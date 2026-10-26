"""Serve VaVAM (``vavam``) or VaVAM + ECO (``vavam_eco``) to HUGSIM over the episode FIFOs.

    python -m agents._vavam.run --agent {vavam,vavam_eco} --output <episode_dir> [--device cuda|cpu] [--ckpt-root DIR]

Run it from the repository root (``agents/<agent>/launch.sh`` does). The model is the released
VaVAM-B (``VAM_width_1024_pretrained_139k.pt``, 318M + 38M parameters), the one the ECO paper uses,
with VaVAM's image tokenizer (``VQ_ds16_16384_llamagen_encoder.jit``), both from
``$HUGSIM_AGENT_CKPTS/<agent>/``.

Each step follows VaVAM's own closed-loop client (``inference/runner.py`` of valeoai/VideoActionModel
at 738050e, written for NeuroNCAP): the front camera alone, through ``NeuroNCAPTransform`` (resized to
288x512, in [-1, 1]); the last ``context_length`` frames, tokenized; HUGSIM's command (0 right,
1 left, 2 straight, VaVAM's and UniAD's encoding); the flow-matching action expert with its default
10 Euler steps, under bf16 autocast (the client's server default); 6 waypoints 0.5 s apart in the ego
frame (x forward, y left), sent to HUGSIM as (right, forward) = (-y, x).

One choice is ours, because the client was driven at 2 Hz and HUGSIM steps every 0.25 s: the context
frames are the ones 0.5 s apart ending at the current frame (t, t - 0.5, ...), VaVAM's training rate,
rather than every simulator frame. At an episode's start the context is shorter, as in the client.

``vavam_eco`` passes the plan through ECO (``agents/_vavam/eco.py``, our reimplementation of arXiv
2609.31383), anchored to the poses executed 1.0 s and 0.5 s before the step (from ``info['ego_box']``).

The flow's noise is drawn each step and the client never seeded it; ``HUGSIM_AGENT_SEED`` seeds torch
before each step (``seed + step``), as for the NAVSIM family.
"""

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The vendored VaVAM package is imported as `vam`, the name its checkpoints give its classes; only
# vendor/ goes on sys.path, so nothing else here can shadow a module of the simulator's process.
_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor')
for p in (_ROOT, _VENDOR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

AGENTS = ('vavam', 'vavam_eco')
CHECKPOINT = 'VAM_width_1024_pretrained_139k.pt'
TOKENIZER = 'VQ_ds16_16384_llamagen_encoder.jit'
FRAME_PERIOD = 0.5  # s, VaVAM's frame and waypoint period (2 Hz)


class VaVAM:
    """VaVAM's closed-loop client, stepped by HUGSIM; see the module docstring."""

    def __init__(self, ckpt_dir, device='cuda', dtype=torch.bfloat16):
        from agents._vavam.transforms import NeuroNCAPTransform
        from vam.action_expert import load_inference_VAM

        for f in (CHECKPOINT, TOKENIZER):
            if not os.path.isfile(os.path.join(ckpt_dir, f)):
                raise FileNotFoundError(f'{os.path.join(ckpt_dir, f)} is missing: see agents/README.md, Checkpoints')
        self.device = torch.device(device)
        self.dtype = dtype
        self.image_tokenizer = torch.jit.load(os.path.join(ckpt_dir, TOKENIZER), map_location=self.device)
        self.vam = load_inference_VAM(os.path.join(ckpt_dir, CHECKPOINT), self.device)
        self.nb_timesteps = self.vam.context_length
        self.preproc = NeuroNCAPTransform()
        self.frames = []  # (timestamp, preprocessed front frame), oldest first

    def context(self):
        """The frames FRAME_PERIOD apart ending at the newest one, oldest first, at most nb_timesteps."""
        picked = [self.frames[-1]]
        for t, f in reversed(self.frames[:-1]):
            want = picked[-1][0] - FRAME_PERIOD
            if abs(t - want) < 1e-3:
                picked.append((t, f))
            elif t < want:
                break
            if len(picked) == self.nb_timesteps:
                break
        return [f for _, f in reversed(picked)]

    @torch.no_grad()
    def plan(self, front_rgb, timestamp, command):
        """(6, 2) waypoints, ego frame (x forward, y left), 0.5 s apart."""
        self.frames.append((float(timestamp), self.preproc(front_rgb)))
        horizon = FRAME_PERIOD * (self.nb_timesteps - 1) + 1e-3
        self.frames = [(t, f) for t, f in self.frames if t >= timestamp - horizon]
        frames = torch.stack(self.context(), dim=0).to(self.device)
        visual_tokens = self.image_tokenizer(frames).unsqueeze(0)
        command_tokens = torch.tensor([int(command)]).unsqueeze(0).to(self.device)
        with torch.amp.autocast(self.device.type, dtype=self.dtype):
            trajectory = self.vam(visual_tokens, command_tokens, self.dtype)
        return trajectory.float().cpu().numpy().reshape(-1, 2)


def make_step(agent, episode_dir, device='cuda', ckpt_root=None, seed=None):
    """``(step, on_done)`` for ``agents.client.serve`` or an in-process call; see the module docstring."""
    from agents._vavam.eco import eco, history_poses

    if agent not in AGENTS:
        raise KeyError(f'unknown agent {agent!r}; one of {AGENTS}')
    ckpt_root = ckpt_root or os.environ.get('HUGSIM_AGENT_CKPTS')
    if not ckpt_root:
        raise RuntimeError('HUGSIM_AGENT_CKPTS is not set')
    device = device or os.environ.get('HUGSIM_AGENT_DEVICE', 'cuda')
    if seed is None and os.environ.get('HUGSIM_AGENT_SEED'):
        seed = int(os.environ['HUGSIM_AGENT_SEED'])
    model = VaVAM(os.path.join(ckpt_root, agent), device=device)
    use_eco = agent == 'vavam_eco'
    past = []  # (t, x, y, yaw) executed world poses
    count = [0]
    log = open(os.path.join(episode_dir, f'{agent}_steps.tsv'), 'w') if episode_dir else None
    if log:
        log.write('step\ttimestamp\tcommand\tcontext\tpolicy_ms\teco_ms\teco_iters\n')

    def step(obs, info):
        if seed is not None:
            torch.manual_seed(seed + count[0])
        t0 = time.perf_counter()
        xy = model.plan(obs['rgb']['CAM_FRONT'], info['timestamp'], info['command'])
        t1 = time.perf_counter()
        box = info['ego_box']
        now = (float(info['timestamp']), float(box[0]), float(box[1]), float(box[6]))
        iters = ''
        if use_eco:
            xy, eco_info = eco(xy, history_poses(past, now, FRAME_PERIOD))
            iters = eco_info['iterations']
        t2 = time.perf_counter()
        past.append(now)
        if log:
            log.write(f'{count[0]}\t{now[0]:.3f}\t{int(info["command"])}\t{len(model.context())}\t'
                      f'{1e3 * (t1 - t0):.1f}\t{1e3 * (t2 - t1):.1f}\t{iters}\n')
            log.flush()
        count[0] += 1
        return np.column_stack([-xy[:, 1], xy[:, 0]])

    def on_done():
        if log:
            log.close()

    step.model = model
    return step, on_done


def main():
    from agents.client import serve

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--agent', required=True, choices=AGENTS)
    parser.add_argument('--output', required=True, help='the episode folder')
    parser.add_argument('--device', default=None, help='cuda or cpu (default: $HUGSIM_AGENT_DEVICE, else cuda)')
    parser.add_argument('--ckpt-root', default=None, help='default: $HUGSIM_AGENT_CKPTS')
    args = parser.parse_args()
    serve(args.output, *make_step(args.agent, args.output, args.device, args.ckpt_root), name=args.agent)


if __name__ == '__main__':
    main()
