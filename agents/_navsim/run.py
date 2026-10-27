"""Serve a NAVSIM-family agent to HUGSIM over the episode FIFOs.

    python -m agents._navsim.run --agent {ltf,drivor,drivor_paper,toad,gtrs,ztrs,gtrs_aug} --output <episode_dir> \
        [--device cuda|cpu] [--ckpt-root DIR]

Run it from the repository root (the launchers ``agents/<agent>/launch.sh`` do). Checkpoints are
read from ``--ckpt-root`` or ``$HUGSIM_AGENT_CKPTS``, laid out as in each agent's ``spec.py``. The device
is ``--device``, else ``$HUGSIM_AGENT_DEVICE``, else ``cuda``. ``HUGSIM_AGENT_VIZ=1`` also writes
the plan drawn on the front camera, and a video, to ``<episode_dir>/<agent>/``. ``HUGSIM_AGENT_SEED``
seeds torch before each step (only TOAD's CEM sampling is random).
"""

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from agents._navsim import AGENTS  # noqa: E402
from agents._navsim.policy import Policy  # noqa: E402
from agents._navsim.viz import save_frame, to_video  # noqa: E402
from agents.client import serve  # noqa: E402


def make_step(agent, episode_dir, device='cuda', ckpt_root=None, seed=None):
    """Build the agent and return ``(step, on_done)`` for ``agents.client.serve`` or an in-process call.

    ``step(obs, info)`` returns the ``(N, 2)`` ``(right, forward)`` plan and does not modify ``obs`` or
    ``info``; ``step.policy`` is the underlying :class:`Policy`. ``on_done()`` writes the video when
    ``HUGSIM_AGENT_VIZ=1`` (the frames go to ``<episode_dir>/<agent>/``), else it does nothing.
    ``device=None`` means ``$HUGSIM_AGENT_DEVICE``, else cuda. ``seed`` (else ``$HUGSIM_AGENT_SEED``)
    seeds torch before each step.
    """
    if seed is None and os.environ.get('HUGSIM_AGENT_SEED'):
        seed = int(os.environ['HUGSIM_AGENT_SEED'])
    policy = Policy(agent, ckpt_root=ckpt_root, device=device, seed=seed)
    viz = os.environ.get('HUGSIM_AGENT_VIZ') == '1' and episode_dir is not None
    viz_folder = os.path.join(episode_dir, agent) if viz else None
    if viz:
        os.makedirs(viz_folder, exist_ok=True)
    count = [0]
    past = []  # (t, x, y, yaw) executed world poses, for ECO's anchors

    def step(obs, info):
        plan = policy.step(obs, info)
        if policy.spec.eco:
            from agents._eco.eco import eco, history_poses

            box = info['ego_box']
            now = (float(info['timestamp']), float(box[0]), float(box[1]), float(box[6]))
            # (right, forward) -> (x forward, y left); NAVSIM headings are already in that frame
            xy = np.column_stack([plan[:, 1], -plan[:, 0]])
            xy, _ = eco(xy, history_poses(past, now, 0.5), headings=policy.last_way_points[:, 2])
            past.append(now)
            plan = np.column_stack([-xy[:, 1], xy[:, 0]])
        if viz:
            save_frame(
                policy.last_raw_imgs, policy.last_way_points, info['cam_params'], os.path.join(viz_folder, f'{count[0]:04d}')
            )
        count[0] += 1
        return plan

    def on_done():
        if viz:
            to_video(viz_folder)

    step.policy = policy
    return step, on_done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--agent', required=True, choices=sorted(AGENTS))
    parser.add_argument('--output', required=True, help='the episode folder')
    parser.add_argument('--device', default=None, help='cuda or cpu (default: $HUGSIM_AGENT_DEVICE, else cuda)')
    parser.add_argument('--ckpt-root', default=None, help='default: $HUGSIM_AGENT_CKPTS')
    args = parser.parse_args()

    serve(args.output, *make_step(args.agent, args.output, args.device, args.ckpt_root), name=args.agent)


if __name__ == '__main__':
    main()
