"""Check `vavam` against VaVAM's own closed-loop client on recorded HUGSIM frames.

    # 1. frames:  HUGSIM_RECORD_OUT=<frames> HUGSIM_RECORD_STEPS=34 python closed_loop.py ... --ad record --ad_path agents/tools/record.sh
    # 2. ours:     python -m agents._vavam.check_port ours --frames <frames> --out ours.npz
    # 3. original: PYTHONPATH=<VideoActionModel checkout>:<its deps> python agents/_vavam/check_port.py original \
    #                  --frames <frames> --out orig.npz
    # 4. compare:  python agents/_vavam/check_port.py compare ours.npz orig.npz

The two must run in separate processes: both import a package named ``vam``. Every step is seeded
alike (``torch.manual_seed(seed + step)`` before the step), so the flow's noise is the same draw.

The original client (``inference/runner.py``, ``VAMRunner``) keeps the last ``context_length`` frames it
was given; ours keeps the frames 0.5 s apart ending at the current one. So the original is given the
frames at 2 Hz: one runner gets the even steps and another the odd ones, and step i is compared with
the runner that holds the frames i, i - 2, i - 4, ... -- the context ours picks at HUGSIM's 0.25 s.
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np


def frames(folder):
    for path in sorted(glob.glob(os.path.join(folder, '*.pkl'))):
        with open(path, 'rb') as f:
            yield pickle.load(f)


def run_ours(args):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, root)
    from agents._vavam.run import make_step

    step, _ = make_step('vavam', None, device=args.device, ckpt_root=args.ckpt_root, seed=args.seed)
    plans = []
    for obs, info in frames(args.frames):
        rf = step(obs, info)
        plans.append(np.column_stack([rf[:, 1], -rf[:, 0]]))  # back to (x forward, y left)
    np.savez(args.out, plans=np.array(plans))
    print(f'ours: {len(plans)} plans -> {args.out}')


def run_original(args):
    import torch
    from inference.runner import VAMInferenceInput, VAMRunner  # VideoActionModel's own client

    ckpt = os.path.join(args.ckpt_root, 'vavam')
    make = lambda: VAMRunner(os.path.join(ckpt, 'VQ_ds16_16384_llamagen_encoder.jit'),  # noqa: E731
                             os.path.join(ckpt, 'VAM_width_1024_pretrained_139k.pt'), torch.device(args.device), torch.bfloat16)
    runners = [make(), make()]
    plans = []
    for i, (obs, info) in enumerate(frames(args.frames)):
        torch.manual_seed(args.seed + i)
        out = runners[i % 2].forward_inference(
            VAMInferenceInput(imgs=[obs['rgb']['CAM_FRONT']], timestamp=info['timestamp'], command=int(info['command'])))
        plans.append(out.trajectory)
    np.savez(args.out, plans=np.array(plans))
    print(f'original: {len(plans)} plans -> {args.out}')


def compare(args):
    a, b = np.load(args.ours)['plans'], np.load(args.original)['plans']
    assert a.shape == b.shape, (a.shape, b.shape)
    d = np.abs(a - b).max(axis=(1, 2))
    print(f'{len(d)} steps, plans {a.shape[1:]}: max |ours - original| per step (m):')
    print(' '.join(f'{x:.2e}' for x in d))
    print(f'identical steps: {int((d == 0).sum())} of {len(d)}; worst {d.max():.3e} m')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    for name in ('ours', 'original'):
        p = sub.add_parser(name)
        p.add_argument('--frames', required=True)
        p.add_argument('--out', required=True)
        p.add_argument('--ckpt-root', default=os.environ.get('HUGSIM_AGENT_CKPTS'))
        p.add_argument('--device', default='cuda')
        p.add_argument('--seed', type=int, default=0)
    p = sub.add_parser('compare')
    p.add_argument('ours')
    p.add_argument('original')
    args = ap.parse_args()
    {'ours': run_ours, 'original': run_original, 'compare': compare}[args.mode](args)


if __name__ == '__main__':
    main()
