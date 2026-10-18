"""Run UniAD or VAD offline over recorded frames and save the plans.

    python plans.py --agent uniad|vad --frames DIR --out plans.npz [--device cuda|cpu] [--seed S]

``DIR`` holds ``*.pkl`` files, each ``pickle.dumps((obs, info))`` as closed_loop.py sends them
(``agents/tools/record.py`` records them), played in file-name order as one episode. ``--seed``
seeds ``random``, numpy and torch before every frame, which makes VAD (stochastic by default,
see runner.py) repeatable. ``--max-frames`` stops early. Writes ``plans`` ``(N, 6, 2)`` and
``names``. The environment variables of runner.py apply.
"""

import argparse
import glob
import os
import pickle
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def seed_all(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--agent', required=True, choices=['uniad', 'vad'])
    parser.add_argument('--frames', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--ckpt', help='default: $HUGSIM_AGENT_CKPTS/{uniad/uniad_base_e2e.pth,vad/VAD_base.pth}')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    # absolute before the runner changes into the agent's tree
    frames = sorted(glob.glob(os.path.join(os.path.abspath(args.frames), '*.pkl')))
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    if not frames:
        sys.exit(f'no *.pkl frames in {args.frames}')
    out = os.path.abspath(args.out)
    sys.path.insert(0, HERE)
    import runner

    ckpt = args.ckpt
    if ckpt is None:
        root = os.environ['HUGSIM_AGENT_CKPTS']
        ckpt = os.path.join(root, runner.CKPT_NAMES[args.agent])
    agent = runner.load_agent(args.agent, ckpt, device=args.device, output=os.path.dirname(out))
    torch = agent.torch

    plans = []
    for i, path in enumerate(frames):
        with open(path, 'rb') as f:
            obs, info = pickle.load(f)
        if args.seed is not None:
            seed_all(args.seed, torch)
        plans.append(agent.step(obs, info))
        print(f'[{args.agent}] {os.path.basename(path)}: {plans[-1][:2].tolist()}', flush=True)
    agent.finish()
    np.savez(out, plans=np.stack(plans), names=np.array([os.path.basename(p) for p in frames]))
    print(f'[{args.agent}] wrote {out}')


if __name__ == '__main__':
    main()
