"""Run an agent's step function over frames recorded by ``record.py``, with no simulator.

    python agents/tools/plans.py --agent ltf --frames /tmp/frames --out plans.npz [--device cpu]

``--frames`` holds ``0000.pkl``, ``0001.pkl``, ... each a pickled ``(obs, info)``. The agents are
stepped in that order (an agent sees the frames as the simulator would send them) and the plans
are saved as ``plan_0000`` ... in ``--out``, each ``(N, 2)`` ``(right, forward)``, with
``choice_0000`` ... the discrete choice behind each (the scorer's proposal or vocabulary index; empty for LTF). This uses the
same ``Policy.step`` as ``python -m agents.navsim.run``. ``--seed`` seeds torch before each frame
(``seed + frame index``) so that DrivoR's CEM sampling is repeatable; ``--seed -1`` leaves it
unseeded, as the simulator-facing launcher does.
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents.navsim.policy import AGENTS  # noqa: E402
from agents.navsim.run import make_step  # noqa: E402


def run(agent, frames, ckpt_root=None, device=None, seed=0):
    step, _ = make_step(agent, None, device, ckpt_root, seed=None if seed < 0 else seed)
    plans, choices = {}, {}
    for path in sorted(glob.glob(os.path.join(frames, '*.pkl'))):
        with open(path, 'rb') as f:
            obs, info = pickle.load(f)
        key = os.path.splitext(os.path.basename(path))[0]
        plans[key] = step(obs, info)
        choices[key] = step.policy.last_choice
    if not plans:
        raise FileNotFoundError(f'no *.pkl frames in {frames}')
    return plans, choices


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--agent', required=True, choices=sorted(AGENTS))
    parser.add_argument('--frames', required=True)
    parser.add_argument('--out', required=True, help='the .npz to write')
    parser.add_argument('--device', default=None, help='cuda or cpu (default: $HUGSIM_AGENT_DEVICE, else cuda)')
    parser.add_argument('--ckpt-root', default=None, help='default: $HUGSIM_AGENT_CKPTS')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    plans, choices = run(args.agent, args.frames, args.ckpt_root, args.device, args.seed)
    np.savez(
        args.out,
        **{f'plan_{k}': v for k, v in plans.items()},
        **{f'choice_{k}': v for k, v in choices.items()},
    )
    print(f'wrote {len(plans)} plans to {args.out}')


if __name__ == '__main__':
    main()
