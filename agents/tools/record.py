"""An agent that records what the simulator sends it, for checking a ported agent offline.

Each request is pickled to ``<out>/<step>.pkl`` as ``(obs, info)``, exactly as it arrived; the
answer is a straight plan at a fixed speed, so the frames cover a moving ego without depending on
any model. After ``--steps`` requests it answers ``None``, which ends the episode.

    HUGSIM_RECORD_OUT=/tmp/frames HUGSIM_RECORD_STEPS=12 python closed_loop.py ... \
        --ad record --ad_path agents/tools/record.sh

``agents/tools/plans.py`` (and ``agents/mmdet3d/plans.py``) then replays these frames through an
agent offline, so two implementations can be compared on the same inputs.
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents.client import serve  # noqa: E402

PLAN_TIMESTEP = 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='the episode folder')
    parser.add_argument('--out', required=True, help='where the frames go')
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--speed', type=float, default=4.0, help='m/s of the straight plan')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    count = [0]

    def step(obs, info):
        if count[0] >= args.steps:
            return None
        with open(os.path.join(args.out, f'{count[0]:04d}.pkl'), 'wb') as f:
            pickle.dump((obs, info), f)
        count[0] += 1
        t = PLAN_TIMESTEP * np.arange(1, 7)
        return np.stack([np.zeros_like(t), args.speed * t], axis=1)  # (right, forward)

    serve(args.output, step, name='record')


if __name__ == '__main__':
    main()
