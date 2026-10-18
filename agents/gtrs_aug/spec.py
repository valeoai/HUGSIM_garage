"""The :class:`AgentSpec` of GTRS-Aug (teacher): build function, checkpoint files and output rule. Its ``parse_raw`` is
the one the GTRS-family clients share (``agents/_navsim/parse.py``).
"""

import numpy as np

from .._navsim.parse import parse_gtrs
from .._navsim.policy import AgentSpec


def _gtrs_aug_choice(output):
    """The first-stage vocabulary index and the one the refinement stage finally picked."""
    refinement = output['refinement'][-1]
    final = refinement['indices_absolute'][0, refinement['scores'].argmax(1)[0]]
    return np.array([int(output['selected_indices'][0]), int(final)], dtype=np.int64)


def _build_gtrs_aug(paths):
    from .agent import GTRSAugAgent
    from .config import GTRSAugConfig

    agent = GTRSAugAgent(GTRSAugConfig(vocab_path=paths['vocab']))
    agent.load_checkpoint(paths['ckpt'])
    return agent


SPEC = AgentSpec(
    'gtrs_aug',
    {'ckpt': 'gtrs_aug/gtrs_aug_vov.ckpt', 'vocab': 'gtrs_aug/8192.npy'},
    _build_gtrs_aug,
    parse_gtrs,
    'get_trajectory_from_agent_input',
    True,
    lambda m: m.model.teacher.model,
    _gtrs_aug_choice,
)
