"""The :class:`AgentSpec` of DrivoR + ECO: ``agents/drivor`` (the released NAVSIM-v1 checkpoint, the
official implementation's default scorer weights, no CEM refinement) with each plan passed through ECO
(``agents/_eco``, our reimplementation of arXiv 2609.31383), DrivoR's own headings as the start of the
solve. The ECO paper drives "DrivoR (w/ SimScale)"; the SimScale-trained checkpoint is not this one.
"""

from .._drivor.client import drivor_choice, parse_drivor
from .._navsim.policy import AgentSpec


def _build_drivor(paths):
    from .._drivor.agent import DrivoRAgent
    from .._drivor.config import DrivoRConfig

    agent = DrivoRAgent(DrivoRConfig(use_cem=False), name='drivor')
    agent.load_checkpoint(paths['ckpt'])
    return agent


SPEC = AgentSpec(
    'drivor_eco',
    {'ckpt': 'drivor_eco/drivor_Nav1_25epochs.pth'},
    _build_drivor,
    parse_drivor,
    'compute_trajectory',
    False,
    lambda m: m._drivor_model,
    drivor_choice,
    fp32=True,
    eco=True,
)
