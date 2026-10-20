"""The :class:`AgentSpec` of DrivoR (valeoai/DrivoR): the released NAVSIM-v1 checkpoint
``drivor_Nav1_25epochs.pth``, the official implementation's default scorer weights (those TOAD uses
too), and no CEM refinement. The DrivoR paper's HUGSIM run (arXiv 2601.05083v2, Table 2) used other
scorer weights; that setup is ``agents/drivor_paper``.

With CEM refinement the same model is TOAD, which is ``agents/toad``.
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
    'drivor',
    {'ckpt': 'drivor/drivor_Nav1_25epochs.pth'},
    _build_drivor,
    parse_drivor,
    'compute_trajectory',
    False,
    lambda m: m._drivor_model,
    drivor_choice,
    fp32=True,
)
