"""The :class:`AgentSpec` of DrivoR (valeoai/DrivoR), as evaluated on HUGSIM in the DrivoR paper
(arXiv 2601.05083, Table 2): the released NAVSIM-v1 checkpoint ``drivor_Nav1_25epochs.pth``, the PDMS
scorer weights, and no CEM refinement. This is the official implementation's default configuration,
and what that evaluation's config (``dynamo_v6_agent_new_scorer_v3_eval_nav3.yaml`` in the NAVSIM
checkout) ran.

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
