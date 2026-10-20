"""The :class:`AgentSpec` of DrivoR as its paper evaluated it on HUGSIM (arXiv 2601.05083v2, Table 2):
the released NAVSIM-v1 checkpoint ``drivor_Nav1_25epochs.pth``, no CEM refinement, and the scorer
weights that run used, NC 1, DAC 1.6, DDC 0, TTC 8.1, EP 11.2, comfort 2 (logged in each episode of
that run, November 2025). Everything else is ``agents/drivor``'s, which keeps the official
implementation's default weights, the ones TOAD uses too.
"""

from .._drivor.client import drivor_choice, parse_drivor
from .._navsim.policy import AgentSpec

#: The scorer weights of the DrivoR paper's HUGSIM run, where they differ from the defaults.
PAPER_WEIGHTS = dict(dac=1.6, ttc=8.1, ep=11.2)


def _build_drivor_paper(paths):
    from .._drivor.agent import DrivoRAgent
    from .._drivor.config import DrivoRConfig

    agent = DrivoRAgent(DrivoRConfig(use_cem=False, **PAPER_WEIGHTS), name='drivor_paper')
    agent.load_checkpoint(paths['ckpt'])
    return agent


SPEC = AgentSpec(
    'drivor_paper',
    {'ckpt': 'drivor_paper/drivor_Nav1_25epochs.pth'},
    _build_drivor_paper,
    parse_drivor,
    'compute_trajectory',
    False,
    lambda m: m._drivor_model,
    drivor_choice,
    fp32=True,
)
