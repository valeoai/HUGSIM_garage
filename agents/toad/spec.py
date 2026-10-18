"""The :class:`AgentSpec` of TOAD (valeoai/TOAD, arXiv 2606.07170): DrivoR with CEM refinement of the
scorer's best proposal, as hyzhou404/NAVSIM's ``dynamo_e2e.py`` client runs it (``CONFIG_NAME =
"drivor"``): DrivoR's released NAVSIM-v1 checkpoint ``drivor_Nav1_25epochs.pth``, the PDMS scorer
weights and CEM on, all the defaults of ``.._drivor.config``. Two CEM settings of that client differ
from TOAD's released code: the update smoothing ``cem_alpha`` is 0.7 (1.0 there) and sampled controls
are clamped at 5.0 times the kinematic limits (1.5 there).
"""

from .._drivor.client import drivor_choice, parse_drivor
from .._navsim.policy import AgentSpec


def _build_toad(paths):
    from .._drivor.agent import DrivoRAgent
    from .._drivor.config import DrivoRConfig

    agent = DrivoRAgent(DrivoRConfig(), name='toad')
    agent.load_checkpoint(paths['ckpt'])
    return agent


SPEC = AgentSpec(
    'toad',
    {'ckpt': 'toad/drivor_Nav1_25epochs.pth'},
    _build_toad,
    parse_drivor,
    'compute_trajectory',
    False,
    lambda m: m._drivor_model,
    drivor_choice,
    fp32=True,
)
