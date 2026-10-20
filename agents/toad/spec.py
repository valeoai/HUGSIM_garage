"""The :class:`AgentSpec` of TOAD (valeoai/TOAD at cfa88e0, arXiv 2606.07170): DrivoR with CEM refinement
of the scorer's best proposal, with the CEM settings of TOAD's released code (``drivor_cem_portable/``):
DrivoR's released NAVSIM-v1 checkpoint ``drivor_Nav1_25epochs.pth``, the PDMS scorer weights and CEM
on, all the defaults of ``.._drivor.config``. Until 2026-10 this agent used our ``dynamo_e2e.py``
HUGSIM client's two other CEM settings (update smoothing ``cem_alpha`` 0.7, controls clamped at 5.0
times the kinematic limits); TOAD results of that time are that version's.
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
