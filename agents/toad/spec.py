"""The :class:`AgentSpec` of TOAD: DrivoR with CEM refinement of the scorer's best proposal, as
hyzhou404/NAVSIM's ``dynamo_e2e.py`` client runs it (``CONFIG_NAME = "drivor"``): the checkpoint
``nav3.ckpt``, the PDMS scorer weights and CEM on, all the defaults of ``.._drivor.config``.
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
    {'ckpt': 'toad/nav3.ckpt'},
    _build_toad,
    parse_drivor,
    'compute_trajectory',
    False,
    lambda m: m._drivor_model,
    drivor_choice,
    fp32=True,
)
