"""The :class:`AgentSpec` of TOAD as released (valeoai/TOAD at cfa88e0, arXiv 2606.07170): DrivoR with
CEM refinement of the scorer's best proposal, with the CEM settings of TOAD's ``drivor_cem_portable/``:
no smoothing of the distribution update (``cem_alpha`` 1.0) and sampled controls clamped at 1.5 times
the kinematic limits. Everything else, checkpoint included, is ``agents/toad``'s; that agent keeps the
HUGSIM client's earlier settings (0.7 and 5.0).
"""

from .._drivor.client import drivor_choice, parse_drivor
from .._navsim.policy import AgentSpec

#: The two CEM settings where TOAD's released code differs from the HUGSIM client's.
RELEASED_CEM = dict(cem_alpha=1.0, cem_control_margin=1.5)


def _build_toad_release(paths):
    from .._drivor.agent import DrivoRAgent
    from .._drivor.config import DrivoRConfig

    agent = DrivoRAgent(DrivoRConfig(**RELEASED_CEM), name='toad_release')
    agent.load_checkpoint(paths['ckpt'])
    return agent


SPEC = AgentSpec(
    'toad_release',
    {'ckpt': 'toad_release/drivor_Nav1_25epochs.pth'},
    _build_toad_release,
    parse_drivor,
    'compute_trajectory',
    False,
    lambda m: m._drivor_model,
    drivor_choice,
    fp32=True,
)
