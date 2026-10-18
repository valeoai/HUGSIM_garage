"""The :class:`AgentSpec` of ZTRS: build function, checkpoint files and output rule. Its ``parse_raw`` is
the one the GTRS-family clients share (``agents/_navsim/parse.py``).
"""

from .._navsim.parse import parse_gtrs, vocab_choice
from .._navsim.policy import AgentSpec



def _build_ztrs(paths):
    from .agent import ZTRSAgent
    from .config import ZTRSConfig

    agent = ZTRSAgent(ZTRSConfig(vocab_path=paths['vocab']))
    agent.load_checkpoint(paths['ckpt'])
    return agent


SPEC = AgentSpec(
    'ztrs',
    {'ckpt': 'ztrs/ztrs_vov.ckpt', 'vocab': 'ztrs/16384.npy'},
    _build_ztrs,
    parse_gtrs,
    'get_trajectory_from_agent_input',
    True,
    lambda m: m.model,
    vocab_choice,
)
