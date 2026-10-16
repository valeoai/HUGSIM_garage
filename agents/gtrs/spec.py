"""The :class:`AgentSpec` of GTRS-Dense: build function, checkpoint files and output rule. Its ``parse_raw`` is
the one the GTRS-family clients share (``agents/_navsim/parse.py``).
"""

from .._navsim.parse import parse_gtrs, vocab_choice
from .._navsim.policy import AgentSpec



def _build_gtrs(paths):
    from .agent import GTRSAgent
    from .config import GTRSConfig

    agent = GTRSAgent(GTRSConfig(vocab_path=paths['vocab']))
    agent.load_checkpoint(paths['ckpt'])
    return agent


SPEC = AgentSpec(
    'gtrs',
    {'ckpt': 'gtrs/gtrs_dense_vov.ckpt', 'vocab': 'gtrs/16384.npy'},
    _build_gtrs,
    parse_gtrs,
    'get_trajectory_from_agent_input',
    True,
    lambda m: m.model,
    vocab_choice,
)
