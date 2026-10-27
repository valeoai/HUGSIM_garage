"""The :class:`AgentSpec` of LTFv7: Latent TransFuser trained with BeyondDrive's hard negatives
(arXiv 2605.19771), its released checkpoint, with BeyondDrive's trajectory head. Inputs and client
are LTF's (``../_ltf/client.py``), as BeyondDrive evaluates it (``agent=transfuser_agent``,
``latent=True``, ``use_delta_traj=True``).
"""

from .._ltf.client import build_ltf, no_choice, parse_ltf
from .._navsim.policy import AgentSpec

SPEC = AgentSpec(
    'ltfv7',
    {'ckpt': 'ltfv7/LTFv7-PDMS_89.841.ckpt'},
    build_ltf('ltfv7', use_delta_traj=True),
    parse_ltf,
    'compute_trajectory',
    False,
    lambda m: m._transfuser_model,
    no_choice,
    fp32=True,
)
