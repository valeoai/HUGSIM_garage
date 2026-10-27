"""The :class:`AgentSpec` of LTF (Latent TransFuser): NAVSIM's released checkpoint, driven as the HUGSIM
client of hyzhou404/NAVSIM drove it (``../_ltf/client.py``).
"""

from .._ltf.client import build_ltf, no_choice, parse_ltf
from .._navsim.policy import AgentSpec

SPEC = AgentSpec(
    'ltf',
    {'ckpt': 'ltf/ltf_seed_0.ckpt'},
    build_ltf('ltf'),
    parse_ltf,
    'compute_trajectory',
    False,
    lambda m: m._transfuser_model,
    no_choice,
    fp32=True,
)
