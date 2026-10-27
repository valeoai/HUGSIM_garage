"""The :class:`AgentSpec` of LTF*: BeyondDrive's own Latent TransFuser baseline (arXiv 2605.19771), trained
as LTFv7 but without hard negatives, with the same trajectory head. Inputs and client are LTF's
(``../_ltf/client.py``).
"""

from .._ltf.client import build_ltf, no_choice, parse_ltf
from .._navsim.policy import AgentSpec

SPEC = AgentSpec(
    'ltf_star',
    {'ckpt': 'ltf_star/LTF_-PDMS_88.7.ckpt'},
    build_ltf('ltf_star', use_delta_traj=True),
    parse_ltf,
    'compute_trajectory',
    False,
    lambda m: m._transfuser_model,
    no_choice,
    fp32=True,
)
