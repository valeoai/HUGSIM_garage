"""The :class:`AgentSpec` of LTFv6: LEAD's Latent TransFuser v6 (arXiv 2512.20563), its released NAVSIM
checkpoint. No HUGSIM client of it exists; this one feeds it what LEAD's NAVSIM workspace does
(``agent.py``): the four cameras (the rear one too), the driving command fixed to "straight" and the
speed and acceleration, as the garage's other NAVSIM-family clients do.
"""

import numpy as np

from .._navsim.common import Camera
from .._navsim.parse import CAMERA_SLOTS, FOUR_CAMERAS, agent_input, resize_cameras, rotated_status
from .._navsim.policy import AgentSpec


def parse_ltfv6(obs, info):
    """Four cameras at 1920x1080; velocity and acceleration as LTF's client (LTFv6 reads only their norms)."""
    raw_imgs, imgs = resize_cameras(obs, FOUR_CAMERAS)
    ego_status = rotated_status(info, np.zeros(3))
    cameras = {CAMERA_SLOTS[c]: Camera(image=imgs[c]) for c in FOUR_CAMERAS}
    return agent_input(ego_status, cameras), raw_imgs


def _no_choice(output):
    return np.zeros(0, dtype=np.int64)


def _build_ltfv6(paths):
    from .agent import LTFv6Agent

    agent = LTFv6Agent(paths['config'])
    agent.load_checkpoint(paths['ckpt'])
    return agent


SPEC = AgentSpec(
    'ltfv6',
    {'ckpt': 'ltfv6/model_0060.pth', 'config': 'ltfv6/config.json'},
    _build_ltfv6,
    parse_ltfv6,
    'compute_trajectory',
    False,
    lambda m: m.net,
    _no_choice,
)
