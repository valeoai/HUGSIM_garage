"""The :class:`AgentSpec` of LTF (Latent TransFuser): its ``parse_raw`` (``hugsim/dataparser.py`` of
hyzhou404/NAVSIM), build function, checkpoint and output rule.
"""

import numpy as np

from .._navsim.common import Camera
from .._navsim.parse import CAMERA_SLOTS, THREE_CAMERAS, agent_input, resize_cameras, rotated_status
from .._navsim.policy import AgentSpec

OPENCV2IMU = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])



def parse_ltf(obs, info):
    """``hugsim/dataparser.py``: three cameras, the pose from ``ego_pos`` (LTF never reads it)."""
    raw_imgs, imgs = resize_cameras(obs, THREE_CAMERAS)
    ego_pose = OPENCV2IMU @ info['ego_pos']
    ego_status = rotated_status(info, ego_pose)
    cameras = {CAMERA_SLOTS[c]: Camera(image=imgs[c]) for c in THREE_CAMERAS}
    return agent_input(ego_status, cameras), raw_imgs


def _no_choice(output):
    return np.zeros(0, dtype=np.int64)


def _build_ltf(paths):
    from .agent import TransfuserAgent
    from .config import TransfuserConfig

    agent = TransfuserAgent(TransfuserConfig())
    agent.load_checkpoint(paths['ckpt'])
    return agent


SPEC = AgentSpec(
    'ltf',
    {'ckpt': 'ltf/ltf_seed_0.ckpt'},
    _build_ltf,
    parse_ltf,
    'compute_trajectory',
    False,
    lambda m: m._transfuser_model,
    _no_choice,
    fp32=True,
)
