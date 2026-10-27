"""What ``ltf``, ``ltfv7`` and ``ltf_star`` share on the HUGSIM side: the ``parse_raw`` of hyzhou404/NAVSIM's
``hugsim/dataparser.py``, the build function and the output rule (no proposals to choose among).
"""

import numpy as np

from .._navsim.common import Camera
from .._navsim.parse import CAMERA_SLOTS, THREE_CAMERAS, agent_input, resize_cameras, rotated_status

OPENCV2IMU = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])


def parse_ltf(obs, info):
    """``hugsim/dataparser.py``: three cameras, the pose from ``ego_pos`` (LTF never reads it)."""
    raw_imgs, imgs = resize_cameras(obs, THREE_CAMERAS)
    ego_pose = OPENCV2IMU @ info['ego_pos']
    ego_status = rotated_status(info, ego_pose)
    cameras = {CAMERA_SLOTS[c]: Camera(image=imgs[c]) for c in THREE_CAMERAS}
    return agent_input(ego_status, cameras), raw_imgs


def no_choice(output):
    return np.zeros(0, dtype=np.int64)


def build_ltf(name, use_delta_traj=False):
    """The build function of an LTF checkpoint: ``use_delta_traj`` for BeyondDrive's trajectory head."""

    def build(paths):
        from .agent import TransfuserAgent
        from .config import TransfuserConfig

        agent = TransfuserAgent(TransfuserConfig(use_delta_traj=use_delta_traj), name=name)
        agent.load_checkpoint(paths['ckpt'])
        return agent

    return build
