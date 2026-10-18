"""What ``drivor`` and ``toad`` share on the HUGSIM side: the ``parse_raw`` of hyzhou404/NAVSIM's
``dynamo_e2e.py`` client and the rule that reports which proposal was driven.
"""

import numpy as np

from .._navsim.common import Camera, EgoStatus
from .._navsim.parse import CAMERA_SLOTS, FOUR_CAMERAS, agent_input, resize_cameras, straight_command

WHEELBASE = 2.7  # m, DrivoR's lateral-acceleration estimate


def _get_intrinsics(dict_intrinsics):
    H, W = dict_intrinsics['H'], dict_intrinsics['W']
    cx, cy = dict_intrinsics['cx'], dict_intrinsics['cy']
    fovx, fovy = dict_intrinsics['fovx'], dict_intrinsics['fovy']  # radians

    fx = W / (2.0 * np.tan(fovx / 2.0))
    fy = H / (2.0 * np.tan(fovy / 2.0))

    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


def parse_drivor(obs, info):
    """``parse_raw`` of ``dynamo_e2e.py``: four cameras with intrinsics and extrinsics, the speed
    unrotated and a bicycle-model lateral acceleration."""
    cam_params = info['cam_params']
    raw_imgs, imgs = resize_cameras(obs, FOUR_CAMERAS)

    velo, acc = np.zeros(2), np.zeros(2)
    ego_pose = np.zeros(3)
    forward_velo = info['ego_velo']
    forward_acc = info['accelerate']
    lateral_acc = forward_velo * forward_velo * np.tan(-info['ego_steer']) / WHEELBASE
    velo[0] = forward_velo
    velo[1] = 0.0
    acc[0] = forward_acc
    acc[1] = lateral_acc
    ego_status = EgoStatus(ego_pose, velo, acc, straight_command())

    cameras = {}
    for cam in FOUR_CAMERAS:
        l2c = cam_params[cam]['l2c']
        c2l = np.linalg.inv(l2c)
        cameras[CAMERA_SLOTS[cam]] = Camera(
            image=imgs[cam],
            intrinsics=_get_intrinsics(cam_params[cam]['intrinsic']),
            sensor2lidar_rotation=c2l[:3, :3],
            sensor2lidar_translation=c2l[:3, 3],
        )
    return agent_input(ego_status, cameras), raw_imgs


def drivor_choice(output):
    """The scorer's best proposal, and whether CEM replaced it (always 0 when CEM is off, since the
    model then reports no ``use_cem``)."""
    token = int(output['pdm_score'].argmax(1)[0])
    use_cem = int(output['use_cem'][0]) if 'use_cem' in output else 0
    return np.array([token, use_cem], dtype=np.int64)
