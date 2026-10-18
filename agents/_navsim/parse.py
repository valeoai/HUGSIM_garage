"""Pieces of the HUGSIM -> NAVSIM frame conversion (``parse_raw``) that several agents share.

Each agent's ``spec.py`` holds its own ``parse``; what two or more of them use is here: the camera
tables, the resize to what the models were fed, the ``AgentInput`` assembly, the ego-status quirk of
LTF/GTRS/ZTRS/GTRS-Aug (velocity and acceleration rotated by the steering angle, the driving command
fixed to "straight"), the ``parse_raw`` of ``gtrs_e2e.py``/``ztrs_e2e.py``/``gtrs_aug_e2e.py`` (byte-identical
in the three clients) and the choice read-out of the vocabulary scorers. The LTF pieces follow
hyzhou404/NAVSIM's ``hugsim/dataparser.py`` and ``ltf_e2e.py`` (Apache-2.0); the GTRS-family ones our
own clients, kept in our NAVSIM checkout; see ``NOTICE.md``.
"""

import cv2
import numpy as np

from .common import AgentInput, Camera, Cameras, EgoStatus, Lidar


# HUGSIM camera name -> NAVSIM camera slot
CAMERA_SLOTS = {'CAM_FRONT': 'cam_f0', 'CAM_FRONT_RIGHT': 'cam_r0', 'CAM_FRONT_LEFT': 'cam_l0', 'CAM_BACK': 'cam_b0'}
THREE_CAMERAS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT']
FOUR_CAMERAS = THREE_CAMERAS + ['CAM_BACK']


def resize_cameras(obs, names):
    """The renders as HUGSIM sent them, and resized to 1920x1080 (what the models were fed)."""
    raw_imgs, imgs = {}, {}
    for cam in names:
        im = obs['rgb'][cam]
        raw_imgs[cam] = im
        imgs[cam] = cv2.resize(im, (1920, 1080))
    return raw_imgs, imgs


def agent_input(ego_status, cameras_dict):
    cameras = Cameras()
    for cam_name, cam in cameras_dict.items():
        setattr(cameras, cam_name, cam)
    lidar = Lidar(np.zeros((6, 100), dtype=np.float32))
    return AgentInput([ego_status], [cameras], [lidar])


def straight_command():
    command = np.zeros(4)
    command[1] = 1  # the driving command is fixed to "straight"
    return command


def rotated_status(info, ego_pose):
    """Velocity and acceleration of LTF/GTRS/ZTRS/GTRS-Aug: the speed rotated by the steering angle."""
    velo, acc = np.zeros(2), np.zeros(2)
    yaw = -info['ego_steer']
    forward_velo = info['ego_velo']
    forward_acc = info['accelerate']
    velo[0] = forward_velo * np.cos(yaw)
    velo[1] = forward_velo * np.sin(yaw)
    acc[0] = forward_acc * np.cos(yaw)
    acc[1] = forward_acc * np.sin(yaw)
    return EgoStatus(ego_pose, velo, acc, straight_command())


def parse_gtrs(obs, info):
    """``parse_raw`` of ``gtrs_e2e.py`` / ``ztrs_e2e.py`` / ``gtrs_aug_e2e.py``: the pose is zero."""
    raw_imgs, imgs = resize_cameras(obs, THREE_CAMERAS)
    ego_status = rotated_status(info, np.zeros(3))
    cameras = {CAMERA_SLOTS[c]: Camera(image=imgs[c]) for c in THREE_CAMERAS}
    return agent_input(ego_status, cameras), raw_imgs


def vocab_choice(output):
    """The vocabulary index the scorer picked."""
    return output['selected_indices'].reshape(-1)[:1].cpu().numpy().astype(np.int64)
