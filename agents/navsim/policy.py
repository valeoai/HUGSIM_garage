"""The five NAVSIM-family agents behind one ``step(obs, info) -> plan`` function.

Each agent reproduces, step for step, what its HUGSIM client script in hyzhou404/NAVSIM did:
``hugsim/dataparser.py`` + ``ltf_e2e.py`` (LTF), ``dynamo_e2e.py`` (DrivoR) and
``gtrs_e2e.py``/``ztrs_e2e.py``/``gtrs_aug_e2e.py``, each with its own ``parse_raw``. The three
GTRS-family clients and LTF share the ego-status quirk (velocity and acceleration rotated by the
steering angle, the driving command fixed to "straight"); DrivoR passes the longitudinal speed and
a bicycle-model lateral acceleration instead. ``step`` is what ``run.py`` serves and ``tools/plans.py``
replays over recorded frames.
"""

import contextlib
import os
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import torch

from .common import AgentInput, Camera, Cameras, EgoStatus, Lidar

OPENCV2IMU = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

# HUGSIM camera name -> NAVSIM camera slot
CAMERA_SLOTS = {'CAM_FRONT': 'cam_f0', 'CAM_FRONT_RIGHT': 'cam_r0', 'CAM_FRONT_LEFT': 'cam_l0', 'CAM_BACK': 'cam_b0'}
THREE_CAMERAS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT']
FOUR_CAMERAS = THREE_CAMERAS + ['CAM_BACK']
WHEELBASE = 2.7  # m, DrivoR's lateral-acceleration estimate


def _resize_cameras(obs, names):
    """The renders as HUGSIM sent them, and resized to 1920x1080 (what the models were fed)."""
    raw_imgs, imgs = {}, {}
    for cam in names:
        im = obs['rgb'][cam]
        raw_imgs[cam] = im
        imgs[cam] = cv2.resize(im, (1920, 1080))
    return raw_imgs, imgs


def _agent_input(ego_status, cameras_dict):
    cameras = Cameras()
    for cam_name, cam in cameras_dict.items():
        setattr(cameras, cam_name, cam)
    lidar = Lidar(np.zeros((6, 100), dtype=np.float32))
    return AgentInput([ego_status], [cameras], [lidar])


def _straight_command():
    command = np.zeros(4)
    command[1] = 1  # the driving command is fixed to "straight"
    return command


def _rotated_status(info, ego_pose):
    """Velocity and acceleration of LTF/GTRS/ZTRS/GTRS-Aug: the speed rotated by the steering angle."""
    velo, acc = np.zeros(2), np.zeros(2)
    yaw = -info['ego_steer']
    forward_velo = info['ego_velo']
    forward_acc = info['accelerate']
    velo[0] = forward_velo * np.cos(yaw)
    velo[1] = forward_velo * np.sin(yaw)
    acc[0] = forward_acc * np.cos(yaw)
    acc[1] = forward_acc * np.sin(yaw)
    return EgoStatus(ego_pose, velo, acc, _straight_command())


def parse_ltf(obs, info):
    """``hugsim/dataparser.py``: three cameras, the pose from ``ego_pos`` (LTF never reads it)."""
    raw_imgs, imgs = _resize_cameras(obs, THREE_CAMERAS)
    ego_pose = OPENCV2IMU @ info['ego_pos']
    ego_status = _rotated_status(info, ego_pose)
    cameras = {CAMERA_SLOTS[c]: Camera(image=imgs[c]) for c in THREE_CAMERAS}
    return _agent_input(ego_status, cameras), raw_imgs


def parse_gtrs(obs, info):
    """``parse_raw`` of ``gtrs_e2e.py`` / ``ztrs_e2e.py`` / ``gtrs_aug_e2e.py``: the pose is zero."""
    raw_imgs, imgs = _resize_cameras(obs, THREE_CAMERAS)
    ego_status = _rotated_status(info, np.zeros(3))
    cameras = {CAMERA_SLOTS[c]: Camera(image=imgs[c]) for c in THREE_CAMERAS}
    return _agent_input(ego_status, cameras), raw_imgs


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
    raw_imgs, imgs = _resize_cameras(obs, FOUR_CAMERAS)

    velo, acc = np.zeros(2), np.zeros(2)
    ego_pose = np.zeros(3)
    forward_velo = info['ego_velo']
    forward_acc = info['accelerate']
    lateral_acc = forward_velo * forward_velo * np.tan(-info['ego_steer']) / WHEELBASE
    velo[0] = forward_velo
    velo[1] = 0.0
    acc[0] = forward_acc
    acc[1] = lateral_acc
    ego_status = EgoStatus(ego_pose, velo, acc, _straight_command())

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
    return _agent_input(ego_status, cameras), raw_imgs


def to_plan(poses, subsample):
    """NAVSIM poses (x forward, y left, heading) -> HUGSIM's ``(right, forward)``.

    ``subsample`` keeps every fifth 0.1 s pose (from the fourth) for the 4 s plans of GTRS & co, so
    the waypoints are 0.5 s apart as HUGSIM expects.
    """
    imu_way_points = poses[:, :3]
    way_points = imu_way_points[:, [1, 0, 2]]
    way_points[:, 0] *= -1
    if subsample:
        way_points = way_points[4::5]
    return way_points


def _no_choice(output):
    return np.zeros(0, dtype=np.int64)


def _drivor_choice(output):
    """The scorer's best proposal, and whether CEM replaced it."""
    token = int(output['pdm_score'].argmax(1)[0])
    return np.array([token, int(output['use_cem'][0])], dtype=np.int64)


def _vocab_choice(output):
    """The vocabulary index the scorer picked."""
    return output['selected_indices'].reshape(-1)[:1].cpu().numpy().astype(np.int64)


def _gtrs_aug_choice(output):
    """The first-stage vocabulary index and the one the refinement stage finally picked."""
    refinement = output['refinement'][-1]
    final = refinement['indices_absolute'][0, refinement['scores'].argmax(1)[0]]
    return np.array([int(output['selected_indices'][0]), int(final)], dtype=np.int64)


@dataclass
class AgentSpec:
    name: str
    files: dict  # role -> path relative to the checkpoint root
    build: Callable  # (paths) -> torch module with load_checkpoint
    parse: Callable
    compute: str  # the method of the module that turns an AgentInput into a Trajectory
    subsample: bool
    network: Callable  # the module whose output holds the discrete choice
    choice: Callable  # that output -> the discrete choices (proposal / vocabulary indices)
    # The original client ran this agent on the CPU (it never moved the model to the GPU). On a
    # GPU, cuDNN's default TF32 convolutions then move its plans by up to 2.5 cm on scene-0383
    # against 2e-5 m with TF32 off, so the step turns TF32 off for itself (see Policy.step).
    fp32: bool = False


def _build_ltf(paths):
    from .transfuser.agent import TransfuserAgent
    from .transfuser.config import TransfuserConfig

    agent = TransfuserAgent(TransfuserConfig())
    agent.load_checkpoint(paths['ckpt'])
    return agent


def _build_drivor(paths):
    from .drivor.agent import DrivoRAgent
    from .drivor.config import DrivoRConfig

    agent = DrivoRAgent(DrivoRConfig())
    agent.load_checkpoint(paths['ckpt'])
    return agent


def _build_gtrs(paths):
    from .gtrs.agent import GTRSAgent
    from .gtrs.config import GTRSConfig

    agent = GTRSAgent(GTRSConfig(vocab_path=paths['vocab']))
    agent.load_checkpoint(paths['ckpt'])
    return agent


def _build_ztrs(paths):
    from .ztrs.agent import ZTRSAgent
    from .ztrs.config import ZTRSConfig

    agent = ZTRSAgent(ZTRSConfig(vocab_path=paths['vocab']))
    agent.load_checkpoint(paths['ckpt'])
    return agent


def _build_gtrs_aug(paths):
    from .gtrs_aug.agent import GTRSAugAgent
    from .gtrs_aug.config import GTRSAugConfig

    agent = GTRSAugAgent(GTRSAugConfig(vocab_path=paths['vocab']))
    agent.load_checkpoint(paths['ckpt'])
    return agent


AGENTS = {
    'ltf': AgentSpec(
        'ltf',
        {'ckpt': 'ltf/ltf_seed_0.ckpt'},
        _build_ltf,
        parse_ltf,
        'compute_trajectory',
        False,
        lambda m: m._transfuser_model,
        _no_choice,
        fp32=True,
    ),
    'drivor': AgentSpec(
        'drivor',
        {'ckpt': 'drivor/nav3.ckpt'},
        _build_drivor,
        parse_drivor,
        'compute_trajectory',
        False,
        lambda m: m._drivor_model,
        _drivor_choice,
        fp32=True,
    ),
    'gtrs': AgentSpec(
        'gtrs',
        {'ckpt': 'gtrs/gtrs_dense_vov.ckpt', 'vocab': 'gtrs/16384.npy'},
        _build_gtrs,
        parse_gtrs,
        'get_trajectory_from_agent_input',
        True,
        lambda m: m.model,
        _vocab_choice,
    ),
    'ztrs': AgentSpec(
        'ztrs',
        {'ckpt': 'ztrs/ztrs_vov.ckpt', 'vocab': 'ztrs/16384.npy'},
        _build_ztrs,
        parse_gtrs,
        'get_trajectory_from_agent_input',
        True,
        lambda m: m.model,
        _vocab_choice,
    ),
    'gtrs_aug': AgentSpec(
        'gtrs_aug',
        {'ckpt': 'gtrs_aug/gtrs_aug_vov.ckpt', 'vocab': 'gtrs_aug/8192.npy'},
        _build_gtrs_aug,
        parse_gtrs,
        'get_trajectory_from_agent_input',
        True,
        lambda m: m.model.teacher.model,
        _gtrs_aug_choice,
    ),
}


def resolve_device(device=None):
    """``device`` or ``$HUGSIM_AGENT_DEVICE`` or ``cuda``; ``cuda`` must be available."""
    device = device or os.environ.get('HUGSIM_AGENT_DEVICE') or 'cuda'
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(
            f'device {device} requested but no CUDA device is visible; pass --device cpu '
            f'(or set HUGSIM_AGENT_DEVICE=cpu) to run on the CPU'
        )
    return torch.device(device)


def ckpt_paths(name, ckpt_root=None):
    """The files the agent needs under the checkpoint root, or an error naming the first missing one."""
    root = ckpt_root or os.environ.get('HUGSIM_AGENT_CKPTS')
    if not root:
        raise RuntimeError('no checkpoint root: pass --ckpt-root or set HUGSIM_AGENT_CKPTS')
    paths = {role: os.path.join(root, rel) for role, rel in AGENTS[name].files.items()}
    for role, path in paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f'{name}: expected {role} file {path} (checkpoint root {root}), not found')
    return paths


@contextlib.contextmanager
def full_fp32(enabled):
    """TF32 off for cuDNN and matmuls inside the block, and the caller's flags back after it.

    Scoped to the step rather than set at import: in closed_loop.py's --in_process mode the agent
    shares the simulator's process, whose own flags must not change.
    """
    if not enabled:
        yield
        return
    saved = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved


class Policy:
    """An agent on a device: ``step(obs, info)`` is the function ``serve`` calls.

    ``seed``, when given, seeds torch before each step with ``seed + step index``. The original
    clients never seeded; this only matters for DrivoR, whose CEM refinement samples noise, and
    makes its plans repeatable (``tools/plans.py`` uses it).
    """

    def __init__(self, name, ckpt_root=None, device=None, seed=None):
        self.spec = AGENTS[name]
        self.device = resolve_device(device)
        paths = ckpt_paths(name, ckpt_root)
        self.model = self.spec.build(paths).to(self.device).eval()
        self.seed = seed
        self.count = 0
        self.last_raw_imgs = None
        self.last_way_points = None
        self.last_choice = None
        self.spec.network(self.model).register_forward_hook(
            lambda module, args, output: setattr(self, 'last_choice', self.spec.choice(output))
        )

    def step(self, obs, info):
        agent_input, raw_imgs = self.spec.parse(obs, info)
        t = time.time()
        # A seeded step draws from a forked RNG: in closed_loop.py's --in_process mode the global
        # one is the simulator's, whose adversary and planner sample from it.
        devices = [self.device] if self.seed is not None and self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices, enabled=self.seed is not None):
            if self.seed is not None:
                torch.manual_seed(self.seed + self.count)
            with torch.no_grad(), full_fp32(self.spec.fp32 and self.device.type == 'cuda'):
                traj = getattr(self.model, self.spec.compute)(agent_input)
        way_points = to_plan(traj.poses, self.spec.subsample)
        print(f'[{self.spec.name}] step {self.count}: {len(way_points)} waypoints in {time.time() - t:.2f} s', flush=True)
        self.count += 1
        self.last_raw_imgs = raw_imgs
        self.last_way_points = way_points
        return way_points[:, :2]
