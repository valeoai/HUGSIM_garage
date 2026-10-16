"""The few NAVSIM data classes inference needs, without the nuplan devkit.

Adapted from NAVSIM (autonomousvision/navsim, Apache-2.0) ``navsim/common/dataclasses.py``,
``navsim/common/enums.py`` and nuplan-devkit's ``TrajectorySampling``, via hyzhou404/NAVSIM, which
HUGSIM's NAVSIM clients were written against. Only the fields the five agents read are kept.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class TrajectorySampling:
    """``num_poses`` poses ``interval_length`` seconds apart, as nuplan's TrajectorySampling."""

    time_horizon: float
    interval_length: float

    @property
    def num_poses(self) -> int:
        return int(round(self.time_horizon / self.interval_length))


@dataclass
class Camera:
    image: Optional[np.ndarray] = None
    sensor2lidar_rotation: Optional[np.ndarray] = None
    sensor2lidar_translation: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None
    distortion: Optional[np.ndarray] = None


@dataclass
class Cameras:
    cam_f0: Camera = field(default_factory=Camera)
    cam_l0: Camera = field(default_factory=Camera)
    cam_l1: Camera = field(default_factory=Camera)
    cam_l2: Camera = field(default_factory=Camera)
    cam_r0: Camera = field(default_factory=Camera)
    cam_r1: Camera = field(default_factory=Camera)
    cam_r2: Camera = field(default_factory=Camera)
    cam_b0: Camera = field(default_factory=Camera)


@dataclass
class Lidar:
    # (6, n): x, y, z, intensity, ring, lidar_id
    lidar_pc: Optional[np.ndarray] = None


@dataclass
class EgoStatus:
    ego_pose: np.ndarray
    ego_velocity: np.ndarray
    ego_acceleration: np.ndarray
    driving_command: np.ndarray
    in_global_frame: bool = False


@dataclass
class AgentInput:
    ego_statuses: List[EgoStatus]
    cameras: List[Cameras]
    lidars: List[Lidar]


@dataclass
class Trajectory:
    poses: np.ndarray  # (num_poses, 3): x forward, y left, heading
    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    def __post_init__(self):
        assert self.poses.ndim == 2, 'Trajectory poses should have two dimensions for samples and poses.'
        assert (
            self.poses.shape[0] == self.trajectory_sampling.num_poses
        ), 'Trajectory poses and sampling have unequal number of poses.'
        assert self.poses.shape[1] == 3, 'Trajectory requires (x, y, heading) at last dim.'
