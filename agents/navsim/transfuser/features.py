"""TransFuser input features (stitched camera, LiDAR histogram, ego status).

From NAVSIM (autonomousvision/navsim, Apache-2.0) ``navsim/agents/transfuser/transfuser_features.py``
via hyzhou404/NAVSIM (``TransfuserFeatureBuilder``); the target builders are dropped.
"""

from typing import Dict

import cv2
import numpy as np
import torch
from torchvision import transforms

from ..common import AgentInput
from .config import TransfuserConfig

LIDAR_POSITION = slice(0, 3)  # LidarIndex.POSITION


class TransfuserFeatureBuilder:
    """Input feature builder for TransFuser."""

    def __init__(self, config: TransfuserConfig):
        self._config = config

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        features = {}

        features['camera_feature'] = self._get_camera_feature(agent_input)
        features['lidar_feature'] = self._get_lidar_feature(agent_input)
        features['status_feature'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),
            ],
        )

        return features

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """Stitched front view image as a torch tensor."""

        cameras = agent_input.cameras[-1]

        # Crop to ensure 4:1 aspect ratio
        l0 = cameras.cam_l0.image[28:-28, 416:-416]
        f0 = cameras.cam_f0.image[28:-28]
        r0 = cameras.cam_r0.image[28:-28, 416:-416]

        # stitch l0, f0, r0 images
        stitched_image = np.concatenate([l0, f0, r0], axis=1)
        resized_image = cv2.resize(stitched_image, (1024, 256))
        tensor_image = transforms.ToTensor()(resized_image)

        return tensor_image

    def _get_lidar_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """LiDAR feature as a 2D histogram, according to Transfuser."""

        # only consider (x,y,z) & swap axes for (N,3) numpy array
        lidar_pc = agent_input.lidars[-1].lidar_pc[LIDAR_POSITION].T

        # NOTE: Code from
        # https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py#L873
        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(
                self._config.lidar_min_x,
                self._config.lidar_max_x,
                (self._config.lidar_max_x - self._config.lidar_min_x) * int(self._config.pixels_per_meter) + 1,
            )
            ybins = np.linspace(
                self._config.lidar_min_y,
                self._config.lidar_max_y,
                (self._config.lidar_max_y - self._config.lidar_min_y) * int(self._config.pixels_per_meter) + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self._config.hist_max_per_pixel] = self._config.hist_max_per_pixel
            overhead_splat = hist / self._config.hist_max_per_pixel
            return overhead_splat

        # Remove points above the vehicle
        lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.max_height_lidar]
        below = lidar_pc[lidar_pc[..., 2] <= self._config.lidar_split_height]
        above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]
        above_features = splat_points(above)
        if self._config.use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)

        return torch.tensor(features)
