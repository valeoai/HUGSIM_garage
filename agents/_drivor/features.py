"""DrivoR input features: the cameras resized and normalised, and the 11-D ego status.

From valeoai/DrivoR ``navsim/agents/drivoR/drivor_features.py`` (``DrivoRFeatureBuilder``). The
LiDAR feature (the shipped config has no LiDAR), the camera intrinsics and extrinsics tensors
(computed but never read by the model) and the target builder are dropped.
"""

from typing import Dict

import numpy as np
import torch
from PIL import Image

from .._navsim.common import AgentInput
from .config import DrivoRConfig


class DrivoRFeatureBuilder:
    def __init__(self, config: DrivoRConfig):
        self._config = config

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        features = {}
        features.update(self._get_camera_feature(agent_input))

        ego_feature_list = []

        for ego_status in agent_input.ego_statuses:
            if ego_status is None:
                continue
            pose = torch.tensor(ego_status.ego_pose, dtype=torch.float32)
            velocity = torch.tensor(ego_status.ego_velocity, dtype=torch.float32)
            acceleration = torch.tensor(ego_status.ego_acceleration, dtype=torch.float32)
            driving_command = torch.tensor(ego_status.driving_command, dtype=torch.float32)
            ego_feature = torch.cat([pose, velocity, acceleration, driving_command], dim=-1)

            ego_feature_list.append(ego_feature)

        features['ego_status'] = torch.stack(ego_feature_list)

        return features

    def _get_camera_feature(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """The cameras in the order front, back, left, right, as (n_cams, 3, H, W)."""

        cameras = agent_input.cameras[-1]

        # this is a change for the focus front cam
        cameras = [
            cameras.cam_f0,
            cameras.cam_b0,
            cameras.cam_l0,
            cameras.cam_l1,
            cameras.cam_l2,
            cameras.cam_r0,
            cameras.cam_r1,
            cameras.cam_r2,
        ]

        images = []
        for cam in cameras:
            if cam is None or cam.image is None:
                continue

            im = Image.fromarray(cam.image)

            # image resize
            im = im.resize(self._config.image_size)

            # PIL to numpy and normalize
            im = np.asarray(im, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            im = (im - mean) / std

            # convert to torch
            im = torch.from_numpy(im).permute(2, 0, 1)

            images.append(im)

        return {'image': torch.stack(images)}
