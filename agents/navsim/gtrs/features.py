"""GTRS / ZTRS input features: the three front cameras stitched to one strip, and the ego status.

From NVlabs/GTRS (Apache-2.0) ``navsim/agents/gtrs_dense/hydra_features.py`` (``HydraFeatureBuilder``),
as in hyzhou404/NAVSIM. The target builder, the left/right/back-view options (off in the shipped
configs) and the unused ``ego_pose``/``ego_velocity``/``ego_acceleration`` outputs are dropped.
"""

from typing import Dict

import cv2
import numpy as np
import torch
from torchvision import transforms

from ..common import AgentInput


class HydraFeatureBuilder:
    def __init__(self, config):
        self._config = config

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        features = {}

        features['camera_feature'] = self._get_camera_feature(agent_input)

        ego_status_list = []
        for i in range(self._config.num_ego_status):
            idx = -(i + 1)

            ego_status = torch.concatenate(
                [
                    torch.tensor(agent_input.ego_statuses[idx].driving_command, dtype=torch.float32),
                    torch.tensor(agent_input.ego_statuses[idx].ego_velocity, dtype=torch.float32),
                    torch.tensor(agent_input.ego_statuses[idx].ego_acceleration, dtype=torch.float32),
                ],
            )

            ego_status_list.append(ego_status)

        features['status_feature'] = ego_status_list
        return features

    def _get_camera_feature(self, agent_input: AgentInput):
        """The stitched front view of the last ``seq_len`` frames, a list of (3, H, W) tensors."""
        seq_len = self._config.seq_len
        cameras = agent_input.cameras[-seq_len:]
        assert len(cameras) == seq_len
        image_list = []
        for camera in cameras:
            image = camera.cam_l0.image
            if image is not None and image.size > 0 and np.any(image):
                l0 = camera.cam_l0.image[28:-28, 416:-416]
                f0 = camera.cam_f0.image[28:-28]
                r0 = camera.cam_r0.image[28:-28, 416:-416]
                # stitch l0, f0, r0 images
                stitched_image = np.concatenate([l0, f0, r0], axis=1)
                resized_image = cv2.resize(stitched_image, (self._config.camera_width, self._config.camera_height))
                tensor_image = transforms.ToTensor()(resized_image)
                image_list.append(tensor_image)
            else:
                image_list.append(None)

        return image_list
