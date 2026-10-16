"""GTRS-Aug input features: the three front cameras stitched to one strip, and the ego status.

From NVlabs/GTRS (Apache-2.0) ``navsim/agents/gtrs_aug/hydra_features_aug.py``
(``HydraAugFeatureBuilder``), and the status features of ``GTRSAugAgent.get_trajectory_from_agent_input``,
as in hyzhou404/NAVSIM. Only the teacher image is built. The original also built the student image
with random colour jitter, blur and solarisation, which the teacher inference never reads, and the
rotated crops, which are only built when training.
"""

from typing import Dict

import cv2
import numpy as np
import torch
from torchvision import transforms

from ..common import AgentInput
from .config import GTRSAugConfig


class HydraAugFeatureBuilder:
    def __init__(self, config: GTRSAugConfig):
        self._config = config
        self.teacher_ori_augmentation = transforms.Compose([transforms.ToTensor()])

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        features = {'ori_teacher': self._get_camera_feature(agent_input)}

        status_list = []
        for i in range(self._config.num_ego_status):
            idx = -(i + 1)
            status_list.append(
                torch.concatenate(
                    [
                        torch.tensor(agent_input.ego_statuses[idx].driving_command, dtype=torch.float32),
                        torch.tensor(agent_input.ego_statuses[idx].ego_velocity, dtype=torch.float32),
                        torch.tensor(agent_input.ego_statuses[idx].ego_acceleration, dtype=torch.float32),
                    ]
                )
            )
        features['status_feature'] = status_list
        return features

    def _get_camera_feature(self, agent_input: AgentInput):
        """The stitched front view of the last ``seq_len`` frames, a list of (3, H, W) tensors."""
        seq_len = self._config.seq_len
        cameras = agent_input.cameras[-seq_len:]  # List[Cameras]
        assert len(cameras) == seq_len

        ori_teacher = []
        for camera in cameras:
            image = camera.cam_l0.image
            if image is not None and image.size > 0 and np.any(image):
                # Crop to ensure 4:1 aspect ratio
                l0 = camera.cam_l0.image[28:-28, 416:-416]
                f0 = camera.cam_f0.image[28:-28]
                r0 = camera.cam_r0.image[28:-28, 416:-416]

                assert self._config.n_camera == 3
                ori_image = np.concatenate([l0, f0, r0], axis=1)

                _ori_image = cv2.resize(ori_image, (self._config.camera_width, self._config.camera_height))
                ori_teacher.append(self.teacher_ori_augmentation(_ori_image))

        return ori_teacher
