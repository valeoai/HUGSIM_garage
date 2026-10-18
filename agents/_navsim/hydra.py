"""Pieces of the GTRS dense model that GTRS and ZTRS share: the VoVNet backbone and the input features.

From NVlabs/GTRS (Apache-2.0) ``navsim/agents/gtrs_dense/hydra_backbone.py`` (``HydraBackbone``) and
``hydra_features.py`` (``HydraFeatureBuilder``), as in hyzhou404/NAVSIM. ZTRS reuses both
unchanged. The module lives here, and not in ``gtrs/``, so that no agent folder imports another.
The pretrained backbone load (the checkpoint carries it), the target builder, the left/right/back
view options (off in the shipped configs) and the unused ``ego_pose``/``ego_velocity``/
``ego_acceleration`` outputs are dropped.
"""

from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from .common import AgentInput
from .vov import VoVNet


class HydraBackbone(nn.Module):
    def __init__(self, config):

        super().__init__()
        self.config = config
        self.backbone_type = config.backbone_type
        if config.backbone_type == 'vov':
            self.image_encoder = VoVNet(
                spec_name='V-99-eSE',
                out_features=['stage4', 'stage5'],
                norm_eval=True,
                with_cp=True,
            )
            vit_channels = 1024
        else:
            raise ValueError('Unsupported backbone')

        self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors))
        self.img_feat_c = vit_channels

    def forward(self, image):
        B, C, H, W = image.shape
        if self.backbone_type == 'vov':
            image_features = self.image_encoder(image)[-1]
        else:
            raise ValueError('Forward wrong backbone')
        return self.avgpool_img(image_features)


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
