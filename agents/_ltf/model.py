"""TransFuser model (the Latent TransFuser planner).

From NAVSIM (autonomousvision/navsim, Apache-2.0) ``navsim/agents/transfuser/transfuser_model.py``
via hyzhou404/NAVSIM; ``StateSE2Index`` comes from the same repo's ``common/enums.py``, as plain
constants. ``AgentHead`` of the same file is in ``agents/_navsim/agent_head.py``. The delta variant of
``TrajectoryHead`` and ``cumsum_traj`` with its statistics are from BeyondDrive (wjl2244/BeyondDrive,
Apache-2.0) ``navsim/agents/transfuser/transfuser_model.py`` and ``utils.py``.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .._navsim.agent_head import AgentHead
from .backbone import TransfuserBackbone
from .config import TransfuserConfig

# StateSE2Index: poses are (x, y, heading)
STATE_SE2_SIZE = 3
HEADING = 2

# BeyondDrive's per-step displacement statistics (``utils.py``), which normalize its delta head's output.
X_DIFF_MIN, X_DIFF_MAX, X_DIFF_MEAN = -1.2698211669921875, 7.475563049316406, 2.950225591659546
Y_DIFF_MIN, Y_DIFF_MAX, Y_DIFF_MEAN = -5.012081146240234, 4.8563690185546875, 0.0607292577624321


def cumsum_traj(norm_trajs):
    """BeyondDrive's ``cumsum_traj``: rows (dx, dy, sin, cos), normalized, to poses (x, y, heading)."""
    heading = torch.atan2(norm_trajs[..., 2:3], norm_trajs[..., 3:4])
    x_diff_range = max(abs(X_DIFF_MAX - X_DIFF_MEAN), abs(X_DIFF_MIN - X_DIFF_MEAN))
    x_diff = norm_trajs[..., 0:1] * x_diff_range + X_DIFF_MEAN
    y_diff_range = max(abs(Y_DIFF_MAX - Y_DIFF_MEAN), abs(Y_DIFF_MIN - Y_DIFF_MEAN))
    y_diff = norm_trajs[..., 1:2] * y_diff_range + Y_DIFF_MEAN
    return torch.cat([x_diff.cumsum(dim=1), y_diff.cumsum(dim=1), heading], -1)


class TransfuserModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = TransfuserBackbone(config)

        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            use_delta_traj=config.use_delta_traj,
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)

        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}
        trajectory = self._trajectory_head(trajectory_query)
        output.update(trajectory)

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output


class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, use_delta_traj: bool = False):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self._use_delta_traj = use_delta_traj

        if use_delta_traj:  # BeyondDrive: (dx, dy, sin, cos) per pose, through a d_model-wide SiLU layer
            self._mlp = nn.Sequential(
                nn.Linear(self._d_model, self._d_model),
                nn.SiLU(),
                nn.Linear(self._d_model, num_poses * (STATE_SE2_SIZE + 1)),
            )
        else:
            self._mlp = nn.Sequential(
                nn.Linear(self._d_model, self._d_ffn),
                nn.ReLU(),
                nn.Linear(self._d_ffn, num_poses * STATE_SE2_SIZE),
            )

    def forward(self, object_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        if self._use_delta_traj:
            delta = self._mlp(object_queries).reshape(-1, self._num_poses, STATE_SE2_SIZE + 1)
            return {"trajectory": cumsum_traj(delta)}
        poses = self._mlp(object_queries).reshape(-1, self._num_poses, STATE_SE2_SIZE)
        poses[..., HEADING] = poses[..., HEADING].tanh() * np.pi
        return {"trajectory": poses}
