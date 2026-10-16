"""TransFuser config, with the values the HUGSIM Latent TransFuser client ran with.

From NAVSIM (autonomousvision/navsim, Apache-2.0) ``navsim/agents/transfuser/transfuser_config.py``
and hyzhou404/NAVSIM's ``config/HUGSIM/transfuser.yaml`` + ``ltf_e2e.py`` (``latent=True``). The
training-only fields (loss weights, the nuplan BEV-map class table) are dropped.
"""

from dataclasses import dataclass

import numpy as np

from ..common import TrajectorySampling


@dataclass
class TransfuserConfig:
    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    image_architecture: str = 'resnet34'
    lidar_architecture: str = 'resnet34'

    latent: bool = True
    latent_rad_thresh: float = 4 * np.pi / 9

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    lidar_seq_len: int = 1

    camera_width: int = 1024
    camera_height: int = 256
    lidar_resolution_width: int = 256
    lidar_resolution_height: int = 256

    img_vert_anchors: int = 256 // 32
    img_horz_anchors: int = 1024 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    block_exp: int = 4
    n_layer: int = 2  # transformer layers used in the vision backbone
    n_head: int = 4
    n_scale: int = 4
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    gpt_linear_layer_init_mean: float = 0.0
    gpt_linear_layer_init_std: float = 0.02
    gpt_layer_norm_init_weight: float = 1.0

    perspective_downsample_factor: int = 1
    transformer_decoder_join: bool = True
    detect_boxes: bool = True
    use_bev_semantic: bool = True
    use_semantic: bool = False
    use_depth: bool = False
    add_features: bool = True

    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    num_bounding_boxes: int = 30

    num_bev_classes: int = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2
