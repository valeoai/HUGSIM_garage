"""DrivoR config, with the values the HUGSIM ``dynamo_e2e.py`` client ran with: TOAD's.

``agents/toad`` uses these defaults; ``agents/drivor`` turns CEM off, as the DrivoR paper's HUGSIM
evaluation ran it (see its ``spec.py``).

The values are those hydra resolved from hyzhou404/NAVSIM's ``config/common/agent/drivoR.yaml``
(valeoai/DrivoR) for ``agent.config``. Training-only keys, the LiDAR branch (``lidar_pc: []`` in the
shipped config) and the pretrained-weight paths (the checkpoint carries the whole backbone) are
dropped. The CEM keys the original read with ``getattr(cfg, key, default)`` are fields here, with
those defaults.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .._navsim.common import TrajectorySampling


@dataclass
class ImageBackboneConfig:
    model_name: str = 'timm/vit_small_patch14_reg4_dinov2.lvd142m'
    use_lora: bool = True
    finetune: bool = False
    lora_rank: int = 32
    focus_front_cam: bool = False
    use_feature_pooling: bool = False
    compress_fc: bool = False
    # filled in by the model from the DrivoR config, as the original did
    image_size: Tuple[int, int] = None
    num_scene_tokens: int = None
    tf_d_model: int = None
    in_chans: int = 3


@dataclass
class DrivoRConfig:
    b2d: bool = False
    shared_refiner: bool = False
    ref_num: int = 4  # number of refining steps
    scorer_ref_num: int = 4

    proposal_num: int = 64  # number of proposals (trajectories)
    num_poses: int = 8  # size of the trajectories

    num_scene_tokens: int = 16
    one_token_per_traj: bool = True
    full_history_status: bool = False

    long_trajectory_additional_poses: int = -1

    noc: float = 1.0
    dac: float = 1.0
    ddc: float = 0.0
    ttc: float = 5.0
    ep: float = 5.0
    comfort: float = 2.0

    tf_d_model: int = 256
    tf_d_ffn: int = 1024

    # cameras in the order of the original sensor config; an empty list is a camera not used
    cams: Dict[str, List[int]] = field(
        default_factory=lambda: {
            'cam_f0': [3],
            'cam_l0': [3],
            'cam_l1': [],
            'cam_l2': [],
            'cam_r0': [3],
            'cam_r1': [],
            'cam_r2': [],
            'cam_b0': [3],
        }
    )

    use_cem: bool = True
    cem_seed_topk: int = 64
    cem_num_samples: int = 64
    cem_num_iterations: int = 5
    cem_dt: float = 0.5
    cem_alpha: float = 0.7
    cem_std_floor_a: float = 0.1
    cem_std_floor_omega: float = 0.025
    cem_comfort_weight: float = 0.05
    cem_anchor_weight: float = 0.5
    cem_init_std_scale: float = 0.5

    image_size: Tuple[int, int] = (1148, 672)
    image_backbone: ImageBackboneConfig = field(default_factory=ImageBackboneConfig)

    refiner_num_heads: int = 1
    refiner_ls_values: float = 0.0

    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
