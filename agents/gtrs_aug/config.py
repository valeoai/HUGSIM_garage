"""GTRS-Aug config, with the values the HUGSIM ``gtrs_aug_e2e.py`` client ran with.

The values are those hydra resolved from our NAVSIM checkout's ``config/HUGSIM/gtrs_aug.yaml`` +
``config/common/agent/gtrs_aug_vov.yaml`` (NVlabs/GTRS) for ``agent.config``, over the defaults of
``HydraConfigAug``/``TransfuserConfig``: the teacher network, one refinement stage, no
augmentation. Only the fields inference reads are kept.
"""

from dataclasses import dataclass, field

from .._navsim.common import TrajectorySampling


@dataclass
class InferConfig:
    model: str = 'teacher'  # teacher or student
    use_aug: bool = True


@dataclass
class RefinementConfig:
    use_multi_stage: bool = True
    refinement_approach: str = 'offset_decoder'
    num_refinement_stage: int = 1
    stage_layers: int = 3
    topks: int = 256

    use_mid_output: bool = True
    use_offset_refinement_v2: bool = False
    use_separate_stage_heads: bool = True


@dataclass
class LabConfig:
    use_first_stage_traj_in_infer: bool = False
    use_imi_learning_in_refinement: bool = True
    optimize_prev_frame_traj_for_ec: bool = False
    refinement_metrics: str = 'all'
    adjust_refinement_score_weight: bool = False
    use_higher_res_feat_in_refinement: bool = False


@dataclass
class GTRSAugConfig:
    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.1)

    backbone_type: str = 'vov'
    img_vert_anchors: int = 16
    img_horz_anchors: int = 64

    tf_d_model: int = 256
    tf_d_ffn: int = 1024

    num_ego_status: int = 1
    vadv2_head_nhead: int = 8
    vadv2_head_nlayers: int = 3
    vocab_size: int = 8192
    vocab_path: str = None  # set from --ckpt-root by the entry point
    normalize_vocab_pos: bool = True

    n_camera: int = 3
    camera_width: int = 2048
    camera_height: int = 512
    seq_len: int = 1

    training: bool = False
    inference: InferConfig = field(default_factory=InferConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    lab: LabConfig = field(default_factory=LabConfig)
