"""GTRS (dense, vocabulary-scoring) config, with the values the HUGSIM ``gtrs_e2e.py`` client ran with.

The values are those hydra resolved from hyzhou404/NAVSIM's ``config/HUGSIM/gtrs.yaml`` +
``config/common/agent/gtrs_dense_vov.yaml`` (NVlabs/GTRS) for ``agent.config``, over the defaults
of ``HydraConfig``/``TransfuserConfig``. Only the fields inference reads are kept.
"""

from dataclasses import dataclass

from .._navsim.common import TrajectorySampling


@dataclass
class GTRSConfig:
    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.1)

    num_bounding_boxes: int = 30
    backbone_type: str = 'vov'
    img_vert_anchors: int = 16
    img_horz_anchors: int = 64

    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    num_ego_status: int = 1
    vadv2_head_nhead: int = 8
    vadv2_head_nlayers: int = 3
    vocab_size: int = 16384
    vocab_path: str = None  # set from --ckpt-root by the entry point
    normalize_vocab_pos: bool = True

    camera_width: int = 2048
    camera_height: int = 512
    seq_len: int = 1
