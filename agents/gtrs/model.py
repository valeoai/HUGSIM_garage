"""GTRS dense model: VoVNet image tokens, a vocabulary of 16384 trajectories scored by a transformer.

From NVlabs/GTRS (Apache-2.0) ``navsim/agents/gtrs_dense/hydra_model.py``, as in our NAVSIM checkout
(``hydra_backbone.py`` is ``agents/_navsim/hydra.py``). Dropped: the pretrained backbone load (the checkpoint carries it), the
training-time vocabulary dropout, the NeRF/back-view options and the diffusion-proposal evaluation
path (all off in the shipped config).
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .._navsim.agent_head import AgentHead
from .._navsim.attn import MemoryEffTransformer
from .._navsim.hydra import HydraBackbone
from .config import GTRSConfig


class HydraModel(nn.Module):
    def __init__(self, config: GTRSConfig):
        super().__init__()

        self._query_splits = [
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = HydraBackbone(config)

        self._keyval_embedding = nn.Embedding(
            config.img_vert_anchors * config.img_horz_anchors, config.tf_d_model
        )  # 8x8 feature grid + trajectory
        # The next three modules are not used at inference; they are kept so that the checkpoint's
        # weights for them load without being reported as unexpected.
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self.downscale_layer = nn.Conv2d(self._backbone.img_feat_c, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear((4 + 2 + 2) * config.num_ego_status, config.tf_d_model)

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

        self._trajectory_head = HydraTrajHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            vocab_path=config.vocab_path,
            config=config,
        )

    def img_feat_blc(self, camera_feature):
        img_features = self._backbone(camera_feature)
        img_features = self.downscale_layer(img_features).flatten(-2, -1)
        img_features = img_features.permute(0, 2, 1)
        return img_features

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        status_feature: torch.Tensor = features['status_feature'][0]
        camera_feature = features['camera_feature']

        if self._config.num_ego_status == 1 and status_feature.shape[1] == 32:
            status_encoding = self._status_encoding(status_feature[:, :8])
        else:
            status_encoding = self._status_encoding(status_feature)

        # original
        if isinstance(camera_feature, list):
            camera_feature = camera_feature[-1]
        img_features = self.img_feat_blc(camera_feature)
        keyval = img_features

        keyval += self._keyval_embedding.weight[None, ...]

        output: Dict[str, torch.Tensor] = {}
        trajectory = self._trajectory_head(keyval, status_encoding)
        output.update(trajectory)
        return output


class HydraTrajHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int, vocab_path: str,
                 nhead: int, nlayers: int, config: GTRSConfig = None
                 ):
        super().__init__()
        self.config = config
        self._num_poses = num_poses
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, d_ffn,
                dropout=0.0, batch_first=True
            ), nlayers
        )
        self.vocab = nn.Parameter(
            torch.from_numpy(np.load(vocab_path)),
            requires_grad=False
        )

        self.heads = nn.ModuleDict({
            'no_at_fault_collisions': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'drivable_area_compliance':
                nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
            'time_to_collision_within_bound': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'ego_progress': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'driving_direction_compliance': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'lane_keeping': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'traffic_light_compliance': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'imi': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            )
        })

        self.normalize_vocab_pos = config.normalize_vocab_pos
        if self.normalize_vocab_pos:
            self.encoder = MemoryEffTransformer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.0
            )
        self.pos_embed = nn.Sequential(
            nn.Linear(num_poses * 3, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, bev_feature, status_encoding) -> Dict[str, torch.Tensor]:
        result = {}
        # vocab: 4096, 40, 3
        # bev_feature: B, 32, C
        # embedded_vocab: B, 4096, C
        vocab = self.vocab.data
        L, HORIZON, _ = vocab.shape
        B = bev_feature.shape[0]

        if self.normalize_vocab_pos:
            embedded_vocab = self.pos_embed(vocab.view(L, -1))[None]
            embedded_vocab = self.encoder(embedded_vocab).repeat(B, 1, 1)
        else:
            embedded_vocab = self.pos_embed(vocab.view(L, -1))[None].repeat(B, 1, 1)
        tr_out = self.transformer(embedded_vocab, bev_feature)
        dist_status = tr_out + status_encoding.unsqueeze(1)

        # selected_indices: B,
        for k, head in self.heads.items():
            result[k] = head(dist_status).squeeze(-1)

        scores = (
                0.03 * result['imi'].softmax(-1).log() +
                0.1 * result['traffic_light_compliance'].sigmoid().log() +
                0.1 * result['no_at_fault_collisions'].sigmoid().log() +
                0.9 * result['drivable_area_compliance'].sigmoid().log() +
                0.2 * result['driving_direction_compliance'].sigmoid().log() +
                6.0 * (7.0 * result['time_to_collision_within_bound'].sigmoid() +
                       7.0 * result['ego_progress'].sigmoid() +
                       3.0 * result['lane_keeping'].sigmoid()
                       ).log()
        )

        selected_indices = scores.argmax(1)
        result["trajectory"] = self.vocab.data[selected_indices]
        result["trajectory_vocab"] = self.vocab.data
        result["selected_indices"] = selected_indices
        return result
