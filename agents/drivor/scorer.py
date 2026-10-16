"""DrivoR scorer: one MLP head per sub-score of the trajectory score.

From valeoai/DrivoR ``navsim/agents/drivoR/score_module/scorer.py``. The agent/area/BEV heads that
the shipped config disables (``double_score``, ``agent_pred``, ``area_pred``, ``bev_map``,
``bev_agent`` are all false) and the training-only outputs are dropped; the sub-score heads are
unchanged, so the checkpoint's keys are the same.
"""

import torch.nn as nn

from .config import DrivoRConfig


def _head(d_model, d_ffn):
    return nn.Sequential(
        nn.Linear(d_model, d_ffn),
        nn.ReLU(),
        nn.Linear(d_ffn, 1),
    )


class Scorer(nn.Module):
    def __init__(self, config: DrivoRConfig):
        super().__init__()

        d_model, d_ffn = config.tf_d_model, config.tf_d_ffn
        self.pred_score = nn.ModuleDict(
            {
                'no_at_fault_collisions': _head(d_model, d_ffn),
                'drivable_area_compliance': _head(d_model, d_ffn),
                'time_to_collision_within_bound': _head(d_model, d_ffn),
                'ego_progress': _head(d_model, d_ffn),
                'driving_direction_compliance': _head(d_model, d_ffn),
                'comfort': _head(d_model, d_ffn),
            }
        )

    def forward(self, proposal_feature):
        return {k: head(proposal_feature).squeeze(-1) for k, head in self.pred_score.items()}
