"""Bounding-box prediction head of TransFuser, shared by LTF, GTRS and ZTRS.

From NAVSIM (autonomousvision/navsim, Apache-2.0) ``navsim/agents/transfuser/transfuser_model.py``
(``AgentHead``) via hyzhou404/NAVSIM; ``BoundingBox2DIndex`` comes from the same repo's
``transfuser_features.py``, as plain constants. The module lives here, and not in ``ltf/``, so that
no agent folder imports another one.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

# BoundingBox2DIndex: boxes are (x, y, heading, length, width)
BBOX_SIZE = 5
BBOX_POINT = slice(0, 2)
BBOX_HEADING = 2


class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BBOX_SIZE),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BBOX_POINT] = agent_states[..., BBOX_POINT].tanh() * 32
        agent_states[..., BBOX_HEADING] = agent_states[..., BBOX_HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}
