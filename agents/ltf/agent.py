"""Latent TransFuser agent (``latent=True``: the LiDAR branch is a learned constant image).

From NAVSIM (autonomousvision/navsim, Apache-2.0) ``navsim/agents/transfuser/transfuser_agent.py``
and ``navsim/agents/abstract_agent.py`` (``compute_trajectory``) via hyzhou404/NAVSIM.
"""

import torch

from .._navsim.base import load_checkpoint, model_device, to_batch
from .._navsim.common import AgentInput, Trajectory
from .config import TransfuserConfig
from .features import TransfuserFeatureBuilder
from .model import TransfuserModel


class TransfuserAgent(torch.nn.Module):
    """Agent interface for TransFuser baseline."""

    def __init__(self, config: TransfuserConfig):
        super().__init__()
        self._config = config
        self._transfuser_model = TransfuserModel(config)
        self._feature_builder = TransfuserFeatureBuilder(config)

    def load_checkpoint(self, checkpoint_path):
        state_dict = load_checkpoint(checkpoint_path, 'ltf')
        result = self.load_state_dict({k.replace('agent.', ''): v for k, v in state_dict.items()})
        print(f'[ltf] loaded {checkpoint_path}: {result}', flush=True)

    def forward(self, features):
        return self._transfuser_model(features)

    @torch.no_grad()
    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        self.eval()
        features = to_batch(self._feature_builder.compute_features(agent_input), model_device(self))
        poses = self.forward(features)['trajectory'].squeeze(0).cpu().numpy()
        return Trajectory(poses, self._config.trajectory_sampling)
