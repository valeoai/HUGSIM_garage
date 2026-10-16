"""GTRS agent: checkpoint loading and inference.

From NVlabs/GTRS (Apache-2.0) ``navsim/agents/gtrs_dense/gtrs_agent.py`` (``initialize`` and
``get_trajectory_from_agent_input``), as in hyzhou404/NAVSIM. The losses, optimisers and callbacks
are dropped.
"""

import torch

from ..base import load_checkpoint, model_device, to_batch
from ..common import AgentInput, Trajectory
from .config import GTRSConfig
from .features import HydraFeatureBuilder
from .model import HydraModel


class GTRSAgent(torch.nn.Module):
    def __init__(self, config: GTRSConfig):
        super().__init__()
        self._config = config
        self._trajectory_sampling = config.trajectory_sampling
        self.model = HydraModel(config)
        self._feature_builder = HydraFeatureBuilder(config)

    def load_checkpoint(self, checkpoint_path):
        state_dict = load_checkpoint(checkpoint_path, 'gtrs')
        # The vocabulary is read from the .npy file, as in the original, not from the checkpoint.
        for k in [k for k in state_dict if 'model._trajectory_head.vocab' in k]:
            del state_dict[k]

        result = self.load_state_dict({k.replace('agent.', ''): v for k, v in state_dict.items()}, strict=False)
        print(f'[gtrs] loaded {checkpoint_path}: {result}', flush=True)
        return result

    def forward(self, features):
        return self.model(features)

    @torch.no_grad()
    def get_trajectory_from_agent_input(self, agent_input: AgentInput) -> Trajectory:
        self.eval()
        features = to_batch(self._feature_builder.compute_features(agent_input), model_device(self.model))
        poses = self.forward(features)['trajectory'].squeeze(0).cpu().numpy()
        return Trajectory(poses, self._trajectory_sampling)
