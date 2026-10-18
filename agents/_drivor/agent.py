"""DrivoR agent, shared by ``drivor`` and ``toad``: checkpoint loading and inference.

From valeoai/DrivoR ``navsim/agents/drivoR/drivor_agent.py`` and NAVSIM's
``navsim/agents/abstract_agent.py`` (``compute_trajectory``), as in hyzhou404/NAVSIM. The training
code of the agent (losses, optimisers, metric caches, ray, progress bars) is dropped.
"""

import torch

from .._navsim.base import load_checkpoint, model_device, to_batch
from .._navsim.common import AgentInput, Trajectory
from .config import DrivoRConfig
from .features import DrivoRFeatureBuilder
from .model import DrivoRModel


class DrivoRAgent(torch.nn.Module):
    def __init__(self, config: DrivoRConfig, name: str = 'drivor'):
        super().__init__()
        self._config = config
        self._name = name  # the --ad name, for messages: drivor or toad
        self._drivor_model = DrivoRModel(config)
        self._feature_builder = DrivoRFeatureBuilder(config)

    def load_checkpoint(self, checkpoint_path):
        state_dict = load_checkpoint(checkpoint_path, self._name)

        # Older checkpoints stored the submodule as ``agent._pad_model`` before it
        # was renamed to ``_drivor_model``; remap both legacy prefixes.
        def _remap(k: str) -> str:
            k = k.replace('agent._drivor_model', '_drivor_model')
            k = k.replace('agent._pad_model', '_drivor_model')
            return k

        result = self.load_state_dict({_remap(k): v for k, v in state_dict.items()})
        print(f'[{self._name}] loaded {checkpoint_path}: {result}', flush=True)

    def forward(self, features):
        return self._drivor_model(features)

    @torch.no_grad()
    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        self.eval()
        features = to_batch(self._feature_builder.compute_features(agent_input), model_device(self))
        poses = self.forward(features)['trajectory'].squeeze(0).cpu().numpy()
        return Trajectory(poses, self._config.trajectory_sampling)
