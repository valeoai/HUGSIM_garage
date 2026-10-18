"""ZTRS agent: checkpoint loading and inference.

From woxihuanjiangguo/ZTRS (Apache-2.0) ``navsim/agents/ztrs/ztrs_agent.py`` (``initialize`` and
``get_trajectory_from_agent_input``), as in our NAVSIM checkout. The losses, optimisers and callbacks
are dropped. The features are GTRS's.
"""

import torch

from .._navsim.base import load_checkpoint, model_device, to_batch
from .._navsim.common import AgentInput, Trajectory
from .._navsim.hydra import HydraFeatureBuilder
from .config import ZTRSConfig
from .model import HydraRLModel


class ZTRSAgent(torch.nn.Module):
    def __init__(self, config: ZTRSConfig):
        super().__init__()
        self._config = config
        self._trajectory_sampling = config.trajectory_sampling
        self.model = HydraRLModel(config)
        self._feature_builder = HydraFeatureBuilder(config)

    def load_checkpoint(self, checkpoint_path):
        state_dict = load_checkpoint(checkpoint_path, 'ztrs')
        # The vocabulary is read from the .npy file, as in the original, not from the checkpoint.
        for k in [k for k in state_dict if 'model._trajectory_head.vocab' in k]:
            del state_dict[k]
        result = self.load_state_dict({k.replace('agent.', ''): v for k, v in state_dict.items()}, strict=False)
        print(f'[ztrs] loaded {checkpoint_path}: {result}', flush=True)
        return result

    def forward(self, features):
        return self.model(features)

    @torch.no_grad()
    def get_trajectory_from_agent_input(self, agent_input: AgentInput) -> Trajectory:
        self.eval()
        features = to_batch(self._feature_builder.compute_features(agent_input), model_device(self.model))
        poses = self.forward(features)['trajectory'].squeeze(0).cpu().numpy()
        return Trajectory(poses, self._trajectory_sampling)
