"""GTRS-Aug agent: checkpoint loading and inference with the teacher network.

From NVlabs/GTRS (Apache-2.0) ``navsim/agents/gtrs_aug/gtrs_aug_agent.py`` (``initialize`` and
``get_trajectory_from_agent_input``), as in hyzhou404/NAVSIM. The losses, optimisers, callbacks and
the augmentation metadata loaded when training are dropped.
"""

import torch

from .._navsim.base import load_checkpoint, model_device, to_batch
from .._navsim.common import AgentInput, Trajectory
from .config import GTRSAugConfig
from .features import HydraAugFeatureBuilder
from .model import AugMetaArch, HydraModel


class GTRSAugAgent(torch.nn.Module):
    def __init__(self, config: GTRSAugConfig):
        super().__init__()
        self._config = config
        self._trajectory_sampling = config.trajectory_sampling
        teacher_model = HydraModel(config)
        student_model = HydraModel(config)
        self.model = AugMetaArch(config, teacher_model, student_model)
        self._feature_builder = HydraAugFeatureBuilder(config)

    def load_checkpoint(self, checkpoint_path):
        state_dict = load_checkpoint(checkpoint_path, 'gtrs_aug')
        result = self.load_state_dict({k.replace('agent.', ''): v for k, v in state_dict.items()}, strict=False)
        print(f'[gtrs_aug] loaded {checkpoint_path}: {result}', flush=True)
        return result

    @torch.no_grad()
    def get_trajectory_from_agent_input(self, agent_input: AgentInput) -> Trajectory:
        self.eval()
        device = model_device(self.model)
        features = to_batch(self._feature_builder.compute_features(agent_input), device)
        predictions = self.model({'camera_feature': features['ori_teacher'], 'status_feature': features['status_feature']})
        if self._config.refinement.use_multi_stage and not self._config.lab.use_first_stage_traj_in_infer:
            poses = predictions['final_traj'].squeeze(0).cpu().numpy()
        else:
            poses = predictions['trajectory'].squeeze(0).cpu().numpy()

        return Trajectory(poses, self._trajectory_sampling)
