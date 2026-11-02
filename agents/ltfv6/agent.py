"""The LTFv6 agent: LEAD's NAVSIM inputs, its model, and its output back in NAVSIM's frame.

What LEAD's NAVSIM v1.1 workspace runs (kesai-labs/lead, branch ``cvpr2026``, commit 730bc1a2f4,
Apache-2.0), with the nuplan devkit, Lightning and the ensemble wrapper removed:

- the inputs of ``navsim/agents/transfuser/transfuser_features.py`` (``TransfuserFeatureBuilder``): the
  four cameras L0, F0, R0, B0 side by side, downscaled 4x (cv2's default bilinear) to 1920x270 and
  passed through JPEG at quality 30, as its feature cache stores them; the status is the driving
  command, then the norms of the velocity and of the acceleration;
- ``navsim/agents/carla_transfuser_agent.py``: the model runs as ``lead/inference/open_loop_inference.py``
  runs it, the released ``model_0060.pth`` (an ensemble of one), under autocast in the
  config's float type (bfloat16: the config was saved on an L40S); the model is trained in CARLA's
  left-handed frame, so y and the heading are negated into NAVSIM's (x forward, y left).
"""

import json

import cv2
import numpy as np
import torch

from .._navsim.base import model_device
from .._navsim.common import AgentInput, Trajectory, TrajectorySampling
from .model import Model, TrainingConfig

JPEG_QUALITY = 30


def camera_feature(cameras):
    """``TransfuserFeatureBuilder._get_camera_feature`` followed by the agent's decode: a (3, 270, 1920) uint8 array."""
    stitched = np.concatenate([cameras.cam_l0.image, cameras.cam_f0.image, cameras.cam_r0.image, cameras.cam_b0.image], axis=1)
    resized = cv2.resize(stitched, (stitched.shape[1] // 4, stitched.shape[0] // 4))
    _, jpeg = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    rgb = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)  # same channel order as encoded
    return np.transpose(rgb, (2, 0, 1))


class LTFv6Agent(torch.nn.Module):
    def __init__(self, config_path, name='ltfv6'):
        super().__init__()
        with open(config_path, encoding='utf-8') as f:
            self._config = TrainingConfig(json.load(f))
        self._name = name
        # Built on the CPU; the module's ``device`` attributes follow it at each step (see _sync_device).
        self.net = Model(device=torch.device('cpu'), config=self._config)
        # NAVSIM's 4 s at 2 Hz: the config's NAVSIM branch (``num_way_points_prediction``, ``waypoints_spacing``).
        assert self._config.num_way_points_prediction == 8, self._config.num_way_points_prediction
        self.sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    def load_checkpoint(self, path):
        """Every weight of the planner, and nothing else silently: the checkpoint also holds LEAD's
        training-only heads (BEV semantics, CenterNet boxes, ...), which the release's model does not
        build and loads past with ``strict=False``; here a missing key, or an extra one inside the
        backbone or the planning decoder, is an error."""
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        result = self.net.load_state_dict(state_dict, strict=False)
        inside = [k for k in result.unexpected_keys if k.startswith(('backbone.', 'planning_decoder.'))]
        if result.missing_keys or inside:
            raise RuntimeError(f'{self._name}: {path} does not fit the model: missing {result.missing_keys}, unexpected {inside}')
        heads = sorted({k.split('.')[0] for k in result.unexpected_keys})
        print(f'[{self._name}] loaded {path}: all {len(self.net.state_dict())} tensors; unused training heads {heads}', flush=True)

    def _sync_device(self, device):
        """The released modules keep the device they were built for and move their inputs to it."""
        for m in self.net.modules():
            if hasattr(m, 'device') and isinstance(m.device, torch.device):
                m.device = device

    def features(self, agent_input: AgentInput):
        status = agent_input.ego_statuses[-1]
        return {
            'rgb': torch.from_numpy(camera_feature(agent_input.cameras[-1])).unsqueeze(0).float(),
            'command': torch.as_tensor(np.asarray(status.driving_command), dtype=torch.float32).reshape(1, 4),
            'speed': torch.tensor([[float(np.linalg.norm(status.ego_velocity))]], dtype=torch.float32),
            'acceleration': torch.tensor([[float(np.linalg.norm(status.ego_acceleration))]], dtype=torch.float32),
        }

    def forward(self, data):
        return self.net(data)

    @torch.no_grad()
    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        self.eval()
        device = model_device(self)
        self._sync_device(device)
        data = {k: v.to(device) for k, v in self.features(agent_input).items()}
        with torch.autocast(
            device_type=device.type,
            dtype=self._config.torch_float_type,
            enabled=self._config.use_mixed_precision_training,
        ):
            prediction = self.forward(data)
        waypoints = prediction.pred_future_waypoints[0].float().cpu().numpy()
        headings = prediction.pred_headings[0].float().cpu().numpy()
        poses = np.stack([waypoints[:, 0], -waypoints[:, 1], -headings], axis=-1)  # CARLA (y right) -> NAVSIM (y left)
        return Trajectory(poses, self.sampling)
