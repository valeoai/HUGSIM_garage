"""DrivoR model: camera tokens -> trajectory proposals -> scorer -> CEM refinement.

The DrivoR model is from valeoai/DrivoR (``navsim/agents/drivoR/drivor_model.py`` at fc6e5aa); the CEM
refinement (``cem_refine`` and its helpers) is TOAD's (valeoai/TOAD), both as the copy in
hyzhou404/NAVSIM's working tree ran them for HUGSIM. That copy's CEM differs from TOAD's released
code (``drivor_cem_portable/`` at cfa88e0) in two settings: ``cem_alpha`` 0.7 (1.0 there) and a
control clamp of 5.0 times the kinematic limits (1.5 there). Changes: the LiDAR branch, the training-only
scorer heads and the per-forward prints are dropped, and the ``getattr(cfg, name, default)`` reads
of the CEM settings read the config (which carries those defaults) directly.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DrivoRConfig, ImageBackboneConfig
from .encoder import ImgEncoder
from .mlp import MLP
from .scorer import Scorer
from .transformer_decoder import TransformerDecoder, TransformerDecoderScorer

# Comfort thresholds, mirror navsim/.../pdm_comfort_metrics.py.
_MAX_ABS_MAG_JERK = 8.37  # [m/s^3]
_MAX_ABS_LAT_ACCEL = 4.89  # [m/s^2]
_MAX_LON_ACCEL = 2.40  # [m/s^2]
_MIN_LON_ACCEL = -4.05  # [m/s^2]
_MAX_ABS_YAW_ACCEL = 1.93  # [rad/s^2]
_MAX_ABS_LON_JERK = 4.13  # [m/s^3]
_MAX_ABS_YAW_RATE = 0.95  # [rad/s]


def _wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


class DrivoRModel(nn.Module):
    def __init__(self, config: DrivoRConfig):
        super().__init__()
        self._config = config
        self.poses_num = config.num_poses
        self.state_size = 3
        self.embed_dims = self._config.tf_d_model

        # camera embedding
        self.num_cams = sum(len(cam) > 0 for cam in config.cams.values())

        # create the image backbone
        backbone = config.image_backbone
        self.image_backbone = ImgEncoder(
            ImageBackboneConfig(
                **{
                    **vars(backbone),
                    'image_size': config.image_size,
                    'num_scene_tokens': config.num_scene_tokens,
                    'tf_d_model': config.tf_d_model,
                }
            )
        )
        self.scene_embeds = nn.Parameter(
            torch.randn(1, self.num_cams, self._config.num_scene_tokens, self.image_backbone.num_features) * 1e-6,
            requires_grad=True,
        )

        # ego status encoder
        if self._config.full_history_status:
            self.hist_encoding = nn.Linear(11 * 4, config.tf_d_model)
        else:
            self.hist_encoding = nn.Linear(11, config.tf_d_model)

        # trajectory embdedding
        if self._config.one_token_per_traj:
            self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.poses_num * self.state_size
        else:
            self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.state_size

        # trajectory decoder
        self.trajectory_decoder = TransformerDecoder(proj_drop=0.1, drop_path=0.2, config=config)

        # scorer decoder
        self.scorer_attention = TransformerDecoderScorer(
            num_layers=config.scorer_ref_num, d_model=config.tf_d_model, proj_drop=0.1, drop_path=0.2, config=config
        )

        self.pos_embed = nn.Sequential(
            nn.Linear(self.poses_num * 3, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )

        # get the trajectory decoders
        ref_num = config.ref_num
        self.traj_head = nn.ModuleList(
            [MLP(config.tf_d_model, config.tf_d_ffn, traj_head_output_size) for _ in range(ref_num + 1)]
        )

        # scorer
        self.scorer = Scorer(config)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Raw ego status (B, T_hist, 11), needed by CEM for initial speed and
        # the past-boundary kinematics used by history_comfort.
        ego_status_raw: torch.Tensor = features['ego_status']

        # ego status and initial traj tokens
        if self._config.full_history_status:
            ego_status: torch.Tensor = ego_status_raw.flatten(-2)
        else:
            ego_status: torch.Tensor = ego_status_raw[:, -1]

        ego_token = self.hist_encoding(ego_status)[:, None]
        traj_tokens = ego_token + self.init_feature.weight[None]

        batch_size = ego_status.shape[0]

        # image features
        img = features['image']
        scene_tokens = self.scene_embeds.repeat(batch_size, 1, 1, 1)
        scene_features = self.image_backbone(img, scene_tokens)

        # initial trajectories
        proposals = self.traj_head[0](traj_tokens).reshape(traj_tokens.shape[0], -1, self.poses_num, self.state_size)
        proposal_list = [proposals]

        # decode the trajectories at each step of the decoder
        token_list = self.trajectory_decoder(traj_tokens, scene_features)
        for i in range(self._config.ref_num):
            tokens = token_list[i]
            proposals = self.traj_head[i + 1](tokens).reshape(tokens.shape[0], -1, self.poses_num, self.state_size)
            proposal_list.append(proposals)

        traj_tokens = token_list[-1]
        proposals = proposal_list[-1]

        output = {}
        output['proposals'] = proposals
        output['proposal_list'] = proposal_list

        # scoring
        B, N, _, _ = proposals.shape

        embedded_traj = self.pos_embed(proposals.reshape(B, N, -1).detach())  # (B, N, d_model)
        tr_out = self.scorer_attention(embedded_traj, scene_features)  # (B, N, d_model)
        tr_out = tr_out + ego_token
        pred_logit = self.scorer(tr_out)

        output['pred_logit'] = pred_logit

        pdm_score = (
            self._config.noc * pred_logit['no_at_fault_collisions'].sigmoid().log()
            + self._config.dac * pred_logit['drivable_area_compliance'].sigmoid().log()
            + self._config.ddc * pred_logit['driving_direction_compliance'].sigmoid().log()
            + (
                self._config.ttc * pred_logit['time_to_collision_within_bound'].sigmoid()
                + self._config.ep * pred_logit['ego_progress'].sigmoid()
                + self._config.comfort * pred_logit['comfort'].sigmoid()
            ).log()
        )

        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]
        argmax_score = pdm_score[torch.arange(batch_size), token]

        output['trajectory'] = trajectory
        output['pdm_score'] = pdm_score

        # Cross-Entropy Method refinement at inference time.
        if (not self.training) and self._config.use_cem:
            cem_traj, cem_score = self.cem_refine(
                proposals,
                scene_features,
                ego_token,
                ego_status_raw,
                init_traj=trajectory,
            )
            # Symmetric scoring: rate the argmax pick with the same combined
            # metric (pdm - lambda * comfort) that CEM is optimizing. Without
            # this, CEM is unfairly penalized by comfort and can only "win" on
            # raw pdm, which encourages scorer-exploitation drift.
            cem_dt = self._config.cem_dt
            comfort_w = self._config.cem_comfort_weight
            argmax_comfort = self._comfort_cost_from_traj(trajectory.unsqueeze(1), ego_status_raw, cem_dt).squeeze(1)
            argmax_full_score = argmax_score - comfort_w * argmax_comfort

            use_cem = cem_score > argmax_full_score
            trajectory = torch.where(use_cem[:, None, None], cem_traj, trajectory)
            output['trajectory'] = trajectory
            output['cem_trajectory'] = cem_traj
            output['cem_score'] = cem_score
            output['argmax_score'] = argmax_score
            output['argmax_full_score'] = argmax_full_score
            output['use_cem'] = use_cem

        return output

    def _score_trajectories(
        self,
        trajectories: torch.Tensor,
        scene_features: torch.Tensor,
        ego_token: torch.Tensor,
    ) -> torch.Tensor:
        """Score a batch of trajectories with the same pipeline used in `forward`.

        Args:
            trajectories: (B, N, num_poses, state_size)
            scene_features: (B, S, d_model)
            ego_token: (B, 1, d_model)

        Returns:
            pdm_score: (B, N) — higher is better.
        """
        B, N = trajectories.shape[:2]
        embedded = self.pos_embed(trajectories.reshape(B, N, -1))
        tr_out = self.scorer_attention(embedded, scene_features)
        tr_out = tr_out + ego_token
        pred_logit = self.scorer(tr_out)

        pdm_score = (
            self._config.noc * pred_logit['no_at_fault_collisions'].sigmoid().log() +
            self._config.dac * pred_logit['drivable_area_compliance'].sigmoid().log() +
            self._config.ddc * pred_logit['driving_direction_compliance'].sigmoid().log() +
            (self._config.ttc * pred_logit['time_to_collision_within_bound'].sigmoid() +
             self._config.ep * pred_logit['ego_progress'].sigmoid() +
             self._config.comfort * pred_logit['comfort'].sigmoid()).log()
        )
        return pdm_score

    @torch.no_grad()
    def cem_refine(
        self,
        proposals: torch.Tensor,
        scene_features: torch.Tensor,
        ego_token: torch.Tensor,
        ego_status_raw: torch.Tensor,
        init_traj: torch.Tensor = None,
    ):
        """Cross-Entropy Method in (lon-accel, yaw-rate) control space.

        Trajectories are produced by a kinematic-bicycle rollout, so all
        candidates are smooth by construction. The cost combines:
            * scorer pdm (collisions / drivable / ttc / progress / comfort head)
              evaluated on the rolled-out trajectory,
            * a closed-form comfort cost over (past history + future), which
              captures both `comfort` and `history_comfort`,
            * an anchor cost that pulls toward the argmax-proposal's controls
              to prevent CEM from drifting into out-of-distribution regions
              where the scorer over-estimates pdm.

        The final trajectory returned is the **converged Gaussian mean** —
        not the highest-scoring random sample. The mean is a smooth function
        of the input scene, which keeps consecutive frames consistent and
        protects `two_frame_extended_comfort`.
        """
        cfg = self._config
        dt = cfg.cem_dt
        num_iterations = cfg.cem_num_iterations
        num_samples = cfg.cem_num_samples
        num_elites = max(num_samples // 8, 1)  # sanity check
        alpha = cfg.cem_alpha
        std_floor_a = cfg.cem_std_floor_a  # [m/s^2]
        std_floor_omega = cfg.cem_std_floor_omega  # [rad/s]
        comfort_weight = cfg.cem_comfort_weight
        anchor_weight = cfg.cem_anchor_weight
        init_std_scale = cfg.cem_init_std_scale

        B = proposals.shape[0]
        P = self.poses_num
        device = proposals.device
        dtype = proposals.dtype

        init_speed = ego_status_raw[:, -1, 3]
        past_a, past_omega = self._past_boundary(ego_status_raw, dt)

        # Inverse-kinematics seeding from proposals.
        controls_init = self._proposals_to_controls(proposals, init_speed, dt)  # (B, N_p, P, 2)
        flat_init = controls_init.reshape(B, controls_init.shape[1], P * 2)

        # Anchor: argmax-proposal's controls. CEM is an L2-regularized refinement of this.
        if init_traj is not None:
            anchor_ctrl = self._proposals_to_controls(init_traj.unsqueeze(1), init_speed, dt)
            anchor_flat = anchor_ctrl.reshape(B, P * 2)
        else:
            anchor_flat = flat_init.mean(dim=1)

        mean = anchor_flat.clone()
        std = flat_init.std(dim=1, unbiased=False) * init_std_scale
        floor = torch.tensor(
            [std_floor_a, std_floor_omega], device=device, dtype=dtype
        ).repeat(P)
        std = torch.maximum(std, floor.unsqueeze(0))

        for _ in range(num_iterations):
            # drawn on the CPU and moved, so that a seeded run samples the same noise on any device
            eps = torch.randn(B, num_samples, P * 2, dtype=dtype).to(device)
            sampled_flat = mean.unsqueeze(1) + eps * std.unsqueeze(1)
            controls = self._clamp_controls(sampled_flat.reshape(B, num_samples, P, 2))

            traj, speeds = self._bicycle_rollout(controls, init_speed, dt)
            scorer_score = self._score_trajectories(traj, scene_features, ego_token)
            comfort_cost = self._direct_comfort_cost(
                controls, speeds, past_a, past_omega, dt
            )
            ctrl_flat = controls.reshape(B, num_samples, P * 2)
            anchor_cost = ((ctrl_flat - anchor_flat.unsqueeze(1)) ** 2).mean(-1)
            score = (
                scorer_score
                - comfort_weight * comfort_cost
                - anchor_weight * anchor_cost
            )

            # Refit Gaussian to elites (in clamped-control space).
            _, elite_idx = torch.topk(score, num_elites, dim=1)
            elite = torch.gather(
                ctrl_flat, 1, elite_idx.unsqueeze(-1).expand(-1, -1, P * 2)
            )
            new_mean = elite.mean(dim=1)
            if num_elites > 1:
                new_std = torch.maximum(
                    elite.std(dim=1, unbiased=False), floor.unsqueeze(0)
                )
            else:
                new_std = std
            mean = alpha * new_mean + (1 - alpha) * mean
            std = alpha * new_std + (1 - alpha) * std

        # Final output: converged Gaussian mean — deterministic given inputs.
        ctrl_mean = self._clamp_controls(mean.reshape(B, 1, P, 2))
        traj_mean, sp_mean = self._bicycle_rollout(ctrl_mean, init_speed, dt)
        sc_mean = self._score_trajectories(traj_mean, scene_features, ego_token).squeeze(1)
        cf_mean = self._direct_comfort_cost(
            ctrl_mean, sp_mean, past_a, past_omega, dt
        ).squeeze(1)
        final_score = sc_mean - comfort_weight * cf_mean
        return traj_mean.squeeze(1), final_score

    @torch.no_grad()
    def _comfort_cost_from_traj(
        self,
        trajectory: torch.Tensor,
        ego_status_raw: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Comfort cost for an arbitrary (B, N, P, 3) trajectory. Used to score
        the argmax candidate under the same metric as the CEM candidate."""
        init_speed = ego_status_raw[:, -1, 3]
        past_a, past_omega = self._past_boundary(ego_status_raw, dt)
        controls = self._proposals_to_controls(trajectory, init_speed, dt)
        _, speeds = self._bicycle_rollout(controls, init_speed, dt)
        return self._direct_comfort_cost(controls, speeds, past_a, past_omega, dt)

    @staticmethod
    def _bicycle_rollout(
        controls: torch.Tensor,
        init_speed: torch.Tensor,
        dt: float,
    ):
        """Forward-Euler kinematic-bicycle rollout with mid-point evaluation.

        Args:
            controls: (B, N, P, 2) — per-step (a, omega) in (m/s^2, rad/s).
            init_speed: (B,) — initial longitudinal speed.
            dt: time step in seconds.

        Returns:
            trajectory: (B, N, P, 3) — (x, y, heading) at each future step
                in the local ego frame at t=0.
            speeds_end: (B, N, P) — speed at the end of each step.
        """
        B, N, P, _ = controls.shape
        a = controls[..., 0]
        omega = controls[..., 1]

        # Speed integration (clamp to forward motion).
        v0 = init_speed.view(B, 1).expand(B, N).contiguous()
        delta_v = a * dt
        speeds_end = (v0[..., None] + torch.cumsum(delta_v, dim=-1)).clamp_min(0.0)
        speeds_start = torch.cat([v0[..., None], speeds_end[..., :-1]], dim=-1)
        v_mid = 0.5 * (speeds_start + speeds_end)

        # Heading integration.
        delta_theta = omega * dt
        headings_end = torch.cumsum(delta_theta, dim=-1)
        zero = torch.zeros(B, N, 1, device=controls.device, dtype=controls.dtype)
        headings_start = torch.cat([zero, headings_end[..., :-1]], dim=-1)
        theta_mid = 0.5 * (headings_start + headings_end)

        # Position integration via mid-point speed/heading.
        dx = v_mid * torch.cos(theta_mid) * dt
        dy = v_mid * torch.sin(theta_mid) * dt
        x = torch.cumsum(dx, dim=-1)
        y = torch.cumsum(dy, dim=-1)

        trajectory = torch.stack([x, y, headings_end], dim=-1)
        return trajectory, speeds_end

    @staticmethod
    def _proposals_to_controls(
        proposals: torch.Tensor,
        init_speed: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Inverse kinematics: (B, N, P, 3) trajectories -> (B, N, P, 2) (a, omega).

        Used to seed the CEM Gaussian from the model's proposals.
        """
        B, N, P, _ = proposals.shape
        x = proposals[..., 0]
        y = proposals[..., 1]
        h = proposals[..., 2]

        zero = torch.zeros(B, N, 1, device=proposals.device, dtype=proposals.dtype)
        x_full = torch.cat([zero, x], dim=-1)
        y_full = torch.cat([zero, y], dim=-1)
        h_full = torch.cat([zero, h], dim=-1)

        dx = torch.diff(x_full, dim=-1)
        dy = torch.diff(y_full, dim=-1)
        dh = _wrap_angle(torch.diff(h_full, dim=-1))

        # Per-step speed (proxy via segment length / dt).
        speed = torch.sqrt(dx * dx + dy * dy) / dt
        omega = dh / dt

        # Acceleration via finite-diff of speed (with init_speed as the prior).
        v0 = init_speed.view(B, 1, 1).expand(B, N, 1)
        speed_full = torch.cat([v0, speed], dim=-1)
        a = torch.diff(speed_full, dim=-1) / dt

        return torch.stack([a, omega], dim=-1)

    @staticmethod
    def _past_boundary(ego_status_raw: torch.Tensor, dt: float):
        """Last past (lon-accel, yaw-rate). Used as the t=-dt sample to
        compute boundary jerk and yaw-acceleration (i.e. history comfort)."""
        B = ego_status_raw.shape[0]
        device = ego_status_raw.device
        dtype = ego_status_raw.dtype
        if ego_status_raw.dim() != 3 or ego_status_raw.shape[1] < 2:
            zero = torch.zeros(B, device=device, dtype=dtype)
            return zero, zero
        # ego_status layout: [x, y, heading, vx, vy, ax, ay, cmd0..cmd3]
        last_ax = ego_status_raw[:, -1, 5]
        h_prev = ego_status_raw[:, -2, 2]
        h_curr = ego_status_raw[:, -1, 2]
        last_omega = _wrap_angle(h_curr - h_prev) / dt
        return last_ax, last_omega

    def _clamp_controls(self, controls: torch.Tensor) -> torch.Tensor:
        """Clamp (a, omega) to slightly-relaxed kinematic envelopes. Keeps the
        rollout in a sane regime even when the Gaussian samples extreme values.
        """
        margin = self._config.cem_control_margin  # 5.0 in the HUGSIM client, 1.5 in TOAD's release
        a = controls[..., 0].clamp(_MIN_LON_ACCEL * margin, _MAX_LON_ACCEL * margin)
        omega = controls[..., 1].clamp(
            -_MAX_ABS_YAW_RATE * margin, _MAX_ABS_YAW_RATE * margin
        )
        return torch.stack([a, omega], dim=-1)

    @staticmethod
    def _direct_comfort_cost(
        controls: torch.Tensor,
        speeds: torch.Tensor,
        past_a: torch.Tensor,
        past_omega: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Closed-form comfort penalty (lower = better) over the joint
        (past-boundary + future) sequence.

        Penalizes any threshold violation of: longitudinal acceleration and
        jerk, lateral acceleration, magnitude jerk, yaw rate, and yaw
        acceleration. The boundary jerk / yaw_accel terms — built by
        prepending the last past (a, omega) — are exactly what makes
        `history_comfort` differ from `comfort`.
        """
        B, N, P, _ = controls.shape
        a = controls[..., 0]      # (B, N, P)
        omega = controls[..., 1]  # (B, N, P)

        # Prepend the past boundary so jerk / yaw-accel cover the history join.
        a_prev = past_a.view(B, 1, 1).expand(B, N, 1)
        om_prev = past_omega.view(B, 1, 1).expand(B, N, 1)
        a_full = torch.cat([a_prev, a], dim=-1)        # (B, N, P+1)
        om_full = torch.cat([om_prev, omega], dim=-1)
        lon_jerk = torch.diff(a_full, dim=-1) / dt     # (B, N, P)
        yaw_accel = torch.diff(om_full, dim=-1) / dt

        # Lateral accel ~ v * omega (centripetal in body frame).
        lat_accel = speeds * omega

        # Magnitude jerk: combine lon and lat accel changes.
        zero = torch.zeros(B, N, 1, device=controls.device, dtype=controls.dtype)
        lat_full = torch.cat([zero, lat_accel], dim=-1)
        d_lat = torch.diff(lat_full, dim=-1) / dt
        mag_jerk = torch.sqrt(lon_jerk * lon_jerk + d_lat * d_lat + 1e-8)

        cost = (
            F.relu(a - _MAX_LON_ACCEL).pow(2).sum(-1)
            + F.relu(_MIN_LON_ACCEL - a).pow(2).sum(-1)
            + F.relu(omega.abs() - _MAX_ABS_YAW_RATE).pow(2).sum(-1)
            + F.relu(lon_jerk.abs() - _MAX_ABS_LON_JERK).pow(2).sum(-1)
            + F.relu(yaw_accel.abs() - _MAX_ABS_YAW_ACCEL).pow(2).sum(-1)
            + F.relu(lat_accel.abs() - _MAX_ABS_LAT_ACCEL).pow(2).sum(-1)
            + F.relu(mag_jerk - _MAX_ABS_MAG_JERK).pow(2).sum(-1)
        )
        return cost
