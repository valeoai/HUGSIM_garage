"""The part of the scene's point clouds that HUGSIM's two scorers can ever look at.

Both scorers only ask point questions about a box at a known pose, and a point inside such a box is
within half its footprint's diagonal of the pose. So the points further than that from every pose
asked about cannot change any score, and an episode can keep only the clouds' corridor around its
drive instead of the whole scene (~40 MB of ground.ply + scene.ply per episode, ~3 MB cropped).

The poses asked about (all in the sim frame, x = world z, y = -world x):

* native no-collision and drivable area (score_calculator.py): each frame's planned poses,
  ``[ego pose] + plan waypoints``;
* native time-to-collision: those poses displaced by velocity * {0.5, 1.0} s, velocity being
  diff(planned) / timestep with its first row repeated;
* realized (realized_score.py): the recorded ego poses, and each displaced by the ego's finite
  difference velocity * tau for tau on the 0.2 s grid up to 1.0 s.

A frame whose plan has fewer than two waypoints is skipped by the native scorer, and the realized
terminal collision keyframe asks no new question, so neither adds a pose. Every frame is used
here, not only the key frames the scorers visit: a superset can only keep more.
"""

import numpy as np
from scipy.spatial import cKDTree

#: Added to the half diagonal, in metres: slack for float32 versus float64 and the like.
MARGIN_M = 0.5
#: The native TTC displacements and the realized TTC grid (realized_score.TTC_STRIDE_S / TTC_HORIZON_S).
NATIVE_TTC_S = (0.5, 1.0)
REALIZED_TAU_S = np.arange(0.0, 1.0 + 1e-9, 0.2)


def scoring_centers(frames):
    """The (n, 2) sim-frame positions both scorers test a box at, for these data.pkl frames."""
    centers = []
    for f in frames:
        x, y, _, _, _, _, yaw = f["ego_box"]
        traj = np.asarray(f["planned_traj"]["traj"], dtype=np.float64)
        if len(traj) < 2:
            continue
        planned = np.concatenate(([np.array([x, y, yaw], dtype=np.float64)], traj), axis=0)[:, :2]
        vel = np.diff(planned, axis=0) / f["planned_traj"]["timestep"]
        vel = np.vstack([vel[0], vel])
        for t in (0.0, *NATIVE_TTC_S):
            centers.append(planned + vel * t)
    if len(frames) >= 2:
        ego = np.array([f["ego_box"] for f in frames], dtype=np.float64)
        ts = np.array([f["time_stamp"] for f in frames], dtype=np.float64)
        dt = float(np.median(np.diff(ts)))
        ego_v = np.zeros((len(frames), 2))
        ego_v[:-1] = np.diff(ego[:, :2], axis=0) / dt
        ego_v[-1] = ego_v[-2]
        for tau in REALIZED_TAU_S:
            centers.append(ego[:, :2] + ego_v * tau)
    return np.concatenate(centers) if centers else np.zeros((0, 2))


def corridor_radius(frames):
    """Half the largest ego footprint diagonal, plus the margin."""
    ego = np.array([f["ego_box"] for f in frames], dtype=np.float64)
    return 0.5 * float(np.hypot(ego[:, 3].max(), ego[:, 4].max())) + MARGIN_M


def crop_to_corridor(xyz_world, centers, radius):
    """The rows of a world-frame cloud (n, 3) within ``radius`` of any sim-frame centre.

    Points that are not finite are dropped as well: no check can find one inside a box.
    """
    if len(centers) == 0:
        return xyz_world[:0]
    sim_xy = np.stack([xyz_world[:, 2], -xyz_world[:, 0]], axis=1)
    finite = np.isfinite(sim_xy).all(axis=1)
    dist = cKDTree(centers).query(np.where(finite[:, None], sim_xy, 0.0), distance_upper_bound=radius)[0]
    return xyz_world[finite & (dist < np.inf)]


def crop_episode(frames, ground_world, scene_world):
    """Both clouds of an episode cropped to its corridor: (ground, scene)."""
    if not frames:
        return ground_world, scene_world
    centers, radius = scoring_centers(frames), corridor_radius(frames)
    return crop_to_corridor(ground_world, centers, radius), crop_to_corridor(scene_world, centers, radius)
