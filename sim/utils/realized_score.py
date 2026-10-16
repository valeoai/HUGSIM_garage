"""The realized score of finished HUGSIM episodes: TerraTransfer's closed-loop HD-Score.

HUGSIM's native scorer (score_calculator.py, eval.json) judges each step's *planned* trajectory:
no-collision, drivable area, time-to-collision and comfort are all computed on the waypoints the
agent returned, and only route completion looks at where the ego went. TerraTransfer
(arXiv 2606.17386, App. B) drives a single-step jerk policy that has no multi-second plan, so it
scores the *realized* trajectory instead, and that is the protocol to match to be comparable
with it. Per keyframe t (every 0.5 s of the rollout):

* NC_t   ego-vs-vehicle oriented-box overlap at t, plus the 3DGS background (HUGSIM's
         single-frame check, unchanged): {0, 1}
* DAC_t  the ego footprint's four quarters against the scene's ground points (HUGSIM's
         single-frame check, unchanged): {0, 0.5, 1}
* TTC_t  ego and partners propagated at their current velocities on a 0.2 s grid up to 1.0 s;
         the earliest contact tau* (with a partner or the background) gives min(1, tau*/1.0)
* COM_t  the realized trajectory over the next 1.0 s against HUGSIM's comfort bounds
         (a_long in [-4.05, 2.40], |jerk| <= 8.37, |yaw acc| <= 1.93, |yaw rate| <= 0.95): {0, 1}
* R_c    arc-length fraction of the densified recorded camera track reached, min(1, max s / S)

score = mean_t [ NC_t * DAC_t * (5 TTC_t + 2 COM_t) / 7 ] * R_c   (TerraTransfer's "HD-Score")

Three HD-Score variants are reported side by side, because TerraTransfer's own texts differ on
route completion and R_c multiplies the whole score:

  hdscore                      HUGSIM's own score, of the planned trajectory, copied from the
                               episode's eval.json (None without one), not recomputed
  hdscore_realized             this score with HUGSIM's index fraction ``rc``, which the ICLR 2027
                               submission reverts to so that its numbers stay comparable with HUGSIM's
  hdscore_realized_arclength   this score with the arc length above, ``rc_arclength`` (arXiv v1/v2,
                               App. B Eq. 14)

Everything is read from what HUGSIM saves per episode (data.pkl, ground.ply, scene.ply) plus the
scene's ground_param.pkl, so any past run can be rescored: score_realized.py does that for
whole runs, and closed_loop.py --realized_score for the episode it just drove.

Frames are the ones HUGSIM recorded, at whatever simulator step the run used: dt is read from
their timestamps and every seconds-parameterized quantity (the 0.5 s keyframe stride, the 1.0 s
TTC horizon, the 1.0 s comfort window) is converted to a frame count against it. The paper runs
at dt = 0.1 s, so a run at a coarser step is NOT directly comparable on COM: the comfort bounds
are checked on finite differences taken at dt, and jerk, a third difference, shrinks sharply as
dt grows (the same trajectory scores COM 0.33 at dt = 0.1 and 1.00 at dt = 0.5). App. B calls the
metric invariant to dt because the *window* is fixed in seconds; the derivatives inside it are
not. Where the paper leaves a choice open this follows HUGSIM: velocities are finite differences
of the recorded frames, and a partner's velocity is its displacement since the previous frame.

Ported from Pictura's scripts/hugsim/closed_loop_score.py; the scoring is unchanged.
"""

import json
import os
import pickle

import numpy as np
from scipy.spatial import cKDTree

KEYFRAME_S = 0.5
TTC_STRIDE_S, TTC_HORIZON_S = 0.2, 1.0
COM_WINDOW_S = 1.0
BOUNDS = {"lon_min": -4.05, "lon_max": 2.40, "jerk": 8.37, "yaw_acc": 1.93, "yaw_rate": 0.95}
# HUGSIM's canonical ego box: x, y in [-0.5, 0.5], z in [0, 1], scaled by (l, w, h).
EGO_VERTS = np.array(
    [
        [0.5, 0.5, 0],
        [0.5, -0.5, 0],
        [0.5, 0.5, 1],
        [0.5, -0.5, 1],
        [-0.5, -0.5, 0],
        [-0.5, 0.5, 0],
        [-0.5, -0.5, 1],
        [-0.5, 0.5, 1],
    ],
    dtype=np.float64,
)
BG_MIN_POINTS = 100  # HUGSIM's bg_collision_det threshold


def rect(x, y, w, l, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    u = np.array([l / 2, l / 2, -l / 2, -l / 2])
    v = np.array([w / 2, -w / 2, -w / 2, w / 2])
    return np.stack([x + c * u - s * v, y + s * u + c * v], axis=1)


def rects_overlap(a, b):
    """Separating-axis test for two convex quadrilaterals given as (4, 2) corner arrays."""
    for poly in (a, b):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


class PointIndex:
    """A cloud with a KD-tree over its xy, so a box test only looks at the box's neighbourhood.

    Every check here asks whether points fall inside the ego's footprint, and a point inside it
    is within half the footprint's diagonal of its centre. Scanning the whole cloud per check
    cost 0.18 s against 1.8M points, 96 times an episode; the lookup returns the same points.
    """

    def __init__(self, xyz):
        self.xyz = np.asarray(xyz)
        self.tree = cKDTree(self.xyz[:, :2]) if len(self.xyz) else None

    def near(self, x, y, radius):
        if self.tree is None:
            return self.xyz[:0]
        return self.xyz[self.tree.query_ball_point([x, y], radius)]


def bg_collision(index, x, y, z, w, l, h, yaw):
    """HUGSIM's bg_collision_det on the ego box at (x, y, z, yaw): >100 scene points inside."""
    points = index.near(x, y, 0.5 * float(np.hypot(l, w)))
    if points.shape[0] == 0:
        return False
    c, s = np.cos(yaw), np.sin(yaw)
    local = points - np.array([x, y, z])
    lx = c * local[:, 0] + s * local[:, 1]
    ly = -s * local[:, 0] + c * local[:, 1]
    # ego_box's z is the camera plane, about 1.45 m above the road, not the road itself, so
    # a box running up from it covers 1.45-2.95 m: it misses parked car bodies and kerbs and
    # catches overhangs and facade Gaussians. HUGSimEnv's own check runs from the camera
    # plane DOWN to the road, and so does this one now (valeoai/HUGSIM_garage#1).
    inside = (np.abs(lx) < l / 2) & (np.abs(ly) < w / 2) & (local[:, 2] > -h) & (local[:, 2] < 0)
    return int(inside.sum()) > BG_MIN_POINTS


def dac_single(ground, x, y, w, l, yaw):
    """HUGSIM's _single_frame_drivable_area_compliance: 2x2 footprint cells with ground points."""
    ground_xy = ground.near(x, y, 0.5 * float(np.hypot(l, w)))[:, :2]
    if ground_xy.shape[0] == 0:
        return 0.0
    c, s = np.cos(yaw), np.sin(yaw)
    rel = ground_xy - np.array([x, y])
    gx = c * rel[:, 0] + s * rel[:, 1]
    gy = -s * rel[:, 0] + c * rel[:, 1]
    xb = np.linspace(-l / 2, l / 2, 3)
    yb = np.linspace(-w / 2, w / 2, 3)
    cnt = 0
    for xi in range(2):
        for yi in range(2):
            m = (xb[xi] < gx) & (gx < xb[xi + 1]) & (yb[yi] < gy) & (gy < yb[yi + 1])
            cnt += bool(m.any())
    ratio = cnt / 4
    return 0.0 if ratio < 0.3 else (0.5 if ratio < 0.5 else 1.0)


def load_ply_xyz(path):
    import open3d as o3d

    return np.asarray(o3d.io.read_point_cloud(path).points)


def camera_track(scene_dir):
    """The scene's recorded camera positions in the sim frame (as ego_box: x = world z, y = -world x)."""
    cam_poses = np.asarray(pickle.load(open(os.path.join(scene_dir, "ground_param.pkl"), "rb"))[0])
    return np.stack([cam_poses[:, 2, 3], -cam_poses[:, 0, 3]], axis=1)


DATASETS = ("kitti360", "nuscenes", "pandaset", "waymo")


def find_scene_dir(data_root, datasets, scene_name):
    """The reconstructed scene an episode ran in: <data_root>/HUGSIM/ss/scenes/<dataset>/<scene>.

    ``datasets`` is one dataset name or several to search, in order.
    """
    for dataset in (datasets,) if isinstance(datasets, str) else datasets:
        path = os.path.join(data_root, "HUGSIM", "ss", "scenes", dataset, scene_name)
        if os.path.isfile(os.path.join(path, "ground_param.pkl")):
            return path
    raise FileNotFoundError(f"no ground_param.pkl for {scene_name} in {datasets} under {data_root}")


def densify(poly, step=0.1):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    q = np.arange(0.0, s[-1] + 1e-9, step)
    return np.stack([np.interp(q, s, poly[:, 0]), np.interp(q, s, poly[:, 1])], axis=1), q


def kinematics_ok(xy_yaw, dt):
    """HUGSIM's comfort bounds on a short realized trajectory (rows x, y, yaw at spacing dt)."""
    if xy_yaw.shape[0] < 4:
        return 1.0
    x, y, yaw = xy_yaw[:, 0], xy_yaw[:, 1], xy_yaw[:, 2]
    speed = np.hypot(np.diff(x), np.diff(y)) / dt
    acc = np.diff(speed) / dt
    jerk = np.diff(acc) / dt
    dyaw = np.diff(yaw)
    dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
    yaw_rate = dyaw / dt
    yaw_acc = np.diff(yaw_rate) / dt
    ok = (
        np.all(acc >= BOUNDS["lon_min"])
        and np.all(acc <= BOUNDS["lon_max"])
        and np.all(np.abs(jerk) <= BOUNDS["jerk"])
        and np.all(np.abs(yaw_acc) <= BOUNDS["yaw_acc"])
        and np.all(np.abs(yaw_rate) <= BOUNDS["yaw_rate"])
    )
    return 1.0 if ok else 0.0


def _finite(xyz):
    """Drop points that are not finite.

    11 of the 128 waymo scenes carry a handful of nan points in ground.ply (154 of 390k in
    881121006469, 0.04%). cKDTree.query refuses them outright ("'x' must be finite"), which
    failed every keyframe of those episodes -- for every agent, so it read as a slow run rather
    than a broken one. A point that is nowhere is near nothing, so dropping it is what the
    checks would have concluded anyway.
    """
    return xyz[np.isfinite(xyz).all(axis=1)]


def score_episode(ep_dir, scene_dir):
    """The realized score of one finished episode (see the module docstring).

    ``ep_dir`` holds what HUGSIM saved (data.pkl, scene.ply, ground.ply); ``scene_dir`` is the
    reconstructed scene, for its recorded camera track. None when fewer than two frames were driven.
    """
    data = pickle.load(open(os.path.join(ep_dir, "data.pkl"), "rb"))[0]
    frames = data["frames"]
    if len(frames) < 2:
        return None
    ts = np.array([f["time_stamp"] for f in frames], dtype=np.float64)
    dt = float(np.median(np.diff(ts)))
    ego = np.array([f["ego_box"] for f in frames], dtype=np.float64)  # x y z w l h yaw

    # Keep only the scene points near the drive; the checks never look further than ~25 m.
    # The clouds are saved in the world (camera) frame; convert to the sim frame exactly as
    # hugsim_evaluate does before scoring.
    raw = load_ply_xyz(os.path.join(ep_dir, "scene.ply"))
    scene_xyz = _finite(np.stack([raw[:, 2], -raw[:, 0], -raw[:, 1]], axis=1))
    raw = load_ply_xyz(os.path.join(ep_dir, "ground.ply"))
    ground_xyz = _finite(np.stack([raw[:, 2], -raw[:, 0], np.zeros(len(raw))], axis=1))
    tree = cKDTree(ego[:, :2])
    near = tree.query(scene_xyz[:, :2], distance_upper_bound=30.0)[0] < np.inf
    scene_xyz = scene_xyz[near]
    ground_xy = ground_xyz[tree.query(ground_xyz[:, :2], distance_upper_bound=10.0)[0] < np.inf]
    scene_xyz, ground_xy = PointIndex(scene_xyz), PointIndex(ground_xy)

    # Per-frame velocities: ego and partners, finite differences of the recorded frames.
    ego_v = np.zeros((len(frames), 2))
    ego_v[:-1] = np.diff(ego[:, :2], axis=0) / dt
    ego_v[-1] = ego_v[-2]

    step = max(1, int(round(KEYFRAME_S / dt)))
    # App. B: n = max(round(W_COM / dt) + 1, 4) realized poses starting at the keyframe, i.e. 11
    # samples at dt = 0.1 s. The arXiv versions (v1/v2) drop the +1 and say n = 10; the ICLR 2027
    # submission restores it. Following the newer text, which is also what this file always did.
    com_n = max(int(round(COM_WINDOW_S / dt)) + 1, 4)
    per = []
    for i in range(0, len(frames), step):
        x, y, z, w, l, h, yaw = ego[i]
        objs = [o for o, n in zip(frames[i]["obj_boxes"], frames[i]["obj_names"], strict=True) if n == "car"]
        prev = frames[i - 1]["obj_boxes"] if i > 0 else None
        obj_v = [
            (np.array(o[:2]) - np.array(prev[k][:2])) / dt
            if prev is not None and len(prev) == len(frames[i]["obj_boxes"])
            else np.zeros(2)
            for k, o in enumerate(frames[i]["obj_boxes"])
            if frames[i]["obj_names"][k] == "car"
        ]
        ego_rect = rect(x, y, w, l, yaw)
        nc = (
            0.0
            if (
                bg_collision(scene_xyz, x, y, z, w, l, h, yaw)
                or any(rects_overlap(ego_rect, rect(o[0], o[1], o[3], o[4], o[6])) for o in objs)
            )
            else 1.0
        )
        dac = dac_single(ground_xy, x, y, w, l, yaw)

        ttc = 1.0
        for tau in np.arange(0.0, TTC_HORIZON_S + 1e-9, TTC_STRIDE_S):
            ex, ey = np.array([x, y]) + ego_v[i] * tau
            er = rect(ex, ey, w, l, yaw)
            hit = bg_collision(scene_xyz, ex, ey, z, w, l, h, yaw) if tau > 0 else nc == 0.0
            if not hit:
                for o, v in zip(objs, obj_v, strict=True):
                    ox, oy = np.array(o[:2]) + v * tau
                    if rects_overlap(er, rect(ox, oy, o[3], o[4], o[6])):
                        hit = True
                        break
            if hit:
                ttc = min(1.0, tau / TTC_HORIZON_S)
                break

        window = ego[i : i + com_n][:, [0, 1, 6]]
        com = kinematics_ok(window, dt)
        per.append(
            {"t": float(ts[i]), "nc": nc, "dac": dac, "ttc": ttc, "c": com, "pdms": nc * dac * (5 * ttc + 2 * com) / 7}
        )

    # HUGSIM records each frame's pose *before* the step and the collision flag *after* it, so
    # the pose at contact is never saved. An episode that ended in a collision gets a terminal
    # keyframe at the contact time with NC = 0, which is what a per-keyframe check sees there.
    if frames[-1].get("collision"):
        per.append(
            {"t": float(ts[-1] + dt), "nc": 0.0, "dac": per[-1]["dac"], "ttc": 0.0, "c": per[-1]["c"], "pdms": 0.0}
        )

    track, s_track = densify(camera_track(scene_dir))
    s_ego = s_track[cKDTree(track).query(ego[:, :2])[1]]
    rc_arclength = float(min(1.0, s_ego.max() / max(s_track[-1], 1e-9)))
    # Two route-completion conventions, because the paper's own differ and R_c multiplies the
    # whole score (native runs ~1.14x arc-length here, so the choice is not cosmetic):
    #   rc            HUGSIM's index fraction min(1, max_t (k_t + 1) / (0.9 N)), which the ICLR 2027
    #                 submission reverts to, "so that every method ... remains comparable with the
    #                 numbers HUGSim publishes". Read straight off the frames HUGSIM recorded.
    #   rc_arclength  arc-length fraction of the densified camera track (arXiv v1/v2, App. B Eq. 14)
    native = [f["rc"] for f in frames if f.get("rc") is not None]
    rc = float(min(1.0, max(native))) if native else rc_arclength
    mean = {k: float(np.mean([p[k] for p in per])) for k in ("nc", "dac", "ttc", "c", "pdms")}
    # `hdscore` is HUGSIM's own score, of the planned trajectory, copied from eval.json (None when
    # the episode has none: --skip-native-eval, or HUGSIM's scorer died), so the three variants sit
    # side by side; this scorer's own is `hdscore_realized`.
    eval_path = os.path.join(ep_dir, "eval.json")
    hdscore = json.load(open(eval_path)).get("hdscore") if os.path.isfile(eval_path) else None
    return {
        "hdscore": hdscore,
        **mean,
        "rc": rc,
        "hdscore_realized": mean["pdms"] * rc,
        "rc_arclength": rc_arclength,
        "hdscore_realized_arclength": mean["pdms"] * rc_arclength,
        "keyframes": len(per),
    }


#: What an AD side leaves in the episode folder when it failed and ended the episode: the garage's
#: shared client (agents/client.py) and Pictura's driver. HUGSIM still writes data.pkl and
#: eval.json for such an episode, so from those alone a crash three seconds in reads as a short
#: drive with low route completion; it is recorded as a crash instead of scored.
CRASH_MARKERS = ("agent_crash.txt", "pictura_crash.json")


def read_crash_marker(ep_dir):
    """``{"marker", "step", "error"}`` when the AD side crashed in this episode, else None."""
    for name in CRASH_MARKERS:
        path = os.path.join(ep_dir, name)
        if not os.path.isfile(path):
            continue
        if name.endswith(".json"):
            with open(path) as f:
                record = json.load(f)
            return {"marker": name, "step": record.get("step"), "error": record.get("error")}
        with open(path) as f:
            lines = [line for line in f.read().splitlines() if line.strip()]
        return {"marker": name, "step": None, "error": lines[-1] if lines else ""}
    return None
