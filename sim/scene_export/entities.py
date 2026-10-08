"""Pack HUGSIM state into the ego-frame entity blocks an external rasterizer consumes.

Two blocks matter here, and both are plain float arrays:

* roads -- ``(N, 7)`` ``[x0, y0, z0, x1, y1, z1, type_code]``, one 3D segment per row.
* agent cuboids -- ``(K, 16)`` ``[center xyz, forward xyz, right xyz, up xyz,
  half_length, half_width, half_height, class_code]``.

Both are expressed in the ego frame (x forward, y left, z relative to the ground under the
ego) and are shared across the rig's cameras -- the renderer applies each camera's own
extrinsics itself. The transforms below mirror ``collect_shared_entities`` in the consumer's
``observation.h``: same visibility box, same z gate, same nearest-first ordering, so a frame
built from HUGSIM state lands in the distribution the policy was trained on.
"""

import numpy as np

# The agent class codes: the renderer picks a face palette from them.
CLASS_VEHICLE = 1
CLASS_PEDESTRIAN = 2
CLASS_CYCLIST = 3

# The visibility box, in meters (drive.ini road_obs_front/behind/side_dist). Geometry outside
# it is not exported at training time, so exporting it here would be off-distribution.
ROAD_OBS_FRONT_M = 120.0
ROAD_OBS_BEHIND_M = 20.0
ROAD_OBS_SIDE_M = 30.0

# Segments whose midpoint sits further than this above or below the ego are dropped: on a
# multi-level map they belong to another deck (drive.ini same_level_z_thresh).
SAME_LEVEL_Z_THRESH_M = 8.0

# Per-camera block capacities. The road figure is POLY_MAX_ROAD_ENTITIES_PER_CAM. The agent
# figure is only an upper bound on what is exported: the policy's agent block is sized by the
# checkpoint's own `max_partner_observations` (16 for every shipped checkpoint), and the
# driver keeps the nearest that many, which is why the export is sorted nearest-first. HUGSIM
# cannot import pufferlib to look the checkpoint's figure up, so exporting a generous 64 lets
# one export serve any checkpoint; it is overridable from the scenario config. Exporting fewer
# than the checkpoint's block leaves slots empty that the ego can still collide with. The
# rasterizer's own ceiling is 85 (its per-camera face budget).
MAX_ROAD_ENTITIES = 512
MAX_AGENT_ENTITIES = 64


def _ego_transform(ego_box):
    """Return ``(xy, cos_heading, sin_heading, z)`` for a HUGSIM ego box."""
    x, y, z = float(ego_box[0]), float(ego_box[1]), float(ego_box[2])
    yaw = float(ego_box[6])
    return np.array([x, y]), np.cos(yaw), np.sin(yaw), z


def _world_to_ego_xy(points_xy, ego_xy, cos_h, sin_h):
    """``(..., 2)`` sim-frame points -> ego frame (x forward, y left)."""
    rel = points_xy - ego_xy
    return np.stack(
        [rel[..., 0] * cos_h + rel[..., 1] * sin_h, -rel[..., 0] * sin_h + rel[..., 1] * cos_h],
        axis=-1,
    )


# Rigs whose estimated mount pitch is below this are treated as level. On nuScenes, pandaset and
# waymo the per-scene estimate scatters within +/-1 deg around zero (noise from bumps and real
# grade changes); KITTI-360's front camera reads +5.8..+6.0 deg on every scene.
RIG_PITCH_DEADBAND_DEG = 2.0


def rig_pitch_from_track(cam_poses, min_step_m=0.05, deadband_deg=RIG_PITCH_DEADBAND_DEG):
    """How far the direction of travel sits above the recorded camera's optical axis, in radians.

    HUGSIM's world frame is the frame of the recorded camera poses, and on KITTI-360 that frame
    is the camera's own: the camera is mounted pitched down, so a level street climbs ~6 deg in
    world coordinates while the camera's optical axis stays horizontal in them. Exported as is,
    a flat street reaches the policy as a 10% hill (the goals 10 m up at 115 m). The travel
    direction is the reliable reference -- the vehicle drives along the road -- so the median
    angle between it and the optical axis is the rig's mount pitch, and rotating the ego-frame
    export by it puts the road back where the vehicle rides on it.

    Args:
        cam_poses: ``(N, 4, 4)`` camera-to-world poses in HUGSIM's world frame (y down, z
            forward), as ``ground_param.pkl`` stores them.
        min_step_m: pose steps shorter than this carry no direction and are skipped.
        deadband_deg: estimates below this magnitude return 0.

    Returns:
        The pitch in radians, positive when travel points above the optical axis.
    """
    cam_poses = np.asarray(cam_poses, dtype=np.float64)
    step = np.diff(cam_poses[:, :3, 3], axis=0)
    horiz = np.linalg.norm(step[:, [0, 2]], axis=1)
    keep = horiz > min_step_m
    if keep.sum() < 5:
        return 0.0
    travel = np.arctan2(-step[keep, 1], horiz[keep])
    optical = np.arcsin(np.clip(-cam_poses[:-1][keep, 1, 2], -1.0, 1.0))
    pitch = float(np.median(travel - optical))
    return pitch if abs(np.degrees(pitch)) >= deadband_deg else 0.0


def _level(x, z, pitch):
    """Rotate ego-frame ``(forward, up)`` by ``-pitch`` about the lateral axis.

    ``pitch`` is the rig's mount pitch (see :func:`rig_pitch_from_track`): the world frame is
    tilted by it relative to the road, and this undoes that in the ego frame.
    """
    if pitch == 0.0:
        return x, z
    c, s = np.cos(pitch), np.sin(pitch)
    return c * x + s * z, -s * x + c * z


def _clip_segments_to_box(p0, p1, x_min, x_max, y_min, y_max):
    """Liang-Barsky clip of a batch of 2D segments against an axis-aligned box.

    Args:
        p0, p1: ``(N, 2)`` segment endpoints.
        x_min, x_max, y_min, y_max: box bounds.

    Returns:
        ``(t_enter, t_exit, keep)``. ``keep`` is the mask of segments with a non-empty
        intersection; the two parameters are the clipped span along ``p1 - p0``, so a caller
        can interpolate z the same way it interpolates x and y.
    """
    d = p1 - p0
    t_enter = np.zeros(p0.shape[0])
    t_exit = np.ones(p0.shape[0])
    keep = np.ones(p0.shape[0], dtype=bool)

    for axis, lo, hi in ((0, x_min, x_max), (1, y_min, y_max)):
        di = d[:, axis]
        pi = p0[:, axis]
        parallel = np.abs(di) < 1e-12
        # Parallel to this pair of planes: the segment is either wholly inside the slab or
        # wholly outside it, and no t bound can be derived either way.
        keep &= ~(parallel & ((pi < lo) | (pi > hi)))

        with np.errstate(divide="ignore", invalid="ignore"):
            t_lo = np.where(parallel, -np.inf, (lo - pi) / di)
            t_hi = np.where(parallel, np.inf, (hi - pi) / di)
        t_near = np.minimum(t_lo, t_hi)
        t_far = np.maximum(t_lo, t_hi)
        t_enter = np.maximum(t_enter, np.where(parallel, 0.0, t_near))
        t_exit = np.minimum(t_exit, np.where(parallel, 1.0, t_far))

    keep &= t_enter <= t_exit
    return t_enter, t_exit, keep


def roads_to_ego(
    segments,
    ego_box,
    front_m=ROAD_OBS_FRONT_M,
    behind_m=ROAD_OBS_BEHIND_M,
    side_m=ROAD_OBS_SIDE_M,
    z_thresh_m=SAME_LEVEL_Z_THRESH_M,
    max_entities=MAX_ROAD_ENTITIES,
    pitch=0.0,
):
    """Cull a scene's road segments to what the ego can see, in the ego frame.

    Args:
        segments: ``(N, 7)`` sim-frame segments from a :mod:`~sim.scene_export.map_source`.
        ego_box: HUGSIM's ``[x, y, z, w, l, h, yaw]``.
        front_m, behind_m, side_m: the visibility box.
        z_thresh_m: drop segments whose midpoint is further than this off the ego's level.
        max_entities: block capacity; the nearest are kept.
        pitch: the rig's mount pitch, undone before the culls (:func:`rig_pitch_from_track`).

    Returns:
        ``(M, 7)`` float32, ``M <= max_entities``, nearest first.
    """
    segments = np.asarray(segments, dtype=np.float64)
    if segments.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float32)

    ego_xy, cos_h, sin_h, ego_z = _ego_transform(ego_box)
    p0 = _world_to_ego_xy(segments[:, 0:2], ego_xy, cos_h, sin_h)
    p1 = _world_to_ego_xy(segments[:, 3:5], ego_xy, cos_h, sin_h)
    z0 = segments[:, 2] - ego_z
    z1 = segments[:, 5] - ego_z
    p0[:, 0], z0 = _level(p0[:, 0], z0, pitch)
    p1[:, 0], z1 = _level(p1[:, 0], z1, pitch)

    on_level = np.abs(0.5 * (z0 + z1)) <= z_thresh_m
    t_enter, t_exit, inside = _clip_segments_to_box(p0, p1, -behind_m, front_m, -side_m, side_m)
    keep = on_level & inside
    if not np.any(keep):
        return np.zeros((0, 7), dtype=np.float32)

    p0, p1 = p0[keep], p1[keep]
    z0, z1 = z0[keep], z1[keep]
    t_enter, t_exit = t_enter[keep][:, None], t_exit[keep][:, None]
    d_xy = p1 - p0
    d_z = (z1 - z0)[:, None]

    clipped = np.empty((p0.shape[0], 7), dtype=np.float64)
    clipped[:, 0:2] = p0 + d_xy * t_enter
    clipped[:, 2] = (z0[:, None] + d_z * t_enter)[:, 0]
    clipped[:, 3:5] = p0 + d_xy * t_exit
    clipped[:, 5] = (z0[:, None] + d_z * t_exit)[:, 0]
    clipped[:, 6] = segments[keep, 6]

    if clipped.shape[0] > max_entities:
        # Over budget the renderer would truncate in whatever order the map happened to be
        # built in, which drops geometry under the ego's nose in favour of geometry 100 m
        # out. Order by midpoint range and let the far end be the part that goes.
        mid = 0.5 * (clipped[:, 0:2] + clipped[:, 3:5])
        nearest = np.argsort(np.einsum("ij,ij->i", mid, mid))[:max_entities]
        clipped = clipped[nearest]

    return clipped.astype(np.float32)


def _boxes_to_ego_pose(boxes, ego_box, pitch=0.0):
    """Ego-frame pose of an ``(M, 7)`` array of ``[x, y, z, w, l, h, yaw]`` boxes.

    The one transform behind both the cuboid export and the pose history, so the two cannot
    drift apart: sim-frame centre -> ego frame, centre z measured off the ego's ground datum,
    then the rig-pitch correction of x (which uses the centre z).

    Returns:
        ``(center_xy, center_z, half_h, rel_yaw)``: ``(M, 2)`` leveled ego-frame centre,
        ``(M,)`` centre z, ``(M,)`` half height, and ``(M,)`` ``box_yaw - ego_yaw`` (unwrapped).
    """
    ego_xy, cos_h, sin_h, ego_z = _ego_transform(ego_box)
    center_xy = _world_to_ego_xy(boxes[:, 0:2], ego_xy, cos_h, sin_h)
    half_h = 0.5 * np.maximum(boxes[:, 5], 0.1)
    center_xy[:, 0], center_z = _level(center_xy[:, 0], boxes[:, 2] + half_h - ego_z, pitch)
    return center_xy, center_z, half_h, boxes[:, 6] - float(ego_box[6])


def box_history_to_ego(history_boxes, ego_box, pitch=0.0):
    """Express a box history in the current ego frame.

    Args:
        history_boxes: ``(M, N, 7)`` ``[x, y, z, w, l, h, yaw]``, sim frame, one row per
            actor and one column per snapshot.
        ego_box: the CURRENT ego box (the frame every snapshot is expressed in).
        pitch: the rig's mount pitch, as for :func:`boxes_to_cuboids`.

    Returns:
        ``(M, N, 3)`` float32 ``(x, y, heading)``; ``(x, y)`` match the cuboid centre and
        ``heading = box_yaw - ego_yaw`` wrapped to ``(-pi, pi]``, so it agrees with
        ``atan2(cuboid[4], cuboid[3])``.
    """
    history_boxes = np.asarray(history_boxes, dtype=np.float64)
    m, n = history_boxes.shape[:2]
    center_xy, _, _, rel_yaw = _boxes_to_ego_pose(history_boxes.reshape(m * n, 7), ego_box, pitch)
    # arctan2(sin, cos) wraps to (-pi, pi] and is what the cuboid's (cos, sin) decode to.
    heading = np.arctan2(np.sin(rel_yaw), np.cos(rel_yaw))
    out = np.concatenate([center_xy, heading[:, None]], axis=1)
    return out.reshape(m, n, 3).astype(np.float32)


def _empty_extra(extra):
    """Zero-row stand-in for ``extra`` (1-D stays ``(0,)``; a 2-D block keeps its width)."""
    width = np.shape(extra)[1:] if np.ndim(extra) > 1 else ()
    return np.zeros((0,) + tuple(width), dtype=np.float32)


def boxes_to_cuboids(
    obj_boxes,
    ego_box,
    front_m=ROAD_OBS_FRONT_M,
    behind_m=ROAD_OBS_BEHIND_M,
    side_m=ROAD_OBS_SIDE_M,
    max_entities=MAX_AGENT_ENTITIES,
    class_code=CLASS_VEHICLE,
    pitch=0.0,
    extra=None,
):
    """Turn HUGSIM's dynamic boxes into ego-frame agent cuboids.

    Args:
        obj_boxes: sequence of ``[x, y, z, w, l, h, yaw]``, as ``info['obj_boxes']`` gives
            them. ``z`` is ground level under the actor, matching the export's convention.
        ego_box: HUGSIM's ego box.
        front_m, behind_m, side_m: the visibility box, shared with the road cull.
        max_entities: block capacity; the nearest are kept.
        class_code: face palette selector. HUGSIM's actors are all 3DRealCar vehicles, so
            this is ``CLASS_VEHICLE`` unless a scenario grows other actor types.
        pitch: the rig's mount pitch, undone before the cull (:func:`rig_pitch_from_track`).
        extra: optional per-box values (e.g. speeds, or a 2-D ``(M, C)`` block of per-box
            rows) carried through the same cull and sort; indexed on the first axis.

    Returns:
        ``(K, 16)`` float32, ``K <= max_entities``, nearest first; with ``extra``, a pair
        ``(cuboids, extra)`` whose rows line up.
    """
    empty = np.zeros((0, 16), dtype=np.float32)
    if obj_boxes is None or len(obj_boxes) == 0:
        return empty if extra is None else (empty, _empty_extra(extra))

    boxes = np.asarray(obj_boxes, dtype=np.float64)
    center_xy, center_z, half_h, all_rel_yaw = _boxes_to_ego_pose(boxes, ego_box, pitch)
    width, length = boxes[:, 3], boxes[:, 4]

    visible = (
        (center_xy[:, 0] >= -behind_m)
        & (center_xy[:, 0] <= front_m)
        & (np.abs(center_xy[:, 1]) <= side_m)
    )
    if not np.any(visible):
        return empty if extra is None else (empty, _empty_extra(extra))

    center_xy = center_xy[visible]
    rel_yaw = all_rel_yaw[visible]
    rel_cos, rel_sin = np.cos(rel_yaw), np.sin(rel_yaw)

    cuboids = np.zeros((center_xy.shape[0], 16), dtype=np.float64)
    cuboids[:, 0:2] = center_xy
    cuboids[:, 2] = center_z[visible]
    cuboids[:, 3] = rel_cos
    cuboids[:, 4] = rel_sin
    cuboids[:, 6] = -rel_sin
    cuboids[:, 7] = rel_cos
    cuboids[:, 11] = 1.0
    cuboids[:, 12] = 0.5 * np.maximum(length[visible], 0.1)
    cuboids[:, 13] = 0.5 * np.maximum(width[visible], 0.1)
    cuboids[:, 14] = half_h[visible]
    cuboids[:, 15] = class_code

    # Always nearest-first, not only on overflow: consumers with fewer slots than
    # max_entities truncate this block, and must lose the far boxes rather than arbitrary ones.
    nearest = np.argsort(np.einsum("ij,ij->i", center_xy, center_xy), kind="stable")[:max_entities]
    cuboids = cuboids[nearest]

    if extra is None:
        return cuboids.astype(np.float32)
    carried = np.asarray(extra, dtype=np.float64)[visible][nearest]
    return cuboids.astype(np.float32), carried.astype(np.float32)


# The static-target block the policy reads: `num_target_waypoints` goals, normalized by
# `max_goal_position` (drive.ini). a consumer's own goals may be sampled destinations; HUGSIM's
# equivalent is the recorded drive, which is also what route completion is scored against.
NUM_TARGET_WAYPOINTS = 3
GOAL_HORIZON_M = 120.0


def route_to_ego(track_xyz, ego_box, num_waypoints=NUM_TARGET_WAYPOINTS, horizon_m=GOAL_HORIZON_M, pitch=0.0):
    """Sample the recorded drive ahead of the ego as goal waypoints, in the ego frame.

    Args:
        track_xyz: ``(N, 3)`` sim-frame ground track (the densified camera poses).
        ego_box: HUGSIM's ego box.
        num_waypoints: how many goals to emit; must match the policy's target block width.
        horizon_m: arc length ahead of the ego covered by the last waypoint.
        pitch: the rig's mount pitch (:func:`rig_pitch_from_track`).

    Returns:
        ``(num_waypoints, 3)`` float32 ego-frame goals. Past the end of the track every
        remaining waypoint repeats its last point, which is what an ego approaching its
        destination should see.
    """
    track_xyz = np.asarray(track_xyz, dtype=np.float64)
    ego_xy, cos_h, sin_h, ego_z = _ego_transform(ego_box)
    if track_xyz.shape[0] < 2:
        return np.zeros((num_waypoints, 3), dtype=np.float32)

    # Start from the track point nearest the ego, not from the track's own start: the ego is
    # somewhere in the middle of the drive and the goals are what lies ahead of it.
    delta = track_xyz[:, :2] - ego_xy
    start = int(np.argmin(np.einsum("ij,ij->i", delta, delta)))

    ahead = track_xyz[start:]
    if ahead.shape[0] < 2:
        ahead = track_xyz[-2:]
    seg_len = np.linalg.norm(np.diff(ahead[:, :2], axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(seg_len)])

    sample_at = np.linspace(horizon_m / num_waypoints, horizon_m, num_waypoints)
    goals = np.stack([np.interp(sample_at, along, ahead[:, axis]) for axis in range(3)], axis=1)

    out = np.empty((num_waypoints, 3), dtype=np.float64)
    out[:, :2] = _world_to_ego_xy(goals[:, :2], ego_xy, cos_h, sin_h)
    out[:, 2] = goals[:, 2] - ego_z
    out[:, 0], out[:, 2] = _level(out[:, 0], out[:, 2], pitch)
    return out.astype(np.float32)
