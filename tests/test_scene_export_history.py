"""Pose-history export: ``box_history_to_ego``, ``PoseHistory`` and the cull alignment.

Numpy only. ``sim.scene_export`` imports cleanly without torch/open3d, so it is imported as
a package.
"""

import numpy as np
import pytest

from sim.scene_export import PoseHistory, box_history_to_ego, boxes_to_cuboids
from sim.scene_export import entities as ent


def _old_boxes_to_cuboids(obj_boxes, ego_box, front_m=120.0, behind_m=20.0, side_m=30.0, max_entities=64,
                          class_code=1, pitch=0.0, extra=None):
    """Verbatim copy of the pre-refactor implementation."""
    empty = np.zeros((0, 16), dtype=np.float32)
    if obj_boxes is None or len(obj_boxes) == 0:
        return empty if extra is None else (empty, np.zeros(0, dtype=np.float32))
    boxes = np.asarray(obj_boxes, dtype=np.float64)
    ego_xy, cos_h, sin_h, ego_z = ent._ego_transform(ego_box)
    ego_yaw = float(ego_box[6])
    center_xy = ent._world_to_ego_xy(boxes[:, 0:2], ego_xy, cos_h, sin_h)
    width, length, height = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    half_h = 0.5 * np.maximum(height, 0.1)
    center_xy[:, 0], center_z = ent._level(center_xy[:, 0], boxes[:, 2] + half_h - ego_z, pitch)
    visible = (
        (center_xy[:, 0] >= -behind_m) & (center_xy[:, 0] <= front_m) & (np.abs(center_xy[:, 1]) <= side_m)
    )
    if not np.any(visible):
        return empty if extra is None else (empty, np.zeros(0, dtype=np.float32))
    center_xy = center_xy[visible]
    rel_yaw = boxes[visible, 6] - ego_yaw
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
    nearest = np.argsort(np.einsum("ij,ij->i", center_xy, center_xy), kind="stable")[:max_entities]
    cuboids = cuboids[nearest]
    if extra is None:
        return cuboids.astype(np.float32)
    carried = np.asarray(extra, dtype=np.float64)[visible][nearest]
    return cuboids.astype(np.float32), carried.astype(np.float32)


def _box(x, y, yaw, z=0.0):
    return np.array([x, y, z, 1.9, 4.5, 1.5, yaw])


def _ego(t):
    """An ego that drives and turns: pose at time t."""
    yaw = 0.3 + 0.2 * t
    return [10.0 + 8.0 * t * np.cos(yaw), -3.0 + 8.0 * t * np.sin(yaw), 0.0, 1.9, 4.5, 1.5, yaw]


def _car(t, x0=40.0, y0=2.0, vx=5.0, vy=1.0):
    yaw = np.arctan2(vy, vx)
    return _box(x0 + vx * t, y0 + vy * t, yaw, z=0.1)


TIMES = [0.0, 0.25, 0.5]


@pytest.mark.parametrize("pitch", [0.0, 0.1])
@pytest.mark.parametrize("with_extra", [False, True])
def test_boxes_to_cuboids_unchanged(pitch, with_extra):
    rng = np.random.default_rng(0)
    boxes = np.column_stack(
        [rng.uniform(-40, 160, 30), rng.uniform(-40, 40, 30), rng.uniform(-1, 1, 30),
         rng.uniform(1, 2.5, 30), rng.uniform(3, 6, 30), rng.uniform(0, 2, 30), rng.uniform(-7, 7, 30)]
    )
    ego = [3.0, -1.0, 0.2, 1.9, 4.5, 1.5, 0.7]
    extra = rng.normal(size=30) if with_extra else None
    new = boxes_to_cuboids(boxes, ego, pitch=pitch, extra=extra, max_entities=12)
    old = _old_boxes_to_cuboids(boxes, ego, pitch=pitch, extra=extra, max_entities=12)
    if with_extra:
        assert new[0].shape[0] > 0
        for a, b in zip(new, old):
            np.testing.assert_array_equal(a, b)
    else:
        np.testing.assert_array_equal(new, old)


def test_extra_2d_and_empty():
    boxes = [_box(50, 0, 0), _box(10, 5, 1)]
    extra = np.arange(10, dtype=float).reshape(2, 5)
    cub, carried = boxes_to_cuboids(boxes, _ego(0), extra=extra)
    assert carried.shape == (2, 5) and cub.shape[0] == 2
    cub, carried = boxes_to_cuboids([], _ego(0), extra=np.zeros((0, 5)))
    assert carried.shape == (0, 5)
    cub, carried = boxes_to_cuboids([_box(5000, 0, 0)], _ego(0), extra=np.zeros((1, 5)))
    assert carried.shape == (0, 5)


@pytest.mark.parametrize("pitch", [0.0, 0.1])
def test_history_matches_ego_frame_transform(pitch):
    h = PoseHistory()
    for t in TIMES:
        h.push(t, {7: _car(t)})
    ego_now = _ego(TIMES[-1])
    hist, valid, times = h.export([7], [_car(TIMES[-1])], ego_now, pitch)
    assert hist.shape == (1, 3, 3) and valid.all()
    for j, t in enumerate(TIMES):
        c = boxes_to_cuboids([_car(t)], ego_now, pitch=pitch)[0]
        np.testing.assert_allclose(hist[0, j], [c[0], c[1], np.arctan2(c[4], c[3])], atol=1e-5)
    c = boxes_to_cuboids([_car(TIMES[-1])], ego_now, pitch=pitch)[0]
    np.testing.assert_array_equal(hist[0, -1], np.array([c[0], c[1], np.arctan2(c[4], c[3])], dtype=np.float32))


def test_heading_wrapped():
    ego = [0, 0, 0, 1.9, 4.5, 1.5, 3.0]
    out = box_history_to_ego(np.array([[_box(5, 0, -3.0)]]), ego)
    assert -np.pi < out[0, 0, 2] <= np.pi
    np.testing.assert_allclose(out[0, 0, 2], -6.0 + 2 * np.pi, atol=1e-6)


def test_actor_appearing_mid_episode():
    h = PoseHistory()
    h.push(0.0, {1: _car(0.0)})
    h.push(0.25, {1: _car(0.25), 2: _car(0.25, x0=60)})
    h.push(0.5, {1: _car(0.5), 2: _car(0.5, x0=60)})
    ego = _ego(0.5)
    cur = [_car(0.5), _car(0.5, x0=60)]
    hist, valid, _ = h.export([1, 2], cur, ego)
    np.testing.assert_array_equal(valid, [[True, True, True], [False, True, True]])
    # Filler is the current pose; the other actor is unaffected.
    np.testing.assert_array_equal(hist[1, 0], hist[1, 2])
    solo = PoseHistory()
    for t in TIMES:
        solo.push(t, {1: _car(t)})
    np.testing.assert_array_equal(solo.export([1], [cur[0]], ego)[0][0], hist[0])


def test_length_grows_then_saturates_and_times_relative():
    h = PoseHistory()
    for k, t in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        h.push(t, {1: _car(t)})
        hist, valid, times = h.export([1], [_car(t)], _ego(t))
        n = min(k + 1, 3)
        assert hist.shape[1] == valid.shape[1] == len(times) == n
        assert times[-1] == 0.0
    np.testing.assert_allclose(times, [-0.5, -0.25, 0.0])
    h.reset()
    h.push(0.0, {1: _car(0)})
    assert len(h.export([1], [_car(0)], _ego(0))[2]) == 1


def test_duplicate_push_replaces():
    h = PoseHistory()
    h.push(0.0, {1: _car(0.0)})
    h.push(0.25, {1: _car(0.25)})
    h.push(0.25, {1: _car(0.25), 2: _car(0.25)})
    hist, valid, times = h.export([1, 2], [_car(0.25), _car(0.25)], _ego(0.25))
    assert len(times) == 2
    np.testing.assert_array_equal(valid, [[True, True], [False, True]])


def test_empty_export():
    h = PoseHistory()
    h.push(0.0, {})
    hist, valid, times = h.export([], np.zeros((0, 7)), _ego(0))
    assert hist.shape == (0, 1, 3) and valid.shape == (0, 1) and len(times) == 1


def test_cull_keeps_past_and_rows_align():
    ids = [10, 11, 12]

    def world(t):
        # 10 approaches from beyond the 120 m cull; 11 is near; 12 is behind 11.
        return {10: _box(150.0 - 80.0 * t, 0.0, np.pi, 0.0), 11: _box(30.0, 5.0, 0.0), 12: _box(15.0, -4.0, 0.2)}

    h = PoseHistory()
    for t in TIMES:
        h.push(t, world(t))
    ego = [0.0, 0.0, 0.0, 1.9, 4.5, 1.5, 0.0]
    cur = [world(TIMES[-1])[i] for i in ids]
    hist, valid, _ = h.export(ids, cur, ego)
    # 10 was at x=150 (outside the visibility box) at t=0 and is at x=110 now.
    assert hist[0, 0, 0] > 120.0 and hist[0, -1, 0] <= 120.0
    extra = np.concatenate([np.asarray(ids, float)[:, None], hist.reshape(3, 9), valid.astype(float)], axis=1)
    cub, carried = boxes_to_cuboids(cur, ego, extra=extra)
    assert cub.shape[0] == 3
    assert carried[:, 0].tolist() == [12.0, 11.0, 10.0]  # nearest first
    got = carried[:, 1:10].reshape(3, 3, 3)
    np.testing.assert_array_equal(got[0], hist[2])
    np.testing.assert_array_equal(got[2], hist[0])
    np.testing.assert_allclose(got[:, -1, 0], cub[:, 0])
    np.testing.assert_allclose(got[:, -1, 1], cub[:, 1])
    np.testing.assert_allclose(got[:, -1, 2], np.arctan2(cub[:, 4], cub[:, 3]), atol=1e-6)
