"""asset_extent_wlh: an actor's size as its Gaussians render it."""

import numpy as np

from sim.scene_export.entities import asset_extent_wlh


def _car(w=1.8, l=4.6, h=1.6, n=20000, seed=0):
    """A box of opaque Gaussians in the asset frame (x length, y down, z across), bottom on y = 0."""
    rng = np.random.default_rng(seed)
    xyz = np.stack([rng.uniform(-l / 2, l / 2, n), rng.uniform(-h, 0.0, n), rng.uniform(-w / 2, w / 2, n)], axis=1)
    return xyz, np.full(n, 0.95)


def test_the_size_is_the_opaque_body():
    xyz, op = _car()
    w, l, h = asset_extent_wlh(xyz, op)
    assert abs(w - 1.8) < 0.05 and abs(l - 4.6) < 0.1 and abs(h - 1.6) < 0.05


def test_faint_floaters_and_ground_splats_do_not_grow_it():
    xyz, op = _car()
    rng = np.random.default_rng(1)
    floaters = rng.uniform([-6, -4, -3], [6, 0, 3], (3000, 3))
    shadow = np.stack([rng.uniform(-3.5, 3.5, 3000), rng.uniform(-0.05, 0.0, 3000), rng.uniform(-1.5, 1.5, 3000)], axis=1)
    xyz = np.concatenate([xyz, floaters, shadow])
    op = np.concatenate([op, np.full(3000, 0.3), np.full(3000, 0.95)])
    w, l, h = asset_extent_wlh(xyz, op)
    assert abs(w - 1.8) < 0.05 and abs(l - 4.6) < 0.1 and abs(h - 1.6) < 0.05


def test_too_few_points_keep_the_json():
    xyz, op = _car(n=50)
    assert asset_extent_wlh(xyz, op) is None
