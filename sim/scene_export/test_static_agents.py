import numpy as np

from sim.scene_export import static_agents as sa


def _row(length, width=2.6, height=3.6, n=40000, seed=0, x0=0.0, y0=8.0):
    """World-frame Gaussians of a solid box ``length`` along sim x, ``width`` along sim y, on z = 0."""
    rng = np.random.default_rng(seed)
    sim = np.stack([x0 + rng.uniform(0, length, n), y0 + rng.uniform(0, width, n), rng.uniform(0, height, n)], 1)
    world = np.stack([-sim[:, 1], -sim[:, 2], sim[:, 0]], 1)  # inverse of _world_to_sim
    return world


def _extract(world):
    n = world.shape[0]
    return sa.extract_static_vehicles(world, np.full(n, sa.CITYSCAPES_TRUCK), np.ones(n), lambda a, b: np.zeros_like(np.atleast_1d(a), dtype=float))


def test_a_row_longer_than_one_vehicle_bound_is_split_not_dropped():
    boxes = _extract(_row(22.0))  # three truck trailers nose to tail, 22 m
    assert boxes.shape[0] >= 4
    span = boxes[:, 0].max() - boxes[:, 0].min() + boxes[:, 4].max()
    assert 18.0 < span < 24.0
    assert (boxes[:, 4] <= sa.MAX_PIECE_LENGTH_M).all() and (boxes[:, 3] <= sa.MAX_WIDTH_M).all()


def test_a_blob_longer_than_any_row_is_still_dropped():
    assert _extract(_row(45.0)).shape[0] == 0


def test_the_blob_bound_is_in_the_fingerprint():
    assert sa.tunables()["max_blob_length_m"] == sa.MAX_BLOB_LENGTH_M


if __name__ == "__main__":
    test_a_row_longer_than_one_vehicle_bound_is_split_not_dropped()
    test_a_blob_longer_than_any_row_is_still_dropped()
    test_the_blob_bound_is_in_the_fingerprint()
    print("3 passed")
