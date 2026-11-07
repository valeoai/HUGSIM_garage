import numpy as np

from sim.scene_export import road_fixes
from sim.scene_export.map_source import GroundTrajectoryMapSource


def _straight_road(half_width=4.0, length=60.0):
    rng = np.random.default_rng(0)
    b = rng.uniform(0, length, 40000)
    a = rng.uniform(-half_width, half_width, 40000)
    track = np.stack([np.zeros(61), np.linspace(0, length, 61)], 1)
    return track, np.stack([a, b], 1)


def test_add_and_remove_change_the_drivable_area_where_asked():
    track, ground = _straight_road()
    base = GroundTrajectoryMapSource(track, ground_ab=ground)
    mask0, lo = base._drivable_mask()
    # a side street on the sim left (y > 0 is planner a < 0), touching the road
    side = [[20.0, 3.0], [28.0, 3.0], [28.0, 20.0], [20.0, 20.0]]
    hole = [[40.0, -5.0], [44.0, -5.0], [44.0, 5.0], [40.0, 5.0]]
    fix = {"note": "", "add": [np.array(side)], "remove": [np.array(hole)]}
    mask1, lo1 = GroundTrajectoryMapSource(track, ground_ab=ground, road_fix=fix)._drivable_mask()
    cell = base.cell_m
    at = lambda m, o, x, y: m[int((-y - o[0]) // cell), int((x - o[1]) // cell)]
    assert not at(mask0, lo, 24.0, 12.0) and at(mask1, lo1, 24.0, 12.0)
    assert at(mask0, lo, 30.0, 0.0) and at(mask1, lo1, 30.0, 0.0)
    # the removed block cuts the road; the far part is no longer the drive's component
    assert at(mask0, lo, 42.0, 0.0) and not at(mask1, lo1, 42.0, 0.0)


def test_an_added_region_across_a_median_is_kept():
    track, ground = _straight_road()
    # a parallel carriageway 10-17 m to the sim right (planner a in [10, 17]), never touching the road
    far = [[5.0, -10.0], [55.0, -10.0], [55.0, -17.0], [5.0, -17.0]]
    fix = {"note": "", "add": [np.array(far)], "remove": []}
    src = GroundTrajectoryMapSource(track, ground_ab=ground, road_fix=fix)
    mask, lo = src._drivable_mask()
    at = lambda x, y: mask[int((-y - lo[0]) // src.cell_m), int((x - lo[1]) // src.cell_m)]
    assert at(30.0, -13.5) and at(30.0, 0.0) and not at(30.0, -7.0)
    roads = src.build(lambda a, b: np.zeros_like(np.asarray(a, dtype=float)))
    edges = roads[roads[:, 6] == 2]
    assert (edges[:, 1] < -9.0).any()  # the far carriageway has its own outline


def test_fill_takes_the_side_street_shape_from_the_gaussians():
    track, ground = _straight_road()
    rng = np.random.default_rng(1)
    # road Gaussians of a 6 m side street at sim x 22..28 running to y = 25; sidewalk beside it
    side = np.stack([-rng.uniform(4, 25, 30000), rng.uniform(22, 28, 30000)], 1)
    kerb = np.stack([-rng.uniform(4, 25, 30000), np.r_[rng.uniform(17, 22, 15000), rng.uniform(28, 33, 15000)]], 1)
    box = [[10.0, 2.0], [40.0, 2.0], [40.0, 30.0], [10.0, 30.0]]  # loose: the Gaussians decide
    fix = {"note": "", "fill": [np.array(box)], "add": [], "remove": []}
    src = GroundTrajectoryMapSource(track, ground_ab=ground, road_fix=fix, fill_road_ab=side, fill_kerb_ab=kerb)
    mask, lo = src._drivable_mask()
    at = lambda x, y: mask[int((-y - lo[0]) // src.cell_m), int((x - lo[1]) // src.cell_m)]
    assert at(25.0, 15.0) and at(25.0, 24.0)  # the street
    assert not at(19.0, 15.0) and not at(31.0, 15.0)  # its sidewalks stay out
    assert not at(25.0, 29.0)  # nothing past the Gaussians


def test_no_fix_file_is_none_and_digest_is_empty(tmp_path):
    assert road_fixes.load(tmp_path / "nowhere" / "000") is None
    assert road_fixes.digest(None) == ""


def test_digest_follows_the_polygons():
    a = {"note": "x", "add": [np.array([[0, 0], [1, 0], [1, 1.0]])], "remove": []}
    b = {"note": "y", "add": [np.array([[0, 0], [1, 0], [1, 1.0]])], "remove": []}
    c = {"note": "x", "add": [np.array([[0, 0], [2, 0], [1, 1.0]])], "remove": []}
    assert road_fixes.digest(a) == road_fixes.digest(b) != road_fixes.digest(c)


if __name__ == "__main__":
    import tempfile, pathlib
    test_add_and_remove_change_the_drivable_area_where_asked()
    test_an_added_region_across_a_median_is_kept()
    test_fill_takes_the_side_street_shape_from_the_gaussians()
    test_no_fix_file_is_none_and_digest_is_empty(pathlib.Path(tempfile.mkdtemp()))
    test_digest_follows_the_polygons()
    print("5 passed")
