"""Map sources: a scene's road geometry as typed segments in the simulator frame.

A map source produces one array of shape ``(N, 7)``: ``[x0, y0, z0, x1, y1, z1, class]``,
each row a single 3D segment carrying one of the five drawable classes. That is
exactly the road entity the renderer takes (``POLY_ROAD_ENTITY_FIELDS``), so nothing
downstream has to know which source produced it.

There are two, because HUGSIM's HD map coverage is thin. :class:`TrajdataMapSource` reads
the real vector map through the same trajdata handle the traffic planner uses, and applies
to the nuScenes scenes whose scenario sets ``load_HD_map``. :class:`GroundTrajectoryMapSource`
applies to every scene: it takes the recorded drive as a lane centerline and measures the
road edges off the ground Gaussians, which are the only description of the drivable surface
the other three datasets carry.

Coordinates
-----------
Three frames meet here, so they are named throughout:

``world``
    The Gaussian model's frame. Y points down; the planner tracks ego as ``(X, Z)``.
``planner``
    The ``(a, b) = (X, Z)`` pair the planner and :class:`~sim.utils.plan.UnifiedMap` use.
``sim``
    The frame ``ego_box`` and ``obj_boxes`` are written in: ``x = b``, ``y = -a``,
    ``z`` up. the world convention every source returns.
"""

import numpy as np

from sim.scene_export import road_fixes

# The drawable road classes (datatypes.h, ROAD_CLASS_*). The values are the
# `type_code` the renderer reads out of the road entity, so they are not ours to renumber.
ROAD_CLASS_LANE = 0
ROAD_CLASS_LINE = 1
ROAD_CLASS_EDGE = 2
ROAD_CLASS_CROSSWALK = 3
ROAD_CLASS_SPEED_BUMP = 4

# Lane divisions to place either side of the centerline. The drivable area is one connected
# surface, so where a car park adjoins the road it is measured as road width, and an
# unbounded division count then paints lane markings across the parking bays. Two per side
# spans a four-lane carriageway, which is the widest road these scenes carry.
MAX_LANE_DIVISIONS = 2

# Below this many back-projected road pixels the sweep saw too little to trace a map from --
# a scene whose semantic head found almost no road, say -- and the Gaussian centres, coarse
# but scene-wide, are the better of two poor options.
MIN_RENDERED_ROAD_POINTS = 20_000

# Polylines are resampled to this spacing before being cut into segments. Two things set it:
# the ground under a segment is sampled at its midpoint only, so a long segment on a slope
# floats or sinks; and the renderer's per-camera budget is 512 segments, which a finer step
# would spend on geometry the ego drives past in a second.
RESAMPLE_STEP_M = 2.0

# The drivable area is traced from road pixels seen along the recorded drive, and the first
# camera never sees the ground under and just ahead of itself (nor the last one past its own
# position). Traced as is, the outline closes across the carriageway 1-6 m into the drive --
# a kerb drawn in front of an ego that starts at the beginning of the track, which a policy
# trained never to cross a road edge will not drive over. So the drive is extended straight
# past both ends by END_EXTENSION_M and a road strip is stamped along it, as wide as the
# surface measured END_PROBE_M in from that end.
END_EXTENSION_M = 30.0
END_PROBE_M = 10.0


# The drivable area is road evidence -- what the cameras label road -- but HUGSIM ends an episode
# on its own collision set: the scene Gaussians that are neither road, sidewalk nor sky
# (HUGSimEnv.points, bg_collision_det). Where the two disagree the raster shows road that HUGSIM
# scores as a crash: a grass traffic island the hole filling closed over (nuScenes scene-0013), bins
# and a hedge on the kerb the road outline runs past (0411, 0418), a cyclist smeared into the static
# scene mid-lane (kitti360 4480_4680). So cells holding those Gaussians at car height are carved
# out of the drivable area. Vehicles are left in: they are exported as boxes, not as road holes.
OBSTACLE_EXCLUDED_CLASSES = (13, 14, 15, 16)
OBSTACLE_HEIGHT_M = (0.05, 1.8)        # above the road surface: the ego box's span, not the canopy over it
# Density, not a per-cell count: HUGSIM ends an episode on > 100 collision Gaussians anywhere in the
# ego box (~8 m2, ~12 per m2), and the fences, hedges and smears it stops on are that sparse. A cell
# is an obstacle when the 1 m window around it holds this many per m2, a little below that bar.
OBSTACLE_MIN_PER_M2 = 8.0
OBSTACLE_WINDOW_M = 1.0
OBSTACLE_MIN_AREA_M2 = 0.25            # smaller blobs are floaters, not something the road stops at
# The road outline is smoothed over smooth_m and simplified to simplify_m, which erases a 1.5 m hole
# (a cyclist) and flattens a 1 m notch (bins on the kerb). The carved obstacles are traced on their
# own, this finely, so the raster draws an edge where HUGSIM counts a collision.
OBSTACLE_OUTLINE_SMOOTH_M = 0.75
OBSTACLE_OUTLINE_SIMPLIFY_M = 0.4
# Only obstacles this close to the drive line get the fine outline: the road block is capped at
# MAX_ROAD_ENTITIES (512) segments a step, nearest first, and outlining every hedge along a street
# pushes the far road ahead out of it. Every collision found was within 1.5 m of the drive line; a
# traffic island at a junction within 6 m. Farther obstacles are still carved, outlined coarsely.
OBSTACLE_OUTLINE_REACH_M = 6.0


def tunables():
    """Every constant that steers the map build, for the sidecar's fingerprint.

    Includes ``GroundTrajectoryMapSource``'s constructor defaults, because that is where the
    grid resolution and the contour smoothing actually live and both move the boundary.
    """
    import inspect

    defaults = {
        name: param.default
        for name, param in inspect.signature(GroundTrajectoryMapSource.__init__).parameters.items()
        # the road fix and its Gaussian evidence are per scene, not a tuning; the sidecar hashes
        # the fix separately and only for the scenes that have one
        if param.default is not inspect.Parameter.empty
        and name not in ("road_fix", "fill_road_ab", "fill_kerb_ab", "obstacle_ab")
    }
    return {
        "resample_step_m": RESAMPLE_STEP_M,
        "max_lane_divisions": MAX_LANE_DIVISIONS,
        "min_rendered_road_points": MIN_RENDERED_ROAD_POINTS,
        "end_extension_m": END_EXTENSION_M,
        "end_probe_m": END_PROBE_M,
        "obstacle_carve": (OBSTACLE_EXCLUDED_CLASSES, OBSTACLE_HEIGHT_M, OBSTACLE_MIN_PER_M2, OBSTACLE_WINDOW_M, OBSTACLE_MIN_AREA_M2,
                           OBSTACLE_OUTLINE_SMOOTH_M, OBSTACLE_OUTLINE_SIMPLIFY_M, OBSTACLE_OUTLINE_REACH_M),
        "ground_source": sorted(defaults.items()),
    }


def _planner_to_sim_xy(ab):
    """``(..., 2)`` planner ``(a, b)`` -> sim ``(x, y)``."""
    ab = np.asarray(ab, dtype=np.float64)
    return np.stack([ab[..., 1], -ab[..., 0]], axis=-1)


def _resample_polyline(points, step=RESAMPLE_STEP_M):
    """Resample an open polyline to roughly uniform spacing, keeping both ends.

    Args:
        points: ``(N, 2)`` planner-frame vertices.
        step: target spacing in meters.

    Returns:
        ``(M, 2)`` resampled vertices, or the input when it is too short to resample.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 2:
        return points
    seg_len = np.linalg.norm(np.diff(points, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = dist[-1]
    if total < 1e-6:
        return points[:1]
    n = max(2, int(np.ceil(total / step)) + 1)
    sample_at = np.linspace(0.0, total, n)
    return np.stack([np.interp(sample_at, dist, points[:, 0]), np.interp(sample_at, dist, points[:, 1])], axis=1)


def _extend_ends(points, length_m, step=RESAMPLE_STEP_M, fit_m=6.0):
    """Extrapolate a resampled polyline straight past both ends along its end tangents.

    Returns ``(extended, head)``: the extended polyline and how many points were prepended,
    so ``extended[head : head + len(points)]`` is the original.
    """
    n = points.shape[0]
    count = int(round(length_m / step))
    if count <= 0 or n < 2:
        return points, 0
    k = min(n - 1, max(1, int(round(fit_m / step))))

    def direction(a, b):
        d = a - b
        return d / max(np.linalg.norm(d), 1e-9)

    back = direction(points[0], points[k])
    fwd = direction(points[-1], points[-1 - k])
    offsets = step * np.arange(1, count + 1)[:, None]
    head = points[0] + back * offsets[::-1]
    tail = points[-1] + fwd * offsets
    return np.vstack([head, points, tail]), count


def _polyline_to_segments(points, road_class, height_fn, close=False):
    """Cut a planner-frame polyline into sim-frame segment rows.

    Args:
        points: ``(N, 2)`` planner-frame vertices.
        road_class: one of the ``ROAD_CLASS_*`` codes.
        height_fn: ``(a, b) -> sim z``, the drivable surface under a planner point.
        close: join the last vertex back to the first (polygons: road areas, crosswalks).

    Returns:
        ``(M, 7)`` float32 rows, empty when the polyline degenerates to a point.
    """
    points = _resample_polyline(points)
    if points.shape[0] < 2:
        return np.zeros((0, 7), dtype=np.float32)
    if close and np.linalg.norm(points[0] - points[-1]) > 1e-6:
        points = np.vstack([points, points[:1]])

    xy = _planner_to_sim_xy(points)
    z = height_fn(points[:, 0], points[:, 1])

    segments = np.empty((points.shape[0] - 1, 7), dtype=np.float32)
    segments[:, 0:2] = xy[:-1]
    segments[:, 2] = z[:-1]
    segments[:, 3:5] = xy[1:]
    segments[:, 5] = z[1:]
    segments[:, 6] = road_class
    return segments


def _finite_rows(points):
    """Drop rows holding a NaN or an inf, keeping the array's shape otherwise."""
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return points
    keep = np.isfinite(points).all(axis=1)
    return points if keep.all() else points[keep]


class MapSource:
    """Produces a scene's road segments once, at episode setup."""

    def build(self, height_fn):
        """Return ``(N, 7)`` sim-frame segments. ``height_fn`` maps planner ``(a, b)`` to sim z."""
        raise NotImplementedError


class TrajdataMapSource(MapSource):
    """Road geometry from the trajdata vector map behind ``load_HD_map``.

    Lane centers become ``LANE``, their left and right edges ``LINE``, road-area outlines
    ``EDGE`` and pedestrian crossings ``CROSSWALK``. The vector map covers a whole nuScenes
    location, so elements are first cut down to a box around the recorded drive.

    The world->planner transform is :class:`~sim.utils.plan.UnifiedMap`'s, anchored on the
    first ego frame. Its accuracy is what limits this source: it was written to pick a lane
    for a traffic agent to follow, and a heading error that is harmless there shows up
    directly as map geometry sliding across the rendered image. Check a rendered frame
    against the RGB one before trusting a scene.
    """

    def __init__(self, unified_map, trajectory_ab, margin_m=80.0):
        """
        Args:
            unified_map: a built :class:`~sim.utils.plan.UnifiedMap`.
            trajectory_ab: ``(N, 2)`` planner-frame recorded drive, used to bound the extract.
            margin_m: how far beyond the drive's bounding box to keep map elements.
        """
        self.unified_map = unified_map
        self.trajectory_ab = np.asarray(trajectory_ab, dtype=np.float64)
        self.margin_m = float(margin_m)

    def _world_to_planner(self, xyz):
        """``(N, 3+)`` world-frame map points -> ``(N, 2)`` planner ``(a, b)``."""
        stat = np.zeros((xyz.shape[0], 4), dtype=np.float64)
        stat[:, :2] = xyz[:, :2]
        local = self.unified_map.batch_xyzr_world2local(stat)
        return local[:, :2]

    def _in_range(self, ab):
        lo = self.trajectory_ab.min(axis=0) - self.margin_m
        hi = self.trajectory_ab.max(axis=0) + self.margin_m
        return np.any(np.all((ab >= lo) & (ab <= hi), axis=1))

    def build(self, height_fn):
        from trajdata.maps.vec_map_elements import MapElementType

        vector_map = self.unified_map.vector_map
        out = []

        def add(points_xyz, road_class, close=False):
            if points_xyz is None or len(points_xyz) < 2:
                return
            ab = self._world_to_planner(np.asarray(points_xyz, dtype=np.float64))
            if not self._in_range(ab):
                return
            segments = _polyline_to_segments(ab, road_class, height_fn, close=close)
            if segments.shape[0]:
                out.append(segments)

        for lane in vector_map.elements.get(MapElementType.ROAD_LANE, {}).values():
            add(lane.center.xyz, ROAD_CLASS_LANE)
            if lane.left_edge is not None:
                add(lane.left_edge.xyz, ROAD_CLASS_LINE)
            if lane.right_edge is not None:
                add(lane.right_edge.xyz, ROAD_CLASS_LINE)

        for area in vector_map.elements.get(MapElementType.ROAD_AREA, {}).values():
            add(area.exterior_polygon.xyz, ROAD_CLASS_EDGE, close=True)
            for hole in area.interior_holes:
                add(hole.xyz, ROAD_CLASS_EDGE, close=True)

        for crosswalk in vector_map.elements.get(MapElementType.PED_CROSSWALK, {}).values():
            add(crosswalk.polygon.xyz, ROAD_CLASS_CROSSWALK, close=True)

        if not out:
            return np.zeros((0, 7), dtype=np.float32)
        return np.vstack(out)


def _smooth_closed(points, window):
    """Moving-average a traced contour, wrapping around because the contour is a loop.

    Marching squares walks a binary mask, so its output steps between cell boundaries: the
    path is in the right place but its heading changes at nearly every vertex, which the
    renderer draws as a ragged kerb. Averaging along the contour removes the stepping without
    moving the boundary, because the steps are symmetric about the true edge.

    Args:
        points: ``(N, 2)`` contour vertices.
        window: filter width in vertices; forced odd and clamped to the contour length.

    Returns:
        ``(N, 2)`` smoothed vertices.
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    window = min(int(window) | 1, n if n % 2 else n - 1)
    if window < 3:
        return points
    pad = window // 2
    kernel = np.ones(window) / window
    wrapped = np.vstack([points[-pad:], points, points[:pad]])
    return np.stack(
        [np.convolve(wrapped[:, axis], kernel, mode="valid") for axis in range(2)], axis=1
    )


def _contiguous_runs(flags):
    """Yield slices of consecutive True in a 1-D bool array."""
    flags = np.asarray(flags, dtype=bool)
    if not flags.any():
        return
    padded = np.concatenate([[False], flags, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, stop in zip(edges[::2], edges[1::2]):
        yield slice(int(start), int(stop))


class GroundTrajectoryMapSource(MapSource):
    """Road geometry recovered from the recorded drive and the ground Gaussians.

    Available on every scene, because both inputs are things HUGSIM already builds: the
    densified camera track in ``ground_param.pkl``, and the Gaussians the scene model labels
    as road surface. The track is the lane the ego is scored on, so it becomes the ``LANE``
    centerline; the road surface itself is measured, and its outline becomes ``EDGE``.

    The road is described as an *area*, not as a corridor around the track. An earlier
    version fitted a left and a right half-width per along-track bin, as a high quantile of
    the lateral offsets of the ground points in that bin. That estimator is only well posed
    where the road really is a ribbon with the ego down the middle of it: at a junction it
    has no answer and ran out to its clamp, and behind a parked car, where the road surface
    is simply not reconstructed, it collapsed onto the car. Measured on the exported
    geometry, the half-width jumped by more than a metre between neighbouring segments in
    26% of steps on kitti360 and 81% on nuScenes, which is what the rendered boundary looked
    like. Rasterizing the drivable surface and tracing its outline has an answer everywhere,
    including at junctions and side roads, and the closing pass below is what bridges the
    occlusion holes rather than a clamp.

    What it still cannot represent: real painted markings. Those are not recoverable from a
    road-surface point cloud, so ``LINE`` is placed at lane-width divisions of the *measured*
    drivable width -- honest about the road's extent, synthetic about where the paint is.
    """

    def __init__(
        self,
        trajectory_ab,
        ground_ab=None,
        nonroad_ab=None,
        lane_width_m=3.5,
        default_half_width_m=4.0,
        cell_m=0.25,
        close_m=1.25,
        min_points_per_cell=2,
        simplify_m=1.5,
        smooth_m=6.0,
        max_half_width_m=20.0,
        road_fix=None,
        fill_road_ab=None,
        fill_kerb_ab=None,
        obstacle_ab=None,
    ):
        """
        Args:
            trajectory_ab: ``(N, 2)`` planner-frame recorded drive.
            ground_ab: ``(M, 2)`` planner-frame evidence that a place is drivable -- either
                the positions of road-labelled Gaussians, or road pixels back-projected out
                of the renderer, which is sharper (see ``nonroad_ab``).
            nonroad_ab: ``(P, 2)`` planner-frame evidence that a place is *not* drivable --
                back-projected sidewalk and terrain pixels. Where both are present a cell
                goes to whichever has more votes, which puts the boundary on the kerb
                instead of wherever the road-labelled splats happen to peter out. None
                disables the veto, and the extent of the drivable evidence decides alone.
            lane_width_m: spacing between ``LINE`` markings.
            default_half_width_m: corridor half-width used when there is no ground to measure.
            cell_m: occupancy grid resolution.
            close_m: radius of the morphological closing that bridges gaps in the
                reconstructed surface -- the shadow under a parked car, a stretch the lidar
                never swept. Large enough to close a car's width, small enough not to swallow
                a traffic island.
            min_points_per_cell: ground Gaussians needed before a cell counts as drivable.
                Above one, so a single stray point does not extend the road.
            simplify_m: Douglas-Peucker tolerance on the traced outline.
            smooth_m: length of the moving average applied along the traced contour before
                simplification. This is what takes the marching-squares stepping out of the
                boundary; swept over two scenes, it drops the median direction change between
                neighbouring edge segments from 25 degrees to 0.
            max_half_width_m: how far the width probe looks before giving up.
            road_fix: the scene's hand correction (:func:`road_fixes.load`), or None.
            fill_road_ab, fill_kerb_ab: planner-frame road- and sidewalk/terrain-labelled
                Gaussians, which decide the drivable cells inside the fix's ``fill`` polygons.
            obstacle_ab: planner-frame positions of HUGSIM's collision Gaussians at car height
                (see ``OBSTACLE_*``; :func:`obstacle_points` selects them), carved out of the
                drivable area. None leaves the area as the road evidence gives it.
        """
        self.trajectory_ab = _finite_rows(np.asarray(trajectory_ab, dtype=np.float64))
        # A scene model can carry a handful of diverged Gaussians -- four out of half a
        # million in the Waymo scenes -- and any non-finite row would poison the grid bounds.
        self.ground_ab = None if ground_ab is None else _finite_rows(np.asarray(ground_ab, dtype=np.float64))
        self.nonroad_ab = None if nonroad_ab is None else _finite_rows(np.asarray(nonroad_ab, dtype=np.float64))
        self.lane_width_m = float(lane_width_m)
        self.default_half_width_m = float(default_half_width_m)
        self.cell_m = float(cell_m)
        self.close_m = float(close_m)
        self.min_points_per_cell = int(min_points_per_cell)
        self.simplify_m = float(simplify_m)
        self.smooth_m = float(smooth_m)
        self.max_half_width_m = float(max_half_width_m)
        self.road_fix = road_fix
        self.obstacle_ab = None if obstacle_ab is None else _finite_rows(np.asarray(obstacle_ab, dtype=np.float64))
        self.fill_road_ab = None if fill_road_ab is None else _finite_rows(np.asarray(fill_road_ab, dtype=np.float64))
        self.fill_kerb_ab = None if fill_kerb_ab is None else _finite_rows(np.asarray(fill_kerb_ab, dtype=np.float64))

    @staticmethod
    def _frenet_frame(centerline):
        """Unit tangents and left normals at every centerline vertex."""
        tangent = np.gradient(centerline, axis=0)
        norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        tangent = tangent / np.maximum(norm, 1e-9)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        return tangent, normal

    def _drivable_mask(self, track=None, core=None):
        """Rasterize the drivable surface. Returns ``(mask, origin_ab)`` or None.

        The mask is the connected component of reconstructed road that the ego's own drive
        runs through, so a car park across the street does not become road the policy is
        invited onto.

        Args:
            track: ``(K, 2)`` resampled drive, possibly extended past its ends
                (:func:`_extend_ends`). Defaults to the recorded drive, unextended.
            core: ``(start, stop)`` indices of the recorded part of ``track``. Everything
                outside it, plus ``END_PROBE_M`` inside each end, gets a road strip stamped
                along it (see ``END_EXTENSION_M``).
        """
        if self.ground_ab is None or self.ground_ab.shape[0] < 100:
            return None

        from scipy import ndimage

        if track is None:
            track = self.trajectory_ab
        margin = self.close_m + 2.0 * self.cell_m + self.default_half_width_m
        extent = [self.ground_ab, track]
        if self.road_fix is not None:
            # an added side street may reach past every road sample, so the grid must hold it
            extent += [road_fixes.sim_to_planner(p) for p in self.road_fix.get("add", []) + (self.road_fix.get("fill") or [])]
        lo = np.min([e.min(axis=0) for e in extent], axis=0) - margin
        hi = np.max([e.max(axis=0) for e in extent], axis=0) + margin
        shape = np.ceil((hi - lo) / self.cell_m).astype(int) + 1
        if shape.max() > 4000 or shape.min() < 2:
            return None

        def vote(points):
            idx = np.floor((points - lo) / self.cell_m).astype(int)
            idx = idx[(idx >= 0).all(axis=1) & (idx[:, 0] < shape[0]) & (idx[:, 1] < shape[1])]
            grid = np.zeros(tuple(shape), dtype=np.int32)
            np.add.at(grid, (idx[:, 0], idx[:, 1]), 1)
            return grid

        counts = vote(self.ground_ab)
        kerb = None
        if self.nonroad_ab is not None and self.nonroad_ab.shape[0]:
            # The kerb is where road stops winning, not where road evidence runs out.
            kerb = vote(self.nonroad_ab) > counts
        mask = counts >= self.min_points_per_cell
        if kerb is not None:
            mask &= ~kerb

        radius = max(1, int(round(self.close_m / self.cell_m)))
        span = np.arange(-radius, radius + 1)
        disk = (span[:, None] ** 2 + span[None, :] ** 2) <= radius * radius
        # Close first, then fill: closing bridges the gap a parked car leaves at the kerb,
        # filling removes the holes it leaves inside the road.
        #
        # Closing is indiscriminate about what it bridges, and a kerb is far narrower than
        # the gap it is sized for, so the kerb is stamped back out afterwards. Without that,
        # a car park adjoining the road joins it *through its own kerb* and the component
        # test can no longer tell them apart: on nuScenes scene-0064 the ego's component came
        # out 3598 m2 instead of 2189 m2, the difference being the car park.
        #
        # Not repeated after the fill. Filling can only close a region already enclosed by
        # road, so it cannot re-bridge a kerb that separates two areas; vetoing again there
        # only punches out strays and traffic islands, which changed the area by 0.5% while
        # adding 7% to the boundary's perimeter -- contour detail the renderer then draws as
        # notches in the kerb.
        mask = ndimage.binary_closing(mask, structure=disk)
        if kerb is not None:
            mask &= ~kerb
        mask = ndimage.binary_fill_holes(mask)
        if core is not None:
            mask = self._stamp_end_strips(mask, lo, track, core)
        # A reviewed scene's hand correction goes in next, then the obstacle carve, so both are what
        # the component test, the outline and the lane lines see.
        fill_votes = None
        if self.road_fix is not None and self.road_fix.get("fill") and self.fill_road_ab is not None:
            empty = np.zeros((0, 2))
            fill_votes = (vote(self.fill_road_ab), vote(self.fill_kerb_ab if self.fill_kerb_ab is not None else empty))
        mask, added = road_fixes.apply(
            mask, lo, self.cell_m, self.road_fix, fill_votes=fill_votes, close_cells=radius
        )
        # The obstacles win over a hand fix too: a fill box drawn over a kerb with bins on it must not
        # hand the raster back road where HUGSIM counts a collision.
        if self.obstacle_ab is not None and self.obstacle_ab.shape[0]:
            n = max(1, int(round(OBSTACLE_WINDOW_M / self.cell_m)))
            density = ndimage.uniform_filter(vote(self.obstacle_ab).astype(np.float64), size=n, mode="constant") / self.cell_m ** 2
            occupied = density >= OBSTACLE_MIN_PER_M2
            lab, count = ndimage.label(occupied)
            if count:
                sizes = np.bincount(lab.ravel())
                big = sizes * self.cell_m ** 2 >= OBSTACLE_MIN_AREA_M2
                big[0] = False
                carved = big[lab] & mask
                mask &= ~big[lab]
                if added is not None:
                    added &= ~big[lab]
                # remembered for build(): where the road evidence said drivable and HUGSIM has an obstacle
                self._carved = (carved, lo)

        track_idx = np.floor((track - lo) / self.cell_m).astype(int)
        track_idx = track_idx[
            (track_idx >= 0).all(axis=1)
            & (track_idx[:, 0] < shape[0])
            & (track_idx[:, 1] < shape[1])
        ]
        if not track_idx.shape[0]:
            return None
        labels, count = ndimage.label(mask)
        if count == 0:
            return None
        on_track = labels[track_idx[:, 0], track_idx[:, 1]]
        on_track = on_track[on_track > 0]
        if not on_track.size:
            return None
        keep = [np.bincount(on_track).argmax()]
        if added is not None:
            # regions a fix added are road even across a median from the drive
            keep += [k for k in np.unique(labels[added]) if k > 0]
        return np.isin(labels, keep), lo

    def _stamp_end_strips(self, mask, origin, track, core):
        """Mark a road strip along each unobserved end of the drive (see ``END_EXTENSION_M``).

        The strip's half-widths are the ones measured ``END_PROBE_M`` in from that end,
        clamped to [half a lane, ``default_half_width_m``], and it runs from that station out
        to the end of the extension, so it also covers the stretch the cameras never saw.
        """
        start, stop = core
        probe = int(round(END_PROBE_M / RESAMPLE_STEP_M))
        _, normal = self._frenet_frame(track)
        n = track.shape[0]
        runs = (
            (min(start + probe, stop - 1), range(0, min(start + probe, stop - 1) + 1)),
            (max(stop - 1 - probe, start), range(max(stop - 1 - probe, start), n)),
        )
        lo_w, hi_w = 0.5 * self.lane_width_m, self.default_half_width_m
        shape = mask.shape
        stamped = mask.copy()
        for station, span in runs:
            left, right = self._probe_half_widths(mask, origin, track[[station]], normal[[station]])
            left_w = float(np.clip(left[0], lo_w, hi_w))
            right_w = float(np.clip(right[0], lo_w, hi_w))
            idx = np.fromiter(span, dtype=int)
            if idx.size < 2:
                continue
            # Densify to the grid so consecutive 2 m stations leave no gaps between them.
            seg = np.linalg.norm(np.diff(track[idx], axis=0), axis=1)
            along = np.r_[0.0, np.cumsum(seg)]
            dense_s = np.arange(0.0, along[-1] + 1e-9, 0.5 * self.cell_m)
            centre = np.stack([np.interp(dense_s, along, track[idx, a]) for a in range(2)], axis=1)
            nrm = np.stack([np.interp(dense_s, along, normal[idx, a]) for a in range(2)], axis=1)
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
            lateral = np.arange(-right_w, left_w + 1e-9, 0.5 * self.cell_m)
            pts = (centre[:, None, :] + nrm[:, None, :] * lateral[None, :, None]).reshape(-1, 2)
            cell = np.floor((pts - origin) / self.cell_m).astype(int)
            ok = (cell >= 0).all(axis=1) & (cell[:, 0] < shape[0]) & (cell[:, 1] < shape[1])
            stamped[cell[ok, 0], cell[ok, 1]] = True
        return stamped

    def _outline_segments(self, mask, origin, height_fn, smooth_m=None, simplify_m=None, min_points=8):
        """Trace the drivable mask's boundary into ``EDGE`` segment rows."""
        from skimage.measure import approximate_polygon, find_contours

        out = []
        smooth_m = self.smooth_m if smooth_m is None else smooth_m
        simplify_m = self.simplify_m if simplify_m is None else simplify_m
        window = max(3, int(round(smooth_m / self.cell_m)))
        for contour in find_contours(mask.astype(float), 0.5):
            if contour.shape[0] < min_points:
                continue
            simplified = approximate_polygon(
                _smooth_closed(contour, window), tolerance=simplify_m / self.cell_m
            )
            if simplified.shape[0] < 3:
                continue
            points_ab = origin + simplified * self.cell_m
            segments = _polyline_to_segments(points_ab, ROAD_CLASS_EDGE, height_fn, close=True)
            if segments.shape[0]:
                out.append(segments)
        return out

    def _probe_half_widths(self, mask, origin, centerline, normal):
        """Drivable half-width left and right of each centerline vertex.

        Marched across the mask rather than estimated from the point cloud, so it inherits
        the closing and hole-filling and cannot be dragged out by a stray point.
        """
        steps = int(round(self.max_half_width_m / self.cell_m))
        shape = mask.shape
        widths = []
        for sign in (1.0, -1.0):
            reach = np.full(centerline.shape[0], 0.0)
            live = np.ones(centerline.shape[0], dtype=bool)
            for step in range(1, steps + 1):
                probe = centerline + sign * normal * (step * self.cell_m)
                idx = np.floor((probe - origin) / self.cell_m).astype(int)
                inside = (
                    (idx >= 0).all(axis=1) & (idx[:, 0] < shape[0]) & (idx[:, 1] < shape[1])
                )
                hit = np.zeros(centerline.shape[0], dtype=bool)
                hit[inside] = mask[idx[inside, 0], idx[inside, 1]]
                live &= hit
                if not live.any():
                    break
                reach[live] = step * self.cell_m
            widths.append(reach)
        return widths[0], widths[1]

    def _lane_lines(self, centerline, normal, left_w, right_w, height_fn):
        """``LINE`` polylines at lane-width divisions of the measured drivable width.

        Painted markings are not in the data. What is in the data is how wide the road is, so
        the markings are placed where lane divisions would fall across that width: the ego's
        own lane always, and further divisions only where the measured road is wide enough to
        hold them. On a narrow street that yields two lines; on a dual carriageway it yields
        the interior ones too, instead of the single invented pair a fixed offset gives.
        """
        half_lane = 0.5 * self.lane_width_m
        out = []
        for sign, reach in ((1.0, left_w), (-1.0, right_w)):
            offset = half_lane
            for _ in range(MAX_LANE_DIVISIONS):
                # Only where the road is actually that wide; elsewhere the line is not drawn
                # rather than being pushed onto the verge.
                on_road = reach >= offset + 0.25 * self.lane_width_m
                if on_road.sum() < 2:
                    break
                for run in _contiguous_runs(on_road):
                    if run.stop - run.start < 2:
                        continue
                    polyline = centerline[run] + sign * normal[run] * offset
                    segments = _polyline_to_segments(polyline, ROAD_CLASS_LINE, height_fn)
                    if segments.shape[0]:
                        out.append(segments)
                offset += self.lane_width_m
        return out

    def build(self, height_fn):
        recorded = _resample_polyline(self.trajectory_ab)
        if recorded.shape[0] < 2:
            return np.zeros((0, 7), dtype=np.float32)
        # The lane, its markings and the road outline all continue past the recorded drive;
        # only the goals (route_to_ego) stay on it.
        centerline, head = _extend_ends(recorded, END_EXTENSION_M)
        _, normal = self._frenet_frame(centerline)

        out = [_polyline_to_segments(centerline, ROAD_CLASS_LANE, height_fn)]
        drivable = self._drivable_mask(centerline, (head, head + recorded.shape[0]))
        if drivable is None:
            # No usable ground: fall back to a fixed-width corridor, which is at least
            # continuous, and say so by giving it the same classes.
            half = self.default_half_width_m
            half_lane = 0.5 * self.lane_width_m
            out += [
                _polyline_to_segments(centerline + normal * half_lane, ROAD_CLASS_LINE, height_fn),
                _polyline_to_segments(centerline - normal * half_lane, ROAD_CLASS_LINE, height_fn),
                _polyline_to_segments(centerline + normal * half, ROAD_CLASS_EDGE, height_fn),
                _polyline_to_segments(centerline - normal * half, ROAD_CLASS_EDGE, height_fn),
            ]
        else:
            mask, origin = drivable
            left_w, right_w = self._probe_half_widths(mask, origin, centerline, normal)
            out += self._lane_lines(centerline, normal, left_w, right_w, height_fn)
            out += self._outline_segments(mask, origin, height_fn)
            carved = getattr(self, "_carved", None)
            if carved is not None and carved[0].any():
                # obstacles inside the road, outlined finely; padded so a blob on the grid border closes
                cmask, corigin = carved
                from scipy import ndimage
                near = np.ones(cmask.shape, dtype=bool)
                idx = np.floor((centerline - corigin) / self.cell_m).astype(int)
                ok = (idx >= 0).all(axis=1) & (idx[:, 0] < cmask.shape[0]) & (idx[:, 1] < cmask.shape[1])
                near[idx[ok, 0], idx[ok, 1]] = False
                cmask = cmask & (ndimage.distance_transform_edt(near) * self.cell_m <= OBSTACLE_OUTLINE_REACH_M)
                out += self._outline_segments(
                    np.pad(cmask, 1), corigin - self.cell_m, height_fn,
                    smooth_m=OBSTACLE_OUTLINE_SMOOTH_M, simplify_m=OBSTACLE_OUTLINE_SIMPLIFY_M, min_points=4,
                )

        out = [s for s in out if s.shape[0]]
        if not out:
            return np.zeros((0, 7), dtype=np.float32)
        return np.vstack(out)


def obstacle_points(xyz_world, semantic, opacity, height_fn):
    """Planner ``(a, b)`` of HUGSIM's collision Gaussians that stand at car height above the road.

    ``xyz_world``, ``semantic``, ``opacity``: the scene Gaussians HUGSimEnv builds ``points`` from;
    the same mask (semantic > 1, != 10, opacity > 0.8) minus ``OBSTACLE_EXCLUDED_CLASSES``.
    ``height_fn``: planner ``(a, b)`` -> road surface z in the sim frame (``sim_ground_height``).
    """
    xyz = np.asarray(xyz_world, dtype=np.float64); sem = np.asarray(semantic).reshape(-1)
    op = np.asarray(opacity, dtype=np.float64).reshape(-1)
    keep = (sem > 1) & (sem != 10) & (op > 0.8) & ~np.isin(sem, OBSTACLE_EXCLUDED_CLASSES) & np.isfinite(xyz).all(axis=1)
    xyz = xyz[keep]
    if not xyz.shape[0]:
        return np.zeros((0, 2))
    a, b = xyz[:, 0], xyz[:, 2]
    dz = -xyz[:, 1] - np.asarray(height_fn(a, b), dtype=np.float64)
    lo, hi = OBSTACLE_HEIGHT_M
    band = (dz >= lo) & (dz <= hi)
    return np.stack([a[band], b[band]], axis=1)


def build_map_source(cfg, unified_map, cam_poses, ground_xyz=None, road_ab=None, nonroad_ab=None, road_fix=None,
                     kerb_xyz=None, obstacle_ab=None):
    """Pick a map source for a scene.

    The HD map wins where the scenario asked for one and it loaded; otherwise the recorded
    drive plus the ground Gaussians. Both return the same array, so the caller does not
    branch -- but which one ran matters when reading results, so it is logged.

    Args:
        cfg: the merged simulation config.
        unified_map: the env's :class:`~sim.utils.plan.UnifiedMap`, or None.
        cam_poses: ``(N, 4, 4)`` densified camera-to-world poses from ``ground_param.pkl``.
        ground_xyz: ``(M, 3)`` world-frame ground Gaussians, or None.
        road_ab: ``(N, 2)`` planner-frame road pixels back-projected out of the renderer, or
            None. Preferred over ``ground_xyz`` when present: it describes the surface the
            renderer shows rather than the primitives it is built from.
        nonroad_ab: ``(P, 2)`` the same for sidewalk and terrain, used to place the kerb.
        road_fix: the scene's hand correction (:mod:`road_fixes`), or None. Ignored by the HD map.
        kerb_xyz: ``(K, 3)`` world-frame sidewalk/terrain Gaussians, for the fix's ``fill``.
        obstacle_ab: :func:`obstacle_points`, carved out of the drivable area. Ignored by the HD map.

    Returns:
        A :class:`MapSource`.
    """
    # cam_poses are world c2w; the planner pair is (X, Z) of the translation.
    trajectory_ab = np.stack([cam_poses[:, 0, 3], cam_poses[:, 2, 3]], axis=1)

    if unified_map is not None:
        print("[scene_export] map source: trajdata HD map")
        return TrajdataMapSource(unified_map, trajectory_ab)

    fill = {}
    if road_fix is not None and road_fix.get("fill"):
        to_ab = lambda xyz: None if xyz is None or not len(xyz) else np.stack([xyz[:, 0], xyz[:, 2]], axis=1)  # noqa: E731
        fill = {"fill_road_ab": to_ab(ground_xyz), "fill_kerb_ab": to_ab(kerb_xyz)}

    if road_ab is not None and len(road_ab) >= MIN_RENDERED_ROAD_POINTS:
        print(
            f"[scene_export] map source: recorded drive + rendered road "
            f"({len(road_ab)} road px, {0 if nonroad_ab is None else len(nonroad_ab)} kerb px)"
        )
        return GroundTrajectoryMapSource(
            trajectory_ab, ground_ab=road_ab, nonroad_ab=nonroad_ab, road_fix=road_fix, obstacle_ab=obstacle_ab, **fill
        )

    ground_ab = None
    if ground_xyz is not None and len(ground_xyz):
        ground_ab = np.stack([ground_xyz[:, 0], ground_xyz[:, 2]], axis=1)
    print(
        "[scene_export] map source: recorded drive"
        + (" + ground Gaussians" if ground_ab is not None else " (fixed-width corridor)")
    )
    return GroundTrajectoryMapSource(trajectory_ab, ground_ab=ground_ab, road_fix=road_fix, obstacle_ab=obstacle_ab, **fill)
