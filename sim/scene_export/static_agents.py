"""Recover the vehicles baked into a scene's background reconstruction as boxes.

HUGSIM's dynamic actors are the ones a scenario inserts: one or two 3DRealCar models driven
by the planner. Everything else on the street -- every parked car, every vehicle that was
standing still when the scene was recorded -- is part of the static Gaussian model. The
Gaussian renderer shows them, so they are in the photorealistic observation the other AD
sides consume, but nothing described them as geometry, so the abstract export shipped
an empty street with one car on it.

That gap is not cosmetic. ``HUGSimEnv`` builds its background collision set from exactly
these Gaussians (``semantic > 1 & != 10 & opacity > 0.8``, which includes the vehicle
classes) and ``bg_collision_det`` ends the episode on contact. Without this module the
policy is failing episodes on obstacles its observation never showed it, which is not a
mistake it can learn its way out of.

The scene models carry a Cityscapes label per Gaussian, so the vehicles can be selected
directly; what is left is turning a labelled point cloud into instances and instances into
oriented boxes. Both steps are approximations, and the rejection rules at the end are what
keep an approximation from inventing an obstacle that is not there.

Coordinates
-----------
``world``
    The Gaussian model's frame. Y points down.
``sim``
    ``x = Z_world``, ``y = -X_world``, ``z = -Y_world``. The frame ``ego_box`` and
    ``obj_boxes`` are written in, and what this module returns.
"""

import numpy as np

# Cityscapes ids carried on the Gaussians' 3D features. The scene's semantics come from
# InverseForm, so this is the standard 19-class train set; `sim.utils` relies on the same
# numbering (0 road, 1 sidewalk, 10 sky) when it splits ground from scene.
CITYSCAPES_CAR = 13
CITYSCAPES_TRUCK = 14
CITYSCAPES_BUS = 15
VEHICLE_CLASSES = (CITYSCAPES_CAR, CITYSCAPES_TRUCK, CITYSCAPES_BUS)

# A Gaussian this faint contributes almost nothing to the render, and including the haze
# around an object inflates its box. Matches the threshold HUGSimEnv uses for its collision
# point set, so what is exported and what the ego can crash into agree.
MIN_OPACITY = 0.8

# Instance clustering. `eps` is in metres, on the ground plane. Adjacent parked cars in a row
# are genuinely within half a metre of each other, so no single eps separates them; the split
# pass below is what deals with that rather than a smaller eps, which would shatter one car.
CLUSTER_EPS_M = 0.6
CLUSTER_MIN_SAMPLES = 20
MAX_CLUSTER_POINTS = 60_000

# Plausibility bounds for a fitted box, in metres. Anything outside is a clustering artifact
# -- a wall fragment mislabelled as a bus, or a few points of a car's shadow -- and is
# dropped rather than exported as an obstacle that is not there.
MIN_LENGTH_M, MAX_LENGTH_M = 1.8, 14.0
MIN_WIDTH_M, MAX_WIDTH_M = 1.0, 3.6
# How long a blob may be before it is split. Not MAX_LENGTH_M: a row of parked cars or a line
# of truck trailers clusters as one blob 15-27 m long (nuScenes 0071, 0528, 0661, several
# pandaset and waymo scenes), and dropping it at 14 m took the whole row out of the export
# while bg_collision_det still ends an episode on it. _split_cluster cuts any length into
# vehicle pieces, which the per-piece bounds below then check; past this length a blob is more
# likely a mislabelled wall than vehicles.
MAX_BLOB_LENGTH_M = 40.0

# Bounds applied to the blob before it is split, rather than to a vehicle. A row of
# bay-parked cars is a car length across, so the blob's minor extent may legitimately reach
# a car length; a single vehicle never exceeds MAX_WIDTH_M once split.
MAX_BLOB_WIDTH_M = 6.5
MAX_PIECE_LENGTH_M = 7.0
MIN_HEIGHT_M, MAX_HEIGHT_M = 0.6, 4.5

# A fitted box longer than this is a row of parked cars that clustered as one blob; it is cut
# into this many metres per vehicle. Measured: rows fuse into 8-10 m clusters on kitti360 and
# pandaset at the eps above.
SPLIT_LENGTH_M = 6.5
NOMINAL_VEHICLE_LENGTH_M = 4.5
NOMINAL_VEHICLE_WIDTH_M = 1.85

# A blob whose minor extent reaches this is not one car wide -- it is one car *long*, seen
# across a row of bay-parked vehicles. Paired with an aspect test so a single car whose
# footprint happens to be squarish is not mistaken for a row of two.
PERPENDICULAR_ROW_MIN_WIDTH_M = 3.0
PERPENDICULAR_ROW_ASPECT = 1.6

# Boxes overlapping the recorded drive are almost always reconstruction smear from a vehicle
# that was moving when the scene was captured, not something parked in the road. The ego is
# scored on driving that line, so a phantom obstacle sitting on it would be unfair as well as
# wrong.
TRACK_CLEARANCE_M = 1.5

# How far a fitted box's base may sit from the road under it before the box is rejected. The
# ground function extrapolates away from the recorded track, so this is loose enough to
# tolerate that and tight enough to catch a cluster that is not on the road at all.
MAX_GROUND_OFFSET_M = 2.5


def tunables():
    """Every constant that steers the extraction, for the sidecar's fingerprint.

    Listed explicitly rather than scraped from the module namespace: a scrape would silently
    start covering an unrelated new constant, and silently stop covering one that got
    renamed, which is exactly the failure a cache key must not have.
    """
    return {
        "vehicle_classes": VEHICLE_CLASSES,
        "min_opacity": MIN_OPACITY,
        "cluster_eps_m": CLUSTER_EPS_M,
        "cluster_min_samples": CLUSTER_MIN_SAMPLES,
        "max_cluster_points": MAX_CLUSTER_POINTS,
        "length_m": (MIN_LENGTH_M, MAX_LENGTH_M),
        "width_m": (MIN_WIDTH_M, MAX_WIDTH_M),
        "height_m": (MIN_HEIGHT_M, MAX_HEIGHT_M),
        "max_blob_width_m": MAX_BLOB_WIDTH_M,
        "max_blob_length_m": MAX_BLOB_LENGTH_M,
        "max_piece_length_m": MAX_PIECE_LENGTH_M,
        "split_length_m": SPLIT_LENGTH_M,
        "nominal_length_m": NOMINAL_VEHICLE_LENGTH_M,
        "nominal_width_m": NOMINAL_VEHICLE_WIDTH_M,
        "perpendicular_min_width_m": PERPENDICULAR_ROW_MIN_WIDTH_M,
        "perpendicular_aspect": PERPENDICULAR_ROW_ASPECT,
        "track_clearance_m": TRACK_CLEARANCE_M,
        "max_ground_offset_m": MAX_GROUND_OFFSET_M,
    }


def _world_to_sim(xyz):
    """``(N, 3)`` Gaussian-world points -> the sim frame (x forward-ish, y left, z up)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    return np.stack([xyz[:, 2], -xyz[:, 0], -xyz[:, 1]], axis=1)


def _fit_box(cluster, max_height_m=MAX_HEIGHT_M, trim_pct=2.0):
    """Fit one oriented box to a cluster of vehicle Gaussians.

    Deliberately percentile-based rather than an exact min-area rectangle over the convex
    hull. A splat cloud has a halo, and a hull is decided entirely by its outermost points,
    so the exact fit is the one most sensitive to exactly the points that are least real: it
    came out 2.3-2.7 m wide on ordinary cars. Trimmed extents along the principal axes give
    up exactness for not being steered by the haze.

    The base comes from the cluster itself, not from the ground function. ``height_fn``
    projects onto the plane of the nearest recorded camera, which is a good description of
    the road under the track and a progressively worse one out where the parked cars are --
    measured up to a metre out. The tyres are a better datum than the extrapolation.

    Args:
        cluster: ``(N, 3)`` sim-frame points.
        max_height_m: points higher than this above the base are dropped before fitting.
            Facade and foliage mislabelled as vehicle sit above the roofline, and they
            inflate the footprint as well as the height.
        trim_pct: percentile trimmed from each end of each axis.

    Returns:
        ``(center_xy, length, width, yaw, base_z, height)``, or None if the cluster does not
        survive trimming.
    """
    cluster = np.asarray(cluster, dtype=np.float64)
    base_z = float(np.percentile(cluster[:, 2], trim_pct))
    body = cluster[cluster[:, 2] <= base_z + max_height_m]
    if body.shape[0] < 8:
        return None

    footprint = body[:, :2]
    centroid = np.median(footprint, axis=0)
    centred = footprint - centroid
    # Principal axis of the footprint is the vehicle's heading. Two-by-two, so an eigen
    # decomposition of the covariance is exact and cheap.
    _, vectors = np.linalg.eigh(np.cov(centred.T))
    major, minor = vectors[:, 1], vectors[:, 0]

    along = centred @ major
    across = centred @ minor
    lo_a, hi_a = np.percentile(along, [trim_pct, 100.0 - trim_pct])
    lo_c, hi_c = np.percentile(across, [trim_pct, 100.0 - trim_pct])
    length, width = float(hi_a - lo_a), float(hi_c - lo_c)
    center = centroid + 0.5 * (lo_a + hi_a) * major + 0.5 * (lo_c + hi_c) * minor
    yaw = float(np.arctan2(major[1], major[0]))
    if width > length:
        length, width = width, length
        yaw += 0.5 * np.pi

    height = float(np.percentile(body[:, 2], 100.0 - trim_pct) - base_z)
    return center, length, width, yaw, base_z, height


def _split_cluster(center, length, width, yaw):
    """Cut a fitted blob into vehicle-sized boxes, oriented the way the vehicles are.

    Adjacent parked cars are within half a metre of each other, so a row of them clusters as
    one blob and no single DBSCAN eps separates them without shattering individual cars. What
    the blob's own shape says is which way the cars inside it face, and it says it in the
    *minor* extent:

    * minor extent about a car wide -- the cars lie along the blob, nose to tail. Parallel
      parking. Split into car lengths along the major axis, heading along it.
    * minor extent about a car *long* -- the cars stand across the blob, side by side. Bay
      parking. Split into car widths along the major axis, heading across it.

    Assuming the first case unconditionally is what rotated every box in a car park by 90
    degrees: measured on nuScenes scene-0064, the big clusters come out 7-14 m along by 4.1 m
    across, and 4.1 m is a car's length.

    Returns:
        A list of ``(center_xy, length, width, yaw)``.
    """
    major = np.array([np.cos(yaw), np.sin(yaw)])

    if width >= PERPENDICULAR_ROW_MIN_WIDTH_M and length >= PERPENDICULAR_ROW_ASPECT * width:
        # Bay parking: the blob's minor extent is the vehicle length, and the major axis
        # counts vehicles rather than measuring one.
        count = max(1, int(round(length / NOMINAL_VEHICLE_WIDTH_M)))
        piece = length / count
        offsets = (np.arange(count) - 0.5 * (count - 1)) * piece
        return [
            (center + offset * major, width, piece, yaw + 0.5 * np.pi) for offset in offsets
        ]

    if length <= SPLIT_LENGTH_M:
        return [(center, length, width, yaw)]

    # Parallel parking, long enough to be more than one car. The split is even rather than
    # fitted: where the individual cars sit inside the blob is not recoverable from it, but
    # the occupied span is.
    count = int(np.ceil(length / NOMINAL_VEHICLE_LENGTH_M))
    piece = length / count
    offsets = (np.arange(count) - 0.5 * (count - 1)) * piece
    return [(center + offset * major, piece, width, yaw) for offset in offsets]


def extract_static_vehicles(
    xyz_world,
    semantic,
    opacity,
    height_fn,
    track_sim=None,
    classes=VEHICLE_CLASSES,
    min_opacity=MIN_OPACITY,
    eps_m=CLUSTER_EPS_M,
    min_samples=CLUSTER_MIN_SAMPLES,
):
    """Vehicles baked into the static scene, as sim-frame boxes.

    Args:
        xyz_world: ``(N, 3)`` Gaussian centres in the world frame.
        semantic: ``(N,)`` Cityscapes class id per Gaussian.
        opacity: ``(N,)`` opacity per Gaussian.
        height_fn: ``(a, b) -> sim z``, the drivable surface under a planner point. The env's
            ``sim_ground_height``; it seats each box on the road rather than on whatever the
            cluster's lowest Gaussian happened to be, which is often shadow bleed.
        track_sim: ``(M, 3)`` recorded drive in the sim frame, or None to skip the clearance
            test. Boxes closer than ``TRACK_CLEARANCE_M`` to it are dropped.
        classes: Cityscapes ids to treat as vehicles.
        min_opacity: drop Gaussians fainter than this.
        eps_m, min_samples: DBSCAN parameters, on the ground plane.

    Returns:
        ``(K, 7)`` float64 ``[x, y, z, w, l, h, yaw]`` in the sim frame, with ``z`` the box's
        base -- the same layout and convention as ``HUGSimEnv.objs_list``, so the two can be
        concatenated before being packed into agent cuboids.
    """
    xyz_world = np.asarray(xyz_world, dtype=np.float64)
    semantic = np.asarray(semantic).reshape(-1)
    opacity = np.asarray(opacity, dtype=np.float64).reshape(-1)

    keep = (
        np.isin(semantic, list(classes))
        & (opacity > min_opacity)
        & np.isfinite(xyz_world).all(axis=1)
    )
    if not np.any(keep):
        return np.zeros((0, 7), dtype=np.float64)

    points = _world_to_sim(xyz_world[keep])
    # DBSCAN is superlinear in the point count and a scene carries a few hundred thousand
    # vehicle Gaussians; a stride keeps the instance structure while bounding the cost.
    stride = max(1, points.shape[0] // MAX_CLUSTER_POINTS)
    points = points[::stride]

    from sklearn.cluster import DBSCAN

    labels = DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(points[:, :2])

    boxes = []
    for label in np.unique(labels[labels >= 0]):
        cluster = points[labels == label]
        if cluster.shape[0] < min_samples:
            continue
        fitted = _fit_box(cluster)
        if fitted is None:
            continue
        center, length, width, yaw, base_z, height = fitted
        # The ground function is not the datum here, but it is still the sanity check: a
        # cluster sitting metres off the road is a mislabelled balcony or roof rack, not a
        # parked car. planner (a, b) = (X_world, Z_world) = (-y_sim, x_sim).
        ground_z = float(np.atleast_1d(height_fn(-center[1], center[0]))[0])
        if abs(base_z - ground_z) > MAX_GROUND_OFFSET_M:
            continue
        if not (MIN_HEIGHT_M <= height <= MAX_HEIGHT_M):
            continue
        # Blob-level extent bounds are deliberately loose: a row of bay-parked cars is a car
        # *length* across and several cars along, so neither extent is a vehicle dimension
        # until the split has run. The per-vehicle bounds are applied to the pieces below.
        if not (MIN_LENGTH_M <= length <= MAX_BLOB_LENGTH_M):
            continue
        if not (MIN_WIDTH_M <= width <= MAX_BLOB_WIDTH_M):
            continue
        for piece_center, piece_length, piece_width, piece_yaw in _split_cluster(
            center, length, width, yaw
        ):
            if not (MIN_LENGTH_M <= piece_length <= MAX_PIECE_LENGTH_M):
                continue
            if not (MIN_WIDTH_M <= piece_width <= MAX_WIDTH_M):
                continue
            boxes.append(
                [
                    piece_center[0],
                    piece_center[1],
                    base_z,
                    piece_width,
                    piece_length,
                    height,
                    piece_yaw,
                ]
            )

    if not boxes:
        return np.zeros((0, 7), dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)

    if track_sim is not None and len(track_sim):
        track_xy = np.asarray(track_sim, dtype=np.float64)[:, :2]
        gap = np.linalg.norm(boxes[:, None, 0:2] - track_xy[None, :, :], axis=2).min(axis=1)
        boxes = boxes[gap > TRACK_CLEARANCE_M]

    return boxes
