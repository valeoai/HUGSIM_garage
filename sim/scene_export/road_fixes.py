"""Per-scene corrections to the drivable area the scene export traces its road map from.

The automatic map (:class:`~sim.scene_export.map_source.GroundTrajectoryMapSource`) sees the
road through the front cameras along the recorded drive, so it can cut a side street off at
the junction, or let the drivable area run onto a kerb, verge or car park that the scene
model labels as road. Where a scene was reviewed against its Gaussians and found wrong, a
fix file sets the drivable area right by hand:

``sim/scene_export/road_fixes/<dataset>/<scene dir>.yaml``::

    note: why the scene needed fixing
    fill:                    # polygons where the scene's Gaussians decide: a cell is drivable
      - [[x0, y0], ...]      #   where road-labelled Gaussians outvote sidewalk/terrain ones
    add:                     # polygons forced drivable, sim frame (x, y), metres
      - [[x0, y0], [x1, y1], [x2, y2], ...]
    remove:                  # polygons forced not drivable, applied after `fill` and `add`
      - [[x0, y0], ...]

``fill`` is for what the camera sweep under-sees -- a side street the drive only passes the
mouth of, a cross street behind the start -- where the reconstruction does hold the road: the
shape comes from the Gaussians, the polygon only says where to look. ``add`` is for where the
Gaussians are wrong too.

Both are applied to the occupancy grid before the component around the drive is picked,
the outline traced and the lane lines placed, so everything downstream follows. A region
``add`` touches is kept even when it is not connected to the drive (an opposite carriageway
across a median), and gets its own outline. Scoring, collisions and the HUGSIM planner never
read this.

The file's content is part of the scene sidecar's fingerprint (:func:`digest`), so editing
a fix rebuilds that scene's map on its next run and no other scene's.
"""

import hashlib
import os

import numpy as np
import yaml

# Closing radius of a fill, in metres (see apply).
FILL_CLOSE_M = 2.0

FIX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "road_fixes")


def fix_path(model_path):
    """``road_fixes/<dataset>/<scene dir>.yaml`` for a scene directory ``.../<dataset>/<scene dir>``."""
    model_path = os.path.normpath(str(model_path))
    return os.path.join(FIX_DIR, os.path.basename(os.path.dirname(model_path)), os.path.basename(model_path) + ".yaml")


def load(model_path):
    """The scene's fix as ``{'add': [(K, 2) sim xy], 'remove': [...], 'note': str}``, or None."""
    path = fix_path(model_path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    fix = {"note": str(raw.get("note", ""))}
    for key in ("fill", "add", "remove"):
        polys = []
        for poly in raw.get(key) or []:
            poly = np.asarray(poly, dtype=np.float64)
            if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
                raise ValueError(f"{path}: every '{key}' polygon needs >= 3 [x, y] vertices, got {poly.shape}")
            polys.append(poly)
        fix[key] = polys
    unknown = set(raw) - {"note", "fill", "add", "remove"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    return fix


def digest(fix):
    """A short digest of a fix for the sidecar fingerprint; '' when there is none."""
    if fix is None:
        return ""
    payload = repr([(k, [np.round(p, 3).tolist() for p in fix.get(k, [])]) for k in ("fill", "add", "remove") if fix.get(k)])
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def sim_to_planner(xy):
    """``(K, 2)`` sim ``(x, y)`` -> planner ``(a, b)`` (sim ``x = b``, ``y = -a``)."""
    xy = np.asarray(xy, dtype=np.float64)
    return np.stack([-xy[:, 1], xy[:, 0]], axis=1)


def polygon_mask(shape, origin_ab, cell_m, polygons):
    """``shape`` bool grid of the cells inside any of ``polygons`` (sim ``(x, y)``)."""
    from skimage.draw import polygon

    out = np.zeros(shape, dtype=bool)
    for poly in polygons:
        # cell (i, j) spans [origin + i * cell, origin + (i + 1) * cell); skimage puts pixel centres on integers
        idx = (sim_to_planner(poly) - np.asarray(origin_ab)) / cell_m - 0.5
        rr, cc = polygon(idx[:, 0], idx[:, 1], shape=shape)
        out[rr, cc] = True
    return out


def apply(mask, origin_ab, cell_m, fix, fill_votes=None, close_cells=5):
    """Apply a fix to a planner-frame occupancy grid: ``fill``, then ``add``, then ``remove``.

    Args:
        mask: ``(A, B)`` bool occupancy grid, axis 0 along planner ``a``, axis 1 along ``b``.
        origin_ab: planner ``(a, b)`` of cell ``(0, 0)``'s corner.
        cell_m: grid resolution.
        fix: a :func:`load` result, or None.
        fill_votes: ``(road, kerb)`` int grids of road- and sidewalk/terrain-labelled Gaussians
            per cell, needed when the fix has ``fill``.
        close_cells: radius, in cells, of the closing that bridges the gaps between the
            Gaussians in a fill; FILL_CLOSE_M wins where it is wider, because the Gaussians are
            sparser than the rendered road samples the base mask is closed over.

    Returns:
        ``(mask, added)``: the corrected grid, and the cells ``fill``/``add`` turned on (None
        without a fix), which the caller keeps as drivable even where they do not touch the
        drive -- an opposite carriageway across a median, say.
    """
    if fix is None or not (fix.get("fill") or fix.get("add") or fix.get("remove")):
        return mask, None
    added = polygon_mask(mask.shape, origin_ab, cell_m, fix.get("add", []))
    if fix.get("fill"):
        if fill_votes is None:
            raise ValueError("a road fix with `fill` needs the scene's Gaussian votes")
        from scipy import ndimage

        road, kerb = fill_votes
        where = polygon_mask(mask.shape, origin_ab, cell_m, fix["fill"])
        r = max(1, int(close_cells), int(round(FILL_CLOSE_M / cell_m)))
        span = np.arange(-r, r + 1)
        disk = (span[:, None] ** 2 + span[None, :] ** 2) <= r * r
        filled = ndimage.binary_closing((road > kerb) & (road > 0), structure=disk)
        filled = ndimage.binary_fill_holes(filled & ~(kerb > road)) & where
        added |= filled
    removed = polygon_mask(mask.shape, origin_ab, cell_m, fix.get("remove", []))
    added &= ~removed
    return (mask | added) & ~removed, added
