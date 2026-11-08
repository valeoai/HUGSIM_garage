"""Cache the scene-level half of the abstract scene export next to the scene.

Neither the road geometry nor the parked cars move: they are properties of the reconstructed
scene, not of the scenario, the ego or the episode. Deriving them costs a sweep of the
Gaussian renderer plus a clustering pass, which is tens of seconds every time an episode
starts and identical every time. This writes the result beside the scene model and reads it
back on the next run.

The risk with any cache is silently serving a stale answer after the code that produced it
has changed -- which is the shape of the very bugs this export has already had. So nothing is
keyed on the scene alone: the fingerprint covers every constant that steers the result, and a
mismatch is a miss, not a warning. Change a threshold and the next run recomputes.

The sidecar lives in the scene's own directory when that is writable, because it belongs to
the scene; otherwise under ``$HUGSIM_SCENE_EXPORT_CACHE`` or ``~/.cache/hugsim_scene_export``.
"""

import hashlib
import os

import numpy as np

# Bump whenever a rebuild would produce different arrays for reasons the tunables do not
# capture: a new column, a different frame, or -- the easy one to forget -- a change to the
# *algorithm* at unchanged settings. Retuning a constant is already covered, because the
# constants are hashed; reordering the morphological passes is not.
#
#   1: initial
#   2: the kerb veto is re-applied after the morphological closing, so an adjoining car park
#      no longer joins the road through its own kerb
#   3: the drive is extended past both ends with a stamped road strip, so the outline no
#      longer closes across the carriageway just ahead of the ego's start
#   4: (tried) parked cars carved out of the drivable area; no measurable effect, removed
#   5: back to the version-3 algorithm, with the HD-map export off by default (the fingerprint's
#      load_hd_map flag now records which source actually ran)
#   6: the obstacle carve runs after the hand road fixes. No constant changed, so without this bump
#      the eight hand-fixed scenes kept serving maps built in the old order
SCHEMA_VERSION = 6

SIDECAR_NAME = "pictura_sidecar.npz"
FALLBACK_DIR = os.environ.get(
    "HUGSIM_SCENE_EXPORT_CACHE", os.path.expanduser("~/.cache/hugsim_scene_export")
)


def fingerprint(scene_name, load_hd_map, **tunables):
    """A short digest of everything that changes what the sidecar would contain.

    Args:
        scene_name: the scene the export was derived from.
        load_hd_map: whether the scenario asked for the HD map, which selects a different
            map source entirely.
        **tunables: every constant the extraction reads. Passed by name so the digest changes
            if one is renamed as well as if it is retuned.

    Returns:
        A hex string.
    """
    payload = repr(
        (SCHEMA_VERSION, str(scene_name), bool(load_hd_map), sorted(tunables.items()))
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _paths(model_path, scene_name):
    """Candidate sidecar paths, most preferred first."""
    candidates = []
    if model_path and os.path.isdir(model_path) and os.access(model_path, os.W_OK):
        candidates.append(os.path.join(model_path, SIDECAR_NAME))
    candidates.append(os.path.join(FALLBACK_DIR, f"{scene_name}_{SIDECAR_NAME}"))
    return candidates


def load(model_path, scene_name, digest):
    """The cached export, or None on a miss.

    A miss is anything other than a file that exists, parses, and carries exactly this
    fingerprint -- including a file written by a different version of the code. ``refined``
    in the result says whether the headings came from the orientation model or the geometry.
    """
    for path in _paths(model_path, scene_name):
        if not os.path.exists(path):
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["fingerprint"]) != digest:
                    continue
                # `refine_agent_yaw.py` may have settled each vehicle's heading against an
                # orientation model and written the result alongside the geometric fit. It is
                # stored beside the base rather than over it, so that a fingerprint miss --
                # which means the base was rederived and the refinement is stale -- discards
                # it by simply overwriting the file.
                refined = "static_agents_refined" in data.files
                return {
                    "roads": data["roads"],
                    "static_agents": (
                        data["static_agents_refined"] if refined else data["static_agents"]
                    ),
                    "track": data["track"],
                    "refined": refined,
                    "path": path,
                }
        except Exception:
            # A truncated or unreadable sidecar is a miss, not a failure: recomputing is
            # always available and always correct.
            continue
    return None


def save(model_path, scene_name, digest, roads, static_agents, track):
    """Write the sidecar. Returns the path written, or None if nowhere was writable.

    Written to a temporary name and moved into place, so a run interrupted mid-write leaves
    the previous sidecar intact rather than a half-file that the next run has to detect. A
    failure to write is reported rather than swallowed: a cache that never hits and never
    says so is indistinguishable from one that works.
    """
    problems = []
    for path in _paths(model_path, scene_name):
        # savez_compressed appends '.npz' unless the name already ends in it, so the
        # temporary name has to carry the suffix or the rename below chases a file that was
        # never written.
        tmp = f"{path}.{os.getpid()}.tmp.npz"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(
                tmp,
                fingerprint=np.array(digest),
                roads=np.asarray(roads, dtype=np.float32),
                static_agents=np.asarray(static_agents, dtype=np.float64),
                track=np.asarray(track, dtype=np.float64),
            )
            os.replace(tmp, path)
            return path
        except OSError as exc:
            problems.append(f"{path}: {exc}")
            if os.path.exists(tmp):
                os.remove(tmp)
    # Not fatal -- the export is already computed and correct -- but silence here would look
    # exactly like a cache that works and never hits.
    print("[scene_export] sidecar not written; " + "; ".join(problems))
    return None
