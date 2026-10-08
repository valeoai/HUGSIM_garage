"""Per-actor pose history, exported in the current ego frame for temporal policies.

A policy that reads past frames needs the scene as it was a moment ago, often at a finer time
step than HUGSIM's snapshots, so it can interpolate each actor's pose between snapshots. Three
arrays are exported for that, aligned row for row with ``info['pictura']['cuboids']``:

* ``cuboid_history`` -- ``(K, N, 3)`` float32, each cuboid's ``(x, y, heading)`` at the last
  ``N`` HUGSIM snapshots, oldest first, the last being now, all in the CURRENT ego frame.
* ``cuboid_history_valid`` -- ``(K, N)`` bool, False where the actor did not exist.
* ``snapshot_times`` -- ``(N,)`` seconds relative to now: ``[-0.5, -0.25, 0.0]`` at HUGSIM's
  default step of 0.25 s, ``[-0.5, -0.4, ..., 0.0]`` at 0.1 s.

The ring holds enough snapshots to look back ``base.scene_export_history_seconds`` (default
``SCENE_EXPORT_HISTORY_SECONDS``) at the env's step (``history_length``). At episode start fewer
exist, so ``N`` grows 1, 2, ... up to that length; a consumer looking back by a stride ``d`` treats
the past as missing until ``snapshot_times[0] <= -d + 1e-6``. The times are rounded to the
microsecond, since the env's clock is a float sum of steps that drifts by an ulp.

``base.scene_export_history_seconds: 0`` turns the history off (``pose_history`` returns None):
a policy that reads only the present then pays nothing for it, and the three keys are left out
of ``info['pictura']``, whose other keys are unchanged. A negative or non-finite value is an error.
"""

import numpy as np

from sim.scene_export.entities import box_history_to_ego

# How far back the history reaches: the longest stride a consumer may look back by. The env
# takes it from base.scene_export_history_seconds when the config sets it.
SCENE_EXPORT_HISTORY_SECONDS = 0.5


def history_length(dt, seconds=SCENE_EXPORT_HISTORY_SECONDS):
    """Snapshots needed to look back ``seconds`` when the env steps every ``dt``.

    ``ceil(seconds / dt) + 1``, the +1 being "now": 3 at dt = 0.25 s, 6 at dt = 0.1 s. The
    tolerance keeps a quotient that is an integer up to rounding from taking one more.
    """
    return int(np.ceil(seconds / dt - 1e-6)) + 1


def pose_history(dt, seconds=SCENE_EXPORT_HISTORY_SECONDS):
    """The env's ring for a look-back of ``seconds``, or None when ``seconds`` is 0 (history off).

    Checked here, at env construction, because a negative look-back would otherwise size a ring
    that never trims and grows for the whole episode.
    """
    seconds = float(seconds)
    if not (np.isfinite(seconds) and seconds >= 0):
        raise ValueError(f"base.scene_export_history_seconds must be a finite number >= 0 (0 turns the history off), got {seconds}")
    if seconds == 0:
        return None
    return PoseHistory(history_length(dt, seconds))


class PoseHistory:
    """Ring of the last ``length`` snapshots, each ``(timestamp, {obj_id: box[7]})``.

    Boxes are sim-frame ``[x, y, z, w, l, h, yaw]``; the transform to the ego frame happens at
    export time, with the current ego, so past poses land in the frame ``cuboids`` uses.
    """

    def __init__(self, length):
        # At least "now"; a length of 0 would slice [:-0] in push, which trims nothing.
        if int(length) < 1:
            raise ValueError(f"PoseHistory needs a length >= 1, got {length}")
        self.length = int(length)
        self._snaps = []

    def reset(self):
        self._snaps = []

    def push(self, timestamp, boxes_by_id):
        """Record the actors' boxes at ``timestamp``.

        A second push at the same timestamp replaces the snapshot instead of adding a
        duplicate, so a caller that rebuilds the info of one state does not add a snapshot.
        """
        snap = (float(timestamp), {k: np.asarray(v, dtype=np.float64) for k, v in boxes_by_id.items()})
        if self._snaps and self._snaps[-1][0] == snap[0]:
            self._snaps[-1] = snap
        else:
            self._snaps.append(snap)
            del self._snaps[: -self.length]

    def export(self, obj_ids, current_boxes, ego_box, pitch=0.0):
        """History of ``obj_ids`` (in that order) in the current ego frame.

        Args:
            obj_ids: the actors to export.
            current_boxes: their current boxes, ``(M, 7)``; the filler for snapshots where
                an actor was absent (valid is False there).
            ego_box: the current ego box.
            pitch: the rig's mount pitch.

        Returns:
            ``(history (M, N, 3) float32, valid (M, N) bool, times (N,) float)``, ``N`` the
            number of snapshots held (at least the latest, which must be pushed first), times
            relative to now and rounded to the microsecond.
        """
        n = len(self._snaps)
        m = len(obj_ids)
        current = np.asarray(current_boxes, dtype=np.float64).reshape(m, 7)
        boxes = np.repeat(current[:, None, :], n, axis=1)
        valid = np.zeros((m, n), dtype=bool)
        for j, (_, snap) in enumerate(self._snaps):
            for i, oid in enumerate(obj_ids):
                if oid in snap:
                    boxes[i, j] = snap[oid]
                    valid[i, j] = True
        times = np.array([t for t, _ in self._snaps], dtype=np.float64)
        times = np.round(times - times[-1], 6) if n else times
        return box_history_to_ego(boxes, ego_box, pitch), valid, times
