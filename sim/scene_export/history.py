"""Per-actor pose history, exported in the current ego frame for temporal policies.

A policy that reads past frames needs the scene as it was a moment ago, often at a finer time
step than HUGSIM's snapshots, so it can interpolate each actor's pose between snapshots. Three
arrays are exported for that, aligned row for row with ``info['pictura']['cuboids']``:

* ``cuboid_history`` -- ``(K, N, 3)`` float32, each cuboid's ``(x, y, heading)`` at the last
  ``N`` HUGSIM snapshots, oldest first, the last being now, all in the CURRENT ego frame.
* ``cuboid_history_valid`` -- ``(K, N)`` bool, False where the actor did not exist.
* ``snapshot_times`` -- ``(N,)`` seconds relative to now (``[-0.5, -0.25, 0.0]``).

At episode start fewer than ``length`` snapshots exist, so ``N`` grows 1, 2, 3.
"""

import numpy as np

from sim.scene_export.entities import box_history_to_ego

# Snapshots kept. The consumer looks back by a stride of at most dmax seconds; HUGSIM steps
# every dt = 0.25 s, so it needs N = ceil(dmax / dt) + 1 snapshots (the +1 is "now").
# dmax = 0.5 s -> ceil(0.5 / 0.25) + 1 = 3.
SCENE_EXPORT_HISTORY_LEN = 3


class PoseHistory:
    """Ring of the last ``length`` snapshots, each ``(timestamp, {obj_id: box[7]})``.

    Boxes are sim-frame ``[x, y, z, w, l, h, yaw]``; the transform to the ego frame happens at
    export time, with the current ego, so past poses land in the frame ``cuboids`` uses.
    """

    def __init__(self, length=SCENE_EXPORT_HISTORY_LEN):
        self.length = int(length)
        self._snaps = []

    def reset(self):
        self._snaps = []

    def push(self, timestamp, boxes_by_id):
        """Record the actors' boxes at ``timestamp``.

        A second push at the same timestamp (``_get_info`` may run twice for one state)
        replaces the snapshot instead of adding a duplicate.
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
            number of snapshots held (at least the latest, which must be pushed first).
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
        times = times - times[-1] if n else times
        return box_history_to_ego(boxes, ego_box, pitch), valid, times
