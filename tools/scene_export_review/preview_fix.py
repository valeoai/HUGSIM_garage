"""Rebuild one scene's road map offline from its dumped evidence, with the worktree's fix file.

usage: preview_fix.py <evidence.npz> <scene_dir> <out_prefix> [--nofix]
Writes <out_prefix>.png/.json via road_bev.py and <out_prefix>_roads.npy. With no fix, also
reports how closely the offline rebuild reproduces the roads the env built.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim.scene_export import road_fixes  # noqa: E402
from sim.scene_export.map_source import GroundTrajectoryMapSource, MIN_RENDERED_ROAD_POINTS  # noqa: E402

import road_bev  # noqa: E402

ev_path, scene, out = sys.argv[1:4]
nofix = "--nofix" in sys.argv
ev = np.load(ev_path)
cam = ev["cam_poses"]
traj = np.stack([cam[:, 0, 3], cam[:, 2, 3]], 1)
fix = None if nofix else road_fixes.load(scene)
fill = {}
if fix is not None and fix.get("fill"):
    # what hug_sim hands a fill: road and sidewalk/terrain Gaussians of the full set (scene + ground model)
    import torch
    mp, _ = torch.load(scene + "/scene.pth", weights_only=False, map_location="cpu")
    parts = [mp, mp[10]]
    xyz = np.concatenate([p[1].detach().numpy() for p in parts]).astype(np.float64)
    sem = np.concatenate([p[4].detach().numpy().argmax(1) for p in parts])
    g = ev["ground_xyz"]
    k = xyz[np.isin(sem, (1, 9))]
    fill = {"fill_road_ab": np.stack([g[:, 0], g[:, 2]], 1), "fill_kerb_ab": np.stack([k[:, 0], k[:, 2]], 1)}
if len(ev["road_ab"]) >= MIN_RENDERED_ROAD_POINTS:
    src = GroundTrajectoryMapSource(traj, ground_ab=ev["road_ab"], nonroad_ab=ev["nonroad_ab"], road_fix=fix, **fill)
else:
    g = ev["ground_xyz"]
    src = GroundTrajectoryMapSource(traj, ground_ab=np.stack([g[:, 0], g[:, 2]], 1) if len(g) else None, road_fix=fix, **fill)
roads = src.build(lambda a, b: np.zeros(np.shape(np.atleast_1d(a)), dtype=float)).astype(np.float64)
ref = ev["roads"]
if fix is None and roads.shape == ref.shape:
    print("reproduces env build: max xy diff %.4f m" % np.abs(roads[:, [0, 1, 3, 4, 6]] - ref[:, [0, 1, 3, 4, 6]]).max())
elif fix is None:
    print("offline rebuild differs in shape", roads.shape, ref.shape)
else:
    print("fix:", road_fixes.fix_path(scene), "-", fix["note"])
np.save(out + "_roads.npy", roads)
road_bev.main(scene, out, out + "_roads.npy")
