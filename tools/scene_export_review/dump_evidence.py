"""Build one scene's HUGSIM env with the sidecar bypassed and save the map-source inputs.

usage: dump_evidence.py <scripts/hugsim dir> <scenario.yaml> <base.yaml> <camera.yaml> <kinematic.yaml> <out_dir>
Writes <out_dir>/<dataset>_<scene dir>.npz with cam_poses, road_ab, nonroad_ab, ground_xyz and
the roads the unfixed build produced (to check the offline rebuild reproduces it).
"""
import os
import sys

import numpy as np

scripts, scenario, base, camera, kinematic, out_dir = sys.argv[1:7]
sys.path.insert(0, scripts)
sys.path.insert(0, os.getcwd())
from capture_alignment import load_sim_cfg  # noqa: E402

import gymnasium  # noqa: E402
import hugsim_env  # noqa: E402,F401
import hugsim_env.envs.hug_sim as hs  # noqa: E402
import sim.scene_export.sidecar as sc  # noqa: E402

sc.load = lambda *a, **k: None
sc.save = lambda *a, **k: None
orig = hs.build_map_source
seen = {}


def spy(cfg, unified_map, cam_poses, ground_xyz=None, road_ab=None, nonroad_ab=None, road_fix=None, **kw):
    src = orig(cfg, unified_map, cam_poses, ground_xyz=ground_xyz, road_ab=road_ab, nonroad_ab=nonroad_ab, road_fix=road_fix, **kw)
    mp = os.path.normpath(cfg.model_path)
    seen.update(name=f"{os.path.basename(os.path.dirname(mp))}_{os.path.basename(mp)}", cam_poses=np.asarray(cam_poses),
                road_ab=np.zeros((0, 2)) if road_ab is None else np.asarray(road_ab),
                nonroad_ab=np.zeros((0, 2)) if nonroad_ab is None else np.asarray(nonroad_ab),
                ground_xyz=np.zeros((0, 3)) if ground_xyz is None else np.asarray(ground_xyz),
                source=type(src).__name__)
    real_build = src.build

    def build(height_fn):
        roads = real_build(height_fn)
        seen["roads"] = roads
        return roads

    src.build = build
    return src


hs.build_map_source = spy
cfg = load_sim_cfg(scenario, base, camera, kinematic, obs_mode="raster")
env = gymnasium.make("hugsim_env/HUGSim-v0", cfg=cfg, output=os.path.join(out_dir, "tmp_env"))
name = seen.pop("name")
np.savez_compressed(os.path.join(out_dir, name + ".npz"), **seen)
print("dumped", name, seen["source"], len(seen["road_ab"]), len(seen["nonroad_ab"]), seen["roads"].shape)
