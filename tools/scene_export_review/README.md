# Reviewing and fixing a scene's exported road map

1. dump_evidence.py: build a scene's env once (GPU) with the sidecar bypassed and save the map
   source's inputs (drive, rendered road / sidewalk-terrain samples, road Gaussians) to an .npz.
   Takes the directory holding Pictura-dev's scripts/hugsim/capture_alignment.py for load_sim_cfg.
2. preview_fix.py: rebuild the map offline (CPU, seconds) from that .npz with the scene's
   sim/scene_export/road_fixes/<dataset>/<scene>.yaml, through the same GroundTrajectoryMapSource,
   and draw it with road_bev.py. With --nocarve and no fix it reproduces the env's build exactly;
   with the obstacle carve it matches to within a segment or two on the scenes checked.
3. road_bev.py: top-down colour and semantic orthophotos of the scene's Gaussians with the
   exported EDGE / LINE / LANE, parked cars and a 10 m grid, plus the raster-vs-semantic-road IoU.

Run from the repository root with PYTHONPATH=sim:. . The fix schema is in sim/scene_export/road_fixes.py.
