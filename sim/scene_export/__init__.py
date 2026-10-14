"""Adapter that exposes HUGSIM's state as an abstract scene description.

A consuming policy may be trained on flat-shaded perspective renders of simulation
primitives -- cuboids for agents, typed line segments for the map -- not on photorealistic
images. Feeding it HUGSIM's Gaussian-splat render would put it straight into the
representation gap its training regime exists to close, so instead HUGSIM exports the
abstract state and the AD side rasterizes it with the same kernel used in training.

Two pieces live here:

* :mod:`sim.scene_export.map_source` turns a scene's map into typed world-frame segments, from
  the HD map where there is one and from the recorded drive plus the ground Gaussians where
  there is not.
* :mod:`sim.scene_export.entities` culls those segments to the ego's visibility box and packs
  them, and the dynamic boxes, into the ego-frame layout the renderer consumes.
* :mod:`sim.scene_export.static_agents` recovers the vehicles baked into the static Gaussian
  model -- every parked car -- which the Gaussian render shows and the abstract export
  otherwise omitted, even though the ego can already collide with them.
* :mod:`sim.scene_export.history` keeps the last few snapshots of every actor's box, and
  exports them in the current ego frame for a temporal policy.
* :mod:`sim.scene_export.sidecar` caches both of those beside the scene. Neither the road nor the
  parked cars move, so they are derived once per scene rather than once per episode.

Everything is expressed in the frame ``ego_box`` / ``obj_boxes`` already use (x forward,
y left, z up, yaw counter-clockwise), a common world convention.
"""

from sim.scene_export.entities import box_history_to_ego, boxes_to_cuboids, rig_pitch_from_track, roads_to_ego, route_to_ego
from sim.scene_export import sidecar
from sim.scene_export.history import SCENE_EXPORT_HISTORY_LEN, PoseHistory
from sim.scene_export.static_agents import VEHICLE_CLASSES, extract_static_vehicles
from sim.scene_export.map_source import (
    ROAD_CLASS_CROSSWALK,
    ROAD_CLASS_EDGE,
    ROAD_CLASS_LANE,
    ROAD_CLASS_LINE,
    ROAD_CLASS_SPEED_BUMP,
    GroundTrajectoryMapSource,
    TrajdataMapSource,
    build_map_source,
)

__all__ = [
    "SCENE_EXPORT_HISTORY_LEN",
    "VEHICLE_CLASSES",
    "ROAD_CLASS_CROSSWALK",
    "ROAD_CLASS_EDGE",
    "ROAD_CLASS_LANE",
    "ROAD_CLASS_LINE",
    "ROAD_CLASS_SPEED_BUMP",
    "GroundTrajectoryMapSource",
    "PoseHistory",
    "TrajdataMapSource",
    "box_history_to_ego",
    "boxes_to_cuboids",
    "build_map_source",
    "rig_pitch_from_track",
    "sidecar",
    "extract_static_vehicles",
    "roads_to_ego",
    "route_to_ego",
]
