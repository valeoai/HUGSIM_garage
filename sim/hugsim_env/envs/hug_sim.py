import torch
import numpy as np
from copy import deepcopy
import gymnasium
from gymnasium import spaces
from copy import deepcopy
from sim.utils.sim_utils import create_cam, rt2pose, pose2rt, load_camera_cfg, dense_cam_poses
from scipy.spatial.transform import Rotation as SCR
from sim.utils.score_calculator import create_rectangle, bg_collision_det
import os
import pickle
from sim.utils.plan import planner, UnifiedMap
from sim.scene_export import (
    SCENE_EXPORT_HISTORY_LEN,
    PoseHistory,
    box_history_to_ego,
    boxes_to_cuboids,
    build_map_source,
    extract_static_vehicles,
    rig_pitch_from_track,
    roads_to_ego,
    route_to_ego,
    sidecar,
)
import sim.scene_export.map_source as scene_export_map_source
import sim.scene_export.static_agents as scene_export_static_agents
from omegaconf import OmegaConf
import math
from gaussian_renderer import GaussianModel
from scene.obj_model import ObjModel
from gaussian_renderer import render
import open3d as o3d


# Cityscapes ids in the scene models' semantic head. 0 road is the carriageway; 1 sidewalk
# and 9 terrain are what it ends against, and voting them against each other is what puts the
# exported boundary on the kerb.
SCENE_EXPORT_ROAD_CLASSES = (0,)
# Cap on the cars exported per step, nearest first; the driver keeps the nearest the
# checkpoint has slots for. See sim.scene_export.entities.MAX_AGENT_ENTITIES.
SCENE_EXPORT_MAX_AGENTS = 64
SCENE_EXPORT_KERB_CLASSES = (1, 9)

def fg_collision_det(ego_box, objs):
    ego_x, ego_y, _, ego_w, ego_l, ego_h, ego_yaw = ego_box
    ego_poly = create_rectangle(ego_x, ego_y, ego_w, ego_l, ego_yaw)
    for obs in objs:
        obs_x, obs_y, _, obs_w, obs_l, _, obs_yaw = obs
        obs_poly = create_rectangle(
            obs_x, obs_y, obs_w, obs_l, obs_yaw)
        if ego_poly.intersects(obs_poly):
            return True
    return False

class HUGSimEnv(gymnasium.Env):
    # Parameters of the road sweep, in one place because the sidecar's fingerprint has to
    # cover exactly what `_render_road_evidence` used.
    ROAD_SWEEP = {
        "cam_names": ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"),
        "num_viewpoints": 80,
        "pixel_stride": 3,
        "max_range_m": 45.0,
    }

    def __init__(self, cfg, output):
        super().__init__()
        
        plan_list = cfg.scenario.plan_list
        for control_param in plan_list:
            control_param[5] = os.path.join(cfg.base.realcar_path, control_param[5])

        # read ground infos
        with open(os.path.join(cfg.model_path, 'ground_param.pkl'), 'rb') as f:
            #numpy.ndarray, float, list
            cam_poses, cam_heights, commands = pickle.load(f)
            cam_poses, commands = dense_cam_poses(cam_poses, commands)
            self.ground_model = (cam_poses, cam_heights, commands)

        if cfg.scenario.load_HD_map:
            unified_map = UnifiedMap(cfg.base.HD_map.path, cfg.base.HD_map.version, cfg.scenario.scene_name)
        else:
            unified_map = None
        
        self.kinematic = OmegaConf.to_container(cfg.kinematic)
        self.kinematic['min_steer'] = -math.radians(cfg.kinematic.min_steer)
        self.kinematic['max_steer'] = math.radians(cfg.kinematic.max_steer)
        self.kinematic['start_vr']= np.array(cfg.scenario.start_euler) / 180 * np.pi
        self.kinematic['start_vab'] = np.array(cfg.scenario.start_ab)
        self.kinematic['start_velo'] = cfg.scenario.start_velo
        self.kinematic['start_steer'] = cfg.scenario.start_steer

        self.gaussians = GaussianModel(cfg.model.sh_degree, affine=cfg.affine)

        """
        plan_list: a, b, height, yaw, v, model_path, controller, params
        Yaw is based on ego car's orientation. 0 means same direction as ego. 
        Right is positive and left is negative.
        """
        self.planner = planner(plan_list, scene_path=cfg.model_path, unified_map=unified_map, ground=self.ground_model, dt=cfg.kinematic.dt)
        
        (model_params, iteration) = torch.load(os.path.join(cfg.model_path, "scene.pth"), weights_only=False)
        self.gaussians.restore(model_params, None)
        
        dynamic_gaussians = {}
        for plan_id in self.planner.ckpts.keys():
            dynamic_gaussians[plan_id] = ObjModel(cfg.model.sh_degree, feat_mutable=False)
            (model_params, iteration) = torch.load(self.planner.ckpts[plan_id], weights_only=False)
            model_params = list(model_params)
            dynamic_gaussians[plan_id].restore(model_params, None)
            
        semantic_idx = torch.argmax(self.gaussians.get_full_3D_features, dim=-1, keepdim=True)
        ground_xyz = self.gaussians.get_full_xyz[(semantic_idx == 0)[:, 0]].detach().cpu().numpy()
        scene_xyz = self.gaussians.get_full_xyz[((semantic_idx > 1) & (semantic_idx != 10))[:, 0]].detach().cpu().numpy()
        ground_pcd = o3d.geometry.PointCloud()
        ground_pcd.points = o3d.utility.Vector3dVector(ground_xyz.astype(float))
        o3d.io.write_point_cloud(os.path.join(output, 'ground.ply'), ground_pcd)
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(scene_xyz.astype(float))
        o3d.io.write_point_cloud(os.path.join(output, 'scene.ply'), scene_pcd)

        if cfg.scenario.load_HD_map:
            self.planner.update_agent_route()

        self.cam_params, cam_align, self.cam_rect = load_camera_cfg(cfg.camera)
        
        self.ego_verts = np.array([[0.5, 0, 0.5], [0.5, 0, -0.5], [0.5, 1.0,  0.5], [0.5, 1.0, -0.5],
                    [-0.5, 0, -0.5], [-0.5, 0, 0.5], [-0.5, 1.0, -0.5], [-0.5, 1.0, 0.5]])
        self.whl = np.array([1.6, 1.5, 3.0])
        self.ego_verts *= self.whl
        self.data_type = cfg.data_type

        self.action_space = spaces.Dict(
            {
                "steer_rate": spaces.Box(self.kinematic['min_steer'], self.kinematic['max_steer'], dtype=float),
                "acc": spaces.Box(self.kinematic['min_acc'], self.kinematic['max_acc'], dtype=float)
            }
        )
        self.observation_space = spaces.Dict(
            {
                'rgb': spaces.Dict({
                    cam_name: spaces.Box(
                        low=0, high=255, 
                        shape=(params['intrinsic']['H'], params['intrinsic']['W'], 3), dtype=np.uint8
                    ) for cam_name, params in self.cam_params.items()
                }),
                'semantic': spaces.Dict({
                    cam_name: spaces.Box(
                        low=0, high=50, 
                        shape=(params['intrinsic']['H'], params['intrinsic']['W']), dtype=np.uint8
                    ) for cam_name, params in self.cam_params.items()
                }),
                'depth': spaces.Dict({
                    cam_name: spaces.Box(
                        low=0, high=1000, 
                        shape=(params['intrinsic']['H'], params['intrinsic']['W']), dtype=np.float32
                    ) for cam_name, params in self.cam_params.items()
                }),
            }
        )
        self.fric = self.kinematic['fric']

        self.start_vr = self.kinematic['start_vr']
        self.start_vab = self.kinematic['start_vab']
        self.start_velo = self.kinematic['start_velo']
        self.vr = deepcopy(self.kinematic['start_vr'])
        self.vab = deepcopy(self.kinematic['start_vab'])
        self.velo = deepcopy(self.kinematic['start_velo'])
        self.steer = deepcopy(self.kinematic['start_steer'])
        self.dt = self.kinematic['dt']

        bg_color = [1, 1, 1] if cfg.model.white_background else [0, 0, 0]
        self.render_fn = render
        self.render_kwargs = {
            "pc": self.gaussians,
            "bg_color": torch.tensor(bg_color, dtype=torch.float32, device="cuda"),
            "dynamic_gaussians": dynamic_gaussians,
            "unicycles": {} # dummy input, unicycle planner is used for unicycle models
        }
        gaussians = self.gaussians
        semantic_idx = torch.argmax(gaussians.get_3D_features, dim=-1, keepdim=True)
        opacities = gaussians.get_opacity[:, 0]
        mask = ((semantic_idx > 1) & (semantic_idx != 10))[:, 0] & (opacities > 0.8)
        self.points = gaussians.get_xyz[mask]

        # A policy may read an abstract render of the scene rather than the Gaussian one, so the
        # map has to be exported as geometry. Built here, at the end of construction, because
        # it renders the scene to find the road and so needs the cameras and render_kwargs.
        # Off by default -- every other AD side ignores it and it is not free to build.
        self.scene_export_map = None
        self.scene_export_static_agents = np.zeros((0, 7))
        # An export cap, not the policy's block width; see sim.scene_export.entities.
        self.scene_export_max_agents = int(cfg.base.get('scene_export_max_agents', SCENE_EXPORT_MAX_AGENTS))
        self.scene_export_rig_pitch = 0.0
        self._scene_export_history = PoseHistory(SCENE_EXPORT_HISTORY_LEN)
        if cfg.base.get('scene_export_obs', False):
            # KITTI-360's world frame is its down-pitched camera's, so a level street climbs
            # ~6 deg in it; the export is rotated back onto the direction of travel.
            self.scene_export_rig_pitch = rig_pitch_from_track(cam_poses)
            # The HD-map export (TrajdataMapSource) does not line up with the reconstruction:
            # rotated ~128 deg on scene-0383-medium-01, off the markings elsewhere. The 11 public
            # scenarios that load it scored 0.20 against 0.54 for nuScenes as a whole, most of them
            # stalling at the start. So the policy's map comes from the drive for every scenario
            # unless `scene_export_hd_map` asks otherwise; HUGSIM's planner keeps the HD map for its
            # actors either way.
            scene_export_hd_map = bool(cfg.scenario.load_HD_map) and bool(cfg.base.get('scene_export_hd_map', False))
            # The recorded drive, in the sim frame and on the ground rather than at camera
            # height, is the route the policy is given as its goal -- and the same track
            # route completion is scored against, so the goal and the score agree.
            track_ab = np.stack([cam_poses[:, 0, 3], cam_poses[:, 2, 3]], axis=1)
            self.scene_export_track = np.stack(
                [track_ab[:, 1], -track_ab[:, 0], self.sim_ground_height(track_ab[:, 0], track_ab[:, 1])], axis=1
            )
            # The map and the parked cars describe the scene, not the episode, so they are
            # derived once and cached beside it. The fingerprint covers every constant that
            # steers them, so retuning any of these invalidates the sidecar rather than
            # quietly serving what the previous tuning produced.
            digest = sidecar.fingerprint(
                cfg.scenario.scene_name,
                scene_export_hd_map,
                road_classes=SCENE_EXPORT_ROAD_CLASSES,
                kerb_classes=SCENE_EXPORT_KERB_CLASSES,
                sweep=self.ROAD_SWEEP,
                map_source=scene_export_map_source.tunables(),
                static_agents=scene_export_static_agents.tunables(),
            )
            cached = sidecar.load(cfg.model_path, cfg.scenario.scene_name, digest)
            if cached is not None:
                self.scene_export_map = cached["roads"]
                self.scene_export_static_agents = cached["static_agents"]
                how = "model-refined headings" if cached.get("refined") else "geometric headings"
                print(f'[scene_export] sidecar hit ({how}): {cached["path"]}')
            else:
                road_ab, nonroad_ab = self._render_road_evidence()
                map_source = build_map_source(
                    cfg, unified_map if scene_export_hd_map else None, cam_poses,
                    ground_xyz=ground_xyz, road_ab=road_ab, nonroad_ab=nonroad_ab,
                )
                self.scene_export_map = map_source.build(self.sim_ground_height)
                # Vehicles that are part of the static reconstruction rather than the
                # scenario: parked cars, and anything standing still when the scene was
                # recorded. The Gaussian render shows them and `self.points` already makes
                # them collidable, so leaving them out asks the policy to avoid what it
                # cannot see. Note the explicit get_full_3D_features: a *different*
                # `semantic_idx`, over the visible-only features, is bound just above for the
                # collision point set.
                self.scene_export_static_agents = extract_static_vehicles(
                    self.gaussians.get_full_xyz.detach().cpu().numpy(),
                    torch.argmax(self.gaussians.get_full_3D_features, dim=-1).detach().cpu().numpy(),
                    self.gaussians.get_full_opacity[:, 0].detach().cpu().numpy(),
                    self.sim_ground_height,
                    track_sim=self.scene_export_track,
                )
                written = sidecar.save(
                    cfg.model_path, cfg.scenario.scene_name, digest,
                    self.scene_export_map, self.scene_export_static_agents, self.scene_export_track,
                )
                if written:
                    print(f'[scene_export] sidecar written: {written}')
            print(
                f'[scene_export] {self.scene_export_map.shape[0]} road segments, '
                f'{self.scene_export_static_agents.shape[0]} static vehicles, '
                f'rig pitch {np.degrees(self.scene_export_rig_pitch):+.2f} deg'
            )

        self.last_accel = 0
        self.last_steer_rate = 0

        self.timestamp = 0
    
    def _render_road_evidence(self, **overrides):
        """Where the road is, according to the renderer, in planner coordinates.

        The map used to be traced from the positions of road-labelled Gaussians. Those are
        the primitives the surface is *made of*, which is a different thing from the surface:
        they thin out under a parked car, they carry a halo past the kerb, and nothing in
        them says where the pavement starts. The rendered image does say, because the same
        model carries a semantic head, and that image is what every other AD side is judged
        on -- so taking the road from it is taking it from the thing being simulated.

        The scene is swept from viewpoints along the recorded drive, which is the route the
        ego will be scored on, so the map covers where it will go. Road pixels and
        sidewalk/terrain pixels are back-projected with the rendered depth and returned
        separately: the second set is what puts the boundary on the kerb rather than at the
        end of the evidence.

        Args:
            **overrides: any of :attr:`ROAD_SWEEP` -- ``cam_names`` (the forward three see
                the carriageway; the rear ones are zeroed outright on waymo and kitti360),
                ``num_viewpoints`` sampled along the recorded drive, ``pixel_stride`` for the
                back-projected pixels, and ``max_range_m`` beyond which rendered depth gets
                unreliable, out where the surface is a few splats seen edge-on.

        Returns:
            ``(road_ab, nonroad_ab)``, each ``(N, 2)`` planner ``(a, b)``.
        """
        sweep = dict(self.ROAD_SWEEP, **overrides)
        cam_names = sweep["cam_names"]
        pixel_stride = int(sweep["pixel_stride"])
        max_range_m = float(sweep["max_range_m"])
        cam_poses, _, _ = self.ground_model
        stride = max(1, cam_poses.shape[0] // max(1, int(sweep["num_viewpoints"])))
        v2front = self.cam_params['CAM_FRONT']['v2c']
        # No actors: the scenario's cars are not part of the road, and rendering them here
        # would punch their footprints out of it.
        kwargs = dict(self.render_kwargs)
        kwargs['planning'] = [{}, {}]

        road, nonroad = [], []
        for pose in cam_poses[::stride]:
            for cam_name in cam_names:
                params = self.cam_params[cam_name]
                c2front = v2front @ np.linalg.inv(params['v2c']) @ self.cam_rect
                viewpoint = create_cam(params['intrinsic'], pose @ c2front)
                with torch.no_grad():
                    pkg = self.render_fn(viewpoint=viewpoint, prev_viewpoint=None, **kwargs)
                    semantic = torch.argmax(pkg['feats'], dim=0)[::pixel_stride, ::pixel_stride]
                    depth = pkg['depth'][0][::pixel_stride, ::pixel_stride]
                    for classes, sink in ((SCENE_EXPORT_ROAD_CLASSES, road),
                                          (SCENE_EXPORT_KERB_CLASSES, nonroad)):
                        hit = torch.zeros_like(semantic, dtype=torch.bool)
                        for c in classes:
                            hit |= semantic == c
                        hit &= (depth > 1.0) & (depth < max_range_m)
                        if not bool(hit.any()):
                            continue
                        sink.append(self._backproject(hit, depth, viewpoint, pixel_stride))
        empty = np.zeros((0, 2))
        return (np.vstack(road) if road else empty, np.vstack(nonroad) if nonroad else empty)

    @staticmethod
    def _backproject(hit, depth, viewpoint, pixel_stride):
        """``(H, W)`` mask + depth -> ``(N, 2)`` planner ``(a, b) = (X_world, Z_world)``."""
        rows, cols = torch.nonzero(hit, as_tuple=True)
        z = depth[rows, cols]
        K = viewpoint.K
        u = cols.float() * pixel_stride
        v = rows.float() * pixel_stride
        cam_pts = torch.stack(
            [(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], dim=1
        )
        world = cam_pts @ viewpoint.c2w[:3, :3].T + viewpoint.c2w[:3, 3]
        return torch.stack([world[:, 0], world[:, 2]], dim=1).cpu().numpy()

    def ground_height(self, u, v):
        cam_poses, cam_height, _ = self.ground_model
        cam_dist = np.sqrt(
            (cam_poses[:, 0, 3] - u)**2 + (cam_poses[:, 2, 3] - v)**2
        )
        nearest_cam_idx = np.argmin(cam_dist, axis=0)
        nearest_c2w = cam_poses[nearest_cam_idx]

        nearest_w2c = np.linalg.inv(nearest_c2w)
        uhv_local = nearest_w2c[:3, :3] @ np.array([u, 0, v]) + nearest_w2c[:3, 3]
        uhv_local[1] = 0
        uhv_world = nearest_c2w[:3, :3] @ uhv_local + nearest_c2w[:3, 3]
        
        return uhv_world[1]
    
    def sim_ground_height(self, u, v):
        """The drivable surface under ``(u, v)``, in the sim frame (z up).

        Vectorized over arrays, because the map exporter needs a height under every one of a
        few thousand map vertices and the scalar path would do one 4x4 inverse at a time.

        This differs from :meth:`ground_height` by the camera height, and deliberately.
        :meth:`ground_height` projects onto the plane the recorded camera swept, which is
        what places the ego: ``vt`` is the camera rig origin, so it belongs on that plane.
        Road geometry belongs on the road, one camera height lower -- the same correction
        ``planner.ground_height`` applies before it seats an actor.
        """
        u = np.atleast_1d(np.asarray(u, dtype=float))
        v = np.atleast_1d(np.asarray(v, dtype=float))
        cam_poses, cam_height, _ = self.ground_model

        out = np.empty(u.shape[0])
        # Chunked: the distance matrix is len(points) x len(cam_poses), and cam_poses is the
        # densified track, which runs to thousands of entries on a long scene.
        for start in range(0, u.shape[0], 4096):
            stop = min(start + 4096, u.shape[0])
            uu, vv = u[start:stop], v[start:stop]
            dist = (cam_poses[None, :, 0, 3] - uu[:, None]) ** 2 + (cam_poses[None, :, 2, 3] - vv[:, None]) ** 2
            c2w = cam_poses[np.argmin(dist, axis=1)]
            w2c = np.linalg.inv(c2w)
            uhv = np.stack([uu, np.zeros_like(uu), vv], axis=1)
            local = np.einsum('nij,nj->ni', w2c[:, :3, :3], uhv) + w2c[:, :3, 3]
            local[:, 1] = 0.0
            world = np.einsum('nij,nj->ni', c2w[:, :3, :3], local) + c2w[:, :3, 3]
            # World Y points down, so the road is at +cam_height from the camera plane;
            # negating puts it back in the sim's upward z.
            out[start:stop] = -(world[:, 1] + cam_height)
        return out

    @property
    def route_completion(self):
        cam_poses, _, _ = self.ground_model
        cam_dist = np.sqrt(
            (cam_poses[:, 0, 3] - self.vab[0])**2 + (cam_poses[:, 2, 3] - self.vab[1])**2
        )
        nearest_cam_idx = np.argmin(cam_dist, axis=0)
        return (nearest_cam_idx + 1) / (cam_poses.shape[0] * 0.9), cam_dist[nearest_cam_idx]
        

    @property
    def vt(self):
        vt = np.zeros(3)
        vt[[0, 2]] = self.vab
        vt[1] = self.ground_height(self.vab[0], self.vab[1])
        return vt
    
    @property
    def ego(self):
        return rt2pose(self.vr, self.vt)
    
    @property
    def ego_state(self):
        return torch.tensor([self.vab[0], self.vab[1], self.vr[1], self.velo])
    
    @property
    def ego_box(self):
        return [self.vt[2], -self.vt[0], -self.vt[1], self.whl[0], self.whl[2], self.whl[1], -self.vr[1]]

    @property
    def objs_list(self):
        return list(self._objs_by_id().values())

    def _objs_by_id(self):
        """``{obj_id: [x, y, z, w, l, h, yaw]}`` of the dynamic actors, in the planner's stable order."""
        obj_boxes = {}
        objs = self.render_kwargs['planning'][0]
        for obj_id, obj_b2w in objs.items():
            yaw = SCR.from_matrix(obj_b2w[:3, :3].detach().cpu().numpy()).as_euler('YXZ')[0]
            # X, Y, Z in IMU, w, l, h
            wlh = self.planner.wlhs[obj_id]
            obj_boxes[obj_id] = [obj_b2w[2, 3].item(), -obj_b2w[0, 3].item(), -obj_b2w[1, 3].item(), wlh[0], wlh[1], wlh[2], -yaw-0.5*np.pi]
        return obj_boxes

    def _get_obs(self):
        rgbs, semantics, depths = {}, {}, {}
        v2front = self.cam_params['CAM_FRONT']["v2c"]
        for cam_name, params in self.cam_params.items():
            intrinsic, v2c = params['intrinsic'], params['v2c']
            c2front = v2front @ np.linalg.inv(v2c) @ self.cam_rect
            c2w = self.ego @ c2front
            viewpoint = create_cam(intrinsic, c2w)
            with torch.no_grad():
                render_pkg = self.render_fn(viewpoint=viewpoint, prev_viewpoint=None, **self.render_kwargs)
            rgb = (torch.permute(render_pkg['render'].clamp(0, 1), (1,2,0)).detach().cpu().numpy() * 255).astype(np.uint8)
            smt = torch.argmax(render_pkg['feats'], dim=0).detach().cpu().numpy().astype(np.uint8)
            depth = render_pkg['depth'][0].detach().cpu().numpy()
            if (self.data_type == 'waymo' or self.data_type == 'kitti360') and 'BACK' in cam_name:
                rgbs[cam_name] = np.zeros_like(rgb)
                semantics[cam_name] = np.zeros_like(smt)
                depths[cam_name] = np.zeros_like(depth)
            else:
                rgbs[cam_name] = rgb
                semantics[cam_name] = smt
                depths[cam_name] = depth

        return {
                'rgb': rgbs, 
                'semantic': semantics,
                'depth': depths,
                }
    
    def _get_info(self):
        wego_r, wego_t = pose2rt(self.ego)
        cam_poses, _, commands = self.ground_model
        dist = np.sum((cam_poses[:, :3, 3] - self.vt) ** 2, axis=-1)
        nearest_cam_idx = np.argmin(dist)
        command = commands[nearest_cam_idx]
        info = {
            'ego_pos'  : wego_t.tolist(),
            'ego_rot'  : wego_r.tolist(),
            'ego_velo' : self.velo,
            'ego_steer': self.steer,
            'accelerate': self.last_accel,
            'steer_rate': self.last_steer_rate,
            'timestamp': self.timestamp,
            # The simulator step. An AD side integrating its own dynamics needs it on the
            # first step too, where there is no previous timestamp to difference, and must
            # not assume the shipped 0.25 s: dt is configurable and a wrong first step
            # displaces the ego before the episode has begun.
            'dt': self.dt,
            'command': command,
            'ego_box': self.ego_box,
            'obj_boxes': self.objs_list,
            'cam_params': self.cam_params,
            # 'ego_verts': verts,
        }
        if self.scene_export_map is not None:
            # Culled and transformed here rather than on the AD side: the map lives in this
            # process, and shipping the whole scene down the pipe every step would cost more
            # than the render it feeds.
            # The exported ego frame puts z = 0 on the road, but ego_box's z is the camera rig
            # origin, one camera height above it. Everything exported here is measured off
            # the road instead, so that the map, the actors and the goals share the datum
            # the renderer assumes -- otherwise the actors sit a camera height into it.
            ego_box = list(self.ego_box)
            ego_box[2] = float(self.sim_ground_height(self.vab[0], self.vab[1])[0])
            # The scenario's actors and the reconstruction's parked cars are one population
            # as far as the policy is concerned, so they are merged before the nearest-first
            # cull rather than each getting their own budget. Merged only here: objs_list
            # feeds fg_collision_det and the episode record, and the static vehicles are
            # already covered by bg_collision_det, so adding them there would double-count.
            actors_by_id = self._objs_by_id()
            actors = list(actors_by_id.values())
            # Speeds of the scenario's actors, from their motion since the last step (the planner
            # does not expose them). A vectorized policy reads them; the rendered one does not.
            # Actors are listed in a stable order, so rows match while the count does; a step
            # where an actor appears or leaves reports zero. Parked cars stand still.
            speeds = np.zeros(len(actors))
            prev = getattr(self, '_scene_export_prev_objs', None)
            if prev is not None and len(prev) == len(actors) and len(actors) and self.dt > 0:
                speeds = np.linalg.norm(np.asarray(actors)[:, :2] - np.asarray(prev)[:, :2], axis=1) / self.dt
            self._scene_export_prev_objs = [list(o) for o in actors]
            pitch = self.scene_export_rig_pitch
            # Pose history of every actor, computed BEFORE the cull so an actor that was far
            # away a moment ago and is near now keeps its past. A repeated call at the same
            # timestamp replaces the snapshot (PoseHistory.push), it does not add one.
            self._scene_export_history.push(self.timestamp, actors_by_id)
            history, history_valid, snapshot_times = self._scene_export_history.export(
                list(actors_by_id.keys()), actors, ego_box, pitch=pitch
            )
            n_snap = len(snapshot_times)
            if len(self.scene_export_static_agents):
                static = self.scene_export_static_agents
                actors.extend(static.tolist())
                speeds = np.concatenate([speeds, np.zeros(len(static))])
                # Parked cars never move: constant pose, valid at every snapshot.
                static_hist = box_history_to_ego(np.repeat(static[:, None, :], n_snap, axis=1), ego_box, pitch=pitch)
                history = np.concatenate([history, static_hist], axis=0)
                history_valid = np.concatenate([history_valid, np.ones((len(static), n_snap), dtype=bool)], axis=0)
            # One 2-D extra [speed | history N*3 | valid N] so all of it shares the cull and sort.
            extra = np.concatenate(
                [speeds[:, None], history.reshape(len(actors), n_snap * 3), history_valid.astype(np.float64)], axis=1
            )
            cuboids, carried = boxes_to_cuboids(
                actors, ego_box, max_entities=self.scene_export_max_agents, pitch=pitch, extra=extra
            )
            cuboid_speed = carried[:, 0]
            cuboid_history = carried[:, 1 : 1 + n_snap * 3].reshape(-1, n_snap, 3)
            cuboid_history_valid = carried[:, 1 + n_snap * 3 :] > 0.5
            info['pictura'] = {
                'roads': roads_to_ego(self.scene_export_map, ego_box, pitch=pitch),
                'cuboids': cuboids,
                'cuboid_speed': cuboid_speed,
                'cuboid_history': cuboid_history,
                'cuboid_history_valid': cuboid_history_valid,
                'snapshot_times': snapshot_times,
                'route': route_to_ego(self.scene_export_track, ego_box, pitch=pitch),
            }
        return info
    
    def reset(self, seed=None, options=None):
        self._scene_export_prev_objs = None
        self._scene_export_history.reset()
        self.vr = deepcopy(self.start_vr)
        self.vab = deepcopy(self.start_vab)
        self.velo = deepcopy(self.start_velo)
        self.timestamp = 0

        if self.planner is not None:
            self.render_kwargs['planning'] = self.planner.plan_traj(self.timestamp, self.ego_state)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        self.timestamp += self.dt
        if self.planner is not None:
            self.render_kwargs['planning'] = self.planner.plan_traj(self.timestamp, self.ego_state)
        steer_rate, acc = action['steer_rate'], action['acc']
        self.last_steer_rate, self.last_accel = steer_rate, acc
        if 'pose_delta' in action:
            # The AD side moved the ego itself (its own dynamics, see closed_loop.py):
            # apply its displacement instead of integrating a command. `pose_delta` is
            # (forward, left, yaw) in the ego frame at the start of the step, sim-frame
            # convention (x forward, y left, yaw counter-clockwise); `theta` here is the
            # planner-frame heading, which is minus the sim yaw.
            forward, left, dyaw = (float(x) for x in action['pose_delta'])
            theta = self.vr[1]
            heading = -theta
            dx = forward * np.cos(heading) - left * np.sin(heading)
            dy = forward * np.sin(heading) + left * np.cos(heading)
            self.vab[1] = self.vab[1] + dx
            self.vab[0] = self.vab[0] - dy
            self.vr[1] = theta - dyaw
            self.last_steer_rate = (float(action['steer']) - self.steer) / self.dt
            self.velo = float(action['velo'])
            self.steer = float(action['steer'])
        else:
            L = self.kinematic['Lr'] + self.kinematic['Lf']
            self.velo += acc * self.dt
            self.steer += steer_rate * self.dt
            theta = self.vr[1]
            # print(theta / np.pi * 180, self.steer / np.pi * 180)
            self.vab[0] = self.vab[0] + self.velo * np.sin(theta) * self.dt
            self.vab[1] = self.vab[1] + self.velo * np.cos(theta) * self.dt
            self.vr[1] = theta + self.velo * np.tan(self.steer) / L * self.dt

        terminated = False
        reward = 0
        verts = (self.ego[:3, :3] @ self.ego_verts.T).T + self.ego[:3, 3]
        verts = torch.from_numpy(verts.astype(np.float32)).cuda()
        
        bg_collision = bg_collision_det(self.points, verts)
        if bg_collision:
            terminated = True
            print('Collision with background')
            reward = -100

        fg_collision = fg_collision_det(self.ego_box, self.objs_list)
        if fg_collision:
            terminated = True
            print('Collision with foreground')
            reward = -100

        rc, dist = self.route_completion
        if dist > 10:
            terminated=True
            print('Far from preset trajectory')
            reward = -50
            
        if rc >= 1:
            terminated = True
            print('Complete')
            reward = 1000

        observation = self._get_obs()
        info = self._get_info()
        info['rc'] = rc
        info['collision'] = bg_collision or fg_collision
        
        return observation, reward, terminated, False, info