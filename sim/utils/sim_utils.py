import numpy as np
from scipy.spatial.transform import Rotation as SCR
import math
from scene.cameras import Camera
from sim.ilqr.lqr import plan2control
from omegaconf import OmegaConf

def rt2pose(r, t, degrees=False):
    pose = np.eye(4)
    pose[:3, :3] = SCR.from_euler('XYZ', r, degrees=degrees).as_matrix()
    pose[:3, 3] = t
    return pose

def pose2rt(pose, degrees=False):
    r = SCR.from_matrix(pose[:3, :3]).as_euler('XYZ', degrees=degrees)
    t = pose[:3, 3]
    return r, t
    
def load_camera_cfg(cfg):
    cam_params = {}
    cams = OmegaConf.to_container(cfg.cams, resolve=True)
    for cam_name, cam in cams.items():
        v2c = rt2pose(cam['extrinsics']['v2c_rot'], cam['extrinsics']['v2c_trans'], degrees=True)
        l2c = rt2pose(cam['extrinsics']['l2c_rot'], cam['extrinsics']['l2c_trans'], degrees=True)
        cam_intrin = cam['intrinsics']
        cam_intrin['fovx'] = cam_intrin['fovx'] / 180.0 * np.pi
        cam_intrin['fovy'] = cam_intrin['fovy'] / 180.0 * np.pi
        cam_params[cam_name] = {'intrinsic': cam_intrin, 'v2c': v2c, 'l2c': l2c}
        
    rect_mat = np.eye(4)
    if 'cam_rect' in cfg:
        rect_mat[:3, :3] = SCR.from_euler('XYZ', cfg.cam_rect.rot, degrees=True).as_matrix()
        rect_mat[:3, 3] = np.array(cfg.cam_rect.trans)
        
    return cam_params, OmegaConf.to_container(cfg.cam_align, resolve=True), rect_mat

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def create_cam(intrinsic, c2w):
    fovx, fovy = intrinsic['fovx'], intrinsic['fovy']
    h, w = intrinsic['H'], intrinsic['W']
    K = np.eye(4)
    K[0, 0], K[1, 1] = fov2focal(fovx, w), fov2focal(fovy, h)
    K[0, 2], K[1, 2] = intrinsic['cx'], intrinsic['cy']
    # No image: the renderer reads width and height only, and building a full-size zero image per
    # camera per step (then copying it to the GPU) was a quarter of a step's time.
    cam = Camera(K=K, c2w=c2w, width=w, height=h, image=None, image_name='')
    return cam

def traj2control(plan_traj, info, plan_dt=0.5, sim_dt=None, heading_order='fixed'):
    """
        The input plan trajectory is under lidar coordinates
        x to right, y to forward and z to upward

        plan_dt: time spacing between consecutive waypoints of plan_traj.
        sim_dt: interval the returned command is applied for by the simulator. The tracker has to
            run at this discretization, otherwise it optimizes a command for a horizon step that is
            longer (or shorter) than the one actually executed, and tracking lags systematically.
            When it differs from plan_dt the reference is resampled onto the simulator's timeline.
            Defaults to plan_dt.
        heading_order: 'fixed' (atan2(lateral, longitudinal), upstream PR #57) or 'upstream', the
            arguments as upstream HUGSIM main still has them (#57 is not merged there): atan2(longitudinal,
            lateral), pi/2 minus the heading. closed_loop.py --as_upstream asks for it.
    """
    if sim_dt is None:
        sim_dt = plan_dt

    # plan_traj columns are (right, forward); the tracker's state frame is (forward, right), which
    # is what the [1, 0] swap below is for. Row 0 is the current pose at the origin, row k the pose
    # at time k * plan_dt.
    ref_fr = np.zeros((plan_traj.shape[0] + 1, 2))  # columns: (forward, right)
    ref_fr[1:] = plan_traj[:, [1, 0]]
    ref_t = np.arange(ref_fr.shape[0]) * plan_dt

    if abs(sim_dt - plan_dt) > 1e-9:
        # Resample onto the simulator's timeline, keeping the same horizon length.
        n_steps = int(ref_t[-1] / sim_dt)
        sample_t = np.arange(n_steps + 1) * sim_dt
        ref_fr = np.stack(
            [np.interp(sample_t, ref_t, ref_fr[:, 0]), np.interp(sample_t, ref_t, ref_fr[:, 1])],
            axis=1,
        )

    plan_traj_stats = np.zeros((ref_fr.shape[0], 5))
    plan_traj_stats[:, :2] = ref_fr
    prev_fwd, prev_right = ref_fr[0]
    for i, (fwd, right) in enumerate(ref_fr[1:]):
        # Heading in the tracker frame: atan2(lateral, longitudinal).
        if heading_order == 'upstream':
            rot = np.arctan2(fwd - prev_fwd, right - prev_right)
        else:
            rot = np.arctan2(right - prev_right, fwd - prev_fwd)
        rot = np.where(rot > np.pi/2, rot - np.pi, rot)
        rot = np.where(rot < -np.pi/2, rot + np.pi, rot)
        plan_traj_stats[i+1, 2] = rot
        prev_fwd, prev_right = fwd, right
    curr_stat = np.array(
        [0.0, 0.0, 0.0, info['ego_velo'], info['ego_steer']]
    )
    acc, steer_rate = plan2control(plan_traj_stats, curr_stat, discretization_time=sim_dt)
    return acc, steer_rate

def dense_cam_poses(cam_poses, cmds):
    
    for i in range(5):
        dense_poses = []
        dense_cmds = []
        for i in range(cam_poses.shape[0]-1):
            cam1 = cam_poses[i]
            cam2 = cam_poses[i+1]
            dense_poses.append(cam1)
            dense_cmds.append(cmds[i])
            if np.linalg.norm(cam1[:3, 3]-cam2[:3, 3]) > 0.1:
                euler1 = SCR.from_matrix(cam1[:3, :3]).as_euler("XYZ")
                euler2 = SCR.from_matrix(cam2[:3, :3]).as_euler("XYZ")
                interp_euler = (euler1 + euler2) / 2
                interp_trans = (cam1[:3, 3] + cam2[:3, 3]) / 2
                interp_pose = np.eye(4)
                interp_pose[:3, :3] = SCR.from_euler("XYZ", interp_euler).as_matrix()
                interp_pose[:3, 3] = interp_trans
                dense_poses.append(interp_pose)
                dense_cmds.append(cmds[i])
        dense_poses.append(cam_poses[-1])
        # The final pose needs its command too, otherwise cmds ends up one shorter than
        # the poses and indexing it with the nearest-pose index raises IndexError as soon
        # as the ego reaches the end of the reference trajectory.
        dense_cmds.append(cmds[-1])
        dense_poses = np.stack(dense_poses)
        cam_poses = dense_poses
        cmds = dense_cmds
        
    return cam_poses, cmds

def traj_transform_to_global(traj, ego_box):
        """
        Transform trajectory from ego-centeric frame to global frame
        """
        ego_x, ego_y, _, _, _, _, ego_yaw = ego_box
        global_points = [
            (
                ego_x
                + px * math.cos(ego_yaw)
                - py * math.sin(ego_yaw),
                ego_y
                + px * math.sin(ego_yaw)
                + py * math.cos(ego_yaw),
            )
            for px, py in traj
        ]
        global_trajs = []
        for i in range(1, len(global_points)):
            x1, y1 = global_points[i - 1]
            x2, y2 = global_points[i]
            dx, dy = x2 - x1, y2 - y1
            # distance = math.sqrt(dx**2 + dy**2)
            yaw = math.atan2(dy, dx)
            global_trajs.append((x1, y1, yaw))
        return global_trajs
        