import sys
import os
import re
import shutil
sys.path.append(os.getcwd())

import gymnasium
import hugsim_env
from hugsim_env.envs.hug_sim import OBS_MODES, RASTER_RIGS, RASTER_VIEWS, resolve_raster
from argparse import ArgumentParser
from sim.utils.sim_utils import traj2control, traj_transform_to_global
import pickle
import json
import pickle
from sim.utils.launch_ad import launch, check_alive
from omegaconf import OmegaConf
import open3d as o3d
from sim.utils.score_calculator import hugsim_evaluate
import numpy as np
from moviepy import ImageSequenceClip

# Time spacing of the waypoints the AD side returns. The tracker needs it to place the plan on a
# timeline, and the scorer needs it to differentiate the plan into speeds/accelerations.
PLAN_TIMESTEP = 0.5
#: Longest episode, in simulated seconds. 400 steps at the shipped dt = 0.25 s.
EPISODE_SECONDS = 100.0
#: Frame rate of a sparse video, in simulated time. The abstract modes render no camera to drive
#: from, so each video frame is a Gaussian render made for the video alone.
SPARSE_VIDEO_HZ = 2.0

def to_video(observations, output_path, fps):
    frames = []
    for obs in observations:
        row1 = np.concatenate([obs['CAM_FRONT_LEFT'], obs['CAM_FRONT'], obs['CAM_FRONT_RIGHT']], axis=1)
        row2 = np.concatenate([obs['CAM_BACK_RIGHT'], obs['CAM_BACK'], obs['CAM_BACK_LEFT']], axis=1)
        frame = np.concatenate([row1, row2], axis=0)
        frames.append(frame)
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path)


def video_stride(video, dt):
    """Steps between two video frames: every step for 'full', SPARSE_VIDEO_HZ for 'sparse'.

    Counted in steps of simulated time, so the video plays in real time whatever dt is.
    """
    return 1 if video == 'full' else max(1, round(1.0 / (SPARSE_VIDEO_HZ * dt)))


def create_gym_env(cfg, output, skip_native_eval=False, video='full'):

    env = gymnasium.make('hugsim_env/HUGSim-v0', cfg=cfg, output=output)

    observations_save, infos_save = [], []
    obs, info = env.reset()
    done = False
    cnt = 0
    save_data = {'type': 'closeloop', 'frames': []}

    obs_pipe = os.path.join(output, 'obs_pipe')
    plan_pipe = os.path.join(output, 'plan_pipe')
    if not os.path.exists(obs_pipe):
        os.mkfifo(obs_pipe)
    if not os.path.exists(plan_pipe):
        os.mkfifo(plan_pipe)
    print('Ready for simulation')

    stride = video_stride(video, cfg.kinematic.dt)
    obs, info = None, None
    while not done:

        if obs is None or info is None:
            obs, info = env.reset()
        if video != 'off' and cnt % stride == 0:
            # The rgb mode has just rendered the cameras for the agent; the abstract modes
            # render nothing per step, so a frame there is rendered for the video alone.
            observations_save.append(
                obs['rgb'] if cfg.base.obs_mode == 'rgb' else env.unwrapped.render_cameras()['rgb']
            )
        infos_save.append(info)

        print(
            'ego x={:.3f}, y={:.3f}, speed={:.3f}, steer={:.3f}, lon_accel={:.3f}, lat_accel={:.3f}'.format(
                info["ego_box"][0],
                info["ego_box"][1],
                info["ego_velo"],
                info["ego_steer"],
                info["accelerate"],
                info["ego_velo"] * info["ego_velo"] * np.tan(-info['ego_steer']) / 2.7
            )
        )

        with open(obs_pipe, "wb") as pipe:
            pipe.write(pickle.dumps((obs, info)))
        with open(plan_pipe, "rb") as pipe:
            plan_traj = pickle.loads(pipe.read())
        # An AD side may also move the ego itself: a dict carrying the plan (still what is
        # scored) plus the displacement to apply this step. an AD side does this to drive with its
        # own vehicle model instead of through the iLQR tracker. A bare array is a plan to track.
        override = None
        if isinstance(plan_traj, dict):
            override = plan_traj
            plan_traj = override['plan']

        if plan_traj is not None:
            # The plan is expressed in the ego frame at the *current* pose, so it has to be
            # anchored to that pose. Building the frame before env.step() keeps the recorded
            # ego_box, obj_boxes and time_stamp in the same frame of reference as the plan.
            imu_plan_traj = plan_traj[:, [1, 0]]
            imu_plan_traj[:, 1] *= -1
            global_traj = traj_transform_to_global(imu_plan_traj, info['ego_box'])
            frame = {
                'time_stamp': info['timestamp'],
                'is_key_frame': True,
                'ego_box': info['ego_box'],
                'obj_boxes': info['obj_boxes'],
                'obj_names': ['car' for _ in info['obj_boxes']],
                'planned_traj': {
                    'traj': global_traj,
                    'timestep': PLAN_TIMESTEP
                },
            }

            # The command is held for exactly one simulator step, so the tracker must be
            # discretized at cfg.kinematic.dt rather than at the AD's waypoint spacing.
            if override is not None:
                action = {
                    'acc': override['acc'],
                    'steer_rate': 0.0,
                    'pose_delta': override['pose_delta'],
                    'velo': override['velo'],
                    'steer': override['steer'],
                }
            else:
                acc, steer_rate = traj2control(
                    plan_traj, info, plan_dt=PLAN_TIMESTEP, sim_dt=cfg.kinematic.dt
                )
                # print(plan_traj, acc, steer_rate)

                action = {'acc': acc, 'steer_rate': steer_rate}
            obs, reward, terminated, truncated, info = env.step(action)
            cnt += 1
            # The cap is a duration, not a step count: at the shipped dt = 0.25 s, 400 steps
            # was 100 s, and a step-count cap silently shortens the episode when dt changes
            # (at dt = 0.1 s the same 400 steps is 40 s, so a route needing 60 s can never be
            # completed and route completion measures the cap rather than the agent).
            done = terminated or truncated or cnt * cfg.kinematic.dt >= EPISODE_SECONDS

            # Episode-level bookkeeping: 'rc' is the progress reached by executing this plan,
            # so it stays post-step (the scorer only takes its maximum over the episode).
            frame['collision'] = info['collision']
            frame['rc'] = info['rc']
            save_data['frames'].append(frame)

        else:  # AD Side Crushed
            done = True

    with open(obs_pipe, "wb") as pipe:
        pipe.write(pickle.dumps('Done'))

    with open(os.path.join(output, 'data.pkl'), 'wb') as wf:
        pickle.dump([save_data], wf)

    if observations_save:
        to_video(observations_save, os.path.join(output, 'video.mp4'), fps=1.0 / (stride * cfg.kinematic.dt))
    with open(os.path.join(output, 'infos.pkl'), 'wb') as wf:
        pickle.dump(infos_save, wf)
    
    # The native scorer reads the two point clouds back and checks every frame against them: it is
    # ~69 s of the ~525 s an episode costs here, far more than the drive's own tail. A sweep that
    # scores offline from data.pkl (which this always writes, and which carries the full state) can
    # skip it; eval.json is then simply absent, and nothing else in the episode depends on it.
    if skip_native_eval:
        print('skipping the native scorer (--skip-native-eval); score offline from data.pkl', flush=True)
    else:
        ground_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'ground.ply')).points)
        scene_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'scene.ply')).points)
        results = hugsim_evaluate([save_data], ground_xyz, scene_xyz)
        with open(os.path.join(output, 'eval.json'), 'w') as f:
            json.dump(results, f)



#: What each variable the sim configs interpolate is for, shown when one is missing.
ENV_VARS = {
    "HUGSIM_DATA": "the installed data tree (HUGSIM-public / HUGSIM-private)",
    "HUGSIM_OUT": "where episode folders are written (or pass --output_dir)",
    "NUSCENES_RAW": "the raw nuScenes release, read only by scenarios with load_HD_map: true",
}

ENV_REF = re.compile(r"\$\{oc\.env:([A-Z0-9_]+)\}")


def check_env(base_cfg, keys):
    """Report, in one message, every unset variable the keys this run reads interpolate.

    The configs hold no absolute paths, so each root arrives from the environment, and each
    agent names its own variable because the agents live in unrelated checkouts. Only the keys
    a run actually reads are required, so running one agent does not need the others' paths.
    Left to OmegaConf these surface one at a time, inside a traceback that names neither the
    variable nor what it is for.
    """
    raw = OmegaConf.to_container(base_cfg, resolve=False)
    missing = {}
    for key in keys:
        node = raw
        for part in key.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        for var in ENV_REF.findall(str(node or "")):
            if var not in os.environ:
                missing.setdefault(var, []).append(key)
    if missing:
        rows = "\n".join(
            f"  {v:20s} {ENV_VARS.get(v, 'the launcher for --ad ' + v.removeprefix('HUGSIM_AD_').lower())}"
            f"   (needed by {', '.join(k)})"
            for v, k in sorted(missing.items())
        )
        raise SystemExit(f"these environment variables are not set:\n{rows}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--scenario_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--camera_path", type=str, required=True)
    parser.add_argument("--kinematic_path", type=str, required=True)
    parser.add_argument('--ad', default="uniad")
    parser.add_argument('--ad_cuda', default="1")
    parser.add_argument(
        '--ad_path', default=None,
        help="Launcher for the AD side, overriding base.<ad>_path. With this, running a new "
             "agent or a variant of one needs no config file and no code change here.")
    parser.add_argument(
        '--output_dir', default=None,
        help="Overrides base.output_dir. The agent name is still appended, as it is for the "
             "config value, so the episode folder is named the same way either way.")
    parser.add_argument(
        '--obs_mode', default=None, choices=OBS_MODES,
        help="What the agent is handed: 'rgb', the Gaussian render; 'vector', the scene as "
             "geometry, to read as is; 'raster', the same geometry, for the agent to draw with its "
             "own renderer. Overrides base.<ad>_obs_mode; rgb when neither is set.")
    parser.add_argument(
        '--raster_view', default=None, choices=RASTER_VIEWS,
        help="Raster mode: the view the agent is asked to draw, perspective or top-down. "
             "Overrides base.<ad>_raster.view; pv by default.")
    parser.add_argument(
        '--raster_rig', default=None, choices=RASTER_RIGS,
        help="Raster mode: the rig the agent is asked to draw on, its own or the scenario's "
             "cameras. Overrides base.<ad>_raster.rig; agent by default.")
    parser.add_argument(
        '--video', default=None, choices=('full', 'sparse', 'off'),
        help=f"video.mp4: every step, {SPARSE_VIDEO_HZ:g} Hz of simulated time, or none. Full by "
             "default in the rgb mode, where the frames are already rendered; sparse in the "
             "vector and raster modes, where every frame is a Gaussian render of its own.")
    parser.add_argument(
        "--skip-native-eval",
        action="store_true",
        help="do not run HUGSIM's own scorer or write eval.json (~69 s/episode); data.pkl is "
             "still written, so the episode can be scored offline",
    )
    args = parser.parse_args()

    scenario_config = OmegaConf.load(args.scenario_path)
    base_config = OmegaConf.load(args.base_path)
    camera_config = OmegaConf.load(args.camera_path)
    kinematic_config = OmegaConf.load(args.kinematic_path)
    cfg = OmegaConf.merge(
        {"scenario": scenario_config},
        {"base": base_config},
        {"camera": camera_config},
        {"kinematic": kinematic_config}
    )
    ad_key = 'dynamo_path' if args.ad.startswith('dynamo') else f'{args.ad}_path'
    needed = ['realcar_path', 'model_base']
    if args.output_dir is None:
        needed.append('output_dir')
    if args.ad_path is None and ad_key in cfg.base:
        needed.append(ad_key)
    if scenario_config.get('load_HD_map', False):
        needed.append('HD_map.path')
    check_env(cfg.base, needed)

    cfg.base.output_dir = (args.output_dir or cfg.base.output_dir) + args.ad
    # An agent declares its mode next to its launcher, base.<ad>_obs_mode, and --obs_mode
    # overrides it. The opt-in it replaces is refused rather than ignored: ignored, it would
    # quietly run that agent on the Gaussian render.
    if f'{args.ad}_scene_export' in cfg.base:
        raise SystemExit(
            f"base.{args.ad}_scene_export is gone: delete it. If the agent read the export, also set "
            f"base.{args.ad}_obs_mode to vector or raster (or pass --obs_mode) and read it from "
            "info['scene_export']")
    cfg.base.obs_mode = args.obs_mode or cfg.base.get(f'{args.ad}_obs_mode', 'rgb')
    if cfg.base.obs_mode not in OBS_MODES:
        # Checked before the agent is launched, which the env's own check would come after.
        raise SystemExit(f"base.{args.ad}_obs_mode is {cfg.base.obs_mode!r}; expected one of {OBS_MODES}")
    if cfg.base.obs_mode == 'raster':
        try:
            # Checked before the agent is launched, as the mode is. argparse holds the flags to
            # their choices, so only what was written in the config, under that key, is checked.
            raster = resolve_raster(cfg.base.get(f'{args.ad}_raster'), where=f"base.{args.ad}_raster")
        except ValueError as e:
            raise SystemExit(str(e))
        if args.raster_view is not None:
            raster['view'] = args.raster_view
        if args.raster_rig is not None:
            raster['rig'] = args.raster_rig
        cfg.base.raster = raster
    elif args.raster_view is not None or args.raster_rig is not None:
        raise SystemExit(f"--raster_view / --raster_rig need the raster mode; this run is {cfg.base.obs_mode}")
    video = args.video or ('full' if cfg.base.obs_mode == 'rgb' else 'sparse')

    model_path = os.path.join(cfg.base.model_base, cfg.scenario.scene_name)
    model_config = OmegaConf.load(os.path.join(model_path, 'cfg.yaml'))
    model_config.model_path = model_path
    cfg.update(model_config)
    
    output = os.path.join(cfg.base.output_dir, cfg.scenario.scene_name+"_"+cfg.scenario.mode)
    os.makedirs(output, exist_ok=True)

    # base.<ad>_path by convention, or --ad_path. 'dynamo*' variants share one launcher.
    if args.ad_path is not None:
        ad_path, source = args.ad_path, '--ad_path'
    else:
        env_var = f"HUGSIM_AD_{args.ad.upper()}"
        if ad_key in cfg.base:
            ad_path, source = cfg.base[ad_key], f'base.{ad_key}'
        elif env_var in os.environ:
            ad_path, source = os.environ[env_var], env_var    # an agent the shipped config does not list
        else:
            raise SystemExit(
                f"no launcher for --ad {args.ad}: pass --ad_path, export {env_var}, or add "
                f"base.{ad_key}. Listed: {sorted(k for k in cfg.base if k.endswith('_path'))}")
    # The shipped agents' launchers are in-repo paths (agents/...): one the cwd does not hold is
    # taken relative to this checkout, so a run started elsewhere still finds them.
    if ad_path and '/' in ad_path and not os.path.isabs(ad_path) and not os.path.isfile(ad_path):
        repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ad_path)
        if os.path.isfile(repo_path):
            ad_path = repo_path
    # bash reads the launcher from that file, relative to the cwd, or for a bare name from PATH.
    # Given anything else it exits at once, and the episode waits on obs_pipe for no one.
    if not ad_path or not (os.path.isfile(ad_path)
                           or ('/' not in ad_path and shutil.which(ad_path, mode=os.F_OK))):
        raise SystemExit(f"the launcher for --ad {args.ad} is {ad_path!r}, from {source}: not a file")

    # Files an earlier run left in the reused episode folder that this run may not rewrite:
    # eval.json when it is refused or passes --skip-native-eval, video.mp4 with --video off.
    for name in ('eval.json', 'video.mp4'):
        if os.path.exists(os.path.join(output, name)):
            os.remove(os.path.join(output, name))
    process = launch(ad_path, args.ad_cuda, output)
    try:
        create_gym_env(cfg, output, skip_native_eval=args.skip_native_eval, video=video)
        check_alive(process)
    except Exception as e:
        import traceback
        traceback.print_exc()
        process.kill()
    
    # For debug
    # create_gym_env(cfg, output)
