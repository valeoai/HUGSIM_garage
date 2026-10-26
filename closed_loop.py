import sys
import os
import re
import shutil
import copy
import threading
import time
sys.path.append(os.getcwd())

import gymnasium
import hugsim_env
from hugsim_env.envs.hug_sim import OBS_CHANNELS, OBS_MODES, RASTER_RIGS, RASTER_VIEWS, resolve_raster
from argparse import ArgumentParser
from sim.utils.sim_utils import traj2control, traj_transform_to_global
from sim.ilqr.lqr import set_max_solve_time
import pickle
import json
import pickle
from sim.utils.launch_ad import launch, check_alive, kill
from omegaconf import OmegaConf
from sim.utils.score_calculator import EGO_VERTS_BEFORE_FIXES, hugsim_evaluate
from sim.utils.realized_score import read_crash_marker, score_episode
from sim.utils.score_corridor import crop_episode
import numpy as np

# Time spacing of the waypoints the AD side returns. The tracker needs it to place the plan on a
# timeline, and the scorer needs it to differentiate the plan into speeds/accelerations.
PLAN_TIMESTEP = 0.5
#: Longest episode, in simulated seconds. 400 steps at the shipped dt = 0.25 s.
EPISODE_SECONDS = 100.0
# --before_fixes (and --as_upstream): upstream HUGSIM's cap, a step count (its gymnasium registration's
# TimeLimit): 100 s at the shipped dt 0.25 s, 40 s at dt 0.1.
UPSTREAM_MAX_STEPS = 400
#: Frame rate of a sparse video, in simulated time. The abstract modes render no camera to drive
#: from, so each video frame is a Gaussian render made for the video alone.
SPARSE_VIDEO_HZ = 2.0

def to_video(observations, output_path, fps):
    from moviepy import ImageSequenceClip  # only an episode with a video needs it
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


class FifoAgent:
    """The agent as a separate process, over obs_pipe and plan_pipe (README.md, Agents)."""

    def __init__(self, output, process):
        self.output, self.process, self.finished = output, process, False
        self.obs_pipe = os.path.join(output, 'obs_pipe')
        self.plan_pipe = os.path.join(output, 'plan_pipe')
        for path in (self.obs_pipe, self.plan_pipe):
            try:
                os.mkfifo(path)
            except FileExistsError:  # an earlier run's, or the agent made it first
                pass
        if process is not None:  # None: an agent started by hand, for debugging
            threading.Thread(target=self._watch, daemon=True).start()

    def _watch(self):
        """Release the simulator from a pipe the agent will never open again.

        Opening a FIFO blocks until the other end is opened too, and nothing else watches the
        agent during the episode, so an agent that exits early (a missing checkpoint, an import
        error, a crash in a compiled op) used to leave the simulator waiting forever. Once the
        process is gone, opening each pipe from the agent's side, without blocking, wakes the
        simulator's open(); its read then finds no data and its write no reader, and ask() fails.
        """
        self.process.wait()
        while not self.finished:
            for path, flags in ((self.obs_pipe, os.O_RDONLY), (self.plan_pipe, os.O_WRONLY)):
                try:
                    os.close(os.open(path, flags | os.O_NONBLOCK))
                except OSError:  # ENXIO: the simulator is not waiting on this one
                    pass
            time.sleep(0.2)

    def _gone(self):
        return RuntimeError(
            f"the agent exited (code {self.process.poll()}) before answering; its log is "
            f"{os.path.join(self.output, 'output.txt')}")

    def ask(self, obs, info):
        message = pickle.dumps((obs, info))
        try:
            with open(self.obs_pipe, "wb") as pipe:
                pipe.write(message)
            with open(self.plan_pipe, "rb") as pipe:
                reply = pipe.read()
            return pickle.loads(reply)
        except (BrokenPipeError, EOFError, pickle.UnpicklingError):
            if self.process is not None and self.process.poll() is not None:
                raise self._gone() from None
            raise

    def done(self):
        try:
            with open(self.obs_pipe, "wb") as pipe:
                pipe.write(pickle.dumps('Done'))
        except BrokenPipeError:
            pass  # the agent is gone already; there is nobody to tell
        self.finished = True


class InProcessAgent:
    """The agent's step function, called in this process (--in_process; agents/registry.py).

    It answers as it would over the pipes: agents.client.answer turns an exception into None,
    which ends the episode as a crashed AD side does. The pipes hand each side its own copy, and
    so does this, so neither the agent nor the simulator can reach what the other keeps.
    """

    def __init__(self, ad, output, device):
        from agents.registry import load_in_process
        self.ad, self.output, self.count = ad, output, 0
        self.step, self.on_done = load_in_process(ad, output, device)

    def ask(self, obs, info):
        from agents.client import answer
        reply = answer(self.step, *copy.deepcopy((obs, info)), self.output, self.ad, self.count)
        self.count += 1
        return copy.deepcopy(reply)

    def done(self):
        # The episode's data.pkl, infos.pkl and eval.json are written after this: an agent's
        # own end-of-episode work (its video, say) failing must not take them with it.
        if self.on_done is not None:
            try:
                self.on_done()
            except Exception:
                import traceback
                traceback.print_exc()


def write_cloud(path, xyz):
    """A binary PLY of float64 x, y, z, the file open3d writes for a cloud without colours, written
    with plyfile: importing open3d costs ~5 s of an episode's start-up."""
    from plyfile import PlyData, PlyElement
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if not len(xyz):
        return  # as open3d did: no file for an empty cloud (the scorers then read it as empty)
    vertex = np.empty(len(xyz), dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
    vertex['x'], vertex['y'], vertex['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    PlyData([PlyElement.describe(vertex, 'vertex')], byte_order='<').write(path)


def create_gym_env(cfg, output, agent, skip_native_eval=False, video='full', realized_score=False, outputs='full',
                   agent_name=None, before_fixes=False, as_upstream=False):

    env = gymnasium.make('hugsim_env/HUGSim-v0', cfg=cfg, output=output)

    observations_save, infos_save = [], []
    obs, info = env.reset()
    done = False
    ad_crashed = False
    cnt = 0
    save_data = {'type': 'closeloop', 'frames': []}
    if before_fixes:
        save_data['frames_before_fixes'] = []

    print('Ready for simulation')

    stride = video_stride(video, cfg.kinematic.dt)
    obs, info = None, None
    while not done:

        if obs is None or info is None:
            obs, info = env.reset()
        if video != 'off' and cnt % stride == 0:
            # The rgb mode has usually just rendered the color images for the agent; the abstract
            # modes render nothing per step, and base.obs_channels may leave color out, so a frame
            # is then rendered for the video alone.
            observations_save.append(
                obs['rgb'] if 'rgb' in obs else env.unwrapped.render_cameras(channels=('rgb',))['rgb']
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

        plan_traj = agent.ask(obs, info)
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
            elif before_fixes:
                # As before the controller-step fix: the tracker at the plan's own 0.5 s spacing;
                # with --as_upstream also the reference headings as upstream main computes them (no #57).
                heading_order = 'upstream' if as_upstream else 'fixed'
                acc, steer_rate = traj2control(plan_traj, info, heading_order=heading_order)
                action = {'acc': acc, 'steer_rate': steer_rate}
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
            if before_fixes:
                done = terminated or truncated or cnt >= UPSTREAM_MAX_STEPS
            else:
                done = terminated or truncated or cnt * cfg.kinematic.dt >= EPISODE_SECONDS

            # Episode-level bookkeeping: 'rc' is the progress reached by executing this plan,
            # so it stays post-step (the scorer only takes its maximum over the episode).
            frame['collision'] = info['collision']
            frame['rc'] = info['rc']
            save_data['frames'].append(frame)
            if before_fixes:
                # The frame HUGSIM's scorer judged before the fixes: the loop built it after
                # env.step(), so the plan was anchored at the pose the step reached, with that
                # step's vehicles and time.
                save_data['frames_before_fixes'].append({
                    'time_stamp': info['timestamp'],
                    'is_key_frame': True,
                    'ego_box': info['ego_box'],
                    'obj_boxes': info['obj_boxes'],
                    'obj_names': ['car' for _ in info['obj_boxes']],
                    'planned_traj': {
                        'traj': traj_transform_to_global(imu_plan_traj, info['ego_box']),
                        'timestep': PLAN_TIMESTEP
                    },
                    'collision': info['collision'],
                    'rc': info['rc'],
                })

        else:  # AD Side Crushed
            done = True
            ad_crashed = True

    agent.done()

    with open(os.path.join(output, 'data.pkl'), 'wb') as wf:
        pickle.dump([save_data], wf)

    if observations_save:
        to_video(observations_save, os.path.join(output, 'video.mp4'), fps=1.0 / (stride * cfg.kinematic.dt))
    if outputs == 'full':
        with open(os.path.join(output, 'infos.pkl'), 'wb') as wf:
            pickle.dump(infos_save, wf)

    # --outputs scoring: the env wrote no clouds (base.write_point_clouds). Keep the corridor of
    # this episode's data in ground.ply / scene.ply, which is all the scorers (and score_realized.py,
    # offline) can look at, and score on those same arrays rather than reading them back. The scores
    # are identical in practice to full clouds (CPU drivable area: bitwise; GPU background check:
    # up to float32 rounding of a point on a box face).
    clouds = None
    if outputs == 'scoring' and save_data['frames']:
        try:
            clouds = crop_episode(save_data['frames'] + save_data.get('frames_before_fixes', []),
                                  env.unwrapped.ground_xyz, env.unwrapped.scene_xyz)
        except Exception:
            import traceback
            traceback.print_exc()
            print('corridor crop failed; scoring and writing the uncropped clouds', flush=True)
            clouds = (env.unwrapped.ground_xyz, env.unwrapped.scene_xyz)
        write_cloud(os.path.join(output, 'ground.ply'), clouds[0])
        write_cloud(os.path.join(output, 'scene.ply'), clouds[1])
    
    # The native scorer reads the two point clouds back and checks every frame against them: it is
    # ~69 s of the ~525 s an episode costs here, far more than the drive's own tail. A sweep that
    # scores offline from data.pkl (which this always writes, and which carries the full state) can
    # skip it; eval.json is then simply absent, and nothing else in the episode depends on it.
    if skip_native_eval:
        print('skipping the native scorer (--skip-native-eval); score offline from data.pkl', flush=True)
    else:
        # HUGSIM's scorer raises on some episodes (no frame driven, say); eval.json is then absent,
        # and the realized score below is still written.
        try:
            if clouds is None:
                import open3d as o3d
                ground_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'ground.ply')).points)
                scene_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'scene.ply')).points)
            else:
                ground_xyz, scene_xyz = clouds
            if before_fixes:
                # HUGSIM's score as the DrivoR paper's results computed it: the frames above, and
                # the collision box from before #76.
                results = hugsim_evaluate([{**save_data, 'frames': save_data['frames_before_fixes']}],
                                          ground_xyz, scene_xyz, ego_verts=EGO_VERTS_BEFORE_FIXES)
            else:
                results = hugsim_evaluate([save_data], ground_xyz, scene_xyz)
            with open(os.path.join(output, 'eval.json'), 'w') as f:
                json.dump(results, f)
        except Exception:
            import traceback
            traceback.print_exc()

    # The realized score, TerraTransfer's closed-loop HD-Score (sim/utils/realized_score.py), beside
    # HUGSIM's own: an episode the AD side ended by answering None is a crash, not a drive. An agent
    # that exits outright never gets here (ask() raises): such an episode has no score file at all.
    if realized_score:
        if ad_crashed:
            marker = read_crash_marker(output)
            result = {"crashed": True, "crash_step": marker["step"] if marker else cnt,
                      "error": marker["error"] if marker else "the AD side answered None",
                      "marker": marker["marker"] if marker else None}
        else:
            result = score_episode(output, cfg.model_path, clouds=clouds)
            if result is None:  # fewer than two frames, no crash: a file of `null` would read as a crash-free score
                result = {"too_short": True}
        # Which --ad drove: a record is otherwise only labelled by its folder name, and the name
        # `drivor` meant TOAD before agents/toad existed (run_benchmark.py refuses those records).
        result["agent"] = agent_name
        if before_fixes:
            result["before_fixes"] = True  # eval.json is then HUGSIM's score before the fixes
        if as_upstream:
            result["as_upstream"] = True  # driven as upstream HUGSIM main drives (closed_loop.py --as_upstream)
        with open(os.path.join(output, 'realized_score.json'), 'w') as f:
            json.dump(result, f)



#: What each variable the sim configs interpolate is for, shown when one is missing.
ENV_VARS = {
    "HUGSIM_DATA": "the data tree holding HUGSIM/ (3DRealCar, ss/scenes, ss/scenarios/<set>); build it with prepare_benchmark.py",
    "HUGSIM_OUT": "where episode folders are written (or pass --output_dir)",
    "NUSCENES_RAW": "the raw nuScenes release, read only by scenarios with load_HD_map: true",
}

#: The benchmark sets prepare_benchmark.py builds; episodes are filed under the one a scenario is in.
SCENARIO_SETS = ("prechallenge345", "official436")

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
        help="Overrides base.output_dir and is used as given: no <set> folder is inserted (it is "
             "for the config value). The agent name is still appended, as it is for the config "
             "value, so end it like one, e.g. /out/nuscenes_ for /out/nuscenes_<agent>/.")
    parser.add_argument(
        '--obs_mode', default=None, choices=OBS_MODES,
        help="What the agent is handed: 'rgb', the Gaussian render; 'vector', the scene as "
             "geometry, to read as is; 'raster', the same geometry, for the agent to draw with its "
             "own renderer. Overrides base.<ad>_obs_mode; rgb when neither is set.")
    parser.add_argument(
        '--obs_channels', nargs='+', default=None, choices=OBS_CHANNELS,
        help="rgb mode: the images handed to the agent each step, among rgb, semantic and depth. "
             "Overrides base.<ad>_obs_channels; rgb alone for the agents under agents/, which read "
             "nothing else, and all three otherwise.")
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
        '--outputs', default='full', choices=('full', 'scoring'),
        help="What an episode folder keeps. full: everything (ground.ply and scene.ply whole, "
             "infos.pkl, the video). scoring: only what scoring needs -- data.pkl, the scores, and "
             "ground.ply / scene.ply cropped to the corridor of the episode's own poses, which no "
             "scorer looks beyond, so every score is unchanged; no infos.pkl; --video defaults to off.")
    parser.add_argument(
        "--skip-native-eval",
        action="store_true",
        help="do not run HUGSIM's own scorer or write eval.json (~69 s/episode); data.pkl is "
             "still written, so the episode can be scored offline",
    )
    parser.add_argument(
        "--before_fixes", action="store_true",
        help="drive and score as the HUGSIM the published results such as the DrivoR paper's were run on: "
             "upstream main (hyzhou404/HUGSIM 62c690d, #56) with its PR #57, and none of this fork's later "
             "fixes that change an episode or a score. The iLQR tracker optimizes each command for the plan's "
             "0.5 s spacing instead of the simulator's step; the AttackPlanner runs on a fixed 0.1 s plan with a "
             "step-count replan (its actors move 2.5x farther at dt 0.1 than at 0.25); the episode is capped at "
             "400 steps (40 s at dt 0.1); HUGSIM's score (eval.json) judges each plan from the pose one step "
             "after it was made, with the collision box from before #76. At the shipped dt 0.25 s only the "
             "tracker and eval.json differ from the default run. realized_score.json records before_fixes.")
    parser.add_argument(
        "--as_upstream", action="store_true",
        help="drive as upstream HUGSIM main (62c690d) does, for comparing with results run on it such as "
             "TerraTransfer++'s: --before_fixes without PR #57, i.e. the tracker's reference headings computed "
             "with swapped atan2 arguments, as #57 is not merged there. realized_score.json records as_upstream.")
    parser.add_argument(
        "--deterministic_tracker",
        action="store_true",
        help="stop the iLQR tracker on its iteration cap and convergence only, not on its 0.05 s "
             "wall-clock budget, so the tracking no longer depends on how busy the machine is "
             "(scenarios with load_HD_map: true still draw actor routes at random). Changes the "
             "controls, hence the episode: run it under an agent name of its own",
    )
    parser.add_argument(
        "--realized_score",
        action="store_true",
        help="also write realized_score.json: TerraTransfer's closed-loop HD-Score, of the realized "
             "trajectory (sim/utils/realized_score.py), next to HUGSIM's own eval.json",
    )
    parser.add_argument(
        "--in_process",
        action="store_true",
        help="call the agent's step function in this process instead of launching it over the "
             "pipes: only for agents that run in this environment, listed in agents/registry.py. "
             "The agent uses this process's GPU; --ad_cuda and --ad_path do not apply",
    )
    args = parser.parse_args()
    if args.as_upstream:
        args.before_fixes = True  # --as_upstream includes it
    if args.before_fixes and args.skip_native_eval:
        # Only the episode's own scoring judges frames_before_fixes with the old box; the offline
        # scorers that --skip-native-eval leaves the score to are the fork's.
        raise SystemExit("--before_fixes needs HUGSIM's own scorer: drop --skip-native-eval")
    if args.deterministic_tracker:
        set_max_solve_time(None)
    if args.in_process and args.ad_path is not None:
        raise SystemExit("--in_process calls the agent from agents/registry.py; it takes no --ad_path")

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
    if args.ad_path is None and not args.in_process and ad_key in cfg.base:
        needed.append(ad_key)
    if scenario_config.get('load_HD_map', False):
        needed.append('HD_map.path')
    check_env(cfg.base, needed)

    output_dir = args.output_dir
    if output_dir is None:
        # .../ss/scenarios/<set>/<dataset>/<scenario>.yaml: a scenario in both benchmark sets
        # must not overwrite the other set's run, so the set joins the folder name.
        scenario_set = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args.scenario_path))))
        head, tail = os.path.split(cfg.base.output_dir)
        output_dir = os.path.join(head, scenario_set, tail) if scenario_set in SCENARIO_SETS else cfg.base.output_dir
    cfg.base.output_dir = output_dir + args.ad
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
    video = args.video or ('off' if args.outputs == 'scoring' else 'full' if cfg.base.obs_mode == 'rgb' else 'sparse')
    cfg.base.write_point_clouds = args.outputs == 'full'

    model_path = os.path.join(cfg.base.model_base, cfg.scenario.scene_name)
    model_config = OmegaConf.load(os.path.join(model_path, 'cfg.yaml'))
    model_config.model_path = model_path
    cfg.update(model_config)
    
    output = os.path.join(cfg.base.output_dir, cfg.scenario.scene_name+"_"+cfg.scenario.mode)
    os.makedirs(output, exist_ok=True)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    if args.in_process:
        launcher_in_repo = True
    else:
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
        # The shipped agents' launchers are in-repo paths (agents/...), taken relative to this
        # checkout before the cwd, so the launcher runs the same agents/ that --in_process imports.
        if ad_path and '/' in ad_path and not os.path.isabs(ad_path) \
                and os.path.isfile(os.path.join(repo_root, ad_path)):
            ad_path = os.path.join(repo_root, ad_path)
        # bash reads the launcher from that file, relative to the cwd, or for a bare name from PATH.
        # Given anything else it exits at once, and the episode waits on obs_pipe for no one.
        if not ad_path or not (os.path.isfile(ad_path)
                               or ('/' not in ad_path and shutil.which(ad_path, mode=os.F_OK))):
            raise SystemExit(f"the launcher for --ad {args.ad} is {ad_path!r}, from {source}: not a file")
        agents_dir = os.path.join(repo_root, 'agents', '')
        launcher_in_repo = os.path.abspath(ad_path).startswith(agents_dir) \
            and not os.path.abspath(ad_path).startswith(os.path.join(agents_dir, 'tools', ''))
    # The agents under agents/ read their weights from HUGSIM_AGENT_CKPTS. Checked here, where
    # the message can say so, rather than by an agent that would only fail once launched.
    if launcher_in_repo and not os.environ.get('HUGSIM_AGENT_CKPTS'):
        raise SystemExit(
            f"--ad {args.ad} runs from agents/, which reads its weights from HUGSIM_AGENT_CKPTS: "
            "set it to the checkpoint root (agents/README.md, Checkpoints)")
    # The images an rgb-mode agent is handed. The agents under agents/ all read RGB alone, so they
    # get only that, and the semantic and depth images are neither computed nor copied back from
    # the GPU. Any other AD side gets all three unless it declares less (base.<ad>_obs_channels).
    channels = (args.obs_channels or cfg.base.get(f'{args.ad}_obs_channels')
                or (['rgb'] if launcher_in_repo else OBS_CHANNELS))
    cfg.base.obs_channels = [channels] if isinstance(channels, str) else list(channels)  # 'rgb' is ['rgb']
    cfg.base.upstream_actors = bool(args.before_fixes)  # read by HUGSimEnv for its planner (the AttackPlanner as upstream calls it)

    if args.in_process:
        try:
            # Loaded before the episode starts, so a missing checkpoint or an unknown name stops
            # the run here rather than inside it. The device is the agent's default (cuda, or
            # HUGSIM_AGENT_DEVICE), as over the pipes.
            agent = InProcessAgent(args.ad, output, device=None)
        except Exception as e:
            raise SystemExit(f"--in_process --ad {args.ad}: {type(e).__name__}: {e}")

    # Files an earlier run left in the reused episode folder that this run may not rewrite:
    # eval.json when it is refused or passes --skip-native-eval, video.mp4 with --video off,
    # agent_crash.txt / pictura_crash.json when the agent does not crash this time,
    # realized_score.json without --realized_score, and data.pkl / infos.pkl when the episode
    # dies before writing them (an agent that exits outright, say), which a scorer would
    # otherwise take for this run's drive.
    stale = ('eval.json', 'video.mp4', 'agent_crash.txt', 'pictura_crash.json', 'realized_score.json',
             'data.pkl', 'infos.pkl')
    if args.outputs == 'scoring':
        stale += ('ground.ply', 'scene.ply')  # an episode that dies early would leave an earlier run's
    for name in stale:
        if os.path.exists(os.path.join(output, name)):
            os.remove(os.path.join(output, name))
    if args.in_process:
        try:
            create_gym_env(cfg, output, agent, skip_native_eval=args.skip_native_eval, video=video,
                           realized_score=args.realized_score, outputs=args.outputs, agent_name=args.ad,
                           before_fixes=args.before_fixes, as_upstream=args.as_upstream)
        except Exception:
            import traceback
            traceback.print_exc()
    else:
        process = launch(ad_path, args.ad_cuda, output)
        try:
            create_gym_env(cfg, output, FifoAgent(output, process),
                           skip_native_eval=args.skip_native_eval, video=video,
                           realized_score=args.realized_score, outputs=args.outputs, agent_name=args.ad,
                           before_fixes=args.before_fixes, as_upstream=args.as_upstream)
            check_alive(process)
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            # Whatever ended the episode (an exception, Ctrl-C), nothing the launcher started
            # outlives it: the agent runs in a session of its own (sim/utils/launch_ad.py).
            kill(process)

    # For debug, without launching the agent: start it by hand on the episode folder, then
    # create_gym_env(cfg, output, FifoAgent(output, process=None))
