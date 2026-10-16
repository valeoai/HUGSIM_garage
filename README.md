# HUGSIM garage

A maintained fork of [HUGSIM](https://github.com/hyzhou404/HUGSIM) and a home for the baseline
agents that drive in it. The simulator and the data are Zhou et al.'s work ([paper](https://arxiv.org/abs/2412.01718),
[project page](https://xdimlab.github.io/HUGSIM/)); please cite them, see [Citation](#citation).

# Why this fork

**A place to keep the fixes.** Upstream's last change is from 2025-11-08, and a correctness fix
we opened four days later is still open. Rather than carry fixes as patches in every project
that uses the simulator, they live here on `main`, with their upstream status recorded below.

**A place to gather the baseline agents.** Running a closed loop needs an AD side. Upstream's
sat in separate repositories (UniAD_SIM, VAD_SIM, NAVSIM), each with its own environment; here
they are in [`agents/`](agents/README.md): UniAD and VAD with the mmcv environment they need,
LTF, DrivoR, GTRS, ZTRS and GTRS-Aug in the simulator's own. Plugging in one of your own needs no
fork, no config file and no edit to `closed_loop.py`, see [Agents](#agents).

## What this fork changes

### Upstreamed

| | |
|---|---|
| [hyzhou404/HUGSIM#56](https://github.com/hyzhou404/HUGSIM/pull/56) | **Merged 2025-11-08.** The comfort check compared longitudinal acceleration against the *lateral* bound (4.89 instead of 2.40 m/s²); the planned-trajectory timestep was 0.25 s where the controller and the baselines use 0.5 s; the heading fed to the controller was measured from the initial position rather than the previous point, with `arctan` instead of `arctan2`. |

### Proposed upstream, not merged, carried here

| | |
|---|---|
| [hyzhou404/HUGSIM#57](https://github.com/hyzhou404/HUGSIM/pull/57) | Open since 2025-11-12. The `arctan2` arguments are swapped, so the heading is `π/2 − heading`. It should be `arctan2(lateral, forward)`. |
| [hyzhou404/HUGSIM#76](https://github.com/hyzhou404/HUGSIM/pull/76) | Open since 2026-10-02. The plan scorer's background-collision box sits **above** the car. `ego_box`'s z is the camera plane, about 1.45 m above the road, and the box runs up from it, so it covers 1.45–2.95 m: it misses parked car bodies, kerbs and bollards, and fires on overhangs, canopies and stray façade Gaussians. `HUGSimEnv`'s own per-step check runs from the camera plane *down* to the road; the scorer now matches it. |

`#76` changes `nc` and `ttc`, so **scores from before and after it are not comparable**. Measured
on the 88 nuScenes scenarios for nine agents, HD-Score moves by less than 0.010 for each and no
ranking changes, but it is not a bias that cancels: 5–16 episodes per agent change `nc`, with
per-episode swings to 0.12, and the direction depends on the agent. Plan-and-tracker agents lose
(8–11 episodes worse against 1–5 better) because the raised box was missing collisions at car
height; reactive agents gain, because it was also inventing collisions overhead.

### Fixed here, not yet proposed upstream

* **The comfort bound is wrong in a second place.** `#56` fixed `_calculate_is_comfortable`;
  `_calculate_actual_comfort` still compared longitudinal acceleration against the lateral bound.
  Inert today, since its only call site is commented out, but a trap for whoever uncomments it.
* **The tracker optimises for the step it executes.** `traj2control` now resamples the reference
  onto the simulator's timeline when `plan_dt != sim_dt` and passes `discretization_time`, instead
  of assuming the two agree.

### Interface, not bug fixes

* **No absolute paths in the configs.** Every root is an `${oc.env:VAR}` interpolation, so one
  checkout serves any machine. See [Agents](#agents).
* **Three observation modes.** Besides the Gaussian render (`rgb`), an agent can be handed the
  scene as geometry, `sim/scene_export/` (typed road segments, actor cuboids, the route), and
  asked to read it as it is (`vector`) or to draw it with its own renderer (`raster`). See
  [Observation modes](#observation-modes).
* **Extra flags from the sweep scripts.** `run_dataset.sh` and `run_single.sh` append
  `HUGSIM_ARGS`, split on whitespace, to every `closed_loop.py` call. `run_dataset.sh` refuses
  `--output_dir`, `--ad`, `--base_path` and `--scenario_path` there, since they pick the episode
  folder its `eval.json` skip reads. See [Running a variant](#running-a-variant).
* **The baseline agents ship in [`agents/`](agents/README.md).** The configs' `base.<ad>_path` is
  `${oc.env:HUGSIM_AD_<NAME>,agents/...}`: the in-repo launcher unless the variable is exported.
  A relative launcher path is taken against this checkout before the cwd. Their weights come from
  `HUGSIM_AGENT_CKPTS`, which `closed_loop.py` checks before launching one of them.
* **`--in_process`** calls an agent's step function in the simulator's process instead of over
  the pipes, for the agents in `agents/registry.py`. The default stays the FIFO.
* **An agent that exits early no longer hangs the episode.** `closed_loop.py` watches the
  launcher's process and, once it is gone, releases whichever pipe the simulator waits on; the
  episode then ends with an error naming the agent's exit code and log. The launcher runs in a
  session of its own and its whole process group is killed at the end, so an agent behind
  `pixi run` or `conda run` (which fork rather than exec) is not left holding its GPU.

# Agents

An agent is a script HUGSIM launches with `(cuda_device, episode_dir)`; it talks to the simulator
over two FIFOs in that directory. Point the simulator at it in whichever way suits you:

```bash
python closed_loop.py --scenario_path ... --base_path ./configs/sim/nuscenes_base.yaml \
    --camera_path ./configs/sim/nuscenes_camera.yaml --kinematic_path ./configs/sim/kinematic.yaml \
    --ad my_agent --ad_path /path/to/my_e2e.sh --output_dir /where/episodes/go
```

`--ad_path` wins; otherwise `base.<ad>_path` from the config; otherwise the environment variable
`HUGSIM_AD_<NAME>`. **Adding an agent needs no config file and no change to `closed_loop.py`.**

`closed_loop.py` refuses a launcher that is not a file before launching it, and just before
launching removes an earlier run's `eval.json` and `video.mp4` from the episode folder, so
neither outlives a run that does not write it (an agent's refusal or `--skip-native-eval` writes
no `eval.json`, `--video off` no `video.mp4`).

The shipped configs name `uniad`, `vad`, `ltf`, `drivor`, `gtrs`, `ztrs` and `gtrs_aug`, whose
launchers are in [`agents/`](agents/README.md) (`base.<ad>_path` is
`${oc.env:HUGSIM_AD_<NAME>,agents/...}`, so an exported variable still replaces one), and
`dynamo` and `pictura`, which live in unrelated checkouts and read their variable alone.

The agents that run in this environment (`ltf`, `drivor`, `gtrs`, `ztrs`, `gtrs_aug`, listed in
`agents/registry.py`) can be called in the simulator's process instead, with `--in_process`: no
launcher, no pipes, and a step of LTF about 1.5 times faster. See
[agents/README.md](agents/README.md#in-process).

## Observation modes

What an agent is handed each step is its mode, `base.<ad>_obs_mode` in the config, or
`--obs_mode`:

| mode | `obs` | what the agent does |
|---|---|---|
| `rgb` (default) | each camera's Gaussian render: `rgb`, `semantic`, `depth` | drives from the images, as UniAD, VAD and NAVSIM do |
| `vector` | `{}` | reads `info['scene_export']` as it is |
| `raster` | `{}` | draws `info['scene_export']` with its own renderer, as `info['raster']` asks, and drives from that |

`vector` and `raster` hand over the same geometry, `info['scene_export']`: `roads` (road
segments), `cuboids` (the actors, the scenario's and the reconstruction's parked cars),
`cuboid_speed` (their speeds) and `route`, laid out as in `sim/scene_export/entities.py`. All
of it is in the ego frame of `ego_box`: x forward, y left, z up with z = 0 on the road, yaw
counter-clockwise. The plan the agent sends back stays `(right, forward)`, as in `rgb`. The
mode is a contract, not a different export. Every step's `info['obs_mode']` names it, so an
agent built for one mode can refuse the other instead of driving from an input it was never
trained on.

A raster request has a view and a rig, in `base.<ad>_raster` or on the command line:

| key | flag | values |
|---|---|---|
| `view` | `--raster_view` | `pv` (default): perspective, one image per camera; `bev`: top-down around the ego |
| `rig` | `--raster_rig` | `agent` (default): the rig the agent was trained on; `hugsim`: the scenario's cameras |

`info['cam_params']` gives the scenario's cameras' intrinsics and extrinsics, but not the rest of
what places them against the export: the camera config's `cam_rect`, the export's rig pitch
(non-zero on KITTI-360), the cameras' height above its z = 0 road, and its xy origin. The export is
centred on the road below `CAM_FRONT`, x along its heading, and the render places each camera
from `CAM_FRONT` (`v2front @ inv(v2c) @ cam_rect`): a rig put at `inv(v2c)` in the export's
frame has every camera 1.72 m forward of the render, `CAM_FRONT`'s position in the vehicle frame
(the same `v2c` in all four camera configs). A `hugsim` raster drawn from `info` alone does not
line up exactly with the `rgb` render.

The agent does the drawing, so it decides what it can honour: Pictura's driver, for instance,
draws `pv` on its own rig. An agent that cannot honour the mode or the raster request refuses it
at its first step: it says why in its log (its stdout and stderr go to the episode's
`output.txt`) and answers `None` on `plan_pipe`. `closed_loop.py` then ends the episode and
sends `'Done'`. The episode folder has no `eval.json` and its `data.pkl` holds no frames, so
`get_score.py` lists the scenario as missing; `closed_loop.py` still exits 0. An agent must
answer, and then read that `'Done'`: one that raises or exits instead leaves `closed_loop.py`
blocked on a pipe until something kills it, since nothing watches the agent's process during
the episode.

Neither abstract mode renders the cameras each step. Before the modes, an agent on the export
still had all six cameras rendered and piped to it every step, and it never read them. The
Gaussian model is still loaded: it gives the point clouds the collision checks and the scorer
use, and the export is derived from it once per scene and cached (`sim/scene_export/sidecar.py`).
The cameras are then rendered only for `video.mp4`, and `--video` sets how often: `full` (every
step, the default in `rgb`, where the frames already exist), `sparse` (2 Hz of simulated time,
the default in the abstract modes) or `off`. Scoring reads none of it.

The shipped configs run `pictura` in `raster`; a vector checkpoint runs as a variant (below).

### Running a variant

Episode folders are named after the agent and the scenario's difficulty, not the observation
mode (`<dataset>_<agent>/<scene>_<difficulty>`, e.g. `nuscenes_pictura/scene-0383_easy_00`), so
a variant, a vector checkpoint say, runs under an agent name of its own. From the sweep scripts,
its flags go in `HUGSIM_ARGS`, an environment variable that `run_dataset.sh` and `run_single.sh`
split on whitespace and append to every `closed_loop.py` call. The variant's name reads none of
the base agent's `base.<agent>_*` keys: its launcher comes from `HUGSIM_AD_<NAME>`, and its mode
and raster request from the flags. The exception is a name starting with `dynamo`: every one
takes the shared `base.dynamo_path` (`HUGSIM_AD_DYNAMO`), so a `dynamo` variant gets its own
launcher from `--ad_path` in `HUGSIM_ARGS`. The launcher is not given the agent name, so the
agent picks its checkpoint from its own environment (Pictura's reads `PICTURA_CHECKPOINT`):

```bash
PICTURA_CHECKPOINT=/path/to/vector.pt HUGSIM_AD_PICTURA_VECTOR=/path/to/Pictura-dev/hugsim_e2e.sh HUGSIM_ARGS="--obs_mode vector" bash run_dataset.sh pictura_vector nuscenes 0 0
```

These episodes land in `$HUGSIM_OUT/nuscenes_pictura_vector/`. `run_dataset.sh`'s skip, which
lets a forward and a reverse worker split the scenes, looks for them under the `OUTPUT_PATH` set
at the top of the script: it must be the same tree as `$HUGSIM_OUT`, as it is in the shipped
`hugsim_env.sh`. `get_score.py` takes no arguments: to score the variant, edit its constants
`MODEL_NAME` and `BENCH_ROOT` to `pictura_vector` and that tree. `run_all.sh` and
`run_sweep.sh` take the agent name and pass it on. Any other `HUGSIM_ARGS` that changes what
an episode is, such as another `--kinematic_path` (dt 0.1), needs an agent name of its own
too, and the mode again: a `pictura` variant without `--obs_mode raster` runs in `rgb`, and
Pictura refuses every episode.

```bash
HUGSIM_AD_PICTURA_10HZ=/path/to/Pictura-dev/hugsim_e2e.sh HUGSIM_ARGS="--obs_mode raster --kinematic_path ./configs/sim/kinematic_10hz.yaml" bash run_dataset.sh pictura_10hz nuscenes 0 0
```

`run_dataset.sh` refuses `--output_dir`, `--ad`, `--base_path` and `--scenario_path` in
`HUGSIM_ARGS`, and any abbreviation of them: they would send the episode somewhere other than the
folder `run_dataset.sh` checks for `eval.json`. `--skip-native-eval` writes no `eval.json`, so
that check never sees a finished scene: run one worker then, not a forward and reverse pair. Calling `closed_loop.py` directly,
`--ad pictura_vector --ad_path ...` and `--output_dir` both work.

## Environment

The configs interpolate these (all but `HUGSIM_AGENT_CKPTS`, which the agents in `agents/` read
themselves); `closed_loop.py` checks only the ones a given run reads and reports them all at once:

| variable | what it is |
|---|---|
| `HUGSIM_DATA` | the data tree (`HUGSIM-public` / `HUGSIM-private`) |
| `HUGSIM_OUT` | where episode folders are written, or pass `--output_dir` |
| `HUGSIM_AD_<NAME>` | an agent's launcher, or pass `--ad_path`; needed only for agents not in `agents/` |
| `HUGSIM_AGENT_CKPTS` | the weights of the agents in `agents/` ([layout](agents/README.md#checkpoints)) |
| `NUSCENES_RAW` | raw nuScenes, read only by scenarios with `load_HD_map: true` |

---

# Installation

First, install [pixi](https://pixi.sh/latest/):

``` bash
curl -fsSL https://pixi.sh/install.sh | sh
```

As the repository depends on some packages that can only be installed from source code, and rely on pytorch and cuda to compile, the installation of pixi environment is seperated as **two steps**:

1. Comment the packages below `# install from source code` in `pixi.toml`, then run `pixi install` to install the packages from pypi.
2. Uncomment the packages in the previous step, then run `pixi install` to install these packages from source code.
3. Install apex (required by InverseForm) by running `pixi run install-apex`

Change into the **pixi environment** by using the command `pixi shell`.

Or you can use `pixi run <command>` to run a command in the **pixi environment**.


# Data Preparation

Please refer to [Data Preparation Document](data/README.md)

You can download sample data from [here](https://huggingface.co/datasets/hyzhou404/HUGSIM/tree/main/sample_data).

# Reconstruction

``` bash
seq=${seq_name}
input_path=${datadir}/${seq}
output_path=${modeldir}/${seq}
mkdir -p ${output_path}
CUDA_VISIBLE_DEVICES=4 \
python -u train_ground.py --data_cfg ./configs/${dataset_name: [kitti360, waymo, nusc, pandaset]}.yaml \
        --source_path ${input_path} --model_path ${output_path}
CUDA_VISIBLE_DEVICES=4 \
python -u train.py --data_cfg ./configs/${dataset_name}.yaml \
        --source_path ${input_path} --model_path ${output_path}
```

# Scene Export

The reconstructed scene folders contain some information that won't be utilized during the simulation. The scenes are expected to be exported as a minimized format to facilitate easier sharing and simulation.
```bash
 python eval_render/export_scene.py --model_path ${recon_scene_path} --output_path ${export_path} --iteration 30000
``` 
We've made some changes in the capturing and reloading code. If you would like to convert scenes from previous version (before commit 1ca821a8) of our code, add `--ver0` in the above command. 

# Vehicles, Scenes and Scenarios

We have released all 3DRealCar files, along with the complete set of scenes and scenarios, at [release link](https://huggingface.co/datasets/XDimLab/HUGSIM). 
We are also holding a competition at [RealADSim @ ICCV 2025](https://huggingface.co/spaces/XDimLab/ICCV2025-RealADSim-ClosedLoop), so some scenarios and scenarios are hosted privately. We welcome participants to join!

# Scenarios configuration with GUI

**Note that this GUI is only used for configuration scenarios, rather than simulation. The rendering quality in GUI is not the results during simulation**

First convert the vehicles and scenes to splat and semantic format.

``` bash
python eval_render/convert_vehicles.py --vehicle_path ${PATH_3DRealCar}
python eval_render/convert_scene.py --model_path ${PATH_Scene}
```

Then, you can run the GUI to configure the scenario. 
**nuscenes_camera.yaml** in gui/static/data provides a camera configuration template, you can modify it to fit your needs.

``` bash
cd gui
python app.py --scene ${PATH_Scene} --car_folder ${PATH_3DRealCar/converted}
```

You can configure the scenario with the GUI, and download the yaml file to use in simulation.

Here is a video for the GUI usage demonstration: [GUI Video](https://github.com/hyzhou404/HUGSIM/blob/main/assets/hugsim_gui.mp4)

# Simulation

**The AD clients are in [`agents/`](agents/README.md)**, vendored from upstream's
[UniAD_SIM](https://github.com/hyzhou404/UniAD_SIM), [VAD_SIM](https://github.com/hyzhou404/VAD_SIM)
and [NAVSIM](https://github.com/hyzhou404/NAVSIM). LTF, DrivoR, GTRS, ZTRS and GTRS-Aug run in
the pixi environment above; UniAD and VAD need the environment in `agents/mmdet3d/` (one
`pixi install` and two build tasks, [agents/mmdet3d/README.md](agents/mmdet3d/README.md)). All of
them read their weights from `HUGSIM_AGENT_CKPTS`.

In **closed_loop.py**, we automatically launch autonomous driving algorithms.

Paths in **configs/sim/\*\_base.yaml** are `${oc.env:VAR}` interpolations in this fork, so you
export the roots rather than editing the files: see [Environment](#environment).

``` bash
CUDA_VISIBLE_DEVICES=${sim_cuda} \
python closed_loop.py --scenario_path ./configs/benchmark/${dataset_name}/${scenario_name}.yaml \
            --base_path ./configs/sim/${dataset_name}_base.yaml \
            --camera_path ./configs/sim/${dataset_name}_camera.yaml \
            --kinematic_path ./configs/sim/kinematic.yaml \
            --ad ${method_name: [uniad, vad, ltf, drivor, gtrs, ztrs, gtrs_aug]} \
            --ad_cuda ${ad_cuda}
```

Run the following commands to execute.

```bash
sim_cuda=0
ad_cuda=1

# change this variable as the scenario path on your machine
scenario_dir=${SCENARIO_PATH} 

for cfg in ${scenario_dir}/*.yaml; do
    echo ${cfg}
    CUDA_VISIBLE_DEVICES=${sim_cuda} \
    python closed_loop.py --scenario_path ${cfg} \
                        --base_path ./configs/sim/nuscenes_base.yaml \
                        --camera_path ./configs/sim/nuscenes_camera.yaml \
                        --kinematic_path ./configs/sim/kinematic.yaml \
                        --ad uniad \
                        --ad_cuda ${ad_cuda}
done
```

In practice, you may encounter errors due to an incorrect environment, path, and etc. The agent's
own output is in `<episode>/output.txt`, and a step that raised leaves its traceback in
`<episode>/agent_crash.txt`. To debug the agent by hand, replace the launch at the end of
`closed_loop.py` with `create_gym_env(cfg, output, FifoAgent(output, process=None))` and start the
agent yourself on the episode folder; an agent in `agents/registry.py` can instead run with
`--in_process`, where a breakpoint in its step stops the simulator too.

# TODO list
- [x] Release sample data and results
- [x] Release unicycle model part
- [x] Release GUI
- [x] Release more scenarios

# Citation

If you find our paper and codes useful, please kindly cite us via:

```bibtex
@article{zhou2024hugsim,
  title={HUGSIM: A Real-Time, Photo-Realistic and Closed-Loop Simulator for Autonomous Driving},
  author={Zhou, Hongyu and Lin, Longzhong and Wang, Jiabao and Lu, Yichong and Bai, Dongfeng and Liu, Bingbing and Wang, Yue and Geiger, Andreas and Liao, Yiyi},
  journal={arXiv preprint arXiv:2412.01718},
  year={2024}
}
```