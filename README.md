# HUGSIM garage

A maintained fork of [HUGSIM](https://github.com/hyzhou404/HUGSIM) and a home for the baseline
agents that drive in it. The simulator and the data are Zhou et al.'s work ([paper](https://arxiv.org/abs/2412.01718),
[project page](https://xdimlab.github.io/HUGSIM/)); please cite them, see [Citation](#citation).

# Why this fork

**A place to keep the fixes.** Upstream's last change is from 2025-11-08, and a correctness fix
we opened four days later is still open. Rather than carry fixes as patches in every project
that uses the simulator, they live here on `main`, with their upstream status recorded below.

**A place to gather the baseline agents.** Running a closed loop needs an AD side, and those sit
in separate repositories (UniAD_SIM, VAD_SIM, NAVSIM, and whatever you are building). The
interface for plugging one in is described under [Agents](#agents) and needs no fork, no config
file and no edit to `closed_loop.py`.

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

The shipped configs name `uniad`, `vad`, `ltf`, `dynamo`, `gtrs`, `ztrs`, `gtrs_aug` and
`pictura`, each reading its own variable, because these live in unrelated checkouts and share no
root.

## Observation modes

What an agent is handed each step is its mode, `base.<ad>_obs_mode` in the config, or
`--obs_mode`:

| mode | `obs` | what the agent does |
|---|---|---|
| `rgb` (default) | each camera's Gaussian render: `rgb`, `semantic`, `depth` | drives from the images, as UniAD, VAD and NAVSIM do |
| `vector` | `{}` | reads `info['scene_export']` as it is |
| `raster` | `{}` | draws `info['scene_export']` with its own renderer, as `info['raster']` asks, and drives from that |

`vector` and `raster` hand over the same geometry: road segments, the actors (the scenario's and
the reconstruction's parked cars) as cuboids with their speeds, and the route. All of it is in
the ego frame, with z = 0 on the road. The mode is a contract, not a different export. Every
step's `info['obs_mode']` names it, so an agent built for one mode can refuse the other instead
of driving from an input it was never trained on.

A raster request has a view and a rig, in `base.<ad>_raster` or on the command line:

| key | flag | values |
|---|---|---|
| `view` | `--raster_view` | `pv` (default): perspective, one image per camera; `bev`: top-down around the ego |
| `rig` | `--raster_rig` | `agent` (default): the rig the agent was trained on; `hugsim`: the scenario's cameras, `info['cam_params']` |

The agent does the drawing, so it decides what it can honour, and refuses the rest at its first
step. Pictura's driver, for instance, draws `pv` on its own rig.

Neither abstract mode renders the cameras each step. Before the modes, an agent on the export
still had all six cameras rendered and piped to it every step, and it never read them. The
Gaussian model is still loaded: it gives the point clouds the collision checks and the scorer
use, and the export is derived from it once per scene and cached (`sim/scene_export/sidecar.py`).
`video.mp4` is then rendered only for the video, and `--video` sets how often: `full` (every
step, the default in `rgb`, where the frames already exist), `sparse` (2 Hz of simulated time,
the default in the abstract modes) or `off`. Scoring reads none of it.

The shipped configs run `pictura` in `raster`. Run a vector checkpoint with `--obs_mode vector`.

## Environment

The configs interpolate these; `closed_loop.py` checks only the ones a given run reads and
reports them all at once:

| variable | what it is |
|---|---|
| `HUGSIM_DATA` | the data tree (`HUGSIM-public` / `HUGSIM-private`) |
| `HUGSIM_OUT` | where episode folders are written, or pass `--output_dir` |
| `HUGSIM_AD_<NAME>` | one per agent, or pass `--ad_path` |
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

**Before simulation, [UniAD_SIM](https://github.com/hyzhou404/UniAD_SIM), [VAD_SIM](https://github.com/hyzhou404/VAD_SIM) and [NAVSIM](https://github.com/hyzhou404/NAVSIM) client should be installed. The client environments are allowed to be separated from the HUGSIM environment.**

The dependencies for NAVSIM are already specified as the pixi environment file, so you don't need to manually install the dependencies.

In **closed_loop.py**, we automatically launch autonomous driving algorithms.

Paths in **configs/sim/\*\_base.yaml** are `${oc.env:VAR}` interpolations in this fork, so you
export the roots rather than editing the files: see [Environment](#environment).

``` bash
CUDA_VISIBLE_DEVICES=${sim_cuda} \
python closed_loop.py --scenario_path ./configs/benchmark/${dataset_name}/${scenario_name}.yaml \
            --base_path ./configs/sim/${dataset_name}_base.yaml \
            --camera_path ./configs/sim/${dataset_name}_camera.yaml \
            --kinematic_path ./configs/sim/kinematic.yaml \
            --ad ${method_name: [uniad, vad, ltf]} \
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

In practice, you may encounter errors due to an incorrect environment, path, and etc. For debugging purposes, you can modify the last part of code as:
```python
# process = launch(ad_path, args.ad_cuda, output)
# try:
#     create_gym_env(cfg, output)
#     check_alive(process)
# except Exception as e:
#     print(e)
#     process.kill()

# For debug
create_gym_env(cfg, output)
```

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