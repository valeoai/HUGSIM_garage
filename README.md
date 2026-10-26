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
LTF, DrivoR, TOAD (as the HUGSIM client ran it and as released), GTRS, ZTRS and GTRS-Aug in the simulator's own. Plugging in one of your own needs no
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
* **The tracker can be made a function of its inputs (opt-in, `--deterministic_tracker`).** The
  iLQR tracker stops a solve after 0.05 s of wall-clock time (`max_solve_time`, from nuPlan), so the
  command it returns depends on how busy the machine is, and two runs of one agent on one scenario
  drift apart from the first step. With four LTF episodes sharing a node, the budget cut 91 % of
  solves short, after a median 14 to 18 of the about 33 iterations that convergence takes; two
  default runs ended 69 and 67 frames long, two runs with the flag were identical frame for frame.
  The flag stops the solver on convergence and its 100-iteration cap only (every solve converged
  by iteration 37): about 45 ms more per step there (median 54 to 98 ms), 3 s over a 68-step
  episode at dt 0.25 s; the cost at dt 0.1 s is not measured. Off by default, because it changes
  the controls and so every score. It makes the tracking reproducible, not every episode: on a
  scenario with `load_HD_map: true` the actors still pick their route at a fork at random
  (`get_route`, `sim/utils/plan.py`), unseeded and from a set whose order changes per process.

### Interface, not bug fixes

* **No absolute paths in the configs.** Every root is an `${oc.env:VAR}` interpolation, so one
  checkout serves any machine. See [Agents](#agents).
* **Two benchmark sets, one scene tree.** `prechallenge345` (the 345 scenarios of the 2025-07
  release) and `official436` (the current one) share one scene tree and one 3DRealCar copy, and
  [`prepare_benchmark.py`](#benchmark-sets) builds the layout from the pinned Hugging Face
  revisions and repairs the scenarios that do not run as shipped. The sweep scripts take the set as
  an argument and episodes are filed under it.
* **Three observation modes.** Besides the Gaussian render (`rgb`), an agent can be handed the
  scene as geometry, `sim/scene_export/` (typed road segments, actor cuboids, the route), and
  asked to read it as it is (`vector`) or to draw it with its own renderer (`raster`). See
  [Observation modes](#observation-modes).
* **The realized score, TerraTransfer's closed-loop HD-Score**, beside HUGSIM's own `hdscore`:
  `--realized_score` per episode, `score_realized.py` for past runs. See [Scoring](#scoring).
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
    --ad my_agent --ad_path /path/to/my_e2e.sh --output_dir /where/episodes/go/nuscenes_
```

`--ad_path` wins; otherwise `base.<ad>_path` from the config; otherwise the environment variable
`HUGSIM_AD_<NAME>`. **Adding an agent needs no config file and no change to `closed_loop.py`.**

`closed_loop.py` refuses a launcher that is not a file before launching it, and just before
launching removes an earlier run's `eval.json` and `video.mp4` from the episode folder, so
neither outlives a run that does not write it (an agent's refusal or `--skip-native-eval` writes
no `eval.json`, `--video off` no `video.mp4`).

The shipped configs name `uniad`, `vad`, `ltf`, `drivor`, `drivor_paper`, `toad`, `gtrs`, `ztrs` and `gtrs_aug`, whose
launchers are in [`agents/`](agents/README.md) (`base.<ad>_path` is
`${oc.env:HUGSIM_AD_<NAME>,agents/...}`, so an exported variable still replaces one), and
`dynamo` and `pictura`, which live in unrelated checkouts and read their variable alone.

The agents that run in this environment (`ltf`, `drivor`, `drivor_paper`, `toad`, `gtrs`, `ztrs`, `gtrs_aug`, listed in
`agents/registry.py`) can be called in the simulator's process instead, with `--in_process`: no
launcher, no pipes, and a step of LTF about 1.5 times faster. See
[agents/README.md](agents/README.md#in-process).

## Observation modes

What an agent is handed each step is its mode, `base.<ad>_obs_mode` in the config, or
`--obs_mode`:

| mode | `obs` | what the agent does |
|---|---|---|
| `rgb` (default) | each camera's Gaussian render: `rgb`, `semantic`, `depth`, or those `obs_channels` selects | drives from the images, as UniAD, VAD and NAVSIM do |
| `vector` | `{}` | reads `info['scene_export']` as it is |
| `raster` | `{}` | draws `info['scene_export']` with its own renderer, as `info['raster']` asks, and drives from that |

In `rgb`, `base.<ad>_obs_channels` or `--obs_channels` picks the images among `rgb`, `semantic` and
`depth`; one left out is neither computed from the render nor copied back from the GPU. The agents
under `agents/` read `rgb` alone and get only that; any other AD side gets all three unless it
declares less.

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
answer, and then read that `'Done'`. One that exits instead is caught: `closed_loop.py` watches
the launcher's process, releases the pipe it waits on, and ends the episode with an error naming
the agent's exit code and `output.txt` (agents/client.py's `serve` keeps both promises for the
agents in `agents/`).

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
mode (`<set>/<dataset>_<agent>/<scene>_<difficulty>`, e.g. `official436/nuscenes_pictura/scene-0383_easy_00`), so
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

These episodes land in `$HUGSIM_OUT/official436/nuscenes_pictura_vector/`. `run_dataset.sh`'s
skip, which lets a forward and a reverse worker split the scenes, looks for them there, under the
same `$HUGSIM_OUT` that `closed_loop.py` writes to. `python get_score.py official436 pictura_vector`
scores the variant. `run_all.sh` and `run_sweep.sh` take the agent name and pass it on. Any other
`HUGSIM_ARGS` that changes what an episode is, such as another `--kinematic_path` (dt 0.1) or
`--deterministic_tracker`, needs an agent name of its own
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

## Scoring

Two scores, for two questions:

| | what it judges | where |
|---|---|---|
| `hdscore`, HUGSIM's HD-Score | each step's **planned** trajectory (NC, DAC, TTC, comfort on the agent's waypoints), times route completion | `eval.json`, written at the end of every episode unless `--skip-native-eval` is passed or HUGSIM's scorer fails on it (`get_score.py` averages them) |
| the realized score, TerraTransfer's closed-loop HD-Score ([arXiv 2606.17386](https://arxiv.org/abs/2606.17386), App. B) | the **realized** trajectory: per 0.5 s keyframe NC and DAC at the ego's pose, TTC propagated 1 s ahead, comfort over the next 1 s | `realized_score.json` with `--realized_score`, or `score_realized.py` for past runs |

`--before_fixes` drives and scores as the HUGSIM the published results such as the DrivoR paper's were
run on: upstream main (hyzhou404/HUGSIM `62c690d`, #56) with its PR #57, and none of this fork's later fixes
that change an episode or a score. It restores, for each such fix, the upstream behaviour:

| upstream behaviour restored | fork fix it undoes |
|---|---|
| the iLQR tracker optimizes each command for the plan's 0.5 s spacing, not the simulator's step | the controller-step fix |
| the AttackPlanner on a fixed 0.1 s plan, one row per call, forecasts on the simulator's step, replanned on a step count: at dt 0.1 its actors move 2.5x the distance they cover at the shipped 0.25 s | 98ee0bf, 5105d46, 3b4f070 |
| the episode capped at 400 steps (100 s at 0.25 s, 40 s at 0.1 s) | 8dda047, a50c161 |
| `eval.json` judges each plan from the pose one step after it was made (the frames kept as `frames_before_fixes` in `data.pkl`), with the collision box from before #76 | the plan-pose and collision-box fixes |

At the shipped dt 0.25 s the step-size fixes change nothing, so there only the tracker and `eval.json`
differ from the default run. `--as_upstream` is `--before_fixes` without #57, which upstream main still
lacks: the tracker's reference headings computed with swapped `atan2` arguments (pi/2 minus the heading).
It is the HUGSIM results such as TerraTransfer++'s (arXiv 2610.30319, at dt 0.1 s) were run on. Kept in
both: the IDM lookahead fix (dd964ea) and the last pose's command (`dense_cam_poses`), whose upstream
behaviour raises and loses the episode rather than driving it differently; the observation modes and the
scene export, which only hand the agent information. `realized_score.json` records `"before_fixes": true`
(and `"as_upstream": true`); keep these episodes in an output folder of their own. Validated against
upstream checkouts: LTF on official436 reproduces their scores.

`hdscore` always means HUGSIM's own score. The realized score is the protocol to match to be
comparable with TerraTransfer, and with any agent that, like its jerk policy, has no
multi-second plan to score. It reads only what an episode already saves, so every past run can
be rescored:

```bash
python score_realized.py --runs official436/nuscenes_uniad official436/nuscenes_vad --out realized.json   # under $HUGSIM_OUT
```

`realized_score.json` (and each entry of `score_realized.py`'s output) carries the three HD-Score
variants side by side:

| key | score | route completion |
|---|---|---|
| `hdscore` | HUGSIM's, of the planned trajectory, copied from `eval.json` (`null` without one, e.g. `--skip-native-eval`) | HUGSIM's |
| `hdscore_realized` | TerraTransfer's, of the realized trajectory | HUGSIM's index fraction, `rc`: the ICLR 2027 submission's choice, comparable with HUGSIM's published numbers |
| `hdscore_realized_arclength` | the same | arc-length fraction of the recorded track, `rc_arclength` (arXiv v1/v2) |

The two realized variants differ only by route completion; the index one is usually the higher
(about 1.14x on average). The comfort term is only comparable at the same simulator step: the window is fixed in
seconds, but jerk is a third finite difference at dt, and the paper runs at dt = 0.1 s
(`configs/sim/kinematic_10hz.yaml`). `sim/utils/realized_score.py` has the definitions.

Not every episode gets a score. With `--realized_score`, an episode the agent ended by answering
`None` is recorded as `{"crashed": true}`; offline, `score_realized.py` records one as crashed
when the agent left a crash marker (`agent_crash.txt` from `agents/client.py`, or Pictura's
`pictura_crash.json`), and one with fewer than two frames (an agent that refused at its first step)
as `{"too_short": true}`. An agent that exits outright ends the episode before anything is written,
so that episode has no score file at all. `score_realized.py` keeps a cached score only while it
matches the episode's `data.pkl`, and retries everything else on every pass.

`--outputs scoring` makes an episode write only what scoring needs. Each episode folder normally
carries `ground.ply` (~11 MB) and `scene.ply` (~31 MB), the scene's Gaussians by semantics, a property
of the scene rather than the episode, plus `infos.pkl` and a ~9 MB `video.mp4`. Both scorers only ask
point questions about the ego's box at poses known from `data.pkl` (the planned poses and the
poses displaced by velocity x 0.5 s and 1.0 s for HUGSIM's, the recorded poses and their 0.2 s
propagations to 1 s for the realized one), so `sim/utils/score_corridor.py` keeps the points within
half the footprint diagonal plus 0.5 m of those poses (~3 MB); no other point can matter to a score.
With `--outputs scoring` the env writes no clouds, there is no `infos.pkl`, `--video` defaults to
`off` (an explicit `--video` wins), the cropped clouds are written as `ground.ply` / `scene.ply` (so
`score_realized.py` still rescores the folder), and both scorers run on those in-memory arrays. The
scores are identical in practice to `--outputs full`, the default, which writes everything as before
(the CPU drivable-area path is bitwise identical; the GPU background check is identical up to float32
rounding of a point lying on a box face, since a cuBLAS matmul over the subset can round 1 ulp
differently from one over the full cloud: no difference was seen over 20,736 box queries). HUGSIM's
drivable-area check also looks only at the ground points near each waypoint (`FAST_SCORING` in
`score_calculator.py`, bitwise identical; set it to `False` for the original full-cloud loop).
Together with the cropped clouds, HUGSIM's scorer went from 53.2 s to 3.2 s an episode on average
over 40 past episodes.

To evaluate agents on a whole set, `run_benchmark.py` runs the episodes in parallel, several per
GPU, each with `--outputs scoring --realized_score` (agents in `agents/registry.py` in-process),
skips the ones already scored, and retries a failure once. A start that finds no room on its GPU
(it waits for the job's memory to be free, at most 30 minutes, then releases the GPU and requeues
the job) or an episode that dies out of memory is requeued without using that retry, up to 5 times.
An agent that crashed of the GPU's memory is re-run too (its crash record is dropped, also from a
run made before this existed); on the last attempt the record is kept, with `"gpu_crash": true`. An
exception in the runner itself requeues the job as a failed attempt. An episode folder that is a
symlink (reused from another set) counts as done if its target holds a `realized_score.json` and is
otherwise skipped, never run into. `--out` and `--data` reach the episodes (as `HUGSIM_OUT` /
`HUGSIM_DATA`). `aggregate_benchmark.py` then prints the three HD-Score variants per agent and
dataset, counting an agent's crash (or exit) as 0 and listing the episodes that produced no score:

```bash
python run_benchmark.py --agents uniad vad ltf drivor toad gtrs ztrs gtrs_aug --set official436 --slots 0:14 1:4
python aggregate_benchmark.py --agents uniad vad ltf drivor toad gtrs ztrs gtrs_aug --set official436
```

### Results written before agents/toad

Until `agents/drivor` and `agents/toad` were separated, `--ad drivor` ran DrivoR with CEM
refinement, which is now `toad` ([agents/README.md](agents/README.md)). Episodes of that time sit in
`<set>/<dataset>_drivor/` and are TOAD's. Every `realized_score.json` written since carries the
`--ad` that drove it (`"agent"`), and `run_benchmark.py` and `aggregate_benchmark.py` stop on a
`drivor` record without one rather than take TOAD's drive for DrivoR's. Rename those folders, and
re-point the symlinks a set reuses them through (`official436` episodes linked from
`prechallenge345`, say):

```bash
cd $HUGSIM_OUT
for set in official436 prechallenge345; do
  for d in $set/*_drivor; do [ -e "$d" ] || [ -L "$d" ] && mv "$d" "${d%_drivor}_toad"; done
done
# symlinks whose target names a _drivor folder
find . -type l -lname '*_drivor*' | while read -r l; do t=$(readlink "$l"); ln -sfn "${t/_drivor/_toad}" "$l"; done
```

then aggregate them as `toad`. The 2026-10 baseline campaign is already renamed. These episodes, and any
`toad` episode run before `toad` took TOAD's released CEM settings (October 2026), are TOAD with our
HUGSIM client's CEM settings (`cem_alpha` 0.7, a 5.0 clamp); do not mix them with later `toad` results.

## Environment

The configs interpolate these (all but `HUGSIM_AGENT_CKPTS`, which the agents in `agents/` read
themselves); `closed_loop.py` checks only the ones a given run reads and reports them all at once:

| variable | what it is |
|---|---|
| `HUGSIM_DATA` | the tree holding `HUGSIM/` (3DRealCar, scenes, the two benchmark sets), built by [`prepare_benchmark.py`](#benchmark-sets) |
| `HUGSIM_OUT` | where episode folders are written (`$HUGSIM_OUT/<set>/<dataset>_<agent>/`), or pass `--output_dir`, which is used as given (no `<set>` folder; the agent name is appended to it) |
| `HUGSIM_AD_<NAME>` | an agent's launcher, or pass `--ad_path`; needed only for agents not in `agents/` |
| `HUGSIM_AGENT_CKPTS` | the weights of the agents in `agents/` ([layout](agents/README.md#checkpoints)); `python agents/download_checkpoints.py` fills it |
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

Upstream released all 3DRealCar files, the scenes and the scenarios at [release link](https://huggingface.co/datasets/XDimLab/HUGSIM).
They are released in two stages, around the [RealADSim @ ICCV 2025](https://huggingface.co/spaces/XDimLab/ICCV2025-RealADSim-ClosedLoop) competition,
and the second changed the scenarios, so a benchmark number needs to say which set it is on.

## Benchmark sets

| set | scenarios | what it is |
|---|---|---|
| `official436` (default) | nuScenes 88, PandaSet 127, KITTI-360 113, Waymo 108 | the current release (2025-10-31), the benchmark published HUGSIM numbers are on |
| `prechallenge345` | nuScenes 87, PandaSet 51, KITTI-360 79, Waymo 128 | the release before the held-back scenes went up (2025-07-07), the protocol older papers report |

The sets overlap. The 2025-10-31 release retired 94 of the 345 and moved the actors in 15 of the
rest, so a scenario file of the same name can differ between them: each set has its own copy of every
file. They share one scene tree (all 82 scenes) and the 3DRealCar vehicles.
`prepare_benchmark.py` builds the layout from the two pinned Hugging Face revisions:

```
$HUGSIM_DATA/HUGSIM/3DRealCar/<id>/{gs.pth,wlh.json}
$HUGSIM_DATA/HUGSIM/nusc_map_cache/nusc_trainval/
$HUGSIM_DATA/HUGSIM/ss/scenes/<dataset>/<scene>/
$HUGSIM_DATA/HUGSIM/ss/scenarios/{prechallenge345,official436}/<dataset>/<scenario>.yaml
```

```bash
python prepare_benchmark.py --root /data/hugsim --from-hf --dry-run      # sizes (about 41 GB of scenes), nothing written
python prepare_benchmark.py --root /data/hugsim --from-hf                # download everything
python prepare_benchmark.py --root /data/hugsim --verify                 # counts, scenes, vehicles, repairs
export HUGSIM_DATA=/data/hugsim
```

The release's scenarios do not run as shipped, so the script repairs them (the vehicle asset
`<id>/postprocess/shadow.pth` becomes `<id>`, `ConstantPlanner`'s `max_t` is dropped, `start_velo:
1R` becomes `1`); `--verify` fails on a scenario that still needs one. A one-time migration of an install from before this layout, without downloading it again, is
`--from-install OLD_ROOT` (it links the old install's data and only fetches the two small
`scenarios.zip` files; `--scenarios-zip-pre` / `--scenarios-zip-official` take local copies).

`run_dataset.sh`, `run_all.sh` and `run_sweep.sh` take the set as their last argument
(`official436` by default). `closed_loop.py` reads the set off the scenario's path
(`.../ss/scenarios/<set>/<dataset>/`), so an episode lands in
`$HUGSIM_OUT/<set>/<dataset>_<agent>/<scene>_<mode>` and a scenario that is in both sets never
overwrites the other's run; a scenario outside those folders (`configs/benchmark/`) keeps
`$HUGSIM_OUT/<dataset>_<agent>/`. `python get_score.py <set> <agent>` scores one set: it counts
only the episodes that are scenarios of that set, lists other folders as ignored and scenarios
never run as missing.

Migrating old results: episode folders from the old install's public scenarios belong under
`$HUGSIM_OUT/prechallenge345/`, and the `*_private_*` ones under `$HUGSIM_OUT/official436/`. Do
not move public folders into `official436`: that set retired 94 of the public scenarios and
changed the actors in 15 more, so they would mix other scenarios into its mean.

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
and [NAVSIM](https://github.com/hyzhou404/NAVSIM). LTF, DrivoR, TOAD, GTRS, ZTRS and GTRS-Aug run in
the pixi environment above; UniAD and VAD need the environment in `agents/_mmdet3d/` (one
`pixi install` and two build tasks, [agents/_mmdet3d/README.md](agents/_mmdet3d/README.md)). All of
them read their weights from `HUGSIM_AGENT_CKPTS`, which `python agents/download_checkpoints.py`
fills (about 5.3 GB).

In **closed_loop.py**, we automatically launch autonomous driving algorithms.

Paths in **configs/sim/\*\_base.yaml** are `${oc.env:VAR}` interpolations in this fork, so you
export the roots rather than editing the files: see [Environment](#environment).

``` bash
CUDA_VISIBLE_DEVICES=${sim_cuda} \
python closed_loop.py --scenario_path ./configs/benchmark/${dataset_name}/${scenario_name}.yaml \
            --base_path ./configs/sim/${dataset_name}_base.yaml \
            --camera_path ./configs/sim/${dataset_name}_camera.yaml \
            --kinematic_path ./configs/sim/kinematic.yaml \
            --ad ${method_name: [uniad, vad, ltf, drivor, drivor_paper, toad, gtrs, ztrs, gtrs_aug]} \
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