# Baseline agents

The agents HUGSIM is benchmarked with, kept here instead of in separate checkouts (UniAD_SIM,
VAD_SIM, NAVSIM). Each one drives as its original client did: same inputs, same model, same
output. An agent that needs nothing beyond the simulator's own environment runs in it; the rest
bring an environment of their own.

| `--ad` | model | code | environment | in-process |
|---|---|---|---|---|
| `uniad` | UniAD (base, e2e) | `uniad/` | `_mmdet3d/pixi.toml` | no |
| `vad` | VAD (base) | `vad/` | `_mmdet3d/pixi.toml` | no |
| `ltf` | Latent TransFuser | `ltf/` | the simulator's | yes |
| `drivor` | [DrivoR](https://github.com/valeoai/DrivoR) (4 cameras), as its paper evaluated it on HUGSIM | `drivor/` | the simulator's | yes |
| `toad` | [TOAD](https://github.com/valeoai/TOAD): DrivoR with CEM refinement, as the HUGSIM client ran it | `toad/` | the simulator's | yes |
| `toad_release` | TOAD with the CEM settings of its released code | `toad_release/` | the simulator's | yes |
| `gtrs` | GTRS-Dense (VoVNet, 16384 vocabulary) | `gtrs/` | the simulator's | yes |
| `ztrs` | ZTRS (VoVNet, 16384 vocabulary) | `ztrs/` | the simulator's | yes |
| `gtrs_aug` | GTRS-Aug teacher (VoVNet, 8192 vocabulary) | `gtrs_aug/` | the simulator's | yes |

UniAD and VAD are built from mmcv 1.x registries and call its compiled CUDA ops (deformable
attention, DCNv2), so they keep that stack, in a pixi workspace of their own: see
[_mmdet3d/README.md](_mmdet3d/README.md) for the build (about 13 minutes) and the launchers. The
NAVSIM-family agents only ever needed NAVSIM's environment for imports (nuplan-devkit, hydra,
pytorch_lightning), not for anything they compute; their folders hold their inference path alone.
Every agent is a folder named like its `--ad` name and like its checkpoint folder
`$HUGSIM_AGENT_CKPTS/<name>/`, with a `launch.sh` and a `NOTICE.md` (where each file came from). The code
the NAVSIM-family agents share is in `_navsim/` (`NOTICE.md` there too; each agent's `spec.py` is what
distinguishes it), the DrivoR model that `drivor` and `toad` both build in `_drivor/`, the mmdet3d environment and runner of UniAD and VAD in `_mmdet3d/`. No agent folder
imports another, so deleting one leaves the rest working (`_navsim/LICENSE` is the Apache-2.0 text of the
NAVSIM-family code).

`drivor`, `toad` and `toad_release` run the same model and checkpoint, and differ only in their settings:

* **`drivor`** is DrivoR as the DrivoR paper (arXiv 2601.05083, Table 2) evaluated it on HUGSIM: the
  PDMS scorer weights (NC 1, DAC 1, DDC 0, TTC 5, EP 5, comfort 2) and no CEM refinement, as NAVSIM's
  `dynamo_v6_agent_new_scorer_v3_eval_nav3.yaml` ran it.
* **`toad`** is what NAVSIM's `dynamo_e2e.sh` launches (`CONFIG_NAME = "drivor"`, not Dynamo):
  DrivoR with CEM refinement of the scorer's best proposal and the PDMS weights, which is TOAD.
  It is what `drivor` ran before the two were separated. Two of its CEM settings differ from TOAD's
  released code: the update smoothing `cem_alpha` is 0.7 (1.0 there) and sampled controls are
  clamped at 5.0 times the kinematic limits (1.5 there).
* **`toad_release`** is TOAD with its released code's CEM settings (`cem_alpha` 1.0, a 1.5 clamp,
  `cem_control_margin` in `_drivor/config.py`); the rest, checkpoint included, is `toad`'s.

`dynamo` stays an external launcher (`HUGSIM_AD_DYNAMO`).

## Running one

```bash
export HUGSIM_AGENT_CKPTS=/path/to/hugsim_agent_ckpts      # see Checkpoints
python closed_loop.py --scenario_path ... --base_path ./configs/sim/nuscenes_base.yaml \
    --camera_path ./configs/sim/nuscenes_camera.yaml --kinematic_path ./configs/sim/kinematic.yaml \
    --ad gtrs --output_dir /where/episodes/go/nuscenes_
```

The shipped configs launch each agent from here (`base.<ad>_path:
${oc.env:HUGSIM_AD_<NAME>,agents/...}`); an exported `HUGSIM_AD_<NAME>` still replaces the
launcher, so unset it to use these. The NAVSIM-family launchers run the simulator's own
interpreter (`closed_loop.py` passes it as `HUGSIM_AGENT_PYTHON`, unless that is already set);
the UniAD and VAD launchers clear the simulator's environment and start their own. An agent
that exits early (a missing checkpoint, an unbuilt environment) ends the episode with an error
naming its log, rather than leaving the simulator waiting on a pipe.

### In-process

The seven agents that run in the simulator's environment can also be called in its process,
with no launcher and no pipes:

```bash
python closed_loop.py ... --ad ltf --in_process
```

`closed_loop.py` imports the step function from `registry.py` and calls it each step, on the
device the agent would pick over the pipes (`cuda`, or `HUGSIM_AGENT_DEVICE`). On LTF,
scene-0383 easy, a step costs 30 to 75 ms instead of 46 to 117 ms over the pipes (no 16 MB pickle
of six cameras per step, one CUDA context instead of two), and a crash cannot leave the episode
waiting on a pipe. The agent runs on the simulator's GPU (`--ad_cuda` does not apply), and its
output goes to the simulator's log rather than `output.txt`. Both ways give the same episode:
`data.pkl` is identical frame for frame (see the iLQR note under Checking a port).

## Checkpoints

Every agent reads its weights from `$HUGSIM_AGENT_CKPTS`, laid out as:

| file | from |
|---|---|
| `uniad/uniad_base_e2e.pth` | [OpenDriveLab/UniAD v1.0.1 release](https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth) |
| `vad/VAD_base.pth` | VAD-Base from [hustvl/VAD](https://github.com/hustvl/VAD)'s README |
| `ltf/ltf_seed_0.ckpt` | [autonomousvision/navsim_baselines, `ltf/`](https://huggingface.co/autonomousvision/navsim_baselines/tree/main/ltf) |
| `drivor/drivor_Nav1_25epochs.pth` | [valeoai/DrivoR's `model_weights` release](https://github.com/valeoai/DrivoR/releases/download/model_weights/drivor_Nav1_25epochs.pth) (the NAVSIM-v1 model; sha256 `e1a678f2…`). NAVSIM's `ckpts/final/nav3.ckpt`, which its `dynamo_e2e.py` loaded, holds the same 467 tensors in another container |
| `toad/drivor_Nav1_25epochs.pth`, `toad_release/drivor_Nav1_25epochs.pth` | the same file as `drivor/drivor_Nav1_25epochs.pth` |
| `gtrs/gtrs_dense_vov.ckpt` | [Zzxxxxxxxx/gtrs](https://huggingface.co/Zzxxxxxxxx/gtrs/blob/main/gtrs_dense_vov.ckpt) |
| `gtrs/16384.npy` | the trajectory vocabulary, `traj_final/16384.npy` in [NVlabs/GTRS](https://github.com/NVlabs/GTRS) |
| `ztrs/ztrs_vov.ckpt` | [Zzxxxxxxxx/ZTRS](https://huggingface.co/Zzxxxxxxxx/ZTRS/blob/main/ztrs_vov.ckpt) |
| `ztrs/16384.npy` | as `gtrs/16384.npy` |
| `gtrs_aug/gtrs_aug_vov.ckpt` | [Zzxxxxxxxx/gtrs](https://huggingface.co/Zzxxxxxxxx/gtrs/blob/main/gtrs_aug_vov.ckpt) |
| `gtrs_aug/8192.npy` | `traj_final/8192.npy` in NVlabs/GTRS |

The checkpoints carry their backbones' weights, so nothing else is downloaded at start-up (no
DINOv2 or DD3D files, no Hugging Face fetch). A missing file stops the agent with the path it
expected. Symlinks into existing checkouts work.

## Behaviour kept from the original clients

These are how the original clients drove HUGSIM, so they are kept:

* **The NAVSIM-family agents always get the "straight" command** (one-hot `[0, 1, 0, 0]`); the
  original clients commented out `info['command']`. UniAD and VAD read it.
* **VAD runs in training mode**: upstream never called `model.eval()`, so GridMask and dropout
  stay active and its plans are stochastic. `HUGSIM_VAD_EVAL=1` turns that off.
* **TOAD's CEM refinement samples noise** and the original client never seeded it.
  `HUGSIM_AGENT_SEED` seeds it (`seed + step` before each step, on a forked RNG, so in-process
  it leaves the simulator's own random stream alone).
* **LTF and DrivoR/TOAD ran on the CPU** (the model was never moved to the GPU). They now run on
  the GPU with TF32 off for their step, which keeps their plans within 2e-5 m of the CPU ones;
  `HUGSIM_AGENT_DEVICE=cpu` runs the old path.

What changed: a step that fails is answered `None` (the episode ends) instead of leaving
`closed_loop.py` blocked on a pipe, and each agent's own visualisation is off unless
`HUGSIM_AGENT_VIZ=1` (it then writes `<episode>/<agent>/`, frames and `video.mp4`, as before).

| variable | |
|---|---|
| `HUGSIM_AGENT_CKPTS` | checkpoint root, required |
| `HUGSIM_AGENT_PYTHON` | python for the NAVSIM-family launchers (default: the simulator's) |
| `HUGSIM_MMDET3D_PYTHON` | python of an environment you manage, for UniAD and VAD (default: `pixi run` in `_mmdet3d/`); `PIXI` names the pixi binary |
| `HUGSIM_AGENT_DEVICE` | `cuda` (default) or `cpu`, NAVSIM family |
| `HUGSIM_AGENT_SEED` | seed the step, NAVSIM family |
| `HUGSIM_VAD_EVAL=1` | VAD in eval mode |
| `HUGSIM_AGENT_VIZ=1` | each agent's own visualisation |

## Writing an agent

`client.py` is the AD side of `closed_loop.py`'s protocol, standard library and numpy only:

```python
from agents.client import serve
serve(episode_dir, step)   # step(obs, info) -> (N, 2) plan, (right, forward) m, 0.5 s apart
```

`serve` answers every request (`None` when `step` raises or the request cannot be unpickled,
with the traceback in `<episode>/agent_crash.txt`) and returns after the simulator's `'Done'`,
or when the simulator closes the pipe without writing. For what `serve` cannot cover (an agent
that dies before serving, or crashes outright), `closed_loop.py` watches the launcher's process. An agent that runs in the simulator's environment can also be
listed in `registry.py`, which makes `--in_process` available to it. A NAVSIM-family agent is one
folder plus its name in `_navsim/__init__.py`'s `AGENTS` (the registry derives from it); its default
launcher also takes a `<name>_path` line in the `configs/sim/*_base.yaml` (or a `HUGSIM_AD_<NAME>`).

## Checking a port

`tools/record.py` is an agent that saves every `(obs, info)` the simulator sends and drives a
straight plan; `tools/plans.py` (NAVSIM family) and `_mmdet3d/plans.py` replay those frames through
an agent offline. That is how each agent here was checked against its original client: on 16
frames of nuScenes scene-0383 easy, GTRS, ZTRS, GTRS-Aug, UniAD and VAD give the same plans to the
bit, LTF and TOAD within 2e-5 m, with the same proposal chosen on every frame.

```bash
HUGSIM_RECORD_OUT=/tmp/frames HUGSIM_RECORD_STEPS=16 python closed_loop.py ... --ad record --ad_path agents/tools/record.sh
python agents/tools/plans.py --agent ltf --frames /tmp/frames --out ltf.npz
```

Comparing whole episodes needs one more step: the iLQR tracker stops on a 50 ms wall-clock
budget (`max_solve_time` in `sim/ilqr/lqr.py`), so the same plan can give a slightly different
control depending on the machine's load, and two runs of one agent drift apart from the first
step. Lift the budget for the comparison (`solver_params = replace(solver_params,
max_solve_time=None)` and clear `_solvers`); with it lifted, the FIFO and in-process runs above
are identical.
