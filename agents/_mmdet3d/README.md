# UniAD and VAD agents

[UniAD](https://github.com/OpenDriveLab/UniAD) and [VAD](https://github.com/hustvl/VAD) as HUGSIM
agents, with an environment of their own. The code under `agents/uniad/` and `agents/vad/` is vendored from
Zhou et al.'s UniAD_SIM and VAD_SIM; see [uniad/NOTICE.md](../uniad/NOTICE.md) and
[vad/NOTICE.md](../vad/NOTICE.md) for the commits, the licence and what was changed. Both read `rgb` observations (6 nuScenes cameras) and answer a `(6, 2)` plan.

```bash
# once: the environment (python 3.9, torch 2.1.2+cu121, mmcv-full 1.7.2, mmdet 2.28, mmdet3d 1.0.0rc6)
pixi install --manifest-path agents/_mmdet3d/pixi.toml
pixi run --manifest-path agents/_mmdet3d/pixi.toml install-mmcv       # compiles the CUDA ops
pixi run --manifest-path agents/_mmdet3d/pixi.toml install-mmdet3d    # compiles mmdet3d's
CUDA_VISIBLE_DEVICES=0 pixi run --manifest-path agents/_mmdet3d/pixi.toml python agents/_mmdet3d/check_env.py

# checkpoints: $HUGSIM_AGENT_CKPTS/uniad/uniad_base_e2e.pth and $HUGSIM_AGENT_CKPTS/vad/VAD_base.pth
export HUGSIM_AGENT_CKPTS=/path/to/hugsim_agent_ckpts
python closed_loop.py ... --ad uniad    # the shipped configs launch agents/uniad/launch.sh
python closed_loop.py ... --ad vad      # and agents/vad/launch.sh
```

The two compiled packages are tasks, not lock entries: they need torch and nvcc in the
environment before they can build (the root `pixi.toml` has the same constraint, hence its
`no-build-isolation` list and `install-apex` task). `TORCH_CUDA_ARCH_LIST` in `pixi.toml` is
`7.0;7.5;8.0;8.6;9.0` (V100, RTX 2080 Ti, A100, RTX 30xx, H100/H200); edit it, or override it
for one build, to compile for fewer. The CUDA toolchain is conda-forge's, not `/usr/local/cuda`.
The machine needs an NVIDIA driver that runs CUDA 12.1 wheels, nothing else.

## Launchers

`agents/uniad/launch.sh` and `agents/vad/launch.sh` take `$1` = cuda id and `$2` = episode folder, as `closed_loop.py` calls
them. They start `run.py` with the environment's python: `pixi run --manifest-path
agents/_mmdet3d/pixi.toml python` (`PIXI` names the pixi binary), or `$HUGSIM_MMDET3D_PYTHON` if
set. A missing checkpoint stops the launcher with the path it expected. `closed_loop.py` starts
them from inside the simulator's environment, so `launch_lib.sh` first clears what would leak
into this one: `PYTHONPATH`/`PYTHONHOME` (the simulator's packages must not shadow the pinned
ones, nor the vendored `tools` and `projects`), `LD_LIBRARY_PATH` entries that point into a
conda or pixi environment, `LD_PRELOAD`, and the `CONDA_*`, `PIXI_*`, `VIRTUAL_ENV` and
`CUDA_HOME` activation state. `CUDA_VISIBLE_DEVICES` stays.

| variable | |
|---|---|
| `HUGSIM_AGENT_CKPTS` | checkpoint root, required |
| `HUGSIM_MMDET3D_PYTHON` | python of an environment you manage yourself, instead of pixi (the launcher clears `LD_PRELOAD`, and pixi's activation is what preloads the environment's `libstdc++`; a pip-wheel env may need its own, see `pixi.toml`) |
| `HUGSIM_AGENT_VIZ=1` | write the agent's visualisation to `<episode>/uniad` or `<episode>/vad` (frames and `video.mp4`); off by default |
| `HUGSIM_VAD_EVAL=1` | put VAD in `eval()` mode (see below) |

## VAD is stochastic by default

Upstream VAD_SIM never calls `model.eval()`, so GridMask (p=0.7) and dropout stay active at
inference: the same observations give different plans.
That is kept as the default, because it is how upstream's client drove HUGSIM, and so how
the HUGSIM VAD baseline was run. With
`HUGSIM_VAD_EVAL=1` the model is in eval mode and deterministic. On the 16 frames of
`scene-0383_easy` the two modes' plans differ by up to 8 m (first frame) and about 1 m after.
UniAD is in eval mode, as upstream.

## Offline

`plans.py` runs the same step on recorded frames (`agents/tools/record.py` records them; each
`*.pkl` is a pickled `(obs, info)`):

```bash
CUDA_VISIBLE_DEVICES=0 pixi run --manifest-path agents/_mmdet3d/pixi.toml python agents/_mmdet3d/plans.py \
    --agent vad --frames /path/to/frames --out plans.npz --seed 0 [--device cpu] [--max-frames 3]
```

`--seed` seeds `random`, numpy and torch before every frame; use it to make default-mode VAD
repeatable. UniAD on `--device cpu` fails in mmcv's pure-PyTorch deformable attention (the
segmentation head's `MultiScaleDeformableAttention`), in this environment and in the original
one alike; VAD on CPU needs `eval` mode, since its GridMask calls `.cuda()`.

## Files

* `pixi.toml`, `pixi.lock`, `install_mmdet3d.sh`, `check_env.py`: the environment and its GPU check.
* `runner.py`: model build, checkpoint load and the per-frame step of both agents (the old `e2e.py`
  without its FIFO loop). `run.py` serves it through `agents/client.py`; `plans.py` runs it offline.
* `launch_lib.sh`: the launcher code `agents/uniad/launch.sh` and `agents/vad/launch.sh` share.
* `../uniad/`, `../vad/`: the vendored trees, each self-contained (they define the same `projects`
  package, so a process imports exactly one; `runner.py` refuses a second).

## Build notes

`install_mmdet3d.sh` builds mmdet3d 1.0.0rc6
([open-mmlab/mmdetection3d](https://github.com/open-mmlab/mmdetection3d) `47285b3`) with one
patch: the upper bound in `mmdet3d/__init__.py` on the mmcv version is raised from 1.7.0 to 1.7.2.
