# Provenance of the vendored UniAD and VAD agents

`uniad/` and `vad/` are copies of the HUGSIM closed-loop clients of two end-to-end driving
models, both under the Apache License 2.0 (`uniad/LICENSE`, `vad/LICENSE`, unchanged).

| tree | HUGSIM client (upstream of this copy) | commit | model code originates from |
|---|---|---|---|
| `uniad/` | [hyzhou404/UniAD_SIM](https://github.com/hyzhou404/UniAD_SIM) | `5fb279e` | [OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD) |
| `vad/` | [hyzhou404/VAD_SIM](https://github.com/hyzhou404/VAD_SIM) | `0c74967` | [hustvl/VAD](https://github.com/hustvl/VAD) (VAD's BEVFormer parts come from [fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer)) |

What was copied is the working tree of the local forks, not the commits: each carries uncommitted
patches that adapt the code to mmdet 2.26+ / mmdet3d 1.0.0rc6. They are part of the copy.

## Local patches (present in the copy, absent from the upstream commits)

UniAD_SIM:

* `projects/mmdet3d_plugin/__init__.py` and `core/bbox/match_costs/__init__.py` import `DiceCost`
  from `mmdet.core.bbox.match_costs`; `core/bbox/match_costs/match_cost.py` has its own `DiceCost`
  commented out. mmdet registers the same name, and a double registration raises.
* `projects/mmdet3d_plugin/losses/dice_loss.py`: `@LOSSES.register_module(force=True)`.
* The runtime data moved from `data_temp/` to `data/` (`data/others/motion_anchor_infos_mode6.pkl`,
  `data/nus_vis_pose_info.pkl`) and `tools/closeloop/visualizer.py` reads the latter from there.

VAD_SIM:

* `projects/mmdet3d_plugin/core/bbox/structures/lidar_box3d.py`:
  `from mmdet3d.ops import points_in_boxes_all as points_in_boxes_gpu`.

## Changes made when vendoring into HUGSIM_garage

* `tools/closeloop/e2e.py` is gone. Its model build, checkpoint load, forward and output
  selection are `../runner.py`; its FIFO loop is `agents/client.py`; the shell wrappers
  (`tools/e2e.sh`, which hard-coded a path and `conda run`) are `../uniad.sh`, `../vad.sh`.
* `tools/closeloop/dataparser.py`: `parse_raw` takes `device='cuda'` where the upstream called
  `.cuda()`, so the same code runs on CPU (`plans.py --device cpu`). Nothing else is changed.
* `tools/closeloop/visualizer.py`: `draw_bev` and `draw_proj` and their imports (open3d, mediapy,
  nuscenes `Box`) are removed; nothing called them, and they were the only users of open3d.
* Pruned, as whole files only, after tracing which files the inference path imports (with and
  without `HUGSIM_AGENT_VIZ=1`): training and test drivers (`uniad/apis`, `VAD/apis`, `bevformer/`,
  training hooks), dataset samplers and evaluation helpers that nothing imports, the configs
  outside the `_base_` chain of `base_e2e.py` / `VAD_base_e2e.py`, `tools/` other than the
  closed-loop and visualiser code. No model code is edited.
* Not copied: checkpoints (`$HUGSIM_AGENT_CKPTS`), docs, data converters, `data_template.pth`.

## Environment

`pixi.toml` pins the stack; `install_mmdet3d.sh` builds mmdet3d 1.0.0rc6
([open-mmlab/mmdetection3d](https://github.com/open-mmlab/mmdetection3d) `47285b3`) with one
patch: the upper bound in `mmdet3d/__init__.py` on the mmcv version is raised from 1.7.0 to 1.7.2.
