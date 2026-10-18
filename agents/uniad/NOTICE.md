# Provenance of the vendored UniAD agent

`agents/uniad/` is a copy of the HUGSIM closed-loop client of UniAD.
HUGSIM client (upstream of this copy): [hyzhou404/UniAD_SIM](https://github.com/hyzhou404/UniAD_SIM), commit `5fb279e`.
Model code originates from [OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD). Licence: Apache License 2.0 (`LICENSE` here, unchanged).

What was copied is the working tree of the local fork, not the commit: it carries uncommitted
patches that adapt the code to mmdet 2.26+ / mmdet3d 1.0.0rc6. They are part of the copy.

## Local patches (present in the copy, absent from the upstream commit)

* `projects/mmdet3d_plugin/__init__.py` and `core/bbox/match_costs/__init__.py` import `DiceCost`
  from `mmdet.core.bbox.match_costs`; `core/bbox/match_costs/match_cost.py` has its own `DiceCost`
  commented out. mmdet registers the same name, and a double registration raises.
* `projects/mmdet3d_plugin/losses/dice_loss.py`: `@LOSSES.register_module(force=True)`.
* The runtime data moved from `data_temp/` to `data/` (`data/others/motion_anchor_infos_mode6.pkl`,
  `data/nus_vis_pose_info.pkl`) and `tools/closeloop/visualizer.py` reads the latter from there.

## Changes made when vendoring into HUGSIM_garage

* `tools/closeloop/e2e.py` is gone. Its model build, checkpoint load, forward and output
  selection are `../_mmdet3d/runner.py`; its FIFO loop is `../client.py`; the shell wrapper
  (`tools/e2e.sh`, which hard-coded a path and `conda run`) is `launch.sh`.
* `tools/closeloop/dataparser.py`: `parse_raw` takes `device='cuda'` where the upstream called
  `.cuda()`, so the same code runs on CPU (`../_mmdet3d/plans.py --device cpu`). Nothing else is changed.
* `tools/closeloop/visualizer.py`: `draw_bev` and `draw_proj` and their imports (open3d, mediapy,
  nuscenes `Box`) are removed; nothing called them, and they were the only users of open3d.
* Pruned, as whole files only, after tracing which files the inference path imports (with and
  without `HUGSIM_AGENT_VIZ=1`): training and test drivers (`uniad/apis`, `bevformer/`, training hooks), dataset samplers and evaluation
  helpers that nothing imports, the configs outside the `_base_` chain of `base_e2e.py`, `tools/`
  other than the closed-loop and visualiser code. No model code is edited.
* Not copied: the checkpoint (`$HUGSIM_AGENT_CKPTS/uniad/`), docs, data converters, `data_template.pth`.

The environment (`pixi.toml`, `install_mmdet3d.sh`) is shared with the other mmdet3d agent and
documented in `../_mmdet3d/README.md`.
