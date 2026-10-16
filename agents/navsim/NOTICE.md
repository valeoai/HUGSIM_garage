# Third-party code in `agents/navsim/`

HUGSIM_garage itself is MIT-licensed (see the repository `LICENSE`). The files in this folder are
adapted from the Apache-2.0 projects below and keep that licence (`LICENSE` here is the Apache-2.0
text). Each file's docstring names its origin. Changes are limited to what running inference
inside HUGSIM's environment needs: the nuplan devkit, hydra, Lightning, ray and every training-only
code path are removed, hydra configs become explicit dataclasses, pretrained backbones are not
fetched at construction (the checkpoints carry them), and the HUGSIM client scripts are folded into
`policy.py`.

All five agents reach this repository through the HUGSIM clients of
[hyzhou404/NAVSIM](https://github.com/hyzhou404/NAVSIM) (`ltf_e2e.py`, `hugsim/dataparser.py`,
`hugsim/visualize.py`, and the `dynamo_e2e.py`, `gtrs_e2e.py`, `ztrs_e2e.py`, `gtrs_aug_e2e.py`
siblings in the working tree this port was made from), a fork of NAVSIM.

| Source | Licence | Files here |
|---|---|---|
| [NAVSIM](https://github.com/autonomousvision/navsim) (autonomousvision) | Apache-2.0 | `common.py` (data classes), `transfuser/*`, `backbones/vov.py` (VoVNet, via NVlabs/GTRS), `attn.py` (via NVlabs/GTRS) |
| [GTRS](https://github.com/NVlabs/GTRS) (NVlabs) | Apache-2.0 | `gtrs/*`, `gtrs_aug/*`, `backbones/vov.py`, `attn.py` |
| [ZTRS](https://github.com/woxihuanjiangguo/ZTRS) (woxihuanjiangguo) | Apache-2.0 | `ztrs/*` (`ztrs/dp_utils.py` is vendored from ZTRS's diffusion-policy module) |
| [DrivoR](https://github.com/valeoai/DrivoR) (valeoai) | Apache-2.0 | `drivor/*` (the DINOv2 LoRA encoder follows mrabiabrn/robustbev; the decoder blocks use timm layers, Apache-2.0) |
| [hyzhou404/NAVSIM](https://github.com/hyzhou404/NAVSIM) | Apache-2.0 | `policy.py` (the per-agent `parse_raw` and frame conversion), `viz.py` |

Written for this port: `base.py`, `policy.py` (the registry, the device and checkpoint handling),
`run.py`, the launchers `*.sh`, and `../tools/plans.py`.

The checkpoints are not part of this repository; they have their own licences (NAVSIM/OpenScene
data licences, and those of the respective projects).
