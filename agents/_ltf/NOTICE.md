# Third-party code in `agents/_ltf/`

HUGSIM_garage itself is MIT-licensed (see the repository `LICENSE`). The files in this folder are adapted from the Apache-2.0 project(s) below and keep that licence (`../_navsim/LICENSE` is the Apache-2.0 text). Each file's docstring names its origin. Changes are limited to what running inference
inside HUGSIM's environment needs: the nuplan devkit, hydra, Lightning, ray and every training-only
code path are removed, hydra configs become explicit dataclasses, pretrained backbones are not
fetched at construction (the checkpoints carry them), and the HUGSIM client scripts are folded into
`../_navsim/policy.py` and this folder's `spec.py`.

The agent reaches this repository through the HUGSIM client of [hyzhou404/NAVSIM](https://github.com/hyzhou404/NAVSIM) (`ltf_e2e.py`, `hugsim/dataparser.py`, `hugsim/visualize.py`), a fork of NAVSIM.

| Source | Licence | Files here |
|---|---|---|
| [NAVSIM](https://github.com/autonomousvision/navsim) (autonomousvision) | Apache-2.0 | `config.py`, `model.py`, `backbone.py`, `features.py`, `agent.py` (Latent TransFuser) |
| [hyzhou404/NAVSIM](https://github.com/hyzhou404/NAVSIM) | Apache-2.0 | `client.py` (`parse_ltf`, from `hugsim/dataparser.py`) |
| [BeyondDrive](https://github.com/wjl2244/BeyondDrive) (wjl2244) | Apache-2.0 | `model.py`: the delta variant of `TrajectoryHead`, `cumsum_traj` and its statistics (from `transfuser_model.py` and `utils.py`) |

Written for this port: the rest of `client.py` (the build function and output rule) and `__init__.py`. The shared files it imports from `../_navsim/` are listed in `../_navsim/NOTICE.md`.


The checkpoints are not part of this repository; each agent folder's `NOTICE.md` says where its own is read from.
