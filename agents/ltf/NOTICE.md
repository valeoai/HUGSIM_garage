# Third-party code in `agents/ltf/`

HUGSIM_garage itself is MIT-licensed (see the repository `LICENSE`). The files in this folder are adapted from the Apache-2.0 project(s) below and keep that licence (`../LICENSE` is the Apache-2.0 text). Each file's docstring names its origin. Changes are limited to what running inference
inside HUGSIM's environment needs: the nuplan devkit, hydra, Lightning, ray and every training-only
code path are removed, hydra configs become explicit dataclasses, pretrained backbones are not
fetched at construction (the checkpoints carry them), and the HUGSIM client scripts are folded into
`_navsim/policy.py` and this folder's `spec.py`.

The agent reaches this repository through the HUGSIM client of [hyzhou404/NAVSIM](https://github.com/hyzhou404/NAVSIM) (`ltf_e2e.py`, `hugsim/dataparser.py`, `hugsim/visualize.py`), a fork of NAVSIM.

| Source | Licence | Files here |
|---|---|---|
| [NAVSIM](https://github.com/autonomousvision/navsim) (autonomousvision) | Apache-2.0 | `config.py`, `model.py`, `backbone.py`, `features.py`, `agent.py` (Latent TransFuser) |
| [hyzhou404/NAVSIM](https://github.com/hyzhou404/NAVSIM) | Apache-2.0 | `spec.py` (`parse_ltf`, from `hugsim/dataparser.py`) |

Written for this port: `spec.py` (the checkpoint, build function and output rule), `launch.sh`. The shared files it imports from `../_navsim/` are listed in `../_navsim/NOTICE.md`.


The checkpoint is not part of this repository; it has its own licence (NAVSIM/OpenScene data licence, and that of the project above). It is read from `$HUGSIM_AGENT_CKPTS/ltf/`.
