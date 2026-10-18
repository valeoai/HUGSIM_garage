# Third-party code in `agents/_drivor/`

HUGSIM_garage itself is MIT-licensed (see the repository `LICENSE`). The files in this folder are adapted from the Apache-2.0 project(s) below and keep that licence (`../_navsim/LICENSE` is the Apache-2.0 text). Each file's docstring names its origin. Changes are limited to what running inference
inside HUGSIM's environment needs: the nuplan devkit, hydra, Lightning, ray and every training-only
code path are removed, hydra configs become explicit dataclasses, pretrained backbones are not
fetched at construction (the checkpoints carry them), and the HUGSIM client scripts are folded into
`../_navsim/policy.py` and this folder's `client.py`. The DrivoR model is shared by `../drivor/` (DrivoR) and
`../toad/` (TOAD, DrivoR with CEM refinement), which differ only in their settings. Apart from the CEM
refinement, the files match valeoai/DrivoR's inference code up to a `None` guard on missing cameras,
the remapping of legacy checkpoint key prefixes, and removed prints and training code.

The agent reaches this repository through the HUGSIM client of [hyzhou404/NAVSIM](https://github.com/hyzhou404/NAVSIM) (`dynamo_e2e.py` and its `parse_raw`), a fork of NAVSIM.

| Source | Licence | Files here |
|---|---|---|
| [DrivoR](https://github.com/valeoai/DrivoR) (valeoai), at `fc6e5aa` | Apache-2.0 | `config.py`, `model.py`, `agent.py`, `features.py`, `scorer.py`, `transformer_decoder.py`, `mlp.py`, `encoder.py` (the DINOv2 LoRA encoder follows mrabiabrn/robustbev), `layers_timm.py` (timm layers, Apache-2.0) |
| [TOAD](https://github.com/valeoai/TOAD) (valeoai), at `cfa88e0` | see that repository | the CEM refinement in `model.py` (`cem_refine` and its helpers) and the `cem_*` fields of `config.py`, as hyzhou404/NAVSIM's copy ran them: `cem_alpha` 0.7 and a control clamp of 5.0 times the kinematic limits, where TOAD's released `drivor_cem_portable/` uses 1.0 and 1.5 |
| [hyzhou404/NAVSIM](https://github.com/hyzhou404/NAVSIM) | Apache-2.0 | `client.py` (`parse_drivor`, from `dynamo_e2e.py`) |

Written for this port: `client.py` (`drivor_choice`). The shared files it imports from `../_navsim/` are listed in `../_navsim/NOTICE.md`.


The checkpoint is not part of this repository; it has its own licence (NAVSIM/OpenScene data licence, and that of the project above). It is read from `$HUGSIM_AGENT_CKPTS/drivor/` and `$HUGSIM_AGENT_CKPTS/toad/`.
