# Third-party code in `agents/gtrs/`

HUGSIM_garage itself is MIT-licensed (see the repository `LICENSE`). The files in this folder are adapted from the Apache-2.0 project(s) below and keep that licence (`../_navsim/LICENSE` is the Apache-2.0 text). Each file's docstring names its origin. Changes are limited to what running inference
inside HUGSIM's environment needs: the nuplan devkit, hydra, Lightning, ray and every training-only
code path are removed, hydra configs become explicit dataclasses, pretrained backbones are not
fetched at construction (the checkpoints carry them), and the HUGSIM client scripts are folded into
`../_navsim/policy.py` and this folder's `spec.py`.

The agent reaches this repository through our own HUGSIM client for it, `gtrs_e2e.py` and its `parse_raw`, kept in our NAVSIM checkout: a working copy of [hyzhou404/NAVSIM](https://github.com/hyzhou404/NAVSIM) to which we added the GTRS-family agents, their configs and their clients. None of these is in hyzhou404/NAVSIM.

| Source | Licence | Files here |
|---|---|---|
| [GTRS](https://github.com/NVlabs/GTRS) (NVlabs) | Apache-2.0 | `config.py`, `model.py`, `agent.py` |
| our NAVSIM checkout (unpublished), derived from [hyzhou404/NAVSIM](https://github.com/hyzhou404/NAVSIM)'s `ltf_e2e.py` and `hugsim/dataparser.py` | Apache-2.0 | `spec.py` (the GTRS-family `parse_raw` is in `../_navsim/parse.py`) |

Written for this port: `spec.py` (the checkpoint, build function and output rule), `launch.sh`. The shared files it imports from `../_navsim/` are listed in `../_navsim/NOTICE.md`. `HydraBackbone` and `HydraFeatureBuilder` (`../_navsim/hydra.py`) and `AgentHead` (`../_navsim/agent_head.py`) were moved out of this folder and `ltf/` so that no agent folder imports another.


The checkpoint is not part of this repository; it has its own licence (NAVSIM/OpenScene data licence, and that of the project above). It is read from `$HUGSIM_AGENT_CKPTS/gtrs/`.
