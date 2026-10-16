# Third-party code in `agents/_navsim/`

HUGSIM_garage itself is MIT-licensed (see the repository `LICENSE`). The files here are shared by the
NAVSIM-family agents (`ltf/`, `drivor/`, `gtrs/`, `ztrs/`, `gtrs_aug/`), adapted from the Apache-2.0
projects below, and keep that licence (`../LICENSE` is the Apache-2.0 text). Each file's docstring names
its origin. Changes are limited to what running inference inside HUGSIM's environment needs: the nuplan
devkit, hydra, Lightning, ray and every training-only code path are removed. Each agent's own files,
and where they come from, are in its folder's `NOTICE.md`.

| Source | Licence | Files here |
|---|---|---|
| [NAVSIM](https://github.com/autonomousvision/navsim) (autonomousvision) | Apache-2.0 | `common.py` (data classes; with nuplan-devkit's `TrajectorySampling`), `agent_head.py` (`AgentHead` of TransFuser), `vov.py` (VoVNet, via NVlabs/GTRS), `attn.py` (via NVlabs/GTRS) |
| [GTRS](https://github.com/NVlabs/GTRS) (NVlabs) | Apache-2.0 | `hydra.py` (`HydraBackbone`, `HydraFeatureBuilder`), `vov.py`, `attn.py` |
| [hyzhou404/NAVSIM](https://github.com/hyzhou404/NAVSIM) | Apache-2.0 | `parse.py` (the `parse_raw` pieces the agents share), `viz.py`, and the client scripts that `policy.py` folds in |

All five agents reach this repository through the HUGSIM clients of hyzhou404/NAVSIM (a fork of NAVSIM).

Written for this port: `base.py`, `policy.py` (the spec loading, the device and checkpoint handling),
`run.py`, `__init__.py`, and `../tools/plans.py`.

The checkpoints are not part of this repository; they have their own licences (NAVSIM/OpenScene
data licences, and those of the respective projects).
