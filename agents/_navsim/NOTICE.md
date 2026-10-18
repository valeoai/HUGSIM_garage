# Third-party code in `agents/_navsim/`

HUGSIM_garage itself is MIT-licensed (see the repository `LICENSE`). The files here are shared by the
NAVSIM-family agents (`ltf/`, `drivor/`, `toad/`, `toad_release/`, `gtrs/`, `ztrs/`, `gtrs_aug/`), adapted from the Apache-2.0
projects below, and keep that licence (`LICENSE` here is the Apache-2.0 text). Each file's docstring names
its origin. Changes are limited to what running inference inside HUGSIM's environment needs: the nuplan
devkit, hydra, Lightning, ray and every training-only code path are removed. Each agent's own files,
and where they come from, are in its folder's `NOTICE.md`.

| Source | Licence | Files here |
|---|---|---|
| [NAVSIM](https://github.com/autonomousvision/navsim) (autonomousvision) | Apache-2.0 | `common.py` (data classes; with nuplan-devkit's `TrajectorySampling`), `agent_head.py` (`AgentHead` of TransFuser), `vov.py` (VoVNet, via NVlabs/GTRS), `attn.py` (via NVlabs/GTRS) |
| [GTRS](https://github.com/NVlabs/GTRS) (NVlabs) | Apache-2.0 | `hydra.py` (`HydraBackbone`, `HydraFeatureBuilder`), `vov.py`, `attn.py` |
| [hyzhou404/NAVSIM](https://github.com/hyzhou404/NAVSIM) | Apache-2.0 | `viz.py`, the LTF pieces of `parse.py`, and the LTF client (`ltf_e2e.py`, `hugsim/dataparser.py`) that `policy.py` folds in |
| our NAVSIM checkout (unpublished), derived from [hyzhou404/NAVSIM](https://github.com/hyzhou404/NAVSIM)'s `ltf_e2e.py` and `hugsim/dataparser.py` | Apache-2.0 | the GTRS-family pieces of `parse.py`, and the DrivoR/TOAD and GTRS-family clients (`dynamo_e2e.py`, `gtrs_e2e.py`, `ztrs_e2e.py`, `gtrs_aug_e2e.py`) that `policy.py` folds in |

LTF reaches this repository through hyzhou404/NAVSIM's HUGSIM client (`ltf_e2e.py`). The other agents reach it through the HUGSIM clients we added in our NAVSIM checkout, a working copy of hyzhou404/NAVSIM; none of those clients, nor the DrivoR, TOAD and GTRS-family agent code they run, is in hyzhou404/NAVSIM. The clients are its
`ltf_e2e.py` (their serving loop, line for line) and `hugsim/dataparser.py` (their input parsing) adapted to each
agent, so they are derivative works of those Apache-2.0 files and keep that licence and attribution.

Written for this port: `base.py`, `policy.py` (the spec loading, the device and checkpoint handling),
`run.py`, `__init__.py`, and `../tools/plans.py`.

The checkpoints are not part of this repository; they have their own licences (NAVSIM/OpenScene
data licences, and those of the respective projects).
