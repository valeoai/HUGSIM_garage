# Third-party code in `agents/ltfv6/`

HUGSIM_garage itself is MIT-licensed (see the repository `LICENSE`). `model.py` is adapted from the
Apache-2.0 project below and keeps that licence (`../_navsim/LICENSE` is the Apache-2.0 text). Its
docstring lists the changes: the beartype runtime checks are removed and timm no longer fetches ImageNet
weights at construction (the checkpoint carries them); nothing else. `agent.py` loads every tensor the model has from the checkpoint and refuses a missing one.

| Source | Licence | Files here |
|---|---|---|
| [LEAD](https://github.com/kesai-labs/lead) (formerly autonomousvision/lead), the model release [ln2697/ltfv6-navsim](https://huggingface.co/ln2697/ltfv6-navsim) at `d27ed76` | Apache-2.0 | `model.py` (`ltfv6.py` of the release: config, TransFuser v6 backbone, planning decoder) |
| LEAD, branch `cvpr2026` (`730bc1a2f4`), `3rd_party/navsim_workspace/navsimv1.1` | Apache-2.0 | `agent.py`: the camera and status features of `navsim/agents/transfuser/transfuser_features.py` and the forward of `navsim/agents/carla_transfuser_agent.py` (CARLA -> NAVSIM frame) |

Written for this port: the rest of `agent.py` (loading, device handling), `spec.py` (the HUGSIM parse,
checkpoint and output rule), `launch.sh`. The shared files it imports from `../_navsim/` are listed in
`../_navsim/NOTICE.md`.

The checkpoint and its `config.json` are not part of this repository; they have their own licence
(NAVSIM/OpenScene data licence, and that of the release above, Apache-2.0). They are read from
`$HUGSIM_AGENT_CKPTS/ltfv6/`.
