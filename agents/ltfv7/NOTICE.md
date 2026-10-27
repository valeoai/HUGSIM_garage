# Code in `agents/ltfv7/`

HUGSIM_garage itself is MIT-licensed (see the repository `LICENSE`). This folder holds only what was
written for this port: `spec.py` (BeyondDrive's released LTFv7 checkpoint, with its trajectory head) and `launch.sh`. The Latent TransFuser model it builds, adapted
from Apache-2.0 code, is in `../_ltf/`; see `../_ltf/NOTICE.md` for where each file comes from.

The checkpoint is not part of this repository; it has its own licence (NAVSIM/OpenScene data licence, and that of wjl2244/BeyondDrive, Apache-2.0). It is read from
`$HUGSIM_AGENT_CKPTS/ltfv7/`.
