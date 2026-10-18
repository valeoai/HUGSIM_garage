# Code in `agents/toad_release/`

HUGSIM_garage itself is MIT-licensed (see the repository `LICENSE`). This folder holds only what was
written for this port: `spec.py` (TOAD as released in valeoai/TOAD, DrivoR with CEM refinement with the CEM settings of its `drivor_cem_portable/`: the checkpoint, the settings, the build function and the
output rule) and `launch.sh`. The DrivoR model it builds, adapted from Apache-2.0 code, is in
`../_drivor/`; see `../_drivor/NOTICE.md` for where each file comes from.

The checkpoint is not part of this repository; it has its own licence (NAVSIM/OpenScene data licence,
and that of valeoai/DrivoR). It is read from `$HUGSIM_AGENT_CKPTS/toad_release/`.
