# Code in `agents/drivor_eco/`

`spec.py` is `agents/drivor/spec.py` (DrivoR, default scorer weights, no CEM) with `eco=True`: each plan
goes through ECO, **our reimplementation** of arXiv 2609.31383 (`../_eco/`, no code was released).
`launch.sh` is `agents/drivor/launch.sh` for this name. The checkpoint is DrivoR's, read from
`$HUGSIM_AGENT_CKPTS/drivor_eco/` (a link to `drivor/`). The ECO paper's DrivoR is the SimScale-trained one;
this is the released NAVSIM-v1 checkpoint.
