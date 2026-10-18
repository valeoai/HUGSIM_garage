#!/usr/bin/env bash
# closed_loop.py launcher for VAD: $1 = cuda id, $2 = episode folder.
# Needs $HUGSIM_AGENT_CKPTS/vad/VAD_base.pth and the pixi environment of agents/_mmdet3d
# (HUGSIM_MMDET3D_PYTHON names another python to use instead; PIXI the pixi binary).
# HUGSIM_AGENT_VIZ=1 writes the agent's own visualisation to <episode>/vad. HUGSIM_VAD_EVAL=1 puts the
# model in eval() mode; upstream never does, so by default GridMask and dropout stay on.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../_mmdet3d/launch_lib.sh"
CUDA_VISIBLE_DEVICES="$1" launch_agent vad vad/VAD_base.pth "$1" "$2"
