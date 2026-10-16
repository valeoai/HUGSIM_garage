#!/usr/bin/env bash
# closed_loop.py launcher for UNIAD: $1 = cuda id, $2 = episode folder.
# Needs $HUGSIM_AGENT_CKPTS/uniad/uniad_base_e2e.pth and the pixi environment of agents/_mmdet3d
# (HUGSIM_MMDET3D_PYTHON names another python to use instead; PIXI the pixi binary).
# HUGSIM_AGENT_VIZ=1 writes the agent's own visualisation to <episode>/uniad.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../_mmdet3d/launch_lib.sh"
CUDA_VISIBLE_DEVICES="$1" launch_agent uniad uniad/uniad_base_e2e.pth "$1" "$2"
