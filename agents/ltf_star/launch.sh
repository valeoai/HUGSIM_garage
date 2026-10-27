#!/usr/bin/env bash
# closed_loop.py launcher for the NAVSIM-family agent `ltf_star`: $1 = cuda id, $2 = episode folder.
# HUGSIM_AGENT_PYTHON picks the interpreter (default: python), HUGSIM_AGENT_CKPTS the checkpoint
# root ($HUGSIM_AGENT_CKPTS/ltf_star/ holds this agent's files), HUGSIM_AGENT_DEVICE cuda or cpu,
# HUGSIM_AGENT_VIZ=1 writes the plan overlay and a video.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
episode="$(realpath -m "$2")"  # resolved before the cd
cd "$here/../.."
CUDA_VISIBLE_DEVICES="$1" exec "${HUGSIM_AGENT_PYTHON:-python}" -m agents._navsim.run --agent ltf_star --output "$episode"
