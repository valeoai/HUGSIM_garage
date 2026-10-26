#!/usr/bin/env bash
# closed_loop.py launcher for `vavam` (agents/_vavam): $1 = cuda id, $2 = episode folder.
# HUGSIM_AGENT_PYTHON picks the interpreter (default: python), HUGSIM_AGENT_CKPTS the checkpoint
# root ($HUGSIM_AGENT_CKPTS/vavam/ holds this agent's files), HUGSIM_AGENT_DEVICE cuda or cpu,
# HUGSIM_AGENT_SEED seeds the flow's noise.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
episode="$(realpath -m "$2")"  # resolved before the cd
cd "$here/../.."
CUDA_VISIBLE_DEVICES="$1" exec "${HUGSIM_AGENT_PYTHON:-python}" -m agents._vavam.run --agent vavam --output "$episode"
