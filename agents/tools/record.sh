#!/usr/bin/env bash
# closed_loop.py launcher for record.py: $1 = cuda id (unused), $2 = episode folder.
# HUGSIM_RECORD_OUT is where the frames go; HUGSIM_RECORD_STEPS how many (default 12).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HUGSIM_AGENT_PYTHON:-python}" "$here/record.py" --output "$2" \
    --out "${HUGSIM_RECORD_OUT:?set HUGSIM_RECORD_OUT to where the frames go}" --steps "${HUGSIM_RECORD_STEPS:-12}"
