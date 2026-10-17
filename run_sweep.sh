#!/bin/bash
# Zero-shot sweep of one agent over all 4 HUGSIM datasets
#   -> $HUGSIM_OUT/{set}/{dataset}_{agent}
# run_all.sh uses `set -e`, so one failing scenario aborts the sweep; retry passes resume
# because completed scenes are skipped via their eval.json.

if [ -z "${1-}" ]; then
    echo "Usage: $0 <agent_name> [sim_cuda] [ad_cuda] [set]"
    echo "  set: official436 (default) or prechallenge345"
    exit 1
fi
AGENT=${1}
sim_cuda=${2:-0}
ad_cuda=${3:-0}
scenario_set=${4:-official436}

cd "$(dirname "$(readlink -f "$0")")" || exit 1
# pixi: $PIXI if set, else the one on PATH.
PIXI=${PIXI:-pixi}
if ! command -v "${PIXI}" > /dev/null; then
    echo "Error: pixi not found: put it on PATH or set PIXI=/path/to/pixi"
    exit 1
fi
# The AD side (NAVSIM/${AGENT}_e2e.sh) is spawned by closed_loop.py and calls `pixi run`,
# so pixi must be on PATH in this non-interactive environment as well.
case "${PIXI}" in */*) export PATH="$(dirname "${PIXI}"):${PATH}" ;; esac

for pass in $(seq 1 30); do
    echo "================ PASS ${pass} ${AGENT} $(date -Is) ================"
    if ${PIXI} run bash run_all.sh "${AGENT}" "${sim_cuda}" "${ad_cuda}" forward "${scenario_set}"; then
        echo "================ SWEEP COMPLETE $(date -Is) ================"
        exit 0
    fi
    echo "!!! pass ${pass} aborted, retrying in 30s"
    command sleep 30
done
echo "!!! giving up after 30 passes"
exit 1
