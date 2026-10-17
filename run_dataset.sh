#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 <agent_name> <dataset> <sim_cuda> <ad_cuda> [scenario_order] [set]"
    echo "  scenario_order: forward (default) or reverse -- run two workers from opposite"
    echo "                  ends of the list; finished scenes are skipped via their eval.json."
    echo "  set:            official436 (default) or prechallenge345 (README.md, Benchmark sets)"
    echo "  HUGSIM_ARGS:    env var, extra closed_loop.py arguments split on whitespace, but not"
    echo "                  --ad/--base_path/--scenario_path/--output_dir. Flags that change the"
    echo "                  episode (--obs_mode, --kinematic_path, --raster_*) need an agent name of"
    echo "                  its own (README.md, Running a variant)."
}

if [ -z "${1-}" ]; then
    echo "Error: Agent name must be provided as the first argument."
    usage; exit 1
fi
AGENT=${1}

if [ -z "${2-}" ]; then
    echo "Error: Dataset must be provided as the second argument."
    usage; exit 1
fi
DATASET=${2}

if [ -z "${3-}" ] || [ -z "${4-}" ]; then
    echo "Error: sim_cuda and ad_cuda must be provided as the third and fourth arguments."
    usage; exit 1
fi
sim_cuda=${3}
ad_cuda=${4}
scenario_order=${5:-forward}
scenario_set=${6:-official436}

if [ "${scenario_order}" != "forward" ] && [ "${scenario_order}" != "reverse" ]; then
    echo "Error: scenario_order must be either 'forward' or 'reverse'."
    usage; exit 1
fi

# The scenarios come from, and the episodes go to, the same roots closed_loop.py reads.
: "${HUGSIM_DATA:?HUGSIM_DATA is not set: the tree prepare_benchmark.py built (README.md, Environment)}"
: "${HUGSIM_OUT:?HUGSIM_OUT is not set: where episode folders are written (README.md, Environment)}"

read -r -d '' -a hugsim_args <<< "${HUGSIM_ARGS-}" || true
# The skip below looks for eval.json in the folder --ad, --base_path, --scenario_path and --output_dir
# pick, so HUGSIM_ARGS must not set them, nor any abbreviation argparse would take for one. Any prefix
# of three characters or more is refused: it either names one of them or is ambiguous, whatever
# other flags closed_loop.py gains or loses.
for arg in ${hugsim_args[@]+"${hugsim_args[@]}"}; do
    flag=${arg%%=*}
    for opt in --ad --base_path --scenario_path --output_dir; do
        if [[ "${#flag}" -gt 2 && "${opt}" == "${flag}"* ]]; then
            echo "Error: HUGSIM_ARGS must not set ${opt} (${arg}): run_dataset.sh picks the episode folder (and its eval.json skip) from its own arguments. Run a variant under an agent name of its own, with HUGSIM_AD_<NAME>=<launcher path> (a dynamo* name: --ad_path in HUGSIM_ARGS) and its flags in HUGSIM_ARGS (README.md, Running a variant)."
            usage; exit 1
        fi
    done
done

case "${scenario_set}" in
    official436|prechallenge345) ;;
    *)
        echo "Error: set must be either 'official436' or 'prechallenge345'."
        usage; exit 1
        ;;
esac
scenario_dir="${HUGSIM_DATA}/HUGSIM/ss/scenarios/${scenario_set}/${DATASET}"
base_cfg="./configs/sim/${DATASET}_base.yaml"
# closed_loop.py files an episode under the set its scenario is in: $HUGSIM_OUT/<set>/<dataset>_<agent>/.
out_group="${scenario_set}/${DATASET}_${AGENT}"

echo "--- Processing dataset: ${DATASET} (${scenario_set}, ${scenario_order}) ---"

# Check if scenario files exist to avoid errors with glob
if ! ls "${scenario_dir}"/*.yaml 1> /dev/null 2>&1; then
    echo "Warning: No .yaml scenario files found in ${scenario_dir}. Skipping."
    exit 0
fi

if [ "${scenario_order}" = "reverse" ]; then
    mapfile -t scenario_files < <(printf '%s\n' "${scenario_dir}"/*.yaml | sort -r)
else
    mapfile -t scenario_files < <(printf '%s\n' "${scenario_dir}"/*.yaml | sort)
fi

for cfg in "${scenario_files[@]}"; do
    echo "Running scenario: ${cfg}"

    # --- extract scene_name and mode from yaml to build output dir path ---
    scene_name=$(grep "scene_name:" "$cfg" | sed 's/scene_name:[ ]*//' | sed 's/ //g' | sed "s/'//g")
    mode=$(grep "mode:" "$cfg" | sed 's/mode:[ ]*//')

    # --- build output dir path ---
    out_dir="${HUGSIM_OUT}/${out_group}/${scene_name}_${mode}"

    # --- skip only if eval output already exists ---
    if [ -f "$out_dir/eval.json" ]; then
        echo "⚠️  eval.json already exists, skipping: $out_dir"
        continue
    fi

    echo "→ Running simulation, output will be saved to: $out_dir"

    CUDA_VISIBLE_DEVICES=${sim_cuda} \
    python closed_loop.py --scenario_path "${cfg}" \
                        --base_path "${base_cfg}" \
                        --camera_path "./configs/sim/${DATASET}_camera.yaml" \
                        --kinematic_path "./configs/sim/kinematic.yaml" \
                        --ad "${AGENT}" \
                        --ad_cuda ${ad_cuda} \
                        ${hugsim_args[@]+"${hugsim_args[@]}"}
done
