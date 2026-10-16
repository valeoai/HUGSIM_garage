# Shared by uniad.sh and vad.sh: launch_agent <uniad|vad> <checkpoint relative to $HUGSIM_AGENT_CKPTS> <cuda id> <episode dir>.
#
# closed_loop.py starts the launcher from inside the simulator's environment (the repo's root pixi
# env, or a conda env), whose variables must not reach the mmdet3d environment:
#   * PYTHONPATH / PYTHONHOME / PYTHONSTARTUP: would put the simulator's packages (or a second
#     numpy, torch, mmcv) in front of the pinned ones; the vendored trees' own `tools` and
#     `projects` packages must not be shadowed either. PYTHONNOUSERSITE=1 closes ~/.local too.
#   * LD_LIBRARY_PATH entries inside an environment prefix (the simulator's CUDA 12.4 / libstdc++
#     would be loaded ahead of the ones torch 2.1 + cu121 and mmcv's compiled ops were built
#     against). Entries outside any environment (a cluster's driver directory) are kept.
#   * LD_PRELOAD, CONDA_* / PIXI_* / VIRTUAL_ENV / CUDA_HOME: stale activation state; `pixi run`
#     sets its own for the mmdet3d workspace.
# GPU choice (CUDA_VISIBLE_DEVICES) is deliberately kept.
launch_agent() {
    local agent="$1" ckpt_rel="$2" cuda_id="$3" episode="$4"
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    if [ -z "${HUGSIM_AGENT_CKPTS:-}" ]; then
        echo "[$agent] HUGSIM_AGENT_CKPTS is not set; expected the checkpoint at \$HUGSIM_AGENT_CKPTS/$ckpt_rel" >&2
        return 1
    fi
    local ckpt="$HUGSIM_AGENT_CKPTS/$ckpt_rel"
    if [ ! -f "$ckpt" ]; then
        echo "[$agent] checkpoint not found: $ckpt (expected \$HUGSIM_AGENT_CKPTS/$ckpt_rel)" >&2
        return 1
    fi
    mkdir -p "$episode"
    episode="$(cd "$episode" && pwd)"  # run.py changes into the agent's tree

    local clean_ld="" entry
    local IFS=':'
    for entry in ${LD_LIBRARY_PATH:-}; do
        case "$entry" in
            */.pixi/envs/*|*/envs/*/lib*|"${CONDA_PREFIX:-/nonexistent}"|"${CONDA_PREFIX:-/nonexistent}"/*) ;;
            *) clean_ld="${clean_ld:+$clean_ld:}$entry" ;;
        esac
    done
    unset IFS

    local python
    if [ -n "${HUGSIM_MMDET3D_PYTHON:-}" ]; then
        python=("$HUGSIM_MMDET3D_PYTHON")
    else
        python=("${PIXI:-pixi}" run --frozen --manifest-path "$here/pixi.toml" python)
    fi
    exec env -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP -u LD_PRELOAD \
        -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u CONDA_PROMPT_MODIFIER \
        -u VIRTUAL_ENV -u CUDA_HOME -u CUDA_PATH \
        -u PIXI_ENVIRONMENT_NAME -u PIXI_ENVIRONMENT_PLATFORMS -u PIXI_PROJECT_NAME \
        -u PIXI_PROJECT_ROOT -u PIXI_PROJECT_MANIFEST -u PIXI_PROJECT_VERSION -u PIXI_PROMPT -u PIXI_IN_SHELL \
        PYTHONNOUSERSITE=1 LD_LIBRARY_PATH="$clean_ld" \
        "${python[@]}" "$here/run.py" --agent "$agent" --ckpt "$ckpt" --output "$episode" --device cuda
}
