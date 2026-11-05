# Source on JSC JUPITER before running HUGSIM from this checkout: the roots the sim configs interpolate (as
# hugsim_env.sh) and the simulator's environment, built by create_env.sh in the group's HUGSIM_garage
# (/e/fscratch/e-reg-2026r01-243/HUGSIM_garage/jupiter/create_env.sh), first on PATH.
F=/e/fscratch/e-reg-2026r01-243
export HUGSIM_DATA=${HUGSIM_DATA:-$F/hugsim_data}         # prepare_benchmark.py --from-hf, official436 + prechallenge345
# episodes: $HUGSIM_OUT/<set>/<dataset>_<agent>/ ($F/hugsim_out is not group-writable)
export HUGSIM_OUT=${HUGSIM_OUT:-/e/scratch/e-reg-2026r01-243/$USER/hugsim_out}
# the vector / raster modes' scene sidecars, cached where the scene tree (read-only here) cannot hold them
export HUGSIM_SCENE_EXPORT_CACHE=${HUGSIM_SCENE_EXPORT_CACHE:-/e/scratch/e-reg-2026r01-243/$USER/cache/hugsim_scene_export}
export HUGSIM_AGENT_CKPTS=${HUGSIM_AGENT_CKPTS:-$F/hugsim_agent_ckpts}
export HUGSIM_ENV=${HUGSIM_ENV:-$F/envs/hugsim}
export PATH=$HUGSIM_ENV/bin:$PATH
# tiny-cuda-nn loads libnvrtc.so.12 at import; the aarch64 torch wheel ships it in torch/lib
export LD_LIBRARY_PATH=$HUGSIM_ENV/lib/python3.11/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# The env's hugsim_env is an editable install of the group checkout's sim/; this checkout's comes first, so the
# simulator that runs is this one's.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH=$HERE/sim${PYTHONPATH:+:$PYTHONPATH}
# NUSCENES_RAW: raw nuScenes (v1.0-trainval's tables and maps/ with the map expansion), read only by the 16
# official436 nuScenes scenarios with load_HD_map: true; those scenarios fail without it, the other 420 run.
export NUSCENES_RAW=${NUSCENES_RAW:-$F/nuscenes_raw}
export PYTHONNOUSERSITE=1
unset F HERE
