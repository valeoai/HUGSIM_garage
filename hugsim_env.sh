# Roots the sim configs interpolate. Source this, or set them yourself.
# The configs hold no absolute paths, so each machine points them at its own tree.
# Must be a tree built by prepare_benchmark.py (HUGSIM/3DRealCar, HUGSIM/ss/scenes, HUGSIM/ss/scenarios/<set>);
# an install from before that layout must be migrated first (prepare_benchmark.py --from-install).
export HUGSIM_DATA=/home/yyin5/mack/datasets_mack_raw
export HUGSIM_OUT=/home/yyin5/mack/yyin5/HUGSIM/outputs/benchmark_drivor_mpc
export NUSCENES_RAW=/home/yyin5/scania/datasets_scania_raw/nuscenes
# The baseline agents under agents/ read their weights from here (agents/README.md, Checkpoints).
export HUGSIM_AGENT_CKPTS=/home/yyin5/workspace/hugsim_agent_ckpts
# uniad, vad, ltf, drivor, gtrs, ztrs and gtrs_aug launch from agents/ by default; a
# HUGSIM_AD_<NAME> exported here would replace that launcher. The agents below live elsewhere.
export HUGSIM_AD_PICTURA=/home/yyin5/workspace/Pictura-dev/scripts/hugsim/hugsim_e2e.sh
export HUGSIM_AD_DYNAMO=/home/yyin5/workspace/NAVSIM/dynamo_e2e.sh
