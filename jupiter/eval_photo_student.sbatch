#!/bin/bash
#SBATCH --account=e-reg-2026r01-243
#SBATCH --job-name=hugsim_photo_student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --partition=booster
# A Pictura photo student (pufferlib.sim2real best_adapter.pt) on HUGSIM official436, one booster node: the four
# datasets through run_benchmark.py, PER_GPU episodes at once on each of the four GPUs. The student drives
# reactive (Pictura's scripts/hugsim/hugsim_e2e.py --adapter) from HUGSIM's renders of PHOTOS, and reads its
# teacher's observation row from the raster mode's export. Resubmit the same command to go on: scored episodes
# are skipped. Episodes go to $HUGSIM_OUT/<TAG>/official436/<dataset>_pictura/, the table to $HUGSIM_OUT/<TAG>/agg.json.
#   TAG=tfa_base_4hz ADAPTER=.../best_adapter.pt sbatch --output=<logdir>/%x-%j.out jupiter/eval_photo_student.sbatch
# Settings: TAG (required), ADAPTER (the student; unset, the teacher drives); PICTURA_TREE (the Pictura tree with hugsim_e2e.py --adapter), PICTURA_CHECKPOINT
# (the teacher the adapter was distilled from), KINEMATIC (configs/sim/kinematic.yaml, dt 0.25 s, default; or
# kinematic_10hz.yaml, dt 0.1 s), CAMERA_SUFFIX (configs/sim/<dataset>_camera<suffix>.yaml; empty default),
# PHOTOS (the HUGSIM cameras rendered for the student), PHOTO_CAMERAS (hugsim_e2e.py --photo-cameras), E2E_ARGS
# (more hugsim_e2e.py flags), PER_GPU (default 4), LIMIT (first N scenarios per dataset, a pilot).
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[ -f closed_loop.py ] || { echo "submit from the HUGSIM_garage root"; exit 1; }
: "${TAG:?}"
# No ADAPTER: the teacher (PICTURA_CHECKPOINT) drives itself from the raster export, no photos rendered.
[ -z "${ADAPTER:-}" ] || [ -f "$ADAPTER" ] || { echo "no adapter $ADAPTER"; exit 1; }
PICTURA_TREE=${PICTURA_TREE:-/e/home/jusers/vu5/jupiter/code/Pictura-dev-hugsim}
export PICTURA_CHECKPOINT=${PICTURA_CHECKPOINT:-/e/scratch/e-reg-2026r01-243/vu5/ckpt/pv_50b_cos/models/model_puffer_drive_1792696908_003189.pt}
KINEMATIC=${KINEMATIC:-configs/sim/kinematic.yaml}
PHOTOS=${PHOTOS:-CAM_FRONT CAM_BACK CAM_FRONT_LEFT CAM_FRONT_RIGHT}
SET=${SET:-official436}
source jupiter/hugsim_jupiter.sh
export HUGSIM_AD_PICTURA=$PICTURA_TREE/scripts/hugsim/hugsim_e2e.sh
export PICTURA_PYTHON=$PICTURA_TREE/scripts/hugsim/jupiter/pictura_python.sh
export PICTURA_E2E_ARGS="--allow-dt-mismatch --control reactive ${ADAPTER:+--adapter $ADAPTER} ${PHOTO_CAMERAS:+--photo-cameras $PHOTO_CAMERAS} ${E2E_ARGS:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
OUT=$HUGSIM_OUT/$TAG
mkdir -p "$OUT"
PER_GPU=${PER_GPU:-4}
echo "$(date -Is) $(hostname) job ${SLURM_JOB_ID:-local}: $TAG on $SET -> $OUT"
echo "  garage $(git log --oneline -1) $(git status --short | wc -l) changed; pictura $PICTURA_TREE @ $(git -C "$PICTURA_TREE" log --oneline -1)"
echo "  adapter $ADAPTER, teacher $PICTURA_CHECKPOINT, $KINEMATIC, photos $PHOTOS, e2e $PICTURA_E2E_ARGS"
nvidia-smi -L
extra=(${ADAPTER:+--obs_photos $PHOTOS} --kinematic_path "$KINEMATIC")
if [ -n "${CAMERA_SUFFIX:-}" ]; then
  # run_benchmark.py names configs/sim/<ds>_camera.yaml; one dataset at a time, so each gets its own camera file
  for ds in ${DATASETS:-nuscenes kitti360 pandaset waymo}; do
    python run_benchmark.py --agents pictura --set "$SET" --datasets "$ds" --limit "${LIMIT:-0}" \
      --slots 0:$PER_GPU 1:$PER_GPU 2:$PER_GPU 3:$PER_GPU --out "$OUT" \
      -- "${extra[@]}" --camera_path "configs/sim/${ds}_camera${CAMERA_SUFFIX}.yaml"
  done
else
  python run_benchmark.py --agents pictura --set "$SET" --datasets ${DATASETS:-nuscenes kitti360 pandaset waymo} \
    --limit "${LIMIT:-0}" --slots 0:$PER_GPU 1:$PER_GPU 2:$PER_GPU 3:$PER_GPU --out "$OUT" -- "${extra[@]}"
fi
python aggregate_benchmark.py --out-root "$OUT" --set "$SET" --agents pictura --json "$OUT/agg.json"
echo "$(date -Is) RUN_DONE"
