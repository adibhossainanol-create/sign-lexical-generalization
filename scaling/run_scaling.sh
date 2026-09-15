#!/usr/bin/env bash
# Vocabulary-scaling curve: train each arm from scratch, sequentially, then
# chain the evaluation. NOT auto-launched -- run explicitly.
#
# WHY train_ PREFIXED SPLITS: src/data/text_motion.py sets
#   is_training = split.startswith("train")
# and only applies condition dropout (drop_cond=0.05, text -> "") when it is
# True. The first scaling run used splits named vocab25 etc., so every arm
# trained with NO unconditional branch; at guidance=15 inference then
# extrapolated from garbage (samples ~3x real scale). train_vocab25.txt etc.
# are byte-identical copies of the same ids, so the arms are unchanged except
# that dropout now happens. A guard below refuses any non-train split.
#
# Reconstructed from ft_scratch/config.json:
#   experiment=wlasl274 dataset=wlasl274, ckpt=null (from scratch),
#   trainer.logger=false (no wandb), val_split=null, trainer.max_epochs=500.
#
# Run names use PREFIX (default scale2) so nothing from the invalid first run
# is overwritten or -- worse -- silently reused by the skip-if-exists logic.
#
# Optional extra arm (not in the default set):
#   vocab99_budget400_steps3500 -> same 404 clips as budget400, trained for
#   3500 steps (875 epochs at 4 batches/epoch) to match vocab99's step count.
set -uo pipefail
cd "${HANDMDM_DIR:?set HANDMDM_DIR to your HandMDM checkout}"
PY="${PY:-python3}"   # python from the handmdm env
PREFIX="${PREFIX:-scale2}"
ARMS="${*:-vocab25 vocab50 vocab99 vocab99_budget200 vocab99_budget400}"
SPLITS=datasets/annotations/wlasl274/splits
done_arms=""

for arm in $ARMS; do
  run="${PREFIX}_${arm}"; split="train_${arm}"; epochs=500; seed=1234
  case "$arm" in
    vocab99) seed=7 ;;   # ft_scratch covers seed 1234 for this condition
    vocab99_budget400_steps3500) split="train_vocab99_budget400"; epochs=875 ;;
  esac

  case "$split" in train*) ;; *) echo "!! REFUSING ${split}: no 'train' prefix -> no condition dropout"; continue ;; esac
  [ -f "${SPLITS}/${split}.txt" ] || { echo "!! missing ${SPLITS}/${split}.txt"; continue; }

  if [ -s "${run}/checkpoints/last.ckpt" ]; then
    echo "[$(date +%H:%M:%S)] SKIP ${run} (last.ckpt present)"
  else
    echo "[$(date +%H:%M:%S)] START ${run}  split=${split}  seed=${seed}  epochs=${epochs}"
    $PY train.py \
        experiment=wlasl274 dataset=wlasl274 \
        run_name="${run}" train_split="${split}" val_split=null ckpt=null \
        seed="${seed}" trainer.logger=false trainer.max_epochs="${epochs}" \
        > "/tmp/${run}.log" 2>&1
    rc=$?
    echo "[$(date +%H:%M:%S)] DONE  ${run} rc=${rc} checkpoints=$(ls "${run}/checkpoints"/*.ckpt 2>/dev/null | wc -l)"
    [ $rc -ne 0 ] && { echo "  !! FAILED, see /tmp/${run}.log"; continue; }
  fi
  # inference.py locates config.json by walking up from the checkpoint
  [ -f "outputs/${run}/config.json" ] && cp -f "outputs/${run}/config.json" "${run}/config.json"
  done_arms="${done_arms} ${arm}"
done
echo "[$(date +%H:%M:%S)] TRAINING FINISHED:${done_arms:- none}"

if [ -n "${done_arms// /}" ]; then
  echo "[$(date +%H:%M:%S)] chaining evaluation"
  env PREFIX="$PREFIX" ARMS="${done_arms# }" bash eval_scaling.sh
fi
