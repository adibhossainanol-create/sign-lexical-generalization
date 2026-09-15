#!/bin/bash
# render.sh — HandMDM motion (.npy) + one photograph -> signing video (.mp4)
#
# Three steps, all in the `lhm` conda environment:
#   1. handmdm_to_lhm.py   274-d motion -> LHM SMPL-X frames
#                          (uses the signer's own betas when given, so body
#                          shape is personalised rather than inherited from
#                          the reference person)
#   2. amplify_hands.py    per-hand articulation refiner, Eq. (5)
#   3. LHM inference.sh    photograph + SMPL-X frames -> mp4
#
# Usage:
#   export LHM_DIR=/path/to/LHM HANDMDM_DIR=/path/to/HandMDM
#   bash pipeline/render.sh motion.npy photo.png out.mp4 [betas.json]
#
# Optional overrides: REF (reference SMPL-X frame), MODEL (LHM checkpoint name).
# The output is LHM's full-resolution render; crop for scoring with
# prep_for_spotter.sh, never before.

set -euo pipefail
: "${LHM_DIR:?set LHM_DIR to your LHM checkout}"
: "${HANDMDM_DIR:?set HANDMDM_DIR to your HandMDM checkout}"
[ $# -ge 3 ] || { sed -n 2,19p "$0"; exit 1; }

HERE=$(cd "$(dirname "$0")" && pwd)
MOTION=$(realpath "$1")
PHOTO=$(realpath "$2")
OUT=$(realpath -m "$3")
BETAS=${4:+$(realpath "$4")}
REF=${REF:-$LHM_DIR/train_data/motion_video/mimo1/smplx_params/00022.json}
MODEL=${MODEL:-LHM-500M-HF}
export HANDMDM_DIR

WORK=$(mktemp -d)
TAG=render_$$
IMG_DIR=$LHM_DIR/train_data/$TAG
trap 'rm -rf "$WORK" "$IMG_DIR"' EXIT

echo "[1/3] motion -> SMPL-X"
conda run --no-capture-output -n lhm python "$HERE/handmdm_to_lhm.py" \
  --handmdm_output "$MOTION" --ref_smplx "$REF" --output_dir "$WORK/smplx" \
  ${BETAS:+--betas "$BETAS"}

echo "[2/3] per-hand refiner"
conda run --no-capture-output -n lhm python "$HERE/amplify_hands.py" \
  --in "$WORK/smplx" --out "$WORK/amp"

echo "[3/3] LHM render"
mkdir -p "$IMG_DIR"
cp "$PHOTO" "$IMG_DIR/"
touch "$WORK/stamp"
(cd "$LHM_DIR" && conda run --no-capture-output -n lhm \
  bash inference.sh "$MODEL" "./train_data/$TAG/" "$WORK/amp" < /dev/null)

V=$(find "$LHM_DIR/exps/videos" -path "*train_data/$TAG/*" -name '*.mp4' -newer "$WORK/stamp" | head -1)
[ -n "$V" ] || { echo "render produced no mp4 under $LHM_DIR/exps/videos"; exit 1; }
mkdir -p "$(dirname "$OUT")"
cp "$V" "$OUT"
echo "wrote $OUT"
