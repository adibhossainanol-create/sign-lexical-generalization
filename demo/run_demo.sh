#!/bin/bash
# run_demo.sh — one photograph + one text prompt -> one signing video
#
# Run from an activated `handmdm` shell (generation); the render step switches
# to `lhm` itself via conda run. See pipeline/README.md for the environments.
#
#   conda activate handmdm
#   export HANDMDM_DIR=/path/to/HandMDM LHM_DIR=/path/to/LHM
#   bash demo/run_demo.sh demo/assets/example_signer.png \
#     "Flat open hand fingers together fingertips touching the chin then moved forward and downward away from the face in a single smooth arc" \
#     thankyou.mp4
#
# Optional overrides: CKPT (relative to HANDMDM_DIR), MOTION_LENGTH (frames).

set -euo pipefail
: "${HANDMDM_DIR:?set HANDMDM_DIR to your HandMDM checkout}"
: "${LHM_DIR:?set LHM_DIR to your LHM checkout}"
[ $# -eq 3 ] || { sed -n 2,13p "$0"; exit 1; }
[ "${CONDA_DEFAULT_ENV:-}" = handmdm ] || { echo "activate the handmdm env first (conda deactivate; conda activate handmdm)"; exit 1; }

HERE=$(cd "$(dirname "$0")" && pwd)
PHOTO=$(realpath "$1")
PROMPT=$2
OUT=$(realpath -m "$3")
CKPT=${CKPT:-models/mdm_bobsl3dt_phonology_hms/checkpoints/last.ckpt}
LEN=${MOTION_LENGTH:-150}

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "[generate] $PROMPT"
(cd "$HANDMDM_DIR" && PYOPENGL_PLATFORM=egl python inference.py checkpoint="$CKPT" \
  "input_text=\"$PROMPT\"" render=false output_path="$WORK/motion.npy" \
  +motion_length="$LEN" < /dev/null)
[ -f "$WORK/motion.npy" ] || { echo "generation produced no motion"; exit 1; }

bash "$HERE/../pipeline/render.sh" "$WORK/motion.npy" "$PHOTO" "$OUT"
