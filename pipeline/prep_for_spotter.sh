#!/bin/bash
# prep_for_spotter.sh — prepare a rendered clip for BSLDict spotter scoring
#
# Applies exactly the prep in controls/spotter_negative_control.py:
#   crop=<box>, scale=256:256, setpts=4*(PTS-STARTPTS), 25 fps, no audio
# Scores are NOT portable across preparations (KNOWN_ISSUES.md): everything
# you compare must go through this script with the same box, and a new prep
# needs its own negative control before any absolute claim.
#
# Usage:
#   bash pipeline/prep_for_spotter.sh in.mp4 out.mp4 s1                # box from crop_boxes.json
#   bash pipeline/prep_for_spotter.sh in.mp4 out.mp4 276:345:217:147   # explicit w:h:x:y

set -euo pipefail
[ $# -eq 3 ] || { sed -n 2,12p "$0"; exit 1; }

HERE=$(cd "$(dirname "$0")" && pwd)
BOX=$3
if ! [[ $BOX =~ ^[0-9]+:[0-9]+:[0-9]+:[0-9]+$ ]]; then
  BOX=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["boxes"][sys.argv[2]])' \
        "$HERE/crop_boxes.json" "$BOX" 2>/dev/null) || { echo "no box for signer '$3' in crop_boxes.json"; exit 1; }
fi

ffmpeg -nostdin -y -loglevel error -i "$1" \
  -vf "crop=$BOX,scale=256:256,setpts=4*(PTS-STARTPTS)" -r 25 -an "$2"
echo "wrote $2 (crop $BOX)"
