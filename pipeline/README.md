# pipeline

Photograph + text prompt → signing video, and the prep that turns a video into
something the spotter can score.

## Three conda environments — never stack them

| Environment | Used for | Scripts |
|---|---|---|
| `handmdm` | motion generation (HandMDM `inference.py`) | `demo/run_demo.sh` (step 1) |
| `lhm` | SMPL-X conversion, hand refiner, rendering | `handmdm_to_lhm.py`, `amplify_hands.py`, `apply_adapter.py`, `render.sh` |
| `bsldict_env` | spotter scoring, **CPU only** | `controls/multi_subject_spotter.py`, `controls/spotter_negative_control.py` |

Run `conda deactivate` before switching. Stacking `conda activate` calls breaks
Python resolution, and scripts that call BSLDict's `demo.py` must be started
from an *activated* `bsldict_env` shell, not through `conda run`. See
[`../KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) for why.

## What you need outside this repository

| Variable | Points to | Needed by |
|---|---|---|
| `HANDMDM_DIR` | a HandMDM checkout and its checkpoint | `handmdm_to_lhm.py` (imports `src.tools.geometry`), `run_demo.sh`, `scaling/run_scaling.sh` |
| `LHM_DIR` | an LHM checkout with `LHM-500M-HF` | `render.sh` |
| `SMPLX_MODEL_PATH` | SMPL-X human model files (default: `$LHM_DIR/pretrained_models/human_model_files`) | `retarget_loss.py`, `adapter_v3.py` |
| `--bsldict` | a BSLDict checkout with `models/` and `bsldict/bsldict_v1.pkl` | the two spotter controls |

## Flow

```
motion.npy (274-d, 6D)
  │  handmdm_to_lhm.py      → SMPL-X frames (JSON), signer's own betas if given
  │  amplify_hands.py       → per-hand refiner, Eq. (5)
  │     (or apply_adapter.py → learned adapter, for the ablation)
  │  LHM inference.sh       → full-resolution mp4
  │                           ── render.sh runs the three steps above
  │  prep_for_spotter.sh    → crop / scale / setpts
  ▼
spotter score
```

`signavatars_to_handmdm.py` converts real SignAvatars clips (182-d axis-angle)
into the same 274-d representation, so real and generated motion go through
identical evaluation code.

## Files

| File | Role |
|---|---|
| `signavatars_to_handmdm.py` | 182-d axis-angle ↔ 274-d 6D (round-trip checked) |
| `handmdm_to_lhm.py` | motion → LHM SMPL-X frames |
| `amplify_hands.py` | per-hand target-matched gain (left 0.0199, right 0.0281), Eq. (5) |
| `apply_adapter.py` | learned alternative to the refiner; pass the adapter weights with `--ckpt` (not included) |
| `adapter_v3.py`, `retarget_loss.py` | the adapter model and the SMPL-X loss it was trained with |
| `render.sh` | motion + photograph → mp4 |
| `prep_for_spotter.sh` | the scoring prep: `crop`, `scale=256:256`, `setpts=4*(PTS-STARTPTS)`, 25 fps |
| `crop_boxes.json` | one crop box per signer, and how the boxes were measured |

## Example

```bash
export HANDMDM_DIR=/path/to/HandMDM LHM_DIR=/path/to/LHM

conda activate lhm
bash pipeline/render.sh gen/thankyou__s0.npy demo/assets/example_signer.png out/s1_thankyou.mp4
bash pipeline/prep_for_spotter.sh out/s1_thankyou.mp4 out/s1_thankyou_prep.mp4 s1
```
