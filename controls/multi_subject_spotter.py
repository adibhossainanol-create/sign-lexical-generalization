#!/usr/bin/env python3
"""
multi_subject_spotter.py

Feeds paper_stats.py's contact_split() (item 8): scores every rendered HQ
video against the BSLDict dictionary entry for its own gloss, using the
same I3D+MLP embedding + cosine-similarity pipeline as bsldict/demo/demo.py
-- headless (no visualization), batched over every signer.

Video naming convention in --raw-dir: s<signer>_<idx>_<gloss>.mp4
(e.g. s1_01_good.mp4). For each gloss, the score is the MEAN across
signers of that signer's own max similarity (max over sliding windows x
dictionary versions) -- "multi-subject" because a single signer's video
being an outlier (bad render, occlusion) shouldn't set the gloss's score.
Per-signer scores are also printed so that can be checked.

Run in the `bsldict_env` conda environment (needs the old torch/cv2 that
environment pins), from anywhere -- paths are resolved relative to --bsldict.

  conda activate bsldict_env
  python multi_subject_spotter.py \
      --raw-dir outputs/hq/raw \
      --bsldict /path/to/bsldict \
      --out spotter_scores.json
"""
import argparse
import json
import math
import pickle as pkl
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import pairwise_distances

NAME_RE = re.compile(r"^s(\d+)_\d+_(.+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="dir of s<signer>_<idx>_<gloss>.mp4")
    ap.add_argument("--bsldict", required=True, help="bsldict repo root")
    ap.add_argument("--out", default="spotter_scores.json")
    ap.add_argument("--arch", default="i3d_mlp", choices=["i3d", "i3d_mlp"])
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--num_in_frames", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--embd_dim", type=int, default=256)
    ap.add_argument("--device", default=None,
                     help="cuda/cpu override; auto-detects an unsupported GPU "
                          "compute capability (old bsldict_env torch) and falls "
                          "back to cpu otherwise")
    a = ap.parse_args()

    bsl_root = Path(a.bsldict)
    sys.path.insert(0, str(bsl_root / "demo"))
    sys.path.insert(0, str(bsl_root))
    from utils import load_model, load_rgb_video, prepare_input, sliding_windows

    ckpt = bsl_root / "models" / f"{a.arch}.pth.tar"
    meta_path = bsl_root / "bsldict" / "bsldict_v1.pkl"
    assert ckpt.exists(), f"missing checkpoint {ckpt}"
    assert meta_path.exists(), f"missing dictionary metadata {meta_path}"

    print(f"loading dictionary metadata from {meta_path}")
    with open(meta_path, "rb") as f:
        meta = pkl.load(f)
    words = np.array(meta["videos"]["word"])
    dict_feats_all = np.array(meta["videos"]["features"]["mlp"])

    if a.device:
        device = torch.device(a.device)
    elif torch.cuda.is_available():
        # Metadata-only check (no kernel launch -- launching a real op on an
        # unsupported arch can just HANG rather than raise/warn).
        cap = torch.cuda.get_device_capability(0)
        cap_tag = f"sm_{cap[0]}{cap[1]}"
        arch_list = torch.cuda.get_arch_list()
        if cap_tag in arch_list:
            device = torch.device("cuda")
        else:
            print(f"[warn] GPU compute capability {cap_tag} not in this torch "
                  f"{torch.__version__} build's arch list {arch_list} -- "
                  f"falling back to CPU. Pass --device cuda to force it anyway.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"loading model from {ckpt}")
    model = load_model(checkpoint_path=ckpt, arch=a.arch).to(device)
    print(f"device: {device}")

    files = sorted(Path(a.raw_dir).glob("*.mp4"))
    if not files:
        raise SystemExit(f"no .mp4 files in {a.raw_dir}")

    by_gloss = defaultdict(dict)   # gloss -> {signer: score}
    for f in files:
        m = NAME_RE.match(f.stem)
        if not m:
            print(f"  [skip] {f.name}: doesn't match s<signer>_<idx>_<gloss>.mp4")
            continue
        signer, gloss = m.group(1), m.group(2)

        dict_ix = np.where(words == gloss)[0]
        if len(dict_ix) == 0:
            print(f"  [skip] {f.name}: gloss '{gloss}' not in dictionary")
            continue
        dict_feats = dict_feats_all[dict_ix]

        rgb_orig = load_rgb_video(video_path=f, fps=a.fps)
        rgb_input = prepare_input(rgb_orig)
        rgb_slides, _ = sliding_windows(
            rgb=rgb_input, stride=a.stride, num_in_frames=a.num_in_frames)
        num_clips = rgb_slides.shape[0]
        num_batches = math.ceil(num_clips / a.batch_size)
        feats = np.empty((0, a.embd_dim), dtype=float)
        with torch.no_grad():
            for b in range(num_batches):
                inp = rgb_slides[b * a.batch_size:(b + 1) * a.batch_size].to(device)
                out = model(inp)
                feats = np.append(
                    feats, out["embds"].cpu().detach().flatten(1).numpy(), axis=0)

        dst = pairwise_distances(feats, dict_feats, metric="cosine")
        sim = 1 - dst / 2
        score = float(sim.max())
        by_gloss[gloss][signer] = score
        print(f"  {f.name:24s} gloss={gloss:10s} signer={signer}  sim={score:.4f}")

    scores = {}
    print("\n" + "=" * 60)
    print("per-gloss (mean across signers)")
    print("=" * 60)
    for gloss, per_signer in sorted(by_gloss.items()):
        vals = list(per_signer.values())
        scores[gloss] = float(np.mean(vals))
        detail = ", ".join(f"{s}={v:.3f}" for s, v in sorted(per_signer.items()))
        print(f"  {gloss:10s} mean={scores[gloss]:.4f}  n={len(vals)}  ({detail})")

    with open(a.out, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nwrote {len(scores)} gloss scores to {a.out}")


if __name__ == "__main__":
    main()
