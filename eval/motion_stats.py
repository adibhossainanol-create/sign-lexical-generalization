

import argparse
import glob
import json
import os

import numpy as np

# 90 = 15 joints x 3 (axis-angle) per hand, left then right (see load_json_clips).
HAND_SPLIT = 45


def _stats_one(c):
    """c: (F,D) single hand-or-pooled block for one clip."""
    d1 = np.diff(c, axis=0)                   # velocity
    d2 = np.diff(c, axis=0, n=2)               # acceleration -> jerk proxy
    v = np.linalg.norm(d1, axis=1)             # per-frame speed
    return {
        "velocity":   float(v.mean()),
        "jerk":       float(np.linalg.norm(d2, axis=1).mean()),
        "joint_var":  float(c.var(axis=0).mean()),
        "range":      float((c.max(0) - c.min(0)).mean()),
        "still_frac": float((v < 0.02).mean()),
    }


def stats_for_clips(clips):
    blocks = {
        "pooled": slice(0, None),
        "left":   slice(0, HAND_SPLIT),
        "right":  slice(HAND_SPLIT, None),
    }
    keys = ("velocity", "jerk", "joint_var", "range", "still_frac")
    acc = {name: {k: [] for k in keys} for name in blocks}
    n_clips = 0
    for c in clips:
        c = np.asarray(c, dtype=np.float32)      # (F,90)
        if c.shape[0] < 3:
            continue
        n_clips += 1
        for name, sl in blocks.items():
            for k, v in _stats_one(c[:, sl]).items():
                acc[name][k].append(v)

    out = {"n_clips": n_clips}
    for name in blocks:
        out[name] = {k: float(np.mean(v)) if v else float("nan")
                      for k, v in acc[name].items()}
    return out


def load_json_clips(d):
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    L = [np.concatenate([
            np.array(json.load(open(f))["lhand_pose"]).reshape(-1),
            np.array(json.load(open(f))["rhand_pose"]).reshape(-1)])
         for f in files]
    return [np.array(L)]           # one clip


def _print_block(name, r):
    print(f"  [{name}]")
    print(f"    velocity   (finger speed)     : {r['velocity']:.4f}")
    print(f"    jerk       (jitter; hi=noisy) : {r['jerk']:.4f}")
    print(f"    joint_var  (articulation used): {r['joint_var']:.4f}")
    print(f"    range      (motion extent)    : {r['range']:.4f}")
    print(f"    still_frac (static frames)    : {r['still_frac']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz")
    ap.add_argument("--json_dir")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    if args.npz:
        source = args.npz
        d = np.load(args.npz, allow_pickle=True)
        clips = list(d["clips"])
    elif args.json_dir:
        source = args.json_dir
        clips = load_json_clips(args.json_dir)
    else:
        raise SystemExit("need --npz or --json_dir")

    s = stats_for_clips(clips)
    print(f"\n--- motion stats: {args.label or 'set'} ---")
    print(f"  source  : {source}")
    print(f"  n_clips : {s['n_clips']}")
    for name in ("pooled", "left", "right"):
        _print_block(name, s[name])
    if s["left"]["joint_var"] > 0:
        rl = s["right"]["joint_var"] / s["left"]["joint_var"]
        print(f"\n  R/L joint_var ratio: {rl:.3f}")
    print("\ninterpret vs the other set:")
    print("  HandMDM jerk >> SignAvatars  -> model jitter as degradation")
    print("  HandMDM velocity/var/range << -> model oversmoothing/collapse")
    print("  HandMDM still_frac >>         -> model mode-collapse to static")


if __name__ == "__main__":
    main()
