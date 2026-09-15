"""
amplify_hands_v3.py
===================
Per-HAND target-matched gain, with v2's per-JOINT clamping kept only as a
safety valve.

Why: HandMDM's under-articulation is per-hand, not global. In `here` and
`thankyou` the active RIGHT hand already sits at ground-truth articulation
(joint_var 0.025 / 0.017 vs the 0.024 SignAvatars anchor) while the LEFT hand
is collapsed (0.0005 / 0.0013). A single --gain for both hands is wrong in
one direction or the other:
  - v1 (one capped gain per hand) throttled the right hand to 2.68 and let
    the left take 6.0
  - v2 (per-joint gain, uncapped by target) pushed the right hand to 16-25x
    ground truth

v3 instead asks each hand how far it is from the target and gains it by
exactly that much:

    gain = sqrt(target / raw_joint_var)      # joint_var is a variance
    gain = clip(gain, 1.0, max_gain)         # never shrink, never wild

A hand already at target gets gain 1.0 and is left untouched. Per-joint
clamping then runs only to stop individual axes exceeding the anatomical
ceiling -- it is a safety valve, not the amplification rule.

Usage:
    python amplify_hands_v3.py --in <smplx dir> --out <dir> \
        --target 0.024 --max-gain 8.0 --ceiling 2.5
"""
import argparse
import glob
import json
import os

import numpy as np


def target_gain(seq, target, max_gain):
    """Per-hand gain that brings joint_var to `target`. Never below 1.0."""
    raw = float(seq.var(0).mean())
    if raw <= 1e-9:
        return 1.0, raw          # nothing to amplify; see note below
    g = np.sqrt(target / raw)
    return float(np.clip(g, 1.0, max_gain)), raw


def clamp_per_joint(seq, mean, gain, ceiling):
    """Reduce gain only on the axes that would exceed `ceiling`."""
    dev = seq - mean
    base = np.abs(mean).reshape(-1)
    peak_dev = np.abs(dev).max(0)
    room = np.maximum(ceiling - base, 0.0)
    allowed = np.where(peak_dev > 1e-6, room / np.maximum(peak_dev, 1e-6), gain)
    gains = np.maximum(np.minimum(gain, allowed), 1.0)
    return gains, int((gains < gain - 1e-6).sum())


def process(seq, target, max_gain, ceiling, label):
    mean = seq.mean(0, keepdims=True)
    g, raw = target_gain(seq, target, max_gain)
    gains, n_clamped = clamp_per_joint(seq, mean, g, ceiling)
    out = mean + (seq - mean) * gains
    new = float(out.var(0).mean())
    note = ""
    if raw <= 1e-9:
        note = "  <- NO SIGNAL: nothing to amplify (collapsed or static sign)"
    elif g <= 1.0 + 1e-6:
        note = "  <- already at/above target, left untouched"
    elif g >= max_gain - 1e-6 and new < target * 0.5:
        note = "  <- CANNOT REACH TARGET: collapsed or genuinely low-articulation"
    print(f"  {label}: raw {raw:.5f} -> {new:.5f}   gain {g:.2f}"
          f"   clamped {n_clamped}/45   max|pose| {np.abs(out).max():.3f}{note}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=float, default=0.024,
                    help="SignAvatars joint_var anchor")
    ap.add_argument("--max-gain", type=float, default=8.0)
    ap.add_argument("--ceiling", type=float, default=2.5,
                    help="inside SignAvatars' observed [-2.48, 2.73]")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.inp, "*.json")))
    if not files:
        raise SystemExit(f"no JSON in {args.inp}")
    data = [json.load(open(f)) for f in files]

    L = np.array([np.array(d["lhand_pose"]).reshape(-1) for d in data])
    R = np.array([np.array(d["rhand_pose"]).reshape(-1) for d in data])

    print(f"target {args.target}, max gain {args.max_gain}, ceiling {args.ceiling}")
    La = process(L, args.target, args.max_gain, args.ceiling, "lhand")
    Ra = process(R, args.target, args.max_gain, args.ceiling, "rhand")

    os.makedirs(args.out, exist_ok=True)
    for i, (f, d) in enumerate(zip(files, data)):
        d["lhand_pose"] = La[i].reshape(15, 3).tolist()
        d["rhand_pose"] = Ra[i].reshape(15, 3).tolist()
        json.dump(d, open(os.path.join(args.out, os.path.basename(f)), "w"))
    print(f"  wrote {len(files)} frames to {args.out}")


if __name__ == "__main__":
    main()
