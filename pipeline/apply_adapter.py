
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import torch


def load_adapter(ckpt_path, sa_dir):
    """Import Adapter from adapter_v3.py and load weights."""
    sys.path.insert(0, sa_dir)
    try:
        from adapter_v3 import Adapter
    except ImportError as e:
        sys.exit(f"could not import Adapter from {sa_dir}/adapter_v3.py: {e}")
    model = Adapter()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sa-dir",
                    default=os.path.dirname(os.path.abspath(__file__)),
                    help="directory holding adapter_v3.py")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.src, "*.json")))
    if not files:
        sys.exit(f"no JSON in {a.src}")

    frames = []
    for f in files:
        with open(f) as fh:
            frames.append(json.load(fh))

    # LHM stores these as nested lists; flatten to (F, 45) each.
    def grab(key):
        return np.array([np.asarray(fr[key], dtype=np.float32).ravel()
                         for fr in frames], dtype=np.float32)

    lh, rh = grab("lhand_pose"), grab("rhand_pose")
    if lh.shape[1] != 45 or rh.shape[1] != 45:
        sys.exit(f"expected 45 dims per hand, got {lh.shape[1]} / {rh.shape[1]}")

    pose = np.concatenate([lh, rh], axis=1)          # (F, 90)
    print(f"loaded {len(frames)} frames from {a.src}, pose {pose.shape}")

    model = load_adapter(a.ckpt, a.sa_dir)
    with torch.no_grad():
        x = torch.from_numpy(pose).unsqueeze(0).transpose(1, 2)   # (1, 90, F)
        y = model(x).transpose(1, 2).squeeze(0).numpy()           # (F, 90)

    delta = np.abs(y - pose)
    print(f"adapter delta: mean {delta.mean():.5f} rad, max {delta.max():.5f} rad")
    v_in = pose.var(axis=0).mean()
    v_out = y.var(axis=0).mean()
    print(f"temporal variance: {v_in:.6f} -> {v_out:.6f}  (ratio {v_out/max(v_in,1e-12):.3f})")

    os.makedirs(a.dst, exist_ok=True)
    for i, (f, fr) in enumerate(zip(files, frames)):
        fr["lhand_pose"] = y[i, :45].reshape(np.asarray(fr["lhand_pose"]).shape).tolist()
        fr["rhand_pose"] = y[i, 45:].reshape(np.asarray(fr["rhand_pose"]).shape).tolist()
        with open(os.path.join(a.dst, os.path.basename(f)), "w") as fh:
            json.dump(fr, fh)

    # carry over any non-JSON siblings (LHM sometimes expects them)
    for extra in glob.glob(os.path.join(a.src, "*")):
        if not extra.endswith(".json"):
            shutil.copy2(extra, a.dst)

    print(f"wrote {len(files)} frames to {a.dst}")


if __name__ == "__main__":
    main()
