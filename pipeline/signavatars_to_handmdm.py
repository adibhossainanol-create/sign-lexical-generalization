
import argparse
import glob
import io
import os
import pickle

import numpy as np
import torch

from src.tools.geometry import to_matrix, matrix_to

LOWER = [0, 1, 3, 4, 6, 7, 9, 10]          # body joints HandMDM drops

SA = dict(glob=(0, 3), body=(3, 66), lhand=(66, 111), rhand=(111, 156),
          extra=(156, 179), transl=(179, 182))
HM = dict(body=(0, 78), lhand=(78, 168), rhand=(168, 258),
          jaw=(258, 264), expr=(264, 274))


def _cpu_load(path):
    class CPU(pickle.Unpickler):
        def find_class(self, m, n):
            if m == "torch.storage" and n == "_load_from_bytes":
                return lambda b: torch.load(io.BytesIO(b), map_location="cpu",
                                            weights_only=False)
            return super().find_class(m, n)
    return CPU(open(path, "rb")).load()


def aa_to_6d(aa):
    """(J,3) axis-angle -> (J,6) rot6d, with HandMDM's sign convention."""
    t = torch.tensor(np.asarray(aa, dtype=np.float32) * -1.0)
    return matrix_to("rot6d", to_matrix("axisangle", t)).numpy()


def sixd_to_aa(six):
    """Inverse of aa_to_6d."""
    t = torch.tensor(np.asarray(six, dtype=np.float32))
    return -matrix_to("axisangle", to_matrix("rot6d", t)).numpy()


def sa_frame_to_274(v):
    """One SignAvatars 182-vector -> one HandMDM 274-vector."""
    body21 = np.asarray(v[SA["body"][0]:SA["body"][1]]).reshape(21, 3)
    body13 = np.delete(body21, LOWER, axis=0)              # 21 - 8 = 13
    lh = np.asarray(v[SA["lhand"][0]:SA["lhand"][1]]).reshape(15, 3)
    rh = np.asarray(v[SA["rhand"][0]:SA["rhand"][1]]).reshape(15, 3)
    jaw = np.asarray(v[SA["extra"][0]:SA["extra"][0] + 3]).reshape(1, 3)
    expr = np.asarray(v[SA["extra"][0] + 3:SA["extra"][0] + 13]).reshape(10)

    return np.concatenate([
        aa_to_6d(body13).ravel(),      # 78
        aa_to_6d(lh).ravel(),          # 90
        aa_to_6d(rh).ravel(),          # 90
        aa_to_6d(jaw).ravel(),         # 6
        expr,                          # 10
    ]).astype(np.float32)


def feats274_from_smplx_dicts(frames):
    """Inverse of HandMDM's np_feats_to_smplx -- used by the self-test."""
    out = []
    for f in frames:
        body = np.asarray(f["body_pose"])
        if body.shape[0] == 21:
            body = np.delete(body, LOWER, axis=0)
        out.append(np.concatenate([
            aa_to_6d(body).ravel(),
            aa_to_6d(np.asarray(f["left_hand_pose"]).reshape(-1, 3)).ravel(),
            aa_to_6d(np.asarray(f["right_hand_pose"]).reshape(-1, 3)).ravel(),
            aa_to_6d(np.asarray(f["jaw_pose"]).reshape(-1, 3)).ravel(),
            np.asarray(f["expression"]).ravel(),
        ]).astype(np.float32))
    return np.stack(out)


def selftest(npy_path):
    from prepare.motion_feats_to_smpl import np_feats_to_smplx
    orig = np.load(npy_path).astype(np.float32)
    print(f"loaded {npy_path}  shape {orig.shape}")
    assert orig.shape[-1] == 274, f"expected 274-dim, got {orig.shape[-1]}"

    smplx = np_feats_to_smplx(orig)
    back = feats274_from_smplx_dicts(smplx)

    err = np.abs(orig - back)
    print(f"round-trip max abs err : {err.max():.3e}")
    print(f"round-trip mean abs err: {err.mean():.3e}")
    for name, (a, b) in HM.items():
        print(f"  {name:6s} [{a:3d}:{b:3d}]  max err {err[:, a:b].max():.3e}")

    # rot6d is not unique: HandMDM's raw output is not orthonormal
    # (|a1| ~ 0.9988, dot(a1,a2) ~ 0.003), so raw-value comparison is wrong.
    # Compare rotations instead.
    def geo(a, b):
        A = to_matrix("rot6d", torch.tensor(orig[:, a:b].reshape(-1, 6)))
        B = to_matrix("rot6d", torch.tensor(back[:, a:b].reshape(-1, 6)))
        rel = A.transpose(-1, -2) @ B
        cos = ((rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]) - 1) / 2
        return torch.arccos(cos.clamp(-1, 1)).max().item()
    worst = max(geo(a, b) for a, b in [(0, 78), (78, 168), (168, 258)])
    print(f"\nworst geodesic error: {worst:.2e} rad")
    if worst < 1e-3:
        print("\nPASS -- the inverse is correct, SignAvatars conversion is trustworthy")
    else:
        print("\nFAIL -- do NOT build on this. Most likely causes:")
        print("  * sign convention (their decoder negates; check aa_to_6d)")
        print("  * rot6d is not unique -- compare rotation MATRICES, not raw 6D")
        print("  * body joint insertion order differs from LOWER")


def convert_dir(pkl_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(pkl_dir, "*.npy")))
    print(f"{len(files)} pkls -> {out_dir}")
    for i, f in enumerate(files):
        s = np.load(f).astype(np.float32)
        feats = np.stack([sa_frame_to_274(v) for v in s])
        np.save(os.path.join(out_dir, os.path.basename(f)), feats)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)}")
    print("done")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--npy", help="an existing HandMDM 274-dim .npy for the self-test")
    p.add_argument("--pkl_dir")
    p.add_argument("--out", default="feats274")
    a = p.parse_args()

    if a.selftest:
        if not a.npy:
            raise SystemExit("--selftest needs --npy <a HandMDM .npy>")
        selftest(a.npy)
    elif a.pkl_dir:
        convert_dir(a.pkl_dir, a.out)
    else:
        raise SystemExit("give --selftest --npy ... or --pkl_dir ...")


if __name__ == "__main__":
    main()
