
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

SEED_RE = re.compile(r"_{1,2}s\d+$")
ORDER = ["base", "pres20", "pres05", "kl01", "pres01", "scratch", "ft"]
LABEL = {"base": "pretrained", "ft": "lambda 0", "scratch": "from scratch",
         "pres01": "lambda 0.1", "kl01": "lambda 0.1 KL",
         "pres05": "lambda 0.5", "pres20": "lambda 2.0"}


def load_raw(p):
    x = np.load(p)
    if x.ndim == 3:
        x = x[0]
    if x.shape[0] < x.shape[1] and x.shape[0] > 100:
        x = x.T
    return x.astype(np.float64)


def sixd_to_R(v):
    """(...,6) -> (...,3,3) via Gram-Schmidt (Zhou et al.)."""
    a, b = v[..., :3], v[..., 3:]
    e1 = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b - (e1 * b).sum(-1, keepdims=True) * e1
    e2 = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=-1)


def axisangle_to_R(v):
    """(...,3) axis-angle -> (...,3,3) via Rodrigues' formula."""
    theta = np.linalg.norm(v, axis=-1)
    small = theta < 1e-12
    theta_safe = np.where(small, 1.0, theta)
    k = v / theta_safe[..., None]
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    z = np.zeros_like(kx)
    K = np.stack([np.stack([z, -kz, ky], -1),
                  np.stack([kz, z, -kx], -1),
                  np.stack([-ky, kx, z], -1)], -2)
    s = np.sin(theta)[..., None, None]
    c = np.cos(theta)[..., None, None]
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    return np.where(small[..., None, None], np.eye(3), R)


def projected_mean(R):
    """Chordal mean on SO(3): SVD-project the arithmetic mean of the matrices."""
    M = R.mean(0)
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def frechet_var(R):
    """Mean squared geodesic angle to the projected mean, in rad^2."""
    Rb = projected_mean(R)
    tr = np.trace(Rb.T @ R, axis1=1, axis2=2)
    th = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    return float((th ** 2).mean())


def axisangle_var(R):
    """The paper's quantity: per-component variance of the axis-angle vector."""
    tr = np.trace(R, axis1=1, axis2=2)
    th = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    s = np.sin(th)
    ax = np.stack([R[:, 2, 1] - R[:, 1, 2],
                   R[:, 0, 2] - R[:, 2, 0],
                   R[:, 1, 0] - R[:, 0, 1]], -1)
    small = s < 1e-6
    ax = np.where(small[:, None], 0.0, ax / (2 * s[:, None] + 1e-12))
    return float(np.var(ax * th[:, None], axis=0).mean())


def _stats_from_R(R):
    """R: (F, J, 3, 3). Returns (geodesic Frechet var, axis-angle var), mean over joints."""
    J = R.shape[1]
    g = [frechet_var(R[:, j]) for j in range(J)]
    a = [axisangle_var(R[:, j]) for j in range(J)]
    return float(np.mean(g)), float(np.mean(a))


def hand_stats(x, lo, hi):
    """Returns (geodesic Frechet var, axis-angle var) averaged over the joints
    in [lo,hi) of the 6D block."""
    blk = x[:, lo:hi]
    J = blk.shape[1] // 6
    if J == 0:
        return np.nan, np.nan
    R = sixd_to_R(blk[:, :J * 6].reshape(len(blk), J, 6))
    return _stats_from_R(R)


def hand_stats_aa(x, lo, hi):
    """Same as hand_stats, but for an axis-angle block (3 params/joint)
    instead of a 6D block -- e.g. adapter_data.npz's (F,90) clips."""
    blk = x[:, lo:hi]
    J = blk.shape[1] // 3
    if J == 0:
        return np.nan, np.nan
    R = axisangle_to_R(blk[:, :J * 3].reshape(len(blk), J, 3))
    return _stats_from_R(R)


def npz_real_signing(npz_path):
    """REAL SIGNING geodesic-vs-axis-angle R/L on adapter_data.npz-style clips:
    (F,90) axis-angle, 15 left-hand joints (0:45) then 15 right-hand joints
    (45:90) -- see motion_stats.py / retrieval_eval.py's documented layout.
    Only the reference block applies here: the 274-dim, 6D-rotation gen_seen_*
    outputs aren't the same representation, so there is no per-config table."""
    d = np.load(npz_path, allow_pickle=True)
    clips = d["clips"]
    rl_g, rl_a, rr_g, rr_a = [], [], [], []
    for c in clips:
        x = np.asarray(c, dtype=np.float64)
        if x.ndim == 3:
            x = x[0]
        if len(x) < 2 or x.shape[1] < 90:
            continue
        g, aa = hand_stats_aa(x, 0, 45); rl_g.append(g); rl_a.append(aa)
        g, aa = hand_stats_aa(x, 45, 90); rr_g.append(g); rr_a.append(aa)

    RLg, RRg = np.median(rl_g), np.median(rr_g)
    RLa, RRa = np.median(rl_a), np.median(rr_a)
    print("=" * 74)
    print(f"REAL SIGNING -- {npz_path} ({len(rl_g)} clips), median per hand")
    print("=" * 74)
    print(f"  geodesic (Frechet, rad^2)   L {RLg:.5f}   R {RRg:.5f}   "
          f"R/L {RRg/RLg:.3f}")
    print(f"  axis-angle (paper's metric) L {RLa:.5f}   R {RRa:.5f}   "
          f"R/L {RRa/RLa:.3f}")
    print("\n  This is the same 866-clip population the paper's R/L = 1.41 was")
    print("  computed on (see motion_stats.py), now checked against the")
    print("  geodesic metric instead of just axis-angle joint_var.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir")
    ap.add_argument("--wlasl")
    ap.add_argument("--glosses", default=None,
                     help="restrict real clips to these glosses; omit for ALL real clips")
    ap.add_argument("--gen-glob", default="gen_seen_*")
    # 274 = root/body block then 30 hand joints x 6D. Defaults assume the hand
    # block runs 94:274 (15 left then 15 right). VERIFY against your converter
    # before quoting numbers -- the script prints what it inferred.
    ap.add_argument("--lh", type=int, nargs=2, default=[94, 184])
    ap.add_argument("--rh", type=int, nargs=2, default=[184, 274])
    ap.add_argument("--npz", help="adapter_data.npz-style file of (F,90) "
                     "axis-angle clips; computes REAL SIGNING R/L only "
                     "(no per-config table -- see npz_real_signing)")
    a = ap.parse_args()

    if a.npz:
        npz_real_signing(a.npz)
        return

    if not a.real_dir or not a.wlasl:
        ap.error("--real-dir and --wlasl are required unless --npz is given")

    print(f"left hand  dims {a.lh[0]}:{a.lh[1]}  -> {(a.lh[1]-a.lh[0])//6} joints")
    print(f"right hand dims {a.rh[0]}:{a.rh[1]}  -> {(a.rh[1]-a.rh[0])//6} joints")
    print("If those joint counts are not 15/15, fix --lh/--rh before reading on.\n")

    id2g = {}
    for e in json.load(open(a.wlasl)):
        for i in e.get("instances", []):
            id2g[str(i.get("video_id", "")).zfill(5)] = e["gloss"]
    keep = set(json.load(open(a.glosses))) if a.glosses else None
    if keep is None:
        print("no --glosses given: using ALL real clips, no gloss filter\n")

    # ---- real reference ------------------------------------------------------
    rl_g, rl_a, rr_g, rr_a = [], [], [], []
    for p in sorted(Path(a.real_dir).glob("*.npy")):
        if keep is not None and id2g.get(p.stem.zfill(5)) not in keep:
            continue
        x = load_raw(p)
        if x.shape[1] < a.rh[1]:
            continue
        g, aa = hand_stats(x, *a.lh); rl_g.append(g); rl_a.append(aa)
        g, aa = hand_stats(x, *a.rh); rr_g.append(g); rr_a.append(aa)
    if not rl_g:
        sys.exit("no usable real clips -- check --real-dir and the slices")

    RLg, RRg = np.median(rl_g), np.median(rr_g)
    RLa, RRa = np.median(rl_a), np.median(rr_a)
    print("=" * 74)
    print(f"REAL SIGNING ({len(rl_g)} clips), median per hand")
    print("=" * 74)
    print(f"  geodesic (Frechet, rad^2)   L {RLg:.5f}   R {RRg:.5f}   "
          f"R/L {RRg/RLg:.3f}")
    print(f"  axis-angle (paper's metric) L {RLa:.5f}   R {RRa:.5f}   "
          f"R/L {RRa/RLa:.3f}")
    print(f"\n  The paper reports R/L = 1.41 in axis-angle. If the geodesic R/L")
    print(f"  is also > 1, the dominance asymmetry is not an artifact.\n")

    # ---- per configuration ---------------------------------------------------
    rows = {}
    for d in sorted(Path(".").glob(a.gen_glob)):
        name = d.name.replace("gen_seen_", "")
        files = sorted(d.glob("*.npy"))
        if not files:
            continue
        lg, la, rg, ra = [], [], [], []
        for f in files:
            x = load_raw(f)
            if x.shape[1] < a.rh[1]:
                continue
            g, aa = hand_stats(x, *a.lh); lg.append(g); la.append(aa)
            g, aa = hand_stats(x, *a.rh); rg.append(g); ra.append(aa)
        if not lg:
            continue
        rows[name] = dict(
            geo_L=np.median(lg) / RLg, geo_R=np.median(rg) / RRg,
            aa_L=np.median(la) / RLa, aa_R=np.median(ra) / RRa, n=len(lg))

    print("=" * 74)
    print("ARTICULATION RATIO vs REAL  (1.0 = real signing)")
    print("=" * 74)
    print(f"{'config':16s} {'geo L':>7s} {'geo R':>7s} | {'aa L':>7s} {'aa R':>7s}"
          f" | {'geo R/L':>8s}")
    seq_g, seq_a = [], []
    for k in ORDER:
        if k not in rows:
            continue
        r = rows[k]
        print(f"{LABEL[k]:16s} {r['geo_L']:7.3f} {r['geo_R']:7.3f} | "
              f"{r['aa_L']:7.3f} {r['aa_R']:7.3f} | {r['geo_R']/r['geo_L']:8.3f}")
        seq_g.append(r['geo_R']); seq_a.append(r['aa_R'])

    if len(seq_g) >= 3:
        # Spearman without scipy
        def rank(v):
            o = np.argsort(np.argsort(v))
            return o.astype(float)
        rg_, ra_ = rank(seq_g), rank(seq_a)
        rho = float(np.corrcoef(rg_, ra_)[0, 1])
        print("\n" + "=" * 74)
        print("DOES THE CONCLUSION SURVIVE?")
        print("=" * 74)
        print(f"  Spearman(geodesic order, axis-angle order) = {rho:.3f}")
        print(f"  identical ordering: {'YES' if rho > 0.999 else 'NO'}")
        print("\n  rho ~ 1.0 -> the metric is representation-dependent but every")
        print("  ordering the paper reports is unchanged on the manifold. State")
        print("  the dependence as a limitation and cite this check.")
        print("  rho well below 1 -> the axis-angle numbers cannot carry the")
        print("  claims and the metric has to be redefined geodesically.")

    print("\nNOTE: this validates the METRIC. The refiner is a separate question:")
    print("its affine rescaling about an axis-angle mean is not the geodesic")
    print("operation R' = Rbar exp(g log(Rbar^T R)). Report that as a known")
    print("approximation rather than claiming the refiner is SO(3)-correct.")


if __name__ == "__main__":
    main()
