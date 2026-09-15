#!/usr/bin/env python3
"""
retrieval_eval.py -- pose-space retrieval evaluation for generated sign motion.

QUESTION IT ANSWERS
  Are generated signs the RIGHT signs, not just the right AMOUNT of motion?
  For each generated clip we ask whether it is closest, in pose space, to REAL
  clips of the SAME gloss rather than to clips of other glosses. Retrieval is
  fairly robust to the over-articulation trap that fools art_ratio: a model
  that just moves the hands more will still sit far from the correct real sign.

THREE NUMBERS (mirrors the bsldict validation discipline)
  1. real  -> real   (leave-one-out) : the CEILING and a validity check on the
     metric itself. If this is near chance the feature space cannot tell signs
     apart and every other number below is meaningless -- STOP and fix it.
  2. base  -> real                    : what the un-fine-tuned model retrieves.
  3. ft    -> real                    : the result. Beating base and approaching
     the real->real ceiling is the correctness signal we are after.

Everything runs on HandMDM-format 274-dim .npy clips of shape (F, 274).
No rendering, no GPU. Two distances are reported side by side (a resampled-L2
metric and DTW) so a single metric can't mislead us -- if they disagree, that
is itself informative.

274 layout: 0:78 body(13x6d) | 78:168 lhand(15x6d) | 168:258 rhand(15x6d)
            | 258:264 jaw | 264:274 expr(zero in SignAvatars)

USAGE (on your machine, in the `handmdm` env)
  # 1) dump one generated clip per held-out gloss from each checkpoint into
  #    gen_base/<gloss>.npy and gen_ft/<gloss>.npy  (reuse eval_finetune.py's
  #    matched-length generation -- see the note at the bottom of this file).
  # 2) then:
  python retrieval_eval.py \
      --real_dir feats274 \
      --split adapter_data_gloss.npz \
      --wlasl $SA/WLASL_v0.3.json \
      --gen base=gen_base --gen ft=gen_ft

  # sanity-check the scorer itself first (no data needed):
  python retrieval_eval.py --selftest
"""

import argparse, glob, json, os, sys
import numpy as np

# ----------------------------------------------------------------------------
# 274-dim feature slices
# ----------------------------------------------------------------------------
SLICES = {
    "all":        (0, 264),    # everything except the always-zero expr block
    "body_hands": (0, 258),    # body + both hands  (default)
    "hands":      (78, 258),   # both hands only -- signs are hand-dominated
    # dims 0:18 are exactly constant in real clips; with the std floor they
    # dominate standardised distances and bias DTW rankings by clip length
    "body_hands_live": (18, 258),
}


def load_clip(path):
    x = np.load(path)
    if x.ndim != 2 or x.shape[1] < 258:
        raise ValueError(f"{path}: expected (F,274)-ish, got {x.shape}")
    return x.astype(np.float64)


def resample(x, L):
    """Linearly resample a (F,d) sequence to (L,d) along time."""
    F = len(x)
    if F == L:
        return x
    src = np.linspace(0.0, 1.0, F)
    dst = np.linspace(0.0, 1.0, L)
    return np.stack([np.interp(dst, src, x[:, k]) for k in range(x.shape[1])], axis=1)


# ----------------------------------------------------------------------------
# distances
# ----------------------------------------------------------------------------
def dtw_dist(a, b):
    """Length-normalised DTW with L2 local cost. a:(n,d) b:(m,d)."""
    n, m = len(a), len(b)
    a2 = (a * a).sum(1)[:, None]
    b2 = (b * b).sum(1)[None, :]
    C = np.sqrt(np.maximum(a2 + b2 - 2.0 * a @ b.T, 0.0))  # (n,m) frame costs
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        Ci = C[i - 1]
        prev = D[i - 1]
        row = D[i]
        for j in range(1, m + 1):
            row[j] = Ci[j - 1] + min(prev[j], row[j - 1], prev[j - 1])
    return D[n, m] / (n + m)  # normalise so long clips aren't penalised


def pdist_rows(Q, G):
    """Euclidean distance, each row of Q (nq,D) to each row of G (ng,D)."""
    q2 = (Q * Q).sum(1)[:, None]
    g2 = (G * G).sum(1)[None, :]
    return np.sqrt(np.maximum(q2 + g2 - 2.0 * Q @ G.T, 0.0))


# ----------------------------------------------------------------------------
# retrieval scoring
# ----------------------------------------------------------------------------
def retrieval_scores(Dqg, q_gloss, g_gloss, self_index=None, topk=(1, 5)):
    """
    Dqg      : (nq, ng) distance from each query to each gallery clip
    q_gloss  : (nq,) gloss per query
    g_gloss  : (ng,) gloss per gallery clip
    self_index: (nq,) gallery column to exclude per query (leave-one-out), or None
    Returns dict with clip-mean and gloss-balanced top-k accuracy + chance.
    """
    nq, ng = Dqg.shape
    g_gloss = np.asarray(g_gloss)
    hits = {k: [] for k in topk}
    chance = []
    per_gloss = {}  # gloss -> list of top1 hits
    for i in range(nq):
        d = Dqg[i].copy()
        if self_index is not None and self_index[i] >= 0:
            d[self_index[i]] = np.inf
        order = np.argsort(d, kind="stable")
        order = order[np.isfinite(d[order])]
        ranked = g_gloss[order]
        same = (g_gloss == q_gloss[i])
        if self_index is not None and self_index[i] >= 0:
            same = same.copy(); same[self_index[i]] = False
        n_same = int(same.sum()); n_gal = int(np.isfinite(d).sum())
        chance.append(n_same / n_gal if n_gal else 0.0)
        for k in topk:
            hits[k].append(1.0 if (ranked[:k] == q_gloss[i]).any() else 0.0)
        per_gloss.setdefault(q_gloss[i], []).append(hits[1][-1])

    out = {"n_queries": nq}
    for k in topk:
        out[f"top{k}"] = float(np.mean(hits[k]))
    out["chance_top1"] = float(np.mean(chance))
    out["gloss_top1"] = float(np.mean([np.mean(v) for v in per_gloss.values()]))
    out["per_gloss"] = {g: float(np.mean(v)) for g, v in per_gloss.items()}
    out["per_gloss_hits"] = per_gloss  # raw {gloss: [0/1, ...]} top-1 flags
    return out


def featurize(clips, slc, standardizer, mode, L):
    """Return either a stacked matrix (resample mode) or a list of seqs (dtw)."""
    lo, hi = slc
    mean, std = standardizer
    if mode == "resample":
        rows = [((resample(c[:, lo:hi], L) - mean) / std).ravel() for c in clips]
        return np.stack(rows)
    else:  # dtw: standardise, cap length so DTW stays tractable
        seqs = []
        for c in clips:
            s = (c[:, lo:hi] - mean) / std
            if len(s) > L:
                s = resample(s, L)
            seqs.append(np.ascontiguousarray(s))
        return seqs


def dtw_matrix(Q, G):
    M = np.empty((len(Q), len(G)))
    for i, q in enumerate(Q):
        for j, g in enumerate(G):
            M[i, j] = dtw_dist(q, g)
    return M


def build_standardizer(real_clips, slc):
    lo, hi = slc
    allframes = np.concatenate([c[:, lo:hi] for c in real_clips], axis=0)
    mean = allframes.mean(0)
    std = allframes.std(0)
    std[std < 1e-6] = 1e-6
    return mean, std


def run_config(real_clips, real_gloss, gens, slc_name, dist, L):
    slc = SLICES[slc_name]
    std = build_standardizer(real_clips, slc)
    print(f"\n=== feature={slc_name}  distance={dist}  (real gallery: "
          f"{len(real_clips)} clips, {len(set(real_gloss))} glosses) ===")

    rg = np.asarray(real_gloss)
    Rf = featurize(real_clips, slc, std, dist, L)

    # 1) real -> real, leave-one-out
    if dist == "resample":
        Drr = pdist_rows(Rf, Rf)
    else:
        Drr = dtw_matrix(Rf, Rf)
    self_idx = np.arange(len(real_clips))
    rr = retrieval_scores(Drr, rg, rg, self_index=self_idx)
    _print_row("real->real (CEILING)", rr)

    # 2,3) each generated set -> real
    rows = {"real->real (CEILING)": rr}
    for name, (gclips, ggloss) in gens.items():
        Gf = featurize(gclips, slc, std, dist, L)
        Dgr = pdist_rows(Gf, Rf) if dist == "resample" else dtw_matrix(Gf, Rf)
        sc = retrieval_scores(Dgr, np.asarray(ggloss), rg, self_index=None)
        _print_row(f"{name}->real", sc)
        rows[f"{name}->real"] = sc
    return rows


def _print_row(label, s):
    print(f"  {label:<24}  top1={s['top1']:.3f}  top5={s['top5']:.3f}  "
          f"gloss-top1={s['gloss_top1']:.3f}  chance={s['chance_top1']:.3f}  "
          f"(n={s['n_queries']})")


# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------
def load_wlasl_map(path):
    W = json.load(open(path))
    m = {}
    for e in W:
        for inst in e["instances"]:
            m[str(inst["video_id"])] = e["gloss"]
    return m


def stem_id(path):
    # strip a "__s<N>" multi-sample suffix so gen dirs can hold several
    # samples per gloss (agree__s0.npy, agree__s1.npy -> "agree")
    return os.path.splitext(os.path.basename(path))[0].split("__")[0]


def load_real(real_dir, id2gloss, test_glosses):
    clips, glosses, missing = [], [], 0
    for p in sorted(glob.glob(os.path.join(real_dir, "*.npy"))):
        gid = stem_id(p)
        g = id2gloss.get(gid) or id2gloss.get(gid.lstrip("0")) \
            or id2gloss.get(gid.zfill(5))
        if g is None:
            missing += 1; continue
        if g not in test_glosses:
            continue
        clips.append(load_clip(p)); glosses.append(g)
    if missing:
        print(f"  [warn] {missing} real clips had no WLASL gloss match "
              f"(check id format between feats274 stems and WLASL_v0.3.json)")
    return clips, glosses


def load_gen(gen_dir):
    clips, glosses = [], []
    for p in sorted(glob.glob(os.path.join(gen_dir, "*.npy"))):
        clips.append(load_clip(p)); glosses.append(stem_id(p))
    return clips, glosses


# ----------------------------------------------------------------------------
# self-test on synthetic data -- validates the scorer, not any real model
# ----------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(0)
    D, G, K = 258, 20, 6
    templates = {}
    for gi in range(G):
        base = np.cumsum(rng.normal(0, 0.15, size=(50, D)), axis=0)  # smooth walk
        templates[f"g{gi:02d}"] = base

    def instance(t, noise, warp):
        L = int(len(t) * warp)
        s = resample(t, L)
        return s + rng.normal(0, noise, size=s.shape)

    real_clips, real_gloss = [], []
    for g, t in templates.items():
        for _ in range(K):
            real_clips.append(instance(t, 0.20, rng.uniform(0.7, 1.3)))
            real_gloss.append(g)

    # "ft" = template + moderate noise (should retrieve near ceiling)
    # "base" = a walk unrelated to its gloss (should sit at chance)
    ft_clips, ft_gloss, base_clips, base_gloss = [], [], [], []
    for g, t in templates.items():
        ft_clips.append(instance(t, 0.35, 1.0)); ft_gloss.append(g)
        junk = np.cumsum(rng.normal(0, 0.15, size=(50, D)), axis=0)
        base_clips.append(junk); base_gloss.append(g)

    # pad to 274 so slices behave exactly like real data
    def pad(cl):
        return [np.concatenate([c, np.zeros((len(c), 274 - D))], 1) for c in cl]
    real_clips, ft_clips, base_clips = pad(real_clips), pad(ft_clips), pad(base_clips)

    gens = {"base": (base_clips, base_gloss), "ft": (ft_clips, ft_gloss)}
    ok = True
    for dist in ("resample", "dtw"):
        rows = run_config(real_clips, real_gloss, gens, "body_hands", dist, 48)
        rr, ba, ft = rows["real->real (CEILING)"], rows["base->real"], rows["ft->real"]
        cond = (rr["top1"] > 0.6 and rr["top1"] > 3 * rr["chance_top1"]
                and ft["top1"] > 0.5 and ft["top1"] > ba["top1"] + 0.2
                and ba["top1"] < rr["chance_top1"] + 0.20)
        print(f"  --> {dist}: {'PASS' if cond else 'FAIL'} "
              f"(ceiling {rr['top1']:.2f} >> chance {rr['chance_top1']:.2f}; "
              f"ft {ft['top1']:.2f} >> base {ba['top1']:.2f})")
        ok = ok and cond
    print(f"\nSELF-TEST {'PASSED' if ok else 'FAILED'} -- the scorer separates "
          f"planted same-gloss structure from noise and reads chance correctly.")
    return 0 if ok else 1


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir")
    ap.add_argument("--split", help="adapter_data_gloss.npz (uses test_glosses)")
    ap.add_argument("--glosses", help="JSON list of glosses to evaluate on "
                    "(alternative to --split, e.g. for a SEEN-gloss run with "
                    "the same slice/standardization/cap as the held-out eval)")
    ap.add_argument("--wlasl", help="WLASL_v0.3.json")
    ap.add_argument("--gen", action="append", default=[],
                    help="name=dir, repeatable (e.g. --gen base=gen_base --gen ft=gen_ft)")
    ap.add_argument("--feature", choices=list(SLICES), default="body_hands")
    ap.add_argument("--dist", choices=["resample", "dtw", "both"], default="both")
    ap.add_argument("--cap", type=int, default=48,
                    help="frames to resample to (resample mode) / cap at (dtw mode)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())

    for req in ("real_dir", "wlasl"):
        if not getattr(a, req):
            ap.error(f"--{req} is required (or use --selftest)")
    if not a.split and not a.glosses:
        ap.error("one of --split or --glosses is required (or use --selftest)")

    if a.glosses:
        test_glosses = set(json.load(open(a.glosses)))
    else:
        d = np.load(a.split, allow_pickle=True)
        test_glosses = set(str(x) for x in d["test_glosses"])
    id2gloss = load_wlasl_map(a.wlasl)
    print(f"glosses ({len(test_glosses)}): {sorted(test_glosses)}")

    real_clips, real_gloss = load_real(a.real_dir, id2gloss, test_glosses)
    if not real_clips:
        ap.error("no real held-out clips loaded -- check --real_dir and id matching")

    gens = {}
    for spec in a.gen:
        name, _, gdir = spec.partition("=")
        gc, gg = load_gen(gdir)
        gg = [g for g in gg if g in test_glosses]
        gc = [c for c, g0 in zip(gc, [stem_id(p) for p in
              sorted(glob.glob(os.path.join(gdir, "*.npy")))]) if g0 in test_glosses]
        if not gc:
            print(f"  [warn] {name}: no generated clips whose gloss is held-out in {gdir}")
        gens[name] = (gc, gg)

    dists = ["resample", "dtw"] if a.dist == "both" else [a.dist]
    for dist in dists:
        rows = run_config(real_clips, real_gloss, gens, a.feature, dist, a.cap)
        # DTW is the headline metric; dump per-clip hits for retrieval_stats.py
        if dist == "dtw":
            for name in gens:
                key = f"{name}->real"
                if key not in rows:
                    continue
                with open(f"hits_seen_{name}.json", "w") as f:
                    json.dump(rows[key]["per_gloss_hits"], f, indent=2)

    print("\nread: ft->real should beat base->real and approach real->real. "
          "If real->real is near chance the metric is invalid -- fix before trusting.")


if __name__ == "__main__":
    main()
