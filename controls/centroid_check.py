#!/usr/bin/env python3
"""
centroid_check.py

ft retrieves its own gloss 0.96 of the time against a real->real ceiling of
0.543, with NN distances matching real-real (ratio 1.021). Not copying — but
above-ceiling retrieval still needs an explanation, and there is a natural one:

  PROTOTYPE     the generated clip sits near the CENTROID of that gloss's real
                clips. A centroid is closer to every member than members are to
                each other, which produces above-ceiling top-1 AND a healthy
                NN-distance ratio at the same time. Genuine generation, but of
                "the average way to sign X", not of the range.

  DIVERSE       generated clips sit among the real clips the way real clips sit
                among each other — no closer to the centroid than a real clip
                of the same gloss is.

Decides whether the paper says "produces correct signs" or "produces a
prototypical realization per gloss". Different claims.

  python centroid_check.py \
      --real-dir feats274 --wlasl $SA/WLASL_v0.3.json \
      --glosses seen_glosses.json \
      --gen base=gen_seen_base --gen ft=gen_seen_ft --gen kl01=gen_seen_kl01
"""
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

SLICE, T = (0, 258), 64
SEED_RE = re.compile(r"__s(\d+)$")


def load(p):
    x = np.load(p)
    if x.ndim == 3:
        x = x[0]
    if x.shape[0] < x.shape[1] and x.shape[0] in (258, 264, 274):
        x = x.T
    x = x[:, SLICE[0]:SLICE[1]].astype(np.float64)
    if len(x) != T:
        s, d = np.linspace(0, 1, len(x)), np.linspace(0, 1, T)
        x = np.stack([np.interp(d, s, x[:, i]) for i in range(x.shape[1])], 1)
    return x.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--wlasl", required=True)
    ap.add_argument("--glosses", required=True)
    ap.add_argument("--gen", action="append", default=[], metavar="NAME=DIR")
    a = ap.parse_args()

    id2g = {}
    for e in json.load(open(a.wlasl)):
        for i in e.get("instances", []):
            id2g[str(i.get("video_id", "")).zfill(5)] = e["gloss"]
    keep = set(json.load(open(a.glosses)))

    by_gloss = defaultdict(list)
    for p in sorted(Path(a.real_dir).glob("*.npy")):
        g = id2g.get(p.stem.zfill(5))
        if g in keep:
            by_gloss[g].append(load(p))
    by_gloss = {g: np.stack(v) for g, v in by_gloss.items() if len(v) >= 3}
    if not by_gloss:
        sys.exit("need >=3 real clips per gloss")
    cent = {g: v.mean(0) for g, v in by_gloss.items()}
    print(f"{len(by_gloss)} glosses with >=3 real clips\n")

    # --- reference: how close is a REAL clip to its own gloss centroid? ------
    # Leave-one-out, else the clip pulls its own centroid toward itself.
    real_ratio = []
    for g, V in by_gloss.items():
        n = len(V)
        for i in range(n):
            loo = (V.sum(0) - V[i]) / (n - 1)
            d_cent = np.linalg.norm(V[i] - loo)
            d_mem = np.mean([np.linalg.norm(V[i] - V[j])
                             for j in range(n) if j != i])
            real_ratio.append(d_cent / d_mem)
    real_ratio = np.array(real_ratio)
    print("=" * 68)
    print("REFERENCE: real clips, (dist to own centroid) / (mean dist to members)")
    print("=" * 68)
    print(f"  median {np.median(real_ratio):.3f}   "
          f"p10 {np.percentile(real_ratio,10):.3f}   "
          f"p90 {np.percentile(real_ratio,90):.3f}")
    print("  A real clip is ~this close to its centroid relative to its peers.")
    print("  A generated clip scoring MUCH LOWER is sitting at the prototype.\n")

    for spec in a.gen:
        name, d = spec.split("=", 1)
        files = sorted(Path(d).glob("*.npy"))
        if not files:
            print(f"{name}: nothing in {d}"); continue

        ratios, cent_closer, seen = [], 0, 0
        per_gloss_spread = defaultdict(list)
        for f in files:
            g = SEED_RE.sub("", f.stem)
            if g not in by_gloss:
                continue
            q = load(f)
            V = by_gloss[g]
            d_cent = np.linalg.norm(q - cent[g])
            d_all = np.array([np.linalg.norm(q - v) for v in V])
            ratios.append(d_cent / d_all.mean())
            cent_closer += int(d_cent < d_all.min())
            per_gloss_spread[g].append(q)
            seen += 1

        if not seen:
            print(f"{name}: no gloss overlap with the gallery\n"); continue
        ratios = np.array(ratios)

        # diversity of the generations themselves, in the same units
        gen_spread, real_spread = [], []
        for g, qs in per_gloss_spread.items():
            if len(qs) < 2:
                continue
            Q = np.stack(qs)
            gen_spread.append(np.mean([np.linalg.norm(Q[i] - Q[j])
                                       for i in range(len(Q))
                                       for j in range(i + 1, len(Q))]))
            V = by_gloss[g]
            real_spread.append(np.mean([np.linalg.norm(V[i] - V[j])
                                        for i in range(len(V))
                                        for j in range(i + 1, len(V))]))

        print("=" * 68)
        print(f"{name}   ({seen} clips)")
        print("=" * 68)
        print(f"  centroid ratio   median {np.median(ratios):.3f}   "
              f"p10 {np.percentile(ratios,10):.3f}   "
              f"p90 {np.percentile(ratios,90):.3f}")
        print(f"  closer to centroid than to ANY real clip: "
              f"{cent_closer}/{seen} ({100*cent_closer/seen:.0f}%)")
        if gen_spread:
            gs, rs = float(np.mean(gen_spread)), float(np.mean(real_spread))
            print(f"  within-gloss spread  generated {gs:.3f}  vs real {rs:.3f}"
                  f"   (ratio {gs/rs:.3f})")

        print("  VERDICT: ", end="")
        if np.median(ratios) < 0.75 * np.median(real_ratio) and cent_closer / seen > 0.5:
            print("PROTOTYPE — generations sit at the gloss average.")
            print("    Write 'produces a prototypical realization per gloss',")
            print("    not 'produces correct signs'.")
        elif np.median(ratios) < 0.9 * np.median(real_ratio):
            print("LEANS PROTOTYPE — closer to the centroid than real clips are.")
        else:
            print("DIVERSE — generations sit among the real clips like real")
            print("    clips do. 'Produces correct signs' is supported.")
        print()

    print("Within-gloss spread ratio well below 1.0 is the same story from the")
    print("other side: less variation than real signing has.")


if __name__ == "__main__":
    main()
