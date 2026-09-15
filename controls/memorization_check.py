#!/usr/bin/env python3
"""
memorization_check.py

The seen-gloss result (top-1 0.920) sits ABOVE the real->real ceiling (0.607).
A generated clip cannot legitimately be more retrievable than real clips are of
each other unless it is reproducing a specific real clip. This script decides
between two readings:

  MEMORIZATION  generated clips are near-duplicates of particular training
                clips. Nearest-neighbour distances will be much smaller than
                real-to-real NN distances, the same clip will be retrieved
                across independent seeds, and few distinct clips will be hit.

  GENUINE       generated clips land in the right gloss neighbourhood without
                copying one exemplar. NN distances look like real-to-real,
                different seeds retrieve different clips of the same gloss.

  nohup python memorization_check.py \
      --real-dir feats274 --wlasl $SA/WLASL_v0.3.json \
      --glosses seen_glosses.json \
      --gen ft=gen_seen_ft --gen kl01=gen_seen_kl01 --gen base=gen_seen_base \
      > /tmp/memo.log 2>&1 &
"""
import argparse, json, re, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

SLICE = (0, 258)
T = 64
SEED_RE = re.compile(r"_s(\d+)$")


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
        for inst in e.get("instances", []):
            id2g[str(inst.get("video_id", "")).zfill(5)] = e["gloss"]
    keep = set(json.load(open(a.glosses)))

    names, labels, vecs = [], [], []
    for p in sorted(Path(a.real_dir).glob("*.npy")):
        g = id2g.get(p.stem.zfill(5))
        if g in keep:
            names.append(p.stem); labels.append(g); vecs.append(load(p))
    if not vecs:
        sys.exit("empty gallery")
    R = np.stack(vecs)
    print(f"gallery {len(R)} clips / {len(set(labels))} glosses\n")

    # --- baseline: how far is a REAL clip from its nearest OTHER real clip ---
    D = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    real_nn = D.min(1)
    print("=" * 66)
    print("REFERENCE: real-clip nearest-neighbour distances")
    print("=" * 66)
    print(f"  median {np.median(real_nn):.3f}   p10 {np.percentile(real_nn,10):.3f}"
          f"   p90 {np.percentile(real_nn,90):.3f}\n")

    for spec in a.gen:
        name, d = spec.split("=", 1)
        files = sorted(Path(d).glob("*.npy"))
        if not files:
            print(f"{name}: nothing in {d}"); continue

        by_gloss = defaultdict(list)      # gloss -> [(seed, nn_index, dist)]
        dists, retrieved = [], []
        for f in files:
            stem = f.stem
            m = SEED_RE.search(stem)
            seed = int(m.group(1)) if m else 0
            gloss = SEED_RE.sub("", stem)
            q = load(f)
            dd = np.linalg.norm(R - q, axis=1)
            j = int(dd.argmin())
            by_gloss[gloss].append((seed, j, float(dd[j])))
            dists.append(float(dd[j])); retrieved.append(names[j])

        dists = np.array(dists)
        ratio = np.median(dists) / np.median(real_nn)

        # across-seed agreement: do independent seeds hit the SAME clip?
        same, multi = 0, 0
        for g, lst in by_gloss.items():
            if len(lst) < 2:
                continue
            multi += 1
            if len({j for _, j, _ in lst}) == 1:
                same += 1

        print("=" * 66)
        print(f"{name}   ({len(files)} clips)")
        print("=" * 66)
        print(f"  NN distance   median {np.median(dists):.3f}"
              f"   p10 {np.percentile(dists,10):.3f}"
              f"   p90 {np.percentile(dists,90):.3f}")
        print(f"  ratio to real-real median: {ratio:.3f}")
        print(f"  distinct clips retrieved: {len(set(retrieved))} of {len(R)}")
        if multi:
            print(f"  all seeds retrieve the SAME clip: {same}/{multi} glosses "
                  f"({100*same/multi:.0f}%)")
        top = Counter(retrieved).most_common(3)
        print(f"  most-retrieved: {top}")

        print("  VERDICT: ", end="")
        if ratio < 0.5 and multi and same / multi > 0.8:
            print("MEMORIZATION — generated clips are near-duplicates of")
            print("    specific training clips, and every seed returns the same one.")
        elif ratio < 0.7:
            print("LIKELY MEMORIZATION — NN distances well below real-real.")
        elif multi and same / multi > 0.8:
            print("SUSPICIOUS — seeds collapse onto one exemplar per gloss.")
        else:
            print("consistent with genuine gloss-level learning.")
        print()

    print("A ratio near 1.0 with seeds spread across different clips is what")
    print("real generalisation looks like. A ratio well below 1.0 means the")
    print("model is reproducing training clips, and 0.920 top-1 is recall,")
    print("not sign production.")


if __name__ == "__main__":
    main()
