#!/usr/bin/env python3
"""
retrieval_stats.py

Takes the per-clip hits_seen_<name>.json files (gloss -> [0/1, ...]) written by
seen_gloss_retrieval.py and turns them into confidence intervals plus a
two-proportion z-test against a baseline config, so the top-1 numbers in the
paper can be reported with error bars and a significance flag instead of bare
point estimates.

  python retrieval_stats.py \
      --hits ft=hits_seen_ft.json --hits pres01=hits_seen_pres01.json \
      --hits scratch=hits_seen_scratch.json --hits kl01=hits_seen_kl01.json \
      --hits pres05=hits_seen_pres05.json --hits pres20=hits_seen_pres20.json \
      --hits base=hits_seen_base.json \
      --baseline base --chance 0.0446
"""
import argparse
import json

import numpy as np

ORDER = ["base", "ft", "scratch", "pres01", "kl01", "pres05", "pres20"]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((center - half) / denom, (center + half) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", action="append", required=True, metavar="NAME=FILE")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--chance", type=float, required=True)
    a = ap.parse_args()

    data = {}
    for spec in a.hits:
        name, path = spec.split("=", 1)
        by_gloss = json.load(open(path))
        flat = [h for v in by_gloss.values() for h in v]
        data[name] = np.array(flat, dtype=float)

    if a.baseline not in data:
        raise SystemExit(f"--baseline {a.baseline} not among --hits names {list(data)}")
    base = data[a.baseline]
    n_base, k_base = len(base), int(base.sum())
    p_base = k_base / n_base

    order = [n for n in ORDER if n in data] + [n for n in data if n not in ORDER]

    print(f"{'config':10s} {'n':>5s} {'top-1':>7s} {'95% CI':>17s} "
          f"{'z vs chance':>12s} {'z vs '+a.baseline:>12s}")
    for name in order:
        h = data[name]
        n, k = len(h), int(h.sum())
        p = k / n
        lo, hi = wilson_ci(k, n)

        se_chance = np.sqrt(a.chance * (1 - a.chance) / n)
        z_chance = (p - a.chance) / se_chance

        if name == a.baseline:
            z_base_str = "—"
        else:
            p_pool = (k + k_base) / (n + n_base)
            se_diff = np.sqrt(p_pool * (1 - p_pool) * (1 / n + 1 / n_base))
            z_base = (p - p_base) / se_diff if se_diff > 0 else float("nan")
            z_base_str = f"{z_base:+.2f}{'  *' if abs(z_base) > 1.96 else ''}"

        star = "  *" if abs(z_chance) > 1.96 else ""
        print(f"{name:10s} {n:5d} {p:7.3f}   [{lo:.3f}, {hi:.3f}] "
              f"{z_chance:+9.2f}{star:3s} {z_base_str:>15s}")

    print(f"\nbaseline = {a.baseline} (top-1 {p_base:.3f}, n={n_base}), chance = {a.chance:.4f}")
    print("95% CI is a Wilson score interval on top-1 accuracy.")
    print("'z vs chance' and 'z vs baseline' use a normal-approximation")
    print("two-proportion test; '*' marks |z| > 1.96 (p < 0.05).")
    print("Caveat: hits within a gloss (across seeds) are not independent,")
    print("so these intervals are optimistic relative to a per-gloss")
    print("clustered/bootstrap estimate — treat as a lower bound on uncertainty.")


if __name__ == "__main__":
    main()
