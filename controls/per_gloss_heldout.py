
import argparse, json
from collections import defaultdict
from statistics import mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", action="append", required=True, metavar="NAME=FILE")
    ap.add_argument("--chance", type=float, default=0.040)
    a = ap.parse_args()

    models = {}
    for spec in a.hits:
        n, p = spec.split("=", 1)
        models[n] = json.load(open(p))

    glosses = sorted({g for d in models.values() for g in d})
    names = list(models)

    print(f"{'gloss':18s}" + "".join(f"{n:>9s}" for n in names) + f"{'#models':>9s}")
    print("-" * (18 + 9 * (len(names) + 1)))

    hit_count = defaultdict(int)
    for g in glosses:
        row, k = f"{g:18s}", 0
        for n in names:
            v = models[n].get(g, [])
            m = mean(v) if v else 0.0
            row += f"{m:9.2f}"
            if m > 0:
                k += 1
        hit_count[g] = k
        print(row + f"{k:9d}")

    n_models = len(names)
    ever = [g for g in glosses if hit_count[g] > 0]
    multi = [g for g in glosses if hit_count[g] >= 3]

    print("\n" + "=" * 64)
    print("IS THE FAILURE UNIFORM?")
    print("=" * 64)
    print(f"  glosses ever retrieved by any model : {len(ever)}/{len(glosses)}")
    print(f"  retrieved by >=3 of {n_models} models        : {len(multi)}")
    if multi:
        print(f"    {', '.join(multi)}")
    exp = a.chance * len(glosses)
    print(f"\n  expected correct per model by chance: {exp:.1f} of {len(glosses)}")
    for n in names:
        k = sum(1 for g in glosses if mean(models[n].get(g, [0])) > 0)
        print(f"    {n:10s} {k} correct")

    print("\n  READ: if the successful glosses DIFFER across models, the")
    print("  failure is uniform and the aggregate is honest -- say so. If the")
    print("  SAME few glosses succeed everywhere, those signs are learnable")
    print("  and that is a finding worth naming.")


if __name__ == "__main__":
    main()
