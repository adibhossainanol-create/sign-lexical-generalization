#!/usr/bin/env python3
"""
make_scaling_splits.py

Builds the training splits for the vocabulary scaling curve.

THE QUESTION: held-out correctness is at chance. Is that because 99 glosses is
too small a vocabulary to generalise from, or simply because 794 clips is not
much data? Those are different papers and the curve has to separate them.

FIVE ARMS:
  vocab25            25 glosses, all their clips      (~200 clips, ~8/gloss)
  vocab50            50 glosses, all their clips      (~400 clips, ~8/gloss)
  vocab99            99 glosses, all their clips      ( 794 clips, ~8/gloss)
  vocab99_budget200  99 glosses, clips matched to vocab25  (~2/gloss)
  vocab99_budget400  99 glosses, clips matched to vocab50  (~4/gloss)

  vocab25 vs vocab99           : vocabulary AND data grow -> the raw trend
  vocab50 vs vocab99_budget400 : SAME data, 2x vocabulary -> isolates vocabulary
  vocab99 vs vocab99_budget400 : SAME vocabulary, 2x data -> isolates volume

vocab99_budget400 (~4/gloss) is the informative control; budget200 (~2/gloss)
is kept as the thinner end of the same axis but is likely too sparse to read.

The gloss sets are NESTED against one shuffled ordering, so
vocab25 c vocab50 c vocab99 -- otherwise the curve confounds vocabulary size
with WHICH glosses happened to be drawn.

Reads the real training split (datasets/annotations/wlasl274/splits/train.txt,
794 clips over 99 glosses) -- the same data train.py consumes. NOT
adapter_data_gloss.npz, which is the SignAvatars adapter subset (694 clips) and
is never read by train.py.

The 25 HELD-OUT glosses are untouched: no arm may contain a clip whose gloss is
held out, and that is asserted per arm rather than assumed.

  python make_scaling_splits.py
  # then, per arm:
  python train.py run_name=scale_vocab25 train_split=vocab25 ckpt=null
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datasets/annotations/wlasl274",
                    help="holds annotations.json and splits/")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--held-split", default="test_unseen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the arms without writing split files")
    a = ap.parse_args()

    root = Path(a.data_dir)
    splits = root / "splits"
    ann = json.load(open(root / "annotations.json"))

    def read_ids(name):
        return [ln.strip() for ln in open(splits / f"{name}.txt") if ln.strip()]

    train_ids = read_ids(a.train_split)
    held_ids = read_ids(a.held_split)
    held_glosses = {ann[k]["gloss"] for k in held_ids}

    by_gloss = defaultdict(list)
    for k in train_ids:
        by_gloss[ann[k]["gloss"]].append(k)
    train_glosses = sorted(by_gloss)

    leak = set(train_glosses) & held_glosses
    if leak:
        raise SystemExit(f"source split is already contaminated: {sorted(leak)[:5]}")
    print(f"source {a.train_split}.txt: {len(train_ids)} clips, "
          f"{len(train_glosses)} glosses, {len(train_ids)/len(train_glosses):.1f}/gloss")
    print(f"held out ({a.held_split}.txt): {len(held_ids)} clips, "
          f"{len(held_glosses)} glosses -- untouched\n")

    # ---- one ordering, used for every nested arm -----------------------------
    rng = np.random.default_rng(a.seed)
    order = sorted(train_glosses)
    rng.shuffle(order)

    arms = []
    for n in (25, 50, len(order)):
        keep = order[:n]
        ids = [i for g in keep for i in by_gloss[g]]
        arms.append((f"vocab{n}", keep, sorted(ids)))

    # ---- budget-matched controls: full vocabulary, clip budget of a small arm
    for label, ref in (("vocab99_budget200", "vocab25"),
                       ("vocab99_budget400", "vocab50")):
        budget = len(dict((n, i) for n, _, i in arms)[ref])
        per = max(1, budget // len(order))
        ids = []
        for g in order:
            pool = by_gloss[g][:]
            rng.shuffle(pool)
            ids.extend(pool[:per])
        # top up to the exact budget so the comparison is clip-for-clip
        chosen = set(ids)
        spare = [i for g in order for i in by_gloss[g] if i not in chosen]
        rng.shuffle(spare)
        ids.extend(spare[:max(0, budget - len(ids))])
        arms.append((label, list(order), sorted(ids[:budget])))

    # ---- checks --------------------------------------------------------------
    named = {n: (keep, ids) for n, keep, ids in arms}

    # nesting, which is the whole point of the shared ordering
    g25, g50, g99 = (set(named[f"vocab{n}"][0]) for n in (25, 50, len(order)))
    assert g25 < g50 < g99, "gloss sets are not nested"
    i25, i50, i99 = (set(named[f"vocab{n}"][1]) for n in (25, 50, len(order)))
    assert i25 < i50 < i99, "clip sets are not nested"

    train_id_set, train_gloss_set = set(train_ids), set(train_glosses)
    for name, keep, ids in arms:
        assert len(set(ids)) == len(ids), f"{name}: duplicate ids"
        unknown = set(ids) - train_id_set
        assert not unknown, f"{name}: {len(unknown)} ids outside {a.train_split}.txt"
        bad = {i for i in ids if ann[i]["gloss"] not in train_gloss_set}
        assert not bad, f"{name}: {len(bad)} clips whose gloss is not a training gloss"
        bled = {i for i in ids if ann[i]["gloss"] in held_glosses}
        assert not bled, f"{name}: HELD-OUT LEAKAGE on {len(bled)} clips"

    # ---- report + write ------------------------------------------------------
    print(f"{'arm':20s} {'glosses':>8s} {'clips':>7s} {'clips/gloss':>12s}   note")
    for name, keep, ids in arms:
        note = ""
        if name.startswith("vocab99_budget"):
            ref = "vocab25" if name.endswith("200") else "vocab50"
            note = f"budget matched to {ref} = {len(named[ref][1])} clips"
        print(f"{name:20s} {len(keep):8d} {len(ids):7d} "
              f"{len(ids)/len(keep):12.1f}   {note}")
        if not a.dry_run:
            (splits / f"{name}.txt").write_text("".join(f"{i}\n" for i in ids))

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return

    print(f"\nwrote {len(arms)} splits to {splits}/")
    print("Nested by construction; no arm contains a held-out gloss (asserted).")
    print("\nNEXT: train each arm FROM SCRATCH (scratch beat pretrained on seen")
    print("glosses, and it removes the BSL-initialisation confound from the curve).")
    for name, _, _ in arms:
        print(f"  python train.py run_name=scale_{name} train_split={name} ckpt=null")
    print(f"Then evaluate every arm on the SAME {len(held_glosses)} held-out "
          f"glosses, 6 seeds.")


if __name__ == "__main__":
    main()
