
import argparse, json, subprocess, sys, tempfile, os, re, time
from pathlib import Path
from statistics import mean, pstdev

CROP = {"s1": "276:345:217:147", "s2": "318:397:196:124",
        "s3": "322:402:194:127", "s4": "276:345:217:148",
        "s5": "308:385:201:126"}

# Deliberate mismatches. Kept away from near-synonyms and from signs that share
# a handshape or location, so a low score means "different sign", not
# "different-but-similar sign".
MISMATCH = [
    ("cold",    "work"),
    ("rain",    "tea"),
    ("today",   "finish"),
    ("weather", "drink"),
    ("good",    "rain"),
    ("finish",  "morning"),
    ("tea",     "cold"),
    ("drink",   "today"),
    ("work",    "weather"),
    ("morning", "good"),
]

PEAKS_RE = re.compile(r"PEAKS(.*?)\|\s*MAX\s*([-\d.]+)")


def prep(src, dst, crop):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src),
                    "-vf", f"crop={crop},scale=256:256,setpts=4*(PTS-STARTPTS)",
                    "-r", "25", "-an", str(dst)],
                   check=True, stdin=subprocess.DEVNULL)


def score(bsldict, clip, keyword):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "demo.py", "--input_path", str(clip),
                        "--keyword", keyword],
                       cwd=str(Path(bsldict) / "demo"), env=env,
                       stdin=subprocess.DEVNULL, capture_output=True, text=True)
    m = PEAKS_RE.search(r.stdout)
    if not m:
        print(f"    !! no PEAKS for {clip.name} / {keyword}")
        return None
    return float(m.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="outputs/hq/raw")
    ap.add_argument("--bsldict", required=True)
    ap.add_argument("--scores", default="spotter_scores.json",
                    help="same-gloss scores from multi_subject_spotter.py")
    ap.add_argument("--signers", nargs="+", default=["s1", "s3", "s5"],
                    help="subset is enough; 3 signers x 10 pairs = 30 comparisons")
    a = ap.parse_args()

    raw = Path(a.raw_dir)
    tmp = Path(tempfile.mkdtemp(prefix="negctl_"))
    neg = []

    for gloss, wrong in MISMATCH:
        print(f"\n=== clip={gloss}  scored as '{wrong}' ===", flush=True)
        for sg in a.signers:
            hits = sorted(raw.glob(f"*{sg}*{gloss}*.mp4")) or \
                   sorted(raw.glob(f"*{gloss}*{sg}*.mp4"))
            if not hits:
                print(f"  {sg}: no render"); continue
            clip = tmp / f"{sg}_{gloss}_as_{wrong}.mp4"
            prep(hits[0], clip, CROP[sg])
            s = score(a.bsldict, clip, wrong)
            if s is not None:
                neg.append(s)
                print(f"  {sg}: {s:.4f}", flush=True)
            time.sleep(1)

    if not neg:
        sys.exit("no scores collected -- check the filename glob and that this "
                 "is running inside an activated bsldict_env shell")

    pos = []
    if Path(a.scores).exists():
        d = json.load(open(a.scores))
        pos = [v for g in d.values() for v in (g.values() if isinstance(g, dict) else [g])]

    print("\n" + "=" * 68)
    print("NEGATIVE CONTROL IN THIS PREP")
    print("=" * 68)
    print(f"  different-gloss  n={len(neg):3d}  mean {mean(neg):.4f}  "
          f"sd {pstdev(neg):.4f}  range {min(neg):.4f}-{max(neg):.4f}")
    if pos:
        print(f"  same-gloss       n={len(pos):3d}  mean {mean(pos):.4f}  "
              f"sd {pstdev(pos):.4f}  range {min(pos):.4f}-{max(pos):.4f}")
        sep = mean(pos) - mean(neg)
        better = sum(1 for p in pos for q in neg if p > q)
        auc = better / (len(pos) * len(neg))
        print(f"\n  separation {sep:+.4f}")
        print(f"  P[same > different] = {auc:.3f}"
              f"   (story1 prep gave 0.854)")
        print("\n  VERDICT: ", end="")
        if auc >= 0.80 and sep > 0.10:
            print("SEPARATES. The metric discriminates in this prep;")
            print("    the cross-signer stability result stands as written.")
        elif auc >= 0.65:
            print("WEAK separation. Report the cross-signer comparison as")
            print("    relative only, and state this AUC beside it.")
        else:
            print("COLLAPSED. These scores cannot support any absolute claim")
            print("    about sign correctness. Keep only the relative")
            print("    cross-signer variation, and say the prep was not")
            print("    validated for absolute scoring.")
    else:
        print(f"\n  (no {a.scores} found -- compare by hand against 0.477-0.610)")


if __name__ == "__main__":
    main()
