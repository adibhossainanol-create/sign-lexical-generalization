# Known issues

Every item here cost us hours and several produced wrong numbers that survived
into a draft. Read this before debugging.

---

## Things that silently give wrong results

**Two reference populations.** Articulation statistics come from SignAvatars
`adapter_data.npz` (866 clips); retrieval comes from WLASL274 `feats274`
(994 clips, 794 training). They share the same 124-gloss vocabulary and the same
99/25 partition, so it is easy to mix them — and a number computed on one and
described as the other looks perfectly plausible. `motion_stats.py` now prints
its source path and clip count for this reason. Always state which population a
number comes from.

**Feature slice.** Retrieval top-1 is stable to ±0.01 under changes to resample
length and DTW cap, but **moves by up to 0.20 when the feature slice changes**.
Dimensions 0–17 never vary in real clips and distort DTW. Seen and held-out
conditions must be computed through the same code path or they are not
comparable — this invalidated an entire first round of results.

**Hand slices in the 274-d representation.** Left hand `78:168`, right hand
`168:258`, verified against `signavatars_to_handmdm.py`. A wrong slice can still
print a clean "15/15 joints" while mixing hand data with jaw/expression channels.

**Per-hand vs pooled targets.** `A* = 0.024` is the *pooled* mean. The per-hand
targets are 0.0199 (left) and 0.0281 (right) — real signing is asymmetric by
1.41×. Calibrating both hands to the pooled value over-corrects one and
under-corrects the other.

**Shared checkpoint directory.** Lightning wrote every run to `./checkpoints/`
and disambiguated collisions with `-v1` suffixes. Set
`dirpath: ${run_name}/checkpoints`. Runs before that fix need their provenance
verified by weight distance, not by directory name.

**Seed suffix regex.** Generated files are `<gloss>__s0.npy` — *double*
underscore. A `_s(\d+)$` pattern matches nothing and scripts report "no gloss
overlap" rather than failing.

---

## Environment traps

**Three conda environments**, and stacking them breaks Python resolution:
`handmdm` (generation), `lhm` (rendering), `bsldict_env` (scoring). Run
`conda deactivate` before switching, not `conda activate` on top.

**The spotter must run on CPU.** `bsldict_env`'s torch predates the RTX 3090's
sm_86; `torch.cuda.is_available()` returns True and then every kernel launch
hangs rather than erroring. Check `get_device_capability()` against
`get_arch_list()` — metadata only, no launch.

**Scripts calling `demo.py` must be launched from an activated `bsldict_env`
shell.** Using `conda run` for earlier steps leaves `demo.py` in the wrong
interpreter, and every PEAKS line comes back empty with no error.

**`ffmpeg` and `demo.py` both read stdin.** Add `-nostdin` to ffmpeg and
`< /dev/null` to python calls, or a loop over a manifest will consume its own
input. Read manifests on fd 3.

**Never name a scratch file `inspect.py`** (or `random.py`, `json.py`,
`types.py`) — numpy imports `inspect` internally and picks up the local file.

---

## Measurement discipline

**Scores are not portable across preparations.** Crop, scale and `setpts` all
shift spotter scores substantially. Anything compared must be prepped
identically, and a new prep needs its own negative control before any absolute
claim.

**Noise floor ≈ 0.10, worst case 0.34.** The same sign generated under four
near-identical requests spans this much. Use ≥4 samples per condition and treat
differences below 0.15 as not established.

**Bootstrap over glosses, not clips.** Six seeds of one gloss are not six
independent observations; resampling clips gives intervals that are far too
tight.

**Above-ceiling retrieval is not good news.** Generated clips can beat the
real-clip ceiling by converging on the per-gloss centroid, which is closer to
every member than members are to each other. Check `centroid_check.py` before
celebrating.
