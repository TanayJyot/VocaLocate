# A5 Part One: profiling five VocaLocate functions

We profiled five functions with Python's built-in profiler (`cProfile` + `pstats`). Each one is
on a path that runs for every clip, every query or every training epoch. We then fixed the main
hotspot in each. The optimised functions return the same results as before (details below).
New tests compare them against verbatim copies of the old code, so a later change cannot
silently alter what gets computed.

| # | Function | Where it runs |
|---|---|---|
| 1 | [`clean_array`](../../pipeline/vocalocate_pipeline/flow_functions/audio.py) | A2 pipeline: cleans every ingested clip and every search query |
| 2 | [`log_mel`](../../pipeline/vocalocate_pipeline/flow_functions/features.py) | A2 pipeline: featurises every clean clip and every query |
| 3 | [`score_metrics`](../../demo/src/vocalocate_demo/evaluate.py) | Validation after every training epoch, the 11-point α sweep in fusion, final test scoring |
| 4 | [`mine`](../../demo/src/vocalocate_demo/mining.py) | Picks the semi-hard negatives before every fine-tuning run |
| 5 | [`_unique_reference_batches`](../../demo/src/vocalocate_demo/train.py) | Shuffles the training pairs into batches for every epoch |

## How to reproduce

```powershell
$env:PYTHONPATH = "pipeline;demo/src"   # use ':' on Linux/macOS
python a5/profiling/profile_functions.py --label after
python -m pstats a5/profiling/results/after/mine.prof   # interactive browsing, or use snakeviz
```

For the "before" numbers, point `PYTHONPATH` at a checkout of `main` from before this change
(we used a separate `git worktree`) and run with `--label before`.

[`profile_functions.py`](profile_functions.py) builds synthetic inputs at the size of our real data
(VimSketch / QVIM scale) with a fixed seed, so the run needs neither the lake nor a GPU. It
profiles each function under `cProfile`. It also times three plain runs without the profiler,
because `cProfile` charges a fixed cost to every Python call, which exaggerates code that makes
millions of small calls (`mine` and the batcher). [`results/before/`](results/before) and
[`results/after/`](results/after) hold the `.prof` dumps, the top-25 tables sorted by cumulative
time, and `summary.json`.

## Results

These are plain timings (best of three, no profiler) on a Windows 11 laptop CPU. We ran the
before and after versions back to back, three rounds in a row. Another job was training on the
same machine, so the table gives the committed round (round 3, the files in `results/`) and the
range over all three rounds. The call counts are exact and do not vary.

| Function | Workload | Before (committed / range) | After (committed / range) | Speed-up | Python calls before → after |
|---|---|---|---|---|---|
| `clean_array` | 200 × 10 s stereo 44.1 kHz clips | 4.15 s / 4.15–8.79 s | 3.52 s / 3.15–3.52 s | **~1.2–1.3×** | 80,621 → 79,821 |
| `log_mel` | 400 × 5 s 32 kHz clips | 4.23 s / 3.81–6.24 s | 1.49 s / 1.19–1.49 s | **~3–5×** | 50,402 → 30,050 |
| `score_metrics` | 1,500 queries × 3,000 candidates | 0.334 s / 0.334–0.348 s | 0.055 s / 0.053–0.056 s | **~6×** | 16,579 → 12,164 |
| `mine` | retrieval + metadata strategies, 1,500 imitations × 1,500 SFX | 2.34 s / 2.34–2.56 s | 0.84 s / 0.79–0.84 s | **~3×** (4.3× under cProfile) | 9,561,175 → 444,248 |
| `_unique_reference_batches` | 12,000 pairs, 3,000 references, batch 32 | 0.080 s / 0.080–0.088 s | 0.010 s / 0.006–0.010 s | **~8–13×** | 2,279,943 → 36,318 |

The profile numbers quoted below come from the committed files.

## Analysis, function by function

### 1. `clean_array`: decode output → 32 kHz mono, trimmed, loudness-normalised

**What the profile showed** ([before/clean_array.txt](results/before/clean_array.txt)).
`resample` (scipy's polyphase `resample_poly`) took 1.33 s of 4.17 s. That is real work and we
kept it. The surprise was numpy's `_mean` at 1.21 s, of which `ndarray.mean` alone was 0.94 s:
the stereo-to-mono mix `data.mean(axis=1)`. Reducing along the short channel axis of a
`(441000, 2)` array makes numpy loop over 441,000 tiny strided rows. A separate micro-benchmark
(`timeit`, best of 5 × 50) put it at **7.2 ms per clip**.

**Fix.** The new `mix_to_mono` adds whole channel columns, one contiguous vector add per
channel, and divides once. It takes **1.1 ms per clip** (0.215 s for all 200 in
[after/clean_array.txt](results/after/clean_array.txt)) and is bit-identical to the old output
on the profiling workload. It accumulates in the input's float type (float64 for integer
input), as `mean` does. We also changed `trim_silence`, which computed `np.abs(x)` twice over
the whole clip, to compute it once (0.272 s → 0.221 s).

**What is left.** `resample_poly` is now about 40% of the time. It re-designs the same Kaiser
FIR filter on every call (`firwin`, about 0.2 s over 200 calls), because every 44.1 kHz file
needs the same 320/441 filter. Caching the taps per `(up, down)` pair would save that. A
dedicated resampler such as soxr would be faster again, but it uses a different filter, so it
changes the samples, and the whole lake would have to be re-cleaned and re-featurised to stay
consistent. We left it.

### 2. `log_mel`: (128, frames) log-mel spectrogram, QVIM baseline settings

**What the profile showed** ([before/log_mel.txt](results/before/log_mel.txt)). 2.88 s of 4.78 s
was *tottime inside `log_mel` itself*, not in the FFT. So the cost was in the numpy expressions
on its own lines. Timing each stage separately found that `mel = banks @ power.T` took
**10–20 ms per clip**. It is a 128 × 513 × 501 product (33 MFLOP) that should take well under
1 ms. Computing the same product as `power @ banks.T` with C-contiguous operands took
**0.46 ms** on the same data. We did not fully isolate why the original order was so slow. With
random data the original order took 2.3 ms, and with real spectra 10–20 ms, so the transposed
operand alone does not explain it. The slow version was also very sensitive to the other job on
the CPU, which is where most of the "before" range comes from. The profile also showed:
* `mel_filterbank` and `np.hanning` rebuilt on every call, though they depend only on the
  settings (400 calls, 0.22 s).
* `numpy.fft.rfft` taking 1.57 s.

**Fix.**
* Compute `(power @ banks_t).T`, with the transposed filterbank stored C-contiguous.
* Cache the transposed filterbank and the window per setting with `functools.lru_cache`. The
  cached arrays are read-only so no caller can corrupt them (tested).
* Use `scipy.fft.rfft` (scipy is already a pipeline dependency) instead of `numpy.fft.rfft`:
  1.57 s → 0.43 s for the 400 clips.

**Output change.** The features differ from the old ones by at most 4.4e-5, in a feature range
of about −1.4 (silence) to +2, from float32 rounding in a different summation order. The test allows
2e-4. The lake decides whether to re-featurise a clip from the settings fingerprint
(`PipelineConfig.fingerprint`), not from the code. Clips already featurised keep their old
features, and new clips get the new ones. We judged a 4e-5 difference too small to matter for
retrieval and did not bump `FEATURE_VERSION`. That label is stored with each feature row, but
nothing compares or hashes it.

**What is left.** In [after/log_mel.txt](results/after/log_mel.txt), 0.97 s of 1.54 s is still
the element-wise work in the body: windowing, `abs()**2`, `log`. Batching many clips into one
call, or computing on the GPU with `torch.stft` as the QVIM baseline does in training, would be
the next step.

### 3. `score_metrics`: Recall@K, MRR, fine-grained MRR, graded nDCG@10

**What the profile showed** ([before/score_metrics.txt](results/before/score_metrics.txt)).
0.217 s of 0.346 s was tottime in the function body: two Python loops over the 1,500 queries.
The fine-MRR loop rebuilt a `cats == cats[t]` mask per query. The nDCG loop allocated a
3,000-element relevance vector and **sorted it** for every query just to read the top 10.
Another 0.102 s was a full `argsort` of the 1,500 × 3,000 score matrix, also only to read 10
columns per row. This function runs after every training epoch and 11 times in the fusion α
sweep, so it adds up.

**Fix.**
* Categories and recordings become integer codes (`value_codes`), so the per-query masks
  become one `(n_q, n_c)` integer comparison. A `None` category still equals `None`, as in the
  old object-array comparison; a `None` recording matches nothing, as before.
* Fine MRR is a single masked count over the matrix.
* `np.argpartition` finds each row's top 10 in O(n_c), and only those 10 get sorted.
* The ideal DCG needs no sort at all. The ideal ranking is always the target (grade 3), then the
  other takes of its recording (grade 2), then the rest of its category (grade 1), so it only
  needs those two counts per query.

Metrics agree with the old loop to 1e-12. The tests cover missing categories and recordings,
recordings shared across categories, fewer than 10 candidates and a single candidate.

### 4. `mine`: candidate-side semi-hard negative mining

**What the profile showed** ([before/mine.txt](results/before/mine.txt)). 9.56 million function
calls for one retrieval run and one metadata run. 4.5 million of them were `relation()`
(2.41 s cumulative), plus 4.5 million `len()` calls (0.39 s). For every imitation, `mine`
recomputed the relation of its positive to **every** SFX in the pool in Python:
`[relation(target_meta, m) for m in pool_meta]`. On top of that:
* The retrieval strategy walked the *entire* ranked row in Python
  (`[int(j) for j in order[i] if j != p][:top_k]`) to keep 20 entries.
* The metadata strategy built its three tiers with a Python loop over the whole pool per
  imitation.

**Fix.**
* `_PoolRelations` computes all relations to a target at once with numpy: integer codes for
  category and recording, and a multi-hot tag matrix whose product with the target's row
  counts shared tags. Imitations that share a positive reuse the cached result.
* Retrieval reads only the first `top_k + 1` ranked entries, since the positive can occupy at
  most one of them.
* Metadata tiers and the semi-hard band test are boolean masks.

We ran all three strategies (`retrieval`, `metadata`, `random`) on the old and new code, with
the default `MiningConfig` and with a non-default one (`top_k=5`, `min_shared_tags=1`, metadata
filter off). In all six cases the mined rows, negatives and statistics were **identical**,
including the random-number draws, because selection order is preserved. The committed tests
check `_PoolRelations` against `relation` for every pair, and check mining determinism and
validity for all three strategies.

**What is left.** In [after/mine.txt](results/after/mine.txt), `np.full(..., dtype=object)` for
the relation labels takes 0.22 s and the full `argsort` 0.12 s. The full sort has to stay,
because `rank_of` records the rank of every chosen negative, including metadata negatives deep
in the list.

### 5. `_unique_reference_batches`: shuffle pairs into batches with no repeated reference

**What the profile showed** ([before/unique_reference_batches.txt](results/before/unique_reference_batches.txt)).
2.26 million `len()` calls for 12,000 pairs. The first-fit loop scanned every batch created so
far, **including full ones**, for each pair, so an epoch's shuffle cost O(pairs × batches),
about O(n²/32). It runs once per epoch, before training starts, for every epoch: 30 baseline
epochs plus 12 per fine-tuning run.

**Fix.** Keep a list of only the *open* (not yet full) batches, in creation order, and drop a
batch from it once it fills. A full batch could never accept another pair, so every pair still
lands in the same batch. The output is identical, and the cost is now proportional to the
number of open batches, which stays small. The tests also cover `batch_size = 1` and an
empty dataset.

## Tests added

* [`pipeline/tests/test_profiling_optimizations.py`](../../pipeline/tests/test_profiling_optimizations.py) (19 tests):
  * `log_mel` against the old implementation for several lengths (including 1 sample) and settings.
  * Read-only, per-setting caches.
  * `mix_to_mono` against `mean` for mono, stereo, 5.1, float64 and int16 input, and no mutation of the input.
  * `trim_silence` against the old version for empty, silent, burst and noise clips.
* [`demo/tests/test_profiling_optimizations.py`](../../demo/tests/test_profiling_optimizations.py) (16 tests):
  * `score_metrics` against the old loops on four shapes.
  * A hand-built perfect ranking that must score 1.0.
  * `_PoolRelations` against `relation` for every pair at three tag thresholds.
  * Mining determinism, with no positive or same-recording negatives, for all three strategies.
  * The batcher against the old version on five shapes.

The existing pipeline and demo suites pass with these changes, and so do the extra tests in the
open coverage PR (`feat/a5-part3-test-coverage`) when run against them. The only failures are
three pipeline tests that compare paths using `/`. They fail the same way on unmodified code on
Windows and are not related to this change:
* `test_clips_record_the_original_format_bit_depth_and_noise_floor`
* `test_vis_maps_ontology_freesound_ids_and_descriptions`
* `test_vis_finds_nested_audio_and_reports_missing_files` (from the coverage PR)
