# VocaLocate demos: semi-hard negative mining

Two working versions of VocaLocate from the "Semi-Hard Negative Mining Approach"
table in *Project Feature - List to Confirm*, built on top of the A2 data
pipeline in [`../pipeline`](../pipeline):

| | **A. Retrieval-driven mining** | **B. Pre-constructed metadata-guided mining** |
| --- | --- | --- |
| Idea | Learn from the real retrieval mistakes the baseline makes | Teach fine distinctions between SFX that metadata says are related, even if they are not current mistakes |
| Negative | A non-target real SFX the baseline already ranks highly (top-K) | A non-target real SFX from a pool built from category / tags before fine-tuning |
| False-negative guard | Drop other takes of the target's recording (metadata) and audio near-duplicates (top 1% SFX-SFX similarity); never take candidates scored above the positive | Other takes of the target's recording are never in the pool; same near-duplicate filter |
| Semi-hard rule | Keep `s_pos − band < s_neg < s_pos`, sample from the band (not just the top wrong result) | Same band on the metadata pool, filled tier by tier: same category + tags → same category → shared tags |
| Training example | same imitation + correct SFX + semi-hard negative SFX | same |
| Code | `mining.py` (`strategy="retrieval"`) | `mining.py` (`strategy="metadata"`) |

Both start from the same QVIM baseline and are fine-tuned the same way
(contrastive + triplet loss). A third fine-tune with **random** real-SFX
negatives is the control from stage 5 of the technical flow: it shows whether
mining helps beyond simply training for longer. Metadata is only read at
training time. Searching a user's library never needs it.

## Models

| Part | Default | Option | Notes |
| --- | --- | --- | --- |
| Voice encoders (imitation + SFX) | small CNN from scratch (`--encoder cnn`) | **`--encoder mobilenetv3`**: EfficientAT `mn10_as`, AudioSet-pretrained | The QVIM challenge baseline's encoder. Both towers start from the same weights and the 960-value pooled features are the embedding. The model code is vendored in `efficientat/` (MIT); weights download to `~/.cache/efficientat` on first use. The CNN stays as the default so tests and CPU demos need no download. |
| Text branch | off | **`--text-branch`**: frozen CLAP, `laion/clap-htsat-unfused` via Hugging Face `transformers` | Stage 6-7: hints such as "metallic short" are scored against every SFX's CLAP audio embedding and fused with the voice score, `α·z(voice) + (1 − α)·z(text)`, with α tuned on validation. We use LAION CLAP instead of Microsoft's msclap (cited in A1) because msclap pins `numpy<2` and needs a CUDA-matched torchaudio, which break the pipeline's pyarrow. LAION-Audio-630K includes AudioCaps, Clotho, Freesound and BBC Sound Effects, so text scores on those sets will look optimistic (A2 Part Two). |

`vocalocate-demo fetch-models` downloads both sets of weights and prints a quick check.

## How it follows the A2 pipeline

```
synthetic SFX + imitations ──► pipeline dataset flow (sfx_meta ingester)
   (A2 "data we can generate")     bronze raw_files → silver clips, pairs, sfx_metadata
                                   → gold log-mel features, grouped splits → DuckDB
                                                       │
            v_training_pairs, v_sfx_candidates ◄───────┘
                     │
   1. baseline dual encoder (contrastive)                         train split
   2-3. mine negatives: retrieval / metadata / random  ──► gold hard_negatives table,
                                                           pipeline_runs (flow "mining")
   4. fine-tune one model per strategy                            train split
   5. keep each model's best epoch by validation MRR              val split only
   8. score the frozen models once                                test split
                     │
   messy library (held-out SFX renamed SFX_0001.wav in project folders)
          ──► pipeline library flow (incremental) ──► gold embeddings table (cached per model)
          ──► search with a held-out imitation
```

What was added to the pipeline for this (all covered by its tests):

* `ingest/sfx_meta.py`: an ingester for any SFX dataset with a metadata CSV
  (`file, role, sound_id, imitates, category, tags, recording_id, ...`). Clips
  cut from one recording share a group key, so they never straddle splits.
* New tables: `sfx_metadata` (silver, the SFX_Metadata wish-list schema),
  `embeddings` and `hard_negatives` (gold; the "mined hard negatives" dataset
  from A2 Part Two, the relevance schema with a model score instead of a rating).
* New views: `v_sfx_candidates` (every clean reference/library SFX with split
  and metadata) and `v_hard_negatives` (mined negatives joined with metadata);
  `v_training_pairs` now carries the feature paths.

## The app

```bash
vocalocate-demo app --workdir demo_out            # after `vocalocate-demo run`; opens on http://127.0.0.1:8765
vocalocate-demo app --workdir demo_mn_491 --text  # MobileNetV3 run with CLAP text hints
vocalocate-demo app --lake demo_out/lake --root ~/SFX --name mine --text
```

A local web UI over the same pipeline and models, following the product flow:

1. **Library**: add folders (paths on this machine, nothing is uploaded). They are indexed by the A2
   library flow (incremental: unchanged files are skipped, moved files are recognised) and embedded
   once per model. Progress shows each folder, its file count and the current step.
2. **Search**: record an imitation in the browser (or upload a file, or pick one of the demo's held-out
   example imitations), add an optional text hint, and pick VocaLocate B, A or the baseline.
3. **Results** (a separate page): file name, project folder, path, similarity, a 5-second preview,
   copy path and drag-and-drop into an editor. Changing the hint or model reruns the search
   without re-embedding the query or the library.

| Search | Results |
| --- | --- |
| ![Search page](docs/app_search.png) | ![Results page](docs/app_results.png) |

The server is standard-library Python (`app.py`) and binds to localhost only. It serves only
files that are in a library index.

## The synthetic data

`synth.py` writes a dataset in the `sfx_meta` layout so the demo runs offline
and in minutes. It copies the structure of the real data:

* 12 categories (metal impact, wood knock, glass clink, whoosh, wind, alarm
  beep, siren, laser zap, engine, rattle, water drip, door creak), each with
  its own generator and tags. Some tags are shared across categories.
* 14 imitated recordings + 4 distractor recordings per category. Every
  recording has a reference take plus a "distant" and a "pitched" take, like
  several takes of one Freesound recording. These are the likely false
  negatives the miners have to avoid.
* 5 imitations per reference from 8 imitators (voice or whistle, different
  vowels and ranges). An imitation keeps the reference's envelope, rhythm and
  pitch contour, folded into the voice range, with a voice timbre, timing and
  pitch errors, background noise and a 16 kHz laptop-mic sample rate.

That is 1,488 files. The pipeline cleans them to 32 kHz mono and computes the
QVIM-baseline log-mels.

## Run it

```bash
pip install -e pipeline -e 'demo[dev,text]'   # from the repo root; needs torch
cd demo
vocalocate-demo -v run                       # CNN, voice only, writes demo_out/ (≈10 min on a laptop CPU)
vocalocate-demo -v run --small               # tiny version, about a minute
vocalocate-demo fetch-models                 # once: MobileNetV3 + CLAP weights
vocalocate-demo -v run --encoder mobilenetv3 --text-branch --sweep --device cuda --seed 491 --workdir demo_mn_491
```

`demo_out/report.md` has the results, `demo_out/results.json` has everything
including training curves. Try the search yourself on the simulated library:

```bash
vocalocate-demo --lake demo_out/lake search --root demo_out/library --name demo \
    --query demo_out/queries/<one of the files>.wav          # compares baseline, A and B
vocalocate-demo --lake demo_out/lake search --root ~/SFX --name mine --query my_whoosh.wav --model metadata
vocalocate-demo --lake demo_mn_491/lake search --root ~/SFX --name mine --query my_clang.wav --text "metallic"
```

`demo_out/queries/queries.csv` says which library file each example query imitates, and its text hint.
A second search on the same folder skips unchanged files and reuses the cached
embeddings.

Other commands: `vocalocate-demo generate --out DIR` writes only the dataset,
and `vocalocate-demo train --source <source>` runs the training comparison on
any source already in a lake. For example, VimSketch after
`vocalocate-pipeline ingest-dataset --source vimsketch`, with
`--frames 1000` for its 10-second clips.

## Results

5 seeds (490–494) on an RTX 3070 Laptop GPU: `vocalocate-demo run --sweep
--device cuda --epochs 60 --ft-epochs 20 --seed <s>`, about 5 minutes per seed.
Each seed's test split has 185 imitation queries against 138 candidate SFX.
Mean ± std over seeds; Δ MRR is each model minus the same seed's baseline.

| model | R@1 | R@5 | MRR | fine MRR | nDCG@10 | Δ MRR vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QVIM baseline | 0.197 ± 0.030 | 0.712 ± 0.038 | 0.423 ± 0.023 | 0.498 ± 0.028 | 0.605 ± 0.015 | – |
| Control: random negatives | 0.205 ± 0.033 | 0.719 ± 0.048 | 0.429 ± 0.025 | 0.503 ± 0.025 | 0.605 ± 0.018 | +0.006 (better on 2/5 seeds) |
| **A. retrieval-driven** | 0.219 ± 0.038 | 0.722 ± 0.046 | 0.434 ± 0.036 | 0.512 ± 0.018 | 0.602 ± 0.031 | +0.011 (3/5) |
| **B. metadata-guided** | 0.224 ± 0.031 | 0.724 ± 0.044 | 0.443 ± 0.034 | 0.517 ± 0.027 | 0.605 ± 0.019 | +0.021 (3/5) |

Mined negatives (seed 491; the other seeds are similar):

| strategy | mean baseline rank | mean gap s+ − s− | relation to target |
| --- | ---: | ---: | --- |
| A. retrieval-driven | 7.1 | 0.13 | 71% same category, 15% unrelated |
| B. metadata-guided | 7.5 | 0.13 | 84% same category, 16% shared tags |
| random | 193 | 0.79 | 68% unrelated |

How to read this:

* The ordering is B > A > random > baseline on R@1, MRR and fine-grained MRR.
  It matches what the mining table predicts: candidate-side negatives that are
  hard and related teach finer distinctions than easy random ones.
* **The gains are small and not yet significant.** Seed-to-seed spread
  (≈0.03 MRR) is larger than the mean gains (+0.01 to +0.02), and each
  approach beats the baseline on only 3 of 5 seeds. nDCG@10 is unchanged:
  pushing apart same-category SFX also pushes down the other takes that nDCG
  counts as usable substitutes. That is the false-negative risk in the doc.
* Validation picked different band / near-duplicate settings on different
  seeds (`summarize` lists them), so the setting matters less than the noise
  at this scale.
* To get a clear answer we need more test queries and a stronger encoder,
  i.e. real data (VimSketch + the Vocal Imitation Set's same-class
  recordings) and the pretrained MobileNetV3, not more seeds of the synthetic set.

The first single-seed CPU run (30 + 12 epochs, no sweep) gave the same
ordering for B but not for A. See the commit history for that table.

## Limits and next steps

* The Results section below is from the small CNN. The MobileNetV3 + CLAP
  results replace it once the GPU runs finish.
* The data is synthetic. With real data, approach B needs real metadata.
  VimSketch labels are one per reference, so they make no related pools. The
  Vocal Imitation Set's same-class recordings and AudioSet classes, or FSD50K
  tags, should go through an `sfx_meta` CSV first. That is the "more
  ingesters" item in the A2 next steps.
* Negatives are mined once from the baseline. Re-mining during fine-tuning
  and tuning `band` / `top_k` on validation are the obvious next experiments.
* Search is a brute-force cosine over cached embeddings. FAISS slots in where
  `search.search` ranks the vectors.
