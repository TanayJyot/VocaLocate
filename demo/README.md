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
pip install -e pipeline -e 'demo[dev]'     # from the repo root; needs torch
cd demo
vocalocate-demo -v run                       # full demo, writes demo_out/ (≈10 min on a laptop CPU)
vocalocate-demo -v run --small               # tiny version, about a minute
vocalocate-demo -v run --device cuda --seed 491 --workdir demo_out_491
```

`demo_out/report.md` has the results, `demo_out/results.json` has everything
including training curves. Try the search yourself on the simulated library:

```bash
vocalocate-demo --lake demo_out/lake search --root demo_out/library --name demo \
    --query demo_out/queries/<one of the files>.wav          # compares baseline, A and B
vocalocate-demo --lake demo_out/lake search --root ~/SFX --name mine --query my_whoosh.wav --model metadata
```

`demo_out/queries/queries.csv` says which library file each example query imitates.
A second search on the same folder skips unchanged files and reuses the cached
embeddings.

Other commands: `vocalocate-demo generate --out DIR` writes only the dataset,
and `vocalocate-demo train --source <source>` runs the training comparison on
any source already in a lake. For example, VimSketch after
`vocalocate-pipeline ingest-dataset --source vimsketch`, with
`--frames 1000` for its 10-second clips.

## Results

RESULTS_PLACEHOLDER

## Limits and next steps

* The encoders are a small CNN trained from scratch so the demo runs on a CPU.
  `model.Encoder` is the only thing to swap for the AudioSet-pretrained
  MobileNetV3 from the QVIM baseline.
* The data is synthetic. With real data, approach B needs real metadata.
  VimSketch labels are one per reference, so they make no related pools. The
  Vocal Imitation Set's same-class recordings and AudioSet classes, or FSD50K
  tags, should go through an `sfx_meta` CSV first. That is the "more
  ingesters" item in the A2 next steps.
* Negatives are mined once from the baseline. Re-mining during fine-tuning
  and tuning `band` / `top_k` on validation are the obvious next experiments.
* Search is a brute-force cosine over cached embeddings. FAISS slots in where
  `search.search` ranks the vectors.
