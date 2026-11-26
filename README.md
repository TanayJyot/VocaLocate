# VocaLocate

VocaLocate lets a video editor or sound designer find a sound effect in their own library by
imitating it with their voice ("pshhh", "bonk"), optionally adding a short text hint such as
"metallic". This repository holds the CSC490 project: the data pipeline that prepares audio for
the search model, the AWS infrastructure that runs it, and the model experiments and local search
app built on top.

## Code coverage

<!-- coverage:start -->
| Python source | Covered lines | Total lines | Coverage |
| --- | ---: | ---: | ---: |
| Pipeline | 1369 | 1422 | 96.27% |
| Demo | 1970 | 2239 | 87.99% |
| Modal training | 0 | 84 | 0.00% |
| **Combined** | 3339 | 3745 | **89.16%** |

Scope: first-party VocaLocate Python code. Frontend JavaScript coverage is unmeasured. Terraform is checked separately by the infra workflow. Vendored EfficientAT and the A3/A4 Nanochat experiments are excluded.
<!-- coverage:end -->

GitHub Actions runs both pytest suites, then combines their line coverage for
`pipeline/vocalocate_pipeline/`, `demo/src/vocalocate_demo/`, and the Modal training
entry point in `demo/modal/`. The report includes Python files that the tests never
execute, so an untested script counts against the combined percentage.

The Python report excludes vendored EfficientAT and the separate A3/A4 Nanochat
experiments. The search UI’s browser JavaScript is part of VocaLocate and its
coverage is currently unmeasured. The independent [infra workflow](.github/workflows/infra.yml)
runs `terraform validate` and `terraform test` for the project’s AWS deployment.

The `code-coverage` artifact includes per-file results. Same-repository PRs get a
coverage comment, and the Actions summary also shows the totals for fork PRs.

## Where each assignment part lives

| A2 part | What it is | Where |
| --- | --- | --- |
| 1. Aspirational datasets | Wish-list schemas | The report; the real tables are in [`pipeline/vocalocate_pipeline/schemas.py`](pipeline/vocalocate_pipeline/schemas.py), and one warehouse view per Part One schema (`v_sfx_file`, `v_retrieval_query`, ...) in [`warehouse.py`](pipeline/vocalocate_pipeline/warehouse.py) |
| 2. Reality check | Datasets we can use | The report; one ingester per chosen dataset in [`pipeline/vocalocate_pipeline/ingest/`](pipeline/vocalocate_pipeline/ingest) |
| 3. Data-processing pipeline | Ingest, clean, transform, warehouse | [`pipeline/`](pipeline) |
| 4. Infrastructure as code | Terraform for AWS, dev and prod | [`infra/`](infra) |
| 5. Disaster recovery | Delete prod and restore it | [`infra/disaster-recovery.md`](infra/disaster-recovery.md) |
| (A1) | Project proposal | [`a1/a1.pdf`](a1/a1.pdf) |

## Repository map

```
VocaLocate/
├── pipeline/                     Part 3: the data pipeline (Python package vocalocate_pipeline)
│   ├── vocalocate_pipeline/      the code, read in this order:
│   │   ├── cli.py                  the vocalocate-pipeline command and its sub-commands
│   │   ├── config.py               every tunable setting, and the config fingerprint
│   │   ├── ingest/                 1. Ingest: one small reader per dataset (see below)
│   │   ├── flow_functions/         the per-clip steps the flows run:
│   │   │   ├── audio.py              2. Clean: mono, 32 kHz, trim, loudness; reject or flag bad clips
│   │   │   ├── features.py           3. Transform: log-mel features, as the QVIM baseline computes them
│   │   │   ├── splits.py                train/val/test by a seeded hash of each reference sound
│   │   │   └── pairs.py                 links each imitation to its reference sound
│   │   ├── flows.py                the two flows (dataset, library) that run stages 1-4 with caching
│   │   ├── schemas.py              the columns and types of every lake table
│   │   ├── lake.py                 4. Load: where each file goes in bronze/silver/gold
│   │   ├── warehouse.py               the DuckDB warehouse and its SQL views
│   │   ├── sync.py                 copy lake files between a local folder and S3
│   │   └── remote.py               what the AWS tasks run: pull from S3, run a flow, push back
│   ├── tests/                    pytest suite on small fake datasets (no downloads)
│   ├── configs/                  settings: default.toml, dev.toml, prod.toml
│   ├── docs/diagrams/            architecture, AWS deployment and library-flow figures
│   ├── Dockerfile                the container image the AWS tasks run
│   └── README.md                 pipeline details: layers, tables, views, commands
├── infra/                        Parts 4-5: Terraform
│   ├── bootstrap/                once per account: state bucket, OIDC providers, budget
│   ├── backup/                   prod's backup buckets in us-east-2 (separate state)
│   ├── envs/dev/, envs/prod/     one short root per environment
│   ├── modules/                  vocalocate_stack (one environment) and the modules it wires:
│   │                             bucket, network, data_lake, pipeline_compute, orchestration,
│   │                             warehouse_catalog, monitoring, access
│   ├── disaster-recovery.md      the Part 5 runbook
│   └── README.md                 deploying and running the pipeline on AWS
├── demo/                         model training, evaluation and the local search app
│   ├── src/vocalocate_demo/      encoders, negative mining, CLAP text branch, fusion, app
│   ├── modal/recipe_app.py       parallel GPU training on Modal, reading the S3 lake
│   └── README.md                 experiments and results
├── a1/a1.pdf                     the A1 proposal
└── .github/workflows/            CI: pipeline tests, demo tests, Terraform checks, image build
```

### The ingesters

Every dataset has its own way of saying which file is which, so each gets a small module in
[`pipeline/vocalocate_pipeline/ingest/`](pipeline/vocalocate_pipeline/ingest) that only parses that
layout. The shared work (reading index files, recording files, missing files, query and caption
IDs) lives in `IngestBuilder` in [`ingest/__init__.py`](pipeline/vocalocate_pipeline/ingest/__init__.py).

| Module | Dataset | Used for |
| --- | --- | --- |
| `vocal_imitation_set.py` (`vis`) | Vocal Imitation Set v1.1 | Main training set: imitation and reference pairs, distractors, text hints |
| `esc50_voice.py` | ESC-50-Voice (+ the ESC-50 clips it imitates) | Imitation pairs from 8 imitators for 1,240 ESC-50 clips; splits shared with VIS for the same Freesound upload |
| `vimsketch.py` | VimSketch (+ VocalSketch excluded takes) | Broader training set; what the QVIM baseline trained on |
| `qvim_dev.py`, `qvim_eval.py` | QVIM challenge DEV and EVAL | Held-out benchmark (always the test split) |
| `captdure.py` | CAPTDURE | Captions to test the CLAP text branch (always the test split) |
| `esc50.py` | ESC-50 | Fast smoke test of the whole pipeline |
| `team.py` | Our own recordings (`recordings.csv`) | Voice + text + target queries for tuning the fusion weight |
| `sfx_meta.py` | Any set with a `metadata.csv` | The generic path; VIS and VimSketch convert to its rows |

## File guide

One line per file. Paths are relative to the folder in each heading.

### `pipeline/vocalocate_pipeline/` (the pipeline code)

| File | What it does |
| --- | --- |
| `cli.py` | The `vocalocate-pipeline` command: `ingest-dataset`, `index-library`, `load-warehouse`, `report`, `sync`, `remote` |
| `config.py` | Every tunable setting (sample rate, thresholds, FFT size, split seed) and the config fingerprint stored with each result |
| `flows.py` | The main pipeline: `run_dataset_flow` and `run_library_flow` run ingest, clean, transform and load, skipping clips already done |
| `flow_functions/audio.py` | Cleans one clip: decode, mono, 32 kHz, trim silence, loudness; rejects or flags bad clips; SHA-256 file hashing |
| `flow_functions/features.py` | Log-mel spectrogram and 256-value summary vector, matching the QVIM baseline |
| `flow_functions/splits.py` | Assigns train/val/test by a seeded hash of each reference sound; benchmarks always go to test |
| `flow_functions/pairs.py` | Links each clean imitation to the clean reference sound it imitates (the `pairs` table) |
| `ingest/__init__.py` | Shared ingest code (`RawFile`, `IngestBuilder`, `read_table`) and the source registry |
| `ingest/<dataset>.py` | One reader per dataset (see the table above) |
| `schemas.py` | Columns, types and allowed values of all 17 lake tables; every write is checked against them |
| `lake.py` | Where each table and file goes in bronze/silver/gold, and crash-safe Parquet writes |
| `warehouse.py` | Rebuilds the DuckDB warehouse from the lake and defines its 16 SQL views |
| `sync.py` | Copies lake files between a local folder and S3, never overwriting a newer file |
| `remote.py` | What the AWS tasks run: pull the lake from S3, run a flow, push back only this run's outputs |

### `pipeline/` (everything around the code)

| File | What it does |
| --- | --- |
| `pyproject.toml` | Package metadata, dependencies and the `vocalocate-pipeline` command |
| `Dockerfile`, `.dockerignore` | The container image the AWS Fargate tasks run |
| `configs/default.toml` | Default settings for local runs |
| `configs/dev.toml`, `configs/prod.toml` | Settings for the AWS environments; only the lake path and worker count differ |
| `docs/diagrams/*.dot`, `*.png` | Graphviz sources and images: pipeline architecture, AWS deployment, library flow |
| `tests/conftest.py` | Builds small fake datasets laid out like the real ones, so tests run offline |
| `tests/test_audio_features.py` | Cleaning rules (resample, trim, silence, clipping, truncation) and log-mel features |
| `tests/test_core.py` | Config loading, environment configs matching, fingerprint, schema checks |
| `tests/test_ingest.py` | ESC-50, VimSketch and QVIM ingesters read their layouts correctly |
| `tests/test_vocal_imitation_set.py`, `test_captdure.py`, `test_sfx_meta.py`, `test_esc50_voice.py` | The VIS, CAPTDURE, generic metadata and ESC-50-Voice ingesters and their flows |
| `tests/test_splits.py` | Splits are deterministic, grouped by sound, and benchmarks stay in test |
| `tests/test_flows.py` | Whole dataset and library flows: caching, duplicates, pairs, team triples, held-out sets |
| `tests/test_part_one_views.py` | The warehouse views that return each A2 Part One schema |
| `tests/test_sync.py`, `test_sync_s3.py` | S3 sync and the remote flow, against local folders and a local S3 server |
| `README.md` | Pipeline details: layers, tables, views, commands |

### `infra/` (Terraform)

| File | What it does |
| --- | --- |
| `bootstrap/main.tf` | Once per account: Terraform state bucket, GitHub and Modal OIDC providers, monthly budget alert |
| `backup/main.tf` | Prod's backup buckets in us-east-2, in their own state so destroying prod leaves them intact |
| `envs/dev/main.tf`, `envs/prod/main.tf` | Each environment: one call to `vocalocate_stack` with its sizes and safety settings |
| `envs/*/variables.tf`, `outputs.tf`, `versions.tf` | Inputs, printed outputs (bucket names, role ARNs), provider and backend versions |
| `envs/*/terraform.tfvars.example` | Template for the values you fill in before `terraform apply` |
| `modules/vocalocate_stack/` | Everything one environment needs; wires the modules below together. `tests/stack.tftest.hcl` checks it offline |
| `modules/bucket/` | A private, encrypted, TLS-only S3 bucket; every bucket in the project uses it |
| `modules/network/` | A small VPC: two public subnets, outbound-only security group, S3 gateway endpoint |
| `modules/data_lake/` | The raw, lake and models buckets, and replication to the backups (prod) |
| `modules/pipeline_compute/` | ECR repository, ECS cluster, Fargate task definition, task roles, logs |
| `modules/orchestration/` | Step Functions state machine (Ingest, LoadWarehouse, Report, Catalog), upload trigger, schedules |
| `modules/warehouse_catalog/` | Glue crawler and catalog over the lake tables, and an Athena workgroup |
| `modules/monitoring/` | SNS email alerts and CloudWatch alarms on failed runs |
| `modules/access/` | IAM roles GitHub Actions and Modal assume with OIDC tokens; no stored keys |
| `disaster-recovery.md` | The Part 5 runbook: destroy prod, rebuild it, restore its data, verify |
| `README.md` | Deploying and running the pipeline on AWS |

### `demo/` (models and the search app)

| File | What it does |
| --- | --- |
| `src/vocalocate_demo/cli.py` | The `vocalocate-demo` command: `run`, `train`, `benchmark`, `recipe`, `app`, ... |
| `src/vocalocate_demo/data.py` | Reads training pairs, candidate sounds and cached features from the pipeline's warehouse |
| `src/vocalocate_demo/model.py` | The dual encoder: one network for imitations, one for sound effects |
| `src/vocalocate_demo/efficientat/` | The MobileNetV3 audio model used by the QVIM baseline (vendored, MIT licence) |
| `src/vocalocate_demo/train.py` | Contrastive baseline training, then fine-tuning with mined negatives |
| `src/vocalocate_demo/mining.py` | Picks hard-but-wrong sounds to train on: retrieval-driven, metadata-guided or random |
| `src/vocalocate_demo/recipe.py`, `augment.py` | The QVIM 2025 winning recipe and its audio augmentations |
| `src/vocalocate_demo/text.py` | The frozen CLAP model that scores text hints against sounds |
| `src/vocalocate_demo/fusion.py` | Blends voice and text scores with the weight α, tuned on validation |
| `src/vocalocate_demo/evaluate.py` | Retrieval metrics: Recall@K, MRR, nDCG |
| `src/vocalocate_demo/experiment.py`, `demo_flow.py` | The end-to-end experiment and the one-command demo |
| `src/vocalocate_demo/synth.py` | A synthetic sound set with imitations, so the demo runs offline |
| `src/vocalocate_demo/search.py`, `app.py`, `ui/` | Indexes your folders and serves the local search web app |
| `src/vocalocate_demo/report.py`, `summarize.py` | Write results as markdown and average them over seeds |
| `modal/recipe_app.py` | Runs every training in its own Modal GPU container, reading from and writing to S3 |
| `configs/demo.toml`, `configs/vimsketch.toml` | Settings for the synthetic demo and the VimSketch replication |
| `tests/` | Mining, fusion, metrics, the recipe, the app and the full demo end to end |
| `docs/` | App screenshots and a walkthrough video |
| `README.md` | Experiments, how to run them, and results |

### `.github/workflows/` (CI)

| File | What it does |
| --- | --- |
| `coverage.yml` | Runs both Python test suites and publishes coverage to the README and pull requests |
| `infra.yml` | `terraform fmt`, `validate` and module tests on `infra/` changes; `plan` against dev on main |
| `pipeline-image.yml` | Builds the pipeline image; when run by hand, pushes it to an environment's ECR |

## Find things fast

| I want to... | Look at |
| --- | --- |
| Change a cleaning threshold or feature setting | `pipeline/configs/*.toml` and `pipeline/vocalocate_pipeline/config.py` |
| Change how a clip is cleaned, featurised, split or paired | `pipeline/vocalocate_pipeline/flow_functions/` |
| Add a dataset | A new module in `pipeline/vocalocate_pipeline/ingest/`, registered in `ingest/__init__.py` |
| See or change a table's columns | `pipeline/vocalocate_pipeline/schemas.py` |
| Add a SQL view | `VIEWS` in `pipeline/vocalocate_pipeline/warehouse.py` |
| Understand what runs on AWS and when | `infra/modules/orchestration/main.tf` (the Step Functions state machine) |
| Change dev or prod sizes | `infra/envs/dev/main.tf`, `infra/envs/prod/main.tf` |
| Change permissions | `infra/modules/access/` (CI and Modal), `infra/modules/pipeline_compute/` (the task) |

## Quick start

```bash
cd pipeline
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q                       # the whole suite on fake datasets, no downloads

# Real audio: ESC-50 (about 600 MB)
git clone https://github.com/karolpiczak/ESC-50
vocalocate-pipeline --lake lake --workers 4 -v ingest-dataset --source esc50 --path ESC-50
vocalocate-pipeline --lake lake report    # data-quality summary from the warehouse

# Your own sound folder, indexed incrementally
vocalocate-pipeline --lake lake index-library --root ~/SFX --name my_library
```

Deploying to AWS is covered in [`infra/README.md`](infra/README.md); the search app and
experiments in [`demo/README.md`](demo/README.md).

## How data flows

```
datasets / team recordings / editor folders
   │  1. ingest     ingest/*.py                  → bronze: raw_files manifest (raw audio never modified)
   │  2. clean      flow_functions/audio.py      → silver: clips table + clean 32 kHz mono WAV
   │  3. transform  flow_functions/features.py,  → gold:   log-mel features, splits
   │                flow_functions/splits.py,       (pairs and queries in silver)
   │                flow_functions/pairs.py
   ▼  4. load       warehouse.py                 → warehouse/vocalocate.duckdb with SQL views
training (v_training_pairs) · fusion tuning (v_retrieval_query) · search app (v_library_index)
· data-quality report (v_clip_quality)
```

On AWS the same code runs in ECS Fargate tasks started by Step Functions, against an S3 lake with
the same layout; see the diagram in [`infra/README.md`](infra/README.md).
