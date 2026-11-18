# A4 Part 2 Modal setup log

## November 12, 2026

1. Checked the VocaLocate checkout. The existing `a3/modal/train_d12.py` launches A3 pretraining with `scripts.base_train`; the current post-training entry point is `a3/nanochat/scripts/chat_sft.py`. Added `modal_a4.py` for A4 storage setup and reproducible checkpoint transfer.
2. Reused the A3 Modal virtual environment (SDK 1.6.3) and authenticated with the existing Modal profile. Listing the A3 files confirmed access to the `carmenjchau` workspace. The destination did not exist before setup. Ran `modal_a4.py prepare` to create `a4-nanochat-volume` (ID `vo-BRnVP7FxocPIE5vbDufrVy`) in `main`.
3. Ran the two transfers in parallel. The tokenizer subagent used `modal volume put a4-nanochat-volume <local-file> /a4/tokenizer/<filename>` for each file in `VocaLocate/a3/tokenizer`. Downloaded both files from A3 and A4 and verified their SHA-256 values against the local originals. No tokenizer blockers.
4. Ran `modal_a4.py copy-checkpoint`. The first attempt stopped before copying because Modal requires the serialized function and container to use the same Python version. Updated the image to match the launching interpreter and reran. The [successful CPU-only Modal job](https://modal.com/apps/carmenjchau/main/ap-AIsKeMVw2twiwdcQgya4tR) mounted the A3 volume read-only and copied only the step-9960 checkpoint. The copy includes rank 0, which is required alongside ranks 1–7 for the original eight-rank optimizer warm start. The source checkpoint remained available for A3.
5. Verified source and destination contents by SHA-256 in the copy job. After its commit, a separate Modal SDK call listed all 12 destination files at their expected sizes and read the committed metadata. It confirmed step 9,960, tag `d20-baseline-8-H100`, depth 20, context 2,048, vocabulary 32,768, and global batch 524,288 tokens.

## Layout and assignment alignment

- Source checkpoint: `nanochat-a3-d20-ablations:/a3/base_checkpoints/d20-baseline-8-H100/`.
- Destination checkpoint: `a4-nanochat-volume:/a4/base_checkpoints/d20-baseline-8-H100/`.
- Destination tokenizer: `a4-nanochat-volume:/a4/tokenizer/`.
- Mount the destination volume at `/vol`, set `NANOCHAT_BASE_DIR=/vol/a4`, and use `--model-tag=d20-baseline-8-H100 --model-step=9960` in `scripts.chat_sft`.
- This stages the existing pretrained model for Part 2’s first bullet. Lighthouse reused a pinned pretrained checkpoint with the current single-stage SFT workflow, and LectureClip used a Modal wrapper for SFT. The A4 volume uses an assignment directory and nanochat’s standard `base_checkpoints` and `tokenizer` subdirectories. The baseline SFT run, W&B logging, and shared checkpoint evaluation are the next step. GPU training and checkpoint deserialization have not been tested in this setup step.

## Verified files

| File | Bytes | SHA-256 |
|---|---:|---|
| `model_009960.pt` | 2,663,446,486 | `0bb5c834e1a111e963ca3678a26fd57d6185ae97f836baf8d69bc86a00c9dea1` |
| `meta_009960.json` | 1,825 | `4b40c46d5e2af19d095bc13f20b73e6e3cd2eb5e07b1c8e335509abe30e1620e` |
| `optim_009960_rank0.pt` | 495,642,677 | `c07e6fc19e6679152bc75ba7910276a310f4377e61fb7b6eed522034b3ca64f0` |
| `optim_009960_rank1.pt` | 495,642,677 | `a69951ba01c7591f36647e705c3bb6ab9f4bfc4b228d7673b5c79c215627b068` |
| `optim_009960_rank2.pt` | 495,642,677 | `320888aefe70bd65913b37c5b441d00a36950f5d5f2c248a6418c880a69b2414` |
| `optim_009960_rank3.pt` | 495,642,677 | `1bc8e36a3f27d1db9f230eae8187fde44e005de9528249b7fde5317f97d0290a` |
| `optim_009960_rank4.pt` | 495,642,677 | `9f36bafb3d24a74e112d220dd18af7bfc7ac6ee6d080225b8ca5f50fd2a01af7` |
| `optim_009960_rank5.pt` | 495,642,677 | `75864610ef44d1d8a1cad79406ccaaf9fa79cbefe119a1ba0e18888db53db567` |
| `optim_009960_rank6.pt` | 495,642,677 | `706580d329dbc2162b05b8f32cff7f3af8653d8fff62db39d79385a6cdcd2362` |
| `optim_009960_rank7.pt` | 495,642,677 | `dd7c0bf9b5a70c0da5295449c8fbb79e26bd25b5be1948888d764d99e1f70826` |
| `tokenizer.pkl` | 412,009 | `46a1a55e1154ec68b217dde7deffbfeb1f754e911982d9d7f53a7523b356ab89` |
| `token_bytes.pt` | 132,649 | `42317d1460274d9f175fe3532d0cc6a04d06396ed06d3338ce3f733c9f986665` |

## Reproduce

Run from the directory that contains the setup script. The existing Modal profile or standard Modal token environment variables supply credentials. To authenticate a new machine, run `modal setup` first. The optional `--volume NAME` flag selects a different destination for both commands.

```bash
../../a3/modal/.venv/bin/python modal_a4.py prepare
../../a3/modal/.venv/bin/python modal_a4.py copy-checkpoint
```

## SFT launcher preparation

Added `modal_chat_sft.py` in `VocaLocate/a4/part-2-sft`. Its `--output-tag` option selects an isolated run directory while both original and expanded-data SFT runs load the same pretrained checkpoint. The SFT optimizer state should be loaded independently from A3 for each run when enabled. An expanded-data run must not load optimizer state from the original SFT run.

The launcher accepts a Modal volume name, a relative base path inside that volume, an input checkpoint tag and step, a distinct output tag, a W&B run name and mode, a GPU count, an optimizer-loading choice, and extra SFT CLI arguments. `NANOCHAT_REPO` selects a team member's local nanochat checkout for the Modal image. The selected checkout contains the dataset implementation, so a future expanded-data recipe can be supplied through its CLI and `--sft-args`. It mounts the volume at `/vol`, sets `NANOCHAT_BASE_DIR=/vol/<base-path>`, checks the required tokenizer/checkpoint files, and streams the training log to `logs/<output-tag>.log`.

Example from the VocaLocate repository root, once the checkpoint audit and planned smoke-test settings are finalized:

```bash
./a3/modal/.venv/bin/modal run a4/part-2-sft/modal_chat_sft.py \
  --volume a4-nanochat-volume --base-path a4 \
  --model-tag d20-baseline-8-H100 --model-step 9960 \
  --output-tag a4-original-sft --run-name a4-original-sft \
  --gpus 8 --load-optimizer --wandb-mode offline \
  --wandb-project a4-nanochat-rl --wandb-entity carmen-chau-0912
```

For a team member's alternate checkout, set `NANOCHAT_REPO=/absolute/path/to/nanochat` before `modal run`, and pass their volume/base path. For an online W&B run, use `--wandb-mode online --wandb-secret <Modal-secret-name>`, where the secret contains `WANDB_API_KEY`. Offline W&B records require a later `wandb sync`.

The launcher's CLI was checked with `modal run ... --help`. A Modal GPU training invocation has not been made.

## W&B logging revision

The original `chat_sft.py` logs a numeric `step` and estimated training FLOPs. It does not log processed-token counts, and its `wandb.init` call fixes the project to `nanochat-sft`. The launcher now instruments only the copy of that script inside the Modal container. It reads `WANDB_PROJECT` from the launcher and logs `total_training_tokens = args.total_batch_size * step` with each existing W&B record. The local tracked source remains unchanged. The caller must pass `--wandb-project` for each run and may pass `--wandb-entity` when needed. The instrumentation was checked by counting the expected source patterns and parsing the resulting Python. It has not been tested in a training run.

## Checkpoint content audit

The audit subagent compared the staged checkpoint with `assignments/a3/assignment_resources/a3_submission.pdf`. The report's Part 2 is the earlier d12 study. The selected `d20-baseline-8-H100` checkpoint is the plain d20 control in Part 5 Table 5.2. The separate Part 5 Table 5.1 final model is `d20-combined`. Source metadata identifies the plain checkpoint as step 9,960, depth 20, width 1,280, 10 attention heads, context 2,048, vocabulary 32,768, per-device batch 16, global token batch 524,288, and validation BPB 0.73590868. These align with the plain d20 result in the A3 report.

In a read-only [Modal CPU audit](https://modal.com/apps/carmenjchau/main/ap-ViZua2vZNcdD8Q0mgPtAef), PyTorch `torch.load(weights_only=True, mmap=True)` opened the model and every rank-0-through-7 optimizer shard. The model has 147 tensor keys, including a `[32768, 1280]` token embedding, with zero nonfinite tensor values. Every optimizer shard has 21 state entries, 10 parameter groups (six AdamW and four Muon), and 42 tensors with zero nonfinite values. The earlier SHA-256 check also established that each A4 file is byte-identical to the corresponding A3 file.

A stricter check using `model.load_state_dict` and `optimizer.load_state_dict` would upload the local nanochat source to a new Modal image. Automatic approval review rejected that audit job because the upload exceeded the checkpoint audit authorization. The strict restore remains unverified. The planned SFT smoke test is the next practical check of full model and optimizer compatibility.

## Launcher revision after review

Restored `VocaLocate/a3/nanochat/scripts/chat_sft.py` to its original tracked contents. The launcher now gives each SFT run its own `/vol/<base-path>/sft_runs/<output-tag>/` base directory. It links the shared `base_checkpoints` and `tokenizer` inputs into that directory and sets `NANOCHAT_BASE_DIR` to it. Unmodified nanochat therefore writes each run's final model under `sft_runs/<output-tag>/chatsft_checkpoints/<model-tag>/`. The name `--output-tag` is a launcher argument and is no longer passed to `chat_sft.py`. The first GPU smoke test must verify that Modal Volume supports these links and that the final checkpoint persists.

## Local environment and Modal image update

Created `VocaLocate/.venv` using local Python 3.12.4. Installed the runtime dependencies declared in nanochat's `pyproject.toml`, plus Modal 1.6.3 and the request, template, dataset, and Hugging Face packages used by the Modal image. An editable install of nanochat failed because setuptools detected multiple top-level directories in that checkout; installing the declared dependencies directly succeeded without changing the project package configuration. Imports of PyTorch 2.9.1, Modal, W&B, kernels, PyArrow, rustbpe, tiktoken, and datasets succeeded from the new environment.

Changed `modal_chat_sft.py` from `modal.Image.debian_slim(python_version="3.10")` to Python 3.12. The 2026.10 Modal image builder supports Python 3.11 and later. The new `.venv/bin/modal run a4/part-2-sft/modal_chat_sft.py --help` command loads the launcher. A remote GPU run has not been started by this environment setup.
