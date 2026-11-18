# Run SFT and evaluate your baseline

Run both commands from the **VocaLocate repository root**, two levels above this README. The launcher uses the Nanochat code in `a3/nanochat/`. Use whichever authenticated Modal CLI you have set up. The examples use `./a3/modal/.venv/bin/modal`, which you can replace with `.venv/bin/modal`.

## Before you start

Both modes require four values that identify your **pretrained** checkpoint. Use the same values in both commands.

| Flag | What to enter | Original d20 example |
|---|---|---|
| `--volume` | The Modal volume that holds your files | `a4-nanochat-volume` |
| `--base-path` | The folder inside that volume that contains `base_checkpoints/` and `tokenizer/` | `a4` |
| `--model-tag` | The subfolder under `base_checkpoints/` that holds your checkpoint | `d20-baseline-8-H100` |
| `--model-step` | The step shared by your model and metadata filenames, as an integer | `9960` |

If you downloaded the files to your machine, upload them to your Modal volume in this layout. Keep the checkpoint filenames and use the tokenizer that belongs to the baseline.

```text
<base-path>/
  base_checkpoints/<model-tag>/
    model_<six-digit-step>.pt
    meta_<six-digit-step>.json
    optim_<six-digit-step>_rank<N>.pt  # one per rank when using --load-optimizer
  tokenizer/
    tokenizer.pkl
    token_bytes.pt
```

SFT loads the optimizer by default, so upload the baseline optimizer shards for the same checkpoint step alongside the weights and metadata. With eight GPUs, include ranks 0 through 7. Evaluation does not need optimizer shards; to run SFT without them, pass `--no-load-optimizer`.

## 1. Train SFT from your baseline

Replace the example values below with yours. Choose a new output tag for each run.

```bash
./a3/modal/.venv/bin/modal run --detach a4/part-2-sft/modal_chat_sft.py \
  --volume YOUR_VOLUME \
  --base-path YOUR_BASE_PATH \
  --model-tag YOUR_BASELINE_TAG \
  --model-step YOUR_BASELINE_STEP \
  --output-tag YOUR_SFT_OUTPUT_TAG \
  --run-name YOUR_WANDB_RUN_NAME \
  --gpus 8 \
  --wandb-mode online \
  --wandb-project YOUR_WANDB_PROJECT \
  --wandb-entity YOUR_WANDB_ENTITY \
  --wandb-secret YOUR_MODAL_SECRET
```

By default, SFT loads the matching optimizer shards for the baseline checkpoint. Upload those shards alongside the weights and metadata (for eight GPUs, include ranks 0 through 7). If you do not want to load the optimizer, or only have weights and metadata, pass `--no-load-optimizer`. For online logging, the named Modal secret must contain `WANDB_API_KEY`. For offline logging, use `--wandb-mode offline` and omit the entity and secret flags. Keep the project flag.

The trained checkpoint goes to `<base-path>/sft_runs/<output-tag>/chatsft_checkpoints/<model-tag>/` on the volume. The training log and run summary go under `<base-path>/logs/` and `<base-path>/results/`.

## 2. Evaluate the pretrained baseline on ChatCORE

Use the same volume, base path, model tag, and step from before SFT. This evaluation can run before or after training. It reads the pretrained checkpoint from `base_checkpoints/`, and needs only the checkpoint and tokenizer.

```bash
./a3/modal/.venv/bin/modal run a4/part-2-sft/modal_chat_sft.py \
  --eval-pretrained \
  --volume YOUR_VOLUME \
  --base-path YOUR_BASE_PATH \
  --model-tag YOUR_BASELINE_TAG \
  --model-step YOUR_BASELINE_STEP \
  --gpus 8 \
  --eval-cache-tag YOUR_SFT_OUTPUT_TAG \
  --eval-log a4/part-2-sft/YOUR_BASELINE-chat-eval.log
```

Keep the terminal attached until the local log is saved. Choose a fresh log filename on each run, or omit `--eval-log` to generate one automatically. Use `--gpus 4` if the eight-H100 request is queued.

The evaluator runs the full ARC-Easy, ARC-Challenge, and MMLU test sets and 24 examples each from GSM8K and HumanEval, matching the standard SFT task limits. To reuse cached test data, pass your SFT output tag with `--eval-cache-tag`. Omit it to download test data into temporary container storage. If a selected cache lacks a task manifest, the evaluator downloads that task. The log records the data source. The log reports per-task accuracy, five-task ChatCORE, and categorical ChatCORE. Compare these with the corresponding SFT scores. The A3 CORE score comes from a different benchmark suite.


## Defaults

- Both modes request eight strict H100s. Use `--gpus` to select 1, 2, 4, or 8.
- Training defaults to loading matching optimizer shards and using offline W&B. It requires `--output-tag`, `--run-name`, and `--wandb-project`. Extra SFT arguments and W&B entity and secret are optional. Online W&B requires the secret.
- Evaluation uses one greedy completion per generative question, temperature 0, a 512-token limit, and the batch size from checkpoint metadata. Without `--eval-log`, it creates a unique log in your current directory.

For a separate GSM8K evaluation of a saved SFT checkpoint, see the [GSM8K replay guide](modal_gsm8k_README.md).
