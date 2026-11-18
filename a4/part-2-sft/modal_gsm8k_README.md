# Replay GSM8K on a saved SFT checkpoint

Run from the repository root with an authenticated Modal CLI. The launcher uses the Nanochat checkout at `a3/nanochat/`. Set `NANOCHAT_REPO` to point to a different checkout.

## Checkpoint layout

Use the same volume, base path, model tag, and output tag that you used for SFT. Pass the **saved SFT step**, which is `932` in the example. Put downloaded files on Modal under these paths, replacing the placeholders with values from your run:

```text
<base-path>/tokenizer/tokenizer.pkl
<base-path>/sft_runs/<sft-output-tag>/chatsft_checkpoints/<model-tag>/model_<six-digit-step>.pt
<base-path>/sft_runs/<sft-output-tag>/chatsft_checkpoints/<model-tag>/meta_<six-digit-step>.json
```

Use the tokenizer that belongs to the checkpoint. The replay reads the volume and saves its output locally. It requires the model weights and metadata, with no optimizer shards or W&B setup.

## Run the replay

The original command was `.venv/bin/modal run a4/part-2-sft/modal_gsm8k_replay.py`. Its new equivalent supplies the checkpoint explicitly:

```bash
.venv/bin/modal run a4/part-2-sft/modal_gsm8k_replay.py \
  --volume a4-nanochat-volume \
  --base-path a4 \
  --sft-output-tag a4-original-sft \
  --model-tag d20-baseline-8-H100 \
  --model-step 932
```

To replay a different SFT checkpoint, substitute the five values above. By default, the evaluator requests eight strict H100s and runs the same 24-question subset used in SFT. Pass `--gpus` with 1, 2, 4, or 8 and optionally `--gpu-type` to change the hardware. Prefix the command with `NANOCHAT_GSM8K_FULL_TEST=1` to evaluate all 1,319 test questions.

The evaluator reuses the selected SFT run's cached GSM8K test data when present. Otherwise, it downloads the test data into temporary container storage. Both modes use greedy decoding, one completion per problem, a 512-token generation cap, and Nanochat's stock scorer.

Keep the terminal attached to receive the timestamped `gsm8k-sft-step<step>-*.jsonl` file in your current directory. Each row contains the question, raw completion, reference and parsed answers, and whether it received credit.
