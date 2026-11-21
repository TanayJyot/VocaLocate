# A4 Part 2: SFT and chat evaluation

This folder records the comparison between the pretrained d20-baseline-long
model, Standard SFT, and three SFT runs with added math or coding examples.
Run launchers from the VocaLocate repository root.

## What the files do

- `modal_chat_sft.py` runs Standard SFT. Its `--eval-pretrained` mode runs
  ChatCORE on the pretrained model.
- `part2b/metamath/` and `part2b/kodcode/` contain the selection code and
  manifests for MetaMathQA and KodCode. `part2b/shared/` supports overlap checks.
- `part2b/integration/` contains a JSONL loader and a patch for a separate
  Nanochat checkout. The patch adds examples to SFT training and records how
  many examples from each source were used. It leaves the reported BPB
  validation mixture and ChatCORE scorer unchanged.
- `part2b/modal_enhanced_sft_runs.py` runs MetaMathQA-only, KodCode-only, or
  combined SFT. Without `--launch`, it checks inputs but does not train.
- `modal_gsm8k_replay.py` evaluates a saved SFT checkpoint with Nanochat's
  GSM8K scorer and saves individual questions, completions, and scores.
  `NANOCHAT_GSM8K_FULL_TEST=1` selects all 1,319 test questions.
- `a4/modal/` holds the selected run summaries, training/evaluation logs,
  and full GSM8K response files. Trained checkpoints remain on Modal.

## Data and training

Standard SFT uses Nanochat's 789,759-row training mixture: SmolTalk,
MMLU auxiliary training data repeated three times, and GSM8K training data
repeated four times. Enhanced SFT adds 60,000 selected MetaMathQA examples,
40,000 selected KodCode examples, or both. Separate 3,000-row MetaMathQA
and 1,000-row KodCode holdouts support dataset auditing; they are not used
in the reported BPB validation score.

The enhanced runs assume the pretrained model, metadata, and eight matching
optimizer shards are at:

`a4-nanochat-volume:/a4/base_checkpoints/d20-baseline-long/`

The historical Standard SFT run and pretrained ChatCORE evaluation instead
read `vocalocate-a3:/a3/base_checkpoints/d20-baseline-long/`. All reported
runs identify pretrained step 21,328 and use eight H100s, 524,288
batch-token positions per update, a 2,048-token sequence limit, and 932
SFT optimizer updates. Added data changes the examples seen within that
fixed update budget. The archived summaries do not include cross-volume
checkpoint hashes.

## Running and reproducing results

Run `modal_chat_sft.py` for Standard SFT, or add `--eval-pretrained` to
evaluate the pretrained checkpoint. Run
`part2b/modal_enhanced_sft_runs.py` with `--run-kind metamath`,
`--run-kind kodcode`, or `--run-kind both` for enhanced SFT. The reported
enhanced runs used `--gpus-per-job 8`, `--total-batch-size 524288`,
`--optimizer-policy restore`, `--kodcode-subdir kodcode_40k_v2`, and
`--launch`. The archived `run_summary.json` files give each historical
command and output tag; older commands include an approval override that
the current launcher no longer needs.

The screened JSONL, raw source files, reference index, original task cache,
and isolated patched Nanochat checkout are kept outside Git. The committed
manifests record selected row counts and hashes, but the repository alone
does not recreate every historical input. Automated screening also does
not independently establish that every solution is correct.