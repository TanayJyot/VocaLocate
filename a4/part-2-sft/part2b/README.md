# Part 2B: Math and coding SFT

This folder documents the screened MetaMathQA and KodCode data used to extend the original Nanochat SFT mixture. The experiments add **60,000 math examples**, **40,000 coding examples**, or both. Separate source holdouts contain 3,000 MetaMathQA and 1,000 KodCode examples; they support dataset auditing and are not part of the reported BPB validation mixture.

## Screening

- `metamath/prepare.py` selects nonempty, deduplicated examples, checks overlap with the original SFT data and similarity to GSM8K test and HumanEval prompts, and holds test-similar examples. MetaMath's GSM8K training lineage is intentional. It also checks tokenizer length and assistant loss tokens, and keeps related examples together when splitting train and holdout.
- `kodcode/finalize_40k.py` repairs the final selection from a completed audit of 15 source shards. Screening checks for function-writing requests, parseable single-function solutions, tests, duplicate or overlapping prompts, benchmark similarity, tokenizer length, and train/holdout group separation. The repair excludes selected named-function prompts that already contain a solution body and fills the resulting slots with screened candidates.
- `shared/fetch_reference.py` fetches reference exports used by the checks. The manifests record counts, input hashes and selected output hashes.

These are automated screens, not guarantees of correctness or complete decontamination. Original-SFT near-duplicate retrieval is approximate, similarity flags and solution quality require review, and KodCode tests were not independently executed.

## Training

The integration patch adds the selected training JSONL to the original 789,759-row SFT mixture. Source holdouts are not added to training or the reported BPB validation mixture. All reported runs use pretrained step 21,328, eight H100s, a 2,048-token sequence limit, 524,288 batch-token positions per update, and **932 optimizer updates**. Added data changes the examples seen within that fixed budget.

`modal_enhanced_sft_runs.py` preflights the checkpoint, tokenizer, original task cache, both screened datasets and output paths on the Modal volume. It defaults to preflight only; `--launch` starts training and records the command, dataset hashes, logs and run summary. From the repository root, select `--run-kind metamath`, `--run-kind kodcode`, or `--run-kind both`. The reported runs used `--gpus-per-job 8 --total-batch-size 524288 --optimizer-policy restore --kodcode-subdir kodcode_40k_v2 --launch`. Run `modal run a4/part-2-sft/part2b/modal_enhanced_sft_runs.py --help` for options. Launching uses paid GPU resources.

## Reproduction

Raw data, screened JSONL, the reference index, audit files, task cache and patched Nanochat checkout are not included in Git. Create a separate Nanochat checkout at commit `e6ca3a27c145bb66ba65b593cd72b52e688e28f5`; check and apply `integration/chat_sft.patch`, then copy `integration/screened_jsonl.py` to its `tasks/` directory. Preparation expects the checkout at `derived_nanochat`; set `NANOCHAT_REPO` to it for Modal. Install the preparation dependencies and pinned Nanochat dependencies, and restore the tokenizer and archived inputs.

From this folder, the preparation entry points are:

```sh
python metamath/prepare.py --tokenizer-dir /path/to/tokenizer
python kodcode/download_remaining.py
python kodcode/finalize_40k.py --tokenizer-dir /path/to/tokenizer
```

See each script's `--help` and manifest for inputs. The KodCode downloader pins the source revision and verifies shard hashes. The finalizer is an incremental repair, not a complete audit: it requires the archived initial selection, pre-repair draft, similarity-review hold list, exact reference index, tokenizer and all 15 original shards. Do not substitute the final JSONL for the draft. Manifest paths are historical and may need restoring locally. Compare regenerated outputs with the manifest hashes; complete reproduction from this folder alone has not been verified. Keep large data outside Git and check licenses before sharing.
