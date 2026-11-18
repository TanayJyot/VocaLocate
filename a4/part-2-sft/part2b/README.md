# Part 2b: MetaMathQA and KodCode SFT

This folder retains the screening code, manifests and isolated Nanochat integration for the 60k MetaMathQA and final 40k KodCode experiments. Raw datasets, generated JSONL, reference index, historical audit files, and Nanochat checkout are not included. Both datasets are **UNAPPROVED** pending quality and similarity review; retrieval is approximate and KodCode tests were not independently executed. Keep large data outside Git and check licenses before sharing.

- `metamath/prepare.py` screens MetaMathQA only. Its historical selection seed remains in the script because changing it would select different rows. The checked-in `metamath/screened_v1/manifest.json` is a MetaMathQA-only view of the original combined audit manifest (SHA-256 `b9954ebbca5119d377a1848c1ba9a039a79a634228e2c01b81d5976272d5d060`); it retains the MetaMath train/validation row counts and hashes while omitting unrelated source metadata. The original manifest remains in the experiment archive.
- `kodcode/finalize_40k.py` contains the final selection repair and its screening helpers. The final 40k split retains an earlier 20k/1k seed split, so exact reconstruction still requires those archived split JSONLs, the **pre-repair** 40k draft, similarity-review hold list, exact reference index, tokenizer and all 15 original shards. This package has no separate 20k generator or manifest. It verifies draft and tokenizer hashes before writing. Compare final hashes to `kodcode/screened_40k_v2/kodcode_manifest.json`; no complete reproduction has been verified from this folder.
- `shared/fetch_reference.py` fetches reference exports. The exact reference index used in screening is an archived experiment input; the exports alone cannot recreate it.
- `integration/` contains the patch and JSONL loader. Create a separate Nanochat checkout based on commit `e6ca3a27c145bb66ba65b593cd72b52e688e28f5`, check `git apply --check` against it, apply the patch, and copy `screened_jsonl.py` into its `tasks/` directory. Do not modify the existing assignment checkout. Set `NANOCHAT_REPO` to this checkout for Modal; preparation expects it at `derived_nanochat`. The patch was generated against the target checkout and must be checked against the pinned revision.

Preparation requires Python, numpy, pyarrow, scikit-learn, huggingface_hub, the pinned Nanochat dependencies and its tokenizer. With the archived inputs restored to the paths documented by `--help`, run from this folder:

```sh
python metamath/prepare.py --tokenizer-dir /path/to/tokenizer
python kodcode/download_remaining.py
python kodcode/finalize_40k.py --tokenizer-dir /path/to/tokenizer
```

The downloader pins KodCode revision `97d81f0695ddd7c5c02bfe16436db9a17a4a21e5` and verifies shard hashes under `/private/tmp`. The finalizer is an incremental repair of the archived full audit, not a standalone audit. Do not run it against the final JSONL in place of the draft. Manifest paths are historical provenance and are not portable.

`modal_kodcode_experiments.py` preflights or launches fixed 932-step runs. It requires both datasets even for a one-source run, plus the checkpoint, tokenizer, Modal volume and isolated checkout. For options run `modal run modal_kodcode_experiments.py --help`; without `--launch` it only preflights. Launching uses paid GPU resources and unapproved data requires the explicit override.
