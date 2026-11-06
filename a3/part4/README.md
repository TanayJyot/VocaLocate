# Part 4 context extension results

The baseline Picochat trains at a sequence length of 512 tokens to step 1260
(checkpoint A), then resumes at 2048 tokens to step 2520 (checkpoint B).
Both stages use one continuous learning-rate schedule, seed 42, and 524,288
tokens per optimizer step. Training uses ClimbMix training shards 0-31 and the
separate validation shard 6542. The tokenizer is the unchanged copy in
`../tokenizer`, inherited from commit `be8e30c`.

## Training

From the repository root, with Modal configured:

```sh
modal run a3/modal/train_part4.py::prep
modal run a3/modal/train_part4.py::smoke
modal run --detach a3/modal/train_part4.py::train
```

The launcher verifies the tokenizer hashes and dataset manifest before training.
Checkpoints are stored on the `vocalocate-a3-part4` Modal volume under
`/part4/base_checkpoints/d12-context-part4/`. Model and optimizer weights are
excluded from Git. The `checkpoints/` directory contains the saved configuration,
validation metric, and data-loader position for A and B.

## Local evaluation

Use the dependencies declared in `a3/nanochat/pyproject.toml`. Download
`model_001260.pt`, `meta_001260.json`, `model_002520.pt`, and `meta_002520.json`
from the volume into `a3/nanochat/dev-ignore/part4/`. From `a3/nanochat`, run:

```sh
python -m scripts.part4_eval --checkpoint-dir dev-ignore/part4 --steps 1260 2520 --trials 128 --seed 42 --output ../part4/results/passkey-results.json
python -m scripts.part4_eval --checkpoint-dir dev-ignore/part4 --steps 1260 2520 --trials 128 --seed 20261104 --competing-records --output ../part4/results/passkey-competing-paired.json
```

Both checkpoints receive identical prompts and candidate words within each task.
The code compares the next-token scores of four candidate words. A correct
answer scores one; an incorrect answer scores zero. Each condition has 128
prompts. Input length and the distance from the target code to the final input
token are controlled separately. Both checkpoints are evaluated with an input
limit of 2048 tokens.

## Saved outputs

- `results/passkey-results.json`: single-record accuracy for A and B.
- `results/passkey-competing-records.json`: original four-record accuracy summary.
- `results/passkey-competing-paired.json`: matching four-record totals with each
  trial's seed, candidate token IDs, expected answer, and prediction.
- `results/passkey-paired-summary.json`: paired differences in four-record
  accuracy and approximate 95% confidence intervals, in percentage points.
- `results/passkey-leakage-audit.json`: checks on B after replacing the stated
  code with `unknown` or a different candidate. `no_answer_correct` counts
  selection of the removed original code, not selection of `unknown`.
- `logs/`: the original training logs for the two stages.
- `figures/`: the training/validation curves and four-record accuracy figure
  used in the report.

For each condition, the confidence interval uses paired correctness differences
`d_i = correct_B_i - correct_A_i`, with values -1, 0, or 1. The estimated gain is
`100 * mean(d)`, and the interval is `100 * (mean(d) +/- 1.96 * stdev(d) / sqrt(n))`,
using the sample standard deviation and `n = 128`.

## Tests

From `a3/nanochat`:

```sh
python -m pytest tests/test_part4_eval.py tests/ablations/test_context_resume.py -q
```
