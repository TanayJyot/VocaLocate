# Local xIPReLU smoke test

From the repository root, using a Python environment with the project dependencies
and pytest installed:

```sh
python -m pytest -q tests/test_xiprelu_smoke.py
```

Alternatively, let uv prepare the CPU dependencies:

```sh
uv run --extra cpu --with pytest python -m pytest -q tests/test_xiprelu_smoke.py
```

The test uses the real `scripts.base_train` entry point with a two-layer,
32-dimensional model, 16-token sequences and 32 tokens per training step. It
generates a byte tokenizer and tiny train/validation shards in pytest's temporary
directory. It trains for three steps and resumes for one more, for both the
default ReLU² baseline, `--mlp-activation=xiprelu`, and xIPReLU combined with
`--n-loops=2`. Existing looped-model configuration and checkpoint fields are
preserved. The two shared blocks still have only four xIPReLU scalars total.

It checks full-graph compilation of the model and optimizer with `aot_eager`,
forward/backward execution, finite validation results and weights, checkpoint
metadata, optimizer resume and learning of both xIPReLU coefficients in each
layer. Production training code and compilation settings are unchanged.

The test itself needs no downloads, credentials, real training data or Modal
calls. uv may download dependencies when setting up the environment. This is
a correctness check, not a comparison of ablation quality. CUDA/Flash Attention,
FP8, GPU memory and Inductor/Triton kernel compilation still need a short check
on the intended GPU before a long run.

The command prints a readable report even without pytest's `-s` flag:

- A table of known inputs, expected xIPReLU outputs, actual outputs and gradients.
  This example deliberately uses different positive/negative coefficients to
  verify that the two branches are implemented correctly.
- A pass message after compilation/training, checkpoint saving and resume.
- Actual parameter counts and validation BPB before and after training.
- Each xIPReLU coefficient at initialization, after three steps and after resume,
  with its change displayed explicitly. All four must learn and keep updating.

Expected success is `4 passed`. A failed numerical or execution check still fails
pytest with a nonzero exit code. BPB is lower-is-better, but the tiny repeated
synthetic dataset and four updates are not evidence that xIPReLU outperforms
ReLU². That requires matched, adequately trained runs on meaningful data.

For the numerical, checkpoint, inference and baseline regression tests as well:

```sh
python -m pytest -q tests/test_xiprelu.py tests/test_xiprelu_smoke.py
```
