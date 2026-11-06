# Looped depth

Looped depth executes a core of Transformer blocks repeatedly with shared
weights. `--depth` still counts unique blocks. The optional `--n-prelude` blocks
run once before the core, and `--n-coda` blocks run once after it.

Append these flags to an existing training command to enable looping:

```sh
--n-loops=3 --n-prelude=1 --n-coda=1
```

The defaults are `n_loops=1`, `n_prelude=0` and `n_coda=0`. Direct Python callers
can set the same fields on `GPTConfig`.

The effective execution depth is:

```text
n_core = n_layer - n_prelude - n_coda
effective_depth = n_prelude + n_core * n_loops + n_coda
```

At least one core block is required. For four unique blocks, one prelude,
one coda and three loops, the executed block indices are
`[0, 1, 2, 1, 2, 1, 2, 3]`: eight execution steps using four unique blocks.
The KV cache uses a separate slot for each execution step.

Run the tests from the repository root:

```sh
OMP_NUM_THREADS=1 python -m pytest -q tests/ablations/test_looped.py
```

The tests cover the layer plan, baseline regression, shared block weights,
compute accounting, gradients and cached decoding. Compiled training and resume

```sh
OMP_NUM_THREADS=1 python -m pytest -q tests/ablations/test_diff_v2_smoke.py
```

See the [ablation index](../README.md) for the other architecture experiments.
