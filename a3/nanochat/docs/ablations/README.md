# Architecture ablations

Each ablation has its own README in this folder. Run the commands below from
the repository root with the project dependencies and pytest installed.

| Ablation | Documentation | Tests in `tests/ablations/` |
| --- | --- | --- |
| Looped depth | [README](looped/README.md) | `test_looped.py` |
| xIPReLU | [README](xiprelu/README.md) | `test_xiprelu.py`, `test_xiprelu_smoke.py` |
| Differential Attention V2 | [README](diff_v2/README.md) | `test_diff_v2.py`, `test_diff_v2_smoke.py` |
| Depth-Attention | [README](depth_attention/README.md) | `test_depth_attention.py`, `test_depth_attention_smoke.py` |
| MGVR | [README](mgvr/README.md) | `test_mgvr.py`, `test_mgvr_smoke.py` |

For controlled Part 2 commands, use the centralized selector in the existing
trainer. It enables the selected architecture and rejects flags that would add
a different ablation:

```sh
python -m scripts.base_train --experiment=baseline ...
python -m scripts.base_train --experiment=mgvr ...
python -m scripts.base_train --experiment=looped --n-loops=2 ...
```

The choices are `baseline`, `looped`, `xiprelu`, `diff-v2`, `depth-attention`,
`mgvr`, and `combined`. For example, `--experiment=mgvr --depth-attention`
raises a parser error before compute/data initialization. Intentional mixtures
must use `--experiment=combined` with at least two enabled ablation flags:

```sh
python -m scripts.base_train --experiment=combined --mgvr --depth-attention ...
```

Replace `...` with your existing training arguments. Calls that omit
`--experiment` retain their existing flag behavior. There is no separate
NanoChat Modal/Part 2 launcher in this checkout; use this selector wherever the
Part 2 command invokes `scripts.base_train`. Selection does not pin training
budgets: keep the intended batch size and iteration/token budget explicit.

Run all ablation tests:

```sh
OMP_NUM_THREADS=1 python -m pytest -q tests/ablations/
```

Run a single ablation, including its smoke tests:

```sh
OMP_NUM_THREADS=1 python -m pytest -q tests/ablations/test_mgvr.py tests/ablations/test_mgvr_smoke.py
```

The smoke tests generate tiny offline datasets and exercise compiled CPU
training, checkpoint saving and resume. CUDA-specific cases skip when the
required hardware is unavailable. The normal `python -m pytest tests/` command
also discovers the ablation tests.

Additional cross-ablation checks live in `test_ablation_interactions.py` and
`test_experiments.py`. `test_baseline_equivalence.py` explicitly disables every
architecture ablation and compares weights, logits, gradients, two optimizer
updates and cached decoding against the original imported NanoChat at commit
`0ed62a1`, including its original optimizer implementation. These historical
checks require that commit to be available in local Git history.
