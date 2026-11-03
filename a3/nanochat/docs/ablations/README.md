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
