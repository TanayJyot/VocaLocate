# A3 tokenizer

The tokenizer used by every d12 run in `a3/modal/` (byte-identical to `/a3/tokenizer` on the `vocalocate-a3` Modal volume).

| Setting | Value |
|---|---|
| Vocab size | 32,768 |
| Max chars | 1,000,000,000 |
| Doc cap | 10,000 chars |
| Training data | ClimbMix train shards, read in order; the 1B limit was reached in `shard_00004` |

Trained with:

```bash
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=1000000000
```

SHA-256:

```
46a1a55e1154ec68b217dde7deffbfeb1f754e911982d9d7f53a7523b356ab89  tokenizer.pkl
42317d1460274d9f175fe3532d0cc6a04d06396ed06d3338ce3f733c9f986665  token_bytes.pt
```

To use it, copy both files to `$NANOCHAT_BASE_DIR/tokenizer/`, or upload them to Modal:

```bash
modal volume put vocalocate-a3 a3/tokenizer/tokenizer.pkl /a3/tokenizer/tokenizer.pkl
modal volume put vocalocate-a3 a3/tokenizer/token_bytes.pt /a3/tokenizer/token_bytes.pt
```

`tokenizer.pkl` is a Python pickle, and loading a pickle can run code. Check the hash before loading a copy from anywhere else.
