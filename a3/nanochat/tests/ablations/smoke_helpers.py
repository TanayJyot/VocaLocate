"""Shared helpers for the offline CPU smoke tests."""
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import tiktoken
import torch

from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS, SPLIT_PATTERN


# Compile the full model and optimizer graphs using the portable CPU backend.
# This changes only the test subprocess; production training keeps its backend.
LOCAL_TRAINER = """
import runpy
import torch

original_compile = torch.compile
def local_compile(model=None, **kwargs):
    kwargs.update(backend="aot_eager", fullgraph=True)
    return original_compile(model, **kwargs)

torch.compile = local_compile
runpy.run_module("scripts.base_train", run_name="__main__")
"""


@pytest.fixture
def local_data(tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    encoding = tiktoken.Encoding(
        name="smoke-test-bytes", pat_str=SPLIT_PATTERN,
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={token: 256 + i for i, token in enumerate(SPECIAL_TOKENS)},
    )
    RustBPETokenizer(encoding, "<|bos|>").save(tokenizer_dir)
    torch.save(torch.tensor([1] * 256 + [0] * len(SPECIAL_TOKENS), dtype=torch.int32),
               tokenizer_dir / "token_bytes.pt")
    data_dir = tmp_path / "base_data_climbmix"
    data_dir.mkdir()
    table = pa.table({"text": ["A tiny offline example for activation testing."] * 1024})
    # The real loader uses the last shard for validation and the rest for training.
    for index in range(2):
        pq.write_table(table, data_dir / f"shard_{index:05d}.parquet", row_group_size=128)
    return tmp_path
