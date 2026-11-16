"""Run nanochat's stock scripts/chat_eval.py, logging every GSM8K completion to jsonl.

Scoring is untouched; tasks.gsm8k.GSM8K.evaluate is wrapped to append
{qidx, ref, pred, correct, completion} per completion to $EVAL_LOG_DIR/<EVAL_LOG_NAME>_rank{N}.jsonl.

Usage (cwd = nanochat repo):
    torchrun --standalone --nproc_per_node=8 /root/eval_logger.py -- -i rl -g <tag> -s <step>
"""
import json
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())
import tasks.gsm8k as gsm8k  # noqa: E402

RANK = int(os.environ.get("RANK", "0"))
os.makedirs(os.environ["EVAL_LOG_DIR"], exist_ok=True)
_out = open(os.path.join(os.environ["EVAL_LOG_DIR"], f"{os.environ['EVAL_LOG_NAME']}_rank{RANK}.jsonl"), "a",
            buffering=1)
_orig_init, _orig_eval = gsm8k.GSM8K.__init__, gsm8k.GSM8K.evaluate


def _init(self, subset, split, **kwargs):
    _orig_init(self, subset, split, **kwargs)
    col, perm = self.ds.table["question"].to_pylist(), self.ds.permutation
    self._qidx = {col[int(p)]: i for i, p in enumerate(perm)}


def _evaluate(self, conversation, response):
    correct = _orig_eval(self, conversation, response)
    q = conversation["messages"][0]["content"]
    _out.write(json.dumps({
        "qidx": self._qidx.get(q),
        "ref": gsm8k.extract_answer(conversation["messages"][-1]["content"][-1]["text"]),
        "pred": gsm8k.extract_answer(response), "correct": int(correct), "completion": response,
    }) + "\n")
    return correct


gsm8k.GSM8K.__init__, gsm8k.GSM8K.evaluate = _init, _evaluate
if sys.argv[1:2] == ["--"]:
    del sys.argv[1]
sys.argv[0] = "scripts/chat_eval.py"
try:
    runpy.run_module("scripts.chat_eval", run_name="__main__", alter_sys=True)
finally:
    _out.close()
