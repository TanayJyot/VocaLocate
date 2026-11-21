"""Run nanochat's stock scripts/chat_rl.py with per-rollout logging.

The training math is untouched. This wrapper only:
  * wraps tasks.gsm8k.GSM8K.reward / .evaluate to append one jsonl record per
    sampled completion (training rollouts and in-training pass@k eval samples);
  * redirects the hardcoded W&B project ("nanochat-rl") to $WANDB_PROJECT;
  * optionally truncates the GSM8K train split to RL_SMOKE_STEPS * examples_per_step
    questions for a short smoke test (num_steps is derived from the split length);
  * optionally replaces the TRAINING reward with a sum of components from rewards.py
    (RL_REWARD, default "strict" = the stock reward). Evaluation stays strict.

Usage (inside the container, cwd = nanochat repo):
    torchrun --standalone --nproc_per_node=8 /root/rl_rollout_logger.py -- --run=NAME ...
Environment:
    RL_ROLLOUT_DIR   directory for rank{N}.jsonl files (required)
    RL_SMOKE_STEPS   optional int; limit training to this many steps
    WANDB_PROJECT    optional W&B project override
    RL_REWARD        optional reward spec for training, e.g. "strict,my_format:0.2" (see rewards.py)
    RL_SCRIPT        optional training script under scripts/ (default "chat_rl"; Part 5 uses "chat_distill")
"""

import json
import os
import runpy
import sys
import time

sys.path.insert(0, os.getcwd())

import wandb  # noqa: E402
import tasks.gsm8k as gsm8k  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # rewards.py sits next to this file
import rewards  # noqa: E402

REWARD_SPEC = rewards.parse_spec(os.environ.get("RL_REWARD", "strict"))  # fail fast on a typo
IS_STOCK_REWARD = REWARD_SPEC == [("strict", 1.0)]

RANK = int(os.environ.get("RANK", "0"))
ROLLOUT_DIR = os.environ["RL_ROLLOUT_DIR"]
os.makedirs(ROLLOUT_DIR, exist_ok=True)
_out = open(os.path.join(ROLLOUT_DIR, f"rank{RANK}.jsonl"), "a", buffering=1)
_t0 = time.time()
SMOKE_STEPS = int(os.environ.get("RL_SMOKE_STEPS", "0") or 0)

# --- W&B project redirect ----------------------------------------------------
_wandb_init = wandb.init
def _patched_init(*args, **kwargs):
    if os.environ.get("WANDB_PROJECT"):
        kwargs["project"] = os.environ["WANDB_PROJECT"]
    return _wandb_init(*args, **kwargs)
wandb.init = _patched_init

# --- GSM8K instrumentation ---------------------------------------------------
_orig_init = gsm8k.GSM8K.__init__
_orig_evaluate = gsm8k.GSM8K.evaluate
_orig_reward = gsm8k.GSM8K.reward


def _patched_init(self, subset, split, **kwargs):
    if split == "train" and SMOKE_STEPS > 0:
        # examples_per_step default is 16; read it from argv if overridden.
        eps = 16
        for i, a in enumerate(sys.argv):
            if a.startswith("--examples-per-step"):
                eps = int(a.split("=", 1)[1]) if "=" in a else int(sys.argv[i + 1])
        kwargs["stop"] = SMOKE_STEPS * eps
    _orig_init(self, subset, split, **kwargs)
    self._split = split
    # Map question text -> index in the (shuffled, seed 42) split, so records are joinable.
    col = self.ds.table["question"].to_pylist()
    perm = self.ds.permutation
    self._qidx = {col[int(p) if perm is not None else i]: i for i, p in enumerate(perm if perm is not None else range(len(col)))}


def _current_step():
    main = sys.modules.get("__main__")
    return getattr(main, "step", None) if main is not None else None


def _record(self, kind, conversation, response, correct, reward=None, parts=None):
    question = conversation["messages"][0]["content"]
    ref = gsm8k.extract_answer(conversation["messages"][-1]["content"][-1]["text"])
    _out.write(json.dumps({
        "kind": kind,                      # "train" rollout or "eval" pass@k sample
        "step": _current_step(),
        "rank": RANK,
        "t": round(time.time() - _t0, 2),  # seconds since this rank started
        "qidx": self._qidx.get(question),
        "ref": ref,
        "pred": gsm8k.extract_answer(response),
        "correct": int(correct),           # always the strict 0/1 answer check
        **({"reward": reward, "reward_parts": parts} if reward is not None else {}),  # train only
        "completion": response,
    }) + "\n")


def _patched_evaluate(self, conversation, assistant_response):
    correct = _orig_evaluate(self, conversation, assistant_response)
    if getattr(self, "_split", None) == "test":
        _record(self, "eval", conversation, assistant_response, correct)
    return correct


def _patched_reward(self, conversation, assistant_response):
    correct = _orig_evaluate(self, conversation, assistant_response)  # strict check, as stock reward() does
    if IS_STOCK_REWARD:
        r, parts = float(correct), {"strict": float(correct)}
    else:
        ctx = {"correct": correct, "pred": gsm8k.extract_answer(assistant_response),
               "ref": gsm8k.extract_answer(conversation["messages"][-1]["content"][-1]["text"]),
               "ref_solution_parts": conversation["messages"][-1]["content"]}
        r, parts = rewards.compute(REWARD_SPEC, conversation, assistant_response, ctx)
    _record(self, "train", conversation, assistant_response, correct, float(r), parts)
    return float(r)


gsm8k.GSM8K.__init__ = _patched_init
gsm8k.GSM8K.evaluate = _patched_evaluate
gsm8k.GSM8K.reward = _patched_reward

# --- run the stock script ----------------------------------------------------
SCRIPT = os.environ.get("RL_SCRIPT", "chat_rl")
if sys.argv[1:2] == ["--"]:
    del sys.argv[1]
sys.argv[0] = f"scripts/{SCRIPT}.py"
RESUME = int(os.environ.get("RL_RESUME_STEP", "0") or 0)


def resume_source(src, start):
    """Edit chat_rl.py's source (in memory only) to continue a preempted run at step `start`:
    load the RL checkpoint saved at the end of step start-1, run steps start..num_steps-1, and skip the
    start*examples_per_rank training examples each rank already consumed. The LR multiplier is computed
    from `step` inside the loop, so the schedule continues unchanged. The optimizer is fresh: chat_rl.py
    never saves optimizer state, so momentum / Adam moments restart at `start`."""
    edits = [
        ('model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step)',
         f'model, tokenizer, meta = load_model("rl", device, phase="eval", model_tag=args.model_tag, step={start - 1})'),
        ("for example_idx in itertools.cycle(rank_indices):",
         f"for example_idx in itertools.islice(itertools.cycle(rank_indices), {start} * examples_per_rank, None):"),
        ("for step in range(num_steps):", f"for step in range({start}, num_steps):"),
    ]
    for old, new in edits:
        if src.count(old) != 1:
            raise RuntimeError(f"{SCRIPT}.py changed; cannot apply resume edit: {old!r}")
        src = src.replace(old, new)
    return src


try:
    if RESUME:
        import types
        path = os.path.join(os.getcwd(), "scripts", f"{SCRIPT}.py")
        src = resume_source(open(path).read(), RESUME)
        main_mod = types.ModuleType("__main__")
        main_mod.__file__ = path
        sys.modules["__main__"] = main_mod
        print(f"[rl_rollout_logger] resuming at step {RESUME} from chatrl checkpoint step {RESUME - 1}", flush=True)
        exec(compile(src, path, "exec"), main_mod.__dict__)
    else:
        runpy.run_module(f"scripts.{SCRIPT}", run_name="__main__", alter_sys=True)
finally:
    _out.close()
