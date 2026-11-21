"""CPU smoke test for chat_distill.py: tiny random models, a few steps, every teacher mode.

Checks, through the same rollout-logger entry point the Modal launcher uses:
  1. a checkpoint teacher identical to the student gives a reverse KL of 0 at step 0
     (so the advantage, and the distillation gradient, vanish when teacher == student);
  2. a different checkpoint teacher gives a positive, finite reverse KL;
  3. the SDPO self-teacher (reference solution in context, frozen and EMA) runs end to end;
  4. every run writes rollouts and a final checkpoint.

Needs torch and nanochat's CPU dependencies, the team tokenizer in a3/tokenizer, and internet
access the first time (GSM8K parquet from the Hugging Face hub, cached afterwards).

    python a4/part-5-distill/local_smoke_test.py [--work-dir DIR]
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
NANOCHAT = REPO_ROOT / "a3" / "nanochat"
LOGGER = REPO_ROOT / "a4" / "part-3-rl" / "rl_rollout_logger.py"
TOKENIZER = REPO_ROOT / "a3" / "tokenizer"
STEPS = 3
TINY = dict(sequence_len=512, vocab_size=32768, n_layer=2, n_head=2, n_kv_head=2, n_embd=64, window_pattern="L")


def save_tiny(checkpoint_dir, seed):
    sys.path.insert(0, str(NANOCHAT))
    import torch
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.checkpoint_manager import save_checkpoint
    torch.manual_seed(seed)
    model = GPT(GPTConfig(**TINY))
    model.init_weights()
    # init_weights zero-initializes the output projections; perturb them so the model is not uniform
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02 * torch.randn_like(p))
    save_checkpoint(str(checkpoint_dir), 0, model.state_dict(), None, {"model_config": TINY})


def run(work, name, distill_args):
    base = work / name
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    for sub in ("tokenizer", "chatsft_checkpoints", "task_data"):
        (base / sub).symlink_to(work / "shared" / sub, target_is_directory=True)
    env = dict(os.environ, NANOCHAT_BASE_DIR=str(base), RL_ROLLOUT_DIR=str(base / "rollouts"),
               RL_SMOKE_STEPS=str(STEPS), RL_SCRIPT="chat_distill", PYTHONUNBUFFERED="1")
    cmd = [sys.executable, str(LOGGER), "--", "--device-type=cpu", "--model-tag=tiny", "--model-step=0",
           "--num-samples=4", "--device-batch-size=4", "--examples-per-step=2", "--max-new-tokens=24",
           "--eval-every=100", "--eval-examples=4", "--save-every=100", *distill_args]
    proc = subprocess.run(cmd, cwd=work / "nanochat", env=env, capture_output=True, text=True)
    if proc.returncode:
        print(proc.stdout[-4000:], proc.stderr[-4000:], sep="\n")
        raise SystemExit(f"{name}: exit code {proc.returncode}")
    kls = [float(m) for m in re.findall(r"Reverse KL: (-?[\d.e+-]+|nan|inf)", proc.stdout)]
    assert len(kls) == STEPS, f"{name}: expected {STEPS} steps, got {len(kls)}"
    assert all(math.isfinite(k) for k in kls), f"{name}: non-finite KL {kls}"
    assert (base / "rollouts" / "rank0.jsonl").stat().st_size > 0, f"{name}: no rollouts logged"
    assert (base / "chatrl_checkpoints" / "tiny" / f"model_{STEPS - 1:06d}.pt").is_file(), f"{name}: no checkpoint"
    print(f"{name:14s} reverse KL per step: {', '.join(f'{k:.5f}' for k in kls)}")
    return kls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default=None, help="reused between runs to keep the GSM8K download")
    args = parser.parse_args()
    work = Path(args.work_dir or Path(tempfile.gettempdir()) / "chat_distill_smoke").resolve()
    shared = work / "shared"
    for sub in ("tokenizer", "chatsft_checkpoints", "task_data"):
        (shared / sub).mkdir(parents=True, exist_ok=True)
    for f in ("tokenizer.pkl", "token_bytes.pt"):
        shutil.copy2(TOKENIZER / f, shared / "tokenizer" / f)
    # A copy of nanochat with chat_distill.py in scripts/, as the Modal image has it
    if (work / "nanochat").exists():
        shutil.rmtree(work / "nanochat")
    shutil.copytree(NANOCHAT, work / "nanochat", ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__"))
    shutil.copy2(HERE / "chat_distill.py", work / "nanochat" / "scripts" / "chat_distill.py")

    save_tiny(shared / "chatsft_checkpoints" / "tiny", seed=0)
    save_tiny(work / "teachers" / "same", seed=0)
    save_tiny(work / "teachers" / "diff", seed=1)

    same = run(work, "teacher_same", [f"--teacher-dir={work / 'teachers' / 'same'}", "--teacher-step=0"])
    assert abs(same[0]) < 1e-5, f"identical teacher should give KL 0 at step 0, got {same[0]}"
    diff = run(work, "teacher_diff", [f"--teacher-dir={work / 'teachers' / 'diff'}", "--teacher-step=0"])
    assert diff[0] > 0.01, f"different teacher should give a positive KL, got {diff[0]}"
    run(work, "teacher_mixed", [f"--teacher-dir={work / 'teachers' / 'diff'}", "--teacher-step=0",
                                "--reward-coef=1.0", "--distill-coef=1.0", "--adv-clip=5"])
    run(work, "self_frozen", ["--teacher=self"])
    run(work, "self_ema", ["--teacher=self", "--teacher-ema=0.05"])
    print("all checks passed")


if __name__ == "__main__":
    main()
