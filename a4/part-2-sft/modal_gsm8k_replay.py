"""Evaluate a saved SFT checkpoint on GSM8K and save its completions."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime, timezone

import modal


repo = Path("/root/nanochat")
if Path(__file__).resolve().parent != Path("/root"):
    default_repo = Path(__file__).resolve().parents[2] / "a3" / "nanochat"
    repo = Path(os.environ.get("NANOCHAT_REPO", str(default_repo))).expanduser()
    if not (repo / "scripts" / "chat_eval.py").is_file():
        raise FileNotFoundError(f"Set NANOCHAT_REPO to the nanochat checkout: {repo}")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "numpy", "tiktoken", "filelock", "psutil", "pyarrow", "requests",
        "jinja2", "datasets", "kernels==0.11.7", "huggingface-hub==0.34.4",
        "rustbpe", "wandb",
    )
    .add_local_dir(
        repo, "/root/nanochat",
        ignore=[".venv", ".git", "**/__pycache__", "**/.pytest_cache"],
    )
)

full_test = os.environ.get("NANOCHAT_GSM8K_FULL_TEST") == "1"
app = modal.App(
    "nanochat-a4-sft-gsm8k-full-test" if full_test else "nanochat-a4-sft-gsm8k-replay",
    image=image,
)

# Run the stock evaluator and capture its inputs and decisions per rank.
CAPTURE_EVAL = r"""
import json
import os
from pathlib import Path
import runpy

from tasks.gsm8k import GSM8K, extract_answer

original_evaluate = GSM8K.evaluate
rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
next_index = rank
output = Path(os.environ["GSM8K_CAPTURE_DIR"]) / f"rank-{rank}.jsonl"

def capture(self, conversation, completion):
    global next_index
    correct = original_evaluate(self, conversation, completion)
    reference = conversation["messages"][-1]["content"][-1]["text"]
    record = {
        "shuffled_test_index": next_index,
        "question": conversation["messages"][0]["content"],
        "completion": completion,
        "reference_answer": extract_answer(reference),
        "parsed_answer": extract_answer(completion),
        "correct": bool(correct),
    }
    with output.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    next_index += world_size
    return correct

GSM8K.evaluate = capture
runpy.run_module("scripts.chat_eval", run_name="__main__", alter_sys=True)
"""


@app.function(cpu=8, timeout=10800)
def evaluate(
    base_path: str,
    sft_output_tag: str,
    model_tag: str,
    model_step: int,
    gpus: int,
    full_test: bool,
):
    base_dir = Path("/vol") / base_path
    run_dir = base_dir / "sft_runs" / sft_output_tag
    checkpoint = run_dir / "chatsft_checkpoints" / model_tag
    tokenizer_dir = base_dir / "tokenizer"
    prefix = f"{model_step:06d}"
    metadata_path = checkpoint / f"meta_{prefix}.json"
    for path in (
        checkpoint / f"model_{prefix}.pt",
        metadata_path,
        tokenizer_dir / "tokenizer.pkl",
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty evaluation input: {path}")
    if json.loads(metadata_path.read_text())["step"] != model_step:
        raise ValueError("Checkpoint step and metadata disagree.")

    launcher = ["python"]
    if gpus > 1:
        launcher = ["torchrun", "--standalone", f"--nproc_per_node={gpus}"]
    arguments = [
        "--source", "sft", "--model-tag", model_tag,
        "--step", str(model_step), "--task-name", "GSM8K",
        "--num-samples", "1", "--temperature", "0", "--max-new-tokens", "512",
        "--top-k", "50", "--device-type", "cuda",
    ]
    expected_count = 1319 if full_test else 24
    if not full_test:
        arguments += ["--max-problems", str(expected_count)]

    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory)
        (scratch / "chatsft_checkpoints").symlink_to(
            run_dir / "chatsft_checkpoints", target_is_directory=True
        )
        (scratch / "tokenizer").symlink_to(tokenizer_dir, target_is_directory=True)

        dataset_path = Path("task_data/openai--gsm8k/main/test")
        cached_test = run_dir / dataset_path
        if (cached_test / "manifest.json").is_file():
            dataset_target = scratch / dataset_path
            dataset_target.parent.mkdir(parents=True)
            dataset_target.symlink_to(cached_test, target_is_directory=True)
            print(f"Using cached GSM8K test data: {cached_test}", flush=True)
        else:
            print("Downloading GSM8K test data into temporary storage.", flush=True)

        capture_script = scratch / "capture_eval.py"
        capture_script.write_text(CAPTURE_EVAL)
        env = dict(
            os.environ,
            NANOCHAT_BASE_DIR=directory,
            GSM8K_CAPTURE_DIR=directory,
            PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS="1",
        )
        env["PYTHONPATH"] = "/root/nanochat" + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        subprocess.run(
            launcher + [str(capture_script)] + arguments,
            cwd="/root/nanochat", env=env, check=True,
        )
        records = [
            json.loads(line)
            for path in Path(directory).glob("rank-*.jsonl")
            for line in path.read_text().splitlines()
        ]
        records.sort(key=lambda record: record["shuffled_test_index"])
        if len(records) != expected_count or any(
            record["shuffled_test_index"] != index for index, record in enumerate(records)
        ):
            raise RuntimeError(
                f"Expected {expected_count} indexed GSM8K completions, got {len(records)} records."
            )
        return records


@app.local_entrypoint()
def main(
    volume: str,
    base_path: str,
    sft_output_tag: str,
    model_tag: str,
    model_step: int,
    gpus: int = 8,
    gpu_type: str = "H100!",
):
    if not volume.strip():
        raise ValueError("Pass a Modal volume name with --volume.")
    if not base_path.strip() or Path(base_path).is_absolute() or ".." in Path(base_path).parts:
        raise ValueError("--base-path must be a relative path within the volume.")
    for flag, value in (("--sft-output-tag", sft_output_tag), ("--model-tag", model_tag)):
        if not value.strip() or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"{flag} must be a single directory name.")
    if model_step < 0:
        raise ValueError("--model-step must be nonnegative.")
    if gpus not in (1, 2, 4, 8):
        raise ValueError("Use 1, 2, 4, or 8 GPUs.")
    gpu_type = gpu_type.strip()
    if not gpu_type or ":" in gpu_type:
        raise ValueError("Pass the GPU type via --gpu-type and the GPU count via --gpus.")

    gpu_request = f"{gpu_type}:{gpus}" if gpus > 1 else gpu_type
    records = evaluate.with_options(
        gpu=gpu_request,
        volumes={"/vol": modal.Volume.from_name(volume).with_mount_options(read_only=True)},
    ).remote(base_path, sft_output_tag, model_tag, model_step, gpus, full_test)
    mode = "full-test" if full_test else "replay-24"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(f"gsm8k-sft-step{model_step}-{mode}-{timestamp}.jsonl")
    with output.open("x") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    print(f"Saved {len(records)} completions to {output.resolve()}")
