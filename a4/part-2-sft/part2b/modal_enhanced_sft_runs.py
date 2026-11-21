"""Part 2b preflight and concurrent SFT launches.

Default invocation checks inputs on a CPU Modal worker. GPU jobs require --launch.
The local image includes NANOCHAT_REPO. JSONL files and the original SFT
dataset cache must be present on the Modal volume.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

import modal


SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent == Path("/root"):
    # Modal imports this module again inside the image; the local source has already been mounted.
    REPO = Path("/root/nanochat")
else:
    REPO = Path(os.environ.get("NANOCHAT_REPO", SCRIPT_PATH.parent / "derived_nanochat")).expanduser().resolve()
if not (REPO / "scripts" / "chat_sft.py").is_file():
    raise FileNotFoundError(f"Set NANOCHAT_REPO to the derived nanochat checkout: {REPO}")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "numpy", "tiktoken", "filelock", "psutil", "pyarrow", "requests",
        "jinja2", "datasets", "kernels==0.11.7", "huggingface-hub==0.34.4",
        "rustbpe", "wandb",
    )
    .add_local_dir(REPO, "/root/nanochat", ignore=[".venv", ".git", "**/__pycache__", "**/.pytest_cache"])
)
app = modal.App("nanochat-a4-part2b-kodcode-sft", image=image)
RUNS = ("metamath", "kodcode", "both")
ORIGINAL_SFT_CACHE_SPLITS = (
    ("HuggingFaceTB--smol-smoltalk", "default", "train"),
    ("cais--mmlu", "all", "auxiliary_train"),
    ("openai--gsm8k", "main", "train"),
    ("HuggingFaceTB--smol-smoltalk", "default", "test"),
    ("cais--mmlu", "all", "test"),
    ("openai--gsm8k", "main", "test"),
    ("allenai--ai2_arc", "ARC-Easy", "test"),
    ("allenai--ai2_arc", "ARC-Challenge", "test"),
    ("openai--openai_humaneval", "openai_humaneval", "test"),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_base(base_path):
    path = Path(base_path)
    if not base_path or path.is_absolute() or ".." in path.parts:
        raise ValueError("base_path must be relative within the Modal volume")
    return Path("/vol") / path


def checked_name(value, label):
    if not value or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a single path component")
    return value


def inputs(base, model_tag, model_step, metamath_file, kodcode_file, gpus_per_job=8,
           optimizer_policy="restore", kodcode_subdir=""):
    prefix = f"{model_step:06d}"
    checkpoint = base / "base_checkpoints" / model_tag
    files = {
        "model": checkpoint / f"model_{prefix}.pt",
        "metadata": checkpoint / f"meta_{prefix}.json",
        "tokenizer": base / "tokenizer" / "tokenizer.pkl",
        "token_bytes": base / "tokenizer" / "token_bytes.pt",
        "metamath": base / "sft_data" / metamath_file,
        "kodcode": base / "sft_data" / kodcode_subdir / kodcode_file,
    }
    if optimizer_policy == "restore":
        files.update({f"optimizer_rank{rank}": checkpoint / f"optim_{prefix}_rank{rank}.pt"
                      for rank in range(gpus_per_job)})
    missing = [str(path) for path in files.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing or empty Modal volume inputs: {missing}")
    metadata = json.loads(files["metadata"].read_text())
    if metadata.get("step") != model_step or metadata.get("model_config", {}).get("vocab_size", 0) <= 0:
        raise ValueError("Checkpoint metadata does not match the requested step/model")
    return files


def original_sft_cache(base):
    root = base / "sft_runs" / "a4-original-sft" / "task_data"
    missing = []
    for parts in ORIGINAL_SFT_CACHE_SPLITS:
        split_dir = root.joinpath(*parts)
        manifest = split_dir / "manifest.json"
        if not manifest.is_file():
            missing.append(str(manifest))
            continue
        try:
            filenames = json.loads(manifest.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid original SFT cache manifest: {manifest}") from exc
        if not filenames or any(
            not isinstance(name, str)
            or Path(name).name != name
            or not (split_dir / name).is_file()
            or (split_dir / name).stat().st_size == 0
            for name in filenames
        ):
            missing.append(str(split_dir))
    if missing:
        raise FileNotFoundError(
            "Original SFT dataset cache is incomplete: "
            f"{missing}. Refusing to download datasets during GPU training."
        )
    return root


def run_args(kind, files):
    args = []
    if kind in ("metamath", "both"):
        args.append(f"--metamath-jsonl={files['metamath']}")
    if kind in ("kodcode", "both"):
        args.append(f"--kodcode-jsonl={files['kodcode']}")
    return args


@app.function(cpu=4, timeout=1800)
def preflight(base_path, model_tag, model_step, metamath_file, kodcode_file, output_prefix,
              gpus_per_job, optimizer_policy, run_kind,
              kodcode_subdir, kodcode_train_rows):
    base = checked_base(base_path)
    files = inputs(base, model_tag, model_step, metamath_file, kodcode_file,
                   gpus_per_job, optimizer_policy, kodcode_subdir)
    cache = original_sft_cache(base)
    help_text = subprocess.run(
        ["python", "-m", "scripts.chat_sft", "--help"], cwd="/root/nanochat",
        capture_output=True, text=True, check=True,
    ).stdout
    for flag in ("--metamath-jsonl", "--kodcode-jsonl", "--max-optimizer-steps", "--total-batch-size"):
        if flag not in help_text:
            raise RuntimeError(f"Derived chat_sft.py lacks {flag}")
    manifest_paths = {
        "metamath": base / "sft_data" / "manifest.json",
        "kodcode": base / "sft_data" / kodcode_subdir / "kodcode_manifest.json",
    }
    manifests = {}
    for source, path in manifest_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset manifest: {path}")
        manifests[source] = json.loads(path.read_text())
    datasets = {}
    for source in ("metamath", "kodcode"):
        split_info = {}
        ids = set()
        groups = set()
        for split in ("train", "val"):
            path = base / "sft_data" / (kodcode_subdir if source == "kodcode" else "") / f"{source}_{split}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"Missing source split: {path}")
            count = 0
            split_ids = set()
            split_groups = set()
            with path.open() as stream:
                for number, line in enumerate(stream, 1):
                    row = json.loads(line)
                    if row.get("source") != source or not isinstance(row.get("messages"), list) or len(row["messages"]) != 2:
                        raise ValueError(f"Invalid source/messages in {path}:{number}")
                    audit_id = row.get("audit_id")
                    if not isinstance(audit_id, str) or not audit_id or audit_id in split_ids:
                        raise ValueError(f"Missing/duplicate audit ID in {path}:{number}")
                    split_ids.add(audit_id)
                    split_groups.add(row.get("source_group", audit_id))
                    count += 1
            digest = sha256(path)
            expected = manifests[source].get("outputs", {}).get(path.name, {})
            if expected.get("rows") != count or expected.get("sha256") != digest:
                raise ValueError(f"Manifest row count or SHA-256 mismatch for {path}")
            if ids.intersection(split_ids) or groups.intersection(split_groups):
                raise ValueError(f"Train/validation overlap in {source}")
            ids.update(split_ids)
            groups.update(split_groups)
            split_info[split] = {"path": str(path), "rows": count, "sha256": digest}
        datasets[source] = split_info
    sizes = {source: "train" if files[source].name == f"{source}_train.jsonl" else None for source in ("metamath", "kodcode")}
    if None in sizes.values():
        raise ValueError("Expected metamath_train.jsonl and kodcode_train.jsonl")
    if (datasets["metamath"]["train"]["rows"], datasets["kodcode"]["train"]["rows"],
        datasets["metamath"]["val"]["rows"], datasets["kodcode"]["val"]["rows"]) != (60000, kodcode_train_rows, 3000, 1000):
        raise ValueError(f"Expected train 60k/{kodcode_train_rows} and validation 3k/1k rows")
    selected_runs = RUNS if run_kind == "all" else (run_kind,)
    for kind in selected_runs:
        output_tag = f"{output_prefix}-{kind}"
        output = base / "sft_runs" / output_tag / "chatsft_checkpoints" / model_tag
        result = base / "results" / f"{output_tag}.json"
        log = base / "logs" / f"{output_tag}.log"
        if any(p.exists() for p in (output, result, log)):
            raise FileExistsError(f"Existing output for {output_tag}; use a new output prefix")
    return {
        "checkpoint": str(files["model"]), "checkpoint_bytes": files["model"].stat().st_size,
        "metadata": str(files["metadata"]), "datasets": datasets,
        "dataset_manifests": {source: {"path": str(path), "sha256": sha256(path)}
                              for source, path in manifest_paths.items()},
        "original_sft_task_cache": str(cache),
        "derived_chat_sft_sha256": sha256(Path("/root/nanochat/scripts/chat_sft.py")),
        "optimizer_policy": optimizer_policy,
        "planned_runs": [f"{output_prefix}-{kind}" for kind in selected_runs],
    }


@app.function(cpu=8, timeout=12 * 3600)
def train_one(base_path, model_tag, model_step, metamath_file, kodcode_file,
              output_prefix, kind, wandb_mode, wandb_project, wandb_entity,
              volume, gpus_per_job, total_batch_size,
              optimizer_policy, kodcode_subdir):
    if kind not in RUNS:
        raise ValueError(kind)
    base = checked_base(base_path)
    files = inputs(base, model_tag, model_step, metamath_file, kodcode_file,
                   gpus_per_job, optimizer_policy, kodcode_subdir)
    output_tag = f"{output_prefix}-{kind}"
    run_dir = base / "sft_runs" / output_tag
    output_dir = run_dir / "chatsft_checkpoints" / model_tag
    log_path = base / "logs" / f"{output_tag}.log"
    result_path = base / "results" / f"{output_tag}.json"
    if output_dir.exists() or log_path.exists() or result_path.exists():
        raise FileExistsError(f"Output already exists for {output_tag}")
    run_dir.mkdir(parents=True)
    for name in ("base_checkpoints", "tokenizer", "sft_data"):
        (run_dir / name).symlink_to(base / name, target_is_directory=True)
    (run_dir / "task_data").symlink_to(original_sft_cache(base), target_is_directory=True)
    for name in ("logs", "results", "wandb"):
        (base / name).mkdir(parents=True, exist_ok=True)
    script_path = Path("/root/nanochat/scripts/chat_sft.py")
    script = script_path.read_text()
    project_call = 'wandb.init(project="nanochat-sft", name=args.run, config=user_config)'
    flops_field = '"total_training_flops": flops_so_far,'
    if script.count(project_call) != 1 or script.count(flops_field) != 3:
        raise RuntimeError("Unexpected W&B source layout; revise container instrumentation")
    script = script.replace(project_call, 'wandb.init(project=os.environ["WANDB_PROJECT"], name=args.run, config=user_config)')
    script = script.replace(flops_field, flops_field + '\n            "total_training_tokens": args.total_batch_size * step,')
    script_path.write_text(script)
    env = dict(os.environ, NANOCHAT_BASE_DIR=str(run_dir), WANDB_MODE=wandb_mode,
               WANDB_DIR=str(base / "wandb" / output_tag), WANDB_PROJECT=wandb_project,
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
    env.pop("NANOCHAT_REFERENCE_RAW_DIR", None)
    Path(env["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    if wandb_entity:
        env["WANDB_ENTITY"] = wandb_entity
    launcher = (["torchrun", "--standalone", f"--nproc_per_node={gpus_per_job}", "-m", "scripts.chat_sft", "--"]
                if gpus_per_job > 1 else ["python", "-m", "scripts.chat_sft"])
    command = launcher + [
        f"--model-tag={model_tag}",
        f"--model-step={model_step}",
        f"--load-optimizer={1 if optimizer_policy == 'restore' else 0}", f"--run={output_tag}",
        "--max-optimizer-steps=932", "--max-seq-len=2048", "--device-batch-size=16",
        f"--total-batch-size={total_batch_size}", *run_args(kind, files),
    ]
    started = time.time()
    returncode = None
    try:
        with log_path.open("x") as log:
            log.write("Launching: " + shlex.join(command) + "\n")
            log.flush()
            process = subprocess.Popen(command, cwd="/root/nanochat", env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            returncode = process.wait()
    finally:
        result = {
            "command": command, "input_checkpoint": str(files["model"].parent),
            "input_step": model_step, "output_checkpoint": str(output_dir),
            "gpus": gpus_per_job, "load_optimizer": optimizer_policy == "restore",
            "optimizer_policy": optimizer_policy, "wandb_mode": wandb_mode,
            "wandb_project": wandb_project, "wandb_entity": wandb_entity,
            "returncode": returncode, "elapsed_seconds": time.time() - started,
            "log_path": str(log_path), "datasets": {
                key: {"path": str(files[key]), "sha256": sha256(files[key])}
                for key in ("metamath", "kodcode") if key in kind or kind == "both"
            },
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        modal.Volume.from_name(volume).commit()
    if returncode:
        raise RuntimeError(f"SFT failed with exit code {returncode}; see {log_path}")
    if len(list(output_dir.glob("model_*.pt"))) != 1 or len(list(output_dir.glob("meta_*.json"))) != 1:
        raise RuntimeError(f"Expected one final model/metadata checkpoint pair in {output_dir}")
    return result


@app.local_entrypoint()
def main(volume: str = "a4-nanochat-volume", base_path: str = "a4",
         model_tag: str = "d20-baseline-long", model_step: int = 21328,
         metamath_file: str = "metamath_train.jsonl",
         kodcode_file: str = "kodcode_train.jsonl",
         output_prefix: str = "a4-part2b-long-kodcode-g8-v1", wandb_mode: str = "offline",
         wandb_project: str = "a4-nanochat-rl", wandb_entity: str = "",
         wandb_secret: str = "", launch: bool = False,
         gpus_per_job: int = 8,
         total_batch_size: int = 524288, optimizer_policy: str = "restore",
         run_kind: str = "kodcode", kodcode_subdir: str = "",
         kodcode_train_rows: int = 40000):
    checked_base(base_path)
    for label, value in (("model_tag", model_tag), ("output_prefix", output_prefix),
                         ("metamath_file", metamath_file), ("kodcode_file", kodcode_file)):
        checked_name(value, label)
    if kodcode_subdir:
        checked_name(kodcode_subdir, "kodcode_subdir")
    if kodcode_train_rows != 40000:
        raise ValueError("kodcode_train_rows must be 40000")
    if model_step < 0 or wandb_mode not in ("offline", "online") or not wandb_project:
        raise ValueError("Invalid step, W&B mode, or project")
    if not 1 <= gpus_per_job <= 8 or optimizer_policy not in ("restore", "fresh"):
        raise ValueError("gpus_per_job must be 1–8 and optimizer_policy restore|fresh")
    if run_kind not in (*RUNS, "all"):
        raise ValueError("run_kind must be metamath|kodcode|both|all")
    if optimizer_policy == "restore" and gpus_per_job != 8:
        raise ValueError("Eight baseline optimizer shards require 8 H100s; use --optimizer-policy fresh otherwise")
    if total_batch_size <= 0 or total_batch_size % (gpus_per_job * 16 * 2048):
        raise ValueError("total_batch_size must be divisible by gpus_per_job × 16 × 2048")
    if wandb_mode == "online" and not wandb_secret:
        raise ValueError("Online W&B requires a named Modal secret with WANDB_API_KEY")
    options = {"volumes": {"/vol": modal.Volume.from_name(volume)}}
    report = preflight.with_options(**options).remote(
        base_path, model_tag, model_step, metamath_file, kodcode_file, output_prefix,
        gpus_per_job, optimizer_policy, run_kind,
        kodcode_subdir, kodcode_train_rows)
    print(json.dumps(report, indent=2))
    if not launch:
        print(f"Preflight passed. GPU jobs were not launched; use --launch to start {run_kind}.")
        return
    gpu_options = dict(options, gpu=f"H100!:{gpus_per_job}" if gpus_per_job > 1 else "H100!", timeout=12 * 3600)
    if wandb_secret:
        gpu_options["secrets"] = [modal.Secret.from_name(wandb_secret)]
    jobs = {
        kind: train_one.with_options(**gpu_options).spawn(
            base_path, model_tag, model_step, metamath_file, kodcode_file,
            output_prefix, kind, wandb_mode, wandb_project, wandb_entity,
            volume, gpus_per_job, total_batch_size,
            optimizer_policy, kodcode_subdir)
        for kind in (RUNS if run_kind == "all" else (run_kind,))
    }
    for kind, job in jobs.items():
        print(kind, json.dumps(job.get(), indent=2))
