"""Run the Part 4 suite using Part 3's unchanged training and evaluation functions."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
from datetime import datetime, timezone

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE if HERE == Path("/root") else HERE.parent / "part-3-rl"))
import modal_chat_rl as rl
from run_protocol import validate_plan

VOLUME_NAME = os.environ.get("RL_VOLUME", "")
app = modal.App("vocalocate-a4-part4")
app.include(rl.app)
train, final_eval = rl.train, rl.final_eval
vol = modal.Volume.from_name(VOLUME_NAME or "vocalocate-a4-part4")
cpu_image = modal.Image.debian_slim(python_version=f"{sys.version_info.major}.{sys.version_info.minor}")
cpu_image = cpu_image.add_local_file(HERE / "run_protocol.py", "/root/run_protocol.py")
gpu_image = (rl.image
             .add_local_file(HERE / "run_protocol.py", "/root/run_protocol.py")
             .add_local_file(Path(rl.__file__), "/root/modal_chat_rl.py")
             .add_local_file(HERE / "modal_part4.py", "/root/modal_part4.py"))


@app.function(image=gpu_image, timeout=24 * 3600, cpu=8, gpu="H100!:8",
              env={"RL_VOLUME": VOLUME_NAME}, scaledown_window=2, volumes={"/vol": vol})
def part4_suite(plan, resume=False):
    import sys
    sys.path.insert(0, "/root")
    from run_protocol import execute_runs, validate_plan, recovery_progress
    validate_plan(plan)
    if VOLUME_NAME != plan["destination_volume"]:
        raise ValueError("Set RL_VOLUME to the configured Part 4 volume")
    base = Path("/vol") / plan["base_path"]
    status_path = base / "results" / "part4_suite_v1.json"
    saved = json.loads(status_path.read_text()) if status_path.exists() else None
    owner_call_id = modal.current_function_call_id()
    same_call = saved is not None and saved.get("owner_call_id") == owner_call_id
    completed = recovery_progress(plan, saved, resume=resume, owner_call_id=owner_call_id)
    if len(completed) == len(plan["runs"]):
        return completed
    recovering = resume or same_call
    pending = plan["runs"][len(completed):]
    for index, run in enumerate(pending):
        if (base / "rl_runs" / run["output_tag"]).exists() and not (recovering and index == 0):
            raise FileExistsError(f"Formal output tag already exists: {run['output_tag']}")
    smoke = json.loads((base / "results" / f"{plan['smoke_output_tag']}.json").read_text())
    if smoke.get("returncode") != 0 or smoke.get("smoke_steps") != 10:
        raise RuntimeError("A successful, separate 10-step smoke test is required")

    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    previous_attempts = saved.get("previous_attempts", []) if saved else []
    if recovering:
        archive = status_path.with_name(f"part4_suite_v1.attempt-{attempt}.json")
        archive.write_text(json.dumps(saved, indent=2) + "\n")
        tag = pending[0]["output_tag"]
        run_dir = base / "rl_runs" / tag
        if run_dir.exists():
            if (run_dir / "reward.txt").read_text().strip() != pending[0]["reward"]:
                raise ValueError("Saved training reward differs from the configured run")
            backup = run_dir / "recovery_backups" / attempt
            (backup / "rollouts").mkdir(parents=True)
            log_path = base / "logs" / f"{tag}.log"
            if log_path.exists():
                shutil.copyfile(log_path, backup / "training.log")
            for path in (run_dir / "rollouts").glob("rank*.jsonl"):
                shutil.copyfile(path, backup / "rollouts" / path.name)
        previous_attempts = previous_attempts + [{"archived_state": str(archive),
                                                 "status": saved["status"], "current": saved["current"],
                                                 "error": saved.get("error")}]
        vol.commit()

    def record(state):
        vol.reload()
        status_path.write_text(json.dumps({"plan": plan, "attempt": attempt, "resumed": resume,
                                          "owner_call_id": owner_call_id,
                                          "previous_attempts": previous_attempts, **state}, indent=2) + "\n")
        vol.commit()

    # Run both original implementations inside this GPU function. A separate CPU
    # controller could be preempted while a nested GPU call was still running.

    def train_one(run):
        result_path = base / "results" / f"{run['output_tag']}.json"
        final_model = base / "rl_runs" / run["output_tag"] / "chatrl_checkpoints" / plan["model_tag"] / "model_000466.pt"
        if final_model.exists() and result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get("returncode") == 0 and result.get("reward") == run["reward"] and not result.get("smoke_steps"):
                return result
        return train.local(VOLUME_NAME, plan["base_path"], plan["sft_tag"],
                                    plan["model_tag"], plan["model_step"], run["output_tag"],
                                    run["output_tag"], 8, 0, (), "offline", "a4-rl", run["reward"])

    def eval_one(run):
        vol.reload()
        run_dir = base / "rl_runs" / run["output_tag"]
        existing_eval = run_dir / "final_eval"
        if existing_eval.exists():
            existing_eval.rename(run_dir / f"final_eval_before_resume_{attempt}")
            vol.commit()
        summary = final_eval.local(VOLUME_NAME, plan["base_path"], run["output_tag"],
                                      plan["model_tag"], (("rl", 466),))
        vol.reload()
        rows = []
        for path in (base / "rl_runs" / run["output_tag"] / "final_eval").glob("gsm8k_rl_step000466_rank*.jsonl"):
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        if len(rows) != 1319 or {row["qidx"] for row in rows} != set(range(1319)):
            raise RuntimeError("Final evaluation must save one answer per GSM8K test question")
        result = summary.get("rl_step000466", {})
        correct = sum(row["correct"] for row in rows)
        if abs(correct / 1319 - result.get("GSM8K", -1)) > 0.000051:
            raise RuntimeError("Saved answers disagree with the final evaluation score")
        result.update(GSM8K_correct=correct, GSM8K_total=len(rows))
        return summary

    return execute_runs(plan, train_one, eval_one, record, completed=completed)


@app.function(serialized=True, image=cpu_image, timeout=24 * 3600, cpu=0.125, memory=256, scaledown_window=2)
def kickoff_part4(plan, resume=False):
    return part4_suite.spawn(plan, resume).object_id


@app.local_entrypoint()
def start(manifest: str, resume: bool = False):
    plan = json.loads(Path(manifest).read_text())
    validate_plan(plan)
    if os.environ.get("MODAL_PROFILE") != plan["workspace"] or VOLUME_NAME != plan["destination_volume"]:
        raise ValueError("Set MODAL_PROFILE=justinzhaoovo and RL_VOLUME=vocalocate-a4-part4")
    import rewards
    for run in plan["runs"]:
        rewards.parse_spec(run["reward"])
    plan["execution"] = {
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_launcher_sha256": hashlib.sha256(Path(rl.__file__).read_bytes()).hexdigest(),
        "reward_sha256": hashlib.sha256((HERE.parent / "part-3-rl" / "rewards.py").read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256((HERE / "run_protocol.py").read_bytes()).hexdigest(),
        "run_manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
    }
    call_id = kickoff_part4.remote(plan, resume)
    print(f"Spawned Part 4 suite {call_id}. Status: /a3/results/part4_suite_v1.json")
