"""Sequential experiment control; training and scoring are supplied by the launcher."""

import copy
import math
import re


def validate_plan(plan):
    required = {"workspace": "justinzhaoovo", "destination_volume": "vocalocate-a4-part4",
                "base_path": "a3", "sft_tag": "d20-baseline-long-sft",
                "model_tag": "d20-baseline-long", "model_step": 932}
    if any(plan.get(k) != v for k, v in required.items()):
        raise ValueError("Part 4 requires the configured workspace, volume and SFT initialization")
    training = {"optimizer_updates": 467, "questions_per_update": 16, "samples_per_question": 16,
                "temperature": 1.0, "top_k": 50, "max_new_tokens": 256, "gpus": 8,
                "learning_rates": "unchanged chat_rl.py defaults"}
    if plan["training"] != training:
        raise ValueError("Use the unchanged Part 3 training configuration")
    runs = plan["runs"]
    tags = [r["output_tag"] for r in runs]
    if (len(runs) != 3 or len(set(tags)) != 3 or plan["smoke_output_tag"] in tags
            or any(not re.fullmatch(r"a4-p4-[a-z0-9-]+", t) for t in tags)):
        raise ValueError("Use three distinct formal Part 4 tags, separate from the smoke test")


def recovery_progress(plan, saved, *, resume=False, owner_call_id=None):
    """Recover a failed suite or retry the same Modal input after preemption."""
    validate_plan(plan)
    if saved is None:
        if resume:
            raise FileNotFoundError("No saved suite to resume")
        return []
    same_call = bool(owner_call_id) and saved.get("owner_call_id") == owner_call_id
    if not resume and not same_call:
        raise FileExistsError("A saved suite exists; inspect it and use --resume to continue")
    experiment = lambda p: {k: v for k, v in p.items() if k not in {"execution", "status"}}
    allowed_states = {"failed", "training", "evaluating", "complete"} if same_call else {"failed"}
    if saved.get("status") not in allowed_states or experiment(saved["plan"]) != experiment(plan):
        raise ValueError("Resume requires a failed attempt or the same Modal call and experiment")
    completed = copy.deepcopy(saved.get("completed", []))
    if len(completed) > len(plan["runs"]):
        raise ValueError("Saved progress exceeds the plan")
    for item, run in zip(completed, plan["runs"]):
        if any(item.get(k) != v for k, v in run.items()):
            raise ValueError("Completed experiments must match the plan in order")
    if len(completed) == len(plan["runs"]):
        if same_call and saved["status"] == "complete" and saved.get("current") is None:
            return completed
        raise ValueError("No unfinished experiment to resume")
    if saved.get("current") != plan["runs"][len(completed)]["output_tag"]:
        raise ValueError("Saved current experiment does not follow the completed experiments")
    return completed


def execute_runs(plan, train_one, eval_one, record, *, completed=()):
    """Stop the suite on a failed call or missing strict score before spending on another run."""
    validate_plan(plan)
    completed = copy.deepcopy(list(completed))
    current = None
    try:
        for run in plan["runs"][len(completed):]:
            current = run["output_tag"]
            record({"status": "training", "current": current, "completed": copy.deepcopy(completed)})
            training = train_one(run)
            if training.get("returncode") != 0:
                raise RuntimeError(f"Training failed: {current}")
            record({"status": "evaluating", "current": current, "completed": copy.deepcopy(completed)})
            evaluation = eval_one(run)
            result = evaluation.get("rl_step000466", {})
            accuracy = result.get("GSM8K", float("nan"))
            if result.get("returncode") != 0 or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
                raise RuntimeError(f"Final strict evaluation failed: {current}")
            completed.append({**run, "training": training, "evaluation": evaluation})
        record({"status": "complete", "current": None, "completed": copy.deepcopy(completed)})
    except Exception as exc:
        record({"status": "failed", "current": current, "completed": copy.deepcopy(completed),
                "error": f"{type(exc).__name__}: {exc}"})
        raise
    return completed
