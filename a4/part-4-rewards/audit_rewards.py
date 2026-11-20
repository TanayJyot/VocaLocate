"""Replay proposed rewards on existing TRAINING outputs; no model or Modal calls.

GSM8K train JSONL must be in its original Hub row order. The seed-42 permutation
below matches nanochat.tasks.common.HubDataset.shuffle and the logger's qidx.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time

import numpy as np


HERE = Path(__file__).resolve().parent
HOOK = HERE.parent / "part-3-rl" / "rewards.py"
spec = importlib.util.spec_from_file_location("part3_rewards", HOOK)
rewards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rewards)
STRICT_ANSWER = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(text):
    match = STRICT_ANSWER.search(text)
    return match.group(1).replace(",", "") if match else None


def reference_parts(solution):
    parts = []
    for text in re.split(r"(<<[^>]+>>)", solution):
        if text.startswith("<<") and text.endswith(">>") and "=" in text:
            expression, output = text[2:-2].rsplit("=", 1)
            parts.extend([{"type": "python", "text": expression},
                          {"type": "python_output", "text": output}])
        else:
            parts.append({"type": "text", "text": text})
    return parts


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    source_rows = [json.loads(line) for line in args.train_data.read_text(encoding="utf-8").splitlines()]
    if len(source_rows) != 7473:
        raise ValueError("Expected the official 7473-row GSM8K train split")
    order = np.random.default_rng(42).permutation(len(source_rows))
    questions = [source_rows[index] for index in order]
    contexts = [{"ref": extract_answer(q["answer"]), "ref_solution_parts": reference_parts(q["answer"])}
                for q in questions]
    manifest = json.loads((HERE / "runs.json").read_text(encoding="utf-8"))
    compositions = {r["name"]: rewards.parse_spec(r["reward"]) for r in manifest["runs"]}
    counts = Counter()
    groups = defaultdict(list)
    examples = {}
    sources = []
    reference_questions = set()
    files = sorted(args.rollouts.glob("rank*.jsonl"))
    if not files:
        raise ValueError("No rank*.jsonl files found")
    for path in files:
        sources.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)})
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["kind"] != "train":
                counts["nontraining_records_excluded"] += 1
                continue
            qidx, text = row["qidx"], row["completion"]
            ctx = dict(contexts[qidx], pred=extract_answer(text), correct=int(row["correct"]))
            if ctx["ref"] != row["ref"] or ctx["correct"] != int(ctx["pred"] == ctx["ref"]):
                raise ValueError(f"Reference/order/scorer mismatch: {path.name}, question {qidx}")
            f = rewards.score_answer_format(text)
            s = rewards.score_reference_calculations(text, ctx["ref_solution_parts"], ctx["ref"])
            scores = {name: rewards.compute(parts, {}, text, ctx)[0] for name, parts in compositions.items()}
            if not all(0 <= score <= 1 for score in scores.values()):
                raise ValueError("Reward outside [0, 1]")
            if ctx["correct"] and any(score != 1 for score in scores.values()):
                raise ValueError("Strict-correct response changed from reward 1")
            counts["train_responses"] += 1
            counts["strict_correct"] += ctx["correct"]
            counts["complete_answer_format"] += int(f > 0)
            counts["any_reference_calculation"] += int(s > 0)
            counts["reference_coverage_sum"] += s
            counts["correct_with_reference_calculations"] += int(ctx["correct"] and s > 0)
            if not ctx["correct"]:
                counts["incorrect"] += 1
                counts["incorrect_with_format_credit"] += int(f > 0)
                counts["incorrect_with_reference_credit"] += int(s > 0)
                counts["incorrect_with_reference_but_no_final_marker"] += int(s > 0 and ctx["pred"] is None)
            keys = rewards.reference_calculation_keys(ctx["ref_solution_parts"], ctx["ref"])
            if keys:
                reference_questions.add(qidx)
            groups[(row["step"], qidx)].append({"correct": ctx["correct"], **scores})
            case = ("partial_calculation_without_answer" if not ctx["correct"] and s > 0 and ctx["pred"] is None
                    else "formatted_wrong_with_reference_calculations" if not ctx["correct"] and f and s > 0
                    else "formatted_wrong_without_reference_calculations" if not ctx["correct"] and f and not s
                    else "correct_without_matched_reference_calculations" if ctx["correct"] and not s and keys
                         and rewards.TOOL_CALL.search(text)
                    else None)
            if case and case not in examples:
                examples[case] = {"qidx": qidx, "step": row["step"], "rank": row["rank"],
                                  "question": questions[qidx]["question"],
                                  "reference_solution": questions[qidx]["answer"],
                                  "completion": text, "strict_correct": ctx["correct"],
                                  "format_score": f, "reference_score": s, "training_rewards": scores}
    group_summary = Counter()
    per_run = {name: Counter() for name in compositions}
    for group in groups.values():
        if len(group) != 16:
            raise ValueError(f"Incomplete question group: {len(group)} samples")
        correct = [item["correct"] for item in group]
        all_wrong = sum(correct) == 0
        constant = min(correct) == max(correct)
        group_summary["question_groups"] += 1
        group_summary["all_wrong_groups"] += all_wrong
        group_summary["strict_zero_variance_groups"] += constant
        for name in compositions:
            values = [item[name] for item in group]
            varied = max(values) - min(values) > 1e-12
            per_run[name]["nonconstant_reward_groups"] += varied
            per_run[name]["all_wrong_groups_with_nonconstant_reward"] += all_wrong and varied
            per_run[name]["previously_constant_groups_with_nonconstant_reward"] += constant and varied
    summary = {
        "purpose": "Offline reward audit, not a trained-model evaluation",
        "selection": "All train records from the downloaded rank files; no test outputs analyzed",
        "group_size": 16,
        "training_split_rows": len(source_rows),
        "train_data_sha256": digest(args.train_data),
        "reward_code_sha256": digest(HOOK),
        "audit_code_sha256": digest(Path(__file__)),
        "run_manifest_sha256": digest(HERE / "runs.json"),
        "sources": sources,
        "counts": dict(counts),
        "groups": dict(group_summary),
        "questions_with_eligible_reference_calculations": len(reference_questions),
        "per_run": {name: dict(counter) for name, counter in per_run.items()},
        "example_selection": "First training completion satisfying each named condition, in rank-file order",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "reward_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "reward_examples.json").write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": summary["counts"], "groups": summary["groups"],
                      "per_run": summary["per_run"], "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
