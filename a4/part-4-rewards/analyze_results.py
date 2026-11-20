"""Analyze saved Part 4 outputs locally without loading a model or contacting Modal."""

import hashlib
import json
from collections import defaultdict, Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re

import numpy as np
import pandas as pd

from audit_rewards import extract_answer, reference_parts, rewards

HERE = Path(__file__).resolve().parent
LOCAL = HERE / "local"
OUT = HERE / "results"
NAMES = ("baseline", "format", "reference", "combined")
LABELS = {"baseline": "Correctness only", "format": "+ Format", "reference": "+ Calculations", "combined": "+ Both"}


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def logged_sequence_lengths(text):
    """Total prompt plus completion tokens, averaged across ranks in the RL log."""
    return {int(s): float(v) for s, v in re.findall(
        r"Step (\d+)/467 \| Average reward: [^|]+\| Average sequence length: ([\d.]+)", text)}


def number(value):
    try:
        return Decimal(str(value).replace(",", ""))
    except InvalidOperation:
        return None


def step_band(steps):
    return "0-1" if steps <= 1 else str(steps) if steps < 4 else "4+"


def error_category(pred, ref, intermediate):
    if pred == ref:
        return "Correct"
    if pred is None or pred == "":
        return "No extracted answer"
    if number(pred) in intermediate:
        return "Reference intermediate"
    return "Other wrong answer"


def paired_comparison(a, b):
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    gained, lost = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    # Paired item bootstrap, computed from the three possible item differences.
    draws = np.random.default_rng(17011).multinomial(len(a), [lost / len(a), 1 - (lost + gained) / len(a), gained / len(a)], size=20000)
    delta = 100 * (draws[:, 2] - draws[:, 0]) / len(a)
    return {"gained": gained, "lost": lost, "net": gained - lost,
            "delta_pp": 100 * (gained - lost) / len(a),
            "ci_low_pp": float(np.quantile(delta, .025)), "ci_high_pp": float(np.quantile(delta, .975)),
            "both_correct": int(((a == 1) & (b == 1)).sum()), "both_wrong": int(((a == 0) & (b == 0)).sum())}


def main():
    OUT.mkdir(exist_ok=True)
    test_path = LOCAL / "artifacts/a3/sft_runs/d20-baseline-long-sft/task_data/openai--gsm8k/main/test/00000.parquet"
    original = pd.read_parquet(test_path).to_dict("records")
    test = [original[i] for i in np.random.default_rng(42).permutation(len(original))]
    refs = [extract_answer(x["answer"]) for x in test]
    raw_train = [json.loads(line) for line in (LOCAL / "gsm8k_train.jsonl").read_text(encoding="utf-8").splitlines()]
    training_refs = [extract_answer(raw_train[i]["answer"]) for i in np.random.default_rng(42).permutation(len(raw_train))]
    intermediates = []
    for x, ref in zip(test, refs):
        values = {number(y) for y in re.findall(r"<<[^>]*=([^>]+)>>", x["answer"])}
        intermediates.append(values - {number(ref), None})
    baseline = pd.read_csv(HERE.parent / "part-3-rl/results/fulltest_sft_vs_rl.csv", dtype=str, keep_default_na=False).sort_values("qidx", key=lambda s: s.astype(int))
    assert baseline.qidx.astype(int).tolist() == list(range(1319))
    assert baseline.ref_sft.tolist() == refs
    final_rows, metrics, examples = {}, {}, {}
    final_rows["baseline"] = [{"qidx": int(r.qidx), "pred": r.pred_rl or None,
                               "ref": r.ref_sft, "correct": int(r.correct_rl)} for r in baseline.itertuples()]
    training_frames, periodic_frames = [], []
    baseline_train = pd.read_csv(HERE.parent / "part-3-rl/results/train_per_step.csv")
    baseline_train["name"] = "baseline"
    baseline_train["strict_accuracy"] = baseline_train.reward
    baseline_train["zero_variance"] = baseline_train.all_wrong + baseline_train.all_right
    # The baseline CSV counts completion tokens; the launcher logs total sequence
    # tokens. Use the same log metric for every run instead of mixing the two.
    baseline_lengths = logged_sequence_lengths((LOCAL / "tanay_baseline/training.log").read_text(encoding="utf-8"))
    baseline_train["mean_tokens"] = baseline_train.step.map(baseline_lengths)
    training_frames.append(baseline_train)
    baseline_periodic = pd.read_csv(HERE.parent / "part-3-rl/results/eval_per_step.csv")
    baseline_periodic["name"] = "baseline"
    periodic_frames.append(baseline_periodic)

    for name in NAMES[1:]:
        root = LOCAL / "report_data" / name
        rows = [r for p in sorted((root / "final_eval").glob("gsm8k_*rank*.jsonl")) for r in read_jsonl(p)]
        rows.sort(key=lambda r: r["qidx"])
        assert [r["qidx"] for r in rows] == list(range(1319))
        assert [r["ref"] for r in rows] == refs
        for r in rows:
            assert r["pred"] == extract_answer(r["completion"])
            assert r["correct"] == int(r["pred"] == r["ref"])
            q = r["qidx"]
            parts = reference_parts(test[q]["answer"])
            r["format"] = rewards.score_answer_format(r["completion"])
            r["coverage"] = rewards.score_reference_calculations(r["completion"], parts, r["ref"])
            r["eligible"] = bool(rewards.reference_calculation_keys(parts, r["ref"]))
        final_rows[name] = rows
        accum = defaultdict(lambda: Counter())
        groups = defaultdict(list)
        evals = defaultdict(list)
        total_train = total_eval = 0
        for path in sorted((root / "rollouts").glob("rank*.jsonl")):
            for r in read_jsonl(path):
                assert r["pred"] == extract_answer(r["completion"])
                assert r["correct"] == int(r["pred"] == r["ref"])
                if r["kind"] == "eval":
                    assert r["ref"] == refs[r["qidx"]]
                    total_eval += 1
                    evals[(r["step"], r["qidx"])].append(r["correct"])
                    continue
                total_train += 1
                step = r["step"]
                assert r["ref"] == training_refs[r["qidx"]] and r["qidx"] // 16 == step
                assert abs(sum(r["reward_parts"].values()) - r["reward"]) < 1e-8
                assert 0 <= r["reward"] <= 1
                assert (r["reward"] == 1) if r["correct"] else (r["reward"] <= (.2 if name == "combined" else .1) + 1e-8)
                a = accum[step]
                a["n"] += 1
                a["strict_accuracy"] += r["correct"]
                a["reward"] += r["reward"]
                a["p_marker"] += r["pred"] is not None
                for key in ("answer_format", "reference_calculation"):
                    a[key] += r["reward_parts"].get(key, 0)
                groups[(step, r["qidx"])].append(r["reward"])
        assert total_train == 119552 and total_eval == 25600, (name, total_train, total_eval)
        assert len(groups) == 7472 and all(len(g) == 16 for g in groups.values())
        assert len(evals) == 3200 and all(len(g) == 8 for g in evals.values())
        zero = Counter()
        for (step, _), g in groups.items():
            zero[step] += max(g) - min(g) < 1e-12
        log = (root / "training.log").read_text(encoding="utf-8")
        lengths = logged_sequence_lengths(log)
        data = []
        for step, a in sorted(accum.items()):
            assert a["n"] == 256
            data.append({"name": name, "step": step, **{k: a[k] / 256 for k in ("strict_accuracy", "reward", "p_marker", "answer_format", "reference_calculation")},
                         "zero_variance": zero[step] / 16, "mean_tokens": lengths.get(step, float("nan"))})
        training_frames.append(pd.DataFrame(data))
        pdata = []
        for step in sorted({s for s, q in evals}):
            g = [evals[(step, q)] for q in range(400)]
            pdata.append({"name": name, "step": step, "pass@1_first_sample": np.mean([x[0] for x in g]),
                          "pass@8": np.mean([any(x) for x in g]), "acc_all_samples": np.mean(g)})
        periodic_frames.append(pd.DataFrame(pdata))

    for name, rows in final_rows.items():
        categories = Counter(error_category(r["pred"], r["ref"], intermediates[r["qidx"]]) for r in rows)
        correct = sum(r["correct"] for r in rows)
        parsed = sum(r["pred"] is not None for r in rows)
        m = {"correct": correct, "n": len(rows), "accuracy": correct / len(rows),
             "parsed": parsed, "parseable_rate": parsed / len(rows), "correct_given_parseable": correct / parsed,
             "categories": dict(categories)}
        if name != "baseline":
            eligible = [r for r in rows if r["eligible"]]
            m.update(format_rate=float(np.mean([r["format"] for r in rows])),
                     eligible=len(eligible), coverage_mean=float(np.mean([r["coverage"] for r in eligible])),
                     any_reference_match=sum(r["coverage"] > 0 for r in eligible),
                     full_reference_match=sum(r["coverage"] == 1 for r in eligible),
                     full_reference_match_correct=sum(r["coverage"] == 1 and r["correct"] for r in eligible),
                     wrong_with_match=sum(r["coverage"] > 0 and not r["correct"] for r in eligible))
            m["paired_vs_baseline"] = paired_comparison([r["correct"] for r in final_rows["baseline"]], [r["correct"] for r in rows])
        metrics[name] = m

    for label, name, predicate in (
        ("format_without_correctness", "format", lambda r, b: b["pred"] is None and r["format"] and not r["correct"]),
        ("reference_gain", "reference", lambda r, b: r["correct"] and not b["correct"] and r["coverage"] > 0),
        ("reference_loss_despite_coverage", "reference", lambda r, b: not r["correct"] and b["correct"] and r["coverage"] == 1),
        ("combined_full_coverage_wrong", "combined", lambda r, b: not r["correct"] and r["coverage"] == 1 and r["format"])):
        for r, b in zip(final_rows[name], final_rows["baseline"]):
            if predicate(r, b):
                examples[label] = {"name": name, **r, "question": test[r["qidx"]]["question"],
                                   "reference_solution": test[r["qidx"]]["answer"], "baseline_pred": b["pred"]}
                break

    train = pd.concat(training_frames, ignore_index=True)
    periodic = pd.concat(periodic_frames, ignore_index=True)
    train.to_csv(OUT / "training_curves.csv", index=False)
    periodic.to_csv(OUT / "periodic_eval.csv", index=False)
    by_steps = []
    joined = []
    for name, rows in final_rows.items():
        for r in rows:
            q = r["qidx"]
            steps = int(baseline.iloc[q].ref_steps)
            joined.append({"name": name, "qidx": q, "correct": r["correct"], "pred": r["pred"], "ref": r["ref"],
                           "ref_steps": steps, "category": error_category(r["pred"], r["ref"], intermediates[q]),
                           "format": r.get("format"), "coverage": r.get("coverage")})
        for band in ("0-1", "2", "3", "4+"):
            subset = [r for r in rows if step_band(int(baseline.iloc[r["qidx"]].ref_steps)) == band]
            by_steps.append({"name": name, "reference_steps": band, "n": len(subset), "correct": sum(r["correct"] for r in subset),
                             "accuracy": np.mean([r["correct"] for r in subset])})
        assert sum(item["n"] for item in by_steps if item["name"] == name) == 1319
    pd.DataFrame(joined).to_csv(OUT / "final_per_question.csv", index=False)
    # Confirm that the coarser shared taxonomy reproduces Part 3's saved labels.
    baseline_categories = {r["qidx"]: r["category"] for r in joined if r["name"] == "baseline"}
    for r in baseline.itertuples():
        expected = ("Correct" if r.fail_rl == "correct" else "No extracted answer" if r.fail_rl.startswith("no ####")
                    else "Reference intermediate" if r.fail_rl == "#### = a reference intermediate" else "Other wrong answer")
        assert baseline_categories[int(r.qidx)] == expected
    pd.DataFrame(by_steps).to_csv(OUT / "accuracy_by_steps.csv", index=False)
    phase_summary = {}
    for name in NAMES:
        x = train[(train.name == name) & (train.step >= 311)]
        phase_summary[name] = {k: float(x[k].mean()) for k in ("strict_accuracy", "reward", "zero_variance", "p_marker", "mean_tokens")}
    summary = {"final": metrics, "late_training_steps_311_to_466": phase_summary,
               "periodic_last": periodic[periodic.step == 420][["name", "step", "pass@1_first_sample", "pass@8", "acc_all_samples"]].to_dict("records"),
               "source_test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
               "item_bootstrap_replicates": 20000, "analysis_scope": "All 8 ranks for every new run; baseline aggregates and per-question labels from Part 3"}
    (OUT / "final_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "final_examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
