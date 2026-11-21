"""Compare the Part 5 distillation runs with each other and with the Part 3 RL baseline.

Inputs
  --runs DIR   one sub-folder per distillation run, as written by a4/part-3-rl/analysis/fetch_run.py
               (logs/<tag>.log, rollouts/rank*.jsonl, final_eval/{summary.json, gsm8k_rl_step*_rank*.jsonl})
  --gsm8k DIR  task_data/openai--gsm8k/main (parquet shards nanochat downloads), for question features
  --tokenizer  tokenizer.pkl, for completion lengths
The RL baseline (and the SFT start) are read from a4/part-3-rl/results/, so Part 3's raw data is not needed.

Outputs (--out): p5_summary.csv, p5_per_step.csv, p5_fulltest.csv, summary.json and three figures.

    python compare_runs.py --runs <dir> --gsm8k <dir> --tokenizer <tokenizer.pkl> --out <dir>
"""
import argparse
import json
import pickle
import re
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PART3 = HERE.parents[1] / "part-3-rl"
sys.path.insert(0, str(PART3 / "analysis"))
from analyze_rollouts import (FAIL_COLORS, FAIL_ORDER, GREY, failure_category, load_split,  # noqa: E402
                              plt, question_features, save)

plt.rcParams["axes.axisbelow"] = True  # gridlines behind bars

RUNS = {  # tag -> (label, teacher description); order = plotting order
    "a4-p5-d00-rl466": ("D00", "RL step 466"),
    "a4-p5-d01-sdpo": ("D01", "self + reference solution (EMA 0.05)"),
    "a4-p5-d02-rl120": ("D02", "RL step 120"),
    "a4-p5-d03-rl466-reward": ("D03", "RL step 466 + reward"),
    # exploratory, not pre-registered: the SDPO paper's small-model fix, run after D01 collapsed
    "a4-p5-d01b-sdpo-grpo-v2": ("D01b", "frozen self + reference, 0.9 reward + 0.1 distill (exploratory)"),
}
# categorical slots 1-5 of a palette checked for colour-blind separation; RL stays neutral grey
COLORS = {"RL": "#4d4d4d", "D00": "#2a78d6", "D01": "#eb6834", "D02": "#1baf7a", "D03": "#eda100", "D01b": "#e87ba4"}
STAMP = r"^\[\s*([\d.]+)s\]\s*"
STEP_RE = re.compile(STAMP + r"Step (\d+)/\d+ \| Average reward: ([\d.eE+-]+) \| Average sequence length: ([\d.]+)"
                     r" \| Reverse KL: ([\d.eE+-]+|nan) \| Log-ratio std: ([\d.eE+-]+|nan) \| Clip frac: ([\d.]+)", re.M)
PASSK_RE = re.compile(STAMP + r"Step (\d+) \| Pass@1: ([\d.]+),.*Pass@8: ([\d.]+)", re.M)


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def parse_log(path):
    text = Path(path).read_text(errors="replace")
    steps = pd.DataFrame([{"t": float(m[0]), "step": int(m[1]), "reward": float(m[2]), "seq_len": float(m[3]),
                           "reverse_kl": float(m[4]), "log_ratio_std": float(m[5]), "clip_frac": float(m[6])}
                          for m in STEP_RE.findall(text)]).drop_duplicates("step", keep="last").set_index("step")
    evals = pd.DataFrame([{"t": float(m[0]), "step": int(m[1]), "pass@1": float(m[2]), "pass@8": float(m[3])}
                          for m in PASSK_RE.findall(text)]).drop_duplicates("step", keep="last").set_index("step")
    return steps, evals


def load_rollouts(run_dir):
    rows = [json.loads(l) for p in sorted((run_dir / "rollouts").glob("rank*.jsonl")) for l in open(p, encoding="utf-8")]
    df = pd.DataFrame(rows)
    df["has_marker"] = df["pred"].notna()
    df["non_ascii"] = df["completion"].map(lambda c: any(ord(ch) > 127 for ch in c))
    return df


def load_final(run_dir, enc, test_q):
    fe = run_dir / "final_eval"
    shards = sorted(fe.glob("gsm8k_rl_step*_rank*.jsonl"))
    if not shards:
        return None, {}
    rows = [json.loads(l) for f in shards for l in open(f, encoding="utf-8")]
    d = pd.DataFrame(rows).drop_duplicates("qidx").set_index("qidx").join(test_q.set_index("qidx"))
    d["ntok"] = [len(enc.encode(c, allowed_special="all")) for c in d.completion]
    # pandas turns a missing answer (JSON null) into NaN; failure_category expects None for "no #### marker"
    d["fail"] = [failure_category({"completion": r.completion, "correct": r.correct,
                                   "pred": r.pred if isinstance(r.pred, str) else None, "ref": r.ref,
                                   "ntok": r.ntok if r.ntok < 512 else 254}, r.ref_intermediates)
                 for r in d.itertuples()]
    summary = json.loads((fe / "summary.json").read_text()) if (fe / "summary.json").exists() else {}
    return d, next(iter(summary.values()), {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--gsm8k", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs_dir, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    enc = pickle.load(open(a.tokenizer, "rb"))
    test_q = question_features(load_split(a.gsm8k, "test"))

    # ---- RL baseline and SFT start (Part 3 results)
    res3 = PART3 / "results"
    rl_train = pd.read_csv(res3 / "train_per_step.csv", index_col="step")
    rl_eval = pd.read_csv(res3 / "eval_per_step.csv", index_col="step").rename(columns={"pass@1_first_sample": "pass@1"})
    rl_full = pd.read_csv(res3 / "fulltest_sft_vs_rl.csv", index_col="qidx")
    rl_other = json.loads((res3 / "final_eval_rl.json").read_text())["rl_step000466"]
    sft_other = json.loads((res3 / "final_eval_sft.json").read_text())
    sft_other = next(iter(sft_other.values()))

    def mix(fails):
        return fails.value_counts(normalize=True).reindex(FAIL_ORDER).fillna(0)

    rows = [{"run": "SFT", "teacher": "-", "steps": 0, "gsm8k": rl_full.correct_sft.mean(),
             **{k: sft_other.get(k) for k in ("ARC-Easy", "ARC-Challenge", "MMLU", "HumanEval", "ChatCORE")}},
            {"run": "RL", "teacher": "- (Part 3, strict reward)", "steps": 467, "gsm8k": rl_full.correct_rl.mean(),
             **{k: rl_other.get(k) for k in ("ARC-Easy", "ARC-Challenge", "MMLU", "HumanEval", "ChatCORE")},
             "max_non_ascii_train": rl_train.p_non_ascii.max(), "marker_rate_test": rl_full.pred_rl.notna().mean(),
             **{f"wrong: {k}": v for k, v in mix(rl_full.loc[rl_full.correct_rl == 0, "fail_rl"]).items()}}]
    per_step, curves, fulltests = [], {"RL": (rl_train, rl_eval)}, {}
    for tag, (label, teacher) in RUNS.items():
        run_dir = runs_dir / tag
        if not run_dir.exists():
            print(f"skip {tag}: not fetched")
            continue
        steps, evals = parse_log(run_dir / "logs" / f"{tag}.log")
        ro = load_rollouts(run_dir)
        tr = ro[ro.kind == "train"].groupby("step")
        steps = steps.join(pd.DataFrame({"p_marker": tr.has_marker.mean(), "p_non_ascii": tr.non_ascii.mean()}))
        curves[label] = (steps, evals)
        per_step.append(steps.assign(run=label).join(evals[["pass@1", "pass@8"]], how="left"))
        row = {"run": label, "teacher": teacher, "steps": int(steps.index.max()) + 1,
               "train_minutes": steps.t.iloc[-1] / 60, "max_non_ascii_train": steps.p_non_ascii.max(),
               "reverse_kl_first10": steps.reverse_kl.iloc[:10].mean(), "reverse_kl_last10": steps.reverse_kl.iloc[-10:].mean(),
               "pass@1_step60": evals["pass@1"].get(60, np.nan)}
        d, other = load_final(run_dir, enc, test_q)
        if d is not None:
            fulltests[label] = d
            j = d[["correct"]].join(rl_full[["correct_rl"]], how="inner")
            b = int(((j.correct == 1) & (j.correct_rl == 0)).sum())  # distillation right, RL wrong
            c = int(((j.correct == 0) & (j.correct_rl == 1)).sum())
            row.update({"gsm8k": d.correct.mean(), "vs_rl_gained": b, "vs_rl_lost": c, "vs_rl_mcnemar_p": mcnemar(b, c),
                        **{k: other.get(k) for k in ("ARC-Easy", "ARC-Challenge", "MMLU", "HumanEval", "ChatCORE")},
                        "marker_rate_test": d.pred.notna().mean(), "mean_tokens_test": d.ntok.mean(),
                        **{f"wrong: {k}": v for k, v in mix(d.loc[d.correct == 0, "fail"]).items()}})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "p5_summary.csv", index=False)
    if per_step:
        pd.concat(per_step).to_csv(out / "p5_per_step.csv")
    if fulltests:
        ft = rl_full[["correct_sft", "correct_rl", "fail_rl", "ref_steps"]].copy()
        for label, d in fulltests.items():
            ft[f"correct_{label}"], ft[f"fail_{label}"] = d.correct, d.fail
        ft.to_csv(out / "p5_fulltest.csv")

    # ---- Figure 1: in-training pass@k vs step
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric in zip(axes, ("pass@1", "pass@8")):
        for label, (_, ev) in curves.items():
            ax.plot(ev.index, ev[metric], marker="o", ms=4, lw=2, color=COLORS[label],
                    label="RL (Part 3, 467 steps)" if label == "RL" else label, ls="--" if label == "RL" else "-")
        ax.set(xlabel="step", ylabel=metric, title=f"In-training {metric} (400 test questions, T = 1)")
        ax.axvline(199, color=GREY, lw=0.8, ls=":")
    axes[0].legend(fontsize=8)
    save(fig, out / "fig_p5_passk.png", "Distillation runs stop at step 199 (dotted line); RL runs 467 steps with a "
                                        "longer LR decay, so equal steps are not equal learning rates.")

    # ---- Figure 2: training dynamics (15-step means, as in Part 3's Figure 13)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    roll = lambda s: s.rolling(15, min_periods=1).mean()  # noqa: E731
    for label, (st, _) in curves.items():
        kw = dict(color=COLORS[label], lw=1.8, label="RL (Part 3)" if label == "RL" else label, ls="--" if label == "RL" else "-")
        axes[0].plot(st.index, roll(st.reward), **kw)
        if label != "RL":
            axes[1].plot(st.index, roll(st.reverse_kl), **kw)
        axes[2].plot(st.index, roll(st.p_marker), **kw)
        axes[3].plot(st.index, roll(st.p_non_ascii), **kw)
    axes[0].set(ylabel="share of rollouts correct", title="Training accuracy")
    axes[1].set(ylabel="nats per token", title="Reverse KL to the teacher")
    axes[2].set(ylabel="share of rollouts", title="Rollouts that write ####", ylim=(0, 1.02))
    axes[3].set(ylabel="share of rollouts", title="Rollouts containing non-ASCII junk")
    for ax in axes:
        ax.set_xlabel("step")
        ax.set_xlim(0, 210)
    axes[0].legend(fontsize=8, ncol=2)
    save(fig, out / "fig_p5_training.png", "15-step trailing means; RL is shown for its first 210 steps. D01 and D01b measure "
                                           "KL to their own reference-conditioned copy, a different scale from the checkpoint teachers.")

    # ---- Figure 3: full test accuracy and why answers are wrong
    if fulltests:
        labels = ["SFT", "RL", *fulltests]
        acc = [rl_full.correct_sft.mean(), rl_full.correct_rl.mean(), *[d.correct.mean() for d in fulltests.values()]]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        ax = axes[0]
        ax.bar(range(len(labels)), acc, color=[GREY, COLORS["RL"], *[COLORS[l] for l in fulltests]])
        for i, v in enumerate(acc):
            ax.text(i, v + 0.004, f"{v:.1%}", ha="center", fontsize=9)
        ax.set_xticks(range(len(labels)), labels)
        ax.grid(axis="x", visible=False)
        ax.set(ylabel="greedy accuracy", title="GSM8K full test (1,319 questions)")
        ax = axes[1]
        mixes = pd.DataFrame({"RL": mix(rl_full.loc[rl_full.correct_rl == 0, "fail_rl"]),
                              **{l: mix(d.loc[d.correct == 0, "fail"]) for l, d in fulltests.items()}})
        bottom = np.zeros(mixes.shape[1])
        for cat, col in zip(FAIL_ORDER, FAIL_COLORS):
            ax.bar(range(mixes.shape[1]), mixes.loc[cat].values, bottom=bottom, color=col, edgecolor="white", label=cat)
            bottom += mixes.loc[cat].values
        ax.set_xticks(range(mixes.shape[1]), mixes.columns)
        ax.grid(axis="x", visible=False)
        ax.set(ylabel="share of wrong answers", ylim=(0, 1), title="Why greedy answers are wrong")
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
        save(fig, out / "fig_p5_fulltest.png")

    summary = table.set_index("run").round(4).to_dict(orient="index")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.round(4).T)


if __name__ == "__main__":
    main()
