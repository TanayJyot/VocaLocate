"""Extra Part 5 analysis: when each run collapsed, accuracy by solution length, and how much of its teacher
each student copied. Uses saved outputs only; no model is run.

Inputs
  --runs DIR     one sub-folder per distillation run (as for compare_runs.py), plus the RL run's final_eval folder
                 at DIR/rl-teacher/final_eval (gsm8k_rl_step000120_rank*.jsonl and gsm8k_rl_step000466_rank*.jsonl,
                 copied from the RL run folder on the Modal volume)
  --results DIR  a4/part-5-distill/results (reads p5_per_step.csv and p5_fulltest.csv written by compare_runs.py)
Part 3 and Part 4 results are read from a4/part-3-rl/results and a4/part-4-rewards/results.

Outputs (in --results): p5_answers.csv, p5_agreement.csv, p5_collapse.csv, p5_by_length.csv,
fig_p5_collapse.png, fig_p5_by_length.png, fig_p5_teacher.png

    python extra_figures.py --runs <dir> --results a4/part-5-distill/results
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_runs import COLORS, GREY, RUNS, plt, save

HERE = Path(__file__).resolve().parent
A4 = HERE.parents[1]
INK, MUTED = "#1f2328", "#59636e"
TEACHER_COLORS = {"SFT": "#c4c9cf", "RL-120": "#9aa3ad", "RL-466": COLORS["RL"]}


def norm(x):
    """Answers as strings without commas or a trailing .0, None for no answer."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    x = str(x).replace(",", "")
    try:
        f = float(x)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return x


def load_eval(pattern_dir, glob):
    rows = [json.loads(l) for f in sorted(Path(pattern_dir).glob(glob)) for l in open(f, encoding="utf-8")]
    d = pd.DataFrame(rows).drop_duplicates("qidx").set_index("qidx").sort_index()
    return pd.DataFrame({"pred": d.pred.map(norm), "correct": d.correct.astype(int)})


def collapse_window(marker, lr_steps):
    """First and last step where fewer than half of the rollouts write #### (5-step centred mean)."""
    m = marker.rolling(5, min_periods=1, center=True).mean()
    low = m[m < 0.5].index
    if not len(low):
        return None
    start, end = int(low.min()), int(low.max())
    return {"start": start, "end": end, "lr_at_start": 1 - start / lr_steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    runs, res = Path(a.runs), Path(a.results)

    # ---- every model's greedy answer on the 1,319 test questions
    models = {"RL-466": load_eval(runs / "rl-teacher" / "final_eval", "gsm8k_rl_step000466_rank*.jsonl"),
              "RL-120": load_eval(runs / "rl-teacher" / "final_eval", "gsm8k_rl_step000120_rank*.jsonl")}
    for tag, (label, _) in RUNS.items():
        models[label] = load_eval(runs / tag / "final_eval", "gsm8k_rl_step*_rank*.jsonl")
    p3 = pd.read_csv(A4 / "part-3-rl" / "results" / "fulltest_sft_vs_rl.csv", index_col="qidx")
    models["SFT"] = pd.DataFrame({"pred": p3.pred_sft.map(norm), "correct": p3.correct_sft})
    p4 = pd.read_csv(A4 / "part-4-rewards" / "results" / "final_per_question.csv")
    p4_names = {"format": "P4 + format", "reference": "P4 + calculations", "combined": "P4 + both"}
    for name, g in p4.groupby("name"):
        g = g.set_index("qidx").sort_index()
        if name == "baseline":  # the same model as RL-466: check, then skip
            assert (g.correct.values == models["RL-466"].correct.values).all()
            continue
        models[p4_names[name]] = pd.DataFrame({"pred": g.pred.map(norm), "correct": g.correct})
    answers = pd.concat({k: v for k, v in models.items()}, axis=1)
    answers.columns = [f"{c}_{m}" for m, c in answers.columns]
    answers.to_csv(res / "p5_answers.csv")

    # ---- agreement: on questions both models get wrong and both answer, how often is it the same number?
    def agree(x, y):
        j = models[x].join(models[y], how="inner", suffixes=("", "_y"))
        both = j[(j.correct == 0) & (j.correct_y == 0) & j.pred.notna() & j.pred_y.notna()]
        same = int((both.pred == both.pred_y).sum())
        right = set(j.index[j.correct == 1]), set(j.index[j.correct_y == 1])
        return {"model": x, "compared_with": y, "both_wrong_and_answered": len(both), "same_wrong_number": same,
                "share_same": same / len(both), "both_right": len(right[0] & right[1]),
                "jaccard_right": len(right[0] & right[1]) / len(right[0] | right[1])}

    pairs = [("D00", "RL-466"), ("D00", "RL-120"), ("D03", "RL-466"), ("D03", "RL-120"), ("D02", "RL-120"),
             ("D02", "RL-466"), ("P4 + format", "RL-466"), ("P4 + calculations", "RL-466"), ("P4 + both", "RL-466"),
             ("RL-120", "RL-466"), ("SFT", "RL-466")]
    agreement = pd.DataFrame([agree(*p) for p in pairs])
    agreement.to_csv(res / "p5_agreement.csv", index=False)

    # ---- collapse windows (all runs that use chat_rl.py's optimizer settings)
    per_step = pd.read_csv(res / "p5_per_step.csv")
    rl_train = pd.read_csv(A4 / "part-3-rl" / "results" / "train_per_step.csv", index_col="step")
    p4_train = pd.read_csv(A4 / "part-4-rewards" / "results" / "training_curves.csv")
    series = {"RL (Part 3)": (rl_train.p_marker, 467)}
    for name in ("format", "reference", "combined"):
        series[p4_names[name]] = (p4_train[p4_train.name == name].set_index("step").p_marker, 467)
    for label in ("D00", "D03", "D02", "D01", "D01b"):
        series[label] = (per_step[per_step.run == label].set_index("step").p_marker, 200)
    collapse = pd.DataFrame([{"run": k, **(collapse_window(m, n) or {})} for k, (m, n) in series.items()])
    collapse.to_csv(res / "p5_collapse.csv", index=False)

    # ---- accuracy by reference solution length
    ft = pd.read_csv(res / "p5_fulltest.csv", index_col="qidx")
    bins = pd.cut(ft.ref_steps, [-1, 1, 2, 3, 4, 5, 99], labels=["0–1", "2", "3", "4", "5", "6+"])
    by_len = pd.DataFrame({m: models[m].correct.reindex(ft.index) for m in ("SFT", "RL-120", "RL-466", "D02", "D00", "D03")})
    by_len = by_len.groupby(bins, observed=False).mean().assign(n=ft.groupby(bins, observed=False).size())
    by_len.to_csv(res / "p5_by_length.csv")

    # ---- Figure: when each run collapsed
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for label, ls in (("RL (Part 3)", "--"), ("D00", "-"), ("D03", "-"), ("D02", "-")):
        m, _ = series[label]
        color = COLORS["RL"] if label.startswith("RL") else COLORS[label]
        ax.plot(m.index, m.rolling(5, min_periods=1, center=True).mean(), color=color, lw=1.8, ls=ls, label=label)
    ax.axhline(0.5, color=GREY, lw=0.8, ls=":")
    ax.set(xlim=(0, 200), ylim=(0, 1.02), xlabel="step", ylabel="share of rollouts",
           title="Rollouts that write #### (5-step centred mean)")
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    order = ["RL (Part 3)", "P4 + format", "P4 + calculations", "P4 + both", "D00", "D03", "D02"]
    for i, run in enumerate(order):
        row = collapse.set_index("run").loc[run]
        color = COLORS.get(run, COLORS["RL"] if run.startswith("RL") else GREY)
        if pd.isna(row.start):
            ax.text(4, i, "never below 50%", va="center", fontsize=8, color=MUTED)
            continue
        ax.barh(i, row.end - row.start, left=row.start, height=0.55, color=color)
        ax.text(row.end + 3, i, f"steps {int(row.start)}–{int(row.end)}, LR {row.lr_at_start:.0%} of peak",
                va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(order)), order)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.set(xlim=(0, 200), xlabel="step", title="Steps where fewer than half write ####")
    save(fig, res / "fig_p5_collapse.png", "P4 = the Part 4 reward runs (467 steps). The learning rate (LR) decays "
                                           "linearly to zero over 467 steps for RL and Part 4 and over 200 for distillation.")

    # ---- Figure: accuracy by solution length
    fig, ax = plt.subplots(figsize=(11, 3.9))
    cols = ["SFT", "RL-120", "RL-466", "D02", "D00", "D03"]
    names = {"SFT": "SFT start", "RL-120": "RL step 120 (D02's teacher)", "RL-466": "RL step 466 (D00/D03's teacher)"}
    colors = [TEACHER_COLORS["SFT"], TEACHER_COLORS["RL-120"], TEACHER_COLORS["RL-466"], COLORS["D02"], COLORS["D00"], COLORS["D03"]]
    w = 0.13
    x = np.arange(len(by_len))
    for k, (c, col) in enumerate(zip(cols, colors)):
        ax.bar(x + (k - 2.5) * w, by_len[c], width=w - 0.015, color=col, label=names.get(c, c))
    ax.set_xticks(x, [f"{g}\n(n = {n})" for g, n in zip(by_len.index, by_len.n)])
    ax.grid(axis="x", visible=False)
    ax.set(xlabel="calculator steps in the reference solution", ylabel="greedy accuracy",
           title="GSM8K full test: accuracy by solution length")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=8, ncol=3)
    save(fig, res / "fig_p5_by_length.png")

    # ---- Figure: what the students copied from their teachers
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ag = agreement.set_index(["model", "compared_with"]).share_same
    bars = [("D00 vs its teacher (RL-466)", ag["D00", "RL-466"], COLORS["D00"], None),
            ("D00 vs RL-120", ag["D00", "RL-120"], COLORS["D00"], "//"),
            ("D03 vs its teacher (RL-466)", ag["D03", "RL-466"], COLORS["D03"], None),
            ("D03 vs RL-120", ag["D03", "RL-120"], COLORS["D03"], "//"),
            ("D02 vs its teacher (RL-120)", ag["D02", "RL-120"], COLORS["D02"], None),
            ("D02 vs RL-466", ag["D02", "RL-466"], COLORS["D02"], "//"),
            ("P4 + format vs RL-466", ag["P4 + format", "RL-466"], GREY, None),
            ("P4 + calculations vs RL-466", ag["P4 + calculations", "RL-466"], GREY, None),
            ("P4 + both vs RL-466", ag["P4 + both", "RL-466"], GREY, None),
            ("RL-120 vs RL-466", ag["RL-120", "RL-466"], GREY, None)]
    for i, (lab, v, col, hatch) in enumerate(bars):
        solid = hatch is None
        ax.barh(i, v, height=0.6, color=col if solid else "white", edgecolor=col, hatch=hatch, lw=1.2)
        ax.text(v + 0.006, i, f"{v:.0%}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(bars)), [b[0] for b in bars], fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set(xlim=(0, 0.52), xlabel="share of shared wrong answers that are the same number",
           title="Same wrong answer as another model")
    ax = axes[1]
    rate = {m: models[m].pred.notna().mean() for m in ("SFT", "RL-120", "D02", "RL-466", "D00", "D03")}
    cols = [TEACHER_COLORS["SFT"], TEACHER_COLORS["RL-120"], COLORS["D02"], TEACHER_COLORS["RL-466"], COLORS["D00"], COLORS["D03"]]
    ax.bar(range(len(rate)), list(rate.values()), color=cols, width=0.6)
    for i, v in enumerate(rate.values()):
        ax.text(i, v + 0.015, f"{v:.0%}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(rate)), ["SFT", "RL-120\n(teacher)", "D02", "RL-466\n(teacher)", "D00", "D03"])
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set(ylim=(0, 1.1), ylabel="share of test answers", title="Test answers that end with a #### answer")
    save(fig, res / "fig_p5_teacher.png", "Left: only questions that both models answer wrongly, each with a number after "
                                          "####. Solid bars: student vs its own teacher; hatched: vs the other teacher; "
                                          "grey: RL runs trained without a teacher.")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(agreement.round(3).to_string(index=False))
        print(collapse.round(3).to_string(index=False))
        print(by_len.round(3))
        print({k: round(v, 3) for k, v in rate.items()})


if __name__ == "__main__":
    main()
