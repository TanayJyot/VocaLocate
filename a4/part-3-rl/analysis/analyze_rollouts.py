"""Exploratory analysis of which GSM8K problems the RL run gets right / wrong, and how that changes.

Inputs : rollouts/rank*.jsonl written by rl_rollout_logger.py (one line per sampled completion:
         kind=train|eval, step, qidx, ref, pred, correct, completion), the GSM8K parquet shards
         (same files nanochat downloads), and nanochat's tokenizer.pkl (for token counts).
Outputs: figures (fig_eda_*.png), tables (*.csv), examples.md, summary.json in --out.

Usage:
  python analyze_rollouts.py --rollouts <dir> --gsm8k <task_data/openai--gsm8k/main> \
      --tokenizer <tokenizer.pkl> --out <dir>

Notes on the data:
  * Training is one epoch: each train question is sampled 16x at exactly one step, so "learning"
    over time is read across questions (early vs late phase), and per-question trajectories come
    from the eval set (the same 400 test questions x 8 samples at steps 0, 60, ..., 420).
  * Rewards are the stock strict scorer: a completion counts only with a '#### <number>' marker.
"""
import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

INK, MUTED, GRID = "#1f2328", "#59636e", "#e3e6ea"
BLUE, ORANGE, GREEN, PURPLE, GREY = "#2f6fb3", "#d1652a", "#2e8b57", "#7b5ea7", "#9aa3ad"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False})

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
CALC_OUT_RE = re.compile(r"<\|output_start\|>(.*?)<\|output_end\|>", re.S)
REF_CALC_RE = re.compile(r"<<([^>]*)>>")
TOPICS = {  # first match wins; ordered from most specific
    "percent": r"%|percent",
    "fraction/ratio": r"\bhalf\b|\bthird\b|\bquarter\b|fraction|\bratio\b|\btwice\b|\btimes as\b",
    "money": r"\$|dollar|\bcents?\b|\bprice\b|\bcosts?\b|\bpays?\b|\bearns?\b|\bsells?\b",
    "time/rate": r"\bhours?\b|\bminutes?\b|\bdays?\b|\bweeks?\b|\bper\b|\bspeed\b|\bmph\b|\byears?\b",
    "measurement": r"\bfeet\b|\binch|\bpounds?\b|\bounces?\b|\bmeters?\b|\bgallons?\b|\bmiles?\b|\bliters?\b",
}
FAIL_ORDER = ["truncated (hit the token limit)", "no ####: ended, right number last", "no ####: ended without marker",
              "#### = a reference intermediate", "#### = one of its own calc results",
              "#### within 10%", "#### off by x10/x100/x60", "#### other wrong"]
FAIL_COLORS = ["#4d4d4d", "#8fb8de", "#c6dbef", ORANGE, "#f2a272", GREEN, PURPLE, GREY]


def to_float(s):
    try:
        return float(str(s).replace(",", "").rstrip("."))
    except ValueError:
        return None


def nums(text):
    return [v for v in (to_float(m) for m in NUM_RE.findall(text)) if v is not None]


def close(a, b, tol=1e-6):
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(b))


def load_split(gsm_dir, split):
    t = pq.read_table(next((Path(gsm_dir) / split).glob("*.parquet")))
    perm = np.random.default_rng(42).permutation(t.num_rows)  # == HubDataset.shuffle(seed=42)
    q, a = t["question"].to_pylist(), t["answer"].to_pylist()
    return pd.DataFrame({"qidx": range(len(perm)), "question": [q[i] for i in perm],
                         "solution": [a[i] for i in perm]})


def question_features(df):
    df = df.copy()
    calcs = df["solution"].map(REF_CALC_RE.findall)
    df["ref_steps"] = calcs.map(len)
    df["ref_has_division"] = calcs.map(lambda cs: any("/" in c for c in cs))
    df["ref_intermediates"] = calcs.map(lambda cs: [to_float(c.rsplit("=", 1)[-1]) for c in cs if "=" in c])
    df["ref_answer"] = df["solution"].map(lambda s: to_float(s.split("####")[-1].strip()))
    df["q_words"] = df["question"].map(lambda s: len(s.split()))
    df["topic"] = df["question"].map(
        lambda s: next((k for k, p in TOPICS.items() if re.search(p, s, re.I)), "other"))
    return df


def failure_category(row, ref_inter):
    c = row["completion"]
    if row["correct"]:
        return "correct"
    has_marker = row["pred"] is not None
    if not has_marker and row["ntok"] >= 254:
        return FAIL_ORDER[0]
    ref = to_float(row["ref"])
    if not has_marker:
        ns = nums(c)
        return FAIL_ORDER[1] if ns and close(ns[-1], ref) else FAIL_ORDER[2]
    pred = to_float(row["pred"])
    if any(close(pred, v) for v in ref_inter[:-1] if v is not None):
        return FAIL_ORDER[3]
    own = [to_float(x.strip()) for x in CALC_OUT_RE.findall(c)]
    if any(close(pred, v) for v in own if v is not None):
        return FAIL_ORDER[4]
    if pred is not None and ref and abs(pred - ref) <= 0.1 * abs(ref):
        return FAIL_ORDER[5]
    if pred and ref and any(close(pred / ref, k) or close(ref / pred, k) for k in (10, 100, 1000, 60)):
        return FAIL_ORDER[6]
    return FAIL_ORDER[7]


def save(fig, path, note=None):
    if note:
        fig.text(0.5, 0.01, note, ha="center", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.05 if note else 0, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--gsm8k", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-clusters", type=int, default=8)
    ap.add_argument("--final-eval", nargs=2, metavar=("SFT_GLOB_DIR_PREFIX", "RL_GLOB_DIR_PREFIX"),
                    help="two path prefixes of greedy full-test GSM8K jsonl shards, e.g. "
                         "dir/gsm8k_sft_step000932 dir/gsm8k_rl_step000466")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    enc = pickle.load(open(a.tokenizer, "rb"))

    rows = [json.loads(l) for p in sorted(Path(a.rollouts).glob("rank*.jsonl"))
            for l in open(p, encoding="utf-8")]
    df = pd.DataFrame(rows)
    # sample index within each (kind, step, qidx) group, in generation order (rank files are per-rank,
    # and each question lives on exactly one rank, so file order == sample order)
    df["sample"] = df.groupby(["kind", "step", "qidx"]).cumcount()
    df["ntok"] = [len(enc.encode(c, allowed_special="all")) for c in df["completion"]]
    df["has_marker"] = df["pred"].notna()
    # crude degeneracy flag: GSM8K solutions are plain ASCII, so any non-ASCII character means junk text
    df["non_ascii"] = df["completion"].map(lambda c: any(ord(ch) > 127 for ch in c))
    train_q = question_features(load_split(a.gsm8k, "train"))
    test_q = question_features(load_split(a.gsm8k, "test"))

    tr = df[df.kind == "train"].merge(train_q, on="qidx", how="left")
    ev = df[df.kind == "eval"].merge(test_q, on="qidx", how="left")
    n_steps = int(tr.step.max()) + 1
    edges = np.linspace(0, n_steps, 4)
    tr["phase"] = pd.cut(tr.step, edges, right=False, include_lowest=True,
                         labels=[f"early (0-{int(edges[1]) - 1})", f"mid ({int(edges[1])}-{int(edges[2]) - 1})",
                                 f"late ({int(edges[2])}-{n_steps - 1})"])
    for d in (tr, ev):
        d["fail"] = [failure_category(r, inter) for r, inter in
                     zip(d[["completion", "correct", "pred", "ref", "ntok"]].to_dict("records"), d["ref_intermediates"])]
    summary = {"train_rollouts": len(tr), "eval_samples": len(ev), "train_steps": n_steps}

    # ---- 1. per-step decomposition: reward = P(####) x P(correct | ####), plus length / truncation
    # Shaped-reward runs (Part 4) log `reward` and `reward_parts` next to the strict `correct`;
    # Part 3 logs only `correct`, where reward == correct.
    shaped = "reward" in tr.columns and tr["reward"].notna().any()
    tr["reward"] = tr["reward"].fillna(tr["correct"]) if shaped else tr["correct"].astype(float)
    g = tr.groupby("step")
    per_step = pd.DataFrame({
        "accuracy": g.correct.mean(),
        "reward": g.reward.mean(),
        "p_marker": g.has_marker.mean(),
        "p_correct_given_marker": g.apply(lambda x: x.correct[x.has_marker].mean() if x.has_marker.any() else 0.0,
                                          include_groups=False),
        "p_truncated": g.apply(lambda x: (x.ntok >= 254).mean(), include_groups=False),
        "mean_tokens": g.ntok.mean(),
        "p_non_ascii": g.non_ascii.mean(),
    })
    if shaped and "reward_parts" in tr.columns:
        parts = pd.DataFrame([p if isinstance(p, dict) else {} for p in tr["reward_parts"]], index=tr.index)
        for c in parts.columns:
            per_step[f"part_{c}"] = parts[c].groupby(tr["step"]).mean()
    # Zero advantage = all 16 samples of a question got the same REWARD (advantage = reward - mean).
    grp = tr.groupby(["step", "qidx"]).agg(acc=("correct", "mean"), n_rewards=("reward", "nunique"))
    grp["zero_adv"] = grp.n_rewards == 1
    zs = pd.DataFrame({
        "all_wrong": (grp.zero_adv & (grp.acc == 0)).groupby("step").mean(),
        "all_right": (grp.zero_adv & (grp.acc == 1)).groupby("step").mean(),
        "equal_reward_other": (grp.zero_adv & grp.acc.between(0, 1, inclusive="neither")).groupby("step").mean(),
    })
    per_step = per_step.join(zs)
    per_step.to_csv(out / "train_per_step.csv")
    roll = lambda s: s.rolling(max(1, n_steps // 30), min_periods=1).mean()
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    ax = axes[0]
    ax.plot(per_step.index, roll(per_step.p_marker), color=BLUE, lw=2, label="P(writes ####)")
    ax.plot(per_step.index, roll(per_step.p_correct_given_marker), color=ORANGE, lw=2, label="P(correct | ####)")
    ax.plot(per_step.index, roll(per_step.accuracy), color=INK, lw=1.5, ls="--", label="accuracy = product")
    if shaped:
        ax.plot(per_step.index, roll(per_step.reward), color=GREEN, lw=1.5, label="shaped training reward")
    ax.set(title="Accuracy decomposition", xlabel="RL step", ylim=(0, None if shaped else 1))
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(per_step.index, roll(per_step.mean_tokens), color=BLUE, lw=2)
    ax.set(title="Completion length", xlabel="RL step", ylabel="mean generated tokens")
    ax = axes[2]
    ax.plot(per_step.index, roll(per_step.p_truncated), color=BLUE, lw=2, label="hit 256-token limit")
    ax.plot(per_step.index, roll(per_step.p_non_ascii), color=ORANGE, lw=2, label="contains non-ASCII junk")
    ax.set(title="Degenerate rollouts", xlabel="RL step", ylabel="share of rollouts", ylim=(0, 1))
    ax.legend(fontsize=8)
    ax = axes[3]
    ax.stackplot(per_step.index, roll(per_step.all_wrong), roll(per_step.all_right),
                 roll(per_step.equal_reward_other), colors=[GREY, GREEN, PURPLE],
                 labels=["all 16 wrong", "all 16 right", "mixed correctness, equal reward"])
    ax.set(title="Zero-advantage questions", xlabel="RL step", ylabel="share of questions", ylim=(0, 1))
    ax.legend(fontsize=8, loc="upper right")
    save(fig, out / "fig_eda_train_dynamics.png",
         f"Rolling means over {max(1, n_steps // 30)} steps. Generated tokens only (the training log's sequence length includes the prompt). "
         "A question whose 16 samples all get the same reward has zero advantage and contributes no gradient.")
    summary["shaped_reward"] = bool(shaped)
    summary["zero_signal_share_overall"] = float(grp.zero_adv.mean())
    summary["marker_rate_first_last_10_steps"] = [float(per_step.p_marker.iloc[:10].mean()),
                                                  float(per_step.p_marker.iloc[-10:].mean())]

    # ---- 2. difficulty: per-question solve rate (16 samples) vs features, by phase
    qtr = tr.groupby(["qidx", "phase"], observed=True).agg(
        solve=("correct", "mean"), ref_steps=("ref_steps", "first"), topic=("topic", "first"),
        q_words=("q_words", "first"), div=("ref_has_division", "first"), question=("question", "first")).reset_index()
    qtr.to_csv(out / "train_questions.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    qtr["steps_bin"] = qtr.ref_steps.clip(upper=6).map(lambda v: f"{v}" if v < 6 else "6+")
    order = sorted(qtr.steps_bin.unique(), key=lambda s: int(s.rstrip("+")))
    phases = list(qtr.phase.cat.categories) if hasattr(qtr.phase, "cat") else sorted(qtr.phase.unique())
    w = 0.8 / len(phases)
    for i, (ph, col) in enumerate(zip(phases, [GREY, BLUE, ORANGE])):
        s = qtr[qtr.phase == ph].groupby("steps_bin").solve.mean().reindex(order)
        ax.bar(np.arange(len(order)) + (i - 1) * w, s.values, width=w, color=col, label=ph)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(range(len(order)), order)
    ax.set(title="Solve rate by reference solution length", xlabel="calculator steps in reference solution",
           ylabel="mean solve rate (16 samples)")
    ax.legend(fontsize=8)
    ax = axes[1]
    tp = qtr.groupby(["topic", "phase"], observed=True).solve.mean().unstack()
    tp = tp.loc[tp.mean(axis=1).sort_values().index]
    for i, (ph, col) in enumerate(zip(tp.columns, [GREY, BLUE, ORANGE])):
        ax.barh(np.arange(len(tp)) + (i - 1) * w, tp[ph].values, height=w, color=col, label=ph)
    ax.grid(axis="y", visible=False)
    ax.set_yticks(range(len(tp)), [f"{t} (n={int((qtr.topic == t).sum())})" for t in tp.index])
    ax.set(title="Solve rate by keyword topic", xlabel="mean solve rate (16 samples)")
    save(fig, out / "fig_eda_difficulty.png",
         "Each train question is sampled once (16 samples) at a single step; phases compare different questions.")
    summary["solve_by_ref_steps"] = qtr.groupby("steps_bin").solve.mean().reindex(order).round(4).to_dict()

    # ---- 3. failure taxonomy by phase (train) and at each eval step
    fail = tr[tr.correct == 0].groupby("phase", observed=True).fail.value_counts(normalize=True).unstack().reindex(
        columns=FAIL_ORDER).fillna(0)
    fail.to_csv(out / "train_failure_mix_by_phase.csv")
    evfail = ev[ev.correct == 0].groupby("step").fail.value_counts(normalize=True).unstack().reindex(
        columns=FAIL_ORDER).fillna(0)
    evfail.to_csv(out / "eval_failure_mix_by_step.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})
    for ax, tab, xl in ((axes[0], fail, "training phase"), (axes[1], evfail, "eval step")):
        bottom = np.zeros(len(tab))
        for cat, col in zip(FAIL_ORDER, FAIL_COLORS):
            ax.bar(range(len(tab)), tab[cat].values, bottom=bottom, color=col, edgecolor="white", lw=1.2,
                   label=cat)
            bottom += tab[cat].values
        ax.grid(axis="x", visible=False)
        ax.set_xticks(range(len(tab)), [str(i) for i in tab.index], fontsize=8.5)
        ax.set(xlabel=xl, ylabel="share of wrong answers", ylim=(0, 1))
    axes[0].set_title("Why answers are wrong: train rollouts")
    axes[1].set_title("Why answers are wrong: eval samples (400 test Qs x 8)")
    axes[1].legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, out / "fig_eda_failure_mix.png", "Rule-based categories, first match wins (order as in legend).")

    # ---- 4. eval set: per-question trajectories, gained/lost, paired bootstrap
    eq = ev.groupby(["step", "qidx"]).agg(acc=("correct", "mean"), any_=("correct", "max"),
                                          first=("correct", "first")).reset_index()
    steps_ev = sorted(eq.step.unique())
    piv = eq.pivot(index="qidx", columns="step", values="any_")
    turnover = []
    for s0, s1 in zip(steps_ev, steps_ev[1:]):
        turnover.append({"from": s0, "to": s1, "gained": int(((piv[s0] == 0) & (piv[s1] == 1)).sum()),
                         "lost": int(((piv[s0] == 1) & (piv[s1] == 0)).sum()),
                         "kept": int(((piv[s0] == 1) & (piv[s1] == 1)).sum())})
    pd.DataFrame(turnover).to_csv(out / "eval_turnover.csv", index=False)
    evs = ev.groupby("step").agg(p_marker=("has_marker", "mean"), p_non_ascii=("non_ascii", "mean"), acc_all_samples=("correct", "mean"),
                                 p_correct_given_marker=("correct", lambda c: c[ev.loc[c.index, "has_marker"]].mean()))
    evs["pass@1_first_sample"] = eq.groupby("step")["first"].mean()
    evs["pass@8"] = eq.groupby("step").any_.mean()
    evs.to_csv(out / "eval_per_step.csv")
    if len(steps_ev) > 1:
        a0 = eq[eq.step == steps_ev[0]].set_index("qidx").acc
        a1 = eq[eq.step == steps_ev[-1]].set_index("qidx").acc.reindex(a0.index)
        diff = (a1 - a0).values
        rng = np.random.default_rng(0)
        boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
        summary["eval_paired_bootstrap"] = {"from": int(steps_ev[0]), "to": int(steps_ev[-1]),
                                            "mean_acc_change": float(diff.mean()),
                                            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
        cats = pd.cut(piv.sum(axis=1), [-1, 0, 2, 5, 99], labels=["never", "1-2 evals", "3-5 evals", "6+ evals"])
        summary["eval_questions_solved_in_n_evals"] = cats.value_counts().sort_index().to_dict()
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        ax = axes[0]
        ax.plot(evs.index, evs.p_marker, color=BLUE, lw=2, marker="o", ms=4, label="P(writes ####)")
        ax.plot(evs.index, evs.p_correct_given_marker, color=ORANGE, lw=2, marker="o", ms=4, label="P(correct | ####)")
        ax.plot(evs.index, evs.acc_all_samples, color=INK, lw=1.5, ls="--", marker="o", ms=4,
                label="accuracy, all 8 samples")
        ax.set(title="Eval decomposition (400 test Qs, T=1.0)", xlabel="RL step", ylim=(0, 1))
        ax.legend(fontsize=8)
        ax = axes[1]
        t = pd.DataFrame(turnover)
        x = np.arange(len(t))
        ax.bar(x, t.gained, color=GREEN, label="gained (0 -> >=1 of 8 correct)")
        ax.bar(x, -t.lost, color=ORANGE, label="lost (>=1 -> 0 of 8 correct)")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.grid(axis="x", visible=False)
        ax.set_xticks(x, [f"{a}->{b}" for a, b in zip(t["from"], t["to"])], fontsize=8, rotation=30)
        ax.set(title="Question turnover between evals (pass@8)", ylabel="questions")
        ax.legend(fontsize=8)
        save(fig, out / "fig_eda_eval_dynamics.png")

    # ---- 5. clustering: TF-IDF + k-means on question text (all train questions seen)
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    qs = tr.groupby("qidx").agg(question=("question", "first"), solve=("correct", "mean"),
                                phase=("phase", "first"), ref_steps=("ref_steps", "first")).reset_index()
    k = min(a.k_clusters, max(2, len(qs) // 10))
    vec = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.5, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
    X = vec.fit_transform(qs.question)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    qs["cluster"] = km.labels_
    terms = np.array(vec.get_feature_names_out())
    names = {c: ", ".join(terms[np.argsort(-km.cluster_centers_[c])[:4]]) for c in range(k)}
    ctab = qs.groupby("cluster").agg(n=("qidx", "size"), solve=("solve", "mean"), ref_steps=("ref_steps", "mean"))
    ctab["top_terms"] = ctab.index.map(names)
    ctab.sort_values("solve").to_csv(out / "clusters.csv")
    heat = qs.groupby(["cluster", "phase"], observed=True).solve.mean().unstack()
    heat = heat.loc[ctab.sort_values("solve").index]
    fig, ax = plt.subplots(figsize=(8, 0.45 * k + 1.6))
    im = ax.imshow(heat.values, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(heat.shape[1]), heat.columns, fontsize=8.5)
    ax.set_yticks(range(k), [f"{names[c]} (n={ctab.n[c]})" for c in heat.index], fontsize=8)
    ax.grid(False)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > np.nanmax(heat.values) * 0.6 else INK)
    fig.colorbar(im, ax=ax, label="mean solve rate")
    ax.set_title(f"TF-IDF k-means clusters of train questions (k={k}) x training phase")
    save(fig, out / "fig_eda_clusters.png")

    # ---- 6. examples per failure category (latest phase / latest eval)
    with open(out / "examples.md", "w", encoding="utf-8") as f:
        f.write("# Example completions per failure category (train rollouts, last phase)\n\n")
        late = tr[tr.phase == tr.phase.cat.categories[-1]] if hasattr(tr.phase, "cat") else tr
        for cat in FAIL_ORDER + ["correct"]:
            sub = late[late.fail == cat]
            f.write(f"## {cat} ({len(sub)} rollouts)\n\n")
            for _, r in sub.sample(min(2, len(sub)), random_state=0).iterrows():
                f.write(f"**Q{r.qidx} (step {r.step}, ref {r.ref}, pred {r.pred})**: {r.question}\n\n```\n"
                        f"{r.completion[:1200]}\n```\n\n")
    # ---- 7. greedy full test: SFT vs RL, per question
    if a.final_eval:
        def load(prefix):
            p = Path(prefix)
            rows = [json.loads(l) for f in sorted(p.parent.glob(p.name + "_rank*.jsonl")) for l in open(f, encoding="utf-8")]
            d = pd.DataFrame(rows).drop_duplicates("qidx").set_index("qidx")
            d["ntok"] = [len(enc.encode(c, allowed_special="all")) for c in d.completion]
            return d
        s_, r_ = load(a.final_eval[0]), load(a.final_eval[1])
        both = s_.join(r_, lsuffix="_sft", rsuffix="_rl", how="inner").join(test_q.set_index("qidx"))
        both["outcome"] = np.select(
            [(both.correct_sft == 1) & (both.correct_rl == 1), (both.correct_sft == 0) & (both.correct_rl == 1),
             (both.correct_sft == 1) & (both.correct_rl == 0)], ["both right", "fixed by RL", "broken by RL"],
            "both wrong")
        for tag in ("sft", "rl"):
            both[f"fail_{tag}"] = [failure_category(
                {"completion": r[f"completion_{tag}"], "correct": r[f"correct_{tag}"], "pred": r[f"pred_{tag}"],
                 "ref": r[f"ref_{tag}"], "ntok": r[f"ntok_{tag}"] if r[f"ntok_{tag}"] < 512 else 254}, r.ref_intermediates)
                for _, r in both.iterrows()]
        both[["outcome", "correct_sft", "correct_rl", "fail_sft", "fail_rl", "ref_steps", "topic", "q_words",
              "pred_sft", "pred_rl", "ref_sft"]].to_csv(out / "fulltest_sft_vs_rl.csv")
        summary["fulltest"] = {"n": len(both), "sft_acc": float(both.correct_sft.mean()),
                               "rl_acc": float(both.correct_rl.mean()),
                               "outcomes": both.outcome.value_counts().to_dict(),
                               "marker_rate": [float(both.pred_sft.notna().mean()), float(both.pred_rl.notna().mean())],
                               "mean_tokens": [float(both.ntok_sft.mean()), float(both.ntok_rl.mean())]}
        b, c = int((both.outcome == "fixed by RL").sum()), int((both.outcome == "broken by RL").sum())
        from math import comb
        n = b + c
        p = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n) if n else 1.0
        summary["fulltest"]["mcnemar_exact_p"] = p
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax = axes[0]
        mix = pd.DataFrame({"SFT": both[both.correct_sft == 0].fail_sft.value_counts(normalize=True),
                            "RL": both[both.correct_rl == 0].fail_rl.value_counts(normalize=True)}).reindex(FAIL_ORDER).fillna(0)
        bottom = np.zeros(2)
        for cat, col in zip(FAIL_ORDER, FAIL_COLORS):
            ax.bar([0, 1], mix.loc[cat].values, bottom=bottom, color=col, edgecolor="white", lw=1.2, label=cat, width=0.6)
            bottom += mix.loc[cat].values
        ax.set_xticks([0, 1], [f"SFT ({both.correct_sft.sum()}/{len(both)} right)", f"RL ({both.correct_rl.sum()}/{len(both)} right)"])
        ax.grid(axis="x", visible=False)
        ax.set(ylabel="share of wrong answers", ylim=(0, 1), title="Why greedy answers are wrong (full test)")
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
        ax = axes[1]
        both["steps_bin"] = both.ref_steps.clip(upper=6).map(lambda v: f"{v}" if v < 6 else "6+")
        order = sorted(both.steps_bin.unique(), key=lambda s: int(s.rstrip("+")))
        g2 = both.groupby("steps_bin")[["correct_sft", "correct_rl"]].mean().reindex(order)
        x = np.arange(len(order))
        ax.bar(x - 0.2, g2.correct_sft, width=0.4, color=GREY, label="SFT")
        ax.bar(x + 0.2, g2.correct_rl, width=0.4, color=BLUE, label="RL")
        ax.set_xticks(x, [o + chr(10) + f"n={int((both.steps_bin == o).sum())}" for o in order], fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.set(xlabel="calculator steps in reference solution", ylabel="greedy accuracy",
               title="Full-test accuracy by solution length")
        ax.legend(fontsize=8)
        save(fig, out / "fig_eda_fulltest_sft_vs_rl.png",
             f"Greedy, 512 new tokens, all {len(both)} test questions. Fixed by RL: {b}, broken by RL: {c} "
             f"(exact McNemar p = {p:.2g}).")

    summary["failure_mix_overall_train"] = tr[tr.correct == 0].fail.value_counts(normalize=True).round(4).to_dict()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
