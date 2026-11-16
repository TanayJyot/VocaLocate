"""Reward / eval curves for our RL run vs Karpathy's d20 run (digitized from his W&B screenshot).

Usage:
    python plot_curves.py --log <run>.log --ref <reference_dir> --out <figures_dir>
        [--final-json final_eval.json]   # optional: {"sft": acc, "rl": acc} full-test greedy GSM8K

Writes:
    fig_rl_reward.png         training reward vs step (ours raw + EMA, Karpathy W&B-smoothed)
    fig_rl_passk_step.png     pass@1 and pass@8 vs step (Karpathy pass@8 mapped eval-index -> step)
    fig_rl_pass8_minutes.png  pass@8 vs wall-clock minutes (both runs, as in Karpathy's panel)
    fig_rl_seqlen.png         mean sampled sequence length vs step (ours only; Karpathy did not plot it)
    curves_summary.csv        per-eval numbers for both runs
"""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#1f2328", "#59636e", "#e3e6ea"
OURS, KARP = "#2f6fb3", "#d1652a"   # blue / orange: CVD-safe pair; Karpathy is also dashed
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False})
OURS_LABEL = "Ours: d20 baseline-long (SFT step 932)"
KARP_LABEL = "Karpathy d20speed (digitized from W&B screenshot)"

STEP_RE = re.compile(r"\[\s*([\d.]+)s\] Step (\d+)/(\d+) \| Average reward: ([\d.eE-]+) \| Average sequence length: ([\d.]+)")
EVAL_RE = re.compile(r"\[\s*([\d.]+)s\] Step (\d+) \| (Pass@1: .*)")
START_RE = re.compile(r"\[\s*([\d.]+)s\] Calculated number of steps")


def parse_log(path):
    steps, evals, t_start = [], [], None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if (m := START_RE.search(line)) and t_start is None:  # first start only (a resumed run prints it again)
            t_start = float(m.group(1))
        elif m := STEP_RE.search(line):
            steps.append(dict(t=float(m.group(1)), step=int(m.group(2)), reward=float(m.group(4)),
                              seqlen=float(m.group(5))))
        elif m := EVAL_RE.search(line):
            ks = dict((int(k), float(v)) for k, v in re.findall(r"Pass@(\d+): ([\d.]+)", m.group(3)))
            # the eval line is printed when the eval FINISHES; Karpathy's W&B x-axis is the log time too
            evals.append(dict(t=float(m.group(1)), step=int(m.group(2)), **{f"pass@{k}": v for k, v in ks.items()}))
    return steps, evals, t_start or 0.0


def ema(xs, w=0.9):
    out, s = [], None
    for x in xs:
        s = x if s is None else w * s + (1 - w) * x
        out.append(s)
    return out


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def finish(fig, ax, out, note=None):
    ax.legend(loc="best", fontsize=8.5)
    if note:
        fig.text(0.5, 0.01, note, ha="center", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.05 if note else 0, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--final-json")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = Path(a.ref)
    steps, evals, t0 = parse_log(a.log)
    k_rew = read_csv(ref / "karpathy_d20_reward_smoothed.csv")
    k_p1 = read_csv(ref / "karpathy_d20_pass1.csv")
    k_p8 = read_csv(ref / "karpathy_d20_pass8.csv")

    # 1. reward
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = [s["step"] for s in steps]
    r = [s["reward"] for s in steps]
    ax.plot(x, r, color=OURS, lw=0.6, alpha=0.3)
    ax.plot(x, ema(r), color=OURS, lw=1.8, label=OURS_LABEL + " (EMA 0.9; faint = raw)")
    ax.plot([int(d["step"]) for d in k_rew], [float(d["reward_wandb_smoothed"]) for d in k_rew],
            color=KARP, lw=1.8, ls="--", label=KARP_LABEL + " (W&B-smoothed)")
    ax.set(xlabel="RL step", ylabel="mean training reward (256 rollouts)", title="Training reward")
    ax.set_xlim(0, 467)
    finish(fig, ax, out / "fig_rl_reward.png",
           "Train split, 16 questions x 16 samples per step. With the stock reward (Part 3 and Karpathy) the reward is "
           "strict GSM8K correctness (#### marker required); shaped-reward runs plot their own reward.")

    # 2. pass@1 / pass@8 vs step
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, k in zip(axes, (1, 8)):
        ax.plot([e["step"] for e in evals], [e[f"pass@{k}"] for e in evals], color=OURS, lw=2,
                marker="o", ms=5, mec="white", mew=1.2, label="Ours")
        if k == 1:
            kx, ky = [int(d["step"]) for d in k_p1], [float(d["pass@1"]) for d in k_p1]
        else:
            kx, ky = [int(d["assumed_step"]) for d in k_p8], [float(d["pass@8"]) for d in k_p8]
        ax.plot(kx, ky, color=KARP, lw=2, ls="--", marker="s", ms=4.5, mec="white", mew=1.0,
                label="Karpathy (digitized)" + (" *" if k == 8 else ""))
        ax.set(xlabel="RL step", ylabel=f"pass@{k}", title=f"pass@{k} on 400 test questions (T=1.0)")
        ax.set_xlim(-10, 480)
        ax.legend(fontsize=8.5)
    fig.text(0.5, 0.01, "* Karpathy plotted pass@8 against minutes; his 8 points are mapped in order to the evals at "
             "steps 0, 60, ..., 420.", ha="center", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "fig_rl_passk_step.png", dpi=200)
    plt.close(fig)
    print("wrote", out / "fig_rl_passk_step.png")

    # 3. pass@8 vs minutes
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([(e["t"] - t0) / 60 for e in evals], [e["pass@8"] for e in evals], color=OURS, lw=2,
            marker="o", ms=5, mec="white", mew=1.2, label=OURS_LABEL)
    ax.plot([float(d["minutes"]) for d in k_p8], [float(d["pass@8"]) for d in k_p8], color=KARP, lw=2,
            ls="--", marker="s", ms=4.5, mec="white", mew=1.0, label=KARP_LABEL)
    ax.set(xlabel="wall-clock minutes since start of training loop", ylabel="pass@8",
           title="pass@8 vs time (8xH100)")
    finish(fig, ax, out / "fig_rl_pass8_minutes.png",
           "Our minutes are measured from the 'Calculated number of steps' log line; his start point is unknown (first eval at ~2.3 min).")

    # 4. sequence length
    fig, ax = plt.subplots(figsize=(7, 3.6))
    sl = [s["seqlen"] for s in steps]
    ax.plot(x, sl, color=OURS, lw=0.6, alpha=0.3)
    ax.plot(x, ema(sl), color=OURS, lw=1.8, label=OURS_LABEL + " (EMA 0.9)")
    ax.set(xlabel="RL step", ylabel="mean sampled tokens", title="Mean rollout length (max 256 new tokens)")
    ax.set_xlim(0, 467)
    finish(fig, ax, out / "fig_rl_seqlen.png", "Karpathy's screenshot has no sequence-length panel.")

    # 5. summary csv
    kp1 = {int(d["step"]): float(d["pass@1"]) for d in k_p1}
    kp8 = {int(d["assumed_step"]): float(d["pass@8"]) for d in k_p8}
    with open(out / "curves_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "ours_minutes", "ours_pass@1", "ours_pass@8", "karpathy_pass@1", "karpathy_pass@8"])
        for e in evals:
            w.writerow([e["step"], round((e["t"] - t0) / 60, 1), e["pass@1"], e["pass@8"],
                        kp1.get(e["step"], ""), kp8.get(e["step"], "")])
    if a.final_json:
        fj = json.loads(Path(a.final_json).read_text())
        print("final full-test GSM8K:", fj, "| Karpathy: SFT 0.0455 -> RL 0.0758")


if __name__ == "__main__":
    main()
