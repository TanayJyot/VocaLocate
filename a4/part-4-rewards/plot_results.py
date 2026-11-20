"""Build publication-style figures from the saved local analysis tables."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
SUMMARY = json.loads((HERE / "results/final_analysis.json").read_text())
NAMES = ["baseline", "format", "reference", "combined"]
LABELS = ["Correctness only", "+ Format", "+ Calculations", "+ Both"]
COLORS = ["#454545", "#257AA5", "#C47722", "#785B9C"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
                     "axes.titlesize": 10, "legend.fontsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.facecolor": "white", "figure.dpi": 150})


def save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=260, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def lines(ax, frame, column, smooth=False, percentage=False):
    for name, label, color in zip(NAMES, LABELS, COLORS):
        x = frame[frame.name == name].sort_values("step")
        y = x[column].rolling(21, min_periods=1).mean() if smooth else x[column]
        ax.plot(x.step, y, color=color, label=label, lw=1.8, marker=None if smooth else "o", ms=3)
    ax.set_xlabel("Optimizer step")
    ax.set_xlim(0, 466)
    ax.set_xticks([0, 120, 240, 360, 466])
    ax.grid(axis="y", alpha=.15)
    if percentage:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


def main():
    fig, ax = plt.subplots(figsize=(7, 2.65))
    values = [0, 1310 / 1637, 1066 / 1637, 1495 / 1637]
    bars = ax.barh(LABELS[::-1], values[::-1], color=COLORS[::-1], height=.55)
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + .018, bar.get_y() + bar.get_height() / 2, f"{100*value:.1f}%", va="center")
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("All-wrong groups receiving different rewards within the group")
    ax.grid(axis="x", alpha=.15)
    save(fig, "fig4_1_reward_audit")

    train = pd.read_csv(HERE / "results/training_curves.csv")
    periodic = pd.read_csv(HERE / "results/periodic_eval.csv")
    fig, axes = plt.subplots(1, 3, figsize=(8.1, 3.1))
    for ax, col, title in zip(axes, ["strict_accuracy", "reward", "zero_variance"],
                              ["Strict training accuracy", "Total training reward", "Groups with equal rewards"]):
        lines(ax, train, col, smooth=True, percentage=col != "reward")
        ax.set_title(title)
        ax.set_ylim(0, .65 if col != "zero_variance" else 1)
    axes[0].set_ylabel("Fraction of responses")
    axes[1].set_ylabel("Mean reward")
    axes[2].set_ylabel("Fraction of 16-response groups")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(.5, 1.07), frameon=False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig4_2_training")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    for ax, col, title in zip(axes, ["pass@1_first_sample", "pass@8"], ["First sampled answer", "At least one correct answer in eight"]):
        lines(ax, periodic, col, percentage=True)
        ax.set_title(title)
        ax.set_ylim(0, .42)
        ax.set_xlim(0, 420)
        ax.set_xticks([0, 60, 120, 180, 240, 300, 360, 420])
    axes[0].set_ylabel("Pass@1")
    axes[1].set_ylabel("Pass@8")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(.5, 1.08), frameon=False)
    fig.tight_layout(w_pad=2)
    save(fig, "fig4_4_periodic_eval")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    for name, col, color, style, label in [("format", "answer_format", COLORS[1], "-", "Format run: format term"),
                                         ("reference", "reference_calculation", COLORS[2], "-", "Calculation run: calculation term"),
                                         ("combined", "answer_format", COLORS[3], "-", "Combined: format term"),
                                         ("combined", "reference_calculation", COLORS[3], "--", "Combined: calculation term")]:
        x = train[train.name == name]
        axes[0].plot(x.step, x[col].rolling(21, min_periods=1).mean(), color=color, ls=style, label=label)
    axes[0].set(xlabel="Optimizer step", ylabel="Weighted contribution to reward", ylim=(0, .075), xlim=(0, 466))
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    axes[0].set_title("Auxiliary reward contributions")
    lines(axes[1], train, "mean_tokens", smooth=True)
    axes[1].set_title("Total training sequence length")
    axes[1].set_ylabel("Prompt plus completion tokens")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(w_pad=2)
    save(fig, "fig4_3_components_length")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8), gridspec_kw={"width_ratios": [1, 1.2]})
    vals = [SUMMARY["final"][n]["accuracy"] * 100 for n in NAMES]
    axes[0].barh(LABELS[::-1], vals[::-1], color=COLORS[::-1], height=.55)
    for i, v in enumerate(vals[::-1]):
        axes[0].text(v + .35, i, f"{v:.2f}%", va="center", fontsize=8)
    axes[0].set(xlim=(0, 27), xlabel="Final greedy accuracy (%)")
    for i, n in enumerate(NAMES[1:]):
        d = SUMMARY["final"][n]["paired_vs_baseline"]
        y = 2 - i
        axes[1].errorbar(d["delta_pp"], y,
                         xerr=[[d["delta_pp"] - d["ci_low_pp"]], [d["ci_high_pp"] - d["delta_pp"]]],
                         color=COLORS[i + 1], fmt="o", capsize=4, markersize=5)
        axes[1].text(3.12, y, f"{d['delta_pp']:+.2f}", va="center", fontsize=8)
    axes[1].axvline(0, color="#666666", ls="--", lw=1)
    axes[1].set(yticks=[2, 1, 0], yticklabels=LABELS[1:], ylim=(-.6, 2.6),
                xlabel="Accuracy difference vs correctness only (pp)", xlim=(-3.1, 4))
    axes[1].set_xticks([-3, -2, -1, 0, 1, 2, 3])
    for ax in axes:
        ax.grid(axis="x", alpha=.15)
    fig.tight_layout(w_pad=1.8)
    save(fig, "fig4_5_final_accuracy")

    cats = ["Correct", "No extracted answer", "Reference intermediate", "Other wrong answer"]
    palette = ["#4D8D6C", "#C9CDD1", "#D6A14C", "#A993B6"]
    fig, ax = plt.subplots(figsize=(7.5, 2.65))
    left = np.zeros(4)
    for cat, color in zip(cats, palette):
        values = np.array([SUMMARY["final"][n]["categories"].get(cat, 0) / 1319 for n in NAMES])
        ax.barh(np.arange(4), values, left=left, color=color, label=cat, height=.58)
        for i, v in enumerate(values):
            if v > .045:
                ax.text(left[i] + v / 2, i, f"{v*100:.1f}", va="center", ha="center", fontsize=8)
        left += values
    ax.set(yticks=range(4), yticklabels=LABELS, xlabel="Percentage of all 1319 test questions", xlim=(0, 1))
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.2), ncol=2, frameon=False)
    save(fig, "fig4_6_errors")

    bands = pd.read_csv(HERE / "results/accuracy_by_steps.csv", dtype={"reference_steps": str})
    fig, ax = plt.subplots(figsize=(7.3, 2.9))
    x = np.arange(4)
    for i, (name, label, color) in enumerate(zip(NAMES, LABELS, COLORS)):
        frame = bands[bands.name == name]
        ax.bar(x + (i - 1.5) * .19, 100 * frame.accuracy.to_numpy(), width=.18, label=label, color=color)
    counts = bands[bands.name == "baseline"].n.tolist()
    ax.set_xticks(x, [f"{b} calculations\n(n={n})" for b, n in zip(["0-1", "2", "3", "4+"], counts)])
    ax.set_ylabel("Final greedy accuracy (%)")
    ax.set_ylim(0, 48)
    ax.legend(loc="upper center", ncol=4, frameon=False)
    ax.grid(axis="y", alpha=.15)
    save(fig, "fig4_7_complexity")
    print("Saved seven figures to", OUT)


if __name__ == "__main__":
    main()
