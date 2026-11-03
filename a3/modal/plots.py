"""
Figures for the A3 Part 2 write-up, built from the committed logs and result JSONs in out/.
    python plots.py    # writes figures/*.png
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIG = os.path.join(HERE, "figures")
TOKENS_PER_STEP = 524288

# categorical slots in fixed order (validated default palette), recessive ink for text/axes
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NAMES = {
    "d12-baseline": "baseline", "d12-baseline-long": "baseline-long (×1.68 tokens)", "d12-looped-l2": "looped-l2",
    "d12-diff-v2": "Diff V2", "d12-depth-attention": "Depth-Attention", "d12-mgvr": "MGVR", "d12-xiprelu": "xIPReLU",
    "d12-looped-u7": "looped-u7", "d12-looped-u9": "looped-u9", "d12-looped-l2-embed": "looped-l2-embed",
    "d12-looped-rand13": "looped-rand13",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "figure.dpi": 150,
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2,
})


def val_curve(tag):
    log = open(os.path.join(OUT, "logs", f"{tag}.log"), encoding="utf-8").read()
    return [(int(s), float(b)) for s, b in re.findall(r"Step (\d+) \| Validation bpb: ([\d.]+)", log)]


def result(tag):
    return json.load(open(os.path.join(OUT, "results", f"{tag}.json"), encoding="utf-8"))


def fig_delta_vs_baseline():
    base = dict(val_curve("d12-baseline"))
    tags = ["d12-looped-l2", "d12-diff-v2", "d12-mgvr", "d12-depth-attention", "d12-xiprelu"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axhline(0, color=INK2, lw=1)
    ax.annotate("baseline", (500, 0), xytext=(2, 4), textcoords="offset points", va="bottom", fontsize=9, color=INK2)
    for color, tag in zip(SERIES, tags):
        pts = [(s, b - base[s]) for s, b in val_curve(tag) if s >= 500 and s in base]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, label=f"{NAMES[tag]}  ({ys[-1]:+.4f})")
        ax.plot([xs[-1]], [ys[-1]], "o", ms=5, color=color, mec=SURFACE, mew=1.5, zorder=5)
    ax.set_xlim(450, 2600)
    ax.set_xlabel("training step (524,288 tokens each)")
    ax.set_ylabel("Δ val bpb vs baseline (lower is better)")
    ax.set_title("Part 2 ablations at equal tokens: gap to the baseline during training")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9, title="final Δ bpb", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_delta_vs_baseline.png"))
    plt.close(fig)


def fig_compute_matched():
    tags = ["d12-baseline", "d12-baseline-long", "d12-looped-l2", "d12-diff-v2", "d12-looped-u9", "d12-looped-u7"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for color, tag in zip(SERIES, tags):
        flops = result(tag)["flops_per_token"]
        pts = [(s * TOKENS_PER_STEP * flops / 1e18, b) for s, b in val_curve(tag) if s >= 750]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, label=f"{NAMES[tag]}  ({ys[-1]:.4f})")
        ax.plot([xs[-1]], [ys[-1]], "o", ms=5, color=color, mec=SURFACE, mew=1.5, zorder=5)
    ax.set_xlim(0.2, 1.8)
    ax.set_xlabel("training compute (EFLOPs = 10¹⁸ FLOPs)")
    ax.set_ylabel("val bpb (lower is better)")
    ax.set_title("Same compute, different spending: more tokens beat more loops")
    ax.legend(loc="upper right", fontsize=9, title="final val bpb", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_compute_matched.png"))
    plt.close(fig)


def fig_loop_sweep():
    rows = [json.load(open(f, encoding="utf-8")) for f in glob.glob(os.path.join(OUT, "sweep", "*.json"))]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for color, tag in zip(SERIES, ["d12-looped-l2", "d12-looped-rand13"]):
        pts = sorted((r["n_loops"], r["val_bpb"], r["core"]) for r in rows if r["tag"] == tag)
        n, bpb, core = zip(*pts)
        label = {"d12-looped-l2": "looped-l2 (trained at 2)", "d12-looped-rand13": "looped-rand13 (trained at 1–3)"}[tag]
        axes[0].plot(n, bpb, "-o", color=color, ms=6, mec=SURFACE, mew=1.5, label=label)
        axes[1].plot(n, core, "-o", color=color, ms=6, mec=SURFACE, mew=1.5, label=label)
    for ax in axes:
        ax.set_xticks([1, 2, 3, 4, 6])
        ax.set_xlabel("core passes at evaluation")
        ax.axvspan(1, 3, color=GRID, alpha=0.35, lw=0)
    axes[0].set_ylabel("val bpb (lower is better)")
    axes[0].set_title("Val bpb by loop count")
    axes[1].set_ylabel("CORE (higher is better)")
    axes[1].set_title("CORE by loop count")
    handles, _ = axes[0].get_legend_handles_labels()
    band = matplotlib.patches.Patch(color=GRID, alpha=0.6, label="rand13's training range (1–3)")
    axes[0].legend(handles=handles + [band], loc="upper center", fontsize=8.5, bbox_to_anchor=(0.62, 0.98))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_loop_sweep.png"))
    plt.close(fig)


def fig_cost_vs_quality():
    tags = ["d12-baseline", "d12-xiprelu", "d12-depth-attention", "d12-mgvr", "d12-diff-v2", "d12-looped-u7",
            "d12-looped-u9", "d12-looped-l2", "d12-looped-l2-embed", "d12-looped-rand13", "d12-baseline-long"]
    rows = sorted(((result(t)["container_s"] / 3600 * 3.95, result(t)["val_bpb"], NAMES[t]) for t in tags))
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    offsets = {"Depth-Attention": (-8, -12, "right"), "MGVR": (7, 4, "left"), "looped-l2": (7, 5, "left"),
               "looped-l2-embed": (-8, -3, "right")}
    for cost, bpb, name in rows:
        dx, dy, ha = offsets.get(name, (7, -3, "left"))
        ax.plot([cost], [bpb], "o", ms=8, color=SERIES[0], mec=SURFACE, mew=1.5)
        ax.annotate(name, (cost, bpb), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=INK2, ha=ha)
    ax.set_xlim(3.2, 6.9)
    ax.set_xlabel("cost per run on one H100 (USD, $3.95/hr, incl. evals)")
    ax.set_ylabel("final val bpb (lower is better)")
    ax.set_title("Cost vs quality of each d12 run")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_cost_vs_quality.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_delta_vs_baseline()
    fig_compute_matched()
    fig_loop_sweep()
    fig_cost_vs_quality()
    print("wrote", sorted(os.listdir(FIG)))
