"""
A3 Part 3: table and figure comparing Diff V2 and looped-l2-embed across GPUs, from the runs' logs and result JSONs.
    python part3.py    # reads out/part3/, writes part3_table.json and figures/fig5_part3_hardware.png
Run with --fetch first time to download them from the Modal volume.
"""
import json
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "part3")
MODAL = r"C:/Users/tjsin/Documents/nanochat experiment/.venv-modal/Scripts/modal.exe"
TOKENS = 2520 * 524288
FIG = os.path.join(HERE, "figures", "fig5_part3_hardware.png")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# price per GPU-hour on Modal, rated peak in nanochat's table (None: not listed), measured dense bf16 peak (peak_bench.py)
HARDWARE = {
    "H100": dict(suffix="", price=3.95, rated=989, measured=779.5),
    "A100": dict(suffix="-a100", price=2.50, rated=312, measured=271.4),
    "L40S": dict(suffix="-l40s", price=1.95, rated=362, measured=254.0),
    "B200": dict(suffix="-b200", price=6.25, rated=2250, measured=None),
    "RTX PRO 6000": dict(suffix="-rtx", price=3.03, rated=None, measured=418.5),
}
MODELS = {"Diff V2": "d12-diff-v2", "looped-l2": "d12-looped-l2-embed"}


def fetch():
    os.makedirs(OUT, exist_ok=True)
    env = dict(os.environ, MSYS_NO_PATHCONV="1", PYTHONUTF8="1")
    for base in MODELS.values():
        for hw in HARDWARE.values():
            tag = base + hw["suffix"]
            for remote, local in ((f"/a3/results/{tag}.json", f"{tag}.json"), (f"/a3/logs/{tag}.log", f"{tag}.log")):
                subprocess.run([MODAL, "volume", "get", "--force", "vocalocate-a3", remote, os.path.join(OUT, local)],
                               env=env, capture_output=True)


def rows():
    table = []
    for model, base in MODELS.items():
        for name, hw in HARDWARE.items():
            tag = base + hw["suffix"]
            path = os.path.join(OUT, f"{tag}.json")
            if not os.path.exists(path):
                continue
            r = json.load(open(path, encoding="utf-8"))
            if r.get("returncode") != 0:
                continue
            achieved = r["tok_per_sec"] * r["flops_per_token"] / 1e12
            hours = r["train_minutes"] / 60
            log = open(os.path.join(OUT, f"{tag}.log"), encoding="utf-8", errors="replace").read()
            curve = [(int(s), float(b)) for s, b in re.findall(r"Step (\d+) \| Validation bpb: ([\d.]+)", log)]
            device_batch = re.findall(r"--device-batch-size=(\d+)", " ".join(r["cmd"]))[-1]
            table.append({
                "model": model, "hardware": name, "tag": tag, "gpu": r.get("gpu", "NVIDIA H100 80GB HBM3"), "fa3": r["fa3"],
                "device_batch": int(device_batch), "attempts": r.get("attempts", 1),
                "train_min": r["train_minutes"], "container_min": r["container_s"] / 60,
                "tok_per_sec": r["tok_per_sec"], "achieved_tflops": achieved,
                "mfu_rated": achieved / hw["rated"] * 100 if hw["rated"] else None,
                "mfu_measured": achieved / hw["measured"] * 100 if hw["measured"] else None,
                "train_cost": hw["price"] * hours, "container_cost": hw["price"] * r["container_s"] / 3600,
                "cost_per_1b": hw["price"] * hours / (TOKENS / 1e9), "peak_gb": r["peak_mem_mib"] / 1024,
                "val_bpb": r["val_bpb"], "core": r["core"], "curve": curve,
            })
    return table


def figure(table):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "figure.dpi": 150, "font.size": 9,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
    })
    color = dict(zip(HARDWARE, SERIES))
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), constrained_layout=True)
    for row, model in enumerate(MODELS):
        data = [t for t in table if t["model"] == model]
        for t in data:
            kept = [(s, b) for s, b in t["curve"] if s >= 500]
            fraction = [s / 2520 for s, _ in kept]
            bpb = [b for _, b in kept]
            axes[row, 0].plot([f * t["train_min"] / 60 for f in fraction], bpb, color=color[t["hardware"]], label=t["hardware"])
            axes[row, 1].plot([f * t["train_cost"] for f in fraction], bpb, color=color[t["hardware"]], label=t["hardware"])
        ax = axes[row, 2]
        names = [t["hardware"] for t in data]
        bars = ax.bar(names, [t["achieved_tflops"] for t in data], color=[color[n] for n in names], width=0.62)
        for bar, t in zip(bars, data):
            ax.annotate(f"{t['achieved_tflops']:.0f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", color=INK, fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_ylim(0, 480)
        ax.set_ylabel("achieved bf16 TFLOP/s")
        ax.set_title(f"{model}: achieved throughput")
        axes[row, 0].set_xlabel("training time (hours)")
        axes[row, 1].set_xlabel("training cost at list price ($)")
        for ax, what in ((axes[row, 0], "training time"), (axes[row, 1], "cost")):
            ax.set_ylabel("val bpb (lower is better)")
            ax.set_title(f"{model}: val bpb vs. {what}")
            ax.legend(fontsize=8)
    fig.savefig(FIG)


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    table = rows()
    json.dump(table, open(os.path.join(HERE, "part3_table.json"), "w", encoding="utf-8"), indent=1)
    figure(table)
    for t in table:
        print(f"{t['model']:10} {t['hardware']:13} batch {t['device_batch']:2} fa3 {str(t['fa3']):5} {t['train_min']:6.1f} min "
              f"{t['container_min']:6.1f} ctr  {t['tok_per_sec'] / 1e3:5.0f}k  {t['achieved_tflops']:4.0f} TF "
              f"rated {t['mfu_rated'] or 0:5.1f}% meas {t['mfu_measured'] or 0:5.1f}%  ${t['train_cost']:5.2f} ${t['container_cost']:5.2f} "
              f"${t['cost_per_1b']:5.2f}/1B  {t['peak_gb']:5.1f} GB  bpb {t['val_bpb']:.4f}  core {t['core']:.4f}")
