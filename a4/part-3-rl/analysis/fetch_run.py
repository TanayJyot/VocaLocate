"""Download an RL run's log, result json and rollout jsonl files from the Modal volume.

Usage (Modal venv python): python fetch_run.py <output-tag> <local_dir> [volume]
"""
import sys
from pathlib import Path
import modal

tag, out = sys.argv[1], Path(sys.argv[2])
vol = modal.Volume.from_name(sys.argv[3] if len(sys.argv) > 3 else "vocalocate-a3")
(out / "rollouts").mkdir(parents=True, exist_ok=True)
(out / "logs").mkdir(exist_ok=True)
items = [(f"a3/logs/{tag}.log", out / "logs" / f"{tag}.log"),
         (f"a3/results/{tag}.json", out / "logs" / f"{tag}.result.json")]
items += [(f"a3/rl_runs/{tag}/rollouts/rank{r}.jsonl", out / "rollouts" / f"rank{r}.jsonl") for r in range(8)]
try:
    (out / "final_eval").mkdir(exist_ok=True)
    items += [(e.path, out / "final_eval" / Path(e.path).name)
              for e in vol.listdir(f"a3/rl_runs/{tag}/final_eval") if e.type == modal.volume.FileEntryType.FILE]
except Exception:
    pass
for src, dst in items:
    try:
        with open(dst, "wb") as f:
            for chunk in vol.read_file(src):
                f.write(chunk)
        print(f"{src} -> {dst} ({dst.stat().st_size:,} B)")
    except Exception as e:
        dst.unlink(missing_ok=True)
        print(f"skip {src}: {type(e).__name__}")
