"""Command-line entry point: `vocalocate-demo <command>`.

    run       everything, on synthetic data: pipeline, baseline, both mining approaches, evaluation, search
    generate  write the synthetic dataset only
    train     baseline + mining + fine-tuning + evaluation on a source already in the lake
    search    index a folder and search it with a recorded imitation
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from vocalocate_pipeline.config import load_config

from .mining import MiningConfig
from .train import TrainConfig

MODELS = ("baseline", "random", "retrieval", "metadata")


def _train_args(p: argparse.ArgumentParser) -> None:
    d = TrainConfig()
    p.add_argument("--epochs", type=int, default=d.epochs, help="baseline epochs")
    p.add_argument("--ft-epochs", type=int, default=d.ft_epochs, help="fine-tuning epochs per strategy")
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--frames", type=int, default=d.frames, help="log-mel frames per clip (10 ms each)")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    m = MiningConfig()
    p.add_argument("--top-k", type=int, default=m.top_k, help="retrieval-driven: candidates per query")
    p.add_argument("--band", type=float, default=m.band, help="semi-hard band below the positive score")
    p.add_argument("--n-neg", type=int, default=m.n_neg, help="negatives kept per imitation")


def _configs(args) -> tuple[TrainConfig, MiningConfig]:
    tcfg = TrainConfig(
        epochs=args.epochs, ft_epochs=args.ft_epochs, batch_size=args.batch_size, frames=args.frames,
        device=args.device,
    )
    return tcfg, MiningConfig(top_k=args.top_k, band=args.band, n_neg=args.n_neg)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vocalocate-demo", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, help="pipeline TOML (default: demo/configs/demo.toml if present)")
    p.add_argument("--lake", type=Path, help="lake root; overrides the config file")
    p.add_argument("--workers", type=int, help="pipeline worker processes")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="full demo on synthetic data")
    r.add_argument("--workdir", type=Path, default=Path("demo_out"))
    r.add_argument("--small", action="store_true", help="tiny dataset, for a quick check")
    _train_args(r)

    g = sub.add_parser("generate", help="write the synthetic dataset")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--small", action="store_true")

    t = sub.add_parser("train", help="train and compare all models on a source in the lake")
    t.add_argument("--source", default="sfx_meta", help="pipeline source with imitation/reference pairs")
    _train_args(t)

    s = sub.add_parser("search", help="search a folder of SFX with a vocal imitation")
    s.add_argument("--root", type=Path, action="append", required=True, help="library folder (repeatable)")
    s.add_argument("--name", default="default", help="library name")
    s.add_argument("--query", type=Path, action="append", required=True, help="imitation WAV (repeatable)")
    s.add_argument("--model", choices=MODELS + ("all",), default="all")
    s.add_argument("--source", default="sfx_meta", help="which trained models to use (lake/models/<source>)")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--json", action="store_true", help="print JSON instead of a table")
    return p


def _default_config(args) -> Path | None:
    if args.config:
        return args.config
    here = Path(__file__).resolve().parents[2] / "configs" / "demo.toml"
    return here if here.exists() else None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(message)s")
    lake = args.lake
    if args.command == "run" and lake is None:
        lake = args.workdir / "lake"
    cfg = load_config(_default_config(args), lake_root=lake, workers=args.workers)

    if args.command == "generate":
        from .synth import generate

        kwargs = dict(recordings_per_category=4, distractors_per_category=1, imitations_per_reference=4)
        print(generate(args.out, **(kwargs if args.small else {})))
    elif args.command == "run":
        from .demo_flow import run_demo

        tcfg, mcfg = _configs(args)
        if args.small and args.epochs == TrainConfig().epochs:
            tcfg.epochs, tcfg.ft_epochs = 6, 3
        run_demo(args.workdir, cfg, tcfg, mcfg, small=args.small)
        print((args.workdir / "report.md").read_text())
    elif args.command == "train":
        from .experiment import run_experiment
        from .report import write_report

        tcfg, mcfg = _configs(args)
        results = run_experiment(cfg, args.source, tcfg, mcfg)
        out = cfg.lake_root / "models" / args.source / "report.md"
        write_report(results, out)
        print(out.read_text())
    elif args.command == "search":
        from . import train as T
        from .experiment import model_dir
        from .search import embed_query, index_library, search

        names = [n for n in MODELS if n != "random"] if args.model == "all" else [args.model]
        report, scanned = {}, False
        for name in names:
            ckpt_path = model_dir(cfg, args.source) / f"{name}.pt"
            if not ckpt_path.exists():
                raise SystemExit(f"no trained model at {ckpt_path}; run `vocalocate-demo run` or `train` first")
            model, tcfg, ckpt = T.load(ckpt_path)
            idx = index_library(cfg, args.root, args.name, model, tcfg, ckpt["model_version"], scan=not scanned)
            scanned = True
            report[name] = {
                str(q): search(idx, embed_query(q, model, tcfg, cfg), args.k) for q in args.query
            }
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            for q in args.query:
                print(f"\nQuery: {q}")
                for name in names:
                    print(f"  [{name}]")
                    for hit in report[name][str(q)]:
                        print(f"    {hit['rank']}. {hit['score']:.3f}  {hit['path']}  ({hit['project'] or '-'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
