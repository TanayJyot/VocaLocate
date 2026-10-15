"""Command-line entry point: `vocalocate-demo <command>`.

    run       everything, on synthetic data: pipeline, baseline, both mining approaches, evaluation, search
    generate  write the synthetic dataset only
    train     baseline + mining + fine-tuning + evaluation on a source already in the lake
    search    index a folder and search it with a recorded imitation
    summarize average results over several seeds
    fetch-models  download the pretrained MobileNetV3 and CLAP weights
    benchmark score trained models on the QVIM DEV set (or any other source's test split)
    app       the VocaLocate web app: index folders, record an imitation, search, preview, drag out
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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
    p.add_argument("--seed", type=int, default=d.seed, help="training and mining seed")
    p.add_argument("--encoder", choices=("cnn", "mobilenetv3"), default=d.encoder,
                   help="cnn: small CNN from scratch; mobilenetv3: AudioSet-pretrained EfficientAT mn10_as")
    p.add_argument("--pretrained", default=d.pretrained,
                   help="EfficientAT weights for --encoder mobilenetv3 ('none' = random init)")
    p.add_argument("--lr", type=float, help="baseline learning rate (default: 1e-3 cnn, 3e-4 mobilenetv3)")
    p.add_argument("--ft-lr", type=float, help="fine-tuning learning rate (default: 1e-4 cnn, 5e-5 mobilenetv3)")
    m = MiningConfig()
    p.add_argument("--top-k", type=int, default=m.top_k, help="retrieval-driven: candidates per query")
    p.add_argument("--band", type=float, default=m.band, help="semi-hard band below the positive score")
    p.add_argument("--n-neg", type=int, default=m.n_neg, help="negatives kept per imitation")
    p.add_argument("--sweep", action="store_true", help="tune band / near-duplicate cutoff per strategy on validation")
    _clap_args(p)
    p.add_argument("--text-branch", action="store_true",
                   help="also score text hints with CLAP and fuse them with each voice model (α tuned on validation)")


def _clap_args(p: argparse.ArgumentParser) -> None:
    from .text import DEFAULT_CLAP

    p.add_argument("--clap-model", default=DEFAULT_CLAP, help="Hugging Face CLAP model for the text branch")
    p.add_argument("--clap-backend", choices=("hf", "fake"), default="hf", help="fake: offline stand-in for tests")


def _clap(args):
    from .text import ClapScorer

    return ClapScorer(args.clap_model, device=getattr(args, "device", "auto"), backend=args.clap_backend)


def _configs(args) -> tuple[TrainConfig, MiningConfig]:
    tcfg = TrainConfig(
        epochs=args.epochs, ft_epochs=args.ft_epochs, batch_size=args.batch_size, frames=args.frames,
        device=args.device, seed=args.seed, encoder=args.encoder,
        pretrained=None if str(args.pretrained).lower() == "none" else args.pretrained,
    )
    mobilenet = args.encoder == "mobilenetv3"
    tcfg.lr = args.lr if args.lr is not None else (3e-4 if mobilenet else tcfg.lr)
    tcfg.ft_lr = args.ft_lr if args.ft_lr is not None else (5e-5 if mobilenet else tcfg.ft_lr)
    return tcfg, MiningConfig(top_k=args.top_k, band=args.band, n_neg=args.n_neg, seed=args.seed)


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

    bm = sub.add_parser("benchmark", help="score trained models on another source's test split, e.g. the QVIM DEV set")
    bm.add_argument("--source", default="sfx_meta", help="the source the models were trained on")
    bm.add_argument("--on", default="qvim_dev", help="benchmark source already in the lake")
    bm.add_argument("--device", default="auto")

    fm = sub.add_parser("fetch-models", help="download the pretrained MobileNetV3 (EfficientAT) and CLAP weights")
    fm.add_argument("--pretrained", default=TrainConfig().pretrained, help="EfficientAT model name")
    _clap_args(fm)

    ap = sub.add_parser("app", help="start the VocaLocate web app (record, search, preview, drag out)")
    ap.add_argument("--workdir", type=Path, help="a demo run folder: uses its lake, indexes its library/, offers its queries/")
    ap.add_argument("--source", default="sfx_meta", help="which trained models to use (lake/models/<source>)")
    ap.add_argument("--root", type=Path, action="append", help="library folder to index at start (repeatable)")
    ap.add_argument("--name", default="default", help="library name for --root")
    ap.add_argument("--text", action="store_true", help="enable CLAP text hints")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    _clap_args(ap)

    sm = sub.add_parser("summarize", help="average several runs' results.json (mean ± std over seeds)")
    sm.add_argument("runs", type=Path, nargs="+", help="workdirs or results.json files")
    sm.add_argument("--out", type=Path, help="also write the table to this file")

    s = sub.add_parser("search", help="search a folder of SFX with a vocal imitation")
    s.add_argument("--root", type=Path, action="append", required=True, help="library folder (repeatable)")
    s.add_argument("--name", default="default", help="library name")
    s.add_argument("--query", type=Path, action="append", required=True, help="imitation WAV (repeatable)")
    s.add_argument("--model", choices=MODELS + ("all",), default="all")
    s.add_argument("--source", default="sfx_meta", help="which trained models to use (lake/models/<source>)")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--json", action="store_true", help="print JSON instead of a table")
    s.add_argument("--text", help='optional text hint, e.g. "metallic short" (uses CLAP; α from training)')
    s.add_argument("--alpha", type=float, help="voice weight for --text (default: the α tuned on validation)")
    _clap_args(s)
    return p


def _default_config(args) -> Path | None:
    if args.config:
        return args.config
    here = Path(__file__).resolve().parents[2] / "configs" / "demo.toml"
    return here if here.exists() else None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Reports contain characters like "−" and "Δ"; Windows consoles and redirects default to cp1252.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(message)s")
    lake = args.lake
    if args.command in ("run", "app") and lake is None and getattr(args, "workdir", None):
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
        clap = _clap(args) if args.text_branch else None
        run_demo(args.workdir, cfg, tcfg, mcfg, small=args.small, sweep=args.sweep, clap=clap)
        print((args.workdir / "report.md").read_text(encoding="utf-8"))
    elif args.command == "train":
        from .experiment import run_experiment
        from .report import write_report

        tcfg, mcfg = _configs(args)
        clap = _clap(args) if args.text_branch else None
        results = run_experiment(cfg, args.source, tcfg, mcfg, sweep=args.sweep, clap=clap)
        out = cfg.lake_root / "models" / args.source / "report.md"
        write_report(results, out)
        print(out.read_text(encoding="utf-8"))
    elif args.command == "app":
        from .app import AppState, serve

        examples = args.workdir / "queries" if args.workdir else None
        state = AppState(cfg, args.source, clap=_clap(args) if args.text else None, examples=examples)
        if args.workdir and (args.workdir / "library").is_dir():
            state.add_library("demo", [str(args.workdir / "library")])
        if args.root:
            state.add_library(args.name, [str(r) for r in args.root])
        server = serve(state, args.host, args.port)
        print(f"VocaLocate is running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    elif args.command == "benchmark":
        from .experiment import MODEL_NAMES, run_benchmark

        out = run_benchmark(cfg, args.source, args.on, device=args.device)
        print(f"Models trained on {args.source}, scored on the {args.on} test split (each imitation ranks every reference)")
        print(f"{'model':<55} {'MRR':>6} {'R@1':>6} {'R@5':>6} {'n':>5}")
        for name, m in out["models"].items():
            print(f"{MODEL_NAMES[name]:<55} {m['mrr']:>6.3f} {m['recall@1']:>6.3f} {m['recall@5']:>6.3f} {m['n_queries']:>5}")
        for label, mrr in out.get("published_mrr", {}).items():
            print(f"{'published: ' + label:<55} {mrr:>6.3f}")
    elif args.command == "fetch-models":
        import torch

        from .model import MobileNetEncoder
        from .text import ClapScorer

        enc = MobileNetEncoder(pretrained=args.pretrained)
        print(f"MobileNetV3 {args.pretrained}: {sum(p.numel() for p in enc.parameters()) / 1e6:.1f}M parameters, "
              f"embedding {tuple(enc(torch.zeros(1, 128, 256)).shape)}")
        clap = ClapScorer(args.clap_model, backend="hf", device="cpu")
        t = clap.embed_text(["metallic", "a dog barking"])
        print(f"CLAP {args.clap_model}: text embedding {t.shape}, cos(metallic, dog) = {float(t[0] @ t[1]):.3f}")
    elif args.command == "summarize":
        from .summarize import summarize

        text = summarize(args.runs)
        if args.out:
            args.out.write_text(text, encoding="utf-8")
        print(text)
    elif args.command == "search":
        from . import train as T
        from .experiment import model_dir
        from .search import embed_query, index_library, search, text_scores

        names = [n for n in MODELS if n != "random"] if args.model == "all" else [args.model]
        clap = _clap(args) if args.text else None
        fusion_path = model_dir(cfg, args.source) / "fusion.json"
        alphas = json.loads(fusion_path.read_text(encoding="utf-8"))["alpha"] if fusion_path.exists() else {}
        report, scanned = {}, False
        for name in names:
            ckpt_path = model_dir(cfg, args.source) / f"{name}.pt"
            if not ckpt_path.exists():
                raise SystemExit(f"no trained model at {ckpt_path}; run `vocalocate-demo run` or `train` first")
            model, tcfg, ckpt = T.load(ckpt_path)
            idx = index_library(cfg, args.root, args.name, model, tcfg, ckpt["model_version"], scan=not scanned)
            scanned = True
            text = text_scores(cfg, args.name, idx, args.text, clap) if clap else None
            alpha = args.alpha if args.alpha is not None else alphas.get(name, 0.5)
            report[name] = {
                str(q): search(idx, embed_query(q, model, tcfg, cfg), args.k, text=text, alpha=alpha)
                for q in args.query
            }
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            for q in args.query:
                print(f"\nQuery: {q}" + (f'  + text "{args.text}"' if args.text else ""))
                for name in names:
                    print(f"  [{name}]")
                    for hit in report[name][str(q)]:
                        print(f"    {hit['rank']}. {hit['score']:.3f}  {hit['path']}  ({hit['project'] or '-'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
