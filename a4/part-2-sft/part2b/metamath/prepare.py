#!/usr/bin/env python3
"""Prepare auditable MetaMathQA SFT train and source-validation JSONL.

Pinned source files are downloaded separately. No GPU or Modal service is used.
Automatic screens do not establish solution correctness.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import HashingVectorizer

ROOT = Path(__file__).resolve().parent
REVISIONS = {
    "metamath": "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18",
    "humaneval": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
}
SEED = "a4-part2b-metamath-xcoder-v1"
TARGETS = {"metamath": (60000, 3000)}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.casefold())).strip()


def rank(s):
    return hashlib.sha256((SEED + ":" + s).encode()).hexdigest()


def read_meta(path):
    rows = json.loads(path.read_text())
    for i, raw in enumerate(rows):
        if not str(raw.get("type", "")).startswith("GSM_"):
            continue
        q, a, original = raw.get("query"), raw.get("response"), raw.get("original_question")
        if not all(isinstance(x, str) and x.strip() for x in (q, a, original)):
            continue
        # Keep original answer and its final-answer convention unchanged.
        if not re.search(r"(?:####\s*[-+]?\d|the answer is\s*:\s*[-+]?\d)", a, re.I):
            continue
        sid = f"metamath:{i}"
        yield {"source": "metamath", "source_id": sid, "audit_id": sid,
               "source_category": raw["type"], "source_group": norm(original),
               "original_question": original, "question": q.strip(), "answer": a.strip(),
               "messages": [{"role": "user", "content": q.strip()},
                            {"role": "assistant", "content": a.strip()}]}


def test_index(gsm_path, human_path):
    gsm = pq.read_table(gsm_path, columns=["question"]).column("question").to_pylist()
    human = pq.read_table(human_path, columns=["prompt", "canonical_solution"])
    prompts = human.column("prompt").to_pylist()
    solutions = human.column("canonical_solution").to_pylist()
    # Concatenate signature/docstring with reference solution for code-copy screening.
    code = [p + s for p, s in zip(prompts, solutions)]
    return {"gsm": gsm, "human_prompt": prompts, "human_code": code}


def write_rows(path, rows):
    with path.open("w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "screened_v1")
    ap.add_argument("--reference-index", type=Path,
                    default=ROOT.parent / "shared" / "audit_v1" / "reference_index.sqlite")
    ap.add_argument("--gsm-test", type=Path,
                    default=ROOT.parent / "shared" / "reference_raw" / "gsm8k_test" / "00000.parquet")
    ap.add_argument("--human-test", type=Path,
                    default=ROOT / "raw" / "openai_humaneval" / "test-00000-of-00001.parquet")
    args = ap.parse_args()
    raw = ROOT / "raw"
    meta_path = raw / "MetaMathQA-395K.json"
    for p in (meta_path, args.gsm_test, args.human_test,
              args.tokenizer_dir / "tokenizer.pkl", args.reference_index):
        if not p.is_file():
            ap.error(f"Missing required input: {p}")
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT.parent / "derived_nanochat"))
    from nanochat.tokenizer import RustBPETokenizer
    tokenizer = RustBPETokenizer.from_directory(str(args.tokenizer_dir))
    refs = test_index(args.gsm_test, args.human_test)
    vec = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                            n_features=2**18, alternate_sign=False, norm="l2", dtype=np.float32)
    mats = {k: vec.transform(v) for k, v in refs.items()}
    db = sqlite3.connect(f"file:{args.reference_index}?mode=ro", uri=True)
    counts = defaultdict(Counter)
    flags, rejects, accepted = [], [], {}
    seen = set()
    for source, reader, cap in (("metamath", read_meta(meta_path), 130000),):
        # Deterministic overdraw; no input-order bias in the selected candidate pool.
        pool = sorted(reader, key=lambda x: rank(x["source_id"]))[:cap]
        counts[source]["candidate_overdraw"] = len(pool)
        kept = []
        for offset in range(0, len(pool), 256):
            batch = pool[offset:offset + 256]
            qmatrix = vec.transform([r["question"] for r in batch])
            score_sets = {k: (qmatrix @ mats[k].T).toarray() for k in ("gsm", "human_prompt")}
            for j, row in enumerate(batch):
                nq = norm(row["question"])
                if not nq or nq in seen:
                    counts[source]["empty_or_duplicate"] += 1
                    continue
                seen.add(nq)
                exact = db.execute("SELECT sources FROM ref_exact WHERE q=?", (nq,)).fetchone()
                if exact:
                    counts[source]["exact_original_reference:" + exact[0]] += 1
                    # GSM8K-train lineage is intentional for MetaMath, but new
                    # data should not repeat other original SFT or test rows.
                    if exact[0] != "gsm_train" or source != "metamath":
                        counts[source]["exact_original_overlap_reject"] += 1
                        continue
                if source == "metamath":
                    orig = db.execute("SELECT sources FROM ref_exact WHERE q=?",
                                      (row["source_group"],)).fetchone()
                    if orig:
                        counts[source]["original_question_reference:" + orig[0]] += 1
                match = None
                for kind, scores in score_sets.items():
                    idx = int(scores[j].argmax())
                    if float(scores[j, idx]) >= 0.60 or nq == norm(refs[kind][idx]):
                        match = (kind, idx, float(scores[j, idx]))
                        break
                if match is not None:
                    kind, idx, score = match
                    counts[source]["test_similarity_hold"] += 1
                    flags.append({"source_id": row["source_id"], "kind": kind,
                                  "test_index": idx, "similarity": round(score, 5),
                                  "question": row["question"], "test_text": refs[kind][idx],
                                  "decision": None})
                    continue
                try:
                    ids, mask = tokenizer.render_conversation(
                        {"messages": row["messages"]}, max_tokens=1_000_000)
                    length, loss = len(ids), sum(mask)
                except Exception as exc:
                    counts[source]["render_error"] += 1
                    rejects.append({"source_id": row["source_id"], "reason": type(exc).__name__})
                    continue
                if not 0 < loss <= length <= 2048:
                    counts[source]["length_or_loss_reject"] += 1
                    continue
                row["rendered_tokens"] = length
                row["assistant_loss_tokens"] = loss
                kept.append(row)
                counts[source]["screened"] += 1
        accepted[source] = kept
        print(source, dict(counts[source]), flush=True)
    write_rows(args.out / "test_similarity_review.jsonl", flags)
    write_rows(args.out / "render_rejects.jsonl", rejects)
    outputs = {}
    selected = {}
    for source, rows in accepted.items():
        train_target, val_target = TARGETS[source]
        groups = defaultdict(list)
        for row in rows:
            groups[row.get("source_group", row["source_id"])].append(row)
        # Partition by group before taking rows, so variants cannot cross train/holdout.
        val, train = [], []
        for group in sorted(groups, key=lambda g: rank("holdout:" + g)):
            if len(val) < val_target:
                val.extend(sorted(groups[group], key=lambda r: rank(r["source_id"])))
            else:
                train.extend(groups[group])
        if len(train) < train_target:
            raise RuntimeError(f"{source}: only {len(train)} train candidates after screens; need {train_target}")
        train.sort(key=lambda r: rank("train:" + r["source_id"]))
        val = sorted(val, key=lambda r: rank("val:" + r["source_id"]))[:val_target]
        selected[source] = {"train": train[:train_target], "val": val}
        for split, subset in selected[source].items():
            p = args.out / f"{source}_{split}.jsonl"
            write_rows(p, subset)
            outputs[p.name] = {"rows": len(subset), "sha256": sha(p),
                               "rendered_tokens": sum(r["rendered_tokens"] for r in subset),
                               "assistant_loss_tokens": sum(r["assistant_loss_tokens"] for r in subset),
                               "categories": dict(Counter(r["source_category"] for r in subset))}
        # Reproducible sample for substantive human review.
        sample = sorted(rows, key=lambda r: rank("review:" + r["source_id"]))[:100]
        write_rows(args.out / f"{source}_review_sample.jsonl", sample)
    manifest = {"status": "Solution quality and test-similarity flags require review",
                "selection_seed_sha256": hashlib.sha256(SEED.encode()).hexdigest(), "revisions": REVISIONS,
                "inputs": {str(p.relative_to(ROOT.parent)): sha(p) for p in (meta_path, args.gsm_test, args.human_test)},
                "tokenizer_sha256": sha(args.tokenizer_dir / "tokenizer.pkl"),
                "reference_index": str(args.reference_index),
                "counts": {k: dict(v) for k, v in counts.items()},
                "test_similarity_flags": len(flags), "outputs": outputs}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"],
                      "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
