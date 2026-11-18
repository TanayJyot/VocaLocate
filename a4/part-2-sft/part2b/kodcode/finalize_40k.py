"""Finish the 40K split using the completed audit's held-ID decisions.

The first full 15-shard audit screened every candidate against the original SFT,
HumanEval, and tokenizer. This pass removes named-function prompts that already
contain a function body and fills the vacant training slots from audited rows
that were not selected in the first draft. It does not weaken any prior screen.
"""

import argparse
import ast
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

import pyarrow.parquet as pq

def norm(s): return re.sub(r'\s+',' ',re.sub(r'[^\w\s]',' ',s.casefold())).strip()
def rank(s): return hashlib.sha256((SEED+':'+s).encode()).hexdigest()
def code_of(s):
 # Match Nanochat's HumanEval extractor exactly: only newline-delimited first fence.
 fs=re.findall(r'```(?:python)?\s*\n(.*?)\n```',s,re.DOTALL)
 return fs[0] if len(fs)<=1 and fs else (s if not fs else '')
def prompt_contains_solution_body(question, function_name):
 try: tree=ast.parse(question)
 except (SyntaxError,ValueError): return False
 for function in tree.body:
  if not isinstance(function,(ast.FunctionDef,ast.AsyncFunctionDef)) or function.name!=function_name: continue
  body=function.body
  if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str): body=body[1:]
  return any(not isinstance(node,ast.Pass) and not (isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and node.value.value is Ellipsis) for node in body)
 return False
NAME_PATTERNS=(
 re.compile(r'\bdef\s+([A-Za-z_]\w*)\s*\('),
 re.compile(r'\bfunction\s+(?:named|called)\s+[`\'\"]?([A-Za-z_]\w*)',re.I),
 re.compile(r'\bimplement\s+(?:the\s+)?[`\'\"]?([A-Za-z_]\w*)\s*\(',re.I),
)
def save(p, rows):
 with open(p,'w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')



ROOT = Path(__file__).resolve().parent
DRAFT = ROOT / "screened_40k_v2"
OLD = ROOT / "screened_v1"
SEED = "a4-kodcode-fifteen-shards-40k-v2"


def read_rows(path):
    return [json.loads(line) for line in path.open()]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((DRAFT / "kodcode_manifest.json").read_text())
    if manifest["seed"] != SEED or manifest["automatically_screened"] != 42243:
        raise RuntimeError("Unexpected draft audit; run the complete preparation instead")
    expected_draft = manifest.get("incremental_quality_repair", {}).get("draft_train_sha256")
    if expected_draft and digest(DRAFT / "kodcode_train.jsonl") != expected_draft:
        raise RuntimeError("Restore the pre-repair draft train JSONL before finalizing")
    if digest(args.tokenizer_dir / "tokenizer.pkl") != manifest["tokenizer_sha256"]:
        raise RuntimeError("Tokenizer differs from the audited tokenizer")
    draft_train = read_rows(DRAFT / "kodcode_train.jsonl")
    draft_val = read_rows(DRAFT / "kodcode_val.jsonl")
    old_train = read_rows(OLD / "kodcode_train.jsonl")
    old_val = read_rows(OLD / "kodcode_val.jsonl")
    if draft_train[:20000] != old_train or draft_val != old_val:
        raise RuntimeError("The draft did not preserve the original 20K/1K split")
    selected = {r["audit_id"] for r in draft_train + draft_val}
    held = {r["source_id"] for r in read_rows(DRAFT / "kodcode_similarity_review.jsonl")}
    val_groups = {r["source_group"] for r in old_val}
    kept_new = [r for r in draft_train[20000:] if not (
        r.get("request_shape") == "named_function"
        and prompt_contains_solution_body(r["question"],
            next(n.name for n in ast.parse(r["extracted_code"]).body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))))
    )]
    removed = 20000 - len(kept_new)
    sys.path.insert(0, str(ROOT.parent / "derived_nanochat"))
    from nanochat.tokenizer import RustBPETokenizer
    tokenizer = RustBPETokenizer.from_directory(str(args.tokenizer_dir))
    db = sqlite3.connect(f"file:{ROOT.parent / 'shared/audit_v1/reference_index.sqlite'}?mode=ro", uri=True)
    exact = {x[0] for x in db.execute("select q from ref_exact")}
    seen = set()
    action = re.compile(r"\b(?:write|create|implement|define|design|develop|build|construct|complete|code|generate)\b.{0,120}\b(?:python\s+)?function\b|\bfunction\b.{0,120}\b(?:write|create|implement|define|design|develop|build|construct|complete|code|generate)\b", re.I | re.S)
    io = re.compile(r"\b(?:input|takes?|accepts?|return|output|given)\b", re.I)
    extra = []
    solved_holds = 0
    global_row = 0
    for shard in range(15):
        path = Path(f"/private/tmp/kodcode_shard{shard}_humaneval_audit.parquet")
        if digest(path) != manifest["inputs"][str(path)]:
            raise RuntimeError(f"Source shard changed: {path}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=[
            "question_id", "question", "solution", "test", "benchmark_similarity"]):
            for raw in batch.to_pylist():
                global_row += 1
                q, answer = raw["question"], raw["solution"]
                if not isinstance(q, str) or not isinstance(answer, str) or not q.strip() or not answer.strip():
                    continue
                code = code_of(answer)
                try:
                    tree = ast.parse(code)
                except (SyntaxError, ValueError):
                    continue
                functions = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                if not code or len(functions) != 1 or not io.search(q):
                    continue
                explicit = bool(action.search(q))
                named = any((m := pat.search(q)) and m.group(1) == functions[0]
                            for pat in NAME_PATTERNS)
                if not (explicit or named):
                    continue
                if not explicit and prompt_contains_solution_body(q, functions[0]):
                    solved_holds += 1
                    continue
                nq = norm(q)
                if nq in seen:
                    continue
                seen.add(nq)
                if nq in exact or not isinstance(raw["test"], str) or not raw["test"].strip():
                    continue
                try:
                    similarity = float(raw["benchmark_similarity"])
                except (TypeError, ValueError):
                    continue
                if similarity >= 0.60:
                    continue
                source_id = f"kodcode:{shard}:{global_row}"
                if source_id in selected or source_id in held:
                    continue
                group = re.sub(r"_[A-Z]$", "", str(raw.get("question_id") or nq))
                if group in val_groups:
                    continue
                messages = [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]
                ids, mask = tokenizer.render_conversation({"messages": messages}, max_tokens=1000000)
                if not 0 < sum(mask) <= len(ids) <= 2048:
                    continue
                extra.append({"source": "kodcode", "source_id": source_id,
                              "audit_id": source_id, "source_group": group,
                              "question": q, "answer": answer, "extracted_code": code,
                              "messages": messages, "benchmark_similarity": similarity,
                              "shard": shard, "request_shape": "explicit" if explicit else "named_function",
                              "rendered_tokens": len(ids), "assistant_loss_tokens": sum(mask)})
    if global_row != manifest["counts"]["raw"]:
        raise RuntimeError("Source row count changed")
    candidates = kept_new + extra
    candidates.sort(key=lambda r: rank("new_train:" + r["source_id"]))
    if len(candidates) < 20000:
        raise RuntimeError(f"Only {len(candidates)} valid new candidates for 20K slots")
    final_train = old_train + candidates[:20000]
    if len({r["audit_id"] for r in final_train}) != 40000:
        raise RuntimeError("Duplicate training ID")
    if {r["source_group"] for r in final_train} & val_groups:
        raise RuntimeError("Training/validation group overlap")
    if {norm(r["question"]) for r in final_train} & {
            norm(r["question"]) for r in old_val}:
        raise RuntimeError("Training/validation prompt overlap")
    draft_hash = digest(DRAFT / "kodcode_train.jsonl")
    save(DRAFT / "kodcode_train.jsonl", final_train)
    save(DRAFT / "kodcode_val.jsonl", old_val)
    save(DRAFT / "kodcode_named_only_review_sample.jsonl",
                 sorted((r for r in final_train[20000:] if r["request_shape"] == "named_function"),
                        key=lambda r: rank("named_review:" + r["source_id"]))[:200])
    manifest["status"] = "UNAPPROVED: original-SFT near retrieval is approximate; solution quality needs review"
    manifest["automatically_screened_after_quality_filter"] = 21000 + len(candidates)
    manifest["incremental_quality_repair"] = {
        "draft_train_sha256": draft_hash,
        "excluded_selected_named_prompts_with_solution_body": removed,
        "all_source_named_prompts_with_solution_body": solved_holds,
        "audited_unselected_replacements_checked": len(extra),
        "final_new_candidate_pool": len(candidates),
        "method": "Reuse full-audit overlap holds; rescan source rows and tokenizer-check unselected candidates",
    }
    manifest["counts"]["final_train"] = 40000
    manifest["counts"]["final_val"] = 1000
    manifest["outputs"]["kodcode_train.jsonl"] = {"rows": 40000, "sha256": digest(DRAFT / "kodcode_train.jsonl")}
    manifest["outputs"]["kodcode_val.jsonl"] = {"rows": 1000, "sha256": digest(DRAFT / "kodcode_val.jsonl")}
    for name in ("manifest.json", "kodcode_manifest.json"):
        (DRAFT / name).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"train": 40000, "val": 1000, "removed": removed,
                      "replacements_screened": len(extra), "candidate_pool": len(candidates),
                      "train_sha256": manifest["outputs"]["kodcode_train.jsonl"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
