"""Validated, deterministic conversation task for the screened Part 2b JSONL files."""

import hashlib
import json
from pathlib import Path

import numpy as np

from tasks.common import Task


class ScreenedJSONL(Task):
    def __init__(self, path, source, **kwargs):
        super().__init__(**kwargs)
        self.path = Path(path).resolve(strict=True)
        if source not in {"metamath", "kodcode"}:
            raise ValueError(f"Unknown source: {source}")
        self.source = source
        if self.path.name != f"{source}_train.jsonl":
            raise ValueError("Only source-matched audited train JSONL files are supported")
        manifest_path = self.path.with_name("kodcode_manifest.json" if source == "kodcode" else "manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("outputs", {}).get(self.path.name)
        if not isinstance(expected, dict) or not isinstance(expected.get("rows"), int) or not isinstance(expected.get("sha256"), str):
            raise ValueError(f"Missing output metadata for {self.path.name}")
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if digest != expected["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {self.path}")
        records = []
        seen_ids = set()
        with self.path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {self.path}:{line_number}") from exc
                if row.get("source") != source:
                    raise ValueError(f"Wrong source at {self.path}:{line_number}")
                audit_id = row.get("audit_id")
                if not isinstance(audit_id, str) or not audit_id or audit_id in seen_ids:
                    raise ValueError(f"Missing or duplicate audit_id at {self.path}:{line_number}")
                seen_ids.add(audit_id)
                messages = row.get("messages")
                if not isinstance(messages, list) or len(messages) != 2:
                    raise ValueError(f"Expected user/assistant messages at {self.path}:{line_number}")
                for message, role in zip(messages, ("user", "assistant")):
                    if not isinstance(message, dict) or message.get("role") != role or not isinstance(message.get("content"), str) or not message["content"]:
                        raise ValueError(f"Invalid {role} message at {self.path}:{line_number}")
                if messages[0]["content"] != row.get("question") or messages[1]["content"] != row.get("answer"):
                    raise ValueError(f"Messages differ from source question/answer at {self.path}:{line_number}")
                if not isinstance(row.get("rendered_tokens"), int) or not isinstance(row.get("assistant_loss_tokens"), int):
                    raise ValueError(f"Missing token counts at {self.path}:{line_number}")
                if not 0 < row["assistant_loss_tokens"] <= row["rendered_tokens"] <= 2048:
                    raise ValueError(f"Invalid sequence length at {self.path}:{line_number}")
                records.append(messages)
        if len(records) != expected["rows"]:
            raise ValueError(f"Row count mismatch: expected {expected['rows']}, got {len(records)}")
        permutation = np.random.default_rng(42).permutation(len(records))
        self.records = [records[int(i)] for i in permutation]
        self.sha256 = digest

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.records)

    def get_example(self, index):
        return {"messages": self.records[index]}
