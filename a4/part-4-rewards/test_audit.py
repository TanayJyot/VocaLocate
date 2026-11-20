"""Data-split and provenance checks for the offline replay entry point."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_rewards


class TrainingOnlyAuditTests(unittest.TestCase):
    def run_fixture(self, directory, *, sample_count=16, logged_reference="8"):
        root = Path(directory)
        data = root / "train.jsonl"
        data.write_text((json.dumps({"question": "Training fixture only",
                                     "answer": "3*2=<<3*2=6>>6; 6+2=<<6+2=8>>8\n#### 8"})
                         + "\n") * 7473, encoding="utf-8")
        logs = root / "rollouts"
        logs.mkdir()
        # Missing fields and an invalid qidx make this record poisonous if the audit
        # tries to inspect test answers or join them to the training references.
        rows = [{"kind": "eval", "qidx": 999999999}]
        rows.extend({"kind": "train", "qidx": 0, "step": 0, "rank": 0,
                     "ref": logged_reference, "correct": 0,
                     "completion": "#### 7" if i % 2 else "No final answer."}
                    for i in range(sample_count))
        (logs / "rank0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        output = root / "results"
        argv = ["audit_rewards.py", "--rollouts", str(logs), "--train-data", str(data), "--output", str(output)]
        with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
            audit_rewards.main()
        return json.loads((output / "reward_audit.json").read_text(encoding="utf-8"))

    def test_test_records_are_excluded_before_any_reference_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_fixture(directory)
        self.assertEqual(result["counts"]["train_responses"], 16)
        self.assertEqual(result["counts"]["nontraining_records_excluded"], 1)
        self.assertEqual(result["per_run"]["format"]["all_wrong_groups_with_nonconstant_reward"], 1)

    def test_incomplete_sample_group_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Incomplete question group"):
                self.run_fixture(directory, sample_count=15)

    def test_mismatched_training_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Reference/order/scorer mismatch"):
                self.run_fixture(directory, logged_reference="9")


if __name__ == "__main__":
    unittest.main()
