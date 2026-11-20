"""Exercise interruption recovery without starting a process or contacting Modal."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

with patch.dict(os.environ, RL_VOLUME="vocalocate-a4-part4"):
    import modal_part4 as launcher


PLAN = json.loads(Path(__file__).with_name("runs.json").read_text())


class SuiteRuntimeTests(unittest.TestCase):
    def test_preempted_gpu_input_retries_without_nested_jobs_or_repeating_completed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "a3"
            results = base / "results"
            results.mkdir(parents=True)
            (results / f"{PLAN['smoke_output_tag']}.json").write_text(
                json.dumps({"returncode": 0, "smoke_steps": 10}))
            events = []
            interrupted = False

            def train(*args):
                tag, reward = args[5], args[-1]
                events.append(("train", tag))
                run = base / "rl_runs" / tag
                model = run / "chatrl_checkpoints" / PLAN["model_tag"] / "model_000466.pt"
                model.parent.mkdir(parents=True, exist_ok=True)
                model.write_bytes(b"test checkpoint")
                (run / "reward.txt").write_text(reward)
                result = {"returncode": 0, "reward": reward, "smoke_steps": 0}
                (results / f"{tag}.json").write_text(json.dumps(result))
                return result

            def evaluate(*args):
                nonlocal interrupted
                tag = args[2]
                events.append(("eval", tag))
                output = base / "rl_runs" / tag / "final_eval"
                output.mkdir()
                if tag == PLAN["runs"][1]["output_tag"] and not interrupted:
                    interrupted = True
                    (output / "partial.log").write_text("interrupted evaluation")
                    raise KeyboardInterrupt("platform preemption")
                rows = [{"qidx": i, "correct": i < 265} for i in range(1319)]
                (output / "gsm8k_rl_step000466_rank0.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows))
                return {"rl_step000466": {"returncode": 0, "GSM8K": 265 / 1319}}

            def path(value):
                return root if str(value) == "/vol" else Path(value)

            with (patch.object(launcher, "Path", side_effect=path),
                  patch.object(launcher, "vol", Mock()),
                  patch.object(launcher.modal, "current_function_call_id", return_value="fc-same"),
                  patch.object(launcher.train, "local", side_effect=train),
                  patch.object(launcher.final_eval, "local", side_effect=evaluate),
                  patch.object(launcher.train, "remote", side_effect=AssertionError("nested GPU job")),
                  patch.object(launcher.final_eval, "remote", side_effect=AssertionError("nested GPU job"))):
                with self.assertRaises(KeyboardInterrupt):
                    launcher.part4_suite.local(PLAN)
                saved = json.loads((results / "part4_suite_v1.json").read_text())
                self.assertEqual(saved["status"], "evaluating")
                self.assertEqual(len(saved["completed"]), 1)
                done = launcher.part4_suite.local(PLAN)
                self.assertEqual(len(done), 3)
                self.assertEqual(launcher.part4_suite.local(PLAN), done)
            names = [run["output_tag"] for run in PLAN["runs"]]
            self.assertEqual(events, [("train", names[0]), ("eval", names[0]),
                                      ("train", names[1]), ("eval", names[1]),
                                      ("eval", names[1]), ("train", names[2]), ("eval", names[2])])
            self.assertEqual(len(list((base / "rl_runs" / names[1]).glob("final_eval_before_resume_*"))), 1)
