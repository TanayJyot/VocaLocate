"""Check expensive-job ordering and stop conditions without contacting Modal."""

import copy
import json
from pathlib import Path
import unittest

from run_protocol import execute_runs, validate_plan, recovery_progress


PLAN = json.loads(Path(__file__).with_name("runs.json").read_text())


class RunProtocolTests(unittest.TestCase):
    def run_plan(self, train_rc=0, eval_rc=0):
        self.events, self.records = [], []

        def train(run):
            self.events.append(("train", run["name"]))
            return {"returncode": train_rc}

        def evaluate(run):
            self.events.append(("eval", run["name"]))
            return {"rl_step000466": {"returncode": eval_rc, "GSM8K": 0.2}}

        return execute_runs(PLAN, train, evaluate, self.records.append)

    def test_runs_are_sequential_and_each_is_evaluated(self):
        result = self.run_plan()
        self.assertEqual(self.events, [(stage, name) for name in ("format", "reference", "combined")
                                       for stage in ("train", "eval")])
        self.assertEqual(len(result), 3)
        self.assertEqual(self.records[-1]["status"], "complete")

    def test_training_failure_stops_before_eval_and_next_run(self):
        with self.assertRaises(RuntimeError):
            self.run_plan(train_rc=1)
        self.assertEqual(self.events, [("train", "format")])
        self.assertEqual(self.records[-1]["status"], "failed")

    def test_eval_failure_stops_before_next_training(self):
        with self.assertRaises(RuntimeError):
            self.run_plan(eval_rc=1)
        self.assertEqual(self.events, [("train", "format"), ("eval", "format")])

    def test_refuses_other_workspace_or_volume(self):
        for field in ("workspace", "destination_volume"):
            plan = copy.deepcopy(PLAN)
            plan[field] = "other"
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_plan(plan)

    def test_refuses_duplicate_or_smoke_output_tags(self):
        for tag in (PLAN["runs"][0]["output_tag"], PLAN["smoke_output_tag"]):
            plan = copy.deepcopy(PLAN)
            plan["runs"][1]["output_tag"] = tag
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                validate_plan(plan)

    def test_requires_original_sft_start_and_full_horizon(self):
        for field, value in (("model_step", 466), ("sft_tag", "baseline-rl")):
            plan = copy.deepcopy(PLAN)
            plan[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_plan(plan)
        plan = copy.deepcopy(PLAN)
        plan["training"]["optimizer_updates"] = 10
        with self.assertRaises(ValueError):
            validate_plan(plan)

    def test_resume_accepts_failed_first_run_without_restarting_the_plan(self):
        saved = {"plan": copy.deepcopy(PLAN), "status": "failed",
                 "current": PLAN["runs"][0]["output_tag"], "completed": []}
        self.assertEqual(recovery_progress(PLAN, saved, resume=True), [])
        with self.assertRaises(FileExistsError):
            recovery_progress(PLAN, saved, resume=False)

    def test_resume_requires_matching_rewards_and_a_failed_state(self):
        saved = {"plan": copy.deepcopy(PLAN), "status": "failed",
                 "current": PLAN["runs"][0]["output_tag"], "completed": []}
        saved["plan"]["runs"][0]["reward"] = "strict"
        with self.assertRaises(ValueError):
            recovery_progress(PLAN, saved, resume=True)
        saved["plan"] = copy.deepcopy(PLAN)
        saved["status"] = "training"
        with self.assertRaises(ValueError):
            recovery_progress(PLAN, saved, resume=True)

    def test_resume_allows_a_new_code_revision_but_keeps_the_experiment(self):
        saved = {"plan": {**copy.deepcopy(PLAN), "execution": {"code_revision": "old"}},
                 "status": "failed", "current": PLAN["runs"][0]["output_tag"], "completed": []}
        plan = {**copy.deepcopy(PLAN), "execution": {"code_revision": "new"}}
        self.assertEqual(recovery_progress(plan, saved, resume=True), [])

    def test_resume_ignores_descriptive_manifest_status(self):
        saved = {"plan": {**copy.deepcopy(PLAN), "status": "recorded"},
                 "status": "failed", "current": PLAN["runs"][0]["output_tag"], "completed": []}
        current = copy.deepcopy(PLAN)
        current.pop("status", None)
        self.assertEqual(recovery_progress(current, saved, resume=True), [])

    def test_completed_runs_are_retained_and_not_run_again(self):
        first = {**PLAN["runs"][0], "training": {"returncode": 0},
                 "evaluation": {"rl_step000466": {"returncode": 0, "GSM8K": 0.2}}}
        saved = {"plan": copy.deepcopy(PLAN), "status": "failed",
                 "current": PLAN["runs"][1]["output_tag"], "completed": [first]}
        progress = recovery_progress(PLAN, saved, resume=True)
        events, records = [], []

        def train(run):
            events.append(("train", run["name"]))
            return {"returncode": 0}

        def evaluate(run):
            events.append(("eval", run["name"]))
            return {"rl_step000466": {"returncode": 0, "GSM8K": 0.2}}

        result = execute_runs(PLAN, train, evaluate, records.append, completed=progress)
        self.assertEqual(events, [(s, n) for n in ("reference", "combined") for s in ("train", "eval")])
        self.assertEqual(result[0], first)
        self.assertEqual(records[-1]["status"], "complete")

    def test_resume_rejects_a_wrong_completed_prefix(self):
        saved = {"plan": copy.deepcopy(PLAN), "status": "failed",
                 "current": PLAN["runs"][1]["output_tag"], "completed": [PLAN["runs"][2]]}
        with self.assertRaises(ValueError):
            recovery_progress(PLAN, saved, resume=True)

    def test_platform_retry_accepts_its_own_in_progress_state(self):
        for status in ("training", "evaluating"):
            for resume in (False, True):
                saved = {"plan": copy.deepcopy(PLAN), "status": status,
                         "current": PLAN["runs"][0]["output_tag"], "completed": [],
                         "owner_call_id": "fc-same"}
                with self.subTest(status=status, resume=resume):
                    self.assertEqual(recovery_progress(
                        PLAN, saved, resume=resume, owner_call_id="fc-same"), [])

    def test_another_call_cannot_take_over_an_active_suite(self):
        saved = {"plan": copy.deepcopy(PLAN), "status": "training",
                 "current": PLAN["runs"][0]["output_tag"], "completed": [],
                 "owner_call_id": "fc-first"}
        with self.assertRaises(ValueError):
            recovery_progress(PLAN, saved, resume=True, owner_call_id="fc-second")

    def test_platform_retry_preserves_completed_runs(self):
        first = {**PLAN["runs"][0], "training": {"returncode": 0},
                 "evaluation": {"rl_step000466": {"returncode": 0, "GSM8K": 0.2}}}
        saved = {"plan": copy.deepcopy(PLAN), "status": "evaluating",
                 "current": PLAN["runs"][1]["output_tag"], "completed": [first],
                 "owner_call_id": "fc-same"}
        self.assertEqual(recovery_progress(PLAN, saved, owner_call_id="fc-same"), [first])

    def test_retry_after_final_status_returns_completed_suite(self):
        done = copy.deepcopy(PLAN["runs"])
        saved = {"plan": copy.deepcopy(PLAN), "status": "complete", "current": None,
                 "completed": done, "owner_call_id": "fc-same"}
        self.assertEqual(recovery_progress(PLAN, saved, owner_call_id="fc-same"), done)


if __name__ == "__main__":
    unittest.main()
