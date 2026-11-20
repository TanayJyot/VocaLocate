"""Local reward checks; these never import Modal or initialize a model."""

import importlib.util
from pathlib import Path
import re
import unittest


HOOK_PATH = Path(__file__).resolve().parents[1] / "part-3-rl" / "rewards.py"
spec = importlib.util.spec_from_file_location("part3_rewards", HOOK_PATH)
rewards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rewards)


def tool(expression, output):
    return (f"<|python_start|>{expression}<|python_end|>"
            f"<|output_start|>{output}<|output_end|>")


def context(response, answer="8", calculations=None):
    calculations = calculations or [("3*2", "6"), ("6*4", "24"), ("24/3", "8")]
    parts = []
    for expression, output in calculations:
        parts.extend([{"type": "python", "text": expression},
                      {"type": "python_output", "text": output}])
    parts.append({"type": "text", "text": f"\n#### {answer}"})
    match = re.search(r"#### (\-?[0-9\.\,]+)", response)
    pred = match.group(1).replace(",", "") if match else None
    return {"correct": int(pred == answer), "pred": pred, "ref": answer,
            "ref_solution_parts": parts}


class AnswerFormatTests(unittest.TestCase):
    def test_accepts_one_complete_final_line(self):
        self.assertEqual(rewards.score_answer_format("Work here.\n#### -1,234.5\n"), 1)

    def test_accepts_integer_decimal_and_fractional_decimal(self):
        for number in ("0", "-0.5", ".5", "12.0"):
            with self.subTest(number=number):
                self.assertEqual(rewards.score_answer_format(f"#### {number}"), 1)

    def test_rejects_empty_or_incomplete_answer(self):
        for response in ("", "Work here.", "####", "#### ", "#### NaN", "#### infinity"):
            with self.subTest(response=response):
                self.assertEqual(rewards.score_answer_format(response), 0)

    def test_rejects_multiple_answers_or_text_after_answer(self):
        for response in ("#### 2\n#### 8", "#### 8\nMaybe not.", "#### 8 or 9", "Work #### 8"):
            with self.subTest(response=response):
                self.assertEqual(rewards.score_answer_format(response), 0)

    def test_rejects_malformed_numbers_and_unsupported_notation(self):
        for number in ("1.2.3", "1,2", "+8", "8e0", "1/2", "\u0661\u0662"):
            with self.subTest(number=number):
                self.assertEqual(rewards.score_answer_format(f"#### {number}"), 0)


class ReferenceCalculationTests(unittest.TestCase):
    def score(self, response, calculations=None, answer="8"):
        ctx = context(response, answer, calculations)
        return rewards.score_reference_calculations(response, ctx["ref_solution_parts"], ctx["ref"])

    def test_one_of_two_intermediate_calculations_earns_half_credit(self):
        self.assertEqual(self.score(tool("3*2", "6")), 0.5)

    def test_all_intermediate_calculations_earn_full_credit(self):
        self.assertEqual(self.score(tool("3*2", "6") + tool("6*4", "24")), 1)

    def test_commutative_operand_order_and_whitespace_are_accepted(self):
        self.assertEqual(self.score(tool(" 2 * 3 ", "6.0") + tool("4*6", "24.0")), 1)

    def test_duplicate_calculations_cannot_inflate_credit(self):
        self.assertEqual(self.score(tool("3*2", "6") * 20), 0.5)

    def test_same_result_from_unrelated_calculation_earns_nothing(self):
        self.assertEqual(self.score(tool("1+5", "6")), 0)

    def test_claimed_calculator_result_must_be_arithmetically_valid(self):
        self.assertEqual(self.score(tool("3*2", "24")), 0)

    def test_printing_intermediate_numbers_is_not_a_calculation(self):
        self.assertEqual(self.score("6 24 8\n#### 7"), 0)

    def test_calculations_after_the_final_answer_do_not_earn_credit(self):
        self.assertEqual(self.score("#### 7\n" + tool("3*2", "6")), 0)

    def test_final_answer_calculation_is_excluded(self):
        self.assertEqual(self.score(tool("24/3", "8")), 0)

    def test_reference_without_intermediates_earns_no_credit(self):
        self.assertEqual(self.score(tool("4*2", "8"), [("4*2", "8")]), 0)

    def test_merely_wrapping_a_number_in_tool_tags_is_not_a_calculation(self):
        self.assertEqual(self.score(tool("6", "6"), [("6", "6"), ("6*4", "24")]), 0)

    def test_bad_reference_outputs_and_duplicate_references_are_filtered(self):
        calculations = [("3*2", "6"), ("2*3", "6.0"), ("4*6", "99"), ("24/3", "8")]
        self.assertEqual(self.score(tool("3*2", "6"), calculations), 1)

    def test_rounding_from_numeric_tool_output_is_tolerated(self):
        self.assertEqual(self.score(tool("1/3", "0.3333333333333333"),
                                    [("1/3", "0.3333333333333333"), ("3+5", "8")]), 1)

    def test_different_valid_solution_is_conservatively_uncredited(self):
        self.assertEqual(self.score(tool("12/2", "6")), 0)

    def test_rejects_unsafe_and_unbounded_expressions(self):
        for expression in ("__import__('os').system('echo unsafe')", "sum([3,3])",
                           "3*unknown", "10**100000000", "1e100000000", "1/0",
                           "(" * 500 + "6" + ")" * 500):
            with self.subTest(expression=expression):
                self.assertEqual(self.score(tool(expression, "6")), 0)

    def test_unclosed_tool_calls_and_non_numeric_outputs_are_ignored(self):
        for response in ("<|python_start|>3*2", tool("3*2", "[6]"), tool("3*2", "NaN")):
            with self.subTest(response=response):
                self.assertEqual(self.score(response), 0)


class RewardCompositionTests(unittest.TestCase):
    SPECS = ("strict,answer_format:0.1", "strict,reference_calculation:0.1",
             "strict,answer_format:0.1,reference_calculation:0.1")

    def test_both_components_are_available_to_existing_hook(self):
        for reward_spec in self.SPECS:
            with self.subTest(reward_spec=reward_spec):
                self.assertTrue(rewards.parse_spec(reward_spec))

    def test_correct_answer_always_keeps_reward_one(self):
        response = tool("3*2", "6") + tool("6*4", "24") + "\n#### 8"
        for reward_spec in self.SPECS:
            with self.subTest(reward_spec=reward_spec):
                total, parts = rewards.compute(rewards.parse_spec(reward_spec), {}, response, context(response))
                self.assertEqual(total, 1)
                self.assertTrue(all(value == 0 for name, value in parts.items() if name != "strict"))

    def test_incorrect_answer_can_receive_only_capped_partial_credit(self):
        response = tool("3*2", "6") + tool("6*4", "24") + "\n#### 7"
        for reward_spec, expected in zip(self.SPECS, (0.1, 0.1, 0.2)):
            with self.subTest(reward_spec=reward_spec):
                total, _ = rewards.compute(rewards.parse_spec(reward_spec), {}, response, context(response))
                self.assertAlmostEqual(total, expected)

    def test_default_strict_reward_is_unchanged(self):
        for response in ("#### 8", "#### 7", "No answer."):
            with self.subTest(response=response):
                ctx = context(response)
                self.assertEqual(rewards.compute(rewards.parse_spec("strict"), {}, response, ctx),
                                 (float(ctx["correct"]), {"strict": float(ctx["correct"])}))

    def test_auxiliary_rewards_do_not_read_the_question(self):
        response = tool("3*2", "6") + "\n#### 7"
        ctx = context(response)
        for name in ("answer_format", "reference_calculation"):
            with self.subTest(name=name):
                self.assertGreater(rewards.REWARDS[name]({}, response, ctx), 0)


if __name__ == "__main__":
    unittest.main()
