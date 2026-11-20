"""Guard paired comparisons, strict scoring, and the units used in report plots."""
import unittest
from decimal import Decimal
from analyze_results import error_category, paired_comparison, logged_sequence_lengths, step_band


class AnalysisTests(unittest.TestCase):
    def test_uses_original_strict_string_match(self):
        self.assertEqual(error_category("15.0", "15", set()), "Other wrong answer")
        self.assertEqual(error_category("15", "15", set()), "Correct")

    def test_missing_answer_and_reference_intermediate_are_distinct(self):
        self.assertEqual(error_category(None, "90", {Decimal(30)}), "No extracted answer")
        self.assertEqual(error_category("30", "90", {Decimal(30)}), "Reference intermediate")

    def test_paired_differences_count_gains_and_losses(self):
        result = paired_comparison([0, 0, 1, 1], [1, 0, 0, 1])
        self.assertEqual((result["gained"], result["lost"], result["net"]), (1, 1, 0))
        self.assertLess(result["ci_low_pp"], 0)
        self.assertGreater(result["ci_high_pp"], 0)

    def test_identical_predictions_have_zero_interval(self):
        result = paired_comparison([0, 1, 0], [0, 1, 0])
        self.assertEqual((result["ci_low_pp"], result["ci_high_pp"]), (0, 0))

    def test_sequence_lengths_come_from_same_logged_unit_for_every_run(self):
        text = "Step 1/467 | Average reward: 0.3 | Average sequence length: 145.96\n"
        self.assertEqual(logged_sequence_lengths(text), {1: 145.96})

    def test_reference_bins_include_questions_without_calculator_annotations(self):
        self.assertEqual([step_band(x) for x in range(6)], ["0-1", "0-1", "2", "3", "4+", "4+"])
