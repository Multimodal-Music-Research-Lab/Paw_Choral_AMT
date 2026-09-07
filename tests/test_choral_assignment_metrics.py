import sys
import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from calculate_choral_scores import (
    _collapse_voice_events,
    _same_output_assignment_metrics,
    aggregate_same_output_summary,
)


def event(pitch, onset, offset):
    return {"midi_note": pitch, "onset_time": onset, "offset_time": offset}


class SameOutputAssignmentMetricsTest(unittest.TestCase):
    def test_swapped_voices_keep_union_score_but_fail_voice_accuracy(self):
        reference = {
            "S": [event(72, 0.0, 1.0)],
            "A": [event(64, 0.0, 1.0)],
            "T": [],
            "B": [],
        }
        estimated = {
            "S": [event(64, 0.0, 1.0)],
            "A": [event(72, 0.0, 1.0)],
            "T": [],
            "B": [],
        }
        result = _same_output_assignment_metrics(
            reference,
            estimated,
            frames_per_second=100.0,
            onset_tolerance=0.05,
            offset_ratio=0.2,
            offset_min_tolerance=0.05,
        )
        self.assertEqual(result["union_note_f1"], 1.0)
        self.assertEqual(result["matched_note_voice_accuracy"], 0.0)
        self.assertEqual(result["matched_note_voice_eligible_count"], 2)
        self.assertEqual(result["matched_note_voice_correct_count"], 0)
        self.assertEqual(result["voice_confusion_S_A"], 1)
        self.assertEqual(result["voice_confusion_A_S"], 1)

    def test_duplicate_voice_estimate_is_counted(self):
        reference = {"S": [event(72, 0.0, 1.0)], "A": [], "T": [], "B": []}
        estimated = {
            "S": [event(72, 0.0, 1.0)],
            "A": [event(72, 0.0, 0.8)],
            "T": [],
            "B": [],
        }
        result = _same_output_assignment_metrics(
            reference,
            estimated,
            frames_per_second=100.0,
            onset_tolerance=0.05,
            offset_ratio=0.2,
            offset_min_tolerance=0.05,
        )
        self.assertEqual(result["duplicate_estimate_excess_count"], 1)
        self.assertEqual(result["duplicate_estimate_rate"], 0.5)

    def test_union_projection_is_independent_of_metric_tolerance(self):
        estimated = {
            "S": [event(72, 0.00, 1.0)],
            "A": [event(72, 0.04, 1.0)],
            "T": [event(72, 0.08, 1.0)],
            "B": [],
        }

        collapsed, duplicate_excess, total = _collapse_voice_events(
            estimated, frames_per_second=100.0
        )

        self.assertEqual(total, 3)
        self.assertEqual(len(collapsed), 3)
        self.assertEqual(duplicate_excess, 0)
        self.assertAlmostEqual(collapsed[0]["onset_time"], 0.0)
        self.assertAlmostEqual(collapsed[1]["onset_time"], 0.04)
        self.assertAlmostEqual(collapsed[2]["onset_time"], 0.08)

    def test_short_rearticulation_in_one_voice_is_not_collapsed(self):
        estimated = {
            "S": [event(72, 0.00, 0.03), event(72, 0.04, 0.08)],
            "A": [],
            "T": [],
            "B": [],
        }

        collapsed, duplicate_excess, total = _collapse_voice_events(
            estimated, frames_per_second=100.0
        )

        self.assertEqual(total, 2)
        self.assertEqual(len(collapsed), 2)
        self.assertEqual(duplicate_excess, 0)

    def test_simultaneous_divisi_notes_are_one_switch_state(self):
        reference = {
            "S": [
                event(72, 0.00, 0.5),
                event(76, 0.03, 0.5),
                event(74, 1.00, 1.5),
            ],
            "A": [],
            "T": [],
            "B": [],
        }
        estimated = {
            "S": [event(72, 0.00, 0.5), event(76, 0.03, 0.5)],
            "A": [event(74, 1.00, 1.5)],
            "T": [],
            "B": [],
        }

        result = _same_output_assignment_metrics(
            reference,
            estimated,
            frames_per_second=100.0,
            onset_tolerance=0.05,
            offset_ratio=0.2,
            offset_min_tolerance=0.05,
        )

        self.assertEqual(result["voice_transition_count"], 1)
        self.assertEqual(result["voice_switch_count"], 1)
        self.assertEqual(result["voice_switch_rate"], 1.0)

    def test_ambiguous_unison_is_excluded_from_assignment_denominator(self):
        reference = {
            "S": [event(72, 0.00, 1.0)],
            "A": [event(72, 0.004, 1.0)],
            "T": [],
            "B": [],
        }
        estimated = {
            "S": [event(72, 0.01, 1.0)],
            "A": [],
            "T": [],
            "B": [],
        }

        result = _same_output_assignment_metrics(
            reference,
            estimated,
            frames_per_second=100.0,
            onset_tolerance=0.05,
            offset_ratio=0.2,
            offset_min_tolerance=0.05,
        )

        self.assertEqual(result["union_matched_note_count"], 1)
        self.assertEqual(result["matched_note_voice_eligible_count"], 0)
        self.assertEqual(
            result["matched_note_voice_ambiguous_reference_count"], 1
        )
        self.assertEqual(result["matched_note_voice_accuracy"], 0.0)

    def test_union_reference_is_collapsed_from_canonical_voice_events(self):
        reference_by_voice = {
            "S": [event(72, 0.0, 1.0)],
            "A": [],
            "T": [],
            "B": [],
        }
        estimated = {
            "S": [event(60, 0.0, 1.0)],
            "A": [],
            "T": [],
            "B": [],
        }

        result = _same_output_assignment_metrics(
            reference_by_voice,
            estimated,
            frames_per_second=100.0,
            onset_tolerance=0.05,
            offset_ratio=0.2,
            offset_min_tolerance=0.05,
        )

        self.assertEqual(result["union_note_f1"], 0.0)
        self.assertEqual(result["union_matched_note_count"], 0)
        self.assertEqual(result["matched_note_voice_eligible_count"], 0)

    def test_aggregate_retention_uses_counts_not_mean_of_song_ratios(self):
        summary = aggregate_same_output_summary(
            {
                "matched_note_voice_correct_count": [1, 9],
                "matched_note_voice_eligible_count": [2, 10],
                "matched_note_voice_accuracy": [0.5, 0.9],
            }
        )

        self.assertAlmostEqual(
            summary["matched_note_voice_accuracy"], 10 / 12
        )
        self.assertGreaterEqual(
            summary["matched_note_voice_accuracy"], 0.0
        )
        self.assertLessEqual(
            summary["matched_note_voice_accuracy"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
