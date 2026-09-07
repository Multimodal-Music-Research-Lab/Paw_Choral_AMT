import sys
import unittest
from pathlib import Path

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_DIR / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from canonical_union import build_canonical_union_rolls, canonical_union_events


def note(pitch, onset, offset):
    return [pitch, 0, 0, onset, offset]


class CanonicalUnionEventsTest(unittest.TestCase):
    def test_nested_cross_voice_unison_retains_both_attacks(self):
        events = canonical_union_events(
            [{"S": [note(60, 0.0, 2.0)], "A": [note(60, 0.1, 1.0)]}],
            100,
        )
        self.assertEqual(events, [(60, 0.0, 0.1), (60, 0.1, 2.0)])

    def test_same_quantized_onset_is_one_attack_with_union_end(self):
        events = canonical_union_events(
            [{"S": [note(64, 0.101, 0.5)], "A": [note(64, 0.104, 1.0)]}],
            10,
        )
        self.assertEqual(events, [(64, 0.101, 1.0)])

    def test_overlapping_same_part_rearticulation_is_stable(self):
        events = canonical_union_events(
            [{"S": [note(67, 0.0, 1.1), note(67, 1.0, 2.0)]}],
            100,
        )
        self.assertEqual(events, [(67, 0.0, 1.0), (67, 1.0, 2.0)])

    def test_adjacent_and_gapped_notes_remain_separate(self):
        events = canonical_union_events(
            [{
                "S": [
                    note(60, 0.0, 1.0),
                    note(60, 1.0, 2.0),
                    note(60, 2.25, 3.0),
                ]
            }],
            100,
        )
        self.assertEqual(
            events,
            [(60, 0.0, 1.0), (60, 1.0, 2.0), (60, 2.25, 3.0)],
        )

    def test_unknown_part_labels_are_included(self):
        events = canonical_union_events(
            [{"mystery_part": [note(72, 0.0, 1.0)]}],
            100,
            begin_note=21,
            classes_num=88,
        )
        self.assertEqual(events, [(72, 0.0, 1.0)])

    def test_non_aligned_segment_uses_its_local_model_grid(self):
        note_bars = [{
            "S": [note(60, 0.06, 0.10), note(60, 0.14, 0.20)]
        }]
        global_events = canonical_union_events(note_bars, 10)
        local_events = canonical_union_events(
            note_bars,
            10,
            quantization_origin=0.04,
        )

        self.assertEqual(len(global_events), 1)
        self.assertEqual(len(local_events), 2)

    def test_invalid_and_out_of_range_notes_are_ignored(self):
        events = canonical_union_events(
            [{
                "S": [
                    note(20, 0.0, 1.0),
                    note(109, 0.0, 1.0),
                    note(60, 1.0, 1.0),
                    note(61.5, 0.0, 1.0),
                    note(62, float("nan"), 1.0),
                    note(63, 0.0, 1.0),
                ]
            }],
            100,
            begin_note=21,
            classes_num=88,
        )
        self.assertEqual(events, [(63, 0.0, 1.0)])


class CanonicalUnionRollsTest(unittest.TestCase):
    def test_long_note_crossing_segment_has_frames_without_false_boundaries(self):
        targets = build_canonical_union_rolls(
            [{"unknown": [note(60, 0.0, 5.0)]}],
            start_time=1.0,
            segment_seconds=2.0,
            frames_per_second=10,
            begin_note=21,
            classes_num=88,
        )
        note_index = 60 - 21
        self.assertEqual(targets["frame_roll"].shape, (21, 88))
        np.testing.assert_array_equal(targets["frame_roll"][:, note_index], 1.0)
        self.assertEqual(float(targets["onset_roll"][:, note_index].sum()), 0.0)
        self.assertEqual(float(targets["offset_roll"][:, note_index].sum()), 0.0)
        for mask_name in (
            "frame_mask_roll",
            "onset_mask_roll",
            "offset_mask_roll",
        ):
            np.testing.assert_array_equal(targets[mask_name], 1.0)

    def test_true_boundaries_inside_segment_are_marked(self):
        targets = build_canonical_union_rolls(
            [{"S": [note(60, 1.25, 2.25)]}],
            start_time=1.0,
            segment_seconds=2.0,
            frames_per_second=4,
            begin_note=60,
            classes_num=1,
        )
        self.assertEqual(targets["onset_roll"][1, 0], 1.0)
        self.assertEqual(targets["offset_roll"][5, 0], 1.0)
        np.testing.assert_array_equal(targets["frame_roll"][1:6, 0], 1.0)

    def test_release_at_final_endpoint_is_not_mislabeled_as_negative(self):
        targets = build_canonical_union_rolls(
            [{"S": [note(60, 0.0, 1.0)]}],
            start_time=0.0,
            segment_seconds=1.0,
            frames_per_second=10,
            begin_note=60,
            classes_num=1,
        )

        self.assertEqual(targets["offset_roll"][10, 0], 1.0)
        self.assertEqual(targets["offset_mask_roll"][10, 0], 1.0)


if __name__ == "__main__":
    unittest.main()
