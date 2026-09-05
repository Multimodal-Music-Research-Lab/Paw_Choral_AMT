import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
for source_dir in (REPO_DIR / "src", REPO_DIR / "tools", REPO_DIR / "experiments"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import inference


class BuildTotalDictTest(unittest.TestCase):
    def test_build_total_dict_keeps_per_voice_rolls_and_masks(self):
        output_dict = {
            "frame_output": np.zeros((2, 88), dtype=np.float32),
            "voice_frame_output": np.zeros((2, 4, 88), dtype=np.float32),
        }
        target_dict = {
            "frame_roll": np.ones((2, 88), dtype=np.float32),
            "onset_roll": np.ones((2, 88), dtype=np.float32),
            "offset_roll": np.ones((2, 88), dtype=np.float32),
            "pedal_onset_roll": np.zeros((2,), dtype=np.float32),
            "pedal_offset_roll": np.zeros((2,), dtype=np.float32),
            "pedal_frame_roll": np.zeros((2,), dtype=np.float32),
            "frame_mask_roll": np.ones((2, 88), dtype=np.float32),
            "onset_mask_roll": np.ones((2, 88), dtype=np.float32),
            "offset_mask_roll": np.ones((2, 88), dtype=np.float32),
            "pedal_mask_roll": np.zeros((2,), dtype=np.float32),
            "voice_frame_roll": np.ones((2, 4, 88), dtype=np.float32),
            "voice_onset_roll": np.ones((2, 4, 88), dtype=np.float32),
            "voice_offset_roll": np.ones((2, 4, 88), dtype=np.float32),
            "voice_frame_mask_roll": np.ones((2, 4, 88), dtype=np.float32),
            "voice_onset_mask_roll": np.ones((2, 4, 88), dtype=np.float32),
            "voice_offset_mask_roll": np.ones((2, 4, 88), dtype=np.float32),
        }

        total_dict = inference.build_total_dict(
            output_dict=output_dict,
            target_dict=target_dict,
            ref_on_off_pairs=np.array([[0.0, 0.5]], dtype=np.float32),
            ref_midi_notes=np.array([60], dtype=np.int32),
            ref_pedal_on_off_pairs=np.zeros((0, 2), dtype=np.float32),
        )

        self.assertIn("voice_frame_roll", total_dict)
        self.assertIn("voice_onset_roll", total_dict)
        self.assertIn("voice_offset_roll", total_dict)
        self.assertIn("voice_frame_mask_roll", total_dict)
        self.assertIn("voice_onset_mask_roll", total_dict)
        self.assertIn("voice_offset_mask_roll", total_dict)
        self.assertEqual(total_dict["voice_frame_roll"].shape, (2, 4, 88))
        self.assertEqual(total_dict["voice_frame_mask_roll"].shape, (2, 4, 88))

    def test_build_choral_target_dict_creates_per_voice_rolls_for_range_prior(self):
        cfg = SimpleNamespace(
            feature=SimpleNamespace(
                begin_note=21,
                classes_num=88,
                frames_per_second=10,
                segment_seconds=2.0,
            ),
            choral=SimpleNamespace(
                num_voices=4,
                voice_names=["S", "A", "T", "B"],
                voice_assignment_method="range_prior",
                voice_assignment_part_penalty=2.0,
                voice_assignment_continuity_weight=0.35,
                voice_assignment_overlap_penalty=4.0,
                voice_assignment_range_mins=[60, 55, 48, 40],
                voice_assignment_range_maxs=[88, 79, 72, 67],
                voice_assignment_range_margin=2.0,
                voice_assignment_mask_penalty=8.0,
            ),
        )
        note_bars = [
            {
                "S": [[80, 0, 0, 0.0, 1.0]],
                "B": [[45, 0, 0, 0.5, 1.5]],
            }
        ]

        target_dict = inference.build_choral_target_dict(
            cfg=cfg,
            note_bars=note_bars,
            segment_seconds=2.0,
            start_time=0.0,
        )

        self.assertEqual(target_dict["voice_frame_roll"].shape, (21, 4, 88))
        self.assertEqual(target_dict["voice_frame_mask_roll"].shape, (21, 4, 88))
        self.assertEqual(float(target_dict["voice_presence"][0]), 1.0)
        self.assertEqual(float(target_dict["voice_presence"][3]), 1.0)
        self.assertGreater(target_dict["voice_frame_roll"][:, 0, 80 - 21].sum(), 0.0)
        self.assertGreater(target_dict["voice_frame_roll"][:, 3, 45 - 21].sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
