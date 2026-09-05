import argparse
import os
import sys
import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
for source_dir in (REPO_DIR / "src", REPO_DIR / "tools", REPO_DIR / "experiments"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import search_best_thresholds


class BuildOverridesTest(unittest.TestCase):
    def make_args(self, **kwargs):
        defaults = {
            "test_set": "youchorale",
            "split": "validation",
            "allow_test_tuning": False,
            "workspace": "./workspaces",
            "ckpt_iteration": "199999",
            "model_name": "",
            "model_arch": "hpt",
            "model_mode": "frame_onset_offset",
            "post_processor_type": "onsets_frames",
            "sample_rate": None,
            "onset_tolerance": None,
            "name_suffix": "demo",
            "choral_enable": True,
            "voice_assignment_method": "part_name",
            "youchorale_dir": "",
            "youchorale_pro_dir": "",
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_build_overrides_includes_model_name_when_provided(self):
        args = self.make_args(
            model_name="hpt_frame_onset_offset_logmel_sr16000_fps100_pro_choral_stream_foff_onsetx2_va_ordered_continuity_vint_self_attn"
        )

        overrides = search_best_thresholds.build_overrides(args)

        self.assertIn(
            "model.name=hpt_frame_onset_offset_logmel_sr16000_fps100_pro_choral_stream_foff_onsetx2_va_ordered_continuity_vint_self_attn",
            overrides,
        )

    def test_build_overrides_includes_youchorale_dir_when_provided(self):
        args = self.make_args(youchorale_dir="../NoteTranscription/dataset/test/YouChorale")

        overrides = search_best_thresholds.build_overrides(args)

        self.assertIn(
            "dataset.youchorale_dir=../NoteTranscription/dataset/test/YouChorale",
            overrides,
        )

    def test_build_overrides_selects_validation_split(self):
        overrides = search_best_thresholds.build_overrides(self.make_args())
        self.assertIn("dataset.eval_split=validation", overrides)

    def test_build_overrides_preserves_training_voice_assignment(self):
        overrides = search_best_thresholds.build_overrides(
            self.make_args(voice_assignment_method="ordered_continuity")
        )
        self.assertIn("choral.voice_assignment_method=ordered_continuity", overrides)

    def test_threshold_search_refuses_unacknowledged_test_tuning(self):
        with self.assertRaisesRegex(ValueError, "Refusing to tune thresholds on the test split"):
            search_best_thresholds.validate_selection_split("test")

    def test_threshold_search_allows_validation(self):
        search_best_thresholds.validate_selection_split("validation")


class RequireMetricValuesTest(unittest.TestCase):
    def test_require_metric_values_raises_on_empty_stats(self):
        with self.assertRaisesRegex(ValueError, "No matching evaluation songs"):
            search_best_thresholds.require_metric_values({}, context="voice=S")


if __name__ == "__main__":
    unittest.main()
