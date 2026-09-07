import argparse
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


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
            "model_arch": "pawct",
            "model_mode": "frame_onset_offset",
            "post_processor_type": "onsets_frames",
            "sample_rate": None,
            "onset_tolerance": None,
            "name_suffix": "demo",
            "choral_enable": True,
            "range_prior_loss_weight": None,
            "continuity_prior_loss_weight": None,
            "config_overrides": [],
            "target_assignment": "part_name",
            "voice_assignment_method": None,
            "youchorale_dir": "",
            "youchorale_split_dir": "",
            "youchorale_pro_dir": "",
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_build_overrides_includes_model_name_when_provided(self):
        args = self.make_args(
            model_name="pawct_frame_onset_offset_logmel_sr16000_fps100_pro_choral_stream_foff_onsetx2_va_ordered_continuity_vint_self_attn"
        )

        overrides = search_best_thresholds.build_overrides(args)

        self.assertIn(
            "model.name=pawct_frame_onset_offset_logmel_sr16000_fps100_pro_choral_stream_foff_onsetx2_va_ordered_continuity_vint_self_attn",
            overrides,
        )

    def test_build_overrides_includes_youchorale_dir_when_provided(self):
        args = self.make_args(youchorale_dir="../NoteTranscription/dataset/test/YouChorale")

        overrides = search_best_thresholds.build_overrides(args)

        self.assertIn(
            "dataset.youchorale_dir=../NoteTranscription/dataset/test/YouChorale",
            overrides,
        )

    def test_build_overrides_includes_external_youchorale_split(self):
        args = self.make_args(
            youchorale_split_dir="../repro/splits/composition_disjoint_v1"
        )

        overrides = search_best_thresholds.build_overrides(args)

        self.assertIn(
            "dataset.youchorale_split_dir=../repro/splits/composition_disjoint_v1",
            overrides,
        )

    def test_build_overrides_selects_validation_split(self):
        overrides = search_best_thresholds.build_overrides(self.make_args())
        self.assertIn("dataset.eval_split=validation", overrides)

    def test_build_overrides_preserves_training_target_assignment(self):
        overrides = search_best_thresholds.build_overrides(
            self.make_args(target_assignment="ordered_continuity")
        )
        self.assertIn("choral.target_assignment=ordered_continuity", overrides)

    def test_build_overrides_preserves_training_prior_weights(self):
        overrides = search_best_thresholds.build_overrides(
            self.make_args(
                target_assignment="ordered_continuity",
                range_prior_loss_weight=0.0,
                continuity_prior_loss_weight=0.01,
            )
        )
        self.assertIn("choral.range_prior_loss_weight=0.0", overrides)
        self.assertIn("choral.continuity_prior_loss_weight=0.01", overrides)

    def test_build_overrides_accepts_repeatable_checkpoint_semantics(self):
        overrides = search_best_thresholds.build_overrides(
            self.make_args(
                config_overrides=[
                    "choral.voice_assignment_range_mins=[61,56,49,41]",
                    "choral.oc_gap_decay_seconds=3.0",
                ]
            )
        )
        self.assertIn(
            "choral.voice_assignment_range_mins=[61,56,49,41]", overrides
        )
        self.assertIn("choral.oc_gap_decay_seconds=3.0", overrides)

    def test_deprecated_assignment_flag_preserves_legacy_config_key(self):
        overrides = search_best_thresholds.build_overrides(
            self.make_args(
                target_assignment=None,
                voice_assignment_method="ordered_continuity",
            )
        )
        self.assertIn(
            "choral.voice_assignment_method=ordered_continuity", overrides
        )
        self.assertFalse(
            any(value.startswith("choral.target_assignment=") for value in overrides)
        )

    def test_build_overrides_rejects_canonical_and_deprecated_flags_together(self):
        with self.assertRaisesRegex(ValueError, "either canonical target_assignment"):
            search_best_thresholds.build_overrides(
                self.make_args(voice_assignment_method="range_prior")
            )

    def test_threshold_search_refuses_unacknowledged_test_tuning(self):
        with self.assertRaisesRegex(ValueError, "Refusing to tune thresholds on the test split"):
            search_best_thresholds.validate_selection_split("test")

    def test_threshold_search_allows_validation(self):
        search_best_thresholds.validate_selection_split("validation")

    def test_legacy_range_masked_assignment_remains_parseable(self):
        argv = [
            "search_best_thresholds.py",
            "--ckpt_iteration",
            "199999",
            "--output_txt",
            "result.tsv",
            "--target-assignment",
            "legacy_range_masked_continuity",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = search_best_thresholds.parse_args()
        self.assertEqual(args.target_assignment, "legacy_range_masked_continuity")
        self.assertIsNone(args.voice_assignment_method)

    def test_argparse_rejects_both_assignment_flag_families(self):
        argv = [
            "search_best_thresholds.py",
            "--ckpt_iteration",
            "199999",
            "--output_txt",
            "result.tsv",
            "--target-assignment",
            "range_prior",
            "--voice-assignment-method",
            "range_prior",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                search_best_thresholds.parse_args()

    def test_assignment_default_is_canonical_part_name(self):
        argv = [
            "search_best_thresholds.py",
            "--ckpt_iteration",
            "199999",
            "--output_txt",
            "result.tsv",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = search_best_thresholds.parse_args()
        self.assertEqual(args.target_assignment, "part_name")
        self.assertIsNone(args.voice_assignment_method)
        self.assertEqual(args.model_arch, "pagct")

    def test_choral_cli_defaults_to_pawct(self):
        argv = [
            "search_best_thresholds.py",
            "--ckpt_iteration",
            "best",
            "--output_txt",
            "result.tsv",
            "--choral_enable",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = search_best_thresholds.parse_args()
        self.assertEqual(args.model_arch, "pawct")


class RequireMetricValuesTest(unittest.TestCase):
    def test_require_metric_values_raises_on_empty_stats(self):
        with self.assertRaisesRegex(ValueError, "No matching evaluation songs"):
            search_best_thresholds.require_metric_values({}, context="voice=S")


class ChoralResultTest(unittest.TestCase):
    def test_same_output_assignment_ratio_is_micro_aggregated(self):
        stats = {
            "mean_satb_note_f1": [0.4, 0.6],
            "matched_note_voice_correct_count": [1, 9],
            "matched_note_voice_eligible_count": [2, 10],
            "union_matched_note_count": [3, 11],
        }
        for voice_name in search_best_thresholds.VOICE_NAMES:
            stats[f"{voice_name}_f1"] = [0.5]
            stats[f"{voice_name}_f1_50ms"] = [0.5]
            stats[f"{voice_name}_f1_100ms"] = [0.5]

        result = search_best_thresholds.build_choral_result(stats)

        self.assertAlmostEqual(
            result["matched_note_voice_accuracy"], 10 / 12
        )
        self.assertEqual(result["matched_note_voice_correct_count"], 10)
        self.assertEqual(result["matched_note_voice_eligible_count"], 12)
        self.assertEqual(result["union_matched_note_count"], 14)

    def test_both_choral_reports_share_explicit_same_output_fields(self):
        required = {
            "union_note_precision",
            "union_note_recall",
            "union_note_f1",
            "union_matched_note_count",
            "matched_note_voice_correct_count",
            "matched_note_voice_eligible_count",
            "matched_note_voice_accuracy",
            "voice_confusion_S_S",
            "voice_confusion_B_B",
        }
        self.assertTrue(required.issubset(search_best_thresholds.CHORAL_RESULT_KEYS))

    def test_count_metrics_are_formatted_as_counts(self):
        self.assertEqual(
            search_best_thresholds.format_metric_value(
                "matched_note_voice_correct_count", 7.0
            ),
            "7",
        )
        self.assertEqual(
            search_best_thresholds.format_metric_value("voice_confusion_S_A", 3.0),
            "3",
        )


if __name__ == "__main__":
    unittest.main()
