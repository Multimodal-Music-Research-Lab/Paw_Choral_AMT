import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


REPO_DIR = Path(__file__).resolve().parents[1]
for source_dir in (REPO_DIR / "src", REPO_DIR / "tools", REPO_DIR / "experiments"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import calculate_scores
import evaluate
import losses
import utilities


class _DummyPostProcessor:
    def output_dict_to_midi_events(self, post_input):
        return (
            [
                {
                    "onset_time": 0.0,
                    "offset_time": 0.5,
                    "midi_note": 60,
                    "velocity": 100,
                }
            ],
            [],
        )


class NoVelocityMainchainTest(unittest.TestCase):
    def make_model_cfg(self, *, mode=None, model_type=None):
        return SimpleNamespace(
            model=SimpleNamespace(
                arch="pagct",
                mode=mode,
                type=model_type,
            )
        )

    def make_score_cfg(self):
        return SimpleNamespace(
            score=SimpleNamespace(
                evaluate_frame=False,
                evaluate_note=True,
                evaluate_pedal=False,
                onset_tolerance=0.05,
                offset_ratio=0.2,
                offset_min_tolerance=0.05,
                pedal_offset_ratio=0.2,
                pedal_offset_min_tolerance=0.05,
            ),
            post=SimpleNamespace(frame_threshold=0.5),
        )

    def test_get_task_spec_rejects_velocity_modes(self):
        with self.assertRaisesRegex(ValueError, "Unsupported model.mode"):
            utilities.get_task_spec(self.make_model_cfg(mode="frame_onset_offset_velo"))

        with self.assertRaisesRegex(ValueError, "Unsupported model.mode"):
            utilities.get_task_spec(self.make_model_cfg(mode="frame_onset_offset_velo_pedal"))

    def test_task_bce_ignores_velocity_output(self):
        dummy_model = SimpleNamespace(cfg=self.make_model_cfg(mode="frame_onset_offset"))

        output_dict = {
            "frame_output": torch.tensor([[0.9]], dtype=torch.float32),
            "onset_output": torch.tensor([[0.8]], dtype=torch.float32),
            "offset_output": torch.tensor([[0.7]], dtype=torch.float32),
            "velocity_output": torch.tensor([[0.6]], dtype=torch.float32),
        }
        target_dict = {
            "frame_roll": torch.tensor([[1.0]], dtype=torch.float32),
            "onset_roll": torch.tensor([[1.0]], dtype=torch.float32),
            "offset_roll": torch.tensor([[0.0]], dtype=torch.float32),
            "frame_mask_roll": torch.tensor([[1.0]], dtype=torch.float32),
            "onset_mask_roll": torch.tensor([[1.0]], dtype=torch.float32),
            "offset_mask_roll": torch.tensor([[1.0]], dtype=torch.float32),
        }

        actual = losses.task_bce(dummy_model, output_dict, target_dict)
        expected = (
            losses.bce(output_dict["frame_output"], target_dict["frame_roll"], target_dict["frame_mask_roll"])
            + losses.bce(output_dict["onset_output"], target_dict["onset_roll"], target_dict["onset_mask_roll"])
            + losses.bce(output_dict["offset_output"], target_dict["offset_roll"], target_dict["offset_mask_roll"])
        )

        self.assertAlmostEqual(actual.item(), expected.item(), places=6)

    def test_segment_evaluator_does_not_report_velocity_metric(self):
        cfg = self.make_model_cfg(mode="frame_onset_offset")
        evaluator = evaluate.SegmentEvaluator(model=SimpleNamespace(), cfg=cfg)
        fake_output = {
            "frame_output": np.array([[0.9]], dtype=np.float32),
            "frame_roll": np.array([[1.0]], dtype=np.float32),
            "frame_mask_roll": np.array([[1.0]], dtype=np.float32),
            "velocity_output": np.array([[0.6]], dtype=np.float32),
            "velocity_roll": np.array([[96.0]], dtype=np.float32),
            "velocity_mask_roll": np.array([[1.0]], dtype=np.float32),
        }

        with mock.patch.object(evaluate, "forward_dataloader", return_value=fake_output):
            stats = evaluator.evaluate(dataloader=None)

        self.assertIn("frame_ap", stats)
        self.assertNotIn("velocity_mae", stats)

    def test_score_calculator_does_not_emit_velocity_metrics(self):
        class _DiagnosticValidator:
            def validate(self, *args, **kwargs):
                return None

        calc = calculate_scores.ScoreCalculator.__new__(calculate_scores.ScoreCalculator)
        calc.cfg = self.make_score_cfg()
        calc.spec = SimpleNamespace(offset=True, pedal=False)
        calc.post_processor = _DummyPostProcessor()
        calc.artifact_validator = _DiagnosticValidator()

        with tempfile.TemporaryDirectory() as tmpdir:
            calc.probs_dir = tmpdir
            total_dict = {
                "ref_on_off_pairs": np.array([[0.0, 0.5]], dtype=np.float32),
                "ref_midi_notes": np.array([60], dtype=np.int32),
                "ref_velocity": np.array([100], dtype=np.int32),
                "velocity_output": np.array([[0.75]], dtype=np.float32),
                "velocity_roll": np.array([[96.0]], dtype=np.float32),
                "velocity_mask_roll": np.array([[1.0]], dtype=np.float32),
            }
            with open(os.path.join(tmpdir, "song.pkl"), "wb") as fw:
                pickle.dump(total_dict, fw)

            stats = calc.calculate_score_per_song([0, "/unused/song.h5"])

        self.assertNotIn("note_with_velocity_f1", stats)
        self.assertNotIn("velocity_mae", stats)


if __name__ == "__main__":
    unittest.main()
