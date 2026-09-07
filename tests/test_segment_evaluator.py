import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

import evaluate
from evaluate import SegmentEvaluator, _masked_average_precision


class MaskedAveragePrecisionTest(unittest.TestCase):
    def test_perfect_ranking(self):
        target = np.array([0, 1, 0, 1], dtype=np.float32)
        output = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
        self.assertEqual(_masked_average_precision(target, output), 1.0)

    def test_empty_positive_target_is_not_a_selection_metric(self):
        target = np.zeros(4, dtype=np.float32)
        output = np.ones(4, dtype=np.float32)
        self.assertIsNone(_masked_average_precision(target, output))

    def test_mask_is_applied(self):
        target = np.array([1, 0, 0], dtype=np.float32)
        output = np.array([0.8, 0.9, 0.1], dtype=np.float32)
        mask = np.array([1, 0, 1], dtype=np.float32)
        self.assertEqual(_masked_average_precision(target, output, mask), 1.0)

    def test_evaluator_keeps_nonfinite_default_selection_keys(self):
        cfg = SimpleNamespace(
            model=SimpleNamespace(arch="pawct", mode="frame_onset", type=None),
            choral=SimpleNamespace(enable=True, voice_names=["S", "A", "T", "B"]),
        )
        shape = (1, 2, 4, 3)
        fake_output = {
            "frame_output": np.zeros((1, 2, 3), dtype=np.float32),
            "frame_roll": np.zeros((1, 2, 3), dtype=np.float32),
            "frame_mask_roll": np.ones((1, 2, 3), dtype=np.float32),
            "voice_frame_output": np.zeros(shape, dtype=np.float32),
            "voice_frame_roll": np.zeros(shape, dtype=np.float32),
            "voice_frame_mask_roll": np.ones(shape, dtype=np.float32),
        }

        with mock.patch.object(evaluate, "forward_dataloader", return_value=fake_output):
            statistics = SegmentEvaluator(SimpleNamespace(), cfg).evaluate(None)

        self.assertTrue(np.isnan(statistics["frame_ap"]))
        self.assertTrue(np.isnan(statistics["mean_voice_frame_ap"]))
        for voice_name in cfg.choral.voice_names:
            self.assertTrue(np.isnan(statistics[f"{voice_name}_frame_ap"]))

    def test_disk_backed_streaming_matches_aggregated_metrics(self):
        class EchoModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, waveform):
                frame = waveform + self.anchor
                return {
                    "frame_output": frame,
                    "voice_frame_output": torch.stack(
                        [frame, 1.0 - frame, frame * 0.8, frame * 0.6],
                        dim=2,
                    ),
                    "onset_output": frame * 0.5,
                    "offset_output": frame * 0.25,
                }

        waveforms = [
            np.array([[[0.9, 0.2], [0.4, 0.8]]], dtype=np.float32),
            np.array([[[0.1, 0.7], [0.6, 0.3]]], dtype=np.float32),
        ]
        batches = []
        for waveform in waveforms:
            frame_roll = np.array(
                [[[1.0, 0.0], [0.0, 1.0]]],
                dtype=np.float32,
            )
            frame_mask = np.array(
                [[[1.0, 1.0], [1.0, 0.0]]],
                dtype=np.float32,
            )
            voice_roll = np.stack(
                [frame_roll, 1.0 - frame_roll, frame_roll, 1.0 - frame_roll],
                axis=2,
            )
            voice_mask = np.repeat(frame_mask[:, :, None, :], 4, axis=2)
            batches.append(
                {
                    "waveform": waveform,
                    "frame_roll": frame_roll,
                    "frame_mask_roll": frame_mask,
                    "voice_frame_roll": voice_roll,
                    "voice_frame_mask_roll": voice_mask,
                    "onset_roll": frame_roll,
                    "onset_mask_roll": frame_mask,
                    "offset_roll": frame_roll,
                    "offset_mask_roll": frame_mask,
                }
            )

        with tempfile.TemporaryDirectory() as workspace:
            cfg = SimpleNamespace(
                model=SimpleNamespace(
                    arch="pawct",
                    mode="frame_onset_offset",
                    type=None,
                ),
                choral=SimpleNamespace(
                    enable=True,
                    voice_names=["S", "A", "T", "B"],
                ),
                exp=SimpleNamespace(workspace=workspace),
            )
            model = EchoModel()
            evaluator = SegmentEvaluator(model, cfg)
            expected = evaluator._evaluate_aggregated(
                evaluate.forward_dataloader(model, batches, return_target=True)
            )
            actual = evaluator.evaluate(batches)

            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assertAlmostEqual(actual[key], expected[key], places=7)
            self.assertEqual(list(Path(workspace).glob(".segment-eval-*")), [])


if __name__ == "__main__":
    unittest.main()
