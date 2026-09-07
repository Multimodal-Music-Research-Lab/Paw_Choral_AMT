import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


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


if __name__ == "__main__":
    unittest.main()
