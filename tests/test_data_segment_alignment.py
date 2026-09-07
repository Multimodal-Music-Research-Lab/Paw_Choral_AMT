import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

import data_generator
from data_generator import ChoralSATBDataset, resolve_segment_bounds
from utilities import int16_to_float32


class RecordingTargetProcessor:
    def __init__(self):
        self.start_times = []

    def process(
        self,
        start_time,
        midi_events_time,
        midi_events,
        extend_pedal,
        note_shift,
    ):
        self.start_times.append(start_time)
        target_dict = {
            "frame_mask_roll": np.ones((2, 1), dtype=np.float32),
            "onset_mask_roll": np.ones((2, 1), dtype=np.float32),
            "offset_mask_roll": np.ones((2, 1), dtype=np.float32),
        }
        return target_dict, [], []


class RecordingChoralTargetBuilder:
    def __init__(self):
        self.start_times = []

    def build(self, note_bars, *, start_time, **masks):
        self.start_times.append(start_time)
        return {"voice_frame_roll": np.zeros((2, 4, 1), dtype=np.float32)}


class SegmentAlignmentTest(unittest.TestCase):
    def test_segment_bounds_report_shifted_audio_start(self):
        self.assertEqual(
            resolve_segment_bounds(
                requested_start_time=0.5,
                waveform_samples=12,
                sample_rate=10,
                segment_samples=10,
            ),
            (2, 12, 0.2),
        )
        self.assertEqual(
            resolve_segment_bounds(
                requested_start_time=3.0,
                waveform_samples=5,
                sample_rate=10,
                segment_samples=10,
            ),
            (0, 10, 0.0),
        )

    def test_choral_global_and_voice_targets_use_actual_audio_start(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_dir = Path(tmpdir) / "hdf5s" / "youchorale_sr10"
            hdf5_dir.mkdir(parents=True)
            with h5py.File(hdf5_dir / "song.h5", "w") as hf:
                hf.create_dataset("waveform", data=np.arange(12, dtype=np.int16))
                hf.create_dataset("midi_event", data=np.asarray([], dtype="S1"))
                hf.create_dataset("midi_event_time", data=np.asarray([], dtype=np.float32))

            cfg = SimpleNamespace(
                exp=SimpleNamespace(workspace=tmpdir, random_seed=17, debug=False),
                feature=SimpleNamespace(
                    sample_rate=10,
                    segment_seconds=1.0,
                    frames_per_second=10,
                    begin_note=21,
                    classes_num=1,
                    max_note_shift=0,
                    augmentor=None,
                ),
            )
            dataset = object.__new__(ChoralSATBDataset)
            dataset.cfg = cfg
            dataset.dataset_name = "youchorale"
            dataset.is_training = False
            dataset.random_state = np.random.RandomState(cfg.exp.random_seed)
            dataset.hdf5s_dir = str(hdf5_dir)
            dataset.segment_samples = 10
            dataset.target_processor = RecordingTargetProcessor()
            dataset.choral_target_builder = RecordingChoralTargetBuilder()
            dataset.note_cache = {"song": []}

            with mock.patch.object(
                data_generator,
                "build_target_masks",
                side_effect=lambda _cfg, target_dict: target_dict,
            ):
                result = dataset[["song.h5", 0.5]]

        np.testing.assert_allclose(
            result["waveform"],
            int16_to_float32(np.arange(2, 12, dtype=np.int16)),
        )
        self.assertEqual(dataset.target_processor.start_times, [0.2])
        self.assertEqual(dataset.choral_target_builder.start_times, [0.2])


if __name__ == "__main__":
    unittest.main()
