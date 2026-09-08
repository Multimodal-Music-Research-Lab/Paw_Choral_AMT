import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "src"))

from piano_vad import (  # noqa: E402
    note_detection_with_onset_offset_regress,
    onsets_frames_note_detection,
)
from utilities import OnsetsFramesPostProcessor, RegressionPostProcessor, TargetProcessor  # noqa: E402


def _postprocessor_cfg():
    return SimpleNamespace(
        feature=SimpleNamespace(
            frames_per_second=100,
            classes_num=1,
            begin_note=60,
            velocity_scale=128,
        ),
        post=SimpleNamespace(
            frame_threshold=0.5,
            onset_threshold=0.5,
            offset_threshold=0.5,
            pedal_offset_threshold=0.5,
            default_velocity=80,
        ),
    )


class NoteDecoderBoundaryTest(unittest.TestCase):
    def test_onsets_frames_sharp_output_vectorizes_strict_local_maxima(self):
        processor = OnsetsFramesPostProcessor.__new__(OnsetsFramesPostProcessor)
        values = np.asarray(
            [
                [0.6, 0.3, 0.9],
                [0.4, 0.8, 0.7],
                [0.7, 0.8, 0.6],
                [0.2, 0.9, 0.8],
            ],
            dtype=np.float32,
        )
        expected = np.asarray(
            [
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(processor.sharp_output(values, 0.5), expected)
        np.testing.assert_array_equal(
            processor.sharp_output(np.asarray([[0.6, 0.5]], dtype=np.float32), 0.5),
            np.asarray([[1.0, 0.0]], dtype=np.float32),
        )

    def test_regression_decoder_keeps_onset_at_frame_zero(self):
        frame = np.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        onset = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        offset = np.zeros_like(onset)
        shift = np.zeros_like(onset)
        velocity = np.full_like(onset, 0.75)

        events = note_detection_with_onset_offset_regress(
            frame, onset, shift, offset, shift, velocity, frame_threshold=0.5
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][:2], [0, 3])

    def test_onsets_frames_decoder_keeps_onset_at_frame_zero(self):
        frame = np.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        onset = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        offset = np.zeros_like(onset)
        velocity = np.full_like(onset, 0.75)

        events = onsets_frames_note_detection(
            frame, onset, offset, velocity, threshold=0.5
        )

        self.assertEqual(events, [[0, 3, np.float32(0.75)]])

    def test_regression_decoder_flushes_held_note_at_stream_end(self):
        frame = np.ones(6, dtype=np.float32)
        onset = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        offset = np.zeros_like(onset)
        shift = np.zeros_like(onset)
        velocity = np.full_like(onset, 0.5)

        events = note_detection_with_onset_offset_regress(
            frame, onset, shift, offset, shift, velocity, frame_threshold=0.5
        )

        self.assertEqual(events, [[1, 5, np.float32(0.0), 0.0, np.float32(0.5)]])

    def test_onsets_frames_decoder_flushes_held_note_at_stream_end(self):
        frame = np.ones(6, dtype=np.float32)
        onset = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        offset = np.zeros_like(onset)
        velocity = np.full_like(onset, 0.5)

        events = onsets_frames_note_detection(
            frame, onset, offset, velocity, threshold=0.5
        )

        self.assertEqual(events, [[2, 5, np.float32(0.5)]])

    def test_last_frame_onset_is_discarded_at_audio_boundary(self):
        frame = np.ones(3, dtype=np.float32)
        onset = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        offset = np.zeros_like(onset)
        shift = np.zeros_like(onset)
        velocity = np.full_like(onset, 0.5)

        regression_events = note_detection_with_onset_offset_regress(
            frame, onset, shift, offset, shift, velocity, frame_threshold=0.5
        )
        onsets_frames_events = onsets_frames_note_detection(
            frame, onset, offset, velocity, threshold=0.5
        )

        self.assertEqual(regression_events, [])
        self.assertEqual(onsets_frames_events, [])

    def test_single_frame_onset_is_discarded(self):
        frame = np.ones(1, dtype=np.float32)
        onset = np.ones(1, dtype=np.float32)
        offset = np.zeros_like(onset)
        shift = np.zeros_like(onset)
        velocity = np.full_like(onset, 0.5)

        regression_events = note_detection_with_onset_offset_regress(
            frame, onset, shift, offset, shift, velocity, frame_threshold=0.5
        )
        onsets_frames_events = onsets_frames_note_detection(
            frame, onset, offset, velocity, threshold=0.5
        )

        self.assertEqual(regression_events, [])
        self.assertEqual(onsets_frames_events, [])

    def test_regression_tail_flush_uses_last_frame_explicit_offset(self):
        frame = np.ones(4, dtype=np.float32)
        onset = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        offset = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        onset_shift = np.zeros_like(onset)
        offset_shift = np.array([0.0, 0.0, 0.0, -0.25], dtype=np.float32)
        velocity = np.full_like(onset, 0.5)

        events = note_detection_with_onset_offset_regress(
            frame,
            onset,
            onset_shift,
            offset,
            offset_shift,
            velocity,
            frame_threshold=0.5,
        )

        self.assertEqual(events[0][:2], [1, 3])
        self.assertEqual(events[0][3], np.float32(-0.25))

    def test_onsets_frames_decoder_uses_explicit_offset_before_frame_drop(self):
        frame = np.ones(6, dtype=np.float32)
        onset = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        offset = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        velocity = np.full_like(onset, 0.5)

        events = onsets_frames_note_detection(
            frame, onset, offset, velocity, threshold=0.5
        )

        self.assertEqual(events, [[1, 3, np.float32(0.5)]])

    def test_onsets_frames_postprocessor_preserves_boundary_peak(self):
        processor = OnsetsFramesPostProcessor(_postprocessor_cfg())
        note_events, _ = processor.output_dict_to_midi_events(
            {
                "frame_output": np.array([[1.0], [1.0], [0.0]], dtype=np.float32),
                "onset_output": np.array([[0.9], [0.2], [0.0]], dtype=np.float32),
                "offset_output": np.zeros((3, 1), dtype=np.float32),
            }
        )

        self.assertEqual(len(note_events), 1)
        self.assertAlmostEqual(note_events[0]["onset_time"], 0.0)
        self.assertAlmostEqual(note_events[0]["offset_time"], 0.02)

    def test_regression_postprocessor_preserves_boundary_peak(self):
        processor = RegressionPostProcessor(_postprocessor_cfg())
        note_events, _ = processor.output_dict_to_midi_events(
            {
                "frame_output": np.array([[1.0], [1.0], [0.0]], dtype=np.float32),
                "onset_output": np.array([[0.9], [0.4], [0.0]], dtype=np.float32),
                "offset_output": np.zeros((3, 1), dtype=np.float32),
            }
        )

        self.assertEqual(len(note_events), 1)
        self.assertAlmostEqual(note_events[0]["onset_time"], 0.0)
        self.assertAlmostEqual(note_events[0]["offset_time"], 0.02)


class RegressionTargetTest(unittest.TestCase):
    def test_inter_peak_falloff_uses_the_next_peaks_subframe_offset(self):
        processor = TargetProcessor.__new__(TargetProcessor)
        processor.frames_per_second = 100
        target = np.ones(12, dtype=np.float64)
        target[2] = 0.004
        target[8] = -0.003

        regression = processor.get_regression(target)

        # Frame 5 belongs to the descending side of the peak at frame 8.
        # Its distance is |-0.03 - (-0.003)| = 0.027 s, hence 1 - 20*0.027.
        self.assertAlmostEqual(regression[5], 0.46, places=7)


class TargetProcessorBoundaryTest(unittest.TestCase):
    def test_final_note_on_is_included_without_end_sentinel(self):
        processor = TargetProcessor(segment_seconds=2.0, cfg=_postprocessor_cfg())
        target_dict, note_events, _ = processor.process(
            start_time=0.0,
            midi_events_time=np.array([0.5], dtype=np.float64),
            midi_events=np.array([
                'note_on channel=0 note=60 velocity=64 time=0',
            ]),
            extend_pedal=False,
        )

        self.assertEqual(len(note_events), 1)
        self.assertEqual(note_events[0]['midi_note'], 60)
        self.assertAlmostEqual(note_events[0]['onset_time'], 0.5)
        self.assertAlmostEqual(note_events[0]['offset_time'], 2.0)
        self.assertEqual(target_dict['onset_roll'][50, 0], 1.0)
        self.assertEqual(target_dict['frame_roll'][50, 0], 1.0)

    def test_final_note_off_is_included_without_end_sentinel(self):
        processor = TargetProcessor(segment_seconds=2.0, cfg=_postprocessor_cfg())
        target_dict, note_events, _ = processor.process(
            start_time=0.0,
            midi_events_time=np.array([0.0, 1.0], dtype=np.float64),
            midi_events=np.array([
                'note_on channel=0 note=60 velocity=64 time=0',
                'note_off channel=0 note=60 velocity=0 time=0',
            ]),
            extend_pedal=False,
        )

        self.assertEqual(len(note_events), 1)
        self.assertEqual(note_events[0]['midi_note'], 60)
        self.assertAlmostEqual(note_events[0]['onset_time'], 0.0)
        self.assertAlmostEqual(note_events[0]['offset_time'], 1.0)
        self.assertEqual(target_dict['onset_roll'][0, 0], 1.0)
        self.assertEqual(target_dict['offset_roll'][100, 0], 1.0)


if __name__ == "__main__":
    unittest.main()
