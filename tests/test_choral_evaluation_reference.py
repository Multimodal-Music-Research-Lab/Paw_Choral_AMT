import pickle
import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import h5py


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from calculate_choral_scores import (
    ChoralScoreCalculator,
    _reference_events_by_voice,
    _reference_frame_roll_by_voice,
    print_merged_channel_metrics,
)


class ChoralEvaluationReferenceTest(unittest.TestCase):
    def _score_stubbed_song(
        self,
        note_bars,
        *,
        packed_reference_pitch,
        estimated_pitch,
        estimated_offset=1.0,
    ):
        total_dict = {
            # This packed reference is deliberately present to prove formal
            # scoring does not use an RP/OC pseudo-target or stale MIDI copy.
            "ref_on_off_pairs": np.array([[0.0, 1.0]], dtype=np.float32),
            "ref_midi_notes": np.array([packed_reference_pitch], dtype=np.int32),
        }
        calculator = ChoralScoreCalculator.__new__(ChoralScoreCalculator)
        calculator.cfg = SimpleNamespace(
            feature=SimpleNamespace(
                begin_note=21,
                classes_num=88,
                frames_per_second=100,
                sample_rate=16000,
            ),
            score=SimpleNamespace(
                onset_tolerance=0.05,
                offset_ratio=0.2,
                offset_min_tolerance=0.05,
            ),
        )
        calculator.post_processor = object()
        calculator.voice_thresholds = {
            voice_name: {
                "frame_threshold": 0.5,
                "onset_threshold": 0.5,
                "offset_threshold": 0.5,
            }
            for voice_name in ("S", "A", "T", "B")
        }
        calculator._load_probability_file = lambda *_args, **_kwargs: total_dict
        calculator._validate_probability_provenance = lambda *_args, **_kwargs: None
        calculator._prepare_formal_reference = (
            lambda note_bars, *_args, **_kwargs: (note_bars, {})
        )

        def decode_stub(_total_dict, voice_idx, _post_processor, thresholds=None):
            del thresholds
            if voice_idx != 0:
                return []
            return [{
                "midi_note": estimated_pitch,
                "onset_time": 0.0,
                "offset_time": estimated_offset,
            }]

        with tempfile.TemporaryDirectory() as tmpdir:
            note_path = Path(tmpdir) / "song.pkl"
            with note_path.open("wb") as note_file:
                pickle.dump(note_bars, note_file)
            with patch("calculate_choral_scores._decode_voice_events", side_effect=decode_stub):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    return calculator.calculate_score_per_song(
                        "/unused/song.pkl",
                        str(note_path),
                    )

    def test_reference_roll_uses_canonical_part_names_and_keeps_divisi(self):
        note_bars = [
            {
                "S1": [[72, 0, 0, 0.0, 0.2]],
                "S2": [[69, 0, 0, 0.1, 0.3]],
                "A": [[64, 0, 0, 0.0, 0.3]],
            }
        ]

        roll = _reference_frame_roll_by_voice(
            note_bars,
            frames_num=5,
            frames_per_second=10,
            begin_note=21,
            classes_num=88,
        )

        self.assertEqual(roll.shape, (5, 4, 88))
        self.assertEqual(float(roll[:, 0, 72 - 21].sum()), 3.0)
        self.assertEqual(float(roll[:, 0, 69 - 21].sum()), 3.0)
        self.assertEqual(float(roll[:, 1, 64 - 21].sum()), 4.0)
        self.assertEqual(float(roll[:, 2:, :].sum()), 0.0)

    def test_reference_roll_does_not_clip_out_of_window_notes_to_boundaries(self):
        note_bars = [{
            "S": [
                [72, 0, 0, -2.0, -1.0],
                [74, 0, 0, 100.0, 101.0],
            ]
        }]

        roll = _reference_frame_roll_by_voice(
            note_bars,
            frames_num=11,
            frames_per_second=10,
            begin_note=21,
            classes_num=88,
        )

        self.assertEqual(float(roll.sum()), 0.0)

    def test_reference_parser_does_not_guess_voice_from_arbitrary_prefix(self):
        note_bars = [{
            "Soprano_2": [[72, 0, 0, 0.0, 0.2]],
            "soloist": [[71, 0, 0, 0.0, 0.2]],
            "ambiguous": [[64, 0, 0, 0.0, 0.2]],
            "track": [[55, 0, 0, 0.0, 0.2]],
            "bassoon": [[48, 0, 0, 0.0, 0.2]],
        }]

        events = _reference_events_by_voice(note_bars, frames_per_second=100)

        self.assertEqual([event["midi_note"] for event in events["S"]], [72])
        self.assertEqual(events["A"], [])
        self.assertEqual(events["T"], [])
        self.assertEqual(events["B"], [])

    def test_formal_union_ignores_conflicting_packed_reference(self):
        result = self._score_stubbed_song(
            [{"S": [[72, 0, 0, 0.0, 1.0]]}],
            packed_reference_pitch=60,
            estimated_pitch=60,
        )

        self.assertEqual(result["union_note_f1"], 0.0)
        self.assertEqual(result["union_matched_note_count"], 0)

        output = StringIO()
        cfg = SimpleNamespace(exp=SimpleNamespace(ckpt_iteration="best"))
        with patch("calculate_choral_scores.get_model_name", return_value="pawct"):
            with redirect_stdout(output):
                print_merged_channel_metrics(
                    cfg,
                    {key: [value] for key, value in result.items()},
                )
        self.assertIn("Canonical SATB Union Evaluation", output.getvalue())
        self.assertIn("note_f1: 0.0000", output.getvalue())

    def test_exact_unison_divisi_matches_one_binary_voice_event(self):
        note_bars = [{
            "S1": [[72, 0, 0, 0.001, 0.5]],
            "S2": [[72, 0, 0, 0.004, 1.0]],
            "A": [[72, 0, 0, 0.001, 1.0]],
        }]

        reference = _reference_events_by_voice(note_bars, frames_per_second=100)
        self.assertEqual(len(reference["S"]), 1)
        self.assertEqual(reference["S"][0]["onset_time"], 0.001)
        self.assertEqual(reference["S"][0]["offset_time"], 1.0)
        self.assertEqual(len(reference["A"]), 1)

        result = self._score_stubbed_song(
            note_bars,
            packed_reference_pitch=72,
            estimated_pitch=72,
            estimated_offset=1.0,
        )
        self.assertEqual(result["S_f1"], 1.0)
        self.assertEqual(result["S_COnPOff"], 1.0)

    def test_formal_preflight_rejects_notes_outside_packed_audio_duration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            probs_dir = root / "probs"
            note_dir = root / "note"
            probs_dir.mkdir()
            note_dir.mkdir()
            hdf5_path = root / "song.h5"
            with h5py.File(hdf5_path, "w") as hdf5_file:
                hdf5_file.create_dataset(
                    "waveform",
                    data=np.zeros(16000, dtype=np.int16),
                )

            calculator = ChoralScoreCalculator.__new__(ChoralScoreCalculator)
            calculator.cfg = SimpleNamespace(
                feature=SimpleNamespace(
                    sample_rate=16000,
                    begin_note=21,
                    classes_num=88,
                )
            )
            calculator.probs_dir = str(probs_dir)
            calculator.note_dir = str(note_dir)
            calculator.probability_names = ("song.pkl",)
            calculator._load_probability_file = lambda *_args, **_kwargs: {}
            calculator.artifact_validator = SimpleNamespace(
                hdf5_path_for_probability=lambda _prob_path: str(hdf5_path)
            )

            invalid_references = (
                [{"S": [[72, 0, 0, -2.0, -1.0]]}],
                [{"S": [[72, 0, 0, 100.0, 101.0]]}],
            )
            for note_bars in invalid_references:
                with self.subTest(note_bars=note_bars):
                    with (note_dir / "song.pkl").open("wb") as note_file:
                        pickle.dump(note_bars, note_file)
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "outside_recording_duration",
                    ):
                        calculator.validate_all_probability_files()


if __name__ == "__main__":
    unittest.main()
