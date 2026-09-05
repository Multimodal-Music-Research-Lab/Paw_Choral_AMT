import os
import sys
import tempfile
import unittest
from pathlib import Path

import mido

REPO_DIR = Path(__file__).resolve().parents[1]
for source_dir in (REPO_DIR / "src", REPO_DIR / "tools", REPO_DIR / "experiments"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import evaluate_pred_midi_voice_assignment


class EvaluateTrackSetsTest(unittest.TestCase):
    def test_evaluate_track_sets_perfect_match(self):
        tracks = {
            "S": [{"midi_note": 72, "onset_time": 0.00, "offset_time": 0.50, "velocity": 100}],
            "A": [{"midi_note": 67, "onset_time": 0.10, "offset_time": 0.60, "velocity": 100}],
            "T": [{"midi_note": 60, "onset_time": 0.20, "offset_time": 0.70, "velocity": 100}],
            "B": [{"midi_note": 48, "onset_time": 0.30, "offset_time": 0.80, "velocity": 100}],
        }

        metrics = evaluate_pred_midi_voice_assignment.evaluate_track_sets(tracks, tracks)

        self.assertAlmostEqual(metrics["S_note_f1_50ms"], 1.0, places=6)
        self.assertAlmostEqual(metrics["A_note_f1_100ms"], 1.0, places=6)
        self.assertAlmostEqual(metrics["T_frame_f1"], 1.0, places=6)
        self.assertAlmostEqual(metrics["B_frame_f1"], 1.0, places=6)
        self.assertAlmostEqual(metrics["conventional_note_f1_50ms"], 1.0, places=6)
        self.assertAlmostEqual(metrics["conventional_note_f1_100ms"], 1.0, places=6)
        self.assertAlmostEqual(metrics["conventional_frame_f1"], 1.0, places=6)

    def test_evaluate_track_sets_swapped_voices_keep_conventional_match(self):
        ref_tracks = {
            "S": [{"midi_note": 72, "onset_time": 0.00, "offset_time": 0.50, "velocity": 100}],
            "A": [{"midi_note": 67, "onset_time": 0.10, "offset_time": 0.60, "velocity": 100}],
            "T": [],
            "B": [],
        }
        pred_tracks = {
            "S": list(ref_tracks["A"]),
            "A": list(ref_tracks["S"]),
            "T": [],
            "B": [],
        }

        metrics = evaluate_pred_midi_voice_assignment.evaluate_track_sets(ref_tracks, pred_tracks)

        self.assertAlmostEqual(metrics["conventional_note_f1_50ms"], 1.0, places=6)
        self.assertAlmostEqual(metrics["conventional_note_f1_100ms"], 1.0, places=6)
        self.assertAlmostEqual(metrics["conventional_frame_f1"], 1.0, places=6)
        self.assertAlmostEqual(metrics["S_note_f1_50ms"], 0.0, places=6)
        self.assertAlmostEqual(metrics["A_note_f1_100ms"], 0.0, places=6)
        self.assertAlmostEqual(metrics["S_frame_f1"], 0.0, places=6)
        self.assertAlmostEqual(metrics["A_frame_f1"], 0.0, places=6)


class ResolvePredMidiPathTest(unittest.TestCase):
    def test_resolve_pred_midi_path_prefers_newest_duplicate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            older = root / "old_run" / "demo.mid"
            newer = root / "new_run" / "demo.mid"
            older.parent.mkdir(parents=True, exist_ok=True)
            newer.parent.mkdir(parents=True, exist_ok=True)
            older.write_bytes(b"old")
            newer.write_bytes(b"new")
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))

            resolved = evaluate_pred_midi_voice_assignment.resolve_pred_midi_path(root, "demo")

            self.assertEqual(resolved, newer)


class ReadPredMidiTest(unittest.TestCase):
    def test_local_mido_reader_is_tempo_aware(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = Path(tmpdir) / "one-note.mid"
            midi_file = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            midi_file.tracks.append(track)
            track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
            track.append(mido.Message("note_on", note=60, velocity=100, channel=0, time=0))
            track.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=480))
            midi_file.save(midi_path)

            notes = evaluate_pred_midi_voice_assignment.load_pred_notes_from_midi(midi_path)

            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0].pitch, 60)
            self.assertAlmostEqual(notes[0].onset, 0.0, places=6)
            self.assertAlmostEqual(notes[0].offset, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
