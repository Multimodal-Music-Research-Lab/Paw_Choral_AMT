import argparse
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
for source_dir in (REPO_DIR / "src", REPO_DIR / "tools", REPO_DIR / "experiments"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import visualize_choral_checkpoint_single_song as viz_song


class VisualizeChoralCheckpointSingleSongTest(unittest.TestCase):
    def test_parse_checkpoint_info_extracts_model_name_and_iteration(self):
        checkpoint_path = (
            "/tmp/workspaces/checkpoints/"
            "hpt_frame_onset_offset_logmel_sr16000_fps100_pro_choral_stream_foff_onsetx2_unionx1_oc_va_ordered_continuity/"
            "299999_iteration.pth"
        )

        model_name, iteration = viz_song.parse_checkpoint_info(checkpoint_path)

        self.assertEqual(
            model_name,
            "hpt_frame_onset_offset_logmel_sr16000_fps100_pro_choral_stream_foff_onsetx2_unionx1_oc_va_ordered_continuity",
        )
        self.assertEqual(iteration, 299999)

    def test_resolve_probs_dir_includes_evaluation_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            expected = workspace / "probs" / "youchorale" / "validation" / "pawct" / "299999_iteration"
            expected.mkdir(parents=True)

            args = argparse.Namespace(
                checkpoint_path="unused.pth",
                workspace=str(workspace),
                test_set="youchorale",
                prob_split="validation",
            )

            self.assertEqual(
                viz_song.resolve_probs_dir(args, model_name="pawct", iteration=299999),
                expected,
            )

    def test_reference_voice_tracks_from_note_bars_merges_split_parts(self):
        note_bars = [
            {
                "S1": [[72, 0.0, 1.0, 0.0, 1.0]],
                "S2": [[74, 1.0, 1.0, 1.0, 2.0]],
                "A1": [[69, 0.5, 1.0, 0.5, 1.5]],
                "T2": [[60, 0.75, 1.0, 0.75, 1.75]],
                "B1": [[48, 0.25, 1.0, 0.25, 1.25]],
                "measure": 4.0,
            }
        ]

        tracks = viz_song.reference_voice_tracks_from_note_bars(note_bars)

        self.assertEqual(sorted(tracks.keys()), ["A", "B", "S", "T"])
        self.assertEqual(len(tracks["S"]), 2)
        self.assertEqual(len(tracks["A"]), 1)
        self.assertEqual(len(tracks["T"]), 1)
        self.assertEqual(len(tracks["B"]), 1)
        self.assertEqual(tracks["S"][0]["midi_note"], 72)
        self.assertEqual(tracks["S"][1]["midi_note"], 74)

    def test_compute_merged_note_f1_from_events_returns_one_for_exact_match(self):
        ref_on_off_pairs = np.asarray([[0.0, 1.0], [1.5, 2.0]], dtype=np.float32)
        ref_midi_notes = np.asarray([60, 64], dtype=np.int32)
        est_events = [
            {"midi_note": 60, "onset_time": 0.0, "offset_time": 1.0},
            {"midi_note": 64, "onset_time": 1.5, "offset_time": 2.0},
        ]

        note_f1 = viz_song.compute_merged_note_f1_from_events(
            ref_on_off_pairs=ref_on_off_pairs,
            ref_midi_notes=ref_midi_notes,
            est_events=est_events,
            onset_tolerance=0.05,
            offset_min_tolerance=0.05,
        )

        self.assertAlmostEqual(note_f1, 1.0, places=6)

    def test_compute_merged_note_f1_from_events_tolerates_zero_length_reference_intervals(self):
        ref_on_off_pairs = np.asarray([[0.0, 0.0], [1.5, 2.0]], dtype=np.float32)
        ref_midi_notes = np.asarray([60, 64], dtype=np.int32)
        est_events = [
            {"midi_note": 60, "onset_time": 0.0, "offset_time": 0.1},
            {"midi_note": 64, "onset_time": 1.5, "offset_time": 2.0},
        ]

        note_f1 = viz_song.compute_merged_note_f1_from_events(
            ref_on_off_pairs=ref_on_off_pairs,
            ref_midi_notes=ref_midi_notes,
            est_events=est_events,
            onset_tolerance=0.05,
            offset_min_tolerance=0.05,
        )

        self.assertGreaterEqual(note_f1, 0.0)

    def test_draw_global_roll_adds_patches_for_each_event(self):
        fig, ax = plt.subplots()
        try:
            viz_song.draw_global_roll(
                ax=ax,
                note_events=[
                    {"midi_note": 60, "onset_time": 0.0, "offset_time": 1.0},
                    {"midi_note": 64, "onset_time": 1.5, "offset_time": 2.5},
                ],
                title="Global",
                alpha=0.8,
                line_width=0.7,
                xlim=(0.0, 3.0),
                ylim=(48, 72),
            )
            self.assertEqual(len(ax.patches), 2)
        finally:
            plt.close(fig)

    def test_apply_axis_style_sets_times_font_and_caps_ymax_at_88(self):
        fig, ax = plt.subplots()
        try:
            ax.set_ylim(40, 96)
            viz_song.apply_axis_style(ax, title="Demo", xlabel="Time (seconds)", ylabel="MIDI Pitch")

            self.assertAlmostEqual(ax.get_ylim()[1], 88.0)
            self.assertIn(viz_song.PLOT_FONT_FAMILY, ax.title.get_fontfamily())
            self.assertIn(viz_song.PLOT_FONT_FAMILY, ax.xaxis.label.get_fontfamily())
            self.assertIn(viz_song.PLOT_FONT_FAMILY, ax.yaxis.label.get_fontfamily())
            self.assertGreaterEqual(ax.title.get_fontsize(), 15)
            self.assertGreaterEqual(ax.xaxis.label.get_fontsize(), 14)
            self.assertGreaterEqual(ax.yaxis.label.get_fontsize(), 14)
            self.assertTrue(ax.get_xticklabels())
            self.assertIn(viz_song.PLOT_FONT_FAMILY, ax.get_xticklabels()[0].get_fontfamily())
            self.assertIn(viz_song.PLOT_FONT_FAMILY, ax.get_yticklabels()[0].get_fontfamily())
        finally:
            plt.close(fig)

    def test_resolve_plot_font_family_prefers_installed_times_like_font(self):
        self.assertIn(viz_song.PLOT_FONT_FAMILY, {"Times New Roman", "Nimbus Roman", "Liberation Serif", "serif"})

    def test_resolve_pitch_ylim_uses_paper_range(self):
        self.assertEqual(viz_song.resolve_pitch_ylim(), (40, 88))

    def test_build_color_legend_handles_uses_full_voice_names(self):
        handles = viz_song.build_color_legend_handles()

        self.assertEqual(
            [handle.get_label() for handle in handles],
            ["Soprano", "Alto", "Tenor", "Bass", "Part-agnostic"],
        )

    def test_save_color_legend_can_save_transparent_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "legend.png"
            captured = {}

            def fake_savefig(self, *args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs

            with mock.patch("matplotlib.figure.Figure.savefig", new=fake_savefig):
                viz_song.save_color_legend(output_path=output_path, dpi=600, transparent=True)

            self.assertEqual(captured["args"][0], output_path)
            self.assertEqual(captured["kwargs"]["dpi"], 600)
            self.assertTrue(captured["kwargs"]["transparent"])
            self.assertEqual(captured["kwargs"]["bbox_inches"], "tight")

    def test_plot_pair_does_not_draw_metrics_text_box(self):
        class Args:
            alpha = 0.8
            line_width = 0.7
            min_pitch = 36
            max_pitch = 90
            dpi = 120
            frame_threshold = 0.055
            onset_threshold = 0.010
            offset_threshold = 0.006

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "plot.png"
            captured = {}

            def fake_close(fig):
                captured["fig"] = fig

            with mock.patch("matplotlib.pyplot.close", side_effect=fake_close):
                viz_song.plot_pair(
                    gt_tracks={"S": [], "A": [], "T": [], "B": []},
                    pred_tracks={"S": [], "A": [], "T": [], "B": []},
                    global_events=[],
                    output_path=output_path,
                    stem="demo",
                    args=Args(),
                )

            fig = captured["fig"]
            try:
                self.assertEqual(len(fig.axes[-1].texts), 0)
            finally:
                plt.close(fig)

    def test_plot_pair_uses_short_panel_titles(self):
        class Args:
            alpha = 0.8
            line_width = 0.7
            min_pitch = 36
            max_pitch = 90
            dpi = 120

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "plot.png"
            captured = {}

            def fake_close(fig):
                captured["fig"] = fig

            with mock.patch("matplotlib.pyplot.close", side_effect=fake_close):
                viz_song.plot_pair(
                    gt_tracks={"S": [], "A": [], "T": [], "B": []},
                    pred_tracks={"S": [], "A": [], "T": [], "B": []},
                    global_events=[],
                    output_path=output_path,
                    stem="demo-song",
                    args=Args(),
                )

            fig = captured["fig"]
            try:
                self.assertEqual([ax.get_title() for ax in fig.axes], ["Ground Truth", "Global Transcription", "Part-aware Transcription"])
                self.assertIsNone(fig._suptitle)
            finally:
                plt.close(fig)

    def test_plot_pair_can_save_transparent_png(self):
        class Args:
            alpha = 0.8
            line_width = 0.7
            min_pitch = 36
            max_pitch = 90
            dpi = 300
            transparent = True

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "plot.png"
            captured = {}

            def fake_savefig(self, *args, **kwargs):
                captured["kwargs"] = kwargs

            with mock.patch("matplotlib.figure.Figure.savefig", new=fake_savefig):
                viz_song.plot_pair(
                    gt_tracks={"S": [], "A": [], "T": [], "B": []},
                    pred_tracks={"S": [], "A": [], "T": [], "B": []},
                    global_events=[],
                    output_path=output_path,
                    stem="demo-song",
                    args=Args(),
                )

            self.assertEqual(captured["kwargs"]["dpi"], 300)
            self.assertTrue(captured["kwargs"]["transparent"])


if __name__ == "__main__":
    unittest.main()
