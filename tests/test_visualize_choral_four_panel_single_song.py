import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt


REPO_DIR = Path(__file__).resolve().parents[1]
for source_dir in (REPO_DIR / "src", REPO_DIR / "tools", REPO_DIR / "experiments"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import visualize_choral_four_panel_single_song as viz_four


class VisualizeChoralFourPanelSingleSongTest(unittest.TestCase):
    def test_plot_four_panel_uses_expected_titles_and_40_second_window(self):
        class Args:
            alpha = 0.82
            line_width = 0.8
            dpi = 180
            transparent = False
            start_sec = 0.0
            end_sec = 40.0

        gt_tracks = {
            "S": [{"midi_note": 72, "onset_time": 2.0, "offset_time": 3.0, "velocity": 90}],
            "A": [],
            "T": [],
            "B": [],
        }
        global_events = [
            {"midi_note": 67, "onset_time": 5.0, "offset_time": 6.0, "velocity": 90},
            {"midi_note": 69, "onset_time": 45.0, "offset_time": 46.0, "velocity": 90},
        ]
        part_aware_tracks = {
            "S": [{"midi_note": 71, "onset_time": 10.0, "offset_time": 12.0, "velocity": 90}],
            "A": [],
            "T": [],
            "B": [],
        }
        va_tracks = {
            "S": [],
            "A": [{"midi_note": 60, "onset_time": 15.0, "offset_time": 16.0, "velocity": 90}],
            "T": [],
            "B": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "plot.png"
            captured = {}

            def fake_close(fig):
                captured["fig"] = fig

            with mock.patch("matplotlib.pyplot.close", side_effect=fake_close):
                viz_four.plot_four_panel(
                    gt_tracks=gt_tracks,
                    global_events=global_events,
                    part_aware_tracks=part_aware_tracks,
                    va_tracks=va_tracks,
                    global_note_f1=0.1234,
                    part_aware_note_f1=0.2345,
                    va_note_f1=0.3456,
                    output_path=output_path,
                    stem="demo-song",
                    args=Args(),
                )

            fig = captured["fig"]
            try:
                self.assertEqual(
                    [ax.get_title() for ax in fig.axes],
                    [
                        "Ground Truth",
                        "PawCT | note F1@50ms=0.2345",
                        "PagCT | note F1@50ms=0.1234",
                        "PagCT + Post-VA | note F1@50ms=0.3456",
                    ],
                )
                self.assertEqual(len(fig.axes), 4)
                width, height = fig.get_size_inches()
                self.assertAlmostEqual(width, 18.0, places=2)
                self.assertAlmostEqual(height, 6.4, places=2)
                for ax in fig.axes:
                    self.assertAlmostEqual(ax.get_xlim()[0], 0.0)
                    self.assertAlmostEqual(ax.get_xlim()[1], 40.0)
                    self.assertIsNone(ax.get_legend())
            finally:
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
