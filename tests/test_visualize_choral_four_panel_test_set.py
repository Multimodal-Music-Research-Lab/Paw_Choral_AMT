import os
import sys
import tempfile
import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
for source_dir in (REPO_DIR / "src", REPO_DIR / "tools", REPO_DIR / "experiments"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import visualize_choral_four_panel_test_set as viz_batch


class VisualizeChoralFourPanelTestSetTest(unittest.TestCase):
    def test_load_split_stems_reads_requested_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir)
            (dataset_dir / "test.json").write_text('["song_a", "song_b"]', encoding="utf-8")
            (dataset_dir / "valid.json").write_text('["song_c"]', encoding="utf-8")

            stems = viz_batch.load_split_stems(dataset_dir=dataset_dir, split="test")

            self.assertEqual(stems, ["song_a", "song_b"])

    def test_resolve_output_dir_matches_single_song_convention(self):
        class Args:
            output_dir = ""
            workspace = ""

        checkpoint_path = Path(
            "/tmp/workspaces/checkpoints/"
            "pawct_frame_onset_offset_logmel_sr16000_fps100_pro_choral_stream_foff_onsetx2_unionx1_oc_va_ordered_continuity/"
            "299999_iteration.pth"
        )

        output_dir = viz_batch.resolve_output_dir(
            args=Args(),
            checkpoint_path=checkpoint_path,
            model_name="pawct_frame_onset_offset_logmel_sr16000_fps100_pro_choral_stream_foff_onsetx2_unionx1_oc_va_ordered_continuity",
            iteration=299999,
        )

        self.assertEqual(
            output_dir,
            Path("/tmp/workspaces/visualizations")
            / "pawct_frame_onset_offset_logmel_sr16000_fps100_pro_choral_stream_foff_onsetx2_unionx1_oc_va_ordered_continuity"
            / "299999_iteration",
        )


if __name__ == "__main__":
    unittest.main()
