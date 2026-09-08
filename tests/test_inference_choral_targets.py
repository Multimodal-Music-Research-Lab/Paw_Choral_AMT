import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf


REPO_DIR = Path(__file__).resolve().parents[1]
for source_dir in (REPO_DIR / "src", REPO_DIR / "tools", REPO_DIR / "experiments"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import inference


def make_cfg(
    *,
    target_assignment="part_name",
    evaluation_reference_assignment="part_name",
):
    return SimpleNamespace(
        feature=SimpleNamespace(
            audio_feature="logmel",
            begin_note=21,
            classes_num=88,
            frames_per_second=10,
            sample_rate=16000,
            segment_seconds=2.0,
        ),
        model=SimpleNamespace(
            arch="pawct",
            mode="frame_onset_offset",
            name="auto",
            type=None,
        ),
        dataset=SimpleNamespace(
            eval_split="validation",
            test_set="youchorale",
        ),
        exp=SimpleNamespace(name_suffix=""),
        choral=SimpleNamespace(
            enable=True,
            num_voices=4,
            voice_names=["S", "A", "T", "B"],
            target_assignment=target_assignment,
            voice_assignment_method=None,
            evaluation_reference_assignment=evaluation_reference_assignment,
            evaluation_reference_duration_policy="strict",
            preserve_known_part_labels=True,
            assignment_module="heads",
            assignment_hidden_channels=64,
            assignment_rnn_hidden_size=128,
            voice_interaction_module="self_attn",
            voice_interaction_dim=192,
            voice_interaction_heads=8,
            voice_interaction_layers=2,
            voice_interaction_dropout=0.25,
            voice_assignment_part_penalty=2.0,
            voice_assignment_continuity_weight=0.35,
            voice_assignment_overlap_penalty=4.0,
            voice_assignment_range_mins=[60, 55, 48, 40],
            voice_assignment_range_maxs=[88, 79, 72, 67],
            voice_assignment_range_margin=2.0,
            voice_assignment_mask_penalty=8.0,
            oc_gap_decay_seconds=2.0,
            oc_overlap_tolerance_seconds=0.05,
            append_assignment_to_name=True,
        ),
    )


class BuildTotalDictTest(unittest.TestCase):
    def test_public_transcriber_name_keeps_legacy_alias(self):
        self.assertIs(inference.PianoTranscriber, inference.ChoralAMTTranscriber)

    def test_checkpoint_selector_accepts_best_or_iteration(self):
        self.assertEqual(
            inference.resolve_inference_checkpoint_path('/tmp/checkpoints', 'best'),
            '/tmp/checkpoints/best.pth',
        )
        self.assertEqual(
            inference.resolve_inference_checkpoint_path('/tmp/checkpoints', '199999'),
            '/tmp/checkpoints/199999_iteration.pth',
        )
        with self.assertRaisesRegex(ValueError, 'ckpt_iteration'):
            inference.resolve_inference_checkpoint_path('/tmp/checkpoints', '')

    def test_stale_probability_files_fail_instead_of_contaminating_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "expected.pkl").touch()
            (Path(tmpdir) / "old_split.pkl").touch()
            with self.assertRaisesRegex(RuntimeError, "stale"):
                inference.reject_stale_probability_files(tmpdir, ["expected"])

            os.unlink(Path(tmpdir) / "old_split.pkl")
            inference.reject_stale_probability_files(tmpdir, ["expected"])

    def test_transcriber_hashes_the_exact_checkpoint_snapshot_it_loads(self):
        cfg = make_cfg()
        cfg.exp.cuda = False
        checkpoint_a = b"checkpoint-snapshot-a"
        checkpoint_b = b"checkpoint-snapshot-b"
        report = SimpleNamespace(
            source_format="checkpoint:model",
            schema_version=2,
            deserialization_mode="weights_only",
            missing_keys=(),
            unexpected_keys=(),
            allowlisted_missing_keys=(),
            allowlisted_unexpected_keys=(),
            missing_key_allowlist=(),
            unexpected_key_allowlist=(),
        )
        model = torch.nn.Linear(3, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "best.pth"
            checkpoint_path.write_bytes(checkpoint_a)

            def fake_load(_model, snapshot, **_kwargs):
                self.assertEqual(snapshot, checkpoint_a)
                checkpoint_path.write_bytes(checkpoint_b)
                return {
                    "iteration": 10,
                    "schema_version": 2,
                    "target_assignment": {"method": "part_name"},
                }, report

            with (
                patch.object(inference, "build_model", return_value=model),
                patch.object(inference, "load_model_checkpoint", side_effect=fake_load),
                patch.object(inference, "validate_checkpoint_behavior"),
                patch.object(inference, "build_post_processor", return_value=object()),
            ):
                transcriber = inference.ChoralAMTTranscriber(
                    cfg,
                    str(checkpoint_path),
                )

        self.assertEqual(
            transcriber.checkpoint_identity["sha256"],
            hashlib.sha256(checkpoint_a).hexdigest(),
        )

    def test_formal_inference_rejects_allowlisted_missing_parameters(self):
        cfg = make_cfg()
        cfg.exp.cuda = False
        model = torch.nn.Linear(3, 2)
        report = SimpleNamespace(missing_keys=("bias",))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "best.pth"
            checkpoint_path.write_bytes(b"checkpoint-snapshot")

            with (
                patch.object(inference, "build_model", return_value=model),
                patch.object(
                    inference,
                    "load_model_checkpoint",
                    return_value=({"model": {}}, report),
                ),
                patch.object(inference, "validate_checkpoint_behavior") as behavior_audit,
                self.assertRaisesRegex(RuntimeError, "missing model parameters"),
            ):
                inference.ChoralAMTTranscriber(cfg, str(checkpoint_path))

        behavior_audit.assert_not_called()

        inference.reject_missing_checkpoint_parameters(
            model,
            SimpleNamespace(missing_keys=()),
        )

    def test_deframe_retains_the_single_final_endpoint_frame(self):
        transcriber = inference.ChoralAMTTranscriber.__new__(
            inference.ChoralAMTTranscriber
        )
        transcriber.segment_frames = 1001
        segments = np.zeros((3, 1001, 2), dtype=np.float32)
        segments[-1, -1, :] = [0.25, 0.75]

        deframed = transcriber.deframe(segments)
        stitched = transcriber.stitch_output(segments, valid_frames=2001)

        self.assertEqual(deframed.shape, (2001, 2))
        np.testing.assert_array_equal(deframed[-1], [0.25, 0.75])
        np.testing.assert_array_equal(stitched, deframed)
        with self.assertRaisesRegex(RuntimeError, "shorter"):
            transcriber.stitch_output(segments, valid_frames=2002)

    def test_build_total_dict_keeps_per_voice_rolls_and_masks(self):
        output_dict = {
            "frame_output": np.zeros((2, 88), dtype=np.float32),
            "voice_frame_output": np.zeros((2, 4, 88), dtype=np.float32),
        }
        target_dict = {
            "frame_roll": np.ones((2, 88), dtype=np.float32),
            "onset_roll": np.ones((2, 88), dtype=np.float32),
            "offset_roll": np.ones((2, 88), dtype=np.float32),
            "pedal_onset_roll": np.zeros((2,), dtype=np.float32),
            "pedal_offset_roll": np.zeros((2,), dtype=np.float32),
            "pedal_frame_roll": np.zeros((2,), dtype=np.float32),
            "frame_mask_roll": np.ones((2, 88), dtype=np.float32),
            "onset_mask_roll": np.ones((2, 88), dtype=np.float32),
            "offset_mask_roll": np.ones((2, 88), dtype=np.float32),
            "pedal_mask_roll": np.zeros((2,), dtype=np.float32),
            "voice_frame_roll": np.ones((2, 4, 88), dtype=np.float32),
            "voice_onset_roll": np.ones((2, 4, 88), dtype=np.float32),
            "voice_offset_roll": np.ones((2, 4, 88), dtype=np.float32),
            "voice_frame_mask_roll": np.ones((2, 4, 88), dtype=np.float32),
            "voice_onset_mask_roll": np.ones((2, 4, 88), dtype=np.float32),
            "voice_offset_mask_roll": np.ones((2, 4, 88), dtype=np.float32),
        }

        total_dict = inference.build_total_dict(
            output_dict=output_dict,
            target_dict=target_dict,
            ref_on_off_pairs=np.array([[0.0, 0.5]], dtype=np.float32),
            ref_midi_notes=np.array([60], dtype=np.int32),
            ref_pedal_on_off_pairs=np.zeros((0, 2), dtype=np.float32),
        )

        self.assertIn("voice_frame_roll", total_dict)
        self.assertIn("voice_onset_roll", total_dict)
        self.assertIn("voice_offset_roll", total_dict)
        self.assertIn("voice_frame_mask_roll", total_dict)
        self.assertIn("voice_onset_mask_roll", total_dict)
        self.assertIn("voice_offset_mask_roll", total_dict)
        self.assertEqual(total_dict["voice_frame_roll"].shape, (2, 4, 88))
        self.assertEqual(total_dict["voice_frame_mask_roll"].shape, (2, 4, 88))

    def test_build_choral_target_dict_creates_per_voice_rolls_for_range_prior(self):
        cfg = make_cfg(target_assignment="range_prior")
        note_bars = [
            {
                "S": [[80, 0, 0, 0.0, 1.0]],
                "B": [[45, 0, 0, 0.5, 1.5]],
            }
        ]

        target_dict = inference.build_choral_target_dict(
            cfg=cfg,
            note_bars=note_bars,
            segment_seconds=2.0,
            start_time=0.0,
        )

        self.assertEqual(target_dict["voice_frame_roll"].shape, (21, 4, 88))
        self.assertEqual(target_dict["voice_frame_mask_roll"].shape, (21, 4, 88))
        self.assertEqual(float(target_dict["voice_presence"][0]), 1.0)
        self.assertEqual(float(target_dict["voice_presence"][3]), 1.0)
        self.assertGreater(target_dict["voice_frame_roll"][:, 0, 80 - 21].sum(), 0.0)
        self.assertGreater(target_dict["voice_frame_roll"][:, 3, 45 - 21].sum(), 0.0)

    def test_new_default_config_ignores_null_legacy_assignment(self):
        cfg = OmegaConf.load(REPO_DIR / "src" / "config.yaml")
        cfg.choral.enable = True
        self.assertIsNone(cfg.choral.target_assignment)
        self.assertIsNone(cfg.choral.voice_assignment_method)
        self.assertEqual(inference.resolve_target_assignment(cfg), "part_name")

        target_dict = inference.build_choral_target_dict(
            cfg=cfg,
            note_bars=[{"S": [[72, 0, 0, 0.0, 1.0]]}],
            segment_seconds=2.0,
        )

        self.assertGreater(target_dict["voice_frame_roll"][:, 0, 72 - 21].sum(), 0.0)
        self.assertEqual(float(target_dict["voice_frame_roll"][:, 1:, :].sum()), 0.0)

    def test_formal_reference_rolls_do_not_use_training_range_prior(self):
        cfg = make_cfg(target_assignment="range_prior")
        note_bars = [
            {
                "unknown": [[76, 0, 0, 0.0, 1.0]],
                "B": [[45, 0, 0, 0.5, 1.5]],
            }
        ]

        training_targets = inference.build_choral_target_dict(
            cfg=cfg,
            note_bars=note_bars,
            segment_seconds=2.0,
        )
        reference_targets = inference.build_evaluation_choral_target_dict(
            cfg=cfg,
            note_bars=note_bars,
            segment_seconds=2.0,
        )

        self.assertGreater(training_targets["voice_frame_roll"][:, :, 76 - 21].sum(), 0.0)
        self.assertEqual(
            float(reference_targets["voice_frame_roll"][:, :, 76 - 21].sum()),
            0.0,
        )
        self.assertGreater(
            reference_targets["voice_frame_roll"][:, 3, 45 - 21].sum(),
            0.0,
        )

    def test_formal_reference_masks_do_not_inherit_packed_midi_gaps(self):
        cfg = make_cfg()
        packed_midi_mask = np.ones((21, 88), dtype=np.float32)
        packed_midi_mask[5:, :] = 0.0

        reference_targets = inference.build_evaluation_choral_target_dict(
            cfg=cfg,
            note_bars=[{"S": [[72, 0, 0, 0.5, 1.0]]}],
            segment_seconds=2.0,
            frame_mask_roll=packed_midi_mask,
            onset_mask_roll=packed_midi_mask,
            offset_mask_roll=packed_midi_mask,
        )

        note_idx = 72 - cfg.feature.begin_note
        self.assertEqual(float(reference_targets["frame_mask_roll"].min()), 1.0)
        self.assertEqual(
            float(reference_targets["voice_frame_mask_roll"].min()),
            1.0,
        )
        self.assertGreater(
            float(reference_targets["frame_roll"][5:, note_idx].sum()),
            0.0,
        )

    def test_formal_global_reference_uses_overlap_aware_canonical_union(self):
        cfg = make_cfg()
        note_idx = 60 - cfg.feature.begin_note
        reference_targets = inference.build_evaluation_choral_target_dict(
            cfg=cfg,
            note_bars=[{
                "S": [[60, 0, 0, 0.0, 2.0]],
                "A": [[60, 0, 0, 0.1, 1.0]],
            }],
            segment_seconds=2.0,
        )

        # The nested A attack is a rearticulation at frame 1. Its release at
        # 1.0 s is not the release of the still-active part-agnostic union.
        hip = reference_targets["onset_roll"][:, note_idx]
        self.assertEqual(float(hip.sum()), 2.0)
        self.assertEqual(float(hip[0]), 1.0)
        self.assertEqual(float(hip[1]), 1.0)
        self.assertEqual(float(reference_targets["offset_roll"][1, note_idx]), 1.0)
        self.assertEqual(float(reference_targets["offset_roll"][10, note_idx]), 0.0)
        np.testing.assert_array_equal(
            reference_targets["frame_roll"][:, note_idx],
            1.0,
        )

    def test_formal_reference_rejects_rp_or_oc_assignment(self):
        cfg = make_cfg(evaluation_reference_assignment="ordered_continuity")

        with self.assertRaisesRegex(ValueError, "evaluation_reference_assignment=part_name"):
            inference.build_evaluation_choral_target_dict(
                cfg=cfg,
                note_bars=[],
                segment_seconds=2.0,
            )

    def test_probs_provenance_records_checkpoint_and_assignment_semantics(self):
        cfg = make_cfg(target_assignment="oc")
        load_report = SimpleNamespace(
            as_dict=lambda: {
                "source_format": "checkpoint:model",
                "schema_version": 1,
                "deserialization_mode": "weights_only",
                "missing_keys": [],
                "unexpected_keys": [],
            }
        )
        transcriber = SimpleNamespace(
            checkpoint_load_report=load_report,
            checkpoint_target_assignment={"method": "ordered_continuity"},
            checkpoint_model_input_identity={
                "model": {"architecture": "pawct"},
                "feature": {"begin_note": 21},
            },
            checkpoint_identity={
                "filename": "best.pth",
                "sha256": "a" * 64,
                "iteration": 15000,
                "schema_version": 1,
            },
        )

        provenance = inference.build_inference_provenance(
            cfg,
            transcriber,
            evaluation_reference_assignment="part_name",
        )

        self.assertEqual(provenance["evaluation_reference_assignment"], "part_name")
        self.assertEqual(provenance["evaluation_reference_duration_policy"], "strict")
        self.assertEqual(provenance["evaluation_split"], "validation")
        self.assertEqual(provenance["checkpoint_identity"]["sha256"], "a" * 64)
        self.assertEqual(
            provenance["checkpoint_model_input_identity"]["feature"]["begin_note"],
            21,
        )
        self.assertFalse(
            provenance["runtime_model_behavior"][
                "allow_legacy_checkpoint_model_identity"
            ]
        )
        self.assertFalse(
            provenance["runtime_model_behavior"][
                "allow_unsafe_legacy_checkpoint_load"
            ]
        )
        self.assertEqual(
            provenance["runtime_model_behavior"]["voice_interaction_heads"],
            8,
        )
        self.assertEqual(
            provenance["runtime_model_behavior"]["assignment_module"],
            "heads",
        )
        self.assertEqual(
            provenance["runtime_model_behavior"]["voice_interaction_dim"],
            192,
        )
        self.assertEqual(
            provenance["runtime_model_behavior"]["voice_interaction_layers"],
            2,
        )
        self.assertEqual(
            provenance["runtime_model_behavior"]["voice_interaction_dropout"],
            0.25,
        )
        self.assertEqual(
            provenance["checkpoint_load_report"]["missing_keys"],
            [],
        )
        self.assertEqual(
            provenance["target_assignment"]["runtime_training_method"],
            "ordered_continuity",
        )
        self.assertEqual(
            provenance["target_assignment"]["model_assignment_module"],
            "heads",
        )
        self.assertEqual(
            provenance["target_assignment"]["checkpoint_metadata"]["method"],
            "ordered_continuity",
        )

        total_dict = inference.build_total_dict(
            output_dict={},
            target_dict={
                "frame_roll": np.zeros((1, 1)),
                "onset_roll": np.zeros((1, 1)),
                "offset_roll": np.zeros((1, 1)),
                "pedal_onset_roll": np.zeros((1,)),
                "pedal_offset_roll": np.zeros((1,)),
                "pedal_frame_roll": np.zeros((1,)),
                "frame_mask_roll": np.ones((1, 1)),
                "onset_mask_roll": np.ones((1, 1)),
                "offset_mask_roll": np.ones((1, 1)),
                "pedal_mask_roll": np.zeros((1,)),
            },
            ref_on_off_pairs=np.zeros((0, 2)),
            ref_midi_notes=np.zeros((0,)),
            ref_pedal_on_off_pairs=np.zeros((0, 2)),
            provenance=provenance,
        )
        self.assertEqual(total_dict["provenance"], provenance)
        self.assertIsNot(total_dict["provenance"], provenance)


if __name__ == "__main__":
    unittest.main()
