import os
import pickle
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from probability_artifacts import (
    ProbabilityArtifactValidator,
    expected_split_probability_stems,
    sha256_file,
    validate_probability_manifest,
)
from checkpointing import (
    checkpoint_data_target_semantics,
    checkpoint_model_input_identity,
    checkpoint_target_semantics,
)
from calculate_scores import ScoreCalculator, _canonical_choral_reference


def make_cfg(workspace="/tmp/unused", require_probability_provenance=True):
    return SimpleNamespace(
        exp=SimpleNamespace(
            workspace=workspace,
            ckpt_iteration="best",
            require_probability_provenance=require_probability_provenance,
            allow_checkpoint_behavior_mismatch=False,
            allow_unknown_checkpoint_target_assignment=False,
            allow_checkpoint_target_assignment_mismatch=False,
            allow_legacy_checkpoint_model_identity=False,
            allow_unsafe_legacy_checkpoint_load=False,
        ),
        dataset=SimpleNamespace(
            test_set="youchorale",
            eval_split="validation",
        ),
        feature=SimpleNamespace(
            audio_feature="logmel",
            begin_note=21,
            classes_num=88,
            sample_rate=16000,
            frames_per_second=100,
            fft_size=2048,
            segment_seconds=10.0,
        ),
        model=SimpleNamespace(arch="pawct", mode="frame_onset_offset", name="auto"),
        choral=SimpleNamespace(
            enable=True,
            evaluation_reference_duration_policy="strict",
            target_assignment="ordered_continuity",
            voice_assignment_method=None,
            num_voices=4,
            voice_names=["S", "A", "T", "B"],
            use_presence_head=True,
            apply_presence_gate=False,
            assignment_module="heads",
            assignment_temperature=1.0,
            assignment_hidden_channels=64,
            assignment_rnn_hidden_size=128,
            voice_interaction_module="none",
            voice_interaction_dim=256,
            voice_interaction_heads=4,
            voice_interaction_layers=1,
            voice_interaction_dropout=0.1,
        ),
    )


def make_provenance(
    hdf5_path,
    stem,
    *,
    sha256="a" * 64,
    iteration=15000,
    run_id="run-1",
):
    target_semantics = checkpoint_target_semantics(make_cfg())
    return {
        "dataset_name": "youchorale",
        "model_name": "pawct_frame_onset_offset_oc",
        "evaluation_split": "validation",
        "evaluation_reference_assignment": "part_name",
        "evaluation_reference_duration_policy": "strict",
        "inference_run_id": run_id,
        "checkpoint_identity": {
            "filename": "best.pth",
            "sha256": sha256,
            "iteration": iteration,
            "schema_version": 1,
        },
        "checkpoint_load_report": {
            "is_compatible": True,
            "deserialization_mode": "weights_only",
            "missing_key_allowlist": [],
            "unexpected_key_allowlist": [],
        },
        "data_target_semantics": {
            "runtime": checkpoint_data_target_semantics(make_cfg()),
            "checkpoint": checkpoint_data_target_semantics(make_cfg()),
        },
        "checkpoint_model_input_identity": checkpoint_model_input_identity(make_cfg()),
        "runtime_model_behavior": {
            "num_voices": 4,
            "use_presence_head": True,
            "apply_presence_gate": False,
            "assignment_module": "heads",
            "assignment_temperature": 1.0,
            "assignment_hidden_channels": 64,
            "assignment_rnn_hidden_size": 128,
            "voice_interaction_module": "none",
            "voice_interaction_dim": 256,
            "voice_interaction_heads": 4,
            "voice_interaction_layers": 1,
            "voice_interaction_dropout": 0.1,
            "voice_names": ["S", "A", "T", "B"],
            "allow_checkpoint_behavior_mismatch": False,
            "allow_unknown_checkpoint_target_assignment": False,
            "allow_checkpoint_target_assignment_mismatch": False,
            "allow_legacy_checkpoint_model_identity": False,
            "allow_unsafe_legacy_checkpoint_load": False,
            "allow_legacy_canonical_union_semantics": False,
        },
        "target_assignment": {
            "runtime_training_method": "ordered_continuity",
            "runtime_semantics": deepcopy(target_semantics),
            "preserve_known_part_labels": True,
            "model_assignment_module": "heads",
            "checkpoint_metadata": deepcopy(target_semantics),
        },
        "source_artifacts": {
            "recording_stem": stem,
            "hdf5_sha256": sha256_file(hdf5_path),
            "reference_note_sha256": None,
        },
    }


def bare_validator(require_probability_provenance=True, expected_identity=None):
    validator = ProbabilityArtifactValidator.__new__(ProbabilityArtifactValidator)
    validator.cfg = make_cfg(
        require_probability_provenance=require_probability_provenance
    )
    validator.model_name = "pawct_frame_onset_offset_oc"
    validator.eval_split = "validation"
    validator.require_provenance = require_probability_provenance
    validator.choral_enabled = True
    validator.expected_target_assignment = "ordered_continuity"
    validator.expected_target_semantics = checkpoint_target_semantics(validator.cfg)
    validator.expected_data_target_semantics = checkpoint_data_target_semantics(
        validator.cfg
    )
    validator.expected_model_input_identity = checkpoint_model_input_identity(
        validator.cfg
    )
    validator.expected_voice_names = ["S", "A", "T", "B"]
    validator.expected_checkpoint_identity = expected_identity
    validator._provenance_presence = None
    validator._probability_checkpoint_identity = None
    validator._inference_run_id = None
    return validator


class ProbabilityManifestTest(unittest.TestCase):
    def test_pagct_scorer_rejects_artifact_ground_truth_that_differs_from_note_pkl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = make_cfg(tmpdir)
            hdf5_path = Path(tmpdir) / "song.h5"
            note_path = Path(tmpdir) / "song.pkl"
            with h5py.File(hdf5_path, "w") as hdf5_file:
                hdf5_file.create_dataset(
                    "waveform",
                    data=np.zeros((16000,), dtype=np.int16),
                )
            with note_path.open("wb") as note_file:
                pickle.dump([{"S": [[60, 0, 0, 0.0, 0.5]]}], note_file)

            frames_num = 101
            corrupted = {
                "ref_on_off_pairs": np.asarray([[0.0, 0.5]], dtype=np.float32),
                "ref_midi_notes": np.asarray([61], dtype=np.int32),
                "frame_roll": np.zeros((frames_num, 88), dtype=np.float32),
                "onset_roll": np.zeros((frames_num, 88), dtype=np.float32),
                "offset_roll": np.zeros((frames_num, 88), dtype=np.float32),
                "frame_mask_roll": np.ones((frames_num, 88), dtype=np.float32),
            }

            with self.assertRaisesRegex(RuntimeError, "canonical reference mismatch"):
                _canonical_choral_reference(
                    cfg,
                    str(hdf5_path),
                    str(note_path),
                    corrupted,
                )

    def test_hash_cache_detects_same_size_rewrite_with_restored_mtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.bin"
            path.write_bytes(b"first-payload")
            original_stat = path.stat()
            first_hash = sha256_file(path)

            path.write_bytes(b"other-payload")
            os.utime(
                path,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )

            self.assertNotEqual(sha256_file(path), first_hash)

    def test_expected_stems_come_from_requested_packed_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = make_cfg(tmpdir)
            hdf5_dir = Path(tmpdir) / "hdf5s" / "youchorale_sr16000"
            hdf5_dir.mkdir(parents=True)
            for stem, split in (
                ("song_b", "validation"),
                ("song_a", "validation"),
                ("song_c", "test"),
            ):
                with h5py.File(hdf5_dir / f"{stem}.h5", "w") as hf:
                    hf.attrs["split"] = split

            self.assertEqual(
                expected_split_probability_stems(cfg),
                ("song_a", "song_b"),
            )

    def test_duplicate_packed_stems_are_rejected_before_inference_or_scoring(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = make_cfg(tmpdir)
            hdf5_dir = Path(tmpdir) / "hdf5s" / "youchorale_sr16000"
            for subdir in ("one", "two"):
                path = hdf5_dir / subdir
                path.mkdir(parents=True)
                with h5py.File(path / "same.h5", "w") as hf:
                    hf.attrs["split"] = "validation"

            with self.assertRaisesRegex(RuntimeError, "duplicate recording stem"):
                expected_split_probability_stems(cfg)

    def test_manifest_rejects_missing_and_stale_probability_files(self):
        with self.assertRaisesRegex(RuntimeError, "missing=1"):
            validate_probability_manifest(["song_a", "song_b"], ["song_a"])
        with self.assertRaisesRegex(RuntimeError, "stale_or_extra=1"):
            validate_probability_manifest(["song_a"], ["song_a", "old_song"])
        validate_probability_manifest(["song_a", "song_b"], ["song_b", "song_a"])


class ProbabilityProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.hdf5_path = Path(self.tempdir.name) / "source.h5"
        self.hdf5_path.write_bytes(b"packed-source-v1")

    def tearDown(self):
        self.tempdir.cleanup()

    def validate(self, validator, provenance, stem="song_a"):
        validator.validate(
            {"provenance": provenance},
            f"{stem}.pkl",
            hdf5_path=self.hdf5_path,
        )

    def test_matching_probability_provenance_is_accepted(self):
        validator = bare_validator()
        self.validate(
            validator,
            make_provenance(self.hdf5_path, "song_a"),
            "song_a",
        )
        self.validate(
            validator,
            make_provenance(self.hdf5_path, "song_b"),
            "song_b",
        )

    def test_missing_or_stale_canonical_union_provenance_is_rejected(self):
        validator = bare_validator()
        provenance = make_provenance(self.hdf5_path, "song_a")
        provenance.pop("data_target_semantics")
        with self.assertRaisesRegex(RuntimeError, "Missing canonical-union"):
            self.validate(validator, provenance)

        validator = bare_validator()
        provenance = make_provenance(self.hdf5_path, "song_a")
        provenance["data_target_semantics"]["runtime"]["canonical_union"][
            "version"
        ] += 1
        with self.assertRaisesRegex(RuntimeError, "runtime mismatch"):
            self.validate(validator, provenance)

        validator = bare_validator()
        provenance = make_provenance(self.hdf5_path, "song_a")
        provenance["data_target_semantics"]["checkpoint"]["canonical_union"][
            "version"
        ] += 1
        with self.assertRaisesRegex(RuntimeError, "missing or stale"):
            self.validate(validator, provenance)

    def test_pre_event_level_oc_probability_semantics_are_rejected(self):
        validator = bare_validator()
        provenance = make_provenance(self.hdf5_path, "song_a")
        provenance["target_assignment"]["runtime_semantics"][
            "prior_loss_semantics_version"
        ] = 2
        with self.assertRaisesRegex(RuntimeError, "target-semantics mismatch"):
            self.validate(validator, provenance)

        validator = bare_validator()
        provenance = make_provenance(self.hdf5_path, "song_a")
        provenance["target_assignment"]["checkpoint_metadata"][
            "prior_loss_semantics_version"
        ] = 2
        with self.assertRaisesRegex(RuntimeError, "target-semantics mismatch"):
            self.validate(validator, provenance)

    def test_mixed_checkpoint_or_inference_run_outputs_are_rejected(self):
        validator = bare_validator()
        self.validate(validator, make_provenance(self.hdf5_path, "song_a"))
        with self.assertRaisesRegex(RuntimeError, "different checkpoints"):
            self.validate(
                validator,
                make_provenance(self.hdf5_path, "song_b", sha256="b" * 64),
                "song_b",
            )

        validator = bare_validator()
        self.validate(validator, make_provenance(self.hdf5_path, "song_a"))
        with self.assertRaisesRegex(RuntimeError, "different inference runs"):
            self.validate(
                validator,
                make_provenance(self.hdf5_path, "song_b", run_id="run-2"),
                "song_b",
            )

    def test_probability_checkpoint_is_bound_to_current_requested_file(self):
        validator = bare_validator(
            expected_identity={"filename": "best.pth", "sha256": "a" * 64}
        )
        provenance = make_provenance(self.hdf5_path, "song_a", sha256="b" * 64)
        with self.assertRaisesRegex(RuntimeError, "requested checkpoint"):
            self.validate(validator, provenance)

    def test_split_model_dataset_and_assignment_mismatches_are_rejected(self):
        for field, wrong_value in (
            ("dataset_name", "other_dataset"),
            ("model_name", "wrong_model"),
            ("evaluation_split", "test"),
            ("evaluation_reference_assignment", "ordered_continuity"),
            ("evaluation_reference_duration_policy", "invented"),
        ):
            validator = bare_validator()
            provenance = make_provenance(self.hdf5_path, "song_a")
            provenance[field] = wrong_value
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, field):
                self.validate(validator, provenance)

        validator = bare_validator()
        provenance = make_provenance(self.hdf5_path, "song_a")
        provenance["target_assignment"]["runtime_training_method"] = "range_prior"
        with self.assertRaisesRegex(RuntimeError, "target-assignment mismatch"):
            self.validate(validator, provenance)

        validator = bare_validator()
        provenance = make_provenance(self.hdf5_path, "song_a")
        provenance["target_assignment"]["checkpoint_metadata"]["method"] = "range_prior"
        with self.assertRaisesRegex(RuntimeError, "Checkpoint target assignment mismatch"):
            self.validate(validator, provenance)

        validator = bare_validator()
        provenance = make_provenance(self.hdf5_path, "song_a")
        provenance["target_assignment"]["checkpoint_metadata"][
            "preserve_known_part_labels"
        ] = False
        with self.assertRaisesRegex(RuntimeError, "preserve-known-label mismatch"):
            self.validate(validator, provenance)

        validator = bare_validator()
        provenance = make_provenance(self.hdf5_path, "song_a")
        provenance["checkpoint_model_input_identity"]["feature"]["begin_note"] = 22
        with self.assertRaisesRegex(RuntimeError, "model/input identity mismatch"):
            self.validate(validator, provenance)

    def test_runtime_voice_order_and_source_content_are_bound(self):
        validator = bare_validator()
        provenance = make_provenance(self.hdf5_path, "song_a")
        provenance["runtime_model_behavior"]["voice_names"] = ["A", "S", "T", "B"]
        with self.assertRaisesRegex(RuntimeError, "voice_names"):
            self.validate(validator, provenance)

        validator = bare_validator()
        provenance = make_provenance(self.hdf5_path, "song_a")
        self.hdf5_path.write_bytes(b"packed-source-v2-is-different")
        with self.assertRaisesRegex(RuntimeError, "HDF5 content changed"):
            self.validate(validator, provenance)

    def test_model_assignment_and_interaction_behavior_are_bound(self):
        mismatches = {
            "assignment_module": "va2_cnn",
            "assignment_hidden_channels": 32,
            "assignment_rnn_hidden_size": 96,
            "voice_interaction_module": "self_attn",
            "voice_interaction_dim": 128,
            "voice_interaction_heads": 8,
            "voice_interaction_layers": 2,
            "voice_interaction_dropout": 0.25,
        }
        for field, wrong_value in mismatches.items():
            validator = bare_validator()
            provenance = make_provenance(self.hdf5_path, "song_a")
            provenance["runtime_model_behavior"][field] = wrong_value
            with self.subTest(field=field), self.assertRaisesRegex(
                RuntimeError,
                field,
            ):
                self.validate(validator, provenance)

        validator = bare_validator()
        provenance = make_provenance(self.hdf5_path, "song_a")
        provenance["target_assignment"]["model_assignment_module"] = "va2_cnn"
        with self.assertRaisesRegex(RuntimeError, "model-assignment mismatch"):
            self.validate(validator, provenance)

    def test_probability_payload_shape_finiteness_and_range_are_validated(self):
        with h5py.File(self.hdf5_path, "w") as hdf5_file:
            hdf5_file.create_dataset(
                "waveform",
                data=np.zeros(16000, dtype=np.int16),
            )

        valid = np.full((101, 4, 88), 0.5, dtype=np.float32)
        validator = bare_validator()
        validator.validate(
            {
                "provenance": make_provenance(self.hdf5_path, "song_a"),
                "voice_frame_output": valid,
            },
            "song_a.pkl",
            hdf5_path=self.hdf5_path,
        )

        invalid_payloads = {
            r"shape mismatch": np.zeros((101, 3, 88), dtype=np.float32),
            r"non-finite": np.where(
                np.indices(valid.shape)[0] == 0,
                np.nan,
                valid,
            ),
            r"outside \[0, 1\]": np.full(
                (101, 4, 88),
                1.01,
                dtype=np.float32,
            ),
        }
        for error, payload in invalid_payloads.items():
            validator = bare_validator()
            with self.subTest(error=error), self.assertRaisesRegex(RuntimeError, error):
                validator.validate(
                    {
                        "provenance": make_provenance(self.hdf5_path, "song_a"),
                        "voice_frame_output": payload,
                    },
                    "song_a.pkl",
                    hdf5_path=self.hdf5_path,
                )

    def test_unsafe_checkpoint_load_provenance_requires_matching_opt_in(self):
        validator = bare_validator()
        provenance = make_provenance(self.hdf5_path, "song_a")
        provenance["checkpoint_load_report"][
            "deserialization_mode"
        ] = "unsafe_legacy_opt_in"
        with self.assertRaisesRegex(RuntimeError, "explicit opt-in"):
            self.validate(validator, provenance)

        validator = bare_validator()
        validator.cfg.exp.allow_unsafe_legacy_checkpoint_load = True
        provenance["runtime_model_behavior"][
            "allow_unsafe_legacy_checkpoint_load"
        ] = True
        self.validate(validator, provenance)

        validator = bare_validator()
        validator.cfg.exp.allow_unsafe_legacy_checkpoint_load = True
        provenance["checkpoint_identity"]["schema_version"] = 2
        with self.assertRaisesRegex(RuntimeError, "Schema-v2"):
            self.validate(validator, provenance)

    def test_satb_reference_content_is_bound_to_inference_artifact(self):
        reference_path = Path(self.tempdir.name) / "song_a.pkl"
        reference_path.write_bytes(b"canonical-reference-v1")
        provenance = make_provenance(self.hdf5_path, "song_a")
        provenance["source_artifacts"]["reference_note_sha256"] = sha256_file(
            reference_path
        )
        validator = bare_validator()
        validator.validate(
            {"provenance": provenance},
            "song_a.pkl",
            hdf5_path=self.hdf5_path,
            reference_path=reference_path,
        )

        reference_path.write_bytes(b"canonical-reference-v2-is-different")
        validator = bare_validator()
        with self.assertRaisesRegex(RuntimeError, "reference content changed"):
            validator.validate(
                {"provenance": provenance},
                "song_a.pkl",
                hdf5_path=self.hdf5_path,
                reference_path=reference_path,
            )

    def test_missing_provenance_requires_explicit_historical_opt_out(self):
        validator = bare_validator()
        with self.assertRaisesRegex(RuntimeError, "Missing probability provenance"):
            validator.validate({}, "song.pkl", hdf5_path=self.hdf5_path)

        historical_validator = bare_validator(require_probability_provenance=False)
        historical_validator.validate({}, "old_song.pkl", hdf5_path=self.hdf5_path)
        with self.assertRaisesRegex(RuntimeError, "mixes artifacts"):
            self.validate(
                historical_validator,
                make_provenance(self.hdf5_path, "new_song"),
                "new_song",
            )

    def test_generic_pagct_scorer_cannot_bypass_shared_validator(self):
        class RejectingValidator:
            def validate(self, *args, **kwargs):
                raise RuntimeError("provenance rejected")

        calculator = ScoreCalculator.__new__(ScoreCalculator)
        calculator.probs_dir = self.tempdir.name
        calculator.artifact_validator = RejectingValidator()
        prob_path = Path(self.tempdir.name) / "source.pkl"
        with prob_path.open("wb") as file_object:
            pickle.dump({}, file_object)

        with self.assertRaisesRegex(RuntimeError, "provenance rejected"):
            calculator.calculate_score_per_song((0, str(self.hdf5_path)))

    def test_generic_pagct_scorer_requires_validator_when_init_is_bypassed(self):
        calculator = ScoreCalculator.__new__(ScoreCalculator)
        calculator.probs_dir = self.tempdir.name
        prob_path = Path(self.tempdir.name) / "source.pkl"
        with prob_path.open("wb") as file_object:
            pickle.dump({}, file_object)

        with self.assertRaisesRegex(AttributeError, "artifact_validator"):
            calculator.calculate_score_per_song((0, str(self.hdf5_path)))


if __name__ == "__main__":
    unittest.main()
