import io
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from main_iter import (
    build_training_semantics_signature,
    capture_dataset_rng_state,
    capture_rng_state,
    evaluation_data_identities,
    periodic_action_due,
    restore_dataset_rng_state,
    restore_rng_state,
    resolve_selection_metric,
    restore_selection_state,
    seed_everything,
    selection_improved,
    should_decay_learning_rate,
    stochastic_training_pipeline_enabled,
    training_has_remaining_updates,
    validate_resume_optimizer_state,
    validate_resume_device_state,
    validate_resume_evaluation_data,
    validate_resume_training_semantics,
)


class CheckpointSelectionTest(unittest.TestCase):
    @staticmethod
    def _training_cfg():
        return SimpleNamespace(
            model=SimpleNamespace(arch="pawct", mode="frame_onset", type=None),
            feature=SimpleNamespace(
                audio_feature="logmel",
                classes_num=88,
                segment_seconds=10.0,
                hop_seconds=1.0,
                sample_rate=16000,
                fft_size=2048,
                frames_per_second=100,
                use_augmentation=False,
                begin_note=21,
                velocity_scale=128,
                max_note_shift=0,
            ),
            dataset=SimpleNamespace(
                train_set="youchorale_pro",
                test_set="youchorale",
                cantoria_f0_source="crepe",
            ),
            choral=SimpleNamespace(
                enable=True,
                target_assignment="ordered_continuity",
                voice_assignment_method=None,
                voice_frame_loss_weight=1.0,
                voice_interaction_heads=4,
            ),
            exp=SimpleNamespace(
                workspace="unused-a",
                resume_iteration=0,
                total_iteration=200000,
                eval_iteration=5000,
                learning_rate=1e-4,
                optim="adam",
                decay=True,
                reduce_iteration=10000,
                loss_type="auto",
                random_seed=86,
                deterministic_cudnn=True,
                mini_data=False,
                cuda=True,
                debug=False,
                num_workers=0,
                batch_size=8,
                max_eval_batches=None,
                selection_dataset="youchorale",
                selection_metric="auto",
                selection_mode="max",
                early_stopping_patience_evals=None,
            ),
        )

    def test_training_semantics_signature_covers_each_update_semantics_group(self):
        baseline_cfg = self._training_cfg()
        baseline = build_training_semantics_signature(baseline_cfg)
        changes = [
            ("loss", lambda cfg: setattr(cfg.choral, "voice_frame_loss_weight", 0.0)),
            ("data", lambda cfg: setattr(cfg.dataset, "train_set", "youchorale")),
            ("feature", lambda cfg: setattr(cfg.feature, "frames_per_second", 50)),
            ("target", lambda cfg: setattr(cfg.choral, "target_assignment", "range_prior")),
            ("model", lambda cfg: setattr(cfg.choral, "voice_interaction_heads", 2)),
            ("optimization", lambda cfg: setattr(cfg.exp, "total_iteration", 210000)),
        ]
        for group, mutate in changes:
            with self.subTest(group=group):
                changed_cfg = copy.deepcopy(baseline_cfg)
                mutate(changed_cfg)
                changed = build_training_semantics_signature(changed_cfg)
                self.assertNotEqual(changed[group], baseline[group])

    def test_resume_rejects_semantic_mismatch_even_with_inexact_opt_in(self):
        saved_cfg = self._training_cfg()
        checkpoint = {
            "resolved_config": {},
        }
        reproducibility = {
            "training_semantics": build_training_semantics_signature(saved_cfg)
        }
        changed_cfg = copy.deepcopy(saved_cfg)
        changed_cfg.choral.voice_frame_loss_weight = 0.0

        for allow_inexact in (False, True):
            with self.subTest(allow_inexact=allow_inexact), self.assertRaisesRegex(
                ValueError,
                "loss.voice_frame_loss_weight",
            ):
                validate_resume_training_semantics(
                    changed_cfg,
                    checkpoint,
                    reproducibility,
                    allow_inexact_resume=allow_inexact,
                )

    def test_resume_can_derive_signature_from_resolved_config(self):
        cfg = self._training_cfg()
        from checkpointing import resolve_config

        validate_resume_training_semantics(
            cfg,
            {"resolved_config": resolve_config(cfg)},
            {},
            allow_inexact_resume=False,
        )

    def test_unknown_training_semantics_require_explicit_inexact_opt_in(self):
        cfg = self._training_cfg()
        with self.assertRaisesRegex(ValueError, "training-semantics"):
            validate_resume_training_semantics(
                cfg,
                {},
                {},
                allow_inexact_resume=False,
            )
        with self.assertLogs(level="WARNING") as captured:
            validate_resume_training_semantics(
                cfg,
                {},
                {},
                allow_inexact_resume=True,
            )
        self.assertIn("not an exact resume", " ".join(captured.output))

    def test_runtime_destinations_do_not_change_training_semantics(self):
        cfg = self._training_cfg()
        baseline = build_training_semantics_signature(cfg)
        cfg.exp.workspace = "unused-b"
        cfg.exp.resume_iteration = 1234
        self.assertEqual(build_training_semantics_signature(cfg), baseline)

    def test_evaluation_data_identity_is_bound_for_exact_resume(self):
        loaders = {
            "youchorale": SimpleNamespace(
                batch_sampler=SimpleNamespace(
                    dataset_type="youchorale",
                    split="validation",
                    segment_identity="sha-a",
                )
            )
        }
        saved = evaluation_data_identities(loaders)
        validate_resume_evaluation_data(
            saved,
            saved,
            allow_inexact_resume=False,
        )
        changed = copy.deepcopy(saved)
        changed["youchorale"]["segment_identity"] = "sha-b"
        for allow_inexact in (False, True):
            with self.subTest(allow_inexact=allow_inexact), self.assertRaisesRegex(
                ValueError,
                "Evaluation data",
            ):
                validate_resume_evaluation_data(
                    saved,
                    changed,
                    allow_inexact_resume=allow_inexact,
                )

    def test_unknown_evaluation_data_identity_requires_exact_opt_out(self):
        with self.assertRaisesRegex(ValueError, "evaluation-data"):
            validate_resume_evaluation_data(
                None,
                {},
                allow_inexact_resume=False,
            )
        with self.assertLogs(level="WARNING"):
            validate_resume_evaluation_data(
                None,
                {},
                allow_inexact_resume=True,
            )

    def test_seed_and_rng_snapshot_reproduce_all_main_process_streams(self):
        import random

        seed_everything(123)
        saved = capture_rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(3))

        random.random()
        np.random.rand()
        torch.rand(3)
        restore_rng_state(saved)
        actual = (random.random(), np.random.rand(), torch.rand(3))

        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_schema_v2_training_rng_payload_is_weights_only_loadable(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters())
        rng_state = capture_rng_state()
        payload = {
            "schema_version": 2,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "reproducibility": {
                "format_version": 2,
                "rng_state": rng_state,
                "train_sampler": {
                    "sampler_state_version": 2,
                    "segment_indexes": [1, 0],
                    "random_state": rng_state["numpy"],
                },
                "train_dataset_rng": {"dataset_numpy": rng_state["numpy"]},
            },
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)

        restored = torch.load(buffer, map_location="cpu", weights_only=True)

        self.assertEqual(restored["schema_version"], 2)
        self.assertIsInstance(
            restored["reproducibility"]["rng_state"]["numpy"]["keys"],
            list,
        )

    def test_dataset_and_augmentor_rng_state_round_trip(self):
        dataset = SimpleNamespace(
            random_state=np.random.RandomState(1),
            cfg=SimpleNamespace(
                feature=SimpleNamespace(
                    augmentor=SimpleNamespace(random_state=np.random.RandomState(2))
                )
            ),
        )
        saved = capture_dataset_rng_state(dataset)
        expected = (dataset.random_state.rand(), dataset.cfg.feature.augmentor.random_state.rand())
        dataset.random_state.rand()
        dataset.cfg.feature.augmentor.random_state.rand()

        restore_dataset_rng_state(dataset, saved)

        self.assertEqual(dataset.random_state.rand(), expected[0])
        self.assertEqual(dataset.cfg.feature.augmentor.random_state.rand(), expected[1])

    def test_exact_dataset_restore_rejects_missing_rng_stream(self):
        dataset = SimpleNamespace(
            random_state=np.random.RandomState(1),
            cfg=SimpleNamespace(feature=SimpleNamespace(augmentor=None)),
        )

        with self.assertRaisesRegex(ValueError, "missing dataset RNG state"):
            restore_dataset_rng_state(dataset, {})

    def test_stochastic_pipeline_detection_is_explicit(self):
        cfg = SimpleNamespace(
            feature=SimpleNamespace(use_augmentation=False, max_note_shift=0)
        )
        self.assertFalse(stochastic_training_pipeline_enabled(cfg))
        cfg.feature.use_augmentation = True
        self.assertTrue(stochastic_training_pipeline_enabled(cfg))

    def test_auto_metric_uses_part_aware_validation_for_pawct(self):
        cfg = SimpleNamespace(
            exp=SimpleNamespace(selection_metric="auto"),
            choral=SimpleNamespace(enable=True),
        )
        self.assertEqual(resolve_selection_metric(cfg), "mean_voice_frame_ap")

    def test_auto_metric_uses_merged_validation_for_pagct(self):
        cfg = SimpleNamespace(
            exp=SimpleNamespace(selection_metric="auto"),
            choral=SimpleNamespace(enable=False),
        )
        self.assertEqual(resolve_selection_metric(cfg), "frame_ap")

    def test_selection_direction_is_explicit(self):
        self.assertTrue(selection_improved(0.6, 0.5, "max"))
        self.assertFalse(selection_improved(0.4, 0.5, "max"))
        self.assertTrue(selection_improved(0.4, 0.5, "min"))
        with self.assertRaises(ValueError):
            selection_improved(0.4, 0.5, "sideways")
        self.assertFalse(selection_improved(float("nan"), None, "max"))
        with self.assertRaisesRegex(ValueError, "finite"):
            selection_improved(0.4, float("nan"), "max")

    def test_lr_decay_is_iteration_based_after_resume(self):
        cfg = SimpleNamespace(exp=SimpleNamespace(decay=True, reduce_iteration=10000))
        self.assertFalse(should_decay_learning_rate(cfg, 0))
        self.assertFalse(should_decay_learning_rate(cfg, 9999))
        self.assertTrue(should_decay_learning_rate(cfg, 10000))
        self.assertTrue(should_decay_learning_rate(cfg, 20000))

    def test_lr_decay_validates_interval(self):
        cfg = SimpleNamespace(exp=SimpleNamespace(decay=True, reduce_iteration=0))
        with self.assertRaisesRegex(ValueError, "reduce_iteration"):
            should_decay_learning_rate(cfg, 10000)

    def test_disabled_lr_decay_does_not_validate_unused_interval(self):
        cfg = SimpleNamespace(exp=SimpleNamespace(decay=False, reduce_iteration=0))
        self.assertFalse(should_decay_learning_rate(cfg, 10000))

    def test_resume_does_not_inject_an_extra_validation_or_save(self):
        self.assertTrue(periodic_action_due(0, 100, fresh_run=True))
        self.assertFalse(periodic_action_due(50, 100, fresh_run=False))
        self.assertTrue(periodic_action_due(99, 100, fresh_run=False))

    def test_exact_resume_requires_optimizer_state(self):
        validate_resume_optimizer_state(True, allow_inexact_resume=False)
        with self.assertRaisesRegex(ValueError, "no optimizer state"):
            validate_resume_optimizer_state(False, allow_inexact_resume=False)

        with self.assertLogs(level="WARNING") as captured:
            validate_resume_optimizer_state(False, allow_inexact_resume=True)
        self.assertIn("not an exact resume", " ".join(captured.output))

    def test_exact_resume_binds_runtime_device_and_cuda_rng(self):
        cpu_state = {"device_type": "cpu", "rng_state": {}}
        validate_resume_device_state(
            cpu_state,
            torch.device("cpu"),
            allow_inexact_resume=False,
        )
        for allow_inexact in (False, True):
            with self.subTest(allow_inexact=allow_inexact), self.assertRaisesRegex(
                ValueError,
                "device type",
            ):
                validate_resume_device_state(
                    cpu_state,
                    torch.device("cuda"),
                    allow_inexact_resume=allow_inexact,
                )

        with self.assertRaisesRegex(ValueError, "CUDA RNG"):
            validate_resume_device_state(
                {"device_type": "cuda", "rng_state": {}},
                torch.device("cuda"),
                allow_inexact_resume=False,
            )

    def test_unknown_resume_device_requires_explicit_inexact_opt_in(self):
        with self.assertRaisesRegex(ValueError, "device identity"):
            validate_resume_device_state(
                {},
                torch.device("cpu"),
                allow_inexact_resume=False,
            )
        with self.assertLogs(level="WARNING"):
            validate_resume_device_state(
                {},
                torch.device("cpu"),
                allow_inexact_resume=True,
            )

    def test_resume_selection_requires_same_objective(self):
        saved = {
            "dataset": "youchorale",
            "metric": "mean_voice_frame_ap",
            "mode": "max",
            "best_value": 0.42,
            "best_iteration": 9999,
            "evaluations_without_improvement": 2,
        }
        self.assertEqual(
            restore_selection_state(
                saved,
                dataset="youchorale",
                metric="mean_voice_frame_ap",
                mode="max",
            ),
            (0.42, 9999, 2),
        )
        with self.assertRaisesRegex(ValueError, "different objective"):
            restore_selection_state(
                saved,
                dataset="youchorale",
                metric="frame_ap",
                mode="max",
            )

    def test_resume_selection_rejects_nonfinite_best(self):
        saved = {
            "dataset": "youchorale",
            "metric": "mean_voice_frame_ap",
            "mode": "max",
            "best_value": float("nan"),
        }
        with self.assertRaisesRegex(ValueError, "finite"):
            restore_selection_state(
                saved,
                dataset="youchorale",
                metric="mean_voice_frame_ap",
                mode="max",
            )

    def test_exact_resume_requires_complete_selection_state(self):
        with self.assertRaisesRegex(ValueError, "complete checkpoint selection state"):
            restore_selection_state(
                None,
                dataset="youchorale",
                metric="mean_voice_frame_ap",
                mode="max",
                require_complete=True,
            )

        incomplete = {
            "dataset": "youchorale",
            "metric": "mean_voice_frame_ap",
            "mode": "max",
        }
        with self.assertRaisesRegex(ValueError, "missing="):
            restore_selection_state(
                incomplete,
                dataset="youchorale",
                metric="mean_voice_frame_ap",
                mode="max",
                require_complete=True,
            )

    def test_inexact_resume_explicitly_resets_invalid_selection_state(self):
        with self.assertLogs(level="WARNING") as captured:
            restored = restore_selection_state(
                None,
                dataset="youchorale",
                metric="mean_voice_frame_ap",
                mode="max",
                require_complete=True,
                allow_inexact_resume=True,
            )
        self.assertEqual(restored, (None, None, 0))
        self.assertIn("Resetting checkpoint-selection history", " ".join(captured.output))

    def test_completed_resume_has_no_remaining_update(self):
        self.assertTrue(training_has_remaining_updates(9999, 10000))
        self.assertFalse(training_has_remaining_updates(10000, 10000))
        self.assertFalse(training_has_remaining_updates(10001, 10000))


if __name__ == "__main__":
    unittest.main()
