import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR / 'src') not in sys.path:
    sys.path.insert(0, str(REPO_DIR / 'src'))

from checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointBehaviorError,
    CheckpointCompatibilityError,
    CheckpointFormatError,
    CheckpointIdentityError,
    UnsafeLegacyCheckpointError,
    build_checkpoint_metadata,
    checkpoint_compatibility_allowlists,
    checkpoint_data_target_semantics,
    checkpoint_model_input_identity,
    checkpoint_target_semantics,
    format_checkpoint_load_report,
    load_model_checkpoint,
    resolve_config,
    validate_checkpoint_behavior,
)


def add_runtime_identity_config(cfg):
    cfg.model = SimpleNamespace(
        arch='pawct',
        mode='frame_onset_offset',
        name='unit-test-model',
    )
    cfg.feature = SimpleNamespace(
        audio_feature='logmel',
        begin_note=21,
        classes_num=88,
        sample_rate=16000,
        frames_per_second=100,
        fft_size=2048,
        segment_seconds=10.0,
    )
    return cfg


def versioned_checkpoint(cfg, **contents):
    checkpoint = {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'model_input_identity': checkpoint_model_input_identity(cfg),
    }
    checkpoint.update(contents)
    return checkpoint


class CheckpointLoadingTest(unittest.TestCase):
    def make_state(self):
        source = torch.nn.Linear(3, 2)
        with torch.no_grad():
            source.weight.fill_(0.25)
            source.bias.fill_(-0.5)
        return source.state_dict()

    def test_loads_legacy_model_container_with_exact_keys(self):
        target = torch.nn.Linear(3, 2)

        payload, report = load_model_checkpoint(target, {'model': self.make_state()})

        self.assertEqual(report.source_format, 'checkpoint:model')
        self.assertTrue(report.is_legacy)
        self.assertEqual(report.missing_keys, ())
        self.assertEqual(report.unexpected_keys, ())
        self.assertTrue(torch.equal(target.weight, payload['model']['weight']))
        self.assertIn('missing_keys=[]', format_checkpoint_load_report(report))
        self.assertIn('unexpected_keys=[]', format_checkpoint_load_report(report))

    def test_loads_raw_legacy_state_dict(self):
        target = torch.nn.Linear(3, 2)
        state = self.make_state()

        payload, report = load_model_checkpoint(target, state)

        self.assertEqual(report.source_format, 'raw_state_dict')
        self.assertIs(payload, state)
        self.assertEqual(report.schema_version, None)

    def test_loads_versioned_checkpoint_from_path(self):
        target = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'model.pth'
            torch.save(
                {
                    'schema_version': CHECKPOINT_SCHEMA_VERSION,
                    'model': self.make_state(),
                },
                checkpoint_path,
            )

            _, report = load_model_checkpoint(target, checkpoint_path, map_location='cpu')

        self.assertEqual(report.source_format, 'checkpoint:model')
        self.assertEqual(report.schema_version, CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(report.deserialization_mode, 'weights_only')
        self.assertFalse(report.is_legacy)

    def test_loads_versioned_checkpoint_from_serialized_snapshot(self):
        target = torch.nn.Linear(3, 2)
        serialized = io.BytesIO()
        torch.save(
            {
                'schema_version': CHECKPOINT_SCHEMA_VERSION,
                'model': self.make_state(),
            },
            serialized,
        )

        _, report = load_model_checkpoint(
            target,
            serialized.getvalue(),
            map_location='cpu',
        )

        self.assertEqual(report.source_format, 'checkpoint:model')
        self.assertEqual(report.schema_version, CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(report.deserialization_mode, 'weights_only')

    def test_unsafe_legacy_pickle_requires_independent_explicit_opt_in(self):
        target = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'legacy-training.pth'
            torch.save(
                {
                    'model': self.make_state(),
                    'legacy_numpy_rng': np.random.get_state(),
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(
                UnsafeLegacyCheckpointError,
                'No unsafe fallback was attempted',
            ):
                load_model_checkpoint(target, checkpoint_path, map_location='cpu')

            _, report = load_model_checkpoint(
                target,
                checkpoint_path,
                map_location='cpu',
                allow_unsafe_legacy_load=True,
            )

        self.assertEqual(report.deserialization_mode, 'unsafe_legacy_opt_in')
        self.assertTrue(report.is_legacy)

    def test_unsafe_legacy_pickle_can_load_from_the_audited_snapshot(self):
        target = torch.nn.Linear(3, 2)
        serialized = io.BytesIO()
        torch.save(
            {
                'model': self.make_state(),
                'legacy_numpy_rng': np.random.get_state(),
            },
            serialized,
        )

        _, report = load_model_checkpoint(
            target,
            serialized.getvalue(),
            map_location='cpu',
            allow_unsafe_legacy_load=True,
        )

        self.assertEqual(report.deserialization_mode, 'unsafe_legacy_opt_in')
        self.assertTrue(report.is_legacy)

    def test_schema_v2_cannot_be_reclassified_as_unsafe_legacy(self):
        target = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'invalid-v2.pth'
            torch.save(
                {
                    'schema_version': CHECKPOINT_SCHEMA_VERSION,
                    'model': self.make_state(),
                    'forbidden_numpy_array': np.arange(3),
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(
                UnsafeLegacyCheckpointError,
                'schema-v2-or-newer',
            ):
                load_model_checkpoint(
                    target,
                    checkpoint_path,
                    map_location='cpu',
                    allow_unsafe_legacy_load=True,
                )

    def test_rejects_unknown_future_schema_before_loading(self):
        target = torch.nn.Linear(3, 2)
        original_weight = target.weight.detach().clone()
        with self.assertRaisesRegex(CheckpointFormatError, "newer"):
            load_model_checkpoint(
                target,
                {
                    'schema_version': CHECKPOINT_SCHEMA_VERSION + 1,
                    'model': self.make_state(),
                },
            )
        self.assertTrue(torch.equal(target.weight, original_weight))

    def test_missing_key_fails_by_default_and_is_reported(self):
        target = torch.nn.Linear(3, 2)
        incomplete_state = {'weight': self.make_state()['weight']}

        with self.assertRaises(CheckpointCompatibilityError) as raised:
            load_model_checkpoint(target, incomplete_state)

        self.assertEqual(raised.exception.report.missing_keys, ('bias',))
        self.assertEqual(raised.exception.report.unapproved_missing_keys, ('bias',))
        self.assertIn("missing_keys=['bias']", str(raised.exception))

    def test_unexpected_key_fails_by_default_and_is_reported(self):
        target = torch.nn.Linear(3, 2)
        state = dict(self.make_state())
        state['legacy_head.weight'] = torch.ones(1)

        with self.assertRaises(CheckpointCompatibilityError) as raised:
            load_model_checkpoint(target, state)

        self.assertEqual(raised.exception.report.unexpected_keys, ('legacy_head.weight',))
        self.assertEqual(
            raised.exception.report.unapproved_unexpected_keys,
            ('legacy_head.weight',),
        )

    def test_only_explicit_allowlist_patterns_permit_key_differences(self):
        target = torch.nn.Linear(3, 2)
        state = {'weight': self.make_state()['weight'], 'legacy_head.weight': torch.ones(1)}

        _, report = load_model_checkpoint(
            target,
            state,
            allowed_missing_keys=['b*'],
            allowed_unexpected_keys=['legacy_head.*'],
        )

        self.assertTrue(report.is_compatible)
        self.assertEqual(report.allowlisted_missing_keys, ('bias',))
        self.assertEqual(report.allowlisted_unexpected_keys, ('legacy_head.weight',))
        self.assertEqual(report.unapproved_missing_keys, ())
        self.assertEqual(report.unapproved_unexpected_keys, ())

    def test_rejects_non_state_checkpoint_mapping(self):
        with self.assertRaises(CheckpointFormatError):
            load_model_checkpoint(torch.nn.Linear(3, 2), {'iteration': 10})


class CheckpointMetadataTest(unittest.TestCase):
    def make_cfg(self):
        return OmegaConf.create(
            {
                'wandb': {'name': '${model.arch}-${model.mode}'},
                'exp': {'workspace': './workspace'},
                'model': {'arch': 'pawct', 'mode': 'frame_onset_offset'},
                'feature': {
                    'audio_feature': 'logmel',
                    'begin_note': 21,
                    'classes_num': 88,
                    'sample_rate': 16000,
                    'frames_per_second': 100,
                    'fft_size': 2048,
                    'segment_seconds': 10.0,
                },
                'choral': {
                    'enable': True,
                    'num_voices': 4,
                    'voice_names': ['S', 'A', 'T', 'B'],
                    'target_assignment': 'oc',
                    'voice_assignment_method': 'range_prior',
                    'preserve_known_part_labels': True,
                    'evaluation_reference_assignment': 'part_name',
                    'assignment_module': 'va2_cnn',
                    'voice_interaction_module': 'self_attn',
                    'voice_interaction_heads': 8,
                    'voice_interaction_dropout': 0.25,
                    'voice_assignment_range_mins': [60, 55, 48, 40],
                    'voice_assignment_range_maxs': [88, 79, 72, 67],
                    'voice_assignment_range_margin': 2.0,
                    'voice_assignment_part_penalty': 2.0,
                    'voice_assignment_continuity_weight': 0.35,
                    'voice_assignment_overlap_penalty': 4.0,
                    'voice_assignment_mask_penalty': 8.0,
                },
            }
        )

    def test_metadata_contains_resolved_config_and_system_assignment(self):
        cfg = self.make_cfg()

        metadata = build_checkpoint_metadata(
            cfg,
            model_name='paper-model',
            model_arch='pawct',
            model_mode='frame_onset_offset',
            model_class='models.PawCT',
        )

        self.assertEqual(metadata['schema_version'], CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(metadata['resolved_config']['wandb']['name'], 'pawct-frame_onset_offset')
        self.assertEqual(metadata['model_system']['system'], 'PawCT')
        self.assertEqual(metadata['model_system']['model_class'], 'models.PawCT')
        self.assertEqual(metadata['model_system']['assignment_module'], 'va2_cnn')
        self.assertEqual(metadata['model_system']['assignment_temperature'], 1.0)
        self.assertEqual(metadata['model_system']['voice_interaction_heads'], 8)
        self.assertEqual(metadata['model_system']['voice_interaction_dropout'], 0.25)
        identity = metadata['model_input_identity']
        self.assertEqual(identity['model']['architecture'], 'pawct')
        self.assertEqual(identity['model']['task_mode'], 'frame_onset_offset')
        self.assertEqual(identity['model']['model_name'], 'paper-model')
        self.assertEqual(identity['model']['assignment_temperature'], 1.0)
        self.assertEqual(identity['model']['voice_interaction_heads'], 8)
        self.assertEqual(identity['model']['voice_interaction_dropout'], 0.25)
        self.assertEqual(identity['feature']['begin_note'], 21)
        self.assertEqual(identity['feature']['classes_num'], 88)
        self.assertEqual(identity['feature']['sample_rate'], 16000)
        self.assertEqual(identity['feature']['frames_per_second'], 100.0)
        self.assertEqual(identity['feature']['n_fft'], 2048)
        self.assertEqual(identity['feature']['hop_length'], 160)
        self.assertEqual(identity['feature']['mel_bins'], 229)
        self.assertEqual(identity['feature']['fmin'], 30.0)
        self.assertEqual(identity['feature']['fmax'], 8000.0)
        self.assertEqual(metadata['target_assignment']['method'], 'ordered_continuity')
        self.assertEqual(metadata['target_assignment']['configured_method'], 'oc')
        self.assertEqual(metadata['target_assignment']['legacy_method'], 'range_prior')
        self.assertTrue(metadata['target_assignment']['preserve_known_part_labels'])
        self.assertEqual(metadata['target_assignment']['voice_names'], ['S', 'A', 'T', 'B'])
        json.dumps(metadata)

    def test_pagct_choral_data_records_canonical_union_semantics(self):
        cfg = self.make_cfg()
        cfg.model.arch = 'pagct'
        cfg.choral.enable = False
        cfg.dataset = OmegaConf.create(
            {'train_set': 'youchorale', 'test_set': 'youchorale'}
        )

        metadata = build_checkpoint_metadata(
            cfg,
            model_name='pagct-frame-onset-offset',
            model_arch='pagct',
            model_mode='frame_onset_offset',
            model_class='models.PagCT',
        )

        self.assertEqual(
            metadata['data_target_semantics'],
            checkpoint_data_target_semantics(cfg),
        )
        self.assertEqual(
            metadata['data_target_semantics']['canonical_union']['source'],
            'note_pkl',
        )

    def test_pagct_rejects_missing_canonical_union_semantics_by_default(self):
        cfg = add_runtime_identity_config(
            SimpleNamespace(
                dataset=SimpleNamespace(
                    train_set='youchorale',
                    test_set='youchorale',
                ),
                choral=SimpleNamespace(enable=False),
                exp=SimpleNamespace(
                    allow_legacy_canonical_union_semantics=False,
                ),
            )
        )
        cfg.model.arch = 'pagct'
        checkpoint = versioned_checkpoint(cfg)

        with self.assertRaisesRegex(
            CheckpointBehaviorError,
            'canonical-union target semantics',
        ):
            validate_checkpoint_behavior(cfg, checkpoint)

        cfg.exp.allow_legacy_canonical_union_semantics = True
        validate_checkpoint_behavior(cfg, checkpoint)

    def test_begin_note_mismatch_is_rejected_without_generic_bypass(self):
        cfg = add_runtime_identity_config(
            SimpleNamespace(
                choral=SimpleNamespace(enable=False),
                exp=SimpleNamespace(
                    allow_checkpoint_behavior_mismatch=True,
                    allow_legacy_checkpoint_model_identity=True,
                ),
            )
        )
        checkpoint = versioned_checkpoint(cfg)
        cfg.feature.begin_note = 22

        with self.assertRaisesRegex(
            CheckpointIdentityError,
            r'feature\.begin_note',
        ):
            validate_checkpoint_behavior(cfg, checkpoint)

    def test_frontend_and_model_identity_mismatches_are_rejected(self):
        mismatch_cases = (
            ('classes_num', ('feature', 'classes_num'), 87),
            ('sample_rate', ('feature', 'sample_rate'), 22050),
            ('frames_per_second', ('feature', 'frames_per_second'), 86),
            ('n_fft', ('feature', 'fft_size'), 1024),
            ('architecture', ('model', 'arch'), 'pagct'),
            ('task_mode', ('model', 'mode'), 'frame_onset'),
            ('model_name', ('model', 'name'), 'different-model'),
        )
        for expected_field, path, replacement in mismatch_cases:
            cfg = add_runtime_identity_config(
                SimpleNamespace(
                    choral=SimpleNamespace(enable=False),
                    exp=SimpleNamespace(
                        allow_checkpoint_behavior_mismatch=True,
                        allow_legacy_checkpoint_model_identity=True,
                    ),
                )
            )
            checkpoint = versioned_checkpoint(cfg)
            setattr(getattr(cfg, path[0]), path[1], replacement)
            with self.subTest(field=expected_field), self.assertRaisesRegex(
                CheckpointIdentityError,
                expected_field,
            ):
                validate_checkpoint_behavior(cfg, checkpoint)

    def test_effective_mel_frontend_fields_are_strictly_checked(self):
        replacements = {
            'hop_length': 161,
            'mel_bins': 128,
            'fmin': 20.0,
            'fmax': 7900.0,
        }
        for field, replacement in replacements.items():
            cfg = add_runtime_identity_config(
                SimpleNamespace(
                    choral=SimpleNamespace(enable=False),
                    exp=SimpleNamespace(),
                )
            )
            checkpoint = versioned_checkpoint(cfg)
            checkpoint['model_input_identity']['feature'][field] = replacement
            with self.subTest(field=field), self.assertRaisesRegex(
                CheckpointIdentityError,
                field,
            ):
                validate_checkpoint_behavior(cfg, checkpoint)

    def test_legacy_identity_requires_its_own_explicit_opt_in(self):
        cfg = add_runtime_identity_config(
            SimpleNamespace(
                choral=SimpleNamespace(enable=False),
                exp=SimpleNamespace(
                    allow_checkpoint_behavior_mismatch=True,
                    allow_legacy_checkpoint_model_identity=False,
                ),
            )
        )
        legacy_checkpoint = {'model': {}}

        with self.assertRaisesRegex(
            CheckpointIdentityError,
            'allow_legacy_checkpoint_model_identity',
        ):
            validate_checkpoint_behavior(cfg, legacy_checkpoint)

        cfg.exp.allow_legacy_checkpoint_model_identity = True
        validate_checkpoint_behavior(cfg, legacy_checkpoint)

    def test_current_schema_missing_identity_is_corrupt_not_legacy(self):
        cfg = add_runtime_identity_config(
            SimpleNamespace(
                choral=SimpleNamespace(enable=False),
                exp=SimpleNamespace(
                    allow_legacy_checkpoint_model_identity=True,
                ),
            )
        )

        with self.assertRaisesRegex(
            CheckpointIdentityError,
            'missing required model_input_identity',
        ):
            validate_checkpoint_behavior(
                cfg,
                {'schema_version': CHECKPOINT_SCHEMA_VERSION, 'model': {}},
            )

    def test_metadata_marks_deprecated_assignment_key_as_legacy_semantics(self):
        cfg = self.make_cfg()
        cfg.choral.target_assignment = None
        cfg.choral.voice_assignment_method = 'ordered_continuity'

        metadata = build_checkpoint_metadata(
            cfg,
            model_name='historical-model',
            model_arch='pawct',
            model_mode='frame_onset_offset',
            model_class='models.PawCT',
        )

        self.assertEqual(
            metadata['target_assignment']['method'],
            'legacy_ordered_continuity',
        )

    def test_runtime_object_is_replaced_by_stable_type_marker(self):
        class RuntimeAugmentor:
            pass

        cfg = SimpleNamespace(feature=SimpleNamespace(augmentor=RuntimeAugmentor()))
        resolved = resolve_config(cfg)

        self.assertEqual(
            resolved['feature']['augmentor']['__python_type__'],
            f'{RuntimeAugmentor.__module__}.{RuntimeAugmentor.__qualname__}',
        )

    def test_compatibility_allowlists_are_absent_by_default_and_opt_in(self):
        default_cfg = SimpleNamespace(exp=SimpleNamespace())
        explicit_cfg = SimpleNamespace(
            exp=SimpleNamespace(
                checkpoint_allow_missing_keys=['legacy.*'],
                checkpoint_allow_unexpected_keys='old_head.*',
            )
        )

        self.assertEqual(checkpoint_compatibility_allowlists(default_cfg), ((), ()))
        self.assertEqual(
            checkpoint_compatibility_allowlists(explicit_cfg),
            (('legacy.*',), ('old_head.*',)),
        )

    def test_unversioned_pawct_requires_historical_presence_gate(self):
        cfg = SimpleNamespace(
            choral=SimpleNamespace(
                enable=True,
                use_presence_head=True,
                apply_presence_gate=False,
            ),
            exp=SimpleNamespace(
                allow_checkpoint_behavior_mismatch=False,
                allow_legacy_checkpoint_model_identity=True,
                allow_unknown_checkpoint_target_assignment=True,
            ),
        )

        with self.assertRaisesRegex(CheckpointBehaviorError, "parameter-free"):
            validate_checkpoint_behavior(cfg, {'model': {}})

        cfg.choral.apply_presence_gate = True
        validate_checkpoint_behavior(cfg, {'model': {}})

    def test_versioned_checkpoint_gate_must_match_runtime(self):
        cfg = add_runtime_identity_config(SimpleNamespace(
            choral=SimpleNamespace(
                enable=True,
                use_presence_head=True,
                apply_presence_gate=True,
            ),
            exp=SimpleNamespace(allow_checkpoint_behavior_mismatch=False),
        ))
        checkpoint = versioned_checkpoint(
            cfg,
            model_system={'apply_presence_gate': False},
            target_assignment={'method': 'part_name'},
        )

        with self.assertRaises(CheckpointBehaviorError):
            validate_checkpoint_behavior(cfg, checkpoint)

        cfg.exp.allow_checkpoint_behavior_mismatch = True
        validate_checkpoint_behavior(cfg, checkpoint)

    def test_voice_order_and_assignment_temperature_are_audited(self):
        cfg = add_runtime_identity_config(SimpleNamespace(
            choral=SimpleNamespace(
                enable=True,
                use_presence_head=True,
                apply_presence_gate=False,
                voice_names=['A', 'S', 'T', 'B'],
                assignment_temperature=0.5,
            ),
            exp=SimpleNamespace(allow_checkpoint_behavior_mismatch=False),
        ))
        checkpoint = versioned_checkpoint(
            cfg,
            model_system={
                'apply_presence_gate': False,
                'assignment_temperature': 1.0,
            },
            target_assignment={
                'method': 'part_name',
                'voice_names': ['S', 'A', 'T', 'B'],
            },
        )

        with self.assertRaises(CheckpointBehaviorError) as raised:
            validate_checkpoint_behavior(cfg, checkpoint)
        self.assertIn('voice_names', str(raised.exception))
        self.assertIn('assignment_temperature', str(raised.exception))

    def test_behavior_is_audited_without_presence_head(self):
        mismatches = {
            'apply_presence_gate': {
                'apply_presence_gate': True,
                'assignment_temperature': 0.5,
                'voice_names': ['S', 'A', 'T', 'B'],
            },
            'voice_names': {
                'apply_presence_gate': False,
                'assignment_temperature': 0.5,
                'voice_names': ['A', 'S', 'T', 'B'],
            },
            'assignment_temperature': {
                'apply_presence_gate': False,
                'assignment_temperature': 1.0,
                'voice_names': ['S', 'A', 'T', 'B'],
            },
        }
        for expected_field, checkpoint_values in mismatches.items():
            cfg = add_runtime_identity_config(SimpleNamespace(
                choral=SimpleNamespace(
                    enable=True,
                    use_presence_head=False,
                    apply_presence_gate=False,
                    voice_names=['S', 'A', 'T', 'B'],
                    assignment_temperature=0.5,
                    target_assignment='part_name',
                    voice_assignment_method=None,
                ),
                exp=SimpleNamespace(allow_checkpoint_behavior_mismatch=False),
            ))
            checkpoint = versioned_checkpoint(
                cfg,
                model_system={
                    'apply_presence_gate': checkpoint_values['apply_presence_gate'],
                    'assignment_temperature': checkpoint_values[
                        'assignment_temperature'
                    ],
                },
                target_assignment={
                    'method': 'part_name',
                    'voice_names': checkpoint_values['voice_names'],
                },
            )

            with self.subTest(field=expected_field), self.assertRaisesRegex(
                CheckpointBehaviorError,
                expected_field,
            ):
                validate_checkpoint_behavior(cfg, checkpoint)

    def test_same_shape_forward_settings_are_part_of_model_identity(self):
        mismatch_cases = (
            ('assignment_temperature', 0.5),
            ('voice_interaction_heads', 8),
            ('voice_interaction_dropout', 0.25),
        )
        for field, replacement in mismatch_cases:
            cfg = add_runtime_identity_config(SimpleNamespace(
                choral=SimpleNamespace(
                    enable=True,
                    use_presence_head=True,
                    target_assignment='part_name',
                    voice_assignment_method=None,
                    assignment_module='va2_cnn',
                    assignment_temperature=1.0,
                    voice_interaction_module='self_attn',
                    voice_interaction_heads=4,
                    voice_interaction_dropout=0.1,
                ),
                exp=SimpleNamespace(),
            ))
            checkpoint = versioned_checkpoint(cfg)
            setattr(cfg.choral, field, replacement)

            with self.subTest(field=field), self.assertRaisesRegex(
                CheckpointIdentityError,
                field,
            ):
                validate_checkpoint_behavior(cfg, checkpoint)

    def test_unknown_target_assignment_fails_closed_with_separate_opt_in(self):
        cfg = SimpleNamespace(
            choral=SimpleNamespace(
                enable=True,
                use_presence_head=False,
                target_assignment='ordered_continuity',
                voice_assignment_method=None,
            ),
            exp=SimpleNamespace(
                allow_legacy_checkpoint_model_identity=True,
                allow_unknown_checkpoint_target_assignment=False,
                allow_checkpoint_target_assignment_mismatch=True,
            ),
        )

        with self.assertRaisesRegex(
            CheckpointBehaviorError,
            'allow_unknown_checkpoint_target_assignment',
        ):
            validate_checkpoint_behavior(cfg, {'model': {}})

        cfg.exp.allow_unknown_checkpoint_target_assignment = True
        validate_checkpoint_behavior(cfg, {'model': {}})

    def test_versioned_target_assignment_must_match_runtime(self):
        cfg = add_runtime_identity_config(SimpleNamespace(
            choral=SimpleNamespace(
                enable=True,
                use_presence_head=False,
                target_assignment='ordered_continuity',
                voice_assignment_method=None,
            ),
            exp=SimpleNamespace(
                allow_checkpoint_behavior_mismatch=True,
                allow_unknown_checkpoint_target_assignment=True,
                allow_checkpoint_target_assignment_mismatch=False,
            ),
        ))
        checkpoint = versioned_checkpoint(
            cfg,
            target_assignment={'method': 'range_prior'},
        )

        with self.assertRaisesRegex(
            CheckpointBehaviorError,
            'allow_checkpoint_target_assignment_mismatch',
        ):
            validate_checkpoint_behavior(cfg, checkpoint)

        cfg.exp.allow_checkpoint_target_assignment_mismatch = True
        validate_checkpoint_behavior(cfg, checkpoint)

    def test_matching_versioned_target_assignment_is_accepted(self):
        cfg = add_runtime_identity_config(SimpleNamespace(
            choral=SimpleNamespace(
                enable=True,
                use_presence_head=False,
                target_assignment='oc',
                voice_assignment_method=None,
            ),
            exp=SimpleNamespace(),
        ))
        checkpoint = versioned_checkpoint(
            cfg,
            target_assignment=checkpoint_target_semantics(cfg),
        )

        validate_checkpoint_behavior(cfg, checkpoint)

    def test_pre_v4_prior_loss_semantics_are_rejected(self):
        cfg = add_runtime_identity_config(SimpleNamespace(
            choral=SimpleNamespace(
                enable=True,
                use_presence_head=False,
                target_assignment='ordered_continuity',
                voice_assignment_method=None,
            ),
            exp=SimpleNamespace(),
        ))
        target_semantics = checkpoint_target_semantics(cfg)
        self.assertEqual(target_semantics['prior_loss_semantics_version'], 4)
        target_semantics['prior_loss_semantics_version'] = 3
        checkpoint = versioned_checkpoint(
            cfg,
            target_assignment=target_semantics,
        )

        with self.assertRaisesRegex(
            CheckpointBehaviorError,
            'target-semantics mismatch',
        ):
            validate_checkpoint_behavior(cfg, checkpoint)

    def test_preserve_known_label_semantics_must_match(self):
        cfg = add_runtime_identity_config(SimpleNamespace(
            choral=SimpleNamespace(
                enable=True,
                use_presence_head=False,
                target_assignment='oc',
                voice_assignment_method=None,
                preserve_known_part_labels=True,
            ),
            exp=SimpleNamespace(
                allow_checkpoint_target_assignment_mismatch=False,
            ),
        ))
        target_semantics = checkpoint_target_semantics(cfg)
        target_semantics['preserve_known_part_labels'] = False
        checkpoint = versioned_checkpoint(
            cfg,
            target_assignment=target_semantics,
        )

        with self.assertRaisesRegex(CheckpointBehaviorError, 'preserve-known-label'):
            validate_checkpoint_behavior(cfg, checkpoint)

        cfg.exp.allow_checkpoint_target_assignment_mismatch = True
        validate_checkpoint_behavior(cfg, checkpoint)


if __name__ == '__main__':
    unittest.main()
