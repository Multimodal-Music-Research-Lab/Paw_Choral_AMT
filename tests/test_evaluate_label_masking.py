import contextlib
import io
import json
import pickle
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
for source_dir in (REPO_DIR / 'src', REPO_DIR / 'tools'):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import evaluate_label_masking as label_masking


def note(pitch, onset, offset):
    return [pitch, 0, 0, onset, offset]


class EvaluateLabelMaskingTest(unittest.TestCase):
    def make_dataset(self, root):
        dataset_dir = root / 'TinyYouChorale'
        note_dir = dataset_dir / 'note'
        note_dir.mkdir(parents=True)
        (dataset_dir / 'train.json').write_text(
            json.dumps(['recording_beta', 'recording_alpha']),
            encoding='utf-8',
        )

        for recording_id, shift in (
            ('recording_alpha', 0),
            ('recording_beta', 1),
        ):
            note_bars = [{
                'S1': [
                    note(74 + (index % 3), index * 0.1, index * 0.1 + 0.08)
                    for index in range(128)
                ],
                'S2': [
                    note(72 + shift, index * 0.1, index * 0.1 + 0.06)
                    for index in range(0, 128, 16)
                ],
                'A': [
                    note(67 + (index % 3), index * 0.1, index * 0.1 + 0.08)
                    for index in range(128)
                ],
                'T': [
                    note(59 + (index % 3), index * 0.1, index * 0.1 + 0.08)
                    for index in range(128)
                ],
                'B': [
                    note(47 + (index % 3), index * 0.1, index * 0.1 + 0.08)
                    for index in range(128)
                ],
            }]
            with (note_dir / f'{recording_id}.pkl').open('wb') as handle:
                pickle.dump(note_bars, handle)
        return dataset_dir

    def test_frozen_manifest_packed_filter_metrics_and_negative_control(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = self.make_dataset(root)
            manifest_path = root / 'frozen' / 'train.json'
            manifest_path.parent.mkdir()
            manifest_path.write_text(
                json.dumps(['recording_alpha'], indent=2) + '\n',
                encoding='utf-8',
            )
            packed_dir = root / 'packed'
            packed_dir.mkdir()
            (packed_dir / 'recording_alpha.h5').touch()
            (packed_dir / 'recording_beta.h5').touch()

            result = label_masking.evaluate_label_masking(
                dataset_dir,
                recording_manifest=manifest_path,
                packed_hdf5_dir=packed_dir,
            )
            repeated = label_masking.evaluate_label_masking(
                dataset_dir,
                recording_manifest=manifest_path,
                packed_hdf5_dir=packed_dir,
            )

        self.assertEqual(result, repeated)
        self.assertEqual(result['protocol']['mask_rates_percent'], [10, 25, 50])
        self.assertTrue(result['protocol']['test_tuning_prohibited'])
        self.assertFalse(result['protocol']['binary_head_projection'])
        self.assertEqual(result['dataset']['recordings'], 1)
        # All eight S2 divisi notes remain distinct evaluation events.
        self.assertEqual(result['dataset']['eligible_notes'], 4 * 128 + 8)
        self.assertEqual(result['dataset']['eligible_notes_by_voice']['S'], 136)
        self.assertEqual(
            result['dataset']['recording_manifest']['filename'],
            'train.json',
        )
        self.assertEqual(
            result['dataset']['packed_coverage']['intersection_recordings'],
            1,
        )

        previous_masked = 0
        for rate in ('10', '25', '50'):
            rate_result = result['rates'][rate]
            masked = rate_result['masked_notes']
            self.assertGreaterEqual(masked, previous_masked)
            previous_masked = masked
            self.assertEqual(
                masked + rate_result['visible_notes'],
                result['dataset']['eligible_notes'],
            )
            self.assertEqual(
                sum(rate_result['masked_notes_by_true_voice'].values()),
                masked,
            )
            self.assertEqual(
                set(rate_result['conditions']),
                {'aligned', 'cyclic_range_negative_control'},
            )
            for condition in rate_result['conditions'].values():
                self.assertEqual(
                    condition['evaluated_masked_event_sequence_sha256'],
                    rate_result['masked_event_sequence_sha256'],
                )
                for metrics in condition['methods'].values():
                    self.assertEqual(metrics['evaluated_masked_notes'], masked)
                    self.assertEqual(
                        sum(sum(row.values()) for row in metrics['confusion'].values()),
                        masked,
                    )
                    for voice in label_masking.VOICE_NAMES:
                        self.assertEqual(
                            metrics['per_voice'][voice]['support'],
                            rate_result['masked_notes_by_true_voice'][voice],
                        )

            aligned_rp = rate_result['conditions']['aligned']['methods'][
                'range_prior'
            ]
            control_rp = rate_result['conditions'][
                'cyclic_range_negative_control'
            ]['methods']['range_prior']
            self.assertEqual(aligned_rp['accuracy'], 1.0)
            self.assertEqual(aligned_rp['macro_f1'], 1.0)
            self.assertEqual(control_rp['accuracy'], 0.0)
            self.assertLess(control_rp['macro_f1'], aligned_rp['macro_f1'])

    def test_cli_output_matches_function_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = self.make_dataset(root)
            output_path = root / 'reports' / 'masking.json'
            expected = label_masking.evaluate_label_masking(dataset_dir)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                return_code = label_masking.main([
                    '--dataset-dir',
                    str(dataset_dir),
                    '--split',
                    'train',
                    '--output-json',
                    str(output_path),
                ])

            self.assertEqual(return_code, 0)
            self.assertEqual(json.loads(stdout.getvalue()), expected)
            self.assertEqual(json.loads(output_path.read_text()), expected)
            self.assertEqual(
                output_path.read_text(),
                f'{json.dumps(expected, indent=2, sort_keys=True, allow_nan=False)}\n',
            )

    def test_mask_identity_percentiles_and_train_only_guard_are_frozen(self):
        identity = label_masking._event_id(
            'recording_alpha',
            3,
            2,
            5,
            61,
            1.25,
            1.75,
        )
        self.assertEqual(
            identity,
            '79f1f025724c462a5b205cd9eeb2846b6016659616bce40e894afba5ade386af',
        )
        mask_memberships = [
            label_masking._is_masked(identity, rate)
            for rate in label_masking.MASK_RATES_PERCENT
        ]
        self.assertEqual(mask_memberships, sorted(mask_memberships))
        with self.assertRaisesRegex(ValueError, 'Unsupported pre-registered'):
            label_masking._is_masked(identity, 20)

        histogram = Counter({40: 1, 50: 2, 60: 1})
        self.assertEqual(label_masking._linear_percentile(histogram, 1), 40.3)
        self.assertEqual(label_masking._linear_percentile(histogram, 99), 59.7)

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = self.make_dataset(Path(temp_dir))
            with self.assertRaisesRegex(ValueError, 'train-only'):
                label_masking.evaluate_label_masking(
                    dataset_dir,
                    split='test',
                )
            validation_manifest = Path(temp_dir) / 'valid.json'
            validation_manifest.write_text('["recording_alpha"]', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'filename is train.json'):
                label_masking.evaluate_label_masking(
                    dataset_dir,
                    recording_manifest=validation_manifest,
                )

    def test_unknown_reference_label_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / 'InvalidChoral'
            note_dir = dataset_dir / 'note'
            note_dir.mkdir(parents=True)
            (dataset_dir / 'train.json').write_text('["bad"]', encoding='utf-8')
            with (note_dir / 'bad.pkl').open('wb') as handle:
                pickle.dump([{'mystery': [note(60, 0.0, 1.0)]}], handle)

            with self.assertRaisesRegex(RuntimeError, 'unlabelled_counts'):
                label_masking.evaluate_label_masking(dataset_dir)


if __name__ == '__main__':
    unittest.main()
