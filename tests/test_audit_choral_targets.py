import contextlib
import hashlib
import io
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
for source_dir in (REPO_DIR / 'src', REPO_DIR / 'tools'):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import audit_choral_targets


def note(pitch, onset=0.0, offset=1.0):
    return [pitch, 0, 0, onset, offset]


class AuditChoralTargetsTest(unittest.TestCase):
    def write_note_file(self, dataset_dir, recording_id, note_bars):
        note_path = dataset_dir / 'note' / f'{recording_id}.pkl'
        with note_path.open('wb') as handle:
            pickle.dump(note_bars, handle)

    def make_dataset(self, root):
        dataset_dir = root / 'TinyChoral'
        (dataset_dir / 'note').mkdir(parents=True)
        (dataset_dir / 'train.json').write_text(
            json.dumps(['song_b', 'song_a']),
            encoding='utf-8',
        )
        (dataset_dir / 'valid.json').write_text(json.dumps(['song_valid']), encoding='utf-8')
        (dataset_dir / 'test.json').write_text(json.dumps(['song_test']), encoding='utf-8')

        self.write_note_file(dataset_dir, 'song_a', [{
            'S1': [note(45)],
            'S2': [note(74)],
            'A': [note(69)],
            'T': [note(60)],
            'B': [note(48)],
            'unknown': [note(80), note(55, onset=1.0)],
        }])
        self.write_note_file(dataset_dir, 'song_b', [{
            'S': [note(80)],
            'B': [note(40)],
        }])
        self.write_note_file(dataset_dir, 'song_valid', [{'A': [note(65)]}])
        self.write_note_file(dataset_dir, 'song_test', [{'T': [note(58)]}])
        return dataset_dir

    def test_audit_reports_label_group_divisi_and_assignment_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = self.make_dataset(Path(temp_dir))
            result = audit_choral_targets.audit_dataset(dataset_dir, 'train')

        self.assertEqual(result['notes']['total'], 9)
        self.assertEqual(result['notes']['canonical_known'], 7)
        self.assertEqual(result['notes']['canonical_unknown_or_ambiguous'], 2)
        self.assertEqual(
            result['notes']['canonical_distribution'],
            {'S': 3, 'A': 1, 'T': 1, 'B': 2},
        )
        self.assertEqual(
            result['notes']['canonical_pitch_statistics']['S']['min'],
            45,
        )
        self.assertEqual(
            result['notes']['canonical_pitch_statistics']['S']['max'],
            80,
        )
        self.assertEqual(
            result['notes']['canonical_pitch_statistics']['A']['median'],
            69.0,
        )
        self.assertEqual(result['onset_groups']['total'], 3)
        self.assertEqual(result['onset_groups']['more_than_four']['count'], 1)
        self.assertAlmostEqual(result['onset_groups']['more_than_four']['ratio'], 1 / 3)
        duplicate = result['onset_groups']['duplicate_canonical_voice']
        self.assertEqual(duplicate['group_count'], 1)
        self.assertAlmostEqual(duplicate['group_ratio'], 1 / 3)
        self.assertEqual(duplicate['participating_note_count'], 2)
        self.assertAlmostEqual(duplicate['participating_note_ratio'], 2 / 7)
        self.assertEqual(duplicate['excess_note_count'], 1)

        for method in ('range_prior', 'ordered_continuity'):
            metrics = result['assignments'][method]
            self.assertEqual(metrics['assigned_notes'], 9)
            self.assertEqual(metrics['known_labels'], 7)
            self.assertEqual(metrics['retained_known_labels'], 7)
            self.assertEqual(metrics['changed_known_labels'], 0)
            self.assertEqual(metrics['retention_ratio'], 1.0)
            self.assertEqual(metrics['confusion']['S']['S'], 3)
            self.assertEqual(metrics['confusion']['B']['B'], 2)

        self.assertGreater(
            result['assignments']['legacy_range_prior']['changed_known_labels'],
            0,
        )
        self.assertGreater(
            result['assignments']['legacy_ordered_continuity']['changed_known_labels'],
            0,
        )
        self.assertEqual(
            {
                metrics['assigned_notes']
                for metrics in result['assignments'].values()
            },
            {9},
        )
        self.assertEqual(
            list(result['assignments']['range_prior']['confusion']),
            ['S', 'A', 'T', 'B'],
        )

    def test_validation_split_accepts_valid_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = self.make_dataset(Path(temp_dir))
            result = audit_choral_targets.audit_dataset(dataset_dir, 'validation')

        self.assertEqual(result['dataset']['split_files'], {'validation': 'valid.json'})
        self.assertEqual(result['dataset']['recordings'], 1)
        self.assertEqual(result['notes']['canonical_distribution']['A'], 1)
        self.assertNotIn('packed_coverage', result['dataset'])

    def test_packed_hdf5_filter_reports_coverage_and_audits_intersection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = self.make_dataset(root)
            packed_dir = root / 'packed'
            (packed_dir / 'nested').mkdir(parents=True)
            (packed_dir / 'nested' / 'song_a.H5').touch()
            (packed_dir / 'song_valid.hdf5').touch()
            (packed_dir / 'not_in_train.h5').touch()
            # A manifest item excluded by the packed filter is never opened.
            (dataset_dir / 'note' / 'song_b.pkl').unlink()

            result = audit_choral_targets.audit_dataset(
                dataset_dir,
                'train',
                packed_hdf5_dir=packed_dir,
            )

        self.assertEqual(result['dataset']['recordings'], 1)
        self.assertEqual(result['notes']['total'], 7)
        coverage = result['dataset']['packed_coverage']
        self.assertEqual(coverage['schema_version'], 1)
        self.assertEqual(
            coverage['match_semantics'],
            'exact_case_sensitive_filename_stem_existence_only',
        )
        self.assertEqual(coverage['manifest_recordings'], 2)
        self.assertEqual(coverage['packed_recordings_total'], 3)
        self.assertEqual(coverage['intersection_recordings'], 1)
        self.assertEqual(coverage['manifest_coverage_ratio'], 0.5)
        self.assertEqual(coverage['packed_selection_ratio'], 1 / 3)
        self.assertEqual(coverage['manifest_missing_packed_recordings'], 1)
        self.assertEqual(
            coverage['manifest_missing_packed_recording_ids'],
            ['song_b'],
        )
        self.assertEqual(
            coverage['packed_not_in_selected_manifest_recordings'],
            2,
        )
        self.assertEqual(
            coverage['intersection_recording_ids_sha256'],
            result['dataset']['recording_ids_sha256'],
        )
        self.assertNotEqual(
            coverage['manifest_recording_ids_sha256'],
            coverage['intersection_recording_ids_sha256'],
        )
        self.assertNotEqual(
            coverage['packed_recording_ids_sha256'],
            coverage['intersection_recording_ids_sha256'],
        )

    def test_packed_hdf5_filter_rejects_duplicate_stems_and_empty_overlap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = self.make_dataset(root)
            packed_dir = root / 'packed'
            packed_dir.mkdir()
            (packed_dir / 'song_a.h5').touch()
            (packed_dir / 'song_a.hdf5').touch()

            with self.assertRaisesRegex(ValueError, 'same recording ID'):
                audit_choral_targets.audit_dataset(
                    dataset_dir,
                    'train',
                    packed_hdf5_dir=packed_dir,
                )

            (packed_dir / 'song_a.h5').unlink()
            (packed_dir / 'song_a.hdf5').unlink()
            (packed_dir / 'outside.h5').touch()
            with self.assertRaisesRegex(ValueError, 'No packed HDF5 recording'):
                audit_choral_targets.audit_dataset(
                    dataset_dir,
                    'train',
                    packed_hdf5_dir=packed_dir,
                )

    def test_packed_hdf5_filter_validates_directory_and_cli_forwards_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = self.make_dataset(root)
            missing_dir = root / 'missing-packed'
            with self.assertRaises(NotADirectoryError):
                audit_choral_targets.audit_dataset(
                    dataset_dir,
                    'train',
                    packed_hdf5_dir=missing_dir,
                )

            packed_dir = root / 'packed'
            packed_dir.mkdir()
            with self.assertRaisesRegex(ValueError, 'No .h5 or .hdf5 files'):
                audit_choral_targets.audit_dataset(
                    dataset_dir,
                    'train',
                    packed_hdf5_dir=packed_dir,
                )

            (packed_dir / 'song_a.h5').touch()
            output_path = root / 'packed-audit.json'
            expected = audit_choral_targets.audit_dataset(
                dataset_dir,
                'train',
                packed_hdf5_dir=packed_dir,
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                return_code = audit_choral_targets.main([
                    '--dataset-dir',
                    str(dataset_dir),
                    '--split',
                    'train',
                    '--packed-hdf5-dir',
                    str(packed_dir),
                    '--output-json',
                    str(output_path),
                ])

            self.assertEqual(return_code, 0)
            self.assertEqual(json.loads(stdout.getvalue()), expected)
            self.assertEqual(json.loads(output_path.read_text()), expected)

    def test_assignment_statistics_keep_original_note_denominator(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / 'DuplicateAttackChoral'
            (dataset_dir / 'note').mkdir(parents=True)
            (dataset_dir / 'train.json').write_text(
                json.dumps(['same_attack']),
                encoding='utf-8',
            )
            self.write_note_file(dataset_dir, 'same_attack', [{
                'S1': [note(72, offset=0.5)],
                'S2': [note(72, offset=1.0)],
            }])

            result = audit_choral_targets.audit_dataset(dataset_dir, 'train')

        self.assertEqual(result['notes']['total'], 2)
        self.assertEqual(
            {
                metrics['assigned_notes']
                for metrics in result['assignments'].values()
            },
            {2},
        )
        self.assertEqual(
            result['assignments']['range_prior']['known_labels'],
            2,
        )

    def test_all_split_output_is_deterministic_and_cli_can_write_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = self.make_dataset(root)
            output_path = root / 'outputs' / 'audit.json'

            first = audit_choral_targets.audit_dataset(dataset_dir, 'all')
            second = audit_choral_targets.audit_dataset(dataset_dir, 'all')
            self.assertEqual(first, second)
            self.assertEqual(first['dataset']['recordings'], 4)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                return_code = audit_choral_targets.main([
                    '--dataset-dir',
                    str(dataset_dir),
                    '--split',
                    'all',
                    '--output-json',
                    str(output_path),
                ])

            self.assertEqual(return_code, 0)
            self.assertEqual(json.loads(stdout.getvalue()), first)
            self.assertEqual(json.loads(output_path.read_text(encoding='utf-8')), first)
            self.assertEqual(
                output_path.read_text(encoding='utf-8'),
                f'{json.dumps(first, indent=2, sort_keys=True)}\n',
            )
            self.assertEqual(
                hashlib.sha256(output_path.read_bytes()).hexdigest(),
                'dfe8e6d70c75ea7a226c2065f7c71206fbd1daf16d8a8d894da41ea208ce1ba5',
            )


if __name__ == '__main__':
    unittest.main()
