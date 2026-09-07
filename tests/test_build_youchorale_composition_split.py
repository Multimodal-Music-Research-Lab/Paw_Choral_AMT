import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_DIR / 'tools'
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import build_youchorale_composition_split as split_builder


class BuildYouChoraleCompositionSplitTest(unittest.TestCase):
    def make_dataset(self, root: Path) -> Path:
        dataset_dir = root / 'YouChorale'
        dataset_dir.mkdir()
        rows = [
            ('rec00a', 'Bach', 'Gloria—In Excelsis', 'https://example/0a'),
            ('rec00b', 'ＢＡＣＨ', 'Gloria In Excelsis', 'https://example/0b'),
        ]
        rows.extend(
            (
                f'rec{index:02d}',
                f'Composer {index}',
                f'Title {index}',
                f'https://example/{index}',
            )
            for index in range(1, 10)
        )
        info_lines = ['id\tcomposer\ttitle\tlink'] + [
            '\t'.join(row) for row in rows
        ]
        (dataset_dir / 'info.csv').write_text(
            '\n'.join(info_lines) + '\n',
            encoding='utf-8',
        )
        manifests = {
            'train': ['rec00a', 'rec01', 'rec02', 'rec03'],
            'valid': ['rec04', 'rec05', 'rec06'],
            'test': ['rec00b', 'rec07', 'rec08', 'rec09'],
        }
        for split_name, values in manifests.items():
            (dataset_dir / f'{split_name}.json').write_text(
                json.dumps(values),
                encoding='utf-8',
            )
        return dataset_dir

    def test_normalization_uses_nfkc_casefold_punctuation_and_whitespace(self):
        self.assertEqual(
            split_builder.normalize_group_field('  Ｊ．S．  BACH—Mass\tNo.1  '),
            'j s bach mass no 1',
        )
        self.assertEqual(split_builder.normalize_group_field('A-B'), 'a b')

    def test_reports_official_overlap_and_builds_disjoint_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = self.make_dataset(root)
            report, payloads = split_builder.build_composition_split(dataset_dir)

            overlap = report['official_recording_split_overlap']
            self.assertEqual(overlap['recording_counts'], {
                'train': 4,
                'valid': 3,
                'test': 4,
            })
            self.assertEqual(overlap['group_counts'], {
                'train': 4,
                'valid': 3,
                'test': 4,
            })
            self.assertEqual(
                overlap['pairwise_work_overlap']['train__test']['shared_groups'],
                1,
            )
            self.assertEqual(
                overlap['evaluation_seen_in_train']['test']['recordings'],
                1,
            )
            self.assertEqual(report['metadata']['manifest_groups'], 10)

            group_map = json.loads(payloads['group_map.json'])
            manifests = {
                split_name: json.loads(payloads[f'{split_name}.json'])
                for split_name in split_builder.SPLIT_NAMES
            }
            self.assertEqual(
                report['composition_disjoint_split']['group_counts'],
                {'train': 8, 'valid': 1, 'test': 1},
            )
            self.assertEqual(
                {
                    pair['shared_groups']
                    for pair in report['composition_disjoint_split'][
                        'work_overlap'
                    ].values()
                },
                {0},
            )
            self.assertEqual(
                report['unicode_database_version'],
                split_builder.unicodedata.unidata_version,
            )
            self.assertEqual(
                set().union(*(set(values) for values in manifests.values())),
                {
                    'rec00a', 'rec00b', 'rec01', 'rec02', 'rec03', 'rec04',
                    'rec05', 'rec06', 'rec07', 'rec08', 'rec09',
                },
            )
            self.assertFalse(set(manifests['train']) & set(manifests['valid']))
            self.assertFalse(set(manifests['train']) & set(manifests['test']))
            self.assertFalse(set(manifests['valid']) & set(manifests['test']))
            self.assertEqual(
                group_map['recordings']['rec00a']['group_key'],
                group_map['recordings']['rec00b']['group_key'],
            )
            self.assertEqual(
                group_map['recordings']['rec00a']['split'],
                group_map['recordings']['rec00b']['split'],
            )
            for filename, payload in payloads.items():
                self.assertEqual(
                    report['composition_disjoint_split']['artifact_sha256'][filename],
                    hashlib.sha256(payload).hexdigest(),
                )

    def test_cli_prints_only_path_independent_json_and_writes_exact_payloads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = self.make_dataset(root)
            output_dir = root / 'published-split'
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                return_code = split_builder.main([
                    '--dataset-dir', str(dataset_dir),
                    '--output-dir', str(output_dir),
                ])

            self.assertEqual(return_code, 0)
            report = json.loads(stdout.getvalue())
            self.assertNotIn(str(root), stdout.getvalue())
            self.assertEqual(set(path.name for path in output_dir.iterdir()), {
                'group_map.json', 'train.json', 'valid.json', 'test.json',
            })
            for filename, expected_hash in report[
                'composition_disjoint_split'
            ]['artifact_sha256'].items():
                self.assertEqual(
                    hashlib.sha256((output_dir / filename).read_bytes()).hexdigest(),
                    expected_hash,
                )

            first_contents = {
                filename: (output_dir / filename).read_bytes()
                for filename in split_builder.OUTPUT_FILENAMES
            }
            with contextlib.redirect_stdout(io.StringIO()):
                split_builder.main([
                    '--dataset-dir', str(dataset_dir),
                    '--output-dir', str(output_dir),
                ])
            self.assertEqual(first_contents, {
                filename: (output_dir / filename).read_bytes()
                for filename in split_builder.OUTPUT_FILENAMES
            })

    def test_existing_different_output_fails_closed_without_changing_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = self.make_dataset(root)
            output_dir = root / 'published-split'
            _, first_payloads = split_builder.build_composition_split(dataset_dir)
            split_builder._write_outputs(
                output_dir,
                first_payloads,
                protected_dirs=[dataset_dir],
            )
            before = {
                filename: (output_dir / filename).read_bytes()
                for filename in split_builder.OUTPUT_FILENAMES
            }
            _, conflicting_payloads = split_builder.build_composition_split(
                dataset_dir,
                salt='different-frozen-protocol',
            )

            with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
                split_builder._write_outputs(
                    output_dir,
                    conflicting_payloads,
                    protected_dirs=[dataset_dir],
                )

            self.assertEqual(before, {
                filename: (output_dir / filename).read_bytes()
                for filename in split_builder.OUTPUT_FILENAMES
            })

    def test_new_output_publish_is_atomic_when_rename_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = self.make_dataset(root)
            output_dir = root / 'published-split'
            _, payloads = split_builder.build_composition_split(dataset_dir)

            with mock.patch.object(
                split_builder.os,
                'replace',
                side_effect=OSError('injected publish failure'),
            ):
                with self.assertRaisesRegex(OSError, 'injected publish failure'):
                    split_builder._write_outputs(
                        output_dir,
                        payloads,
                        protected_dirs=[dataset_dir],
                    )

            self.assertFalse(output_dir.exists())
            self.assertEqual(
                list(root.glob('.published-split.staging.*')),
                [],
            )

    def test_existing_empty_output_directory_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = self.make_dataset(root)
            output_dir = root / 'already-created'
            output_dir.mkdir()
            _, payloads = split_builder.build_composition_split(dataset_dir)

            with self.assertRaisesRegex(ValueError, 'already exists and is empty'):
                split_builder._write_outputs(
                    output_dir,
                    payloads,
                    protected_dirs=[dataset_dir],
                )

            self.assertEqual(list(output_dir.iterdir()), [])

    def test_packed_hdf5_filter_is_recursive_and_stem_matching_is_case_sensitive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = self.make_dataset(root)
            packed_dir = root / 'packed'
            (packed_dir / 'nested').mkdir(parents=True)
            available_ids = [
                'rec00a', 'rec01', 'rec02', 'rec03', 'rec04', 'rec05',
                'rec06', 'rec07', 'rec08', 'rec09',
            ]
            for index, recording_id in enumerate(available_ids):
                suffix = '.H5' if index == 0 else '.hdf5' if index == 1 else '.h5'
                (packed_dir / 'nested' / f'{recording_id}{suffix}').touch()
            (packed_dir / 'REC00b.h5').touch()
            (packed_dir / 'not-in-manifest.h5').touch()

            report, payloads = split_builder.build_composition_split(
                dataset_dir,
                packed_hdf5_dir=packed_dir,
            )

            coverage = report['packed_hdf5_coverage']
            self.assertEqual(coverage['packed_stems'], 12)
            self.assertEqual(coverage['intersection_recordings'], 10)
            self.assertEqual(coverage['manifest_missing_packed'], 1)
            self.assertEqual(coverage['packed_not_in_manifests'], 2)
            self.assertEqual(
                coverage['by_official_split']['test']['intersection_recordings'],
                3,
            )
            self.assertEqual(
                coverage['by_official_split']['test']['coverage_fraction'],
                0.75,
            )
            self.assertEqual(report['selection']['groups'], 10)
            group_map = json.loads(payloads['group_map.json'])
            self.assertNotIn('rec00b', group_map['recordings'])
            self.assertNotIn('REC00b', group_map['recordings'])

    def test_rejects_duplicate_hdf5_stems(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = self.make_dataset(root)
            packed_dir = root / 'packed'
            (packed_dir / 'nested').mkdir(parents=True)
            (packed_dir / 'rec01.h5').touch()
            (packed_dir / 'nested' / 'rec01.hdf5').touch()

            with self.assertRaisesRegex(ValueError, 'Duplicate packed HDF5'):
                split_builder.build_composition_split(
                    dataset_dir,
                    packed_hdf5_dir=packed_dir,
                )

    def test_rejects_invalid_metadata_and_manifests(self):
        mutations = {
            'missing metadata ID': lambda dataset: (
                dataset / 'info.csv'
            ).write_text(
                (dataset / 'info.csv').read_text(encoding='utf-8').replace(
                    'rec09\tComposer 9\tTitle 9\thttps://example/9\n',
                    '',
                ),
                encoding='utf-8',
            ),
            'duplicate metadata ID': lambda dataset: (
                dataset / 'info.csv'
            ).write_text(
                (dataset / 'info.csv').read_text(encoding='utf-8')
                + 'rec01\tOther\tOther\thttps://example/duplicate\n',
                encoding='utf-8',
            ),
            'split overlap': lambda dataset: (
                dataset / 'valid.json'
            ).write_text(json.dumps(['rec04', 'rec01']), encoding='utf-8'),
            'duplicate split ID': lambda dataset: (
                dataset / 'train.json'
            ).write_text(json.dumps(['rec00a', 'rec01', 'rec01']), encoding='utf-8'),
            'empty split': lambda dataset: (
                dataset / 'test.json'
            ).write_text('[]', encoding='utf-8'),
            'empty normalized group': lambda dataset: (
                dataset / 'info.csv'
            ).write_text(
                (dataset / 'info.csv').read_text(encoding='utf-8').replace(
                    'rec01\tComposer 1\tTitle 1\thttps://example/1',
                    'rec01\t---\t...\thttps://example/1',
                ),
                encoding='utf-8',
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp_dir:
                dataset_dir = self.make_dataset(Path(temp_dir))
                mutate(dataset_dir)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    split_builder.build_composition_split(dataset_dir)

    def test_rejects_output_inside_read_only_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = self.make_dataset(root)
            _, payloads = split_builder.build_composition_split(dataset_dir)
            with self.assertRaisesRegex(ValueError, 'output-dir must not'):
                split_builder._write_outputs(
                    dataset_dir / 'generated',
                    payloads,
                    protected_dirs=[dataset_dir],
                )


if __name__ == '__main__':
    unittest.main()
