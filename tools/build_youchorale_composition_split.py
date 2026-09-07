#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Audit and build deterministic composition-disjoint YouChorale splits.

The command only reads ``info.csv`` and the three official JSON manifests.
When ``--packed-hdf5-dir`` is supplied, filename stems are used as a read-only
availability filter; HDF5 contents are never opened.  The only writes are the
four manifest artifacts requested through ``--output-dir``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
DEFAULT_SALT = 'pawct-icassp2027-v1'
SPLIT_NAMES = ('train', 'valid', 'test')
SPLIT_FILENAMES = {name: f'{name}.json' for name in SPLIT_NAMES}
OUTPUT_FILENAMES = ('group_map.json', 'train.json', 'valid.json', 'test.json')
REQUIRED_METADATA_COLUMNS = ('id', 'composer', 'title', 'link')


def normalize_group_field(value: str) -> str:
    """Apply the pre-registered normalization to one metadata field."""

    normalized = unicodedata.normalize('NFKC', value).casefold()
    punctuation_spaced = ''.join(
        ' ' if unicodedata.category(character).startswith('P') else character
        for character in normalized
    )
    return ' '.join(punctuation_spaced.split())


def make_group_key(composer: str, title: str) -> tuple[str, str, str]:
    """Return normalized fields and an unambiguous serialized pair key."""

    normalized_composer = normalize_group_field(composer)
    normalized_title = normalize_group_field(title)
    if not normalized_composer or not normalized_title:
        raise ValueError(
            'Every selected metadata row must have a non-empty normalized '
            'composer and title'
        )
    group_key = json.dumps(
        [normalized_composer, normalized_title],
        ensure_ascii=False,
        separators=(',', ':'),
    )
    return normalized_composer, normalized_title, group_key


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + '\n'
    ).encode('utf-8')


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sequence_sha256(values: Iterable[str]) -> str:
    payload = ''.join(f'{value}\n' for value in sorted(values)).encode('utf-8')
    return _sha256_bytes(payload)


def _validate_recording_id(value: Any, *, source_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f'{source_name} contains a missing or non-string ID')
    if value != value.strip():
        raise ValueError(f'{source_name} contains an ID with surrounding whitespace')
    return value


def _read_metadata(dataset_dir: Path) -> dict[str, dict[str, str]]:
    metadata_path = dataset_dir / 'info.csv'
    if not metadata_path.is_file():
        raise FileNotFoundError('dataset-dir is missing info.csv')

    rows: dict[str, dict[str, str]] = {}
    with metadata_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError('info.csv is missing its header')
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError('info.csv contains duplicate header names')
        missing_columns = [
            column for column in REQUIRED_METADATA_COLUMNS if column not in fieldnames
        ]
        if missing_columns:
            raise ValueError(
                'info.csv is missing required columns: ' + ', '.join(missing_columns)
            )

        for row in reader:
            if None in row:
                raise ValueError(
                    f'info.csv row {reader.line_num} has more fields than its header'
                )
            if not any((value or '').strip() for value in row.values()):
                continue
            recording_id = _validate_recording_id(
                row.get('id'),
                source_name=f'info.csv row {reader.line_num}',
            )
            if recording_id in rows:
                raise ValueError(f'info.csv contains duplicate metadata ID {recording_id!r}')
            composer = row.get('composer')
            title = row.get('title')
            link = row.get('link')
            if not all(isinstance(value, str) for value in (composer, title, link)):
                raise ValueError(f'info.csv row {reader.line_num} has missing fields')
            rows[recording_id] = {
                'composer': composer,
                'title': title,
                'link': link,
            }

    if not rows:
        raise ValueError('info.csv contains no metadata rows')
    return rows


def _read_manifest(dataset_dir: Path, split_name: str) -> list[str]:
    filename = SPLIT_FILENAMES[split_name]
    manifest_path = dataset_dir / filename
    if not manifest_path.is_file():
        raise FileNotFoundError(f'dataset-dir is missing {filename}')
    try:
        values = json.loads(manifest_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as error:
        raise ValueError(f'{filename} is not valid JSON: {error.msg}') from error
    if not isinstance(values, list):
        raise ValueError(f'{filename} must contain a JSON list of recording IDs')

    recording_ids = [
        _validate_recording_id(value, source_name=filename) for value in values
    ]
    if not recording_ids:
        raise ValueError(f'{filename} must not be empty')
    duplicates = sorted(
        recording_id
        for recording_id, count in Counter(recording_ids).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f'{filename} contains duplicate IDs: {duplicates}')
    return recording_ids


def _read_official_manifests(dataset_dir: Path) -> dict[str, list[str]]:
    manifests = {
        split_name: _read_manifest(dataset_dir, split_name)
        for split_name in SPLIT_NAMES
    }
    memberships: dict[str, list[str]] = defaultdict(list)
    for split_name, recording_ids in manifests.items():
        for recording_id in recording_ids:
            memberships[recording_id].append(split_name)
    overlaps = {
        recording_id: split_names
        for recording_id, split_names in memberships.items()
        if len(split_names) > 1
    }
    if overlaps:
        raise ValueError(f'Recording IDs occur in multiple split files: {overlaps}')
    return manifests


def _read_packed_stems(packed_hdf5_dir: Path) -> list[str]:
    if not packed_hdf5_dir.is_dir():
        raise NotADirectoryError('packed-hdf5-dir is not a directory')

    relative_paths_by_stem: dict[str, list[str]] = defaultdict(list)
    for path in packed_hdf5_dir.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.h5', '.hdf5'}:
            relative_paths_by_stem[path.stem].append(
                path.relative_to(packed_hdf5_dir).as_posix()
            )
    if not relative_paths_by_stem:
        raise ValueError('packed-hdf5-dir contains no .h5 or .hdf5 files')

    duplicates = {
        stem: sorted(relative_paths)
        for stem, relative_paths in relative_paths_by_stem.items()
        if len(relative_paths) > 1
    }
    if duplicates:
        raise ValueError(f'Duplicate packed HDF5 filename stems: {duplicates}')
    return sorted(relative_paths_by_stem)


def _group_selected_recordings(
    selected_recording_ids: Iterable[str],
    metadata: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for recording_id in sorted(selected_recording_ids):
        row = metadata[recording_id]
        normalized_composer, normalized_title, group_key = make_group_key(
            row['composer'], row['title']
        )
        group = groups.setdefault(
            group_key,
            {
                'group_key': group_key,
                'normalized_composer': normalized_composer,
                'normalized_title': normalized_title,
                'recording_ids': [],
            },
        )
        group['recording_ids'].append(recording_id)
    if not groups:
        raise ValueError('The selected recording intersection contains no groups')
    return groups


def _overlap_report(
    manifests: Mapping[str, Sequence[str]],
    recording_group_keys: Mapping[str, str],
) -> dict[str, Any]:
    group_sets = {
        split_name: {
            recording_group_keys[recording_id]
            for recording_id in recording_ids
            if recording_id in recording_group_keys
        }
        for split_name, recording_ids in manifests.items()
    }
    recording_sets = {
        split_name: [
            recording_id
            for recording_id in recording_ids
            if recording_id in recording_group_keys
        ]
        for split_name, recording_ids in manifests.items()
    }
    pairs: dict[str, Any] = {}
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            shared = group_sets[left] & group_sets[right]
            pairs[f'{left}__{right}'] = {
                'shared_groups': len(shared),
                'shared_group_keys_sha256': _sequence_sha256(shared),
                'left_group_fraction': (
                    len(shared) / len(group_sets[left]) if group_sets[left] else 0.0
                ),
                'right_group_fraction': (
                    len(shared) / len(group_sets[right]) if group_sets[right] else 0.0
                ),
            }

    evaluation_seen_in_train: dict[str, Any] = {}
    for split_name in ('valid', 'test'):
        seen_groups = group_sets['train'] & group_sets[split_name]
        seen_recordings = [
            recording_id
            for recording_id in recording_sets[split_name]
            if recording_group_keys[recording_id] in seen_groups
        ]
        evaluation_seen_in_train[split_name] = {
            'groups': len(seen_groups),
            'group_fraction': (
                len(seen_groups) / len(group_sets[split_name])
                if group_sets[split_name]
                else 0.0
            ),
            'recordings': len(seen_recordings),
            'recording_fraction': (
                len(seen_recordings) / len(recording_sets[split_name])
                if recording_sets[split_name]
                else 0.0
            ),
            'recording_ids_sha256': _sequence_sha256(seen_recordings),
        }

    return {
        'recording_counts': {
            split_name: len(recording_sets[split_name])
            for split_name in SPLIT_NAMES
        },
        'recording_ids_sha256': {
            split_name: _sequence_sha256(recording_sets[split_name])
            for split_name in SPLIT_NAMES
        },
        'group_counts': {
            split_name: len(group_sets[split_name]) for split_name in SPLIT_NAMES
        },
        'group_keys_sha256': {
            split_name: _sequence_sha256(group_sets[split_name])
            for split_name in SPLIT_NAMES
        },
        'pairwise_work_overlap': pairs,
        'evaluation_seen_in_train': evaluation_seen_in_train,
    }


def _assign_groups(
    groups: Mapping[str, Mapping[str, Any]],
    *,
    salt: str,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    if not salt:
        raise ValueError('salt must not be empty')

    ranked_groups: list[dict[str, Any]] = []
    for group_key, group in groups.items():
        rank_sha256 = hashlib.sha256((salt + group_key).encode('utf-8')).hexdigest()
        ranked_groups.append(
            {
                **group,
                'rank_sha256': rank_sha256,
            }
        )
    ranked_groups.sort(key=lambda group: (group['rank_sha256'], group['group_key']))

    group_count = len(ranked_groups)
    train_cut = (8 * group_count) // 10
    valid_cut = (9 * group_count) // 10
    slices = {
        'train': ranked_groups[:train_cut],
        'valid': ranked_groups[train_cut:valid_cut],
        'test': ranked_groups[valid_cut:],
    }
    empty_splits = [name for name, values in slices.items() if not values]
    if empty_splits:
        raise ValueError(
            'The deterministic group assignment produces empty splits '
            f'{empty_splits}; provide more distinct selected groups'
        )

    manifests: dict[str, list[str]] = {}
    for split_name, split_groups in slices.items():
        manifests[split_name] = [
            recording_id
            for group in split_groups
            for recording_id in sorted(group['recording_ids'])
        ]
        for group in split_groups:
            group['split'] = split_name
    return manifests, ranked_groups


def _build_group_map(
    ranked_groups: Sequence[Mapping[str, Any]],
    *,
    salt: str,
) -> dict[str, Any]:
    recordings: dict[str, Any] = {}
    group_rows: list[dict[str, Any]] = []
    for group in ranked_groups:
        group_rows.append(
            {
                'group_key': group['group_key'],
                'normalized_composer': group['normalized_composer'],
                'normalized_title': group['normalized_title'],
                'rank_sha256': group['rank_sha256'],
                'recording_ids': sorted(group['recording_ids']),
                'split': group['split'],
            }
        )
        for recording_id in group['recording_ids']:
            recordings[recording_id] = {
                'group_key': group['group_key'],
                'split': group['split'],
            }
    return {
        'schema_version': SCHEMA_VERSION,
        'unicode_database_version': unicodedata.unidata_version,
        'normalization': (
            'Unicode NFKC; casefold; replace every Unicode punctuation '
            'character with a space; collapse whitespace'
        ),
        'assignment': {
            'salt': salt,
            'rank': 'SHA256(UTF-8(salt + group_key)) ascending',
            'group_key': 'compact UTF-8 JSON array [normalized_composer,normalized_title]',
            'cuts': 'floor(0.8 * groups), floor(0.9 * groups)',
        },
        'groups': group_rows,
        'recordings': recordings,
    }


def build_composition_split(
    dataset_dir: Path,
    *,
    packed_hdf5_dir: Path | None = None,
    salt: str = DEFAULT_SALT,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Return a path-independent report and deterministic output payloads."""

    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise NotADirectoryError('dataset-dir is not a directory')

    metadata = _read_metadata(dataset_dir)
    official_manifests = _read_official_manifests(dataset_dir)
    official_recording_ids = {
        recording_id
        for recording_ids in official_manifests.values()
        for recording_id in recording_ids
    }
    missing_metadata_ids = sorted(official_recording_ids - set(metadata))
    if missing_metadata_ids:
        raise ValueError(
            'Split manifests contain IDs missing from info.csv: '
            f'{missing_metadata_ids}'
        )

    official_groups = _group_selected_recordings(official_recording_ids, metadata)
    official_recording_group_keys = {
        recording_id: group_key
        for group_key, group in official_groups.items()
        for recording_id in group['recording_ids']
    }

    selected_recording_ids = set(official_recording_ids)
    packed_report = None
    if packed_hdf5_dir is not None:
        packed_stems = _read_packed_stems(Path(packed_hdf5_dir))
        packed_stem_set = set(packed_stems)
        selected_recording_ids &= packed_stem_set
        if not selected_recording_ids:
            raise ValueError(
                'No exact case-sensitive packed HDF5 stem matches a manifest ID'
            )
        missing_packed = official_recording_ids - packed_stem_set
        packed_extra = packed_stem_set - official_recording_ids
        packed_report = {
            'match_semantics': (
                'recursive case-insensitive .h5/.hdf5 extension discovery; '
                'exact case-sensitive stem equality; file contents are not read'
            ),
            'hash_semantics': (
                'SHA256 of sorted, exact stems encoded as UTF-8, one per line '
                'with a trailing newline'
            ),
            'packed_stems': len(packed_stems),
            'packed_stems_sha256': _sequence_sha256(packed_stems),
            'manifest_recordings': len(official_recording_ids),
            'manifest_recordings_sha256': _sequence_sha256(official_recording_ids),
            'intersection_recordings': len(selected_recording_ids),
            'intersection_recordings_sha256': _sequence_sha256(
                selected_recording_ids
            ),
            'manifest_coverage_fraction': (
                len(selected_recording_ids) / len(official_recording_ids)
            ),
            'manifest_missing_packed': len(missing_packed),
            'manifest_missing_packed_sha256': _sequence_sha256(missing_packed),
            'packed_not_in_manifests': len(packed_extra),
            'packed_not_in_manifests_sha256': _sequence_sha256(packed_extra),
            'by_official_split': {
                split_name: {
                    'manifest_recordings': len(recording_ids),
                    'intersection_recordings': sum(
                        recording_id in packed_stem_set
                        for recording_id in recording_ids
                    ),
                    'coverage_fraction': (
                        sum(
                            recording_id in packed_stem_set
                            for recording_id in recording_ids
                        )
                        / len(recording_ids)
                    ),
                    'intersection_recording_ids_sha256': _sequence_sha256(
                        recording_id
                        for recording_id in recording_ids
                        if recording_id in packed_stem_set
                    ),
                }
                for split_name, recording_ids in official_manifests.items()
            },
        }

    selected_groups = _group_selected_recordings(selected_recording_ids, metadata)
    selected_recording_group_keys = {
        recording_id: group_key
        for group_key, group in selected_groups.items()
        for recording_id in group['recording_ids']
    }
    composition_manifests, ranked_groups = _assign_groups(
        selected_groups,
        salt=salt,
    )
    group_map = _build_group_map(ranked_groups, salt=salt)
    composition_overlap = _overlap_report(
        composition_manifests,
        selected_recording_group_keys,
    )
    generated_shared_groups = {
        pair_name: values['shared_groups']
        for pair_name, values in composition_overlap[
            'pairwise_work_overlap'
        ].items()
        if values['shared_groups']
    }
    if generated_shared_groups:
        raise ValueError(
            'Internal error: generated split contains cross-split work overlap: '
            f'{generated_shared_groups}'
        )

    output_objects: dict[str, Any] = {'group_map.json': group_map}
    output_objects.update(
        {
            SPLIT_FILENAMES[split_name]: recording_ids
            for split_name, recording_ids in composition_manifests.items()
        }
    )
    output_payloads = {
        filename: _canonical_json_bytes(value)
        for filename, value in output_objects.items()
    }
    artifact_sha256 = {
        filename: _sha256_bytes(payload)
        for filename, payload in sorted(output_payloads.items())
    }

    selected_official_manifests = {
        split_name: [
            recording_id
            for recording_id in recording_ids
            if recording_id in selected_recording_ids
        ]
        for split_name, recording_ids in official_manifests.items()
    }
    report: dict[str, Any] = {
        'schema_version': SCHEMA_VERSION,
        'unicode_database_version': unicodedata.unidata_version,
        'normalization': group_map['normalization'],
        'sequence_hash_semantics': (
            'SHA256 of values after lexicographic sorting, encoded as UTF-8 '
            'one per line with a trailing newline'
        ),
        'metadata': {
            'rows': len(metadata),
            'recording_ids_sha256': _sequence_sha256(metadata),
            'manifest_recordings': len(official_recording_ids),
            'manifest_recording_ids_sha256': _sequence_sha256(
                official_recording_ids
            ),
            'manifest_groups': len(official_groups),
            'manifest_group_keys_sha256': _sequence_sha256(official_groups),
        },
        'official_recording_split_overlap': _overlap_report(
            official_manifests,
            official_recording_group_keys,
        ),
        'selection': {
            'mode': (
                'manifest_and_packed_stem_intersection'
                if packed_hdf5_dir is not None
                else 'manifest'
            ),
            'recordings': len(selected_recording_ids),
            'recording_ids_sha256': _sequence_sha256(selected_recording_ids),
            'groups': len(selected_groups),
            'group_keys_sha256': _sequence_sha256(selected_groups),
        },
        'composition_disjoint_split': {
            'salt': salt,
            'rank': 'SHA256(UTF-8(salt + group_key)) ascending',
            'group_key': (
                'compact UTF-8 JSON array '
                '[normalized_composer,normalized_title]'
            ),
            'cuts': {
                'train_end': (8 * len(selected_groups)) // 10,
                'valid_end': (9 * len(selected_groups)) // 10,
                'total_groups': len(selected_groups),
            },
            'recording_counts': {
                split_name: len(recording_ids)
                for split_name, recording_ids in composition_manifests.items()
            },
            'recording_ids_sha256': {
                split_name: _sequence_sha256(recording_ids)
                for split_name, recording_ids in composition_manifests.items()
            },
            'group_counts': {
                split_name: sum(
                    group['split'] == split_name for group in ranked_groups
                )
                for split_name in SPLIT_NAMES
            },
            'work_overlap': composition_overlap['pairwise_work_overlap'],
            'artifact_sha256': artifact_sha256,
        },
    }
    if packed_report is not None:
        report['packed_hdf5_coverage'] = packed_report
        report['selected_official_split_overlap'] = _overlap_report(
            selected_official_manifests,
            selected_recording_group_keys,
        )
    return report, output_payloads


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _write_outputs(
    output_dir: Path,
    output_payloads: Mapping[str, bytes],
    *,
    protected_dirs: Sequence[Path],
) -> None:
    resolved_output = output_dir.resolve()
    for protected_dir in protected_dirs:
        if _is_within(resolved_output, protected_dir.resolve()):
            raise ValueError(
                'output-dir must not be the dataset-dir, packed-hdf5-dir, '
                'or one of their descendants'
            )

    if output_dir.is_symlink():
        raise ValueError('output-dir must not be a symbolic link')
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError('output-dir exists and is not a directory')
    if output_dir.is_dir():
        present_targets = [
            filename
            for filename in OUTPUT_FILENAMES
            if (output_dir / filename).exists()
        ]
        if present_targets and len(present_targets) != len(OUTPUT_FILENAMES):
            raise ValueError(
                'output-dir contains an incomplete manifest set; refusing to '
                'modify any file'
            )
        if present_targets:
            non_files = [
                filename
                for filename in OUTPUT_FILENAMES
                if not (output_dir / filename).is_file()
            ]
            if non_files:
                raise ValueError(f'Output targets are not files: {non_files}')
            conflicts = [
                filename
                for filename in OUTPUT_FILENAMES
                if (output_dir / filename).read_bytes()
                != output_payloads[filename]
            ]
            if conflicts:
                raise ValueError(
                    'Existing frozen manifest files differ; refusing to '
                    f'overwrite: {conflicts}'
                )
            return
        if any(output_dir.iterdir()):
            raise ValueError(
                'output-dir already contains unrelated files; use a new directory'
            )
        raise ValueError(
            'output-dir already exists and is empty; use a new, nonexistent directory'
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f'.{output_dir.name}.staging.',
            dir=output_dir.parent,
        )
    )
    try:
        for filename in OUTPUT_FILENAMES:
            with (staging_dir / filename).open('wb') as handle:
                handle.write(output_payloads[filename])
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(staging_dir, output_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Audit the official YouChorale work overlap and build a '
            'deterministic composition-disjoint split.'
        )
    )
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        required=True,
        help='Directory containing info.csv and train/valid/test JSON files.',
    )
    parser.add_argument(
        '--packed-hdf5-dir',
        type=Path,
        help='Optional recursive packed-HDF5 availability filter.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Optional destination for group_map.json and three manifests.',
    )
    parser.add_argument(
        '--salt',
        default=DEFAULT_SALT,
        help=f'Deterministic assignment salt (default: {DEFAULT_SALT}).',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report, output_payloads = build_composition_split(
            args.dataset_dir,
            packed_hdf5_dir=args.packed_hdf5_dir,
            salt=args.salt,
        )
        if args.output_dir is not None:
            protected_dirs = [args.dataset_dir]
            if args.packed_hdf5_dir is not None:
                protected_dirs.append(args.packed_hdf5_dir)
            _write_outputs(
                args.output_dir,
                output_payloads,
                protected_dirs=protected_dirs,
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    json.dump(
        report,
        sys.stdout,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    sys.stdout.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
