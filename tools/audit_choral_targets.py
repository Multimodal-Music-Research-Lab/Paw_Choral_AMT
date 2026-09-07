#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Audit SATB annotations and PawCT target-assignment behaviour.

The command is CPU-only and never modifies the dataset. Because note files are
Python pickles, only run it on data from a trusted source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from choral_targets import ChoralTargetBuilder


VOICE_NAMES = ('S', 'A', 'T', 'B')
AUDITED_ASSIGNMENTS = (
    'range_prior',
    'ordered_continuity',
    'legacy_range_prior',
    'legacy_ordered_continuity',
)
SPLIT_FILES = {
    'train': ('train.json',),
    'validation': ('validation.json', 'valid.json'),
    'test': ('test.json',),
}


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _new_confusion():
    return {
        source: {target: 0 for target in VOICE_NAMES}
        for source in VOICE_NAMES
    }


def _pitch_summary(pitches):
    if not pitches:
        return {'count': 0}
    values = np.asarray(pitches, dtype=np.float64)
    percentiles = np.percentile(values, [1, 5, 50, 95, 99])
    return {
        'count': int(values.size),
        'min': int(np.min(values)),
        'p01': float(percentiles[0]),
        'p05': float(percentiles[1]),
        'median': float(percentiles[2]),
        'p95': float(percentiles[3]),
        'p99': float(percentiles[4]),
        'max': int(np.max(values)),
    }


def _make_builder_cfg(
    target_assignment: str,
    *,
    begin_note: int,
    classes_num: int,
    frames_per_second: float,
):
    return SimpleNamespace(
        feature=SimpleNamespace(
            begin_note=int(begin_note),
            classes_num=int(classes_num),
            frames_per_second=float(frames_per_second),
            segment_seconds=10.0,
        ),
        choral=SimpleNamespace(
            num_voices=4,
            voice_names=list(VOICE_NAMES),
            target_assignment=target_assignment,
            preserve_known_part_labels=True,
            voice_assignment_part_penalty=2.0,
            voice_assignment_continuity_weight=0.35,
            voice_assignment_overlap_penalty=4.0,
            voice_assignment_range_mins=[60, 55, 48, 40],
            voice_assignment_range_maxs=[88, 79, 72, 67],
            voice_assignment_range_margin=2.0,
            voice_assignment_mask_penalty=8.0,
            oc_gap_decay_seconds=2.0,
            oc_overlap_tolerance_seconds=0.05,
        ),
    )


class _AuditTargetBuilder(ChoralTargetBuilder):
    """Keep the source event object attached to each inferred voice."""

    def _events_to_voice_tuples(self, assigned_events):
        return list(assigned_events)


def _load_split_file(dataset_dir: Path, split: str):
    for filename in SPLIT_FILES[split]:
        path = dataset_dir / filename
        if not path.is_file():
            continue
        values = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f'{path} must contain a JSON list of recording IDs')
        return filename, sorted(set(value.strip() for value in values if value.strip()))
    return None, []


def _load_packed_recording_ids(packed_hdf5_dir: Path):
    packed_hdf5_dir = Path(packed_hdf5_dir)
    if not packed_hdf5_dir.is_dir():
        raise NotADirectoryError(
            f'Packed HDF5 directory does not exist: {packed_hdf5_dir}'
        )

    files_by_id = defaultdict(list)
    for path in packed_hdf5_dir.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.h5', '.hdf5'}:
            files_by_id[path.stem].append(
                path.relative_to(packed_hdf5_dir).as_posix()
            )
    duplicate_ids = {
        recording_id: sorted(filenames)
        for recording_id, filenames in files_by_id.items()
        if len(filenames) > 1
    }
    if duplicate_ids:
        raise ValueError(
            'Multiple packed HDF5 files resolve to the same recording ID: '
            f'{duplicate_ids}'
        )
    if not files_by_id:
        raise ValueError(f'No .h5 or .hdf5 files found in {packed_hdf5_dir}')
    return sorted(files_by_id)


def load_selected_recordings(
    dataset_dir: Path,
    split: str,
    *,
    packed_hdf5_dir: Path | None = None,
):
    selected_splits = tuple(SPLIT_FILES) if split == 'all' else (split,)
    split_files = {}
    memberships = defaultdict(list)
    missing_splits = []

    for split_name in selected_splits:
        filename, recording_ids = _load_split_file(dataset_dir, split_name)
        if filename is None:
            missing_splits.append(split_name)
            continue
        split_files[split_name] = filename
        for recording_id in recording_ids:
            memberships[recording_id].append(split_name)

    if split != 'all' and missing_splits:
        candidates = ', '.join(SPLIT_FILES[split])
        raise FileNotFoundError(
            f'Missing split file for {split!r} in {dataset_dir}; expected {candidates}'
        )
    if not memberships:
        raise ValueError(f'No recording IDs found for split={split!r} in {dataset_dir}')

    duplicate_memberships = {
        recording_id: sorted(split_names)
        for recording_id, split_names in memberships.items()
        if len(split_names) > 1
    }
    manifest_recording_ids = sorted(memberships)
    result = {
        'recording_ids': manifest_recording_ids,
        'split_files': dict(sorted(split_files.items())),
        'missing_splits': sorted(missing_splits),
        'duplicate_memberships': dict(sorted(duplicate_memberships.items())),
    }
    if packed_hdf5_dir is None:
        return result

    packed_recording_ids = _load_packed_recording_ids(packed_hdf5_dir)
    manifest_set = set(manifest_recording_ids)
    packed_set = set(packed_recording_ids)
    selected_recording_ids = sorted(manifest_set & packed_set)
    missing_recording_ids = sorted(manifest_set - packed_set)
    packed_not_in_selected_manifest_ids = sorted(packed_set - manifest_set)
    if not selected_recording_ids:
        raise ValueError(
            f'No packed HDF5 recording matches split={split!r} in {packed_hdf5_dir}'
        )
    result['recording_ids'] = selected_recording_ids
    result['packed_coverage'] = {
        'schema_version': 1,
        'match_semantics': 'exact_case_sensitive_filename_stem_existence_only',
        'manifest_recordings': len(manifest_recording_ids),
        'manifest_recording_ids_sha256': _recording_id_hash(
            manifest_recording_ids
        ),
        'packed_recordings_total': len(packed_recording_ids),
        'packed_recording_ids_sha256': _recording_id_hash(
            packed_recording_ids
        ),
        'intersection_recordings': len(selected_recording_ids),
        'intersection_recording_ids_sha256': _recording_id_hash(
            selected_recording_ids
        ),
        'manifest_coverage_ratio': _safe_ratio(
            len(selected_recording_ids),
            len(manifest_recording_ids),
        ),
        'packed_selection_ratio': _safe_ratio(
            len(selected_recording_ids),
            len(packed_recording_ids),
        ),
        'manifest_missing_packed_recordings': len(missing_recording_ids),
        'manifest_missing_packed_recording_ids': missing_recording_ids,
        'packed_not_in_selected_manifest_recordings': len(
            packed_not_in_selected_manifest_ids
        ),
        'packed_not_in_selected_manifest_recording_ids_sha256': _recording_id_hash(
            packed_not_in_selected_manifest_ids
        ),
    }
    return result


def _resolve_note_path(note_dir: Path, recording_id: str) -> Path:
    safe_name = Path(recording_id).name
    candidates = [note_dir / f'{safe_name}.pkl']
    suffix = Path(safe_name).suffix.lower()
    if suffix in {'.pkl', '.mid', '.midi', '.wav', '.mp3', '.flac', '.m4a'}:
        candidates.insert(0, note_dir / f'{Path(safe_name).stem}.pkl')
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = ', '.join(candidate.name for candidate in candidates)
    raise FileNotFoundError(
        f'Missing note annotation for recording {recording_id!r}; checked {rendered}'
    )


def _sanitize_note_bars(note_bars, begin_note: int, classes_num: int):
    if not isinstance(note_bars, list):
        raise ValueError('Each note pickle must contain a list of bar dictionaries')

    total_notes = 0
    in_model_range = 0
    malformed_notes = 0
    sanitized_bars = []
    for bar in note_bars:
        if not isinstance(bar, dict):
            continue
        sanitized_bar = {}
        for part_name, note_list in bar.items():
            if str(part_name).strip().lower() == 'measure':
                sanitized_bar[part_name] = note_list
                continue
            if not isinstance(note_list, list):
                continue
            sanitized_notes = []
            for note in note_list:
                try:
                    if len(note) < 5:
                        raise ValueError
                    midi_note = int(note[0])
                    onset_time = float(note[3])
                    offset_time = float(note[4])
                    if not np.isfinite(onset_time) or not np.isfinite(offset_time):
                        raise ValueError
                except (TypeError, ValueError, IndexError):
                    malformed_notes += 1
                    continue
                total_notes += 1
                if begin_note <= midi_note < begin_note + classes_num:
                    in_model_range += 1
                sanitized_notes.append(note)
            sanitized_bar[part_name] = sanitized_notes
        sanitized_bars.append(sanitized_bar)

    return sanitized_bars, {
        'total': total_notes,
        'in_model_range': in_model_range,
        'outside_model_range': total_notes - in_model_range,
        'malformed': malformed_notes,
    }


def _recording_id_hash(recording_ids) -> str:
    payload = ''.join(f'{recording_id}\n' for recording_id in sorted(recording_ids))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _initial_assignment_stats():
    return {
        method: {
            'assigned_notes': 0,
            'known_labels': 0,
            'retained_known_labels': 0,
            'changed_known_labels': 0,
            'confusion': _new_confusion(),
        }
        for method in AUDITED_ASSIGNMENTS
    }


def _finalize_assignment_stats(stats):
    finalized = {}
    for method in AUDITED_ASSIGNMENTS:
        values = stats[method]
        known = values['known_labels']
        finalized[method] = {
            **values,
            'retention_ratio': _safe_ratio(values['retained_known_labels'], known),
            'change_ratio': _safe_ratio(values['changed_known_labels'], known),
        }
    return finalized


def audit_dataset(
    dataset_dir,
    split: str,
    *,
    begin_note: int = 21,
    classes_num: int = 88,
    frames_per_second: float = 100.0,
    packed_hdf5_dir=None,
):
    dataset_dir = Path(dataset_dir)
    if classes_num <= 0:
        raise ValueError('classes_num must be positive')
    if frames_per_second <= 0:
        raise ValueError('frames_per_second must be positive')
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f'Dataset directory does not exist: {dataset_dir}')
    note_dir = dataset_dir / 'note'
    if not note_dir.is_dir():
        raise NotADirectoryError(f'Missing note directory: {note_dir}')

    selection = load_selected_recordings(
        dataset_dir,
        split,
        packed_hdf5_dir=packed_hdf5_dir,
    )
    recording_ids = selection['recording_ids']
    cfg_by_method = {
        method: _make_builder_cfg(
            method,
            begin_note=begin_note,
            classes_num=classes_num,
            frames_per_second=frames_per_second,
        )
        for method in AUDITED_ASSIGNMENTS
    }
    builders = {
        method: _AuditTargetBuilder(cfg_by_method[method])
        for method in AUDITED_ASSIGNMENTS
    }
    parser_builder = builders['range_prior']

    note_totals = Counter()
    canonical_distribution = Counter({voice: 0 for voice in VOICE_NAMES})
    canonical_pitches = {voice: [] for voice in VOICE_NAMES}
    onset_group_count = 0
    more_than_four_group_count = 0
    duplicate_group_count = 0
    duplicate_participating_notes = 0
    duplicate_excess_notes = 0
    assignment_stats = _initial_assignment_stats()

    for recording_id in recording_ids:
        note_path = _resolve_note_path(note_dir, recording_id)
        with note_path.open('rb') as handle:
            note_bars = pickle.load(handle)
        note_bars, file_note_counts = _sanitize_note_bars(
            note_bars,
            begin_note=begin_note,
            classes_num=classes_num,
        )
        note_totals.update(file_note_counts)

        source_events = parser_builder.note_bars_to_events(note_bars)
        for event in source_events:
            voice_idx = event['part_voice_idx']
            if voice_idx is not None:
                voice_name = VOICE_NAMES[voice_idx]
                canonical_distribution[voice_name] += 1
                canonical_pitches[voice_name].append(int(event['midi_note']))

        groups = defaultdict(list)
        for event in source_events:
            onset_frame = int(np.round(event['onset_time'] * frames_per_second))
            groups[onset_frame].append(event)
        onset_group_count += len(groups)
        for group in groups.values():
            if len(group) > len(VOICE_NAMES):
                more_than_four_group_count += 1
            known_counts = Counter(
                event['part_voice_idx']
                for event in group
                if event['part_voice_idx'] is not None
            )
            duplicate_counts = [count for count in known_counts.values() if count > 1]
            if duplicate_counts:
                duplicate_group_count += 1
                duplicate_participating_notes += sum(duplicate_counts)
                duplicate_excess_notes += sum(count - 1 for count in duplicate_counts)

        for method, builder in builders.items():
            # Compare assignment strategies over identical source-note
            # denominators. The later binary-head projection is a target
            # representability operation, not a label-assignment decision.
            assigned_events = builder.assign_voice_events(
                note_bars,
                project_representable=False,
            )
            method_stats = assignment_stats[method]
            method_stats['assigned_notes'] += len(assigned_events)
            for assigned_voice_idx, event in assigned_events:
                source_voice_idx = event['part_voice_idx']
                if source_voice_idx is None:
                    continue
                source_voice = VOICE_NAMES[source_voice_idx]
                assigned_voice = VOICE_NAMES[assigned_voice_idx]
                method_stats['known_labels'] += 1
                method_stats['confusion'][source_voice][assigned_voice] += 1
                if source_voice_idx == assigned_voice_idx:
                    method_stats['retained_known_labels'] += 1
                else:
                    method_stats['changed_known_labels'] += 1

    canonical_known = sum(canonical_distribution.values())
    in_model_range = int(note_totals['in_model_range'])
    dataset_result = {
        'name': dataset_dir.name,
        'split': split,
        'split_files': selection['split_files'],
        'missing_splits_for_all': selection['missing_splits'],
        'recordings': len(recording_ids),
        'recording_ids_sha256': _recording_id_hash(recording_ids),
        'duplicate_split_membership_count': len(selection['duplicate_memberships']),
        'duplicate_split_memberships': selection['duplicate_memberships'],
    }
    if 'packed_coverage' in selection:
        dataset_result['packed_coverage'] = selection['packed_coverage']

    result = {
        'schema_version': 1,
        'dataset': dataset_result,
        'analysis_config': {
            'begin_note': int(begin_note),
            'classes_num': int(classes_num),
            'onset_group_frames_per_second': float(frames_per_second),
            'voice_order': list(VOICE_NAMES),
            'preserve_known_part_labels': True,
            'range_mins': list(parser_builder.range_mins),
            'range_maxs': list(parser_builder.range_maxs),
            'range_margin': parser_builder.range_margin,
            'range_mask_penalty': parser_builder.mask_penalty,
            'part_label_mismatch_penalty': parser_builder.part_penalty,
            'continuity_weight': parser_builder.continuity_weight,
            'overlap_penalty': parser_builder.overlap_penalty,
            'oc_gap_decay_seconds': parser_builder.gap_decay_seconds,
            'oc_overlap_tolerance_seconds': parser_builder.overlap_tolerance_seconds,
        },
        'notes': {
            'total': int(note_totals['total']),
            'in_model_range': in_model_range,
            'outside_model_range': int(note_totals['outside_model_range']),
            'malformed': int(note_totals['malformed']),
            'canonical_known': canonical_known,
            'canonical_unknown_or_ambiguous': in_model_range - canonical_known,
            'canonical_known_ratio': _safe_ratio(canonical_known, in_model_range),
            'canonical_distribution': {
                voice: int(canonical_distribution[voice]) for voice in VOICE_NAMES
            },
            'canonical_pitch_statistics': {
                voice: _pitch_summary(canonical_pitches[voice])
                for voice in VOICE_NAMES
            },
        },
        'onset_groups': {
            'total': onset_group_count,
            'more_than_four': {
                'count': more_than_four_group_count,
                'ratio': _safe_ratio(more_than_four_group_count, onset_group_count),
            },
            'duplicate_canonical_voice': {
                'group_count': duplicate_group_count,
                'group_ratio': _safe_ratio(duplicate_group_count, onset_group_count),
                'participating_note_count': duplicate_participating_notes,
                'participating_note_ratio': _safe_ratio(
                    duplicate_participating_notes,
                    canonical_known,
                ),
                'excess_note_count': duplicate_excess_notes,
            },
        },
        'assignments': _finalize_assignment_stats(assignment_stats),
    }
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Audit choral note labels and RP/OC target construction (CPU-only, read-only).'
    )
    parser.add_argument('--dataset-dir', required=True, help='Dataset root containing split JSON files and note/*.pkl')
    parser.add_argument(
        '--split',
        required=True,
        choices=('train', 'validation', 'test', 'all'),
        help='Split to audit; validation accepts validation.json or valid.json.',
    )
    parser.add_argument('--output-json', default='', help='Optional path for the same JSON printed to stdout')
    parser.add_argument('--begin-note', type=int, default=21, help='Lowest modeled MIDI pitch (default: 21)')
    parser.add_argument('--classes-num', type=int, default=88, help='Number of modeled pitches (default: 88)')
    parser.add_argument(
        '--frames-per-second',
        type=float,
        default=100.0,
        help='Frame rate used to group near-synchronous onsets (default: 100).',
    )
    parser.add_argument(
        '--packed-hdf5-dir',
        default='',
        help=(
            'Optional directory of packed .h5/.hdf5 files. When set, audit '
            'only split recordings with a matching packed file and report '
            'manifest coverage; source annotations remain read-only.'
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = audit_dataset(
        args.dataset_dir,
        args.split,
        begin_note=args.begin_note,
        classes_num=args.classes_num,
        frames_per_second=args.frames_per_second,
        packed_hdf5_dir=args.packed_hdf5_dir or None,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f'{rendered}\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
