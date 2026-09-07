#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Evaluate RP/OC label recovery with deterministic training-note masking.

This is a CPU-only, read-only diagnostic over trusted YouChorale annotation
pickles.  It is deliberately restricted to a training manifest: the fixed
10/25/50% masks and all prior parameters must not become a test-tuning path.
Only masked canonical SATB notes are scored.  Unmasked labels estimate the
per-rate pitch ranges and provide the observed trajectory context used by OC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audit_choral_targets import (
    _recording_id_hash,
    _resolve_note_path,
    load_selected_recordings,
)
from choral_targets import (
    ChoralTargetBuilder,
    canonical_voice_code,
    require_complete_satb_reference,
)


SCHEMA_VERSION = 1
VOICE_NAMES = ('S', 'A', 'T', 'B')
METHODS = ('range_prior', 'ordered_continuity')
MASK_RATES_PERCENT = (10, 25, 50)
MASK_SALT = 'pawct-icassp2027-label-mask-v1'
CONDITIONS = ('aligned', 'cyclic_range_negative_control')

# These are fixed algorithm settings, not command-line search dimensions.
CONTINUITY_WEIGHT = 0.35
OVERLAP_PENALTY = 4.0
PART_LABEL_MISMATCH_PENALTY = 2.0
RANGE_MARGIN = 2.0
RANGE_MASK_PENALTY = 8.0
OC_GAP_DECAY_SECONDS = 2.0
OC_OVERLAP_TOLERANCE_SECONDS = 0.05


class _IdentityTargetBuilder(ChoralTargetBuilder):
    """Run the existing assignment logic while retaining source event IDs."""

    def note_bars_to_events(self, note_events):
        return list(note_events)

    def _events_to_voice_tuples(self, assigned_events):
        return list(assigned_events)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _new_confusion():
    return {
        truth: {prediction: 0 for prediction in VOICE_NAMES}
        for truth in VOICE_NAMES
    }


def _event_identity_payload(
    recording_id: str,
    bar_index: int,
    part_index: int,
    note_index: int,
    midi_note: int,
    onset_time: float,
    offset_time: float,
) -> bytes:
    value = [
        MASK_SALT,
        recording_id,
        int(bar_index),
        int(part_index),
        int(note_index),
        int(midi_note),
        float(onset_time).hex(),
        float(offset_time).hex(),
    ]
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(',', ':'),
    ).encode('utf-8')


def _event_id(
    recording_id: str,
    bar_index: int,
    part_index: int,
    note_index: int,
    midi_note: int,
    onset_time: float,
    offset_time: float,
) -> str:
    return hashlib.sha256(
        _event_identity_payload(
            recording_id,
            bar_index,
            part_index,
            note_index,
            midi_note,
            onset_time,
            offset_time,
        )
    ).hexdigest()


def _is_masked(event_id: str, rate_percent: int) -> bool:
    if rate_percent not in MASK_RATES_PERCENT:
        raise ValueError(f'Unsupported pre-registered mask rate: {rate_percent}')
    rank = int(event_id, 16)
    threshold = (rate_percent * (1 << 256)) // 100
    return rank < threshold


def _extract_reference_events(recording_id: str, note_bars, builder):
    events = []
    seen_ids = set()
    for bar_index, bar in enumerate(note_bars):
        for part_index, (part_name, note_list) in enumerate(bar.items()):
            if str(part_name).strip().lower() == 'measure':
                continue
            voice_code = canonical_voice_code(part_name)
            # require_complete_satb_reference has already rejected this case.
            if voice_code not in builder.voice_code_to_index:
                raise RuntimeError(
                    f'Canonical voice {voice_code!r} is absent from voice_names'
                )
            truth_voice_idx = builder.voice_code_to_index[voice_code]
            for note_index, note in enumerate(note_list):
                midi_note = int(float(note[0]))
                onset_time = float(note[3])
                offset_time = float(note[4])
                identity = _event_id(
                    recording_id,
                    bar_index,
                    part_index,
                    note_index,
                    midi_note,
                    onset_time,
                    offset_time,
                )
                if identity in seen_ids:
                    raise RuntimeError(
                        f'Event identity collision in recording {recording_id!r}'
                    )
                seen_ids.add(identity)
                events.append({
                    'event_id': identity,
                    'part_name': str(part_name),
                    'truth_voice_idx': truth_voice_idx,
                    'midi_note': midi_note,
                    'onset_time': onset_time,
                    'offset_time': offset_time,
                })

    events.sort(key=lambda event: (
        event['onset_time'],
        -event['midi_note'],
        event['offset_time'],
    ))
    return events


def _load_reference_events(
    note_dir: Path,
    recording_id: str,
    validation_builder,
    *,
    expected_pickle_sha256: str | None = None,
):
    note_path = _resolve_note_path(note_dir, recording_id)
    payload = note_path.read_bytes()
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    if (
        expected_pickle_sha256 is not None
        and payload_sha256 != expected_pickle_sha256
    ):
        raise RuntimeError(
            f'Annotation changed between audit passes for {recording_id!r}'
        )
    note_bars = pickle.loads(payload)
    require_complete_satb_reference(
        note_bars,
        note_path.name,
        begin_note=validation_builder.begin_note,
        classes_num=validation_builder.classes_num,
    )
    events = _extract_reference_events(
        recording_id,
        note_bars,
        validation_builder,
    )
    return payload_sha256, events


def _histogram_value_at_rank(histogram: Counter, rank: int) -> int:
    if rank < 0:
        raise ValueError('rank must be non-negative')
    consumed = 0
    for value in sorted(histogram):
        consumed += histogram[value]
        if rank < consumed:
            return int(value)
    raise IndexError(f'Histogram rank {rank} is out of bounds')


def _linear_percentile(histogram: Counter, percentile: int) -> float:
    count = sum(histogram.values())
    if count <= 0:
        raise ValueError('Cannot estimate a percentile from zero observations')
    position = Fraction((count - 1) * int(percentile), 100)
    lower_rank = position.numerator // position.denominator
    upper_rank = math.ceil(position)
    lower = _histogram_value_at_rank(histogram, lower_rank)
    upper = _histogram_value_at_rank(histogram, upper_rank)
    fraction = position - lower_rank
    return float(Fraction(lower) + fraction * (upper - lower))


def _make_builder_cfg(
    range_mins,
    range_maxs,
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
            target_assignment='part_name',
            preserve_known_part_labels=True,
            voice_assignment_part_penalty=PART_LABEL_MISMATCH_PENALTY,
            voice_assignment_continuity_weight=CONTINUITY_WEIGHT,
            voice_assignment_overlap_penalty=OVERLAP_PENALTY,
            voice_assignment_range_mins=list(range_mins),
            voice_assignment_range_maxs=list(range_maxs),
            voice_assignment_range_margin=RANGE_MARGIN,
            voice_assignment_mask_penalty=RANGE_MASK_PENALTY,
            oc_gap_decay_seconds=OC_GAP_DECAY_SECONDS,
            oc_overlap_tolerance_seconds=OC_OVERLAP_TOLERANCE_SECONDS,
        ),
    )


def _condition_ranges(range_mins, range_maxs, condition: str):
    if condition == 'aligned':
        return list(range_mins), list(range_maxs), list(VOICE_NAMES)
    if condition == 'cyclic_range_negative_control':
        indices = [1, 2, 3, 0]
        return (
            [range_mins[index] for index in indices],
            [range_maxs[index] for index in indices],
            [VOICE_NAMES[index] for index in indices],
        )
    raise ValueError(f'Unknown condition: {condition}')


def _prepare_masked_events(reference_events, rate_percent: int):
    prepared = []
    truth_by_id = {}
    masked_ids = set()
    for reference in reference_events:
        identity = reference['event_id']
        truth_voice_idx = reference['truth_voice_idx']
        masked = _is_masked(identity, rate_percent)
        truth_by_id[identity] = truth_voice_idx
        if masked:
            masked_ids.add(identity)
        prepared.append({
            'event_id': identity,
            # Mask both modern and legacy label fields. The true label exists
            # only in truth_by_id, which ChoralTargetBuilder never receives.
            'part_name': None if masked else reference['part_name'],
            'part_voice_idx': None if masked else truth_voice_idx,
            'legacy_part_voice_idx': None,
            'midi_note': reference['midi_note'],
            'onset_time': reference['onset_time'],
            'offset_time': reference['offset_time'],
        })
    return prepared, truth_by_id, masked_ids


def _finalize_metrics(confusion):
    evaluated = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[voice][voice] for voice in VOICE_NAMES)
    per_voice = {}
    f1_values = []
    for voice in VOICE_NAMES:
        true_positive = confusion[voice][voice]
        false_positive = sum(
            confusion[truth][voice]
            for truth in VOICE_NAMES
            if truth != voice
        )
        false_negative = sum(
            confusion[voice][prediction]
            for prediction in VOICE_NAMES
            if prediction != voice
        )
        support = sum(confusion[voice].values())
        predicted = sum(confusion[truth][voice] for truth in VOICE_NAMES)
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
        f1_values.append(f1)
        per_voice[voice] = {
            'support': support,
            'predicted': predicted,
            'true_positive': true_positive,
            'false_positive': false_positive,
            'false_negative': false_negative,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    return {
        'evaluated_masked_notes': evaluated,
        'correct': correct,
        'accuracy': _safe_ratio(correct, evaluated),
        'macro_f1': sum(f1_values) / len(VOICE_NAMES),
        'per_voice': per_voice,
        'confusion': confusion,
    }


def evaluate_label_masking(
    dataset_dir,
    split: str = 'train',
    *,
    recording_manifest=None,
    packed_hdf5_dir=None,
    begin_note: int = 21,
    classes_num: int = 88,
    frames_per_second: float = 100.0,
):
    if split != 'train':
        raise ValueError(
            'The pre-registered label-masking diagnostic is train-only; '
            'validation/test labels must not be used to tune RP/OC priors.'
        )
    if (
        recording_manifest is not None
        and Path(recording_manifest).name != 'train.json'
    ):
        raise ValueError(
            'The train-only diagnostic accepts an explicit manifest only when '
            'its filename is train.json'
        )
    if classes_num <= 0:
        raise ValueError('classes_num must be positive')
    if not math.isfinite(frames_per_second) or frames_per_second <= 0:
        raise ValueError('frames_per_second must be finite and positive')

    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f'Dataset directory does not exist: {dataset_dir}')
    note_dir = dataset_dir / 'note'
    if not note_dir.is_dir():
        raise NotADirectoryError(f'Missing note directory: {note_dir}')

    selection = load_selected_recordings(
        dataset_dir,
        split,
        recording_manifest=recording_manifest,
        packed_hdf5_dir=packed_hdf5_dir,
    )
    recording_ids = selection['recording_ids']
    validation_cfg = _make_builder_cfg(
        [0.0] * 4,
        [0.0] * 4,
        begin_note=begin_note,
        classes_num=classes_num,
        frames_per_second=frames_per_second,
    )
    validation_builder = _IdentityTargetBuilder(validation_cfg)

    eligible_notes = 0
    eligible_by_voice = Counter({voice: 0 for voice in VOICE_NAMES})
    masked_by_rate = {
        rate: Counter({voice: 0 for voice in VOICE_NAMES})
        for rate in MASK_RATES_PERCENT
    }
    recordings_with_masked = {rate: 0 for rate in MASK_RATES_PERCENT}
    unmasked_pitch_histograms = {
        rate: [Counter() for _ in VOICE_NAMES]
        for rate in MASK_RATES_PERCENT
    }
    annotation_hashes = {}
    eligible_sequence_hasher = hashlib.sha256()
    masked_sequence_hashers = {
        rate: hashlib.sha256() for rate in MASK_RATES_PERCENT
    }

    # Pass 1 validates the entire reference, fixes denominators, and estimates
    # each rate's prior exclusively from labels that remain visible.
    for recording_id in recording_ids:
        payload_sha256, events = _load_reference_events(
            note_dir,
            recording_id,
            validation_builder,
        )
        annotation_hashes[recording_id] = payload_sha256
        eligible_notes += len(events)
        recording_mask_counts = {rate: 0 for rate in MASK_RATES_PERCENT}
        for event in events:
            identity = event['event_id']
            eligible_sequence_hasher.update(f'{identity}\n'.encode('ascii'))
            voice_idx = event['truth_voice_idx']
            voice_name = VOICE_NAMES[voice_idx]
            eligible_by_voice[voice_name] += 1
            for rate in MASK_RATES_PERCENT:
                if _is_masked(identity, rate):
                    masked_by_rate[rate][voice_name] += 1
                    recording_mask_counts[rate] += 1
                    masked_sequence_hashers[rate].update(
                        f'{identity}\n'.encode('ascii')
                    )
                else:
                    unmasked_pitch_histograms[rate][voice_idx][
                        event['midi_note']
                    ] += 1
        for rate, count in recording_mask_counts.items():
            if count:
                recordings_with_masked[rate] += 1

    if eligible_notes == 0:
        raise ValueError('The selected training manifest contains no eligible notes')

    ranges_by_rate = {}
    for rate in MASK_RATES_PERCENT:
        missing_masked = [
            voice for voice in VOICE_NAMES if masked_by_rate[rate][voice] == 0
        ]
        if missing_masked:
            raise ValueError(
                f'{rate}% mask has no held-out notes for voices: {missing_masked}'
            )
        missing_visible = [
            VOICE_NAMES[index]
            for index, histogram in enumerate(unmasked_pitch_histograms[rate])
            if not histogram
        ]
        if missing_visible:
            raise ValueError(
                f'{rate}% mask leaves no visible notes for voices: {missing_visible}'
            )
        ranges_by_rate[rate] = {
            'mins': [
                _linear_percentile(histogram, 1)
                for histogram in unmasked_pitch_histograms[rate]
            ],
            'maxs': [
                _linear_percentile(histogram, 99)
                for histogram in unmasked_pitch_histograms[rate]
            ],
            'visible_notes_by_voice': {
                VOICE_NAMES[index]: sum(histogram.values())
                for index, histogram in enumerate(
                    unmasked_pitch_histograms[rate]
                )
            },
        }

    confusions = {
        rate: {
            condition: {method: _new_confusion() for method in METHODS}
            for condition in CONDITIONS
        }
        for rate in MASK_RATES_PERCENT
    }
    builders = {}
    mappings = {}
    for rate in MASK_RATES_PERCENT:
        for condition in CONDITIONS:
            range_mins, range_maxs, source_voices = _condition_ranges(
                ranges_by_rate[rate]['mins'],
                ranges_by_rate[rate]['maxs'],
                condition,
            )
            mappings[(rate, condition)] = {
                VOICE_NAMES[index]: source_voices[index]
                for index in range(len(VOICE_NAMES))
            }
            cfg = _make_builder_cfg(
                range_mins,
                range_maxs,
                begin_note=begin_note,
                classes_num=classes_num,
                frames_per_second=frames_per_second,
            )
            for method in METHODS:
                builders[(rate, condition, method)] = _IdentityTargetBuilder(
                    cfg,
                    target_assignment=method,
                )

    # Pass 2 verifies immutable input bytes, runs the existing builder, and
    # evaluates exactly the same masked event IDs in every condition/method.
    for recording_id in recording_ids:
        _, reference_events = _load_reference_events(
            note_dir,
            recording_id,
            validation_builder,
            expected_pickle_sha256=annotation_hashes[recording_id],
        )
        for rate in MASK_RATES_PERCENT:
            prepared, truth_by_id, masked_ids = _prepare_masked_events(
                reference_events,
                rate,
            )
            for condition in CONDITIONS:
                for method in METHODS:
                    assigned = builders[
                        (rate, condition, method)
                    ].assign_voice_events(
                        prepared,
                        project_representable=False,
                    )
                    if len(assigned) != len(prepared):
                        raise RuntimeError(
                            f'{method} changed the source-note denominator for '
                            f'{recording_id!r}'
                        )
                    seen_assigned_ids = set()
                    for predicted_voice_idx, event in assigned:
                        identity = event['event_id']
                        if identity in seen_assigned_ids:
                            raise RuntimeError(
                                f'{method} emitted duplicate event {identity}'
                            )
                        seen_assigned_ids.add(identity)
                        truth_voice_idx = truth_by_id[identity]
                        if identity not in masked_ids:
                            if predicted_voice_idx != truth_voice_idx:
                                raise RuntimeError(
                                    f'{method} failed to anchor a visible label'
                                )
                            continue
                        truth_voice = VOICE_NAMES[truth_voice_idx]
                        predicted_voice = VOICE_NAMES[predicted_voice_idx]
                        confusions[rate][condition][method][truth_voice][
                            predicted_voice
                        ] += 1
                    if seen_assigned_ids != set(truth_by_id):
                        raise RuntimeError(
                            f'{method} did not return the exact source event set'
                        )

    annotation_sequence_hasher = hashlib.sha256()
    for recording_id in sorted(annotation_hashes):
        encoded = json.dumps(
            [recording_id, annotation_hashes[recording_id]],
            separators=(',', ':'),
        )
        annotation_sequence_hasher.update(f'{encoded}\n'.encode('utf-8'))

    rate_results = {}
    previous_masked = 0
    for rate in MASK_RATES_PERCENT:
        masked_total = sum(masked_by_rate[rate].values())
        if masked_total < previous_masked:
            raise RuntimeError('Pre-registered mask sets are not nested')
        previous_masked = masked_total
        condition_results = {}
        for condition in CONDITIONS:
            method_results = {
                method: _finalize_metrics(confusions[rate][condition][method])
                for method in METHODS
            }
            for method, metrics in method_results.items():
                if metrics['evaluated_masked_notes'] != masked_total:
                    raise RuntimeError(
                        f'{condition}/{method} denominator does not match mask'
                    )
            range_mins, range_maxs, _ = _condition_ranges(
                ranges_by_rate[rate]['mins'],
                ranges_by_rate[rate]['maxs'],
                condition,
            )
            condition_results[condition] = {
                'evaluated_masked_event_sequence_sha256': (
                    masked_sequence_hashers[rate].hexdigest()
                ),
                'range_source_voice_for_output_voice': mappings[(rate, condition)],
                'range_mins': range_mins,
                'range_maxs': range_maxs,
                'methods': method_results,
            }
        rate_results[str(rate)] = {
            'requested_mask_rate_percent': rate,
            'eligible_notes': eligible_notes,
            'masked_notes': masked_total,
            'visible_notes': eligible_notes - masked_total,
            'realized_mask_rate': _safe_ratio(masked_total, eligible_notes),
            'masked_notes_by_true_voice': {
                voice: masked_by_rate[rate][voice] for voice in VOICE_NAMES
            },
            'recordings_with_masked_notes': recordings_with_masked[rate],
            'masked_event_sequence_sha256': masked_sequence_hashers[
                rate
            ].hexdigest(),
            'range_estimation': {
                'source': 'same_training_manifest_visible_labels_only',
                'percentiles': [1, 99],
                'method': 'exact_linear_type7_position_(n-1)*q',
                **ranges_by_rate[rate],
            },
            'conditions': condition_results,
        }

    dataset_result = {
        'name': dataset_dir.name,
        'split': split,
        'split_files': selection['split_files'],
        'recordings': len(recording_ids),
        'recording_ids_sha256': _recording_id_hash(recording_ids),
        'eligible_notes': eligible_notes,
        'eligible_notes_by_voice': {
            voice: eligible_by_voice[voice] for voice in VOICE_NAMES
        },
        'eligible_event_sequence_sha256': eligible_sequence_hasher.hexdigest(),
        'annotation_pickle_sequence_sha256': annotation_sequence_hasher.hexdigest(),
        'duplicate_split_membership_count': len(selection['duplicate_memberships']),
        'duplicate_split_memberships': selection['duplicate_memberships'],
    }
    if 'recording_manifest' in selection:
        dataset_result['recording_manifest'] = selection['recording_manifest']
    if 'packed_coverage' in selection:
        dataset_result['packed_coverage'] = selection['packed_coverage']

    return {
        'schema_version': SCHEMA_VERSION,
        'protocol': {
            'name': 'pawct_train_label_masking_v1',
            'intended_use': 'training_only_semi_supervised_label_recovery_diagnostic',
            'test_tuning_prohibited': True,
            'methods': list(METHODS),
            'mask_rates_percent': list(MASK_RATES_PERCENT),
            'mask_salt': MASK_SALT,
            'mask_hash': (
                'SHA256(canonical_JSON([salt,recording_id,bar_index,'
                'part_index,note_index,midi_note,onset_float_hex,offset_float_hex]))'
            ),
            'mask_rule': 'unsigned_digest_integer < floor(rate_percent*2^256/100)',
            'mask_sets_nested': True,
            'score_scope': 'masked_canonical_notes_only',
            'visible_label_role': (
                'range estimation for RP/OC and within-recording trajectory '
                'context for OC'
            ),
            'binary_head_projection': False,
            'divisi_semantics': (
                'each source note remains distinct; S1/S2 etc. share their '
                'canonical SATB truth without target projection or merging'
            ),
            'negative_control': (
                'fixed cyclic range mapping S<-A,A<-T,T<-B,B<-S with identical '
                'masked event IDs and visible labels'
            ),
            'metric_definition': (
                'accuracy and one-vs-rest per-voice F1 over masked notes; '
                'macro-F1 is the unweighted S/A/T/B mean with zero_division=0'
            ),
        },
        'dataset': dataset_result,
        'assignment_config': {
            'begin_note': int(begin_note),
            'classes_num': int(classes_num),
            'frames_per_second': float(frames_per_second),
            'voice_order': list(VOICE_NAMES),
            'preserve_known_part_labels': True,
            'part_label_mismatch_penalty': PART_LABEL_MISMATCH_PENALTY,
            'continuity_weight': CONTINUITY_WEIGHT,
            'overlap_penalty': OVERLAP_PENALTY,
            'range_margin': RANGE_MARGIN,
            'range_mask_penalty': RANGE_MASK_PENALTY,
            'oc_gap_decay_seconds': OC_GAP_DECAY_SECONDS,
            'oc_overlap_tolerance_seconds': OC_OVERLAP_TOLERANCE_SECONDS,
            'active_cost_terms': {
                'range_prior': ['range_cost'],
                'ordered_continuity': [
                    'range_cost',
                    'gap_decayed_pitch_continuity',
                    'overlap_penalty',
                ],
            },
            'inactive_for_evaluated_methods': [
                'part_label_mismatch_penalty_for_masked_notes',
                'range_margin',
                'range_mask_penalty',
            ],
        },
        'rates': rate_results,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'CPU-only deterministic RP/OC label-masking diagnostic over a '
            'trusted YouChorale training manifest.'
        )
    )
    parser.add_argument(
        '--dataset-dir',
        required=True,
        help='YouChorale root containing note/*.pkl and split JSON files.',
    )
    parser.add_argument(
        '--split',
        required=True,
        choices=('train',),
        help='Intentionally restricted to train to prevent test-set tuning.',
    )
    parser.add_argument(
        '--recording-manifest',
        default='',
        help=(
            'Optional strict frozen training-manifest JSON; replaces the '
            'dataset-dir train.json selection.'
        ),
    )
    parser.add_argument(
        '--packed-hdf5-dir',
        default='',
        help=(
            'Optional recursive .h5/.hdf5 filename-stem availability filter; '
            'HDF5 contents are never opened.'
        ),
    )
    parser.add_argument(
        '--output-json',
        default='',
        help='Optional path for the same JSON printed to stdout.',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = evaluate_label_masking(
        args.dataset_dir,
        args.split,
        recording_manifest=args.recording_manifest or None,
        packed_hdf5_dir=args.packed_hdf5_dir or None,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f'{rendered}\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
