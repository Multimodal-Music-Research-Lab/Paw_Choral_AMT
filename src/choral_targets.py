# SPDX-License-Identifier: Apache-2.0

"""SATB target construction for PawCT training.

Range-prior (RP) and ordered-continuity (OC) are target-label construction
strategies, not inference-time model components. Their canonical variants keep
trusted SATB annotations and only infer a voice for unknown or ambiguous part
labels. Explicit ``legacy_*`` variants reproduce the earlier behaviour that
reassigned every note.
"""

from __future__ import annotations

import math
import re
from itertools import combinations

import numpy as np

from canonical_union import canonical_union_events


TARGET_ASSIGNMENT_ALIASES = {
    'part_name': 'part_name',
    'range_prior': 'range_prior',
    'rp': 'range_prior',
    'ordered_continuity': 'ordered_continuity',
    'oc': 'ordered_continuity',
    'range_masked_continuity': 'range_masked_continuity',
    'legacy_range_prior': 'legacy_range_prior',
    'legacy_ordered_continuity': 'legacy_ordered_continuity',
    'legacy_range_masked_continuity': 'legacy_range_masked_continuity',
}

_FULL_VOICE_NAMES = {
    'soprano': 'S',
    'sop': 'S',
    'alto': 'A',
    'alt': 'A',
    'tenor': 'T',
    'ten': 'T',
    'bass': 'B',
}


def canonical_voice_code(part_name) -> str | None:
    """Return an exact SATB code without guessing from arbitrary metadata.

    Divisi labels such as ``S1`` and full names such as ``Soprano_2`` are
    accepted.  Prefix matches are intentionally rejected: ``soloist``,
    ``ambiguous``, ``track`` and ``bassoon`` are not trusted voice labels.
    """

    if part_name is None:
        return None
    label = str(part_name).strip()
    if not label:
        return None

    compact = re.sub(r'[\s_.-]+', '', label).lower()
    match = re.fullmatch(r'([satb])\d*', compact)
    if match:
        return match.group(1).upper()

    for full_name, voice_code in _FULL_VOICE_NAMES.items():
        if re.fullmatch(rf'{full_name}\d*', compact):
            return voice_code
    return None


def merge_quantized_voice_events(
    voice_events,
    frames_per_second,
    *,
    quantization_origin=0.0,
):
    """Collapse events that one binary voice head cannot distinguish.

    Events are merged only when their assigned/canonical voice, MIDI pitch,
    and quantized onset frame are identical.  The retained event uses the
    earliest source onset and latest source offset.  Keeping the voice in the
    key prevents cross-voice unisons from being collapsed.
    """

    frames_per_second = float(frames_per_second)
    if not math.isfinite(frames_per_second) or frames_per_second <= 0.0:
        raise ValueError(
            'frames_per_second must be finite and positive, '
            f'got {frames_per_second!r}'
        )
    quantization_origin = float(quantization_origin)
    if not math.isfinite(quantization_origin):
        raise ValueError(
            'quantization_origin must be finite, '
            f'got {quantization_origin!r}'
        )

    merged = {}
    event_style = None
    for item in voice_events:
        if len(item) == 2 and isinstance(item[1], dict):
            voice, source_event = item
            midi_note = source_event['midi_note']
            onset_time = source_event['onset_time']
            offset_time = source_event['offset_time']
            item_style = 'mapping'
        elif len(item) == 4:
            voice, midi_note, onset_time, offset_time = item
            source_event = None
            item_style = 'tuple'
        else:
            raise ValueError(
                'voice_events must contain (voice, event_dict) pairs or '
                '(voice, pitch, onset, offset) tuples'
            )
        if event_style is None:
            event_style = item_style
        elif event_style != item_style:
            raise ValueError('voice_events cannot mix pair and tuple representations')

        midi_note = int(midi_note)
        onset_time = float(onset_time)
        offset_time = float(offset_time)
        onset_frame = int(
            np.round(
                (onset_time - quantization_origin) * frames_per_second
            )
        )
        key = (voice, midi_note, onset_frame)
        if key not in merged:
            merged[key] = {
                'voice': voice,
                'midi_note': midi_note,
                'onset_time': onset_time,
                'offset_time': offset_time,
                'source_event': dict(source_event) if source_event is not None else None,
            }
        else:
            retained = merged[key]
            if onset_time < retained['onset_time']:
                retained['onset_time'] = onset_time
                if source_event is not None:
                    retained['source_event'] = dict(source_event)
            retained['offset_time'] = max(retained['offset_time'], offset_time)

    ordered = sorted(
        merged.values(),
        key=lambda event: (
            float(event['onset_time']),
            str(event['voice']),
            -int(event['midi_note']),
            float(event['offset_time']),
        ),
    )
    if event_style == 'mapping':
        result = []
        for event in ordered:
            source_event = event['source_event']
            source_event['onset_time'] = event['onset_time']
            source_event['offset_time'] = event['offset_time']
            result.append((event['voice'], source_event))
        return result
    return [
        (
            event['voice'],
            event['midi_note'],
            event['onset_time'],
            event['offset_time'],
        )
        for event in ordered
    ]


def unlabelled_note_counts(note_bars) -> dict[str, int]:
    """Count note events whose part label cannot support SATB evaluation."""

    counts = {}
    for bar in note_bars:
        if not isinstance(bar, dict):
            continue
        for part_name, note_list in bar.items():
            if str(part_name).strip().lower() == 'measure' or not isinstance(note_list, list):
                continue
            note_count = sum(
                1
                for note in note_list
                if isinstance(note, (list, tuple, np.ndarray)) and len(note) >= 5
            )
            if note_count and canonical_voice_code(part_name) is None:
                label = str(part_name)
                counts[label] = counts.get(label, 0) + note_count
    return dict(sorted(counts.items()))


def require_complete_satb_reference(
    note_bars,
    source='<in-memory>',
    *,
    begin_note: int | None = None,
    classes_num: int | None = None,
    recording_duration: float | None = None,
) -> None:
    """Fail when primary SATB metrics would have undefined/impossible notes."""

    if not isinstance(note_bars, list):
        raise RuntimeError(
            'Formal SATB evaluation requires note_bars to be a materialized list; '
            f'top_level_schema={type(note_bars).__name__} in {source}'
        )
    if recording_duration is not None:
        recording_duration = float(recording_duration)
        if not math.isfinite(recording_duration) or recording_duration < 0.0:
            raise ValueError(
                'recording_duration must be finite and non-negative, '
                f'got {recording_duration!r}'
            )

    unknown = {}
    schema_errors = []
    invalid_values = []
    outside_range = []
    outside_duration = []
    upper_note = (
        int(begin_note) + int(classes_num)
        if begin_note is not None and classes_num is not None
        else None
    )
    for bar_index, bar in enumerate(note_bars):
        if not isinstance(bar, dict):
            schema_errors.append(f'bar[{bar_index}]={type(bar).__name__}')
            continue
        for part_name, note_list in bar.items():
            if str(part_name).strip().lower() == 'measure':
                continue
            if not isinstance(note_list, list):
                schema_errors.append(
                    f'bar[{bar_index}][{part_name!r}]={type(note_list).__name__}'
                )
                continue
            voice_code = canonical_voice_code(part_name)
            for note_index, note in enumerate(note_list):
                location = f'bar[{bar_index}][{part_name!r}][{note_index}]'
                if not isinstance(note, (list, tuple, np.ndarray)) or len(note) < 5:
                    schema_errors.append(location)
                    continue
                try:
                    pitch_value = float(note[0])
                    onset_time = float(note[3])
                    offset_time = float(note[4])
                except (TypeError, ValueError):
                    invalid_values.append(location)
                    continue
                if (
                    not np.isfinite(pitch_value)
                    or not float(pitch_value).is_integer()
                    or not np.isfinite(onset_time)
                    or not np.isfinite(offset_time)
                    or offset_time <= onset_time
                ):
                    invalid_values.append(location)
                    continue
                midi_note = int(pitch_value)
                if upper_note is not None and not int(begin_note) <= midi_note < upper_note:
                    outside_range.append((location, midi_note))
                if recording_duration is not None and (
                    onset_time < 0.0
                    or offset_time < 0.0
                    or onset_time > recording_duration
                    or offset_time > recording_duration
                ):
                    outside_duration.append(
                        (location, onset_time, offset_time, recording_duration)
                    )
                if voice_code is None:
                    label = str(part_name)
                    unknown[label] = unknown.get(label, 0) + 1

    problems = []
    if unknown:
        problems.append(f'unlabelled_counts={dict(sorted(unknown.items()))}')
    if schema_errors:
        problems.append(f'schema_errors={schema_errors[:10]}')
    if invalid_values:
        problems.append(f'invalid_note_values={invalid_values[:10]}')
    if outside_range:
        problems.append(f'outside_model_range={outside_range[:10]}')
    if outside_duration:
        problems.append(f'outside_recording_duration={outside_duration[:10]}')
    if problems:
        raise RuntimeError(
            'Formal SATB evaluation requires every reference note to have a canonical '
            'part, valid schema/timing, and a model-representable pitch; '
            + '; '.join(problems)
            + f' in {source}. Define and preregister an explicit data policy first.'
        )


def resolve_target_assignment(cfg) -> str:
    """Return the canonical SATB target-assignment strategy.

    ``choral.target_assignment`` is authoritative when present. Historical
    configurations that do not define it may continue to use
    ``choral.voice_assignment_method``.
    """

    choral_cfg = getattr(cfg, 'choral', None)
    value = getattr(choral_cfg, 'target_assignment', None)
    used_legacy_key = value is None or not str(value).strip()
    if used_legacy_key:
        value = getattr(choral_cfg, 'voice_assignment_method', None)
    if value is None or not str(value).strip():
        value = 'part_name'

    key = str(value).strip().lower().replace('-', '_')
    try:
        resolved = TARGET_ASSIGNMENT_ALIASES[key]
        # Before ``target_assignment`` existed, every RP/OC variant rewrote all
        # notes. Preserve that behavior when an old config uses only the old
        # key; canonical new runs must opt into the anchored semantics through
        # ``target_assignment``.
        if used_legacy_key and resolved in {
            'range_prior',
            'ordered_continuity',
            'range_masked_continuity',
        }:
            return f'legacy_{resolved}'
        return resolved
    except KeyError as exc:
        supported = ', '.join(sorted(TARGET_ASSIGNMENT_ALIASES))
        raise ValueError(
            f'Unsupported choral target assignment {value!r}; expected one of: {supported}'
        ) from exc


class ChoralTargetBuilder:
    """Convert note-bar annotations into frame-wise SATB training targets."""

    def __init__(
        self,
        cfg,
        segment_seconds: float | None = None,
        target_assignment: str | None = None,
    ):
        self.cfg = cfg
        self.segment_seconds = float(
            cfg.feature.segment_seconds if segment_seconds is None else segment_seconds
        )
        self.frames_per_second = float(cfg.feature.frames_per_second)
        self.frames_num = int(round(self.segment_seconds * self.frames_per_second)) + 1
        self.begin_note = int(cfg.feature.begin_note)
        self.classes_num = int(cfg.feature.classes_num)

        choral_cfg = cfg.choral
        self.num_voices = int(getattr(choral_cfg, 'num_voices', 4))
        self.voice_names = list(getattr(choral_cfg, 'voice_names', ['S', 'A', 'T', 'B']))
        if len(self.voice_names) != self.num_voices:
            raise ValueError(
                f'voice_names must contain {self.num_voices} entries, got {len(self.voice_names)}'
            )
        self.voice_code_to_index = {
            str(name).strip()[0].upper(): index
            for index, name in enumerate(self.voice_names)
            if str(name).strip()
        }

        if target_assignment is None:
            self.target_assignment = resolve_target_assignment(cfg)
        else:
            assignment_key = str(target_assignment).strip().lower().replace('-', '_')
            if assignment_key not in TARGET_ASSIGNMENT_ALIASES:
                supported = ', '.join(sorted(TARGET_ASSIGNMENT_ALIASES))
                raise ValueError(
                    f'Unsupported choral target assignment {target_assignment!r}; '
                    f'expected one of: {supported}'
                )
            self.target_assignment = TARGET_ASSIGNMENT_ALIASES[assignment_key]
        self.preserve_known_part_labels = bool(
            getattr(choral_cfg, 'preserve_known_part_labels', True)
        )
        self.part_penalty = float(getattr(choral_cfg, 'voice_assignment_part_penalty', 2.0))
        self.continuity_weight = float(
            getattr(choral_cfg, 'voice_assignment_continuity_weight', 0.35)
        )
        self.overlap_penalty = float(
            getattr(choral_cfg, 'voice_assignment_overlap_penalty', 4.0)
        )
        self.range_mins = self._normalize_voice_setting(
            getattr(choral_cfg, 'voice_assignment_range_mins', None),
            [60, 55, 48, 40],
        )
        self.range_maxs = self._normalize_voice_setting(
            getattr(choral_cfg, 'voice_assignment_range_maxs', None),
            [88, 79, 72, 67],
        )
        self.range_margin = float(getattr(choral_cfg, 'voice_assignment_range_margin', 2.0))
        self.mask_penalty = float(getattr(choral_cfg, 'voice_assignment_mask_penalty', 8.0))
        self.gap_decay_seconds = float(getattr(choral_cfg, 'oc_gap_decay_seconds', 2.0))
        self.overlap_tolerance_seconds = float(
            getattr(choral_cfg, 'oc_overlap_tolerance_seconds', 0.05)
        )

    def _normalize_voice_setting(self, values, defaults):
        values = defaults if values is None else list(values)
        if len(values) != self.num_voices:
            raise ValueError(
                f'voice assignment config expects {self.num_voices} values, got {len(values)}'
            )
        return [float(value) for value in values]

    def _canonical_voice_index(self, part_name) -> int | None:
        """Recognize trusted SATB labels without guessing from arbitrary text."""

        voice_code = canonical_voice_code(part_name)
        return self.voice_code_to_index.get(voice_code) if voice_code else None

    def _legacy_voice_index(self, part_name) -> int | None:
        """Reproduce the former first-character label heuristic."""

        if part_name is None:
            return None
        label = str(part_name).strip()
        if not label:
            return None
        return self.voice_code_to_index.get(label[0].upper())

    def note_bars_to_events(self, note_bars):
        events = []
        for bar in note_bars:
            if not isinstance(bar, dict):
                continue
            for part_name, note_list in bar.items():
                if str(part_name).strip().lower() == 'measure' or not isinstance(note_list, list):
                    continue
                for note in note_list:
                    if len(note) < 5:
                        continue
                    midi_note = int(note[0])
                    onset_time = float(note[3])
                    offset_time = float(note[4])
                    if offset_time <= onset_time:
                        offset_time = onset_time + 1e-4
                    if not self.begin_note <= midi_note < self.begin_note + self.classes_num:
                        continue
                    events.append(
                        {
                            'part_name': part_name,
                            'part_voice_idx': self._canonical_voice_index(part_name),
                            'legacy_part_voice_idx': self._legacy_voice_index(part_name),
                            'midi_note': midi_note,
                            'onset_time': onset_time,
                            'offset_time': offset_time,
                        }
                    )
        events.sort(key=lambda event: (
            event['onset_time'],
            -event['midi_note'],
            event['offset_time'],
        ))
        return events

    def _range_cost(self, midi_note: int, voice_idx: int) -> float:
        low = self.range_mins[voice_idx]
        high = self.range_maxs[voice_idx]
        center = 0.5 * (low + high)
        below = max(0.0, low - midi_note)
        above = max(0.0, midi_note - high)
        return below + above + 0.25 * abs(midi_note - center) / 12.0

    def _voice_in_masked_range(self, midi_note: int, voice_idx: int) -> bool:
        low = self.range_mins[voice_idx] - self.range_margin
        high = self.range_maxs[voice_idx] + self.range_margin
        return low <= midi_note <= high

    def _assignment_cost(
        self,
        event,
        voice_idx: int,
        last_pitch_by_voice=None,
        last_offset_by_voice=None,
        *,
        legacy: bool = False,
        masked: bool = False,
    ) -> float:
        cost = self._range_cost(event['midi_note'], voice_idx)
        part_key = 'legacy_part_voice_idx' if legacy else 'part_voice_idx'
        part_voice_idx = event.get(part_key)
        if part_voice_idx is not None and part_voice_idx != voice_idx:
            cost += self.part_penalty

        if last_pitch_by_voice is not None and voice_idx in last_pitch_by_voice:
            continuity_scale = 1.0
            if not legacy and last_offset_by_voice is not None and voice_idx in last_offset_by_voice:
                gap = max(0.0, event['onset_time'] - last_offset_by_voice[voice_idx])
                if self.gap_decay_seconds > 0:
                    continuity_scale = math.exp(-gap / self.gap_decay_seconds)
                elif gap > 0:
                    continuity_scale = 0.0
            cost += (
                self.continuity_weight
                * continuity_scale
                * abs(event['midi_note'] - last_pitch_by_voice[voice_idx])
                / 12.0
            )

        if last_offset_by_voice is not None and voice_idx in last_offset_by_voice:
            tolerance = 0.0 if legacy else self.overlap_tolerance_seconds
            if event['onset_time'] < last_offset_by_voice[voice_idx] - tolerance:
                cost += self.overlap_penalty

        if masked and not self._voice_in_masked_range(event['midi_note'], voice_idx):
            cost += self.mask_penalty
        return cost

    def _events_to_voice_tuples(self, assigned_events):
        return [
            (voice_idx, event['midi_note'], event['onset_time'], event['offset_time'])
            for voice_idx, event in assigned_events
        ]

    def _assign_part_name(self, note_events):
        return self._events_to_voice_tuples(
            (event['part_voice_idx'], event)
            for event in note_events
            if event['part_voice_idx'] is not None
        )

    def _assign_range_prior(self, note_events, *, preserve_known: bool, legacy: bool):
        assigned_events = []
        for event in note_events:
            if preserve_known and event['part_voice_idx'] is not None:
                voice_idx = event['part_voice_idx']
            else:
                voice_idx = min(
                    range(self.num_voices),
                    key=lambda candidate: self._assignment_cost(
                        event,
                        candidate,
                        legacy=legacy,
                    ),
                )
            assigned_events.append((voice_idx, event))
        return self._events_to_voice_tuples(assigned_events)

    def _group_by_onset_frame(self, note_events):
        grouped = {}
        for event in note_events:
            onset_frame = int(np.round(event['onset_time'] * self.frames_per_second))
            grouped.setdefault(onset_frame, []).append(event)
        return [grouped[key] for key in sorted(grouped)]

    def _assign_unknown_group(
        self,
        group,
        last_pitch_by_voice,
        last_offset_by_voice,
        *,
        legacy: bool,
        masked: bool,
    ):
        group = sorted(
            group,
            key=lambda event: (-event['midi_note'], event['onset_time'], event['offset_time']),
        )
        if len(group) > self.num_voices:
            return [
                (
                    min(
                        range(self.num_voices),
                        key=lambda voice_idx: self._assignment_cost(
                            event,
                            voice_idx,
                            last_pitch_by_voice,
                            last_offset_by_voice,
                            legacy=legacy,
                            masked=masked,
                        ),
                    ),
                    event,
                )
                for event in group
            ]

        best_assignment = None
        best_cost = float('inf')
        for voice_combo in combinations(range(self.num_voices), len(group)):
            candidate = list(zip(voice_combo, group))
            cost = sum(
                self._assignment_cost(
                    event,
                    voice_idx,
                    last_pitch_by_voice,
                    last_offset_by_voice,
                    legacy=legacy,
                    masked=masked,
                )
                for voice_idx, event in candidate
            )
            if cost < best_cost:
                best_cost = cost
                best_assignment = candidate
        return best_assignment or []

    @staticmethod
    def _update_modern_continuity_state(
        group_assignment,
        last_pitch_by_voice,
        last_offset_by_voice,
    ) -> None:
        events_by_voice = {}
        for voice_idx, event in group_assignment:
            events_by_voice.setdefault(voice_idx, []).append(event)
        for voice_idx, events in events_by_voice.items():
            # Keep the pitch centre of the event(s) that release last. A short
            # nested divisi note must not erase a longer still-active line.
            latest_release = max(event['offset_time'] for event in events)
            previous_release = last_offset_by_voice.get(voice_idx, float('-inf'))
            if latest_release >= previous_release:
                last_pitch_by_voice[voice_idx] = float(np.mean([
                    event['midi_note']
                    for event in events
                    if abs(event['offset_time'] - latest_release) <= 1e-12
                ]))
                last_offset_by_voice[voice_idx] = latest_release

    def _assign_ordered_continuity(
        self,
        note_events,
        *,
        preserve_known: bool,
        legacy: bool,
        masked: bool,
    ):
        last_pitch_by_voice = {}
        last_offset_by_voice = {}
        assigned_events = []

        for group in self._group_by_onset_frame(note_events):
            if preserve_known:
                known_assignment = [
                    (event['part_voice_idx'], event)
                    for event in group
                    if event['part_voice_idx'] is not None
                ]
                unknown_events = [
                    event for event in group if event['part_voice_idx'] is None
                ]
            else:
                known_assignment = []
                unknown_events = group

            inferred_assignment = self._assign_unknown_group(
                unknown_events,
                last_pitch_by_voice,
                last_offset_by_voice,
                legacy=legacy,
                masked=masked,
            )
            group_assignment = known_assignment + inferred_assignment
            assigned_events.extend(group_assignment)

            if legacy:
                # Preserve the exact sequential state update used by the old
                # all-note OC implementation.
                for voice_idx, event in group_assignment:
                    last_pitch_by_voice[voice_idx] = event['midi_note']
                    last_offset_by_voice[voice_idx] = event['offset_time']
            else:
                self._update_modern_continuity_state(
                    group_assignment,
                    last_pitch_by_voice,
                    last_offset_by_voice,
                )

        assigned_events.sort(key=lambda item: (
            item[1]['onset_time'],
            item[0],
            item[1]['midi_note'],
        ))
        return self._events_to_voice_tuples(assigned_events)

    def assign_voice_events(self, note_bars, *, quantization_origin=0.0):
        """Return ``(voice, pitch, onset, offset)`` tuples for all retained notes."""

        note_events = self.note_bars_to_events(note_bars)
        method = self.target_assignment
        if method == 'part_name':
            assigned_events = self._assign_part_name(note_events)
        elif method == 'range_prior':
            assigned_events = self._assign_range_prior(
                note_events,
                preserve_known=self.preserve_known_part_labels,
                legacy=False,
            )
        elif method == 'ordered_continuity':
            assigned_events = self._assign_ordered_continuity(
                note_events,
                preserve_known=self.preserve_known_part_labels,
                legacy=False,
                masked=False,
            )
        elif method == 'range_masked_continuity':
            assigned_events = self._assign_ordered_continuity(
                note_events,
                preserve_known=self.preserve_known_part_labels,
                legacy=False,
                masked=True,
            )
        elif method == 'legacy_range_prior':
            assigned_events = self._assign_range_prior(
                note_events,
                preserve_known=False,
                legacy=True,
            )
        elif method == 'legacy_ordered_continuity':
            assigned_events = self._assign_ordered_continuity(
                note_events,
                preserve_known=False,
                legacy=True,
                masked=False,
            )
        elif method == 'legacy_range_masked_continuity':
            assigned_events = self._assign_ordered_continuity(
                note_events,
                preserve_known=False,
                legacy=True,
                masked=True,
            )
        else:
            raise AssertionError(f'Unreachable target-assignment method: {method}')

        merged_events = merge_quantized_voice_events(
            assigned_events,
            self.frames_per_second,
            quantization_origin=quantization_origin,
        )
        # A binary voice head cannot retain two overlapping notes of the same
        # pitch. Preserve every representable attack and end the preceding
        # event at the next attack, matching the global canonical projector.
        projected_events = []
        mapping_style = bool(
            merged_events
            and len(merged_events[0]) == 2
            and isinstance(merged_events[0][1], dict)
        )
        for voice_idx in range(self.num_voices):
            voice_events = [
                event for event in merged_events if event[0] == voice_idx
            ]
            if mapping_style:
                event_values = [
                    (
                        int(source_event['midi_note']),
                        float(source_event['onset_time']),
                        float(source_event['offset_time']),
                    )
                    for _, source_event in voice_events
                ]
                source_by_attack = {
                    (midi_note, onset_time): source_event
                    for (_, source_event), (
                        midi_note,
                        onset_time,
                        _offset_time,
                    ) in zip(voice_events, event_values)
                }
            else:
                event_values = [
                    (midi_note, onset_time, offset_time)
                    for _, midi_note, onset_time, offset_time in voice_events
                ]
            synthetic_bar = {
                str(voice_idx): [
                    [midi_note, 0, 0, onset_time, offset_time]
                    for midi_note, onset_time, offset_time in event_values
                ]
            }
            projected_voice_events = canonical_union_events(
                [synthetic_bar],
                self.frames_per_second,
                begin_note=self.begin_note,
                classes_num=self.classes_num,
                quantization_origin=quantization_origin,
            )
            if mapping_style:
                for midi_note, onset_time, offset_time in projected_voice_events:
                    source_event = dict(source_by_attack[(midi_note, onset_time)])
                    source_event['onset_time'] = onset_time
                    source_event['offset_time'] = offset_time
                    projected_events.append((voice_idx, source_event))
            else:
                projected_events.extend(
                    (voice_idx, midi_note, onset_time, offset_time)
                    for midi_note, onset_time, offset_time in projected_voice_events
                )
        return sorted(
            projected_events,
            key=(
                (lambda event: (
                    event[1]['onset_time'],
                    event[0],
                    event[1]['midi_note'],
                    event[1]['offset_time'],
                ))
                if mapping_style
                else (lambda event: (event[2], event[0], event[1], event[3]))
            ),
        )

    def _repeat_mask(self, mask_roll, name: str):
        if mask_roll is None:
            mask_roll = np.ones((self.frames_num, self.classes_num), dtype=np.float32)
        mask_roll = np.asarray(mask_roll)
        expected_shape = (self.frames_num, self.classes_num)
        if mask_roll.shape != expected_shape:
            raise ValueError(f'{name} must have shape {expected_shape}, got {mask_roll.shape}')
        return np.repeat(mask_roll[:, None, :], self.num_voices, axis=1)

    def build(
        self,
        note_bars,
        *,
        start_time: float = 0.0,
        frame_mask_roll=None,
        onset_mask_roll=None,
        offset_mask_roll=None,
    ):
        """Build voice rolls, masks, and segment-level voice presence."""

        start_time = float(start_time)
        segment_end = start_time + self.segment_seconds
        shape = (self.frames_num, self.num_voices, self.classes_num)
        voice_frame_roll = np.zeros(shape, dtype=np.float32)
        voice_onset_roll = np.zeros(shape, dtype=np.float32)
        voice_offset_roll = np.zeros(shape, dtype=np.float32)
        voice_presence = np.zeros((self.num_voices,), dtype=np.float32)

        for voice_idx, midi_note, onset_time, offset_time in self.assign_voice_events(
            note_bars,
            quantization_origin=start_time,
        ):
            if offset_time <= start_time or onset_time >= segment_end:
                continue

            note_idx = midi_note - self.begin_note
            local_onset = max(onset_time, start_time)
            local_offset = min(offset_time, segment_end)
            onset_frame = int(np.clip(
                np.round((local_onset - start_time) * self.frames_per_second),
                0,
                self.frames_num - 1,
            ))
            offset_frame = int(np.clip(
                np.round((local_offset - start_time) * self.frames_per_second),
                0,
                self.frames_num - 1,
            ))
            offset_frame = max(offset_frame, onset_frame)

            voice_frame_roll[onset_frame:offset_frame + 1, voice_idx, note_idx] = 1.0
            if start_time <= onset_time < segment_end:
                true_onset_frame = int(np.clip(
                    np.round((onset_time - start_time) * self.frames_per_second),
                    0,
                    self.frames_num - 1,
                ))
                voice_onset_roll[true_onset_frame, voice_idx, note_idx] = 1.0
            if start_time < offset_time <= segment_end:
                true_offset_frame = int(np.clip(
                    np.round((offset_time - start_time) * self.frames_per_second),
                    0,
                    self.frames_num - 1,
                ))
                voice_offset_roll[true_offset_frame, voice_idx, note_idx] = 1.0
            voice_presence[voice_idx] = 1.0

        return {
            'voice_frame_roll': voice_frame_roll,
            'voice_onset_roll': voice_onset_roll,
            'voice_offset_roll': voice_offset_roll,
            'voice_presence': voice_presence,
            'voice_frame_mask_roll': self._repeat_mask(frame_mask_roll, 'frame_mask_roll'),
            'voice_onset_mask_roll': self._repeat_mask(onset_mask_roll, 'onset_mask_roll'),
            'voice_offset_mask_roll': self._repeat_mask(offset_mask_roll, 'offset_mask_roll'),
        }


def build_choral_target_dict(
    cfg,
    note_bars,
    segment_seconds: float,
    start_time: float = 0.0,
    frame_mask_roll=None,
    onset_mask_roll=None,
    offset_mask_roll=None,
    target_assignment: str | None = None,
):
    """Functional wrapper around :class:`ChoralTargetBuilder`."""

    return ChoralTargetBuilder(
        cfg,
        segment_seconds=segment_seconds,
        target_assignment=target_assignment,
    ).build(
        note_bars,
        start_time=start_time,
        frame_mask_roll=frame_mask_roll,
        onset_mask_roll=onset_mask_roll,
        offset_mask_roll=offset_mask_roll,
    )
