# SPDX-License-Identifier: Apache-2.0

"""Canonical part-agnostic note-union targets.

The SATB annotations in ``note.pkl`` may contain divisi parts, unknown part
labels, and simultaneous unisons.  A single union transcription head cannot
represent the part identity of those notes, but it *can* retain every distinct
same-pitch attack.  This module defines that projection independently of any
voice-assignment strategy (part-name, RP, or OC).

All public functions are pure: their output depends only on their arguments
and they never mutate ``note_bars``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np


UnionEvent = tuple[int, float, float]
CANONICAL_UNION_SEMANTICS_VERSION = 1
CANONICAL_UNION_DATASETS = frozenset({
    'youchorale',
    'youchorale_pro',
    'csd',
    'cantoria',
})


def canonical_union_semantics(dataset_name: str) -> dict | None:
    """Return the versioned contract used by a supported choral dataset."""

    if str(dataset_name).strip().lower() not in CANONICAL_UNION_DATASETS:
        return None
    return {
        'version': CANONICAL_UNION_SEMANTICS_VERSION,
        'source': 'note_pkl',
        'attack_quantization': 'round((onset-origin)*fps)',
        'same_pitch_overlap': 'preserve_attacks_split_at_next_attack',
        'segment_masks': 'complete_interval_all_observed',
    }


def _validate_projection_parameters(
    frames_per_second: float,
    begin_note: int,
    classes_num: int,
) -> tuple[float, int, int]:
    frames_per_second = float(frames_per_second)
    if not math.isfinite(frames_per_second) or frames_per_second <= 0.0:
        raise ValueError(
            "frames_per_second must be finite and positive, "
            f"got {frames_per_second!r}"
        )

    begin_note = int(begin_note)
    classes_num = int(classes_num)
    if classes_num <= 0:
        raise ValueError(f"classes_num must be positive, got {classes_num!r}")
    return frames_per_second, begin_note, classes_num


def _extract_representable_events(
    note_bars: Iterable,
    *,
    begin_note: int,
    classes_num: int,
) -> list[UnionEvent]:
    """Extract valid in-range notes while deliberately ignoring part labels."""

    upper_note = begin_note + classes_num
    events: list[UnionEvent] = []
    for bar in note_bars:
        if not isinstance(bar, Mapping):
            continue
        for part_name, notes in bar.items():
            if str(part_name).strip().lower() == "measure":
                continue
            if not isinstance(notes, (list, tuple)):
                continue
            for note in notes:
                if not isinstance(note, (list, tuple, np.ndarray)) or len(note) < 5:
                    continue
                try:
                    pitch_value = float(note[0])
                    onset = float(note[3])
                    offset = float(note[4])
                except (TypeError, ValueError, OverflowError):
                    continue
                if (
                    not math.isfinite(pitch_value)
                    or not pitch_value.is_integer()
                    or not math.isfinite(onset)
                    or not math.isfinite(offset)
                    or offset <= onset
                ):
                    continue
                pitch = int(pitch_value)
                if begin_note <= pitch < upper_note:
                    events.append((pitch, onset, offset))
    return events


def canonical_union_events(
    note_bars: Iterable,
    frames_per_second: float,
    *,
    begin_note: int = 0,
    classes_num: int = 128,
    quantization_origin: float = 0.0,
) -> list[UnionEvent]:
    """Project annotated notes to one canonical, part-agnostic note stream.

    The projection has two deterministic stages for each pitch:

    1. Attacks whose
       ``round((onset - quantization_origin) * frames_per_second)`` values
       agree are collapsed to the earliest source onset and latest source
       offset. Full-recording references use the default origin zero; segment
       targets use their local start so the projection matches the model grid.
    2. Within each strictly overlapping interval component, every remaining
       attack is retained.  An active segment ends at the next attack and the
       final segment ends at the union end of the component.

    Therefore nested cross-voice unisons such as ``[0, 2]`` and ``[.1, 1]``
    become ``[0, .1]`` and ``[.1, 2]``.  Touching half-open intervals are not
    coalesced, so adjacent rearticulations remain separate.  Part names are
    intentionally ignored: valid notes under unknown labels are included.

    Returns:
        ``(midi_pitch, onset_seconds, offset_seconds)`` tuples.  Intervals are
        half-open and ordered by onset, then pitch, then offset.
    """

    frames_per_second, begin_note, classes_num = _validate_projection_parameters(
        frames_per_second,
        begin_note,
        classes_num,
    )
    quantization_origin = float(quantization_origin)
    if not math.isfinite(quantization_origin):
        raise ValueError(
            'quantization_origin must be finite, '
            f'got {quantization_origin!r}'
        )
    raw_events = _extract_representable_events(
        note_bars,
        begin_note=begin_note,
        classes_num=classes_num,
    )

    # First collapse attacks that the binary onset head cannot distinguish.
    quantized_attacks: dict[tuple[int, int], list[float]] = {}
    for pitch, onset, offset in raw_events:
        onset_frame = int(
            np.round((onset - quantization_origin) * frames_per_second)
        )
        key = (pitch, onset_frame)
        if key not in quantized_attacks:
            quantized_attacks[key] = [onset, offset]
        else:
            attack = quantized_attacks[key]
            attack[0] = min(attack[0], onset)
            attack[1] = max(attack[1], offset)

    attacks_by_pitch: dict[int, list[tuple[float, float]]] = {}
    for (pitch, _), (onset, offset) in quantized_attacks.items():
        attacks_by_pitch.setdefault(pitch, []).append((onset, offset))

    projected: list[UnionEvent] = []
    for pitch, attacks in attacks_by_pitch.items():
        attacks.sort(key=lambda interval: (interval[0], interval[1]))

        # Build strict-overlap components.  Equality starts a new component
        # because the returned intervals are half-open and because an attack
        # at the previous release is a genuine rearticulation.
        component: list[tuple[float, float]] = []
        component_end = float("-inf")

        def emit_component() -> None:
            if not component:
                return
            for index, (attack_onset, _) in enumerate(component):
                attack_offset = (
                    component[index + 1][0]
                    if index + 1 < len(component)
                    else component_end
                )
                if attack_offset > attack_onset:
                    projected.append((pitch, attack_onset, attack_offset))

        for attack in attacks:
            onset, offset = attack
            if component and onset >= component_end:
                emit_component()
                component = []
                component_end = float("-inf")
            component.append(attack)
            component_end = max(component_end, offset)
        emit_component()

    projected.sort(key=lambda event: (event[1], event[0], event[2]))
    return projected


def build_canonical_union_rolls(
    note_bars: Iterable,
    *,
    start_time: float,
    segment_seconds: float,
    frames_per_second: float,
    begin_note: int = 21,
    classes_num: int = 88,
) -> dict[str, np.ndarray]:
    """Build binary canonical-union rolls for one audio segment.

    Frame targets include notes that began before ``start_time`` and remain
    active in this segment; their onset is not spuriously repeated.  Likewise,
    an offset is marked only when the true release occurs inside the segment.
    Every returned mask is one because the complete note list, rather than a
    segment-local MIDI event stream, defines these targets.

    The roll length follows the repository convention
    ``round(segment_seconds * fps) + 1``.  Frame occupancy uses the matching
    endpoint-inclusive discrete convention, while :func:`canonical_union_events`
    remains the authoritative half-open continuous representation.
    """

    frames_per_second, begin_note, classes_num = _validate_projection_parameters(
        frames_per_second,
        begin_note,
        classes_num,
    )
    start_time = float(start_time)
    segment_seconds = float(segment_seconds)
    if not math.isfinite(start_time):
        raise ValueError(f"start_time must be finite, got {start_time!r}")
    if not math.isfinite(segment_seconds) or segment_seconds <= 0.0:
        raise ValueError(
            "segment_seconds must be finite and positive, "
            f"got {segment_seconds!r}"
        )

    frames_num = int(round(segment_seconds * frames_per_second)) + 1
    segment_end = start_time + segment_seconds
    shape = (frames_num, classes_num)
    frame_roll = np.zeros(shape, dtype=np.float32)
    onset_roll = np.zeros(shape, dtype=np.float32)
    offset_roll = np.zeros(shape, dtype=np.float32)

    for pitch, onset, offset in canonical_union_events(
        note_bars,
        frames_per_second,
        begin_note=begin_note,
        classes_num=classes_num,
        quantization_origin=start_time,
    ):
        # Both continuous representations are half-open.
        if offset <= start_time or onset >= segment_end:
            continue

        note_index = pitch - begin_note
        local_onset = max(onset, start_time)
        local_offset = min(offset, segment_end)
        onset_frame = int(np.clip(
            np.round((local_onset - start_time) * frames_per_second),
            0,
            frames_num - 1,
        ))
        offset_frame = int(np.clip(
            np.round((local_offset - start_time) * frames_per_second),
            0,
            frames_num - 1,
        ))
        offset_frame = max(offset_frame, onset_frame)
        frame_roll[onset_frame : offset_frame + 1, note_index] = 1.0

        if start_time <= onset < segment_end:
            true_onset_frame = int(np.clip(
                np.round((onset - start_time) * frames_per_second),
                0,
                frames_num - 1,
            ))
            onset_roll[true_onset_frame, note_index] = 1.0
        # The repository stores N * fps + 1 frames, so the true release at the
        # recording/segment endpoint is representable and must not be treated
        # as an all-one-mask negative. An offset exactly at the segment start
        # belongs to the preceding window instead.
        if start_time < offset <= segment_end:
            true_offset_frame = int(np.clip(
                np.round((offset - start_time) * frames_per_second),
                0,
                frames_num - 1,
            ))
            offset_roll[true_offset_frame, note_index] = 1.0

    return {
        "frame_roll": frame_roll,
        "onset_roll": onset_roll,
        "offset_roll": offset_roll,
        "frame_mask_roll": np.ones(shape, dtype=np.float32),
        "onset_mask_roll": np.ones(shape, dtype=np.float32),
        "offset_mask_roll": np.ones(shape, dtype=np.float32),
    }


__all__ = [
    "CANONICAL_UNION_DATASETS",
    "CANONICAL_UNION_SEMANTICS_VERSION",
    "UnionEvent",
    "build_canonical_union_rolls",
    "canonical_union_events",
    "canonical_union_semantics",
]
