from __future__ import annotations

import os
import pickle
import sys
from copy import deepcopy

import h5py
import mir_eval
import numpy as np
from hydra import compose, initialize
from sklearn import metrics as sk_metrics

from canonical_union import canonical_union_events
from calculate_scores import build_post_processor
from choral_targets import (
    canonical_voice_code,
    merge_quantized_voice_events,
    prepare_formal_satb_reference,
    resolve_reference_duration_policy,
)
from probability_artifacts import (
    ProbabilityArtifactValidator,
    expected_split_probability_stems,
    validate_probability_manifest,
)
from utilities import get_filename, get_model_name, note_to_freq


VOICE_NAMES = ["S", "A", "T", "B"]
VOICE_TO_INDEX = {name: idx for idx, name in enumerate(VOICE_NAMES)}
FIXED_ONSET_TOLERANCES = (0.05, 0.10)
SAME_OUTPUT_REPORT_KEYS = (
    "union_note_precision",
    "union_note_recall",
    "union_note_f1",
    "union_average_overlap_ratio",
    "union_note_with_offset_f1",
    "union_offset_average_overlap_ratio",
    "union_matched_note_count",
    "matched_note_voice_correct_count",
    "matched_note_voice_eligible_count",
    "matched_note_voice_ambiguous_reference_count",
    "matched_note_voice_accuracy",
    "duplicate_estimate_excess_count",
    "estimated_voice_note_count",
    "duplicate_estimate_rate",
    "voice_switch_count",
    "voice_transition_count",
    "voice_switch_rate",
    *(
        f"voice_confusion_{source}_{target}"
        for source in VOICE_NAMES
        for target in VOICE_NAMES
    ),
)


def _new_confusion():
    return {
        source: {target: 0 for target in VOICE_NAMES}
        for source in VOICE_NAMES
    }


def _tolerance_tag(onset_tolerance):
    return f"{int(round(onset_tolerance * 1000.0))}ms"


def _dataset_note_dir(cfg):
    if cfg.dataset.test_set == "youchorale":
        return os.path.join(cfg.dataset.youchorale_dir, "note")
    if cfg.dataset.test_set == "youchorale_pro":
        return os.path.join(cfg.dataset.youchorale_pro_dir, "note")
    if cfg.dataset.test_set == "csd":
        return os.path.join(cfg.dataset.csd_dir, "note")
    if cfg.dataset.test_set == "cantoria":
        return os.path.join(cfg.dataset.cantoria_dir, f"note_{cfg.dataset.cantoria_f0_source}")
    raise ValueError(f"Unsupported choral dataset: {cfg.dataset.test_set}")


def _load_note_bars(note_path):
    with open(note_path, "rb") as f:
        return pickle.load(f)


def _packed_recording_duration(hdf5_path, sample_rate):
    """Return the duration represented by the packed evaluation recording."""

    sample_rate = float(sample_rate)
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError(f"sample_rate must be finite and positive, got {sample_rate!r}")
    with h5py.File(hdf5_path, "r") as hdf5_file:
        if "waveform" in hdf5_file:
            duration = hdf5_file["waveform"].shape[0] / sample_rate
        elif "duration" in hdf5_file.attrs:
            duration = float(hdf5_file.attrs["duration"])
        else:
            raise RuntimeError(
                "Packed evaluation recording has neither waveform samples nor a "
                f"duration attribute: {hdf5_path}"
            )
    if not np.isfinite(duration) or duration < 0.0:
        raise RuntimeError(
            f"Packed evaluation recording has invalid duration={duration!r}: {hdf5_path}"
        )
    return float(duration)


def _safe_intervals(intervals):
    intervals = np.asarray(intervals, dtype=np.float32)
    if intervals.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    bad = np.nonzero(intervals[:, 1] <= intervals[:, 0])[0]
    intervals = intervals.copy()
    intervals[bad, 1] = intervals[bad, 0] + 1e-4
    return intervals


def _reference_events_by_voice(note_bars, frames_per_second):
    voice_map = {name: [] for name in VOICE_NAMES}
    for bar in note_bars:
        if not isinstance(bar, dict):
            continue
        for part_name, note_list in bar.items():
            if part_name == "measure" or not part_name:
                continue
            voice_name = canonical_voice_code(part_name)
            if voice_name not in voice_map:
                continue
            for note in note_list:
                if len(note) < 5:
                    continue
                onset_time = float(note[3])
                offset_time = float(note[4])
                if offset_time <= onset_time:
                    offset_time = onset_time + 1e-4
                voice_map[voice_name].append(
                    {
                        "midi_note": int(note[0]),
                        "onset_time": onset_time,
                        "offset_time": offset_time,
                    }
                )
    merged_events = merge_quantized_voice_events(
        (
            (
                voice_name,
                event["midi_note"],
                event["onset_time"],
                event["offset_time"],
            )
            for voice_name, events in voice_map.items()
            for event in events
        ),
        frames_per_second,
    )
    merged_voice_map = {name: [] for name in VOICE_NAMES}
    for voice_name, midi_note, onset_time, offset_time in merged_events:
        merged_voice_map[voice_name].append({
            "midi_note": midi_note,
            "onset_time": onset_time,
            "offset_time": offset_time,
        })
    projected_voice_map = {name: [] for name in VOICE_NAMES}
    for voice_name, events in merged_voice_map.items():
        synthetic_bars = [{
            voice_name: [
                [
                    event["midi_note"],
                    0,
                    0,
                    event["onset_time"],
                    event["offset_time"],
                ]
                for event in events
            ]
        }]
        projected_voice_map[voice_name] = [
            {
                "midi_note": midi_note,
                "onset_time": onset_time,
                "offset_time": offset_time,
            }
            for midi_note, onset_time, offset_time in canonical_union_events(
                synthetic_bars,
                frames_per_second,
                begin_note=0,
                classes_num=128,
            )
        ]
    return projected_voice_map


def _reference_presence_vector(note_bars, frames_per_second):
    ref_voice_events = _reference_events_by_voice(note_bars, frames_per_second)
    return np.asarray(
        [1.0 if len(ref_voice_events[voice_name]) > 0 else 0.0 for voice_name in VOICE_NAMES],
        dtype=np.float32,
    )


def _reference_frame_roll_by_voice(note_bars, frames_num, frames_per_second, begin_note, classes_num):
    """Build the immutable SATB frame reference from the source part labels.

    Probability files can contain training-target rolls for diagnostics.  Those
    rolls are not valid evaluation ground truth when RP/OC changed the training
    targets, so scoring always reconstructs the reference from ``note/*.pkl``.
    """
    frame_roll = np.zeros((frames_num, len(VOICE_NAMES), classes_num), dtype=np.float32)
    if frames_num <= 0:
        return frame_roll
    final_frame_time = (frames_num - 1) / float(frames_per_second)
    for voice_name, events in _reference_events_by_voice(
        note_bars,
        frames_per_second,
    ).items():
        voice_idx = VOICE_TO_INDEX[voice_name]
        for event in events:
            note_idx = int(event["midi_note"]) - int(begin_note)
            if not 0 <= note_idx < classes_num:
                continue
            if event["offset_time"] < 0.0 or event["onset_time"] > final_frame_time:
                continue
            onset_frame = int(np.clip(np.round(event["onset_time"] * frames_per_second), 0, frames_num - 1))
            offset_frame = int(np.clip(np.round(event["offset_time"] * frames_per_second), 0, frames_num - 1))
            if offset_frame < onset_frame:
                offset_frame = onset_frame
            frame_roll[onset_frame : offset_frame + 1, voice_idx, note_idx] = 1.0
    return frame_roll


def _events_to_arrays(events):
    if len(events) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    intervals = np.array([[e["onset_time"], e["offset_time"]] for e in events], dtype=np.float32)
    pitches = np.array([e["midi_note"] for e in events], dtype=np.int32)
    intervals = _safe_intervals(intervals)
    sort_idx = np.argsort(intervals[:, 0], kind='mergesort')
    return intervals[sort_idx], pitches[sort_idx]


def _flatten_voice_events(events_by_voice):
    flattened = []
    for voice_name in VOICE_NAMES:
        for event in events_by_voice.get(voice_name, []):
            flattened.append({**event, "voice_name": voice_name})
    return sorted(
        flattened,
        key=lambda event: (
            event["onset_time"],
            event["midi_note"],
            event["offset_time"],
            VOICE_TO_INDEX[event["voice_name"]],
        ),
    )


def _project_voice_events(events_by_voice, frames_per_second):
    """Project voice notes to the same frame-defined union used for training.

    Evaluation tolerances deliberately do not participate in this projection:
    changing a 50 ms report to 100 ms must not rewrite its ground truth.
    ``events`` on each projected attack retain the emitting voice set for the
    assignment diagnostics below.
    """

    frames_per_second = float(frames_per_second)
    flattened = _flatten_voice_events(events_by_voice)
    note_bars = [{
        voice_name: [
            [
                event["midi_note"],
                0,
                0,
                event["onset_time"],
                event["offset_time"],
            ]
            for event in events_by_voice.get(voice_name, [])
        ]
        for voice_name in VOICE_NAMES
    }]
    projected = canonical_union_events(
        note_bars,
        frames_per_second,
        begin_note=0,
        classes_num=128,
    )

    attack_sources = {}
    for event in flattened:
        try:
            midi_note = int(event["midi_note"])
            onset_time = float(event["onset_time"])
            offset_time = float(event["offset_time"])
        except (TypeError, ValueError, OverflowError, KeyError):
            continue
        if not (
            0 <= midi_note < 128
            and np.isfinite(onset_time)
            and np.isfinite(offset_time)
            and offset_time > onset_time
        ):
            continue
        key = (midi_note, int(np.round(onset_time * frames_per_second)))
        attack_sources.setdefault(key, []).append(event)

    clusters = []
    for midi_note, onset_time, offset_time in projected:
        key = (midi_note, int(np.round(onset_time * frames_per_second)))
        clusters.append({
            "midi_note": midi_note,
            "onset_time": onset_time,
            "offset_time": offset_time,
            "events": attack_sources.get(key, []),
        })
    return clusters


def _collapse_voice_events(events_by_voice, frames_per_second):
    """Return canonical union notes and duplicate-emission counts."""

    clusters = _project_voice_events(events_by_voice, frames_per_second)
    collapsed = [
        {
            "midi_note": cluster["midi_note"],
            "onset_time": cluster["onset_time"],
            "offset_time": cluster["offset_time"],
        }
        for cluster in clusters
    ]
    total_events = sum(len(cluster["events"]) for cluster in clusters)
    duplicate_excess = total_events - len(clusters)
    return collapsed, duplicate_excess, total_events


def _same_output_assignment_metrics(
    reference_by_voice,
    estimated_by_voice,
    *,
    frames_per_second,
    onset_tolerance,
    offset_ratio,
    offset_min_tolerance,
):
    """Measure union transcription and assignment from the same PawCT output.

    ``matched_note_voice_accuracy`` is the fraction of matched
    union notes with an unambiguous reference voice that are emitted only by
    the correct SATB head.  Its numerator and denominator are explicit counts,
    so the value is bounded to [0, 1].
    """
    reference_clusters = _project_voice_events(
        reference_by_voice, frames_per_second
    )
    estimated_clusters = _project_voice_events(
        estimated_by_voice, frames_per_second
    )
    canonical_reference_union, _, _ = _collapse_voice_events(
        reference_by_voice, frames_per_second
    )
    estimated_union, duplicate_excess, estimated_count = _collapse_voice_events(
        estimated_by_voice, frames_per_second
    )
    ref_intervals, ref_pitches = _events_to_arrays(canonical_reference_union)
    est_intervals, est_pitches = _events_to_arrays(estimated_union)
    union_precision, union_recall, union_f1, union_overlap = (
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=note_to_freq(ref_pitches),
            est_intervals=est_intervals,
            est_pitches=note_to_freq(est_pitches),
            onset_tolerance=onset_tolerance,
            offset_ratio=None,
            offset_min_tolerance=offset_min_tolerance,
        )
    )
    _, _, union_offset_f1, union_offset_overlap = (
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=note_to_freq(ref_pitches),
            est_intervals=est_intervals,
            est_pitches=note_to_freq(est_pitches),
            onset_tolerance=onset_tolerance,
            offset_ratio=offset_ratio,
            offset_min_tolerance=offset_min_tolerance,
        )
    )

    union_matches = mir_eval.transcription.match_notes(
        ref_intervals=ref_intervals,
        ref_pitches=note_to_freq(ref_pitches),
        est_intervals=est_intervals,
        est_pitches=note_to_freq(est_pitches),
        onset_tolerance=onset_tolerance,
        pitch_tolerance=50.0,
        offset_ratio=None,
        offset_min_tolerance=offset_min_tolerance,
    )

    canonical_ref_intervals, canonical_ref_pitches = _events_to_arrays(
        canonical_reference_union
    )
    assignment_matches = mir_eval.transcription.match_notes(
        ref_intervals=canonical_ref_intervals,
        ref_pitches=note_to_freq(canonical_ref_pitches),
        est_intervals=est_intervals,
        est_pitches=note_to_freq(est_pitches),
        onset_tolerance=onset_tolerance,
        pitch_tolerance=50.0,
        offset_ratio=None,
        offset_min_tolerance=offset_min_tolerance,
    )

    confusion = _new_confusion()
    correct = 0
    eligible = 0
    ambiguous = 0
    predicted_by_reference = {voice_name: [] for voice_name in VOICE_NAMES}
    for ref_idx, est_idx in assignment_matches:
        reference_cluster = reference_clusters[ref_idx]
        estimated_cluster = estimated_clusters[est_idx]
        reference_voices = {
            event["voice_name"] for event in reference_cluster["events"]
        }
        if len(reference_voices) != 1:
            ambiguous += 1
            continue

        reference_voice = next(iter(reference_voices))
        predicted_voices = {
            event["voice_name"] for event in estimated_cluster["events"]
        }
        eligible += 1
        if predicted_voices == {reference_voice}:
            correct += 1
        if len(predicted_voices) != 1:
            continue

        predicted_voice = next(iter(predicted_voices))
        confusion[reference_voice][predicted_voice] += 1
        predicted_by_reference[reference_voice].append(
            (float(reference_cluster["onset_time"]), predicted_voice)
        )

    switch_count = 0
    transition_count = 0
    for items in predicted_by_reference.values():
        items.sort()
        onset_groups = []
        for onset_time, predicted_voice in items:
            if (
                not onset_groups
                or onset_time - onset_groups[-1]["start"]
                > onset_tolerance + 1e-12
            ):
                onset_groups.append(
                    {"start": onset_time, "predicted_voices": [predicted_voice]}
                )
            else:
                onset_groups[-1]["predicted_voices"].append(predicted_voice)
        for previous_group, next_group in zip(onset_groups, onset_groups[1:]):
            previous_voices = set(previous_group["predicted_voices"])
            next_voices = set(next_group["predicted_voices"])
            if len(previous_voices) != 1 or len(next_voices) != 1:
                continue
            previous_voice = next(iter(previous_voices))
            next_voice = next(iter(next_voices))
            transition_count += 1
            switch_count += int(previous_voice != next_voice)

    matched_voice_accuracy = correct / eligible if eligible else 0.0

    return {
        "union_note_precision": float(union_precision),
        "union_note_recall": float(union_recall),
        "union_note_f1": float(union_f1),
        "union_average_overlap_ratio": float(union_overlap),
        "union_note_with_offset_f1": float(union_offset_f1),
        "union_offset_average_overlap_ratio": float(union_offset_overlap),
        "union_matched_note_count": int(len(union_matches)),
        "duplicate_estimate_excess_count": int(duplicate_excess),
        "estimated_voice_note_count": int(estimated_count),
        "duplicate_estimate_rate": float(duplicate_excess / max(estimated_count, 1)),
        "matched_note_voice_eligible_count": int(eligible),
        "matched_note_voice_correct_count": int(correct),
        "matched_note_voice_ambiguous_reference_count": int(ambiguous),
        "matched_note_voice_accuracy": float(matched_voice_accuracy),
        "voice_switch_count": int(switch_count),
        "voice_transition_count": int(transition_count),
        "voice_switch_rate": float(switch_count / max(transition_count, 1)),
        **{
            f"voice_confusion_{source}_{target}": int(confusion[source][target])
            for source in VOICE_NAMES
            for target in VOICE_NAMES
        },
    }


def _correct_onset_metrics(ref_intervals, est_intervals, onset_tolerance):
    ref_onsets = ref_intervals[:, 0] if len(ref_intervals) else np.zeros((0,), dtype=np.float32)
    est_onsets = est_intervals[:, 0] if len(est_intervals) else np.zeros((0,), dtype=np.float32)
    ref_onsets = np.sort(ref_onsets)
    est_onsets = np.sort(est_onsets)
    con_f1, con_precision, con_recall = mir_eval.onset.f_measure(
        reference_onsets=ref_onsets,
        estimated_onsets=est_onsets,
        window=onset_tolerance,
    )
    return con_precision, con_recall, con_f1


def _frame_metrics(y_true, y_pred, mask=None, threshold=0.5):
    if y_true.size == 0 or y_pred.size == 0:
        return {}
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    if mask is not None:
        valid = mask.flatten() > 0
        min_len = min(len(y_true), len(y_pred), len(valid))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        valid = valid[:min_len]
        if not np.any(valid):
            return {}
        y_true = y_true[valid]
        y_pred = y_pred[valid]
    y_hat = (y_pred >= threshold).astype(np.float32)
    precision, recall, f1, _ = sk_metrics.precision_recall_fscore_support(
        y_true,
        y_hat,
        average="binary",
        zero_division=0,
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _decode_voice_events(total_dict, voice_idx, post_processor, thresholds=None):
    if thresholds is not None:
        post_processor = deepcopy(post_processor)
        post_processor.frame_threshold = thresholds["frame_threshold"]
        post_processor.onset_threshold = thresholds["onset_threshold"]
        post_processor.offset_threshold = thresholds["offset_threshold"]
    post_input = {}
    if "voice_frame_output" in total_dict:
        post_input["frame_output"] = total_dict["voice_frame_output"][:, voice_idx, :]
    if "voice_onset_output" in total_dict:
        post_input["onset_output"] = total_dict["voice_onset_output"][:, voice_idx, :]
    if "voice_offset_output" in total_dict:
        post_input["offset_output"] = total_dict["voice_offset_output"][:, voice_idx, :]
    note_events, _ = post_processor.output_dict_to_midi_events(deepcopy(post_input))
    return note_events


def _normalize_voice_thresholds(values, default_value):
    if values is None:
        return [float(default_value)] * len(VOICE_NAMES)
    values = list(values)
    if len(values) != len(VOICE_NAMES):
        raise ValueError(f"Expected {len(VOICE_NAMES)} thresholds, got {len(values)}")
    return [float(v) for v in values]


def _cfg_voice_thresholds(cfg):
    if getattr(cfg.choral, "use_per_voice_thresholds", False):
        frame_thresholds = _normalize_voice_thresholds(
            getattr(cfg.choral, "voice_frame_thresholds", None),
            cfg.post.frame_threshold,
        )
        onset_thresholds = _normalize_voice_thresholds(
            getattr(cfg.choral, "voice_onset_thresholds", None),
            cfg.post.onset_threshold,
        )
        offset_thresholds = _normalize_voice_thresholds(
            getattr(cfg.choral, "voice_offset_thresholds", None),
            cfg.post.offset_threshold,
        )
    else:
        frame_thresholds = [float(cfg.post.frame_threshold)] * len(VOICE_NAMES)
        onset_thresholds = [float(cfg.post.onset_threshold)] * len(VOICE_NAMES)
        offset_thresholds = [float(cfg.post.offset_threshold)] * len(VOICE_NAMES)

    return {
        voice_name: {
            "frame_threshold": frame_thresholds[idx],
            "onset_threshold": onset_thresholds[idx],
            "offset_threshold": offset_thresholds[idx],
        }
        for idx, voice_name in enumerate(VOICE_NAMES)
    }


class ChoralScoreCalculator:
    def __init__(self, cfg, voice_thresholds=None):
        self.cfg = cfg
        self.model_name = get_model_name(cfg)
        self.eval_split = str(getattr(cfg.dataset, "eval_split", "validation"))
        if self.eval_split not in {"validation", "test"}:
            raise ValueError("dataset.eval_split must be 'validation' or 'test'")
        self.probs_dir = os.path.join(
            cfg.exp.workspace,
            "probs",
            cfg.dataset.test_set,
            self.eval_split,
            self.model_name,
            f"{cfg.exp.ckpt_iteration}_iteration",
        )
        self.note_dir = _dataset_note_dir(cfg)
        self.post_processor = build_post_processor(cfg)
        self.voice_thresholds = voice_thresholds or _cfg_voice_thresholds(cfg)
        reference_assignment = str(
            getattr(cfg.choral, "evaluation_reference_assignment", "part_name")
        ).strip().lower()
        if reference_assignment != "part_name":
            raise ValueError(
                "Paper metrics require choral.evaluation_reference_assignment=part_name; "
                "RP/OC pseudo-targets may only be inspected as diagnostics."
            )

        self.artifact_validator = ProbabilityArtifactValidator(
            cfg,
            probs_dir=self.probs_dir,
            eval_split=self.eval_split,
        )
        self.probability_names = self.artifact_validator.probability_names

    def _validate_probability_provenance(self, total_dict, prob_path, note_path=None):
        self.artifact_validator.validate(
            total_dict,
            prob_path,
            reference_path=note_path,
        )

    def _load_probability_file(self, prob_path, note_path=None):
        with open(prob_path, "rb") as probability_file:
            total_dict = pickle.load(probability_file)
        self._validate_probability_provenance(total_dict, prob_path, note_path)
        return total_dict

    def _prepare_formal_reference(self, note_bars, note_path, prob_path):
        hdf5_path = self.artifact_validator.hdf5_path_for_probability(prob_path)
        recording_duration = _packed_recording_duration(
            hdf5_path,
            self.cfg.feature.sample_rate,
        )
        return prepare_formal_satb_reference(
            note_bars,
            note_path,
            begin_note=int(self.cfg.feature.begin_note),
            classes_num=int(self.cfg.feature.classes_num),
            recording_duration=recording_duration,
            duration_policy=resolve_reference_duration_policy(self.cfg),
        )

    def calculate_voice_score_per_song(
        self,
        prob_path,
        note_path,
        voice_name,
        thresholds=None,
        total_dict=None,
        note_bars=None,
        reference_validated=False,
    ):
        voice_idx = VOICE_TO_INDEX[voice_name]
        if total_dict is None:
            total_dict = self._load_probability_file(prob_path, note_path)
        else:
            self._validate_probability_provenance(total_dict, prob_path, note_path)

        if note_bars is None:
            note_bars = _load_note_bars(note_path)
        if not reference_validated:
            note_bars, _duration_adjustment = self._prepare_formal_reference(
                note_bars,
                note_path,
                prob_path,
            )
        ref_voice_events = _reference_events_by_voice(
            note_bars,
            float(self.cfg.feature.frames_per_second),
        )
        est_events = _decode_voice_events(
            total_dict,
            voice_idx,
            self.post_processor,
            thresholds=thresholds or self.voice_thresholds[voice_name],
        )
        ref_intervals, ref_pitches = _events_to_arrays(ref_voice_events[voice_name])
        est_intervals, est_pitches = _events_to_arrays(est_events)

        con_precision, con_recall, con_f1 = _correct_onset_metrics(
            ref_intervals,
            est_intervals,
            onset_tolerance=self.cfg.score.onset_tolerance,
        )
        conp_precision, conp_recall, conp_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=note_to_freq(ref_pitches),
            est_intervals=est_intervals,
            est_pitches=note_to_freq(est_pitches),
            onset_tolerance=self.cfg.score.onset_tolerance,
            offset_ratio=None,
            offset_min_tolerance=self.cfg.score.offset_min_tolerance,
        )
        conpoff_precision, conpoff_recall, conpoff_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=note_to_freq(ref_pitches),
            est_intervals=est_intervals,
            est_pitches=note_to_freq(est_pitches),
            onset_tolerance=self.cfg.score.onset_tolerance,
            offset_ratio=self.cfg.score.offset_ratio,
            offset_min_tolerance=self.cfg.score.offset_min_tolerance,
        )
        result = {
            "precision": conp_precision,
            "recall": conp_recall,
            "f1": conp_f1,
            "COn_precision": con_precision,
            "COn_recall": con_recall,
            "COn": con_f1,
            "COnP_precision": conp_precision,
            "COnP_recall": conp_recall,
            "COnP": conp_f1,
            "COnPOff_precision": conpoff_precision,
            "COnPOff_recall": conpoff_recall,
            "COnPOff": conpoff_f1,
        }
        if "voice_frame_output" in total_dict:
            voice_frame_output = np.asarray(total_dict["voice_frame_output"], dtype=np.float32)
            reference_roll = _reference_frame_roll_by_voice(
                note_bars=note_bars,
                frames_num=voice_frame_output.shape[0],
                frames_per_second=float(self.cfg.feature.frames_per_second),
                begin_note=int(self.cfg.feature.begin_note),
                classes_num=voice_frame_output.shape[-1],
            )
            frame_mask = total_dict.get("frame_mask_roll")
            if frame_mask is None:
                voice_mask = total_dict.get("voice_frame_mask_roll")
                if voice_mask is not None:
                    frame_mask = voice_mask[:, voice_idx, :]
            frame_scores = _frame_metrics(
                reference_roll[:, voice_idx, :],
                voice_frame_output[:, voice_idx, :],
                mask=frame_mask,
                threshold=(thresholds or self.voice_thresholds[voice_name])["frame_threshold"],
            )
            for key, value in frame_scores.items():
                result[f"frame_{key}"] = value
        for onset_tolerance in FIXED_ONSET_TOLERANCES:
            tag = _tolerance_tag(onset_tolerance)
            fixed_precision, fixed_recall, fixed_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=ref_intervals,
                ref_pitches=note_to_freq(ref_pitches),
                est_intervals=est_intervals,
                est_pitches=note_to_freq(est_pitches),
                onset_tolerance=onset_tolerance,
                offset_ratio=None,
                offset_min_tolerance=self.cfg.score.offset_min_tolerance,
            )
            result[f"note_precision_{tag}"] = fixed_precision
            result[f"note_recall_{tag}"] = fixed_recall
            result[f"note_f1_{tag}"] = fixed_f1
            result[f"f1_{tag}"] = fixed_f1
        return result

    def metrics_for_voice(self, voice_name, thresholds=None):
        stats = {}
        for name in self.probability_names:
            stem = os.path.splitext(name)[0]
            prob_path = os.path.join(self.probs_dir, name)
            note_path = os.path.join(self.note_dir, f"{stem}.pkl")
            if not os.path.exists(note_path):
                raise FileNotFoundError(f"Missing SATB reference for probability file {prob_path}: {note_path}")

            song_stats = self.calculate_voice_score_per_song(prob_path, note_path, voice_name, thresholds=thresholds)
            for key, value in song_stats.items():
                stats.setdefault(key, []).append(value)
        return stats

    def calculate_score_per_song(self, prob_path, note_path):
        return_dict = {}
        per_voice_f1 = []
        per_voice_f1_by_tolerance = {_tolerance_tag(t): [] for t in FIXED_ONSET_TOLERANCES}
        estimated_by_voice = {}
        total_dict = self._load_probability_file(prob_path, note_path)
        note_bars = _load_note_bars(note_path)
        note_bars, _duration_adjustment = self._prepare_formal_reference(
            note_bars,
            note_path,
            prob_path,
        )
        for voice_name in VOICE_NAMES:
            song_stats = self.calculate_voice_score_per_song(
                prob_path,
                note_path,
                voice_name,
                total_dict=total_dict,
                note_bars=note_bars,
                reference_validated=True,
            )
            estimated_by_voice[voice_name] = _decode_voice_events(
                total_dict,
                VOICE_TO_INDEX[voice_name],
                self.post_processor,
                thresholds=self.voice_thresholds[voice_name],
            )
            return_dict[f"{voice_name}_precision"] = song_stats["precision"]
            return_dict[f"{voice_name}_recall"] = song_stats["recall"]
            return_dict[f"{voice_name}_f1"] = song_stats["f1"]
            return_dict[f"{voice_name}_COn"] = song_stats["COn"]
            return_dict[f"{voice_name}_COnP"] = song_stats["COnP"]
            return_dict[f"{voice_name}_COnPOff"] = song_stats["COnPOff"]
            for metric_name in ("precision", "recall", "f1"):
                key = f"frame_{metric_name}"
                if key in song_stats:
                    return_dict[f"{voice_name}_frame_{metric_name}"] = song_stats[key]
            for onset_tolerance in FIXED_ONSET_TOLERANCES:
                tag = _tolerance_tag(onset_tolerance)
                return_dict[f"{voice_name}_f1_{tag}"] = song_stats[f"f1_{tag}"]
                per_voice_f1_by_tolerance[tag].append(song_stats[f"f1_{tag}"])
            per_voice_f1.append(song_stats["f1"])
        return_dict["mean_satb_note_f1"] = float(np.mean(per_voice_f1)) if per_voice_f1 else 0.0
        for tag, values in per_voice_f1_by_tolerance.items():
            return_dict[f"mean_satb_note_f1_{tag}"] = float(np.mean(values)) if values else 0.0
        reference_by_voice = _reference_events_by_voice(
            note_bars,
            float(self.cfg.feature.frames_per_second),
        )
        return_dict.update(
            _same_output_assignment_metrics(
                reference_by_voice,
                estimated_by_voice,
                frames_per_second=float(self.cfg.feature.frames_per_second),
                onset_tolerance=float(self.cfg.score.onset_tolerance),
                offset_ratio=float(self.cfg.score.offset_ratio),
                offset_min_tolerance=float(self.cfg.score.offset_min_tolerance),
            )
        )
        return return_dict

    def metrics(self):
        stats = {}
        for name in self.probability_names:
            stem = os.path.splitext(name)[0]
            prob_path = os.path.join(self.probs_dir, name)
            note_path = os.path.join(self.note_dir, f"{stem}.pkl")
            if not os.path.exists(note_path):
                raise FileNotFoundError(f"Missing SATB reference for probability file {prob_path}: {note_path}")

            song_stats = self.calculate_score_per_song(prob_path, note_path)
            for key, value in song_stats.items():
                stats.setdefault(key, []).append(value)
        return stats

    def validate_all_probability_files(self):
        """Preflight every artifact before a combined report prints any metric."""

        for name in self.probability_names:
            stem = os.path.splitext(name)[0]
            prob_path = os.path.join(self.probs_dir, name)
            note_path = os.path.join(self.note_dir, f"{stem}.pkl")
            self._load_probability_file(prob_path, note_path)
            note_bars = _load_note_bars(note_path)
            self._prepare_formal_reference(note_bars, note_path, prob_path)

    def presence_arrays(self, threshold=0.5):
        ref_list = []
        pred_list = []

        for name in self.probability_names:
            stem = os.path.splitext(name)[0]
            prob_path = os.path.join(self.probs_dir, name)
            note_path = os.path.join(self.note_dir, f"{stem}.pkl")
            if not os.path.exists(note_path):
                raise FileNotFoundError(f"Missing SATB reference for probability file {prob_path}: {note_path}")

            total_dict = self._load_probability_file(prob_path, note_path)

            if "voice_presence_output" not in total_dict:
                continue

            pred_presence = np.asarray(total_dict["voice_presence_output"], dtype=np.float32).reshape(-1)
            if pred_presence.size != len(VOICE_NAMES):
                continue

            note_bars = _load_note_bars(note_path)
            note_bars, _duration_adjustment = self._prepare_formal_reference(
                note_bars,
                note_path,
                prob_path,
            )
            ref_presence = _reference_presence_vector(
                note_bars,
                float(self.cfg.feature.frames_per_second),
            )
            ref_list.append(ref_presence)
            pred_list.append((pred_presence >= threshold).astype(np.float32))

        if not ref_list:
            return None, None

        return np.stack(ref_list, axis=0), np.stack(pred_list, axis=0)

    def presence_summary(self, threshold=0.5):
        ref_array, pred_array = self.presence_arrays(threshold=threshold)
        if ref_array is None or pred_array is None:
            return {}

        summary = {
            "presence_accuracy": float(np.mean(pred_array == ref_array)),
            "presence_exact_match_accuracy": float(np.mean(np.all(pred_array == ref_array, axis=1))),
        }

        per_voice_acc = []
        per_voice_f1 = []
        for voice_idx, voice_name in enumerate(VOICE_NAMES):
            ref_voice = ref_array[:, voice_idx]
            pred_voice = pred_array[:, voice_idx]

            tp = float(np.sum((pred_voice == 1.0) & (ref_voice == 1.0)))
            fp = float(np.sum((pred_voice == 1.0) & (ref_voice == 0.0)))
            fn = float(np.sum((pred_voice == 0.0) & (ref_voice == 1.0)))

            acc = float(np.mean(pred_voice == ref_voice))
            precision = tp / max(tp + fp, 1.0)
            recall = tp / max(tp + fn, 1.0)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)

            summary[f"{voice_name}_presence_accuracy"] = acc
            summary[f"{voice_name}_presence_precision"] = precision
            summary[f"{voice_name}_presence_recall"] = recall
            summary[f"{voice_name}_presence_f1"] = f1
            per_voice_acc.append(acc)
            per_voice_f1.append(f1)

        summary["mean_presence_accuracy"] = float(np.mean(per_voice_acc))
        summary["mean_presence_f1"] = float(np.mean(per_voice_f1))
        return summary


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def aggregate_same_output_summary(stats):
    """Aggregate same-output metrics, using micro counts for all ratios."""

    def total(key):
        return int(np.sum(stats.get(key, [])))

    summary = {
        "union_note_precision": _mean(stats.get("union_note_precision", [])),
        "union_note_recall": _mean(stats.get("union_note_recall", [])),
        "union_note_f1": _mean(stats.get("union_note_f1", [])),
        "union_average_overlap_ratio": _mean(
            stats.get("union_average_overlap_ratio", [])
        ),
        "union_note_with_offset_f1": _mean(
            stats.get("union_note_with_offset_f1", [])
        ),
        "union_offset_average_overlap_ratio": _mean(
            stats.get("union_offset_average_overlap_ratio", [])
        ),
        "union_matched_note_count": total("union_matched_note_count"),
        "matched_note_voice_correct_count": total(
            "matched_note_voice_correct_count"
        ),
        "matched_note_voice_eligible_count": total(
            "matched_note_voice_eligible_count"
        ),
        "matched_note_voice_ambiguous_reference_count": total(
            "matched_note_voice_ambiguous_reference_count"
        ),
        "duplicate_estimate_excess_count": total(
            "duplicate_estimate_excess_count"
        ),
        "estimated_voice_note_count": total("estimated_voice_note_count"),
        "voice_switch_count": total("voice_switch_count"),
        "voice_transition_count": total("voice_transition_count"),
        **{
            f"voice_confusion_{source}_{target}": total(
                f"voice_confusion_{source}_{target}"
            )
            for source in VOICE_NAMES
            for target in VOICE_NAMES
        },
    }
    eligible = summary["matched_note_voice_eligible_count"]
    summary["matched_note_voice_accuracy"] = (
        summary["matched_note_voice_correct_count"] / eligible
        if eligible
        else 0.0
    )
    estimated = summary["estimated_voice_note_count"]
    summary["duplicate_estimate_rate"] = (
        summary["duplicate_estimate_excess_count"] / estimated
        if estimated
        else 0.0
    )
    transitions = summary["voice_transition_count"]
    summary["voice_switch_rate"] = (
        summary["voice_switch_count"] / transitions if transitions else 0.0
    )
    return summary


def aggregate_voice_f1_summary(stats):
    voice_f1_means = [_mean(stats.get(f"{voice_name}_f1", [])) for voice_name in VOICE_NAMES]
    if not voice_f1_means:
        return {
            "mean_satb_note_f1": 0.0,
            "min_satb_note_f1": 0.0,
            "harmonic_satb_note_f1": 0.0,
            "balanced_satb_note_f1": 0.0,
            "satb_f1_std": 0.0,
        }

    voice_f1_means = np.asarray(voice_f1_means, dtype=np.float32)
    eps = 1e-8
    mean_f1 = float(np.mean(voice_f1_means))
    min_f1 = float(np.min(voice_f1_means))
    harmonic_f1 = float(len(voice_f1_means) / np.sum(1.0 / np.clip(voice_f1_means, eps, None)))
    std_f1 = float(np.std(voice_f1_means))
    balanced_f1 = float((mean_f1 + min_f1) / 2.0)
    return {
        "mean_satb_note_f1": mean_f1,
        "min_satb_note_f1": min_f1,
        "harmonic_satb_note_f1": harmonic_f1,
        "balanced_satb_note_f1": balanced_f1,
        "satb_f1_std": std_f1,
    }


def aggregate_voice_metric_mean(stats, metric_key):
    values = [_mean(stats.get(f"{voice_name}_{metric_key}", [])) for voice_name in VOICE_NAMES]
    return float(np.mean(values)) if values else 0.0


def print_merged_channel_metrics(cfg, choral_stats):
    """Print the canonical SATB union; never use packed pseudo-reference GT."""

    if not choral_stats:
        return

    summary = aggregate_same_output_summary(choral_stats)
    print("=" * 80)
    print(f"Canonical SATB Union Evaluation | {get_model_name(cfg)} | ckpt={cfg.exp.ckpt_iteration}")
    print("=" * 80)
    display_keys = {
        "note_precision": "union_note_precision",
        "note_recall": "union_note_recall",
        "note_f1": "union_note_f1",
        "note_with_offset_f1": "union_note_with_offset_f1",
    }
    for display_key, summary_key in display_keys.items():
        print(f"{display_key}: {summary[summary_key]:.4f}")


def main():
    initialize(config_path="./", job_name="choral_eval", version_base=None)
    cfg = compose(config_name="config", overrides=sys.argv[1:])

    calculator = ChoralScoreCalculator(cfg)
    calculator.validate_all_probability_files()
    stats = calculator.metrics()
    print_merged_channel_metrics(cfg, stats)

    print("=" * 80)
    print(f"Choral Voice Evaluation | {calculator.model_name} | ckpt={cfg.exp.ckpt_iteration}")
    print("=" * 80)
    for voice_name in VOICE_NAMES:
        p = _mean(stats.get(f"{voice_name}_precision", []))
        r = _mean(stats.get(f"{voice_name}_recall", []))
        f1 = _mean(stats.get(f"{voice_name}_f1", []))
        print(f"{voice_name}_precision: {p:.4f}")
        print(f"{voice_name}_recall: {r:.4f}")
        print(f"{voice_name}_f1: {f1:.4f}")
        print(f"{voice_name}_frame_precision: {_mean(stats.get(f'{voice_name}_frame_precision', [])):.4f}")
        print(f"{voice_name}_frame_recall: {_mean(stats.get(f'{voice_name}_frame_recall', [])):.4f}")
        print(f"{voice_name}_frame_f1: {_mean(stats.get(f'{voice_name}_frame_f1', [])):.4f}")
        for onset_tolerance in FIXED_ONSET_TOLERANCES:
            tag = _tolerance_tag(onset_tolerance)
            print(f"{voice_name}_f1_{tag}: {_mean(stats.get(f'{voice_name}_f1_{tag}', [])):.4f}")
        print(f"{voice_name}_COn: {_mean(stats.get(f'{voice_name}_COn', [])):.4f}")
        print(f"{voice_name}_COnP: {_mean(stats.get(f'{voice_name}_COnP', [])):.4f}")
        print(f"{voice_name}_COnPOff: {_mean(stats.get(f'{voice_name}_COnPOff', [])):.4f}")
    summary = aggregate_voice_f1_summary(stats)
    print(f"mean_satb_note_f1: {summary['mean_satb_note_f1']:.4f}")
    for onset_tolerance in FIXED_ONSET_TOLERANCES:
        tag = _tolerance_tag(onset_tolerance)
        print(f"mean_satb_note_f1_{tag}: {_mean(stats.get(f'mean_satb_note_f1_{tag}', [])):.4f}")
    print(f"min_satb_note_f1: {summary['min_satb_note_f1']:.4f}")
    print(f"harmonic_satb_note_f1: {summary['harmonic_satb_note_f1']:.4f}")
    print(f"balanced_satb_note_f1: {summary['balanced_satb_note_f1']:.4f}")
    print(f"satb_f1_std: {summary['satb_f1_std']:.4f}")
    print(f"mean_satb_COn: {aggregate_voice_metric_mean(stats, 'COn'):.4f}")
    print(f"mean_satb_COnP: {aggregate_voice_metric_mean(stats, 'COnP'):.4f}")
    print(f"mean_satb_COnPOff: {aggregate_voice_metric_mean(stats, 'COnPOff'):.4f}")

    same_output_summary = aggregate_same_output_summary(stats)
    print("=" * 80)
    print(f"Same-output Union and Assignment | {calculator.model_name} | ckpt={cfg.exp.ckpt_iteration}")
    print("=" * 80)
    for key in SAME_OUTPUT_REPORT_KEYS:
        value = same_output_summary[key]
        if key.endswith("_count") or key.startswith("voice_confusion_"):
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.4f}")

    presence_summary = calculator.presence_summary()
    if presence_summary:
        print("=" * 80)
        print(f"Voice Presence Evaluation | {calculator.model_name} | ckpt={cfg.exp.ckpt_iteration}")
        print("=" * 80)
        for voice_name in VOICE_NAMES:
            print(f"{voice_name}_presence_accuracy: {presence_summary[f'{voice_name}_presence_accuracy']:.4f}")
            print(f"{voice_name}_presence_f1: {presence_summary[f'{voice_name}_presence_f1']:.4f}")
        print(f"presence_accuracy: {presence_summary['presence_accuracy']:.4f}")
        print(f"presence_exact_match_accuracy: {presence_summary['presence_exact_match_accuracy']:.4f}")
        print(f"mean_presence_accuracy: {presence_summary['mean_presence_accuracy']:.4f}")
        print(f"mean_presence_f1: {presence_summary['mean_presence_f1']:.4f}")


if __name__ == "__main__":
    main()
