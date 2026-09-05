from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import mido
import numpy as np
from mir_eval.multipitch import evaluate as multipitch_evaluate
from mir_eval.transcription import precision_recall_f1_overlap
from mir_eval.util import midi_to_hz

from infer_midi_voice_assignment import (
    build_model_from_checkpoint,
    infer_song,
    load_checkpoint_args,
    resolve_device,
    write_satb_midi,
)
from train_midi_voice_assignment import (
    DEFAULT_RANGE_MAXS,
    DEFAULT_RANGE_MINS,
    NoteEvent,
    VOICE_NAMES,
    build_song_events,
    create_folder,
    seed_everything,
    setup_logging,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted MIDI files with the symbolic SATB voice-assignment model.",
    )
    parser.add_argument("--va-checkpoint", type=str, required=True)
    parser.add_argument("--pred-midi-dir", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, default="")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--workspace", type=str, default="./workspaces")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--stem", type=str, default="")
    parser.add_argument("--max-songs", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--segment-notes", type=int, default=0)
    parser.add_argument("--eval-stride", type=int, default=0)
    parser.add_argument("--onset-quantization-hz", type=int, default=0)
    parser.add_argument("--range-margin", type=float, default=-1.0)
    parser.add_argument("--apply-range-mask-at-infer", action="store_true")
    parser.add_argument("--frame-hop-sec", type=float, default=0.0625)
    parser.add_argument("--export-midi", action="store_true")
    parser.add_argument("--seed", type=int, default=86)
    return parser


def f1_measure(precision: float, recall: float) -> float:
    if np.isnan(precision) or np.isnan(recall):
        return float("nan")
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def average_numeric_dicts(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = [key for key, value in rows[0].items() if isinstance(value, (int, float, np.floating))]
    out = {}
    for key in keys:
        out[key] = float(np.nanmean([row[key] for row in rows]))
    return out


def resolve_pred_midi_path(pred_midi_dir: Path, stem: str) -> Path:
    candidates = (
        pred_midi_dir / f"{stem}.mid",
        pred_midi_dir / f"{stem}_pred.mid",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    recursive_matches = sorted(pred_midi_dir.rglob(f"{stem}.mid")) + sorted(pred_midi_dir.rglob(f"{stem}_pred.mid"))
    if len(recursive_matches) == 1:
        return recursive_matches[0]
    if len(recursive_matches) > 1:
        newest_match = max(recursive_matches, key=lambda path: path.stat().st_mtime)
        logging.warning(
            "Multiple predicted MIDIs found for stem '%s' in %s; using newest file: %s",
            stem,
            pred_midi_dir,
            newest_match,
        )
        return newest_match
    raise FileNotFoundError(f"Missing predicted MIDI for stem '{stem}' in {pred_midi_dir}")


def load_stems(dataset_dir: Path, split: str, single_stem: str, max_songs: int) -> List[str]:
    if single_stem:
        stems = [single_stem]
    else:
        split_path = dataset_dir / f"{split}.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        with split_path.open("r") as f:
            stems = json.load(f)
        if not isinstance(stems, list):
            raise ValueError(f"Split file must contain a list of stems: {split_path}")
    if max_songs > 0:
        stems = stems[:max_songs]
    return stems


@dataclass(frozen=True)
class ParsedMIDINote:
    pitch: int
    onset: float
    offset: float
    is_drum: bool = False


def load_pred_notes_from_midi(midi_path: Path) -> List[ParsedMIDINote]:
    """Read pitched note events without relying on a sibling YourMT3 checkout.

    Iterating over ``mido.MidiFile`` yields a merged, tempo-aware stream whose
    delta times are expressed in seconds. Overlapping repeated notes are paired
    FIFO per MIDI channel and pitch. Unclosed notes are ignored with a warning.
    """
    midi_file = mido.MidiFile(str(midi_path))
    now = 0.0
    active: Dict[tuple[int, int], List[float]] = {}
    notes: List[ParsedMIDINote] = []

    for message in midi_file:
        now += float(message.time)
        if message.type not in {"note_on", "note_off"}:
            continue
        channel = int(getattr(message, "channel", 0))
        pitch = int(message.note)
        key = (channel, pitch)
        is_note_on = message.type == "note_on" and int(message.velocity) > 0

        if is_note_on:
            active.setdefault(key, []).append(now)
            continue

        starts = active.get(key, [])
        if not starts:
            continue
        onset = starts.pop(0)
        if not starts:
            active.pop(key, None)
        notes.append(
            ParsedMIDINote(
                pitch=pitch,
                onset=onset,
                offset=max(now, onset + 1e-4),
                is_drum=(channel == 9),
            )
        )

    if active:
        logging.warning("Ignoring %d unclosed MIDI note-on events in %s", sum(map(len, active.values())), midi_path)
    return sorted((note for note in notes if not note.is_drum), key=lambda note: (note.onset, note.pitch, note.offset))


def predicted_midi_notes_to_input_events(
    stem: str,
    notes,
    onset_quantization_hz: int,
) -> List[NoteEvent]:
    grouped: Dict[int, List[object]] = {}
    for note in notes:
        onset_group = int(round(float(note.onset) * onset_quantization_hz))
        grouped.setdefault(onset_group, []).append(note)

    final_events: List[NoteEvent] = []
    for onset_group in sorted(grouped):
        group = sorted(
            grouped[onset_group],
            key=lambda x: (-int(x.pitch), float(x.offset), float(x.onset)),
        )
        group_size = len(group)
        for rank_in_group, note in enumerate(group):
            onset_time = float(note.onset)
            offset_time = float(note.offset)
            final_events.append(
                NoteEvent(
                    stem=stem,
                    midi_note=int(note.pitch),
                    onset_time=onset_time,
                    offset_time=offset_time,
                    duration_sec=max(1e-4, offset_time - onset_time),
                    beat_position=0.0,
                    duration_beats=0.0,
                    measure_beats=4.0,
                    voice_idx=-1,
                    onset_group=onset_group,
                    rank_in_group=rank_in_group,
                    group_size=group_size,
                )
            )
    return final_events


def voice_tracks_from_note_events(
    events: Sequence[NoteEvent],
    voice_indices: Sequence[int],
) -> Dict[str, List[Dict[str, float]]]:
    tracks = {voice_name: [] for voice_name in VOICE_NAMES}
    for event, voice_idx in zip(events, voice_indices):
        tracks[VOICE_NAMES[int(voice_idx)]].append(
            {
                "midi_note": int(event.midi_note),
                "onset_time": float(event.onset_time),
                "offset_time": float(event.offset_time),
                "velocity": 100,
            }
        )
    return tracks


def merge_voice_tracks(voice_tracks: Dict[str, Sequence[Dict[str, float]]]) -> List[Dict[str, float]]:
    merged = []
    for voice_name in VOICE_NAMES:
        merged.extend(voice_tracks.get(voice_name, []))
    merged.sort(key=lambda x: (float(x["onset_time"]), int(x["midi_note"]), float(x["offset_time"])))
    return merged


def note_arrays(events: Sequence[Dict[str, float]]):
    if len(events) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    intervals = np.array(
        [[float(event["onset_time"]), float(event["offset_time"])] for event in events],
        dtype=np.float32,
    )
    pitches_hz = np.array([midi_to_hz(int(event["midi_note"])) for event in events], dtype=np.float32)
    return intervals, pitches_hz


def note_metrics(
    ref_events: Sequence[Dict[str, float]],
    est_events: Sequence[Dict[str, float]],
    onset_tolerance: float,
    with_offset: bool = False,
):
    ref_intervals, ref_pitches = note_arrays(ref_events)
    est_intervals, est_pitches = note_arrays(est_events)
    if len(ref_pitches) == 0 and len(est_pitches) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(ref_pitches) == 0 and len(est_pitches) != 0:
        return 0.0, float("nan"), float("nan")
    if len(ref_pitches) != 0 and len(est_pitches) == 0:
        return float("nan"), 0.0, 0.0
    precision, recall, f1, _ = precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=onset_tolerance,
        offset_ratio=0.2 if with_offset else None,
        offset_min_tolerance=0.05,
    )
    return float(precision), float(recall), float(f1)


def extract_frame_time_freq_from_events(
    events: Sequence[Dict[str, float]],
    hop_size_sec: float = 0.0625,
):
    if len(events) == 0:
        return {
            "time": np.array([]),
            "freqs": [[]],
            "roll": np.zeros((0, 128), dtype=np.float32),
        }

    last_offset = max(float(event["offset_time"]) for event in events)
    frames_num = max(1, int(last_offset / hop_size_sec))
    roll = np.zeros((frames_num, 128), dtype=np.float32)
    for event in events:
        onset_frame = int(float(event["onset_time"]) / hop_size_sec)
        onset_frame = int(np.clip(onset_frame, 0, frames_num - 1))
        offset_frame = max(int(float(event["offset_time"]) / hop_size_sec), onset_frame + 1)
        offset_frame = int(np.clip(offset_frame, onset_frame + 1, frames_num))
        roll[onset_frame:offset_frame, int(event["midi_note"])] = 1.0

    roll[:, :16] = 0
    roll[:, 110:] = 0

    time = np.arange(frames_num, dtype=np.float32) * hop_size_sec
    freqs = [np.array([midi_to_hz(pitch) for pitch in roll[t].nonzero()[0]], dtype=np.float32) for t in range(frames_num)]
    return {
        "time": time,
        "freqs": freqs,
        "roll": roll,
    }


def frame_metrics(
    ref_events: Sequence[Dict[str, float]],
    est_events: Sequence[Dict[str, float]],
    hop_size_sec: float,
):
    ref_tf = extract_frame_time_freq_from_events(ref_events, hop_size_sec=hop_size_sec)
    est_tf = extract_frame_time_freq_from_events(est_events, hop_size_sec=hop_size_sec)
    ref_sum = float(np.sum(ref_tf["roll"]))
    est_sum = float(np.sum(est_tf["roll"]))
    if ref_sum == 0 and est_sum == 0:
        return float("nan"), float("nan"), float("nan")
    if ref_sum == 0 and est_sum != 0:
        return 0.0, float("nan"), float("nan")
    if ref_sum != 0 and est_sum == 0:
        return float("nan"), 0.0, 0.0
    result = multipitch_evaluate(
        ref_time=ref_tf["time"],
        ref_freqs=ref_tf["freqs"],
        est_time=est_tf["time"],
        est_freqs=est_tf["freqs"],
    )
    precision = float(result["Precision"])
    recall = float(result["Recall"])
    return precision, recall, f1_measure(precision, recall)


def evaluate_track_sets(
    ref_tracks: Dict[str, Sequence[Dict[str, float]]],
    pred_tracks: Dict[str, Sequence[Dict[str, float]]],
    frame_hop_sec: float = 0.0625,
) -> Dict[str, float]:
    row: Dict[str, float] = {}
    for tolerance, tag in ((0.05, "50ms"), (0.10, "100ms")):
        voice_f1s_no_offset = []
        voice_f1s_with_offset = []
        for voice_name in VOICE_NAMES:
            _, _, f1_no_offset = note_metrics(
                ref_tracks.get(voice_name, []),
                pred_tracks.get(voice_name, []),
                onset_tolerance=tolerance,
                with_offset=False,
            )
            _, _, f1_with_offset = note_metrics(
                ref_tracks.get(voice_name, []),
                pred_tracks.get(voice_name, []),
                onset_tolerance=tolerance,
                with_offset=True,
            )
            # Align default note F1 with calculate_choral_scores.py (offset_ratio=None).
            row[f"{voice_name}_note_f1_{tag}"] = f1_no_offset
            row[f"{voice_name}_note_f1_{tag}_no_offset"] = f1_no_offset
            row[f"{voice_name}_note_f1_{tag}_with_offset"] = f1_with_offset
            voice_f1s_no_offset.append(f1_no_offset)
            voice_f1s_with_offset.append(f1_with_offset)
        _, _, conventional_f1_no_offset = note_metrics(
            merge_voice_tracks(ref_tracks),
            merge_voice_tracks(pred_tracks),
            onset_tolerance=tolerance,
            with_offset=False,
        )
        _, _, conventional_f1_with_offset = note_metrics(
            merge_voice_tracks(ref_tracks),
            merge_voice_tracks(pred_tracks),
            onset_tolerance=tolerance,
            with_offset=True,
        )
        row[f"mean_satb_note_f1_{tag}"] = float(np.nanmean(voice_f1s_no_offset))
        row[f"mean_satb_note_f1_{tag}_no_offset"] = float(np.nanmean(voice_f1s_no_offset))
        row[f"mean_satb_note_f1_{tag}_with_offset"] = float(np.nanmean(voice_f1s_with_offset))
        row[f"conventional_note_f1_{tag}"] = conventional_f1_no_offset
        row[f"conventional_note_f1_{tag}_no_offset"] = conventional_f1_no_offset
        row[f"conventional_note_f1_{tag}_with_offset"] = conventional_f1_with_offset

    voice_frame_f1s = []
    for voice_name in VOICE_NAMES:
        _, _, f1 = frame_metrics(
            ref_tracks.get(voice_name, []),
            pred_tracks.get(voice_name, []),
            hop_size_sec=frame_hop_sec,
        )
        row[f"{voice_name}_frame_f1"] = f1
        voice_frame_f1s.append(f1)
    _, _, conventional_frame_f1 = frame_metrics(
        merge_voice_tracks(ref_tracks),
        merge_voice_tracks(pred_tracks),
        hop_size_sec=frame_hop_sec,
    )
    row["mean_satb_frame_f1"] = float(np.nanmean(voice_frame_f1s))
    row["conventional_frame_f1"] = conventional_frame_f1
    return row


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)

    va_checkpoint = Path(args.va_checkpoint)
    checkpoint = load_checkpoint_args(va_checkpoint)
    saved_args = checkpoint["args"]

    dataset_dir = Path(args.dataset_dir or saved_args["dataset_dir"])
    segment_notes = int(args.segment_notes or saved_args["segment_notes"])
    default_eval_stride = max(1, segment_notes // 2)
    eval_stride = int(args.eval_stride or saved_args.get("segment_stride", default_eval_stride))
    onset_quantization_hz = int(args.onset_quantization_hz or saved_args["onset_quantization_hz"])
    range_margin = float(args.range_margin if args.range_margin >= 0 else saved_args.get("range_margin", 2.0))

    pred_midi_dir = Path(args.pred_midi_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.workspace) / "midi_voice_assignment" / "midi_input_eval" / pred_midi_dir.parent.name / pred_midi_dir.name
    )
    create_folder(output_dir)
    setup_logging(output_dir / "eval.log")

    stems = load_stems(dataset_dir, args.split, args.stem, args.max_songs)
    device = resolve_device(args.device)
    model = build_model_from_checkpoint(checkpoint, device)

    logging.info("VA checkpoint: %s", va_checkpoint)
    logging.info("Pred MIDI dir: %s", pred_midi_dir)
    logging.info("Dataset dir: %s", dataset_dir)
    logging.info("Split: %s | num_songs=%d", args.split, len(stems))
    logging.info("Output dir: %s", output_dir)

    pred_satb_dir = output_dir / "pred_midis"
    ref_satb_dir = output_dir / "ref_midis"
    if args.export_midi:
        create_folder(pred_satb_dir)
        create_folder(ref_satb_dir)

    range_mins = DEFAULT_RANGE_MINS.copy()
    range_maxs = DEFAULT_RANGE_MAXS.copy()
    song_rows: List[Dict[str, float]] = []

    for song_index, stem in enumerate(stems, start=1):
        pred_midi_path = resolve_pred_midi_path(pred_midi_dir, stem)
        pred_notes = load_pred_notes_from_midi(pred_midi_path)
        pred_input_events = predicted_midi_notes_to_input_events(
            stem=stem,
            notes=pred_notes,
            onset_quantization_hz=onset_quantization_hz,
        )
        ref_input_events = build_song_events(dataset_dir, stem, onset_quantization_hz)

        if pred_input_events:
            infer_dict = infer_song(
                model=model,
                events=pred_input_events,
                device=device,
                segment_notes=segment_notes,
                eval_stride=eval_stride,
                range_mins=range_mins,
                range_maxs=range_maxs,
                range_margin=range_margin,
                apply_range_mask_at_infer=args.apply_range_mask_at_infer,
            )
            pred_voice = infer_dict["pred_voice"].astype(np.int64)
            pred_tracks = voice_tracks_from_note_events(pred_input_events, pred_voice)
        else:
            pred_tracks = {voice_name: [] for voice_name in VOICE_NAMES}

        ref_tracks = voice_tracks_from_note_events(
            ref_input_events,
            [event.voice_idx for event in ref_input_events],
        )

        row: Dict[str, float] = {
            "stem": stem,
            "num_pred_notes": len(pred_input_events),
            "num_ref_notes": len(ref_input_events),
        }
        row.update(evaluate_track_sets(ref_tracks, pred_tracks, frame_hop_sec=args.frame_hop_sec))
        song_rows.append(row)

        if args.export_midi:
            write_satb_midi(pred_tracks, pred_satb_dir / f"{stem}_pred.mid")
            write_satb_midi(ref_tracks, ref_satb_dir / f"{stem}_ref.mid")

        logging.info(
            "[%d/%d] %s | pred_notes=%d | note50=%.4f | note100=%.4f | frame=%.4f",
            song_index,
            len(stems),
            stem,
            len(pred_input_events),
            row["mean_satb_note_f1_50ms"],
            row["mean_satb_note_f1_100ms"],
            row["mean_satb_frame_f1"],
        )

    summary = average_numeric_dicts(song_rows)
    summary["num_songs"] = len(song_rows)
    summary["va_checkpoint"] = str(va_checkpoint)
    summary["pred_midi_dir"] = str(pred_midi_dir)
    summary["dataset_dir"] = str(dataset_dir)
    summary["split"] = args.split
    summary["frame_hop_sec"] = float(args.frame_hop_sec)

    report_path = output_dir / "report.json"
    with report_path.open("w") as f:
        json.dump(summary, f, indent=2)

    if song_rows:
        csv_path = output_dir / "song_metrics.csv"
        fieldnames = list(song_rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(song_rows)

    logging.info("Summary: %s", summary)
    logging.info("Saved report to %s", report_path)


if __name__ == "__main__":
    main()
