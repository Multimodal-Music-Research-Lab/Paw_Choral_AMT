from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import torch
from mir_eval.transcription import precision_recall_f1_overlap

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
    seed_everything,
)
from utilities import (
    OnsetsFramesPostProcessor,
    RegressionPostProcessor,
    note_to_freq,
)
from visualize_midi_voice_assignment import plot_pair


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a symbolic MIDI voice-assignment model on audio-model predicted notes from probs/*.pkl.",
    )
    parser.add_argument("--va-checkpoint", type=str, required=True)
    parser.add_argument("--probs-dir", type=str, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./data/YouChorale",
    )
    parser.add_argument("--workspace", type=str, default="./workspaces")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--stem", type=str, default="")
    parser.add_argument("--max-songs", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--post-processor-type", type=str, default="regression")
    parser.add_argument("--frame-threshold", type=float, default=0.07)
    parser.add_argument("--onset-threshold", type=float, default=0.03)
    parser.add_argument("--offset-threshold", type=float, default=0.003)
    parser.add_argument("--default-velocity", type=int, default=80)
    parser.add_argument("--frames-per-second", type=int, default=100)
    parser.add_argument("--classes-num", type=int, default=88)
    parser.add_argument("--begin-note", type=int, default=21)
    parser.add_argument("--velocity-scale", type=int, default=128)
    parser.add_argument("--segment-notes", type=int, default=0)
    parser.add_argument("--eval-stride", type=int, default=0)
    parser.add_argument("--onset-quantization-hz", type=int, default=0)
    parser.add_argument("--range-margin", type=float, default=-1.0)
    parser.add_argument("--apply-range-mask-at-infer", action="store_true")
    parser.add_argument("--export-midi", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--seed", type=int, default=86)
    return parser


def create_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logging(log_path: Path) -> None:
    create_folder(log_path.parent)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )


def build_post_processor(args: argparse.Namespace):
    cfg = SimpleNamespace(
        feature=SimpleNamespace(
            frames_per_second=args.frames_per_second,
            classes_num=args.classes_num,
            begin_note=args.begin_note,
            velocity_scale=args.velocity_scale,
        ),
        post=SimpleNamespace(
            frame_threshold=args.frame_threshold,
            onset_threshold=args.onset_threshold,
            offset_threshold=args.offset_threshold,
            pedal_offset_threshold=0.2,
            default_velocity=args.default_velocity,
        ),
    )
    if args.post_processor_type.lower() == "regression":
        return RegressionPostProcessor(cfg)
    if args.post_processor_type.lower() in {"onsets_frames", "onf", "onset_frames"}:
        return OnsetsFramesPostProcessor(cfg)
    raise ValueError(f"Unsupported post processor: {args.post_processor_type}")


def probs_files(probs_dir: Path, single_stem: str, max_songs: int) -> List[Path]:
    files = sorted(probs_dir.glob("*.pkl"))
    if single_stem:
        files = [path for path in files if path.stem == single_stem]
    if max_songs > 0:
        files = files[:max_songs]
    return files


def predicted_note_events_to_input_events(
    stem: str,
    note_events: Sequence[Dict[str, float]],
    onset_quantization_hz: int,
) -> List[NoteEvent]:
    grouped: Dict[int, List[Dict[str, float]]] = {}
    for note_event in note_events:
        onset_group = int(round(float(note_event["onset_time"]) * onset_quantization_hz))
        grouped.setdefault(onset_group, []).append(note_event)

    final_events: List[NoteEvent] = []
    for onset_group in sorted(grouped):
        group = sorted(
            grouped[onset_group],
            key=lambda x: (-float(x["midi_note"]), float(x["offset_time"]), float(x["onset_time"])),
        )
        group_size = len(group)
        for rank_in_group, note_event in enumerate(group):
            onset_time = float(note_event["onset_time"])
            offset_time = float(note_event["offset_time"])
            final_events.append(
                NoteEvent(
                    stem=stem,
                    midi_note=int(note_event["midi_note"]),
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


def voice_tracks_from_note_events(events: Sequence[NoteEvent], voice_indices: Sequence[int]) -> Dict[str, List[Dict[str, float]]]:
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


def note_arrays(events: Sequence[Dict[str, float]]):
    if len(events) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    intervals = np.array(
        [[float(event["onset_time"]), float(event["offset_time"])] for event in events],
        dtype=np.float32,
    )
    pitches_hz = note_to_freq(np.array([int(event["midi_note"]) for event in events], dtype=np.float32)).astype(np.float32)
    return intervals, pitches_hz


def voice_note_metrics(ref_events: Sequence[Dict[str, float]], est_events: Sequence[Dict[str, float]], onset_tolerance: float):
    ref_intervals, ref_pitches = note_arrays(ref_events)
    est_intervals, est_pitches = note_arrays(est_events)
    precision, recall, f1, _ = precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=onset_tolerance,
        offset_ratio=0.2,
        offset_min_tolerance=0.05,
    )
    return float(precision), float(recall), float(f1)


def average_numeric_dicts(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = [key for key, value in rows[0].items() if isinstance(value, (int, float, np.floating))]
    out = {}
    for key in keys:
        out[key] = float(np.mean([row[key] for row in rows]))
    return out


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)

    va_checkpoint = Path(args.va_checkpoint)
    checkpoint = load_checkpoint_args(va_checkpoint)
    saved_args = checkpoint["args"]

    segment_notes = int(args.segment_notes or saved_args["segment_notes"])
    eval_stride = int(args.eval_stride or saved_args.get("segment_stride", max(1, segment_notes // 2)))
    onset_quantization_hz = int(args.onset_quantization_hz or saved_args["onset_quantization_hz"])
    range_margin = float(args.range_margin if args.range_margin >= 0 else saved_args.get("range_margin", 2.0))

    probs_dir = Path(args.probs_dir)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.workspace) / "midi_voice_assignment" / "audio_conditioned_eval" / probs_dir.parent.name / probs_dir.name
    )
    create_folder(output_dir)
    setup_logging(output_dir / "eval.log")

    device = resolve_device(args.device)
    model = build_model_from_checkpoint(checkpoint, device)
    post_processor = build_post_processor(args)

    prob_files = probs_files(probs_dir, args.stem, args.max_songs)
    if not prob_files:
        raise FileNotFoundError(f"No probs files found in {probs_dir}")

    logging.info("VA checkpoint: %s", va_checkpoint)
    logging.info("Probs dir: %s", probs_dir)
    logging.info("Num songs: %d", len(prob_files))
    logging.info("Output dir: %s", output_dir)

    pred_midi_dir = output_dir / "pred_midis"
    ref_midi_dir = output_dir / "ref_midis"
    viz_dir = output_dir / "visualizations"
    if args.export_midi:
        create_folder(pred_midi_dir)
        create_folder(ref_midi_dir)
    if args.visualize:
        create_folder(viz_dir)

    range_mins = DEFAULT_RANGE_MINS.copy()
    range_maxs = DEFAULT_RANGE_MAXS.copy()

    song_rows: List[Dict[str, float]] = []

    for index, prob_path in enumerate(prob_files, start=1):
        stem = prob_path.stem
        with prob_path.open("rb") as f:
            total_dict = pickle.load(f)

        post_input = {}
        for key, value in total_dict.items():
            if isinstance(value, np.ndarray):
                post_input[key] = np.array(value, copy=True)

        est_note_events, _ = post_processor.output_dict_to_midi_events(post_input)
        pred_input_events = predicted_note_events_to_input_events(
            stem=stem,
            note_events=est_note_events,
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
        ref_tracks = voice_tracks_from_note_events(ref_input_events, [event.voice_idx for event in ref_input_events])

        row: Dict[str, float] = {"stem": stem, "num_pred_notes": len(pred_input_events), "num_ref_notes": len(ref_input_events)}
        for tolerance, tag in ((0.05, "50ms"), (0.1, "100ms")):
            voice_f1s = []
            for voice_name in VOICE_NAMES:
                precision, recall, f1 = voice_note_metrics(
                    ref_tracks[voice_name],
                    pred_tracks[voice_name],
                    onset_tolerance=tolerance,
                )
                row[f"{voice_name}_precision_{tag}"] = precision
                row[f"{voice_name}_recall_{tag}"] = recall
                row[f"{voice_name}_f1_{tag}"] = f1
                voice_f1s.append(f1)
            row[f"mean_satb_note_f1_{tag}"] = float(np.mean(voice_f1s))
            row[f"min_satb_note_f1_{tag}"] = float(np.min(voice_f1s))

        song_rows.append(row)

        if args.export_midi:
            pred_midi_path = pred_midi_dir / f"{stem}_pred.mid"
            ref_midi_path = ref_midi_dir / f"{stem}_ref.mid"
            write_satb_midi(pred_tracks, pred_midi_path)
            write_satb_midi(ref_tracks, ref_midi_path)
            if args.visualize:
                plot_pair(
                    pred_path=pred_midi_path,
                    ref_path=ref_midi_path,
                    output_path=viz_dir / f"{stem}.png",
                    min_pitch=36,
                    max_pitch=90,
                    alpha=0.82,
                    line_width=0.8,
                    dpi=180,
                )

        logging.info(
            "[%d/%d] %s | pred_notes=%d | mean_F1_50ms=%.4f | mean_F1_100ms=%.4f",
            index,
            len(prob_files),
            stem,
            len(pred_input_events),
            row["mean_satb_note_f1_50ms"],
            row["mean_satb_note_f1_100ms"],
        )

    summary = average_numeric_dicts(song_rows)
    summary["num_songs"] = len(song_rows)
    summary["va_checkpoint"] = str(va_checkpoint)
    summary["probs_dir"] = str(probs_dir)
    summary["post_processor_type"] = args.post_processor_type
    summary["frame_threshold"] = args.frame_threshold
    summary["onset_threshold"] = args.onset_threshold
    summary["offset_threshold"] = args.offset_threshold

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
