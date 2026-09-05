from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from infer_midi_voice_assignment import write_satb_midi
from train_midi_voice_assignment import build_song_events, create_folder, load_split, seed_everything, setup_logging
from train_symbolic_satb_editor import (
    FIXED_ONSET_TOLERANCES,
    build_clean_song_input_rolls,
    build_model_from_args,
    compute_choral_note_summary,
    corrupt_merged_events,
    decode_satb_tracks,
    infer_song_rolls,
    reference_voice_tracks,
    resolve_device,
    stable_segment_seed,
    voice_events_to_merged_events,
)
from utilities import read_midi


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer and evaluate the symbolic SATB editor.",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, default="")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--stem", type=str, default="")
    parser.add_argument("--midi-path", type=str, default="")
    parser.add_argument("--workspace", type=str, default="./workspaces")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--segment-seconds", type=float, default=0.0)
    parser.add_argument("--segment-stride-seconds", type=float, default=0.0)
    parser.add_argument("--frames-per-second", type=int, default=0)
    parser.add_argument("--frame-threshold", type=float, default=0.50)
    parser.add_argument("--onset-threshold", type=float, default=0.50)
    parser.add_argument("--offset-threshold", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=86)
    parser.add_argument("--max-songs", type=int, default=0)
    parser.add_argument("--use-corrupted-input", action="store_true")
    parser.add_argument("--export-midi", action="store_true")
    parser.add_argument("--export-reference-midi", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    return parser


def load_checkpoint(checkpoint_path: Path) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "args" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain saved args: {checkpoint_path}")
    return checkpoint


def load_model_from_checkpoint(
    checkpoint: Dict[str, object],
    device: torch.device,
) -> torch.nn.Module:
    saved_args = argparse.Namespace(**checkpoint["args"])
    model = build_model_from_args(saved_args).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_note_events_from_midi(midi_path: Path) -> List[Dict[str, float]]:
    midi_dict = read_midi(str(midi_path), dataset="youchorale")
    buffers: Dict[int, List[tuple[float, int]]] = defaultdict(list)
    note_events: List[Dict[str, float]] = []
    last_time = 0.0

    for midi_event, event_time in zip(midi_dict["midi_event"], midi_dict["midi_event_time"]):
        last_time = max(last_time, float(event_time))
        parts = str(midi_event).split(" ")
        if not parts or parts[0] not in {"note_on", "note_off"}:
            continue
        midi_note = int(parts[2].split("=")[1])
        velocity = int(parts[3].split("=")[1])

        if parts[0] == "note_on" and velocity > 0:
            buffers[midi_note].append((float(event_time), velocity))
            continue

        if buffers[midi_note]:
            onset_time, onset_velocity = buffers[midi_note].pop(0)
            if event_time > onset_time:
                note_events.append(
                    {
                        "midi_note": midi_note,
                        "onset_time": float(onset_time),
                        "offset_time": float(event_time),
                        "velocity": int(onset_velocity),
                    }
                )

    for midi_note, queued in buffers.items():
        for onset_time, onset_velocity in queued:
            note_events.append(
                {
                    "midi_note": int(midi_note),
                    "onset_time": float(onset_time),
                    "offset_time": float(last_time + 0.1),
                    "velocity": int(onset_velocity),
                }
            )

    note_events.sort(key=lambda x: (x["onset_time"], x["midi_note"], x["offset_time"]))
    return note_events


def build_input_rolls_from_note_events(
    note_events: Sequence[Dict[str, float]],
    frames_per_second: int,
) -> np.ndarray:
    max_offset = max((float(event["offset_time"]) for event in note_events), default=0.0)
    frames_num = max(1, int(round(max_offset * frames_per_second)) + 1)
    from train_symbolic_satb_editor import build_roll_triplet

    roll_dict = build_roll_triplet(note_events, frames_num, frames_per_second)
    return np.stack([roll_dict["onset"], roll_dict["frame"], roll_dict["offset"]], axis=1).astype(np.float32)


def maybe_corrupt_song_input(
    song_events,
    stem: str,
    saved_args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    merged_events = voice_events_to_merged_events(song_events)
    segment_seconds = max((event["offset_time"] for event in merged_events), default=0.0)
    rng = np.random.default_rng(stable_segment_seed(seed, stem, 0))
    corrupted = corrupt_merged_events(
        events=merged_events,
        rng=rng,
        segment_seconds=max(segment_seconds, 1e-3),
        frames_per_second=int(saved_args.frames_per_second),
        drop_note_prob=float(saved_args.drop_note_prob),
        pitch_shift_prob=float(saved_args.pitch_shift_prob),
        max_pitch_shift=int(saved_args.max_pitch_shift),
        onset_jitter_prob=float(saved_args.onset_jitter_prob),
        max_onset_jitter_frames=int(saved_args.max_onset_jitter_frames),
        offset_jitter_prob=float(saved_args.offset_jitter_prob),
        max_offset_jitter_frames=int(saved_args.max_offset_jitter_frames),
        extra_note_ratio=float(saved_args.extra_note_ratio),
        extra_pitch_span=int(saved_args.extra_pitch_span),
        min_note_frames=int(saved_args.min_note_frames),
    )
    return build_input_rolls_from_note_events(corrupted, int(saved_args.frames_per_second))


def resolve_output_dir(
    args: argparse.Namespace,
    checkpoint_path: Path,
) -> Path:
    if args.output_dir:
        return Path(args.output_dir)

    experiment_name = checkpoint_path.parent.parent.name
    workspace = Path(args.workspace)
    if args.midi_path:
        stem_name = Path(args.midi_path).stem
        return workspace / "symbolic_satb_editor" / experiment_name / "infer_midi" / stem_name
    if args.stem:
        return workspace / "symbolic_satb_editor" / experiment_name / f"infer_{args.stem}"
    return workspace / "symbolic_satb_editor" / experiment_name / f"infer_{args.split}"


def mean_report(song_reports: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not song_reports:
        return {}
    numeric_keys = sorted(song_reports[0].keys())
    result = {}
    for key in numeric_keys:
        values = [float(report[key]) for report in song_reports if key in report]
        if values:
            result[key] = float(np.mean(values))
    return result


def write_song_metrics_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    create_folder(path.parent)
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    seed_everything(args.seed)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    saved_args = argparse.Namespace(**checkpoint["args"])
    output_dir = resolve_output_dir(args, checkpoint_path)
    create_folder(output_dir)
    setup_logging(output_dir / "infer.log")

    device = resolve_device(args.device)
    model = load_model_from_checkpoint(checkpoint, device)

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(saved_args.dataset_dir)
    frames_per_second = int(args.frames_per_second) if args.frames_per_second > 0 else int(saved_args.frames_per_second)
    segment_seconds = float(args.segment_seconds) if args.segment_seconds > 0 else float(saved_args.segment_seconds)
    segment_stride_seconds = (
        float(args.segment_stride_seconds)
        if args.segment_stride_seconds > 0
        else float(saved_args.segment_stride_seconds)
    )

    logging.info("Using device: %s", device)
    logging.info("Output dir: %s", output_dir)
    logging.info(
        "Inference config | fps=%d | segment_seconds=%.2f | stride_seconds=%.2f",
        frames_per_second,
        segment_seconds,
        segment_stride_seconds,
    )

    if args.midi_path:
        stems = [args.stem or Path(args.midi_path).stem]
    elif args.stem:
        stems = [args.stem]
    else:
        stems = load_split(dataset_dir, args.split)
    if args.max_songs > 0:
        stems = stems[: args.max_songs]

    pred_midi_dir = output_dir / "pred_midis"
    ref_midi_dir = output_dir / "ref_midis"
    vis_dir = output_dir / "visualizations"
    if args.export_midi:
        create_folder(pred_midi_dir)
    if args.export_reference_midi:
        create_folder(ref_midi_dir)
    if args.visualize:
        create_folder(vis_dir)

    song_rows: List[Dict[str, object]] = []
    song_reports: List[Dict[str, float]] = []

    for stem in stems:
        logging.info("Processing %s", stem)
        reference_tracks = None

        if args.midi_path:
            input_rolls = build_input_rolls_from_note_events(
                load_note_events_from_midi(Path(args.midi_path)),
                frames_per_second=frames_per_second,
            )
            note_path = dataset_dir / "note" / f"{stem}.pkl"
            if note_path.exists():
                song_events = build_song_events(dataset_dir, stem, frames_per_second)
                reference_tracks = reference_voice_tracks(song_events)
        else:
            song_events = build_song_events(dataset_dir, stem, frames_per_second)
            reference_tracks = reference_voice_tracks(song_events)
            if args.use_corrupted_input:
                input_rolls = maybe_corrupt_song_input(song_events, stem, saved_args, args.seed)
            else:
                input_rolls = build_clean_song_input_rolls(song_events, frames_per_second)

        averaged_outputs = infer_song_rolls(
            model=model,
            input_rolls=input_rolls,
            device=device,
            segment_seconds=segment_seconds,
            segment_stride_seconds=segment_stride_seconds,
            frames_per_second=frames_per_second,
        )
        pred_tracks = decode_satb_tracks(
            averaged_outputs=averaged_outputs,
            frames_per_second=frames_per_second,
            frame_threshold=args.frame_threshold,
            onset_threshold=args.onset_threshold,
            offset_threshold=args.offset_threshold,
        )

        row: Dict[str, object] = {
            "stem": stem,
            "num_pred_notes": int(sum(len(pred_tracks[name]) for name in pred_tracks)),
        }

        if reference_tracks is not None:
            report = compute_choral_note_summary(reference_tracks, pred_tracks)
            report["num_ref_notes"] = int(sum(len(reference_tracks[name]) for name in reference_tracks))
            row.update(report)
            song_reports.append(report)

        song_rows.append(row)

        if args.export_midi:
            pred_path = pred_midi_dir / f"{stem}_pred.mid"
            write_satb_midi(pred_tracks, pred_path)
        else:
            pred_path = None

        if args.export_reference_midi and reference_tracks is not None:
            ref_path = ref_midi_dir / f"{stem}_ref.mid"
            write_satb_midi(reference_tracks, ref_path)
        else:
            ref_path = None

        if args.visualize and pred_path is not None and ref_path is not None:
            from visualize_midi_voice_assignment import plot_pair

            plot_pair(
                pred_path=pred_path,
                ref_path=ref_path,
                output_path=vis_dir / f"{stem}.png",
                min_pitch=36,
                max_pitch=90,
                alpha=0.82,
                line_width=0.8,
                dpi=180,
            )

    aggregate_report = mean_report(song_reports)
    aggregate_report["num_songs"] = len(stems)
    aggregate_report["checkpoint"] = str(checkpoint_path)
    aggregate_report["frame_threshold"] = float(args.frame_threshold)
    aggregate_report["onset_threshold"] = float(args.onset_threshold)
    aggregate_report["offset_threshold"] = float(args.offset_threshold)
    aggregate_report["mode"] = "midi_path" if args.midi_path else "split"
    if not args.midi_path:
        aggregate_report["split"] = args.split if not args.stem else args.stem
    aggregate_report["onset_tolerances"] = [float(t) for t in FIXED_ONSET_TOLERANCES]

    with (output_dir / "report.json").open("w") as f:
        json.dump(aggregate_report, f, indent=2)
    write_song_metrics_csv(output_dir / "song_metrics.csv", song_rows)

    logging.info("Finished %d song(s)", len(stems))
    if song_reports:
        logging.info(
            "mean_satb_note_f1_50ms=%.4f | mean_satb_note_f1_100ms=%.4f | presence_accuracy=%.4f",
            aggregate_report.get("mean_satb_note_f1_50ms", 0.0),
            aggregate_report.get("mean_satb_note_f1_100ms", 0.0),
            aggregate_report.get("presence_accuracy", 0.0),
        )


if __name__ == "__main__":
    main()
