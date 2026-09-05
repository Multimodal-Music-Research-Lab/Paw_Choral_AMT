from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from train_midi_voice_assignment import (
    DEFAULT_RANGE_MAXS,
    DEFAULT_RANGE_MINS,
    VOICE_NAMES,
    build_segment_sample,
    build_song_events,
    create_folder,
    init_confusion,
    metrics_from_confusion,
    seed_everything,
    setup_logging,
    SymbolicVoiceAssignmentNet,
    update_confusion,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer and evaluate a trained symbolic MIDI voice-assignment model.",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="",
        help="Override dataset dir saved in the checkpoint. Defaults to the training value.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate: train | valid | test",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="",
        help="Run only one YouChorale stem instead of a full split.",
    )
    parser.add_argument("--workspace", type=str, default="./workspaces")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--segment-notes", type=int, default=0)
    parser.add_argument("--eval-stride", type=int, default=0)
    parser.add_argument("--onset-quantization-hz", type=int, default=0)
    parser.add_argument("--range-margin", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=86)
    parser.add_argument("--export-midi", action="store_true")
    parser.add_argument("--export-reference-midi", action="store_true")
    parser.add_argument("--max-songs", type=int, default=0)
    parser.add_argument("--apply-range-mask-at-infer", action="store_true")
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_args(checkpoint_path: Path) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "args" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain saved args: {checkpoint_path}")
    return checkpoint


def build_model_from_checkpoint(
    checkpoint: Dict[str, object],
    device: torch.device,
) -> SymbolicVoiceAssignmentNet:
    saved_args = checkpoint["args"]
    model = SymbolicVoiceAssignmentNet(
        pitch_embed_dim=int(saved_args["pitch_embed_dim"]),
        pitch_class_embed_dim=int(saved_args["pitch_class_embed_dim"]),
        feature_hidden_dim=int(saved_args["feature_hidden_dim"]),
        model_dim=int(saved_args["model_dim"]),
        hidden_size=int(saved_args["hidden_size"]),
        num_layers=int(saved_args["num_layers"]),
        dropout=float(saved_args["dropout"]),
        num_voices=len(VOICE_NAMES),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_stems(dataset_dir: Path, split: str, single_stem: str) -> List[str]:
    if single_stem:
        return [single_stem]
    split_path = dataset_dir / f"{split}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    with split_path.open("r") as f:
        stems = json.load(f)
    if not isinstance(stems, list):
        raise ValueError(f"Split file must contain a list of stems: {split_path}")
    return stems


def write_satb_midi(
    voice_events: Dict[str, Sequence[Dict[str, float]]],
    midi_path: Path,
    tempo_bpm: float = 120.0,
) -> None:
    try:
        from mido import Message, MetaMessage, MidiFile, MidiTrack
    except Exception as exc:
        raise ImportError("mido is required to export MIDI files.") from exc

    create_folder(midi_path.parent)
    ticks_per_beat = 384
    ticks_per_second = ticks_per_beat * tempo_bpm / 60.0
    microseconds_per_beat = int(60.0 * 1e6 / tempo_bpm)

    midi_file = MidiFile(ticks_per_beat=ticks_per_beat)

    meta_track = MidiTrack()
    meta_track.append(MetaMessage("set_tempo", tempo=microseconds_per_beat, time=0))
    meta_track.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta_track.append(MetaMessage("end_of_track", time=1))
    midi_file.tracks.append(meta_track)

    for voice_name in VOICE_NAMES:
        track = MidiTrack()
        track.append(MetaMessage("track_name", name=voice_name, time=0))
        events = []
        for note_event in voice_events.get(voice_name, []):
            velocity = int(note_event.get("velocity", 100))
            events.append((float(note_event["onset_time"]), 1, int(note_event["midi_note"]), velocity))
            events.append((float(note_event["offset_time"]), 0, int(note_event["midi_note"]), 0))
        events.sort(key=lambda x: (x[0], -x[1], x[2]))

        previous_ticks = 0
        for abs_time, is_onset, midi_note, velocity in events:
            this_ticks = max(0, int(round(abs_time * ticks_per_second)))
            diff_ticks = this_ticks - previous_ticks
            previous_ticks = this_ticks
            track.append(
                Message(
                    "note_on" if is_onset else "note_off",
                    note=midi_note,
                    velocity=velocity,
                    time=diff_ticks,
                )
            )
        track.append(MetaMessage("end_of_track", time=1))
        midi_file.tracks.append(track)

    midi_file.save(str(midi_path))


def events_to_voice_tracks(events, voice_indices: np.ndarray) -> Dict[str, List[Dict[str, float]]]:
    voice_tracks = {voice_name: [] for voice_name in VOICE_NAMES}
    for event, voice_idx in zip(events, voice_indices.tolist()):
        voice_name = VOICE_NAMES[int(voice_idx)]
        voice_tracks[voice_name].append(
            {
                "midi_note": int(event.midi_note),
                "onset_time": float(event.onset_time),
                "offset_time": float(event.offset_time),
                "velocity": 100,
            }
        )
    return voice_tracks


def infer_song(
    model: SymbolicVoiceAssignmentNet,
    events,
    device: torch.device,
    segment_notes: int,
    eval_stride: int,
    range_mins: np.ndarray,
    range_maxs: np.ndarray,
    range_margin: float,
    apply_range_mask_at_infer: bool,
) -> Dict[str, np.ndarray]:
    total_notes = len(events)
    logits_sum = np.zeros((total_notes, len(VOICE_NAMES)), dtype=np.float32)
    logits_count = np.zeros((total_notes, 1), dtype=np.float32)
    presence_logits_list: List[np.ndarray] = []

    for start in range(0, total_notes, eval_stride):
        end = min(total_notes, start + segment_notes)
        if end <= start:
            continue
        segment_events = events[start:end]
        sample = build_segment_sample(
            events=segment_events,
            segment_notes=segment_notes,
            range_mins=range_mins,
            range_maxs=range_maxs,
            range_margin=range_margin,
        )
        valid_len = len(segment_events)

        batch = {
            "pitch_ids": torch.from_numpy(sample["pitch_ids"]).unsqueeze(0).to(device),
            "pitch_classes": torch.from_numpy(sample["pitch_classes"]).unsqueeze(0).to(device),
            "cont_features": torch.from_numpy(sample["cont_features"]).unsqueeze(0).to(device),
            "valid_mask": torch.from_numpy(sample["valid_mask"]).unsqueeze(0).to(device),
        }
        with torch.no_grad():
            outputs = model(
                batch["pitch_ids"],
                batch["pitch_classes"],
                batch["cont_features"],
                batch["valid_mask"],
            )
        logits = outputs["logits"][0, :valid_len].detach().cpu().numpy()
        if apply_range_mask_at_infer:
            logits = logits + np.log(sample["range_mask"][:valid_len] + 1e-6)

        logits_sum[start:end] += logits
        logits_count[start:end] += 1.0
        presence_logits_list.append(outputs["presence_logits"][0].detach().cpu().numpy())

        if end == total_notes:
            break

    logits_avg = logits_sum / np.clip(logits_count, 1.0, None)
    pred_voice = np.argmax(logits_avg, axis=-1)
    mean_presence_logits = np.mean(np.stack(presence_logits_list, axis=0), axis=0)
    presence_pred = (1.0 / (1.0 + np.exp(-mean_presence_logits)) >= 0.5).astype(np.float32)

    return {
        "logits": logits_avg,
        "pred_voice": pred_voice,
        "presence_pred": presence_pred,
    }


def song_metrics(true_voice: np.ndarray, pred_voice: np.ndarray) -> Dict[str, float]:
    confusion = init_confusion(len(VOICE_NAMES))
    update_confusion(confusion, true_voice, pred_voice)
    return metrics_from_confusion(confusion)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    seed_everything(args.seed)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint_args(checkpoint_path)
    saved_args = checkpoint["args"]

    dataset_dir = Path(args.dataset_dir or saved_args["dataset_dir"])
    segment_notes = int(args.segment_notes or saved_args["segment_notes"])
    default_eval_stride = max(1, segment_notes // 2)
    eval_stride = int(args.eval_stride or saved_args.get("segment_stride", default_eval_stride))
    onset_quantization_hz = int(args.onset_quantization_hz or saved_args["onset_quantization_hz"])
    range_margin = float(args.range_margin if args.range_margin >= 0 else saved_args.get("range_margin", 2.0))
    experiment_name = saved_args.get("experiment_name", checkpoint_path.parent.parent.name)

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.workspace) / "midi_voice_assignment" / experiment_name / f"infer_{args.split}"
    )
    create_folder(output_dir)
    setup_logging(output_dir / "infer.log")

    device = resolve_device(args.device)
    model = build_model_from_checkpoint(checkpoint, device)

    stems = load_stems(dataset_dir, args.split, args.stem)
    if args.max_songs > 0:
        stems = stems[: args.max_songs]

    logging.info("Checkpoint: %s", checkpoint_path)
    logging.info("Dataset dir: %s", dataset_dir)
    logging.info("Split: %s | num_songs=%d", args.split, len(stems))
    logging.info("Device: %s", device)
    logging.info("Output dir: %s", output_dir)

    range_mins = DEFAULT_RANGE_MINS.copy()
    range_maxs = DEFAULT_RANGE_MAXS.copy()

    overall_confusion = init_confusion(len(VOICE_NAMES))
    total_presence_correct = 0.0
    total_presence_total = 0.0
    song_rows: List[Dict[str, float]] = []

    pred_midi_dir = output_dir / "pred_midis"
    ref_midi_dir = output_dir / "ref_midis"
    if args.export_midi:
        create_folder(pred_midi_dir)
    if args.export_reference_midi:
        create_folder(ref_midi_dir)

    for song_index, stem in enumerate(stems, start=1):
        events = build_song_events(dataset_dir, stem, onset_quantization_hz)
        if not events:
            logging.warning("Skip empty stem: %s", stem)
            continue

        infer_dict = infer_song(
            model=model,
            events=events,
            device=device,
            segment_notes=segment_notes,
            eval_stride=eval_stride,
            range_mins=range_mins,
            range_maxs=range_maxs,
            range_margin=range_margin,
            apply_range_mask_at_infer=args.apply_range_mask_at_infer,
        )

        true_voice = np.array([event.voice_idx for event in events], dtype=np.int64)
        pred_voice = infer_dict["pred_voice"].astype(np.int64)
        update_confusion(overall_confusion, true_voice, pred_voice)

        presence_true = np.zeros((len(VOICE_NAMES),), dtype=np.float32)
        presence_true[np.unique(true_voice)] = 1.0
        total_presence_correct += float((infer_dict["presence_pred"] == presence_true).sum())
        total_presence_total += float(len(VOICE_NAMES))

        row = {"stem": stem}
        row.update(song_metrics(true_voice, pred_voice))
        row["presence_accuracy"] = float(np.mean(infer_dict["presence_pred"] == presence_true))
        song_rows.append(row)

        if args.export_midi:
            write_satb_midi(
                events_to_voice_tracks(events, pred_voice),
                pred_midi_dir / f"{stem}_pred.mid",
            )
        if args.export_reference_midi:
            write_satb_midi(
                events_to_voice_tracks(events, true_voice),
                ref_midi_dir / f"{stem}_ref.mid",
            )

        logging.info(
            "[%d/%d] %s | acc=%.4f | macro_f1=%.4f | S=%.4f A=%.4f T=%.4f B=%.4f",
            song_index,
            len(stems),
            stem,
            row["note_assignment_accuracy"],
            row["macro_note_f1"],
            row["S_f1"],
            row["A_f1"],
            row["T_f1"],
            row["B_f1"],
        )

    overall_metrics = metrics_from_confusion(overall_confusion)
    overall_metrics["presence_accuracy"] = total_presence_correct / max(1.0, total_presence_total)
    overall_metrics["num_songs"] = len(song_rows)
    overall_metrics["checkpoint"] = str(checkpoint_path)
    overall_metrics["split"] = args.split

    report_path = output_dir / "report.json"
    with report_path.open("w") as f:
        json.dump(overall_metrics, f, indent=2)

    if song_rows:
        csv_path = output_dir / "song_metrics.csv"
        fieldnames = list(song_rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(song_rows)

    logging.info("Overall metrics: %s", overall_metrics)
    logging.info("Saved report to %s", report_path)


if __name__ == "__main__":
    main()
