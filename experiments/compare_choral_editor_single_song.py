from __future__ import annotations

import argparse
import copy
import json
import logging
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from infer_midi_voice_assignment import write_satb_midi
from infer_symbolic_satb_editor import build_input_rolls_from_note_events, load_model_from_checkpoint
from train_midi_voice_assignment import VOICE_NAMES, build_song_events
from train_symbolic_satb_editor import (
    compute_choral_note_summary,
    decode_satb_tracks,
    infer_song_rolls,
    reference_voice_tracks,
    resolve_device,
)
from utilities import OnsetsFramesPostProcessor, RegressionPostProcessor, write_events_to_midi
from visualize_midi_voice_assignment import VOICE_COLORS, create_folder, draw_voice_roll


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare direct choral decoding against symbolic SATB editor on a single song.",
    )
    parser.add_argument("--editor-checkpoint", type=str, required=True)
    parser.add_argument("--probs-dir", type=str, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./data/YouChorale",
    )
    parser.add_argument("--stem", type=str, default="")
    parser.add_argument("--workspace", type=str, default="./workspaces")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--post-processor-type", type=str, default="regression")
    parser.add_argument("--voice-frame-thresholds", type=str, default="0.02,0.02,0.02,0.02")
    parser.add_argument("--voice-onset-thresholds", type=str, default="0.01,0.01,0.01,0.01")
    parser.add_argument("--voice-offset-thresholds", type=str, default="0.003,0.003,0.003,0.003")
    parser.add_argument("--merged-frame-threshold", type=float, default=-1.0)
    parser.add_argument("--merged-onset-threshold", type=float, default=-1.0)
    parser.add_argument("--merged-offset-threshold", type=float, default=-1.0)
    parser.add_argument("--editor-frame-threshold", type=float, default=0.10)
    parser.add_argument("--editor-onset-threshold", type=float, default=0.50)
    parser.add_argument("--editor-offset-threshold", type=float, default=0.50)
    parser.add_argument("--frames-per-second", type=int, default=100)
    parser.add_argument("--classes-num", type=int, default=88)
    parser.add_argument("--begin-note", type=int, default=21)
    parser.add_argument("--velocity-scale", type=int, default=128)
    parser.add_argument("--default-velocity", type=int, default=80)
    parser.add_argument("--export-midi", action="store_true")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--min-pitch", type=int, default=36)
    parser.add_argument("--max-pitch", type=int, default=90)
    parser.add_argument("--alpha", type=float, default=0.82)
    parser.add_argument("--line-width", type=float, default=0.8)
    return parser


def parse_thresholds(value: str) -> List[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if len(items) != len(VOICE_NAMES):
        raise ValueError(f"Expected {len(VOICE_NAMES)} thresholds, got {len(items)} from {value!r}")
    return [float(item) for item in items]


def create_post_processor(
    args: argparse.Namespace,
    frame_threshold: float,
    onset_threshold: float,
    offset_threshold: float,
):
    cfg = SimpleNamespace(
        feature=SimpleNamespace(
            frames_per_second=args.frames_per_second,
            classes_num=args.classes_num,
            begin_note=args.begin_note,
            velocity_scale=args.velocity_scale,
        ),
        post=SimpleNamespace(
            frame_threshold=frame_threshold,
            onset_threshold=onset_threshold,
            offset_threshold=offset_threshold,
            pedal_offset_threshold=0.2,
            default_velocity=args.default_velocity,
        ),
    )
    if args.post_processor_type.lower() == "regression":
        return RegressionPostProcessor(cfg)
    if args.post_processor_type.lower() in {"onsets_frames", "onf", "onset_frames"}:
        return OnsetsFramesPostProcessor(cfg)
    raise ValueError(f"Unsupported post processor: {args.post_processor_type}")


def pick_stem(probs_dir: Path, stem: str) -> str:
    if stem:
        target = probs_dir / f"{stem}.pkl"
        if not target.exists():
            raise FileNotFoundError(f"Missing probs file for stem {stem}: {target}")
        return stem

    candidates = sorted(path.stem for path in probs_dir.glob("*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No probs files found in {probs_dir}")
    return candidates[0]


def decode_merged_events(total_dict: Dict[str, np.ndarray], args: argparse.Namespace) -> List[Dict[str, float]]:
    frame_thresholds = parse_thresholds(args.voice_frame_thresholds)
    onset_thresholds = parse_thresholds(args.voice_onset_thresholds)
    offset_thresholds = parse_thresholds(args.voice_offset_thresholds)

    frame_threshold = args.merged_frame_threshold if args.merged_frame_threshold >= 0 else frame_thresholds[0]
    onset_threshold = args.merged_onset_threshold if args.merged_onset_threshold >= 0 else onset_thresholds[0]
    offset_threshold = args.merged_offset_threshold if args.merged_offset_threshold >= 0 else offset_thresholds[0]

    post_processor = create_post_processor(
        args=args,
        frame_threshold=frame_threshold,
        onset_threshold=onset_threshold,
        offset_threshold=offset_threshold,
    )
    post_input = {}
    for key in ("frame_output", "onset_output", "offset_output", "velocity_output"):
        if key in total_dict:
            post_input[key] = np.array(total_dict[key], copy=True)
    note_events, _ = post_processor.output_dict_to_midi_events(post_input)
    return note_events


def decode_direct_voice_tracks(total_dict: Dict[str, np.ndarray], args: argparse.Namespace) -> Dict[str, List[Dict[str, float]]]:
    frame_thresholds = parse_thresholds(args.voice_frame_thresholds)
    onset_thresholds = parse_thresholds(args.voice_onset_thresholds)
    offset_thresholds = parse_thresholds(args.voice_offset_thresholds)

    tracks = {}
    for voice_idx, voice_name in enumerate(VOICE_NAMES):
        post_processor = create_post_processor(
            args=args,
            frame_threshold=frame_thresholds[voice_idx],
            onset_threshold=onset_thresholds[voice_idx],
            offset_threshold=offset_thresholds[voice_idx],
        )
        post_input = {
            "frame_output": np.array(total_dict["voice_frame_output"][:, voice_idx, :], copy=True),
            "onset_output": np.array(total_dict["voice_onset_output"][:, voice_idx, :], copy=True),
        }
        if "voice_offset_output" in total_dict:
            post_input["offset_output"] = np.array(total_dict["voice_offset_output"][:, voice_idx, :], copy=True)
        note_events, _ = post_processor.output_dict_to_midi_events(copy.deepcopy(post_input))
        tracks[voice_name] = note_events
    return tracks


def merged_note_events_to_editor_tracks(
    editor_checkpoint: Path,
    merged_note_events: Sequence[Dict[str, float]],
    args: argparse.Namespace,
) -> Dict[str, List[Dict[str, float]]]:
    checkpoint = torch.load(editor_checkpoint, map_location="cpu")
    saved_args = argparse.Namespace(**checkpoint["args"])
    device = resolve_device(args.device)
    model = load_model_from_checkpoint(checkpoint, device)

    input_rolls = build_input_rolls_from_note_events(
        note_events=merged_note_events,
        frames_per_second=int(saved_args.frames_per_second),
    )
    averaged_outputs = infer_song_rolls(
        model=model,
        input_rolls=input_rolls,
        device=device,
        segment_seconds=float(saved_args.segment_seconds),
        segment_stride_seconds=float(saved_args.segment_stride_seconds),
        frames_per_second=int(saved_args.frames_per_second),
    )
    return decode_satb_tracks(
        averaged_outputs=averaged_outputs,
        frames_per_second=int(saved_args.frames_per_second),
        frame_threshold=args.editor_frame_threshold,
        onset_threshold=args.editor_onset_threshold,
        offset_threshold=args.editor_offset_threshold,
    )


def collect_time_pitch_bounds_multi(
    tracks_list: Sequence[Dict[str, List[Dict[str, float]]]],
    min_pitch: int,
    max_pitch: int,
):
    all_events = []
    for tracks in tracks_list:
        for voice_name in VOICE_NAMES:
            all_events.extend(tracks.get(voice_name, []))
    if not all_events:
        return 0.0, 1.0, min_pitch, max_pitch

    max_time = max(event["offset_time"] for event in all_events)
    min_seen_pitch = min(event["midi_note"] for event in all_events)
    max_seen_pitch = max(event["midi_note"] for event in all_events)
    low_pitch = min(min_pitch, min_seen_pitch - 1)
    high_pitch = max(max_pitch, max_seen_pitch + 1)
    return 0.0, max_time, low_pitch, high_pitch


def plot_triplet(
    gt_tracks: Dict[str, List[Dict[str, float]]],
    editor_tracks: Dict[str, List[Dict[str, float]]],
    direct_tracks: Dict[str, List[Dict[str, float]]],
    output_path: Path,
    stem: str,
    editor_summary: Dict[str, float],
    direct_summary: Dict[str, float],
    args: argparse.Namespace,
) -> None:
    x0, x1, y0, y1 = collect_time_pitch_bounds_multi(
        [gt_tracks, editor_tracks, direct_tracks],
        args.min_pitch,
        args.max_pitch,
    )
    fig, axes = plt.subplots(3, 1, figsize=(18, 11), sharex=True, constrained_layout=True)

    draw_voice_roll(
        axes[0],
        gt_tracks,
        title=f"Ground Truth: {stem}",
        alpha=args.alpha,
        line_width=args.line_width,
        xlim=(x0, x1),
        ylim=(y0, y1),
    )
    draw_voice_roll(
        axes[1],
        editor_tracks,
        title=(
            f"MIDI Editor Prediction: {stem} | "
            f"meanF1@50={editor_summary.get('mean_satb_note_f1_50ms', 0.0):.4f} | "
            f"meanF1@100={editor_summary.get('mean_satb_note_f1_100ms', 0.0):.4f}"
        ),
        alpha=args.alpha,
        line_width=args.line_width,
        xlim=(x0, x1),
        ylim=(y0, y1),
    )
    draw_voice_roll(
        axes[2],
        direct_tracks,
        title=(
            f"Direct Choral Decode: {stem} | "
            f"meanF1@50={direct_summary.get('mean_satb_note_f1_50ms', 0.0):.4f} | "
            f"meanF1@100={direct_summary.get('mean_satb_note_f1_100ms', 0.0):.4f}"
        ),
        alpha=args.alpha,
        line_width=args.line_width,
        xlim=(x0, x1),
        ylim=(y0, y1),
    )
    axes[2].set_xlabel("Time (seconds)")
    legend_handles = [Patch(facecolor=VOICE_COLORS[v], edgecolor="black", label=v) for v in VOICE_NAMES]
    axes[0].legend(handles=legend_handles, loc="upper right", ncol=4, frameon=True)

    create_folder(output_path.parent)
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()

    probs_dir = Path(args.probs_dir)
    editor_checkpoint = Path(args.editor_checkpoint)
    dataset_dir = Path(args.dataset_dir)
    stem = pick_stem(probs_dir, args.stem)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.workspace) / "symbolic_satb_editor" / "choral_editor_compare" / stem
    create_folder(output_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(output_dir / "compare.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    prob_path = probs_dir / f"{stem}.pkl"
    with prob_path.open("rb") as f:
        total_dict = pickle.load(f)

    song_events = build_song_events(dataset_dir, stem, args.frames_per_second)
    gt_tracks = reference_voice_tracks(song_events)

    direct_tracks = decode_direct_voice_tracks(total_dict, args)
    direct_summary = compute_choral_note_summary(gt_tracks, direct_tracks)

    merged_note_events = decode_merged_events(total_dict, args)
    editor_tracks = merged_note_events_to_editor_tracks(
        editor_checkpoint=editor_checkpoint,
        merged_note_events=merged_note_events,
        args=args,
    )
    editor_summary = compute_choral_note_summary(gt_tracks, editor_tracks)

    if args.export_midi:
        create_folder(output_dir / "midis")
        write_satb_midi(gt_tracks, output_dir / "midis" / f"{stem}_gt.mid")
        write_satb_midi(editor_tracks, output_dir / "midis" / f"{stem}_editor.mid")
        write_satb_midi(direct_tracks, output_dir / "midis" / f"{stem}_direct.mid")
        write_events_to_midi(0, list(merged_note_events), None, str(output_dir / "midis" / f"{stem}_merged_input.mid"))

    plot_triplet(
        gt_tracks=gt_tracks,
        editor_tracks=editor_tracks,
        direct_tracks=direct_tracks,
        output_path=output_dir / f"{stem}_triplet.png",
        stem=stem,
        editor_summary=editor_summary,
        direct_summary=direct_summary,
        args=args,
    )

    report = {
        "stem": stem,
        "editor_checkpoint": str(editor_checkpoint),
        "probs_dir": str(probs_dir),
        "post_processor_type": args.post_processor_type,
        "voice_frame_thresholds": parse_thresholds(args.voice_frame_thresholds),
        "voice_onset_thresholds": parse_thresholds(args.voice_onset_thresholds),
        "voice_offset_thresholds": parse_thresholds(args.voice_offset_thresholds),
        "editor_frame_threshold": args.editor_frame_threshold,
        "editor_onset_threshold": args.editor_onset_threshold,
        "editor_offset_threshold": args.editor_offset_threshold,
        "merged_num_notes": len(merged_note_events),
        "direct": direct_summary,
        "editor": editor_summary,
    }
    with (output_dir / "report.json").open("w") as f:
        json.dump(report, f, indent=2)

    logging.info("Stem: %s", stem)
    logging.info(
        "Direct decode | mean_F1_50ms=%.4f | mean_F1_100ms=%.4f",
        direct_summary.get("mean_satb_note_f1_50ms", 0.0),
        direct_summary.get("mean_satb_note_f1_100ms", 0.0),
    )
    logging.info(
        "MIDI editor | mean_F1_50ms=%.4f | mean_F1_100ms=%.4f",
        editor_summary.get("mean_satb_note_f1_50ms", 0.0),
        editor_summary.get("mean_satb_note_f1_100ms", 0.0),
    )
    logging.info("Saved visualization to %s", output_dir / f"{stem}_triplet.png")


if __name__ == "__main__":
    main()
