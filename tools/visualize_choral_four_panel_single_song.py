from __future__ import annotations
# Run from the repository root with PYTHONPATH=src:tools.

# python visualize_choral_four_panel_single_song.py \
#   --checkpoint-path        checkpoints/pawct-oc.pth \
#   --global-checkpoint-path checkpoints/pagct.pth \
#   --va-checkpoint          ./workspaces/midi_voice_assignment/youchorale_midi_va_bilstm/checkpoints/best_macro_f1.pth \
#   --test-set youchorale \
#   --stem jd2_r4PK5dc \
#   --post-processor-type regression \
#   --frame-threshold  0.03 \
#   --onset-threshold  0.01 \
#   --offset-threshold 0.005 \
#   --start-sec 10 --end-sec 100 \
#   --output-dir ./workspaces/visualizations/retest_pro_single_with_partaware_oc_299999_jd2_th003_001_0005_no_title \
#   --transparent

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_audio_midi_voice_assignment import predicted_note_events_to_input_events, voice_tracks_from_note_events
from infer_midi_voice_assignment import (
    build_model_from_checkpoint,
    infer_song,
    load_checkpoint_args,
    resolve_device,
)
from train_midi_voice_assignment import DEFAULT_RANGE_MAXS, DEFAULT_RANGE_MINS
from visualize_choral_checkpoint_single_song import (
    apply_axis_style,
    compute_mean_satb_note_f1,
    compute_merged_note_f1_from_events,
    create_folder,
    decode_direct_voice_tracks,
    decode_merged_events,
    derive_workspace,
    draw_choral_roll,
    draw_global_roll,
    load_note_bars,
    parse_checkpoint_info,
    pick_stem,
    reference_voice_tracks_from_note_bars,
    resolve_pitch_ylim,
    resolve_probs_dir,
    save_color_legend,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize GT, global, part-aware, and global+BLSTM-VA transcriptions for one song.",
    )
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Part-aware choral checkpoint path.")
    parser.add_argument("--global-checkpoint-path", type=str, required=True, help="Global checkpoint path.")
    parser.add_argument("--va-checkpoint", type=str, required=True, help="BLSTM MIDI voice-assignment checkpoint.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./data/YouChorale",
    )
    parser.add_argument("--workspace", type=str, default="", help="Optional workspace override.")
    parser.add_argument("--test-set", type=str, default="youchorale", help="Probs test set name.")
    parser.add_argument(
        "--prob-split",
        choices=("validation", "test"),
        default="test",
        help="Probability split produced by inference.py.",
    )
    parser.add_argument("--stem", type=str, default="", help="Song stem. If empty, use the first probs pkl.")
    parser.add_argument("--output-dir", type=str, default="", help="Optional output directory override.")
    parser.add_argument("--legend-output-path", type=str, default="", help="Optional standalone legend image path.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--post-processor-type", type=str, default="onsets_frames")
    parser.add_argument("--frame-threshold", type=float, default=0.05)
    parser.add_argument("--onset-threshold", type=float, default=0.01)
    parser.add_argument("--offset-threshold", type=float, default=0.01)
    parser.add_argument("--frames-per-second", type=int, default=100)
    parser.add_argument("--classes-num", type=int, default=88)
    parser.add_argument("--begin-note", type=int, default=21)
    parser.add_argument("--velocity-scale", type=int, default=128)
    parser.add_argument("--default-velocity", type=int, default=80)
    parser.add_argument("--segment-notes", type=int, default=0)
    parser.add_argument("--eval-stride", type=int, default=0)
    parser.add_argument("--onset-quantization-hz", type=int, default=0)
    parser.add_argument("--range-margin", type=float, default=-1.0)
    parser.add_argument("--apply-range-mask-at-infer", action="store_true")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=40.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--alpha", type=float, default=0.82)
    parser.add_argument("--line-width", type=float, default=0.8)
    parser.add_argument("--transparent", action="store_true", help="Save figure with transparent background.")
    return parser


def crop_note_events(
    note_events: Sequence[Dict[str, float]],
    start_sec: float,
    end_sec: float,
) -> List[Dict[str, float]]:
    cropped = []
    for event in note_events:
        onset_time = float(event["onset_time"])
        offset_time = float(event["offset_time"])
        if offset_time <= start_sec or onset_time >= end_sec:
            continue
        clipped_event = dict(event)
        clipped_event["onset_time"] = max(onset_time, start_sec)
        clipped_event["offset_time"] = min(offset_time, end_sec)
        if clipped_event["offset_time"] <= clipped_event["onset_time"]:
            continue
        cropped.append(clipped_event)
    return cropped


def crop_voice_tracks(
    voice_tracks: Dict[str, Sequence[Dict[str, float]]],
    start_sec: float,
    end_sec: float,
) -> Dict[str, List[Dict[str, float]]]:
    return {
        voice_name: crop_note_events(note_events, start_sec=start_sec, end_sec=end_sec)
        for voice_name, note_events in voice_tracks.items()
    }


def build_global_va_tracks(
    global_events: Sequence[Dict[str, float]],
    va_checkpoint_path: Path,
    args: argparse.Namespace,
) -> Dict[str, List[Dict[str, float]]]:
    checkpoint = load_checkpoint_args(va_checkpoint_path)
    saved_args = checkpoint["args"]
    segment_notes = int(args.segment_notes or saved_args["segment_notes"])
    eval_stride = int(args.eval_stride or saved_args.get("segment_stride", max(1, segment_notes // 2)))
    onset_quantization_hz = int(args.onset_quantization_hz or saved_args["onset_quantization_hz"])
    range_margin = float(args.range_margin if args.range_margin >= 0 else saved_args.get("range_margin", 2.0))

    input_events = predicted_note_events_to_input_events(
        stem="prediction",
        note_events=global_events,
        onset_quantization_hz=onset_quantization_hz,
    )
    if not input_events:
        return {voice_name: [] for voice_name in ("S", "A", "T", "B")}

    device = resolve_device(args.device)
    model = build_model_from_checkpoint(checkpoint, device)
    infer_dict = infer_song(
        model=model,
        events=input_events,
        device=device,
        segment_notes=segment_notes,
        eval_stride=eval_stride,
        range_mins=DEFAULT_RANGE_MINS.copy(),
        range_maxs=DEFAULT_RANGE_MAXS.copy(),
        range_margin=range_margin,
        apply_range_mask_at_infer=bool(args.apply_range_mask_at_infer),
    )
    return voice_tracks_from_note_events(input_events, infer_dict["pred_voice"].astype(np.int64))


def plot_four_panel(
    gt_tracks: Dict[str, List[Dict[str, float]]],
    global_events: Sequence[Dict[str, float]],
    part_aware_tracks: Dict[str, List[Dict[str, float]]],
    va_tracks: Dict[str, List[Dict[str, float]]],
    global_note_f1: float,
    part_aware_note_f1: float,
    va_note_f1: float,
    output_path: Path,
    stem: str,
    args: argparse.Namespace,
) -> None:
    xlim = (float(args.start_sec), float(args.end_sec))
    ylim = resolve_pitch_ylim()

    gt_tracks = crop_voice_tracks(gt_tracks, start_sec=xlim[0], end_sec=xlim[1])
    part_aware_tracks = crop_voice_tracks(part_aware_tracks, start_sec=xlim[0], end_sec=xlim[1])
    va_tracks = crop_voice_tracks(va_tracks, start_sec=xlim[0], end_sec=xlim[1])
    global_events = crop_note_events(global_events, start_sec=xlim[0], end_sec=xlim[1])

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18.0, 6.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes)

    draw_choral_roll(
        axes[0, 0],
        gt_tracks,
        alpha=args.alpha,
        line_width=args.line_width,
        xlim=xlim,
        ylim=ylim,
    )
    apply_axis_style(axes[0, 0], title="Ground Truth", ylabel="MIDI Pitch")

    draw_choral_roll(
        axes[0, 1],
        part_aware_tracks,
        alpha=args.alpha,
        line_width=args.line_width,
        xlim=xlim,
        ylim=ylim,
    )
    apply_axis_style(
        axes[0, 1],
        title=f"PawCT | note F1@50ms={part_aware_note_f1:.4f}",
        ylabel="MIDI Pitch",
    )

    draw_global_roll(
        axes[1, 0],
        global_events,
        title="PagCT",
        alpha=args.alpha,
        line_width=args.line_width,
        xlim=xlim,
        ylim=ylim,
    )
    apply_axis_style(
        axes[1, 0],
        title=f"PagCT | note F1@50ms={global_note_f1:.4f}",
        xlabel="Time (seconds)",
        ylabel="MIDI Pitch",
    )

    draw_choral_roll(
        axes[1, 1],
        va_tracks,
        alpha=args.alpha,
        line_width=args.line_width,
        xlim=xlim,
        ylim=ylim,
    )
    apply_axis_style(
        axes[1, 1],
        title=f"PagCT + Post-VA | note F1@50ms={va_note_f1:.4f}",
        xlabel="Time (seconds)",
        ylabel="MIDI Pitch",
    )

    create_folder(output_path.parent)
    fig.savefig(output_path, dpi=args.dpi, transparent=bool(getattr(args, "transparent", False)))
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    checkpoint_path = Path(args.checkpoint_path)
    global_checkpoint_path = Path(args.global_checkpoint_path)
    va_checkpoint_path = Path(args.va_checkpoint)

    model_name, iteration = parse_checkpoint_info(checkpoint_path)
    global_model_name, global_iteration = parse_checkpoint_info(global_checkpoint_path)

    probs_dir = resolve_probs_dir(args, model_name=model_name, iteration=iteration, checkpoint_path=checkpoint_path)
    global_probs_dir = resolve_probs_dir(
        args,
        model_name=global_model_name,
        iteration=global_iteration,
        checkpoint_path=global_checkpoint_path,
    )
    stem = pick_stem(probs_dir, args.stem)

    note_path = Path(args.dataset_dir) / "note" / f"{stem}.pkl"
    if not note_path.exists():
        raise FileNotFoundError(f"Missing GT note file: {note_path}")

    prob_path = probs_dir / f"{stem}.pkl"
    global_prob_path = global_probs_dir / f"{stem}.pkl"
    if not global_prob_path.exists():
        raise FileNotFoundError(f"Missing global probs file for stem {stem}: {global_prob_path}")

    with prob_path.open("rb") as f:
        total_dict = pickle.load(f)
    with global_prob_path.open("rb") as f:
        global_total_dict = pickle.load(f)

    note_bars = load_note_bars(note_path)
    gt_tracks = reference_voice_tracks_from_note_bars(note_bars)
    part_aware_tracks = decode_direct_voice_tracks(total_dict, args)
    global_events = decode_merged_events(global_total_dict, args)
    va_tracks = build_global_va_tracks(global_events, va_checkpoint_path=va_checkpoint_path, args=args)
    global_note_f1 = compute_merged_note_f1_from_events(
        ref_on_off_pairs=global_total_dict["ref_on_off_pairs"],
        ref_midi_notes=global_total_dict["ref_midi_notes"],
        est_events=global_events,
        onset_tolerance=0.05,
        offset_min_tolerance=0.05,
    )
    part_aware_note_f1 = compute_mean_satb_note_f1(
        ref_tracks=gt_tracks,
        pred_tracks=part_aware_tracks,
        onset_tolerance=0.05,
        offset_min_tolerance=0.05,
    )
    va_note_f1 = compute_mean_satb_note_f1(
        ref_tracks=gt_tracks,
        pred_tracks=va_tracks,
        onset_tolerance=0.05,
        offset_min_tolerance=0.05,
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        workspace = Path(args.workspace) if args.workspace else derive_workspace(checkpoint_path)
        output_dir = workspace / "visualizations" / model_name / f"{iteration}_iteration"

    output_path = output_dir / f"{stem}_four_panel.png"
    plot_four_panel(
        gt_tracks=gt_tracks,
        global_events=global_events,
        part_aware_tracks=part_aware_tracks,
        va_tracks=va_tracks,
        global_note_f1=global_note_f1,
        part_aware_note_f1=part_aware_note_f1,
        va_note_f1=va_note_f1,
        output_path=output_path,
        stem=stem,
        args=args,
    )

    if args.legend_output_path:
        save_color_legend(
            output_path=Path(args.legend_output_path),
            dpi=args.dpi,
            transparent=bool(getattr(args, "transparent", False)),
        )

    logging.info("Stem: %s", stem)
    logging.info("Saved visualization to %s", output_path)
    if args.legend_output_path:
        logging.info("Saved legend to %s", args.legend_output_path)


if __name__ == "__main__":
    main()
