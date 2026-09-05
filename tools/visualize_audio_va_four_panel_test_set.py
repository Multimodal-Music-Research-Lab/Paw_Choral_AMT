from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_midi_voice_assignment import VOICE_NAMES
from visualize_choral_checkpoint_single_song import (
    apply_axis_style,
    compute_mean_satb_note_f1,
    compute_merged_note_f1_from_events,
    create_folder,
    decode_direct_voice_tracks,
    decode_merged_events,
    draw_choral_roll,
    draw_global_roll,
    load_note_bars,
    reference_voice_tracks_from_note_bars,
    resolve_pitch_ylim,
)
from visualize_choral_four_panel_single_song import (
    build_global_va_tracks,
    build_arg_parser as build_single_arg_parser,
    crop_note_events,
    crop_voice_tracks,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = build_single_arg_parser()
    parser.description = "Batch four-panel visualization from audio-conditioned eval outputs."
    parser.set_defaults(dataset_dir="./data/YouChorale")
    # This batch script resolves checkpoints from eval-dir/report.json, so these
    # inherited required flags should not block CLI execution.
    for action in parser._actions:
        if action.dest in {"checkpoint_path", "global_checkpoint_path", "va_checkpoint"}:
            action.required = False
            if action.default is None:
                action.default = ""
    parser.add_argument(
        "--eval-dir",
        type=str,
        required=True,
        help="Directory containing report.json from evaluate_audio_midi_voice_assignment.py",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split JSON name.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of stems to render.")
    parser.add_argument("--overwrite", action="store_true", help="Re-render existing png files.")
    parser.add_argument(
        "--part-aware-probs-dir",
        type=str,
        default="",
        help="Optional probs directory for part-aware panel.",
    )
    parser.add_argument(
        "--hide-stem-title",
        action="store_true",
        help="Do not render the song stem title above the figure.",
    )
    parser.add_argument("--gt-title", type=str, default="Ground Truth",
                        help="Title for the GT panel (top-left).")
    parser.add_argument("--global-title-prefix", type=str, default="PCT w/o VA",
                        help="Text shown before '| note F1@50ms=...' on the global panel.")
    parser.add_argument("--part-aware-title-prefix", type=str, default="PCT-OC",
                        help="Text shown before '| mean SATB F1@50ms=...' on the part-aware panel.")
    parser.add_argument("--va-title-prefix", type=str, default="Post-VA + PCT w/o VA",
                        help="Text shown before '| mean SATB F1@50ms=...' on the VA panel.")
    return parser


def load_split_stems(dataset_dir: Path, split: str) -> List[str]:
    split_path = dataset_dir / f"{split}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    with split_path.open("r", encoding="utf-8") as f:
        stems = json.load(f)
    if not isinstance(stems, list):
        raise ValueError(f"Split JSON must contain a list of stems: {split_path}")
    return [str(stem) for stem in stems]


def compute_per_voice_note_f1(
    ref_tracks: Dict[str, Sequence[Dict[str, float]]],
    pred_tracks: Dict[str, Sequence[Dict[str, float]]],
    onset_tolerance: float = 0.05,
    offset_min_tolerance: float = 0.05,
) -> Dict[str, float]:
    from visualize_choral_checkpoint_single_song import compute_track_note_f1

    scores = {}
    for voice_name in VOICE_NAMES:
        scores[voice_name] = float(
            compute_track_note_f1(
                ref_events=ref_tracks.get(voice_name, []),
                pred_events=pred_tracks.get(voice_name, []),
                onset_tolerance=onset_tolerance,
                offset_min_tolerance=offset_min_tolerance,
            )
        )
    return scores


def plot_four_panel_from_audio_eval(
    gt_tracks: Dict[str, List[Dict[str, float]]],
    global_events: Sequence[Dict[str, float]],
    part_aware_tracks: Dict[str, List[Dict[str, float]]],
    va_tracks: Dict[str, List[Dict[str, float]]],
    global_note_f1: float,
    part_aware_note_f1: float,
    va_note_f1: float,
    stem: str,
    output_path: Path,
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
        figsize=(18.0, 6.8),
        sharex=False,
        sharey=False,
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

    draw_global_roll(
        axes[0, 1],
        global_events,
        title="PCT w/o VA",
        alpha=args.alpha,
        line_width=args.line_width,
        xlim=xlim,
        ylim=ylim,
    )
    apply_axis_style(
        axes[0, 1],
        title=f"{args.global_title_prefix} | note F1@50ms={global_note_f1:.4f}",
        ylabel="MIDI Pitch",
    )

    draw_choral_roll(
        axes[1, 0],
        part_aware_tracks,
        alpha=args.alpha,
        line_width=args.line_width,
        xlim=xlim,
        ylim=ylim,
    )
    apply_axis_style(
        axes[1, 0],
        title=f"{args.part_aware_title_prefix} | mean SATB F1@50ms={part_aware_note_f1:.4f}",
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
        title=f"{args.va_title_prefix} | mean SATB F1@50ms={va_note_f1:.4f}",
        xlabel="Time (seconds)",
        ylabel="MIDI Pitch",
    )

    if not bool(getattr(args, "hide_stem_title", False)):
        fig.suptitle(stem, fontsize=16, y=1.01)
    create_folder(output_path.parent)
    fig.savefig(output_path, dpi=args.dpi, transparent=bool(getattr(args, "transparent", False)))
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    eval_dir = Path(args.eval_dir)
    report_path = eval_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report.json: {report_path}")
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    args.va_checkpoint = str(report["va_checkpoint"])
    args.probs_dir = str(report["probs_dir"])
    # Keep decoding thresholds from CLI/default arguments so visualization is reproducible
    # under user-selected threshold settings (do not force report.json thresholds here).

    dataset_dir = Path(args.dataset_dir)
    probs_dir = Path(args.probs_dir)
    part_aware_probs_dir = Path(args.part_aware_probs_dir) if args.part_aware_probs_dir else None
    va_checkpoint_path = Path(args.va_checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else (eval_dir / "four_panel")
    create_folder(output_dir)

    stems = load_split_stems(dataset_dir=dataset_dir, split=args.split)
    if args.limit and args.limit > 0:
        stems = stems[: args.limit]
    if args.stem:
        stems = [args.stem]

    total = len(stems)
    if total == 0:
        raise ValueError("No stems selected for rendering.")

    for index, stem in enumerate(stems, start=1):
        output_path = output_dir / f"{stem}_four_panel.png"
        if output_path.exists() and not args.overwrite:
            logging.info("[%d/%d] %s | skip existing %s", index, total, stem, output_path)
            continue

        note_path = dataset_dir / "note" / f"{stem}.pkl"
        prob_path = probs_dir / f"{stem}.pkl"
        part_aware_prob_path = (part_aware_probs_dir / f"{stem}.pkl") if part_aware_probs_dir else None
        if not note_path.exists():
            logging.warning("[%d/%d] %s | missing GT note file %s", index, total, stem, note_path)
            continue
        if not prob_path.exists():
            logging.warning("[%d/%d] %s | missing probs file %s", index, total, stem, prob_path)
            continue
        if part_aware_prob_path is not None and (not part_aware_prob_path.exists()):
            logging.warning("[%d/%d] %s | missing part-aware probs file %s", index, total, stem, part_aware_prob_path)
            continue

        with prob_path.open("rb") as f:
            total_dict = pickle.load(f)
        part_aware_dict = None
        if part_aware_prob_path is not None:
            with part_aware_prob_path.open("rb") as f:
                part_aware_dict = pickle.load(f)
        note_bars = load_note_bars(note_path)
        gt_tracks = reference_voice_tracks_from_note_bars(note_bars)
        global_events = decode_merged_events(total_dict, args)
        part_aware_tracks = (
            decode_direct_voice_tracks(part_aware_dict, args) if part_aware_dict is not None else {voice: [] for voice in VOICE_NAMES}
        )
        va_tracks = build_global_va_tracks(global_events, va_checkpoint_path=va_checkpoint_path, args=args)
        global_note_f1 = compute_merged_note_f1_from_events(
            ref_on_off_pairs=total_dict["ref_on_off_pairs"],
            ref_midi_notes=total_dict["ref_midi_notes"],
            est_events=global_events,
            onset_tolerance=0.05,
            offset_min_tolerance=0.05,
        )
        _ = compute_per_voice_note_f1(
            ref_tracks=gt_tracks,
            pred_tracks=va_tracks,
            onset_tolerance=0.05,
            offset_min_tolerance=0.05,
        )
        va_note_f1 = compute_mean_satb_note_f1(
            ref_tracks=gt_tracks,
            pred_tracks=va_tracks,
            onset_tolerance=0.05,
            offset_min_tolerance=0.05,
        )
        part_aware_note_f1 = compute_mean_satb_note_f1(
            ref_tracks=gt_tracks,
            pred_tracks=part_aware_tracks,
            onset_tolerance=0.05,
            offset_min_tolerance=0.05,
        )

        plot_four_panel_from_audio_eval(
            gt_tracks=gt_tracks,
            global_events=global_events,
            part_aware_tracks=part_aware_tracks,
            va_tracks=va_tracks,
            global_note_f1=global_note_f1,
            part_aware_note_f1=part_aware_note_f1,
            va_note_f1=va_note_f1,
            stem=stem,
            output_path=output_path,
            args=args,
        )
        logging.info("[%d/%d] %s | saved %s", index, total, stem, output_path)


if __name__ == "__main__":
    main()
