from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

from visualize_choral_checkpoint_single_song import (
    compute_mean_satb_note_f1,
    compute_merged_note_f1_from_events,
    create_folder,
    decode_direct_voice_tracks,
    decode_merged_events,
    derive_workspace,
    load_note_bars,
    parse_checkpoint_info,
    reference_voice_tracks_from_note_bars,
    resolve_probs_dir,
)
from visualize_choral_four_panel_single_song import (
    build_arg_parser as build_single_arg_parser,
    build_global_va_tracks,
    plot_four_panel,
)


def build_arg_parser():
    parser = build_single_arg_parser()
    parser.description = "Batch-render GT, global, part-aware, and global+BLSTM-VA transcriptions."
    parser.add_argument("--split", type=str, default="test", help="Dataset split JSON to render, e.g. test or valid.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of stems to render.")
    parser.add_argument("--overwrite", action="store_true", help="Re-render figures even if output PNGs already exist.")
    return parser


def load_split_stems(dataset_dir: Path, split: str) -> list[str]:
    split_path = dataset_dir / f"{split}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    with split_path.open("r", encoding="utf-8") as f:
        stems = json.load(f)
    if not isinstance(stems, list):
        raise ValueError(f"Split JSON must contain a list of stems: {split_path}")
    return [str(stem) for stem in stems]


def resolve_output_dir(args, checkpoint_path: Path, model_name: str, iteration: int) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    workspace = Path(args.workspace) if args.workspace else derive_workspace(checkpoint_path)
    return workspace / "visualizations" / model_name / f"{iteration}_iteration"


def resolve_stems(args, dataset_dir: Path) -> list[str]:
    if args.stem:
        return [args.stem]
    stems = load_split_stems(dataset_dir=dataset_dir, split=args.split)
    if args.limit and args.limit > 0:
        stems = stems[: args.limit]
    return stems


def render_stem(
    stem: str,
    args,
    dataset_dir: Path,
    probs_dir: Path,
    global_probs_dir: Path,
    va_checkpoint_path: Path,
    output_dir: Path,
) -> Path:
    note_path = dataset_dir / "note" / f"{stem}.pkl"
    prob_path = probs_dir / f"{stem}.pkl"
    global_prob_path = global_probs_dir / f"{stem}.pkl"

    if not note_path.exists():
        raise FileNotFoundError(f"Missing GT note file: {note_path}")
    if not prob_path.exists():
        raise FileNotFoundError(f"Missing part-aware probs file for stem {stem}: {prob_path}")
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
    return output_path


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dataset_dir = Path(args.dataset_dir)
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
    output_dir = resolve_output_dir(args=args, checkpoint_path=checkpoint_path, model_name=model_name, iteration=iteration)
    create_folder(output_dir)

    stems = resolve_stems(args=args, dataset_dir=dataset_dir)
    total = len(stems)
    if total == 0:
        raise ValueError("No stems selected for rendering.")

    for index, stem in enumerate(stems, start=1):
        output_path = output_dir / f"{stem}_four_panel.png"
        if output_path.exists() and not args.overwrite:
            logging.info("[%d/%d] %s | skip existing %s", index, total, stem, output_path)
            continue
        output_path = render_stem(
            stem=stem,
            args=args,
            dataset_dir=dataset_dir,
            probs_dir=probs_dir,
            global_probs_dir=global_probs_dir,
            va_checkpoint_path=va_checkpoint_path,
            output_dir=output_dir,
        )
        logging.info("[%d/%d] %s | saved %s", index, total, stem, output_path)


if __name__ == "__main__":
    main()
