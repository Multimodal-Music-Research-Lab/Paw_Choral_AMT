from __future__ import annotations

import argparse
import logging
import pickle
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mir_eval
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle

from utilities import OnsetsFramesPostProcessor, RegressionPostProcessor, note_to_freq
from visualize_midi_voice_assignment import VOICE_COLORS, VOICE_NAMES, create_folder

GLOBAL_COLOR = "#4f772d"
VOICE_DISPLAY_NAMES = {
    "S": "Soprano",
    "A": "Alto",
    "T": "Tenor",
    "B": "Bass",
}


def resolve_plot_font_family() -> str:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for family in ("Times New Roman", "Nimbus Roman", "Liberation Serif"):
        if family in available_fonts:
            return family
    return "serif"


PLOT_FONT_FAMILY = resolve_plot_font_family()
TITLE_FONT_SIZE = 22
LABEL_FONT_SIZE = 22
TICK_FONT_SIZE = 18
PLOT_YLIM = (40, 88)
FIGSIZE_THREE_PANEL = (11.8, 8.2)
FIGSIZE_TWO_PANEL = (11.8, 5.9)
GRID_COLOR = "#7a7a7a"
GRID_ALPHA = 0.28
GRID_LINE_WIDTH = 1.0
RECT_EDGE_WIDTH = 0.25

matplotlib.rcParams["font.family"] = PLOT_FONT_FAMILY
matplotlib.rcParams["font.serif"] = [PLOT_FONT_FAMILY]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize one choral checkpoint song with GT on top and prediction below.",
    )
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to *_iteration.pth checkpoint.")
    parser.add_argument(
        "--global-checkpoint-path",
        type=str,
        default="",
        help="Optional checkpoint for middle global-transcription panel.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./data/YouChorale",
        help="Dataset directory containing note/*.pkl for GT.",
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
    parser.add_argument(
        "--legend-output-path",
        type=str,
        default="",
        help="Optional standalone legend image path.",
    )
    parser.add_argument("--post-processor-type", type=str, default="onsets_frames")
    parser.add_argument("--frame-threshold", type=float, default=0.05)
    parser.add_argument("--onset-threshold", type=float, default=0.01)
    parser.add_argument("--offset-threshold", type=float, default=0.01)
    parser.add_argument("--frames-per-second", type=int, default=100)
    parser.add_argument("--classes-num", type=int, default=88)
    parser.add_argument("--begin-note", type=int, default=21)
    parser.add_argument("--velocity-scale", type=int, default=128)
    parser.add_argument("--default-velocity", type=int, default=80)
    parser.add_argument("--onset-tolerance", type=float, default=0.05)
    parser.add_argument("--offset-min-tolerance", type=float, default=0.05)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--min-pitch", type=int, default=36)
    parser.add_argument("--max-pitch", type=int, default=90)
    parser.add_argument("--alpha", type=float, default=0.82)
    parser.add_argument("--line-width", type=float, default=0.8)
    parser.add_argument("--transparent", action="store_true", help="Save figure with transparent background.")
    return parser


def parse_checkpoint_info(checkpoint_path: str | Path) -> tuple[str, int]:
    checkpoint_path = Path(checkpoint_path)
    match = re.match(r"^(\d+)_iteration\.pth$", checkpoint_path.name)
    if not match:
        raise ValueError(f"Checkpoint must look like 299999_iteration.pth: {checkpoint_path}")
    return checkpoint_path.parent.name, int(match.group(1))


def derive_workspace(checkpoint_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.parent.parent.name != "checkpoints":
        raise ValueError(
            f"Expected checkpoint under .../workspaces/checkpoints/<model>/<iter>_iteration.pth, got {checkpoint_path}"
        )
    return checkpoint_path.parent.parent.parent


def resolve_probs_dir(
    args: argparse.Namespace,
    model_name: str,
    iteration: int,
    checkpoint_path: str | Path | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else Path(args.checkpoint_path)
    workspace = Path(args.workspace) if args.workspace else derive_workspace(checkpoint_path)
    prob_split = getattr(args, "prob_split", "test")
    probs_dir = workspace / "probs" / args.test_set / prob_split / model_name / f"{iteration}_iteration"
    if not probs_dir.is_dir():
        raise FileNotFoundError(
            f"Missing probs dir: {probs_dir}\nRun inference first for checkpoint {iteration}."
        )
    return probs_dir


def pick_stem(probs_dir: Path, stem: str) -> str:
    if stem:
        prob_path = probs_dir / f"{stem}.pkl"
        if not prob_path.exists():
            raise FileNotFoundError(f"Missing probs file for stem {stem}: {prob_path}")
        return stem

    candidates = sorted(path.stem for path in probs_dir.glob("*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No probs files found in {probs_dir}")
    return candidates[0]


def load_note_bars(note_path: Path):
    with note_path.open("rb") as f:
        return pickle.load(f)


def reference_voice_tracks_from_note_bars(note_bars) -> Dict[str, List[Dict[str, float]]]:
    tracks = {voice_name: [] for voice_name in VOICE_NAMES}
    for bar in note_bars:
        if not isinstance(bar, dict):
            continue
        for part_name, note_list in bar.items():
            if part_name == "measure" or not isinstance(note_list, list) or not part_name:
                continue
            voice_name = part_name[0].upper()
            if voice_name not in tracks:
                continue
            for note in note_list:
                if len(note) < 5:
                    continue
                onset_time = float(note[3])
                offset_time = float(note[4])
                if offset_time <= onset_time:
                    offset_time = onset_time + 1e-4
                tracks[voice_name].append(
                    {
                        "midi_note": int(note[0]),
                        "onset_time": onset_time,
                        "offset_time": offset_time,
                        "velocity": 100,
                    }
                )

    for voice_name in VOICE_NAMES:
        tracks[voice_name].sort(key=lambda x: (x["onset_time"], x["midi_note"], x["offset_time"]))
    return tracks


def sanitize_intervals(intervals: np.ndarray) -> np.ndarray:
    intervals = np.asarray(intervals, dtype=np.float32)
    if intervals.size == 0:
        return intervals.reshape(0, 2)
    bad = intervals[:, 1] <= intervals[:, 0]
    if np.any(bad):
        intervals = intervals.copy()
        intervals[bad, 1] = intervals[bad, 0] + 1e-4
    return intervals


def events_to_intervals_and_pitches(events: Sequence[Dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    if len(events) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    intervals = sanitize_intervals([[event["onset_time"], event["offset_time"]] for event in events])
    pitches = np.asarray([event["midi_note"] for event in events], dtype=np.int32)
    order = np.argsort(intervals[:, 0], kind="mergesort")
    return intervals[order], pitches[order]


def compute_track_note_f1(
    ref_events: Sequence[Dict[str, float]],
    pred_events: Sequence[Dict[str, float]],
    onset_tolerance: float,
    offset_min_tolerance: float,
) -> float:
    ref_intervals, ref_pitches = events_to_intervals_and_pitches(ref_events)
    pred_intervals, pred_pitches = events_to_intervals_and_pitches(pred_events)
    _, _, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals=ref_intervals,
        ref_pitches=note_to_freq(ref_pitches),
        est_intervals=pred_intervals,
        est_pitches=note_to_freq(pred_pitches),
        onset_tolerance=onset_tolerance,
        offset_ratio=None,
        offset_min_tolerance=offset_min_tolerance,
    )
    return float(f1)


def compute_mean_satb_note_f1(
    ref_tracks: Dict[str, Sequence[Dict[str, float]]],
    pred_tracks: Dict[str, Sequence[Dict[str, float]]],
    onset_tolerance: float,
    offset_min_tolerance: float,
) -> float:
    values = [
        compute_track_note_f1(
            ref_events=ref_tracks[voice_name],
            pred_events=pred_tracks[voice_name],
            onset_tolerance=onset_tolerance,
            offset_min_tolerance=offset_min_tolerance,
        )
        for voice_name in VOICE_NAMES
    ]
    return float(np.mean(values)) if values else 0.0


def compute_merged_note_f1_from_events(
    ref_on_off_pairs: np.ndarray,
    ref_midi_notes: np.ndarray,
    est_events: Sequence[Dict[str, float]],
    onset_tolerance: float,
    offset_min_tolerance: float,
) -> float:
    ref_on_off_pairs = sanitize_intervals(ref_on_off_pairs)
    ref_midi_notes = np.asarray(ref_midi_notes, dtype=np.int32)
    est_intervals, est_pitches = events_to_intervals_and_pitches(est_events)
    _, _, note_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals=ref_on_off_pairs,
        ref_pitches=note_to_freq(ref_midi_notes),
        est_intervals=est_intervals,
        est_pitches=note_to_freq(est_pitches),
        onset_tolerance=onset_tolerance,
        offset_ratio=None,
        offset_min_tolerance=offset_min_tolerance,
    )
    return float(note_f1)


def create_post_processor(args: argparse.Namespace):
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
    post_type = args.post_processor_type.lower()
    if post_type == "regression":
        return RegressionPostProcessor(cfg)
    if post_type in {"onsets_frames", "onf", "onset_frames"}:
        return OnsetsFramesPostProcessor(cfg)
    raise ValueError(f"Unsupported post processor: {args.post_processor_type}")


def decode_direct_voice_tracks(total_dict: Dict[str, np.ndarray], args: argparse.Namespace) -> Dict[str, List[Dict[str, float]]]:
    required_keys = ("voice_frame_output", "voice_onset_output")
    for key in required_keys:
        if key not in total_dict:
            raise KeyError(f"Missing {key} in probs pkl; rerun inference if needed.")

    tracks = {}
    for voice_idx, voice_name in enumerate(VOICE_NAMES):
        post_processor = create_post_processor(args)
        post_input = {
            "frame_output": np.array(total_dict["voice_frame_output"][:, voice_idx, :], copy=True),
            "onset_output": np.array(total_dict["voice_onset_output"][:, voice_idx, :], copy=True),
        }
        if "voice_offset_output" in total_dict:
            post_input["offset_output"] = np.array(total_dict["voice_offset_output"][:, voice_idx, :], copy=True)
        note_events, _ = post_processor.output_dict_to_midi_events(post_input)
        tracks[voice_name] = note_events
    return tracks


def decode_merged_events(total_dict: Dict[str, np.ndarray], args: argparse.Namespace) -> List[Dict[str, float]]:
    post_processor = create_post_processor(args)
    post_input = {}
    for key in ("frame_output", "onset_output", "offset_output", "velocity_output"):
        if key in total_dict:
            post_input[key] = np.array(total_dict[key], copy=True)
    note_events, _ = post_processor.output_dict_to_midi_events(post_input)
    return note_events


def resolve_pitch_ylim() -> tuple[int, int]:
    return PLOT_YLIM


def draw_note_rectangles(
    ax,
    note_events: Sequence[Dict[str, float]],
    color: str,
    alpha: float,
    line_width: float,
) -> None:
    edge_width = max(RECT_EDGE_WIDTH, line_width * 0.35)
    for event in note_events:
        width = max(1e-4, event["offset_time"] - event["onset_time"])
        rect = Rectangle(
            (event["onset_time"], event["midi_note"] - 0.45),
            width,
            0.9,
            facecolor=color,
            edgecolor=color,
            linewidth=edge_width,
            alpha=alpha,
            joinstyle="round",
        )
        ax.add_patch(rect)


def draw_choral_roll(
    ax,
    voice_events: Dict[str, List[Dict[str, float]]],
    alpha: float,
    line_width: float,
    xlim,
    ylim,
) -> None:
    for voice_name in VOICE_NAMES:
        draw_note_rectangles(
            ax=ax,
            note_events=voice_events.get(voice_name, []),
            color=VOICE_COLORS[voice_name],
            alpha=alpha,
            line_width=line_width,
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, axis="x", linestyle="--", linewidth=GRID_LINE_WIDTH, alpha=GRID_ALPHA, color=GRID_COLOR)
    ax.set_axisbelow(True)


def draw_global_roll(
    ax,
    note_events: Sequence[Dict[str, float]],
    title: str,
    alpha: float,
    line_width: float,
    xlim,
    ylim,
    color: str = GLOBAL_COLOR,
) -> None:
    draw_note_rectangles(ax=ax, note_events=note_events, color=color, alpha=alpha, line_width=line_width)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, axis="x", linestyle="--", linewidth=GRID_LINE_WIDTH, alpha=GRID_ALPHA, color=GRID_COLOR)
    ax.set_axisbelow(True)


def apply_axis_style(ax, title: str, xlabel: str | None = None, ylabel: str = "MIDI Pitch") -> None:
    bottom, _ = ax.get_ylim()
    ax.set_ylim(bottom, 88)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontfamily=PLOT_FONT_FAMILY)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE, fontfamily=PLOT_FONT_FAMILY)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE, fontfamily=PLOT_FONT_FAMILY)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(PLOT_FONT_FAMILY)
        label.set_fontsize(TICK_FONT_SIZE)


def build_color_legend_handles() -> List[Patch]:
    handles = [
        Patch(facecolor=VOICE_COLORS[voice_name], edgecolor=VOICE_COLORS[voice_name], label=VOICE_DISPLAY_NAMES[voice_name])
        for voice_name in VOICE_NAMES
    ]
    handles.append(Patch(facecolor=GLOBAL_COLOR, edgecolor=GLOBAL_COLOR, label="Part-agnostic"))
    return handles


def save_color_legend(output_path: Path, dpi: int, transparent: bool) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 1.0))
    handles = build_color_legend_handles()
    legend = ax.legend(
        handles=handles,
        loc="center",
        ncol=len(handles),
        frameon=False,
        handlelength=1.4,
        columnspacing=1.3,
        handletextpad=0.5,
        fontsize=TICK_FONT_SIZE + 1,
    )
    for text in legend.get_texts():
        text.set_fontfamily(PLOT_FONT_FAMILY)
        text.set_fontsize(TICK_FONT_SIZE + 1)
    ax.axis("off")
    create_folder(output_path.parent)
    fig.savefig(output_path, dpi=dpi, transparent=transparent, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_pair(
    gt_tracks: Dict[str, List[Dict[str, float]]],
    pred_tracks: Dict[str, List[Dict[str, float]]],
    global_events: Sequence[Dict[str, float]] | None,
    output_path: Path,
    stem: str,
    args: argparse.Namespace,
) -> None:
    all_events = []
    for tracks in (gt_tracks, pred_tracks):
        for voice_name in VOICE_NAMES:
            all_events.extend(tracks.get(voice_name, []))
    if global_events is not None:
        all_events.extend(global_events)

    if all_events:
        max_time = max(event["offset_time"] for event in all_events)
        xlim = (0.0, max_time)
        ylim = resolve_pitch_ylim()
    else:
        xlim = (0.0, 1.0)
        ylim = resolve_pitch_ylim()

    if global_events is not None:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=FIGSIZE_THREE_PANEL,
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
    else:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=FIGSIZE_TWO_PANEL,
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
    draw_choral_roll(
        axes[0],
        gt_tracks,
        alpha=args.alpha,
        line_width=args.line_width,
        xlim=xlim,
        ylim=ylim,
    )
    apply_axis_style(axes[0], title="Ground Truth")
    pred_ax = axes[1]
    if global_events is not None:
        draw_global_roll(
            axes[1],
            global_events,
            title="Global Transcription",
            alpha=args.alpha,
            line_width=args.line_width,
            xlim=xlim,
            ylim=ylim,
        )
        apply_axis_style(axes[1], title="Global Transcription")
        pred_ax = axes[2]

    draw_choral_roll(
        pred_ax,
        pred_tracks,
        alpha=args.alpha,
        line_width=args.line_width,
        xlim=xlim,
        ylim=ylim,
    )
    apply_axis_style(pred_ax, title="Part-aware Transcription", xlabel="Time (seconds)")

    create_folder(output_path.parent)
    fig.savefig(output_path, dpi=args.dpi, transparent=bool(getattr(args, "transparent", False)))
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    model_name, iteration = parse_checkpoint_info(checkpoint_path)
    probs_dir = resolve_probs_dir(args, model_name=model_name, iteration=iteration)
    stem = pick_stem(probs_dir, args.stem)

    note_path = Path(args.dataset_dir) / "note" / f"{stem}.pkl"
    if not note_path.exists():
        raise FileNotFoundError(f"Missing GT note file: {note_path}")

    prob_path = probs_dir / f"{stem}.pkl"
    with prob_path.open("rb") as f:
        total_dict = pickle.load(f)

    global_events = None
    global_note_f1 = None
    if args.global_checkpoint_path:
        global_checkpoint_path = Path(args.global_checkpoint_path)
        global_model_name, global_iteration = parse_checkpoint_info(global_checkpoint_path)
        global_probs_dir = resolve_probs_dir(
            args,
            model_name=global_model_name,
            iteration=global_iteration,
            checkpoint_path=global_checkpoint_path,
        )
        global_prob_path = global_probs_dir / f"{stem}.pkl"
        if not global_prob_path.exists():
            raise FileNotFoundError(f"Missing global probs file for stem {stem}: {global_prob_path}")
        with global_prob_path.open("rb") as f:
            global_total_dict = pickle.load(f)
        global_events = decode_merged_events(global_total_dict, args)
        global_note_f1 = compute_merged_note_f1_from_events(
            ref_on_off_pairs=global_total_dict["ref_on_off_pairs"],
            ref_midi_notes=global_total_dict["ref_midi_notes"],
            est_events=global_events,
            onset_tolerance=args.onset_tolerance,
            offset_min_tolerance=args.offset_min_tolerance,
        )

    note_bars = load_note_bars(note_path)
    gt_tracks = reference_voice_tracks_from_note_bars(note_bars)
    pred_tracks = decode_direct_voice_tracks(total_dict, args)
    merged_events = decode_merged_events(total_dict, args)

    merged_note_f1 = compute_merged_note_f1_from_events(
        ref_on_off_pairs=total_dict["ref_on_off_pairs"],
        ref_midi_notes=total_dict["ref_midi_notes"],
        est_events=merged_events,
        onset_tolerance=args.onset_tolerance,
        offset_min_tolerance=args.offset_min_tolerance,
    )
    mean_satb_note_f1 = compute_mean_satb_note_f1(
        ref_tracks=gt_tracks,
        pred_tracks=pred_tracks,
        onset_tolerance=args.onset_tolerance,
        offset_min_tolerance=args.offset_min_tolerance,
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        workspace = Path(args.workspace) if args.workspace else derive_workspace(checkpoint_path)
        output_dir = workspace / "visualizations" / model_name / f"{iteration}_iteration"

    output_path = output_dir / f"{stem}_gt_vs_pred.png"
    create_folder(output_dir)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    plot_pair(
        gt_tracks=gt_tracks,
        pred_tracks=pred_tracks,
        global_events=global_events,
        output_path=output_path,
        stem=stem,
        args=args,
    )

    if args.legend_output_path:
        legend_output_path = Path(args.legend_output_path)
        save_color_legend(
            output_path=legend_output_path,
            dpi=args.dpi,
            transparent=bool(getattr(args, "transparent", False)),
        )

    logging.info("Stem: %s", stem)
    logging.info("Merged note-F1@50ms: %.4f", merged_note_f1)
    logging.info("Mean SATB note-F1@50ms: %.4f", mean_satb_note_f1)
    if global_note_f1 is not None:
        logging.info("Global note-F1@50ms: %.4f", global_note_f1)
    logging.info("Saved visualization to %s", output_path)
    if args.legend_output_path:
        logging.info("Saved legend to %s", args.legend_output_path)


if __name__ == "__main__":
    main()
