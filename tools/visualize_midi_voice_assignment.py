from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


VOICE_NAMES = ("S", "A", "T", "B")
VOICE_COLORS = {
    "S": "#d1495b",
    "A": "#edae49",
    "T": "#00798c",
    "B": "#30638e",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize reference and predicted SATB MIDI files with voice-colored piano rolls.",
    )
    parser.add_argument("--pred-midi", type=str, default="")
    parser.add_argument("--ref-midi", type=str, default="")
    parser.add_argument("--pred-dir", type=str, default="")
    parser.add_argument("--ref-dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--stem", type=str, default="")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--min-pitch", type=int, default=36)
    parser.add_argument("--max-pitch", type=int, default=90)
    parser.add_argument("--alpha", type=float, default=0.82)
    parser.add_argument("--line-width", type=float, default=0.8)
    return parser


def create_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ticks_to_seconds(ticks: int, ticks_per_beat: int, tempo: int) -> float:
    return ticks * tempo / 1e6 / ticks_per_beat


def parse_satb_midi(midi_path: Path) -> Dict[str, List[Dict[str, float]]]:
    try:
        from mido import MidiFile
    except Exception as exc:
        raise ImportError("mido is required to visualize MIDI files.") from exc

    midi = MidiFile(str(midi_path))
    ticks_per_beat = midi.ticks_per_beat
    tempo = 500000
    voice_events = {voice: [] for voice in VOICE_NAMES}

    for track in midi.tracks:
        current_ticks = 0
        current_seconds = 0.0
        track_name = ""
        active_notes: Dict[Tuple[int, int], Tuple[float, int]] = {}

        for msg in track:
            delta_ticks = msg.time
            current_ticks += delta_ticks
            current_seconds += ticks_to_seconds(delta_ticks, ticks_per_beat, tempo)

            if msg.type == "set_tempo":
                tempo = msg.tempo
                continue

            if msg.type == "track_name":
                track_name = msg.name.strip()
                continue

            if track_name not in VOICE_NAMES:
                continue

            if msg.type == "note_on" and msg.velocity > 0:
                active_notes[(msg.note, getattr(msg, "channel", 0))] = (current_seconds, int(msg.velocity))
            elif msg.type in {"note_off", "note_on"}:
                key = (msg.note, getattr(msg, "channel", 0))
                if key not in active_notes:
                    continue
                onset_time, velocity = active_notes.pop(key)
                if current_seconds <= onset_time:
                    continue
                voice_events[track_name].append(
                    {
                        "midi_note": int(msg.note),
                        "onset_time": float(onset_time),
                        "offset_time": float(current_seconds),
                        "velocity": int(velocity),
                    }
                )

    for voice_name in VOICE_NAMES:
        voice_events[voice_name].sort(key=lambda x: (x["onset_time"], x["midi_note"], x["offset_time"]))
    return voice_events


def infer_stem_name(pred_path: Path, ref_path: Path) -> str:
    for path in (pred_path, ref_path):
        stem = path.stem
        for suffix in ("_pred", "_ref"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem
    return "midi_comparison"


def collect_time_pitch_bounds(
    pred_events: Dict[str, List[Dict[str, float]]],
    ref_events: Dict[str, List[Dict[str, float]]],
    min_pitch: int,
    max_pitch: int,
) -> Tuple[float, float, int, int]:
    all_events = []
    for source in (pred_events, ref_events):
        for voice_name in VOICE_NAMES:
            all_events.extend(source.get(voice_name, []))

    if not all_events:
        return 0.0, 1.0, min_pitch, max_pitch

    max_time = max(event["offset_time"] for event in all_events)
    min_seen_pitch = min(event["midi_note"] for event in all_events)
    max_seen_pitch = max(event["midi_note"] for event in all_events)
    low_pitch = min(min_pitch, min_seen_pitch - 1)
    high_pitch = max(max_pitch, max_seen_pitch + 1)
    return 0.0, max_time, low_pitch, high_pitch


def draw_voice_roll(
    ax,
    voice_events: Dict[str, List[Dict[str, float]]],
    title: str,
    alpha: float,
    line_width: float,
    xlim: Tuple[float, float],
    ylim: Tuple[int, int],
) -> None:
    for voice_name in VOICE_NAMES:
        color = VOICE_COLORS[voice_name]
        for event in voice_events.get(voice_name, []):
            width = max(1e-4, event["offset_time"] - event["onset_time"])
            rect = Rectangle(
                (event["onset_time"], event["midi_note"] - 0.45),
                width,
                0.9,
                facecolor=color,
                edgecolor="black",
                linewidth=line_width,
                alpha=alpha,
            )
            ax.add_patch(rect)

    ax.set_title(title, fontsize=13)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_ylabel("MIDI Pitch")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)


def plot_pair(
    pred_path: Path,
    ref_path: Path,
    output_path: Path,
    min_pitch: int,
    max_pitch: int,
    alpha: float,
    line_width: float,
    dpi: int,
) -> None:
    pred_events = parse_satb_midi(pred_path)
    ref_events = parse_satb_midi(ref_path)
    x0, x1, y0, y1 = collect_time_pitch_bounds(pred_events, ref_events, min_pitch, max_pitch)

    stem_name = infer_stem_name(pred_path, ref_path)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(18, 8),
        sharex=True,
        constrained_layout=True,
    )

    draw_voice_roll(
        axes[0],
        ref_events,
        title=f"Ground Truth: {stem_name}",
        alpha=alpha,
        line_width=line_width,
        xlim=(x0, x1),
        ylim=(y0, y1),
    )
    draw_voice_roll(
        axes[1],
        pred_events,
        title=f"Prediction: {stem_name}",
        alpha=alpha,
        line_width=line_width,
        xlim=(x0, x1),
        ylim=(y0, y1),
    )
    axes[1].set_xlabel("Time (seconds)")

    legend_handles = [Patch(facecolor=VOICE_COLORS[v], edgecolor="black", label=v) for v in VOICE_NAMES]
    axes[0].legend(handles=legend_handles, loc="upper right", ncol=4, frameon=True)

    create_folder(output_path.parent)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def collect_pairs_from_dirs(pred_dir: Path, ref_dir: Path, stem_filter: str) -> List[Tuple[Path, Path]]:
    pairs = []
    for pred_path in sorted(pred_dir.glob("*_pred.mid")):
        stem = pred_path.stem[:-5]
        if stem_filter and stem != stem_filter:
            continue
        ref_path = ref_dir / f"{stem}_ref.mid"
        if ref_path.exists():
            pairs.append((pred_path, ref_path))
    return pairs


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_logging()

    if args.pred_midi and args.ref_midi:
        pred_path = Path(args.pred_midi)
        ref_path = Path(args.ref_midi)
        output_dir = Path(args.output_dir) if args.output_dir else pred_path.parent / "visualizations"
        output_path = output_dir / f"{infer_stem_name(pred_path, ref_path)}.png"
        plot_pair(
            pred_path=pred_path,
            ref_path=ref_path,
            output_path=output_path,
            min_pitch=args.min_pitch,
            max_pitch=args.max_pitch,
            alpha=args.alpha,
            line_width=args.line_width,
            dpi=args.dpi,
        )
        logging.info("Saved visualization to %s", output_path)
        return

    if args.pred_dir and args.ref_dir:
        pred_dir = Path(args.pred_dir)
        ref_dir = Path(args.ref_dir)
        output_dir = Path(args.output_dir) if args.output_dir else pred_dir.parent / "visualizations"
        create_folder(output_dir)
        pairs = collect_pairs_from_dirs(pred_dir, ref_dir, args.stem)
        if not pairs:
            raise FileNotFoundError(f"No matching *_pred.mid / *_ref.mid pairs found in {pred_dir} and {ref_dir}")

        for pred_path, ref_path in pairs:
            output_path = output_dir / f"{infer_stem_name(pred_path, ref_path)}.png"
            plot_pair(
                pred_path=pred_path,
                ref_path=ref_path,
                output_path=output_path,
                min_pitch=args.min_pitch,
                max_pitch=args.max_pitch,
                alpha=args.alpha,
                line_width=args.line_width,
                dpi=args.dpi,
            )
            logging.info("Saved visualization to %s", output_path)
        return

    raise ValueError("Provide either --pred-midi/--ref-midi or --pred-dir/--ref-dir.")


if __name__ == "__main__":
    main()
