from __future__ import annotations

import argparse
from pathlib import Path

import mido


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Postprocess predicted MIDIs before VA evaluation.",
    )
    parser.add_argument("--src-dir", type=str, required=True)
    parser.add_argument("--dst-dir", type=str, required=True)
    parser.add_argument("--min-duration-sec", type=float, default=0.04)
    parser.add_argument("--dedup-onset-sec", type=float, default=0.05)
    parser.add_argument("--min-pitch", type=int, default=36)
    parser.add_argument("--max-pitch", type=int, default=90)
    parser.add_argument("--program", type=int, default=0)
    return parser


def ticks_to_seconds(mid: mido.MidiFile, abs_ticks: int, tempo: int) -> float:
    return mido.tick2second(abs_ticks, mid.ticks_per_beat, tempo)


def seconds_to_ticks(mid: mido.MidiFile, sec: float, tempo: int) -> int:
    return int(round(mido.second2tick(sec, mid.ticks_per_beat, tempo)))


def extract_notes(mid: mido.MidiFile):
    tempo = 500000
    notes = []
    active = {}
    for track in mid.tracks:
        abs_ticks = 0
        for msg in track:
            abs_ticks += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "note_on" and msg.velocity > 0:
                active.setdefault((msg.channel, msg.note), []).append((abs_ticks, msg.velocity))
            elif msg.type in {"note_off", "note_on"} and (msg.type == "note_off" or msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active and active[key]:
                    start_tick, vel = active[key].pop()
                    notes.append(
                        {
                            "channel": msg.channel,
                            "pitch": int(msg.note),
                            "velocity": int(vel),
                            "start_sec": ticks_to_seconds(mid, start_tick, tempo),
                            "end_sec": ticks_to_seconds(mid, abs_ticks, tempo),
                        }
                    )
    return notes


def dedup_and_filter(notes, min_duration_sec: float, dedup_onset_sec: float, min_pitch: int, max_pitch: int):
    cleaned = []
    for n in notes:
        if not (min_pitch <= n["pitch"] <= max_pitch):
            continue
        if n["end_sec"] <= n["start_sec"]:
            continue
        if (n["end_sec"] - n["start_sec"]) < min_duration_sec:
            continue
        cleaned.append(n)

    # Deduplicate by pitch within close onsets; keep the longer note.
    cleaned.sort(key=lambda x: (x["pitch"], x["start_sec"], x["end_sec"]))
    out = []
    i = 0
    while i < len(cleaned):
        cur = cleaned[i]
        j = i + 1
        best = cur
        while j < len(cleaned):
            nxt = cleaned[j]
            if nxt["pitch"] != cur["pitch"]:
                break
            if abs(nxt["start_sec"] - cur["start_sec"]) > dedup_onset_sec:
                break
            if (nxt["end_sec"] - nxt["start_sec"]) > (best["end_sec"] - best["start_sec"]):
                best = nxt
            j += 1
        out.append(best)
        i = j
    out.sort(key=lambda x: (x["start_sec"], x["pitch"], x["end_sec"]))
    return out


def write_midi(mid: mido.MidiFile, notes, dst_path: Path, program: int):
    tempo = 500000
    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    tr = mido.MidiTrack()
    out.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    tr.append(mido.Message("program_change", program=program, channel=0, time=0))

    events = []
    for n in notes:
        st = seconds_to_ticks(out, n["start_sec"], tempo)
        ed = seconds_to_ticks(out, n["end_sec"], tempo)
        ed = max(ed, st + 1)
        events.append((st, mido.Message("note_on", note=n["pitch"], velocity=max(1, n["velocity"]), channel=0, time=0)))
        events.append((ed, mido.Message("note_off", note=n["pitch"], velocity=0, channel=0, time=0)))
    events.sort(key=lambda x: (x[0], 0 if x[1].type == "note_off" else 1, x[1].note))

    prev = 0
    for tick, msg in events:
        dt = max(0, tick - prev)
        msg.time = dt
        tr.append(msg)
        prev = tick

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(dst_path))


def main() -> None:
    args = build_arg_parser().parse_args()
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.mid"))
    if not files:
        raise FileNotFoundError(f"No MIDI files found in {src_dir}")

    kept_notes = 0
    raw_notes = 0
    for p in files:
        mid = mido.MidiFile(str(p))
        raw = extract_notes(mid)
        fixed = dedup_and_filter(
            raw,
            min_duration_sec=float(args.min_duration_sec),
            dedup_onset_sec=float(args.dedup_onset_sec),
            min_pitch=int(args.min_pitch),
            max_pitch=int(args.max_pitch),
        )
        write_midi(mid, fixed, dst_dir / p.name, program=int(args.program))
        raw_notes += len(raw)
        kept_notes += len(fixed)

    ratio = (kept_notes / raw_notes) if raw_notes > 0 else 0.0
    print(f"Processed {len(files)} files")
    print(f"Raw notes: {raw_notes} | Kept notes: {kept_notes} | Keep ratio: {ratio:.4f}")
    print(f"Saved to: {dst_dir}")


if __name__ == "__main__":
    main()
