from __future__ import annotations

import argparse
import json
import logging
import math
import zlib
from pathlib import Path
from typing import Dict, List, Sequence

import mir_eval
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from piano_vad import onsets_frames_note_detection
from train_midi_voice_assignment import (
    NoteEvent,
    VOICE_NAMES,
    build_song_events,
    create_folder,
    load_split,
    seed_everything,
    setup_logging,
)
from utilities import note_to_freq


BEGIN_NOTE = 21
CLASSES_NUM = 88
FIXED_ONSET_TOLERANCES = (0.05, 0.10)


def event_pitch(event) -> int:
    return int(event["midi_note"] if isinstance(event, dict) else event.midi_note)


def event_onset(event) -> float:
    return float(event["onset_time"] if isinstance(event, dict) else event.onset_time)


def event_offset(event) -> float:
    return float(event["offset_time"] if isinstance(event, dict) else event.offset_time)


def event_voice(event) -> int | None:
    if isinstance(event, dict):
        return event.get("voice_idx")
    return int(event.voice_idx)


def stable_segment_seed(seed: int, stem: str, start_frame: int) -> int:
    token = f"{seed}:{stem}:{start_frame}".encode("utf-8")
    return zlib.crc32(token) & 0xFFFFFFFF


def note_to_index(midi_note: int) -> int | None:
    idx = int(midi_note) - BEGIN_NOTE
    if 0 <= idx < CLASSES_NUM:
        return idx
    return None


def select_segment_events(
    events: Sequence[NoteEvent],
    start_time: float,
    end_time: float,
) -> List[NoteEvent]:
    selected = []
    for event in events:
        if event_offset(event) <= start_time:
            continue
        if event_onset(event) >= end_time:
            continue
        selected.append(event)
    return selected


def voice_events_to_merged_events(events: Sequence[NoteEvent]) -> List[Dict[str, float]]:
    merged = []
    for event in events:
        merged.append(
            {
                "midi_note": int(event.midi_note),
                "onset_time": float(event.onset_time),
                "offset_time": float(event.offset_time),
            }
        )
    return merged


def localize_events(
    events: Sequence[NoteEvent | Dict[str, float]],
    start_time: float,
) -> List[Dict[str, float]]:
    localized = []
    for event in events:
        item = {
            "midi_note": event_pitch(event),
            "onset_time": event_onset(event) - start_time,
            "offset_time": event_offset(event) - start_time,
        }
        voice_idx = event_voice(event)
        if voice_idx is not None:
            item["voice_idx"] = int(voice_idx)
        localized.append(item)
    return localized


def clone_event(event: Dict[str, float]) -> Dict[str, float]:
    return {
        "midi_note": int(event["midi_note"]),
        "onset_time": float(event["onset_time"]),
        "offset_time": float(event["offset_time"]),
    }


def jitter_note_event(
    event: Dict[str, float],
    rng: np.random.Generator,
    segment_seconds: float,
    frames_per_second: int,
    pitch_shift_prob: float,
    max_pitch_shift: int,
    onset_jitter_prob: float,
    max_onset_jitter_frames: int,
    offset_jitter_prob: float,
    max_offset_jitter_frames: int,
    min_note_frames: int,
) -> Dict[str, float]:
    note = clone_event(event)

    if pitch_shift_prob > 0 and max_pitch_shift > 0 and rng.random() < pitch_shift_prob:
        shift = int(rng.integers(-max_pitch_shift, max_pitch_shift + 1))
        note["midi_note"] = int(np.clip(note["midi_note"] + shift, BEGIN_NOTE, BEGIN_NOTE + CLASSES_NUM - 1))

    contained = 0.0 <= note["onset_time"] < segment_seconds and 0.0 < note["offset_time"] <= segment_seconds
    if contained and onset_jitter_prob > 0 and max_onset_jitter_frames > 0 and rng.random() < onset_jitter_prob:
        jitter = int(rng.integers(-max_onset_jitter_frames, max_onset_jitter_frames + 1))
        note["onset_time"] += jitter / float(frames_per_second)

    if contained and offset_jitter_prob > 0 and max_offset_jitter_frames > 0 and rng.random() < offset_jitter_prob:
        jitter = int(rng.integers(-max_offset_jitter_frames, max_offset_jitter_frames + 1))
        note["offset_time"] += jitter / float(frames_per_second)

    min_duration = max(1.0 / frames_per_second, min_note_frames / float(frames_per_second))
    note["onset_time"] = float(np.clip(note["onset_time"], -segment_seconds, max(segment_seconds - min_duration, 0.0)))
    note["offset_time"] = float(np.clip(note["offset_time"], note["onset_time"] + min_duration, segment_seconds + segment_seconds))
    return note


def sample_extra_notes(
    base_events: Sequence[Dict[str, float]],
    rng: np.random.Generator,
    segment_seconds: float,
    frames_per_second: int,
    extra_note_ratio: float,
    extra_pitch_span: int,
    min_note_frames: int,
) -> List[Dict[str, float]]:
    if extra_note_ratio <= 0:
        return []

    if len(base_events) == 0:
        count = 1 if rng.random() < extra_note_ratio else 0
    else:
        lam = extra_note_ratio * len(base_events)
        count = int(rng.poisson(lam))

    min_duration = max(1.0 / frames_per_second, min_note_frames / float(frames_per_second))
    extras: List[Dict[str, float]] = []
    for _ in range(count):
        if base_events:
            base = base_events[int(rng.integers(0, len(base_events)))]
            pitch = int(base["midi_note"] + rng.integers(-extra_pitch_span, extra_pitch_span + 1))
            onset = float(base["onset_time"] + rng.uniform(-0.2, 0.2))
            base_duration = max(min_duration, float(base["offset_time"] - base["onset_time"]))
            duration = float(base_duration * rng.uniform(0.5, 1.5))
        else:
            pitch = int(rng.integers(48, 72))
            onset = float(rng.uniform(0.0, max(segment_seconds - min_duration, 1e-4)))
            duration = float(rng.uniform(min_duration, max(0.5, min_duration)))

        pitch = int(np.clip(pitch, BEGIN_NOTE, BEGIN_NOTE + CLASSES_NUM - 1))
        onset = float(np.clip(onset, 0.0, max(segment_seconds - min_duration, 0.0)))
        offset = float(np.clip(onset + duration, onset + min_duration, segment_seconds))
        extras.append(
            {
                "midi_note": pitch,
                "onset_time": onset,
                "offset_time": offset,
            }
        )
    return extras


def corrupt_merged_events(
    events: Sequence[Dict[str, float]],
    rng: np.random.Generator,
    segment_seconds: float,
    frames_per_second: int,
    drop_note_prob: float,
    pitch_shift_prob: float,
    max_pitch_shift: int,
    onset_jitter_prob: float,
    max_onset_jitter_frames: int,
    offset_jitter_prob: float,
    max_offset_jitter_frames: int,
    extra_note_ratio: float,
    extra_pitch_span: int,
    min_note_frames: int,
) -> List[Dict[str, float]]:
    corrupted: List[Dict[str, float]] = []
    for event in events:
        if drop_note_prob > 0 and rng.random() < drop_note_prob:
            continue
        corrupted.append(
            jitter_note_event(
                event=event,
                rng=rng,
                segment_seconds=segment_seconds,
                frames_per_second=frames_per_second,
                pitch_shift_prob=pitch_shift_prob,
                max_pitch_shift=max_pitch_shift,
                onset_jitter_prob=onset_jitter_prob,
                max_onset_jitter_frames=max_onset_jitter_frames,
                offset_jitter_prob=offset_jitter_prob,
                max_offset_jitter_frames=max_offset_jitter_frames,
                min_note_frames=min_note_frames,
            )
        )

    corrupted.extend(
        sample_extra_notes(
            base_events=events,
            rng=rng,
            segment_seconds=segment_seconds,
            frames_per_second=frames_per_second,
            extra_note_ratio=extra_note_ratio,
            extra_pitch_span=extra_pitch_span,
            min_note_frames=min_note_frames,
        )
    )
    corrupted.sort(key=lambda x: (x["onset_time"], x["midi_note"], x["offset_time"]))
    return corrupted


def build_roll_triplet(
    events: Sequence[Dict[str, float]],
    frames_num: int,
    frames_per_second: int,
) -> Dict[str, np.ndarray]:
    onset = np.zeros((frames_num, CLASSES_NUM), dtype=np.float32)
    frame = np.zeros((frames_num, CLASSES_NUM), dtype=np.float32)
    offset = np.zeros((frames_num, CLASSES_NUM), dtype=np.float32)

    for event in events:
        note_idx = note_to_index(event["midi_note"])
        if note_idx is None:
            continue

        bgn_frame = int(round(event["onset_time"] * frames_per_second))
        fin_frame = int(round(event["offset_time"] * frames_per_second))
        if fin_frame < 0 or bgn_frame >= frames_num:
            continue

        frame_bgn = max(bgn_frame, 0)
        frame_fin = min(fin_frame, frames_num - 1)
        if frame_fin < frame_bgn:
            continue

        frame[frame_bgn : frame_fin + 1, note_idx] = 1.0
        if 0 <= bgn_frame < frames_num:
            onset[bgn_frame, note_idx] = 1.0
        if 0 <= fin_frame < frames_num:
            offset[fin_frame, note_idx] = 1.0

    return {
        "onset": onset,
        "frame": frame,
        "offset": offset,
    }


def build_voice_target_rolls(
    events: Sequence[Dict[str, float]],
    frames_num: int,
    frames_per_second: int,
) -> Dict[str, np.ndarray]:
    onset = np.zeros((frames_num, len(VOICE_NAMES), CLASSES_NUM), dtype=np.float32)
    frame = np.zeros_like(onset)
    offset = np.zeros_like(onset)
    presence = np.zeros((len(VOICE_NAMES),), dtype=np.float32)

    for event in events:
        voice_idx = event.get("voice_idx")
        if voice_idx is None or not (0 <= int(voice_idx) < len(VOICE_NAMES)):
            continue
        note_idx = note_to_index(event["midi_note"])
        if note_idx is None:
            continue

        bgn_frame = int(round(event["onset_time"] * frames_per_second))
        fin_frame = int(round(event["offset_time"] * frames_per_second))
        if fin_frame < 0 or bgn_frame >= frames_num:
            continue

        voice_idx = int(voice_idx)
        presence[voice_idx] = 1.0
        frame_bgn = max(bgn_frame, 0)
        frame_fin = min(fin_frame, frames_num - 1)
        if frame_fin < frame_bgn:
            continue

        frame[frame_bgn : frame_fin + 1, voice_idx, note_idx] = 1.0
        if 0 <= bgn_frame < frames_num:
            onset[bgn_frame, voice_idx, note_idx] = 1.0
        if 0 <= fin_frame < frames_num:
            offset[fin_frame, voice_idx, note_idx] = 1.0

    return {
        "voice_onset_roll": onset,
        "voice_frame_roll": frame,
        "voice_offset_roll": offset,
        "voice_presence": presence,
    }


def build_segment_sample(
    song_events: Sequence[NoteEvent],
    start_frame: int,
    segment_seconds: float,
    frames_per_second: int,
    apply_corruption: bool,
    rng: np.random.Generator,
    clean_input_prob: float,
    drop_note_prob: float,
    pitch_shift_prob: float,
    max_pitch_shift: int,
    onset_jitter_prob: float,
    max_onset_jitter_frames: int,
    offset_jitter_prob: float,
    max_offset_jitter_frames: int,
    extra_note_ratio: float,
    extra_pitch_span: int,
    min_note_frames: int,
) -> Dict[str, np.ndarray]:
    frames_num = int(round(segment_seconds * frames_per_second)) + 1
    start_time = start_frame / float(frames_per_second)
    end_time = start_time + segment_seconds

    segment_voice_events = select_segment_events(song_events, start_time, end_time)
    local_voice_events = localize_events(segment_voice_events, start_time)
    merged_clean_events = voice_events_to_merged_events(segment_voice_events)
    local_merged_clean_events = localize_events(merged_clean_events, start_time)

    if apply_corruption and (clean_input_prob <= 0.0 or rng.random() > clean_input_prob):
        input_events = corrupt_merged_events(
            events=local_merged_clean_events,
            rng=rng,
            segment_seconds=segment_seconds,
            frames_per_second=frames_per_second,
            drop_note_prob=drop_note_prob,
            pitch_shift_prob=pitch_shift_prob,
            max_pitch_shift=max_pitch_shift,
            onset_jitter_prob=onset_jitter_prob,
            max_onset_jitter_frames=max_onset_jitter_frames,
            offset_jitter_prob=offset_jitter_prob,
            max_offset_jitter_frames=max_offset_jitter_frames,
            extra_note_ratio=extra_note_ratio,
            extra_pitch_span=extra_pitch_span,
            min_note_frames=min_note_frames,
        )
    else:
        input_events = [clone_event(event) for event in local_merged_clean_events]

    input_rolls = build_roll_triplet(input_events, frames_num, frames_per_second)
    target_rolls = build_voice_target_rolls(local_voice_events, frames_num, frames_per_second)
    union_rolls = {
        "onset_roll": np.max(target_rolls["voice_onset_roll"], axis=1),
        "frame_roll": np.max(target_rolls["voice_frame_roll"], axis=1),
        "offset_roll": np.max(target_rolls["voice_offset_roll"], axis=1),
    }

    return {
        "input_rolls": np.stack(
            [input_rolls["onset"], input_rolls["frame"], input_rolls["offset"]],
            axis=1,
        ).astype(np.float32),
        "voice_onset_roll": target_rolls["voice_onset_roll"],
        "voice_frame_roll": target_rolls["voice_frame_roll"],
        "voice_offset_roll": target_rolls["voice_offset_roll"],
        "onset_roll": union_rolls["onset_roll"],
        "frame_roll": union_rolls["frame_roll"],
        "offset_roll": union_rolls["offset_roll"],
        "voice_presence": target_rolls["voice_presence"],
    }


def build_clean_song_input_rolls(
    song_events: Sequence[NoteEvent],
    frames_per_second: int,
) -> np.ndarray:
    max_offset = max((event.offset_time for event in song_events), default=0.0)
    frames_num = max(1, int(round(max_offset * frames_per_second)) + 1)
    merged_events = voice_events_to_merged_events(song_events)
    input_rolls = build_roll_triplet(merged_events, frames_num, frames_per_second)
    return np.stack(
        [input_rolls["onset"], input_rolls["frame"], input_rolls["offset"]],
        axis=1,
    ).astype(np.float32)


class YouChoraleSymbolicSATBEditorDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        split: str,
        segment_seconds: float,
        segment_stride_seconds: float,
        frames_per_second: int,
        seed: int,
        apply_corruption: bool = True,
        clean_input_prob: float = 0.05,
        drop_note_prob: float = 0.10,
        pitch_shift_prob: float = 0.10,
        max_pitch_shift: int = 2,
        onset_jitter_prob: float = 0.20,
        max_onset_jitter_frames: int = 4,
        offset_jitter_prob: float = 0.20,
        max_offset_jitter_frames: int = 6,
        extra_note_ratio: float = 0.06,
        extra_pitch_span: int = 7,
        min_note_frames: int = 2,
        deterministic_corruption: bool = False,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.segment_seconds = float(segment_seconds)
        self.segment_stride_seconds = float(segment_stride_seconds)
        self.frames_per_second = int(frames_per_second)
        self.segment_frames = int(round(self.segment_seconds * self.frames_per_second))
        self.segment_stride_frames = max(1, int(round(self.segment_stride_seconds * self.frames_per_second)))
        self.seed = int(seed)
        self.apply_corruption = bool(apply_corruption)
        self.clean_input_prob = float(clean_input_prob)
        self.drop_note_prob = float(drop_note_prob)
        self.pitch_shift_prob = float(pitch_shift_prob)
        self.max_pitch_shift = int(max_pitch_shift)
        self.onset_jitter_prob = float(onset_jitter_prob)
        self.max_onset_jitter_frames = int(max_onset_jitter_frames)
        self.offset_jitter_prob = float(offset_jitter_prob)
        self.max_offset_jitter_frames = int(max_offset_jitter_frames)
        self.extra_note_ratio = float(extra_note_ratio)
        self.extra_pitch_span = int(extra_pitch_span)
        self.min_note_frames = int(min_note_frames)
        self.deterministic_corruption = bool(deterministic_corruption)

        self.stems = load_split(self.dataset_dir, split)
        self.song_events: Dict[str, List[NoteEvent]] = {}
        self.segments: List[tuple[str, int]] = []

        for stem in self.stems:
            events = build_song_events(self.dataset_dir, stem, self.frames_per_second)
            if not events:
                continue
            self.song_events[stem] = events
            max_offset = max(event.offset_time for event in events)
            total_frames = max(1, int(round(max_offset * self.frames_per_second)) + 1)

            if total_frames <= self.segment_frames:
                self.segments.append((stem, 0))
                continue

            for start_frame in range(0, total_frames, self.segment_stride_frames):
                self.segments.append((stem, start_frame))
                if start_frame + self.segment_frames >= total_frames:
                    break

        if not self.segments:
            raise RuntimeError(f"No segments found for split={split} in {self.dataset_dir}")

    def __len__(self) -> int:
        return len(self.segments)

    def _rng_for_item(self, stem: str, start_frame: int) -> np.random.Generator:
        if self.deterministic_corruption:
            return np.random.default_rng(stable_segment_seed(self.seed, stem, start_frame))
        return np.random.default_rng()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        stem, start_frame = self.segments[index]
        rng = self._rng_for_item(stem, start_frame)
        sample = build_segment_sample(
            song_events=self.song_events[stem],
            start_frame=start_frame,
            segment_seconds=self.segment_seconds,
            frames_per_second=self.frames_per_second,
            apply_corruption=self.apply_corruption,
            rng=rng,
            clean_input_prob=self.clean_input_prob,
            drop_note_prob=self.drop_note_prob,
            pitch_shift_prob=self.pitch_shift_prob,
            max_pitch_shift=self.max_pitch_shift,
            onset_jitter_prob=self.onset_jitter_prob,
            max_onset_jitter_frames=self.max_onset_jitter_frames,
            offset_jitter_prob=self.offset_jitter_prob,
            max_offset_jitter_frames=self.max_offset_jitter_frames,
            extra_note_ratio=self.extra_note_ratio,
            extra_pitch_span=self.extra_pitch_span,
            min_note_frames=self.min_note_frames,
        )
        return {key: torch.from_numpy(value) for key, value in sample.items()}


class SymbolicSATBEditorNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        classes_num: int = CLASSES_NUM,
        model_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_voices: int = 4,
    ):
        super().__init__()
        self.num_voices = int(num_voices)
        self.classes_num = int(classes_num)
        input_dim = input_channels * classes_num

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(model_dim, model_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.encoder = nn.GRU(
            input_size=model_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        encoded_dim = hidden_size * 2
        self.onset_head = nn.Linear(encoded_dim, num_voices * classes_num)
        self.frame_head = nn.Linear(encoded_dim, num_voices * classes_num)
        self.offset_head = nn.Linear(encoded_dim, num_voices * classes_num)
        self.presence_head = nn.Sequential(
            nn.Linear(encoded_dim, encoded_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoded_dim, num_voices),
        )

    def forward(self, input_rolls: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, frames_num, channels, classes_num = input_rolls.shape
        x = input_rolls.reshape(batch_size, frames_num, channels * classes_num)
        x = self.input_projection(x)

        conv_in = x.transpose(1, 2)
        x = x + self.temporal_conv(conv_in).transpose(1, 2)
        encoded, _ = self.encoder(x)

        onset_logits = self.onset_head(encoded).reshape(batch_size, frames_num, self.num_voices, self.classes_num)
        frame_logits = self.frame_head(encoded).reshape(batch_size, frames_num, self.num_voices, self.classes_num)
        offset_logits = self.offset_head(encoded).reshape(batch_size, frames_num, self.num_voices, self.classes_num)

        pooled = encoded.mean(dim=1)
        presence_logits = self.presence_head(pooled)

        voice_onset_probs = torch.sigmoid(onset_logits)
        voice_frame_probs = torch.sigmoid(frame_logits)
        voice_offset_probs = torch.sigmoid(offset_logits)

        return {
            "voice_onset_logits": onset_logits,
            "voice_frame_logits": frame_logits,
            "voice_offset_logits": offset_logits,
            "voice_onset_probs": voice_onset_probs,
            "voice_frame_probs": voice_frame_probs,
            "voice_offset_probs": voice_offset_probs,
            "onset_probs": torch.max(voice_onset_probs, dim=2).values,
            "frame_probs": torch.max(voice_frame_probs, dim=2).values,
            "offset_probs": torch.max(voice_offset_probs, dim=2).values,
            "presence_logits": presence_logits,
        }


def weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = 1.0 + (positive_weight - 1.0) * targets
    return (loss * weights).mean()


def weighted_bce_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    loss = F.binary_cross_entropy(probs.clamp(1e-6, 1.0 - 1e-6), targets, reduction="none")
    weights = 1.0 + (positive_weight - 1.0) * targets
    return (loss * weights).mean()


def frame_f1_from_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    pred = probs >= threshold
    tgt = targets >= 0.5
    tp = float((pred & tgt).sum().item())
    fp = float((pred & (~tgt)).sum().item())
    fn = float(((~pred) & tgt).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return 2.0 * precision * recall / max(precision + recall, 1e-8)


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    voice_frame_loss_weight: float,
    voice_onset_loss_weight: float,
    voice_offset_loss_weight: float,
    union_frame_loss_weight: float,
    union_onset_loss_weight: float,
    union_offset_loss_weight: float,
    presence_loss_weight: float,
    frame_positive_weight: float,
    onset_positive_weight: float,
    offset_positive_weight: float,
) -> Dict[str, torch.Tensor]:
    voice_frame_loss = weighted_bce_with_logits(
        outputs["voice_frame_logits"],
        batch["voice_frame_roll"],
        positive_weight=frame_positive_weight,
    )
    voice_onset_loss = weighted_bce_with_logits(
        outputs["voice_onset_logits"],
        batch["voice_onset_roll"],
        positive_weight=onset_positive_weight,
    )
    voice_offset_loss = weighted_bce_with_logits(
        outputs["voice_offset_logits"],
        batch["voice_offset_roll"],
        positive_weight=offset_positive_weight,
    )
    union_frame_loss = weighted_bce_probs(
        outputs["frame_probs"],
        batch["frame_roll"],
        positive_weight=frame_positive_weight,
    )
    union_onset_loss = weighted_bce_probs(
        outputs["onset_probs"],
        batch["onset_roll"],
        positive_weight=onset_positive_weight,
    )
    union_offset_loss = weighted_bce_probs(
        outputs["offset_probs"],
        batch["offset_roll"],
        positive_weight=offset_positive_weight,
    )
    presence_loss = F.binary_cross_entropy_with_logits(outputs["presence_logits"], batch["voice_presence"])

    total_loss = (
        voice_frame_loss_weight * voice_frame_loss
        + voice_onset_loss_weight * voice_onset_loss
        + voice_offset_loss_weight * voice_offset_loss
        + union_frame_loss_weight * union_frame_loss
        + union_onset_loss_weight * union_onset_loss
        + union_offset_loss_weight * union_offset_loss
        + presence_loss_weight * presence_loss
    )

    return {
        "loss": total_loss,
        "voice_frame_loss": voice_frame_loss.detach(),
        "voice_onset_loss": voice_onset_loss.detach(),
        "voice_offset_loss": voice_offset_loss.detach(),
        "union_frame_loss": union_frame_loss.detach(),
        "union_onset_loss": union_onset_loss.detach(),
        "union_offset_loss": union_offset_loss.detach(),
        "presence_loss": presence_loss.detach(),
    }


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "voice_frame_loss": 0.0,
        "voice_onset_loss": 0.0,
        "voice_offset_loss": 0.0,
        "union_frame_loss": 0.0,
        "union_onset_loss": 0.0,
        "union_offset_loss": 0.0,
        "presence_loss": 0.0,
    }
    steps = 0
    voice_frame_f1 = 0.0
    union_frame_f1 = 0.0
    presence_correct = 0.0
    presence_total = 0.0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = move_batch_to_device(batch, device)
        outputs = model(batch["input_rolls"])
        loss_dict = compute_loss(
            outputs=outputs,
            batch=batch,
            voice_frame_loss_weight=args.voice_frame_loss_weight,
            voice_onset_loss_weight=args.voice_onset_loss_weight,
            voice_offset_loss_weight=args.voice_offset_loss_weight,
            union_frame_loss_weight=args.union_frame_loss_weight,
            union_onset_loss_weight=args.union_onset_loss_weight,
            union_offset_loss_weight=args.union_offset_loss_weight,
            presence_loss_weight=args.presence_loss_weight,
            frame_positive_weight=args.frame_positive_weight,
            onset_positive_weight=args.onset_positive_weight,
            offset_positive_weight=args.offset_positive_weight,
        )
        for key in totals:
            totals[key] += float(loss_dict[key].item())

        voice_frame_f1 += frame_f1_from_probs(outputs["voice_frame_probs"], batch["voice_frame_roll"])
        union_frame_f1 += frame_f1_from_probs(outputs["frame_probs"], batch["frame_roll"])
        presence_pred = (torch.sigmoid(outputs["presence_logits"]) >= 0.5).float()
        presence_correct += float((presence_pred == batch["voice_presence"]).sum().item())
        presence_total += float(batch["voice_presence"].numel())
        steps += 1

    metrics = {key: value / max(1, steps) for key, value in totals.items()}
    metrics["voice_frame_f1"] = voice_frame_f1 / max(1, steps)
    metrics["union_frame_f1"] = union_frame_f1 / max(1, steps)
    metrics["presence_accuracy"] = presence_correct / max(1.0, presence_total)
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    grad_clip: float,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "voice_frame_loss": 0.0,
        "voice_onset_loss": 0.0,
        "voice_offset_loss": 0.0,
        "union_frame_loss": 0.0,
        "union_onset_loss": 0.0,
        "union_offset_loss": 0.0,
        "presence_loss": 0.0,
    }
    steps = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["input_rolls"])
        loss_dict = compute_loss(
            outputs=outputs,
            batch=batch,
            voice_frame_loss_weight=args.voice_frame_loss_weight,
            voice_onset_loss_weight=args.voice_onset_loss_weight,
            voice_offset_loss_weight=args.voice_offset_loss_weight,
            union_frame_loss_weight=args.union_frame_loss_weight,
            union_onset_loss_weight=args.union_onset_loss_weight,
            union_offset_loss_weight=args.union_offset_loss_weight,
            presence_loss_weight=args.presence_loss_weight,
            frame_positive_weight=args.frame_positive_weight,
            onset_positive_weight=args.onset_positive_weight,
            offset_positive_weight=args.offset_positive_weight,
        )
        loss_dict["loss"].backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for key in totals:
            totals[key] += float(loss_dict[key].item())
        steps += 1

    return {key: value / max(1, steps) for key, value in totals.items()}


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_metric: float,
    metric_name: str,
    args: argparse.Namespace,
) -> None:
    create_folder(checkpoint_path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_metric": best_metric,
            "metric_name": metric_name,
            "args": vars(args),
        },
        checkpoint_path,
    )


def build_model_from_args(args: argparse.Namespace) -> SymbolicSATBEditorNet:
    return SymbolicSATBEditorNet(
        input_channels=3,
        classes_num=CLASSES_NUM,
        model_dim=args.model_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_voices=len(VOICE_NAMES),
    )


def infer_song_rolls(
    model: nn.Module,
    input_rolls: np.ndarray,
    device: torch.device,
    segment_seconds: float,
    segment_stride_seconds: float,
    frames_per_second: int,
) -> Dict[str, np.ndarray]:
    frames_num = input_rolls.shape[0]
    segment_frames = int(round(segment_seconds * frames_per_second)) + 1
    stride_frames = max(1, int(round(segment_stride_seconds * frames_per_second)))

    accum = {
        "voice_onset_probs": np.zeros((frames_num, len(VOICE_NAMES), CLASSES_NUM), dtype=np.float32),
        "voice_frame_probs": np.zeros((frames_num, len(VOICE_NAMES), CLASSES_NUM), dtype=np.float32),
        "voice_offset_probs": np.zeros((frames_num, len(VOICE_NAMES), CLASSES_NUM), dtype=np.float32),
    }
    counts = np.zeros((frames_num, 1, 1), dtype=np.float32)
    presence_logits_list: List[np.ndarray] = []

    model.eval()
    for start in range(0, frames_num, stride_frames):
        end = min(frames_num, start + segment_frames)
        valid_len = end - start
        if valid_len <= 0:
            continue

        chunk = np.zeros((segment_frames, 3, CLASSES_NUM), dtype=np.float32)
        chunk[:valid_len] = input_rolls[start:end]
        batch = torch.from_numpy(chunk).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(batch)

        accum["voice_onset_probs"][start:end] += outputs["voice_onset_probs"][0, :valid_len].detach().cpu().numpy()
        accum["voice_frame_probs"][start:end] += outputs["voice_frame_probs"][0, :valid_len].detach().cpu().numpy()
        accum["voice_offset_probs"][start:end] += outputs["voice_offset_probs"][0, :valid_len].detach().cpu().numpy()
        counts[start:end] += 1.0
        presence_logits_list.append(outputs["presence_logits"][0].detach().cpu().numpy())

        if end == frames_num:
            break

    counts = np.clip(counts, 1.0, None)
    averaged = {key: value / counts for key, value in accum.items()}
    averaged["voice_presence_probs"] = 1.0 / (1.0 + np.exp(-np.mean(np.stack(presence_logits_list, axis=0), axis=0)))
    averaged["onset_probs"] = np.max(averaged["voice_onset_probs"], axis=1)
    averaged["frame_probs"] = np.max(averaged["voice_frame_probs"], axis=1)
    averaged["offset_probs"] = np.max(averaged["voice_offset_probs"], axis=1)
    return averaged


def sharp_output(x: np.ndarray, threshold: float) -> np.ndarray:
    frames_num, classes_num = x.shape
    y = np.zeros_like(x, dtype=np.float32)
    for piano_note in range(classes_num):
        for i in range(1, frames_num - 1):
            if x[i, piano_note] > threshold and x[i, piano_note] > x[i - 1, piano_note] and x[i, piano_note] > x[i + 1, piano_note]:
                y[i, piano_note] = 1.0
    return y


def decode_note_events(
    frame_probs: np.ndarray,
    onset_probs: np.ndarray,
    offset_probs: np.ndarray,
    frames_per_second: int,
    frame_threshold: float,
    onset_threshold: float,
    offset_threshold: float,
) -> List[Dict[str, float]]:
    onset_binary = sharp_output(onset_probs, onset_threshold)
    offset_binary = sharp_output(offset_probs, offset_threshold)
    velocity_output = np.ones_like(frame_probs, dtype=np.float32)

    events: List[Dict[str, float]] = []
    for piano_note in range(CLASSES_NUM):
        tuples = onsets_frames_note_detection(
            frame_output=frame_probs[:, piano_note],
            onset_output=onset_binary[:, piano_note],
            offset_output=offset_binary[:, piano_note],
            velocity_output=velocity_output[:, piano_note],
            threshold=frame_threshold,
        )
        for bgn, fin, velocity in tuples:
            fin = max(fin, bgn + 1)
            events.append(
                {
                    "midi_note": piano_note + BEGIN_NOTE,
                    "onset_time": bgn / float(frames_per_second),
                    "offset_time": fin / float(frames_per_second),
                    "velocity": int(np.clip(velocity * 127.0, 1, 127)),
                }
            )
    events.sort(key=lambda x: (x["onset_time"], x["midi_note"], x["offset_time"]))
    return events


def decode_satb_tracks(
    averaged_outputs: Dict[str, np.ndarray],
    frames_per_second: int,
    frame_threshold: float,
    onset_threshold: float,
    offset_threshold: float,
) -> Dict[str, List[Dict[str, float]]]:
    voice_tracks = {}
    for voice_idx, voice_name in enumerate(VOICE_NAMES):
        voice_tracks[voice_name] = decode_note_events(
            frame_probs=averaged_outputs["voice_frame_probs"][:, voice_idx, :],
            onset_probs=averaged_outputs["voice_onset_probs"][:, voice_idx, :],
            offset_probs=averaged_outputs["voice_offset_probs"][:, voice_idx, :],
            frames_per_second=frames_per_second,
            frame_threshold=frame_threshold,
            onset_threshold=onset_threshold,
            offset_threshold=offset_threshold,
        )
    return voice_tracks


def reference_voice_tracks(song_events: Sequence[NoteEvent]) -> Dict[str, List[Dict[str, float]]]:
    tracks = {voice_name: [] for voice_name in VOICE_NAMES}
    for event in song_events:
        voice_name = VOICE_NAMES[int(event.voice_idx)]
        tracks[voice_name].append(
            {
                "midi_note": int(event.midi_note),
                "onset_time": float(event.onset_time),
                "offset_time": float(event.offset_time),
                "velocity": 100,
            }
        )
    return tracks


def events_to_intervals_and_pitches(events: Sequence[Dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    if len(events) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    intervals = np.asarray([[event["onset_time"], event["offset_time"]] for event in events], dtype=np.float32)
    pitches = np.asarray([event["midi_note"] for event in events], dtype=np.int32)
    bad = intervals[:, 1] <= intervals[:, 0]
    if np.any(bad):
        intervals = intervals.copy()
        intervals[bad, 1] = intervals[bad, 0] + 1e-4
    order = np.argsort(intervals[:, 0], kind="mergesort")
    return intervals[order], pitches[order]


def compute_track_note_metrics(
    ref_events: Sequence[Dict[str, float]],
    pred_events: Sequence[Dict[str, float]],
    onset_tolerance: float,
) -> Dict[str, float]:
    ref_intervals, ref_pitches = events_to_intervals_and_pitches(ref_events)
    pred_intervals, pred_pitches = events_to_intervals_and_pitches(pred_events)
    precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals=ref_intervals,
        ref_pitches=note_to_freq(ref_pitches),
        est_intervals=pred_intervals,
        est_pitches=note_to_freq(pred_pitches),
        onset_tolerance=onset_tolerance,
        offset_ratio=None,
        offset_min_tolerance=0.05,
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_choral_note_summary(
    ref_tracks: Dict[str, Sequence[Dict[str, float]]],
    pred_tracks: Dict[str, Sequence[Dict[str, float]]],
) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for onset_tolerance in FIXED_ONSET_TOLERANCES:
        tag = f"{int(round(onset_tolerance * 1000.0))}ms"
        f1_values = []
        for voice_name in VOICE_NAMES:
            metrics = compute_track_note_metrics(
                ref_events=ref_tracks[voice_name],
                pred_events=pred_tracks[voice_name],
                onset_tolerance=onset_tolerance,
            )
            summary[f"{voice_name}_precision_{tag}"] = metrics["precision"]
            summary[f"{voice_name}_recall_{tag}"] = metrics["recall"]
            summary[f"{voice_name}_f1_{tag}"] = metrics["f1"]
            f1_values.append(metrics["f1"])
        summary[f"mean_satb_note_f1_{tag}"] = float(np.mean(f1_values)) if f1_values else 0.0
        summary[f"min_satb_note_f1_{tag}"] = float(np.min(f1_values)) if f1_values else 0.0

    ref_presence = np.asarray([1.0 if len(ref_tracks[name]) > 0 else 0.0 for name in VOICE_NAMES], dtype=np.float32)
    pred_presence = np.asarray([1.0 if len(pred_tracks[name]) > 0 else 0.0 for name in VOICE_NAMES], dtype=np.float32)
    summary["presence_accuracy"] = float(np.mean(ref_presence == pred_presence))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a symbolic SATB editor that maps noisy merged MIDI rolls to four voice-specific rolls.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./data/YouChorale",
    )
    parser.add_argument("--workspace", type=str, default="./workspaces")
    parser.add_argument("--experiment-name", type=str, default="youchorale_symbolic_satb_editor")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--valid-split", type=str, default="valid")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--segment-seconds", type=float, default=12.0)
    parser.add_argument("--segment-stride-seconds", type=float, default=6.0)
    parser.add_argument("--frames-per-second", type=int, default=100)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--voice-frame-loss-weight", type=float, default=1.0)
    parser.add_argument("--voice-onset-loss-weight", type=float, default=2.0)
    parser.add_argument("--voice-offset-loss-weight", type=float, default=1.0)
    parser.add_argument("--union-frame-loss-weight", type=float, default=0.5)
    parser.add_argument("--union-onset-loss-weight", type=float, default=1.0)
    parser.add_argument("--union-offset-loss-weight", type=float, default=0.5)
    parser.add_argument("--presence-loss-weight", type=float, default=0.1)
    parser.add_argument("--frame-positive-weight", type=float, default=1.5)
    parser.add_argument("--onset-positive-weight", type=float, default=4.0)
    parser.add_argument("--offset-positive-weight", type=float, default=4.0)
    parser.add_argument("--clean-input-prob", type=float, default=0.05)
    parser.add_argument("--drop-note-prob", type=float, default=0.10)
    parser.add_argument("--pitch-shift-prob", type=float, default=0.10)
    parser.add_argument("--max-pitch-shift", type=int, default=2)
    parser.add_argument("--onset-jitter-prob", type=float, default=0.20)
    parser.add_argument("--max-onset-jitter-frames", type=int, default=4)
    parser.add_argument("--offset-jitter-prob", type=float, default=0.20)
    parser.add_argument("--max-offset-jitter-frames", type=int, default=6)
    parser.add_argument("--extra-note-ratio", type=float, default=0.06)
    parser.add_argument("--extra-pitch-span", type=int, default=7)
    parser.add_argument("--min-note-frames", type=int, default=2)
    parser.add_argument("--clean-valid-input", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=86)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-valid-batches", type=int, default=0)
    parser.add_argument("--resume", type=str, default="")
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    seed_everything(args.seed)

    workspace = Path(args.workspace)
    experiment_dir = workspace / "symbolic_satb_editor" / args.experiment_name
    checkpoints_dir = experiment_dir / "checkpoints"
    logs_dir = experiment_dir / "logs"
    tensorboard_dir = experiment_dir / "tensorboard"
    create_folder(experiment_dir)
    setup_logging(logs_dir / "train.log")

    with (experiment_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    device = resolve_device(args.device)
    logging.info("Using device: %s", device)
    logging.info("Experiment dir: %s", experiment_dir)

    train_dataset = YouChoraleSymbolicSATBEditorDataset(
        dataset_dir=args.dataset_dir,
        split=args.train_split,
        segment_seconds=args.segment_seconds,
        segment_stride_seconds=args.segment_stride_seconds,
        frames_per_second=args.frames_per_second,
        seed=args.seed,
        apply_corruption=True,
        clean_input_prob=args.clean_input_prob,
        drop_note_prob=args.drop_note_prob,
        pitch_shift_prob=args.pitch_shift_prob,
        max_pitch_shift=args.max_pitch_shift,
        onset_jitter_prob=args.onset_jitter_prob,
        max_onset_jitter_frames=args.max_onset_jitter_frames,
        offset_jitter_prob=args.offset_jitter_prob,
        max_offset_jitter_frames=args.max_offset_jitter_frames,
        extra_note_ratio=args.extra_note_ratio,
        extra_pitch_span=args.extra_pitch_span,
        min_note_frames=args.min_note_frames,
        deterministic_corruption=False,
    )
    valid_dataset = YouChoraleSymbolicSATBEditorDataset(
        dataset_dir=args.dataset_dir,
        split=args.valid_split,
        segment_seconds=args.segment_seconds,
        segment_stride_seconds=args.segment_seconds,
        frames_per_second=args.frames_per_second,
        seed=args.seed,
        apply_corruption=not args.clean_valid_input,
        clean_input_prob=0.0 if not args.clean_valid_input else 1.0,
        drop_note_prob=args.drop_note_prob,
        pitch_shift_prob=args.pitch_shift_prob,
        max_pitch_shift=args.max_pitch_shift,
        onset_jitter_prob=args.onset_jitter_prob,
        max_onset_jitter_frames=args.max_onset_jitter_frames,
        offset_jitter_prob=args.offset_jitter_prob,
        max_offset_jitter_frames=args.max_offset_jitter_frames,
        extra_note_ratio=args.extra_note_ratio,
        extra_pitch_span=args.extra_pitch_span,
        min_note_frames=args.min_note_frames,
        deterministic_corruption=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model_from_args(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=6,
    )

    start_epoch = 1
    best_metric = -float("inf")
    best_metric_name = "voice_frame_f1"
    if args.resume:
        resume_path = Path(args.resume)
        logging.info("Resuming from %s", resume_path)
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))
        best_metric_name = str(checkpoint.get("metric_name", best_metric_name))

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    limit_train = args.limit_train_batches if args.limit_train_batches > 0 else None
    limit_valid = args.limit_valid_batches if args.limit_valid_batches > 0 else None

    logging.info(
        "Train segments=%d, valid segments=%d, batch_size=%d",
        len(train_dataset),
        len(valid_dataset),
        args.batch_size,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
            grad_clip=args.grad_clip,
            max_batches=limit_train,
        )
        valid_stats = evaluate(
            model=model,
            loader=valid_loader,
            device=device,
            args=args,
            max_batches=limit_valid,
        )
        scheduler.step(valid_stats["voice_frame_f1"])

        lr = optimizer.param_groups[0]["lr"]
        logging.info(
            "Epoch %03d | lr=%.6f | train_loss=%.4f | valid_loss=%.4f | "
            "voice_frame_f1=%.4f | union_frame_f1=%.4f | presence_acc=%.4f",
            epoch,
            lr,
            train_stats["loss"],
            valid_stats["loss"],
            valid_stats["voice_frame_f1"],
            valid_stats["union_frame_f1"],
            valid_stats["presence_accuracy"],
        )

        writer.add_scalar("train/loss", train_stats["loss"], epoch)
        writer.add_scalar("train/voice_frame_loss", train_stats["voice_frame_loss"], epoch)
        writer.add_scalar("train/voice_onset_loss", train_stats["voice_onset_loss"], epoch)
        writer.add_scalar("train/voice_offset_loss", train_stats["voice_offset_loss"], epoch)
        writer.add_scalar("train/union_frame_loss", train_stats["union_frame_loss"], epoch)
        writer.add_scalar("train/union_onset_loss", train_stats["union_onset_loss"], epoch)
        writer.add_scalar("train/union_offset_loss", train_stats["union_offset_loss"], epoch)
        writer.add_scalar("train/presence_loss", train_stats["presence_loss"], epoch)
        for key, value in valid_stats.items():
            writer.add_scalar(f"valid/{key}", value, epoch)
        writer.add_scalar("train/lr", lr, epoch)
        writer.flush()

        selection_metric = valid_stats["voice_frame_f1"]
        if selection_metric > best_metric:
            best_metric = selection_metric
            save_checkpoint(
                checkpoint_path=checkpoints_dir / "best_voice_frame_f1.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_metric=best_metric,
                metric_name=best_metric_name,
                args=args,
            )
        save_checkpoint(
            checkpoint_path=checkpoints_dir / "last.pth",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_metric=best_metric,
            metric_name=best_metric_name,
            args=args,
        )

    writer.close()
    logging.info("Training finished. Best %s=%.4f", best_metric_name, best_metric)


if __name__ == "__main__":
    main()
