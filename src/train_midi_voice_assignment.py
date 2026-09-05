from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


VOICE_NAMES = ("S", "A", "T", "B")
VOICE_TO_INDEX = {name: idx for idx, name in enumerate(VOICE_NAMES)}
DEFAULT_RANGE_MINS = np.array([60, 55, 48, 40], dtype=np.float32)
DEFAULT_RANGE_MAXS = np.array([88, 79, 72, 67], dtype=np.float32)


@dataclass(frozen=True)
class NoteEvent:
    stem: str
    midi_note: int
    onset_time: float
    offset_time: float
    duration_sec: float
    beat_position: float
    duration_beats: float
    measure_beats: float
    voice_idx: int
    onset_group: int
    rank_in_group: int
    group_size: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logging(log_path: Path) -> None:
    create_folder(log_path.parent)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )


def voice_index(part_name: str) -> int | None:
    if not part_name:
        return None
    return VOICE_TO_INDEX.get(part_name[0].upper())


def range_mask_for_pitch(
    midi_note: int,
    range_mins: np.ndarray,
    range_maxs: np.ndarray,
    margin: float,
) -> np.ndarray:
    low = range_mins - margin
    high = range_maxs + margin
    return ((midi_note >= low) & (midi_note <= high)).astype(np.float32)


def range_costs_for_pitch(
    midi_note: int,
    range_mins: np.ndarray,
    range_maxs: np.ndarray,
) -> np.ndarray:
    centers = 0.5 * (range_mins + range_maxs)
    below = np.maximum(0.0, range_mins - midi_note)
    above = np.maximum(0.0, midi_note - range_maxs)
    return (below + above + 0.25 * np.abs(midi_note - centers) / 12.0).astype(np.float32)


def load_split(dataset_dir: Path, split: str) -> List[str]:
    split_path = dataset_dir / f"{split}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    with split_path.open("r") as f:
        stems = json.load(f)
    if not isinstance(stems, list):
        raise ValueError(f"Split file must contain a list of stems: {split_path}")
    return stems


def parse_note_bars(note_path: Path) -> List[Dict[str, float]]:
    with note_path.open("rb") as f:
        note_bars = pickle.load(f)

    raw_events = []
    for bar in note_bars:
        if not isinstance(bar, dict):
            continue
        measure_beats = float(bar.get("measure", 4.0) or 4.0)
        for part_name, note_list in bar.items():
            if part_name == "measure" or not isinstance(note_list, list):
                continue
            v_idx = voice_index(part_name)
            if v_idx is None:
                continue
            for note in note_list:
                if len(note) < 5:
                    continue
                midi_note = int(note[0])
                beat_position = float(note[1])
                duration_beats = float(note[2])
                onset_time = float(note[3])
                offset_time = float(note[4])
                if offset_time <= onset_time:
                    continue
                raw_events.append(
                    {
                        "midi_note": midi_note,
                        "onset_time": onset_time,
                        "offset_time": offset_time,
                        "duration_sec": offset_time - onset_time,
                        "beat_position": beat_position,
                        "duration_beats": duration_beats,
                        "measure_beats": measure_beats,
                        "voice_idx": v_idx,
                    }
                )

    raw_events.sort(
        key=lambda x: (
            x["onset_time"],
            -x["midi_note"],
            x["offset_time"],
            x["voice_idx"],
        )
    )
    return raw_events


def attach_onset_groups(
    stem: str,
    raw_events: Sequence[Dict[str, float]],
    onset_quantization_hz: int,
) -> List[NoteEvent]:
    grouped: Dict[int, List[Dict[str, float]]] = {}
    for event in raw_events:
        onset_group = int(round(event["onset_time"] * onset_quantization_hz))
        grouped.setdefault(onset_group, []).append(event)

    final_events: List[NoteEvent] = []
    for onset_group in sorted(grouped):
        group = sorted(
            grouped[onset_group],
            key=lambda x: (-x["midi_note"], x["offset_time"], x["voice_idx"]),
        )
        group_size = len(group)
        for rank_in_group, event in enumerate(group):
            final_events.append(
                NoteEvent(
                    stem=stem,
                    midi_note=int(event["midi_note"]),
                    onset_time=float(event["onset_time"]),
                    offset_time=float(event["offset_time"]),
                    duration_sec=float(event["duration_sec"]),
                    beat_position=float(event["beat_position"]),
                    duration_beats=float(event["duration_beats"]),
                    measure_beats=float(event["measure_beats"]),
                    voice_idx=int(event["voice_idx"]),
                    onset_group=onset_group,
                    rank_in_group=rank_in_group,
                    group_size=group_size,
                )
            )
    return final_events


def build_song_events(dataset_dir: Path, stem: str, onset_quantization_hz: int) -> List[NoteEvent]:
    note_path = dataset_dir / "note" / f"{stem}.pkl"
    if not note_path.exists():
        raise FileNotFoundError(f"Missing note file: {note_path}")
    raw_events = parse_note_bars(note_path)
    return attach_onset_groups(stem, raw_events, onset_quantization_hz)


def build_segment_sample(
    events: Sequence[NoteEvent],
    segment_notes: int,
    range_mins: np.ndarray,
    range_maxs: np.ndarray,
    range_margin: float,
) -> Dict[str, np.ndarray]:
    pitch_ids = np.zeros((segment_notes,), dtype=np.int64)
    pitch_classes = np.zeros((segment_notes,), dtype=np.int64)
    cont_features = np.zeros((segment_notes, 11), dtype=np.float32)
    labels = np.full((segment_notes,), -100, dtype=np.int64)
    valid_mask = np.zeros((segment_notes,), dtype=np.float32)
    group_ids = np.full((segment_notes,), -1, dtype=np.int64)
    range_mask = np.zeros((segment_notes, len(VOICE_NAMES)), dtype=np.float32)

    previous_onset = None
    first_group = events[0].onset_group if events else 0

    for i, event in enumerate(events[:segment_notes]):
        pitch_ids[i] = int(np.clip(event.midi_note - 21, 0, 87))
        pitch_classes[i] = int(event.midi_note % 12)
        labels[i] = int(event.voice_idx)
        valid_mask[i] = 1.0
        group_ids[i] = int(event.onset_group - first_group)
        range_mask[i] = range_mask_for_pitch(
            event.midi_note,
            range_mins,
            range_maxs,
            range_margin,
        )

        onset_delta = 0.0 if previous_onset is None else max(0.0, event.onset_time - previous_onset)
        previous_onset = event.onset_time

        rank_den = max(1, event.group_size - 1)
        measure_beats = max(1e-6, event.measure_beats)

        cont_features[i, 0] = (event.midi_note - 21.0) / 87.0
        cont_features[i, 1] = math.log1p(max(event.duration_sec, 0.0))
        cont_features[i, 2] = math.log1p(onset_delta)
        cont_features[i, 3] = event.beat_position / measure_beats
        cont_features[i, 4] = min(event.duration_beats / measure_beats, 4.0)
        cont_features[i, 5] = event.rank_in_group / rank_den
        cont_features[i, 6] = min(event.group_size, 8) / 8.0
        cont_features[i, 7:] = range_costs_for_pitch(
            event.midi_note,
            range_mins,
            range_maxs,
        )

    presence_target = np.zeros((len(VOICE_NAMES),), dtype=np.float32)
    valid_labels = labels[labels >= 0]
    if valid_labels.size:
        presence_target[np.unique(valid_labels)] = 1.0

    return {
        "pitch_ids": pitch_ids,
        "pitch_classes": pitch_classes,
        "cont_features": cont_features,
        "labels": labels,
        "valid_mask": valid_mask,
        "group_ids": group_ids,
        "range_mask": range_mask,
        "presence_target": presence_target,
    }


class YouChoraleMIDIVoiceAssignmentDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        split: str,
        segment_notes: int = 256,
        segment_stride: int = 128,
        onset_quantization_hz: int = 100,
        range_mins: Sequence[float] = DEFAULT_RANGE_MINS,
        range_maxs: Sequence[float] = DEFAULT_RANGE_MAXS,
        range_margin: float = 2.0,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.segment_notes = int(segment_notes)
        self.segment_stride = int(segment_stride)
        self.onset_quantization_hz = int(onset_quantization_hz)
        self.range_mins = np.asarray(range_mins, dtype=np.float32)
        self.range_maxs = np.asarray(range_maxs, dtype=np.float32)
        self.range_margin = float(range_margin)

        self.stems = load_split(self.dataset_dir, split)
        self.song_events: Dict[str, List[NoteEvent]] = {}
        self.segments: List[tuple[str, int, int]] = []

        for stem in self.stems:
            events = build_song_events(self.dataset_dir, stem, self.onset_quantization_hz)
            if not events:
                continue
            self.song_events[stem] = events
            stride = self.segment_stride if split == "train" else self.segment_notes
            for start in range(0, len(events), stride):
                end = min(len(events), start + self.segment_notes)
                if end <= start:
                    continue
                self.segments.append((stem, start, end))
                if end == len(events):
                    break

        if not self.segments:
            raise RuntimeError(f"No note segments found for split={split} in {self.dataset_dir}")

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        stem, start, end = self.segments[index]
        events = self.song_events[stem][start:end]
        sample = build_segment_sample(
            events=events,
            segment_notes=self.segment_notes,
            range_mins=self.range_mins,
            range_maxs=self.range_maxs,
            range_margin=self.range_margin,
        )
        return {key: torch.from_numpy(value) for key, value in sample.items()}


class SymbolicVoiceAssignmentNet(nn.Module):
    def __init__(
        self,
        cont_dim: int = 11,
        pitch_embed_dim: int = 64,
        pitch_class_embed_dim: int = 16,
        feature_hidden_dim: int = 64,
        model_dim: int = 192,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_voices: int = 4,
    ):
        super().__init__()
        self.num_voices = num_voices
        self.pitch_embedding = nn.Embedding(88, pitch_embed_dim)
        self.pitch_class_embedding = nn.Embedding(12, pitch_class_embed_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(cont_dim, feature_hidden_dim),
            nn.LayerNorm(feature_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_hidden_dim, feature_hidden_dim),
            nn.GELU(),
        )
        input_dim = pitch_embed_dim + pitch_class_embed_dim + feature_hidden_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.encoder = nn.LSTM(
            input_size=model_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        encoded_dim = hidden_size * 2
        self.classifier = nn.Linear(encoded_dim, num_voices)
        self.presence_head = nn.Sequential(
            nn.Linear(encoded_dim, encoded_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoded_dim, num_voices),
        )

    def forward(
        self,
        pitch_ids: torch.Tensor,
        pitch_classes: torch.Tensor,
        cont_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pitch_emb = self.pitch_embedding(pitch_ids)
        pc_emb = self.pitch_class_embedding(pitch_classes)
        feat_emb = self.feature_mlp(cont_features)
        x = torch.cat([pitch_emb, pc_emb, feat_emb], dim=-1)
        x = self.input_projection(x)
        encoded, _ = self.encoder(x)
        logits = self.classifier(encoded)

        pooled_mask = valid_mask.unsqueeze(-1)
        pooled = (encoded * pooled_mask).sum(dim=1) / pooled_mask.sum(dim=1).clamp_min(1.0)
        presence_logits = self.presence_head(pooled)
        return {
            "logits": logits,
            "presence_logits": presence_logits,
        }


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    flat_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    )
    flat_mask = valid_mask.reshape(-1)
    return (flat_loss * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)


def range_regularizer(
    logits: torch.Tensor,
    range_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    invalid_mass = probs * (1.0 - range_mask)
    invalid_mass = invalid_mass.sum(dim=-1)
    weighted = invalid_mass * valid_mask
    return weighted.sum() / valid_mask.sum().clamp_min(1.0)


def crossing_regularizer(
    logits: torch.Tensor,
    group_ids: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    voice_order = torch.arange(logits.size(-1), device=logits.device, dtype=probs.dtype)
    expected_voice = (probs * voice_order).sum(dim=-1)

    same_group = (
        (group_ids[:, :-1] >= 0)
        & (group_ids[:, :-1] == group_ids[:, 1:])
        & (valid_mask[:, :-1] > 0.0)
        & (valid_mask[:, 1:] > 0.0)
    )
    if not torch.any(same_group):
        return logits.new_tensor(0.0)

    crossing_penalty = F.relu(expected_voice[:, :-1] - expected_voice[:, 1:])
    crossing_penalty = crossing_penalty * same_group.float()
    return crossing_penalty.sum() / same_group.float().sum().clamp_min(1.0)


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    presence_loss_weight: float,
    range_loss_weight: float,
    crossing_loss_weight: float,
) -> Dict[str, torch.Tensor]:
    note_loss = masked_cross_entropy(outputs["logits"], batch["labels"], batch["valid_mask"])
    presence_loss = F.binary_cross_entropy_with_logits(
        outputs["presence_logits"],
        batch["presence_target"],
    )
    range_loss = range_regularizer(outputs["logits"], batch["range_mask"], batch["valid_mask"])
    crossing_loss = crossing_regularizer(outputs["logits"], batch["group_ids"], batch["valid_mask"])
    total_loss = (
        note_loss
        + presence_loss_weight * presence_loss
        + range_loss_weight * range_loss
        + crossing_loss_weight * crossing_loss
    )
    return {
        "loss": total_loss,
        "note_loss": note_loss.detach(),
        "presence_loss": presence_loss.detach(),
        "range_loss": range_loss.detach(),
        "crossing_loss": crossing_loss.detach(),
    }


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def init_confusion(num_classes: int = 4) -> np.ndarray:
    return np.zeros((num_classes, num_classes), dtype=np.int64)


def update_confusion(confusion: np.ndarray, labels: np.ndarray, preds: np.ndarray) -> None:
    for target, pred in zip(labels, preds):
        confusion[target, pred] += 1


def metrics_from_confusion(confusion: np.ndarray) -> Dict[str, float]:
    eps = 1e-8
    totals = confusion.sum()
    accuracy = float(np.trace(confusion) / max(1, totals))
    macro_f1 = 0.0
    metrics: Dict[str, float] = {"note_assignment_accuracy": accuracy}

    for voice_idx, voice_name in enumerate(VOICE_NAMES):
        tp = float(confusion[voice_idx, voice_idx])
        fp = float(confusion[:, voice_idx].sum() - tp)
        fn = float(confusion[voice_idx, :].sum() - tp)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        metrics[f"{voice_name}_precision"] = precision
        metrics[f"{voice_name}_recall"] = recall
        metrics[f"{voice_name}_f1"] = f1
        macro_f1 += f1

    metrics["macro_note_f1"] = macro_f1 / len(VOICE_NAMES)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    presence_loss_weight: float,
    range_loss_weight: float,
    crossing_loss_weight: float,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.eval()
    confusion = init_confusion(len(VOICE_NAMES))
    total_loss = 0.0
    total_presence_correct = 0.0
    total_presence_elements = 0.0
    steps = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        outputs = model(
            batch["pitch_ids"],
            batch["pitch_classes"],
            batch["cont_features"],
            batch["valid_mask"],
        )
        loss_dict = compute_loss(
            outputs,
            batch,
            presence_loss_weight=presence_loss_weight,
            range_loss_weight=range_loss_weight,
            crossing_loss_weight=crossing_loss_weight,
        )
        total_loss += float(loss_dict["loss"].item())
        steps += 1

        preds = torch.argmax(outputs["logits"], dim=-1)
        valid_mask = batch["valid_mask"] > 0.0
        labels_np = batch["labels"][valid_mask].detach().cpu().numpy()
        preds_np = preds[valid_mask].detach().cpu().numpy()
        update_confusion(confusion, labels_np, preds_np)

        presence_pred = (torch.sigmoid(outputs["presence_logits"]) >= 0.5).float()
        total_presence_correct += float((presence_pred == batch["presence_target"]).sum().item())
        total_presence_elements += float(batch["presence_target"].numel())

    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / max(1, steps)
    metrics["presence_accuracy"] = total_presence_correct / max(1.0, total_presence_elements)
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    presence_loss_weight: float,
    range_loss_weight: float,
    crossing_loss_weight: float,
    grad_clip: float,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_note_loss = 0.0
    total_presence_loss = 0.0
    total_range_loss = 0.0
    total_crossing_loss = 0.0
    steps = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch["pitch_ids"],
            batch["pitch_classes"],
            batch["cont_features"],
            batch["valid_mask"],
        )
        loss_dict = compute_loss(
            outputs,
            batch,
            presence_loss_weight=presence_loss_weight,
            range_loss_weight=range_loss_weight,
            crossing_loss_weight=crossing_loss_weight,
        )
        loss_dict["loss"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss_dict["loss"].item())
        total_note_loss += float(loss_dict["note_loss"].item())
        total_presence_loss += float(loss_dict["presence_loss"].item())
        total_range_loss += float(loss_dict["range_loss"].item())
        total_crossing_loss += float(loss_dict["crossing_loss"].item())
        steps += 1

    return {
        "loss": total_loss / max(1, steps),
        "note_loss": total_note_loss / max(1, steps),
        "presence_loss": total_presence_loss / max(1, steps),
        "range_loss": total_range_loss / max(1, steps),
        "crossing_loss": total_crossing_loss / max(1, steps),
    }


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_metric: float,
    args: argparse.Namespace,
) -> None:
    create_folder(checkpoint_path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_metric": best_metric,
            "args": vars(args),
        },
        checkpoint_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a symbolic MIDI voice-assignment network on YouChorale.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./data/YouChorale",
    )
    parser.add_argument("--workspace", type=str, default="./workspaces")
    parser.add_argument("--experiment-name", type=str, default="youchorale_midi_voice_assignment")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--valid-split", type=str, default="valid")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--segment-notes", type=int, default=256)
    parser.add_argument("--segment-stride", type=int, default=128)
    parser.add_argument("--onset-quantization-hz", type=int, default=100)
    parser.add_argument("--pitch-embed-dim", type=int, default=64)
    parser.add_argument("--pitch-class-embed-dim", type=int, default=16)
    parser.add_argument("--feature-hidden-dim", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=192)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--presence-loss-weight", type=float, default=0.1)
    parser.add_argument("--range-loss-weight", type=float, default=0.02)
    parser.add_argument("--crossing-loss-weight", type=float, default=0.05)
    parser.add_argument("--range-margin", type=float, default=2.0)
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
    experiment_dir = workspace / "midi_voice_assignment" / args.experiment_name
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

    train_dataset = YouChoraleMIDIVoiceAssignmentDataset(
        dataset_dir=args.dataset_dir,
        split=args.train_split,
        segment_notes=args.segment_notes,
        segment_stride=args.segment_stride,
        onset_quantization_hz=args.onset_quantization_hz,
        range_margin=args.range_margin,
    )
    valid_dataset = YouChoraleMIDIVoiceAssignmentDataset(
        dataset_dir=args.dataset_dir,
        split=args.valid_split,
        segment_notes=args.segment_notes,
        segment_stride=args.segment_stride,
        onset_quantization_hz=args.onset_quantization_hz,
        range_margin=args.range_margin,
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

    model = SymbolicVoiceAssignmentNet(
        pitch_embed_dim=args.pitch_embed_dim,
        pitch_class_embed_dim=args.pitch_class_embed_dim,
        feature_hidden_dim=args.feature_hidden_dim,
        model_dim=args.model_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_voices=len(VOICE_NAMES),
    ).to(device)
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
    if args.resume:
        resume_path = Path(args.resume)
        logging.info("Resuming from %s", resume_path)
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))

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
            model,
            train_loader,
            optimizer,
            device,
            presence_loss_weight=args.presence_loss_weight,
            range_loss_weight=args.range_loss_weight,
            crossing_loss_weight=args.crossing_loss_weight,
            grad_clip=args.grad_clip,
            max_batches=limit_train,
        )
        valid_stats = evaluate(
            model,
            valid_loader,
            device,
            presence_loss_weight=args.presence_loss_weight,
            range_loss_weight=args.range_loss_weight,
            crossing_loss_weight=args.crossing_loss_weight,
            max_batches=limit_valid,
        )
        scheduler.step(valid_stats["macro_note_f1"])

        lr = optimizer.param_groups[0]["lr"]
        logging.info(
            "Epoch %03d | lr=%.6f | train_loss=%.4f | valid_loss=%.4f | "
            "valid_acc=%.4f | valid_macro_f1=%.4f | presence_acc=%.4f",
            epoch,
            lr,
            train_stats["loss"],
            valid_stats["loss"],
            valid_stats["note_assignment_accuracy"],
            valid_stats["macro_note_f1"],
            valid_stats["presence_accuracy"],
        )
        logging.info(
            "Valid per-voice F1 | S=%.4f A=%.4f T=%.4f B=%.4f",
            valid_stats["S_f1"],
            valid_stats["A_f1"],
            valid_stats["T_f1"],
            valid_stats["B_f1"],
        )

        writer.add_scalar("train/loss", train_stats["loss"], epoch)
        writer.add_scalar("train/note_loss", train_stats["note_loss"], epoch)
        writer.add_scalar("train/presence_loss", train_stats["presence_loss"], epoch)
        writer.add_scalar("train/range_loss", train_stats["range_loss"], epoch)
        writer.add_scalar("train/crossing_loss", train_stats["crossing_loss"], epoch)
        for key, value in valid_stats.items():
            writer.add_scalar(f"valid/{key}", value, epoch)
        writer.add_scalar("train/lr", lr, epoch)
        writer.flush()

        if valid_stats["macro_note_f1"] > best_metric:
            best_metric = valid_stats["macro_note_f1"]
            save_checkpoint(
                checkpoints_dir / "best_macro_f1.pth",
                epoch,
                model,
                optimizer,
                best_metric,
                args,
            )
        save_checkpoint(
            checkpoints_dir / "last.pth",
            epoch,
            model,
            optimizer,
            best_metric,
            args,
        )

    writer.close()
    logging.info("Training finished. Best valid macro_note_f1=%.4f", best_metric)


if __name__ == "__main__":
    main()
