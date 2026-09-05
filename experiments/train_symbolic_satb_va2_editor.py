from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from train_midi_voice_assignment import VOICE_NAMES, create_folder, seed_everything, setup_logging
from train_symbolic_satb_editor import (
    CLASSES_NUM,
    YouChoraleSymbolicSATBEditorDataset,
    frame_f1_from_probs,
)


DEFAULT_RANGE_MINS = np.array([60, 55, 48, 40], dtype=np.int64)
DEFAULT_RANGE_MAXS = np.array([88, 79, 72, 67], dtype=np.int64)


def build_range_mask(
    range_mins: Sequence[int] = DEFAULT_RANGE_MINS,
    range_maxs: Sequence[int] = DEFAULT_RANGE_MAXS,
) -> torch.Tensor:
    mask = np.zeros((len(VOICE_NAMES), CLASSES_NUM), dtype=np.float32)
    for voice_idx, (lo, hi) in enumerate(zip(range_mins, range_maxs)):
        begin = max(0, int(lo) - 21)
        end = min(CLASSES_NUM, int(hi) - 21 + 1)
        if end > begin:
            mask[voice_idx, begin:end] = 1.0
    return torch.from_numpy(mask)


class ConvBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SymbolicSATBVA2EditorNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 64,
        assignment_hidden_channels: int = 64,
        dropout: float = 0.2,
        temperature: float = 1.0,
        num_voices: int = 4,
        classes_num: int = CLASSES_NUM,
        use_presence_head: bool = True,
    ):
        super().__init__()
        self.num_voices = int(num_voices)
        self.classes_num = int(classes_num)
        self.temperature = float(temperature)
        self.use_presence_head = bool(use_presence_head)

        self.shared_stem = nn.Sequential(
            ConvBlock2d(input_channels, hidden_channels, dropout=dropout),
            ConvBlock2d(hidden_channels, hidden_channels, dropout=dropout),
        )
        self.shared_onset_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.shared_frame_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.shared_offset_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        assignment_in_channels = hidden_channels + 3
        self.assignment_net = nn.Sequential(
            ConvBlock2d(assignment_in_channels, assignment_hidden_channels, dropout=dropout),
            nn.Conv2d(assignment_hidden_channels, self.num_voices, kernel_size=1),
        )

        if self.use_presence_head:
            self.presence_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, self.num_voices),
            )
        else:
            self.presence_head = None

    def forward(self, input_rolls: torch.Tensor) -> Dict[str, torch.Tensor]:
        # [B, T, C, P] -> [B, C, T, P]
        x = input_rolls.permute(0, 2, 1, 3)
        shared_features = self.shared_stem(x)

        shared_onset_logits = self.shared_onset_head(shared_features).squeeze(1)
        shared_frame_logits = self.shared_frame_head(shared_features).squeeze(1)
        shared_offset_logits = self.shared_offset_head(shared_features).squeeze(1)

        shared_onset_probs = torch.sigmoid(shared_onset_logits)
        shared_frame_probs = torch.sigmoid(shared_frame_logits)
        shared_offset_probs = torch.sigmoid(shared_offset_logits)

        assignment_input = torch.cat(
            [
                shared_features,
                shared_onset_probs.unsqueeze(1),
                shared_frame_probs.unsqueeze(1),
                shared_offset_probs.unsqueeze(1),
            ],
            dim=1,
        )
        assignment_logits = self.assignment_net(assignment_input)
        assignment_probs = torch.softmax(assignment_logits / max(self.temperature, 1e-6), dim=1)

        if self.use_presence_head:
            presence_logits = self.presence_head(shared_features)
            presence_gate = torch.sigmoid(presence_logits).unsqueeze(-1).unsqueeze(-1)
        else:
            presence_logits = None
            presence_gate = 1.0

        voice_onset_probs = shared_onset_probs.unsqueeze(1) * assignment_probs * presence_gate
        voice_frame_probs = shared_frame_probs.unsqueeze(1) * assignment_probs * presence_gate
        voice_offset_probs = shared_offset_probs.unsqueeze(1) * assignment_probs * presence_gate

        return {
            "shared_onset_probs": shared_onset_probs,
            "shared_frame_probs": shared_frame_probs,
            "shared_offset_probs": shared_offset_probs,
            "assignment_logits": assignment_logits,
            "assignment_probs": assignment_probs,
            "voice_onset_probs": voice_onset_probs.permute(0, 2, 1, 3),
            "voice_frame_probs": voice_frame_probs.permute(0, 2, 1, 3),
            "voice_offset_probs": voice_offset_probs.permute(0, 2, 1, 3),
            "presence_logits": presence_logits,
        }


def weighted_bce_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    loss = F.binary_cross_entropy(probs.clamp(1e-6, 1.0 - 1e-6), targets, reduction="none")
    weights = 1.0 + (positive_weight - 1.0) * targets
    return (loss * weights).mean()


def assignment_entropy_loss(assignment_probs: torch.Tensor) -> torch.Tensor:
    entropy = -(assignment_probs.clamp_min(1e-8) * torch.log(assignment_probs.clamp_min(1e-8))).sum(dim=1)
    return entropy.mean()


def assignment_range_loss(assignment_probs: torch.Tensor, range_mask: torch.Tensor) -> torch.Tensor:
    invalid_mass = assignment_probs * (1.0 - range_mask)
    return invalid_mass.mean()


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    range_mask: torch.Tensor,
    voice_frame_loss_weight: float,
    voice_onset_loss_weight: float,
    voice_offset_loss_weight: float,
    union_frame_loss_weight: float,
    union_onset_loss_weight: float,
    union_offset_loss_weight: float,
    presence_loss_weight: float,
    assignment_entropy_loss_weight: float,
    assignment_range_loss_weight: float,
    frame_positive_weight: float,
    onset_positive_weight: float,
    offset_positive_weight: float,
) -> Dict[str, torch.Tensor]:
    voice_frame_loss = weighted_bce_probs(
        outputs["voice_frame_probs"],
        batch["voice_frame_roll"],
        positive_weight=frame_positive_weight,
    )
    voice_onset_loss = weighted_bce_probs(
        outputs["voice_onset_probs"],
        batch["voice_onset_roll"],
        positive_weight=onset_positive_weight,
    )
    voice_offset_loss = weighted_bce_probs(
        outputs["voice_offset_probs"],
        batch["voice_offset_roll"],
        positive_weight=offset_positive_weight,
    )
    union_frame_loss = weighted_bce_probs(
        outputs["shared_frame_probs"],
        batch["frame_roll"],
        positive_weight=frame_positive_weight,
    )
    union_onset_loss = weighted_bce_probs(
        outputs["shared_onset_probs"],
        batch["onset_roll"],
        positive_weight=onset_positive_weight,
    )
    union_offset_loss = weighted_bce_probs(
        outputs["shared_offset_probs"],
        batch["offset_roll"],
        positive_weight=offset_positive_weight,
    )

    if outputs["presence_logits"] is not None:
        presence_loss = F.binary_cross_entropy_with_logits(outputs["presence_logits"], batch["voice_presence"])
    else:
        presence_loss = outputs["voice_frame_probs"].new_tensor(0.0)

    assignment_entropy = assignment_entropy_loss(outputs["assignment_probs"])
    assignment_range = assignment_range_loss(outputs["assignment_probs"], range_mask)

    total_loss = (
        voice_frame_loss_weight * voice_frame_loss
        + voice_onset_loss_weight * voice_onset_loss
        + voice_offset_loss_weight * voice_offset_loss
        + union_frame_loss_weight * union_frame_loss
        + union_onset_loss_weight * union_onset_loss
        + union_offset_loss_weight * union_offset_loss
        + presence_loss_weight * presence_loss
        + assignment_entropy_loss_weight * assignment_entropy
        + assignment_range_loss_weight * assignment_range
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
        "assignment_entropy_loss": assignment_entropy.detach(),
        "assignment_range_loss": assignment_range.detach(),
    }


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    range_mask: torch.Tensor,
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
        "assignment_entropy_loss": 0.0,
        "assignment_range_loss": 0.0,
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
            range_mask=range_mask,
            voice_frame_loss_weight=args.voice_frame_loss_weight,
            voice_onset_loss_weight=args.voice_onset_loss_weight,
            voice_offset_loss_weight=args.voice_offset_loss_weight,
            union_frame_loss_weight=args.union_frame_loss_weight,
            union_onset_loss_weight=args.union_onset_loss_weight,
            union_offset_loss_weight=args.union_offset_loss_weight,
            presence_loss_weight=args.presence_loss_weight,
            assignment_entropy_loss_weight=args.assignment_entropy_loss_weight,
            assignment_range_loss_weight=args.assignment_range_loss_weight,
            frame_positive_weight=args.frame_positive_weight,
            onset_positive_weight=args.onset_positive_weight,
            offset_positive_weight=args.offset_positive_weight,
        )
        for key in totals:
            totals[key] += float(loss_dict[key].item())

        voice_frame_f1 += frame_f1_from_probs(
            outputs["voice_frame_probs"],
            batch["voice_frame_roll"],
            threshold=args.eval_frame_threshold,
        )
        union_frame_f1 += frame_f1_from_probs(
            outputs["shared_frame_probs"],
            batch["frame_roll"],
            threshold=args.eval_frame_threshold,
        )
        if outputs["presence_logits"] is not None:
            presence_pred = (torch.sigmoid(outputs["presence_logits"]) >= 0.5).float()
            presence_correct += float((presence_pred == batch["voice_presence"]).sum().item())
            presence_total += float(batch["voice_presence"].numel())
        steps += 1

    metrics = {key: value / max(1, steps) for key, value in totals.items()}
    metrics["voice_frame_f1"] = voice_frame_f1 / max(1, steps)
    metrics["union_frame_f1"] = union_frame_f1 / max(1, steps)
    metrics["presence_accuracy"] = presence_correct / max(1.0, presence_total) if presence_total > 0 else 0.0
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    range_mask: torch.Tensor,
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
        "assignment_entropy_loss": 0.0,
        "assignment_range_loss": 0.0,
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
            range_mask=range_mask,
            voice_frame_loss_weight=args.voice_frame_loss_weight,
            voice_onset_loss_weight=args.voice_onset_loss_weight,
            voice_offset_loss_weight=args.voice_offset_loss_weight,
            union_frame_loss_weight=args.union_frame_loss_weight,
            union_onset_loss_weight=args.union_onset_loss_weight,
            union_offset_loss_weight=args.union_offset_loss_weight,
            presence_loss_weight=args.presence_loss_weight,
            assignment_entropy_loss_weight=args.assignment_entropy_loss_weight,
            assignment_range_loss_weight=args.assignment_range_loss_weight,
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


def build_model_from_args(args: argparse.Namespace) -> SymbolicSATBVA2EditorNet:
    return SymbolicSATBVA2EditorNet(
        input_channels=3,
        hidden_channels=args.hidden_channels,
        assignment_hidden_channels=args.assignment_hidden_channels,
        dropout=args.dropout,
        temperature=args.assignment_temperature,
        num_voices=len(VOICE_NAMES),
        classes_num=CLASSES_NUM,
        use_presence_head=args.use_presence_head,
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
        "shared_onset_probs": np.zeros((frames_num, CLASSES_NUM), dtype=np.float32),
        "shared_frame_probs": np.zeros((frames_num, CLASSES_NUM), dtype=np.float32),
        "shared_offset_probs": np.zeros((frames_num, CLASSES_NUM), dtype=np.float32),
        "assignment_probs": np.zeros((frames_num, len(VOICE_NAMES), CLASSES_NUM), dtype=np.float32),
    }
    counts = np.zeros((frames_num, 1), dtype=np.float32)
    presence_logits_list = []

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
        accum["shared_onset_probs"][start:end] += outputs["shared_onset_probs"][0, :valid_len].detach().cpu().numpy()
        accum["shared_frame_probs"][start:end] += outputs["shared_frame_probs"][0, :valid_len].detach().cpu().numpy()
        accum["shared_offset_probs"][start:end] += outputs["shared_offset_probs"][0, :valid_len].detach().cpu().numpy()
        accum["assignment_probs"][start:end] += outputs["assignment_probs"][0, :, :valid_len, :].permute(1, 0, 2).detach().cpu().numpy()
        counts[start:end] += 1.0
        if outputs["presence_logits"] is not None:
            presence_logits_list.append(outputs["presence_logits"][0].detach().cpu().numpy())

        if end == frames_num:
            break

    counts = np.clip(counts, 1.0, None)
    averaged = {
        "voice_onset_probs": accum["voice_onset_probs"] / counts[:, None, :],
        "voice_frame_probs": accum["voice_frame_probs"] / counts[:, None, :],
        "voice_offset_probs": accum["voice_offset_probs"] / counts[:, None, :],
        "shared_onset_probs": accum["shared_onset_probs"] / counts,
        "shared_frame_probs": accum["shared_frame_probs"] / counts,
        "shared_offset_probs": accum["shared_offset_probs"] / counts,
        "assignment_probs": accum["assignment_probs"] / counts[:, None, :],
    }
    if presence_logits_list:
        averaged["voice_presence_probs"] = 1.0 / (1.0 + np.exp(-np.mean(np.stack(presence_logits_list, axis=0), axis=0)))
    return averaged


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a VA-2 inspired symbolic MIDI-to-SATB editor with shared salience and voice-assignment masks.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./data/YouChorale",
    )
    parser.add_argument("--workspace", type=str, default="./workspaces")
    parser.add_argument("--experiment-name", type=str, default="youchorale_symbolic_satb_va2_editor")
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
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--assignment-hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--assignment-temperature", type=float, default=1.0)
    parser.add_argument("--voice-frame-loss-weight", type=float, default=1.0)
    parser.add_argument("--voice-onset-loss-weight", type=float, default=2.0)
    parser.add_argument("--voice-offset-loss-weight", type=float, default=1.0)
    parser.add_argument("--union-frame-loss-weight", type=float, default=1.0)
    parser.add_argument("--union-onset-loss-weight", type=float, default=2.0)
    parser.add_argument("--union-offset-loss-weight", type=float, default=1.0)
    parser.add_argument("--presence-loss-weight", type=float, default=0.1)
    parser.add_argument("--assignment-entropy-loss-weight", type=float, default=0.01)
    parser.add_argument("--assignment-range-loss-weight", type=float, default=0.05)
    parser.add_argument("--frame-positive-weight", type=float, default=6.0)
    parser.add_argument("--onset-positive-weight", type=float, default=8.0)
    parser.add_argument("--offset-positive-weight", type=float, default=8.0)
    parser.add_argument("--eval-frame-threshold", type=float, default=0.10)
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
    parser.add_argument("--clean-valid-input", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-presence-head", action=argparse.BooleanOptionalAction, default=True)
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
        clean_input_prob=1.0 if args.clean_valid_input else 0.0,
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

    range_mask = build_range_mask().to(device).unsqueeze(0).unsqueeze(2)

    start_epoch = 1
    best_metric = -float("inf")
    metric_name = "voice_frame_f1"
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
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
            range_mask=range_mask,
            grad_clip=args.grad_clip,
            max_batches=limit_train,
        )
        valid_stats = evaluate(
            model=model,
            loader=valid_loader,
            device=device,
            args=args,
            range_mask=range_mask,
            max_batches=limit_valid,
        )
        scheduler.step(valid_stats["voice_frame_f1"])

        lr = optimizer.param_groups[0]["lr"]
        logging.info(
            "Epoch %03d | lr=%.6f | train_loss=%.4f | valid_loss=%.4f | "
            "voice_frame_f1=%.4f | union_frame_f1=%.4f | presence_acc=%.4f | "
            "assign_entropy=%.4f | assign_range=%.4f",
            epoch,
            lr,
            train_stats["loss"],
            valid_stats["loss"],
            valid_stats["voice_frame_f1"],
            valid_stats["union_frame_f1"],
            valid_stats["presence_accuracy"],
            valid_stats["assignment_entropy_loss"],
            valid_stats["assignment_range_loss"],
        )

        for key, value in train_stats.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in valid_stats.items():
            writer.add_scalar(f"valid/{key}", value, epoch)
        writer.add_scalar("train/lr", lr, epoch)
        writer.flush()

        if valid_stats["voice_frame_f1"] > best_metric:
            best_metric = valid_stats["voice_frame_f1"]
            save_checkpoint(
                checkpoint_path=checkpoints_dir / "best_voice_frame_f1.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_metric=best_metric,
                metric_name=metric_name,
                args=args,
            )
        save_checkpoint(
            checkpoint_path=checkpoints_dir / "last.pth",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_metric=best_metric,
            metric_name=metric_name,
            args=args,
        )

    writer.close()
    logging.info("Training finished. Best %s=%.4f", metric_name, best_metric)


if __name__ == "__main__":
    main()
