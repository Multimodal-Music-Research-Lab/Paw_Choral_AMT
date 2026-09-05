from __future__ import annotations

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def parse_args():
    parser = argparse.ArgumentParser(description="Create a clean GT vs prediction piano-roll figure from probs/*.pkl.")
    parser.add_argument("--pkl_path", type=str, required=True, help="Path to a probs pkl file.")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the figure.")
    parser.add_argument("--frame_threshold", type=float, default=0.1, help="Threshold for predicted frame roll.")
    parser.add_argument("--max_frames", type=int, default=1200, help="Max displayed time bins after downsampling.")
    parser.add_argument("--title", type=str, default=None, help="Optional custom title.")
    return parser.parse_args()


def pool_time(arr: np.ndarray, max_frames: int) -> np.ndarray:
    frames = arr.shape[0]
    if frames <= max_frames:
        return arr
    scale = int(np.ceil(frames / max_frames))
    pad = (-frames) % scale
    if pad:
        arr = np.pad(arr, ((0, pad), (0, 0)), mode="constant")
    arr = arr.reshape(-1, scale, arr.shape[1])
    return arr.max(axis=1)


def prepare_roll(arr: np.ndarray, max_frames: int) -> np.ndarray:
    arr = pool_time(arr, max_frames)
    return arr.T[::-1, :]


def make_cmap(hex_color: str):
    return LinearSegmentedColormap.from_list("", ["#ffffff", hex_color])


def add_panel(ax, arr, cmap, title, ytick_labels):
    ax.imshow(arr, aspect="auto", origin="upper", cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(title, loc="left", fontsize=12, fontweight="semibold", pad=8)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d9d9d9")
    ax.spines["bottom"].set_color("#d9d9d9")
    ax.tick_params(axis="both", colors="#555555", labelsize=9)
    ax.set_ylabel("Pitch", fontsize=10, color="#444444")
    ax.set_yticks([0, 21, 43, 65, 87])
    ax.set_yticklabels(ytick_labels)


def main():
    args = parse_args()

    with open(args.pkl_path, "rb") as f:
        data = pickle.load(f)

    gt = np.asarray(data["frame_roll"], dtype=np.float32)
    pred = (np.asarray(data["frame_output"], dtype=np.float32) > args.frame_threshold).astype(np.float32)

    gt_roll = prepare_roll(gt, args.max_frames)
    pred_roll = prepare_roll(pred, args.max_frames)
    diff_roll = np.clip(pred_roll - gt_roll, 0.0, 1.0)

    title = args.title or os.path.splitext(os.path.basename(args.pkl_path))[0]
    ytick_labels = ["108", "87", "65", "43", "21"]

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
        }
    )

    fig = plt.figure(figsize=(12.5, 6.8), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.65])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    add_panel(ax1, gt_roll, make_cmap("#1f1f1f"), "Ground Truth", ytick_labels)
    add_panel(ax2, pred_roll, make_cmap("#1f7a4f"), f"Prediction (frame > {args.frame_threshold})", ytick_labels)
    add_panel(ax3, diff_roll, make_cmap("#d95f02"), "Extra Predicted Activity", ytick_labels)

    for ax in [ax1, ax2]:
        ax.set_xticklabels([])
    ax3.set_xlabel("Time (downsampled frame bins)", fontsize=10, color="#444444")

    gt_density = float(gt.mean())
    pred_density = float(pred.mean())
    fig.suptitle(title, fontsize=16, fontweight="bold", x=0.06, ha="left", color="#222222")
    fig.text(
        0.06,
        0.965,
        f"GT density: {gt_density:.3f}    Pred density: {pred_density:.3f}",
        ha="left",
        va="top",
        fontsize=10,
        color="#666666",
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fig.savefig(args.output_path, dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(args.output_path)


if __name__ == "__main__":
    main()
