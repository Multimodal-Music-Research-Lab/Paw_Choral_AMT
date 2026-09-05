from __future__ import annotations

import os
import pickle
import sys
from copy import deepcopy

import mir_eval
import numpy as np
from hydra import compose, initialize
from sklearn import metrics as sk_metrics

from calculate_scores import ScoreCalculator, build_post_processor
from utilities import get_filename, get_model_name, note_to_freq


VOICE_NAMES = ["S", "A", "T", "B"]
VOICE_TO_INDEX = {name: idx for idx, name in enumerate(VOICE_NAMES)}
FIXED_ONSET_TOLERANCES = (0.05, 0.10)


def _tolerance_tag(onset_tolerance):
    return f"{int(round(onset_tolerance * 1000.0))}ms"


def _dataset_note_dir(cfg):
    if cfg.dataset.test_set == "youchorale":
        return os.path.join(cfg.dataset.youchorale_dir, "note")
    if cfg.dataset.test_set == "youchorale_pro":
        return os.path.join(cfg.dataset.youchorale_pro_dir, "note")
    if cfg.dataset.test_set == "csd":
        return os.path.join(cfg.dataset.csd_dir, "note")
    if cfg.dataset.test_set == "cantoria":
        return os.path.join(cfg.dataset.cantoria_dir, f"note_{cfg.dataset.cantoria_f0_source}")
    raise ValueError(f"Unsupported choral dataset: {cfg.dataset.test_set}")


def _load_note_bars(note_path):
    with open(note_path, "rb") as f:
        return pickle.load(f)


def _safe_intervals(intervals):
    intervals = np.asarray(intervals, dtype=np.float32)
    if intervals.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    bad = np.nonzero(intervals[:, 1] <= intervals[:, 0])[0]
    intervals = intervals.copy()
    intervals[bad, 1] = intervals[bad, 0] + 1e-4
    return intervals


def _reference_events_by_voice(note_bars):
    voice_map = {name: [] for name in VOICE_NAMES}
    for bar in note_bars:
        if not isinstance(bar, dict):
            continue
        for part_name, note_list in bar.items():
            if part_name == "measure" or not part_name:
                continue
            voice_name = part_name[0].upper()
            if voice_name not in voice_map:
                continue
            for note in note_list:
                if len(note) < 5:
                    continue
                onset_time = float(note[3])
                offset_time = float(note[4])
                if offset_time <= onset_time:
                    offset_time = onset_time + 1e-4
                voice_map[voice_name].append(
                    {
                        "midi_note": int(note[0]),
                        "onset_time": onset_time,
                        "offset_time": offset_time,
                    }
                )
    return voice_map


def _reference_presence_vector(note_bars):
    ref_voice_events = _reference_events_by_voice(note_bars)
    return np.asarray(
        [1.0 if len(ref_voice_events[voice_name]) > 0 else 0.0 for voice_name in VOICE_NAMES],
        dtype=np.float32,
    )


def _events_to_arrays(events):
    if len(events) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    intervals = np.array([[e["onset_time"], e["offset_time"]] for e in events], dtype=np.float32)
    pitches = np.array([e["midi_note"] for e in events], dtype=np.int32)
    intervals = _safe_intervals(intervals)
    sort_idx = np.argsort(intervals[:, 0], kind='mergesort')
    return intervals[sort_idx], pitches[sort_idx]


def _correct_onset_metrics(ref_intervals, est_intervals, onset_tolerance):
    ref_onsets = ref_intervals[:, 0] if len(ref_intervals) else np.zeros((0,), dtype=np.float32)
    est_onsets = est_intervals[:, 0] if len(est_intervals) else np.zeros((0,), dtype=np.float32)
    ref_onsets = np.sort(ref_onsets)
    est_onsets = np.sort(est_onsets)
    con_f1, con_precision, con_recall = mir_eval.onset.f_measure(
        reference_onsets=ref_onsets,
        estimated_onsets=est_onsets,
        window=onset_tolerance,
    )
    return con_precision, con_recall, con_f1


def _frame_metrics(y_true, y_pred, mask=None, threshold=0.5):
    if y_true.size == 0 or y_pred.size == 0:
        return {}
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    if mask is not None:
        valid = mask.flatten() > 0
        min_len = min(len(y_true), len(y_pred), len(valid))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        valid = valid[:min_len]
        if not np.any(valid):
            return {}
        y_true = y_true[valid]
        y_pred = y_pred[valid]
    y_hat = (y_pred >= threshold).astype(np.float32)
    precision, recall, f1, _ = sk_metrics.precision_recall_fscore_support(
        y_true,
        y_hat,
        average="binary",
        zero_division=0,
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _decode_voice_events(total_dict, voice_idx, post_processor, thresholds=None):
    if thresholds is not None:
        post_processor = deepcopy(post_processor)
        post_processor.frame_threshold = thresholds["frame_threshold"]
        post_processor.onset_threshold = thresholds["onset_threshold"]
        post_processor.offset_threshold = thresholds["offset_threshold"]
    post_input = {}
    if "voice_frame_output" in total_dict:
        post_input["frame_output"] = total_dict["voice_frame_output"][:, voice_idx, :]
    if "voice_onset_output" in total_dict:
        post_input["onset_output"] = total_dict["voice_onset_output"][:, voice_idx, :]
    if "voice_offset_output" in total_dict:
        post_input["offset_output"] = total_dict["voice_offset_output"][:, voice_idx, :]
    note_events, _ = post_processor.output_dict_to_midi_events(deepcopy(post_input))
    return note_events


def _normalize_voice_thresholds(values, default_value):
    if values is None:
        return [float(default_value)] * len(VOICE_NAMES)
    values = list(values)
    if len(values) != len(VOICE_NAMES):
        raise ValueError(f"Expected {len(VOICE_NAMES)} thresholds, got {len(values)}")
    return [float(v) for v in values]


def _cfg_voice_thresholds(cfg):
    if getattr(cfg.choral, "use_per_voice_thresholds", False):
        frame_thresholds = _normalize_voice_thresholds(
            getattr(cfg.choral, "voice_frame_thresholds", None),
            cfg.post.frame_threshold,
        )
        onset_thresholds = _normalize_voice_thresholds(
            getattr(cfg.choral, "voice_onset_thresholds", None),
            cfg.post.onset_threshold,
        )
        offset_thresholds = _normalize_voice_thresholds(
            getattr(cfg.choral, "voice_offset_thresholds", None),
            cfg.post.offset_threshold,
        )
    else:
        frame_thresholds = [float(cfg.post.frame_threshold)] * len(VOICE_NAMES)
        onset_thresholds = [float(cfg.post.onset_threshold)] * len(VOICE_NAMES)
        offset_thresholds = [float(cfg.post.offset_threshold)] * len(VOICE_NAMES)

    return {
        voice_name: {
            "frame_threshold": frame_thresholds[idx],
            "onset_threshold": onset_thresholds[idx],
            "offset_threshold": offset_thresholds[idx],
        }
        for idx, voice_name in enumerate(VOICE_NAMES)
    }


class ChoralScoreCalculator:
    def __init__(self, cfg, voice_thresholds=None):
        self.cfg = cfg
        self.model_name = get_model_name(cfg)
        self.eval_split = str(getattr(cfg.dataset, "eval_split", "validation"))
        if self.eval_split not in {"validation", "test"}:
            raise ValueError("dataset.eval_split must be 'validation' or 'test'")
        self.probs_dir = os.path.join(
            cfg.exp.workspace,
            "probs",
            cfg.dataset.test_set,
            self.eval_split,
            self.model_name,
            f"{cfg.exp.ckpt_iteration}_iteration",
        )
        self.note_dir = _dataset_note_dir(cfg)
        self.post_processor = build_post_processor(cfg)
        self.voice_thresholds = voice_thresholds or _cfg_voice_thresholds(cfg)

        if not os.path.isdir(self.probs_dir):
            raise FileNotFoundError(f"Missing probs dir: {self.probs_dir}")

    def calculate_voice_score_per_song(self, prob_path, note_path, voice_name, thresholds=None):
        voice_idx = VOICE_TO_INDEX[voice_name]
        with open(prob_path, "rb") as f:
            total_dict = pickle.load(f)

        ref_voice_events = _reference_events_by_voice(_load_note_bars(note_path))
        est_events = _decode_voice_events(
            total_dict,
            voice_idx,
            self.post_processor,
            thresholds=thresholds or self.voice_thresholds[voice_name],
        )
        ref_intervals, ref_pitches = _events_to_arrays(ref_voice_events[voice_name])
        est_intervals, est_pitches = _events_to_arrays(est_events)

        con_precision, con_recall, con_f1 = _correct_onset_metrics(
            ref_intervals,
            est_intervals,
            onset_tolerance=self.cfg.score.onset_tolerance,
        )
        conp_precision, conp_recall, conp_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=note_to_freq(ref_pitches),
            est_intervals=est_intervals,
            est_pitches=note_to_freq(est_pitches),
            onset_tolerance=self.cfg.score.onset_tolerance,
            offset_ratio=None,
            offset_min_tolerance=self.cfg.score.offset_min_tolerance,
        )
        conpoff_precision, conpoff_recall, conpoff_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=note_to_freq(ref_pitches),
            est_intervals=est_intervals,
            est_pitches=note_to_freq(est_pitches),
            onset_tolerance=self.cfg.score.onset_tolerance,
            offset_ratio=self.cfg.score.offset_ratio,
            offset_min_tolerance=self.cfg.score.offset_min_tolerance,
        )
        result = {
            "precision": conp_precision,
            "recall": conp_recall,
            "f1": conp_f1,
            "COn_precision": con_precision,
            "COn_recall": con_recall,
            "COn": con_f1,
            "COnP_precision": conp_precision,
            "COnP_recall": conp_recall,
            "COnP": conp_f1,
            "COnPOff_precision": conpoff_precision,
            "COnPOff_recall": conpoff_recall,
            "COnPOff": conpoff_f1,
        }
        if "voice_frame_output" in total_dict and "voice_frame_roll" in total_dict:
            frame_mask = total_dict.get("voice_frame_mask_roll")
            if frame_mask is not None:
                frame_mask = frame_mask[:, voice_idx, :]
            frame_scores = _frame_metrics(
                total_dict["voice_frame_roll"][:, voice_idx, :],
                total_dict["voice_frame_output"][:, voice_idx, :],
                mask=frame_mask,
                threshold=(thresholds or self.voice_thresholds[voice_name])["frame_threshold"],
            )
            for key, value in frame_scores.items():
                result[f"frame_{key}"] = value
        for onset_tolerance in FIXED_ONSET_TOLERANCES:
            tag = _tolerance_tag(onset_tolerance)
            fixed_precision, fixed_recall, fixed_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=ref_intervals,
                ref_pitches=note_to_freq(ref_pitches),
                est_intervals=est_intervals,
                est_pitches=note_to_freq(est_pitches),
                onset_tolerance=onset_tolerance,
                offset_ratio=None,
                offset_min_tolerance=self.cfg.score.offset_min_tolerance,
            )
            result[f"note_precision_{tag}"] = fixed_precision
            result[f"note_recall_{tag}"] = fixed_recall
            result[f"note_f1_{tag}"] = fixed_f1
            result[f"f1_{tag}"] = fixed_f1
        return result

    def metrics_for_voice(self, voice_name, thresholds=None):
        stats = {}
        for name in sorted(os.listdir(self.probs_dir)):
            if not name.endswith(".pkl"):
                continue

            stem = os.path.splitext(name)[0]
            prob_path = os.path.join(self.probs_dir, name)
            note_path = os.path.join(self.note_dir, f"{stem}.pkl")
            if not os.path.exists(note_path):
                continue

            song_stats = self.calculate_voice_score_per_song(prob_path, note_path, voice_name, thresholds=thresholds)
            for key, value in song_stats.items():
                stats.setdefault(key, []).append(value)
        return stats

    def calculate_score_per_song(self, prob_path, note_path):
        return_dict = {}
        per_voice_f1 = []
        per_voice_f1_by_tolerance = {_tolerance_tag(t): [] for t in FIXED_ONSET_TOLERANCES}
        for voice_name in VOICE_NAMES:
            song_stats = self.calculate_voice_score_per_song(prob_path, note_path, voice_name)
            return_dict[f"{voice_name}_precision"] = song_stats["precision"]
            return_dict[f"{voice_name}_recall"] = song_stats["recall"]
            return_dict[f"{voice_name}_f1"] = song_stats["f1"]
            return_dict[f"{voice_name}_COn"] = song_stats["COn"]
            return_dict[f"{voice_name}_COnP"] = song_stats["COnP"]
            return_dict[f"{voice_name}_COnPOff"] = song_stats["COnPOff"]
            for metric_name in ("precision", "recall", "f1"):
                key = f"frame_{metric_name}"
                if key in song_stats:
                    return_dict[f"{voice_name}_frame_{metric_name}"] = song_stats[key]
            for onset_tolerance in FIXED_ONSET_TOLERANCES:
                tag = _tolerance_tag(onset_tolerance)
                return_dict[f"{voice_name}_f1_{tag}"] = song_stats[f"f1_{tag}"]
                per_voice_f1_by_tolerance[tag].append(song_stats[f"f1_{tag}"])
            per_voice_f1.append(song_stats["f1"])
        return_dict["mean_satb_note_f1"] = float(np.mean(per_voice_f1)) if per_voice_f1 else 0.0
        for tag, values in per_voice_f1_by_tolerance.items():
            return_dict[f"mean_satb_note_f1_{tag}"] = float(np.mean(values)) if values else 0.0
        return return_dict

    def metrics(self):
        stats = {}
        for name in sorted(os.listdir(self.probs_dir)):
            if not name.endswith(".pkl"):
                continue

            stem = os.path.splitext(name)[0]
            prob_path = os.path.join(self.probs_dir, name)
            note_path = os.path.join(self.note_dir, f"{stem}.pkl")
            if not os.path.exists(note_path):
                continue

            song_stats = self.calculate_score_per_song(prob_path, note_path)
            for key, value in song_stats.items():
                stats.setdefault(key, []).append(value)
        return stats

    def presence_arrays(self, threshold=0.5):
        ref_list = []
        pred_list = []

        for name in sorted(os.listdir(self.probs_dir)):
            if not name.endswith(".pkl"):
                continue

            stem = os.path.splitext(name)[0]
            prob_path = os.path.join(self.probs_dir, name)
            note_path = os.path.join(self.note_dir, f"{stem}.pkl")
            if not os.path.exists(note_path):
                continue

            with open(prob_path, "rb") as f:
                total_dict = pickle.load(f)

            if "voice_presence_output" not in total_dict:
                continue

            pred_presence = np.asarray(total_dict["voice_presence_output"], dtype=np.float32).reshape(-1)
            if pred_presence.size != len(VOICE_NAMES):
                continue

            ref_presence = _reference_presence_vector(_load_note_bars(note_path))
            ref_list.append(ref_presence)
            pred_list.append((pred_presence >= threshold).astype(np.float32))

        if not ref_list:
            return None, None

        return np.stack(ref_list, axis=0), np.stack(pred_list, axis=0)

    def presence_summary(self, threshold=0.5):
        ref_array, pred_array = self.presence_arrays(threshold=threshold)
        if ref_array is None or pred_array is None:
            return {}

        summary = {
            "presence_accuracy": float(np.mean(pred_array == ref_array)),
            "presence_exact_match_accuracy": float(np.mean(np.all(pred_array == ref_array, axis=1))),
        }

        per_voice_acc = []
        per_voice_f1 = []
        for voice_idx, voice_name in enumerate(VOICE_NAMES):
            ref_voice = ref_array[:, voice_idx]
            pred_voice = pred_array[:, voice_idx]

            tp = float(np.sum((pred_voice == 1.0) & (ref_voice == 1.0)))
            fp = float(np.sum((pred_voice == 1.0) & (ref_voice == 0.0)))
            fn = float(np.sum((pred_voice == 0.0) & (ref_voice == 1.0)))

            acc = float(np.mean(pred_voice == ref_voice))
            precision = tp / max(tp + fp, 1.0)
            recall = tp / max(tp + fn, 1.0)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)

            summary[f"{voice_name}_presence_accuracy"] = acc
            summary[f"{voice_name}_presence_precision"] = precision
            summary[f"{voice_name}_presence_recall"] = recall
            summary[f"{voice_name}_presence_f1"] = f1
            per_voice_acc.append(acc)
            per_voice_f1.append(f1)

        summary["mean_presence_accuracy"] = float(np.mean(per_voice_acc))
        summary["mean_presence_f1"] = float(np.mean(per_voice_f1))
        return summary


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def aggregate_voice_f1_summary(stats):
    voice_f1_means = [_mean(stats.get(f"{voice_name}_f1", [])) for voice_name in VOICE_NAMES]
    if not voice_f1_means:
        return {
            "mean_satb_note_f1": 0.0,
            "min_satb_note_f1": 0.0,
            "harmonic_satb_note_f1": 0.0,
            "balanced_satb_note_f1": 0.0,
            "satb_f1_std": 0.0,
        }

    voice_f1_means = np.asarray(voice_f1_means, dtype=np.float32)
    eps = 1e-8
    mean_f1 = float(np.mean(voice_f1_means))
    min_f1 = float(np.min(voice_f1_means))
    harmonic_f1 = float(len(voice_f1_means) / np.sum(1.0 / np.clip(voice_f1_means, eps, None)))
    std_f1 = float(np.std(voice_f1_means))
    balanced_f1 = float((mean_f1 + min_f1) / 2.0)
    return {
        "mean_satb_note_f1": mean_f1,
        "min_satb_note_f1": min_f1,
        "harmonic_satb_note_f1": harmonic_f1,
        "balanced_satb_note_f1": balanced_f1,
        "satb_f1_std": std_f1,
    }


def aggregate_voice_metric_mean(stats, metric_key):
    values = [_mean(stats.get(f"{voice_name}_{metric_key}", [])) for voice_name in VOICE_NAMES]
    return float(np.mean(values)) if values else 0.0


def print_merged_channel_metrics(cfg):
    merged_stats = ScoreCalculator(cfg).metrics()
    if not merged_stats:
        return

    print("=" * 80)
    print(f"Merged Channel Evaluation | {get_model_name(cfg)} | ckpt={cfg.exp.ckpt_iteration}")
    print("=" * 80)
    ordered_keys = [
        "frame_precision",
        "frame_recall",
        "frame_f1",
        "COn",
        "COnP",
        "COnPOff",
        "note_precision",
        "note_recall",
        "note_f1",
        "note_f1_50ms",
        "note_f1_100ms",
        "note_with_offset_precision",
        "note_with_offset_recall",
        "note_with_offset_f1",
    ]
    for key in ordered_keys:
        if key in merged_stats:
            print(f"{key}: {np.mean(merged_stats[key]):.4f}")


def main():
    initialize(config_path="./", job_name="choral_eval", version_base=None)
    cfg = compose(config_name="config", overrides=sys.argv[1:])

    print_merged_channel_metrics(cfg)

    calculator = ChoralScoreCalculator(cfg)
    stats = calculator.metrics()

    print("=" * 80)
    print(f"Choral Voice Evaluation | {calculator.model_name} | ckpt={cfg.exp.ckpt_iteration}")
    print("=" * 80)
    for voice_name in VOICE_NAMES:
        p = _mean(stats.get(f"{voice_name}_precision", []))
        r = _mean(stats.get(f"{voice_name}_recall", []))
        f1 = _mean(stats.get(f"{voice_name}_f1", []))
        print(f"{voice_name}_precision: {p:.4f}")
        print(f"{voice_name}_recall: {r:.4f}")
        print(f"{voice_name}_f1: {f1:.4f}")
        print(f"{voice_name}_frame_precision: {_mean(stats.get(f'{voice_name}_frame_precision', [])):.4f}")
        print(f"{voice_name}_frame_recall: {_mean(stats.get(f'{voice_name}_frame_recall', [])):.4f}")
        print(f"{voice_name}_frame_f1: {_mean(stats.get(f'{voice_name}_frame_f1', [])):.4f}")
        for onset_tolerance in FIXED_ONSET_TOLERANCES:
            tag = _tolerance_tag(onset_tolerance)
            print(f"{voice_name}_f1_{tag}: {_mean(stats.get(f'{voice_name}_f1_{tag}', [])):.4f}")
        print(f"{voice_name}_COn: {_mean(stats.get(f'{voice_name}_COn', [])):.4f}")
        print(f"{voice_name}_COnP: {_mean(stats.get(f'{voice_name}_COnP', [])):.4f}")
        print(f"{voice_name}_COnPOff: {_mean(stats.get(f'{voice_name}_COnPOff', [])):.4f}")
    summary = aggregate_voice_f1_summary(stats)
    print(f"mean_satb_note_f1: {summary['mean_satb_note_f1']:.4f}")
    for onset_tolerance in FIXED_ONSET_TOLERANCES:
        tag = _tolerance_tag(onset_tolerance)
        print(f"mean_satb_note_f1_{tag}: {_mean(stats.get(f'mean_satb_note_f1_{tag}', [])):.4f}")
    print(f"min_satb_note_f1: {summary['min_satb_note_f1']:.4f}")
    print(f"harmonic_satb_note_f1: {summary['harmonic_satb_note_f1']:.4f}")
    print(f"balanced_satb_note_f1: {summary['balanced_satb_note_f1']:.4f}")
    print(f"satb_f1_std: {summary['satb_f1_std']:.4f}")
    print(f"mean_satb_COn: {aggregate_voice_metric_mean(stats, 'COn'):.4f}")
    print(f"mean_satb_COnP: {aggregate_voice_metric_mean(stats, 'COnP'):.4f}")
    print(f"mean_satb_COnPOff: {aggregate_voice_metric_mean(stats, 'COnPOff'):.4f}")

    presence_summary = calculator.presence_summary()
    if presence_summary:
        print("=" * 80)
        print(f"Voice Presence Evaluation | {calculator.model_name} | ckpt={cfg.exp.ckpt_iteration}")
        print("=" * 80)
        for voice_name in VOICE_NAMES:
            print(f"{voice_name}_presence_accuracy: {presence_summary[f'{voice_name}_presence_accuracy']:.4f}")
            print(f"{voice_name}_presence_f1: {presence_summary[f'{voice_name}_presence_f1']:.4f}")
        print(f"presence_accuracy: {presence_summary['presence_accuracy']:.4f}")
        print(f"presence_exact_match_accuracy: {presence_summary['presence_exact_match_accuracy']:.4f}")
        print(f"mean_presence_accuracy: {presence_summary['mean_presence_accuracy']:.4f}")
        print(f"mean_presence_f1: {presence_summary['mean_presence_f1']:.4f}")


if __name__ == "__main__":
    main()
