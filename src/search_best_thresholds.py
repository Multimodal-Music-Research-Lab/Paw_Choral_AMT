from __future__ import annotations

import argparse
import os
from itertools import product

import numpy as np
from hydra import compose, initialize_config_dir

from calculate_scores import ScoreCalculator
from calculate_choral_scores import (
    SAME_OUTPUT_REPORT_KEYS,
    ChoralScoreCalculator,
    VOICE_NAMES,
    aggregate_same_output_summary,
)
from utilities import get_model_name


CHORAL_RESULT_KEYS = (
    "mean_satb_note_f1",
    "mean_satb_note_f1_50ms",
    "mean_satb_note_f1_100ms",
    "min_satb_note_f1",
    "harmonic_satb_note_f1",
    "balanced_satb_note_f1",
    "satb_f1_std",
    *SAME_OUTPUT_REPORT_KEYS,
    *(
        f"{voice_name}_{metric_name}"
        for voice_name in VOICE_NAMES
        for metric_name in ("f1", "f1_50ms", "f1_100ms")
    ),
)


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Grid-search decoding thresholds for note F1.")
    parser.add_argument("--config_dir", type=str, default=".", help="Directory containing config.yaml")
    parser.add_argument("--workspace", type=str, default="./workspaces", help="Workspace root")
    parser.add_argument("--test_set", type=str, default="youchorale", help="Dataset name")
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Data split used for threshold selection (default: validation).",
    )
    parser.add_argument(
        "--allow-test-tuning",
        action="store_true",
        help="Explicitly acknowledge non-reportable diagnostic tuning on the test split.",
    )
    parser.add_argument("--ckpt_iteration", type=str, required=True, help="Checkpoint iteration, e.g. 17999")
    parser.add_argument("--model_name", type=str, default="", help="Optional explicit model name override")
    parser.add_argument(
        "--model_arch",
        type=str,
        default=None,
        help="Model architecture (default: pawct for choral, pagct otherwise)",
    )
    parser.add_argument("--model_mode", type=str, default="frame_onset_offset", help="Model mode")
    parser.add_argument("--post_processor_type", type=str, default="onsets_frames", help="Post processor")
    parser.add_argument("--sample_rate", type=int, default=None, help="Optional sample rate override")
    parser.add_argument("--onset_tolerance", type=float, default=None, help="Optional onset tolerance override in seconds")
    parser.add_argument("--name_suffix", type=str, default="", help="Optional experiment name suffix used in model_name")
    parser.add_argument("--youchorale-dir", type=str, default="", help="Optional YouChorale dataset root override")
    parser.add_argument(
        "--youchorale-split-dir",
        type=str,
        default="",
        help="Optional external YouChorale train/valid/test manifest directory",
    )
    parser.add_argument("--youchorale-pro-dir", type=str, default="", help="Optional YouChorale-Pro dataset root override")
    parser.add_argument("--choral_enable", action="store_true", help="Use choral SATB evaluation instead of single-stream evaluation")
    parser.add_argument("--choral_per_voice", action="store_true", help="Search independent thresholds for S/A/T/B and combine them")
    parser.add_argument(
        "--reference-duration-policy",
        choices=("strict", "clip_offsets_drop_unobservable_onsets_v1"),
        default=None,
        help=(
            "Explicit policy for source notes crossing the packed waveform boundary. "
            "Use the same frozen value as inference."
        ),
    )
    parser.add_argument(
        "--range-prior-loss-weight",
        "--range_prior_loss_weight",
        dest="range_prior_loss_weight",
        type=float,
        default=None,
        help="Training-time RP weight recorded by the checkpoint.",
    )
    parser.add_argument(
        "--continuity-prior-loss-weight",
        "--continuity_prior_loss_weight",
        dest="continuity_prior_loss_weight",
        type=float,
        default=None,
        help="Training-time OC weight recorded by the checkpoint.",
    )
    assignment_group = parser.add_mutually_exclusive_group()
    assignment_group.add_argument(
        "--target-assignment",
        "--target_assignment",
        dest="target_assignment",
        choices=(
            "part_name",
            "range_prior",
            "ordered_continuity",
            "range_masked_continuity",
            "legacy_range_prior",
            "legacy_ordered_continuity",
            "legacy_range_masked_continuity",
        ),
        default=None,
        help="Canonical SATB target construction used for the checkpoint.",
    )
    assignment_group.add_argument(
        "--voice-assignment-method",
        "--voice_assignment_method",
        dest="voice_assignment_method",
        choices=(
            "part_name",
            "range_prior",
            "ordered_continuity",
            "range_masked_continuity",
        ),
        default=None,
        help=(
            "Deprecated compatibility flag: preserves legacy all-note reassignment "
            "and the historical _va_* model path."
        ),
    )
    parser.add_argument("--objective", type=str, default="", help="Metric key to maximize")
    parser.add_argument("--frame_thresholds", type=str, default="0.05,0.1,0.15,0.2,0.3")
    parser.add_argument("--onset_thresholds", type=str, default="0.003,0.005,0.01,0.02,0.03,0.05")
    parser.add_argument("--offset_thresholds", type=str, default="0.003,0.005,0.01,0.02,0.03,0.05")
    parser.add_argument("--output_txt", type=str, required=True, help="Path to txt summary")
    parser.add_argument(
        "--config-override",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Repeatable Hydra override for checkpoint-recorded settings not covered "
            "by dedicated flags (for example an audited RP pitch range)."
        ),
    )
    args = parser.parse_args()
    if args.model_arch is None:
        args.model_arch = "pawct" if args.choral_enable else "pagct"
    if args.target_assignment is None and args.voice_assignment_method is None:
        args.target_assignment = "part_name"
    return args


def validate_selection_split(split: str, allow_test_tuning: bool = False) -> None:
    if split == "test" and not allow_test_tuning:
        raise ValueError(
            "Refusing to tune thresholds on the test split. Tune on validation, "
            "then run the test evaluator once with frozen thresholds."
        )


def metric_mean(stats_dict, key: str) -> float:
    values = stats_dict.get(key, [])
    if not values:
        return float("nan")
    return float(np.mean(values))


def add_choral_summary_metrics(result: dict) -> dict:
    voice_f1s = [result.get(f"{voice_name}_f1", float("nan")) for voice_name in VOICE_NAMES]
    voice_f1s = np.asarray(voice_f1s, dtype=np.float32)
    if np.any(np.isnan(voice_f1s)):
        result["min_satb_note_f1"] = float("nan")
        result["harmonic_satb_note_f1"] = float("nan")
        result["balanced_satb_note_f1"] = float("nan")
        result["satb_f1_std"] = float("nan")
        return result

    eps = 1e-8
    result["min_satb_note_f1"] = float(np.min(voice_f1s))
    result["harmonic_satb_note_f1"] = float(len(voice_f1s) / np.sum(1.0 / np.clip(voice_f1s, eps, None)))
    result["balanced_satb_note_f1"] = float((result["mean_satb_note_f1"] + result["min_satb_note_f1"]) / 2.0)
    result["satb_f1_std"] = float(np.std(voice_f1s))
    return result


def require_metric_values(stats_dict, context: str):
    if any(values for values in stats_dict.values()):
        return
    raise ValueError(
        f"No matching evaluation songs for {context}. "
        "Check dataset directory overrides and whether probs files match available note annotations."
    )


def build_choral_result(stats_dict: dict) -> dict:
    result = {
        "mean_satb_note_f1": metric_mean(stats_dict, "mean_satb_note_f1"),
        "mean_satb_note_f1_50ms": metric_mean(stats_dict, "mean_satb_note_f1_50ms"),
        "mean_satb_note_f1_100ms": metric_mean(stats_dict, "mean_satb_note_f1_100ms"),
    }
    result.update(aggregate_same_output_summary(stats_dict))
    for voice_name in VOICE_NAMES:
        result[f"{voice_name}_f1"] = metric_mean(stats_dict, f"{voice_name}_f1")
        result[f"{voice_name}_f1_50ms"] = metric_mean(stats_dict, f"{voice_name}_f1_50ms")
        result[f"{voice_name}_f1_100ms"] = metric_mean(stats_dict, f"{voice_name}_f1_100ms")
    return add_choral_summary_metrics(result)


def format_metric_value(key: str, value) -> str:
    if key.endswith("_count") or key.startswith("voice_confusion_"):
        return str(int(value))
    return f"{float(value):.4f}"


def artifact_identity(calculator) -> dict:
    validator = getattr(calculator, "artifact_validator", None)
    if validator is None:
        return {}
    return validator.identity_summary()


def write_artifact_identity(file_object, identity: dict) -> None:
    for key in (
        "checkpoint_filename",
        "checkpoint_iteration",
        "checkpoint_sha256",
        "inference_run_id",
    ):
        file_object.write(f"actual_{key}: {identity.get(key)}\n")


def build_voice_search_result(voice_name: str, thresholds: dict, voice_stats: dict) -> dict:
    return {
        "voice": voice_name,
        "frame_threshold": thresholds["frame_threshold"],
        "onset_threshold": thresholds["onset_threshold"],
        "offset_threshold": thresholds["offset_threshold"],
        "f1": metric_mean(voice_stats, "f1"),
        "f1_50ms": metric_mean(voice_stats, "f1_50ms"),
        "f1_100ms": metric_mean(voice_stats, "f1_100ms"),
        "precision": metric_mean(voice_stats, "precision"),
        "recall": metric_mean(voice_stats, "recall"),
    }


def search_best_choral_per_voice(calculator, combos):
    per_voice_best = {}
    stats_grid = calculator.metrics_for_voice_threshold_grid(combos)
    for voice_name in VOICE_NAMES:
        voice_results = []
        for idx, (frame_th, onset_th, offset_th) in enumerate(combos, start=1):
            thresholds = {
                "frame_threshold": frame_th,
                "onset_threshold": onset_th,
                "offset_threshold": offset_th,
            }
            voice_stats = stats_grid[voice_name][idx - 1]
            require_metric_values(
                voice_stats,
                context=(
                    f"voice={voice_name}, probs_dir={calculator.probs_dir}, "
                    f"note_dir={calculator.note_dir}"
                ),
            )
            result = build_voice_search_result(voice_name, thresholds, voice_stats)
            print(
                f"[{voice_name} {idx}/{len(combos)}] "
                f"frame={frame_th:.4f} onset={onset_th:.4f} offset={offset_th:.4f} "
                f"f1={result['f1']:.4f}"
            )
            voice_results.append(result)

        voice_results.sort(
            key=lambda x: (
                -np.nan_to_num(x["f1"], nan=-1.0),
                -np.nan_to_num(x["precision"], nan=-1.0),
                -np.nan_to_num(x["recall"], nan=-1.0),
            )
        )
        per_voice_best[voice_name] = voice_results[0]
    return per_voice_best


def build_overrides(args, extra_overrides=None):
    overrides = [
        f"dataset.test_set={args.test_set}",
        f"dataset.eval_split={args.split}",
        f"exp.workspace={args.workspace}",
        f"exp.ckpt_iteration={args.ckpt_iteration}",
        f"model.arch={args.model_arch}",
        f"model.mode={args.model_mode}",
        f"post.post_processor_type={args.post_processor_type}",
    ]
    if getattr(args, "model_name", ""):
        overrides.append(f"model.name={args.model_name}")
    if args.sample_rate is not None:
        overrides.append(f"feature.sample_rate={args.sample_rate}")
    if args.onset_tolerance is not None:
        overrides.append(f"score.onset_tolerance={args.onset_tolerance}")
    if args.name_suffix:
        overrides.append(f"exp.name_suffix={args.name_suffix}")
    if getattr(args, "youchorale_dir", ""):
        overrides.append(f"dataset.youchorale_dir={args.youchorale_dir}")
    if getattr(args, "youchorale_split_dir", ""):
        overrides.append(
            f"dataset.youchorale_split_dir={args.youchorale_split_dir}"
        )
    if getattr(args, "youchorale_pro_dir", ""):
        overrides.append(f"dataset.youchorale_pro_dir={args.youchorale_pro_dir}")
    if args.choral_enable:
        overrides.append("choral.enable=true")
        if getattr(args, "reference_duration_policy", None) is not None:
            overrides.append(
                "choral.evaluation_reference_duration_policy="
                f"{args.reference_duration_policy}"
            )
        target_assignment = getattr(args, "target_assignment", None)
        legacy_assignment = getattr(args, "voice_assignment_method", None)
        if target_assignment is not None and legacy_assignment is not None:
            raise ValueError(
                "Choose either canonical target_assignment or deprecated "
                "voice_assignment_method, not both."
            )
        if legacy_assignment is not None:
            overrides.append(f"choral.voice_assignment_method={legacy_assignment}")
        else:
            overrides.append(
                f"choral.target_assignment={target_assignment or 'part_name'}"
            )
        range_weight = getattr(args, "range_prior_loss_weight", None)
        continuity_weight = getattr(args, "continuity_prior_loss_weight", None)
        if range_weight is not None:
            overrides.append(f"choral.range_prior_loss_weight={range_weight}")
        if continuity_weight is not None:
            overrides.append(f"choral.continuity_prior_loss_weight={continuity_weight}")
    overrides.extend(getattr(args, "config_overrides", ()) or ())
    if extra_overrides:
        overrides.extend(extra_overrides)
    return overrides


def main():
    args = parse_args()
    validate_selection_split(args.split, args.allow_test_tuning)
    frame_thresholds = parse_float_list(args.frame_thresholds)
    onset_thresholds = parse_float_list(args.onset_thresholds)
    offset_thresholds = parse_float_list(args.offset_thresholds)
    if args.objective:
        objective_key = args.objective
    elif args.choral_enable:
        objective_key = "mean_satb_note_f1"
    else:
        objective_key = "note_f1"
    calculator_cls = ChoralScoreCalculator if args.choral_enable else ScoreCalculator

    with initialize_config_dir(config_dir=os.path.abspath(args.config_dir), job_name="threshold_search", version_base=None):
        base_cfg = compose(
            config_name="config",
            overrides=build_overrides(args),
        )

    model_name = get_model_name(base_cfg)
    probs_dir = os.path.join(
        args.workspace,
        "probs",
        args.test_set,
        args.split,
        model_name,
        f"{args.ckpt_iteration}_iteration",
    )
    if not os.path.isdir(probs_dir):
        raise FileNotFoundError(
            f"Missing probs directory: {probs_dir}\n"
            f"Run inference first for checkpoint {args.ckpt_iteration}."
        )

    if args.choral_per_voice and not args.choral_enable:
        raise ValueError("--choral_per_voice requires --choral_enable")

    if args.choral_per_voice:
        calculator = ChoralScoreCalculator(base_cfg)
        combos = list(product(frame_thresholds, onset_thresholds, offset_thresholds))
        per_voice_best = search_best_choral_per_voice(calculator, combos)
        validated_identity = artifact_identity(calculator)

        best_voice_thresholds = {
            voice_name: {
                "frame_threshold": item["frame_threshold"],
                "onset_threshold": item["onset_threshold"],
                "offset_threshold": item["offset_threshold"],
            }
            for voice_name, item in per_voice_best.items()
        }
        combined_stats = ChoralScoreCalculator(base_cfg, voice_thresholds=best_voice_thresholds).metrics()
        require_metric_values(
            combined_stats,
            context=f"combined choral evaluation, probs_dir={calculator.probs_dir}, note_dir={calculator.note_dir}",
        )
        frame_list = [per_voice_best[v]["frame_threshold"] for v in VOICE_NAMES]
        onset_list = [per_voice_best[v]["onset_threshold"] for v in VOICE_NAMES]
        offset_list = [per_voice_best[v]["offset_threshold"] for v in VOICE_NAMES]
        combined_summary = build_choral_result(combined_stats)
        presence_summary = ChoralScoreCalculator(base_cfg, voice_thresholds=best_voice_thresholds).presence_summary()

        output_dir = os.path.dirname(args.output_txt)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_txt, "w", encoding="utf-8") as f:
            f.write(f"checkpoint: {args.ckpt_iteration}\n")
            write_artifact_identity(f, validated_identity)
            f.write(f"test_set: {args.test_set}\n")
            f.write(f"selection_split: {args.split}\n")
            f.write(f"model_name: {model_name}\n")
            f.write(f"probs_dir: {probs_dir}\n")
            f.write("choral_enable: True\n")
            f.write("choral_per_voice: True\n")
            f.write("search_strategy: per-voice independent threshold search\n")
            f.write(f"reported_objective: {objective_key}\n\n")
            if args.onset_tolerance is not None:
                f.write(f"search_onset_tolerance: {args.onset_tolerance}\n\n")
            f.write("Best thresholds per voice:\n")
            f.write("voice\tframe_th\tonset_th\toffset_th\tf1\tf1_50ms\tf1_100ms\tprecision\trecall\n")
            for voice_name in VOICE_NAMES:
                item = per_voice_best[voice_name]
                f.write(
                    f"{voice_name}\t"
                    f"{item['frame_threshold']:.4f}\t"
                    f"{item['onset_threshold']:.4f}\t"
                    f"{item['offset_threshold']:.4f}\t"
                    f"{item['f1']:.4f}\t"
                    f"{item['f1_50ms']:.4f}\t"
                    f"{item['f1_100ms']:.4f}\t"
                    f"{item['precision']:.4f}\t"
                    f"{item['recall']:.4f}\n"
                )
            f.write("\nCombined evaluation with per-voice thresholds:\n")
            for key in CHORAL_RESULT_KEYS:
                f.write(
                    f"{key}: {format_metric_value(key, combined_summary[key])}\n"
                )
            if presence_summary:
                f.write("\nVoice presence evaluation:\n")
                for key in [
                    "presence_accuracy",
                    "presence_exact_match_accuracy",
                    "mean_presence_accuracy",
                    "mean_presence_f1",
                    "S_presence_accuracy",
                    "A_presence_accuracy",
                    "T_presence_accuracy",
                    "B_presence_accuracy",
                ]:
                    f.write(f"{key}: {presence_summary[key]:.4f}\n")
            f.write("\nHydra overrides:\n")
            f.write("choral.use_per_voice_thresholds=true\n")
            f.write(f"choral.voice_frame_thresholds={frame_list}\n")
            f.write(f"choral.voice_onset_thresholds={onset_list}\n")
            f.write(f"choral.voice_offset_thresholds={offset_list}\n")

        print("\nBest thresholds per voice:")
        for voice_name in VOICE_NAMES:
            print(voice_name, per_voice_best[voice_name])
        print("\nCombined evaluation:")
        print(combined_summary)
        if presence_summary:
            print("\nVoice presence evaluation:")
            print(presence_summary)
        print(f"\nSaved txt to: {args.output_txt}")
        return

    results = []
    validated_identity = None
    combos = list(product(frame_thresholds, onset_thresholds, offset_thresholds))
    for idx, (frame_th, onset_th, offset_th) in enumerate(combos, start=1):
        with initialize_config_dir(config_dir=os.path.abspath(args.config_dir), job_name=f"threshold_search_{idx}", version_base=None):
            cfg = compose(
                config_name="config",
                overrides=build_overrides(
                    args,
                    [
                        f"post.frame_threshold={frame_th}",
                        f"post.onset_threshold={onset_th}",
                        f"post.offset_threshold={offset_th}",
                    ],
                ),
            )
        calculator = calculator_cls(cfg)
        stats = calculator.metrics()
        current_identity = artifact_identity(calculator)
        if validated_identity is None:
            validated_identity = current_identity
        elif current_identity != validated_identity:
            raise RuntimeError(
                "Probability artifact identity changed during threshold search"
            )
        require_metric_values(
            stats,
            context=f"threshold search, test_set={args.test_set}, probs_dir={probs_dir}",
        )
        result = {"frame_threshold": frame_th, "onset_threshold": onset_th, "offset_threshold": offset_th}
        if args.choral_enable:
            result.update(build_choral_result(stats))
        else:
            result.update(
                {
                    "note_f1": metric_mean(stats, "note_f1"),
                    "note_precision": metric_mean(stats, "note_precision"),
                    "note_recall": metric_mean(stats, "note_recall"),
                    "note_with_offset_f1": metric_mean(stats, "note_with_offset_f1"),
                    "frame_f1": metric_mean(stats, "frame_f1"),
                    "frame_precision": metric_mean(stats, "frame_precision"),
                    "frame_recall": metric_mean(stats, "frame_recall"),
                }
            )
        print(
            f"[{idx}/{len(combos)}] "
            f"frame={frame_th:.4f} onset={onset_th:.4f} offset={offset_th:.4f} "
            f"{objective_key}={result.get(objective_key, float('nan')):.4f}"
        )
        results.append(result)

    if args.choral_enable:
        results.sort(
            key=lambda x: (
                -np.nan_to_num(x.get(objective_key), nan=-1.0),
                -np.nan_to_num(x.get("S_f1"), nan=-1.0),
                -np.nan_to_num(x.get("A_f1"), nan=-1.0),
                -np.nan_to_num(x.get("T_f1"), nan=-1.0),
                -np.nan_to_num(x.get("B_f1"), nan=-1.0),
            )
        )
    else:
        results.sort(
            key=lambda x: (
                -np.nan_to_num(x.get(objective_key), nan=-1.0),
                -np.nan_to_num(x.get("note_with_offset_f1"), nan=-1.0),
                -np.nan_to_num(x.get("frame_f1"), nan=-1.0),
            )
        )

    output_dir = os.path.dirname(args.output_txt)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write(f"checkpoint: {args.ckpt_iteration}\n")
        write_artifact_identity(f, validated_identity or {})
        f.write(f"test_set: {args.test_set}\n")
        f.write(f"selection_split: {args.split}\n")
        f.write(f"model_name: {model_name}\n")
        f.write(f"probs_dir: {probs_dir}\n")
        f.write(f"choral_enable: {args.choral_enable}\n")
        f.write(f"objective: {objective_key}\n")
        if args.onset_tolerance is not None:
            f.write(f"search_onset_tolerance: {args.onset_tolerance}\n")
        f.write(f"num_combinations: {len(results)}\n\n")
        f.write(f"Top results sorted by {objective_key}:\n")
        if args.choral_enable:
            f.write(
                "\t".join(
                    ("rank", "frame_th", "onset_th", "offset_th", *CHORAL_RESULT_KEYS)
                )
                + "\n"
            )
            for rank, item in enumerate(results, start=1):
                values = [
                    str(rank),
                    f"{item['frame_threshold']:.4f}",
                    f"{item['onset_threshold']:.4f}",
                    f"{item['offset_threshold']:.4f}",
                    *(format_metric_value(key, item[key]) for key in CHORAL_RESULT_KEYS),
                ]
                f.write("\t".join(values) + "\n")
        else:
            f.write(
                "rank\tframe_th\tonset_th\toffset_th\tnote_f1\tnote_prec\tnote_rec\tnote_off_f1\tframe_f1\tframe_prec\tframe_rec\n"
            )
            for rank, item in enumerate(results, start=1):
                f.write(
                    f"{rank}\t"
                    f"{item['frame_threshold']:.4f}\t"
                    f"{item['onset_threshold']:.4f}\t"
                    f"{item['offset_threshold']:.4f}\t"
                    f"{item['note_f1']:.4f}\t"
                    f"{item['note_precision']:.4f}\t"
                    f"{item['note_recall']:.4f}\t"
                    f"{item['note_with_offset_f1']:.4f}\t"
                    f"{item['frame_f1']:.4f}\t"
                    f"{item['frame_precision']:.4f}\t"
                    f"{item['frame_recall']:.4f}\n"
                )

    best = results[0]
    print("\nBest thresholds:")
    print(best)
    print(f"\nSaved txt to: {args.output_txt}")


if __name__ == "__main__":
    main()
