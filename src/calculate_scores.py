# SPDX-License-Identifier: Apache-2.0
# Includes modifications to ByteDance's piano_transcription implementation.

from __future__ import annotations

import gc
import os
import pickle
import sys
import time
from copy import deepcopy

import h5py
import mir_eval
import numpy as np
from hydra import compose, initialize
from sklearn import metrics

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import wandb
except Exception:
    wandb = None

from utilities import (
    get_task_spec,
    resolve_post_processor_type,
    OnsetsFramesPostProcessor,
    RegressionPostProcessor,
    get_filename,
    get_model_name,
    note_to_freq,
)
from canonical_union import build_canonical_union_rolls, canonical_union_events
from choral_targets import require_complete_satb_reference
from probability_artifacts import ProbabilityArtifactValidator


FIXED_ONSET_TOLERANCES = (0.05, 0.10)
CHORAL_DATASETS = frozenset({'youchorale', 'youchorale_pro', 'csd', 'cantoria'})


def tolerance_tag(onset_tolerance: float) -> str:
    return f"{int(round(onset_tolerance * 1000.0))}ms"


def _choral_reference_path(cfg, recording_stem: str) -> str | None:
    """Resolve the canonical note annotation used by choral PagCT scoring."""

    if cfg is None or not hasattr(cfg, 'dataset'):
        return None
    dataset_name = str(cfg.dataset.test_set)
    if dataset_name not in CHORAL_DATASETS:
        return None
    if dataset_name == 'youchorale':
        note_dir = os.path.join(cfg.dataset.youchorale_dir, 'note')
    elif dataset_name == 'youchorale_pro':
        note_dir = os.path.join(cfg.dataset.youchorale_pro_dir, 'note')
    elif dataset_name == 'csd':
        note_dir = os.path.join(cfg.dataset.csd_dir, 'note')
    else:
        note_dir = os.path.join(
            cfg.dataset.cantoria_dir,
            f'note_{cfg.dataset.cantoria_f0_source}',
        )
    note_path = os.path.join(note_dir, f'{recording_stem}.pkl')
    if not os.path.isfile(note_path):
        raise FileNotFoundError(
            f'Missing canonical choral reference for {recording_stem}: {note_path}'
        )
    return note_path


def _canonical_choral_reference(cfg, hdf5_path, note_path, total_dict):
    """Rebuild and verify formal PagCT ground truth from immutable annotations."""

    with h5py.File(hdf5_path, 'r') as hdf5_file:
        if 'waveform' not in hdf5_file:
            raise RuntimeError(f'Packed recording has no waveform: {hdf5_path}')
        duration = hdf5_file['waveform'].shape[0] / float(cfg.feature.sample_rate)
    with open(note_path, 'rb') as note_file:
        note_bars = pickle.load(note_file)
    require_complete_satb_reference(
        note_bars,
        note_path,
        begin_note=int(cfg.feature.begin_note),
        classes_num=int(cfg.feature.classes_num),
        recording_duration=duration,
    )

    union_events = canonical_union_events(
        note_bars,
        float(cfg.feature.frames_per_second),
        begin_note=int(cfg.feature.begin_note),
        classes_num=int(cfg.feature.classes_num),
    )
    reference = build_canonical_union_rolls(
        note_bars,
        start_time=0.0,
        segment_seconds=duration,
        frames_per_second=float(cfg.feature.frames_per_second),
        begin_note=int(cfg.feature.begin_note),
        classes_num=int(cfg.feature.classes_num),
    )
    reference['ref_on_off_pairs'] = np.asarray(
        [[onset, offset] for _, onset, offset in union_events],
        dtype=np.float32,
    ).reshape(-1, 2)
    reference['ref_midi_notes'] = np.asarray(
        [pitch for pitch, _, _ in union_events],
        dtype=np.int32,
    )

    # Formal artifacts are caches, not authorities. Reject a cache whose
    # embedded ground truth disagrees with the source annotation so corruption
    # cannot silently change a paper metric.
    for key in (
        'ref_on_off_pairs',
        'ref_midi_notes',
        'frame_roll',
        'onset_roll',
        'offset_roll',
        'frame_mask_roll',
    ):
        if key not in total_dict:
            raise RuntimeError(
                f'Probability artifact is missing canonical reference field {key}: '
                f'{hdf5_path}'
            )
        observed = np.asarray(total_dict[key])
        expected = np.asarray(reference[key])
        if observed.shape != expected.shape or not np.array_equal(observed, expected):
            raise RuntimeError(
                f'Probability artifact canonical reference mismatch for {key}: '
                f'{hdf5_path}. Re-run inference from the current source annotations.'
            )
    return reference



def build_post_processor(cfg):
    post_type = resolve_post_processor_type(cfg)
    if post_type == 'regression':
        return RegressionPostProcessor(cfg)
    if post_type in {'onsets_frames', 'onf'}:
        return OnsetsFramesPostProcessor(cfg)
    raise ValueError(f'Unsupported post.post_processor_type: {post_type}')


class ScoreCalculator(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.spec = get_task_spec(cfg)
        self.eval_split = str(getattr(cfg.dataset, 'eval_split', 'validation'))
        if self.eval_split not in {'validation', 'test'}:
            raise ValueError("dataset.eval_split must be 'validation' or 'test'")
        model_name = get_model_name(cfg)
        self.probs_dir = os.path.join(
            cfg.exp.workspace,
            'probs',
            cfg.dataset.test_set,
            self.eval_split,
            model_name,
            f'{cfg.exp.ckpt_iteration}_iteration',
        )
        self.post_processor = build_post_processor(cfg)
        self.artifact_validator = ProbabilityArtifactValidator(
            cfg,
            probs_dir=self.probs_dir,
            eval_split=self.eval_split,
        )
        self.hdf5_paths = tuple(self.artifact_validator.hdf5_by_stem.values())

    def metrics(self):
        list_args = [[n, hdf5_path] for n, hdf5_path in enumerate(self.hdf5_paths)]
        stats_list = [self.calculate_score_per_song(arg) for arg in list_args]
        if not stats_list:
            return {}
        keys = sorted({key for item in stats_list for key in item.keys()})
        return {key: [e[key] for e in stats_list if key in e] for key in keys}

    def _frame_metrics(self, y_true, y_pred, mask=None, threshold=0.5):
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
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(
            y_true,
            y_hat,
            average='binary',
            zero_division=0,
        )
        return {'precision': precision, 'recall': recall, 'f1': f1}

    def _safe_note_arrays(self, est_note_events):
        min_duration = 1e-4
        if len(est_note_events) == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )
        est_on_offs = np.array([[event['onset_time'], event['offset_time']] for event in est_note_events], dtype=np.float32)
        est_midi_notes = np.array([event['midi_note'] for event in est_note_events], dtype=np.int32)
        abnormal_index = np.nonzero(est_on_offs[:, 1] <= est_on_offs[:, 0])[0]
        est_on_offs[abnormal_index, 1] = est_on_offs[abnormal_index, 0] + min_duration
        sort_idx = np.argsort(est_on_offs[:, 0], kind='mergesort')
        est_on_offs = est_on_offs[sort_idx]
        est_midi_notes = est_midi_notes[sort_idx]
        return est_on_offs, est_midi_notes

    def _safe_ref_arrays(self, ref_on_off_pairs, ref_midi_notes):
        min_duration = 1e-4
        ref_on_off_pairs = np.asarray(ref_on_off_pairs, dtype=np.float32)
        ref_midi_notes = np.asarray(ref_midi_notes, dtype=np.int32)

        if ref_on_off_pairs.size == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        abnormal_index = np.nonzero(ref_on_off_pairs[:, 1] <= ref_on_off_pairs[:, 0])[0]
        ref_on_off_pairs = ref_on_off_pairs.copy()
        ref_on_off_pairs[abnormal_index, 1] = ref_on_off_pairs[abnormal_index, 0] + min_duration
        sort_idx = np.argsort(ref_on_off_pairs[:, 0], kind='mergesort')
        ref_on_off_pairs = ref_on_off_pairs[sort_idx]
        ref_midi_notes = ref_midi_notes[sort_idx]
        return ref_on_off_pairs, ref_midi_notes

    def _correct_onset_metrics(self, ref_on_off_pairs, est_on_offs, onset_tolerance=None):
        ref_onsets = ref_on_off_pairs[:, 0] if len(ref_on_off_pairs) else np.zeros((0,), dtype=np.float32)
        est_onsets = est_on_offs[:, 0] if len(est_on_offs) else np.zeros((0,), dtype=np.float32)
        ref_onsets = np.sort(ref_onsets)
        est_onsets = np.sort(est_onsets)
        con_f1, con_precision, con_recall = mir_eval.onset.f_measure(
            reference_onsets=ref_onsets,
            estimated_onsets=est_onsets,
            window=self.cfg.score.onset_tolerance if onset_tolerance is None else onset_tolerance,
        )
        return con_precision, con_recall, con_f1

    def _note_metrics(self, ref_on_off_pairs, ref_midi_notes, est_on_offs, est_midi_notes, onset_tolerance, offset_ratio=None):
        note_precision, note_recall, note_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_on_off_pairs,
            ref_pitches=note_to_freq(ref_midi_notes),
            est_intervals=est_on_offs,
            est_pitches=note_to_freq(est_midi_notes),
            onset_tolerance=onset_tolerance,
            offset_ratio=offset_ratio,
            offset_min_tolerance=self.cfg.score.offset_min_tolerance,
        )
        return note_precision, note_recall, note_f1

    def calculate_score_per_song(self, args):
        hdf5_path = args[1]
        prob_path = os.path.join(self.probs_dir, f'{get_filename(hdf5_path)}.pkl')
        with open(prob_path, 'rb') as fr:
            total_dict = pickle.load(fr)
        # Formal scoring must never silently fall back to unverified artifacts.
        # Tests or downstream diagnostic callers that bypass ``__init__`` must
        # install an explicit validator rather than accidentally disabling it.
        reference_path = _choral_reference_path(
            getattr(self, 'cfg', None),
            get_filename(hdf5_path),
        )
        self.artifact_validator.validate(
            total_dict,
            prob_path,
            hdf5_path=hdf5_path,
            reference_path=reference_path,
        )
        canonical_reference = (
            _canonical_choral_reference(
                self.cfg,
                hdf5_path,
                reference_path,
                total_dict,
            )
            if reference_path is not None
            else None
        )
        post_input = deepcopy(total_dict)
        est_note_events, est_pedal_events = self.post_processor.output_dict_to_midi_events(post_input)
        est_on_offs, est_midi_notes = self._safe_note_arrays(est_note_events)

        reference_source = canonical_reference or total_dict
        ref_on_off_pairs, ref_midi_notes = self._safe_ref_arrays(
            reference_source['ref_on_off_pairs'],
            reference_source['ref_midi_notes'],
        )
        return_dict = {}

        if self.cfg.score.evaluate_frame and 'frame_output' in total_dict:
            for suffix, values in self._frame_metrics(
                reference_source['frame_roll'],
                total_dict['frame_output'],
                mask=reference_source.get('frame_mask_roll'),
                threshold=self.cfg.post.frame_threshold,
            ).items():
                return_dict[f'frame_{suffix}'] = values

        if self.cfg.score.evaluate_note:
            con_precision, con_recall, con_f1 = self._correct_onset_metrics(ref_on_off_pairs, est_on_offs)
            return_dict.update({'COn_precision': con_precision, 'COn_recall': con_recall, 'COn': con_f1})

            note_precision, note_recall, note_f1 = self._note_metrics(
                ref_on_off_pairs,
                ref_midi_notes,
                est_on_offs,
                est_midi_notes,
                onset_tolerance=self.cfg.score.onset_tolerance,
                offset_ratio=None,
            )
            return_dict.update(
                {
                    'note_precision': note_precision,
                    'note_recall': note_recall,
                    'note_f1': note_f1,
                    'COnP_precision': note_precision,
                    'COnP_recall': note_recall,
                    'COnP': note_f1,
                }
            )

            note_off_precision, note_off_recall, note_off_f1 = self._note_metrics(
                ref_on_off_pairs,
                ref_midi_notes,
                est_on_offs,
                est_midi_notes,
                onset_tolerance=self.cfg.score.onset_tolerance,
                offset_ratio=self.cfg.score.offset_ratio,
            )
            return_dict.update(
                {
                    'note_with_offset_precision': note_off_precision,
                    'note_with_offset_recall': note_off_recall,
                    'note_with_offset_f1': note_off_f1,
                    'COnPOff_precision': note_off_precision,
                    'COnPOff_recall': note_off_recall,
                    'COnPOff': note_off_f1,
                }
            )

            for fixed_tolerance in FIXED_ONSET_TOLERANCES:
                tag = tolerance_tag(fixed_tolerance)
                con_precision_t, con_recall_t, con_f1_t = self._correct_onset_metrics(
                    ref_on_off_pairs,
                    est_on_offs,
                    onset_tolerance=fixed_tolerance,
                )
                return_dict[f'COn_{tag}'] = con_f1_t
                return_dict[f'COn_precision_{tag}'] = con_precision_t
                return_dict[f'COn_recall_{tag}'] = con_recall_t

                note_precision_t, note_recall_t, note_f1_t = self._note_metrics(
                    ref_on_off_pairs,
                    ref_midi_notes,
                    est_on_offs,
                    est_midi_notes,
                    onset_tolerance=fixed_tolerance,
                    offset_ratio=None,
                )
                return_dict[f'note_precision_{tag}'] = note_precision_t
                return_dict[f'note_recall_{tag}'] = note_recall_t
                return_dict[f'note_f1_{tag}'] = note_f1_t

                note_off_precision_t, note_off_recall_t, note_off_f1_t = self._note_metrics(
                    ref_on_off_pairs,
                    ref_midi_notes,
                    est_on_offs,
                    est_midi_notes,
                    onset_tolerance=fixed_tolerance,
                    offset_ratio=self.cfg.score.offset_ratio,
                )
                return_dict[f'note_with_offset_precision_{tag}'] = note_off_precision_t
                return_dict[f'note_with_offset_recall_{tag}'] = note_off_recall_t
                return_dict[f'note_with_offset_f1_{tag}'] = note_off_f1_t

        has_pedal_eval = (
            self.spec.pedal
            and self.cfg.score.evaluate_pedal
            and 'pedal_frame_output' in total_dict
            and np.sum(total_dict.get('pedal_mask_roll', np.ones_like(total_dict['pedal_frame_roll']))) > 0
        )
        if has_pedal_eval:
            ref_pedal = total_dict.get('ref_pedal_on_off_pairs', np.zeros((0, 2), dtype=np.float32))
            est_pedal = np.array([[event['onset_time'], event['offset_time']] for event in est_pedal_events], dtype=np.float32) if est_pedal_events else np.zeros((0, 2), dtype=np.float32)
            if len(ref_pedal) > 0:
                pedal_precision, pedal_recall, pedal_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                    ref_intervals=ref_pedal,
                    ref_pitches=np.ones(ref_pedal.shape[0]),
                    est_intervals=est_pedal,
                    est_pitches=np.ones(est_pedal.shape[0]),
                    onset_tolerance=0.2,
                    offset_ratio=self.cfg.score.pedal_offset_ratio,
                    offset_min_tolerance=self.cfg.score.pedal_offset_min_tolerance,
                )
                return_dict.update(
                    {
                        'pedal_precision': pedal_precision,
                        'pedal_recall': pedal_recall,
                        'pedal_f1': pedal_f1,
                    }
                )
            for suffix, values in self._frame_metrics(
                total_dict['pedal_frame_roll'],
                total_dict['pedal_frame_output'],
                mask=total_dict.get('pedal_mask_roll'),
                threshold=0.5,
            ).items():
                return_dict[f'pedal_frame_{suffix}'] = values

        return return_dict


if __name__ == '__main__':
    initialize(config_path='./', job_name='eval', version_base=None)
    cfg = compose(config_name='config', overrides=sys.argv[1:])
    spec = get_task_spec(cfg)

    print('=' * 80)
    print(f'Evaluation Mode : {cfg.exp.run_infer.upper()}')
    print(f'Model Name      : {get_model_name(cfg)}')
    print(f'Architecture    : {spec.arch}')
    print(f'Task Mode       : {spec.mode}')
    print(f'Test Set        : {cfg.dataset.test_set}')
    print(f'Evaluation Split: {cfg.dataset.eval_split}')
    print(f'Post Processor  : {resolve_post_processor_type(cfg)}')
    print('=' * 80)

    if cfg.exp.run_infer == 'single':
        t1 = time.time()
        score_calculator = ScoreCalculator(cfg)
        stats_dict = score_calculator.metrics()
        print(f'\n[Done] Score Calculation Time: {time.time() - t1:.2f} sec')
        for key, values in stats_dict.items():
            print(f'{key}: {np.mean(values):.4f}')

    elif cfg.exp.run_infer == 'multi':
        model_name = get_model_name(cfg)
        ckpt_dir = os.path.join(cfg.exp.workspace, 'checkpoints', model_name)
        ckpt_files = sorted(
            [f for f in os.listdir(ckpt_dir) if f.endswith('_iteration.pth')],
            key=lambda x: int(x.replace('_iteration.pth', '')),
        )
        print(f'Found {len(ckpt_files)} checkpoints in {ckpt_dir}')

        wb = None
        if getattr(cfg.wandb, 'enable', False) and wandb is not None:
            wb = wandb.init(project=cfg.wandb.project, name=f'eval_{cfg.dataset.test_set}_{model_name}')

        records = []
        for idx, ckpt_file in enumerate(ckpt_files):
            ckpt_iteration = ckpt_file.replace('_iteration.pth', '')
            cfg.exp.ckpt_iteration = ckpt_iteration
            print('-' * 60)
            print(f'[{idx + 1}/{len(ckpt_files)}] Evaluating: {ckpt_iteration}_iteration.pth')
            t1 = time.time()
            score_calculator = ScoreCalculator(cfg)
            stats_dict = score_calculator.metrics()
            elapsed = time.time() - t1
            print(f'[Done] Time: {elapsed:.2f} sec')

            eval_results = {'iteration': int(ckpt_iteration)}
            for key, values in stats_dict.items():
                val = float(np.mean(values))
                eval_results[key] = val
                print(f'{key}: {val:.4f}')
            if wb is not None:
                wandb.log(eval_results, step=idx)
            records.append(eval_results)
            del score_calculator
            del stats_dict
            gc.collect()

        if pd is not None and records:
            df = pd.DataFrame(records)
            csv_path = os.path.join(cfg.exp.workspace, 'logs', f'{model_name}_{cfg.dataset.test_set}.csv')
            df.to_csv(csv_path, index=False)
            print(f'\n[Saved] Summary CSV: {csv_path}')

        if wb is not None:
            wandb.finish()
        print('=' * 80)
        print('All checkpoint scores completed.')
        print('=' * 80)
    else:
        raise ValueError("cfg.exp.run_infer must be 'single' or 'multi'")
