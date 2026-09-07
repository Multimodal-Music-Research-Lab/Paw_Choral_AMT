# SPDX-License-Identifier: Apache-2.0
# Includes modifications to ByteDance's piano_transcription implementation.

from __future__ import annotations

import hashlib
import os
import pickle
import sys
import time
import uuid
from collections.abc import Mapping
from copy import deepcopy

import h5py
import numpy as np
import torch
from hydra import compose, initialize
from tqdm import tqdm

from checkpointing import (
    checkpoint_target_semantics,
    checkpoint_data_target_semantics,
    checkpoint_compatibility_allowlists,
    format_checkpoint_load_report,
    load_model_checkpoint,
    validate_checkpoint_behavior,
)
from canonical_union import build_canonical_union_rolls, canonical_union_events
from choral_targets import (
    build_choral_target_dict,
    require_complete_satb_reference,
    resolve_target_assignment,
)
from models import build_model
from probability_artifacts import (
    CANONICAL_VOICE_NAMES,
    resolve_inference_checkpoint_path,
    sha256_file,
)
from utilities import (
    build_target_masks,
    forward,
    get_task_spec,
    resolve_post_processor_type,
    OnsetsFramesPostProcessor,
    RegressionPostProcessor,
    TargetProcessor,
    create_folder,
    get_dataset_hdf5s_dir,
    get_filename,
    get_model_name,
    int16_to_float32,
    traverse_folder,
    write_events_to_midi,
    decode_hdf5_attr,
)


def build_post_processor(cfg):
    post_type = resolve_post_processor_type(cfg)
    if post_type == 'regression':
        return RegressionPostProcessor(cfg)
    if post_type in {'onsets_frames', 'onf'}:
        return OnsetsFramesPostProcessor(cfg)
    raise ValueError(f'Unsupported post.post_processor_type: {post_type}')


def select_split_hdf5_paths(hdf5_paths, eval_split: str) -> list[str]:
    """Return deterministic packed examples belonging to one evaluation split."""

    selected = []
    for hdf5_path in sorted(hdf5_paths):
        with h5py.File(hdf5_path, 'r') as hf:
            if decode_hdf5_attr(hf.attrs['split']) == eval_split:
                selected.append(hdf5_path)
    stems = [get_filename(path) for path in selected]
    if len(stems) != len(set(stems)):
        duplicates = sorted(stem for stem in set(stems) if stems.count(stem) > 1)
        raise RuntimeError(
            f'Packed evaluation split contains duplicate recording stems: {duplicates[:10]}'
        )
    return selected


def reject_stale_probability_files(probs_dir: str, expected_stems) -> None:
    """Fail rather than silently mix outputs from an older split/manifest."""

    expected = {str(stem) for stem in expected_stems}
    existing = {
        os.path.splitext(name)[0]
        for name in os.listdir(probs_dir)
        if name.endswith('.pkl')
    }
    unexpected = sorted(existing - expected)
    if unexpected:
        preview = unexpected[:10]
        raise RuntimeError(
            f'Probability directory contains {len(unexpected)} stale file(s) '
            f'not present in the current {len(expected)}-recording split: {preview}. '
            'Use a fresh workspace/output directory before inference.'
        )


def read_checkpoint_snapshot(checkpoint_path: str) -> tuple[bytes, str]:
    """Read once so deserialization and provenance refer to identical bytes."""

    with open(checkpoint_path, 'rb') as checkpoint_file:
        snapshot = checkpoint_file.read()
    return snapshot, hashlib.sha256(snapshot).hexdigest()


def reject_missing_checkpoint_parameters(model, load_report) -> None:
    """Do not let an allowlist turn random model initialization into inference."""

    parameter_names = {name for name, _ in model.named_parameters()}
    missing_parameters = tuple(
        key for key in load_report.missing_keys if key in parameter_names
    )
    if missing_parameters:
        raise RuntimeError(
            'Formal inference refuses checkpoint-missing model parameters, even '
            'when a compatibility allowlist matches them, because their retained '
            'initial values are not checkpoint-bound: '
            f'{list(missing_parameters)}'
        )


def _get_choral_note_dir(cfg):
    dataset_name = cfg.dataset.test_set
    if dataset_name == 'youchorale':
        return os.path.join(cfg.dataset.youchorale_dir, 'note')
    if dataset_name == 'youchorale_pro':
        return os.path.join(cfg.dataset.youchorale_pro_dir, 'note')
    if dataset_name == 'csd':
        return os.path.join(cfg.dataset.csd_dir, 'note')
    if dataset_name == 'cantoria':
        return os.path.join(cfg.dataset.cantoria_dir, f'note_{cfg.dataset.cantoria_f0_source}')
    return None


def resolve_evaluation_reference_assignment(cfg) -> str:
    """Resolve the immutable assignment used by formal reference rolls."""

    if bool(getattr(cfg.choral, 'enable', False)):
        voice_names = tuple(
            getattr(cfg.choral, 'voice_names', CANONICAL_VOICE_NAMES)
        )
        if voice_names != CANONICAL_VOICE_NAMES:
            raise ValueError(
                "Formal PawCT inference requires choral.voice_names=['S','A','T','B']; "
                f'received {list(voice_names)!r}'
            )

    reference_assignment = str(
        getattr(cfg.choral, 'evaluation_reference_assignment', 'part_name')
    ).strip().lower().replace('-', '_')
    if reference_assignment != 'part_name':
        raise ValueError(
            'Formal evaluation requires choral.evaluation_reference_assignment=part_name; '
            'training RP/OC assignments cannot be used to rewrite references.'
        )
    return reference_assignment


def _evaluation_reference_cfg(cfg, reference_assignment: str):
    reference_cfg = deepcopy(cfg)
    if hasattr(reference_cfg.choral, 'target_assignment'):
        reference_cfg.choral.target_assignment = reference_assignment
    else:
        reference_cfg.choral.voice_assignment_method = reference_assignment
    if resolve_target_assignment(reference_cfg) != 'part_name':
        raise RuntimeError('Evaluation reference configuration did not resolve to part_name')
    return reference_cfg


def build_evaluation_choral_target_dict(
    cfg,
    note_bars,
    segment_seconds,
    start_time=0.0,
    frame_mask_roll=None,
    onset_mask_roll=None,
    offset_mask_roll=None,
):
    """Build formal SATB reference rolls and masks solely from ``note.pkl``."""

    reference_assignment = resolve_evaluation_reference_assignment(cfg)
    reference_cfg = _evaluation_reference_cfg(cfg, reference_assignment)
    # Packed MIDI masks describe the independently packed MIDI stream. They
    # cannot censor a valid canonical SATB annotation during formal scoring.
    del frame_mask_roll, onset_mask_roll, offset_mask_roll
    reference_targets = build_choral_target_dict(
        cfg=reference_cfg,
        note_bars=note_bars,
        segment_seconds=segment_seconds,
        start_time=start_time,
    )
    spec = get_task_spec(reference_cfg)
    union_targets = build_canonical_union_rolls(
        note_bars,
        start_time=start_time,
        segment_seconds=segment_seconds,
        frames_per_second=float(reference_cfg.feature.frames_per_second),
        begin_note=int(reference_cfg.feature.begin_note),
        classes_num=int(reference_cfg.feature.classes_num),
    )
    canonical_mask = union_targets['frame_mask_roll']
    onset_mask = canonical_mask if spec.onset else np.zeros_like(canonical_mask)
    offset_mask = canonical_mask if spec.offset else np.zeros_like(canonical_mask)
    union_targets['onset_mask_roll'] = onset_mask.copy()
    union_targets['offset_mask_roll'] = offset_mask.copy()
    reference_targets.update(union_targets)
    reference_targets.update({
        'voice_frame_mask_roll': np.repeat(
            canonical_mask[:, None, :], len(CANONICAL_VOICE_NAMES), axis=1
        ),
        'voice_onset_mask_roll': np.repeat(
            onset_mask[:, None, :], len(CANONICAL_VOICE_NAMES), axis=1
        ),
        'voice_offset_mask_roll': np.repeat(
            offset_mask[:, None, :], len(CANONICAL_VOICE_NAMES), axis=1
        ),
    })
    return reference_targets


def build_inference_provenance(cfg, transcriber, evaluation_reference_assignment: str):
    """Summarize checkpoint and label semantics embedded in every probs file."""

    choral_enabled = bool(getattr(cfg.choral, 'enable', False))
    runtime_target_assignment = (
        resolve_target_assignment(cfg) if choral_enabled else 'part_agnostic'
    )
    runtime_target_semantics = (
        checkpoint_target_semantics(cfg)
        if choral_enabled
        else {'method': 'part_agnostic'}
    )
    return {
        'dataset_name': str(cfg.dataset.test_set),
        'model_name': get_model_name(cfg),
        'evaluation_split': str(getattr(cfg.dataset, 'eval_split', 'validation')),
        'evaluation_reference_assignment': evaluation_reference_assignment,
        'inference_run_id': uuid.uuid4().hex,
        'checkpoint_identity': deepcopy(transcriber.checkpoint_identity),
        'checkpoint_model_input_identity': deepcopy(
            getattr(transcriber, 'checkpoint_model_input_identity', None)
        ),
        'checkpoint_load_report': transcriber.checkpoint_load_report.as_dict(),
        'data_target_semantics': {
            'runtime': checkpoint_data_target_semantics(cfg),
            'checkpoint': deepcopy(
                getattr(transcriber, 'checkpoint_data_target_semantics', None)
            ),
        },
        'runtime_model_behavior': {
            'num_voices': int(getattr(cfg.choral, 'num_voices', 4)),
            'use_presence_head': bool(
                getattr(cfg.choral, 'use_presence_head', True)
            ),
            'apply_presence_gate': bool(
                getattr(cfg.choral, 'apply_presence_gate', True)
            ),
            'assignment_module': str(
                getattr(cfg.choral, 'assignment_module', 'heads')
            ).strip(),
            'assignment_temperature': float(
                getattr(cfg.choral, 'assignment_temperature', 1.0)
            ),
            'assignment_hidden_channels': int(
                getattr(cfg.choral, 'assignment_hidden_channels', 64)
            ),
            'assignment_rnn_hidden_size': int(
                getattr(cfg.choral, 'assignment_rnn_hidden_size', 128)
            ),
            'voice_interaction_module': str(
                getattr(cfg.choral, 'voice_interaction_module', 'none')
            ).strip(),
            'voice_interaction_dim': int(
                getattr(cfg.choral, 'voice_interaction_dim', 256)
            ),
            'voice_interaction_heads': int(
                getattr(cfg.choral, 'voice_interaction_heads', 4)
            ),
            'voice_interaction_layers': int(
                getattr(cfg.choral, 'voice_interaction_layers', 1)
            ),
            'voice_interaction_dropout': float(
                getattr(cfg.choral, 'voice_interaction_dropout', 0.1)
            ),
            'voice_names': list(
                getattr(cfg.choral, 'voice_names', ['S', 'A', 'T', 'B'])
            ),
            'allow_checkpoint_behavior_mismatch': bool(
                getattr(
                    getattr(cfg, 'exp', None),
                    'allow_checkpoint_behavior_mismatch',
                    False,
                )
            ),
            'allow_legacy_checkpoint_model_identity': bool(
                getattr(
                    getattr(cfg, 'exp', None),
                    'allow_legacy_checkpoint_model_identity',
                    False,
                )
            ),
            'allow_unsafe_legacy_checkpoint_load': bool(
                getattr(
                    getattr(cfg, 'exp', None),
                    'allow_unsafe_legacy_checkpoint_load',
                    False,
                )
            ),
            'allow_legacy_canonical_union_semantics': bool(
                getattr(
                    getattr(cfg, 'exp', None),
                    'allow_legacy_canonical_union_semantics',
                    False,
                )
            ),
            'allow_unknown_checkpoint_target_assignment': bool(
                getattr(
                    getattr(cfg, 'exp', None),
                    'allow_unknown_checkpoint_target_assignment',
                    False,
                )
            ),
            'allow_checkpoint_target_assignment_mismatch': bool(
                getattr(
                    getattr(cfg, 'exp', None),
                    'allow_checkpoint_target_assignment_mismatch',
                    False,
                )
            ),
        },
        'target_assignment': {
            'runtime_training_method': runtime_target_assignment,
            'runtime_semantics': runtime_target_semantics,
            'preserve_known_part_labels': bool(
                getattr(cfg.choral, 'preserve_known_part_labels', True)
            ),
            'model_assignment_module': str(
                getattr(cfg.choral, 'assignment_module', 'heads')
            ).strip(),
            'checkpoint_metadata': deepcopy(transcriber.checkpoint_target_assignment),
        },
    }


def build_total_dict(
    output_dict,
    target_dict,
    ref_on_off_pairs,
    ref_midi_notes,
    ref_pedal_on_off_pairs,
    provenance=None,
):
    total_dict = {key: output_dict[key] for key in output_dict.keys()}
    total_dict.update(
        {
            'frame_roll': target_dict['frame_roll'],
            'onset_roll': target_dict['onset_roll'],
            'offset_roll': target_dict['offset_roll'],
            'pedal_onset_roll': target_dict['pedal_onset_roll'],
            'pedal_offset_roll': target_dict['pedal_offset_roll'],
            'pedal_frame_roll': target_dict['pedal_frame_roll'],
            'frame_mask_roll': target_dict['frame_mask_roll'],
            'onset_mask_roll': target_dict['onset_mask_roll'],
            'offset_mask_roll': target_dict['offset_mask_roll'],
            'pedal_mask_roll': target_dict['pedal_mask_roll'],
            'ref_on_off_pairs': ref_on_off_pairs,
            'ref_midi_notes': ref_midi_notes,
            'ref_pedal_on_off_pairs': ref_pedal_on_off_pairs,
        }
    )
    for key in (
        'voice_frame_roll',
        'voice_onset_roll',
        'voice_offset_roll',
        'voice_frame_mask_roll',
        'voice_onset_mask_roll',
        'voice_offset_mask_roll',
    ):
        if key in target_dict:
            total_dict[key] = target_dict[key]
    if provenance is not None:
        total_dict['provenance'] = deepcopy(provenance)
    return total_dict


class ChoralAMTTranscriber:
    def __init__(self, cfg, checkpoint_path):
        self.cfg = cfg
        self.spec = get_task_spec(cfg)
        self.device = torch.device('cuda') if cfg.exp.cuda and torch.cuda.is_available() else torch.device('cpu')
        self.segment_samples = int(cfg.feature.sample_rate * cfg.feature.segment_seconds)
        self.segment_frames = int(round(cfg.feature.frames_per_second * cfg.feature.segment_seconds)) + 1
        self.model = build_model(cfg)

        allowed_missing_keys, allowed_unexpected_keys = checkpoint_compatibility_allowlists(cfg)
        checkpoint_snapshot, checkpoint_sha256 = read_checkpoint_snapshot(
            checkpoint_path
        )
        checkpoint, load_report = load_model_checkpoint(
            self.model,
            checkpoint_snapshot,
            map_location=self.device,
            allowed_missing_keys=allowed_missing_keys,
            allowed_unexpected_keys=allowed_unexpected_keys,
            allow_unsafe_legacy_load=bool(
                getattr(
                    cfg.exp,
                    'allow_unsafe_legacy_checkpoint_load',
                    False,
                )
            ),
        )
        del checkpoint_snapshot
        reject_missing_checkpoint_parameters(self.model, load_report)
        validate_checkpoint_behavior(cfg, checkpoint)
        self.checkpoint_load_report = load_report
        checkpoint_iteration = checkpoint.get('iteration')
        if isinstance(checkpoint_iteration, torch.Tensor) and checkpoint_iteration.numel() == 1:
            checkpoint_iteration = checkpoint_iteration.item()
        if isinstance(checkpoint_iteration, np.generic):
            checkpoint_iteration = checkpoint_iteration.item()
        self.checkpoint_identity = {
            'filename': os.path.basename(checkpoint_path),
            'sha256': checkpoint_sha256,
            'iteration': checkpoint_iteration,
            'schema_version': checkpoint.get('schema_version'),
        }
        self.checkpoint_model_input_identity = deepcopy(
            checkpoint.get('model_input_identity')
        )
        checkpoint_target_assignment = checkpoint.get('target_assignment')
        self.checkpoint_target_assignment = (
            deepcopy(checkpoint_target_assignment)
            if isinstance(checkpoint_target_assignment, Mapping)
            else checkpoint_target_assignment
        )
        self.checkpoint_data_target_semantics = deepcopy(
            checkpoint.get('data_target_semantics')
        )
        print(f'Checkpoint load audit: {format_checkpoint_load_report(load_report)}')
        self.model.to(self.device)
        self.post_processor = build_post_processor(cfg)

    def enframe(self, x: np.ndarray, is_audio: bool = True) -> np.ndarray:
        segment_length = self.segment_samples if is_audio else self.segment_frames
        length = x.shape[1] if is_audio else x.shape[0]
        assert length % segment_length == 0
        batch = []
        for pointer in range(0, length - segment_length + 1, segment_length // 2):
            if is_audio:
                batch.append(x[:, pointer : pointer + segment_length])
            else:
                batch.append(x[pointer : pointer + segment_length, :])
        return np.concatenate(batch, axis=0) if is_audio else np.stack(batch, axis=0)

    def deframe(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] == 1:
            return x[0]
        # Adjacent windows share their endpoint frame. Drop every duplicated
        # endpoint while overlap-cropping, then restore the recording's one
        # genuine final endpoint from the last window.
        final_endpoint = x[-1, -1:, ...]
        x = x[:, :-1, ...]
        segment_frames = x.shape[1]
        assert segment_frames % 4 == 0
        y = [x[0, : int(segment_frames * 0.75)]]
        for i in range(1, x.shape[0] - 1):
            y.append(x[i, int(segment_frames * 0.25) : int(segment_frames * 0.75)])
        y.append(x[-1, int(segment_frames * 0.25) :])
        y.append(final_endpoint)
        return np.concatenate(y, axis=0)

    def stitch_output(self, x: np.ndarray, valid_frames: int) -> np.ndarray:
        # Frame-like outputs are segment x time x ... and need overlap-add deframing.
        if x.ndim >= 3 and x.shape[1] == self.segment_frames:
            stitched = self.deframe(x)
            if stitched.shape[0] < valid_frames:
                raise RuntimeError(
                    'Deframed model output is shorter than the audio-derived frame '
                    f'count: output={stitched.shape[0]}, required={valid_frames}'
                )
            return stitched[:valid_frames]

        # Segment-level outputs (e.g. voice presence logits) are aggregated to song-level.
        if x.ndim >= 2:
            return np.max(x, axis=0)

        return x

    def transcribe(self, audio: np.ndarray, midi_path: str | None = None):
        audio = audio[None, :]
        audio_len = audio.shape[1]
        segments_num = int(np.ceil(audio_len / self.segment_samples))
        pad_len = segments_num * self.segment_samples - audio_len
        pad_audio = np.pad(audio, ((0, 0), (0, pad_len)), mode='constant')
        segments = self.enframe(pad_audio, is_audio=True)

        output_dict = forward(self.model, segments, batch_size=max(1, self.cfg.exp.batch_size))
        audio_duration = audio_len / self.cfg.feature.sample_rate
        valid_frames = int(round(audio_duration * self.cfg.feature.frames_per_second)) + 1
        for key in list(output_dict.keys()):
            output_dict[key] = self.stitch_output(output_dict[key], valid_frames)

        post_input = deepcopy(output_dict)
        est_note_events, est_pedal_events = self.post_processor.output_dict_to_midi_events(post_input)

        if midi_path is not None:
            write_events_to_midi(0, est_note_events, est_pedal_events, midi_path)

        return {
            'output_dict': output_dict,
            'est_note_events': est_note_events,
            'est_pedal_events': est_pedal_events,
        }


def infer(cfg):
    model_name = get_model_name(cfg)
    eval_split = str(getattr(cfg.dataset, 'eval_split', 'validation'))
    if eval_split not in {'validation', 'test'}:
        raise ValueError("dataset.eval_split must be 'validation' or 'test'")
    evaluation_reference_assignment = resolve_evaluation_reference_assignment(cfg)
    checkpoints_dir = os.path.join(cfg.exp.workspace, 'checkpoints', model_name)
    checkpoint_path = resolve_inference_checkpoint_path(
        checkpoints_dir,
        cfg.exp.ckpt_iteration,
    )
    hdf5s_dir = get_dataset_hdf5s_dir(cfg, cfg.dataset.test_set)
    choral_note_dir = _get_choral_note_dir(cfg)
    if getattr(cfg.choral, 'enable', False) and choral_note_dir is None:
        raise ValueError(
            f'No SATB note-directory mapping for dataset={cfg.dataset.test_set}'
        )
    _, hdf5_paths = traverse_folder(hdf5s_dir)
    hdf5_paths = select_split_hdf5_paths(hdf5_paths, eval_split)
    if not hdf5_paths:
        raise RuntimeError(
            f'No packed recordings found for split={eval_split} in {hdf5s_dir}'
        )

    probs_dir = os.path.join(
        cfg.exp.workspace,
        'probs',
        cfg.dataset.test_set,
        eval_split,
        model_name,
        f'{cfg.exp.ckpt_iteration}_iteration',
    )
    create_folder(probs_dir)
    reject_stale_probability_files(
        probs_dir,
        (get_filename(path) for path in hdf5_paths),
    )

    transcriber = ChoralAMTTranscriber(cfg, checkpoint_path)
    provenance = build_inference_provenance(
        cfg,
        transcriber,
        evaluation_reference_assignment=evaluation_reference_assignment,
    )

    progress_bar = tqdm(hdf5_paths, desc=f'Infer {cfg.exp.ckpt_iteration}', unit='file', ncols=90)
    for hdf5_path in progress_bar:
        note_path = None
        note_bars = None
        with h5py.File(hdf5_path, 'r') as hf:
            audio = int16_to_float32(hf['waveform'][:])
            midi_events = [e.decode() for e in hf['midi_event'][:]]
            midi_events_time = hf['midi_event_time'][:]

        segment_seconds = len(audio) / cfg.feature.sample_rate
        target_processor = TargetProcessor(segment_seconds=segment_seconds, cfg=cfg)
        target_dict, note_events, pedal_events = target_processor.process(
            start_time=0,
            midi_events_time=midi_events_time,
            midi_events=midi_events,
            extend_pedal=True,
        )
        target_dict = build_target_masks(cfg, target_dict)
        if choral_note_dir is not None:
            note_path = os.path.join(
                choral_note_dir,
                f'{get_filename(hdf5_path)}.pkl',
            )
            if not os.path.exists(note_path):
                raise FileNotFoundError(
                    f'Missing canonical choral reference annotation: {note_path}'
                )
            with open(note_path, 'rb') as f:
                note_bars = pickle.load(f)
            require_complete_satb_reference(
                note_bars,
                note_path,
                begin_note=int(cfg.feature.begin_note),
                classes_num=int(cfg.feature.classes_num),
                recording_duration=segment_seconds,
            )
            canonical_targets = build_canonical_union_rolls(
                note_bars,
                start_time=0.0,
                segment_seconds=segment_seconds,
                frames_per_second=float(cfg.feature.frames_per_second),
                begin_note=int(cfg.feature.begin_note),
                classes_num=int(cfg.feature.classes_num),
            )
            spec = get_task_spec(cfg)
            if not spec.onset:
                canonical_targets['onset_mask_roll'].fill(0.0)
            if not spec.offset:
                canonical_targets['offset_mask_roll'].fill(0.0)
            target_dict.update(canonical_targets)

            if getattr(cfg.choral, 'enable', False):
                target_dict.update(build_evaluation_choral_target_dict(
                    cfg=cfg,
                    note_bars=note_bars,
                    segment_seconds=segment_seconds,
                    start_time=0.0,
                ))

        if note_bars is not None:
            union_events = canonical_union_events(
                note_bars,
                float(cfg.feature.frames_per_second),
                begin_note=int(cfg.feature.begin_note),
                classes_num=int(cfg.feature.classes_num),
            )
            ref_on_off_pairs = np.asarray(
                [[onset, offset] for _, onset, offset in union_events],
                dtype=np.float32,
            ).reshape(-1, 2)
            ref_midi_notes = np.asarray(
                [pitch for pitch, _, _ in union_events],
                dtype=np.int32,
            )
        else:
            ref_on_off_pairs = np.asarray(
                [
                    [event['onset_time'], event['offset_time']]
                    for event in note_events
                ],
                dtype=np.float32,
            ).reshape(-1, 2)
            ref_midi_notes = np.asarray(
                [event['midi_note'] for event in note_events],
                dtype=np.int32,
            )
        ref_pedal_on_off_pairs = np.array([[event['onset_time'], event['offset_time']] for event in pedal_events], dtype=np.float32)

        transcribed_dict = transcriber.transcribe(audio, midi_path=None)
        output_dict = transcribed_dict['output_dict']

        file_provenance = deepcopy(provenance)
        file_provenance['source_artifacts'] = {
            'recording_stem': get_filename(hdf5_path),
            'hdf5_sha256': sha256_file(hdf5_path),
            'reference_note_sha256': (
                sha256_file(note_path) if note_path is not None else None
            ),
        }

        total_dict = build_total_dict(
            output_dict=output_dict,
            target_dict=target_dict,
            ref_on_off_pairs=ref_on_off_pairs,
            ref_midi_notes=ref_midi_notes,
            ref_pedal_on_off_pairs=ref_pedal_on_off_pairs,
            provenance=file_provenance,
        )

        prob_path = os.path.join(probs_dir, f'{get_filename(hdf5_path)}.pkl')
        with open(prob_path, 'wb') as fw:
            pickle.dump(total_dict, fw)

# Historical public import retained for downstream scripts.
PianoTranscriber = ChoralAMTTranscriber


if __name__ == '__main__':
    initialize(config_path='./', job_name='infer', version_base=None)
    cfg = compose(config_name='config', overrides=sys.argv[1:])
    spec = get_task_spec(cfg)

    print('=' * 80)
    print(f'Inference Mode : {cfg.exp.run_infer.upper()}')
    print(f'Model Name     : {get_model_name(cfg)}')
    print(f'Architecture   : {spec.arch}')
    print(f'Task Mode      : {spec.mode}')
    print(f'Test Set       : {cfg.dataset.test_set}')
    print(f'Evaluation Split: {cfg.dataset.eval_split}')
    print(f'Post Processor : {resolve_post_processor_type(cfg)}')
    print(f'Using Device   : {torch.device("cuda") if cfg.exp.cuda and torch.cuda.is_available() else torch.device("cpu")}')
    print('=' * 80)

    if cfg.exp.run_infer == 'single':
        t1 = time.time()
        infer(cfg)
        print(f'\n[Done] Inference time: {time.time() - t1:.2f} sec')
    elif cfg.exp.run_infer == 'multi':
        model_name = get_model_name(cfg)
        ckpt_dir = os.path.join(cfg.exp.workspace, 'checkpoints', model_name)
        ckpt_files = sorted(
            [f for f in os.listdir(ckpt_dir) if f.endswith('_iteration.pth')],
            key=lambda x: int(x.replace('_iteration.pth', '')),
        )
        total_start = time.time()
        for idx, ckpt_file in enumerate(ckpt_files):
            cfg.exp.ckpt_iteration = ckpt_file.replace('_iteration.pth', '')
            tqdm.write('-' * 60)
            tqdm.write(f'[{idx + 1}/{len(ckpt_files)}] {ckpt_file}')
            t1 = time.time()
            infer(cfg)
            tqdm.write(f'[Done] Time: {time.time() - t1:.2f} sec')
        print('\n' + '=' * 80)
        print(f'All checkpoint inference completed in {time.time() - total_start:.2f} sec')
        print('=' * 80)
    else:
        raise ValueError("cfg.exp.run_infer must be 'single' or 'multi'")
