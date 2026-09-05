# SPDX-License-Identifier: Apache-2.0
# Includes modifications to ByteDance's piano_transcription implementation.

from __future__ import annotations

import os
import pickle
import sys
import time
from copy import deepcopy
from itertools import combinations

import h5py
import numpy as np
import torch
from hydra import compose, initialize
from tqdm import tqdm

from models import build_model
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


class _ChoralTargetBuilder:
    def __init__(self, cfg, segment_seconds: float, start_time: float):
        self.cfg = cfg
        self.segment_seconds = float(segment_seconds)
        self.start_time = float(start_time)
        self.frames_per_second = cfg.feature.frames_per_second
        self.frames_num = int(round(self.segment_seconds * self.frames_per_second)) + 1
        self.begin_note = cfg.feature.begin_note
        self.classes_num = cfg.feature.classes_num
        self.num_voices = int(getattr(cfg.choral, 'num_voices', 4))
        self.voice_names = list(getattr(cfg.choral, 'voice_names', ['S', 'A', 'T', 'B']))
        self.voice_assignment_method = str(getattr(cfg.choral, 'voice_assignment_method', 'part_name')).strip()
        self.voice_assignment_part_penalty = float(getattr(cfg.choral, 'voice_assignment_part_penalty', 2.0))
        self.voice_assignment_continuity_weight = float(getattr(cfg.choral, 'voice_assignment_continuity_weight', 0.35))
        self.voice_assignment_overlap_penalty = float(getattr(cfg.choral, 'voice_assignment_overlap_penalty', 4.0))
        self.voice_assignment_range_mins = self._normalize_voice_setting(
            getattr(cfg.choral, 'voice_assignment_range_mins', [60, 55, 48, 40]),
            [60, 55, 48, 40],
        )
        self.voice_assignment_range_maxs = self._normalize_voice_setting(
            getattr(cfg.choral, 'voice_assignment_range_maxs', [88, 79, 72, 67]),
            [88, 79, 72, 67],
        )
        self.voice_assignment_range_margin = float(getattr(cfg.choral, 'voice_assignment_range_margin', 2.0))
        self.voice_assignment_mask_penalty = float(getattr(cfg.choral, 'voice_assignment_mask_penalty', 8.0))

    def _normalize_voice_setting(self, values, default_values):
        values = default_values if values is None else list(values)
        if len(values) != self.num_voices:
            raise ValueError(
                f'voice assignment config expects {self.num_voices} values, got {len(values)}'
            )
        return [float(v) for v in values]

    def _voice_index(self, part_name: str):
        if not part_name:
            return None
        part_head = part_name[0].upper()
        if part_head not in self.voice_names:
            return None
        return self.voice_names.index(part_head)

    def _bars_to_note_events(self, note_bars):
        events = []
        for bar in note_bars:
            if not isinstance(bar, dict):
                continue
            for part_name, note_list in bar.items():
                if part_name == 'measure' or not isinstance(note_list, list):
                    continue
                for note in note_list:
                    if len(note) < 5:
                        continue
                    midi_note = int(note[0])
                    onset_time = float(note[3])
                    offset_time = float(note[4])
                    if offset_time <= onset_time:
                        offset_time = onset_time + 1e-4
                    if self.begin_note <= midi_note < self.begin_note + self.classes_num:
                        events.append(
                            {
                                'part_name': part_name,
                                'part_voice_idx': self._voice_index(part_name),
                                'midi_note': midi_note,
                                'onset_time': onset_time,
                                'offset_time': offset_time,
                            }
                        )
        events.sort(key=lambda x: (x['onset_time'], -x['midi_note'], x['offset_time']))
        return events

    def _range_cost(self, midi_note: int, voice_idx: int):
        low = self.voice_assignment_range_mins[voice_idx]
        high = self.voice_assignment_range_maxs[voice_idx]
        center = 0.5 * (low + high)
        below = max(0.0, low - midi_note)
        above = max(0.0, midi_note - high)
        return below + above + 0.25 * abs(midi_note - center) / 12.0

    def _voice_in_masked_range(self, midi_note: int, voice_idx: int):
        low = self.voice_assignment_range_mins[voice_idx] - self.voice_assignment_range_margin
        high = self.voice_assignment_range_maxs[voice_idx] + self.voice_assignment_range_margin
        return low <= midi_note <= high

    def _assignment_cost(self, event, voice_idx: int, last_pitch_by_voice=None, last_offset_by_voice=None):
        cost = self._range_cost(event['midi_note'], voice_idx)
        if event.get('part_voice_idx') is not None and event['part_voice_idx'] != voice_idx:
            cost += self.voice_assignment_part_penalty
        if last_pitch_by_voice is not None and voice_idx in last_pitch_by_voice:
            cost += self.voice_assignment_continuity_weight * abs(
                event['midi_note'] - last_pitch_by_voice[voice_idx]
            ) / 12.0
        if (
            last_offset_by_voice is not None
            and voice_idx in last_offset_by_voice
            and event['onset_time'] < last_offset_by_voice[voice_idx]
        ):
            cost += self.voice_assignment_overlap_penalty
        return cost

    def _masked_assignment_cost(self, event, voice_idx: int, last_pitch_by_voice=None, last_offset_by_voice=None):
        cost = self._assignment_cost(
            event,
            voice_idx,
            last_pitch_by_voice=last_pitch_by_voice,
            last_offset_by_voice=last_offset_by_voice,
        )
        if not self._voice_in_masked_range(event['midi_note'], voice_idx):
            cost += self.voice_assignment_mask_penalty
        return cost

    def _events_to_voice_tuples(self, assigned_events):
        return [
            (voice_idx, event['midi_note'], event['onset_time'], event['offset_time'])
            for voice_idx, event in assigned_events
        ]

    def _assign_events_part_name(self, note_events):
        assigned_events = []
        for event in note_events:
            voice_idx = event.get('part_voice_idx')
            if voice_idx is None:
                continue
            assigned_events.append((voice_idx, event))
        return self._events_to_voice_tuples(assigned_events)

    def _assign_events_range_prior(self, note_events):
        assigned_events = []
        for event in note_events:
            best_voice_idx = min(
                range(self.num_voices),
                key=lambda voice_idx: self._assignment_cost(event, voice_idx),
            )
            assigned_events.append((best_voice_idx, event))
        return self._events_to_voice_tuples(assigned_events)

    def _group_by_onset_frame(self, note_events):
        grouped = {}
        for event in note_events:
            onset_frame = int(np.round(event['onset_time'] * self.frames_per_second))
            grouped.setdefault(onset_frame, []).append(event)
        return [grouped[key] for key in sorted(grouped)]

    def _assign_group_ordered_continuity(self, group, last_pitch_by_voice, last_offset_by_voice):
        group = sorted(group, key=lambda x: (-x['midi_note'], x['onset_time'], x['offset_time']))
        if len(group) > self.num_voices:
            return [
                (
                    min(
                        range(self.num_voices),
                        key=lambda voice_idx: self._assignment_cost(
                            event,
                            voice_idx,
                            last_pitch_by_voice=last_pitch_by_voice,
                            last_offset_by_voice=last_offset_by_voice,
                        ),
                    ),
                    event,
                )
                for event in group
            ]

        best_assignment = None
        best_cost = float('inf')
        for voice_combo in combinations(range(self.num_voices), len(group)):
            cost = 0.0
            candidate = []
            for event, voice_idx in zip(group, voice_combo):
                cost += self._assignment_cost(
                    event,
                    voice_idx,
                    last_pitch_by_voice=last_pitch_by_voice,
                    last_offset_by_voice=last_offset_by_voice,
                )
                candidate.append((voice_idx, event))
            if cost < best_cost:
                best_cost = cost
                best_assignment = candidate
        return best_assignment or []

    def _assign_events_ordered_continuity(self, note_events):
        last_pitch_by_voice = {}
        last_offset_by_voice = {}
        assigned_events = []
        for group in self._group_by_onset_frame(note_events):
            group_assignment = self._assign_group_ordered_continuity(
                group,
                last_pitch_by_voice,
                last_offset_by_voice,
            )
            for voice_idx, event in group_assignment:
                last_pitch_by_voice[voice_idx] = event['midi_note']
                last_offset_by_voice[voice_idx] = event['offset_time']
                assigned_events.append((voice_idx, event))
        assigned_events.sort(key=lambda x: (x[1]['onset_time'], x[0], x[1]['midi_note']))
        return self._events_to_voice_tuples(assigned_events)

    def _assign_group_range_masked_continuity(self, group, last_pitch_by_voice, last_offset_by_voice):
        group = sorted(group, key=lambda x: (-x['midi_note'], x['onset_time'], x['offset_time']))
        if len(group) > self.num_voices:
            return [
                (
                    min(
                        range(self.num_voices),
                        key=lambda voice_idx: self._masked_assignment_cost(
                            event,
                            voice_idx,
                            last_pitch_by_voice=last_pitch_by_voice,
                            last_offset_by_voice=last_offset_by_voice,
                        ),
                    ),
                    event,
                )
                for event in group
            ]

        best_assignment = None
        best_cost = float('inf')
        for voice_combo in combinations(range(self.num_voices), len(group)):
            cost = 0.0
            candidate = []
            for event, voice_idx in zip(group, voice_combo):
                cost += self._masked_assignment_cost(
                    event,
                    voice_idx,
                    last_pitch_by_voice=last_pitch_by_voice,
                    last_offset_by_voice=last_offset_by_voice,
                )
                candidate.append((voice_idx, event))
            if cost < best_cost:
                best_cost = cost
                best_assignment = candidate
        return best_assignment or []

    def _assign_events_range_masked_continuity(self, note_events):
        last_pitch_by_voice = {}
        last_offset_by_voice = {}
        assigned_events = []
        for group in self._group_by_onset_frame(note_events):
            group_assignment = self._assign_group_range_masked_continuity(
                group,
                last_pitch_by_voice,
                last_offset_by_voice,
            )
            for voice_idx, event in group_assignment:
                last_pitch_by_voice[voice_idx] = event['midi_note']
                last_offset_by_voice[voice_idx] = event['offset_time']
                assigned_events.append((voice_idx, event))
        assigned_events.sort(key=lambda x: (x[1]['onset_time'], x[0], x[1]['midi_note']))
        return self._events_to_voice_tuples(assigned_events)

    def _bars_to_voice_events(self, note_bars):
        note_events = self._bars_to_note_events(note_bars)
        method = self.voice_assignment_method
        if method == 'part_name':
            return self._assign_events_part_name(note_events)
        if method == 'range_prior':
            return self._assign_events_range_prior(note_events)
        if method == 'ordered_continuity':
            return self._assign_events_ordered_continuity(note_events)
        if method == 'range_masked_continuity':
            return self._assign_events_range_masked_continuity(note_events)
        raise ValueError(f'Unsupported choral.voice_assignment_method: {method}')

    def build(self, note_bars, frame_mask_roll=None, onset_mask_roll=None, offset_mask_roll=None):
        voice_frame_roll = np.zeros((self.frames_num, self.num_voices, self.classes_num), dtype=np.float32)
        voice_onset_roll = np.zeros_like(voice_frame_roll)
        voice_offset_roll = np.zeros_like(voice_frame_roll)
        voice_presence = np.zeros((self.num_voices,), dtype=np.float32)

        segment_end = self.start_time + self.segment_seconds
        events = self._bars_to_voice_events(note_bars)

        for voice_idx, midi_note, onset_time, offset_time in events:
            if offset_time <= self.start_time or onset_time >= segment_end:
                continue

            note_idx = midi_note - self.begin_note
            local_onset = max(onset_time, self.start_time)
            local_offset = min(offset_time, segment_end)

            onset_frame = int(np.clip(np.round((local_onset - self.start_time) * self.frames_per_second), 0, self.frames_num - 1))
            offset_frame = int(np.clip(np.round((local_offset - self.start_time) * self.frames_per_second), 0, self.frames_num - 1))
            if offset_frame < onset_frame:
                offset_frame = onset_frame

            voice_frame_roll[onset_frame : offset_frame + 1, voice_idx, note_idx] = 1.0
            if self.start_time <= onset_time < segment_end:
                true_onset_frame = int(np.clip(np.round((onset_time - self.start_time) * self.frames_per_second), 0, self.frames_num - 1))
                voice_onset_roll[true_onset_frame, voice_idx, note_idx] = 1.0
            if self.start_time <= offset_time < segment_end:
                true_offset_frame = int(np.clip(np.round((offset_time - self.start_time) * self.frames_per_second), 0, self.frames_num - 1))
                voice_offset_roll[true_offset_frame, voice_idx, note_idx] = 1.0
            voice_presence[voice_idx] = 1.0

        frame_mask_roll = np.ones((self.frames_num, self.classes_num), dtype=np.float32) if frame_mask_roll is None else frame_mask_roll
        onset_mask_roll = np.ones((self.frames_num, self.classes_num), dtype=np.float32) if onset_mask_roll is None else onset_mask_roll
        offset_mask_roll = np.ones((self.frames_num, self.classes_num), dtype=np.float32) if offset_mask_roll is None else offset_mask_roll

        return {
            'voice_frame_roll': voice_frame_roll,
            'voice_onset_roll': voice_onset_roll,
            'voice_offset_roll': voice_offset_roll,
            'voice_presence': voice_presence,
            'voice_frame_mask_roll': np.repeat(frame_mask_roll[:, None, :], self.num_voices, axis=1),
            'voice_onset_mask_roll': np.repeat(onset_mask_roll[:, None, :], self.num_voices, axis=1),
            'voice_offset_mask_roll': np.repeat(offset_mask_roll[:, None, :], self.num_voices, axis=1),
        }


def build_choral_target_dict(
    cfg,
    note_bars,
    segment_seconds,
    start_time=0.0,
    frame_mask_roll=None,
    onset_mask_roll=None,
    offset_mask_roll=None,
):
    builder = _ChoralTargetBuilder(cfg=cfg, segment_seconds=segment_seconds, start_time=start_time)
    return builder.build(
        note_bars,
        frame_mask_roll=frame_mask_roll,
        onset_mask_roll=onset_mask_roll,
        offset_mask_roll=offset_mask_roll,
    )


def build_total_dict(output_dict, target_dict, ref_on_off_pairs, ref_midi_notes, ref_pedal_on_off_pairs):
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
    return total_dict


class PianoTranscriber:
    def __init__(self, cfg, checkpoint_path):
        self.cfg = cfg
        self.spec = get_task_spec(cfg)
        self.device = torch.device('cuda') if cfg.exp.cuda and torch.cuda.is_available() else torch.device('cpu')
        self.segment_samples = int(cfg.feature.sample_rate * cfg.feature.segment_seconds)
        self.segment_frames = int(round(cfg.feature.frames_per_second * cfg.feature.segment_seconds)) + 1
        self.model = build_model(cfg)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=False)
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
        x = x[:, :-1, ...]
        segment_frames = x.shape[1]
        assert segment_frames % 4 == 0
        y = [x[0, : int(segment_frames * 0.75)]]
        for i in range(1, x.shape[0] - 1):
            y.append(x[i, int(segment_frames * 0.25) : int(segment_frames * 0.75)])
        y.append(x[-1, int(segment_frames * 0.25) :])
        return np.concatenate(y, axis=0)

    def stitch_output(self, x: np.ndarray, valid_frames: int) -> np.ndarray:
        # Frame-like outputs are segment x time x ... and need overlap-add deframing.
        if x.ndim >= 3 and x.shape[1] == self.segment_frames:
            return self.deframe(x)[:valid_frames]

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
    checkpoint_path = os.path.join(cfg.exp.workspace, 'checkpoints', model_name, f'{cfg.exp.ckpt_iteration}_iteration.pth')
    hdf5s_dir = get_dataset_hdf5s_dir(cfg, cfg.dataset.test_set)
    _, hdf5_paths = traverse_folder(hdf5s_dir)

    probs_dir = os.path.join(
        cfg.exp.workspace,
        'probs',
        cfg.dataset.test_set,
        eval_split,
        model_name,
        f'{cfg.exp.ckpt_iteration}_iteration',
    )
    create_folder(probs_dir)

    transcriber = PianoTranscriber(cfg, checkpoint_path)

    progress_bar = tqdm(hdf5_paths, desc=f'Infer {cfg.exp.ckpt_iteration}', unit='file', ncols=90)
    for hdf5_path in progress_bar:
        with h5py.File(hdf5_path, 'r') as hf:
            if decode_hdf5_attr(hf.attrs['split']) != eval_split:
                continue

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
        if getattr(cfg.choral, 'enable', False):
            note_dir = _get_choral_note_dir(cfg)
            if note_dir is not None:
                note_path = os.path.join(note_dir, f'{get_filename(hdf5_path)}.pkl')
                if os.path.exists(note_path):
                    with open(note_path, 'rb') as f:
                        note_bars = pickle.load(f)
                    target_dict.update(
                        build_choral_target_dict(
                            cfg=cfg,
                            note_bars=note_bars,
                            segment_seconds=segment_seconds,
                            start_time=0.0,
                            frame_mask_roll=target_dict['frame_mask_roll'],
                            onset_mask_roll=target_dict['onset_mask_roll'],
                            offset_mask_roll=target_dict['offset_mask_roll'],
                        )
                    )

        ref_on_off_pairs = np.array([[event['onset_time'], event['offset_time']] for event in note_events], dtype=np.float32)
        ref_midi_notes = np.array([event['midi_note'] for event in note_events], dtype=np.int32)
        ref_pedal_on_off_pairs = np.array([[event['onset_time'], event['offset_time']] for event in pedal_events], dtype=np.float32)

        transcribed_dict = transcriber.transcribe(audio, midi_path=None)
        output_dict = transcribed_dict['output_dict']

        total_dict = build_total_dict(
            output_dict=output_dict,
            target_dict=target_dict,
            ref_on_off_pairs=ref_on_off_pairs,
            ref_midi_notes=ref_midi_notes,
            ref_pedal_on_off_pairs=ref_pedal_on_off_pairs,
        )

        prob_path = os.path.join(probs_dir, f'{get_filename(hdf5_path)}.pkl')
        with open(prob_path, 'wb') as fw:
            pickle.dump(total_dict, fw)


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
