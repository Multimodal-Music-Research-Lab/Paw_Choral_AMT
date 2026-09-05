# SPDX-License-Identifier: Apache-2.0
# Includes modifications to ByteDance's piano_transcription implementation.

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import time
from itertools import combinations

import h5py
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
try:
    import sox
except Exception:
    sox = None

from utilities import (
    TargetProcessor,
    build_target_masks,
    create_folder,
    create_logging,
    decode_hdf5_attr,
    float32_to_int16,
    get_dataset_hdf5s_dir,
    get_filename,
    int16_to_float32,
    pad_truncate_sequence,
    plot_waveform_midi_targets,
    read_metadata,
    read_midi,
    traverse_folder,
)


def _sr_tag(cfg):
    return f"sr{int(cfg.feature.sample_rate)}"



def _write_packed_hdf5(hdf5_path, attrs, midi_dict, waveform):
    create_folder(os.path.dirname(hdf5_path))
    with h5py.File(hdf5_path, 'w') as hf:
        for key, value in attrs.items():
            if isinstance(value, str):
                hf.attrs.create(key, data=value.encode())
            else:
                hf.attrs.create(key, data=np.float32(value))
        hf.create_dataset('midi_event', data=[e.encode() for e in midi_dict['midi_event']])
        hf.create_dataset('midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32)
        hf.create_dataset('waveform', data=float32_to_int16(waveform), dtype=np.int16)


def _is_valid_packed_hdf5(hdf5_path):
    if not os.path.exists(hdf5_path):
        return False
    try:
        with h5py.File(hdf5_path, 'r') as hf:
            required_attrs = ['split', 'duration', 'midi_filename', 'audio_filename']
            required_datasets = ['midi_event', 'midi_event_time', 'waveform']
            for key in required_attrs:
                if key not in hf.attrs:
                    return False
            for key in required_datasets:
                if key not in hf:
                    return False
            if hf['waveform'].shape[0] == 0:
                return False
            if len(hf['midi_event']) != len(hf['midi_event_time']):
                return False
        return True
    except Exception:
        return False


def _load_split_map(dataset_dir):
    split_map = {}
    split_jsons = {
        'train': 'train.json',
        'validation': 'valid.json',
        'test': 'test.json',
    }
    for split_name, json_name in split_jsons.items():
        json_path = os.path.join(dataset_dir, json_name)
        if not os.path.exists(json_path):
            continue
        for stem in json.load(open(json_path)):
            split_map[stem] = split_name
    return split_map


def _pack_youchorale_like_dataset_to_hdf5(cfg, dataset_name: str, dataset_dir: str):
    audio_dir = os.path.join(dataset_dir, 'audio')
    midi_dir = os.path.join(dataset_dir, 'midi')
    hdf5s_dir = get_dataset_hdf5s_dir(cfg, dataset_name)
    logs_dir = os.path.join(cfg.exp.workspace, 'logs', f'{get_filename(__file__)}_{dataset_name}_{_sr_tag(cfg)}')
    create_logging(logs_dir, filemode='w')

    logging.info('Packing %s from %s', dataset_name, dataset_dir)
    start_time = time.time()
    count = 0
    skipped = 0
    split_map = _load_split_map(dataset_dir)

    audio_map = {}
    for name in sorted(os.listdir(audio_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() in {'.wav', '.mp3', '.flac', '.m4a'}:
            audio_map[stem] = os.path.join(audio_dir, name)

    for name in sorted(os.listdir(midi_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in {'.mid', '.midi'}:
            continue

        audio_path = audio_map.get(stem)
        if audio_path is None:
            logging.warning('Skip %s: missing paired audio', name)
            continue

        split = split_map.get(stem)
        if split is None:
            logging.warning('Skip %s: missing split assignment', name)
            continue

        hdf5_path = os.path.join(hdf5s_dir, f'{stem}.h5')
        if _is_valid_packed_hdf5(hdf5_path):
            skipped += 1
            logging.info('skip %s (already packed)', stem)
            continue

        midi_path = os.path.join(midi_dir, name)
        midi_dict = read_midi(midi_path, 'youchorale')
        waveform, _ = librosa.load(audio_path, sr=cfg.feature.sample_rate, mono=True)
        duration = librosa.get_duration(y=waveform, sr=cfg.feature.sample_rate)

        attrs = {
            'split': split,
            'duration': duration,
            'midi_filename': name,
            'audio_filename': os.path.basename(audio_path),
        }
        _write_packed_hdf5(hdf5_path, attrs, midi_dict, waveform)
        count += 1
        logging.info('%d %s', count, stem)

    logging.info('Write HDF5 to %s', hdf5s_dir)
    logging.info('Total files: %d', count)
    logging.info('Skipped existing valid files: %d', skipped)
    logging.info('Time: %.3f s', time.time() - start_time)



def _hz_to_midi_number(freq_hz: float) -> int:
    return int(np.round(69.0 + 12.0 * np.log2(freq_hz / 440.0)))


def _load_audio_mono_resampled(audio_path: str, sample_rate: int) -> np.ndarray:
    waveform, original_sr = sf.read(audio_path, dtype='float32')
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    if int(original_sr) != int(sample_rate):
        gcd = math.gcd(int(original_sr), int(sample_rate))
        waveform = resample_poly(waveform, int(sample_rate) // gcd, int(original_sr) // gcd).astype(np.float32)
    return waveform.astype(np.float32)


def _read_csd_note_lab(lab_path: str):
    note_events = []
    with open(lab_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            onset_time = float(parts[0])
            freq_hz = float(parts[1])
            duration = float(parts[2])
            if freq_hz <= 0.0 or duration <= 0.0:
                continue
            note_events.append(
                {
                    'midi_note': _hz_to_midi_number(freq_hz),
                    'onset_time': onset_time,
                    'offset_time': onset_time + duration,
                    'velocity': 100,
                }
            )
    note_events.sort(key=lambda x: (x['onset_time'], x['midi_note'], x['offset_time']))
    return note_events


def _note_events_to_midi_dict(note_events):
    midi_messages = []
    for event in note_events:
        midi_note = int(event['midi_note'])
        velocity = int(event.get('velocity', 100))
        midi_messages.append((float(event['onset_time']), 1, f'note_on channel=0 note={midi_note} velocity={velocity} time=0'))
        midi_messages.append((float(event['offset_time']), 0, f'note_off channel=0 note={midi_note} velocity=0 time=0'))

    midi_messages.sort(key=lambda x: (x[0], x[1], x[2]))
    return {
        'midi_event': np.array([msg for _, _, msg in midi_messages]),
        'midi_event_time': np.array([time_sec for time_sec, _, _ in midi_messages], dtype=np.float32),
    }


def _voice_note_bars_from_csd(voice_events):
    bar = {'measure': 0}
    for voice_name in ['soprano', 'alto', 'tenor', 'bass']:
        bar[voice_name] = [
            [event['midi_note'], event.get('velocity', 100), 0, event['onset_time'], event['offset_time']]
            for event in voice_events.get(voice_name, [])
        ]
    return [bar]


def pack_csd_dataset_to_hdf5(cfg):
    dataset_dir = cfg.dataset.csd_dir
    hdf5s_dir = get_dataset_hdf5s_dir(cfg, 'csd')
    note_dir = os.path.join(dataset_dir, 'note')
    logs_dir = os.path.join(cfg.exp.workspace, 'logs', f'{get_filename(__file__)}_csd_{_sr_tag(cfg)}')
    create_logging(logs_dir, filemode='w')
    create_folder(hdf5s_dir)
    create_folder(note_dir)

    logging.info('Packing CSD from %s', dataset_dir)
    start_time = time.time()
    song_codes = sorted({
        name.split('_')[1]
        for name in os.listdir(dataset_dir)
        if name.startswith('CSD_') and name.endswith('_notes.lab')
    })
    voice_names = ['soprano', 'alto', 'tenor', 'bass']

    for song_code in song_codes:
        stem = f'CSD_{song_code}'
        voice_events = {}
        merged_note_events = []
        mix_waveform = None
        num_tracks = 0
        durations = []

        for voice_name in voice_names:
            lab_path = os.path.join(dataset_dir, f'{stem}_{voice_name}_notes.lab')
            voice_events[voice_name] = _read_csd_note_lab(lab_path)
            merged_note_events.extend(voice_events[voice_name])

            for singer_idx in range(1, 5):
                wav_path = os.path.join(dataset_dir, f'{stem}_{voice_name}_{singer_idx}.wav')
                waveform = _load_audio_mono_resampled(wav_path, cfg.feature.sample_rate)
                durations.append(len(waveform) / float(cfg.feature.sample_rate))
                if mix_waveform is None:
                    mix_waveform = waveform
                else:
                    if len(waveform) > len(mix_waveform):
                        mix_waveform = np.pad(mix_waveform, (0, len(waveform) - len(mix_waveform)))
                    elif len(waveform) < len(mix_waveform):
                        waveform = np.pad(waveform, (0, len(mix_waveform) - len(waveform)))
                    mix_waveform += waveform
                num_tracks += 1

        if mix_waveform is None or num_tracks == 0:
            logging.warning('Skip %s: no audio tracks found', stem)
            continue

        mix_waveform /= float(num_tracks)
        merged_note_events.sort(key=lambda x: (x['onset_time'], x['midi_note'], x['offset_time']))
        midi_dict = _note_events_to_midi_dict(merged_note_events)

        hdf5_path = os.path.join(hdf5s_dir, f'{stem}.h5')
        attrs = {
            'split': 'test',
            'duration': max(durations) if durations else len(mix_waveform) / float(cfg.feature.sample_rate),
            'midi_filename': f'{stem}_merged.mid',
            'audio_filename': f'{stem}_mix16.wav',
        }
        _write_packed_hdf5(hdf5_path, attrs, midi_dict, mix_waveform)

        with open(os.path.join(note_dir, f'{stem}.pkl'), 'wb') as f:
            pickle.dump(_voice_note_bars_from_csd(voice_events), f)

        logging.info('Packed %s', stem)

    logging.info('Write HDF5 to %s', hdf5s_dir)
    logging.info('Write note pkls to %s', note_dir)
    logging.info('Time: %.3f s', time.time() - start_time)


def _f0_csv_to_note_events(csv_path: str, min_confidence: float = 0.35, min_note_duration: float = 0.08):
    rows = []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 3:
                continue
            time_sec, freq_hz, confidence = map(float, parts[:3])
            rows.append((time_sec, freq_hz, confidence))

    if not rows:
        return []

    times = np.asarray([row[0] for row in rows], dtype=np.float32)
    freqs = np.asarray([row[1] for row in rows], dtype=np.float32)
    confs = np.asarray([row[2] for row in rows], dtype=np.float32)
    voiced = (freqs > 0.0) & (confs >= float(min_confidence))

    if not np.any(voiced):
        return []

    midi_float = np.zeros_like(freqs)
    valid_freq = freqs > 0.0
    midi_float[valid_freq] = 69.0 + 12.0 * np.log2(freqs[valid_freq] / 440.0)
    midi_int = np.round(midi_float).astype(np.int32)
    frame_step = float(np.median(np.diff(times))) if len(times) > 1 else 0.01

    note_events = []
    idx = 0
    while idx < len(times):
        if not voiced[idx]:
            idx += 1
            continue

        start_idx = idx
        current_pitch = midi_int[idx]
        idx += 1
        while idx < len(times) and voiced[idx] and midi_int[idx] == current_pitch:
            idx += 1
        end_idx = idx - 1

        onset_time = float(times[start_idx])
        offset_time = float(times[end_idx] + frame_step)
        if offset_time - onset_time >= float(min_note_duration):
            note_events.append(
                {
                    'midi_note': int(current_pitch),
                    'onset_time': onset_time,
                    'offset_time': offset_time,
                    'velocity': 100,
                }
            )

    return note_events


def _voice_note_bars_from_letter_events(voice_events):
    bar = {'measure': 0}
    for voice_name in ['S', 'A', 'T', 'B']:
        bar[voice_name] = [
            [event['midi_note'], event.get('velocity', 100), 0, event['onset_time'], event['offset_time']]
            for event in voice_events.get(voice_name, [])
        ]
    return [bar]


def pack_cantoria_dataset_to_hdf5(cfg):
    dataset_dir = cfg.dataset.cantoria_dir
    audio_dir = os.path.join(dataset_dir, 'Audio')
    f0_source = str(getattr(cfg.dataset, 'cantoria_f0_source', 'crepe')).lower()
    f0_dir = os.path.join(dataset_dir, 'F0_crepe' if f0_source == 'crepe' else 'F0_pyin')
    note_dir = os.path.join(dataset_dir, f'note_{f0_source}')
    hdf5s_dir = get_dataset_hdf5s_dir(cfg, 'cantoria')
    logs_dir = os.path.join(cfg.exp.workspace, 'logs', f'{get_filename(__file__)}_cantoria_{f0_source}_{_sr_tag(cfg)}')
    create_logging(logs_dir, filemode='w')
    create_folder(hdf5s_dir)
    create_folder(note_dir)

    logging.info('Packing Cantoria from %s using %s F0', dataset_dir, f0_source)
    start_time = time.time()
    stems = sorted(name[:-8] for name in os.listdir(audio_dir) if name.endswith('_Mix.wav'))

    for stem in stems:
        mix_path = os.path.join(audio_dir, f'{stem}_Mix.wav')
        waveform = _load_audio_mono_resampled(mix_path, cfg.feature.sample_rate)
        voice_events = {}
        merged_note_events = []

        for voice_name in ['S', 'A', 'T', 'B']:
            csv_path = os.path.join(f0_dir, f'{stem}_{voice_name}.csv')
            note_events = _f0_csv_to_note_events(
                csv_path,
                min_confidence=float(getattr(cfg.dataset, 'cantoria_min_confidence', 0.35)),
                min_note_duration=float(getattr(cfg.dataset, 'cantoria_min_note_duration', 0.08)),
            )
            voice_events[voice_name] = note_events
            merged_note_events.extend(note_events)

        merged_note_events.sort(key=lambda x: (x['onset_time'], x['midi_note'], x['offset_time']))
        midi_dict = _note_events_to_midi_dict(merged_note_events)

        hdf5_path = os.path.join(hdf5s_dir, f'{stem}.h5')
        attrs = {
            'split': 'test',
            'duration': len(waveform) / float(cfg.feature.sample_rate),
            'midi_filename': f'{stem}_merged_from_{f0_source}.mid',
            'audio_filename': os.path.basename(mix_path),
        }
        _write_packed_hdf5(hdf5_path, attrs, midi_dict, waveform)

        with open(os.path.join(note_dir, f'{stem}.pkl'), 'wb') as f:
            pickle.dump(_voice_note_bars_from_letter_events(voice_events), f)

        logging.info('Packed %s', stem)

    logging.info('Write HDF5 to %s', hdf5s_dir)
    logging.info('Write note pkls to %s', note_dir)
    logging.info('Time: %.3f s', time.time() - start_time)


def pack_maestro_dataset_to_hdf5(cfg):
    dataset_dir = cfg.dataset.maestro_dir
    csv_path = os.path.join(dataset_dir, 'maestro-v3.0.0.csv')
    hdf5s_dir = get_dataset_hdf5s_dir(cfg, 'maestro')
    logs_dir = os.path.join(cfg.exp.workspace, 'logs', f'{get_filename(__file__)}_maestro_{_sr_tag(cfg)}')
    create_logging(logs_dir, filemode='w')

    meta_dict = read_metadata(csv_path)
    audios_num = len(meta_dict['audio_filename'])
    logging.info('Packing MAESTRO from %s', dataset_dir)
    logging.info('Total files: %d', audios_num)
    start_time = time.time()

    for idx in range(audios_num):
        midi_path = os.path.join(dataset_dir, meta_dict['midi_filename'][idx])
        audio_path = os.path.join(dataset_dir, meta_dict['audio_filename'][idx])
        midi_dict = read_midi(midi_path, 'maestro')
        waveform, _ = librosa.load(audio_path, sr=cfg.feature.sample_rate, mono=True)

        hdf5_path = os.path.join(
            hdf5s_dir,
            f"{os.path.splitext(meta_dict['audio_filename'][idx])[0]}.h5",
        )
        attrs = {
            'canonical_composer': meta_dict['canonical_composer'][idx],
            'canonical_title': meta_dict['canonical_title'][idx],
            'split': meta_dict['split'][idx],
            'year': meta_dict['year'][idx],
            'midi_filename': meta_dict['midi_filename'][idx],
            'audio_filename': meta_dict['audio_filename'][idx],
            'duration': meta_dict['duration'][idx],
        }
        _write_packed_hdf5(hdf5_path, attrs, midi_dict, waveform)
        logging.info('%d/%d %s', idx + 1, audios_num, meta_dict['audio_filename'][idx])

    logging.info('Write HDF5 to %s', hdf5s_dir)
    logging.info('Time: %.3f s', time.time() - start_time)



def pack_maps_dataset_to_hdf5(cfg):
    dataset_dir = cfg.dataset.maps_dir
    hdf5s_dir = get_dataset_hdf5s_dir(cfg, 'maps')
    logs_dir = os.path.join(cfg.exp.workspace, 'logs', f'{get_filename(__file__)}_maps_{_sr_tag(cfg)}')
    create_logging(logs_dir, filemode='w')

    pianos = ['ENSTDkCl', 'ENSTDkAm']
    logging.info('Packing MAPS from %s', dataset_dir)
    start_time = time.time()
    count = 0

    for piano in pianos:
        sub_dir = os.path.join(dataset_dir, piano, 'MUS')
        for name in sorted(os.listdir(sub_dir)):
            audio_name, ext = os.path.splitext(name)
            if ext.lower() != '.mid':
                continue

            midi_path = os.path.join(sub_dir, f'{audio_name}.mid')
            audio_path = os.path.join(sub_dir, f'{audio_name}.wav')
            midi_dict = read_midi(midi_path, 'maps')
            waveform, _ = librosa.load(audio_path, sr=cfg.feature.sample_rate, mono=True)
            duration = librosa.get_duration(y=waveform, sr=cfg.feature.sample_rate)

            hdf5_path = os.path.join(hdf5s_dir, f'{audio_name}.h5')
            attrs = {
                'split': 'test',
                'duration': duration,
                'midi_filename': f'{audio_name}.mid',
                'audio_filename': f'{audio_name}.wav',
            }
            _write_packed_hdf5(hdf5_path, attrs, midi_dict, waveform)
            count += 1
            logging.info('%d %s', count, audio_name)

    logging.info('Write HDF5 to %s', hdf5s_dir)
    logging.info('Total files: %d', count)
    logging.info('Time: %.3f s', time.time() - start_time)



def pack_smd_dataset_to_hdf5(cfg):
    dataset_dir = cfg.dataset.smd_dir
    hdf5s_dir = get_dataset_hdf5s_dir(cfg, 'smd')
    logs_dir = os.path.join(cfg.exp.workspace, 'logs', f'{get_filename(__file__)}_smd_{_sr_tag(cfg)}')
    create_logging(logs_dir, filemode='w')

    logging.info('Packing SMD from %s', dataset_dir)
    start_time = time.time()
    count = 0

    for name in sorted(os.listdir(dataset_dir)):
        audio_name, ext = os.path.splitext(name)
        if ext.lower() != '.mid':
            continue

        midi_path = os.path.join(dataset_dir, f'{audio_name}.mid')
        audio_path = os.path.join(dataset_dir, f'{audio_name}.mp3')
        if not os.path.exists(audio_path):
            continue

        midi_dict = read_midi(midi_path, 'smd')
        waveform, _ = librosa.load(audio_path, sr=cfg.feature.sample_rate, mono=True)
        duration = librosa.get_duration(y=waveform, sr=cfg.feature.sample_rate)

        hdf5_path = os.path.join(hdf5s_dir, f'{audio_name}.h5')
        attrs = {
            'split': 'test',
            'duration': duration,
            'midi_filename': f'{audio_name}.mid',
            'audio_filename': f'{audio_name}.mp3',
        }
        _write_packed_hdf5(hdf5_path, attrs, midi_dict, waveform)
        count += 1
        logging.info('%d %s', count, audio_name)

    logging.info('Write HDF5 to %s', hdf5s_dir)
    logging.info('Total files: %d', count)
    logging.info('Time: %.3f s', time.time() - start_time)


def pack_youchorale_dataset_to_hdf5(cfg):
    _pack_youchorale_like_dataset_to_hdf5(cfg, 'youchorale', cfg.dataset.youchorale_dir)


def pack_youchorale_pro_dataset_to_hdf5(cfg):
    _pack_youchorale_like_dataset_to_hdf5(cfg, 'youchorale_pro', cfg.dataset.youchorale_pro_dir)


def _get_choral_note_dir(cfg, dataset_name: str) -> str:
    if dataset_name == 'youchorale':
        return os.path.join(cfg.dataset.youchorale_dir, 'note')
    if dataset_name == 'youchorale_pro':
        return os.path.join(cfg.dataset.youchorale_pro_dir, 'note')
    if dataset_name == 'csd':
        return os.path.join(cfg.dataset.csd_dir, 'note')
    if dataset_name == 'cantoria':
        return os.path.join(cfg.dataset.cantoria_dir, f'note_{cfg.dataset.cantoria_f0_source}')
    raise ValueError(f'No choral note directory configured for dataset={dataset_name}')


class BasePianoDataset:
    def __init__(self, cfg, dataset_name: str, is_training: bool):
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.is_training = is_training
        self.random_state = np.random.RandomState(cfg.exp.random_seed)
        self.hdf5s_dir = get_dataset_hdf5s_dir(cfg, dataset_name)
        self.segment_samples = int(cfg.feature.sample_rate * cfg.feature.segment_seconds)
        self.target_processor = TargetProcessor(cfg.feature.segment_seconds, cfg)

    def _get_hdf5_path(self, meta):
        if self.dataset_name == 'maestro':
            year, hdf5_name, _ = meta
            return os.path.join(self.hdf5s_dir, year, hdf5_name)
        hdf5_name, _ = meta
        return os.path.join(self.hdf5s_dir, hdf5_name)

    def _get_start_time(self, meta):
        return float(meta[-1])

    def _get_note_shift(self):
        if not self.is_training or self.cfg.feature.max_note_shift <= 0:
            return 0
        return int(self.random_state.randint(-self.cfg.feature.max_note_shift, self.cfg.feature.max_note_shift + 1))

    def __getitem__(self, meta):
        hdf5_path = self._get_hdf5_path(meta)
        start_time = self._get_start_time(meta)
        note_shift = self._get_note_shift()

        with h5py.File(hdf5_path, 'r') as hf:
            start_sample = int(start_time * self.cfg.feature.sample_rate)
            end_sample = start_sample + self.segment_samples
            if end_sample >= hf['waveform'].shape[0]:
                start_sample = max(0, hf['waveform'].shape[0] - self.segment_samples)
                end_sample = start_sample + self.segment_samples

            waveform = int16_to_float32(hf['waveform'][start_sample:end_sample])
            waveform = pad_truncate_sequence(waveform, self.segment_samples).astype(np.float32)

            if self.is_training and getattr(self.cfg.feature, 'augmentor', None) is not None:
                waveform = self.cfg.feature.augmentor.augment(waveform)

            if note_shift != 0:
                waveform = librosa.effects.pitch_shift(
                    waveform,
                    sr=self.cfg.feature.sample_rate,
                    n_steps=note_shift,
                    bins_per_octave=12,
                )
                waveform = pad_truncate_sequence(waveform, self.segment_samples).astype(np.float32)

            midi_events = [e.decode() for e in hf['midi_event'][:]]
            midi_events_time = hf['midi_event_time'][:]
            target_dict, note_events, _ = self.target_processor.process(
                start_time,
                midi_events_time,
                midi_events,
                extend_pedal=True,
                note_shift=note_shift,
            )

        target_dict = build_target_masks(self.cfg, target_dict)
        data_dict = {'waveform': waveform}
        data_dict.update(target_dict)

        if self.cfg.exp.debug:
            plot_waveform_midi_targets(data_dict, start_time, note_events, self.cfg)
            raise SystemExit

        return data_dict


class Maestro_Dataset(BasePianoDataset):
    def __init__(self, cfg, is_training: bool = True):
        super().__init__(cfg, 'maestro', is_training)


class MAPS_Dataset(BasePianoDataset):
    def __init__(self, cfg, is_training: bool = True):
        super().__init__(cfg, 'maps', is_training)


class SMD_Dataset(BasePianoDataset):
    def __init__(self, cfg, is_training: bool = True):
        super().__init__(cfg, 'smd', is_training)


class ChoralSATBDataset(BasePianoDataset):
    def __init__(self, cfg, dataset_name: str, is_training: bool = True):
        super().__init__(cfg, dataset_name, is_training)
        if int(getattr(cfg.feature, 'max_note_shift', 0)) != 0:
            raise ValueError(
                'Online note shifting is disabled for ChoralSATBDataset because the current '
                'voice-target builder does not shift SATB labels. Use a pre-generated, '
                'synchronously transposed dataset or implement and test label shifting first.'
            )
        self.note_dir = _get_choral_note_dir(cfg, dataset_name)
        self.frames_per_second = cfg.feature.frames_per_second
        self.frames_num = int(round(cfg.feature.segment_seconds * self.frames_per_second)) + 1
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
        self.note_cache = {}

    def _get_stem(self, meta):
        return os.path.splitext(meta[0])[0]

    def _load_note_bars(self, stem: str):
        if stem not in self.note_cache:
            note_path = os.path.join(self.note_dir, f'{stem}.pkl')
            if not os.path.exists(note_path):
                raise FileNotFoundError(
                    f'Missing SATB note annotation for {stem}: {note_path}. '
                    'Refusing to train against an implicit all-zero label.'
                )
            with open(note_path, 'rb') as f:
                self.note_cache[stem] = pickle.load(f)
        return self.note_cache[stem]

    def _voice_index(self, part_name: str):
        if not part_name:
            return None
        part_head = part_name[0].upper()
        if part_head not in self.voice_names:
            return None
        return self.voice_names.index(part_head)

    def _normalize_voice_setting(self, values, default_values):
        values = default_values if values is None else list(values)
        if len(values) != self.num_voices:
            raise ValueError(
                f'voice assignment config expects {self.num_voices} values, got {len(values)}'
            )
        return [float(v) for v in values]

    def _bars_to_note_events(self, note_bars):
        events = []
        for bar in note_bars:
            if not isinstance(bar, dict):
                continue
            for part_name, note_list in bar.items():
                if part_name == 'measure':
                    continue
                if not isinstance(note_list, list):
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

    def _build_voice_rolls(self, stem: str, start_time: float):
        voice_frame_roll = np.zeros((self.frames_num, self.num_voices, self.classes_num), dtype=np.float32)
        voice_onset_roll = np.zeros_like(voice_frame_roll)
        voice_offset_roll = np.zeros_like(voice_frame_roll)
        voice_presence = np.zeros((self.num_voices,), dtype=np.float32)

        segment_end = start_time + self.cfg.feature.segment_seconds
        events = self._bars_to_voice_events(self._load_note_bars(stem))

        for voice_idx, midi_note, onset_time, offset_time in events:
            if offset_time <= start_time or onset_time >= segment_end:
                continue

            note_idx = midi_note - self.begin_note
            local_onset = max(onset_time, start_time)
            local_offset = min(offset_time, segment_end)

            onset_frame = int(np.clip(np.round((local_onset - start_time) * self.frames_per_second), 0, self.frames_num - 1))
            offset_frame = int(np.clip(np.round((local_offset - start_time) * self.frames_per_second), 0, self.frames_num - 1))
            if offset_frame < onset_frame:
                offset_frame = onset_frame

            voice_frame_roll[onset_frame : offset_frame + 1, voice_idx, note_idx] = 1.0
            if start_time <= onset_time < segment_end:
                true_onset_frame = int(np.clip(np.round((onset_time - start_time) * self.frames_per_second), 0, self.frames_num - 1))
                voice_onset_roll[true_onset_frame, voice_idx, note_idx] = 1.0
            if getattr(self.target_processor, 'frames_per_second', self.frames_per_second) and start_time <= offset_time < segment_end:
                true_offset_frame = int(np.clip(np.round((offset_time - start_time) * self.frames_per_second), 0, self.frames_num - 1))
                voice_offset_roll[true_offset_frame, voice_idx, note_idx] = 1.0
            voice_presence[voice_idx] = 1.0

        return {
            'voice_frame_roll': voice_frame_roll,
            'voice_onset_roll': voice_onset_roll,
            'voice_offset_roll': voice_offset_roll,
            'voice_presence': voice_presence,
        }

    def __getitem__(self, meta):
        data_dict = super().__getitem__(meta)
        start_time = self._get_start_time(meta)
        stem = self._get_stem(meta)

        voice_target_dict = self._build_voice_rolls(stem, start_time)
        frame_mask_roll = data_dict['frame_mask_roll']
        onset_mask_roll = data_dict['onset_mask_roll']
        offset_mask_roll = data_dict['offset_mask_roll']

        voice_target_dict['voice_frame_mask_roll'] = np.repeat(frame_mask_roll[:, None, :], self.num_voices, axis=1)
        voice_target_dict['voice_onset_mask_roll'] = np.repeat(onset_mask_roll[:, None, :], self.num_voices, axis=1)
        voice_target_dict['voice_offset_mask_roll'] = np.repeat(offset_mask_roll[:, None, :], self.num_voices, axis=1)

        data_dict.update(voice_target_dict)
        return data_dict


class Augmentor:
    def __init__(self, cfg):
        self.sample_rate = cfg.feature.sample_rate
        self.random_state = np.random.RandomState(cfg.exp.random_seed)

    def augment(self, x):
        if sox is None:
            raise ImportError('sox is required for waveform augmentation')

        clip_samples = len(x)
        logging.getLogger('sox').propagate = False

        tfm = sox.Transformer()
        tfm.set_globals(verbosity=0)
        tfm.pitch(self.random_state.uniform(-0.1, 0.1, 1)[0])
        tfm.contrast(self.random_state.uniform(0, 100, 1)[0])
        tfm.equalizer(
            frequency=self.loguniform(32, 4096, 1)[0],
            width_q=self.random_state.uniform(1, 2, 1)[0],
            gain_db=self.random_state.uniform(-30, 10, 1)[0],
        )
        tfm.equalizer(
            frequency=self.loguniform(32, 4096, 1)[0],
            width_q=self.random_state.uniform(1, 2, 1)[0],
            gain_db=self.random_state.uniform(-30, 10, 1)[0],
        )
        tfm.reverb(reverberance=self.random_state.uniform(0, 70, 1)[0])

        aug_x = tfm.build_array(input_array=x, sample_rate_in=self.sample_rate)
        return pad_truncate_sequence(aug_x, clip_samples).astype(np.float32)

    def loguniform(self, low, high, size):
        return np.exp(self.random_state.uniform(np.log(low), np.log(high), size))


class Sampler:
    def __init__(self, cfg, split, is_eval=None):
        assert split in ['train', 'validation', 'test']
        self.cfg = cfg
        self.split = split
        self.batch_size = cfg.exp.batch_size
        self.segment_seconds = cfg.feature.segment_seconds
        self.hop_seconds = cfg.feature.hop_seconds
        self.random_state = np.random.RandomState(cfg.exp.random_seed)
        self.mini_data = cfg.exp.mini_data
        # For evaluation loaders, `is_eval` tells us which dataset we are actually
        # iterating over (e.g. validate on youchorale while training on
        # youchorale_pro). Falling back to train_set here breaks cross-dataset
        # validation because the sampler and dataset point at different HDF5 roots.
        self.dataset_type = is_eval if is_eval is not None else cfg.dataset.train_set
        self.hdf5s_dir = get_dataset_hdf5s_dir(cfg, self.dataset_type)

        _, hdf5_paths = traverse_folder(self.hdf5s_dir)
        self.segment_list = []
        file_counter = 0

        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as hf:
                if decode_hdf5_attr(hf.attrs['split']) != split:
                    continue

                audio_name = os.path.basename(hdf5_path)
                if self.dataset_type == 'maestro':
                    file_id = [decode_hdf5_attr(hf.attrs['year']), audio_name]
                else:
                    file_id = [audio_name]

                start_time = 0.0
                duration = float(hf.attrs['duration'])
                while start_time + self.segment_seconds < duration:
                    self.segment_list.append(file_id + [start_time])
                    start_time += self.hop_seconds

                file_counter += 1
                if self.mini_data and file_counter >= 10:
                    break

        logging.info('%s %s segments: %d', 'eval' if is_eval else 'train', split, len(self.segment_list))
        self.pointer = 0
        self.segment_indexes = np.arange(len(self.segment_list))
        if len(self.segment_indexes) > 0:
            self.random_state.shuffle(self.segment_indexes)
        else:
            raise RuntimeError(
                f'No segments found for dataset={self.dataset_type}, split={split}, hdf5s_dir={self.hdf5s_dir}'
            )

    def __iter__(self):
        while True:
            batch_segment_list = []
            for _ in range(self.batch_size):
                index = self.segment_indexes[self.pointer]
                self.pointer += 1
                if self.pointer >= len(self.segment_indexes):
                    self.pointer = 0
                    self.random_state.shuffle(self.segment_indexes)
                batch_segment_list.append(self.segment_list[index])
            yield batch_segment_list

    def __len__(self):
        if self.batch_size == 0:
            return 0
        return int(np.ceil(len(self.segment_list) / self.batch_size))

    def state_dict(self):
        return {'pointer': self.pointer, 'segment_indexes': self.segment_indexes}

    def load_state_dict(self, state):
        self.pointer = state['pointer']
        self.segment_indexes = state['segment_indexes']


class EvalSampler(Sampler):
    def __init__(self, cfg, split, is_eval=None):
        super().__init__(cfg, split, is_eval=is_eval)
        self.max_evaluate_iteration = 20

    def __iter__(self):
        pointer = 0
        iteration = 0
        while iteration < self.max_evaluate_iteration and len(self.segment_indexes) > 0:
            batch_segment_list = []
            for _ in range(self.batch_size):
                if pointer >= len(self.segment_indexes):
                    pointer = 0
                index = self.segment_indexes[pointer]
                pointer += 1
                batch_segment_list.append(self.segment_list[index])
            iteration += 1
            yield batch_segment_list



def collate_fn(list_data_dict):
    return {key: np.array([data_dict[key] for data_dict in list_data_dict]) for key in list_data_dict[0].keys()}



def main():
    from hydra import compose, initialize

    parser = argparse.ArgumentParser(description='Data packing utilities')
    subparsers = parser.add_subparsers(dest='mode', required=True)
    subparsers.add_parser('pack_maestro_dataset_to_hdf5')
    subparsers.add_parser('pack_maps_dataset_to_hdf5')
    subparsers.add_parser('pack_smd_dataset_to_hdf5')
    subparsers.add_parser('pack_cantoria_dataset_to_hdf5')
    subparsers.add_parser('pack_csd_dataset_to_hdf5')
    subparsers.add_parser('pack_youchorale_dataset_to_hdf5')
    subparsers.add_parser('pack_youchorale_pro_dataset_to_hdf5')
    args, hydra_overrides = parser.parse_known_args()

    initialize(config_path='./', job_name='data_generator', version_base=None)
    cfg = compose(config_name='config', overrides=hydra_overrides)

    mode_to_function = {
        'pack_maestro_dataset_to_hdf5': pack_maestro_dataset_to_hdf5,
        'pack_maps_dataset_to_hdf5': pack_maps_dataset_to_hdf5,
        'pack_smd_dataset_to_hdf5': pack_smd_dataset_to_hdf5,
        'pack_cantoria_dataset_to_hdf5': pack_cantoria_dataset_to_hdf5,
        'pack_csd_dataset_to_hdf5': pack_csd_dataset_to_hdf5,
        'pack_youchorale_dataset_to_hdf5': pack_youchorale_dataset_to_hdf5,
        'pack_youchorale_pro_dataset_to_hdf5': pack_youchorale_pro_dataset_to_hdf5,
    }
    mode_to_function[args.mode](cfg)


if __name__ == '__main__':
    main()
