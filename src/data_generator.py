# SPDX-License-Identifier: Apache-2.0
# Includes modifications to ByteDance's piano_transcription implementation.

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import pickle
import time
from collections.abc import Mapping

import h5py
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
try:
    import sox
except Exception:
    sox = None

from canonical_union import build_canonical_union_rolls
from choral_targets import ChoralTargetBuilder, require_complete_satb_reference
from split_manifests import (
    configured_split_manifest_identity,
    select_split_hdf5_paths,
)
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


NUMPY_RANDOM_STATE_FORMAT_VERSION = 1


def serialize_numpy_random_state(random_state):
    """Encode ``RandomState`` using only restricted-unpickler-safe values."""

    if isinstance(random_state, np.random.RandomState):
        random_state = random_state.get_state()
    if not isinstance(random_state, (list, tuple)) or len(random_state) != 5:
        raise TypeError(
            'random_state must be numpy.random.RandomState or its five-item state, got '
            f'{type(random_state).__name__}'
        )
    algorithm, keys, position, has_gauss, cached_gaussian = random_state
    return {
        'format_version': NUMPY_RANDOM_STATE_FORMAT_VERSION,
        'algorithm': str(algorithm),
        'keys': [int(key) for key in np.asarray(keys, dtype=np.uint32)],
        'position': int(position),
        'has_gauss': int(has_gauss),
        'cached_gaussian': float(cached_gaussian),
    }


def deserialize_numpy_random_state(state):
    """Decode a safe RNG mapping or an already-loaded historical tuple."""

    if isinstance(state, Mapping):
        required = {
            'format_version',
            'algorithm',
            'keys',
            'position',
            'has_gauss',
            'cached_gaussian',
        }
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(f'NumPy RNG state is incomplete: missing {missing}')
        version = state['format_version']
        if isinstance(version, bool) or not isinstance(version, (int, np.integer)):
            raise ValueError(f'NumPy RNG format_version must be an integer, got {version!r}')
        if int(version) != NUMPY_RANDOM_STATE_FORMAT_VERSION:
            raise ValueError(
                'Unsupported NumPy RNG format_version '
                f'{version!r}; expected {NUMPY_RANDOM_STATE_FORMAT_VERSION}'
            )

        keys = state['keys']
        if not isinstance(keys, list) or not keys:
            raise ValueError('NumPy RNG keys must be a non-empty list of uint32 integers')
        normalized_keys = []
        for key in keys:
            if isinstance(key, bool) or not isinstance(key, (int, np.integer)):
                raise ValueError('NumPy RNG keys must contain only uint32 integers')
            key = int(key)
            if key < 0 or key > np.iinfo(np.uint32).max:
                raise ValueError(f'NumPy RNG key is outside uint32 range: {key}')
            normalized_keys.append(key)

        position = state['position']
        has_gauss = state['has_gauss']
        if isinstance(position, bool) or not isinstance(position, (int, np.integer)):
            raise ValueError(f'NumPy RNG position must be an integer, got {position!r}')
        if isinstance(has_gauss, bool) or not isinstance(has_gauss, (int, np.integer)):
            raise ValueError(f'NumPy RNG has_gauss must be 0 or 1, got {has_gauss!r}')
        has_gauss = int(has_gauss)
        if has_gauss not in {0, 1}:
            raise ValueError(f'NumPy RNG has_gauss must be 0 or 1, got {has_gauss!r}')
        try:
            cached_gaussian = float(state['cached_gaussian'])
        except (TypeError, ValueError) as exc:
            raise ValueError('NumPy RNG cached_gaussian must be numeric') from exc
        if not math.isfinite(cached_gaussian):
            raise ValueError('NumPy RNG cached_gaussian must be finite')

        decoded = (
            str(state['algorithm']),
            np.asarray(normalized_keys, dtype=np.uint32),
            int(position),
            has_gauss,
            cached_gaussian,
        )
    elif isinstance(state, (list, tuple)) and len(state) == 5:
        # Historical tuples can only enter a file-backed resume after the
        # caller explicitly opts into unsafe legacy checkpoint loading.
        algorithm, keys, position, has_gauss, cached_gaussian = state
        decoded = (
            algorithm,
            np.asarray(keys, dtype=np.uint32),
            position,
            has_gauss,
            cached_gaussian,
        )
    else:
        raise ValueError('NumPy RNG state must be a versioned mapping')

    validator = np.random.RandomState()
    try:
        validator.set_state(decoded)
    except (IndexError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError('Invalid NumPy RNG state') from exc
    return validator.get_state()


def segment_start_times(duration, segment_seconds, hop_seconds):
    """Return deterministic segment starts with complete recording coverage.

    Every non-negative duration produces at least the start at zero. Longer
    recordings also include all regular hop starts that fit before the final
    full-length segment and an exact tail-aligned start. This keeps short and
    exactly-one-segment recordings in the dataset and avoids dropping a tail
    when the hop does not land on it.
    """

    duration = float(duration)
    segment_seconds = float(segment_seconds)
    hop_seconds = float(hop_seconds)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError(f'duration must be finite and non-negative, got {duration!r}')
    if not math.isfinite(segment_seconds) or segment_seconds <= 0:
        raise ValueError(
            f'segment_seconds must be finite and positive, got {segment_seconds!r}'
        )
    if not math.isfinite(hop_seconds) or hop_seconds <= 0:
        raise ValueError(f'hop_seconds must be finite and positive, got {hop_seconds!r}')

    tail_start = max(0.0, duration - segment_seconds)
    starts = [0.0]
    regular_start = hop_seconds
    while regular_start < tail_start:
        starts.append(float(regular_start))
        regular_start += hop_seconds

    if tail_start > 0:
        if math.isclose(starts[-1], tail_start, rel_tol=1e-9, abs_tol=1e-9):
            starts[-1] = float(tail_start)
        else:
            starts.append(float(tail_start))
    return starts


def resolve_segment_bounds(
    requested_start_time,
    waveform_samples,
    sample_rate,
    segment_samples,
):
    """Resolve a requested segment to waveform bounds and its actual start."""

    requested_start_time = float(requested_start_time)
    waveform_samples = int(waveform_samples)
    sample_rate = float(sample_rate)
    segment_samples = int(segment_samples)
    if not math.isfinite(requested_start_time) or requested_start_time < 0:
        raise ValueError(
            'requested_start_time must be finite and non-negative, '
            f'got {requested_start_time!r}'
        )
    if waveform_samples < 0:
        raise ValueError(f'waveform_samples must be non-negative, got {waveform_samples}')
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError(f'sample_rate must be finite and positive, got {sample_rate!r}')
    if segment_samples <= 0:
        raise ValueError(f'segment_samples must be positive, got {segment_samples}')

    requested_start_sample = int(round(requested_start_time * sample_rate))
    maximum_start_sample = max(0, waveform_samples - segment_samples)
    start_sample = min(requested_start_sample, maximum_start_sample)
    end_sample = start_sample + segment_samples
    actual_start_time = start_sample / sample_rate
    return start_sample, end_sample, actual_start_time


_FILE_SHA256_CACHE = {}


def _file_sha256(path, chunk_size=1024 * 1024):
    """Hash immutable sampler inputs without repeatedly rereading large HDF5s."""

    stat_result = os.stat(path)
    cache_key = (
        os.path.realpath(path),
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )
    cached = _FILE_SHA256_CACHE.get(cache_key)
    if cached is not None:
        return cached

    digest = hashlib.sha256()
    with open(path, 'rb') as file_handle:
        while True:
            chunk = file_handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_SHA256_CACHE[cache_key] = value
    return value



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

    def _load_base_item(self, meta):
        hdf5_path = self._get_hdf5_path(meta)
        requested_start_time = self._get_start_time(meta)
        note_shift = self._get_note_shift()

        with h5py.File(hdf5_path, 'r') as hf:
            start_sample, end_sample, actual_start_time = resolve_segment_bounds(
                requested_start_time=requested_start_time,
                waveform_samples=hf['waveform'].shape[0],
                sample_rate=self.cfg.feature.sample_rate,
                segment_samples=self.segment_samples,
            )

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
                actual_start_time,
                midi_events_time,
                midi_events,
                extend_pedal=True,
                note_shift=note_shift,
            )

        target_dict = build_target_masks(self.cfg, target_dict)
        data_dict = {'waveform': waveform}
        data_dict.update(target_dict)

        if self.cfg.exp.debug:
            plot_waveform_midi_targets(data_dict, actual_start_time, note_events, self.cfg)
            raise SystemExit

        return data_dict, actual_start_time

    def __getitem__(self, meta):
        data_dict, _ = self._load_base_item(meta)
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


class ChoralUnionDataset(BasePianoDataset):
    """Use ``note.pkl`` as the single canonical source for choral note targets.

    The packed merged MIDI is retained for backwards-compatible audio storage,
    but it cannot represent overlapping same-pitch notes from different parts:
    the historical event parser keyed active notes by pitch and overwrote one
    voice with another.  Both PagCT and PawCT therefore build their shared
    part-agnostic targets from the complete, track-aware note annotations.
    """

    def __init__(
        self,
        cfg,
        dataset_name: str,
        is_training: bool = True,
        formal_evaluation: bool = False,
    ):
        super().__init__(cfg, dataset_name, is_training)
        self.formal_evaluation = bool(formal_evaluation)
        if int(getattr(cfg.feature, 'max_note_shift', 0)) != 0:
            raise ValueError(
                'Online note shifting is disabled for choral datasets because waveform '
                'pitch shifting is not yet paired with an identical transformation of '
                'note.pkl targets. Use a pre-generated, synchronously transposed dataset.'
            )
        self.note_dir = _get_choral_note_dir(cfg, dataset_name)
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
            if self.formal_evaluation:
                require_complete_satb_reference(
                    self.note_cache[stem],
                    note_path,
                    begin_note=int(self.cfg.feature.begin_note),
                    classes_num=int(self.cfg.feature.classes_num),
                )
        return self.note_cache[stem]

    def _build_union_rolls(self, stem: str, start_time: float):
        return build_canonical_union_rolls(
            self._load_note_bars(stem),
            start_time=start_time,
            segment_seconds=float(self.cfg.feature.segment_seconds),
            frames_per_second=float(self.cfg.feature.frames_per_second),
            begin_note=int(self.cfg.feature.begin_note),
            classes_num=int(self.cfg.feature.classes_num),
        )

    def __getitem__(self, meta):
        data_dict, actual_start_time = self._load_base_item(meta)
        stem = self._get_stem(meta)
        data_dict.update(self._build_union_rolls(stem, actual_start_time))
        return data_dict


class ChoralSATBDataset(ChoralUnionDataset):
    """Canonical choral union targets plus explicit SATB voice targets."""

    def __init__(
        self,
        cfg,
        dataset_name: str,
        is_training: bool = True,
        formal_evaluation: bool = False,
    ):
        super().__init__(
            cfg,
            dataset_name,
            is_training=is_training,
            formal_evaluation=formal_evaluation,
        )
        evaluation_assignment = None
        if not is_training:
            evaluation_assignment = str(
                getattr(cfg.choral, 'evaluation_reference_assignment', 'part_name')
            ).strip().lower().replace('-', '_')
            if evaluation_assignment != 'part_name':
                raise ValueError(
                    'Validation/test targets require '
                    'choral.evaluation_reference_assignment=part_name'
                )
        self.choral_target_builder = ChoralTargetBuilder(
            cfg,
            segment_seconds=cfg.feature.segment_seconds,
            target_assignment=evaluation_assignment,
        )

    def _build_voice_rolls(
        self,
        stem: str,
        start_time: float,
        frame_mask_roll=None,
        onset_mask_roll=None,
        offset_mask_roll=None,
    ):
        return self.choral_target_builder.build(
            self._load_note_bars(stem),
            start_time=start_time,
            frame_mask_roll=frame_mask_roll,
            onset_mask_roll=onset_mask_roll,
            offset_mask_roll=offset_mask_roll,
        )

    def __getitem__(self, meta):
        data_dict, actual_start_time = self._load_base_item(meta)
        stem = self._get_stem(meta)

        # Replace the merged-MIDI global targets before deriving voice masks.
        # Complete note.pkl intervals make all three target masks observable,
        # including notes crossing a segment boundary.
        data_dict.update(self._build_union_rolls(stem, actual_start_time))

        voice_target_dict = self._build_voice_rolls(
            stem,
            actual_start_time,
            frame_mask_roll=data_dict['frame_mask_roll'],
            onset_mask_roll=data_dict['onset_mask_roll'],
            offset_mask_roll=data_dict['offset_mask_roll'],
        )

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
    STATE_VERSION = 2

    def __init__(self, cfg, split, is_eval=None):
        assert split in ['train', 'validation', 'test']
        self.cfg = cfg
        self.split = split
        self.batch_size = int(cfg.exp.batch_size)
        if self.batch_size <= 0:
            raise ValueError(f'exp.batch_size must be positive, got {self.batch_size}')
        self.segment_seconds = cfg.feature.segment_seconds
        self.hop_seconds = cfg.feature.hop_seconds
        self.random_seed = int(cfg.exp.random_seed)
        self.random_state = np.random.RandomState(self.random_seed)
        self.mini_data = cfg.exp.mini_data
        # For evaluation loaders, `is_eval` tells us which dataset we are actually
        # iterating over (e.g. validate on youchorale while training on
        # youchorale_pro). Falling back to train_set here breaks cross-dataset
        # validation because the sampler and dataset point at different HDF5 roots.
        self.dataset_type = is_eval if is_eval is not None else cfg.dataset.train_set
        self.hdf5s_dir = get_dataset_hdf5s_dir(cfg, self.dataset_type)
        self.choral_note_dir = None
        if self.dataset_type in {
            'youchorale',
            'youchorale_pro',
            'csd',
            'cantoria',
        }:
            self.choral_note_dir = _get_choral_note_dir(cfg, self.dataset_type)

        _, hdf5_paths = traverse_folder(self.hdf5s_dir)
        self.split_manifest_identity = configured_split_manifest_identity(
            cfg,
            self.dataset_type,
        )
        if self.split_manifest_identity is not None:
            hdf5_paths = select_split_hdf5_paths(
                cfg,
                self.dataset_type,
                hdf5_paths,
                split,
            )
        self.segment_list = []
        self.hdf5_content_manifest = []
        file_counter = 0

        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as hf:
                if (
                    self.split_manifest_identity is None
                    and decode_hdf5_attr(hf.attrs['split']) != split
                ):
                    continue
                audio_name = os.path.basename(hdf5_path)
                if self.dataset_type == 'maestro':
                    file_id = [decode_hdf5_attr(hf.attrs['year']), audio_name]
                else:
                    file_id = [audio_name]

                # Waveform samples are the extraction boundary used by the
                # dataset loader, so they are also authoritative for segment
                # coverage. Metadata duration can differ after resampling.
                duration = hf['waveform'].shape[0] / float(cfg.feature.sample_rate)
                for start_time in segment_start_times(
                    duration,
                    self.segment_seconds,
                    self.hop_seconds,
                ):
                    self.segment_list.append(file_id + [start_time])

            relative_path = os.path.relpath(hdf5_path, self.hdf5s_dir).replace(
                os.sep,
                '/',
            )
            manifest_entry = {
                'path': relative_path,
                'sha256': _file_sha256(hdf5_path),
            }
            if self.choral_note_dir is not None:
                stem = os.path.splitext(audio_name)[0]
                note_path = os.path.join(self.choral_note_dir, f'{stem}.pkl')
                if not os.path.isfile(note_path):
                    raise FileNotFoundError(
                        f'Missing SATB note annotation for {stem}: {note_path}. '
                        'Cannot establish the sampler data identity.'
                    )
                manifest_entry['satb_reference'] = {
                    'path': f'{stem}.pkl',
                    'sha256': _file_sha256(note_path),
                }
            self.hdf5_content_manifest.append(manifest_entry)

            file_counter += 1
            if self.mini_data and file_counter >= 10:
                break

        logging.info('%s %s segments: %d', 'eval' if is_eval else 'train', split, len(self.segment_list))
        if not self.segment_list:
            raise RuntimeError(
                f'No segments found for dataset={self.dataset_type}, split={split}, hdf5s_dir={self.hdf5s_dir}'
            )
        self.segment_identity = self._compute_segment_identity()
        self.pointer = 0
        self.next_batch_index = 0
        self.segment_indexes = np.arange(len(self.segment_list))
        self.random_state.shuffle(self.segment_indexes)

    def _compute_segment_identity(self):
        encoded_segments = json.dumps(
            {
                'segments': self.segment_list,
                'hdf5_content_manifest': self.hdf5_content_manifest,
                'split_manifest_identity': getattr(
                    self,
                    'split_manifest_identity',
                    None,
                ),
            },
            ensure_ascii=True,
            allow_nan=False,
            separators=(',', ':'),
        ).encode('utf-8')
        return hashlib.sha256(encoded_segments).hexdigest()

    def _make_state_dict(
        self,
        *,
        pointer,
        segment_indexes,
        random_state,
        next_batch_index,
    ):
        return {
            'sampler_state_version': self.STATE_VERSION,
            'dataset_type': self.dataset_type,
            'split': self.split,
            'batch_size': self.batch_size,
            'random_seed': self.random_seed,
            'segment_identity': self.segment_identity,
            'hdf5_content_manifest': [
                dict(item) for item in self.hdf5_content_manifest
            ],
            'pointer': int(pointer),
            'segment_indexes': [
                int(index) for index in np.asarray(segment_indexes, dtype=np.int64)
            ],
            'random_state': serialize_numpy_random_state(random_state),
            'next_batch_index': int(next_batch_index),
        }

    def state_dict_for_batch(self, batch_index):
        """Return exact state before a zero-based logical batch, without mutation.

        This reconstructs from the configured seed instead of copying the live
        pointer, which may already be ahead of the optimizer because a
        multi-worker DataLoader prefetches batches.
        """

        if isinstance(batch_index, bool) or not isinstance(batch_index, (int, np.integer)):
            raise TypeError(f'batch_index must be an integer, got {type(batch_index).__name__}')
        batch_index = int(batch_index)
        if batch_index < 0:
            raise ValueError(f'batch_index must be non-negative, got {batch_index}')

        random_state = np.random.RandomState(self.random_seed)
        segment_indexes = np.arange(len(self.segment_list))
        random_state.shuffle(segment_indexes)
        pointer = 0
        remaining_examples = batch_index * self.batch_size

        while remaining_examples > 0:
            examples_until_shuffle = len(segment_indexes) - pointer
            consumed = min(remaining_examples, examples_until_shuffle)
            pointer += consumed
            remaining_examples -= consumed
            if pointer == len(segment_indexes):
                pointer = 0
                random_state.shuffle(segment_indexes)

        return self._make_state_dict(
            pointer=pointer,
            segment_indexes=segment_indexes,
            random_state=random_state,
            next_batch_index=batch_index,
        )

    def seek_batch(self, batch_index):
        """Move this sampler to the state before ``batch_index``."""

        self.load_state_dict(self.state_dict_for_batch(batch_index))

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
            self.next_batch_index += 1
            yield batch_segment_list

    def __len__(self):
        if self.batch_size == 0:
            return 0
        return int(np.ceil(len(self.segment_list) / self.batch_size))

    def state_dict(self):
        return self._make_state_dict(
            pointer=self.pointer,
            segment_indexes=self.segment_indexes,
            random_state=self.random_state,
            next_batch_index=self.next_batch_index,
        )

    def load_state_dict(self, state):
        required_keys = {
            'sampler_state_version',
            'dataset_type',
            'split',
            'batch_size',
            'random_seed',
            'segment_identity',
            'hdf5_content_manifest',
            'pointer',
            'segment_indexes',
            'random_state',
            'next_batch_index',
        }
        if not isinstance(state, dict):
            raise TypeError(f'sampler state must be a dict, got {type(state).__name__}')
        missing_keys = sorted(required_keys.difference(state))
        if missing_keys:
            raise ValueError(
                'Sampler state is missing exact-resume metadata '
                f'{missing_keys}. Refusing an inexact legacy restore; reconstruct '
                'explicitly with seek_batch(batch_index) instead.'
            )
        state_version = state['sampler_state_version']
        if isinstance(state_version, bool) or not isinstance(
            state_version,
            (int, np.integer),
        ):
            raise ValueError(
                f'sampler_state_version must be an integer, got {state_version!r}'
            )
        state_version = int(state_version)
        if state_version not in {1, self.STATE_VERSION}:
            raise ValueError(
                'Unsupported sampler_state_version '
                f'{state["sampler_state_version"]!r}; expected 1 or {self.STATE_VERSION}'
            )

        if state_version == self.STATE_VERSION:
            if not isinstance(state['segment_indexes'], list):
                raise ValueError(
                    'Sampler state v2 segment_indexes must be a primitive list'
                )
            if not isinstance(state['random_state'], Mapping):
                raise ValueError(
                    'Sampler state v2 random_state must be a versioned mapping'
                )

        expected_metadata = {
            'dataset_type': self.dataset_type,
            'split': self.split,
            'batch_size': self.batch_size,
            'random_seed': self.random_seed,
            'segment_identity': self.segment_identity,
            'hdf5_content_manifest': self.hdf5_content_manifest,
        }
        mismatches = {
            key: (expected, state[key])
            for key, expected in expected_metadata.items()
            if state[key] != expected
        }
        if mismatches:
            mismatch_text = ', '.join(
                f'{key}: current={current!r}, checkpoint={saved!r}'
                for key, (current, saved) in sorted(mismatches.items())
            )
            raise ValueError(f'Sampler state does not match the current segments/config: {mismatch_text}')

        indexes = np.asarray(state['segment_indexes'])
        if not np.issubdtype(indexes.dtype, np.integer):
            raise ValueError('segment_indexes must contain integers')
        indexes = indexes.astype(np.int64, copy=True)
        expected_indexes = np.arange(len(self.segment_list), dtype=np.int64)
        if indexes.shape != expected_indexes.shape or not np.array_equal(
            np.sort(indexes),
            expected_indexes,
        ):
            raise ValueError('segment_indexes must be a permutation of the current segment list')

        pointer_value = state['pointer']
        if isinstance(pointer_value, bool) or not isinstance(
            pointer_value,
            (int, np.integer),
        ):
            raise ValueError(f'pointer must be an integer, got {pointer_value!r}')
        pointer = int(pointer_value)
        if pointer < 0 or pointer >= len(indexes):
            raise ValueError(
                f'pointer must be in [0, {len(indexes) - 1}], got {pointer}'
            )
        next_batch_value = state['next_batch_index']
        if isinstance(next_batch_value, bool) or not isinstance(
            next_batch_value,
            (int, np.integer),
        ):
            raise ValueError(
                f'next_batch_index must be an integer, got {next_batch_value!r}'
            )
        next_batch_index = int(next_batch_value)
        if next_batch_index < 0:
            raise ValueError(
                f'next_batch_index must be non-negative, got {next_batch_index}'
            )
        expected_pointer = (next_batch_index * self.batch_size) % len(indexes)
        if pointer != expected_pointer:
            raise ValueError(
                'Sampler pointer is inconsistent with next_batch_index: '
                f'pointer={pointer}, expected={expected_pointer}'
            )

        restored_random_state = np.random.RandomState()
        try:
            restored_random_state.set_state(
                deserialize_numpy_random_state(state['random_state'])
            )
        except (IndexError, OverflowError, TypeError, ValueError) as exc:
            raise ValueError('Invalid sampler random_state') from exc

        self.pointer = pointer
        self.segment_indexes = indexes
        self.random_state = restored_random_state
        self.next_batch_index = next_batch_index


class EvalSampler(Sampler):
    def __init__(self, cfg, split, is_eval=None):
        super().__init__(cfg, split, is_eval=is_eval)
        # Evaluation must be deterministic and cover each segment at most once.
        self.segment_indexes = np.arange(len(self.segment_list))
        limit_key = 'max_train_eval_batches' if split == 'train' else 'max_eval_batches'
        configured_limit = getattr(cfg.exp, limit_key, None)
        self.max_evaluate_iteration = (
            None if configured_limit is None else int(configured_limit)
        )
        if self.max_evaluate_iteration is not None and self.max_evaluate_iteration <= 0:
            raise ValueError('exp.max_eval_batches must be null or a positive integer')

    def __iter__(self):
        indexes = self.segment_indexes
        if self.max_evaluate_iteration is not None:
            max_examples = self.max_evaluate_iteration * self.batch_size
            if len(indexes) > max_examples:
                # Even coverage avoids selecting only the alphabetically first
                # few recordings when a lightweight validation proxy is used.
                positions = np.linspace(0, len(indexes) - 1, max_examples, dtype=np.int64)
                indexes = indexes[positions]
        for start in range(0, len(indexes), self.batch_size):
            batch_indexes = indexes[start : start + self.batch_size]
            yield [self.segment_list[index] for index in batch_indexes]

    def __len__(self):
        batches = int(np.ceil(len(self.segment_indexes) / self.batch_size))
        if self.max_evaluate_iteration is None:
            return batches
        return min(batches, self.max_evaluate_iteration)



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
