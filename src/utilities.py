# SPDX-License-Identifier: Apache-2.0
# Includes modifications to ByteDance's piano_transcription implementation.

import os
import logging
import librosa
import audioread
import numpy as np
import csv
import datetime
import collections
import pickle
from dataclasses import dataclass
from typing import Dict

import soundfile as sf
import torch

from choral_targets import resolve_target_assignment

try:
    from mido import MidiFile, merge_tracks
except Exception:
    MidiFile = None
    merge_tracks = None

from piano_vad import (
    note_detection_with_onset_offset_regress,
    pedal_detection_with_onset_offset_regress,
    onsets_frames_note_detection,
    onsets_frames_pedal_detection,
)


MODE_SPECS = {
    'frame_onset': {
        'frame': True,
        'onset': True,
        'offset': False,
        'pedal': False,
    },
    'frame_onset_offset': {
        'frame': True,
        'onset': True,
        'offset': True,
        'pedal': False,
    },
}

LEGACY_MODE_MAP = {
    'note': 'frame_onset_offset',
}

UNSUPPORTED_LEGACY_MODEL_TYPES = {'note_velo', 'note_velo_pedal'}


@dataclass(frozen=True)
class TaskSpec:
    arch: str
    mode: str
    frame: bool
    onset: bool
    offset: bool
    pedal: bool



def resolve_model_arch(cfg) -> str:
    arch = str(getattr(cfg.model, 'arch', 'pagct')).lower()
    if arch not in {'pagct', 'pawct', 'hpt', 'onf'}:
        raise ValueError(f'Unsupported model.arch: {arch}')
    return arch



def resolve_model_mode(cfg) -> str:
    mode = getattr(cfg.model, 'mode', None)
    if mode is None:
        legacy_type = getattr(cfg.model, 'type', None)
        if legacy_type in UNSUPPORTED_LEGACY_MODEL_TYPES:
            raise ValueError(f'Legacy model.type={legacy_type} is no longer supported after velocity cleanup')
        mode = LEGACY_MODE_MAP.get(legacy_type, 'frame_onset_offset')
    mode = str(mode)
    if mode not in MODE_SPECS:
        raise ValueError(f'Unsupported model.mode: {mode}')
    return mode



def get_task_spec(cfg) -> TaskSpec:
    arch = resolve_model_arch(cfg)
    mode = resolve_model_mode(cfg)
    spec = MODE_SPECS[mode]
    if arch == 'onf' and mode not in {'frame_onset', 'frame_onset_offset'}:
        raise ValueError(f"model.arch='onf' only supports frame_onset and frame_onset_offset, got {mode}")
    return TaskSpec(
        arch=arch,
        mode=mode,
        frame=spec['frame'],
        onset=spec['onset'],
        offset=spec['offset'],
        pedal=spec['pedal'],
    )



def resolve_post_processor_type(cfg) -> str:
    post_type = str(getattr(cfg.post, 'post_processor_type', 'onsets_frames')).lower()
    if post_type == 'auto':
        return 'onsets_frames'
    if post_type in {'onsets_frames', 'onf', 'onset_frame', 'onset_frames'}:
        return 'onsets_frames'
    if post_type in {'regression', 'regressive'}:
        return 'regression'
    raise ValueError(f'Unsupported post.post_processor_type: {post_type}')


def build_target_masks(cfg, target_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    spec = get_task_spec(cfg)
    base_mask = target_dict['mask_roll'].astype(np.float32)
    pedal_frame_roll = target_dict['pedal_frame_roll'].astype(np.float32)

    target_dict['frame_mask_roll'] = base_mask
    target_dict['onset_mask_roll'] = base_mask if spec.onset else np.zeros_like(base_mask, dtype=np.float32)
    target_dict['offset_mask_roll'] = base_mask if spec.offset else np.zeros_like(base_mask, dtype=np.float32)
    target_dict['pedal_mask_roll'] = np.ones_like(pedal_frame_roll, dtype=np.float32) if spec.pedal else np.zeros_like(pedal_frame_roll, dtype=np.float32)
    return target_dict



def move_data_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if 'float' in str(x.dtype):
        x = torch.tensor(x, dtype=torch.float32)
    elif 'int' in str(x.dtype):
        x = torch.tensor(x, dtype=torch.long)
    else:
        return x
    return x.to(device)



def append_to_dict(storage, key, value):
    if key not in storage:
        storage[key] = []
    storage[key].append(value)



def forward_dataloader(model, dataloader, return_target=True):
    output_dict = {}
    device = next(model.parameters()).device
    model.eval()

    for batch_data_dict in dataloader:
        batch_waveform = move_data_to_device(batch_data_dict['waveform'], device)
        with torch.no_grad():
            batch_output_dict = model(batch_waveform)

        for key, value in batch_output_dict.items():
            if '_list' not in key:
                append_to_dict(output_dict, key, value.detach().cpu().numpy())

        if return_target:
            for key, value in batch_data_dict.items():
                if 'roll' in key or 'mask' in key or 'reg_distance' in key or 'reg_tail' in key:
                    append_to_dict(output_dict, key, value)

    for key in output_dict:
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    return output_dict



def forward(model, x, batch_size):
    output_dict = {}
    device = next(model.parameters()).device
    model.eval()
    pointer = 0

    while pointer < len(x):
        batch_waveform = move_data_to_device(x[pointer:pointer + batch_size], device)
        pointer += batch_size
        with torch.no_grad():
            batch_output_dict = model(batch_waveform)
        for key, value in batch_output_dict.items():
            append_to_dict(output_dict, key, value.detach().cpu().numpy())

    for key in output_dict:
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    return output_dict



def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def traverse_folder(folder):
    paths = []
    names = []

    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for name in sorted(files):
            filepath = os.path.join(root, name)
            names.append(name)
            paths.append(filepath)

    return names, paths


def note_to_freq(piano_note):
    return 2 ** ((piano_note - 39) / 12) * 440


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1

    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging


def float32_to_int16(x):
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    peak = np.max(np.abs(x))
    if peak > 1.0:
        x = x / peak
    return (x * 32767.).astype(np.int16)


def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)


def decode_hdf5_attr(value):
    if isinstance(value, bytes):
        return value.decode()
    return value


def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]


def read_metadata(csv_path):
    """Read metadata of MAESTRO dataset from csv file.

    Args:
      csv_path: str

    Returns:
      meta_dict, dict, e.g. {
        'canonical_composer': ['Alban Berg', ...],
        'canonical_title': ['Sonata Op. 1', ...],
        'split': ['train', ...],
        'year': ['2018', ...]
        'midi_filename': ['2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi', ...],
        'audio_filename': ['2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav', ...],
        'duration': [698.66116031, ...]}
    """

    with open(csv_path, 'r') as fr:
        reader = csv.reader(fr, delimiter=',')
        lines = list(reader)

    meta_dict = {'canonical_composer': [], 'canonical_title': [], 'split': [],
        'year': [], 'midi_filename': [], 'audio_filename': [], 'duration': []}

    for n in range(1, len(lines)):
        meta_dict['canonical_composer'].append(lines[n][0])
        meta_dict['canonical_title'].append(lines[n][1])
        meta_dict['split'].append(lines[n][2])
        meta_dict['year'].append(lines[n][3])
        meta_dict['midi_filename'].append(lines[n][4])
        meta_dict['audio_filename'].append(lines[n][5])
        meta_dict['duration'].append(float(lines[n][6]))

    for key in meta_dict.keys():
        meta_dict[key] = np.array(meta_dict[key])

    return meta_dict


def read_midi(midi_path, dataset='maestro'):
    """
    Parse a MIDI file and return events with timestamps.

    Args:
        midi_path (str): Path to the MIDI file.
        dataset (str): One of 'maestro', 'smd', 'maps', or 'youchorale'. Determines where tempo and events are stored.

            - 'maestro': 2 tracks.
              • Track 0 holds all meta messages (set_tempo, time_signature, end_of_track).
              • Track 1 holds piano events.

            - 'smd': 2 tracks.
              • Track 0 holds meta messages, but tempo is the second index (track_name, set_tempo, time_signature, end_of_track).
              • Track 1 holds piano events.

            - 'maps': 1 track.
              • That track holds both meta messages and piano events (tempo is the first message).

    Returns:
        dict: {
            'midi_event': np.ndarray of message strings,
            'midi_event_time': np.ndarray of timestamps in seconds
        }
    """
    if MidiFile is None:
        raise ImportError('mido is required to read MIDI files but could not be imported.')
    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat

    ds = dataset.lower()
    if ds == 'youchorale':
        if merge_tracks is None:
            raise ImportError('mido.merge_tracks is required to read YouChorale MIDI files.')

        current_tempo = 500000
        ticks_accum = 0
        seconds_accum = 0.0
        message_list = []
        time_in_second = []

        # Merge all SATB tracks and follow tempo changes from track 0.
        for msg in merge_tracks(midi_file.tracks):
            delta_ticks = msg.time
            ticks_accum += delta_ticks
            seconds_accum += delta_ticks * current_tempo / 1e6 / ticks_per_beat

            if msg.type == 'set_tempo':
                current_tempo = msg.tempo
                continue

            if msg.type in {'note_on', 'note_off', 'control_change'}:
                message_list.append(str(msg))
                time_in_second.append(seconds_accum)

        return {
            'midi_event': np.array(message_list),
            'midi_event_time': np.array(time_in_second)
        }

    if ds == 'maestro':
        # Expect 2 tracks: track 0 for meta (tempo at index 0), track 1 for piano events
        assert len(midi_file.tracks) == 2, f"{dataset} format requires 2 tracks, found {len(midi_file.tracks)}"
        microseconds_per_beat = midi_file.tracks[0][0].tempo
        play_track_idx = 1

    elif ds == 'smd':
        # Expect 2 tracks: track 0 for meta (tempo at index 1), track 1 for piano events
        assert len(midi_file.tracks) == 2, f"SMD format requires 2 tracks, found {len(midi_file.tracks)}"
        microseconds_per_beat = midi_file.tracks[0][1].tempo
        play_track_idx = 1

    elif ds == 'maps':
        # Expect 1 track: contains both meta and piano events (tempo at index 0)
        assert len(midi_file.tracks) == 1, f"MAPS format requires 1 track, found {len(midi_file.tracks)}"
        microseconds_per_beat = midi_file.tracks[0][0].tempo
        play_track_idx = 0

    else:
        raise ValueError(f"Dataset not supported: {dataset}")

    # Convert ticks to seconds
    beats_per_second = 1e6 / microseconds_per_beat
    ticks_per_second = ticks_per_beat * beats_per_second

    message_list = []
    time_in_second = []
    ticks_accum = 0

    # Iterate over piano event track
    for msg in midi_file.tracks[play_track_idx]:
        message_list.append(str(msg))
        ticks_accum += msg.time
        time_in_second.append(ticks_accum / ticks_per_second)

    return {
        'midi_event': np.array(message_list),
        'midi_event_time': np.array(time_in_second)
    }


class TargetProcessor(object):
    def __init__(self, segment_seconds, cfg):
        """Class for processing MIDI events to target.

        Args:
          segment_seconds: float
          frames_per_second: int
          begin_note: int, A0 MIDI note of a piano
          classes_num: int
        """
        self.segment_seconds = segment_seconds
        self.frames_per_second = cfg.feature.frames_per_second
        self.begin_note = cfg.feature.begin_note
        self.classes_num = cfg.feature.classes_num
        self.max_piano_note = self.classes_num - 1

    def process(self, start_time, midi_events_time, midi_events,
        extend_pedal=True, note_shift=0):
        """Process MIDI events of an audio segment to target for training,
        includes:
        1. Parse MIDI events
        2. Prepare note targets
        3. Prepare pedal targets

        Args:
          start_time: float, start time of a segment
          midi_events_time: list of float, times of MIDI events of a recording,
            e.g. [0, 3.3, 5.1, ...]
          midi_events: list of str, MIDI events of a recording, e.g.
            ['note_on channel=0 note=75 velocity=37 time=14',
             'control_change channel=0 control=64 value=54 time=20',
             ...]
          extend_pedal, bool, True: Notes will be set to ON until pedal is
            released. False: Ignore pedal events.

        Returns:
          target_dict: {
            'onset_roll': (frames_num, classes_num),
            'offset_roll': (frames_num, classes_num),
            'reg_onset_roll': (frames_num, classes_num),
            'reg_offset_roll': (frames_num, classes_num),
            'frame_roll': (frames_num, classes_num),
            'velocity_roll': (frames_num, classes_num),
            'mask_roll':  (frames_num, classes_num),
            'pedal_onset_roll': (frames_num,),
            'pedal_offset_roll': (frames_num,),
            'reg_pedal_onset_roll': (frames_num,),
            'reg_pedal_offset_roll': (frames_num,),
            'pedal_frame_roll': (frames_num,)}

          note_events: list of dict, e.g. [
            {'midi_note': 51, 'onset_time': 696.64, 'offset_time': 697.00, 'velocity': 44},
            {'midi_note': 58, 'onset_time': 697.00, 'offset_time': 697.19, 'velocity': 50}
            ...]

          pedal_events: list of dict, e.g. [
            {'onset_time': 149.37, 'offset_time': 150.35},
            {'onset_time': 150.54, 'offset_time': 152.06},
            ...]
        """

        # ------ 1. Parse MIDI events ------
        # Search the half-open event-index bounds explicitly. ``searchsorted``
        # returns ``len(midi_events_time)`` when there is no later event, unlike
        # the former ``for`` loop which left ``fin_idx`` pointing at (and then
        # excluded) the final event.
        event_times = np.asarray(midi_events_time)
        bgn_idx = int(np.searchsorted(event_times, start_time, side='right'))
        """E.g., start_time: 709.0, bgn_idx: 18003, event_time: 709.0146"""

        # Search the end index of a segment
        fin_idx = int(np.searchsorted(
            event_times,
            start_time + self.segment_seconds,
            side='right',
        ))
        """E.g., start_time: 709.0, bgn_idx: 18196, event_time: 719.0115"""

        note_events = []
        """E.g. [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44},
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]"""

        pedal_events = []
        """E.g. [
            {'onset_time': 696.46875, 'offset_time': 696.62604},
            {'onset_time': 696.8063, 'offset_time': 698.50836},
            ...]"""

        buffer_dict = {}    # Used to store onset of notes to be paired with offsets
        pedal_dict = {}     # Used to store onset of pedal to be paired with offset of pedal

        # Backtrack bgn_idx to earlier indexes: ex_bgn_idx, which is used for
        # searching cross segment pedal and note events. E.g.: bgn_idx: 1149,
        # ex_bgn_idx: 981
        _delta = int((fin_idx - bgn_idx) * 1.)
        ex_bgn_idx = max(bgn_idx - _delta, 0)

        for i in range(ex_bgn_idx, fin_idx):
            # Parse MIDI messiage
            attribute_list = midi_events[i].split(' ')

            # Note
            if attribute_list[0] in ['note_on', 'note_off']:
                """E.g. attribute_list: ['note_on', 'channel=0', 'note=41', 'velocity=0', 'time=10']"""

                midi_note = int(attribute_list[2].split('=')[1])
                velocity = int(attribute_list[3].split('=')[1])

                # Onset
                if attribute_list[0] == 'note_on' and velocity > 0:
                    buffer_dict[midi_note] = {
                        'onset_time': midi_events_time[i],
                        'velocity': velocity}

                # Offset
                else:
                    if midi_note in buffer_dict.keys():
                        note_events.append({
                            'midi_note': midi_note,
                            'onset_time': buffer_dict[midi_note]['onset_time'],
                            'offset_time': midi_events_time[i],
                            'velocity': buffer_dict[midi_note]['velocity']})
                        del buffer_dict[midi_note]

            # Pedal
            elif attribute_list[0] == 'control_change' and attribute_list[2] == 'control=64':
                """control=64 corresponds to pedal MIDI event. E.g.
                attribute_list: ['control_change', 'channel=0', 'control=64', 'value=45', 'time=43']"""

                ped_value = int(attribute_list[3].split('=')[1])
                if ped_value >= 64:
                    if 'onset_time' not in pedal_dict:
                        pedal_dict['onset_time'] = midi_events_time[i]
                else:
                    if 'onset_time' in pedal_dict:
                        pedal_events.append({
                            'onset_time': pedal_dict['onset_time'],
                            'offset_time': midi_events_time[i]})
                        pedal_dict = {}

        # Add unpaired onsets to events
        for midi_note in buffer_dict.keys():
            note_events.append({
                'midi_note': midi_note,
                'onset_time': buffer_dict[midi_note]['onset_time'],
                'offset_time': start_time + self.segment_seconds,
                'velocity': buffer_dict[midi_note]['velocity']})

        # Add unpaired pedal onsets to data
        if 'onset_time' in pedal_dict.keys():
            pedal_events.append({
                'onset_time': pedal_dict['onset_time'],
                'offset_time': start_time + self.segment_seconds})

        # Set notes to ON until pedal is released
        if extend_pedal:
            note_events = self.extend_pedal(note_events, pedal_events)

        # Prepare targets
        frames_num = int(round(self.segment_seconds * self.frames_per_second)) + 1
        onset_roll = np.zeros((frames_num, self.classes_num))
        offset_roll = np.zeros((frames_num, self.classes_num))
        reg_onset_roll = np.ones((frames_num, self.classes_num))
        reg_offset_roll = np.ones((frames_num, self.classes_num))
        frame_roll = np.zeros((frames_num, self.classes_num))
        velocity_roll = np.zeros((frames_num, self.classes_num))
        mask_roll = np.ones((frames_num, self.classes_num))
        """mask_roll is used for masking out cross segment notes"""

        pedal_onset_roll = np.zeros(frames_num)
        pedal_offset_roll = np.zeros(frames_num)
        reg_pedal_onset_roll = np.ones(frames_num)
        reg_pedal_offset_roll = np.ones(frames_num)
        pedal_frame_roll = np.zeros(frames_num)

        # ------ 2. Get note targets ------
        # Process note events to target
        for note_event in note_events:
            """note_event: e.g., {'midi_note': 60, 'onset_time': 722.0719, 'offset_time': 722.47815, 'velocity': 103}"""

            piano_note = np.clip(note_event['midi_note'] - self.begin_note + note_shift, 0, self.max_piano_note)
            """There are 88 keys on a piano"""

            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((note_event['onset_time'] - start_time) * self.frames_per_second))
                fin_frame = int(round((note_event['offset_time'] - start_time) * self.frames_per_second))

                if fin_frame >= 0:
                    frame_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = 1

                    offset_roll[fin_frame, piano_note] = 1
                    velocity_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = note_event['velocity']

                    # Vector from the center of a frame to ground truth offset
                    reg_offset_roll[fin_frame, piano_note] = \
                        (note_event['offset_time'] - start_time) - (fin_frame / self.frames_per_second)

                    if bgn_frame >= 0:
                        onset_roll[bgn_frame, piano_note] = 1

                        # Vector from the center of a frame to ground truth onset
                        reg_onset_roll[bgn_frame, piano_note] = \
                            (note_event['onset_time'] - start_time) - (bgn_frame / self.frames_per_second)

                    # Mask out segment notes
                    else:
                        mask_roll[: fin_frame + 1, piano_note] = 0

        for k in range(self.classes_num):
            """Get regression targets"""
            reg_onset_roll[:, k] = self.get_regression(reg_onset_roll[:, k])
            reg_offset_roll[:, k] = self.get_regression(reg_offset_roll[:, k])

        # Process unpaired onsets to target
        for midi_note in buffer_dict.keys():
            piano_note = np.clip(midi_note - self.begin_note + note_shift, 0, self.max_piano_note)
            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((buffer_dict[midi_note]['onset_time'] - start_time) * self.frames_per_second))
                mask_roll[bgn_frame :, piano_note] = 0

        # ------ 3. Get pedal targets ------
        # Process pedal events to target
        for pedal_event in pedal_events:
            bgn_frame = int(round((pedal_event['onset_time'] - start_time) * self.frames_per_second))
            fin_frame = int(round((pedal_event['offset_time'] - start_time) * self.frames_per_second))

            if fin_frame >= 0:
                pedal_frame_roll[max(bgn_frame, 0) : fin_frame + 1] = 1

                pedal_offset_roll[fin_frame] = 1
                reg_pedal_offset_roll[fin_frame] = \
                    (pedal_event['offset_time'] - start_time) - (fin_frame / self.frames_per_second)

                if bgn_frame >= 0:
                    pedal_onset_roll[bgn_frame] = 1
                    reg_pedal_onset_roll[bgn_frame] = \
                        (pedal_event['onset_time'] - start_time) - (bgn_frame / self.frames_per_second)

        # Get regresssion padal targets
        reg_pedal_onset_roll = self.get_regression(reg_pedal_onset_roll)
        reg_pedal_offset_roll = self.get_regression(reg_pedal_offset_roll)

        target_dict = {
            'onset_roll': onset_roll, 'offset_roll': offset_roll,
            'reg_onset_roll': reg_onset_roll, 'reg_offset_roll': reg_offset_roll,
            'frame_roll': frame_roll, 'velocity_roll': velocity_roll,
            'mask_roll': mask_roll, 'reg_pedal_onset_roll': reg_pedal_onset_roll,
            'pedal_onset_roll': pedal_onset_roll, 'pedal_offset_roll': pedal_offset_roll,
            'reg_pedal_offset_roll': reg_pedal_offset_roll, 'pedal_frame_roll': pedal_frame_roll
            }

        return target_dict, note_events, pedal_events

    def extend_pedal(self, note_events, pedal_events):
        """Update the offset of all notes until pedal is released.

        Args:
          note_events: list of dict, e.g., [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44},
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
          pedal_events: list of dict, e.g., [
            {'onset_time': 696.46875, 'offset_time': 696.62604},
            {'onset_time': 696.8063, 'offset_time': 698.50836},
            ...]

        Returns:
          ex_note_events: list of dict, e.g., [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44},
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
        """
        note_events = collections.deque(note_events)
        pedal_events = collections.deque(pedal_events)
        ex_note_events = []

        idx = 0     # Index of note events
        while pedal_events: # Go through all pedal events
            pedal_event = pedal_events.popleft()
            buffer_dict = {}    # keys: midi notes, value for each key: event index

            while note_events:
                note_event = note_events.popleft()

                # If a note offset is between the onset and offset of a pedal,
                # Then set the note offset to when the pedal is released.
                if pedal_event['onset_time'] < note_event['offset_time'] < pedal_event['offset_time']:

                    midi_note = note_event['midi_note']

                    if midi_note in buffer_dict.keys():
                        """Multiple same note inside a pedal"""
                        _idx = buffer_dict[midi_note]
                        del buffer_dict[midi_note]
                        ex_note_events[_idx]['offset_time'] = note_event['onset_time']

                    # Set note offset to pedal offset
                    note_event['offset_time'] = pedal_event['offset_time']
                    buffer_dict[midi_note] = idx

                ex_note_events.append(note_event)
                idx += 1

                # Break loop and pop next pedal
                if note_event['offset_time'] > pedal_event['offset_time']:
                    break

        while note_events:
            """Append left notes"""
            ex_note_events.append(note_events.popleft())

        return ex_note_events

    def get_regression(self, input):
        """Get regression target. See Fig. 2 of [1] for an example.
        [1] Q. Kong, et al., High-resolution Piano Transcription with Pedals by
        Regressing Onsets and Offsets Times, 2020.

        input:
          input: (frames_num,)

        Returns: (frames_num,), e.g., [0, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.9, 0.7, 0.5, 0.3, 0.1, 0, 0, ...]
        """
        step = 1. / self.frames_per_second
        output = np.ones_like(input)

        locts = np.where(input < 0.5)[0]
        if len(locts) > 0:
            for t in range(0, locts[0]):
                output[t] = step * (t - locts[0]) - input[locts[0]]

            for i in range(0, len(locts) - 1):
                for t in range(locts[i], (locts[i] + locts[i + 1]) // 2):
                    output[t] = step * (t - locts[i]) - input[locts[i]]

                for t in range((locts[i] + locts[i + 1]) // 2, locts[i + 1]):
                    output[t] = step * (t - locts[i + 1]) - input[locts[i + 1]]

            for t in range(locts[-1], len(input)):
                output[t] = step * (t - locts[-1]) - input[locts[-1]]

        output = np.clip(np.abs(output), 0., 0.05) * 20
        output = (1. - output)

        return output


def write_events_to_midi(start_time, note_events, pedal_events, midi_path):
    """Write out note events to MIDI file.

    Args:
      start_time: float
      note_events: list of dict, e.g. [
        {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44},
        {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
        ...]
      midi_path: str
    """
    try:
        from mido import Message, MidiFile, MidiTrack, MetaMessage
    except Exception as exc:
        raise ImportError('mido is required to write MIDI files but could not be imported.') from exc

    # This configuration is the same as MIDIs in MAESTRO dataset
    ticks_per_beat = 384
    beats_per_second = 2
    ticks_per_second = ticks_per_beat * beats_per_second
    microseconds_per_beat = int(1e6 // beats_per_second)

    midi_file = MidiFile()
    midi_file.ticks_per_beat = ticks_per_beat

    # Track 0
    track0 = MidiTrack()
    track0.append(MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))
    track0.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    track0.append(MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track0)

    # Track 1
    track1 = MidiTrack()

    # Message rolls of MIDI
    message_roll = []

    for note_event in note_events:
        # Onset
        message_roll.append({
            'time': note_event['onset_time'],
            'midi_note': note_event['midi_note'],
            'velocity': note_event['velocity']})

        # Offset
        message_roll.append({
            'time': note_event['offset_time'],
            'midi_note': note_event['midi_note'],
            'velocity': 0})

    if pedal_events:
        for pedal_event in pedal_events:
            message_roll.append({'time': pedal_event['onset_time'], 'control_change': 64, 'value': 127})
            message_roll.append({'time': pedal_event['offset_time'], 'control_change': 64, 'value': 0})

    # Sort MIDI messages by time
    message_roll.sort(key=lambda note_event: note_event['time'])

    previous_ticks = 0
    for message in message_roll:
        this_ticks = int((message['time'] - start_time) * ticks_per_second)
        if this_ticks >= 0:
            diff_ticks = this_ticks - previous_ticks
            previous_ticks = this_ticks
            if 'midi_note' in message.keys():
                track1.append(Message('note_on', note=message['midi_note'], velocity=message['velocity'], time=diff_ticks))
            elif 'control_change' in message.keys():
                track1.append(Message('control_change', channel=0, control=message['control_change'], value=message['value'], time=diff_ticks))
    track1.append(MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track1)

    midi_file.save(midi_path)


def plot_waveform_midi_targets(data_dict, start_time, note_events, cfg):
    """Debug helper for one segment."""
    import matplotlib.pyplot as plt

    create_folder('debug')
    audio_path = 'debug/debug.wav'
    midi_path = 'debug/debug.mid'
    fig_path = 'debug/debug.png'

    sf.write(audio_path, data_dict['waveform'], cfg.feature.sample_rate)
    write_events_to_midi(start_time, note_events, None, midi_path)
    x = librosa.core.stft(y=data_dict['waveform'], n_fft=2048, hop_length=int(cfg.feature.sample_rate / cfg.feature.frames_per_second), window='hann', center=True)
    x = np.abs(x) ** 2

    fig, axs = plt.subplots(8, 1, sharex=True, figsize=(24, 20))
    axs[0].matshow(np.log(np.maximum(x, 1e-8)), origin='lower', aspect='auto', cmap='jet')
    axs[1].matshow(data_dict['onset_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[2].matshow(data_dict['frame_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[3].matshow(data_dict['reg_onset_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[4].matshow(data_dict['reg_offset_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[5].matshow(data_dict['velocity_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[6].matshow(data_dict['frame_mask_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[7].matshow(data_dict['velocity_mask_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[0].set_title('Log spectrogram')
    axs[1].set_title('onset_roll')
    axs[2].set_title('frame_roll')
    axs[3].set_title('reg_onset_roll')
    axs[4].set_title('reg_offset_roll')
    axs[5].set_title('velocity_roll')
    axs[6].set_title('frame_mask_roll')
    axs[7].set_title('velocity_mask_roll')
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)
    print(f'Write out to {audio_path}, {midi_path}, {fig_path}!')


class RegressionPostProcessor(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.frames_per_second = cfg.feature.frames_per_second
        self.classes_num = cfg.feature.classes_num
        self.begin_note = cfg.feature.begin_note
        self.velocity_scale = cfg.feature.velocity_scale

        self.frame_threshold = cfg.post.frame_threshold
        self.onset_threshold = cfg.post.onset_threshold
        self.offset_threshold = cfg.post.offset_threshold
        self.pedal_offset_threshold = cfg.post.pedal_offset_threshold

    def output_dict_to_midi_events(self, output_dict):
        est_on_off_note_vels, est_pedal_on_offs = self.output_dict_to_note_pedal_arrays(output_dict)
        est_note_events = self.detected_notes_to_events(est_on_off_note_vels)
        est_pedal_events = None if est_pedal_on_offs is None else self.detected_pedals_to_events(est_pedal_on_offs)
        return est_note_events, est_pedal_events

    def output_dict_to_note_pedal_arrays(self, output_dict):
        onset_output, onset_shift_output = self.get_binarized_output_from_regression(
            reg_output=output_dict['onset_output'],
            threshold=self.onset_threshold,
            neighbour=2,
        )
        output_dict['onset_output'] = onset_output
        output_dict['onset_shift_output'] = onset_shift_output

        if 'offset_output' in output_dict:
            offset_output, offset_shift_output = self.get_binarized_output_from_regression(
                reg_output=output_dict['offset_output'],
                threshold=self.offset_threshold,
                neighbour=4,
            )
            output_dict['offset_output'] = offset_output
            output_dict['offset_shift_output'] = offset_shift_output
        else:
            output_dict['offset_output'] = np.zeros_like(output_dict['frame_output'])
            output_dict['offset_shift_output'] = np.zeros_like(output_dict['frame_output'])

        if 'pedal_offset_output' in output_dict:
            pedal_offset_output, pedal_offset_shift_output = self.get_binarized_output_from_regression(
                reg_output=output_dict['pedal_offset_output'],
                threshold=self.pedal_offset_threshold,
                neighbour=4,
            )
            output_dict['pedal_offset_output'] = pedal_offset_output
            output_dict['pedal_offset_shift_output'] = pedal_offset_shift_output
        else:
            output_dict['pedal_offset_shift_output'] = np.zeros_like(output_dict['pedal_frame_output']) if 'pedal_frame_output' in output_dict else None

        est_on_off_note_vels = self.output_dict_to_detected_notes(output_dict)
        est_pedal_on_offs = self.output_dict_to_detected_pedals(output_dict) if 'pedal_frame_output' in output_dict else None
        return est_on_off_note_vels, est_pedal_on_offs

    def get_binarized_output_from_regression(self, reg_output, threshold, neighbour):
        binary_output = np.zeros_like(reg_output)
        shift_output = np.zeros_like(reg_output)
        frames_num, classes_num = reg_output.shape

        for k in range(classes_num):
            x = reg_output[:, k]
            if frames_num == 0:
                continue
            if frames_num == 1:
                if x[0] > threshold:
                    binary_output[0, k] = 1
                continue

            # The regular neighbourhood test below cannot visit boundary
            # frames. Treat a boundary as a peak only when the available side
            # moves monotonically away from it; sub-frame shift is unknown at
            # a one-sided boundary and remains zero.
            left_width = min(neighbour, frames_num - 1)
            left_steps = x[:left_width] - x[1:left_width + 1]
            if x[0] > threshold and np.all(left_steps >= 0) and np.any(left_steps > 0):
                binary_output[0, k] = 1

            for n in range(neighbour, frames_num - neighbour):
                if x[n] > threshold and self.is_monotonic_neighbour(x, n, neighbour):
                    binary_output[n, k] = 1
                    if x[n - 1] > x[n + 1]:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2
                    else:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2
                    shift_output[n, k] = shift

            right_width = min(neighbour, frames_num - 1)
            right_steps = x[-right_width:] - x[-right_width - 1:-1]
            if x[-1] > threshold and np.all(right_steps >= 0) and np.any(right_steps > 0):
                binary_output[-1, k] = 1

        return binary_output, shift_output

    def is_monotonic_neighbour(self, x, n, neighbour):
        monotonic = True
        for i in range(neighbour):
            if x[n - i] < x[n - i - 1]:
                monotonic = False
            if x[n + i] < x[n + i + 1]:
                monotonic = False
        return monotonic

    def output_dict_to_detected_notes(self, output_dict):
        est_tuples = []
        est_midi_notes = []
        classes_num = output_dict['frame_output'].shape[-1]
        if 'velocity_output' in output_dict:
            velocity_output = output_dict['velocity_output']
        else:
            default_velocity = float(self.cfg.post.default_velocity) / float(self.velocity_scale)
            velocity_output = np.full_like(output_dict['frame_output'], default_velocity, dtype=np.float32)

        if 'offset_output' not in output_dict:
            output_dict['offset_output'] = np.zeros_like(output_dict['frame_output'], dtype=np.float32)
        if 'offset_shift_output' not in output_dict:
            output_dict['offset_shift_output'] = np.zeros_like(output_dict['frame_output'], dtype=np.float32)

        for piano_note in range(classes_num):
            est_tuples_per_note = note_detection_with_onset_offset_regress(
                frame_output=output_dict['frame_output'][:, piano_note],
                onset_output=output_dict['onset_output'][:, piano_note],
                onset_shift_output=output_dict['onset_shift_output'][:, piano_note],
                offset_output=output_dict['offset_output'][:, piano_note],
                offset_shift_output=output_dict['offset_shift_output'][:, piano_note],
                velocity_output=velocity_output[:, piano_note],
                frame_threshold=self.frame_threshold,
            )
            est_tuples += est_tuples_per_note
            est_midi_notes += [piano_note + self.begin_note] * len(est_tuples_per_note)

        if len(est_tuples) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        est_tuples = np.array(est_tuples, dtype=np.float32)
        est_midi_notes = np.array(est_midi_notes, dtype=np.float32)
        onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
        offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
        velocities = est_tuples[:, 4]
        return np.stack((onset_times, offset_times, est_midi_notes, velocities), axis=-1).astype(np.float32)

    def output_dict_to_detected_pedals(self, output_dict):
        if 'pedal_offset_output' not in output_dict:
            output_dict['pedal_offset_output'] = np.zeros_like(output_dict['pedal_frame_output'], dtype=np.float32)
        if 'pedal_offset_shift_output' not in output_dict:
            output_dict['pedal_offset_shift_output'] = np.zeros_like(output_dict['pedal_frame_output'], dtype=np.float32)

        est_tuples = pedal_detection_with_onset_offset_regress(
            frame_output=output_dict['pedal_frame_output'][:, 0],
            offset_output=output_dict['pedal_offset_output'][:, 0],
            offset_shift_output=output_dict['pedal_offset_shift_output'][:, 0],
            frame_threshold=0.5,
        )
        est_tuples = np.array(est_tuples)

        if len(est_tuples) == 0:
            return np.array([])

        onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
        offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
        return np.stack((onset_times, offset_times), axis=-1).astype(np.float32)

    def detected_notes_to_events(self, est_on_off_note_vels):
        midi_events = []
        for i in range(est_on_off_note_vels.shape[0]):
            midi_events.append({
                'onset_time': est_on_off_note_vels[i][0],
                'offset_time': est_on_off_note_vels[i][1],
                'midi_note': int(est_on_off_note_vels[i][2]),
                'velocity': int(est_on_off_note_vels[i][3] * self.velocity_scale),
            })
        return midi_events

    def detected_pedals_to_events(self, pedal_on_offs):
        pedal_events = []
        for i in range(len(pedal_on_offs)):
            pedal_events.append({
                'onset_time': pedal_on_offs[i, 0],
                'offset_time': pedal_on_offs[i, 1],
            })
        return pedal_events


class OnsetsFramesPostProcessor(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.frames_per_second = cfg.feature.frames_per_second
        self.classes_num = cfg.feature.classes_num
        self.begin_note = cfg.feature.begin_note
        self.velocity_scale = cfg.feature.velocity_scale

        self.frame_threshold = cfg.post.frame_threshold
        self.onset_threshold = cfg.post.onset_threshold
        self.offset_threshold = cfg.post.offset_threshold
        self.pedal_offset_threshold = cfg.post.pedal_offset_threshold

    def output_dict_to_midi_events(self, output_dict):
        est_on_off_note_vels, est_pedal_on_offs = self.output_dict_to_note_pedal_arrays(output_dict)
        est_note_events = self.detected_notes_to_events(est_on_off_note_vels)
        est_pedal_events = None if est_pedal_on_offs is None else self.detected_pedals_to_events(est_pedal_on_offs)
        return est_note_events, est_pedal_events

    def output_dict_to_note_pedal_arrays(self, output_dict):
        output_dict = self.sharp_output_dict(output_dict)
        est_on_off_note_vels = self.output_dict_to_detected_notes(output_dict)
        est_pedal_on_offs = self.output_dict_to_detected_pedals(output_dict) if 'pedal_frame_output' in output_dict else None
        return est_on_off_note_vels, est_pedal_on_offs

    def sharp_output_dict(self, output_dict):
        if 'onset_output' in output_dict:
            output_dict['onset_output'] = self.sharp_output(output_dict['onset_output'], self.onset_threshold)
        if 'offset_output' in output_dict:
            output_dict['offset_output'] = self.sharp_output(output_dict['offset_output'], self.offset_threshold)
        if 'pedal_offset_output' in output_dict:
            output_dict['pedal_offset_output'] = self.sharp_output(output_dict['pedal_offset_output'], self.pedal_offset_threshold)
        return output_dict

    def sharp_output(self, x, threshold):
        frames_num, _classes_num = x.shape
        y = np.zeros_like(x)
        if frames_num == 0:
            return y
        if frames_num == 1:
            y[0] = x[0] > threshold
            return y

        y[0] = (x[0] > threshold) & (x[0] > x[1])
        y[-1] = (x[-1] > threshold) & (x[-1] > x[-2])
        if frames_num > 2:
            y[1:-1] = (
                (x[1:-1] > threshold)
                & (x[1:-1] > x[:-2])
                & (x[1:-1] > x[2:])
            )
        return y

    def output_dict_to_detected_notes(self, output_dict):
        est_tuples = []
        est_midi_notes = []
        if 'velocity_output' in output_dict:
            velocity_output = output_dict['velocity_output']
        else:
            default_velocity = float(self.cfg.post.default_velocity) / float(self.velocity_scale)
            velocity_output = np.full_like(output_dict['frame_output'], default_velocity, dtype=np.float32)

        if 'offset_output' not in output_dict:
            output_dict['offset_output'] = np.zeros_like(output_dict['frame_output'], dtype=np.float32)

        for piano_note in range(self.classes_num):
            est_tuples_per_note = onsets_frames_note_detection(
                frame_output=output_dict['frame_output'][:, piano_note],
                onset_output=output_dict['onset_output'][:, piano_note],
                offset_output=output_dict['offset_output'][:, piano_note],
                velocity_output=velocity_output[:, piano_note],
                threshold=self.frame_threshold,
            )
            est_tuples += est_tuples_per_note
            est_midi_notes += [piano_note + self.begin_note] * len(est_tuples_per_note)

        if len(est_midi_notes) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        est_tuples = np.array(est_tuples, dtype=np.float32)
        est_midi_notes = np.array(est_midi_notes, dtype=np.float32)
        onset_times = est_tuples[:, 0] / self.frames_per_second
        offset_times = est_tuples[:, 1] / self.frames_per_second
        velocities = est_tuples[:, 2]
        return np.stack((onset_times, offset_times, est_midi_notes, velocities), axis=-1).astype(np.float32)

    def output_dict_to_detected_pedals(self, output_dict):
        if 'pedal_offset_output' not in output_dict:
            output_dict['pedal_offset_output'] = np.zeros_like(output_dict['pedal_frame_output'], dtype=np.float32)

        est_tuples = onsets_frames_pedal_detection(
            frame_output=output_dict['pedal_frame_output'][:, 0],
            offset_output=output_dict['pedal_offset_output'][:, 0],
            frame_threshold=0.5,
        )
        est_tuples = np.array(est_tuples)

        if len(est_tuples) == 0:
            return np.array([])

        onset_times = est_tuples[:, 0] / self.frames_per_second
        offset_times = est_tuples[:, 1] / self.frames_per_second
        return np.stack((onset_times, offset_times), axis=-1).astype(np.float32)

    def detected_notes_to_events(self, est_on_off_note_vels):
        midi_events = []
        for i in range(len(est_on_off_note_vels)):
            midi_events.append({
                'onset_time': est_on_off_note_vels[i][0],
                'offset_time': est_on_off_note_vels[i][1],
                'midi_note': int(est_on_off_note_vels[i][2]),
                'velocity': int(est_on_off_note_vels[i][3] * self.velocity_scale),
            })
        return midi_events

    def detected_pedals_to_events(self, pedal_on_offs):
        pedal_events = []
        for i in range(len(pedal_on_offs)):
            pedal_events.append({
                'onset_time': pedal_on_offs[i, 0],
                'offset_time': pedal_on_offs[i, 1],
            })
        return pedal_events


# ============================
# Project helpers
# ============================

def get_dataset_hdf5s_dir(cfg, dataset_name):
    return os.path.join(cfg.exp.workspace, 'hdf5s', f'{dataset_name}_sr{int(cfg.feature.sample_rate)}')


def get_model_name(cfg):
    model_name = str(getattr(cfg.model, 'name', 'auto'))
    if model_name and model_name != 'auto':
        return model_name
    arch = resolve_model_arch(cfg)
    mode = resolve_model_mode(cfg)
    model_name = f"{arch}_{mode}_{cfg.feature.audio_feature}_sr{int(cfg.feature.sample_rate)}_fps{int(cfg.feature.frames_per_second)}"
    name_suffix = str(getattr(cfg.exp, 'name_suffix', '')).strip()
    suffixes = []
    if name_suffix:
        suffixes.append(name_suffix)
    if getattr(cfg.choral, 'enable', False) and getattr(cfg.choral, 'append_assignment_to_name', True):
        configured_assignment = getattr(cfg.choral, 'target_assignment', None)
        legacy_assignment = getattr(cfg.choral, 'voice_assignment_method', None)
        uses_legacy_assignment_key = (
            configured_assignment is None or not str(configured_assignment).strip()
        ) and legacy_assignment is not None and str(legacy_assignment).strip()
        assignment_method = resolve_target_assignment(cfg)
        if uses_legacy_assignment_key and str(legacy_assignment).strip() != 'part_name':
            # Preserve historical checkpoint-directory names exactly when an
            # old resolved config/CLI uses the deprecated key.
            suffixes.append(f'va_{str(legacy_assignment).strip()}')
        elif assignment_method and assignment_method != 'part_name':
            assignment_tags = {
                'range_prior': 'rp',
                'ordered_continuity': 'oc',
                'range_masked_continuity': 'rmc',
                'legacy_range_prior': 'legacy_rp',
                'legacy_ordered_continuity': 'legacy_oc',
                'legacy_range_masked_continuity': 'legacy_rmc',
            }
            suffixes.append(assignment_tags.get(assignment_method, f'target_{assignment_method}'))
        assignment_module = str(getattr(cfg.choral, 'assignment_module', 'heads')).strip()
        if assignment_module and assignment_module != 'heads':
            suffixes.append(f'vamod_{assignment_module}')
        voice_interaction_module = str(getattr(cfg.choral, 'voice_interaction_module', 'none')).strip()
        if voice_interaction_module and voice_interaction_module != 'none':
            suffixes.append(f'vint_{voice_interaction_module}')
    if suffixes:
        model_name = f"{model_name}_{'_'.join(suffixes)}"
    return model_name
