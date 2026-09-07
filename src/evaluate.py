# SPDX-License-Identifier: Apache-2.0
# Includes modifications to ByteDance's piano_transcription implementation.

from __future__ import annotations

import numpy as np
from sklearn import metrics

from utilities import forward_dataloader, get_task_spec



def _mae(target, output, mask=None):
    if mask is None:
        return float(np.mean(np.abs(target - output)))
    denom = np.clip(np.sum(mask), 1e-8, np.inf)
    return float(np.sum(np.abs(target - output) * mask) / denom)


def _masked_average_precision(target, output, mask=None):
    if mask is None:
        mask = np.ones_like(output)
    valid = np.asarray(mask).flatten() > 0
    if not np.any(valid):
        return None
    target = np.asarray(target).flatten()[valid]
    output = np.asarray(output).flatten()[valid]
    if not np.any(target > 0):
        return None
    return float(metrics.average_precision_score(target, output))


class SegmentEvaluator(object):
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.spec = get_task_spec(cfg)

    def evaluate(self, dataloader):
        statistics = {}
        output_dict = forward_dataloader(self.model, dataloader, return_target=True)

        if 'frame_output' in output_dict:
            frame_mask = output_dict.get('frame_mask_roll', np.ones_like(output_dict['frame_output']))
            frame_ap = _masked_average_precision(
                output_dict['frame_roll'], output_dict['frame_output'], frame_mask
            )
            statistics['frame_ap'] = float('nan') if frame_ap is None else frame_ap

        if 'voice_frame_output' in output_dict and 'voice_frame_roll' in output_dict:
            voice_mask = output_dict.get(
                'voice_frame_mask_roll',
                np.ones_like(output_dict['voice_frame_output']),
            )
            voice_aps = []
            for voice_idx, voice_name in enumerate(getattr(self.cfg.choral, 'voice_names', ['S', 'A', 'T', 'B'])):
                voice_ap = _masked_average_precision(
                    output_dict['voice_frame_roll'][:, :, voice_idx, :],
                    output_dict['voice_frame_output'][:, :, voice_idx, :],
                    voice_mask[:, :, voice_idx, :],
                )
                statistics[f'{voice_name}_frame_ap'] = (
                    float('nan') if voice_ap is None else voice_ap
                )
                if voice_ap is not None:
                    voice_aps.append(voice_ap)
            expected_voice_count = output_dict['voice_frame_output'].shape[2]
            statistics['mean_voice_frame_ap'] = (
                float(np.mean(voice_aps))
                if len(voice_aps) == expected_voice_count
                else float('nan')
            )

        if 'onset_output' in output_dict:
            onset_mask = output_dict.get('onset_mask_roll', np.ones_like(output_dict['onset_output']))
            if np.sum(onset_mask) > 0:
                statistics['onset_mae'] = _mae(output_dict['onset_roll'], output_dict['onset_output'], onset_mask)

        if 'offset_output' in output_dict:
            offset_mask = output_dict.get('offset_mask_roll', np.ones_like(output_dict['offset_output']))
            if np.sum(offset_mask) > 0:
                statistics['offset_mae'] = _mae(output_dict['offset_roll'], output_dict['offset_output'], offset_mask)

        if 'pedal_onset_output' in output_dict:
            pedal_mask = output_dict.get('pedal_mask_roll')
            if pedal_mask is None or np.sum(pedal_mask) > 0:
                statistics['pedal_onset_mae'] = _mae(
                    output_dict['pedal_onset_roll'].flatten(),
                    output_dict['pedal_onset_output'].flatten(),
                    None if pedal_mask is None else pedal_mask.flatten(),
                )

        if 'pedal_offset_output' in output_dict:
            pedal_mask = output_dict.get('pedal_mask_roll')
            if pedal_mask is None or np.sum(pedal_mask) > 0:
                statistics['pedal_offset_mae'] = _mae(
                    output_dict['pedal_offset_roll'].flatten(),
                    output_dict['pedal_offset_output'].flatten(),
                    None if pedal_mask is None else pedal_mask.flatten(),
                )

        if 'pedal_frame_output' in output_dict:
            pedal_mask = output_dict.get('pedal_mask_roll')
            if pedal_mask is None or np.sum(pedal_mask) > 0:
                statistics['pedal_frame_mae'] = _mae(
                    output_dict['pedal_frame_roll'].flatten(),
                    output_dict['pedal_frame_output'].flatten(),
                    None if pedal_mask is None else pedal_mask.flatten(),
                )

        return statistics
