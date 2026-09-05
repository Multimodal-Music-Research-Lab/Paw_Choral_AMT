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
            valid = frame_mask.flatten() > 0
            if np.any(valid):
                statistics['frame_ap'] = metrics.average_precision_score(
                    output_dict['frame_roll'].flatten()[valid],
                    output_dict['frame_output'].flatten()[valid],
                    average='macro',
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

        for key in list(statistics.keys()):
            statistics[key] = np.around(statistics[key], decimals=4)
        return statistics
