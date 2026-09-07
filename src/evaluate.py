# SPDX-License-Identifier: Apache-2.0
# Includes modifications to ByteDance's piano_transcription implementation.

from __future__ import annotations

import gc
import logging
import os
import tempfile

import numpy as np
import torch
from sklearn import metrics

from utilities import forward_dataloader, get_task_spec, move_data_to_device



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


class _DiskBackedAveragePrecision:
    """Accumulate exact AP inputs without retaining a full split in RAM."""

    def __init__(self, directory, name):
        self.score_path = os.path.join(directory, f"{name}.scores.f32")
        self.target_path = os.path.join(directory, f"{name}.targets.u8")
        self.score_file = open(self.score_path, "wb")
        self.target_file = open(self.target_path, "wb")
        self.count = 0
        self.positive_count = 0
        self.closed = False

    def update(self, target, output, mask=None):
        target = np.asarray(target)
        output = np.asarray(output)
        if target.size != output.size:
            raise ValueError(
                f"AP target/output size mismatch: {target.size} != {output.size}"
            )
        target = target.reshape(-1)
        output = output.reshape(-1)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.size != output.size:
                raise ValueError(
                    f"AP mask size mismatch: {mask.size} != {output.size}"
                )
            valid = mask.reshape(-1) > 0
            target = target[valid]
            output = output[valid]
        binary_target = np.asarray(target > 0, dtype=np.uint8)
        scores = np.asarray(output, dtype=np.float32)
        binary_target.tofile(self.target_file)
        scores.tofile(self.score_file)
        self.count += int(scores.size)
        self.positive_count += int(np.count_nonzero(binary_target))

    def close(self):
        if self.closed:
            return
        self.score_file.close()
        self.target_file.close()
        self.closed = True

    def compute(self):
        self.close()
        if self.count == 0 or self.positive_count == 0:
            return None
        targets = np.memmap(
            self.target_path,
            mode="r",
            dtype=np.uint8,
            shape=(self.count,),
        )
        scores = np.memmap(
            self.score_path,
            mode="r",
            dtype=np.float32,
            shape=(self.count,),
        )
        try:
            return float(metrics.average_precision_score(targets, scores))
        finally:
            del scores
            del targets
            gc.collect()


class _StreamingMae:
    def __init__(self):
        self.absolute_error = 0.0
        self.weight = 0.0

    def update(self, target, output, mask=None):
        target = np.asarray(target)
        output = np.asarray(output)
        if target.size != output.size:
            raise ValueError(
                f"MAE target/output size mismatch: {target.size} != {output.size}"
            )
        target = target.reshape(-1)
        output = output.reshape(-1)
        error = np.abs(target - output)
        if mask is None:
            self.absolute_error += float(np.sum(error, dtype=np.float64))
            self.weight += float(error.size)
            return
        mask = np.asarray(mask)
        if mask.size != output.size:
            raise ValueError(
                f"MAE mask size mismatch: {mask.size} != {output.size}"
            )
        mask = mask.reshape(-1)
        self.absolute_error += float(np.sum(error * mask, dtype=np.float64))
        self.weight += float(np.sum(mask, dtype=np.float64))

    def compute(self):
        if self.weight <= 0:
            return None
        return self.absolute_error / self.weight


class SegmentEvaluator(object):
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.spec = get_task_spec(cfg)

    def evaluate(self, dataloader):
        # Keep the aggregation path available for small injected fixtures and
        # compatibility tests. Real loaders use the bounded-memory path below.
        if dataloader is None:
            return self._evaluate_aggregated(
                forward_dataloader(self.model, dataloader, return_target=True)
            )
        return self._evaluate_streaming(dataloader)

    def _evaluate_aggregated(self, output_dict):
        statistics = {}

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

    def _evaluate_streaming(self, dataloader):
        statistics = {}
        device = next(self.model.parameters()).device
        self.model.eval()
        workspace = os.path.abspath(str(self.cfg.exp.workspace))
        os.makedirs(workspace, exist_ok=True)
        voice_names = list(
            getattr(self.cfg.choral, "voice_names", ["S", "A", "T", "B"])
        )

        with tempfile.TemporaryDirectory(
            prefix=".segment-eval-",
            dir=workspace,
        ) as temporary_directory:
            frame_ap = None
            voice_aps = None
            onset_mae = _StreamingMae()
            offset_mae = _StreamingMae()
            pedal_onset_mae = _StreamingMae()
            pedal_offset_mae = _StreamingMae()
            pedal_frame_mae = _StreamingMae()
            saw_onset = False
            saw_offset = False
            saw_pedal_onset = False
            saw_pedal_offset = False
            saw_pedal_frame = False

            try:
                total_batches = len(dataloader)
            except TypeError:
                total_batches = None

            for batch_index, batch_data_dict in enumerate(dataloader, start=1):
                batch_waveform = move_data_to_device(
                    batch_data_dict["waveform"],
                    device,
                )
                with torch.no_grad():
                    batch_output_dict = self.model(batch_waveform)

                if "frame_output" in batch_output_dict:
                    if frame_ap is None:
                        frame_ap = _DiskBackedAveragePrecision(
                            temporary_directory,
                            "frame",
                        )
                    frame_ap.update(
                        batch_data_dict["frame_roll"],
                        batch_output_dict["frame_output"].detach().cpu().numpy(),
                        batch_data_dict.get("frame_mask_roll"),
                    )

                if (
                    "voice_frame_output" in batch_output_dict
                    and "voice_frame_roll" in batch_data_dict
                ):
                    voice_output = (
                        batch_output_dict["voice_frame_output"].detach().cpu().numpy()
                    )
                    voice_target = np.asarray(batch_data_dict["voice_frame_roll"])
                    voice_mask = batch_data_dict.get("voice_frame_mask_roll")
                    if voice_output.shape != voice_target.shape:
                        raise ValueError(
                            "Voice-frame target/output shape mismatch: "
                            f"{voice_target.shape} != {voice_output.shape}"
                        )
                    voice_count = voice_output.shape[2]
                    if voice_count != len(voice_names):
                        raise ValueError(
                            f"Model emitted {voice_count} voices but config names "
                            f"{len(voice_names)}"
                        )
                    if voice_aps is None:
                        voice_aps = [
                            _DiskBackedAveragePrecision(
                                temporary_directory,
                                f"voice_{voice_index}",
                            )
                            for voice_index in range(voice_count)
                        ]
                    for voice_index, accumulator in enumerate(voice_aps):
                        accumulator.update(
                            voice_target[:, :, voice_index, :],
                            voice_output[:, :, voice_index, :],
                            None
                            if voice_mask is None
                            else np.asarray(voice_mask)[:, :, voice_index, :],
                        )

                if "onset_output" in batch_output_dict:
                    saw_onset = True
                    onset_mae.update(
                        batch_data_dict["onset_roll"],
                        batch_output_dict["onset_output"].detach().cpu().numpy(),
                        batch_data_dict.get("onset_mask_roll"),
                    )
                if "offset_output" in batch_output_dict:
                    saw_offset = True
                    offset_mae.update(
                        batch_data_dict["offset_roll"],
                        batch_output_dict["offset_output"].detach().cpu().numpy(),
                        batch_data_dict.get("offset_mask_roll"),
                    )

                pedal_mask = batch_data_dict.get("pedal_mask_roll")
                if "pedal_onset_output" in batch_output_dict:
                    saw_pedal_onset = True
                    pedal_onset_mae.update(
                        batch_data_dict["pedal_onset_roll"],
                        batch_output_dict["pedal_onset_output"].detach().cpu().numpy(),
                        pedal_mask,
                    )
                if "pedal_offset_output" in batch_output_dict:
                    saw_pedal_offset = True
                    pedal_offset_mae.update(
                        batch_data_dict["pedal_offset_roll"],
                        batch_output_dict["pedal_offset_output"].detach().cpu().numpy(),
                        pedal_mask,
                    )
                if "pedal_frame_output" in batch_output_dict:
                    saw_pedal_frame = True
                    pedal_frame_mae.update(
                        batch_data_dict["pedal_frame_roll"],
                        batch_output_dict["pedal_frame_output"].detach().cpu().numpy(),
                        pedal_mask,
                    )

                if batch_index % 100 == 0 or batch_index == total_batches:
                    logging.info(
                        "Evaluation progress: %d%s batches",
                        batch_index,
                        "" if total_batches is None else f"/{total_batches}",
                    )

            if frame_ap is not None:
                value = frame_ap.compute()
                statistics["frame_ap"] = float("nan") if value is None else value

            if voice_aps is not None:
                finite_voice_aps = []
                for voice_name, accumulator in zip(voice_names, voice_aps):
                    value = accumulator.compute()
                    statistics[f"{voice_name}_frame_ap"] = (
                        float("nan") if value is None else value
                    )
                    if value is not None:
                        finite_voice_aps.append(value)
                statistics["mean_voice_frame_ap"] = (
                    float(np.mean(finite_voice_aps))
                    if len(finite_voice_aps) == len(voice_aps)
                    else float("nan")
                )

            for key, saw_value, accumulator in (
                ("onset_mae", saw_onset, onset_mae),
                ("offset_mae", saw_offset, offset_mae),
                ("pedal_onset_mae", saw_pedal_onset, pedal_onset_mae),
                ("pedal_offset_mae", saw_pedal_offset, pedal_offset_mae),
                ("pedal_frame_mae", saw_pedal_frame, pedal_frame_mae),
            ):
                value = accumulator.compute()
                if saw_value and value is not None:
                    statistics[key] = value

            for accumulator in [frame_ap, *(voice_aps or [])]:
                if accumulator is not None:
                    accumulator.close()

        return statistics
