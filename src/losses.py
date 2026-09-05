# SPDX-License-Identifier: Apache-2.0
# Includes modifications to ByteDance's piano_transcription implementation.

from __future__ import annotations

import torch

from utilities import get_task_spec, resolve_model_mode



def _safe_mask(mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(reference)
    return mask.to(reference.dtype)



def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0



def bce(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    eps = 1e-7
    output = torch.clamp(output, eps, 1.0 - eps)
    mask = _safe_mask(mask, output)
    matrix = -target * torch.log(output) - (1.0 - target) * torch.log(1.0 - output)
    denom = torch.clamp(torch.sum(mask), min=1.0)
    return torch.sum(matrix * mask) / denom



def task_bce(model, output_dict, target_dict):
    spec = get_task_spec(model.cfg)
    losses = []

    frame_mask = target_dict.get('frame_mask_roll')
    if 'frame_output' in output_dict and (frame_mask is None or torch.sum(frame_mask) > 0):
        losses.append(bce(output_dict['frame_output'], target_dict['frame_roll'], frame_mask))
    elif 'frame_output' in output_dict:
        losses.append(_zero_loss(output_dict['frame_output']))

    onset_mask = target_dict.get('onset_mask_roll')
    if 'onset_output' in output_dict and (onset_mask is None or torch.sum(onset_mask) > 0):
        losses.append(bce(output_dict['onset_output'], target_dict['onset_roll'], onset_mask))
    elif 'onset_output' in output_dict:
        losses.append(_zero_loss(output_dict['onset_output']))

    offset_mask = target_dict.get('offset_mask_roll')
    if 'offset_output' in output_dict and (offset_mask is None or torch.sum(offset_mask) > 0):
        losses.append(bce(output_dict['offset_output'], target_dict['offset_roll'], offset_mask))
    elif 'offset_output' in output_dict:
        losses.append(_zero_loss(output_dict['offset_output']))

    if spec.pedal and 'pedal_onset_output' in output_dict:
        pedal_mask = target_dict['pedal_mask_roll'][:, :, None]
        losses.append(bce(output_dict['pedal_onset_output'], target_dict['pedal_onset_roll'][:, :, None], pedal_mask))
        losses.append(bce(output_dict['pedal_offset_output'], target_dict['pedal_offset_roll'][:, :, None], pedal_mask))
        losses.append(bce(output_dict['pedal_frame_output'], target_dict['pedal_frame_roll'][:, :, None], pedal_mask))

    if not losses:
        reference = next(iter(output_dict.values()))
        return _zero_loss(reference)
    return sum(losses)


def choral_bce(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    eps = 1e-7
    output = torch.clamp(output, eps, 1.0 - eps)
    if mask is None:
        mask = torch.ones_like(output)
    else:
        mask = mask.to(output.dtype)
    matrix = -target * torch.log(output) - (1.0 - target) * torch.log(1.0 - output)
    denom = torch.clamp(torch.sum(mask), min=1.0)
    return torch.sum(matrix * mask) / denom


def choral_task_bce(model, output_dict, target_dict):
    cfg = model.cfg
    spec = get_task_spec(cfg)
    losses = []

    if 'voice_frame_output' in output_dict:
        losses.append(
            float(getattr(cfg.choral, 'voice_frame_loss_weight', 1.0))
            * choral_bce(output_dict['voice_frame_output'], target_dict['voice_frame_roll'], target_dict.get('voice_frame_mask_roll'))
        )
    if 'voice_onset_output' in output_dict:
        losses.append(
            float(getattr(cfg.choral, 'voice_onset_loss_weight', 1.0))
            * choral_bce(output_dict['voice_onset_output'], target_dict['voice_onset_roll'], target_dict.get('voice_onset_mask_roll'))
        )
    if spec.offset and 'voice_offset_output' in output_dict and 'voice_offset_roll' in target_dict:
        losses.append(
            float(getattr(cfg.choral, 'voice_offset_loss_weight', 1.0))
            * choral_bce(output_dict['voice_offset_output'], target_dict['voice_offset_roll'], target_dict.get('voice_offset_mask_roll'))
        )

    if 'frame_output' in output_dict:
        losses.append(
            float(getattr(cfg.choral, 'union_frame_loss_weight', 0.5))
            * bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict.get('frame_mask_roll'))
        )
    if 'onset_output' in output_dict:
        losses.append(
            float(getattr(cfg.choral, 'union_onset_loss_weight', 0.5))
            * bce(output_dict['onset_output'], target_dict['onset_roll'], target_dict.get('onset_mask_roll'))
        )
    if spec.offset and 'offset_output' in output_dict and 'offset_roll' in target_dict:
        losses.append(
            float(getattr(cfg.choral, 'union_offset_loss_weight', 0.5))
            * bce(output_dict['offset_output'], target_dict['offset_roll'], target_dict.get('offset_mask_roll'))
        )

    if 'voice_presence_logits' in output_dict and 'voice_presence' in target_dict:
        losses.append(
            float(getattr(cfg.choral, 'presence_loss_weight', 0.2))
            * torch.nn.functional.binary_cross_entropy_with_logits(
                output_dict['voice_presence_logits'],
                target_dict['voice_presence'],
            )
        )

    if 'voice_assignment_output' in output_dict:
        assignment = output_dict['voice_assignment_output']
        assignment_eps = 1e-7
        entropy = -(assignment.clamp(min=assignment_eps) * torch.log(assignment.clamp(min=assignment_eps))).sum(dim=2).mean()
        entropy_weight = float(getattr(cfg.choral, 'assignment_entropy_loss_weight', 0.0))
        if entropy_weight > 0.0:
            losses.append(entropy_weight * entropy)

        range_weight = float(getattr(cfg.choral, 'assignment_range_loss_weight', 0.0))
        if range_weight > 0.0:
            begin_note = int(cfg.feature.begin_note)
            classes_num = int(cfg.feature.classes_num)
            margin = float(getattr(cfg.choral, 'voice_assignment_range_margin', 0.0))
            midi_notes = torch.arange(
                begin_note,
                begin_note + classes_num,
                device=assignment.device,
                dtype=assignment.dtype,
            )
            voice_mins = torch.tensor(
                list(getattr(cfg.choral, 'voice_assignment_range_mins', [60, 55, 48, 40])),
                device=assignment.device,
                dtype=assignment.dtype,
            )[:, None]
            voice_maxs = torch.tensor(
                list(getattr(cfg.choral, 'voice_assignment_range_maxs', [88, 79, 72, 67])),
                device=assignment.device,
                dtype=assignment.dtype,
            )[:, None]
            in_range = (
                (midi_notes[None, :] >= (voice_mins - margin))
                & (midi_notes[None, :] <= (voice_maxs + margin))
            ).to(assignment.dtype)
            out_of_range = 1.0 - in_range
            range_penalty = (assignment * out_of_range[None, None, :, :]).mean()
            losses.append(range_weight * range_penalty)

    if not losses:
        reference = next(iter(output_dict.values()))
        return _zero_loss(reference)
    return sum(losses)


LOSS_FUNC_DICT = {
    'task_bce': task_bce,
    'choral_task_bce': choral_task_bce,
    'frame_onset_bce': task_bce,
    'frame_onset_offset_bce': task_bce,
}



def resolve_loss_type(cfg) -> str:
    loss_type = getattr(cfg.exp, 'loss_type', 'auto')
    if loss_type != 'auto':
        return loss_type
    if getattr(cfg.choral, 'enable', False):
        return 'choral_task_bce'
    mode = resolve_model_mode(cfg)
    mapping = {
        'frame_onset': 'frame_onset_bce',
        'frame_onset_offset': 'frame_onset_offset_bce',
    }
    return mapping[mode]



def get_loss_func(loss_type):
    if loss_type not in LOSS_FUNC_DICT:
        raise ValueError(f'Incorrect loss_type: {loss_type}')
    return LOSS_FUNC_DICT[loss_type]
