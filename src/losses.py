# SPDX-License-Identifier: Apache-2.0
# Includes modifications to ByteDance's piano_transcription implementation.

from __future__ import annotations

import math

import torch

from utilities import get_task_spec, resolve_model_mode



def _safe_mask(mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(reference)
    return mask.to(reference.dtype)



def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _stable_bce_matrix(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Elementwise BCE in a dtype where both clamp boundaries are representable."""

    compute_dtype = torch.float64 if output.dtype == torch.float64 else torch.float32
    probability = output.to(compute_dtype)
    target = target.to(compute_dtype)
    eps = torch.finfo(compute_dtype).eps
    probability = torch.clamp(probability, eps, 1.0 - eps)
    return -target * torch.log(probability) - (1.0 - target) * torch.log1p(-probability)



def bce(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    mask = _safe_mask(mask, output)
    matrix = _stable_bce_matrix(output, target)
    mask = mask.to(matrix.dtype)
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


def choral_bce(
    output: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    positive_weight: float = 1.0,
) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(output)
    else:
        mask = mask.to(output.dtype)
    matrix = _stable_bce_matrix(output, target)
    target = target.to(matrix.dtype)
    mask = mask.to(matrix.dtype)
    example_weight = 1.0 + (max(float(positive_weight), 0.0) - 1.0) * target
    matrix = matrix * example_weight
    denom = torch.clamp(torch.sum(mask * example_weight), min=1.0)
    return torch.sum(matrix * mask) / denom


def voice_range_prior_loss(
    output: torch.Tensor,
    cfg,
    target: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize predicted probability outside each voice's configured range.

    This operates on the PawCT voice heads themselves, unlike the historical
    assignment loss which was active only for the optional VA2 module.
    """
    begin_note = int(cfg.feature.begin_note)
    classes_num = int(cfg.feature.classes_num)
    margin = float(getattr(cfg.choral, 'voice_assignment_range_margin', 0.0))
    midi_notes = torch.arange(
        begin_note,
        begin_note + classes_num,
        device=output.device,
        dtype=output.dtype,
    )
    voice_mins = torch.as_tensor(
        list(getattr(cfg.choral, 'voice_assignment_range_mins', [60, 55, 48, 40])),
        device=output.device,
        dtype=output.dtype,
    )[:, None]
    voice_maxs = torch.as_tensor(
        list(getattr(cfg.choral, 'voice_assignment_range_maxs', [88, 79, 72, 67])),
        device=output.device,
        dtype=output.dtype,
    )[:, None]
    out_of_range = (
        (midi_notes[None, :] < (voice_mins - margin))
        | (midi_notes[None, :] > (voice_maxs + margin))
    ).to(output.dtype)
    penalty_mask = out_of_range[None, None, :, :].expand_as(output)
    if target is not None:
        # A range prior must never overwrite trusted supervision: unusual but
        # annotated notes remain valid positives. The prior only suppresses
        # unsupported out-of-range probability mass.
        penalty_mask = penalty_mask * (1.0 - target.to(output.dtype))
    if mask is not None:
        penalty_mask = penalty_mask * mask.to(output.dtype)
    denom = torch.clamp(penalty_mask.sum(), min=1.0)
    return torch.sum(output * penalty_mask) / denom


def voice_continuity_prior_loss(
    output: torch.Tensor,
    target_events: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    activity_target: torch.Tensor | None = None,
    activity_mask: torch.Tensor | None = None,
    frames_per_second: float = 100.0,
    gap_decay_seconds: float = 2.0,
) -> torch.Tensor:
    """Match predicted and annotated pitch motion between consecutive events.

    The former OC regularizer always pushed the predicted interval toward zero.
    That also penalized a perfectly predicted legitimate leap and could flatten
    melodic motion.  A frame-adjacent residual still gave almost no useful
    signal: hundreds of within-note hold pairs diluted a single transition, and
    a one-frame rest removed that transition entirely.  This objective instead
    links consecutive annotated onset events in each voice.  It is zero when
    the prediction follows the annotated interval and penalizes unsupported
    jumps or voice swaps across rests.  When a frame-activity target is
    supplied, silent gaps are measured from the last active frame before the
    next onset. Long rests are down-weighted rather than treated as equally
    reliable continuity evidence.

    A pitch-selective mask could move a probability centre merely by changing
    its support.  Therefore an event is eligible only when every pitch bin at
    that voice/frame is supervised.
    """
    if output.shape != target_events.shape:
        raise ValueError(
            "output and target_events must have identical [B, T, V, P] shapes, "
            f"got {tuple(output.shape)} and {tuple(target_events.shape)}"
        )
    if output.ndim != 4:
        raise ValueError(
            "continuity inputs must have shape [batch, time, voice, pitch], "
            f"got {tuple(output.shape)}"
        )
    if activity_target is not None and activity_target.shape != output.shape:
        raise ValueError(
            "activity_target must match the event tensor shape, got "
            f"{tuple(activity_target.shape)} and {tuple(output.shape)}"
        )
    if activity_mask is not None and activity_mask.shape != output.shape:
        raise ValueError(
            "activity_mask must match the event tensor shape, got "
            f"{tuple(activity_mask.shape)} and {tuple(output.shape)}"
        )
    frames_per_second = float(frames_per_second)
    gap_decay_seconds = float(gap_decay_seconds)
    if not math.isfinite(frames_per_second) or frames_per_second <= 0.0:
        raise ValueError("frames_per_second must be finite and positive")
    if not math.isfinite(gap_decay_seconds) or gap_decay_seconds < 0.0:
        raise ValueError("gap_decay_seconds must be finite and non-negative")
    if output.shape[1] < 2:
        return _zero_loss(output)

    effective_mask = _safe_mask(mask, output)
    masked_output = output * effective_mask
    masked_target = target_events.to(output.dtype) * effective_mask
    pitch_axis = torch.arange(output.shape[-1], device=output.device, dtype=output.dtype)
    output_mass = masked_output.sum(dim=-1).clamp(min=1e-6)
    target_mass = masked_target.sum(dim=-1).clamp(min=1e-6)
    output_center = (
        masked_output * pitch_axis[None, None, None, :]
    ).sum(dim=-1) / output_mass
    target_center = (
        masked_target * pitch_axis[None, None, None, :]
    ).sum(dim=-1) / target_mass

    fully_supervised = torch.all(effective_mask > 0, dim=-1)
    event_active = (target_events.to(output.dtype).sum(dim=-1) > 0) & fully_supervised
    weighted_loss = _zero_loss(output)
    pair_count = output.new_zeros((), dtype=torch.float32)

    if activity_target is not None:
        activity_effective_mask = _safe_mask(activity_mask, output)
        activity_fully_supervised = torch.all(
            activity_effective_mask > 0,
            dim=-1,
        )
        activity_active = (
            activity_target.to(output.dtype).sum(dim=-1) > 0
        ) & activity_fully_supervised
    else:
        activity_fully_supervised = fully_supervised
        activity_active = event_active

    time_indices = torch.arange(
        output.shape[1],
        device=output.device,
        dtype=torch.long,
    ).view(1, -1, 1).expand(
        output.shape[0],
        -1,
        output.shape[2],
    )

    # The shifted cumulative maximum identifies the previous onset for every
    # current event without a CUDA-synchronising nonzero/Python loop.
    event_locations = torch.where(
        event_active,
        time_indices,
        torch.full_like(time_indices, -1),
    )
    latest_event = torch.cummax(event_locations, dim=1).values
    previous_event = torch.cat((
        torch.full_like(latest_event[:, :1, :], -1),
        latest_event[:, :-1, :],
    ), dim=1)
    previous_event_safe = torch.clamp(previous_event, min=0)

    previous_output_center = torch.gather(
        output_center,
        dim=1,
        index=previous_event_safe,
    )
    previous_target_center = torch.gather(
        target_center,
        dim=1,
        index=previous_event_safe,
    )
    output_delta = (output_center - previous_output_center) / 12.0
    target_delta = (target_center - previous_target_center) / 12.0
    pair_loss = torch.nn.functional.smooth_l1_loss(
        output_delta,
        target_delta,
        reduction='none',
    )

    # Exclude a pair if any frame in [previous onset, current onset] lacks a
    # complete pitch mask. This prevents masked activity from masquerading as
    # a rest and prevents a changing pitch support from shifting the centre.
    invalid_prefix = torch.cat((
        torch.zeros(
            output.shape[0],
            1,
            output.shape[2],
            device=output.device,
            dtype=torch.long,
        ),
        torch.cumsum((~activity_fully_supervised).to(torch.long), dim=1),
    ), dim=1)
    invalid_before_previous = torch.gather(
        invalid_prefix,
        dim=1,
        index=previous_event_safe,
    )
    invalid_through_current = invalid_prefix[:, 1:, :]
    pair_valid = (
        event_active
        & (previous_event >= 0)
        & ((invalid_through_current - invalid_before_previous) == 0)
    )
    pair_valid_float = pair_valid.to(torch.float32)

    if gap_decay_seconds == 0.0:
        pair_weight = torch.ones_like(pair_loss, dtype=torch.float32)
    else:
        activity_locations = torch.where(
            activity_active,
            time_indices,
            torch.full_like(time_indices, -1),
        )
        latest_activity = torch.cummax(activity_locations, dim=1).values
        last_active_before_onset = torch.cat((
            torch.full_like(latest_activity[:, :1, :], -1),
            latest_activity[:, :-1, :],
        ), dim=1)
        last_active_before_onset = torch.maximum(
            last_active_before_onset,
            previous_event,
        )
        rest_frames = torch.clamp(
            time_indices - last_active_before_onset - 1,
            min=0,
        )
        gap_seconds = rest_frames.to(torch.float32) / frames_per_second
        # Python floats below float32.tiny cast to zero. Clamp before the
        # tensor division so a zero-length rest remains 0 rather than 0/0.
        decay = max(gap_decay_seconds, torch.finfo(torch.float32).tiny)
        pair_weight = torch.exp(-gap_seconds / decay)

    pair_weight = pair_weight * pair_valid_float
    weighted_loss = weighted_loss + torch.sum(pair_loss * pair_weight)
    pair_count = pair_count + torch.sum(pair_valid_float)

    return weighted_loss / torch.clamp(pair_count, min=1.0)


def choral_task_bce(model, output_dict, target_dict):
    cfg = model.cfg
    spec = get_task_spec(cfg)
    losses = []

    if 'voice_frame_output' in output_dict:
        losses.append(
            float(getattr(cfg.choral, 'voice_frame_loss_weight', 1.0))
            * choral_bce(
                output_dict['voice_frame_output'],
                target_dict['voice_frame_roll'],
                target_dict.get('voice_frame_mask_roll'),
                positive_weight=float(getattr(cfg.choral, 'voice_frame_positive_weight', 1.0)),
            )
        )
    if 'voice_onset_output' in output_dict:
        losses.append(
            float(getattr(cfg.choral, 'voice_onset_loss_weight', 1.0))
            * choral_bce(
                output_dict['voice_onset_output'],
                target_dict['voice_onset_roll'],
                target_dict.get('voice_onset_mask_roll'),
                positive_weight=float(getattr(cfg.choral, 'voice_onset_positive_weight', 1.0)),
            )
        )
    if spec.offset and 'voice_offset_output' in output_dict and 'voice_offset_roll' in target_dict:
        losses.append(
            float(getattr(cfg.choral, 'voice_offset_loss_weight', 1.0))
            * choral_bce(
                output_dict['voice_offset_output'],
                target_dict['voice_offset_roll'],
                target_dict.get('voice_offset_mask_roll'),
                positive_weight=float(getattr(cfg.choral, 'voice_offset_positive_weight', 1.0)),
            )
        )

    range_prior_weight = float(getattr(cfg.choral, 'range_prior_loss_weight', 0.0))
    if range_prior_weight > 0.0 and 'voice_frame_output' in output_dict:
        prior_streams = [
            (
                output_dict['voice_frame_output'],
                target_dict['voice_frame_roll'],
                target_dict.get('voice_frame_mask_roll'),
            )
        ]
        for output_key, target_key, mask_key in (
            ('voice_onset_output', 'voice_onset_roll', 'voice_onset_mask_roll'),
            ('voice_offset_output', 'voice_offset_roll', 'voice_offset_mask_roll'),
        ):
            if output_key in output_dict and target_key in target_dict:
                prior_streams.append(
                    (
                        output_dict[output_key],
                        target_dict[target_key],
                        target_dict.get(mask_key),
                    )
                )
        losses.append(
            range_prior_weight
            * torch.stack([
                voice_range_prior_loss(stream, cfg, target=target, mask=mask)
                for stream, target, mask in prior_streams
            ]).mean()
        )

    continuity_weight = float(getattr(cfg.choral, 'continuity_prior_loss_weight', 0.0))
    if continuity_weight > 0.0:
        required_continuity_keys = {
            'voice_onset_roll',
            'voice_frame_roll',
            'voice_frame_mask_roll',
        }
        missing_continuity_keys = sorted(required_continuity_keys - target_dict.keys())
        if 'voice_onset_output' not in output_dict or missing_continuity_keys:
            raise ValueError(
                'continuity_prior_loss_weight > 0 requires PawCT voice onset '
                'outputs plus onset/frame activity targets and frame masks; '
                'use model.mode=frame_onset or frame_onset_offset. Missing: '
                f'{missing_continuity_keys}'
            )
        losses.append(
            continuity_weight
            * voice_continuity_prior_loss(
                output_dict['voice_onset_output'],
                target_dict['voice_onset_roll'],
                target_dict.get('voice_onset_mask_roll'),
                activity_target=target_dict['voice_frame_roll'],
                activity_mask=target_dict['voice_frame_mask_roll'],
                frames_per_second=float(cfg.feature.frames_per_second),
                gap_decay_seconds=float(
                    getattr(cfg.choral, 'oc_gap_decay_seconds', 2.0)
                ),
            )
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
            float(getattr(cfg.choral, 'union_offset_loss_weight', 0.0))
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
