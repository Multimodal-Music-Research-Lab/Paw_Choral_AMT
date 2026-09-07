# SPDX-License-Identifier: Apache-2.0
# Includes modifications to ByteDance's piano_transcription implementation.

from __future__ import annotations

from collections.abc import Mapping
import logging
import math
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.utils.data
from hydra import compose, initialize
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except Exception:
    wandb = None

from checkpointing import (
    PRIOR_LOSS_SEMANTICS_VERSION,
    build_checkpoint_metadata,
    checkpoint_data_target_semantics,
    checkpoint_compatibility_allowlists,
    format_checkpoint_load_report,
    load_model_checkpoint,
    resolve_config,
    validate_checkpoint_behavior,
)
from data_generator import (
    Augmentor,
    ChoralSATBDataset,
    ChoralUnionDataset,
    EvalSampler,
    MAPS_Dataset,
    Maestro_Dataset,
    SMD_Dataset,
    BasePianoDataset,
    Sampler,
    collate_fn,
    deserialize_numpy_random_state,
    serialize_numpy_random_state,
)
from evaluate import SegmentEvaluator
from losses import get_loss_func, resolve_loss_type
from models import build_model
from utilities import create_folder, create_logging, get_model_name, get_task_spec, move_data_to_device


DATASET_CLASS_MAP = {
    'maestro': Maestro_Dataset,
    'smd': SMD_Dataset,
    'maps': MAPS_Dataset,
    'cantoria': lambda cfg, is_training: BasePianoDataset(cfg, 'cantoria', is_training),
    'csd': lambda cfg, is_training: BasePianoDataset(cfg, 'csd', is_training),
    'youchorale': lambda cfg, is_training: BasePianoDataset(cfg, 'youchorale', is_training),
    'youchorale_pro': lambda cfg, is_training: BasePianoDataset(cfg, 'youchorale_pro', is_training),
}

REPRODUCIBILITY_FORMAT_VERSION = 2
TRAINING_SEMANTICS_FORMAT_VERSION = 1


def _config_value(cfg, *path, default=None):
    """Read a nested value from either an OmegaConf/object or plain mapping."""

    current = cfg
    for key in path:
        if isinstance(current, Mapping):
            if key not in current:
                return default
            current = current[key]
        else:
            if not hasattr(current, key):
                return default
            current = getattr(current, key)
    return current


def _canonical_target_assignment(cfg) -> str:
    """Resolve the effective SATB target method for object or mapping configs."""

    value = _config_value(cfg, 'choral', 'target_assignment', default=None)
    used_legacy_key = value is None or not str(value).strip()
    if used_legacy_key:
        value = _config_value(
            cfg,
            'choral',
            'voice_assignment_method',
            default=None,
        )
    if value is None or not str(value).strip():
        value = 'part_name'
    aliases = {
        'part_name': 'part_name',
        'range_prior': 'range_prior',
        'rp': 'range_prior',
        'ordered_continuity': 'ordered_continuity',
        'oc': 'ordered_continuity',
        'range_masked_continuity': 'range_masked_continuity',
        'legacy_range_prior': 'legacy_range_prior',
        'legacy_ordered_continuity': 'legacy_ordered_continuity',
        'legacy_range_masked_continuity': 'legacy_range_masked_continuity',
    }
    key = str(value).strip().lower().replace('-', '_')
    if key not in aliases:
        raise ValueError(f'Unsupported choral target assignment {value!r}')
    resolved = aliases[key]
    if used_legacy_key and resolved in {
        'range_prior',
        'ordered_continuity',
        'range_masked_continuity',
    }:
        return f'legacy_{resolved}'
    return resolved


def _canonical_model_mode(cfg) -> str:
    mode = _config_value(cfg, 'model', 'mode', default=None)
    if mode is None:
        legacy_type = _config_value(cfg, 'model', 'type', default=None)
        mode = {'note': 'frame_onset_offset'}.get(
            legacy_type,
            'frame_onset_offset',
        )
    return str(mode).strip().lower()


def _canonical_loss_type(cfg) -> str:
    loss_type = str(
        _config_value(cfg, 'exp', 'loss_type', default='auto')
    ).strip()
    if loss_type != 'auto':
        return loss_type
    if bool(_config_value(cfg, 'choral', 'enable', default=False)):
        return 'choral_task_bce'
    return {
        'frame_onset': 'frame_onset_bce',
        'frame_onset_offset': 'frame_onset_offset_bce',
    }[_canonical_model_mode(cfg)]


def build_training_semantics_signature(cfg) -> dict:
    """Return all configured semantics that can change continued training.

    Runtime-only destinations and resume coordinates are intentionally omitted;
    data bytes are bound independently by the sampler content manifest.
    """

    resolved = resolve_config(cfg)
    get = lambda *path, default=None: _config_value(
        resolved,
        *path,
        default=default,
    )
    choral_enabled = bool(get('choral', 'enable', default=False))
    model_mode = _canonical_model_mode(resolved)

    return {
        'format_version': TRAINING_SEMANTICS_FORMAT_VERSION,
        'model': {
            'architecture': str(get('model', 'arch', default='pagct')).strip().lower(),
            'mode': model_mode,
            'choral_enabled': choral_enabled,
            'num_voices': int(get('choral', 'num_voices', default=4)),
            'use_presence_head': bool(
                get('choral', 'use_presence_head', default=True)
            ),
            'apply_presence_gate': bool(
                get('choral', 'apply_presence_gate', default=True)
            ),
            'assignment_module': str(
                get('choral', 'assignment_module', default='heads')
            ).strip(),
            'assignment_temperature': float(
                get('choral', 'assignment_temperature', default=1.0)
            ),
            'assignment_hidden_channels': int(
                get('choral', 'assignment_hidden_channels', default=64)
            ),
            'assignment_rnn_hidden_size': int(
                get('choral', 'assignment_rnn_hidden_size', default=128)
            ),
            'voice_interaction_module': str(
                get('choral', 'voice_interaction_module', default='none')
            ).strip(),
            'voice_interaction_dim': int(
                get('choral', 'voice_interaction_dim', default=256)
            ),
            'voice_interaction_heads': int(
                get('choral', 'voice_interaction_heads', default=4)
            ),
            'voice_interaction_layers': int(
                get('choral', 'voice_interaction_layers', default=1)
            ),
            'voice_interaction_dropout': float(
                get('choral', 'voice_interaction_dropout', default=0.1)
            ),
        },
        'feature': {
            'audio_feature': str(
                get('feature', 'audio_feature', default='logmel')
            ).strip().lower(),
            'classes_num': int(get('feature', 'classes_num', default=88)),
            'segment_seconds': float(
                get('feature', 'segment_seconds', default=10.0)
            ),
            'hop_seconds': float(get('feature', 'hop_seconds', default=1.0)),
            'sample_rate': int(get('feature', 'sample_rate', default=16000)),
            'fft_size': int(get('feature', 'fft_size', default=2048)),
            'frames_per_second': float(
                get('feature', 'frames_per_second', default=100)
            ),
            'use_augmentation': bool(
                get('feature', 'use_augmentation', default=False)
            ),
            'begin_note': int(get('feature', 'begin_note', default=21)),
            'velocity_scale': float(
                get('feature', 'velocity_scale', default=128)
            ),
            'max_note_shift': int(
                get('feature', 'max_note_shift', default=0)
            ),
        },
        'data': {
            'target_semantics': checkpoint_data_target_semantics(resolved),
            'train_set': str(get('dataset', 'train_set', default='maestro')),
            'selection_set': str(
                get(
                    'exp',
                    'selection_dataset',
                    default=get('dataset', 'test_set', default='smd'),
                )
            ),
            'test_set': str(get('dataset', 'test_set', default='smd')),
            'cantoria_f0_source': str(
                get('dataset', 'cantoria_f0_source', default='crepe')
            ).strip().lower(),
            'mini_data': bool(get('exp', 'mini_data', default=False)),
            'batch_size': int(get('exp', 'batch_size', default=8)),
            'num_workers': int(get('exp', 'num_workers', default=0)),
        },
        'target': {
            'assignment': _canonical_target_assignment(resolved),
            'voice_names': list(
                get('choral', 'voice_names', default=['S', 'A', 'T', 'B'])
            ),
            'preserve_known_part_labels': bool(
                get('choral', 'preserve_known_part_labels', default=True)
            ),
            'evaluation_reference_assignment': str(
                get(
                    'choral',
                    'evaluation_reference_assignment',
                    default='part_name',
                )
            ).strip().lower(),
            'range_mins': list(
                get(
                    'choral',
                    'voice_assignment_range_mins',
                    default=[60, 55, 48, 40],
                )
            ),
            'range_maxs': list(
                get(
                    'choral',
                    'voice_assignment_range_maxs',
                    default=[88, 79, 72, 67],
                )
            ),
            'range_margin': float(
                get(
                    'choral',
                    'voice_assignment_range_margin',
                    default=2.0,
                )
            ),
            'mask_penalty': float(
                get(
                    'choral',
                    'voice_assignment_mask_penalty',
                    default=8.0,
                )
            ),
            'part_penalty': float(
                get(
                    'choral',
                    'voice_assignment_part_penalty',
                    default=2.0,
                )
            ),
            'continuity_weight': float(
                get(
                    'choral',
                    'voice_assignment_continuity_weight',
                    default=0.35,
                )
            ),
            'overlap_penalty': float(
                get(
                    'choral',
                    'voice_assignment_overlap_penalty',
                    default=4.0,
                )
            ),
            'oc_gap_decay_seconds': float(
                get('choral', 'oc_gap_decay_seconds', default=2.0)
            ),
            'oc_overlap_tolerance_seconds': float(
                get(
                    'choral',
                    'oc_overlap_tolerance_seconds',
                    default=0.05,
                )
            ),
        },
        'loss': {
            'prior_loss_semantics_version': PRIOR_LOSS_SEMANTICS_VERSION,
            'configured_type': str(get('exp', 'loss_type', default='auto')).strip(),
            'resolved_type': _canonical_loss_type(resolved),
            **{
                key: float(get('choral', key, default=default))
                for key, default in {
                    'voice_frame_loss_weight': 1.0,
                    'voice_onset_loss_weight': 1.0,
                    'voice_offset_loss_weight': 1.0,
                    'voice_frame_positive_weight': 1.0,
                    'voice_onset_positive_weight': 1.0,
                    'voice_offset_positive_weight': 1.0,
                    'range_prior_loss_weight': 0.0,
                    'continuity_prior_loss_weight': 0.0,
                    'union_frame_loss_weight': 0.5,
                    'union_onset_loss_weight': 0.5,
                    'union_offset_loss_weight': 0.0,
                    'presence_loss_weight': 0.2,
                    'assignment_range_loss_weight': 0.0,
                    'assignment_entropy_loss_weight': 0.0,
                }.items()
            },
        },
        'optimization': {
            'optimizer': str(get('exp', 'optim', default='adam')).strip().lower(),
            'learning_rate': float(get('exp', 'learning_rate', default=1e-4)),
            'decay': bool(get('exp', 'decay', default=True)),
            'reduce_iteration': int(
                get('exp', 'reduce_iteration', default=10_000)
            ),
            'random_seed': int(get('exp', 'random_seed', default=86)),
            'deterministic_cudnn': bool(
                get('exp', 'deterministic_cudnn', default=True)
            ),
            'cuda': bool(get('exp', 'cuda', default=True)),
            'total_iteration': int(
                get('exp', 'total_iteration', default=200_000)
            ),
            'eval_iteration': int(
                get('exp', 'eval_iteration', default=5_000)
            ),
            'selection_metric': str(
                get('exp', 'selection_metric', default='auto')
            ).strip(),
            'selection_mode': str(
                get('exp', 'selection_mode', default='max')
            ).strip().lower(),
            'early_stopping_patience_evals': get(
                'exp',
                'early_stopping_patience_evals',
                default=None,
            ),
            'max_eval_batches': get(
                'exp',
                'max_eval_batches',
                default=None,
            ),
            'debug': bool(get('exp', 'debug', default=False)),
        },
    }


def _training_semantics_mismatches(runtime, checkpoint):
    mismatches = []

    def compare(expected, actual, path):
        if isinstance(expected, Mapping):
            if not isinstance(actual, Mapping):
                mismatches.append(
                    f'{path}: checkpoint={actual!r}, runtime=mapping'
                )
                return
            for key, value in expected.items():
                nested_path = f'{path}.{key}' if path else str(key)
                if key not in actual:
                    mismatches.append(
                        f'{nested_path}: checkpoint=<missing>, runtime={value!r}'
                    )
                else:
                    compare(value, actual[key], nested_path)
            return
        if actual != expected:
            mismatches.append(
                f'{path}: checkpoint={actual!r}, runtime={expected!r}'
            )

    compare(runtime, checkpoint, '')
    return tuple(mismatches)


def validate_resume_training_semantics(
    cfg,
    checkpoint,
    reproducibility,
    *,
    allow_inexact_resume: bool,
) -> None:
    """Reject a continuation whose next update would use changed semantics."""

    saved_signature = None
    if isinstance(reproducibility, Mapping):
        saved_signature = reproducibility.get('training_semantics')
    if saved_signature is None and isinstance(checkpoint, Mapping):
        saved_config = checkpoint.get('resolved_config')
        if isinstance(saved_config, Mapping):
            saved_signature = build_training_semantics_signature(saved_config)

    if not isinstance(saved_signature, Mapping):
        if allow_inexact_resume:
            logging.warning(
                'Checkpoint has no verifiable training-semantics signature; '
                'continuing only because exp.allow_inexact_resume=true. This '
                'run is not an exact resume.'
            )
            return
        raise ValueError(
            'Exact resume requires a checkpoint training-semantics signature '
            'or resolved_config metadata'
        )

    runtime_signature = build_training_semantics_signature(cfg)
    mismatches = _training_semantics_mismatches(
        runtime_signature,
        saved_signature,
    )
    if mismatches:
        preview = '; '.join(mismatches[:12])
        if len(mismatches) > 12:
            preview += f'; ... ({len(mismatches) - 12} more)'
        raise ValueError(
            'Resume training semantics do not match the checkpoint; refusing '
            f'to change the continued optimization trajectory: {preview}'
        )


def evaluation_data_identities(eval_loaders: Mapping) -> dict:
    """Return stable identities for data that can select/stop a training run."""

    identities = {}
    for name, loader in sorted(eval_loaders.items()):
        sampler = loader.batch_sampler
        identities[str(name)] = {
            'dataset_type': str(sampler.dataset_type),
            'split': str(sampler.split),
            'segment_identity': str(sampler.segment_identity),
        }
    return identities


def validate_resume_evaluation_data(
    saved_identities,
    runtime_identities,
    *,
    allow_inexact_resume: bool,
) -> None:
    """Fail closed when validation bytes used for selection have changed."""

    if not isinstance(saved_identities, Mapping):
        if allow_inexact_resume:
            logging.warning(
                'Checkpoint has no evaluation-data identity; continuing only '
                'because exp.allow_inexact_resume=true. This run is not an '
                'exact resume.'
            )
            return
        raise ValueError(
            'Exact resume requires checkpoint evaluation-data identities'
        )
    mismatches = _training_semantics_mismatches(
        runtime_identities,
        saved_identities,
    )
    if mismatches:
        raise ValueError(
            'Evaluation data used for checkpoint selection changed since the '
            'checkpoint: ' + '; '.join(mismatches[:12])
        )


def seed_everything(seed: int, *, deterministic_cudnn: bool = True) -> None:
    """Seed model initialization and all main-process stochastic operations."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = not deterministic_cudnn
        torch.backends.cudnn.deterministic = deterministic_cudnn


def capture_rng_state() -> dict:
    """Return serializable main-process RNG state for exact continuation."""

    state = {
        'python': random.getstate(),
        'numpy': serialize_numpy_random_state(np.random.get_state()),
        'torch_cpu': torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Mapping) -> None:
    """Restore RNG state saved by :func:`capture_rng_state`."""

    required = {'python', 'numpy', 'torch_cpu'}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f'Checkpoint RNG state is incomplete: missing {missing}')
    random.setstate(state['python'])
    np.random.set_state(deserialize_numpy_random_state(state['numpy']))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if 'torch_cuda' in state:
        if not torch.cuda.is_available():
            raise ValueError('Checkpoint contains CUDA RNG state but CUDA is unavailable')
        saved_cuda = [item.cpu() for item in state['torch_cuda']]
        if len(saved_cuda) != torch.cuda.device_count():
            raise ValueError(
                'Checkpoint CUDA RNG device count does not match this run: '
                f'{len(saved_cuda)} != {torch.cuda.device_count()}'
            )
        torch.cuda.set_rng_state_all(saved_cuda)


def stochastic_training_pipeline_enabled(cfg) -> bool:
    return bool(
        getattr(cfg.feature, 'use_augmentation', False)
        or int(getattr(cfg.feature, 'max_note_shift', 0)) != 0
    )


def seed_data_worker(worker_id: int) -> None:
    """Give each DataLoader worker independent deterministic dataset RNGs."""

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    worker_seed = int(worker_info.seed % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    dataset = worker_info.dataset
    if hasattr(dataset, 'random_state'):
        dataset.random_state = np.random.RandomState(worker_seed)
    augmentor = getattr(getattr(dataset, 'cfg', None), 'feature', None)
    augmentor = getattr(augmentor, 'augmentor', None)
    if augmentor is not None and hasattr(augmentor, 'random_state'):
        augmentor.random_state = np.random.RandomState((worker_seed + 1) % (2**32))


def make_loader_generator(seed: int, stream: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed) + 10_000 * int(stream))
    return generator


def capture_dataset_rng_state(dataset) -> dict:
    state = {}
    if hasattr(dataset, 'random_state'):
        state['dataset_numpy'] = serialize_numpy_random_state(dataset.random_state)
    augmentor = getattr(getattr(dataset, 'cfg', None), 'feature', None)
    augmentor = getattr(augmentor, 'augmentor', None)
    if augmentor is not None and hasattr(augmentor, 'random_state'):
        state['augmentor_numpy'] = serialize_numpy_random_state(
            augmentor.random_state
        )
    return state


def restore_dataset_rng_state(dataset, state: Mapping) -> None:
    if hasattr(dataset, 'random_state'):
        if 'dataset_numpy' not in state:
            raise ValueError('Checkpoint is missing dataset RNG state')
        dataset.random_state.set_state(
            deserialize_numpy_random_state(state['dataset_numpy'])
        )
    augmentor = getattr(getattr(dataset, 'cfg', None), 'feature', None)
    augmentor = getattr(augmentor, 'augmentor', None)
    if augmentor is not None and hasattr(augmentor, 'random_state'):
        if 'augmentor_numpy' not in state:
            raise ValueError('Checkpoint is missing augmentor RNG state')
        augmentor.random_state.set_state(
            deserialize_numpy_random_state(state['augmentor_numpy'])
        )



def _build_dataset(
    cfg,
    dataset_name: str,
    is_training: bool,
    formal_evaluation: bool = False,
):
    if dataset_name in {'youchorale', 'youchorale_pro', 'csd', 'cantoria'}:
        dataset_class = (
            ChoralSATBDataset
            if getattr(cfg.choral, 'enable', False)
            else ChoralUnionDataset
        )
        return dataset_class(
            cfg,
            dataset_name,
            is_training=is_training,
            formal_evaluation=formal_evaluation,
        )
    dataset_builder = DATASET_CLASS_MAP[dataset_name]
    return dataset_builder(cfg, is_training=is_training)



def _build_optimizer(cfg, model):
    optim_name = getattr(cfg.exp, 'optim', 'adam').lower()
    if optim_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=cfg.exp.learning_rate)
    if optim_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=cfg.exp.learning_rate)
    raise ValueError(f'Unsupported optimizer: {cfg.exp.optim}')



def _wandb_init(cfg, run_id=None):
    if not getattr(cfg.wandb, 'enable', False):
        return None
    if wandb is None:
        logging.warning('wandb is enabled in config but not installed. Continue without wandb.')
        return None
    spec = get_task_spec(cfg)
    return wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        id=run_id,
        resume='must' if run_id else 'allow',
        config={
            'model_arch': spec.arch,
            'model_mode': spec.mode,
            'audio_feature': cfg.feature.audio_feature,
            'sample_rate': cfg.feature.sample_rate,
            'frames_per_second': cfg.feature.frames_per_second,
            'train_set': cfg.dataset.train_set,
        },
    )


def _tensorboard_init(cfg, model_name):
    if not getattr(cfg.tensorboard, 'enable', True):
        return None
    tb_dir = os.path.join(cfg.tensorboard.dir, model_name)
    create_folder(tb_dir)
    return SummaryWriter(log_dir=tb_dir)



def _prepare_batch(batch_data_dict, device):
    for key in batch_data_dict.keys():
        batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
    return batch_data_dict



def forward_pass(cfg, model, batch_data_dict, device):
    batch_data_dict = _prepare_batch(batch_data_dict, device)
    batch_output_dict = model(batch_data_dict['waveform'])
    loss_type = resolve_loss_type(cfg)
    loss = get_loss_func(loss_type)(model, batch_output_dict, batch_data_dict)
    return batch_output_dict, loss



def get_sampler(cfg, purpose, split, is_eval=None):
    sampler_cls = {'train': Sampler, 'eval': EvalSampler}[purpose]
    return sampler_cls(cfg, split=split, is_eval=is_eval)


def resolve_selection_metric(cfg) -> str:
    metric = str(getattr(cfg.exp, 'selection_metric', 'auto')).strip()
    if metric == 'auto':
        return 'mean_voice_frame_ap' if getattr(cfg.choral, 'enable', False) else 'frame_ap'
    return metric


def selection_improved(value: float, best_value: float | None, mode: str) -> bool:
    if mode not in {'max', 'min'}:
        raise ValueError("exp.selection_mode must be 'max' or 'min'")
    if not math.isfinite(value):
        return False
    if best_value is None:
        return True
    if not math.isfinite(best_value):
        raise ValueError(f'Checkpoint best selection value must be finite, got {best_value}')
    return value > best_value if mode == 'max' else value < best_value


def should_decay_learning_rate(cfg, iteration: int) -> bool:
    """Return whether the update at ``iteration`` starts a decay interval."""

    if not cfg.exp.decay:
        return False
    reduce_iteration = int(cfg.exp.reduce_iteration)
    if reduce_iteration <= 0:
        raise ValueError('exp.reduce_iteration must be positive')
    return bool(cfg.exp.decay and iteration > 0 and iteration % reduce_iteration == 0)


def periodic_action_due(
    iteration: int,
    interval: int,
    *,
    fresh_run: bool,
) -> bool:
    """Return whether logging/saving is due without perturbing resume cadence."""

    iteration = int(iteration)
    interval = int(interval)
    if interval <= 0:
        raise ValueError(f'periodic interval must be positive, got {interval}')
    return bool((fresh_run and iteration == 0) or (iteration + 1) % interval == 0)


def validate_resume_optimizer_state(
    has_optimizer_state: bool,
    *,
    allow_inexact_resume: bool,
) -> None:
    """Require optimizer moments for an exact training continuation."""

    if has_optimizer_state:
        return
    if not allow_inexact_resume:
        raise ValueError(
            'The requested checkpoint has no optimizer state, so an exact '
            'continuation cannot be established. Set exp.allow_inexact_resume=true '
            'only for a clearly labelled historical diagnostic.'
        )
    logging.warning(
        'Checkpoint has no optimizer state; model weights were loaded but the '
        'optimizer is newly initialized because exp.allow_inexact_resume=true. '
        'This run is not an exact resume.'
    )


def validate_resume_device_state(
    reproducibility,
    runtime_device,
    *,
    allow_inexact_resume: bool,
) -> None:
    """Bind exact continuation to the device type and its RNG stream."""

    if not isinstance(reproducibility, Mapping):
        return
    saved_device = reproducibility.get('device_type')
    if saved_device is None:
        if allow_inexact_resume:
            logging.warning(
                'Checkpoint has no runtime device identity; continuing only '
                'because exp.allow_inexact_resume=true. This run is not an '
                'exact resume.'
            )
            return
        raise ValueError('Exact resume requires checkpoint runtime device identity')

    current_device = str(torch.device(runtime_device).type)
    if str(saved_device) != current_device:
        raise ValueError(
            'Resume device type does not match the checkpoint: '
            f'{current_device!r} != {saved_device!r}'
        )
    rng_state = reproducibility.get('rng_state')
    if (
        current_device == 'cuda'
        and isinstance(rng_state, Mapping)
        and 'torch_cuda' not in rng_state
    ):
        if allow_inexact_resume:
            logging.warning(
                'Checkpoint has no CUDA RNG state; continuing only because '
                'exp.allow_inexact_resume=true. This run is not an exact resume.'
            )
            return
        raise ValueError('Exact CUDA resume requires checkpoint CUDA RNG state')


def restore_selection_state(
    saved_selection,
    *,
    dataset: str,
    metric: str,
    mode: str,
    require_complete: bool = False,
    allow_inexact_resume: bool = False,
) -> tuple[float | None, int | None, int]:
    """Restore best-checkpoint state only when its objective is unchanged."""

    default_state = (None, None, 0)

    try:
        if not isinstance(saved_selection, Mapping):
            if require_complete:
                raise ValueError(
                    'Exact resume requires a complete checkpoint selection state'
                )
            return default_state

        required_keys = {
            'dataset',
            'metric',
            'mode',
            'best_value',
            'best_iteration',
            'evaluations_without_improvement',
        }
        if require_complete:
            missing = sorted(required_keys.difference(saved_selection))
            if missing:
                raise ValueError(
                    'Exact resume requires a complete checkpoint selection state; '
                    f'missing={missing}'
                )

        expected = {'dataset': dataset, 'metric': metric, 'mode': mode}
        mismatches = {
            key: (saved_selection.get(key), value)
            for key, value in expected.items()
            if saved_selection.get(key) != value
        }
        if mismatches:
            raise ValueError(
                'Cannot resume checkpoint-selection state with a different objective: '
                f'{mismatches}'
            )

        best_value = saved_selection.get('best_value')
        if best_value is not None:
            best_value = float(best_value)
            if not math.isfinite(best_value):
                raise ValueError(
                    f'Checkpoint best selection value must be finite, got {best_value}'
                )

        best_iteration_value = saved_selection.get('best_iteration')
        if require_complete and best_iteration_value is not None and (
            isinstance(best_iteration_value, bool)
            or not isinstance(best_iteration_value, (int, np.integer))
        ):
            raise ValueError(
                'Checkpoint best_iteration must be an integer or null for exact resume'
            )
        best_iteration = (
            None if best_iteration_value is None else int(best_iteration_value)
        )

        counter_value = saved_selection.get('evaluations_without_improvement', 0)
        if require_complete and (
            isinstance(counter_value, bool)
            or not isinstance(counter_value, (int, np.integer))
        ):
            raise ValueError(
                'Checkpoint evaluations_without_improvement must be an integer '
                'for exact resume'
            )
        evaluations_without_improvement = int(counter_value)
        if evaluations_without_improvement < 0:
            raise ValueError('evaluations_without_improvement cannot be negative')
        if (best_value is None) != (best_iteration is None):
            raise ValueError(
                'Checkpoint best_value and best_iteration must both be set or both be null'
            )
        return best_value, best_iteration, evaluations_without_improvement
    except (OverflowError, TypeError, ValueError) as exc:
        if require_complete and allow_inexact_resume:
            logging.warning(
                'Resetting checkpoint-selection history because '
                'exp.allow_inexact_resume=true: %s',
                exc,
            )
            return default_state
        raise


def training_has_remaining_updates(start_iteration: int, total_iteration: int) -> bool:
    """Whether the half-open update interval ``[start, total)`` is nonempty."""

    return int(start_iteration) < int(total_iteration)



def train(cfg):
    seed_everything(
        cfg.exp.random_seed,
        deterministic_cudnn=bool(getattr(cfg.exp, 'deterministic_cudnn', True)),
    )
    spec = get_task_spec(cfg)
    device = torch.device('cuda') if cfg.exp.cuda and torch.cuda.is_available() else torch.device('cpu')

    if getattr(cfg.feature, 'use_augmentation', False):
        cfg.feature.augmentor = Augmentor(cfg)
    else:
        cfg.feature.augmentor = None

    model = build_model(cfg).to(device)
    optimizer = _build_optimizer(cfg, model)

    model_name = get_model_name(cfg)
    checkpoints_dir = os.path.join(cfg.exp.workspace, 'checkpoints', model_name)
    logs_dir = os.path.join(cfg.exp.workspace, 'logs', model_name)
    create_folder(checkpoints_dir)
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info(cfg)
    logging.info('Using device: %s', device)
    logging.info('Resolved task: arch=%s mode=%s', spec.arch, spec.mode)
    logging.info('Resolved post processor: %s', cfg.post.post_processor_type)

    start_iteration = 0
    wandb_run_id = None
    resume_selection = None
    resume_reproducibility = None
    resume_requested = int(cfg.exp.resume_iteration) > 0
    allow_inexact_resume = bool(getattr(cfg.exp, 'allow_inexact_resume', False))
    if resume_requested:
        checkpoint_path = os.path.join(checkpoints_dir, f'{cfg.exp.resume_iteration}_iteration.pth')
        if os.path.exists(checkpoint_path):
            logging.info('Loading checkpoint %s', checkpoint_path)
            allowed_missing_keys, allowed_unexpected_keys = checkpoint_compatibility_allowlists(cfg)
            checkpoint, load_report = load_model_checkpoint(
                model,
                checkpoint_path,
                map_location=device,
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
            validate_checkpoint_behavior(cfg, checkpoint)
            logging.info('Checkpoint load audit: %s', format_checkpoint_load_report(load_report))

            is_training_checkpoint = (
                load_report.source_format != 'raw_state_dict'
                and isinstance(checkpoint, Mapping)
                and 'optimizer' in checkpoint
            )
            validate_resume_optimizer_state(
                is_training_checkpoint,
                allow_inexact_resume=allow_inexact_resume,
            )
            if is_training_checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if load_report.source_format != 'raw_state_dict' and isinstance(checkpoint, Mapping):
                # Saved iteration N already contains update N, so resume at N+1.
                start_iteration = int(checkpoint.get('iteration', cfg.exp.resume_iteration)) + 1
                wandb_run_id = checkpoint.get('wandb_run_id')
                resume_selection = checkpoint.get('selection')
                resume_reproducibility = checkpoint.get('reproducibility')
            else:
                start_iteration = int(cfg.exp.resume_iteration)
        else:
            raise FileNotFoundError(
                f'Requested resume checkpoint does not exist: {checkpoint_path}'
            )

        validate_resume_training_semantics(
            cfg,
            checkpoint,
            resume_reproducibility,
            allow_inexact_resume=allow_inexact_resume,
        )

        if not isinstance(resume_reproducibility, Mapping):
            if not allow_inexact_resume:
                raise ValueError(
                    'The requested checkpoint has no reproducibility state, so an exact '
                    'continuation cannot be established. Set exp.allow_inexact_resume=true '
                    'only for a clearly labelled historical diagnostic.'
                )
            logging.warning(
                'Continuing without checkpoint RNG state because '
                'exp.allow_inexact_resume=true; this run is not an exact resume.'
            )
        elif int(resume_reproducibility.get('random_seed', -1)) != int(cfg.exp.random_seed):
            raise ValueError(
                'Resume random seed does not match the checkpoint: '
                f"{cfg.exp.random_seed} != {resume_reproducibility.get('random_seed')}"
            )
        elif bool(resume_reproducibility.get('deterministic_cudnn', True)) != bool(
            getattr(cfg.exp, 'deterministic_cudnn', True)
        ):
            raise ValueError(
                'Resume deterministic_cudnn setting does not match the checkpoint'
            )
        validate_resume_device_state(
            resume_reproducibility,
            device,
            allow_inexact_resume=allow_inexact_resume,
        )

        if (
            stochastic_training_pipeline_enabled(cfg)
            and int(cfg.exp.num_workers) > 0
            and not allow_inexact_resume
        ):
            raise ValueError(
                'Exact resume with stochastic waveform augmentation or pitch shifting '
                'requires exp.num_workers=0; prefetched worker-local RNG state cannot be '
                'recovered. Set exp.allow_inexact_resume=true only for a diagnostic.'
            )

    if not training_has_remaining_updates(start_iteration, cfg.exp.total_iteration):
        logging.info(
            'No updates remaining: start_iteration=%d, total_iteration=%d',
            start_iteration,
            cfg.exp.total_iteration,
        )
        return

    wandb_run = _wandb_init(cfg, wandb_run_id)
    tb_writer = _tensorboard_init(cfg, model_name)

    train_dataset = _build_dataset(cfg, cfg.dataset.train_set, is_training=True)
    eval_train_dataset = _build_dataset(cfg, cfg.dataset.train_set, is_training=False)

    train_sampler = get_sampler(cfg, purpose='train', split='train')
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
        worker_init_fn=seed_data_worker,
        generator=make_loader_generator(cfg.exp.random_seed, 1),
    )
    eval_train_loader = torch.utils.data.DataLoader(
        dataset=eval_train_dataset,
        batch_sampler=get_sampler(cfg, purpose='eval', split='train'),
        collate_fn=collate_fn,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
        worker_init_fn=seed_data_worker,
        generator=make_loader_generator(cfg.exp.random_seed, 2),
    )

    eval_loaders = {}
    eval_split = 'validation'
    for dataset_index, dataset_name in enumerate(
        dict.fromkeys([cfg.dataset.train_set, cfg.dataset.test_set]),
        start=3,
    ):
        eval_dataset = _build_dataset(
            cfg,
            dataset_name,
            is_training=False,
            formal_evaluation=True,
        )
        eval_loaders[dataset_name] = torch.utils.data.DataLoader(
            dataset=eval_dataset,
            batch_sampler=get_sampler(cfg, purpose='eval', split=eval_split, is_eval=dataset_name),
            collate_fn=collate_fn,
            num_workers=cfg.exp.num_workers,
            pin_memory=True,
            worker_init_fn=seed_data_worker,
            generator=make_loader_generator(cfg.exp.random_seed, dataset_index),
        )

    if resume_requested:
        saved_evaluation_data = (
            resume_reproducibility.get('evaluation_data')
            if isinstance(resume_reproducibility, Mapping)
            else None
        )
        validate_resume_evaluation_data(
            saved_evaluation_data,
            evaluation_data_identities(eval_loaders),
            allow_inexact_resume=allow_inexact_resume,
        )

    evaluator = SegmentEvaluator(model, cfg)

    if start_iteration > 0:
        if isinstance(resume_reproducibility, Mapping):
            reproducibility_version = resume_reproducibility.get(
                'format_version',
                -1,
            )
            if isinstance(reproducibility_version, bool) or not isinstance(
                reproducibility_version,
                (int, np.integer),
            ):
                raise ValueError(
                    'Checkpoint reproducibility format_version must be an integer, '
                    f'got {reproducibility_version!r}'
                )
            reproducibility_version = int(reproducibility_version)
            if reproducibility_version not in {1, REPRODUCIBILITY_FORMAT_VERSION}:
                raise ValueError(
                    'Unsupported checkpoint reproducibility format: '
                    f'{reproducibility_version!r}; expected 1 or '
                    f'{REPRODUCIBILITY_FORMAT_VERSION}'
                )
            saved_next_iteration = int(
                resume_reproducibility.get('next_iteration', -1)
            )
            if saved_next_iteration != start_iteration:
                raise ValueError(
                    'Checkpoint iteration and reproducibility progress disagree: '
                    f'{start_iteration} != {saved_next_iteration}'
                )
            exact_data_resume = bool(
                resume_reproducibility.get('exact_data_resume_supported', False)
            )
            if not exact_data_resume and not allow_inexact_resume:
                raise ValueError(
                    'This checkpoint records a stochastic prefetched data pipeline, '
                    'which cannot be resumed exactly. Use the original run from the '
                    'beginning, or set exp.allow_inexact_resume=true only for a '
                    'labelled diagnostic.'
                )
            sampler_state = resume_reproducibility.get('train_sampler')
            if not isinstance(sampler_state, Mapping):
                raise ValueError('Checkpoint is missing exact train-sampler state')
            train_sampler.load_state_dict(dict(sampler_state))
            if exact_data_resume:
                dataset_state = resume_reproducibility.get('train_dataset_rng')
                if not isinstance(dataset_state, Mapping):
                    raise ValueError('Checkpoint is missing training-dataset RNG state')
                restore_dataset_rng_state(train_dataset, dataset_state)
            rng_state = resume_reproducibility.get('rng_state')
            if not isinstance(rng_state, Mapping):
                raise ValueError('Checkpoint is missing main-process RNG state')
            # Restore last: object construction and checkpoint loading are
            # allowed to consume RNG, but the first resumed forward pass is not.
            restore_rng_state(rng_state)
        else:
            # Explicit legacy opt-in still starts from the logically correct
            # data batch, but cannot reconstruct dropout/augmentation RNG.
            train_sampler.seek_batch(start_iteration)

    iteration = start_iteration
    train_bgn_time = time.time()
    running_train_loss = 0.0
    running_train_steps = 0
    loss_type = resolve_loss_type(cfg)
    logging.info('Resolved loss_type: %s', loss_type)
    selection_dataset = str(getattr(cfg.exp, 'selection_dataset', cfg.dataset.test_set))
    selection_metric = resolve_selection_metric(cfg)
    selection_mode = str(getattr(cfg.exp, 'selection_mode', 'max')).strip().lower()
    if selection_mode not in {'max', 'min'}:
        raise ValueError("exp.selection_mode must be 'max' or 'min'")
    early_stopping_patience = getattr(cfg.exp, 'early_stopping_patience_evals', None)
    if early_stopping_patience is not None:
        early_stopping_patience = int(early_stopping_patience)
        if early_stopping_patience <= 0:
            raise ValueError('exp.early_stopping_patience_evals must be null or positive')
    (
        best_selection_value,
        best_selection_iteration,
        evaluations_without_improvement,
    ) = restore_selection_state(
        resume_selection,
        dataset=selection_dataset,
        metric=selection_metric,
        mode=selection_mode,
        require_complete=resume_requested,
        allow_inexact_resume=allow_inexact_resume,
    )
    logging.info(
        'Checkpoint selection: dataset=%s metric=%s mode=%s patience=%s',
        selection_dataset,
        selection_metric,
        selection_mode,
        early_stopping_patience,
    )

    optimizer.zero_grad(set_to_none=True)

    for batch_data_dict in train_loader:
        if should_decay_learning_rate(cfg, iteration):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9

        model.train()
        _, loss = forward_pass(cfg, model, batch_data_dict, device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        running_train_loss += float(loss.item())
        running_train_steps += 1

        should_log = periodic_action_due(
            iteration,
            cfg.exp.eval_iteration,
            fresh_run=not resume_requested,
        )
        should_save = periodic_action_due(
            iteration,
            cfg.exp.save_iteration,
            fresh_run=not resume_requested,
        )
        improved_this_iteration = False
        stop_after_iteration = False

        if should_log:
            train_fin_time = time.time()
            averaged_train_loss = running_train_loss / max(1, running_train_steps)
            train_statistics = evaluator.evaluate(eval_train_loader)
            valid_statistics = {name: evaluator.evaluate(loader) for name, loader in eval_loaders.items()}

            logging.info('------------------------------------')
            logging.info('Iteration: %d / %d', iteration, cfg.exp.total_iteration)
            logging.info('Train loss: %.4f', averaged_train_loss)
            logging.info('Train statistics: %s', train_statistics)
            for name, stats in valid_statistics.items():
                logging.info('Eval %s statistics: %s', name, stats)

            if selection_dataset not in valid_statistics:
                raise KeyError(
                    f'Selection dataset {selection_dataset!r} is unavailable; '
                    f'choices={sorted(valid_statistics)}'
                )
            selection_stats = valid_statistics[selection_dataset]
            if selection_metric not in selection_stats:
                raise KeyError(
                    f'Selection metric {selection_metric!r} is unavailable for '
                    f'{selection_dataset}; choices={sorted(selection_stats)}'
                )
            selection_value = float(selection_stats[selection_metric])
            improved_this_iteration = selection_improved(
                selection_value,
                best_selection_value,
                selection_mode,
            )
            if improved_this_iteration:
                best_selection_value = selection_value
                best_selection_iteration = iteration
                evaluations_without_improvement = 0
                logging.info(
                    'New best checkpoint candidate: %s/%s=%.6f at iteration %d',
                    selection_dataset,
                    selection_metric,
                    selection_value,
                    iteration,
                )
            else:
                evaluations_without_improvement += 1
                if not math.isfinite(selection_value):
                    logging.warning(
                        'Selection metric %s/%s is non-finite at iteration %d; '
                        'checkpoint is not eligible for best.pth',
                        selection_dataset,
                        selection_metric,
                        iteration,
                    )
            stop_after_iteration = (
                early_stopping_patience is not None
                and evaluations_without_improvement >= early_stopping_patience
            )

            if wandb_run is not None:
                log_dict = {'iteration': iteration, 'train_loss': averaged_train_loss}
                for key, value in train_statistics.items():
                    log_dict[f'train/{key}'] = value
                for dataset_name, stats in valid_statistics.items():
                    for key, value in stats.items():
                        log_dict[f'{dataset_name}/{key}'] = value
                wandb.log(log_dict)

            if tb_writer is not None:
                tb_writer.add_scalar('train/loss', averaged_train_loss, iteration)
                for key, value in train_statistics.items():
                    tb_writer.add_scalar(f'train/{key}', value, iteration)
                for dataset_name, stats in valid_statistics.items():
                    for key, value in stats.items():
                        tb_writer.add_scalar(f'{dataset_name}/{key}', value, iteration)
                tb_writer.flush()

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time
            logging.info('Train time: %.3f s, validate time: %.3f s', train_time, validate_time)
            running_train_loss = 0.0
            running_train_steps = 0
            train_bgn_time = time.time()

        if should_save or improved_this_iteration:
            exact_data_resume_supported = not (
                stochastic_training_pipeline_enabled(cfg)
                and int(cfg.exp.num_workers) > 0
            )
            checkpoint = {
                'iteration': iteration,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'wandb_run_id': wandb_run.id if wandb_run is not None else None,
                'model_arch': spec.arch,
                'model_mode': spec.mode,
                'selection': {
                    'dataset': selection_dataset,
                    'metric': selection_metric,
                    'mode': selection_mode,
                    'best_value': best_selection_value,
                    'best_iteration': best_selection_iteration,
                    'evaluations_without_improvement': evaluations_without_improvement,
                },
                'reproducibility': {
                    'format_version': REPRODUCIBILITY_FORMAT_VERSION,
                    'training_semantics': build_training_semantics_signature(cfg),
                    'random_seed': int(cfg.exp.random_seed),
                    'deterministic_cudnn': bool(
                        getattr(cfg.exp, 'deterministic_cudnn', True)
                    ),
                    'device_type': device.type,
                    'next_iteration': int(iteration + 1),
                    'exact_data_resume_supported': exact_data_resume_supported,
                    'rng_state': capture_rng_state(),
                    # Reconstruct logical progress instead of serializing the
                    # potentially prefetched live pointer.
                    'train_sampler': train_sampler.state_dict_for_batch(iteration + 1),
                    'train_dataset_rng': capture_dataset_rng_state(train_dataset),
                    'evaluation_data': evaluation_data_identities(eval_loaders),
                },
            }
            checkpoint.update(
                build_checkpoint_metadata(
                    cfg,
                    model_name=model_name,
                    model_arch=spec.arch,
                    model_mode=spec.mode,
                    model_class=f'{type(model).__module__}.{type(model).__qualname__}',
                )
            )
            # Every best checkpoint also gets an immutable numeric path. This
            # lets threshold selection freeze the exact weights instead of
            # referring forever to the mutable ``best.pth`` alias.
            checkpoint_path = os.path.join(checkpoints_dir, f'{iteration}_iteration.pth')
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to %s', checkpoint_path)
            if improved_this_iteration:
                best_checkpoint_path = os.path.join(checkpoints_dir, 'best.pth')
                temporary_best_path = f'{best_checkpoint_path}.tmp'
                shutil.copyfile(checkpoint_path, temporary_best_path)
                os.replace(temporary_best_path, best_checkpoint_path)
                logging.info('Best model saved to %s', best_checkpoint_path)

        iteration += 1
        if stop_after_iteration:
            logging.info(
                'Early stopping after %d validation evaluations without improvement; '
                'best iteration=%s, best value=%s',
                evaluations_without_improvement,
                best_selection_iteration,
                best_selection_value,
            )
            break
        if iteration >= cfg.exp.total_iteration:
            break

    if wandb_run is not None:
        wandb.finish()
    if tb_writer is not None:
        tb_writer.close()


if __name__ == '__main__':
    initialize(config_path='./', job_name='train', version_base=None)
    cfg = compose(config_name='config', overrides=sys.argv[1:])
    train(cfg)
