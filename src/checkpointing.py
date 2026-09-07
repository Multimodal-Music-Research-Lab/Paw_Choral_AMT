# SPDX-License-Identifier: Apache-2.0

"""Versioned checkpoint metadata and audited model-state loading."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from fnmatch import fnmatchcase
from io import BytesIO
from os import PathLike
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from canonical_union import canonical_union_semantics
from choral_targets import resolve_target_assignment


CHECKPOINT_SCHEMA_VERSION = 2
MODEL_INPUT_IDENTITY_SCHEMA_VERSION = 2
# Version 4 uses negative-label BCE for effective RP gradients and excludes
# ambiguous multi-pitch onset frames from OC trajectory pairs. Version 3 had
# the event-level OC objective and true silent-gap decay but neither hardening.
PRIOR_LOSS_SEMANTICS_VERSION = 4


class CheckpointFormatError(ValueError):
    """Raised when a file is not a supported model checkpoint."""


class UnsafeLegacyCheckpointError(CheckpointFormatError):
    """Raised when a checkpoint would require unsafe pickle deserialization."""


class CheckpointBehaviorError(RuntimeError):
    """Raised when parameter-compatible settings would change model behavior."""


class CheckpointIdentityError(CheckpointBehaviorError):
    """Raised when model or preprocessing semantics cannot be verified."""


@dataclass(frozen=True)
class CheckpointLoadReport:
    """Structured result of a model-state compatibility audit."""

    source_format: str
    schema_version: int | None
    deserialization_mode: str
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    allowlisted_missing_keys: tuple[str, ...]
    allowlisted_unexpected_keys: tuple[str, ...]
    missing_key_allowlist: tuple[str, ...]
    unexpected_key_allowlist: tuple[str, ...]

    @property
    def unapproved_missing_keys(self) -> tuple[str, ...]:
        approved = set(self.allowlisted_missing_keys)
        return tuple(key for key in self.missing_keys if key not in approved)

    @property
    def unapproved_unexpected_keys(self) -> tuple[str, ...]:
        approved = set(self.allowlisted_unexpected_keys)
        return tuple(key for key in self.unexpected_keys if key not in approved)

    @property
    def is_compatible(self) -> bool:
        return not self.unapproved_missing_keys and not self.unapproved_unexpected_keys

    @property
    def is_legacy(self) -> bool:
        return self.schema_version is None

    def as_dict(self) -> dict[str, Any]:
        return {
            'source_format': self.source_format,
            'schema_version': self.schema_version,
            'deserialization_mode': self.deserialization_mode,
            'missing_keys': list(self.missing_keys),
            'unexpected_keys': list(self.unexpected_keys),
            'allowlisted_missing_keys': list(self.allowlisted_missing_keys),
            'allowlisted_unexpected_keys': list(self.allowlisted_unexpected_keys),
            'missing_key_allowlist': list(self.missing_key_allowlist),
            'unexpected_key_allowlist': list(self.unexpected_key_allowlist),
            'is_compatible': self.is_compatible,
            'is_legacy': self.is_legacy,
        }


class CheckpointCompatibilityError(RuntimeError):
    """Raised when checkpoint/model key differences were not explicitly approved."""

    def __init__(self, report: CheckpointLoadReport):
        self.report = report
        message = (
            'Checkpoint model state is incompatible. '
            f'missing_keys={list(report.missing_keys)}, '
            f'unexpected_keys={list(report.unexpected_keys)}, '
            f'unapproved_missing_keys={list(report.unapproved_missing_keys)}, '
            f'unapproved_unexpected_keys={list(report.unapproved_unexpected_keys)}. '
            'No key mismatch is accepted by default; add only intentional legacy '
            'differences to the explicit checkpoint compatibility allowlists.'
        )
        super().__init__(message)


def _normalise_patterns(
    patterns: Sequence[str] | str | None,
    argument_name: str,
) -> tuple[str, ...]:
    if patterns is None:
        return ()
    if isinstance(patterns, str):
        patterns = (patterns,)
    normalised = tuple(str(pattern) for pattern in patterns)
    if any(not pattern for pattern in normalised):
        raise ValueError(f'{argument_name} cannot contain an empty pattern')
    return normalised


def _allowlisted(keys: tuple[str, ...], patterns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(key for key in keys if any(fnmatchcase(key, pattern) for pattern in patterns))


def _looks_like_raw_state_dict(payload: Mapping[str, Any]) -> bool:
    if not payload:
        return True
    if not all(isinstance(key, str) for key in payload):
        return False
    checkpoint_only_keys = {
        'iteration',
        'optimizer',
        'schema_version',
        'resolved_config',
        'model_system',
        'model_input_identity',
        'target_assignment',
        'wandb_run_id',
    }
    if checkpoint_only_keys.intersection(payload):
        return False
    return any(torch.is_tensor(value) for value in payload.values())


def extract_model_state_dict(payload: Any) -> tuple[Mapping[str, Any], str]:
    """Return the model state and identify its historical container format."""

    if not isinstance(payload, Mapping):
        raise CheckpointFormatError(
            f'Checkpoint must be a mapping, received {type(payload).__name__}'
        )

    if 'model' in payload:
        state_dict = payload['model']
        source_format = 'checkpoint:model'
    elif 'state_dict' in payload:
        state_dict = payload['state_dict']
        source_format = 'checkpoint:state_dict'
    elif _looks_like_raw_state_dict(payload):
        state_dict = payload
        source_format = 'raw_state_dict'
    else:
        raise CheckpointFormatError(
            "Checkpoint has neither a 'model'/'state_dict' mapping nor a raw state_dict"
        )

    if not isinstance(state_dict, Mapping):
        raise CheckpointFormatError(
            f'Checkpoint model state must be a mapping, received {type(state_dict).__name__}'
        )
    if not all(isinstance(key, str) for key in state_dict):
        raise CheckpointFormatError('Checkpoint model-state keys must all be strings')
    return state_dict, source_format


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint: (
        str
        | PathLike[str]
        | Mapping[str, Any]
        | bytes
        | bytearray
        | memoryview
    ),
    *,
    map_location: Any = None,
    allowed_missing_keys: Sequence[str] | str | None = (),
    allowed_unexpected_keys: Sequence[str] | str | None = (),
    allow_unsafe_legacy_load: bool = False,
) -> tuple[Mapping[str, Any], CheckpointLoadReport]:
    """Load a checkpoint while making every state-key difference auditable.

    Loading is exact by default. Internally, PyTorch's non-strict result is used
    only to obtain its authoritative missing/unexpected key lists; any mismatch
    not matched by an explicit allowlist pattern immediately raises
    :class:`CheckpointCompatibilityError`.

    File-backed checkpoints are first loaded with PyTorch's restricted
    ``weights_only=True`` unpickler. A historical file that contains unsupported
    pickle globals is rejected unless ``allow_unsafe_legacy_load`` is set
    explicitly; that opt-in is accepted only for checkpoints older than the
    current schema and may execute arbitrary code from the checkpoint.

    Both versioned training checkpoints and historical ``{'model': state}`` or
    raw ``state_dict`` files are supported when their serialized values are
    compatible with the selected deserialization mode. Serialized byte
    snapshots are accepted so callers can hash and deserialize exactly the
    same immutable content instead of reopening a mutable path.
    """

    if not isinstance(allow_unsafe_legacy_load, bool):
        raise TypeError('allow_unsafe_legacy_load must be a bool')

    is_serialized_snapshot = isinstance(
        checkpoint,
        (bytes, bytearray, memoryview),
    )
    if isinstance(checkpoint, (str, PathLike, Path)) or is_serialized_snapshot:
        load_source = (
            BytesIO(bytes(checkpoint)) if is_serialized_snapshot else checkpoint
        )
        try:
            payload = torch.load(
                load_source,
                map_location=map_location,
                weights_only=True,
            )
            deserialization_mode = 'weights_only'
        except pickle.UnpicklingError as exc:
            if not allow_unsafe_legacy_load:
                raise UnsafeLegacyCheckpointError(
                    'Checkpoint could not be loaded with PyTorch weights_only=True. '
                    'The file may contain historical NumPy/Python pickle objects or '
                    'may be corrupt. No unsafe fallback was attempted. If and only if '
                    'this is a trusted checkpoint predating schema v2, set '
                    'exp.allow_unsafe_legacy_checkpoint_load=true; unsafe pickle '
                    'loading can execute arbitrary code.'
                ) from exc
            if is_serialized_snapshot:
                load_source.seek(0)
            payload = torch.load(
                load_source,
                map_location=map_location,
                weights_only=False,
            )
            deserialization_mode = 'unsafe_legacy_opt_in'

            unsafe_schema = payload.get('schema_version') if isinstance(payload, Mapping) else None
            if unsafe_schema is not None:
                try:
                    unsafe_schema = int(unsafe_schema)
                except (TypeError, ValueError) as schema_exc:
                    raise CheckpointFormatError(
                        'Unsafe-loaded checkpoint schema_version must be an integer, '
                        f'received {unsafe_schema!r}'
                    ) from schema_exc
                if unsafe_schema >= CHECKPOINT_SCHEMA_VERSION:
                    raise UnsafeLegacyCheckpointError(
                        'A schema-v2-or-newer checkpoint required unsafe pickle '
                        'deserialization, violating the versioned checkpoint format. '
                        'Refusing to treat it as legacy; regenerate the checkpoint '
                        'with the current serializer.'
                    )
    else:
        payload = checkpoint
        deserialization_mode = 'in_memory'

    state_dict, source_format = extract_model_state_dict(payload)
    missing_patterns = _normalise_patterns(allowed_missing_keys, 'allowed_missing_keys')
    unexpected_patterns = _normalise_patterns(allowed_unexpected_keys, 'allowed_unexpected_keys')

    schema_version = payload.get('schema_version') if source_format != 'raw_state_dict' else None
    if schema_version is not None:
        try:
            schema_version = int(schema_version)
        except (TypeError, ValueError) as exc:
            raise CheckpointFormatError(
                f'Checkpoint schema_version must be an integer, received {schema_version!r}'
            ) from exc
        if schema_version > CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointFormatError(
                f'Checkpoint schema_version={schema_version} is newer than the '
                f'supported version {CHECKPOINT_SCHEMA_VERSION}; update the code '
                'before loading it.'
            )

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = tuple(incompatible.missing_keys)
    unexpected_keys = tuple(incompatible.unexpected_keys)

    report = CheckpointLoadReport(
        source_format=source_format,
        schema_version=schema_version,
        deserialization_mode=deserialization_mode,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        allowlisted_missing_keys=_allowlisted(missing_keys, missing_patterns),
        allowlisted_unexpected_keys=_allowlisted(unexpected_keys, unexpected_patterns),
        missing_key_allowlist=missing_patterns,
        unexpected_key_allowlist=unexpected_patterns,
    )
    if not report.is_compatible:
        raise CheckpointCompatibilityError(report)
    return payload, report


def format_checkpoint_load_report(report: CheckpointLoadReport) -> str:
    version = report.schema_version if report.schema_version is not None else 'legacy/unversioned'
    return (
        f'format={report.source_format}, schema_version={version}, '
        f'deserialization_mode={report.deserialization_mode}, '
        f'missing_keys={list(report.missing_keys)}, '
        f'unexpected_keys={list(report.unexpected_keys)}, '
        f'allowlisted_missing_keys={list(report.allowlisted_missing_keys)}, '
        f'allowlisted_unexpected_keys={list(report.allowlisted_unexpected_keys)}, '
        f'missing_key_allowlist={list(report.missing_key_allowlist)}, '
        f'unexpected_key_allowlist={list(report.unexpected_key_allowlist)}'
    )


def _cfg_get(cfg: Any, *path: str, default: Any = None) -> Any:
    current = cfg
    for key in path:
        if isinstance(current, Mapping):
            if key not in current:
                return default
            current = current[key]
        else:
            current = getattr(current, key, default)
            if current is default:
                return default
    return current


def checkpoint_compatibility_allowlists(cfg: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read opt-in compatibility patterns from ``exp`` without changing defaults.

    Hydra users can supply these absent-by-default fields with, for example,
    ``+exp.checkpoint_allow_missing_keys=[legacy_head.*]``.
    """

    missing = _cfg_get(cfg, 'exp', 'checkpoint_allow_missing_keys', default=())
    unexpected = _cfg_get(cfg, 'exp', 'checkpoint_allow_unexpected_keys', default=())
    return (
        _normalise_patterns(missing, 'exp.checkpoint_allow_missing_keys'),
        _normalise_patterns(unexpected, 'exp.checkpoint_allow_unexpected_keys'),
    )


def checkpoint_target_semantics(cfg: Any) -> dict[str, Any]:
    """Return the training-target settings that define a PawCT experiment row."""

    return {
        'method': resolve_target_assignment(cfg),
        'prior_loss_semantics_version': PRIOR_LOSS_SEMANTICS_VERSION,
        'preserve_known_part_labels': bool(
            _cfg_get(cfg, 'choral', 'preserve_known_part_labels', default=True)
        ),
        'range_prior_loss_weight': float(
            _cfg_get(cfg, 'choral', 'range_prior_loss_weight', default=0.0)
        ),
        'continuity_prior_loss_weight': float(
            _cfg_get(cfg, 'choral', 'continuity_prior_loss_weight', default=0.0)
        ),
        'evaluation_reference_assignment': str(
            _cfg_get(
                cfg,
                'choral',
                'evaluation_reference_assignment',
                default='part_name',
            )
        ),
        'voice_names': list(
            _cfg_get(cfg, 'choral', 'voice_names', default=['S', 'A', 'T', 'B'])
        ),
        'range_mins': list(
            _cfg_get(
                cfg,
                'choral',
                'voice_assignment_range_mins',
                default=[60, 55, 48, 40],
            )
        ),
        'range_maxs': list(
            _cfg_get(
                cfg,
                'choral',
                'voice_assignment_range_maxs',
                default=[88, 79, 72, 67],
            )
        ),
        'range_margin': float(
            _cfg_get(cfg, 'choral', 'voice_assignment_range_margin', default=2.0)
        ),
        'part_penalty': float(
            _cfg_get(cfg, 'choral', 'voice_assignment_part_penalty', default=2.0)
        ),
        'continuity_weight': float(
            _cfg_get(
                cfg,
                'choral',
                'voice_assignment_continuity_weight',
                default=0.35,
            )
        ),
        'overlap_penalty': float(
            _cfg_get(cfg, 'choral', 'voice_assignment_overlap_penalty', default=4.0)
        ),
        'mask_penalty': float(
            _cfg_get(cfg, 'choral', 'voice_assignment_mask_penalty', default=8.0)
        ),
        'oc_gap_decay_seconds': float(
            _cfg_get(cfg, 'choral', 'oc_gap_decay_seconds', default=2.0)
        ),
        'oc_overlap_tolerance_seconds': float(
            _cfg_get(cfg, 'choral', 'oc_overlap_tolerance_seconds', default=0.05)
        ),
    }


def checkpoint_data_target_semantics(cfg: Any) -> dict[str, Any]:
    """Return versioned data-target semantics independent of model family."""

    for field in ('train_set', 'test_set'):
        dataset_name = _cfg_get(cfg, 'dataset', field, default='')
        semantics = canonical_union_semantics(dataset_name)
        if semantics is not None:
            return {'canonical_union': semantics}
    return {'canonical_union': None}


def checkpoint_model_input_identity(
    cfg: Any,
    *,
    model_name: str | None = None,
    model_arch: str | None = None,
    model_mode: str | None = None,
) -> dict[str, Any]:
    """Return the model/output-coordinate and frontend identity for a run.

    The mel settings below are the effective constants used by
    ``LogMelExtractor``.  Recording derived values such as ``hop_length`` makes
    the identity independent of whether two configurations express the same
    frontend through integer or floating-point frame rates.
    """

    if model_name is None or model_arch is None or model_mode is None:
        # Lazy import avoids making checkpoint deserialization depend on the
        # heavier utility module unless runtime identity derivation is needed.
        from utilities import get_model_name, get_task_spec

        task_spec = get_task_spec(cfg)
        if model_name is None:
            model_name = get_model_name(cfg)
        if model_arch is None:
            model_arch = task_spec.arch
        if model_mode is None:
            model_mode = task_spec.mode

    audio_feature = str(
        _cfg_get(cfg, 'feature', 'audio_feature', default='logmel')
    ).strip().lower()
    sample_rate = int(_cfg_get(cfg, 'feature', 'sample_rate', default=16000))
    frames_per_second = float(
        _cfg_get(cfg, 'feature', 'frames_per_second', default=100)
    )
    if frames_per_second <= 0.0:
        raise ValueError('feature.frames_per_second must be positive')

    if audio_feature in {'logmel', 'mel'}:
        hop_length = int(sample_rate // frames_per_second)
        mel_bins = 229
        fmin = 30.0
        fmax = float(sample_rate // 2)
    elif audio_feature in {'bark', 'sone', 'ntot'}:
        hop_length = int(round(sample_rate / frames_per_second))
        mel_bins = None
        fmin = None
        fmax = None
    else:
        # MoSQITo-backed extractors own their internal framing. Keeping the
        # fields explicit prevents a missing key from being mistaken for an
        # unaudited legacy checkpoint.
        hop_length = None
        mel_bins = None
        fmin = None
        fmax = None

    choral_enabled = bool(
        _cfg_get(cfg, 'choral', 'enable', default=False)
    )
    model_identity = {
        'architecture': str(model_arch).strip().lower(),
        'task_mode': str(model_mode).strip().lower(),
        'model_name': str(model_name),
        'choral_enabled': choral_enabled,
    }
    if choral_enabled:
        # These settings change the forward function without changing any
        # state-dict key or tensor shape. They therefore must be part of the
        # identity rather than relying on strict parameter loading.
        model_identity.update(
            {
                'assignment_temperature': float(
                    _cfg_get(
                        cfg,
                        'choral',
                        'assignment_temperature',
                        default=1.0,
                    )
                ),
                'voice_interaction_heads': int(
                    _cfg_get(
                        cfg,
                        'choral',
                        'voice_interaction_heads',
                        default=4,
                    )
                ),
                'voice_interaction_dropout': float(
                    _cfg_get(
                        cfg,
                        'choral',
                        'voice_interaction_dropout',
                        default=0.1,
                    )
                ),
            }
        )

    return {
        'model': model_identity,
        'feature': {
            'audio_feature': audio_feature,
            'begin_note': int(
                _cfg_get(cfg, 'feature', 'begin_note', default=21)
            ),
            'classes_num': int(
                _cfg_get(cfg, 'feature', 'classes_num', default=88)
            ),
            'sample_rate': sample_rate,
            'frames_per_second': frames_per_second,
            'n_fft': int(_cfg_get(cfg, 'feature', 'fft_size', default=2048)),
            'hop_length': hop_length,
            'mel_bins': mel_bins,
            'fmin': fmin,
            'fmax': fmax,
            'segment_seconds': float(
                _cfg_get(cfg, 'feature', 'segment_seconds', default=10.0)
            ),
        },
    }


def _identity_mismatches(expected: Mapping[str, Any], actual: Mapping[str, Any]):
    mismatches = []

    def compare(expected_value, actual_value, path):
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping):
                mismatches.append(
                    f'{path}: checkpoint={actual_value!r}, runtime=mapping'
                )
                return
            for key, nested_expected in expected_value.items():
                nested_path = f'{path}.{key}' if path else str(key)
                if key not in actual_value:
                    mismatches.append(
                        f'{nested_path}: checkpoint=<missing>, '
                        f'runtime={nested_expected!r}'
                    )
                else:
                    compare(nested_expected, actual_value[key], nested_path)
            return
        if actual_value != expected_value:
            mismatches.append(
                f'{path}: checkpoint={actual_value!r}, runtime={expected_value!r}'
            )

    compare(expected, actual, '')
    return tuple(mismatches)


def validate_checkpoint_model_input_identity(
    cfg: Any,
    checkpoint: Mapping[str, Any],
) -> None:
    """Fail closed when checkpoint model/frontend semantics are unverified."""

    if not isinstance(checkpoint, Mapping):
        raise CheckpointIdentityError('Checkpoint identity requires a mapping payload')

    raw_schema_version = checkpoint.get('schema_version')
    try:
        schema_version = (
            int(raw_schema_version) if raw_schema_version is not None else None
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointIdentityError(
            f'Invalid checkpoint schema_version for identity audit: '
            f'{raw_schema_version!r}'
        ) from exc

    actual_identity = checkpoint.get('model_input_identity')
    if actual_identity is None:
        is_legacy_schema = (
            schema_version is None
            or schema_version < MODEL_INPUT_IDENTITY_SCHEMA_VERSION
        )
        allow_legacy = bool(
            _cfg_get(
                cfg,
                'exp',
                'allow_legacy_checkpoint_model_identity',
                default=False,
            )
        )
        if is_legacy_schema and allow_legacy:
            return
        if is_legacy_schema:
            raise CheckpointIdentityError(
                'Legacy checkpoint has no verifiable model/preprocessing identity. '
                'Set exp.allow_legacy_checkpoint_model_identity=true only for a '
                'clearly labelled historical run after manually confirming its '
                'model and frontend settings.'
            )
        raise CheckpointIdentityError(
            'Versioned checkpoint is missing required model_input_identity metadata'
        )
    if not isinstance(actual_identity, Mapping):
        raise CheckpointIdentityError(
            'Checkpoint model_input_identity metadata must be a mapping'
        )

    expected_identity = checkpoint_model_input_identity(cfg)
    mismatches = _identity_mismatches(expected_identity, actual_identity)
    if mismatches:
        raise CheckpointIdentityError(
            'Checkpoint model/preprocessing identity mismatch: '
            f'mismatches={list(mismatches)}. Model/input-coordinate and frontend '
            'settings must match exactly; generic behavior-mismatch overrides do '
            'not bypass this audit.'
        )


def validate_checkpoint_behavior(cfg: Any, checkpoint: Mapping[str, Any]) -> None:
    """Reject semantic runtime changes that strict state loading cannot see.

    Historical PawCT releases always multiplied note probabilities by the
    segment-level presence gate.  Because that switch has no parameters, a
    state-dict key audit alone cannot detect changing it at inference or on
    resume. Versioned checkpoints also record the training target-assignment
    method. Unknown target semantics and known mismatches require separate,
    explicit opt-ins so that a checkpoint cannot silently change RP/OC identity.
    """

    validate_checkpoint_model_input_identity(cfg, checkpoint)

    expected_data_semantics = checkpoint_data_target_semantics(cfg)
    if expected_data_semantics['canonical_union'] is not None:
        checkpoint_data_semantics = checkpoint.get('data_target_semantics')
        if checkpoint_data_semantics != expected_data_semantics and not bool(
            _cfg_get(
                cfg,
                'exp',
                'allow_legacy_canonical_union_semantics',
                default=False,
            )
        ):
            raise CheckpointBehaviorError(
                'Checkpoint does not prove the current canonical-union target '
                'semantics. Retrain from this release, or set '
                'exp.allow_legacy_canonical_union_semantics=true only for a '
                'clearly labelled historical diagnostic: '
                f'checkpoint={checkpoint_data_semantics!r}, '
                f'expected={expected_data_semantics!r}'
            )

    if not bool(_cfg_get(cfg, 'choral', 'enable', default=False)):
        return

    runtime_target_assignment = resolve_target_assignment(cfg)
    schema_version = checkpoint.get('schema_version') if isinstance(checkpoint, Mapping) else None
    target_metadata = checkpoint.get('target_assignment') if isinstance(checkpoint, Mapping) else None
    checkpoint_target_assignment = (
        target_metadata.get('method') if isinstance(target_metadata, Mapping) else None
    )
    runtime_preserve_known = bool(
        _cfg_get(cfg, 'choral', 'preserve_known_part_labels', default=True)
    )
    checkpoint_preserve_known = (
        target_metadata.get('preserve_known_part_labels')
        if isinstance(target_metadata, Mapping)
        else None
    )
    has_known_target_assignment = (
        schema_version is not None
        and isinstance(checkpoint_target_assignment, str)
        and bool(checkpoint_target_assignment.strip())
    )

    if not has_known_target_assignment:
        allow_unknown_target_assignment = bool(
            _cfg_get(
                cfg,
                'exp',
                'allow_unknown_checkpoint_target_assignment',
                default=False,
            )
        )
        if not allow_unknown_target_assignment:
            raise CheckpointBehaviorError(
                'Checkpoint target-assignment semantics are unknown: a versioned '
                "PawCT checkpoint with target_assignment.method is required. Set "
                'exp.allow_unknown_checkpoint_target_assignment=true only for a '
                'clearly labelled historical checkpoint whose training targets '
                'cannot be verified.'
            )
    elif checkpoint_target_assignment != runtime_target_assignment:
        allow_target_assignment_mismatch = bool(
            _cfg_get(
                cfg,
                'exp',
                'allow_checkpoint_target_assignment_mismatch',
                default=False,
            )
        )
        if not allow_target_assignment_mismatch:
            raise CheckpointBehaviorError(
                'Checkpoint target-assignment mismatch: '
                f'checkpoint={checkpoint_target_assignment!r}, '
                f'runtime={runtime_target_assignment!r}. Use the checkpoint '
                'training-target method for a faithful resume/evaluation. For a '
                'deliberate cross-target weight-reuse ablation, set '
                'exp.allow_checkpoint_target_assignment_mismatch=true and record '
                'that override with the results.'
            )
    elif (
        checkpoint_preserve_known is None
        and runtime_target_assignment != 'part_name'
        and not bool(
            _cfg_get(
                cfg,
                'exp',
                'allow_unknown_checkpoint_target_assignment',
                default=False,
            )
        )
    ):
        raise CheckpointBehaviorError(
            'Checkpoint preserve-known-label semantics are unknown. Set '
            'exp.allow_unknown_checkpoint_target_assignment=true only for a '
            'clearly labelled historical diagnostic.'
        )
    elif (
        checkpoint_preserve_known is not None
        and bool(checkpoint_preserve_known) != runtime_preserve_known
        and not bool(
            _cfg_get(
                cfg,
                'exp',
                'allow_checkpoint_target_assignment_mismatch',
                default=False,
            )
        )
    ):
        raise CheckpointBehaviorError(
            'Checkpoint preserve-known-label mismatch: '
            f'checkpoint={bool(checkpoint_preserve_known)!r}, '
            f'runtime={runtime_preserve_known!r}. Use the training setting or '
            'explicitly mark a cross-target weight-reuse ablation with '
            'exp.allow_checkpoint_target_assignment_mismatch=true.'
        )

    if has_known_target_assignment and runtime_target_assignment != 'part_name':
        expected_semantics = checkpoint_target_semantics(cfg)
        remaining_keys = tuple(
            key
            for key in expected_semantics
            if key not in {'method', 'preserve_known_part_labels'}
        )
        missing_semantics = [
            key for key in remaining_keys if key not in target_metadata
        ]
        mismatched_semantics = {
            key: (target_metadata.get(key), expected_semantics[key])
            for key in remaining_keys
            if key in target_metadata and target_metadata.get(key) != expected_semantics[key]
        }
        if missing_semantics and not bool(
            _cfg_get(
                cfg,
                'exp',
                'allow_unknown_checkpoint_target_assignment',
                default=False,
            )
        ):
            raise CheckpointBehaviorError(
                'Checkpoint target semantics are incomplete: '
                f'missing={missing_semantics}. Set '
                'exp.allow_unknown_checkpoint_target_assignment=true only for '
                'a clearly labelled historical diagnostic.'
            )
        if mismatched_semantics and not bool(
            _cfg_get(
                cfg,
                'exp',
                'allow_checkpoint_target_assignment_mismatch',
                default=False,
            )
        ):
            raise CheckpointBehaviorError(
                'Checkpoint target-semantics mismatch: '
                f'{mismatched_semantics}. Use the recorded training settings, or '
                'set exp.allow_checkpoint_target_assignment_mismatch=true only '
                'for a clearly labelled weight-reuse ablation.'
            )

    runtime_gate = bool(_cfg_get(cfg, 'choral', 'apply_presence_gate', default=True))
    runtime_voice_names = list(
        _cfg_get(cfg, 'choral', 'voice_names', default=['S', 'A', 'T', 'B'])
    )
    runtime_temperature = float(
        _cfg_get(cfg, 'choral', 'assignment_temperature', default=1.0)
    )
    model_system = checkpoint.get('model_system') if isinstance(checkpoint, Mapping) else None
    if isinstance(model_system, Mapping) and 'apply_presence_gate' in model_system:
        checkpoint_gate = bool(model_system['apply_presence_gate'])
        source = 'versioned checkpoint metadata'
        checkpoint_temperature = float(model_system.get('assignment_temperature', 1.0))
        target_metadata = checkpoint.get('target_assignment')
        checkpoint_voice_names = list(
            target_metadata.get('voice_names', ['S', 'A', 'T', 'B'])
            if isinstance(target_metadata, Mapping)
            else ['S', 'A', 'T', 'B']
        )
    else:
        checkpoint_gate = True
        checkpoint_temperature = 1.0
        checkpoint_voice_names = ['S', 'A', 'T', 'B']
        source = 'historical unversioned PawCT behavior'

    mismatches = []
    if runtime_gate != checkpoint_gate:
        mismatches.append(
            f'apply_presence_gate checkpoint={checkpoint_gate} runtime={runtime_gate}'
        )
    if runtime_voice_names != checkpoint_voice_names:
        mismatches.append(
            f'voice_names checkpoint={checkpoint_voice_names} runtime={runtime_voice_names}'
        )
    if runtime_temperature != checkpoint_temperature:
        mismatches.append(
            'assignment_temperature '
            f'checkpoint={checkpoint_temperature} runtime={runtime_temperature}'
        )
    if not mismatches:
        return

    allow_mismatch = bool(
        _cfg_get(
            cfg,
            'exp',
            'allow_checkpoint_behavior_mismatch',
            default=False,
        )
    )
    if allow_mismatch:
        return

    raise CheckpointBehaviorError(
        'Checkpoint parameter-free behavior is ambiguous or incompatible: '
        f'{source}; mismatches={mismatches}. Use the checkpoint values for a '
        'faithful resume/evaluation. For a deliberate weight-reuse ablation, '
        'set exp.allow_checkpoint_behavior_mismatch=true and record that '
        'override with the results.'
    )


def _plain_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _plain_value(value.value)
    if isinstance(value, PathLike):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _plain_value(asdict(value))
    if isinstance(value, SimpleNamespace):
        return _plain_value(vars(value))
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain_value(item) for item in value), key=repr)

    # Runtime-only objects (for example feature.augmentor) must not make the
    # checkpoint depend on pickling a process-local instance or memory address.
    value_type = type(value)
    return {'__python_type__': f'{value_type.__module__}.{value_type.__qualname__}'}


def resolve_config(cfg: Any) -> dict[str, Any]:
    """Resolve Hydra/OmegaConf interpolation into stable, pickle-safe values."""

    try:
        from omegaconf import OmegaConf
    except ImportError:
        OmegaConf = None

    if OmegaConf is not None and OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(
            cfg,
            resolve=True,
            throw_on_missing=True,
            enum_to_str=True,
        )
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)
    elif is_dataclass(cfg) and not isinstance(cfg, type):
        cfg = asdict(cfg)

    resolved = _plain_value(cfg)
    if not isinstance(resolved, dict):
        raise TypeError(f'Resolved config must be a mapping, received {type(resolved).__name__}')
    return resolved


def build_checkpoint_metadata(
    cfg: Any,
    *,
    model_name: str,
    model_arch: str,
    model_mode: str,
    model_class: str,
) -> dict[str, Any]:
    """Build the versioned, reproducibility-focused portion of a checkpoint."""

    resolved_config = resolve_config(cfg)
    choral_enabled = bool(_cfg_get(resolved_config, 'choral', 'enable', default=False))
    architecture = str(model_arch)
    if architecture == 'pawct' or (architecture == 'hpt' and choral_enabled):
        system = 'PawCT'
    elif architecture == 'pagct' or (architecture == 'hpt' and not choral_enabled):
        system = 'PagCT'
    else:
        system = architecture

    voice_names = _cfg_get(
        resolved_config,
        'choral',
        'voice_names',
        default=['S', 'A', 'T', 'B'],
    )
    configured_target_method = _cfg_get(
        resolved_config,
        'choral',
        'target_assignment',
        default=None,
    )
    legacy_target_method = _cfg_get(
        resolved_config,
        'choral',
        'voice_assignment_method',
        default=None,
    )
    raw_target_method = configured_target_method
    used_legacy_target_key = raw_target_method is None or not str(raw_target_method).strip()
    if used_legacy_target_key:
        raw_target_method = legacy_target_method
    if raw_target_method is None or not str(raw_target_method).strip():
        raw_target_method = 'part_name'
    target_aliases = {
        'rp': 'range_prior',
        'oc': 'ordered_continuity',
    }
    normalized_target_method = str(raw_target_method).strip().lower().replace('-', '_')
    target_method = target_aliases.get(normalized_target_method, normalized_target_method)
    if used_legacy_target_key and target_method in {
        'range_prior',
        'ordered_continuity',
        'range_masked_continuity',
    }:
        target_method = f'legacy_{target_method}'
    if not choral_enabled:
        target_method = 'part_agnostic'

    return {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'data_target_semantics': checkpoint_data_target_semantics(
            resolved_config
        ),
        'resolved_config': resolved_config,
        'model_input_identity': checkpoint_model_input_identity(
            resolved_config,
            model_name=model_name,
            model_arch=model_arch,
            model_mode=model_mode,
        ),
        'model_system': {
            'system': system,
            'model_name': str(model_name),
            'model_class': str(model_class),
            'architecture': architecture,
            'task_mode': str(model_mode),
            'choral_enabled': choral_enabled,
            'num_voices': int(
                _cfg_get(resolved_config, 'choral', 'num_voices', default=4)
            ),
            'assignment_module': str(
                _cfg_get(resolved_config, 'choral', 'assignment_module', default='heads')
            ),
            'voice_interaction_module': str(
                _cfg_get(
                    resolved_config,
                    'choral',
                    'voice_interaction_module',
                    default='none',
                )
            ),
            'voice_interaction_heads': int(
                _cfg_get(
                    resolved_config,
                    'choral',
                    'voice_interaction_heads',
                    default=4,
                )
            ),
            'voice_interaction_dropout': float(
                _cfg_get(
                    resolved_config,
                    'choral',
                    'voice_interaction_dropout',
                    default=0.1,
                )
            ),
            'apply_presence_gate': bool(
                _cfg_get(
                    resolved_config,
                    'choral',
                    'apply_presence_gate',
                    default=True,
                )
            ),
            'assignment_temperature': float(
                _cfg_get(
                    resolved_config,
                    'choral',
                    'assignment_temperature',
                    default=1.0,
                )
            ),
        },
        'target_assignment': {
            'enabled': choral_enabled,
            'method': target_method,
            'prior_loss_semantics_version': PRIOR_LOSS_SEMANTICS_VERSION,
            'configured_method': _plain_value(configured_target_method),
            'legacy_method': _plain_value(legacy_target_method),
            'preserve_known_part_labels': bool(
                _cfg_get(
                    resolved_config,
                    'choral',
                    'preserve_known_part_labels',
                    default=True,
                )
            ),
            'range_prior_loss_weight': float(
                _cfg_get(
                    resolved_config,
                    'choral',
                    'range_prior_loss_weight',
                    default=0.0,
                )
            ),
            'continuity_prior_loss_weight': float(
                _cfg_get(
                    resolved_config,
                    'choral',
                    'continuity_prior_loss_weight',
                    default=0.0,
                )
            ),
            'evaluation_reference_assignment': str(
                _cfg_get(
                    resolved_config,
                    'choral',
                    'evaluation_reference_assignment',
                    default='part_name',
                )
            ),
            'voice_names': _plain_value(voice_names),
            'range_mins': _plain_value(
                _cfg_get(
                    resolved_config,
                    'choral',
                    'voice_assignment_range_mins',
                    default=[60, 55, 48, 40],
                )
            ),
            'range_maxs': _plain_value(
                _cfg_get(
                    resolved_config,
                    'choral',
                    'voice_assignment_range_maxs',
                    default=[88, 79, 72, 67],
                )
            ),
            'range_margin': float(
                _cfg_get(
                    resolved_config,
                    'choral',
                    'voice_assignment_range_margin',
                    default=2.0,
                )
            ),
            'part_penalty': float(
                _cfg_get(
                    resolved_config,
                    'choral',
                    'voice_assignment_part_penalty',
                    default=2.0,
                )
            ),
            'continuity_weight': float(
                _cfg_get(
                    resolved_config,
                    'choral',
                    'voice_assignment_continuity_weight',
                    default=0.35,
                )
            ),
            'overlap_penalty': float(
                _cfg_get(
                    resolved_config,
                    'choral',
                    'voice_assignment_overlap_penalty',
                    default=4.0,
                )
            ),
            'mask_penalty': float(
                _cfg_get(
                    resolved_config,
                    'choral',
                    'voice_assignment_mask_penalty',
                    default=8.0,
                )
            ),
            'oc_gap_decay_seconds': float(
                _cfg_get(
                    resolved_config,
                    'choral',
                    'oc_gap_decay_seconds',
                    default=2.0,
                )
            ),
            'oc_overlap_tolerance_seconds': float(
                _cfg_get(
                    resolved_config,
                    'choral',
                    'oc_overlap_tolerance_seconds',
                    default=0.05,
                )
            ),
        },
    }
