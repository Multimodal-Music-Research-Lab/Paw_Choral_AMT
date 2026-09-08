# SPDX-License-Identifier: Apache-2.0

"""Integrity checks for inference probability artifacts.

Formal metrics must be tied to one packed split, one checkpoint, one inference
run, and the runtime semantics used to produce the outputs.  These helpers are
shared by merged (PagCT) and part-aware (PawCT) scorers so neither path can
silently accept a partial or stale directory.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping

import h5py
import numpy as np

from checkpointing import (
    checkpoint_data_target_semantics,
    checkpoint_compatibility_allowlists,
    checkpoint_model_input_identity,
    checkpoint_target_semantics,
)
from choral_targets import (
    resolve_reference_duration_policy,
    resolve_target_assignment,
)
from split_manifests import (
    configured_split_manifest_identity,
    select_split_hdf5_paths,
)
from utilities import (
    get_dataset_hdf5s_dir,
    get_filename,
    get_model_name,
    traverse_folder,
)


CANONICAL_VOICE_NAMES = ("S", "A", "T", "B")
_NOTE_PROBABILITY_OUTPUT_KEYS = frozenset({
    "frame_output",
    "onset_output",
    "offset_output",
    "reg_onset_output",
    "reg_offset_output",
})
_VOICE_PROBABILITY_OUTPUT_KEYS = frozenset({
    "voice_frame_output",
    "voice_onset_output",
    "voice_offset_output",
    "voice_assignment_output",
})
_PEDAL_PROBABILITY_OUTPUT_KEYS = frozenset({
    "pedal_frame_output",
    "pedal_onset_output",
    "pedal_offset_output",
    "reg_pedal_onset_output",
    "reg_pedal_offset_output",
})
_CLIP_PROBABILITY_OUTPUT_KEYS = frozenset({"voice_presence_output"})
_FINITE_LOGIT_OUTPUT_KEYS = frozenset({
    "voice_assignment_logits",
    "voice_presence_logits",
})
_PROBABILITY_OUTPUT_KEYS = (
    _NOTE_PROBABILITY_OUTPUT_KEYS
    | _VOICE_PROBABILITY_OUTPUT_KEYS
    | _PEDAL_PROBABILITY_OUTPUT_KEYS
    | _CLIP_PROBABILITY_OUTPUT_KEYS
)
_TEMPORAL_OUTPUT_KEYS = (
    _NOTE_PROBABILITY_OUTPUT_KEYS
    | _VOICE_PROBABILITY_OUTPUT_KEYS
    | _PEDAL_PROBABILITY_OUTPUT_KEYS
    | frozenset({"voice_assignment_logits"})
)


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Return a content-derived SHA-256 without trusting file timestamps."""

    absolute_path = os.path.realpath(path)
    digest = hashlib.sha256()
    with open(absolute_path, "rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_inference_checkpoint_path(checkpoints_dir: str, selector) -> str:
    """Resolve an iteration number or the validation-selected ``best`` file."""

    selector = str(selector).strip()
    if not selector:
        raise ValueError('exp.ckpt_iteration must be an iteration number or "best"')
    filename = "best.pth" if selector in {"best", "best.pth"} else f"{selector}_iteration.pth"
    return os.path.join(checkpoints_dir, filename)


def checkpoint_path_from_cfg(cfg) -> str:
    checkpoints_dir = os.path.join(cfg.exp.workspace, "checkpoints", get_model_name(cfg))
    return resolve_inference_checkpoint_path(checkpoints_dir, cfg.exp.ckpt_iteration)


def expected_split_hdf5_by_stem(cfg, eval_split=None):
    """Return the exact packed recordings defining an evaluation split."""

    split = str(
        getattr(cfg.dataset, "eval_split", "validation")
        if eval_split is None
        else eval_split
    )
    hdf5s_dir = get_dataset_hdf5s_dir(cfg, cfg.dataset.test_set)
    if not os.path.isdir(hdf5s_dir):
        raise FileNotFoundError(
            f"Missing packed evaluation directory required for manifest audit: {hdf5s_dir}"
        )

    _, hdf5_paths = traverse_folder(hdf5s_dir)
    selected = {}
    selected_paths = select_split_hdf5_paths(
        cfg,
        str(cfg.dataset.test_set),
        hdf5_paths,
        split,
    )
    for hdf5_path in selected_paths:
        stem = get_filename(hdf5_path)
        if stem in selected:
            raise RuntimeError(
                "Packed evaluation manifest contains duplicate recording stem "
                f"{stem!r}: {selected[stem]} and {hdf5_path}"
            )
        selected[stem] = hdf5_path

    if not selected:
        raise RuntimeError(f"No packed recordings found for split={split} in {hdf5s_dir}")
    return dict(sorted(selected.items()))


def expected_split_probability_stems(cfg, eval_split=None):
    return tuple(expected_split_hdf5_by_stem(cfg, eval_split))


def validate_probability_manifest(expected_stems, probability_stems):
    """Require exactly one probability file for every split recording."""

    expected = {str(stem) for stem in expected_stems}
    observed = {str(stem) for stem in probability_stems}
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={len(missing)} {missing[:10]}")
        if extra:
            details.append(f"stale_or_extra={len(extra)} {extra[:10]}")
        raise RuntimeError(
            "Probability manifest does not exactly match the packed evaluation split ("
            + "; ".join(details)
            + "). Finish inference into a consistent output directory before scoring."
        )


def probability_directory_from_cfg(cfg, eval_split=None) -> str:
    split = str(
        getattr(cfg.dataset, "eval_split", "validation")
        if eval_split is None
        else eval_split
    )
    return os.path.join(
        cfg.exp.workspace,
        "probs",
        cfg.dataset.test_set,
        split,
        get_model_name(cfg),
        f"{cfg.exp.ckpt_iteration}_iteration",
    )


def _expected_probability_frames(hdf5_path, *, sample_rate, frames_per_second):
    """Match inference's inclusive endpoint frame count for one recording."""

    sample_rate = float(sample_rate)
    frames_per_second = float(frames_per_second)
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError(f"sample_rate must be finite and positive, got {sample_rate!r}")
    if not np.isfinite(frames_per_second) or frames_per_second <= 0.0:
        raise ValueError(
            "frames_per_second must be finite and positive, got "
            f"{frames_per_second!r}"
        )

    with h5py.File(hdf5_path, "r") as hdf5_file:
        if "waveform" in hdf5_file:
            duration = hdf5_file["waveform"].shape[0] / sample_rate
        elif "duration" in hdf5_file.attrs:
            duration = float(hdf5_file.attrs["duration"])
        else:
            raise RuntimeError(
                "Packed evaluation recording has neither waveform samples nor a "
                f"duration attribute: {hdf5_path}"
            )
    if not np.isfinite(duration) or duration < 0.0:
        raise RuntimeError(
            f"Packed evaluation recording has invalid duration={duration!r}: {hdf5_path}"
        )
    return int(round(float(duration) * frames_per_second)) + 1


class ProbabilityArtifactValidator:
    """Validate a complete directory and every artifact loaded from it."""

    def __init__(self, cfg, probs_dir=None, eval_split=None):
        self.cfg = cfg
        self.model_name = get_model_name(cfg)
        self.eval_split = str(
            getattr(cfg.dataset, "eval_split", "validation")
            if eval_split is None
            else eval_split
        )
        if self.eval_split not in {"validation", "test"}:
            raise ValueError("dataset.eval_split must be 'validation' or 'test'")
        self.probs_dir = probs_dir or probability_directory_from_cfg(cfg, self.eval_split)
        if not os.path.isdir(self.probs_dir):
            raise FileNotFoundError(f"Missing probs dir: {self.probs_dir}")

        self.hdf5_by_stem = expected_split_hdf5_by_stem(cfg, self.eval_split)
        self.expected_split_manifest_identity = configured_split_manifest_identity(
            cfg,
            str(cfg.dataset.test_set),
        )
        self.probability_names = tuple(
            sorted(name for name in os.listdir(self.probs_dir) if name.endswith(".pkl"))
        )
        validate_probability_manifest(
            self.hdf5_by_stem,
            (os.path.splitext(name)[0] for name in self.probability_names),
        )

        self.require_provenance = bool(
            getattr(cfg.exp, "require_probability_provenance", True)
        )
        self.choral_enabled = bool(getattr(cfg.choral, "enable", False))
        self.expected_target_assignment = (
            resolve_target_assignment(cfg) if self.choral_enabled else "part_agnostic"
        )
        self.expected_target_semantics = (
            checkpoint_target_semantics(cfg)
            if self.choral_enabled
            else {"method": "part_agnostic"}
        )
        self.expected_data_target_semantics = checkpoint_data_target_semantics(cfg)
        self.expected_model_input_identity = checkpoint_model_input_identity(cfg)
        self.expected_voice_names = list(
            getattr(cfg.choral, "voice_names", CANONICAL_VOICE_NAMES)
        )
        if self.choral_enabled and tuple(self.expected_voice_names) != CANONICAL_VOICE_NAMES:
            raise ValueError(
                "Formal PawCT scoring requires choral.voice_names=['S','A','T','B']; "
                f"received {self.expected_voice_names!r}"
            )

        self.checkpoint_path = checkpoint_path_from_cfg(cfg)
        self.expected_checkpoint_identity = None
        if os.path.isfile(self.checkpoint_path):
            self.expected_checkpoint_identity = {
                "filename": os.path.basename(self.checkpoint_path),
                "sha256": sha256_file(self.checkpoint_path),
            }
        elif self.require_provenance:
            raise FileNotFoundError(
                "Cannot bind probability artifacts to the requested checkpoint because it "
                f"does not exist: {self.checkpoint_path}"
            )

        self._provenance_presence = None
        self._probability_checkpoint_identity = None
        self._inference_run_id = None

    def hdf5_path_for_probability(self, prob_path):
        stem = get_filename(prob_path)
        try:
            return self.hdf5_by_stem[stem]
        except KeyError as exc:
            raise RuntimeError(
                f"Probability artifact is outside the validated split manifest: {prob_path}"
            ) from exc

    def identity_summary(self):
        """Return the identity proven while loading the current directory."""

        if self._provenance_presence == "present" and self._probability_checkpoint_identity:
            filename, sha256, iteration = self._probability_checkpoint_identity
            return {
                "checkpoint_filename": filename,
                "checkpoint_sha256": sha256,
                "checkpoint_iteration": iteration,
                "inference_run_id": self._inference_run_id,
            }
        return {
            "checkpoint_filename": None,
            "checkpoint_sha256": None,
            "checkpoint_iteration": None,
            "inference_run_id": None,
        }

    def _validate_checkpoint_load_report(self, provenance, prob_path):
        report = provenance.get("checkpoint_load_report")
        if not isinstance(report, Mapping):
            raise RuntimeError(f"Missing checkpoint load audit in {prob_path}")
        if report.get("is_compatible") is not True:
            raise RuntimeError(f"Checkpoint load audit is not compatible in {prob_path}")

        deserialization_mode = report.get("deserialization_mode")
        if deserialization_mode not in {"weights_only", "unsafe_legacy_opt_in"}:
            raise RuntimeError(
                f"Checkpoint deserialization audit is missing or invalid in {prob_path}"
            )
        runtime_behavior = provenance.get("runtime_model_behavior")
        unsafe_opt_in = (
            runtime_behavior.get("allow_unsafe_legacy_checkpoint_load")
            if isinstance(runtime_behavior, Mapping)
            else None
        )
        if deserialization_mode == "unsafe_legacy_opt_in" and unsafe_opt_in is not True:
            raise RuntimeError(
                f"Unsafe checkpoint load is not paired with its explicit opt-in in {prob_path}"
            )
        checkpoint_identity = provenance.get("checkpoint_identity")
        checkpoint_schema = (
            checkpoint_identity.get("schema_version")
            if isinstance(checkpoint_identity, Mapping)
            else None
        )
        if (
            deserialization_mode == "unsafe_legacy_opt_in"
            and isinstance(checkpoint_schema, int)
            and not isinstance(checkpoint_schema, bool)
            and checkpoint_schema >= 2
        ):
            raise RuntimeError(
                f"Schema-v2 checkpoint cannot use unsafe deserialization in {prob_path}"
            )

        expected_missing, expected_unexpected = checkpoint_compatibility_allowlists(self.cfg)
        if tuple(report.get("missing_key_allowlist", ())) != tuple(expected_missing):
            raise RuntimeError(f"Checkpoint missing-key allowlist mismatch in {prob_path}")
        if tuple(report.get("unexpected_key_allowlist", ())) != tuple(expected_unexpected):
            raise RuntimeError(f"Checkpoint unexpected-key allowlist mismatch in {prob_path}")

    def _validate_runtime_behavior(self, provenance, prob_path):
        behavior = provenance.get("runtime_model_behavior")
        if not isinstance(behavior, Mapping):
            raise RuntimeError(f"Missing runtime model behavior in {prob_path}")
        expected = {
            "num_voices": int(getattr(self.cfg.choral, "num_voices", 4)),
            "use_presence_head": bool(getattr(self.cfg.choral, "use_presence_head", True)),
            "apply_presence_gate": bool(getattr(self.cfg.choral, "apply_presence_gate", True)),
            "assignment_module": str(
                getattr(self.cfg.choral, "assignment_module", "heads")
            ).strip(),
            "assignment_temperature": float(
                getattr(self.cfg.choral, "assignment_temperature", 1.0)
            ),
            "assignment_hidden_channels": int(
                getattr(self.cfg.choral, "assignment_hidden_channels", 64)
            ),
            "assignment_rnn_hidden_size": int(
                getattr(self.cfg.choral, "assignment_rnn_hidden_size", 128)
            ),
            "voice_interaction_module": str(
                getattr(self.cfg.choral, "voice_interaction_module", "none")
            ).strip(),
            "voice_interaction_dim": int(
                getattr(self.cfg.choral, "voice_interaction_dim", 256)
            ),
            "voice_interaction_heads": int(
                getattr(self.cfg.choral, "voice_interaction_heads", 4)
            ),
            "voice_interaction_layers": int(
                getattr(self.cfg.choral, "voice_interaction_layers", 1)
            ),
            "voice_interaction_dropout": float(
                getattr(self.cfg.choral, "voice_interaction_dropout", 0.1)
            ),
            "voice_names": self.expected_voice_names,
            "allow_checkpoint_behavior_mismatch": bool(
                getattr(self.cfg.exp, "allow_checkpoint_behavior_mismatch", False)
            ),
            "allow_unknown_checkpoint_target_assignment": bool(
                getattr(self.cfg.exp, "allow_unknown_checkpoint_target_assignment", False)
            ),
            "allow_checkpoint_target_assignment_mismatch": bool(
                getattr(self.cfg.exp, "allow_checkpoint_target_assignment_mismatch", False)
            ),
            "allow_legacy_checkpoint_model_identity": bool(
                getattr(self.cfg.exp, "allow_legacy_checkpoint_model_identity", False)
            ),
            "allow_unsafe_legacy_checkpoint_load": bool(
                getattr(self.cfg.exp, "allow_unsafe_legacy_checkpoint_load", False)
            ),
            "allow_legacy_canonical_union_semantics": bool(
                getattr(
                    self.cfg.exp,
                    "allow_legacy_canonical_union_semantics",
                    False,
                )
            ),
        }
        for key, expected_value in expected.items():
            if behavior.get(key) != expected_value:
                raise RuntimeError(
                    f"Probability runtime behavior mismatch for {key} in {prob_path}: "
                    f"expected {expected_value!r}, found {behavior.get(key)!r}"
                )

    def _validate_probability_payload(self, total_dict, prob_path, hdf5_path=None):
        output_keys = (
            _PROBABILITY_OUTPUT_KEYS | _FINITE_LOGIT_OUTPUT_KEYS
        ).intersection(total_dict)
        if not output_keys:
            return

        temporal_keys = _TEMPORAL_OUTPUT_KEYS.intersection(output_keys)
        expected_frames = None
        if temporal_keys:
            hdf5_path = hdf5_path or self.hdf5_path_for_probability(prob_path)
            expected_frames = _expected_probability_frames(
                hdf5_path,
                sample_rate=self.cfg.feature.sample_rate,
                frames_per_second=self.cfg.feature.frames_per_second,
            )

        classes_num = int(self.cfg.feature.classes_num)
        num_voices = int(getattr(self.cfg.choral, "num_voices", 4))
        for key in sorted(output_keys):
            try:
                values = np.asarray(total_dict[key])
            except Exception as exc:
                raise RuntimeError(
                    f"Probability payload {key} is not array-like in {prob_path}"
                ) from exc
            if (
                not np.issubdtype(values.dtype, np.number)
                or np.issubdtype(values.dtype, np.complexfloating)
            ):
                raise RuntimeError(
                    f"Probability payload {key} must be a real numeric array in {prob_path}; "
                    f"found dtype={values.dtype}"
                )

            if key in _NOTE_PROBABILITY_OUTPUT_KEYS:
                expected_shape = (expected_frames, classes_num)
            elif key in _VOICE_PROBABILITY_OUTPUT_KEYS or key == "voice_assignment_logits":
                expected_shape = (expected_frames, num_voices, classes_num)
            elif key in _PEDAL_PROBABILITY_OUTPUT_KEYS:
                expected_shape = (expected_frames, 1)
            else:
                expected_shape = (num_voices,)
            if values.shape != expected_shape:
                raise RuntimeError(
                    f"Probability payload shape mismatch for {key} in {prob_path}: "
                    f"expected {expected_shape}, found {values.shape}"
                )
            if not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"Probability payload {key} contains non-finite values in {prob_path}"
                )
            if key in _PROBABILITY_OUTPUT_KEYS and (
                np.any(values < 0.0) or np.any(values > 1.0)
            ):
                raise RuntimeError(
                    f"Probability payload {key} contains values outside [0, 1] in {prob_path}"
                )

    def validate(self, total_dict, prob_path, *, hdf5_path=None, reference_path=None):
        if not isinstance(total_dict, Mapping):
            raise TypeError(f"Probability artifact is not a mapping: {prob_path}")

        self._validate_probability_payload(
            total_dict,
            prob_path,
            hdf5_path=hdf5_path,
        )

        provenance = total_dict.get("provenance")
        presence = "present" if isinstance(provenance, Mapping) else "missing"
        if provenance is not None and not isinstance(provenance, Mapping):
            raise RuntimeError(f"Malformed probability provenance in {prob_path}")
        if self._provenance_presence is None:
            self._provenance_presence = presence
        elif self._provenance_presence != presence:
            raise RuntimeError(
                "Probability directory mixes artifacts with and without provenance; "
                f"encountered {prob_path}"
            )

        if provenance is None:
            if self.require_provenance:
                raise RuntimeError(
                    f"Missing probability provenance in {prob_path}. Re-run inference, or set "
                    "exp.require_probability_provenance=false only for a clearly labelled "
                    "historical diagnostic."
                )
            return

        expected_fields = {
            "dataset_name": str(self.cfg.dataset.test_set),
            "model_name": self.model_name,
            "evaluation_split": self.eval_split,
            "evaluation_reference_assignment": "part_name",
            "evaluation_reference_duration_policy": (
                resolve_reference_duration_policy(self.cfg)
            ),
        }
        for key, expected_value in expected_fields.items():
            if provenance.get(key) != expected_value:
                raise RuntimeError(
                    f"Probability provenance mismatch for {key} in {prob_path}: "
                    f"expected {expected_value!r}, found {provenance.get(key)!r}"
                )

        expected_split_manifest = getattr(
            self,
            "expected_split_manifest_identity",
            configured_split_manifest_identity(
                self.cfg,
                str(self.cfg.dataset.test_set),
            ),
        )
        if provenance.get("split_manifest") != expected_split_manifest:
            raise RuntimeError(
                f"Probability split-manifest mismatch in {prob_path}: expected "
                f"{expected_split_manifest!r}, found "
                f"{provenance.get('split_manifest')!r}"
            )

        expected_union = self.expected_data_target_semantics.get(
            "canonical_union"
        )
        if expected_union is not None:
            data_semantics = provenance.get("data_target_semantics")
            if not isinstance(data_semantics, Mapping):
                raise RuntimeError(
                    f"Missing canonical-union provenance in {prob_path}"
                )
            runtime_data_semantics = data_semantics.get("runtime")
            if runtime_data_semantics != self.expected_data_target_semantics:
                raise RuntimeError(
                    f"Probability canonical-union runtime mismatch in {prob_path}: "
                    f"expected {self.expected_data_target_semantics!r}, found "
                    f"{runtime_data_semantics!r}"
                )
            checkpoint_data_semantics = data_semantics.get("checkpoint")
            allow_legacy_union = bool(
                getattr(
                    self.cfg.exp,
                    "allow_legacy_canonical_union_semantics",
                    False,
                )
            )
            if (
                checkpoint_data_semantics != self.expected_data_target_semantics
                and not allow_legacy_union
            ):
                raise RuntimeError(
                    f"Checkpoint canonical-union semantics are missing or stale in "
                    f"{prob_path}: expected {self.expected_data_target_semantics!r}, "
                    f"found {checkpoint_data_semantics!r}"
                )

        run_id = provenance.get("inference_run_id")
        if self.require_provenance and (not isinstance(run_id, str) or not run_id):
            raise RuntimeError(f"Missing inference run identity in {prob_path}")
        if self._inference_run_id is None:
            self._inference_run_id = run_id
        elif self._inference_run_id != run_id:
            raise RuntimeError(
                "Probability directory mixes outputs from different inference runs; "
                f"encountered {prob_path}"
            )

        checkpoint_identity = provenance.get("checkpoint_identity")
        if not isinstance(checkpoint_identity, Mapping):
            raise RuntimeError(f"Missing checkpoint identity in {prob_path}")
        sha256 = checkpoint_identity.get("sha256")
        if not (
            isinstance(sha256, str)
            and len(sha256) == 64
            and all(character in "0123456789abcdefABCDEF" for character in sha256)
        ):
            raise RuntimeError(f"Invalid checkpoint SHA-256 identity in {prob_path}")
        identity = (
            checkpoint_identity.get("filename"),
            sha256.lower(),
            checkpoint_identity.get("iteration"),
        )
        if self._probability_checkpoint_identity is None:
            self._probability_checkpoint_identity = identity
        elif self._probability_checkpoint_identity != identity:
            raise RuntimeError(
                "Probability directory mixes outputs from different checkpoints; "
                f"encountered {prob_path}"
            )

        if self.expected_checkpoint_identity is not None:
            for key in ("filename", "sha256"):
                expected_value = self.expected_checkpoint_identity[key]
                actual_value = checkpoint_identity.get(key)
                if key == "sha256" and isinstance(actual_value, str):
                    actual_value = actual_value.lower()
                if actual_value != expected_value:
                    raise RuntimeError(
                        f"Probability checkpoint {key} does not match the requested checkpoint "
                        f"in {prob_path}: expected {expected_value!r}, found {actual_value!r}"
                    )

        actual_iteration = checkpoint_identity.get("iteration")
        if self.require_provenance:
            if not isinstance(actual_iteration, int) or isinstance(actual_iteration, bool):
                raise RuntimeError(f"Missing actual checkpoint iteration in {prob_path}")
            selector = str(self.cfg.exp.ckpt_iteration).strip()
            if selector not in {"best", "best.pth"}:
                try:
                    expected_iteration = int(selector)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid numeric checkpoint selector: {selector!r}"
                    ) from exc
                if actual_iteration != expected_iteration:
                    raise RuntimeError(
                        f"Probability checkpoint iteration mismatch in {prob_path}: expected "
                        f"{expected_iteration}, found {actual_iteration}"
                    )

        target_assignment = provenance.get("target_assignment")
        if not isinstance(target_assignment, Mapping):
            raise RuntimeError(f"Missing target-assignment provenance in {prob_path}")
        if target_assignment.get("runtime_training_method") != self.expected_target_assignment:
            raise RuntimeError(
                f"Probability target-assignment mismatch in {prob_path}: expected "
                f"{self.expected_target_assignment!r}, found "
                f"{target_assignment.get('runtime_training_method')!r}"
            )
        runtime_semantics = target_assignment.get("runtime_semantics")
        if runtime_semantics != self.expected_target_semantics:
            raise RuntimeError(
                f"Probability target-semantics mismatch in {prob_path}: expected "
                f"{self.expected_target_semantics!r}, found {runtime_semantics!r}"
            )
        expected_preserve_known = bool(
            getattr(self.cfg.choral, "preserve_known_part_labels", True)
        )
        if target_assignment.get("preserve_known_part_labels") != expected_preserve_known:
            raise RuntimeError(
                f"Probability preserve-known-label mismatch in {prob_path}: expected "
                f"{expected_preserve_known!r}, found "
                f"{target_assignment.get('preserve_known_part_labels')!r}"
            )
        expected_model_assignment = str(
            getattr(self.cfg.choral, "assignment_module", "heads")
        ).strip()
        if target_assignment.get("model_assignment_module") != expected_model_assignment:
            raise RuntimeError(
                f"Probability model-assignment mismatch in {prob_path}: expected "
                f"{expected_model_assignment!r}, found "
                f"{target_assignment.get('model_assignment_module')!r}"
            )
        checkpoint_target = target_assignment.get("checkpoint_metadata")
        allow_unknown = bool(
            getattr(self.cfg.exp, "allow_unknown_checkpoint_target_assignment", False)
        )
        allow_mismatch = bool(
            getattr(self.cfg.exp, "allow_checkpoint_target_assignment_mismatch", False)
        )
        checkpoint_method = (
            checkpoint_target.get("method")
            if isinstance(checkpoint_target, Mapping)
            else None
        )
        checkpoint_preserve_known = (
            checkpoint_target.get("preserve_known_part_labels")
            if isinstance(checkpoint_target, Mapping)
            else None
        )
        if self.choral_enabled and checkpoint_method is None and not allow_unknown:
            raise RuntimeError(f"Checkpoint target assignment is unknown in {prob_path}")
        if (
            checkpoint_method is not None
            and checkpoint_method != self.expected_target_assignment
            and not allow_mismatch
        ):
            raise RuntimeError(
                f"Checkpoint target assignment mismatch in {prob_path}: expected "
                f"{self.expected_target_assignment!r}, found {checkpoint_method!r}"
            )
        if (
            self.choral_enabled
            and self.expected_target_assignment != "part_name"
            and checkpoint_preserve_known is None
            and not allow_unknown
        ):
            raise RuntimeError(
                f"Checkpoint preserve-known-label semantics are unknown in {prob_path}"
            )
        if (
            self.choral_enabled
            and checkpoint_preserve_known is not None
            and bool(checkpoint_preserve_known) != expected_preserve_known
            and not allow_mismatch
        ):
            raise RuntimeError(
                f"Checkpoint preserve-known-label mismatch in {prob_path}: expected "
                f"{expected_preserve_known!r}, found {bool(checkpoint_preserve_known)!r}"
            )
        if self.choral_enabled and isinstance(checkpoint_target, Mapping):
            semantic_keys = tuple(
                key
                for key in self.expected_target_semantics
                if key not in {"method", "preserve_known_part_labels"}
            )
            missing_semantics = [
                key for key in semantic_keys if key not in checkpoint_target
            ]
            mismatched_semantics = {
                key: (checkpoint_target.get(key), self.expected_target_semantics[key])
                for key in semantic_keys
                if key in checkpoint_target
                and checkpoint_target.get(key) != self.expected_target_semantics[key]
            }
            if missing_semantics and not allow_unknown:
                raise RuntimeError(
                    f"Checkpoint target semantics are incomplete in {prob_path}: "
                    f"missing={missing_semantics}"
                )
            if mismatched_semantics and not allow_mismatch:
                raise RuntimeError(
                    f"Checkpoint target-semantics mismatch in {prob_path}: "
                    f"{mismatched_semantics}"
                )

        checkpoint_input_identity = provenance.get("checkpoint_model_input_identity")
        allow_legacy_identity = bool(
            getattr(self.cfg.exp, "allow_legacy_checkpoint_model_identity", False)
        )
        if checkpoint_input_identity is None:
            if not allow_legacy_identity:
                raise RuntimeError(
                    f"Checkpoint model/input identity is missing in {prob_path}"
                )
        elif checkpoint_input_identity != self.expected_model_input_identity:
            raise RuntimeError(
                f"Checkpoint model/input identity mismatch in {prob_path}: expected "
                f"{self.expected_model_input_identity!r}, found "
                f"{checkpoint_input_identity!r}"
            )

        self._validate_runtime_behavior(provenance, prob_path)
        self._validate_checkpoint_load_report(provenance, prob_path)

        hdf5_path = hdf5_path or self.hdf5_path_for_probability(prob_path)
        source_artifacts = provenance.get("source_artifacts")
        if not isinstance(source_artifacts, Mapping):
            raise RuntimeError(f"Missing source artifact identity in {prob_path}")
        expected_stem = get_filename(prob_path)
        if source_artifacts.get("recording_stem") != expected_stem:
            raise RuntimeError(f"Probability recording identity mismatch in {prob_path}")
        expected_hdf5_hash = sha256_file(hdf5_path)
        if source_artifacts.get("hdf5_sha256") != expected_hdf5_hash:
            raise RuntimeError(f"Packed HDF5 content changed after inference for {prob_path}")
        if reference_path is not None:
            if not os.path.isfile(reference_path):
                raise FileNotFoundError(f"Missing SATB reference: {reference_path}")
            expected_reference_hash = sha256_file(reference_path)
            if source_artifacts.get("reference_note_sha256") != expected_reference_hash:
                raise RuntimeError(f"SATB reference content changed after inference for {prob_path}")
