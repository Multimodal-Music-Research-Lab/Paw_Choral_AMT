# SPDX-License-Identifier: Apache-2.0

"""Strict split-manifest selection for pre-packed datasets.

Packed YouChorale files retain the official split in their ``split`` HDF5
attribute.  A composition-disjoint experiment must therefore select files by
an explicit external manifest rather than silently continuing to use that
attribute.  This module is shared by training, inference, and scoring so all
three stages use exactly the same recording set.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence

import h5py

from utilities import decode_hdf5_attr, get_filename


SPLIT_FILES = {
    "train": "train.json",
    "validation": "valid.json",
    "test": "test.json",
}
SPLIT_MANIFEST_FORMAT_VERSION = 1


def _configured_manifest_dir(cfg, dataset_name: str) -> str | None:
    """Return the explicit split directory for a supported dataset, if any."""

    if str(dataset_name) != "youchorale":
        return None
    dataset_cfg = (
        cfg.get("dataset") if isinstance(cfg, Mapping) else getattr(cfg, "dataset", None)
    )
    value = (
        dataset_cfg.get("youchorale_split_dir")
        if isinstance(dataset_cfg, Mapping)
        else getattr(dataset_cfg, "youchorale_split_dir", None)
    )
    if value is None or not str(value).strip():
        return None
    return os.path.abspath(os.path.expanduser(str(value).strip()))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def load_configured_split_manifest(cfg, dataset_name: str) -> dict | None:
    """Load and validate one external train/validation/test protocol.

    The returned identity is independent of the local absolute path.  All
    manifests must be non-empty lists of unique filename stems, and the three
    sets must be pairwise disjoint.
    """

    manifest_dir = _configured_manifest_dir(cfg, dataset_name)
    if manifest_dir is None:
        return None
    if not os.path.isdir(manifest_dir):
        raise FileNotFoundError(
            "dataset.youchorale_split_dir is not a directory: "
            f"{manifest_dir}"
        )

    splits = {}
    file_sha256 = {}
    for split_name, filename in SPLIT_FILES.items():
        path = os.path.join(manifest_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing required {split_name} split manifest: {path}"
            )
        with open(path, "rb") as manifest_file:
            payload = manifest_file.read()
        file_sha256[filename] = _sha256_bytes(payload)
        try:
            values = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid UTF-8 JSON split manifest: {path}") from exc
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Split manifest must be a non-empty JSON list: {path}"
            )

        stems = []
        seen = set()
        for index, stem in enumerate(values):
            if not isinstance(stem, str) or not stem:
                raise ValueError(
                    f"Split manifest entry {index} must be a non-empty string: {path}"
                )
            if stem != os.path.basename(stem) or stem in {".", ".."}:
                raise ValueError(
                    f"Split manifest entries must be filename stems, got {stem!r}: {path}"
                )
            if stem in seen:
                raise ValueError(
                    f"Duplicate recording stem {stem!r} in split manifest: {path}"
                )
            seen.add(stem)
            stems.append(stem)
        splits[split_name] = tuple(stems)

    owners = {}
    overlaps = {}
    for split_name, stems in splits.items():
        for stem in stems:
            previous = owners.setdefault(stem, split_name)
            if previous != split_name:
                overlaps.setdefault(stem, {previous}).add(split_name)
    if overlaps:
        preview = [
            f"{stem}:{'/'.join(sorted(split_names))}"
            for stem, split_names in sorted(overlaps.items())[:10]
        ]
        raise ValueError(
            "External split manifests are not pairwise disjoint: "
            f"{preview}"
        )

    canonical_splits = {
        split_name: list(splits[split_name]) for split_name in SPLIT_FILES
    }
    identity = {
        "format_version": SPLIT_MANIFEST_FORMAT_VERSION,
        "dataset_name": str(dataset_name),
        "hash_semantics": {
            "content_sha256": (
                "SHA256 of compact UTF-8 canonical JSON for the three split "
                "arrays; object keys sorted and manifest order preserved"
            ),
            "file_sha256": "SHA256 of exact manifest file bytes",
            "ordered_recording_ids_sha256": (
                "SHA256 of a compact UTF-8 JSON array in manifest order"
            ),
        },
        "content_sha256": _canonical_json_sha256(canonical_splits),
        "file_sha256": dict(sorted(file_sha256.items())),
        "recording_counts": {
            split_name: len(splits[split_name]) for split_name in SPLIT_FILES
        },
        "ordered_recording_ids_sha256": {
            split_name: _canonical_json_sha256(list(splits[split_name]))
            for split_name in SPLIT_FILES
        },
        "total_recordings": len(owners),
    }
    return {
        "directory": manifest_dir,
        "splits": splits,
        "identity": identity,
    }


def configured_split_manifest_identity(cfg, dataset_name: str) -> dict | None:
    """Return a path-independent identity for the configured protocol."""

    manifest = load_configured_split_manifest(cfg, dataset_name)
    return None if manifest is None else manifest["identity"]


def _hdf5_paths_by_stem(hdf5_paths: Sequence[str]) -> dict[str, str]:
    paths_by_stem = {}
    for hdf5_path in sorted(str(path) for path in hdf5_paths):
        stem = get_filename(hdf5_path)
        if stem in paths_by_stem:
            raise RuntimeError(
                "Packed dataset contains duplicate recording stem "
                f"{stem!r}: {paths_by_stem[stem]} and {hdf5_path}"
            )
        paths_by_stem[stem] = hdf5_path
    return paths_by_stem


def select_split_hdf5_paths(
    cfg,
    dataset_name: str,
    hdf5_paths: Sequence[str],
    split: str,
) -> list[str]:
    """Select packed files using an external manifest or legacy HDF5 attrs.

    When an external protocol is configured, its union must equal the complete
    packed stem set.  This rejects both missing audio and unnoticed extra files
    instead of allowing filesystem contents to redefine the experiment.
    """

    split = str(split)
    if split not in SPLIT_FILES:
        raise ValueError(
            f"split must be one of {tuple(SPLIT_FILES)}, got {split!r}"
        )
    paths_by_stem = _hdf5_paths_by_stem(hdf5_paths)
    manifest = load_configured_split_manifest(cfg, dataset_name)
    if manifest is not None:
        manifest_stems = {
            stem
            for split_stems in manifest["splits"].values()
            for stem in split_stems
        }
        packed_stems = set(paths_by_stem)
        missing = sorted(manifest_stems - packed_stems)
        unassigned = sorted(packed_stems - manifest_stems)
        if missing or unassigned:
            details = []
            if missing:
                details.append(
                    f"missing_from_pack={len(missing)} {missing[:10]}"
                )
            if unassigned:
                details.append(
                    f"unassigned_packed={len(unassigned)} {unassigned[:10]}"
                )
            raise RuntimeError(
                "External split manifests do not exactly cover the packed dataset ("
                + "; ".join(details)
                + ")"
            )
        return [paths_by_stem[stem] for stem in manifest["splits"][split]]

    selected = []
    for stem, hdf5_path in sorted(paths_by_stem.items()):
        with h5py.File(hdf5_path, "r") as hdf5_file:
            if "split" not in hdf5_file.attrs:
                raise KeyError(
                    f"Packed file has no split attribute: {hdf5_path}"
                )
            if str(decode_hdf5_attr(hdf5_file.attrs["split"])) == split:
                selected.append(hdf5_path)
    return selected


def configured_split_identities(cfg, dataset_names) -> dict:
    """Return configured path-independent identities keyed by dataset name."""

    identities = {}
    for dataset_name in sorted({str(name) for name in dataset_names}):
        identity = configured_split_manifest_identity(cfg, dataset_name)
        if identity is not None:
            identities[dataset_name] = identity
    return identities


def validate_checkpoint_split_manifest(cfg, dataset_name: str, checkpoint) -> None:
    """Bind alternate-manifest inference to a checkpoint trained on it."""

    expected = configured_split_manifest_identity(cfg, dataset_name)
    if expected is None:
        return
    saved = None
    if isinstance(checkpoint, Mapping):
        reproducibility = checkpoint.get("reproducibility")
        if isinstance(reproducibility, Mapping):
            training_semantics = reproducibility.get("training_semantics")
            if isinstance(training_semantics, Mapping):
                data = training_semantics.get("data")
                if isinstance(data, Mapping):
                    identities = data.get("split_manifests")
                    if isinstance(identities, Mapping):
                        saved = identities.get(str(dataset_name))
    if saved != expected:
        raise ValueError(
            "The checkpoint is not bound to the configured external split "
            f"manifest for dataset={dataset_name!r}; refusing cross-protocol inference"
        )
