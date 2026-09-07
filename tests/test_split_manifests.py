import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from data_generator import Sampler
from probability_artifacts import expected_split_probability_stems
from split_manifests import (
    configured_split_manifest_identity,
    load_configured_split_manifest,
    select_split_hdf5_paths,
    validate_checkpoint_split_manifest,
)


def write_manifests(directory: Path, *, train, valid, test):
    directory.mkdir(parents=True, exist_ok=True)
    for filename, values in (
        ("train.json", train),
        ("valid.json", valid),
        ("test.json", test),
    ):
        (directory / filename).write_text(
            json.dumps(values, indent=2) + "\n",
            encoding="utf-8",
        )


def make_cfg(root: Path, split_dir: Path | None):
    return SimpleNamespace(
        exp=SimpleNamespace(
            workspace=str(root),
            batch_size=1,
            random_seed=17,
            mini_data=False,
        ),
        feature=SimpleNamespace(
            sample_rate=10,
            segment_seconds=1.0,
            hop_seconds=1.0,
        ),
        dataset=SimpleNamespace(
            train_set="youchorale",
            test_set="youchorale",
            eval_split="validation",
            youchorale_dir=str(root / "YouChorale"),
            youchorale_split_dir=(
                None if split_dir is None else str(split_dir)
            ),
        ),
    )


def write_hdf5(path: Path, official_split: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as hdf5_file:
        hdf5_file.attrs["split"] = official_split
        hdf5_file.create_dataset(
            "waveform",
            data=np.zeros(20, dtype=np.int16),
        )


class ExternalSplitManifestTest(unittest.TestCase):
    def test_manifest_identity_is_path_independent_and_records_exact_files(self):
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first_root = Path(first_tmp)
            second_root = Path(second_tmp)
            values = dict(train=["a"], valid=["b"], test=["c"])
            write_manifests(first_root / "splits", **values)
            write_manifests(second_root / "elsewhere", **values)

            first = configured_split_manifest_identity(
                make_cfg(first_root, first_root / "splits"),
                "youchorale",
            )
            second = configured_split_manifest_identity(
                make_cfg(second_root, second_root / "elsewhere"),
                "youchorale",
            )

        self.assertEqual(first, second)
        self.assertEqual(first["recording_counts"], {
            "train": 1,
            "validation": 1,
            "test": 1,
        })
        self.assertEqual(first["total_recordings"], 3)
        self.assertEqual(len(first["content_sha256"]), 64)
        self.assertIn("ordered_recording_ids_sha256", first)
        self.assertIn("hash_semantics", first)

    def test_malformed_duplicate_and_overlapping_manifests_fail_closed(self):
        cases = (
            (dict(train=[], valid=["b"], test=["c"]), "non-empty"),
            (dict(train=["a", "a"], valid=["b"], test=["c"]), "Duplicate"),
            (dict(train=["a"], valid=["a"], test=["c"]), "pairwise disjoint"),
            (dict(train=["../a"], valid=["b"], test=["c"]), "filename stems"),
        )
        for values, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                split_dir = root / "splits"
                write_manifests(split_dir, **values)
                with self.assertRaisesRegex(ValueError, message):
                    load_configured_split_manifest(
                        make_cfg(root, split_dir),
                        "youchorale",
                    )

    def test_missing_manifest_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "splits"
            write_manifests(split_dir, train=["a"], valid=["b"], test=["c"])
            (split_dir / "valid.json").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "validation"):
                load_configured_split_manifest(
                    make_cfg(root, split_dir),
                    "youchorale",
                )

    def test_external_manifest_overrides_old_hdf5_split_attributes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "splits"
            write_manifests(split_dir, train=["a"], valid=["b"], test=["c"])
            hdf5_dir = root / "hdf5s" / "youchorale_sr10"
            paths = []
            for stem, old_split in (
                ("a", "test"),
                ("b", "train"),
                ("c", "validation"),
            ):
                path = hdf5_dir / f"{stem}.h5"
                write_hdf5(path, old_split)
                paths.append(str(path))
            cfg = make_cfg(root, split_dir)

            selected = select_split_hdf5_paths(
                cfg,
                "youchorale",
                paths,
                "train",
            )

        self.assertEqual([Path(path).stem for path in selected], ["a"])

    def test_pack_and_manifest_must_cover_exactly_the_same_stems(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "splits"
            write_manifests(split_dir, train=["a"], valid=["b"], test=["c"])
            hdf5_dir = root / "hdf5s" / "youchorale_sr10"
            paths = []
            for stem in ("a", "b", "extra"):
                path = hdf5_dir / f"{stem}.h5"
                write_hdf5(path, "train")
                paths.append(str(path))

            with self.assertRaisesRegex(
                RuntimeError,
                "missing_from_pack=1.*unassigned_packed=1",
            ):
                select_split_hdf5_paths(
                    make_cfg(root, split_dir),
                    "youchorale",
                    paths,
                    "train",
                )

    def test_training_sampler_and_scoring_use_the_same_external_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "splits"
            write_manifests(split_dir, train=["a"], valid=["b"], test=["c"])
            hdf5_dir = root / "hdf5s" / "youchorale_sr10"
            note_dir = root / "YouChorale" / "note"
            note_dir.mkdir(parents=True)
            for stem, old_split in (
                ("a", "test"),
                ("b", "train"),
                ("c", "validation"),
            ):
                write_hdf5(hdf5_dir / f"{stem}.h5", old_split)
                with (note_dir / f"{stem}.pkl").open("wb") as note_file:
                    pickle.dump([{"S": [(0.0, 1.0, 60)]}], note_file)
            cfg = make_cfg(root, split_dir)

            train_sampler = Sampler(cfg, "train")
            validation_stems = expected_split_probability_stems(cfg)

        self.assertEqual(
            {Path(item[0]).stem for item in train_sampler.segment_list},
            {"a"},
        )
        self.assertEqual(validation_stems, ("b",))

    def test_alternate_manifest_inference_requires_a_bound_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "splits"
            write_manifests(split_dir, train=["a"], valid=["b"], test=["c"])
            cfg = make_cfg(root, split_dir)
            identity = configured_split_manifest_identity(cfg, "youchorale")
            checkpoint = {
                "reproducibility": {
                    "training_semantics": {
                        "data": {
                            "split_manifests": {"youchorale": identity},
                        }
                    }
                }
            }

            validate_checkpoint_split_manifest(cfg, "youchorale", checkpoint)
            with self.assertRaisesRegex(ValueError, "cross-protocol inference"):
                validate_checkpoint_split_manifest(cfg, "youchorale", {})


if __name__ == "__main__":
    unittest.main()
