import io
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from data_generator import EvalSampler, Sampler, segment_start_times
from utilities import traverse_folder


def make_training_sampler():
    sampler = object.__new__(Sampler)
    sampler.dataset_type = "youchorale"
    sampler.split = "train"
    sampler.batch_size = 2
    sampler.random_seed = 17
    sampler.segment_list = [[f"song-{idx}", float(idx)] for idx in range(5)]
    sampler.hdf5_content_manifest = []
    sampler.segment_identity = sampler._compute_segment_identity()
    sampler.pointer = 0
    sampler.next_batch_index = 0
    sampler.random_state = np.random.RandomState(sampler.random_seed)
    sampler.segment_indexes = np.arange(len(sampler.segment_list))
    sampler.random_state.shuffle(sampler.segment_indexes)
    return sampler


def assert_sampler_states_equal(test_case, left, right):
    test_case.assertEqual(left, right)


class SegmentStartTimesTest(unittest.TestCase):
    def test_short_equal_and_tail_aligned_recordings_are_covered(self):
        self.assertEqual(segment_start_times(5.0, 10.0, 2.0), [0.0])
        self.assertEqual(segment_start_times(10.0, 10.0, 2.0), [0.0])
        self.assertEqual(segment_start_times(25.0, 10.0, 10.0), [0.0, 10.0, 15.0])
        self.assertEqual(segment_start_times(30.0, 10.0, 10.0), [0.0, 10.0, 20.0])

    def test_invalid_segment_inputs_fail_loudly(self):
        for args in [(-1.0, 10.0, 1.0), (10.0, 0.0, 1.0), (10.0, 1.0, 0.0)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                segment_start_times(*args)


class TrainingSamplerResumeTest(unittest.TestCase):
    def test_seek_batch_matches_uninterrupted_sequence(self):
        uninterrupted = make_training_sampler()
        uninterrupted_iterator = iter(uninterrupted)
        expected_batches = [next(uninterrupted_iterator) for _ in range(12)]

        resumed = make_training_sampler()
        resumed.seek_batch(7)
        resumed_iterator = iter(resumed)
        actual_batches = [next(resumed_iterator) for _ in range(5)]

        self.assertEqual(actual_batches, expected_batches[7:])
        self.assertEqual(resumed.next_batch_index, 12)

    def test_state_dict_for_batch_does_not_mutate_live_prefetched_state(self):
        sampler = make_training_sampler()
        iterator = iter(sampler)
        for _ in range(4):
            next(iterator)
        before = sampler.state_dict()

        logical_state = sampler.state_dict_for_batch(2)

        after = sampler.state_dict()
        assert_sampler_states_equal(self, before, after)
        self.assertEqual(logical_state["next_batch_index"], 2)

    def test_state_round_trip_restores_indexes_pointer_and_rng(self):
        source = make_training_sampler()
        source_iterator = iter(source)
        for _ in range(3):
            next(source_iterator)
        state = source.state_dict()
        expected = [next(source_iterator) for _ in range(8)]

        restored = make_training_sampler()
        restored.load_state_dict(state)
        restored_iterator = iter(restored)
        actual = [next(restored_iterator) for _ in range(8)]

        self.assertEqual(actual, expected)

    def test_versioned_sampler_state_uses_only_primitive_containers(self):
        sampler = make_training_sampler()
        state = sampler.state_dict_for_batch(3)

        self.assertEqual(state["sampler_state_version"], 2)
        self.assertIsInstance(state["segment_indexes"], list)
        self.assertTrue(all(type(index) is int for index in state["segment_indexes"]))
        self.assertIsInstance(state["random_state"], dict)
        self.assertIsInstance(state["random_state"]["keys"], list)
        self.assertTrue(all(type(key) is int for key in state["random_state"]["keys"]))

        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        safely_loaded = torch.load(buffer, weights_only=True)
        self.assertEqual(safely_loaded, state)

    def test_historical_sampler_state_still_restores_after_file_load_gate(self):
        source = make_training_sampler()
        state = source.state_dict_for_batch(3)
        state["sampler_state_version"] = 1
        state["segment_indexes"] = np.asarray(state["segment_indexes"], dtype=np.int64)
        rng = state["random_state"]
        state["random_state"] = (
            rng["algorithm"],
            np.asarray(rng["keys"], dtype=np.uint32),
            rng["position"],
            rng["has_gauss"],
            rng["cached_gaussian"],
        )

        restored = make_training_sampler()
        restored.load_state_dict(state)

        self.assertEqual(restored.state_dict()["next_batch_index"], 3)

    def test_legacy_or_wrong_segment_state_fails_closed(self):
        sampler = make_training_sampler()
        with self.assertRaisesRegex(ValueError, "inexact legacy restore"):
            sampler.load_state_dict({"pointer": 0, "segment_indexes": np.arange(5)})

        state = sampler.state_dict_for_batch(2)
        state["segment_identity"] = "changed-dataset"
        with self.assertRaisesRegex(ValueError, "segment_identity"):
            sampler.load_state_dict(state)

    def test_constructor_uses_complete_segment_coverage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_dir = Path(tmpdir) / "hdf5s" / "smd_sr10"
            hdf5_dir.mkdir(parents=True)
            for name, duration in [
                ("a-short.h5", 5.0),
                ("b-equal.h5", 10.0),
                ("c-long.h5", 25.0),
            ]:
                with h5py.File(hdf5_dir / name, "w") as hf:
                    hf.attrs["split"] = "validation"
                    hf.attrs["duration"] = duration
                    hf.create_dataset(
                        "waveform",
                        data=np.zeros(int(round(duration * 10)), dtype=np.int16),
                    )

            cfg = SimpleNamespace(
                exp=SimpleNamespace(
                    workspace=tmpdir,
                    batch_size=2,
                    random_seed=17,
                    mini_data=False,
                    max_eval_batches=None,
                    max_train_eval_batches=None,
                ),
                feature=SimpleNamespace(
                    sample_rate=10,
                    segment_seconds=10.0,
                    hop_seconds=10.0,
                ),
                dataset=SimpleNamespace(train_set="smd"),
            )

            sampler = EvalSampler(cfg, "validation", is_eval="smd")

        self.assertEqual(
            sampler.segment_list,
            [
                ["a-short.h5", 0.0],
                ["b-equal.h5", 0.0],
                ["c-long.h5", 0.0],
                ["c-long.h5", 10.0],
                ["c-long.h5", 15.0],
            ],
        )

    def test_waveform_length_prevents_duplicate_from_inexact_duration_attr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_dir = Path(tmpdir) / "hdf5s" / "smd_sr16000"
            hdf5_dir.mkdir(parents=True)
            with h5py.File(hdf5_dir / "exact-ten.h5", "w") as hf:
                hf.attrs["split"] = "validation"
                hf.attrs["duration"] = 10.00001
                hf.create_dataset(
                    "waveform",
                    data=np.zeros(160000, dtype=np.int16),
                )

            cfg = SimpleNamespace(
                exp=SimpleNamespace(
                    workspace=tmpdir,
                    batch_size=2,
                    random_seed=17,
                    mini_data=False,
                    max_eval_batches=None,
                    max_train_eval_batches=None,
                ),
                feature=SimpleNamespace(
                    sample_rate=16000,
                    segment_seconds=10.0,
                    hop_seconds=1.0,
                ),
                dataset=SimpleNamespace(train_set="smd"),
            )

            sampler = EvalSampler(cfg, "validation", is_eval="smd")

        self.assertEqual(sampler.segment_list, [["exact-ten.h5", 0.0]])

    def test_changed_hdf5_content_rejects_exact_sampler_restore(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_dir = Path(tmpdir) / "hdf5s" / "smd_sr10"
            hdf5_dir.mkdir(parents=True)
            hdf5_path = hdf5_dir / "song.h5"
            with h5py.File(hdf5_path, "w") as hf:
                hf.attrs["split"] = "train"
                hf.attrs["duration"] = 2.0
                hf.create_dataset("waveform", data=np.zeros(20, dtype=np.int16))

            cfg = SimpleNamespace(
                exp=SimpleNamespace(
                    workspace=tmpdir,
                    batch_size=1,
                    random_seed=17,
                    mini_data=False,
                ),
                feature=SimpleNamespace(
                    sample_rate=10,
                    segment_seconds=1.0,
                    hop_seconds=1.0,
                ),
                dataset=SimpleNamespace(train_set="smd"),
            )
            original = Sampler(cfg, "train")
            saved_state = original.state_dict_for_batch(1)

            with h5py.File(hdf5_path, "r+") as hf:
                hf["waveform"][0] = 1

            changed = Sampler(cfg, "train")

            with self.assertRaisesRegex(
                ValueError,
                "segment_identity|hdf5_content_manifest",
            ):
                changed.load_state_dict(saved_state)

    def test_changed_satb_reference_rejects_exact_sampler_restore(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_dir = Path(tmpdir) / "hdf5s" / "youchorale_sr10"
            hdf5_dir.mkdir(parents=True)
            with h5py.File(hdf5_dir / "song.h5", "w") as hf:
                hf.attrs["split"] = "train"
                hf.create_dataset("waveform", data=np.zeros(20, dtype=np.int16))

            dataset_dir = Path(tmpdir) / "YouChorale"
            note_dir = dataset_dir / "note"
            note_dir.mkdir(parents=True)
            note_path = note_dir / "song.pkl"
            with note_path.open("wb") as file_handle:
                pickle.dump([{"S": [(0.0, 1.0, 60)]}], file_handle)

            cfg = SimpleNamespace(
                exp=SimpleNamespace(
                    workspace=tmpdir,
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
                    youchorale_dir=str(dataset_dir),
                ),
                choral=SimpleNamespace(enable=True),
            )
            original = Sampler(cfg, "train")
            saved_state = original.state_dict_for_batch(1)

            with note_path.open("wb") as file_handle:
                pickle.dump([{"S": [(0.0, 1.0, 61)]}], file_handle)

            changed = Sampler(cfg, "train")
            with self.assertRaisesRegex(
                ValueError,
                "segment_identity|hdf5_content_manifest",
            ):
                changed.load_state_dict(saved_state)


class EvalSamplerIterationTest(unittest.TestCase):
    def make_sampler(self, max_batches=None):
        sampler = object.__new__(EvalSampler)
        sampler.segment_list = [[f"song-{idx}", float(idx)] for idx in range(5)]
        sampler.segment_indexes = np.arange(5)
        sampler.batch_size = 2
        sampler.max_evaluate_iteration = max_batches
        return sampler

    def test_complete_eval_has_no_shuffle_or_duplicate_padding(self):
        sampler = self.make_sampler()
        batches = list(sampler)
        self.assertEqual([item for batch in batches for item in batch], sampler.segment_list)
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        self.assertEqual(len(sampler), 3)

    def test_optional_limit_is_explicit_and_deterministic(self):
        sampler = self.make_sampler(max_batches=2)
        self.assertEqual(
            list(sampler),
            [
                [sampler.segment_list[0], sampler.segment_list[1]],
                [sampler.segment_list[2], sampler.segment_list[4]],
            ],
        )
        self.assertEqual(len(sampler), 2)

    def test_filesystem_discovery_is_sorted_across_machines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "z_dir").mkdir()
            (root / "a_dir").mkdir()
            (root / "z_file.h5").touch()
            (root / "a_file.h5").touch()
            (root / "z_dir" / "b.h5").touch()
            (root / "a_dir" / "c.h5").touch()

            names, paths = traverse_folder(tmpdir)

        self.assertEqual(names, ["a_file.h5", "z_file.h5", "c.h5", "b.h5"])
        self.assertEqual(
            [Path(path).name for path in paths],
            ["a_file.h5", "z_file.h5", "c.h5", "b.h5"],
        )


if __name__ == "__main__":
    unittest.main()
