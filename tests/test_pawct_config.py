import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

import main_iter
from utilities import get_model_name, resolve_model_arch


def make_cfg(arch="pawct", target_assignment=None, legacy_assignment=None):
    return SimpleNamespace(
        model=SimpleNamespace(arch=arch, mode="frame_onset_offset", name="auto"),
        feature=SimpleNamespace(
            audio_feature="logmel",
            sample_rate=16000,
            frames_per_second=100,
        ),
        exp=SimpleNamespace(name_suffix=""),
        choral=SimpleNamespace(
            enable=arch in {"pawct", "hpt"},
            append_assignment_to_name=True,
            target_assignment=target_assignment,
            voice_assignment_method=legacy_assignment,
            assignment_module="heads",
            voice_interaction_module="none",
        ),
    )


class PawCTConfigTest(unittest.TestCase):
    def test_canonical_architectures_are_accepted(self):
        self.assertEqual(resolve_model_arch(make_cfg("pawct")), "pawct")
        self.assertEqual(resolve_model_arch(make_cfg("pagct")), "pagct")

    def test_target_assignment_has_clear_rp_and_oc_tags(self):
        self.assertTrue(get_model_name(make_cfg(target_assignment="range_prior")).endswith("_rp"))
        self.assertTrue(get_model_name(make_cfg(target_assignment="ordered_continuity")).endswith("_oc"))

    def test_legacy_assignment_key_is_still_resolved(self):
        cfg = make_cfg(target_assignment=None, legacy_assignment="ordered_continuity")
        self.assertTrue(get_model_name(cfg).endswith("_va_ordered_continuity"))

    def test_explicit_legacy_target_has_auditable_new_name(self):
        cfg = make_cfg(target_assignment="legacy_ordered_continuity")
        self.assertTrue(get_model_name(cfg).endswith("_legacy_oc"))

    def test_choral_pagct_and_pawct_share_the_note_pkl_union_loader(self):
        pagct_cfg = make_cfg("pagct")
        pawct_cfg = make_cfg("pawct")
        pagct_dataset = object()
        pawct_dataset = object()

        with (
            mock.patch.object(
                main_iter,
                "ChoralUnionDataset",
                return_value=pagct_dataset,
            ) as union_builder,
            mock.patch.object(
                main_iter,
                "ChoralSATBDataset",
                return_value=pawct_dataset,
            ) as satb_builder,
        ):
            self.assertIs(
                main_iter._build_dataset(
                    pagct_cfg,
                    "youchorale",
                    is_training=True,
                ),
                pagct_dataset,
            )
            self.assertIs(
                main_iter._build_dataset(
                    pawct_cfg,
                    "youchorale",
                    is_training=False,
                    formal_evaluation=True,
                ),
                pawct_dataset,
            )

        union_builder.assert_called_once_with(
            pagct_cfg,
            "youchorale",
            is_training=True,
            formal_evaluation=False,
        )
        satb_builder.assert_called_once_with(
            pawct_cfg,
            "youchorale",
            is_training=False,
            formal_evaluation=True,
        )


if __name__ == "__main__":
    unittest.main()
