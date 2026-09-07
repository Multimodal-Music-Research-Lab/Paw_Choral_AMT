import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_DIR / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import models


class ModelPublicNamesTest(unittest.TestCase):
    def make_cfg(self, choral_enabled):
        return SimpleNamespace(choral=SimpleNamespace(enable=choral_enabled))

    def test_historical_class_names_are_exact_aliases(self):
        self.assertIs(models.FlexibleHPT, models.PagCT)
        self.assertIs(models.FlexibleHPTChoralStream, models.PawCT)
        self.assertEqual(models.PagCT.__name__, "PagCT")
        self.assertEqual(models.PawCT.__name__, "PawCT")

    def test_factory_routes_canonical_pagct_name(self):
        cfg = self.make_cfg(choral_enabled=False)
        instance = object()
        with (
            mock.patch.object(models, "get_task_spec", return_value=SimpleNamespace(arch="pagct")),
            mock.patch.object(models, "PagCT", return_value=instance) as constructor,
        ):
            self.assertIs(models.build_model(cfg), instance)
        constructor.assert_called_once_with(cfg)

    def test_factory_routes_canonical_pawct_name(self):
        cfg = self.make_cfg(choral_enabled=True)
        instance = object()
        with (
            mock.patch.object(models, "get_task_spec", return_value=SimpleNamespace(arch="pawct")),
            mock.patch.object(models, "PawCT", return_value=instance) as constructor,
        ):
            self.assertIs(models.build_model(cfg), instance)
        constructor.assert_called_once_with(cfg)

    def test_factory_preserves_historical_hpt_selection(self):
        with mock.patch.object(models, "get_task_spec", return_value=SimpleNamespace(arch="hpt")):
            pagct_cfg = self.make_cfg(choral_enabled=False)
            pagct_instance = object()
            with mock.patch.object(models, "PagCT", return_value=pagct_instance):
                self.assertIs(models.build_model(pagct_cfg), pagct_instance)

            pawct_cfg = self.make_cfg(choral_enabled=True)
            pawct_instance = object()
            with mock.patch.object(models, "PawCT", return_value=pawct_instance):
                self.assertIs(models.build_model(pawct_cfg), pawct_instance)

    def test_factory_rejects_conflicting_canonical_configuration(self):
        cases = (("pagct", True), ("pawct", False), ("onf", True))
        for arch, choral_enabled in cases:
            with self.subTest(arch=arch, choral_enabled=choral_enabled):
                cfg = self.make_cfg(choral_enabled=choral_enabled)
                with mock.patch.object(models, "get_task_spec", return_value=SimpleNamespace(arch=arch)):
                    with self.assertRaises(ValueError):
                        models.build_model(cfg)


if __name__ == "__main__":
    unittest.main()
