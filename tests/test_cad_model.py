#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import unittest
from pathlib import Path

import numpy as np

from aubo_workbench.cad_model import (
    CadModel,
    load_cad_model_json,
    parse_step_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CadModelTests(unittest.TestCase):
    def test_generated_json_is_mm_top_z_plus_z_and_11_holes(self):
        path = Path(r"C:\MM\aubo_tools\data\cad_model\cad_hole_model.json")
        self.assertTrue(path.is_file(), path)
        model = load_cad_model_json(path)
        self.assertEqual(model.unit, "mm")
        self.assertEqual(model.hole_count, 11)
        self.assertAlmostEqual(model.top_surface_z_mm, 6.0, places=3)
        np.testing.assert_allclose(model.top_normal_cad, [0.0, 0.0, 1.0], atol=1e-9)
        self.assertAlmostEqual(model.holes[0].center_cad_mm[0], -150.0, places=3)
        self.assertAlmostEqual(model.holes[0].center_cad_mm[1], -90.0, places=3)
        self.assertAlmostEqual(model.holes[0].diameter_mm, 68.5, places=3)

    def test_json_loader_rejects_non_mm(self):
        data = {
            "unit": "inch",
            "top_surface_z_mm": 6,
            "top_normal_cad": [0, 0, 1],
            "holes": [
                {
                    "id": f"CAD-{index:02d}",
                    "center_cad_mm": [index, 0, 6],
                    "diameter_mm": 68.5,
                    "top_z_mm": 6,
                }
                for index in range(11)
            ],
        }
        with self.assertRaises(ValueError):
            CadModel.from_dict(data)

    def test_step_parser_returns_authoritative_geometry(self):
        step = PROJECT_ROOT / "孔位板_JXDZ26-KWB-001.STEP"
        self.assertTrue(step.is_file(), step)
        model = parse_step_model(step, expected_hole_count=11)
        self.assertEqual(model.hole_count, 11)
        self.assertAlmostEqual(model.top_surface_z_mm, 6.0, places=3)
        np.testing.assert_allclose(model.top_normal_cad, [0.0, 0.0, 1.0], atol=1e-9)
        self.assertAlmostEqual(float(model.bounds_min_mm[0]), -215.0, places=3)
        self.assertAlmostEqual(float(model.bounds_max_mm[0]), 215.0, places=3)
        self.assertAlmostEqual(float(model.bounds_min_mm[1]), -140.0, places=3)
        self.assertAlmostEqual(float(model.bounds_max_mm[1]), 140.0, places=3)
        diameters = sorted(round(hole.diameter_mm, 3) for hole in model.holes)
        self.assertEqual(diameters.count(68.5), 4)
        self.assertEqual(diameters.count(73.5), 3)
        self.assertEqual(diameters.count(78.5), 4)


if __name__ == "__main__":
    unittest.main()

