from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aubo_workbench.hole_map_seed_correction import (
    apply_runtime_aligned_seed_correction,
    apply_seed_correction_to_hole,
    build_seed_correction_model,
    fit_runtime_seed_alignment,
    load_seed_correction_model,
    save_seed_correction_model,
    validate_model_binding,
)


class HoleMapSeedCorrectionTests(unittest.TestCase):
    def _map_payload(self) -> dict:
        return {
            "schema_version": 2,
            "kind": "coarse_hole_map",
            "map_id": "map-a",
            "holes": {
                "H01": {
                    "hole_id": 1,
                    "status": "ready",
                    "coarse_center_base_mm": [0.0, 0.0, 30.0],
                    "coarse_plane_point_base_mm": [0.5, 0.5, 31.0],
                    "coarse_normal_toward_camera_base": [0.0, 0.0, 1.0],
                },
                "H02": {
                    "hole_id": 2,
                    "status": "ready",
                    "coarse_center_base_mm": [100.0, 0.0, 30.0],
                    "coarse_plane_point_base_mm": [100.5, 0.5, 31.0],
                    "coarse_normal_toward_camera_base": [0.0, 0.0, 1.0],
                },
                "H03": {
                    "hole_id": 3,
                    "status": "ready",
                    "coarse_center_base_mm": [0.0, 100.0, 30.0],
                    "coarse_plane_point_base_mm": [0.5, 100.5, 31.0],
                    "coarse_normal_toward_camera_base": [0.0, 0.0, 1.0],
                },
            },
        }

    def _reference_report(self) -> dict:
        # Reference IDs intentionally do not match map IDs. Spatial identity is
        # the only safe correspondence for manually selected historical runs.
        return {
            "final_result": {
                "holes": [
                    {
                        "hole_id": 91,
                        "status": "completed",
                        "hole_center_base_mm": [1.0, 101.5, 30.0],
                        "fine_quality_status": "strict",
                        "valid_frames": 15,
                    },
                    {
                        "hole_id": 77,
                        "status": "completed",
                        "hole_center_base_mm": [1.0, 1.5, 30.0],
                        "fine_quality_status": "strict",
                        "valid_frames": 15,
                    },
                    {
                        "hole_id": 88,
                        "status": "completed",
                        "hole_center_base_mm": [101.0, 1.5, 30.0],
                        "fine_quality_status": "strict",
                        "valid_frames": 15,
                    },
                    {
                        "hole_id": 999,
                        "status": "completed",
                        "hole_center_base_mm": [500.0, 500.0, 30.0],
                    },
                ]
            }
        }

    def test_build_save_load_and_apply_xy_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_path = root / "hole_map.json"
            report_path = root / "reference" / "report.json"
            report_path.parent.mkdir()
            map_payload = self._map_payload()
            reference = self._reference_report()
            map_path.write_text(json.dumps(map_payload), encoding="utf-8")
            report_path.write_text(json.dumps(reference), encoding="utf-8")

            model = build_seed_correction_model(
                map_payload,
                reference,
                map_path=map_path,
                reference_report_path=report_path,
                sector_id=None,
                match_gate_mm=5.0,
                max_correction_mm=3.0,
            )
            self.assertEqual(model["matching"]["matched_count"], 3)
            self.assertEqual(model["matching"]["unmatched_reference_hole_ids"], [999])
            self.assertEqual(model["holes"]["H01"]["reference_hole_id"], 77)
            self.assertTrue(np.allclose(
                model["holes"]["H01"]["correction_xy_base_mm"], [1.0, 1.5],
            ))

            model_path = save_seed_correction_model(model, root / "correction.json")
            loaded = load_seed_correction_model(model_path)
            validate_model_binding(
                loaded,
                map_payload=map_payload,
                map_path=map_path,
                sector_id=None,
            )
            corrected, metadata = apply_seed_correction_to_hole(
                map_payload["holes"]["H01"], loaded,
            )
            self.assertTrue(metadata["applied"])
            self.assertTrue(np.allclose(
                corrected["coarse_center_base_mm"], [1.0, 1.5, 30.0],
            ))
            self.assertTrue(np.allclose(
                corrected["coarse_plane_point_base_mm"], [1.5, 2.0, 31.0],
            ))
            self.assertEqual(
                corrected["coarse_normal_toward_camera_base"], [0.0, 0.0, 1.0],
            )
            self.assertEqual(
                map_payload["holes"]["H01"]["coarse_center_base_mm"],
                [0.0, 0.0, 30.0],
            )

    def test_binding_rejects_changed_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_path = root / "hole_map.json"
            report_path = root / "report.json"
            payload = self._map_payload()
            reference = self._reference_report()
            map_path.write_text(json.dumps(payload), encoding="utf-8")
            report_path.write_text(json.dumps(reference), encoding="utf-8")
            model = build_seed_correction_model(
                payload,
                reference,
                map_path=map_path,
                reference_report_path=report_path,
                sector_id=None,
                match_gate_mm=5.0,
                max_correction_mm=3.0,
            )
            map_path.write_text(json.dumps({**payload, "updated": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "地图文件已变化"):
                validate_model_binding(
                    model,
                    map_payload=payload,
                    map_path=map_path,
                    sector_id=None,
                )

    def test_runtime_alignment_is_required_before_planning_correction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_path = root / "hole_map.json"
            report_path = root / "report.json"
            payload = self._map_payload()
            reference = self._reference_report()
            map_path.write_text(json.dumps(payload), encoding="utf-8")
            report_path.write_text(json.dumps(reference), encoding="utf-8")
            model = build_seed_correction_model(
                payload,
                reference,
                map_path=map_path,
                reference_report_path=report_path,
                sector_id=None,
                match_gate_mm=5.0,
                max_correction_mm=5.0,
            )
            model["safety"]["runtime_min_anchor_holes"] = 3
            angle = np.deg2rad(0.2)
            rotation = np.asarray([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ])
            translation = np.asarray([0.4, -0.3])
            observations = {}
            for hole_id in (1, 2, 3):
                reference_xy = np.asarray(
                    model["holes"][f"H{hole_id:02d}"]["reference_xy_base_mm"]
                )
                observations[hole_id] = rotation @ reference_xy + translation
            alignment = fit_runtime_seed_alignment(
                model, observations, {1: 1, 2: 1, 3: 2},
            )
            self.assertTrue(alignment["success"])
            self.assertEqual(alignment["inlier_hole_ids"], [1, 2, 3])

            planning_hole = {
                **payload["holes"]["H01"],
                "initial_center_base_mm": np.asarray([0.0, 0.0, 30.0]),
            }
            unchanged, waiting = apply_runtime_aligned_seed_correction(
                planning_hole, model, {"success": False},
            )
            self.assertFalse(waiting["applied"])
            self.assertTrue(np.allclose(unchanged["initial_center_base_mm"], [0, 0, 30]))

            corrected, details = apply_runtime_aligned_seed_correction(
                planning_hole, model, alignment,
            )
            self.assertTrue(details["applied"])
            self.assertTrue(np.allclose(
                np.asarray(corrected["initial_center_base_mm"])[:2],
                observations[1],
            ))


if __name__ == "__main__":
    unittest.main()
