from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from aubo_workbench.hole_map import (
    build_rotary_sector_map_payload,
    build_rotary_sector_hole_repair_payload,
    get_hole,
    publish_current_hole_map,
    validate_hole_map,
)


def _hole_result(hole_id: int) -> dict:
    return {
        "status": "completed",
        "hole_id": hole_id,
        "hole_center_base_mm": np.asarray([100.0 + hole_id * 20.0, 200.0 + (hole_id % 2) * 30.0, 300.0]),
        "coarse_center_base_mm": np.asarray([100.0 + hole_id * 20.0, 200.0 + (hole_id % 2) * 30.0, 301.0]),
        "coarse_plane_point_base_mm": np.asarray([100.0 + hole_id * 20.0, 200.0 + (hole_id % 2) * 30.0, 301.0]),
        "coarse_normal_toward_camera_base": [0.0, 0.0, 1.0],
        "plane_normal_toward_camera_base": [0.0, 0.0, 1.0],
        "coarse_fine_reference_tcp_pose_m_rad": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
    }


class RotarySectorMapTests(unittest.TestCase):
    def test_legacy_v3_map_requires_rebuild(self) -> None:
        with self.assertRaisesRegex(ValueError, "重新建立"):
            validate_hole_map({
                "schema_version": 3,
                "kind": "rotary_six_sector_hole_map",
            })

    def test_sector_merge_and_partial_publish(self) -> None:
        report = {
            "camera": {"serial_number": "camera-1"},
            "final_result": {"holes": [_hole_result(1), _hole_result(2), _hole_result(3)]},
            "stages": {"home_selection": {"capture_tcp_pose_m_rad": [0.1, 0.2, 0.3, 0, 0, 0]}},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = build_rotary_sector_map_payload(
                report,
                map_id="rotary-1",
                source_run_dir=root,
                handeye_path=None,
                camera_identity={"serial_number": "camera-1"},
                charuco_model_source=None,
                charuco_model_matrix=np.eye(2),
                charuco_model_bias_mm=[0.0, 0.0],
                sector_id=1,
            )
            second = build_rotary_sector_map_payload(
                report,
                map_id="rotary-2",
                source_run_dir=root,
                handeye_path=None,
                camera_identity={"serial_number": "camera-1"},
                charuco_model_source=None,
                charuco_model_matrix=np.eye(2),
                charuco_model_bias_mm=[0.0, 0.0],
                sector_id=2,
                existing_payload=first,
            )
            self.assertEqual(second["schema_version"], 4)
            self.assertEqual(second["target_policy"], "coarse_map_requires_fresh_fine")
            self.assertTrue(second["requires_fresh_fine"])
            self.assertEqual(second["quality_summary"]["ready_sector_ids"], [1, 2])
            self.assertEqual(validate_hole_map(second, sector_id=2), [1, 2, 3])
            self.assertEqual(get_hole(second, 1, sector_id=2)["hole_id"], 1)
            record = get_hole(second, 1, sector_id=2)
            self.assertEqual(record["coarse_center_base_mm"], [120.0, 230.0, 301.0])
            for forbidden in (
                "fine_xy_base_mm",
                "fine_result_center_base_mm",
                "visual_center_base_mm",
                "execution_final_tcp_pose_m_rad",
                "execution_final_tcp_target_base_mm",
                "execution_target_details",
            ):
                self.assertNotIn(forbidden, record)
            map_path = root / "rotary-2" / "hole_map.json"
            current = root / "current.json"
            map_path.parent.mkdir()
            map_path.write_text("{}", encoding="utf-8")
            # 发布校验使用内存中的 payload；这里仅验证 v4 partial 允许更新指针。
            published = publish_current_hole_map(second, map_path, current)
            self.assertEqual(published, current)

    def test_single_hole_repair_preserves_other_holes_and_sectors(self) -> None:
        report = {
            "camera": {"serial_number": "camera-1"},
            "final_result": {"holes": [_hole_result(1), _hole_result(2), _hole_result(3)]},
            "stages": {"home_selection": {"capture_tcp_pose_m_rad": [0.1, 0.2, 0.3, 0, 0, 0]}},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = build_rotary_sector_map_payload(
                report,
                map_id="rotary-1",
                source_run_dir=root,
                handeye_path=None,
                camera_identity={"serial_number": "camera-1"},
                charuco_model_source=None,
                charuco_model_matrix=np.eye(2),
                charuco_model_bias_mm=[0.0, 0.0],
                sector_id=1,
            )
            with_sector_two = build_rotary_sector_map_payload(
                report,
                map_id="rotary-2",
                source_run_dir=root,
                handeye_path=None,
                camera_identity={"serial_number": "camera-1"},
                charuco_model_source=None,
                charuco_model_matrix=np.eye(2),
                charuco_model_bias_mm=[0.0, 0.0],
                sector_id=2,
                existing_payload=first,
            )
            changed = _hole_result(2)
            changed["hole_center_base_mm"] = np.asarray([999.0, 888.0, 777.0])
            changed_report = {
                "camera": {"serial_number": "camera-1"},
                "configuration": {},
                "final_result": {"holes": [changed]},
            }
            repaired = build_rotary_sector_hole_repair_payload(
                with_sector_two,
                changed_report,
                map_id="rotary-repair",
                source_run_dir=root / "repair-run",
                handeye_path=None,
                camera_identity={"serial_number": "camera-1"},
                charuco_model_source=None,
                charuco_model_matrix=np.eye(2),
                charuco_model_bias_mm=[0.0, 0.0],
                sector_id=1,
                hole_id=2,
            )
            self.assertEqual(repaired["map_id"], "rotary-repair")
            self.assertEqual(
                get_hole(repaired, 2, sector_id=1)["coarse_center_base_mm"],
                [140.0, 200.0, 301.0],
            )
            self.assertEqual(
                get_hole(repaired, 1, sector_id=1)["coarse_center_base_mm"],
                [120.0, 230.0, 301.0],
            )
            self.assertEqual(
                get_hole(repaired, 3, sector_id=1)["coarse_center_base_mm"],
                [160.0, 230.0, 301.0],
            )
            self.assertEqual(
                get_hole(repaired, 1, sector_id=2)["coarse_center_base_mm"],
                [120.0, 230.0, 301.0],
            )
            self.assertNotIn("fine_xy_base_mm", get_hole(repaired, 2, sector_id=1))
            self.assertEqual(validate_hole_map(repaired, sector_id=1, requested_hole_ids=[2]), [2])

    def test_repair_replaces_coarse_record_without_fine_fields(self) -> None:
        report = {
            "camera": {"serial_number": "camera-1"},
            "final_result": {"holes": [_hole_result(1), _hole_result(2)]},
        }
        existing = build_rotary_sector_map_payload(
            report,
            map_id="rotary-base",
            source_run_dir="source-run",
            handeye_path=None,
            camera_identity={"serial_number": "camera-1"},
            charuco_model_source=None,
            charuco_model_matrix=np.eye(2),
            charuco_model_bias_mm=[0.0, 0.0],
            sector_id=1,
        )
        fresh = _hole_result(2)
        fresh["coarse_center_base_mm"] = np.asarray([700.0, 800.0, 900.0])
        fresh["coarse_plane_point_base_mm"] = np.asarray([700.0, 800.0, 900.0])
        repaired = build_rotary_sector_hole_repair_payload(
            existing,
            {"final_result": {"holes": [fresh]}},
            map_id="rotary-coarse-repair",
            source_run_dir="repair-run",
            handeye_path=None,
            camera_identity={"serial_number": "camera-1"},
            charuco_model_source=None,
            charuco_model_matrix=np.eye(2),
            charuco_model_bias_mm=[0.0, 0.0],
            sector_id=1,
            hole_id=2,
        )
        hole = get_hole(repaired, 2, sector_id=1)
        self.assertEqual(hole["coarse_center_base_mm"], [700.0, 800.0, 900.0])
        self.assertEqual(hole["coarse_plane_point_base_mm"], [700.0, 800.0, 900.0])
        self.assertNotIn("fine_xy_base_mm", hole)
        self.assertNotIn("visual_center_base_mm", hole)
        self.assertNotIn("execution_final_tcp_pose_m_rad", hole)
        self.assertEqual(hole["coarse_center_base_mm"], [700.0, 800.0, 900.0])
