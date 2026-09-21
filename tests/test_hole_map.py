from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from aubo_workbench.hole_map import (
    build_hole_map_payload,
    get_hole,
    load_hole_map,
    publish_current_hole_map,
    resolve_hole_map_path,
    save_hole_map,
    validate_hole_map,
)
from aubo_workbench.hole_map_visualization import export_hole_map_artifacts


def _completed_result(hole_id: int) -> dict:
    return {
        "status": "completed",
        "hole_id": hole_id,
        "tracking_identity": f"initial_selection_hole_{hole_id}",
        "initial_selection_order": hole_id,
        "hole_center_base_mm": np.array([100.0 + hole_id, 200.0, 300.0]),
        "coarse_center_base_mm": np.array([100.0 + hole_id, 200.0, 301.0]),
        "coarse_plane_point_base_mm": np.array([100.0 + hole_id, 200.0, 301.0]),
        "coarse_normal_toward_camera_base": np.array([0.0, 0.0, 1.0]),
        "plane_normal_toward_camera_base": np.array([0.0, 0.0, 1.0]),
        "coarse_fine_reference_tcp_pose_m_rad": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        "diameter_estimate_mm": 70.0,
        "matched_diameter_mm": 70.0,
        "fine_xy_source": "shared_batch_fine",
        "fine_z_source": "per_hole_coarse_center_z",
        "batch_fine_source": "batch_fine_at_260mm",
        "batch_fine_capture_round": 0,
        "batch_fine_joint_applied": True,
        "coarse_valid_frames": 10,
        "coarse_center_scatter_p95_px": 0.2,
        "coarse_plane_rmse_mm": 0.5,
        "fine_quality_status": "strict",
        "valid_frames": 8,
        "center_scatter_p95_px": 0.1,
        "batch_fine_joint_summary": {"success": True},
        "hole_result_type": "base_frame_3d_point",
        "final_pose_source": "per_hole_coarse_pose_with_batch_fine_xy_only",
    }


class HoleMapTests(unittest.TestCase):
    def test_build_save_load_and_select_holes(self) -> None:
        report = {
            "configuration": {
                "batch_coarse_localization": True,
                "batch_fine_localization": True,
                "batch_fine_joint_localization": True,
            },
            "camera": {"serial_number": "camera-1"},
            "final_result": {
                "holes": [
                    _completed_result(1),
                    {"status": "deferred_fine_quality", "hole_id": 2, "error": "not enough frames"},
                    _completed_result(3),
                ],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source-run"
            source.mkdir()
            handeye = Path(directory) / "handeye.json"
            handeye.write_text("{}", encoding="utf-8")
            payload = build_hole_map_payload(
                report,
                map_id="hole-map-test",
                source_run_dir=source,
                handeye_path=handeye,
                camera_identity={"serial_number": "camera-1"},
                charuco_model_source=handeye,
                charuco_model_matrix=np.eye(2),
                charuco_model_bias_mm=[0.0, 0.0],
            )
            self.assertEqual(payload["status"], "partial")
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["target_policy"], "coarse_map_requires_fresh_fine")
            self.assertTrue(payload["requires_fresh_fine"])
            self.assertEqual(payload["quality_summary"]["ready_holes"], 2)
            self.assertEqual(validate_hole_map(payload), [1, 3])
            self.assertEqual(validate_hole_map(payload, requested_hole_ids=[3, 1]), [3, 1])
            record = get_hole(payload, 3)
            self.assertEqual(record["coarse_center_base_mm"], [103.0, 200.0, 301.0])
            for forbidden in (
                "fine_xy_base_mm",
                "fine_result_center_base_mm",
                "visual_center_base_mm",
                "execution_final_tcp_pose_m_rad",
                "execution_final_tcp_target_base_mm",
                "execution_target_details",
            ):
                self.assertNotIn(forbidden, record)

            target = Path(directory) / "hole_map.json"
            save_hole_map(payload, target)
            loaded = load_hole_map(target)
            self.assertEqual(loaded["map_id"], "hole-map-test")
            self.assertEqual(validate_hole_map(loaded), [1, 3])

    def test_execution_only_fields_are_not_persisted_or_required(self) -> None:
        report = {
            "final_result": {"holes": [_completed_result(1)]},
        }
        payload = build_hole_map_payload(
            report,
            map_id="hole-map-test",
            source_run_dir="source-run",
            handeye_path=None,
            camera_identity=None,
            charuco_model_source=None,
        )
        self.assertEqual(validate_hole_map(payload, expected_target_mode="normal"), [1])
        record = get_hole(payload, 1)
        self.assertNotIn("final_point_mode", record)
        self.assertNotIn("final_point_offset_base_mm", record)

    def test_coarse_batch_map_rejects_unaccepted_group_refinement(self) -> None:
        report = {
            "configuration": {"batch_coarse_localization": True},
            "map_build_localization_mode": "coarse_only",
            "stages": {
                "batch_coarse_results": {
                    "groups": [{
                        "group_index": 1,
                        "hole_ids": [1],
                        "pose_refinement": {
                            "accepted": False,
                            "reason": "pose_correction_exceeds_motion_gate",
                        },
                    }],
                },
            },
            "final_result": {"holes": [_completed_result(1)]},
        }
        with self.assertRaisesRegex(ValueError, "未收敛"):
            build_hole_map_payload(
                report,
                map_id="coarse-map-rejected",
                source_run_dir="source-run",
                handeye_path=None,
                camera_identity=None,
            )

    def test_coarse_batch_map_allows_singleton_group_without_shared_refinement(self) -> None:
        report = {
            "configuration": {"batch_coarse_localization": True},
            "map_build_localization_mode": "coarse_only",
            "stages": {
                "batch_coarse_results": {
                    "groups": [{
                        "group_index": 1,
                        "hole_ids": [1],
                        "pose_refinement": {
                            "accepted": False,
                            "map_build_safe": True,
                            "reason": "singleton_group_not_shared",
                        },
                    }],
                },
            },
            "final_result": {"holes": [_completed_result(1)]},
        }
        payload = build_hole_map_payload(
            report,
            map_id="coarse-map-singleton",
            source_run_dir="source-run",
            handeye_path=None,
            camera_identity=None,
        )
        self.assertEqual(validate_hole_map(payload), [1])

    def test_per_hole_map_persists_fine_reference_without_final_tcp(self) -> None:
        result = _completed_result(1)
        result.update({
            "map_build_localization_mode": "per_hole",
            "map_build_fine_reference": True,
            "fine_tcp_pose_m_rad": np.array([0.41, 0.52, 0.26, 0.0, 0.0, 1.57]),
            "fine_plane_intersection_mm": np.array([101.2, 199.8, 300.2]),
            "fine_center_source": "strict_ellipse_yolo",
            "fine_quality_note": "strict quality gate passed",
            "ellipse_residual_median_px": 0.18,
            "fine_height_estimate_mm": 260.1,
        })
        report = {
            "map_build_localization_mode": "per_hole",
            "final_result": {"holes": [result]},
        }
        payload = build_hole_map_payload(
            report,
            map_id="per-hole-map",
            source_run_dir="source-run",
            handeye_path=None,
            camera_identity=None,
        )
        self.assertEqual(payload["map_build_localization_mode"], "per_hole")
        self.assertEqual(payload["quality_summary"]["fine_reference_ready_holes"], [1])
        self.assertEqual(payload["quality_summary"]["fine_reference_missing_holes"], [])
        record = get_hole(payload, 1)
        reference = record["fine_reference"]
        self.assertEqual(reference["status"], "ready")
        self.assertEqual(reference["hole_center_base_mm"], [101.0, 200.0, 300.0])
        self.assertEqual(reference["fine_plane_intersection_mm"], [101.2, 199.8, 300.2])
        self.assertEqual(reference["fine_capture_tcp_pose_m_rad"], [0.41, 0.52, 0.26, 0.0, 0.0, 1.57])
        self.assertEqual(validate_hole_map(payload), [1])
        self.assertNotIn("final_tcp_pose_m_rad", reference)
        self.assertNotIn("target_point_base_mm", reference)

    def test_invalid_requested_hole_is_rejected(self) -> None:
        report = {"final_result": {"holes": [_completed_result(1)]}}
        payload = build_hole_map_payload(
            report,
            map_id="hole-map-test",
            source_run_dir="source-run",
            handeye_path=None,
            camera_identity=None,
            charuco_model_source=None,
        )
        with self.assertRaises(ValueError):
            validate_hole_map(payload, requested_hole_ids=[2])

    def test_current_pointer_publishes_only_complete_map(self) -> None:
        report = {"final_result": {"holes": [_completed_result(1)]}}
        payload = build_hole_map_payload(
            report,
            map_id="hole-map-complete",
            source_run_dir="source-run",
            handeye_path=None,
            camera_identity=None,
            charuco_model_source=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "hole_localization_maps"
            map_path = root / "hole-map-complete" / "hole_map.json"
            current = root / "current.json"
            save_hole_map(payload, map_path)
            published = publish_current_hole_map(payload, map_path, current)
            self.assertEqual(published, current)
            self.assertEqual(resolve_hole_map_path(current), map_path.resolve())
            self.assertEqual(load_hole_map(current)["map_id"], "hole-map-complete")

            partial = dict(payload)
            partial["status"] = "partial"
            self.assertIsNone(publish_current_hole_map(partial, map_path, current))
            self.assertEqual(load_hole_map(current)["map_id"], "hole-map-complete")

    def test_pointcloud_artifacts_are_self_contained(self) -> None:
        report = {"final_result": {"holes": [_completed_result(1), _completed_result(2)]}}
        payload = build_hole_map_payload(
            report,
            map_id="hole-map-cloud",
            source_run_dir="source-run",
            handeye_path=None,
            camera_identity=None,
            charuco_model_source=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "batch_pointcloud.npz"
            np.savez_compressed(
                source,
                points_camera_mm=np.asarray([[0.0, 0.0, 340.0], [10.0, 0.0, 340.0]], dtype=np.float32),
                point_hole_ids=np.asarray([1, 2], dtype=np.int32),
                point_frame_indices=np.asarray([0, 0], dtype=np.int32),
                T_base_camera=np.eye(4, dtype=np.float64),
            )
            artifacts = export_hole_map_artifacts(source, root / "map", payload)
            self.assertEqual(artifacts["status"], "ready")
            for name in ("raw_npz", "ply", "centers_ply", "preview_jpg"):
                self.assertTrue((root / "map" / artifacts[name]).is_file())
            with np.load(root / "map" / "pointcloud_raw.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["points_base_mm"].shape, (2, 3))
            preview = cv2.imread(str(root / "map" / "pointcloud_preview.jpg"))
            self.assertIsNotNone(preview)
            self.assertGreater(int(preview.shape[1]), 1000)
