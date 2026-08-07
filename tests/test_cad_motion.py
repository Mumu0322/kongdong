import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import run_yolo_eye_in_hand_optimized as module
from aubo_workbench.cad_registration import CameraIntrinsics


class CadMotionPlanningTests(unittest.TestCase):
    @staticmethod
    def _synthetic_intrinsics():
        return SimpleNamespace(
            fx=600.0,
            fy=600.0,
            cx=640.0,
            cy=360.0,
            width=1280,
            height=720,
            distortion=(),
        )

    def test_cad_entry_plan_hits_260_mm_and_keeps_cad_center(self):
        handeye = SimpleNamespace(T_tcp_rgb_camera=np.eye(4, dtype=np.float64))
        hole = {
            "hole_id": "CAD-01",
            "point_base_mm": np.array([10.0, 20.0, 0.0]),
            "normal_base": np.array([0.0, 0.0, 1.0]),
            "normal_toward_camera_base": np.array([0.0, 0.0, 1.0]),
            "diameter_mm": 68.5,
            "top_z_mm": 6.0,
        }
        reference = np.eye(4, dtype=np.float64)
        target, entry = module._cad_hole_entry_plan(
            hole, reference, handeye, fixed_rz_rad=0.0, fine_height_mm=260.0,
        )

        self.assertAlmostEqual(entry["achieved_camera_height_mm"], 260.0, places=6)
        np.testing.assert_allclose(entry["cad_center_base_mm"], hole["point_base_mm"])
        self.assertAlmostEqual(
            module.camera_height_to_plane_mm(target, handeye.T_tcp_rgb_camera, hole["point_base_mm"]),
            260.0,
            places=6,
        )

    def test_registration_loader_rejects_failed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "cad_registration_report.json"
            report_path.write_text(json.dumps({
                "result": {
                    "success": False,
                    "multi_frame_gate_pass": False,
                    "failure_reasons": ["synthetic failure"],
                },
            }), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                module._load_cad_motion_input(
                    report_path,
                    module.DEFAULT_CAD_MODEL_JSON,
                    module.CadMotionConfig(),
                )

    def test_registration_loader_accepts_successful_report_and_validates_model(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "cad_registration_report.json"
            report_path.write_text(json.dumps({
                "motion_allowed": False,
                "result": {
                    "success": True,
                    "multi_frame_gate_pass": True,
                    "valid_frame_indices": [0, 1, 2],
                    "cross_frame_center_p95_mm": 1.2,
                    "T_base_cad": np.eye(4).tolist(),
                },
                "inputs": {
                    "cad_json": str(module.DEFAULT_CAD_MODEL_JSON),
                },
            }), encoding="utf-8")
            result = module._load_cad_motion_input(
                report_path,
                module.DEFAULT_CAD_MODEL_JSON,
                module.CadMotionConfig(),
            )
            self.assertEqual(result.cad_model.hole_count, 11)
            np.testing.assert_allclose(result.T_base_cad, np.eye(4))

    def test_fresh_cad_model_loader_does_not_read_previous_report_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            previous_report = Path(directory) / "old_report.json"
            previous_report.write_text("not a registration report", encoding="utf-8")
            model_path, model, previous = module._load_cad_model_for_fresh_motion(
                module.DEFAULT_CAD_MODEL_JSON,
                previous_report,
            )
            self.assertEqual(model_path, module.DEFAULT_CAD_MODEL_JSON)
            self.assertEqual(model.hole_count, 11)
            self.assertEqual(previous, previous_report)

    def test_fresh_cad_detections_undistort_yolo_center_with_cad_intrinsics(self):
        cad_intrinsics = CameraIntrinsics(
            width=1280,
            height=720,
            fx=600.0,
            fy=600.0,
            cx=640.0,
            cy=360.0,
            dist_coeffs=np.zeros(5, dtype=np.float64),
        )
        detection = {
            "box": [100.0, 200.0, 180.0, 280.0],
            "center": [140.0, 240.0],
            "confidence": 0.95,
            "class_id": 0,
        }
        with patch.object(module, "detect", return_value=[detection]), patch.object(
            module, "fit_hole_ellipse", side_effect=RuntimeError("ellipse unavailable")
        ):
            result = module._build_fresh_cad_detections(
                np.zeros((720, 1280, 3), dtype=np.uint8),
                object(),
                0.35,
                self._synthetic_intrinsics(),
                cad_intrinsics,
            )

        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0].center_px, [140.0, 240.0])

    def test_group_plan_centers_selected_holes_and_checks_circle_margin(self):
        handeye = SimpleNamespace(T_tcp_rgb_camera=np.eye(4, dtype=np.float64))
        holes = []
        for hole_id, point in enumerate((
            (-100.0, -60.0, 0.0),
            (100.0, -60.0, 0.0),
            (-100.0, 60.0, 0.0),
            (100.0, 60.0, 0.0),
        ), start=1):
            holes.append({
                "hole_id": f"CAD-{hole_id:02d}",
                "point_base_mm": np.asarray(point, dtype=np.float64),
                "normal_toward_camera_base": np.array([0.0, 0.0, 1.0]),
                "diameter_mm": 60.0,
            })
        target, plan = module._cad_group_entry_plan(
            holes,
            [hole["hole_id"] for hole in holes],
            np.eye(4, dtype=np.float64),
            handeye,
            fixed_rz_rad=0.0,
            intrinsics=self._synthetic_intrinsics(),
            depth_height_mm=340.0,
            view_margin_px=35.0,
        )

        np.testing.assert_allclose(plan["group_center_px"], [640.0, 360.0], atol=1e-6)
        self.assertTrue(all(value > 0.0 for value in plan["projected_hole_radius_px"].values()))
        self.assertAlmostEqual(
            module.camera_height_to_plane_mm(
                target, handeye.T_tcp_rgb_camera, plan["group_point_base_mm"],
            ),
            340.0,
            places=6,
        )

    def test_shared_depth_offset_is_converted_to_base_z_without_changing_xy(self):
        handeye = SimpleNamespace(T_tcp_rgb_camera=np.eye(4, dtype=np.float64))
        hole = {
            "hole_id": "CAD-01",
            "point_base_mm": np.array([10.0, 20.0, 0.0]),
            "normal_base": np.array([0.0, 0.0, 1.0]),
            "normal_toward_camera_base": np.array([0.0, 0.0, 1.0]),
            "diameter_mm": 68.5,
            "top_z_mm": 6.0,
        }
        nominal, _ = module._cad_hole_entry_plan(
            hole, np.eye(4, dtype=np.float64), handeye, fixed_rz_rad=0.0, fine_height_mm=260.0,
        )
        corrected, base_z_delta, axis_z = module._apply_shared_cad_depth_height_offset(
            nominal, handeye.T_tcp_rgb_camera, depth_height_offset_mm=10.0,
        )
        actual_plane = hole["point_base_mm"] + 10.0 * module.camera_transform(
            nominal, handeye.T_tcp_rgb_camera,
        )[:3, 2]
        self.assertAlmostEqual(
            module.camera_height_to_plane_mm(corrected, handeye.T_tcp_rgb_camera, actual_plane),
            260.0,
            places=6,
        )
        self.assertAlmostEqual(corrected[0, 3], nominal[0, 3], places=6)
        self.assertAlmostEqual(corrected[1, 3], nominal[1, 3], places=6)
        self.assertAlmostEqual(base_z_delta * axis_z, 10.0, places=6)

    def test_cad_depth_policy_uses_single_hole_and_clamped_multi_hole_gates(self):
        self.assertEqual(module._cad_depth_policy(1, 4), ("single_hole", 1))
        self.assertEqual(module._cad_depth_policy(2, 4), ("multi_hole_shared", 2))
        self.assertEqual(module._cad_depth_policy(3, 4), ("multi_hole_shared", 3))
        self.assertEqual(module._cad_depth_policy(3, 1), ("multi_hole_shared", 3))
        self.assertEqual(module._cad_depth_policy(11, 4), ("multi_hole_shared", 4))
        with self.assertRaises(ValueError):
            module._cad_depth_policy(0, 4)

    def test_cad_fine_visualization_saves_overlay_and_final_result(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        detection = {
            "box": [600.0, 320.0, 680.0, 400.0],
            "center": [640.0, 360.0],
            "confidence": 0.91,
        }
        with tempfile.TemporaryDirectory() as directory:
            frame_path = module._save_cad_fine_overlay(
                Path(directory) / "CAD-01" / "frame_00.png",
                image,
                hole_id="CAD-01",
                frame_index=0,
                expected_center_px=np.array([642.0, 358.0]),
                detection=detection,
                all_detections=[detection],
                median_center_px=np.array([641.0, 359.0]),
                expected_radius_px=40.0,
                distance_px=2.83,
                valid_frames=12,
                total_frames=12,
                scatter_p95_px=0.31,
                mean_distance_px=1.10,
                summary=True,
            )
            self.assertIsNotNone(frame_path)
            self.assertTrue(Path(frame_path).is_file())

            result_path = module._annotate_cad_fine_result_overlay(
                frame_path,
                "CAD-01",
                {"xy_correction_mm": np.array([0.5, -0.2])},
                {"target_base_z_mm": 6.0},
                {"delta_base_y_mm": 0.3},
            )
            self.assertIsNotNone(result_path)
            result_image = cv2.imread(str(result_path), cv2.IMREAD_COLOR)
            self.assertIsNotNone(result_image)
            self.assertEqual(tuple(result_image.shape), tuple(image.shape))


if __name__ == "__main__":
    unittest.main()
