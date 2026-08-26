#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手眼诊断/E7正式验证的纯离线测试；不得连接相机或机械臂。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from aubo_workbench.calibration_readiness import assess_handeye_cross_validation
from aubo_workbench.capture import board_view_metadata, pose_bracket_report
from aubo_workbench.charuco_detect import estimate_rgb_board_pose
from aubo_workbench.config import AUTO_PRUNE_CFG, CAMERA_CFG, ROBOT_CAMERA_INTEGRATION_CFG, SOLVE_CFG
from aubo_workbench.e7_handeye import (
    assess_e7_dataset,
    deterministic_e7_split,
    run_e7_cross_validation,
)
from aubo_workbench.gui_handeye import _format_e7_candidate_summary
from aubo_workbench.geometry import invert_transform, make_transform
from aubo_workbench.camera import CameraIntrinsics
from aubo_workbench.samples import CalibSample, archive_samples, load_existing_samples
from aubo_workbench.solve import auto_prune_samples, solve_and_save


def _transform(rotation_vector: tuple[float, float, float], translation: tuple[float, float, float]) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rotation_vector, dtype=np.float64))
    return make_transform(rotation, np.asarray(translation, dtype=np.float64))


def _write_raw_files(root: Path, index: int) -> dict[str, str]:
    sample_dir = root / "samples"
    image_dir = root / "images"
    sample_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sample_json_path": sample_dir / f"sample_{index:03d}_synthetic.json",
        "rgb_path": image_dir / f"sample_{index:03d}_synthetic_rgb.png",
        "overlay_path": image_dir / f"sample_{index:03d}_synthetic_overlay.png",
        "depth_vis_path": image_dir / f"sample_{index:03d}_synthetic_depth_vis.png",
    }
    for kind, path in paths.items():
        path.write_bytes(f"synthetic-e7-{index}-{kind}".encode("ascii"))
    return {key: str(path) for key, path in paths.items()}


def _synthetic_samples(root: Path, count: int = 30) -> tuple[list[CalibSample], np.ndarray]:
    true_T_tcp_rgb = _transform((0.11, -0.07, 0.05), (32.0, -18.0, 91.0))
    fixed_T_base_board = _transform((-0.08, 0.04, 0.03), (420.0, 35.0, 560.0))
    regions = ("center", "left", "right", "top", "bottom")
    region_centers = {
        "center": [640.0, 360.0],
        "left": [180.0, 360.0],
        "right": [1100.0, 360.0],
        "top": [640.0, 100.0],
        "bottom": [640.0, 620.0],
    }
    samples: list[CalibSample] = []
    for offset in range(count):
        index = offset + 1
        angle = float(offset) * 0.37
        rotation_vector = (
            0.42 * np.sin(angle) + 0.03 * offset,
            0.36 * np.cos(angle * 0.73) - 0.02 * offset,
            -0.31 + 0.045 * offset,
        )
        translation = (
            160.0 + 38.0 * np.sin(angle * 0.81),
            -95.0 + 44.0 * np.cos(angle * 0.59),
            310.0 + 6.0 * offset + 24.0 * np.sin(angle * 0.41),
        )
        T_base_tool = _transform(rotation_vector, translation)
        T_rgb_board = (
            invert_transform(true_T_tcp_rgb)
            @ invert_transform(T_base_tool)
            @ fixed_T_base_board
        )
        raw_paths = _write_raw_files(root, index)
        pose_values = [*translation, *(np.rad2deg(rotation_vector))]
        region = regions[offset % len(regions)]
        samples.append(CalibSample(
            index=index,
            timestamp=f"20260717_1200{index:02d}",
            T_base_tool=T_base_tool,
            T_pointcloud_board=None,
            robot_snapshot={
                "robot_brand": "AUBO",
                "robot_name": "aubo-i5h-01",
                "robot_type": "i5H",
                "pose_source": "tcp",
                "pose_values": pose_values,
                "camera_frame_pose_bracket": {
                    "ok": True,
                    "xyz_delta_mm": [0.01, 0.0, 0.0],
                    "abc_delta_deg": [0.002, 0.0, 0.0],
                    "before_snapshot": {"pose_values": pose_values},
                    "after_snapshot": {"pose_values": pose_values},
                },
            },
            board_status="ok",
            charuco_count=84,
            valid_3d_count=120,
            corner_rmse_mm=0.10,
            corner_max_error_mm=0.20,
            plane_rmse_mm=0.10,
            plane_inlier_count=500,
            board_mask_point_count=1000,
            camera_metadata={
                "width": 1280,
                "height": 720,
                "color_timestamp_us": 1_000_000 + index * 33_333,
                "depth_timestamp_us": 1_000_000 + index * 33_333,
                "host_timestamp_ns": 1_000_000_000 + index * 33_333_000,
                "color_frame_index": index,
                "depth_frame_index": index,
                "capture_mode": "rgb_only_no_depth_or_pointcloud",
                "intrinsics": {
                    "width": 1280, "height": 720,
                    "fx": 900.0, "fy": 900.0, "cx": 640.0, "cy": 360.0,
                    "distortion": [0.0] * 8,
                },
                "depth_processing": {"mode": "none", "order": ["align_depth_to_color", "point_cloud_from_processed_depth"]},
                "device": {
                    "name": "Gemini 435Le",
                    "serial_number": ROBOT_CAMERA_INTEGRATION_CFG.production_camera_serial,
                    "pid": "435",
                    "connection_type": "USB",
                },
            },
            capture_quality={
                "ok": True,
                "score": 98.0,
                "label": "合格",
                "reasons": [],
                "advice": [],
                "brightness": 110.0,
                "contrast": 45.0,
                "sharpness": 180.0,
                "board_area_ratio": 0.15,
                "center_offset_ratio": 0.10,
                "charuco_count": 84,
                "valid_3d_count": 120,
                "corner_rmse_mm": 0.10,
                "corner_max_error_mm": 0.20,
                "plane_rmse_mm": 0.10,
                "calibration_frame": "rgb_camera",
                "rgb_pnp_inlier_count": 84,
                "rgb_reprojection_rmse_px": 0.10,
                "rgb_reprojection_max_px": 0.25,
            },
            board_center_uv=region_centers[region],
            image_size_wh=[1280, 720],
            view_region=region,
            calibration_frame="rgb_camera",
            T_rgb_board=T_rgb_board,
            rgb_pnp_inlier_count=84,
            rgb_reprojection_rmse_px=0.10,
            rgb_reprojection_max_px=0.25,
            **raw_paths,
        ))
    return samples, true_T_tcp_rgb


class HandEyeE7Tests(unittest.TestCase):
    def test_auto_prune_can_remove_from_eighteen_down_to_e7_minimum(self):
        with tempfile.TemporaryDirectory() as temporary:
            samples, _ = _synthetic_samples(Path(temporary), count=18)
            samples[2].T_base_tool = np.eye(4, dtype=np.float64)
            samples[2].robot_snapshot["pose_values"] = [0.0] * 6
            remaining, removed, _ = auto_prune_samples(samples)
            self.assertEqual(AUTO_PRUNE_CFG.min_remaining_samples, 11)
            self.assertIn(3, [int(item["index"]) for item in removed])
            self.assertNotIn(3, [sample.index for sample in remaining])

    def test_gui_formats_flat_e7_candidate_without_legacy_nested_key(self):
        text = _format_e7_candidate_summary({
            "validation_center_scatter_rms_mm": 265.43408684606925,
            "validation_numeric_pass": False,
        })
        self.assertIn("265.434087 mm", text)
        self.assertIn("数值验证=不通过", text)

    def test_rgb_diagnostic_output_uses_rgb_coordinate_names_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples, true_transform = _synthetic_samples(root, count=8)
            output = root / "handeye_diagnostic_current.json"
            with patch.object(SOLVE_CFG, "output_json", str(output)):
                result = solve_and_save(samples)
            self.assertIsNotNone(result)
            self.assertEqual(result["record_type"], "handeye_rgb_diagnostic_result")
            self.assertEqual(result["calibration_frame"], "rgb_camera")
            self.assertTrue(np.allclose(result["T_tcp_rgb_camera"], true_transform, atol=1e-5))
            self.assertNotIn("T_tcp_pointcloud", result)
            self.assertNotIn("T_tool_pointcloud", result)
            self.assertTrue(output.is_file())

    def test_rgb_charuco_pnp_recovers_board_pose_without_depth_input(self):
        object_points = np.asarray(
            [[30.0 * x, 30.0 * y, 0.0] for y in range(8) for x in range(11)],
            dtype=np.float64,
        )

        class SyntheticBoard:
            def getChessboardCorners(self):
                return object_points.astype(np.float32)

        intrinsics = CameraIntrinsics(
            width=1280, height=720,
            fx=920.0, fy=918.0, cx=640.0, cy=360.0,
            distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
        )
        true_transform = _transform((0.17, -0.11, 0.06), (-140.0, -90.0, 620.0))
        rvec, _ = cv2.Rodrigues(true_transform[:3, :3])
        image_points, _ = cv2.projectPoints(
            object_points, rvec, true_transform[:3, 3],
            intrinsics.camera_matrix(), np.zeros((5, 1), dtype=np.float64),
        )
        detected = (
            None,
            None,
            image_points.astype(np.float32),
            np.arange(object_points.shape[0], dtype=np.int32).reshape(-1, 1),
            44,
        )
        image = np.full((720, 1280, 3), 128, dtype=np.uint8)
        with patch("aubo_workbench.charuco_detect.detect_charuco", return_value=detected):
            result = estimate_rgb_board_pose(image, intrinsics, SyntheticBoard(), None)

        self.assertTrue(result.ok, result.status)
        self.assertEqual(result.calibration_frame, "rgb_camera")
        self.assertIsNone(result.T_pointcloud_board)
        self.assertEqual(result.rgb_pnp_inlier_count, 88)
        self.assertLess(result.rgb_reprojection_rmse_px, 1e-3)
        self.assertTrue(np.allclose(result.T_rgb_board, true_transform, atol=1e-4))

    def test_board_region_and_frame_pose_bracket_are_computed_from_measurements(self):
        pose = SimpleNamespace(image_points=np.asarray([[620.0, 340.0], [660.0, 380.0]]))
        center = board_view_metadata(pose, (720, 1280, 3))
        self.assertEqual(center["view_region"], "center")
        pose.image_points = np.asarray([[80.0, 300.0], [120.0, 340.0]])
        left = board_view_metadata(pose, (720, 1280, 3))
        self.assertEqual(left["view_region"], "left")

        before = {"pose_values": [0.0, 0.0, 0.0, 179.999, 0.0, 0.0]}
        after_ok = {"pose_values": [0.04, 0.01, 0.0, -179.999, 0.009, 0.0]}
        after_bad = {"pose_values": [0.051, 0.0, 0.0, 179.999, 0.0, 0.0]}
        self.assertTrue(pose_bracket_report(before, after_ok)["ok"])
        self.assertFalse(pose_bracket_report(before, after_bad)["ok"])

    def test_e7_uses_preassigned_independent_validation_and_writes_unvalidated_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples, true_transform = _synthetic_samples(root)
            output = root / "candidate" / "e7_handeye_candidate_current.json"
            result = run_e7_cross_validation(
                samples,
                fixed_board_confirmed=True,
                candidate_path=output,
            )

            self.assertTrue(output.is_file())
            self.assertFalse(result["validated"])
            self.assertTrue(result["do_not_use_for_motion"])
            self.assertFalse(result["production_eligible"])
            self.assertTrue(result["split_assignment_before_fit"])
            self.assertFalse(result["split_assignment_uses_measurement_residuals"])
            self.assertFalse(
                set(result["calibration_sample_indices"])
                & set(result["validation_sample_indices"])
            )
            self.assertEqual(result["validation_reference_source"], "calibration_set_T_base_board_mean")
            self.assertLess(result["validation_center_scatter_rms_mm"], 1e-5)
            self.assertEqual(result["calibration_frame"], "rgb_camera")
            self.assertTrue(np.allclose(result["T_tcp_rgb_camera"], true_transform, atol=1e-5))

            blocked = assess_handeye_cross_validation(result)
            self.assertFalse(blocked["ok"])
            self.assertIn("marked_validated", blocked["failed_checks"])

            reviewed = deepcopy(result)
            reviewed["validated"] = True
            reviewed["do_not_use_for_motion"] = False
            reviewed["production_eligible"] = True
            reviewed["camera_mount_id"] = "gemini-mount-01"
            reviewed["verified_by"] = "independent-reviewer"
            accepted = assess_handeye_cross_validation(reviewed)
            self.assertTrue(accepted["ok"], accepted)

            tampered = deepcopy(reviewed)
            tampered["validation_center_scatter_rms_mm"] += 0.01
            rejected = assess_handeye_cross_validation(tampered)
            self.assertFalse(rejected["ok"])
            self.assertIn("validation_result_rows_match_reported_rms", rejected["failed_checks"])

    def test_split_is_deterministic_and_does_not_read_residuals(self):
        with tempfile.TemporaryDirectory() as temporary:
            samples, _ = _synthetic_samples(Path(temporary))
            calibration_a, validation_a, report_a = deterministic_e7_split(samples)
            for sample in samples:
                sample.corner_rmse_mm = 0.01 * sample.index
                sample.plane_rmse_mm = 0.02 * sample.index
            calibration_b, validation_b, report_b = deterministic_e7_split(list(reversed(samples)))
            self.assertEqual([sample.index for sample in calibration_a], [sample.index for sample in calibration_b])
            self.assertEqual([sample.index for sample in validation_a], [sample.index for sample in validation_b])
            self.assertEqual(report_a, report_b)
            self.assertFalse(report_a["assignment_uses_measurement_residuals"])

    def test_e7_rejects_manual_pose_and_missing_frame_bracket(self):
        with tempfile.TemporaryDirectory() as temporary:
            samples, _ = _synthetic_samples(Path(temporary))
            samples[0].robot_snapshot["pose_source"] = "manual"
            samples[1].robot_snapshot.pop("camera_frame_pose_bracket")
            report = assess_e7_dataset(samples, fixed_board_confirmed=True)
            self.assertFalse(report["ok"])
            self.assertFalse(report["checks"]["all_samples_use_tcp_pose"])
            self.assertFalse(report["checks"]["manual_pose_not_used"])
            self.assertFalse(report["checks"]["camera_frame_pose_bracket_verified"])

    def test_e7_rejects_all_zero_robot_pose(self):
        with tempfile.TemporaryDirectory() as temporary:
            samples, _ = _synthetic_samples(Path(temporary))
            sample = samples[0]
            sample.T_base_tool = np.eye(4, dtype=np.float64)
            sample.robot_snapshot["pose_values_sdk_m_rad"] = [0.0] * 6
            bracket = sample.robot_snapshot["camera_frame_pose_bracket"]
            bracket["before_snapshot"]["pose_values_sdk_m_rad"] = [0.0] * 6
            bracket["after_snapshot"]["pose_values_sdk_m_rad"] = [0.0] * 6
            report = assess_e7_dataset(samples, fixed_board_confirmed=True)
            self.assertFalse(report["checks"]["all_robot_pose_values_valid"])
            self.assertFalse(report["checks"]["camera_frame_pose_bracket_verified"])

    def test_e7_total_pose_threshold_is_more_than_ten(self):
        with tempfile.TemporaryDirectory() as temporary:
            samples, _ = _synthetic_samples(Path(temporary), count=11)
            report_10 = assess_e7_dataset(samples[:10], fixed_board_confirmed=True)
            report_11 = assess_e7_dataset(samples, fixed_board_confirmed=True)
            self.assertFalse(report_10["checks"]["minimum_total_poses"])
            self.assertTrue(report_11["checks"]["minimum_total_poses"])
            calibration, validation, _ = deterministic_e7_split(samples)
            self.assertEqual(len(calibration), 8)
            self.assertEqual(len(validation), 3)

    def test_user_remove_archives_disk_files_and_rewrites_active_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples, _ = _synthetic_samples(root, count=1)
            sample = samples[0]
            with patch.object(CAMERA_CFG, "save_dir", str(root)):
                archive_dir = archive_samples([sample], [], reason="unit_test_remove")
                self.assertIsNotNone(archive_dir)
                self.assertTrue((archive_dir / "archive_report.json").is_file())
                self.assertFalse(Path(sample.sample_json_path).exists())
                self.assertFalse(Path(sample.rgb_path).exists())
                self.assertFalse(Path(sample.overlay_path).exists())
                self.assertFalse(Path(sample.depth_vis_path).exists())
                self.assertEqual(load_existing_samples(), [])
                csv_text = (root / "charuco_pointcloud_samples.csv").read_text(encoding="utf-8-sig")
                self.assertNotIn("synthetic", csv_text)

    def test_archive_never_moves_a_path_outside_active_capture_directory(self):
        with tempfile.TemporaryDirectory() as active_text, tempfile.TemporaryDirectory() as external_text:
            active = Path(active_text)
            external = Path(external_text) / "must_remain.png"
            external.write_bytes(b"external-user-file")
            sample = CalibSample(
                index=1,
                timestamp="external_path_test",
                T_base_tool=np.eye(4),
                T_pointcloud_board=np.eye(4),
                robot_snapshot={"pose_source": "tcp"},
                board_status="ok",
                charuco_count=84,
                valid_3d_count=120,
                corner_rmse_mm=0.1,
                corner_max_error_mm=0.2,
                plane_rmse_mm=0.1,
                plane_inlier_count=500,
                board_mask_point_count=1000,
                rgb_path=str(external),
            )
            with patch.object(CAMERA_CFG, "save_dir", str(active)):
                archive_samples([sample], [], reason="external_path_safety_test")
            self.assertTrue(external.is_file())
            self.assertEqual(external.read_bytes(), b"external-user-file")

    def test_diagnostic_solver_rejects_mixed_pose_sources_before_solving(self):
        identity = np.eye(4, dtype=np.float64)
        samples = []
        for index in range(8):
            samples.append(CalibSample(
                index=index + 1,
                timestamp=str(index),
                T_base_tool=identity.copy(),
                T_pointcloud_board=identity.copy(),
                robot_snapshot={"pose_source": "tcp" if index < 7 else "tool"},
                board_status="ok",
                charuco_count=24,
                valid_3d_count=120,
                corner_rmse_mm=0.1,
                corner_max_error_mm=0.2,
                plane_rmse_mm=0.1,
                plane_inlier_count=500,
                board_mask_point_count=1000,
            ))
        with self.assertRaisesRegex(RuntimeError, "位姿源必须唯一"):
            solve_and_save(samples)


if __name__ == "__main__":
    unittest.main()
