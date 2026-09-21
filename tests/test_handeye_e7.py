#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手眼诊断/E7正式验证的纯离线测试；不得连接相机或机械臂。"""

from __future__ import annotations

from copy import deepcopy
import json
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
from aubo_workbench.config import CAMERA_CFG, ROBOT_CAMERA_INTEGRATION_CFG, SOLVE_CFG
from aubo_workbench.e7_handeye import (
    assess_board_fixity,
    assess_e7_dataset,
    deterministic_e7_split,
    get_or_create_fixed_e7_split,
    run_e7_cross_validation,
    view_coverage_report,
)
from aubo_workbench.gui_handeye import _format_e7_candidate_summary
from aubo_workbench.geometry import average_transforms, invert_transform, make_transform, rotation_error_deg
from aubo_workbench.camera import CameraIntrinsics
from aubo_workbench.samples import CalibSample, archive_samples, load_existing_samples
from aubo_workbench.solve import pose_coverage_report, solve_and_save


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
                "actual_tcp_offset_sdk_m_rad": [0.00001, 0.00002, 0.21030, 3.139, 0.002, 1.5719],
                "configured_tcp_offset_sdk_m_rad": [0.00001, 0.00002, 0.21030, 3.139, 0.002, 1.5719],
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
    def test_rotation_mean_across_pi_is_near_pi_and_frame_invariant(self):
        transforms = [_transform((0.0, 0.0, np.deg2rad(a)), (a, 0.0, 0.0)) for a in (179.0, -179.0)]
        mean = average_transforms(transforms)
        expected = _transform((0.0, 0.0, np.pi), (0.0, 0.0, 0.0))
        self.assertLess(rotation_error_deg(expected[:3, :3], mean[:3, :3]), 1e-6)
        frame = _transform((0.4, -0.2, 0.1), (20.0, 30.0, -10.0))
        self.assertTrue(np.allclose(average_transforms([frame @ t for t in transforms]), frame @ mean))

    def test_fixity_rejects_equal_radius_position_errors(self):
        samples = [SimpleNamespace(
            T_base_tool=np.eye(4),
            T_rgb_board=_transform((0.0, 0.0, 0.0), (x, 0.0, 500.0)),
            calibration_frame="rgb_camera", camera_metadata={},
        ) for x in (-10.0, 10.0)]
        report = assess_board_fixity(samples, T_pose_source_sensor=np.eye(4))
        self.assertFalse(report["board_position_stable"])
        self.assertAlmostEqual(report["position_scatter_rms_mm"], 10.0)

    def test_fixity_rotation_is_order_independent_and_checks_maximum(self):
        samples = [SimpleNamespace(
            T_base_tool=np.eye(4),
            T_rgb_board=_transform((0.0, 0.0, np.deg2rad(a)), (0.0, 0.0, 500.0)),
            calibration_frame="rgb_camera", camera_metadata={},
        ) for a in ([0.0] * 19 + [1.0])]
        report = assess_board_fixity(samples, T_pose_source_sensor=np.eye(4))
        reversed_report = assess_board_fixity(list(reversed(samples)), T_pose_source_sensor=np.eye(4))
        self.assertLess(report["orientation_scatter_rms_deg"], 0.3)
        self.assertGreater(report["orientation_scatter_max_deg"], 0.6)
        self.assertFalse(report["board_orientation_stable"])
        self.assertAlmostEqual(report["orientation_scatter_rms_deg"], reversed_report["orientation_scatter_rms_deg"])

    def test_good_fit_does_not_hide_shifted_holdout_or_remove_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples, true_transform = _synthetic_samples(root, count=18)
            _, preview_validation, _ = deterministic_e7_split(samples)
            for sample in preview_validation:
                # A board displacement in base coordinates must survive validation.
                base_board = sample.T_base_tool @ true_transform @ sample.T_rgb_board
                base_board[0, 3] += 10.0
                sample.T_rgb_board = invert_transform(true_transform) @ invert_transform(sample.T_base_tool) @ base_board
            original_indices = [sample.index for sample in samples]
            with patch.object(SOLVE_CFG, "output_json", str(root / "result.json")):
                result = solve_and_save(samples)
            self.assertLess(result["quality"]["translation_rmse_mm"], 1e-4)
            self.assertAlmostEqual(result["validation_quality"]["translation_rmse_mm"], 10.0, places=3)
            self.assertEqual(result["conclusion"], "未达标")
            self.assertTrue(result["do_not_use_for_motion"])
            self.assertEqual([sample.index for sample in samples], original_indices)
            self.assertEqual(set(result["used_sample_indices"]) | set(result["validation_sample_indices"]), set(original_indices))
            self.assertNotIn("excluded_sample_indices", result)

    def test_fixed_split_is_reused_after_calibration_samples_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            samples, _ = _synthetic_samples(Path(temporary), count=18)
            calibration_a, validation_a, report_a = get_or_create_fixed_e7_split(samples)
            self.assertTrue(Path(report_a["split_path"]).is_file())
            removed_calibration = calibration_a[0].index
            reduced = [sample for sample in samples if sample.index != removed_calibration]
            calibration_b, validation_b, report_b = get_or_create_fixed_e7_split(reduced)
            self.assertEqual(
                [sample.index for sample in validation_a],
                [sample.index for sample in validation_b],
            )
            self.assertNotIn(removed_calibration, [sample.index for sample in calibration_b])
            self.assertEqual(report_a["validation_sample_indices"], report_b["validation_sample_indices"])

    def test_fixed_split_rejects_same_index_with_replaced_timestamp_or_pose(self):
        with tempfile.TemporaryDirectory() as temporary:
            samples, _ = _synthetic_samples(Path(temporary), count=18)
            _, validation, _ = get_or_create_fixed_e7_split(samples)
            replaced_index = validation[0].index
            replacement = deepcopy(next(sample for sample in samples if sample.index == replaced_index))
            replacement.timestamp = "replaced-timestamp"
            replacement.T_base_tool = replacement.T_base_tool.copy()
            replacement.T_base_tool[0, 3] += 1.0
            changed = [
                replacement if sample.index == replaced_index else sample
                for sample in samples
            ]
            with self.assertRaisesRegex(RuntimeError, "内容已变化或被替换"):
                get_or_create_fixed_e7_split(changed)

    def test_fixed_split_rejects_rewritten_original_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            samples, _ = _synthetic_samples(Path(temporary), count=18)
            _, validation, _ = get_or_create_fixed_e7_split(samples)
            path = Path(validation[0].rgb_path)
            path.write_bytes(path.read_bytes() + b"-rewritten")
            with self.assertRaisesRegex(RuntimeError, "内容已变化或被替换"):
                get_or_create_fixed_e7_split(samples)

    def test_fixed_split_rejects_legacy_index_only_record_and_duplicate_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            samples, _ = _synthetic_samples(Path(temporary), count=18)
            _, _, report = get_or_create_fixed_e7_split(samples)
            split_path = Path(report["split_path"])
            payload = json.loads(split_path.read_text(encoding="utf-8"))
            payload.pop("validation_sample_records", None)
            split_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "仅index记录不能静默信任"):
                get_or_create_fixed_e7_split(samples)

            split_path.unlink()
            duplicate = [*samples, deepcopy(samples[0])]
            with self.assertRaisesRegex(RuntimeError, "重复index"):
                get_or_create_fixed_e7_split(duplicate)

    def test_fixed_split_survives_active_data_path_relocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            old_root = Path(temporary) / "old_capture"
            new_root = Path(temporary) / "moved_capture"
            samples, _ = _synthetic_samples(old_root, count=18)
            _, validation, report = get_or_create_fixed_e7_split(samples)
            new_root.mkdir()
            (old_root / "samples").rename(new_root / "samples")
            (old_root / "images").rename(new_root / "images")
            Path(report["split_path"]).rename(new_root / "e7_validation_split_current.json")
            with patch.object(CAMERA_CFG, "save_dir", str(new_root)):
                _, moved_validation, moved_report = get_or_create_fixed_e7_split(samples)
            self.assertEqual(
                [sample.index for sample in validation],
                [sample.index for sample in moved_validation],
            )
            self.assertEqual(moved_report["split_source"], "persisted")

    def test_preflight_reuses_fixed_calibration_after_calibration_sample_removal(self):
        with tempfile.TemporaryDirectory() as temporary:
            samples, true_transform = _synthetic_samples(Path(temporary), count=18)
            calibration, validation, _ = get_or_create_fixed_e7_split(samples)
            removed_index = calibration[0].index
            reduced = [sample for sample in samples if sample.index != removed_index]
            seen_indices = []

            def preliminary_estimate(items, quiet=False):
                seen_indices.append([int(sample.index) for sample in items])
                return {"T_final": true_transform}

            with patch(
                "aubo_workbench.e7_handeye.solve_handeye_estimate",
                side_effect=preliminary_estimate,
            ):
                report = assess_e7_dataset(reduced, fixed_board_confirmed=True)

            expected = [
                int(sample.index)
                for sample in reduced
                if sample.index in {item.index for item in calibration}
            ]
            self.assertEqual(seen_indices, [expected])
            self.assertTrue(set(seen_indices[0]).isdisjoint({sample.index for sample in validation}))
            self.assertEqual(report["board_fixity_preliminary_fit_scope"], "persisted_fixed_calibration")


    def test_pose_coverage_reports_nearly_single_axis_rotation(self):
        with tempfile.TemporaryDirectory() as temporary:
            samples, _ = _synthetic_samples(Path(temporary), count=12)
            for i, sample in enumerate(samples):
                angle = np.deg2rad(float(i * 12.0))
                sample.T_base_tool[:3, :3] = np.asarray([
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ])
            report = pose_coverage_report(samples)
        self.assertIn("relative_rotation_axes_nearly_collinear", report["warnings"])





    def test_gui_formats_flat_e7_candidate_without_legacy_nested_key(self):
        text = _format_e7_candidate_summary({
            "validation_center_scatter_rms_mm": 265.43408684606925,
            "validation_numeric_pass": False,
        })
        self.assertIn("265.434 mm", text)
        self.assertIn("独立验证未达标", text)

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
            self.assertEqual(result["conclusion"], "待验证")
            self.assertIsNone(result["validation_quality"])

    def test_solver_uses_fixed_calibration_only_and_keeps_primary_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples, true_transform = _synthetic_samples(root, count=18)
            calibration, validation, _ = get_or_create_fixed_e7_split(samples)
            output = root / "handeye_diagnostic_current.json"
            with patch.object(SOLVE_CFG, "output_json", str(output)):
                result = solve_and_save(samples)
            self.assertIsNotNone(result)
            self.assertEqual(
                result["used_sample_indices"],
                [int(sample.index) for sample in calibration],
            )
            self.assertEqual(
                result["validation_sample_indices"],
                [int(sample.index) for sample in validation],
            )
            self.assertTrue(result["validation_poses_excluded_from_fit"])
            self.assertNotIn("sample_conflict_diagnosis", result)
            self.assertTrue(np.allclose(result["T_tcp_rgb_camera"], true_transform, atol=1e-5))

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
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            ROBOT_CAMERA_INTEGRATION_CFG,
            "production_camera_serial",
            "SYNTHETIC-CAMERA-001",
        ):
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

            tampered_max = deepcopy(reviewed)
            tampered_max["validation_center_scatter_max_mm"] += 0.01
            rejected_max = assess_handeye_cross_validation(tampered_max)
            self.assertFalse(rejected_max["ok"])
            self.assertIn("validation_result_rows_match_reported_max", rejected_max["failed_checks"])

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
