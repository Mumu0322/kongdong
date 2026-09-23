"""保留手眼诊断和采集安全检查的离线回归测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from aubo_workbench.camera import CameraIntrinsics
from aubo_workbench.capture import board_view_metadata, pose_bracket_report
from aubo_workbench.charuco_detect import estimate_rgb_board_pose
from aubo_workbench.config import CAMERA_CFG, SOLVE_CFG
from aubo_workbench.geometry import invert_transform, make_transform
from aubo_workbench.handeye_consistency import tcp_offset_consistency
from aubo_workbench.samples import CalibSample, archive_samples
from aubo_workbench.solve import solve_and_save


def _transform(rotation_vector, translation) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rotation_vector, dtype=np.float64))
    return make_transform(rotation, np.asarray(translation, dtype=np.float64))


def _samples(count: int = 8) -> tuple[list[CalibSample], np.ndarray]:
    true_handeye = _transform((0.11, -0.07, 0.05), (32.0, -18.0, 91.0))
    fixed_board = _transform((-0.08, 0.04, 0.03), (420.0, 35.0, 560.0))
    samples = []
    for index in range(count):
        angle = float(index) * 0.37
        T_base_tcp = _transform(
            (0.42 * np.sin(angle) + 0.03 * index,
             0.36 * np.cos(angle * 0.73) - 0.02 * index,
             -0.31 + 0.045 * index),
            (160.0 + 38.0 * np.sin(angle * 0.81),
             -95.0 + 44.0 * np.cos(angle * 0.59),
             310.0 + 6.0 * index + 24.0 * np.sin(angle * 0.41)),
        )
        T_rgb_board = invert_transform(true_handeye) @ invert_transform(T_base_tcp) @ fixed_board
        offset = [0.00001, 0.00002, 0.21030, 3.139, 0.002, 1.5719]
        samples.append(CalibSample(
            index=index + 1,
            timestamp=str(index + 1),
            T_base_tool=T_base_tcp,
            T_pointcloud_board=None,
            robot_snapshot={
                "pose_source": "tcp",
                "actual_tcp_offset_sdk_m_rad": offset.copy(),
                "configured_tcp_offset_sdk_m_rad": offset.copy(),
            },
            board_status="ok",
            charuco_count=84,
            valid_3d_count=120,
            corner_rmse_mm=0.1,
            corner_max_error_mm=0.2,
            plane_rmse_mm=0.1,
            plane_inlier_count=500,
            board_mask_point_count=1000,
            calibration_frame="rgb_camera",
            T_rgb_board=T_rgb_board,
        ))
    return samples, true_handeye


class HandeyeDiagnosticsTests(unittest.TestCase):
    def test_rgb_diagnostic_uses_all_samples_without_motion_authorization(self) -> None:
        samples, expected = _samples()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "diagnostic.json"
            with patch.object(SOLVE_CFG, "output_json", str(output)):
                result = solve_and_save(samples)
            self.assertTrue(output.is_file())
        self.assertIsNotNone(result)
        self.assertEqual(result["used_sample_indices"], list(range(1, 9)))
        self.assertEqual(result["record_type"], "handeye_rgb_diagnostic_result")
        self.assertTrue(np.allclose(result["T_tcp_rgb_camera"], expected, atol=1e-4))
        self.assertFalse(result["validated"])
        self.assertTrue(result["do_not_use_for_motion"])

    def test_diagnostic_rejects_mixed_pose_sources(self) -> None:
        samples, _ = _samples()
        samples[-1].robot_snapshot["pose_source"] = "tool"
        with self.assertRaisesRegex(RuntimeError, "位姿源必须唯一"):
            solve_and_save(samples)

    def test_tcp_change_is_reported(self) -> None:
        samples, _ = _samples()
        samples[-1].robot_snapshot["actual_tcp_offset_sdk_m_rad"][0] += 0.001
        report = tcp_offset_consistency(samples)
        self.assertTrue(report["actual_metadata_complete"])
        self.assertFalse(report["actual_offset_consistent"])
        self.assertFalse(report["actual_configured_offsets_match"])

    def test_rgb_pnp_needs_no_depth(self) -> None:
        object_points = np.asarray(
            [[30.0 * x, 30.0 * y, 0.0] for y in range(8) for x in range(11)],
            dtype=np.float64,
        )

        class SyntheticBoard:
            def getChessboardCorners(self):
                return object_points.astype(np.float32)

        intrinsics = CameraIntrinsics(
            width=1280, height=720, fx=920.0, fy=918.0, cx=640.0, cy=360.0,
            distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
        )
        expected = _transform((0.17, -0.11, 0.06), (-140.0, -90.0, 620.0))
        rvec, _ = cv2.Rodrigues(expected[:3, :3])
        image_points, _ = cv2.projectPoints(
            object_points, rvec, expected[:3, 3],
            intrinsics.camera_matrix(), np.zeros((5, 1), dtype=np.float64),
        )
        detected = (
            None, None, image_points.astype(np.float32),
            np.arange(object_points.shape[0], dtype=np.int32).reshape(-1, 1), 44,
        )
        image = np.full((720, 1280, 3), 128, dtype=np.uint8)
        with patch("aubo_workbench.charuco_detect.detect_charuco", return_value=detected):
            result = estimate_rgb_board_pose(image, intrinsics, SyntheticBoard(), None)
        self.assertTrue(result.ok, result.status)
        self.assertIsNone(result.T_pointcloud_board)
        self.assertEqual(result.rgb_pnp_inlier_count, 88)
        self.assertTrue(np.allclose(result.T_rgb_board, expected, atol=1e-4))

    def test_board_region_and_pose_bracket_use_measurements(self) -> None:
        pose = SimpleNamespace(image_points=np.asarray([[620.0, 340.0], [660.0, 380.0]]))
        self.assertEqual(board_view_metadata(pose, (720, 1280, 3))["view_region"], "center")
        pose.image_points = np.asarray([[80.0, 300.0], [120.0, 340.0]])
        self.assertEqual(board_view_metadata(pose, (720, 1280, 3))["view_region"], "left")
        before = {"pose_values": [0.0, 0.0, 0.0, 179.999, 0.0, 0.0]}
        after_ok = {"pose_values": [0.04, 0.01, 0.0, -179.999, 0.009, 0.0]}
        after_bad = {"pose_values": [0.051, 0.0, 0.0, 179.999, 0.0, 0.0]}
        self.assertTrue(pose_bracket_report(before, after_ok)["ok"])
        self.assertFalse(pose_bracket_report(before, after_bad)["ok"])

    def test_archiving_samples_keeps_external_files(self) -> None:
        samples, _ = _samples(1)
        with tempfile.TemporaryDirectory() as active_dir, tempfile.TemporaryDirectory() as external_dir:
            external = Path(external_dir) / "must_remain.png"
            external.write_bytes(b"external-user-file")
            samples[0].rgb_path = str(external)
            with patch.object(CAMERA_CFG, "save_dir", active_dir):
                archive_samples(samples, [], reason="external_path_safety_test")
            self.assertEqual(external.read_bytes(), b"external-user-file")
