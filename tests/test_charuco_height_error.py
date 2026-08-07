from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import run_charuco_height_error_experiment as experiment_module

from aubo_workbench.charuco_height_error import (
    DEFAULT_HEIGHTS_MM,
    HeightExperimentConfig,
    HeightErrorExperiment,
    RgbPrecisionRangeConfig,
    RgbPrecisionRangeExperiment,
    build_rgb_height_summaries,
    build_rgb_pose_summaries,
    build_field_position_summaries,
    build_batch_summaries,
    build_height_summaries,
    fit_tcp_rgb_geometry,
    make_height_schedule,
    normal_angle_deg,
    rgb_precision_targets,
    scalar_statistics,
    select_contiguous_height_ranges,
    transform_cross_metrics,
    trend_summary,
)
from aubo_workbench.config import BOARD_CFG
from aubo_workbench.geometry import make_transform, rotx


def _row(height: float, frame: int, rgb_z: float, depth_z: float) -> dict:
    return {
        "height_target_mm": height,
        "frame_index": frame,
        "batch_index": (frame - 1) // 10 + 1,
        "valid": True,
        "rgb_ok": True,
        "depth_ok": True,
        "rgb_reprojection_rmse_px": 0.1 + height / 10000.0,
        "rgb_reprojection_max_px": 0.3,
        "rgb_x_mm": 1.0, "rgb_y_mm": 2.0, "rgb_z_mm": rgb_z,
        "depth_x_mm": 1.2, "depth_y_mm": 1.8, "depth_z_mm": depth_z,
        "depth_corner_rmse_mm": 0.5,
        "depth_corner_max_error_mm": 1.0,
        "depth_plane_rmse_mm": 0.4,
        "cross_dx_mm": 0.2, "cross_dy_mm": -0.2,
        "cross_dz_mm": depth_z - rgb_z,
        "cross_distance_mm": float(np.linalg.norm([0.2, -0.2, depth_z - rgb_z])),
        "cross_normal_angle_deg": 0.2,
        "control_residual_mm": rgb_z - height,
        "brightness": 120.0, "contrast": 40.0, "sharpness": 200.0,
        "rgb_depth_timestamp_delta_us": 100.0,
        "tcp_x_mm": 500.0, "tcp_y_mm": 0.0, "tcp_z_mm": 300.0,
    }


class HeightErrorStatisticsTests(unittest.TestCase):
    def test_default_height_schedule(self) -> None:
        self.assertEqual(DEFAULT_HEIGHTS_MM, (300.0, 320.0, 340.0, 360.0))

    def test_board_configuration_matches_physical_board(self) -> None:
        self.assertEqual((BOARD_CFG.pattern_width_mm, BOARD_CFG.pattern_height_mm), (360.0, 270.0))
        self.assertEqual((BOARD_CFG.squares_x, BOARD_CFG.squares_y), (12, 9))
        self.assertEqual(BOARD_CFG.square_length_mm, 30.0)
        self.assertEqual(BOARD_CFG.marker_length_mm, 22.5)

    def test_scalar_statistics_and_p95(self) -> None:
        stats = scalar_statistics(range(1, 101))
        self.assertEqual(stats["count"], 100)
        self.assertAlmostEqual(stats["median"], 50.5)
        self.assertAlmostEqual(stats["p95"], 95.05)
        self.assertAlmostEqual(stats["range"], 99.0)

    def test_cross_metrics_have_expected_sign_and_angle(self) -> None:
        rgb = make_transform(np.eye(3), np.array([10.0, 20.0, 400.0]))
        depth = make_transform(rotx(np.deg2rad(2.0)), np.array([11.0, 18.0, 403.0]))
        result = transform_cross_metrics(rgb, depth)
        self.assertEqual(result["cross_dx_mm"], 1.0)
        self.assertEqual(result["cross_dy_mm"], -2.0)
        self.assertEqual(result["cross_dz_mm"], 3.0)
        self.assertAlmostEqual(result["cross_distance_mm"], np.sqrt(14.0))
        self.assertAlmostEqual(result["cross_normal_angle_deg"], 2.0, places=6)
        self.assertAlmostEqual(normal_angle_deg([0, 0, 1], [0, 0, -1]), 0.0)

    def test_batches_and_heights_group_without_dropping_frames(self) -> None:
        rows = []
        for height in (300.0, 350.0):
            rows.extend(_row(height, frame, height + 0.01 * frame, height + 1.0) for frame in range(1, 21))
        batches = build_batch_summaries(rows)
        heights = build_height_summaries(rows)
        self.assertEqual(len(batches), 4)
        self.assertTrue(all(batch["frame_count"] == 10 for batch in batches))
        self.assertEqual(len(heights), 2)
        self.assertTrue(all(item["frame_count"] == 20 for item in heights))
        self.assertTrue(all(item["valid_ratio"] == 1.0 for item in heights))

    def test_height_trend_uses_height_medians(self) -> None:
        rows = []
        for height in (300.0, 350.0, 400.0):
            rows.extend(_row(height, frame, height, height + height / 100.0) for frame in range(1, 11))
        trend = trend_summary(build_height_summaries(rows))
        self.assertAlmostEqual(trend["cross_dz_mm"]["slope_per_100mm"], 1.0, places=6)

    def test_config_rejects_incomplete_batch(self) -> None:
        with self.assertRaises(ValueError):
            HeightExperimentConfig(frames_per_height=201, batch_size=10).validate()

    def test_field_grid_groups_height_and_position(self) -> None:
        rows = []
        for position, row, col in (("R1C1", 1, 1), ("R2C2", 2, 2)):
            for frame in range(1, 6):
                item = _row(340.0, frame, 340.0, 340.5)
                item.update({
                    "grid_position": position, "grid_row": row, "grid_col": col,
                    "grid_target_u_px": 384.0, "grid_target_v_px": 240.0,
                    "rgb_center_u_px": 384.0 + frame * 0.1,
                    "rgb_center_v_px": 240.0,
                })
                rows.append(item)
        summaries = build_field_position_summaries(rows)
        self.assertEqual(len(summaries), 2)
        self.assertTrue(all(item["frame_count"] == 5 for item in summaries))

    def test_finalize_writes_json_and_csv_without_hardware(self) -> None:
        with TemporaryDirectory() as directory:
            config = HeightExperimentConfig(heights_mm=(400.0,), frames_per_height=10)
            experiment = HeightErrorExperiment(config, Path(directory))
            experiment.rows = [_row(400.0, frame, 400.0, 401.0) for frame in range(1, 11)]
            report = experiment.finalize("synthetic_test")
            report_path = experiment.output_dir / "report.json"
            self.assertTrue(report_path.is_file())
            self.assertTrue((experiment.output_dir / "frames.csv").is_file())
            self.assertTrue((experiment.output_dir / "batch_summary.csv").is_file())
            self.assertTrue((experiment.output_dir / "height_summary.csv").is_file())
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "synthetic_test")
            self.assertEqual(saved["frame_count"], 10)
            self.assertFalse(report["absolute_accuracy_claimed"])

    def test_rgb_precision_height_schedule_is_inclusive(self) -> None:
        self.assertEqual(
            make_height_schedule(200.0, 300.0, 10.0),
            tuple(float(value) for value in range(200, 301, 10)),
        )
        self.assertEqual(make_height_schedule(200.0, 225.0, 10.0), (200.0, 210.0, 220.0, 225.0))

    def test_rgb_precision_targets_use_intrinsics_and_metric_step(self) -> None:
        targets = rgb_precision_targets(
            {"fx": 600.0, "fy": 500.0, "cx": 640.0, "cy": 400.0},
            height_mm=200.0,
            center_step_mm=20.0,
        )
        self.assertEqual([item[0] for item in targets], ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"])
        self.assertEqual(targets[0][1:], (640.0, 400.0))
        self.assertEqual(targets[1][1:], (580.0, 400.0))
        self.assertEqual(targets[2][1:], (700.0, 400.0))
        self.assertEqual(targets[3][1:], (640.0, 350.0))
        self.assertEqual(targets[4][1:], (640.0, 450.0))

    def test_rgb_geometry_fit_is_zero_for_exact_rigid_mapping(self) -> None:
        tcp_points = np.asarray([
            [0.0, 0.0, 10.0], [-20.0, 0.0, 10.0], [20.0, 0.0, 10.0],
            [0.0, -20.0, 10.0], [0.0, 20.0, 10.0],
        ])
        mapping = np.diag([-1.0, 1.0, -1.0])
        rgb_points = tcp_points @ mapping + np.asarray([100.0, 200.0, 300.0])
        rows = [
            {"tcp_xyz_median_mm": tcp.tolist(), "rgb_xyz_median_mm": rgb.tolist()}
            for tcp, rgb in zip(tcp_points, rgb_points)
        ]
        result = fit_tcp_rgb_geometry(rows)
        self.assertAlmostEqual(result["distance_error_mm"]["p95"], 0.0, places=9)
        self.assertAlmostEqual(result["rigid_rms_mm"], 0.0, places=9)
        self.assertAlmostEqual(result["scale"], 1.0, places=9)

    def test_rgb_geometry_reports_known_scale_error(self) -> None:
        tcp_points = np.asarray([
            [0.0, 0.0, 0.0], [-20.0, 0.0, 0.0], [20.0, 0.0, 0.0],
            [0.0, -20.0, 0.0], [0.0, 20.0, 0.0],
        ])
        rgb_points = tcp_points * 1.01 + np.asarray([10.0, 20.0, 250.0])
        result = fit_tcp_rgb_geometry([
            {"tcp_xyz_median_mm": tcp.tolist(), "rgb_xyz_median_mm": rgb.tolist()}
            for tcp, rgb in zip(tcp_points, rgb_points)
        ])
        self.assertAlmostEqual(result["scale"], 1.01, places=9)
        self.assertAlmostEqual(result["scale_error_percent"], 1.0, places=9)
        self.assertGreater(result["distance_error_mm"]["p95"], 0.0)

    def test_rgb_precision_selects_contiguous_passing_range(self) -> None:
        summaries = [
            {"height_target_mm": 200.0, "passed": False, "distance_error_p95_mm": 0.7},
            {"height_target_mm": 210.0, "passed": True, "distance_error_p95_mm": 0.3,
             "rigid_rms_mm": 0.2, "rgb_reprojection_rmse_p95_px": 0.15},
            {"height_target_mm": 220.0, "passed": True, "distance_error_p95_mm": 0.2,
             "rigid_rms_mm": 0.1, "rgb_reprojection_rmse_p95_px": 0.14},
            {"height_target_mm": 230.0, "passed": False, "distance_error_p95_mm": 0.6},
            {"height_target_mm": 240.0, "passed": True, "distance_error_p95_mm": 0.4,
             "rigid_rms_mm": 0.2, "rgb_reprojection_rmse_p95_px": 0.16},
        ]
        result = select_contiguous_height_ranges(summaries)
        self.assertEqual(result["best_height_mm"], 220.0)
        self.assertEqual(
            result["recommended_range"],
            {"start_mm": 210.0, "stop_mm": 220.0, "heights_mm": [210.0, 220.0]},
        )

    def test_rgb_precision_height_summary_passes_exact_motion(self) -> None:
        tcp_by_pose = {
            "CENTER": [0.0, 0.0, 0.0],
            "LEFT": [-20.0, 0.0, 0.0],
            "RIGHT": [20.0, 0.0, 0.0],
            "UP": [0.0, -20.0, 0.0],
            "DOWN": [0.0, 20.0, 0.0],
        }
        rows = []
        for pose_index, (pose_name, tcp) in enumerate(tcp_by_pose.items(), 1):
            rgb = np.asarray(tcp) + np.asarray([100.0, 200.0, 250.0])
            for frame_index in range(1, 21):
                rows.append({
                    "height_target_mm": 250.0,
                    "pose_name": pose_name,
                    "pose_index": pose_index,
                    "frame_index": frame_index,
                    "valid": True,
                    "rgb_x_mm": rgb[0], "rgb_y_mm": rgb[1], "rgb_z_mm": rgb[2],
                    "tcp_x_mm": tcp[0], "tcp_y_mm": tcp[1], "tcp_z_mm": tcp[2],
                    "rgb_reprojection_rmse_px": 0.1,
                    "rgb_reprojection_max_px": 0.3,
                    "rgb_charuco_count": 88,
                    "rgb_pnp_inlier_count": 88,
                    "target_distance_px": 1.0,
                })
        config = RgbPrecisionRangeConfig(heights_mm=(250.0,))
        poses = build_rgb_pose_summaries(rows)
        heights = build_rgb_height_summaries(rows, poses, config)
        self.assertEqual(len(poses), 5)
        self.assertEqual(len(heights), 1)
        self.assertTrue(heights[0]["passed"])
        self.assertAlmostEqual(heights[0]["distance_error_rms_mm"], 0.0, places=9)
        self.assertAlmostEqual(heights[0]["distance_error_p95_mm"], 0.0, places=9)

    def test_rgb_precision_pipeline_does_not_start_depth(self) -> None:
        bundle = SimpleNamespace(
            intrinsics=SimpleNamespace(as_dict=lambda: {"fx": 600.0}),
            metadata_dict=lambda: {"capture_mode": "rgb_only_no_depth_or_pointcloud"},
        )
        motion = MagicMock()
        with TemporaryDirectory() as directory, \
                patch.object(experiment_module, "init_rgb_handeye_pipeline", return_value=object()), \
                patch.object(experiment_module, "init_pipeline", side_effect=AssertionError("depth started")), \
                patch.object(experiment_module, "get_color_profile", return_value=object()), \
                patch.object(experiment_module, "_video_profile_metadata", return_value={"stream": "color"}), \
                patch.object(experiment_module, "get_device_identity", return_value={"serial_number": "RGB"}), \
                patch.object(experiment_module, "read_and_lock_color_exposure", return_value={"locked": True}), \
                patch.object(experiment_module, "get_rgb_frame_bundle", return_value=bundle), \
                patch.object(experiment_module, "AuboMotionSession", return_value=motion):
            config = RgbPrecisionRangeConfig(
                heights_mm=(250.0,), expected_camera_serial="RGB",
            )
            experiment = RgbPrecisionRangeExperiment(config, Path(directory))
            experiment.start()
            motion.connect.assert_called_once()

    def test_rgb_precision_auto_xy_uses_eight_twenty_mm_segments(self) -> None:
        with TemporaryDirectory() as directory:
            experiment = RgbPrecisionRangeExperiment(
                RgbPrecisionRangeConfig(heights_mm=(250.0,), execute_motion=True),
                Path(directory),
            )
            experiment.motion = MagicMock()
            experiment.motion.snapshot.return_value = {
                "power_on": True,
                "collision": False,
                "steady": True,
                "tcp_pose_m_rad": [0.5, -0.1, 0.3, 0.0, 0.0, 0.0],
            }
            experiment.capture_pose = MagicMock()
            experiment.move_tcp_xy = MagicMock()
            with patch("builtins.input", return_value="m"):
                experiment.capture_auto_xy_layer(250.0)
            self.assertEqual(
                [call.args[1] for call in experiment.capture_pose.call_args_list],
                ["CENTER", "X_MINUS", "X_PLUS", "Y_MINUS", "Y_PLUS"],
            )
            offsets = [
                (call.args[1], call.args[2])
                for call in experiment.move_tcp_xy.call_args_list
            ]
            self.assertEqual(
                offsets,
                [(-20.0, 0.0), (0.0, 0.0), (20.0, 0.0), (0.0, 0.0),
                 (0.0, -20.0), (0.0, 0.0), (0.0, 20.0), (0.0, 0.0)],
            )


if __name__ == "__main__":
    unittest.main()
