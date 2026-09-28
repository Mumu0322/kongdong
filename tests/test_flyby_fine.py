from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from aubo_workbench.flyby_fine import (
    PoseSample, frame_time_ns, interpolate_pose, robust_xy, rows_from_map,
    scan_endpoints, scan_pose_waypoints, retain_nearest_frame,
    continuous_scan_waypoints, translation_speed_mm_s, validate_camera_preflight,
    build_contact_sheet, draw_flyby_overlay, thumbnail,
)


class FlybyFineTests(unittest.TestCase):
    def test_camera_preflight_rejects_unsynchronized_or_slow_frames(self) -> None:
        origin = 1_800_000_000_000_000_000
        frames = [SimpleNamespace(
            color_system_timestamp_us=(origin + index * 16_667_000) // 1000,
            host_timestamp_ns=origin + index * 16_667_000 + 10_000_000,
            color_exposure_us=5000,
        ) for index in range(12)]
        self.assertGreater(validate_camera_preflight(frames)["measured_fps"], 55)
        frames[3].color_system_timestamp_us = None
        with self.assertRaisesRegex(RuntimeError, "时间戳"):
            validate_camera_preflight(frames)
        frames[3].color_system_timestamp_us = (origin + 3 * 16_667_000) // 1000
        frames[3].color_exposure_us = 12000
        with self.assertRaisesRegex(RuntimeError, "曝光"):
            validate_camera_preflight(frames)
        self.assertIsNone(frame_time_ns(SimpleNamespace(
            color_system_timestamp_us=origin // 1000,
            host_timestamp_ns=origin + 100_000_000,
        )))

    def test_camera_preflight_accepts_small_sdk_timestamp_lead(self) -> None:
        origin = 1_800_000_000_000_000_000
        frames = [SimpleNamespace(
            color_system_timestamp_us=(origin + index * 16_667_000) // 1000,
            host_timestamp_ns=origin + index * 16_667_000 + (400_000 if index % 2 else -800_000),
            color_exposure_us=100,
        ) for index in range(12)]
        result = validate_camera_preflight(frames)
        self.assertEqual(result["clock_offset_ns"], 0)
        self.assertGreater(result["measured_fps"], 55)
        frames[0].host_timestamp_ns -= 3_000_000
        with self.assertRaisesRegex(RuntimeError, "无法对齐"):
            validate_camera_preflight(frames)

    def test_pose_interpolation_requires_bracketing_fast_reads(self) -> None:
        left = np.eye(4)
        right = np.eye(4)
        right[0, 3] = 10.0
        samples = [
            PoseSample(100_000_000, left, 2_000_000),
            PoseSample(120_000_000, right, 2_000_000),
        ]
        self.assertAlmostEqual(interpolate_pose(samples, 110_000_000)[0, 3], 5.0)
        self.assertAlmostEqual(translation_speed_mm_s(samples, 110_000_000), 500.0)
        self.assertIsNone(interpolate_pose(samples, 99_000_000))
        self.assertIsNone(interpolate_pose(samples, 130_000_000))
        self.assertIsNone(interpolate_pose([
            samples[0], PoseSample(140_000_000, right, 2_000_000),
        ], 110_000_000))
        self.assertIsNone(interpolate_pose([
            samples[0], PoseSample(120_000_000, right, 9_000_000),
        ], 110_000_000))

    def test_scan_geometry_and_row_identity(self) -> None:
        seeds = [
            {"hole_id": index + 1, "initial_center_base_mm": [index * 30., 0., 0.],
             "coarse_map_source": {"layout_row": 1, "layout_column": index + 1}}
            for index in range(3)
        ]
        self.assertEqual([item["hole_id"] for item in rows_from_map(seeds)[0]], [1, 2, 3])
        tcp = np.diag([1., -1., -1., 1.])
        start, end, geometry = scan_endpoints(seeds, tcp, np.eye(4), 300.)
        self.assertAlmostEqual(start[2, 3], 300.)
        self.assertAlmostEqual(end[0, 3] - start[0, 3], 90.)
        self.assertAlmostEqual(geometry["maximum_row_deviation_mm"], 0.)
        seeds[1]["initial_center_base_mm"][1] = 50.
        with self.assertRaisesRegex(ValueError, "偏离"):
            scan_endpoints(seeds, tcp, np.eye(4), 300.)

    def test_scatter_gate_never_turns_few_frames_into_success(self) -> None:
        self.assertFalse(robust_xy([np.zeros(2)] * 4)["success"])
        self.assertTrue(robust_xy([np.array([1., 2.])] * 5)["success"])

    def test_pose_waypoints_change_orientation_without_stopping_at_holes(self) -> None:
        poses = []
        for index in range(3):
            pose = np.eye(4)
            angle = np.radians(index * 3.0)
            pose[:3, :3] = [
                [1., 0., 0.],
                [0., np.cos(angle), -np.sin(angle)],
                [0., np.sin(angle), np.cos(angle)],
            ]
            pose[:3, 3] = [index * 70., index * 5., 300.]
            poses.append(pose)
        start, waypoints, geometry = scan_pose_waypoints(poses)
        self.assertEqual(len(waypoints), 4)
        self.assertLess(start[0, 3], poses[0][0, 3])
        self.assertGreater(waypoints[-1][0, 3], poses[-1][0, 3])
        self.assertAlmostEqual(geometry["maximum_pose_rotation_step_deg"], 3.0)
        poses[-1][:3, :3] = np.diag([1., -1., -1.])
        with self.assertRaisesRegex(ValueError, "姿态变化"):
            scan_pose_waypoints(poses)

    def test_selected_frames_are_spaced_and_prefer_target_pose(self) -> None:
        selected = []
        for index in range(12):
            retain_nearest_frame(
                selected, score=float(12 - index), timestamp_ns=index * 55_000_000,
                bundle=index, max_frames=6,
            )
        self.assertEqual(sorted(item[2] for item in selected), list(range(6, 12)))
        retain_nearest_frame(
            selected, score=0.1, timestamp_ns=11 * 55_000_000 + 10_000_000,
            bundle="closer", max_frames=6,
        )
        self.assertIn("closer", [item[2] for item in selected])
        self.assertEqual(len(selected), 6)

    def test_continuous_scan_uses_diagonal_safe_transfers(self) -> None:
        def pose(x, y, z, angle=0.0):
            value = np.eye(4)
            value[:3, 3] = [x, y, z]
            value[:3, :3] = Rotation.from_euler("z", angle, degrees=True).as_matrix()
            return value

        rows = [
            [pose(0, 0, 300), pose(80, 0, 300)],
            [pose(80, 100, 280, 3), pose(0, 100, 280, 6)],
        ]
        start, path, geometry = continuous_scan_waypoints(rows)
        self.assertEqual(geometry["transfer_count"], 1)
        self.assertEqual(geometry["waypoint_count"], len(path))
        # Only the first row's exit is dropped; the last row keeps its exit
        # so the final hole is passed at scan speed.
        self.assertEqual(geometry["omitted_exit_waypoint_count"], 1)
        self.assertAlmostEqual(path[-1][0, 3], -15.0)
        self.assertGreater(path[2][2, 3], rows[0][-1][2, 3])
        self.assertGreaterEqual(path[2][0, 3], min(path[1][0, 3], path[3][0, 3]))
        self.assertLessEqual(path[2][0, 3], max(path[1][0, 3], path[3][0, 3]))

    def test_continuous_scan_can_include_initial_tcp_without_separate_attitude_move(self) -> None:
        def pose(x, y, z, angle=0.0):
            value = np.eye(4)
            value[:3, 3] = [x, y, z]
            value[:3, :3] = Rotation.from_euler("z", angle, degrees=True).as_matrix()
            return value

        initial = pose(-120, -40, 340, 0)
        rows = [[pose(0, 0, 300, 12), pose(80, 0, 300, 15)]]
        start, path, geometry = continuous_scan_waypoints(
            rows, initial_tcp=initial, transfer_clearance_mm=20,
        )
        self.assertTrue(np.array_equal(start, initial))
        self.assertTrue(geometry["continuous_from_initial_tcp"])
        self.assertGreater(geometry["approach_waypoint_count"], 0)
        self.assertAlmostEqual(path[0][2, 3], 320.0)
        self.assertTrue(np.allclose(path[geometry["approach_waypoint_count"] - 1],
                                    path[geometry["approach_waypoint_count"] - 1]))
        self.assertAlmostEqual(path[geometry["approach_waypoint_count"] - 1][2, 3], 300.0)

    def test_overlay_and_contact_sheet_render_without_modifying_source(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        overlay = draw_flyby_overlay(
            image, header_lines=["row 1 image 1"],
            holes=[
                {"hole_id": 3, "anchor_px": np.array([320.0, 240.0]), "status": "accepted",
                 "ellipse_center_px": [322.0, 241.0], "ellipse_axes_px": [80.0, 76.0],
                 "ellipse_angle_deg": 10.0},
                {"hole_id": 4, "anchor_px": np.array([100.0, 100.0]), "status": "no_detection"},
            ],
            detections=[{"box": [280, 200, 360, 280]}],
        )
        self.assertEqual(overlay.shape, image.shape)
        self.assertFalse(image.any())
        self.assertTrue(overlay.any())
        tiles = [(thumbnail(overlay), "a"), (thumbnail(image), "b"), (thumbnail(overlay), "c")]
        sheet = build_contact_sheet(tiles, columns=2)
        self.assertEqual(sheet.shape, (2 * 360, 2 * 480, 3))
        with self.assertRaises(ValueError):
            build_contact_sheet([])


if __name__ == "__main__":
    unittest.main()
