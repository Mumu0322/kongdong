from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import run_yolo_eye_in_hand_optimized as module
from aubo_workbench.camera import CameraIntrinsics
from aubo_workbench.geometry import make_transform


class TwoStageGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.T_tcp_camera = np.eye(4)
        self.R_down = np.diag([1.0, -1.0, -1.0])
        self.intrinsics = CameraIntrinsics(1280, 720, 800.0, 800.0, 640.0, 360.0, ())

    def test_base_z_target_changes_only_base_z(self) -> None:
        current = make_transform(self.R_down, np.array([0.0, 0.0, 340.0]))
        target, measured = module.base_z_target_for_camera_height(
            current, self.T_tcp_camera, np.zeros(3), 260.0,
        )
        self.assertAlmostEqual(measured, 340.0)
        self.assertTrue(np.allclose(target[:2, 3], current[:2, 3]))
        self.assertAlmostEqual(target[2, 3], 260.0)
        self.assertTrue(np.allclose(target[:3, :3], current[:3, :3]))

    def test_final_tcp_xy_preserves_tcp_z_and_orientation(self) -> None:
        current = make_transform(self.R_down, np.array([0.0, 0.0, 260.0]))
        target, before = module.plan_final_tcp_xy(
            current, np.array([25.0, -10.0, 0.0]), xy_offset_mm=(0.0, 0.0),
        )
        self.assertTrue(np.allclose(before, [0.0, 0.0, 260.0]))
        self.assertTrue(np.allclose(target[:3, 3], [25.0, -10.0, 260.0]))
        self.assertTrue(np.allclose(target[:3, :3], current[:3, :3]))

    def test_principal_ray_intersects_known_plane(self) -> None:
        tcp = make_transform(self.R_down, np.array([0.0, 0.0, 340.0]))
        result = module.pixel_to_base_plane(
            np.array([640.0, 360.0]), self.intrinsics, tcp, self.T_tcp_camera,
            np.zeros(3), np.array([0.0, 0.0, 1.0]),
        )
        self.assertTrue(np.allclose(result, [0.0, 0.0, 0.0], atol=1e-8))

    def test_camera_ray_undistorts_pixel_before_back_projection(self) -> None:
        distorted_intrinsics = CameraIntrinsics(
            1280, 720, 800.0, 800.0, 640.0, 360.0,
            (0.15, -0.05, 0.001, -0.002, 0.0, 0.0, 0.0, 0.0),
        )
        point_3d = np.array([[0.2, -0.1, 1.0]], dtype=np.float64)
        image_point, _ = cv2.projectPoints(
            point_3d, np.zeros(3), np.zeros(3), distorted_intrinsics.camera_matrix(),
            np.asarray(distorted_intrinsics.distortion, dtype=np.float64),
        )
        ray = module.camera_ray(distorted_intrinsics, image_point.reshape(2))
        self.assertTrue(np.allclose(ray, point_3d.reshape(3) / np.linalg.norm(point_3d), atol=1e-7))

    def test_sdk_pose_converts_mm_to_m(self) -> None:
        T = make_transform(np.eye(3), np.array([1000.0, -500.0, 250.0]))
        pose = module.transform_to_sdk_pose_m_rad(T)
        self.assertEqual(pose[:3], [1.0, -0.5, 0.25])

    def test_circle_ellipse_fit_returns_center(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.ellipse(image, (640, 360), (100, 96), 0.0, 0.0, 360.0, (255, 255, 255), 3)
        detection = {"box": [520.0, 250.0, 760.0, 470.0], "center": [640.0, 360.0]}
        ellipse = module.fit_hole_ellipse(image, detection)
        self.assertIsNotNone(ellipse)
        assert ellipse is not None
        self.assertLess(np.linalg.norm(np.asarray(ellipse["center_px"]) - [640.0, 360.0]), 2.0)
        self.assertGreater(ellipse["roundness"], 0.9)

    def test_fine_capture_uses_yolo_center_even_when_ellipse_is_valid(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        bundle = SimpleNamespace(
            color_bgr=image,
            intrinsics=self.intrinsics,
            host_timestamp_ns=123,
        )
        detection = {
            "box": [600.0, 300.0, 700.0, 400.0],
            "center": [650.0, 360.0],
            "class_id": 0,
        }
        ellipse = {
            "center_px": np.array([640.0, 360.0]),
            "center_px_distorted": np.array([640.0, 360.0]),
            "coverage_deg": 360.0,
            "residual_px": 0.1,
            "roundness": 0.99,
            "axes_px": [180.0, 178.0],
        }
        cfg = module.TwoStageConfig(fine_frames=1, min_fine_valid=1, fine_stable_min_frames=1)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, "get_rgb_frame_bundle", return_value=bundle), \
                patch.object(module, "detect", return_value=[detection]), \
                patch.object(module, "fit_hole_ellipse", return_value=ellipse), \
                patch.object(module, "_overlay", return_value=image), \
                patch.object(module.cv2, "imwrite", return_value=True):
            observations, _, _, _ = module._capture_fine_burst(
                pipeline=object(), model=object(), confidence=0.5,
                chosen=detection, cfg=cfg, run_dir=Path(directory), name="fine",
            )

        self.assertEqual(len(observations), 1)
        self.assertTrue(np.allclose(observations[0].center_px, [650.0, 360.0]))
        self.assertEqual(observations[0].center_source, "yolo")
        self.assertFalse(np.allclose(observations[0].center_px, ellipse["center_px"]))

    def test_fine_fusion_rejects_outlier_and_keeps_median(self) -> None:
        cfg = module.TwoStageConfig(fine_frames=4, min_fine_valid=3)
        items = []
        for index, center in enumerate(((640.0, 360.0), (640.2, 360.0), (639.8, 360.1), (640.1, 359.9))):
            items.append(module.Observation(
                "fine", index, np.asarray(center),
                ellipse={"residual_px": 0.2, "roundness": 0.98, "axes_px": [180.0, 178.0]},
                center_source="yolo",
            ))
        summary = module._fuse_fine(items, cfg)
        self.assertAlmostEqual(summary["center_px"][0], 640.05, places=6)
        self.assertLess(summary["center_scatter_p95_px"], 1.0)

    def test_fine_fusion_reports_yolo_as_the_only_center_source(self) -> None:
        cfg = module.TwoStageConfig(fine_frames=4, min_fine_valid=3)
        items = []
        for index, center in enumerate(((648.5, 377.5), (648.7, 377.6), (648.6, 377.4), (648.6, 377.5))):
            items.append(module.Observation(
                "fine", index, np.asarray(center),
                ellipse={"residual_px": 2.4, "coverage_deg": 360.0,
                         "roundness": 1.0, "axes_px": [180.0, 180.0]},
                center_source="yolo",
                quality_note="strict_ellipse_rejected",
            ))
        summary = module._fuse_fine(items, cfg)
        self.assertEqual(summary["center_source"], "yolo")
        self.assertEqual(summary["yolo_frames"], 4)
        self.assertEqual(summary["yolo_fallback_frames"], 4)
        self.assertEqual(summary["strict_ellipse_frames"], 0)
        self.assertLess(summary["center_scatter_p95_px"], 0.35)

    def test_fine_recovery_retries_then_accepts_degraded_result(self) -> None:
        cfg = module.TwoStageConfig(
            fine_frames=4,
            min_fine_valid=3,
            fine_stable_min_frames=3,
            fine_retry_count=2,
            fine_settle_discard_frames=10,
            fine_degraded_max_center_scatter_p95_px=0.6,
        )
        observations = [
            module.Observation(
                "fine", index, np.asarray([640.0 + index * 0.1, 360.0]),
                ellipse={"residual_px": 0.2, "coverage_deg": 360.0,
                         "roundness": 0.98, "axes_px": [180.0, 178.0]},
            )
            for index in range(4)
        ]
        capture_calls: list[dict[str, object]] = []

        def fake_capture(*_args: object, **kwargs: object) -> tuple[list[module.Observation], object, np.ndarray, None]:
            capture_calls.append(kwargs)
            return observations, self.intrinsics, np.zeros((8, 8, 3), dtype=np.uint8), None

        fusion_calls = {"count": 0}

        def fake_fuse(*_args: object, **kwargs: object) -> dict[str, object]:
            fusion_calls["count"] += 1
            if kwargs.get("max_center_scatter_p95_px") is None:
                raise RuntimeError("strict gate")
            return {"valid_frames": 4, "center_scatter_p95_px": 0.5}

        hole = {"tracking_events": []}
        timing = module.TimingRecorder()
        rows: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, "_capture_fine_burst", side_effect=fake_capture), \
                patch.object(module, "_fuse_fine", side_effect=fake_fuse):
            result = module._capture_fine_with_recovery(
                pipeline=object(), model=object(), confidence=0.5, chosen={"class_id": 0},
                cfg=cfg, run_dir=Path(directory), hole=hole, hole_id=9,
                processing_order=3, expected_anchor_px=np.array([640.0, 360.0]),
                timing=timing, rows=rows,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "degraded_fine")
        self.assertEqual(result["fine"]["fine_quality_status"], "degraded_fine")
        self.assertEqual(len(capture_calls), 3)
        self.assertTrue(all(call["settle_discard_frames"] == 10 for call in capture_calls))
        self.assertEqual(fusion_calls["count"], 4)
        self.assertEqual(len(hole["tracking_events"]), 3)
        self.assertEqual(len(rows), 12)

    def test_fine_recovery_defers_without_raising_after_all_attempts_fail(self) -> None:
        cfg = module.TwoStageConfig(
            fine_frames=4,
            min_fine_valid=3,
            fine_stable_min_frames=3,
            fine_retry_count=1,
        )
        observations = [
            module.Observation(
                "fine", index, np.asarray([640.0, 360.0]),
                ellipse={"residual_px": 0.2, "coverage_deg": 360.0,
                         "roundness": 0.98, "axes_px": [180.0, 178.0]},
            )
            for index in range(4)
        ]

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(
                    module, "_capture_fine_burst",
                    return_value=(observations, self.intrinsics, np.zeros((8, 8, 3), dtype=np.uint8), None),
                ), \
                patch.object(module, "_fuse_fine", side_effect=RuntimeError("strict and degraded gate")):
            result = module._capture_fine_with_recovery(
                pipeline=object(), model=object(), confidence=0.5, chosen={"class_id": 0},
                cfg=cfg, run_dir=Path(directory), hole={"tracking_events": []}, hole_id=9,
                processing_order=3, expected_anchor_px=np.array([640.0, 360.0]),
                timing=module.TimingRecorder(), rows=[],
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "deferred_fine_quality")
        self.assertIsNone(result["fine"])
        self.assertEqual(len(result["attempts"]), 3)
        self.assertEqual(result["attempts"][-1]["fusion_status"], "degraded_failed")

    def test_next_hole_confirmation_is_kept_between_selected_holes(self) -> None:
        output = StringIO()
        with patch("builtins.input", return_value="m"), redirect_stdout(output):
            command = module._request_next_hole_confirmation(1, 2)
        self.assertEqual(command, "m")
        self.assertIn("[NEXT_HOLE_CONFIRM_REQUIRED]", output.getvalue())
        self.assertIn("下一个检测孔=2", output.getvalue())

    def test_next_cycle_confirmation_blocks_second_initial_capture(self) -> None:
        with patch("builtins.input", return_value="m"):
            command = module._request_next_cycle_confirmation(1)
        self.assertEqual(command, "m")

    def test_initial_capture_waits_for_two_steady_reads(self) -> None:
        first = np.eye(4)
        second = np.eye(4)
        second[0, 3] = 1.0
        with patch.object(
            module,
            "_wait_robot_steady",
            side_effect=[({}, first), ({}, second)],
        ) as wait_steady, patch.object(module.time, "sleep") as sleep:
            actual = module._wait_robot_steady_before_initial_capture(
                object(), settle_delay_s=0.5,
            )
        self.assertEqual(wait_steady.call_count, 2)
        sleep.assert_called_once_with(0.5)
        self.assertTrue(np.allclose(actual, second))

    def test_timing_recorder_persists_completed_and_failed_events(self) -> None:
        report: dict[str, object] = {}
        output = StringIO()
        timer = module.TimingRecorder()
        timer.attach_report(report)
        with redirect_stdout(output):
            with timer.measure("hole_01/coarse_capture", hole_id=1):
                pass
            with self.assertRaises(RuntimeError):
                with timer.measure("hole_01/coarse_geometry", hole_id=1):
                    raise RuntimeError("synthetic timing failure")
        timing = report["timing"]
        self.assertEqual(len(timing["events"]), 2)
        self.assertEqual(timing["events"][0]["status"], "completed")
        self.assertEqual(timing["events"][1]["status"], "failed")
        self.assertEqual(timing["aggregates"]["hole_01/coarse_capture"]["count"], 1)
        self.assertIn("[TIMING] hole_01/coarse_geometry", output.getvalue())

    def test_transit_motion_profile_uses_separate_speed(self) -> None:
        class FakeMotion:
            def __init__(self) -> None:
                self.calls: list[tuple[float, float]] = []

            def move_line(self, pose: list[float], speed_m_s: float, acc_m_s2: float) -> list[int]:
                self.calls.append((speed_m_s, acc_m_s2))
                return [0, 0]

        args = SimpleNamespace(
            speed_m_s=0.03, acc_m_s2=0.10,
            transit_speed_m_s=0.05, transit_acc_m_s2=0.15,
            approach_speed_m_s=0.04, approach_acc_m_s2=0.12,
        )
        motion = FakeMotion()
        with patch.object(module, "_wait_robot_steady", return_value=({}, np.eye(4))):
            module._confirm_and_move_line(
                "synthetic transit", np.eye(4), np.eye(4), args,
                motion, None, require_confirmation=False, motion_profile="transit",
            )
            module._confirm_and_move_line(
                "synthetic approach", np.eye(4), np.eye(4), args,
                motion, None, require_confirmation=False, motion_profile="approach",
            )
            module._confirm_and_move_line(
                "synthetic precision", np.eye(4), np.eye(4), args,
                motion, None, require_confirmation=False, motion_profile="precision",
            )
        self.assertEqual(
            motion.calls,
            [(0.05, 0.15), (0.04, 0.12), (0.03, 0.10)],
        )

    def test_choose_boxes_uses_clicked_count_when_count_is_omitted(self) -> None:
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        detections = [
            {"box": [5.0, 5.0, 25.0, 25.0], "center": [15.0, 15.0],
             "class_name": "circle", "confidence": 0.9},
            {"box": [65.0, 5.0, 85.0, 25.0], "center": [75.0, 15.0],
             "class_name": "circle", "confidence": 0.9},
        ]
        state: dict[str, object] = {"callback": None, "clicked": False}

        def set_callback(_name: str, callback: object) -> None:
            state["callback"] = callback

        def wait_key(_delay_ms: int) -> int:
            if not bool(state["clicked"]):
                callback = state["callback"]
                assert callable(callback)
                callback(module.cv2.EVENT_LBUTTONDOWN, 15, 15, 0, None)
                state["clicked"] = True
            return 13

        with patch.object(module.cv2, "namedWindow"), \
                patch.object(module.cv2, "setMouseCallback", side_effect=set_callback), \
                patch.object(module.cv2, "imshow"), \
                patch.object(module.cv2, "waitKey", side_effect=wait_key):
            selection = module.choose_boxes(image, detections, return_clicks=True)
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection[0], [0])

    def test_initial_selection_degrades_one_bad_plane_to_shared_navigation_plane(self) -> None:
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        bundle = SimpleNamespace(
            color_bgr=image,
            xyz_map_mm=np.zeros((120, 160, 3), dtype=np.float64),
            intrinsics=self.intrinsics,
        )
        detections = [
            {"box": [620.0, 340.0, 660.0, 380.0], "center": [640.0, 360.0], "class_id": 0},
            {"box": [680.0, 340.0, 720.0, 380.0], "center": [700.0, 360.0], "class_id": 0},
        ]
        good_plane = module.PlaneEstimate(
            point_camera_mm=np.array([0.0, 0.0, 1000.0]),
            normal_camera=np.array([0.0, 0.0, 1.0]),
            rmse_mm=1.0,
            ring_points=20,
            surface_model="ring",
        )
        bad_plane = module.PlaneEstimate(
            point_camera_mm=np.array([0.0, 0.0, 1010.0]),
            normal_camera=np.array([0.0, 0.0, 1.0]),
            rmse_mm=22.9,
            ring_points=20,
            surface_model="ring",
        )
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, "get_aligned_frame_bundle", return_value=bundle), \
                patch.object(module, "detect", return_value=detections), \
                patch.object(
                    module, "choose_boxes",
                    return_value=([0, 1], [[640.0, 360.0], [700.0, 360.0]]),
                ), \
                patch.object(module, "hole_camera_point", return_value=(np.array([0.0, 0.0, 1000.0]), {})), \
                patch.object(module, "_plane_estimate_from_info", side_effect=[good_plane, bad_plane]), \
                patch.object(module.cv2, "imwrite", return_value=True):
            _, holes, group_center, group_plane, _ = module._capture_initial_multi_hole_selection(
                pipeline=object(), align=object(), chain=object(), model=object(),
                confidence=0.5, run_dir=Path(directory), max_plane_rmse_mm=3.5,
            )

        self.assertEqual(len(holes), 2)
        self.assertFalse(holes[0]["initial_geometry_fallback"])
        self.assertTrue(holes[1]["initial_geometry_fallback"])
        self.assertEqual(holes[1]["initial_surface_model"], "initial_group_plane_fallback")
        self.assertEqual(holes[1]["initial_geometry_fallback_reason"].split(":", 1)[0], "ValueError")
        self.assertTrue(np.allclose(group_center, [0.0, 0.0, 1000.0]))
        self.assertAlmostEqual(group_plane.rmse_mm, 1.0)

    def test_diameter_categories_cover_all_supported_holes(self) -> None:
        for diameter in (64.6, 70.2, 74.7):
            self.assertEqual(min(module.HOLE_DIAMETERS_MM, key=lambda value: abs(value - diameter)), round(diameter / 5.0) * 5.0)

    def test_motion_defaults_are_preview_and_validated_only(self) -> None:
        parser = module.build_parser()
        defaults = parser.parse_args([])
        self.assertFalse(defaults.execute)
        self.assertFalse(defaults.allow_experimental_handeye)
        self.assertFalse(defaults.move_final_xy)
        self.assertFalse(parser.parse_args(["--require-validated-handeye"]).allow_experimental_handeye)
        self.assertTrue(parser.parse_args(["--execute"]).execute)
        self.assertTrue(parser.parse_args(["--allow-experimental-handeye"]).allow_experimental_handeye)
        self.assertTrue(parser.parse_args(["--move-final-xy"]).move_final_xy)
        self.assertEqual(parser.parse_args([]).final_target_mode, "gripper")
        self.assertEqual(parser.parse_args(["--final-target-mode", "normal"]).final_target_mode, "normal")
        self.assertIsNone(parser.parse_args([]).tcp_xy_offset_mm)
        self.assertEqual(tuple(parser.parse_args(["--tcp-xy-offset-mm", "0", "0"]).tcp_xy_offset_mm), (0.0, 0.0))
        self.assertAlmostEqual(parser.parse_args([]).speed_m_s, 0.08)
        self.assertAlmostEqual(parser.parse_args([]).acc_m_s2, 0.25)
        self.assertAlmostEqual(parser.parse_args([]).transit_speed_m_s, 0.15)
        self.assertAlmostEqual(parser.parse_args([]).transit_acc_m_s2, 0.45)
        self.assertAlmostEqual(parser.parse_args([]).approach_speed_m_s, 0.12)
        self.assertAlmostEqual(parser.parse_args([]).approach_acc_m_s2, 0.35)
        self.assertEqual(parser.parse_args([]).fine_settle_discard_frames, 10)
        self.assertEqual(parser.parse_args([]).fine_retries, 2)


if __name__ == "__main__":
    unittest.main()
