from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
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

    def test_robust_circle_refinement_rejects_horizontal_line_edges(self) -> None:
        rng = np.random.default_rng(4)
        angles = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
        circle = np.column_stack((
            100.0 + 40.0 * np.cos(angles),
            80.0 + 40.0 * np.sin(angles),
        )) + rng.normal(0.0, 0.15, (360, 2))
        horizontal_reflection = np.column_stack((
            np.linspace(55.0, 145.0, 500),
            np.full(500, 80.0),
        ))
        center, radius = module._robust_refine_circle_from_edges(
            np.vstack((circle, horizontal_reflection)),
            np.array([101.5, 79.0]),
            40.8,
        )
        self.assertLess(float(np.linalg.norm(center - [100.0, 80.0])), 0.1)
        self.assertAlmostEqual(radius, 40.0, delta=0.1)

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

    def test_coarse_recapture_settle_uses_rgbd_pipeline_and_discards_frames(self) -> None:
        settled = np.eye(4)
        with patch.object(
            module,
            "_wait_robot_steady_before_initial_capture",
            return_value=settled,
        ) as wait_steady, patch.object(
            module,
            "get_aligned_frame_bundle",
            return_value=object(),
        ) as get_frame:
            actual, discarded = module._settle_and_discard_coarse_recapture_frames(
                object(), "rgbd-pipeline", "align", "chain",
                settle_delay_s=0.3, discard_frames=3,
            )
        wait_steady.assert_called_once()
        self.assertEqual(get_frame.call_count, 3)
        self.assertEqual(get_frame.call_args.args[:3], ("rgbd-pipeline", "align", "chain"))
        self.assertEqual(discarded, 3)
        self.assertTrue(np.allclose(actual, settled))

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

    def test_position_move_waits_for_motion_session_after_pose_session(self) -> None:
        class FakeMotion:
            def __init__(self) -> None:
                self.steady_reads: list[bool] = []
                self._states = [False, True, True]
                self.move_calls = 0

            def snapshot(self) -> dict[str, object]:
                steady = self._states.pop(0) if self._states else True
                self.steady_reads.append(steady)
                return {"power_on": True, "steady": steady, "collision": False}

            def move_line(self, _pose: list[float], _speed: float, _acc: float) -> list[int]:
                self.move_calls += 1
                self.assert_last_state_was_steady()
                return [0, 0]

            def assert_last_state_was_steady(self) -> None:
                if not self.steady_reads[-1]:
                    raise AssertionError("moveLine 在运动会话判稳前被调用")

        args = SimpleNamespace(speed_m_s=0.03, acc_m_s2=0.10)
        motion = FakeMotion()
        with patch.object(module, "_wait_robot_steady", return_value=({}, np.eye(4))), \
                patch.object(module.time, "sleep") as sleep:
            module._confirm_and_move_line(
                "synthetic handoff", np.eye(4), np.eye(4), args,
                motion, object(), require_confirmation=False,
            )

        self.assertEqual(motion.move_calls, 1)
        self.assertEqual(motion.steady_reads, [False, True, True])
        sleep.assert_called_once_with(module.ROBOT_STEADY_POLL_INTERVAL_S)

    def test_shared_fine_motion_has_pure_lift_and_pure_descent_clearance(self) -> None:
        current = make_transform(np.eye(3), np.array([0.0, 0.0, 120.0]))
        target = make_transform(np.eye(3), np.array([100.0, 50.0, 90.0]))
        args = SimpleNamespace(
            speed_m_s=0.08, acc_m_s2=0.25,
            transit_speed_m_s=0.15, transit_acc_m_s2=0.45,
            approach_speed_m_s=0.12, approach_acc_m_s2=0.35,
        )
        calls: list[tuple[np.ndarray, str]] = []

        def fake_move(_label, _current, destination, *_args, **kwargs):
            calls.append((np.asarray(destination).copy(), kwargs["motion_profile"]))
            return np.asarray(destination).copy()

        with patch.object(module, "_confirm_and_move_line", side_effect=fake_move):
            result = module._move_to_shared_fine_pose(
                "共享精定位", 1, current, target, args, object(), object(),
            )

        self.assertEqual(len(calls), 4)
        self.assertEqual(
            [item[1] for item in calls],
            ["transit", "transit", "approach", "approach"],
        )
        # 第一段必须只抬Z，且至少抬10 mm。
        self.assertTrue(np.allclose(calls[0][0][:2, 3], current[:2, 3]))
        self.assertTrue(np.allclose(calls[0][0][:3, :3], current[:3, :3]))
        self.assertGreaterEqual(
            float(calls[0][0][2, 3] - current[2, 3]),
            module.SHARED_OBSERVATION_MIN_LIFT_MM,
        )
        # 第二段才允许横移/调姿态，仍保持在抬升后的高度。
        self.assertTrue(np.allclose(calls[1][0][2, 3], calls[0][0][2, 3]))
        # 第三段先下降到目标上方10 mm，仍保持XY和姿态不变。
        self.assertTrue(np.allclose(calls[2][0][:2, 3], target[:2, 3]))
        self.assertTrue(np.allclose(calls[2][0][:3, :3], target[:3, :3]))
        self.assertAlmostEqual(
            float(calls[2][0][2, 3]),
            float(target[2, 3] + module.SHARED_OBSERVATION_MIN_DESCENT_MM),
        )
        # 第四段才穿过最后10 mm纯Z安全下降空间到观察位。
        self.assertTrue(np.allclose(calls[3][0][:2, 3], target[:2, 3]))
        self.assertTrue(np.allclose(calls[3][0][:3, :3], target[:3, :3]))
        self.assertGreaterEqual(
            float(calls[1][0][2, 3] - calls[2][0][2, 3]),
            module.SHARED_OBSERVATION_MIN_DESCENT_MM,
        )
        self.assertAlmostEqual(
            float(calls[2][0][2, 3] - calls[3][0][2, 3]),
            module.SHARED_OBSERVATION_MIN_DESCENT_MM,
        )
        self.assertTrue(np.allclose(result, target))

    def test_shared_coarse_motion_has_pure_lift_and_pure_descent_clearance(self) -> None:
        current = make_transform(np.eye(3), np.array([-20.0, 30.0, 140.0]))
        target = make_transform(np.eye(3), np.array([80.0, -40.0, 100.0]))
        args = SimpleNamespace(
            speed_m_s=0.08, acc_m_s2=0.25,
            transit_speed_m_s=0.15, transit_acc_m_s2=0.45,
            approach_speed_m_s=0.12, approach_acc_m_s2=0.35,
        )
        calls: list[tuple[np.ndarray, str]] = []

        def fake_move(_label, _current, destination, *_args, **kwargs):
            calls.append((np.asarray(destination).copy(), kwargs["motion_profile"]))
            return np.asarray(destination).copy()

        with patch.object(module, "_confirm_and_move_line", side_effect=fake_move):
            result = module._move_to_shared_coarse_pose(
                1, 1, current, target, args, object(), object(),
            )

        self.assertEqual(len(calls), 4)
        self.assertEqual(
            [item[1] for item in calls],
            ["transit", "transit", "transit", "transit"],
        )
        self.assertTrue(np.allclose(calls[0][0][:2, 3], current[:2, 3]))
        self.assertTrue(np.allclose(calls[0][0][:3, :3], current[:3, :3]))
        self.assertGreaterEqual(
            float(calls[0][0][2, 3] - current[2, 3]),
            module.SHARED_OBSERVATION_MIN_LIFT_MM,
        )
        self.assertTrue(np.allclose(calls[1][0][2, 3], calls[0][0][2, 3]))
        self.assertTrue(np.allclose(calls[2][0][:2, 3], target[:2, 3]))
        self.assertTrue(np.allclose(calls[2][0][:3, :3], target[:3, :3]))
        self.assertAlmostEqual(
            float(calls[2][0][2, 3]),
            float(target[2, 3] + module.SHARED_OBSERVATION_MIN_DESCENT_MM),
        )
        self.assertTrue(np.allclose(calls[3][0][:2, 3], target[:2, 3]))
        self.assertTrue(np.allclose(calls[3][0][:3, :3], target[:3, :3]))
        self.assertGreaterEqual(
            float(calls[1][0][2, 3] - calls[2][0][2, 3]),
            module.SHARED_OBSERVATION_MIN_DESCENT_MM,
        )
        self.assertAlmostEqual(
            float(calls[2][0][2, 3] - calls[3][0][2, 3]),
            module.SHARED_OBSERVATION_MIN_DESCENT_MM,
        )
        self.assertTrue(np.allclose(result, target))

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

    def test_coarse_burst_reanchors_after_initial_projection(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        bundle = SimpleNamespace(
            color_bgr=image,
            xyz_map_mm=np.zeros((720, 1280, 3), dtype=np.float32),
            intrinsics=self.intrinsics,
            host_timestamp_ns=123,
        )
        plane = module.PlaneEstimate(
            point_camera_mm=np.array([0.0, 0.0, 340.0]),
            normal_camera=np.array([0.0, 0.0, 1.0]),
            rmse_mm=1.0,
            ring_points=100,
            surface_model="ring",
        )
        detections = [
            {"box": [620.0, 340.0, 660.0, 380.0], "center": [655.0, 360.0], "class_id": 0},
            {"box": [621.0, 340.0, 661.0, 380.0], "center": [656.0, 360.0], "class_id": 0},
            {"box": [620.5, 340.0, 660.5, 380.0], "center": [655.5, 360.0], "class_id": 0},
        ]
        cfg = module.TwoStageConfig(
            coarse_frames=3,
            min_coarse_valid=3,
            coarse_settle_frames=0,
            coarse_max_attempt_multiplier=1,
            max_coarse_center_scatter_p95_px=0.8,
        )
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, "get_aligned_frame_bundle", return_value=bundle), \
                patch.object(module, "detect", side_effect=[[item] for item in detections]), \
                patch.object(module, "hole_camera_point", return_value=(np.array([0.0, 0.0, 340.0]), {})), \
                patch.object(module, "_plane_estimate_from_info", return_value=plane), \
                patch.object(module.cv2, "imwrite", return_value=True):
            observations, _ = module._capture_coarse_burst(
                pipeline=object(), align=object(), chain=object(), model=object(),
                confidence=0.5, chosen={"class_id": 0}, cfg=cfg,
                run_dir=Path(directory), name="coarse",
                initial_anchor_px=np.array([640.0, 360.0]),
                tracking_tolerance_px=70.0,
                lock_anchor=False,
                include_points=True,
            )

        self.assertEqual(len(observations), 3)
        self.assertAlmostEqual(float(observations[0].tracking_distance_px), 15.0)
        self.assertAlmostEqual(float(observations[1].tracking_distance_px), 1.0)
        self.assertAlmostEqual(float(observations[2].tracking_distance_px), 0.5)

    def test_batch_coarse_captures_all_holes_from_each_rgbd_frame(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        xyz = np.zeros((720, 1280, 3), dtype=np.float32)
        bundle = SimpleNamespace(
            color_bgr=image,
            xyz_map_mm=xyz,
            intrinsics=self.intrinsics,
            host_timestamp_ns=123,
        )
        selected = [
            {
                "hole_id": 1,
                "initial_center_base_mm": np.array([0.0, 0.0, 1000.0]),
                "initial_detection": {"class_id": 0},
            },
            {
                "hole_id": 2,
                "initial_center_base_mm": np.array([100.0, 0.0, 1000.0]),
                "initial_detection": {"class_id": 0},
            },
        ]
        detections = [
            {"box": [620.0, 340.0, 660.0, 380.0], "center": [640.0, 360.0], "class_id": 0},
            {"box": [700.0, 340.0, 740.0, 380.0], "center": [720.0, 360.0], "class_id": 0},
        ]
        plane_info = {
            "plane_point_camera_mm": [0.0, 0.0, 1000.0],
            "plane_normal_camera": [0.0, 0.0, 1.0],
            "plane_rmse_mm": 1.0,
            "ring_points": 30,
            "local_plane_point_camera_mm": [0.0, 0.0, 1000.0],
            "points_camera_mm": [[0.0, 0.0, 1000.0], [1.0, 0.0, 1000.0]],
            "surface_model": "front_surface_outer_ring_v2",
            "surface_selection_policy": module.COARSE_SURFACE_SELECTION_POLICY,
            "front_surface_z_mm": 1000.0,
            "ring_points_raw": 40,
            "surface_points_selected": 30,
        }
        cfg = module.TwoStageConfig(
            batch_coarse_frames=3,
            batch_coarse_min_valid=2,
            batch_coarse_min_holes_per_frame=2,
            batch_coarse_settle_discard_frames=2,
            multi_coarse_tracking_tolerance_px=80.0,
        )
        handeye = SimpleNamespace(T_tcp_rgb_camera=np.eye(4, dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, "get_aligned_frame_bundle", return_value=bundle) as get_frame, \
                patch.object(module, "detect", return_value=detections), \
                patch.object(module, "hole_camera_point", return_value=(np.array([0.0, 0.0, 1000.0]), plane_info)), \
                patch.object(module.cv2, "imwrite", return_value=True), \
                patch.object(module, "_save_coarse_pointcloud_image", return_value=Path(directory) / "cloud.png"):
            result = module._batch_coarse_localization_at_340mm(
                selected, np.eye(4, dtype=np.float64), handeye,
                object(), object(), object(), object(), 0.5, cfg,
                self.intrinsics, Path(directory), module.TimingRecorder(), [],
            )
            archive_created = (
                Path(directory) / "batch_coarse_340_all_holes_pointcloud.npz"
            ).is_file()

        self.assertEqual(get_frame.call_count, 5)
        self.assertTrue(result[1]["success"])
        self.assertTrue(result[2]["success"])
        self.assertEqual(result[1]["valid_frames"], 3)
        self.assertEqual(result[2]["valid_frames"], 3)
        self.assertTrue(archive_created)
        self.assertIn("_batch_metadata", result)
        self.assertEqual(len(result["_batch_metadata"]["frame_records"]), 3)
        self.assertEqual(result["_batch_metadata"]["discarded_frame_count"], 2)

    def test_batch_defaults_use_stable_bursts_for_all_selected_holes(self) -> None:
        cfg = module.TwoStageConfig()
        self.assertEqual(cfg.batch_coarse_frames, 15)
        self.assertEqual(cfg.batch_coarse_min_valid, 10)
        self.assertEqual(cfg.batch_fine_frames, 8)
        self.assertEqual(cfg.batch_fine_min_valid, 5)
        self.assertEqual(cfg.batch_fine_stable_min_frames, 5)
        self.assertEqual(cfg.batch_fine_supplement_rounds, 1)
        args = module.build_parser().parse_args([])
        self.assertEqual(args.batch_coarse_frames, 15)
        self.assertEqual(args.batch_coarse_min_valid, 10)
        self.assertEqual(args.batch_fine_frames, 8)
        self.assertEqual(args.batch_fine_min_valid, 5)
        self.assertEqual(args.batch_fine_stable_min_frames, 5)
        self.assertEqual(args.batch_fine_supplement_rounds, 1)

    def test_current_initial_pointcloud_overrides_stale_cache_geometry(self) -> None:
        hole = {
            "hole_id": 2,
            "initial_center_px": [720.0, 360.0],
            "initial_center_base_mm": [100.0, 20.0, 900.0],
            "initial_point_camera_mm": [80.0, 0.0, 900.0],
            "initial_plane_point_base_mm": [100.0, 20.0, 901.0],
            "initial_plane_point_camera_mm": [80.0, 0.0, 901.0],
            "initial_plane_normal_base": [0.0, 0.0, -1.0],
            "initial_plane_normal_camera": [0.0, 0.0, -1.0],
            "initial_shared_plane_point_base_mm": [0.0, 0.0, 905.0],
            "initial_shared_plane_point_camera_mm": [0.0, 0.0, 905.0],
            "initial_shared_plane_normal_base": [0.0, -0.1, -0.995],
            "initial_shared_plane_normal_camera": [0.0, -0.1, -0.995],
            "initial_plane_rmse_mm": 0.5,
            "initial_ring_points": 80,
            "coarse_center_base_mm": [999.0, 999.0, 999.0],
            "coarse_normal_toward_camera_base": [1.0, 0.0, 0.0],
            "coarse_source": "persistent_base_cache",
        }

        reused = module._reuse_initial_pointcloud_geometry_for_batch_fine(hole)

        self.assertTrue(reused)
        np.testing.assert_allclose(hole["coarse_center_base_mm"], [100.0, 20.0, 900.0])
        np.testing.assert_allclose(
            hole["coarse_normal_toward_camera_base"], [0.0, -0.1, -0.995]
        )
        np.testing.assert_allclose(hole["coarse_plane_point_base_mm"], [0.0, 0.0, 905.0])
        self.assertEqual(
            hole["coarse_source"], "initial_selection_shared_pointcloud_reuse"
        )
        self.assertEqual(
            hole["coarse_geometry_type"],
            "shared_workpiece_plane_with_per_hole_center_ray",
        )
        self.assertTrue(hole["initial_pointcloud_reused_for_batch_fine"])
        self.assertEqual(
            hole["coarse_captures"][0]["source_frame"],
            "current_cycle_initial_rgbd_selection",
        )

    def test_batch_fine_captures_and_fuses_all_holes_from_one_260mm_view(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        bundle = SimpleNamespace(
            color_bgr=image,
            intrinsics=self.intrinsics,
            host_timestamp_ns=456,
        )
        selected = [
            {
                "hole_id": 1,
                "initial_center_base_mm": np.array([0.0, 0.0, 1000.0]),
                "initial_detection": {
                    "class_id": 0, "box": [620.0, 340.0, 660.0, 380.0],
                },
            },
            {
                "hole_id": 2,
                "initial_center_base_mm": np.array([100.0, 0.0, 1000.0]),
                "initial_detection": {
                    "class_id": 0, "box": [700.0, 340.0, 740.0, 380.0],
                },
            },
        ]
        detections = [
            {"box": [620.0, 340.0, 660.0, 380.0], "center": [640.0, 360.0], "class_id": 0, "confidence": 0.9},
            {"box": [700.0, 340.0, 740.0, 380.0], "center": [720.0, 360.0], "class_id": 0, "confidence": 0.8},
        ]
        def geometric_fit(_image, detection, _intrinsics):
            return {
                "center_px": np.asarray(detection["center"], dtype=np.float64)
                + np.array([1.25, -0.75]),
                "axes_px": np.array([40.0, 40.0]),
                "residual_px": 0.2,
                "coverage_deg": 300.0,
                "roundness": 0.98,
                "fit_method": "hough_circle",
            }
        cfg = module.TwoStageConfig(
            fine_frames=3,
            min_fine_valid=2,
            fine_stable_min_frames=2,
            batch_fine_frames=2,
            batch_fine_min_valid=2,
            batch_fine_stable_min_frames=2,
            fine_stable_center_scatter_p95_px=1.0,
            max_fine_center_scatter_p95_px=1.0,
            fine_settle_discard_frames=1,
            fine_pointcloud_anchor_tolerance_px=30.0,
        )
        handeye = SimpleNamespace(T_tcp_rgb_camera=np.eye(4, dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, "get_rgb_frame_bundle", return_value=bundle) as get_frame, \
                patch.object(module, "detect", return_value=detections), \
                patch.object(module, "fit_hole_ellipse", side_effect=geometric_fit), \
                patch.object(module.cv2, "imwrite", return_value=True):
            result = module._batch_fine_localization_at_260mm(
                selected, np.eye(4, dtype=np.float64), handeye,
                object(), object(), 0.5, cfg, self.intrinsics,
                Path(directory), module.TimingRecorder(), [],
            )

        self.assertGreaterEqual(get_frame.call_count, 3)
        self.assertTrue(result[1]["success"])
        self.assertTrue(result[2]["success"])
        self.assertEqual(result[1]["fine"]["valid_frames"], 2)
        self.assertEqual(result[2]["fine"]["valid_frames"], 2)
        self.assertTrue(np.allclose(result[1]["fine"]["center_px"], [641.25, 359.25]))
        self.assertTrue(np.allclose(result[2]["fine"]["center_px"], [721.25, 359.25]))
        self.assertEqual(result[1]["fine"]["center_source"], "hough_circle")
        self.assertEqual(result[1]["fine"]["yolo_frames"], 0)
        self.assertEqual(len(result[1]["observations"]), 2)
        self.assertEqual(len(result[2]["observations"]), 2)

    def test_batch_fine_flushes_fast_queue_until_two_fresh_frame_intervals(self) -> None:
        timestamps_ns = [
            1_000_000_000,
            1_001_000_000,
            1_002_000_000,
            1_020_000_000,
            1_040_000_000,
        ]
        bundles = [
            SimpleNamespace(
                host_timestamp_ns=timestamp,
                color_frame_index=index,
                color_timestamp_us=timestamp // 1000,
            )
            for index, timestamp in enumerate(timestamps_ns, start=1)
        ]
        with patch.object(module, "get_rgb_frame_bundle", side_effect=bundles) as get_frame:
            result = module._flush_rgb_queue_until_fresh(
                object(), 3,
                maximum_extra_frames=5,
                fresh_host_interval_ms=10.0,
                required_fresh_intervals=2,
            )

        self.assertEqual(get_frame.call_count, 5)
        self.assertEqual(result["discarded_frame_count"], 5)
        self.assertTrue(result["fresh_frame_confirmed"])
        self.assertEqual(result["reason"], "minimum_and_fresh_intervals_reached")
        self.assertEqual(result["records"][-1]["fresh_interval_streak"], 2)

    def test_projection_matching_is_one_to_one_and_respects_class(self) -> None:
        detections = [
            {"center": [100.0, 100.0], "box": [90.0, 90.0, 110.0, 110.0], "class_id": 0, "confidence": 0.9},
            {"center": [104.0, 100.0], "box": [94.0, 90.0, 114.0, 110.0], "class_id": 1, "confidence": 0.9},
        ]
        assignments = module._assign_detections_to_projection(
            detections,
            {"1": np.array([101.0, 100.0]), "2": np.array([104.0, 100.0])},
            10.0,
            expected_metadata={"1": {"class_id": 0}, "2": {"class_id": 1}},
        )
        self.assertEqual(set(assignments), {"1", "2"})
        self.assertNotEqual(
            assignments["1"]["detection_index"],
            assignments["2"]["detection_index"],
        )

    def test_projection_matching_without_scipy_keeps_maximum_cardinality(self) -> None:
        cost = np.array([[1.0, 2.0], [1.0, np.inf]], dtype=np.float64)
        pairs = module._minimum_cost_maximum_assignment_without_scipy(cost)
        self.assertEqual(set(pairs), {(0, 1), (1, 0)})
        with patch.dict(sys.modules, {"scipy": None, "scipy.optimize": None}):
            assignments = module._assign_detections_to_projection(
                [
                    {"center": [0.0, 0.0], "class_id": 0, "confidence": 0.0},
                    {"center": [2.0, 0.0], "class_id": 0, "confidence": 0.0},
                ],
                {"1": np.array([0.0, 0.0]), "2": np.array([2.0, 0.0])},
                3.0,
            )
        self.assertEqual(set(assignments), {"1", "2"})

    def test_batch_fine_caps_frames_and_locks_stable_holes(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        bundle = SimpleNamespace(
            color_bgr=image,
            intrinsics=self.intrinsics,
            host_timestamp_ns=789,
        )
        selected = [
            {
                "hole_id": 1,
                "initial_center_base_mm": np.array([0.0, 0.0, 1000.0]),
                "initial_detection": {
                    "class_id": 0, "box": [620.0, 340.0, 660.0, 380.0],
                },
            },
            {
                "hole_id": 2,
                "initial_center_base_mm": np.array([100.0, 0.0, 1000.0]),
                "initial_detection": {
                    "class_id": 0, "box": [700.0, 340.0, 740.0, 380.0],
                },
            },
        ]
        detections = [
            {"box": [620.0, 340.0, 660.0, 380.0], "center": [640.0, 360.0], "class_id": 0, "confidence": 0.9},
            {"box": [700.0, 340.0, 740.0, 380.0], "center": [720.0, 360.0], "class_id": 0, "confidence": 0.8},
        ]
        good_ellipse = {
            "center_px": np.array([640.0, 360.0]), "axes_px": np.array([40.0, 40.0]),
            "residual_px": 0.2, "coverage_deg": 300.0, "roundness": 0.98,
        }
        # 可通过旧“放宽椭圆门”，但不能通过严格门；共享精定位必须拒绝，
        # 不允许再用YOLO框中心把这一帧计为有效。
        bad_ellipse = {
            "center_px": np.array([720.0, 360.0]), "axes_px": np.array([40.0, 40.0]),
            "residual_px": 1.8, "coverage_deg": 300.0, "roundness": 1.0,
        }
        cfg = module.TwoStageConfig(
            batch_fine_frames=3,
            batch_fine_min_valid=2,
            batch_fine_stable_min_frames=2,
            batch_fine_settle_discard_frames=0,
            fine_pointcloud_anchor_tolerance_px=30.0,
        )
        handeye = SimpleNamespace(T_tcp_rgb_camera=np.eye(4, dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, "get_rgb_frame_bundle", return_value=bundle) as get_frame, \
                patch.object(module, "detect", return_value=detections) as detect, \
                patch.object(module, "fit_hole_ellipse", side_effect=[good_ellipse, bad_ellipse, good_ellipse, bad_ellipse, bad_ellipse]) as fit_ellipse, \
                patch.object(module.cv2, "imwrite", return_value=True):
            result = module._batch_fine_localization_at_260mm(
                selected, np.eye(4, dtype=np.float64), handeye,
                object(), object(), 0.5, cfg, self.intrinsics,
                Path(directory), module.TimingRecorder(), [],
            )

        self.assertEqual(get_frame.call_count, 3)
        self.assertEqual(detect.call_count, 3)
        self.assertEqual(fit_ellipse.call_count, 5)
        self.assertTrue(result[1]["success"])
        self.assertEqual(result[1]["fine"]["valid_frames"], 2)
        self.assertFalse(result[2]["success"])
        self.assertEqual(result["_batch_metadata"]["max_attempts"], 3)
        self.assertEqual(result["_batch_metadata"]["locked_holes"], [1])

    def test_batch_fine_final_result_overlay_projects_fine_target_and_tcp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            source = run_dir / "shared_fine.png"
            self.assertTrue(cv2.imwrite(str(source), np.zeros((720, 1280, 3), dtype=np.uint8)))
            batch_results = {
                1: {
                    "success": True,
                    "fine_capture_overlay_path": str(source),
                    "capture_tcp": np.eye(4, dtype=np.float64),
                    "intrinsics": self.intrinsics,
                },
            }
            final_results = [{
                "status": "completed",
                "hole_id": 1,
                "batch_fine_source": "batch_fine_at_260mm",
                "hole_center_base_mm": [0.0, 0.0, 1000.0],
                "target_point_base_mm": [10.0, 0.0, 1000.0],
                "final_tcp_pose_m_rad": [0.020, 0.0, 1.0, 0.0, 0.0, 0.0],
                "final_xy_motion": {},
            }]
            handeye = SimpleNamespace(T_tcp_rgb_camera=np.eye(4, dtype=np.float64))
            overlay = module._save_batch_fine_final_result_overlay(
                run_dir, batch_results, final_results, handeye,
            )

            self.assertIsNotNone(overlay)
            assert overlay is not None
            self.assertTrue(Path(overlay["image_path"]).is_file())
            self.assertEqual(overlay["holes"][0]["hole_id"], 1)
            self.assertGreater(
                float(overlay["holes"][0]["final_tcp_px"][0]),
                float(overlay["holes"][0]["fine_point_px"][0]),
            )

    def test_batch_fine_final_motion_restores_coarse_reference_pose_safely(self) -> None:
        current = make_transform(np.eye(3), np.array([300.0, 0.0, 120.0]))
        captured = make_transform(self.R_down, np.array([100.0, 40.0, 260.0]))
        args = SimpleNamespace(
            speed_m_s=0.08, acc_m_s2=0.25,
            transit_speed_m_s=0.15, transit_acc_m_s2=0.45,
            approach_speed_m_s=0.12, approach_acc_m_s2=0.35,
        )
        with patch.object(module, "_move_to_fine_pose", return_value=captured) as move_fine:
            result = module._restore_batch_fine_pose_for_final_motion(
                "2", 1, current, captured, args, object(), object(),
                target_height_mm=255.0,
            )
        move_fine.assert_called_once()
        self.assertEqual(
            move_fine.call_args.kwargs["descent_guard_mm"],
            module.SHARED_OBSERVATION_MIN_DESCENT_MM,
        )
        self.assertEqual(move_fine.call_args.kwargs["target_height_mm"], 255.0)
        self.assertTrue(np.allclose(result, captured))

    def test_batch_coarse_per_hole_fusion_keeps_valid_shared_cache_hole(self) -> None:
        """A missing peer must not discard another hole's shared-pose frames."""
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        bundle = SimpleNamespace(
            color_bgr=image,
            xyz_map_mm=np.zeros((720, 1280, 3), dtype=np.float32),
            intrinsics=self.intrinsics,
            host_timestamp_ns=123,
        )
        selected = [
            {
                "hole_id": 1,
                "initial_center_base_mm": np.array([0.0, 0.0, 1000.0]),
                "initial_detection": {"class_id": 0},
            },
            {
                "hole_id": 2,
                "initial_center_base_mm": np.array([100.0, 0.0, 1000.0]),
                "initial_detection": {"class_id": 0},
            },
        ]
        plane_info = {
            "plane_point_camera_mm": [0.0, 0.0, 1000.0],
            "plane_normal_camera": [0.0, 0.0, 1.0],
            "plane_rmse_mm": 1.0,
            "ring_points": 30,
            "local_plane_point_camera_mm": [0.0, 0.0, 1000.0],
            "points_camera_mm": [[0.0, 0.0, 1000.0], [1.0, 0.0, 1000.0]],
            "surface_model": "front_surface_outer_ring_v2",
            "surface_selection_policy": module.COARSE_SURFACE_SELECTION_POLICY,
            "front_surface_z_mm": 1000.0,
            "ring_points_raw": 40,
            "surface_points_selected": 30,
        }
        first_hole = {
            "box": [620.0, 340.0, 660.0, 380.0],
            "center": [640.0, 360.0],
            "class_id": 0,
        }
        cfg = module.TwoStageConfig(
            batch_coarse_frames=3,
            batch_coarse_min_valid=2,
            # Shared cache validation uses this setting so each hole can be
            # fused from its own valid observations at the common pose.
            batch_coarse_min_holes_per_frame=1,
            batch_coarse_settle_discard_frames=0,
            multi_coarse_tracking_tolerance_px=80.0,
        )
        handeye = SimpleNamespace(T_tcp_rgb_camera=np.eye(4, dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, "get_aligned_frame_bundle", return_value=bundle), \
                patch.object(module, "detect", side_effect=[[first_hole], [first_hole], []]), \
                patch.object(
                    module, "hole_camera_point",
                    return_value=(np.array([0.0, 0.0, 1000.0]), plane_info),
                ), \
                patch.object(module.cv2, "imwrite", return_value=True), \
                patch.object(module, "_save_coarse_pointcloud_image", return_value=None):
            result = module._batch_coarse_localization_at_340mm(
                selected, np.eye(4, dtype=np.float64), handeye,
                object(), object(), object(), object(), 0.5, cfg,
                self.intrinsics, Path(directory), module.TimingRecorder(), [],
            )

        self.assertTrue(result[1]["success"])
        self.assertEqual(result[1]["valid_frames"], 2)
        self.assertFalse(result[2]["success"])
        self.assertEqual(result[2]["valid_frames"], 0)
        self.assertEqual(result["_batch_metadata"]["required_holes_per_frame"], 1)

    def test_batch_anchor_correction_removes_systematic_projection_error(self) -> None:
        projected = {
            "1": np.array([500.0, 250.0]),
            "2": np.array([650.0, 250.0]),
            "3": np.array([500.0, 450.0]),
            "4": np.array([650.0, 450.0]),
        }
        detections = [
            {"center": [517.0, 242.0], "box": [500.0, 225.0, 534.0, 259.0]},
            {"center": [661.0, 238.0], "box": [644.0, 221.0, 678.0, 255.0]},
            {"center": [517.0, 434.0], "box": [500.0, 417.0, 534.0, 451.0]},
            {"center": [661.0, 430.0], "box": [644.0, 413.0, 678.0, 447.0]},
        ]

        corrected, info = module._fit_batch_projected_anchor_correction(
            detections, projected, max_distance_px=70.0, min_matches=4,
        )

        self.assertTrue(info["enabled"])
        self.assertEqual(info["match_count"], 4)
        np.testing.assert_allclose(corrected["1"], [517.0, 242.0], atol=1e-6)
        np.testing.assert_allclose(corrected["4"], [661.0, 430.0], atol=1e-6)
        self.assertLess(info["residual_p95_px"], 1e-6)

    def test_comparison_diagnostics_excludes_wait_and_preserves_path(self) -> None:
        result = {
            "status": "completed",
            "coarse_source": "batch_coarse_localization",
            "batch_coarse_requested": True,
            "coarse_cache_event": {"cache_reused": False, "cache_validation_failed": False},
            "initial_center_base_mm": [0.0, 0.0, 0.0],
            "coarse_center_base_mm": [3.0, 4.0, 0.0],
            "hole_center_base_naive_mm": [3.0, 4.0, 2.0],
            "hole_center_base_mm": [3.0, 4.0, 2.1],
            "coarse_normal_toward_camera_base": [0.0, 0.0, 1.0],
            "plane_normal_toward_camera_base": [0.0, 0.0, 1.0],
            "estimated_height_mm": 260.2,
            "timing": {"events": [
                {"name": "hole_01/navigate_to_coarse", "elapsed_s": 2.0},
                {"name": "hole_01/coarse_capture_1", "elapsed_s": 1.0},
                {"name": "hole_01/coarse_settle_buffer", "elapsed_s": 9.0},
                {"name": "hole_01/wait_next_hole_confirmation", "elapsed_s": 20.0},
            ]},
        }

        diagnostics = module._build_comparison_hole_diagnostics(result)

        self.assertTrue(diagnostics["coarse_path"]["batch_used"])
        self.assertEqual(diagnostics["geometry_change"]["initial_to_coarse"]["norm_mm"], 5.0)
        self.assertAlmostEqual(
            diagnostics["motion_quality"]["effective_timing"]["effective_total_s"], 3.0,
        )
        self.assertAlmostEqual(
            diagnostics["motion_quality"]["effective_timing"]["excluded_wait_s"], 29.0,
        )

    def test_batch_coarse_fusion_rejects_unstable_centers(self) -> None:
        cfg = module.TwoStageConfig(min_coarse_valid=3)
        observations = [
            module.Observation(
                "batch_coarse",
                index,
                np.asarray(center, dtype=np.float64),
                plane=module.PlaneEstimate(
                    point_camera_mm=np.array([0.0, 0.0, 340.0]),
                    normal_camera=np.array([0.0, 0.0, 1.0]),
                    rmse_mm=1.0,
                    ring_points=100,
                ),
                tracking_distance_px=float(distance),
            )
            for index, (center, distance) in enumerate(
                [([640.0, 360.0], 1.0), ([650.0, 360.0], 2.0), ([660.0, 360.0], 3.0)]
            )
        ]

        with self.assertRaisesRegex(RuntimeError, "中心稳定性失败"):
            module._fuse_coarse(
                observations,
                cfg,
                min_valid_frames=3,
                max_center_scatter_p95_px=0.8,
                max_tracking_distance_p95_px=12.0,
            )

    def test_sequential_coarse_geometry_does_not_accept_unstable_capture(self) -> None:
        cfg = module.TwoStageConfig(min_coarse_valid=3)
        observations = [
            module.Observation(
                "coarse",
                index,
                np.asarray([640.0 + index * 3.0, 360.0], dtype=np.float64),
                plane=module.PlaneEstimate(
                    point_camera_mm=np.array([0.0, 0.0, 340.0]),
                    normal_camera=np.array([0.0, 0.0, 1.0]),
                    rmse_mm=1.0,
                    ring_points=100,
                ),
                tracking_distance_px=float(index + 1),
            )
            for index in range(3)
        ]

        with self.assertRaisesRegex(RuntimeError, "中心稳定性失败"):
            module._apply_coarse_geometry_to_hole(
                {"hole_id": 1}, observations, np.eye(4, dtype=np.float64), cfg,
            )

    def test_batch_success_skips_per_hole_340_capture_and_moves_to_fine_pose(self) -> None:
        class StopAfterBatchFine(RuntimeError):
            pass

        selected = []
        for hole_id, x_mm in ((1, 0.0), (2, 100.0)):
            selected.append({
                "hole_id": hole_id,
                "initial_center_base_mm": np.array([x_mm, 0.0, 0.0]),
                "initial_plane_normal_base": np.array([0.0, 0.0, 1.0]),
                "initial_detection": {"class_id": 0},
            })
        batch_result = {
            "success": True,
            "center_px": [640.0, 360.0],
            "center_camera_mm": [0.0, 0.0, 340.0],
            "center_base_mm": [0.0, 0.0, 0.0],
            "plane_point_camera_mm": [0.0, 0.0, 340.0],
            "plane_point_base_mm": [0.0, 0.0, 0.0],
            "normal_camera": [0.0, 0.0, -1.0],
            "normal_base": [0.0, 0.0, 1.0],
            "plane_rmse_mm": 1.0,
            "valid_frames": 15,
            "total_frames": 15,
            "center_scatter_p95_px": 0.2,
            "coarse_captures": [{"capture_index": 0, "mode": "batch_coarse_at_340mm"}],
            "pointcloud_image_path": "cloud.png",
            "batch_pointcloud_archive_path": "batch.npz",
        }
        batch_results = {
            1: dict(batch_result),
            2: {**batch_result, "center_base_mm": [100.0, 0.0, 0.0]},
            "_batch_metadata": {},
        }
        plan = {
            "hole_order": [1, 2],
            "target_tcp_pose_m_rad": [0.0] * 6,
            "group_point_base_mm": np.zeros(3),
            "group_normal_toward_camera_base": np.array([0.0, 0.0, 1.0]),
            "projected_holes_px": {1: np.array([640.0, 360.0]), 2: np.array([720.0, 360.0])},
            "group_bbox_px": [640.0, 360.0, 720.0, 360.0],
            "group_center_px": np.array([680.0, 360.0]),
            "view_margin_px": 50.0,
        }
        args = SimpleNamespace(
            execute=True,
            reuse_coarse_cache=False,
            reuse_persistent_coarse_cache=False,
            confidence=0.5,
            speed_m_s=0.08,
            acc_m_s2=0.25,
            transit_speed_m_s=0.15,
            transit_acc_m_s2=0.45,
            approach_speed_m_s=0.12,
            approach_acc_m_s2=0.35,
            move_final_xy=False,
            tcp_xy_offset_mm=None,
            final_target_mode="normal",
            handeye="handeye.json",
        )
        cfg = module.TwoStageConfig(
            batch_coarse_localization=True,
            batch_coarse_frames=15,
            batch_coarse_min_valid=10,
            batch_fine_localization=False,
        )
        report = {"stages": {}, "camera": {}}
        runtime = {"rgbd_pipeline": object(), "align": object(), "chain": object()}
        handeye = SimpleNamespace(
            T_tcp_rgb_camera=np.eye(4, dtype=np.float64),
            validated_for_motion=True,
        )
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, "_plan_batch_coarse_group_pose", return_value=(np.eye(4), plan)), \
                patch.object(module, "_confirm_and_move_line", side_effect=lambda _label, _current, target, *_args, **_kwargs: target), \
                patch.object(module, "_batch_coarse_localization_at_340mm", return_value=batch_results), \
                patch.object(module, "_plan_hole_tcp_pose_fixed_rz", return_value=(np.eye(4), {})), \
                patch.object(module, "_move_to_fine_pose", return_value=np.eye(4)) as move_fine, \
                patch.object(module, "_require_safe_snapshot", return_value=({}, np.eye(4))), \
                patch.object(module, "camera_height_to_plane_mm", return_value=260.0), \
                patch.object(module, "_project_base_point_to_pixel", return_value=np.array([640.0, 360.0])), \
                patch.object(module, "_capture_coarse_burst") as capture_coarse, \
                patch.object(module, "_capture_fine_with_recovery", side_effect=StopAfterBatchFine("reached fine capture")):
            with self.assertRaisesRegex(StopAfterBatchFine, "reached fine capture"):
                module._run_sequential_hole_workflow(
                    args, handeye, object(), cfg, Path(directory), report,
                    module.TimingRecorder(), [], runtime, object(), object(),
                    np.eye(4), selected, self.intrinsics,
                )

        capture_coarse.assert_not_called()
        move_fine.assert_called_once()

    def test_multi_hole_two_capture_mode_never_splits_selected_holes(self) -> None:
        selected = []
        for hole_id, x_mm in ((1, -40.0), (2, 0.0), (3, 50.0)):
            selected.append({
                "hole_id": hole_id,
                "initial_center_px": [640.0 + x_mm, 360.0],
                "initial_center_base_mm": [x_mm, 0.0, 0.0],
                "initial_point_camera_mm": [x_mm, 0.0, 900.0],
                "initial_plane_point_base_mm": [0.0, 0.0, 0.0],
                "initial_plane_point_camera_mm": [0.0, 0.0, 900.0],
                "initial_plane_normal_base": [0.0, 0.0, 1.0],
                "initial_plane_normal_camera": [0.0, 0.0, -1.0],
                "initial_shared_plane_point_base_mm": [0.0, 0.0, 0.0],
                "initial_shared_plane_point_camera_mm": [0.0, 0.0, 900.0],
                "initial_shared_plane_normal_base": [0.0, 0.0, 1.0],
                "initial_shared_plane_normal_camera": [0.0, 0.0, -1.0],
                "initial_detection": {"class_id": 0, "center": [640.0 + x_mm, 360.0]},
                "pointcloud_segmentation": "test",
            })
        coarse_results = {
            hole["hole_id"]: {
                "success": True,
                "center_base_mm": hole["initial_center_base_mm"],
                "normal_base": [0.0, 0.0, 1.0],
            }
            for hole in selected
        }
        coarse_results["_batch_metadata"] = {}
        fine_results = {
            hole["hole_id"]: {"success": False, "error": "test_stop_after_shared_capture"}
            for hole in selected
        }
        fine_results["_batch_metadata"] = {}
        planning_calls = []

        def plan_all(group, _current, _handeye, _rz, _intrinsics, height, _margin):
            planning_calls.append((
                [int(hole["hole_id"]) for hole in group],
                float(height),
                [np.asarray(hole["initial_center_base_mm"]).tolist() for hole in group],
            ))
            return np.eye(4), {
                "target_tcp_pose_m_rad": [0.0] * 6,
                "projected_holes_px": {
                    int(hole["hole_id"]): np.array([640.0, 360.0]) for hole in group
                },
                "group_bbox_px": [600.0, 340.0, 680.0, 380.0],
                "group_center_px": np.array([640.0, 360.0]),
            }

        args = SimpleNamespace(
            execute=True, reuse_coarse_cache=False, reuse_persistent_coarse_cache=False,
            shared_cache_validation=False, confidence=0.5, speed_m_s=0.08,
            acc_m_s2=0.25, transit_speed_m_s=0.15, transit_acc_m_s2=0.45,
            approach_speed_m_s=0.12, approach_acc_m_s2=0.35,
            move_final_xy=False, tcp_xy_offset_mm=None, final_target_mode="normal",
            handeye="handeye.json", optimize_hole_order=False,
        )
        cfg = module.TwoStageConfig(
            batch_coarse_frames=1, batch_coarse_min_valid=1,
            batch_fine_localization=True, batch_fine_frames=1,
            batch_fine_min_valid=1, batch_fine_stable_min_frames=1,
        )
        report = {"stages": {}, "camera": {}, "coarse_cache": {}}
        runtime = {"rgbd_pipeline": object(), "align": object(), "chain": object()}
        handeye = SimpleNamespace(T_tcp_rgb_camera=np.eye(4), validated_for_motion=True)

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, "_split_shared_cache_validation_groups") as split_groups, \
                patch.object(module, "_plan_batch_coarse_group_pose", side_effect=plan_all), \
                patch.object(module, "_confirm_and_move_line", side_effect=lambda _l, _c, target, *_a, **_k: target), \
                patch.object(module, "_move_to_shared_fine_pose", side_effect=lambda _l, _o, _c, target, *_a, **_k: target), \
                patch.object(module, "_batch_coarse_localization_at_340mm", return_value=coarse_results) as coarse_capture, \
                patch.object(module, "_batch_fine_localization_at_260mm", return_value=fine_results) as fine_capture:
            result = module._run_sequential_hole_workflow(
                args, handeye, object(), cfg, Path(directory), report,
                module.TimingRecorder(), [], runtime, object(), object(),
                np.eye(4), selected, self.intrinsics,
            )

        self.assertEqual(result, 0)
        split_groups.assert_not_called()
        coarse_capture.assert_called_once()
        self.assertEqual(fine_capture.call_count, 2)
        self.assertEqual(
            [call[0] for call in planning_calls],
            [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        )
        self.assertEqual([call[1] for call in planning_calls], [340.0, 260.0, 260.0])
        self.assertEqual(planning_calls[1][2], [[-40.0, 0.0, 0.0], [0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
        self.assertTrue(report["stages"]["batch_coarse_plan"]["all_selected_holes_single_group"])
        self.assertTrue(report["stages"]["batch_fine_plan"]["all_selected_holes_single_group"])

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
        self.assertEqual(parser.parse_args([]).coarse_recapture_settle_discard_frames, 10)
        self.assertEqual(parser.parse_args([]).coarse_max_corrections, 2)
        self.assertEqual(parser.parse_args([]).fine_settle_discard_frames, 10)
        self.assertEqual(parser.parse_args([]).batch_fine_settle_discard_frames, 10)
        self.assertEqual(parser.parse_args([]).fine_retries, 2)


if __name__ == "__main__":
    unittest.main()
