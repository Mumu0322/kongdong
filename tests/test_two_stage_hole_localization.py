from __future__ import annotations

import json
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
from aubo_workbench.auto_sector_selection import AutoSectorConfig
from aubo_workbench import hole_capture_workflow
from aubo_workbench.hole_localization_models import Observation, PlaneEstimate
from aubo_workbench import group_pose_workflow
from aubo_workbench import sequential_hole_execution


class TwoStageGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.T_tcp_camera = np.eye(4)
        self.R_down = np.diag([1.0, -1.0, -1.0])
        self.intrinsics = CameraIntrinsics(1280, 720, 800.0, 800.0, 640.0, 360.0, ())

    def test_coarse_map_build_never_moves_to_per_hole_fine_fallback(self) -> None:
        args = SimpleNamespace(map_build_coarse_only=True)
        self.assertEqual(
            sequential_hole_execution._per_hole_fine_route(
                args, batch_fine_available=False,
            ),
            "coarse_map_only",
        )
        self.assertFalse(
            sequential_hole_execution._should_move_to_per_hole_fine(
                args, batch_fine_available=False,
            )
        )
        args.map_build_coarse_only = False
        self.assertEqual(
            sequential_hole_execution._per_hole_fine_route(
                args, batch_fine_available=False,
            ),
            "per_hole_fallback",
        )
        self.assertTrue(
            sequential_hole_execution._should_move_to_per_hole_fine(
                args, batch_fine_available=False,
            )
        )
        self.assertEqual(
            sequential_hole_execution._per_hole_fine_route(
                args, batch_fine_available=True,
            ),
            "batch_fine_result",
        )
        self.assertFalse(
            sequential_hole_execution._should_move_to_per_hole_fine(
                args, batch_fine_available=True,
            )
        )
        args.coarse_direct_final = True
        self.assertEqual(
            sequential_hole_execution._per_hole_fine_route(
                args, batch_fine_available=False,
            ),
            "coarse_direct_final",
        )
        self.assertFalse(
            sequential_hole_execution._should_move_to_per_hole_fine(
                args, batch_fine_available=False,
            )
        )

    def test_fuse_coarse_uses_only_pointcloud_plane_for_direct_strategy(self) -> None:
        cfg = SimpleNamespace(
            min_coarse_valid=3,
            coarse_direct_final=True,
        )
        observations = []
        for frame_index, center in enumerate(
            (
                [100.0, 100.0],
                [102.0, 100.0],
                [102.2, 100.1],
            )
        ):
            observations.append(Observation(
                "batch_coarse", frame_index, np.asarray(center, dtype=np.float64),
                plane=PlaneEstimate(
                    point_camera_mm=np.asarray([
                        12.0,
                        20.0,
                        300.0,
                    ]),
                    normal_camera=np.asarray([0.0, 0.0, 1.0]),
                    rmse_mm=0.5,
                    ring_points=100,
                ),
                center_source="pointcloud_center",
                tracking_distance_px=0.2,
            ))

        summary = module._fuse_coarse(
            observations, cfg, min_valid_frames=3,
            max_center_scatter_p95_px=10.0,
            max_tracking_distance_p95_px=1.0,
        )

        np.testing.assert_allclose(summary["center_px"], [102.0, 100.0])
        np.testing.assert_allclose(summary["plane_point_camera_mm"], [12.0, 20.0, 300.0])
        self.assertEqual(summary["center_fusion_source"], "pointcloud_center")
        self.assertEqual(summary["geometric_valid_frames"], 0)
        self.assertEqual(summary["center_source_counts"], {"pointcloud_center": 3})

    def test_direct_strategy_routes_successful_batch_pointcloud_to_direct_handler(self) -> None:
        """批量点云成功时不能误入通用260 mm精定位分支。"""
        current_tcp = np.eye(4, dtype=np.float64)
        batch_result = {
            "success": True,
            "center_px": [640.0, 360.0],
            "center_camera_mm": [10.0, 20.0, 320.0],
            "center_base_mm": [10.0, 20.0, 320.0],
            "plane_point_camera_mm": [10.0, 20.0, 320.0],
            "plane_point_base_mm": [10.0, 20.0, 320.0],
            "normal_camera": [0.0, 0.0, 1.0],
            "normal_base": [0.0, 0.0, 1.0],
            "plane_rmse_mm": 0.5,
            "valid_frames": 10,
            "total_frames": 15,
            "center_scatter_p95_px": 0.2,
            "coarse_captures": [{
                "capture_index": 0,
                "mode": "batch_coarse_at_320mm_pointcloud_only",
                "capture_height_mm": 320.0,
            }],
            "capture_height_mm": 320.0,
        }
        hole = {
            "hole_id": 1,
            "initial_selection_order": 1,
            "initial_center_px": [640.0, 360.0],
            "initial_center_base_mm": np.array([10.0, 20.0, 320.0]),
            "initial_plane_normal_base": np.array([0.0, 0.0, 1.0]),
            "initial_detection": {"class_id": 0},
        }
        args = SimpleNamespace(
            coarse_direct_final=True,
            coarse_direct_final_capture_only=True,
            move_final_xy=False,
            map_build_coarse_only=False,
        )
        cfg = module.TwoStageConfig(
            coarse_direct_final=True,
            coarse_height_mm=320.0,
            batch_coarse_localization=True,
            batch_fine_localization=False,
            batch_fine_joint_localization=False,
            batch_fine_pointcloud_xy_fusion=False,
        )
        report = {"stages": {}}
        ctx = SimpleNamespace(
            args=args,
            handeye=SimpleNamespace(T_tcp_rgb_camera=np.eye(4, dtype=np.float64)),
            model=object(),
            cfg=cfg,
            run_dir=Path(tempfile.mkdtemp()),
            report=report,
            timing=module.TimingRecorder(),
            rows=[],
            runtime={},
            pose_session=object(),
            motion_session=object(),
            current_tcp=current_tcp.copy(),
            initial_holes=[hole],
            initial_intrinsics=self.intrinsics,
            fixed_rz_rad=0.0,
            results=[],
            order_ids=[1],
            initial_pointcloud_reused_holes=[],
            all_selected_two_capture_mode=False,
            cache_entries={},
            cache_sources={},
            cache_source_ids={},
            cache_gates=object(),
            cache_enabled=False,
            persistent_enabled=False,
            persistent_entries={},
            coarse_cache_dir=None,
            persistent_cache_dir=None,
            cache_built_ids=set(),
            batch_coarse_results={1: batch_result},
            batch_coarse_for_cache=False,
            shared_cache_results={},
            shared_cache_failed_ids=set(),
            invalidated_cache_ids=set(),
            batch_fine_results={},
            batch_fine_plan={},
            ensure_rgbd_pipeline=lambda: (_ for _ in ()).throw(
                AssertionError("第四策略不应进入260 mm批量精定位")
            ),
        )
        direct_result = {
            "status": "capture_only",
            "hole_id": 1,
            "coarse_direct_decision": "capture_only",
            "localization_path": "coarse_320_direct",
            "target_point_base_mm": np.array([10.0, 20.0, 320.0]),
        }
        with (
            patch.object(sequential_hole_execution, "np", np, create=True),
            patch.object(
                sequential_hole_execution, "_unit", module._unit, create=True,
            ),
            patch.object(
                sequential_hole_execution,
                "_confirm_next_hole_if_needed",
                create=True,
            ),
            patch.object(
                sequential_hole_execution,
                "_run_coarse_direct_final",
                return_value=(current_tcp.copy(), direct_result),
            ) as direct_handler,
            patch.object(
                sequential_hole_execution,
                "_build_comparison_hole_diagnostics",
                return_value={},
                create=True,
            ),
            patch.object(
                sequential_hole_execution,
                "_write_progress_checkpoint",
                create=True,
            ),
        ):
            sequential_hole_execution._process_one_hole(ctx, 1, hole)

        direct_handler.assert_called_once()
        self.assertEqual(ctx.results[0]["status"], "capture_only")
        self.assertEqual(hole["coarse_center_base_mm"], [10.0, 20.0, 320.0])

    def test_direct_capture_only_skips_final_planning_and_motion(self) -> None:
        current_tcp = np.eye(4, dtype=np.float64)
        hole = {
            "hole_id": 2,
            "tracking_identity": "initial_selection_hole_2",
            "coarse_center_base_mm": np.array([12.0, 20.0, 320.0]),
            "coarse_plane_point_base_mm": np.array([12.0, 20.0, 320.0]),
            "coarse_normal_toward_camera_base": np.array([0.0, 0.0, 1.0]),
            "coarse_captures": [{
                "capture_index": 0,
                "mode": "batch_coarse_at_320mm_pointcloud_only",
                "capture_height_mm": 320.0,
            }],
        }
        ctx = SimpleNamespace(
            args=SimpleNamespace(
                coarse_direct_final_capture_only=True,
                move_final_xy=False,
                tcp_xy_offset_mm=None,
            ),
            handeye=SimpleNamespace(T_tcp_rgb_camera=np.eye(4, dtype=np.float64)),
            cfg=SimpleNamespace(coarse_height_mm=320.0),
            timing=module.TimingRecorder(),
            report={},
            fixed_rz_rad=0.0,
            motion_session=object(),
            pose_session=object(),
        )
        with (
            patch.object(sequential_hole_execution, "np", np, create=True),
            patch.object(
                sequential_hole_execution, "_unit", module._unit, create=True,
            ),
            patch.object(
                sequential_hole_execution,
                "_plan_hole_tcp_pose_fixed_rz",
                return_value=(current_tcp.copy(), {}),
                create=True,
            ),
            patch.object(
                sequential_hole_execution,
                "transform_to_sdk_pose_m_rad",
                side_effect=lambda pose: [float(value) for value in np.asarray(pose).reshape(-1)],
                create=True,
            ),
            patch.object(
                sequential_hole_execution,
                "camera_height_to_plane_mm",
                return_value=320.0,
                create=True,
            ),
            patch.object(
                sequential_hole_execution,
                "_move_to_batch_final_tcp_direct",
                create=True,
            ) as final_motion,
            patch.object(
                sequential_hole_execution,
                "plan_final_tcp_xy",
                create=True,
            ) as final_xy_plan,
            patch.object(
                sequential_hole_execution,
                "plan_final_tcp_base_z",
                create=True,
            ) as final_z_plan,
            patch.object(sequential_hole_execution, "THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM", 60.0, create=True),
            patch.object(sequential_hole_execution, "SHARED_OBSERVATION_MIN_DESCENT_MM", 10.0, create=True),
        ):
            _, result = sequential_hole_execution._run_coarse_direct_final(
                ctx,
                1,
                hole,
                current_tcp,
                cache_event={"cache_reused": False},
                coarse_captures=hole["coarse_captures"],
                batch_coarse_for_cache=False,
            )

        self.assertEqual(result["status"], "capture_only")
        self.assertEqual(result["localization_path"], "coarse_320_direct")
        self.assertEqual(result["coarse_capture_height_mm"], 320.0)
        self.assertTrue(result["coarse_direct_final_capture_only"])
        np.testing.assert_allclose(
            result["target_point_base_mm"], hole["coarse_center_base_mm"],
        )
        self.assertNotIn("final_point_mode", result)
        self.assertNotIn("final_point_offset_base_mm", result)
        self.assertIsNone(result["final_xy_motion"])
        self.assertIsNone(result["final_z_motion"])
        self.assertIsNone(result["final_motion_direct"])
        final_motion.assert_not_called()
        final_xy_plan.assert_not_called()
        final_z_plan.assert_not_called()

    def test_coarse_map_build_returns_home_after_map_is_saved(self) -> None:
        final_tcp = np.eye(4, dtype=np.float64)
        final_tcp[:3, 3] = [100.0, 200.0, 300.0]
        report: dict[str, object] = {}
        runtime: dict[str, object] = {}
        timing = module.TimingRecorder()
        with (
            patch.object(
                module, "_resolve_home_joints",
                return_value=(np.arange(6, dtype=np.float64), "controller_home"),
            ),
            patch.object(module, "_confirm_and_move_home", return_value=final_tcp) as move_home,
            patch.object(module, "_wait_robot_steady", return_value=({}, final_tcp)) as wait_steady,
            patch.object(module, "_write_report"),
        ):
            actual = module._return_home_after_coarse_map_build(
                home=None,
                motion_session=object(),
                pose_session=object(),
                timing=timing,
                report=report,
                rows=[],
                run_dir=Path("unused"),
                pipeline_runtime=runtime,
            )
        move_home.assert_called_once()
        wait_steady.assert_called_once()
        np.testing.assert_allclose(actual, final_tcp)
        np.testing.assert_allclose(runtime["current_tcp"], final_tcp)
        self.assertEqual(
            report["session_end_reason"],
            "coarse_map_build_completed_returned_home",
        )
        self.assertTrue(report["return_to_home"]["completed"])

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

    def test_timing_recorder_separates_artifact_io_from_effective_time(self) -> None:
        report: dict[str, object] = {}
        timer = module.TimingRecorder()
        timer.attach_report(report)
        with timer.measure("hole_01/fine_capture"):
            with timer.measure_artifact(
                "fine_overlay/write",
                artifact_kind="fine_frame_overlay",
                paths=["fine_overlay.png"],
            ):
                pass

        timing = report["timing"]
        artifact_event = next(
            event for event in timing["events"]
            if event["name"] == "artifact_io/fine_overlay/write"
        )
        parent_event = next(
            event for event in timing["events"]
            if event["name"] == "hole_01/fine_capture"
        )
        self.assertEqual(artifact_event["category"], "artifact_io")
        self.assertTrue(artifact_event["exclude_from_effective"])
        self.assertEqual(timing["artifact_io"]["event_count"], 1)
        self.assertGreaterEqual(timing["artifact_io"]["total_elapsed_s"], 0.0)
        self.assertLessEqual(
            parent_event["effective_elapsed_s"], parent_event["elapsed_s"],
        )
        effective = module._effective_timing_summary(timing["events"])
        self.assertEqual(effective["artifact_event_count"], 1)
        self.assertGreaterEqual(effective["artifact_io_s"], 0.0)

    def test_write_report_persists_artifact_timing_in_report_json(self) -> None:
        report: dict[str, object] = {"status": "running"}
        timer = module.TimingRecorder()
        timer.attach_report(report)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            module._write_report(run_dir, report, [], timing=timer)
            saved = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))

        timing = saved["timing"]
        self.assertGreaterEqual(timing["artifact_io"]["event_count"], 1)
        self.assertTrue(any(
            event["name"] == "artifact_io/report/write_frames_csv"
            for event in timing["events"]
        ))

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
                current, target, args, object(), object(),
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

    def test_shared_fine_in_group_pose_adjustment_clips_motion_and_locks_rz(self) -> None:
        current_rotation = (
            module.rotz(0.4) @ module.roty(0.05) @ module.rotx(-0.03)
        )
        planned_rotation = (
            module.rotz(0.4) @ module.roty(0.20) @ module.rotx(0.16)
        )
        current = make_transform(current_rotation, np.array([0.0, 0.0, 100.0]))
        planned = make_transform(planned_rotation, np.array([20.0, 0.0, 110.0]))

        target, details = module._bounded_shared_fine_pose_adjustment(
            current,
            planned,
            max_xy_mm=8.0,
            max_z_mm=3.0,
            max_rotation_deg=2.0,
        )

        self.assertAlmostEqual(float(np.linalg.norm(target[:2, 3])), 8.0)
        self.assertAlmostEqual(float(target[2, 3]), 103.0)
        self.assertLessEqual(details["applied_rotation_deg"], 2.0 + 1.0e-6)
        self.assertTrue(details["translation_clipped"])
        self.assertTrue(details["rotation_clipped"])
        self.assertTrue(details["rz_locked"])
        current_rz = module._matrix_to_rpy_zyx(current[:3, :3])[2]
        target_rz = module._matrix_to_rpy_zyx(target[:3, :3])[2]
        self.assertAlmostEqual(float(current_rz), float(target_rz), places=8)

    def test_shared_fine_in_group_adjustment_uses_one_direct_move_line(self) -> None:
        current = make_transform(np.eye(3), np.array([0.0, 0.0, 100.0]))
        target = make_transform(
            module.roty(np.radians(1.0)),
            np.array([6.0, -2.0, 102.0]),
        )
        calls: list[tuple[str, np.ndarray, str]] = []

        def fake_move(label, _current, destination, *_args, **kwargs):
            calls.append((label, np.asarray(destination).copy(), kwargs["motion_profile"]))
            return np.asarray(destination).copy()

        with patch.object(module, "_confirm_and_move_line", side_effect=fake_move):
            result = module._move_shared_fine_group_adjustment(
                "synthetic in-group adjustment",
                current,
                target,
                SimpleNamespace(),
                object(),
                object(),
                max_xy_mm=8.0,
                max_z_mm=3.0,
                max_rotation_deg=2.0,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "synthetic in-group adjustment")
        self.assertEqual(calls[0][2], "approach")
        np.testing.assert_allclose(result, target)

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

    def test_shared_coarse_motion_forwards_direct_strategy_height(self) -> None:
        current = make_transform(np.eye(3), np.array([-20.0, 30.0, 140.0]))
        target = make_transform(np.eye(3), np.array([80.0, -40.0, 100.0]))
        args = SimpleNamespace(
            speed_m_s=0.08, acc_m_s2=0.25,
            transit_speed_m_s=0.15, transit_acc_m_s2=0.45,
            approach_speed_m_s=0.12, approach_acc_m_s2=0.35,
        )
        with patch.object(
            module, "_move_to_shared_observation_pose", return_value=target,
        ) as move_observation:
            result = module._move_to_shared_coarse_pose(
                1, 1, current, target, args, object(), object(),
                target_height_mm=320.0,
            )

        self.assertTrue(np.allclose(result, target))
        self.assertEqual(
            move_observation.call_args.kwargs["target_height_mm"], 320.0,
        )

    def test_map_build_motion_uses_extra_z_margin_and_separate_attitude_move(self) -> None:
        current = make_transform(np.eye(3), np.array([-20.0, 30.0, 140.0]))
        target_rotation = module.rotz(np.deg2rad(30.0))
        target = make_transform(target_rotation, np.array([80.0, -40.0, 100.0]))
        args = SimpleNamespace(
            map_build_coarse_only=True,
            map_build_safe_z_margin_mm=100.0,
            speed_m_s=0.08, acc_m_s2=0.25,
            transit_speed_m_s=0.15, transit_acc_m_s2=0.45,
            approach_speed_m_s=0.12, approach_acc_m_s2=0.35,
        )
        calls: list[np.ndarray] = []

        def fake_move(_label, _current, destination, *_args, **_kwargs):
            calls.append(np.asarray(destination).copy())
            return np.asarray(destination).copy()

        with patch.object(module, "_confirm_and_move_line", side_effect=fake_move):
            result = module._move_to_shared_coarse_pose(
                1, 1, current, target, args, object(), object(),
            )

        self.assertEqual(len(calls), 5)
        self.assertAlmostEqual(float(calls[0][2, 3]), 200.0)
        self.assertTrue(np.allclose(calls[1][:3, :3], current[:3, :3]))
        self.assertTrue(np.allclose(calls[1][:2, 3], target[:2, 3]))
        self.assertTrue(np.allclose(calls[2][:3, :3], target[:3, :3]))
        self.assertAlmostEqual(float(calls[2][2, 3]), 200.0)
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

    def test_map_manual_selection_is_limited_to_sector_candidates_and_numbered_row_major(self) -> None:
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        bundle = SimpleNamespace(
            color_bgr=image,
            xyz_map_mm=np.zeros((120, 160, 3), dtype=np.float64),
            intrinsics=self.intrinsics,
        )
        detections = [
            {"box": [5.0, 5.0, 25.0, 25.0], "center": [15.0, 15.0], "class_id": 0,
             "class_name": "hole", "confidence": 0.95},
            {"box": [65.0, 7.0, 85.0, 27.0], "center": [75.0, 17.0], "class_id": 0,
             "class_name": "hole", "confidence": 0.90},
            {"box": [5.0, 55.0, 25.0, 75.0], "center": [15.0, 65.0], "class_id": 0,
             "class_name": "hole", "confidence": 0.85},
        ]
        config = AutoSectorConfig(
            partition_mode="polygons",
            regions=({"sector_id": 1, "polygon_px": [[0, 0], [160, 0], [160, 120], [0, 120]]},),
            image_size_px=(160, 120),
            dedup_distance_px=0.0,
        )
        plane = module.PlaneEstimate(
            point_camera_mm=np.array([0.0, 0.0, 1000.0]),
            normal_camera=np.array([0.0, 0.0, 1.0]),
            rmse_mm=1.0,
            ring_points=20,
            surface_model="ring",
        )
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, "get_aligned_frame_bundle", return_value=bundle), \
                patch.object(module, "detect", return_value=detections), \
                patch.object(
                    module, "choose_boxes",
                    return_value=([2, 1, 0], {2: [15.0, 65.0], 1: [75.0, 17.0], 0: [15.0, 15.0]}),
                ), \
                patch.object(module, "hole_camera_point", return_value=(np.array([0.0, 0.0, 1000.0]), {})), \
                patch.object(module, "_plane_estimate_from_info", return_value=plane), \
                patch.object(hole_capture_workflow, "write_sector_info_snapshots", return_value={}), \
                patch.object(module.cv2, "imwrite", return_value=True):
            _, holes, _, _, _ = module._capture_initial_multi_hole_selection(
                pipeline=object(), align=object(), chain=object(), model=object(),
                confidence=0.5, run_dir=Path(directory), max_plane_rmse_mm=3.5,
                auto_sector_config=config, map_hole_selection_mode="manual",
            )

        self.assertEqual([hole["hole_id"] for hole in holes], [1, 2, 3])
        self.assertEqual([hole["initial_center_px"] for hole in holes], [[15.0, 15.0], [75.0, 17.0], [15.0, 65.0]])
        self.assertEqual([hole["operator_selection_order"] for hole in holes], [3, 2, 1])
        self.assertEqual({hole["numbering_policy"] for hole in holes}, {"image_row_major_v1"})

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

    def test_direct_batch_capture_uses_configured_320mm_artifacts(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        xyz = np.zeros((720, 1280, 3), dtype=np.float32)
        bundle = SimpleNamespace(
            color_bgr=image,
            xyz_map_mm=xyz,
            intrinsics=self.intrinsics,
            host_timestamp_ns=123,
        )
        selected = [{
            "hole_id": 7,
            "initial_center_base_mm": np.array([0.0, 0.0, 320.0]),
            "initial_detection": {"class_id": 0},
        }]
        detections = [{
            "box": [620.0, 340.0, 660.0, 380.0],
            "center": [640.0, 360.0],
            "class_id": 0,
        }]
        plane_info = {
            "plane_point_camera_mm": [0.0, 0.0, 320.0],
            "plane_normal_camera": [0.0, 0.0, 1.0],
            "plane_rmse_mm": 1.0,
            "ring_points": 30,
            "local_plane_point_camera_mm": [0.0, 0.0, 320.0],
            "points_camera_mm": [[0.0, 0.0, 320.0], [1.0, 0.0, 320.0]],
            "surface_model": "front_surface_outer_ring_v2",
            "surface_selection_policy": module.COARSE_SURFACE_SELECTION_POLICY,
            "front_surface_z_mm": 320.0,
            "ring_points_raw": 40,
            "surface_points_selected": 30,
        }
        cfg = module.TwoStageConfig(
            coarse_direct_final=True,
            coarse_height_mm=320.0,
            batch_coarse_localization=True,
            batch_coarse_frames=1,
            batch_coarse_min_valid=1,
            batch_coarse_min_holes_per_frame=1,
            batch_coarse_settle_discard_frames=0,
            batch_coarse_early_stop_extra_frames=0,
            multi_coarse_tracking_tolerance_px=80.0,
        )
        handeye = SimpleNamespace(T_tcp_rgb_camera=np.eye(4, dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, "get_aligned_frame_bundle", return_value=bundle), \
                patch.object(module, "detect", return_value=detections), \
                patch.object(
                    module,
                    "hole_camera_point",
                    return_value=(np.array([0.0, 0.0, 320.0]), plane_info),
                ), \
                patch.object(module.cv2, "imwrite", return_value=True), \
                patch.object(module, "_save_coarse_pointcloud_image", return_value=None):
            result = module._batch_coarse_localization_at_340mm(
                selected, np.eye(4, dtype=np.float64), handeye,
                object(), object(), object(), object(), 0.5, cfg,
                self.intrinsics, Path(directory), module.TimingRecorder(), [],
            )

        self.assertTrue(result[7]["success"])
        self.assertEqual(result[7]["capture_height_mm"], 320.0)
        self.assertEqual(
            result[7]["coarse_captures"][0]["mode"],
            "batch_coarse_at_320mm_pointcloud_only",
        )
        self.assertEqual(result["_batch_metadata"]["target_height_mm"], 320.0)
        self.assertEqual(result["_batch_metadata"]["strategy"], "coarse_direct_pointcloud_only")
        self.assertEqual(
            result[7]["batch_pointcloud_archive_path"].split("\\")[-1],
            "batch_coarse_320_all_holes_pointcloud.npz",
        )

    def test_direct_batch_capture_stops_after_consecutive_frame_failures(self) -> None:
        selected = [{
            "hole_id": 7,
            "initial_center_base_mm": np.array([0.0, 0.0, 340.0]),
            "initial_detection": {"class_id": 0},
        }]
        cfg = module.TwoStageConfig(
            coarse_direct_final=True,
            coarse_height_mm=340.0,
            batch_coarse_localization=True,
            batch_coarse_frames=20,
            batch_coarse_min_valid=1,
            batch_coarse_min_holes_per_frame=1,
            batch_coarse_settle_discard_frames=0,
            coarse_direct_final_max_consecutive_frame_failures=2,
        )
        handeye = SimpleNamespace(T_tcp_rgb_camera=np.eye(4, dtype=np.float64))
        progress: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, "get_aligned_frame_bundle", return_value=None) as get_frame:
            result = module._batch_coarse_localization_at_340mm(
                selected, np.eye(4, dtype=np.float64), handeye,
                object(), object(), object(), object(), 0.5, cfg,
                self.intrinsics, Path(directory), module.TimingRecorder(), [],
                progress_callback=progress.append,
            )

        metadata = result["_batch_metadata"]
        self.assertEqual(get_frame.call_count, 2)
        self.assertEqual(metadata["capture_stop_reason"], "consecutive_frame_failures")
        self.assertTrue(metadata["max_consecutive_frame_failures_reached"])
        self.assertEqual(metadata["consecutive_frame_failures"], 2)
        self.assertTrue(metadata["capture_timed_out"] is False)
        self.assertIn("capture_frame", [item["phase"] for item in progress])
        self.assertEqual(progress[-1]["phase"], "complete")

    def test_batch_defaults_use_stable_bursts_for_all_selected_holes(self) -> None:
        cfg = module.TwoStageConfig()
        self.assertEqual(cfg.batch_coarse_frames, 15)
        self.assertEqual(cfg.batch_coarse_min_valid, 10)
        self.assertEqual(cfg.batch_coarse_max_view_span_ratio, 0.35)
        self.assertEqual(cfg.batch_coarse_max_group_size, 5)
        self.assertEqual(cfg.batch_coarse_group_max_aspect_ratio, 2.0)
        self.assertEqual(cfg.batch_coarse_group_max_xy_diameter_mm, 150.0)
        self.assertEqual(cfg.batch_coarse_group_max_normal_spread_deg, 3.0)
        self.assertEqual(cfg.batch_fine_frames, 8)
        self.assertEqual(cfg.batch_fine_min_valid, 5)
        self.assertEqual(cfg.batch_fine_stable_min_frames, 5)
        self.assertEqual(cfg.batch_fine_inplace_recovery_frames, 4)
        self.assertEqual(cfg.batch_fine_supplement_rounds, 2)
        self.assertEqual(cfg.batch_fine_max_view_span_ratio, 0.55)
        self.assertEqual(cfg.batch_fine_max_geometric_anchor_distance_px, 5.0)
        self.assertEqual(cfg.batch_fine_fallback_max_coarse_to_fine_xy_mm, 3.0)
        self.assertEqual(cfg.batch_fine_supplement_max_view_span_ratio, 0.55)
        self.assertEqual(cfg.batch_fine_max_group_size, 4)
        self.assertTrue(cfg.batch_fine_in_group_pose_adjustment)
        self.assertEqual(cfg.batch_fine_in_group_max_adjustments, 2)
        self.assertEqual(cfg.batch_fine_in_group_max_xy_mm, 8.0)
        self.assertEqual(cfg.batch_fine_in_group_max_z_mm, 3.0)
        self.assertEqual(cfg.batch_fine_in_group_max_rotation_deg, 2.0)
        self.assertEqual(cfg.batch_fine_in_group_min_normal_holes, 2)
        self.assertEqual(cfg.batch_fine_in_group_max_normal_spread_deg, 3.0)
        self.assertEqual(cfg.batch_fine_group_max_aspect_ratio, 1.8)
        self.assertTrue(cfg.batch_fine_per_hole_fallback)
        args = module.build_parser().parse_args([])
        self.assertEqual(args.batch_coarse_frames, 15)
        self.assertEqual(args.batch_coarse_min_valid, 10)
        self.assertEqual(args.batch_fine_inplace_recovery_frames, 4)
        self.assertEqual(args.batch_coarse_max_view_span_ratio, 0.35)
        self.assertEqual(args.batch_coarse_max_group_size, 5)
        self.assertEqual(args.batch_coarse_group_max_xy_diameter_mm, 150.0)
        self.assertEqual(args.batch_coarse_group_max_normal_spread_deg, 3.0)
        self.assertEqual(args.batch_fine_frames, 8)
        self.assertEqual(args.batch_fine_min_valid, 5)
        self.assertEqual(args.batch_fine_stable_min_frames, 5)
        self.assertEqual(args.batch_fine_supplement_rounds, 2)
        self.assertEqual(args.batch_fine_max_view_span_ratio, 0.55)
        self.assertEqual(args.batch_fine_max_geometric_anchor_distance_px, 5.0)
        self.assertEqual(args.batch_fine_fallback_max_coarse_to_fine_xy_mm, 3.0)
        self.assertEqual(args.batch_fine_supplement_max_view_span_ratio, 0.55)
        self.assertEqual(args.batch_fine_max_group_size, 4)
        self.assertTrue(args.batch_fine_in_group_pose_adjustment)
        self.assertEqual(args.batch_fine_in_group_max_adjustments, 2)
        self.assertEqual(args.batch_fine_in_group_max_xy_mm, 8.0)
        self.assertEqual(args.batch_fine_in_group_max_z_mm, 3.0)
        self.assertEqual(args.batch_fine_in_group_max_rotation_deg, 2.0)
        self.assertEqual(args.batch_fine_in_group_min_normal_holes, 2)
        self.assertEqual(args.batch_fine_in_group_max_normal_spread_deg, 3.0)
        self.assertTrue(args.batch_fine_per_hole_fallback)
        self.assertTrue(args.batch_fine_joint_localization)
        self.assertTrue(cfg.batch_fine_pointcloud_xy_fusion)
        self.assertEqual(cfg.batch_fine_joint_max_residual_mm, 1.0)
        self.assertEqual(cfg.batch_fine_pointcloud_xy_weight, 0.35)
        self.assertEqual(cfg.batch_fine_pointcloud_xy_max_correction_mm, 0.6)
        self.assertEqual(cfg.batch_fine_pointcloud_xy_agreement_gate_mm, 2.5)
        self.assertTrue(args.batch_fine_pointcloud_xy_fusion)
        self.assertEqual(args.batch_fine_joint_max_residual_mm, 1.0)
        self.assertEqual(args.batch_fine_pointcloud_xy_weight, 0.35)
        self.assertEqual(args.batch_fine_pointcloud_xy_max_correction_mm, 0.6)
        self.assertEqual(args.batch_fine_pointcloud_xy_agreement_gate_mm, 2.5)
        self.assertFalse(
            module.build_parser().parse_args(["--no-batch-fine-joint-localization"])
            .batch_fine_joint_localization
        )
        self.assertFalse(
            module.build_parser().parse_args([
                "--no-batch-fine-pointcloud-xy-fusion",
            ]).batch_fine_pointcloud_xy_fusion
        )

    def test_batch_fine_joint_recovers_translation_and_yaw(self) -> None:
        source_points = np.asarray([
            [0.0, 0.0], [100.0, 0.0], [0.0, 80.0], [100.0, 80.0],
        ])
        yaw_rad = np.deg2rad(1.0)
        rotation = np.asarray([
            [np.cos(yaw_rad), -np.sin(yaw_rad)],
            [np.sin(yaw_rad), np.cos(yaw_rad)],
        ])
        translation = np.asarray([1.2, -0.7])
        target_points = (rotation @ source_points.T).T + translation
        result = module._fit_batch_fine_joint_transform(
            [1, 2, 3, 4],
            {hole_id: point for hole_id, point in zip([1, 2, 3, 4], source_points)},
            {hole_id: point for hole_id, point in zip([1, 2, 3, 4], target_points)},
            {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            min_holes=4,
            max_residual_mm=0.01,
            max_translation_mm=5.0,
            max_yaw_deg=3.0,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["inlier_hole_ids"], [1, 2, 3, 4])
        np.testing.assert_allclose(result["translation_mm"], translation, atol=1.0e-8)
        self.assertAlmostEqual(result["yaw_deg"], 1.0, places=7)
        np.testing.assert_allclose(
            np.asarray(result["predicted_xy_by_hole"][4]), target_points[3], atol=1.0e-8,
        )

    def test_batch_fine_joint_two_holes_uses_translation_only(self) -> None:
        source_points = {
            1: np.asarray([0.0, 0.0]),
            2: np.asarray([100.0, 0.0]),
        }
        translation = np.asarray([1.2, -0.7])
        target_points = {
            hole_id: point + translation
            for hole_id, point in source_points.items()
        }
        result = module._fit_batch_fine_joint_transform(
            [1, 2], source_points, target_points, {1: 1.0, 2: 1.0},
            min_holes=2,
            max_residual_mm=0.01,
            max_translation_mm=5.0,
            max_yaw_deg=3.0,
            estimate_rotation=False,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["fit_mode"], "translation_only")
        self.assertEqual(result["inlier_hole_ids"], [1, 2])
        self.assertAlmostEqual(result["yaw_deg"], 0.0, places=7)
        np.testing.assert_allclose(result["rotation"], np.eye(2), atol=1.0e-12)
        np.testing.assert_allclose(result["translation_mm"], translation, atol=1.0e-8)

    def test_batch_fine_joint_gates_local_hole_motion_not_origin_translation(self) -> None:
        source_points = np.asarray([
            [620.0, -120.0], [700.0, -120.0],
            [620.0, -40.0], [700.0, -40.0],
        ])
        centroid = np.mean(source_points, axis=0)
        yaw_rad = np.deg2rad(1.0)
        rotation = np.asarray([
            [np.cos(yaw_rad), -np.sin(yaw_rad)],
            [np.sin(yaw_rad), np.cos(yaw_rad)],
        ])
        local_shift = np.asarray([0.8, -0.6])
        target_points = (
            (rotation @ (source_points - centroid).T).T + centroid + local_shift
        )

        result = module._fit_batch_fine_joint_transform(
            [1, 2, 3, 4],
            {hole_id: point for hole_id, point in zip([1, 2, 3, 4], source_points)},
            {hole_id: point for hole_id, point in zip([1, 2, 3, 4], target_points)},
            {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            min_holes=4,
            max_residual_mm=0.01,
            max_translation_mm=5.0,
            max_yaw_deg=3.0,
        )

        self.assertTrue(result["success"])
        self.assertGreater(result["translation_norm_mm"], 5.0)
        self.assertLess(result["max_point_correction_mm"], 5.0)

    def test_batch_fine_joint_two_hole_conflict_falls_back_at_one_mm(self) -> None:
        # 复现 two-stage-20260903_182104 的 G03：两个精拍中心都很稳定，
        # 但各自粗到精位移互相矛盾，不能用一个共同平移可靠解释。
        source = {
            11: np.asarray([609.0222514232581, -178.14734778471097]),
            12: np.asarray([669.5143676269554, -222.94317560016646]),
        }
        target = {
            11: np.asarray([610.6233795896544, -177.9139667790402]),
            12: np.asarray([669.3774129378667, -221.53809583002908]),
        }
        weights = {11: 34.20316784493929, 12: 17.370281192419895}

        with self.assertRaisesRegex(ValueError, "联合平移残差过大"):
            module._fit_batch_fine_joint_transform(
                [12, 11], source, target, weights,
                min_holes=2,
                max_residual_mm=1.0,
                max_translation_mm=5.0,
                max_yaw_deg=3.0,
                estimate_rotation=False,
            )

    def test_batch_fine_joint_keeps_good_holes_when_three_hole_outlier_exists(self) -> None:
        source_points = {
            1: np.asarray([0.0, 0.0]),
            2: np.asarray([100.0, 0.0]),
            3: np.asarray([0.0, 100.0]),
        }
        yaw_rad = np.deg2rad(0.8)
        rotation = np.asarray([
            [np.cos(yaw_rad), -np.sin(yaw_rad)],
            [np.sin(yaw_rad), np.cos(yaw_rad)],
        ])
        translation = np.asarray([0.8, -0.5])
        good_target = {
            hole_id: rotation @ point + translation
            for hole_id, point in source_points.items()
        }
        target_points = dict(good_target)
        target_points[3] = target_points[3] + np.asarray([8.0, -6.0])
        result = module._fit_batch_fine_joint_transform(
            [1, 2, 3], source_points, target_points,
            {1: 1.0, 2: 1.0, 3: 1.0},
            min_holes=2,
            max_residual_mm=1.5,
            max_translation_mm=5.0,
            max_yaw_deg=3.0,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["inlier_hole_ids"], [1, 2])
        self.assertAlmostEqual(result["yaw_deg"], 0.8, places=7)

    def test_batch_fine_rejects_two_stable_center_clusters(self) -> None:
        centers = [
            [215.00, 483.00], [215.05, 483.02], [214.96, 482.98],
            [230.75, 469.20], [230.79, 469.18], [230.73, 469.23],
        ]
        observations = [
            module.Observation(
                "batch_fine", index, np.asarray(center, dtype=np.float64),
                ellipse={
                    "center_px": np.asarray(center, dtype=np.float64),
                    "center_px_distorted": np.asarray(center, dtype=np.float64),
                    "axes_px": np.asarray([40.0, 40.0]),
                    "residual_px": 0.2,
                    "coverage_deg": 300.0,
                    "roundness": 0.98,
                },
                center_source="hough_circle",
            )
            for index, center in enumerate(centers)
        ]
        cfg = module.TwoStageConfig(min_fine_valid=3)

        with self.assertRaisesRegex(RuntimeError, "双峰歧义"):
            module._fuse_fine(observations, cfg, reject_multimodal=True)

    def test_supplement_groups_split_wide_failed_hole_set(self) -> None:
        holes = [
            {"hole_id": index + 1, "initial_center_base_mm": [x, 0.0, 0.0]}
            for index, x in enumerate((-120.0, -40.0, 40.0, 120.0))
        ]

        def fake_plan(group, *_args, **_kwargs):
            projected = {
                int(hole["hole_id"]): np.asarray([
                    640.0 + float(hole["initial_center_base_mm"][0]) * 4.0,
                    360.0,
                ])
                for hole in group
            }
            values = np.asarray(list(projected.values()))
            bbox_min = np.min(values, axis=0)
            bbox_max = np.max(values, axis=0)
            return np.eye(4), {
                "projected_holes_px": projected,
                "group_bbox_px": [*bbox_min, *bbox_max],
            }

        with patch.object(module, "_plan_batch_coarse_group_pose", side_effect=fake_plan):
            groups = module._split_batch_fine_supplement_groups(
                holes, np.eye(4), object(), 0.0, self.intrinsics,
                260.0, 50.0, 0.60,
                max_aspect_ratio=1.8,
            )

        self.assertEqual(
            [[int(hole["hole_id"]) for hole in group] for group in groups],
            [[1, 2], [3, 4]],
        )

    def test_batch_fine_joint_rejects_one_outlier(self) -> None:
        source_points = np.asarray([
            [0.0, 0.0], [100.0, 0.0], [0.0, 80.0], [100.0, 80.0],
        ])
        target_points = source_points + np.asarray([1.0, -0.5])
        target_points[3] += np.asarray([12.0, -9.0])
        result = module._fit_batch_fine_joint_transform(
            [1, 2, 3, 4],
            {hole_id: point for hole_id, point in zip([1, 2, 3, 4], source_points)},
            {hole_id: point for hole_id, point in zip([1, 2, 3, 4], target_points)},
            {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            min_holes=3,
            max_residual_mm=1.5,
            max_translation_mm=5.0,
            max_yaw_deg=3.0,
        )

        self.assertEqual(result["inlier_hole_ids"], [1, 2, 3])
        np.testing.assert_allclose(result["translation_mm"], [1.0, -0.5], atol=1.0e-8)
        self.assertGreater(result["residual_mm_by_hole"][4], 1.5)

    def test_batch_fine_joint_stability_gate(self) -> None:
        stable = [
            {"translation_mm": [1.0, 2.0], "yaw_rad": np.deg2rad(0.10)},
            {"translation_mm": [1.08, 1.96], "yaw_rad": np.deg2rad(0.14)},
            {"translation_mm": [0.94, 2.03], "yaw_rad": np.deg2rad(0.06)},
        ]
        unstable = stable + [
            {"translation_mm": [1.8, 2.0], "yaw_rad": np.deg2rad(0.60)},
        ]
        self.assertTrue(
            module._batch_fine_joint_transform_stable(stable, 3, 0.25, 0.15)
        )
        self.assertFalse(
            module._batch_fine_joint_transform_stable(unstable, 4, 0.25, 0.15)
        )

    def test_batch_fine_joint_preserves_tilt_and_limits_local_residual(self) -> None:
        result, details = module._compose_batch_fine_joint_xy_with_tilt(
            np.asarray([10.0, -4.0, 900.0]),
            np.asarray([10.1, -3.8, 898.0]),
            np.asarray([8.0, -5.0, 900.0]),
            local_residual_weight=0.25,
            local_residual_limit_mm=0.5,
        )

        self.assertAlmostEqual(result[2], 898.0)
        self.assertAlmostEqual(details["local_residual_norm_mm"], np.sqrt(5.0))
        np.testing.assert_allclose(
            result[:2], [8.2118034, -4.7440983], atol=1.0e-6,
        )

    def test_batch_fine_pointcloud_prior_blends_and_limits_correction(self) -> None:
        result, details = module.fuse_batch_fine_xy_with_pointcloud_prior(
            np.asarray([10.0, 20.0, 900.0]),
            np.asarray([12.0, 20.0, 898.0]),
            pointcloud_weight=0.5,
            max_correction_mm=0.6,
            agreement_gate_mm=2.5,
        )

        np.testing.assert_allclose(result, [10.6, 20.0, 900.0], atol=1.0e-9)
        self.assertTrue(details["applied"])
        self.assertTrue(details["correction_limited"])
        self.assertAlmostEqual(details["disagreement_mm"], 2.0)
        self.assertAlmostEqual(details["correction_norm_mm"], 0.6)

    def test_batch_fine_pointcloud_prior_rejects_large_disagreement(self) -> None:
        visual = np.asarray([10.0, 20.0, 900.0])
        result, details = module.fuse_batch_fine_xy_with_pointcloud_prior(
            visual,
            np.asarray([13.0, 20.0, 898.0]),
            pointcloud_weight=0.35,
            max_correction_mm=0.6,
            agreement_gate_mm=2.5,
        )

        np.testing.assert_allclose(result, visual, atol=1.0e-9)
        self.assertFalse(details["applied"])
        self.assertEqual(details["status"], "rejected_disagreement")
        self.assertAlmostEqual(details["correction_norm_mm"], 0.0)

    def test_batch_fine_pointcloud_prior_adapts_weight_to_joint_residual(self) -> None:
        result, details = module.fuse_batch_fine_xy_with_pointcloud_prior(
            np.asarray([10.0, 20.0, 900.0]),
            np.asarray([12.0, 20.0, 898.0]),
            pointcloud_weight=0.5,
            max_correction_mm=0.6,
            agreement_gate_mm=2.5,
            joint_residual_mm=0.2,
            max_joint_residual_mm=1.0,
            adaptive_weight=True,
        )

        np.testing.assert_allclose(result, [10.4, 20.0, 900.0], atol=1.0e-9)
        self.assertAlmostEqual(details["configured_pointcloud_weight"], 0.5)
        self.assertAlmostEqual(details["pointcloud_weight"], 0.2)
        self.assertAlmostEqual(details["adaptive_weight_ratio"], 0.4)
        self.assertEqual(
            details["adaptive_weight_reason"], "scaled_by_joint_residual",
        )

    def test_batch_fine_pointcloud_prior_without_joint_uses_conservative_weight(self) -> None:
        result, details = module.fuse_batch_fine_xy_with_pointcloud_prior(
            np.asarray([10.0, 20.0, 900.0]),
            np.asarray([11.0, 20.0, 898.0]),
            pointcloud_weight=0.5,
            max_correction_mm=0.6,
            agreement_gate_mm=2.5,
            adaptive_weight=True,
        )

        np.testing.assert_allclose(result, [10.2, 20.0, 900.0], atol=1.0e-9)
        self.assertAlmostEqual(details["pointcloud_weight"], 0.2)
        self.assertEqual(
            details["adaptive_weight_reason"],
            "missing_joint_residual_uses_minimum",
        )

    def test_batch_fine_pointcloud_prior_smoothly_restores_weight(self) -> None:
        result, details = module.fuse_batch_fine_xy_with_pointcloud_prior(
            np.asarray([10.0, 20.0, 900.0]),
            np.asarray([11.0, 20.0, 898.0]),
            pointcloud_weight=0.5,
            max_correction_mm=0.6,
            agreement_gate_mm=2.5,
            joint_residual_mm=0.4,
            max_joint_residual_mm=1.0,
            adaptive_weight=True,
        )

        np.testing.assert_allclose(result, [10.35, 20.0, 900.0], atol=1.0e-9)
        self.assertAlmostEqual(details["adaptive_weight_ratio"], 0.7)
        self.assertAlmostEqual(details["pointcloud_weight"], 0.35)

    def test_batch_fine_pointcloud_prior_inherits_joint_residual_gate(self) -> None:
        visual = np.asarray([10.0, 20.0, 900.0])
        result, details = module.fuse_batch_fine_xy_with_pointcloud_prior(
            visual,
            np.asarray([9.0, 19.5, 898.0]),
            pointcloud_weight=0.35,
            max_correction_mm=0.6,
            agreement_gate_mm=2.5,
            joint_residual_mm=1.2,
            max_joint_residual_mm=1.0,
        )

        np.testing.assert_allclose(result, visual, atol=1.0e-9)
        self.assertFalse(details["applied"])
        self.assertEqual(details["status"], "rejected_joint_residual")

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
                "coarse_plane_point_base_mm": np.array([0.0, 0.0, 1000.0]),
                "coarse_normal_toward_camera_base": np.array([0.0, 0.0, 1.0]),
                "initial_detection": {
                    "class_id": 0, "box": [620.0, 340.0, 660.0, 380.0],
                },
            },
            {
                "hole_id": 2,
                "initial_center_base_mm": np.array([100.0, 0.0, 1000.0]),
                "coarse_plane_point_base_mm": np.array([100.0, 0.0, 1000.0]),
                "coarse_normal_toward_camera_base": np.array([0.0, 0.0, 1.0]),
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
            batch_fine_joint_min_valid_frames=2,
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
        self.assertTrue(result[1]["batch_fine_joint_success"])
        self.assertTrue(result[1]["batch_fine_joint_summary"]["success"])
        self.assertEqual(
            result[1]["batch_fine_joint_summary"]["fused_frame_count"], 2,
        )

    def test_batch_fine_uses_per_hole_fusion_when_frame_joint_scatter_is_large(self) -> None:
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
                "coarse_plane_point_base_mm": np.array([0.0, 0.0, 1000.0]),
                "coarse_normal_toward_camera_base": np.array([0.0, 0.0, 1.0]),
                "initial_detection": {
                    "class_id": 0, "box": [620.0, 340.0, 660.0, 380.0],
                },
            },
            {
                "hole_id": 2,
                "initial_center_base_mm": np.array([100.0, 0.0, 1000.0]),
                "coarse_plane_point_base_mm": np.array([100.0, 0.0, 1000.0]),
                "coarse_normal_toward_camera_base": np.array([0.0, 0.0, 1.0]),
                "initial_detection": {
                    "class_id": 0, "box": [700.0, 340.0, 740.0, 380.0],
                },
            },
        ]
        frame_offsets_px = [0.0, 0.5, -0.5]
        detect_call_index = 0

        def detect_frame(_model, _image, _confidence):
            nonlocal detect_call_index
            offset = frame_offsets_px[detect_call_index]
            detect_call_index += 1
            return [
                {
                    "box": [620.0 + offset, 340.0, 660.0 + offset, 380.0],
                    "center": [640.0 + offset, 360.0],
                    "class_id": 0, "confidence": 0.9,
                },
                {
                    "box": [700.0 + offset, 340.0, 740.0 + offset, 380.0],
                    "center": [720.0 + offset, 360.0],
                    "class_id": 0, "confidence": 0.8,
                },
            ]

        def geometric_fit(_image, detection, _intrinsics):
            center = np.asarray(detection["center"], dtype=np.float64)
            return {
                "center_px": center,
                "center_px_distorted": center.copy(),
                "axes_px": np.array([40.0, 40.0]),
                "residual_px": 0.2,
                "coverage_deg": 300.0,
                "roundness": 0.98,
                "fit_method": "hough_circle",
            }

        cfg = module.TwoStageConfig(
            batch_fine_frames=3,
            batch_fine_min_valid=3,
            batch_fine_stable_min_frames=3,
            fine_stable_center_scatter_p95_px=1.0,
            max_fine_center_scatter_p95_px=1.0,
            fine_settle_discard_frames=0,
            batch_fine_joint_min_valid_frames=3,
            batch_fine_joint_stable_translation_mm=0.25,
            batch_fine_joint_stable_yaw_deg=0.15,
        )
        handeye = SimpleNamespace(T_tcp_rgb_camera=np.eye(4, dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, "get_rgb_frame_bundle", return_value=bundle), \
                patch.object(module, "detect", side_effect=detect_frame), \
                patch.object(module, "fit_hole_ellipse", side_effect=geometric_fit), \
                patch.object(module.cv2, "imwrite", return_value=True):
            result = module._batch_fine_localization_at_260mm(
                selected, np.eye(4, dtype=np.float64), handeye,
                object(), object(), 0.5, cfg, self.intrinsics,
                Path(directory), module.TimingRecorder(), [],
            )

        joint_summary = result[1]["batch_fine_joint_summary"]
        self.assertTrue(joint_summary["success"])
        self.assertFalse(joint_summary["joint_frame_stability_gate_applied"])
        diagnostic_transforms = [
            {
                "translation_mm": record["translation_mm"],
                "yaw_rad": np.deg2rad(float(record["yaw_deg"])),
            }
            for record in joint_summary["frame_records"]
            if record["success"]
        ]
        self.assertFalse(module._batch_fine_joint_transform_stable(
            diagnostic_transforms,
            3, 0.25, 0.15,
        ))
        self.assertTrue(result[1]["batch_fine_joint_success"])
        self.assertTrue(result[2]["batch_fine_joint_success"])
        self.assertEqual(joint_summary["fused_frame_count"], 3)

    def test_batch_fine_rejects_geometric_center_far_from_coarse_anchor(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        bundle = SimpleNamespace(
            color_bgr=image, intrinsics=self.intrinsics, host_timestamp_ns=456,
        )
        selected = [{
            "hole_id": 18,
            "initial_center_base_mm": np.array([0.0, 0.0, 1000.0]),
            "coarse_plane_point_base_mm": np.array([0.0, 0.0, 1000.0]),
            "coarse_normal_toward_camera_base": np.array([0.0, 0.0, 1.0]),
            "initial_detection": {
                "class_id": 0, "box": [620.0, 340.0, 660.0, 380.0],
            },
        }]
        detections = [{
            "box": [620.0, 340.0, 660.0, 380.0],
            "center": [640.0, 360.0], "class_id": 0, "confidence": 0.9,
        }]
        wrong_center = np.asarray([649.0, 356.0])
        ellipse = {
            "center_px": wrong_center,
            "center_px_distorted": wrong_center,
            "axes_px": np.asarray([40.0, 40.0]),
            "residual_px": 0.2,
            "coverage_deg": 300.0,
            "roundness": 0.98,
            "fit_method": "hough_circle",
        }
        cfg = module.TwoStageConfig(
            batch_fine_frames=5,
            batch_fine_min_valid=3,
            batch_fine_stable_min_frames=3,
            batch_fine_settle_discard_frames=0,
            batch_fine_max_geometric_anchor_distance_px=5.0,
        )
        handeye = SimpleNamespace(T_tcp_rgb_camera=np.eye(4, dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, "get_rgb_frame_bundle", return_value=bundle), \
                patch.object(module, "detect", return_value=detections), \
                patch.object(module, "fit_hole_ellipse", return_value=ellipse), \
                patch.object(module.cv2, "imwrite", return_value=True):
            result = module._batch_fine_localization_at_260mm(
                selected, np.eye(4, dtype=np.float64), handeye,
                object(), object(), 0.5, cfg, self.intrinsics,
                Path(directory), module.TimingRecorder(), [],
            )

        self.assertFalse(result[18]["success"])
        self.assertTrue(all(
            "geometric_anchor_distance" in str(observation.error)
            for observation in result[18]["observations"]
        ))

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
                "2", current, captured, args, object(), object(),
                target_height_mm=255.0,
            )
        move_fine.assert_called_once()
        self.assertEqual(
            move_fine.call_args.kwargs["descent_guard_mm"],
            module.SHARED_OBSERVATION_MIN_DESCENT_MM,
        )
        self.assertEqual(move_fine.call_args.kwargs["target_height_mm"], 255.0)
        self.assertTrue(np.allclose(result, captured))

    def test_per_hole_fine_motion_uses_measured_tcp_and_local_clearance(self) -> None:
        cached = make_transform(np.eye(3), np.array([700.0, 70.0, 20.0]))
        measured = make_transform(np.eye(3), np.array([706.0, 77.0, 100.0]))
        target = make_transform(self.R_down, np.array([500.0, 40.0, 260.0]))
        args = SimpleNamespace(
            speed_m_s=0.08, acc_m_s2=0.25,
            transit_speed_m_s=0.15, transit_acc_m_s2=0.45,
            approach_speed_m_s=0.12, approach_acc_m_s2=0.35,
        )
        calls = []

        def fake_move(label, actual, planned, *_args, **kwargs):
            calls.append((label, np.asarray(actual).copy(), np.asarray(planned).copy(), kwargs))
            return np.asarray(planned).copy()

        with patch.object(module, "_require_safe_snapshot", return_value=({}, measured)), \
                patch.object(module, "_confirm_and_move_line", side_effect=fake_move):
            result = module._move_to_fine_pose(
                "4", cached, target, args, object(), object(),
                safe_margin_mm=20.0,
                descent_guard_mm=10.0,
            )

        self.assertEqual(len(calls), 4)
        # 缓存TCP与实测TCP相差6/7 mm；第一段必须以实测XY为准做纯Z抬升。
        self.assertTrue(np.allclose(calls[0][1], measured))
        self.assertTrue(np.allclose(calls[0][2][:2, 3], measured[:2, 3]))
        self.assertAlmostEqual(calls[0][2][2, 3], 280.0)
        # 20 mm只替代原来的60 mm额外净空，最后10 mm仍保持纯Z保护段。
        self.assertTrue(np.allclose(calls[1][2][:2, 3], target[:2, 3]))
        self.assertAlmostEqual(calls[1][2][2, 3], 280.0)
        self.assertAlmostEqual(calls[2][2][2, 3], 270.0)
        self.assertTrue(np.allclose(calls[3][2], target))
        self.assertTrue(np.allclose(result, target))

    def test_batch_fine_final_motion_direct_path_keeps_guarded_pure_z_descent(self) -> None:
        current = make_transform(np.eye(3), np.array([300.0, 0.0, 120.0]))
        target = make_transform(self.R_down, np.array([100.0, 40.0, 40.0]))
        args = SimpleNamespace(
            speed_m_s=0.08, acc_m_s2=0.25,
            transit_speed_m_s=0.15, transit_acc_m_s2=0.45,
            approach_speed_m_s=0.12, approach_acc_m_s2=0.35,
        )
        calls = []

        def fake_move(label, actual, planned, *_args, **kwargs):
            calls.append((label, np.asarray(actual).copy(), np.asarray(planned).copy(), kwargs))
            return np.asarray(planned).copy()

        with patch.object(module, "_confirm_and_move_line", side_effect=fake_move), patch.object(
            module, "_require_safe_snapshot", return_value=({}, current),
        ):
            result = module._move_to_batch_final_tcp_direct(
                "2", current, target, args, object(), object(),
            )

        # 当前TCP已经处于目标上方80 mm，满足60 mm安全余量；不再重复
        # 额外抬升一个完整安全余量，直接在当前安全高度横移。
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            [call[3]["motion_profile"] for call in calls],
            ["transit", "approach", "precision"],
        )
        # 60 mm safe margin, followed by a final 10 mm pure-Z descent guard.
        self.assertTrue(np.allclose(calls[0][2][:2, 3], target[:2, 3]))
        self.assertAlmostEqual(calls[0][2][2, 3], 120.0)
        self.assertTrue(np.allclose(calls[1][2][:2, 3], target[:2, 3]))
        self.assertAlmostEqual(calls[1][2][2, 3], 50.0)
        self.assertTrue(np.allclose(calls[2][2], target))
        self.assertTrue(np.allclose(result, target))

    def test_final_direct_path_from_below_lifts_before_xy(self) -> None:
        current = make_transform(np.eye(3), np.array([300.0, 0.0, 10.0]))
        stale_current = make_transform(np.eye(3), np.array([999.0, 999.0, 80.0]))
        target = make_transform(self.R_down, np.array([100.0, 40.0, 40.0]))
        args = SimpleNamespace(speed_m_s=0.08, acc_m_s2=0.25)
        calls = []

        def fake_move(label, actual, planned, *_args, **kwargs):
            calls.append((np.asarray(actual).copy(), np.asarray(planned).copy()))
            return np.asarray(planned).copy()

        with patch.object(module, "_confirm_and_move_line", side_effect=fake_move), patch.object(
            module, "_require_safe_snapshot", return_value=({}, current),
        ):
            result = module._move_to_batch_final_tcp_direct(
                "2", stale_current, target, args, object(), object(),
            )

        self.assertEqual(len(calls), 4)
        # 先沿原XY纯Z升至最终点上方60 mm，再允许横移和姿态变化。
        np.testing.assert_allclose(calls[0][0], current)
        np.testing.assert_allclose(calls[0][1][:2, 3], current[:2, 3])
        self.assertAlmostEqual(calls[0][1][2, 3], 100.0)
        np.testing.assert_allclose(calls[1][1][:2, 3], target[:2, 3])
        self.assertAlmostEqual(calls[1][1][2, 3], 100.0)
        self.assertAlmostEqual(calls[2][1][2, 3], 50.0)
        np.testing.assert_allclose(calls[3][1], target)
        np.testing.assert_allclose(result, target)

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

    def test_direct_capture_only_diagnostics_do_not_treat_320mm_as_fine_height_error(self) -> None:
        diagnostics = module._build_comparison_hole_diagnostics({
            "status": "capture_only",
            "fine_quality_status": "coarse_direct_capture_only",
            "coarse_direct_decision": "capture_only",
            "coarse_source": "batch_coarse_pointcloud",
            "coarse_capture_height_mm": 320.0,
            "estimated_height_mm": 320.0,
            "batch_coarse_requested": True,
            "coarse_cache_event": {},
            "timing": {"events": []},
        })

        self.assertTrue(diagnostics["coarse_path"]["batch_used"])
        self.assertTrue(diagnostics["motion_quality"]["coarse_direct_final"])
        self.assertTrue(diagnostics["motion_quality"]["capture_only"])
        self.assertIsNone(diagnostics["motion_quality"]["fine_height_error_mm"])
        self.assertEqual(diagnostics["coarse_quality"]["capture_height_mm"], 320.0)

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
        self.assertEqual(
            move_fine.call_args.kwargs["safe_margin_mm"],
            cfg.per_hole_fine_safe_z_margin_mm,
        )

    def test_compact_multi_hole_shared_captures_keep_one_group(self) -> None:
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
                "center_px": hole["initial_center_px"],
                "center_camera_mm": [hole["initial_center_base_mm"][0], 0.0, 340.0],
                "center_base_mm": hole["initial_center_base_mm"],
                "plane_point_camera_mm": [hole["initial_center_base_mm"][0], 0.0, 340.0],
                "plane_point_base_mm": hole["initial_center_base_mm"],
                "normal_camera": [0.0, 0.0, -1.0],
                "normal_base": [0.0, 0.0, 1.0],
                "plane_rmse_mm": 1.0,
                "valid_frames": 1,
                "total_frames": 1,
                "center_scatter_p95_px": 0.0,
                "coarse_captures": [],
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
            target = np.eye(4)
            if len(planning_calls) == 3:
                target[0, 3] = 6.0
            return target, {
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
            move_final_xy=False, tcp_xy_offset_mm=None,
            handeye="handeye.json", optimize_hole_order=False,
        )
        cfg = module.TwoStageConfig(
            batch_coarse_frames=1, batch_coarse_min_valid=1,
            batch_fine_localization=True, batch_fine_frames=1,
            batch_fine_min_valid=1, batch_fine_stable_min_frames=1,
            batch_fine_per_hole_fallback=False,
        )
        report = {"stages": {}, "camera": {}, "coarse_cache": {}}
        runtime = {"rgbd_pipeline": object(), "align": object(), "chain": object()}
        handeye = SimpleNamespace(T_tcp_rgb_camera=np.eye(4), validated_for_motion=True)

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, "_split_shared_cache_validation_groups") as split_groups, \
                patch.object(module, "_split_batch_localization_groups", return_value=([selected], [])), \
                patch.object(module, "_plan_batch_coarse_group_pose", side_effect=plan_all), \
                patch.object(module, "_confirm_and_move_line", side_effect=lambda _l, _c, target, *_a, **_k: target), \
                patch.object(module, "_move_to_shared_fine_pose", side_effect=lambda _current, target, *_a, **_k: target) as shared_move, \
                patch.object(module, "_move_shared_fine_group_adjustment", side_effect=lambda _label, _current, target, *_a, **_k: target) as in_group_move, \
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
        shared_move.assert_called_once()
        in_group_move.assert_called_once()
        self.assertEqual(fine_capture.call_count, 2)
        self.assertEqual(
            [call[0] for call in planning_calls],
            [[1, 2, 3]] * 3,
        )
        self.assertEqual(
            [call[1] for call in planning_calls],
            [340.0, 260.0, 260.0],
        )
        self.assertEqual(planning_calls[1][2], [[-40.0, 0.0, 0.0], [0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
        self.assertTrue(report["stages"]["batch_coarse_plan"]["all_selected_holes_single_group"])
        self.assertTrue(report["stages"]["batch_fine_plan"]["all_selected_holes_single_group"])

    def test_shared_coarse_pose_refinement_fuses_the_whole_group_before_accepting(self) -> None:
        group = [
            {
                "hole_id": 1,
                "initial_center_base_mm": np.array([0.0, 0.0, 340.0]),
                "initial_center_px": np.array([640.0, 360.0]),
                "initial_plane_normal_base": np.array([0.0, 0.0, -1.0]),
            },
            {
                "hole_id": 2,
                "initial_center_base_mm": np.array([20.0, 0.0, 340.0]),
                "initial_center_px": np.array([660.0, 360.0]),
                "initial_plane_normal_base": np.array([0.0, 0.0, -1.0]),
            },
        ]
        def plane_info_for_center(center, *_args, **_kwargs):
            info = {
                "plane_point_camera_mm": [float(center[0]) - 640.0, 0.0, 340.0],
                "plane_normal_camera": [0.0, 0.0, -1.0],
                "plane_rmse_mm": 1.0,
                "ring_points": 100,
                "surface_model": "test",
            }
            return np.zeros(3), info
        bundle = SimpleNamespace(
            color_bgr=np.full((720, 1280, 3), 220, dtype=np.uint8),
            xyz_map_mm=np.zeros((720, 1280, 3), dtype=np.float32),
            intrinsics=self.intrinsics,
            host_timestamp_ns=1,
        )
        detections = [
            {"box": [630.0, 350.0, 650.0, 370.0], "center": [640.0, 360.0]},
            {"box": [650.0, 350.0, 670.0, 370.0], "center": [660.0, 360.0]},
        ]
        cfg = module.TwoStageConfig(
            batch_coarse_localization=True,
            batch_coarse_pose_refine_frames=5,
            batch_coarse_pose_refine_min_valid_frames=3,
            batch_coarse_pose_refine_max_iterations=2,
            coarse_settle_delay_s=0.0,
        )
        current_tcp = np.eye(4, dtype=np.float64)
        handeye = SimpleNamespace(T_tcp_rgb_camera=np.eye(4, dtype=np.float64))
        geometry = {
            "target_tcp_pose_m_rad": [0.0] * 6,
            "target_tcp_transform_mm": current_tcp.copy(),
            "group_point_base_mm": np.array([10.0, 0.0, 340.0]),
            "group_normal_toward_camera_base": np.array([0.0, 0.0, -1.0]),
            "projected_holes_px": {1: np.array([640.0, 360.0]), 2: np.array([660.0, 360.0])},
            "group_bbox_px": [640.0, 360.0, 660.0, 360.0],
            "group_center_px": np.array([640.0, 360.0]),
        }

        def project(point, *_args):
            return np.array([640.0 + float(np.asarray(point).reshape(3)[0]), 360.0])

        def assign(_detections, anchors, _tolerance):
            return {
                hole_id: {"detection": detections[index], "distance_px": 0.0}
                for index, hole_id in enumerate(sorted(anchors))
            }

        pose_session = SimpleNamespace(read_pose_snapshot=lambda: {})
        frames = iter([bundle] * 10)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(module, "get_aligned_frame_bundle", side_effect=lambda *_a: next(frames)), \
                    patch.object(module, "detect", return_value=detections), \
                    patch.object(module, "_assign_detections_to_projection", side_effect=assign), \
                    patch.object(module, "hole_camera_point", side_effect=plane_info_for_center), \
                    patch.object(module, "_project_base_point_to_pixel", side_effect=project), \
                    patch.object(module, "_plan_batch_coarse_group_pose", return_value=(current_tcp.copy(), geometry)), \
                    patch.object(module, "_wait_robot_steady_before_initial_capture", return_value=current_tcp.copy()), \
                    patch.object(module, "camera_height_to_plane_mm", return_value=340.0), \
                    patch.object(module, "_move_to_shared_coarse_pose") as move_shared:
                result = module._refine_shared_coarse_group_pose(
                    1, 1, group, current_tcp, current_tcp.copy(), geometry,
                    handeye, object(), object(), object(), object(), 0.5, cfg,
                    self.intrinsics, Path(directory), module.TimingRecorder(),
                    [], SimpleNamespace(), object(), pose_session, 0.0,
                )

        self.assertTrue(result["report"]["accepted"])
        self.assertEqual(result["report"]["iterations"][0]["success_hole_count"], 2)
        self.assertIn("coarse_pose_refined_center_base_mm", group[0])
        self.assertIn("coarse_pose_refined_center_base_mm", group[1])
        move_shared.assert_not_called()

    def test_diameter_categories_cover_all_supported_holes(self) -> None:
        for diameter in (64.6, 70.2, 74.7):
            self.assertEqual(min(module.HOLE_DIAMETERS_MM, key=lambda value: abs(value - diameter)), round(diameter / 5.0) * 5.0)

    def test_motion_defaults_use_current_calibrations_but_remain_preview(self) -> None:
        parser = module.build_parser()
        defaults = parser.parse_args([])
        self.assertFalse(defaults.execute)
        self.assertTrue(defaults.allow_experimental_handeye)
        self.assertTrue(defaults.move_final_xy)
        self.assertFalse(defaults.coarse_direct_final)
        self.assertTrue(
            parser.parse_args(["--coarse-direct-final"]).coarse_direct_final
        )
        direct_defaults = parser.parse_args([])
        self.assertFalse(direct_defaults.coarse_direct_final)
        self.assertAlmostEqual(direct_defaults.coarse_direct_final_height_mm, 340.0)
        self.assertEqual(direct_defaults.coarse_direct_final_max_group_size, 5)
        self.assertEqual(direct_defaults.coarse_direct_final_early_stop_extra_frames, 5)
        self.assertAlmostEqual(direct_defaults.coarse_direct_final_settle_delay_s, 1.0)
        self.assertAlmostEqual(direct_defaults.coarse_direct_final_group_planning_timeout_s, 10.0)
        self.assertAlmostEqual(direct_defaults.coarse_direct_final_capture_timeout_s, 60.0)
        self.assertAlmostEqual(direct_defaults.coarse_direct_final_steady_timeout_s, 45.0)
        self.assertEqual(direct_defaults.coarse_direct_final_max_consecutive_frame_failures, 3)
        self.assertFalse(direct_defaults.coarse_direct_final_capture_only)
        explicit_direct = parser.parse_args([
            "--coarse-direct-final",
            "--coarse-direct-final-height-mm", "340",
            "--coarse-direct-final-max-group-size", "1",
            "--coarse-direct-final-early-stop-extra-frames", "0",
            "--coarse-direct-final-settle-delay-s", "2.5",
            "--coarse-direct-final-capture-only",
        ])
        self.assertTrue(explicit_direct.coarse_direct_final)
        self.assertEqual(
            module.TwoStageConfig.from_namespace(explicit_direct).coarse_direct_final_height_mm,
            340.0,
        )
        self.assertEqual(
            module.TwoStageConfig.from_namespace(explicit_direct).coarse_direct_final_max_group_size,
            1,
        )
        self.assertAlmostEqual(
            module.TwoStageConfig.from_namespace(explicit_direct).coarse_direct_final_settle_delay_s,
            2.5,
        )
        self.assertTrue(
            module.TwoStageConfig.from_namespace(explicit_direct).coarse_direct_final_capture_only
        )
        with self.assertRaisesRegex(ValueError, "每组最多孔数"):
            module.TwoStageConfig(
                coarse_direct_final_max_group_size=6,
            ).validate()
        self.assertFalse(hasattr(direct_defaults, "coarse_direct_min_valid_frames"))
        self.assertFalse(hasattr(direct_defaults, "coarse_direct_center_recheck"))
        self.assertFalse(parser.parse_args(["--require-validated-handeye"]).allow_experimental_handeye)
        self.assertTrue(parser.parse_args(["--execute"]).execute)
        self.assertTrue(parser.parse_args(["--allow-experimental-handeye"]).allow_experimental_handeye)
        self.assertTrue(parser.parse_args(["--move-final-xy"]).move_final_xy)
        self.assertFalse(hasattr(parser.parse_args([]), "final_target_mode"))
        self.assertNotIn("--final-target-mode", parser._option_string_actions)
        self.assertIsNone(parser.parse_args([]).tcp_xy_offset_mm)
        self.assertEqual(tuple(parser.parse_args(["--tcp-xy-offset-mm", "0", "0"]).tcp_xy_offset_mm), (0.0, 0.0))
        self.assertAlmostEqual(parser.parse_args([]).speed_m_s, 0.08)
        self.assertAlmostEqual(parser.parse_args([]).acc_m_s2, 0.25)
        self.assertAlmostEqual(parser.parse_args([]).transit_speed_m_s, 0.15)
        self.assertAlmostEqual(parser.parse_args([]).transit_acc_m_s2, 0.45)
        self.assertAlmostEqual(parser.parse_args([]).approach_speed_m_s, 0.12)
        self.assertAlmostEqual(parser.parse_args([]).approach_acc_m_s2, 0.35)
        self.assertAlmostEqual(defaults.per_hole_fine_safe_z_margin_mm, 20.0)
        self.assertEqual(parser.parse_args([]).coarse_recapture_settle_discard_frames, 10)
        self.assertEqual(parser.parse_args([]).coarse_max_corrections, 2)
        self.assertEqual(parser.parse_args([]).fine_settle_discard_frames, 10)
        self.assertEqual(parser.parse_args([]).batch_fine_settle_discard_frames, 10)
        self.assertEqual(parser.parse_args([]).fine_retries, 2)
        cfg = module.TwoStageConfig.from_namespace(defaults)
        for name in module.TwoStageConfig.__dataclass_fields__:
            if hasattr(defaults, name):
                self.assertEqual(
                    getattr(defaults, name),
                    getattr(module.TwoStageConfig(), name),
                    name,
                )
        self.assertEqual(
            cfg.coarse_recapture_settle_discard_frames,
            module.TwoStageConfig().coarse_recapture_settle_discard_frames,
        )
        self.assertEqual(cfg.fine_retry_count, defaults.fine_retries)
        cfg.validate(reuse_coarse_cache=defaults.reuse_coarse_cache)

    def test_map_build_parser_defaults_to_manual_selection_and_has_motion_gates(self) -> None:
        defaults = module.build_parser().parse_args([])
        self.assertEqual(defaults.map_hole_selection_mode, "manual")
        self.assertEqual(defaults.map_build_localization_mode, "coarse_only")
        per_hole = module.build_parser().parse_args([
            "--map-build-localization-mode", "per_hole",
        ])
        self.assertEqual(per_hole.map_build_localization_mode, "per_hole")
        self.assertAlmostEqual(defaults.batch_coarse_pose_refine_max_correction_mm, 5.0)
        self.assertAlmostEqual(defaults.batch_coarse_pose_refine_max_correction_rotation_deg, 2.0)
        self.assertTrue(defaults.batch_coarse_pose_refine_direct_motion)
        self.assertFalse(module.build_parser().parse_args([
            "--no-batch-coarse-pose-refine-direct-motion",
        ]).batch_coarse_pose_refine_direct_motion)
        self.assertAlmostEqual(defaults.batch_coarse_pose_refine_max_total_correction_mm, 25.0)
        self.assertAlmostEqual(defaults.batch_coarse_pose_refine_max_total_correction_rotation_deg, 7.0)
        self.assertAlmostEqual(defaults.map_build_safe_z_margin_mm, 100.0)

    def test_large_live_pose_correction_is_rejected_before_motion(self) -> None:
        cfg = module.TwoStageConfig(
            batch_coarse_pose_refine_max_correction_mm=10.0,
            batch_coarse_pose_refine_max_correction_rotation_deg=2.0,
        )
        self.assertEqual(
            group_pose_workflow._pose_correction_motion_gate(9.9, 1.9, cfg),
            (True, None),
        )
        self.assertEqual(
            group_pose_workflow._pose_correction_motion_gate(10.1, 0.1, cfg),
            (False, "translation_exceeds_motion_gate"),
        )
        self.assertEqual(
            group_pose_workflow._pose_correction_motion_gate(1.0, 2.1, cfg),
            (False, "rotation_exceeds_motion_gate"),
        )

    def test_large_shared_pose_correction_is_split_into_a_safe_step(self) -> None:
        group_pose_workflow.install_runtime(module.__dict__)
        cfg = module.TwoStageConfig()
        start = np.eye(4, dtype=np.float64)
        target = np.eye(4, dtype=np.float64)
        target[:3, 3] = [18.0, 0.0, 0.0]
        angle = np.deg2rad(5.0)
        target[:3, :3] = np.asarray([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])

        step, fraction, step_translation, step_rotation = group_pose_workflow._bounded_pose_step(
            start, target, cfg,
        )

        self.assertLess(fraction, 1.0)
        self.assertLessEqual(step_translation, cfg.batch_coarse_pose_refine_max_correction_mm + 1e-9)
        self.assertLessEqual(
            step_rotation,
            cfg.batch_coarse_pose_refine_max_correction_rotation_deg + 1e-9,
        )
        self.assertEqual(
            group_pose_workflow._pose_correction_motion_gate(
                step_translation, step_rotation, cfg,
            ),
            (True, None),
        )
        self.assertGreater(float(np.linalg.norm(step[:3, 3] - start[:3, 3])), 0.0)

    def test_same_group_bounded_pose_correction_moves_directly_in_place(self) -> None:
        group_pose_workflow.install_runtime(module.__dict__)
        cfg = module.TwoStageConfig()
        start = np.eye(4, dtype=np.float64)
        start[:3, 3] = [100.0, 200.0, 340.0]
        target = start.copy()
        target[:3, 3] += [3.0, -2.0, 1.0]
        angle = np.deg2rad(0.5)
        target[:3, :3] = np.asarray([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])

        def fake_direct_move(_label, _current, destination, *_args, **kwargs):
            self.assertEqual(kwargs["motion_profile"], "approach")
            return np.asarray(destination, dtype=np.float64).copy()

        with patch.object(
            group_pose_workflow, "_confirm_and_move_line", side_effect=fake_direct_move,
        ) as direct_move, patch.object(
            group_pose_workflow, "_move_to_shared_coarse_pose",
        ) as safe_move:
            moved, policy = group_pose_workflow._move_shared_coarse_pose_correction(
                2, 9, start, target, SimpleNamespace(), object(), object(), cfg,
            )

        direct_move.assert_called_once()
        safe_move.assert_not_called()
        self.assertEqual(policy, "direct_same_group_bounded_move_line")
        self.assertTrue(np.allclose(moved, target))

    def test_same_group_pose_correction_falls_back_when_step_is_over_limit(self) -> None:
        group_pose_workflow.install_runtime(module.__dict__)
        cfg = module.TwoStageConfig()
        start = np.eye(4, dtype=np.float64)
        target = start.copy()
        target[0, 3] = cfg.batch_coarse_pose_refine_max_correction_mm + 0.1

        with patch.object(
            group_pose_workflow, "_confirm_and_move_line",
        ) as direct_move, patch.object(
            group_pose_workflow, "_move_to_shared_coarse_pose", return_value=target,
        ) as safe_move:
            moved, policy = group_pose_workflow._move_shared_coarse_pose_correction(
                1, 1, start, target, SimpleNamespace(), object(), object(), cfg,
                target_height_mm=340.0,
            )

        direct_move.assert_not_called()
        safe_move.assert_called_once()
        self.assertEqual(
            policy,
            "safe_lift_horizontal_descent_translation_exceeds_motion_gate",
        )
        self.assertTrue(np.allclose(moved, target))

    def test_stable_340_capture_can_be_used_without_repositioning_for_map(self) -> None:
        group_pose_workflow.install_runtime(module.__dict__)
        cfg = module.TwoStageConfig()
        metrics = {
            "pose_height_error_mm": -1.3,
            "current_group_center_error_px": 3.3,
            "target_group_center_error_px": 0.22,
            "reprojection_error_p95_px": 0.09,
        }
        self.assertTrue(
            group_pose_workflow._measured_340_pose_is_map_safe(
                metrics, observed_hole_count=4, expected_hole_count=4, cfg=cfg,
            )
        )
        metrics["reprojection_error_p95_px"] = cfg.batch_coarse_pose_refine_max_reprojection_error_px + 0.1
        self.assertFalse(
            group_pose_workflow._measured_340_pose_is_map_safe(
                metrics, observed_hole_count=4, expected_hole_count=4, cfg=cfg,
            )
        )

    def test_two_stage_config_validation_is_centralized(self) -> None:
        parser = module.build_parser()
        args = parser.parse_args([
            "--batch-fine-frames", "4",
            "--batch-fine-min-valid", "5",
        ])
        cfg = module.TwoStageConfig.from_namespace(args)
        with self.assertRaisesRegex(ValueError, "批量精定位最少有效帧数"):
            cfg.validate(reuse_coarse_cache=args.reuse_coarse_cache)

    def test_hole_map_build_rejects_coarse_cache_manifest(self) -> None:
        with self.assertRaisesRegex(ValueError, "只能自动保存到 hole_localization_maps"):
            module._resolve_hole_map_path(
                module.HOLE_LOCALIZATION_COARSE_CACHE_DIR / "manifest.json",
                for_build=True,
            )


if __name__ == "__main__":
    unittest.main()
