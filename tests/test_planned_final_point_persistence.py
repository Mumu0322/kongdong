from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from aubo_workbench import sequential_hole_execution as execution
from aubo_workbench.geometry import transform_to_sdk_pose_m_rad
from aubo_workbench.hole_localization_planning import (
    plan_final_tcp_base_z, plan_final_tcp_xy,
)


class PlannedFinalPointPersistenceTests(unittest.TestCase):
    def test_precaptured_final_target_uses_its_own_orientation(self) -> None:
        current = np.eye(4)
        current[:3, 3] = [0.0, 0.0, 100.0]
        current_before = current.copy()
        fine_capture = np.eye(4)
        angle = np.deg2rad(15.0)
        fine_capture[:3, :3] = [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
        fine_capture[:3, 3] = [10.0, 20.0, 260.0]
        point = np.array([50.0, 60.0, 40.0])
        with patch.object(execution, "np", np, create=True), patch.object(
            execution, "plan_final_tcp_xy", plan_final_tcp_xy, create=True,
        ), patch.object(
            execution, "plan_final_tcp_base_z", plan_final_tcp_base_z, create=True,
        ):
            reference, xy_target, final = execution._plan_per_hole_final_target(
                current, fine_capture, point, None,
                use_charuco_model=False, precaptured=True,
            )
        np.testing.assert_allclose(reference[:3, :3], fine_capture[:3, :3])
        np.testing.assert_allclose(xy_target[:3, :3], fine_capture[:3, :3])
        np.testing.assert_allclose(final[:3, :3], fine_capture[:3, :3])
        np.testing.assert_allclose(final[:3, 3], point)
        np.testing.assert_allclose(current, current_before)

    def test_precaptured_pose_forces_safe_path_even_above_final(self) -> None:
        current = np.eye(4)
        current[2, 3] = 100.0
        final = current.copy()
        final[2, 3] = 40.0
        final[0, 3] = 50.0
        timing = SimpleNamespace(measure=lambda *a, **k: nullcontext())
        with patch.object(execution, "np", np, create=True), patch.object(
            execution, "_move_to_batch_final_tcp_direct", return_value=final,
            create=True,
        ) as safe, patch.object(
            execution, "_confirm_and_move_line", side_effect=AssertionError("unsafe XY first"),
            create=True,
        ):
            reached, after_xy, policy = execution._execute_per_hole_final_motion(
                7, 7, current, final, final, "model", object(),
                object(), object(), timing, force_safe_path=True,
            )
        safe.assert_called_once()
        self.assertIsNone(after_xy)
        self.assertEqual(policy, "safe_z_lift_xy_guarded_z_descent")
        np.testing.assert_allclose(reached, final)

    def test_current_below_final_uses_safe_z_xy_z_route(self) -> None:
        current = np.eye(4)
        current[:3, 3] = [0.0, 0.0, 10.0]
        xy_target = current.copy()
        xy_target[:2, 3] = [50.0, 60.0]
        final = xy_target.copy()
        final[2, 3] = 40.0
        calls = []
        timing = SimpleNamespace(measure=lambda *a, **k: nullcontext())

        def safe_route(*args, **kwargs):
            calls.append("safe_z_xy_z")
            return final.copy()

        with patch.object(execution, "np", np, create=True), patch.object(
            execution, "_move_to_batch_final_tcp_direct", side_effect=safe_route,
            create=True,
        ), patch.object(
            execution, "_confirm_and_move_line", side_effect=AssertionError("unsafe XY first"),
            create=True,
        ):
            reached, after_xy, policy = execution._execute_per_hole_final_motion(
                1, 1, current, xy_target, final, "model", object(),
                object(), object(), timing,
            )
        self.assertEqual(calls, ["safe_z_xy_z"])
        self.assertIsNone(after_xy)
        self.assertEqual(policy, "safe_z_lift_xy_guarded_z_descent")
        np.testing.assert_allclose(reached, final)

    def test_current_above_final_keeps_xy_then_z(self) -> None:
        current = np.eye(4)
        current[:3, 3] = [0.0, 0.0, 100.0]
        xy_target = current.copy()
        xy_target[:2, 3] = [50.0, 60.0]
        final = xy_target.copy()
        final[2, 3] = 40.0
        calls = []
        timing = SimpleNamespace(measure=lambda *a, **k: nullcontext())

        def move(label, actual, target, *args, **kwargs):
            calls.append((np.asarray(actual).copy(), np.asarray(target).copy()))
            return np.asarray(target).copy()

        with patch.object(execution, "np", np, create=True), patch.object(
            execution, "_move_to_batch_final_tcp_direct", side_effect=AssertionError("unneeded lift"),
            create=True,
        ), patch.object(
            execution, "_confirm_and_move_line", side_effect=move, create=True,
        ), patch.object(
            execution, "plan_final_tcp_base_z", plan_final_tcp_base_z, create=True,
        ):
            reached, after_xy, policy = execution._execute_per_hole_final_motion(
                1, 1, current, xy_target, final, "model", object(),
                object(), object(), timing,
            )
        self.assertEqual(len(calls), 2)
        np.testing.assert_allclose(calls[0][1], xy_target)
        np.testing.assert_allclose(calls[1][1], final)
        self.assertEqual(policy, "xy_then_z")
        np.testing.assert_allclose(after_xy, xy_target)
        np.testing.assert_allclose(reached, final)

    def test_plan_is_on_disk_before_motion_and_survives_failure_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            report = {"stages": {"active_hole": {"hole_id": 1, "status": "processing"}}}
            ctx = SimpleNamespace(
                run_dir=run_dir,
                report=report,
                rows=[],
                timing=SimpleNamespace(scoped_snapshot=lambda _: {"events": []}),
                results=[],
                batch_fine_plan={},
                cfg=SimpleNamespace(batch_fine_localization=False),
                batch_coarse_for_cache=False,
            )
            hole = {"hole_id": 1}
            tcp = np.eye(4)
            tcp[:3, 3] = [10.5, 20.5, 30.5]

            def write_report(path, value, rows, *, timing):
                (path / "report.json").write_text(json.dumps(value), encoding="utf-8")

            deferred = []

            def append_deferred(hole, result, results, report, run_dir, timing):
                deferred.append(result)
                report["stages"]["hole_1"] = result

            with patch.object(execution, "np", np, create=True), patch.object(
                execution, "transform_to_sdk_pose_m_rad", transform_to_sdk_pose_m_rad,
                create=True,
            ), patch.object(execution, "_write_report", write_report, create=True), patch.object(
                execution, "_append_deferred_hole_result", append_deferred, create=True,
            ):
                execution._persist_planned_final_point(
                    ctx, hole, np.array([1.0, 2.0, 3.0]),
                    np.array([1.0, 2.0, 4.0]), tcp,
                    motion_path="per_hole_fine_xy_then_z",
                )
                on_disk = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(
                    on_disk["stages"]["hole_1"]["planned_final_point"]
                    ["planned_final_tcp_xyz_mm"],
                    [10.5, 20.5, 30.5],
                )
                execution._record_unexpected_hole_failure(
                    ctx, 1, hole, RuntimeError("motion failed"),
                )

            self.assertEqual(deferred[0]["status"], "deferred_unexpected_error")
            self.assertEqual(deferred[0]["target_point_base_mm"], [1.0, 2.0, 4.0])
            self.assertEqual(
                deferred[0]["planned_final_point"]["planned_final_tcp_xyz_mm"],
                [10.5, 20.5, 30.5],
            )
