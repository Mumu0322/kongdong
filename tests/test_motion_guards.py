#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""aubo_workbench.motion_guards 的纯离线测试；不连接相机或机械臂。

这批逻辑原来分散在两个入口脚本里，只有间接覆盖。合并到一处后在这里直接
锁定行为，避免以后再次分叉。
"""

from __future__ import annotations

import math
import unittest
from typing import Any

from aubo_workbench.motion_guards import (
    angular_delta_rad,
    pose_error,
    validate_robot_ready,
    wait_for_target,
)


def _ready_snapshot(**overrides: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "power_on": True,
        "collision": False,
        "steady": True,
        "tcp_pose_m_rad": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    snapshot.update(overrides)
    return snapshot


class FakeSession:
    """按预设序列返回状态；记录 stop_motion 是否被调用。"""

    def __init__(self, snapshots: list[dict[str, Any]]) -> None:
        self._snapshots = list(snapshots)
        self.stop_calls = 0

    def snapshot(self) -> dict[str, Any]:
        if len(self._snapshots) > 1:
            return self._snapshots.pop(0)
        return self._snapshots[0]

    def stop_motion(self) -> str:
        self.stop_calls += 1
        return "stopped"


class AngularDeltaTests(unittest.TestCase):
    def test_wraps_across_pi_boundary(self) -> None:
        # 从 -pi+0.01 到 pi-0.01 实际只差 0.02 rad，不应算成接近 2pi。
        delta = angular_delta_rad(math.pi - 0.01, -math.pi + 0.01)
        self.assertAlmostEqual(abs(delta), 0.02, places=9)

    def test_zero_for_identical_angles(self) -> None:
        self.assertAlmostEqual(angular_delta_rad(1.234, 1.234), 0.0, places=12)


class PoseErrorTests(unittest.TestCase):
    def test_translation_converted_to_mm(self) -> None:
        target = [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]
        current = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        xyz_mm, rotation_rad = pose_error(target, current)
        self.assertAlmostEqual(xyz_mm, 1.0, places=9)
        self.assertAlmostEqual(rotation_rad, 0.0, places=12)

    def test_rotation_uses_wrapped_delta(self) -> None:
        target = [0.0, 0.0, 0.0, 0.0, 0.0, math.pi - 0.01]
        current = [0.0, 0.0, 0.0, 0.0, 0.0, -math.pi + 0.01]
        _, rotation_rad = pose_error(target, current)
        self.assertAlmostEqual(rotation_rad, 0.02, places=9)


class ValidateRobotReadyTests(unittest.TestCase):
    def test_accepts_ready_snapshot(self) -> None:
        validate_robot_ready(_ready_snapshot())

    def test_rejects_power_off(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "未上电"):
            validate_robot_ready(_ready_snapshot(power_on=False))

    def test_rejects_collision_flag(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "碰撞"):
            validate_robot_ready(_ready_snapshot(collision=True))

    def test_rejects_unsteady(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "未稳定"):
            validate_robot_ready(_ready_snapshot(steady=False))

    def test_rejects_short_pose(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "TCP位姿不可用"):
            validate_robot_ready(_ready_snapshot(tcp_pose_m_rad=[0.0, 0.0, 0.0]))

    def test_rejects_non_finite_pose(self) -> None:
        pose = [0.0, 0.0, float("nan"), 0.0, 0.0, 0.0]
        with self.assertRaisesRegex(RuntimeError, "TCP位姿不可用"):
            validate_robot_ready(_ready_snapshot(tcp_pose_m_rad=pose))

    def test_rejects_missing_pose(self) -> None:
        snapshot = _ready_snapshot()
        del snapshot["tcp_pose_m_rad"]
        with self.assertRaisesRegex(RuntimeError, "TCP位姿不可用"):
            validate_robot_ready(snapshot)


class WaitForTargetTests(unittest.TestCase):
    def test_returns_errors_once_within_tolerance(self) -> None:
        session = FakeSession([_ready_snapshot()])
        arrival = wait_for_target(
            session,
            target_m_rad=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            timeout_s=1.0,
            position_tolerance_mm=0.5,
            rotation_tolerance_rad=0.01,
            poll_interval_s=0.01,
        )
        self.assertAlmostEqual(arrival["position_error_mm"], 0.0, places=9)
        self.assertAlmostEqual(arrival["rotation_error_rad"], 0.0, places=12)
        self.assertIs(arrival["snapshot"], session.snapshot())
        self.assertEqual(session.stop_calls, 0)

    def test_waits_until_steady_before_returning(self) -> None:
        # 位置已到但尚未稳定时不得判定到位。
        session = FakeSession([
            _ready_snapshot(steady=False),
            _ready_snapshot(steady=True),
        ])
        arrival = wait_for_target(
            session,
            target_m_rad=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            timeout_s=1.0,
            position_tolerance_mm=0.5,
            rotation_tolerance_rad=0.01,
            poll_interval_s=0.01,
        )
        self.assertTrue(arrival["snapshot"]["steady"])

    def test_collision_stops_motion_and_raises(self) -> None:
        session = FakeSession([_ready_snapshot(collision=True)])
        with self.assertRaisesRegex(RuntimeError, "碰撞"):
            wait_for_target(
                session,
                target_m_rad=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                timeout_s=1.0,
                position_tolerance_mm=0.5,
                rotation_tolerance_rad=0.01,
                poll_interval_s=0.01,
            )
        self.assertEqual(session.stop_calls, 1)

    def test_collision_raises_even_if_stop_fails(self) -> None:
        class FailingStopSession(FakeSession):
            def stop_motion(self) -> str:
                self.stop_calls += 1
                raise OSError("控制器无响应")

        session = FailingStopSession([_ready_snapshot(collision=True)])
        # 停止指令自身失败也必须抛出，不能静默继续运动。
        with self.assertRaises(Exception) as caught:
            wait_for_target(
                session,
                target_m_rad=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                timeout_s=1.0,
                position_tolerance_mm=0.5,
                rotation_tolerance_rad=0.01,
                poll_interval_s=0.01,
            )
        self.assertNotIsInstance(caught.exception, TimeoutError)
        self.assertEqual(session.stop_calls, 1)

    def test_timeout_reports_last_measured_error(self) -> None:
        far = _ready_snapshot(tcp_pose_m_rad=[0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
        session = FakeSession([far])
        with self.assertRaisesRegex(TimeoutError, r"位置误差=50\.000 mm"):
            wait_for_target(
                session,
                target_m_rad=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                timeout_s=0.05,
                position_tolerance_mm=0.5,
                rotation_tolerance_rad=0.01,
                poll_interval_s=0.01,
            )

    def test_timeout_without_any_snapshot(self) -> None:
        session = FakeSession([_ready_snapshot()])
        with self.assertRaisesRegex(TimeoutError, "未读到机器人状态"):
            wait_for_target(
                session,
                target_m_rad=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                timeout_s=0.0,
                position_tolerance_mm=0.5,
                rotation_tolerance_rad=0.01,
                poll_interval_s=0.01,
            )


if __name__ == "__main__":
    unittest.main()
