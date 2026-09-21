#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""机械臂运动前置检查与到位等待。

这里集中放置"动之前必须确认什么"和"怎么判断到位"的逻辑。原来
``run_charuco_height_error_experiment.py`` 和 ``run_handeye_pose_sequence.py``
各自维护了一份语义相同但措辞和实现细节不同的副本；安全检查出现分叉时，
两个脚本会对同一个控制器状态给出不同判断，因此统一到这里。

本模块刻意不导入 tkinter，也不导入 ``motion_control``（后者在模块级引入
tkinter），这样纯离线测试和无 GUI 的实验脚本都能直接使用。
"""

from __future__ import annotations

import math
import time
from typing import Any, Protocol


class MotionSession(Protocol):
    """``wait_for_target`` 对运动会话的最小要求。

    ``AuboMotionSession`` 满足该协议；测试可以传入任何提供这两个方法的替身。
    """

    def snapshot(self) -> dict[str, Any]: ...

    def stop_motion(self) -> Any: ...


def angular_delta_rad(target: float, current: float) -> float:
    """把角度差折算到 (-pi, pi]，避免 ±pi 附近的跳变被当成巨大误差。"""
    return (float(target) - float(current) + math.pi) % (2.0 * math.pi) - math.pi


def pose_error(
    target_m_rad: list[float], current_m_rad: list[float],
) -> tuple[float, float]:
    """返回 (位置误差_mm, 姿态误差_rad)。

    位置取 XYZ 欧氏距离并换算到毫米；姿态取三个轴角差（已折算到 ±pi）的范数。
    """
    xyz_mm = math.sqrt(
        sum(
            (float(target_m_rad[index]) - float(current_m_rad[index])) ** 2
            for index in range(3)
        )
    ) * 1000.0
    rotation_rad = math.sqrt(
        sum(
            angular_delta_rad(target_m_rad[index], current_m_rad[index]) ** 2
            for index in range(3, 6)
        )
    )
    return xyz_mm, rotation_rad


def validate_robot_ready(snapshot: dict[str, Any]) -> None:
    """确认控制器状态允许运动；不满足则抛 RuntimeError。

    这里只做检查，不会自动上电、不会清碰撞标志、不会下发任何运动指令。
    """
    if not bool(snapshot.get("power_on")):
        raise RuntimeError("机械臂未上电；本脚本不会自动上电")
    if bool(snapshot.get("collision")):
        raise RuntimeError("控制器存在碰撞标志，拒绝继续")
    if not bool(snapshot.get("steady")):
        raise RuntimeError("机械臂当前未稳定，等待稳定后再试")
    pose = snapshot.get("tcp_pose_m_rad")
    if (
        not isinstance(pose, list)
        or len(pose) < 6
        or not all(math.isfinite(float(value)) for value in pose[:6])
    ):
        raise RuntimeError("当前TCP位姿不可用")


def wait_for_target(
    session: MotionSession,
    target_m_rad: list[float],
    timeout_s: float,
    position_tolerance_mm: float,
    rotation_tolerance_rad: float,
    poll_interval_s: float = 0.1,
) -> dict[str, Any]:
    """轮询等待到位，返回到位时的状态和实际误差。

    运动期间一旦读到碰撞标志，立即请求停止并抛 RuntimeError；即使停止指令
    本身失败，也保证异常向上抛出，不会静默继续。超时抛 TimeoutError 并带上
    最后一次实测误差，便于判断是公差太紧还是根本没动到位。
    """
    deadline = time.monotonic() + float(timeout_s)
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = session.snapshot()
        if bool(last.get("collision")):
            try:
                session.stop_motion()
            finally:
                raise RuntimeError("运动期间检测到碰撞标志，已请求停止")
        current = [float(value) for value in last["tcp_pose_m_rad"][:6]]
        xyz_error_mm, rotation_error_rad = pose_error(target_m_rad, current)
        if (
            bool(last.get("steady"))
            and xyz_error_mm <= float(position_tolerance_mm)
            and rotation_error_rad <= float(rotation_tolerance_rad)
        ):
            return {
                "snapshot": last,
                "position_error_mm": xyz_error_mm,
                "rotation_error_rad": rotation_error_rad,
            }
        time.sleep(float(poll_interval_s))

    if last is None:
        raise TimeoutError("等待到位超时，且未读到机器人状态")
    current = [float(value) for value in last["tcp_pose_m_rad"][:6]]
    xyz_error_mm, rotation_error_rad = pose_error(target_m_rad, current)
    raise TimeoutError(
        f"等待到位超时：位置误差={xyz_error_mm:.3f} mm，"
        f"姿态误差={rotation_error_rad:.6f} rad"
    )
