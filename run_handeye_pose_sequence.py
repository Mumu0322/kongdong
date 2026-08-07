#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交互式移动到40个手眼采集点；与手眼GUI分进程同时运行，拍照由人工完成。

点位文件采用 [X_mm, Y_mm, Z_mm, Rx_rad, Ry_rad, Rz_rad]；发送给 AUBO
``moveLine`` 前仅将 XYZ 从 mm 转成 m，姿态保持 rad 不变。

默认只预览。必须显式传 ``--execute`` 并在终端输入确认短语后才连接和运动。
脚本不负责上电、启动、拍照或自动连续运行，每个点都需要人工确认。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from aubo_workbench.motion_control import (  # noqa: E402
    AuboMotionSession,
    DEFAULT_IP,
    DEFAULT_PASSWORD,
    DEFAULT_PORT,
    DEFAULT_TIMEOUT_MS,
    DEFAULT_USER,
    sdk_ok,
)


DEFAULT_PLAN = Path(r"C:\MM\aubo_tools\data\handeye_40_pose_plan_current.json")
DEFAULT_PROGRESS = Path(r"C:\MM\aubo_tools\data\handeye_40_pose_progress_current.json")
LIVE_CONFIRMATION = "MOVE HAND-EYE POSES"


def load_plan(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    poses = payload.get("poses")
    if payload.get("record_type") != "offline_handeye_40_pose_capture_plan":
        raise ValueError("不是预期的40点手眼采集计划")
    if int(payload.get("total_pose_count", 0)) != 40 or not isinstance(poses, list) or len(poses) != 40:
        raise ValueError("点位计划不完整，必须恰好包含40个点")
    indices = [int(item.get("plan_index", -1)) for item in poses]
    if indices != list(range(1, 41)):
        raise ValueError(f"点位编号必须连续为1..40，当前={indices}")

    result: list[dict[str, Any]] = []
    for item in poses:
        values = item.get("pose_mm_rad_rxryrz")
        if not isinstance(values, list) or len(values) != 6:
            raise ValueError(f"点位{item['plan_index']}缺少 pose_mm_rad_rxryrz")
        numeric = [float(value) for value in values]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"点位{item['plan_index']}包含非有限数")
        if max(abs(value) for value in numeric[3:]) > math.pi:
            raise ValueError(f"点位{item['plan_index']}姿态疑似不是rad：{numeric[3:]}")
        projection = item.get("projection_check") or {}
        if item.get("source") == "planned_new" and projection.get("ok") is not True:
            raise ValueError(f"点位{item['plan_index']}未通过离线画面边界检查")
        result.append({**item, "pose_mm_rad_rxryrz": numeric})
    return result


def pose_mm_rad_to_sdk_m_rad(values: list[float]) -> list[float]:
    if len(values) != 6:
        raise ValueError("pose must contain 6 values")
    return [
        float(values[0]) / 1000.0,
        float(values[1]) / 1000.0,
        float(values[2]) / 1000.0,
        float(values[3]),
        float(values[4]),
        float(values[5]),
    ]


def angular_delta_rad(target: float, current: float) -> float:
    return (float(target) - float(current) + math.pi) % (2.0 * math.pi) - math.pi


def pose_error(target_m_rad: list[float], current_m_rad: list[float]) -> tuple[float, float]:
    xyz_mm = math.sqrt(sum((target_m_rad[index] - current_m_rad[index]) ** 2 for index in range(3))) * 1000.0
    rotation = math.sqrt(sum(angular_delta_rad(target_m_rad[index], current_m_rad[index]) ** 2 for index in range(3, 6)))
    return xyz_mm, rotation


def format_pose_mm_rad(values: list[float]) -> str:
    return (
        f"XYZ(mm)=({values[0]:.3f}, {values[1]:.3f}, {values[2]:.3f})  "
        f"RPY(rad)=({values[3]:.6f}, {values[4]:.6f}, {values[5]:.6f})"
    )


def validate_robot_ready(snapshot: dict[str, Any]) -> None:
    if not bool(snapshot.get("power_on")):
        raise RuntimeError("机械臂未上电；本脚本不会自动上电")
    if bool(snapshot.get("collision")):
        raise RuntimeError("控制器存在碰撞标志，拒绝运动")
    if not bool(snapshot.get("steady")):
        raise RuntimeError("机械臂当前未稳定，等待稳定后再试")
    current = snapshot.get("tcp_pose_m_rad")
    if not isinstance(current, list) or len(current) < 6 or not all(math.isfinite(float(v)) for v in current[:6]):
        raise RuntimeError("当前TCP位姿不可用")


def wait_for_target(
    session: AuboMotionSession,
    target_m_rad: list[float],
    timeout_s: float,
    position_tolerance_mm: float,
    rotation_tolerance_rad: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + float(timeout_s)
    last_snapshot: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        snapshot = session.snapshot()
        last_snapshot = snapshot
        if bool(snapshot.get("collision")):
            try:
                session.stop_motion()
            finally:
                raise RuntimeError("运动期间检测到碰撞标志，已请求停止")
        current = [float(value) for value in snapshot["tcp_pose_m_rad"][:6]]
        xyz_error_mm, rotation_error_rad = pose_error(target_m_rad, current)
        if (
            bool(snapshot.get("steady"))
            and xyz_error_mm <= float(position_tolerance_mm)
            and rotation_error_rad <= float(rotation_tolerance_rad)
        ):
            return {
                "snapshot": snapshot,
                "position_error_mm": xyz_error_mm,
                "rotation_error_rad": rotation_error_rad,
            }
        time.sleep(0.10)
    if last_snapshot is None:
        raise TimeoutError("等待到位超时，且未读到机器人状态")
    current = [float(value) for value in last_snapshot["tcp_pose_m_rad"][:6]]
    xyz_error_mm, rotation_error_rad = pose_error(target_m_rad, current)
    raise TimeoutError(
        f"等待到位超时：位置误差={xyz_error_mm:.3f} mm，姿态误差={rotation_error_rad:.6f} rad"
    )


def write_progress(path: Path, pose: dict[str, Any], arrival: dict[str, Any], captured_confirmed: bool) -> None:
    payload = {
        "record_type": "handeye_40_pose_manual_capture_progress",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_arrived_plan_index": int(pose["plan_index"]),
        "manual_capture_confirmed": bool(captured_confirmed),
        "target_pose_mm_rad_rxryrz": pose["pose_mm_rad_rxryrz"],
        "position_error_mm": float(arrival["position_error_mm"]),
        "rotation_error_rad": float(arrival["rotation_error_rad"]),
        "robot_motion_command_sent": True,
        "photo_capture_command_sent": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def print_plan(poses: list[dict[str, Any]], start_index: int, end_index: int) -> None:
    for pose in poses[start_index - 1:end_index]:
        print(
            f"P{pose['plan_index']:02d} {str(pose.get('layer', '-')):>4} "
            f"{format_pose_mm_rad(pose['pose_mm_rad_rxryrz'])}"
        )


def run_interactive(args: argparse.Namespace, poses: list[dict[str, Any]]) -> int:
    selected = poses[args.start_index - 1:args.end_index]
    if not args.execute:
        print("[DRY-RUN] 未连接机器人、未发送运动。加 --execute 才会进入实机模式。")
        print_plan(poses, args.start_index, args.end_index)
        return 0

    print("即将连接机器人并允许逐点 moveLine。")
    print("此计划只有画面投影估算，没有机器人/治具碰撞模型。")
    print(f"范围：P{args.start_index:02d}..P{args.end_index:02d}，速度={args.speed_mm_s:.1f} mm/s，"
          f"加速度={args.acc_mm_s2:.1f} mm/s²")
    typed = input(f"请输入 {LIVE_CONFIRMATION!r} 确认进入实机模式：").strip()
    if typed != LIVE_CONFIRMATION:
        print("确认短语不匹配，退出；未连接机器人。")
        return 2

    session = AuboMotionSession()
    try:
        robot_name = session.connect(args.ip, args.port, args.user, args.password, args.timeout_ms)
        print(f"已连接：{robot_name}。本脚本不会上电或启动机械臂。")
        cursor = 0
        while 0 <= cursor < len(selected):
            pose = selected[cursor]
            plan_index = int(pose["plan_index"])
            target_mm_rad = pose["pose_mm_rad_rxryrz"]
            target_m_rad = pose_mm_rad_to_sdk_m_rad(target_mm_rad)
            snapshot = session.snapshot()
            validate_robot_ready(snapshot)
            current = [float(value) for value in snapshot["tcp_pose_m_rad"][:6]]
            delta_mm, delta_rad = pose_error(target_m_rad, current)
            print("\n" + "=" * 72)
            print(f"下一目标 P{plan_index:02d} ({pose.get('layer', '-')})")
            print(format_pose_mm_rad(target_mm_rad))
            print(f"相对当前位置：平移 {delta_mm:.1f} mm，姿态 {delta_rad:.4f} rad")
            if delta_mm > float(args.max_step_mm) or delta_rad > float(args.max_step_rotation_rad):
                raise RuntimeError(
                    "目标与当前位置跨度超过保护门："
                    f"平移 {delta_mm:.1f}/{args.max_step_mm:.1f} mm，"
                    f"姿态 {delta_rad:.4f}/{args.max_step_rotation_rad:.4f} rad。"
                    "请先用示教器移动到附近，或在确认路径安全后显式调整限制。"
                )
            command = input("输入 m 移动；s 跳过；b 上一点；q 退出：").strip().lower()
            if command == "q":
                break
            if command == "b":
                cursor = max(0, cursor - 1)
                continue
            if command == "s":
                cursor += 1
                continue
            if command != "m":
                print("未识别命令，没有运动。")
                continue

            validate_robot_ready(session.snapshot())
            returns = session.move_line(
                target_m_rad,
                float(args.speed_mm_s) / 1000.0,
                float(args.acc_mm_s2) / 1000.0,
            )
            if not returns or not sdk_ok(returns[-1]):
                raise RuntimeError(f"moveLine返回失败：{returns}")
            print(f"P{plan_index:02d} 已发送，等待机器人稳定到位……")
            arrival = wait_for_target(
                session,
                target_m_rad,
                args.motion_timeout_s,
                args.position_tolerance_mm,
                args.rotation_tolerance_rad,
            )
            print(
                f"P{plan_index:02d} 已到位：位置误差={arrival['position_error_mm']:.3f} mm，"
                f"姿态误差={arrival['rotation_error_rad']:.6f} rad"
            )
            write_progress(args.progress, pose, arrival, captured_confirmed=False)
            capture = input("请在手眼界面手动拍照；确认样本已保存后输入 c（r重走/q退出）：").strip().lower()
            if capture == "q":
                break
            if capture == "r":
                continue
            if capture != "c":
                print("没有收到拍照完成确认，停留在当前点。")
                continue
            write_progress(args.progress, pose, arrival, captured_confirmed=True)
            cursor += 1
        print("点位流程结束。")
        return 0
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在请求停止运动……")
        try:
            session.stop_motion()
        except Exception as exc:
            print(f"停止请求异常：{exc}")
        return 130
    finally:
        session.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AUBO手眼40点交互式运动；拍照由人工完成")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=40)
    parser.add_argument("--speed-mm-s", type=float, default=20.0)
    parser.add_argument("--acc-mm-s2", type=float, default=30.0)
    parser.add_argument("--motion-timeout-s", type=float, default=90.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=0.50)
    parser.add_argument("--rotation-tolerance-rad", type=float, default=0.002)
    # 原始三层样本的两个角点之间最大直线跨度约566 mm；650 mm可覆盖完整计划，
    # 同时仍能拦截明显错误或从远处工作位直接切入的运动。
    parser.add_argument("--max-step-mm", type=float, default=650.0)
    parser.add_argument("--max-step-rotation-rad", type=float, default=1.50)
    parser.add_argument("--ip", default=DEFAULT_IP)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--execute", action="store_true", help="允许连接机器人并逐点运动")
    args = parser.parse_args()
    if not 1 <= args.start_index <= args.end_index <= 40:
        parser.error("点位范围必须满足 1 <= start-index <= end-index <= 40")
    if args.speed_mm_s <= 0.0 or args.acc_mm_s2 <= 0.0:
        parser.error("速度和加速度必须大于0")
    if (
        args.motion_timeout_s <= 0.0
        or args.position_tolerance_mm <= 0.0
        or args.rotation_tolerance_rad <= 0.0
        or args.max_step_mm <= 0.0
        or args.max_step_rotation_rad <= 0.0
    ):
        parser.error("超时和到位容差必须大于0")
    return args


def main() -> int:
    args = parse_args()
    poses = load_plan(args.plan.resolve())
    return run_interactive(args, poses)


if __name__ == "__main__":
    raise SystemExit(main())
