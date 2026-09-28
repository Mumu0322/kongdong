#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""实机验证 AUBO 连续交融路径的下发方式（飞拍"一顿一顿"问题诊断）。

在当前 TCP 所在高度走一个小正方形（4 段直线，姿态不变，回到起点），
分别用以下方式各下发一次，每段只下发一次、不重试：

    baseline  现状：RuntimeMachine 停止状态下直接 moveLine(blend)
    runtime   先 RuntimeMachine.start()，再 moveLine(blend)，结束 stop()
    process   RuntimeMachine.start() + moveProcess(blend)，结束 stop()

独立连接以约 100 Hz 记录 TCP 位姿/速度、getQueueSize、getExecId、
isBlending，据此判断拐角是否停顿、是否有段被丢弃。

默认只打印计划，不运动。实机运行需要 --execute 并在终端输入 YES：

    python tools/blend_motion_probe.py
    python tools/blend_motion_probe.py --execute
    python tools/blend_motion_probe.py --execute --modes runtime,process --dx -1

运行前确认：正方形范围内无障碍、急停在手边、示教器上无程序运行。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from aubo_workbench.paths import (  # noqa: E402
    DATA_DIR,
    DEFAULT_ROBOT_IP,
    DEFAULT_ROBOT_PASSWORD,
    DEFAULT_ROBOT_PORT,
    DEFAULT_ROBOT_TIMEOUT_MS,
    DEFAULT_ROBOT_USER,
)

MODES = ("baseline", "runtime", "process")
MAX_SIDE_MM = 60.0
MAX_SPEED_M_S = 0.03
OUTPUT_DIR = DATA_DIR / "blend_motion_probe"


def plan_square(start_pose: Sequence[float], side_mm: float, dx_sign: int, dy_sign: int) -> list[list[float]]:
    """Four corners of a square in the start pose's Z plane, ending at the start."""
    x, y = float(start_pose[0]), float(start_pose[1])
    side = side_mm / 1000.0
    ox, oy = dx_sign * side, dy_sign * side
    rest = [float(v) for v in start_pose[2:6]]
    return [
        [x + ox, y, *rest],
        [x + ox, y + oy, *rest],
        [x, y + oy, *rest],
        [x, y, *rest],
    ]


def windowed_speeds(samples: list[dict[str, Any]], window_s: float = 0.05) -> list[float | None]:
    """Speed over >= ``window_s``; per-sample differences read 0 whenever the
    controller returns the same pose twice (probe 20260928_105034)."""
    speeds: list[float | None] = []
    j = 0
    for s in samples:
        while j + 1 < len(samples) and s["t_s"] - samples[j + 1]["t_s"] >= window_s:
            j += 1
        dt = s["t_s"] - samples[j]["t_s"]
        speeds.append(math.dist(s["xyz_m"], samples[j]["xyz_m"]) / dt if dt >= window_s else None)
    return speeds


def summarize_samples(
    samples: list[dict[str, Any]],
    waypoints: list[list[float]],
    *,
    speed_m_s: float,
    blend_m: float,
) -> dict[str, Any]:
    """Classify each interior corner as blended, stopped or dropped."""
    speeds = windowed_speeds(samples)
    moving = [s for s, v in zip(samples, speeds) if v is not None and v > 0.1 * speed_m_s]
    corners = []
    for index, corner in enumerate(waypoints[:-1]):
        near = [
            v for s, v in zip(samples, speeds)
            if v is not None and math.dist(s["xyz_m"][:2], corner[:2]) <= max(2.5 * blend_m, 0.008)
        ]
        closest = min((math.dist(s["xyz_m"][:2], corner[:2]) for s in samples), default=math.inf)
        min_speed = min(near, default=None)
        if closest > 2.0 * blend_m:
            verdict = "dropped"
        elif min_speed is not None and min_speed < 0.1 * speed_m_s:
            verdict = "stopped"
        else:
            verdict = "blended"
        corners.append({
            "corner": index + 1,
            "closest_mm": round(closest * 1000.0, 2),
            "min_speed_mm_s": None if min_speed is None else round(min_speed * 1000.0, 2),
            "verdict": verdict,
        })
    motion_s = (moving[-1]["t_s"] - moving[0]["t_s"]) if len(moving) >= 2 else 0.0
    # The last waypoint is the start pose, so the closed loop covers every segment.
    loop = [waypoints[-1], *waypoints]
    path_m = sum(math.dist(a[:3], b[:3]) for a, b in zip(loop, loop[1:]))
    return {
        "corners": corners,
        "motion_s": round(motion_s, 3),
        "ideal_constant_speed_s": round(path_m / speed_m_s, 3),
        "max_queue_size": max((s["queue_size"] for s in samples if s["queue_size"] is not None), default=None),
        "blending_samples": sum(1 for s in samples if s["blending"]),
        "continuous": bool(corners) and all(c["verdict"] == "blended" for c in corners),
    }


class Robot:
    """One RPC connection; the sampler uses its own instance."""

    def __init__(self, args: argparse.Namespace) -> None:
        from aubo_workbench.sdk_paths import add_aubo_sdk_to_path

        add_aubo_sdk_to_path()
        import pyaubo_sdk as aubo

        self.aubo = aubo
        self.client = aubo.RpcClient()
        self.client.setRequestTimeout(int(args.timeout_ms))
        if self.client.connect(args.ip, int(args.port)) != aubo.AUBO_OK:
            raise RuntimeError("连接控制器失败")
        if self.client.login(args.user, args.password) != aubo.AUBO_OK:
            self.client.disconnect()
            raise RuntimeError("登录控制器失败")
        robot = self.client.getRobotInterface(self.client.getRobotNames()[0])
        self.state = robot.getRobotState()
        self.motion = robot.getMotionControl()
        self.manage = robot.getRobotManage()
        self.runtime = self.client.getRuntimeMachine()

    def close(self) -> None:
        try:
            self.client.disconnect()
        except Exception:
            pass

    def tcp(self) -> list[float]:
        return [float(v) for v in self.state.getTcpPose()]

    def preflight(self) -> None:
        problems = []
        if not self.state.isPowerOn():
            problems.append("未上电")
        if not self.state.isSteady():
            problems.append("未静止")
        if self.state.isCollisionOccurred():
            problems.append("碰撞标志")
        if "Normal" not in str(self.state.getSafetyModeType()):
            problems.append(f"安全模式 {self.state.getSafetyModeType()}")
        if self.manage.isFreedriveEnabled():
            problems.append("拖动示教已开启")
        if "Stopped" not in str(self.runtime.getStatus()):
            problems.append(f"运行机状态 {self.runtime.getStatus()}（示教器可能有程序在跑）")
        if problems:
            raise RuntimeError("预检失败：" + "、".join(problems))


def sampler(args: argparse.Namespace, stop: threading.Event, out: list[dict[str, Any]]) -> None:
    robot = Robot(args)
    t0 = time.monotonic()
    prev: tuple[float, list[float]] | None = None
    try:
        while not stop.is_set():
            now = time.monotonic() - t0
            pose = robot.tcp()
            try:
                queue_size = int(robot.motion.getQueueSize())
            except Exception:
                queue_size = None
            speed = 0.0
            if prev is not None and now > prev[0]:
                speed = math.dist(pose[:3], prev[1][:3]) / (now - prev[0])
            prev = (now, pose)
            out.append({
                "t_s": round(now, 4),
                "xyz_m": pose[:3],
                "speed_m_s": speed,
                "queue_size": queue_size,
                "exec_id": int(robot.motion.getExecId()),
                "blending": bool(robot.motion.isBlending()),
                "steady": bool(robot.state.isSteady()),
                "collision": bool(robot.state.isCollisionOccurred()),
            })
            time.sleep(0.008)
    finally:
        robot.close()


def wait_done(robot: Robot, samples: list[dict[str, Any]], timeout_s: float) -> str:
    """Wait until queue is empty and the arm is steady for 0.3 s."""
    deadline = time.monotonic() + timeout_s
    quiet_since = None
    time.sleep(0.3)
    while time.monotonic() < deadline:
        if robot.state.isCollisionOccurred():
            robot.motion.stopMove(False, True)
            return "collision"
        idle = robot.state.isSteady() and int(robot.motion.getQueueSize()) == 0
        if idle:
            quiet_since = quiet_since or time.monotonic()
            if time.monotonic() - quiet_since >= 0.3:
                return "done"
        else:
            quiet_since = None
        time.sleep(0.02)
    robot.motion.stopMove(False, True)
    return "timeout"


def run_mode(
    robot: Robot, args: argparse.Namespace, mode: str, waypoints: list[list[float]],
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    stop = threading.Event()
    thread = threading.Thread(target=sampler, args=(args, stop, samples), daemon=True)
    use_runtime = mode in ("runtime", "process")
    record: dict[str, Any] = {"mode": mode, "returns": []}
    robot.motion.clearPath()
    thread.start()
    time.sleep(0.3)
    try:
        if use_runtime:
            record["runtime_start"] = int(robot.runtime.start())
        sent_at = time.monotonic()
        for index, pose in enumerate(waypoints):
            blend = args.blend_mm / 1000.0 if index < len(waypoints) - 1 else 0.0
            if mode == "process":
                ret = robot.motion.moveProcess(pose, args.acc, args.speed, blend)
            else:
                ret = robot.motion.moveLine(pose, args.acc, args.speed, blend, 0.0)
            record["returns"].append({
                "segment": index + 1,
                "ret": int(ret),
                "t_s": round(time.monotonic() - sent_at, 3),
            })
        record["send_s"] = round(time.monotonic() - sent_at, 3)
        record["finish"] = wait_done(robot, samples, timeout_s=60.0)
    except BaseException:
        robot.motion.stopMove(False, True)
        raise
    finally:
        if use_runtime:
            record["runtime_stop"] = int(robot.runtime.stop())
        time.sleep(0.3)
        stop.set()
        thread.join(timeout=5.0)
    # The pendant speed slider scales the commanded speed (field value 0.5).
    fraction = float(robot.motion.getSpeedFraction())
    record["speed_fraction"] = fraction
    record["summary"] = summarize_samples(
        samples, waypoints, speed_m_s=args.speed * fraction, blend_m=args.blend_mm / 1000.0,
    )
    record["samples"] = samples
    return record


def return_to_start(robot: Robot, args: argparse.Namespace, start: list[float]) -> None:
    """Straight move back inside the square if a mode ended off the start pose."""
    if math.dist(robot.tcp()[:3], start[:3]) <= 0.001:
        return
    print("[PROBE] 末端未回到起点，直线移回起点", flush=True)
    robot.motion.clearPath()
    ret = robot.motion.moveLine(start, args.acc, args.speed, 0.0, 0.0)
    print(f"[PROBE]   moveLine 返回 {ret}", flush=True)
    if wait_done(robot, [], timeout_s=30.0) != "done":
        raise RuntimeError("回到起点未完成，停止实验")
    if math.dist(robot.tcp()[:3], start[:3]) > 0.001:
        raise RuntimeError("回到起点后位置偏差仍大于 1 mm，停止实验")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="实际运动；否则只打印计划")
    parser.add_argument("--modes", default=",".join(MODES), help="逗号分隔：baseline,runtime,process")
    parser.add_argument("--side-mm", type=float, default=40.0)
    parser.add_argument("--speed", type=float, default=0.02, help="m/s")
    parser.add_argument("--acc", type=float, default=0.25, help="m/s^2")
    parser.add_argument("--blend-mm", type=float, default=5.0)
    parser.add_argument("--dx", type=int, choices=(-1, 1), default=1, help="正方形沿基座 X 的方向")
    parser.add_argument("--dy", type=int, choices=(-1, 1), default=1, help="正方形沿基座 Y 的方向")
    parser.add_argument("--lookahead", type=int, default=None,
                        help="实验期间临时设置 setLookAheadSize，结束后恢复原值")
    parser.add_argument("--ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--port", type=int, default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--user", default=DEFAULT_ROBOT_USER)
    parser.add_argument("--password", default=DEFAULT_ROBOT_PASSWORD)
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_ROBOT_TIMEOUT_MS)
    args = parser.parse_args(argv)
    args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in args.modes if m not in MODES]
    if unknown:
        parser.error(f"未知模式：{unknown}")
    if not 0.0 < args.side_mm <= MAX_SIDE_MM:
        parser.error(f"--side-mm 必须在 (0, {MAX_SIDE_MM:g}]")
    if not 0.0 < args.speed <= MAX_SPEED_M_S:
        parser.error(f"--speed 必须在 (0, {MAX_SPEED_M_S:g}] m/s")
    if not 0.0 < args.acc <= 0.5:
        parser.error("--acc 必须在 (0, 0.5] m/s^2")
    if not 1.0 <= args.blend_mm <= min(20.0, args.side_mm / 2.0):
        parser.error("--blend-mm 必须在 1 mm 到边长一半之间")
    return args


def print_summary(record: dict[str, Any]) -> None:
    s = record["summary"]
    print(f"\n=== {record['mode']}  结束={record.get('finish')}  下发耗时={record.get('send_s')}s")
    print("  返回码：" + ", ".join(f"#{r['segment']}={r['ret']}@{r['t_s']}s" for r in record["returns"]))
    print(f"  运动时长 {s['motion_s']}s（恒速理想 {s['ideal_constant_speed_s']}s），"
          f"最大队列 {s['max_queue_size']}，isBlending 采样数 {s['blending_samples']}")
    for c in s["corners"]:
        print(f"  拐角{c['corner']}: {c['verdict']:8s} 最近 {c['closest_mm']} mm，"
              f"附近最低速度 {c['min_speed_mm_s']} mm/s")
    print("  结论：" + ("连续交融" if s["continuous"] else "不连续"))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    robot = Robot(args)
    original_lookahead = None
    try:
        robot.preflight()
        start = robot.tcp()
        waypoints = plan_square(start, args.side_mm, args.dx, args.dy)
        print(f"[PROBE] 起点 TCP (mm): {[round(v * 1000, 1) for v in start[:3]]}，姿态不变")
        for i, p in enumerate(waypoints, 1):
            print(f"[PROBE]   点{i}: {[round(v * 1000, 1) for v in p[:3]]}")
        print(f"[PROBE] 速度 {args.speed * 1000:g} mm/s，加速度 {args.acc:g} m/s²，"
              f"交融 {args.blend_mm:g} mm，模式 {args.modes}")
        print(f"[PROBE] 控制器 lookahead={robot.motion.getLookAheadSize()}，"
              f"运行机={robot.runtime.getStatus()}，速度倍率={robot.motion.getSpeedFraction()}")
        if not args.execute:
            print("[PROBE] 仅打印计划。确认范围安全后加 --execute 运行。")
            return 0
        if input("确认正方形范围无障碍、急停在手边，输入 YES 开始：").strip() != "YES":
            print("[PROBE] 已取消")
            return 1
        if args.lookahead is not None:
            original_lookahead = int(robot.motion.getLookAheadSize())
            print(f"[PROBE] setLookAheadSize({args.lookahead}) -> "
                  f"{robot.motion.setLookAheadSize(args.lookahead)}")
        results = []
        for mode in args.modes:
            robot.preflight()
            return_to_start(robot, args, start)
            print(f"\n[PROBE] 开始 {mode} ...", flush=True)
            record = run_mode(robot, args, mode, waypoints)
            print_summary(record)
            results.append(record)
            if record["finish"] != "done":
                print("[PROBE] 本模式未正常结束，停止后续实验")
                break
        return_to_start(robot, args, start)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / f"probe-{datetime.now():%Y%m%d_%H%M%S}.json"
        out.write_text(json.dumps({
            "start_tcp_m_rad": start, "waypoints": waypoints,
            "args": {k: v for k, v in vars(args).items() if k != "password"},
            "original_lookahead": original_lookahead, "results": results,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n[PROBE] 结果已保存：{out}")
        return 0
    except KeyboardInterrupt:
        robot.motion.stopMove(False, True)
        print("\n[PROBE] 已中断并下发停止")
        return 130
    finally:
        if original_lookahead is not None:
            robot.motion.setLookAheadSize(original_lookahead)
            print(f"[PROBE] 已恢复 lookahead={original_lookahead}")
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
