#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 ChArUco 标定板采集“视觉角点 -> 实际触碰 TCP XY”模型（PyCharm 直接运行）。

这是 TCP 示教和手眼标定之外的第三类数据：它只记录当前工具、当前高度、当前
yaw 下，视觉预测的 ChArUco 角点基坐标与人工实际触碰该角点时 TCP 基坐标的差。

程序只会自动执行“基坐标 Z”运动，把相机调到固定板面距离；每段都需输入 m 确认。
它不会自动下降到板面，也不会自动移动最终 XY。最终触碰的 Z 必须由人工示教器完成。
触碰时若发现 XY 未对准，应人工微调 XY 后再下 Z；否则不能得到 XY 真值。
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aubo_workbench.camera import get_device_identity, get_rgb_frame_bundle, init_rgb_handeye_pipeline
from aubo_workbench.charuco_detect import create_charuco_board, estimate_rgb_board_pose, get_board_chessboard_corners
from aubo_workbench.charuco_point_experiment import (
    load_handeye_experiment_result,
    nearest_detected_corner,
)
from aubo_workbench.config import ROBOT_CFG
from aubo_workbench.geometry import average_transforms
from aubo_workbench.motion_control import AuboMotionSession, sdk_ok
from aubo_workbench.robot import AuboPoseSession


# ============================================================================
# 用户配置区：在 PyCharm 中直接 Run；通常只修改这里
# ============================================================================

HANDEYE_PATH = Path(r"C:\MM\aubo_tools\data\e7_candidates\e7_handeye_candidate_current.json")
OUTPUT_ROOT = Path(r"C:\MM\aubo_tools\data\tcp_absolute_xy_model")

# 选择 3x3 个分散的 ChArUco 内部角点；每个格点只触碰一次。
# 当前目标是测量视野位置误差，因此每次必须选择不同角点，不做同点重复。
GRID_ROWS = 3
GRID_COLS = 3
REPEATS_PER_CORNER = 1

# 不再写死相机-板面距离：第一次检测到板时，以当前距离为基准再抬升该值，
# 并在本次运行内锁定。这样“只允许自动抬升Z”和高度固定不再冲突。
INITIAL_CAPTURE_HEIGHT_RAISE_MM = 20.0
HEIGHT_TOLERANCE_MM = 3.0
MAX_Z_CORRECTIONS = 3
MAX_AUTO_Z_DELTA_MM = 200.0
# 当前原点距离较远：程序绝不回原点；自动高度对准也只允许基坐标 Z 抬升。
# 若需要下降，脚本会拒绝并要求用示教器人工处理。
AUTO_RAISE_BASE_Z_ONLY = True
MOVE_SPEED_M_S = 0.020
MOVE_ACC_M_S2 = 0.080

# 当前 9 点 ChArUco 模型（charuco-tcp-xy-20260727_174559/report.json）：
# touch_tcp_xy_mm = MATRIX @ visual_corner_xy_mm + BIAS。
# 留一角点验证 P95=0.535 mm；仅适用于当前手眼、TCP、工具、固定 yaw、
# 标定板固定以及约 317~322 mm 的采集高度。
AUTO_MOVE_TO_VISUAL_XY = True
VISUAL_TO_TCP_MATRIX_2X2 = np.array([
    [1.0008588303603987, -0.0011632075996384716],
    [-0.0013284394231714038, 0.999581433830417],
], dtype=np.float64)
VISUAL_TO_TCP_BIAS_MM = np.array([0.05229713949213546, 3.1959138367376676], dtype=np.float64)
VISUAL_TO_TCP_MODEL_SOURCE = Path(
    r"C:\MM\aubo_tools\data\tcp_absolute_xy_model\charuco-tcp-xy-20260727_174559\report.json"
)

CAPTURE_VALID_FRAMES = 20
HEIGHT_MEASURE_FRAMES = 8
MAX_CAPTURE_ATTEMPTS = 80

# 只有 3 个不共线点可以拟合；默认 3x3×3=27 组用于验证，不会把训练误差当精度。
MIN_FIELD_SPAN_MM = 120.0
MAX_LOOCV_P95_MM = 0.50
WAIT_BEFORE_EXIT = True
WINDOW = "ChArUco TCP absolute XY recorder"

# 自动读档：优先恢复 OUTPUT_ROOT 下最新的 collecting / failed / stopped_by_user 运行。
# 如需强制新实验，将它改为 False；如需指定档案，填写 RESUME_RUN_DIR。
RESUME_LATEST_INCOMPLETE = True
RESUME_RUN_DIR: Path | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")


def _load_resume_report() -> tuple[Path, dict[str, Any]] | None:
    """读取指定或最新未完成档案；只接受本脚本生成的 ChArUco XY 记录。"""
    candidates: list[Path] = []
    if RESUME_RUN_DIR is not None:
        candidates.append(Path(RESUME_RUN_DIR) / "report.json")
    elif RESUME_LATEST_INCOMPLETE and OUTPUT_ROOT.exists():
        candidates.extend(sorted(OUTPUT_ROOT.glob("charuco-tcp-xy-*/report.json"),
                                 key=lambda item: item.stat().st_mtime, reverse=True))
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("mode") != "charuco_tcp_absolute_xy_model":
            continue
        if payload.get("status") not in {"collecting", "failed", "stopped_by_user"}:
            continue
        if not isinstance(payload.get("records"), list):
            continue
        saved = payload.get("configuration", {})
        if (int(saved.get("grid_rows", GRID_ROWS)) != GRID_ROWS
                or int(saved.get("grid_cols", GRID_COLS)) != GRID_COLS
                or int(saved.get("repeats", REPEATS_PER_CORNER)) != REPEATS_PER_CORNER):
            # 旧 3 次重复档案可保留，但不能被新的“每点一次”方案误续跑。
            continue
        return path.parent, payload
    return None


def _pose_to_transform(pose_session: AuboPoseSession, snapshot: dict[str, Any]) -> np.ndarray:
    if not snapshot["power_on"] or not snapshot["steady"] or snapshot["collision"]:
        raise RuntimeError(
            "机器人状态不满足要求：需要上电、稳定、无碰撞 "
            f"(power={snapshot['power_on']}, steady={snapshot['steady']}, collision={snapshot['collision']})"
        )
    return pose_session.pose_sdk_to_transform_mm(snapshot["pose_values_sdk_m_rad"])


def _current_tcp(pose_session: AuboPoseSession) -> tuple[dict[str, Any], np.ndarray]:
    snapshot = pose_session.read_pose_snapshot()
    return snapshot, _pose_to_transform(pose_session, snapshot)


def _wait_steady(pose_session: AuboPoseSession, timeout_s: float = 45.0) -> tuple[dict[str, Any], np.ndarray]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        snapshot = pose_session.read_pose_snapshot()
        if snapshot["collision"]:
            raise RuntimeError("运动后检测到碰撞标志，停止采集")
        if snapshot["power_on"] and snapshot["steady"]:
            return snapshot, pose_session.pose_sdk_to_transform_mm(snapshot["pose_values_sdk_m_rad"])
        time.sleep(0.25)
    raise RuntimeError("等待机器人稳定超时")


def _transform_to_sdk_pose_m_rad(T: np.ndarray) -> list[float]:
    """AUBO 需要 [x,y,z,rx,ry,rz]，平移由 mm 转 m。"""
    R = np.asarray(T[:3, :3], dtype=np.float64)
    ry = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    if abs(math.cos(ry)) > 1e-8:
        rx = math.atan2(R[2, 1], R[2, 2])
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = 0.0
        rz = math.atan2(-R[0, 1], R[1, 1])
    return [float(T[0, 3] / 1000.0), float(T[1, 3] / 1000.0), float(T[2, 3] / 1000.0), rx, ry, rz]


def _wait_motion_queue_empty(motion: AuboMotionSession, timeout_s: float = 2.0) -> bool:
    """AUBO clearPath 后等待规划队列释放；部分控制器此过程不是同步完成。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            queue_size = motion.snapshot().get("queue_size")
        except Exception:
            return True  # 无法读取队列时交由下发结果判断。
        if queue_size is None or int(queue_size) <= 0:
            return True
        time.sleep(0.10)
    return False


def _submit_move_line(motion: AuboMotionSession, target: np.ndarray) -> list[Any]:
    """处理 AUBO 的短暂规划队列满；只在 ret=2 时清队列后重试一次。"""
    pose = _transform_to_sdk_pose_m_rad(target)
    _wait_motion_queue_empty(motion)
    response = motion.move_line(pose, MOVE_SPEED_M_S, MOVE_ACC_M_S2)
    if response and sdk_ok(response[-1]):
        return response
    try:
        code = int(response[-1]) if response else None
    except (TypeError, ValueError):
        code = None
    if code != 2:
        return response
    print("[MOTION] AUBO 规划队列满；清空队列并等待后重试一次。")
    clear_response = motion.clear_path()
    if not sdk_ok(clear_response):
        return [clear_response, *response]
    _wait_motion_queue_empty(motion, timeout_s=3.0)
    time.sleep(0.15)
    retry = motion.move_line(pose, MOVE_SPEED_M_S, MOVE_ACC_M_S2)
    return retry


def _visual_to_tcp_target_xy(visual_xy_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """应用 ChArUco 9 点仿射模型，返回目标 TCP XY 和本次位置相关修正量。"""
    visual = np.asarray(visual_xy_mm, dtype=np.float64).reshape(2)
    target = VISUAL_TO_TCP_MATRIX_2X2 @ visual + VISUAL_TO_TCP_BIAS_MM
    return target, target - visual


def _move_base_z_only(label: str, current: np.ndarray, target_z_mm: float,
                      motion: AuboMotionSession, pose_session: AuboPoseSession) -> np.ndarray:
    target = np.asarray(current, dtype=np.float64).copy()
    target[2, 3] = float(target_z_mm)
    delta = float(target[2, 3] - current[2, 3])
    if abs(delta) < 0.02:
        return current
    if AUTO_RAISE_BASE_Z_ONLY and delta < 0.0:
        raise RuntimeError(
            f"当前高度需要基坐标 Z 下降 {abs(delta):.1f} mm 才能达到目标；"
            "本脚本只允许自动抬升 Z，请用示教器人工处理。"
        )
    if abs(delta) > MAX_AUTO_Z_DELTA_MM:
        raise RuntimeError(f"自动 Z 修正 {delta:.1f} mm 超过安全上限 {MAX_AUTO_Z_DELTA_MM:.0f} mm")
    print(
        f"\n[MOTION] {label}\n"
        f"  current XYZ(mm): {np.round(current[:3, 3], 3).tolist()}\n"
        f"  target  XYZ(mm): {np.round(target[:3, 3], 3).tolist()}\n"
        f"  仅修改基坐标 Z: {delta:+.3f} mm；XY、姿态、yaw 均保持不变"
    )
    if input("输入 m 确认 Z 运动，其他任意键取消：").strip().lower() != "m":
        raise RuntimeError("用户取消自动高度调整")
    response = _submit_move_line(motion, target)
    print("[MOTION]", response)
    if not response or not sdk_ok(response[-1]):
        raise RuntimeError(f"{label} moveLine 下发失败：{response}")
    _, actual = _wait_steady(pose_session)
    return actual


def _move_to_target_xy(label: str, current: np.ndarray, target_xy_mm: np.ndarray,
                       motion: AuboMotionSession, pose_session: AuboPoseSession) -> np.ndarray:
    """仅移动 TCP 基坐标 XY，严格保持当前 Z 和全部姿态（含 yaw）。"""
    target = np.asarray(current, dtype=np.float64).copy()
    target[:2, 3] = np.asarray(target_xy_mm, dtype=np.float64).reshape(2)
    delta = target[:3, 3] - current[:3, 3]
    if float(np.linalg.norm(delta[:2])) < 0.02:
        return current
    print(
        f"\n[MOTION] {label}\n"
        f"  current XYZ(mm): {np.round(current[:3, 3], 3).tolist()}\n"
        f"  target  XYZ(mm): {np.round(target[:3, 3], 3).tolist()}\n"
        f"  仅修改基坐标 XY: {np.round(delta[:2], 3).tolist()} mm；Z、姿态、yaw 均保持不变"
    )
    if input("输入 m 确认自动 XY 运动，其他任意键取消：").strip().lower() != "m":
        raise RuntimeError("用户取消自动 XY 对准")
    response = _submit_move_line(motion, target)
    print("[MOTION]", response)
    if not response or not sdk_ok(response[-1]):
        raise RuntimeError(f"{label} moveLine 下发失败：{response}")
    _, actual = _wait_steady(pose_session)
    return actual


def _annotate_corners(image: np.ndarray, result: Any, selected_id: int | None) -> None:
    for corner_id, point in zip(result.used_corner_ids or [], result.image_points or []):
        center = tuple(np.rint(np.asarray(point)).astype(int))
        selected = int(corner_id) == selected_id
        color = (0, 255, 255) if selected else (255, 255, 0)
        cv2.circle(image, center, 9 if selected else 4, color, 2)
        cv2.putText(image, str(int(corner_id)), (center[0] + 5, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def _clean_live_view(image: np.ndarray, result: Any, selected_id: int | None,
                     line1: str, line2: str, ok: bool) -> np.ndarray:
    """交互界面只显示必要角点，避免正式 PnP 诊断叠加层遮挡画面。"""
    display = image.copy()
    for corner_id, point in zip(result.used_corner_ids or [], result.image_points or []):
        center = tuple(np.rint(np.asarray(point)).astype(int))
        if int(corner_id) == selected_id:
            cv2.circle(display, center, 10, (0, 255, 255), 2)
            cv2.putText(display, f"ID {int(corner_id)}", (center[0] + 8, center[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(display, center, 3, (255, 255, 0), -1)
    cv2.rectangle(display, (0, 0), (min(display.shape[1], 920), 74), (24, 24, 24), -1)
    color = (40, 230, 40) if ok else (0, 180, 255)
    cv2.putText(display, line1, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)
    cv2.putText(display, line2, (14, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.53, color, 1, cv2.LINE_AA)
    return display


def _select_corner(pipeline: Any, board: Any, dictionary: Any, label: str) -> int:
    """从当前固定高度画面中鼠标选择一个 ChArUco 角点。"""
    clicked: list[tuple[int, int] | None] = [None]
    selected: list[int | None] = [None]

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked[0] = (x, y)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        bundle = get_rgb_frame_bundle(pipeline)
        if bundle is None:
            continue
        result = estimate_rgb_board_pose(bundle.color_bgr, bundle.intrinsics, board, dictionary)
        if clicked[0] is not None:
            candidate = nearest_detected_corner(clicked[0], result.used_corner_ids, result.image_points)
            if candidate is None:
                print("[SELECT] 点击位置附近没有已检测 ChArUco 角点")
            else:
                selected[0] = candidate
                print(f"[SELECT] {label} 选择角点 ID={candidate}")
            clicked[0] = None
        display = _clean_live_view(
            bundle.color_bgr, result, selected[0],
            f"{label}: click a corner, then ENTER", "Q / ESC: cancel", result.ok,
        )
        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise KeyboardInterrupt("用户取消角点选择")
        if key in (13, 10, 32) and selected[0] is not None:
            if result.ok and selected[0] in (result.valid_corner_ids or []):
                return int(selected[0])
            print("[SELECT] 当前帧 PnP/内点质量未通过，请调整画面后再确认")


def _preview_board_until_confirm(pipeline: Any, board: Any, dictionary: Any, label: str) -> None:
    """第一步就显示实时画面，避免高度计算阶段看不到相机界面。"""
    clicked: list[bool] = [False]

    def on_mouse(event: int, _x: int, _y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked[0] = True

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print("[PREVIEW] 请在相机窗口内点击左键或按 Enter，开始自动高度对准；按 Q/Esc 退出。")
    while True:
        bundle = get_rgb_frame_bundle(pipeline)
        if bundle is None:
            continue
        result = estimate_rgb_board_pose(bundle.color_bgr, bundle.intrinsics, board, dictionary)
        display = _clean_live_view(
            bundle.color_bgr, result, None,
            f"{label}: safe board view", "LEFT CLICK / ENTER: align Z | Q: quit", result.ok,
        )
        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise KeyboardInterrupt("用户取消")
        if clicked[0] or key in (13, 10, 32):
            if result.ok:
                return
            print("[PREVIEW] 当前 ChArUco PnP 未通过，请调整至可见、清晰的板面后再按 Enter")
            clicked[0] = False


def _capture_board_burst(pipeline: Any, board: Any, dictionary: Any, required: int) -> dict[str, Any]:
    poses: list[np.ndarray] = []
    rmses: list[float] = []
    maxes: list[float] = []
    counts: list[int] = []
    last_result = None
    last_bundle = None
    for attempt in range(MAX_CAPTURE_ATTEMPTS):
        if len(poses) >= required:
            break
        bundle = get_rgb_frame_bundle(pipeline)
        if bundle is None:
            continue
        result = estimate_rgb_board_pose(bundle.color_bgr, bundle.intrinsics, board, dictionary)
        last_result, last_bundle = result, bundle
        if not result.ok or result.T_rgb_board is None:
            continue
        poses.append(np.asarray(result.T_rgb_board, dtype=np.float64))
        rmses.append(float(result.rgb_reprojection_rmse_px))
        maxes.append(float(result.rgb_reprojection_max_px))
        counts.append(int(result.charuco_count))
    if len(poses) < required or last_result is None or last_bundle is None:
        raise RuntimeError(f"ChArUco 有效 PnP 帧不足：{len(poses)}/{required}")
    translation = np.asarray([T[:3, 3] for T in poses], dtype=np.float64)
    center = np.median(translation, axis=0)
    scatter = np.linalg.norm(translation - center, axis=1)
    return {
        "T_rgb_board": average_transforms(poses), "intrinsics": last_bundle.intrinsics,
        "last_result": last_result, "last_image": last_bundle.color_bgr,
        "valid_frames": len(poses), "attempts": attempt + 1,
        "charuco_count_median": float(np.median(counts)),
        "reprojection_rmse_p95_px": float(np.percentile(rmses, 95)),
        "reprojection_max_p95_px": float(np.percentile(maxes, 95)),
        "board_translation_scatter_p95_mm": float(np.percentile(scatter, 95)),
    }


def _board_height_and_z_target(T_base_tcp: np.ndarray, T_tcp_camera: np.ndarray,
                               T_rgb_board: np.ndarray, target_height_mm: float) -> tuple[float, float]:
    T_base_camera = np.asarray(T_base_tcp, dtype=np.float64) @ np.asarray(T_tcp_camera, dtype=np.float64)
    board_origin_base = T_base_camera[:3, :3] @ T_rgb_board[:3, 3] + T_base_camera[:3, 3]
    current_height = float(T_base_camera[:3, 2] @ (board_origin_base - T_base_camera[:3, 3]))
    base_z_projection = float(T_base_camera[2, 2])
    if abs(base_z_projection) < 0.1:
        raise RuntimeError("相机光轴几乎平行基坐标 XY，不能仅通过基坐标 Z 调整高度")
    target_tcp_z = float(T_base_tcp[2, 3] + (current_height - float(target_height_mm)) / base_z_projection)
    return current_height, target_tcp_z


def _corner_base_point(T_base_tcp: np.ndarray, T_tcp_camera: np.ndarray,
                       T_rgb_board: np.ndarray, corner_board_mm: np.ndarray) -> np.ndarray:
    point = np.asarray(corner_board_mm, dtype=np.float64).reshape(3)
    T_base_board = np.asarray(T_base_tcp) @ np.asarray(T_tcp_camera) @ np.asarray(T_rgb_board)
    return T_base_board[:3, :3] @ point + T_base_board[:3, 3]


def _fit_affine(source_xy: np.ndarray, target_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_xy, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_xy, dtype=np.float64).reshape(-1, 2)
    system = np.c_[source, np.ones(len(source))]
    if len(source) < 3 or np.linalg.matrix_rank(system) < 3:
        raise ValueError("至少需要 3 个不共线 ChArUco 角点")
    coefficients, *_ = np.linalg.lstsq(system, target, rcond=None)
    return coefficients[:2].T, coefficients[2]


def _predict(source_xy: np.ndarray, matrix: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.asarray(source_xy, dtype=np.float64) @ matrix.T + bias


def _statistics(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(len(array)), "median_mm": float(np.median(array)),
        "rms_mm": float(np.sqrt(np.mean(array ** 2))), "p95_mm": float(np.percentile(array, 95)),
        "max_mm": float(np.max(array)),
    }


def build_xy_model(records: list[dict[str, Any]]) -> dict[str, Any]:
    """按角点分组拟合，留出一个角点交叉验证；绝不把训练误差称为精度。"""
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        groups[int(item["corner_id"])].append(item)
    ids = sorted(groups)
    heights = np.asarray([float(item.get("capture_height_mm", np.nan)) for item in records], dtype=np.float64)
    finite_heights = heights[np.isfinite(heights)]
    base = {"sample_count": len(records), "corner_count": len(ids), "corner_ids": ids}
    if len(finite_heights):
        base["capture_height_mm"] = {
            "min": float(np.min(finite_heights)), "max": float(np.max(finite_heights)),
            "range": float(np.ptp(finite_heights)),
        }
    if len(ids) < 3:
        return base | {"status": "insufficient_data", "reason": "至少 3 个不共线角点后才可拟合"}
    source = np.asarray([np.median([x["visual_corner_base_mm"][:2] for x in groups[c]], axis=0) for c in ids])
    target = np.asarray([np.median([x["touch_tcp_pose_m_rad"][:2] for x in groups[c]], axis=0) * 1000.0 for c in ids])
    span = np.ptp(source, axis=0)
    base["visual_field_span_mm"] = {"x": float(span[0]), "y": float(span[1])}
    if min(span) < MIN_FIELD_SPAN_MM:
        return base | {"status": "insufficient_field_coverage", "reason": f"角点覆盖不足 {MIN_FIELD_SPAN_MM:.0f} mm"}
    matrix, bias = _fit_affine(source, target)
    train = np.linalg.norm(_predict(source, matrix, bias) - target, axis=1)
    loo: list[float] = []
    if len(ids) >= 4:
        for index in range(len(ids)):
            m, b = _fit_affine(np.delete(source, index, 0), np.delete(target, index, 0))
            loo.append(float(np.linalg.norm(_predict(source[index:index + 1], m, b)[0] - target[index])))
    sample_source = np.asarray([x["visual_corner_base_mm"][:2] for x in records])
    sample_target = np.asarray([x["touch_tcp_pose_m_rad"][:2] for x in records]) * 1000.0
    sample = np.linalg.norm(_predict(sample_source, matrix, bias) - sample_target, axis=1)
    loo_stats = _statistics(np.asarray(loo)) if loo else None
    status = "ready" if loo_stats and loo_stats["p95_mm"] <= MAX_LOOCV_P95_MM else "needs_more_or_better_touch_data"
    return base | {
        "status": status, "definition": "touch_tcp_xy_mm = matrix_2x2 @ visual_corner_xy_mm + bias_mm",
        "matrix_2x2": matrix, "bias_mm": bias, "training_corner_residual": _statistics(train),
        "all_sample_residual": _statistics(sample), "leave_one_corner_out_error": loo_stats,
        "acceptance_loocv_p95_mm": MAX_LOOCV_P95_MM,
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "label", "row", "col", "repeat", "corner_id", "board_x_mm", "board_y_mm",
        "visual_x_mm", "visual_y_mm", "visual_z_mm", "touch_x_mm", "touch_y_mm", "touch_z_mm",
        "residual_x_mm", "residual_y_mm", "capture_height_mm", "pnp_rmse_p95_px",
        "pnp_max_p95_px", "corner_pnp_scatter_p95_mm",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            visual, touch, board = item["visual_corner_base_mm"], item["touch_tcp_pose_m_rad"], item["corner_board_mm"]
            writer.writerow({
                "label": item["label"], "row": item["row"], "col": item["col"], "repeat": item["repeat"],
                "corner_id": item["corner_id"], "board_x_mm": board[0], "board_y_mm": board[1],
                "visual_x_mm": visual[0], "visual_y_mm": visual[1], "visual_z_mm": visual[2],
                "touch_x_mm": touch[0] * 1000.0, "touch_y_mm": touch[1] * 1000.0, "touch_z_mm": touch[2] * 1000.0,
                "residual_x_mm": item["touch_minus_visual_xy_mm"][0], "residual_y_mm": item["touch_minus_visual_xy_mm"][1],
                "capture_height_mm": item["capture_height_mm"],
                "pnp_rmse_p95_px": item["pnp_quality"]["reprojection_rmse_p95_px"],
                "pnp_max_p95_px": item["pnp_quality"]["reprojection_max_p95_px"],
                "corner_pnp_scatter_p95_mm": item["pnp_quality"]["board_translation_scatter_p95_mm"],
            })


def _label(row: int, col: int, repeat: int) -> str:
    return f"r{row + 1}c{col + 1}_rep{repeat + 1}"


def main() -> int:
    if min(GRID_ROWS, GRID_COLS, REPEATS_PER_CORNER) < 1:
        raise ValueError("网格和重复次数必须为正")
    resumed = _load_resume_report()
    if resumed is None:
        run_dir = OUTPUT_ROOT / f"charuco-tcp-xy-{datetime.now():%Y%m%d_%H%M%S}"
        run_dir.mkdir(parents=True, exist_ok=False)
        records: list[dict[str, Any]] = []
        report: dict[str, Any] = {
            "status": "collecting", "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run_dir), "mode": "charuco_tcp_absolute_xy_model",
            "limitations": [
                "只适用于当前手眼、当前TCP、当前工具、固定yaw和当前配置的相机高度",
                "最终触碰Z由人工完成；若不把工具尖端实际对到角点，不能得到XY真值",
                "标定板一旦移动，本次绝对XY模型即失效",
            ],
            "configuration": {"grid_rows": GRID_ROWS, "grid_cols": GRID_COLS, "repeats": REPEATS_PER_CORNER,
                              "initial_capture_height_raise_mm": INITIAL_CAPTURE_HEIGHT_RAISE_MM,
                              "visual_to_tcp_model_source": str(VISUAL_TO_TCP_MODEL_SOURCE),
                              "visual_to_tcp_matrix_2x2": VISUAL_TO_TCP_MATRIX_2X2,
                              "visual_to_tcp_bias_mm": VISUAL_TO_TCP_BIAS_MM},
            "records": records,
        }
    else:
        run_dir, report = resumed
        records = report["records"]
        saved = report.get("configuration", {})
        if (int(saved.get("grid_rows", GRID_ROWS)) != GRID_ROWS
                or int(saved.get("grid_cols", GRID_COLS)) != GRID_COLS
                or int(saved.get("repeats", REPEATS_PER_CORNER)) != REPEATS_PER_CORNER):
            raise RuntimeError("读档网格配置与当前代码配置不一致；请恢复原配置或关闭自动读档新建实验")
        report["status"] = "collecting"
        report["resumed_at"] = datetime.now().isoformat(timespec="seconds")
        print(f"[RESUME] 已读档：{run_dir} | 已保存 {len(records)} 组，将从下一未完成组继续。")
    report.setdefault("configuration", {}).update({
        "visual_to_tcp_model_source": str(VISUAL_TO_TCP_MODEL_SOURCE),
        "visual_to_tcp_matrix_2x2": VISUAL_TO_TCP_MATRIX_2X2,
        "visual_to_tcp_bias_mm": VISUAL_TO_TCP_BIAS_MM,
    })
    handeye = load_handeye_experiment_result(HANDEYE_PATH)
    board, dictionary = create_charuco_board()
    board_corners = get_board_chessboard_corners(board)
    pipeline = pose_session = motion = None
    cell_corner_ids: dict[tuple[int, int], int] = {
        (int(item["row"]), int(item["col"])): int(item["corner_id"])
        for item in records
    }
    completed_labels = {str(item.get("label")) for item in records}
    try:
        pipeline = init_rgb_handeye_pipeline()
        report["camera"] = get_device_identity(pipeline)
        pose_session = AuboPoseSession()
        pose_session.connect()
        motion = AuboMotionSession()
        motion.connect(ROBOT_CFG.ip, ROBOT_CFG.rpc_port, ROBOT_CFG.user, ROBOT_CFG.password, ROBOT_CFG.request_timeout_ms)
        total = GRID_ROWS * GRID_COLS * REPEATS_PER_CORNER
        print("\n[流程] 每组：人工到可见位置 -> 程序仅抬升基坐标Z至本次锁定高度 -> 选 ChArUco 角点 -> 人工触碰角点并记录TCP。")
        print("[要求] 3x3格选择分散的9个内部角点；每格只记录一次且必须选择新角点。最终Z由人工下降。")
        locked_value = report.get("configuration", {}).get("locked_capture_height_mm")
        locked_capture_height_mm: float | None = float(locked_value) if locked_value is not None else None
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                for repeat in range(REPEATS_PER_CORNER):
                    label = _label(row, col, repeat)
                    if label in completed_labels:
                        print(f"[RESUME] 跳过已保存组 {label}")
                        continue
                    print(f"\n{'=' * 72}\n[{label}] {len(records) + 1}/{total}")
                    _preview_board_until_confirm(pipeline, board, dictionary, label)
                    # 最多三次：PnP测得相机-板面距离，程序只修改基坐标 Z。
                    for correction in range(MAX_Z_CORRECTIONS + 1):
                        _, current_tcp = _current_tcp(pose_session)
                        height_observation = _capture_board_burst(pipeline, board, dictionary, HEIGHT_MEASURE_FRAMES)
                        raw_height, _ = _board_height_and_z_target(current_tcp, handeye.T_tcp_rgb_camera,
                                                                     height_observation["T_rgb_board"], 0.0)
                        if locked_capture_height_mm is None:
                            locked_capture_height_mm = raw_height + INITIAL_CAPTURE_HEIGHT_RAISE_MM
                            report["configuration"]["locked_capture_height_mm"] = locked_capture_height_mm
                            _save_json(run_dir / "report.json", report)
                            print(
                                f"[HEIGHT] 首次检测={raw_height:.2f} mm，锁定采集高度="
                                f"{locked_capture_height_mm:.2f} mm（仅抬升 {INITIAL_CAPTURE_HEIGHT_RAISE_MM:.1f} mm）"
                            )
                        height, target_z = _board_height_and_z_target(
                            current_tcp, handeye.T_tcp_rgb_camera, height_observation["T_rgb_board"],
                            locked_capture_height_mm,
                        )
                        print(f"[HEIGHT] 当前={height:.2f} mm，锁定目标={locked_capture_height_mm:.2f} mm")
                        if abs(height - locked_capture_height_mm) <= HEIGHT_TOLERANCE_MM:
                            break
                        # 现场限制为“只能自动抬升Z”。若人工把相机放得比锁定高度更高，
                        # 不自动下降、不退出；保留真实高度并在该样本报告中显式标记。
                        if AUTO_RAISE_BASE_Z_ONLY and target_z < current_tcp[2, 3]:
                            deviation = float(height - locked_capture_height_mm)
                            print(
                                f"[HEIGHT-WARN] 当前比锁定高度高 {deviation:.2f} mm；"
                                "只抬升策略不允许自动下降，本组按当前高度继续并写入报告。"
                            )
                            report.setdefault("height_warnings", []).append({
                                "label": label, "current_height_mm": height,
                                "locked_height_mm": locked_capture_height_mm,
                                "deviation_mm": deviation,
                                "policy": "accepted_without_auto_descent",
                            })
                            _save_json(run_dir / "report.json", report)
                            break
                        if correction >= MAX_Z_CORRECTIONS:
                            raise RuntimeError("自动高度修正次数已用尽，仍未达到目标高度")
                        _move_base_z_only(f"ChArUco 高度修正 {correction + 1}/{MAX_Z_CORRECTIONS}", current_tcp, target_z,
                                          motion, pose_session)

                    expected_id = cell_corner_ids.get((row, col))
                    if expected_id is None:
                        selected_id = _select_corner(pipeline, board, dictionary, label)
                    else:
                        # 仅为兼容旧档案保留；新配置每格只有一次，不会进入这里。
                        selected_id = expected_id
                        print(f"[SELECT] {label} 自动锁定本格角点 ID={selected_id}")
                    if expected_id is None and selected_id in cell_corner_ids.values():
                        raise RuntimeError(f"角点 ID={selected_id} 已属于其它网格位置；请为每格选择不同角点")
                    cell_corner_ids[(row, col)] = selected_id

                    before_snapshot, capture_tcp = _current_tcp(pose_session)
                    observation = _capture_board_burst(pipeline, board, dictionary, CAPTURE_VALID_FRAMES)
                    after_snapshot, after_tcp = _current_tcp(pose_session)
                    if np.linalg.norm(after_tcp[:3, 3] - capture_tcp[:3, 3]) > 0.20:
                        raise RuntimeError("ChArUco采集期间 TCP 位置变化超过0.20mm")
                    if selected_id >= len(board_corners):
                        raise RuntimeError(f"无效 ChArUco 角点 ID={selected_id}")
                    corner_board = np.asarray(board_corners[selected_id], dtype=np.float64)
                    visual_corner = _corner_base_point(capture_tcp, handeye.T_tcp_rgb_camera,
                                                        observation["T_rgb_board"], corner_board)
                    overlay = observation["last_result"].rgb_overlay.copy()
                    _annotate_corners(overlay, observation["last_result"], selected_id)
                    cv2.imwrite(str(run_dir / f"{label}_capture.png"), overlay)
                    print(f"[视觉] corner={selected_id} | predicted XY={np.round(visual_corner[:2], 3).tolist()} mm")
                    planned_tcp = capture_tcp.copy()
                    model_target_xy, model_correction_xy = _visual_to_tcp_target_xy(visual_corner[:2])
                    planned_tcp[:2, 3] = model_target_xy
                    print(
                        f"[MODEL] ChArUco仿射修正={np.round(model_correction_xy, 3).tolist()} mm | "
                        f"target XY={np.round(model_target_xy, 3).tolist()} mm"
                    )
                    if AUTO_MOVE_TO_VISUAL_XY:
                        aligned_tcp = _move_to_target_xy(
                            f"自动移动至 ChArUco 角点 ID={selected_id} 的目标 XY", capture_tcp,
                            planned_tcp[:2, 3], motion, pose_session,
                        )
                    else:
                        aligned_tcp = capture_tcp.copy()
                    if input("请人工将工具尖端真正触到该 ChArUco 角点；最终Z手动下降，必要时微调XY；稳定后按 Enter（q结束）：").strip().lower() == "q":
                        raise KeyboardInterrupt
                    touch_snapshot, touch_tcp = _current_tcp(pose_session)
                    residual_xy = touch_tcp[:2, 3] - visual_corner[:2]
                    record = {
                        "label": label, "row": row, "col": col, "repeat": repeat, "corner_id": selected_id,
                        "recorded_at": datetime.now().isoformat(timespec="seconds"), "corner_board_mm": corner_board,
                        "visual_corner_base_mm": visual_corner, "capture_tcp_pose_m_rad": _transform_to_sdk_pose_m_rad(capture_tcp),
                        "planned_target_tcp_pose_m_rad": _transform_to_sdk_pose_m_rad(planned_tcp),
                        "aligned_tcp_pose_m_rad": _transform_to_sdk_pose_m_rad(aligned_tcp),
                        "visual_to_tcp_model_source": str(VISUAL_TO_TCP_MODEL_SOURCE),
                        "visual_to_tcp_xy_correction_mm": model_correction_xy,
                        "touch_tcp_pose_m_rad": _transform_to_sdk_pose_m_rad(touch_tcp),
                        "capture_height_mm": height, "locked_capture_height_mm": locked_capture_height_mm,
                        "capture_height_deviation_mm": float(height - locked_capture_height_mm),
                        "touch_minus_visual_xy_mm": residual_xy,
                        "pnp_quality": {key: observation[key] for key in (
                            "valid_frames", "attempts", "charuco_count_median", "reprojection_rmse_p95_px",
                            "reprojection_max_p95_px", "board_translation_scatter_p95_mm",
                        )},
                        "capture_robot_snapshot_before": before_snapshot, "capture_robot_snapshot_after": after_snapshot,
                        "touch_robot_snapshot": touch_snapshot,
                    }
                    records.append(record)
                    completed_labels.add(label)
                    report["model"] = build_xy_model(records)
                    _write_csv(run_dir / "charuco_tcp_xy_samples.csv", records)
                    _save_json(run_dir / "report.json", report)
                    loo = (report["model"].get("leave_one_corner_out_error") or {}).get("p95_mm", float("nan"))
                    print(f"[触碰差] TCP - visual = {np.round(residual_xy, 3).tolist()} mm | LOOCV P95={loo:.3f} mm")
                    print("[NEXT] 请人工仅抬升 Z，使标定板重新清晰可见；随后在相机窗口点击左键进入下一组。")
        report["status"] = "completed"
        report["model"] = build_xy_model(records)
        _write_csv(run_dir / "charuco_tcp_xy_samples.csv", records)
        _save_json(run_dir / "report.json", report)
        print("\n[完成]", run_dir)
        print(json.dumps(_jsonable(report["model"]), ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        report["status"] = "stopped_by_user"
        report["model"] = build_xy_model(records)
        _write_csv(run_dir / "charuco_tcp_xy_samples.csv", records)
        _save_json(run_dir / "report.json", report)
        print(f"\n[停止] 已保存 {len(records)} 组记录：{run_dir}")
        return 130
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["model"] = build_xy_model(records)
        _write_csv(run_dir / "charuco_tcp_xy_samples.csv", records)
        _save_json(run_dir / "report.json", report)
        print(f"\n[FAILED] {type(exc).__name__}: {exc}")
        print(f"[FAILED] 已保留 {len(records)} 组记录：{run_dir}")
        traceback.print_exc()
        return 1
    finally:
        if pipeline is not None:
            pipeline.stop()
        if pose_session is not None:
            pose_session.disconnect()
        if motion is not None:
            motion.disconnect()
        cv2.destroyAllWindows()
        if WAIT_BEFORE_EXIT:
            try:
                input("\n按 Enter 结束程序...")
            except (EOFError, KeyboardInterrupt):
                pass


if __name__ == "__main__":
    raise SystemExit(main())
