#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ChArUco 多高度相机误差实验：纯统计函数与现场采集流程。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from aubo_workbench.camera import (
    CameraFrameBundle,
    RgbFrameBundle,
    depth_to_vis,
    get_aligned_frame_bundle,
    get_color_profile,
    get_depth_profile,
    get_device_identity,
    get_rgb_frame_bundle,
    init_pipeline,
    init_rgb_handeye_pipeline,
)
from aubo_workbench.charuco_detect import (
    BoardPoseResult,
    create_charuco_board,
    estimate_pointcloud_board_pose,
    estimate_rgb_board_pose,
)
from aubo_workbench.config import (
    BOARD_CFG,
    CAMERA_CFG,
    ROBOT_CFG,
    ROBOT_CAMERA_INTEGRATION_CFG,
)
from aubo_workbench.io_utils import atomic_write_json, timestamp_str
from aubo_workbench.motion_control import (
    AuboMotionSession,
    sdk_ok,
)
from aubo_workbench.quality import evaluate_image_quality


DEFAULT_OUTPUT_ROOT = Path(r"C:\MM\aubo_tools\data\charuco_height_error")
# 默认视野实验高度：300/320/340/360 mm。
DEFAULT_HEIGHTS_MM = (300.0, 320.0, 340.0, 360.0)
# 1280x800 RGB画面中的3x3目标板中心；XY由操作者手动移动。
DEFAULT_FIELD_GRID = (
    ("R1C1", 0.30, 0.30), ("R1C2", 0.50, 0.30), ("R1C3", 0.70, 0.30),
    ("R2C1", 0.30, 0.50), ("R2C2", 0.50, 0.50), ("R2C3", 0.70, 0.50),
    ("R3C1", 0.30, 0.70), ("R3C2", 0.50, 0.70), ("R3C3", 0.70, 0.70),
)
RGB_PRECISION_DEFAULT_HEIGHTS_MM = tuple(float(value) for value in range(200, 301, 10))
RGB_PRECISION_POSE_NAMES = ("CENTER", "LEFT", "RIGHT", "UP", "DOWN")
RGB_PRECISION_FRAME_FIELDS = (
    "run_id", "height_target_mm", "pose_name", "pose_index", "frame_index",
    "captured_at", "valid", "failure_reasons",
    "color_timestamp_us", "color_frame_index",
    "rgb_status", "rgb_charuco_count", "rgb_pnp_inlier_count",
    "rgb_reprojection_rmse_px", "rgb_reprojection_max_px",
    "rgb_x_mm", "rgb_y_mm", "rgb_z_mm",
    "rgb_center_u_px", "rgb_center_v_px", "target_u_px", "target_v_px",
    "target_distance_px", "brightness", "contrast", "sharpness",
    "tcp_x_mm", "tcp_y_mm", "tcp_z_mm",
    "tcp_rx_rad", "tcp_ry_rad", "tcp_rz_rad",
    "tcp_power_on", "tcp_steady", "tcp_collision",
)
CSV_FIELDS = (
    "run_id", "height_target_mm", "frame_index", "batch_index", "batch_frame_index",
    "captured_at", "valid", "failure_reasons",
    "color_timestamp_us", "depth_timestamp_us", "rgb_depth_timestamp_delta_us",
    "color_frame_index", "depth_frame_index",
    "rgb_ok", "rgb_status", "rgb_charuco_count", "rgb_pnp_inlier_count",
    "rgb_reprojection_rmse_px", "rgb_reprojection_max_px",
    "rgb_x_mm", "rgb_y_mm", "rgb_z_mm", "rgb_normal_x", "rgb_normal_y", "rgb_normal_z",
    "rgb_board_area_ratio", "rgb_center_offset_ratio",
    "grid_position", "grid_row", "grid_col", "grid_target_u_px", "grid_target_v_px",
    "rgb_center_u_px", "rgb_center_v_px", "rgb_center_u_norm", "rgb_center_v_norm",
    "depth_ok", "depth_status", "depth_valid_3d_count",
    "depth_corner_rmse_mm", "depth_corner_max_error_mm", "depth_plane_rmse_mm",
    "depth_plane_inlier_count", "depth_board_mask_point_count",
    "depth_x_mm", "depth_y_mm", "depth_z_mm",
    "depth_normal_x", "depth_normal_y", "depth_normal_z",
    "cross_dx_mm", "cross_dy_mm", "cross_dz_mm", "cross_distance_mm",
    "cross_normal_angle_deg", "control_residual_mm",
    "brightness", "contrast", "sharpness",
    "tcp_x_mm", "tcp_y_mm", "tcp_z_mm", "tcp_rx_rad", "tcp_ry_rad", "tcp_rz_rad",
    "tcp_power_on", "tcp_steady", "tcp_collision",
)


@dataclass(frozen=True)
class HeightExperimentConfig:
    heights_mm: tuple[float, ...] = DEFAULT_HEIGHTS_MM
    frames_per_height: int = 200
    batch_size: int = 10
    warmup_frames: int = 90
    target_tolerance_mm: float = 2.0
    max_step_mm: float = 70.0
    max_corrections: int = 3
    speed_mm_s: float = 20.0
    acceleration_mm_s2: float = 30.0
    motion_timeout_s: float = 90.0
    position_tolerance_mm: float = 0.50
    rotation_tolerance_rad: float = 0.002
    expected_camera_serial: str = ""
    depth_filter_mode: str = "none"
    execute_motion: bool = False
    field_grid: bool = False
    frames_per_position: int = 20
    grid_image_width: int = 1280
    grid_image_height: int = 800

    def validate(self) -> None:
        if not self.heights_mm or any(not math.isfinite(v) or v <= 0 for v in self.heights_mm):
            raise ValueError("heights_mm 必须是非空正数")
        if self.frames_per_height <= 0 or self.batch_size <= 0:
            raise ValueError("frames_per_height 和 batch_size 必须大于0")
        if self.frames_per_height % self.batch_size != 0:
            raise ValueError("frames_per_height 必须能被 batch_size 整除")
        if self.frames_per_position <= 0 or self.grid_image_width <= 0 or self.grid_image_height <= 0:
            raise ValueError("frames_per_position 和网格图像尺寸必须大于0")
        if self.warmup_frames < 0 or self.max_corrections <= 0:
            raise ValueError("warmup_frames 不能为负，max_corrections 必须大于0")
        if min(
            self.target_tolerance_mm, self.max_step_mm, self.speed_mm_s,
            self.acceleration_mm_s2, self.motion_timeout_s,
            self.position_tolerance_mm, self.rotation_tolerance_rad,
        ) <= 0:
            raise ValueError("运动和容差参数必须大于0")
        if self.depth_filter_mode not in {"none", "temporal"}:
            raise ValueError("depth_filter_mode 只支持 none/temporal")

@dataclass(frozen=True)
class RgbPrecisionRangeConfig:
    heights_mm: tuple[float, ...] = RGB_PRECISION_DEFAULT_HEIGHTS_MM
    frames_per_position: int = 20
    batch_size: int = 10
    warmup_frames: int = 90
    center_step_mm: float = 20.0
    confirm_tolerance_px: float = 20.0
    min_tcp_step_mm: float = 15.0
    max_tcp_step_mm: float = 30.0
    max_tcp_z_drift_mm: float = 0.2
    max_tcp_rotation_drift_deg: float = 0.05
    minimum_valid_ratio: float = 0.95
    maximum_distance_p95_mm: float = 0.5
    target_tolerance_mm: float = 2.0
    max_step_mm: float = 70.0
    max_corrections: int = 3
    speed_mm_s: float = 20.0
    acceleration_mm_s2: float = 30.0
    motion_timeout_s: float = 90.0
    position_tolerance_mm: float = 0.50
    rotation_tolerance_rad: float = 0.002
    expected_camera_serial: str = ""
    execute_motion: bool = False
    auto_xy: bool = True

    def validate(self) -> None:
        if not self.heights_mm or any(not math.isfinite(v) or v <= 0 for v in self.heights_mm):
            raise ValueError("RGB精度实验高度必须是非空正数")
        if any(b <= a for a, b in zip(self.heights_mm, self.heights_mm[1:])):
            raise ValueError("RGB精度实验高度必须严格递增")
        if self.frames_per_position <= 0 or self.batch_size <= 0:
            raise ValueError("frames_per_position和batch_size必须大于0")
        if self.frames_per_position % self.batch_size != 0:
            raise ValueError("frames_per_position必须能被batch_size整除")
        positive = (
            self.center_step_mm, self.confirm_tolerance_px, self.min_tcp_step_mm,
            self.max_tcp_step_mm, self.max_tcp_z_drift_mm,
            self.max_tcp_rotation_drift_deg, self.minimum_valid_ratio,
            self.maximum_distance_p95_mm, self.target_tolerance_mm,
            self.max_step_mm, self.speed_mm_s, self.acceleration_mm_s2,
            self.motion_timeout_s, self.position_tolerance_mm,
            self.rotation_tolerance_rad,
        )
        if min(positive) <= 0:
            raise ValueError("RGB精度实验的数值门槛必须大于0")
        if self.min_tcp_step_mm >= self.max_tcp_step_mm:
            raise ValueError("min_tcp_step_mm必须小于max_tcp_step_mm")
        if self.minimum_valid_ratio > 1.0:
            raise ValueError("minimum_valid_ratio不能大于1")


def _finite(values: Iterable[Any]) -> np.ndarray:
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return np.asarray(result, dtype=np.float64)


def _angular_delta_rad(target: float, current: float) -> float:
    return (float(target) - float(current) + math.pi) % (2.0 * math.pi) - math.pi


def _pose_error(target_m_rad: list[float], current_m_rad: list[float]) -> tuple[float, float]:
    xyz_mm = float(np.linalg.norm(
        np.asarray(target_m_rad[:3], dtype=np.float64)
        - np.asarray(current_m_rad[:3], dtype=np.float64)
    ) * 1000.0)
    rotation_rad = float(np.linalg.norm([
        _angular_delta_rad(target_m_rad[index], current_m_rad[index])
        for index in range(3, 6)
    ]))
    return xyz_mm, rotation_rad


def validate_robot_ready(snapshot: dict[str, Any]) -> None:
    if not bool(snapshot.get("power_on")):
        raise RuntimeError("机械臂未上电；实验脚本不会自动上电")
    if bool(snapshot.get("collision")):
        raise RuntimeError("控制器存在碰撞标志，拒绝继续")
    if not bool(snapshot.get("steady")):
        raise RuntimeError("机械臂当前未稳定")
    pose = snapshot.get("tcp_pose_m_rad")
    if not isinstance(pose, list) or len(pose) < 6 or not np.isfinite(
        np.asarray(pose[:6], dtype=np.float64)
    ).all():
        raise RuntimeError("当前TCP位姿不可用")


def wait_for_target(
    session: AuboMotionSession,
    target_m_rad: list[float],
    timeout_s: float,
    position_tolerance_mm: float,
    rotation_tolerance_rad: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + float(timeout_s)
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = session.snapshot()
        if bool(last.get("collision")):
            try:
                session.stop_motion()
            finally:
                raise RuntimeError("运动期间检测到碰撞标志，已请求停止")
        current = [float(v) for v in last["tcp_pose_m_rad"][:6]]
        xyz_error_mm, rotation_error_rad = _pose_error(target_m_rad, current)
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
        time.sleep(0.1)
    if last is None:
        raise TimeoutError("等待到位超时，且未读到机器人状态")
    xyz_error_mm, rotation_error_rad = _pose_error(
        target_m_rad, [float(v) for v in last["tcp_pose_m_rad"][:6]],
    )
    raise TimeoutError(
        f"等待到位超时：位置误差={xyz_error_mm:.3f} mm，"
        f"姿态误差={rotation_error_rad:.6f} rad"
    )


def scalar_statistics(values: Iterable[Any]) -> dict[str, Any]:
    arr = _finite(values)
    if arr.size == 0:
        return {
            "count": 0, "mean": None, "median": None, "std": None, "mad": None,
            "min": None, "max": None, "range": None, "p95": None,
        }
    median = float(np.median(arr))
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": median,
        "std": float(np.std(arr)),
        "mad": float(np.median(np.abs(arr - median))),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.ptp(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def make_height_schedule(start_mm: float, stop_mm: float, step_mm: float) -> tuple[float, ...]:
    start = float(start_mm)
    stop = float(stop_mm)
    step = float(step_mm)
    if not all(math.isfinite(v) for v in (start, stop, step)):
        raise ValueError("高度起点、终点和步长必须是有限数")
    if start <= 0 or stop < start or step <= 0:
        raise ValueError("高度范围必须满足0 < start <= stop且step > 0")
    count = int(math.floor((stop - start) / step + 1e-9))
    values = [start + index * step for index in range(count + 1)]
    if not math.isclose(values[-1], stop, abs_tol=1e-8):
        values.append(stop)
    return tuple(float(round(value, 9)) for value in values)


def rgb_precision_targets(
    intrinsics: Any,
    height_mm: float,
    center_step_mm: float,
) -> tuple[tuple[str, float, float], ...]:
    def parameter(name: str) -> float:
        if isinstance(intrinsics, dict):
            return float(intrinsics[name])
        return float(getattr(intrinsics, name))

    fx, fy = parameter("fx"), parameter("fy")
    cx, cy = parameter("cx"), parameter("cy")
    z = float(height_mm)
    step = float(center_step_mm)
    if min(fx, fy, z, step) <= 0:
        raise ValueError("内参、目标高度和中心位移必须大于0")
    du = fx * step / z
    dv = fy * step / z
    return (
        ("CENTER", cx, cy),
        ("LEFT", cx - du, cy),
        ("RIGHT", cx + du, cy),
        ("UP", cx, cy - dv),
        ("DOWN", cx, cy + dv),
    )


def fit_tcp_rgb_geometry(pose_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        row for row in pose_rows
        if row.get("tcp_xyz_median_mm") is not None and row.get("rgb_xyz_median_mm") is not None
    ]
    if len(rows) < 3:
        return {
            "pose_count": len(rows), "pair_count": 0,
            "distance_error_mm": scalar_statistics([]),
            "rigid_residual_mm": scalar_statistics([]),
            "rigid_rms_mm": None, "scale": None, "scale_error_percent": None,
        }
    tcp = np.asarray([row["tcp_xyz_median_mm"] for row in rows], dtype=np.float64)
    rgb = np.asarray([row["rgb_xyz_median_mm"] for row in rows], dtype=np.float64)
    tcp_centered = tcp - np.mean(tcp, axis=0)
    rgb_centered = rgb - np.mean(rgb, axis=0)
    u, singular_values, vt = np.linalg.svd(tcp_centered.T @ rgb_centered)
    rotation = u @ vt  # Eye-in-hand translation mapping may legitimately have det=-1.
    predicted = tcp_centered @ rotation + np.mean(rgb, axis=0)
    rigid_residuals = np.linalg.norm(predicted - rgb, axis=1)
    denominator = float(np.sum(tcp_centered * tcp_centered))
    scale = None if denominator <= 1e-12 else float(np.sum(singular_values) / denominator)
    pair_errors: list[float] = []
    for first in range(len(rows)):
        for second in range(first + 1, len(rows)):
            tcp_distance = float(np.linalg.norm(tcp[first] - tcp[second]))
            rgb_distance = float(np.linalg.norm(rgb[first] - rgb[second]))
            pair_errors.append(abs(rgb_distance - tcp_distance))
    return {
        "pose_count": len(rows),
        "pair_count": len(pair_errors),
        "distance_error_mm": scalar_statistics(pair_errors),
        "rigid_residual_mm": scalar_statistics(rigid_residuals),
        "rigid_rms_mm": float(np.sqrt(np.mean(rigid_residuals ** 2))),
        "scale": scale,
        "scale_error_percent": None if scale is None else float((scale - 1.0) * 100.0),
        "rotation_determinant": float(np.linalg.det(rotation)),
    }


def select_contiguous_height_ranges(
    height_summaries: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    rows = sorted(height_summaries, key=lambda row: float(row["height_target_mm"]))
    passing_ranges: list[list[float]] = []
    current: list[float] = []
    for row in rows:
        height = float(row["height_target_mm"])
        if bool(row.get("passed")):
            current.append(height)
        elif current:
            passing_ranges.append(current)
            current = []
    if current:
        passing_ranges.append(current)
    candidates = [
        row for row in rows
        if bool(row.get("passed")) and row.get("distance_error_p95_mm") is not None
    ]
    best = min(
        candidates,
        key=lambda row: (
            float(row["distance_error_p95_mm"]),
            float(row.get("rigid_rms_mm") or math.inf),
            float(row.get("rgb_reprojection_rmse_p95_px") or math.inf),
        ),
        default=None,
    )
    recommended = None
    if best is not None:
        best_height = float(best["height_target_mm"])
        recommended = next(
            (
                {"start_mm": band[0], "stop_mm": band[-1], "heights_mm": band}
                for band in passing_ranges if best_height in band
            ),
            None,
        )
    return {
        "passing_ranges": [
            {"start_mm": band[0], "stop_mm": band[-1], "heights_mm": band}
            for band in passing_ranges
        ],
        "recommended_range": recommended,
        "best_height_mm": None if best is None else float(best["height_target_mm"]),
    }


def normal_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(3)
    bv = np.asarray(b, dtype=np.float64).reshape(3)
    if not np.isfinite(av).all() or not np.isfinite(bv).all():
        return float("nan")
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-12:
        return float("nan")
    # 板法向的正负号在两个拟合器之间没有物理区别。
    cosine = abs(float(av @ bv) / denom)
    return float(math.degrees(math.acos(np.clip(cosine, -1.0, 1.0))))


def transform_cross_metrics(
    T_rgb_board: np.ndarray | None,
    T_depth_board: np.ndarray | None,
) -> dict[str, float | None]:
    if T_rgb_board is None or T_depth_board is None:
        return {
            "cross_dx_mm": None, "cross_dy_mm": None, "cross_dz_mm": None,
            "cross_distance_mm": None, "cross_normal_angle_deg": None,
        }
    rgb = np.asarray(T_rgb_board, dtype=np.float64).reshape(4, 4)
    depth = np.asarray(T_depth_board, dtype=np.float64).reshape(4, 4)
    delta = depth[:3, 3] - rgb[:3, 3]
    return {
        "cross_dx_mm": float(delta[0]),
        "cross_dy_mm": float(delta[1]),
        "cross_dz_mm": float(delta[2]),
        "cross_distance_mm": float(np.linalg.norm(delta)),
        "cross_normal_angle_deg": normal_angle_deg(rgb[:3, 2], depth[:3, 2]),
    }


def _transform_values(prefix: str, transform: np.ndarray | None) -> dict[str, Any]:
    if transform is None:
        return {
            f"{prefix}_x_mm": None, f"{prefix}_y_mm": None, f"{prefix}_z_mm": None,
            f"{prefix}_normal_x": None, f"{prefix}_normal_y": None,
            f"{prefix}_normal_z": None,
        }
    T = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return {
        f"{prefix}_x_mm": float(T[0, 3]),
        f"{prefix}_y_mm": float(T[1, 3]),
        f"{prefix}_z_mm": float(T[2, 3]),
        f"{prefix}_normal_x": float(T[0, 2]),
        f"{prefix}_normal_y": float(T[1, 2]),
        f"{prefix}_normal_z": float(T[2, 2]),
    }


def build_frame_record(
    run_id: str,
    target_height_mm: float,
    frame_index: int,
    batch_size: int,
    bundle: CameraFrameBundle,
    rgb: BoardPoseResult,
    depth: BoardPoseResult,
    robot_snapshot: dict[str, Any],
    grid_position: str = "",
    grid_target_uv: tuple[float, float] | None = None,
) -> dict[str, Any]:
    quality = evaluate_image_quality(bundle.color_bgr, rgb)
    reasons: list[str] = []
    if not rgb.ok or rgb.T_rgb_board is None:
        reasons.append(f"rgb:{rgb.status}")
    if not depth.ok or depth.T_pointcloud_board is None:
        reasons.append(f"depth:{depth.status}")
    tcp = [float(v) for v in robot_snapshot.get("tcp_pose_m_rad", [float("nan")] * 6)[:6]]
    tcp_mm_rad = [v * 1000.0 for v in tcp[:3]] + tcp[3:6]
    metadata = bundle.metadata_dict()
    image_points = np.asarray(rgb.image_points or [], dtype=np.float64).reshape(-1, 2)
    if image_points.size and np.isfinite(image_points).all():
        image_center = np.mean(image_points, axis=0)
        center_u, center_v = float(image_center[0]), float(image_center[1])
    else:
        center_u, center_v = float("nan"), float("nan")
    target_u = None if grid_target_uv is None else float(grid_target_uv[0])
    target_v = None if grid_target_uv is None else float(grid_target_uv[1])
    grid_row = int(grid_position[1]) if len(grid_position) == 4 and grid_position.startswith("R") else None
    grid_col = int(grid_position[3]) if len(grid_position) == 4 and grid_position.startswith("R") else None
    record: dict[str, Any] = {
        "run_id": run_id,
        "height_target_mm": float(target_height_mm),
        "frame_index": int(frame_index),
        "batch_index": int((frame_index - 1) // batch_size + 1),
        "batch_frame_index": int((frame_index - 1) % batch_size + 1),
        "captured_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "valid": not reasons,
        "failure_reasons": "|".join(reasons),
        "color_timestamp_us": metadata.get("color_timestamp_us"),
        "depth_timestamp_us": metadata.get("depth_timestamp_us"),
        "rgb_depth_timestamp_delta_us": metadata.get("rgb_depth_timestamp_delta_us"),
        "color_frame_index": metadata.get("color_frame_index"),
        "depth_frame_index": metadata.get("depth_frame_index"),
        "rgb_ok": bool(rgb.ok),
        "rgb_status": rgb.status,
        "rgb_charuco_count": int(rgb.charuco_count),
        "rgb_pnp_inlier_count": int(rgb.rgb_pnp_inlier_count),
        "rgb_reprojection_rmse_px": _finite_or_none(rgb.rgb_reprojection_rmse_px),
        "rgb_reprojection_max_px": _finite_or_none(rgb.rgb_reprojection_max_px),
        "rgb_board_area_ratio": _finite_or_none(quality.board_area_ratio),
        "rgb_center_offset_ratio": _finite_or_none(quality.center_offset_ratio),
        "grid_position": grid_position,
        "grid_row": grid_row,
        "grid_col": grid_col,
        "grid_target_u_px": target_u,
        "grid_target_v_px": target_v,
        "rgb_center_u_px": _finite_or_none(center_u),
        "rgb_center_v_px": _finite_or_none(center_v),
        "rgb_center_u_norm": _finite_or_none(center_u / max(1, bundle.color_bgr.shape[1])),
        "rgb_center_v_norm": _finite_or_none(center_v / max(1, bundle.color_bgr.shape[0])),
        "depth_ok": bool(depth.ok),
        "depth_status": depth.status,
        "depth_valid_3d_count": int(depth.valid_3d_count),
        "depth_corner_rmse_mm": _finite_or_none(depth.corner_rmse_mm),
        "depth_corner_max_error_mm": _finite_or_none(depth.corner_max_error_mm),
        "depth_plane_rmse_mm": _finite_or_none(depth.plane_rmse_mm),
        "depth_plane_inlier_count": int(depth.plane_inlier_count),
        "depth_board_mask_point_count": int(depth.board_mask_point_count),
        "control_residual_mm": (
            None if rgb.T_rgb_board is None
            else float(np.asarray(rgb.T_rgb_board)[2, 3] - target_height_mm)
        ),
        "brightness": _finite_or_none(quality.brightness),
        "contrast": _finite_or_none(quality.contrast),
        "sharpness": _finite_or_none(quality.sharpness),
        "tcp_x_mm": _finite_or_none(tcp_mm_rad[0]),
        "tcp_y_mm": _finite_or_none(tcp_mm_rad[1]),
        "tcp_z_mm": _finite_or_none(tcp_mm_rad[2]),
        "tcp_rx_rad": _finite_or_none(tcp_mm_rad[3]),
        "tcp_ry_rad": _finite_or_none(tcp_mm_rad[4]),
        "tcp_rz_rad": _finite_or_none(tcp_mm_rad[5]),
        "tcp_power_on": bool(robot_snapshot.get("power_on")),
        "tcp_steady": bool(robot_snapshot.get("steady")),
        "tcp_collision": bool(robot_snapshot.get("collision")),
    }
    record.update(_transform_values("rgb", rgb.T_rgb_board))
    record.update(_transform_values("depth", depth.T_pointcloud_board))
    record.update(transform_cross_metrics(rgb.T_rgb_board, depth.T_pointcloud_board))
    return record


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean_transform_rows(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    xyz = np.asarray([
        [row.get(f"{prefix}_x_mm"), row.get(f"{prefix}_y_mm"), row.get(f"{prefix}_z_mm")]
        for row in rows
        if all(row.get(f"{prefix}_{axis}_mm") is not None for axis in ("x", "y", "z"))
    ], dtype=np.float64)
    if xyz.size == 0:
        return {
            "count": 0, "xyz_median_mm": None, "xyz_scatter_p95_mm": None,
            "normal_median": None, "normal_scatter_p95_deg": None,
        }
    center = np.median(xyz, axis=0)
    distances = np.linalg.norm(xyz - center, axis=1)
    result = {
        "count": int(len(xyz)),
        "xyz_median_mm": center.astype(float).tolist(),
        "xyz_axis_std_mm": np.std(xyz, axis=0).astype(float).tolist(),
        "xyz_scatter_p95_mm": float(np.percentile(distances, 95)),
        "xyz_scatter_max_mm": float(np.max(distances)),
    }
    normals = np.asarray([
        [row.get(f"{prefix}_normal_x"), row.get(f"{prefix}_normal_y"), row.get(f"{prefix}_normal_z")]
        for row in rows
        if all(row.get(f"{prefix}_normal_{axis}") is not None for axis in ("x", "y", "z"))
    ], dtype=np.float64)
    if normals.size == 0:
        result.update({
            "normal_median": None,
            "normal_scatter_p95_deg": None,
            "normal_scatter_max_deg": None,
        })
        return result
    # 统一法向符号后再计算中心方向，避免同一平面的 ±n 抵消。
    reference = normals[0] / max(float(np.linalg.norm(normals[0])), 1e-12)
    aligned = np.asarray([
        normal if float(normal @ reference) >= 0.0 else -normal
        for normal in normals
    ], dtype=np.float64)
    mean_normal = np.mean(aligned, axis=0)
    mean_normal /= max(float(np.linalg.norm(mean_normal)), 1e-12)
    angular_errors = np.asarray(
        [normal_angle_deg(normal, mean_normal) for normal in aligned], dtype=np.float64,
    )
    result.update({
        "normal_median": mean_normal.astype(float).tolist(),
        "normal_scatter_p95_deg": float(np.percentile(angular_errors, 95)),
        "normal_scatter_max_deg": float(np.max(angular_errors)),
    })
    return result


def _video_profile_metadata(profile: Any) -> dict[str, Any]:
    try:
        video = profile.as_video_stream_profile()
    except Exception:
        video = profile
    result: dict[str, Any] = {}
    for key, method in (
        ("width", "get_width"), ("height", "get_height"), ("fps", "get_fps"),
        ("format", "get_format"), ("stream_type", "get_type"),
    ):
        try:
            value = getattr(video, method)()
            result[key] = str(value) if key in {"format", "stream_type"} else int(value)
        except Exception as exc:
            result[f"{key}_error"] = str(exc)
    return result


def summarize_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"frame_count": 0, "valid_count": 0, "valid_ratio": 0.0}
    return {
        "height_target_mm": float(rows[0]["height_target_mm"]),
        "batch_index": int(rows[0]["batch_index"]),
        "frame_count": len(rows),
        "valid_count": sum(bool(row.get("valid")) for row in rows),
        "valid_ratio": float(np.mean([bool(row.get("valid")) for row in rows])),
        "rgb": _mean_transform_rows(rows, "rgb"),
        "depth": _mean_transform_rows(rows, "depth"),
        "rgb_reprojection_rmse_px": scalar_statistics(
            row.get("rgb_reprojection_rmse_px") for row in rows
        ),
        "depth_plane_rmse_mm": scalar_statistics(row.get("depth_plane_rmse_mm") for row in rows),
        "cross_dz_mm": scalar_statistics(row.get("cross_dz_mm") for row in rows),
        "cross_distance_mm": scalar_statistics(row.get("cross_distance_mm") for row in rows),
        "control_residual_mm": scalar_statistics(row.get("control_residual_mm") for row in rows),
    }


def build_batch_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (float(row["height_target_mm"]), int(row["batch_index"]))
        groups.setdefault(key, []).append(row)
    return [summarize_batch(groups[key]) for key in sorted(groups)]


def summarize_height(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"frame_count": 0, "valid_count": 0, "valid_ratio": 0.0}
    rgb_gate = [
        bool(row.get("rgb_ok"))
        and _at_most(row.get("rgb_reprojection_rmse_px"), BOARD_CFG.max_rgb_reprojection_rmse_px)
        and _at_most(row.get("rgb_reprojection_max_px"), BOARD_CFG.max_rgb_reprojection_error_px)
        for row in rows
    ]
    depth_gate = [
        bool(row.get("depth_ok"))
        and _at_most(row.get("depth_plane_rmse_mm"), BOARD_CFG.max_plane_rmse_mm)
        for row in rows
    ]
    numeric_fields = (
        "rgb_reprojection_rmse_px", "rgb_reprojection_max_px",
        "rgb_x_mm", "rgb_y_mm", "rgb_z_mm",
        "depth_corner_rmse_mm", "depth_corner_max_error_mm", "depth_plane_rmse_mm",
        "depth_x_mm", "depth_y_mm", "depth_z_mm",
        "cross_dx_mm", "cross_dy_mm", "cross_dz_mm", "cross_distance_mm",
        "cross_normal_angle_deg", "control_residual_mm",
        "brightness", "contrast", "sharpness",
        "rgb_depth_timestamp_delta_us", "tcp_x_mm", "tcp_y_mm", "tcp_z_mm",
        "tcp_rx_rad", "tcp_ry_rad", "tcp_rz_rad",
    )
    result = {
        "height_target_mm": float(rows[0]["height_target_mm"]),
        "frame_count": len(rows),
        "valid_count": sum(bool(row.get("valid")) for row in rows),
        "valid_ratio": float(np.mean([bool(row.get("valid")) for row in rows])),
        "rgb_gate_pass_ratio": float(np.mean(rgb_gate)),
        "depth_gate_pass_ratio": float(np.mean(depth_gate)),
        "rgb_pose_repeatability": _mean_transform_rows(rows, "rgb"),
        "depth_pose_repeatability": _mean_transform_rows(rows, "depth"),
        "statistics": {
            field: scalar_statistics(row.get(field) for row in rows)
            for field in numeric_fields
        },
    }
    return result


def _at_most(value: Any, limit: float) -> bool:
    number = _finite_or_none(value)
    return number is not None and number <= float(limit)


def build_height_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(float(row["height_target_mm"]), []).append(row)
    return [summarize_height(groups[key]) for key in sorted(groups)]


def build_field_position_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按高度和3x3网格位置汇总，供后续生成视野误差热力图。"""
    groups: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for row in rows:
        position = str(row.get("grid_position", "")).strip()
        if position:
            groups.setdefault((float(row["height_target_mm"]), position), []).append(row)
    result: list[dict[str, Any]] = []
    metrics = (
        "rgb_center_u_px", "rgb_center_v_px", "rgb_x_mm", "rgb_y_mm", "rgb_z_mm",
        "rgb_reprojection_rmse_px", "rgb_reprojection_max_px",
        "depth_x_mm", "depth_y_mm", "depth_z_mm", "depth_plane_rmse_mm",
        "cross_dx_mm", "cross_dy_mm", "cross_dz_mm", "cross_distance_mm",
    )
    for key in sorted(groups):
        group = groups[key]
        result.append({
            "height_target_mm": key[0],
            "grid_position": key[1],
            "grid_row": group[0].get("grid_row"),
            "grid_col": group[0].get("grid_col"),
            "grid_target_u_px": group[0].get("grid_target_u_px"),
            "grid_target_v_px": group[0].get("grid_target_v_px"),
            "frame_count": len(group),
            "valid_count": sum(bool(row.get("valid")) for row in group),
            "valid_ratio": float(np.mean([bool(row.get("valid")) for row in group])),
            "statistics": {
                metric: scalar_statistics(row.get(metric) for row in group)
                for metric in metrics
            },
            "rgb_pose_repeatability": _mean_transform_rows(group, "rgb"),
            "depth_pose_repeatability": _mean_transform_rows(group, "depth"),
        })
    return result


def trend_summary(height_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "rgb_reprojection_rmse_px", "rgb_reprojection_max_px", "depth_plane_rmse_mm",
        "cross_dz_mm", "cross_distance_mm", "cross_normal_angle_deg",
    )
    result: dict[str, Any] = {}
    for metric in metrics:
        pairs = []
        for summary in height_summaries:
            median = summary.get("statistics", {}).get(metric, {}).get("median")
            if median is not None:
                pairs.append((float(summary["height_target_mm"]), float(median)))
        if len(pairs) < 2:
            result[metric] = {"count": len(pairs), "slope_per_100mm": None, "worst_height_mm": None}
            continue
        x = np.asarray([item[0] for item in pairs], dtype=np.float64)
        y = np.asarray([item[1] for item in pairs], dtype=np.float64)
        slope, intercept = np.polyfit(x, y, 1)
        worst_index = int(np.argmax(np.abs(y))) if metric in {"cross_dz_mm"} else int(np.argmax(y))
        result[metric] = {
            "count": len(pairs),
            "slope_per_100mm": float(slope * 100.0),
            "intercept": float(intercept),
            "worst_height_mm": float(x[worst_index]),
            "worst_median": float(y[worst_index]),
        }
    return result


def _flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "height_target_mm": summary.get("height_target_mm"),
        "frame_count": summary.get("frame_count"),
        "valid_count": summary.get("valid_count"),
        "valid_ratio": summary.get("valid_ratio"),
        "rgb_gate_pass_ratio": summary.get("rgb_gate_pass_ratio"),
        "depth_gate_pass_ratio": summary.get("depth_gate_pass_ratio"),
    }
    for metric, stats in summary.get("statistics", {}).items():
        for name in ("mean", "median", "std", "mad", "range", "p95"):
            flat[f"{metric}_{name}"] = stats.get(name)
    return flat


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        field_list = sorted({key for row in rows for key in row})
    else:
        field_list = list(fields)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_list, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_plots(output_dir: Path, height_summaries: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[str]:
    if not height_summaries:
        return ["plots_skipped:no_height_data"]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"plots_skipped:{exc}"]
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    heights = [float(item["height_target_mm"]) for item in height_summaries]
    metrics = [
        ("rgb_reprojection_rmse_px", "RGB reprojection RMSE (px)"),
        ("depth_plane_rmse_mm", "Depth plane RMSE (mm)"),
        ("cross_dz_mm", "Depth - RGB board Z (mm)"),
        ("cross_distance_mm", "Depth - RGB center distance (mm)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        medians = [item["statistics"][metric]["median"] for item in height_summaries]
        p95s = [item["statistics"][metric]["p95"] for item in height_summaries]
        axis.plot(heights, medians, "o-", label="median")
        axis.plot(heights, p95s, "s--", label="p95")
        axis.set_xlabel("PnP target Z (mm)")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    trend_path = plot_dir / "height_error_trends.png"
    fig.savefig(trend_path, dpi=150)
    plt.close(fig)
    saved.append(str(trend_path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, (metric, label) in zip(axes, metrics[2:]):
        grouped = [
            _finite(row.get(metric) for row in rows if float(row["height_target_mm"]) == height)
            for height in heights
        ]
        axis.boxplot(grouped, tick_labels=[f"{height:.0f}" for height in heights], showfliers=False)
        axis.set_xlabel("PnP target Z (mm)")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    box_path = plot_dir / "cross_error_boxplots.png"
    fig.savefig(box_path, dpi=150)
    plt.close(fig)
    saved.append(str(box_path))

    fig, axis = plt.subplots(figsize=(8, 5))
    valid = [float(item["valid_ratio"]) for item in height_summaries]
    rgb_pass = [float(item["rgb_gate_pass_ratio"]) for item in height_summaries]
    depth_pass = [float(item["depth_gate_pass_ratio"]) for item in height_summaries]
    x = np.arange(len(heights))
    width = 0.25
    axis.bar(x - width, valid, width, label="both valid")
    axis.bar(x, rgb_pass, width, label="RGB gate")
    axis.bar(x + width, depth_pass, width, label="Depth gate")
    axis.set_xticks(x, [f"{height:.0f}" for height in heights])
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("PnP target Z (mm)")
    axis.set_ylabel("pass ratio")
    axis.legend()
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    pass_path = plot_dir / "height_pass_ratios.png"
    fig.savefig(pass_path, dpi=150)
    plt.close(fig)
    saved.append(str(pass_path))
    return saved


def write_field_grid_plots(output_dir: Path, summaries: list[dict[str, Any]]) -> list[str]:
    if not summaries:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"field_plots_skipped:{exc}"]
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    heights = sorted({float(item["height_target_mm"]) for item in summaries})
    metrics = (
        ("rgb_reprojection_rmse_px", "RGB reprojection RMSE (px)"),
        ("cross_dx_mm", "Depth - RGB X (mm)"),
        ("cross_dy_mm", "Depth - RGB Y (mm)"),
        ("cross_dz_mm", "Depth - RGB Z (mm)"),
    )
    saved: list[str] = []
    for metric, title in metrics:
        columns = 2
        rows_count = int(math.ceil(len(heights) / columns))
        fig, axes = plt.subplots(rows_count, columns, figsize=(10, 4.5 * rows_count), squeeze=False)
        finite_values = [
            item["statistics"][metric]["median"]
            for item in summaries
            if item["statistics"][metric]["median"] is not None
        ]
        vmin = min(finite_values) if finite_values else 0.0
        vmax = max(finite_values) if finite_values else 1.0
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-9
        image = None
        for axis, height in zip(axes.ravel(), heights):
            matrix = np.full((3, 3), np.nan, dtype=np.float64)
            for item in summaries:
                if float(item["height_target_mm"]) != height:
                    continue
                value = item["statistics"][metric]["median"]
                if value is not None:
                    matrix[int(item["grid_row"]) - 1, int(item["grid_col"]) - 1] = float(value)
            image = axis.imshow(matrix, cmap="coolwarm", vmin=vmin, vmax=vmax)
            for row in range(3):
                for col in range(3):
                    text_value = matrix[row, col]
                    axis.text(
                        col, row, "NA" if not np.isfinite(text_value) else f"{text_value:.3f}",
                        ha="center", va="center", color="black", fontsize=10,
                    )
            axis.set_title(f"{height:.0f} mm")
            axis.set_xticks(range(3), ["left", "center", "right"])
            axis.set_yticks(range(3), ["top", "middle", "bottom"])
        for axis in axes.ravel()[len(heights):]:
            axis.axis("off")
        fig.suptitle(title)
        if image is not None:
            fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75)
        path = plot_dir / f"field_{metric}_heatmaps.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))
    return saved


def read_and_lock_color_exposure(pipeline: Any, warmup_frames: int, frame_reader: Any) -> dict[str, Any]:
    """自动曝光预热后读取当前值并锁定；不支持的SDK会明确返回失败。"""
    required = int(warmup_frames)
    valid_count = 0
    attempt_count = 0
    max_attempts = max(required * 4, required + 30, 30)
    while valid_count < required and attempt_count < max_attempts:
        attempt_count += 1
        if frame_reader() is None:
            if attempt_count <= 5 or attempt_count % 20 == 0:
                print(
                    f"[WARN] 曝光预热暂未取得完整RGB-D帧："
                    f"attempt={attempt_count}, valid={valid_count}/{required}，继续重试"
                )
            time.sleep(0.03)
            continue
        valid_count += 1
    if valid_count < required:
        raise RuntimeError(
            f"曝光预热失败：尝试{attempt_count}次，仅取得{valid_count}/{required}个有效RGB-D帧"
        )
    result: dict[str, Any] = {
        "warmup_frames": required,
        "warmup_attempts": attempt_count,
        "warmup_dropped_frames": attempt_count - valid_count,
        "locked": False,
        "errors": [],
    }
    try:
        from pyorbbecsdk import OBPropertyID
        device = pipeline.get_device()
        exposure_prop = OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT
        gain_prop = OBPropertyID.OB_PROP_COLOR_GAIN_INT
        auto_prop = OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL
        exposure = int(device.get_int_property(exposure_prop))
        gain = int(device.get_int_property(gain_prop))
        device.set_bool_property(auto_prop, False)
        device.set_int_property(exposure_prop, exposure)
        device.set_int_property(gain_prop, gain)
        result.update({
            "locked": True,
            "auto_exposure": bool(device.get_bool_property(auto_prop)),
            "exposure": int(device.get_int_property(exposure_prop)),
            "gain": int(device.get_int_property(gain_prop)),
        })
    except Exception as exc:
        result["errors"].append(str(exc))
    return result


def _rgb_image_center(
    rgb: BoardPoseResult,
    intrinsics: Any | None = None,
) -> tuple[float, float] | None:
    if rgb.T_rgb_board is not None and intrinsics is not None:
        transform = np.asarray(rgb.T_rgb_board, dtype=np.float64).reshape(4, 4)
        rotation_vector, _ = cv2.Rodrigues(transform[:3, :3])
        object_center = np.asarray([[
            BOARD_CFG.pattern_width_mm * 0.5,
            BOARD_CFG.pattern_height_mm * 0.5,
            0.0,
        ]], dtype=np.float64)
        distortion = np.asarray(getattr(intrinsics, "distortion", ()), dtype=np.float64)
        projected, _ = cv2.projectPoints(
            object_center,
            rotation_vector,
            transform[:3, 3],
            intrinsics.camera_matrix(),
            distortion,
        )
        center = np.asarray(projected, dtype=np.float64).reshape(-1, 2)[0]
        if np.isfinite(center).all():
            return float(center[0]), float(center[1])
    image_points = np.asarray(rgb.image_points or [], dtype=np.float64).reshape(-1, 2)
    if not image_points.size or not np.isfinite(image_points).all():
        return None
    center = np.mean(image_points, axis=0)
    return float(center[0]), float(center[1])


def build_rgb_precision_frame_record(
    run_id: str,
    height_mm: float,
    pose_name: str,
    pose_index: int,
    frame_index: int,
    bundle: RgbFrameBundle,
    rgb: BoardPoseResult,
    robot_snapshot: dict[str, Any],
    target_uv: tuple[float, float] | None,
) -> dict[str, Any]:
    quality = evaluate_image_quality(bundle.color_bgr, rgb)
    valid = bool(rgb.ok and rgb.T_rgb_board is not None)
    center = _rgb_image_center(rgb, bundle.intrinsics)
    target_distance = (
        None if center is None or target_uv is None
        else float(math.hypot(center[0] - target_uv[0], center[1] - target_uv[1]))
    )
    tcp = [float(v) for v in robot_snapshot.get("tcp_pose_m_rad", [float("nan")] * 6)[:6]]
    tcp_mm_rad = [value * 1000.0 for value in tcp[:3]] + tcp[3:]
    record = {
        "run_id": run_id,
        "height_target_mm": float(height_mm),
        "pose_name": pose_name,
        "pose_index": int(pose_index),
        "frame_index": int(frame_index),
        "captured_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "valid": valid,
        "failure_reasons": "" if valid else f"rgb:{rgb.status}",
        "color_timestamp_us": bundle.color_timestamp_us,
        "color_frame_index": bundle.color_frame_index,
        "rgb_status": rgb.status,
        "rgb_charuco_count": int(rgb.charuco_count),
        "rgb_pnp_inlier_count": int(rgb.rgb_pnp_inlier_count),
        "rgb_reprojection_rmse_px": _finite_or_none(rgb.rgb_reprojection_rmse_px),
        "rgb_reprojection_max_px": _finite_or_none(rgb.rgb_reprojection_max_px),
        "rgb_center_u_px": None if center is None else center[0],
        "rgb_center_v_px": None if center is None else center[1],
        "target_u_px": None if target_uv is None else float(target_uv[0]),
        "target_v_px": None if target_uv is None else float(target_uv[1]),
        "target_distance_px": target_distance,
        "brightness": _finite_or_none(quality.brightness),
        "contrast": _finite_or_none(quality.contrast),
        "sharpness": _finite_or_none(quality.sharpness),
        "tcp_x_mm": _finite_or_none(tcp_mm_rad[0]),
        "tcp_y_mm": _finite_or_none(tcp_mm_rad[1]),
        "tcp_z_mm": _finite_or_none(tcp_mm_rad[2]),
        "tcp_rx_rad": _finite_or_none(tcp_mm_rad[3]),
        "tcp_ry_rad": _finite_or_none(tcp_mm_rad[4]),
        "tcp_rz_rad": _finite_or_none(tcp_mm_rad[5]),
        "tcp_power_on": bool(robot_snapshot.get("power_on")),
        "tcp_steady": bool(robot_snapshot.get("steady")),
        "tcp_collision": bool(robot_snapshot.get("collision")),
    }
    record.update(_transform_values("rgb", rgb.T_rgb_board))
    return record


def build_rgb_pose_summaries(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (float(row["height_target_mm"]), str(row["pose_name"])), [],
        ).append(row)
    summaries: list[dict[str, Any]] = []
    for (height, pose_name), group in sorted(grouped.items()):
        valid = [
            row for row in group
            if bool(row.get("valid"))
            and all(row.get(f"rgb_{axis}_mm") is not None for axis in ("x", "y", "z"))
        ]
        rgb_xyz = np.asarray(
            [[row[f"rgb_{axis}_mm"] for axis in ("x", "y", "z")] for row in valid],
            dtype=np.float64,
        )
        tcp_xyz = np.asarray(
            [[row[f"tcp_{axis}_mm"] for axis in ("x", "y", "z")] for row in valid],
            dtype=np.float64,
        )
        rgb_median = None
        tcp_median = None
        scatter = scalar_statistics([])
        if rgb_xyz.size:
            rgb_center = np.median(rgb_xyz, axis=0)
            tcp_center = np.median(tcp_xyz, axis=0)
            rgb_median = [float(value) for value in rgb_center]
            tcp_median = [float(value) for value in tcp_center]
            scatter = scalar_statistics(np.linalg.norm(rgb_xyz - rgb_center, axis=1))
        summaries.append({
            "height_target_mm": height,
            "pose_name": pose_name,
            "pose_index": int(group[0]["pose_index"]),
            "frame_count": len(group),
            "valid_count": len(valid),
            "valid_ratio": len(valid) / len(group) if group else 0.0,
            "rgb_xyz_median_mm": rgb_median,
            "tcp_xyz_median_mm": tcp_median,
            "rgb_scatter_mm": scatter,
            "rgb_reprojection_rmse_px": scalar_statistics(
                row.get("rgb_reprojection_rmse_px") for row in valid
            ),
            "rgb_reprojection_max_px": scalar_statistics(
                row.get("rgb_reprojection_max_px") for row in valid
            ),
            "rgb_charuco_count": scalar_statistics(row.get("rgb_charuco_count") for row in group),
            "rgb_pnp_inlier_count": scalar_statistics(
                row.get("rgb_pnp_inlier_count") for row in group
            ),
            "target_distance_px": scalar_statistics(row.get("target_distance_px") for row in valid),
        })
    return summaries


def build_rgb_height_summaries(
    rows: Iterable[dict[str, Any]],
    pose_summaries: Iterable[dict[str, Any]],
    config: RgbPrecisionRangeConfig,
) -> list[dict[str, Any]]:
    frame_rows = list(rows)
    pose_rows = list(pose_summaries)
    summaries: list[dict[str, Any]] = []
    for height in sorted({float(row["height_target_mm"]) for row in frame_rows}):
        height_frames = [row for row in frame_rows if float(row["height_target_mm"]) == height]
        height_poses = [row for row in pose_rows if float(row["height_target_mm"]) == height]
        geometry = fit_tcp_rgb_geometry(height_poses)
        valid_count = sum(bool(row.get("valid")) for row in height_frames)
        valid_ratio = valid_count / len(height_frames) if height_frames else 0.0
        reprojection_rmse = scalar_statistics(
            row.get("rgb_reprojection_rmse_px")
            for row in height_frames if bool(row.get("valid"))
        )
        reprojection_max = scalar_statistics(
            row.get("rgb_reprojection_max_px")
            for row in height_frames if bool(row.get("valid"))
        )
        distance_p95 = geometry["distance_error_mm"].get("p95")
        failure_reasons: list[str] = []
        if len(height_poses) != len(RGB_PRECISION_POSE_NAMES):
            failure_reasons.append("five_poses_incomplete")
        if valid_ratio < config.minimum_valid_ratio:
            failure_reasons.append("rgb_valid_ratio_low")
        if (
            reprojection_rmse.get("p95") is None
            or reprojection_rmse["p95"] > BOARD_CFG.max_rgb_reprojection_rmse_px
        ):
            failure_reasons.append("rgb_reprojection_rmse_high")
        if (
            reprojection_max.get("p95") is None
            or reprojection_max["p95"] > BOARD_CFG.max_rgb_reprojection_error_px
        ):
            failure_reasons.append("rgb_reprojection_max_high")
        if distance_p95 is None or distance_p95 > config.maximum_distance_p95_mm:
            failure_reasons.append("tcp_rgb_distance_p95_high")
        summaries.append({
            "height_target_mm": height,
            "frame_count": len(height_frames),
            "valid_count": valid_count,
            "valid_ratio": valid_ratio,
            "pose_count": len(height_poses),
            "pair_count": geometry["pair_count"],
            "distance_error_median_mm": geometry["distance_error_mm"].get("median"),
            "distance_error_rms_mm": (
                None if geometry["distance_error_mm"].get("count", 0) == 0
                else float(np.sqrt(np.mean(
                    np.square(_finite(
                        [
                            abs(
                                np.linalg.norm(
                                    np.asarray(first["rgb_xyz_median_mm"], dtype=np.float64)
                                    - np.asarray(second["rgb_xyz_median_mm"], dtype=np.float64)
                                )
                                - np.linalg.norm(
                                    np.asarray(first["tcp_xyz_median_mm"], dtype=np.float64)
                                    - np.asarray(second["tcp_xyz_median_mm"], dtype=np.float64)
                                )
                            )
                            for index, first in enumerate(height_poses)
                            for second in height_poses[index + 1:]
                            if first.get("rgb_xyz_median_mm") is not None
                            and second.get("rgb_xyz_median_mm") is not None
                        ]
                    ))
                )))
            ),
            "distance_error_p95_mm": distance_p95,
            "distance_error_max_mm": geometry["distance_error_mm"].get("max"),
            "rigid_rms_mm": geometry["rigid_rms_mm"],
            "rigid_residual_p95_mm": geometry["rigid_residual_mm"].get("p95"),
            "rigid_residual_max_mm": geometry["rigid_residual_mm"].get("max"),
            "scale": geometry["scale"],
            "scale_error_percent": geometry["scale_error_percent"],
            "rgb_reprojection_rmse_median_px": reprojection_rmse.get("median"),
            "rgb_reprojection_rmse_p95_px": reprojection_rmse.get("p95"),
            "rgb_reprojection_max_p95_px": reprojection_max.get("p95"),
            "rgb_scatter_p95_worst_mm": max(
                (
                    float(pose["rgb_scatter_mm"]["p95"])
                    for pose in height_poses
                    if pose["rgb_scatter_mm"].get("p95") is not None
                ),
                default=None,
            ),
            "passed": not failure_reasons,
            "failure_reasons": "|".join(failure_reasons),
        })
    return summaries


def _flatten_rgb_pose_summary(summary: dict[str, Any]) -> dict[str, Any]:
    rgb = summary.get("rgb_xyz_median_mm") or [None, None, None]
    tcp = summary.get("tcp_xyz_median_mm") or [None, None, None]
    return {
        "height_target_mm": summary["height_target_mm"],
        "pose_name": summary["pose_name"],
        "pose_index": summary["pose_index"],
        "frame_count": summary["frame_count"],
        "valid_count": summary["valid_count"],
        "valid_ratio": summary["valid_ratio"],
        "rgb_x_median_mm": rgb[0], "rgb_y_median_mm": rgb[1], "rgb_z_median_mm": rgb[2],
        "tcp_x_median_mm": tcp[0], "tcp_y_median_mm": tcp[1], "tcp_z_median_mm": tcp[2],
        "rgb_scatter_p95_mm": summary["rgb_scatter_mm"].get("p95"),
        "rgb_reprojection_rmse_p95_px": summary["rgb_reprojection_rmse_px"].get("p95"),
        "rgb_reprojection_max_p95_px": summary["rgb_reprojection_max_px"].get("p95"),
        "charuco_count_median": summary["rgb_charuco_count"].get("median"),
        "pnp_inlier_count_median": summary["rgb_pnp_inlier_count"].get("median"),
        "target_distance_p95_px": summary["target_distance_px"].get("p95"),
    }


def write_rgb_precision_plots(
    output_dir: Path,
    height_summaries: list[dict[str, Any]],
) -> list[str]:
    if not height_summaries:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"plots_skipped:{exc}"]

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    heights = np.asarray([row["height_target_mm"] for row in height_summaries], dtype=float)
    metrics = (
        ("distance_error_p95_mm", "TCP-RGB distance error P95 (mm)", 0.5),
        ("rigid_rms_mm", "TCP-RGB rigid-fit RMS (mm)", None),
        ("valid_ratio", "RGB valid ratio", 0.95),
        ("rgb_reprojection_rmse_p95_px", "RGB reprojection RMSE P95 (px)", 0.35),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for axis, (key, label, limit) in zip(axes.ravel(), metrics):
        values = np.asarray([
            np.nan if row.get(key) is None else float(row[key]) for row in height_summaries
        ])
        axis.plot(heights, values, "o-", linewidth=2)
        if limit is not None:
            axis.axhline(limit, color="red", linestyle="--", label=f"gate={limit:g}")
            axis.legend()
        axis.set_xlabel("Nominal RGB PnP Z (mm)")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    fig.suptitle("RGB precision range")
    fig.tight_layout()
    path = plot_dir / "rgb_precision_range.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [str(path)]


class RgbPrecisionRangeExperiment:
    def __init__(
        self,
        config: RgbPrecisionRangeConfig,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
    ) -> None:
        config.validate()
        self.config = config
        self.run_id = f"charuco-rgb-precision-{timestamp_str()}"
        self.output_dir = Path(output_root) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.raw_csv = self.output_dir / "rgb_frames.csv"
        self.rows: list[dict[str, Any]] = []
        self.pipeline: Any = None
        self.motion: AuboMotionSession | None = None
        self.board, self.dictionary = create_charuco_board()
        self.device_identity: dict[str, Any] = {}
        self.exposure: dict[str, Any] = {}
        self.intrinsics: Any = None
        self.actual_color_profile: dict[str, Any] = {}
        self.first_frame_metadata: dict[str, Any] | None = None

    def read_bundle(self) -> RgbFrameBundle | None:
        return get_rgb_frame_bundle(self.pipeline)

    def require_bundle(self) -> RgbFrameBundle:
        bundle = self.read_bundle()
        if bundle is None:
            raise RuntimeError("Gemini RGB帧读取失败")
        return bundle

    def observe(self) -> tuple[RgbFrameBundle, BoardPoseResult]:
        bundle = self.require_bundle()
        if bundle.intrinsics is None:
            raise RuntimeError("RGB内参不可用")
        rgb = estimate_rgb_board_pose(
            bundle.color_bgr, bundle.intrinsics, self.board, self.dictionary,
        )
        return bundle, rgb

    def start(self) -> None:
        self.pipeline = init_rgb_handeye_pipeline()
        self.actual_color_profile = _video_profile_metadata(get_color_profile(self.pipeline))
        self.device_identity = get_device_identity(self.pipeline)
        actual_serial = str(self.device_identity.get("serial_number", "")).strip()
        expected = self.config.expected_camera_serial.strip()
        if expected and actual_serial != expected:
            raise RuntimeError(f"相机序列号不匹配：期望={expected}，当前={actual_serial}")
        self.exposure = read_and_lock_color_exposure(
            self.pipeline, self.config.warmup_frames, self.read_bundle,
        )
        if not self.exposure.get("locked"):
            raise RuntimeError(f"彩色曝光/增益锁定失败：{self.exposure.get('errors')}")
        bundle = self.require_bundle()
        if bundle.intrinsics is None:
            raise RuntimeError("RGB内参不可用")
        self.intrinsics = bundle.intrinsics
        self.first_frame_metadata = bundle.metadata_dict()
        self.motion = AuboMotionSession()
        self.motion.connect(
            ROBOT_CFG.ip, ROBOT_CFG.rpc_port, ROBOT_CFG.user, ROBOT_CFG.password,
            ROBOT_CFG.request_timeout_ms,
        )

    def close(self) -> None:
        if self.motion is not None:
            self.motion.disconnect()
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()

    def current_rgb_z(self, sample_count: int = 7) -> float:
        values: list[float] = []
        for _ in range(max(sample_count * 3, sample_count)):
            _bundle, rgb = self.observe()
            if rgb.T_rgb_board is not None:
                values.append(float(np.asarray(rgb.T_rgb_board)[2, 3]))
            if len(values) >= sample_count:
                break
        if len(values) < max(3, sample_count // 2):
            raise RuntimeError(f"无法稳定测量RGB-PnP Z：有效帧={len(values)}/{sample_count}")
        return float(np.median(values))

    def move_tcp_z(self, target_z_m: float, description: str) -> dict[str, Any]:
        if self.motion is None:
            raise RuntimeError("AUBO会话未连接")
        snapshot = self.motion.snapshot()
        validate_robot_ready(snapshot)
        current = [float(value) for value in snapshot["tcp_pose_m_rad"][:6]]
        target = current.copy()
        target[2] = float(target_z_m)
        delta_mm = (target[2] - current[2]) * 1000.0
        if abs(delta_mm) > self.config.max_step_mm + 1e-9:
            raise RuntimeError(
                f"{description}单步Z移动{delta_mm:+.3f} mm超过"
                f"{self.config.max_step_mm:.1f} mm限制"
            )
        print(
            f"[Z-MOVE] {description}：TCP Z {current[2]*1000.0:.3f} -> "
            f"{target[2]*1000.0:.3f} mm，移动={delta_mm:+.3f} mm"
        )
        if not self.config.execute_motion:
            raise RuntimeError("当前为运动预览模式；使用--execute-motion后才会执行")
        answer = input("确认路径安全，输入 m 执行Z移动，其他键取消：").strip().lower()
        if answer != "m":
            raise RuntimeError("用户取消Z移动")
        returns = self.motion.move_line(
            target,
            self.config.speed_mm_s / 1000.0,
            self.config.acceleration_mm_s2 / 1000.0,
        )
        if not returns or not sdk_ok(returns[-1]):
            raise RuntimeError(f"moveLine调用失败：{returns}")
        arrival = wait_for_target(
            self.motion, target, self.config.motion_timeout_s,
            self.config.position_tolerance_mm, self.config.rotation_tolerance_rad,
        )
        time.sleep(0.4)
        return arrival

    def move_tcp_xy(
        self,
        reference_tcp_m_rad: list[float],
        offset_x_mm: float,
        offset_y_mm: float,
        description: str,
    ) -> dict[str, Any]:
        """Move exactly one automatic XY segment while keeping Z and orientation fixed."""
        if self.motion is None:
            raise RuntimeError("AUBO会话未连接")
        snapshot = self.motion.snapshot()
        validate_robot_ready(snapshot)
        current = [float(value) for value in snapshot["tcp_pose_m_rad"][:6]]
        target = [float(value) for value in reference_tcp_m_rad[:6]]
        target[0] += float(offset_x_mm) / 1000.0
        target[1] += float(offset_y_mm) / 1000.0
        segment_xy_mm = float(np.linalg.norm(
            np.asarray(target[:2], dtype=np.float64)
            - np.asarray(current[:2], dtype=np.float64)
        ) * 1000.0)
        if segment_xy_mm > self.config.max_tcp_step_mm + 1e-6:
            raise RuntimeError(
                f"{description}的单段XY移动={segment_xy_mm:.3f} mm，超过"
                f"{self.config.max_tcp_step_mm:.1f} mm限制"
            )
        print(
            f"[XY-AUTO] {description}："
            f"ΔX={target[0]*1000.0-current[0]*1000.0:+.3f} mm，"
            f"ΔY={target[1]*1000.0-current[1]*1000.0:+.3f} mm，"
            f"段长={segment_xy_mm:.3f} mm"
        )
        returns = self.motion.move_line(
            target,
            self.config.speed_mm_s / 1000.0,
            self.config.acceleration_mm_s2 / 1000.0,
        )
        if not returns or not sdk_ok(returns[-1]):
            raise RuntimeError(f"自动XY moveLine调用失败：{returns}")
        arrival = wait_for_target(
            self.motion, target, self.config.motion_timeout_s,
            self.config.position_tolerance_mm, self.config.rotation_tolerance_rad,
        )
        time.sleep(0.3)
        return arrival

    def reach_initial_height(self, target_mm: float) -> float:
        if self.motion is None:
            raise RuntimeError("AUBO会话未连接")
        last_z = self.current_rgb_z()
        for correction in range(1, self.config.max_corrections + 1):
            residual = float(target_mm - last_z)
            print(
                f"[RGB-HEIGHT] 目标={target_mm:.1f} mm，"
                f"当前PnP Z={last_z:.3f} mm，残差={residual:+.3f} mm"
            )
            if abs(residual) <= self.config.target_tolerance_mm:
                return last_z
            snapshot = self.motion.snapshot()
            validate_robot_ready(snapshot)
            current_tcp_z = float(snapshot["tcp_pose_m_rad"][2])
            step_mm = float(np.clip(
                residual, -self.config.max_step_mm, self.config.max_step_mm,
            ))
            self.move_tcp_z(
                current_tcp_z + step_mm / 1000.0,
                f"首次对准{target_mm:.1f} mm，第{correction}次修正",
            )
            measured_z = self.current_rgb_z()
            if abs(target_mm - measured_z) >= abs(residual):
                raise RuntimeError(
                    "首次高度修正后PnP误差没有减小；"
                    f"移动前={last_z:.3f}，移动后={measured_z:.3f}，目标={target_mm:.3f} mm"
                )
            last_z = measured_z
        if abs(last_z - target_mm) > self.config.target_tolerance_mm:
            raise RuntimeError(
                f"达到最大修正次数后仍未进入高度容差："
                f"目标={target_mm:.3f}，当前={last_z:.3f} mm"
            )
        return last_z

    def confirm_manual_pose(
        self,
        height_mm: float,
        pose_name: str,
        target_uv: tuple[float, float],
        center_snapshot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if self.motion is None:
            raise RuntimeError("AUBO会话未连接")
        window_name = "RGB precision range | C/Enter=confirm Q=quit"
        loop_index = 0
        while True:
            loop_index += 1
            bundle, rgb = self.observe()
            view = (
                rgb.rgb_overlay.copy()
                if rgb.rgb_overlay is not None
                else bundle.color_bgr.copy()
            )
            target = (int(round(target_uv[0])), int(round(target_uv[1])))
            cv2.circle(view, target, int(round(self.config.confirm_tolerance_px)), (0, 255, 255), 2)
            cv2.circle(view, target, 5, (0, 255, 255), -1)
            center = _rgb_image_center(rgb, bundle.intrinsics)
            distance_px = math.inf
            if center is not None:
                distance_px = float(math.hypot(
                    center[0] - target_uv[0], center[1] - target_uv[1],
                ))
                cv2.circle(
                    view, (int(round(center[0])), int(round(center[1]))),
                    8, (0, 255, 0), 2,
                )
            ready = bool(
                rgb.ok and rgb.T_rgb_board is not None
                and distance_px <= self.config.confirm_tolerance_px
            )
            status = (
                f"{height_mm:.0f}mm {pose_name} "
                f"target=({target_uv[0]:.1f},{target_uv[1]:.1f}) "
                f"d={distance_px:.1f}px {'READY' if ready else 'MOVE'}"
            )
            cv2.putText(
                view, status, (18, view.shape[0] - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (0, 255, 0) if ready else (0, 165, 255), 2,
            )
            cv2.imshow(window_name, view)
            key = cv2.waitKey(30) & 0xFF
            if loop_index == 1 or loop_index % 20 == 0:
                print(f"[RGB-POSITION] {status}")
            if key in (ord("q"), ord("Q")):
                raise KeyboardInterrupt
            if key not in (ord("c"), ord("C"), 13, 32):
                continue
            if not ready:
                print(
                    f"[RGB-POSITION] 尚未进入{self.config.confirm_tolerance_px:.1f}px确认圈"
                )
                continue
            snapshot = self.motion.snapshot()
            validate_robot_ready(snapshot)
            if center_snapshot is not None:
                reference = np.asarray(center_snapshot["tcp_pose_m_rad"][:6], dtype=np.float64)
                current = np.asarray(snapshot["tcp_pose_m_rad"][:6], dtype=np.float64)
                xy_step_mm = float(np.linalg.norm(current[:2] - reference[:2]) * 1000.0)
                z_drift_mm = abs(float(current[2] - reference[2])) * 1000.0
                rotation_drift_deg = math.degrees(float(np.linalg.norm([
                    _angular_delta_rad(reference[index], current[index])
                    for index in range(3, 6)
                ])))
                problems: list[str] = []
                if pose_name != "CENTER_RETURN" and not (
                    self.config.min_tcp_step_mm
                    <= xy_step_mm
                    <= self.config.max_tcp_step_mm
                ):
                    problems.append(
                        f"TCP XY位移={xy_step_mm:.3f} mm，要求"
                        f"{self.config.min_tcp_step_mm:.1f}–"
                        f"{self.config.max_tcp_step_mm:.1f} mm"
                    )
                if z_drift_mm > self.config.max_tcp_z_drift_mm:
                    problems.append(
                        f"TCP Z漂移={z_drift_mm:.3f} mm>"
                        f"{self.config.max_tcp_z_drift_mm:.3f} mm"
                    )
                if rotation_drift_deg > self.config.max_tcp_rotation_drift_deg:
                    problems.append(
                        f"TCP姿态漂移={rotation_drift_deg:.4f}°>"
                        f"{self.config.max_tcp_rotation_drift_deg:.4f}°"
                    )
                if problems:
                    print("[RGB-POSITION] 位置不满足实验条件：" + "；".join(problems))
                    continue
            return snapshot

    def capture_pose(
        self,
        height_mm: float,
        pose_name: str,
        pose_index: int,
        target_uv: tuple[float, float] | None,
    ) -> None:
        if self.motion is None:
            raise RuntimeError("AUBO会话未连接")
        image_dir = self.output_dir / "images" / f"height_{height_mm:.0f}mm" / pose_name
        image_dir.mkdir(parents=True, exist_ok=True)
        pose_rows: list[dict[str, Any]] = []
        for frame_index in range(1, self.config.frames_per_position + 1):
            snapshot = self.motion.snapshot()
            validate_robot_ready(snapshot)
            bundle, rgb = self.observe()
            record = build_rgb_precision_frame_record(
                self.run_id, height_mm, pose_name, pose_index, frame_index,
                bundle, rgb, snapshot, target_uv,
            )
            self.rows.append(record)
            pose_rows.append(record)
            write_csv(self.raw_csv, self.rows, RGB_PRECISION_FRAME_FIELDS)
            if frame_index % self.config.batch_size == 0:
                batch_index = frame_index // self.config.batch_size
                cv2.imwrite(
                    str(image_dir / f"batch_{batch_index:02d}_rgb_overlay.png"),
                    rgb.rgb_overlay,
                )
                recent = pose_rows[-self.config.batch_size:]
                print(
                    f"[RGB-CAPTURE] {height_mm:.0f}mm {pose_name} "
                    f"{frame_index}/{self.config.frames_per_position}，"
                    f"有效={sum(bool(row['valid']) for row in recent)}/{len(recent)}"
                )

    def capture_auto_xy_layer(self, height_mm: float) -> None:
        """Capture the center micro-cross using robot-base XY moves only."""
        if self.motion is None:
            raise RuntimeError("AUBO会话未连接")
        snapshot = self.motion.snapshot()
        validate_robot_ready(snapshot)
        reference = [float(value) for value in snapshot["tcp_pose_m_rad"][:6]]
        step = float(self.config.center_step_mm)
        print(
            f"[XY-AUTO-PLAN] {height_mm:.0f} mm："
            f"中心 → X-{step:g} → 中心 → X+{step:g} → 中心 → "
            f"Y-{step:g} → 中心 → Y+{step:g} → 中心；"
            f"每一段XY移动{step:g} mm，Z和姿态锁定。"
        )
        answer = input(
            "确认该层自动XY路径安全，输入 m 开始五点自动采集，其他键取消："
        ).strip().lower()
        if answer != "m":
            raise RuntimeError("用户取消自动XY五点采集")

        self.capture_pose(height_mm, "CENTER", 1, None)
        points = (
            ("X_MINUS", -step, 0.0),
            ("X_PLUS", step, 0.0),
            ("Y_MINUS", 0.0, -step),
            ("Y_PLUS", 0.0, step),
        )
        for pose_index, (pose_name, offset_x_mm, offset_y_mm) in enumerate(points, 2):
            self.move_tcp_xy(
                reference, offset_x_mm, offset_y_mm,
                f"{height_mm:.0f} mm / {pose_name}",
            )
            self.capture_pose(height_mm, pose_name, pose_index, None)
            self.move_tcp_xy(
                reference, 0.0, 0.0,
                f"{height_mm:.0f} mm / 返回中心",
            )

    def finalize(self, status: str, error: str | None = None) -> dict[str, Any]:
        pose_summaries = build_rgb_pose_summaries(self.rows)
        height_summaries = build_rgb_height_summaries(
            self.rows, pose_summaries, self.config,
        )
        selection = select_contiguous_height_ranges(height_summaries)
        write_csv(self.raw_csv, self.rows, RGB_PRECISION_FRAME_FIELDS)
        write_csv(
            self.output_dir / "rgb_pose_summary.csv",
            [_flatten_rgb_pose_summary(row) for row in pose_summaries],
        )
        write_csv(self.output_dir / "rgb_height_summary.csv", height_summaries)
        plot_outputs = write_rgb_precision_plots(self.output_dir, height_summaries)
        report = {
            "record_type": "charuco_rgb_robot_relative_precision_range_experiment",
            "run_id": self.run_id,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": status,
            "error": error,
            "absolute_accuracy_claimed": False,
            "interpretation": "机器人TCP相对运动作为参考，不代表外部绝对精度。",
            "capture_mode": "rgb_only_no_depth_or_pointcloud",
            "configuration": asdict(self.config),
            "board": {
                "pattern_width_mm": BOARD_CFG.pattern_width_mm,
                "pattern_height_mm": BOARD_CFG.pattern_height_mm,
                "square_length_mm": BOARD_CFG.square_length_mm,
                "marker_length_mm": BOARD_CFG.marker_length_mm,
                "squares_x": BOARD_CFG.squares_x,
                "squares_y": BOARD_CFG.squares_y,
                "dictionary": BOARD_CFG.aruco_dict_name,
            },
            "gates": {
                "minimum_valid_ratio": self.config.minimum_valid_ratio,
                "rgb_reprojection_rmse_p95_max_px": BOARD_CFG.max_rgb_reprojection_rmse_px,
                "rgb_reprojection_max_p95_max_px": BOARD_CFG.max_rgb_reprojection_error_px,
                "tcp_rgb_distance_error_p95_max_mm": self.config.maximum_distance_p95_mm,
            },
            "camera": self.device_identity,
            "color_profile": self.actual_color_profile,
            "intrinsics": None if self.intrinsics is None else self.intrinsics.as_dict(),
            "first_frame_metadata": self.first_frame_metadata,
            "exposure_lock": self.exposure,
            "frame_count": len(self.rows),
            "expected_frame_count": (
                len(self.config.heights_mm)
                * len(RGB_PRECISION_POSE_NAMES)
                * self.config.frames_per_position
            ),
            "pose_summaries": pose_summaries,
            "height_summaries": height_summaries,
            "precision_selection": selection,
            "plot_outputs": plot_outputs,
            "frames": self.rows,
        }
        atomic_write_json(self.output_dir / "report.json", report)
        return report

    @staticmethod
    def print_summary_table(report: dict[str, Any]) -> None:
        print("\nRGB精度范围汇总")
        print(
            "高度(mm)  有效率   距离RMS(mm)  距离P95(mm)  "
            "刚体RMS(mm)  重投影P95(px)  结果"
        )
        for row in report.get("height_summaries", []):
            def number(key: str, digits: int = 3) -> str:
                value = row.get(key)
                return "—" if value is None else f"{float(value):.{digits}f}"
            print(
                f"{float(row['height_target_mm']):8.0f}  "
                f"{float(row['valid_ratio']):7.1%}  "
                f"{number('distance_error_rms_mm'):>12}  "
                f"{number('distance_error_p95_mm'):>12}  "
                f"{number('rigid_rms_mm'):>11}  "
                f"{number('rgb_reprojection_rmse_p95_px'):>14}  "
                f"{'PASS' if row.get('passed') else 'FAIL'}"
            )
        selection = report.get("precision_selection", {})
        recommended = selection.get("recommended_range")
        if recommended:
            print(
                f"[RESULT] 最佳连续范围={recommended['start_mm']:.0f}–"
                f"{recommended['stop_mm']:.0f} mm；"
                f"最佳高度={selection['best_height_mm']:.0f} mm"
            )
        else:
            print("[RESULT] 没有高度满足全部RGB精度门槛")

    def run(self) -> int:
        try:
            self.start()
            if not self.config.execute_motion:
                current_z = self.current_rgb_z()
                print(f"[PREVIEW] 当前PnP Z={current_z:.3f} mm")
                print(
                    f"[PREVIEW] 首次对准{self.config.heights_mm[0]:.1f} mm，随后TCP Z按 "
                    f"{list(self.config.heights_mm)} mm高度表移动；"
                    f"XY模式={'自动20mm五点' if self.config.auto_xy else '人工五点'}。"
                )
                report = self.finalize("motion_preview")
                self.print_summary_table(report)
                print(f"[PREVIEW] 输出目录：{self.output_dir}")
                return 0
            first_height = float(self.config.heights_mm[0])
            input(
                f"准备首次对准{first_height:.0f} mm。确认标定板固定且Z路径安全后按Enter："
            )
            reached = self.reach_initial_height(first_height)
            print(f"[RGB-HEIGHT-READY] PnP Z={reached:.3f} mm")
            if self.motion is None:
                raise RuntimeError("AUBO会话未连接")
            initial_snapshot = self.motion.snapshot()
            validate_robot_ready(initial_snapshot)
            anchor_tcp_z_m = float(initial_snapshot["tcp_pose_m_rad"][2])

            for height_index, height in enumerate(self.config.heights_mm):
                height = float(height)
                if height_index > 0:
                    scheduled_tcp_z = anchor_tcp_z_m + (height - first_height) / 1000.0
                    self.move_tcp_z(scheduled_tcp_z, f"进入{height:.0f} mm层")
                print("\n" + "=" * 76)
                if self.config.auto_xy:
                    print(f"[RGB-LAYER] {height:.0f} mm，自动基座XY五点，每点20帧")
                    self.capture_auto_xy_layer(height)
                else:
                    print(f"[RGB-LAYER] {height:.0f} mm，人工图像五点，每点20帧")
                    targets = rgb_precision_targets(
                        self.intrinsics, height, self.config.center_step_mm,
                    )
                    center_snapshot: dict[str, Any] | None = None
                    for pose_index, (pose_name, target_u, target_v) in enumerate(targets, 1):
                        target_uv = (target_u, target_v)
                        snapshot = self.confirm_manual_pose(
                            height, pose_name, target_uv, center_snapshot,
                        )
                        if pose_name == "CENTER":
                            center_snapshot = snapshot
                        self.capture_pose(
                            height, pose_name, pose_index, target_uv,
                        )
                    center_target = (targets[0][1], targets[0][2])
                    self.confirm_manual_pose(
                        height, "CENTER_RETURN", center_target, center_snapshot,
                    )
                input(
                    f"{height:.0f} mm五点完成并已返回中心。"
                    "确认现场安全后按Enter进入下一层："
                )

            report = self.finalize("completed")
            self.print_summary_table(report)
            print(f"[DONE] RGB精度实验目录：{self.output_dir}")
            return 0
        except KeyboardInterrupt:
            if self.motion is not None:
                try:
                    self.motion.stop_motion()
                except Exception:
                    pass
            report = self.finalize("interrupted", "KeyboardInterrupt")
            self.print_summary_table(report)
            return 130
        except Exception as exc:
            report = self.finalize("failed", str(exc))
            self.print_summary_table(report)
            print(f"[ERROR] {exc}")
            print(f"[PARTIAL] 已保存部分结果：{self.output_dir}")
            return 2
        finally:
            self.close()


class HeightErrorExperiment:
    def __init__(self, config: HeightExperimentConfig, output_root: Path = DEFAULT_OUTPUT_ROOT) -> None:
        config.validate()
        self.config = config
        self.run_id = "charuco-height-" + timestamp_str()
        self.output_dir = Path(output_root) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.raw_csv = self.output_dir / "frames.csv"
        self.rows: list[dict[str, Any]] = []
        self.pipeline: Any = None
        self.align: Any = None
        self.depth_processor: Any = None
        self.motion: AuboMotionSession | None = None
        self.board, self.dictionary = create_charuco_board()
        self.device_identity: dict[str, Any] = {}
        self.exposure: dict[str, Any] = {}
        self.intrinsics: dict[str, Any] | None = None
        self.first_frame_metadata: dict[str, Any] | None = None
        self.actual_stream_profiles: dict[str, Any] = {}

    def start(self) -> None:
        self.pipeline, self.align, self.depth_processor = init_pipeline(self.config.depth_filter_mode)
        self.actual_stream_profiles = {
            "color": _video_profile_metadata(get_color_profile(self.pipeline)),
            "depth": _video_profile_metadata(get_depth_profile(self.pipeline)),
        }
        self.device_identity = get_device_identity(self.pipeline)
        actual_serial = str(self.device_identity.get("serial_number", "")).strip()
        expected = self.config.expected_camera_serial.strip()
        if expected and actual_serial != expected:
            raise RuntimeError(f"相机序列号不匹配：期望={expected}，当前={actual_serial}")
        self.exposure = read_and_lock_color_exposure(
            self.pipeline, self.config.warmup_frames, self.read_bundle,
        )
        if not self.exposure.get("locked"):
            raise RuntimeError(f"彩色曝光/增益锁定失败：{self.exposure.get('errors')}")
        bundle = self.require_bundle()
        self.intrinsics = None if bundle.intrinsics is None else bundle.intrinsics.as_dict()
        self.first_frame_metadata = bundle.metadata_dict()
        if bundle.intrinsics is None:
            raise RuntimeError("RGB内参不可用")
        self.motion = AuboMotionSession()
        self.motion.connect(
            ROBOT_CFG.ip, ROBOT_CFG.rpc_port, ROBOT_CFG.user, ROBOT_CFG.password,
            ROBOT_CFG.request_timeout_ms,
        )

    def close(self) -> None:
        if self.motion is not None:
            self.motion.disconnect()
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()

    def read_bundle(self) -> CameraFrameBundle | None:
        return get_aligned_frame_bundle(self.pipeline, self.align, self.depth_processor)

    def require_bundle(self) -> CameraFrameBundle:
        bundle = self.read_bundle()
        if bundle is None:
            raise RuntimeError("Gemini RGB-D 对齐帧读取失败")
        return bundle

    def observe(self) -> tuple[CameraFrameBundle, BoardPoseResult, BoardPoseResult]:
        bundle = self.require_bundle()
        rgb = estimate_rgb_board_pose(
            bundle.color_bgr, bundle.intrinsics, self.board, self.dictionary,
        )
        depth = estimate_pointcloud_board_pose(
            bundle.color_bgr, bundle.depth_mm, bundle.xyz_map_mm, self.board, self.dictionary,
        )
        return bundle, rgb, depth

    def current_rgb_z(self) -> float:
        _bundle, rgb, _depth = self.observe()
        if rgb.T_rgb_board is None:
            raise RuntimeError(f"无法测量当前PnP Z：{rgb.status}")
        return float(np.asarray(rgb.T_rgb_board)[2, 3])

    def reach_height(self, target_mm: float) -> float:
        if self.motion is None:
            raise RuntimeError("AUBO会话未启动")
        last_z = self.current_rgb_z()
        for correction in range(1, self.config.max_corrections + 1):
            residual = float(target_mm - last_z)
            print(f"[HEIGHT] 目标={target_mm:.1f} mm，当前PnP Z={last_z:.3f} mm，差={residual:+.3f} mm")
            if abs(residual) <= self.config.target_tolerance_mm:
                return last_z
            snapshot = self.motion.snapshot()
            validate_robot_ready(snapshot)
            current = [float(v) for v in snapshot["tcp_pose_m_rad"][:6]]
            # 当前相机朝向下，TCP基坐标Z增加会使板中心的RGB光轴Z减小，
            # 因此机器人TCP Z修正方向与视觉高度残差相反。
            step_mm = float(np.clip(residual, -self.config.max_step_mm, self.config.max_step_mm))
            target = current.copy()
            target[2] += step_mm / 1000.0
            print(
                f"[PROPOSE] 第{correction}/{self.config.max_corrections}次修正：仅TCP Z移动 "
                f"{step_mm:+.3f} mm；XY和姿态保持不变。"
            )
            if not self.config.execute_motion:
                raise RuntimeError(
                    "当前为运动预览模式，未发送moveLine。使用--execute-motion并人工确认后才能移动。"
                )
            answer = input("确认路径安全并执行本次移动？输入 m，其他内容取消：").strip().lower()
            if answer != "m":
                raise RuntimeError("操作者取消本次高度移动")
            returns = self.motion.move_line(
                target,
                self.config.speed_mm_s / 1000.0,
                self.config.acceleration_mm_s2 / 1000.0,
            )
            if not returns or not sdk_ok(returns[-1]):
                raise RuntimeError(f"moveLine返回失败：{returns}")
            arrival = wait_for_target(
                self.motion, target, self.config.motion_timeout_s,
                self.config.position_tolerance_mm, self.config.rotation_tolerance_rad,
            )
            print(
                f"[ARRIVED] TCP位置残差={arrival['position_error_mm']:.3f} mm，"
                f"姿态残差={arrival['rotation_error_rad']:.6f} rad"
            )
            time.sleep(0.4)
            measured_z = self.current_rgb_z()
            if abs(target_mm - measured_z) >= abs(residual) and abs(residual) > self.config.target_tolerance_mm:
                raise RuntimeError(
                    "TCP Z修正后PnP高度误差没有减小，运动方向或相机安装方向与假设不符；"
                    f"移动前Z={last_z:.3f}，移动后Z={measured_z:.3f}，目标={target_mm:.3f} mm。"
                )
            last_z = measured_z
        if abs(last_z - target_mm) > self.config.target_tolerance_mm:
            raise RuntimeError(
                f"三次修正后仍未到达目标高度：目标={target_mm:.3f}，实测={last_z:.3f} mm"
            )
        return last_z

    def capture_height(self, target_mm: float) -> None:
        if self.motion is None:
            raise RuntimeError("AUBO会话未启动")
        print(f"[CAPTURE] 开始采集 {target_mm:.0f} mm，共 {self.config.frames_per_height} 帧")
        image_dir = self.output_dir / "images" / f"height_{target_mm:.0f}mm"
        image_dir.mkdir(parents=True, exist_ok=True)
        height_rows: list[dict[str, Any]] = []
        for frame_index in range(1, self.config.frames_per_height + 1):
            snapshot = self.motion.snapshot()
            validate_robot_ready(snapshot)
            bundle, rgb, depth = self.observe()
            record = build_frame_record(
                self.run_id, target_mm, frame_index, self.config.batch_size,
                bundle, rgb, depth, snapshot,
            )
            self.rows.append(record)
            height_rows.append(record)
            write_csv(self.raw_csv, self.rows, CSV_FIELDS)
            # 标定板完全丢失属于当前高度的硬中止；深度质量失败仍原样保存并继续统计。
            if rgb.T_rgb_board is None:
                raise RuntimeError(f"第{frame_index}帧标定板丢失，已保存失败记录并停止当前高度")
            if frame_index % self.config.batch_size == 0:
                batch = int(record["batch_index"])
                cv2.imwrite(str(image_dir / f"batch_{batch:02d}_rgb_overlay.png"), rgb.rgb_overlay)
                cv2.imwrite(str(image_dir / f"batch_{batch:02d}_depth.png"), depth_to_vis(bundle.depth_mm))
                print(
                    f"[CAPTURE] {target_mm:.0f} mm：{frame_index}/{self.config.frames_per_height}，"
                    f"当前批次有效={sum(bool(row['valid']) for row in height_rows[-self.config.batch_size:])}"
                    f"/{self.config.batch_size}"
                )

    def confirm_manual_grid_position(
        self,
        target_height_mm: float,
        position_name: str,
        u_norm: float,
        v_norm: float,
    ) -> tuple[float, float]:
        """持续刷新当前板中心与目标像素；操作者用窗口按键确认。"""
        window_name = "ChArUco field grid: yellow=target green=current"
        print(
            f"[GRID] {position_name}：请手动移动XY。"
            "窗口按 C 确认采集，R 继续观察，Q 退出。"
        )
        loop_index = 0
        while True:
            loop_index += 1
            bundle, rgb, _depth = self.observe()
            height, width = bundle.color_bgr.shape[:2]
            target_u, target_v = float(u_norm * width), float(v_norm * height)
            view = rgb.rgb_overlay.copy()
            cv2.drawMarker(
                view, (int(round(target_u)), int(round(target_v))), (0, 255, 255),
                cv2.MARKER_CROSS, 32, 3,
            )
            if rgb.T_rgb_board is None or not rgb.image_points:
                message = (
                    f"{position_name} target=({target_u:.0f},{target_v:.0f}) "
                    "current=NOT DETECTED | C=confirm R=retry Q=quit"
                )
                cv2.putText(
                    view, message, (18, height - 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (0, 0, 255), 2,
                )
                current_text = "未检测到完整ChArUco板"
            else:
                points = np.asarray(rgb.image_points, dtype=np.float64).reshape(-1, 2)
                center = np.mean(points, axis=0)
                du, dv = float(center[0] - target_u), float(center[1] - target_v)
                measured_z = float(np.asarray(rgb.T_rgb_board)[2, 3])
                cv2.circle(
                    view, (int(round(center[0])), int(round(center[1]))), 10, (0, 255, 0), 3,
                )
                cv2.putText(
                    view,
                    f"{position_name} target=({target_u:.0f},{target_v:.0f}) "
                    f"current=({center[0]:.1f},{center[1]:.1f}) "
                    f"d=({du:+.1f},{dv:+.1f})px Z={measured_z:.1f}mm | C=confirm R=retry Q=quit",
                    (18, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2,
                )
                current_text = (
                    f"目标=({target_u:.0f},{target_v:.0f})，当前=({center[0]:.1f},{center[1]:.1f})，"
                    f"偏差=({du:+.1f},{dv:+.1f}) px，PnP Z={measured_z:.3f} mm"
                )
            cv2.imshow(window_name, view)
            key = cv2.waitKey(30) & 0xFF
            if loop_index == 1 or loop_index % 10 == 0:
                print(f"[GRID] 高度={target_height_mm:.0f} mm，位置={position_name}，{current_text}")
            if key in (ord("q"), ord("Q")):
                raise KeyboardInterrupt
            if key in (ord("c"), ord("C"), 13, 32):
                return target_u, target_v

    def capture_field_position(
        self,
        target_height_mm: float,
        position_index: int,
        position_name: str,
        target_uv: tuple[float, float],
    ) -> None:
        if self.motion is None:
            raise RuntimeError("AUBO会话未启动")
        count = int(self.config.frames_per_position)
        image_dir = (
            self.output_dir / "images" / f"height_{target_height_mm:.0f}mm" / position_name
        )
        image_dir.mkdir(parents=True, exist_ok=True)
        position_rows: list[dict[str, Any]] = []
        print(f"[GRID-CAPTURE] {target_height_mm:.0f} mm / {position_name}：采集 {count} 帧")
        for local_index in range(1, count + 1):
            frame_index = (position_index - 1) * count + local_index
            snapshot = self.motion.snapshot()
            validate_robot_ready(snapshot)
            bundle, rgb, depth = self.observe()
            record = build_frame_record(
                self.run_id, target_height_mm, frame_index, self.config.batch_size,
                bundle, rgb, depth, snapshot,
                grid_position=position_name, grid_target_uv=target_uv,
            )
            self.rows.append(record)
            position_rows.append(record)
            write_csv(self.raw_csv, self.rows, CSV_FIELDS)
            if rgb.T_rgb_board is None:
                raise RuntimeError(
                    f"{position_name}第{local_index}帧标定板丢失，已保存失败记录"
                )
            if local_index % self.config.batch_size == 0 or local_index == count:
                batch = int(math.ceil(local_index / self.config.batch_size))
                cv2.imwrite(str(image_dir / f"batch_{batch:02d}_rgb_overlay.png"), rgb.rgb_overlay)
                cv2.imwrite(str(image_dir / f"batch_{batch:02d}_depth.png"), depth_to_vis(bundle.depth_mm))
                recent = position_rows[-min(self.config.batch_size, len(position_rows)):]
                print(
                    f"[GRID-CAPTURE] {position_name}：{local_index}/{count}，"
                    f"最近批次有效={sum(bool(row['valid']) for row in recent)}/{len(recent)}"
                )

    def run_field_grid(self) -> None:
        for target_height in self.config.heights_mm:
            print("\n" + "=" * 76)
            input(
                f"准备进入 {target_height:.0f} mm 高度。确认Z方向路径安全后按 Enter："
            )
            reached = self.reach_height(target_height)
            print(
                f"[HEIGHT-READY] 目标={target_height:.1f} mm，"
                f"当前PnP Z={reached:.3f} mm。接下来XY全部由人工移动。"
            )
            for position_index, (name, u_norm, v_norm) in enumerate(DEFAULT_FIELD_GRID, 1):
                print(
                    f"\n[GRID {position_index}/9] {name}，"
                    f"归一化目标=({u_norm:.2f},{v_norm:.2f})"
                )
                target_uv = self.confirm_manual_grid_position(
                    target_height, name, u_norm, v_norm,
                )
                # 3x3视野实验期间固定当前机械高度。PnP Z随XY位置的变化正是
                # 待测量的视野误差，因此只记录，不在九个网格点之间闭环调高。
                self.capture_field_position(
                    target_height, position_index, name, target_uv,
                )
            input(
                f"{target_height:.0f} mm 的3x3网格已完成。"
                "检查现场后按 Enter 自动进入下一高度："
            )

    def finalize(self, status: str = "completed", error: str | None = None) -> dict[str, Any]:
        batches = build_batch_summaries(self.rows)
        heights = build_height_summaries(self.rows)
        field_positions = build_field_position_summaries(self.rows)
        expected_count = (
            len(self.config.heights_mm) * len(DEFAULT_FIELD_GRID) * self.config.frames_per_position
            if self.config.field_grid
            else len(self.config.heights_mm) * self.config.frames_per_height
        )
        report = {
            "record_type": (
                "charuco_multi_height_field_grid_error_experiment"
                if self.config.field_grid
                else "charuco_multi_height_camera_error_experiment"
            ),
            "run_id": self.run_id,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": status,
            "error": error,
            "absolute_accuracy_claimed": False,
            "interpretation": (
                "本报告分析RGB-PnP、对齐深度和二者交叉差异的内部一致性与重复性；"
                "没有外部长度真值，不代表相机绝对测量精度。"
            ),
            "height_definition": "T_rgb_board[2,3], board origin Z in RGB optical frame, mm",
            "control_residual_is_accuracy_metric": False,
            "configuration": asdict(self.config),
            "board": {
                "pattern_width_mm": BOARD_CFG.pattern_width_mm,
                "pattern_height_mm": BOARD_CFG.pattern_height_mm,
                "squares_x": BOARD_CFG.squares_x,
                "squares_y": BOARD_CFG.squares_y,
                "square_length_mm": BOARD_CFG.square_length_mm,
                "marker_length_mm": BOARD_CFG.marker_length_mm,
                "aruco_dict_name": BOARD_CFG.aruco_dict_name,
            },
            "gates": {
                "rgb_reprojection_rmse_px_max": BOARD_CFG.max_rgb_reprojection_rmse_px,
                "rgb_reprojection_max_px_max": BOARD_CFG.max_rgb_reprojection_error_px,
                "depth_plane_rmse_mm_max": BOARD_CFG.max_plane_rmse_mm,
            },
            "camera": self.device_identity,
            "intrinsics": self.intrinsics,
            "stream_configuration": {
                "requested_color_width": CAMERA_CFG.preferred_color_width,
                "requested_color_height": CAMERA_CFG.preferred_color_height,
                "requested_color_fps": CAMERA_CFG.preferred_color_fps,
                "preferred_color_formats": list(CAMERA_CFG.preferred_color_formats),
                "align_depth_to_color": CAMERA_CFG.align_to_color,
                "actual_profiles": self.actual_stream_profiles,
                "first_frame_metadata": self.first_frame_metadata,
            },
            "exposure_lock": self.exposure,
            "frame_count": len(self.rows),
            "expected_frame_count": expected_count,
            "field_grid": [
                {
                    "position": name, "u_norm": u_norm, "v_norm": v_norm,
                    "target_u_px": u_norm * self.config.grid_image_width,
                    "target_v_px": v_norm * self.config.grid_image_height,
                }
                for name, u_norm, v_norm in DEFAULT_FIELD_GRID
            ] if self.config.field_grid else [],
            "field_position_summaries": field_positions,
            "batch_summaries": batches,
            "height_summaries": heights,
            "height_trends": trend_summary(heights),
            "frames": self.rows,
        }
        write_csv(self.raw_csv, self.rows, CSV_FIELDS)
        write_csv(
            self.output_dir / "height_summary.csv",
            [_flatten_summary(item) for item in heights],
        )
        if field_positions:
            write_csv(
                self.output_dir / "field_position_summary.csv",
                [
                    {
                        "height_target_mm": item["height_target_mm"],
                        "grid_position": item["grid_position"],
                        "grid_row": item["grid_row"],
                        "grid_col": item["grid_col"],
                        "grid_target_u_px": item["grid_target_u_px"],
                        "grid_target_v_px": item["grid_target_v_px"],
                        "frame_count": item["frame_count"],
                        "valid_count": item["valid_count"],
                        "valid_ratio": item["valid_ratio"],
                        **{
                            f"{metric}_{stat_name}": stats.get(stat_name)
                            for metric, stats in item["statistics"].items()
                            for stat_name in ("mean", "median", "std", "mad", "p95", "range")
                        },
                    }
                    for item in field_positions
                ],
            )
        write_csv(
            self.output_dir / "batch_summary.csv",
            [
                {
                    "height_target_mm": item.get("height_target_mm"),
                    "batch_index": item.get("batch_index"),
                    "frame_count": item.get("frame_count"),
                    "valid_count": item.get("valid_count"),
                    "valid_ratio": item.get("valid_ratio"),
                    "rgb_xyz_median_mm": json.dumps(item.get("rgb", {}).get("xyz_median_mm")),
                    "depth_xyz_median_mm": json.dumps(item.get("depth", {}).get("xyz_median_mm")),
                    "rgb_scatter_p95_mm": item.get("rgb", {}).get("xyz_scatter_p95_mm"),
                    "depth_scatter_p95_mm": item.get("depth", {}).get("xyz_scatter_p95_mm"),
                    "rgb_normal_scatter_p95_deg": item.get("rgb", {}).get("normal_scatter_p95_deg"),
                    "depth_normal_scatter_p95_deg": item.get("depth", {}).get("normal_scatter_p95_deg"),
                    "cross_dz_median_mm": item.get("cross_dz_mm", {}).get("median"),
                    "cross_distance_p95_mm": item.get("cross_distance_mm", {}).get("p95"),
                }
                for item in batches
            ],
        )
        report["plot_outputs"] = (
            write_plots(self.output_dir, heights, self.rows)
            + write_field_grid_plots(self.output_dir, field_positions)
        )
        atomic_write_json(self.output_dir / "report.json", report)
        return report

    def run(self) -> int:
        error: str | None = None
        try:
            self.start()
            if not self.config.execute_motion:
                current = self.current_rgb_z()
                snapshot = self.motion.snapshot() if self.motion is not None else {}
                validate_robot_ready(snapshot)
                print("[PREVIEW] 未指定 --execute-motion，不发送任何运动。")
                print(f"[PREVIEW] 当前PnP Z={current:.3f} mm，当前TCP Z={snapshot['tcp_pose_m_rad'][2] * 1000.0:.3f} mm")
                for target in self.config.heights_mm:
                    residual = float(target - current)
                    steps = int(math.ceil(abs(residual) / self.config.max_step_mm))
                    print(
                        f"[PREVIEW] 目标 {target:.1f} mm：首步建议TCP Z "
                        f"{np.clip(-residual, -self.config.max_step_mm, self.config.max_step_mm):+.3f} mm，"
                        f"预计至少 {steps} 步人工确认"
                    )
                if self.config.field_grid:
                    print(
                        f"[PREVIEW] 视野模式：每个高度9个人工XY位置，每位置"
                        f"{self.config.frames_per_position}帧；程序不会发送XY运动。"
                    )
                    for name, u_norm, v_norm in DEFAULT_FIELD_GRID:
                        print(
                            f"[PREVIEW] {name}: 归一化({u_norm:.2f},{v_norm:.2f})，"
                            f"目标像素约({u_norm*self.config.grid_image_width:.0f},"
                            f"{v_norm*self.config.grid_image_height:.0f})"
                        )
                self.finalize("motion_preview")
                print(f"[PREVIEW] 预览记录目录：{self.output_dir}")
                return 0
            if self.config.field_grid:
                self.run_field_grid()
            else:
                for target in self.config.heights_mm:
                    print("\n" + "=" * 76)
                    input(f"准备进入 {target:.0f} mm 高度。确认标定板固定、路径安全后按 Enter：")
                    reached = self.reach_height(target)
                    print(f"[READY] 目标={target:.1f} mm，当前PnP Z={reached:.3f} mm")
                    input("确认画面、光照和机器人状态正常，按 Enter 开始采集：")
                    self.capture_height(target)
                    input(f"{target:.0f} mm 已完成。检查现场后按 Enter 进入下一高度：")
            self.finalize("completed")
            print(f"[DONE] 报告目录：{self.output_dir}")
            return 0
        except KeyboardInterrupt:
            error = "KeyboardInterrupt"
            if self.motion is not None:
                try:
                    self.motion.stop_motion()
                except Exception:
                    pass
            self.finalize("interrupted", error)
            return 130
        except Exception as exc:
            error = str(exc)
            self.finalize("failed", error)
            print(f"[ERROR] {error}")
            print(f"[PARTIAL] 已保存部分报告：{self.output_dir}")
            return 2
        finally:
            self.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="固定ChArUco板，机器人保持XY/姿态、只改TCP Z的多高度相机误差实验"
    )
    parser.add_argument("--heights-mm", type=float, nargs="+", default=list(DEFAULT_HEIGHTS_MM))
    parser.add_argument(
        "--rgb-precision-range", action="store_true",
        help="运行纯RGB机器人相对精度范围实验，不启动Depth或PointCloud",
    )
    parser.add_argument("--rgb-height-start-mm", type=float, default=200.0)
    parser.add_argument("--rgb-height-stop-mm", type=float, default=300.0)
    parser.add_argument("--rgb-height-step-mm", type=float, default=10.0)
    parser.add_argument("--rgb-center-step-mm", type=float, default=20.0)
    parser.add_argument("--rgb-confirm-tolerance-px", type=float, default=20.0)
    parser.add_argument("--rgb-min-tcp-step-mm", type=float, default=15.0)
    parser.add_argument("--rgb-max-tcp-step-mm", type=float, default=30.0)
    parser.add_argument("--rgb-max-tcp-z-drift-mm", type=float, default=0.2)
    parser.add_argument("--rgb-max-tcp-rotation-drift-deg", type=float, default=0.05)
    parser.add_argument("--rgb-valid-ratio-min", type=float, default=0.95)
    parser.add_argument("--rgb-distance-p95-max-mm", type=float, default=0.5)
    parser.set_defaults(rgb_auto_xy=True)
    parser.add_argument(
        "--rgb-auto-xy", dest="rgb_auto_xy", action="store_true",
        help="RGB精度实验使用机器人自动20 mm XY五点（默认）",
    )
    parser.add_argument(
        "--rgb-manual-xy", dest="rgb_auto_xy", action="store_false",
        help="改回人工按图像目标移动XY五点",
    )
    parser.add_argument("--frames-per-height", type=int, default=200)
    parser.set_defaults(field_grid=True)
    parser.add_argument(
        "--field-grid", dest="field_grid", action="store_true",
        help="启用默认模式：每个高度做3x3人工XY视野网格",
    )
    parser.add_argument(
        "--height-only", dest="field_grid", action="store_false",
        help="关闭视野网格，恢复每个高度直接采frames-per-height帧",
    )
    parser.add_argument(
        "--frames-per-position", type=int, default=20,
        help="3x3视野模式下每个网格位置采集帧数，默认20",
    )
    parser.add_argument("--grid-image-width", type=int, default=1280)
    parser.add_argument("--grid-image-height", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--warmup-frames", type=int, default=90)
    parser.add_argument("--target-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--max-step-mm", type=float, default=70.0)
    parser.add_argument("--max-corrections", type=int, default=3)
    parser.add_argument("--speed-mm-s", type=float, default=20.0)
    parser.add_argument("--acc-mm-s2", type=float, default=30.0)
    parser.add_argument("--motion-timeout-s", type=float, default=90.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=0.50)
    parser.add_argument("--rotation-tolerance-rad", type=float, default=0.002)
    parser.add_argument(
        "--camera-serial", default=ROBOT_CAMERA_INTEGRATION_CFG.production_camera_serial,
        help="必须匹配的相机序列号；传空字符串可关闭检查（不推荐）",
    )
    parser.add_argument("--depth-filter-mode", choices=("none", "temporal"), default="none")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--robot-ip", default=ROBOT_CFG.ip)
    parser.add_argument("--robot-port", type=int, default=ROBOT_CFG.rpc_port)
    parser.add_argument("--robot-user", default=ROBOT_CFG.user)
    parser.add_argument("--robot-password", default=ROBOT_CFG.password)
    parser.add_argument("--robot-timeout-ms", type=int, default=ROBOT_CFG.request_timeout_ms)
    parser.add_argument(
        "--execute-motion", action="store_true",
        help="允许逐次人工确认后发送moveLine；不指定时只测量并预览所需Z修正",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ROBOT_CFG.ip = args.robot_ip
    ROBOT_CFG.rpc_port = args.robot_port
    ROBOT_CFG.user = args.robot_user
    ROBOT_CFG.password = args.robot_password
    ROBOT_CFG.request_timeout_ms = args.robot_timeout_ms
    if args.rgb_precision_range:
        rgb_config = RgbPrecisionRangeConfig(
            heights_mm=make_height_schedule(
                args.rgb_height_start_mm,
                args.rgb_height_stop_mm,
                args.rgb_height_step_mm,
            ),
            frames_per_position=args.frames_per_position,
            batch_size=args.batch_size,
            warmup_frames=args.warmup_frames,
            center_step_mm=args.rgb_center_step_mm,
            confirm_tolerance_px=args.rgb_confirm_tolerance_px,
            min_tcp_step_mm=args.rgb_min_tcp_step_mm,
            max_tcp_step_mm=args.rgb_max_tcp_step_mm,
            max_tcp_z_drift_mm=args.rgb_max_tcp_z_drift_mm,
            max_tcp_rotation_drift_deg=args.rgb_max_tcp_rotation_drift_deg,
            minimum_valid_ratio=args.rgb_valid_ratio_min,
            maximum_distance_p95_mm=args.rgb_distance_p95_max_mm,
            target_tolerance_mm=args.target_tolerance_mm,
            max_step_mm=args.max_step_mm,
            max_corrections=args.max_corrections,
            speed_mm_s=args.speed_mm_s,
            acceleration_mm_s2=args.acc_mm_s2,
            motion_timeout_s=args.motion_timeout_s,
            position_tolerance_mm=args.position_tolerance_mm,
            rotation_tolerance_rad=args.rotation_tolerance_rad,
            expected_camera_serial=args.camera_serial,
            execute_motion=args.execute_motion,
            auto_xy=args.rgb_auto_xy,
        )
        return RgbPrecisionRangeExperiment(rgb_config, args.output_root).run()
    config = HeightExperimentConfig(
        heights_mm=tuple(args.heights_mm),
        frames_per_height=args.frames_per_height,
        batch_size=args.batch_size,
        warmup_frames=args.warmup_frames,
        target_tolerance_mm=args.target_tolerance_mm,
        max_step_mm=args.max_step_mm,
        max_corrections=args.max_corrections,
        speed_mm_s=args.speed_mm_s,
        acceleration_mm_s2=args.acc_mm_s2,
        motion_timeout_s=args.motion_timeout_s,
        position_tolerance_mm=args.position_tolerance_mm,
        rotation_tolerance_rad=args.rotation_tolerance_rad,
        expected_camera_serial=args.camera_serial,
        depth_filter_mode=args.depth_filter_mode,
        execute_motion=args.execute_motion,
        field_grid=args.field_grid,
        frames_per_position=args.frames_per_position,
        grid_image_width=args.grid_image_width,
        grid_image_height=args.grid_image_height,
    )
    return HeightErrorExperiment(config, args.output_root).run()


if __name__ == "__main__":
    raise SystemExit(main())
