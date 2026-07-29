#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO + RGB-D + RGB 手眼标定的“眼在手上”孔中心定位脚本。

流程：YOLO 找孔 -> 鼠标点击选择一个孔 -> 用孔外环带的深度拟合局部平面
得到 RGB 相机坐标 -> T_tcp_rgb_camera -> 当前 TCP 位姿 -> 机器人基坐标。

默认进入两阶段流程并启用实验运动；原点和大幅移动需要输入 m，小幅修正自动执行。
默认允许使用当前实验手眼结果，仅适用于现场诊断，不代表生产授权。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aubo_workbench.camera import (  # noqa: E402
    get_aligned_frame_bundle,
    get_device_identity,
    get_rgb_frame_bundle,
    init_pipeline,
    init_rgb_handeye_pipeline,
)
from aubo_workbench.charuco_point_experiment import load_handeye_experiment_result  # noqa: E402
from aubo_workbench.config import ROBOT_CFG  # noqa: E402
from aubo_workbench.geometry import invert_transform, make_transform, transform_to_pose6_rzryrx  # noqa: E402


DEFAULT_MODEL = Path(r"C:\MM\models\small_silu.pt")
DEFAULT_HANDEYE = Path(r"C:\MM\aubo_tools\data\e7_candidates\e7_handeye_candidate_current.json")
WINDOW = "YOLO eye-in-hand hole selection (click hole, Enter=confirm, Esc=quit)"
RUNS_DIR = ROOT.parent / "data" / "hole_localization_runs"
HOLE_DIAMETERS_MM = (65.0, 70.0, 75.0)
FINAL_BASE_Y_AFTER_Z_MM = 0.2

# 当前使用的 ChArUco 9 点 XY 仿射模型；不启用工作区限制。
CHARUCO_XY_MODEL_MATRIX = np.array([
    [1.0008588303603987, -0.0011632075996384716],
    [-0.0013284394231714038, 0.999581433830417],
], dtype=np.float64)
CHARUCO_XY_MODEL_BIAS_MM = np.array([0.05229713949213546, 3.1959138367376676], dtype=np.float64)
CHARUCO_XY_MODEL_SOURCE = Path(
    r"C:\MM\aubo_tools\data\tcp_absolute_xy_model\charuco-tcp-xy-20260727_174559\report.json"
)

# PyCharm 直接运行的默认模式：不需要额外命令行参数。
# 原点和大幅移动仍显示目标并要求输入 m；小幅闭环修正自动执行。
DEFAULT_TWO_STAGE_HOLE_LOCALIZATION = True
DEFAULT_EXECUTE_MOTION = True
DEFAULT_ALLOW_EXPERIMENTAL_HANDEYE = True


@dataclass(frozen=True)
class TwoStageConfig:
    coarse_height_mm: float = 340.0
    fine_height_mm: float = 260.0
    coarse_frames: int = 15
    fine_frames: int = 30
    height_tolerance_mm: float = 2.0
    center_tolerance_px: float = 5.0
    # 当前 Gemini 深度平面法向跨帧/跨视角重复性约 1–2°；粗阶段不应追逐到 0.5°。
    # 精定位仍锁定最终粗姿态并使用 RGB 进行 XY 微调。
    normal_tolerance_deg: float = 2.0
    max_z_corrections: int = 4
    min_coarse_valid: int = 12
    min_fine_valid: int = 24
    # 球面伞具的初始环带深度允许少量结构化噪声；粗/精阶段门限保持不变。
    initial_max_plane_rmse_mm: float = 3.5
    max_plane_rmse_mm: float = 3.5
    coarse_settle_frames: int = 10
    coarse_max_attempt_multiplier: int = 6
    max_ellipse_residual_px: float = 0.9
    min_coarse_ellipse_coverage_deg: float = 120.0
    min_ellipse_coverage_deg: float = 200.0
    max_fine_center_scatter_p95_px: float = 0.35
    enable_tilt_center_correction: bool = True
    tilt_correction_iterations: int = 4
    tilt_correction_samples: int = 240
    max_tilt_correction_mm: float = 3.0


@dataclass
class PlaneEstimate:
    point_camera_mm: np.ndarray
    normal_camera: np.ndarray
    rmse_mm: float
    ring_points: int
    surface_model: str = "sphere"
    sphere_center_camera_mm: np.ndarray | None = None
    sphere_radius_mm: float | None = None


@dataclass
class Observation:
    stage: str
    frame_index: int
    center_px: np.ndarray
    ellipse: dict[str, Any] | None = None
    plane: PlaneEstimate | None = None
    timestamp_ns: int | None = None
    error: str | None = None


def load_yolo(model_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("缺少 ultralytics，请安装：pip install ultralytics") from exc
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO 模型不存在: {model_path}")
    return YOLO(str(model_path))


def detect(model: Any, image: np.ndarray, confidence: float) -> list[dict[str, Any]]:
    result = model.predict(source=image, conf=confidence, verbose=False)[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    names = getattr(result, "names", {})
    output = []
    for box, score, cls in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = [float(v) for v in box]
        output.append({
            "box": [x1, y1, x2, y2], "confidence": float(score), "class_id": int(cls),
            "class_name": str(names.get(int(cls), cls)) if isinstance(names, dict) else str(cls),
            "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
        })
    return output


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 12:
        raise ValueError("孔周围有效深度点不足，无法拟合平面")
    center = np.median(points, axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    if normal[2] > 0:
        normal = -normal
    residual = np.abs((points - center) @ normal)
    keep = residual <= max(1.0, float(np.percentile(residual, 85)) * 2.5)
    if keep.sum() >= 12:
        inliers = points[keep]
        center = np.mean(inliers, axis=0)
        _, _, vh = np.linalg.svd(inliers - center, full_matrices=False)
        normal = vh[-1]
        if normal[2] > 0:
            normal = -normal
        # 用剔除外点后的最终平面重新计算残差；否则 rmse 对应的是第一轮
        # (未剔除外点的) 平面，和实际返回的 normal/center 不一致，会让
        # plane_rmse_mm 质量门形同虚设。
        residual = np.abs((inliers - center) @ normal)
    else:
        residual = residual[keep]
    return normal / np.linalg.norm(normal), float(np.sqrt(np.mean(residual ** 2)))


def fit_sphere(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """鲁棒拟合未知半径球面，返回球心、半径和径向RMSE。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 20:
        raise ValueError("球面拟合有效深度点不足")
    origin = np.median(pts, axis=0)
    active = np.ones(len(pts), dtype=bool)
    center = radius = None
    for _ in range(5):
        work = pts[active]
        q = work - origin
        A = np.column_stack((2.0 * q, np.ones(len(q))))
        b = np.sum(q * q, axis=1)
        solution, *_ = np.linalg.lstsq(A, b, rcond=None)
        center_local = solution[:3]
        radius_sq = float(solution[3] + center_local @ center_local)
        if not math.isfinite(radius_sq) or radius_sq <= 0.0:
            raise ValueError("球面半径估计无效")
        center = origin + center_local
        radius = math.sqrt(radius_sq)
        residual = np.abs(np.linalg.norm(pts - center, axis=1) - radius)
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        threshold = max(2.0, median + 4.0 * 1.4826 * max(mad, 1e-6))
        new_active = residual <= threshold
        if new_active.sum() < 20 or np.array_equal(new_active, active):
            active = new_active if new_active.sum() >= 20 else active
            break
        active = new_active
    if center is None or radius is None:
        raise ValueError("球面拟合失败")
    final_residual = np.linalg.norm(pts[active] - center, axis=1) - radius
    return center, float(radius), float(np.sqrt(np.mean(final_residual ** 2)))


def hole_camera_point(
    ring_center_xy: tuple[float, float],
    xyz_map_mm: np.ndarray,
    intrinsics: Any,
    radius_px: float,
    *,
    ray_center_xy: tuple[float, float] | np.ndarray | None = None,
    ray_center_is_undistorted: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """用孔外环带拟合平面，再将孔中心射线与平面求交。

    ring_center_xy 必须是原始图像坐标，用于在 aligned xyz_map 中选择深度环带。
    ray_center_xy 可以是去畸变轮廓拟合得到的中心；这样深度采样坐标和几何射线
    分别使用各自正确的坐标域，避免把去畸变像素直接拿去索引原始深度图。
    """
    ring_u, ring_v = map(float, ring_center_xy)
    h, w = xyz_map_mm.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - ring_u, yy - ring_v)
    ring = (rr >= max(4.0, radius_px * 1.08)) & (rr <= max(8.0, radius_px * 1.35))
    points = xyz_map_mm[ring]
    points = points[np.isfinite(points).all(axis=1)]
    points = points[(points[:, 2] > 100.0) & (points[:, 2] < 3000.0)]
    sphere_center, sphere_radius, rmse = fit_sphere(points)

    ray_center = np.asarray(
        ring_center_xy if ray_center_xy is None else ray_center_xy,
        dtype=np.float64,
    ).reshape(2)
    ray = camera_ray(intrinsics, ray_center, already_undistorted=ray_center_is_undistorted)
    b = -2.0 * float(sphere_center @ ray)
    c = float(sphere_center @ sphere_center - sphere_radius * sphere_radius)
    discriminant = b * b - 4.0 * c
    if discriminant <= 0.0:
        raise ValueError("孔中心视线未与拟合球面相交")
    roots = [(-b - math.sqrt(discriminant)) / 2.0, (-b + math.sqrt(discriminant)) / 2.0]
    positive_roots = [value for value in roots if value > 0.0 and math.isfinite(value)]
    if not positive_roots:
        raise ValueError("球面交点位于相机反向")
    scale = min(positive_roots)
    point = ray * scale
    if not np.isfinite(point).all() or point[2] <= 0:
        raise ValueError("孔中心反投影得到无效深度")
    normal = _unit(point - sphere_center, "sphere surface normal")
    if normal[2] > 0:
        normal = -normal
    return point, {
        "ring_points": int(len(points)),
        "plane_rmse_mm": rmse,
        "plane_normal_camera": normal.tolist(),
        "plane_point_camera_mm": point.tolist(),
        "surface_model": "sphere",
        "sphere_center_camera_mm": sphere_center.tolist(),
        "sphere_radius_mm": sphere_radius,
        "ring_center_px_distorted": [ring_u, ring_v],
        "ray_center_px": ray_center.tolist(),
        "ray_center_is_undistorted": bool(ray_center_is_undistorted),
    }


def choose_box(image: np.ndarray, detections: list[dict[str, Any]]) -> int | None:
    selected = [0]

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        distances = [np.hypot(x - d["center"][0], y - d["center"][1]) for d in detections]
        if distances:
            selected[0] = int(np.argmin(distances))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        view = image.copy()
        for i, d in enumerate(detections):
            x1, y1, x2, y2 = map(int, d["box"])
            color = (0, 255, 0) if i == selected[0] else (0, 180, 255)
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
            cv2.circle(view, tuple(map(int, d["center"])), 4, color, -1)
            cv2.putText(view, f"{i}: {d['class_name']} {d['confidence']:.2f}",
                        (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(view, "click hole, Enter confirm, Esc quit",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):
            return selected[0] if detections else None
        if key == 27:
            return None


def _load_intrinsics(path: Path) -> Any:
    from aubo_workbench.camera import CameraIntrinsics
    data = json.loads(path.read_text(encoding="utf-8"))
    return CameraIntrinsics(int(data["width"]), int(data["height"]), float(data["fx"]),
                            float(data["fy"]), float(data["cx"]), float(data["cy"]),
                            tuple(data.get("distortion", [])))


def _unit(vec: np.ndarray, label: str = "vector") -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(vec))
    if not np.isfinite(length) or length < 1e-9:
        raise ValueError(f"{label} 无法归一化")
    return vec / length


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.degrees(math.acos(np.clip(float(_unit(a) @ _unit(b)), -1.0, 1.0))))


def _matrix_to_rpy_zyx(R: np.ndarray) -> np.ndarray:
    """AUBO 使用的 [rx, ry, rz]（Rz @ Ry @ Rx）逆变换。"""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    sy = float(-R[2, 0])
    ry = math.asin(float(np.clip(sy, -1.0, 1.0)))
    cy = math.cos(ry)
    if abs(cy) > 1e-7:
        rx = math.atan2(float(R[2, 1]), float(R[2, 2]))
        rz = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        rx = math.atan2(float(-R[1, 2]), float(R[1, 1]))
        rz = 0.0
    return np.array([rx, ry, rz], dtype=np.float64)


def transform_to_sdk_pose_m_rad(T_base_tcp: np.ndarray) -> list[float]:
    T_base_tcp = np.asarray(T_base_tcp, dtype=np.float64).reshape(4, 4)
    return ((T_base_tcp[:3, 3] / 1000.0).tolist()
            + _matrix_to_rpy_zyx(T_base_tcp[:3, :3]).tolist())


def camera_transform(T_base_tcp: np.ndarray, T_tcp_camera: np.ndarray) -> np.ndarray:
    return np.asarray(T_base_tcp, dtype=np.float64) @ np.asarray(T_tcp_camera, dtype=np.float64)


def _camera_matrix(intrinsics: Any) -> np.ndarray:
    return np.asarray(
        [[intrinsics.fx, 0.0, intrinsics.cx],
         [0.0, intrinsics.fy, intrinsics.cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _distortion(intrinsics: Any) -> np.ndarray:
    values = np.asarray(getattr(intrinsics, "distortion", ()), dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        return np.zeros(0, dtype=np.float64)
    return values


def undistort_pixels(intrinsics: Any, points_px: np.ndarray, *, pixel_output: bool = True) -> np.ndarray:
    """把原始畸变像素转换成去畸变像素或归一化坐标。"""
    points = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
    distortion = _distortion(intrinsics)
    if distortion.size == 0 or not np.any(np.abs(distortion) > 1e-12):
        if pixel_output:
            return points.reshape(-1, 2)
        K = _camera_matrix(intrinsics)
        flat = points.reshape(-1, 2)
        return np.column_stack(((flat[:, 0] - K[0, 2]) / K[0, 0],
                                (flat[:, 1] - K[1, 2]) / K[1, 1]))
    P = _camera_matrix(intrinsics) if pixel_output else None
    result = cv2.undistortPoints(points, _camera_matrix(intrinsics), distortion.reshape(1, -1), P=P)
    return result.reshape(-1, 2)


def camera_ray(intrinsics: Any, center_px: np.ndarray, *, already_undistorted: bool = False) -> np.ndarray:
    u, v = np.asarray(center_px, dtype=np.float64).reshape(2)
    if already_undistorted:
        x = (u - intrinsics.cx) / intrinsics.fx
        y = (v - intrinsics.cy) / intrinsics.fy
    else:
        x, y = undistort_pixels(intrinsics, np.asarray([[u, v]]), pixel_output=False)[0]
    return _unit(np.array([x, y, 1.0], dtype=np.float64), "pixel ray")


def ray_plane_intersection(ray_origin: np.ndarray, ray_direction: np.ndarray,
                           plane_point: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    denom = float(np.asarray(plane_normal) @ np.asarray(ray_direction))
    if abs(denom) < 1e-8:
        raise ValueError("相机光线与孔面近乎平行")
    scale = float(np.asarray(plane_normal) @ (np.asarray(plane_point) - np.asarray(ray_origin))) / denom
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("孔面位于相机光线反向，拒绝计算")
    return np.asarray(ray_origin, dtype=np.float64) + scale * np.asarray(ray_direction, dtype=np.float64)


def pixel_to_base_plane(center_px: np.ndarray, intrinsics: Any, T_base_tcp: np.ndarray,
                        T_tcp_camera: np.ndarray, plane_point_base: np.ndarray,
                        plane_normal_base: np.ndarray, *, center_is_undistorted: bool = False) -> np.ndarray:
    T_base_camera = camera_transform(T_base_tcp, T_tcp_camera)
    return ray_plane_intersection(
        T_base_camera[:3, 3],
        T_base_camera[:3, :3] @ camera_ray(
            intrinsics, center_px, already_undistorted=center_is_undistorted,
        ),
        plane_point_base,
        plane_normal_base,
    )


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = _unit(normal, "circle plane normal")
    hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(hint @ n)) > 0.9:
        hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis_x = _unit(hint - n * float(hint @ n), "circle plane x")
    axis_y = _unit(np.cross(n, axis_x), "circle plane y")
    return axis_x, axis_y


def _project_undistorted_pixels(points_camera_mm: np.ndarray, intrinsics: Any) -> np.ndarray:
    points = np.asarray(points_camera_mm, dtype=np.float64).reshape(-1, 3)
    if np.any(points[:, 2] <= 1e-6):
        raise ValueError("圆轮廓投影中存在相机后方点")
    return np.column_stack((
        intrinsics.fx * points[:, 0] / points[:, 2] + intrinsics.cx,
        intrinsics.fy * points[:, 1] / points[:, 2] + intrinsics.cy,
    ))


def correct_projected_circle_center(
    observed_ellipse_center_px: np.ndarray,
    intrinsics: Any,
    T_base_tcp: np.ndarray,
    T_tcp_camera: np.ndarray,
    plane_point_base: np.ndarray,
    plane_normal_base: np.ndarray,
    diameter_mm: float,
    *,
    iterations: int = 4,
    samples: int = 240,
    max_correction_mm: float = 3.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """从投影椭圆中心反解真实圆心。

    不使用小角度近似。每次迭代在当前候选圆心处生成已知半径的三维圆，
    投影后拟合椭圆，计算“投影圆心→椭圆几何中心”的偏差并反向修正。
    输入中心必须是去畸变像素坐标。
    """
    if not math.isfinite(diameter_mm) or diameter_mm <= 0.0:
        raise ValueError("孔径必须是正数")
    if samples < 24:
        raise ValueError("tilt correction samples 不能小于24")

    T_base_camera = camera_transform(T_base_tcp, T_tcp_camera)
    R_base_camera = T_base_camera[:3, :3]
    t_base_camera = T_base_camera[:3, 3]
    R_camera_base = R_base_camera.T
    plane_point_camera = R_camera_base @ (np.asarray(plane_point_base, dtype=np.float64) - t_base_camera)
    plane_normal_camera = _unit(R_camera_base @ np.asarray(plane_normal_base, dtype=np.float64), "camera plane normal")

    observed = np.asarray(observed_ellipse_center_px, dtype=np.float64).reshape(2)
    candidate = ray_plane_intersection(
        np.zeros(3, dtype=np.float64),
        camera_ray(intrinsics, observed, already_undistorted=True),
        plane_point_camera,
        plane_normal_camera,
    )
    naive_camera = candidate.copy()
    radius = float(diameter_mm) / 2.0
    axis_x, axis_y = _plane_basis(plane_normal_camera)
    theta = np.linspace(0.0, 2.0 * math.pi, int(samples), endpoint=False)
    final_bias_px = np.zeros(2, dtype=np.float64)

    for _ in range(max(1, int(iterations))):
        circle = (
            candidate[None, :]
            + radius * np.cos(theta)[:, None] * axis_x[None, :]
            + radius * np.sin(theta)[:, None] * axis_y[None, :]
        )
        projected = _project_undistorted_pixels(circle, intrinsics)
        ellipse = cv2.fitEllipse(projected.astype(np.float32).reshape(-1, 1, 2))
        projected_ellipse_center = np.asarray(ellipse[0], dtype=np.float64)
        projected_circle_center = _project_undistorted_pixels(candidate.reshape(1, 3), intrinsics)[0]
        final_bias_px = projected_ellipse_center - projected_circle_center
        corrected_center_px = observed - final_bias_px
        updated = ray_plane_intersection(
            np.zeros(3, dtype=np.float64),
            camera_ray(intrinsics, corrected_center_px, already_undistorted=True),
            plane_point_camera,
            plane_normal_camera,
        )
        if float(np.linalg.norm(updated - candidate)) < 1e-6:
            candidate = updated
            break
        candidate = updated

    correction_camera = candidate - naive_camera
    correction_norm = float(np.linalg.norm(correction_camera))
    if correction_norm > float(max_correction_mm):
        raise RuntimeError(
            f"倾斜圆心修正过大：{correction_norm:.3f} mm > {max_correction_mm:.3f} mm，"
            "拒绝使用，需检查法向、孔径或轮廓"
        )

    naive_base = R_base_camera @ naive_camera + t_base_camera
    corrected_base = R_base_camera @ candidate + t_base_camera
    tilt_deg = float(math.degrees(math.acos(np.clip(abs(float(plane_normal_camera[2])), 0.0, 1.0))))
    return corrected_base, {
        "method": "iterative_projected_circle_center",
        "diameter_mm": float(diameter_mm),
        "tilt_deg": tilt_deg,
        "ellipse_center_bias_px": final_bias_px.tolist(),
        "correction_camera_mm": correction_camera.tolist(),
        "correction_base_mm": (corrected_base - naive_base).tolist(),
        "correction_norm_mm": correction_norm,
        "naive_point_base_mm": naive_base.tolist(),
        "corrected_point_base_mm": corrected_base.tolist(),
        "iterations": int(iterations),
        "samples": int(samples),
    }


def camera_height_to_plane_mm(T_base_tcp: np.ndarray, T_tcp_camera: np.ndarray,
                              plane_point_base: np.ndarray) -> float:
    T_base_camera = camera_transform(T_base_tcp, T_tcp_camera)
    return float(T_base_camera[:3, 2] @ (np.asarray(plane_point_base) - T_base_camera[:3, 3]))


def camera_orientation_from_hole_plane(plane_normal_base: np.ndarray,
                                       reference_camera_x_base: np.ndarray) -> np.ndarray:
    """构造光轴指向孔面的相机姿态，平面内滚转继承原点相机 X 方向。"""
    z_axis = -_unit(plane_normal_base, "hole plane normal")
    x_hint = np.asarray(reference_camera_x_base, dtype=np.float64).reshape(3)
    x_axis = x_hint - z_axis * float(x_hint @ z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_hint = np.array([1.0, 0.0, 0.0])
        x_axis = x_hint - z_axis * float(x_hint @ z_axis)
    x_axis = _unit(x_axis, "camera x axis")
    y_axis = _unit(np.cross(z_axis, x_axis), "camera y axis")
    return np.column_stack((x_axis, y_axis, z_axis))


def plan_centered_tcp_pose(plane_point_base: np.ndarray, plane_normal_base: np.ndarray,
                           reference_T_base_tcp: np.ndarray, T_tcp_camera: np.ndarray,
                           height_mm: float) -> np.ndarray:
    """让孔位于主点，光轴垂直本孔局部平面，返回 TCP 基坐标目标。"""
    reference_camera = camera_transform(reference_T_base_tcp, T_tcp_camera)
    R_base_camera = camera_orientation_from_hole_plane(
        plane_normal_base, reference_camera[:3, 0],
    )
    t_base_camera = np.asarray(plane_point_base, dtype=np.float64) - height_mm * R_base_camera[:, 2]
    return make_transform(R_base_camera, t_base_camera) @ invert_transform(T_tcp_camera)


def base_z_target_for_camera_height(T_base_tcp: np.ndarray, T_tcp_camera: np.ndarray,
                                    plane_point_base: np.ndarray, target_height_mm: float) -> tuple[np.ndarray, float]:
    """只改 TCP 基坐标 Z，使孔面在 RGB 坐标的估计 Z 达到目标高度。"""
    current_height = camera_height_to_plane_mm(T_base_tcp, T_tcp_camera, plane_point_base)
    z_axis_base_z = float(camera_transform(T_base_tcp, T_tcp_camera)[2, 2])
    if abs(z_axis_base_z) < 0.15:
        raise ValueError("相机光轴过于接近基坐标水平面，不能以基坐标 Z 控制高度")
    delta_base_z = (current_height - target_height_mm) / z_axis_base_z
    target = np.asarray(T_base_tcp, dtype=np.float64).copy()
    target[2, 3] += delta_base_z
    return target, current_height


def plan_final_tcp_xy(T_base_tcp: np.ndarray, hole_center_base: np.ndarray,
                      xy_offset_mm: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """只调整当前 TCP 的基坐标 XY，姿态（含 yaw）和 Z 保持不变。

    默认使用 ChArUco 9 点仿射模型；显式提供固定偏置时覆盖该模型。
    """
    target = np.asarray(T_base_tcp, dtype=np.float64).copy()
    visual_xy = np.asarray(hole_center_base, dtype=np.float64)[:2]
    if xy_offset_mm is None:
        target[:2, 3] = CHARUCO_XY_MODEL_MATRIX @ visual_xy + CHARUCO_XY_MODEL_BIAS_MM
    else:
        target[:2, 3] = visual_xy + np.asarray(xy_offset_mm, dtype=np.float64)
    return target, np.asarray(T_base_tcp, dtype=np.float64)[:3, 3].copy()


def plan_final_tcp_base_z(T_base_tcp: np.ndarray, hole_center_base: np.ndarray) -> np.ndarray:
    """保持当前 TCP 的 XY 和姿态，仅令基坐标 Z 等于孔中心基坐标 Z。"""
    target = np.asarray(T_base_tcp, dtype=np.float64).copy()
    target[2, 3] = float(np.asarray(hole_center_base, dtype=np.float64).reshape(3)[2])
    return target


def plan_final_tcp_base_y_trim(T_base_tcp: np.ndarray, delta_y_mm: float = FINAL_BASE_Y_AFTER_Z_MM) -> np.ndarray:
    """保持当前 TCP 的 X、Z 和姿态，仅沿基坐标 Y 做最终微调。"""
    target = np.asarray(T_base_tcp, dtype=np.float64).copy()
    target[1, 3] += float(delta_y_mm)
    return target


def fit_hole_ellipse(
    image_bgr: np.ndarray,
    detection: dict[str, Any],
    intrinsics: Any | None = None,
) -> dict[str, Any] | None:
    """YOLO仅给ROI；将轮廓点去畸变后再拟合椭圆。

    center_px / axes_px / angle_deg 位于去畸变像素域，用于几何计算。
    *_distorted 字段位于原始图像域，仅用于深度环带索引和显示。
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in detection["box"]]
    pad = 0.18 * max(x2 - x1, y2 - y1)
    xa, ya = max(0, int(math.floor(x1 - pad))), max(0, int(math.floor(y1 - pad)))
    xb, yb = min(w, int(math.ceil(x2 + pad))), min(h, int(math.ceil(y2 + pad)))
    if xb - xa < 24 or yb - ya < 24:
        return None

    gray = cv2.cvtColor(image_bgr[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    lo, hi = int(max(5.0, 0.66 * median)), int(min(250.0, 1.33 * median + 20.0))
    edges = cv2.Canny(gray, lo, max(lo + 10, hi))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    expected_distorted = np.asarray(detection["center"], dtype=np.float64)
    expected_undistorted = (
        undistort_pixels(intrinsics, expected_distorted.reshape(1, 2), pixel_output=True)[0]
        if intrinsics is not None else expected_distorted
    )
    best: dict[str, Any] | None = None

    for contour in contours:
        if len(contour) < 30:
            continue
        local_points = contour.reshape(-1, 2).astype(np.float64)
        global_distorted = local_points + np.array([xa, ya], dtype=np.float64)
        try:
            raw_ellipse = cv2.fitEllipse(global_distorted.astype(np.float32).reshape(-1, 1, 2))
        except cv2.error:
            continue
        raw_center = np.asarray(raw_ellipse[0], dtype=np.float64)
        raw_axes = np.asarray(raw_ellipse[1], dtype=np.float64)
        raw_width, raw_height = float(raw_axes[0]), float(raw_axes[1])
        raw_major, raw_minor = max(raw_width, raw_height), min(raw_width, raw_height)
        if raw_minor < 12.0 or raw_major / raw_minor > 1.60:
            continue

        undistorted_points = (
            undistort_pixels(intrinsics, global_distorted, pixel_output=True)
            if intrinsics is not None else global_distorted
        )
        try:
            (center_tuple, axes_tuple, angle) = cv2.fitEllipse(
                undistorted_points.astype(np.float32).reshape(-1, 1, 2)
            )
        except cv2.error:
            continue
        center = np.asarray(center_tuple, dtype=np.float64)
        axes = np.asarray(axes_tuple, dtype=np.float64)
        fit_width, fit_height = float(axes[0]), float(axes[1])
        major, minor = max(fit_width, fit_height), min(fit_width, fit_height)
        if minor < 12.0 or major / minor > 1.45:
            continue

        theta = math.radians(float(angle))
        c, s = math.cos(theta), math.sin(theta)
        dx = undistorted_points[:, 0] - center[0]
        dy = undistorted_points[:, 1] - center[1]
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        # 使用OpenCV返回的轴顺序和角度计算残差，避免把长短轴排序后与angle错配。
        radial = np.sqrt((local_x / (fit_width / 2.0)) ** 2 + (local_y / (fit_height / 2.0)) ** 2)
        residual = float(np.median(np.abs(radial - 1.0)) * (fit_width + fit_height) / 4.0)
        angles = np.mod(
            np.degrees(np.arctan2(local_y / (fit_height / 2.0), local_x / (fit_width / 2.0))),
            360.0,
        )
        bins = np.unique(np.floor(angles / 10.0).astype(int))
        coverage = float(len(bins) * 10.0)
        center_distance = float(np.linalg.norm(center - expected_undistorted))
        score = center_distance / max(major, 1.0) + residual / 2.0 + abs(1.0 - minor / major)

        legacy_center_undistorted = (
            undistort_pixels(intrinsics, raw_center.reshape(1, 2), pixel_output=True)[0]
            if intrinsics is not None else raw_center
        )
        item = {
            "center_px": center.tolist(),
            "axes_px": [major, minor],
            "angle_deg": float(angle),
            "center_px_distorted": raw_center.tolist(),
            "axes_px_distorted": [raw_width, raw_height],
            "axes_px_distorted_major_minor": [raw_major, raw_minor],
            "angle_deg_distorted": float(raw_ellipse[2]),
            "legacy_center_px_undistorted": legacy_center_undistorted.tolist(),
            "contour_undistortion_shift_px": (center - legacy_center_undistorted).tolist(),
            "residual_px": residual,
            "coverage_deg": coverage,
            "roundness": float(minor / major),
            "contour_points": int(len(contour)),
            "score": score,
            "center_coordinate_domain": "undistorted_pixel",
        }
        if best is None or score < best["score"]:
            best = item
    return best


def _nearest_detection(detections: list[dict[str, Any]], anchor_px: np.ndarray,
                       class_id: int | None = None) -> dict[str, Any] | None:
    if class_id is not None:
        same_class = [item for item in detections if item["class_id"] == class_id]
        if same_class:
            detections = same_class
    if not detections:
        return None
    return min(detections, key=lambda item: float(np.linalg.norm(np.asarray(item["center"]) - anchor_px)))


def _ellipse_ok(ellipse: dict[str, Any] | None, cfg: TwoStageConfig,
                min_coverage_deg: float | None = None) -> bool:
    coverage_gate = cfg.min_ellipse_coverage_deg if min_coverage_deg is None else float(min_coverage_deg)
    return bool(ellipse and ellipse["residual_px"] <= cfg.max_ellipse_residual_px
                and ellipse["coverage_deg"] >= coverage_gate
                and ellipse["roundness"] >= 0.70)


def _fuse_vectors(vectors: list[np.ndarray], label: str) -> np.ndarray:
    if not vectors:
        raise ValueError(f"没有可融合的 {label}")
    values = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    median = np.median(values, axis=0)
    distances = np.linalg.norm(values - median, axis=1)
    keep = distances <= max(1e-6, float(np.percentile(distances, 85)) * 2.5)
    return np.median(values[keep], axis=0)


def _fuse_normals(normals: list[np.ndarray]) -> np.ndarray:
    if not normals:
        raise ValueError("没有可融合的孔面法向")
    reference = _unit(normals[0], "normal")
    aligned = [(_unit(item, "normal") if float(_unit(item, "normal") @ reference) >= 0 else -_unit(item, "normal"))
               for item in normals]
    return _unit(np.median(np.asarray(aligned), axis=0), "fused normal")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row}) if rows else ["stage", "frame_index", "error"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


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


def _overlay(image: np.ndarray, detection: dict[str, Any] | None,
             ellipse: dict[str, Any] | None, title: str) -> np.ndarray:
    view = image.copy()
    if detection is not None:
        x1, y1, x2, y2 = map(int, detection["box"])
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 180, 255), 2)
    if ellipse is not None:
        display_center = ellipse.get("center_px_distorted", ellipse["center_px"])
        display_axes = ellipse.get("axes_px_distorted", ellipse["axes_px"])
        display_angle = ellipse.get("angle_deg_distorted", ellipse["angle_deg"])
        center = tuple(np.rint(display_center).astype(int))
        axes = tuple(max(1, int(round(value / 2.0))) for value in display_axes)
        cv2.ellipse(view, center, axes, float(display_angle), 0, 360, (0, 255, 0), 2)
        cv2.drawMarker(view, center, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.putText(view, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    return view


def _require_safe_snapshot(pose_session: Any) -> tuple[dict[str, Any], np.ndarray]:
    snapshot = pose_session.read_pose_snapshot()
    if not snapshot["power_on"] or not snapshot["steady"] or snapshot["collision"]:
        raise RuntimeError("机器人状态不满足安全门：需要上电、稳定且无碰撞")
    return snapshot, pose_session.pose_sdk_to_transform_mm(snapshot["pose_values_sdk_m_rad"])


def _wait_robot_steady(pose_session: Any, timeout_s: float = 45.0) -> tuple[dict[str, Any], np.ndarray]:
    deadline = time.monotonic() + timeout_s
    last_reason = ""
    while time.monotonic() < deadline:
        snapshot = pose_session.read_pose_snapshot()
        if snapshot["collision"]:
            raise RuntimeError("运动后检测到碰撞标志，立即停止流程")
        if snapshot["power_on"] and snapshot["steady"]:
            return snapshot, pose_session.pose_sdk_to_transform_mm(snapshot["pose_values_sdk_m_rad"])
        last_reason = f"power={snapshot['power_on']} steady={snapshot['steady']}"
        time.sleep(0.25)
    raise RuntimeError(f"等待机器人稳定超时：{last_reason}")


def _print_motion_preview(label: str, current: np.ndarray, target: np.ndarray, extra: str = "") -> None:
    delta = np.asarray(target[:3, 3]) - np.asarray(current[:3, 3])
    rotation = _angle_deg(current[:3, 2], target[:3, 2])
    print(
        f"\n[MOTION] {label}\n"
        f"  current XYZ(mm): {np.round(current[:3, 3], 3).tolist()}\n"
        f"  target  XYZ(mm): {np.round(target[:3, 3], 3).tolist()}\n"
        f"  delta XYZ(mm): {np.round(delta, 3).tolist()} | translation={np.linalg.norm(delta):.2f} mm | optical-axis change={rotation:.3f} deg"
    )
    if extra:
        print("  " + extra)


def _request_motion_confirmation(label: str, prompt: str) -> str:
    """控制台保持原有 m 确认，同时向工作台输出可逐行读取的确认事件。"""
    print(f"[MOTION_CONFIRM_REQUIRED] {label}", flush=True)
    return input(prompt).strip().lower()


def _confirm_and_move_line(label: str, current: np.ndarray, target: np.ndarray, args: Any,
                           motion_session: Any, pose_session: Any, extra: str = "",
                           require_confirmation: bool = True) -> np.ndarray:
    _print_motion_preview(label, current, target, extra)
    if require_confirmation:
        command = _request_motion_confirmation(label, "输入 m 确认运动，其他任意键取消：")
        if command != "m":
            raise RuntimeError(f"用户取消：{label}")
    else:
        print("[MOTION] 小幅自动修正，无需输入 m")
    from aubo_workbench.motion_control import sdk_ok
    response = motion_session.move_line(transform_to_sdk_pose_m_rad(target), args.speed_m_s, args.acc_m_s2)
    print("[MOTION]", response)
    if not response or not sdk_ok(response[-1]):
        raise RuntimeError(f"{label} moveLine 下发失败：{response}")
    _, actual = _wait_robot_steady(pose_session)
    return actual


def _confirm_and_move_home(home: Any, args: Any, motion_session: Any, pose_session: Any) -> np.ndarray:
    snapshot, current = _require_safe_snapshot(pose_session)
    current_joints = np.asarray(snapshot.get("joints_rad", []), dtype=np.float64)
    home_joints = np.asarray(home.joints_rad, dtype=np.float64)
    if current_joints.size == home_joints.size and current_joints.size > 0:
        max_joint_error_deg = float(np.max(np.abs(current_joints - home_joints)) * 180.0 / math.pi)
        if max_joint_error_deg <= 0.5:
            print(f"[MOTION] 当前已在原点关节位，最大关节偏差={max_joint_error_deg:.3f}°，跳过回原点运动")
            return current
    home_target = pose_session.pose_sdk_to_transform_mm(home.tcp_pose_m_rad)
    _print_motion_preview("回机械臂原点（关节运动）", current, home_target,
                          f"home={home.name} created_at={home.created_at}")
    command = _request_motion_confirmation("回机械臂原点", "输入 m 确认回原点，其他任意键取消：")
    if command != "m":
        raise RuntimeError("用户取消回原点")
    speed = math.radians(20.0)
    acc = math.radians(40.0)
    response = motion_session.move_joint(home.joints_rad, speed, acc)
    print("[MOTION]", response)
    from aubo_workbench.motion_control import sdk_ok
    settled_snapshot, actual = _wait_robot_steady(pose_session)
    if not response or not sdk_ok(response[-1]):
        settled_joints = np.asarray(settled_snapshot.get("joints_rad", []), dtype=np.float64)
        reached_home = (
            settled_joints.size == home_joints.size
            and settled_joints.size > 0
            and float(np.max(np.abs(settled_joints - home_joints)) * 180.0 / math.pi) <= 0.5
        )
        if not reached_home:
            raise RuntimeError(f"回原点 moveJoint 下发失败：{response}")
        print("[MOTION] 控制器返回非成功码，但关节已处于原点，按到位处理")
    return actual


def _capture_initial_selection(pipeline: Any, align: Any, chain: Any, model: Any, confidence: float,
                               run_dir: Path, max_plane_rmse_mm: float, hole_count: int = 1
                               ) -> tuple[Any, dict[str, Any], np.ndarray, PlaneEstimate, Any]:
    bundle = None
    for _ in range(10):
        bundle = get_aligned_frame_bundle(pipeline, align, chain)
        if bundle is not None and bundle.intrinsics is not None:
            break
    if bundle is None or bundle.intrinsics is None:
        raise RuntimeError("无法获得带RGB内参的初始 RGB-D 帧")
    detections = detect(model, bundle.color_bgr, confidence)
    if not detections:
        raise RuntimeError("原点画面中 YOLO 未检测到孔")
    selected = choose_box(bundle.color_bgr, detections)
    if selected is None:
        raise RuntimeError("未选择目标孔")
    chosen = detections[selected]
    if hole_count > 1:
        candidates = [item for item in detections if item.get("class_id") == chosen.get("class_id")]
        if len(candidates) < hole_count:
            raise RuntimeError(f"当前画面同类孔数量不足：检测到 {len(candidates)} 个，需要 {hole_count} 个")
        # 保留用户点选的参考孔，再按置信度补齐目标孔，最后按图像X/Y排序固定编号。
        selected_key = tuple(chosen["box"])
        ranked = sorted(candidates, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        group = [chosen]
        group.extend(item for item in ranked if tuple(item["box"]) != selected_key)
        group = group[:hole_count]
        group.sort(key=lambda item: (float(item["center"][1]), float(item["center"][0])))
        chosen["tracked_holes"] = [
            {"hole_id": index + 1, "initial_center_px": list(item["center"]), "class_id": item["class_id"]}
            for index, item in enumerate(group)
        ]
        chosen["reference_hole_id"] = next(
            index + 1 for index, item in enumerate(group) if tuple(item["box"]) == selected_key
        )
    ellipse = fit_hole_ellipse(bundle.color_bgr, chosen, bundle.intrinsics)
    radius = max(chosen["box"][2] - chosen["box"][0], chosen["box"][3] - chosen["box"][1]) / 2.0
    ring_center = tuple(ellipse["center_px_distorted"] if ellipse is not None else chosen["center"])
    ray_center = np.asarray(ellipse["center_px"] if ellipse is not None else chosen["center"], dtype=np.float64)
    point, info = hole_camera_point(
        ring_center, bundle.xyz_map_mm, bundle.intrinsics, radius,
        ray_center_xy=ray_center, ray_center_is_undistorted=ellipse is not None,
    )
    plane = PlaneEstimate(
        point_camera_mm=np.asarray(info["plane_point_camera_mm"], dtype=np.float64),
        normal_camera=_unit(np.asarray(info["plane_normal_camera"], dtype=np.float64), "initial plane normal"),
        rmse_mm=float(info["plane_rmse_mm"]), ring_points=int(info["ring_points"]),
        surface_model=str(info.get("surface_model", "sphere")),
        sphere_center_camera_mm=(
            np.asarray(info["sphere_center_camera_mm"], dtype=np.float64)
            if info.get("sphere_center_camera_mm") is not None else None
        ),
        sphere_radius_mm=(float(info["sphere_radius_mm"]) if info.get("sphere_radius_mm") is not None else None),
    )
    cv2.imwrite(str(run_dir / "01_home_selected.png"), _overlay(bundle.color_bgr, chosen, ellipse, "home selected hole"))
    if plane.rmse_mm > float(max_plane_rmse_mm):
        raise RuntimeError(
            f"初始孔面深度拟合RMSE过大：{plane.rmse_mm:.3f} mm "
            f"> {float(max_plane_rmse_mm):.3f} mm"
        )
    return bundle, chosen, np.asarray(point, dtype=np.float64), plane, bundle.intrinsics


def _capture_coarse_burst(pipeline: Any, align: Any, chain: Any, model: Any, confidence: float,
                          chosen: dict[str, Any], cfg: TwoStageConfig, run_dir: Path,
                          name: str) -> tuple[list[Observation], np.ndarray, dict[str, Any] | None]:
    observations: list[Observation] = []
    latest_image: np.ndarray | None = None
    latest_detection: dict[str, Any] | None = None
    latest_ellipse: dict[str, Any] | None = None
    anchor = np.array([0.0, 0.0])
    for _ in range(cfg.coarse_settle_frames):
        get_aligned_frame_bundle(pipeline, align, chain)
    for attempt in range(cfg.coarse_frames * cfg.coarse_max_attempt_multiplier):
        if len([item for item in observations if item.error is None]) >= cfg.coarse_frames:
            break
        bundle = get_aligned_frame_bundle(pipeline, align, chain)
        if bundle is None or bundle.intrinsics is None:
            continue
        latest_image = bundle.color_bgr
        if attempt == 0:
            anchor = np.array([bundle.color_bgr.shape[1] / 2.0, bundle.color_bgr.shape[0] / 2.0])
        detection = _nearest_detection(detect(model, bundle.color_bgr, confidence), anchor, chosen["class_id"])
        if detection is None:
            observations.append(Observation(name, attempt, anchor, timestamp_ns=bundle.host_timestamp_ns, error="yolo_missing"))
            continue
        ellipse = fit_hole_ellipse(bundle.color_bgr, detection, bundle.intrinsics)
        ellipse_valid = _ellipse_ok(ellipse, cfg, cfg.min_coarse_ellipse_coverage_deg)
        center = np.asarray(ellipse["center_px"] if ellipse_valid else detection["center"], dtype=np.float64)
        radius = max(detection["box"][2] - detection["box"][0], detection["box"][3] - detection["box"][1]) / 2.0
        try:
            ring_center = tuple(ellipse["center_px_distorted"] if ellipse is not None else detection["center"])
            _, plane_info = hole_camera_point(
                ring_center, bundle.xyz_map_mm, bundle.intrinsics, radius,
                ray_center_xy=center, ray_center_is_undistorted=ellipse_valid,
            )
            plane = PlaneEstimate(
                point_camera_mm=np.asarray(plane_info["plane_point_camera_mm"], dtype=np.float64),
                normal_camera=_unit(np.asarray(plane_info["plane_normal_camera"], dtype=np.float64), "coarse plane normal"),
                rmse_mm=float(plane_info["plane_rmse_mm"]), ring_points=int(plane_info["ring_points"]),
                surface_model=str(plane_info.get("surface_model", "sphere")),
                sphere_center_camera_mm=(
                    np.asarray(plane_info["sphere_center_camera_mm"], dtype=np.float64)
                    if plane_info.get("sphere_center_camera_mm") is not None else None
                ),
                sphere_radius_mm=(float(plane_info["sphere_radius_mm"]) if plane_info.get("sphere_radius_mm") is not None else None),
            )
            valid = ellipse_valid and plane.rmse_mm <= cfg.max_plane_rmse_mm
            observations.append(Observation(name, attempt, center, ellipse, plane, bundle.host_timestamp_ns,
                                            None if valid else "ellipse_or_plane_quality"))
        except Exception as exc:
            observations.append(Observation(name, attempt, center, ellipse, timestamp_ns=bundle.host_timestamp_ns,
                                            error=f"plane_error:{exc}"))
        anchor = center
        latest_detection, latest_ellipse = detection, ellipse
    if latest_image is None:
        raise RuntimeError("粗定位期间未获得相机帧")
    cv2.imwrite(str(run_dir / f"{name}_overlay.png"), _overlay(latest_image, latest_detection, latest_ellipse, name))
    return observations, latest_image, latest_ellipse


def _capture_fine_burst(pipeline: Any, model: Any, confidence: float, chosen: dict[str, Any],
                        cfg: TwoStageConfig, run_dir: Path) -> tuple[list[Observation], Any, np.ndarray, dict[str, Any] | None]:
    observations: list[Observation] = []
    latest_bundle = None
    latest_detection = latest_ellipse = None
    anchor = None
    for attempt in range(cfg.fine_frames * 3):
        if len([item for item in observations if item.error is None]) >= cfg.fine_frames:
            break
        bundle = get_rgb_frame_bundle(pipeline)
        if bundle is None or bundle.intrinsics is None:
            continue
        latest_bundle = bundle
        if anchor is None:
            anchor = np.array([bundle.color_bgr.shape[1] / 2.0, bundle.color_bgr.shape[0] / 2.0])
        detection = _nearest_detection(detect(model, bundle.color_bgr, confidence), anchor, chosen["class_id"])
        if detection is None:
            observations.append(Observation("fine", attempt, anchor, timestamp_ns=bundle.host_timestamp_ns, error="yolo_missing"))
            continue
        ellipse = fit_hole_ellipse(bundle.color_bgr, detection, bundle.intrinsics)
        if not _ellipse_ok(ellipse, cfg):
            observations.append(Observation("fine", attempt, np.asarray(detection["center"]), ellipse,
                                            timestamp_ns=bundle.host_timestamp_ns, error="ellipse_quality"))
            continue
        center = np.asarray(ellipse["center_px"], dtype=np.float64)
        observations.append(Observation("fine", attempt, center, ellipse, timestamp_ns=bundle.host_timestamp_ns))
        anchor, latest_detection, latest_ellipse = center, detection, ellipse
    if latest_bundle is None:
        raise RuntimeError("精定位期间未获得 RGB 帧")
    cv2.imwrite(str(run_dir / "04_fine_overlay.png"),
                _overlay(latest_bundle.color_bgr, latest_detection, latest_ellipse, "fine RGB ellipse"))
    return observations, latest_bundle.intrinsics, latest_bundle.color_bgr, latest_ellipse


def _overlay_multi(image: np.ndarray, detections: list[tuple[dict[str, Any], dict[str, Any] | None]], title: str) -> np.ndarray:
    view = image.copy()
    for index, (detection, ellipse) in enumerate(detections, start=1):
        view = _overlay(view, detection, ellipse, f"{title}  hole-{index}")
        cv2.putText(view, f"H{index}", tuple(map(int, detection["center"])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    return view


def _capture_multi_fine_burst(
    pipeline: Any, model: Any, confidence: float, tracked_holes: list[dict[str, Any]],
    cfg: TwoStageConfig, run_dir: Path,
) -> tuple[dict[int, list[Observation]], Any, np.ndarray]:
    """一次 RGB 精拍同时跟踪多个孔；检测按上一帧中心贪心匹配，避免孔编号跳变。"""
    if not tracked_holes:
        raise RuntimeError("没有可跟踪的多孔目标")
    observations = {int(item["hole_id"]): [] for item in tracked_holes}
    anchors = {int(item["hole_id"]): np.asarray(item["initial_center_px"], dtype=np.float64) for item in tracked_holes}
    latest_bundle = None
    latest_pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    reference_class = int(tracked_holes[0]["class_id"])
    max_attempts = cfg.fine_frames * 4
    for attempt in range(max_attempts):
        if all(len([item for item in values if item.error is None]) >= cfg.fine_frames for values in observations.values()):
            break
        bundle = get_rgb_frame_bundle(pipeline)
        if bundle is None or bundle.intrinsics is None:
            continue
        latest_bundle = bundle
        detections = [item for item in detect(model, bundle.color_bgr, confidence)
                      if int(item.get("class_id", -1)) == reference_class]
        candidates: list[tuple[dict[str, Any], dict[str, Any] | None, np.ndarray, bool]] = []
        for detection in detections:
            ellipse = fit_hole_ellipse(bundle.color_bgr, detection, bundle.intrinsics)
            valid = _ellipse_ok(ellipse, cfg)
            center = np.asarray(ellipse["center_px"] if valid else detection["center"], dtype=np.float64)
            candidates.append((detection, ellipse, center, valid))
        unused = set(range(len(candidates)))
        pairs_for_overlay: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
        for hole in tracked_holes:
            hole_id = int(hole["hole_id"])
            if not unused:
                observations[hole_id].append(Observation(
                    f"fine_hole_{hole_id}", attempt, anchors[hole_id], timestamp_ns=bundle.host_timestamp_ns,
                    error="yolo_missing",
                ))
                continue
            candidate_index = min(unused, key=lambda idx: float(np.linalg.norm(candidates[idx][2] - anchors[hole_id])))
            unused.remove(candidate_index)
            detection, ellipse, center, valid = candidates[candidate_index]
            pairs_for_overlay.append((detection, ellipse))
            if valid:
                observations[hole_id].append(Observation(
                    f"fine_hole_{hole_id}", attempt, center, ellipse, timestamp_ns=bundle.host_timestamp_ns,
                ))
                anchors[hole_id] = center
            else:
                observations[hole_id].append(Observation(
                    f"fine_hole_{hole_id}", attempt, center, ellipse, timestamp_ns=bundle.host_timestamp_ns,
                    error="ellipse_quality",
                ))
        latest_pairs = pairs_for_overlay
    if latest_bundle is None:
        raise RuntimeError("多孔精定位期间未获得 RGB 帧")
    cv2.imwrite(str(run_dir / "04_fine_multi_overlay.png"),
                _overlay_multi(latest_bundle.color_bgr, latest_pairs, "fine RGB multi-hole"))
    return observations, latest_bundle.intrinsics, latest_bundle.color_bgr


def _ray_sphere_intersection_base(
    pixel_xy: np.ndarray, intrinsics: Any, T_base_camera: np.ndarray,
    sphere_center_base_mm: np.ndarray, sphere_radius_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    ray_camera = camera_ray(intrinsics, pixel_xy, already_undistorted=True)
    origin = np.asarray(T_base_camera[:3, 3], dtype=np.float64)
    direction = _unit(T_base_camera[:3, :3] @ ray_camera, "base camera ray")
    center = np.asarray(sphere_center_base_mm, dtype=np.float64).reshape(3)
    offset = origin - center
    b = 2.0 * float(direction @ offset)
    c = float(offset @ offset - sphere_radius_mm * sphere_radius_mm)
    discriminant = b * b - 4.0 * c
    if discriminant < 0.0:
        raise RuntimeError(f"RGB射线与粗定位球面无交点：D={discriminant:.3f}")
    roots = [(-b - math.sqrt(discriminant)) / 2.0, (-b + math.sqrt(discriminant)) / 2.0]
    positive = [value for value in roots if value > 0.0]
    if not positive:
        raise RuntimeError("RGB射线与粗定位球面的交点在相机后方")
    point = origin + min(positive) * direction
    normal = _unit(point - center, "hole sphere normal")
    if float(normal @ (origin - point)) < 0.0:
        normal = -normal
    return point, normal


def _hole_surface_pose(
    point_base_mm: np.ndarray, normal_toward_camera_base: np.ndarray,
    reference_x_base: np.ndarray,
) -> np.ndarray:
    """构造孔面局部姿态：Z=朝相机法向，X=相机X在孔面内的投影。"""
    z_axis = _unit(normal_toward_camera_base, "hole surface z axis")
    x_axis = np.asarray(reference_x_base, dtype=np.float64) - float(reference_x_base @ z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis -= float(x_axis @ z_axis) * z_axis
    x_axis = _unit(x_axis, "hole surface x axis")
    y_axis = _unit(np.cross(z_axis, x_axis), "hole surface y axis")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform[:3, 3] = np.asarray(point_base_mm, dtype=np.float64)
    return transform


def _fuse_coarse(observations: list[Observation], cfg: TwoStageConfig) -> dict[str, Any]:
    valid = [item for item in observations if item.error is None and item.plane is not None]
    if len(valid) < cfg.min_coarse_valid:
        raise RuntimeError(f"粗定位有效帧不足：{len(valid)}/{cfg.min_coarse_valid}")
    centers = np.asarray([item.center_px for item in valid], dtype=np.float64)
    plane_points = _fuse_vectors([item.plane.point_camera_mm for item in valid], "coarse plane points")
    normals = _fuse_normals([item.plane.normal_camera for item in valid])
    center = np.median(centers, axis=0)
    scatter = np.linalg.norm(centers - center, axis=1)
    sphere_centers = [item.plane.sphere_center_camera_mm for item in valid if item.plane.sphere_center_camera_mm is not None]
    sphere_radii = [item.plane.sphere_radius_mm for item in valid if item.plane.sphere_radius_mm is not None]
    return {
        "valid_frames": len(valid), "total_frames": len(observations), "center_px": center,
        "center_scatter_p95_px": float(np.percentile(scatter, 95)),
        "plane_point_camera_mm": plane_points, "plane_normal_camera": normals,
        "plane_rmse_median_mm": float(np.median([item.plane.rmse_mm for item in valid])),
        "surface_model": valid[0].plane.surface_model,
        "sphere_center_camera_mm": _fuse_vectors(sphere_centers, "sphere centers") if sphere_centers else None,
        "sphere_radius_mm": float(np.median(sphere_radii)) if sphere_radii else None,
    }


def _fuse_fine(observations: list[Observation], cfg: TwoStageConfig) -> dict[str, Any]:
    valid = [item for item in observations if item.error is None and item.ellipse is not None]
    if len(valid) < cfg.min_fine_valid:
        raise RuntimeError(f"精定位有效帧不足：{len(valid)}/{cfg.min_fine_valid}")
    centers = np.asarray([item.center_px for item in valid], dtype=np.float64)
    median = np.median(centers, axis=0)
    distance = np.linalg.norm(centers - median, axis=1)
    summary = {
        "valid_frames": len(valid), "total_frames": len(observations), "center_px": median,
        "center_scatter_p95_px": float(np.percentile(distance, 95)),
        "ellipse_residual_median_px": float(np.median([item.ellipse["residual_px"] for item in valid])),
        "ellipse_roundness_median": float(np.median([item.ellipse["roundness"] for item in valid])),
        "axes_px_median": np.median(np.asarray([item.ellipse["axes_px"] for item in valid]), axis=0),
    }
    if summary["center_scatter_p95_px"] > cfg.max_fine_center_scatter_p95_px:
        raise RuntimeError(
            f"精定位圆心帧间散布过大：P95={summary['center_scatter_p95_px']:.3f} px"
        )
    return summary


def _observation_rows(observations: list[Observation]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in observations:
        row: dict[str, Any] = {
            "stage": item.stage, "frame_index": item.frame_index, "timestamp_ns": item.timestamp_ns,
            "center_u_px": float(item.center_px[0]), "center_v_px": float(item.center_px[1]), "error": item.error,
        }
        if item.plane is not None:
            row.update({
                "plane_rmse_mm": item.plane.rmse_mm, "ring_points": item.plane.ring_points,
                "plane_point_z_mm": float(item.plane.point_camera_mm[2]),
                "surface_model": item.plane.surface_model,
            })
        if item.ellipse is not None:
            row.update({
                "ellipse_residual_px": item.ellipse["residual_px"],
                "ellipse_coverage_deg": item.ellipse["coverage_deg"],
                "ellipse_roundness": item.ellipse["roundness"],
            })
        rows.append(row)
    return rows


def _write_report(run_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    _write_csv(run_dir / "frames.csv", rows)
    (run_dir / "report.json").write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")


def _stage_table(report: dict[str, Any]) -> str:
    stages = report.get("stages", {})
    lines = ["阶段              有效帧/总帧    中心散布P95       平面RMSE/最终高度"]
    for key, label in (("coarse", "340mm 粗定位"), ("coarse_validation", "粗定位校正确认"), ("fine", "260mm RGB精定位")):
        item = stages.get(key)
        if not item:
            continue
        valid = item.get("valid_frames", "-")
        total = item.get("total_frames", "-")
        scatter = item.get("center_scatter_p95_px", "-")
        right = item.get("plane_rmse_median_mm", item.get("estimated_height_mm", "-"))
        scatter_text = f"{scatter:.3f}px" if isinstance(scatter, (float, int)) else "-"
        right_text = f"{right:.3f}mm" if isinstance(right, (float, int)) else "-"
        lines.append(f"{label:<16} {valid}/{total:<10} {scatter_text:<16} {right_text}")
    return "\n".join(lines)


def run_two_stage_hole_localization(args: Any, handeye: Any, model: Any) -> int:
    """执行“原点选孔 -> 340 mm粗定位 -> 260 mm RGB精定位”的受确认流程。"""
    cfg = TwoStageConfig(
        coarse_height_mm=float(args.coarse_height_mm), fine_height_mm=float(args.fine_height_mm),
        coarse_frames=int(args.coarse_frames), fine_frames=int(args.fine_frames),
    )
    if cfg.fine_height_mm >= cfg.coarse_height_mm:
        raise ValueError("精定位高度必须小于粗定位高度")
    run_dir = RUNS_DIR / f"two-stage-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "status": "running", "mode": "two_stage_hole_localization", "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"), "configuration": cfg.__dict__,
        "hole_count": int(args.hole_count),
        "handeye_path": str(args.handeye), "stages": {}, "motion_executed": bool(args.execute),
        "experimental_handeye_override": bool(args.allow_experimental_handeye),
        "final_xy_compensation": (
            {
                "mode": "charuco_affine_model",
                "source": str(CHARUCO_XY_MODEL_SOURCE),
                "matrix_2x2": CHARUCO_XY_MODEL_MATRIX,
                "bias_mm": CHARUCO_XY_MODEL_BIAS_MM,
            }
            if args.tcp_xy_offset_mm is None else {
                "mode": "fixed_offset_override",
                "tcp_xy_offset_mm": [float(value) for value in args.tcp_xy_offset_mm],
            }
        ),
    }
    rows: list[dict[str, Any]] = []
    rgbd_pipeline = rgb_pipeline = align = chain = None
    pose_session = motion_session = None
    try:
        from aubo_workbench.motion_control import AuboMotionSession, load_home_point
        from aubo_workbench.robot import AuboPoseSession

        home = load_home_point()
        if home is None:
            raise RuntimeError("未找到 aubo_home_point.json；请先在机械臂运动界面设置原始点")
        pose_session = AuboPoseSession()
        pose_session.connect()
        initial_snapshot, initial_tcp = _require_safe_snapshot(pose_session)
        report["robot_initial_tcp_pose_m_rad"] = initial_snapshot["pose_values_sdk_m_rad"]
        report["home_point"] = home.to_dict()

        if args.execute:
            if not handeye.validated_for_motion and not args.allow_experimental_handeye:
                raise RuntimeError(
                    "手眼证据未通过生产运动门，拒绝两阶段自动运动；"
                    "若仅用于现场实验验证，请显式添加 --allow-experimental-handeye"
                )
            if not handeye.validated_for_motion:
                print(
                    "[EXPERIMENTAL] 使用未通过生产验证的当前手眼结果。"
                    "本次仅可作实验验证，结果不会被标记为生产可用。"
                )
            motion_session = AuboMotionSession()
            motion_session.connect(ROBOT_CFG.ip, ROBOT_CFG.rpc_port, ROBOT_CFG.user,
                                  ROBOT_CFG.password, ROBOT_CFG.request_timeout_ms)
            current_tcp = _confirm_and_move_home(home, args, motion_session, pose_session)
        else:
            current_tcp = initial_tcp
            print("[PREVIEW] 未指定 --execute：不会回原点或下发运动；请手动将机器人置于原点后核对规划。")

        rgbd_pipeline, align, chain = init_pipeline()
        identity = get_device_identity(rgbd_pipeline)
        expected = str(handeye.payload.get("camera_serial", "")).strip()
        actual = str(identity.get("serial_number", "")).strip()
        if expected and actual and expected != actual:
            raise RuntimeError(f"相机序列号不匹配：手眼={expected}，当前={actual}")
        report["camera"] = identity

        initial_bundle, chosen, selected_point_camera, initial_plane_camera, intrinsics = _capture_initial_selection(
            rgbd_pipeline, align, chain, model, args.confidence, run_dir,
            cfg.initial_max_plane_rmse_mm, args.hole_count,
        )
        T_base_camera_home = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
        selected_point_base = T_base_camera_home[:3, :3] @ selected_point_camera + T_base_camera_home[:3, 3]
        plane_point_base = T_base_camera_home[:3, :3] @ initial_plane_camera.point_camera_mm + T_base_camera_home[:3, 3]
        plane_normal_base = _unit(T_base_camera_home[:3, :3] @ initial_plane_camera.normal_camera, "initial base plane normal")
        rough_target = plan_centered_tcp_pose(selected_point_base, plane_normal_base, current_tcp,
                                              handeye.T_tcp_rgb_camera, cfg.coarse_height_mm)
        report["stages"]["home_selection"] = {
            "selected": chosen, "hole_center_camera_mm": selected_point_camera,
            "hole_center_base_mm": selected_point_base, "plane_point_base_mm": plane_point_base,
            "plane_normal_base": plane_normal_base, "plane_rmse_mm": initial_plane_camera.rmse_mm,
            "ring_points": initial_plane_camera.ring_points,
            "surface_model": initial_plane_camera.surface_model,
        }

        if not args.execute:
            fine_preview, predicted_height = base_z_target_for_camera_height(
                rough_target, handeye.T_tcp_rgb_camera, plane_point_base, cfg.fine_height_mm,
            )
            _print_motion_preview("预览：原点到孔上方340 mm", current_tcp, rough_target)
            _print_motion_preview("预览：340 mm到260 mm（仅基坐标Z）", rough_target, fine_preview,
                                  f"粗拍位预测高度={predicted_height:.2f} mm")
            report["status"] = "preview_complete"
            report["planned_rough_tcp_pose_m_rad"] = transform_to_sdk_pose_m_rad(rough_target)
            report["planned_fine_tcp_pose_m_rad"] = transform_to_sdk_pose_m_rad(fine_preview)
            _write_report(run_dir, report, rows)
            print(f"[PREVIEW] 已写入: {run_dir}")
            return 0

        current_tcp = _confirm_and_move_line("自动到孔上方340 mm粗拍位", current_tcp, rough_target,
                                              args, motion_session, pose_session,
                                              "目标：光轴垂直所选孔面，孔中心投影至RGB主点")
        coarse_obs, _, _ = _capture_coarse_burst(rgbd_pipeline, align, chain, model, args.confidence,
                                                  chosen, cfg, run_dir, "02_coarse")
        rows.extend(_observation_rows(coarse_obs))
        coarse = _fuse_coarse(coarse_obs, cfg)
        T_base_camera = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
        coarse_plane_base = T_base_camera[:3, :3] @ coarse["plane_point_camera_mm"] + T_base_camera[:3, 3]
        coarse_normal_base = _unit(T_base_camera[:3, :3] @ coarse["plane_normal_camera"], "coarse base normal")
        coarse_hole_base = pixel_to_base_plane(
            coarse["center_px"], intrinsics, current_tcp, handeye.T_tcp_rgb_camera,
            coarse_plane_base, coarse_normal_base, center_is_undistorted=True,
        )
        center_offset = float(np.linalg.norm(coarse["center_px"] - np.array([intrinsics.cx, intrinsics.cy])))
        normal_error = _angle_deg(T_base_camera[:3, 2], -coarse_normal_base)
        coarse.update({
            "hole_center_base_mm": coarse_hole_base, "plane_point_base_mm": coarse_plane_base,
            "plane_normal_base": coarse_normal_base, "center_offset_px": center_offset,
            "normal_error_deg": normal_error,
        })
        report["stages"]["coarse"] = coarse

        validation = None
        # 最多两次小幅闭环校正；闭环运动自动执行。
        for correction_index in range(2):
            if center_offset <= cfg.center_tolerance_px and normal_error <= cfg.normal_tolerance_deg:
                break
            corrected_target = plan_centered_tcp_pose(
                coarse_hole_base, coarse_normal_base, current_tcp,
                handeye.T_tcp_rgb_camera, cfg.coarse_height_mm,
            )
            current_tcp = _confirm_and_move_line(
                f"340 mm粗定位闭环校正 {correction_index + 1}/2",
                current_tcp, corrected_target, args, motion_session, pose_session,
                f"center offset={center_offset:.2f}px, normal error={normal_error:.3f}deg",
                require_confirmation=False,
            )
            validation_obs, _, _ = _capture_coarse_burst(
                rgbd_pipeline, align, chain, model, args.confidence,
                chosen, cfg, run_dir,
                "03_coarse_validation" if correction_index == 0 else "04_coarse_validation_2",
            )
            rows.extend(_observation_rows(validation_obs))
            validation = _fuse_coarse(validation_obs, cfg)
            T_base_camera = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
            coarse_plane_base = T_base_camera[:3, :3] @ validation["plane_point_camera_mm"] + T_base_camera[:3, 3]
            coarse_normal_base = _unit(
                T_base_camera[:3, :3] @ validation["plane_normal_camera"], "validated base normal"
            )
            coarse_hole_base = pixel_to_base_plane(
                validation["center_px"], intrinsics, current_tcp, handeye.T_tcp_rgb_camera,
                coarse_plane_base, coarse_normal_base, center_is_undistorted=True,
            )
            center_offset = float(
                np.linalg.norm(validation["center_px"] - np.array([intrinsics.cx, intrinsics.cy]))
            )
            normal_error = _angle_deg(T_base_camera[:3, 2], -coarse_normal_base)
            validation.update({
                "hole_center_base_mm": coarse_hole_base,
                "plane_point_base_mm": coarse_plane_base,
                "plane_normal_base": coarse_normal_base,
                "center_offset_px": center_offset,
                "normal_error_deg": normal_error,
                "correction_index": correction_index + 1,
            })
            report["stages"]["coarse_validation" if correction_index == 0 else "coarse_validation_2"] = validation
        if center_offset > cfg.center_tolerance_px or normal_error > cfg.normal_tolerance_deg:
            raise RuntimeError(
                "粗定位两次闭环校正后仍未满足居中/垂直质量门："
                f"offset={center_offset:.2f}px, normal={normal_error:.3f}deg"
            )

        estimated_height = None
        for correction_index in range(cfg.max_z_corrections):
            _, actual_tcp = _require_safe_snapshot(pose_session)
            estimated_height = camera_height_to_plane_mm(actual_tcp, handeye.T_tcp_rgb_camera, coarse_plane_base)
            if abs(estimated_height - cfg.fine_height_mm) <= cfg.height_tolerance_mm:
                current_tcp = actual_tcp
                break
            z_target, _ = base_z_target_for_camera_height(actual_tcp, handeye.T_tcp_rgb_camera,
                                                           coarse_plane_base, cfg.fine_height_mm)
            current_tcp = _confirm_and_move_line(
                f"仅基坐标Z下降至{cfg.fine_height_mm:.0f} mm（第{correction_index + 1}次）",
                actual_tcp, z_target, args, motion_session, pose_session,
                f"当前估计高度={estimated_height:.2f} mm；XY与姿态锁定",
                require_confirmation=bool(abs(float(z_target[2, 3] - actual_tcp[2, 3])) > 30.0),
            )
        _, current_tcp = _require_safe_snapshot(pose_session)
        estimated_height = camera_height_to_plane_mm(current_tcp, handeye.T_tcp_rgb_camera, coarse_plane_base)
        if abs(estimated_height - cfg.fine_height_mm) > cfg.height_tolerance_mm:
            raise RuntimeError(f"仅Z修正后仍未达到精拍高度：{estimated_height:.2f} mm")

        try:
            rgbd_pipeline.stop()
        finally:
            rgbd_pipeline = None
        rgb_pipeline = init_rgb_handeye_pipeline()
        if args.hole_count == 3:
            tracked_holes = list(chosen.get("tracked_holes", []))
            if len(tracked_holes) != 3:
                raise RuntimeError("三孔模式没有建立完整的三个目标跟踪组")
            multi_obs, fine_intrinsics, _, = _capture_multi_fine_burst(
                rgb_pipeline, model, args.confidence, tracked_holes, cfg, run_dir,
            )
            rows.extend(row for hole_rows in multi_obs.values() for row in _observation_rows(hole_rows))
            T_base_camera_fine = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
            sphere_center_camera = coarse.get("sphere_center_camera_mm")
            sphere_radius = coarse.get("sphere_radius_mm")
            if sphere_center_camera is None or sphere_radius is None:
                raise RuntimeError("三孔模式需要粗定位球面模型，但当前粗定位未返回球心/半径")
            sphere_center_base = (
                T_base_camera[:3, :3] @ np.asarray(sphere_center_camera, dtype=np.float64)
                + T_base_camera[:3, 3]
            )
            holes_result: list[dict[str, Any]] = []
            for hole in tracked_holes:
                hole_id = int(hole["hole_id"])
                summary = _fuse_fine(multi_obs[hole_id], cfg)
                point_base, normal_base = _ray_sphere_intersection_base(
                    summary["center_px"], fine_intrinsics, T_base_camera_fine,
                    sphere_center_base, float(sphere_radius),
                )
                hole_pose = _hole_surface_pose(
                    point_base, normal_base, T_base_camera_fine[:3, 0],
                )
                diameter_px = float(max(float(summary["axes_px_median"][0]), float(summary["axes_px_median"][1])))
                ray_distance = float(np.linalg.norm(point_base - T_base_camera_fine[:3, 3]))
                diameter_estimate = diameter_px * ray_distance / ((fine_intrinsics.fx + fine_intrinsics.fy) / 2.0)
                matched_diameter = min(HOLE_DIAMETERS_MM, key=lambda value: abs(value - diameter_estimate))
                summary.update({
                    "hole_id": hole_id,
                    "center_px": summary["center_px"],
                    "hole_center_base_mm": point_base,
                    "plane_normal_toward_camera_base": normal_base,
                    "hole_pose_m_rad": transform_to_sdk_pose_m_rad(hole_pose),
                    "diameter_estimate_mm": diameter_estimate,
                    "matched_diameter_mm": matched_diameter,
                    "shared_fine_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                })
                holes_result.append(summary)
            holes_result.sort(key=lambda item: int(item["hole_id"]))
            report["stages"]["fine_multi"] = {
                "hole_count": 3,
                "surface_model": "sphere",
                "sphere_center_base_mm": sphere_center_base,
                "sphere_radius_mm": float(sphere_radius),
                "shared_fine_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "holes": holes_result,
            }
            report["final_result"] = {
                "hole_count": 3,
                "reference_hole_id": chosen.get("reference_hole_id"),
                "holes": holes_result,
                "shared_fine_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "final_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "motion": "skipped_multi_hole_output_only",
            }
            report["status"] = "completed_experimental_handeye" if not handeye.validated_for_motion else "completed"
            _write_report(run_dir, report, rows)
            print("\n[三孔结果] 粗定位和精定位各执行一次；最终不移动到单个孔。")
            print(json.dumps(_jsonable(report["final_result"]), ensure_ascii=False, indent=2))
            print(f"[DONE] 结果目录: {run_dir}")
            return 0
        fine_obs, fine_intrinsics, _, _ = _capture_fine_burst(rgb_pipeline, model, args.confidence, chosen, cfg, run_dir)
        rows.extend(_observation_rows(fine_obs))
        fine = _fuse_fine(fine_obs, cfg)
        naive_final_point_base = pixel_to_base_plane(
            fine["center_px"], fine_intrinsics, current_tcp, handeye.T_tcp_rgb_camera,
            coarse_plane_base, coarse_normal_base, center_is_undistorted=True,
        )
        # 倾斜时短轴会明显缩短，使用长轴估算孔径比几何平均更稳定。
        diameter_px = float(max(float(fine["axes_px_median"][0]), float(fine["axes_px_median"][1])))
        diameter_estimate = diameter_px * estimated_height / ((fine_intrinsics.fx + fine_intrinsics.fy) / 2.0)
        nearest_diameter = min(HOLE_DIAMETERS_MM, key=lambda value: abs(value - diameter_estimate))
        T_base_camera_fine = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
        final_normal = coarse_normal_base if float(coarse_normal_base @ T_base_camera_fine[:3, 2]) < 0 else -coarse_normal_base

        tilt_correction = None
        final_point_base = naive_final_point_base
        if cfg.enable_tilt_center_correction:
            final_point_base, tilt_correction = correct_projected_circle_center(
                fine["center_px"], fine_intrinsics, current_tcp, handeye.T_tcp_rgb_camera,
                coarse_plane_base, coarse_normal_base, nearest_diameter,
                iterations=cfg.tilt_correction_iterations,
                samples=cfg.tilt_correction_samples,
                max_correction_mm=cfg.max_tilt_correction_mm,
            )

        fine.update({
            "estimated_height_mm": estimated_height,
            "hole_center_base_mm": final_point_base,
            "hole_center_base_naive_mm": naive_final_point_base,
            "plane_normal_toward_camera_base": final_normal,
            "diameter_estimate_mm": diameter_estimate,
            "matched_diameter_mm": nearest_diameter,
            "tilt_center_correction": tilt_correction,
            "tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
        })
        report["stages"]["fine"] = fine
        fine_tcp = current_tcp.copy()
        final_xy_motion: dict[str, Any] | None = None
        final_z_motion: dict[str, Any] | None = None
        final_y_trim_motion: dict[str, Any] | None = None
        if args.move_final_xy:
            fixed_offset = (None if args.tcp_xy_offset_mm is None
                            else (float(args.tcp_xy_offset_mm[0]), float(args.tcp_xy_offset_mm[1])))
            xy_target, tcp_before = plan_final_tcp_xy(current_tcp, final_point_base, fixed_offset)
            correction_xy = xy_target[:2, 3] - final_point_base[:2]
            compensation = (
                f"ChArUco仿射模型修正={np.round(correction_xy, 3).tolist()} mm"
                if fixed_offset is None else f"显式固定补偿={list(fixed_offset)} mm"
            )
            current_tcp = _confirm_and_move_line(
                "精定位后将TCP移动至最终孔XY", current_tcp, xy_target,
                args, motion_session, pose_session,
                f"保持精拍TCP Z与姿态；{compensation}",
                require_confirmation=False,
            )
            final_xy_motion = {
                "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(xy_target),
                "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "tcp_position_before_mm": tcp_before,
                "hole_center_base_mm": final_point_base,
                "compensation_mode": "charuco_affine_model" if fixed_offset is None else "fixed_offset_override",
                "xy_correction_mm": correction_xy,
                "charuco_model_source": str(CHARUCO_XY_MODEL_SOURCE) if fixed_offset is None else None,
                "charuco_model_matrix_2x2": CHARUCO_XY_MODEL_MATRIX if fixed_offset is None else None,
                "charuco_model_bias_mm": CHARUCO_XY_MODEL_BIAS_MM if fixed_offset is None else None,
                "tcp_xy_offset_mm": None if fixed_offset is None else list(fixed_offset),
            }
            report["stages"]["final_xy_motion"] = final_xy_motion
            z_target = plan_final_tcp_base_z(current_tcp, final_point_base)
            z_delta_mm = float(z_target[2, 3] - current_tcp[2, 3])
            current_tcp = _confirm_and_move_line(
                "最终孔中心基坐标Z运动", current_tcp, z_target,
                args, motion_session, pose_session,
                f"保持最终XY与姿态；目标TCP基坐标Z={z_target[2, 3]:.3f} mm",
                require_confirmation=bool(abs(z_delta_mm) > 30.0),
            )
            final_z_motion = {
                "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(z_target),
                "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "target_base_z_mm": float(z_target[2, 3]),
                "delta_base_z_mm": z_delta_mm,
                "motion_frame": "base_z_only",
            }
            report["stages"]["final_z_motion"] = final_z_motion
            y_trim_target = plan_final_tcp_base_y_trim(current_tcp)
            current_tcp = _confirm_and_move_line(
                "最终基坐标+Y微调", current_tcp, y_trim_target,
                args, motion_session, pose_session,
                f"保持X、Z与姿态；基坐标Y增加 {FINAL_BASE_Y_AFTER_Z_MM:.3f} mm",
                require_confirmation=False,
            )
            final_y_trim_motion = {
                "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(y_trim_target),
                "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "delta_base_y_mm": FINAL_BASE_Y_AFTER_Z_MM,
                "motion_frame": "base_y_only",
            }
            report["stages"]["final_y_trim_motion"] = final_y_trim_motion
        report["final_result"] = {
            "hole_center_base_mm": final_point_base,
            "hole_center_base_naive_mm": naive_final_point_base,
            "tilt_center_correction": tilt_correction,
            "plane_normal_toward_camera_base": final_normal,
            "fine_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(fine_tcp),
            "final_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
            "matched_diameter_mm": nearest_diameter,
            "final_xy_motion": final_xy_motion,
            "final_z_motion": final_z_motion,
            "final_y_trim_motion": final_y_trim_motion,
        }
        report["status"] = "completed_experimental_handeye" if not handeye.validated_for_motion else "completed"
        _write_report(run_dir, report, rows)
        print("\n" + _stage_table(report))
        print(json.dumps(_jsonable(report["final_result"]), ensure_ascii=False, indent=2))
        print(f"[DONE] 结果目录: {run_dir}")
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        _write_report(run_dir, report, rows)
        raise
    finally:
        if rgbd_pipeline is not None:
            try:
                rgbd_pipeline.stop()
            except Exception:
                pass
        if rgb_pipeline is not None:
            try:
                rgb_pipeline.stop()
            except Exception:
                pass
        if pose_session is not None:
            pose_session.disconnect()
        if motion_session is not None:
            motion_session.disconnect()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLO 选孔并计算眼在手目标位姿")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--handeye", type=Path, default=DEFAULT_HANDEYE)
    p.add_argument("--confidence", type=float, default=0.35)
    p.add_argument("--image", type=Path, help="离线 RGB 图；不指定则使用 Gemini RGB-D")
    p.add_argument("--intrinsics", type=Path, help="离线图像使用的 RGB 内参 JSON")
    p.add_argument("--target-depth-mm", type=float, help="离线无深度图时使用的平面深度（仅粗略预览）")
    p.add_argument("--execute", dest="execute", action="store_true", default=DEFAULT_EXECUTE_MOTION,
                   help="兼容参数：默认已启用真实运动，大幅移动仍需输入 m 确认")
    p.add_argument("--no-execute", dest="execute", action="store_false",
                   help="仅预览：不连接运动控制或下发机器人运动")
    p.add_argument("--allow-experimental-handeye", dest="allow_experimental_handeye", action="store_true",
                   default=DEFAULT_ALLOW_EXPERIMENTAL_HANDEYE,
                   help="兼容参数：默认允许当前实验手眼结果用于现场诊断")
    p.add_argument("--require-validated-handeye", dest="allow_experimental_handeye", action="store_false",
                   help="只允许已获生产授权的手眼结果")
    p.add_argument("--speed-m-s", type=float, default=0.03)
    p.add_argument("--acc-m-s2", type=float, default=0.10)
    p.add_argument("--offset-mm", type=float, default=0.0, help="沿机器人基坐标 Z 方向的安全偏置")
    p.add_argument("--two-stage-hole-localization", dest="two_stage_hole_localization", action="store_true",
                   default=DEFAULT_TWO_STAGE_HOLE_LOCALIZATION,
                   help="兼容参数：默认执行 原点选孔->340mm粗定位->260mm RGB精定位")
    p.add_argument("--single-stage", dest="two_stage_hole_localization", action="store_false",
                   help="仅诊断使用：关闭默认两阶段流程")
    p.add_argument("--coarse-height-mm", type=float, default=340.0,
                   help="两阶段模式的孔面RGB-Z粗定位高度，默认340")
    p.add_argument("--fine-height-mm", type=float, default=260.0,
                   help="两阶段模式的孔面RGB-Z精定位高度，默认260")
    p.add_argument("--coarse-frames", type=int, default=15, help="两阶段粗定位有效RGB-D帧数")
    p.add_argument("--fine-frames", type=int, default=30, help="两阶段精定位有效RGB帧数")
    p.add_argument("--hole-count", type=int, choices=(1, 3), default=1,
                   help="两阶段输出孔数量；3表示一次粗/精定位同时推算三个孔")
    p.add_argument("--move-final-xy", dest="move_final_xy", action="store_true", default=True,
                   help="兼容参数：两阶段流程默认已启用最终 TCP XY 微调")
    p.add_argument("--no-move-final-xy", dest="move_final_xy", action="store_false",
                   help="仅排障使用：关闭精定位后的最终 TCP XY 微调")
    p.add_argument("--tcp-xy-offset-mm", type=float, nargs=2, metavar=("DX", "DY"), default=None,
                   help="临时固定TCP XY补偿(mm)；默认不施加任何XY偏置")
    # 工作台 GUI 可用这些参数覆盖本机默认连接配置；命令行既有用法保持兼容。
    p.add_argument("--robot-ip", type=str, help="AUBO RPC IP（工作台传入）")
    p.add_argument("--robot-port", type=int, help="AUBO RPC 端口（工作台传入）")
    p.add_argument("--robot-user", type=str, help="AUBO 用户名（工作台传入）")
    p.add_argument("--robot-password", type=str, help="AUBO 密码（工作台传入）")
    p.add_argument("--robot-timeout-ms", type=int, help="AUBO 请求超时毫秒（工作台传入）")
    return p


def _apply_robot_connection_overrides(args: Any) -> None:
    """让工作台顶栏连接参数对独立定位进程生效。"""
    for arg_name, config_name in (
        ("robot_ip", "ip"),
        ("robot_port", "rpc_port"),
        ("robot_user", "user"),
        ("robot_password", "password"),
        ("robot_timeout_ms", "request_timeout_ms"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(ROBOT_CFG, config_name, value)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _apply_robot_connection_overrides(args)
    handeye = load_handeye_experiment_result(args.handeye)
    model = load_yolo(args.model)
    if args.two_stage_hole_localization:
        if args.image:
            raise RuntimeError("两阶段模式必须使用实时 Gemini RGB-D/RGB 流，不能使用 --image")
        return run_two_stage_hole_localization(args, handeye, model)
    pipeline = align = chain = None
    pose_session = motion_session = None
    try:
        if args.image:
            image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"无法读取图像: {args.image}")
            detections = detect(model, image, args.confidence)
            intrinsics = _load_intrinsics(args.intrinsics) if args.intrinsics else None
            if intrinsics is None:
                raise RuntimeError("离线模式必须提供 --intrinsics JSON")
            xyz = None
        else:
            pipeline, align, chain = init_pipeline()
            bundle = None
            while bundle is None:
                bundle = get_aligned_frame_bundle(pipeline, align, chain)
            image, xyz, intrinsics = bundle.color_bgr, bundle.xyz_map_mm, bundle.intrinsics
            if intrinsics is None:
                raise RuntimeError("相机没有返回 RGB 内参")
            detections = detect(model, image, args.confidence)
            expected = str(handeye.payload.get("camera_serial", "")).strip()
            actual = str(get_device_identity(pipeline).get("serial_number", "")).strip()
            if expected and actual and expected != actual:
                raise RuntimeError(f"相机序列号不匹配：手眼={expected}，当前={actual}")
        if not detections:
            raise RuntimeError("YOLO 未检测到孔")
        idx = choose_box(image, detections)
        if idx is None:
            print("已取消。")
            return 0
        chosen = detections[idx]
        ellipse = fit_hole_ellipse(image, chosen, intrinsics)
        center_distorted = np.asarray(
            ellipse["center_px_distorted"] if ellipse is not None else chosen["center"], dtype=np.float64,
        )
        center_for_ray = np.asarray(
            ellipse["center_px"] if ellipse is not None else chosen["center"], dtype=np.float64,
        )
        radius = max(chosen["box"][2] - chosen["box"][0], chosen["box"][3] - chosen["box"][1]) / 2.0
        if xyz is not None:
            p_cam, depth_info = hole_camera_point(
                tuple(center_distorted), xyz, intrinsics, radius,
                ray_center_xy=center_for_ray, ray_center_is_undistorted=ellipse is not None,
            )
        else:
            if args.target_depth_mm is None:
                raise RuntimeError("离线模式请提供 --intrinsics 和 --target-depth-mm，或改用 RGB-D 实时模式")
            ray = camera_ray(intrinsics, center_for_ray, already_undistorted=ellipse is not None)
            p_cam = ray * (float(args.target_depth_mm) / float(ray[2]))
            depth_info = {"mode": "constant_depth_preview", "ellipse": ellipse}

        # 只有真正要把像素点变成机器人基坐标时才加载 AUBO SDK；
        # 这样 --help/模型检测/离线预览在没有 SDK 的电脑上也能运行。
        from aubo_workbench.robot import AuboPoseSession
        from aubo_workbench.motion_control import AuboMotionSession

        pose_session = AuboPoseSession()
        pose_session.connect()
        snapshot = pose_session.read_pose_snapshot()
        if not snapshot["power_on"] or not snapshot["steady"] or snapshot["collision"]:
            raise RuntimeError("机器人状态不满足运动前安全门：需要上电、稳定且无碰撞")
        T_base_tcp = pose_session.pose_sdk_to_transform_mm(snapshot["pose_values_sdk_m_rad"])
        p_tcp = handeye.T_tcp_rgb_camera[:3, :3] @ p_cam + handeye.T_tcp_rgb_camera[:3, 3]
        p_base = T_base_tcp[:3, :3] @ p_tcp + T_base_tcp[:3, 3]
        p_base[2] += float(args.offset_mm)
        target = T_base_tcp.copy()
        target[:3, 3] = p_base
        print(json.dumps({"selected": chosen, "camera_point_mm": p_cam.tolist(),
                          "base_target_mm": p_base.tolist(), "depth": depth_info,
                          "handeye_validated": handeye.validated_for_motion,
                          "target_pose_xyzrpy": list(transform_to_pose6_rzryrx(target))},
                         ensure_ascii=False, indent=2))
        if args.execute:
            if not handeye.validated_for_motion:
                raise RuntimeError("手眼证据未通过生产运动门，拒绝下发；请先安装 validated E7 结果")
            motion_session = AuboMotionSession()
            motion_session.connect(ROBOT_CFG.ip, ROBOT_CFG.rpc_port, ROBOT_CFG.user,
                                    ROBOT_CFG.password, ROBOT_CFG.request_timeout_ms)
            pose = list(snapshot["pose_values_sdk_m_rad"])
            pose[:3] = (p_base / 1000.0).tolist()
            print("[MOTION] 下发 moveLine，目标 XYZ(m):", pose[:3])
            print(motion_session.move_line(pose, args.speed_m_s, args.acc_m_s2))
        return 0
    finally:
        if pipeline is not None:
            try: pipeline.stop()
            except Exception: pass
        if pose_session is not None: pose_session.disconnect()
        if motion_session is not None: motion_session.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
