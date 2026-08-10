#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""相机投影几何：去畸变、像素光线、光线与平面求交、倾斜圆心修正、相机高度控制。

内参对象按属性读取（fx/fy/cx/cy 以及 distortion 或 dist_coeffs），
因此同时兼容 ``aubo_workbench.camera.CameraIntrinsics``
与 ``aubo_workbench.cad_registration.CameraIntrinsics`` 两种数据类。

注意：``aubo_workbench.cad_registration`` 里另有一个同名
``undistort_pixels(points_px, intrinsics)``，参数顺序与本模块相反。
本模块统一为 intrinsics 在前，混用会静默得到错误结果，不要交叉调用。
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from aubo_workbench.geometry import unit_vector


def camera_transform(T_base_tcp: np.ndarray, T_tcp_camera: np.ndarray) -> np.ndarray:
    return np.asarray(T_base_tcp, dtype=np.float64) @ np.asarray(T_tcp_camera, dtype=np.float64)


def camera_matrix(intrinsics: Any) -> np.ndarray:
    return np.asarray(
        [[intrinsics.fx, 0.0, intrinsics.cx],
         [0.0, intrinsics.fy, intrinsics.cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def distortion_coeffs(intrinsics: Any) -> np.ndarray:
    """读取畸变系数，兼容 ``distortion`` 与 ``dist_coeffs`` 两种字段名。

    两个字段都不存在时抛错，而不是静默当作无畸变——后者会让标定结果偏移
    却看不出任何异常。
    """
    if hasattr(intrinsics, "distortion"):
        raw = getattr(intrinsics, "distortion")
    elif hasattr(intrinsics, "dist_coeffs"):
        raw = getattr(intrinsics, "dist_coeffs")
    else:
        raise AttributeError(
            f"{type(intrinsics).__name__} 既没有 distortion 也没有 dist_coeffs，无法确定畸变参数"
        )
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        return np.zeros(0, dtype=np.float64)
    return values


def undistort_pixels(intrinsics: Any, points_px: np.ndarray, *, pixel_output: bool = True) -> np.ndarray:
    """把原始畸变像素转换成去畸变像素或归一化坐标。"""
    points = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
    distortion = distortion_coeffs(intrinsics)
    if distortion.size == 0 or not np.any(np.abs(distortion) > 1e-12):
        if pixel_output:
            return points.reshape(-1, 2)
        K = camera_matrix(intrinsics)
        flat = points.reshape(-1, 2)
        return np.column_stack(((flat[:, 0] - K[0, 2]) / K[0, 0],
                                (flat[:, 1] - K[1, 2]) / K[1, 1]))
    P = camera_matrix(intrinsics) if pixel_output else None
    result = cv2.undistortPoints(points, camera_matrix(intrinsics), distortion.reshape(1, -1), P=P)
    return result.reshape(-1, 2)


def camera_ray(intrinsics: Any, center_px: np.ndarray, *, already_undistorted: bool = False) -> np.ndarray:
    u, v = np.asarray(center_px, dtype=np.float64).reshape(2)
    if already_undistorted:
        x = (u - intrinsics.cx) / intrinsics.fx
        y = (v - intrinsics.cy) / intrinsics.fy
    else:
        x, y = undistort_pixels(intrinsics, np.asarray([[u, v]]), pixel_output=False)[0]
    return unit_vector(np.array([x, y, 1.0], dtype=np.float64), "pixel ray")


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


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = unit_vector(normal, "circle plane normal")
    hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(hint @ n)) > 0.9:
        hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis_x = unit_vector(hint - n * float(hint @ n), "circle plane x")
    axis_y = unit_vector(np.cross(n, axis_x), "circle plane y")
    return axis_x, axis_y


def project_undistorted_pixels(points_camera_mm: np.ndarray, intrinsics: Any) -> np.ndarray:
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
    plane_normal_camera = unit_vector(R_camera_base @ np.asarray(plane_normal_base, dtype=np.float64), "camera plane normal")

    observed = np.asarray(observed_ellipse_center_px, dtype=np.float64).reshape(2)
    candidate = ray_plane_intersection(
        np.zeros(3, dtype=np.float64),
        camera_ray(intrinsics, observed, already_undistorted=True),
        plane_point_camera,
        plane_normal_camera,
    )
    naive_camera = candidate.copy()
    radius = float(diameter_mm) / 2.0
    axis_x, axis_y = plane_basis(plane_normal_camera)
    theta = np.linspace(0.0, 2.0 * math.pi, int(samples), endpoint=False)
    final_bias_px = np.zeros(2, dtype=np.float64)

    for _ in range(max(1, int(iterations))):
        circle = (
            candidate[None, :]
            + radius * np.cos(theta)[:, None] * axis_x[None, :]
            + radius * np.sin(theta)[:, None] * axis_y[None, :]
        )
        projected = project_undistorted_pixels(circle, intrinsics)
        ellipse = cv2.fitEllipse(projected.astype(np.float32).reshape(-1, 1, 2))
        projected_ellipse_center = np.asarray(ellipse[0], dtype=np.float64)
        projected_circle_center = project_undistorted_pixels(candidate.reshape(1, 3), intrinsics)[0]
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
