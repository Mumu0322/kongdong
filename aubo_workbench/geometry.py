#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""4x4 齐次变换、旋转、位姿格式转换等纯数学工具，不依赖相机/机械臂/GUI。"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def ensure_finite_array(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")


def rotx(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def roty(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rotz(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_to_pose6_rzryrx(T: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """仅用于输出近似 [x, y, z, rz, ry, rx]，运行时应使用 4x4 矩阵。"""
    R = np.asarray(T[:3, :3], dtype=np.float64)
    beta = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    cb = math.cos(beta)
    if abs(cb) > 1e-8:
        alpha = math.atan2(R[1, 0], R[0, 0])
        gamma = math.atan2(R[2, 1], R[2, 2])
    else:
        alpha = math.atan2(-R[0, 1], R[1, 1])
        gamma = 0.0
    return (
        float(T[0, 3]), float(T[1, 3]), float(T[2, 3]),
        math.degrees(alpha), math.degrees(beta), math.degrees(gamma),
    )


def transform_to_vec6(T: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(T[:3, :3], dtype=np.float64))
    return np.r_[rvec.reshape(3), np.asarray(T[:3, 3], dtype=np.float64).reshape(3)]


def vec6_to_transform(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(6)
    R, _ = cv2.Rodrigues(vec[:3].reshape(3, 1))
    return make_transform(R, vec[3:6])


def rotation_error_deg(R_ref: np.ndarray, R: np.ndarray) -> float:
    R_err = np.asarray(R_ref, dtype=np.float64).T @ np.asarray(R, dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R_err)
    return float(np.linalg.norm(rvec) * 180.0 / math.pi)


def average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        return np.eye(4, dtype=np.float64)
    ts = np.asarray([T[:3, 3] for T in transforms], dtype=np.float64)
    rvecs = []
    for T in transforms:
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        rvecs.append(rvec.reshape(3))
    mean_rvec = np.mean(np.asarray(rvecs), axis=0)
    R_mean, _ = cv2.Rodrigues(mean_rvec.reshape(3, 1))
    return make_transform(R_mean, np.mean(ts, axis=0))


def fmt_vec(v: Any, digits: int = 3) -> str:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    return ", ".join(f"{x:.{digits}f}" for x in arr)


def rotation_angle_deg(R: np.ndarray) -> float:
    """R 的旋转角（0~180 度），用于统计位姿相对旋转覆盖度。"""
    value = (float(np.trace(R)) - 1.0) * 0.5
    return float(math.degrees(math.acos(max(-1.0, min(1.0, value)))))


def angle_span_deg(values: np.ndarray) -> float:
    """一组角度（度）在圆周上的最大连续覆盖跨度。"""
    if values.size == 0:
        return 0.0
    wrapped = (values.astype(np.float64) + 180.0) % 360.0 - 180.0
    ordered = np.sort(wrapped)
    gaps = np.diff(np.r_[ordered, ordered[0] + 360.0])
    return float(360.0 - np.max(gaps))


def circular_angle_abs_diff_deg(a: float, b: float) -> float:
    return float(abs((float(a) - float(b) + 180.0) % 360.0 - 180.0))
