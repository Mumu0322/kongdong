#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""点云鲁棒拟合：平面、球面。纯 numpy，不依赖相机/机械臂/GUI。"""

from __future__ import annotations

import math

import numpy as np


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
