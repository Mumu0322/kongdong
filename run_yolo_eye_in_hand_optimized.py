#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO + RGB-D + RGB 手眼标定的“眼在手上”孔中心定位脚本。

流程：初始画面选择任意数量的孔并按初始孔号顺序处理 -> 对当前孔导航到340 mm
粗定位位并逐孔居中 -> 对当前YOLO框的外环带深度拟合局部平面和中心 -> 到260 mm
精定位，仅用当前孔的RGB观测修正XY -> 移动到当前目标点 -> 进入下一个初始孔。

默认进入两阶段流程并启用实验运动；运动自动执行，仅在开始检测下一个已选孔前输入 m。
默认允许使用当前实验手眼结果，仅适用于现场诊断，不代表生产授权。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
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
)
from aubo_workbench.charuco_point_experiment import load_handeye_experiment_result  # noqa: E402
from aubo_workbench.config import ROBOT_CFG  # noqa: E402
from aubo_workbench.geometry import (  # noqa: E402
    invert_transform,
    make_transform,
    rotx,
    roty,
    rotz,
    transform_to_pose6_rzryrx,
)


DEFAULT_MODEL = Path(r"C:\MM\models\small_silu.pt")
DEFAULT_HANDEYE = Path(r"C:\MM\aubo_tools\data\e7_candidates\e7_handeye_candidate_current.json")
WINDOW = "YOLO eye-in-hand hole selection (click hole, Enter=confirm, Esc=quit)"
RUNS_DIR = ROOT.parent / "data" / "hole_localization_runs"
HOLE_DIAMETERS_MM = (65.0, 70.0, 75.0)
FINAL_BASE_Y_AFTER_Z_MM = 0.2
# 三孔逐孔安放时，所有低位横移前先抬到该安全余量。
THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM = 60.0
# 机器人到位检测只影响轮询响应，不改变控制器的运动轨迹。
ROBOT_STEADY_POLL_INTERVAL_S = 0.10


class TimingRecorder:
    """记录流程阶段耗时，并在每个阶段结束时同步到运行报告。"""

    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self.events: list[dict[str, Any]] = []
        self._report: dict[str, Any] | None = None

    def attach_report(self, report: dict[str, Any]) -> None:
        self._report = report
        self._sync()

    def snapshot(self) -> dict[str, Any]:
        aggregates: dict[str, dict[str, Any]] = {}
        for event in self.events:
            name = str(event["name"])
            bucket = aggregates.setdefault(
                name,
                {"count": 0, "total_elapsed_s": 0.0, "max_elapsed_s": 0.0},
            )
            elapsed_s = float(event["elapsed_s"])
            bucket["count"] += 1
            bucket["total_elapsed_s"] += elapsed_s
            bucket["max_elapsed_s"] = max(float(bucket["max_elapsed_s"]), elapsed_s)
        for bucket in aggregates.values():
            bucket["total_elapsed_s"] = round(float(bucket["total_elapsed_s"]), 6)
            bucket["max_elapsed_s"] = round(float(bucket["max_elapsed_s"]), 6)
        return {
            "total_elapsed_s": round(time.perf_counter() - self.started_at, 6),
            "events": list(self.events),
            "aggregates": aggregates,
        }

    def _sync(self) -> None:
        if self._report is not None:
            self._report["timing"] = self.snapshot()

    def scoped_snapshot(self, prefix: str) -> dict[str, Any]:
        events = [item for item in self.events if str(item["name"]).startswith(prefix)]
        return {
            "prefix": prefix,
            "sum_elapsed_s": round(sum(float(item["elapsed_s"]) for item in events), 6),
            "events": events,
        }

    def record(
        self, name: str, elapsed_s: float, status: str = "completed", **details: Any,
    ) -> None:
        event: dict[str, Any] = {
            "name": name,
            "elapsed_s": round(float(elapsed_s), 6),
            "status": status,
        }
        event.update(details)
        self.events.append(event)
        self._sync()
        print(
            f"[TIMING] {name}: {float(elapsed_s):.3f}s [{status}]",
            flush=True,
        )

    @contextmanager
    def measure(self, name: str, **details: Any):
        started_at = time.perf_counter()
        status = "completed"
        error: str | None = None
        try:
            yield
        except BaseException as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed_s = time.perf_counter() - started_at
            if error is not None:
                details = {**details, "error": error}
            self.record(name, elapsed_s, status=status, **details)

    def print_summary(self) -> None:
        snapshot = self.snapshot()
        print(
            f"[TIMING_SUMMARY] total={float(snapshot['total_elapsed_s']):.3f}s",
            flush=True,
        )
        aggregates = snapshot["aggregates"]
        ranked = sorted(
            aggregates.items(),
            key=lambda item: float(item[1]["total_elapsed_s"]),
            reverse=True,
        )
        for name, item in ranked[:12]:
            print(
                f"  {name}: total={float(item['total_elapsed_s']):.3f}s "
                f"count={int(item['count'])} max={float(item['max_elapsed_s']):.3f}s",
                flush=True,
            )

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
# 运动自动执行，仅在开始检测下一个已选孔前输入 m。
DEFAULT_TWO_STAGE_HOLE_LOCALIZATION = True
DEFAULT_EXECUTE_MOTION = True
DEFAULT_ALLOW_EXPERIMENTAL_HANDEYE = True


@dataclass(frozen=True)
class TwoStageConfig:
    coarse_height_mm: float = 340.0
    fine_height_mm: float = 260.0
    # 默认帧数按最近一次运行的稳定性下调；达到稳定质量门时还会提前结束。
    coarse_frames: int = 10
    fine_frames: int = 20
    preliminary_coarse_frames: int = 6
    height_tolerance_mm: float = 2.0
    center_tolerance_px: float = 5.0
    # 当前 Gemini 深度平面法向跨帧/跨视角重复性约 1–2°；粗阶段不应追逐到 0.5°。
    # 精定位仍锁定最终粗姿态并使用 RGB 进行 XY 微调。
    normal_tolerance_deg: float = 2.0
    max_z_corrections: int = 4
    min_coarse_valid: int = 8
    min_preliminary_coarse_valid: int = 5
    min_fine_valid: int = 12
    # 镀膜曲面工件的初始环带深度允许少量结构化噪声；粗/精阶段门限保持不变。
    initial_max_plane_rmse_mm: float = 3.5
    max_plane_rmse_mm: float = 3.5
    coarse_settle_frames: int = 5
    coarse_max_attempt_multiplier: int = 4
    max_coarse_center_scatter_p95_px: float = 0.8
    fine_stable_min_frames: int = 15
    fine_stable_center_scatter_p95_px: float = 0.6
    # 机械臂到达精定位高度后，先丢弃相机队列和末端微振动产生的预热帧。
    fine_settle_discard_frames: int = 10
    # 单孔精定位质量门失败时只重拍当前孔，不中断整个多孔流程。
    fine_retry_count: int = 2
    # 多次重拍仍略超严格门槛时，允许稳定但降级的结果继续执行并留痕。
    fine_degraded_max_center_scatter_p95_px: float = 0.6
    max_ellipse_residual_px: float = 0.9
    # 多孔镀膜表面边缘常有高光，椭圆残差会显著高于单孔测试；
    # YOLO中心散布仍由 max_fine_center_scatter_p95_px 严格约束。
    coarse_max_ellipse_residual_px: float = 2.5
    multi_fine_max_ellipse_residual_px: float = 2.5
    # 三孔中靠近视野边缘的孔可能只看到局部圆弧；只要弧段拟合
    # 稳定，允许用较低覆盖度参与圆心融合。
    multi_fine_min_ellipse_coverage_deg: float = 45.0
    # 镀膜和反光会造成YOLO框中心在帧间抖动；单孔精定位仍使用
    # 0.35 px，三孔批量输出单独保留可审计门槛。
    multi_fine_max_center_scatter_p95_px: float = 1.0
    # 为反光离群帧预留补采量；原始帧全部写入CSV，最终只融合稳定内点。
    # 只在稳定门未通过时补采；默认最多补4帧，而不是固定补12帧。
    multi_fine_extra_frames: int = 4
    multi_fine_match_tolerance_px: float = 80.0
    multi_fine_tracking_tolerance_px: float = 55.0
    # 粗定位三孔身份关联使用固定锚点；比精拍适当放宽，兼容粗拍时
    # YOLO框中心在局部倾斜和深度噪声下的少量变化。
    multi_coarse_tracking_tolerance_px: float = 70.0
    coarse_reference_lock_tolerance_px: float = 70.0
    min_coarse_ellipse_coverage_deg: float = 120.0
    min_ellipse_coverage_deg: float = 200.0
    max_fine_center_scatter_p95_px: float = 0.35
    # 精定位先使用粗定位点云中心在当前RGB相机中的投影作为身份锚点。
    # 该门限明显小于相邻孔间距，避免密集孔阵列中锁到邻孔。
    fine_pointcloud_anchor_tolerance_px: float = 30.0
    # 反光会让严格椭圆残差门（0.9 px）间歇性失败；只有在点云投影锚点
    # 已锁定目标且拟合仍满足较宽的几何门时，才允许该帧继续使用YOLO中心。
    fine_yolo_fallback_max_ellipse_residual_px: float = 2.5
    fine_yolo_fallback_min_ellipse_coverage_deg: float = 45.0
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
    surface_plane_point_camera_mm: np.ndarray | None = None


@dataclass
class Observation:
    stage: str
    frame_index: int
    center_px: np.ndarray
    ellipse: dict[str, Any] | None = None
    plane: PlaneEstimate | None = None
    timestamp_ns: int | None = None
    error: str | None = None
    center_source: str = "unknown"
    quality_note: str | None = None


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
    # 只在孔附近建立环带网格，避免每个粗定位帧都为整幅RGB-D图创建
    # 1280x800级别的坐标矩阵；环带定义与原实现保持一致。
    ring_outer_px = max(8.0, radius_px * 1.35)
    x0 = max(0, int(math.floor(ring_u - ring_outer_px - 1.0)))
    x1 = min(w, int(math.ceil(ring_u + ring_outer_px + 2.0)))
    y0 = max(0, int(math.floor(ring_v - ring_outer_px - 1.0)))
    y1 = min(h, int(math.ceil(ring_v + ring_outer_px + 2.0)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("孔环带超出深度图范围，无法提取点云")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - ring_u, yy - ring_v)
    ring = (rr >= max(4.0, radius_px * 1.08)) & (rr <= max(8.0, radius_px * 1.35))
    points = xyz_map_mm[y0:y1, x0:x1][ring]
    points = points[np.isfinite(points).all(axis=1)]
    points = points[(points[:, 2] > 100.0) & (points[:, 2] < 3000.0)]
    raw_ring_points = int(len(points))
    if len(points) < 12:
        raise ValueError("孔周围有效深度点不足，无法拟合局部切平面")

    # 镀膜伞具的孔壁/孔内深度会落入环带；它们通常比外表面离相机更远，
    # 混入后会把局部平面RMSE拉到二十多毫米。按前景深度簇筛选外表面，
    # 同时保留足够带宽覆盖孔面倾斜与局部曲率。
    front_surface_z = float(np.percentile(points[:, 2], 20.0))
    surface_band_mm = 8.0
    surface_points = points[np.abs(points[:, 2] - front_surface_z) <= surface_band_mm]
    if len(surface_points) >= max(80, int(len(points) * 0.10)):
        points = surface_points
    # 粗拍姿态只需要选中孔处的局部切平面。对单孔窄环带拟合整球半径
    # 数值病态，且镀膜反光会使球心/法向跳变；不能用它直接规划TCP姿态。
    normal, plane_rmse = fit_plane(points)
    plane_point = np.median(points, axis=0)
    sphere_center = sphere_radius = None
    sphere_rmse = None
    try:
        # 仍记录球面参考，供显式三孔模式的后续坐标推算使用；失败不影响
        # 单孔粗定位和TCP垂直孔面的姿态规划。
        sphere_center, sphere_radius, sphere_rmse = fit_sphere(points)
    except ValueError:
        pass

    ray_center = np.asarray(
        ring_center_xy if ray_center_xy is None else ray_center_xy,
        dtype=np.float64,
    ).reshape(2)
    ray = camera_ray(intrinsics, ray_center, already_undistorted=ray_center_is_undistorted)
    denominator = float(normal @ ray)
    if abs(denominator) < 1e-6:
        raise ValueError("孔中心视线与局部切平面近似平行")
    scale = float(normal @ plane_point) / denominator
    if scale <= 0.0 or not math.isfinite(scale):
        raise ValueError("孔中心局部切平面交点位于相机反向")
    point = ray * scale
    if not np.isfinite(point).all() or point[2] <= 0:
        raise ValueError("孔中心反投影得到无效深度")
    return point, {
        "ring_points": int(len(points)),
        "ring_points_raw": raw_ring_points,
        "front_surface_z_mm": front_surface_z,
        "plane_rmse_mm": plane_rmse,
        "plane_normal_camera": normal.tolist(),
        # 与孔中心 point_camera_mm 分开记录环带拟合平面上的代表点；
        # 当前中心仍由中心射线与该局部平面求交得到。
        "local_plane_point_camera_mm": plane_point.tolist(),
        "plane_point_camera_mm": point.tolist(),
        "surface_model": "local_tangent_plane",
        "sphere_center_camera_mm": sphere_center.tolist() if sphere_center is not None else None,
        "sphere_radius_mm": sphere_radius,
        "sphere_rmse_mm": sphere_rmse,
        "ring_center_px_distorted": [ring_u, ring_v],
        "ray_center_px": ray_center.tolist(),
        "ray_center_is_undistorted": bool(ray_center_is_undistorted),
    }


def choose_boxes(image: np.ndarray, detections: list[dict[str, Any]], count: int | None = None,
                 preselected_indices: list[int] | None = None,
                 locked_indices: list[int] | None = None,
                 return_clicks: bool = False) -> list[int] | tuple[list[int], dict[int, list[float]]] | None:
    """手动选择孔；count=None 时由用户点击任意数量并按 Enter 结束。"""
    required = None if count is None else max(1, int(count))
    if required is not None and len(detections) < required:
        raise ValueError(f"当前画面仅检测到 {len(detections)} 个孔，无法选择 {required} 个")
    if not detections:
        raise ValueError("当前画面未检测到孔，无法选择")
    selected = list(dict.fromkeys(preselected_indices or []))
    selected = [index for index in selected if 0 <= index < len(detections)]
    if required is not None:
        selected = selected[:required]
    locked = set(locked_indices or []) & set(selected)
    # 未手选的锁定参考孔仍以检测中心作锚点；输出孔以用户点击的真实孔中心作锚点。
    clicks: dict[int, list[float]] = {
        index: list(map(float, detections[index]["center"])) for index in locked
    }

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        distances = [np.hypot(x - d["center"][0], y - d["center"][1]) for d in detections]
        if not distances:
            return
        index = int(np.argmin(distances))
        if index in selected:
            if index not in locked:
                selected.remove(index)  # 再点一次即可取消该孔。
                clicks.pop(index, None)
        elif required is None or len(selected) < required:
            selected.append(index)
            clicks[index] = [float(x), float(y)]

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        view = image.copy()
        for i, d in enumerate(detections):
            x1, y1, x2, y2 = map(int, d["box"])
            if i in selected:
                order = selected.index(i) + 1
                color = (0, 255, 0) if order == 1 else (255, 220, 0)
            else:
                order = None
                color = (0, 180, 255)
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
            cv2.circle(view, tuple(map(int, d["center"])), 4, color, -1)
            cv2.putText(view, f"{i}: {d['class_name']} {d['confidence']:.2f}",
                        (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if order is not None:
                label = f"{order}: reference" if i in locked or order == 1 else f"{order}: output"
                cv2.putText(view, label, (x1, min(view.shape[0] - 10, y2 + 24)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
                if i in clicks:
                    cx, cy = map(int, clicks[i])
                    cv2.drawMarker(view, (cx, cy), color, cv2.MARKER_CROSS, 16, 2)
        if required is None:
            prompt = f"click hole center, select any number: {len(selected)}; Enter=confirm; Esc=quit"
        else:
            prompt = f"click hole center, select {required}: {len(selected)}/{required}; Enter=confirm; Esc=quit"
        cv2.putText(view, prompt,
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):
            if len(selected) >= 1 and (required is None or len(selected) == required):
                if return_clicks:
                    return selected, {index: clicks.get(index, list(map(float, detections[index]["center"])))
                                      for index in selected}
                return selected
        if key == 27:
            return None


def choose_box(image: np.ndarray, detections: list[dict[str, Any]]) -> int | None:
    """保留旧调用接口，单孔选择。"""
    selected = choose_boxes(image, detections, count=1)
    return selected[0] if selected else None


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
    expected_center_px: np.ndarray | list[float] | tuple[float, float] | None = None,
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

    expected_distorted = np.asarray(
        detection["center"] if expected_center_px is None else expected_center_px,
        dtype=np.float64,
    ).reshape(2)
    box_width = float(x2 - x1)
    box_height = float(y2 - y1)
    # 密集孔阵列中，一个YOLO框的扩展ROI会同时包含相邻孔。椭圆轮廓必须
    # 仍属于该检测框附近；否则看似“残差很小”的相邻孔会被错误地拿来使用。
    # 该门槛只用于身份关联，不是降低几何质量要求。
    max_raw_center_offset_px = max(12.0, 0.45 * min(box_width, box_height))
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
        raw_center_offset = float(np.linalg.norm(raw_center - expected_distorted))
        if raw_center_offset > max_raw_center_offset_px:
            continue
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
            "detection_center_offset_px": raw_center_offset,
            "max_detection_center_offset_px": max_raw_center_offset_px,
            "score": score,
            "center_coordinate_domain": "undistorted_pixel",
        }
        if best is None or score < best["score"]:
            best = item

    # 镀膜内壁会让Canny轮廓在上、下两段之间跳变；对于当前这类近圆孔，
    # 固定局部ROI内的霍夫圆心通常更稳定。它仍受同一身份锚点约束，绝不会
    # 在整张图中搜索到邻孔。椭圆拟合保留在上方，供非圆或倾斜明显的情况使用。
    min_radius = max(12, int(0.26 * min(box_width, box_height)))
    max_radius = max(min_radius + 2, int(0.72 * max(box_width, box_height)))
    circles = cv2.HoughCircles(
        cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(24.0, 0.70 * min(box_width, box_height)),
        param1=max(50.0, float(hi)), param2=20.0,
        minRadius=min_radius, maxRadius=max_radius,
    )
    hough_best: dict[str, Any] | None = None
    if circles is not None:
        edge_y, edge_x = np.nonzero(edges)
        raw_edge_points = np.column_stack((edge_x + xa, edge_y + ya)).astype(np.float64)
        for local_u, local_v, radius in np.asarray(circles[0], dtype=np.float64):
            raw_center = np.array([local_u + xa, local_v + ya], dtype=np.float64)
            raw_center_offset = float(np.linalg.norm(raw_center - expected_distorted))
            if raw_center_offset > max_raw_center_offset_px:
                continue
            # HoughCircles 的 dp=1.2 输出天然量化到约1.2 px，不能直接用于
            # 1 px级精定位。用其作为初值，在同一环带边缘做两次亚像素圆最小二乘。
            refined_center = raw_center.copy()
            refined_radius = float(radius)
            for _ in range(2):
                radial_distance = np.linalg.norm(raw_edge_points - refined_center, axis=1)
                annulus = np.abs(radial_distance - refined_radius) <= max(3.0, 0.08 * refined_radius)
                fit_points = raw_edge_points[annulus]
                if len(fit_points) < 30:
                    break
                local_fit = fit_points - np.array([xa, ya], dtype=np.float64)
                A = np.column_stack((2.0 * local_fit[:, 0], 2.0 * local_fit[:, 1], np.ones(len(local_fit))))
                b = np.sum(local_fit * local_fit, axis=1)
                try:
                    solution, *_ = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    break
                local_center = solution[:2]
                radius_sq = float(solution[2] + local_center @ local_center)
                if not math.isfinite(radius_sq) or radius_sq <= 0.0:
                    break
                refined_center = local_center + np.array([xa, ya], dtype=np.float64)
                refined_radius = math.sqrt(radius_sq)
            raw_center = refined_center
            radius = refined_radius
            raw_center_offset = float(np.linalg.norm(raw_center - expected_distorted))
            if raw_center_offset > max_raw_center_offset_px:
                continue
            radial_distance = np.linalg.norm(raw_edge_points - raw_center, axis=1)
            annulus = np.abs(radial_distance - radius) <= max(2.5, 0.06 * radius)
            if not np.any(annulus):
                continue
            annulus_points = raw_edge_points[annulus]
            residual = float(np.median(np.abs(radial_distance[annulus] - radius)))
            angles = np.mod(
                np.degrees(np.arctan2(annulus_points[:, 1] - raw_center[1], annulus_points[:, 0] - raw_center[0])),
                360.0,
            )
            coverage = float(len(np.unique(np.floor(angles / 10.0).astype(int))) * 10.0)
            center = (
                undistort_pixels(intrinsics, raw_center.reshape(1, 2), pixel_output=True)[0]
                if intrinsics is not None else raw_center
            )
            # 相机畸变在本工作区较小；圆轴用于圆心定位，姿态仍来自冻结深度局部面。
            item = {
                "center_px": center.tolist(),
                "axes_px": [float(2.0 * radius), float(2.0 * radius)],
                "angle_deg": 0.0,
                "center_px_distorted": raw_center.tolist(),
                "axes_px_distorted": [float(2.0 * radius), float(2.0 * radius)],
                "axes_px_distorted_major_minor": [float(2.0 * radius), float(2.0 * radius)],
                "angle_deg_distorted": 0.0,
                "legacy_center_px_undistorted": center.tolist(),
                "contour_undistortion_shift_px": [0.0, 0.0],
                "residual_px": residual,
                "coverage_deg": coverage,
                "roundness": 1.0,
                "contour_points": int(len(annulus_points)),
                "detection_center_offset_px": raw_center_offset,
                "max_detection_center_offset_px": max_raw_center_offset_px,
                "score": raw_center_offset / max(radius, 1.0) + residual / 2.0,
                "center_coordinate_domain": "undistorted_pixel",
                "fit_method": "hough_circle",
            }
            if hough_best is None or item["score"] < hough_best["score"]:
                hough_best = item
    # 只要霍夫圆具有足够的真实边缘覆盖，它优先作为精拍圆心；否则沿用轮廓椭圆。
    if hough_best is not None and hough_best["coverage_deg"] >= 80.0 and hough_best["residual_px"] <= 2.5:
        return hough_best
    return best


def _detection_roi_at(
    detection: dict[str, Any], center_px: np.ndarray | list[float] | tuple[float, float],
    size_px: np.ndarray | list[float] | tuple[float, float] | None = None,
) -> dict[str, Any]:
    """保留YOLO类别/置信度，但把拟合ROI重新居中到人工确认的孔位置。"""
    center = np.asarray(center_px, dtype=np.float64).reshape(2)
    x1, y1, x2, y2 = [float(value) for value in detection["box"]]
    if size_px is None:
        width, height = x2 - x1, y2 - y1
    else:
        width, height = np.asarray(size_px, dtype=np.float64).reshape(2)
    width, height = max(24.0, float(width)), max(24.0, float(height))
    result = dict(detection)
    result["center"] = center.tolist()
    result["box"] = [
        float(center[0] - width / 2.0), float(center[1] - height / 2.0),
        float(center[0] + width / 2.0), float(center[1] + height / 2.0),
    ]
    return result


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
                min_coverage_deg: float | None = None,
                max_residual_px: float | None = None) -> bool:
    coverage_gate = cfg.min_ellipse_coverage_deg if min_coverage_deg is None else float(min_coverage_deg)
    residual_gate = cfg.max_ellipse_residual_px if max_residual_px is None else float(max_residual_px)
    return bool(ellipse and ellipse["residual_px"] <= residual_gate
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
             ellipse: dict[str, Any] | None, title: str,
             reference_center_px: np.ndarray | None = None,
             reference_label: str = "pointcloud anchor") -> np.ndarray:
    view = image.copy()
    if detection is not None:
        x1, y1, x2, y2 = map(int, detection["box"])
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 180, 255), 2)
        detection_center = tuple(np.rint(np.asarray(detection["center"], dtype=np.float64)).astype(int))
        cv2.drawMarker(view, detection_center, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.putText(
            view, "YOLO", (detection_center[0] + 8, detection_center[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 0, 255), 1,
        )
    if ellipse is not None:
        display_center = ellipse.get("center_px_distorted", ellipse["center_px"])
        display_axes = ellipse.get("axes_px_distorted", ellipse["axes_px"])
        display_angle = ellipse.get("angle_deg_distorted", ellipse["angle_deg"])
        center = tuple(np.rint(display_center).astype(int))
        axes = tuple(max(1, int(round(value / 2.0))) for value in display_axes)
        cv2.ellipse(view, center, axes, float(display_angle), 0, 360, (0, 255, 0), 2)
        cv2.drawMarker(view, center, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    if reference_center_px is not None:
        reference = tuple(np.rint(np.asarray(reference_center_px, dtype=np.float64).reshape(2)).astype(int))
        cv2.circle(view, reference, 10, (255, 0, 0), 2)
        cv2.drawMarker(view, reference, (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(
            view, reference_label, (reference[0] + 10, reference[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 0, 0), 1,
        )
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
        time.sleep(ROBOT_STEADY_POLL_INTERVAL_S)
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
    """兼容旧流程的运动确认；当前顺序流程不再调用它。"""
    print(f"[MOTION_CONFIRM_REQUIRED] {label}", flush=True)
    return input(prompt).strip().lower()


def _request_next_hole_confirmation(current_hole_id: int, next_hole_id: int) -> str:
    """当前孔完成后，仅在开始检测下一个已选孔前暂停一次。"""
    print(
        f"[NEXT_HOLE_CONFIRM_REQUIRED] 当前孔={int(current_hole_id)}，"
        f"下一个检测孔={int(next_hole_id)}",
        flush=True,
    )
    return input(
        f"孔{int(current_hole_id)}已完成定位和目标点运动；"
        f"输入 m 开始检测孔{int(next_hole_id)}，其他任意键停止："
    ).strip().lower()


def _confirm_and_move_line(label: str, current: np.ndarray, target: np.ndarray, args: Any,
                           motion_session: Any, pose_session: Any, extra: str = "",
                           require_confirmation: bool = True,
                           motion_profile: str = "precision") -> np.ndarray:
    if motion_profile == "transit":
        speed_m_s = float(getattr(args, "transit_speed_m_s", args.speed_m_s))
        acc_m_s2 = float(getattr(args, "transit_acc_m_s2", args.acc_m_s2))
    elif motion_profile == "approach":
        speed_m_s = float(getattr(
            args, "approach_speed_m_s", getattr(args, "transit_speed_m_s", args.speed_m_s),
        ))
        acc_m_s2 = float(getattr(
            args, "approach_acc_m_s2", getattr(args, "transit_acc_m_s2", args.acc_m_s2),
        ))
    elif motion_profile == "precision":
        speed_m_s = float(args.speed_m_s)
        acc_m_s2 = float(args.acc_m_s2)
    else:
        raise ValueError(f"未知运动速度档位：{motion_profile}")
    if speed_m_s <= 0.0 or acc_m_s2 <= 0.0:
        raise ValueError(
            f"运动速度和加速度必须大于0：profile={motion_profile}, "
            f"speed={speed_m_s}, acc={acc_m_s2}"
        )
    _print_motion_preview(label, current, target, extra)
    if require_confirmation:
        command = _request_motion_confirmation(label, "输入 m 确认运动，其他任意键取消：")
        if command != "m":
            raise RuntimeError(f"用户取消：{label}")
    else:
        print("[MOTION] 自动执行，无需输入 m")
    print(
        f"[MOTION] profile={motion_profile}, speed={speed_m_s:.4f} m/s, "
        f"acc={acc_m_s2:.4f} m/s^2",
        flush=True,
    )
    from aubo_workbench.motion_control import sdk_ok
    response = motion_session.move_line(
        transform_to_sdk_pose_m_rad(target), speed_m_s, acc_m_s2,
    )
    print("[MOTION]", response)
    if not response or not sdk_ok(response[-1]):
        raise RuntimeError(f"{label} moveLine 下发失败：{response}")
    _, actual = _wait_robot_steady(pose_session)
    return actual




def _confirm_and_move_home(home: Any, args: Any, motion_session: Any, pose_session: Any) -> np.ndarray:
    snapshot, current = _require_safe_snapshot(pose_session)
    current_joints = np.asarray(snapshot.get("joints_rad", []), dtype=np.float64)
    home_joints = np.asarray(home.joints_rad, dtype=np.float64)
    home_target = pose_session.pose_sdk_to_transform_mm(home.tcp_pose_m_rad)
    if current_joints.size == home_joints.size and current_joints.size > 0:
        max_joint_error_deg = float(np.max(np.abs(current_joints - home_joints)) * 180.0 / math.pi)
        if max_joint_error_deg <= 0.5:
            tcp_position_error_mm = float(np.linalg.norm(current[:3, 3] - home_target[:3, 3]))
            tcp_axis_error_deg = _angle_deg(current[:3, 2], home_target[:3, 2])
            if tcp_position_error_mm > 5.0 or tcp_axis_error_deg > 2.0:
                print(
                    "\n[SAFETY WARNING] 原点关节一致，但当前活动TCP与保存原点TCP明显不一致：\n"
                    f"  TCP位置差={tcp_position_error_mm:.3f} mm，光轴差={tcp_axis_error_deg:.3f}°\n"
                    f"  当前TCP XYZ(mm)={np.round(current[:3, 3], 3).tolist()}\n"
                    f"  保存TCP XYZ(mm)={np.round(home_target[:3, 3], 3).tolist()}\n"
                    "  请确认当前活动TCP与手眼标定时一致；当前自动流程不会逐段暂停确认。",
                    flush=True,
                )
            print(f"[MOTION] 当前已在原点关节位，最大关节偏差={max_joint_error_deg:.3f}°，跳过回原点运动")
            return current
    _print_motion_preview("回机械臂原点（关节运动）", current, home_target,
                          f"home={home.name} created_at={home.created_at}")
    print("[MOTION] 自动执行回原点，无需输入 m", flush=True)
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




def _capture_initial_multi_hole_selection(
    pipeline: Any, align: Any, chain: Any, model: Any, confidence: float,
    run_dir: Path, max_plane_rmse_mm: float, count: int | None = None,
) -> tuple[Any, list[dict[str, Any]], np.ndarray, PlaneEstimate, Any]:
    """初始画面选择任意数量的孔，按 Enter 结束并记录每孔联合几何导航数据。"""
    required = None if count is None else max(1, int(count))
    bundle = None
    for _ in range(10):
        bundle = get_aligned_frame_bundle(pipeline, align, chain)
        if bundle is not None and bundle.intrinsics is not None:
            break
    if bundle is None or bundle.intrinsics is None:
        raise RuntimeError("无法获得带RGB内参的初始多孔 RGB-D 帧")
    detections = detect(model, bundle.color_bgr, confidence)
    if required is not None and len(detections) < required:
        raise RuntimeError(f"初始画面仅检测到 {len(detections)} 个孔，无法选择 {required} 个")
    selection = choose_boxes(
        bundle.color_bgr, detections, count=required, return_clicks=True,
    )
    if selection is None:
        raise RuntimeError("未选择初始孔")
    selected_indices, selection_clicks = selection
    selected = [detections[index] for index in selected_indices]
    class_ids = {int(item.get("class_id", -1)) for item in selected}
    if len(class_ids) != 1:
        raise RuntimeError("初始选择的目标不是同一个YOLO孔类别，无法进行稳定身份关联")

    holes: list[dict[str, Any]] = []
    points_camera: list[np.ndarray] = []
    normals_camera: list[np.ndarray] = []
    overlays: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for hole_id, (detection_index, detection) in enumerate(zip(selected_indices, selected), start=1):
        selection_click = np.asarray(selection_clicks[detection_index], dtype=np.float64).reshape(2)
        fit_detection = _detection_roi_at(detection, selection_click)
        ellipse = fit_hole_ellipse(
            bundle.color_bgr, fit_detection, bundle.intrinsics,
            expected_center_px=selection_click,
        )
        radius = max(
            float(fit_detection["box"][2] - fit_detection["box"][0]),
            float(fit_detection["box"][3] - fit_detection["box"][1]),
        ) / 2.0
        ring_center = tuple(
            ellipse["center_px_distorted"] if ellipse is not None else selection_click
        )
        ray_center = np.asarray(
            ellipse["center_px"] if ellipse is not None else selection_click,
            dtype=np.float64,
        )
        try:
            point, info = hole_camera_point(
                ring_center, bundle.xyz_map_mm, bundle.intrinsics, radius,
                ray_center_xy=ray_center, ray_center_is_undistorted=ellipse is not None,
            )
            plane = _plane_estimate_from_info(info, f"initial multi-hole {hole_id} plane normal")
        except Exception as exc:
            raise RuntimeError(f"初始孔{hole_id}点云几何计算失败：{exc}") from exc
        if plane.rmse_mm > float(max_plane_rmse_mm):
            raise RuntimeError(
                f"初始孔{hole_id}深度拟合RMSE过大：{plane.rmse_mm:.3f} mm "
                f"> {float(max_plane_rmse_mm):.3f} mm"
            )

        display_detection = dict(detection)
        display_detection["_hole_id"] = hole_id
        overlays.append((display_detection, ellipse))
        points_camera.append(np.asarray(point, dtype=np.float64))
        normals_camera.append(np.asarray(plane.normal_camera, dtype=np.float64))
        holes.append({
            "hole_id": hole_id,
            "initial_selection_order": hole_id,
            "class_id": int(detection["class_id"]),
            "initial_detection": dict(detection),
            "initial_center_px": list(map(float, detection["center"])),
            "initial_box": list(map(float, detection["box"])),
            "initial_selection_click_px": selection_click.tolist(),
            "tracking_anchor_px": list(map(float, detection["center"])),
            "tracking_roi_size_px": [
                float(detection["box"][2] - detection["box"][0]),
                float(detection["box"][3] - detection["box"][1]),
            ],
            "initial_point_camera_mm": np.asarray(point, dtype=np.float64).tolist(),
            "initial_plane_point_camera_mm": plane.point_camera_mm.tolist(),
            "initial_plane_normal_camera": plane.normal_camera.tolist(),
            "initial_plane_rmse_mm": float(plane.rmse_mm),
            "initial_ring_points": int(plane.ring_points),
            "initial_surface_model": plane.surface_model,
            "pointcloud_segmentation": "yolo_box_annular_depth_ring",
        })

    group_center_camera = np.mean(np.asarray(points_camera, dtype=np.float64), axis=0)
    group_normal_camera = _fuse_normals(normals_camera)
    group_plane = PlaneEstimate(
        point_camera_mm=group_center_camera,
        normal_camera=group_normal_camera,
        rmse_mm=float(np.max([float(item["initial_plane_rmse_mm"]) for item in holes])),
        ring_points=int(sum(int(item["initial_ring_points"]) for item in holes)),
        surface_model="initial_multi_hole_group",
    )
    cv2.imwrite(
        str(run_dir / "01_home_selected.png"),
        _overlay_multi(bundle.color_bgr, overlays, "initial multi-hole group selection"),
    )
    return bundle, holes, group_center_camera, group_plane, bundle.intrinsics


def _center_scatter_p95(observations: list[Observation]) -> float:
    """返回一组有效观测中心相对中位数的P95散布。"""
    centers = np.asarray([
        item.center_px for item in observations if item.error is None
    ], dtype=np.float64)
    if len(centers) < 2:
        return math.inf
    median = np.median(centers, axis=0)
    return float(np.percentile(np.linalg.norm(centers - median, axis=1), 95.0))


def _coarse_burst_stable(
    observations_by_hole: dict[int, list[Observation]],
    min_valid: int,
    max_scatter_p95_px: float,
) -> bool:
    """判断粗定位是否已达到可以提前结束的稳定状态。"""
    for observations in observations_by_hole.values():
        valid = [item for item in observations if item.error is None and item.plane is not None]
        if len(valid) < int(min_valid):
            return False
        if _center_scatter_p95(valid) > float(max_scatter_p95_px):
            return False
    return True


def _fine_burst_stable(
    observations_by_hole: dict[int, list[Observation]],
    min_valid: int,
    max_scatter_p95_px: float,
) -> bool:
    """判断RGB精定位多孔观测是否已足够稳定，可以停止继续补帧。"""
    for observations in observations_by_hole.values():
        valid = [item for item in observations if item.error is None and item.ellipse is not None]
        if len(valid) < int(min_valid):
            return False
        if _center_scatter_p95(valid) > float(max_scatter_p95_px):
            return False
    return True


def _capture_coarse_burst(pipeline: Any, align: Any, chain: Any, model: Any, confidence: float,
                          chosen: dict[str, Any], cfg: TwoStageConfig, run_dir: Path,
                          name: str,
                          initial_anchor_px: np.ndarray | None = None,
                          tracking_tolerance_px: float | None = None,
                          lock_anchor: bool = False,
                          ) -> tuple[list[Observation], np.ndarray, dict[str, Any] | None]:
    observations: list[Observation] = []
    latest_image: np.ndarray | None = None
    latest_detection: dict[str, Any] | None = None
    latest_ellipse: dict[str, Any] | None = None
    anchor = None if initial_anchor_px is None else np.asarray(initial_anchor_px, dtype=np.float64).reshape(2)
    locked_anchor = anchor.copy() if anchor is not None and lock_anchor else None
    for _ in range(cfg.coarse_settle_frames):
        get_aligned_frame_bundle(pipeline, align, chain)
    for attempt in range(cfg.coarse_frames * cfg.coarse_max_attempt_multiplier):
        valid_count = len([item for item in observations if item.error is None and item.plane is not None])
        if (
            valid_count >= cfg.coarse_frames
            or (
                valid_count >= cfg.min_coarse_valid
                and _coarse_burst_stable(
                    {1: observations}, cfg.min_coarse_valid,
                    cfg.max_coarse_center_scatter_p95_px,
                )
            )
        ):
            break
        bundle = get_aligned_frame_bundle(pipeline, align, chain)
        if bundle is None or bundle.intrinsics is None:
            continue
        latest_image = bundle.color_bgr
        if anchor is None:
            anchor = np.array([bundle.color_bgr.shape[1] / 2.0, bundle.color_bgr.shape[0] / 2.0])
            if lock_anchor:
                locked_anchor = anchor.copy()
        search_anchor = locked_anchor if lock_anchor and locked_anchor is not None else anchor
        detection = _nearest_detection(
            detect(model, bundle.color_bgr, confidence), search_anchor, chosen["class_id"],
        )
        if detection is None:
            observations.append(Observation(
                name, attempt, search_anchor, timestamp_ns=bundle.host_timestamp_ns, error="yolo_missing",
            ))
            continue
        detection_center = np.asarray(detection["center"], dtype=np.float64)
        tracking_distance = float(np.linalg.norm(detection_center - search_anchor))
        if tracking_tolerance_px is not None and tracking_distance > float(tracking_tolerance_px):
            observations.append(Observation(
                name, attempt, search_anchor, timestamp_ns=bundle.host_timestamp_ns,
                error="tracking_distance",
            ))
            continue
        ellipse = fit_hole_ellipse(bundle.color_bgr, detection, bundle.intrinsics)
        # 粗定位只负责将参考孔带到主点并测得局部深度法向。镀膜边缘的
        # 椭圆轮廓会在不同弧段间切换，不能再用它作为粗阶段的中心来源。
        # 使用YOLO框中心保证跨帧一致；椭圆圆心仅用于精定位。
        center = detection_center
        radius = max(detection["box"][2] - detection["box"][0], detection["box"][3] - detection["box"][1]) / 2.0
        try:
            ring_center = tuple(detection["center"])
            _, plane_info = hole_camera_point(
                ring_center, bundle.xyz_map_mm, bundle.intrinsics, radius,
                ray_center_xy=center, ray_center_is_undistorted=False,
            )
            plane = _plane_estimate_from_info(plane_info, "coarse plane normal")
            valid = plane.rmse_mm <= cfg.max_plane_rmse_mm
            observations.append(Observation(name, attempt, center, ellipse, plane, bundle.host_timestamp_ns,
                                            None if valid else "plane_quality"))
        except Exception as exc:
            observations.append(Observation(name, attempt, center, ellipse, timestamp_ns=bundle.host_timestamp_ns,
                                            error=f"plane_error:{exc}"))
        if not lock_anchor:
            anchor = detection_center
        latest_detection, latest_ellipse = detection, ellipse
    if latest_image is None:
        raise RuntimeError("粗定位期间未获得相机帧")
    cv2.imwrite(str(run_dir / f"{name}_overlay.png"), _overlay(latest_image, latest_detection, latest_ellipse, name))
    return observations, latest_image, latest_ellipse


def _capture_fine_burst(
    pipeline: Any, model: Any, confidence: float, chosen: dict[str, Any],
    cfg: TwoStageConfig, run_dir: Path,
    initial_anchor_px: np.ndarray | None = None,
    tracking_tolerance_px: float | None = None,
    lock_anchor: bool = False,
    name: str = "04_fine",
    stable_scatter_p95_px: float | None = None,
    allow_yolo_fallback: bool = False,
    fallback_min_coverage_deg: float | None = None,
    fallback_max_residual_px: float | None = None,
    settle_discard_frames: int = 0,
) -> tuple[list[Observation], Any, np.ndarray, dict[str, Any] | None]:
    observations: list[Observation] = []
    latest_bundle = None
    latest_detection = latest_ellipse = None
    latest_center_source = "none"
    anchor = None if initial_anchor_px is None else np.asarray(initial_anchor_px, dtype=np.float64).reshape(2)
    locked_anchor = anchor.copy() if anchor is not None and lock_anchor else None
    fallback_coverage_gate = (
        cfg.fine_yolo_fallback_min_ellipse_coverage_deg
        if fallback_min_coverage_deg is None else float(fallback_min_coverage_deg)
    )
    fallback_residual_gate = (
        cfg.fine_yolo_fallback_max_ellipse_residual_px
        if fallback_max_residual_px is None else float(fallback_max_residual_px)
    )
    for _ in range(max(0, int(settle_discard_frames))):
        # 只取RGB帧而不运行YOLO；这些帧用于跨越机器人到位后的相机/末端稳定期。
        get_rgb_frame_bundle(pipeline)
    for attempt in range(cfg.fine_frames * 3):
        valid_count = len([item for item in observations if item.error is None and item.ellipse is not None])
        if (
            valid_count >= cfg.fine_frames
            or (
                valid_count >= cfg.fine_stable_min_frames
                and _fine_burst_stable(
                    {1: observations}, cfg.fine_stable_min_frames,
                    cfg.fine_stable_center_scatter_p95_px
                    if stable_scatter_p95_px is None else float(stable_scatter_p95_px),
                )
            )
        ):
            break
        bundle = get_rgb_frame_bundle(pipeline)
        if bundle is None or bundle.intrinsics is None:
            continue
        latest_bundle = bundle
        if anchor is None:
            anchor = np.array([bundle.color_bgr.shape[1] / 2.0, bundle.color_bgr.shape[0] / 2.0])
            if lock_anchor:
                locked_anchor = anchor.copy()
        search_anchor = locked_anchor if lock_anchor and locked_anchor is not None else anchor
        detection = _nearest_detection(
            detect(model, bundle.color_bgr, confidence), search_anchor, chosen["class_id"],
        )
        if detection is None:
            observations.append(Observation(
                name, attempt, search_anchor, timestamp_ns=bundle.host_timestamp_ns, error="yolo_missing",
            ))
            continue
        detection_center = np.asarray(detection["center"], dtype=np.float64)
        tracking_distance = float(np.linalg.norm(detection_center - search_anchor))
        if tracking_tolerance_px is not None and tracking_distance > float(tracking_tolerance_px):
            observations.append(Observation(
                name, attempt, search_anchor, timestamp_ns=bundle.host_timestamp_ns,
                error="tracking_distance",
            ))
            continue
        ellipse = fit_hole_ellipse(bundle.color_bgr, detection, bundle.intrinsics)
        # 即使严格椭圆门失败，也保存当前检测用于最后的诊断叠加图；
        # 否则“检测到了但拟合没通过”会被误看成“YOLO没有找到孔”。
        latest_detection, latest_ellipse = detection, ellipse
        strict_ellipse_ok = _ellipse_ok(ellipse, cfg)
        relaxed_ellipse_ok = _ellipse_ok(
            ellipse, cfg,
            min_coverage_deg=fallback_coverage_gate,
            max_residual_px=fallback_residual_gate,
        )
        if not strict_ellipse_ok and not (
            allow_yolo_fallback and relaxed_ellipse_ok
        ):
            observations.append(Observation(name, attempt, detection_center, ellipse,
                                            timestamp_ns=bundle.host_timestamp_ns,
                                            error="ellipse_quality",
                                            center_source="rejected",
                                            quality_note=(
                                                "strict_ellipse_gate_failed;"
                                                f"fallback_gate={fallback_residual_gate:.3f}px/"
                                                f"{fallback_coverage_gate:.1f}deg"
                                            )))
            latest_center_source = "rejected_ellipse_quality"
            continue

        # 精定位中心统一来自 YOLO 检测框；椭圆拟合只负责质量门和诊断，
        # 不再把 ellipse["center_px"] 混入中心融合，避免两种中心在帧间跳变。
        # YOLO 中心属于原始畸变像素域；下游 pixel_to_base_plane 要求去畸变域，
        # 因此统一先用当前 RGB 内参转换到去畸变像素坐标。
        center = undistort_pixels(
            bundle.intrinsics, detection_center.reshape(1, 2), pixel_output=True,
        )[0]
        center_source = "yolo"
        quality_note = (
            None if strict_ellipse_ok
            else "strict_ellipse_rejected;accepted_by_pointcloud_anchor_and_relaxed_ellipse"
        )
        observations.append(Observation(
            name, attempt, center, ellipse,
            timestamp_ns=bundle.host_timestamp_ns,
            center_source=center_source,
            quality_note=quality_note,
        ))
        latest_center_source = center_source
        if not lock_anchor:
            # detection["center"] 属于原始像素域；不能用去畸变椭圆中心更新跟踪锚点。
            anchor = detection_center
    if latest_bundle is None:
        raise RuntimeError("精定位期间未获得 RGB 帧")
    reference_anchor = locked_anchor if lock_anchor and locked_anchor is not None else anchor
    cv2.imwrite(str(run_dir / f"{name}_overlay.png"),
                _overlay(
                    latest_bundle.color_bgr, latest_detection, latest_ellipse,
                    f"{name} RGB center={latest_center_source}",
                    reference_center_px=reference_anchor,
                ))
    return observations, latest_bundle.intrinsics, latest_bundle.color_bgr, latest_ellipse


def _capture_fine_with_recovery(
    pipeline: Any, model: Any, confidence: float, chosen: dict[str, Any],
    cfg: TwoStageConfig, run_dir: Path, hole: dict[str, Any], hole_id: int,
    processing_order: int, expected_anchor_px: np.ndarray,
    timing: TimingRecorder, rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """单孔精定位恢复流程：预热丢帧 -> 严格重拍 -> 稳定降级。"""
    attempts: list[dict[str, Any]] = []
    last_observations: list[Observation] = []
    last_intrinsics: Any = None
    retry_count = max(0, int(cfg.fine_retry_count))

    for attempt_index in range(retry_count + 1):
        attempt_number = attempt_index + 1
        fine_name = (
            f"hole_{hole_id:02d}_fine"
            if attempt_number == 1
            else f"hole_{hole_id:02d}_fine_retry_{attempt_number - 1}"
        )
        attempt_record: dict[str, Any] = {
            "attempt": attempt_number,
            "name": fine_name,
            "settle_discard_frames": int(cfg.fine_settle_discard_frames),
        }
        try:
            with timing.measure(
                f"hole_{hole_id:02d}/fine_capture_attempt_{attempt_number}",
                hole_id=hole_id,
                processing_order=processing_order,
                attempt=attempt_number,
            ):
                observations, fine_intrinsics, _, _ = _capture_fine_burst(
                    pipeline, model, confidence, chosen, cfg, run_dir,
                    initial_anchor_px=expected_anchor_px,
                    tracking_tolerance_px=cfg.fine_pointcloud_anchor_tolerance_px,
                    lock_anchor=True,
                    name=fine_name,
                    stable_scatter_p95_px=cfg.max_fine_center_scatter_p95_px,
                    allow_yolo_fallback=True,
                    fallback_min_coverage_deg=cfg.fine_yolo_fallback_min_ellipse_coverage_deg,
                    fallback_max_residual_px=cfg.fine_yolo_fallback_max_ellipse_residual_px,
                    settle_discard_frames=cfg.fine_settle_discard_frames,
                )
            rows.extend(_observation_rows(observations))
            _record_hole_tracking_event(
                hole, fine_name, expected_anchor_px, observations,
                fine_intrinsics, "undistorted_pixel_yolo_center",
            )
            last_observations = observations
            last_intrinsics = fine_intrinsics
            attempt_record.update({
                "capture_status": "completed",
                "total_frames": len(observations),
                "valid_frames": len([
                    item for item in observations
                    if item.error is None and item.ellipse is not None
                ]),
            })
        except Exception as exc:
            attempt_record.update({
                "capture_status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            attempts.append(attempt_record)
            if attempt_index < retry_count:
                print(
                    f"[FINE_RETRY] 孔{hole_id}第{attempt_number}次采集失败，"
                    f"准备重拍：{attempt_record['error']}",
                    flush=True,
                )
                continue
            break

        try:
            with timing.measure(
                f"hole_{hole_id:02d}/fine_fusion_attempt_{attempt_number}",
                hole_id=hole_id,
                processing_order=processing_order,
                attempt=attempt_number,
                quality_mode="strict",
            ):
                fine = _fuse_fine(last_observations, cfg)
            fine["fine_quality_status"] = "strict"
            fine["fine_quality_note"] = None
            fine["fine_recovery_attempts"] = list(attempts) + [attempt_record]
            attempt_record.update({
                "fusion_status": "strict_pass",
                "center_scatter_p95_px": float(fine["center_scatter_p95_px"]),
            })
            attempts.append(attempt_record)
            fine["fine_recovery_attempts"] = list(attempts)
            return {
                "success": True,
                "status": "strict",
                "fine": fine,
                "observations": last_observations,
                "intrinsics": last_intrinsics,
                "attempts": attempts,
                "error": None,
            }
        except Exception as exc:
            attempt_record.update({
                "fusion_status": "strict_failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            attempts.append(attempt_record)
            if attempt_index < retry_count:
                print(
                    f"[FINE_RETRY] 孔{hole_id}第{attempt_number}次精定位未通过，"
                    f"准备预热后重拍：{attempt_record['error']}",
                    flush=True,
                )

    degraded_error: str | None = None
    valid_count = len([
        item for item in last_observations
        if item.error is None and item.ellipse is not None
    ])
    if last_intrinsics is not None and valid_count >= int(cfg.fine_stable_min_frames):
        try:
            with timing.measure(
                f"hole_{hole_id:02d}/fine_fusion_degraded",
                hole_id=hole_id,
                processing_order=processing_order,
                quality_mode="degraded",
            ):
                fine = _fuse_fine(
                    last_observations, cfg,
                    max_center_scatter_p95_px=cfg.fine_degraded_max_center_scatter_p95_px,
                )
            if int(fine["valid_frames"]) < int(cfg.fine_stable_min_frames):
                raise RuntimeError(
                    f"降级融合有效帧不足：{fine['valid_frames']}/{cfg.fine_stable_min_frames}"
                )
            fine["fine_quality_status"] = "degraded_fine"
            fine["fine_quality_note"] = (
                f"严格精定位门未通过；稳定P95={float(fine['center_scatter_p95_px']):.3f}px，"
                f"降级门={float(cfg.fine_degraded_max_center_scatter_p95_px):.3f}px"
            )
            fine["fine_recovery_attempts"] = list(attempts)
            attempts.append({
                "attempt": len(attempts) + 1,
                "name": "degraded_fusion",
                "fusion_status": "degraded_pass",
                "center_scatter_p95_px": float(fine["center_scatter_p95_px"]),
            })
            fine["fine_recovery_attempts"] = list(attempts)
            print(
                f"[FINE_DEGRADED] 孔{hole_id}采用稳定降级精定位，"
                f"P95={float(fine['center_scatter_p95_px']):.3f}px",
                flush=True,
            )
            return {
                "success": True,
                "status": "degraded_fine",
                "fine": fine,
                "observations": last_observations,
                "intrinsics": last_intrinsics,
                "attempts": attempts,
                "error": None,
            }
        except Exception as exc:
            degraded_error = f"{type(exc).__name__}: {exc}"

    if degraded_error is None:
        degraded_error = (
            f"降级融合前有效帧不足：{valid_count}/{int(cfg.fine_stable_min_frames)}"
        )
    attempts.append({
        "attempt": len(attempts) + 1,
        "name": "degraded_fusion",
        "fusion_status": "degraded_failed",
        "error": degraded_error,
    })
    print(
        f"[FINE_DEFERRED] 孔{hole_id}多次精定位仍未获得可靠中心，"
        f"当前孔标记deferred并继续后续孔：{degraded_error}",
        flush=True,
    )
    return {
        "success": False,
        "status": "deferred_fine_quality",
        "fine": None,
        "observations": last_observations,
        "intrinsics": last_intrinsics,
        "attempts": attempts,
        "error": degraded_error,
    }


def _overlay_multi(
    image: np.ndarray,
    detections: list[tuple[dict[str, Any], dict[str, Any] | None]],
    title: str,
    pointcloud_centers_px: dict[int, np.ndarray] | None = None,
    pointcloud_medians_px: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """绘制多孔叠加图，并可显示粗阶段点云中心在当前视图中的投影。"""
    view = image.copy()
    pointcloud_centers_px = pointcloud_centers_px or {}
    pointcloud_medians_px = pointcloud_medians_px or {}
    for index, (detection, ellipse) in enumerate(detections, start=1):
        hole_id = int(detection.get("_hole_id", index))
        view = _overlay(view, detection, ellipse, f"{title}  hole-{hole_id}")
        yolo_center = tuple(np.rint(np.asarray(detection["center"], dtype=np.float64)).astype(int))
        # 橙色小点明确表示当前精定位实际使用的YOLO框中心；
        # _overlay中的红十字仍表示绿色拟合圆/椭圆的中心。
        cv2.circle(view, yolo_center, 5, (0, 140, 255), 2)
        cv2.putText(view, f"H{hole_id}", (yolo_center[0] + 8, yolo_center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        pointcloud_center = pointcloud_centers_px.get(hole_id)
        if pointcloud_center is not None:
            pointcloud_center = np.asarray(pointcloud_center, dtype=np.float64).reshape(2)
            if np.isfinite(pointcloud_center).all():
                pc_xy = tuple(np.rint(pointcloud_center).astype(int))
                # 蓝色十字：粗定位“YOLO中心射线与点云局部平面交点”。
                cv2.drawMarker(view, pc_xy, (255, 0, 0), cv2.MARKER_TILTED_CROSS, 22, 2)
                cv2.circle(view, pc_xy, 7, (255, 0, 0), 1)
                cv2.line(view, yolo_center, pc_xy, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(view, "PC", (pc_xy[0] + 8, pc_xy[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
        pointcloud_median = pointcloud_medians_px.get(hole_id)
        if pointcloud_median is not None:
            pointcloud_median = np.asarray(pointcloud_median, dtype=np.float64).reshape(2)
            if np.isfinite(pointcloud_median).all():
                median_xy = tuple(np.rint(pointcloud_median).astype(int))
                # 青色菱形：分割出的前表面环带点云三维中位点投影。
                cv2.drawMarker(view, median_xy, (255, 255, 0), cv2.MARKER_DIAMOND, 20, 2)
                cv2.circle(view, median_xy, 6, (255, 255, 0), 1)
                cv2.putText(view, "PC-med", (median_xy[0] + 8, median_xy[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2)
    if pointcloud_centers_px or pointcloud_medians_px:
        cv2.putText(
            view,
            "orange=YOLO  red=green fit  blue PC=ray-plane  cyan PC-med=cloud median",
            (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2,
        )
    return view


def _match_selected_holes_at_fine(
    detections: list[dict[str, Any]], tracked_holes: list[dict[str, Any]], intrinsics: Any,
    predicted_centers_px: dict[int, np.ndarray] | None = None,
    max_match_distance_px: float = 80.0,
) -> dict[int, dict[str, Any] | None]:
    """按手动选择的相对几何关系匹配精拍三孔，避免视角变化后跳到邻孔。

    第一个手选孔是粗拍参考孔，因此在精拍时应最接近RGB主点。其余孔使用
    相对参考孔的像素向量，并按参考孔检测框尺度缩放后进行匹配。
    """
    result = {int(item["hole_id"]): None for item in tracked_holes}
    if not detections or not tracked_holes:
        return result
    # 已知机器人运动时，以原点RGB-D测得的三维孔位投影作为精拍预测。
    # 这能处理大姿态变化下二维相对位置的旋转、缩放和透视变形。
    if predicted_centers_px:
        used: set[int] = set()
        for hole in tracked_holes:
            hole_id = int(hole["hole_id"])
            expected = predicted_centers_px.get(hole_id)
            if expected is None:
                continue
            remaining = [item for item in detections if id(item) not in used]
            if not remaining:
                continue
            matched = min(remaining, key=lambda item: float(np.linalg.norm(np.asarray(item["center"]) - expected)))
            if float(np.linalg.norm(np.asarray(matched["center"]) - expected)) <= float(max_match_distance_px):
                result[hole_id] = matched
                used.add(id(matched))
        return result
    reference = next((item for item in tracked_holes if int(item["hole_id"]) == 1), tracked_holes[0])
    principal = np.array([float(intrinsics.cx), float(intrinsics.cy)], dtype=np.float64)
    reference_detection = min(
        detections,
        key=lambda item: float(np.linalg.norm(np.asarray(item["center"], dtype=np.float64) - principal)),
    )
    result[int(reference["hole_id"])] = reference_detection
    used = {id(reference_detection)}
    reference_center = np.asarray(reference_detection["center"], dtype=np.float64)
    initial_reference_center = np.asarray(reference["initial_center_px"], dtype=np.float64)
    ref_box = np.asarray(reference.get("initial_box", reference_detection["box"]), dtype=np.float64)
    ref_area = max(1.0, (ref_box[2] - ref_box[0]) * (ref_box[3] - ref_box[1]))
    current_box = np.asarray(reference_detection["box"], dtype=np.float64)
    current_area = max(1.0, (current_box[2] - current_box[0]) * (current_box[3] - current_box[1]))
    scale = math.sqrt(current_area / ref_area)

    for hole in tracked_holes:
        hole_id = int(hole["hole_id"])
        if hole_id == int(reference["hole_id"]):
            continue
        expected = reference_center + scale * (
            np.asarray(hole["initial_center_px"], dtype=np.float64) - initial_reference_center
        )
        remaining = [item for item in detections if id(item) not in used]
        if not remaining:
            continue
        matched = min(remaining, key=lambda item: float(np.linalg.norm(np.asarray(item["center"]) - expected)))
        result[hole_id] = matched
        used.add(id(matched))
    return result


def _project_base_point_to_pixel(
    point_base_mm: np.ndarray, T_base_camera: np.ndarray, intrinsics: Any,
) -> np.ndarray:
    """将基坐标三维点投影到当前RGB原始像素坐标（包含镜头畸变）。"""
    T_camera_base = invert_transform(T_base_camera)
    point_camera = T_camera_base[:3, :3] @ np.asarray(point_base_mm, dtype=np.float64) + T_camera_base[:3, 3]
    if point_camera[2] <= 1e-6:
        raise ValueError("目标孔位于当前相机后方")
    projected, _ = cv2.projectPoints(
        point_camera.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
        _camera_matrix(intrinsics), _distortion(intrinsics),
    )
    return np.asarray(projected, dtype=np.float64).reshape(2)


def _plane_estimate_from_info(info: dict[str, Any], label: str) -> PlaneEstimate:
    """把单次环带点云拟合结果统一转换成跨帧融合使用的平面估计。"""
    return PlaneEstimate(
        point_camera_mm=np.asarray(info["plane_point_camera_mm"], dtype=np.float64),
        normal_camera=_unit(np.asarray(info["plane_normal_camera"], dtype=np.float64), label),
        rmse_mm=float(info["plane_rmse_mm"]),
        ring_points=int(info["ring_points"]),
        surface_model=str(info.get("surface_model", "local_tangent_plane")),
        sphere_center_camera_mm=(
            np.asarray(info["sphere_center_camera_mm"], dtype=np.float64)
            if info.get("sphere_center_camera_mm") is not None else None
        ),
        sphere_radius_mm=(
            float(info["sphere_radius_mm"])
            if info.get("sphere_radius_mm") is not None else None
        ),
        surface_plane_point_camera_mm=(
            np.asarray(info["local_plane_point_camera_mm"], dtype=np.float64)
            if info.get("local_plane_point_camera_mm") is not None else None
        ),
    )


def _wrap_angle_rad(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def _rotation_distance_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """返回两个姿态旋转矩阵之间的最小旋转角。"""
    relative = np.asarray(R_a, dtype=np.float64).reshape(3, 3).T @ np.asarray(R_b, dtype=np.float64).reshape(3, 3)
    cosine = np.clip((float(np.trace(relative)) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.degrees(math.acos(float(cosine))))


def _tcp_rotation_with_fixed_rz_for_camera_axis(
    camera_axis_base: np.ndarray,
    reference_T_base_tcp: np.ndarray,
    T_tcp_camera: np.ndarray,
    fixed_rz_rad: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """在固定 AUBO RZ 的约束下，让相机光轴精确指向目标方向。

    T_tcp_camera 的旋转部分把相机坐标转换到 TCP 坐标。固定 RZ 后，RX/RY
    仍有两个自由度，足够在正常手眼安装姿态下把相机 Z 轴对准每个孔的法向。
    两个解析分支中选择与当前 TCP 姿态旋转距离最小的分支，避免逐孔姿态跳变。
    """
    target_axis = _unit(camera_axis_base, "target camera optical axis")
    reference_R = np.asarray(reference_T_base_tcp, dtype=np.float64).reshape(4, 4)[:3, :3]
    R_tcp_camera = np.asarray(T_tcp_camera, dtype=np.float64).reshape(4, 4)[:3, :3]
    camera_axis_tcp = _unit(R_tcp_camera[:, 2], "camera optical axis in TCP")
    fixed_rz = (
        _matrix_to_rpy_zyx(reference_R)[2]
        if fixed_rz_rad is None else float(fixed_rz_rad)
    )

    # R_tcp = Rz(fixed_rz) Ry(ry) Rx(rx)，先把目标轴变换到固定RZ之后的坐标系。
    target_after_rz = rotz(-fixed_rz) @ target_axis
    cy, cz = float(camera_axis_tcp[1]), float(camera_axis_tcp[2])
    yz_radius = math.hypot(cy, cz)
    if yz_radius < 1e-8:
        raise RuntimeError("手眼安装使固定RZ无法调整相机光轴方向")
    if abs(float(target_after_rz[1])) > yz_radius + 1e-7:
        raise RuntimeError(
            "固定RZ约束下无法将相机光轴完全对准该孔法向："
            f"required_y={target_after_rz[1]:.6f}, reachable={yz_radius:.6f}"
        )

    phase = math.atan2(cz, cy)
    ratio = float(np.clip(target_after_rz[1] / yz_radius, -1.0, 1.0))
    delta = math.acos(ratio)
    rx_candidates = (delta - phase, -delta - phase)
    candidates: list[tuple[float, np.ndarray, float, float]] = []
    for rx in rx_candidates:
        q = rotx(rx) @ camera_axis_tcp
        if math.hypot(float(q[0]), float(q[2])) < 1e-8:
            continue
        ry = math.atan2(float(target_after_rz[0]), float(target_after_rz[2])) - math.atan2(
            float(q[0]), float(q[2])
        )
        R_tcp = rotz(fixed_rz) @ roty(ry) @ rotx(rx)
        axis_error_deg = float(math.degrees(math.acos(np.clip(
            float(_unit(R_tcp @ camera_axis_tcp) @ target_axis), -1.0, 1.0,
        ))))
        rz_error = abs(_wrap_angle_rad(_matrix_to_rpy_zyx(R_tcp)[2] - fixed_rz))
        if axis_error_deg <= 1e-5 and rz_error <= 1e-5:
            score = _rotation_distance_deg(reference_R, R_tcp)
            candidates.append((score, R_tcp, axis_error_deg, rz_error))

    if not candidates:
        raise RuntimeError("固定RZ求解未得到满足光轴和RZ约束的TCP姿态")
    _, R_best, axis_error_deg, rz_error = min(candidates, key=lambda item: item[0])
    return R_best, {
        "fixed_rz_rad": fixed_rz,
        "camera_axis_target_base": target_axis,
        "camera_axis_result_base": R_best @ camera_axis_tcp,
        "camera_axis_error_deg": axis_error_deg,
        "rz_error_rad": rz_error,
        "rotation_change_deg": _rotation_distance_deg(reference_R, R_best),
    }


def _plan_hole_tcp_pose_fixed_rz(
    point_base_mm: np.ndarray,
    normal_toward_camera_base: np.ndarray,
    reference_T_base_tcp: np.ndarray,
    T_tcp_camera: np.ndarray,
    *,
    fixed_rz_rad: float | None = None,
    camera_height_mm: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """规划单孔TCP位姿：相机光轴对准该孔，RZ固定且姿态变化取最小分支。"""
    R_tcp, info = _tcp_rotation_with_fixed_rz_for_camera_axis(
        -_unit(normal_toward_camera_base, "hole normal toward camera"),
        reference_T_base_tcp,
        T_tcp_camera,
        fixed_rz_rad,
    )
    point = np.asarray(point_base_mm, dtype=np.float64).reshape(3)
    if camera_height_mm is None:
        target = make_transform(R_tcp, point)
    else:
        R_base_camera = R_tcp @ np.asarray(T_tcp_camera, dtype=np.float64).reshape(4, 4)[:3, :3]
        camera_origin = point - float(camera_height_mm) * R_base_camera[:, 2]
        target = make_transform(R_base_camera, camera_origin) @ invert_transform(T_tcp_camera)
    info.update({
        "hole_point_base_mm": point,
        "camera_height_mm": None if camera_height_mm is None else float(camera_height_mm),
        "tcp_pose_m_rad": transform_to_sdk_pose_m_rad(target),
    })
    return target, info




def _apply_coarse_geometry_to_hole(
    hole: dict[str, Any], observations: list[Observation],
    T_base_camera: np.ndarray, cfg: TwoStageConfig,
    min_valid_frames: int | None = None,
) -> dict[str, Any]:
    """将某个孔在当前居中相机位采集的结果写回该孔记录。"""
    hole_id = int(hole["hole_id"])
    summary = _fuse_coarse(observations, cfg, min_valid_frames=min_valid_frames)
    R_base_camera = np.asarray(T_base_camera, dtype=np.float64)[:3, :3]
    t_base_camera = np.asarray(T_base_camera, dtype=np.float64)[:3, 3]
    camera_origin = t_base_camera.copy()
    point_camera = np.asarray(summary["plane_point_camera_mm"], dtype=np.float64)
    normal_camera = _unit(
        np.asarray(summary["plane_normal_camera"], dtype=np.float64),
        f"coarse hole {hole_id} normal",
    )
    point_base = R_base_camera @ point_camera + t_base_camera
    normal_base = _unit(R_base_camera @ normal_camera, f"coarse hole {hole_id} base normal")
    if float(normal_base @ (camera_origin - point_base)) < 0.0:
        normal_base = -normal_base
    local_plane_point_camera = summary.get("surface_plane_point_camera_mm")
    local_plane_point_base = (
        R_base_camera @ np.asarray(local_plane_point_camera, dtype=np.float64) + t_base_camera
        if local_plane_point_camera is not None else None
    )
    hole.update({
        "coarse_center_px": summary["center_px"],
        "coarse_center_camera_mm": point_camera,
        "coarse_center_base_mm": point_base,
        "coarse_plane_point_camera_mm": local_plane_point_camera,
        "coarse_plane_point_base_mm": local_plane_point_base,
        "coarse_normal_camera": normal_camera,
        "coarse_normal_toward_camera_base": normal_base,
        "coarse_plane_rmse_mm": summary["plane_rmse_median_mm"],
        "coarse_valid_frames": summary["valid_frames"],
        "coarse_total_frames": summary["total_frames"],
        "coarse_center_scatter_p95_px": summary["center_scatter_p95_px"],
        "coarse_ring_points_median": int(np.median([
            item.plane.ring_points for item in observations
            if item.error is None and item.plane is not None
        ])),
        "coarse_surface_model": summary["surface_model"],
        "coarse_sphere_center_camera_mm": summary.get("sphere_center_camera_mm"),
        "coarse_sphere_radius_mm": summary.get("sphere_radius_mm"),
    })
    return summary










def _ray_sphere_intersection_base(
    pixel_xy: np.ndarray, intrinsics: Any, T_base_camera: np.ndarray,
    sphere_center_base_mm: np.ndarray, sphere_radius_mm: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    ray_camera = camera_ray(intrinsics, pixel_xy, already_undistorted=True)
    origin = np.asarray(T_base_camera[:3, 3], dtype=np.float64)
    direction = _unit(T_base_camera[:3, :3] @ ray_camera, "base camera ray")
    center = np.asarray(sphere_center_base_mm, dtype=np.float64).reshape(3)
    offset = origin - center
    b = 2.0 * float(direction @ offset)
    c = float(offset @ offset - sphere_radius_mm * sphere_radius_mm)
    discriminant = b * b - 4.0 * c
    if discriminant < 0.0:
        # 单孔局部深度环带对整球半径的约束较弱，镀膜反光还可能使
        # 拟合半径略偏小。采用射线到球心的最近点作为局部曲面近似，
        # 并在报告中标记，避免三孔实验因单个孔直接中断。
        depth = float((center - origin) @ direction)
        if depth <= 0.0:
            raise RuntimeError(f"RGB射线与粗定位球面无交点且最近点在相机后方：D={discriminant:.3f}")
        point = origin + depth * direction
        intersection_mode = "closest_ray_fallback"
    else:
        roots = [(-b - math.sqrt(discriminant)) / 2.0, (-b + math.sqrt(discriminant)) / 2.0]
        positive = [value for value in roots if value > 0.0]
        if not positive:
            raise RuntimeError("RGB射线与粗定位球面的交点在相机后方")
        point = origin + min(positive) * direction
        intersection_mode = "sphere"
    normal = _unit(point - center, "hole sphere normal")
    if float(normal @ (origin - point)) < 0.0:
        normal = -normal
    return point, normal, intersection_mode


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


def _fuse_coarse(
    observations: list[Observation], cfg: TwoStageConfig,
    min_valid_frames: int | None = None,
) -> dict[str, Any]:
    valid = [item for item in observations if item.error is None and item.plane is not None]
    required = cfg.min_coarse_valid if min_valid_frames is None else int(min_valid_frames)
    if len(valid) < required:
        raise RuntimeError(f"粗定位有效帧不足：{len(valid)}/{required}")
    centers = np.asarray([item.center_px for item in valid], dtype=np.float64)
    plane_points = _fuse_vectors([item.plane.point_camera_mm for item in valid], "coarse plane points")
    normals = _fuse_normals([item.plane.normal_camera for item in valid])
    center = np.median(centers, axis=0)
    scatter = np.linalg.norm(centers - center, axis=1)
    surface_plane_points = [
        item.plane.surface_plane_point_camera_mm
        for item in valid
        if item.plane.surface_plane_point_camera_mm is not None
    ]
    sphere_centers = [item.plane.sphere_center_camera_mm for item in valid if item.plane.sphere_center_camera_mm is not None]
    sphere_radii = [item.plane.sphere_radius_mm for item in valid if item.plane.sphere_radius_mm is not None]
    return {
        "valid_frames": len(valid), "total_frames": len(observations), "center_px": center,
        "center_scatter_p95_px": float(np.percentile(scatter, 95)),
        "plane_point_camera_mm": plane_points, "plane_normal_camera": normals,
        "surface_plane_point_camera_mm": (
            _fuse_vectors(surface_plane_points, "surface plane points")
            if surface_plane_points else None
        ),
        "plane_rmse_median_mm": float(np.median([item.plane.rmse_mm for item in valid])),
        "surface_model": valid[0].plane.surface_model,
        "sphere_center_camera_mm": _fuse_vectors(sphere_centers, "sphere centers") if sphere_centers else None,
        "sphere_radius_mm": float(np.median(sphere_radii)) if sphere_radii else None,
    }


def _fuse_fine(observations: list[Observation], cfg: TwoStageConfig,
               max_center_scatter_p95_px: float | None = None) -> dict[str, Any]:
    raw_valid = [item for item in observations if item.error is None and item.ellipse is not None]
    if len(raw_valid) < cfg.min_fine_valid:
        raise RuntimeError(f"精定位有效帧不足：{len(raw_valid)}/{cfg.min_fine_valid}")
    raw_centers = np.asarray([item.center_px for item in raw_valid], dtype=np.float64)
    raw_median = np.median(raw_centers, axis=0)
    raw_distance = np.linalg.norm(raw_centers - raw_median, axis=1)
    summary = {
        "valid_frames_raw": len(raw_valid), "total_frames": len(observations),
        "center_px_raw": raw_median,
        "center_scatter_p95_px_raw": float(np.percentile(raw_distance, 95)),
    }
    scatter_gate = (
        cfg.max_fine_center_scatter_p95_px
        if max_center_scatter_p95_px is None else float(max_center_scatter_p95_px)
    )
    summary["center_scatter_gate_px"] = scatter_gate
    # 固定相机/工件下，真实圆心应形成单一紧密簇。用中位数+MAD识别反光
    # 离群帧；严格门槛仍是scatter_gate，不以放宽门槛换取“通过”。
    mad = float(np.median(np.abs(raw_distance - np.median(raw_distance))))
    robust_radius = min(
        scatter_gate,
        max(1.0e-6, float(np.median(raw_distance)) + 3.0 * 1.4826 * max(mad, 1.0e-6)),
    )
    keep_mask = raw_distance <= robust_radius
    filtered_valid = [item for item, keep in zip(raw_valid, keep_mask) if keep]

    def candidate_metrics(items: list[Observation]) -> tuple[np.ndarray, np.ndarray, float]:
        candidate_centers = np.asarray([item.center_px for item in items], dtype=np.float64)
        candidate_median = np.median(candidate_centers, axis=0)
        candidate_distance = np.linalg.norm(candidate_centers - candidate_median, axis=1)
        return candidate_median, candidate_distance, float(np.percentile(candidate_distance, 95))

    # MAD剔除可能破坏原本近似对称的双边高光分布，使中位数偏向一侧，
    # 进而出现“删掉帧后P95反而增大”。只在过滤结果确实更优时才采用它；
    # 无论选择哪组，最终仍必须通过原来的严格scatter_gate。
    raw_candidate = (raw_valid, raw_median, raw_distance, summary["center_scatter_p95_px_raw"])
    filtered_candidate = None
    if len(filtered_valid) >= cfg.min_fine_valid:
        filtered_median, filtered_distance, filtered_p95 = candidate_metrics(filtered_valid)
        filtered_candidate = (filtered_valid, filtered_median, filtered_distance, filtered_p95)
    if filtered_candidate is not None and filtered_candidate[3] < raw_candidate[3]:
        valid, median, distance, selected_p95 = filtered_candidate
        selection_rule = "mad_filtered_lower_p95"
    else:
        valid, median, distance, selected_p95 = raw_candidate
        selection_rule = "raw_lower_or_equal_p95"
    if len(valid) < cfg.min_fine_valid:
        raise RuntimeError(
            f"精定位稳定内点不足：{len(valid)}/{cfg.min_fine_valid}；"
            f"原始有效帧={len(raw_valid)}，原始P95={summary['center_scatter_p95_px_raw']:.3f}px，"
            f"严格门槛={scatter_gate:.3f}px"
        )
    # 当前精定位的有效中心全部按 YOLO 来源统计。兼容旧观测记录中的
    # ellipse / yolo_fallback 标签，但不再让它们改变实际中心来源。
    center_source_counts = {"yolo": len(valid)}
    relaxed_yolo_frames = sum(
        "strict_ellipse_rejected" in str(item.quality_note or "") for item in valid
    )
    strict_ellipse_frames = len(valid) - relaxed_yolo_frames
    center_source = "yolo"
    summary.update({
        "valid_frames": len(valid), "rejected_outlier_frames": len(raw_valid) - len(valid),
        "outlier_rule": "choose lower P95 of raw and MAD-filtered sets; strict gate unchanged",
        "fusion_selection": selection_rule,
        "mad_filtered_frames": len(filtered_valid),
        "mad_filtered_p95_px": (
            float(filtered_candidate[3]) if filtered_candidate is not None else None
        ),
        "center_px": median,
        "center_scatter_p95_px": float(selected_p95),
        "ellipse_residual_median_px": float(np.median([item.ellipse["residual_px"] for item in valid])),
        "ellipse_roundness_median": float(np.median([item.ellipse["roundness"] for item in valid])),
        "axes_px_median": np.median(np.asarray([item.ellipse["axes_px"] for item in valid]), axis=0),
        "center_source": center_source,
        "center_source_counts": center_source_counts,
        "yolo_frames": len(valid),
        "strict_ellipse_frames": int(strict_ellipse_frames),
        "yolo_relaxed_ellipse_frames": int(relaxed_yolo_frames),
        # 保留旧字段，便于已有报告解析器读取；它表示放宽椭圆质量门后
        # 仍采用 YOLO 中心的帧数，不表示中心来自椭圆。
        "yolo_fallback_frames": int(relaxed_yolo_frames),
    })
    if summary["center_scatter_p95_px"] > scatter_gate:
        raise RuntimeError(
            f"精定位稳定内点仍超门槛：P95={summary['center_scatter_p95_px']:.3f} px；"
            f"已剔除离群帧={summary['rejected_outlier_frames']}"
        )
    return summary


def _observation_rows(observations: list[Observation]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in observations:
        row: dict[str, Any] = {
            "stage": item.stage, "frame_index": item.frame_index, "timestamp_ns": item.timestamp_ns,
            "center_u_px": float(item.center_px[0]), "center_v_px": float(item.center_px[1]),
            "center_source": item.center_source, "quality_note": item.quality_note,
            "error": item.error,
        }
        if item.plane is not None:
            row.update({
                "plane_rmse_mm": item.plane.rmse_mm, "ring_points": item.plane.ring_points,
                "plane_point_z_mm": float(item.plane.point_camera_mm[2]),
                "surface_plane_point_z_mm": (
                    float(item.plane.surface_plane_point_camera_mm[2])
                    if item.plane.surface_plane_point_camera_mm is not None else None
                ),
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




def _record_hole_tracking_event(
    hole: dict[str, Any], stage: str, expected_anchor_px: np.ndarray,
    observations: list[Observation], intrinsics: Any, observation_domain: str,
) -> None:
    """把一次孔身份跟踪的预测锚点和实际观测写入该孔审计记录。"""
    expected = np.asarray(expected_anchor_px, dtype=np.float64).reshape(2)
    valid = [item for item in observations if item.error is None]
    event: dict[str, Any] = {
        "stage": stage,
        "expected_detection_center_px": expected.tolist(),
        "expected_center_coordinate_domain": "distorted_pixel",
        "expected_center_px_undistorted": undistort_pixels(
            intrinsics, expected.reshape(1, 2), pixel_output=True,
        )[0].tolist(),
        "observation_center_px": [
            np.asarray(item.center_px, dtype=np.float64).tolist() for item in observations
        ],
        "observation_center_coordinate_domain": observation_domain,
        "observation_center_sources": [item.center_source for item in observations],
        "quality_notes": [item.quality_note for item in observations if item.quality_note is not None],
        "valid_frames": len(valid),
        "total_frames": len(observations),
        "errors": [item.error for item in observations if item.error is not None],
    }
    hole.setdefault("tracking_events", []).append(event)


def _move_to_sequential_coarse_pose(
    hole_id: int, order: int, current_tcp: np.ndarray, target: np.ndarray,
    args: Any, motion_session: Any, pose_session: Any,
) -> np.ndarray:
    """将当前TCP安全移动到指定孔的340 mm粗定位位。"""
    if order == 1:
        return _confirm_and_move_line(
            f"移动到孔{hole_id}上方340 mm粗定位位",
            current_tcp, target, args, motion_session, pose_session,
            "初始选定孔的三维点投影导航；光轴对准该孔初始法向",
            require_confirmation=False,
            motion_profile="transit",
        )

    safe_z = max(float(current_tcp[2, 3]), float(target[2, 3])) + THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM
    actual = np.asarray(current_tcp, dtype=np.float64).copy()
    lift = actual.copy()
    lift[2, 3] = safe_z
    if abs(float(lift[2, 3] - actual[2, 3])) > 0.2:
        actual = _confirm_and_move_line(
            f"孔{hole_id}粗定位前抬升",
            actual, lift, args, motion_session, pose_session,
            "仅修改基坐标Z；离开上一个孔的目标点",
            require_confirmation=False,
            motion_profile="transit",
        )
    high_target = np.asarray(target, dtype=np.float64).copy()
    high_target[2, 3] = safe_z
    actual = _confirm_and_move_line(
        f"移动到孔{hole_id}上方安全高度",
        actual, high_target, args, motion_session, pose_session,
        "安全高度横移并调整到该孔的固定RZ姿态；不下降",
        require_confirmation=False,
        motion_profile="transit",
    )
    return _confirm_and_move_line(
        f"下降到孔{hole_id}上方340 mm粗定位位",
        actual, target, args, motion_session, pose_session,
        "仅基坐标Z下降；XY和孔法向姿态锁定",
        require_confirmation=False,
        motion_profile="transit",
    )


def _run_sequential_hole_workflow(
    args: Any, handeye: Any, model: Any, cfg: TwoStageConfig, run_dir: Path,
    report: dict[str, Any], timing: TimingRecorder, rows: list[dict[str, Any]],
    runtime: dict[str, Any],
    pose_session: Any, motion_session: Any, current_tcp: np.ndarray,
    initial_holes: list[dict[str, Any]], initial_intrinsics: Any,
) -> int:
    """按初始孔号逐个执行：粗定位 -> 精定位 -> 最终目标点 -> 下一个孔。"""
    if not initial_holes:
        raise RuntimeError("没有初始选定孔，无法执行顺序定位")

    fixed_rz_rad = _matrix_to_rpy_zyx(current_tcp[:3, :3])[2]
    results: list[dict[str, Any]] = []
    order_ids = [int(item["hole_id"]) for item in initial_holes]
    report["stages"]["sequential_plan"] = {
        "mode": "initial_selection_then_one_hole_complete",
        "hole_count": len(initial_holes),
        "hole_order": order_ids,
        "fixed_rz_rad": fixed_rz_rad,
        "tracking_identity_source": "initial_selection_order_and_initial_rgbd_3d_projection",
        "confirmation_policy": "only_before_starting_next_selected_hole",
        "camera_pipeline_policy": "reuse_single_rgbd_pipeline_for_coarse_and_rgb_fine",
    }

    def ensure_rgbd_pipeline() -> tuple[Any, Any, Any]:
        if runtime.get("rgbd_pipeline") is None:
            with timing.measure("camera/restart_rgbd_pipeline"):
                pipeline, align, chain = init_pipeline()
            runtime["rgbd_pipeline"] = pipeline
            runtime["align"] = align
            runtime["chain"] = chain
        return runtime["rgbd_pipeline"], runtime["align"], runtime["chain"]

    if not args.execute:
        preview_holes: list[dict[str, Any]] = []
        for order, hole in enumerate(initial_holes, start=1):
            point_base = np.asarray(hole["initial_center_base_mm"], dtype=np.float64)
            normal_base = _unit(
                np.asarray(hole["initial_plane_normal_base"], dtype=np.float64),
                f"preview hole {int(hole['hole_id'])} normal",
            )
            coarse_target, geometry = _plan_hole_tcp_pose_fixed_rz(
                point_base, normal_base, current_tcp, handeye.T_tcp_rgb_camera,
                fixed_rz_rad=fixed_rz_rad, camera_height_mm=cfg.coarse_height_mm,
            )
            preview_holes.append({
                "order": order,
                "hole_id": int(hole["hole_id"]),
                "initial_center_base_mm": point_base,
                "coarse_target_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(coarse_target),
                "coarse_pose_geometry": geometry,
            })
        report["stages"]["sequential_preview"] = {
            "hole_count": len(preview_holes),
            "hole_order": order_ids,
            "fixed_rz_rad": fixed_rz_rad,
            "holes": preview_holes,
        }
        report["status"] = "preview_complete"
        _write_report(run_dir, report, rows)
        print(f"[PREVIEW] 已写入顺序孔定位计划: {run_dir}")
        return 0

    for order, hole in enumerate(initial_holes, start=1):
        hole_id = int(hole["hole_id"])
        chosen = hole["initial_detection"]
        point_base = np.asarray(hole["initial_center_base_mm"], dtype=np.float64).reshape(3)
        normal_base = _unit(
            np.asarray(hole["initial_plane_normal_base"], dtype=np.float64),
            f"hole {hole_id} initial normal",
        )
        hole["tracking_identity"] = f"initial_selection_hole_{hole_id}"
        hole["processing_order"] = order

        rgbd_pipeline, align, chain = ensure_rgbd_pipeline()
        coarse_target, coarse_pose_geometry = _plan_hole_tcp_pose_fixed_rz(
            point_base, normal_base, current_tcp, handeye.T_tcp_rgb_camera,
            fixed_rz_rad=fixed_rz_rad, camera_height_mm=cfg.coarse_height_mm,
        )
        hole["initial_coarse_target_tcp_pose_m_rad"] = transform_to_sdk_pose_m_rad(coarse_target)
        hole["initial_coarse_pose_geometry"] = coarse_pose_geometry
        with timing.measure(
            f"hole_{hole_id:02d}/navigate_to_coarse",
            hole_id=hole_id,
            processing_order=order,
        ):
            current_tcp = _move_to_sequential_coarse_pose(
                hole_id, order, current_tcp, coarse_target,
                args, motion_session, pose_session,
            )

        coarse_captures: list[dict[str, Any]] = []
        final_center_offset = math.inf
        final_normal_error = math.inf
        for capture_index in range(1, 4):
            T_base_camera = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
            tracking_point_base = np.asarray(
                hole.get("coarse_center_base_mm", point_base), dtype=np.float64,
            ).reshape(3)
            expected_anchor_px = _project_base_point_to_pixel(
                tracking_point_base, T_base_camera, initial_intrinsics,
            )
            capture_name = f"hole_{hole_id:02d}_coarse_{capture_index}"
            with timing.measure(
                f"hole_{hole_id:02d}/coarse_capture_{capture_index}",
                hole_id=hole_id,
                processing_order=order,
                capture_index=capture_index,
            ):
                coarse_observations, _, _ = _capture_coarse_burst(
                    rgbd_pipeline, align, chain, model, args.confidence,
                    chosen, cfg, run_dir, capture_name,
                    initial_anchor_px=expected_anchor_px,
                    tracking_tolerance_px=cfg.multi_coarse_tracking_tolerance_px,
                    lock_anchor=True,
                )
            rows.extend(_observation_rows(coarse_observations))
            _record_hole_tracking_event(
                hole, capture_name, expected_anchor_px, coarse_observations,
                initial_intrinsics, "distorted_pixel_yolo_center",
            )
            T_base_camera = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
            with timing.measure(
                f"hole_{hole_id:02d}/coarse_pointcloud_geometry_{capture_index}",
                hole_id=hole_id,
                processing_order=order,
                capture_index=capture_index,
            ):
                summary = _apply_coarse_geometry_to_hole(
                    hole, coarse_observations, T_base_camera, cfg,
                )
            center_offset = float(np.linalg.norm(
                np.asarray(summary["center_px"], dtype=np.float64)
                - np.array([initial_intrinsics.cx, initial_intrinsics.cy])
            ))
            normal_error = _angle_deg(
                T_base_camera[:3, 2],
                -np.asarray(hole["coarse_normal_toward_camera_base"], dtype=np.float64),
            )
            final_center_offset = center_offset
            final_normal_error = normal_error
            coarse_captures.append({
                "capture_index": capture_index,
                "tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "center_offset_px": center_offset,
                "normal_error_deg": normal_error,
                "summary": summary,
            })
            if center_offset <= cfg.center_tolerance_px and normal_error <= cfg.normal_tolerance_deg:
                break
            if capture_index >= 3:
                break
            correction_target, correction_geometry = _plan_hole_tcp_pose_fixed_rz(
                np.asarray(hole["coarse_center_base_mm"], dtype=np.float64),
                np.asarray(hole["coarse_normal_toward_camera_base"], dtype=np.float64),
                current_tcp, handeye.T_tcp_rgb_camera,
                fixed_rz_rad=fixed_rz_rad, camera_height_mm=cfg.coarse_height_mm,
            )
            with timing.measure(
                f"hole_{hole_id:02d}/coarse_correction_motion_{capture_index}",
                hole_id=hole_id,
                processing_order=order,
                capture_index=capture_index,
            ):
                current_tcp = _confirm_and_move_line(
                    f"孔{hole_id}粗定位闭环校正 {capture_index}/2",
                    current_tcp, correction_target, args, motion_session, pose_session,
                    f"center offset={center_offset:.2f}px, normal error={normal_error:.3f}deg；"
                    f"camera_axis_error={float(correction_geometry['camera_axis_error_deg']):.5f}deg",
                    require_confirmation=False,
                    motion_profile="approach",
                )
        hole["coarse_captures"] = coarse_captures
        if final_center_offset > cfg.center_tolerance_px or final_normal_error > cfg.normal_tolerance_deg:
            raise RuntimeError(
                f"孔{hole_id}粗定位闭环后仍未通过质量门："
                f"offset={final_center_offset:.2f}px, normal={final_normal_error:.3f}deg"
            )

        coarse_plane_base_value = hole.get("coarse_plane_point_base_mm")
        coarse_plane_base = np.asarray(
            hole["coarse_center_base_mm"] if coarse_plane_base_value is None else coarse_plane_base_value,
            dtype=np.float64,
        ).reshape(3)
        coarse_normal_base = _unit(
            np.asarray(hole["coarse_normal_toward_camera_base"], dtype=np.float64),
            f"hole {hole_id} coarse normal",
        )
        hole["coarse_plane_point_base_mm"] = coarse_plane_base
        hole["coarse_normal_toward_camera_base"] = coarse_normal_base

        estimated_height = None
        for height_index in range(cfg.max_z_corrections):
            _, actual_tcp = _require_safe_snapshot(pose_session)
            estimated_height = camera_height_to_plane_mm(
                actual_tcp, handeye.T_tcp_rgb_camera, coarse_plane_base,
            )
            if abs(estimated_height - cfg.fine_height_mm) <= cfg.height_tolerance_mm:
                current_tcp = actual_tcp
                break
            z_target, _ = base_z_target_for_camera_height(
                actual_tcp, handeye.T_tcp_rgb_camera, coarse_plane_base, cfg.fine_height_mm,
            )
            with timing.measure(
                f"hole_{hole_id:02d}/move_to_fine_height_{height_index + 1}",
                hole_id=hole_id,
                processing_order=order,
                correction_index=height_index + 1,
            ):
                current_tcp = _confirm_and_move_line(
                    f"孔{hole_id}仅基坐标Z下降至{cfg.fine_height_mm:.0f} mm",
                    actual_tcp, z_target, args, motion_session, pose_session,
                    f"当前孔估计高度={estimated_height:.2f} mm；XY、姿态和RZ锁定",
                    require_confirmation=False,
                    motion_profile="approach",
                )
        _, current_tcp = _require_safe_snapshot(pose_session)
        estimated_height = camera_height_to_plane_mm(
            current_tcp, handeye.T_tcp_rgb_camera, coarse_plane_base,
        )
        if abs(estimated_height - cfg.fine_height_mm) > cfg.height_tolerance_mm:
            raise RuntimeError(
                f"孔{hole_id}仅Z修正后仍未达到精拍高度：{estimated_height:.2f} mm"
            )
        hole["fine_height_estimate_mm"] = estimated_height

        # RGB 精定位只需要 RGB 帧；get_rgb_frame_bundle 不会读取深度或生成点云，
        # 因此直接复用当前 RGB-D pipeline，避免每个孔重复 stop/start 两套相机管线。
        with timing.measure(
            f"hole_{hole_id:02d}/reuse_rgbd_pipeline_for_rgb",
            hole_id=hole_id,
            processing_order=order,
        ):
            fine_pipeline = rgbd_pipeline
        T_base_camera_fine = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
        expected_fine_anchor_px = _project_base_point_to_pixel(
            np.asarray(hole["coarse_center_base_mm"], dtype=np.float64),
            T_base_camera_fine, initial_intrinsics,
        )
        fine_recovery = _capture_fine_with_recovery(
            fine_pipeline, model, args.confidence, chosen, cfg, run_dir,
            hole, hole_id, order, expected_fine_anchor_px, timing, rows,
        )
        fine_observations = fine_recovery["observations"]
        fine_intrinsics = fine_recovery["intrinsics"]
        fine = fine_recovery["fine"]
        if not fine_recovery["success"] or fine is None or fine_intrinsics is None:
            deferred_result = {
                "status": "deferred_fine_quality",
                "hole_id": hole_id,
                "processing_order": order,
                "tracking_identity": hole["tracking_identity"],
                "initial_selection_order": hole.get("initial_selection_order"),
                "initial_center_px": hole.get("initial_center_px"),
                "initial_center_base_mm": hole.get("initial_center_base_mm"),
                "fine_quality_status": "deferred_fine_quality",
                "fine_quality_note": fine_recovery["error"],
                "fine_center_source": "unavailable",
                "fine_center_source_counts": {},
                "fine_recovery_attempts": fine_recovery["attempts"],
                "fine_valid_frames_last_attempt": len([
                    item for item in fine_observations
                    if item.error is None and item.ellipse is not None
                ]),
                "hole_center_base_mm": None,
                "hole_center_base_naive_mm": None,
                "coarse_center_base_mm": hole["coarse_center_base_mm"],
                "coarse_center_camera_mm": hole["coarse_center_camera_mm"],
                "pointcloud_center_base_mm": hole["coarse_center_base_mm"],
                "pointcloud_center_camera_mm": hole["coarse_center_camera_mm"],
                "pointcloud_center_definition": (
                    "coarse_yolo_center_ray_intersection_with_fused_local_pointcloud_plane"
                ),
                "coarse_plane_point_base_mm": coarse_plane_base,
                "coarse_plane_point_camera_mm": hole["coarse_plane_point_camera_mm"],
                "coarse_normal_camera": hole["coarse_normal_camera"],
                "coarse_normal_toward_camera_base": coarse_normal_base,
                "coarse_plane_rmse_mm": hole["coarse_plane_rmse_mm"],
                "coarse_valid_frames": hole["coarse_valid_frames"],
                "coarse_center_scatter_p95_px": hole["coarse_center_scatter_p95_px"],
                "coarse_captures": hole["coarse_captures"],
                "pointcloud_segmentation": hole["pointcloud_segmentation"],
                "fine_z_source": "not_applied",
                "estimated_height_mm": estimated_height,
                "fine_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "final_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "final_xy_motion": None,
                "final_z_motion": None,
                "final_y_trim_motion": None,
                "tracking_events": hole.get("tracking_events", []),
                "initial_detection": hole.get("initial_detection"),
                "deferred_reason": fine_recovery["error"],
                "timing": timing.scoped_snapshot(f"hole_{hole_id:02d}/"),
            }
            hole["final_result"] = deferred_result
            results.append(deferred_result)
            report["stages"][f"hole_{hole_id}"] = deferred_result
            completed_results = [item for item in results if item.get("status") == "completed"]
            deferred_results = [
                item for item in results if item.get("status") == "deferred_fine_quality"
            ]
            report["stages"]["processed_holes"] = {
                "completed_count": len(completed_results),
                "deferred_count": len(deferred_results),
                "total_count": len(results),
                "hole_order": [int(item["hole_id"]) for item in results],
                "holes": results,
            }
            _write_report(run_dir, report, rows)
            print(
                f"[SEQUENTIAL_HOLE] order={order} hole={hole_id} "
                f"status=deferred_fine_quality "
                f"coarse_center={np.round(np.asarray(hole['coarse_center_base_mm']), 3).tolist()} ",
                flush=True,
            )
            if order < len(initial_holes):
                next_hole_id = int(initial_holes[order]["hole_id"])
                with timing.measure(
                    f"hole_{hole_id:02d}/wait_next_hole_confirmation",
                    hole_id=hole_id,
                    next_hole_id=next_hole_id,
                ):
                    command = _request_next_hole_confirmation(hole_id, next_hole_id)
                if command != "m":
                    raise RuntimeError(f"用户在孔{hole_id}完成后停止流程")
            continue

        # 每个孔都按单孔精定位的严格中心稳定性门验收；若严格门失败但
        # 重拍后稳定，则fine_recovery会返回degraded_fine并把降级原因写入报告。
        with timing.measure(
            f"hole_{hole_id:02d}/fine_pixel_to_base",
            hole_id=hole_id,
            processing_order=order,
        ):
            naive_final_point_base = pixel_to_base_plane(
                fine["center_px"], fine_intrinsics, current_tcp, handeye.T_tcp_rgb_camera,
                coarse_plane_base, coarse_normal_base, center_is_undistorted=True,
            )
        diameter_px = float(max(float(fine["axes_px_median"][0]), float(fine["axes_px_median"][1])))
        diameter_estimate = diameter_px * estimated_height / (
            (fine_intrinsics.fx + fine_intrinsics.fy) / 2.0
        )
        nearest_diameter = min(HOLE_DIAMETERS_MM, key=lambda value: abs(value - diameter_estimate))
        T_base_camera_fine = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
        final_normal = (
            coarse_normal_base
            if float(coarse_normal_base @ T_base_camera_fine[:3, 2]) < 0.0
            else -coarse_normal_base
        )
        tilt_correction = None
        final_point_base = naive_final_point_base
        if cfg.enable_tilt_center_correction:
            with timing.measure(
                f"hole_{hole_id:02d}/tilt_center_correction",
                hole_id=hole_id,
                processing_order=order,
            ):
                final_point_base, tilt_correction = correct_projected_circle_center(
                    fine["center_px"], fine_intrinsics, current_tcp, handeye.T_tcp_rgb_camera,
                    coarse_plane_base, coarse_normal_base, nearest_diameter,
                    iterations=cfg.tilt_correction_iterations,
                    samples=cfg.tilt_correction_samples,
                    max_correction_mm=cfg.max_tilt_correction_mm,
                )

        fine_tcp = current_tcp.copy()
        final_xy_motion: dict[str, Any] | None = None
        final_z_motion: dict[str, Any] | None = None
        final_y_trim_motion: dict[str, Any] | None = None
        hole_pose, hole_pose_geometry = _plan_hole_tcp_pose_fixed_rz(
            final_point_base, coarse_normal_base, current_tcp,
            handeye.T_tcp_rgb_camera, fixed_rz_rad=fixed_rz_rad,
        )
        if args.move_final_xy:
            fixed_offset = (
                None if args.tcp_xy_offset_mm is None
                else (float(args.tcp_xy_offset_mm[0]), float(args.tcp_xy_offset_mm[1]))
            )
            xy_target, tcp_before = plan_final_tcp_xy(current_tcp, final_point_base, fixed_offset)
            correction_xy = xy_target[:2, 3] - final_point_base[:2]
            compensation = (
                f"ChArUco仿射模型修正={np.round(correction_xy, 3).tolist()} mm"
                if fixed_offset is None else f"显式固定补偿={list(fixed_offset)} mm"
            )
            with timing.measure(
                f"hole_{hole_id:02d}/final_motion_xy",
                hole_id=hole_id,
                processing_order=order,
            ):
                current_tcp = _confirm_and_move_line(
                    f"孔{hole_id}精定位后移动到最终XY",
                    current_tcp, xy_target, args, motion_session, pose_session,
                    f"保持孔{hole_id}精拍Z与姿态；{compensation}",
                    require_confirmation=False,
                    motion_profile="precision",
                )
            final_xy_motion = {
                "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(xy_target),
                "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "tcp_position_before_mm": tcp_before,
                "hole_center_base_mm": final_point_base,
                "compensation_mode": (
                    "charuco_affine_model" if fixed_offset is None else "fixed_offset_override"
                ),
                "xy_correction_mm": correction_xy,
                "charuco_model_source": str(CHARUCO_XY_MODEL_SOURCE) if fixed_offset is None else None,
                "charuco_model_matrix_2x2": CHARUCO_XY_MODEL_MATRIX if fixed_offset is None else None,
                "charuco_model_bias_mm": CHARUCO_XY_MODEL_BIAS_MM if fixed_offset is None else None,
                "tcp_xy_offset_mm": None if fixed_offset is None else list(fixed_offset),
            }
            z_target = plan_final_tcp_base_z(current_tcp, final_point_base)
            z_delta_mm = float(z_target[2, 3] - current_tcp[2, 3])
            with timing.measure(
                f"hole_{hole_id:02d}/final_motion_z",
                hole_id=hole_id,
                processing_order=order,
            ):
                current_tcp = _confirm_and_move_line(
                    f"孔{hole_id}移动到最终Z",
                    current_tcp, z_target, args, motion_session, pose_session,
                    f"保持最终XY与姿态；目标TCP基坐标Z={z_target[2, 3]:.3f} mm",
                    require_confirmation=False,
                    motion_profile="precision",
                )
            final_z_motion = {
                "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(z_target),
                "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "target_base_z_mm": float(z_target[2, 3]),
                "delta_base_z_mm": z_delta_mm,
                "motion_frame": "base_z_only",
            }
            y_trim_target = plan_final_tcp_base_y_trim(current_tcp)
            with timing.measure(
                f"hole_{hole_id:02d}/final_motion_y_plus_0_2mm",
                hole_id=hole_id,
                processing_order=order,
            ):
                current_tcp = _confirm_and_move_line(
                    f"孔{hole_id}最终基坐标+Y微调",
                    current_tcp, y_trim_target, args, motion_session, pose_session,
                    f"保持X、Z与姿态；基坐标Y增加 {FINAL_BASE_Y_AFTER_Z_MM:.3f} mm",
                    require_confirmation=False,
                    motion_profile="precision",
                )
            final_y_trim_motion = {
                "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(y_trim_target),
                "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "delta_base_y_mm": FINAL_BASE_Y_AFTER_Z_MM,
                "motion_frame": "base_y_only",
            }
        result = dict(fine)
        result.update({
            "status": "completed",
            "hole_id": hole_id,
            "processing_order": order,
            "tracking_identity": hole["tracking_identity"],
            "initial_selection_order": hole.get("initial_selection_order"),
            "initial_center_px": hole.get("initial_center_px"),
            "initial_center_base_mm": hole.get("initial_center_base_mm"),
            "fine_center_source": fine.get("center_source", "unknown"),
            "fine_center_source_counts": fine.get("center_source_counts", {}),
            "fine_quality_status": fine.get("fine_quality_status", "strict"),
            "fine_quality_note": fine.get("fine_quality_note"),
            "fine_recovery_attempts": fine.get("fine_recovery_attempts", []),
            "hole_center_base_mm": final_point_base,
            "hole_center_base_naive_mm": naive_final_point_base,
            "coarse_center_base_mm": hole["coarse_center_base_mm"],
            "coarse_center_camera_mm": hole["coarse_center_camera_mm"],
            "pointcloud_center_base_mm": hole["coarse_center_base_mm"],
            "pointcloud_center_camera_mm": hole["coarse_center_camera_mm"],
            "pointcloud_center_definition": (
                "coarse_yolo_center_ray_intersection_with_fused_local_pointcloud_plane"
            ),
            "coarse_plane_point_base_mm": coarse_plane_base,
            "coarse_plane_point_camera_mm": hole["coarse_plane_point_camera_mm"],
            "coarse_normal_camera": hole["coarse_normal_camera"],
            "coarse_normal_toward_camera_base": coarse_normal_base,
            "coarse_plane_rmse_mm": hole["coarse_plane_rmse_mm"],
            "coarse_valid_frames": hole["coarse_valid_frames"],
            "coarse_center_scatter_p95_px": hole["coarse_center_scatter_p95_px"],
            "coarse_captures": hole["coarse_captures"],
            "pointcloud_segmentation": hole["pointcloud_segmentation"],
            "fine_plane_intersection_mm": naive_final_point_base,
            "fine_xy_source": (
                "pointcloud_anchor_locked_yolo_center_on_coarse_local_plane"
            ),
            "fine_z_source": "coarse_per_hole_center_z",
            "tilt_center_correction": tilt_correction,
            "estimated_height_mm": estimated_height,
            "diameter_estimate_mm": diameter_estimate,
            "matched_diameter_mm": nearest_diameter,
            "plane_normal_toward_camera_base": final_normal,
            "hole_pose_m_rad": transform_to_sdk_pose_m_rad(hole_pose),
            "hole_pose_geometry": hole_pose_geometry,
            "fixed_rz_rad": fixed_rz_rad,
            "fine_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(fine_tcp),
            "final_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
            "final_xy_motion": final_xy_motion,
            "final_z_motion": final_z_motion,
            "final_y_trim_motion": final_y_trim_motion,
            "timing": timing.scoped_snapshot(f"hole_{hole_id:02d}/"),
            "tcp_target_pose_m_rad": (
                final_xy_motion["planned_tcp_pose_m_rad"]
                if final_xy_motion is not None else None
            ),
            "tracking_events": hole.get("tracking_events", []),
            "initial_detection": hole.get("initial_detection"),
        })
        hole["final_result"] = result
        results.append(result)
        report["stages"][f"hole_{hole_id}"] = result
        completed_results = [item for item in results if item.get("status") == "completed"]
        deferred_results = [
            item for item in results if item.get("status") == "deferred_fine_quality"
        ]
        report["stages"]["processed_holes"] = {
            "completed_count": len(completed_results),
            "deferred_count": len(deferred_results),
            "total_count": len(results),
            "hole_order": [int(item["hole_id"]) for item in results],
            "holes": results,
        }
        _write_report(run_dir, report, rows)
        print(
            f"[SEQUENTIAL_HOLE] order={order} hole={hole_id} "
            f"coarse_center={np.round(np.asarray(hole['coarse_center_base_mm']), 3).tolist()} "
            f"final_center={np.round(np.asarray(final_point_base), 3).tolist()} "
            f"tracking_events={len(hole.get('tracking_events', []))}",
            flush=True,
        )

        if order < len(initial_holes):
            next_hole_id = int(initial_holes[order]["hole_id"])
            with timing.measure(
                f"hole_{hole_id:02d}/wait_next_hole_confirmation",
                hole_id=hole_id,
                next_hole_id=next_hole_id,
            ):
                command = _request_next_hole_confirmation(hole_id, next_hole_id)
            if command != "m":
                raise RuntimeError(f"用户在孔{hole_id}完成后停止流程")

    completed_results = [item for item in results if item.get("status") == "completed"]
    deferred_results = [
        item for item in results if item.get("status") == "deferred_fine_quality"
    ]
    report["stages"]["sequential_holes"] = {
        "mode": "one_hole_complete_then_next",
        "hole_count": len(results),
        "completed_count": len(completed_results),
        "deferred_count": len(deferred_results),
        "hole_order": order_ids,
        "fixed_rz_rad": fixed_rz_rad,
        "tracking_identity_source": "initial_selection_order_and_locked_projected_anchor",
        "holes": results,
    }
    report["final_result"] = {
        "hole_count": len(results),
        "completed_count": len(completed_results),
        "deferred_count": len(deferred_results),
        "hole_order": order_ids,
        "fixed_rz_rad": fixed_rz_rad,
        "holes": results,
        "final_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
    }
    if deferred_results:
        report["status"] = (
            "completed_with_deferred_holes_experimental_handeye"
            if not handeye.validated_for_motion else "completed_with_deferred_holes"
        )
    else:
        report["status"] = "completed_experimental_handeye" if not handeye.validated_for_motion else "completed"
    _write_report(run_dir, report, rows)
    if deferred_results:
        print(
            f"\n[顺序孔定位结果] 流程已继续完成：成功={len(completed_results)}，"
            f"deferred={len(deferred_results)}；deferred孔未执行最终目标点运动。",
            flush=True,
        )
    else:
        print("\n[顺序孔定位结果] 已按初始孔号逐个完成粗定位、精定位和目标点运动。")
    print(json.dumps(_jsonable(report["final_result"]), ensure_ascii=False, indent=2))
    print(f"[DONE] 结果目录: {run_dir}")
    return 0


def run_two_stage_hole_localization(args: Any, handeye: Any, model: Any) -> int:
    """执行“初始多孔选择 -> 按孔号逐个粗/精定位 -> 逐孔目标点运动”的受确认流程。"""
    cfg = TwoStageConfig(
        coarse_height_mm=float(args.coarse_height_mm), fine_height_mm=float(args.fine_height_mm),
        coarse_frames=int(args.coarse_frames), fine_frames=int(args.fine_frames),
        preliminary_coarse_frames=int(args.preliminary_coarse_frames),
        multi_fine_extra_frames=int(args.fine_extra_frames),
        fine_settle_discard_frames=int(args.fine_settle_discard_frames),
        fine_retry_count=int(args.fine_retries),
    )
    precision_speed_m_s = float(args.speed_m_s)
    precision_acc_m_s2 = float(args.acc_m_s2)
    transit_speed_m_s = float(getattr(args, "transit_speed_m_s", precision_speed_m_s))
    transit_acc_m_s2 = float(getattr(args, "transit_acc_m_s2", precision_acc_m_s2))
    approach_speed_m_s = float(getattr(args, "approach_speed_m_s", 0.12))
    approach_acc_m_s2 = float(getattr(args, "approach_acc_m_s2", 0.35))
    if precision_speed_m_s <= 0.0 or precision_acc_m_s2 <= 0.0:
        raise ValueError(
            f"精确运动速度和加速度必须大于0：speed={precision_speed_m_s}, "
            f"acc={precision_acc_m_s2}"
        )
    if transit_speed_m_s <= 0.0 or transit_acc_m_s2 <= 0.0:
        raise ValueError(
            f"安全过渡速度和加速度必须大于0：speed={transit_speed_m_s}, "
            f"acc={transit_acc_m_s2}"
        )
    if approach_speed_m_s <= 0.0 or approach_acc_m_s2 <= 0.0:
        raise ValueError(
            f"非接触接近速度和加速度必须大于0：speed={approach_speed_m_s}, "
            f"acc={approach_acc_m_s2}"
        )
    if cfg.fine_settle_discard_frames < 0:
        raise ValueError("精定位预热丢弃帧数不能小于0")
    if cfg.fine_retry_count < 0:
        raise ValueError("精定位重试次数不能小于0")
    if cfg.fine_height_mm >= cfg.coarse_height_mm:
        raise ValueError("精定位高度必须小于粗定位高度")
    if cfg.coarse_frames < cfg.min_coarse_valid:
        raise ValueError(
            f"粗定位最大帧数必须不少于最小有效帧数：{cfg.coarse_frames} < {cfg.min_coarse_valid}"
        )
    if cfg.preliminary_coarse_frames < cfg.min_preliminary_coarse_valid:
        raise ValueError(
            "三孔共享粗定位帧数必须不少于其最小有效帧数："
            f"{cfg.preliminary_coarse_frames} < {cfg.min_preliminary_coarse_valid}"
        )
    if cfg.fine_frames < cfg.fine_stable_min_frames:
        raise ValueError(
            f"精定位最大帧数必须不少于稳定门帧数：{cfg.fine_frames} < {cfg.fine_stable_min_frames}"
        )
    if cfg.fine_frames + cfg.multi_fine_extra_frames < cfg.min_fine_valid:
        raise ValueError(
            "精定位最大总帧数必须不少于最小有效帧数："
            f"{cfg.fine_frames + cfg.multi_fine_extra_frames} < {cfg.min_fine_valid}"
        )
    run_dir = RUNS_DIR / f"two-stage-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "status": "running", "mode": "two_stage_hole_localization", "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"), "configuration": cfg.__dict__,
        "hole_count": None,
        "selection_mode": "click_any_count_then_enter",
        "handeye_path": str(args.handeye), "stages": {}, "motion_executed": bool(args.execute),
        "experimental_handeye_override": bool(args.allow_experimental_handeye),
        "motion_profiles": {
            "precision": {
                "speed_m_s": precision_speed_m_s,
                "acc_m_s2": precision_acc_m_s2,
                "used_for": "final_xy_final_z_final_y_trim",
            },
            "transit": {
                "speed_m_s": transit_speed_m_s,
                "acc_m_s2": transit_acc_m_s2,
                "used_for": "coarse_pose_navigation_between_selected_holes",
            },
            "approach": {
                "speed_m_s": approach_speed_m_s,
                "acc_m_s2": approach_acc_m_s2,
                "used_for": "coarse_correction_and_noncontact_descent_to_fine_height",
            },
        },
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
    timing = TimingRecorder()
    timing.attach_report(report)
    rows: list[dict[str, Any]] = []
    rgbd_pipeline = align = chain = None
    pipeline_runtime: dict[str, Any] = {
        "rgbd_pipeline": None, "align": None, "chain": None,
    }
    pose_session = motion_session = None
    try:
        from aubo_workbench.motion_control import AuboMotionSession, load_home_point
        from aubo_workbench.robot import AuboPoseSession

        with timing.measure("robot/load_home_point"):
            home = load_home_point()
        if home is None:
            raise RuntimeError("未找到 aubo_home_point.json；请先在机械臂运动界面设置原始点")
        with timing.measure("robot/connect_pose_session"):
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
            with timing.measure("robot/connect_motion_session"):
                motion_session = AuboMotionSession()
                motion_session.connect(ROBOT_CFG.ip, ROBOT_CFG.rpc_port, ROBOT_CFG.user,
                                      ROBOT_CFG.password, ROBOT_CFG.request_timeout_ms)
            with timing.measure("robot/move_home"):
                current_tcp = _confirm_and_move_home(home, args, motion_session, pose_session)
        else:
            current_tcp = initial_tcp
            print("[PREVIEW] 未指定 --execute：不会回原点或下发运动；请手动将机器人置于原点后核对规划。")

        with timing.measure("camera/start_rgbd_pipeline"):
            report["stages"]["rgbd_startup"] = {"status": "starting"}
            rgbd_pipeline, align, chain = init_pipeline()
            pipeline_runtime.update({
                "rgbd_pipeline": rgbd_pipeline, "align": align, "chain": chain,
            })
            report["stages"]["rgbd_startup"] = {"status": "ready"}
            identity = get_device_identity(rgbd_pipeline)
            expected = str(handeye.payload.get("camera_serial", "")).strip()
            actual = str(identity.get("serial_number", "")).strip()
            if expected and actual and expected != actual:
                raise RuntimeError(f"相机序列号不匹配：手眼={expected}，当前={actual}")
            report["camera"] = identity

        with timing.measure(
            "initial_selection/yolo_rgbd_pointcloud",
            selection_mode="click_any_count_then_enter",
        ):
            (
                initial_bundle,
                initial_selected_holes,
                selected_point_camera,
                initial_plane_camera,
                intrinsics,
            ) = _capture_initial_multi_hole_selection(
                rgbd_pipeline, align, chain, model, args.confidence, run_dir,
                cfg.initial_max_plane_rmse_mm,
            )
        report["hole_count"] = len(initial_selected_holes)
        chosen = initial_selected_holes[0]["initial_detection"]
        T_base_camera_home = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
        selected_point_base = T_base_camera_home[:3, :3] @ selected_point_camera + T_base_camera_home[:3, 3]
        plane_point_base = T_base_camera_home[:3, :3] @ initial_plane_camera.point_camera_mm + T_base_camera_home[:3, 3]
        camera_origin_home = T_base_camera_home[:3, 3]
        plane_normal_base = _unit(
            T_base_camera_home[:3, :3] @ initial_plane_camera.normal_camera,
            "initial group base normal",
        )
        if float(plane_normal_base @ (camera_origin_home - selected_point_base)) < 0.0:
            plane_normal_base = -plane_normal_base
        for hole in initial_selected_holes:
            point_camera = np.asarray(hole["initial_point_camera_mm"], dtype=np.float64).reshape(3)
            plane_point_camera = np.asarray(
                hole["initial_plane_point_camera_mm"], dtype=np.float64,
            ).reshape(3)
            normal_camera = _unit(
                np.asarray(hole["initial_plane_normal_camera"], dtype=np.float64),
                f"initial hole {int(hole['hole_id'])} base normal",
            )
            point_base = T_base_camera_home[:3, :3] @ point_camera + T_base_camera_home[:3, 3]
            normal_base = _unit(
                T_base_camera_home[:3, :3] @ normal_camera,
                f"initial hole {int(hole['hole_id'])} base normal",
            )
            if float(normal_base @ (camera_origin_home - point_base)) < 0.0:
                normal_base = -normal_base
            hole.update({
                "initial_center_base_mm": point_base.tolist(),
                "initial_plane_point_base_mm": (
                    T_base_camera_home[:3, :3] @ plane_point_camera + T_base_camera_home[:3, 3]
                ).tolist(),
                "initial_plane_normal_base": normal_base.tolist(),
                "tracking_identity": f"initial_selection_hole_{int(hole['hole_id'])}",
            })
        report["stages"]["home_selection"] = {
            "selection_mode": "initial_multi_hole_selection",
            "hole_count": len(initial_selected_holes),
            "selected": chosen,
            "selected_holes": initial_selected_holes,
            "group_center_camera_mm": selected_point_camera,
            "group_center_base_mm": selected_point_base,
            "group_plane_point_base_mm": plane_point_base,
            "group_plane_normal_base": plane_normal_base,
            "group_plane_rmse_mm": initial_plane_camera.rmse_mm,
            "group_ring_points": initial_plane_camera.ring_points,
            "group_surface_model": initial_plane_camera.surface_model,
        }

        return _run_sequential_hole_workflow(
            args, handeye, model, cfg, run_dir, report, timing, rows, pipeline_runtime,
            pose_session, motion_session, current_tcp,
            initial_selected_holes, intrinsics,
        )

    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report["timing"] = timing.snapshot()
        _write_report(run_dir, report, rows)
        raise
    finally:
        cleanup_started = time.perf_counter()
        try:
            stopped_pipeline_ids: set[int] = set()
            for pipeline in (
                rgbd_pipeline,
                pipeline_runtime.get("rgbd_pipeline"),
            ):
                if pipeline is None or id(pipeline) in stopped_pipeline_ids:
                    continue
                stopped_pipeline_ids.add(id(pipeline))
                try:
                    pipeline.stop()
                except Exception:
                    pass
            if pose_session is not None:
                pose_session.disconnect()
            if motion_session is not None:
                motion_session.disconnect()
            cv2.destroyAllWindows()
        finally:
            timing.record("runtime/cleanup", time.perf_counter() - cleanup_started)
            report["timing"] = timing.snapshot()
            try:
                _write_report(run_dir, report, rows)
            except Exception as exc:
                print(f"[TIMING] 最终报告写入失败：{type(exc).__name__}: {exc}", flush=True)
            timing.print_summary()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLO 选孔并计算眼在手目标位姿")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--handeye", type=Path, default=DEFAULT_HANDEYE)
    p.add_argument("--confidence", type=float, default=0.35)
    p.add_argument("--image", type=Path, help="离线 RGB 图；不指定则使用 Gemini RGB-D")
    p.add_argument("--intrinsics", type=Path, help="离线图像使用的 RGB 内参 JSON")
    p.add_argument("--target-depth-mm", type=float, help="离线无深度图时使用的平面深度（仅粗略预览）")
    p.add_argument("--execute", dest="execute", action="store_true", default=DEFAULT_EXECUTE_MOTION,
                   help="兼容参数：默认已启用真实运动，仅开始检测下一个已选孔前需要输入 m")
    p.add_argument("--no-execute", dest="execute", action="store_false",
                   help="仅预览：不连接运动控制或下发机器人运动")
    p.add_argument("--allow-experimental-handeye", dest="allow_experimental_handeye", action="store_true",
                   default=DEFAULT_ALLOW_EXPERIMENTAL_HANDEYE,
                   help="兼容参数：默认允许当前实验手眼结果用于现场诊断")
    p.add_argument("--require-validated-handeye", dest="allow_experimental_handeye", action="store_false",
                   help="只允许已获生产授权的手眼结果")
    p.add_argument("--speed-m-s", type=float, default=0.08,
                   help="精确靠近/闭环修正的 moveLine 速度(m/s)，默认0.08")
    p.add_argument("--acc-m-s2", type=float, default=0.25,
                   help="精确靠近/闭环修正的 moveLine 加速度(m/s²)，默认0.25")
    p.add_argument("--transit-speed-m-s", type=float, default=0.15,
                   help="孔间安全过渡及粗定位导航速度(m/s)，默认0.15")
    p.add_argument("--transit-acc-m-s2", type=float, default=0.45,
                   help="孔间安全过渡及粗定位导航加速度(m/s²)，默认0.45")
    p.add_argument("--approach-speed-m-s", type=float, default=0.12,
                   help="粗定位校正及非接触下降速度(m/s)，默认0.12")
    p.add_argument("--approach-acc-m-s2", type=float, default=0.35,
                   help="粗定位校正及非接触下降加速度(m/s²)，默认0.35")
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
    p.add_argument("--coarse-frames", type=int, default=10, help="两阶段最终粗定位最大有效RGB-D帧数")
    p.add_argument("--fine-frames", type=int, default=20, help="两阶段精定位最大有效RGB帧数")
    p.add_argument("--preliminary-coarse-frames", type=int, default=6,
                   help="三孔共享粗定位导航最大有效帧数；每孔最终居中采集仍单独执行")
    p.add_argument("--fine-extra-frames", type=int, default=4,
                   help="精定位稳定门未通过时的补采帧数")
    p.add_argument("--fine-settle-discard-frames", type=int, default=10,
                   help="每次精定位采集前丢弃的机器人/相机预热RGB帧数，默认10")
    p.add_argument("--fine-retries", type=int, default=2,
                   help="单孔精定位质量门失败后的自动重拍次数，默认2")
    # 兼容旧命令行参数；两阶段流程现在完全以用户点击后按 Enter 的数量为准。
    p.add_argument("--hole-count", type=int, default=None, help=argparse.SUPPRESS)
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
