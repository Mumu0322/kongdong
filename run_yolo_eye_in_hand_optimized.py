#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO + RGB-D + RGB 手眼标定的“眼在手上”孔中心定位脚本。

流程：真实 CAD 运动启动前先在当前位置用 5 帧 RGB/TCP 自动重新配准 CAD；通过质量门后回原点，
在 CAD 投影画面中选择一组目标孔 -> CAD 将整组孔
尽量居中并规划共同的340 mm深度采集位 -> 一次 RGB-D 采集整组孔的共同顶面平面，
深度只提供共享高度偏差 -> 逐孔使用 CAD 中心/法向规划260 mm位姿 -> 仅用当前孔
的RGB/YOLO观测修正最终XY -> 移动到当前目标点。

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
    init_rgb_handeye_pipeline,
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
DEFAULT_CAD_MODEL_JSON = Path(r"C:\MM\aubo_tools\data\cad_model\cad_hole_model.json")
WINDOW = "YOLO eye-in-hand hole selection (click hole, Enter=confirm, Esc=quit)"
RUNS_DIR = ROOT.parent / "data" / "hole_localization_runs"
CAD_REGISTRATION_RUNS_DIR = ROOT.parent / "data" / "cad_registration_runs"
CAD_MOTION_RUNS_DIR = ROOT.parent / "data" / "cad_motion_runs"
HOLE_DIAMETERS_MM = (65.0, 70.0, 75.0)
FINAL_TARGET_MODE_GRIPPER = "gripper"
FINAL_TARGET_MODE_NORMAL = "normal"
DEFAULT_FINAL_TARGET_MODE = FINAL_TARGET_MODE_GRIPPER
GRIPPER_BASE_X_OFFSET_MM = 64.0
GRIPPER_BASE_Z_OFFSET_MM = 50.0
FINAL_BASE_Y_AFTER_Z_MM = 0.3
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
DEFAULT_CAD_MOTION = False


@dataclass(frozen=True)
class TwoStageConfig:
    coarse_height_mm: float = 340.0
    fine_height_mm: float = 260.0
    # 默认帧数按最近一次运行的稳定性下调；达到稳定质量门时还会提前结束。
    coarse_frames: int = 10
    fine_frames: int = 20
    height_tolerance_mm: float = 2.0
    center_tolerance_px: float = 5.0
    # 当前 Gemini 深度平面法向跨帧/跨视角重复性约 1–2°；粗阶段不应追逐到 0.5°。
    # 精定位仍锁定最终粗姿态并使用 RGB 进行 XY 微调。
    normal_tolerance_deg: float = 2.0
    max_z_corrections: int = 4
    min_coarse_valid: int = 8
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
    # 粗定位身份关联使用固定锚点；比精拍适当放宽，兼容粗拍时
    # YOLO框中心在局部倾斜和深度噪声下的少量变化。
    multi_coarse_tracking_tolerance_px: float = 70.0
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


@dataclass(frozen=True)
class CadMotionConfig:
    """CAD 分组深度基准、目标位姿和 260 mm RGB 精定位的质量门。"""

    fine_height_mm: float = 260.0
    depth_height_mm: float = 340.0
    depth_frames: int = 8
    min_depth_valid_frames: int = 5
    min_depth_valid_holes: int = 4
    max_depth_plane_rmse_mm: float = 3.5
    max_depth_height_scatter_p95_mm: float = 2.0
    group_view_margin_px: float = 35.0
    fine_frames: int = 12
    min_fine_valid_frames: int = 6
    max_fine_center_scatter_p95_px: float = 1.5
    yolo_match_tolerance_px: float = 70.0
    settle_discard_frames: int = 10
    max_registration_center_p95_mm: float = 1.5

    def __post_init__(self) -> None:
        if float(self.fine_height_mm) <= 0.0:
            raise ValueError("CAD 精定位高度必须大于0")
        if float(self.depth_height_mm) <= float(self.fine_height_mm):
            raise ValueError("CAD 深度采集高度必须大于精定位高度")
        if int(self.depth_frames) < 1:
            raise ValueError("CAD 深度采集帧数必须大于0")
        if int(self.min_depth_valid_frames) < 1:
            raise ValueError("CAD 深度采集最少有效帧数必须大于0")
        if int(self.min_depth_valid_frames) > int(self.depth_frames):
            raise ValueError("CAD 深度采集最少有效帧数不能超过总帧数")
        if int(self.min_depth_valid_holes) < 1:
            raise ValueError("CAD 深度采集每帧最少有效孔数必须大于0")
        if float(self.max_depth_plane_rmse_mm) <= 0.0:
            raise ValueError("CAD 深度平面 RMSE 门限必须大于0")
        if float(self.max_depth_height_scatter_p95_mm) <= 0.0:
            raise ValueError("CAD 深度高度稳定性门限必须大于0")
        if float(self.group_view_margin_px) < 0.0:
            raise ValueError("CAD 分组视野边缘余量不能为负数")
        if int(self.fine_frames) < 1:
            raise ValueError("CAD 精定位帧数必须大于0")
        if int(self.min_fine_valid_frames) < 4:
            raise ValueError("CAD 精定位至少需要4帧有效YOLO观测")
        if int(self.min_fine_valid_frames) > int(self.fine_frames):
            raise ValueError("CAD 精定位最少有效帧数不能超过总帧数")
        if float(self.max_fine_center_scatter_p95_px) <= 0.0:
            raise ValueError("CAD 精定位中心稳定性门限必须大于0")
        if float(self.yolo_match_tolerance_px) <= 0.0:
            raise ValueError("CAD/YOLO 关联距离门限必须大于0")


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


def apply_final_point_base_offsets(
    point_base: np.ndarray,
    delta_x_mm: float = GRIPPER_BASE_X_OFFSET_MM,
    delta_z_mm: float = GRIPPER_BASE_Z_OFFSET_MM,
) -> np.ndarray:
    """在规划最终运动前，先对最终点施加基坐标 X/Z 偏移。"""
    target = np.asarray(point_base, dtype=np.float64).reshape(3).copy()
    target[0] += float(delta_x_mm)
    target[2] += float(delta_z_mm)
    return target


def final_point_offsets_for_mode(mode: str) -> tuple[float, float]:
    """返回最终点在基坐标 X/Z 方向的偏置，单位 mm。"""
    normalized = str(mode).strip().lower()
    if normalized == FINAL_TARGET_MODE_GRIPPER:
        return GRIPPER_BASE_X_OFFSET_MM, GRIPPER_BASE_Z_OFFSET_MM
    if normalized == FINAL_TARGET_MODE_NORMAL:
        return 0.0, 0.0
    raise ValueError(
        f"未知最终点模式：{mode!r}；可选模式为 "
        f"{FINAL_TARGET_MODE_GRIPPER!r} 或 {FINAL_TARGET_MODE_NORMAL!r}"
    )


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


def _cad_fine_overlay(
    image: np.ndarray,
    hole_id: str,
    frame_index: int,
    expected_center_px: np.ndarray,
    detection: dict[str, Any] | None,
    *,
    all_detections: list[dict[str, Any]] | None = None,
    median_center_px: np.ndarray | None = None,
    expected_radius_px: float | None = None,
    distance_px: float | None = None,
    valid_frames: int | None = None,
    total_frames: int | None = None,
    scatter_p95_px: float | None = None,
    mean_distance_px: float | None = None,
    summary: bool = False,
) -> np.ndarray:
    """绘制 CAD 260 mm 精定位的可审计叠加图。

    图像坐标均使用当前 RGB 原始/畸变像素域：青色是 CAD 投影中心，
    橙色是 YOLO 检测框，绿色是多帧 YOLO 中位中心，红色箭头是本帧
    CAD 投影到 YOLO 中心的偏差。
    """

    view = np.asarray(image).copy()
    height, width = view.shape[:2]

    for item in all_detections or []:
        if item is detection:
            continue
        try:
            x1, y1, x2, y2 = [int(round(value)) for value in item["box"]]
            cv2.rectangle(view, (x1, y1), (x2, y2), (120, 120, 120), 1, cv2.LINE_AA)
        except (KeyError, TypeError, ValueError, cv2.error):
            continue

    expected = np.asarray(expected_center_px, dtype=np.float64).reshape(2)
    expected_point = tuple(np.rint(expected).astype(int))
    if expected_radius_px is not None and np.isfinite(float(expected_radius_px)):
        cv2.circle(
            view,
            expected_point,
            max(3, int(round(float(expected_radius_px)))),
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.drawMarker(view, expected_point, (255, 255, 0), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
    cv2.putText(
        view,
        "CAD expected",
        (expected_point[0] + 10, expected_point[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )

    detection_center: np.ndarray | None = None
    if detection is not None:
        try:
            x1, y1, x2, y2 = [int(round(value)) for value in detection["box"]]
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 165, 255), 2, cv2.LINE_AA)
            detection_center = np.asarray(detection["center"], dtype=np.float64).reshape(2)
            detection_point = tuple(np.rint(detection_center).astype(int))
            cv2.drawMarker(
                view, detection_point, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA,
            )
            cv2.putText(
                view,
                f"YOLO {float(detection.get('confidence', 0.0)):.2f}",
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 165, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.arrowedLine(
                view,
                expected_point,
                detection_point,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
                tipLength=0.18,
            )
        except (KeyError, TypeError, ValueError, cv2.error):
            detection_center = None

    if median_center_px is not None:
        median = np.asarray(median_center_px, dtype=np.float64).reshape(2)
        median_point = tuple(np.rint(median).astype(int))
        cv2.drawMarker(view, median_point, (0, 255, 0), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
        cv2.circle(view, median_point, 8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(
            view,
            "YOLO median",
            (median_point[0] + 10, median_point[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    lines = [
        f"CAD FINE 260 | hole={hole_id} | frame={int(frame_index):02d}",
        "cyan=CAD expected | orange=YOLO box | green=YOLO median | red=error",
    ]
    if distance_px is not None:
        lines.append(f"CAD->YOLO={float(distance_px):.2f}px")
    if valid_frames is not None and total_frames is not None:
        lines.append(
            f"valid={int(valid_frames)}/{int(total_frames)}"
            + (f"  P95={float(scatter_p95_px):.2f}px" if scatter_p95_px is not None else "")
            + (f"  mean={float(mean_distance_px):.2f}px" if mean_distance_px is not None else "")
        )
    if summary:
        lines.append("SUMMARY: median YOLO center is used for final XY correction")
    for index, line in enumerate(lines):
        y = 30 + index * 24
        cv2.putText(view, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(view, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 1, cv2.LINE_AA)

    # Keep the diagnostic labels visible even when the expected point is near an edge.
    cv2.rectangle(view, (0, 0), (min(width - 1, 720), min(height - 1, 112)), (0, 0, 0), 1)
    return view


def _save_cad_fine_overlay(path: Path, image: np.ndarray, **kwargs: Any) -> str | None:
    """保存精定位图；可视化失败只记警告，不阻断机器人质量门。"""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        overlay = _cad_fine_overlay(image, **kwargs)
        if not cv2.imwrite(str(path), overlay):
            raise RuntimeError("cv2.imwrite 返回 False")
        return str(path)
    except Exception as exc:
        print(f"[CAD_FINE][WARN] 无法保存可视化图 {path}: {type(exc).__name__}: {exc}", flush=True)
        return None


def _annotate_cad_fine_result_overlay(
    summary_path: str | Path | None,
    hole_id: str,
    final_xy_motion: dict[str, Any] | None,
    final_z_motion: dict[str, Any] | None,
    final_y_trim_motion: dict[str, Any] | None,
) -> str | None:
    """在 260 mm 精定位汇总图上补充最终 XY/Z/+Y 执行结果。"""

    if summary_path is None:
        return None
    source = Path(summary_path)
    if not source.is_file():
        return None
    try:
        view = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if view is None:
            return None
        lines = [f"FINAL RESULT | hole={hole_id}"]
        if final_xy_motion is None:
            lines.append("XY correction: disabled")
        else:
            correction = np.asarray(final_xy_motion.get("xy_correction_mm", [0.0, 0.0]), dtype=np.float64).reshape(-1)
            lines.append(f"XY correction: [{correction[0]:+.3f}, {correction[1]:+.3f}] mm")
        lines.append(f"Z motion: {'executed' if final_z_motion is not None else 'disabled'}")
        lines.append(
            f"base +Y {FINAL_BASE_Y_AFTER_Z_MM:.1f} mm: "
            f"{'executed' if final_y_trim_motion is not None else 'disabled'}"
        )
        panel_height = 26 + 24 * len(lines)
        cv2.rectangle(view, (0, max(0, view.shape[0] - panel_height)),
                      (min(view.shape[1] - 1, 780), view.shape[0] - 1), (0, 0, 0), -1)
        for index, line in enumerate(lines):
            y = view.shape[0] - panel_height + 22 + index * 24
            cv2.putText(view, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(view, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 1, cv2.LINE_AA)
        result_path = source.with_name(f"{source.stem}_result{source.suffix}")
        if not cv2.imwrite(str(result_path), view):
            return None
        return str(result_path)
    except Exception as exc:
        print(f"[CAD_FINE][WARN] 无法补写最终结果图 {source}: {type(exc).__name__}: {exc}", flush=True)
        return None


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
        radius = max(
            float(detection["box"][2] - detection["box"][0]),
            float(detection["box"][3] - detection["box"][1]),
        ) / 2.0
        # 初始选孔只把点击用于确定目标身份；几何中心统一采用YOLO框中心。
        # 椭圆中心不再参与初始点云中心、法向或导航位姿计算。
        yolo_center = np.asarray(detection["center"], dtype=np.float64).reshape(2)
        ring_center = tuple(yolo_center.tolist())
        try:
            point, info = hole_camera_point(
                ring_center, bundle.xyz_map_mm, bundle.intrinsics, radius,
                ray_center_xy=yolo_center, ray_center_is_undistorted=False,
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
        overlays.append((display_detection, None))
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
                          ) -> tuple[list[Observation], np.ndarray]:
    observations: list[Observation] = []
    latest_image: np.ndarray | None = None
    latest_detection: dict[str, Any] | None = None
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
        # 粗定位只使用YOLO框中心和点云环带，测量孔中心及局部法向。
        # 椭圆拟合只用于旧诊断叠加图，不参与粗定位结果，因此不在这里计算。
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
            observations.append(Observation(name, attempt, center, None, plane, bundle.host_timestamp_ns,
                                            None if valid else "plane_quality"))
        except Exception as exc:
            observations.append(Observation(name, attempt, center, None, timestamp_ns=bundle.host_timestamp_ns,
                                            error=f"plane_error:{exc}"))
        if not lock_anchor:
            anchor = detection_center
        latest_detection = detection
    if latest_image is None:
        raise RuntimeError("粗定位期间未获得相机帧")
    cv2.imwrite(str(run_dir / f"{name}_overlay.png"), _overlay(latest_image, latest_detection, None, name))
    return observations, latest_image


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


def _cad_hole_projected_radius_px(
    hole: dict[str, Any], T_base_camera: np.ndarray, intrinsics: Any,
) -> float | None:
    """按当前 260 mm 相机位姿估计 CAD 孔圆在 RGB 中的显示半径。"""

    try:
        point_base = np.asarray(hole["point_base_mm"], dtype=np.float64).reshape(3)
        T_camera_base = invert_transform(T_base_camera)
        point_camera = T_camera_base[:3, :3] @ point_base + T_camera_base[:3, 3]
        depth = float(point_camera[2])
        diameter_mm = float(hole.get("diameter_mm", 0.0))
        focal = 0.5 * (float(intrinsics.fx) + float(intrinsics.fy))
        if depth <= 1e-6 or diameter_mm <= 0.0 or focal <= 0.0:
            return None
        radius_px = 0.5 * diameter_mm * focal / depth
        return float(radius_px) if np.isfinite(radius_px) else None
    except (AttributeError, KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return None


@dataclass(frozen=True)
class CadMotionInput:
    """已通过 Stage 0 质量门、可供运动规划消费的 CAD 输入。"""

    report_path: Path
    cad_model_path: Path
    cad_model: Any
    T_base_cad: np.ndarray
    payload: dict[str, Any]


def _validate_rigid_transform_for_motion(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} 含有非有限值")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} 不是合法齐次变换")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ValueError(f"{name} 的旋转矩阵不正交")
    if float(np.linalg.det(rotation)) <= 0.0:
        raise ValueError(f"{name} 的旋转矩阵行列式必须为正")
    return matrix.copy()


def _read_cad_registration_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"CAD 配准报告不存在: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"CAD 配准报告 JSON 无法解析: {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
        raise ValueError(f"CAD 配准报告缺少 result 对象: {path}")
    return payload


def _load_cad_model_for_fresh_motion(
    cad_model_path: Path | None,
    previous_report_path: Path | None,
) -> tuple[Path, Any, Path | None]:
    """为每次实时配准加载 CAD 几何；不读取旧报告中的位姿。"""

    from aubo_workbench.cad_model import load_cad_model_json

    candidates: list[tuple[Path, Path | None]] = []
    if cad_model_path is not None:
        candidates.append((Path(cad_model_path), previous_report_path))
    candidates.append((DEFAULT_CAD_MODEL_JSON, previous_report_path))
    if previous_report_path is not None:
        candidates.append((Path(previous_report_path), previous_report_path))
    candidates.extend(
        (path, path)
        for path in sorted(
            CAD_REGISTRATION_RUNS_DIR.glob("*/cad_registration_report.json"),
            key=lambda item: item.parent.name,
            reverse=True,
        )
    )

    checked: set[Path] = set()
    for source, report_source in candidates:
        source = Path(source)
        if source in checked:
            continue
        checked.add(source)
        model_source = source
        if source.name == "cad_registration_report.json":
            try:
                payload = _read_cad_registration_payload(source)
                inputs = payload.get("inputs", {})
                cad_json = inputs.get("cad_json") if isinstance(inputs, dict) else None
                model_source = Path(str(cad_json)) if cad_json else DEFAULT_CAD_MODEL_JSON
            except Exception:
                continue
        if not model_source.is_file():
            continue
        try:
            model = load_cad_model_json(model_source)
        except Exception:
            continue
        if int(model.hole_count) != 11:
            raise RuntimeError(
                f"当前 CAD 运动路径要求11个孔，模型实际为{model.hole_count}个：{model_source}"
            )
        return model_source, model, report_source
    raise FileNotFoundError(
        f"没有找到可用于实时 CAD 配准的 11 孔 CAD JSON：{cad_model_path or DEFAULT_CAD_MODEL_JSON}"
    )


def _cad_registration_intrinsics_from_rgb(rgb_intrinsics: Any) -> Any:
    """把相机模块的实时内参转换为 CAD 配准模块使用的内参对象。"""

    from aubo_workbench.cad_registration import CameraIntrinsics as CadCameraIntrinsics

    distortion = np.asarray(
        getattr(rgb_intrinsics, "distortion", getattr(rgb_intrinsics, "dist_coeffs", ())),
        dtype=np.float64,
    ).reshape(-1)
    if distortion.size == 0:
        distortion = np.zeros(5, dtype=np.float64)
    return CadCameraIntrinsics(
        width=int(rgb_intrinsics.width),
        height=int(rgb_intrinsics.height),
        fx=float(rgb_intrinsics.fx),
        fy=float(rgb_intrinsics.fy),
        cx=float(rgb_intrinsics.cx),
        cy=float(rgb_intrinsics.cy),
        dist_coeffs=distortion,
    )


def _build_fresh_cad_detections(
    image: np.ndarray,
    yolo_model: Any,
    confidence: float,
    rgb_intrinsics: Any,
    cad_intrinsics: Any,
) -> list[Any]:
    """用当前主流程 YOLO 结果构造 CAD PnP 所需的检测对象。"""

    from aubo_workbench.cad_registration import CadDetection, undistort_pixels as cad_undistort_pixels

    detections: list[Any] = []
    for detection_id, item in enumerate(detect(yolo_model, image, confidence)):
        box = np.asarray(item["box"], dtype=np.float64).reshape(4)
        center_distorted = np.asarray(item["center"], dtype=np.float64).reshape(2)
        ellipse = None
        try:
            ellipse = fit_hole_ellipse(image, item, rgb_intrinsics)
        except Exception:
            ellipse = None
        center = (
            np.asarray(ellipse["center_px"], dtype=np.float64).reshape(2)
            if isinstance(ellipse, dict) and ellipse.get("center_px") is not None
            else cad_undistort_pixels(center_distorted.reshape(1, 2), cad_intrinsics)[0]
        )
        detections.append(
            CadDetection(
                detection_id=detection_id,
                box_xyxy=box,
                center_px_distorted=center_distorted,
                center_px=center,
                confidence=float(item.get("confidence", 0.0)),
                class_id=int(item.get("class_id", 0)),
                ellipse=ellipse,
                source="cad_motion_fresh_yolo",
            )
        )
    return detections


def _register_fresh_cad_at_motion_start(
    *,
    model_source: Path,
    cad_model: Any,
    yolo_model: Any,
    args: Any,
    handeye: Any,
    pose_session: Any,
    output_dir: Path,
    max_cross_frame_p95_mm: float,
) -> CadMotionInput:
    """每次 CAD 运动启动前，用当前位置的 RGB/TCP 自动更新 CAD 位姿。"""

    from aubo_workbench.cad_registration import (
        CadRegistrationConfig,
        draw_cad_overlay,
        draw_detection_boxes,
        register_detections_frames,
        save_registration_report,
    )

    registration_dir = Path(output_dir) / "cad_registration_auto"
    registration_dir.mkdir(parents=True, exist_ok=True)
    rgb_pipeline = None
    images: list[np.ndarray] = []
    detections_by_frame: list[list[Any]] = []
    first_base_camera: np.ndarray | None = None
    rgb_intrinsics: Any | None = None
    cad_intrinsics: Any | None = None
    report_path: Path | None = None
    try:
        rgb_pipeline = init_rgb_handeye_pipeline()
        for _ in range(8):
            warmup = get_rgb_frame_bundle(rgb_pipeline)
            if warmup is not None and warmup.intrinsics is not None:
                break
        else:
            warmup = None
        if warmup is None or warmup.intrinsics is None:
            raise RuntimeError("CAD启动前自动配准没有取得带内参的 RGB 帧")

        rgb_intrinsics = warmup.intrinsics
        cad_intrinsics = _cad_registration_intrinsics_from_rgb(rgb_intrinsics)
        for frame_index in range(5):
            bundle = warmup if frame_index == 0 else get_rgb_frame_bundle(rgb_pipeline)
            if bundle is None or bundle.intrinsics is None:
                raise RuntimeError(f"CAD启动前自动配准第{frame_index + 1}帧 RGB 获取失败")
            current_snapshot, current_tcp = _require_safe_snapshot(pose_session)
            current_base_camera = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
            if first_base_camera is None:
                first_base_camera = current_base_camera.copy()
            else:
                translation_delta = float(
                    np.linalg.norm(current_base_camera[:3, 3] - first_base_camera[:3, 3])
                )
                rotation_delta = _rotation_distance_deg(
                    current_base_camera[:3, :3], first_base_camera[:3, :3]
                )
                if translation_delta > 1.0 or rotation_delta > 0.25:
                    raise RuntimeError(
                        f"CAD自动配准采集期间 TCP 变化过大："
                        f"平移={translation_delta:.3f} mm，旋转={rotation_delta:.3f} deg"
                    )
            image = np.asarray(bundle.color_bgr).copy()
            detections = _build_fresh_cad_detections(
                image, yolo_model, args.confidence, rgb_intrinsics, cad_intrinsics,
            )
            images.append(image)
            detections_by_frame.append(detections)
            cv2.imwrite(str(registration_dir / f"rgb_original_{frame_index:03d}.png"), image)
            print(
                f"[CAD_AUTO] frame={frame_index + 1}/5 detections={len(detections)} "
                f"TCP={np.round(current_tcp[:3, 3], 2).tolist()}",
                flush=True,
            )

        if first_base_camera is None or cad_intrinsics is None:
            raise RuntimeError("CAD自动配准没有有效的 T_base_camera 或 RGB内参")
        registration_config = CadRegistrationConfig(
            frame_count=5,
            min_valid_frames=3,
            cross_frame_center_p95_mm=float(max_cross_frame_p95_mm),
            allow_single_frame_preview=False,
        )
        registration_result = register_detections_frames(
            detections_by_frame,
            cad_model,
            cad_intrinsics,
            prior_T_camera_cad=None,
            T_base_camera=first_base_camera,
            config=registration_config,
            manual_mapping=None,
            sequential_prior=True,
        )
        for index, (image, detections, frame) in enumerate(
            zip(images, detections_by_frame, registration_result.frames)
        ):
            overlay = draw_cad_overlay(
                image,
                cad_model,
                cad_intrinsics,
                frame,
                title=f"CAD fresh registration frame {index}",
            )
            overlay = draw_detection_boxes(overlay, detections, frame, cad_intrinsics)
            cv2.imwrite(str(registration_dir / f"frame_{index:03d}_overlay.png"), overlay)
        report_path = save_registration_report(
            registration_dir,
            registration_result,
            model=cad_model,
            intrinsics=cad_intrinsics,
            inputs={
                "cad_source": str(model_source),
                "cad_json": str(model_source),
                "intrinsics": "live_rgb_device_intrinsics",
                "images": [str(registration_dir / f"rgb_original_{i:03d}.png") for i in range(len(images))],
                "prior_transform": None,
                "base_camera": first_base_camera,
                "handeye": str(args.handeye),
                "live_robot_capture": True,
                "fresh_registration_at_motion_start": True,
                "mapping_mode": "automatic_geometry",
                "yolo_model": str(args.model),
                "stage0_only": True,
            },
            detections_by_frame=detections_by_frame,
        )
        if not registration_result.success or not registration_result.multi_frame_gate_pass:
            reasons = "；".join(registration_result.failure_reasons) or "质量门未通过"
            raise RuntimeError(
                f"CAD启动前自动配准失败，已禁止继续使用旧位姿：{reasons}；报告={report_path}"
            )
        if registration_result.T_base_cad is None:
            raise RuntimeError(f"CAD自动配准通过但没有 T_base_cad：{report_path}")
        payload = _read_cad_registration_payload(report_path)
        print(
            f"[CAD_AUTO] 当前运动已应用最新 CAD 配准：{report_path} "
            f"cross_frame_p95={float(registration_result.cross_frame_center_p95_mm or 0.0):.3f} mm",
            flush=True,
        )
        return CadMotionInput(
            report_path=report_path,
            cad_model_path=Path(model_source),
            cad_model=cad_model,
            T_base_cad=np.asarray(registration_result.T_base_cad, dtype=np.float64),
            payload=payload,
        )
    finally:
        if rgb_pipeline is not None:
            try:
                rgb_pipeline.stop()
            except Exception:
                pass


def _load_cad_motion_input(
    report_path: Path | None,
    cad_model_path: Path | None,
    cfg: CadMotionConfig,
) -> CadMotionInput:
    """读取最后一次通过 Stage 0 门的报告；失败报告绝不进入运动规划。"""

    candidates: list[Path]
    if report_path is not None:
        candidates = [Path(report_path)]
    else:
        candidates = sorted(
            CAD_REGISTRATION_RUNS_DIR.glob("*/cad_registration_report.json"),
            key=lambda item: item.parent.name,
            reverse=True,
        )
    if not candidates:
        raise FileNotFoundError(
            f"没有找到 CAD 配准报告: {CAD_REGISTRATION_RUNS_DIR}；"
            "请先运行 CAD 配准预览并通过多帧质量门"
        )

    selected_payload: dict[str, Any] | None = None
    selected_path: Path | None = None
    rejection_notes: list[str] = []
    for candidate in candidates:
        try:
            payload = _read_cad_registration_payload(candidate)
            result = payload["result"]
            if not bool(result.get("success")):
                rejection_notes.append(f"{candidate}: success=false")
                continue
            if not bool(result.get("multi_frame_gate_pass")):
                rejection_notes.append(f"{candidate}: multi_frame_gate_pass=false")
                continue
            selected_payload = payload
            selected_path = candidate
            break
        except Exception as exc:
            rejection_notes.append(f"{candidate}: {type(exc).__name__}: {exc}")
            if report_path is not None:
                raise
    if selected_payload is None or selected_path is None:
        latest = candidates[0]
        raise RuntimeError(
            "没有找到通过 CAD 多帧质量门的报告。最近报告为 "
            f"{latest}；" + ("；".join(rejection_notes) if rejection_notes else "请重新运行 CAD 预览")
        )

    result = selected_payload["result"]
    valid_indices = result.get("valid_frame_indices")
    if not isinstance(valid_indices, list) or len(valid_indices) < 3:
        raise RuntimeError(
            f"CAD 报告有效帧不足3帧，拒绝运动：{selected_path}，有效帧={valid_indices}"
        )
    cross_frame_p95_mm = result.get("cross_frame_center_p95_mm")
    if cross_frame_p95_mm is None or not math.isfinite(float(cross_frame_p95_mm)):
        raise RuntimeError(f"CAD 报告缺少跨帧中心稳定性指标：{selected_path}")
    if float(cross_frame_p95_mm) > float(cfg.max_registration_center_p95_mm):
        raise RuntimeError(
            f"CAD 跨帧中心 P95={float(cross_frame_p95_mm):.3f} mm，"
            f"> {float(cfg.max_registration_center_p95_mm):.3f} mm，拒绝运动"
        )

    T_base_cad = _validate_rigid_transform_for_motion(result.get("T_base_cad"), "result.T_base_cad")
    inputs = selected_payload.get("inputs", {})
    if not isinstance(inputs, dict):
        inputs = {}
    model_source = Path(cad_model_path) if cad_model_path is not None else Path(
        str(inputs.get("cad_json") or DEFAULT_CAD_MODEL_JSON)
    )
    if not model_source.is_file() and cad_model_path is None and DEFAULT_CAD_MODEL_JSON.is_file():
        model_source = DEFAULT_CAD_MODEL_JSON
    from aubo_workbench.cad_model import load_cad_model_json

    cad_model = load_cad_model_json(model_source)
    if int(cad_model.hole_count) != 11:
        raise RuntimeError(
            f"当前 CAD 运动路径要求11个孔，模型实际为{cad_model.hole_count}个：{model_source}"
        )
    report_holes = result.get("final_holes_base")
    if isinstance(report_holes, list) and len(report_holes) != cad_model.hole_count:
        raise RuntimeError(
            f"CAD 报告孔数量={len(report_holes)} 与模型孔数量={cad_model.hole_count} 不一致"
        )

    print(
        f"[CAD] 使用通过质量门的报告: {selected_path}\n"
        f"[CAD] success=true, valid_frames={len(valid_indices)}, "
        f"cross_frame_center_p95={float(cross_frame_p95_mm):.3f} mm\n"
        f"[CAD] motion_allowed={selected_payload.get('motion_allowed')} "
        "（Stage 0 报告本身只读；真实运动还要经过本流程的手眼质量门）",
        flush=True,
    )
    return CadMotionInput(
        report_path=selected_path,
        cad_model_path=model_source,
        cad_model=cad_model,
        T_base_cad=T_base_cad,
        payload=selected_payload,
    )


def _cad_holes_for_current_camera(
    cad_input: CadMotionInput, T_base_camera: np.ndarray,
) -> list[dict[str, Any]]:
    from aubo_workbench.cad_registration import orient_base_hole_points

    holes = orient_base_hole_points(
        cad_input.cad_model, cad_input.T_base_cad, T_base_camera,
    )
    if len(holes) != int(cad_input.cad_model.hole_count):
        raise RuntimeError("CAD 基坐标孔位数量与 CAD 模型不一致")
    return holes


def _select_cad_holes(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    holes_base: list[dict[str, Any]],
    T_base_camera: np.ndarray,
    intrinsics: Any,
) -> list[str] | None:
    """在当前 RGB 画面中直接点击 CAD 投影孔位，返回有序 CAD 孔号。"""

    projected: dict[str, np.ndarray] = {}
    for hole in holes_base:
        try:
            center = _project_base_point_to_pixel(
                np.asarray(hole["point_base_mm"], dtype=np.float64),
                T_base_camera,
                intrinsics,
            )
        except Exception:
            continue
        if np.isfinite(center).all():
            projected[str(hole["hole_id"])] = center
    if not projected:
        raise RuntimeError("当前相机位姿下没有可见的 CAD 孔投影，无法选择目标孔")

    window = "CAD target hole selection"
    selected: list[str] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        click = np.array([float(x), float(y)], dtype=np.float64)
        hole_id, distance = min(
            ((key, float(np.linalg.norm(point - click))) for key, point in projected.items()),
            key=lambda item: item[1],
        )
        if distance > 55.0:
            return
        if hole_id in selected:
            selected.remove(hole_id)
        else:
            selected.append(hole_id)

    try:
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, on_mouse)
        while True:
            view = image.copy()
            for index, detection in enumerate(detections):
                x1, y1, x2, y2 = [int(round(value)) for value in detection["box"]]
                center = tuple(np.rint(np.asarray(detection["center"], dtype=np.float64)).astype(int))
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 130, 255), 2, cv2.LINE_AA)
                cv2.drawMarker(view, center, (0, 80, 255), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
                cv2.putText(
                    view, f"det-{index}", (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 130, 255), 2, cv2.LINE_AA,
                )
            for hole_id, center_value in projected.items():
                center = tuple(np.rint(center_value).astype(int))
                is_selected = hole_id in selected
                color = (0, 255, 0) if is_selected else (0, 255, 255)
                cv2.circle(view, center, 18, color, 2, cv2.LINE_AA)
                cv2.drawMarker(view, center, color, cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
                label = f"{selected.index(hole_id) + 1}: {hole_id}" if is_selected else hole_id
                cv2.putText(
                    view, label, (center[0] + 12, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA,
                )
            prompt = (
                "CAD TARGET: click CAD hole | Enter/Space=confirm | "
                f"selected={len(selected)} | R=reset Z=undo Q=quit"
            )
            cv2.putText(view, prompt, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            cv2.imshow(window, view)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 32) and selected:
                return list(selected)
            if key in (ord("r"), ord("R")):
                selected.clear()
            elif key in (ord("z"), ord("Z")) and selected:
                selected.pop()
            elif key in (ord("q"), ord("Q"), 27):
                return None
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass


def _cad_group_entry_plan(
    holes_base: list[dict[str, Any]],
    selected_ids: list[str],
    reference_tcp: np.ndarray,
    handeye: Any,
    fixed_rz_rad: float,
    intrinsics: Any,
    depth_height_mm: float,
    view_margin_px: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """为一组 CAD 孔规划共同的深度采集位姿。

    组中心使用选中 CAD 孔在目标相机坐标中的投影包围盒中心，保证整组孔
    尽量平均地分布在视野中央。深度采集位姿的 XY/姿态/孔号都来自 CAD，
    后续深度只用于估计整块顶面的共享高度偏差。
    """

    selected_id_set = {str(item) for item in selected_ids}
    selected = [
        hole for hole in holes_base if str(hole["hole_id"]) in selected_id_set
    ]
    if len(selected) != len(selected_ids):
        selected_found = {str(hole["hole_id"]) for hole in selected}
        missing = [str(item) for item in selected_ids if str(item) not in selected_found]
        raise RuntimeError(f"CAD 分组规划缺少孔位：{missing}")

    reference_normal = _unit(
        np.asarray(selected[0]["normal_toward_camera_base"], dtype=np.float64),
        "CAD group normal",
    )
    aligned_normals: list[np.ndarray] = []
    for hole in selected:
        normal = _unit(
            np.asarray(hole["normal_toward_camera_base"], dtype=np.float64),
            f"CAD hole {hole['hole_id']} group normal",
        )
        aligned_normals.append(normal if float(normal @ reference_normal) >= 0.0 else -normal)
    group_normal = _unit(np.median(np.asarray(aligned_normals), axis=0), "CAD group normal")

    R_tcp, rotation_info = _tcp_rotation_with_fixed_rz_for_camera_axis(
        -group_normal,
        reference_tcp,
        handeye.T_tcp_rgb_camera,
        fixed_rz_rad,
    )
    R_base_camera = R_tcp @ np.asarray(handeye.T_tcp_rgb_camera, dtype=np.float64)[:3, :3]
    points_base = np.asarray([
        np.asarray(hole["point_base_mm"], dtype=np.float64).reshape(3) for hole in selected
    ])
    # 先在目标相机方向下求选中孔投影包围盒中心，再转回基坐标。
    points_camera_origin = (R_base_camera.T @ points_base.T).T
    group_camera_point = np.asarray([
        0.5 * (float(np.min(points_camera_origin[:, 0])) + float(np.max(points_camera_origin[:, 0]))),
        0.5 * (float(np.min(points_camera_origin[:, 1])) + float(np.max(points_camera_origin[:, 1]))),
        float(np.median(points_camera_origin[:, 2])),
    ])
    group_point_base = R_base_camera @ group_camera_point
    target, pose_geometry = _plan_hole_tcp_pose_fixed_rz(
        group_point_base,
        group_normal,
        reference_tcp,
        handeye.T_tcp_rgb_camera,
        fixed_rz_rad=fixed_rz_rad,
        camera_height_mm=float(depth_height_mm),
    )
    T_base_camera_target = camera_transform(target, handeye.T_tcp_rgb_camera)

    width = float(getattr(intrinsics, "width", 0))
    height = float(getattr(intrinsics, "height", 0))
    if width <= 0.0 or height <= 0.0:
        raise RuntimeError("RGB 内参缺少有效图像宽高，无法检查共同340 mm位姿的视野范围")

    projected: dict[str, np.ndarray] = {}
    projected_radius_px: dict[str, float] = {}
    T_camera_base_target = invert_transform(T_base_camera_target)
    for hole in selected:
        hole_id = str(hole["hole_id"])
        point_base = np.asarray(hole["point_base_mm"], dtype=np.float64).reshape(3)
        projected[hole_id] = _project_base_point_to_pixel(
            point_base,
            T_base_camera_target,
            intrinsics,
        )
        point_camera = (
            T_camera_base_target[:3, :3] @ point_base + T_camera_base_target[:3, 3]
        )
        if point_camera[2] <= 1e-6:
            raise RuntimeError(f"CAD孔{hole_id}在共同340 mm位姿下位于相机后方")
        diameter_mm = float(hole.get("diameter_mm", 0.0))
        projected_radius_px[hole_id] = (
            0.5 * diameter_mm * max(float(intrinsics.fx), float(intrinsics.fy))
            / float(point_camera[2])
            if diameter_mm > 0.0 else 0.0
        )
    projected_values = np.asarray(list(projected.values()), dtype=np.float64)
    margin = float(view_margin_px)
    out_of_view = [
        hole_id for hole_id, point in projected.items()
        if not np.isfinite(point).all()
        or float(point[0]) < margin + projected_radius_px[hole_id]
        or float(point[0]) > width - margin - projected_radius_px[hole_id]
        or float(point[1]) < margin + projected_radius_px[hole_id]
        or float(point[1]) > height - margin - projected_radius_px[hole_id]
    ]
    if out_of_view:
        raise RuntimeError(
            "选中 CAD 孔无法在共同 340 mm 位姿下全部保持在视野内："
            + ", ".join(out_of_view)
            + "；请减少数量或选择更紧凑的一组孔"
        )

    bbox_min = np.min(projected_values, axis=0)
    bbox_max = np.max(projected_values, axis=0)
    group_center_px = 0.5 * (bbox_min + bbox_max)
    return target, {
        "hole_order": [str(item) for item in selected_ids],
        "group_point_base_mm": group_point_base,
        "group_normal_toward_camera_base": group_normal,
        "projected_holes_px": {key: value for key, value in projected.items()},
        "group_bbox_px": [bbox_min[0], bbox_min[1], bbox_max[0], bbox_max[1]],
        "group_center_px": group_center_px,
        "view_margin_px": margin,
        "projected_hole_radius_px": projected_radius_px,
        "depth_height_mm": float(depth_height_mm),
        "target_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(target),
        "target_tcp_transform_mm": target,
        "pose_geometry": pose_geometry,
        "rotation_info": rotation_info,
    }


def _assign_detections_to_cad_projection(
    detections: list[dict[str, Any]],
    projected_holes_px: dict[str, np.ndarray],
    max_distance_px: float,
) -> dict[str, dict[str, Any]]:
    """按 CAD 投影最近邻分配检测框，保证一个检测不会被多个 CAD 孔复用。"""

    pairs: list[tuple[float, str, int]] = []
    for hole_id, expected in projected_holes_px.items():
        expected_value = np.asarray(expected, dtype=np.float64).reshape(2)
        for detection_index, detection in enumerate(detections):
            center = np.asarray(detection["center"], dtype=np.float64).reshape(2)
            distance = float(np.linalg.norm(center - expected_value))
            if distance <= float(max_distance_px):
                pairs.append((distance, str(hole_id), int(detection_index)))
    assigned: dict[str, dict[str, Any]] = {}
    used_detection_indices: set[int] = set()
    for distance, hole_id, detection_index in sorted(pairs, key=lambda item: item[0]):
        if hole_id in assigned or detection_index in used_detection_indices:
            continue
        assigned[hole_id] = {
            "detection": detections[detection_index],
            "detection_index": detection_index,
            "distance_px": distance,
        }
        used_detection_indices.add(detection_index)
    return assigned


def _cad_depth_policy(
    selected_hole_count: int,
    configured_min_valid_holes: int,
) -> tuple[str, int]:
    """根据选孔数量决定共同340 mm深度的有效孔质量门。

    单孔时只用该孔的环带深度；多孔时融合选中孔的环带深度。
    选中2/3孔时不能要求4孔，必须要求本组所有孔有效；选中4孔以上
    时保留至少4个有效孔的鲁棒门槛。
    """

    count = int(selected_hole_count)
    if count < 1:
        raise ValueError("CAD深度采集至少需要选择1个孔")
    if count == 1:
        return "single_hole", 1
    if count < 4:
        return "multi_hole_shared", count
    return "multi_hole_shared", min(int(configured_min_valid_holes), count)


def _capture_cad_group_depth(
    pipeline: Any,
    align: Any,
    chain: Any,
    model: Any,
    confidence: float,
    selected_holes: list[dict[str, Any]],
    T_base_camera: np.ndarray,
    cfg: CadMotionConfig,
    run_dir: Path,
) -> dict[str, Any]:
    """在共同 340 mm 位姿一次采集整组孔的顶面深度。

    每个孔只用 YOLO 框外环提取深度平面点；CAD 仍然提供孔中心、孔号和法向。
    单孔模式只使用该孔的环带平面；多孔模式在同一帧内融合所选孔的平面点，
    最终输出一个共享的平面点和高度偏差。
    """

    depth_mode, min_holes = _cad_depth_policy(
        len(selected_holes), cfg.min_depth_valid_holes,
    )
    for _ in range(max(0, int(cfg.settle_discard_frames))):
        get_aligned_frame_bundle(pipeline, align, chain)

    camera_origin_base = np.asarray(T_base_camera[:3, 3], dtype=np.float64)
    camera_axis_base = _unit(np.asarray(T_base_camera[:3, 2], dtype=np.float64), "CAD group camera axis")
    frame_records: list[dict[str, Any]] = []
    valid_frame_points_base: list[np.ndarray] = []
    valid_heights: list[float] = []

    for frame_index in range(max(1, int(cfg.depth_frames))):
        bundle = get_aligned_frame_bundle(pipeline, align, chain)
        if bundle is None or bundle.intrinsics is None:
            frame_records.append({"frame_index": frame_index, "valid": False, "reason": "rgbd_frame_missing"})
            continue
        intrinsics = bundle.intrinsics
        expected_current = {
            str(hole["hole_id"]): _project_base_point_to_pixel(
                np.asarray(hole["point_base_mm"], dtype=np.float64),
                T_base_camera,
                intrinsics,
            )
            for hole in selected_holes
        }
        detections = detect(model, bundle.color_bgr, confidence)
        assignments = _assign_detections_to_cad_projection(
            detections,
            expected_current,
            cfg.yolo_match_tolerance_px,
        )
        plane_points_base: list[np.ndarray] = []
        hole_records: list[dict[str, Any]] = []
        for hole in selected_holes:
            hole_id = str(hole["hole_id"])
            assigned = assignments.get(hole_id)
            if assigned is None:
                hole_records.append({"hole_id": hole_id, "valid": False, "reason": "yolo_missing_or_far"})
                continue
            detection = assigned["detection"]
            box = np.asarray(detection["box"], dtype=np.float64).reshape(4)
            radius = max(float(box[2] - box[0]), float(box[3] - box[1])) / 2.0
            try:
                _point_camera, info = hole_camera_point(
                    tuple(np.asarray(detection["center"], dtype=np.float64).reshape(2).tolist()),
                    bundle.xyz_map_mm,
                    intrinsics,
                    radius,
                    ray_center_xy=np.asarray(detection["center"], dtype=np.float64),
                    ray_center_is_undistorted=False,
                )
                plane_rmse = float(info["plane_rmse_mm"])
                if plane_rmse > float(cfg.max_depth_plane_rmse_mm):
                    raise ValueError(
                        f"plane_rmse={plane_rmse:.3f}mm>{float(cfg.max_depth_plane_rmse_mm):.3f}mm"
                    )
                measured_normal_camera = _unit(
                    np.asarray(info["plane_normal_camera"], dtype=np.float64),
                    f"CAD group depth hole {hole_id} measured normal",
                )
                expected_normal_camera = _unit(
                    T_base_camera[:3, :3].T
                    @ np.asarray(hole["normal_toward_camera_base"], dtype=np.float64),
                    f"CAD group depth hole {hole_id} expected normal",
                )
                normal_direction_dot = float(measured_normal_camera @ expected_normal_camera)
                if normal_direction_dot < 0.70:
                    raise ValueError(
                        f"normal_direction_dot={normal_direction_dot:.3f}<0.700"
                    )
                point_camera = np.asarray(info["plane_point_camera_mm"], dtype=np.float64).reshape(3)
                point_base = T_base_camera[:3, :3] @ point_camera + T_base_camera[:3, 3]
                if not np.isfinite(point_base).all():
                    raise ValueError("plane point contains non-finite value")
                plane_points_base.append(point_base)
                hole_records.append({
                    "hole_id": hole_id,
                    "valid": True,
                    "distance_to_cad_projection_px": float(assigned["distance_px"]),
                    "plane_point_base_mm": point_base,
                    "plane_point_camera_mm": point_camera,
                    "plane_normal_camera": np.asarray(info["plane_normal_camera"], dtype=np.float64),
                    "normal_direction_dot": normal_direction_dot,
                    "plane_rmse_mm": plane_rmse,
                    "ring_points": int(info["ring_points"]),
                })
            except Exception as exc:
                hole_records.append({
                    "hole_id": hole_id,
                    "valid": False,
                    "distance_to_cad_projection_px": float(assigned["distance_px"]),
                    "reason": f"depth_plane_failed:{type(exc).__name__}:{exc}",
                })

        view = bundle.color_bgr.copy()
        for hole_id, expected in expected_current.items():
            point = tuple(np.rint(np.asarray(expected, dtype=np.float64)).astype(int))
            assigned = assignments.get(hole_id)
            color = (0, 255, 0) if any(
                item.get("hole_id") == hole_id and item.get("valid") for item in hole_records
            ) else (0, 165, 255)
            cv2.circle(view, point, 18, color, 2, cv2.LINE_AA)
            cv2.drawMarker(view, point, color, cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
            cv2.putText(
                view, f"CAD-{hole_id}", (point[0] + 10, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA,
            )
            if assigned is not None:
                det_center = tuple(np.rint(np.asarray(assigned["detection"]["center"], dtype=np.float64)).astype(int))
                cv2.drawMarker(view, det_center, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)
        valid_holes = [item for item in hole_records if item.get("valid")]
        if len(valid_holes) >= min_holes:
            frame_plane_point = np.median(
                np.asarray([item["plane_point_base_mm"] for item in valid_holes], dtype=np.float64),
                axis=0,
            )
            frame_height = float(camera_axis_base @ (frame_plane_point - camera_origin_base))
            frame_rmse = float(np.median([float(item["plane_rmse_mm"]) for item in valid_holes]))
            valid_frame_points_base.append(frame_plane_point)
            valid_heights.append(frame_height)
            frame_valid = True
        else:
            frame_plane_point = None
            frame_height = None
            frame_rmse = None
            frame_valid = False
        frame_record = {
            "frame_index": frame_index,
            "valid": frame_valid,
            "valid_hole_count": len(valid_holes),
            "required_hole_count": min_holes,
            "height_mm": frame_height,
            "plane_rmse_mm": frame_rmse,
            "holes": hole_records,
        }
        frame_records.append(frame_record)
        cv2.putText(
            view,
            f"CAD group depth 340mm frame={frame_index} valid_holes={len(valid_holes)}/{len(selected_holes)}",
            (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.imwrite(str(run_dir / f"cad_depth_340_frame_{frame_index:02d}.png"), view)

    valid_frame_count = len(valid_heights)
    if valid_heights:
        median_height = float(np.median(np.asarray(valid_heights, dtype=np.float64)))
        height_scatter = float(np.percentile(
            np.abs(np.asarray(valid_heights, dtype=np.float64) - median_height), 95.0,
        ))
        median_plane_point = np.median(np.asarray(valid_frame_points_base), axis=0)
    else:
        median_height = None
        height_scatter = None
        median_plane_point = None
    failure_reasons: list[str] = []
    if valid_frame_count < int(cfg.min_depth_valid_frames):
        failure_reasons.append(
            f"有效深度帧不足：{valid_frame_count}/{int(cfg.min_depth_valid_frames)}"
        )
    if height_scatter is not None and height_scatter > float(cfg.max_depth_height_scatter_p95_mm):
        failure_reasons.append(
            f"共同高度跨帧P95过大：{height_scatter:.3f}mm>{float(cfg.max_depth_height_scatter_p95_mm):.3f}mm"
        )
    if median_plane_point is None or median_height is None:
        failure_reasons.append("没有得到有效的共同顶面深度平面")
    success = not failure_reasons
    return {
        "success": success,
        "depth_mode": depth_mode,
        "selected_hole_count": len(selected_holes),
        "depth_height_mm": float(cfg.depth_height_mm),
        "valid_frame_count": valid_frame_count,
        "total_frame_count": int(cfg.depth_frames),
        "min_valid_frame_count": int(cfg.min_depth_valid_frames),
        "min_valid_hole_count": min_holes,
        "height_median_mm": median_height,
        "height_scatter_p95_mm": height_scatter,
        "height_offset_mm": None if median_height is None else median_height - float(cfg.depth_height_mm),
        "plane_point_base_mm": median_plane_point,
        "frame_heights_mm": valid_heights,
        "failure_reasons": failure_reasons,
        "frames": frame_records,
        "camera_axis_base": camera_axis_base,
        "depth_source": (
            "RGB-D aligned single-hole outer-ring plane; depth supplies this hole height offset"
            if depth_mode == "single_hole" else
            "RGB-D aligned multi-hole outer-ring planes; depth only supplies shared height offset"
        ),
    }


def _capture_cad_fine_yolo(
    pipeline: Any,
    model: Any,
    confidence: float,
    expected_center_px_distorted: np.ndarray,
    fallback_intrinsics: Any,
    cfg: CadMotionConfig,
    hole_id: str,
    *,
    run_dir: Path | None = None,
    expected_radius_px: float | None = None,
) -> dict[str, Any]:
    """在已到260 mm的位姿采RGB；YOLO只用于最终XY修正。"""

    for _ in range(max(0, int(cfg.settle_discard_frames))):
        get_rgb_frame_bundle(pipeline)

    expected = np.asarray(expected_center_px_distorted, dtype=np.float64).reshape(2)
    valid: list[dict[str, Any]] = []
    last_intrinsics = fallback_intrinsics
    fine_visualization_dir = None
    if run_dir is not None:
        fine_visualization_dir = Path(run_dir) / "fine_visualizations" / str(hole_id)
        fine_visualization_dir.mkdir(parents=True, exist_ok=True)
    last_valid_image: np.ndarray | None = None
    last_valid_detection: dict[str, Any] | None = None
    for frame_index in range(max(1, int(cfg.fine_frames))):
        bundle = get_rgb_frame_bundle(pipeline)
        if bundle is None:
            continue
        if bundle.intrinsics is not None:
            last_intrinsics = bundle.intrinsics
        detections = detect(model, bundle.color_bgr, confidence)
        if not detections:
            continue
        distances = [
            float(np.linalg.norm(np.asarray(item["center"], dtype=np.float64).reshape(2) - expected))
            for item in detections
        ]
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])
        if best_distance > float(cfg.yolo_match_tolerance_px):
            continue
        selected_detection = detections[best_index]
        last_valid_image = np.asarray(bundle.color_bgr).copy()
        last_valid_detection = selected_detection
        visualization_path = None
        if fine_visualization_dir is not None:
            visualization_path = _save_cad_fine_overlay(
                fine_visualization_dir / f"frame_{frame_index:02d}.png",
                bundle.color_bgr,
                hole_id=str(hole_id),
                frame_index=frame_index,
                expected_center_px=expected,
                detection=selected_detection,
                all_detections=detections,
                expected_radius_px=expected_radius_px,
                distance_px=best_distance,
                summary=False,
            )
        valid.append({
            "frame_index": frame_index,
            "detection": selected_detection,
            "distance_to_cad_projection_px": best_distance,
            "all_detection_count": len(detections),
            "visualization_path": visualization_path,
        })

    if len(valid) < int(cfg.min_fine_valid_frames):
        raise RuntimeError(
            f"CAD 孔{hole_id} 260 mm精定位YOLO有效帧不足："
            f"{len(valid)}/{int(cfg.min_fine_valid_frames)}"
        )
    centers = np.asarray([
        np.asarray(item["detection"]["center"], dtype=np.float64).reshape(2)
        for item in valid
    ])
    median_center = np.median(centers, axis=0)
    scatter = np.linalg.norm(centers - median_center[None, :], axis=1)
    scatter_p95 = float(np.percentile(scatter, 95.0))
    if scatter_p95 > float(cfg.max_fine_center_scatter_p95_px):
        raise RuntimeError(
            f"CAD 孔{hole_id} 260 mm精定位YOLO中心不稳定："
            f"P95={scatter_p95:.3f}px > {float(cfg.max_fine_center_scatter_p95_px):.3f}px"
        )
    center_undistorted = undistort_pixels(
        last_intrinsics, median_center.reshape(1, 2), pixel_output=True,
    )[0]
    mean_distance = float(np.mean([
        item["distance_to_cad_projection_px"] for item in valid
    ]))
    visualization: dict[str, Any] = {
        "directory": None if fine_visualization_dir is None else str(fine_visualization_dir),
        "frame_images": [
            str(item["visualization_path"])
            for item in valid
            if item.get("visualization_path")
        ],
        "summary_image": None,
        "result_image": None,
        "legend": {
            "cad_expected": "cyan circle/cross",
            "yolo_box": "orange box/magenta center",
            "yolo_median": "green cross/circle",
            "error_vector": "red arrow from CAD expected to YOLO center",
        },
    }
    if fine_visualization_dir is not None and valid and last_valid_image is not None and last_valid_detection is not None:
        summary_image = _save_cad_fine_overlay(
            fine_visualization_dir / "summary.png",
            last_valid_image,
            hole_id=str(hole_id),
            frame_index=int(valid[-1]["frame_index"]),
            expected_center_px=expected,
            detection=last_valid_detection,
            all_detections=None,
            median_center_px=median_center,
            expected_radius_px=expected_radius_px,
            distance_px=float(np.linalg.norm(median_center - expected)),
            valid_frames=len(valid),
            total_frames=int(cfg.fine_frames),
            scatter_p95_px=scatter_p95,
            mean_distance_px=mean_distance,
            summary=True,
        )
        visualization["summary_image"] = summary_image
    return {
        "hole_id": str(hole_id),
        "cad_expected_center_px_distorted": expected,
        "center_px_distorted": median_center,
        "center_px": center_undistorted,
        "valid_frames": len(valid),
        "total_frames": int(cfg.fine_frames),
        "center_scatter_p95_px": scatter_p95,
        "mean_distance_to_cad_projection_px": mean_distance,
        "median_error_vector_px": median_center - expected,
        "expected_radius_px": expected_radius_px,
        "observations": valid,
        "intrinsics": last_intrinsics,
        "visualization": visualization,
    }


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
        valid, median, _, selected_p95 = filtered_candidate
        selection_rule = "mad_filtered_lower_p95"
    else:
        valid, median, _, selected_p95 = raw_candidate
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
                coarse_observations, _ = _capture_coarse_burst(
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

        # 先按最终点模式施加基坐标偏移，再用偏移后的点规划最终移动。
        final_target_mode = str(getattr(args, "final_target_mode", DEFAULT_FINAL_TARGET_MODE))
        final_x_offset_mm, final_z_offset_mm = final_point_offsets_for_mode(final_target_mode)
        final_target_point_base = apply_final_point_base_offsets(
            final_point_base, final_x_offset_mm, final_z_offset_mm,
        )
        fine_tcp = current_tcp.copy()
        final_xy_motion: dict[str, Any] | None = None
        final_z_motion: dict[str, Any] | None = None
        final_y_trim_motion: dict[str, Any] | None = None
        hole_pose, hole_pose_geometry = _plan_hole_tcp_pose_fixed_rz(
            final_target_point_base, coarse_normal_base, current_tcp,
            handeye.T_tcp_rgb_camera, fixed_rz_rad=fixed_rz_rad,
        )
        if args.move_final_xy:
            fixed_offset = (
                None if args.tcp_xy_offset_mm is None
                else (float(args.tcp_xy_offset_mm[0]), float(args.tcp_xy_offset_mm[1]))
            )
            xy_target, tcp_before = plan_final_tcp_xy(
                current_tcp, final_target_point_base, fixed_offset,
            )
            correction_xy = xy_target[:2, 3] - final_target_point_base[:2]
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
                    f"保持孔{hole_id}精拍Z与姿态；{compensation}；"
                    f"模式={final_target_mode}；最终点基坐标"
                    f"X{final_x_offset_mm:+.1f} mm、Z{final_z_offset_mm:+.1f} mm",
                    require_confirmation=False,
                    motion_profile="precision",
                )
            final_xy_motion = {
                "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(xy_target),
                "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "tcp_position_before_mm": tcp_before,
                "hole_center_base_mm": final_point_base,
                "target_point_base_mm": final_target_point_base,
                "final_point_mode": final_target_mode,
                "final_point_offset_base_mm": [final_x_offset_mm, 0.0, final_z_offset_mm],
                "compensation_mode": (
                    "charuco_affine_model" if fixed_offset is None else "fixed_offset_override"
                ),
                "xy_correction_mm": correction_xy,
                "charuco_model_source": str(CHARUCO_XY_MODEL_SOURCE) if fixed_offset is None else None,
                "charuco_model_matrix_2x2": CHARUCO_XY_MODEL_MATRIX if fixed_offset is None else None,
                "charuco_model_bias_mm": CHARUCO_XY_MODEL_BIAS_MM if fixed_offset is None else None,
                "tcp_xy_offset_mm": None if fixed_offset is None else list(fixed_offset),
            }
            z_target = plan_final_tcp_base_z(current_tcp, final_target_point_base)
            z_delta_mm = float(z_target[2, 3] - current_tcp[2, 3])
            with timing.measure(
                f"hole_{hole_id:02d}/final_motion_z",
                hole_id=hole_id,
                processing_order=order,
            ):
                current_tcp = _confirm_and_move_line(
                    f"孔{hole_id}移动到最终Z",
                    current_tcp, z_target, args, motion_session, pose_session,
                    f"模式={final_target_mode}；最终点基坐标Z{final_z_offset_mm:+.1f} mm；"
                    f"目标TCP基坐标Z={z_target[2, 3]:.3f} mm",
                    require_confirmation=False,
                    motion_profile="precision",
                )
            final_z_motion = {
                "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(z_target),
                "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "target_base_z_mm": float(z_target[2, 3]),
                "target_point_base_mm": final_target_point_base,
                "final_point_mode": final_target_mode,
                "final_point_offset_base_mm": [final_x_offset_mm, 0.0, final_z_offset_mm],
                "delta_base_z_mm": z_delta_mm,
                "motion_frame": "base_z_only",
            }
            y_trim_target = plan_final_tcp_base_y_trim(current_tcp)
            with timing.measure(
                f"hole_{hole_id:02d}/final_motion_y_plus_0_3mm",
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
            "target_point_base_mm": final_target_point_base,
            "hole_center_base_naive_mm": naive_final_point_base,
            "final_point_mode": final_target_mode,
            "final_point_offset_base_mm": [final_x_offset_mm, 0.0, final_z_offset_mm],
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
            f"measured_final_point={np.round(np.asarray(final_point_base), 3).tolist()} "
            f"target_final_point={np.round(np.asarray(final_target_point_base), 3).tolist()} "
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


def _request_next_cad_hole_confirmation(current_hole_id: str, next_hole_id: str) -> str:
    print(
        f"[NEXT_CAD_HOLE_CONFIRM_REQUIRED] 当前孔={current_hole_id}，"
        f"下一个目标孔={next_hole_id}",
        flush=True,
    )
    return input(
        f"孔{current_hole_id}已完成；输入 m 开始 CAD 孔{next_hole_id}，其他任意键停止："
    ).strip().lower()


def _move_to_cad_fine_pose(
    hole_id: str,
    order: int,
    current_tcp: np.ndarray,
    target: np.ndarray,
    args: Any,
    motion_session: Any,
    pose_session: Any,
    *,
    target_height_mm: float = 260.0,
    target_stage: str = "精定位",
) -> np.ndarray:
    """沿安全高度移动到 CAD 规划的目标相机高度。"""

    safe_z = max(float(current_tcp[2, 3]), float(target[2, 3])) + THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM
    actual = np.asarray(current_tcp, dtype=np.float64).copy()
    lift = actual.copy()
    lift[2, 3] = safe_z
    if abs(float(lift[2, 3] - actual[2, 3])) > 0.2:
        actual = _confirm_and_move_line(
            f"CAD孔{hole_id}进入安全高度",
            actual,
            lift,
            args,
            motion_session,
            pose_session,
            "仅修改基坐标Z；不经过工件低位区域",
            require_confirmation=False,
            motion_profile="transit",
        )
    high_target = np.asarray(target, dtype=np.float64).copy()
    high_target[2, 3] = safe_z
    actual = _confirm_and_move_line(
        f"移动到CAD孔{hole_id}上方安全位姿",
        actual,
        high_target,
        args,
        motion_session,
        pose_session,
        "安全高度横移并调整CAD法向姿态；不下降",
        require_confirmation=False,
        motion_profile="transit",
    )
    return _confirm_and_move_line(
        f"CAD孔{hole_id}下降到{float(target_height_mm):.0f} mm{target_stage}位",
        actual,
        target,
        args,
        motion_session,
        pose_session,
        f"CAD孔中心和+Z法向已冻结；仅下降到指定RGB相机高度{float(target_height_mm):.0f} mm",
        require_confirmation=False,
        motion_profile="approach",
    )


def _cad_hole_entry_plan(
    hole: dict[str, Any],
    reference_tcp: np.ndarray,
    handeye: Any,
    fixed_rz_rad: float,
    fine_height_mm: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    point_base = np.asarray(hole["point_base_mm"], dtype=np.float64).reshape(3)
    normal_toward_camera = _unit(
        np.asarray(hole["normal_toward_camera_base"], dtype=np.float64),
        f"CAD孔{hole['hole_id']} toward-camera normal",
    )
    target, geometry = _plan_hole_tcp_pose_fixed_rz(
        point_base,
        normal_toward_camera,
        reference_tcp,
        handeye.T_tcp_rgb_camera,
        fixed_rz_rad=fixed_rz_rad,
        camera_height_mm=float(fine_height_mm),
    )
    achieved_height = camera_height_to_plane_mm(
        target, handeye.T_tcp_rgb_camera, point_base,
    )
    if abs(float(achieved_height) - float(fine_height_mm)) > 0.5:
        raise RuntimeError(
            f"CAD孔{hole['hole_id']}规划高度校验失败："
            f"{float(achieved_height):.3f} mm != {float(fine_height_mm):.3f} mm"
        )
    return target, {
        "hole_id": str(hole["hole_id"]),
        "cad_center_base_mm": point_base,
        "cad_normal_base": np.asarray(hole["normal_base"], dtype=np.float64),
        "normal_toward_camera_base": normal_toward_camera,
        "diameter_mm": float(hole["diameter_mm"]),
        "top_z_mm": float(hole["top_z_mm"]),
        "fine_height_mm": float(fine_height_mm),
        "achieved_camera_height_mm": float(achieved_height),
        "target_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(target),
        "pose_geometry": geometry,
    }


def _apply_shared_cad_depth_height_offset(
    nominal_target: np.ndarray,
    T_tcp_camera: np.ndarray,
    depth_height_offset_mm: float,
) -> tuple[np.ndarray, float, float]:
    """把共同340 mm深度测得的高度偏差应用到一个CAD 260 mm目标。

    深度采集只返回一个共享的相机光轴高度偏差；CAD目标的XY、孔号和法向
    不在这里重新估计。由于机器人执行的是基坐标Z移动，换算时使用当前目标
    姿态的相机光轴在基坐标Z上的分量，并返回该目标实际使用的基坐标Z修正量。
    """

    nominal = np.asarray(nominal_target, dtype=np.float64).reshape(4, 4)
    T_base_camera = camera_transform(nominal, T_tcp_camera)
    camera_axis_z_base = float(T_base_camera[2, 2])
    if abs(camera_axis_z_base) < 0.15:
        raise RuntimeError(
            "CAD 260 mm目标的相机光轴过于接近基坐标水平面，不能应用共享高度偏差"
        )
    base_z_correction_mm = float(depth_height_offset_mm) / camera_axis_z_base
    corrected = nominal.copy()
    corrected[2, 3] += base_z_correction_mm
    return corrected, base_z_correction_mm, camera_axis_z_base


def run_cad_motion_workflow(args: Any, handeye: Any, yolo_model: Any) -> int:
    """CAD孔位运动路径：CAD定姿态/定Z，RGB YOLO只做最终XY修正。"""

    cfg = CadMotionConfig(
        fine_height_mm=float(args.cad_fine_height_mm),
        depth_height_mm=float(args.cad_depth_height_mm),
        depth_frames=int(args.cad_depth_frames),
        min_depth_valid_frames=int(args.cad_depth_min_valid),
        min_depth_valid_holes=int(args.cad_depth_min_holes),
        max_depth_plane_rmse_mm=float(args.cad_depth_plane_rmse_mm),
        max_depth_height_scatter_p95_mm=float(args.cad_depth_height_p95_mm),
        group_view_margin_px=float(args.cad_group_view_margin_px),
        fine_frames=int(args.cad_fine_frames),
        min_fine_valid_frames=int(args.cad_fine_min_valid),
        max_fine_center_scatter_p95_px=float(args.cad_fine_center_p95_px),
        yolo_match_tolerance_px=float(args.cad_yolo_match_tolerance_px),
        settle_discard_frames=int(args.cad_settle_discard_frames),
    )
    model_source, cad_model, previous_registration_report = _load_cad_model_for_fresh_motion(
        args.cad_model_json,
        args.cad_registration_report,
    )
    run_dir = CAD_MOTION_RUNS_DIR / datetime.now().strftime("cad-motion-%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": "cad_motion_group_depth_v2",
        "status": "running",
        "mode": "cad_motion_group_depth_340mm_to_cad_260mm",
        "run_dir": str(run_dir),
        "cad_registration_report": None,
        "cad_registration_policy": "fresh_live_auto_each_run",
        "cad_registration_previous_report": (
            None if previous_registration_report is None else str(previous_registration_report)
        ),
        "cad_model_json": str(model_source),
        "T_base_cad": None,
        "motion_requested": bool(args.execute),
        "motion_executed": False,
        "handeye": {
            "path": str(args.handeye),
            "validated_for_motion": bool(handeye.validated_for_motion),
            "experimental_motion_override": bool(
                args.execute and not handeye.validated_for_motion
            ),
        },
        "configuration": {
            "fine_height_mm": float(cfg.fine_height_mm),
            "depth_height_mm": float(cfg.depth_height_mm),
            "depth_frames": int(cfg.depth_frames),
            "min_depth_valid_frames": int(cfg.min_depth_valid_frames),
            "min_depth_valid_holes": int(cfg.min_depth_valid_holes),
            "max_depth_plane_rmse_mm": float(cfg.max_depth_plane_rmse_mm),
            "max_depth_height_scatter_p95_mm": float(cfg.max_depth_height_scatter_p95_mm),
            "group_view_margin_px": float(cfg.group_view_margin_px),
            "fine_frames": int(cfg.fine_frames),
            "min_fine_valid_frames": int(cfg.min_fine_valid_frames),
            "max_fine_center_scatter_p95_px": float(cfg.max_fine_center_scatter_p95_px),
            "yolo_match_tolerance_px": float(cfg.yolo_match_tolerance_px),
            "settle_discard_frames": int(cfg.settle_discard_frames),
            "final_target_mode": str(getattr(args, "final_target_mode", DEFAULT_FINAL_TARGET_MODE)),
            "move_final_xy": bool(args.move_final_xy),
            "tcp_xy_offset_mm": None if args.tcp_xy_offset_mm is None else [
                float(value) for value in args.tcp_xy_offset_mm
            ],
            "base_y_trim_mm": float(FINAL_BASE_Y_AFTER_Z_MM),
        },
    }
    timing = TimingRecorder()
    timing.attach_report(report)
    pipeline = None
    align = None
    chain = None
    pose_session = None
    motion_session = None
    selected_ids: list[str] = []
    experimental_handeye_motion = bool(args.execute and not handeye.validated_for_motion)
    try:
        from aubo_workbench.motion_control import AuboMotionSession, load_home_point
        from aubo_workbench.robot import AuboPoseSession

        if experimental_handeye_motion and not getattr(args, "cad_allow_experimental_handeye", False):
            raise RuntimeError(
                "CAD真实运动被安全门拒绝：当前手眼结果未通过生产验证。"
                "如确实进行现场实验，必须在入口中明确打开 CAD 实验手眼运动；"
                "生产运行仍必须使用 validated_for_motion=true 的手眼结果。"
            )
        if experimental_handeye_motion:
            print(
                "[EXPERIMENTAL] CAD运动将使用未通过生产验证的手眼结果；"
                "本次不得作为生产精度结论。",
                flush=True,
            )

        with timing.measure("robot/connect_pose_session"):
            pose_session = AuboPoseSession()
            pose_session.connect()
            initial_snapshot, initial_tcp = _require_safe_snapshot(pose_session)
        report["robot_initial_tcp_pose_m_rad"] = initial_snapshot["pose_values_sdk_m_rad"]
        report["robot_initial_snapshot"] = initial_snapshot

        # 工件可能在上一次运行后被移动；旧 T_base_cad 只作为历史记录，绝不直接用于本次运动。
        report["cad_registration_report"] = str(
            run_dir / "cad_registration_auto" / "cad_registration_report.json"
        )
        with timing.measure("cad_registration/fresh_live_auto"):
            cad_input = _register_fresh_cad_at_motion_start(
                model_source=model_source,
                cad_model=cad_model,
                yolo_model=yolo_model,
                args=args,
                handeye=handeye,
                pose_session=pose_session,
                output_dir=run_dir,
                max_cross_frame_p95_mm=cfg.max_registration_center_p95_mm,
            )
        report["cad_registration_report"] = str(cad_input.report_path)
        report["cad_model_json"] = str(cad_input.cad_model_path)
        report["T_base_cad"] = cad_input.T_base_cad

        registration_inputs = cad_input.payload.get("inputs", {})
        registration_handeye = (
            str(registration_inputs.get("handeye", "")).strip()
            if isinstance(registration_inputs, dict) else ""
        )
        if registration_handeye and Path(registration_handeye).resolve() != Path(args.handeye).resolve():
            message = (
                f"CAD报告使用的手眼文件与当前文件不同：报告={registration_handeye}，"
                f"当前={args.handeye}"
            )
            if args.execute:
                raise RuntimeError(message + "；真实运动拒绝")
            print("[CAD][WARN] " + message, flush=True)

        current_tcp = np.asarray(initial_tcp, dtype=np.float64).copy()
        if args.execute:
            with timing.measure("robot/connect_motion_session"):
                motion_session = AuboMotionSession()
                motion_session.connect(
                    ROBOT_CFG.ip, ROBOT_CFG.rpc_port, ROBOT_CFG.user,
                    ROBOT_CFG.password, ROBOT_CFG.request_timeout_ms,
                )
            home = load_home_point()
            if home is None:
                raise RuntimeError("未找到 aubo_home_point.json；请先在机器人控制界面设置原点")
            with timing.measure("robot/move_home_before_cad_group_selection"):
                current_tcp = _confirm_and_move_home(home, args, motion_session, pose_session)
            report["home_point"] = home.to_dict()
            report["robot_selection_tcp_pose_m_rad"] = transform_to_sdk_pose_m_rad(current_tcp)
            print("[CAD] 已回到原点，准备选择一组 CAD 目标孔。", flush=True)

        # 真实运动必须在原点选孔；预览模式保持不运动，使用启动时当前TCP做规划。
        selection_tcp = np.asarray(current_tcp, dtype=np.float64).copy()
        T_base_camera_selection = camera_transform(selection_tcp, handeye.T_tcp_rgb_camera)
        holes_selection = _cad_holes_for_current_camera(cad_input, T_base_camera_selection)
        holes_by_id = {str(item["hole_id"]): item for item in holes_selection}
        fixed_rz_rad = _matrix_to_rpy_zyx(selection_tcp[:3, :3])[2]
        report["selection_T_base_camera"] = T_base_camera_selection
        report["fixed_rz_rad"] = float(fixed_rz_rad)
        report["cad_holes_base_selection"] = holes_selection

        if args.execute:
            with timing.measure("camera/start_rgbd_group_depth_pipeline"):
                pipeline, align, chain = init_pipeline()
        else:
            with timing.measure("camera/start_rgb_only_preview_pipeline"):
                pipeline = init_rgb_handeye_pipeline()
        identity = get_device_identity(pipeline)
        expected_serial = str(handeye.payload.get("camera_serial", "")).strip()
        actual_serial = str(identity.get("serial_number", "")).strip()
        if expected_serial and actual_serial and expected_serial != actual_serial:
            raise RuntimeError(f"相机序列号不匹配：手眼={expected_serial}，当前={actual_serial}")
        report["camera"] = identity

        bundle = None
        for _ in range(20):
            if args.execute:
                bundle = get_aligned_frame_bundle(pipeline, align, chain)
            else:
                bundle = get_rgb_frame_bundle(pipeline)
            if bundle is not None and bundle.intrinsics is not None:
                break
        if bundle is None or bundle.intrinsics is None:
            raise RuntimeError("CAD 选择画面未返回带内参的相机帧")
        initial_rgb_path = run_dir / "rgb_selection.png"
        if cv2.imwrite(str(initial_rgb_path), bundle.color_bgr):
            report["initial_rgb_image"] = str(initial_rgb_path)
        detections = detect(yolo_model, bundle.color_bgr, args.confidence)
        if not detections:
            raise RuntimeError("CAD目标孔选择画面中 YOLO 未检测到孔")
        selected = _select_cad_holes(
            bundle.color_bgr,
            detections,
            holes_selection,
            T_base_camera_selection,
            bundle.intrinsics,
        )
        if selected is None:
            print("[CAD] 用户取消目标孔选择。")
            report["status"] = "cancelled"
            _write_report(run_dir, report, [])
            return 0
        selected_ids = list(selected)
        if any(hole_id not in holes_by_id for hole_id in selected_ids):
            raise RuntimeError(f"目标孔选择包含未知 CAD 孔号：{selected_ids}")
        depth_mode, required_depth_holes = _cad_depth_policy(
            len(selected_ids), cfg.min_depth_valid_holes,
        )

        group_target_340, group_plan = _cad_group_entry_plan(
            holes_selection,
            selected_ids,
            selection_tcp,
            handeye,
            fixed_rz_rad,
            bundle.intrinsics,
            cfg.depth_height_mm,
            cfg.group_view_margin_px,
        )
        report["selected_hole_ids"] = selected_ids
        report["stages"] = {
            "cad_group_depth_340": {
                "description": (
                    "单孔共同居中到340 mm；一次RGB-D采集该孔高度，不覆盖CAD中心/法向"
                    if depth_mode == "single_hole" else
                    "选中孔组共同居中到340 mm；一次RGB-D融合多孔深度得到共享高度，不覆盖CAD中心/法向"
                ),
                "group_plan": group_plan,
                "depth_mode": depth_mode,
                "selected_hole_count": len(selected_ids),
                "required_valid_hole_count": required_depth_holes,
                "status": "planned",
            },
        }

        # 预览模式只生成共同340位姿和逐孔CAD 260位姿，不回原点、不启动深度采集、不运动。
        planned_entries: list[dict[str, Any]] = []
        for order, hole_id in enumerate(selected_ids, start=1):
            target, entry = _cad_hole_entry_plan(
                holes_by_id[hole_id], selection_tcp, handeye, fixed_rz_rad, cfg.fine_height_mm,
            )
            entry.update({
                "order": order,
                "cad_260_nominal_target_tcp_transform_mm": target,
                "target_tcp_transform_mm": target,
                "depth_height_offset_mm": 0.0,
                "cad_260_height_correction_applied": False,
            })
            planned_entries.append(entry)
        report["stages"]["cad_target_pose_plan"] = {
            "description": (
                "CAD中心/法向规划逐孔260 mm位姿；单孔340 mm深度只测该孔高度"
                if depth_mode == "single_hole" else
                "CAD中心/法向规划逐孔260 mm位姿；共同340 mm深度只采一次并共享高度"
            ),
            "hole_order": selected_ids,
            "holes": planned_entries,
        }
        _write_report(run_dir, report, [])
        print("\n[CAD_GROUP_TARGET_PLAN] 已生成共同340 mm深度采集位姿：", flush=True)
        print(
            f"  group_center_px={np.round(np.asarray(group_plan['group_center_px']), 3).tolist()} "
            f"bbox_px={np.round(np.asarray(group_plan['group_bbox_px']), 3).tolist()} "
            f"holes={selected_ids}",
            flush=True,
        )

        if not args.execute:
            report["status"] = "preview_complete"
            _write_report(run_dir, report, [])
            print(
                f"[PREVIEW] CAD分组目标位姿已写入: {run_dir}；未回原点、未启动深度采集、未下发机器人运动。",
                flush=True,
            )
            return 0

        # 只移动一次到整组孔的共同340 mm位姿，深度在这里一次采集。
        with timing.measure("cad_group/move_to_340mm_depth_pose"):
            current_tcp = _move_to_cad_fine_pose(
                "选中孔组",
                0,
                current_tcp,
                group_target_340,
                args,
                motion_session,
                pose_session,
                target_height_mm=cfg.depth_height_mm,
                target_stage="共同深度采集",
            )
        report["motion_executed"] = True
        T_base_camera_group = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
        depth_result = _capture_cad_group_depth(
            pipeline,
            align,
            chain,
            yolo_model,
            args.confidence,
            [holes_by_id[hole_id] for hole_id in selected_ids],
            T_base_camera_group,
            cfg,
            run_dir,
        )
        depth_stage = report["stages"]["cad_group_depth_340"]
        depth_stage.update(depth_result)
        depth_stage["status"] = "passed" if bool(depth_result.get("success")) else "failed"
        if not bool(depth_result.get("success")):
            reasons = "；".join(str(item) for item in depth_result.get("failure_reasons", []))
            raise RuntimeError(f"共同340 mm深度质量门失败：{reasons}")
        depth_plane_point_base = np.asarray(depth_result["plane_point_base_mm"], dtype=np.float64).reshape(3)
        depth_height_offset_mm = float(depth_result["height_offset_mm"])
        group_camera_axis_z_base = float(T_base_camera_group[2, 2])
        if abs(group_camera_axis_z_base) < 0.15:
            raise RuntimeError("共同340 mm深度采集位姿的相机光轴过于接近基坐标水平面，不能做共享基坐标Z修正")
        shared_base_z_correction_mm = depth_height_offset_mm / group_camera_axis_z_base
        report["stages"]["cad_group_depth_340"].update({
            "group_camera_axis_z_base": group_camera_axis_z_base,
            "shared_base_z_correction_mm": shared_base_z_correction_mm,
            "height_correction_policy": "shared_depth_offset_only; CAD XY/normal/IDs locked",
            "depth_plane_point_base_mm": depth_plane_point_base,
        })
        print(
            f"[CAD_GROUP_DEPTH] height={float(depth_result['height_median_mm']):.3f} mm, "
            f"nominal={float(cfg.depth_height_mm):.3f} mm, "
            f"offset={depth_height_offset_mm:+.3f} mm, "
            f"shared_base_z_correction={shared_base_z_correction_mm:+.3f} mm",
            flush=True,
        )

        results: list[dict[str, Any]] = []
        for order, hole_id in enumerate(selected_ids, start=1):
            T_base_camera_current = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
            holes_current = _cad_holes_for_current_camera(cad_input, T_base_camera_current)
            hole = {str(item["hole_id"]): item for item in holes_current}[hole_id]
            target_260_nominal, entry = _cad_hole_entry_plan(
                hole, current_tcp, handeye, fixed_rz_rad, cfg.fine_height_mm,
            )
            # 340 mm 深度只给出整块顶面的共享高度偏差；CAD 孔中心、XY 和法向不改。
            target_260, hole_base_z_correction_mm, hole_camera_axis_z_base = (
                _apply_shared_cad_depth_height_offset(
                    target_260_nominal,
                    handeye.T_tcp_rgb_camera,
                    depth_height_offset_mm,
                )
            )
            if abs(hole_camera_axis_z_base - group_camera_axis_z_base) > 0.02:
                raise RuntimeError(
                    f"CAD孔{hole_id}与共同340 mm位姿的相机光轴方向不一致，"
                    f"不能安全复用共享高度偏差："
                    f"group={group_camera_axis_z_base:.6f}, hole={hole_camera_axis_z_base:.6f}"
                )
            corrected_height = camera_height_to_plane_mm(
                target_260,
                handeye.T_tcp_rgb_camera,
                depth_plane_point_base,
            )
            if abs(float(corrected_height) - float(cfg.fine_height_mm)) > 0.75:
                raise RuntimeError(
                    f"CAD孔{hole_id}应用共同深度偏差后高度校验失败："
                    f"{float(corrected_height):.3f} mm != {float(cfg.fine_height_mm):.3f} mm"
                )
            entry.update({
                "cad_260_nominal_target_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(target_260_nominal),
                "cad_260_nominal_target_tcp_transform_mm": target_260_nominal,
                "depth_height_offset_mm": depth_height_offset_mm,
                "shared_base_z_correction_mm": shared_base_z_correction_mm,
                "hole_base_z_correction_mm": hole_base_z_correction_mm,
                "hole_camera_axis_z_base": hole_camera_axis_z_base,
                "cad_260_height_correction_applied": True,
                "achieved_camera_height_mm": float(corrected_height),
                "target_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(target_260),
                "target_tcp_transform_mm": target_260,
            })
            plan_entries = report["stages"]["cad_target_pose_plan"].get("holes", [])
            for plan_entry in plan_entries:
                if str(plan_entry.get("hole_id")) == str(hole_id):
                    plan_entry.update({
                        "cad_260_corrected_target_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(target_260),
                        "cad_260_corrected_target_tcp_transform_mm": target_260,
                        "depth_height_offset_mm": depth_height_offset_mm,
                        "shared_base_z_correction_mm": shared_base_z_correction_mm,
                        "hole_base_z_correction_mm": hole_base_z_correction_mm,
                        "cad_260_height_correction_applied": True,
                    })
                    break
            with timing.measure(f"hole_{hole_id}/move_to_260mm", hole_id=hole_id, order=order):
                current_tcp = _move_to_cad_fine_pose(
                    hole_id, order, current_tcp, target_260,
                    args, motion_session, pose_session,
                )
            _, current_tcp = _require_safe_snapshot(pose_session)
            tcp_at_260 = current_tcp.copy()
            T_base_camera_fine = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
            expected_center = _project_base_point_to_pixel(
                np.asarray(hole["point_base_mm"], dtype=np.float64),
                T_base_camera_fine,
                bundle.intrinsics,
            )
            expected_radius_px = _cad_hole_projected_radius_px(
                hole, T_base_camera_fine, bundle.intrinsics,
            )
            with timing.measure(f"hole_{hole_id}/rgb_yolo_fine", hole_id=hole_id, order=order):
                fine = _capture_cad_fine_yolo(
                    pipeline,
                    yolo_model,
                    args.confidence,
                    expected_center,
                    bundle.intrinsics,
                    cfg,
                    hole_id,
                    run_dir=run_dir,
                    expected_radius_px=expected_radius_px,
                )
            fine_intrinsics = fine["intrinsics"]
            observed_point_base = pixel_to_base_plane(
                fine["center_px"],
                fine_intrinsics,
                current_tcp,
                handeye.T_tcp_rgb_camera,
                np.asarray(hole["point_base_mm"], dtype=np.float64),
                np.asarray(hole["normal_base"], dtype=np.float64),
                center_is_undistorted=True,
            )
            # CAD继续提供平面Z；YOLO只把最终XY拉回真实孔中心。
            hole_center_base = np.asarray(observed_point_base, dtype=np.float64).copy()
            hole_center_base[2] = float(np.asarray(hole["point_base_mm"])[2])
            final_target_mode = str(getattr(args, "final_target_mode", DEFAULT_FINAL_TARGET_MODE))
            final_x_offset_mm, final_z_offset_mm = final_point_offsets_for_mode(final_target_mode)
            final_target_point_base = apply_final_point_base_offsets(
                hole_center_base, final_x_offset_mm, final_z_offset_mm,
            )
            final_xy_motion: dict[str, Any] | None = None
            final_z_motion: dict[str, Any] | None = None
            final_y_trim_motion: dict[str, Any] | None = None
            if args.move_final_xy:
                fixed_offset = (
                    None if args.tcp_xy_offset_mm is None else (
                        float(args.tcp_xy_offset_mm[0]), float(args.tcp_xy_offset_mm[1]),
                    )
                )
                xy_target, tcp_before = plan_final_tcp_xy(
                    current_tcp, final_target_point_base, fixed_offset,
                )
                correction_xy = xy_target[:2, 3] - final_target_point_base[:2]
                with timing.measure(f"hole_{hole_id}/final_xy", hole_id=hole_id, order=order):
                    current_tcp = _confirm_and_move_line(
                        f"CAD孔{hole_id} YOLO最终XY修正",
                        current_tcp,
                        xy_target,
                        args,
                        motion_session,
                        pose_session,
                        "只使用YOLO/RGB孔中心修正XY；CAD继续提供Z和法向",
                        require_confirmation=False,
                        motion_profile="precision",
                    )
                final_xy_motion = {
                    "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(xy_target),
                    "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                    "tcp_position_before_mm": tcp_before,
                    "hole_center_base_mm": hole_center_base,
                    "target_point_base_mm": final_target_point_base,
                    "xy_correction_mm": correction_xy,
                    "compensation_mode": "charuco_affine_model" if fixed_offset is None else "fixed_offset_override",
                    "charuco_model_source": str(CHARUCO_XY_MODEL_SOURCE) if fixed_offset is None else None,
                    "tcp_xy_offset_mm": None if fixed_offset is None else list(fixed_offset),
                }
                z_target = plan_final_tcp_base_z(current_tcp, final_target_point_base)
                with timing.measure(f"hole_{hole_id}/final_z", hole_id=hole_id, order=order):
                    current_tcp = _confirm_and_move_line(
                        f"CAD孔{hole_id}移动到最终Z",
                        current_tcp,
                        z_target,
                        args,
                        motion_session,
                        pose_session,
                        f"保留CAD孔Z与机械爪偏置；最终点模式={final_target_mode}",
                        require_confirmation=False,
                        motion_profile="precision",
                    )
                final_z_motion = {
                    "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(z_target),
                    "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                    "target_base_z_mm": float(z_target[2, 3]),
                    "final_point_offset_base_mm": [final_x_offset_mm, 0.0, final_z_offset_mm],
                }
                y_target = plan_final_tcp_base_y_trim(current_tcp)
                with timing.measure(f"hole_{hole_id}/final_y_plus_0_3mm", hole_id=hole_id, order=order):
                    current_tcp = _confirm_and_move_line(
                        f"CAD孔{hole_id}最终基坐标+Y微调",
                        current_tcp,
                        y_target,
                        args,
                        motion_session,
                        pose_session,
                        f"基坐标Y增加 {FINAL_BASE_Y_AFTER_Z_MM:.3f} mm",
                        require_confirmation=False,
                        motion_profile="precision",
                    )
                final_y_trim_motion = {
                    "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(y_target),
                    "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                    "delta_base_y_mm": FINAL_BASE_Y_AFTER_Z_MM,
                }
            fine_visualization = fine.get("visualization")
            if isinstance(fine_visualization, dict):
                result_image = _annotate_cad_fine_result_overlay(
                    fine_visualization.get("summary_image"),
                    hole_id,
                    final_xy_motion,
                    final_z_motion,
                    final_y_trim_motion,
                )
                if result_image is not None:
                    fine_visualization["result_image"] = result_image
                print(
                    f"[CAD_FINE] hole={hole_id} 260mm "
                    f"valid={int(fine.get('valid_frames', 0))}/{int(fine.get('total_frames', 0))} "
                    f"P95={float(fine.get('center_scatter_p95_px', float('nan'))):.3f}px "
                    f"mean_CAD_error={float(fine.get('mean_distance_to_cad_projection_px', float('nan'))):.3f}px "
                    f"summary={fine_visualization.get('result_image') or fine_visualization.get('summary_image')}",
                    flush=True,
                )
            fine_report = dict(fine)
            fine_report.pop("intrinsics", None)
            result = {
                "status": "completed",
                "order": order,
                "hole_id": hole_id,
                "cad_center_base_mm": np.asarray(hole["point_base_mm"], dtype=np.float64),
                "cad_normal_base": np.asarray(hole["normal_base"], dtype=np.float64),
                "normal_toward_camera_base": np.asarray(hole["normal_toward_camera_base"], dtype=np.float64),
                "cad_260_target_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(target_260),
                "cad_260_nominal_target_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(target_260_nominal),
                "cad_260_actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(tcp_at_260),
                "cad_260_pose_geometry": entry["pose_geometry"],
                "depth_height_offset_mm": depth_height_offset_mm,
                "shared_base_z_correction_mm": shared_base_z_correction_mm,
                "hole_base_z_correction_mm": hole_base_z_correction_mm,
                "hole_camera_axis_z_base": hole_camera_axis_z_base,
                "fine_yolo": fine_report,
                "hole_center_base_mm": hole_center_base,
                "target_point_base_mm": final_target_point_base,
                "final_point_mode": final_target_mode,
                "final_point_offset_base_mm": [final_x_offset_mm, 0.0, final_z_offset_mm],
                "final_xy_motion": final_xy_motion,
                "final_z_motion": final_z_motion,
                "final_y_trim_motion": final_y_trim_motion,
            }
            results.append(result)
            report.setdefault("stages", {})[f"hole_{hole_id}"] = result
            report["stages"]["processed_holes"] = {
                "completed_count": len(results),
                "total_count": len(selected_ids),
                "hole_order": selected_ids,
                "holes": results,
            }
            report["motion_executed"] = True
            _write_report(run_dir, report, [])
            if order < len(selected_ids):
                next_hole_id = selected_ids[order]
                if _request_next_cad_hole_confirmation(hole_id, next_hole_id) != "m":
                    raise RuntimeError(f"用户在CAD孔{hole_id}完成后停止流程")

        report["status"] = (
            "completed_experimental_handeye"
            if experimental_handeye_motion else "completed"
        )
        report["final_tcp_pose_m_rad"] = transform_to_sdk_pose_m_rad(current_tcp)
        _write_report(run_dir, report, [])
        print(f"[DONE] CAD运动流程完成，结果目录: {run_dir}", flush=True)
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        _write_report(run_dir, report, [])
        raise
    finally:
        try:
            if pipeline is not None:
                pipeline.stop()
        except Exception:
            pass
        if pose_session is not None:
            pose_session.disconnect()
        if motion_session is not None:
            motion_session.disconnect()
        cv2.destroyAllWindows()
        report["timing"] = timing.snapshot()
        try:
            _write_report(run_dir, report, [])
        except Exception as exc:
            print(f"[CAD] 最终报告写入失败：{type(exc).__name__}: {exc}", flush=True)
        timing.print_summary()


def run_two_stage_hole_localization(args: Any, handeye: Any, model: Any) -> int:
    """执行“初始多孔选择 -> 按孔号逐个粗/精定位 -> 逐孔目标点运动”的受确认流程。"""
    cfg = TwoStageConfig(
        coarse_height_mm=float(args.coarse_height_mm), fine_height_mm=float(args.fine_height_mm),
        coarse_frames=int(args.coarse_frames), fine_frames=int(args.fine_frames),
        fine_settle_discard_frames=int(args.fine_settle_discard_frames),
        fine_retry_count=int(args.fine_retries),
    )
    precision_speed_m_s = float(args.speed_m_s)
    precision_acc_m_s2 = float(args.acc_m_s2)
    transit_speed_m_s = float(getattr(args, "transit_speed_m_s", precision_speed_m_s))
    transit_acc_m_s2 = float(getattr(args, "transit_acc_m_s2", precision_acc_m_s2))
    approach_speed_m_s = float(getattr(args, "approach_speed_m_s", 0.12))
    approach_acc_m_s2 = float(getattr(args, "approach_acc_m_s2", 0.35))
    final_target_mode = str(getattr(args, "final_target_mode", DEFAULT_FINAL_TARGET_MODE))
    final_x_offset_mm, final_z_offset_mm = final_point_offsets_for_mode(final_target_mode)
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
    if cfg.fine_frames < cfg.fine_stable_min_frames:
        raise ValueError(
            f"精定位最大帧数必须不少于稳定门帧数：{cfg.fine_frames} < {cfg.fine_stable_min_frames}"
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
        "final_target_mode": final_target_mode,
        "final_point_offset_base_mm": [final_x_offset_mm, 0.0, final_z_offset_mm],
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
                _,
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
    p.add_argument(
        "--cad-motion",
        dest="cad_motion",
        action="store_true",
        default=DEFAULT_CAD_MOTION,
        help="CAD运动路径：回原点选孔组，共同340 mm采一次深度，再逐孔CAD 260 mm并用RGB/YOLO做最终XY修正",
    )
    p.add_argument(
        "--cad-allow-experimental-handeye",
        dest="cad_allow_experimental_handeye",
        action="store_true",
        default=False,
        help="仅CAD实验运动：允许未通过生产验证的手眼结果；报告会标记experimental_handeye",
    )
    p.add_argument(
        "--cad-registration-report",
        type=Path,
        default=None,
        help="历史 CAD 配准报告（仅在缺省 CAD JSON 时取模型路径；每次 CAD 运动都会重新实时配准）",
    )
    p.add_argument(
        "--cad-model-json",
        type=Path,
        default=None,
        help="CAD孔位JSON；默认使用配准报告中的路径或 data/cad_model/cad_hole_model.json",
    )
    p.add_argument("--cad-fine-height-mm", type=float, default=260.0,
                   help="CAD路径的RGB相机孔面精定位高度，默认260 mm")
    p.add_argument("--cad-depth-height-mm", type=float, default=340.0,
                   help="CAD路径整组孔共同深度采集高度，默认340 mm")
    p.add_argument("--cad-depth-frames", type=int, default=8,
                   help="CAD路径共同340 mm深度采集帧数")
    p.add_argument("--cad-depth-min-valid", type=int, default=5,
                   help="CAD路径共同深度最少有效帧数")
    p.add_argument("--cad-depth-min-holes", type=int, default=4,
                   help="CAD路径多孔模式每帧最少有效孔数；单孔模式自动按1个孔计算，2/3孔时按选中数量计算")
    p.add_argument("--cad-depth-plane-rmse-mm", type=float, default=3.5,
                   help="CAD路径共同深度孔口环带平面RMSE门限(mm)")
    p.add_argument("--cad-depth-height-p95-mm", type=float, default=2.0,
                   help="CAD路径共同深度高度跨帧P95门限(mm)")
    p.add_argument("--cad-group-view-margin-px", type=float, default=35.0,
                   help="CAD路径共同340 mm位姿的视野边缘安全余量(px)")
    p.add_argument("--cad-fine-frames", type=int, default=12,
                   help="CAD路径每孔RGB/YOLO精定位采集帧数")
    p.add_argument("--cad-fine-min-valid", type=int, default=6,
                   help="CAD路径每孔最少有效YOLO帧数")
    p.add_argument("--cad-fine-center-p95-px", type=float, default=1.5,
                   help="CAD路径YOLO中心跨帧P95门限(px)")
    p.add_argument("--cad-yolo-match-tolerance-px", type=float, default=70.0,
                   help="CAD投影中心与YOLO中心的最大关联距离(px)")
    p.add_argument("--cad-settle-discard-frames", type=int, default=10,
                   help="到达260 mm后丢弃的RGB预热帧数")
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
    p.add_argument("--fine-settle-discard-frames", type=int, default=10,
                   help="每次精定位采集前丢弃的机器人/相机预热RGB帧数，默认10")
    p.add_argument("--fine-retries", type=int, default=2,
                   help="单孔精定位质量门失败后的自动重拍次数，默认2")
    p.add_argument(
        "--final-target-mode",
        choices=(FINAL_TARGET_MODE_GRIPPER, FINAL_TARGET_MODE_NORMAL),
        default=DEFAULT_FINAL_TARGET_MODE,
        help="最终点模式：gripper=机械爪模式(X+64,Z+50)，normal=平常模式(无X/Z偏置)",
    )
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
    if args.cad_motion:
        if args.image:
            raise RuntimeError("CAD运动路径必须使用实时RGB流，不能使用 --image")
        return run_cad_motion_workflow(args, handeye, model)
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
