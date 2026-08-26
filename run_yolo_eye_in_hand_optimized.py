#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO + RGB-D + RGB 手眼标定的“眼在手上”孔中心定位脚本。

流程：在初始 RGB-D 画面中选择目标孔 -> 340 mm 粗定位建立局部点云、平面和法向
-> 260 mm RGB/YOLO 精定位修正最终 XY -> 移动到当前目标点。

默认进入两阶段流程但只做预览；必须显式启用运动和相应的验证开关后，才会连接运动控制。
默认不允许使用未通过生产验证的实验手眼结果。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, replace
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
from aubo_workbench.coarse_cache import (  # noqa: E402
    CacheValidationGates,
    CoarseCacheEntry,
    cache_entry_compatibility_reasons,
    load_cache_entries,
    load_persistent_cache_entries,
    match_entries_by_base_point,
    rekey_cache_entry,
    save_cache_entries,
    save_persistent_cache_entries,
    transform_cached_points_to_camera,
    validate_cache_entry,
    replace_base_z,
)
from aubo_workbench.config import (  # noqa: E402
    ROBOT_CFG,
    apply_robot_connection_overrides,
)
from aubo_workbench.io_utils import jsonable, write_dict_rows  # noqa: E402
from tools.visualize_coarse_cache import CacheCloud, render_cache_cloud  # noqa: E402
from aubo_workbench.fitting import fit_plane, fit_sphere  # noqa: E402
from aubo_workbench.geometry import (  # noqa: E402
    angle_between_deg,
    invert_transform,
    make_transform,
    matrix_to_rpy_zyx,
    rotx,
    roty,
    rotz,
    transform_to_pose6_rzryrx,
    transform_to_sdk_pose_m_rad,
    unit_vector,
)
from aubo_workbench.optics import (  # noqa: E402
    base_z_target_for_camera_height,
    camera_height_to_plane_mm,
    camera_matrix,
    camera_ray,
    camera_transform,
    correct_projected_circle_center,
    distortion_coeffs,
    pixel_to_base_plane,
    plane_basis,
    project_undistorted_pixels,
    ray_plane_intersection,
    undistort_pixels,
)
from aubo_workbench.paths import (  # noqa: E402
    HANDEYE_CANDIDATE_PATH,
    HOLE_LOCALIZATION_COARSE_CACHE_DIR,
    MODEL_PATH,
    TCP_ABSOLUTE_XY_MODEL_DIR,
    HOLE_LOCALIZATION_RUNS_DIR,
)


DEFAULT_MODEL = MODEL_PATH
DEFAULT_HANDEYE = HANDEYE_CANDIDATE_PATH
WINDOW = "YOLO eye-in-hand hole selection (click hole, Enter=confirm, Esc=quit)"
RUNS_DIR = HOLE_LOCALIZATION_RUNS_DIR
HOLE_DIAMETERS_MM = (65.0, 70.0, 75.0)
# 旧两阶段点云只允许使用孔口外侧、朝相机最近的表面。
COARSE_SURFACE_SELECTION_POLICY = "front_surface_outer_ring_v2"
COARSE_SURFACE_MODEL = "local_tangent_plane_front_surface_outer_ring_v2"
FINAL_TARGET_MODE_GRIPPER = "gripper"
FINAL_TARGET_MODE_NORMAL = "normal"
DEFAULT_FINAL_TARGET_MODE = FINAL_TARGET_MODE_GRIPPER
GRIPPER_BASE_X_OFFSET_MM = 64.0
GRIPPER_BASE_Z_OFFSET_MM = 50.0
FINAL_BASE_Y_AFTER_Z_MM = 0.2
# 与上面的基坐标Y微调融合为一次移动执行，避免最终Z到位后再做第二次单独的+Y移动。
FINAL_TOOL_Y_AFTER_Z_MM = 1.0
# 三孔逐孔安放时，所有低位横移前先抬到该安全余量。
THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM = 60.0
SHARED_OBSERVATION_MIN_LIFT_MM = 10.0
SHARED_OBSERVATION_MIN_DESCENT_MM = 10.0
# 机器人到位检测只影响轮询响应，不改变控制器的运动轨迹。
ROBOT_STEADY_POLL_INTERVAL_S = 0.10
# 回到原点后，开始下一轮初始拍摄前再留出一小段静止缓冲，避免相机抓到末端
# 刚停止时的残余振动帧或控制器状态切换瞬间。
ROBOT_INITIAL_CAPTURE_SETTLE_DELAY_S = 0.50


class TwoStageSelectionCancelled(RuntimeError):
    """用户在回原点后的新一轮初始选孔窗口中按 Esc 退出。"""


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
    TCP_ABSOLUTE_XY_MODEL_DIR / "charuco-tcp-xy-20260727_174559" / "report.json"
)

# 直接运行的默认模式：不需要额外命令行参数即可做离线/现场预览。
# 只有明确传入 --execute 时才允许连接运动控制和下发机器人命令。
DEFAULT_TWO_STAGE_HOLE_LOCALIZATION = True
DEFAULT_EXECUTE_MOTION = False
DEFAULT_ALLOW_EXPERIMENTAL_HANDEYE = False
DEFAULT_MOVE_FINAL_XY = False


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
    # 跨运行缓存现场验证：到位后丢弃相机/末端预热帧，再用更多稳定帧验证。
    cache_validation_settle_discard_frames: int = 10
    cache_validation_frames: int = 8
    cache_validation_min_valid: int = 5
    # 缓存缺失或损坏回退完整粗定位时，到达340 mm后给机械臂/相机的
    # 固定停稳缓冲；直接命中base缓存时不等待、不做现场复核。
    coarse_settle_delay_s: float = 0.6
    # 每次单孔粗拍重拍前重新确认机器人停稳，并丢弃相机队列中的预热帧。
    coarse_recapture_settle_discard_frames: int = 15
    # 单孔粗定位最多允许两次姿态纠偏，最后一次采集只做验证，不再盲目运动。
    coarse_max_corrections: int = 2
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
    # 批量粗定位模式参数
    batch_coarse_localization: bool = False
    batch_coarse_frames: int = 15
    batch_coarse_min_valid: int = 10
    batch_coarse_min_holes_per_frame: int | None = None
    batch_coarse_view_margin_px: float = 50.0
    shared_cache_validation: bool = False
    shared_cache_validation_frames: int = 3
    shared_cache_validation_min_valid: int = 2
    shared_cache_validation_view_margin_px: float = 50.0
    # 批量移动完成后先丢弃相机队列中的旧帧，避免把运动过程帧融合进固定TCP位姿。
    batch_coarse_settle_discard_frames: int = 10
    # 所有340mm粗定位（批量和逐孔）共用同一套稳定性门。
    max_coarse_tracking_distance_p95_px: float = 15.0
    # 260mm批量精定位：在同一视野内一次采集并精定位全部已选孔。
    batch_fine_localization: bool = True
    batch_fine_frames: int = 8
    batch_fine_min_valid: int = 5
    batch_fine_stable_min_frames: int = 5
    # 260mm下降后RGB pipeline里会残留运动过程帧。现场实测丢3帧只耗时
    # 3.7ms，仍未清空队列；默认与可靠的逐孔精定位一致，至少丢10帧。
    batch_fine_settle_discard_frames: int = 10
    # 共享精拍某些孔的严格几何有效帧不足时，允许移动到失败孔的共同
    # 260mm观察位补拍；补拍仍按孔集合共享，不退化为逐孔精拍。
    batch_fine_supplement_rounds: int = 1
    batch_fine_view_margin_px: float = 50.0


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
    points_camera_mm: np.ndarray | None = None
    surface_selection_policy: str = "legacy"
    front_surface_z_mm: float | None = None
    ring_points_raw: int | None = None
    surface_points_selected: int | None = None


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
    tracking_distance_px: float | None = None


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




def hole_camera_point(
    ring_center_xy: tuple[float, float],
    xyz_map_mm: np.ndarray,
    intrinsics: Any,
    radius_px: float,
    *,
    ray_center_xy: tuple[float, float] | np.ndarray | None = None,
    ray_center_is_undistorted: bool = False,
    include_points: bool = False,
    surface_selection_policy: str = "legacy",
) -> tuple[np.ndarray, dict[str, Any]]:
    """用孔外环带拟合平面，再将孔中心射线与平面求交。

    ring_center_xy 必须是原始图像坐标，用于在 aligned xyz_map 中选择深度环带。
    ray_center_xy 可以是去畸变轮廓拟合得到的中心；这样深度采样坐标和几何射线
    分别使用各自正确的坐标域，避免把去畸变像素直接拿去索引原始深度图。
    """
    ring_u, ring_v = map(float, ring_center_xy)
    h, w = xyz_map_mm.shape[:2]
    policy = str(surface_selection_policy).strip().lower()
    if policy == COARSE_SURFACE_SELECTION_POLICY:
        # 当前工件孔距较密，旧1.25R～1.50R环带会越过孔口附近曲面，
        # 接近相邻孔边和凸起结构，使局部Z/法向变成大范围曲面平均值。
        # 收回到孔口外侧的窄环；仍保留0.10R安全距离，并继续由前景深度
        # 簇过滤孔壁/孔底，不依靠把环带无限外移来规避无效深度。
        ring_inner_factor = 1.10
        ring_outer_factor = 1.30
        front_percentile = 5.0
        surface_band_mm = 8.0
        min_surface_fraction = 0.03
        surface_model = COARSE_SURFACE_MODEL
    elif policy == "legacy":
        ring_inner_factor = 1.08
        ring_outer_factor = 1.35
        front_percentile = 20.0
        surface_band_mm = 8.0
        min_surface_fraction = 0.10
        surface_model = "local_tangent_plane"
    else:
        raise ValueError(f"未知点云表面筛选策略：{surface_selection_policy!r}")

    # 只在孔附近建立环带网格，避免每个粗定位帧都为整幅RGB-D图创建
    # 1280x800级别的坐标矩阵；环带定义按策略选择。
    ring_outer_px = max(8.0, radius_px * ring_outer_factor)
    x0 = max(0, int(math.floor(ring_u - ring_outer_px - 1.0)))
    x1 = min(w, int(math.ceil(ring_u + ring_outer_px + 2.0)))
    y0 = max(0, int(math.floor(ring_v - ring_outer_px - 1.0)))
    y1 = min(h, int(math.ceil(ring_v + ring_outer_px + 2.0)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("孔环带超出深度图范围，无法提取点云")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - ring_u, yy - ring_v)
    ring = (
        (rr >= max(4.0, radius_px * ring_inner_factor))
        & (rr <= max(8.0, radius_px * ring_outer_factor))
    )
    points = xyz_map_mm[y0:y1, x0:x1][ring]
    points = points[np.isfinite(points).all(axis=1)]
    points = points[(points[:, 2] > 100.0) & (points[:, 2] < 3000.0)]
    raw_ring_points = int(len(points))
    if len(points) < 12:
        raise ValueError("孔周围有效深度点不足，无法拟合局部切平面")

    # 镀膜伞具的孔壁/孔内深度会落入环带；它们通常比外表面离相机更远，
    # 混入后会把局部平面RMSE拉到二十多毫米。按前景深度簇筛选外表面，
    # 同时保留足够带宽覆盖孔面倾斜与局部曲率。
    front_surface_z = float(np.percentile(points[:, 2], front_percentile))
    # 相机坐标Z沿视线向远处增加，因此孔口前表面必须取最小深度簇，
    # 不能用“离第20百分位绝对值相近”把更远的孔底簇带回来。
    surface_points = points[points[:, 2] <= front_surface_z + surface_band_mm]
    min_surface_points = max(80, int(len(points) * min_surface_fraction))
    if len(surface_points) < min_surface_points:
        if policy == COARSE_SURFACE_SELECTION_POLICY:
            raise ValueError(
                "孔口前表面有效点不足，拒绝把孔底/孔壁混入平面："
                f"{len(surface_points)}/{min_surface_points}"
            )
    else:
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
    result = {
        "ring_points": int(len(points)),
        "ring_points_raw": raw_ring_points,
        "front_surface_z_mm": front_surface_z,
        "plane_rmse_mm": plane_rmse,
        "plane_normal_camera": normal.tolist(),
        # 与孔中心 point_camera_mm 分开记录环带拟合平面上的代表点；
        # 当前中心仍由中心射线与该局部平面求交得到。
        "local_plane_point_camera_mm": plane_point.tolist(),
        "plane_point_camera_mm": point.tolist(),
        "surface_model": surface_model,
        "surface_selection_policy": policy,
        "ring_inner_factor": float(ring_inner_factor),
        "ring_outer_factor": float(ring_outer_factor),
        "front_surface_percentile": float(front_percentile),
        "surface_band_mm": float(surface_band_mm),
        "surface_points_selected": int(len(points)),
        "sphere_center_camera_mm": sphere_center.tolist() if sphere_center is not None else None,
        "sphere_radius_mm": sphere_radius,
        "sphere_rmse_mm": sphere_rmse,
        "ring_center_px_distorted": [ring_u, ring_v],
        "ray_center_px": ray_center.tolist(),
        "ray_center_is_undistorted": bool(ray_center_is_undistorted),
    }
    if include_points:
        result["points_camera_mm"] = np.asarray(points, dtype=np.float32)
    return point, result


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


# 纯几何/光学层已抽到 aubo_workbench.geometry 与 aubo_workbench.optics。
# 下划线别名保留给本文件内的既有调用点与 tests 的属性式访问。
_unit = unit_vector
_angle_deg = angle_between_deg
_matrix_to_rpy_zyx = matrix_to_rpy_zyx
_camera_matrix = camera_matrix
_distortion = distortion_coeffs
_plane_basis = plane_basis
_project_undistorted_pixels = project_undistorted_pixels


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


def compose_batch_fine_xy_with_coarse_z(
    fine_point_base: np.ndarray,
    coarse_point_base: np.ndarray,
) -> np.ndarray:
    """共享精定位只提供XY；最终点Z继续采用该孔的粗定位结果。"""
    target = np.asarray(fine_point_base, dtype=np.float64).reshape(3).copy()
    target[2] = float(np.asarray(coarse_point_base, dtype=np.float64).reshape(3)[2])
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


def plan_final_tcp_combined_y_trim(
    T_base_tcp: np.ndarray,
    base_delta_y_mm: float = FINAL_BASE_Y_AFTER_Z_MM,
    tool_delta_y_mm: float = FINAL_TOOL_Y_AFTER_Z_MM,
) -> np.ndarray:
    """保持当前 TCP 姿态，把基坐标+Y微调和工具系+Y微调合并成一次平移执行。

    两次平移都不改变姿态（旋转矩阵不变），所以可以直接把基坐标Y分量和
    工具系Y轴（当前旋转矩阵第2列，在base系下的方向）分量相加成一个
    位移向量，一次性移动到位，避免拆成两次串行移动。
    """
    T_base_tcp = np.asarray(T_base_tcp, dtype=np.float64)
    target = T_base_tcp.copy()
    tool_y_axis_base = T_base_tcp[:3, 1]
    delta_base_mm = np.array([0.0, float(base_delta_y_mm), 0.0]) + tool_y_axis_base * float(tool_delta_y_mm)
    target[:3, 3] += delta_base_mm
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
            # 最小二乘初值之后再做确定性的鲁棒圆拟合。反光横线会贡献
            # 大量非圆边缘；只用与同一圆一致的径向内点重新求解亚像素圆心。
            refined_center, refined_radius = _robust_refine_circle_from_edges(
                raw_edge_points, refined_center, refined_radius,
            )
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
    # 霍夫圆只有达到与正式精定位一致的严格几何质量时才优先。旧的
    # 80deg/2.5px放宽门会让反光横线污染的圆覆盖掉更可靠的轮廓椭圆。
    if (
        hough_best is not None
        and hough_best["coverage_deg"] >= 200.0
        and hough_best["residual_px"] <= 0.9
    ):
        return hough_best
    return best


def _robust_refine_circle_from_edges(
    edge_points: np.ndarray,
    initial_center: np.ndarray,
    initial_radius: float,
    *,
    iterations: int = 180,
    inlier_threshold_px: float = 1.5,
) -> tuple[np.ndarray, float]:
    """在霍夫圆附近用RANSAC径向内点抑制横线、反光和缺口。"""
    points = np.asarray(edge_points, dtype=np.float64).reshape(-1, 2)
    center0 = np.asarray(initial_center, dtype=np.float64).reshape(2)
    radius0 = float(initial_radius)
    if len(points) < 30 or not np.isfinite(center0).all() or radius0 <= 0.0:
        return center0.copy(), radius0

    radial0 = np.linalg.norm(points - center0, axis=1)
    candidate = points[np.abs(radial0 - radius0) <= max(4.0, 0.10 * radius0)]
    if len(candidate) < 30:
        return center0.copy(), radius0

    def circle_from_three(sample: np.ndarray) -> tuple[np.ndarray, float] | None:
        a = 2.0 * (sample[1:] - sample[0])
        b = np.sum(sample[1:] * sample[1:], axis=1) - float(sample[0] @ sample[0])
        if abs(float(np.linalg.det(a))) < 1.0e-6:
            return None
        try:
            center = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            return None
        radius = float(np.linalg.norm(sample[0] - center))
        if not np.isfinite(center).all() or not math.isfinite(radius) or radius <= 0.0:
            return None
        return center, radius

    rng = np.random.default_rng(0)
    best_mask: np.ndarray | None = None
    best_key: tuple[int, int, float] | None = None
    for _ in range(max(1, int(iterations))):
        sample = candidate[rng.choice(len(candidate), size=3, replace=False)]
        fitted = circle_from_three(sample)
        if fitted is None:
            continue
        center, radius = fitted
        if (
            np.linalg.norm(center - center0) > max(6.0, 0.08 * radius0)
            or abs(radius - radius0) > max(6.0, 0.12 * radius0)
        ):
            continue
        residual = np.abs(np.linalg.norm(candidate - center, axis=1) - radius)
        mask = residual <= float(inlier_threshold_px)
        if int(np.count_nonzero(mask)) < 24:
            continue
        inlier_points = candidate[mask]
        angles = np.mod(
            np.degrees(np.arctan2(
                inlier_points[:, 1] - center[1], inlier_points[:, 0] - center[0],
            )),
            360.0,
        )
        coverage_bins = int(len(np.unique(np.floor(angles / 10.0).astype(int))))
        key = (
            coverage_bins,
            int(np.count_nonzero(mask)),
            -float(np.median(residual[mask])),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_mask = mask

    if best_mask is None:
        return center0.copy(), radius0

    inliers = candidate[best_mask]
    center = center0.copy()
    radius = radius0
    for _ in range(3):
        A = np.column_stack((2.0 * inliers[:, 0], 2.0 * inliers[:, 1], np.ones(len(inliers))))
        b = np.sum(inliers * inliers, axis=1)
        try:
            solution, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            break
        center = np.asarray(solution[:2], dtype=np.float64)
        radius_sq = float(solution[2] + center @ center)
        if not math.isfinite(radius_sq) or radius_sq <= 0.0:
            return center0.copy(), radius0
        radius = math.sqrt(radius_sq)
        residual = np.abs(np.linalg.norm(candidate - center, axis=1) - radius)
        next_inliers = candidate[residual <= float(inlier_threshold_px)]
        if len(next_inliers) < 24:
            break
        inliers = next_inliers
    return center, radius


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
    """按本脚本的空表回退列写 CSV；落盘细节统一在 io_utils.write_dict_rows。"""
    write_dict_rows(path, rows, fallback_fields=("stage", "frame_index", "error"))


# 实现已统一到 aubo_workbench.io_utils.jsonable。
_jsonable = jsonable


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


def _wait_motion_session_steady(
    motion_session: Any,
    timeout_s: float = 45.0,
) -> dict[str, Any] | None:
    """用真正下发运动命令的会话确认控制器已停稳。

    ``pose_session`` 和 ``motion_session`` 是两条独立的 AUBO RPC 连接。
    只读位姿会话先读到 ``steady=True`` 时，运动会话的状态对象可能还保留
    一个短暂的 ``steady=False``，从而在下一次 ``moveLine`` 的底层安全门处
    被拒绝。位置运动前后都检查运动会话自身的状态，消除这段交接竞争。
    不具备 ``snapshot`` 接口的离线替身保持原有行为。
    """
    snapshot_fn = getattr(motion_session, "snapshot", None)
    if not callable(snapshot_fn):
        return None
    deadline = time.monotonic() + float(timeout_s)
    last_reason = ""
    while time.monotonic() < deadline:
        snapshot = snapshot_fn()
        if bool(snapshot.get("collision")):
            raise RuntimeError("运动会话检测到碰撞标志，拒绝继续位置运动")
        power_on = bool(snapshot.get("power_on"))
        steady = bool(snapshot.get("steady"))
        if power_on and steady:
            return snapshot
        last_reason = f"power={power_on} steady={steady}"
        time.sleep(ROBOT_STEADY_POLL_INTERVAL_S)
    raise RuntimeError(f"等待运动会话稳定超时：{last_reason}")


def _wait_robot_steady_before_initial_capture(
    pose_session: Any,
    *,
    settle_delay_s: float = ROBOT_INITIAL_CAPTURE_SETTLE_DELAY_S,
) -> np.ndarray:
    """在初始选孔拍摄前再次确认停稳，并丢开停止瞬间的残余振动。"""
    _, actual = _wait_robot_steady(pose_session)
    delay = max(0.0, float(settle_delay_s))
    if delay > 0.0:
        time.sleep(delay)
    _, actual = _wait_robot_steady(pose_session)
    return np.asarray(actual, dtype=np.float64).copy()


def _settle_and_discard_coarse_recapture_frames(
    pose_session: Any,
    pipeline: Any,
    align: Any,
    chain: Any,
    *,
    settle_delay_s: float,
    discard_frames: int,
) -> tuple[np.ndarray, int]:
    """重拍前重新确认停稳，并丢弃纠偏后相机队列中的预热帧。"""
    actual = _wait_robot_steady_before_initial_capture(
        pose_session,
        settle_delay_s=settle_delay_s,
    )
    discarded = 0
    for _ in range(max(0, int(discard_frames))):
        if get_aligned_frame_bundle(pipeline, align, chain) is not None:
            discarded += 1
    return np.asarray(actual, dtype=np.float64).copy(), discarded


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
    _wait_motion_session_steady(motion_session)
    from aubo_workbench.motion_control import sdk_ok
    sdk_pose = transform_to_sdk_pose_m_rad(target)
    try:
        response = motion_session.move_line(sdk_pose, speed_m_s, acc_m_s2)
    except RuntimeError as exc:
        # 在“状态检查”和真正下发之间仍可能发生一次很短的控制器状态竞争。
        # move_line 的底层 steady 门在下发前检查，不会在这里已经执行半段运动。
        if "尚未静止" not in str(exc):
            raise
        print("[MOTION] 运动会话仍在收敛，等待稳定后重试一次 moveLine", flush=True)
        _wait_motion_session_steady(motion_session)
        response = motion_session.move_line(sdk_pose, speed_m_s, acc_m_s2)
    print("[MOTION]", response)
    if not response or not sdk_ok(response[-1]):
        raise RuntimeError(f"{label} moveLine 下发失败：{response}")
    _, actual = _wait_robot_steady(pose_session)
    _wait_motion_session_steady(motion_session)
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
    _wait_motion_session_steady(motion_session)
    try:
        response = motion_session.move_joint(home.joints_rad, speed, acc)
    except RuntimeError as exc:
        if "尚未静止" not in str(exc):
            raise
        print("[MOTION] 运动会话仍在收敛，等待稳定后重试一次 moveJoint", flush=True)
        _wait_motion_session_steady(motion_session)
        response = motion_session.move_joint(home.joints_rad, speed, acc)
    print("[MOTION]", response)
    from aubo_workbench.motion_control import sdk_ok
    settled_snapshot, actual = _wait_robot_steady(pose_session)
    _wait_motion_session_steady(motion_session)
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
    """初始画面选择任意数量的孔，并允许单孔深度异常降级到共享平面导航。"""
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
        raise TwoStageSelectionCancelled("用户取消初始选孔")
    selected_indices, selection_clicks = selection
    selected = [detections[index] for index in selected_indices]
    class_ids = {int(item.get("class_id", -1)) for item in selected}
    if len(class_ids) != 1:
        raise RuntimeError("初始选择的目标不是同一个YOLO孔类别，无法进行稳定身份关联")

    holes: list[dict[str, Any]] = []
    points_camera: list[np.ndarray] = []
    normals_camera: list[np.ndarray] = []
    overlays: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    geometries: list[dict[str, Any] | None] = []
    geometry_errors: list[str | None] = []

    for hole_id, (detection_index, detection) in enumerate(
        zip(selected_indices, selected), start=1,
    ):
        selection_click = np.asarray(
            selection_clicks[detection_index], dtype=np.float64,
        ).reshape(2)
        radius = max(
            float(detection["box"][2] - detection["box"][0]),
            float(detection["box"][3] - detection["box"][1]),
        ) / 2.0
        # 初始选孔只把点击用于确定目标身份；几何中心统一采用YOLO框中心。
        yolo_center = np.asarray(detection["center"], dtype=np.float64).reshape(2)
        display_detection = dict(detection)
        display_detection["_hole_id"] = hole_id
        overlays.append((display_detection, None))
        try:
            point, info = hole_camera_point(
                tuple(yolo_center.tolist()),
                bundle.xyz_map_mm,
                bundle.intrinsics,
                radius,
                ray_center_xy=yolo_center,
                ray_center_is_undistorted=False,
                surface_selection_policy=COARSE_SURFACE_SELECTION_POLICY,
            )
            plane = _plane_estimate_from_info(
                info, f"initial multi-hole {hole_id} plane normal",
            )
            if plane.rmse_mm > float(max_plane_rmse_mm):
                raise ValueError(
                    f"深度拟合RMSE过大：{plane.rmse_mm:.3f} mm "
                    f"> {float(max_plane_rmse_mm):.3f} mm"
                )
            geometries.append({
                "point": np.asarray(point, dtype=np.float64),
                "plane": plane,
                "fallback": False,
            })
            geometry_errors.append(None)
            points_camera.append(np.asarray(point, dtype=np.float64))
            normals_camera.append(np.asarray(plane.normal_camera, dtype=np.float64))
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            geometries.append(None)
            geometry_errors.append(reason)
            print(
                f"[INITIAL_GEOMETRY_FALLBACK] hole={hole_id} reason={reason}",
                flush=True,
            )

    if not points_camera:
        reasons = "; ".join(
            f"hole_{index + 1}:{reason}"
            for index, reason in enumerate(geometry_errors)
            if reason
        )
        raise RuntimeError(f"初始选孔没有任何可用局部平面：{reasons}")

    # 共享平面只用于异常孔的安全340 mm导航和缓存匹配兜底；正常孔仍使用自己的局部平面。
    group_center_camera = np.mean(np.asarray(points_camera, dtype=np.float64), axis=0)
    group_normal_camera = _fuse_normals(normals_camera)
    successful_geometries = [
        geometry for geometry in geometries if geometry is not None
    ]
    group_rmse_mm = float(np.max([
        float(geometry["plane"].rmse_mm) for geometry in successful_geometries
    ]))
    group_ring_points = int(sum(
        int(geometry["plane"].ring_points) for geometry in successful_geometries
    ))
    group_plane = PlaneEstimate(
        point_camera_mm=group_center_camera,
        normal_camera=group_normal_camera,
        rmse_mm=group_rmse_mm,
        ring_points=group_ring_points,
        surface_model="initial_multi_hole_group",
    )

    for hole_id, (detection_index, detection), geometry, geometry_error in zip(
        range(1, len(selected) + 1),
        zip(selected_indices, selected),
        geometries,
        geometry_errors,
    ):
        selection_click = np.asarray(
            selection_clicks[detection_index], dtype=np.float64,
        ).reshape(2)
        yolo_center = np.asarray(detection["center"], dtype=np.float64).reshape(2)
        fallback = geometry is None
        if fallback:
            # 该孔的当前环带深度不可信时，只用共享平面求一个保守导航点。
            # 真正进入340 mm后仍优先直接调用缓存；无缓存则走完整粗定位。
            fallback_point = ray_plane_intersection(
                np.zeros(3, dtype=np.float64),
                camera_ray(bundle.intrinsics, yolo_center, already_undistorted=False),
                group_center_camera,
                group_normal_camera,
            )
            point = fallback_point
            plane = PlaneEstimate(
                point_camera_mm=fallback_point,
                normal_camera=group_normal_camera,
                rmse_mm=group_rmse_mm,
                ring_points=group_ring_points,
                surface_model="initial_group_plane_fallback",
                surface_plane_point_camera_mm=group_center_camera,
                surface_selection_policy=COARSE_SURFACE_SELECTION_POLICY,
            )
        else:
            point = np.asarray(geometry["point"], dtype=np.float64)
            plane = geometry["plane"]

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
            "initial_surface_selection_policy": plane.surface_selection_policy,
            "initial_front_surface_z_mm": plane.front_surface_z_mm,
            "initial_ring_points_raw": plane.ring_points_raw,
            "initial_surface_points_selected": plane.surface_points_selected,
            "initial_geometry_fallback": bool(fallback),
            "initial_geometry_fallback_reason": geometry_error,
            "pointcloud_segmentation": f"yolo_box_{COARSE_SURFACE_SELECTION_POLICY}",
        })

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
                          stop_when_stable: bool = True,
                          include_points: bool = False,
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
        if valid_count >= cfg.coarse_frames or (
            stop_when_stable and (
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
                error="tracking_distance", tracking_distance_px=tracking_distance,
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
                include_points=include_points,
                surface_selection_policy=COARSE_SURFACE_SELECTION_POLICY,
            )
            plane = _plane_estimate_from_info(plane_info, "coarse plane normal")
            valid = plane.rmse_mm <= cfg.max_plane_rmse_mm
            observations.append(Observation(
                name, attempt, center, None, plane, bundle.host_timestamp_ns,
                None if valid else "plane_quality", tracking_distance_px=tracking_distance,
            ))
        except Exception as exc:
            observations.append(Observation(name, attempt, center, None, timestamp_ns=bundle.host_timestamp_ns,
                                            error=f"plane_error:{exc}",
                                            tracking_distance_px=tracking_distance))
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


def _save_batch_fine_final_result_overlay(
    run_dir: Path,
    batch_fine_results: dict[int, dict[str, Any]],
    final_results: list[dict[str, Any]],
    handeye: Any,
) -> dict[str, Any] | None:
    """把共享精拍测量点、孔级目标和最终TCP画回同一张260图像。"""
    capture = next((
        item for item in batch_fine_results.values()
        if item.get("success")
        and item.get("fine_capture_overlay_path")
        and item.get("capture_tcp") is not None
        and item.get("intrinsics") is not None
    ), None)
    if capture is None:
        return None
    source_path = Path(str(capture["fine_capture_overlay_path"]))
    view = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if view is None:
        return None

    T_base_camera = camera_transform(
        np.asarray(capture["capture_tcp"], dtype=np.float64).reshape(4, 4),
        handeye.T_tcp_rgb_camera,
    )
    intrinsics = capture["intrinsics"]
    records: list[dict[str, Any]] = []

    def projected(point: Any) -> np.ndarray | None:
        if point is None:
            return None
        try:
            value = np.asarray(point, dtype=np.float64).reshape(3)
            if not np.isfinite(value).all():
                return None
            return _project_base_point_to_pixel(value, T_base_camera, intrinsics)
        except Exception:
            return None

    for result in final_results:
        if (
            result.get("status") != "completed"
            or not str(result.get("batch_fine_source", "")).startswith("batch_fine")
        ):
            continue
        hole_id = int(result["hole_id"])
        fine_point = np.asarray(result["hole_center_base_mm"], dtype=np.float64).reshape(3)
        target_point = np.asarray(result["target_point_base_mm"], dtype=np.float64).reshape(3)
        final_pose = result.get("final_tcp_pose_m_rad")
        final_tcp_point = (
            np.asarray(final_pose[:3], dtype=np.float64) * 1000.0
            if final_pose is not None and len(final_pose) >= 3
            and result.get("final_xy_motion") is not None else None
        )
        fine_px = projected(fine_point)
        target_px = projected(target_point)
        final_tcp_px = projected(final_tcp_point)

        if fine_px is not None:
            p = tuple(np.rint(fine_px).astype(int))
            cv2.drawMarker(view, p, (0, 0, 255), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
            cv2.putText(view, f"H{hole_id} FINE", (p[0] + 8, p[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1, cv2.LINE_AA)
        if target_px is not None:
            p = tuple(np.rint(target_px).astype(int))
            cv2.drawMarker(view, p, (255, 255, 0), cv2.MARKER_DIAMOND, 22, 2, cv2.LINE_AA)
            cv2.putText(view, f"H{hole_id} TARGET", (p[0] + 8, p[1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1, cv2.LINE_AA)
        if final_tcp_px is not None:
            p = tuple(np.rint(final_tcp_px).astype(int))
            cv2.drawMarker(view, p, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 22, 2, cv2.LINE_AA)
            cv2.putText(view, f"H{hole_id} TCP", (p[0] + 8, p[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
            if fine_px is not None:
                cv2.arrowedLine(
                    view, tuple(np.rint(fine_px).astype(int)), p,
                    (0, 165, 255), 2, cv2.LINE_AA, tipLength=0.15,
                )

        records.append({
            "hole_id": hole_id,
            "fine_point_base_mm": fine_point,
            "fine_point_px": fine_px,
            "target_point_base_mm": target_point,
            "target_point_px": target_px,
            "final_tcp_base_mm": final_tcp_point,
            "final_tcp_px": final_tcp_px,
            "fine_to_final_tcp_xy_mm": (
                None if final_tcp_point is None else
                np.asarray(final_tcp_point[:2] - fine_point[:2], dtype=np.float64)
            ),
        })

    if not records:
        return None
    cv2.rectangle(view, (8, 8), (700, 72), (0, 0, 0), -1)
    cv2.putText(view, "FINAL RESULT ON SHARED 260mm IMAGE", (18, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        view, "red+=FINE  cyan diamond=TARGET  yellow x=FINAL TCP  orange arrow=compensation",
        (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA,
    )
    output_path = run_dir / "batch_fine_260_final_result_overlay.png"
    if not cv2.imwrite(str(output_path), view):
        return None
    return {
        "image_path": str(output_path),
        "source_image_path": str(source_path),
        "coordinate_frame": "shared_260mm_capture_rgb",
        "legend": {
            "red_cross": "fine_3d_hole_center_reprojected",
            "cyan_diamond": "final_hole_target_fine_xy_with_coarse_z",
            "yellow_tilted_cross": "actual_final_tcp_after_xy_compensation_and_y_trim",
            "orange_arrow": "fine_center_to_actual_final_tcp",
        },
        "holes": records,
    }




def _assign_detections_to_projection(
    detections: list[dict[str, Any]],
    projected_holes_px: dict[str, np.ndarray],
    max_distance_px: float,
    expected_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """将检测框与预计孔位做带门限的一对一全局匹配。

    代价以预计像素位置为主，并可选叠加类别、框尺寸和置信度信息。
    优先使用 scipy 的匈牙利算法；现场环境没有 scipy 时回退到确定性的
    距离排序算法，仍保证一个检测框只会分配给一个孔。
    """

    hole_ids = [str(value) for value in projected_holes_px]
    if not hole_ids or not detections:
        return {}
    metadata = expected_metadata or {}
    max_distance = float(max_distance_px)
    cost = np.full((len(hole_ids), len(detections)), np.inf, dtype=np.float64)
    for hole_id, expected in projected_holes_px.items():
        expected_value = np.asarray(expected, dtype=np.float64).reshape(2)
        for detection_index, detection in enumerate(detections):
            center = np.asarray(detection["center"], dtype=np.float64).reshape(2)
            distance = float(np.linalg.norm(center - expected_value))
            if distance > max_distance:
                continue
            hole_metadata = metadata.get(str(hole_id), {})
            expected_class = hole_metadata.get("class_id")
            if expected_class is not None and int(detection.get("class_id", -1)) != int(expected_class):
                continue
            value = distance
            expected_box = hole_metadata.get("box")
            detection_box = detection.get("box")
            if expected_box is not None and detection_box is not None:
                expected_size = max(
                    float(expected_box[2]) - float(expected_box[0]),
                    float(expected_box[3]) - float(expected_box[1]),
                    1.0,
                )
                detected_size = max(
                    float(detection_box[2]) - float(detection_box[0]),
                    float(detection_box[3]) - float(detection_box[1]),
                    1.0,
                )
                value += 4.0 * abs(math.log(detected_size / expected_size))
            value -= 2.0 * float(detection.get("confidence", 0.0))
            cost[hole_ids.index(str(hole_id)), detection_index] = value

    assignments: dict[str, dict[str, Any]] = {}
    finite_pairs = np.argwhere(np.isfinite(cost))
    if finite_pairs.size == 0:
        return assignments

    try:
        from scipy.optimize import linear_sum_assignment

        safe_cost = np.where(np.isfinite(cost), cost, 1.0e9)
        row_indices, column_indices = linear_sum_assignment(safe_cost)
        selected_pairs = [
            (int(row), int(column))
            for row, column in zip(row_indices, column_indices)
            if np.isfinite(cost[row, column])
        ]
    except Exception:
        # 机器人现场可能只部署最小依赖；无 scipy 时使用纯 Python/Numpy
        # 的最大匹配、最小代价回退，不能退化为简单的逐边贪心。
        selected_pairs = _minimum_cost_maximum_assignment_without_scipy(cost)

    used_rows: set[int] = set()
    assigned: dict[str, dict[str, Any]] = {}
    used_detection_indices: set[int] = set()
    for row, detection_index in selected_pairs:
        if row in used_rows or detection_index in used_detection_indices:
            continue
        hole_id = hole_ids[row]
        expected_value = np.asarray(projected_holes_px[hole_id], dtype=np.float64).reshape(2)
        distance = float(np.linalg.norm(
            np.asarray(detections[detection_index]["center"], dtype=np.float64).reshape(2)
            - expected_value
        ))
        assigned[hole_id] = {
            "detection": detections[detection_index],
            "detection_index": detection_index,
            "distance_px": distance,
        }
        used_rows.add(row)
        used_detection_indices.add(detection_index)
    return assigned


def _minimum_cost_maximum_assignment_without_scipy(
    cost: np.ndarray,
) -> list[tuple[int, int]]:
    """无 scipy 时计算最大基数、最小代价的一对一匹配。

    通过虚拟未匹配行/列把“最大匹配数量”编码为主目标，再用方阵
    Hungarian 算法优化有限边代价。有限边代价按当前匹配代价范围设置，
    因此任何合法匹配都优先于把孔位留空，非法边永远不会被选中。
    """
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        return []
    row_count, column_count = matrix.shape
    finite_values = matrix[np.isfinite(matrix)]
    if finite_values.size == 0:
        return []

    unmatched_cost = max(1.0, float(np.max(np.abs(finite_values))) + 1.0)
    forbidden_cost = unmatched_cost * 3.0
    size = row_count + column_count
    padded = np.zeros((size, size), dtype=np.float64)
    padded[:row_count, :column_count] = np.where(
        np.isfinite(matrix), matrix, forbidden_cost,
    )
    # 真实孔位匹配虚拟列表示“不匹配”；虚拟行匹配真实检测框表示
    # “该检测框未被选中”。两者的配对代价分别是 unmatched_cost 和0。
    padded[:row_count, column_count:] = unmatched_cost

    # 方阵 Hungarian 最小化实现，兼容负代价。
    u = np.zeros(size + 1, dtype=np.float64)
    v = np.zeros(size + 1, dtype=np.float64)
    p = np.zeros(size + 1, dtype=np.int32)
    way = np.zeros(size + 1, dtype=np.int32)
    for row in range(1, size + 1):
        p[0] = row
        min_value = np.full(size + 1, np.inf, dtype=np.float64)
        used = np.zeros(size + 1, dtype=bool)
        column0 = 0
        while True:
            used[column0] = True
            row0 = int(p[column0])
            delta = math.inf
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = padded[row0 - 1, column - 1] - u[row0] - v[column]
                if current < min_value[column]:
                    min_value[column] = current
                    way[column] = column0
                if min_value[column] < delta:
                    delta = float(min_value[column])
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[int(p[column])] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            previous = int(way[column0])
            p[column0] = p[previous]
            column0 = previous
            if column0 == 0:
                break

    assigned_columns = np.full(size, -1, dtype=np.int32)
    for column in range(1, size + 1):
        if p[column] > 0:
            assigned_columns[int(p[column]) - 1] = column - 1
    return [
        (row, int(column))
        for row, column in enumerate(assigned_columns[:row_count])
        if 0 <= int(column) < column_count and np.isfinite(matrix[row, int(column)])
    ]


def _fit_batch_projected_anchor_correction(
    detections: list[dict[str, Any]],
    projected_holes_px: dict[str, np.ndarray],
    max_distance_px: float,
    min_matches: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """用首帧的宽匹配结果校正批量投影锚点的整体成像误差。

    公共观察位姿下，手眼/内参与实际到位误差通常表现为小的平移、缩放
    或剪切。先用宽门限做一次一一匹配，再拟合二维仿射映射；后续帧仍由
    原有严格跟踪门限和融合质量门限验收，避免单纯放宽匹配距离。
    """
    broad = _assign_detections_to_projection(
        detections, projected_holes_px, float(max_distance_px),
    )
    required = max(3, min(int(min_matches), len(projected_holes_px)))
    if len(broad) < required:
        raise RuntimeError(
            f"批量锚点校正有效匹配不足：{len(broad)}/{required}"
        )

    source = np.asarray(
        [np.asarray(projected_holes_px[hole_id], dtype=np.float64).reshape(2)
         for hole_id in broad],
        dtype=np.float64,
    )
    target = np.asarray(
        [np.asarray(item["detection"]["center"], dtype=np.float64).reshape(2)
         for item in broad.values()],
        dtype=np.float64,
    )
    design = np.column_stack((source, np.ones(len(source), dtype=np.float64)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if int(rank) < 3 or not np.isfinite(coefficients).all():
        raise RuntimeError("批量锚点校正仿射模型退化")

    linear = np.asarray(coefficients[:2, :], dtype=np.float64).T
    singular_values = np.linalg.svd(linear, compute_uv=False)
    translation_norm = float(np.linalg.norm(coefficients[2, :]))
    if (
        not np.isfinite(singular_values).all()
        or float(np.min(singular_values)) < 0.85
        or float(np.max(singular_values)) > 1.15
        or translation_norm > 50.0
    ):
        raise RuntimeError(
            "批量锚点校正模型偏离单位变换："
            f"scale={singular_values.tolist()}，平移={translation_norm:.2f}px"
        )

    corrected_values = np.column_stack((
        np.asarray(list(projected_holes_px.values()), dtype=np.float64),
        np.ones(len(projected_holes_px), dtype=np.float64),
    )) @ coefficients
    corrected = {
        hole_id: corrected_values[index]
        for index, hole_id in enumerate(projected_holes_px)
    }
    fitted = design @ coefficients
    residuals = np.linalg.norm(fitted - target, axis=1)
    residual_p95 = float(np.percentile(residuals, 95))
    residual_max = float(np.max(residuals))
    if residual_p95 > 5.0 or residual_max > 8.0:
        raise RuntimeError(
            "批量锚点校正残差过大："
            f"P95={residual_p95:.2f}px，最大={residual_max:.2f}px"
        )

    return corrected, {
        "enabled": True,
        "model": "affine_expected_to_detected",
        "match_count": len(broad),
        "match_hole_ids": [str(hole_id) for hole_id in broad],
        "calibration_max_distance_px": float(max_distance_px),
        "residual_p95_px": residual_p95,
        "residual_max_px": residual_max,
        "linear_singular_values": singular_values.tolist(),
        "translation_norm_px": translation_norm,
        "matrix": coefficients.tolist(),
        "raw_anchors_px": {
            str(hole_id): np.asarray(point, dtype=np.float64).tolist()
            for hole_id, point in projected_holes_px.items()
        },
        "corrected_anchors_px": {
            str(hole_id): np.asarray(point, dtype=np.float64).tolist()
            for hole_id, point in corrected.items()
        },
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
        points_camera_mm=(
            np.asarray(info["points_camera_mm"], dtype=np.float32).reshape(-1, 3)
            if info.get("points_camera_mm") is not None else None
        ),
        surface_plane_point_camera_mm=(
            np.asarray(info["local_plane_point_camera_mm"], dtype=np.float64)
            if info.get("local_plane_point_camera_mm") is not None else None
        ),
        surface_selection_policy=str(info.get("surface_selection_policy", "legacy")),
        front_surface_z_mm=(
            float(info["front_surface_z_mm"])
            if info.get("front_surface_z_mm") is not None else None
        ),
        ring_points_raw=(
            int(info["ring_points_raw"])
            if info.get("ring_points_raw") is not None else None
        ),
        surface_points_selected=(
            int(info["surface_points_selected"])
            if info.get("surface_points_selected") is not None else None
        ),
    )


def _wrap_angle_rad(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def _rotation_distance_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """返回两个姿态旋转矩阵之间的最小旋转角。"""
    relative = np.asarray(R_a, dtype=np.float64).reshape(3, 3).T @ np.asarray(R_b, dtype=np.float64).reshape(3, 3)
    cosine = np.clip((float(np.trace(relative)) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.degrees(math.acos(float(cosine))))


def _vector_change_metrics(reference: Any, measured: Any) -> dict[str, Any] | None:
    """把两个基坐标点的变化写成统一的对比指标。"""
    if reference is None or measured is None:
        return None
    try:
        reference_value = np.asarray(reference, dtype=np.float64).reshape(3)
        measured_value = np.asarray(measured, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(reference_value).all() or not np.isfinite(measured_value).all():
        return None
    delta = measured_value - reference_value
    return {
        "delta_base_mm": delta,
        "xy_norm_mm": float(np.linalg.norm(delta[:2])),
        "z_mm": float(delta[2]),
        "norm_mm": float(np.linalg.norm(delta)),
    }


def _normal_angle_deg(reference: Any, measured: Any) -> float | None:
    if reference is None or measured is None:
        return None
    try:
        reference_unit = _unit(np.asarray(reference, dtype=np.float64), "reference normal")
        measured_unit = _unit(np.asarray(measured, dtype=np.float64), "measured normal")
    except (TypeError, ValueError):
        return None
    return float(math.degrees(math.acos(np.clip(
        float(reference_unit @ measured_unit), -1.0, 1.0,
    ))))


def _planned_actual_xy_error_mm(motion: dict[str, Any] | None) -> float | None:
    if not isinstance(motion, dict):
        return None
    planned, actual = motion.get("planned_tcp_pose_m_rad"), motion.get("actual_tcp_pose_m_rad")
    if planned is None or actual is None:
        return None
    try:
        delta_m = np.asarray(actual, dtype=np.float64).reshape(6)[:2] - np.asarray(
            planned, dtype=np.float64,
        ).reshape(6)[:2]
    except (TypeError, ValueError):
        return None
    return float(np.linalg.norm(delta_m) * 1000.0)


def _effective_timing_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    """统一统计方案效率，明确排除用户等待和停稳缓冲。"""
    totals = {"motion_s": 0.0, "vision_compute_s": 0.0, "other_s": 0.0, "excluded_wait_s": 0.0}
    for event in events:
        name, elapsed_s = str(event.get("name", "")), float(event.get("elapsed_s", 0.0))
        if "wait_" in name or "settle" in name or "discard" in name:
            totals["excluded_wait_s"] += elapsed_s
        elif any(token in name for token in ("navigate", "move_to", "correction_motion", "final_motion", "return_home")):
            totals["motion_s"] += elapsed_s
        elif any(token in name for token in ("capture", "geometry", "fusion", "pixel_to_base", "tilt_center", "rgbd_pipeline")):
            totals["vision_compute_s"] += elapsed_s
        else:
            totals["other_s"] += elapsed_s
    totals = {key: round(value, 6) for key, value in totals.items()}
    totals["effective_total_s"] = round(
        totals["motion_s"] + totals["vision_compute_s"] + totals["other_s"], 6,
    )
    return totals


def _build_comparison_hole_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    """为多拍多、缓存复用和一拍多输出同一套可比诊断字段。"""
    coarse_event = result.get("coarse_cache_event") or {}
    source = str(result.get("coarse_source", "fresh_per_hole"))
    return {
        "coarse_path": {
            "source": source,
            "batch_requested": bool(result.get("batch_coarse_requested", False)),
            "batch_used": source == "batch_coarse_localization",
            "batch_fallback_reason": result.get("batch_coarse_fallback_reason"),
            "cache_reused": bool(coarse_event.get("cache_reused", False)),
            "cache_validation_failed": bool(coarse_event.get("cache_validation_failed", False)),
            "coarse_quality_recovery": result.get("coarse_quality_recovery"),
        },
        "coarse_quality": {
            "valid_frames": result.get("coarse_valid_frames"),
            "total_frames": result.get("coarse_total_frames"),
            "center_scatter_p95_px": result.get("coarse_center_scatter_p95_px"),
            "tracking_distance_p95_px": result.get("coarse_tracking_distance_p95_px"),
            "plane_rmse_mm": result.get("coarse_plane_rmse_mm"),
            "ring_points_median": result.get("coarse_ring_points_median"),
        },
        "fine_quality": {
            "status": result.get("fine_quality_status"),
            "valid_frames": result.get("valid_frames"),
            "center_scatter_p95_px": result.get("center_scatter_p95_px"),
            "ellipse_residual_median_px": result.get("ellipse_residual_median_px"),
            "ellipse_roundness_median": result.get("ellipse_roundness_median"),
            "rejected_outlier_frames": result.get("rejected_outlier_frames"),
            "recovery_attempt_count": len(result.get("fine_recovery_attempts") or []),
        },
        "geometry_change": {
            "initial_to_coarse": _vector_change_metrics(result.get("initial_center_base_mm"), result.get("coarse_center_base_mm")),
            "coarse_to_fine_naive": _vector_change_metrics(result.get("coarse_center_base_mm"), result.get("hole_center_base_naive_mm")),
            "fine_tilt_correction": _vector_change_metrics(result.get("hole_center_base_naive_mm"), result.get("hole_center_base_mm")),
            "initial_to_coarse_normal_deg": _normal_angle_deg(result.get("initial_plane_normal_base"), result.get("coarse_normal_toward_camera_base")),
            "coarse_to_final_normal_deg": _normal_angle_deg(result.get("coarse_normal_toward_camera_base"), result.get("plane_normal_toward_camera_base")),
        },
        "motion_quality": {
            "fine_height_error_mm": None if result.get("estimated_height_mm") is None else float(result["estimated_height_mm"] - 260.0),
            "final_xy_planned_actual_error_mm": _planned_actual_xy_error_mm(result.get("final_xy_motion")),
            "effective_timing": _effective_timing_summary(list((result.get("timing") or {}).get("events") or [])),
        },
    }


def _metric_distribution(values: list[Any]) -> dict[str, Any] | None:
    finite: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            finite.append(numeric)
    if not finite:
        return None
    data = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "p95": float(np.percentile(data, 95)),
        "max": float(np.max(data)),
    }


def _build_comparison_run_diagnostics(
    results: list[dict[str, Any]], report: dict[str, Any], cfg: TwoStageConfig,
) -> dict[str, Any]:
    """输出可直接用于三种方案横向比较的整轮汇总。"""
    completed = [item for item in results if item.get("status") == "completed"]
    sources: dict[str, int] = {}
    batch_fallbacks: list[dict[str, Any]] = []
    for item in results:
        source = str(item.get("coarse_source", "unknown"))
        sources[source] = sources.get(source, 0) + 1
        reason = item.get("batch_coarse_fallback_reason")
        if reason:
            batch_fallbacks.append({"hole_id": item.get("hole_id"), "reason": reason})

    def metric(path: str) -> list[Any]:
        keys = path.split(".")
        output: list[Any] = []
        for item in completed:
            value: Any = item.get("comparison_diagnostics", {})
            for key in keys:
                value = value.get(key) if isinstance(value, dict) else None
            output.append(value)
        return output

    batch_stage = (report.get("stages") or {}).get("batch_coarse_results") or {}
    # 兼容单组旧报告的 capture 字段；多视野批量报告以 groups/captures
    # 保存每个观察位，不能只保留最后一组。
    batch_capture = batch_stage.get("capture") or {}
    batch_captures = batch_stage.get("captures") or []
    if not batch_capture and len(batch_captures) == 1:
        batch_capture = batch_captures[0].get("capture") or {}
    return {
        "schema_version": 1,
        "method_configuration": {
            "batch_coarse_requested": bool(cfg.batch_coarse_localization),
            "persistent_cache_requested": bool(report.get("reuse_persistent_coarse_cache", False)),
            "session_cache_requested": bool(report.get("reuse_coarse_cache", False)),
        },
        "outcome": {
            "selected_holes": len(results),
            "completed_holes": len(completed),
            "deferred_holes": len(results) - len(completed),
            "coarse_source_counts": sources,
            "batch_fallbacks": batch_fallbacks,
        },
        "batch_coarse": {
            "success_count": batch_stage.get("success_count"),
            "total_count": batch_stage.get("total_count"),
            "failed_holes": batch_stage.get("failed_holes", []),
            "group_count": batch_stage.get("group_count"),
            "groups": batch_stage.get("groups", []),
            "captures": batch_captures,
            "anchor_correction": batch_capture.get("anchor_correction"),
        },
        "quality_distributions": {
            "coarse_center_scatter_p95_px": _metric_distribution(metric("coarse_quality.center_scatter_p95_px")),
            "coarse_tracking_distance_p95_px": _metric_distribution(metric("coarse_quality.tracking_distance_p95_px")),
            "coarse_plane_rmse_mm": _metric_distribution(metric("coarse_quality.plane_rmse_mm")),
            "fine_center_scatter_p95_px": _metric_distribution(metric("fine_quality.center_scatter_p95_px")),
            "fine_ellipse_residual_median_px": _metric_distribution(metric("fine_quality.ellipse_residual_median_px")),
            "initial_to_coarse_norm_mm": _metric_distribution(metric("geometry_change.initial_to_coarse.norm_mm")),
            "coarse_to_fine_norm_mm": _metric_distribution(metric("geometry_change.coarse_to_fine_naive.norm_mm")),
            "tilt_correction_norm_mm": _metric_distribution(metric("geometry_change.fine_tilt_correction.norm_mm")),
            "final_xy_planned_actual_error_mm": _metric_distribution(metric("motion_quality.final_xy_planned_actual_error_mm")),
        },
        "effective_timing": _effective_timing_summary(
            list((report.get("timing") or {}).get("events") or []),
        ),
    }


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
    enforce_tracking_gate: bool = True,
) -> dict[str, Any]:
    """将某个孔在当前居中相机位采集的结果写回该孔记录。"""
    hole_id = int(hole["hole_id"])
    summary = _fuse_coarse(
        observations,
        cfg,
        min_valid_frames=min_valid_frames,
        max_center_scatter_p95_px=cfg.max_coarse_center_scatter_p95_px,
        max_tracking_distance_p95_px=(
            cfg.max_coarse_tracking_distance_p95_px
            if enforce_tracking_gate else None
        ),
    )
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
        "coarse_tracking_distance_p95_px": summary.get("tracking_distance_p95_px"),
        "coarse_ring_points_median": int(np.median([
            item.plane.ring_points for item in observations
            if item.error is None and item.plane is not None
        ])),
        "coarse_surface_model": summary["surface_model"],
        "coarse_surface_selection_policy": summary.get("surface_selection_policy"),
        "coarse_front_surface_z_mm": summary.get("front_surface_z_median_mm"),
        "coarse_ring_points_raw_median": summary.get("ring_points_raw_median"),
        "coarse_surface_points_selected_median": summary.get(
            "surface_points_selected_median"
        ),
        "coarse_sphere_center_camera_mm": summary.get("sphere_center_camera_mm"),
        "coarse_sphere_radius_mm": summary.get("sphere_radius_mm"),
    })
    if not enforce_tracking_gate:
        summary["quality_recovery"] = "tracking_degraded_but_geometry_valid"
        hole["coarse_quality_recovery"] = summary["quality_recovery"]
    return summary


def _cache_intrinsics_dict(intrinsics: Any) -> dict[str, Any]:
    if hasattr(intrinsics, "as_dict") and callable(intrinsics.as_dict):
        return dict(intrinsics.as_dict())
    return {
        "width": int(intrinsics.width), "height": int(intrinsics.height),
        "fx": float(intrinsics.fx), "fy": float(intrinsics.fy),
        "cx": float(intrinsics.cx), "cy": float(intrinsics.cy),
        "distortion": [float(value) for value in getattr(intrinsics, "distortion", ())],
    }


def _cache_measurements_from_observations(
    observations: list[Observation],
) -> list[dict[str, Any]]:
    measurements: list[dict[str, Any]] = []
    for item in observations:
        measurement: dict[str, Any] = {
            "frame_index": int(item.frame_index),
            "timestamp_ns": item.timestamp_ns,
            "center_px": np.asarray(item.center_px, dtype=np.float64),
            "tracking_distance_px": item.tracking_distance_px,
            "error": item.error,
        }
        if item.plane is not None:
            measurement.update({
                "plane_point_camera_mm": np.asarray(item.plane.point_camera_mm, dtype=np.float64),
                "surface_plane_point_camera_mm": (
                    np.asarray(item.plane.surface_plane_point_camera_mm, dtype=np.float64)
                    if item.plane.surface_plane_point_camera_mm is not None
                    else np.asarray(item.plane.point_camera_mm, dtype=np.float64)
                ),
                "normal_camera": np.asarray(item.plane.normal_camera, dtype=np.float64),
                "plane_rmse_mm": float(item.plane.rmse_mm),
                "ring_points": int(item.plane.ring_points),
                "surface_model": item.plane.surface_model,
                "surface_selection_policy": item.plane.surface_selection_policy,
                "front_surface_z_mm": item.plane.front_surface_z_mm,
                "ring_points_raw": item.plane.ring_points_raw,
                "surface_points_selected": item.plane.surface_points_selected,
            })
        measurements.append(measurement)
    return measurements


def _cache_cloud_from_coarse_observations(
    hole_id: int,
    observations: list[Observation],
    *,
    intrinsics: Any,
    T_base_camera: np.ndarray,
    point_camera_mm: Any,
    plane_point_camera_mm: Any,
    normal_camera: Any,
    plane_rmse_mm: float | None,
    center_scatter_p95_px: float | None,
    source_label: str,
) -> CacheCloud | None:
    """把本次实时观测（含每帧局部点云）转换成可视化工具所需的CacheCloud。

    只使用同时具备平面估计和points_camera_mm的有效帧；没有任何有效帧时返回
    None，交给调用方回退到缓存重投影点云或直接跳过保存。
    """
    valid = [
        item for item in observations
        if item.error is None
        and item.plane is not None
        and item.plane.points_camera_mm is not None
        and len(item.plane.points_camera_mm) > 0
    ]
    if not valid:
        return None
    return CacheCloud(
        hole_id=int(hole_id),
        points_camera_mm_by_frame=tuple(
            np.asarray(item.plane.points_camera_mm, dtype=np.float32).reshape(-1, 3)
            for item in valid
        ),
        frame_indices=np.asarray([item.frame_index for item in valid], dtype=np.int64),
        frame_centers_px=np.asarray(
            [item.center_px for item in valid], dtype=np.float64
        ).reshape(-1, 2),
        frame_plane_rmse_mm=np.asarray([item.plane.rmse_mm for item in valid], dtype=np.float64),
        intrinsics=_cache_intrinsics_dict(intrinsics),
        T_base_camera_build=np.asarray(T_base_camera, dtype=np.float64).reshape(4, 4),
        point_camera_mm=np.asarray(point_camera_mm, dtype=np.float64).reshape(3),
        plane_point_camera_mm=np.asarray(plane_point_camera_mm, dtype=np.float64).reshape(3),
        normal_camera=_unit(
            np.asarray(normal_camera, dtype=np.float64), f"hole {hole_id} pointcloud image normal"
        ),
        point_base_mm=None,
        plane_point_base_mm=None,
        normal_base=None,
        plane_rmse_mm=None if plane_rmse_mm is None else float(plane_rmse_mm),
        center_scatter_p95_px=(
            None if center_scatter_p95_px is None else float(center_scatter_p95_px)
        ),
        source_path=Path(f"live_observations:{source_label}"),
        cache_scope=source_label,
    )


def _cache_cloud_from_cached_entry_reprojected(
    cache_entry: CoarseCacheEntry,
    hole_id: int,
    *,
    T_base_camera: np.ndarray,
    point_camera_mm: Any,
    plane_point_camera_mm: Any,
    normal_camera: Any,
    source_label: str,
) -> CacheCloud:
    """当现场观测缺少点云（例如未通过质量门）时，把缓存点重投影到当前相机位。"""
    points_by_frame = transform_cached_points_to_camera(cache_entry, T_base_camera)
    return CacheCloud(
        hole_id=int(hole_id),
        points_camera_mm_by_frame=points_by_frame,
        frame_indices=np.asarray(cache_entry.frame_indices, dtype=np.int64),
        frame_centers_px=np.asarray(cache_entry.frame_centers_px, dtype=np.float64),
        frame_plane_rmse_mm=np.asarray(cache_entry.frame_plane_rmse_mm, dtype=np.float64),
        intrinsics=dict(cache_entry.intrinsics),
        T_base_camera_build=np.asarray(T_base_camera, dtype=np.float64).reshape(4, 4),
        point_camera_mm=np.asarray(point_camera_mm, dtype=np.float64).reshape(3),
        plane_point_camera_mm=np.asarray(plane_point_camera_mm, dtype=np.float64).reshape(3),
        normal_camera=_unit(
            np.asarray(normal_camera, dtype=np.float64), f"hole {hole_id} cached pointcloud normal"
        ),
        point_base_mm=None,
        plane_point_base_mm=None,
        normal_base=None,
        plane_rmse_mm=float(cache_entry.plane_rmse_mm),
        center_scatter_p95_px=float(cache_entry.center_scatter_p95_px),
        source_path=Path(f"cache_reprojected:{source_label}"),
        cache_scope=source_label,
    )


def _save_coarse_pointcloud_image(
    hole: dict[str, Any],
    hole_id: int,
    run_dir: Path,
    *,
    observations: list[Observation] | None,
    cache_entry: CoarseCacheEntry | None,
    T_base_camera: np.ndarray,
    intrinsics: Any,
    rgb_path: Path | None,
    source_label: str,
) -> Path | None:
    """保存单孔340mm粗定位点云PNG，供后续人工复核；任何异常都只警告不中断流程。"""
    anchor = hole.get("coarse_center_camera_mm")
    normal = hole.get("coarse_normal_camera")
    if anchor is None or normal is None:
        return None
    plane_anchor = hole.get("coarse_plane_point_camera_mm")
    if plane_anchor is None:
        plane_anchor = anchor
    cloud: CacheCloud | None = None
    try:
        if observations is not None:
            cloud = _cache_cloud_from_coarse_observations(
                hole_id, observations,
                intrinsics=intrinsics, T_base_camera=T_base_camera,
                point_camera_mm=anchor, plane_point_camera_mm=plane_anchor,
                normal_camera=normal,
                plane_rmse_mm=hole.get("coarse_plane_rmse_mm"),
                center_scatter_p95_px=hole.get("coarse_center_scatter_p95_px"),
                source_label=source_label,
            )
        if cloud is None and cache_entry is not None:
            cloud = _cache_cloud_from_cached_entry_reprojected(
                cache_entry, hole_id,
                T_base_camera=T_base_camera,
                point_camera_mm=anchor, plane_point_camera_mm=plane_anchor,
                normal_camera=normal,
                source_label=f"{source_label}_cache_reprojected",
            )
    except Exception as exc:
        print(f"[COARSE_POINTCLOUD_IMAGE_WARNING] hole={hole_id} reason={exc}", flush=True)
        return None
    if cloud is None:
        return None
    output_path = run_dir / f"hole_{hole_id:02d}_coarse_pointcloud.png"
    try:
        return render_cache_cloud(
            cloud,
            rgb_path=(rgb_path if rgb_path is not None and rgb_path.is_file() else None),
            output_path=output_path,
            show=False,
        )
    except Exception as exc:
        print(f"[COARSE_POINTCLOUD_IMAGE_WARNING] hole={hole_id} reason={exc}", flush=True)
        return None


def _cache_entry_from_observations(
    hole_id: int,
    observations: list[Observation],
    *,
    T_base_camera: np.ndarray,
    T_tcp_camera: np.ndarray,
    tcp_pose_m_rad: list[float],
    camera_serial: str,
    handeye_path: str,
    intrinsics: Any,
    cfg: TwoStageConfig,
    min_valid_frames: int,
    cache_source: str = "unknown",
) -> CoarseCacheEntry:
    valid = [
        item for item in observations
        if item.error is None and item.plane is not None and item.plane.points_camera_mm is not None
    ]
    summary = _fuse_coarse(observations, cfg, min_valid_frames=min_valid_frames)
    if len(valid) < int(min_valid_frames):
        raise RuntimeError(f"孔{hole_id}缓存有效帧不足：{len(valid)}/{min_valid_frames}")
    T_base_camera = np.asarray(T_base_camera, dtype=np.float64).reshape(4, 4)
    R_base_camera = T_base_camera[:3, :3]
    t_base_camera = T_base_camera[:3, 3]
    point_camera = np.asarray(summary["plane_point_camera_mm"], dtype=np.float64)
    plane_point_camera_value = summary.get("surface_plane_point_camera_mm")
    plane_point_camera = (
        point_camera if plane_point_camera_value is None
        else np.asarray(plane_point_camera_value, dtype=np.float64)
    )
    normal_camera = _unit(np.asarray(summary["plane_normal_camera"], dtype=np.float64), f"缓存孔{hole_id}法向")
    point_base = R_base_camera @ point_camera + t_base_camera
    plane_point_base = R_base_camera @ plane_point_camera + t_base_camera
    normal_base = _unit(R_base_camera @ normal_camera, f"缓存孔{hole_id}基坐标法向")
    if float(normal_base @ (t_base_camera - point_base)) < 0.0:
        normal_base = -normal_base
    frame_indices = np.asarray([item.frame_index for item in valid], dtype=np.int64)
    frame_centers_px = np.asarray([item.center_px for item in valid], dtype=np.float64)
    frame_plane_rmse_mm = np.asarray([item.plane.rmse_mm for item in valid], dtype=np.float64)
    points_by_frame = tuple(
        np.asarray(item.plane.points_camera_mm, dtype=np.float32).reshape(-1, 3)
        for item in valid
    )
    return CoarseCacheEntry(
        hole_id=int(hole_id),
        T_base_camera_build=T_base_camera,
        T_tcp_camera=np.asarray(T_tcp_camera, dtype=np.float64),
        tcp_pose_m_rad=list(tcp_pose_m_rad),
        camera_serial=str(camera_serial),
        handeye_path=str(handeye_path),
        intrinsics=_cache_intrinsics_dict(intrinsics),
        center_px=np.asarray(summary["center_px"], dtype=np.float64),
        point_camera_mm=point_camera,
        plane_point_camera_mm=plane_point_camera,
        normal_camera=normal_camera,
        point_base_mm=point_base,
        plane_point_base_mm=plane_point_base,
        normal_base=normal_base,
        plane_rmse_mm=float(summary["plane_rmse_median_mm"]),
        valid_frames=len(valid),
        total_frames=len(observations),
        center_scatter_p95_px=float(summary["center_scatter_p95_px"]),
        ring_points_median=float(np.median([item.plane.ring_points for item in valid])),
        surface_model=str(summary["surface_model"]),
        frame_indices=frame_indices,
        frame_centers_px=frame_centers_px,
        frame_plane_rmse_mm=frame_plane_rmse_mm,
        points_camera_mm_by_frame=points_by_frame,
        cache_source=str(cache_source),
    )


def _apply_cached_geometry_to_hole(
    hole: dict[str, Any], entry: CoarseCacheEntry, validation: Any,
    *, source: str = "current_run_initial_cache",
    persistent_hole_id: int | None = None,
) -> None:
    """复用缓存的XY/法向，并用现场深度结果替换基坐标Z。"""
    current_point_base = replace_base_z(entry.point_base_mm, validation.current_point_base_mm)
    current_plane_base = replace_base_z(
        entry.plane_point_base_mm, validation.current_plane_point_base_mm,
    )
    hole.update({
        "coarse_center_px": validation.current_center_px,
        "coarse_center_camera_mm": validation.current_point_camera_mm,
        "coarse_center_base_mm": current_point_base,
        "coarse_plane_point_camera_mm": validation.current_plane_point_camera_mm,
        "coarse_plane_point_base_mm": current_plane_base,
        "coarse_normal_camera": entry.normal_camera,
        "coarse_normal_toward_camera_base": entry.normal_base,
        "coarse_plane_rmse_mm": validation.max_plane_rmse_mm,
        "coarse_valid_frames": validation.valid_frames,
        "coarse_total_frames": validation.total_frames,
        "coarse_center_scatter_p95_px": validation.center_scatter_p95_px,
        "coarse_ring_points_median": entry.ring_points_median,
        "coarse_surface_model": f"cached_{COARSE_SURFACE_MODEL}_with_live_z",
        "coarse_surface_selection_policy": COARSE_SURFACE_SELECTION_POLICY,
        "coarse_front_surface_z_mm": None,
        "coarse_ring_points_raw_median": None,
        "coarse_surface_points_selected_median": None,
        "coarse_sphere_center_camera_mm": None,
        "coarse_sphere_radius_mm": None,
        "coarse_cache_source": str(source),
        "coarse_cache_entry_created_at": entry.created_at,
        "coarse_cache_persistent_hole_id": (
            None if persistent_hole_id is None else int(persistent_hole_id)
        ),
    })


def _reuse_initial_pointcloud_geometry_for_batch_fine(
    hole: dict[str, Any],
) -> bool:
    """把本轮选孔RGB-D帧的逐孔点云几何直接提升为粗几何。

    该数据与用户点击选择来自同一帧，不依赖跨运行缓存的base坐标匹配；
    批量精定位模式下以它覆盖历史缓存几何，保证所有已选孔来源一致。
    """
    required = {
        "center_base": hole.get("initial_center_base_mm"),
        "center_camera": hole.get("initial_point_camera_mm"),
        "plane_normal_base": hole.get(
            "initial_shared_plane_normal_base",
            hole.get("initial_plane_normal_base"),
        ),
        "plane_normal_camera": hole.get(
            "initial_shared_plane_normal_camera",
            hole.get("initial_plane_normal_camera"),
        ),
    }
    arrays: dict[str, np.ndarray] = {}
    for name, value in required.items():
        if value is None:
            return False
        array = np.asarray(value, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(array)):
            return False
        arrays[name] = array.copy()

    plane_point_base = np.asarray(
        hole.get(
            "initial_shared_plane_point_base_mm",
            hole.get("initial_plane_point_base_mm", arrays["center_base"]),
        ),
        dtype=np.float64,
    ).reshape(3)
    plane_point_camera = np.asarray(
        hole.get(
            "initial_shared_plane_point_camera_mm",
            hole.get("initial_plane_point_camera_mm", arrays["center_camera"]),
        ),
        dtype=np.float64,
    ).reshape(3)
    if not np.all(np.isfinite(plane_point_base)) or not np.all(np.isfinite(plane_point_camera)):
        return False

    hole.update({
        "coarse_center_px": hole.get("initial_center_px"),
        "coarse_center_camera_mm": arrays["center_camera"],
        "coarse_center_base_mm": arrays["center_base"],
        "coarse_plane_point_camera_mm": plane_point_camera.copy(),
        "coarse_plane_point_base_mm": plane_point_base.copy(),
        "coarse_normal_camera": arrays["plane_normal_camera"],
        "coarse_normal_toward_camera_base": arrays["plane_normal_base"],
        "coarse_plane_rmse_mm": hole.get(
            "initial_shared_plane_rmse_mm", hole.get("initial_plane_rmse_mm")
        ),
        "coarse_valid_frames": 1,
        "coarse_total_frames": 1,
        "coarse_center_scatter_p95_px": 0.0,
        "coarse_tracking_distance_p95_px": 0.0,
        "coarse_ring_points_median": hole.get(
            "initial_shared_ring_points", hole.get("initial_ring_points")
        ),
        "coarse_surface_model": hole.get(
            "initial_shared_surface_model", hole.get("initial_surface_model")
        ),
        "coarse_surface_selection_policy": hole.get(
            "initial_shared_surface_selection_policy",
            hole.get("initial_surface_selection_policy"),
        ),
        "coarse_front_surface_z_mm": hole.get(
            "initial_shared_front_surface_z_mm", hole.get("initial_front_surface_z_mm")
        ),
        "coarse_ring_points_raw_median": hole.get("initial_ring_points_raw"),
        "coarse_surface_points_selected_median": hole.get(
            "initial_surface_points_selected"
        ),
        "coarse_sphere_center_camera_mm": None,
        "coarse_sphere_radius_mm": None,
        "coarse_source": "initial_selection_shared_pointcloud_reuse",
        "coarse_geometry_type": "shared_workpiece_plane_with_per_hole_center_ray",
        "initial_pointcloud_reused_for_batch_fine": True,
        "coarse_captures": [{
            "capture_index": 0,
            "mode": "initial_selection_pointcloud_reuse",
            "source_frame": "current_cycle_initial_rgbd_selection",
        }],
    })
    return True


def _validate_coarse_cache_at_current_pose(
    hole: dict[str, Any], entry: CoarseCacheEntry, *,
    pipeline: Any, align: Any, chain: Any, model: Any, confidence: float,
    run_dir: Path, cfg: TwoStageConfig, gates: CacheValidationGates,
    current_T_base_camera: np.ndarray, intrinsics: Any,
) -> tuple[Any, list[Observation], Any]:
    validation_cfg = replace(
        cfg,
        coarse_frames=int(gates.validation_frames),
        min_coarse_valid=int(gates.min_valid_frames),
        coarse_settle_frames=max(
            int(cfg.coarse_settle_frames),
            int(cfg.cache_validation_settle_discard_frames),
        ),
        coarse_max_attempt_multiplier=1,
        max_coarse_center_scatter_p95_px=float(gates.max_center_scatter_p95_px),
        max_plane_rmse_mm=float(gates.max_plane_rmse_mm),
    )
    expected_anchor = _project_base_point_to_pixel(
        entry.point_base_mm, current_T_base_camera, intrinsics,
    )
    observations, _ = _capture_coarse_burst(
        pipeline, align, chain, model, confidence,
        hole["initial_detection"], validation_cfg, run_dir,
        f"hole_{int(hole['hole_id']):02d}_coarse_cache_verify",
        initial_anchor_px=expected_anchor,
        tracking_tolerance_px=float(gates.max_tracking_distance_px),
        # 只用缓存投影作为首帧身份锚点；后续帧跟踪上一帧检测，
        # 避免固定投影偏差污染跨帧稳定性统计。
        lock_anchor=False,
        stop_when_stable=False,
        include_points=True,
    )
    validation = validate_cache_entry(
        entry,
        _cache_measurements_from_observations(observations),
        current_T_base_camera=current_T_base_camera,
        intrinsics=intrinsics,
        gates=gates,
    )
    return validation, observations, expected_anchor


def _upsert_persistent_cache_entry(
    entry: CoarseCacheEntry,
    *,
    current_hole_id: int,
    persistent_entries: dict[int, CoarseCacheEntry],
    persistent_source_ids: dict[int, int],
    max_match_distance_mm: float = 30.0,
    min_match_margin_mm: float = 5.0,
) -> int:
    """把当前成功粗定位结果写回 base 坐标持久化缓存。"""

    hole_id = int(current_hole_id)
    source_id = persistent_source_ids.get(hole_id)
    if source_id is None and persistent_entries:
        current_point = np.asarray(entry.point_base_mm, dtype=np.float64).reshape(3)
        used_source_ids = {
            int(value) for key, value in persistent_source_ids.items()
            if int(key) != hole_id
        }
        candidates = sorted(
            (
                float(np.linalg.norm(
                    current_point[:2]
                    - np.asarray(candidate.point_base_mm, dtype=np.float64).reshape(3)[:2],
                )),
                int(candidate_id),
            )
            for candidate_id, candidate in persistent_entries.items()
            if int(candidate_id) not in used_source_ids
        )
        if candidates and candidates[0][0] <= float(max_match_distance_mm):
            if (
                len(candidates) == 1
                or candidates[1][0] - candidates[0][0] >= float(min_match_margin_mm)
            ):
                source_id = candidates[0][1]
    if source_id is None:
        source_id = max([int(value) for value in persistent_entries] or [0]) + 1
    persistent_entries[int(source_id)] = rekey_cache_entry(entry, int(source_id))
    persistent_source_ids[hole_id] = int(source_id)
    return int(source_id)


def _load_persistent_coarse_cache_for_run(
    *,
    run_dir: Path,
    camera_serial: str,
    handeye: Any,
    handeye_path: str,
    intrinsics: Any,
    persistent_cache_dir: Path,
    required_surface_model: str | None = None,
) -> tuple[dict[int, CoarseCacheEntry], dict[int, str], dict[str, Any]]:
    """读取固定 base 坐标缓存；首次升级时兼容导入最近一次运行缓存。"""

    errors: dict[int, str] = {}
    audit: dict[str, Any] = {
        "requested_dir": str(persistent_cache_dir),
        "source": None,
        "bootstrap_source": None,
        "loaded_count": 0,
        "required_surface_model": required_surface_model,
        "compatibility_rejected": {},
        "load_errors": {},
    }
    entries: dict[int, CoarseCacheEntry] = {}
    if (persistent_cache_dir / "manifest.json").is_file():
        try:
            entries = load_persistent_cache_entries(persistent_cache_dir, errors=errors)
            audit["source"] = str(persistent_cache_dir)
        except Exception as exc:
            audit["load_errors"]["manifest"] = f"{type(exc).__name__}:{exc}"
    else:
        # 旧版本只把缓存写在每次运行目录。第一次启用持久化时自动把最近
        # 一个可读的旧缓存作为种子，避免用户必须重新做一遍完整粗定位。
        candidates = sorted(
            (
                path for path in RUNS_DIR.glob("two-stage-*/coarse_cache/manifest.json")
                if path.parent.parent.resolve() != run_dir.resolve()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for manifest_path in candidates:
            try:
                entries = load_cache_entries(manifest_path.parent)
            except Exception as exc:
                audit["load_errors"][str(manifest_path)] = f"{type(exc).__name__}:{exc}"
                continue
            if entries:
                audit["source"] = str(manifest_path.parent)
                audit["bootstrap_source"] = str(manifest_path.parent)
                break

    compatible: dict[int, CoarseCacheEntry] = {}
    for persistent_id, entry in entries.items():
        reasons: list[str] = []
        if (
            required_surface_model is not None
            and str(entry.surface_model) != str(required_surface_model)
        ):
            reasons.append("surface_selection_policy_mismatch")
        reasons.extend(cache_entry_compatibility_reasons(
            entry,
            camera_serial=camera_serial,
            T_tcp_camera=handeye.T_tcp_rgb_camera,
            intrinsics=intrinsics,
            handeye_path=handeye_path,
        ))
        if reasons:
            audit["compatibility_rejected"][str(int(persistent_id))] = reasons
        else:
            compatible[int(persistent_id)] = entry
    audit["loaded_count"] = len(compatible)
    audit["load_errors"].update({str(key): value for key, value in errors.items()})
    return compatible, errors, audit


def _fuse_coarse(
    observations: list[Observation], cfg: TwoStageConfig,
    min_valid_frames: int | None = None,
    max_center_scatter_p95_px: float | None = None,
    max_tracking_distance_p95_px: float | None = None,
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
    center_scatter_p95_px = float(np.percentile(scatter, 95))
    tracking_distances = np.asarray(
        [item.tracking_distance_px for item in valid if item.tracking_distance_px is not None],
        dtype=np.float64,
    )
    tracking_distance_p95_px = (
        float(np.percentile(tracking_distances, 95))
        if tracking_distances.size else None
    )
    surface_plane_points = [
        item.plane.surface_plane_point_camera_mm
        for item in valid
        if item.plane.surface_plane_point_camera_mm is not None
    ]
    sphere_centers = [item.plane.sphere_center_camera_mm for item in valid if item.plane.sphere_center_camera_mm is not None]
    sphere_radii = [item.plane.sphere_radius_mm for item in valid if item.plane.sphere_radius_mm is not None]
    front_surface_zs = [
        item.plane.front_surface_z_mm for item in valid
        if item.plane.front_surface_z_mm is not None
    ]
    ring_points_raw = [
        item.plane.ring_points_raw for item in valid
        if item.plane.ring_points_raw is not None
    ]
    surface_points_selected = [
        item.plane.surface_points_selected for item in valid
        if item.plane.surface_points_selected is not None
    ]
    summary = {
        "valid_frames": len(valid), "total_frames": len(observations), "center_px": center,
        "center_scatter_p95_px": center_scatter_p95_px,
        "tracking_distance_p95_px": tracking_distance_p95_px,
        "plane_point_camera_mm": plane_points, "plane_normal_camera": normals,
        "surface_plane_point_camera_mm": (
            _fuse_vectors(surface_plane_points, "surface plane points")
            if surface_plane_points else None
        ),
        "plane_rmse_median_mm": float(np.median([item.plane.rmse_mm for item in valid])),
        "ring_points_median": int(np.median([
            item.plane.ring_points for item in valid
        ])),
        "surface_model": valid[0].plane.surface_model,
        "surface_selection_policy": valid[0].plane.surface_selection_policy,
        "front_surface_z_median_mm": (
            float(np.median(front_surface_zs)) if front_surface_zs else None
        ),
        "ring_points_raw_median": (
            int(np.median(ring_points_raw)) if ring_points_raw else None
        ),
        "surface_points_selected_median": (
            int(np.median(surface_points_selected)) if surface_points_selected else None
        ),
        "sphere_center_camera_mm": _fuse_vectors(sphere_centers, "sphere centers") if sphere_centers else None,
        "sphere_radius_mm": float(np.median(sphere_radii)) if sphere_radii else None,
    }
    if (
        max_center_scatter_p95_px is not None
        and center_scatter_p95_px > float(max_center_scatter_p95_px)
    ):
        raise RuntimeError(
            "粗定位中心稳定性失败："
            f"P95={center_scatter_p95_px:.3f}px > "
            f"{float(max_center_scatter_p95_px):.3f}px"
        )
    if (
        max_tracking_distance_p95_px is not None
        and tracking_distance_p95_px is not None
        and tracking_distance_p95_px > float(max_tracking_distance_p95_px)
    ):
        raise RuntimeError(
            "粗定位跟踪距离失败："
            f"P95={tracking_distance_p95_px:.3f}px > "
            f"{float(max_tracking_distance_p95_px):.3f}px"
        )
    return summary


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
    center_source_counts: dict[str, int] = {}
    for item in valid:
        source = str(item.center_source or "unknown")
        center_source_counts[source] = center_source_counts.get(source, 0) + 1
    relaxed_yolo_frames = sum(
        "strict_ellipse_rejected" in str(item.quality_note or "") for item in valid
    )
    strict_ellipse_frames = sum(
        str(item.center_source or "unknown")
        in {"ellipse", "hough_circle", "geometric_circle"}
        for item in valid
    )
    center_source = (
        next(iter(center_source_counts))
        if len(center_source_counts) == 1 else "mixed"
    )
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
        "yolo_frames": int(center_source_counts.get("yolo", 0)),
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
            "error": item.error, "tracking_distance_px": item.tracking_distance_px,
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
                "surface_selection_policy": item.plane.surface_selection_policy,
                "front_surface_z_mm": item.plane.front_surface_z_mm,
                "ring_points_raw": item.plane.ring_points_raw,
                "surface_points_selected": item.plane.surface_points_selected,
                "pointcloud_points": int(
                    0 if item.plane.points_camera_mm is None else len(item.plane.points_camera_mm)
                ),
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


def _move_to_shared_observation_pose(
    label: str,
    current_tcp: np.ndarray,
    target: np.ndarray,
    args: Any,
    motion_session: Any,
    pose_session: Any,
    *,
    target_height_mm: float,
    descent_profile: str,
) -> np.ndarray:
    """共享粗/精定位共同观察位的安全移动路径。

    共享模式也必须先做纯基坐标Z抬升，再允许XY和姿态变化；横移完成后，
    最后至少保留 ``SHARED_OBSERVATION_MIN_DESCENT_MM`` 的纯基坐标Z下降段。
    抬升量至少为 ``SHARED_OBSERVATION_MIN_LIFT_MM``，同时继续使用原有的
    60 mm安全高度余量，避免把“至少10 mm”误解成只抬10 mm就横移。正式
    采集前的steady检查和相机残帧清理仍由调用方保留。
    """
    actual = np.asarray(current_tcp, dtype=np.float64).reshape(4, 4).copy()
    desired = np.asarray(target, dtype=np.float64).reshape(4, 4).copy()
    position_error = float(np.linalg.norm(actual[:3, 3] - desired[:3, 3]))
    rotation_error = _rotation_distance_deg(actual[:3, :3], desired[:3, :3])
    if position_error <= 0.5 and rotation_error <= 0.5:
        return actual

    safe_z = max(float(actual[2, 3]), float(desired[2, 3])) + THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM
    lift_z = max(
        float(actual[2, 3]) + SHARED_OBSERVATION_MIN_LIFT_MM,
        safe_z,
    )
    lift = actual.copy()
    lift[2, 3] = lift_z
    actual = _confirm_and_move_line(
        f"{label}：先纯Z抬升{SHARED_OBSERVATION_MIN_LIFT_MM:.0f}mm以上",
        actual,
        lift,
        args,
        motion_session,
        pose_session,
        "仅修改基坐标Z；XY和姿态保持不变",
        require_confirmation=False,
        motion_profile="transit",
    )

    high_target = desired.copy()
    high_target[2, 3] = lift_z
    actual = _confirm_and_move_line(
        f"{label}：安全高度横移并调整共同视野姿态",
        actual,
        high_target,
        args,
        motion_session,
        pose_session,
        "已完成纯Z抬升；此段不下降到观察高度",
        require_confirmation=False,
        motion_profile="transit",
    )
    descent_clearance_mm = float(lift_z - desired[2, 3])
    if descent_clearance_mm < SHARED_OBSERVATION_MIN_DESCENT_MM:
        raise RuntimeError(
            f"{label}纯Z下降安全余量不足：{descent_clearance_mm:.3f}mm < "
            f"{SHARED_OBSERVATION_MIN_DESCENT_MM:.3f}mm"
        )
    descent_guard = desired.copy()
    descent_guard[2, 3] = float(desired[2, 3]) + SHARED_OBSERVATION_MIN_DESCENT_MM
    actual = _confirm_and_move_line(
        f"{label}：纯Z下降到目标上方{SHARED_OBSERVATION_MIN_DESCENT_MM:.0f}mm安全位",
        actual,
        descent_guard,
        args,
        motion_session,
        pose_session,
        f"保持XY和姿态不变；先纯Z下降到最终观察位上方"
        f"{SHARED_OBSERVATION_MIN_DESCENT_MM:.0f}mm，禁止斜向接近",
        require_confirmation=False,
        motion_profile=descent_profile,
    )
    return _confirm_and_move_line(
        f"{label}：再纯Z下降{SHARED_OBSERVATION_MIN_DESCENT_MM:.0f}mm到"
        f"{float(target_height_mm):.0f}mm共同观察位",
        actual,
        desired,
        args,
        motion_session,
        pose_session,
        f"共同视野孔集合已冻结；保持XY和姿态不变，最后"
        f"纯Z下降{SHARED_OBSERVATION_MIN_DESCENT_MM:.0f}mm到指定RGB相机高度"
        f"{float(target_height_mm):.0f}mm",
        require_confirmation=False,
        motion_profile=descent_profile,
    )


def _move_to_shared_coarse_pose(
    group_index: int,
    group_count: int,
    current_tcp: np.ndarray,
    target: np.ndarray,
    args: Any,
    motion_session: Any,
    pose_session: Any,
) -> np.ndarray:
    """共享粗定位专用的共同340 mm观察位移动。"""
    return _move_to_shared_observation_pose(
        f"共享粗定位：第{int(group_index)}组/{int(group_count)}组",
        current_tcp,
        target,
        args,
        motion_session,
        pose_session,
        target_height_mm=340.0,
        descent_profile="transit",
    )


def _plan_batch_coarse_group_pose(
    selected_holes: list[dict[str, Any]],
    current_tcp: np.ndarray,
    handeye: Any,
    fixed_rz_rad: float,
    intrinsics: Any,
    target_height_mm: float,
    view_margin_px: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """为批量粗定位规划能同时看到所有选中孔的340mm共同位姿。

    用于两阶段流程批量粗定位模式的共同进入位姿规划。
    使用选中孔的初始位置计算组中心和共同观察位姿。
    """
    if not selected_holes:
        raise RuntimeError("批量粗定位规划缺少选中孔")

    # 收集所有选中孔的初始位置和法向
    points_base = []
    normals_base = []
    for hole in selected_holes:
        point = np.asarray(hole["initial_center_base_mm"], dtype=np.float64).reshape(3)
        normal = np.asarray(
            hole.get(
                "planning_normal_base",
                hole.get(
                    "initial_shared_plane_normal_base",
                    hole["initial_plane_normal_base"],
                ),
            ),
            dtype=np.float64,
        ).reshape(3)
        points_base.append(point)
        normals_base.append(normal)

    points_base = np.asarray(points_base, dtype=np.float64)
    normals_base = np.asarray(normals_base, dtype=np.float64)

    # 使用第一个孔的法向作为参考，对齐所有法向
    reference_normal = _unit(normals_base[0], "batch coarse reference normal")
    aligned_normals = []
    for normal in normals_base:
        normal_unit = _unit(normal, "batch coarse normal")
        # 如果法向与参考法向夹角大于90度，则翻转
        aligned_normals.append(
            normal_unit if float(normal_unit @ reference_normal) >= 0.0 else -normal_unit
        )
    aligned_normals = np.asarray(aligned_normals, dtype=np.float64)

    # 计算组平均法向
    group_normal = _unit(np.median(aligned_normals, axis=0), "batch coarse group normal")

    # 计算目标相机姿态（固定RZ）
    R_tcp, rotation_info = _tcp_rotation_with_fixed_rz_for_camera_axis(
        -group_normal,
        current_tcp,
        handeye.T_tcp_rgb_camera,
        fixed_rz_rad,
    )
    R_base_camera = R_tcp @ np.asarray(handeye.T_tcp_rgb_camera, dtype=np.float64)[:3, :3]

    # 在目标相机坐标系下计算选中孔的包围盒中心
    points_camera_origin = (R_base_camera.T @ points_base.T).T
    group_camera_point = np.asarray([
        0.5 * (float(np.min(points_camera_origin[:, 0])) + float(np.max(points_camera_origin[:, 0]))),
        0.5 * (float(np.min(points_camera_origin[:, 1])) + float(np.max(points_camera_origin[:, 1]))),
        float(np.median(points_camera_origin[:, 2])),
    ])
    group_point_base = R_base_camera @ group_camera_point

    # 规划共同观察位姿
    target, pose_geometry = _plan_hole_tcp_pose_fixed_rz(
        group_point_base,
        group_normal,
        current_tcp,
        handeye.T_tcp_rgb_camera,
        fixed_rz_rad=fixed_rz_rad,
        camera_height_mm=float(target_height_mm),
    )

    T_base_camera_target = camera_transform(target, handeye.T_tcp_rgb_camera)

    # 验证所有孔是否在视野内
    width = float(getattr(intrinsics, "width", 0))
    height = float(getattr(intrinsics, "height", 0))
    if width <= 0.0 or height <= 0.0:
        raise RuntimeError("RGB内参缺少有效图像宽高，无法检查批量粗定位共同位姿的视野范围")

    projected_holes = {}
    out_of_view = []
    margin = float(view_margin_px)

    for hole in selected_holes:
        hole_id = int(hole["hole_id"])
        point_base = np.asarray(hole["initial_center_base_mm"], dtype=np.float64).reshape(3)
        try:
            projected = _project_base_point_to_pixel(
                point_base,
                T_base_camera_target,
                intrinsics,
            )
            projected_holes[hole_id] = projected

            # 检查是否在视野内（包含边缘余量）
            if (not np.isfinite(projected).all()
                or float(projected[0]) < margin
                or float(projected[0]) > width - margin
                or float(projected[1]) < margin
                or float(projected[1]) > height - margin):
                out_of_view.append(hole_id)
        except Exception:
            out_of_view.append(hole_id)

    if out_of_view:
        raise RuntimeError(
            f"批量粗定位：选中孔无法在共同340mm位姿下全部保持在视野内：{out_of_view}；"
            "请减少数量或选择更紧凑的一组孔"
        )

    # 计算投影包围盒
    projected_values = np.asarray(list(projected_holes.values()), dtype=np.float64)
    bbox_min = np.min(projected_values, axis=0)
    bbox_max = np.max(projected_values, axis=0)
    group_center_px = 0.5 * (bbox_min + bbox_max)

    return target, {
        "hole_order": [int(hole["hole_id"]) for hole in selected_holes],
        "group_point_base_mm": group_point_base,
        "group_normal_toward_camera_base": group_normal,
        "projected_holes_px": projected_holes,
        "group_bbox_px": [bbox_min[0], bbox_min[1], bbox_max[0], bbox_max[1]],
        "group_center_px": group_center_px,
        "view_margin_px": margin,
        "target_height_mm": float(target_height_mm),
        "target_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(target),
        "target_tcp_transform_mm": target,
        "pose_geometry": pose_geometry,
        "rotation_info": rotation_info,
    }


def _split_shared_cache_validation_groups(
    holes: list[dict[str, Any]], current_tcp: np.ndarray, handeye: Any,
    fixed_rz_rad: float, intrinsics: Any, target_height_mm: float,
    view_margin_px: float,
) -> list[list[dict[str, Any]]]:
    """Greedily split an arbitrary selection into groups visible from one 340 mm pose.

    Group membership is derived from the same conservative planner used by batch
    coarse localization.  A hole that cannot be planned even by itself is kept
    in a singleton group so the caller can record the planning failure and use
    the existing per-hole fallback path.
    """
    groups: list[list[dict[str, Any]]] = []
    candidate: list[dict[str, Any]] = []
    for hole in holes:
        expanded = [*candidate, hole]
        try:
            _plan_batch_coarse_group_pose(
                expanded, current_tcp, handeye, fixed_rz_rad, intrinsics,
                target_height_mm, view_margin_px,
            )
        except Exception:
            if candidate:
                groups.append(candidate)
                candidate = [hole]
            else:
                groups.append([hole])
                candidate = []
        else:
            candidate = expanded
    if candidate:
        groups.append(candidate)
    return groups


def _batch_coarse_localization_at_340mm(
    selected_holes: list[dict[str, Any]],
    current_tcp: np.ndarray,
    handeye: Any,
    pipeline: Any,
    align: Any,
    chain: Any,
    model: Any,
    confidence: float,
    cfg: TwoStageConfig,
    intrinsics: Any,
    run_dir: Path,
    timing: TimingRecorder,
    rows: list[dict[str, Any]],
    *,
    artifact_prefix: str | None = None,
) -> dict[Any, dict[str, Any]]:
    """在340mm共同位姿一次性检测所有孔的位姿和深度。

    返回: {hole_id: {
        "center_base_mm": [x, y, z],
        "normal_base": [nx, ny, nz],
        "plane_point_base_mm": [x, y, z],
        "depth_mm": float,
        "valid_frames": int,
        "center_scatter_p95_px": float,
        "plane_rmse_mm": float,
        "coarse_captures": [...],
        "coarse_center_camera_mm": [x, y, z],
        "coarse_plane_point_camera_mm": [x, y, z],
        "coarse_normal_camera": [nx, ny, nz],
        其他字段...
    }}
    """
    if not selected_holes:
        raise RuntimeError("批量粗定位：没有选中孔")

    T_base_camera = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
    expected_anchors = {
        int(hole["hole_id"]): _project_base_point_to_pixel(
            np.asarray(hole["initial_center_base_mm"], dtype=np.float64).reshape(3),
            T_base_camera,
            intrinsics,
        )
        for hole in selected_holes
    }
    holes_by_id = {int(hole["hole_id"]): hole for hole in selected_holes}
    batch_frames = max(1, int(cfg.batch_coarse_frames))
    min_valid = int(cfg.batch_coarse_min_valid)
    min_holes_per_frame = cfg.batch_coarse_min_holes_per_frame
    if min_holes_per_frame is None:
        # 批量模式的目标是一次获得整组孔的几何；单孔缺失时交给外层逐孔流程回退。
        min_holes_per_frame = len(selected_holes)
    min_holes_per_frame = max(1, min(int(min_holes_per_frame), len(selected_holes)))
    settle_discard_frames = max(0, int(cfg.batch_coarse_settle_discard_frames))
    hole_observations: dict[int, list[Observation]] = {
        hole_id: [] for hole_id in holes_by_id
    }
    frame_records: list[dict[str, Any]] = []
    latest_overlay_path: Path | None = None
    artifact_tag = "" if not artifact_prefix else f"{str(artifact_prefix).strip()}_"
    last_intrinsics = intrinsics
    anchor_correction: dict[str, np.ndarray] | None = None
    anchor_correction_info: dict[str, Any] | None = None
    anchor_correction_attempted = False

    # moveLine 已等待控制器的 steady 标志，但相机管线里仍可能有运动期间的
    # RGB-D 帧。先清空这段队列，避免把不同TCP位姿下的点云融合到同一个变换。
    discarded_frame_count = 0
    if settle_discard_frames > 0:
        with timing.measure(
            "batch_coarse/discard_settle_frames",
            target_frames=settle_discard_frames,
        ):
            for _ in range(settle_discard_frames):
                bundle = get_aligned_frame_bundle(pipeline, align, chain)
                if bundle is not None:
                    discarded_frame_count += 1

    with timing.measure(
        "batch_coarse/capture_all_holes",
        hole_count=len(selected_holes),
        target_frames=batch_frames,
    ):
        for frame_index in range(batch_frames):
            bundle = get_aligned_frame_bundle(pipeline, align, chain)
            if bundle is None or bundle.intrinsics is None:
                reason = "rgbd_frame_missing"
                for hole_id, anchor in expected_anchors.items():
                    hole_observations[hole_id].append(Observation(
                        "batch_coarse", frame_index, anchor,
                        error=reason,
                    ))
                frame_records.append({
                    "frame_index": frame_index,
                    "valid": False,
                    "valid_hole_count": 0,
                    "reason": reason,
                })
                continue

            last_intrinsics = bundle.intrinsics
            raw_anchors = {
                str(hole_id): _project_base_point_to_pixel(
                    np.asarray(hole["initial_center_base_mm"], dtype=np.float64).reshape(3),
                    T_base_camera,
                    bundle.intrinsics,
                )
                for hole_id, hole in holes_by_id.items()
            }
            try:
                detections = detect(model, bundle.color_bgr, confidence)
                if not anchor_correction_attempted:
                    anchor_correction_attempted = True
                    try:
                        anchor_correction, anchor_correction_info = (
                            _fit_batch_projected_anchor_correction(
                                detections,
                                raw_anchors,
                                cfg.multi_coarse_tracking_tolerance_px,
                                min_matches=max(4, min(6, len(selected_holes))),
                            )
                        )
                    except Exception as correction_exc:
                        # 校正只解决公共位姿的系统性投影偏差；校正本身不可靠时，
                        # 保留原始锚点并让后面的严格跟踪门和外层回退机制接管。
                        anchor_correction = None
                        anchor_correction_info = {
                            "enabled": False,
                            "error": f"{type(correction_exc).__name__}:{correction_exc}",
                        }
                current_anchors = (
                    anchor_correction
                    if anchor_correction is not None else raw_anchors
                )
                assignments = _assign_detections_to_projection(
                    detections,
                    current_anchors,
                    cfg.multi_coarse_tracking_tolerance_px,
                )
            except Exception as exc:
                reason = f"detection_failed:{type(exc).__name__}:{exc}"
                for hole_id, anchor in expected_anchors.items():
                    hole_observations[hole_id].append(Observation(
                        "batch_coarse", frame_index, anchor,
                        timestamp_ns=bundle.host_timestamp_ns,
                        error=reason,
                    ))
                frame_records.append({
                    "frame_index": frame_index,
                    "valid": False,
                    "valid_hole_count": 0,
                    "reason": reason,
                })
                continue

            view = bundle.color_bgr.copy()
            valid_hole_count = 0
            hole_records: list[dict[str, Any]] = []
            for hole_id, hole in holes_by_id.items():
                anchor = np.asarray(current_anchors[str(hole_id)], dtype=np.float64)
                assigned = assignments.get(str(hole_id))
                if assigned is None:
                    reason = "yolo_missing_or_far"
                    hole_observations[hole_id].append(Observation(
                        "batch_coarse", frame_index, anchor,
                        timestamp_ns=bundle.host_timestamp_ns,
                        error=reason,
                    ))
                    hole_records.append({"hole_id": hole_id, "valid": False, "reason": reason})
                    continue

                detection = assigned["detection"]
                center = np.asarray(detection["center"], dtype=np.float64).reshape(2)
                box = np.asarray(detection["box"], dtype=np.float64).reshape(4)
                radius = max(float(box[2] - box[0]), float(box[3] - box[1])) / 2.0
                try:
                    _, plane_info = hole_camera_point(
                        tuple(center.tolist()),
                        bundle.xyz_map_mm,
                        bundle.intrinsics,
                        radius,
                        ray_center_xy=center,
                        ray_center_is_undistorted=False,
                        include_points=True,
                        surface_selection_policy=COARSE_SURFACE_SELECTION_POLICY,
                    )
                    plane = _plane_estimate_from_info(
                        plane_info, f"batch coarse hole {hole_id} normal",
                    )
                    error = (
                        None if plane.rmse_mm <= float(cfg.max_plane_rmse_mm)
                        else f"plane_quality:{plane.rmse_mm:.3f}mm"
                    )
                    observation = Observation(
                        "batch_coarse", frame_index, center,
                        plane=plane,
                        timestamp_ns=bundle.host_timestamp_ns,
                        error=error,
                        tracking_distance_px=float(assigned["distance_px"]),
                    )
                    hole_observations[hole_id].append(observation)
                    valid = error is None
                    if valid:
                        valid_hole_count += 1
                    hole_records.append({
                        "hole_id": hole_id,
                        "valid": valid,
                        "distance_px": float(assigned["distance_px"]),
                        "plane_rmse_mm": float(plane.rmse_mm),
                        "ring_points": int(plane.ring_points),
                        "reason": error,
                    })
                except Exception as exc:
                    reason = f"pointcloud_failed:{type(exc).__name__}:{exc}"
                    hole_observations[hole_id].append(Observation(
                        "batch_coarse", frame_index, center,
                        timestamp_ns=bundle.host_timestamp_ns,
                        error=reason,
                        tracking_distance_px=float(assigned["distance_px"]),
                    ))
                    hole_records.append({
                        "hole_id": hole_id,
                        "valid": False,
                        "distance_px": float(assigned["distance_px"]),
                        "reason": reason,
                    })

            for hole_id, expected in current_anchors.items():
                point = tuple(np.rint(expected).astype(int))
                record = next(item for item in hole_records if str(item["hole_id"]) == hole_id)
                color = (0, 255, 0) if record["valid"] else (0, 165, 255)
                cv2.circle(view, point, 18, color, 2, cv2.LINE_AA)
                cv2.putText(
                    view, f"H{hole_id}", (point[0] + 10, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )
                assigned = assignments.get(hole_id)
                if assigned is not None:
                    detected = tuple(np.rint(assigned["detection"]["center"]).astype(int))
                    cv2.drawMarker(
                        view, detected, (255, 0, 255), cv2.MARKER_TILTED_CROSS,
                        12, 2, cv2.LINE_AA,
                    )
            cv2.putText(
                view,
                f"Batch RGB-D 340mm frame={frame_index} valid={valid_hole_count}/{len(selected_holes)}",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
            latest_overlay_path = run_dir / f"{artifact_tag}batch_coarse_340_frame_{frame_index:02d}.png"
            cv2.imwrite(str(latest_overlay_path), view)
            frame_records.append({
                "frame_index": frame_index,
                "valid": valid_hole_count >= min_holes_per_frame,
                "valid_hole_count": valid_hole_count,
                "required_hole_count": min_holes_per_frame,
                "detection_count": len(detections),
                "holes": hole_records,
                "overlay_path": str(latest_overlay_path),
            })

    # 融合每个孔的多帧观测
    batch_results: dict[int, dict[str, Any]] = {}

    for hole in selected_holes:
        hole_id = int(hole["hole_id"])
        observations = hole_observations[hole_id]
        valid_group_frame_indices = {
            int(record["frame_index"])
            for record in frame_records
            if bool(record.get("valid", False))
        }
        # 默认要求整组孔同帧有效；否则不同孔会来自不同的运动/曝光阶段，
        # 虽然每个孔单独看似有足够帧，批量共同位姿却没有一致的数据基础。
        fusion_observations = observations
        if min_holes_per_frame >= len(selected_holes):
            fusion_observations = [
                item for item in observations
                if int(item.frame_index) in valid_group_frame_indices
            ]
        valid_observations = [
            item for item in fusion_observations
            if item.error is None and item.plane is not None
        ]
        error_counts: dict[str, int] = {}
        for item in observations:
            if item.error:
                key = str(item.error).split(":", 1)[0]
                error_counts[key] = error_counts.get(key, 0) + 1

        # 记录全部观测数据；其中未进入融合的帧也要保留，便于诊断运动/队列问题。
        rows.extend(_observation_rows(observations))

        if len(valid_observations) < min_valid:
            # 当前孔的有效帧数不足，标记为失败
            batch_results[hole_id] = {
                "success": False,
                "valid_frames": len(valid_observations),
                "total_frames": len(observations),
                "min_valid_frames": min_valid,
                "error": (
                    f"整组有效帧不足: {len(valid_observations)} < {min_valid}"
                    if min_holes_per_frame >= len(selected_holes)
                    else f"有效帧数不足: {len(valid_observations)} < {min_valid}"
                ),
                "error_counts": error_counts,
                "observations": observations,
            }
            continue

        # 融合几何信息
        try:
            summary = _fuse_coarse(
                fusion_observations,
                cfg,
                min_valid_frames=min_valid,
                max_center_scatter_p95_px=cfg.max_coarse_center_scatter_p95_px,
                max_tracking_distance_p95_px=cfg.max_coarse_tracking_distance_p95_px,
            )

            # 转换到base坐标系
            R_base_camera = T_base_camera[:3, :3]
            t_base_camera = T_base_camera[:3, 3]
            camera_origin = t_base_camera

            point_camera = np.asarray(summary["plane_point_camera_mm"], dtype=np.float64).reshape(3)
            point_base = R_base_camera @ point_camera + t_base_camera

            normal_camera = _unit(
                np.asarray(summary["plane_normal_camera"], dtype=np.float64),
                f"batch coarse hole {hole_id} camera normal",
            )
            normal_base = _unit(
                R_base_camera @ normal_camera,
                f"batch coarse hole {hole_id} base normal",
            )

            # 确保法向指向相机
            if float(normal_base @ (camera_origin - point_base)) < 0.0:
                normal_base = -normal_base
                normal_camera = -normal_camera

            local_plane_point_camera = summary.get("surface_plane_point_camera_mm")
            local_plane_point_base = (
                R_base_camera @ np.asarray(local_plane_point_camera, dtype=np.float64) + t_base_camera
                if local_plane_point_camera is not None else None
            )

            batch_results[hole_id] = {
                "success": True,
                "center_base_mm": point_base.tolist(),
                "center_camera_mm": point_camera.tolist(),
                "normal_base": normal_base.tolist(),
                "normal_camera": normal_camera.tolist(),
                "plane_point_base_mm": (
                    local_plane_point_base.tolist()
                    if local_plane_point_base is not None
                    else point_base.tolist()
                ),
                "plane_point_camera_mm": (
                    point_camera.tolist()
                    if local_plane_point_camera is None
                    else np.asarray(local_plane_point_camera, dtype=np.float64).tolist()
                ),
                "depth_mm": float(point_camera[2]),
                "valid_frames": summary["valid_frames"],
                "total_frames": summary["total_frames"],
                "center_scatter_p95_px": summary["center_scatter_p95_px"],
                "tracking_distance_p95_px": summary["tracking_distance_p95_px"],
                "plane_rmse_mm": summary["plane_rmse_median_mm"],
                "ring_points_median": summary.get("ring_points_median"),
                "center_px": summary["center_px"],
                "surface_model": summary["surface_model"],
                "surface_selection_policy": summary.get("surface_selection_policy"),
                "front_surface_z_mm": summary.get("front_surface_z_median_mm"),
                "ring_points_raw_median": summary.get("ring_points_raw_median"),
                "surface_points_selected_median": summary.get("surface_points_selected_median"),
                "sphere_center_camera_mm": summary.get("sphere_center_camera_mm"),
                "sphere_radius_mm": summary.get("sphere_radius_mm"),
                "coarse_captures": [{
                    "capture_index": 0,
                    "mode": "batch_coarse_at_340mm",
                    "tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                    "summary": summary,
                }],
                "observations": observations,
                "error_counts": error_counts,
            }

        except Exception as exc:
            batch_results[hole_id] = {
                "success": False,
                "valid_frames": len(valid_observations),
                "total_frames": len(observations),
                "error": f"几何融合失败: {type(exc).__name__}: {exc}",
                "error_counts": error_counts,
                "observations": observations,
            }

    artifact_warnings: list[str] = []
    successful = {
        hole_id: result for hole_id, result in batch_results.items()
        if result.get("success")
    }
    if successful:
        point_chunks: list[np.ndarray] = []
        hole_labels: list[np.ndarray] = []
        frame_labels: list[np.ndarray] = []
        for hole_id, result in successful.items():
            for item in result["observations"]:
                if (
                    item.error is not None or item.plane is None
                    or item.plane.points_camera_mm is None
                ):
                    continue
                points = np.asarray(item.plane.points_camera_mm, dtype=np.float32).reshape(-1, 3)
                point_chunks.append(points)
                hole_labels.append(np.full(len(points), hole_id, dtype=np.int32))
                frame_labels.append(np.full(len(points), item.frame_index, dtype=np.int32))
        if point_chunks:
            archive_path = run_dir / f"{artifact_tag}batch_coarse_340_all_holes_pointcloud.npz"
            try:
                np.savez_compressed(
                    archive_path,
                    points_camera_mm=np.concatenate(point_chunks, axis=0),
                    point_hole_ids=np.concatenate(hole_labels),
                    point_frame_indices=np.concatenate(frame_labels),
                    hole_ids=np.asarray(sorted(successful), dtype=np.int32),
                    T_base_camera=np.asarray(T_base_camera, dtype=np.float64),
                )
            except Exception as exc:
                artifact_warnings.append(f"npz:{type(exc).__name__}:{exc}")
                archive_path = None
        else:
            archive_path = None

        for hole_id, result in successful.items():
            pointcloud_path = _save_coarse_pointcloud_image(
                {
                    "coarse_center_camera_mm": result["center_camera_mm"],
                    "coarse_plane_point_camera_mm": result["plane_point_camera_mm"],
                    "coarse_normal_camera": result["normal_camera"],
                    "coarse_plane_rmse_mm": result["plane_rmse_mm"],
                    "coarse_center_scatter_p95_px": result["center_scatter_p95_px"],
                },
                hole_id,
                run_dir,
                observations=result["observations"],
                cache_entry=None,
                T_base_camera=T_base_camera,
                intrinsics=last_intrinsics,
                rgb_path=latest_overlay_path,
                source_label="batch_coarse_340mm",
            )
            result["pointcloud_image_path"] = (
                None if pointcloud_path is None else str(pointcloud_path)
            )
            result["batch_pointcloud_archive_path"] = (
                None if archive_path is None else str(archive_path)
            )

    batch_results["_batch_metadata"] = {
        "frame_records": frame_records,
        "anchor_correction": anchor_correction_info,
        "required_holes_per_frame": min_holes_per_frame,
        "settle_discard_frames": settle_discard_frames,
        "discarded_frame_count": discarded_frame_count,
        "max_center_scatter_p95_px": float(cfg.max_coarse_center_scatter_p95_px),
        "max_tracking_distance_p95_px": float(cfg.max_coarse_tracking_distance_p95_px),
        "latest_overlay_path": None if latest_overlay_path is None else str(latest_overlay_path),
        "artifact_warnings": artifact_warnings,
    }
    return batch_results


def _flush_rgb_queue_until_fresh(
    pipeline: Any,
    minimum_discard_frames: int,
    *,
    maximum_extra_frames: int = 20,
    fresh_host_interval_ms: float = 10.0,
    required_fresh_intervals: int = 2,
) -> dict[str, Any]:
    """丢弃运动过程RGB帧，直到确认pipeline已经返回实时新帧。

    Orbbec队列中的旧帧会在几毫秒内连续返回；队列清空后，wait_for_frames
    必须等待下一个相机周期。RgbFrameBundle的host_timestamp_ns记录在
    wait_for_frames返回后，因此连续两个足够长的主机时间间隔可以作为
    “已追上实时流”的证据。测试替身或旧调用方没有设备帧元数据时，保持
    原有行为，只执行配置的最少丢帧数。
    """
    minimum = max(0, int(minimum_discard_frames))
    maximum = minimum + max(0, int(maximum_extra_frames))
    required = max(1, int(required_fresh_intervals))
    threshold_ns = max(0, int(float(fresh_host_interval_ms) * 1_000_000.0))
    if minimum == 0:
        return {
            "discarded_frame_count": 0,
            "fresh_frame_confirmed": False,
            "reason": "disabled",
            "records": [],
        }

    discarded = 0
    attempts = 0
    fresh_streak = 0
    previous_host_timestamp_ns: int | None = None
    metadata_available = False
    records: list[dict[str, Any]] = []
    max_attempts = max(maximum + 5, minimum)
    while discarded < maximum and attempts < max_attempts:
        attempts += 1
        bundle = get_rgb_frame_bundle(pipeline)
        if bundle is None:
            records.append({"attempt": attempts, "valid": False})
            continue
        discarded += 1
        has_frame_metadata = bool(
            hasattr(bundle, "color_frame_index")
            or hasattr(bundle, "color_timestamp_us")
        )
        metadata_available = metadata_available or has_frame_metadata
        host_timestamp_ns = int(bundle.host_timestamp_ns)
        host_interval_ms = None
        if previous_host_timestamp_ns is not None:
            interval_ns = host_timestamp_ns - previous_host_timestamp_ns
            host_interval_ms = float(interval_ns / 1_000_000.0)
            if has_frame_metadata and interval_ns >= threshold_ns:
                fresh_streak += 1
            elif has_frame_metadata:
                fresh_streak = 0
        previous_host_timestamp_ns = host_timestamp_ns
        records.append({
            "attempt": attempts,
            "discard_index": discarded,
            "valid": True,
            "host_timestamp_ns": host_timestamp_ns,
            "host_interval_ms": host_interval_ms,
            "color_timestamp_us": getattr(bundle, "color_timestamp_us", None),
            "color_frame_index": getattr(bundle, "color_frame_index", None),
            "fresh_interval_streak": fresh_streak,
        })
        if discarded < minimum:
            continue
        if not metadata_available:
            return {
                "discarded_frame_count": discarded,
                "attempt_count": attempts,
                "fresh_frame_confirmed": False,
                "reason": "minimum_reached_without_frame_metadata",
                "fresh_host_interval_ms": float(fresh_host_interval_ms),
                "required_fresh_intervals": required,
                "records": records,
            }
        if fresh_streak >= required:
            return {
                "discarded_frame_count": discarded,
                "attempt_count": attempts,
                "fresh_frame_confirmed": True,
                "reason": "minimum_and_fresh_intervals_reached",
                "fresh_host_interval_ms": float(fresh_host_interval_ms),
                "required_fresh_intervals": required,
                "records": records,
            }

    return {
        "discarded_frame_count": discarded,
        "attempt_count": attempts,
        "fresh_frame_confirmed": bool(metadata_available and fresh_streak >= required),
        "reason": "maximum_discard_reached_before_fresh_confirmation",
        "fresh_host_interval_ms": float(fresh_host_interval_ms),
        "required_fresh_intervals": required,
        "records": records,
    }


def _batch_fine_localization_at_260mm(
    selected_holes: list[dict[str, Any]],
    current_tcp: np.ndarray,
    handeye: Any,
    pipeline: Any,
    model: Any,
    confidence: float,
    cfg: TwoStageConfig,
    intrinsics: Any,
    run_dir: Path,
    timing: TimingRecorder,
    rows: list[dict[str, Any]],
    *,
    artifact_prefix: str | None = None,
) -> dict[Any, dict[str, Any]]:
    """在一个260mm共同位姿同时精定位当前视野内的全部选中孔。

    机器人在本函数调用前已经移动到共同260mm位姿。本函数只采集RGB帧，
    每帧运行一次YOLO并把检测框一对一分配给所有目标孔；椭圆质量门和
    融合/验收沿用单孔精定位逻辑；同一共同位姿执行一次短连拍，达到
    有效帧数与稳定性门后一次性输出当前孔集合。移动到另一个共享位姿的
    补拍由外层工作流负责，本函数本身不移动机器人。
    """
    if not selected_holes:
        raise RuntimeError("批量精定位：没有选中孔")

    T_base_camera = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
    projected_holes = {
        str(int(hole["hole_id"])): _project_base_point_to_pixel(
            np.asarray(hole["initial_center_base_mm"], dtype=np.float64).reshape(3),
            T_base_camera, intrinsics,
        )
        for hole in selected_holes
    }
    expected_metadata = {
        str(int(hole["hole_id"])): {
            "class_id": int(hole["initial_detection"].get("class_id", -1)),
            "box": hole["initial_detection"].get("box"),
        }
        for hole in selected_holes
    }
    hole_ids = [int(hole["hole_id"]) for hole in selected_holes]
    observations_by_hole: dict[int, list[Observation]] = {
        hole_id: [] for hole_id in hole_ids
    }
    frame_records: list[dict[str, Any]] = []
    latest_bundle: Any = None
    artifact_tag = "" if not artifact_prefix else f"{str(artifact_prefix).strip()}_"
    anchor_correction: dict[str, np.ndarray] | None = None
    anchor_correction_info: dict[str, Any] | None = None
    anchor_correction_attempted = False
    discarded_frame_count = 0
    settle_flush: dict[str, Any] = {
        "discarded_frame_count": 0,
        "fresh_frame_confirmed": False,
        "reason": "disabled",
        "records": [],
    }
    batch_cfg = replace(
        cfg,
        fine_frames=int(cfg.batch_fine_frames),
        min_fine_valid=int(cfg.batch_fine_min_valid),
        fine_stable_min_frames=int(cfg.batch_fine_stable_min_frames),
        fine_settle_discard_frames=int(cfg.batch_fine_settle_discard_frames),
    )
    settle_discard_frames = max(0, int(batch_cfg.fine_settle_discard_frames))
    if settle_discard_frames > 0:
        with timing.measure(
            "batch_fine/discard_settle_frames",
            target_frames=settle_discard_frames,
        ):
            settle_flush = _flush_rgb_queue_until_fresh(
                pipeline, settle_discard_frames,
            )
            discarded_frame_count = int(settle_flush["discarded_frame_count"])

    max_frames = max(1, int(batch_cfg.fine_frames))
    # batch_fine_frames定义同一共同位姿下短连拍的正式帧数上限；达到
    # 当前孔集合的有效帧数和稳定门即可提前结束。
    max_attempts = max_frames
    stable_gate_frames = max(
        1, int(batch_cfg.fine_stable_min_frames), int(batch_cfg.min_fine_valid),
    )
    locked_holes: set[int] = set()
    with timing.measure(
        "batch_fine/capture_all_holes",
        hole_count=len(selected_holes), target_frames=max_frames,
    ):
        for frame_index in range(max_attempts):
            if len(locked_holes) == len(hole_ids):
                break

            bundle = get_rgb_frame_bundle(pipeline)
            if bundle is None or bundle.intrinsics is None:
                for hole_id, anchor in projected_holes.items():
                    if int(hole_id) in locked_holes:
                        continue
                    observations_by_hole[int(hole_id)].append(Observation(
                        "batch_fine", frame_index,
                        np.asarray(anchor, dtype=np.float64),
                        timestamp_ns=None, error="rgb_frame_missing",
                    ))
                frame_records.append({
                    "frame_index": frame_index, "valid": False,
                    "valid_hole_count": 0, "reason": "rgb_frame_missing",
                })
                continue

            latest_bundle = bundle
            detections = detect(model, bundle.color_bgr, confidence)
            raw_anchors = {
                hole_id: _project_base_point_to_pixel(
                    np.asarray(next(
                        hole["initial_center_base_mm"]
                        for hole in selected_holes
                        if str(int(hole["hole_id"])) == hole_id
                    ), dtype=np.float64).reshape(3),
                    camera_transform(current_tcp, handeye.T_tcp_rgb_camera),
                    bundle.intrinsics,
                )
                for hole_id in projected_holes
            }
            if not anchor_correction_attempted:
                anchor_correction_attempted = True
                try:
                    anchor_correction, anchor_correction_info = (
                        _fit_batch_projected_anchor_correction(
                            detections,
                            raw_anchors,
                            cfg.multi_coarse_tracking_tolerance_px,
                            min_matches=max(3, min(6, len(selected_holes))),
                        )
                    )
                except Exception as correction_exc:
                    anchor_correction = None
                    anchor_correction_info = {
                        "enabled": False,
                        "error": f"{type(correction_exc).__name__}:{correction_exc}",
                    }
            current_anchors = anchor_correction if anchor_correction is not None else raw_anchors
            assignments = _assign_detections_to_projection(
                detections,
                current_anchors,
                batch_cfg.fine_pointcloud_anchor_tolerance_px,
                expected_metadata=expected_metadata,
            )
            valid_hole_count = 0
            hole_records: list[dict[str, Any]] = []
            for hole in selected_holes:
                hole_id = int(hole["hole_id"])
                key = str(hole_id)
                if hole_id in locked_holes:
                    hole_records.append({
                        "hole_id": hole_id, "valid": True, "locked": True,
                    })
                    continue
                anchor = np.asarray(current_anchors[key], dtype=np.float64)
                assigned = assignments.get(key)
                if assigned is None:
                    observations_by_hole[hole_id].append(Observation(
                        "batch_fine", frame_index, anchor,
                        timestamp_ns=bundle.host_timestamp_ns,
                        error="yolo_missing_or_far",
                    ))
                    hole_records.append({
                        "hole_id": hole_id, "valid": False,
                        "reason": "yolo_missing_or_far",
                    })
                    continue

                detection = assigned["detection"]
                detection_center = np.asarray(
                    detection["center"], dtype=np.float64,
                ).reshape(2)
                ellipse = fit_hole_ellipse(bundle.color_bgr, detection, bundle.intrinsics)
                strict_ok = _ellipse_ok(ellipse, cfg)
                if not strict_ok:
                    rejected_center = (
                        np.asarray(ellipse["center_px"], dtype=np.float64).reshape(2)
                        if ellipse is not None else
                        undistort_pixels(
                            bundle.intrinsics, detection_center.reshape(1, 2),
                            pixel_output=True,
                        )[0]
                    )
                    observations_by_hole[hole_id].append(Observation(
                        "batch_fine", frame_index, rejected_center, ellipse,
                        timestamp_ns=bundle.host_timestamp_ns,
                        error="ellipse_quality", center_source="rejected",
                        quality_note="strict_geometric_center_gate_failed_no_yolo_fallback",
                        tracking_distance_px=float(assigned["distance_px"]),
                    ))
                    hole_records.append({
                        "hole_id": hole_id, "valid": False,
                        "distance_px": float(assigned["distance_px"]),
                        "reason": "ellipse_quality",
                    })
                    continue

                center = np.asarray(ellipse["center_px"], dtype=np.float64).reshape(2)
                center_source = str(ellipse.get("fit_method") or "ellipse")
                observations_by_hole[hole_id].append(Observation(
                    "batch_fine", frame_index, center, ellipse,
                    timestamp_ns=bundle.host_timestamp_ns,
                    center_source=center_source,
                    quality_note="strict_geometric_center",
                    tracking_distance_px=float(assigned["distance_px"]),
                ))
                valid_hole_count += 1
                hole_records.append({
                    "hole_id": hole_id, "valid": True,
                    "distance_px": float(assigned["distance_px"]),
                    "center_source": center_source,
                    "geometric_center_px": center.tolist(),
                    "geometric_center_px_distorted": np.asarray(
                        ellipse.get("center_px_distorted", center), dtype=np.float64,
                    ).reshape(2).tolist(),
                    "ellipse_residual_px": float(ellipse["residual_px"]),
                })

            for hole_id in hole_ids:
                if hole_id in locked_holes:
                    continue
                if _fine_burst_stable(
                    {hole_id: observations_by_hole[hole_id]},
                    stable_gate_frames,
                    float(batch_cfg.fine_stable_center_scatter_p95_px),
                ):
                    locked_holes.add(hole_id)

            geometric_centers = {
                str(record["hole_id"]): np.asarray(
                    record["geometric_center_px_distorted"], dtype=np.float64,
                )
                for record in hole_records
                if record.get("geometric_center_px_distorted") is not None
            }
            view = bundle.color_bgr.copy()
            for detection in detections:
                box = tuple(np.rint(np.asarray(detection["box"], dtype=np.float64)).astype(int))
                cv2.rectangle(view, (box[0], box[1]), (box[2], box[3]), (255, 180, 0), 1)
            for hole_id, anchor in current_anchors.items():
                point = tuple(np.rint(np.asarray(anchor, dtype=np.float64)).astype(int))
                assigned = assignments.get(hole_id)
                color = (0, 255, 0) if assigned is not None else (0, 165, 255)
                cv2.circle(view, point, 14, color, 2, cv2.LINE_AA)
                cv2.putText(
                    view, f"H{hole_id}", (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )
                if assigned is not None:
                    detected_point = tuple(np.rint(np.asarray(
                        assigned["detection"]["center"], dtype=np.float64,
                    )).astype(int))
                    cv2.drawMarker(
                        view, detected_point, (255, 0, 255),
                        cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA,
                    )
                geometric_center = geometric_centers.get(str(hole_id))
                if geometric_center is not None:
                    geometric_point = tuple(np.rint(geometric_center).astype(int))
                    cv2.drawMarker(
                        view, geometric_point, (0, 0, 255),
                        cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA,
                    )
            cv2.putText(
                view,
                f"Batch RGB 260mm frame={frame_index} valid={valid_hole_count}/{len(selected_holes)}",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA,
            )
            overlay_path = run_dir / f"{artifact_tag}batch_fine_260_frame_{frame_index:02d}.png"
            cv2.imwrite(str(overlay_path), view)
            frame_records.append({
                "frame_index": frame_index,
                "valid": valid_hole_count > 0,
                "valid_hole_count": valid_hole_count,
                "detection_count": len(detections),
                "holes": hole_records,
                "locked_holes": sorted(locked_holes),
                "overlay_path": str(overlay_path),
            })

    if latest_bundle is None:
        raise RuntimeError("批量精定位期间未获得RGB帧")

    batch_results: dict[int, dict[str, Any]] = {}
    for hole_id in hole_ids:
        observations = observations_by_hole[hole_id]
        rows.extend(_observation_rows(observations))
        try:
            summary = _fuse_fine(
                observations, batch_cfg,
                max_center_scatter_p95_px=batch_cfg.max_fine_center_scatter_p95_px,
            )
            summary.update({
                "fine_quality_status": "strict",
                "fine_quality_note": None,
                "fine_recovery_attempts": [{
                    "attempt": 1,
                    "name": "batch_fine",
                    "capture_status": "completed",
                    "total_frames": len(observations),
                    "valid_frames": int(summary["valid_frames"]),
                    "mode": "batch_fine_at_260mm",
                }],
            })
            batch_results[hole_id] = {
                "success": True,
                "fine": summary,
                "observations": observations,
                "intrinsics": latest_bundle.intrinsics,
                "capture_tcp": np.asarray(current_tcp, dtype=np.float64).copy(),
                "expected_anchor_px": np.asarray(
                    projected_holes[str(hole_id)], dtype=np.float64,
                ).copy(),
                "fine_capture_overlay_path": frame_records[-1].get("overlay_path")
                if frame_records else None,
            }
        except Exception as exc:
            batch_results[hole_id] = {
                "success": False,
                "fine": None,
                "observations": observations,
                "intrinsics": latest_bundle.intrinsics,
                "capture_tcp": np.asarray(current_tcp, dtype=np.float64).copy(),
                "expected_anchor_px": np.asarray(
                    projected_holes[str(hole_id)], dtype=np.float64,
                ).copy(),
                "error": f"{type(exc).__name__}:{exc}",
            }

    batch_results["_batch_metadata"] = {
        "frame_records": frame_records,
        "anchor_correction": anchor_correction_info,
        "settle_discard_frames": settle_discard_frames,
        "discarded_frame_count": discarded_frame_count,
        "settle_flush": settle_flush,
        "max_frames": max_frames,
        "max_attempts": max_attempts,
        "locked_holes": sorted(locked_holes),
        "max_tracking_distance_px": float(batch_cfg.fine_pointcloud_anchor_tolerance_px),
        "latest_overlay_path": (
            frame_records[-1].get("overlay_path") if frame_records else None
        ),
    }
    return batch_results


def optimize_hole_order(
    hole_ids: list[int], target_xy_by_hole: dict[int, Any], start_xy: Any | None = None,
) -> tuple[list[int], float]:
    """按目标 XY 位置稳定地生成最近邻孔顺序。"""
    ordered = [int(value) for value in hole_ids]
    valid = {
        key: np.asarray(target_xy_by_hole[key], dtype=np.float64).reshape(2)
        for key in ordered if key in target_xy_by_hole
    }
    if any(not np.isfinite(value).all() for value in valid.values()) or len(valid) != len(ordered):
        return ordered, 0.0
    remaining = list(ordered)
    result: list[int] = []
    current = None if start_xy is None else np.asarray(start_xy, dtype=np.float64).reshape(2)
    while remaining:
        next_id = (
            remaining[0]
            if current is None
            else min(
                remaining,
                key=lambda key: (
                    float(np.linalg.norm(valid[key] - current)),
                    ordered.index(key),
                ),
            )
        )
        result.append(next_id)
        remaining.remove(next_id)
        current = valid[next_id]
    distance = sum(
        float(np.linalg.norm(valid[right] - valid[left]))
        for left, right in zip(result, result[1:])
    )
    return result, distance


def _run_sequential_hole_workflow(
    args: Any, handeye: Any, model: Any, cfg: TwoStageConfig, run_dir: Path,
    report: dict[str, Any], timing: TimingRecorder, rows: list[dict[str, Any]],
    runtime: dict[str, Any],
    pose_session: Any, motion_session: Any, current_tcp: np.ndarray,
    initial_holes: list[dict[str, Any]], initial_intrinsics: Any,
    *,
    coarse_cache_entries: dict[int, CoarseCacheEntry] | None = None,
    coarse_cache_sources: dict[int, str] | None = None,
    coarse_cache_source_ids: dict[int, int] | None = None,
    coarse_cache_dir: Path | None = None,
    coarse_cache_gates: CacheValidationGates | None = None,
    coarse_cache_metadata: dict[str, Any] | None = None,
    persistent_cache_entries: dict[int, CoarseCacheEntry] | None = None,
    persistent_cache_dir: Path | None = None,
    persistent_cache_metadata: dict[str, Any] | None = None,
) -> int:
    """按初始孔号逐个执行：粗定位 -> 精定位 -> 最终目标点 -> 下一个孔。"""
    if not initial_holes:
        raise RuntimeError("没有初始选定孔，无法执行顺序定位")

    initial_pointcloud_reused_holes: list[int] = []
    if cfg.batch_fine_localization and len(initial_holes) > 1:
        initial_pointcloud_reused_holes = [
            int(hole["hole_id"])
            for hole in initial_holes
            if _reuse_initial_pointcloud_geometry_for_batch_fine(hole)
        ]
    all_selected_two_capture_mode = bool(
        len(initial_pointcloud_reused_holes) == len(initial_holes)
        and len(initial_holes) > 1
    )

    fixed_rz_rad = _matrix_to_rpy_zyx(current_tcp[:3, :3])[2]
    results: list[dict[str, Any]] = []
    order_ids = [int(item["hole_id"]) for item in initial_holes]
    cache_entries = coarse_cache_entries if coarse_cache_entries is not None else {}
    cache_sources = coarse_cache_sources if coarse_cache_sources is not None else {}
    cache_source_ids = coarse_cache_source_ids if coarse_cache_source_ids is not None else {}
    cache_gates = coarse_cache_gates or CacheValidationGates()
    cache_enabled = bool(getattr(args, "reuse_coarse_cache", True)) and coarse_cache_dir is not None
    persistent_enabled = bool(
        cache_enabled
        and getattr(args, "reuse_persistent_coarse_cache", True)
        and persistent_cache_dir is not None
    )
    persistent_entries = persistent_cache_entries if persistent_cache_entries is not None else {}
    # 本轮新生成的正式缓存孔号。缓存只允许来自 340 mm 一拍多批量粗定位。
    cache_built_ids: set[int] = set(
        int(value) for value in report.get("coarse_cache", {}).get("cache_built", [])
    )
    report.setdefault("coarse_cache", {}).setdefault("cache_reused", [])
    report.setdefault("coarse_cache", {}).setdefault("persistent_cache_loaded", [])
    report.setdefault("coarse_cache", {}).setdefault("persistent_cache_reused", [])
    report.setdefault("coarse_cache", {}).setdefault("cache_validation_skipped", [])
    report.setdefault("coarse_cache", {}).setdefault("cache_validation_failed", [])
    report.setdefault("coarse_cache", {}).setdefault("cache_invalidated", [])
    report.setdefault("coarse_cache", {}).setdefault("full_coarse_fallback", [])
    report["stages"]["sequential_plan"] = {
        "mode": "single_shared_coarse_capture_then_shared_fine_capture_with_supplement",
        "hole_count": len(initial_holes),
        "hole_order": order_ids,
        "fixed_rz_rad": fixed_rz_rad,
        "coarse_settle_buffer_s": float(cfg.coarse_settle_delay_s),
        "tracking_identity_source": "initial_selection_order_and_initial_rgbd_3d_projection",
        "confirmation_policy": "no_per_hole_pause_after_batch_fine_capture",
        "camera_pipeline_policy": "reuse_single_rgbd_pipeline_for_coarse_and_rgb_fine",
        "initial_pointcloud_reused_holes": initial_pointcloud_reused_holes,
        "capture_policy": (
            "all_selected_holes_in_one_initial_pose_per_stage_then_failed_holes"
            "_in_shared_supplement_pose"
        ),
    }

    def ensure_rgbd_pipeline() -> tuple[Any, Any, Any]:
        if runtime.get("rgbd_pipeline") is None:
            with timing.measure("camera/restart_rgbd_pipeline"):
                pipeline, align, chain = init_pipeline()
            runtime["rgbd_pipeline"] = pipeline
            runtime["align"] = align
            runtime["chain"] = chain
        return runtime["rgbd_pipeline"], runtime["align"], runtime["chain"]

    # 正式多孔流程固定先在340 mm共同位姿拍摄全部选中孔。初始RGB-D
    # 点云只负责规划这个共同粗定位位姿，不能代替正式粗定位拍摄。
    batch_coarse_results: dict[int, dict[str, Any]] = {}
    batch_coarse_for_cache = bool(
        all_selected_two_capture_mode
        or cfg.batch_coarse_localization
        or (cache_enabled and not cache_entries)
    )

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

    # 执行批量粗定位（如果启用且有多个孔）
    if batch_coarse_for_cache and len(initial_holes) > 1:
        print(
            f"[BATCH_COARSE] 启用批量粗定位模式: {len(initial_holes)}个孔",
            flush=True,
        )

        rgbd_pipeline, align, chain = ensure_rgbd_pipeline()

        batch_groups = (
            [list(initial_holes)]
            if all_selected_two_capture_mode else
            _split_shared_cache_validation_groups(
                initial_holes, current_tcp, handeye, fixed_rz_rad, initial_intrinsics,
                cfg.coarse_height_mm, cfg.batch_coarse_view_margin_px,
            )
        )
        batch_plan: dict[str, Any] = {
            "enabled": True,
            "hole_count": len(initial_holes),
            "hole_order": order_ids,
            "group_count": len(batch_groups),
            "all_selected_holes_single_group": bool(all_selected_two_capture_mode),
            "capture_frames": int(cfg.batch_coarse_frames),
            "combined_position_policy": "projected_bbox_center_above_all_selected_holes",
            "motion_policy": "shared_vertical_lift_min10_safe_horizontal_descent_guard10_then_pure_descent10_then_steady_capture",
            "view_margin_px": float(cfg.batch_coarse_view_margin_px),
            "groups": [],
        }
        batch_capture_metadata: list[dict[str, Any]] = []
        # 每个批量组有自己的340 mm TCP/相机变换；缓存点云必须使用产生该
        # 组观测的变换，不能在循环结束后统一使用最后一组的位姿。
        batch_cache_transforms: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for group_index, group in enumerate(batch_groups, start=1):
            group_ids = [int(hole["hole_id"]) for hole in group]
            group_report: dict[str, Any] = {
                "group_index": group_index,
                "hole_ids": group_ids,
                "hole_count": len(group),
                "accepted_holes": [],
                "fallback_holes": list(group_ids),
            }
            try:
                with timing.measure(
                    f"batch_coarse/group_{group_index:02d}/plan_group_pose",
                    hole_count=len(group), group_index=group_index,
                ):
                    batch_target, batch_geometry = _plan_batch_coarse_group_pose(
                        group, current_tcp, handeye, fixed_rz_rad,
                        initial_intrinsics, cfg.coarse_height_mm,
                        cfg.batch_coarse_view_margin_px,
                    )
                group_report.update({
                    "target_tcp_pose_m_rad": batch_geometry["target_tcp_pose_m_rad"],
                    "combined_point_base_mm": batch_geometry.get("group_point_base_mm"),
                    "combined_normal_base": batch_geometry.get(
                        "group_normal_toward_camera_base"
                    ),
                    "pose_policy": "above_projected_bbox_center_of_all_selected_holes",
                    "projected_holes_px": {
                        str(key): value.tolist()
                        for key, value in batch_geometry["projected_holes_px"].items()
                    },
                    "group_bbox_px": batch_geometry["group_bbox_px"],
                    "group_center_px": batch_geometry["group_center_px"].tolist(),
                })

                with timing.measure(
                    f"batch_coarse/group_{group_index:02d}/navigate_to_group_pose",
                    hole_count=len(group), group_index=group_index,
                ):
                    current_tcp = _move_to_shared_coarse_pose(
                        group_index,
                        len(batch_groups),
                        current_tcp,
                        batch_target,
                        args,
                        motion_session,
                        pose_session,
                    )

                coarse_settle_delay_s = max(0.0, float(cfg.coarse_settle_delay_s))
                if coarse_settle_delay_s > 0.0:
                    with timing.measure(
                        f"batch_coarse/group_{group_index:02d}/settle_delay",
                        delay_s=coarse_settle_delay_s, group_index=group_index,
                    ):
                        time.sleep(coarse_settle_delay_s)

                group_results = _batch_coarse_localization_at_340mm(
                    group, current_tcp, handeye, rgbd_pipeline, align, chain,
                    model, args.confidence, cfg, initial_intrinsics, run_dir,
                    timing, rows,
                    artifact_prefix=f"batch_coarse_group_{group_index:02d}",
                )
                group_metadata = group_results.pop("_batch_metadata", {})
                group_tcp = np.asarray(current_tcp, dtype=np.float64).copy()
                group_T_base_camera = camera_transform(
                    group_tcp, handeye.T_tcp_rgb_camera,
                )
                batch_capture_metadata.append({
                    "group_index": group_index, "hole_ids": group_ids,
                    "capture": group_metadata,
                })
                batch_coarse_results.update({
                    int(hole_id): result
                    for hole_id, result in group_results.items()
                    if isinstance(hole_id, int)
                })
                for hole_id in group_results:
                    if isinstance(hole_id, int):
                        batch_cache_transforms[int(hole_id)] = (
                            group_tcp.copy(), group_T_base_camera.copy(),
                        )
                accepted = [
                    int(hole_id) for hole_id, result in group_results.items()
                    if isinstance(hole_id, int) and result.get("success", False)
                ]
                group_report["accepted_holes"] = accepted
                group_report["fallback_holes"] = [
                    hole_id for hole_id in group_ids if hole_id not in accepted
                ]
                group_report["capture"] = group_metadata
                if group_report["fallback_holes"] and not all_selected_two_capture_mode:
                    print(
                        f"[BATCH_COARSE] 第{group_index}组失败孔将逐孔回退: "
                        f"{group_report['fallback_holes']}", flush=True,
                    )
            except Exception as exc:
                # 规划/移动/采集只影响当前视野组；其余组已经得到的结果和缓存
                # 继续保留，当前组的孔走后面的逐孔压轴兜底路径。
                group_report["error"] = f"{type(exc).__name__}:{exc}"
                group_report["fallback_reason"] = "group_failed"
                print(
                    f"[BATCH_COARSE] 第{group_index}组失败；"
                    + (
                        "全部选中孔不再拆分或逐孔补拍"
                        if all_selected_two_capture_mode else
                        "回退该组逐孔粗定位"
                    )
                    + f": {exc}",
                    flush=True,
                )
            batch_plan["groups"].append(group_report)

        success_count = sum(
            1 for result in batch_coarse_results.values()
            if result.get("success", False)
        )
        failed_holes = [
            int(hole["hole_id"]) for hole in initial_holes
            if not batch_coarse_results.get(int(hole["hole_id"]), {}).get("success", False)
        ]
        report["stages"]["batch_coarse_plan"] = batch_plan
        report["stages"]["batch_coarse_results"] = {
            "success_count": success_count,
            "total_count": len(initial_holes),
            "failed_holes": failed_holes,
            "group_count": len(batch_groups),
            "groups": batch_plan["groups"],
            "captures": batch_capture_metadata,
            "capture": (
                batch_capture_metadata[0].get("capture")
                if len(batch_capture_metadata) == 1 else
                {"groups": batch_capture_metadata}
            ),
            "results": {
                int(hole_id): {
                    "success": result.get("success", False),
                    "valid_frames": result.get("valid_frames", 0),
                    "error": result.get("error"),
                    "error_counts": result.get("error_counts", {}),
                    "pointcloud_image_path": result.get("pointcloud_image_path"),
                    "batch_pointcloud_archive_path": result.get("batch_pointcloud_archive_path"),
                }
                for hole_id, result in batch_coarse_results.items()
            },
        }
        print(
            f"[BATCH_COARSE] 完成: 成功{success_count}/{len(initial_holes)}个孔，"
            f"视野组{len(batch_groups)}组", flush=True,
        )

        # 一拍多成功结果是正式缓存的唯一新来源。缓存写入发生在共享
        # 340 mm采集完成后、逐孔精定位之前；逐孔回退不会覆盖这些条目。
        if cache_enabled and batch_coarse_results:
            batch_cache_entries: dict[int, CoarseCacheEntry] = {}
            for hole in initial_holes:
                hole_id = int(hole["hole_id"])
                batch_result = batch_coarse_results.get(hole_id, {})
                if not batch_result.get("success"):
                    continue
                try:
                    batch_cache_tcp, batch_cache_T_base_camera = batch_cache_transforms.get(
                        hole_id,
                        (
                            np.asarray(current_tcp, dtype=np.float64).copy(),
                            camera_transform(current_tcp, handeye.T_tcp_rgb_camera),
                        ),
                    )
                    entry = _cache_entry_from_observations(
                        hole_id,
                        list(batch_result.get("observations", [])),
                        T_base_camera=batch_cache_T_base_camera,
                        T_tcp_camera=handeye.T_tcp_rgb_camera,
                        tcp_pose_m_rad=transform_to_sdk_pose_m_rad(batch_cache_tcp),
                        camera_serial=str((report.get("camera") or {}).get("serial_number", "")),
                        handeye_path=str(args.handeye),
                        intrinsics=initial_intrinsics,
                        cfg=cfg,
                        min_valid_frames=int(coarse_cache_gates.min_valid_frames),
                        cache_source="batch_coarse_source",
                    )
                    batch_cache_entries[hole_id] = entry
                except Exception as exc:
                    report["coarse_cache"].setdefault("batch_cache_errors", {})[
                        str(hole_id)
                    ] = f"{type(exc).__name__}:{exc}"
            if batch_cache_entries:
                cache_entries.update(batch_cache_entries)
                cache_sources.update({
                    int(hole_id): "batch_coarse_source"
                    for hole_id in batch_cache_entries
                })
                cache_built_ids.update(batch_cache_entries)
                try:
                    save_cache_entries(
                        coarse_cache_dir,
                        cache_entries,
                        metadata={
                            "mode": "two_stage_hole_localization",
                            "source": "batch_coarse_source",
                            "handeye_path": str(args.handeye),
                            "camera_serial": str((report.get("camera") or {}).get("serial_number", "")),
                            "gates": coarse_cache_gates.to_dict(),
                            "surface_selection_policy": COARSE_SURFACE_SELECTION_POLICY,
                            "surface_model": COARSE_SURFACE_MODEL,
                        },
                    )
                    if persistent_enabled and persistent_cache_dir is not None:
                        for hole_id, entry in batch_cache_entries.items():
                            cache_source_ids[hole_id] = _upsert_persistent_cache_entry(
                                entry,
                                current_hole_id=hole_id,
                                persistent_entries=persistent_entries,
                                persistent_source_ids=cache_source_ids,
                            )
                        save_persistent_cache_entries(
                            persistent_cache_dir,
                            persistent_entries,
                            metadata={
                                "mode": "two_stage_hole_localization",
                                "source": "batch_coarse_source",
                                "handeye_path": str(args.handeye),
                                "camera_serial": str((report.get("camera") or {}).get("serial_number", "")),
                                "gates": coarse_cache_gates.to_dict(),
                                "surface_selection_policy": COARSE_SURFACE_SELECTION_POLICY,
                                "surface_model": COARSE_SURFACE_MODEL,
                            },
                        )
                except Exception as exc:
                    report["coarse_cache"]["cache_persist_error"] = (
                        f"{type(exc).__name__}:{exc}"
                    )
                report["coarse_cache"]["cache_built"] = sorted(cache_built_ids)
                report["coarse_cache"]["cache_available"] = sorted(cache_entries)
    else:
        report["stages"]["batch_coarse_plan"] = {
            "enabled": False,
            "reason": (
                "single_hole" if len(initial_holes) == 1
                else "batch_mode_disabled"
            ),
        }

    # Fixed-point-cloud reuse normally verifies each hole at its own 340 mm
    # pose.  This opt-in path validates any number of cached holes in dynamic
    # shared-view groups, then lets only rejected holes use that strict path.
    shared_cache_results: dict[int, dict[str, Any]] = {}
    shared_cache_failed_ids: set[int] = set()
    invalidated_cache_ids: set[int] = set()
    # 缓存策略默认就是“340 mm一拍多验证”；命令行开关仅保留为兼容旧
    # 调用方的显式开启方式。批量粗定位模式本身优先使用新采集结果。
    shared_cache_requested = bool(
        not all_selected_two_capture_mode
        and (
            getattr(args, "shared_cache_validation", False)
            or (cache_enabled and bool(cache_entries) and not cfg.batch_coarse_localization)
        )
    )
    shared_cache_allowed = (
        shared_cache_requested and cache_enabled and not batch_coarse_for_cache
    )
    report["stages"]["shared_cache_validation"] = {
        "requested": shared_cache_requested,
        "enabled": shared_cache_allowed,
        "reason": (
            "enabled" if shared_cache_allowed else
            "cache_disabled" if not cache_enabled else
            "incompatible_with_batch"
        ),
        "groups": [],
        "holes": {},
    }
    if shared_cache_allowed:
        initial_holes_by_id = {int(item["hole_id"]): item for item in initial_holes}
        # Group planning and association anchors deliberately use the trusted
        # cache geometry, not a possibly shifted initial selection estimate.
        cached_holes = []
        for hole in initial_holes:
            hole_id = int(hole["hole_id"])
            entry = cache_entries.get(hole_id)
            if entry is None:
                continue
            cached_holes.append({
                **hole,
                "initial_center_base_mm": np.asarray(entry.point_base_mm, dtype=np.float64),
                "initial_plane_normal_base": np.asarray(entry.normal_base, dtype=np.float64),
            })
        groups = _split_shared_cache_validation_groups(
            cached_holes, current_tcp, handeye, fixed_rz_rad, initial_intrinsics,
            cfg.coarse_height_mm, cfg.shared_cache_validation_view_margin_px,
        )
        rgbd_pipeline, align, chain = ensure_rgbd_pipeline()
        for group_index, group in enumerate(groups, start=1):
            group_ids = [int(hole["hole_id"]) for hole in group]
            group_report: dict[str, Any] = {
                "group_index": group_index, "hole_ids": group_ids,
                "target_height_mm": float(cfg.coarse_height_mm), "accepted_holes": [],
                "fallback_holes": list(group_ids),
            }
            try:
                target, geometry = _plan_batch_coarse_group_pose(
                    group, current_tcp, handeye, fixed_rz_rad, initial_intrinsics,
                    cfg.coarse_height_mm, cfg.shared_cache_validation_view_margin_px,
                )
                group_report["target_tcp_pose_m_rad"] = geometry["target_tcp_pose_m_rad"]
                group_report["projected_holes_px"] = {
                    str(key): value.tolist() for key, value in geometry["projected_holes_px"].items()
                }
                with timing.measure(
                    f"shared_cache/group_{group_index:02d}/navigate_to_340mm",
                    group_index=group_index, hole_count=len(group),
                ):
                    current_tcp = _confirm_and_move_line(
                        f"固定点云快速验证：第{group_index}组 {len(group)}孔共同340mm观察位",
                        current_tcp, target, args, motion_session, pose_session,
                        "共享340mm少帧RGB-D验证；未通过孔将逐孔完整粗定位",
                        require_confirmation=False, motion_profile="transit",
                    )
                validation_cfg = replace(
                    cfg,
                    batch_coarse_frames=int(cfg.shared_cache_validation_frames),
                    batch_coarse_min_valid=int(cfg.shared_cache_validation_min_valid),
                    # 共享验证仍在同一个已停稳的 340 mm 位姿采集，但缓存是否
                    # 可复用必须按孔判断。一个孔漏检不能丢弃同帧其他孔的有效
                    # 观测，否则会把局部失败扩大为整组逐孔 340 mm 回退。
                    batch_coarse_min_holes_per_frame=1,
                    batch_coarse_settle_discard_frames=int(
                        cfg.cache_validation_settle_discard_frames
                    ),
                )
                with timing.measure(
                    f"shared_cache/group_{group_index:02d}/capture_and_validate",
                    group_index=group_index, hole_count=len(group),
                    target_frames=int(cfg.shared_cache_validation_frames),
                ):
                    observations_by_hole = _batch_coarse_localization_at_340mm(
                        group, current_tcp, handeye, rgbd_pipeline, align, chain, model,
                        args.confidence, validation_cfg, initial_intrinsics, run_dir, timing, rows,
                        artifact_prefix=f"shared_cache_group_{group_index:02d}",
                    )
                metadata = observations_by_hole.pop("_batch_metadata", {})
                group_report["capture"] = metadata
                T_base_camera = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
                accepted: list[int] = []
                for hole in group:
                    hole_id = int(hole["hole_id"])
                    result = observations_by_hole.get(hole_id, {})
                    entry = cache_entries[hole_id]
                    # A shared 340 mm view deliberately leaves most holes away
                    # from the optical principal point.  Keep the cache gate's
                    # strict offset threshold, but anchor it to this hole's
                    # trusted cached geometry projected into the shared pose.
                    # The local intrinsics copy is only consumed by
                    # validate_cache_entry's center-offset calculation.
                    expected_anchor = _project_base_point_to_pixel(
                        entry.point_base_mm, T_base_camera, initial_intrinsics,
                    )
                    validation_intrinsics = replace(
                        initial_intrinsics,
                        cx=float(expected_anchor[0]),
                        cy=float(expected_anchor[1]),
                    )
                    validation = validate_cache_entry(
                        entry,
                        _cache_measurements_from_observations(result.get("observations", [])),
                        current_T_base_camera=T_base_camera, intrinsics=validation_intrinsics,
                        gates=replace(
                            cache_gates,
                            validation_frames=int(cfg.shared_cache_validation_frames),
                            min_valid_frames=int(cfg.shared_cache_validation_min_valid),
                        ),
                    )
                    item = {
                        "accepted": bool(validation.accepted),
                        "expected_anchor_px": expected_anchor.tolist(),
                        "validation": validation.to_dict(),
                    }
                    report["stages"]["shared_cache_validation"]["holes"][str(hole_id)] = item
                    if not validation.accepted:
                        shared_cache_failed_ids.add(hole_id)
                        invalidated_cache_ids.add(hole_id)
                        report["coarse_cache"]["cache_invalidated"].append({
                            "hole_id": hole_id,
                            "reason": f"shared_cache_validation_rejected:{validation.reason}",
                        })
                        continue
                    _apply_cached_geometry_to_hole(
                        initial_holes_by_id[hole_id], entry, validation,
                        source=cache_sources.get(hole_id, "current_run_initial_cache"),
                        persistent_hole_id=cache_source_ids.get(hole_id),
                    )
                    shared_cache_results[hole_id] = {
                        "entry": entry, "validation": validation,
                        "observations": result.get("observations", []),
                        "group_index": group_index, "source": cache_sources.get(hole_id),
                    }
                    accepted.append(hole_id)
                group_report["accepted_holes"] = accepted
                group_report["fallback_holes"] = [hole_id for hole_id in group_ids if hole_id not in accepted]
            except Exception as exc:
                group_report["error"] = f"{type(exc).__name__}:{exc}"
                for hole_id in group_ids:
                    shared_cache_failed_ids.add(int(hole_id))
                    invalidated_cache_ids.add(int(hole_id))
                    report["coarse_cache"]["cache_invalidated"].append({
                        "hole_id": int(hole_id),
                        "reason": f"shared_cache_validation_error:{group_report['error']}",
                    })
                    report["stages"]["shared_cache_validation"]["holes"][str(hole_id)] = {
                        "accepted": False, "reason": group_report["error"],
                    }
            report["stages"]["shared_cache_validation"]["groups"].append(group_report)

    # 260 mm 批量精定位：所有已选孔优先复用本轮初始选择RGB-D帧中已经
    # 计算完成的逐孔点云几何。跨运行缓存是否匹配不再影响选中孔进入精拍。
    # 这些字段已在进入流程时统一建立；此处只消费，不再触发任何采集。
    batch_fine_results: dict[int, dict[str, Any]] = {}
    batch_fine_plan: dict[str, Any] = {
        "enabled": False,
        "requested": bool(cfg.batch_fine_localization and len(initial_holes) > 1),
        "hole_count": len(initial_holes),
        "groups": [],
        "fallback_holes": [],
        "initial_pointcloud_reused_holes": initial_pointcloud_reused_holes,
        "coarse_geometry_policy": "fresh_shared_340mm_capture_required",
        "failure_policy": "move_to_shared_260mm_supplement_capture",
        "combined_position_policy": "projected_bbox_center_from_all_coarse_holes",
        "motion_policy": "shared_vertical_lift_min10_safe_horizontal_descent_guard10_then_pure_descent10_then_steady_capture",
        "final_pose_policy": "per_hole_coarse_z_and_orientation_with_batch_fine_xy_only",
        "fine_output_components": ["base_x", "base_y"],
        "coarse_output_components": ["base_z", "rx", "ry", "rz"],
        "supplement_rounds": int(cfg.batch_fine_supplement_rounds),
    }
    if cfg.batch_fine_localization and len(initial_holes) > 1:
        fine_planning_holes: list[dict[str, Any]] = []
        fine_planning_holes_by_id: dict[int, dict[str, Any]] = {}
        missing_fine_geometry: list[int] = []
        for hole in initial_holes:
            hole_id = int(hole["hole_id"])
            coarse_result = batch_coarse_results.get(hole_id, {})
            if all_selected_two_capture_mode:
                # 正式两拍流程必须使用同一次340mm拍摄的结果来规划260mm；
                # 初始选孔点云不能在粗定位漏检时悄悄顶替。
                point_value = (
                    coarse_result.get("center_base_mm")
                    if coarse_result.get("success") else None
                )
                normal_value = (
                    coarse_result.get("normal_base")
                    if coarse_result.get("success") else None
                )
            else:
                point_value = hole.get("coarse_center_base_mm")
                normal_value = hole.get("coarse_normal_toward_camera_base")
                if point_value is None and coarse_result.get("success"):
                    point_value = coarse_result.get("center_base_mm")
                if normal_value is None and coarse_result.get("success"):
                    normal_value = coarse_result.get("normal_base")
            if point_value is None or normal_value is None:
                missing_fine_geometry.append(hole_id)
                continue
            fine_planning_holes.append({
                **hole,
                # 共同260mm位姿规划器复用已有的批量视野算法；这里把
                # 粗定位后的可信几何映射到规划器要求的通用字段。
                "initial_center_base_mm": np.asarray(point_value, dtype=np.float64).copy(),
                "initial_plane_normal_base": np.asarray(normal_value, dtype=np.float64).copy(),
                "planning_normal_base": np.asarray(normal_value, dtype=np.float64).copy(),
            })
        fine_planning_holes_by_id = {
            int(hole["hole_id"]): hole for hole in fine_planning_holes
        }

        if all_selected_two_capture_mode and missing_fine_geometry:
            # 两次共享拍摄都必须覆盖同一套完整选孔。粗定位少一个孔时，
            # 不能拿剩余子集重新计算所谓“综合位置”并继续精拍。
            fine_planning_holes = []

        batch_fine_plan.update({
            "hole_count_with_coarse_geometry": len(fine_planning_holes),
            "missing_coarse_geometry_holes": missing_fine_geometry,
            "view_margin_px": float(cfg.batch_fine_view_margin_px),
            "target_height_mm": float(cfg.fine_height_mm),
            "frames": int(cfg.batch_fine_frames),
            "min_valid_frames": int(cfg.batch_fine_min_valid),
            "stable_min_frames": int(cfg.batch_fine_stable_min_frames),
            "settle_discard_frames": int(cfg.batch_fine_settle_discard_frames),
            "supplement_rounds": int(cfg.batch_fine_supplement_rounds),
        })
        if fine_planning_holes:
            batch_fine_groups = (
                [list(fine_planning_holes)]
                if all_selected_two_capture_mode else
                _split_shared_cache_validation_groups(
                    fine_planning_holes, current_tcp, handeye, fixed_rz_rad,
                    initial_intrinsics, cfg.fine_height_mm, cfg.batch_fine_view_margin_px,
                )
            )
            batch_fine_plan.update({
                "enabled": True,
                "group_count": len(batch_fine_groups),
                "all_selected_holes_single_group": bool(all_selected_two_capture_mode),
            })
            rgbd_pipeline, align, chain = ensure_rgbd_pipeline()
            fine_hole_groups_by_id: dict[int, int] = {}

            def _summarize_batch_fine_results(
                values: dict[Any, dict[str, Any]],
            ) -> dict[str, dict[str, Any]]:
                return {
                    str(hole_id): {
                        "success": bool(result.get("success", False)),
                        "error": result.get("error"),
                        "batch_fine_source": result.get("batch_fine_source"),
                        "capture_round": result.get("batch_fine_capture_round", 0),
                        "fine_quality_status": (
                            result.get("fine") or {}
                        ).get("fine_quality_status"),
                        "valid_frames": (
                            result.get("fine") or {}
                        ).get("valid_frames", 0),
                    }
                    for hole_id, result in values.items()
                    if isinstance(hole_id, int)
                }

            for group_index, group in enumerate(batch_fine_groups, start=1):
                group_ids = [int(hole["hole_id"]) for hole in group]
                group_report: dict[str, Any] = {
                    "group_index": group_index,
                    "hole_ids": group_ids,
                    "hole_count": len(group),
                    "accepted_holes": [],
                    "fallback_holes": list(group_ids),
                    "supplement_captures": [],
                }
                for hole_id in group_ids:
                    fine_hole_groups_by_id[hole_id] = group_index
                try:
                    batch_target, batch_geometry = _plan_batch_coarse_group_pose(
                        group, current_tcp, handeye, fixed_rz_rad,
                        initial_intrinsics, cfg.fine_height_mm,
                        cfg.batch_fine_view_margin_px,
                    )
                    group_report.update({
                        "target_tcp_pose_m_rad": batch_geometry["target_tcp_pose_m_rad"],
                        "combined_point_base_mm": batch_geometry.get("group_point_base_mm"),
                        "combined_normal_base": batch_geometry.get(
                            "group_normal_toward_camera_base"
                        ),
                        "pose_policy": "above_projected_bbox_center_of_all_coarse_holes",
                        "projected_holes_px": {
                            str(key): value.tolist()
                            for key, value in batch_geometry["projected_holes_px"].items()
                        },
                        "group_bbox_px": batch_geometry["group_bbox_px"],
                        "group_center_px": batch_geometry["group_center_px"].tolist(),
                    })
                    group_results: dict[Any, dict[str, Any]] = {}
                    initial_capture_error: str | None = None
                    with timing.measure(
                        f"batch_fine/group_{group_index:02d}/navigate_to_260mm",
                        hole_count=len(group), group_index=group_index,
                    ):
                        current_tcp = _move_to_shared_fine_pose(
                            f"fine_group_{group_index:02d}", group_index,
                            current_tcp, batch_target, args, motion_session, pose_session,
                            target_height_mm=cfg.fine_height_mm,
                            target_stage="批量精定位",
                        )
                    try:
                        with timing.measure(
                            f"batch_fine/group_{group_index:02d}/capture_and_localize",
                            hole_count=len(group), group_index=group_index,
                        ):
                            group_results = _batch_fine_localization_at_260mm(
                                group, current_tcp, handeye, rgbd_pipeline, model,
                                args.confidence, cfg, initial_intrinsics, run_dir,
                                timing, rows,
                                artifact_prefix=f"batch_fine_group_{group_index:02d}",
                            )
                    except Exception as exc:
                        initial_capture_error = f"{type(exc).__name__}:{exc}"
                        group_report["initial_capture_error"] = initial_capture_error
                        print(
                            f"[BATCH_FINE] 第{group_index}组首拍失败，准备共享补拍: "
                            f"{initial_capture_error}",
                            flush=True,
                        )
                    group_metadata = group_results.pop("_batch_metadata", {})
                    for hole_id, result in group_results.items():
                        if isinstance(hole_id, int):
                            result.setdefault("batch_fine_source", "batch_fine_at_260mm")
                            result.setdefault("batch_fine_capture_round", 0)
                    batch_fine_results.update({
                        int(hole_id): result
                        for hole_id, result in group_results.items()
                        if isinstance(hole_id, int)
                    })
                    if initial_capture_error is not None and not group_results:
                        for hole_id in group_ids:
                            batch_fine_results[hole_id] = {
                                "success": False,
                                "error": initial_capture_error,
                                "batch_fine_source": "batch_fine_at_260mm",
                                "batch_fine_capture_round": 0,
                            }
                    group_report["capture"] = group_metadata
                    group_report["initial_results"] = _summarize_batch_fine_results(
                        group_results
                    )

                    # 首拍中只有质量门失败的孔进入补拍。补拍仍以失败孔集合
                    # 规划共同260mm观察位，因此不会退化为逐孔精定位。
                    pending_holes = [
                        hole_id for hole_id in group_ids
                        if not batch_fine_results.get(hole_id, {}).get("success", False)
                    ]
                    for supplement_round in range(
                        1, max(0, int(cfg.batch_fine_supplement_rounds)) + 1
                    ):
                        if not pending_holes:
                            break
                        supplement_group = [
                            fine_planning_holes_by_id[hole_id]
                            for hole_id in pending_holes
                            if hole_id in fine_planning_holes_by_id
                        ]
                        supplement_report: dict[str, Any] = {
                            "round": supplement_round,
                            "hole_ids": list(pending_holes),
                            "hole_count": len(supplement_group),
                            "accepted_holes": [],
                            "fallback_holes": list(pending_holes),
                        }
                        if not supplement_group:
                            supplement_report["error"] = (
                                "补拍缺少可用粗定位几何，无法规划共享260mm补拍位"
                            )
                            group_report["supplement_captures"].append(supplement_report)
                            break
                        try:
                            supplement_target, supplement_geometry = (
                                _plan_batch_coarse_group_pose(
                                    supplement_group, current_tcp, handeye, fixed_rz_rad,
                                    initial_intrinsics, cfg.fine_height_mm,
                                    cfg.batch_fine_view_margin_px,
                                )
                            )
                            supplement_report.update({
                                "target_tcp_pose_m_rad": supplement_geometry[
                                    "target_tcp_pose_m_rad"
                                ],
                                "combined_point_base_mm": supplement_geometry.get(
                                    "group_point_base_mm"
                                ),
                                "combined_normal_base": supplement_geometry.get(
                                    "group_normal_toward_camera_base"
                                ),
                                "pose_policy": (
                                    "above_projected_bbox_center_of_failed_coarse_holes"
                                ),
                                "projected_holes_px": {
                                    str(key): value.tolist()
                                    for key, value in supplement_geometry[
                                        "projected_holes_px"
                                    ].items()
                                },
                                "group_bbox_px": supplement_geometry["group_bbox_px"],
                                "group_center_px": supplement_geometry[
                                    "group_center_px"
                                ].tolist(),
                            })
                            with timing.measure(
                                f"batch_fine/group_{group_index:02d}/supplement_{supplement_round:02d}/navigate_to_260mm",
                                hole_count=len(supplement_group),
                                group_index=group_index,
                                supplement_round=supplement_round,
                            ):
                                current_tcp = _move_to_shared_fine_pose(
                                    f"fine_group_{group_index:02d}_supplement_{supplement_round:02d}",
                                    group_index,
                                    current_tcp,
                                    supplement_target,
                                    args,
                                    motion_session,
                                    pose_session,
                                    target_height_mm=cfg.fine_height_mm,
                                    target_stage="共享精定位补拍",
                                )
                            with timing.measure(
                                f"batch_fine/group_{group_index:02d}/supplement_{supplement_round:02d}/capture_and_localize",
                                hole_count=len(supplement_group),
                                group_index=group_index,
                                supplement_round=supplement_round,
                            ):
                                supplement_results = _batch_fine_localization_at_260mm(
                                    supplement_group,
                                    current_tcp,
                                    handeye,
                                    rgbd_pipeline,
                                    model,
                                    args.confidence,
                                    cfg,
                                    initial_intrinsics,
                                    run_dir,
                                    timing,
                                    rows,
                                    artifact_prefix=(
                                        f"batch_fine_group_{group_index:02d}"
                                        f"_supplement_{supplement_round:02d}"
                                    ),
                                )
                            supplement_metadata = supplement_results.pop(
                                "_batch_metadata", {}
                            )
                            for hole_id, result in supplement_results.items():
                                if not isinstance(hole_id, int):
                                    continue
                                result["batch_fine_source"] = (
                                    "batch_fine_supplement_at_260mm"
                                )
                                result["batch_fine_capture_round"] = supplement_round
                                batch_fine_results[int(hole_id)] = result
                            supplement_report["capture"] = supplement_metadata
                            supplement_report["results"] = (
                                _summarize_batch_fine_results(supplement_results)
                            )
                            supplement_report["accepted_holes"] = [
                                int(hole_id) for hole_id, result in supplement_results.items()
                                if isinstance(hole_id, int)
                                and result.get("success", False)
                            ]
                            pending_holes = [
                                hole_id for hole_id in group_ids
                                if not batch_fine_results.get(hole_id, {}).get(
                                    "success", False
                                )
                            ]
                            supplement_report["fallback_holes"] = list(pending_holes)
                            print(
                                f"[BATCH_FINE] 第{group_index}组第{supplement_round}轮"
                                f"共享补拍完成；accepted={supplement_report['accepted_holes']} "
                                f"fallback={pending_holes}",
                                flush=True,
                            )
                        except Exception as exc:
                            supplement_report["error"] = (
                                f"{type(exc).__name__}:{exc}"
                            )
                            supplement_report["fallback_reason"] = (
                                "batch_fine_supplement_failed"
                            )
                            for hole_id in pending_holes:
                                previous = dict(batch_fine_results.get(hole_id, {}))
                                previous.update({
                                    "success": False,
                                    "error": supplement_report["error"],
                                    "batch_fine_source": (
                                        "batch_fine_supplement_at_260mm"
                                    ),
                                    "batch_fine_capture_round": supplement_round,
                                })
                                batch_fine_results[hole_id] = previous
                            print(
                                f"[BATCH_FINE] 第{group_index}组第{supplement_round}轮"
                                f"共享补拍失败: {supplement_report['error']}",
                                flush=True,
                            )
                        group_report["supplement_captures"].append(supplement_report)

                    group_report["accepted_holes"] = [
                        hole_id for hole_id in group_ids
                        if batch_fine_results.get(hole_id, {}).get("success", False)
                    ]
                    group_report["fallback_holes"] = [
                        hole_id for hole_id in group_ids
                        if hole_id not in group_report["accepted_holes"]
                    ]
                    group_report["results"] = _summarize_batch_fine_results({
                        hole_id: batch_fine_results[hole_id]
                        for hole_id in group_ids
                        if hole_id in batch_fine_results
                    })
                except Exception as exc:
                    group_report["error"] = f"{type(exc).__name__}:{exc}"
                    group_report["fallback_reason"] = "batch_fine_group_failed"
                    for hole_id in group_ids:
                        batch_fine_results[hole_id] = {
                            "success": False,
                            "error": group_report["error"],
                            "batch_fine_source": "batch_fine_group_failed",
                            "batch_fine_capture_round": 0,
                        }
                    print(
                        f"[BATCH_FINE] 第{group_index}组失败，无法规划共享补拍: {exc}",
                        flush=True,
                    )
                batch_fine_plan["groups"].append(group_report)

            batch_fine_plan["fallback_holes"] = sorted({
                *missing_fine_geometry,
                *[
                    hole_id for group in batch_fine_plan["groups"]
                    for hole_id in group.get("fallback_holes", [])
                ],
            })
            batch_fine_plan["hole_group_indices"] = {
                str(hole_id): group_index
                for hole_id, group_index in fine_hole_groups_by_id.items()
            }
        else:
            batch_fine_plan["fallback_holes"] = missing_fine_geometry
            batch_fine_plan["reason"] = (
                "shared_coarse_capture_incomplete_abort_fine_capture"
                if all_selected_two_capture_mode and missing_fine_geometry else
                "no_hole_has_reliable_coarse_geometry"
            )
    elif len(initial_holes) <= 1:
        batch_fine_plan["reason"] = "single_hole"
    else:
        batch_fine_plan["reason"] = "batch_fine_disabled"

    report["stages"]["batch_fine_plan"] = batch_fine_plan
    report["stages"]["batch_fine_results"] = {
        "success_count": sum(
            1 for result in batch_fine_results.values() if result.get("success", False)
        ),
        "total_count": len(initial_holes),
        "fallback_holes": batch_fine_plan.get("fallback_holes", []),
        "group_count": len(batch_fine_plan.get("groups", [])),
        "groups": batch_fine_plan.get("groups", []),
    }

    if bool(getattr(args, "optimize_hole_order", False)) and batch_coarse_results:
        original_order = list(order_ids)
        targets = {
            hole_id: np.asarray(result["center_base_mm"], dtype=np.float64)[:2]
            for hole_id, result in batch_coarse_results.items()
            if isinstance(hole_id, int) and result.get("success") and result.get("center_base_mm") is not None
        }
        optimized_order, estimated_distance_mm = optimize_hole_order(
            original_order, targets, start_xy=np.asarray(current_tcp, dtype=np.float64)[:2, 3],
        )
        if optimized_order != original_order:
            by_id = {int(item["hole_id"]): item for item in initial_holes}
            initial_holes[:] = [by_id[hole_id] for hole_id in optimized_order]
            order_ids = optimized_order
        report["stages"]["hole_order_optimization"] = {
            "enabled": True,
            "original_order": original_order,
            "optimized_order": order_ids,
            "estimated_inter_hole_distance_mm": estimated_distance_mm,
        }
        report["stages"]["sequential_plan"]["hole_order"] = order_ids
    else:
        report["stages"]["hole_order_optimization"] = {
            "enabled": bool(getattr(args, "optimize_hole_order", False)),
            "original_order": list(order_ids), "optimized_order": list(order_ids),
            "estimated_inter_hole_distance_mm": 0.0,
        }

    for order, hole in enumerate(initial_holes, start=1):
        hole_id = int(hole["hole_id"])
        chosen = hole["initial_detection"]
        current_selection_point_base = np.asarray(
            hole["initial_center_base_mm"], dtype=np.float64,
        ).reshape(3)
        current_selection_normal_base = _unit(
            np.asarray(hole["initial_plane_normal_base"], dtype=np.float64),
            f"hole {hole_id} initial normal",
        )
        point_base = current_selection_point_base.copy()
        normal_base = current_selection_normal_base.copy()
        hole["tracking_identity"] = f"initial_selection_hole_{hole_id}"
        hole["processing_order"] = order

        # 缓存候选已经通过相机/手眼/内参兼容性检查，先用缓存几何导航到
        # 340 mm，再做少量现场验证。验证失败时仍回退到本轮初始几何，
        # 因此工件重新装夹不会直接复用未经验证的旧点云。
        navigation_point_base = point_base.copy()
        navigation_normal_base = normal_base.copy()
        navigation_source = "current_initial_selection"
        if (
            cache_enabled
            and hole_id in cache_entries
            and hole_id not in shared_cache_failed_ids
        ):
            cache_candidate = cache_entries[hole_id]
            navigation_point_base = np.asarray(
                cache_candidate.point_base_mm, dtype=np.float64,
            ).reshape(3)
            navigation_normal_base = _unit(
                np.asarray(cache_candidate.normal_base, dtype=np.float64),
                f"hole {hole_id} cached navigation normal",
            )
            navigation_source = (
                "persistent_cache_candidate"
                if cache_sources.get(hole_id) == "persistent_base_cache"
                else "current_run_cache_candidate"
            )
            hole["coarse_cache_reference_point_base_mm"] = navigation_point_base.copy()
        hole["coarse_navigation_source"] = navigation_source
        hole["coarse_navigation_point_base_mm"] = navigation_point_base.copy()
        hole["coarse_navigation_normal_base"] = navigation_normal_base.copy()

        # 检查是否可以使用批量粗定位结果
        batch_coarse_available = (
            hole_id in batch_coarse_results
            and batch_coarse_results[hole_id].get("success", False)
        )
        shared_cache_available = hole_id in shared_cache_results
        batch_fine_available = (
            hole_id in batch_fine_results
            and batch_fine_results[hole_id].get("success", False)
        )
        batch_fine_result = batch_fine_results.get(hole_id, {})
        initial_pointcloud_reuse_available = bool(
            hole.get("coarse_source") == "initial_selection_shared_pointcloud_reuse"
            and hole.get("coarse_center_base_mm") is not None
            and hole.get("coarse_normal_toward_camera_base") is not None
        )
        coarse_settle_delay_s = max(0.0, float(cfg.coarse_settle_delay_s))
        coarse_captures: list[dict[str, Any]] = []
        final_center_offset: float | None = None
        final_normal_error: float | None = None
        last_coarse_observations: list[Observation] | None = None
        last_coarse_T_base_camera: np.ndarray | None = None

        # 批量模式的正式RGB图像已经在进入逐孔结果计算前统一拍完；首拍失败
        # 的孔会在此前的共享精拍阶段尝试移动到补拍位。到这里仍失败时才延后，
        # 不再悄悄转成逐孔精拍。
        if (
            cfg.batch_fine_localization
            and len(initial_holes) > 1
            and all_selected_two_capture_mode
            and not batch_fine_available
        ):
            group_index = batch_fine_plan.get("hole_group_indices", {}).get(str(hole_id))
            group_error = next((
                group.get("error")
                for group in batch_fine_plan.get("groups", [])
                if hole_id in group.get("hole_ids", []) and group.get("error")
            ), None)
            failure_reason = (
                batch_fine_result.get("error")
                or group_error
                or "selected_hole_missing_from_single_batch_fine_frame"
            )
            deferred_result = {
                "status": "deferred_batch_fine",
                "hole_id": hole_id,
                "processing_order": order,
                "tracking_identity": hole["tracking_identity"],
                "initial_selection_order": hole.get("initial_selection_order"),
                "initial_center_px": hole.get("initial_center_px"),
                "initial_center_base_mm": hole.get("initial_center_base_mm"),
                "initial_plane_normal_base": hole.get("initial_plane_normal_base"),
                "initial_pointcloud_reused_for_batch_fine": bool(
                    hole.get("initial_pointcloud_reused_for_batch_fine")
                ),
                "coarse_source": hole.get("coarse_source"),
                "coarse_center_base_mm": hole.get("coarse_center_base_mm"),
                "coarse_plane_point_base_mm": hole.get("coarse_plane_point_base_mm"),
                "coarse_normal_toward_camera_base": hole.get(
                    "coarse_normal_toward_camera_base"
                ),
                "batch_fine_requested": True,
                "batch_fine_source": str(
                    batch_fine_result.get(
                        "batch_fine_source", "shared_batch_fine_failed"
                    )
                ),
                "batch_fine_capture_round": int(
                    batch_fine_result.get("batch_fine_capture_round", 0)
                ),
                "batch_fine_group_index": group_index,
                "fine_quality_status": "deferred_batch_fine",
                "fine_quality_note": failure_reason,
                "deferred_reason": failure_reason,
                "final_xy_motion": None,
                "final_z_motion": None,
                "final_y_trim_motion": None,
                "tracking_events": hole.get("tracking_events", []),
                "initial_detection": hole.get("initial_detection"),
                "timing": timing.scoped_snapshot(f"hole_{hole_id:02d}/"),
            }
            deferred_result["comparison_diagnostics"] = _build_comparison_hole_diagnostics(
                deferred_result
            )
            hole["final_result"] = deferred_result
            results.append(deferred_result)
            report["stages"][f"hole_{hole_id}"] = deferred_result
            report.setdefault("deferred_holes", []).append(hole_id)
            report["stages"]["processed_holes"] = {
                "completed_count": sum(item.get("status") == "completed" for item in results),
                "deferred_count": sum(item.get("status") != "completed" for item in results),
                "total_count": len(results),
                "hole_order": [int(item["hole_id"]) for item in results],
                "holes": results,
            }
            _write_report(run_dir, report, rows)
            print(
                f"[BATCH_FINE] hole={hole_id} status=deferred_batch_fine；"
                f"单拍失败且禁止逐孔补拍：{failure_reason}",
                flush=True,
            )
            continue

        if batch_coarse_available:
            # 使用批量粗定位结果，跳过逐孔粗定位流程
            print(
                f"[SEQUENTIAL_HOLE] order={order} hole={hole_id} "
                f"使用批量粗定位结果，跳过逐孔340mm采集",
                flush=True,
            )

            batch_result = batch_coarse_results[hole_id]

            # 将批量粗定位结果写入hole字典
            hole.update({
                "coarse_center_px": batch_result["center_px"],
                "coarse_center_camera_mm": batch_result["center_camera_mm"],
                "coarse_center_base_mm": batch_result["center_base_mm"],
                "coarse_plane_point_camera_mm": batch_result["plane_point_camera_mm"],
                "coarse_plane_point_base_mm": batch_result["plane_point_base_mm"],
                "coarse_normal_camera": batch_result["normal_camera"],
                "coarse_normal_toward_camera_base": batch_result["normal_base"],
                "coarse_plane_rmse_mm": batch_result["plane_rmse_mm"],
                "coarse_valid_frames": batch_result["valid_frames"],
                "coarse_total_frames": batch_result["total_frames"],
                "coarse_center_scatter_p95_px": batch_result["center_scatter_p95_px"],
                "coarse_tracking_distance_p95_px": batch_result.get("tracking_distance_p95_px"),
                "coarse_ring_points_median": batch_result.get("ring_points_median"),
                "coarse_captures": batch_result["coarse_captures"],
                "coarse_surface_model": batch_result.get("surface_model"),
                "coarse_surface_selection_policy": batch_result.get("surface_selection_policy"),
                "coarse_front_surface_z_mm": batch_result.get("front_surface_z_mm"),
                "coarse_ring_points_raw_median": batch_result.get("ring_points_raw_median"),
                "coarse_surface_points_selected_median": batch_result.get("surface_points_selected_median"),
                "coarse_sphere_center_camera_mm": batch_result.get("sphere_center_camera_mm"),
                "coarse_sphere_radius_mm": batch_result.get("sphere_radius_mm"),
                "coarse_source": "batch_coarse_localization",
                "batch_observed_center_base_mm": batch_result.get("batch_observed_center_base_mm"),
                "batch_observed_plane_point_base_mm": batch_result.get("batch_observed_plane_point_base_mm"),
                "batch_observed_normal_base": batch_result.get("batch_observed_normal_base"),
                "coarse_pointcloud_image_path": batch_result.get("pointcloud_image_path"),
                "batch_pointcloud_archive_path": batch_result.get("batch_pointcloud_archive_path"),
            })
            coarse_captures = list(batch_result["coarse_captures"])

            # 不需要单独导航到该孔的340mm位置，直接从批量粗定位共同位姿导航到260mm精定位位
            coarse_plane_base = np.asarray(
                hole["coarse_plane_point_base_mm"], dtype=np.float64,
            ).reshape(3)
            coarse_normal_base = _unit(
                np.asarray(hole["coarse_normal_toward_camera_base"], dtype=np.float64),
                f"hole {hole_id} coarse normal from batch",
            )

            hole["coarse_plane_point_base_mm"] = coarse_plane_base
            hole["coarse_normal_toward_camera_base"] = coarse_normal_base

            # 初始化批量粗定位路径使用的变量
            cache_event: dict[str, Any] = {
                "enabled": False,
                "hole_id": hole_id,
                "cache_available": False,
                "cache_source": "batch_coarse_localization",
                "cache_reused": False,
                "cache_validation_skipped": True,
                "cache_validation_failed": False,
                "full_coarse_fallback": False,
                "coarse_settle_buffer_s": 0.0,
                "navigation_source": "batch_coarse_group_pose",
                "navigation_point_base_mm": batch_result["center_base_mm"],
            }
            cache_reused = False

            rgbd_pipeline, align, chain = ensure_rgbd_pipeline()
            if not batch_fine_available:
                # 没有成功的共同260mm批量结果时，保留原有逐孔精定位兜底。
                batch_fine_target, batch_fine_geometry = _plan_hole_tcp_pose_fixed_rz(
                    np.asarray(hole["coarse_center_base_mm"], dtype=np.float64),
                    np.asarray(hole["coarse_normal_toward_camera_base"], dtype=np.float64),
                    current_tcp,
                    handeye.T_tcp_rgb_camera,
                    fixed_rz_rad=fixed_rz_rad,
                    camera_height_mm=cfg.fine_height_mm,
                )
                hole["batch_fine_target_tcp_pose_m_rad"] = transform_to_sdk_pose_m_rad(
                    batch_fine_target
                )
                hole["batch_fine_pose_geometry"] = batch_fine_geometry
                with timing.measure(
                    f"hole_{hole_id:02d}/batch_move_to_fine_pose",
                    hole_id=hole_id,
                    processing_order=order,
                ):
                    current_tcp = _move_to_fine_pose(
                        str(hole_id),
                        order,
                        current_tcp,
                        batch_fine_target,
                        args,
                        motion_session,
                        pose_session,
                        target_height_mm=cfg.fine_height_mm,
                        target_stage="批量粗定位后的单孔精定位兜底",
                    )
            else:
                hole["batch_fine_localization_source"] = str(
                    batch_fine_result.get("batch_fine_source", "batch_fine_at_260mm")
                )
                hole["batch_fine_capture_tcp_pose_m_rad"] = transform_to_sdk_pose_m_rad(
                    batch_fine_result["capture_tcp"]
                )

        elif shared_cache_available:
            # Shared validation already completed the only 340 mm observation.
            shared = shared_cache_results[hole_id]
            validation = shared["validation"]
            cache_event = {
                "enabled": True, "hole_id": hole_id, "cache_available": True,
                "cache_source": shared.get("source"),
                "persistent_hole_id": cache_source_ids.get(hole_id),
                "cache_reused": True, "cache_validation_skipped": False,
                "cache_validation_failed": False, "full_coarse_fallback": False,
                "shared_cache_validation": True,
                "shared_cache_group_index": shared["group_index"],
                "coarse_settle_buffer_s": 0.0,
                "navigation_source": "shared_cache_group_340mm",
                "navigation_point_base_mm": np.asarray(hole["coarse_center_base_mm"], dtype=np.float64).copy(),
                "cache_validation_result": validation.to_dict(),
            }
            cache_reused = True
            report["coarse_cache"]["cache_reused"].append(hole_id)
            if shared.get("source") == "persistent_base_cache":
                report["coarse_cache"]["persistent_cache_reused"].append(hole_id)
            coarse_captures = [{
                "capture_index": 0, "mode": "shared_cache_validated_reuse",
                "group_index": shared["group_index"],
                "summary": {"validation": validation.to_dict()},
            }]
            rgbd_pipeline, align, chain = ensure_rgbd_pipeline()
            if not batch_fine_available:
                fine_target, fine_geometry = _plan_hole_tcp_pose_fixed_rz(
                    np.asarray(hole["coarse_center_base_mm"], dtype=np.float64),
                    np.asarray(hole["coarse_normal_toward_camera_base"], dtype=np.float64),
                    current_tcp, handeye.T_tcp_rgb_camera, fixed_rz_rad=fixed_rz_rad,
                    camera_height_mm=cfg.fine_height_mm,
                )
                hole["shared_cache_fine_target_tcp_pose_m_rad"] = transform_to_sdk_pose_m_rad(fine_target)
                hole["shared_cache_fine_pose_geometry"] = fine_geometry
                with timing.measure(
                    f"hole_{hole_id:02d}/shared_cache_move_to_fine_pose",
                    hole_id=hole_id, processing_order=order,
                ):
                    current_tcp = _move_to_fine_pose(
                        str(hole_id), order, current_tcp, fine_target, args,
                        motion_session, pose_session, target_height_mm=cfg.fine_height_mm,
                        target_stage="共享缓存验证后的单孔精定位兜底",
                    )
            else:
                hole["batch_fine_localization_source"] = str(
                    batch_fine_result.get("batch_fine_source", "batch_fine_at_260mm")
                )
                hole["batch_fine_capture_tcp_pose_m_rad"] = transform_to_sdk_pose_m_rad(
                    batch_fine_result["capture_tcp"]
                )

        elif initial_pointcloud_reuse_available:
            # 直接复用本轮初始选孔帧的逐孔点云。这不是历史缓存命中，
            # 不做340mm验证、不移动、不重新读取点云。
            print(
                f"[POINTCLOUD_REUSE] hole={hole_id} 复用本轮初始RGB-D点云，"
                "跳过逐孔340mm采集",
                flush=True,
            )
            cache_event = {
                "enabled": False,
                "hole_id": hole_id,
                "cache_available": False,
                "cache_source": "initial_selection_shared_pointcloud_reuse",
                "cache_reused": False,
                "initial_pointcloud_reused": True,
                "reuse_scope": "current_cycle_initial_rgbd_frame",
                "cache_validation_skipped": True,
                "cache_validation_failed": False,
                "full_coarse_fallback": False,
                "coarse_settle_buffer_s": 0.0,
                "navigation_source": "batch_fine_group_already_captured",
                "navigation_point_base_mm": np.asarray(
                    hole["coarse_center_base_mm"], dtype=np.float64
                ).copy(),
            }
            # 内部沿用“粗几何已就绪”布尔量；报告通过独立字段与缓存区分。
            cache_reused = True
            coarse_captures = list(hole.get("coarse_captures", []))
            hole["batch_fine_localization_source"] = str(
                batch_fine_result.get("batch_fine_source", "batch_fine_at_260mm")
            )
            hole["batch_fine_capture_tcp_pose_m_rad"] = transform_to_sdk_pose_m_rad(
                batch_fine_result["capture_tcp"]
            )

        else:
            # 使用传统逐孔粗定位流程
            if batch_coarse_results and hole_id in batch_coarse_results:
                print(
                    f"[SEQUENTIAL_HOLE] order={order} hole={hole_id} "
                    f"批量粗定位失败，回退到逐孔粗定位: {batch_coarse_results[hole_id].get('error')}",
                    flush=True,
                )

            rgbd_pipeline, align, chain = ensure_rgbd_pipeline()
            coarse_target, coarse_pose_geometry = _plan_hole_tcp_pose_fixed_rz(
                navigation_point_base, navigation_normal_base,
                current_tcp, handeye.T_tcp_rgb_camera,
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

            cache_event: dict[str, Any] = {
                "enabled": cache_enabled,
                "hole_id": hole_id,
                "cache_available": hole_id in cache_entries,
                "cache_source": cache_sources.get(hole_id),
                "persistent_hole_id": cache_source_ids.get(hole_id),
                "cache_reused": False,
                "cache_validation_skipped": False,
                "cache_validation_failed": hole_id in shared_cache_failed_ids,
                "cache_invalidated": hole_id in invalidated_cache_ids,
                "full_coarse_fallback": False,
                "coarse_settle_buffer_s": 0.0,
                "navigation_source": hole.get("coarse_navigation_source"),
                "navigation_point_base_mm": navigation_point_base.copy(),
            }
            cache_reused = False
            if (
                cache_enabled
                and hole_id in cache_entries
                and hole_id not in shared_cache_failed_ids
            ):
                cache_entry = cache_entries[hole_id]
                T_base_camera = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
                try:
                    if cache_sources.get(hole_id) == "persistent_base_cache":
                        disk_entries = load_persistent_cache_entries(
                            persistent_cache_dir, [int(cache_source_ids[hole_id])],
                        )
                        cache_entry = disk_entries.get(int(cache_source_ids[hole_id]))
                        if cache_entry is None:
                            raise FileNotFoundError(
                                f"持久化base缓存缺少孔{cache_source_ids[hole_id]}条目"
                            )
                        cache_entry = rekey_cache_entry(cache_entry, hole_id)
                    else:
                        # 当前运行缓存仍从本次运行目录重新读取，NPZ损坏时回退完整粗定位。
                        disk_entries = load_cache_entries(coarse_cache_dir, [hole_id])
                        cache_entry = disk_entries.get(hole_id)
                        if cache_entry is None:
                            raise FileNotFoundError(f"当前运行缓存缺少孔{hole_id}条目")
                    cache_source = cache_sources.get(hole_id, "current_run_initial_cache")
                    # 缓存几何落在base坐标系，复用前必须在340mm现场验证
                    # （aubo_workbench/paths.py 的设计约束）；不再直接信任缓存。
                    with timing.measure(
                        f"hole_{hole_id:02d}/coarse_cache_validation",
                        hole_id=hole_id, processing_order=order,
                    ):
                        validation, cache_observations, _expected_anchor = _validate_coarse_cache_at_current_pose(
                            hole, cache_entry,
                            pipeline=rgbd_pipeline, align=align, chain=chain,
                            model=model, confidence=args.confidence,
                            run_dir=run_dir, cfg=cfg, gates=cache_gates,
                            current_T_base_camera=T_base_camera, intrinsics=initial_intrinsics,
                        )
                    rows.extend(_observation_rows(cache_observations))
                    cache_event["cache_validation_result"] = validation.to_dict()
                    if not validation.accepted:
                        cache_event["cache_validation_skipped"] = False
                        cache_event["cache_validation_failed"] = True
                        cache_event["failure_reason"] = f"cache_validation_rejected:{validation.reason}"
                        report["coarse_cache"]["cache_validation_failed"].append({
                            "hole_id": hole_id, "reason": cache_event["failure_reason"],
                        })
                    else:
                        _apply_cached_geometry_to_hole(
                            hole, cache_entry, validation,
                            source=cache_source,
                            persistent_hole_id=cache_source_ids.get(hole_id),
                        )
                        if cache_source == "persistent_base_cache":
                            transformed_points = transform_cached_points_to_camera(
                                cache_entry, T_base_camera,
                            )
                            hole["coarse_cache_transformed_point_count"] = int(
                                sum(len(points) for points in transformed_points)
                            )
                            hole["coarse_cache_transform"] = "T_base_camera_build_to_current_camera"
                        cache_reused = True
                        cache_event["cache_reused"] = True
                        cache_event["cache_validation_skipped"] = False
                        cache_event["cache_validation_failed"] = False
                        cache_event["coarse_settle_buffer_s"] = 0.0
                        report["coarse_cache"]["cache_reused"].append(hole_id)
                        if cache_source == "persistent_base_cache":
                            report["coarse_cache"]["persistent_cache_reused"].append(hole_id)
                        coarse_captures.append({
                            "capture_index": 0,
                            "mode": "cache_validated_reuse",
                            "tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                            "summary": {
                                "reuse_mode": "validated_base_coordinate_cache_live_z",
                                "validation_skipped": False,
                                "validation": validation.to_dict(),
                                "cached_valid_frames": int(cache_entry.valid_frames),
                                "cached_plane_rmse_mm": float(cache_entry.plane_rmse_mm),
                                "cached_center_scatter_p95_px": float(
                                    cache_entry.center_scatter_p95_px
                                ),
                            },
                        })
                        overlay_path = run_dir / f"hole_{hole_id:02d}_coarse_cache_verify_overlay.png"
                        pointcloud_path = _save_coarse_pointcloud_image(
                            hole, hole_id, run_dir,
                            observations=cache_observations,
                            cache_entry=cache_entry,
                            T_base_camera=T_base_camera,
                            intrinsics=initial_intrinsics,
                            rgb_path=overlay_path,
                            source_label=f"{cache_source}_validated",
                        )
                        hole["coarse_pointcloud_image_path"] = (
                            None if pointcloud_path is None else str(pointcloud_path)
                        )
                except Exception as exc:
                    cache_event["cache_validation_failed"] = True
                    cache_event["failure_reason"] = f"{type(exc).__name__}:{exc}"
                    report["coarse_cache"]["cache_validation_failed"].append({
                        "hole_id": hole_id, "reason": cache_event["failure_reason"],
                    })
            elif cache_enabled:
                cache_event["cache_validation_failed"] = bool(
                    hole_id in shared_cache_failed_ids
                )
                cache_event["cache_invalidated"] = bool(
                    hole_id in invalidated_cache_ids
                )
                cache_event["failure_reason"] = (
                    "shared_cache_validation_rejected"
                    if hole_id in shared_cache_failed_ids else "cache_unavailable"
                )
                report["coarse_cache"]["cache_validation_failed"].append({
                    "hole_id": hole_id, "reason": cache_event["failure_reason"],
                })

        coarse_capture_ready = bool(batch_coarse_available or cache_reused)
        # 批量粗定位和缓存命中都已经提供粗几何；仅缓存缺失/损坏时
        # 才等待停稳并重新执行逐孔340 mm采集。
        with timing.measure(
            f"hole_{hole_id:02d}/coarse_settle_buffer",
            hole_id=hole_id,
            processing_order=order,
            delay_s=(coarse_settle_delay_s if not coarse_capture_ready else 0.0),
            skipped=coarse_capture_ready,
        ):
            if not coarse_capture_ready and coarse_settle_delay_s > 0.0:
                time.sleep(coarse_settle_delay_s)
        cache_event["coarse_settle_buffer_s"] = (
            0.0 if coarse_capture_ready else coarse_settle_delay_s
        )
        cache_event["coarse_settle_buffer_reason"] = (
            "batch_coarse_already_complete"
            if batch_coarse_available else
            "direct_cache_reuse_no_wait"
            if cache_reused else
            "full_coarse_fallback"
        )

        if not coarse_capture_ready:
            cache_event["full_coarse_fallback"] = bool(cache_enabled)
            if cache_enabled:
                report["coarse_cache"]["full_coarse_fallback"].append({
                    "hole_id": hole_id,
                    "reason": cache_event.get("failure_reason", "cache_unavailable"),
                })
        coarse_correction_count = 0
        coarse_failure: dict[str, Any] | None = None
        for capture_index in range(1, 5) if not coarse_capture_ready else []:
            settle_discarded_frames = 0
            settle_delay_s = 0.0
            if capture_index > 1:
                # 纠偏后控制器先报告到位，但相机队列和末端仍可能保留运动过程帧。
                # 重拍前重新确认停稳，并主动丢弃预热帧，避免把连续漂移融合进当前TCP位姿。
                with timing.measure(
                    f"hole_{hole_id:02d}/coarse_recapture_settle_{capture_index}",
                    hole_id=hole_id,
                    processing_order=order,
                    capture_index=capture_index,
                ):
                    current_tcp, settle_discarded_frames = _settle_and_discard_coarse_recapture_frames(
                        pose_session,
                        rgbd_pipeline,
                        align,
                        chain,
                        settle_delay_s=coarse_settle_delay_s,
                        discard_frames=cfg.coarse_recapture_settle_discard_frames,
                    )
                    settle_delay_s = float(coarse_settle_delay_s)
            capture_tcp = np.asarray(current_tcp, dtype=np.float64).copy()
            T_base_camera = camera_transform(capture_tcp, handeye.T_tcp_rgb_camera)
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
                    # 第一个检测仍需落在预测锚点附近（70 px身份门）；
                    # 后续帧跟踪上一帧检测，避免把固定手眼投影偏差
                    # 计入15 px的帧间稳定性统计。
                    lock_anchor=False,
                    include_points=True,
                )
            rows.extend(_observation_rows(coarse_observations))
            _record_hole_tracking_event(
                hole, capture_name, expected_anchor_px, coarse_observations,
                initial_intrinsics, "distorted_pixel_yolo_center",
            )
            # capture_tcp 与采集期间的 current_tcp 相同，复用上面已计算的变换。
            last_coarse_observations = coarse_observations
            last_coarse_T_base_camera = np.asarray(T_base_camera, dtype=np.float64).copy()
            with timing.measure(
                f"hole_{hole_id:02d}/coarse_pointcloud_geometry_{capture_index}",
                hole_id=hole_id,
                processing_order=order,
                capture_index=capture_index,
            ):
                summary = None
                try:
                    summary = _apply_coarse_geometry_to_hole(
                        hole, coarse_observations, T_base_camera, cfg,
                    )
                except RuntimeError as exc:
                    # 跟踪距离门是帧间身份/漂移门，不代表点云几何一定失效。
                    # 对 tracking-only 失败复用同一批点云做一次严格几何复核，
                    # 不放宽中心散布、点云平面或后续精定位质量要求。
                    if "粗定位跟踪距离失败" in str(exc):
                        try:
                            summary = _apply_coarse_geometry_to_hole(
                                hole, coarse_observations, T_base_camera, cfg,
                                enforce_tracking_gate=False,
                            )
                            tracking_p95 = summary.get("tracking_distance_p95_px")
                            tracking_limit = float(cfg.max_coarse_tracking_distance_p95_px)
                            # 只容许门限附近的数值抖动进入几何复核，避免把
                            # 严重身份漂移误当成可用点云。几何门本身仍保持原阈值。
                            if (
                                tracking_p95 is not None
                                and float(tracking_p95) > tracking_limit * 1.2
                            ):
                                raise RuntimeError(
                                    "跟踪距离超出容错范围："
                                    f"P95={float(tracking_p95):.3f}px > "
                                    f"{tracking_limit * 1.2:.3f}px"
                                )
                            summary["tracking_gate_error"] = str(exc)
                            print(
                                f"[COARSE_RECOVERY] 孔{hole_id} 第{capture_index}次粗拍"
                                "跟踪门超限，但几何质量复核通过；继续后续流程",
                                flush=True,
                            )
                        except RuntimeError:
                            # 几何门也失败时，继续走统一的重拍/跳过处理。
                            summary = None
                    coarse_captures.append({
                        "capture_index": capture_index,
                        "status": "rejected_quality",
                        "error": str(exc),
                        "valid_frames": sum(
                            item.error is None and item.plane is not None
                            for item in coarse_observations
                        ),
                        "total_frames": len(coarse_observations),
                        "settle_delay_s": settle_delay_s,
                        "settle_discarded_frames": settle_discarded_frames,
                    }) if summary is None else None
                    hole["coarse_captures"] = list(coarse_captures)
                    if summary is None:
                        print(
                            f"[COARSE_RETRY] 孔{hole_id} 第{capture_index}次粗拍未通过稳定性门，"
                            f"不使用该结果修正位姿：{exc}",
                            flush=True,
                        )
                        if capture_index >= 4:
                            coarse_failure = {
                                "error": str(exc),
                                "capture_index": capture_index,
                            }
                            break
                        # 原地重拍；坏帧不能参与下一次机械臂修正。
                        continue
            center_offset = float(np.linalg.norm(
                np.asarray(summary["center_px"], dtype=np.float64)
                - np.array([initial_intrinsics.cx, initial_intrinsics.cy])
            ))
            normal_error = _angle_deg(
                T_base_camera[:3, 2],
                -np.asarray(hole["coarse_normal_toward_camera_base"], dtype=np.float64),
            )
            previous_normal_error = final_normal_error
            previous_center_offset = final_center_offset
            final_center_offset = center_offset
            final_normal_error = normal_error
            coarse_captures.append({
                "capture_index": capture_index,
                "status": (
                    "accepted_degraded_tracking"
                    if summary.get("quality_recovery") == "tracking_degraded_but_geometry_valid"
                    else "accepted"
                ),
                "tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "tracking_anchor_mode": "initial_projection_then_previous_detection",
                "center_offset_px": center_offset,
                "normal_error_deg": normal_error,
                "normal_error_delta_deg": (
                    None if previous_normal_error is None
                    else normal_error - previous_normal_error
                ),
                "center_offset_delta_px": (
                    None if previous_center_offset is None
                    else center_offset - previous_center_offset
                ),
                "coarse_correction_count": coarse_correction_count,
                "settle_delay_s": settle_delay_s,
                "settle_discarded_frames": settle_discarded_frames,
                "summary": summary,
            })
            hole["coarse_captures"] = list(coarse_captures)
            if center_offset <= cfg.center_tolerance_px and normal_error <= cfg.normal_tolerance_deg:
                break
            if (
                capture_index >= 4
                or coarse_correction_count >= cfg.coarse_max_corrections
            ):
                break
            coarse_correction_count += 1
            # 相机旋转会改变光轴与孔面的交点；即使当前像素中心已合格，
            # 也必须联立更新位置和姿态，才能让旋转后的目标继续落在主点。
            correction_mode = "center_and_orientation"
            correction_target, correction_geometry = _plan_hole_tcp_pose_fixed_rz(
                np.asarray(hole["coarse_center_base_mm"], dtype=np.float64),
                np.asarray(hole["coarse_normal_toward_camera_base"], dtype=np.float64),
                current_tcp, handeye.T_tcp_rgb_camera,
                fixed_rz_rad=fixed_rz_rad, camera_height_mm=cfg.coarse_height_mm,
            )
            coarse_captures[-1].update({
                "next_correction_index": coarse_correction_count,
                "next_correction_mode": correction_mode,
            })
            hole["coarse_captures"] = list(coarse_captures)
            with timing.measure(
                f"hole_{hole_id:02d}/coarse_correction_motion_{capture_index}",
                hole_id=hole_id,
                processing_order=order,
                capture_index=capture_index,
            ):
                current_tcp = _confirm_and_move_line(
                    f"孔{hole_id}粗定位闭环校正 {coarse_correction_count}/{cfg.coarse_max_corrections}",
                    current_tcp, correction_target, args, motion_session, pose_session,
                    f"center offset={center_offset:.2f}px, normal error={normal_error:.3f}deg；"
                    f"mode={correction_mode}, "
                    f"camera_axis_error={float(correction_geometry['camera_axis_error_deg']):.5f}deg",
                    require_confirmation=False,
                    motion_profile="approach",
                )
        hole["coarse_captures"] = coarse_captures
        hole["coarse_cache_event"] = cache_event
        if (
            coarse_failure is None
            and not batch_coarse_available
            and not cache_reused
            and (
                final_center_offset is None
                or final_normal_error is None
                or final_center_offset > cfg.center_tolerance_px
                or final_normal_error > cfg.normal_tolerance_deg
            )
        ):
            coarse_failure = {
                "error": (
                    f"孔{hole_id}粗定位闭环后仍未通过质量门："
                    f"offset={float(final_center_offset or math.inf):.2f}px, "
                    f"normal={float(final_normal_error or math.inf):.3f}deg"
                ),
                "capture_index": int(coarse_captures[-1].get("capture_index", 0))
                if coarse_captures else None,
            }
        if coarse_failure is not None:
            # 单孔粗定位最终失败不应终止整批任务。记录为 deferred，
            # 不执行任何基于不可靠几何的精定位/末端运动，然后继续下一孔。
            deferred_result = {
                "status": "deferred_coarse_quality",
                "hole_id": hole_id,
                "processing_order": order,
                "tracking_identity": hole["tracking_identity"],
                "initial_selection_order": hole.get("initial_selection_order"),
                "initial_center_px": hole.get("initial_center_px"),
                "initial_center_base_mm": hole.get("initial_center_base_mm"),
                "initial_plane_normal_base": hole.get("initial_plane_normal_base"),
                "batch_coarse_requested": bool(batch_coarse_for_cache),
                "batch_fine_requested": bool(cfg.batch_fine_localization),
                "batch_fine_source": (
                    str(batch_fine_result.get("batch_fine_source", "batch_fine_at_260mm"))
                    if batch_fine_available else "per_hole_fine_fallback"
                ),
                "batch_fine_capture_round": int(
                    batch_fine_result.get("batch_fine_capture_round", 0)
                ),
                "batch_fine_group_index": batch_fine_plan.get("hole_group_indices", {}).get(str(hole_id)),
                "batch_coarse_fallback_reason": (
                    batch_coarse_results.get(hole_id, {}).get("error")
                    if batch_coarse_results else None
                ),
                "coarse_source": "fresh_per_hole_coarse",
                "coarse_quality_status": "deferred_coarse_quality",
                "coarse_quality_note": coarse_failure["error"],
                "batch_observed_center_base_mm": hole.get("batch_observed_center_base_mm"),
                "batch_observed_plane_point_base_mm": hole.get("batch_observed_plane_point_base_mm"),
                "batch_observed_normal_base": hole.get("batch_observed_normal_base"),
                "coarse_captures": coarse_captures,
                "coarse_cache_event": cache_event,
                "tracking_events": hole.get("tracking_events", []),
                "timing": timing.scoped_snapshot(f"hole_{hole_id:02d}/"),
            }
            deferred_result["comparison_diagnostics"] = _build_comparison_hole_diagnostics(
                deferred_result
            )
            hole["final_result"] = deferred_result
            results.append(deferred_result)
            report["stages"][f"hole_{hole_id}"] = deferred_result
            report["stages"]["processed_holes"] = {
                "completed_count": sum(item.get("status") == "completed" for item in results),
                "deferred_count": sum(item.get("status") != "completed" for item in results),
                "total_count": len(results),
                "hole_order": [int(item["hole_id"]) for item in results],
                "holes": results,
            }
            report.setdefault("deferred_holes", []).append(hole_id)
            _write_report(run_dir, report, rows)
            print(
                f"[SEQUENTIAL_HOLE] order={order} hole={hole_id} "
                f"status=deferred_coarse_quality；继续后续孔",
                flush=True,
            )
            if order < len(initial_holes):
                next_hole_id = int(initial_holes[order]["hole_id"])
                command = _request_next_hole_confirmation(hole_id, next_hole_id)
                if command != "m":
                    raise RuntimeError(f"用户在孔{hole_id}失败后停止流程")
            continue
        if not cache_reused and last_coarse_observations is not None:
            last_capture_index = (
                int(coarse_captures[-1].get("capture_index", 0)) if coarse_captures else 0
            )
            overlay_path = run_dir / f"hole_{hole_id:02d}_coarse_{last_capture_index}_overlay.png"
            pointcloud_path = _save_coarse_pointcloud_image(
                hole, hole_id, run_dir,
                observations=last_coarse_observations,
                cache_entry=None,
                T_base_camera=(
                    last_coarse_T_base_camera
                    if last_coarse_T_base_camera is not None
                    else camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
                ),
                intrinsics=initial_intrinsics,
                rgb_path=overlay_path,
                source_label="fresh_coarse_capture",
            )
            hole["coarse_pointcloud_image_path"] = (
                None if pointcloud_path is None else str(pointcloud_path)
            )
        # 逐孔粗定位只是本轮失败缓存的压轴兜底，不得把其点云写回正式
        # 缓存；正式缓存仅在上方一拍多批量成功后建立或更新。
        if hole.get("coarse_quality_recovery") == "tracking_degraded_but_geometry_valid":
            cache_event["cache_refresh_skipped"] = "tracking_degraded_but_geometry_valid"
        elif not batch_coarse_available and not cache_reused and cache_enabled:
            cache_event["cache_refresh_skipped"] = "per_hole_fallback_not_cache_source"

        if batch_coarse_available:
            cache_event["coarse_capture_frames_skipped"] = max(
                0, int(cfg.coarse_frames),
            )
            cache_event["coarse_capture_time_saved_s"] = None
            cache_event["coarse_capture_time_saved_basis"] = (
                "batch_coarse_shared_rgbd_capture"
            )
        elif cache_reused:
            cache_event["coarse_capture_frames_skipped"] = max(
                0, int(cfg.coarse_frames),
            )
            cache_event["coarse_capture_time_saved_s"] = None
            cache_event["coarse_capture_time_saved_basis"] = (
                "current_initial_rgbd_pointcloud_reuse"
                if cache_event.get("initial_pointcloud_reused") else
                "direct_cache_reuse_no_live_validation_baseline"
            )
        else:
            cache_event["coarse_capture_frames_skipped"] = 0
            cache_event["coarse_capture_time_saved_s"] = 0.0
            cache_event["coarse_capture_time_saved_basis"] = "no_cache_reuse"

        # 粗定位完成（批量或逐孔），继续执行高度修正和精定位
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

        # 260 mm共享精拍只是一处同时观察全部孔的相机位姿，不能作为每个孔
        # 的最终姿态底稿。逐孔保存由该孔粗定位中心和法向生成的260 mm参考
        # 位姿；执行最终动作时先恢复这个孔自己的Z和姿态，再仅写入精拍XY。
        coarse_fine_reference_tcp: np.ndarray | None = None
        coarse_fine_reference_geometry: dict[str, Any] | None = None
        if batch_fine_available:
            coarse_fine_reference_tcp, coarse_fine_reference_geometry = (
                _plan_hole_tcp_pose_fixed_rz(
                    np.asarray(hole["coarse_center_base_mm"], dtype=np.float64),
                    coarse_normal_base,
                    current_tcp,
                    handeye.T_tcp_rgb_camera,
                    fixed_rz_rad=fixed_rz_rad,
                    camera_height_mm=cfg.fine_height_mm,
                )
            )
            hole["coarse_fine_reference_tcp_pose_m_rad"] = (
                transform_to_sdk_pose_m_rad(coarse_fine_reference_tcp)
            )
            hole["coarse_fine_reference_pose_geometry"] = coarse_fine_reference_geometry

        if batch_fine_available:
            # 批量精定位已经在共同260mm位姿完成；这里仅使用采集位姿做
            # 像素到基坐标的转换，不再为当前孔重复移动、等待或拍照。
            fine_capture_tcp = np.asarray(
                batch_fine_result["capture_tcp"], dtype=np.float64,
            ).copy()
            estimated_height = camera_height_to_plane_mm(
                fine_capture_tcp, handeye.T_tcp_rgb_camera, coarse_plane_base,
            )
            hole["fine_height_estimate_mm"] = estimated_height
            fine_intrinsics = batch_fine_result["intrinsics"]
            fine_observations = list(batch_fine_result.get("observations", []))
            fine = batch_fine_result["fine"]
            expected_fine_anchor_px = np.asarray(
                batch_fine_result["expected_anchor_px"], dtype=np.float64,
            ).copy()
            fine_recovery = {
                "success": True,
                "observations": fine_observations,
                "intrinsics": fine_intrinsics,
                "fine": fine,
                "attempts": fine.get("fine_recovery_attempts", []),
                "error": None,
            }
            _record_hole_tracking_event(
                hole, "batch_fine", expected_fine_anchor_px,
                fine_observations, fine_intrinsics,
                "undistorted_pixel_yolo_center",
            )
        else:
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
            fine_capture_tcp = np.asarray(current_tcp, dtype=np.float64).copy()

            # RGB 精定位只需要 RGB 帧；get_rgb_frame_bundle 不会读取深度或生成点云，
            # 因此直接复用当前 RGB-D pipeline，避免每个孔重复 stop/start 两套相机管线。
            with timing.measure(
                f"hole_{hole_id:02d}/reuse_rgbd_pipeline_for_rgb",
                hole_id=hole_id,
                processing_order=order,
            ):
                fine_pipeline = rgbd_pipeline
            T_base_camera_fine = camera_transform(fine_capture_tcp, handeye.T_tcp_rgb_camera)
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
                "initial_plane_normal_base": hole.get("initial_plane_normal_base"),
                "batch_coarse_requested": bool(batch_coarse_for_cache),
                "batch_fine_requested": bool(cfg.batch_fine_localization),
                "batch_fine_source": (
                    str(batch_fine_result.get("batch_fine_source", "batch_fine_at_260mm"))
                    if batch_fine_available else "per_hole_fine_fallback"
                ),
                "batch_fine_capture_round": int(
                    batch_fine_result.get("batch_fine_capture_round", 0)
                ),
                "batch_fine_group_index": batch_fine_plan.get("hole_group_indices", {}).get(str(hole_id)),
                "batch_coarse_fallback_reason": (
                    batch_coarse_results.get(hole_id, {}).get("error")
                    if batch_coarse_results else None
                ),
                "coarse_source": (
                    "batch_coarse_localization" if batch_coarse_available else
                    str(cache_event.get("cache_source") or "cache_validated_reuse")
                    if cache_reused else "fresh_per_hole_coarse"
                ),
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
                "coarse_surface_model": hole.get("coarse_surface_model"),
                "coarse_surface_selection_policy": hole.get("coarse_surface_selection_policy"),
                "coarse_front_surface_z_mm": hole.get("coarse_front_surface_z_mm"),
                "coarse_ring_points_raw_median": hole.get("coarse_ring_points_raw_median"),
                "coarse_surface_points_selected_median": hole.get(
                    "coarse_surface_points_selected_median"
                ),
                "batch_observed_center_base_mm": hole.get("batch_observed_center_base_mm"),
                "batch_observed_plane_point_base_mm": hole.get("batch_observed_plane_point_base_mm"),
                "batch_observed_normal_base": hole.get("batch_observed_normal_base"),
                "coarse_captures": hole["coarse_captures"],
                "coarse_cache_event": hole.get("coarse_cache_event"),
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
            deferred_result["comparison_diagnostics"] = _build_comparison_hole_diagnostics(
                deferred_result
            )
            hole["final_result"] = deferred_result
            results.append(deferred_result)
            report["stages"][f"hole_{hole_id}"] = deferred_result
            completed_results = [item for item in results if item.get("status") == "completed"]
            deferred_results = [
                item for item in results if item.get("status", "").startswith("deferred_")
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
                fine["center_px"], fine_intrinsics, fine_capture_tcp, handeye.T_tcp_rgb_camera,
                coarse_plane_base, coarse_normal_base, center_is_undistorted=True,
            )
        diameter_px = float(max(float(fine["axes_px_median"][0]), float(fine["axes_px_median"][1])))
        diameter_estimate = diameter_px * estimated_height / (
            (fine_intrinsics.fx + fine_intrinsics.fy) / 2.0
        )
        nearest_diameter = min(HOLE_DIAMETERS_MM, key=lambda value: abs(value - diameter_estimate))
        T_base_camera_fine = camera_transform(fine_capture_tcp, handeye.T_tcp_rgb_camera)
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
                    fine["center_px"], fine_intrinsics, fine_capture_tcp, handeye.T_tcp_rgb_camera,
                    coarse_plane_base, coarse_normal_base, nearest_diameter,
                    iterations=cfg.tilt_correction_iterations,
                    samples=cfg.tilt_correction_samples,
                    max_correction_mm=cfg.max_tilt_correction_mm,
                )

        # 共享精拍只更新每孔最终XY。该孔的最终Z与姿态仍由粗定位决定；
        # 非共享精拍流程保持原有三维最终点行为。
        pose_point_base = np.asarray(final_point_base, dtype=np.float64).copy()
        if batch_fine_available:
            pose_point_base = compose_batch_fine_xy_with_coarse_z(
                pose_point_base,
                np.asarray(hole["coarse_center_base_mm"], dtype=np.float64),
            )

        # 先按最终点模式施加基坐标偏移，再用偏移后的点规划最终移动。
        final_target_mode = str(getattr(args, "final_target_mode", DEFAULT_FINAL_TARGET_MODE))
        final_x_offset_mm, final_z_offset_mm = final_point_offsets_for_mode(final_target_mode)
        final_target_point_base = apply_final_point_base_offsets(
            pose_point_base, final_x_offset_mm, final_z_offset_mm,
        )
        fine_tcp = np.asarray(fine_capture_tcp, dtype=np.float64).copy()
        final_xy_motion: dict[str, Any] | None = None
        final_z_motion: dict[str, Any] | None = None
        final_y_trim_motion: dict[str, Any] | None = None
        if args.move_final_xy and batch_fine_available:
            assert coarse_fine_reference_tcp is not None
            with timing.measure(
                f"hole_{hole_id:02d}/restore_coarse_pose_for_final_motion",
                hole_id=hole_id,
                processing_order=order,
            ):
                current_tcp = _restore_batch_fine_pose_for_final_motion(
                    str(hole_id), order, current_tcp, coarse_fine_reference_tcp,
                    args, motion_session, pose_session,
                    target_height_mm=cfg.fine_height_mm,
                )
            hole["coarse_pose_restored_for_batch_fine_final_motion"] = True
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
                "pose_point_base_mm": pose_point_base,
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
            y_trim_target = plan_final_tcp_combined_y_trim(current_tcp)
            with timing.measure(
                f"hole_{hole_id:02d}/final_motion_y_plus_combined",
                hole_id=hole_id,
                processing_order=order,
            ):
                current_tcp = _confirm_and_move_line(
                    f"孔{hole_id}最终+Y微调（基坐标+工具系合并一次移动）",
                    current_tcp, y_trim_target, args, motion_session, pose_session,
                    f"保持姿态；基坐标Y增加 {FINAL_BASE_Y_AFTER_Z_MM:.3f} mm 并叠加工具系+Y "
                    f"{FINAL_TOOL_Y_AFTER_Z_MM:.3f} mm，合并为一次移动",
                    require_confirmation=False,
                    motion_profile="precision",
                )
            final_y_trim_motion = {
                "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(y_trim_target),
                "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "delta_base_y_mm": FINAL_BASE_Y_AFTER_Z_MM,
                "delta_tool_y_mm": FINAL_TOOL_Y_AFTER_Z_MM,
                "motion_frame": "base_y_plus_tool_y_combined",
            }
        if batch_coarse_available:
            coarse_source = "batch_coarse_localization"
            batch_fallback_reason = None
        elif cache_reused:
            coarse_source = str(cache_event.get("cache_source") or "cache_validated_reuse")
            batch_fallback_reason = None
        else:
            coarse_source = "fresh_per_hole_coarse"
            batch_fallback_reason = (
                batch_coarse_results.get(hole_id, {}).get("error")
                if batch_coarse_results else None
            )

        result = dict(fine)
        result.update({
            "status": "completed",
            "hole_id": hole_id,
            "processing_order": order,
            "tracking_identity": hole["tracking_identity"],
            "initial_selection_order": hole.get("initial_selection_order"),
            "initial_center_px": hole.get("initial_center_px"),
            "initial_center_base_mm": hole.get("initial_center_base_mm"),
            "initial_plane_normal_base": hole.get("initial_plane_normal_base"),
            "batch_coarse_requested": bool(batch_coarse_for_cache),
            "batch_fine_requested": bool(cfg.batch_fine_localization),
            "batch_fine_source": (
                str(batch_fine_result.get("batch_fine_source", "batch_fine_at_260mm"))
                if batch_fine_available else "per_hole_fine"
            ),
            "batch_fine_capture_round": int(
                batch_fine_result.get("batch_fine_capture_round", 0)
            ),
            "batch_fine_group_index": batch_fine_plan.get("hole_group_indices", {}).get(str(hole_id)),
            "batch_coarse_fallback_reason": batch_fallback_reason,
            "coarse_source": coarse_source,
            "fine_center_source": fine.get("center_source", "unknown"),
            "fine_center_source_counts": fine.get("center_source_counts", {}),
            "fine_quality_status": fine.get("fine_quality_status", "strict"),
            "fine_quality_note": fine.get("fine_quality_note"),
            "fine_recovery_attempts": fine.get("fine_recovery_attempts", []),
            "hole_center_base_mm": final_point_base,
            "hole_result_type": "base_frame_3d_point",
            "final_pose_source": (
                "per_hole_coarse_pose_with_batch_fine_xy_only"
                if batch_fine_available else "per_hole_fine_pose_and_center"
            ),
            "batch_fine_xy_only": bool(batch_fine_available),
            "coarse_fine_reference_tcp_pose_m_rad": (
                transform_to_sdk_pose_m_rad(coarse_fine_reference_tcp)
                if coarse_fine_reference_tcp is not None else None
            ),
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
            "coarse_tracking_distance_p95_px": hole.get("coarse_tracking_distance_p95_px"),
            "coarse_ring_points_median": hole.get("coarse_ring_points_median"),
            "coarse_surface_model": hole.get("coarse_surface_model"),
            "coarse_surface_selection_policy": hole.get("coarse_surface_selection_policy"),
            "coarse_front_surface_z_mm": hole.get("coarse_front_surface_z_mm"),
            "coarse_ring_points_raw_median": hole.get("coarse_ring_points_raw_median"),
            "coarse_surface_points_selected_median": hole.get(
                "coarse_surface_points_selected_median"
            ),
            "batch_observed_center_base_mm": hole.get("batch_observed_center_base_mm"),
            "batch_observed_plane_point_base_mm": hole.get("batch_observed_plane_point_base_mm"),
            "batch_observed_normal_base": hole.get("batch_observed_normal_base"),
            "coarse_captures": hole["coarse_captures"],
            "coarse_cache_event": hole.get("coarse_cache_event"),
            "pointcloud_segmentation": hole["pointcloud_segmentation"],
            "fine_plane_intersection_mm": naive_final_point_base,
            "fine_xy_source": (
                "pointcloud_anchor_locked_yolo_center_on_coarse_local_plane"
            ),
            "fine_z_source": (
                "per_hole_coarse_center_z"
                if batch_fine_available else
                "coarse_front_surface_plane_intersection_z"
            ),
            "tilt_center_correction": tilt_correction,
            "estimated_height_mm": estimated_height,
            "diameter_estimate_mm": diameter_estimate,
            "matched_diameter_mm": nearest_diameter,
            "plane_normal_toward_camera_base": final_normal,
            "fixed_rz_rad": fixed_rz_rad,
            "shared_batch_capture_tcp_pose_m_rad": (
                transform_to_sdk_pose_m_rad(fine_capture_tcp)
                if batch_fine_available else None
            ),
            "fine_tcp_pose_m_rad": (
                None if batch_fine_available else transform_to_sdk_pose_m_rad(fine_tcp)
            ),
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
        result["comparison_diagnostics"] = _build_comparison_hole_diagnostics(result)
        hole["final_result"] = result
        results.append(result)
        report["stages"][f"hole_{hole_id}"] = result
        completed_results = [item for item in results if item.get("status") == "completed"]
        deferred_results = [
            item for item in results if item.get("status", "").startswith("deferred_")
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

        if (
            order < len(initial_holes)
            and (
                bool(getattr(args, "move_final_xy", False))
                or not all_selected_two_capture_mode
            )
        ):
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
        item for item in results if item.get("status", "").startswith("deferred_")
    ]
    report["stages"]["sequential_holes"] = {
        "mode": (
            "single_batch_fine_capture_then_compute_all_holes"
            if all_selected_two_capture_mode else
            "one_hole_complete_then_next"
        ),
        "hole_count": len(results),
        "completed_count": len(completed_results),
        "deferred_count": len(deferred_results),
        "failed_holes": [int(item["hole_id"]) for item in deferred_results],
        "hole_order": order_ids,
        "fixed_rz_rad": fixed_rz_rad,
        "tracking_identity_source": "initial_selection_order_and_locked_projected_anchor",
        "holes": results,
    }
    try:
        final_overlay = _save_batch_fine_final_result_overlay(
            run_dir, batch_fine_results, results, handeye,
        )
    except Exception as exc:
        final_overlay = {
            "error": f"{type(exc).__name__}:{exc}",
        }
        print(f"[FINAL_RESULT_OVERLAY_WARNING] {exc}", flush=True)
    report["stages"]["batch_fine_final_result_overlay"] = final_overlay
    if final_overlay and final_overlay.get("image_path"):
        for item in results:
            if str(item.get("batch_fine_source", "")).startswith("batch_fine"):
                item["final_result_overlay_path"] = final_overlay["image_path"]
    report["final_result"] = {
        "hole_count": len(results),
        "completed_count": len(completed_results),
        "deferred_count": len(deferred_results),
        "failed_holes": [
            int(item["hole_id"]) for item in deferred_results
            if item.get("status") == "deferred_coarse_quality"
        ],
        "hole_order": order_ids,
        "fixed_rz_rad": fixed_rz_rad,
        "holes": results,
        "final_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
    }
    report["comparison_diagnostics"] = _build_comparison_run_diagnostics(results, report, cfg)
    # 由外层会话在本轮完成后安全回原点，再开始下一轮初始选孔。
    # 保留在 runtime 中，避免改变现有函数的返回值兼容性。
    runtime["current_tcp"] = np.asarray(current_tcp, dtype=np.float64).copy()
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


def _request_next_cycle_confirmation(cycle_index: int) -> str:
    return input(
        f"第{int(cycle_index)}轮已完成且机器人已停稳；"
        "输入 m 开始下一轮初始拍摄，其他任意键结束："
    ).strip().lower()




def _restore_batch_fine_pose_for_final_motion(
    hole_id: str,
    order: int,
    current_tcp: np.ndarray,
    target_tcp: np.ndarray,
    args: Any,
    motion_session: Any,
    pose_session: Any,
    *,
    target_height_mm: float = 260.0,
    position_tolerance_mm: float = 0.5,
    rotation_tolerance_deg: float = 0.5,
) -> np.ndarray:
    """共享精拍后，安全恢复当前孔由粗定位生成的260 mm参考位姿。

    共享精拍会先完成全部孔的图像采集，随后逐孔执行最终动作；不能只把
    target_tcp赋给current_tcp，因为那不会让机器人真实移动。位姿不一致时
    复用安全的抬升-横移-分段下降路径，避免低位横移或沿用上一孔姿态。
    """
    actual = np.asarray(current_tcp, dtype=np.float64).reshape(4, 4).copy()
    desired = np.asarray(target_tcp, dtype=np.float64).reshape(4, 4).copy()
    position_error = float(np.linalg.norm(actual[:3, 3] - desired[:3, 3]))
    rotation_error = _rotation_distance_deg(actual[:3, :3], desired[:3, :3])
    if (
        position_error <= float(position_tolerance_mm)
        and rotation_error <= float(rotation_tolerance_deg)
    ):
        return actual
    return _move_to_fine_pose(
        hole_id,
        order,
        actual,
        desired,
        args,
        motion_session,
        pose_session,
        target_height_mm=target_height_mm,
        target_stage="恢复该孔粗定位位姿后执行精定位XY",
        descent_guard_mm=SHARED_OBSERVATION_MIN_DESCENT_MM,
    )


def _move_to_shared_fine_pose(
    group_label: str,
    order: int,
    current_tcp: np.ndarray,
    target: np.ndarray,
    args: Any,
    motion_session: Any,
    pose_session: Any,
    *,
    target_height_mm: float = 260.0,
    target_stage: str = "共享精定位",
) -> np.ndarray:
    """共享精定位专用的共同260 mm观察位移动。"""
    del order
    return _move_to_shared_observation_pose(
        target_stage,
        current_tcp,
        target,
        args,
        motion_session,
        pose_session,
        target_height_mm=target_height_mm,
        descent_profile="approach",
    )


def _move_to_fine_pose(
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
    descent_guard_mm: float = 0.0,
) -> np.ndarray:
    """沿安全高度移动到两阶段流程的目标相机高度。"""

    del order
    safe_z = max(float(current_tcp[2, 3]), float(target[2, 3])) + THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM
    actual = np.asarray(current_tcp, dtype=np.float64).copy()
    lift = actual.copy()
    lift[2, 3] = safe_z
    if abs(float(lift[2, 3] - actual[2, 3])) > 0.2:
        actual = _confirm_and_move_line(
            f"孔{hole_id}进入安全高度", actual, lift, args, motion_session, pose_session,
            "仅修改基坐标Z；不经过工件低位区域", require_confirmation=False,
            motion_profile="transit",
        )
    high_target = np.asarray(target, dtype=np.float64).copy()
    high_target[2, 3] = safe_z
    actual = _confirm_and_move_line(
        f"移动到孔{hole_id}上方安全位姿", actual, high_target, args,
        motion_session, pose_session, "安全高度横移并调整目标姿态；不下降",
        require_confirmation=False, motion_profile="transit",
    )
    guard_mm = max(0.0, float(descent_guard_mm))
    if guard_mm > 0.0:
        if float(safe_z - target[2, 3]) < guard_mm:
            raise RuntimeError(
                f"孔{hole_id}纯Z下降安全余量不足："
                f"{float(safe_z - target[2, 3]):.3f}mm < {guard_mm:.3f}mm"
            )
        descent_guard = np.asarray(target, dtype=np.float64).copy()
        descent_guard[2, 3] = float(target[2, 3]) + guard_mm
        actual = _confirm_and_move_line(
            f"孔{hole_id}纯Z下降到目标上方{guard_mm:.0f}mm安全位",
            actual,
            descent_guard,
            args,
            motion_session,
            pose_session,
            f"保持XY和姿态不变；先纯Z下降到目标上方{guard_mm:.0f}mm",
            require_confirmation=False,
            motion_profile="approach",
        )
    return _confirm_and_move_line(
        (
            f"孔{hole_id}再纯Z下降{guard_mm:.0f}mm到"
            f"{float(target_height_mm):.0f} mm{target_stage}位"
            if guard_mm > 0.0 else
            f"孔{hole_id}下降到{float(target_height_mm):.0f} mm{target_stage}位"
        ),
        actual, target, args, motion_session, pose_session,
        (
            f"孔中心和法向已冻结；保持XY和姿态不变，最后纯Z下降{guard_mm:.0f}mm"
            f"到指定RGB相机高度{float(target_height_mm):.0f} mm"
            if guard_mm > 0.0 else
            f"孔中心和法向已冻结；仅下降到指定RGB相机高度{float(target_height_mm):.0f} mm"
        ),
        require_confirmation=False, motion_profile="approach",
    )


def _new_two_stage_run_dir(cycle_index: int) -> Path:
    """为可重复选孔会话创建独立的一轮报告目录。"""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if int(cycle_index) == 1 else f"-cycle{int(cycle_index):02d}"
    stem = f"two-stage-{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    candidate = RUNS_DIR / stem
    serial = 2
    while candidate.exists():
        candidate = RUNS_DIR / f"{stem}-{serial:02d}"
        serial += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _new_two_stage_report(
    args: Any, handeye: Any, cfg: TwoStageConfig, run_dir: Path, cycle_index: int,
) -> dict[str, Any]:
    """创建一轮两阶段报告；机器人/相机资源由外层会话共享。"""
    precision_speed_m_s = float(args.speed_m_s)
    precision_acc_m_s2 = float(args.acc_m_s2)
    transit_speed_m_s = float(getattr(args, "transit_speed_m_s", precision_speed_m_s))
    transit_acc_m_s2 = float(getattr(args, "transit_acc_m_s2", precision_acc_m_s2))
    approach_speed_m_s = float(getattr(args, "approach_speed_m_s", 0.12))
    approach_acc_m_s2 = float(getattr(args, "approach_acc_m_s2", 0.35))
    final_target_mode = str(getattr(args, "final_target_mode", DEFAULT_FINAL_TARGET_MODE))
    final_x_offset_mm, final_z_offset_mm = final_point_offsets_for_mode(final_target_mode)
    return {
        "status": "running",
        "mode": "two_stage_hole_localization",
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cycle_index": int(cycle_index),
        "session_reselect_enabled": bool(args.execute),
        "configuration": cfg.__dict__,
        "hole_count": None,
        "selection_mode": "click_any_count_then_enter",
        "handeye_path": str(args.handeye),
        "stages": {},
        "motion_executed": bool(args.execute),
        "experimental_handeye_override": bool(args.allow_experimental_handeye),
        "reuse_coarse_cache": bool(getattr(args, "reuse_coarse_cache", True)),
        "reuse_persistent_coarse_cache": bool(
            getattr(args, "reuse_persistent_coarse_cache", True)
        ),
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


def _run_two_stage_localization_cycle(
    args: Any, handeye: Any, model: Any, cfg: TwoStageConfig, run_dir: Path,
    report: dict[str, Any], timing: TimingRecorder, rows: list[dict[str, Any]],
    pipeline_runtime: dict[str, Any], pose_session: Any, motion_session: Any,
    current_tcp: np.ndarray, cycle_index: int,
) -> int:
    """执行一轮初始选孔到逐孔完成；相机和机器人会话由外层复用。"""
    rgbd_pipeline = pipeline_runtime.get("rgbd_pipeline")
    align = pipeline_runtime.get("align")
    chain = pipeline_runtime.get("chain")
    if rgbd_pipeline is None or align is None or chain is None:
        raise RuntimeError("RGB-D 管线尚未就绪，无法开始新一轮初始选孔")

    if pose_session is not None:
        with timing.measure(
            "robot/verify_steady_before_initial_capture",
            cycle_index=int(cycle_index),
        ):
            current_tcp = _wait_robot_steady_before_initial_capture(
                pose_session,
            )
        pipeline_runtime["current_tcp"] = current_tcp.copy()
        print(
            f"[CAMERA_READY] cycle={int(cycle_index)} 机器人已完全停止，"
            "开始采集本轮初始画面。",
            flush=True,
        )

    print(
        f"[INITIAL_SELECTION_REQUIRED] cycle={int(cycle_index)} "
        "机器人已在原点并完全停止，请选择本轮目标孔后按 Enter；按 Esc 结束会话。",
        flush=True,
    )
    with timing.measure(
        "initial_selection/yolo_rgbd_pointcloud",
        selection_mode="click_any_count_then_enter",
        cycle_index=int(cycle_index),
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
    selected_point_base = (
        T_base_camera_home[:3, :3] @ selected_point_camera
        + T_base_camera_home[:3, 3]
    )
    plane_point_base = (
        T_base_camera_home[:3, :3] @ initial_plane_camera.point_camera_mm
        + T_base_camera_home[:3, 3]
    )
    camera_origin_home = T_base_camera_home[:3, 3]
    group_plane_normal_camera = _unit(
        np.asarray(initial_plane_camera.normal_camera, dtype=np.float64),
        "initial group camera normal",
    )
    plane_normal_base = _unit(
        T_base_camera_home[:3, :3] @ group_plane_normal_camera,
        "initial group base normal",
    )
    if float(plane_normal_base @ (camera_origin_home - selected_point_base)) < 0.0:
        plane_normal_base = -plane_normal_base
        group_plane_normal_camera = -group_plane_normal_camera
    for hole in initial_selected_holes:
        point_camera = np.asarray(
            hole["initial_point_camera_mm"], dtype=np.float64,
        ).reshape(3)
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
                T_base_camera_home[:3, :3] @ plane_point_camera
                + T_base_camera_home[:3, 3]
            ).tolist(),
            "initial_plane_normal_base": normal_base.tolist(),
            # 同一批孔属于同一工件平面：批量精定位统一使用这一个平面和
            # 法向；逐孔字段仅保留作初始点云质量诊断。
            "initial_shared_plane_point_camera_mm": (
                np.asarray(initial_plane_camera.point_camera_mm, dtype=np.float64).tolist()
            ),
            "initial_shared_plane_point_base_mm": plane_point_base.tolist(),
            "initial_shared_plane_normal_camera": group_plane_normal_camera.tolist(),
            "initial_shared_plane_normal_base": plane_normal_base.tolist(),
            "initial_shared_plane_rmse_mm": float(initial_plane_camera.rmse_mm),
            "initial_shared_ring_points": int(initial_plane_camera.ring_points),
            "initial_shared_surface_model": initial_plane_camera.surface_model,
            "initial_shared_surface_selection_policy": (
                initial_plane_camera.surface_selection_policy
            ),
            "initial_shared_front_surface_z_mm": initial_plane_camera.front_surface_z_mm,
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
        "initial_geometry_fallback_holes": [
            int(hole["hole_id"])
            for hole in initial_selected_holes
            if bool(hole.get("initial_geometry_fallback"))
        ],
        "initial_geometry_fallback_reasons": {
            str(int(hole["hole_id"])): str(hole.get("initial_geometry_fallback_reason"))
            for hole in initial_selected_holes
            if bool(hole.get("initial_geometry_fallback"))
        },
        "initial_geometry_policy": (
            "per_hole_plane_or_shared_plane_fallback_for_safe_340mm_navigation"
        ),
    }

    coarse_cache_gates = CacheValidationGates(
        validation_frames=int(cfg.cache_validation_frames),
        min_valid_frames=int(cfg.cache_validation_min_valid),
        max_tracking_distance_px=cfg.multi_coarse_tracking_tolerance_px,
        max_center_offset_px=cfg.center_tolerance_px,
        max_center_scatter_p95_px=cfg.max_coarse_center_scatter_p95_px,
        max_plane_rmse_mm=cfg.max_plane_rmse_mm,
        max_normal_error_deg=cfg.normal_tolerance_deg,
    )
    coarse_cache_dir = run_dir / "coarse_cache"
    persistent_cache_dir = HOLE_LOCALIZATION_COARSE_CACHE_DIR
    cache_entries: dict[int, CoarseCacheEntry] = {}
    cache_sources: dict[int, str] = {}
    cache_source_ids: dict[int, int] = {}
    cache_built_ids: set[int] = set()
    persistent_entries: dict[int, CoarseCacheEntry] = {}
    persistent_load_errors: dict[int, str] = {}
    persistent_audit: dict[str, Any] = {
        "requested_dir": str(persistent_cache_dir),
        "loaded_count": 0,
        "match": {},
    }
    cache_enabled = bool(args.execute and getattr(args, "reuse_coarse_cache", True))
    persistent_enabled = bool(
        cache_enabled and getattr(args, "reuse_persistent_coarse_cache", True)
    )
    report["coarse_cache"] = {
        "enabled": cache_enabled,
        "cache_scope": "current_run_and_base_persistent",
        "cache_dir": str(coarse_cache_dir),
        "persistent_enabled": persistent_enabled,
        "persistent_cache_dir": str(persistent_cache_dir),
        "surface_selection_policy": COARSE_SURFACE_SELECTION_POLICY,
        "surface_model": COARSE_SURFACE_MODEL,
        "gates": coarse_cache_gates.to_dict(),
        "cache_built": [],
        "cache_available": [],
        "persistent_cache_loaded": [],
        "cache_reused": [],
        "persistent_cache_reused": [],
        "cache_validation_skipped": [],
        "cache_validation_failed": [],
        "cache_invalidated": [],
        "full_coarse_fallback": [],
        "cache_persist_error": None,
        "persistent_cache_persist_error": None,
    }
    if cache_enabled:
        if persistent_enabled:
            (
                persistent_entries,
                persistent_load_errors,
                persistent_audit,
            ) = _load_persistent_coarse_cache_for_run(
                run_dir=run_dir,
                camera_serial=str((report.get("camera") or {}).get("serial_number", "")),
                handeye=handeye,
                handeye_path=str(args.handeye),
                intrinsics=intrinsics,
                persistent_cache_dir=persistent_cache_dir,
                required_surface_model=COARSE_SURFACE_MODEL,
            )
            target_points = {
                int(hole["hole_id"]): np.asarray(
                    hole["initial_center_base_mm"], dtype=np.float64,
                )
                for hole in initial_selected_holes
            }
            (
                matched_entries,
                matched_source_ids,
                match_audit,
            ) = match_entries_by_base_point(target_points, persistent_entries)
            # 仅复用由“一拍多”正式批量粗定位写入的条目；旧版本的
            # 初始选孔/逐孔缓存没有可靠来源标记，强制重新建立批量缓存。
            rejected_non_batch = {
                int(hole_id): str(entry.cache_source)
                for hole_id, entry in matched_entries.items()
                if str(getattr(entry, "cache_source", "unknown")) != "batch_coarse_source"
            }
            for hole_id in rejected_non_batch:
                match_audit[int(hole_id)] = {
                    "matched": False,
                    "reason": "cache_source_not_batch_coarse",
                    "cache_source": rejected_non_batch[hole_id],
                }
            matched_entries = {
                int(hole_id): entry for hole_id, entry in matched_entries.items()
                if int(hole_id) not in rejected_non_batch
            }
            matched_source_ids = {
                int(hole_id): source_id for hole_id, source_id in matched_source_ids.items()
                if int(hole_id) not in rejected_non_batch
            }
            persistent_audit["match"] = match_audit
            cache_entries.update({
                int(hole_id): rekey_cache_entry(entry, int(hole_id))
                for hole_id, entry in matched_entries.items()
            })
            cache_sources.update({
                int(hole_id): "persistent_base_cache"
                for hole_id in matched_entries
            })
            cache_source_ids.update({
                int(hole_id): int(source_id)
                for hole_id, source_id in matched_source_ids.items()
            })
            report["coarse_cache"]["persistent_cache"] = persistent_audit
            report["coarse_cache"]["persistent_cache_loaded"] = sorted(
                int(hole_id) for hole_id in matched_entries
            )
            report["coarse_cache"]["persistent_cache_load_errors"] = {
                str(key): value for key, value in persistent_load_errors.items()
            }

        # 缓存只能由340 mm一拍多批量粗定位产生。初始选孔画面只提供导航
        # 锚点，不再在原地追加帧建立缓存，避免把低精度初始点云写成正式缓存。
        report["coarse_cache"]["cache_built"] = sorted(cache_built_ids)
        report["coarse_cache"]["cache_available"] = sorted(cache_entries)
        report["coarse_cache"]["initial_build_failures"] = {}
        report["coarse_cache"]["persistent_cache_ids"] = sorted(persistent_entries)
    report["coarse_cache"]["rgb_snapshot_images"] = [
        str(path)
        for path in sorted(coarse_cache_dir.glob("initial_cache_rgb_frame_*.png"))
        if path.is_file()
    ]
    report["stages"]["home_selection"]["coarse_cache_built"] = report[
        "coarse_cache"
    ].get("cache_built", [])
    report["stages"]["home_selection"]["coarse_cache_available"] = sorted(cache_entries)
    report["stages"]["home_selection"]["coarse_cache_failures"] = {}
    report["stages"]["home_selection"]["coarse_cache_rgb_images"] = report[
        "coarse_cache"
    ]["rgb_snapshot_images"]
    _write_report(run_dir, report, rows)

    result = _run_sequential_hole_workflow(
        args, handeye, model, cfg, run_dir, report, timing, rows, pipeline_runtime,
        pose_session, motion_session, current_tcp,
        initial_selected_holes, intrinsics,
        coarse_cache_entries=cache_entries,
        coarse_cache_sources=cache_sources,
        coarse_cache_source_ids=cache_source_ids,
        coarse_cache_dir=coarse_cache_dir,
        coarse_cache_gates=coarse_cache_gates,
        coarse_cache_metadata={
            "mode": "two_stage_hole_localization",
            "source": "initial_multi_hole_selection_or_fallback_refresh",
            "handeye_path": str(args.handeye),
            "camera_serial": str((report.get("camera") or {}).get("serial_number", "")),
            "gates": coarse_cache_gates.to_dict(),
            "surface_selection_policy": COARSE_SURFACE_SELECTION_POLICY,
            "surface_model": COARSE_SURFACE_MODEL,
        },
        persistent_cache_entries=persistent_entries,
        persistent_cache_dir=persistent_cache_dir,
        persistent_cache_metadata={
            "mode": "two_stage_hole_localization",
            "source": "batch_coarse_source",
            "handeye_path": str(args.handeye),
            "camera_serial": str((report.get("camera") or {}).get("serial_number", "")),
            "gates": coarse_cache_gates.to_dict(),
            "surface_selection_policy": COARSE_SURFACE_SELECTION_POLICY,
            "surface_model": COARSE_SURFACE_MODEL,
        },
    )
    return int(result)




def run_two_stage_hole_localization(args: Any, handeye: Any, model: Any) -> int:
    """持续运行旧两阶段流程：每轮完成后回原点并重新进入初始选孔。"""
    cfg = TwoStageConfig(
        coarse_height_mm=float(args.coarse_height_mm),
        fine_height_mm=float(args.fine_height_mm),
        coarse_frames=int(args.coarse_frames),
        fine_frames=int(args.fine_frames),
        coarse_settle_delay_s=float(args.coarse_settle_delay_s),
        coarse_recapture_settle_discard_frames=int(
            args.coarse_recapture_settle_discard_frames
        ),
        coarse_max_corrections=int(args.coarse_max_corrections),
        fine_settle_discard_frames=int(args.fine_settle_discard_frames),
        fine_retry_count=int(args.fine_retries),
        batch_coarse_localization=bool(args.batch_coarse_localization),
        batch_coarse_frames=int(getattr(args, "batch_coarse_frames", 15)),
        batch_coarse_min_valid=int(getattr(args, "batch_coarse_min_valid", 10)),
        batch_coarse_settle_discard_frames=int(args.batch_coarse_settle_discard_frames),
        batch_coarse_min_holes_per_frame=int(args.batch_coarse_min_holes_per_frame) if args.batch_coarse_min_holes_per_frame is not None else None,
        batch_coarse_view_margin_px=float(args.batch_coarse_view_margin_px),
        batch_fine_localization=bool(getattr(args, "batch_fine_localization", True)),
        batch_fine_frames=int(getattr(args, "batch_fine_frames", 8)),
        batch_fine_min_valid=int(getattr(args, "batch_fine_min_valid", 5)),
        batch_fine_stable_min_frames=int(getattr(args, "batch_fine_stable_min_frames", 5)),
        batch_fine_settle_discard_frames=int(getattr(args, "batch_fine_settle_discard_frames", 10)),
        batch_fine_supplement_rounds=int(
            getattr(args, "batch_fine_supplement_rounds", 1)
        ),
        batch_fine_view_margin_px=float(getattr(args, "batch_fine_view_margin_px", 50.0)),
        shared_cache_validation=bool(getattr(args, "shared_cache_validation", False)),
        shared_cache_validation_frames=int(args.shared_cache_validation_frames),
        shared_cache_validation_min_valid=int(args.shared_cache_validation_min_valid),
        shared_cache_validation_view_margin_px=float(args.shared_cache_validation_view_margin_px),
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
    if cfg.coarse_settle_delay_s < 0.0:
        raise ValueError("粗定位停稳缓冲时间不能小于0")
    if cfg.cache_validation_settle_discard_frames < 0:
        raise ValueError("缓存验证预热丢弃帧数不能小于0")
    if cfg.cache_validation_frames < 1:
        raise ValueError("缓存验证帧数必须大于0")
    if cfg.cache_validation_min_valid < 1 or (
        cfg.cache_validation_min_valid > cfg.cache_validation_frames
    ):
        raise ValueError("缓存验证最少有效帧数必须在验证帧数范围内")
    if cfg.shared_cache_validation:
        if not bool(getattr(args, "reuse_coarse_cache", True)):
            raise ValueError("共享缓存快速验证要求启用粗定位缓存复用")
        if cfg.shared_cache_validation_frames < 1:
            raise ValueError("共享缓存验证帧数必须大于0")
        if (
            cfg.shared_cache_validation_min_valid < 1
            or cfg.shared_cache_validation_min_valid > cfg.shared_cache_validation_frames
        ):
            raise ValueError("共享缓存验证最少有效帧数必须在验证帧数范围内")
        if (
            not math.isfinite(cfg.shared_cache_validation_view_margin_px)
            or cfg.shared_cache_validation_view_margin_px < 0.0
        ):
            raise ValueError("共享缓存验证视野边缘余量必须是大于等于0的有限数字")
        if cfg.batch_coarse_localization:
            raise ValueError("共享缓存快速验证不能与批量粗定位同时启用")
    if cfg.coarse_recapture_settle_discard_frames < 0:
        raise ValueError("粗定位重拍停稳丢弃帧数不能小于0")
    if cfg.coarse_max_corrections < 0:
        raise ValueError("粗定位最大姿态纠偏次数不能小于0")
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
    if cfg.batch_coarse_localization:
        if cfg.batch_coarse_frames < cfg.batch_coarse_min_valid:
            raise ValueError(
                f"批量粗定位最大帧数必须不少于最小有效帧数：{cfg.batch_coarse_frames} < {cfg.batch_coarse_min_valid}"
            )
        if cfg.batch_coarse_settle_discard_frames < 0:
            raise ValueError("批量粗定位停稳丢弃帧数不能小于0")
        if cfg.max_coarse_center_scatter_p95_px <= 0.0:
            raise ValueError("粗定位中心散布P95门限必须大于0")
        if cfg.max_coarse_tracking_distance_p95_px <= 0.0:
            raise ValueError("粗定位跟踪距离P95门限必须大于0")
        if cfg.batch_coarse_view_margin_px < 0:
            raise ValueError("批量粗定位视野边缘余量不能小于0")
    if cfg.batch_fine_view_margin_px < 0.0 or not math.isfinite(cfg.batch_fine_view_margin_px):
        raise ValueError("批量精定位视野边缘余量必须是大于等于0的有限数字")
    if cfg.batch_fine_localization:
        if cfg.batch_fine_frames < 1 or cfg.batch_fine_min_valid < 1:
            raise ValueError("批量精定位帧数和最少有效帧数必须大于0")
        if cfg.batch_fine_frames < cfg.batch_fine_min_valid:
            raise ValueError("批量精定位帧数必须不少于最少有效帧数")
        if (
            cfg.batch_fine_stable_min_frames < 1
            or cfg.batch_fine_stable_min_frames > cfg.batch_fine_frames
        ):
            raise ValueError("批量精定位稳定门帧数必须在批量帧数范围内")
        if cfg.batch_fine_settle_discard_frames < 0:
            raise ValueError("批量精定位停稳丢弃帧数不能小于0")
        if cfg.batch_fine_supplement_rounds < 0:
            raise ValueError("批量精定位共享补拍轮数不能小于0")
    cycle_index = 1
    run_dir = _new_two_stage_run_dir(cycle_index)
    report = _new_two_stage_report(args, handeye, cfg, run_dir, cycle_index)
    timing = TimingRecorder()
    timing.attach_report(report)
    rows: list[dict[str, Any]] = []
    rgbd_pipeline = None
    pipeline_runtime: dict[str, Any] = {
        "rgbd_pipeline": None,
        "align": None,
        "chain": None,
        "current_tcp": None,
    }
    pose_session = motion_session = None
    current_tcp: np.ndarray | None = None
    home = None
    camera_identity: dict[str, Any] = {}

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
        current_tcp = np.asarray(initial_tcp, dtype=np.float64).copy()
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
                    "本次仅可作实验验证，结果不会被标记为生产可用。",
                    flush=True,
                )
            with timing.measure("robot/connect_motion_session"):
                motion_session = AuboMotionSession()
                motion_session.connect(
                    ROBOT_CFG.ip,
                    ROBOT_CFG.rpc_port,
                    ROBOT_CFG.user,
                    ROBOT_CFG.password,
                    ROBOT_CFG.request_timeout_ms,
                )
            with timing.measure("robot/move_home"):
                current_tcp = _confirm_and_move_home(
                    home, args, motion_session, pose_session,
                )
        else:
            print(
                "[PREVIEW] 未指定 --execute：不会回原点或下发运动；"
                "只执行一轮预览。请手动将机器人置于原点后核对规划。",
                flush=True,
            )

        with timing.measure("camera/start_rgbd_pipeline"):
            report["stages"]["rgbd_startup"] = {"status": "starting"}
            rgbd_pipeline, align, chain = init_pipeline()
            pipeline_runtime.update({
                "rgbd_pipeline": rgbd_pipeline,
                "align": align,
                "chain": chain,
            })
            report["stages"]["rgbd_startup"] = {"status": "ready"}
            camera_identity = get_device_identity(rgbd_pipeline)
            expected = str(handeye.payload.get("camera_serial", "")).strip()
            actual = str(camera_identity.get("serial_number", "")).strip()
            if expected and actual and expected != actual:
                raise RuntimeError(f"相机序列号不匹配：手眼={expected}，当前={actual}")
            report["camera"] = camera_identity
        report["cycle_start_tcp_pose_m_rad"] = transform_to_sdk_pose_m_rad(current_tcp)
        _write_report(run_dir, report, rows)

        while True:
            if cycle_index > 1:
                run_dir = _new_two_stage_run_dir(cycle_index)
                report = _new_two_stage_report(args, handeye, cfg, run_dir, cycle_index)
                timing = TimingRecorder()
                timing.attach_report(report)
                rows = []
                report["robot_initial_tcp_pose_m_rad"] = (
                    transform_to_sdk_pose_m_rad(current_tcp)
                )
                report["home_point"] = home.to_dict()
                report["camera"] = camera_identity
                report["stages"]["rgbd_startup"] = {
                    "status": "shared_ready",
                    "reused_session": True,
                }
                report["cycle_start_tcp_pose_m_rad"] = (
                    transform_to_sdk_pose_m_rad(current_tcp)
                )
                _write_report(run_dir, report, rows)

            result = _run_two_stage_localization_cycle(
                args,
                handeye,
                model,
                cfg,
                run_dir,
                report,
                timing,
                rows,
                pipeline_runtime,
                pose_session,
                motion_session,
                np.asarray(current_tcp, dtype=np.float64),
                cycle_index,
            )
            if not args.execute:
                return int(result)

            current_tcp = np.asarray(
                pipeline_runtime.get("current_tcp"), dtype=np.float64,
            ).copy()
            pipeline_runtime["current_tcp"] = current_tcp.copy()
            report["return_to_home"] = {
                "completed": False,
                "reason": "waiting_for_next_cycle_confirmation_before_return_home",
                "tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
            }
            report["final_tcp_pose_m_rad"] = transform_to_sdk_pose_m_rad(current_tcp)
            report["next_cycle_ready"] = False
            report["next_cycle_confirmation_required"] = True
            report["status"] = (
                "completed_experimental_handeye_waiting_for_next_cycle_confirmation"
                if not handeye.validated_for_motion else
                "completed_waiting_for_next_cycle_confirmation"
            )
            _write_report(run_dir, report, rows)
            print(
                f"[NEXT_CYCLE_CONFIRM_REQUIRED] cycle={int(cycle_index)} 已完成；"
                "机器人保持当前位置，不会自动回到初始点。"
                "点击“开始下一轮检测”（命令行输入 m）后才回到初始点；"
                "回到初始点并完全停止后才会拍摄下一轮画面。",
                flush=True,
            )
            command = _request_next_cycle_confirmation(cycle_index)
            if command != "m":
                raise TwoStageSelectionCancelled("用户未确认开始下一轮检测")
            report["next_cycle_confirmation_required"] = False
            report["next_cycle_confirmed"] = True
            report["next_cycle_confirmed_at"] = datetime.now().isoformat(timespec="seconds")
            report["next_cycle_ready"] = False
            report["next_cycle_motion"] = "return_home_after_confirmation"
            _write_report(run_dir, report, rows)
            with timing.measure(
                "robot/return_home_after_cycle_confirmation",
                cycle_index=int(cycle_index),
            ):
                current_tcp = _confirm_and_move_home(
                    home, args, motion_session, pose_session,
                )
            with timing.measure(
                "robot/verify_home_steady_before_next_cycle_capture",
                cycle_index=int(cycle_index),
            ):
                _, current_tcp = _wait_robot_steady(pose_session)
            pipeline_runtime["current_tcp"] = current_tcp.copy()
            report["return_to_home"] = {
                "completed": True,
                "reason": "next_cycle_confirmation_received",
                "tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
            }
            report["next_cycle_ready"] = True
            report["next_cycle_capture_waits_for_steady"] = True
            _write_report(run_dir, report, rows)
            cycle_index += 1

    except TwoStageSelectionCancelled as exc:
        report["status"] = "stopped_by_user"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["next_cycle_ready"] = False
        report["session_stopped_by_user"] = True
        _write_report(run_dir, report, rows)
        print(f"[STOPPED] 用户结束重复选孔会话：{exc}", flush=True)
        return 0
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
                print(
                    f"[TIMING] 最终报告写入失败：{type(exc).__name__}: {exc}",
                    flush=True,
                )
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
                   help="显式启用真实运动；不传入时只预览，不连接运动控制")
    p.add_argument("--no-execute", dest="execute", action="store_false",
                   help="仅预览：不连接运动控制或下发机器人运动")
    p.add_argument("--allow-experimental-handeye", dest="allow_experimental_handeye", action="store_true",
                   default=DEFAULT_ALLOW_EXPERIMENTAL_HANDEYE,
                   help="显式允许当前实验手眼结果；默认只接受已验证手眼")
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
    p.add_argument(
        "--coarse-settle-delay-s", type=float, default=0.6,
        help="到达340 mm粗定位位后等待的停稳缓冲时间(秒)，默认0.6",
    )
    p.add_argument(
        "--coarse-recapture-settle-discard-frames",
        type=int,
        default=10,
        help="每次粗定位重拍前重新停稳后丢弃的RGB-D预热帧数，默认10",
    )
    p.add_argument(
        "--coarse-max-corrections",
        type=int,
        default=2,
        help="单孔粗定位最多姿态纠偏次数，默认2；最后一次采集只做验证",
    )
    p.add_argument(
        "--batch-coarse-localization",
        dest="batch_coarse_localization",
        action="store_true",
        default=False,
        help="批量粗定位模式：在340mm一次性检测所有选中孔的位姿和深度，然后按视野批量精定位",
    )
    p.add_argument(
        "--batch-coarse-frames",
        type=int,
        default=15,
        help="340mm共同位姿稳定连拍帧数，默认15；同一批帧覆盖全部选中孔",
    )
    p.add_argument(
        "--batch-coarse-min-valid",
        type=int,
        default=10,
        help="340mm共同粗定位每孔最少有效帧数，默认10",
    )
    p.add_argument(
        "--batch-coarse-settle-discard-frames",
        type=int,
        default=10,
        help="批量粗定位正式采集前丢弃的停稳预热RGB-D帧数，默认10",
    )
    p.add_argument(
        "--batch-coarse-min-holes-per-frame",
        type=int,
        default=None,
        help="批量粗定位模式每帧最少有效孔数，默认为None（表示全部选中孔）",
    )
    p.add_argument(
        "--batch-coarse-view-margin-px",
        type=float,
        default=50.0,
        help="批量粗定位共同位姿的视野边缘安全余量(px)，默认50",
    )
    p.add_argument(
        "--batch-fine-localization",
        dest="batch_fine_localization",
        action="store_true",
        default=True,
        help="260mm批量精定位：每个视野一次采集并精定位全部选中孔，默认开启",
    )
    p.add_argument(
        "--no-batch-fine-localization",
        dest="batch_fine_localization",
        action="store_false",
        help="关闭260mm批量精定位，恢复逐孔精定位",
    )
    p.add_argument(
        "--batch-fine-view-margin-px",
        type=float,
        default=50.0,
        help="260mm批量精定位共同位姿的视野边缘安全余量(px)，默认50",
    )
    p.add_argument(
        "--batch-fine-frames", type=int, default=8,
        help="260mm共同位姿严格几何圆心连拍最大帧数，默认8",
    )
    p.add_argument(
        "--batch-fine-min-valid", type=int, default=5,
        help="260mm共同精定位每孔最少严格几何圆心帧数，默认5",
    )
    p.add_argument(
        "--batch-fine-stable-min-frames", type=int, default=5,
        help="260mm共同精定位稳定验收所需严格几何圆心帧数，默认5",
    )
    p.add_argument(
        "--batch-fine-settle-discard-frames", type=int, default=10,
        help="260mm批量精定位正式采集前最少丢弃的停稳RGB帧数；随后自动确认已追上实时帧，默认10",
    )
    p.add_argument(
        "--batch-fine-supplement-rounds", type=int, default=1,
        help="260mm共享精拍质量不足时移动到失败孔共同观察位的补拍轮数，默认1",
    )
    p.add_argument(
        "--shared-cache-validation", action="store_true", default=False,
        help="固定点云复用：按共同视野分组，在每组340mm少帧验证后跳过通过孔的逐孔粗定位",
    )
    p.add_argument(
        "--shared-cache-validation-frames", type=int, default=3,
        help="共享缓存快速验证每组RGB-D帧数，默认3",
    )
    p.add_argument(
        "--shared-cache-validation-min-valid", type=int, default=2,
        help="共享缓存快速验证每孔最少有效帧数，默认2",
    )
    p.add_argument(
        "--shared-cache-validation-view-margin-px", type=float, default=50.0,
        help="共享缓存验证共同340mm位姿的视野边缘余量(px)，默认50",
    )
    p.add_argument(
        "--optimize-hole-order", action="store_true", default=False,
        help="按当前批量精定位目标XY的最近邻顺序处理孔；默认保持初始选择顺序",
    )
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
    p.add_argument(
        "--reuse-coarse-cache", dest="reuse_coarse_cache", action="store_true", default=True,
        help="旧两阶段流程复用本次运行初始多孔局部点云缓存；验证失败自动回退完整粗定位",
    )
    p.add_argument(
        "--no-reuse-coarse-cache", dest="reuse_coarse_cache", action="store_false",
        help="关闭旧两阶段局部点云缓存复用，保持原始粗定位流程",
    )
    p.add_argument(
        "--reuse-persistent-coarse-cache",
        dest="reuse_persistent_coarse_cache",
        action="store_true",
        default=True,
        help="旧两阶段按机器人基坐标复用跨运行粗定位缓存；验证失败自动回退",
    )
    p.add_argument(
        "--no-reuse-persistent-coarse-cache",
        dest="reuse_persistent_coarse_cache",
        action="store_false",
        help="关闭跨运行基坐标粗定位缓存，仅使用当前运行缓存",
    )
    p.add_argument("--move-final-xy", dest="move_final_xy", action="store_true", default=DEFAULT_MOVE_FINAL_XY,
                   help="显式启用精定位后的最终 TCP XY 微调")
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


# 实现已统一到 aubo_workbench.config.apply_robot_connection_overrides。
_apply_robot_connection_overrides = apply_robot_connection_overrides


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
