#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""孔中心定位精度诊断工具（PyCharm 直接运行版）。

将本文件和 run_yolo_eye_in_hand_optimized.py 放在同一目录。
只修改“用户配置区”的 TEST_MODE，然后在 PyCharm 中直接 Run。

测试模式：
- repeat       ：机器人不动，只测 RGB 椭圆圆心重复性和轮廓去畸变影响。
- plane_repeat ：机器人不动，测局部深度平面的点、法向和孔点稳定性。
- multiview    ：手动把机器人移动到不同姿态，每个姿态运行一次；比较同一孔在基坐标中的一致性。
- auto_multiview：选孔后自动在受限的 TCP 相对位姿上采集，测试完整 RGB-D/手眼链路。
- tilt_sim     ：纯几何仿真，验证主脚本的倾斜圆心修正是否能消除透视偏差。

诊断逻辑：
1. repeat 很差：先解决光照、曝光、轮廓提取和运动振动。
2. repeat 很好但 plane_repeat 很差：深度平面是主要瓶颈。
3. repeat、plane_repeat 都好，但 multiview 修正后仍差：重点检查手眼旋转、机器人姿态/TCP定义。
4. multiview 未修正差、修正后明显变好：主要是投影椭圆中心的倾斜几何偏差。
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from aubo_workbench.paths import (
    CAMERA_CALIBRATION_PATH,
    HANDEYE_CANDIDATE_PATH,
    HOLE_LOCALIZATION_RUNS_DIR,
    MODEL_PATH as DEFAULT_MODEL_PATH,
)


# ============================================================================
# 用户配置区：在 PyCharm 中只改这里
# ============================================================================

TEST_MODE = "auto_multiview"  # repeat / plane_repeat / multiview / auto_multiview / tilt_sim

TARGET_SCRIPT = Path(__file__).with_name("run_yolo_eye_in_hand_optimized.py")
MODEL_PATH = DEFAULT_MODEL_PATH
HANDEYE_PATH = HANDEYE_CANDIDATE_PATH
INTRINSICS_JSON = CAMERA_CALIBRATION_PATH
OUTPUT_ROOT = HOLE_LOCALIZATION_RUNS_DIR / "precision_diagnostics"

CONFIDENCE = 0.35
REPEAT_FRAMES = 60
PLANE_REPEAT_FRAMES = 40
MULTIVIEW_FRAMES = 20

# 当前精拍距离，仅用于把像素散布近似换算成毫米。
TEST_HEIGHT_MM = 260.0

# 多视角测试必须始终选择同一个物理孔。
MULTIVIEW_TAG = "pose_01"
SHOW_MULTIVIEW_HISTORY = True
KNOWN_HOLE_DIAMETER_MM = 70.0
USE_FIRST_VIEW_AS_REFERENCE_PLANE = True

# 自动多视角：选定孔后不再需要人工移动。每次非参考位姿采集后都会自动回到参考 TCP，
# 因此每一段实际运动都严格限制在下列范围内。
AUTO_CAPTURE_FRAMES = 20
AUTO_TRANSLATION_MM = 40.0
AUTO_ROTATION_DEG = 20.0
AUTO_MAX_TRANSLATION_MM = 50.0  # 硬上限：不能通过修改 AUTO_TRANSLATION_MM 绕过
AUTO_MAX_ROTATION_DEG = 40.0    # 硬上限：不能通过修改 AUTO_ROTATION_DEG 绕过
AUTO_SPEED_M_S = 0.020
AUTO_ACC_M_S2 = 0.080
AUTO_SETTLE_S = 0.50
AUTO_MAX_TARGET_PIXEL_ERROR_PX = 140.0
AUTO_MAX_ELLIPSE_RESIDUAL_PX = 0.80
AUTO_MIN_ELLIPSE_COVERAGE_DEG = 300.0
AUTO_MIN_VALID_FRAMES = 12
AUTO_RETURN_TO_REFERENCE = True

# 已成功插入后读取到的机器人 TCP 基坐标。它不是孔口中心真值，
# 仅用于测量“孔口中心 -> 插入 TCP”的几何偏移，绝不自动写成补偿。
KNOWN_INSERTED_TCP_BASE_MM: tuple[float, float, float] | None = (
    445.000, -158.210, -139.520,
)

# 只有拿到“孔口平面中心”的独立基坐标真值时才填入；否则保持 None。
# 该值才可用于评价相机/手眼的绝对误差。
ABSOLUTE_REFERENCE_HOLE_BASE_MM: tuple[float, float, float] | None = None

# tilt_sim 设置
SIM_TILT_DEG = (0.0, 0.5, 1.0, 2.0, 3.0, 10.0, 20.0)
SIM_HEIGHT_MM = 260.0
SIM_DIAMETER_MM = 70.0
SIM_CENTER_OFFSET_PX = (0.0, 0.0)

WAIT_BEFORE_EXIT = True


# ============================================================================
# 通用工具
# ============================================================================


def load_locator_module() -> Any:
    if not TARGET_SCRIPT.is_file():
        raise FileNotFoundError(f"找不到主脚本：{TARGET_SCRIPT}")
    spec = importlib.util.spec_from_file_location("hole_locator_optimized", TARGET_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载主脚本：{TARGET_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12 or not math.isfinite(norm):
        raise ValueError("无法归一化向量")
    return vector / norm


def angle_deg(a: np.ndarray, b: np.ndarray, unsigned: bool = False) -> float:
    cosine = float(unit(a) @ unit(b))
    if unsigned:
        cosine = abs(cosine)
    return float(math.degrees(math.acos(np.clip(cosine, -1.0, 1.0))))


@dataclass(frozen=True)
class AutoView:
    """参考 TCP 局部坐标系下的一次自动诊断视角。"""

    label: str
    translation_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_deg_rxyz: tuple[float, float, float] = (0.0, 0.0, 0.0)


def rotation_matrix_rxyz_deg(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """与主脚本一致，按 Rz @ Ry @ Rx 构造局部相对旋转。"""
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = math.cos(float(rx)), math.sin(float(rx))
    cy, sy = math.cos(float(ry)), math.sin(float(ry))
    cz, sz = math.cos(float(rz)), math.sin(float(rz))
    R_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    R_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    R_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return R_z @ R_y @ R_x


def relative_rotation_deg(R_from: np.ndarray, R_to: np.ndarray) -> float:
    R_delta = np.asarray(R_from, dtype=np.float64).T @ np.asarray(R_to, dtype=np.float64)
    cosine = float(np.clip((np.trace(R_delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def automatic_view_sequence() -> tuple[AutoView, ...]:
    """返回对称视角；每个非参考视角均从参考 TCP 往返，避免累计大位移。"""
    t = float(AUTO_TRANSLATION_MM)
    r = float(AUTO_ROTATION_DEG)
    return (
        AutoView("x_plus", (t, 0.0, 0.0)),
        AutoView("x_minus", (-t, 0.0, 0.0)),
        AutoView("y_plus", (0.0, t, 0.0)),
        AutoView("y_minus", (0.0, -t, 0.0)),
        AutoView("z_plus", (0.0, 0.0, t)),
        AutoView("z_minus", (0.0, 0.0, -t)),
        AutoView("rx_plus", rotation_deg_rxyz=(r, 0.0, 0.0)),
        AutoView("rx_minus", rotation_deg_rxyz=(-r, 0.0, 0.0)),
        AutoView("ry_plus", rotation_deg_rxyz=(0.0, r, 0.0)),
        AutoView("ry_minus", rotation_deg_rxyz=(0.0, -r, 0.0)),
        AutoView("rz_plus", rotation_deg_rxyz=(0.0, 0.0, r)),
        AutoView("rz_minus", rotation_deg_rxyz=(0.0, 0.0, -r)),
    )


def validate_automatic_views(views: tuple[AutoView, ...]) -> None:
    if not views:
        raise ValueError("自动多视角序列为空")
    for view in views:
        translation = np.asarray(view.translation_mm, dtype=np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError(f"{view.label} 的平移参数无效")
        translation_norm = float(np.linalg.norm(translation))
        if translation_norm > float(AUTO_MAX_TRANSLATION_MM) + 1e-9:
            raise ValueError(
                f"{view.label} 平移 {translation_norm:.3f} mm 超过硬上限 "
                f"{AUTO_MAX_TRANSLATION_MM:.1f} mm"
            )
        rotation = rotation_matrix_rxyz_deg(*view.rotation_deg_rxyz)
        rotation_deg = relative_rotation_deg(np.eye(3), rotation)
        if rotation_deg > float(AUTO_MAX_ROTATION_DEG) + 1e-9:
            raise ValueError(
                f"{view.label} 转角 {rotation_deg:.3f}° 超过硬上限 "
                f"{AUTO_MAX_ROTATION_DEG:.1f}°"
            )


def automatic_target_pose(T_base_tcp_reference: np.ndarray, view: AutoView) -> np.ndarray:
    """把局部相对位姿映射为基坐标 TCP 目标；不修改参考 TCP 本身。"""
    T_reference = np.asarray(T_base_tcp_reference, dtype=np.float64).reshape(4, 4)
    T_relative = np.eye(4, dtype=np.float64)
    T_relative[:3, :3] = rotation_matrix_rxyz_deg(*view.rotation_deg_rxyz)
    T_relative[:3, 3] = np.asarray(view.translation_mm, dtype=np.float64)
    return T_reference @ T_relative


def project_base_point_to_distorted_pixel(point_base_mm: np.ndarray, intrinsics: Any,
                                          T_base_tcp: np.ndarray,
                                          T_tcp_camera: np.ndarray) -> np.ndarray | None:
    """用当前手眼预测同一物理孔的 RGB 像素，仅用于从同类孔中锁定目标。"""
    T_base_camera = np.asarray(T_base_tcp, dtype=np.float64) @ np.asarray(T_tcp_camera, dtype=np.float64)
    point_camera = T_base_camera[:3, :3].T @ (
        np.asarray(point_base_mm, dtype=np.float64).reshape(3) - T_base_camera[:3, 3]
    )
    if not np.isfinite(point_camera).all() or float(point_camera[2]) <= 1e-6:
        return None
    K = np.array(
        [[float(intrinsics.fx), 0.0, float(intrinsics.cx)],
         [0.0, float(intrinsics.fy), float(intrinsics.cy)],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.asarray(getattr(intrinsics, "distortion", ()), dtype=np.float64).reshape(-1)
    pixels, _ = cv2.projectPoints(
        point_camera.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K,
        None if distortion.size == 0 else distortion,
    )
    return pixels.reshape(2)


def robust_center_stats(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("统计数据为空")
    center = np.median(points, axis=0)
    residual = points - center
    distance = np.linalg.norm(residual, axis=1)
    return {
        "count": int(len(points)),
        "median": center,
        "axis_std": np.std(points, axis=0),
        "axis_range": np.ptp(points, axis=0),
        "radial_rms": float(np.sqrt(np.mean(distance ** 2))),
        "radial_p95": float(np.percentile(distance, 95)),
        "radial_max": float(np.max(distance)),
    }


def make_run_dir(mode: str) -> Path:
    path = OUTPUT_ROOT / f"{mode}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def load_intrinsics_from_json(locator: Any) -> Any:
    from aubo_workbench.camera import CameraIntrinsics

    if not INTRINSICS_JSON.is_file():
        # 使用用户实际读取到的1280x800 SDK内参作为后备值。
        return CameraIntrinsics(
            1280, 800,
            610.048522949, 610.270812988,
            648.242248535, 406.749450684,
            (-0.033531755208969116, 0.03746772184967995,
             0.00020212080562487245, -9.991253318730742e-05,
             -0.01323756854981184, 0.0, 0.0, 0.0),
        )

    payload = json.loads(INTRINSICS_JSON.read_text(encoding="utf-8"))
    data = payload.get("intrinsics", payload)
    return CameraIntrinsics(
        int(data["width"]), int(data["height"]),
        float(data["fx"]), float(data["fy"]),
        float(data["cx"]), float(data["cy"]),
        tuple(float(v) for v in data.get("distortion", [])),
    )


def acquire_rgb_target(locator: Any, pipeline: Any) -> tuple[Any, dict[str, Any]]:
    bundle = None
    for _ in range(30):
        bundle = locator.get_rgb_frame_bundle(pipeline)
        if bundle is not None and bundle.intrinsics is not None:
            break
    if bundle is None or bundle.intrinsics is None:
        raise RuntimeError("无法读取RGB帧或内参")
    detections = locator.detect(locator._DIAG_MODEL, bundle.color_bgr, CONFIDENCE)
    if not detections:
        raise RuntimeError("YOLO没有检测到孔")
    selected = locator.choose_box(bundle.color_bgr, detections)
    if selected is None:
        raise RuntimeError("用户取消选孔")
    return bundle, detections[selected]


def acquire_rgbd_target(locator: Any, pipeline: Any, align: Any, chain: Any) -> tuple[Any, dict[str, Any]]:
    bundle = None
    for _ in range(30):
        bundle = locator.get_aligned_frame_bundle(pipeline, align, chain)
        if bundle is not None and bundle.intrinsics is not None:
            break
    if bundle is None or bundle.intrinsics is None:
        raise RuntimeError("无法读取RGB-D对齐帧或内参")
    detections = locator.detect(locator._DIAG_MODEL, bundle.color_bgr, CONFIDENCE)
    if not detections:
        raise RuntimeError("YOLO没有检测到孔")
    selected = locator.choose_box(bundle.color_bgr, detections)
    if selected is None:
        raise RuntimeError("用户取消选孔")
    return bundle, detections[selected]


# ============================================================================
# repeat：纯RGB重复性 + 畸变管线差异
# ============================================================================


def run_repeat(locator: Any) -> dict[str, Any]:
    run_dir = make_run_dir("repeat")
    pipeline = None
    try:
        pipeline = locator.init_rgb_handeye_pipeline()
        initial, chosen = acquire_rgb_target(locator, pipeline)
        anchor = np.asarray(chosen["center"], dtype=np.float64)
        optimized: list[np.ndarray] = []
        legacy: list[np.ndarray] = []
        shifts: list[np.ndarray] = []
        residuals: list[float] = []
        coverages: list[float] = []
        last_image = initial.color_bgr
        last_detection = chosen
        last_ellipse = None

        attempts = 0
        while len(optimized) < REPEAT_FRAMES and attempts < REPEAT_FRAMES * 4:
            attempts += 1
            bundle = locator.get_rgb_frame_bundle(pipeline)
            if bundle is None or bundle.intrinsics is None:
                continue
            detections = locator.detect(locator._DIAG_MODEL, bundle.color_bgr, CONFIDENCE)
            detection = locator._nearest_detection(detections, anchor, chosen["class_id"])
            if detection is None:
                continue
            ellipse = locator.fit_hole_ellipse(bundle.color_bgr, detection, bundle.intrinsics)
            if ellipse is None:
                continue
            optimized.append(np.asarray(ellipse["center_px"], dtype=np.float64))
            legacy.append(np.asarray(ellipse["legacy_center_px_undistorted"], dtype=np.float64))
            shifts.append(np.asarray(ellipse["contour_undistortion_shift_px"], dtype=np.float64))
            residuals.append(float(ellipse["residual_px"]))
            coverages.append(float(ellipse["coverage_deg"]))
            anchor = np.asarray(ellipse["center_px_distorted"], dtype=np.float64)
            last_image, last_detection, last_ellipse = bundle.color_bgr, detection, ellipse

        if len(optimized) < max(10, REPEAT_FRAMES // 2):
            raise RuntimeError(f"有效椭圆帧不足：{len(optimized)}/{REPEAT_FRAMES}")

        opt = np.asarray(optimized)
        leg = np.asarray(legacy)
        shift = np.asarray(shifts)
        intrinsics = initial.intrinsics
        mm_per_px_x = TEST_HEIGHT_MM / float(intrinsics.fx)
        mm_per_px_y = TEST_HEIGHT_MM / float(intrinsics.fy)
        opt_stats = robust_center_stats(opt)
        legacy_stats = robust_center_stats(leg)
        shift_stats = robust_center_stats(shift)

        report = {
            "mode": "repeat",
            "height_for_px_to_mm": TEST_HEIGHT_MM,
            "valid_frames": len(opt),
            "optimized_center_px": opt_stats,
            "legacy_center_px": legacy_stats,
            "contour_undistortion_shift_px": shift_stats,
            "optimized_axis_std_mm_approx": [
                float(opt_stats["axis_std"][0] * mm_per_px_x),
                float(opt_stats["axis_std"][1] * mm_per_px_y),
            ],
            "optimized_radial_p95_mm_approx": float(
                opt_stats["radial_p95"] * 0.5 * (mm_per_px_x + mm_per_px_y)
            ),
            "undistortion_shift_median_mm_approx": [
                float(np.median(shift[:, 0]) * mm_per_px_x),
                float(np.median(shift[:, 1]) * mm_per_px_y),
            ],
            "ellipse_residual_median_px": float(np.median(residuals)),
            "ellipse_residual_p95_px": float(np.percentile(residuals, 95)),
            "coverage_median_deg": float(np.median(coverages)),
        }
        cv2.imwrite(str(run_dir / "last_overlay.png"), locator._overlay(last_image, last_detection, last_ellipse, "repeat"))
        save_json(run_dir / "report.json", report)
        print(json.dumps(jsonable(report), ensure_ascii=False, indent=2))
        print(f"[DONE] {run_dir}")
        return report
    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()


# ============================================================================
# plane_repeat：深度平面稳定性
# ============================================================================


def run_plane_repeat(locator: Any) -> dict[str, Any]:
    run_dir = make_run_dir("plane_repeat")
    pipeline = align = chain = None
    try:
        pipeline, align, chain = locator.init_pipeline()
        initial, chosen = acquire_rgbd_target(locator, pipeline, align, chain)
        anchor = np.asarray(chosen["center"], dtype=np.float64)
        plane_points: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        hole_points: list[np.ndarray] = []
        plane_rmse: list[float] = []
        ring_counts: list[int] = []
        last_image = initial.color_bgr
        last_detection = chosen
        last_ellipse = None

        attempts = 0
        while len(hole_points) < PLANE_REPEAT_FRAMES and attempts < PLANE_REPEAT_FRAMES * 5:
            attempts += 1
            bundle = locator.get_aligned_frame_bundle(pipeline, align, chain)
            if bundle is None or bundle.intrinsics is None:
                continue
            detections = locator.detect(locator._DIAG_MODEL, bundle.color_bgr, CONFIDENCE)
            detection = locator._nearest_detection(detections, anchor, chosen["class_id"])
            if detection is None:
                continue
            ellipse = locator.fit_hole_ellipse(bundle.color_bgr, detection, bundle.intrinsics)
            if ellipse is None:
                continue
            radius = max(detection["box"][2] - detection["box"][0], detection["box"][3] - detection["box"][1]) / 2.0
            try:
                point, info = locator.hole_camera_point(
                    tuple(ellipse["center_px_distorted"]),
                    bundle.xyz_map_mm,
                    bundle.intrinsics,
                    radius,
                    ray_center_xy=np.asarray(ellipse["center_px"]),
                    ray_center_is_undistorted=True,
                )
            except Exception:
                continue
            plane_points.append(np.asarray(info["plane_point_camera_mm"], dtype=np.float64))
            normals.append(unit(np.asarray(info["plane_normal_camera"], dtype=np.float64)))
            hole_points.append(np.asarray(point, dtype=np.float64))
            plane_rmse.append(float(info["plane_rmse_mm"]))
            ring_counts.append(int(info["ring_points"]))
            anchor = np.asarray(ellipse["center_px_distorted"], dtype=np.float64)
            last_image, last_detection, last_ellipse = bundle.color_bgr, detection, ellipse

        if len(hole_points) < max(10, PLANE_REPEAT_FRAMES // 2):
            raise RuntimeError(f"有效深度帧不足：{len(hole_points)}/{PLANE_REPEAT_FRAMES}")

        reference = unit(normals[0])
        aligned_normals = np.asarray([n if float(n @ reference) >= 0 else -n for n in normals])
        median_normal = unit(np.median(aligned_normals, axis=0))
        normal_angles = np.asarray([angle_deg(n, median_normal) for n in aligned_normals])
        plane_stats = robust_center_stats(np.asarray(plane_points))
        hole_stats = robust_center_stats(np.asarray(hole_points))

        report = {
            "mode": "plane_repeat",
            "valid_frames": len(hole_points),
            "plane_point_camera_mm": plane_stats,
            "hole_point_camera_mm": hole_stats,
            "normal_median": median_normal,
            "normal_scatter_p95_deg": float(np.percentile(normal_angles, 95)),
            "normal_scatter_max_deg": float(np.max(normal_angles)),
            "plane_rmse_median_mm": float(np.median(plane_rmse)),
            "plane_rmse_p95_mm": float(np.percentile(plane_rmse, 95)),
            "ring_points_median": float(np.median(ring_counts)),
        }
        cv2.imwrite(str(run_dir / "last_overlay.png"), locator._overlay(last_image, last_detection, last_ellipse, "plane repeat"))
        save_json(run_dir / "report.json", report)
        print(json.dumps(jsonable(report), ensure_ascii=False, indent=2))
        print(f"[DONE] {run_dir}")
        return report
    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()


# ============================================================================
# multiview：同一孔多姿态一致性
# ============================================================================


def summarize_multiview(records: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    values = [np.asarray(record[field], dtype=np.float64) for record in records
              if record.get(field) is not None]
    if len(values) < 2:
        return None
    return robust_center_stats(np.asarray(values))


def summarize_absolute_errors(records: list[dict[str, Any]], field: str,
                              truth_base_mm: np.ndarray) -> dict[str, Any] | None:
    """相对外部真值的绝对误差；正方向定义为估计值减真值。"""
    estimates = [np.asarray(record[field], dtype=np.float64) for record in records
                 if record.get(field) is not None]
    if not estimates:
        return None
    truth = np.asarray(truth_base_mm, dtype=np.float64).reshape(3)
    errors = np.asarray(estimates, dtype=np.float64) - truth
    norms = np.linalg.norm(errors, axis=1)
    fixed_bias = np.median(errors, axis=0)
    debiased_norms = np.linalg.norm(errors - fixed_bias, axis=1)
    return {
        "count": int(len(errors)),
        "truth_base_mm": truth,
        "median_estimate_base_mm": np.median(np.asarray(estimates), axis=0),
        "median_error_estimate_minus_truth_mm": fixed_bias,
        "median_error_norm_mm": float(np.median(norms)),
        "p95_error_norm_mm": float(np.percentile(norms, 95)),
        "max_error_norm_mm": float(np.max(norms)),
        "axis_error_std_mm": np.std(errors, axis=0, ddof=0),
        "debiased_scatter_rms_mm": float(np.sqrt(np.mean(debiased_norms ** 2))),
        "debiased_scatter_p95_mm": float(np.percentile(debiased_norms, 95)),
    }


def summarize_absolute_groups(records: list[dict[str, Any]], field: str,
                              truth_base_mm: np.ndarray) -> dict[str, Any]:
    groups = {
        "translation": {"x_plus", "x_minus", "y_plus", "y_minus", "z_plus", "z_minus"},
        "roll_pitch": {"rx_plus", "rx_minus", "ry_plus", "ry_minus"},
        "yaw": {"rz_plus", "rz_minus"},
    }
    return {
        name: summarize_absolute_errors([record for record in records if record.get("label") in labels],
                                        field, truth_base_mm)
        for name, labels in groups.items()
    }


def diagnose_absolute_bias(stats: dict[str, Any] | None) -> dict[str, Any] | None:
    """把绝对偏置和去偏置后的跨视角离散分开，避免把两者混为相机精度。"""
    if stats is None:
        return None
    bias = np.asarray(stats["median_error_estimate_minus_truth_mm"], dtype=np.float64)
    norm = float(np.linalg.norm(bias))
    energy = bias ** 2
    axis_index = int(np.argmax(energy))
    axis_names = ("X", "Y", "Z")
    return {
        "fixed_bias_estimate_minus_truth_mm": bias,
        "fixed_bias_norm_mm": norm,
        "dominant_bias_axis": axis_names[axis_index],
        "dominant_axis_energy_ratio": float(energy[axis_index] / max(float(np.sum(energy)), 1e-12)),
        "relative_scatter_after_removing_fixed_bias_rms_mm": stats["debiased_scatter_rms_mm"],
        "relative_scatter_after_removing_fixed_bias_p95_mm": stats["debiased_scatter_p95_mm"],
        "fixed_bias_to_relative_scatter_ratio": float(
            norm / max(float(stats["debiased_scatter_rms_mm"]), 1e-12)
        ),
    }


def summarize_scalar(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "mad": float(np.median(np.abs(array - np.median(array)))),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def estimate_circle_range_from_rgb_mm(axes_px: np.ndarray, intrinsics: Any,
                                      diameter_mm: float) -> float:
    """由已知圆孔直径和去畸变椭圆轴反推名义相机距离。

    只作为独立诊断：孔面接近正对相机时该值应与深度 Z 接近；不参与孔点计算。
    """
    axes = np.asarray(axes_px, dtype=np.float64).reshape(2)
    if np.any(axes <= 1e-9) or diameter_mm <= 0.0:
        raise ValueError("圆孔直径或椭圆轴无效")
    estimates = np.array([
        float(intrinsics.fx) * float(diameter_mm) / axes[0],
        float(intrinsics.fy) * float(diameter_mm) / axes[1],
    ])
    return float(np.median(estimates))


def estimate_circle_diameter_from_rgb_depth_mm(axes_px: np.ndarray, intrinsics: Any,
                                                depth_z_mm: float) -> float:
    """由 RGB 椭圆尺寸和已对齐深度 Z 反推孔径，供 65/70/75 mm 类别核查。"""
    axes = np.asarray(axes_px, dtype=np.float64).reshape(2)
    if np.any(axes <= 1e-9) or depth_z_mm <= 0.0:
        raise ValueError("深度 Z 或椭圆轴无效")
    estimates = np.array([
        float(depth_z_mm) * axes[0] / float(intrinsics.fx),
        float(depth_z_mm) * axes[1] / float(intrinsics.fy),
    ])
    return float(np.median(estimates))


def run_multiview(locator: Any) -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    history_path = OUTPUT_ROOT / "multiview_consistency.json"
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        history = {"reference_plane": None, "records": []}

    pipeline = align = chain = None
    pose_session = None
    run_dir = make_run_dir("multiview")
    try:
        handeye = locator.load_handeye_experiment_result(HANDEYE_PATH)
        from aubo_workbench.robot import AuboPoseSession

        pose_session = AuboPoseSession()
        pose_session.connect()
        snapshot, T_base_tcp = locator._require_safe_snapshot(pose_session)

        pipeline, align, chain = locator.init_pipeline()
        initial, chosen = acquire_rgbd_target(locator, pipeline, align, chain)
        anchor = np.asarray(chosen["center"], dtype=np.float64)
        centers: list[np.ndarray] = []
        plane_points_camera: list[np.ndarray] = []
        normals_camera: list[np.ndarray] = []
        axes: list[np.ndarray] = []
        plane_rmse: list[float] = []
        last_image = initial.color_bgr
        last_detection = chosen
        last_ellipse = None

        attempts = 0
        while len(centers) < MULTIVIEW_FRAMES and attempts < MULTIVIEW_FRAMES * 5:
            attempts += 1
            bundle = locator.get_aligned_frame_bundle(pipeline, align, chain)
            if bundle is None or bundle.intrinsics is None:
                continue
            detections = locator.detect(locator._DIAG_MODEL, bundle.color_bgr, CONFIDENCE)
            detection = locator._nearest_detection(detections, anchor, chosen["class_id"])
            if detection is None:
                continue
            ellipse = locator.fit_hole_ellipse(bundle.color_bgr, detection, bundle.intrinsics)
            if ellipse is None:
                continue
            radius = max(detection["box"][2] - detection["box"][0], detection["box"][3] - detection["box"][1]) / 2.0
            try:
                _, info = locator.hole_camera_point(
                    tuple(ellipse["center_px_distorted"]), bundle.xyz_map_mm, bundle.intrinsics, radius,
                    ray_center_xy=np.asarray(ellipse["center_px"]), ray_center_is_undistorted=True,
                )
            except Exception:
                continue
            centers.append(np.asarray(ellipse["center_px"], dtype=np.float64))
            plane_points_camera.append(np.asarray(info["plane_point_camera_mm"], dtype=np.float64))
            normals_camera.append(unit(np.asarray(info["plane_normal_camera"], dtype=np.float64)))
            axes.append(np.asarray(ellipse["axes_px"], dtype=np.float64))
            plane_rmse.append(float(info["plane_rmse_mm"]))
            anchor = np.asarray(ellipse["center_px_distorted"], dtype=np.float64)
            last_image, last_detection, last_ellipse = bundle.color_bgr, detection, ellipse

        if len(centers) < max(8, MULTIVIEW_FRAMES // 2):
            raise RuntimeError(f"有效多视角采样帧不足：{len(centers)}/{MULTIVIEW_FRAMES}")

        center_px = np.median(np.asarray(centers), axis=0)
        plane_point_camera = np.median(np.asarray(plane_points_camera), axis=0)
        reference_normal = unit(normals_camera[0])
        aligned_normals = np.asarray([n if float(n @ reference_normal) >= 0 else -n for n in normals_camera])
        plane_normal_camera = unit(np.median(aligned_normals, axis=0))
        axes_px = np.median(np.asarray(axes), axis=0)

        T_base_camera = locator.camera_transform(T_base_tcp, handeye.T_tcp_rgb_camera)
        current_plane_point_base = T_base_camera[:3, :3] @ plane_point_camera + T_base_camera[:3, 3]
        current_plane_normal_base = unit(T_base_camera[:3, :3] @ plane_normal_camera)

        if history.get("reference_plane") is None or not USE_FIRST_VIEW_AS_REFERENCE_PLANE:
            reference_plane = {
                "point_base_mm": current_plane_point_base.tolist(),
                "normal_base": current_plane_normal_base.tolist(),
                "created_from_tag": MULTIVIEW_TAG,
            }
            if USE_FIRST_VIEW_AS_REFERENCE_PLANE:
                history["reference_plane"] = reference_plane
        else:
            reference_plane = history["reference_plane"]

        plane_point_base = np.asarray(reference_plane["point_base_mm"], dtype=np.float64)
        plane_normal_base = unit(np.asarray(reference_plane["normal_base"], dtype=np.float64))
        intrinsics = initial.intrinsics

        naive = locator.pixel_to_base_plane(
            center_px, intrinsics, T_base_tcp, handeye.T_tcp_rgb_camera,
            plane_point_base, plane_normal_base, center_is_undistorted=True,
        )
        corrected, correction = locator.correct_projected_circle_center(
            center_px, intrinsics, T_base_tcp, handeye.T_tcp_rgb_camera,
            plane_point_base, plane_normal_base, KNOWN_HOLE_DIAMETER_MM,
        )
        estimated_height = locator.camera_height_to_plane_mm(
            T_base_tcp, handeye.T_tcp_rgb_camera, plane_point_base,
        )
        current_plane_normal_distance = float(
            plane_normal_base @ (current_plane_point_base - plane_point_base)
        )
        current_plane_angle = angle_deg(current_plane_normal_base, plane_normal_base, unsigned=True)

        record = {
            "tag": MULTIVIEW_TAG,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "tcp_pose_m_rad": snapshot["pose_values_sdk_m_rad"],
            "valid_frames": len(centers),
            "center_px_undistorted": center_px,
            "axes_px_undistorted": axes_px,
            "estimated_height_mm": estimated_height,
            "plane_rmse_median_mm": float(np.median(plane_rmse)),
            "current_plane_point_base_mm": current_plane_point_base,
            "current_plane_normal_base": current_plane_normal_base,
            "current_plane_offset_from_reference_mm": current_plane_normal_distance,
            "current_plane_angle_from_reference_deg": current_plane_angle,
            "hole_point_base_naive_mm": naive,
            "hole_point_base_corrected_mm": corrected,
            "tilt_correction": correction,
        }
        history.setdefault("records", []).append(jsonable(record))
        save_json(history_path, history)

        raw_stats = summarize_multiview(history["records"], "hole_point_base_naive_mm")
        corrected_stats = summarize_multiview(history["records"], "hole_point_base_corrected_mm")
        summary = {
            "latest": record,
            "history_count": len(history["records"]),
            "naive_multiview_stats_mm": raw_stats,
            "corrected_multiview_stats_mm": corrected_stats,
            "history_path": history_path,
        }
        cv2.imwrite(str(run_dir / "overlay.png"), locator._overlay(last_image, last_detection, last_ellipse, MULTIVIEW_TAG))
        save_json(run_dir / "report.json", summary)
        print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))
        print(f"[DONE] {run_dir}")
        if SHOW_MULTIVIEW_HISTORY:
            print(f"[HISTORY] {history_path}")
        return summary
    finally:
        if pipeline is not None:
            pipeline.stop()
        if pose_session is not None:
            pose_session.disconnect()
        cv2.destroyAllWindows()


# ============================================================================
# auto_multiview：自动受限多视角完整链路测试
# ============================================================================


class TargetNotVisibleError(RuntimeError):
    """目标孔在当前测试视角无法获得足够可靠帧；该视角可跳过。"""


def _capture_auto_view(locator: Any, pipeline: Any, align: Any, chain: Any,
                       chosen: dict[str, Any], expected_center_px: np.ndarray | None) -> tuple[dict[str, Any], Any, Any, Any]:
    """在一个稳定 TCP 位姿采集多帧，返回融合观测及最后一张叠加图所需数据。"""
    centers: list[np.ndarray] = []
    plane_points_camera: list[np.ndarray] = []
    normals_camera: list[np.ndarray] = []
    axes: list[np.ndarray] = []
    plane_rmse: list[float] = []
    pixel_prediction_errors: list[float] = []
    last_image = last_detection = last_ellipse = None

    max_attempts = int(AUTO_CAPTURE_FRAMES) * 5
    for _ in range(max_attempts):
        if len(centers) >= int(AUTO_CAPTURE_FRAMES):
            break
        bundle = locator.get_aligned_frame_bundle(pipeline, align, chain)
        if bundle is None or bundle.intrinsics is None:
            continue
        detections = locator.detect(locator._DIAG_MODEL, bundle.color_bgr, CONFIDENCE)
        if not detections:
            continue
        anchor = (np.asarray(expected_center_px, dtype=np.float64)
                  if expected_center_px is not None else np.asarray(chosen["center"], dtype=np.float64))
        detection = locator._nearest_detection(detections, anchor, chosen["class_id"])
        if detection is None:
            continue
        prediction_error = float(np.linalg.norm(np.asarray(detection["center"], dtype=np.float64) - anchor))
        if expected_center_px is not None and prediction_error > float(AUTO_MAX_TARGET_PIXEL_ERROR_PX):
            continue
        ellipse = locator.fit_hole_ellipse(bundle.color_bgr, detection, bundle.intrinsics)
        if ellipse is None:
            continue
        if (float(ellipse["residual_px"]) > float(AUTO_MAX_ELLIPSE_RESIDUAL_PX)
                or float(ellipse["coverage_deg"]) < float(AUTO_MIN_ELLIPSE_COVERAGE_DEG)):
            continue
        radius = max(detection["box"][2] - detection["box"][0], detection["box"][3] - detection["box"][1]) / 2.0
        try:
            _, info = locator.hole_camera_point(
                tuple(ellipse["center_px_distorted"]), bundle.xyz_map_mm, bundle.intrinsics, radius,
                ray_center_xy=np.asarray(ellipse["center_px"]), ray_center_is_undistorted=True,
            )
        except Exception:
            continue
        centers.append(np.asarray(ellipse["center_px"], dtype=np.float64))
        plane_points_camera.append(np.asarray(info["plane_point_camera_mm"], dtype=np.float64))
        normals_camera.append(unit(np.asarray(info["plane_normal_camera"], dtype=np.float64)))
        axes.append(np.asarray(ellipse["axes_px"], dtype=np.float64))
        plane_rmse.append(float(info["plane_rmse_mm"]))
        pixel_prediction_errors.append(prediction_error)
        last_image, last_detection, last_ellipse = bundle.color_bgr, detection, ellipse

    if len(centers) < int(AUTO_MIN_VALID_FRAMES):
        raise TargetNotVisibleError(
            f"自动视角有效帧不足：{len(centers)}/{AUTO_CAPTURE_FRAMES}，"
            f"至少需要 {AUTO_MIN_VALID_FRAMES} 帧"
        )

    reference_normal = unit(normals_camera[0])
    aligned_normals = np.asarray([
        normal if float(normal @ reference_normal) >= 0.0 else -normal
        for normal in normals_camera
    ])
    observation = {
        "valid_frames": len(centers),
        "center_px_undistorted": np.median(np.asarray(centers), axis=0),
        "axes_px_undistorted": np.median(np.asarray(axes), axis=0),
        "plane_point_camera_mm": np.median(np.asarray(plane_points_camera), axis=0),
        "plane_normal_camera": unit(np.median(aligned_normals, axis=0)),
        "plane_rmse_median_mm": float(np.median(plane_rmse)),
        "plane_rmse_p95_mm": float(np.percentile(plane_rmse, 95)),
        "target_pixel_prediction_error_median_px": float(np.median(pixel_prediction_errors)),
        "target_pixel_prediction_error_p95_px": float(np.percentile(pixel_prediction_errors, 95)),
    }
    return observation, last_image, last_detection, last_ellipse


def _auto_move_to(locator: Any, motion_session: Any, pose_session: Any,
                  target: np.ndarray, label: str) -> tuple[dict[str, Any], np.ndarray]:
    """低速直线运动并检查控制器状态；调用者保证目标来自受限参考位姿。"""
    _, current = locator._require_safe_snapshot(pose_session)
    delta_mm = float(np.linalg.norm(target[:3, 3] - current[:3, 3]))
    delta_deg = relative_rotation_deg(current[:3, :3], target[:3, :3])
    if delta_mm > float(AUTO_MAX_TRANSLATION_MM) + 1e-6:
        raise RuntimeError(f"{label} 单段平移 {delta_mm:.3f} mm 超过硬上限 {AUTO_MAX_TRANSLATION_MM:.1f} mm")
    if delta_deg > float(AUTO_MAX_ROTATION_DEG) + 1e-6:
        raise RuntimeError(f"{label} 单段转角 {delta_deg:.3f}° 超过硬上限 {AUTO_MAX_ROTATION_DEG:.1f}°")
    response = motion_session.move_line(
        locator.transform_to_sdk_pose_m_rad(target), float(AUTO_SPEED_M_S), float(AUTO_ACC_M_S2),
    )
    from aubo_workbench.motion_control import sdk_ok
    if not response or not sdk_ok(response[-1]):
        raise RuntimeError(f"{label} moveLine 下发失败：{response}")
    snapshot, actual = locator._wait_robot_steady(pose_session, timeout_s=8.0)
    time.sleep(float(AUTO_SETTLE_S))
    snapshot, actual = locator._require_safe_snapshot(pose_session)
    return snapshot, actual


def needs_reference_return(current_tcp: np.ndarray, reference_tcp: np.ndarray,
                           position_tolerance_mm: float = 0.10,
                           rotation_tolerance_deg: float = 0.02) -> bool:
    """避免向已处于参考位姿的控制器重复下发 moveLine。"""
    current = np.asarray(current_tcp, dtype=np.float64).reshape(4, 4)
    reference = np.asarray(reference_tcp, dtype=np.float64).reshape(4, 4)
    position_error = float(np.linalg.norm(current[:3, 3] - reference[:3, 3]))
    rotation_error = relative_rotation_deg(current[:3, :3], reference[:3, :3])
    return position_error > position_tolerance_mm or rotation_error > rotation_tolerance_deg


def _auto_record(locator: Any, label: str, planned_tcp: np.ndarray, actual_tcp: np.ndarray,
                 reference_tcp: np.ndarray, observation: dict[str, Any], handeye: Any,
                 intrinsics: Any, reference_plane_point_base: np.ndarray,
                 reference_plane_normal_base: np.ndarray,
                 robot_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    """把一个视角转成可直接横向比较的基坐标记录。"""
    T_base_camera = np.asarray(actual_tcp, dtype=np.float64) @ np.asarray(handeye.T_tcp_rgb_camera, dtype=np.float64)
    current_plane_point_base = T_base_camera[:3, :3] @ observation["plane_point_camera_mm"] + T_base_camera[:3, 3]
    current_plane_normal_base = unit(T_base_camera[:3, :3] @ observation["plane_normal_camera"])
    naive = None
    corrected = None
    correction: Any = None
    correction_error: str | None = None
    try:
        # 这里按冻结的参考孔面回投，才能直接测到手眼/姿态带来的跨视角差异。
        naive = locator.pixel_to_base_plane(
            observation["center_px_undistorted"], intrinsics, actual_tcp, handeye.T_tcp_rgb_camera,
            reference_plane_point_base, reference_plane_normal_base, center_is_undistorted=True,
        )
        corrected, correction = locator.correct_projected_circle_center(
            observation["center_px_undistorted"], intrinsics, actual_tcp, handeye.T_tcp_rgb_camera,
            reference_plane_point_base, reference_plane_normal_base, KNOWN_HOLE_DIAMETER_MM,
        )
    except Exception as exc:
        correction_error = f"{type(exc).__name__}: {exc}"
        if naive is None:
            raise
    absolute_truth = (None if ABSOLUTE_REFERENCE_HOLE_BASE_MM is None
                      else np.asarray(ABSOLUTE_REFERENCE_HOLE_BASE_MM, dtype=np.float64))
    absolute_error = (None if corrected is None or absolute_truth is None
                      else np.asarray(corrected, dtype=np.float64) - absolute_truth)
    inserted_tcp_reference = (None if KNOWN_INSERTED_TCP_BASE_MM is None
                              else np.asarray(KNOWN_INSERTED_TCP_BASE_MM, dtype=np.float64))
    mouth_to_inserted_tcp = (None if corrected is None or inserted_tcp_reference is None
                             else inserted_tcp_reference - np.asarray(corrected, dtype=np.float64))
    insertion_along_normal = (None if mouth_to_inserted_tcp is None
                              else float(mouth_to_inserted_tcp @ reference_plane_normal_base))
    insertion_tangential = (None if mouth_to_inserted_tcp is None else float(np.linalg.norm(
        mouth_to_inserted_tcp - insertion_along_normal * reference_plane_normal_base
    )))
    rgb_circle_range = estimate_circle_range_from_rgb_mm(
        observation["axes_px_undistorted"], intrinsics, KNOWN_HOLE_DIAMETER_MM,
    )
    depth_plane_z = float(observation["plane_point_camera_mm"][2])
    rgb_depth_equivalent_diameter = estimate_circle_diameter_from_rgb_depth_mm(
        observation["axes_px_undistorted"], intrinsics, depth_plane_z,
    )
    return {
        "label": label,
        "planned_tcp_pose_m_rad": locator.transform_to_sdk_pose_m_rad(planned_tcp),
        "actual_tcp_pose_m_rad": locator.transform_to_sdk_pose_m_rad(actual_tcp),
        "plan_position_error_mm": float(np.linalg.norm(planned_tcp[:3, 3] - actual_tcp[:3, 3])),
        "plan_rotation_error_deg": relative_rotation_deg(planned_tcp[:3, :3], actual_tcp[:3, :3]),
        "reference_tcp_translation_mm": float(np.linalg.norm(actual_tcp[:3, 3] - reference_tcp[:3, 3])),
        "reference_tcp_rotation_deg": relative_rotation_deg(reference_tcp[:3, :3], actual_tcp[:3, :3]),
        "valid_frames": int(observation["valid_frames"]),
        "center_px_undistorted": observation["center_px_undistorted"],
        "axes_px_undistorted": observation["axes_px_undistorted"],
        "plane_rmse_median_mm": float(observation["plane_rmse_median_mm"]),
        "plane_rmse_p95_mm": float(observation["plane_rmse_p95_mm"]),
        "target_pixel_prediction_error_median_px": float(observation["target_pixel_prediction_error_median_px"]),
        "target_pixel_prediction_error_p95_px": float(observation["target_pixel_prediction_error_p95_px"]),
        "plane_point_camera_mm": observation["plane_point_camera_mm"],
        "camera_to_plane_z_mm": depth_plane_z,
        "rgb_circle_nominal_range_mm": rgb_circle_range,
        "depth_z_minus_rgb_circle_range_mm": depth_plane_z - rgb_circle_range,
        # 轮廓可能落在倒角或内缘；这是有效椭圆直径，不等同于名义孔径类别。
        "rgb_depth_effective_ellipse_diameter_mm": rgb_depth_equivalent_diameter,
        "current_plane_point_base_mm": current_plane_point_base,
        "current_plane_normal_base": current_plane_normal_base,
        "current_plane_offset_from_reference_mm": float(
            reference_plane_normal_base @ (current_plane_point_base - reference_plane_point_base)
        ),
        "current_plane_angle_from_reference_deg": angle_deg(
            current_plane_normal_base, reference_plane_normal_base, unsigned=True,
        ),
        "hole_point_base_naive_mm": naive,
        "hole_point_base_corrected_mm": corrected,
        "absolute_truth_hole_base_mm": absolute_truth,
        "absolute_error_estimate_minus_truth_mm": absolute_error,
        "absolute_error_norm_mm": (None if absolute_error is None
                                   else float(np.linalg.norm(absolute_error))),
        "known_inserted_tcp_base_mm": inserted_tcp_reference,
        "mouth_to_inserted_tcp_vector_base_mm": mouth_to_inserted_tcp,
        "mouth_to_inserted_tcp_norm_mm": (None if mouth_to_inserted_tcp is None
                                            else float(np.linalg.norm(mouth_to_inserted_tcp))),
        "mouth_to_inserted_tcp_along_plane_normal_mm": insertion_along_normal,
        "mouth_to_inserted_tcp_tangential_mm": insertion_tangential,
        "tilt_correction": correction,
        "tilt_correction_error": correction_error,
        "robot_pose_source": None if robot_snapshot is None else robot_snapshot.get("pose_source"),
        "robot_tcp_pose_sdk_m_rad": None if robot_snapshot is None else robot_snapshot.get("tcp_pose_sdk_m_rad"),
        "robot_tool_pose_sdk_m_rad": None if robot_snapshot is None else robot_snapshot.get("tool_pose_sdk_m_rad"),
        "robot_actual_tcp_offset_sdk_m_rad": None if robot_snapshot is None else robot_snapshot.get("actual_tcp_offset_sdk_m_rad"),
        "robot_configured_tcp_offset_sdk_m_rad": (
            None if robot_snapshot is None else robot_snapshot.get("configured_tcp_offset_sdk_m_rad")
        ),
    }
def run_auto_multiview(locator: Any) -> dict[str, Any]:
    """选孔后自动执行参考位姿 + 12 个受限视角，并返回参考位姿。"""
    views = automatic_view_sequence()
    validate_automatic_views(views)
    from aubo_workbench.config import ROBOT_CFG
    from aubo_workbench.motion_control import AuboMotionSession
    from aubo_workbench.robot import AuboPoseSession

    run_dir = make_run_dir("auto-multiview")
    pipeline = align = chain = None
    pose_session = motion_session = None
    reference_tcp: np.ndarray | None = None
    collision_seen = False
    report: dict[str, Any] = {
        "mode": "auto_multiview",
        "status": "running",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experimental_handeye": True,
        "constraints": {
            "translation_hard_limit_mm": AUTO_MAX_TRANSLATION_MM,
            "rotation_hard_limit_deg": AUTO_MAX_ROTATION_DEG,
            "translation_command_mm": AUTO_TRANSLATION_MM,
            "rotation_command_deg": AUTO_ROTATION_DEG,
            "speed_m_s": AUTO_SPEED_M_S,
            "acc_m_s2": AUTO_ACC_M_S2,
            "capture_frames_per_view": AUTO_CAPTURE_FRAMES,
        },
        "planned_views": [view.__dict__ for view in views],
        "absolute_reference_hole_base_mm": ABSOLUTE_REFERENCE_HOLE_BASE_MM,
        "known_inserted_tcp_base_mm": KNOWN_INSERTED_TCP_BASE_MM,
        "records": [],
        "skipped_views": [],
    }
    try:
        handeye = locator.load_handeye_experiment_result(HANDEYE_PATH)
        report["handeye_path"] = str(HANDEYE_PATH)
        report["handeye_validated_for_motion"] = bool(handeye.validated_for_motion)
        pose_session = AuboPoseSession()
        pose_session.connect()
        motion_session = AuboMotionSession()
        motion_session.connect(
            ROBOT_CFG.ip, int(ROBOT_CFG.rpc_port), ROBOT_CFG.user,
            ROBOT_CFG.password, int(ROBOT_CFG.request_timeout_ms),
        )
        reference_snapshot, reference_tcp = locator._require_safe_snapshot(pose_session)
        pipeline, align, chain = locator.init_pipeline()
        initial, chosen = acquire_rgbd_target(locator, pipeline, align, chain)
        report["selected_hole"] = chosen
        report["reference_tcp_pose_m_rad"] = locator.transform_to_sdk_pose_m_rad(reference_tcp)

        print("[AUTO] 已选孔；开始自动多视角测试。每个视角均从参考 TCP 往返。")
        print("[AUTO] 视角:", ", ".join(view.label for view in views))
        reference_observation, image, detection, ellipse = _capture_auto_view(
            locator, pipeline, align, chain, chosen, np.asarray(chosen["center"], dtype=np.float64),
        )
        reference_camera = locator.camera_transform(reference_tcp, handeye.T_tcp_rgb_camera)
        reference_plane_point_base = (
            reference_camera[:3, :3] @ reference_observation["plane_point_camera_mm"]
            + reference_camera[:3, 3]
        )
        reference_plane_normal_base = unit(
            reference_camera[:3, :3] @ reference_observation["plane_normal_camera"]
        )
        reference_hole_base = locator.pixel_to_base_plane(
            reference_observation["center_px_undistorted"], initial.intrinsics,
            reference_tcp, handeye.T_tcp_rgb_camera,
            reference_plane_point_base, reference_plane_normal_base,
            center_is_undistorted=True,
        )
        reference_record = _auto_record(
            locator,
            "reference_start", reference_tcp, reference_tcp, reference_tcp, reference_observation,
            handeye, initial.intrinsics, reference_plane_point_base, reference_plane_normal_base,
            reference_snapshot,
        )
        report["records"].append(jsonable(reference_record))
        cv2.imwrite(str(run_dir / "00_reference_start_overlay.png"), locator._overlay(image, detection, ellipse, "auto reference start"))

        for index, view in enumerate(views, start=1):
            target = automatic_target_pose(reference_tcp, view)
            arrived_at_view = False
            actual_tcp: np.ndarray | None = None
            try:
                view_snapshot, actual_tcp = _auto_move_to(
                    locator, motion_session, pose_session, target, f"去 {view.label}"
                )
                arrived_at_view = True
                expected_pixel = project_base_point_to_distorted_pixel(
                    reference_hole_base, initial.intrinsics, actual_tcp, handeye.T_tcp_rgb_camera,
                )
                observation, image, detection, ellipse = _capture_auto_view(
                    locator, pipeline, align, chain, chosen, expected_pixel,
                )
                record = _auto_record(
                    locator,
                    view.label, target, actual_tcp, reference_tcp, observation,
                    handeye, initial.intrinsics, reference_plane_point_base, reference_plane_normal_base,
                    view_snapshot,
                )
                report["records"].append(jsonable(record))
                cv2.imwrite(str(run_dir / f"{index:02d}_{view.label}_overlay.png"),
                            locator._overlay(image, detection, ellipse, f"auto {view.label}"))
            except TargetNotVisibleError as exc:
                report["skipped_views"].append(jsonable({
                    "label": view.label,
                    "reason": str(exc),
                    "planned_tcp_pose_m_rad": locator.transform_to_sdk_pose_m_rad(target),
                    "actual_tcp_pose_m_rad": (
                        None if actual_tcp is None else locator.transform_to_sdk_pose_m_rad(actual_tcp)
                    ),
                }))
                print(f"[SKIP] {view.label}：当前位姿未获得有效目标，返回参考位姿并继续。{exc}")
            finally:
                # 只有“目标未见”会被跳过；返程失败、碰撞或机器人状态异常必须停止。
                if arrived_at_view:
                    _auto_move_to(
                        locator, motion_session, pose_session, reference_tcp, f"{view.label} 返回参考位姿"
                    )
                save_json(run_dir / "report.partial.json", report)

        end_snapshot, actual_reference_tcp = locator._require_safe_snapshot(pose_session)
        expected_reference_pixel = project_base_point_to_distorted_pixel(
            reference_hole_base, initial.intrinsics, actual_reference_tcp, handeye.T_tcp_rgb_camera,
        )
        end_record: dict[str, Any] | None = None
        try:
            end_observation, image, detection, ellipse = _capture_auto_view(
                locator, pipeline, align, chain, chosen, expected_reference_pixel,
            )
            end_record = _auto_record(
                locator,
                "reference_end", reference_tcp, actual_reference_tcp, reference_tcp, end_observation,
                handeye, initial.intrinsics, reference_plane_point_base, reference_plane_normal_base,
                end_snapshot,
            )
            report["records"].append(jsonable(end_record))
            cv2.imwrite(str(run_dir / "99_reference_end_overlay.png"), locator._overlay(image, detection, ellipse, "auto reference end"))
        except TargetNotVisibleError as exc:
            report["skipped_views"].append(jsonable({
                "label": "reference_end",
                "reason": str(exc),
                "planned_tcp_pose_m_rad": locator.transform_to_sdk_pose_m_rad(reference_tcp),
                "actual_tcp_pose_m_rad": locator.transform_to_sdk_pose_m_rad(actual_reference_tcp),
            }))
            print(f"[SKIP] reference_end：未获得有效目标，保留已有视角结果。{exc}")

        naive_stats = summarize_multiview(report["records"], "hole_point_base_naive_mm")
        corrected_stats = summarize_multiview(report["records"], "hole_point_base_corrected_mm")
        absolute_stats = None
        absolute_groups: dict[str, Any] | None = None
        absolute_diagnosis = None
        insertion_tcp_offset_stats = None
        if ABSOLUTE_REFERENCE_HOLE_BASE_MM is not None:
            truth = np.asarray(ABSOLUTE_REFERENCE_HOLE_BASE_MM, dtype=np.float64)
            absolute_stats = summarize_absolute_errors(report["records"], "hole_point_base_corrected_mm", truth)
            absolute_groups = summarize_absolute_groups(report["records"], "hole_point_base_corrected_mm", truth)
            absolute_diagnosis = diagnose_absolute_bias(absolute_stats)
        if KNOWN_INSERTED_TCP_BASE_MM is not None:
            insertion_tcp_offset_stats = {
                "mouth_to_inserted_tcp_vector_base_mm": summarize_multiview(
                    report["records"], "mouth_to_inserted_tcp_vector_base_mm"
                ),
                "along_plane_normal_mm": summarize_scalar([
                    float(record["mouth_to_inserted_tcp_along_plane_normal_mm"])
                    for record in report["records"]
                    if record.get("mouth_to_inserted_tcp_along_plane_normal_mm") is not None
                ]),
                "tangential_mm": summarize_scalar([
                    float(record["mouth_to_inserted_tcp_tangential_mm"])
                    for record in report["records"]
                    if record.get("mouth_to_inserted_tcp_tangential_mm") is not None
                ]),
            }
        depth_rgb_range_stats = summarize_scalar([
            float(record["depth_z_minus_rgb_circle_range_mm"])
            for record in report["records"]
            if record.get("depth_z_minus_rgb_circle_range_mm") is not None
        ])
        rgb_depth_diameter_stats = summarize_scalar([
            float(record["rgb_depth_effective_ellipse_diameter_mm"])
            for record in report["records"]
            if record.get("rgb_depth_effective_ellipse_diameter_mm") is not None
        ])
        report.update({
            "status": "completed_with_skips" if report["skipped_views"] else "completed",
            "reference_plane_point_base_mm": reference_plane_point_base,
            "reference_plane_normal_base": reference_plane_normal_base,
            "reference_hole_base_mm": reference_hole_base,
            "naive_multiview_stats_mm": naive_stats,
            "corrected_multiview_stats_mm": corrected_stats,
            "absolute_corrected_error_stats_mm": absolute_stats,
            "absolute_corrected_error_groups_mm": absolute_groups,
            "absolute_error_diagnosis": absolute_diagnosis,
            "depth_z_minus_rgb_circle_range_stats_mm": depth_rgb_range_stats,
            "rgb_depth_effective_ellipse_diameter_stats_mm": rgb_depth_diameter_stats,
            "mouth_to_inserted_tcp_offset_stats": insertion_tcp_offset_stats,
            "reference_end_hole_drift_mm": float(np.linalg.norm(
                np.asarray(end_record["hole_point_base_corrected_mm"], dtype=np.float64)
                - np.asarray(reference_record["hole_point_base_corrected_mm"], dtype=np.float64)
            )) if (end_record is not None
                  and end_record.get("hole_point_base_corrected_mm") is not None
                  and reference_record.get("hole_point_base_corrected_mm") is not None) else None,
        })
        save_json(run_dir / "report.json", report)
        locator._write_csv(run_dir / "per_pose.csv", report["records"])
        print("\n视角                 有效帧   修正后孔点(mm)                     绝对误差(mm)  平面偏移(mm)  平面角(°)")
        for row in report["records"]:
            point = row.get("hole_point_base_corrected_mm")
            point_text = "-" if point is None else str(np.round(np.asarray(point, dtype=np.float64), 3).tolist())
            absolute_text = "-" if row.get("absolute_error_norm_mm") is None else f"{row['absolute_error_norm_mm']:.3f}"
            print(f"{row['label']:<20} {row['valid_frames']:>3}    {point_text:<34} "
                  f"{absolute_text:>10}  {row['current_plane_offset_from_reference_mm']:>9.3f}  "
                  f"{row['current_plane_angle_from_reference_deg']:>8.3f}")
        print(json.dumps(jsonable({
            "naive": naive_stats, "corrected": corrected_stats,
            "absolute_corrected": absolute_stats,
            "absolute_diagnosis": absolute_diagnosis,
            "depth_z_minus_rgb_circle_range": depth_rgb_range_stats,
            "rgb_depth_effective_ellipse_diameter": rgb_depth_diameter_stats,
            "mouth_to_inserted_tcp_offset": insertion_tcp_offset_stats,
            "reference_end_hole_drift_mm": report["reference_end_hole_drift_mm"],
        }), ensure_ascii=False, indent=2))
        print(f"[DONE] {run_dir}")
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        if pose_session is not None:
            try:
                snapshot, _ = locator._require_safe_snapshot(pose_session)
                collision_seen = bool(snapshot.get("collision", False))
            except Exception:
                pass
        save_json(run_dir / "report.json", report)
        raise
    finally:
        if (AUTO_RETURN_TO_REFERENCE and reference_tcp is not None and motion_session is not None
                and pose_session is not None and not collision_seen):
            try:
                _, current_tcp = locator._require_safe_snapshot(pose_session)
                if needs_reference_return(current_tcp, reference_tcp):
                    _auto_move_to(locator, motion_session, pose_session, reference_tcp, "异常/结束后返回参考位姿")
                else:
                    print("[AUTO] 已在参考位姿，跳过重复返回命令。")
            except Exception as exc:
                print(f"[WARN] 未能自动返回参考位姿：{exc}")
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        if pose_session is not None:
            pose_session.disconnect()
        if motion_session is not None:
            motion_session.disconnect()
        cv2.destroyAllWindows()


# ============================================================================
# tilt_sim：验证精确几何修正
# ============================================================================


def run_tilt_sim(locator: Any) -> dict[str, Any]:
    intrinsics = load_intrinsics_from_json(locator)
    identity = np.eye(4, dtype=np.float64)
    offset_x = SIM_CENTER_OFFSET_PX[0] * SIM_HEIGHT_MM / float(intrinsics.fx)
    offset_y = SIM_CENTER_OFFSET_PX[1] * SIM_HEIGHT_MM / float(intrinsics.fy)
    true_center = np.array([offset_x, offset_y, SIM_HEIGHT_MM], dtype=np.float64)
    rows = []

    for tilt in SIM_TILT_DEG:
        angle = math.radians(float(tilt))
        # 绕相机X轴倾斜；法向符号对平面和修正结果无影响。
        normal = unit(np.array([0.0, math.sin(angle), -math.cos(angle)], dtype=np.float64))
        axis_x, axis_y = locator._plane_basis(normal)
        theta = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
        radius = SIM_DIAMETER_MM / 2.0
        circle = (
            true_center[None, :]
            + radius * np.cos(theta)[:, None] * axis_x[None, :]
            + radius * np.sin(theta)[:, None] * axis_y[None, :]
        )
        projected = locator._project_undistorted_pixels(circle, intrinsics)
        ellipse = cv2.fitEllipse(projected.astype(np.float32).reshape(-1, 1, 2))
        observed_center = np.asarray(ellipse[0], dtype=np.float64)
        naive = locator.pixel_to_base_plane(
            observed_center, intrinsics, identity, identity,
            true_center, normal, center_is_undistorted=True,
        )
        corrected, diagnostics = locator.correct_projected_circle_center(
            observed_center, intrinsics, identity, identity,
            true_center, normal, SIM_DIAMETER_MM,
        )
        rows.append({
            "tilt_deg": float(tilt),
            "observed_ellipse_center_px": observed_center,
            "naive_error_mm": float(np.linalg.norm(naive - true_center)),
            "corrected_error_mm": float(np.linalg.norm(corrected - true_center)),
            "applied_correction_mm": diagnostics["correction_norm_mm"],
            "bias_px": diagnostics["ellipse_center_bias_px"],
        })

    report = {
        "mode": "tilt_sim",
        "height_mm": SIM_HEIGHT_MM,
        "diameter_mm": SIM_DIAMETER_MM,
        "center_offset_px": SIM_CENTER_OFFSET_PX,
        "rows": rows,
    }
    run_dir = make_run_dir("tilt_sim")
    save_json(run_dir / "report.json", report)
    print("倾斜角 | 未修正误差(mm) | 修正后误差(mm) | 施加修正(mm)")
    for row in rows:
        print(
            f"{row['tilt_deg']:7.2f} | {row['naive_error_mm']:14.6f} | "
            f"{row['corrected_error_mm']:14.6f} | {row['applied_correction_mm']:12.6f}"
        )
    print(f"[DONE] {run_dir}")
    return report


# ============================================================================
# 入口
# ============================================================================


def main() -> None:
    locator = load_locator_module()
    locator._DIAG_MODEL = locator.load_yolo(MODEL_PATH) if TEST_MODE != "tilt_sim" else None

    if TEST_MODE == "repeat":
        run_repeat(locator)
    elif TEST_MODE == "plane_repeat":
        run_plane_repeat(locator)
    elif TEST_MODE == "multiview":
        run_multiview(locator)
    elif TEST_MODE == "auto_multiview":
        run_auto_multiview(locator)
    elif TEST_MODE == "tilt_sim":
        run_tilt_sim(locator)
    else:
        raise ValueError(f"不支持的 TEST_MODE：{TEST_MODE}")


if __name__ == "__main__":
    try:
        main()
    finally:
        if WAIT_BEFORE_EXIT:
            try:
                input("\n按 Enter 结束程序...")
            except EOFError:
                pass
