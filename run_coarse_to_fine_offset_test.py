#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""粗定位后精定位视野偏移容忍度测试。

该脚本是独立诊断入口，不执行正式流程的最终 XY、最终 Z 和基坐标 Y+0.2 mm
动作。它先按正式流程完成单孔粗定位并下降到精定位高度，然后把相机在自身
X/Y 平面内横向偏移，使孔洞分别落在 0/5/10/15/20 mm 的圆形范围内，逐点
运行当前 RGB 精定位质量门，输出 JSON、CSV 和极坐标图。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_yolo_eye_in_hand_optimized as loc  # noqa: E402
from aubo_workbench.camera import get_device_identity, init_pipeline  # noqa: E402
from aubo_workbench.charuco_point_experiment import load_handeye_experiment_result  # noqa: E402
from aubo_workbench.config import ROBOT_CFG  # noqa: E402


DEFAULT_RADII_MM = (0.0, 5.0, 10.0, 15.0, 20.0)
DEFAULT_ANGLES_DEG = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
PLOT_SIZE = 900


def build_offset_plan(
    radii_mm: Iterable[float] = DEFAULT_RADII_MM,
    angles_deg: Iterable[float] = DEFAULT_ANGLES_DEG,
) -> list[dict[str, Any]]:
    """生成不重复的圆环采样计划，半径0只采中心点一次。"""
    radii = sorted({round(float(value), 6) for value in radii_mm})
    if not radii or any(value < 0.0 for value in radii):
        raise ValueError("偏移半径必须是非负数，且至少提供一个半径")
    angles = [float(value) % 360.0 for value in angles_deg]
    if not angles:
        raise ValueError("非零半径至少需要一个方向")

    plan: list[dict[str, Any]] = []
    sample_id = 1
    for radius in radii:
        direction_values = [0.0] if math.isclose(radius, 0.0, abs_tol=1e-9) else angles
        for angle_deg in direction_values:
            angle_rad = math.radians(angle_deg)
            offset = np.array(
                [radius * math.cos(angle_rad), radius * math.sin(angle_rad)],
                dtype=np.float64,
            )
            plan.append({
                "sample_id": sample_id,
                "radius_mm": radius,
                "angle_deg": float(angle_deg),
                "angle_rad": float(angle_rad),
                "offset_camera_xy_mm": offset.tolist(),
            })
            sample_id += 1
    return plan


def plan_camera_plane_offset_tcp(
    reference_tcp: np.ndarray,
    T_tcp_camera: np.ndarray,
    offset_camera_xy_mm: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    """规划目标TCP，使孔洞在相机图像中产生指定的横向偏移。

    ``offset_camera_xy_mm`` 定义为孔洞相对于相机主点的期望偏移。
    因此相机需要沿相反方向移动；姿态、相机光轴和相机到孔面的距离不变。
    """
    reference = np.asarray(reference_tcp, dtype=np.float64).reshape(4, 4)
    T_base_camera = loc.camera_transform(reference, T_tcp_camera)
    offset = np.asarray(list(offset_camera_xy_mm), dtype=np.float64).reshape(2)
    camera_lateral_delta_base = -T_base_camera[:3, :2] @ offset
    target = reference.copy()
    target[:3, 3] += camera_lateral_delta_base
    return target, camera_lateral_delta_base


def _expected_center_offset_px(
    hole_center_base_mm: np.ndarray,
    T_base_camera: np.ndarray,
    intrinsics: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回畸变像素、去畸变像素以及相对主点的去畸变像素偏移。"""
    distorted = loc._project_base_point_to_pixel(
        hole_center_base_mm, T_base_camera, intrinsics,
    )
    undistorted = loc.undistort_pixels(
        intrinsics, distorted.reshape(1, 2), pixel_output=True,
    )[0]
    principal = np.array([float(intrinsics.cx), float(intrinsics.cy)], dtype=np.float64)
    return distorted, undistorted, undistorted - principal


def _populate_initial_base_geometry(
    hole: dict[str, Any], current_tcp: np.ndarray, handeye: Any,
) -> None:
    T_base_camera = loc.camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
    point_camera = np.asarray(hole["initial_point_camera_mm"], dtype=np.float64).reshape(3)
    plane_point_camera = np.asarray(
        hole["initial_plane_point_camera_mm"], dtype=np.float64,
    ).reshape(3)
    normal_camera = loc._unit(
        np.asarray(hole["initial_plane_normal_camera"], dtype=np.float64),
        "initial hole normal",
    )
    point_base = T_base_camera[:3, :3] @ point_camera + T_base_camera[:3, 3]
    normal_base = loc._unit(
        T_base_camera[:3, :3] @ normal_camera, "initial hole base normal",
    )
    camera_origin = T_base_camera[:3, 3]
    if float(normal_base @ (camera_origin - point_base)) < 0.0:
        normal_base = -normal_base
    hole.update({
        "initial_center_base_mm": point_base.tolist(),
        "initial_plane_point_base_mm": (
            T_base_camera[:3, :3] @ plane_point_camera + T_base_camera[:3, 3]
        ).tolist(),
        "initial_plane_normal_base": normal_base.tolist(),
        "tracking_identity": "offset_test_initial_hole_1",
        "tracking_events": [],
    })


def _run_single_hole_coarse(
    *,
    args: Any,
    cfg: Any,
    handeye: Any,
    model: Any,
    hole: dict[str, Any],
    current_tcp: np.ndarray,
    fixed_rz_rad: float,
    pipeline: Any,
    align: Any,
    chain: Any,
    intrinsics: Any,
    run_dir: Path,
    timing: loc.TimingRecorder,
    rows: list[dict[str, Any]],
    motion_session: Any,
    pose_session: Any,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """复用正式流程的单孔粗定位闭环，但不进入最终目标点运动。"""
    point_base = np.asarray(hole["initial_center_base_mm"], dtype=np.float64).reshape(3)
    normal_base = loc._unit(
        np.asarray(hole["initial_plane_normal_base"], dtype=np.float64),
        "offset test initial normal",
    )
    coarse_target, coarse_geometry = loc._plan_hole_tcp_pose_fixed_rz(
        point_base, normal_base, current_tcp, handeye.T_tcp_rgb_camera,
        fixed_rz_rad=fixed_rz_rad, camera_height_mm=cfg.coarse_height_mm,
    )
    hole["coarse_target_tcp_pose_m_rad"] = loc.transform_to_sdk_pose_m_rad(coarse_target)
    hole["coarse_pose_geometry"] = coarse_geometry
    current_tcp = loc._confirm_and_move_line(
        "偏移测试：移动到孔上方340 mm粗定位位",
        current_tcp, coarse_target, args, motion_session, pose_session,
        "仅用于测试粗定位闭环，不执行最终目标点运动",
        require_confirmation=False, motion_profile="transit",
    )

    captures: list[dict[str, Any]] = []
    final_center_offset = math.inf
    final_normal_error = math.inf
    for capture_index in range(1, 4):
        T_base_camera = loc.camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
        tracking_point_base = np.asarray(
            hole.get("coarse_center_base_mm", point_base), dtype=np.float64,
        ).reshape(3)
        expected_anchor_px = loc._project_base_point_to_pixel(
            tracking_point_base, T_base_camera, intrinsics,
        )
        capture_name = f"offset_test_coarse_{capture_index}"
        with timing.measure(
            f"coarse/capture_{capture_index}", capture_index=capture_index,
        ):
            observations, _, _ = loc._capture_coarse_burst(
                pipeline, align, chain, model, args.confidence,
                hole["initial_detection"], cfg, run_dir, capture_name,
                initial_anchor_px=expected_anchor_px,
                tracking_tolerance_px=cfg.multi_coarse_tracking_tolerance_px,
                lock_anchor=True,
            )
        rows.extend(loc._observation_rows(observations))
        loc._record_hole_tracking_event(
            hole, capture_name, expected_anchor_px, observations,
            intrinsics, "distorted_pixel_yolo_center",
        )
        T_base_camera = loc.camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
        with timing.measure(
            f"coarse/geometry_{capture_index}", capture_index=capture_index,
        ):
            summary = loc._apply_coarse_geometry_to_hole(
                hole, observations, T_base_camera, cfg,
            )
        principal = np.array([float(intrinsics.cx), float(intrinsics.cy)])
        final_center_offset = float(
            np.linalg.norm(np.asarray(summary["center_px"]) - principal)
        )
        final_normal_error = loc._angle_deg(
            T_base_camera[:3, 2],
            -np.asarray(hole["coarse_normal_toward_camera_base"], dtype=np.float64),
        )
        captures.append({
            "capture_index": capture_index,
            "tcp_pose_m_rad": loc.transform_to_sdk_pose_m_rad(current_tcp),
            "center_offset_px": final_center_offset,
            "normal_error_deg": final_normal_error,
            "summary": summary,
        })
        if (
            final_center_offset <= cfg.center_tolerance_px
            and final_normal_error <= cfg.normal_tolerance_deg
        ):
            break
        if capture_index >= 3:
            break
        correction_target, correction_geometry = loc._plan_hole_tcp_pose_fixed_rz(
            np.asarray(hole["coarse_center_base_mm"], dtype=np.float64),
            np.asarray(hole["coarse_normal_toward_camera_base"], dtype=np.float64),
            current_tcp, handeye.T_tcp_rgb_camera,
            fixed_rz_rad=fixed_rz_rad, camera_height_mm=cfg.coarse_height_mm,
        )
        with timing.measure(
            f"coarse/correction_{capture_index}", capture_index=capture_index,
        ):
            current_tcp = loc._confirm_and_move_line(
                f"偏移测试：粗定位闭环校正 {capture_index}/2",
                current_tcp, correction_target, args, motion_session, pose_session,
                f"center offset={final_center_offset:.2f}px, normal error={final_normal_error:.3f}deg; "
                f"camera_axis_error={float(correction_geometry['camera_axis_error_deg']):.5f}deg",
                require_confirmation=False, motion_profile="approach",
            )
    if final_center_offset > cfg.center_tolerance_px or final_normal_error > cfg.normal_tolerance_deg:
        raise RuntimeError(
            "偏移测试粗定位闭环后未通过质量门："
            f"offset={final_center_offset:.2f}px, normal={final_normal_error:.3f}deg"
        )
    hole["coarse_captures"] = captures
    return current_tcp, captures


def _move_to_fine_height(
    *, args: Any, cfg: Any, handeye: Any, hole: dict[str, Any],
    current_tcp: np.ndarray, motion_session: Any, pose_session: Any,
    timing: loc.TimingRecorder,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    coarse_plane_base = np.asarray(
        hole.get("coarse_plane_point_base_mm")
        if hole.get("coarse_plane_point_base_mm") is not None
        else hole["coarse_center_base_mm"],
        dtype=np.float64,
    ).reshape(3)
    coarse_normal_base = loc._unit(
        np.asarray(hole["coarse_normal_toward_camera_base"], dtype=np.float64),
        "offset test coarse normal",
    )
    estimated_height = math.inf
    for height_index in range(cfg.max_z_corrections):
        _, actual_tcp = loc._require_safe_snapshot(pose_session)
        estimated_height = loc.camera_height_to_plane_mm(
            actual_tcp, handeye.T_tcp_rgb_camera, coarse_plane_base,
        )
        if abs(estimated_height - cfg.fine_height_mm) <= cfg.height_tolerance_mm:
            current_tcp = actual_tcp
            break
        z_target, _ = loc.base_z_target_for_camera_height(
            actual_tcp, handeye.T_tcp_rgb_camera, coarse_plane_base,
            cfg.fine_height_mm,
        )
        with timing.measure(
            f"fine_height/correction_{height_index + 1}",
            correction_index=height_index + 1,
        ):
            current_tcp = loc._confirm_and_move_line(
                f"偏移测试：下降到精定位高度{cfg.fine_height_mm:.0f} mm",
                actual_tcp, z_target, args, motion_session, pose_session,
                f"当前孔估计高度={estimated_height:.2f} mm；XY和姿态保持不变",
                require_confirmation=False, motion_profile="approach",
            )
    _, current_tcp = loc._require_safe_snapshot(pose_session)
    estimated_height = loc.camera_height_to_plane_mm(
        current_tcp, handeye.T_tcp_rgb_camera, coarse_plane_base,
    )
    if abs(estimated_height - cfg.fine_height_mm) > cfg.height_tolerance_mm:
        raise RuntimeError(
            f"偏移测试下降后未达到精定位高度：{estimated_height:.2f} mm"
        )
    return current_tcp, coarse_plane_base, coarse_normal_base, estimated_height


def _compute_fine_base_point(
    fine: dict[str, Any], fine_intrinsics: Any, current_tcp: np.ndarray,
    handeye: Any, coarse_plane_base: np.ndarray, coarse_normal_base: np.ndarray,
    estimated_height_mm: float, cfg: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
    naive = loc.pixel_to_base_plane(
        fine["center_px"], fine_intrinsics, current_tcp,
        handeye.T_tcp_rgb_camera, coarse_plane_base, coarse_normal_base,
        center_is_undistorted=True,
    )
    diameter_px = float(max(
        float(fine["axes_px_median"][0]), float(fine["axes_px_median"][1]),
    ))
    diameter_estimate = diameter_px * estimated_height_mm / (
        (float(fine_intrinsics.fx) + float(fine_intrinsics.fy)) / 2.0
    )
    nearest_diameter = min(
        loc.HOLE_DIAMETERS_MM,
        key=lambda value: abs(value - diameter_estimate),
    )
    corrected = naive
    tilt_correction = None
    if cfg.enable_tilt_center_correction:
        corrected, tilt_correction = loc.correct_projected_circle_center(
            fine["center_px"], fine_intrinsics, current_tcp,
            handeye.T_tcp_rgb_camera, coarse_plane_base, coarse_normal_base,
            nearest_diameter, iterations=cfg.tilt_correction_iterations,
            samples=cfg.tilt_correction_samples,
            max_correction_mm=cfg.max_tilt_correction_mm,
        )
    return naive, corrected, tilt_correction


def _actual_camera_offset_mm(
    reference_tcp: np.ndarray, actual_tcp: np.ndarray, T_tcp_camera: np.ndarray,
) -> np.ndarray:
    reference_camera = loc.camera_transform(reference_tcp, T_tcp_camera)
    actual_camera = loc.camera_transform(actual_tcp, T_tcp_camera)
    delta_base = actual_camera[:3, 3] - reference_camera[:3, 3]
    # 与 plan_camera_plane_offset_tcp 的符号定义一致：返回孔洞的实际相对偏移。
    return -(reference_camera[:3, :2].T @ delta_base)


def _motion_pose_error(planned: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    """记录机器人到位位姿相对规划位姿的执行误差。"""
    planned = np.asarray(planned, dtype=np.float64).reshape(4, 4)
    actual = np.asarray(actual, dtype=np.float64).reshape(4, 4)
    delta = actual[:3, 3] - planned[:3, 3]
    return {
        "translation_error_mm": delta.tolist(),
        "translation_error_norm_mm": float(np.linalg.norm(delta)),
        "rotation_error_deg": float(
            loc._rotation_distance_deg(planned[:3, :3], actual[:3, :3])
        ),
    }


def _execute_final_motion(
    *, args: Any, current_tcp: np.ndarray, reference_tcp: np.ndarray,
    fine_point_base: np.ndarray, timing: loc.TimingRecorder,
    sample_id: int, motion_session: Any, pose_session: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """按正式流程执行最终XY、降Z和基坐标Y+0.2，并停在最终目标点。"""
    fixed_offset = (
        None if args.tcp_xy_offset_mm is None
        else (float(args.tcp_xy_offset_mm[0]), float(args.tcp_xy_offset_mm[1]))
    )
    motion: dict[str, Any] = {
        "motion_sequence": ["final_xy", "final_z", "final_y_plus_0_2", "retract", "return"],
        "hole_center_base_mm": np.asarray(fine_point_base, dtype=np.float64),
        "compensation_mode": (
            "charuco_affine_model" if fixed_offset is None else "fixed_offset_override"
        ),
        "tcp_xy_offset_mm": None if fixed_offset is None else list(fixed_offset),
    }

    xy_target, tcp_before_xy = loc.plan_final_tcp_xy(
        current_tcp, fine_point_base, fixed_offset,
    )
    with timing.measure(
        f"sample_{sample_id:02d}/final_motion_xy", sample_id=sample_id,
    ):
        current_tcp = loc._confirm_and_move_line(
            f"偏移测试点 {sample_id}：最终XY",
            current_tcp, xy_target, args, motion_session, pose_session,
            "按正式流程执行最终XY；随后继续最终Z和Y+0.2 mm",
            require_confirmation=False, motion_profile="precision",
        )
    motion["final_xy"] = {
        "planned_tcp_pose_m_rad": loc.transform_to_sdk_pose_m_rad(xy_target),
        "actual_tcp_pose_m_rad": loc.transform_to_sdk_pose_m_rad(current_tcp),
        "tcp_position_before_mm": tcp_before_xy,
        "xy_correction_mm": xy_target[:2, 3] - np.asarray(fine_point_base)[:2],
        "pose_error": _motion_pose_error(xy_target, current_tcp),
    }

    z_target = loc.plan_final_tcp_base_z(current_tcp, fine_point_base)
    with timing.measure(
        f"sample_{sample_id:02d}/final_motion_z", sample_id=sample_id,
    ):
        current_tcp = loc._confirm_and_move_line(
            f"偏移测试点 {sample_id}：最终降Z",
            current_tcp, z_target, args, motion_session, pose_session,
            f"目标TCP基坐标Z={float(z_target[2, 3]):.3f} mm",
            require_confirmation=False, motion_profile="precision",
        )
    motion["final_z"] = {
        "planned_tcp_pose_m_rad": loc.transform_to_sdk_pose_m_rad(z_target),
        "actual_tcp_pose_m_rad": loc.transform_to_sdk_pose_m_rad(current_tcp),
        "target_base_z_mm": float(z_target[2, 3]),
        "pose_error": _motion_pose_error(z_target, current_tcp),
    }

    y_target = loc.plan_final_tcp_base_y_trim(current_tcp)
    with timing.measure(
        f"sample_{sample_id:02d}/final_motion_y_plus_0_2", sample_id=sample_id,
    ):
        current_tcp = loc._confirm_and_move_line(
            f"偏移测试点 {sample_id}：最终基坐标+Y 0.2 mm",
            current_tcp, y_target, args, motion_session, pose_session,
            "保持X、Z和姿态；基坐标Y增加0.2 mm",
            require_confirmation=False, motion_profile="precision",
        )
    motion["final_y_plus_0_2"] = {
        "planned_tcp_pose_m_rad": loc.transform_to_sdk_pose_m_rad(y_target),
        "actual_tcp_pose_m_rad": loc.transform_to_sdk_pose_m_rad(current_tcp),
        "delta_base_y_mm": loc.FINAL_BASE_Y_AFTER_Z_MM,
        "pose_error": _motion_pose_error(y_target, current_tcp),
    }
    motion["final_pose_before_retract_m_rad"] = loc.transform_to_sdk_pose_m_rad(current_tcp)
    return current_tcp, motion


def _recover_after_final_target(
    *, args: Any, current_tcp: np.ndarray, reference_tcp: np.ndarray,
    motion: dict[str, Any], timing: loc.TimingRecorder, sample_id: int,
    motion_session: Any, pose_session: Any,
) -> np.ndarray:
    """最终目标点确认后，先回升再返回精定位中心。"""
    # 目标点暂停期间保持当前位姿；确认或取消后都先沿基坐标Z回升，
    # 避免从插入深度直接斜向下一个测试点运动。
    retract_target = np.asarray(current_tcp, dtype=np.float64).copy()
    retract_target[2, 3] = float(reference_tcp[2, 3])
    if abs(float(retract_target[2, 3] - current_tcp[2, 3])) > 0.2:
        with timing.measure(
            f"sample_{sample_id:02d}/final_motion_retract", sample_id=sample_id,
        ):
            current_tcp = loc._confirm_and_move_line(
                f"偏移测试点 {sample_id}：最终动作后回升",
                current_tcp, retract_target, args, motion_session, pose_session,
                "先回到精定位高度，再返回下一个测试点中心",
                require_confirmation=False, motion_profile="approach",
            )
    motion["retract"] = {
        "planned_tcp_pose_m_rad": loc.transform_to_sdk_pose_m_rad(retract_target),
        "actual_tcp_pose_m_rad": loc.transform_to_sdk_pose_m_rad(current_tcp),
        "pose_error": _motion_pose_error(retract_target, current_tcp),
    }

    with timing.measure(
        f"sample_{sample_id:02d}/final_motion_return", sample_id=sample_id,
    ):
        current_tcp = loc._confirm_and_move_line(
            f"偏移测试点 {sample_id}：返回精定位中心",
            current_tcp, reference_tcp, args, motion_session, pose_session,
            "最终动作测试完成，返回中心后再进行下一偏移点",
            require_confirmation=False, motion_profile="precision",
        )
    motion["return"] = {
        "planned_tcp_pose_m_rad": loc.transform_to_sdk_pose_m_rad(reference_tcp),
        "actual_tcp_pose_m_rad": loc.transform_to_sdk_pose_m_rad(current_tcp),
        "pose_error": _motion_pose_error(reference_tcp, current_tcp),
    }
    return current_tcp


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row}) if rows else ["sample_id", "status"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_polar_plot(
    path: Path, results: list[dict[str, Any]], max_radius_mm: float,
) -> None:
    canvas = np.full((PLOT_SIZE, PLOT_SIZE, 3), 255, dtype=np.uint8)
    center = np.array([PLOT_SIZE // 2, PLOT_SIZE // 2], dtype=int)
    plot_radius = 330
    max_radius = max(1.0, float(max_radius_mm))
    scale = plot_radius / max_radius
    cv2.line(canvas, (center[0] - plot_radius, center[1]),
             (center[0] + plot_radius, center[1]), (180, 180, 180), 1)
    cv2.line(canvas, (center[0], center[1] - plot_radius),
             (center[0], center[1] + plot_radius), (180, 180, 180), 1)
    for radius in sorted({float(item["radius_mm"]) for item in results}):
        r_px = max(1, int(round(radius * scale)))
        cv2.circle(canvas, tuple(center), r_px, (210, 210, 210), 1)
        cv2.putText(
            canvas, f"{radius:g} mm", (center[0] + r_px + 4, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA,
        )
    for item in results:
        offset = np.asarray(item.get("offset_camera_xy_mm", [0.0, 0.0]), dtype=float)
        point = (
            int(round(center[0] + offset[0] * scale)),
            int(round(center[1] - offset[1] * scale)),
        )
        if item.get("manual_error_marked"):
            color = (180, 0, 180)
        elif item.get("accepted_pass"):
            color = (40, 170, 40)
        elif item.get("strict_pass"):
            color = (0, 150, 255)
        elif item.get("status") == "not_run":
            color = (140, 140, 140)
        else:
            color = (40, 40, 220)
        cv2.circle(canvas, point, 8, color, -1)
        cv2.circle(canvas, point, 10, (40, 40, 40), 1)
    cv2.putText(canvas, "Coarse-to-fine offset tolerance", (24, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "green=accepted  purple=manual-error  orange=strict-only  red=failed", (24, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, "+X camera", (center[0] + plot_radius - 75, center[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, "+Y camera", (center[0] + 8, center[1] - plot_radius + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def _summarize_radii(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_radius: dict[float, list[dict[str, Any]]] = {}
    for item in results:
        by_radius.setdefault(float(item["radius_mm"]), []).append(item)
    rows: list[dict[str, Any]] = []
    accepted_radii: list[float] = []
    strict_radii: list[float] = []
    for radius in sorted(by_radius):
        items = by_radius[radius]
        accepted = bool(items) and all(
            bool(item.get("accepted_pass")) and not bool(item.get("manual_error_marked"))
            for item in items
        )
        strict = bool(items) and all(
            bool(item.get("strict_pass")) and not bool(item.get("manual_error_marked"))
            for item in items
        )
        if accepted:
            accepted_radii.append(radius)
        if strict:
            strict_radii.append(radius)
        rows.append({
            "radius_mm": radius,
            "sample_count": len(items),
            "accepted_count": sum(bool(item.get("accepted_pass")) for item in items),
            "strict_count": sum(bool(item.get("strict_pass")) for item in items),
            "manual_error_count": sum(bool(item.get("manual_error_marked")) for item in items),
            "accepted_all_directions": accepted,
            "strict_all_directions": strict,
        })
    return {
        "by_radius": rows,
        "max_supported_radius_mm": max(accepted_radii) if accepted_radii else None,
        "max_strict_radius_mm": max(strict_radii) if strict_radii else None,
        "accepted_definition": "all sampled directions pass current fine recovery quality gate and have no manual error mark",
        "strict_definition": "all sampled directions finish with strict fine quality status and have no manual error mark",
        "manual_error_count": sum(bool(item.get("manual_error_marked")) for item in results),
        "manual_error_sample_ids": [
            int(item["sample_id"]) for item in results
            if bool(item.get("manual_error_marked"))
        ],
    }


def _summarize_final_motion(results: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总最终XY、Z和Y微调相对规划位姿的实际到位误差。"""
    step_names = {
        "final_xy": "final_xy",
        "final_z": "final_z",
        "final_y_plus_0_2": "final_y_plus_0_2",
    }
    summary: dict[str, Any] = {
        "enabled": False,
        "sample_count": 0,
        "steps": {},
    }
    for result in results:
        final_motion = result.get("final_motion")
        if not isinstance(final_motion, dict):
            continue
        summary["enabled"] = True
        summary["sample_count"] += 1
        for output_name, step_name in step_names.items():
            pose_error = (
                final_motion.get(step_name, {}).get("pose_error")
                if isinstance(final_motion.get(step_name), dict) else None
            )
            if not isinstance(pose_error, dict):
                continue
            step = summary["steps"].setdefault(output_name, {"translation_errors_mm": [], "rotation_errors_deg": []})
            step["translation_errors_mm"].append(float(pose_error.get("translation_error_norm_mm", math.inf)))
            step["rotation_errors_deg"].append(float(pose_error.get("rotation_error_deg", math.inf)))
    for step in summary["steps"].values():
        translation_errors = step.pop("translation_errors_mm")
        rotation_errors = step.pop("rotation_errors_deg")
        step.update({
            "max_translation_error_mm": max(translation_errors) if translation_errors else None,
            "mean_translation_error_mm": float(np.mean(translation_errors)) if translation_errors else None,
            "max_rotation_error_deg": max(rotation_errors) if rotation_errors else None,
            "mean_rotation_error_deg": float(np.mean(rotation_errors)) if rotation_errors else None,
        })
    return summary


def _make_cfg(args: Any) -> Any:
    return loc.TwoStageConfig(
        coarse_height_mm=float(args.coarse_height_mm),
        fine_height_mm=float(args.fine_height_mm),
        coarse_frames=int(args.coarse_frames),
        fine_frames=int(args.fine_frames),
        fine_settle_discard_frames=int(args.fine_settle_discard_frames),
        fine_retry_count=int(args.fine_retries),
    )


def _plan_only(args: Any, run_dir: Path) -> int:
    plan = build_offset_plan(args.radii_mm, args.angles_deg)
    report = {
        "status": "plan_only",
        "mode": "coarse_to_fine_offset_tolerance_test",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "configuration": {
            "coarse_height_mm": float(args.coarse_height_mm),
            "fine_height_mm": float(args.fine_height_mm),
            "radii_mm": [float(value) for value in args.radii_mm],
            "angles_deg": [float(value) for value in args.angles_deg],
            "fine_frames": int(args.fine_frames),
            "fine_settle_discard_frames": int(args.fine_settle_discard_frames),
            "fine_retries": int(args.fine_retries),
            "include_final_motion": bool(args.include_final_motion),
        },
        "offset_coordinate_frame": "camera_transverse_xy_mm; positive value means hole offset in image",
        "sample_count": len(plan),
        "plan": plan,
        "final_motion_summary": {
            "enabled": False,
            "sample_count": 0,
            "steps": {},
        },
    }
    loc._write_report(run_dir, report, [])
    _write_polar_plot(run_dir / "offset_polar_plan.png", [
        {**item, "status": "not_run", "accepted_pass": False, "strict_pass": False}
        for item in plan
    ], max(max(float(value) for value in args.radii_mm), 1.0))
    print(f"[PLAN] 已写入偏移测试计划：{run_dir}")
    return 0


def run_offset_test(args: Any) -> int:
    plan = build_offset_plan(args.radii_mm, args.angles_deg)
    run_dir = loc.RUNS_DIR / f"coarse-to-fine-offset-{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=False)
    if not args.execute:
        if args.include_final_motion:
            raise ValueError("最终XY/Z/Y动作必须在 --execute 实机模式下运行")
        return _plan_only(args, run_dir)

    cfg = _make_cfg(args)
    if cfg.fine_height_mm >= cfg.coarse_height_mm:
        raise ValueError("精定位高度必须小于粗定位高度")
    handeye = load_handeye_experiment_result(args.handeye)
    if not args.allow_experimental_handeye and not handeye.validated_for_motion:
        # 当前测试会移动机器人；即使是诊断，也不让调用者无意间使用未验证手眼。
        raise RuntimeError(
            "偏移测试需要显式添加 --allow-experimental-handeye，"
            "或使用已通过生产运动验证的手眼结果"
        )
    model = loc.load_yolo(args.model)
    loc._apply_robot_connection_overrides(args)
    if not bool(getattr(args, "start_confirmed", False)):
        print(
            "[OFFSET_TEST_CONFIRM_REQUIRED] 偏移测试将只执行粗定位、下降和横向偏移采集，"
            "不执行最终插入动作；输入 m 开始：",
            flush=True,
        )
        if input().strip().lower() != "m":
            raise RuntimeError("用户取消偏移测试")

    report: dict[str, Any] = {
        "status": "running",
        "mode": "coarse_to_fine_offset_tolerance_test",
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "handeye_path": str(args.handeye),
        "experimental_handeye_override": bool(args.allow_experimental_handeye),
        "motion_executed": True,
        "final_motion_executed": bool(args.include_final_motion),
        "offset_coordinate_frame": "camera_transverse_xy_mm; positive value means hole offset in image",
        "configuration": {
            "coarse_height_mm": float(cfg.coarse_height_mm),
            "fine_height_mm": float(cfg.fine_height_mm),
            "radii_mm": [float(value) for value in args.radii_mm],
            "angles_deg": [float(value) for value in args.angles_deg],
            "coarse_frames": int(cfg.coarse_frames),
            "fine_frames": int(cfg.fine_frames),
            "fine_settle_discard_frames": int(cfg.fine_settle_discard_frames),
            "fine_retries": int(cfg.fine_retry_count),
            "include_final_motion": bool(args.include_final_motion),
        },
        "plan": plan,
        "results": [],
        "final_motion_summary": {
            "enabled": bool(args.include_final_motion),
            "sample_count": 0,
            "steps": {},
        },
    }
    timing = loc.TimingRecorder()
    timing.attach_report(report)
    rows: list[dict[str, Any]] = []
    pipeline = align = chain = None
    pose_session = motion_session = None
    current_tcp: np.ndarray | None = None
    try:
        from aubo_workbench.motion_control import AuboMotionSession, load_home_point
        from aubo_workbench.robot import AuboPoseSession

        home = load_home_point()
        if home is None:
            raise RuntimeError("未找到 aubo_home_point.json，无法安全开始偏移测试")
        pose_session = AuboPoseSession()
        pose_session.connect()
        _, current_tcp = loc._require_safe_snapshot(pose_session)
        report["robot_initial_tcp_pose_m_rad"] = loc.transform_to_sdk_pose_m_rad(current_tcp)
        if not handeye.validated_for_motion:
            print(
                "[EXPERIMENTAL] 当前偏移测试使用未通过生产验证的手眼结果，"
                "只用于实验诊断，不代表生产可用。"
            )
        motion_session = AuboMotionSession()
        motion_session.connect(
            ROBOT_CFG.ip, ROBOT_CFG.rpc_port, ROBOT_CFG.user,
            ROBOT_CFG.password, ROBOT_CFG.request_timeout_ms,
        )
        current_tcp = loc._confirm_and_move_home(home, args, motion_session, pose_session)

        with timing.measure("camera/start_rgbd_pipeline"):
            pipeline, align, chain = init_pipeline()
            report["camera"] = get_device_identity(pipeline)
        cfg_initial = cfg.initial_max_plane_rmse_mm
        with timing.measure("initial_selection/single_hole_rgbd"):
            _, selected_holes, _, _, intrinsics = loc._capture_initial_multi_hole_selection(
                pipeline, align, chain, model, args.confidence, run_dir, cfg_initial,
            )
        if len(selected_holes) != 1:
            raise RuntimeError(
                f"偏移测试必须只选择一个孔，当前选择了 {len(selected_holes)} 个"
            )
        hole = selected_holes[0]
        _populate_initial_base_geometry(hole, current_tcp, handeye)
        fixed_rz_rad = loc._matrix_to_rpy_zyx(current_tcp[:3, :3])[2]
        report["selected_hole"] = {
            "hole_id": 1,
            "initial_detection": hole["initial_detection"],
            "initial_center_px": hole["initial_center_px"],
            "initial_center_base_mm": hole["initial_center_base_mm"],
            "fixed_rz_rad": fixed_rz_rad,
        }

        current_tcp, coarse_captures = _run_single_hole_coarse(
            args=args, cfg=cfg, handeye=handeye, model=model, hole=hole,
            current_tcp=current_tcp, fixed_rz_rad=fixed_rz_rad,
            pipeline=pipeline, align=align, chain=chain, intrinsics=intrinsics,
            run_dir=run_dir, timing=timing, rows=rows,
            motion_session=motion_session, pose_session=pose_session,
        )
        report["coarse_result"] = {
            "coarse_captures": coarse_captures,
            "coarse_center_base_mm": hole["coarse_center_base_mm"],
            "coarse_normal_toward_camera_base": hole["coarse_normal_toward_camera_base"],
            "coarse_plane_rmse_mm": hole["coarse_plane_rmse_mm"],
            "coarse_center_scatter_p95_px": hole["coarse_center_scatter_p95_px"],
        }
        current_tcp, coarse_plane_base, coarse_normal_base, estimated_height = _move_to_fine_height(
            args=args, cfg=cfg, handeye=handeye, hole=hole,
            current_tcp=current_tcp, motion_session=motion_session,
            pose_session=pose_session, timing=timing,
        )
        reference_tcp = np.asarray(current_tcp, dtype=np.float64).copy()
        reference_camera = loc.camera_transform(reference_tcp, handeye.T_tcp_rgb_camera)
        hole_center_base = np.asarray(hole["coarse_center_base_mm"], dtype=np.float64)
        report["fine_reference"] = {
            "tcp_pose_m_rad": loc.transform_to_sdk_pose_m_rad(reference_tcp),
            "camera_pose_base": reference_camera,
            "estimated_height_mm": estimated_height,
            "hole_center_base_mm": hole_center_base,
        }

        baseline_point_base: np.ndarray | None = None
        for item in plan:
            sample_id = int(item["sample_id"])
            offset = np.asarray(item["offset_camera_xy_mm"], dtype=np.float64)
            target_tcp, planned_delta_base = plan_camera_plane_offset_tcp(
                reference_tcp, handeye.T_tcp_rgb_camera, offset,
            )
            with timing.measure(
                f"sample_{sample_id:02d}/move_to_offset", sample_id=sample_id,
            ):
                current_tcp = loc._confirm_and_move_line(
                    f"偏移测试点 {sample_id}/{len(plan)}：移动到{item['radius_mm']:.1f} mm",
                    current_tcp, target_tcp, args, motion_session, pose_session,
                    "保持精定位高度、姿态和固定RZ，不执行最终目标点运动",
                    require_confirmation=False, motion_profile="precision",
                )
            _, current_tcp = loc._require_safe_snapshot(pose_session)
            T_base_camera = loc.camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
            expected_anchor_px = loc._project_base_point_to_pixel(
                hole_center_base, T_base_camera, intrinsics,
            )
            sample_hole: dict[str, Any] = {"tracking_events": []}
            with timing.measure(
                f"sample_{sample_id:02d}/fine_recovery", sample_id=sample_id,
            ):
                recovery = loc._capture_fine_with_recovery(
                    pipeline, model, args.confidence, hole["initial_detection"], cfg,
                    run_dir, sample_hole, sample_id, sample_id, expected_anchor_px,
                    timing, rows,
                )
            fine = recovery.get("fine")
            sample_result: dict[str, Any] = {
                **item,
                "status": "failed" if not recovery["success"] else "accepted",
                "accepted_pass": bool(recovery["success"]),
                "strict_pass": bool(recovery["status"] == "strict"),
                "quality_status": recovery["status"],
                "failure_reason": recovery.get("error"),
                "planned_tcp_pose_m_rad": loc.transform_to_sdk_pose_m_rad(target_tcp),
                "actual_tcp_pose_m_rad": loc.transform_to_sdk_pose_m_rad(current_tcp),
                "planned_delta_base_mm": planned_delta_base,
                "actual_offset_camera_xy_mm": _actual_camera_offset_mm(
                    reference_tcp, current_tcp, handeye.T_tcp_rgb_camera,
                ),
                "expected_center_px_distorted": None,
                "expected_center_px_undistorted": None,
                "expected_offset_px": None,
                "observed_center_px": None,
                "observed_offset_px": None,
                "valid_frames": 0,
                "total_frames": 0,
                "center_scatter_p95_px": None,
                "center_source": None,
                "fine_center_base_mm": None,
                "fine_center_naive_base_mm": None,
                "relative_center_error_mm": None,
                "final_motion": None,
                "manual_error_marked": False,
                "manual_error_note": None,
                "fine_recovery_attempts": recovery.get("attempts", []),
                "tracking_events": sample_hole.get("tracking_events", []),
            }
            if fine is not None and recovery.get("intrinsics") is not None:
                fine_intrinsics = recovery["intrinsics"]
                _, expected_undistorted, expected_offset_px = _expected_center_offset_px(
                    hole_center_base, T_base_camera, fine_intrinsics,
                )
                naive_point, corrected_point, tilt_correction = _compute_fine_base_point(
                    fine, fine_intrinsics, current_tcp, handeye,
                    coarse_plane_base, coarse_normal_base, estimated_height, cfg,
                )
                if baseline_point_base is None:
                    baseline_point_base = corrected_point.copy()
                sample_result.update({
                    "expected_center_px_distorted": loc._project_base_point_to_pixel(
                        hole_center_base, T_base_camera, fine_intrinsics,
                    ),
                    "expected_center_px_undistorted": expected_undistorted,
                    "expected_offset_px": expected_offset_px,
                    "observed_center_px": fine["center_px"],
                    "observed_offset_px": np.asarray(fine["center_px"]) - np.array([
                        float(fine_intrinsics.cx), float(fine_intrinsics.cy),
                    ]),
                    "valid_frames": int(fine.get("valid_frames", 0)),
                    "total_frames": int(fine.get("total_frames", 0)),
                    "center_scatter_p95_px": float(fine.get("center_scatter_p95_px", math.inf)),
                    "center_source": fine.get("center_source"),
                    "fine_center_base_mm": corrected_point,
                    "fine_center_naive_base_mm": naive_point,
                    "relative_center_error_mm": float(np.linalg.norm(corrected_point - baseline_point_base)),
                    "tilt_center_correction": tilt_correction,
                })
                if args.include_final_motion:
                    current_tcp, final_motion = _execute_final_motion(
                        args=args, current_tcp=current_tcp, reference_tcp=reference_tcp,
                        fine_point_base=corrected_point, timing=timing,
                        sample_id=sample_id, motion_session=motion_session,
                        pose_session=pose_session,
                    )
                    sample_result["final_motion"] = final_motion
            report["results"].append(sample_result)
            _write_csv(run_dir / "offset_samples.csv", [loc._jsonable(item) for item in report["results"]])
            report["timing"] = timing.snapshot()
            report["summary"] = _summarize_radii(report["results"])
            report["final_motion_summary"] = _summarize_final_motion(report["results"])
            loc._write_report(run_dir, report, rows)
            print(
                f"[OFFSET_TEST] sample={sample_id}/{len(plan)} radius={item['radius_mm']:.1f} mm "
                f"angle={item['angle_deg']:.1f}° status={sample_result['status']} "
                f"scatter={sample_result['center_scatter_p95_px']}",
                flush=True,
            )
            if args.include_final_motion:
                confirmed = True
                manual_error_marked = False
                if bool(args.confirm_each_offset):
                    print(
                        f"[OFFSET_TARGET_CONFIRM_REQUIRED] 偏移点 {sample_id}/{len(plan)} 已到达最终目标点；"
                        "机械臂保持当前位置；输入 m=确认无误继续，输入 e=标记当前点有误差并继续：",
                        flush=True,
                    )
                    decision = input().strip().lower()
                    confirmed = decision in {"m", "e"}
                    manual_error_marked = decision == "e"
                current_tcp = _recover_after_final_target(
                    args=args, current_tcp=current_tcp, reference_tcp=reference_tcp,
                    motion=sample_result["final_motion"], timing=timing,
                    sample_id=sample_id, motion_session=motion_session,
                    pose_session=pose_session,
                )
                sample_result["final_target_confirmation"] = {
                    "confirmed": bool(confirmed),
                    "manual_error_marked": bool(manual_error_marked),
                    "held_at_target_before_retract": True,
                }
                sample_result["manual_error_marked"] = bool(manual_error_marked)
                sample_result["manual_error_note"] = (
                    "人工确认该目标点存在误差" if manual_error_marked else None
                )
                report["timing"] = timing.snapshot()
                report["summary"] = _summarize_radii(report["results"])
                report["final_motion_summary"] = _summarize_final_motion(report["results"])
                loc._write_report(run_dir, report, rows)
                if not confirmed:
                    raise RuntimeError(
                        f"用户在偏移点 {sample_id} 最终目标点处停止偏移测试"
                    )
            if (
                not args.include_final_motion
                and not math.isclose(float(item["radius_mm"]), 0.0, abs_tol=1e-9)
                and (
                    float(np.linalg.norm(current_tcp[:3, 3] - reference_tcp[:3, 3])) > 0.2
                    or loc._rotation_distance_deg(
                        current_tcp[:3, :3], reference_tcp[:3, :3],
                    ) > 0.01
                )
            ):
                with timing.measure(
                    f"sample_{sample_id:02d}/return_to_reference", sample_id=sample_id,
                ):
                    current_tcp = loc._confirm_and_move_line(
                        f"偏移测试点 {sample_id}/{len(plan)}：返回中心",
                        current_tcp, reference_tcp, args, motion_session, pose_session,
                        "返回精定位中心位，不执行最终目标点运动",
                        require_confirmation=False, motion_profile="precision",
                    )
            if (
                bool(args.confirm_each_offset)
                and sample_id < len(plan)
            ):
                print(
                    f"[OFFSET_NEXT_CONFIRM_REQUIRED] 偏移点 {sample_id}/{len(plan)} 已完成，"
                    f"已返回中心；确认后移动到偏移点 {sample_id + 1}/{len(plan)}，输入 m 继续：",
                    flush=True,
                )
                if input().strip().lower() != "m":
                    raise RuntimeError(
                        f"用户在偏移点 {sample_id} 完成后停止偏移测试"
                    )
        current_translation_delta = float(
            np.linalg.norm(current_tcp[:3, 3] - reference_tcp[:3, 3])
        )
        current_rotation_delta = float(
            loc._rotation_distance_deg(current_tcp[:3, :3], reference_tcp[:3, :3])
        )
        if current_translation_delta > 0.2 or current_rotation_delta > 0.01:
            current_tcp = loc._confirm_and_move_line(
                "偏移测试结束：返回精定位中心",
                current_tcp, reference_tcp, args, motion_session, pose_session,
                "测试结束，不执行最终 XY、最终 Z 和 Y+0.2 mm",
                require_confirmation=False, motion_profile="precision",
            )
        report["status"] = "completed"
        report["summary"] = _summarize_radii(report["results"])
        report["final_motion_summary"] = _summarize_final_motion(report["results"])
        report["timing"] = timing.snapshot()
        _write_csv(run_dir / "offset_samples.csv", [loc._jsonable(item) for item in report["results"]])
        _write_polar_plot(
            run_dir / "offset_polar_result.png", report["results"],
            max(float(value) for value in args.radii_mm),
        )
        loc._write_report(run_dir, report, rows)
        print(json.dumps(loc._jsonable(report["summary"]), ensure_ascii=False, indent=2))
        print(f"[DONE] 偏移测试结果目录：{run_dir}")
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report["timing"] = timing.snapshot()
        try:
            loc._write_report(run_dir, report, rows)
        except Exception:
            pass
        raise
    finally:
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


def build_parser() -> argparse.ArgumentParser:
    parser = loc.build_parser()
    parser.description = "粗定位后精定位视野偏移容忍度测试"
    parser.set_defaults(
        execute=False,
        allow_experimental_handeye=False,
        coarse_frames=15,
        fine_frames=30,
        fine_settle_discard_frames=10,
        fine_retries=2,
    )
    parser.add_argument(
        "--radii-mm", nargs="+", type=float, default=list(DEFAULT_RADII_MM),
        help="测试半径，默认0 5 10 15 20 mm",
    )
    parser.add_argument(
        "--angles-deg", nargs="+", type=float, default=list(DEFAULT_ANGLES_DEG),
        help="非零半径测试方向，默认每45度一个方向",
    )
    parser.add_argument(
        "--start-confirmed", action="store_true",
        help="由GUI等外部控制器完成开始确认，不再等待命令行输入",
    )
    parser.add_argument(
        "--include-final-motion", action="store_true",
        help="实机测试中按正式流程执行最终XY、降Z和基坐标Y+0.2 mm，并安全回到中心",
    )
    parser.add_argument(
        "--auto-continue-offset", dest="confirm_each_offset", action="store_false",
        help="偏移测试点之间不等待人工确认，自动继续（不建议实机使用）",
    )
    parser.set_defaults(confirm_each_offset=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    args = build_parser().parse_args(argv)
    loc._apply_robot_connection_overrides(args)
    return run_offset_test(args)


if __name__ == "__main__":
    raise SystemExit(main())
