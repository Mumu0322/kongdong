#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO + RGB-D + RGB 手眼标定的“眼在手上”孔中心定位脚本。

流程：在初始 RGB-D 画面中选择目标孔 -> 340 mm 粗定位建立局部点云、平面和法向
-> 260 mm RGB/YOLO 精定位修正最终 XY -> 移动到当前目标点。建图模式还支持
逐孔保存340 mm粗定位和260 mm精定位参考，但地图调用仍要求现场重新精定位。

默认进入两阶段流程但只做预览；必须显式启用运动和相应的验证开关后，才会连接运动控制。
默认不允许使用未通过生产验证的实验手眼结果。

本文件保留命令行入口和兼容导出；视觉、采集、共享定位、分组运动、缓存、
地图辅助、诊断和参数解析分别由 aubo_workbench 下的专用模块实现。
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from functools import wraps
import json
import math
import shutil
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

# Windows 传统 GBK 控制台无法编码帮助文本中的部分符号（例如 m²），
# 入口脚本应能在现场直接执行 --help 或输出诊断信息。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

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
from aubo_workbench.camera_stream_health import (
    camera_stream_session,
    monitored_camera_read,
    require_camera_stream_healthy,
)

get_aligned_frame_bundle = monitored_camera_read(get_aligned_frame_bundle)
get_rgb_frame_bundle = monitored_camera_read(get_rgb_frame_bundle)
from aubo_workbench.localization_errors import (
    LocalizationHardwareError, MotionExecutionError, motion_failure_boundary,
)
from aubo_workbench.auto_sector_selection import (  # noqa: E402
    load_auto_sector_config,
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
from aubo_workbench.io_utils import atomic_write_json, jsonable, write_dict_rows  # noqa: E402
from aubo_workbench.hole_map import (  # noqa: E402
    build_hole_map_payload,
    build_rotary_sector_map_payload,
    build_rotary_sector_hole_repair_payload,
    file_sha256,
    get_hole,
    load_hole_map,
    publish_current_hole_map,
    resolve_hole_map_path,
    save_hole_map,
    validate_hole_map,
)
from aubo_workbench.hole_map_seed_correction import (  # noqa: E402
    DEFAULT_SEED_CORRECTION_FILENAME,
    load_seed_correction_model,
    validate_model_binding,
)
from aubo_workbench.rotary_sector_map import (  # noqa: E402
    sector_key,
    validate_sector_id,
)
from aubo_workbench.hole_map_visualization import (  # noqa: E402
    export_hole_map_artifacts,
)
from aubo_workbench.batch_grouping import (  # noqa: E402
    classify_selected_hole_boundary_layers,
    group_holes_spatially_with_metadata,
)
from tools.visualize_coarse_cache import CacheCloud, render_cache_cloud  # noqa: E402
from aubo_workbench.fitting import fit_sphere  # noqa: E402
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
    HANDEYE_DIAGNOSTIC_PATH,
    HOLE_LOCALIZATION_COARSE_CACHE_DIR,
    HOLE_LOCALIZATION_MAPS_DIR,
    MODEL_PATH,
    HOLE_LOCALIZATION_RUNS_DIR,
    HOLE_LOCALIZATION_SECTOR_INFO_DIR,
)


DEFAULT_MODEL = MODEL_PATH
DEFAULT_HANDEYE = HANDEYE_DIAGNOSTIC_PATH
RUNS_DIR = HOLE_LOCALIZATION_RUNS_DIR
from aubo_workbench.hole_localization_planning import (  # noqa: E402
    CHARUCO_XY_MODEL_BIAS_MM,
    CHARUCO_XY_MODEL_MATRIX,
    CHARUCO_XY_MODEL_READY,
    CHARUCO_XY_MODEL_SOURCE,
    FINAL_BASE_Y_AFTER_Z_MM,
    FINAL_TOOL_Y_AFTER_Z_MM,
    THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM,
    _compose_batch_fine_joint_xy_with_tilt,
    compose_batch_fine_xy_with_coarse_z,
    fuse_batch_fine_xy_with_pointcloud_prior,
    plan_final_tcp_base_y_trim,
    plan_final_tcp_base_z,
    # 保留旧脚本的兼容导出；地图调用链已不再运行时调用该微调规划。
    plan_final_tcp_combined_y_trim,
    plan_final_tcp_xy,
)
SHARED_OBSERVATION_MIN_LIFT_MM = 10.0
SHARED_OBSERVATION_MIN_DESCENT_MM = 10.0
# AUBO SDK 的正数是警告码；13 是 AUBO_REQUEST_IGNORE（请求被忽略）。
# 该码不能在全局 sdk_ok 中当作成功，但目标已经到位时，重复 moveLine
# 属于安全的无动作请求，可以通过实际 TCP 位姿确认后继续。
AUBO_REQUEST_IGNORE_CODE = 13
MOTION_REQUEST_IGNORE_POSITION_TOLERANCE_MM = 0.5
MOTION_REQUEST_IGNORE_ROTATION_TOLERANCE_DEG = 0.1
# 机器人到位检测只影响轮询响应，不改变控制器的运动轨迹。
ROBOT_STEADY_POLL_INTERVAL_S = 0.10
# 回到原点后，开始下一轮初始拍摄前再留出一小段静止缓冲，避免相机抓到末端
# 刚停止时的残余振动帧或控制器状态切换瞬间。
ROBOT_INITIAL_CAPTURE_SETTLE_DELAY_S = 0.50
from aubo_workbench.hole_localization_models import (  # noqa: E402
    Observation,
    PlaneEstimate,
    TimingRecorder,
    TwoStageConfig,
    TwoStageSelectionCancelled,
    artifact_measure,
)
# 直接运行的默认模式：不需要额外命令行参数即可做离线/现场预览。
# 只有明确传入 --execute 时才允许连接运动控制和下发机器人命令。
DEFAULT_TWO_STAGE_HOLE_LOCALIZATION = True
DEFAULT_EXECUTE_MOTION = False
DEFAULT_ALLOW_EXPERIMENTAL_HANDEYE = True
DEFAULT_MOVE_FINAL_XY = True


from aubo_workbench.hole_localization_vision import (  # noqa: E402
    COARSE_SURFACE_MODEL,
    COARSE_SURFACE_SELECTION_POLICY,
    HOLE_DIAMETERS_MM,
    _load_intrinsics,
    _robust_refine_circle_from_edges,
    choose_box,
    choose_boxes,
    detect,
    fit_hole_ellipse,
    hole_camera_point,
    load_yolo,
)
# 纯几何/光学层已抽到 aubo_workbench.geometry 与 aubo_workbench.optics。
# 下划线别名保留给本文件内的既有调用点与 tests 的属性式访问。
_unit = unit_vector
_angle_deg = angle_between_deg
_matrix_to_rpy_zyx = matrix_to_rpy_zyx
_camera_matrix = camera_matrix
_distortion = distortion_coeffs
_plane_basis = plane_basis
_project_undistorted_pixels = project_undistorted_pixels


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
        raise MotionExecutionError("机器人状态不满足安全门：需要上电、稳定且无碰撞")
    return snapshot, pose_session.pose_sdk_to_transform_mm(snapshot["pose_values_sdk_m_rad"])


@motion_failure_boundary
def _wait_robot_steady(pose_session: Any, timeout_s: float = 45.0) -> tuple[dict[str, Any], np.ndarray]:
    deadline = time.monotonic() + timeout_s
    last_reason = ""
    while time.monotonic() < deadline:
        snapshot = pose_session.read_pose_snapshot()
        if not snapshot["power_on"]:
            raise MotionExecutionError("机器人未上电，停止本轮定位")
        if snapshot["collision"]:
            raise RuntimeError("运动后检测到碰撞标志，立即停止流程")
        if snapshot["power_on"] and snapshot["steady"]:
            return snapshot, pose_session.pose_sdk_to_transform_mm(snapshot["pose_values_sdk_m_rad"])
        last_reason = f"power={snapshot['power_on']} steady={snapshot['steady']}"
        time.sleep(ROBOT_STEADY_POLL_INTERVAL_S)
    raise RuntimeError(f"等待机器人稳定超时：{last_reason}")


@motion_failure_boundary
def _wait_robot_reached(
    pose_session: Any,
    target: np.ndarray,
    timeout_s: float = 45.0,
    *,
    position_tolerance_mm: float = MOTION_REQUEST_IGNORE_POSITION_TOLERANCE_MM,
    rotation_tolerance_deg: float = MOTION_REQUEST_IGNORE_ROTATION_TOLERANCE_DEG,
) -> tuple[dict[str, Any], np.ndarray]:
    """等待机器人真正到达目标，而不是仅凭可能滞后的 steady 标志返回。

    AUBO 的 steady 状态在 moveLine 刚下发时可能仍是上一次动作的旧值。
    只有新鲜 TCP 位姿与目标同时满足位置、姿态误差门限，才允许上层规划
    下一段路径；不具备 ``read_pose_snapshot`` 的离线替身继续走旧等待逻辑。
    """
    read_snapshot = getattr(pose_session, "read_pose_snapshot", None)
    if not callable(read_snapshot):
        return _wait_robot_steady(pose_session, timeout_s)
    desired = np.asarray(target, dtype=np.float64).reshape(4, 4)
    pos_tol = max(0.0, float(position_tolerance_mm))
    rot_tol = max(0.0, float(rotation_tolerance_deg))
    deadline = time.monotonic() + float(timeout_s)
    last_reason = ""
    while time.monotonic() < deadline:
        snapshot = read_snapshot()
        if not bool(snapshot.get("power_on")):
            raise MotionExecutionError("机器人未上电，停止等待目标到位")
        if bool(snapshot.get("collision")):
            raise RuntimeError("运动后检测到碰撞标志，立即停止流程")
        actual = pose_session.pose_sdk_to_transform_mm(
            snapshot["pose_values_sdk_m_rad"]
        )
        position_error_mm = float(np.linalg.norm(actual[:3, 3] - desired[:3, 3]))
        rotation_error_deg = _rotation_delta_between_transforms_deg(actual, desired)
        power_on = bool(snapshot.get("power_on"))
        steady = bool(snapshot.get("steady"))
        if power_on and steady and position_error_mm <= pos_tol and rotation_error_deg <= rot_tol:
            return snapshot, np.asarray(actual, dtype=np.float64).copy()
        last_reason = (
            f"power={power_on} steady={steady} "
            f"position_error={position_error_mm:.3f}mm "
            f"rotation_error={rotation_error_deg:.3f}deg"
        )
        time.sleep(ROBOT_STEADY_POLL_INTERVAL_S)
    raise RuntimeError(f"等待机器人到达目标超时：{last_reason}")


def _joint_error_deg(actual_joints: Any, target_joints: Any) -> float:
    """返回两组关节角的最大最小圆周差（度）。"""
    actual = np.asarray(actual_joints, dtype=np.float64).reshape(-1)
    target = np.asarray(target_joints, dtype=np.float64).reshape(-1)
    if actual.size != target.size or actual.size == 0:
        return float("inf")
    delta = np.arctan2(np.sin(actual - target), np.cos(actual - target))
    return float(np.max(np.abs(delta)) * 180.0 / math.pi)


@motion_failure_boundary
def _wait_robot_joints_reached(
    pose_session: Any,
    target_joints: Any,
    timeout_s: float = 45.0,
    *,
    joint_tolerance_deg: float = 0.5,
) -> tuple[dict[str, Any], np.ndarray]:
    """等待 moveJoint 的关节目标到位。

    关节回原点不能用保存时的 TCP 位姿作为唯一完成条件：保存的 home
    TCP 可能因工具/TCP 配置变化而失效，但关节目标本身仍然是控制器实际
    执行的目标。到位后由调用方单独报告 TCP 不一致，而不是无期限超时。
    """
    read_snapshot = getattr(pose_session, "read_pose_snapshot", None)
    if not callable(read_snapshot):
        return _wait_robot_steady(pose_session, timeout_s)
    target = np.asarray(target_joints, dtype=np.float64).reshape(-1)
    tolerance = max(0.0, float(joint_tolerance_deg))
    deadline = time.monotonic() + float(timeout_s)
    last_reason = ""
    while time.monotonic() < deadline:
        snapshot = read_snapshot()
        if not bool(snapshot.get("power_on")):
            raise MotionExecutionError("机器人未上电，停止等待关节到位")
        if bool(snapshot.get("collision")):
            raise RuntimeError("回原点后检测到碰撞标志，立即停止流程")
        actual = pose_session.pose_sdk_to_transform_mm(
            snapshot["pose_values_sdk_m_rad"]
        )
        joints = snapshot.get("joints_rad", [])
        joint_error_deg = _joint_error_deg(joints, target)
        power_on = bool(snapshot.get("power_on"))
        steady = bool(snapshot.get("steady"))
        if power_on and steady and joint_error_deg <= tolerance:
            return snapshot, np.asarray(actual, dtype=np.float64).copy()
        last_reason = (
            f"power={power_on} steady={steady} "
            f"joint_error={joint_error_deg:.3f}deg"
        )
        time.sleep(ROBOT_STEADY_POLL_INTERVAL_S)
    raise RuntimeError(f"等待机器人关节目标到位超时：{last_reason}")


@motion_failure_boundary
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
        if not bool(snapshot.get("power_on")):
            raise MotionExecutionError("运动会话检测到机器人未上电，停止本轮，不再等待或重试后续分组")
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
    timeout_s: float = 45.0,
) -> np.ndarray:
    """在拍摄前确认停稳，并在延迟期间发现运动时重新计时。

    ``timeout_s`` 是这一段等待的总预算。具备实时快照接口的机器人会在
    延迟期间轮询 ``steady``；离线替身没有该接口时保留原来的两次稳态读
    取和一次阻塞等待，避免改变旧测试/脚本的调用契约。
    """
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError(f"机器人停稳等待超时必须是正数：{timeout_s}")
    deadline = time.monotonic() + timeout
    remaining = max(0.01, deadline - time.monotonic())
    _, actual = _wait_robot_steady(pose_session, timeout_s=remaining)
    delay = max(0.0, float(settle_delay_s))
    if delay > 0.0:
        read_snapshot = getattr(pose_session, "read_pose_snapshot", None)
        if callable(read_snapshot):
            stable_until = time.monotonic() + delay
            while time.monotonic() < stable_until:
                now = time.monotonic()
                if now >= deadline:
                    raise RuntimeError("停稳后额外等待超时")
                time.sleep(min(ROBOT_STEADY_POLL_INTERVAL_S, stable_until - now))
                snapshot = read_snapshot()
                if not bool(snapshot.get("power_on")):
                    raise MotionExecutionError("停稳等待期间机器人未上电")
                if bool(snapshot.get("collision")):
                    raise RuntimeError("停稳等待期间检测到碰撞标志")
                if not bool(snapshot.get("steady")):
                    stable_until = time.monotonic() + delay
                try:
                    actual = pose_session.pose_sdk_to_transform_mm(
                        snapshot["pose_values_sdk_m_rad"]
                    )
                except Exception:
                    # 快照中的姿态只用于刷新返回值；稳态门本身仍由字段判断。
                    pass
        else:
            time.sleep(delay)
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        raise RuntimeError("停稳后确认超时")
    _, actual = _wait_robot_steady(pose_session, timeout_s=remaining)
    return np.asarray(actual, dtype=np.float64).copy()


def _rotation_delta_between_transforms_deg(
    reference: np.ndarray,
    current: np.ndarray,
) -> float:
    """返回两个TCP旋转矩阵之间的完整姿态差，而不是只比较光轴。"""
    relative = np.asarray(reference, dtype=np.float64)[:3, :3].T @ np.asarray(
        current, dtype=np.float64,
    )[:3, :3]
    cosine = float((np.trace(relative) - 1.0) * 0.5)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


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


def _resolve_home_joints(
    home: Any,
    motion_session: Any | None,
) -> tuple[np.ndarray, str]:
    """读取本次运动应使用的原始点关节目标。

    控制器的 ``getHomePosition`` 是示教器保存的原始点；项目 JSON 只保留
    兼容备用值。这样控制器原始点更新后，自动流程不会继续发送旧文件中的
    关节角。
    """
    if motion_session is not None:
        getter = getattr(motion_session, "get_controller_home_joints", None)
        if callable(getter):
            try:
                values = np.asarray(getter(), dtype=np.float64).reshape(-1)
                if values.size == 6 and np.all(np.isfinite(values)):
                    return values, "controller_home"
            except Exception as exc:
                print(f"[HOME] 读取控制器保存原始点失败，回退软件文件：{exc}", flush=True)
    if home is None:
        raise RuntimeError("没有可用的原始点：控制器原始点读取失败，且软件备用文件为空")
    values = np.asarray(home.joints_rad, dtype=np.float64).reshape(-1)
    if values.size != 6 or not np.all(np.isfinite(values)):
        raise RuntimeError(f"软件备用原始点关节数据无效：{values.tolist()}")
    return values, "software_file"


def _request_motion_confirmation(label: str, prompt: str) -> str:
    """兼容旧流程的运动确认；当前顺序流程不再调用它。"""
    print(f"[MOTION_CONFIRM_REQUIRED] {label}", flush=True)
    return input(prompt).strip().lower()


def _request_next_hole_confirmation(current_hole_id: int, next_hole_id: int) -> str:
    """当前孔完成后，仅在开始处理下一个已选孔前暂停一次。"""
    print(
        f"[NEXT_HOLE_CONFIRM_REQUIRED] 当前孔={int(current_hole_id)}，"
        f"下一个检测孔={int(next_hole_id)}",
        flush=True,
    )
    return input(
        f"孔{int(current_hole_id)}已完成定位和目标点运动；"
        f"输入 m 开始处理孔{int(next_hole_id)}，其他任意键停止："
    ).strip().lower()


def _confirm_map_next_hole_if_needed(
    args: Any,
    current_hole_id: int,
    next_hole_id: int,
    *,
    context: str,
) -> None:
    """地图执行的孔间安全确认；预览不暂停，真实运动默认逐孔等待。"""
    if not bool(getattr(args, "execute", False)):
        return
    command = _request_next_hole_confirmation(current_hole_id, next_hole_id)
    if command != "m":
        raise RuntimeError(
            f"用户在{context}孔{int(current_hole_id)}完成后停止流程"
        )


def _numeric_sdk_return_code(value: Any) -> int | None:
    """尽量把SDK返回值转换为整数码；未知对象返回None。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _accept_ignored_move_line_if_target_reached(
    label: str,
    target: np.ndarray,
    response: Any,
    pose_session: Any,
    motion_session: Any,
    steady_timeout_s: float | None = None,
) -> np.ndarray | None:
    """仅在控制器忽略了无效重复运动且实测已到目标时放行。

    AUBO 的 ``AUBO_REQUEST_IGNORE=13`` 不是成功码，不能直接吞掉。常见
    的安全场景是程序上一次异常退出后机器人已经停在验证位姿，本轮再次
    下发完全相同的安全高度位姿，控制器会忽略这个零位移请求。只有重新
    读取TCP并确认误差足够小，才把它视为“已到位”；目标仍有明显误差时
    继续抛错，避免把真正被控制器拒绝的运动误判为成功。
    """
    if _numeric_sdk_return_code(response) != AUBO_REQUEST_IGNORE_CODE:
        return None
    if pose_session is None:
        return None

    wait_timeout = 45.0 if steady_timeout_s is None else float(steady_timeout_s)
    try:
        _, actual = _wait_robot_reached(
            pose_session, target, timeout_s=wait_timeout,
        )
        actual = np.asarray(actual, dtype=np.float64).copy()
        desired = np.asarray(target, dtype=np.float64)
        position_error_mm = float(np.linalg.norm(actual[:3, 3] - desired[:3, 3]))
        rotation_error_deg = _rotation_delta_between_transforms_deg(actual, desired)
    except Exception as exc:
        raise RuntimeError(
            f"{label} 收到AUBO_REQUEST_IGNORE(13)，且无法确认实际TCP位姿：{exc}"
        ) from exc

    if (
        position_error_mm > MOTION_REQUEST_IGNORE_POSITION_TOLERANCE_MM
        or rotation_error_deg > MOTION_REQUEST_IGNORE_ROTATION_TOLERANCE_DEG
    ):
        return None

    print(
        f"[MOTION] {label} moveLine返回AUBO_REQUEST_IGNORE(13)，"
        f"但目标已到位：位置误差={position_error_mm:.3f}mm，"
        f"姿态误差={rotation_error_deg:.3f}deg；按无动作成功处理",
        flush=True,
    )
    _wait_motion_session_steady(motion_session, wait_timeout)
    return actual


def _record_motion_distance(
    args: Any,
    label: str,
    current: np.ndarray,
    target: np.ndarray,
    *,
    motion_type: str,
    motion_profile: str | None = None,
    status: str = "completed",
    extra: str = "",
) -> None:
    """把一次已完成运动交给本轮计时器保存距离信息。

    运动辅助函数分布在多个工作流模块中，统一从 ``args`` 取当前轮的
    TimingRecorder，避免给每个兼容调用点增加一个必填参数。统计失败时
    静默回退，不影响原有运动流程。
    """
    timing = getattr(args, "_timing_recorder", None)
    recorder = getattr(timing, "record_motion", None)
    if not callable(recorder):
        return
    details: dict[str, Any] = {}
    if motion_profile:
        details["motion_profile"] = str(motion_profile)
    if extra:
        details["description"] = str(extra)
    try:
        recorder(
            label,
            np.asarray(current, dtype=np.float64),
            np.asarray(target, dtype=np.float64),
            motion_type=motion_type,
            status=status,
            **details,
        )
    except Exception:
        # 距离统计是审计增强项，绝不能让它改变运动安全链路。
        return


@motion_failure_boundary
def _confirm_and_move_line(label: str, current: np.ndarray, target: np.ndarray, args: Any,
                           motion_session: Any, pose_session: Any, extra: str = "",
                           require_confirmation: bool = True,
                           motion_profile: str = "precision",
                           steady_timeout_s: float | None = None) -> np.ndarray:
    require_camera_stream_healthy()
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
    wait_timeout = 45.0 if steady_timeout_s is None else float(steady_timeout_s)
    if not math.isfinite(wait_timeout) or wait_timeout <= 0.0:
        raise ValueError(f"机器人停稳等待超时必须是正数：{steady_timeout_s}")
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
    _wait_motion_session_steady(motion_session, wait_timeout)
    # 调用方的 current_tcp 在异常恢复或旧兼容路径中可能滞后；距离统计
    # 优先使用下发前重新读取的实际TCP起点，但不让读取失败阻断运动。
    motion_start = np.asarray(current, dtype=np.float64).copy()
    try:
        _, measured_start = _require_safe_snapshot(pose_session)
        motion_start = np.asarray(measured_start, dtype=np.float64).copy()
    except LocalizationHardwareError:
        raise
    except Exception:
        pass
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
        _wait_motion_session_steady(motion_session, wait_timeout)
        response = motion_session.move_line(sdk_pose, speed_m_s, acc_m_s2)
    print("[MOTION]", response)
    if not response:
        raise RuntimeError(f"{label} moveLine 下发失败：空响应")
    if not sdk_ok(response[-1]):
        ignored_target = _accept_ignored_move_line_if_target_reached(
            label,
            target,
            response[-1],
            pose_session,
            motion_session,
            steady_timeout_s=wait_timeout,
        )
        if ignored_target is not None:
            _record_motion_distance(
                args, label, motion_start, target,
                motion_type="move_line",
                motion_profile=motion_profile,
                extra=extra,
            )
            return ignored_target
        response_code = _numeric_sdk_return_code(response[-1])
        if response_code == AUBO_REQUEST_IGNORE_CODE:
            raise RuntimeError(
                f"{label} moveLine 下发失败：{response}；"
                "SDK返回13(AUBO_REQUEST_IGNORE)，但实测TCP尚未到达目标"
            )
        raise RuntimeError(f"{label} moveLine 下发失败：{response}")
    _, actual = _wait_robot_reached(
        pose_session, target, timeout_s=wait_timeout,
    )
    _wait_motion_session_steady(motion_session, wait_timeout)
    _record_motion_distance(
        args, label, motion_start, target,
        motion_type="move_line",
        motion_profile=motion_profile,
        extra=extra,
    )
    return actual


def _confirm_and_move_home(
    home: Any,
    motion_session: Any,
    pose_session: Any,
    *,
    timing: TimingRecorder | None = None,
    target_joints: Any | None = None,
    home_source: str | None = None,
) -> np.ndarray:
    snapshot, current = _require_safe_snapshot(pose_session)
    current_joints = np.asarray(snapshot.get("joints_rad", []), dtype=np.float64)
    home_joints = np.asarray(
        home.joints_rad if target_joints is None else target_joints,
        dtype=np.float64,
    ).reshape(-1)
    if home_joints.size != 6 or not np.all(np.isfinite(home_joints)):
        raise RuntimeError(f"原始点关节目标无效：{home_joints.tolist()}")
    source = str(
        home_source
        or ("software_file" if target_joints is None else "controller_home")
    )
    # 控制器 getHomePosition 只返回关节原始点，没有可靠的对应 TCP 记录。
    # 只有使用软件备用点时才用 JSON 中保存的 TCP 做一致性提示。
    home_target = None
    if target_joints is None:
        home_target = pose_session.pose_sdk_to_transform_mm(home.tcp_pose_m_rad)
    if current_joints.size == home_joints.size and current_joints.size > 0:
        max_joint_error_deg = _joint_error_deg(current_joints, home_joints)
        if max_joint_error_deg <= 0.5:
            if home_target is not None:
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
            print(
                f"[MOTION] 当前已在{source}原点关节位，"
                f"最大关节偏差={max_joint_error_deg:.3f}°，跳过回原点运动",
                flush=True,
            )
            return current
    if home_target is not None:
        _print_motion_preview(
            "回机械臂原点（关节运动）", current, home_target,
            f"home={home.name} created_at={home.created_at} source={source}",
        )
    else:
        print(
            "\n[MOTION] 回机械臂原点（关节运动）\n"
            f"  current TCP XYZ(mm): {np.round(current[:3, 3], 3).tolist()}\n"
            f"  target joints(rad): {np.round(home_joints, 9).tolist()}\n"
            f"  source: {source}",
            flush=True,
        )
    print("[MOTION] 自动执行回原点，无需输入 m", flush=True)
    speed = math.radians(20.0)
    acc = math.radians(40.0)
    _wait_motion_session_steady(motion_session)
    home_joints_list = [float(value) for value in home_joints]
    try:
        response = motion_session.move_joint(home_joints_list, speed, acc)
    except RuntimeError as exc:
        if "尚未静止" not in str(exc):
            raise
        print("[MOTION] 运动会话仍在收敛，等待稳定后重试一次 moveJoint", flush=True)
        _wait_motion_session_steady(motion_session)
        response = motion_session.move_joint(home_joints_list, speed, acc)
    print("[MOTION]", response)
    from aubo_workbench.motion_control import sdk_ok
    # moveJoint 的完成条件应以关节目标为准。保存的 TCP 位姿可能因当前
    # 工具/TCP 配置变化而与同一组关节的真实 TCP 不一致，不能因此把已经
    # 到位的关节运动判成超时。
    settled_snapshot, actual = _wait_robot_joints_reached(
        pose_session, home_joints,
    )
    _wait_motion_session_steady(motion_session)
    if not response or not sdk_ok(response[-1]):
        settled_joints = np.asarray(settled_snapshot.get("joints_rad", []), dtype=np.float64)
        reached_home = _joint_error_deg(settled_joints, home_joints) <= 0.5
        if not reached_home:
            raise RuntimeError(f"回原点 moveJoint 下发失败：{response}")
        print("[MOTION] 控制器返回非成功码，但关节已处于原点，按到位处理")
    if home_target is not None:
        tcp_position_error_mm = float(np.linalg.norm(actual[:3, 3] - home_target[:3, 3]))
        tcp_axis_error_deg = _angle_deg(actual[:3, 2], home_target[:3, 2])
        if tcp_position_error_mm > 5.0 or tcp_axis_error_deg > 2.0:
            print(
                "\n[SAFETY WARNING] 回原点关节已到位，但保存的 home TCP 与当前 TCP 不一致：\n"
                f"  TCP位置差={tcp_position_error_mm:.3f} mm，光轴差={tcp_axis_error_deg:.3f}°\n"
                f"  当前TCP XYZ(mm)={np.round(actual[:3, 3], 3).tolist()}\n"
                f"  保存TCP XYZ(mm)={np.round(home_target[:3, 3], 3).tolist()}\n"
                "  将以当前实际 TCP 继续流程；如需修正，请在机械臂控制页重新设置原始点。",
                flush=True,
            )
    recorder = getattr(timing, "record_motion", None)
    if callable(recorder):
        try:
            record_target = home_target if home_target is not None else actual
            recorder(
                "回机械臂原点（关节运动）",
                np.asarray(current, dtype=np.float64),
                np.asarray(record_target, dtype=np.float64),
                motion_type="move_joint",
                motion_profile="home",
                description=f"home={getattr(home, 'name', '原始点')} source={source}",
            )
        except Exception:
            pass
    return actual




from aubo_workbench import hole_capture_workflow as _hole_capture_workflow  # noqa: E402


def _runtime_facade(module: Any, name: str) -> Any:
    target = getattr(module, name)

    @wraps(target)
    def facade(*args: Any, **kwargs: Any) -> Any:
        module.install_runtime(globals())
        return target(*args, **kwargs)

    facade._runtime_facade_target = name
    return facade


_capture_initial_multi_hole_selection = _runtime_facade(
    _hole_capture_workflow, "_capture_initial_multi_hole_selection",
)
_center_scatter_p95 = _runtime_facade(_hole_capture_workflow, "_center_scatter_p95")
_coarse_burst_stable = _runtime_facade(_hole_capture_workflow, "_coarse_burst_stable")
_fine_burst_stable = _runtime_facade(_hole_capture_workflow, "_fine_burst_stable")
_capture_coarse_burst = _runtime_facade(_hole_capture_workflow, "_capture_coarse_burst")
_capture_fine_burst = _runtime_facade(_hole_capture_workflow, "_capture_fine_burst")
_capture_fine_with_recovery = _runtime_facade(
    _hole_capture_workflow, "_capture_fine_with_recovery",
)

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


from aubo_workbench.hole_localization_visualization import (  # noqa: E402
    _write_visualization_pair,
    _draw_black_text,
    _save_grouping_plan_visualization,
    _save_group_capture_visualization,
    _project_base_point_to_pixel,
    _save_batch_fine_final_result_overlay,
)


from aubo_workbench.batch_localization_math import (  # noqa: E402
    _assign_detections_to_projection,
    _minimum_cost_maximum_assignment_without_scipy,
    _fit_batch_projected_anchor_correction,
    _fit_batch_fine_joint_transform,
    _batch_fine_joint_transform_stable,
)


def _plane_estimate_from_info(info: dict[str, Any], label: str) -> PlaneEstimate:
    """把单次环带点云拟合结果统一转换成跨帧融合使用的平面估计。"""
    return PlaneEstimate(
        ring_coverage_ratio=info.get("ring_coverage_ratio"),
        ring_max_gap_deg=info.get("ring_max_gap_deg"),
        raw_points_camera_mm=info.get("raw_points_camera_mm"),
        raw_pixels=info.get("raw_pixels"),
        ring_pixels=info.get("ring_pixels"),
        surface_selected_mask=info.get("surface_selected_mask"),
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


from aubo_workbench.hole_localization_diagnostics import (  # noqa: E402
    _wrap_angle_rad,
    _rotation_distance_deg,
    _effective_timing_summary,
    _build_comparison_hole_diagnostics,
    _build_comparison_run_diagnostics,
)
from aubo_workbench.hole_localization_report import (  # noqa: E402
    write_progress_checkpoint,
    write_result_summary,
)


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
        "coarse_center_source": summary.get("center_source"),
        "coarse_center_source_counts": summary.get("center_source_counts", {}),
        "coarse_geometric_valid_frames": summary.get("geometric_valid_frames", 0),
        "coarse_geometric_center_scatter_p95_px": summary.get(
            "geometric_center_scatter_p95_px"
        ),
        "coarse_center_fusion_source": summary.get("center_fusion_source"),
        "coarse_tracking_distance_p95_px": summary.get("tracking_distance_p95_px"),
        "coarse_ring_points_median": int(np.median([
            item.plane.ring_points for item in observations
            if item.error is None and item.plane is not None
        ])),
        "coarse_ring_coverage_min_ratio": summary.get("ring_coverage_min_ratio"),
        "coarse_ring_max_gap_deg": summary.get("ring_max_gap_deg"),
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


from aubo_workbench import hole_localization_cache as _localization_cache  # noqa: E402


_cache_intrinsics_dict = _runtime_facade(_localization_cache, "_cache_intrinsics_dict")
_cache_measurements_from_observations = _runtime_facade(_localization_cache, "_cache_measurements_from_observations")
_cache_cloud_from_coarse_observations = _runtime_facade(_localization_cache, "_cache_cloud_from_coarse_observations")
_cache_cloud_from_cached_entry_reprojected = _runtime_facade(_localization_cache, "_cache_cloud_from_cached_entry_reprojected")
_save_coarse_pointcloud_image = _runtime_facade(_localization_cache, "_save_coarse_pointcloud_image")
_cache_entry_from_observations = _runtime_facade(_localization_cache, "_cache_entry_from_observations")
_apply_cached_geometry_to_hole = _runtime_facade(_localization_cache, "_apply_cached_geometry_to_hole")
_reuse_initial_pointcloud_geometry_for_batch_fine = _runtime_facade(_localization_cache, "_reuse_initial_pointcloud_geometry_for_batch_fine")
_validate_coarse_cache_at_current_pose = _runtime_facade(_localization_cache, "_validate_coarse_cache_at_current_pose")
_upsert_persistent_cache_entry = _runtime_facade(_localization_cache, "_upsert_persistent_cache_entry")
_load_persistent_coarse_cache_for_run = _runtime_facade(_localization_cache, "_load_persistent_coarse_cache_for_run")

def _fuse_coarse(
    observations: list[Observation], cfg: TwoStageConfig,
    min_valid_frames: int | None = None,
    max_center_scatter_p95_px: float | None = None,
    max_tracking_distance_p95_px: float | None = None,
) -> dict[str, Any]:
    valid = [item for item in observations if item.error is None and item.plane is not None]
    # Live pose estimation may use incomplete arcs for navigation only. Formal
    # capture and its early-stop decision must pass the per-frame coverage gate.
    ring_rejected = 0
    if not all(item.stage == "batch_coarse_pose_refine" for item in valid):
        filtered = []
        for item in valid:
            coverage = item.plane.ring_coverage_ratio
            gap = item.plane.ring_max_gap_deg
            if (coverage is not None and (not np.isfinite(coverage) or coverage < cfg.coarse_min_ring_coverage_ratio)) or (
                gap is not None and (not np.isfinite(gap) or gap > cfg.coarse_max_ring_gap_deg)
            ):
                ring_rejected += 1
            else:
                filtered.append(item)
        valid = filtered
    required = cfg.min_coarse_valid if min_valid_frames is None else int(min_valid_frames)
    if len(valid) < required:
        raise RuntimeError(f"粗定位有效帧不足：{len(valid)}/{required}；环带覆盖不足帧={ring_rejected}")
    center_source_counts: dict[str, int] = {}
    for item in valid:
        source = str(getattr(item, "center_source", "unknown") or "unknown")
        center_source_counts[source] = center_source_counts.get(source, 0) + 1

    centers = np.asarray(
        [item.center_px for item in valid],
        dtype=np.float64,
    )
    center_fusion_source = "pointcloud_center"
    plane_points = _fuse_vectors(
        [item.plane.point_camera_mm for item in valid], "coarse plane points",
    )
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
        "ring_quality_rejected_frames": ring_rejected,
        "accepted_frame_indices": [int(x.frame_index) for x in valid],
        "ring_coverage_min_ratio": min((x.plane.ring_coverage_ratio for x in valid if x.plane.ring_coverage_ratio is not None), default=None),
        "ring_max_gap_deg": max((x.plane.ring_max_gap_deg for x in valid if x.plane.ring_max_gap_deg is not None), default=None),
        "valid_frames": len(valid), "total_frames": len(observations), "center_px": center,
        "center_scatter_p95_px": center_scatter_p95_px,
        "center_source": (
            str(getattr(valid[0], "center_source", "unknown"))
            if len(valid) == 1 else center_fusion_source
        ),
        "center_source_counts": center_source_counts,
        "geometric_valid_frames": 0,
        "geometric_center_scatter_p95_px": None,
        "center_fusion_source": center_fusion_source,
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


def _fine_center_multimodal_summary(
    centers_px: np.ndarray,
    scatter_gate_px: float,
    *,
    min_cluster_frames: int = 2,
) -> dict[str, Any]:
    """识别两组都稳定、彼此明显分离的精定位圆心候选。"""
    centers = np.asarray(centers_px, dtype=np.float64).reshape(-1, 2)
    if len(centers) < max(4, 2 * int(min_cluster_frames)):
        return {"multimodal": False, "clusters": []}
    link_radius = max(0.5, 2.0 * float(scatter_gate_px))
    remaining = set(range(len(centers)))
    components: list[list[int]] = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            neighbors = {
                index for index in remaining
                if float(np.linalg.norm(centers[index] - centers[current])) <= link_radius
            }
            remaining.difference_update(neighbors)
            component.update(neighbors)
            frontier.extend(neighbors)
        components.append(sorted(component))

    stable_clusters: list[dict[str, Any]] = []
    for indices in components:
        if len(indices) < int(min_cluster_frames):
            continue
        values = centers[indices]
        center = np.median(values, axis=0)
        distances = np.linalg.norm(values - center, axis=1)
        p95 = float(np.percentile(distances, 95))
        if p95 <= float(scatter_gate_px):
            stable_clusters.append({
                "frame_indices": indices,
                "frame_count": len(indices),
                "center_px": center,
                "scatter_p95_px": p95,
            })
    stable_clusters.sort(key=lambda item: int(item["frame_count"]), reverse=True)
    minimum_separation = max(3.0, 4.0 * float(scatter_gate_px))
    separation = None
    multimodal = False
    if len(stable_clusters) >= 2:
        separation = float(np.linalg.norm(
            np.asarray(stable_clusters[0]["center_px"], dtype=np.float64)
            - np.asarray(stable_clusters[1]["center_px"], dtype=np.float64)
        ))
        multimodal = separation >= minimum_separation
    return {
        "multimodal": multimodal,
        "clusters": stable_clusters,
        "largest_cluster_separation_px": separation,
        "minimum_separation_px": minimum_separation,
        "link_radius_px": link_radius,
    }


def _fuse_fine(observations: list[Observation], cfg: TwoStageConfig,
               max_center_scatter_p95_px: float | None = None,
               reject_multimodal: bool = False) -> dict[str, Any]:
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
    multimodal = _fine_center_multimodal_summary(raw_centers, scatter_gate)
    summary["center_multimodality"] = multimodal
    if reject_multimodal and multimodal["multimodal"]:
        raise RuntimeError(
            "共享精定位圆心存在双峰歧义："
            f"稳定候选簇={len(multimodal['clusters'])}，"
            f"前两簇间距={float(multimodal['largest_cluster_separation_px']):.3f}px"
        )
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
        "center_px_distorted": np.median(np.asarray([
            item.ellipse.get("center_px_distorted", item.center_px)
            for item in valid
        ], dtype=np.float64), axis=0),
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
            "geometric_anchor_distance_px": item.geometric_anchor_distance_px,
        }
        if item.plane is not None:
            row.update({
                "plane_rmse_mm": item.plane.rmse_mm, "ring_points": item.plane.ring_points,
                "ring_coverage_ratio": item.plane.ring_coverage_ratio,
                "ring_max_gap_deg": item.plane.ring_max_gap_deg,
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


def _write_report(
    run_dir: Path,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    timing: TimingRecorder | None = None,
) -> None:
    """写入运行报告，并单独记录报告文件的生成时间。"""
    frames_path = run_dir / "frames.csv"
    with artifact_measure(
        timing,
        "report/write_frames_csv",
        artifact_kind="report_csv",
        paths=[str(frames_path)],
        row_count=len(rows),
    ):
        _write_csv(frames_path, rows)

    # 在序列化前同步一次，让 report.json 至少包含本次 CSV 写入的耗时。
    if timing is not None:
        report["timing"] = timing.snapshot()
    report_path = run_dir / "report.json"
    report_payload = json.dumps(_jsonable(report), ensure_ascii=False, indent=2)
    with artifact_measure(
        timing,
        "report/write_json",
        artifact_kind="report_json",
        paths=[str(report_path)],
    ):
        report_path.write_text(report_payload, encoding="utf-8")


def _write_progress_checkpoint(
    run_dir: Path,
    report: dict[str, Any],
    *,
    timing: TimingRecorder | None = None,
) -> Path:
    """只写轻量进度文件，避免每个孔都重写完整诊断报告。"""
    return write_progress_checkpoint(run_dir, report, timing=timing)


def _write_result_summary(
    run_dir: Path,
    report: dict[str, Any],
    *,
    timing: TimingRecorder | None = None,
) -> dict[str, str]:
    """写入现场使用的简明TXT/JSON结果摘要。"""
    return write_result_summary(run_dir, report, timing=timing)




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


from aubo_workbench import group_pose_workflow as _group_pose_workflow  # noqa: E402


_move_to_sequential_coarse_pose = _runtime_facade(_group_pose_workflow, "_move_to_sequential_coarse_pose")
_move_to_shared_observation_pose = _runtime_facade(_group_pose_workflow, "_move_to_shared_observation_pose")
_move_to_shared_coarse_pose = _runtime_facade(_group_pose_workflow, "_move_to_shared_coarse_pose")
_plan_batch_coarse_group_pose = _runtime_facade(_group_pose_workflow, "_plan_batch_coarse_group_pose")
_bounded_shared_fine_pose_adjustment = _runtime_facade(
    _group_pose_workflow, "_bounded_shared_fine_pose_adjustment",
)
_move_shared_fine_group_adjustment = _runtime_facade(
    _group_pose_workflow, "_move_shared_fine_group_adjustment",
)
_save_group_pose_refinement_visualization = _runtime_facade(_group_pose_workflow, "_save_group_pose_refinement_visualization")
_refine_shared_coarse_group_pose = _runtime_facade(_group_pose_workflow, "_refine_shared_coarse_group_pose")

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


def _split_batch_localization_groups(
    holes: list[dict[str, Any]],
    current_tcp: np.ndarray,
    handeye: Any,
    fixed_rz_rad: float,
    intrinsics: Any,
    target_height_mm: float,
    view_margin_px: float,
    max_view_span_ratio: float,
    *,
    max_group_size: int | None = None,
    max_aspect_ratio: float | None = None,
    adjacency_distance_factor: float = 1.8,
    max_normal_spread_deg: float | None = None,
    max_depth_span_mm: float | None = None,
    max_xy_diameter_mm: float | None = None,
    preferred_min_group_size: int | None = None,
    search_timeout_s: float | None = None,
    return_diagnostics: bool = False,
    return_metadata: bool = False,
) -> (
    list[list[dict[str, Any]]]
    | tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]
    | tuple[list[list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]
):
    """按空间连通性、紧凑度和共享视野拆分定位组。

    保留旧的前八个位置参数，新增约束全部使用关键字参数，避免破坏
    现有调用和测试。默认不强制新上限，正式共享粗/精定位路径会显式传入
    各自的最大孔数、紧凑度和法向门限。
    """
    groups, diagnostics, metadata = _split_batch_localization_groups_with_metadata(
        holes,
        current_tcp,
        handeye,
        fixed_rz_rad,
        intrinsics,
        target_height_mm,
        view_margin_px,
        max_view_span_ratio,
        max_group_size=max_group_size,
        max_aspect_ratio=max_aspect_ratio,
        adjacency_distance_factor=adjacency_distance_factor,
        max_normal_spread_deg=max_normal_spread_deg,
        max_depth_span_mm=max_depth_span_mm,
        max_xy_diameter_mm=max_xy_diameter_mm,
        preferred_min_group_size=preferred_min_group_size,
        search_timeout_s=search_timeout_s,
    )
    if return_metadata:
        return groups, diagnostics, metadata
    return (groups, diagnostics) if return_diagnostics else groups


def _split_batch_localization_groups_with_metadata(
    holes: list[dict[str, Any]],
    current_tcp: np.ndarray,
    handeye: Any,
    fixed_rz_rad: float,
    intrinsics: Any,
    target_height_mm: float,
    view_margin_px: float,
    max_view_span_ratio: float,
    *,
    max_group_size: int | None = None,
    max_aspect_ratio: float | None = None,
    adjacency_distance_factor: float = 1.8,
    max_normal_spread_deg: float | None = None,
    max_depth_span_mm: float | None = None,
    max_xy_diameter_mm: float | None = None,
    preferred_min_group_size: int | None = None,
    search_timeout_s: float | None = None,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    """返回共享定位分组及其可视化/审计诊断。"""
    if not holes:
        return [], [], {
            "algorithm": "minimum_feasible_exact_cover_compact_v2",
            "hole_count": 0,
            "theoretical_min_group_count": 0,
            "minimum_feasible_group_count": 0,
            "selected_group_count": 0,
            "search_complete": True,
            "fallback_used": False,
        }
    width = float(getattr(intrinsics, "width", 0.0))
    height = float(getattr(intrinsics, "height", 0.0))
    if width <= 0.0 or height <= 0.0:
        raise RuntimeError("共享精定位补拍分组缺少有效图像尺寸")
    ratio_gate = float(max_view_span_ratio)
    if not (0.0 < ratio_gate <= 1.0):
        raise ValueError("共享精定位补拍视野跨度比例必须在(0, 1]内")

    def grouping_planner(group: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            _, geometry = _plan_batch_coarse_group_pose(
                group, current_tcp, handeye, fixed_rz_rad, intrinsics,
                target_height_mm, view_margin_px,
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        bbox = np.asarray(geometry["group_bbox_px"], dtype=np.float64).reshape(4)
        span_px = np.asarray([
            float(bbox[2] - bbox[0]),
            float(bbox[3] - bbox[1]),
        ])
        span_ratio = span_px / np.asarray([width, height], dtype=np.float64)
        geometry = dict(geometry)
        geometry.update({
            "projected_bbox_span_px": span_px,
            "projected_bbox_span_ratio": span_ratio,
            "view_span_ratio": float(np.max(span_ratio)),
        })
        if float(np.max(span_ratio)) > ratio_gate + 1e-9:
            geometry.update({
                "ok": False,
                "error": (
                    f"projected_bbox_span_ratio={float(np.max(span_ratio)):.4f} "
                    f"> max_view_span_ratio={ratio_gate:.4f}"
                ),
            })
        else:
            geometry["ok"] = True
        return geometry

    return group_holes_spatially_with_metadata(
        holes,
        planner=grouping_planner,
        max_group_size=max_group_size,
        max_aspect_ratio=max_aspect_ratio,
        adjacency_distance_factor=adjacency_distance_factor,
        max_normal_spread_deg=max_normal_spread_deg,
        max_depth_span_mm=max_depth_span_mm,
        max_xy_diameter_mm=max_xy_diameter_mm,
        preferred_min_group_size=preferred_min_group_size,
        search_timeout_s=search_timeout_s,
    )


def _split_batch_localization_groups_edge_first(
    holes: list[dict[str, Any]],
    current_tcp: np.ndarray,
    handeye: Any,
    fixed_rz_rad: float,
    intrinsics: Any,
    target_height_mm: float,
    view_margin_px: float,
    max_view_span_ratio: float,
    *,
    max_group_size: int | None = None,
    max_aspect_ratio: float | None = None,
    adjacency_distance_factor: float = 1.8,
    max_normal_spread_deg: float | None = None,
    max_depth_span_mm: float | None = None,
    max_xy_diameter_mm: float | None = None,
    preferred_min_group_size: int | None = None,
    search_timeout_s: float | None = None,
    return_diagnostics: bool = True,
    return_metadata: bool = True,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    """分两阶段生成策略四的分组：选区外围先拍，内部随后拍。

    外围阶段的组上限固定为 ``min(3, 配置上限)``，内部阶段沿用策略四
    的配置上限（默认五孔）。两个阶段使用同一套空间、紧凑度、视野和
    法向硬约束；外围优先只改变处理顺序和组上限，不为了凑满三孔放宽
    长宽比或其它紧凑性门限。
    """
    classified_holes, boundary_metadata = classify_selected_hole_boundary_layers(
        holes,
        adjacency_distance_factor=adjacency_distance_factor,
    )
    effective_max_group_size = (
        5 if max_group_size is None else int(max_group_size)
    )
    if effective_max_group_size < 1 or effective_max_group_size > 5:
        raise ValueError("策略四每组最多孔数必须在1到5之间")
    if preferred_min_group_size is not None and int(preferred_min_group_size) < 1:
        raise ValueError("策略四优选最小组大小必须至少为1")

    edge_holes = [
        hole for hole in classified_holes
        if str(hole.get("boundary_class", "edge")) == "edge"
    ]
    interior_holes = [
        hole for hole in classified_holes
        if str(hole.get("boundary_class", "edge")) != "edge"
    ]
    preferred_group_floor = (
        3 if preferred_min_group_size is None
        else max(1, min(3, int(preferred_min_group_size)))
    )
    started = time.monotonic()

    def remaining_timeout() -> float | None:
        if search_timeout_s is None:
            return None
        remaining = float(search_timeout_s) - (time.monotonic() - started)
        # The underlying solver requires a positive timeout.  A tiny bounded
        # slice still lets it return its audited fallback instead of hanging.
        return max(0.001, remaining)

    all_groups: list[list[dict[str, Any]]] = []
    all_diagnostics: list[dict[str, Any]] = []
    phase_metadata: dict[str, Any] = {}

    def run_phase(
        phase_holes: list[dict[str, Any]],
        *,
        phase: str,
        boundary_class: str,
        phase_max_group_size: int,
    ) -> None:
        phase_started = time.monotonic()
        if not phase_holes:
            phase_metadata[phase] = {
                "hole_count": 0,
                "group_count": 0,
                "max_group_size": int(phase_max_group_size),
                "elapsed_s": 0.0,
                "groups": [],
            }
            return
        phase_timeout = remaining_timeout()
        groups, diagnostics, metadata = _split_batch_localization_groups_with_metadata(
            phase_holes,
            current_tcp,
            handeye,
            fixed_rz_rad,
            intrinsics,
            target_height_mm,
            view_margin_px,
            max_view_span_ratio,
            max_group_size=min(int(phase_max_group_size), len(phase_holes)),
            # 外围组同样必须通过长宽比门限。近似共线的三孔会拆成
            # 两孔/单孔，避免共同点云在长条形区域内不完整。
            max_aspect_ratio=max_aspect_ratio,
            adjacency_distance_factor=adjacency_distance_factor,
            max_normal_spread_deg=max_normal_spread_deg,
            max_depth_span_mm=max_depth_span_mm,
            max_xy_diameter_mm=max_xy_diameter_mm,
            preferred_min_group_size=min(
                preferred_group_floor, int(phase_max_group_size)
            ),
            search_timeout_s=phase_timeout,
        )
        phase_offset = len(all_groups)
        for local_index, group in enumerate(groups, start=1):
            diagnostic = (
                dict(diagnostics[local_index - 1])
                if local_index - 1 < len(diagnostics)
                and isinstance(diagnostics[local_index - 1], dict)
                else {}
            )
            group_number = phase_offset + local_index
            for hole in group:
                hole["group_phase"] = phase
                hole["group_boundary_class"] = boundary_class
                hole["edge_first_group_index"] = int(group_number)
                hole["edge_first_phase_group_index"] = int(local_index)
            diagnostic["group_index"] = int(group_number)
            diagnostic["group_phase"] = phase
            diagnostic["boundary_class"] = boundary_class
            diagnostic["edge_first_phase_group_index"] = int(local_index)
            diagnostic["edge_first"] = True
            diagnostic["aspect_ratio_gate_disabled_for_edge"] = False
            diagnostic["aspect_ratio_gate_applied_for_edge"] = (
                boundary_class == "edge" and max_aspect_ratio is not None
            )
            all_groups.append(group)
            all_diagnostics.append(diagnostic)
        phase_metadata[phase] = {
            "hole_count": len(phase_holes),
            "group_count": len(groups),
            "max_group_size": int(phase_max_group_size),
            "elapsed_s": float(time.monotonic() - phase_started),
            "preferred_min_group_size": min(
                preferred_group_floor, int(phase_max_group_size)
            ),
            "max_aspect_ratio": max_aspect_ratio,
            "aspect_ratio_gate_disabled_for_edge": False,
            "aspect_ratio_gate_applied_for_edge": (
                boundary_class == "edge" and max_aspect_ratio is not None
            ),
            "search": metadata,
            "groups": [
                {
                    "group_index": len(all_groups) - len(groups) + index,
                    "hole_ids": [int(hole["hole_id"]) for hole in group],
                    "hole_count": len(group),
                }
                for index, group in enumerate(groups, start=1)
            ],
        }

    # Ordering is deliberate: all outer-layer groups are complete before an
    # inner group can move the robot.  A geometry-constrained remainder is still
    # legal and remains in its phase as a singleton or pair.
    run_phase(
        edge_holes,
        phase="edge_first",
        boundary_class="edge",
        phase_max_group_size=min(3, effective_max_group_size),
    )
    run_phase(
        interior_holes,
        phase="interior_after_edge",
        boundary_class="interior",
        phase_max_group_size=effective_max_group_size,
    )

    expected_ids = sorted(int(hole["hole_id"]) for hole in classified_holes)
    actual_ids = [
        int(hole["hole_id"])
        for group in all_groups
        for hole in group
    ]
    coverage_ok = (
        len(actual_ids) == len(set(actual_ids))
        and sorted(actual_ids) == expected_ids
    )
    if not coverage_ok:
        raise RuntimeError(
            "外围优先分组未完整覆盖选区或存在重复分配："
            f"expected={expected_ids}, actual={sorted(actual_ids)}"
        )

    metadata = {
        "algorithm": "edge_first_two_phase_boundary_exact_cover_v1",
        "hole_count": len(classified_holes),
        "selected_group_count": len(all_groups),
        "grouping_phase_order": ["edge_first", "interior_after_edge"],
        "edge_first": True,
        "edge_max_group_size": min(3, effective_max_group_size),
        "interior_max_group_size": effective_max_group_size,
        "edge_hole_count": len(edge_holes),
        "interior_hole_count": len(interior_holes),
        "coverage_check": {
            "ok": coverage_ok,
            "expected_hole_ids": expected_ids,
            "actual_hole_ids": sorted(actual_ids),
        },
        "boundary_classification": boundary_metadata,
        "phases": phase_metadata,
        "compactness_policy": (
            "same_aspect_ratio_xy_diameter_and_adjacency_gates_for_edge_and_interior"
        ),
        "edge_aspect_ratio_gate": {
            "enabled": bool(max_aspect_ratio is not None),
            "max_aspect_ratio": max_aspect_ratio,
        },
        "search_timeout_s": (
            None if search_timeout_s is None else float(search_timeout_s)
        ),
        "elapsed_s": float(time.monotonic() - started),
        "group_size_policy": (
            "edge_first_max3_then_interior_max5_prefer3_to5_allow1_to2"
        ),
    }
    return all_groups, all_diagnostics, metadata


def _unpack_batch_grouping_result(
    result: Any,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    """兼容旧测试/调用方返回的两元分组结果。"""
    if isinstance(result, tuple):
        if len(result) == 3:
            groups, diagnostics, metadata = result
            return list(groups), list(diagnostics), dict(metadata or {})
        if len(result) == 2:
            groups, diagnostics = result
            return list(groups), list(diagnostics), {}
    return list(result or []), [], {}


def _split_batch_fine_supplement_groups(
    holes: list[dict[str, Any]],
    current_tcp: np.ndarray,
    handeye: Any,
    fixed_rz_rad: float,
    intrinsics: Any,
    target_height_mm: float,
    view_margin_px: float,
    max_view_span_ratio: float,
    *,
    max_group_size: int | None = None,
    max_aspect_ratio: float | None = None,
    adjacency_distance_factor: float = 1.8,
    max_normal_spread_deg: float | None = None,
    max_depth_span_mm: float | None = None,
    return_diagnostics: bool = False,
) -> list[list[dict[str, Any]]] | tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """兼容旧调用名；补拍和首次共享拍摄现在使用同一套空间分组规则。"""
    return _split_batch_localization_groups(
        holes, current_tcp, handeye, fixed_rz_rad, intrinsics,
        target_height_mm, view_margin_px, max_view_span_ratio,
        max_group_size=max_group_size,
        max_aspect_ratio=max_aspect_ratio,
        adjacency_distance_factor=adjacency_distance_factor,
        max_normal_spread_deg=max_normal_spread_deg,
        max_depth_span_mm=max_depth_span_mm,
        return_diagnostics=return_diagnostics,
    )


def _coarse_expected_point_base(hole: dict[str, Any]) -> np.ndarray:
    """Return the current coarse-stage association anchor for one hole.

    The live group-pose refinement writes a temporary refined anchor here.  A
    normal per-hole or legacy call has no such field and continues to use the
    original selection point.
    """
    value = hole.get(
        "coarse_pose_refined_center_base_mm",
        hole.get("initial_center_base_mm"),
    )
    return np.asarray(value, dtype=np.float64).reshape(3)


from aubo_workbench import shared_localization_workflow as _shared_workflow  # noqa: E402


_batch_coarse_localization_at_340mm = _runtime_facade(
    _shared_workflow, "_batch_coarse_localization_at_340mm",
)
_flush_rgb_queue_until_fresh = _runtime_facade(
    _shared_workflow, "_flush_rgb_queue_until_fresh",
)
_batch_fine_localization_at_260mm = _runtime_facade(
    _shared_workflow, "_batch_fine_localization_at_260mm",
)

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


from aubo_workbench import sequential_hole_workflow as _sequential_workflow  # noqa: E402


_run_sequential_hole_workflow = _runtime_facade(
    _sequential_workflow, "_run_sequential_hole_workflow",
)

def _request_next_cycle_confirmation(
    cycle_index: int, *, timing: TimingRecorder | None = None,
) -> str:
    def request() -> str:
        return input(
            f"第{int(cycle_index)}轮已完成且机器人已停稳；"
            "输入 m 开始下一轮初始拍摄，其他任意键结束："
        ).strip().lower()

    if timing is None:
        return request()
    with timing.measure(
        "operator/next_cycle_confirmation",
        category="operator_wait",
        level="leaf",
        cycle_index=int(cycle_index),
    ):
        return request()




def _restore_batch_fine_pose_for_final_motion(
    hole_id: str,
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
        actual,
        desired,
        args,
        motion_session,
        pose_session,
        target_height_mm=target_height_mm,
        target_stage="恢复该孔粗定位位姿后执行精定位XY",
        descent_guard_mm=SHARED_OBSERVATION_MIN_DESCENT_MM,
    )


def _move_to_batch_final_tcp_direct(
    hole_id: str,
    current_tcp: np.ndarray,
    target_tcp: np.ndarray,
    args: Any,
    motion_session: Any,
    pose_session: Any,
    *,
    safe_margin_mm: float = THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM,
    descent_guard_mm: float = SHARED_OBSERVATION_MIN_DESCENT_MM,
) -> np.ndarray:
    """沿安全路径直接到共享精拍计算出的最终TCP。

    共享精拍结束后，粗定位参考位姿只用于生成每孔自己的姿态和粗定位Z，
    不再需要让机器人先真实回到每个孔的260 mm观察位。最终TCP已经把
    ChArUco XY补偿已经合并进去，因此可以在安全高度直接平移到
    最终XY/姿态，再用纯基坐标Z下降到安全位和最终点。

    该函数只由“共享精拍后的最终动作”调用，逐孔精定位和共享拍摄路径
    继续使用原有函数，避免新路径改变其它模式。
    """
    actual = np.asarray(current_tcp, dtype=np.float64).reshape(4, 4).copy()
    desired = np.asarray(target_tcp, dtype=np.float64).reshape(4, 4).copy()
    margin = max(10.0, float(safe_margin_mm))
    guard_mm = max(10.0, float(descent_guard_mm))

    # 当前TCP如果已经高于目标安全高度，不要在每次重试时继续叠加
    # margin；否则程序中途退出后再次运行会逐次向上抬升。
    safe_z = max(float(actual[2, 3]), float(desired[2, 3]) + margin)
    lift = actual.copy()
    lift[2, 3] = safe_z
    if abs(float(lift[2, 3] - actual[2, 3])) > 0.2:
        actual = _confirm_and_move_line(
            f"孔{hole_id}最终动作前纯Z抬升",
            actual,
            lift,
            args,
            motion_session,
            pose_session,
            f"共享精拍最终TCP安全路径；纯Z抬升，安全余量={margin:.1f}mm",
            require_confirmation=False,
            motion_profile="transit",
        )

    high_target = desired.copy()
    high_target[2, 3] = safe_z
    actual = _confirm_and_move_line(
        f"孔{hole_id}最终动作安全高度平移",
        actual,
        high_target,
        args,
        motion_session,
        pose_session,
        "在安全高度直接到最终XY和姿态；不下降到工件低位区域",
        require_confirmation=False,
        motion_profile="transit",
    )

    descent_clearance_mm = float(safe_z - desired[2, 3])
    if descent_clearance_mm < guard_mm:
        raise RuntimeError(
            f"孔{hole_id}最终TCP纯Z下降安全余量不足："
            f"{descent_clearance_mm:.3f}mm < {guard_mm:.3f}mm"
        )
    descent_guard = desired.copy()
    descent_guard[2, 3] = float(desired[2, 3]) + guard_mm
    actual = _confirm_and_move_line(
        f"孔{hole_id}最终TCP纯Z下降到上方{guard_mm:.0f}mm",
        actual,
        descent_guard,
        args,
        motion_session,
        pose_session,
        "保持最终XY和姿态不变；进入最后纯Z下降安全段",
        require_confirmation=False,
        motion_profile="approach",
    )
    return _confirm_and_move_line(
        f"孔{hole_id}最终TCP纯Z下降{guard_mm:.0f}mm",
        actual,
        desired,
        args,
        motion_session,
        pose_session,
        f"保持最终XY和姿态不变；最后纯Z下降{guard_mm:.0f}mm到最终点",
        require_confirmation=False,
        motion_profile="precision",
    )


def _move_to_shared_fine_pose(
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
    current_tcp: np.ndarray,
    target: np.ndarray,
    args: Any,
    motion_session: Any,
    pose_session: Any,
    *,
    target_height_mm: float = 260.0,
    target_stage: str = "精定位",
    descent_guard_mm: float = 0.0,
    safe_margin_mm: float = THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM,
) -> np.ndarray:
    """沿安全高度移动到两阶段流程的目标相机高度。"""
    # current_tcp可能来自上一段规划缓存。先读取下发前的实测TCP，保证
    # 离开低位的第一段只改变基坐标Z，不会因缓存漂移夹带XY横移。
    _, measured_tcp = _require_safe_snapshot(pose_session)
    actual = np.asarray(measured_tcp, dtype=np.float64).reshape(4, 4).copy()
    desired = np.asarray(target, dtype=np.float64).reshape(4, 4).copy()
    margin = max(10.0, float(safe_margin_mm))
    # 已经位于目标上方安全高度时保持当前Z，避免重复启动/异常重试时
    # 每次再抬升一个完整安全余量。
    safe_z = max(
        float(actual[2, 3]),
        float(desired[2, 3]) + margin,
    )
    lift = actual.copy()
    lift[2, 3] = safe_z
    if abs(float(lift[2, 3] - actual[2, 3])) > 0.2:
        actual = _confirm_and_move_line(
            f"孔{hole_id}进入安全高度", actual, lift, args, motion_session, pose_session,
            "仅修改基坐标Z；不经过工件低位区域", require_confirmation=False,
            motion_profile="transit",
        )
    high_target = desired.copy()
    high_target[2, 3] = safe_z
    actual = _confirm_and_move_line(
        f"移动到孔{hole_id}上方安全位姿", actual, high_target, args,
        motion_session, pose_session, "安全高度横移并调整目标姿态；不下降",
        require_confirmation=False, motion_profile="transit",
    )
    guard_mm = max(0.0, float(descent_guard_mm))
    if guard_mm > 0.0:
        if float(safe_z - desired[2, 3]) < guard_mm:
            raise RuntimeError(
                f"孔{hole_id}纯Z下降安全余量不足："
                f"{float(safe_z - desired[2, 3]):.3f}mm < {guard_mm:.3f}mm"
            )
        descent_guard = desired.copy()
        descent_guard[2, 3] = float(desired[2, 3]) + guard_mm
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
        actual, desired, args, motion_session, pose_session,
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


def _resolve_hole_map_path(raw_path: Any = None, *, for_build: bool = False) -> tuple[Path, str]:
    """解析孔位地图路径；建图默认每次使用独立版本目录。

    建图路径只能落在专用地图目录的版本子目录中，不能写入粗定位缓存、
    ``current.json`` 或任意外部 JSON。未指定调用路径时自动使用当前地图。
    """
    if raw_path:
        path = Path(raw_path).expanduser()
        if path.suffix.lower() != ".json":
            path = path / "hole_map.json"
        if for_build:
            root = HOLE_LOCALIZATION_MAPS_DIR.resolve()
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    "建立孔位地图只能自动保存到 hole_localization_maps 的版本目录，"
                    f"禁止使用该路径：{resolved}"
                ) from exc
            if (
                resolved.name.lower() != "hole_map.json"
                or len(relative.parts) != 2
                or relative.parts[0].lower() == "calls"
            ):
                raise ValueError(
                    "建立孔位地图的自定义路径必须是地图目录下一级版本目录中的 hole_map.json"
                )
        else:
            path = resolve_hole_map_path(path)
        return path, path.parent.name or path.stem
    if not for_build:
        current_pointer = HOLE_LOCALIZATION_MAPS_DIR / "current.json"
        if not current_pointer.is_file():
            raise ValueError(
                "当前没有可调用的孔位地图，请先建立地图或选择一个地图 JSON"
            )
        path = resolve_hole_map_path(current_pointer)
        return path, path.parent.name or path.stem
    HOLE_LOCALIZATION_MAPS_DIR.mkdir(parents=True, exist_ok=True)
    map_id = f"hole-map-{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
    path = HOLE_LOCALIZATION_MAPS_DIR / map_id / "hole_map.json"
    return path, map_id


def _find_batch_pointcloud_archive(run_dir: Path, report: dict[str, Any]) -> Path | None:
    """找出本轮共享340 mm粗定位产生的全孔点云归档。"""
    candidates: list[Path] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if "pointcloud" in str(key).lower() and isinstance(item, (str, Path)):
                    candidate = Path(item).expanduser()
                    if candidate.suffix.lower() == ".npz":
                        candidates.append(candidate)
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit((report.get("stages") or {}).get("batch_coarse_results") or {})
    candidates.extend(run_dir.glob("*batch_coarse_340_all_holes_pointcloud.npz"))
    existing: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved.is_file() and resolved not in seen:
            seen.add(resolved)
            existing.append(resolved)
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def _write_hole_map_from_report(
    args: Any,
    report: dict[str, Any],
    run_dir: Path,
    *,
    timing: TimingRecorder | None = None,
) -> Path:
    """把本轮粗定位结果固化为地图，并记录可选的逐孔精定位参考。"""
    sector_id = getattr(args, "sector_id", None)
    rotary_map = sector_id is not None
    if rotary_map:
        sector_id = validate_sector_id(int(sector_id))
        # current.json 是入口指针，不能把新的版本正文写到它上面；每次
        # 建立/替换一个扇区都生成不可变版本，便于回退和追溯。
        raw_map_path = getattr(args, "hole_map_path", None)
        if raw_map_path and Path(raw_map_path).name.lower() != "current.json":
            map_path, map_id = _resolve_hole_map_path(raw_map_path, for_build=True)
        else:
            map_path, map_id = _resolve_hole_map_path(None, for_build=True)
        existing_payload: dict[str, Any] | None = None
        existing_source = HOLE_LOCALIZATION_MAPS_DIR / "current.json"
        if raw_map_path and Path(raw_map_path).name.lower() != "current.json":
            existing_source = Path(raw_map_path).expanduser()
        try:
            if existing_source.is_file():
                candidate = load_hole_map(existing_source)
                if int(candidate.get("schema_version", -1)) == 4:
                    existing_payload = candidate
                else:
                    report.setdefault("hole_map", {})["legacy_map_not_merged"] = True
        except (OSError, ValueError) as exc:
            # current 指针损坏不应阻塞重新建第一个扇区；把原因写进报告，
            # 同时保留旧目录中的文件，不静默删除任何历史地图。
            report.setdefault("hole_map", {})["existing_map_load_warning"] = str(exc)
        payload = build_rotary_sector_map_payload(
            report,
            map_id=map_id,
            source_run_dir=run_dir,
            handeye_path=getattr(args, "handeye", None),
            camera_identity=report.get("camera"),
            charuco_model_source=CHARUCO_XY_MODEL_SOURCE,
            charuco_model_matrix=CHARUCO_XY_MODEL_MATRIX,
            charuco_model_bias_mm=CHARUCO_XY_MODEL_BIAS_MM,
            tcp_xy_offset_mm=getattr(args, "tcp_xy_offset_mm", None),
            sector_id=sector_id,
            existing_payload=existing_payload,
        )
    else:
        map_path, map_id = _resolve_hole_map_path(
            getattr(args, "hole_map_path", None), for_build=True,
        )
        payload = build_hole_map_payload(
            report,
            map_id=map_id,
            source_run_dir=run_dir,
            handeye_path=getattr(args, "handeye", None),
            camera_identity=report.get("camera"),
            charuco_model_source=CHARUCO_XY_MODEL_SOURCE,
            charuco_model_matrix=CHARUCO_XY_MODEL_MATRIX,
            charuco_model_bias_mm=CHARUCO_XY_MODEL_BIAS_MM,
            tcp_xy_offset_mm=getattr(args, "tcp_xy_offset_mm", None),
        )
    pointcloud_archive = _find_batch_pointcloud_archive(run_dir, report)
    overlay_source = (
        (report.get("stages") or {}).get("batch_coarse_results") or {}
    ).get("overlay_path")
    if overlay_source is None:
        overlay_source = (
            (report.get("stages") or {}).get("batch_coarse_plan") or {}
        ).get("latest_overlay_path")
    with artifact_measure(
        timing,
        "hole_map/write_artifacts",
        artifact_kind="hole_map_artifacts",
        paths=[str(map_path.parent)],
    ):
        if rotary_map:
            sector_name = sector_key(int(sector_id))
            artifact_dir = map_path.parent / "sectors" / sector_name
            sector_payload = {
                "holes": ((payload.get("sectors") or {}).get(sector_name) or {}).get("holes", {}),
            }
            sector_artifacts = export_hole_map_artifacts(
                pointcloud_archive,
                artifact_dir,
                sector_payload,
                overlay_path=overlay_source,
            )
            relative_prefix = Path("sectors") / sector_name
            prefixed_artifacts: dict[str, Any] = {}
            for key, value in sector_artifacts.items():
                if key in {"raw_npz", "ply", "centers_ply", "preview_jpg", "overlay_jpg"} and value:
                    prefixed_artifacts[key] = (relative_prefix / str(value)).as_posix()
                else:
                    prefixed_artifacts[key] = value
            payload["sectors"][sector_name]["artifacts"] = prefixed_artifacts
            payload["artifacts"] = {
                "status": "ready",
                "coordinate_frame": "robot_base_mm",
                "sectors": {
                    str(key): (value.get("artifacts") or {})
                    for key, value in (payload.get("sectors") or {}).items()
                    if isinstance(value, dict) and value.get("artifacts")
                },
            }
        else:
            payload["artifacts"] = export_hole_map_artifacts(
                pointcloud_archive,
                map_path.parent,
                payload,
                overlay_path=overlay_source,
            )
    with artifact_measure(
        timing,
        "hole_map/write_json",
        artifact_kind="hole_map_json",
        paths=[str(map_path)],
    ):
        saved = save_hole_map(payload, map_path)
    current_map_path = HOLE_LOCALIZATION_MAPS_DIR / "current.json"
    selected_sector_ready = True
    current_update_blocked_reason: str | None = None
    if rotary_map:
        selected_sector_ready = str(
            ((payload.get("sectors") or {}).get(sector_key(int(sector_id))) or {}).get(
                "status", ""
            )
        ).lower() in {"valid", "ready", "completed"}
    if rotary_map and not selected_sector_ready:
        # 当前扇区本轮还有失败/遗漏孔时只保存草稿版本，不能把它发布成
        # current；否则会用不完整的新版本覆盖旧的可用扇区地图。
        current_path = None
        current_update_blocked_reason = "selected_sector_partial"
    else:
        with artifact_measure(
            timing,
            "hole_map/write_current_pointer",
            artifact_kind="hole_map_current_pointer",
            paths=[str(current_map_path)],
        ):
            current_path = publish_current_hole_map(
                payload,
                saved,
                current_map_path,
            )
    if rotary_map:
        sector_name = sector_key(int(sector_id))
        current_sector = (payload.get("sectors") or {}).get(sector_name) or {}
        ready_hole_ids = sorted(
            int(item["hole_id"])
            for item in (current_sector.get("holes") or {}).values()
            if isinstance(item, dict) and str(item.get("status", "ready")) in {"ready", "completed"}
        )
        deferred_value: Any = current_sector.get("quality_summary", {}).get("deferred_holes", [])
    else:
        ready_hole_ids = sorted(
            int(item["hole_id"])
            for item in (payload.get("holes") or {}).values()
        )
        deferred_value = payload.get("quality_summary", {}).get("deferred_holes", [])
    effective_map_mode = str(payload.get("map_build_localization_mode", "coarse_only"))
    if rotary_map:
        effective_map_mode = str(
            ((payload.get("sectors") or {}).get(sector_key(int(sector_id))) or {}).get(
                "map_build_localization_mode", effective_map_mode
            )
        )
    hole_map_summary = {
        "path": str(saved),
        "map_id": map_id,
        "status": payload.get("status"),
        "map_build_localization_mode": effective_map_mode,
        "reference_policy": payload.get("reference_policy"),
        "sector_id": int(sector_id) if rotary_map else None,
        "ready_holes": ready_hole_ids,
        "deferred_holes": deferred_value,
        "ready_sector_ids": (payload.get("quality_summary") or {}).get("ready_sector_ids", []) if rotary_map else None,
        "fine_reference_ready_holes": (
            (payload.get("quality_summary") or {}).get("fine_reference_ready_holes", [])
            if not rotary_map else
            (payload.get("quality_summary") or {}).get("fine_reference_ready_holes", 0)
        ),
        "fine_reference_missing_holes": (
            (payload.get("quality_summary") or {}).get("fine_reference_missing_holes", [])
            if not rotary_map else
            (payload.get("quality_summary") or {}).get("fine_reference_missing_holes", 0)
        ),
        "scope": payload.get("scope"),
        "current_path": None if current_path is None else str(current_path),
        "current_updated": current_path is not None,
        "current_update_blocked_reason": current_update_blocked_reason,
        "artifacts": payload.get("artifacts", {}),
    }
    report["hole_map"] = hole_map_summary
    report.setdefault("stages", {})["hole_map_build"] = {
        "status": "ready" if payload.get("status") == "valid" else "partial",
        "path": str(saved),
        "map_id": map_id,
        "source_run_dir": str(run_dir),
        "ready_holes": hole_map_summary["ready_holes"],
        "deferred_holes": hole_map_summary["deferred_holes"],
        "current_path": hole_map_summary["current_path"],
        "current_updated": hole_map_summary["current_updated"],
        "current_update_blocked_reason": hole_map_summary[
            "current_update_blocked_reason"
        ],
        "artifacts": hole_map_summary["artifacts"],
        "map_build_localization_mode": hole_map_summary[
            "map_build_localization_mode"
        ],
        "fine_reference_ready_holes": hole_map_summary[
            "fine_reference_ready_holes"
        ],
        "fine_reference_missing_holes": hole_map_summary[
            "fine_reference_missing_holes"
        ],
        "policy": (
            "build_one_rotary_sector_with_per_hole_340mm_coarse_and_260mm_fine_reference;"
            "call_requires_runtime_260mm_fine_localization"
            if rotary_map and effective_map_mode == "per_hole" else
            "build_one_rotary_sector_with_340mm_coarse_only;"
            "call_requires_runtime_260mm_fine_localization"
            if rotary_map else
            "build_per_hole_340mm_coarse_and_260mm_fine_reference;"
            "call_requires_runtime_fine_localization"
            if effective_map_mode == "per_hole" else
            "build_340mm_coarse_only;call_requires_runtime_fine_localization"
        ),
    }
    report["status"] = (
        "hole_map_ready_with_deferred_holes"
        if payload.get("status") == "partial" else "hole_map_ready"
    )
    print(
        f"[HOLE_MAP_READY] path={saved} "
        f"ready_holes={hole_map_summary['ready_holes']} "
        f"map_mode={hole_map_summary['map_build_localization_mode']} "
        f"fine_reference_ready={hole_map_summary['fine_reference_ready_holes']} "
        f"deferred={hole_map_summary['deferred_holes']} "
        f"current={hole_map_summary['current_path'] or 'unchanged'}",
        flush=True,
    )
    return saved


def _copy_hole_map_version_artifacts(source_map_path: Path, target_map_path: Path) -> None:
    """把旧版本的相对产物复制到返修生成的新版本目录。"""
    source_dir = source_map_path.parent.resolve()
    target_dir = target_map_path.parent.resolve()
    if source_dir == target_dir or not source_dir.is_dir():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        if item.name.lower() in {"hole_map.json", "calls"}:
            continue
        destination = target_dir / item.name
        try:
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            elif item.is_file():
                shutil.copy2(item, destination)
        except OSError as exc:
            raise RuntimeError(
                f"复制旧地图产物失败：{item} -> {destination}: {exc}"
            ) from exc


@camera_stream_session
def run_hole_map_repair(args: Any, handeye: Any, model: Any) -> int:
    """现场重新定位一个坏孔，并以新版本原子替换该孔的地图记录。"""
    if not bool(getattr(args, "execute", False)):
        raise ValueError("单孔地图返修必须显式使用 --execute，避免只改地图不做现场采集")
    sector_id_raw = getattr(args, "sector_id", None)
    repair_hole_id_raw = getattr(args, "repair_hole_id", None)
    if sector_id_raw is None or repair_hole_id_raw is None:
        raise ValueError("单孔地图返修必须同时指定 --sector-id 和 --repair-hole-id")
    sector_id = validate_sector_id(int(sector_id_raw))
    repair_hole_id = int(repair_hole_id_raw)
    repair_mode = "coarse_refresh"

    source_map_path, _ = _resolve_hole_map_path(getattr(args, "hole_map_path", None))
    source_payload = load_hole_map(source_map_path)
    if int(source_payload.get("schema_version", -1)) != 4:
        raise ValueError("单孔粗定位更新只支持六扇区 v4 粗地图，请先重新建立地图")
    validate_hole_map(
        source_payload,
        sector_id=sector_id,
        requested_hole_ids=[repair_hole_id],
    )
    source_hole = get_hole(source_payload, repair_hole_id, sector_id=sector_id)
    environment = source_payload.get("environment") or {}
    stored_handeye_hash = environment.get("handeye_sha256")
    current_handeye_hash = file_sha256(getattr(args, "handeye", ""))
    if stored_handeye_hash and stored_handeye_hash != current_handeye_hash:
        raise ValueError("现有六扇区地图的手眼标定已变化，禁止单孔返修")
    cfg = replace(
        TwoStageConfig.from_namespace(args),
        # 粗地图更新使用新鲜单孔粗/精定位；精定位结果只用于本轮验证，
        # 不写回地图，也不从旧地图恢复 XY。
        batch_coarse_localization=False,
        batch_fine_localization=False,
        batch_fine_joint_localization=False,
        batch_fine_pointcloud_xy_fusion=False,
        shared_cache_validation=False,
    )
    args.batch_coarse_localization = False
    args.batch_fine_localization = False
    args.batch_fine_joint_localization = False
    args.batch_fine_pointcloud_xy_fusion = False
    args.shared_cache_validation = False
    args.reuse_coarse_cache = False
    args.reuse_persistent_coarse_cache = False
    args.move_final_xy = False
    cfg.validate(reuse_coarse_cache=False)

    cycle_index = 1
    run_dir = _new_two_stage_run_dir(cycle_index)
    report = _new_two_stage_report(args, cfg, run_dir, cycle_index)
    report.update({
        "mode": "hole_map_repair",
        "hole_map_mode": "repair",
        "session_reselect_enabled": False,
        "selection_mode": "coarse_map_seed",
        "map_repair": {
            "source_map_path": str(source_map_path),
            "source_map_id": source_payload.get("map_id"),
            "sector_id": sector_id,
            "hole_id": repair_hole_id,
            "repair_mode": repair_mode,
            "policy": (
                "fresh_per_hole_coarse_and_fine_without_persisting_fine_result"
            ),
            "original_hole": source_hole,
        },
    })
    timing = TimingRecorder()
    timing.attach_report(report)
    setattr(args, "_timing_recorder", timing)
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
        from aubo_workbench.motion_control import AuboMotionSession, HOME_POINT_FILE, load_home_point
        from aubo_workbench.robot import AuboPoseSession

        with timing.measure("robot/load_home_point"):
            home = load_home_point()
        if home is None:
            print(
                f"[HOME] 软件备用原始点不存在或不可读：{HOME_POINT_FILE}；"
                "连接运动控制后将优先读取控制器原始点。",
                flush=True,
            )
        else:
            print(
                f"[HOME] 已加载软件备用原始点：{HOME_POINT_FILE}；"
                f"created_at={home.created_at}；关节目标(rad)={home.joints_rad}",
                flush=True,
            )
        with timing.measure("robot/connect_pose_session"):
            pose_session = AuboPoseSession()
            pose_session.connect()
            initial_snapshot, initial_tcp = _require_safe_snapshot(pose_session)
        current_tcp = np.asarray(initial_tcp, dtype=np.float64).copy()
        report["robot_initial_tcp_pose_m_rad"] = initial_snapshot["pose_values_sdk_m_rad"]
        report["home_point"] = home.to_dict() if home is not None else None
        if not handeye.validated_for_motion and not args.allow_experimental_handeye:
            raise RuntimeError(
                "手眼证据未通过生产运动门，拒绝地图单孔返修；"
                "实验验证请显式添加 --allow-experimental-handeye"
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
        controller_home_joints, home_source = _resolve_home_joints(home, motion_session)
        report["home_point_source"] = home_source
        report["controller_home_joints_rad"] = controller_home_joints.tolist()
        print(
            f"[HOME] 实际运动目标来源={home_source}；"
            f"关节目标(rad)={controller_home_joints.tolist()}",
            flush=True,
        )
        with timing.measure("robot/move_home"):
            current_tcp = _confirm_and_move_home(
                home,
                motion_session,
                pose_session,
                timing=timing,
                target_joints=controller_home_joints,
                home_source=home_source,
            )

        with timing.measure("camera/start_rgbd_pipeline"):
            report["stages"]["rgbd_startup"] = {"status": "starting"}
            rgbd_pipeline, align, chain = init_pipeline()
            pipeline_runtime.update({
                "rgbd_pipeline": rgbd_pipeline,
                "align": align,
                "chain": chain,
            })
            camera_identity = get_device_identity(rgbd_pipeline)
            expected_serial = str(environment.get("camera_serial") or "").strip()
            actual_serial = str(camera_identity.get("serial_number") or "").strip()
            if expected_serial and actual_serial and expected_serial != actual_serial:
                raise RuntimeError(
                    f"返修相机序列号不匹配：地图={expected_serial}，当前={actual_serial}"
                )
            report["camera"] = camera_identity
            report["stages"]["rgbd_startup"] = {"status": "ready"}
        report["cycle_start_tcp_pose_m_rad"] = transform_to_sdk_pose_m_rad(current_tcp)

        coarse_center = np.asarray(
            source_hole.get("coarse_center_base_mm"), dtype=np.float64,
        ).reshape(3)
        coarse_plane = np.asarray(
            source_hole.get("coarse_plane_point_base_mm", coarse_center),
            dtype=np.float64,
        ).reshape(3)
        coarse_normal = _unit(
            np.asarray(
                source_hole.get("coarse_normal_toward_camera_base"),
                dtype=np.float64,
            ).reshape(3),
            "map repair source coarse normal",
        )
        seed_hole = {
            "hole_id": repair_hole_id,
            "class_id": int(source_hole.get("class_id") or 0),
            "class_name": source_hole.get("class_name") or "hole",
            "initial_center_base_mm": coarse_center.tolist(),
            "initial_plane_point_base_mm": coarse_plane.tolist(),
            "initial_plane_normal_base": coarse_normal.tolist(),
        }
        report["map_repair"]["seed"] = {
            "coarse_center_base_mm": coarse_center.tolist(),
            "coarse_plane_point_base_mm": coarse_plane.tolist(),
            "coarse_normal_toward_camera_base": coarse_normal.tolist(),
        }
        _write_report(run_dir, report, rows, timing=timing)
        result_code = _run_two_stage_localization_cycle(
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
            initial_holes_override=[seed_hole],
        )
        if int(result_code) != 0:
            raise RuntimeError(f"地图单孔返修定位流程未完成：exit={result_code}")
        repaired_results = ((report.get("final_result") or {}).get("holes") or [])
        repaired = next(
            (
                item for item in repaired_results
                if isinstance(item, dict)
                and int(item.get("hole_id", -1)) == repair_hole_id
                and str(item.get("status", "")).lower() == "completed"
            ),
            None,
        )
        if repaired is None:
            raise RuntimeError(
                f"孔 {repair_hole_id} 返修未产生 completed 结果，地图保持不变"
            )

        map_path, map_id = _resolve_hole_map_path(None, for_build=True)
        _copy_hole_map_version_artifacts(source_map_path, map_path)
        repaired_payload = build_rotary_sector_hole_repair_payload(
            source_payload,
            report,
            map_id=map_id,
            source_run_dir=run_dir,
            handeye_path=getattr(args, "handeye", None),
            camera_identity=camera_identity,
            charuco_model_source=CHARUCO_XY_MODEL_SOURCE,
            sector_id=sector_id,
            hole_id=repair_hole_id,
            tcp_xy_offset_mm=None,
            charuco_model_matrix=None,
            charuco_model_bias_mm=None,
        )
        saved = save_hole_map(repaired_payload, map_path)
        current_path = publish_current_hole_map(
            repaired_payload,
            saved,
            HOLE_LOCALIZATION_MAPS_DIR / "current.json",
        )
        report["hole_map"] = {
            "path": str(saved),
            "map_id": map_id,
            "current_path": None if current_path is None else str(current_path),
            "current_updated": current_path is not None,
            "sector_id": sector_id,
            "hole_id": repair_hole_id,
            "repair_mode": repair_mode,
            "source_map_id": source_payload.get("map_id"),
        }
        report["map_repair"].update({
            "status": "committed",
            "new_map_path": str(saved),
            "new_map_id": map_id,
            "current_updated": current_path is not None,
            "new_hole": ((repaired_payload.get("sectors") or {})
                          .get(sector_key(sector_id), {}).get("holes", {})
                          .get(f"H{repair_hole_id:02d}")),
        })
        report["status"] = "map_repair_ready"
        report["session_end_reason"] = "map_repair_committed"
        _write_report(run_dir, report, rows, timing=timing)
        print(
            f"[HOLE_MAP_REPAIR_DONE] sector={sector_id} hole={repair_hole_id} "
            f"mode={repair_mode} map={saved} current={current_path or 'unchanged'}",
            flush=True,
        )
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["session_end_reason"] = "map_repair_failed_map_unchanged"
        if isinstance(exc, LocalizationHardwareError):
            report["failure_type"] = exc.failure_type
            print(f"[HARDWARE_FAILED] {exc}", flush=True)
        report["traceback"] = traceback.format_exc()
        _write_report(run_dir, report, rows, timing=timing)
        raise
    finally:
        cleanup_started = time.perf_counter()
        try:
            stopped_pipeline_ids: set[int] = set()
            for pipeline in (rgbd_pipeline, pipeline_runtime.get("rgbd_pipeline")):
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
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        finally:
            timing.record("runtime/cleanup", time.perf_counter() - cleanup_started)
            if not report.get("session_end_reason"):
                report["session_end_reason"] = (
                    "map_repair_committed" if report.get("status") == "map_repair_ready"
                    else "startup_or_runtime_exit"
                )
            report["timing"] = timing.snapshot()
            try:
                _write_progress_checkpoint(run_dir, report, timing=timing)
            except Exception as exc:
                print(f"[PROGRESS_WARNING] 单孔返修进度写入失败：{exc}", flush=True)
            try:
                summary_paths = _write_result_summary(run_dir, report, timing=timing)
                report["result_summary"] = summary_paths
            except Exception as exc:
                print(f"[SUMMARY_WARNING] 单孔返修摘要写入失败：{exc}", flush=True)
            try:
                _write_report(run_dir, report, rows, timing=timing)
            except Exception as exc:
                print(f"[TIMING] 单孔返修最终报告写入失败：{exc}", flush=True)
            timing.print_summary()


def _rotary_sector_map_ready_for_auto_call(args: Any) -> bool:
    """判断指定扇区是否已有可供实时精定位使用的粗地图。"""
    sector_id = getattr(args, "sector_id", None)
    if sector_id is None:
        return False
    try:
        map_path, _ = _resolve_hole_map_path(getattr(args, "hole_map_path", None))
        payload = load_hole_map(map_path)
        if int(payload.get("schema_version", -1)) != 4:
            return False
        sector = (payload.get("sectors") or {}).get(sector_key(validate_sector_id(int(sector_id))))
        if not isinstance(sector, dict):
            return False
        if str(sector.get("status", "")).lower() not in {"valid", "ready", "completed"}:
            return False
        holes = sector.get("holes") or {}
        return sum(
            1 for hole in holes.values()
            if isinstance(hole, dict) and str(hole.get("status", "ready")) in {"ready", "completed"}
        ) >= 1
    except (OSError, ValueError, TypeError):
        return False


def _prepare_rotary_sector_rebuild_args(args: Any) -> Any:
    """为地图缺失时准备纯340 mm粗定位建图参数。"""
    args.hole_map_mode = "build"
    args.batch_coarse_localization = True
    args.batch_fine_localization = False
    args.batch_fine_joint_localization = False
    args.batch_fine_pointcloud_xy_fusion = False
    args.coarse_direct_final = False
    args.coarse_direct_final_capture_only = False
    args.move_final_xy = False
    args.reuse_coarse_cache = False
    args.reuse_persistent_coarse_cache = False
    args.shared_cache_validation = False
    return args


def run_hole_map_execution(args: Any) -> int:
    """读取粗地图并启动一次新的相机/精定位流程。"""
    if getattr(args, "image", None):
        raise RuntimeError("粗定位地图调用必须使用实时 RGB-D/RGB 流，不能使用 --image")
    map_path, _ = _resolve_hole_map_path(getattr(args, "hole_map_path", None))
    payload = load_hole_map(map_path)
    version = int(payload.get("schema_version", -1))
    if version not in {2, 4}:
        raise ValueError(
            f"旧版孔位地图 v{version} 含历史最终点或不受支持，不能直接调用；请重新建立粗定位地图"
        )
    requested_ids = getattr(args, "hole_ids", None)
    sector_id = getattr(args, "sector_id", None)
    if version == 4 and sector_id is None:
        raise ValueError("调用旋转扇区粗定位地图必须指定 --sector-id 1..6")
    hole_ids = validate_hole_map(
        payload,
        sector_id=None if version == 2 else validate_sector_id(int(sector_id))
        if sector_id is not None else None,
        requested_hole_ids=requested_ids,
    )
    if not hole_ids:
        raise ValueError("粗定位地图中没有可执行孔")
    handeye = load_handeye_experiment_result(args.handeye)
    model = load_yolo(args.model)
    environment = payload.get("environment") or {}
    stored_handeye_hash = environment.get("handeye_sha256")
    current_handeye_hash = file_sha256(getattr(args, "handeye", ""))
    if stored_handeye_hash and stored_handeye_hash != current_handeye_hash:
        raise ValueError("粗定位地图对应的手眼文件已变化，请重新建立地图")
    coarse_direct_final = bool(getattr(args, "coarse_direct_final", False))
    correction_model: dict[str, Any] | None = None
    correction_path: Path | None = None
    correction_enabled = bool(
        getattr(args, "hole_map_seed_correction", True)
    ) and not coarse_direct_final
    explicit_correction_path = getattr(args, "hole_map_seed_correction_path", None)
    if correction_enabled:
        correction_path = (
            Path(explicit_correction_path).expanduser().resolve()
            if explicit_correction_path is not None
            else map_path.resolve().parent / DEFAULT_SEED_CORRECTION_FILENAME
        )
        if correction_path.is_file():
            correction_model = load_seed_correction_model(correction_path)
            validate_model_binding(
                correction_model,
                map_payload=payload,
                map_path=map_path,
                sector_id=None if sector_id is None else int(sector_id),
            )
        elif explicit_correction_path is not None:
            raise FileNotFoundError(f"指定的地图种子纠正模型不存在：{correction_path}")
    seeds: list[dict[str, Any]] = []
    selected_sector = int(sector_id) if sector_id is not None else None
    for order, hole_id in enumerate(hole_ids, start=1):
        source = deepcopy(get_hole(payload, int(hole_id), sector_id=selected_sector))
        center = np.asarray(source["coarse_center_base_mm"], dtype=np.float64).reshape(3)
        plane = np.asarray(source["coarse_plane_point_base_mm"], dtype=np.float64).reshape(3)
        normal = np.asarray(source["coarse_normal_toward_camera_base"], dtype=np.float64).reshape(3)
        if not np.isfinite(center).all() or not np.isfinite(plane).all() or not np.isfinite(normal).all():
            raise ValueError(f"粗地图孔 H{int(hole_id):02d} 含非有限几何")
        seeds.append({
            "hole_id": int(hole_id),
            "class_id": int(source.get("class_id") or 0),
            "class_name": source.get("class_name") or "hole",
            "initial_center_base_mm": center.tolist(),
            "initial_plane_point_base_mm": plane.tolist(),
            "initial_plane_normal_base": normal.tolist(),
            "initial_selection_order": order,
            "diameter_estimate_mm": source.get("diameter_estimate_mm"),
            "matched_diameter_mm": source.get("matched_diameter_mm"),
            "coarse_map_source": deepcopy(source),
        })
    args.hole_map_mode = "execute"
    args.auto_select_holes = False
    args.reuse_coarse_cache = False
    args.reuse_persistent_coarse_cache = False
    args.shared_cache_validation = False
    args.batch_coarse_localization = True
    args.batch_fine_localization = not coarse_direct_final
    args.batch_fine_joint_localization = not coarse_direct_final
    args.batch_fine_pointcloud_xy_fusion = not coarse_direct_final
    # 地图调用在成功后必须执行完整最终点运动；点云中心不可用的孔由
    # 逐孔流程记录为失败，不会下发最终目标。第四策略不执行260 mm精拍。
    args.move_final_xy = True
    args._coarse_map_path = str(map_path)
    args._coarse_map_id = payload.get("map_id")
    args._coarse_map_expected_camera_serial = str(environment.get("camera_serial") or "").strip()
    args._coarse_map_seed_correction_model = correction_model
    args._coarse_map_seed_correction_summary = {
        "enabled": bool(correction_enabled),
        "loaded": correction_model is not None,
        "path": None if correction_path is None else str(correction_path),
        "model_status": (
            None if correction_model is None else correction_model.get("status")
        ),
        "reference_run_id": (
            None if correction_model is None else
            (correction_model.get("reference") or {}).get("run_id")
        ),
        "matched_model_holes": (
            0 if correction_model is None else
            int((correction_model.get("matching") or {}).get("matched_count", 0))
        ),
        "requested_holes": len(hole_ids),
        "applied_holes": 0,
        "unmodeled_requested_hole_ids": (
            sorted(
                int(item) for item in hole_ids
                if correction_model is not None
                and f"H{int(item):02d}" not in (correction_model.get("holes") or {})
            )
        ),
        "max_applied_correction_mm": 0.0,
        "application_policy": (
            "defer_until_live_alignment_then_correct_remaining_xy_seeds_and_shared_capture_poses"
        ),
        "static_reference_applied_before_live_alignment": False,
        "holes": [],
    }
    args._coarse_map_seed_results = {
        int(seed["hole_id"]): dict(seed["coarse_map_source"])
        for seed in seeds
    }
    return run_two_stage_hole_localization(
        args,
        handeye,
        model,
        initial_holes_override=seeds,
    )


def _new_two_stage_report(
    args: Any, cfg: TwoStageConfig, run_dir: Path, cycle_index: int,
) -> dict[str, Any]:
    """创建一轮两阶段报告；机器人/相机资源由外层会话共享。"""
    precision_speed_m_s = float(args.speed_m_s)
    precision_acc_m_s2 = float(args.acc_m_s2)
    transit_speed_m_s = float(getattr(args, "transit_speed_m_s", precision_speed_m_s))
    transit_acc_m_s2 = float(getattr(args, "transit_acc_m_s2", precision_acc_m_s2))
    approach_speed_m_s = float(getattr(args, "approach_speed_m_s", 0.12))
    approach_acc_m_s2 = float(getattr(args, "approach_acc_m_s2", 0.35))
    auto_selection_enabled = bool(getattr(args, "auto_select_holes", False))
    auto_config_path = getattr(args, "auto_sector_config", None)
    hole_map_mode = str(getattr(args, "hole_map_mode", "none"))
    map_build_mode = str(
        getattr(args, "map_build_localization_mode", "coarse_only")
    ).strip().lower()
    return {
        "status": "running",
        "session_end_reason": None,
        "result_summary": None,
        "mode": "two_stage_hole_localization",
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cycle_index": int(cycle_index),
        "hole_map_mode": hole_map_mode,
        "map_build_localization_mode": (
            map_build_mode if hole_map_mode == "build" else None
        ),
        "map_hole_selection_mode": (
            str(getattr(args, "map_hole_selection_mode", "manual"))
            if hole_map_mode == "build" else None
        ),
        "localization_strategy": (
            f"coarse_{float(cfg.coarse_height_mm):g}_pointcloud_center_only_edge_first"
            if bool(getattr(args, "coarse_direct_final", False)) else
            "per_hole_coarse_340_then_fine_260_reference_only"
            if hole_map_mode == "build" and map_build_mode == "per_hole" else
            "shared_or_per_hole_fine"
        ),
        "coarse_direct_final_height_mm": (
            float(cfg.coarse_height_mm)
            if bool(getattr(args, "coarse_direct_final", False)) else None
        ),
        "coarse_direct_final_max_group_size": (
            int(cfg.batch_coarse_max_group_size)
            if bool(getattr(args, "coarse_direct_final", False)) else None
        ),
        "coarse_direct_final_settle_delay_s": (
            float(cfg.coarse_direct_final_settle_delay_s)
            if bool(getattr(args, "coarse_direct_final", False)) else None
        ),
        "coarse_direct_final_group_planning_timeout_s": (
            float(cfg.coarse_direct_final_group_planning_timeout_s)
            if bool(getattr(args, "coarse_direct_final", False)) else None
        ),
        "coarse_direct_final_settle_discard_timeout_s": (
            float(cfg.coarse_direct_final_settle_discard_timeout_s)
            if bool(getattr(args, "coarse_direct_final", False)) else None
        ),
        "coarse_direct_final_capture_timeout_s": (
            float(cfg.coarse_direct_final_capture_timeout_s)
            if bool(getattr(args, "coarse_direct_final", False)) else None
        ),
        "coarse_direct_final_steady_timeout_s": (
            float(cfg.coarse_direct_final_steady_timeout_s)
            if bool(getattr(args, "coarse_direct_final", False)) else None
        ),
        "coarse_direct_final_max_consecutive_frame_failures": (
            int(cfg.coarse_direct_final_max_consecutive_frame_failures)
            if bool(getattr(args, "coarse_direct_final", False)) else None
        ),
        "coarse_direct_final_group_size_policy": (
            "edge_first_max3_then_interior_max5_prefer3_to5_allow1_to2"
            if bool(getattr(args, "coarse_direct_final", False)) else None
        ),
        "coarse_direct_final_boundary_policy": (
            "selected_region_outer_layer_first_local_boundary_classification"
            if bool(getattr(args, "coarse_direct_final", False)) else None
        ),
        "coarse_direct_final_edge_first": bool(
            getattr(args, "coarse_direct_final", False)
        ),
        "coarse_direct_final_capture_only": (
            bool(getattr(args, "coarse_direct_final_capture_only", False))
            if bool(getattr(args, "coarse_direct_final", False)) else False
        ),
        "fine_stage_policy": (
            "coarse_direct_pointcloud_capture_only_no_final_motion"
            if bool(getattr(args, "coarse_direct_final", False))
            and bool(getattr(args, "coarse_direct_final_capture_only", False)) else
            "disabled_for_coarse_direct_pointcloud_only"
            if bool(getattr(args, "coarse_direct_final", False)) else
            "per_hole_260mm_fine_reference_only_no_final_motion"
            if hole_map_mode == "build" and map_build_mode == "per_hole" else
            "disabled_for_coarse_map_build"
            if hole_map_mode == "build" else
            "configured_shared_or_per_hole_fine"
        ),
        "fine_stage_skipped": None,
        "sector_id": (
            int(getattr(args, "sector_id"))
            if getattr(args, "sector_id", None) is not None else None
        ),
        "session_reselect_enabled": bool(args.execute),
        "configuration": cfg.__dict__,
        "hole_count": None,
        "selection_mode": (
            "auto_sector_selection" if auto_selection_enabled
            else "click_any_count_then_enter"
        ),
        "auto_sector_selection": {
            "enabled": auto_selection_enabled,
            "config_path": str(auto_config_path) if auto_config_path else None,
            "report_filename": "auto_sector_selection.json",
            "overlay_filename": "01_home_auto_sector_selection.png",
            "sector_info_dir": str(HOLE_LOCALIZATION_SECTOR_INFO_DIR),
            "sector_info_layout": "Sxx/sector_definition.json + Sxx/latest.json + Sxx/runs/<snapshot>/",
            "initial_frame_role": "candidate_range_and_sector_only",
            "coarse_map_generated_from_initial_frame": False,
            "map_hole_selection_mode": (
                str(getattr(args, "map_hole_selection_mode", "manual"))
                if str(getattr(args, "hole_map_mode", "none")) == "build" else None
            ),
        },
        "handeye_path": str(args.handeye),
        "stages": {},
        "motion_executed": bool(args.execute),
        "experimental_handeye_override": bool(args.allow_experimental_handeye),
        "reuse_coarse_cache": bool(getattr(args, "reuse_coarse_cache", True)),
        "reuse_persistent_coarse_cache": bool(
            getattr(args, "reuse_persistent_coarse_cache", True)
        ),
        "motion_profiles": {
            "precision": {
                "speed_m_s": precision_speed_m_s,
                "acc_m_s2": precision_acc_m_s2,
                "used_for": "final_xy_final_z",
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


def _prepare_coarse_map_seed_selection(
    seeds: list[dict[str, Any]],
    *,
    rgbd_pipeline: Any,
    align: Any,
    chain: Any,
    current_tcp: np.ndarray,
    handeye: Any,
    run_dir: Path,
    timing: TimingRecorder,
    cycle_index: int,
) -> tuple[list[dict[str, Any]], np.ndarray, PlaneEstimate, Any]:
    """把地图中的340 mm粗几何投影到当前初始画面，作为本轮选孔种子。

    这里仅使用地图保存的粗中心、粗平面和粗法向。当前帧只提供像素投影和
    RGB内参；真正的260 mm精定位仍由后续共享/逐孔精定位阶段重新采集。
    """
    if not seeds:
        raise ValueError("粗定位地图没有可用孔位种子")
    bundle = None
    for _ in range(20):
        candidate = get_aligned_frame_bundle(rgbd_pipeline, align, chain)
        if candidate is not None and candidate.intrinsics is not None:
            bundle = candidate
            break
    if bundle is None or bundle.intrinsics is None:
        raise RuntimeError("粗定位地图调用无法获得带RGB内参的初始帧")

    intrinsics = bundle.intrinsics
    T_base_camera = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
    R_base_camera = T_base_camera[:3, :3]
    t_base_camera = T_base_camera[:3, 3]
    point_cameras: list[np.ndarray] = []
    plane_points_camera: list[np.ndarray] = []
    normals_camera: list[np.ndarray] = []
    selected: list[dict[str, Any]] = []

    for order, raw_seed in enumerate(seeds, start=1):
        seed = dict(raw_seed)
        hole_id = int(seed["hole_id"])
        point_base = np.asarray(
            seed.get("initial_center_base_mm"), dtype=np.float64,
        ).reshape(3)
        plane_base = np.asarray(
            seed.get("initial_plane_point_base_mm", point_base),
            dtype=np.float64,
        ).reshape(3)
        normal_base = _unit(
            np.asarray(seed.get("initial_plane_normal_base"), dtype=np.float64).reshape(3),
            f"coarse map seed hole {hole_id} normal",
        )
        if not np.isfinite(point_base).all() or not np.isfinite(plane_base).all():
            raise ValueError(f"粗定位地图孔 H{hole_id:02d} 含非有限坐标")
        point_camera = R_base_camera.T @ (point_base - t_base_camera)
        plane_point_camera = R_base_camera.T @ (plane_base - t_base_camera)
        normal_camera = _unit(
            R_base_camera.T @ normal_base,
            f"coarse map seed hole {hole_id} camera normal",
        )
        if float(point_camera[2]) <= 1e-6:
            raise ValueError(f"粗定位地图孔 H{hole_id:02d} 位于当前相机后方")
        try:
            center_px = _project_base_point_to_pixel(
                point_base, T_base_camera, intrinsics,
            )
        except Exception as exc:
            raise ValueError(f"粗定位地图孔 H{hole_id:02d} 无法投影到当前初始画面：{exc}") from exc
        detection = {
            "class_id": int(seed.get("class_id", 0) or 0),
            "class_name": str(seed.get("class_name") or "hole"),
            "confidence": 1.0,
            "center": [float(center_px[0]), float(center_px[1])],
            "box": [
                float(center_px[0] - 25.0), float(center_px[1] - 25.0),
                float(center_px[0] + 25.0), float(center_px[1] + 25.0),
            ],
        }
        seed.update({
            "initial_detection": detection,
            "initial_center_px": detection["center"],
            "initial_box": detection["box"],
            "initial_point_camera_mm": point_camera.tolist(),
            "initial_plane_point_camera_mm": plane_point_camera.tolist(),
            "initial_plane_normal_camera": normal_camera.tolist(),
            "initial_plane_rmse_mm": float(seed.get("coarse_plane_rmse_mm") or 0.0),
            "initial_ring_points": int(seed.get("coarse_ring_points_median") or 0),
            "initial_surface_model": seed.get("coarse_surface_model") or "coarse_map_seed",
            "initial_surface_selection_policy": seed.get(
                "coarse_surface_selection_policy"
            ) or COARSE_SURFACE_SELECTION_POLICY,
            "initial_front_surface_z_mm": seed.get("coarse_front_surface_z_mm"),
            "initial_geometry_fallback": False,
            "initial_geometry_fallback_reason": None,
            "pointcloud_segmentation": "coarse_map_seed",
            "initial_selection_order": order,
            "tracking_identity": f"coarse_map_hole_{hole_id}",
            "coarse_map_seed": True,
        })
        point_cameras.append(point_camera)
        plane_points_camera.append(plane_point_camera)
        normals_camera.append(normal_camera)
        selected.append(seed)

    selected_point_camera = np.mean(np.stack(point_cameras), axis=0)
    shared_plane_point_camera = np.mean(np.stack(plane_points_camera), axis=0)
    shared_normal_camera = _unit(
        np.mean(np.stack(normals_camera), axis=0),
        "coarse map seed shared camera normal",
    )
    rmse_values = [
        float(seed.get("initial_plane_rmse_mm") or 0.0) for seed in selected
    ]
    ring_values = [
        int(seed.get("initial_ring_points") or 0) for seed in selected
    ]
    surface_models = [
        str(seed.get("initial_surface_model") or "coarse_map_seed")
        for seed in selected
    ]
    surface_policies = [
        str(seed.get("initial_surface_selection_policy") or COARSE_SURFACE_SELECTION_POLICY)
        for seed in selected
    ]
    front_z_values = [
        float(seed["initial_front_surface_z_mm"])
        for seed in selected
        if seed.get("initial_front_surface_z_mm") is not None
    ]
    initial_plane = PlaneEstimate(
        point_camera_mm=shared_plane_point_camera,
        normal_camera=shared_normal_camera,
        rmse_mm=float(np.median(rmse_values)) if rmse_values else 0.0,
        ring_points=int(np.median(ring_values)) if ring_values else 0,
        surface_model=surface_models[0] if len(set(surface_models)) == 1 else "coarse_map_seed",
        surface_selection_policy=(
            surface_policies[0]
            if len(set(surface_policies)) == 1
            else "coarse_map_seed"
        ),
        front_surface_z_mm=(float(np.median(front_z_values)) if front_z_values else None),
    )
    frame_path = run_dir / "01_coarse_map_seed.png"
    try:
        if not cv2.imwrite(str(frame_path), bundle.color_bgr):
            frame_path = None
    except Exception:
        frame_path = None
    for seed in selected:
        seed["coarse_map_seed_frame"] = None if frame_path is None else str(frame_path)
    timing.mark(
        "initial_selection/coarse_map_seed",
        cycle_index=int(cycle_index),
        hole_count=len(selected),
        frame_path=None if frame_path is None else str(frame_path),
    )
    return selected, selected_point_camera, initial_plane, intrinsics


def _run_two_stage_localization_cycle(
    args: Any, handeye: Any, model: Any, cfg: TwoStageConfig, run_dir: Path,
    report: dict[str, Any], timing: TimingRecorder, rows: list[dict[str, Any]],
    pipeline_runtime: dict[str, Any], pose_session: Any, motion_session: Any,
    current_tcp: np.ndarray, cycle_index: int,
    initial_holes_override: list[dict[str, Any]] | None = None,
) -> int:
    """执行一轮初始选孔到逐孔完成；相机和机器人会话由外层复用。

    ``initial_holes_override`` 用于粗地图调用/单孔返修：它把地图中的粗定位
    几何投影到当前初始画面作为导航种子，后续精定位仍然重新采集。
    """
    rgbd_pipeline = pipeline_runtime.get("rgbd_pipeline")
    align = pipeline_runtime.get("align")
    chain = pipeline_runtime.get("chain")
    if rgbd_pipeline is None or align is None or chain is None:
        raise RuntimeError("RGB-D 管线尚未就绪，无法开始新一轮初始选孔")

    timing.mark("cycle/task_start", cycle_index=int(cycle_index))
    timing.mark("cycle/initial_selection_start", cycle_index=int(cycle_index))

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

    if initial_holes_override is None:
        auto_sector_config = getattr(args, "_auto_sector_config", None)
        map_selection_mode = (
            getattr(args, "_map_hole_selection_mode", None)
            if str(getattr(args, "hole_map_mode", "none")) == "build"
            else None
        )
        selection_mode = (
            "auto_sector_selection"
            if auto_sector_config is not None and map_selection_mode != "manual"
            else "manual_sector_selection"
            if auto_sector_config is not None and map_selection_mode == "manual"
            else "click_any_count_then_enter"
        )
        print(
            f"[INITIAL_SELECTION_REQUIRED] cycle={int(cycle_index)} "
            + (
                "机器人已在原点并完全停止，自动按静态扇区配置筛选初始候选。"
                if auto_sector_config is not None else
                "机器人已在原点并完全停止，请选择本轮目标孔后按 Enter；按 Esc 结束会话。"
            ),
            flush=True,
        )
        with timing.measure(
            "initial_selection/yolo_rgbd_pointcloud",
            selection_mode=selection_mode,
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
                timing=timing,
                auto_sector_config=auto_sector_config,
                map_hole_selection_mode=map_selection_mode,
            )
    else:
        with timing.measure(
            "initial_selection/coarse_map_seed_frame",
            selection_mode="coarse_map_seed",
            cycle_index=int(cycle_index),
        ):
            (
                initial_selected_holes,
                selected_point_camera,
                initial_plane_camera,
                intrinsics,
            ) = _prepare_coarse_map_seed_selection(
                initial_holes_override,
                rgbd_pipeline=rgbd_pipeline,
                align=align,
                chain=chain,
                current_tcp=current_tcp,
                handeye=handeye,
                run_dir=run_dir,
                timing=timing,
                cycle_index=cycle_index,
            )
        print(
            f"[COARSE_MAP_SEED] cycle={int(cycle_index)} "
            f"使用粗地图投影的{len(initial_selected_holes)}个孔作为导航种子；"
            + (
                f"第四策略将重新采集{float(cfg.coarse_height_mm):g} mm点云中心，"
                "点云不可用时记录失败。"
                if bool(getattr(args, "coarse_direct_final", False)) else
                "后续重新执行当前相机的260 mm精定位。"
            ),
            flush=True,
        )
    timing.mark("cycle/selection_confirmed", cycle_index=int(cycle_index))
    timing.mark("cycle/automatic_execution_start", cycle_index=int(cycle_index))
    report["hole_count"] = len(initial_selected_holes)
    report["selection_mode"] = (
        "coarse_map_seed" if initial_holes_override is not None
        else (
            "auto_sector_selection"
            if getattr(args, "_auto_sector_config", None) is not None
            and not (
                str(getattr(args, "hole_map_mode", "none")) == "build"
                and getattr(args, "_map_hole_selection_mode", None) == "manual"
            )
            else "manual_sector_selection"
            if getattr(args, "_auto_sector_config", None) is not None
            and str(getattr(args, "hole_map_mode", "none")) == "build"
            and getattr(args, "_map_hole_selection_mode", None) == "manual"
            else "click_any_count_then_enter"
        )
    )
    if str(getattr(args, "hole_map_mode", "none")) == "build":
        row_groups: dict[int, int] = {}
        row_tolerances = {
            float(hole["numbering_row_tolerance_px"])
            for hole in initial_selected_holes
            if hole.get("numbering_row_tolerance_px") is not None
        }
        for hole in initial_selected_holes:
            row = hole.get("layout_row")
            if row is not None:
                row_groups[int(row)] = row_groups.get(int(row), 0) + 1
        report["map_numbering"] = {
            "policy": "image_row_major_v1",
            "selection_mode": str(getattr(args, "_map_hole_selection_mode", "manual")),
            "axis": "image_y_then_image_x",
            "hole_count": int(len(initial_selected_holes)),
            "row_count": int(len(row_groups)),
            "row_sizes": [int(row_groups[key]) for key in sorted(row_groups)],
            "row_tolerance_px": (
                float(sorted(row_tolerances)[0]) if row_tolerances else None
            ),
            "image_size_px": [
                int(getattr(intrinsics, "width", 0) or 0),
                int(getattr(intrinsics, "height", 0) or 0),
            ],
            "preview_path": str(run_dir / "01_home_selected.png"),
            "confirmation_required": True,
        }
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
        "selection_mode": (
            "coarse_map_seed" if initial_holes_override is not None
            else (
                "manual_sector_selection"
                if (
                    getattr(args, "_auto_sector_config", None) is not None
                    and str(getattr(args, "hole_map_mode", "none")) == "build"
                    and getattr(args, "_map_hole_selection_mode", None) == "manual"
                ) else "auto_sector_selection"
                if getattr(args, "_auto_sector_config", None) is not None
                else "initial_multi_hole_selection"
            )
        ),
        "map_hole_selection_mode": (
            str(getattr(args, "_map_hole_selection_mode", "manual"))
            if str(getattr(args, "hole_map_mode", "none")) == "build" else None
        ),
        "hole_count": len(initial_selected_holes),
        # 保留建图时的共同视野位姿用于地图记录和旧地图格式兼容；调用
        # 地图时不再回到该位姿做现场视觉验证或整体位姿纠偏。
        "capture_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
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
    auto_report_path = run_dir / "auto_sector_selection.json"
    if initial_holes_override is None and auto_report_path.is_file():
        try:
            report["stages"]["home_selection"]["auto_sector_selection"] = json.loads(
                auto_report_path.read_text(encoding="utf-8")
            )
            if str(getattr(args, "hole_map_mode", "none")) == "build":
                auto_info = report["stages"]["home_selection"]["auto_sector_selection"]
                if isinstance(auto_info, dict):
                    report["map_numbering"].update({
                        "candidate_count": int(auto_info.get("candidate_count", 0) or 0),
                        "auto_selected_count": int(
                            len(auto_info.get("auto_selected_detection_indices") or
                                auto_info.get("selected_detection_indices") or [])
                        ),
                        "excluded_count": int(
                            max(
                                0,
                                int(auto_info.get("candidate_count", 0) or 0)
                                - int(len(auto_info.get("operator_selected_detection_indices") or
                                        auto_info.get("selected_detection_indices") or [])),
                            )
                        ),
                        "excluded_reasons": list(
                            auto_info.get("rejected") or []
                        ) + [
                            {
                                "detection_index": int(item.get("detection_index")),
                                "reason": item.get("selection_reason"),
                            }
                            for item in (auto_info.get("candidates") or [])
                            if isinstance(item, dict)
                            and (
                                item.get("selection_reason") == "operator_deselected"
                                or str(item.get("selection_reason") or "").startswith(
                                    "duplicate_of_detection_"
                                )
                            )
                        ],
                    })
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            report["stages"]["home_selection"]["auto_sector_selection_read_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

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
    _write_report(run_dir, report, rows, timing=timing)

    result = _run_sequential_hole_workflow(
        args, handeye, model, cfg, run_dir, report, timing, rows, pipeline_runtime,
        pose_session, motion_session, current_tcp,
        initial_selected_holes, intrinsics,
        coarse_cache_entries=cache_entries,
        coarse_cache_sources=cache_sources,
        coarse_cache_source_ids=cache_source_ids,
        coarse_cache_dir=coarse_cache_dir,
        coarse_cache_gates=coarse_cache_gates,
        persistent_cache_entries=persistent_entries,
        persistent_cache_dir=persistent_cache_dir,
    )
    if str(getattr(args, "hole_map_mode", "none")) == "build":
        # 预览模式只生成选孔/分组计划；只有实际完成本模式要求的定位后才写地图。
        if not report.get("final_result"):
            report["hole_map"] = {
                "status": "not_written_preview_only",
                "reason": "没有执行建图定位",
            }
            _write_report(run_dir, report, rows, timing=timing)
            return int(result)
        _write_hole_map_from_report(args, report, run_dir, timing=timing)
        # 资源字典名是 pipeline_runtime；这里仅记录路径供本轮报告/外层调试，
        # 不应因为地图已经保存成功而引用不存在的 runtime 局部变量。
        pipeline_runtime["hole_map_path"] = report["hole_map"]["path"]
    return int(result)


def _return_home_after_coarse_map_build(
    *,
    home: Any,
    motion_session: Any,
    pose_session: Any,
    timing: TimingRecorder,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    run_dir: Path,
    pipeline_runtime: dict[str, Any],
) -> np.ndarray:
    """地图保存成功后自动回控制器原始点并等待机械臂稳定。"""
    controller_home_joints, home_source = _resolve_home_joints(home, motion_session)
    report["home_point_source"] = home_source
    report["controller_home_joints_rad"] = controller_home_joints.tolist()
    report["return_to_home"] = {
        "completed": False,
        "reason": "hole_map_saved_returning_home",
    }
    _write_report(run_dir, report, rows, timing=timing)
    print(
        f"[HOLE_MAP_BUILD] 地图已保存，自动返回{home_source}原始点；"
        f"关节目标(rad)={controller_home_joints.tolist()}",
        flush=True,
    )
    with timing.measure("robot/return_home_after_hole_map_build"):
        current_tcp = _confirm_and_move_home(
            home,
            motion_session,
            pose_session,
            timing=timing,
            target_joints=controller_home_joints,
            home_source=home_source,
        )
    with timing.measure("robot/verify_home_steady_after_hole_map_build"):
        _, current_tcp = _wait_robot_steady(pose_session)
    current_tcp = np.asarray(current_tcp, dtype=np.float64).copy()
    pipeline_runtime["current_tcp"] = current_tcp.copy()
    report["return_to_home"] = {
        "completed": True,
        "reason": "hole_map_saved",
        "tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
    }
    report["final_tcp_pose_m_rad"] = transform_to_sdk_pose_m_rad(current_tcp)
    # 保留旧报告枚举值，避免下游工具按历史建图结束状态解析失败。
    report["session_end_reason"] = "coarse_map_build_completed_returned_home"
    _write_report(run_dir, report, rows, timing=timing)
    print("[HOLE_MAP_BUILD_DONE] 孔位地图已保存，机械臂已回原点并完全停止。", flush=True)
    return current_tcp




@camera_stream_session
def run_two_stage_hole_localization(
    args: Any,
    handeye: Any,
    model: Any,
    *,
    initial_holes_override: list[dict[str, Any]] | None = None,
) -> int:
    """持续运行旧两阶段流程：每轮完成后回原点并重新进入初始选孔。"""
    auto_sector_config = None
    if bool(getattr(args, "auto_select_holes", False)):
        config_path = getattr(args, "auto_sector_config", None)
        if config_path is None:
            raise ValueError(
                "启用自动扇区选孔必须同时指定 --auto-sector-config；"
                "请先在固定观察位用鼠标画区并保存配置"
            )
        auto_sector_config = load_auto_sector_config(
            config_path,
            active_sector_ids=getattr(args, "auto_sector_ids", None),
            include_boundary_candidates=(
                False if bool(getattr(args, "auto_exclude_boundary_candidates", False))
                else None
            ),
        )
    setattr(args, "_auto_sector_config", auto_sector_config)
    cfg = TwoStageConfig.from_namespace(args)
    hole_map_mode = str(getattr(args, "hole_map_mode", "none"))
    raw_map_hole_selection_mode = getattr(args, "map_hole_selection_mode", "manual")
    map_hole_selection_mode = str(
        "manual" if raw_map_hole_selection_mode is None else raw_map_hole_selection_mode
    ).strip().lower()
    if map_hole_selection_mode not in {"auto", "manual"}:
        raise ValueError("地图建图选孔模式必须是auto或manual")
    if hole_map_mode == "build" and map_hole_selection_mode == "auto" and auto_sector_config is None:
        raise ValueError(
            "地图建图选择auto模式时必须启用--auto-select-holes并提供--auto-sector-config"
        )
    setattr(args, "_map_hole_selection_mode", map_hole_selection_mode)
    coarse_direct_final = bool(getattr(args, "coarse_direct_final", False))
    if auto_sector_config is not None and hole_map_mode == "build":
        requested_sector = getattr(args, "sector_id", None)
        if requested_sector is not None:
            requested_sector = validate_sector_id(int(requested_sector))
            configured_sectors = set(auto_sector_config.active_sector_ids or ())
            if configured_sectors and configured_sectors != {requested_sector}:
                raise ValueError(
                    "旋转扇区建图时，自动分区活动扇区必须只包含 --sector-id "
                    f"{requested_sector}；否则检测到的孔会被错误写入同一扇区"
                )
            auto_sector_config = load_auto_sector_config(
                getattr(args, "auto_sector_config"),
                active_sector_ids=[requested_sector],
                include_boundary_candidates=(
                    False if bool(getattr(args, "auto_exclude_boundary_candidates", False))
                    else None
                ),
            )
            setattr(args, "_auto_sector_config", auto_sector_config)
    if hole_map_mode == "build":
        map_build_localization_mode = str(
            getattr(args, "map_build_localization_mode", "coarse_only")
            or "coarse_only"
        ).strip().lower()
        if map_build_localization_mode not in {"coarse_only", "per_hole"}:
            raise ValueError(
                "地图建图定位方式必须是coarse_only或per_hole"
            )
        args.map_build_localization_mode = map_build_localization_mode
        args._map_build_localization_mode = map_build_localization_mode

        # 逐孔模式关闭共享粗/精定位，让顺序工作流对每个孔依次执行
        # 340 mm粗定位和260 mm精定位；粗定位模式保留原有的共享340 mm建图。
        batch_coarse_for_map = map_build_localization_mode == "coarse_only"
        args.batch_coarse_localization = batch_coarse_for_map
        args.batch_fine_localization = False
        args.batch_fine_joint_localization = False
        args.batch_fine_pointcloud_xy_fusion = False
        cfg = replace(
            cfg,
            batch_coarse_localization=batch_coarse_for_map,
            coarse_direct_final=False,
            coarse_direct_final_capture_only=False,
            batch_fine_localization=False,
            batch_fine_joint_localization=False,
            batch_fine_pointcloud_xy_fusion=False,
            shared_cache_validation=False,
        )
        args.map_build_coarse_only = batch_coarse_for_map
        args.map_build_per_hole_reference = not batch_coarse_for_map
        # 两种建图方式都不执行最终安放动作；逐孔模式的260 mm结果只作为
        # 每孔精定位参考写入地图。
        args.coarse_direct_final = False
        args.coarse_direct_final_capture_only = False
        coarse_direct_final = False
        if bool(getattr(args, "move_final_xy", False)):
            raise ValueError(
                "建立孔位地图阶段不执行最终安放动作，请关闭 --move-final-xy"
            )
        # 地图只作为后续实时精定位的导航基准，必须来自本轮新鲜定位；
        # 不允许用历史粗缓存节省时间而把旧工件坐标写进新地图。
        args.reuse_coarse_cache = False
        args.reuse_persistent_coarse_cache = False
        args.shared_cache_validation = False
    elif hole_map_mode == "execute":
        # 地图调用默认执行现场精定位；第四策略重新按独立高度采集点云中心。
        args.map_build_coarse_only = False
        args.move_final_xy = True
        if coarse_direct_final:
            args.batch_fine_localization = False
            args.batch_fine_joint_localization = False
            args.batch_fine_pointcloud_xy_fusion = False
            args.shared_cache_validation = False
            cfg = replace(
                cfg,
                coarse_direct_final=True,
                batch_fine_localization=False,
                batch_fine_joint_localization=False,
                batch_fine_pointcloud_xy_fusion=False,
                batch_coarse_group_pose_refinement=False,
                shared_cache_validation=False,
            )
        else:
            args.batch_fine_localization = True
            cfg = replace(cfg, batch_fine_localization=True)
        if initial_holes_override is None:
            raise ValueError("粗定位地图调用缺少地图孔位种子")
    elif coarse_direct_final:
        # 第四策略强制走独立高度点云采集；单孔也使用同一批量采集函数。
        args.batch_coarse_localization = True
        args.batch_fine_localization = False
        args.batch_fine_joint_localization = False
        args.batch_fine_pointcloud_xy_fusion = False
        args.shared_cache_validation = False
        args.reuse_coarse_cache = False
        args.reuse_persistent_coarse_cache = False
        cfg = replace(
            cfg,
            coarse_direct_final=True,
            batch_coarse_localization=True,
            batch_fine_localization=False,
            batch_fine_joint_localization=False,
            batch_fine_pointcloud_xy_fusion=False,
            batch_coarse_group_pose_refinement=False,
            shared_cache_validation=False,
        )
    # 第四策略的高度、分组上限和采集提前结束参数独立于普通两阶段粗定位。
    # 将其折叠进本轮有效配置，确保分组规划、点云采集和最终点规划使用同一组值。
    if coarse_direct_final:
        direct_height_mm = float(cfg.coarse_direct_final_height_mm)
        direct_max_group_size = int(cfg.coarse_direct_final_max_group_size)
        direct_extra_frames = int(cfg.coarse_direct_final_early_stop_extra_frames)
        cfg = replace(
            cfg,
            coarse_height_mm=direct_height_mm,
            batch_coarse_max_group_size=direct_max_group_size,
            batch_coarse_early_stop_extra_frames=direct_extra_frames,
        )
        setattr(args, "_coarse_direct_final_height_mm", direct_height_mm)
        setattr(args, "_coarse_direct_final_max_group_size", direct_max_group_size)
        setattr(args, "_coarse_direct_final_capture_only", bool(
            getattr(args, "coarse_direct_final_capture_only", False)
        ))
        if bool(getattr(args, "coarse_direct_final_capture_only", False)):
            # 仅采集评估必须保留观察位运动，但严禁进入最终XY/Z动作。
            args.move_final_xy = False
    else:
        setattr(args, "_coarse_direct_final_height_mm", None)
        setattr(args, "_coarse_direct_final_max_group_size", None)
        setattr(args, "_coarse_direct_final_capture_only", False)
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
    cfg.validate(
        reuse_coarse_cache=bool(getattr(args, "reuse_coarse_cache", True)),
    )
    if (
        coarse_direct_final
        and getattr(args, "tcp_xy_offset_mm", None) is not None
    ):
        raise ValueError(
            "第四策略只允许使用ChArUco XY纠偏，不能同时指定TCP XY固定补偿"
        )
    if (
        bool(getattr(args, "execute", False))
        and bool(getattr(args, "move_final_xy", False))
        and getattr(args, "tcp_xy_offset_mm", None) is None
        and not CHARUCO_XY_MODEL_READY
    ):
        raise RuntimeError(
            "TCP-XY 补偿模型尚未完成或无效，拒绝执行最终 XY 微调；"
            "请先重新采集并复核 current.json，或仅在受控实验中显式提供 --tcp-xy-offset-mm"
        )
    cycle_index = 1
    run_dir = _new_two_stage_run_dir(cycle_index)
    report = _new_two_stage_report(args, cfg, run_dir, cycle_index)
    timing = TimingRecorder()
    timing.attach_report(report)
    setattr(args, "_timing_recorder", timing)
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
        from aubo_workbench.motion_control import AuboMotionSession, HOME_POINT_FILE, load_home_point
        from aubo_workbench.robot import AuboPoseSession

        with timing.measure("robot/load_home_point"):
            home = load_home_point()
        if home is None:
            print(
                f"[HOME] 软件备用原始点不存在或不可读：{HOME_POINT_FILE}；"
                "真实运动时将优先读取控制器原始点，预览模式不需要原始点文件。",
                flush=True,
            )
        else:
            print(
                f"[HOME] 已加载软件备用原始点：{HOME_POINT_FILE}；"
                f"created_at={home.created_at}；关节目标(rad)={home.joints_rad}",
                flush=True,
            )
        with timing.measure("robot/connect_pose_session"):
            pose_session = AuboPoseSession()
            pose_session.connect()
            initial_snapshot, initial_tcp = _require_safe_snapshot(pose_session)
        current_tcp = np.asarray(initial_tcp, dtype=np.float64).copy()
        report["robot_initial_tcp_pose_m_rad"] = initial_snapshot["pose_values_sdk_m_rad"]
        report["home_point"] = home.to_dict() if home is not None else None

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
            controller_home_joints, home_source = _resolve_home_joints(home, motion_session)
            report["home_point_source"] = home_source
            report["controller_home_joints_rad"] = controller_home_joints.tolist()
            print(
                f"[HOME] 实际运动目标来源={home_source}；"
                f"关节目标(rad)={controller_home_joints.tolist()}",
                flush=True,
            )
            with timing.measure("robot/move_home"):
                current_tcp = _confirm_and_move_home(
                    home,
                    motion_session,
                    pose_session,
                    timing=timing,
                    target_joints=controller_home_joints,
                    home_source=home_source,
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
            map_expected = str(
                getattr(args, "_coarse_map_expected_camera_serial", "") or ""
            ).strip()
            if map_expected and actual and map_expected != actual:
                raise RuntimeError(
                    f"粗定位地图相机序列号不匹配：地图={map_expected}，当前={actual}"
                )
            report["camera"] = camera_identity
            if hole_map_mode == "execute":
                report["hole_map_execution"] = {
                    "map_path": str(getattr(args, "_coarse_map_path", "")),
                    "map_id": getattr(args, "_coarse_map_id", None),
                    "requires_fresh_fine": not bool(
                        getattr(args, "coarse_direct_final", False)
                    ),
                    "fine_capture_policy": (
                        f"disabled_coarse_{float(cfg.coarse_height_mm):g}_pointcloud_center_only"
                        if bool(getattr(args, "coarse_direct_final", False)) else
                        "always"
                    ),
                    "coarse_seed_source": "coarse_hole_map",
                    "seed_correction": deepcopy(getattr(
                        args, "_coarse_map_seed_correction_summary", None,
                    )),
                    "fine_result_persisted": False,
                    "localization_strategy": (
                        f"coarse_{float(cfg.coarse_height_mm):g}_pointcloud_center_only"
                        if bool(getattr(args, "coarse_direct_final", False)) else
                        "fresh_fine_from_coarse_map"
                    ),
                }
        report["cycle_start_tcp_pose_m_rad"] = transform_to_sdk_pose_m_rad(current_tcp)
        _write_report(run_dir, report, rows, timing=timing)

        while True:
            if cycle_index > 1:
                run_dir = _new_two_stage_run_dir(cycle_index)
                report = _new_two_stage_report(args, cfg, run_dir, cycle_index)
                timing = TimingRecorder()
                timing.attach_report(report)
                setattr(args, "_timing_recorder", timing)
                rows = []
                report["robot_initial_tcp_pose_m_rad"] = (
                    transform_to_sdk_pose_m_rad(current_tcp)
                )
                report["home_point"] = home.to_dict() if home is not None else None
                report["camera"] = camera_identity
                report["stages"]["rgbd_startup"] = {
                    "status": "shared_ready",
                    "reused_session": True,
                }
                report["cycle_start_tcp_pose_m_rad"] = (
                    transform_to_sdk_pose_m_rad(current_tcp)
                )
                _write_report(run_dir, report, rows, timing=timing)

            cycle_override = initial_holes_override if cycle_index == 1 else None
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
                initial_holes_override=cycle_override,
            )
            task_map_mode = str(getattr(args, "hole_map_mode", "none"))
            if task_map_mode == "build":
                # 地图已在本轮内部保存。真实建图结束后自动回原点，等待完全
                # 稳定再退出；不得进入精定位或旧的重复选孔循环。
                if args.execute:
                    current_tcp = _return_home_after_coarse_map_build(
                        home=home,
                        motion_session=motion_session,
                        pose_session=pose_session,
                        timing=timing,
                        report=report,
                        rows=rows,
                        run_dir=run_dir,
                        pipeline_runtime=pipeline_runtime,
                    )
                else:
                    report["session_end_reason"] = "coarse_map_build_preview_complete"
                    _write_report(run_dir, report, rows, timing=timing)
                return int(result)
            if task_map_mode in {"execute", "repair"}:
                # 地图调用/返修都是单轮任务；不能回到旧的重复选孔循环。
                return int(result)
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
            _write_report(run_dir, report, rows, timing=timing)
            print(
                f"[NEXT_CYCLE_CONFIRM_REQUIRED] cycle={int(cycle_index)} 已完成；"
                "机器人保持当前位置，不会自动回到初始点。"
                "点击“开始下一轮检测”（命令行输入 m）后才回到初始点；"
                "回到初始点并完全停止后才会拍摄下一轮画面。",
                flush=True,
            )
            command = _request_next_cycle_confirmation(cycle_index, timing=timing)
            if command != "m":
                raise TwoStageSelectionCancelled("用户未确认开始下一轮检测")
            report["next_cycle_confirmation_required"] = False
            report["next_cycle_confirmed"] = True
            report["next_cycle_confirmed_at"] = datetime.now().isoformat(timespec="seconds")
            report["next_cycle_ready"] = False
            report["next_cycle_motion"] = "return_home_after_confirmation"
            controller_home_joints, home_source = _resolve_home_joints(home, motion_session)
            report["home_point_source"] = home_source
            report["controller_home_joints_rad"] = controller_home_joints.tolist()
            print(
                f"[HOME] 实际运动目标来源={home_source}；"
                f"关节目标(rad)={controller_home_joints.tolist()}",
                flush=True,
            )
            _write_report(run_dir, report, rows, timing=timing)
            with timing.measure(
                "robot/return_home_after_cycle_confirmation",
                cycle_index=int(cycle_index),
            ):
                current_tcp = _confirm_and_move_home(
                    home,
                    motion_session,
                    pose_session,
                    timing=timing,
                    target_joints=controller_home_joints,
                    home_source=home_source,
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
            _write_report(run_dir, report, rows, timing=timing)
            cycle_index += 1

    except TwoStageSelectionCancelled as exc:
        report["status"] = "stopped_by_user"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["session_end_reason"] = (
            "user_exit_after_cycle"
            if (report.get("final_result") or {}).get("completed_count") is not None
            else "user_cancelled"
        )
        report["next_cycle_ready"] = False
        report["session_stopped_by_user"] = True
        _write_report(run_dir, report, rows, timing=timing)
        print(f"[STOPPED] 用户结束重复选孔会话：{exc}", flush=True)
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["session_end_reason"] = "error"
        if isinstance(exc, LocalizationHardwareError):
            report["failure_type"] = exc.failure_type
            report["session_end_reason"] = exc.failure_type
            print(f"[HARDWARE_FAILED] {exc}", flush=True)
        report["traceback"] = traceback.format_exc()
        report["timing"] = timing.snapshot()
        _write_report(run_dir, report, rows, timing=timing)
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
            if not report.get("session_end_reason"):
                if report.get("status") == "preview_complete":
                    report["session_end_reason"] = "preview_complete"
                elif report.get("final_result"):
                    report["session_end_reason"] = "completed"
                else:
                    report["session_end_reason"] = "startup_or_runtime_exit"
            report["timing"] = timing.snapshot()
            try:
                _write_progress_checkpoint(run_dir, report, timing=timing)
            except Exception as exc:
                print(
                    f"[PROGRESS_WARNING] 最终进度文件写入失败：{type(exc).__name__}: {exc}",
                    flush=True,
                )
            try:
                summary_paths = _write_result_summary(
                    run_dir, report, timing=timing,
                )
                report["result_summary"] = summary_paths
                print(
                    f"[SUMMARY] 简明结果：{summary_paths.get('text')}；"
                    f"JSON：{summary_paths.get('json')}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[SUMMARY_WARNING] 简明结果报告写入失败：{type(exc).__name__}: {exc}",
                    flush=True,
                )
            report["timing"] = timing.snapshot()
            try:
                _write_report(run_dir, report, rows, timing=timing)
            except Exception as exc:
                print(
                    f"[TIMING] 最终报告写入失败：{type(exc).__name__}: {exc}",
                    flush=True,
                )
            timing.print_summary()


from aubo_workbench.hole_localization_cli import build_parser as _build_parser  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    return _build_parser(
        default_model=DEFAULT_MODEL,
        default_handeye=DEFAULT_HANDEYE,
        default_execute_motion=DEFAULT_EXECUTE_MOTION,
        default_allow_experimental_handeye=DEFAULT_ALLOW_EXPERIMENTAL_HANDEYE,
        default_two_stage_hole_localization=DEFAULT_TWO_STAGE_HOLE_LOCALIZATION,
        default_move_final_xy=DEFAULT_MOVE_FINAL_XY,
    )


# 实现已统一到 aubo_workbench.config.apply_robot_connection_overrides。
_apply_robot_connection_overrides = apply_robot_connection_overrides


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _apply_robot_connection_overrides(args)
    if args.hole_map_mode == "repair":
        handeye = load_handeye_experiment_result(args.handeye)
        model = load_yolo(args.model)
        if args.image:
            raise RuntimeError("地图单孔返修必须使用实时 Gemini RGB-D/RGB 流，不能使用 --image")
        return run_hole_map_repair(args, handeye, model)
    if args.hole_map_mode == "execute":
        # 当前扇区没有地图时进入首轮340 mm粗定位建图；已有地图则把
        # 粗几何作为种子，并启动当前相机重新执行260 mm精定位。
        if getattr(args, "sector_id", None) is not None and not _rotary_sector_map_ready_for_auto_call(args):
            print(
                f"[ROTARY_SECTOR_MAP_MISSING] sector={int(args.sector_id)} "
                "当前扇区无可用地图，进入首轮共享粗/精定位建图。",
                flush=True,
            )
            _prepare_rotary_sector_rebuild_args(args)
            handeye = load_handeye_experiment_result(args.handeye)
            model = load_yolo(args.model)
            if args.image:
                raise RuntimeError("六扇区首轮建图必须使用实时 Gemini RGB-D/RGB 流，不能使用 --image")
            return run_two_stage_hole_localization(args, handeye, model)
        return run_hole_map_execution(args)
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
