#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO + RGB-D + RGB 手眼标定的“眼在手上”孔中心定位脚本。

流程：在初始 RGB-D 画面中选择目标孔 -> 340 mm 粗定位建立局部点云、平面和法向
-> 260 mm RGB/YOLO 精定位修正最终 XY -> 移动到当前目标点。

默认进入两阶段流程但只做预览；必须显式启用运动和相应的验证开关后，才会连接运动控制。
默认不允许使用未通过生产验证的实验手眼结果。

本文件保留命令行入口和兼容导出；视觉、采集、共享定位、分组运动、缓存、
地图辅助、诊断和参数解析分别由 aubo_workbench 下的专用模块实现。
"""

from __future__ import annotations

import argparse
from functools import wraps
import json
import math
import sys
import time
import traceback
from dataclasses import replace
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
from aubo_workbench.io_utils import atomic_write_json, jsonable, write_dict_rows  # noqa: E402
from aubo_workbench.hole_map import (  # noqa: E402
    build_hole_map_payload,
    file_sha256,
    get_hole,
    load_hole_map,
    publish_current_hole_map,
    resolve_hole_map_path,
    save_hole_map,
    validate_hole_map,
)
from aubo_workbench.hole_map_visualization import (  # noqa: E402
    export_hole_map_artifacts,
)
from aubo_workbench.batch_grouping import (  # noqa: E402
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
    HANDEYE_CANDIDATE_PATH,
    HOLE_LOCALIZATION_COARSE_CACHE_DIR,
    HOLE_LOCALIZATION_MAPS_DIR,
    MODEL_PATH,
    HOLE_LOCALIZATION_RUNS_DIR,
)


DEFAULT_MODEL = MODEL_PATH
DEFAULT_HANDEYE = HANDEYE_CANDIDATE_PATH
RUNS_DIR = HOLE_LOCALIZATION_RUNS_DIR
from aubo_workbench.hole_localization_planning import (  # noqa: E402
    CHARUCO_XY_MODEL_BIAS_MM,
    CHARUCO_XY_MODEL_MATRIX,
    CHARUCO_XY_MODEL_SOURCE,
    DEFAULT_FINAL_TARGET_MODE,
    FINAL_BASE_Y_AFTER_Z_MM,
    FINAL_TARGET_MODE_GRIPPER,
    FINAL_TARGET_MODE_NORMAL,
    FINAL_TOOL_Y_AFTER_Z_MM,
    THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM,
    _compose_batch_fine_joint_xy_with_tilt,
    apply_final_point_base_offsets,
    compose_batch_fine_xy_with_coarse_z,
    final_point_offsets_for_mode,
    plan_final_tcp_base_y_trim,
    plan_final_tcp_base_z,
    plan_final_tcp_combined_y_trim,
    plan_final_tcp_xy,
)
SHARED_OBSERVATION_MIN_LIFT_MM = 10.0
SHARED_OBSERVATION_MIN_DESCENT_MM = 10.0
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
DEFAULT_ALLOW_EXPERIMENTAL_HANDEYE = False
DEFAULT_MOVE_FINAL_XY = False


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


def _confirm_and_move_home(home: Any, motion_session: Any, pose_session: Any) -> np.ndarray:
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
    )


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
    ChArUco XY补偿和固定的Y微调合并进去，因此可以在安全高度直接平移到
    最终XY/姿态，再用纯基坐标Z下降到安全位和最终点。

    该函数只由“共享精拍后的最终动作”调用，逐孔精定位和共享拍摄路径
    继续使用原有函数，避免新路径改变其它模式。
    """
    actual = np.asarray(current_tcp, dtype=np.float64).reshape(4, 4).copy()
    desired = np.asarray(target_tcp, dtype=np.float64).reshape(4, 4).copy()
    margin = max(10.0, float(safe_margin_mm))
    guard_mm = max(10.0, float(descent_guard_mm))

    safe_z = max(float(actual[2, 3]), float(desired[2, 3])) + margin
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
) -> np.ndarray:
    """沿安全高度移动到两阶段流程的目标相机高度。"""
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
    """把本轮最终定位结果固化为独立孔位地图。"""
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
        final_target_mode=str(getattr(args, "final_target_mode", DEFAULT_FINAL_TARGET_MODE)),
        final_point_offset_base_mm=report.get("final_point_offset_base_mm", [0.0, 0.0, 0.0]),
        tcp_xy_offset_mm=getattr(args, "tcp_xy_offset_mm", None),
    )
    pointcloud_archive = _find_batch_pointcloud_archive(run_dir, report)
    overlay_source = (
        (report.get("stages") or {}).get("batch_fine_final_result_overlay") or {}
    ).get("image_path")
    with artifact_measure(
        timing,
        "hole_map/write_artifacts",
        artifact_kind="hole_map_artifacts",
        paths=[str(map_path.parent)],
    ):
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
    hole_map_summary = {
        "path": str(saved),
        "map_id": map_id,
        "status": payload.get("status"),
        "ready_holes": sorted(
            int(item["hole_id"])
            for item in (payload.get("holes") or {}).values()
        ),
        "deferred_holes": payload.get("quality_summary", {}).get("deferred_holes", []),
        "scope": payload.get("scope"),
        "current_path": None if current_path is None else str(current_path),
        "current_updated": current_path is not None,
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
        "artifacts": hole_map_summary["artifacts"],
        "policy": "localize_all_selected_holes_then_execute_by_hole_id_without_camera",
    }
    report["status"] = (
        "hole_map_ready_with_deferred_holes"
        if payload.get("status") == "partial" else "hole_map_ready"
    )
    print(
        f"[HOLE_MAP_READY] path={saved} "
        f"ready_holes={hole_map_summary['ready_holes']} "
        f"deferred={hole_map_summary['deferred_holes']} "
        f"current={hole_map_summary['current_path'] or 'unchanged'}",
        flush=True,
    )
    return saved


def _map_hole_target_tcp(
    hole: dict[str, Any],
    pose_session: Any,
    *,
    final_target_mode: str,
    tcp_xy_offset_mm: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """从地图中的原始孔几何重新计算一次最终 TCP 目标。"""
    reference_pose = pose_session.pose_sdk_to_transform_mm(
        list(hole["execution_reference_tcp_pose_m_rad"]),
    )
    visual_point = np.asarray(hole["visual_center_base_mm"], dtype=np.float64).reshape(3)
    final_x_offset_mm, final_z_offset_mm = final_point_offsets_for_mode(final_target_mode)
    target_point = apply_final_point_base_offsets(
        visual_point,
        final_x_offset_mm,
        final_z_offset_mm,
    )
    fixed_offset = None if tcp_xy_offset_mm is None else (
        float(tcp_xy_offset_mm[0]), float(tcp_xy_offset_mm[1]),
    )
    xy_target, _ = plan_final_tcp_xy(reference_pose, target_point, fixed_offset)
    final_target = plan_final_tcp_base_z(xy_target, target_point)
    return final_target, {
        "visual_center_base_mm": visual_point.copy(),
        "target_point_base_mm": target_point.copy(),
        "reference_tcp_pose_m_rad": list(hole["execution_reference_tcp_pose_m_rad"]),
        "final_target_mode": str(final_target_mode),
        "final_point_offset_base_mm": [final_x_offset_mm, 0.0, final_z_offset_mm],
        "compensation_mode": (
            "charuco_affine_model" if fixed_offset is None else "fixed_offset_override"
        ),
        "charuco_model_source": str(CHARUCO_XY_MODEL_SOURCE) if fixed_offset is None else None,
        "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(final_target),
    }


def _new_hole_map_call_dir(map_path: Path) -> Path:
    """为一次地图调用创建独立的可追溯报告目录。"""
    parent = map_path.parent / "calls"
    parent.mkdir(parents=True, exist_ok=True)
    stem = f"call-{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
    candidate = parent / stem
    serial = 2
    while candidate.exists():
        candidate = parent / f"{stem}-{serial:02d}"
        serial += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def run_hole_map_execution(args: Any) -> int:
    """不启动相机、不运行YOLO，直接按孔位地图执行指定孔。"""
    map_path, _ = _resolve_hole_map_path(getattr(args, "hole_map_path", None))
    payload = load_hole_map(map_path)
    expected_mode = str(getattr(args, "final_target_mode", DEFAULT_FINAL_TARGET_MODE))
    requested_ids = getattr(args, "hole_ids", None)
    environment = payload.get("environment") or {}
    stored_handeye_hash = environment.get("handeye_sha256")
    current_handeye_hash = file_sha256(getattr(args, "handeye", ""))
    if stored_handeye_hash and stored_handeye_hash != current_handeye_hash:
        raise ValueError(
            "孔位地图对应的手眼文件已变化，请使用同一手眼标定或重新建立地图"
        )
    stored_charuco_hash = environment.get("charuco_model_sha256")
    current_charuco_hash = file_sha256(CHARUCO_XY_MODEL_SOURCE)
    if stored_charuco_hash and stored_charuco_hash != current_charuco_hash:
        raise ValueError(
            "孔位地图对应的 ChArUco 模型文件已变化，请重新建立地图"
        )
    stored_charuco_matrix = environment.get("charuco_model_matrix_2x2")
    stored_charuco_bias = environment.get("charuco_model_bias_mm")
    if stored_charuco_matrix is not None and not np.allclose(
        np.asarray(stored_charuco_matrix, dtype=np.float64),
        CHARUCO_XY_MODEL_MATRIX,
        atol=1e-12,
    ):
        raise ValueError("孔位地图使用的 ChArUco XY 模型已变化，请重新建立地图")
    if stored_charuco_bias is not None and not np.allclose(
        np.asarray(stored_charuco_bias, dtype=np.float64),
        CHARUCO_XY_MODEL_BIAS_MM,
        atol=1e-12,
    ):
        raise ValueError("孔位地图使用的 ChArUco XY 偏置已变化，请重新建立地图")
    hole_ids = validate_hole_map(
        payload,
        requested_hole_ids=requested_ids,
        expected_target_mode=expected_mode,
        expected_tcp_xy_offset_mm=getattr(args, "tcp_xy_offset_mm", None),
    )
    if not hole_ids:
        raise ValueError("孔位地图中没有可执行孔")

    from aubo_workbench.motion_control import AuboMotionSession
    from aubo_workbench.robot import AuboPoseSession

    call_dir = _new_hole_map_call_dir(map_path)
    call_report: dict[str, Any] = {
        "status": "running",
        "mode": "hole_map_execution",
        "map_path": str(map_path),
        "map_id": payload.get("map_id"),
        "requested_hole_ids": hole_ids,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "motion_executed": bool(args.execute),
        "final_target_mode": expected_mode,
        "scope": payload.get("scope"),
        "camera_started": False,
        "yolo_started": False,
        "holes": [],
        "safety_policy": payload.get("safety", {}),
    }
    report_path = call_dir / "report.json"
    pose_session = None
    motion_session = None
    current_tcp: np.ndarray | None = None
    try:
        pose_session = AuboPoseSession()
        pose_session.connect()
        _, current_tcp = _require_safe_snapshot(pose_session)
        if args.execute:
            motion_session = AuboMotionSession()
            motion_session.connect(
                ROBOT_CFG.ip,
                ROBOT_CFG.rpc_port,
                ROBOT_CFG.user,
                ROBOT_CFG.password,
                ROBOT_CFG.request_timeout_ms,
            )
        for order, hole_id in enumerate(hole_ids, start=1):
            hole = get_hole(payload, int(hole_id))
            target, target_details = _map_hole_target_tcp(
                hole,
                pose_session,
                final_target_mode=expected_mode,
                tcp_xy_offset_mm=getattr(args, "tcp_xy_offset_mm", None),
            )
            item: dict[str, Any] = {
                "hole_id": int(hole_id),
                "processing_order": order,
                "status": "preview" if not args.execute else "planned",
                "target": target_details,
                "current_tcp_before_m_rad": (
                    transform_to_sdk_pose_m_rad(current_tcp)
                    if current_tcp is not None else None
                ),
            }
            if args.execute:
                assert motion_session is not None and current_tcp is not None
                current_tcp = _move_to_fine_pose(
                    f"map_H{int(hole_id):02d}",
                    current_tcp,
                    target,
                    args,
                    motion_session,
                    pose_session,
                    target_height_mm=float(target[2, 3]),
                    target_stage="孔位地图调用",
                    descent_guard_mm=SHARED_OBSERVATION_MIN_DESCENT_MM,
                )
                y_trim_target = plan_final_tcp_combined_y_trim(current_tcp)
                current_tcp = _confirm_and_move_line(
                    f"地图孔{int(hole_id)}最终+Y微调",
                    current_tcp,
                    y_trim_target,
                    args,
                    motion_session,
                    pose_session,
                    "保持姿态；执行现有基坐标Y与工具系Y合并微调",
                    require_confirmation=False,
                    motion_profile="precision",
                )
                item["status"] = "completed"
                item["final_tcp_pose_m_rad"] = transform_to_sdk_pose_m_rad(current_tcp)
                item["final_y_trim_motion"] = {
                    "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(y_trim_target),
                    "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                    "delta_base_y_mm": FINAL_BASE_Y_AFTER_Z_MM,
                    "delta_tool_y_mm": FINAL_TOOL_Y_AFTER_Z_MM,
                }
            call_report["holes"].append(item)
            call_report["current_tcp_pose_m_rad"] = (
                transform_to_sdk_pose_m_rad(current_tcp) if current_tcp is not None else None
            )
            atomic_write_json(report_path, jsonable(call_report))
        call_report["status"] = "completed" if args.execute else "preview_completed"
        call_report["completed_holes"] = len(call_report["holes"])
        atomic_write_json(report_path, jsonable(call_report))
        print(
            f"[HOLE_MAP_CALL_DONE] map={map_path} holes={hole_ids} "
            f"motion_executed={bool(args.execute)} report={report_path}",
            flush=True,
        )
        return 0
    except Exception as exc:
        call_report["status"] = "failed"
        call_report["error"] = f"{type(exc).__name__}: {exc}"
        call_report["traceback"] = traceback.format_exc()
        atomic_write_json(report_path, jsonable(call_report))
        raise
    finally:
        if pose_session is not None:
            pose_session.disconnect()
        if motion_session is not None:
            motion_session.disconnect()


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
    final_target_mode = str(getattr(args, "final_target_mode", DEFAULT_FINAL_TARGET_MODE))
    final_x_offset_mm, final_z_offset_mm = final_point_offsets_for_mode(final_target_mode)
    return {
        "status": "running",
        "session_end_reason": None,
        "result_summary": None,
        "mode": "two_stage_hole_localization",
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cycle_index": int(cycle_index),
        "hole_map_mode": str(getattr(args, "hole_map_mode", "none")),
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
            timing=timing,
        )
    timing.mark("cycle/selection_confirmed", cycle_index=int(cycle_index))
    timing.mark("cycle/automatic_execution_start", cycle_index=int(cycle_index))
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
        # 建图模式只负责完成共享粗/精定位并固化结果；最终孔位动作由
        # 后续 hole-map execution 单独触发，避免建图和安放耦合。
        _write_hole_map_from_report(args, report, run_dir, timing=timing)
        # 资源字典名是 pipeline_runtime；这里仅记录路径供本轮报告/外层调试，
        # 不应因为地图已经保存成功而引用不存在的 runtime 局部变量。
        pipeline_runtime["hole_map_path"] = report["hole_map"]["path"]
    return int(result)




def run_two_stage_hole_localization(args: Any, handeye: Any, model: Any) -> int:
    """持续运行旧两阶段流程：每轮完成后回原点并重新进入初始选孔。"""
    cfg = TwoStageConfig.from_namespace(args)
    hole_map_mode = str(getattr(args, "hole_map_mode", "none"))
    if hole_map_mode == "build":
        if not cfg.batch_coarse_localization or not cfg.batch_fine_localization:
            raise ValueError(
                "建立孔位地图要求同时启用340 mm共享粗定位和260 mm共享精定位"
            )
        if bool(getattr(args, "move_final_xy", False)):
            raise ValueError(
                "建立孔位地图阶段不执行最终安放动作，请关闭 --move-final-xy"
            )
        # 地图是后续无相机调用的精确基准，必须来自本轮新鲜的340 mm
        # 共享点云；不允许用历史粗缓存节省时间而把旧工件坐标写进新地图。
        args.reuse_coarse_cache = False
        args.reuse_persistent_coarse_cache = False
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
    cycle_index = 1
    run_dir = _new_two_stage_run_dir(cycle_index)
    report = _new_two_stage_report(args, cfg, run_dir, cycle_index)
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
                    home, motion_session, pose_session,
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
        _write_report(run_dir, report, rows, timing=timing)

        while True:
            if cycle_index > 1:
                run_dir = _new_two_stage_run_dir(cycle_index)
                report = _new_two_stage_report(args, cfg, run_dir, cycle_index)
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
                _write_report(run_dir, report, rows, timing=timing)

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
            if str(getattr(args, "hole_map_mode", "none")) == "build":
                # 一次建图完成后结束当前进程，不回原点、不重新选孔，避免
                # 建图模式意外进入旧的循环选孔逻辑。
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
            _write_report(run_dir, report, rows, timing=timing)
            with timing.measure(
                "robot/return_home_after_cycle_confirmation",
                cycle_index=int(cycle_index),
            ):
                current_tcp = _confirm_and_move_home(
                    home, motion_session, pose_session,
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
        default_final_target_mode=DEFAULT_FINAL_TARGET_MODE,
        final_target_mode_gripper=FINAL_TARGET_MODE_GRIPPER,
        final_target_mode_normal=FINAL_TARGET_MODE_NORMAL,
    )


# 实现已统一到 aubo_workbench.config.apply_robot_connection_overrides。
_apply_robot_connection_overrides = apply_robot_connection_overrides


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _apply_robot_connection_overrides(args)
    if args.hole_map_mode == "execute":
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
