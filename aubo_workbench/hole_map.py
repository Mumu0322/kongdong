#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""粗定位导航地图的数据层。

地图默认以 340 mm 粗定位几何作为导航基准。逐孔建图模式还可以保存260 mm
精定位参考，或保存同拍340 mm精定位参考及通过严格粗质量门的点云中心回退；
运行地图时仍必须重新执行现场精定位，最终孔心和最终 TCP 目标不能写入地图。
旧版把最终 TCP 烘焙进地图的格式仍可被识别，但正常流程会明确拒绝它。
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io_utils import atomic_write_json, jsonable
from .rotary_sector_map import (
    ROTARY_SECTOR_COUNT,
    nominal_sector_angle_deg,
    sector_key,
    validate_sector_id,
)


HOLE_MAP_SCHEMA_VERSION = 2
ROTARY_HOLE_MAP_SCHEMA_VERSION = 4
LEGACY_HOLE_MAP_SCHEMA_VERSION = 1
LEGACY_ROTARY_HOLE_MAP_SCHEMA_VERSION = 3
HOLE_MAP_READY_STATUSES = frozenset({"ready", "completed"})
ROTARY_SECTOR_READY_STATUSES = frozenset({"valid", "ready", "completed"})
CURRENT_HOLE_MAP_POINTER_KIND = "current_hole_map_pointer"
CURRENT_HOLE_MAP_POINTER_SCHEMA_VERSION = 1

_FORBIDDEN_PERSISTED_FIELDS = frozenset({
    "fine_xy_base_mm",
    "fine_result_center_base_mm",
    "visual_center_base_mm",
    "hole_center_base_mm",
    "hole_center_base_naive_mm",
    "execution_final_tcp_pose_m_rad",
    "execution_final_tcp_target_base_mm",
    "execution_target_details",
    "target_point_base_mm",
    "final_tcp_pose_m_rad",
    "fine_tcp_pose_m_rad",
})


def file_sha256(path: str | Path) -> str | None:
    """返回文件指纹；文件不存在或读取失败时返回 ``None``。"""
    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _finite_vector(value: Any, length: int, field: str) -> list[float]:
    try:
        values = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"地图字段 {field} 不是有效向量") from exc
    if values.size != length or not np.isfinite(values).all():
        raise ValueError(f"地图字段 {field} 必须是 {length} 个有限数字")
    return [float(item) for item in values]


def _optional_vector(value: Any, length: int, field: str) -> list[float] | None:
    if value is None:
        return None
    return _finite_vector(value, length, field)


def _optional_scalar(value: Any) -> float | int | str | None:
    if value is None or isinstance(value, (str, int)):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _report_holes(report: dict[str, Any]) -> list[dict[str, Any]]:
    final_result = report.get("final_result") or {}
    raw_results = final_result.get("holes")
    if not isinstance(raw_results, list):
        raw_results = (report.get("stages") or {}).get("processed_holes", {}).get("holes")
    if not isinstance(raw_results, list):
        raw_results = []
    return [item for item in raw_results if isinstance(item, dict)]


def _coarse_capture_pose(result: dict[str, Any]) -> list[float] | None:
    for key in (
        "coarse_capture_tcp_pose_m_rad",
        "coarse_reference_tcp_pose_m_rad",
        "execution_reference_tcp_pose_m_rad",
    ):
        if result.get(key) is not None:
            return _finite_vector(result[key], 6, key)
    captures = result.get("coarse_captures")
    if isinstance(captures, list):
        for capture in reversed(captures):
            if isinstance(capture, dict) and capture.get("tcp_pose_m_rad") is not None:
                return _finite_vector(capture["tcp_pose_m_rad"], 6, "粗定位采集TCP位姿")
    return None


def _fine_reference_record(result: dict[str, Any]) -> dict[str, Any] | None:
    """按白名单保留逐孔精定位或点云回退参考，不保留最终运动目标。"""
    mode = str(result.get("map_build_localization_mode") or "").strip().lower()
    requested = bool(result.get("map_build_fine_reference")) or mode in {
        "per_hole",
        "per_hole_coarse_and_fine",
        "same_capture_340",
    }
    if not requested:
        return None
    pointcloud_fallback = bool(result.get("pointcloud_center_fallback"))
    if pointcloud_fallback:
        capture_policy = "per_hole_same_capture_340mm_pointcloud_fallback_reference_only"
    elif mode == "same_capture_340":
        capture_policy = "per_hole_same_capture_340mm_reference_only"
    else:
        capture_policy = "per_hole_fine_260mm_reference_only"

    hole_id = int(result.get("hole_id", 0) or 0)
    point = result.get("hole_center_base_mm")
    if point is None:
        return {
            "status": "missing",
            "capture_policy": capture_policy,
            "fine_quality_status": result.get("fine_quality_status"),
            "fine_quality_note": result.get("fine_quality_note"),
        }
    try:
        fine_point = _finite_vector(point, 3, f"孔{hole_id}.fine_reference.hole_center_base_mm")
    except ValueError as exc:
        return {
            "status": "invalid",
            "capture_policy": capture_policy,
            "fine_quality_status": result.get("fine_quality_status"),
            "fine_quality_note": f"精定位参考点无效：{exc}",
        }

    reference: dict[str, Any] = {
        "status": "coarse_fallback" if pointcloud_fallback else "ready",
        "reference_source": (
            "coarse_pointcloud_center_fallback" if pointcloud_fallback
            else "rgb_fine_localization"
        ),
        "capture_policy": capture_policy,
        "hole_center_base_mm": fine_point,
        "fine_plane_intersection_mm": _optional_vector(
            result.get("fine_plane_intersection_mm"),
            3,
            f"孔{hole_id}.fine_reference.fine_plane_intersection_mm",
        ),
        "fine_capture_tcp_pose_m_rad": _optional_vector(
            result.get("fine_tcp_pose_m_rad")
            if result.get("fine_tcp_pose_m_rad") is not None
            else result.get("shared_batch_capture_tcp_pose_m_rad"),
            6,
            f"孔{hole_id}.fine_reference.fine_capture_tcp_pose_m_rad",
        ),
        "fine_quality_status": result.get("fine_quality_status"),
        "fine_quality_note": result.get("fine_quality_note"),
        "fine_center_source": result.get("fine_center_source"),
        "fine_center_source_counts": jsonable(result.get("fine_center_source_counts") or {}),
        "fine_xy_source": result.get("fine_xy_source"),
        "fine_z_source": result.get("fine_z_source"),
        "valid_frames": _optional_scalar(result.get("valid_frames")),
        "total_frames": _optional_scalar(result.get("total_frames")),
        "rejected_outlier_frames": _optional_scalar(result.get("rejected_outlier_frames")),
        "center_scatter_p95_px": _optional_scalar(result.get("center_scatter_p95_px")),
        "ellipse_residual_median_px": _optional_scalar(
            result.get("ellipse_residual_median_px")
        ),
        "ellipse_roundness_median": _optional_scalar(
            result.get("ellipse_roundness_median")
        ),
        "axes_px_median": jsonable(result.get("axes_px_median")),
        "strict_ellipse_frames": _optional_scalar(result.get("strict_ellipse_frames")),
        "yolo_frames": _optional_scalar(result.get("yolo_frames")),
        "yolo_relaxed_ellipse_frames": _optional_scalar(
            result.get("yolo_relaxed_ellipse_frames")
        ),
        "yolo_fallback_frames": _optional_scalar(result.get("yolo_fallback_frames")),
        "fine_height_estimate_mm": _optional_scalar(result.get("fine_height_estimate_mm")),
    }
    if pointcloud_fallback:
        reference["pointcloud_fallback_reason"] = result.get(
            "pointcloud_center_fallback_reason"
        ) or result.get("fine_quality_note")
        reference["pointcloud_fallback_validation"] = jsonable(
            result.get("pointcloud_center_fallback_validation") or {}
        )
    coarse_point = result.get("coarse_center_base_mm")
    if coarse_point is not None and not pointcloud_fallback:
        try:
            coarse_values = np.asarray(coarse_point, dtype=np.float64).reshape(-1)
            fine_values = np.asarray(fine_point, dtype=np.float64).reshape(-1)
            if coarse_values.size == 3 and np.isfinite(coarse_values).all():
                delta_xy = fine_values[:2] - coarse_values[:2]
                reference["coarse_to_fine_xy_delta_mm"] = [
                    float(delta_xy[0]), float(delta_xy[1]),
                ]
                reference["coarse_to_fine_xy_distance_mm"] = float(
                    np.linalg.norm(delta_xy)
                )
        except (TypeError, ValueError):
            pass
    return {key: value for key, value in reference.items() if value is not None}


def _coarse_hole_record(result: dict[str, Any]) -> dict[str, Any]:
    """从单孔运行结果按白名单构造地图记录。"""
    hole_id = int(result["hole_id"])
    coarse_center = _finite_vector(
        result.get("coarse_center_base_mm"), 3, f"孔{hole_id}.coarse_center_base_mm",
    )
    coarse_plane = _finite_vector(
        result.get("coarse_plane_point_base_mm", coarse_center),
        3,
        f"孔{hole_id}.coarse_plane_point_base_mm",
    )
    coarse_normal = _finite_vector(
        result.get("coarse_normal_toward_camera_base"),
        3,
        f"孔{hole_id}.coarse_normal_toward_camera_base",
    )
    initial_detection = result.get("initial_detection")
    record: dict[str, Any] = {
        "hole_id": hole_id,
        "status": "ready",
        "tracking_identity": result.get("tracking_identity"),
        "class_id": (
            initial_detection.get("class_id")
            if isinstance(initial_detection, dict) else result.get("class_id")
        ),
        "class_name": (
            initial_detection.get("class_name")
            if isinstance(initial_detection, dict) else result.get("class_name")
        ),
        "initial_selection_order": result.get("initial_selection_order"),
        "execution_order": result.get("processing_order", result.get("execution_order")),
        "operator_selection_order": result.get("operator_selection_order"),
        "selection_source": result.get("selection_source"),
        "numbering_policy": result.get("numbering_policy"),
        "map_order": result.get("map_order"),
        "layout_row": result.get("layout_row"),
        "layout_column": result.get("layout_column"),
        "numbering_center_px": result.get("numbering_center_px"),
        "numbering_row_tolerance_px": _optional_scalar(
            result.get("numbering_row_tolerance_px")
        ),
        "map_hole_key": result.get("map_hole_key"),
        "diameter_estimate_mm": _optional_scalar(result.get("diameter_estimate_mm")),
        "matched_diameter_mm": _optional_scalar(result.get("matched_diameter_mm")),
        "coarse_center_base_mm": coarse_center,
        "coarse_z_base_mm": float(coarse_center[2]),
        "coarse_plane_point_base_mm": coarse_plane,
        "coarse_normal_toward_camera_base": coarse_normal,
        "coarse_center_camera_mm": _optional_vector(
            result.get("coarse_center_camera_mm"), 3, f"孔{hole_id}.coarse_center_camera_mm",
        ),
        "coarse_plane_point_camera_mm": _optional_vector(
            result.get("coarse_plane_point_camera_mm"),
            3,
            f"孔{hole_id}.coarse_plane_point_camera_mm",
        ),
        "coarse_normal_camera": _optional_vector(
            result.get("coarse_normal_camera"), 3, f"孔{hole_id}.coarse_normal_camera",
        ),
        "coarse_capture_tcp_pose_m_rad": _coarse_capture_pose(result),
        "coarse_valid_frames": result.get("coarse_valid_frames"),
        "coarse_total_frames": result.get("coarse_total_frames"),
        "coarse_plane_rmse_mm": _optional_scalar(result.get("coarse_plane_rmse_mm")),
        "coarse_center_scatter_p95_px": _optional_scalar(result.get("coarse_center_scatter_p95_px")),
        "coarse_tracking_distance_p95_px": _optional_scalar(result.get("coarse_tracking_distance_p95_px")),
        "coarse_ring_points_median": result.get("coarse_ring_points_median"),
        "coarse_ring_coverage_min_ratio": result.get("coarse_ring_coverage_min_ratio"),
        "coarse_ring_max_gap_deg": result.get("coarse_ring_max_gap_deg"),
        "coarse_quality_retry": result.get("coarse_quality_retry", False),
        "coarse_surface_model": result.get("coarse_surface_model"),
        "coarse_surface_selection_policy": result.get("coarse_surface_selection_policy"),
        "coarse_front_surface_z_mm": _optional_scalar(result.get("coarse_front_surface_z_mm")),
        "coarse_ring_points_raw_median": result.get("coarse_ring_points_raw_median"),
        "coarse_surface_points_selected_median": result.get("coarse_surface_points_selected_median"),
        "coarse_sphere_center_camera_mm": _optional_vector(
            result.get("coarse_sphere_center_camera_mm"),
            3,
            f"孔{hole_id}.coarse_sphere_center_camera_mm",
        ),
        "coarse_sphere_radius_mm": _optional_scalar(result.get("coarse_sphere_radius_mm")),
        "coarse_source": result.get("coarse_source"),
        "batch_coarse_group_index": result.get("batch_coarse_group_index"),
        "batch_coarse_group_hole_ids": result.get("batch_coarse_group_hole_ids"),
        "batch_coarse_pose_refinement_accepted": result.get(
            "batch_coarse_pose_refinement_accepted"
        ),
        "batch_coarse_pose_refinement_map_build_safe": result.get(
            "batch_coarse_pose_refinement_map_build_safe"
        ),
        "batch_coarse_pose_refinement_reason": result.get(
            "batch_coarse_pose_refinement_reason"
        ),
        "batch_coarse_pose_refinement_iterations": result.get(
            "batch_coarse_pose_refinement_iterations"
        ),
        "batch_coarse_pose_refinement_final_metrics": jsonable(
            result.get("batch_coarse_pose_refinement_final_metrics")
        ),
        "coarse_captures": jsonable(result.get("coarse_captures") or []),
        "coarse_pointcloud_image_path": result.get("coarse_pointcloud_image_path"),
        "batch_pointcloud_archive_path": result.get("batch_pointcloud_archive_path"),
        "quality": {
            "valid_frames": result.get("coarse_valid_frames"),
            "plane_rmse_mm": _optional_scalar(result.get("coarse_plane_rmse_mm")),
            "center_scatter_p95_px": _optional_scalar(result.get("coarse_center_scatter_p95_px")),
            "tracking_distance_p95_px": _optional_scalar(result.get("coarse_tracking_distance_p95_px")),
            "ring_points_median": result.get("coarse_ring_points_median"),
        },
        "source_result_keys": {
            "hole_result_type": result.get("hole_result_type"),
            "coarse_source": result.get("coarse_source"),
        },
        "pointcloud_center_fallback": bool(
            result.get("pointcloud_center_fallback", False)
        ),
        "fine_reference": _fine_reference_record(result),
    }
    if result.get("pointcloud_center_fallback"):
        record["pointcloud_center_fallback_reason"] = (
            result.get("pointcloud_center_fallback_reason")
            or result.get("fine_quality_note")
        )
        record["pointcloud_center_fallback_validation"] = jsonable(
            result.get("pointcloud_center_fallback_validation") or {}
        )
    return {key: value for key, value in record.items() if value is not None}


def _environment_from_report(
    report: dict[str, Any],
    *,
    handeye_path: str | Path | None,
    camera_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    camera = dict(camera_identity or report.get("camera") or {})
    home_selection = (report.get("stages") or {}).get("home_selection") or {}
    auto_selection = home_selection.get("auto_sector_selection") or {}
    return {
        "robot_id": report.get("robot_id") or report.get("robot_name"),
        "robot_initial_tcp_pose_m_rad": report.get("robot_initial_tcp_pose_m_rad"),
        "home_point": report.get("home_point"),
        "camera_serial": camera.get("serial_number") or camera.get("serial"),
        "camera_identity": camera,
        "handeye_path": None if handeye_path is None else str(handeye_path),
        "handeye_sha256": None if handeye_path is None else file_sha256(handeye_path),
        "coarse_capture_tcp_pose_m_rad": home_selection.get("capture_tcp_pose_m_rad"),
        "observation_pose_id": home_selection.get("observation_pose_id"),
        "observation_image_size": home_selection.get("image_size"),
        "sector_config_path": auto_selection.get("config_path"),
        "sector_config_version": auto_selection.get("config_version"),
    }


def build_hole_map_payload(
    report: dict[str, Any],
    *,
    map_id: str,
    source_run_dir: str | Path,
    handeye_path: str | Path | None,
    camera_identity: dict[str, Any] | None,
    charuco_model_source: str | Path | None = None,
    tcp_xy_offset_mm: Any = None,
    charuco_model_matrix: Any = None,
    charuco_model_bias_mm: Any = None,
) -> dict[str, Any]:
    """从运行报告生成粗定位地图，并按需附带逐孔精定位参考。

    补偿和 ChArUco 参数保留在签名中只是为了让旧调用方平滑迁移；
    它们不会写入地图，也不会参与地图结果计算。
    """
    del charuco_model_source
    del tcp_xy_offset_mm, charuco_model_matrix, charuco_model_bias_mm
    raw_results = _report_holes(report)
    if not raw_results:
        raise ValueError("运行报告中没有孔定位结果")
    report_hole_map = report.get("hole_map")
    report_hole_map = report_hole_map if isinstance(report_hole_map, dict) else {}
    reported_mode = str(
        report.get("map_build_localization_mode")
        or report_hole_map.get("map_build_localization_mode")
        or "coarse_only"
    ).strip().lower()
    fine_reference_requested = reported_mode in {
        "per_hole",
        "per_hole_coarse_and_fine",
        "same_capture_340",
    } or any(
        bool(item.get("map_build_fine_reference"))
        or str(item.get("map_build_localization_mode") or "").strip().lower()
        in {"per_hole", "per_hole_coarse_and_fine", "same_capture_340"}
        for item in raw_results
    )
    map_build_localization_mode = (
        "same_capture_340" if reported_mode == "same_capture_340" else
        "per_hole" if fine_reference_requested else "coarse_only"
    )
    batch_requested = bool(
        (report.get("configuration") or {}).get("batch_coarse_localization", False)
    )
    if map_build_localization_mode == "coarse_only" and batch_requested:
        # 共享粗定位地图只能来自已经完成共享位姿闭环的组。运行时主流程
        # 会在正式采集前拦截未收敛组；这里再加一层写图防线，避免旧入口
        # 或外部脚本把“纠偏失败但仍有结果”的报告写成可执行地图。
        batch_stage = (report.get("stages") or {}).get("batch_coarse_results") or {}
        groups = batch_stage.get("groups") or []
        recovered_by_parent: dict[int, set[int]] = {}
        for group in groups:
            if not isinstance(group, dict) or not group.get("coarse_quality_retry"):
                continue
            refinement = group.get("pose_refinement") or {}
            if not isinstance(refinement, dict) or not (
                bool(refinement.get("accepted", False))
                or bool(refinement.get("map_build_safe", False))
            ):
                continue
            try:
                parent_group = int(group.get("coarse_quality_retry_parent_group"))
            except (TypeError, ValueError):
                continue
            recovered_by_parent.setdefault(parent_group, set()).update(
                int(hole_id) for hole_id in (group.get("accepted_holes") or [])
            )
        rejected_groups = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            refinement = group.get("pose_refinement")
            if not isinstance(refinement, dict):
                continue
            if not bool(refinement.get("accepted", False)) and not bool(
                refinement.get("map_build_safe", False)
            ):
                retry_holes = {
                    int(hole_id) for hole_id in (group.get("quality_retry_holes") or [])
                }
                if retry_holes:
                    try:
                        group_index = int(group.get("group_index"))
                    except (TypeError, ValueError):
                        group_index = None
                    recovered_holes = (
                        recovered_by_parent.get(group_index, set())
                        if group_index is not None else set()
                    )
                    original_accepted = {
                        int(hole_id) for hole_id in (group.get("accepted_holes") or [])
                    }
                    # A rejected shared pose is recoverable only when every hole
                    # that was queued for singleton recapture has a successful,
                    # accepted replacement. Any holes captured before the failure
                    # may remain covered by the original group result.
                    covered_holes = original_accepted | recovered_holes
                    if retry_holes <= recovered_holes and set(
                        int(hole_id) for hole_id in (group.get("hole_ids") or [])
                    ) <= covered_holes:
                        continue
                rejected_groups.append({
                    "group_index": group.get("group_index"),
                    "hole_ids": group.get("hole_ids") or [],
                    "reason": refinement.get("reason") or "pose_refinement_not_accepted",
                })
        if rejected_groups:
            raise ValueError(
                "共享粗定位地图拒绝写入：存在未收敛的共享位姿组；"
                f"rejected_groups={rejected_groups}"
            )
    holes: dict[str, dict[str, Any]] = {}
    deferred: list[dict[str, Any]] = []
    errors: list[str] = []
    fine_reference_missing: list[int] = []
    fine_reference_ready: list[int] = []
    fine_localization_passed: list[int] = []
    pointcloud_fallback_holes: list[int] = []
    for result in raw_results:
        try:
            hole_id = int(result["hole_id"])
        except (KeyError, TypeError, ValueError):
            errors.append("发现没有有效 hole_id 的结果")
            continue
        status = str(result.get("status", "")).lower()
        if status not in HOLE_MAP_READY_STATUSES:
            if fine_reference_requested:
                fine_reference_missing.append(hole_id)
            deferred.append({
                "hole_id": hole_id,
                "status": result.get("status"),
                "error": (
                    result.get("error") or result.get("deferred_reason")
                    or result.get("fine_quality_note") or result.get("coarse_quality_note")
                ),
            })
            continue
        try:
            record = _coarse_hole_record(result)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc))
            deferred.append({"hole_id": hole_id, "status": "invalid_coarse_record", "error": str(exc)})
            continue
        key = f"H{hole_id:02d}"
        if key in holes:
            errors.append(f"孔 {hole_id} 重复出现在运行报告中")
            continue
        if fine_reference_requested:
            fine_reference = record.get("fine_reference") or {}
            reference_status = str(fine_reference.get("status", "")).lower()
            if reference_status in {"ready", "coarse_fallback"}:
                fine_reference_ready.append(hole_id)
                if reference_status == "coarse_fallback":
                    pointcloud_fallback_holes.append(hole_id)
                else:
                    fine_localization_passed.append(hole_id)
            else:
                fine_reference_missing.append(hole_id)
                reason = (
                    fine_reference.get("fine_quality_note")
                    or "逐孔精定位参考缺失或无效"
                )
                errors.append(f"孔 {hole_id} {reason}")
                deferred.append({
                    "hole_id": hole_id,
                    "status": "invalid_fine_reference",
                    "error": reason,
                })
        holes[key] = record
    if not holes:
        detail = "; ".join(errors or ["没有可用的已完成粗定位孔"])
        raise ValueError(f"无法建立粗定位地图：{detail}")
    # 文件中的键顺序也按地图编号排列，避免执行路径优化后的处理顺序
    # 让导出的JSON/PLY预览看起来像是重新编号了。
    holes = dict(sorted(holes.items(), key=lambda item: int(item[1]["hole_id"])))
    source_path = Path(source_run_dir)
    payload: dict[str, Any] = {
        "schema_version": HOLE_MAP_SCHEMA_VERSION,
        "kind": "coarse_hole_map",
        "map_id": str(map_id),
        "status": "valid" if not deferred and not errors else "partial",
        "scope": "current_robot_cycle",
        "coordinate_frame": "robot_base_mm",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_run_id": source_path.name,
        "source_report_path": str(source_path / "report.json"),
        "map_build_localization_mode": map_build_localization_mode,
        "reference_policy": (
            "per_hole_same_capture_340mm_fine_reference_saved_for_diagnostics;"
            "failed_fine_may_use_quality_gated_pointcloud_center_fallback;"
            "runtime_requires_fresh_fine_and_does_not_use_persisted_final_tcp"
            if map_build_localization_mode == "same_capture_340" else
            "per_hole_260mm_fine_reference_saved_for_diagnostics_and_seed_correction;"
            "runtime_requires_fresh_fine_and_does_not_use_persisted_final_tcp"
            if fine_reference_requested else
            "340mm_coarse_geometry_only;runtime_requires_fresh_fine"
        ),
        "target_policy": "coarse_map_requires_fresh_fine",
        "requires_fresh_fine": True,
        "numbering": jsonable(report.get("map_numbering") or {}),
        "environment": _environment_from_report(
            report, handeye_path=handeye_path, camera_identity=camera_identity,
        ),
        "quality_summary": {
            "ready_holes": len(holes),
            "deferred_holes": deferred,
            "map_errors": errors,
            "coarse_only": not fine_reference_requested,
            "fine_reference_requested": fine_reference_requested,
            "fine_reference_ready_holes": sorted(fine_reference_ready),
            "fine_reference_missing_holes": sorted(fine_reference_missing),
            "fine_localization_passed_holes": sorted(fine_localization_passed),
            "pointcloud_fallback_holes": sorted(pointcloud_fallback_holes),
            "batch_coarse_requested": batch_requested,
        },
        "safety": {
            "requires_same_workpiece_pose": True,
            "requires_same_robot_tcp_and_calibration": True,
            "requires_runtime_camera_and_fine_localization": True,
            "minimum_lift_mm": 10.0,
            "minimum_final_descent_guard_mm": 10.0,
        },
        "holes": holes,
    }
    return jsonable(payload)


def _rotary_sector_common_payload(
    source_payload: dict[str, Any],
    *,
    map_id: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "schema_version": ROTARY_HOLE_MAP_SCHEMA_VERSION,
        "kind": "coarse_sector_hole_map",
        "map_id": str(map_id),
        "status": "partial",
        "scope": "rotary_six_sector",
        "coordinate_frame": "robot_base_mm",
        "created_at": source_payload.get("created_at", now),
        "updated_at": now,
        "source_run_id": source_payload.get("source_run_id", report.get("run_id")),
        "source_report_path": source_payload.get("source_report_path"),
        "map_build_localization_mode": source_payload.get(
            "map_build_localization_mode", "coarse_only"
        ),
        "reference_policy": source_payload.get("reference_policy"),
        "target_policy": "coarse_map_requires_fresh_fine",
        "requires_fresh_fine": True,
        "numbering": deepcopy(source_payload.get("numbering") or {}),
        "environment": deepcopy(source_payload.get("environment") or {}),
        "safety": {
            "requires_same_workpiece_rotation_axis": True,
            "requires_same_robot_tcp_and_calibration": True,
            "requires_runtime_camera_and_fine_localization": True,
            "minimum_lift_mm": 10.0,
            "minimum_final_descent_guard_mm": 10.0,
        },
        "sectors": {},
        "quality_summary": {
            "sector_count": ROTARY_SECTOR_COUNT,
            "ready_sector_ids": [],
            "missing_sector_ids": list(range(1, ROTARY_SECTOR_COUNT + 1)),
            "ready_holes": 0,
            "deferred_holes": 0,
            "fine_reference_ready_holes": 0,
            "fine_reference_missing_holes": 0,
            "fine_localization_passed_holes": 0,
            "pointcloud_fallback_holes": 0,
        },
    }


def build_rotary_sector_map_payload(
    report: dict[str, Any],
    *,
    map_id: str,
    source_run_dir: str | Path,
    handeye_path: str | Path | None,
    camera_identity: dict[str, Any] | None,
    charuco_model_source: str | Path | None = None,
    sector_id: int,
    existing_payload: dict[str, Any] | None = None,
    tcp_xy_offset_mm: Any = None,
    charuco_model_matrix: Any = None,
    charuco_model_bias_mm: Any = None,
) -> dict[str, Any]:
    """建立或替换一个扇区的纯粗定位记录。"""
    del charuco_model_source
    del tcp_xy_offset_mm, charuco_model_matrix, charuco_model_bias_mm
    sector_number = validate_sector_id(sector_id)
    sector_name = sector_key(sector_number)
    sector_source = build_hole_map_payload(
        report,
        map_id=map_id,
        source_run_dir=source_run_dir,
        handeye_path=handeye_path,
        camera_identity=camera_identity,
    )
    holes = dict(sector_source.get("holes") or {})
    if existing_payload is not None:
        candidate = deepcopy(jsonable(existing_payload))
        if int(candidate.get("schema_version", -1)) != ROTARY_HOLE_MAP_SCHEMA_VERSION:
            raise ValueError("只能把新扇区合并到 v4 粗定位地图；旧最终点地图请重新建立")
        validate_hole_map(candidate)
        stored_environment = candidate.get("environment") or {}
        current_environment = sector_source.get("environment") or {}
        for field, label in (("handeye_sha256", "手眼标定"), ("camera_serial", "相机序列号")):
            stored = str(stored_environment.get(field) or "").strip()
            current = str(current_environment.get(field) or "").strip()
            if stored and current and stored != current:
                raise ValueError(f"现有六扇区地图的{label}不同，不能混合建图")
        payload = candidate
        payload["map_id"] = str(map_id)
        payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    else:
        payload = _rotary_sector_common_payload(sector_source, map_id=map_id, report=report)
    sector_mode = str(
        sector_source.get("map_build_localization_mode", "coarse_only")
    ).strip().lower()
    existing_mode = str(payload.get("map_build_localization_mode", "")).strip().lower()
    if existing_payload is not None and existing_mode and existing_mode != sector_mode:
        payload["map_build_localization_mode"] = "mixed"
        payload["reference_policy"] = (
            "mixed_sector_modes;each_sector_records_its_own_coarse_and_optional_fine_reference;"
            "runtime_requires_fresh_fine"
        )
    else:
        payload["map_build_localization_mode"] = sector_mode
        payload["reference_policy"] = sector_source.get("reference_policy")
    payload.setdefault("environment", {}).update({
        key: value for key, value in (sector_source.get("environment") or {}).items()
        if value is not None
    })
    payload.setdefault("sectors", {})[sector_name] = {
        "sector_id": sector_number,
        "sector_key": sector_name,
        "map_build_localization_mode": sector_mode,
        "status": "valid" if sector_source.get("status") == "valid" else "partial",
        "nominal_angle_deg": nominal_sector_angle_deg(sector_number),
        "reference_policy": (
            "per_hole_same_capture_340mm_fine_reference_or_quality_gated_pointcloud_fallback"
            if sector_mode == "same_capture_340" else
            "per_hole_260mm_fine_reference_saved_for_diagnostics_and_seed_correction"
            if sector_source.get("map_build_localization_mode") == "per_hole" else
            "coarse_geometry_from_this_sector_build"
        ),
        "target_policy": "coarse_map_requires_fresh_fine",
        "requires_fresh_fine": True,
        "source_run_id": sector_source.get("source_run_id"),
        "source_report_path": sector_source.get("source_report_path"),
        "numbering": deepcopy(sector_source.get("numbering") or {}),
        "quality_summary": sector_source.get("quality_summary", {}),
        "holes": holes,
        "artifacts": {},
    }
    _recompute_rotary_quality_summary(payload)
    return jsonable(payload)


def _recompute_rotary_quality_summary(payload: dict[str, Any]) -> None:
    sectors = payload.get("sectors") if isinstance(payload.get("sectors"), dict) else {}
    ready_sector_ids: list[int] = []
    ready_hole_count = 0
    deferred_hole_count = 0
    fine_reference_ready_count = 0
    fine_reference_missing_count = 0
    fine_localization_passed_count = 0
    pointcloud_fallback_count = 0
    for sector in sectors.values():
        if not isinstance(sector, dict):
            continue
        sid = int(sector.get("sector_id", 0))
        ready = str(sector.get("status", "")).lower() in ROTARY_SECTOR_READY_STATUSES
        holes = sector.get("holes") if isinstance(sector.get("holes"), dict) else {}
        ready_hole_count += sum(
            1 for hole in holes.values()
            if isinstance(hole, dict) and str(hole.get("status", "ready")) in HOLE_MAP_READY_STATUSES
        )
        deferred_hole_count += int((sector.get("quality_summary") or {}).get("deferred_holes_count", 0) or 0)
        sector_quality = sector.get("quality_summary") or {}
        fine_reference_ready_count += int(
            sector_quality.get("fine_reference_ready_holes", [])
            if isinstance(sector_quality.get("fine_reference_ready_holes"), int)
            else len(sector_quality.get("fine_reference_ready_holes") or [])
        )
        fine_reference_missing_count += int(
            sector_quality.get("fine_reference_missing_holes", [])
            if isinstance(sector_quality.get("fine_reference_missing_holes"), int)
            else len(sector_quality.get("fine_reference_missing_holes") or [])
        )
        fine_localization_passed_count += int(
            sector_quality.get("fine_localization_passed_holes", [])
            if isinstance(sector_quality.get("fine_localization_passed_holes"), int)
            else len(sector_quality.get("fine_localization_passed_holes") or [])
        )
        pointcloud_fallback_count += int(
            sector_quality.get("pointcloud_fallback_holes", [])
            if isinstance(sector_quality.get("pointcloud_fallback_holes"), int)
            else len(sector_quality.get("pointcloud_fallback_holes") or [])
        )
        if ready:
            ready_sector_ids.append(sid)
    payload["quality_summary"] = {
        "sector_count": ROTARY_SECTOR_COUNT,
        "ready_sector_ids": sorted(ready_sector_ids),
        "missing_sector_ids": [sid for sid in range(1, ROTARY_SECTOR_COUNT + 1) if sid not in ready_sector_ids],
        "ready_holes": int(ready_hole_count),
        "deferred_holes": int(deferred_hole_count),
        "fine_reference_ready_holes": int(fine_reference_ready_count),
        "fine_reference_missing_holes": int(fine_reference_missing_count),
        "fine_localization_passed_holes": int(fine_localization_passed_count),
        "pointcloud_fallback_holes": int(pointcloud_fallback_count),
        "map_build_localization_mode": payload.get(
            "map_build_localization_mode", "coarse_only"
        ),
    }
    payload["status"] = "valid" if len(ready_sector_ids) == ROTARY_SECTOR_COUNT else "partial"


def build_rotary_sector_hole_repair_payload(
    existing_payload: dict[str, Any],
    report: dict[str, Any],
    *,
    map_id: str,
    source_run_dir: str | Path,
    handeye_path: str | Path | None,
    camera_identity: dict[str, Any] | None,
    charuco_model_source: str | Path | None = None,
    sector_id: int,
    hole_id: int,
    tcp_xy_offset_mm: Any = None,
    charuco_model_matrix: Any = None,
    charuco_model_bias_mm: Any = None,
) -> dict[str, Any]:
    """用新鲜粗定位替换一个孔；不保留、不拼接旧精定位结果。"""
    del charuco_model_source
    del tcp_xy_offset_mm, charuco_model_matrix, charuco_model_bias_mm
    sector_number = validate_sector_id(sector_id)
    validate_rotary_sector_map(
        existing_payload,
        sector_id=sector_number,
        requested_hole_ids=[int(hole_id)],
    )
    source = build_hole_map_payload(
        {"final_result": {"holes": _report_holes(report)}},
        map_id=map_id,
        source_run_dir=source_run_dir,
        handeye_path=handeye_path,
        camera_identity=camera_identity,
    )
    fresh = (source.get("holes") or {}).get(f"H{int(hole_id):02d}")
    if not isinstance(fresh, dict):
        raise ValueError(f"粗定位返修报告中没有孔 {int(hole_id)} 的有效结果")
    payload = deepcopy(jsonable(existing_payload))
    payload["map_id"] = str(map_id)
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    payload["source_run_id"] = Path(source_run_dir).name
    payload["source_report_path"] = str(Path(source_run_dir) / "report.json")
    history = payload.setdefault("repair_history", [])
    history.append({
        "sector_id": sector_number,
        "hole_id": int(hole_id),
        "repair_mode": "coarse_refresh",
        "source_run_id": Path(source_run_dir).name,
        "previous_map_id": existing_payload.get("map_id"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    sector = (payload.get("sectors") or {}).get(sector_key(sector_number))
    if not isinstance(sector, dict):
        raise ValueError(f"现有地图缺少扇区 {sector_key(sector_number)}")
    hole_key = f"H{int(hole_id):02d}"
    previous = (sector.get("holes") or {}).get(hole_key)
    if isinstance(previous, dict):
        # 返修只刷新粗定位几何，既有空间编号和行列元数据必须保持不变。
        for key in (
            "numbering_policy", "map_order", "layout_row", "layout_column",
            "numbering_center_px", "numbering_row_tolerance_px", "map_hole_key",
            "selection_source", "operator_selection_order",
        ):
            if fresh.get(key) is None and previous.get(key) is not None:
                fresh[key] = previous[key]
    sector.setdefault("holes", {})[hole_key] = fresh
    sector["last_repair"] = {
        "hole_id": int(hole_id),
        "repair_mode": "coarse_refresh",
        "updated_at": payload["updated_at"],
    }
    _recompute_rotary_quality_summary(payload)
    validate_hole_map(payload)
    return jsonable(payload)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"孔位地图无法读取：{path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"孔位地图根节点必须是对象：{path}")
    return raw


def resolve_hole_map_path(path: str | Path) -> Path:
    """解析地图文件；``current.json`` 是受限于同一目录的指针。"""
    target = Path(path).expanduser()
    if target.name.lower() != "current.json" or not target.is_file():
        return target
    raw = _read_json_object(target)
    if raw.get("kind") != CURRENT_HOLE_MAP_POINTER_KIND:
        return target
    if int(raw.get("schema_version", -1)) != CURRENT_HOLE_MAP_POINTER_SCHEMA_VERSION:
        raise ValueError(f"当前孔位地图指针版本不兼容：{target}")
    raw_path = str(raw.get("map_path", "")).strip()
    if not raw_path:
        raise ValueError(f"当前孔位地图指针缺少 map_path：{target}")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = target.parent / candidate
    candidate = candidate.resolve()
    root = target.parent.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"当前孔位地图指针越过地图目录：{candidate}") from exc
    relative = candidate.relative_to(root)
    if candidate.name.lower() != "hole_map.json" or len(relative.parts) != 2:
        raise ValueError(f"当前孔位地图指针不是有效地图文件：{candidate}")
    return candidate


def _validate_hole_record(key: str, hole: dict[str, Any]) -> int:
    if not isinstance(hole, dict):
        raise ValueError(f"孔位地图条目 {key} 不是对象")
    try:
        hole_id = int(hole["hole_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"孔位地图条目 {key} 的 hole_id 无效") from exc
    if str(key) != f"H{hole_id:02d}":
        raise ValueError(f"孔位地图键 {key} 与 hole_id={hole_id} 不一致")
    if hole.get("numbering_policy") == "image_row_major_v1":
        try:
            map_order = int(hole.get("map_order"))
            row = int(hole.get("layout_row"))
            column = int(hole.get("layout_column"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"孔位地图条目 {key} 的行优先编号元数据无效") from exc
        if map_order != hole_id or row < 1 or column < 1:
            raise ValueError(f"孔位地图条目 {key} 的行优先编号元数据不一致")
    forbidden = sorted(_FORBIDDEN_PERSISTED_FIELDS.intersection(hole))
    if forbidden:
        raise ValueError(f"粗定位地图条目 {key} 含禁止持久化字段：{', '.join(forbidden)}")
    _finite_vector(hole.get("coarse_center_base_mm"), 3, f"{key}.coarse_center_base_mm")
    _finite_vector(hole.get("coarse_plane_point_base_mm"), 3, f"{key}.coarse_plane_point_base_mm")
    _finite_vector(hole.get("coarse_normal_toward_camera_base"), 3, f"{key}.coarse_normal_toward_camera_base")
    _optional_vector(hole.get("coarse_center_camera_mm"), 3, f"{key}.coarse_center_camera_mm")
    _optional_vector(hole.get("coarse_plane_point_camera_mm"), 3, f"{key}.coarse_plane_point_camera_mm")
    _optional_vector(hole.get("coarse_normal_camera"), 3, f"{key}.coarse_normal_camera")
    _optional_vector(hole.get("coarse_capture_tcp_pose_m_rad"), 6, f"{key}.coarse_capture_tcp_pose_m_rad")
    fine_reference = hole.get("fine_reference")
    if fine_reference is not None:
        if not isinstance(fine_reference, dict):
            raise ValueError(f"孔位地图条目 {key} 的 fine_reference 不是对象")
        fine_status = str(fine_reference.get("status", "")).lower()
        if fine_status in {"ready", "coarse_fallback"}:
            _finite_vector(
                fine_reference.get("hole_center_base_mm"),
                3,
                f"{key}.fine_reference.hole_center_base_mm",
            )
        _optional_vector(
            fine_reference.get("fine_plane_intersection_mm"),
            3,
            f"{key}.fine_reference.fine_plane_intersection_mm",
        )
        _optional_vector(
            fine_reference.get("fine_capture_tcp_pose_m_rad"),
            6,
            f"{key}.fine_reference.fine_capture_tcp_pose_m_rad",
        )
        _optional_vector(
            fine_reference.get("coarse_to_fine_xy_delta_mm"),
            2,
            f"{key}.fine_reference.coarse_to_fine_xy_delta_mm",
        )
    return hole_id


def _validate_common_hole_map_fields(payload: dict[str, Any]) -> None:
    if payload.get("status") not in {"valid", "partial"}:
        raise ValueError(f"孔位地图状态不可执行：{payload.get('status')}")
    if payload.get("coordinate_frame") != "robot_base_mm":
        raise ValueError("孔位地图坐标系不是 robot_base_mm")
    if payload.get("target_policy") != "coarse_map_requires_fresh_fine":
        raise ValueError("地图不是纯粗定位地图，禁止正常调用；请重新建立粗定位地图")
    if payload.get("requires_fresh_fine") is not True:
        raise ValueError("粗定位地图缺少 requires_fresh_fine=true")


def validate_rotary_sector_map(
    payload: dict[str, Any],
    *,
    sector_id: int | None = None,
    requested_hole_ids: Iterable[int] | None = None,
    expected_target_mode: str | None = None,
) -> list[int]:
    """校验 v4 六扇区粗定位地图并返回可用孔号。"""
    del expected_target_mode
    version = int(payload.get("schema_version", -1))
    if version == LEGACY_ROTARY_HOLE_MAP_SCHEMA_VERSION:
        raise ValueError("旧 v3 地图保存了最终TCP目标，不能直接执行；请重新建立粗定位地图")
    if version != ROTARY_HOLE_MAP_SCHEMA_VERSION:
        raise ValueError(f"不支持的旋转扇区地图版本：{payload.get('schema_version')}")
    if payload.get("kind") != "coarse_sector_hole_map":
        raise ValueError("地图不是 coarse_sector_hole_map")
    if payload.get("scope") != "rotary_six_sector":
        raise ValueError("旋转扇区地图 scope 不正确")
    _validate_common_hole_map_fields(payload)
    sectors = payload.get("sectors")
    if not isinstance(sectors, dict) or not sectors:
        raise ValueError("六扇区地图没有 sectors")
    if sector_id is not None:
        key = sector_key(validate_sector_id(sector_id))
        selected_sectors = {key: sectors.get(key)}
    else:
        selected_sectors = sectors
    requested = None if requested_hole_ids is None else [int(item) for item in requested_hole_ids]
    ready_ids: list[int] = []
    for key, sector in selected_sectors.items():
        if sector is None:
            raise ValueError(f"扇区 {key} 尚未建立粗定位地图")
        if not isinstance(sector, dict):
            raise ValueError(f"扇区 {key} 不是对象")
        try:
            sid = int(sector["sector_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"扇区 {key} 缺少有效 sector_id") from exc
        if key != sector_key(sid):
            raise ValueError(f"扇区键 {key} 与 sector_id={sid} 不一致")
        if sector.get("target_policy") != "coarse_map_requires_fresh_fine":
            raise ValueError(f"扇区 {key} 不是纯粗定位扇区")
        holes = sector.get("holes")
        if not isinstance(holes, dict) or not holes:
            if sector_id is not None:
                raise ValueError(f"扇区 {key} 没有孔位")
            continue
        sector_ready = str(sector.get("status", "")).lower() in ROTARY_SECTOR_READY_STATUSES
        for hole_key, hole in holes.items():
            hole_id = _validate_hole_record(str(hole_key), hole)
            if sector_ready and str(hole.get("status", "ready")) in HOLE_MAP_READY_STATUSES:
                ready_ids.append(hole_id)
    if requested is not None:
        if len(set(requested)) != len(requested):
            raise ValueError("请求调用的孔号不能重复")
        missing = [item for item in requested if item not in ready_ids]
        if missing:
            label = f"扇区{int(sector_id)}" if sector_id is not None else "六扇区地图"
            raise ValueError(f"请求调用的孔不在有效{label}中：{missing}")
        return requested
    return sorted(ready_ids)


def validate_hole_map(
    payload: dict[str, Any],
    *,
    requested_hole_ids: Iterable[int] | None = None,
    expected_target_mode: str | None = None,
    sector_id: int | None = None,
) -> list[int]:
    """校验地图结构并返回可用孔号。"""
    if not isinstance(payload, dict):
        raise ValueError("孔位地图根节点必须是对象")
    version = int(payload.get("schema_version", -1))
    kind = payload.get("kind")
    if kind == "rotary_six_sector_hole_map" and version == LEGACY_ROTARY_HOLE_MAP_SCHEMA_VERSION:
        raise ValueError("旧 v3 最终点地图不能直接执行，请重新建立粗定位地图")
    if version == ROTARY_HOLE_MAP_SCHEMA_VERSION or kind == "coarse_sector_hole_map":
        return validate_rotary_sector_map(
            payload,
            sector_id=sector_id,
            requested_hole_ids=requested_hole_ids,
            expected_target_mode=expected_target_mode,
        )
    if version == LEGACY_HOLE_MAP_SCHEMA_VERSION:
        raise ValueError("旧版地图可能含最终点结果，不能直接执行；请重新建立粗定位地图")
    if version != HOLE_MAP_SCHEMA_VERSION:
        raise ValueError(f"不支持的孔位地图版本：{payload.get('schema_version')}")
    if kind != "coarse_hole_map":
        raise ValueError("普通地图不是 coarse_hole_map")
    if sector_id is not None:
        raise ValueError("普通粗定位地图不支持 sector_id")
    del expected_target_mode
    if payload.get("scope") != "current_robot_cycle":
        raise ValueError("当前只允许调用同一机器人循环内建立的孔位地图")
    _validate_common_hole_map_fields(payload)
    holes = payload.get("holes")
    if not isinstance(holes, dict) or not holes:
        raise ValueError("孔位地图没有 holes")
    ready_ids: list[int] = []
    for key, hole in holes.items():
        hole_id = _validate_hole_record(str(key), hole)
        if str(hole.get("status", "ready")) in HOLE_MAP_READY_STATUSES:
            ready_ids.append(hole_id)
    requested = None if requested_hole_ids is None else [int(item) for item in requested_hole_ids]
    if requested is not None:
        if len(set(requested)) != len(requested):
            raise ValueError("请求调用的孔号不能重复")
        missing = [item for item in requested if item not in ready_ids]
        if missing:
            raise ValueError(f"请求调用的孔不在有效地图中：{missing}")
        return requested
    return sorted(ready_ids)


def get_sector(payload: dict[str, Any], sector_id: int) -> dict[str, Any]:
    if int(payload.get("schema_version", -1)) != ROTARY_HOLE_MAP_SCHEMA_VERSION:
        raise ValueError("只有六扇区 v4 粗定位地图支持 get_sector")
    key = sector_key(sector_id)
    sector = (payload.get("sectors") or {}).get(key)
    if not isinstance(sector, dict):
        raise KeyError(f"六扇区地图中不存在 {key}")
    validate_rotary_sector_map(payload, sector_id=int(sector_id))
    return sector


def get_hole(payload: dict[str, Any], hole_id: int, sector_id: int | None = None) -> dict[str, Any]:
    key = f"H{int(hole_id):02d}"
    if int(payload.get("schema_version", -1)) == ROTARY_HOLE_MAP_SCHEMA_VERSION:
        sectors = payload.get("sectors") or {}
        selected = sectors
        if sector_id is not None:
            selected = {sector_key(sector_id): sectors.get(sector_key(sector_id))}
        candidates: list[dict[str, Any]] = []
        for sector in selected.values():
            if isinstance(sector, dict):
                hole = (sector.get("holes") or {}).get(key)
                if isinstance(hole, dict) and int(hole.get("hole_id", -1)) == int(hole_id):
                    candidates.append(hole)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1 and sector_id is None:
            raise KeyError(f"孔 H{int(hole_id):02d} 在多个扇区存在，请指定 sector_id")
        raise KeyError(f"扇区地图中不存在有效孔 H{int(hole_id):02d}")
    hole = (payload.get("holes") or {}).get(key)
    if not isinstance(hole, dict) or int(hole.get("hole_id", -1)) != int(hole_id):
        raise KeyError(f"孔位地图中不存在有效孔 H{int(hole_id):02d}")
    return hole


def publish_current_hole_map(
    payload: dict[str, Any],
    map_path: str | Path,
    current_path: str | Path,
) -> Path | None:
    """把可用粗定位地图发布为当前入口。"""
    validate_hole_map(payload)
    is_rotary_map = int(payload.get("schema_version", -1)) == ROTARY_HOLE_MAP_SCHEMA_VERSION
    if payload.get("status") != "valid" and not (
        is_rotary_map and bool((payload.get("quality_summary") or {}).get("ready_sector_ids"))
    ):
        return None
    target = Path(map_path).expanduser().resolve()
    current = Path(current_path).expanduser().resolve()
    root = current.parent
    if current.name.lower() != "current.json":
        raise ValueError(f"当前地图入口必须命名为 current.json：{current}")
    try:
        relative_map = target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"地图文件必须位于地图目录内：{target}") from exc
    if target.name.lower() != "hole_map.json" or len(relative_map.parts) != 2:
        raise ValueError(f"地图文件必须是版本目录下的 hole_map.json：{target}")
    pointer = {
        "kind": CURRENT_HOLE_MAP_POINTER_KIND,
        "schema_version": CURRENT_HOLE_MAP_POINTER_SCHEMA_VERSION,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "map_id": payload.get("map_id"),
        "map_status": payload.get("status"),
        "map_path": relative_map.as_posix(),
        "source_run_id": payload.get("source_run_id"),
    }
    return atomic_write_json(current, pointer)


def save_hole_map(payload: dict[str, Any], path: str | Path) -> Path:
    """校验并原子保存地图。"""
    validate_hole_map(payload)
    return atomic_write_json(path, jsonable(payload))


def load_hole_map(path: str | Path) -> dict[str, Any]:
    target = resolve_hole_map_path(path)
    if not target.is_file():
        raise FileNotFoundError(f"孔位地图不存在：{target}")
    payload = _read_json_object(target)
    validate_hole_map(payload)
    return payload
