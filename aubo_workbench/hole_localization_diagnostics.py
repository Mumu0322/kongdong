"""Run-comparison diagnostics for hole localization."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from aubo_workbench.geometry import unit_vector
from aubo_workbench.hole_localization_models import TwoStageConfig
from aubo_workbench.hole_localization_report import summarize_timing_events


_unit = unit_vector


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
    """统一统计方案效率，并按事件区间避免父子计时重复相加。"""
    return summarize_timing_events(events)


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
            "joint_enabled": bool(result.get("batch_fine_joint_enabled", False)),
            "joint_applied": bool(result.get("batch_fine_joint_applied", False)),
            "joint_success": bool(
                (result.get("batch_fine_joint_summary") or {}).get("success", False)
            ),
            "joint_translation_mm": (
                (result.get("batch_fine_joint_summary") or {}).get("translation_mm")
            ),
            "joint_yaw_deg": (
                (result.get("batch_fine_joint_summary") or {}).get("yaw_deg")
            ),
            "joint_residual_p95_mm": (
                (result.get("batch_fine_joint_summary") or {}).get(
                    "residual_p95_mm",
                    (result.get("batch_fine_joint_summary") or {}).get(
                        "frame_residual_p95_mm"
                    ),
                )
            ),
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
            "batch_fine_joint_requested": bool(
                cfg.batch_fine_localization and cfg.batch_fine_joint_localization
            ),
            "batch_fine_joint_method": (
                "per_hole_temporal_robust_fusion_then_group_xy_transform"
            ),
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
            "joint_residual_p95_mm": _metric_distribution(
                metric("fine_quality.joint_residual_p95_mm")
            ),
            "joint_translation_norm_mm": _metric_distribution(
                [
                    None if value is None else float(np.linalg.norm(value))
                    for value in metric("fine_quality.joint_translation_mm")
                ]
            ),
            "final_xy_planned_actual_error_mm": _metric_distribution(metric("motion_quality.final_xy_planned_actual_error_mm")),
        },
        "effective_timing": _effective_timing_summary(
            list((report.get("timing") or {}).get("events") or []),
        ),
    }
