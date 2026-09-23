"""Per-hole execution for the sequential multi-hole workflow.

Shared captures only provide validated geometry. This module handles per-hole
fallbacks, ChArUco-aware final point construction, safe motion, and result
recording.
"""

from __future__ import annotations

from aubo_workbench.localization_errors import LocalizationHardwareError
from typing import Any


_RUNTIME_DEPENDENCIES = {
    'CHARUCO_XY_MODEL_BIAS_MM',
    'CHARUCO_XY_MODEL_MATRIX',
    'CHARUCO_XY_MODEL_READY',
    'CHARUCO_XY_MODEL_SOURCE',
    'COARSE_SURFACE_MODEL',
    'COARSE_SURFACE_SELECTION_POLICY',
    'CacheValidationGates',
    'FINAL_BASE_Y_AFTER_Z_MM',
    'FINAL_TOOL_Y_AFTER_Z_MM',
    'HOLE_DIAMETERS_MM',
    'SHARED_OBSERVATION_MIN_DESCENT_MM',
    'THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM',
    '_angle_deg',
    '_append_deferred_hole_result',
    '_apply_cached_geometry_to_hole',
    '_apply_coarse_geometry_to_hole',
    '_batch_coarse_localization_at_340mm',
    '_batch_fine_localization_at_260mm',
    '_build_comparison_hole_diagnostics',
    '_cache_entry_from_observations',
    '_cache_measurements_from_observations',
    '_capture_coarse_burst',
    '_capture_fine_with_recovery',
    '_fuse_fine',
    '_compose_batch_fine_joint_xy_with_tilt',
    '_confirm_and_move_line',
    '_confirm_next_hole_if_needed',
    '_matrix_to_rpy_zyx',
    '_move_to_batch_final_tcp_direct',
    '_move_to_fine_pose',
    '_move_to_sequential_coarse_pose',
    '_move_to_shared_coarse_pose',
    '_move_to_shared_fine_pose',
    '_observation_rows',
    '_plan_batch_coarse_group_pose',
    '_plan_hole_tcp_pose_fixed_rz',
    '_project_base_point_to_pixel',
    '_record_hole_tracking_event',
    '_refine_shared_coarse_group_pose',
    '_request_next_hole_confirmation',
    '_require_safe_snapshot',
    '_reuse_initial_pointcloud_geometry_for_batch_fine',
    '_save_batch_fine_final_result_overlay',
    '_save_coarse_pointcloud_image',
    '_save_group_capture_visualization',
    '_save_grouping_plan_visualization',
    '_settle_and_discard_coarse_recapture_frames',
    '_split_batch_fine_supplement_groups',
    '_split_batch_localization_groups',
    '_split_shared_cache_validation_groups',
    '_unit',
    '_unpack_batch_grouping_result',
    '_upsert_persistent_cache_entry',
    '_validate_coarse_cache_at_current_pose',
    '_write_progress_checkpoint',
    '_write_report',
    'base_z_target_for_camera_height',
    'camera_height_to_plane_mm',
    'camera_transform',
    'compose_batch_fine_xy_with_coarse_z',
    'correct_projected_circle_center',
    'fuse_batch_fine_xy_with_pointcloud_prior',
    'init_pipeline',
    'load_cache_entries',
    'load_persistent_cache_entries',
    'math',
    'np',
    'optimize_hole_order',
    'pixel_to_base_plane',
    'plan_final_tcp_base_z',
    'plan_final_tcp_combined_y_trim',
    'plan_final_tcp_xy',
    'rekey_cache_entry',
    'replace',
    'save_cache_entries',
    'save_persistent_cache_entries',
    'time',
    'transform_cached_points_to_camera',
    'transform_to_sdk_pose_m_rad',
    'validate_cache_entry',
}


def install_runtime(symbols: dict[str, object]) -> None:
    for name in _RUNTIME_DEPENDENCIES:
        if name in symbols:
            globals()[name] = symbols[name]


def _persist_planned_final_point(
    ctx: Any,
    hole: dict[str, Any],
    visual_point_base: np.ndarray,
    target_point_base: np.ndarray,
    planned_tcp: np.ndarray,
    *,
    motion_path: str,
) -> dict[str, Any]:
    """Durably record the computed target before issuing any final motion."""
    visual = np.asarray(visual_point_base, dtype=np.float64).reshape(3)
    target = np.asarray(target_point_base, dtype=np.float64).reshape(3)
    tcp = np.asarray(planned_tcp, dtype=np.float64).reshape(4, 4)
    if not (np.isfinite(visual).all() and np.isfinite(target).all() and np.isfinite(tcp).all()):
        raise ValueError("最终点规划包含非有限坐标，拒绝下发运动")
    plan = {
        "visual_hole_center_base_mm": visual.tolist(),
        "target_point_base_mm": target.tolist(),
        "planned_final_tcp_xyz_mm": tcp[:3, 3].tolist(),
        "planned_final_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(tcp),
        "motion_path": str(motion_path),
    }
    hole["planned_final_point"] = plan
    stages = ctx.report.setdefault("stages", {})
    stages.setdefault(f"hole_{int(hole['hole_id'])}", {})["planned_final_point"] = plan
    active = stages.get("active_hole")
    if isinstance(active, dict):
        active["planned_final_point"] = plan
    _write_report(ctx.run_dir, ctx.report, ctx.rows, timing=ctx.timing)
    print(
        f"[FINAL_PLAN] 孔{int(hole['hole_id'])} 视觉孔中心={np.round(visual, 3).tolist()} mm; "
        f"目标点={np.round(target, 3).tolist()} mm; "
        f"计划TCP={np.round(tcp[:3, 3], 3).tolist()} mm（尚未到位）",
        flush=True,
    )
    return plan


def _execute_per_hole_final_motion(
    hole_id: int,
    order: int,
    current_tcp: np.ndarray,
    xy_target: np.ndarray,
    final_target: np.ndarray,
    compensation: str,
    args: Any,
    motion_session: Any,
    pose_session: Any,
    timing: Any,
    *,
    force_safe_path: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Choose the final route from measured base Z and pose-change clearance."""
    actual = np.asarray(current_tcp, dtype=np.float64).reshape(4, 4).copy()
    final = np.asarray(final_target, dtype=np.float64).reshape(4, 4).copy()
    if force_safe_path or float(actual[2, 3]) < float(final[2, 3]):
        with timing.measure(
            f"hole_{hole_id:02d}/final_motion_safe_z_xy_z",
            hole_id=hole_id,
            processing_order=order,
        ):
            reached = _move_to_batch_final_tcp_direct(
                str(hole_id), actual, final, args, motion_session, pose_session,
            )
        return reached, None, "safe_z_lift_xy_guarded_z_descent"

    with timing.measure(
        f"hole_{hole_id:02d}/final_motion_xy",
        hole_id=hole_id,
        processing_order=order,
    ):
        after_xy = _confirm_and_move_line(
            f"孔{hole_id}精定位后移动到最终XY",
            actual, xy_target, args, motion_session, pose_session,
            f"保持孔{hole_id}精拍Z与姿态；{compensation}；"
            "最终点使用定位结果，不附加X/Z偏移",
            require_confirmation=False,
            motion_profile="precision",
        )
    z_target = plan_final_tcp_base_z(after_xy, final[:3, 3])
    with timing.measure(
        f"hole_{hole_id:02d}/final_motion_z",
        hole_id=hole_id,
        processing_order=order,
    ):
        reached = _confirm_and_move_line(
            f"孔{hole_id}移动到最终Z",
            after_xy, z_target, args, motion_session, pose_session,
            "最终点不附加基坐标Z偏移；"
            f"目标TCP基坐标Z={z_target[2, 3]:.3f} mm",
            require_confirmation=False,
            motion_profile="precision",
        )
    return reached, after_xy, "xy_then_z"


def _plan_per_hole_final_target(
    current_tcp: np.ndarray,
    fine_capture_tcp: np.ndarray,
    target_point_base: np.ndarray,
    fixed_offset: tuple[float, float] | None,
    *,
    use_charuco_model: bool,
    precaptured: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep this hole's capture orientation even when final motion occurs later."""
    reference = np.asarray(
        fine_capture_tcp if precaptured else current_tcp, dtype=np.float64,
    ).reshape(4, 4).copy()
    xy_target, _ = plan_final_tcp_xy(
        reference, target_point_base, fixed_offset,
        use_charuco_model=use_charuco_model,
    )
    final_target = plan_final_tcp_base_z(xy_target, target_point_base)
    return reference, xy_target, final_target


def _per_hole_fine_route(args: Any, batch_fine_available: bool) -> str:
    """选择逐孔定位路线，确保同拍建图不进入260 mm运动分支。"""
    if bool(getattr(args, "coarse_direct_final", False)):
        return "coarse_direct_final"
    if bool(batch_fine_available):
        return "batch_fine_result"
    if bool(getattr(args, "map_build_coarse_only", False)):
        return "coarse_map_only"
    if bool(getattr(args, "map_build_same_capture_340", False)):
        return "same_capture_340"
    return "per_hole_fallback"


def _should_move_to_per_hole_fine(args: Any, batch_fine_available: bool) -> bool:
    """建图只记录粗定位结果，不能为了精定位兜底再移动到每个孔。"""
    return _per_hole_fine_route(args, batch_fine_available) == "per_hole_fallback"


def _apply_batch_coarse_result_to_hole(
    hole: dict[str, Any], batch_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """把共享批量粗定位结果复制到逐孔上下文，供各策略复用。"""
    hole["batch_coarse_center_px"] = batch_result.get("center_px")
    hole["batch_coarse_center_base_mm"] = batch_result.get("center_base_mm")
    hole["batch_coarse_plane_point_base_mm"] = batch_result.get(
        "plane_point_base_mm"
    )
    hole["batch_coarse_normal_toward_camera_base"] = batch_result.get(
        "normal_base"
    )
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
        "coarse_center_source": batch_result.get("center_source"),
        "coarse_center_source_counts": batch_result.get(
            "center_source_counts", {}
        ),
        "coarse_geometric_valid_frames": batch_result.get(
            "geometric_valid_frames", 0
        ),
        "coarse_geometric_center_scatter_p95_px": batch_result.get(
            "geometric_center_scatter_p95_px"
        ),
        "coarse_center_fusion_source": batch_result.get(
            "center_fusion_source"
        ),
        "coarse_tracking_distance_p95_px": batch_result.get(
            "tracking_distance_p95_px"
        ),
        "coarse_ring_points_median": batch_result.get("ring_points_median"),
        "coarse_ring_coverage_min_ratio": batch_result.get("ring_coverage_min_ratio"),
        "coarse_ring_max_gap_deg": batch_result.get("ring_max_gap_deg"),
        "coarse_quality_retry": batch_result.get("coarse_quality_retry", False),
        "coarse_surface_model": batch_result.get("surface_model"),
        "coarse_surface_selection_policy": batch_result.get(
            "surface_selection_policy"
        ),
        "coarse_front_surface_z_mm": batch_result.get("front_surface_z_mm"),
        "coarse_ring_points_raw_median": batch_result.get(
            "ring_points_raw_median"
        ),
        "coarse_surface_points_selected_median": batch_result.get(
            "surface_points_selected_median"
        ),
        "coarse_sphere_center_camera_mm": batch_result.get(
            "sphere_center_camera_mm"
        ),
        "coarse_sphere_radius_mm": batch_result.get("sphere_radius_mm"),
        "coarse_source": ("singleton_coarse_quality_retry" if batch_result.get("coarse_quality_retry")
                          else "batch_coarse_localization"),
        "batch_observed_center_base_mm": batch_result.get(
            "batch_observed_center_base_mm"
        ),
        "batch_observed_plane_point_base_mm": batch_result.get(
            "batch_observed_plane_point_base_mm"
        ),
        "batch_observed_normal_base": batch_result.get("batch_observed_normal_base"),
        "coarse_pointcloud_image_path": batch_result.get("pointcloud_image_path"),
        "batch_pointcloud_archive_path": batch_result.get(
            "batch_pointcloud_archive_path"
        ),
        "coarse_capture_height_mm": batch_result.get("capture_height_mm"),
        "batch_coarse_group_index": batch_result.get("batch_coarse_group_index"),
        "batch_coarse_group_hole_ids": batch_result.get(
            "batch_coarse_group_hole_ids"
        ),
    })
    for key in (
        "boundary_class", "boundary_layer", "boundary_component_index",
        "boundary_edge_score_deg", "boundary_local_degree", "boundary_distance",
        "boundary_distance_mm", "boundary_distance_px", "boundary_distance_unit",
        "boundary_coordinate_source", "boundary_classification_reason",
        "boundary_override", "boundary_override_source", "group_phase",
        "group_boundary_class",
    ):
        if key in batch_result:
            hole[key] = batch_result[key]
    return list(batch_result.get("coarse_captures", []))


def _commit_completed_hole_result(
    ctx: Any, order: int, hole: dict[str, Any], result: dict[str, Any],
) -> None:
    """写入一个已完成孔的统一进度记录。"""
    for key in (
        "initial_selection_order", "operator_selection_order", "selection_source",
        "numbering_policy", "map_order", "layout_row", "layout_column",
        "numbering_center_px", "numbering_row_tolerance_px", "map_hole_key",
        "global_hole_key", "sector_id",
        "boundary_class", "boundary_layer", "boundary_component_index",
        "boundary_edge_score_deg", "boundary_local_degree", "boundary_distance",
        "boundary_distance_mm", "boundary_distance_px", "boundary_distance_unit",
        "boundary_coordinate_source", "boundary_classification_reason",
        "boundary_override", "boundary_override_source", "group_phase",
        "group_boundary_class", "edge_first_group_index",
        "edge_first_phase_group_index", "batch_coarse_group_index",
        "batch_coarse_group_hole_ids",
    ):
        if result.get(key) is None and hole.get(key) is not None:
            result[key] = hole.get(key)
    result["comparison_diagnostics"] = _build_comparison_hole_diagnostics(result)
    hole["final_result"] = result
    results = ctx.results
    results.append(result)
    report = ctx.report
    hole_id = int(result["hole_id"])
    report["stages"][f"hole_{hole_id}"] = result
    completed_results = [item for item in results if item.get("status") == "completed"]
    capture_only_results = [item for item in results if item.get("status") == "capture_only"]
    deferred_results = [
        item for item in results if item.get("status", "").startswith("deferred_")
    ]
    report["stages"]["processed_holes"] = {
        "completed_count": len(completed_results),
        "capture_only_count": len(capture_only_results),
        "deferred_count": len(deferred_results),
        "total_count": len(results),
        "hole_order": [int(item["hole_id"]) for item in results],
        "holes": results,
    }
    _write_progress_checkpoint(ctx.run_dir, report, timing=ctx.timing)


def _defer_coarse_direct_without_pointcloud(
    ctx: Any,
    order: int,
    hole: dict[str, Any],
    reason: str,
) -> None:
    """第四策略没有点云中心时只记录失败，不转入其它定位流程。"""
    hole_id = int(hole["hole_id"])
    capture_height_mm = float(getattr(
        ctx.cfg, "coarse_height_mm",
        getattr(ctx.cfg, "coarse_direct_final_height_mm", 340.0),
    ))
    capture_height_label = f"{capture_height_mm:g}"
    result = {
        "status": "deferred_coarse_pointcloud",
        "failure_type": "pointcloud_center_unavailable",
        "error": str(reason),
        "hole_id": hole_id,
        "processing_order": int(order),
        "tracking_identity": hole.get("tracking_identity"),
        "initial_selection_order": hole.get("initial_selection_order"),
        "initial_center_px": hole.get("initial_center_px"),
        "initial_center_base_mm": hole.get("initial_center_base_mm"),
        "initial_plane_normal_base": hole.get("initial_plane_normal_base"),
        "coarse_source": "batch_coarse_pointcloud_only",
        "batch_coarse_requested": bool(getattr(ctx, "batch_coarse_for_cache", False)),
        "batch_fine_requested": False,
        "batch_fine_source": "disabled_coarse_direct_pointcloud_only",
        "fine_stage_skipped": True,
        "fine_quality_status": "deferred_coarse_pointcloud",
        "fine_quality_note": str(reason),
        "localization_path": f"coarse_{capture_height_label}_pointcloud_only_unavailable",
        "coarse_capture_height_mm": capture_height_mm,
        "coarse_direct_decision": "pointcloud_unavailable",
        "coarse_captures": hole.get("coarse_captures", []),
        "coarse_cache_event": hole.get("coarse_cache_event"),
        "tracking_events": hole.get("tracking_events", []),
        "final_xy_motion": None,
        "final_z_motion": None,
        "final_motion_direct": None,
        "target_point_base_mm": None,
        "hole_center_base_mm": None,
        "timing": ctx.timing.scoped_snapshot(f"hole_{hole_id:02d}/"),
    }
    _append_deferred_hole_result(
        hole, result, ctx.results, ctx.report, ctx.run_dir, ctx.timing,
    )


def _run_coarse_direct_final(
    ctx: Any,
    order: int,
    hole: dict[str, Any],
    current_tcp: np.ndarray,
    *,
    cache_event: dict[str, Any],
    coarse_captures: list[dict[str, Any]],
    batch_coarse_for_cache: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """用第四策略点云中心直接规划最终点，完全跳过260 mm精定位。"""
    args = ctx.args
    handeye = ctx.handeye
    cfg = ctx.cfg
    timing = ctx.timing
    report = ctx.report
    hole_id = int(hole["hole_id"])
    fixed_rz_rad = ctx.fixed_rz_rad
    capture_height_mm = float(getattr(
        cfg, "coarse_height_mm",
        getattr(cfg, "coarse_direct_final_height_mm", 340.0),
    ))
    capture_height_label = f"{capture_height_mm:g}"
    capture_only = bool(getattr(args, "coarse_direct_final_capture_only", False))

    coarse_center = np.asarray(
        hole["coarse_center_base_mm"], dtype=np.float64,
    ).reshape(3).copy()
    coarse_plane = np.asarray(
        hole.get("coarse_plane_point_base_mm", coarse_center), dtype=np.float64,
    ).reshape(3).copy()
    coarse_normal = _unit(
        np.asarray(hole["coarse_normal_toward_camera_base"], dtype=np.float64),
        f"hole {hole_id} coarse direct normal",
    )
    if not np.isfinite(coarse_center).all() or not np.isfinite(coarse_plane).all():
        raise RuntimeError(f"孔{hole_id}粗定位中心或平面不是有限坐标")

    # 即使批量粗定位使用的是共同观察位，也为当前孔生成一个只用于最终
    # 姿态/高度规划的参考位姿；不去该位姿重新拍摄。
    reference_tcp, reference_geometry = _plan_hole_tcp_pose_fixed_rz(
        coarse_center,
        coarse_normal,
        np.asarray(current_tcp, dtype=np.float64),
        handeye.T_tcp_rgb_camera,
        fixed_rz_rad=fixed_rz_rad,
        camera_height_mm=capture_height_mm,
    )
    hole["coarse_direct_reference_tcp_pose_m_rad"] = transform_to_sdk_pose_m_rad(
        reference_tcp
    )
    hole["coarse_direct_reference_pose_geometry"] = reference_geometry

    try:
        estimated_height = camera_height_to_plane_mm(
            reference_tcp, handeye.T_tcp_rgb_camera, coarse_plane,
        )
    except Exception:
        estimated_height = None

    pose_point_base = coarse_center.copy()
    final_target_point_base = pose_point_base.copy()
    explicit_fixed_offset = getattr(args, "tcp_xy_offset_mm", None)
    use_charuco_model = bool(getattr(args, "use_charuco_xy_correction", True))
    if explicit_fixed_offset is not None:
        raise RuntimeError(
            "第四策略不接受TCP XY固定补偿"
        )
    if capture_only:
        # 评估模式只报告“如果执行最终动作会到哪里”，不调用补偿后的
        # 运动目标，也不要求现场已有可用的ChArUco模型。
        planned_xy_target = np.asarray(reference_tcp, dtype=np.float64).copy()
        planned_final_tcp = np.asarray(reference_tcp, dtype=np.float64).copy()
        correction_xy = None
        compensation = "仅采集评估：未执行最终动作"
    else:
        planned_xy_target, _ = plan_final_tcp_xy(
            reference_tcp, final_target_point_base,
            use_charuco_model=use_charuco_model,
        )
        planned_final_tcp = plan_final_tcp_base_z(
            planned_xy_target, final_target_point_base,
        )
        correction_xy = planned_xy_target[:2, 3] - final_target_point_base[:2]
        compensation = (
            f"ChArUco仿射模型修正={np.round(correction_xy, 3).tolist()} mm"
            if use_charuco_model else "不使用ChArUco纠偏"
        )
    hole["coarse_direct_final_target_tcp_pose_m_rad"] = transform_to_sdk_pose_m_rad(
        planned_final_tcp
    )

    final_xy_motion: dict[str, Any] | None = None
    final_z_motion: dict[str, Any] | None = None
    final_motion_direct: dict[str, Any] | None = None
    motion_start_tcp = np.asarray(current_tcp, dtype=np.float64).copy()
    direct_safe_z = max(
        float(motion_start_tcp[2, 3]),
        float(planned_final_tcp[2, 3]) + THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM,
    )
    direct_guard_z = float(planned_final_tcp[2, 3]) + SHARED_OBSERVATION_MIN_DESCENT_MM

    if bool(getattr(args, "move_final_xy", False)) and not capture_only:
        _persist_planned_final_point(
            ctx, hole, coarse_center, final_target_point_base, planned_final_tcp,
            motion_path="coarse_direct_safe_final_tcp",
        )
        with timing.measure(
            f"hole_{hole_id:02d}/coarse_direct_final_motion",
            hole_id=hole_id,
            processing_order=order,
        ):
            current_tcp = _move_to_batch_final_tcp_direct(
                str(hole_id), motion_start_tcp, planned_final_tcp,
                args, ctx.motion_session, ctx.pose_session,
            )
        final_xy_motion = {
            "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(planned_final_tcp),
            "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
            "tcp_position_before_mm": motion_start_tcp[:3, 3].copy(),
            "hole_center_base_mm": coarse_center,
            "pose_point_base_mm": pose_point_base,
            "target_point_base_mm": final_target_point_base,
            "batch_fine_joint_applied": False,
            "compensation_mode": "charuco_affine_model" if use_charuco_model else "none",
            "xy_correction_mm": correction_xy,
            "charuco_model_source": str(CHARUCO_XY_MODEL_SOURCE) if use_charuco_model else None,
            "charuco_model_matrix_2x2": CHARUCO_XY_MODEL_MATRIX if use_charuco_model else None,
            "charuco_model_bias_mm": CHARUCO_XY_MODEL_BIAS_MM if use_charuco_model else None,
            "tcp_xy_offset_mm": None,
            "motion_path": "coarse_direct_safe_final_tcp",
            "reference_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(reference_tcp),
            "planned_safe_z_mm": direct_safe_z,
            "planned_guard_z_mm": direct_guard_z,
            "separate_y_trim_executed": False,
            "compensation_description": compensation,
        }
        final_z_motion = {
            "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(planned_final_tcp),
            "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
            "target_base_z_mm": float(planned_final_tcp[2, 3]),
            "target_point_base_mm": final_target_point_base,
            "delta_base_z_mm": float(planned_final_tcp[2, 3] - reference_tcp[2, 3]),
            "motion_frame": "base_z_only_guard10_then_precision10",
            "descent_guard_mm": SHARED_OBSERVATION_MIN_DESCENT_MM,
        }
        final_motion_direct = {
            "path_policy": (
                "pure_z_lift_safe_horizontal_to_coarse_direct_final_tcp_"
                "pure_z_descent_guard10_then_pure_z_final10"
            ),
            "reference_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(reference_tcp),
            "planned_final_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(planned_final_tcp),
            "planned_safe_z_mm": direct_safe_z,
            "planned_guard_z_mm": direct_guard_z,
            "safe_margin_mm": THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM,
            "descent_guard_mm": SHARED_OBSERVATION_MIN_DESCENT_MM,
            "post_final_y_offset_applied": False,
            "actual_final_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
        }

    coarse_source = "batch_coarse_pointcloud"

    coarse_capture_pose = hole.get("coarse_capture_tcp_pose_m_rad")
    if coarse_capture_pose is None:
        map_source = hole.get("coarse_map_source")
        if isinstance(map_source, dict):
            coarse_capture_pose = map_source.get("coarse_capture_tcp_pose_m_rad")
    if coarse_capture_pose is None:
        for capture in reversed(coarse_captures):
            if isinstance(capture, dict) and capture.get("tcp_pose_m_rad") is not None:
                coarse_capture_pose = capture["tcp_pose_m_rad"]
                break
    if coarse_capture_pose is None:
        coarse_capture_pose = transform_to_sdk_pose_m_rad(reference_tcp)

    result: dict[str, Any] = {
        "status": "capture_only" if capture_only else "completed",
        "hole_id": hole_id,
        "processing_order": order,
        "tracking_identity": hole["tracking_identity"],
        "initial_selection_order": hole.get("initial_selection_order"),
        "initial_center_px": hole.get("initial_center_px"),
        "initial_center_base_mm": hole.get("initial_center_base_mm"),
        "initial_plane_normal_base": hole.get("initial_plane_normal_base"),
        "batch_coarse_requested": bool(batch_coarse_for_cache),
        "batch_fine_requested": False,
        "batch_fine_source": "disabled_coarse_direct_pointcloud_only",
        "coarse_source": coarse_source,
        "fine_center_source": "pointcloud_center_base_mm",
        "fine_center_source_counts": {},
        "fine_quality_status": "coarse_direct_capture_only" if capture_only else "coarse_direct",
        "fine_quality_note": (
            f"第四策略使用{capture_height_label} mm点云中心；"
            + (
                "仅采集评估，未执行最终XY/Z动作，未采集260 mm精定位"
                if capture_only else
                "直接规划最终点，未采集260 mm精定位"
            )
        ),
        "fine_recovery_attempts": [],
        "localization_path": f"coarse_{capture_height_label}_direct",
        "fine_stage_skipped": True,
        "coarse_direct_decision": (
            "capture_only" if capture_only else hole.get(
                "coarse_direct_decision", "coarse_direct"
            )
        ),
        "batch_coarse_center_base_mm": hole.get(
            "batch_coarse_center_base_mm"
        ),
        "hole_center_base_mm": coarse_center,
        "hole_center_base_naive_mm": coarse_center.copy(),
        "hole_result_type": "base_frame_3d_point",
        "final_pose_source": (
            "coarse_center_capture_only_no_final_motion"
            if capture_only else "coarse_center_direct_with_charuco"
        ),
        "batch_fine_xy_only": False,
        "coarse_fine_reference_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(reference_tcp),
        "coarse_direct_reference_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(reference_tcp),
        "coarse_direct_final_target_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(planned_final_tcp),
        "coarse_capture_height_mm": capture_height_mm,
        "batch_coarse_group_index": hole.get("batch_coarse_group_index"),
        "batch_coarse_group_hole_ids": hole.get("batch_coarse_group_hole_ids"),
        "coarse_direct_final_capture_only": capture_only,
        "charuco_compensation_applied": bool(final_xy_motion is not None and use_charuco_model),
        "charuco_model_ready": bool(globals().get("CHARUCO_XY_MODEL_READY", False)),
        "charuco_xy_correction_mm": correction_xy,
        "target_point_base_mm": final_target_point_base,
        "planned_final_point": hole.get("planned_final_point"),
        "coarse_center_base_mm": coarse_center,
        "coarse_center_camera_mm": hole.get("coarse_center_camera_mm"),
        "pointcloud_center_base_mm": coarse_center.copy(),
        "pointcloud_center_camera_mm": hole.get("coarse_center_camera_mm"),
        "pointcloud_center_definition": (
            "coarse_center_ray_intersection_with_fused_local_pointcloud_plane"
        ),
        "coarse_plane_point_base_mm": coarse_plane,
        "coarse_plane_point_camera_mm": hole.get("coarse_plane_point_camera_mm"),
        "coarse_normal_camera": hole.get("coarse_normal_camera"),
        "coarse_normal_toward_camera_base": coarse_normal,
        "coarse_plane_rmse_mm": hole.get("coarse_plane_rmse_mm"),
        "coarse_valid_frames": hole.get("coarse_valid_frames"),
        "coarse_total_frames": hole.get("coarse_total_frames"),
        "valid_frames": hole.get("coarse_valid_frames"),
        "total_frames": hole.get("coarse_total_frames"),
        "coarse_center_scatter_p95_px": hole.get("coarse_center_scatter_p95_px"),
        "coarse_tracking_distance_p95_px": hole.get("coarse_tracking_distance_p95_px"),
        "coarse_ring_points_median": hole.get("coarse_ring_points_median"),
        "coarse_ring_coverage_min_ratio": hole.get("coarse_ring_coverage_min_ratio"),
        "coarse_ring_max_gap_deg": hole.get("coarse_ring_max_gap_deg"),
        "coarse_quality_retry": hole.get("coarse_quality_retry"),
        "coarse_surface_model": hole.get("coarse_surface_model"),
        "coarse_surface_selection_policy": hole.get("coarse_surface_selection_policy"),
        "coarse_front_surface_z_mm": hole.get("coarse_front_surface_z_mm"),
        "coarse_ring_points_raw_median": hole.get("coarse_ring_points_raw_median"),
        "coarse_surface_points_selected_median": hole.get(
            "coarse_surface_points_selected_median"
        ),
        "coarse_sphere_center_camera_mm": hole.get("coarse_sphere_center_camera_mm"),
        "coarse_sphere_radius_mm": hole.get("coarse_sphere_radius_mm"),
        "coarse_capture_tcp_pose_m_rad": coarse_capture_pose,
        "coarse_captures": hole.get("coarse_captures", coarse_captures),
        "coarse_cache_event": cache_event,
        "coarse_quality_recovery": hole.get("coarse_quality_recovery"),
        "pointcloud_segmentation": hole.get("pointcloud_segmentation"),
        "fine_plane_intersection_mm": coarse_center.copy(),
        "fine_xy_source": "coarse_center_direct",
        "fine_z_source": "coarse_center_z",
        "tilt_center_correction": None,
        "estimated_height_mm": estimated_height,
        "diameter_estimate_mm": hole.get("diameter_estimate_mm"),
        "matched_diameter_mm": hole.get("matched_diameter_mm"),
        "plane_normal_toward_camera_base": coarse_normal,
        "fixed_rz_rad": fixed_rz_rad,
        "shared_batch_capture_tcp_pose_m_rad": None,
        "fine_tcp_pose_m_rad": None,
        "final_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
        "final_xy_motion": final_xy_motion,
        "final_z_motion": final_z_motion,
        "final_y_trim_motion": None,
        "final_motion_direct": final_motion_direct,
        "final_motion_path": (
            "coarse_direct_safe_final_tcp" if final_motion_direct is not None else None
        ),
        "capture_only_reason": (
            "evaluation_only_no_final_xy_z_motion"
            if capture_only else None
        ),
        "planned_final_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(planned_final_tcp),
        "tcp_target_pose_m_rad": (
            transform_to_sdk_pose_m_rad(planned_final_tcp)
            if final_xy_motion is not None else None
        ),
        "tracking_events": hole.get("tracking_events", []),
        "initial_detection": hole.get("initial_detection"),
        "timing": timing.scoped_snapshot(f"hole_{hole_id:02d}/"),
    }
    return np.asarray(current_tcp, dtype=np.float64), result


def _same_capture_pointcloud_fallback_validation(
    hole: dict[str, Any],
    cfg: Any,
    *,
    map_build_enabled: bool,
    motion_disabled: bool,
    same_capture_available: bool,
) -> dict[str, Any]:
    """Fail closed unless the accepted 340 mm point-cloud center is map-safe."""
    captures = hole.get("coarse_captures") or []
    accepted_capture = next((
        item for item in reversed(captures)
        if isinstance(item, dict)
        and str(item.get("status", "")).startswith("accepted")
    ), None)

    def finite_float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    coarse_center = hole.get("coarse_center_base_mm")
    try:
        center = np.asarray(coarse_center, dtype=np.float64).reshape(3)
        center_is_finite = bool(np.isfinite(center).all())
    except (TypeError, ValueError):
        center = None
        center_is_finite = False

    metrics = {
        "center_offset_px": finite_float(
            accepted_capture.get("center_offset_px") if accepted_capture else None
        ),
        "normal_error_deg": finite_float(
            accepted_capture.get("normal_error_deg") if accepted_capture else None
        ),
        "valid_frames": finite_float(hole.get("coarse_valid_frames")),
        "plane_rmse_mm": finite_float(hole.get("coarse_plane_rmse_mm")),
        "center_scatter_p95_px": finite_float(
            hole.get("coarse_center_scatter_p95_px")
        ),
        "ring_coverage_min_ratio": finite_float(
            hole.get("coarse_ring_coverage_min_ratio")
        ),
        "ring_max_gap_deg": finite_float(hole.get("coarse_ring_max_gap_deg")),
        "camera_height_mm": finite_float(
            hole.get("fine_height_estimate_mm")
            if hole.get("fine_height_estimate_mm") is not None
            else hole.get("estimated_height_mm")
        ),
    }
    thresholds = {
        "center_offset_max_px": float(cfg.center_tolerance_px),
        "normal_error_max_deg": float(cfg.normal_tolerance_deg),
        "valid_frames_min": int(cfg.min_coarse_valid),
        "plane_rmse_max_mm": float(cfg.max_plane_rmse_mm),
        "center_scatter_p95_max_px": float(cfg.max_coarse_center_scatter_p95_px),
        "ring_coverage_min_ratio": float(cfg.coarse_min_ring_coverage_ratio),
        "ring_max_gap_max_deg": float(cfg.coarse_max_ring_gap_deg),
        "camera_height_target_mm": float(cfg.coarse_height_mm),
        "camera_height_tolerance_mm": float(cfg.height_tolerance_mm),
    }
    checks = {
        "same_capture_340_map_build": bool(map_build_enabled),
        "final_motion_disabled": bool(motion_disabled),
        "same_capture_rgbd_present": bool(same_capture_available),
        "accepted_coarse_capture_present": accepted_capture is not None,
        "finite_pointcloud_center": center_is_finite,
        "camera_centered": (
            metrics["center_offset_px"] is not None
            and metrics["center_offset_px"] <= thresholds["center_offset_max_px"]
        ),
        "normal_aligned": (
            metrics["normal_error_deg"] is not None
            and metrics["normal_error_deg"] <= thresholds["normal_error_max_deg"]
        ),
        "enough_valid_frames": (
            metrics["valid_frames"] is not None
            and metrics["valid_frames"] >= thresholds["valid_frames_min"]
        ),
        "plane_fit_passed": (
            metrics["plane_rmse_mm"] is not None
            and metrics["plane_rmse_mm"] <= thresholds["plane_rmse_max_mm"]
        ),
        "center_scatter_passed": (
            metrics["center_scatter_p95_px"] is not None
            and metrics["center_scatter_p95_px"]
            <= thresholds["center_scatter_p95_max_px"]
        ),
        "ring_coverage_passed": (
            metrics["ring_coverage_min_ratio"] is not None
            and metrics["ring_coverage_min_ratio"]
            >= thresholds["ring_coverage_min_ratio"]
        ),
        "ring_gap_passed": (
            metrics["ring_max_gap_deg"] is not None
            and metrics["ring_max_gap_deg"] <= thresholds["ring_max_gap_max_deg"]
        ),
        "capture_height_is_340mm": (
            metrics["camera_height_mm"] is not None
            and abs(
                metrics["camera_height_mm"] - thresholds["camera_height_target_mm"]
            ) <= thresholds["camera_height_tolerance_mm"]
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "eligible": not failures,
        "checks": checks,
        "metrics": metrics,
        "thresholds": thresholds,
        "failed_checks": failures,
        "accepted_capture_index": (
            accepted_capture.get("capture_index") if accepted_capture else None
        ),
        "pointcloud_center_base_mm": (
            center.copy() if center_is_finite else None
        ),
    }


def _process_one_hole(ctx: Any, order: int, hole: dict[str, Any]) -> None:
    """Execute one hole while preserving outer-loop failure recovery semantics."""
    args = ctx.args
    handeye = ctx.handeye
    model = ctx.model
    cfg = ctx.cfg
    run_dir = ctx.run_dir
    report = ctx.report
    timing = ctx.timing
    rows = ctx.rows
    runtime = ctx.runtime
    pose_session = ctx.pose_session
    motion_session = ctx.motion_session
    current_tcp = ctx.current_tcp
    initial_holes = ctx.initial_holes
    initial_intrinsics = ctx.initial_intrinsics
    fixed_rz_rad = ctx.fixed_rz_rad
    results = ctx.results
    order_ids = ctx.order_ids
    initial_pointcloud_reused_holes = ctx.initial_pointcloud_reused_holes
    all_selected_two_capture_mode = ctx.all_selected_two_capture_mode
    cache_entries = ctx.cache_entries
    cache_sources = ctx.cache_sources
    cache_source_ids = ctx.cache_source_ids
    cache_gates = ctx.cache_gates
    cache_enabled = ctx.cache_enabled
    persistent_enabled = ctx.persistent_enabled
    persistent_entries = ctx.persistent_entries
    coarse_cache_dir = ctx.coarse_cache_dir
    persistent_cache_dir = ctx.persistent_cache_dir
    cache_built_ids = ctx.cache_built_ids
    batch_coarse_results = ctx.batch_coarse_results
    batch_coarse_for_cache = ctx.batch_coarse_for_cache
    shared_cache_results = ctx.shared_cache_results
    shared_cache_failed_ids = ctx.shared_cache_failed_ids
    invalidated_cache_ids = ctx.invalidated_cache_ids
    batch_fine_results = ctx.batch_fine_results
    batch_fine_plan = ctx.batch_fine_plan
    ensure_rgbd_pipeline = ctx.ensure_rgbd_pipeline

    try:
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
        precaptured_fine = getattr(ctx, "in_group_fine_recoveries", {}).get(hole_id)

        failed_coarse = batch_coarse_results.get(hole_id, {})
        if failed_coarse.get("coarse_quality_retry") and not batch_coarse_available:
            # The shared stage already used the one allowed singleton retry.
            # Do not silently enter another per-hole coarse/fine motion cycle.
            reason = failed_coarse.get("error") or "单孔粗定位补拍仍未通过质量门"
            _append_deferred_hole_result(
                hole,
                {"status": "deferred_coarse_quality", "hole_id": hole_id,
                 "processing_order": order, "coarse_quality_retry": True,
                 "coarse_source": "singleton_coarse_quality_retry",
                 "coarse_quality_status": "deferred_coarse_quality",
                 "error": reason, "coarse_quality_note": reason,
                 "final_xy_motion": None, "final_z_motion": None},
                results, report, run_dir, timing,
            )
            print(f"[COARSE_QUALITY_RETRY] hole={hole_id} exhausted: {reason}", flush=True)
            return

        # 第四策略只接受当前独立高度批量点云得到的中心。批量点云失败时
        # 直接记录该孔不可用，不允许再走缓存、逐孔粗定位或260mm托底。
        if bool(getattr(args, "coarse_direct_final", False)) and not batch_coarse_available:
            batch_result = batch_coarse_results.get(hole_id, {}) or {}
            pointcloud_reason = (
                batch_result.get("error")
                or "独立高度批量点云未生成该孔中心"
            )
            _defer_coarse_direct_without_pointcloud(
                ctx, order, hole, str(pointcloud_reason),
            )
            print(
                f"[COARSE_DIRECT] order={order} hole={hole_id} "
                f"点云中心不可用，跳过该孔：{pointcloud_reason}",
                flush=True,
            )
            _confirm_next_hole_if_needed(
                order,
                initial_holes,
                hole_id,
                timing,
                enabled=(
                    bool(getattr(args, "move_final_xy", False))
                    or (
                        not bool(getattr(args, "coarse_direct_final_capture_only", False))
                        and not all_selected_two_capture_mode
                    )
                ),
                stop_message="点云失败",
            )
            return

        per_hole_fine_route = _per_hole_fine_route(args, batch_fine_available)
        batch_fine_fallback_active = bool(
            cfg.batch_fine_localization
            and len(initial_holes) > 1
            and hole_id in batch_fine_results
            and not batch_fine_available
            and bool(cfg.batch_fine_per_hole_fallback)
        )
        if batch_fine_fallback_active:
            hole["batch_fine_fallback_from_shared"] = True
            hole["batch_fine_fallback_reason"] = (
                batch_fine_result.get("error")
                or "shared_batch_fine_quality_gate_failed"
            )
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

        # 批量模式的正式图像已经在进入逐孔结果计算前统一拍完；共享精定位
        # 的失败处理只适用于其它策略，第四策略在前面的点云缺失分支结束。
        if (
            cfg.batch_fine_localization
            and len(initial_holes) > 1
            and all_selected_two_capture_mode
            and not batch_fine_available
            and not cfg.batch_fine_per_hole_fallback
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
                "batch_fine_joint_enabled": bool(cfg.batch_fine_joint_localization),
                "batch_fine_joint_applied": False,
                "batch_fine_joint_summary": batch_fine_result.get(
                    "batch_fine_joint_summary"
                ),
                "batch_fine_source": str(
                    batch_fine_result.get(
                        "batch_fine_source", "shared_batch_fine_failed"
                    )
                ),
                "batch_fine_capture_round": int(
                    batch_fine_result.get("batch_fine_capture_round", 0)
                ),
                "batch_fine_fallback_from_shared": bool(batch_fine_fallback_active),
                "batch_fine_fallback_reason": hole.get("batch_fine_fallback_reason"),
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
            _append_deferred_hole_result(
                hole, deferred_result, results, report, run_dir, timing,
            )
            print(
                f"[BATCH_FINE] hole={hole_id} status=deferred_batch_fine；"
                f"共享补拍失败且已关闭逐孔补拍：{failure_reason}",
                flush=True,
            )
            return

        if bool(getattr(args, "coarse_direct_final", False)):
            # 第四策略复用本轮共享观察位已经完成的点云结果，但不进入
            # 通用批量粗定位后的260 mm分支。这里先补齐逐孔上下文，随后
            # 直接执行点云中心直达或“仅采集评估”路径。
            batch_result = batch_coarse_results[hole_id]
            coarse_captures = _apply_batch_coarse_result_to_hole(
                hole, batch_result,
            )
            cache_event = {
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
                "navigation_point_base_mm": np.asarray(
                    batch_result["center_base_mm"], dtype=np.float64,
                ).copy(),
                "capture_height_mm": batch_result.get("capture_height_mm"),
            }
            capture_only = bool(getattr(args, "coarse_direct_final_capture_only", False))
            hole["coarse_direct_decision"] = (
                "capture_only" if capture_only else "pointcloud_direct"
            )
            current_tcp, direct_result = _run_coarse_direct_final(
                ctx,
                order,
                hole,
                current_tcp,
                cache_event=cache_event,
                coarse_captures=coarse_captures,
                batch_coarse_for_cache=batch_coarse_for_cache,
            )
            _commit_completed_hole_result(ctx, order, hole, direct_result)
            print(
                f"[COARSE_DIRECT] order={order} hole={hole_id} "
                + (
                    "完成点云采集评估，不执行最终XY/Z动作；"
                    if capture_only else
                    "使用点云中心直达，保留ChArUco XY纠偏；"
                )
                + f"center={np.round(np.asarray(hole['coarse_center_base_mm']), 3).tolist()} "
                f"target={np.round(np.asarray(direct_result['target_point_base_mm']), 3).tolist()}",
                flush=True,
            )
            _confirm_next_hole_if_needed(
                order,
                initial_holes,
                hole_id,
                timing,
                enabled=(
                    bool(getattr(args, "move_final_xy", False))
                    or (
                        not capture_only
                        and not all_selected_two_capture_mode
                    )
                ),
            )
            return

        if batch_coarse_available:
            # 使用批量粗定位结果，跳过逐孔粗定位流程
            print(
                f"[SEQUENTIAL_HOLE] order={order} hole={hole_id} "
                f"使用批量粗定位结果，跳过逐孔340mm采集",
                flush=True,
            )

            batch_result = batch_coarse_results[hole_id]

            # 将批量粗定位结果写入hole字典；普通两阶段随后进入260 mm
            # 精定位，第四策略则在下面直接使用同一份点云中心。
            coarse_captures = _apply_batch_coarse_result_to_hole(
                hole, batch_result,
            )

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
            if per_hole_fine_route == "per_hole_fallback" and precaptured_fine is None:
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
                        current_tcp,
                        batch_fine_target,
                        args,
                        motion_session,
                        pose_session,
                        target_height_mm=cfg.fine_height_mm,
                        target_stage="批量粗定位后的单孔精定位兜底",
                        safe_margin_mm=cfg.per_hole_fine_safe_z_margin_mm,
                        descent_guard_mm=(
                            SHARED_OBSERVATION_MIN_DESCENT_MM
                            if all_selected_two_capture_mode else 0.0
                        ),
                    )
            elif per_hole_fine_route == "batch_fine_result":
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
            if per_hole_fine_route == "per_hole_fallback" and precaptured_fine is None:
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
                        str(hole_id), current_tcp, fine_target, args,
                        motion_session, pose_session, target_height_mm=cfg.fine_height_mm,
                        target_stage="共享缓存验证后的单孔精定位兜底",
                        safe_margin_mm=cfg.per_hole_fine_safe_z_margin_mm,
                        descent_guard_mm=(
                            SHARED_OBSERVATION_MIN_DESCENT_MM
                            if all_selected_two_capture_mode else 0.0
                        ),
                    )
            elif per_hole_fine_route == "batch_fine_result":
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
            if per_hole_fine_route == "per_hole_fallback" and precaptured_fine is None:
                # 初始点云复用只提供每孔粗几何；共享精拍失败时仍必须真实
                # 移到该孔自己的260 mm位姿，再走单孔RGB质量门。
                fine_target, fine_geometry = _plan_hole_tcp_pose_fixed_rz(
                    np.asarray(hole["coarse_center_base_mm"], dtype=np.float64),
                    np.asarray(hole["coarse_normal_toward_camera_base"], dtype=np.float64),
                    current_tcp, handeye.T_tcp_rgb_camera, fixed_rz_rad=fixed_rz_rad,
                    camera_height_mm=cfg.fine_height_mm,
                )
                hole["initial_pointcloud_reuse_fine_target_tcp_pose_m_rad"] = (
                    transform_to_sdk_pose_m_rad(fine_target)
                )
                hole["initial_pointcloud_reuse_fine_pose_geometry"] = fine_geometry
                with timing.measure(
                    f"hole_{hole_id:02d}/initial_pointcloud_reuse_move_to_fine_pose",
                    hole_id=hole_id, processing_order=order,
                ):
                    current_tcp = _move_to_fine_pose(
                        str(hole_id), current_tcp, fine_target, args,
                        motion_session, pose_session, target_height_mm=cfg.fine_height_mm,
                        target_stage="初始点云复用后的单孔精定位兜底",
                        safe_margin_mm=cfg.per_hole_fine_safe_z_margin_mm,
                        descent_guard_mm=(
                            SHARED_OBSERVATION_MIN_DESCENT_MM
                            if all_selected_two_capture_mode else 0.0
                        ),
                    )
            elif per_hole_fine_route == "batch_fine_result":
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
                        validation, cache_observations = _validate_coarse_cache_at_current_pose(
                            hole, cache_entry,
                            pipeline=rgbd_pipeline, align=align, chain=chain,
                            model=model, confidence=args.confidence,
                            run_dir=run_dir, cfg=cfg, gates=cache_gates,
                            current_T_base_camera=T_base_camera, intrinsics=initial_intrinsics,
                            timing=timing,
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
                            timing=timing,
                        )
                        hole["coarse_pointcloud_image_path"] = (
                            None if pointcloud_path is None else str(pointcloud_path)
                        )
                except LocalizationHardwareError:
                    raise
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
        accepted_same_capture_fine: dict[str, Any] | None = None
        accepted_same_capture_tcp: np.ndarray | None = None
        accepted_same_capture_anchor: np.ndarray | None = None
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
            same_capture_fine = (
                {} if bool(getattr(args, "map_build_same_capture_340", False)) else None
            )
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
                    timing=timing,
                    T_base_camera=(
                        T_base_camera
                        if bool(getattr(args, "map_build_per_hole_reference", False))
                        else None
                    ),
                    same_capture_fine=same_capture_fine,
                )
            rows.extend(_observation_rows(coarse_observations))
            if same_capture_fine is not None:
                rows.extend(_observation_rows(same_capture_fine["observations"]))
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
            if same_capture_fine is not None:
                accepted_same_capture_fine = same_capture_fine
                accepted_same_capture_tcp = capture_tcp.copy()
                accepted_same_capture_anchor = expected_anchor_px.copy()
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
                "batch_fine_joint_enabled": bool(
                    cfg.batch_fine_localization and cfg.batch_fine_joint_localization
                ),
                "batch_fine_joint_applied": False,
                "batch_fine_joint_summary": batch_fine_result.get(
                    "batch_fine_joint_summary"
                ),
                "batch_fine_source": (
                    str(batch_fine_result.get("batch_fine_source", "batch_fine_at_260mm"))
                    if batch_fine_available else "per_hole_fine_fallback"
                ),
                "batch_fine_capture_round": int(
                    batch_fine_result.get("batch_fine_capture_round", 0)
                ),
                "batch_fine_fallback_from_shared": bool(batch_fine_fallback_active),
                "batch_fine_fallback_reason": hole.get("batch_fine_fallback_reason"),
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
            _append_deferred_hole_result(
                hole, deferred_result, results, report, run_dir, timing,
            )
            print(
                f"[SEQUENTIAL_HOLE] order={order} hole={hole_id} "
                f"status=deferred_coarse_quality；继续后续孔",
                flush=True,
            )
            _confirm_next_hole_if_needed(
                order, initial_holes, hole_id, timing, stop_message="失败",
            )
            return
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
                timing=timing,
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
        coarse_center_base = np.asarray(
            hole["coarse_center_base_mm"], dtype=np.float64,
        ).reshape(3)
        hole["coarse_plane_point_base_mm"] = coarse_plane_base
        hole["coarse_normal_toward_camera_base"] = coarse_normal_base

        if bool(getattr(args, "map_build_coarse_only", False)):
            # 建图阶段在340 mm完成粗定位后立即记录结果；260 mm精定位和
            # 最终运动只在后续地图调用中执行，避免把变形后的伞架精定位
            # 坐标写入地图。
            result = {
                "status": "completed",
                "hole_id": hole_id,
                "processing_order": order,
                "tracking_identity": hole["tracking_identity"],
                "initial_selection_order": hole.get("initial_selection_order"),
                "initial_center_px": hole.get("initial_center_px"),
                "initial_center_base_mm": hole.get("initial_center_base_mm"),
                "initial_plane_normal_base": hole.get("initial_plane_normal_base"),
                "batch_coarse_requested": bool(batch_coarse_for_cache),
                "batch_fine_requested": False,
                "batch_fine_source": "disabled_for_coarse_map_build",
                "map_build_localization_mode": "coarse_only",
                "map_build_fine_reference": False,
                "batch_fine_joint_enabled": False,
                "batch_fine_joint_applied": False,
                "batch_fine_pointcloud_fusion_applied": False,
                "coarse_source": (
                    "batch_coarse_localization"
                    if batch_coarse_available else "fresh_per_hole_coarse"
                ),
                "fine_quality_status": "not_run_coarse_only_build",
                "fine_quality_note": "粗定位建图阶段不执行260 mm精定位",
                "fine_recovery_attempts": [],
                "coarse_center_base_mm": hole["coarse_center_base_mm"],
                "coarse_center_camera_mm": hole.get("coarse_center_camera_mm"),
                "coarse_plane_point_base_mm": coarse_plane_base,
                "coarse_plane_point_camera_mm": hole.get("coarse_plane_point_camera_mm"),
                "coarse_normal_camera": hole.get("coarse_normal_camera"),
                "coarse_normal_toward_camera_base": coarse_normal_base,
                "coarse_plane_rmse_mm": hole.get("coarse_plane_rmse_mm"),
                "coarse_valid_frames": hole.get("coarse_valid_frames"),
                "coarse_total_frames": hole.get("coarse_total_frames"),
                "coarse_center_scatter_p95_px": hole.get("coarse_center_scatter_p95_px"),
                "coarse_tracking_distance_p95_px": hole.get("coarse_tracking_distance_p95_px"),
                "coarse_ring_points_median": hole.get("coarse_ring_points_median"),
                "coarse_ring_coverage_min_ratio": hole.get("coarse_ring_coverage_min_ratio"),
                "coarse_ring_max_gap_deg": hole.get("coarse_ring_max_gap_deg"),
                "coarse_quality_retry": hole.get("coarse_quality_retry"),
                "coarse_surface_model": hole.get("coarse_surface_model"),
                "coarse_surface_selection_policy": hole.get("coarse_surface_selection_policy"),
                "coarse_front_surface_z_mm": hole.get("coarse_front_surface_z_mm"),
                "coarse_ring_points_raw_median": hole.get("coarse_ring_points_raw_median"),
                "coarse_surface_points_selected_median": hole.get(
                    "coarse_surface_points_selected_median"
                ),
                "coarse_sphere_center_camera_mm": hole.get("coarse_sphere_center_camera_mm"),
                "coarse_sphere_radius_mm": hole.get("coarse_sphere_radius_mm"),
                "coarse_captures": hole.get("coarse_captures", coarse_captures),
                "coarse_cache_event": cache_event,
                "pointcloud_segmentation": hole.get("pointcloud_segmentation"),
                "diameter_estimate_mm": hole.get("diameter_estimate_mm"),
                "matched_diameter_mm": hole.get("matched_diameter_mm"),
                "coarse_map_build": True,
                "coarse_capture_tcp_pose_m_rad": (
                    hole.get("coarse_capture_tcp_pose_m_rad")
                    or next(
                        (
                            item.get("tcp_pose_m_rad")
                            for item in reversed(coarse_captures)
                            if isinstance(item, dict) and item.get("tcp_pose_m_rad") is not None
                        ),
                        transform_to_sdk_pose_m_rad(current_tcp),
                    )
                ),
                "tracking_events": hole.get("tracking_events", []),
                "initial_detection": hole.get("initial_detection"),
                "timing": timing.scoped_snapshot(f"hole_{hole_id:02d}/"),
            }
            for key in (
                "initial_selection_order", "operator_selection_order", "selection_source",
                "numbering_policy", "map_order", "layout_row", "layout_column",
                "numbering_center_px", "map_hole_key", "global_hole_key", "sector_id",
            ):
                if result.get(key) is None and hole.get(key) is not None:
                    result[key] = hole.get(key)
            result["comparison_diagnostics"] = _build_comparison_hole_diagnostics(result)
            hole["final_result"] = result
            results.append(result)
            report["stages"][f"hole_{hole_id}"] = result
            completed_results = [item for item in results if item.get("status") == "completed"]
            report["stages"]["processed_holes"] = {
                "completed_count": len(completed_results),
                "deferred_count": len(results) - len(completed_results),
                "total_count": len(results),
                "hole_order": [int(item["hole_id"]) for item in results],
                "holes": results,
            }
            runtime["current_tcp"] = np.asarray(current_tcp, dtype=np.float64).copy()
            _write_progress_checkpoint(run_dir, report, timing=timing)
            print(
                f"[COARSE_MAP_BUILD] order={order} hole={hole_id} "
                f"coarse={np.round(np.asarray(hole['coarse_center_base_mm']), 3).tolist()}",
                flush=True,
            )
            _confirm_next_hole_if_needed(
                order, initial_holes, hole_id, timing, enabled=False,
            )
            return

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
        elif precaptured_fine is not None:
            # The fallback RGB burst was already captured next to its shared
            # group. Use its measured TCP for geometry; do not revisit the
            # hole before the normal final-motion phase.
            fine_capture_tcp = np.asarray(
                precaptured_fine["capture_tcp"], dtype=np.float64,
            ).copy()
            estimated_height = float(precaptured_fine["height_mm"])
            hole["fine_height_estimate_mm"] = estimated_height
            expected_fine_anchor_px = np.asarray(
                precaptured_fine["expected_anchor_px"], dtype=np.float64,
            ).copy()
            fine_recovery = precaptured_fine["recovery"]
        elif bool(getattr(args, "map_build_same_capture_340", False)):
            # The accepted coarse capture already contains the RGB frames used
            # for strict fine localization. Keep their TCP and intrinsics paired.
            capture = accepted_same_capture_fine or {}
            fine_observations = list(capture.get("observations") or [])
            fine_intrinsics = capture.get("intrinsics")
            fine_capture_tcp = np.asarray(
                accepted_same_capture_tcp if accepted_same_capture_tcp is not None
                else current_tcp, dtype=np.float64,
            ).copy()
            estimated_height = camera_height_to_plane_mm(
                fine_capture_tcp, handeye.T_tcp_rgb_camera, coarse_plane_base,
            )
            hole["fine_height_estimate_mm"] = estimated_height
            error = None
            fine = None
            if fine_intrinsics is None:
                error = "340 mm同拍精定位没有有效RGB内参"
            else:
                try:
                    with timing.measure(
                        f"hole_{hole_id:02d}/same_capture_340_fine_fusion",
                        hole_id=hole_id,
                        processing_order=order,
                    ):
                        fine = _fuse_fine(fine_observations, cfg)
                    fine["fine_quality_status"] = "strict"
                    fine["fine_recovery_attempts"] = []
                except RuntimeError as exc:
                    error = str(exc)
            fine_recovery = {
                "success": fine is not None,
                "observations": fine_observations,
                "intrinsics": fine_intrinsics,
                "fine": fine,
                "attempts": [],
                "error": error,
            }
            if accepted_same_capture_anchor is not None and fine_intrinsics is not None:
                _record_hole_tracking_event(
                    hole, "same_capture_340_fine", accepted_same_capture_anchor,
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
                # 初始点云复用分支为了避免 340 mm 重拍，可能没有给本地
                # ``rgbd_pipeline`` 变量赋值。精拍回退仍统一从上下文取当前
                # 管线，兼容首次启动和按需重启两条路径。
                fine_pipeline, _, _ = ensure_rgbd_pipeline()
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
                "batch_fine_joint_enabled": bool(
                    cfg.batch_fine_localization and cfg.batch_fine_joint_localization
                ),
                "batch_fine_joint_applied": False,
                "batch_fine_joint_summary": batch_fine_result.get(
                    "batch_fine_joint_summary"
                ),
                "batch_fine_source": (
                    str(batch_fine_result.get("batch_fine_source", "batch_fine_at_260mm"))
                    if batch_fine_available else
                    "same_capture_340_fine"
                    if bool(getattr(args, "map_build_same_capture_340", False)) else
                    "per_hole_fine_fallback"
                ),
                "map_build_localization_mode": (
                    getattr(args, "map_build_localization_mode", None)
                    if bool(getattr(args, "map_build_per_hole_reference", False)) else None
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
                "in_group_per_hole_fallback_precaptured": bool(precaptured_fine is not None),
                "fine_quality_note": fine_recovery["error"],
                "localization_path": (
                    "same_capture_340_deferred"
                    if bool(getattr(args, "map_build_same_capture_340", False))
                    else "fine_260_deferred"
                ),
                "fine_stage_skipped": False,
                "batch_coarse_center_base_mm": hole.get(
                    "batch_coarse_center_base_mm"
                ),
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
                    "coarse_center_ray_intersection_with_fused_local_pointcloud_plane"
                ),
                "coarse_plane_point_base_mm": coarse_plane_base,
                "coarse_plane_point_camera_mm": hole["coarse_plane_point_camera_mm"],
                "coarse_normal_camera": hole["coarse_normal_camera"],
                "coarse_normal_toward_camera_base": coarse_normal_base,
                "coarse_plane_rmse_mm": hole["coarse_plane_rmse_mm"],
                "coarse_valid_frames": hole["coarse_valid_frames"],
                "coarse_total_frames": hole.get("coarse_total_frames"),
                "coarse_center_scatter_p95_px": hole["coarse_center_scatter_p95_px"],
                "coarse_ring_points_median": hole.get("coarse_ring_points_median"),
                "coarse_ring_coverage_min_ratio": hole.get(
                    "coarse_ring_coverage_min_ratio"
                ),
                "coarse_ring_max_gap_deg": hole.get("coarse_ring_max_gap_deg"),
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
            if bool(getattr(args, "map_build_same_capture_340", False)):
                fallback_validation = _same_capture_pointcloud_fallback_validation(
                    hole,
                    cfg,
                    map_build_enabled=bool(
                        getattr(args, "map_build_per_hole_reference", False)
                    ),
                    motion_disabled=not bool(getattr(args, "move_final_xy", False)),
                    same_capture_available=accepted_same_capture_fine is not None,
                )
                if fallback_validation["eligible"]:
                    coarse_center = np.asarray(
                        hole["coarse_center_base_mm"], dtype=np.float64,
                    ).reshape(3).copy()
                    valid_fine_frames = sum(
                        item.error is None and item.ellipse is not None
                        for item in fine_observations
                    )
                    fallback_result = dict(deferred_result)
                    fallback_result.update({
                        "status": "completed",
                        "deferred_reason": None,
                        "pointcloud_center_fallback": True,
                        "pointcloud_center_fallback_reason": fine_recovery["error"]
                        or "严格RGB精定位结果不可用",
                        "pointcloud_center_fallback_validation": fallback_validation,
                        "fine_quality_status": "pointcloud_center_fallback",
                        "fine_quality_note": fine_recovery["error"]
                        or "严格RGB精定位结果不可用；使用通过质量门的340 mm点云中心",
                        "fine_center_source": "coarse_pointcloud_center_fallback",
                        "fine_center_source_counts": {
                            "coarse_pointcloud_center_fallback": 1,
                        },
                        "fine_xy_source": "coarse_pointcloud_center_fallback",
                        "fine_z_source": "coarse_pointcloud_center_z",
                        "localization_path": "same_capture_340_pointcloud_center_fallback",
                        "map_build_localization_mode": "same_capture_340",
                        "map_build_fine_reference": True,
                        "map_build_final_motion_disabled": True,
                        "hole_result_type": "base_frame_3d_point",
                        "hole_center_base_mm": coarse_center.copy(),
                        "hole_center_base_naive_mm": coarse_center.copy(),
                        "pointcloud_center_base_mm": coarse_center.copy(),
                        "fine_plane_intersection_mm": None,
                        "fine_tcp_pose_m_rad": None,
                        "valid_frames": valid_fine_frames,
                        "total_frames": len(fine_observations),
                        "fine_valid_frames": valid_fine_frames,
                        "target_point_base_mm": None,
                        "final_pose_source": "map_reference_only_pointcloud_center_fallback",
                    })
                    for key in (
                        "initial_selection_order", "operator_selection_order",
                        "selection_source", "numbering_policy", "map_order",
                        "layout_row", "layout_column", "numbering_center_px",
                        "numbering_row_tolerance_px", "map_hole_key", "global_hole_key",
                        "sector_id", "boundary_class", "boundary_layer",
                        "boundary_component_index", "boundary_edge_score_deg",
                        "boundary_local_degree", "boundary_distance",
                        "boundary_distance_mm", "boundary_distance_px",
                        "boundary_distance_unit", "boundary_coordinate_source",
                        "boundary_classification_reason", "boundary_override",
                        "boundary_override_source", "group_phase", "group_boundary_class",
                        "edge_first_group_index", "edge_first_phase_group_index",
                        "batch_coarse_group_index", "batch_coarse_group_hole_ids",
                    ):
                        if fallback_result.get(key) is None and hole.get(key) is not None:
                            fallback_result[key] = hole.get(key)
                    fallback_result["comparison_diagnostics"] = (
                        _build_comparison_hole_diagnostics(fallback_result)
                    )
                    hole["final_result"] = fallback_result
                    results.append(fallback_result)
                    report["stages"][f"hole_{hole_id}"] = fallback_result
                    report["stages"]["processed_holes"] = {
                        "completed_count": sum(
                            item.get("status") == "completed" for item in results
                        ),
                        "deferred_count": sum(
                            item.get("status") != "completed" for item in results
                        ),
                        "total_count": len(results),
                        "hole_order": [int(item["hole_id"]) for item in results],
                        "holes": results,
                    }
                    _write_progress_checkpoint(run_dir, report, timing=timing)
                    print(
                        f"[SEQUENTIAL_HOLE] order={order} hole={hole_id} "
                        "status=completed_pointcloud_center_fallback "
                        f"center={np.round(coarse_center, 3).tolist()} "
                        f"fine_failure={fine_recovery['error']}",
                        flush=True,
                    )
                    _confirm_next_hole_if_needed(
                        order, initial_holes, hole_id, timing,
                    )
                    return
                deferred_result["pointcloud_center_fallback_validation"] = (
                    fallback_validation
                )
            _append_deferred_hole_result(
                hole, deferred_result, results, report, run_dir, timing,
            )
            print(
                f"[SEQUENTIAL_HOLE] order={order} hole={hole_id} "
                f"status=deferred_fine_quality "
                f"coarse_center={np.round(np.asarray(hole['coarse_center_base_mm']), 3).tolist()} ",
                flush=True,
            )
            _confirm_next_hole_if_needed(
                order, initial_holes, hole_id, timing,
            )
            return

        # 每个孔都按单孔精定位的严格中心稳定性门验收；若严格门失败但
        # 重拍后稳定，则fine_recovery会返回degraded_fine并把降级原因写入报告。
        with timing.measure(
            f"hole_{hole_id:02d}/fine_pixel_to_base",
            hole_id=hole_id,
            processing_order=order,
            category="vision_compute",
            visual_stage="coordinate_transform",
            level="leaf",
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
                category="vision_compute",
                visual_stage="final_point_calculation",
                level="leaf",
            ):
                final_point_base, tilt_correction = correct_projected_circle_center(
                    fine["center_px"], fine_intrinsics, fine_capture_tcp, handeye.T_tcp_rgb_camera,
                    coarse_plane_base, coarse_normal_base, nearest_diameter,
                    iterations=cfg.tilt_correction_iterations,
                    samples=cfg.tilt_correction_samples,
                    max_correction_mm=cfg.max_tilt_correction_mm,
                )

        # 共享精拍只更新每孔最终XY。联合结果是同一共享视野内多孔
        # 平面刚体变换的主结果；在其上只保留受限的逐孔局部残差，
        # 并保留已有倾斜圆心纠偏。联合门控失败的孔不会使用共享帧
        # 作为最终点，而是在上游转入真正的逐孔精定位。
        batch_fine_joint_enabled = bool(
            batch_fine_available and cfg.batch_fine_joint_localization
        )
        batch_fine_joint_applied = False
        batch_fine_joint_predicted_point_base: np.ndarray | None = None
        batch_fine_joint_details: dict[str, Any] = {}
        batch_fine_joint_summary = batch_fine_result.get(
            "batch_fine_joint_summary"
        ) or {}
        if batch_fine_joint_enabled and batch_fine_result.get(
            "batch_fine_joint_success", False
        ):
            joint_predictions = (
                batch_fine_joint_summary.get("predicted_xy_by_hole") or {}
            )
            joint_xy = joint_predictions.get(
                hole_id, joint_predictions.get(str(hole_id))
            )
            if joint_xy is not None:
                try:
                    coarse_center_for_joint = np.asarray(
                        hole["coarse_center_base_mm"], dtype=np.float64,
                    ).reshape(3)
                    batch_fine_joint_predicted_point_base = np.asarray(
                        [
                            float(np.asarray(joint_xy).reshape(2)[0]),
                            float(np.asarray(joint_xy).reshape(2)[1]),
                            float(coarse_center_for_joint[2]),
                        ],
                        dtype=np.float64,
                    )
                    with timing.measure(
                        f"hole_{hole_id:02d}/joint_final_point_calculation",
                        hole_id=hole_id,
                        processing_order=order,
                        category="vision_compute",
                        visual_stage="final_point_calculation",
                        level="leaf",
                    ):
                        final_point_base, batch_fine_joint_details = (
                            _compose_batch_fine_joint_xy_with_tilt(
                                naive_final_point_base,
                                final_point_base,
                                batch_fine_joint_predicted_point_base,
                                local_residual_weight=(
                                    cfg.batch_fine_joint_local_residual_weight
                                ),
                                local_residual_limit_mm=(
                                    cfg.batch_fine_joint_local_residual_limit_mm
                                ),
                            )
                        )
                    batch_fine_joint_applied = True
                except Exception as joint_apply_exc:
                    batch_fine_joint_details = {
                        "error": f"{type(joint_apply_exc).__name__}:{joint_apply_exc}",
                    }

        # 粗拍点云支撑中心和共享精拍中心是两个独立高度下的观测。只有
        # 两者在安全门内一致时，才让点云中心对最终XY做有限权重拉回；
        # 超门限只记录冲突，不让一次粗定位异常污染精定位结果。
        batch_fine_visual_point_before_pointcloud = np.asarray(
            final_point_base, dtype=np.float64,
        ).copy()
        batch_fine_pointcloud_fusion_details: dict[str, Any] = {
            "status": "disabled" if batch_fine_available else "not_applicable",
            "applied": False,
        }
        batch_fine_pointcloud_fusion_applied = False
        if batch_fine_available and cfg.batch_fine_pointcloud_xy_fusion:
            try:
                joint_residual_for_pointcloud: float | None = None
                if batch_fine_joint_applied:
                    joint_residuals = (
                        batch_fine_joint_summary.get("residual_mm_by_hole") or {}
                    )
                    joint_residual_value = joint_residuals.get(
                        hole_id, joint_residuals.get(str(hole_id))
                    )
                    if joint_residual_value is not None:
                        joint_residual_for_pointcloud = float(joint_residual_value)
                with timing.measure(
                    f"hole_{hole_id:02d}/pointcloud_prior_fusion",
                    hole_id=hole_id,
                    processing_order=order,
                    category="vision_compute",
                    visual_stage="fusion_quality",
                    level="leaf",
                ):
                    final_point_base, batch_fine_pointcloud_fusion_details = (
                        fuse_batch_fine_xy_with_pointcloud_prior(
                            batch_fine_visual_point_before_pointcloud,
                            np.asarray(hole["coarse_center_base_mm"], dtype=np.float64),
                            pointcloud_weight=cfg.batch_fine_pointcloud_xy_weight,
                            max_correction_mm=(
                                cfg.batch_fine_pointcloud_xy_max_correction_mm
                            ),
                            agreement_gate_mm=(
                                cfg.batch_fine_pointcloud_xy_agreement_gate_mm
                            ),
                            joint_residual_mm=joint_residual_for_pointcloud,
                            max_joint_residual_mm=(
                                cfg.batch_fine_joint_max_residual_mm
                                if batch_fine_joint_applied else None
                            ),
                            adaptive_weight=True,
                        )
                    )
                batch_fine_pointcloud_fusion_applied = bool(
                    batch_fine_pointcloud_fusion_details.get("applied", False)
                )
            except Exception as pointcloud_fusion_exc:
                final_point_base = batch_fine_visual_point_before_pointcloud.copy()
                batch_fine_pointcloud_fusion_details = {
                    "status": "error",
                    "applied": False,
                    "error": (
                        f"{type(pointcloud_fusion_exc).__name__}:"
                        f"{pointcloud_fusion_exc}"
                    ),
                }

        # 共享精拍只更新每孔最终XY。该孔的最终Z与姿态仍由粗定位决定；
        # 非共享精拍流程保持原有三维最终点行为。
        pose_point_base = np.asarray(final_point_base, dtype=np.float64).copy()
        if batch_fine_available:
            pose_point_base = compose_batch_fine_xy_with_coarse_z(
                pose_point_base,
                np.asarray(hole["coarse_center_base_mm"], dtype=np.float64),
            )

        # 直接使用定位结果作为最终点，不附加X/Z偏移。
        final_target_point_base = np.asarray(
            pose_point_base, dtype=np.float64,
        ).copy()
        fine_tcp = np.asarray(fine_capture_tcp, dtype=np.float64).copy()
        final_xy_motion: dict[str, Any] | None = None
        final_z_motion: dict[str, Any] | None = None
        final_y_trim_motion: dict[str, Any] | None = None
        final_motion_direct: dict[str, Any] | None = None
        if args.move_final_xy and batch_fine_available:
            assert coarse_fine_reference_tcp is not None
            use_charuco_model = bool(getattr(args, "use_charuco_xy_correction", True)) and args.tcp_xy_offset_mm is None
            fixed_offset = (
                None if args.tcp_xy_offset_mm is None
                else (float(args.tcp_xy_offset_mm[0]), float(args.tcp_xy_offset_mm[1]))
            )
            # 共享精拍只提供最终XY；粗定位参考位姿只作为每孔姿态和
            # 粗定位Z的数学底稿，不再让机械臂先回到这个260mm位姿。
            reference_tcp = np.asarray(
                coarse_fine_reference_tcp, dtype=np.float64,
            ).reshape(4, 4).copy()
            motion_start_tcp = np.asarray(current_tcp, dtype=np.float64).copy()
            xy_target, _ = plan_final_tcp_xy(
                reference_tcp, final_target_point_base, fixed_offset,
                use_charuco_model=use_charuco_model,
            )
            correction_xy = xy_target[:2, 3] - final_target_point_base[:2]
            z_target = plan_final_tcp_base_z(xy_target, final_target_point_base)
            # 最终目标只包含孔位规划和 ChArUco XY 纠偏，不再追加固定 Y 偏置。
            direct_target = z_target
            _persist_planned_final_point(
                ctx, hole, final_point_base, final_target_point_base, direct_target,
                motion_path="batch_fine_direct_safe_final_tcp",
            )
            direct_safe_z = max(
                float(motion_start_tcp[2, 3]),
                float(direct_target[2, 3]) + THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM,
            )
            direct_guard_z = float(direct_target[2, 3]) + SHARED_OBSERVATION_MIN_DESCENT_MM
            with timing.measure(
                f"hole_{hole_id:02d}/final_motion_direct_safe_path",
                hole_id=hole_id,
                processing_order=order,
            ):
                current_tcp = _move_to_batch_final_tcp_direct(
                    str(hole_id), current_tcp, direct_target,
                    args, motion_session, pose_session,
                )
            hole["coarse_pose_restored_for_batch_fine_final_motion"] = False
            hole["batch_fine_final_motion_path"] = "direct_safe_final_tcp"
            compensation = (
                f"ChArUco仿射模型修正={np.round(correction_xy, 3).tolist()} mm"
                if use_charuco_model else
                f"显式固定补偿={list(fixed_offset)} mm" if fixed_offset is not None
                else "不使用ChArUco纠偏"
            )
            final_xy_motion = {
                "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(direct_target),
                "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "tcp_position_before_mm": motion_start_tcp[:3, 3].copy(),
                "hole_center_base_mm": final_point_base,
                "pose_point_base_mm": pose_point_base,
                "target_point_base_mm": final_target_point_base,
                "batch_fine_joint_applied": batch_fine_joint_applied,
                "compensation_mode": (
                    "charuco_affine_model" if use_charuco_model else
                    "fixed_offset_override" if fixed_offset is not None else "none"
                ),
                "xy_correction_mm": correction_xy,
                "charuco_model_source": str(CHARUCO_XY_MODEL_SOURCE) if use_charuco_model else None,
                "charuco_model_matrix_2x2": CHARUCO_XY_MODEL_MATRIX if use_charuco_model else None,
                "charuco_model_bias_mm": CHARUCO_XY_MODEL_BIAS_MM if use_charuco_model else None,
                "tcp_xy_offset_mm": None if fixed_offset is None else list(fixed_offset),
                "motion_path": "direct_safe_final_tcp",
                "reference_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(reference_tcp),
                "planned_safe_z_mm": direct_safe_z,
                "planned_guard_z_mm": direct_guard_z,
                "separate_y_trim_executed": False,
                "compensation_description": compensation,
            }
            final_z_motion = {
                "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(direct_target),
                "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "target_base_z_mm": float(direct_target[2, 3]),
                "target_point_base_mm": final_target_point_base,
                "delta_base_z_mm": float(direct_target[2, 3] - reference_tcp[2, 3]),
                "motion_frame": "base_z_only_guard10_then_precision10",
                "descent_guard_mm": SHARED_OBSERVATION_MIN_DESCENT_MM,
            }
            final_motion_direct = {
                "path_policy": (
                    "pure_z_lift_safe_horizontal_to_final_tcp_"
                    "pure_z_descent_guard10_then_pure_z_final10"
                ),
                "reference_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(reference_tcp),
                "planned_final_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(direct_target),
                "planned_safe_z_mm": direct_safe_z,
                "planned_guard_z_mm": direct_guard_z,
                "safe_margin_mm": THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM,
                "descent_guard_mm": SHARED_OBSERVATION_MIN_DESCENT_MM,
                "post_final_y_offset_applied": False,
                "actual_final_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
            }
        if args.move_final_xy and not batch_fine_available:
            use_charuco_model = bool(getattr(args, "use_charuco_xy_correction", True)) and args.tcp_xy_offset_mm is None
            # 按下发前实测TCP判断高低，避免异常恢复时缓存位姿导致路径选错。
            _, measured_tcp = _require_safe_snapshot(pose_session)
            current_tcp = np.asarray(measured_tcp, dtype=np.float64).copy()
            # In-group fallback RGB capture happened before the other groups
            # and final-hole sequence. The live TCP now belongs to the previous
            # completed hole; it is only the motion START, never this hole's
            # target orientation. Before early capture existed these were the
            # same pose, because capture immediately preceded final motion.
            fixed_offset = (
                None if args.tcp_xy_offset_mm is None
                else (float(args.tcp_xy_offset_mm[0]), float(args.tcp_xy_offset_mm[1]))
            )
            pose_reference_tcp, xy_target, planned_final_tcp = _plan_per_hole_final_target(
                current_tcp, fine_capture_tcp, final_target_point_base, fixed_offset,
                use_charuco_model=use_charuco_model,
                precaptured=precaptured_fine is not None,
            )
            tcp_before = current_tcp[:3, 3].copy()
            correction_xy = xy_target[:2, 3] - final_target_point_base[:2]
            compensation = (
                f"ChArUco仿射模型修正={np.round(correction_xy, 3).tolist()} mm"
                if use_charuco_model else
                f"显式固定补偿={list(fixed_offset)} mm" if fixed_offset is not None
                else "不使用ChArUco纠偏"
            )
            needs_safe_lift = bool(
                precaptured_fine is not None
                or float(current_tcp[2, 3]) < float(planned_final_tcp[2, 3])
            )
            final_path_policy = (
                "safe_z_lift_xy_guarded_z_descent"
                if needs_safe_lift else "xy_then_z"
            )
            _persist_planned_final_point(
                ctx, hole, final_point_base, final_target_point_base, planned_final_tcp,
                motion_path=final_path_policy,
            )
            motion_start_tcp = current_tcp.copy()
            current_tcp, after_xy_tcp, executed_path_policy = (
                _execute_per_hole_final_motion(
                    hole_id, order, current_tcp, xy_target, planned_final_tcp,
                    compensation, args, motion_session, pose_session, timing,
                    force_safe_path=precaptured_fine is not None,
                )
            )
            final_xy_motion = {
                "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(
                    planned_final_tcp if needs_safe_lift else xy_target
                ),
                "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "tcp_position_before_mm": tcp_before,
                "pose_reference_source": (
                    "in_group_per_hole_fine_capture"
                    if precaptured_fine is not None else "current_tcp"
                ),
                "pose_reference_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(
                    pose_reference_tcp
                ),
                "hole_center_base_mm": final_point_base,
                "pose_point_base_mm": pose_point_base,
                "target_point_base_mm": final_target_point_base,
                "batch_fine_joint_applied": batch_fine_joint_applied,
                "compensation_mode": (
                    "charuco_affine_model" if use_charuco_model else
                    "fixed_offset_override" if fixed_offset is not None else "none"
                ),
                "xy_correction_mm": correction_xy,
                "charuco_model_source": str(CHARUCO_XY_MODEL_SOURCE) if use_charuco_model else None,
                "charuco_model_matrix_2x2": CHARUCO_XY_MODEL_MATRIX if use_charuco_model else None,
                "charuco_model_bias_mm": CHARUCO_XY_MODEL_BIAS_MM if use_charuco_model else None,
                "tcp_xy_offset_mm": None if fixed_offset is None else list(fixed_offset),
                "motion_path": executed_path_policy,
            }
            final_z_motion = {
                "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(planned_final_tcp),
                "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "target_base_z_mm": float(planned_final_tcp[2, 3]),
                "target_point_base_mm": final_target_point_base,
                "delta_base_z_mm": float(
                    planned_final_tcp[2, 3] - (
                        motion_start_tcp[2, 3] if after_xy_tcp is None
                        else after_xy_tcp[2, 3]
                    )
                ),
                "motion_frame": (
                    "base_z_lift_xy_guarded_z_descent"
                    if needs_safe_lift else "base_z_only"
                ),
            }
            if needs_safe_lift:
                safe_z = max(
                    float(motion_start_tcp[2, 3]),
                    float(planned_final_tcp[2, 3]) + THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM,
                )
                final_motion_direct = {
                    "path_policy": "pure_z_lift_safe_xy_guarded_pure_z_descent",
                    "planned_final_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(
                        planned_final_tcp
                    ),
                    "planned_safe_z_mm": safe_z,
                    "planned_guard_z_mm": float(planned_final_tcp[2, 3])
                    + SHARED_OBSERVATION_MIN_DESCENT_MM,
                    "safe_margin_mm": THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM,
                    "descent_guard_mm": SHARED_OBSERVATION_MIN_DESCENT_MM,
                    "actual_final_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(
                        current_tcp
                    ),
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
            "map_build_localization_mode": (
                getattr(args, "map_build_localization_mode", "per_hole")
                if bool(getattr(args, "map_build_per_hole_reference", False))
                else None
            ),
            "map_build_fine_reference": bool(
                getattr(args, "map_build_per_hole_reference", False)
            ),
            "map_build_final_motion_disabled": (
                not bool(getattr(args, "move_final_xy", False))
                if bool(getattr(args, "map_build_per_hole_reference", False))
                else None
            ),
            "batch_fine_source": (
                str(batch_fine_result.get("batch_fine_source", "batch_fine_at_260mm"))
                if batch_fine_available else
                "same_capture_340_fine"
                if bool(getattr(args, "map_build_same_capture_340", False)) else
                "per_hole_fine"
            ),
            "batch_fine_capture_round": int(
                batch_fine_result.get("batch_fine_capture_round", 0)
            ),
            "batch_fine_fallback_from_shared": bool(batch_fine_fallback_active),
            "batch_fine_fallback_reason": hole.get("batch_fine_fallback_reason"),
            "batch_fine_group_index": batch_fine_plan.get("hole_group_indices", {}).get(str(hole_id)),
            "in_group_per_hole_fallback_precaptured": bool(precaptured_fine is not None),
            "batch_coarse_fallback_reason": batch_fallback_reason,
            "batch_fine_joint_enabled": batch_fine_joint_enabled,
            "batch_fine_joint_applied": batch_fine_joint_applied,
            "batch_fine_joint_predicted_xy_base_mm": (
                None if batch_fine_joint_predicted_point_base is None else
                batch_fine_joint_predicted_point_base[:2].copy()
            ),
            "batch_fine_joint_predicted_point_base_mm": (
                batch_fine_joint_predicted_point_base
            ),
            "batch_fine_joint_visual_point_base_mm": (
                batch_fine_visual_point_before_pointcloud.copy()
                if batch_fine_joint_applied else None
            ),
            "batch_fine_joint_details": batch_fine_joint_details,
            "batch_fine_joint_summary": batch_fine_joint_summary,
            "batch_fine_visual_point_before_pointcloud_mm": (
                batch_fine_visual_point_before_pointcloud
                if batch_fine_available else None
            ),
            "batch_fine_pointcloud_fusion_applied": (
                batch_fine_pointcloud_fusion_applied
            ),
            "batch_fine_pointcloud_fused_point_base_mm": (
                final_point_base.copy()
                if batch_fine_pointcloud_fusion_applied else None
            ),
            "batch_fine_pointcloud_fusion_details": (
                batch_fine_pointcloud_fusion_details
            ),
            "batch_fine_direct_point_base_mm": batch_fine_result.get(
                "batch_fine_direct_point_base_mm"
            ),
            "coarse_source": coarse_source,
            "fine_center_source": fine.get("center_source", "unknown"),
            "fine_center_source_counts": fine.get("center_source_counts", {}),
            "fine_quality_status": fine.get("fine_quality_status", "strict"),
            "fine_quality_note": fine.get("fine_quality_note"),
            "fine_recovery_attempts": fine.get("fine_recovery_attempts", []),
            "localization_path": (
                "batch_fine_260" if batch_fine_available else
                "same_capture_340_fine"
                if bool(getattr(args, "map_build_same_capture_340", False)) else
                "per_hole_fine_260"
            ),
            "fine_stage_skipped": False,
            "batch_coarse_center_base_mm": hole.get(
                "batch_coarse_center_base_mm"
            ),
            "hole_center_base_mm": final_point_base,
            "hole_result_type": "base_frame_3d_point",
            "final_pose_source": (
                "per_hole_coarse_pose_with_joint_batch_fine_xy_pointcloud_prior_and_charuco"
                if batch_fine_joint_applied and batch_fine_pointcloud_fusion_applied else
                "per_hole_coarse_pose_with_batch_fine_xy_pointcloud_prior_and_charuco"
                if batch_fine_pointcloud_fusion_applied else
                "per_hole_coarse_pose_with_joint_batch_fine_xy_and_charuco"
                if batch_fine_joint_applied else
                "per_hole_coarse_pose_with_batch_fine_xy_only"
                if batch_fine_available else
                "per_hole_fine_pose_and_center"
            ),
            "batch_fine_xy_only": bool(batch_fine_available),
            "coarse_fine_reference_tcp_pose_m_rad": (
                transform_to_sdk_pose_m_rad(coarse_fine_reference_tcp)
                if coarse_fine_reference_tcp is not None else None
            ),
            "target_point_base_mm": final_target_point_base,
            "planned_final_point": hole.get("planned_final_point"),
            "hole_center_base_naive_mm": naive_final_point_base,
            "coarse_center_base_mm": hole["coarse_center_base_mm"],
            "coarse_center_camera_mm": hole["coarse_center_camera_mm"],
            "pointcloud_center_base_mm": hole["coarse_center_base_mm"],
            "pointcloud_center_camera_mm": hole["coarse_center_camera_mm"],
            "pointcloud_center_definition": (
                "coarse_center_ray_intersection_with_fused_local_pointcloud_plane"
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
            "coarse_ring_coverage_min_ratio": hole.get("coarse_ring_coverage_min_ratio"),
            "coarse_ring_max_gap_deg": hole.get("coarse_ring_max_gap_deg"),
            "coarse_quality_retry": hole.get("coarse_quality_retry"),
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
                "shared_batch_fine_robust_planar_joint_xy_with_limited_local_residual_and_guarded_pointcloud_prior"
                if batch_fine_joint_applied and batch_fine_pointcloud_fusion_applied else
                "shared_batch_fine_xy_with_guarded_pointcloud_prior"
                if batch_fine_pointcloud_fusion_applied else
                "shared_batch_fine_robust_planar_joint_xy_with_limited_local_residual"
                if batch_fine_joint_applied else
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
            "final_motion_direct": final_motion_direct,
            "final_motion_path": (
                "direct_safe_final_tcp"
                if final_motion_direct is not None else
                "legacy_restore_xy_z_ytrim"
                if final_xy_motion is not None else None
            ),
            "timing": timing.scoped_snapshot(f"hole_{hole_id:02d}/"),
            "tcp_target_pose_m_rad": (
                final_xy_motion["planned_tcp_pose_m_rad"]
                if final_xy_motion is not None else None
            ),
            "tracking_events": hole.get("tracking_events", []),
            "initial_detection": hole.get("initial_detection"),
        })
        for key in (
            "initial_selection_order", "operator_selection_order", "selection_source",
            "numbering_policy", "map_order", "layout_row", "layout_column",
            "numbering_center_px", "numbering_row_tolerance_px", "map_hole_key",
            "global_hole_key", "sector_id",
        ):
            if result.get(key) is None and hole.get(key) is not None:
                result[key] = hole.get(key)
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
        _write_progress_checkpoint(run_dir, report, timing=timing)
        print(
            f"[SEQUENTIAL_HOLE] order={order} hole={hole_id} "
            f"coarse_center={np.round(np.asarray(hole['coarse_center_base_mm']), 3).tolist()} "
            f"measured_final_point={np.round(np.asarray(final_point_base), 3).tolist()} "
            f"target_final_point={np.round(np.asarray(final_target_point_base), 3).tolist()} "
            f"tracking_events={len(hole.get('tracking_events', []))}",
            flush=True,
        )

        _confirm_next_hole_if_needed(
            order,
            initial_holes,
            hole_id,
            timing,
            enabled=(
                bool(getattr(args, "move_final_xy", False))
                or not all_selected_two_capture_mode
            ),
        )
    finally:
        ctx.current_tcp = current_tcp
        ctx.results = results

def _record_unexpected_hole_failure(
    ctx: Any,
    order: int,
    hole: dict[str, Any],
    exc: Exception,
) -> None:
    """把未预期异常记录为当前孔的 deferred 结果后再继续抛出异常。

    这里不把编程错误转换成“成功”，也不自动继续驱动机械臂；记录的目的
    只是让中断后的 progress/report/summary 能准确说明停在哪个孔以及还剩
    哪些孔没有处理。
    """
    try:
        hole_id = int(hole.get("hole_id", 0))
    except (TypeError, ValueError):
        hole_id = 0

    # 如果异常发生在当前孔已经写入结果之后（例如最后的人工确认阶段），
    # 不要重复追加同一个孔，避免摘要出现重复孔号。
    if any(
        int(item.get("hole_id", -1)) == hole_id
        for item in (ctx.results or [])
        if isinstance(item, dict)
        and str(item.get("hole_id", "")).lstrip("-").isdigit()
    ):
        ctx.report.setdefault("stages", {}).pop("active_hole", None)
        return

    report = ctx.report
    stages = report.setdefault("stages", {})
    error_text = f"{type(exc).__name__}: {exc}"
    stages["active_hole"] = {
        "hole_id": hole_id,
        "processing_order": int(order),
        "status": "unexpected_error",
        "error": error_text,
    }

    batch_fine_plan = getattr(ctx, "batch_fine_plan", {}) or {}
    group_indices = batch_fine_plan.get("hole_group_indices") or {}
    batch_fine_group_index = group_indices.get(
        str(hole_id), group_indices.get(hole_id),
    )
    try:
        hole_timing = ctx.timing.scoped_snapshot(f"hole_{hole_id:02d}/")
    except Exception:
        hole_timing = {"events": []}
    result = {
        "status": "deferred_unexpected_error",
        "failure_type": "unexpected_exception",
        "error": error_text,
        "hole_id": hole_id,
        "processing_order": int(order),
        "tracking_identity": hole.get("tracking_identity"),
        "initial_selection_order": hole.get("initial_selection_order"),
        "initial_center_px": hole.get("initial_center_px"),
        "initial_center_base_mm": hole.get("initial_center_base_mm"),
        "initial_plane_normal_base": hole.get("initial_plane_normal_base"),
        "coarse_source": hole.get("coarse_source", "unexpected_error"),
        "batch_coarse_requested": bool(getattr(ctx, "batch_coarse_for_cache", False)),
        "batch_fine_requested": bool(
            getattr(getattr(ctx, "cfg", None), "batch_fine_localization", False)
        ),
        "batch_fine_group_index": batch_fine_group_index,
        "batch_fine_source": "not_completed",
        "timing": hole_timing,
    }
    plan = hole.get("planned_final_point")
    if isinstance(plan, dict):
        result["planned_final_point"] = plan
        result["hole_center_base_mm"] = plan.get("visual_hole_center_base_mm")
        result["target_point_base_mm"] = plan.get("target_point_base_mm")
    try:
        _append_deferred_hole_result(
            hole,
            result,
            ctx.results,
            report,
            ctx.run_dir,
            ctx.timing,
        )
    except Exception as record_exc:
        # 保留原始异常给上层处理；最终 finally 仍会再次写 progress/report。
        stages["unexpected_hole_record_error"] = (
            f"{type(record_exc).__name__}: {record_exc}"
        )
        print(
            f"[PROGRESS_WARNING] 孔{hole_id}异常结果写入失败：{stages['unexpected_hole_record_error']}",
            flush=True,
        )


def run_per_hole_stage(ctx: Any) -> None:
    for order, hole in enumerate(ctx.initial_holes, start=1):
        try:
            hole_id = int(hole.get("hole_id", 0))
        except (TypeError, ValueError):
            hole_id = 0
        ctx.report.setdefault("stages", {})["active_hole"] = {
            "hole_id": hole_id,
            "processing_order": int(order),
            "status": "processing",
        }
        try:
            _process_one_hole(ctx, order, hole)
        except Exception as exc:
            _record_unexpected_hole_failure(ctx, order, hole, exc)
            raise
        else:
            ctx.report.setdefault("stages", {}).pop("active_hole", None)
