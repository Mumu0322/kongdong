"""Shared capture stages for sequential multi-hole localization.

This module contains planning, shared coarse capture, cache validation, shared
fine capture, and order optimization. It is deliberately independent from the
legacy runner; runtime symbols are installed by the compatibility facade.
"""

from __future__ import annotations
from aubo_workbench.coarse_recovery import queue_singleton_recaptures

from aubo_workbench.localization_errors import LocalizationHardwareError
from aubo_workbench.hole_map_seed_correction import (
    apply_runtime_aligned_seed_correction,
    fit_runtime_seed_alignment,
)
from typing import Any


_RUNTIME_DEPENDENCIES = {
    'CHARUCO_XY_MODEL_BIAS_MM',
    'CHARUCO_XY_MODEL_MATRIX',
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
    '_bounded_shared_fine_pose_adjustment',
    '_build_comparison_hole_diagnostics',
    '_cache_entry_from_observations',
    '_cache_measurements_from_observations',
    '_capture_coarse_burst',
    '_capture_fine_with_recovery',
    '_compose_batch_fine_joint_xy_with_tilt',
    '_confirm_and_move_line',
    '_confirm_next_hole_if_needed',
    '_matrix_to_rpy_zyx',
    '_move_to_batch_final_tcp_direct',
    '_move_to_fine_pose',
    '_move_to_sequential_coarse_pose',
    '_move_to_shared_coarse_pose',
    '_move_to_shared_fine_pose',
    '_move_shared_fine_group_adjustment',
    '_observation_rows',
    '_plan_batch_coarse_group_pose',
    '_plan_hole_tcp_pose_fixed_rz',
    '_project_base_point_to_pixel',
    '_record_hole_tracking_event',
    '_refine_shared_coarse_group_pose',
    '_request_next_hole_confirmation',
    '_require_safe_snapshot',
    '_wait_robot_steady_before_initial_capture',
    '_reuse_initial_pointcloud_geometry_for_batch_fine',
    '_save_batch_fine_final_result_overlay',
    '_save_coarse_pointcloud_image',
    '_save_group_capture_visualization',
    '_save_grouping_plan_visualization',
    '_settle_and_discard_coarse_recapture_frames',
    '_split_batch_fine_supplement_groups',
    '_split_batch_localization_groups_edge_first',
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


def run_preview_stage(ctx: Any) -> int | None:
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
    direct_strategy = bool(getattr(args, "coarse_direct_final", False))
    if direct_strategy:
        # 预览也必须使用第四策略的独立高度和分组上限；正式工作流通常已
        # 在构造上下文前折叠一次，这里保留直接调用本阶段时的兼容性。
        cfg = replace(
            cfg,
            coarse_direct_final=True,
            coarse_height_mm=float(getattr(
                cfg, "coarse_direct_final_height_mm", cfg.coarse_height_mm,
            )),
            batch_coarse_max_group_size=int(getattr(
                cfg, "coarse_direct_final_max_group_size", 5,
            )),
            batch_coarse_early_stop_extra_frames=int(getattr(
                cfg, "coarse_direct_final_early_stop_extra_frames", 5,
            )),
        )
        if int(cfg.batch_coarse_max_group_size) < 1 or int(
            cfg.batch_coarse_max_group_size
        ) > 5:
            raise ValueError("第四策略每组最多孔数必须在1到5之间")
        ctx.cfg = cfg
    coarse_height_label = f"{float(cfg.coarse_height_mm):g}"

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
        if len(initial_holes) > 1:
            # 预览模式也生成与正式执行完全相同的空间分组图；这里使用
            # 初始点云几何估算精定位分组，只用于检查组边界和排序，不下发运动。
            try:
                coarse_group_splitter = (
                    globals().get("_split_batch_localization_groups_edge_first")
                    if direct_strategy else _split_batch_localization_groups
                )
                if not callable(coarse_group_splitter):
                    raise RuntimeError("策略四外围优先分组函数未安装")
                preview_coarse_grouping_result = coarse_group_splitter(
                    initial_holes, current_tcp, handeye, fixed_rz_rad,
                    initial_intrinsics, cfg.coarse_height_mm,
                    cfg.batch_coarse_view_margin_px,
                    cfg.batch_coarse_max_view_span_ratio,
                    max_group_size=cfg.batch_coarse_max_group_size,
                    max_aspect_ratio=cfg.batch_coarse_group_max_aspect_ratio,
                    adjacency_distance_factor=cfg.batch_coarse_group_adjacency_factor,
                    max_normal_spread_deg=cfg.batch_coarse_group_max_normal_spread_deg,
                    max_xy_diameter_mm=cfg.batch_coarse_group_max_xy_diameter_mm,
                    preferred_min_group_size=(3 if direct_strategy else None),
                    search_timeout_s=(
                        float(getattr(cfg, "coarse_direct_final_group_planning_timeout_s", 10.0))
                        if direct_strategy else None
                    ),
                    return_diagnostics=True,
                    return_metadata=True,
                )
                (
                    preview_coarse_groups,
                    preview_coarse_diagnostics,
                    preview_coarse_grouping_metadata,
                ) = _unpack_batch_grouping_result(preview_coarse_grouping_result)
                report["stages"]["batch_coarse_plan"] = {
                    "enabled": True,
                    "preview": True,
                    "strategy": (
                        "coarse_direct_pointcloud_only" if direct_strategy
                        else "shared_coarse"
                    ),
                    "capture_height_mm": float(cfg.coarse_height_mm),
                    "target_height_mm": float(cfg.coarse_height_mm),
                    "direct_capture_only": bool(
                        getattr(args, "coarse_direct_final_capture_only", False)
                    ) if direct_strategy else False,
                    "group_count": len(preview_coarse_groups),
                    "grouping_policy": (
                        "edge_first_two_phase_boundary_exact_cover_v1"
                        if direct_strategy else
                        "minimum_feasible_exact_cover_compact_v2_spatial_groups"
                    ),
                    "grouping_search": preview_coarse_grouping_metadata,
                    "boundary_classification": (
                        preview_coarse_grouping_metadata.get("boundary_classification")
                        if direct_strategy else None
                    ),
                    "edge_first_boundary_enabled": bool(direct_strategy),
                    "edge_max_group_size": (
                        min(3, int(cfg.batch_coarse_max_group_size))
                        if direct_strategy else None
                    ),
                    "edge_hole_count": (
                        preview_coarse_grouping_metadata.get("edge_hole_count")
                        if direct_strategy else None
                    ),
                    "interior_hole_count": (
                        preview_coarse_grouping_metadata.get("interior_hole_count")
                        if direct_strategy else None
                    ),
                    "edge_group_count": (
                        preview_coarse_grouping_metadata.get("phases", {})
                        .get("edge_first", {})
                        .get("group_count")
                        if direct_strategy else None
                    ),
                    "interior_group_count": (
                        preview_coarse_grouping_metadata.get("phases", {})
                        .get("interior_after_edge", {})
                        .get("group_count")
                        if direct_strategy else None
                    ),
                    "grouping_phase_order": (
                        preview_coarse_grouping_metadata.get("grouping_phase_order")
                        if direct_strategy else None
                    ),
                    "compactness_policy": (
                        preview_coarse_grouping_metadata.get("compactness_policy")
                        if direct_strategy else None
                    ),
                    "edge_aspect_ratio_gate": (
                        preview_coarse_grouping_metadata.get("edge_aspect_ratio_gate")
                        if direct_strategy else None
                    ),
                    "grouping_order_policy": (
                        "edge_first_boundary_then_home_image_v_then_u_row_major"
                        if direct_strategy else "home_image_v_then_u_row_major"
                    ),
                    "max_group_size": int(cfg.batch_coarse_max_group_size),
                    "group_size_policy": (
                        "edge_first_max3_then_interior_max5_prefer3_to5_allow1_to2"
                        if direct_strategy else None
                    ),
                    "preferred_min_group_size": (
                        min(3, int(cfg.batch_coarse_max_group_size))
                        if direct_strategy else None
                    ),
                    "settle_delay_s": (
                        float(getattr(cfg, "coarse_direct_final_settle_delay_s", 1.0))
                        if direct_strategy else float(cfg.coarse_settle_delay_s)
                    ),
                    "steady_timeout_s": (
                        float(getattr(cfg, "coarse_direct_final_steady_timeout_s", 45.0))
                        if direct_strategy else None
                    ),
                    "settle_discard_timeout_s": (
                        float(getattr(cfg, "coarse_direct_final_settle_discard_timeout_s", 10.0))
                        if direct_strategy else None
                    ),
                    "capture_timeout_s": (
                        float(getattr(cfg, "coarse_direct_final_capture_timeout_s", 60.0))
                        if direct_strategy else None
                    ),
                    "capture_frames": int(cfg.batch_coarse_frames),
                    "early_stop_extra_frames": int(
                        cfg.batch_coarse_early_stop_extra_frames
                    ),
                    "max_view_span_ratio": float(cfg.batch_coarse_max_view_span_ratio),
                    "group_max_xy_diameter_mm": float(
                        cfg.batch_coarse_group_max_xy_diameter_mm
                    ),
                    "groups": [
                        {
                            "group_index": index,
                            "hole_ids": [int(hole["hole_id"]) for hole in group],
                            "hole_count": len(group),
                            "boundary_class": (
                                str(group[0].get("group_boundary_class", group[0].get("boundary_class", "interior")))
                                if direct_strategy and group else None
                            ),
                            "group_phase": (
                                str(group[0].get("group_phase", "interior_after_edge"))
                                if direct_strategy and group else None
                            ),
                            # 外围优先不等于放宽紧凑度；外围组同样经过
                            # 长宽比门限，近似直线的三孔会拆分。
                            "aspect_ratio_gate_disabled_for_edge": False,
                            "aspect_ratio_gate_applied_for_edge": bool(
                                direct_strategy and group
                                and str(group[0].get("group_boundary_class", group[0].get("boundary_class", "interior"))) == "edge"
                                and cfg.batch_coarse_group_max_aspect_ratio is not None
                            ),
                            "boundary_component_indices": sorted({
                                int(hole["boundary_component_index"])
                                for hole in group
                                if direct_strategy and hole.get("boundary_component_index") is not None
                            }),
                            "boundary_holes": ([
                                {
                                    "hole_id": int(hole["hole_id"]),
                                    "boundary_distance": hole.get("boundary_distance"),
                                    "boundary_distance_mm": hole.get("boundary_distance_mm"),
                                    "boundary_distance_px": hole.get("boundary_distance_px"),
                                    "boundary_distance_unit": hole.get("boundary_distance_unit"),
                                    "boundary_edge_score_deg": hole.get("boundary_edge_score_deg"),
                                    "boundary_local_degree": hole.get("boundary_local_degree"),
                                    "boundary_override": hole.get("boundary_override", False),
                                    "boundary_override_source": hole.get("boundary_override_source"),
                                    "reason": hole.get("boundary_classification_reason"),
                                }
                                for hole in group
                            ] if direct_strategy else []),
                            "group_size_reason": (
                                (
                                    "edge_preferred_3_satisfied"
                                    if len(group) >= 3 else
                                    "edge_remainder_or_geometry_constraint"
                                )
                                if group and str(group[0].get("group_boundary_class", group[0].get("boundary_class", "interior"))) == "edge"
                                else (
                                    "preferred_3_to_5_satisfied" if len(group) >= 3
                                    else (
                                        "selected_hole_count_below_preferred"
                                        if len(initial_holes) < 3 else
                                        "configured_max_group_size_below_preferred"
                                        if int(cfg.batch_coarse_max_group_size) < 3 else
                                        "remainder_or_geometry_constraint"
                                    )
                                )
                            ) if direct_strategy else None,
                            "grouping_diagnostics": preview_coarse_diagnostics[index - 1],
                        }
                        for index, group in enumerate(preview_coarse_groups, start=1)
                    ],
                    "grouping_diagnostics": preview_coarse_diagnostics,
                    "grouping_visualization": _save_grouping_plan_visualization(
                        run_dir / "01_home_selected.png",
                        run_dir / "batch_coarse_grouping_plan",
                        initial_holes,
                        preview_coarse_groups,
                        f"{coarse_height_label}mm "
                        + (
                            "POINTCLOUD-ONLY GROUP PREVIEW"
                            if direct_strategy else "SHARED COARSE GROUP PREVIEW"
                        ),
                        preview_coarse_diagnostics,
                        timing=timing,
                    ),
                }

                preview_fine_grouping_result = _split_batch_localization_groups(
                    initial_holes, current_tcp, handeye, fixed_rz_rad,
                    initial_intrinsics, cfg.fine_height_mm,
                    cfg.batch_fine_view_margin_px,
                    cfg.batch_fine_max_view_span_ratio,
                    max_group_size=cfg.batch_fine_max_group_size,
                    max_aspect_ratio=cfg.batch_fine_group_max_aspect_ratio,
                    adjacency_distance_factor=cfg.batch_fine_group_adjacency_factor,
                    max_normal_spread_deg=cfg.batch_fine_group_max_normal_spread_deg,
                    return_diagnostics=True,
                    return_metadata=True,
                )
                (
                    preview_fine_groups,
                    preview_fine_diagnostics,
                    preview_fine_grouping_metadata,
                ) = _unpack_batch_grouping_result(preview_fine_grouping_result)
                report["stages"]["batch_fine_plan"] = {
                    "enabled": True,
                    "preview": True,
                    "group_count": len(preview_fine_groups),
                    "grouping_policy": (
                        "minimum_feasible_exact_cover_compact_v2_spatial_groups"
                    ),
                    "grouping_search": preview_fine_grouping_metadata,
                    "max_group_size": int(cfg.batch_fine_max_group_size),
                    "max_view_span_ratio": float(cfg.batch_fine_max_view_span_ratio),
                    "groups": [
                        {
                            "group_index": index,
                            "hole_ids": [int(hole["hole_id"]) for hole in group],
                            "hole_count": len(group),
                            "grouping_diagnostics": preview_fine_diagnostics[index - 1],
                        }
                        for index, group in enumerate(preview_fine_groups, start=1)
                    ],
                    "grouping_diagnostics": preview_fine_diagnostics,
                    "grouping_visualization": _save_grouping_plan_visualization(
                        run_dir / "01_home_selected.png",
                        run_dir / "batch_fine_grouping_plan",
                        initial_holes,
                        preview_fine_groups,
                        "260mm SHARED FINE GROUP PREVIEW",
                        preview_fine_diagnostics,
                        timing=timing,
                    ),
                    "preview_geometry_source": "initial_selection_pointcloud_geometry",
                }
            except Exception as exc:
                report["stages"]["batch_grouping_preview_error"] = (
                    f"{type(exc).__name__}:{exc}"
                )
        report["status"] = "preview_complete"
        _write_report(run_dir, report, rows, timing=timing)
        print(f"[PREVIEW] 已写入顺序孔定位计划: {run_dir}")
        return 0
    return None

def run_shared_coarse_stage(ctx: Any) -> None:
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
    coarse_cache_gates = ctx.cache_gates
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
    direct_strategy = bool(getattr(args, "coarse_direct_final", False))
    # 第四策略的试验参数独立于普通两阶段粗定位。这里再次折叠有效值，
    # 兼容直接调用本阶段的旧测试/脚本，避免遗漏340 mm或5孔上限。
    if direct_strategy:
        cfg = replace(
            cfg,
            coarse_direct_final=True,
            coarse_height_mm=float(getattr(
                cfg, "coarse_direct_final_height_mm", cfg.coarse_height_mm,
            )),
            batch_coarse_max_group_size=int(getattr(
                cfg, "coarse_direct_final_max_group_size", 5,
            )),
            batch_coarse_early_stop_extra_frames=int(getattr(
                cfg, "coarse_direct_final_early_stop_extra_frames", 5,
            )),
        )
        # Keep the effective configuration on the shared context as well. The
        # normal workflow folds these values before constructing the context,
        # but this stage is also callable directly by compatibility runners;
        # later stages must then observe the same 340 mm/five-hole settings.
        ctx.cfg = cfg
    # 第四策略不做共享粗位姿的二次纠偏；只在规划好的独立高度观察位采集点云。
    coarse_pose_cfg = (
        replace(cfg, batch_coarse_group_pose_refinement=False)
        if direct_strategy else cfg
    )
    coarse_height_label = f"{float(cfg.coarse_height_mm):g}"

    # 地图调用只把保存的340 mm几何作为导航种子；这里不再重新建立历史
    # 粗地图，也不把历史精定位结果带入本轮。后续共享/逐孔精定位仍然会
    # 使用当前相机重新采集260 mm画面。
    coarse_map_seed_records = getattr(args, "_coarse_map_seed_results", None)
    if (
        not direct_strategy
        and str(getattr(args, "hole_map_mode", "none")) == "execute"
        and isinstance(coarse_map_seed_records, dict)
    ):
        T_base_camera = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
        R_base_camera = T_base_camera[:3, :3]
        t_base_camera = T_base_camera[:3, 3]
        seed_capture = {
            "capture_index": 0,
            "mode": "coarse_map_seed_reuse",
            "source": "coarse_hole_map",
            "tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
        }
        for hole in initial_holes:
            hole_id = int(hole["hole_id"])
            source = coarse_map_seed_records.get(hole_id)
            if not isinstance(source, dict):
                source = hole.get("coarse_map_source")
            if not isinstance(source, dict):
                raise ValueError(f"粗定位地图缺少孔 H{hole_id:02d} 的粗定位记录")
            center_base = np.asarray(
                source.get("coarse_center_base_mm"), dtype=np.float64,
            ).reshape(3)
            plane_base = np.asarray(
                source.get("coarse_plane_point_base_mm", center_base),
                dtype=np.float64,
            ).reshape(3)
            normal_base = _unit(
                np.asarray(source.get("coarse_normal_toward_camera_base"), dtype=np.float64),
                f"coarse map hole {hole_id} normal",
            )
            center_camera = R_base_camera.T @ (center_base - t_base_camera)
            plane_camera = R_base_camera.T @ (plane_base - t_base_camera)
            normal_camera = _unit(
                R_base_camera.T @ normal_base,
                f"coarse map hole {hole_id} camera normal",
            )
            batch_coarse_results[hole_id] = {
                "success": True,
                "hole_id": hole_id,
                "center_px": hole.get("initial_center_px"),
                "center_camera_mm": center_camera.tolist(),
                "center_base_mm": center_base.tolist(),
                "plane_point_camera_mm": plane_camera.tolist(),
                "plane_point_base_mm": plane_base.tolist(),
                "normal_camera": normal_camera.tolist(),
                "normal_base": normal_base.tolist(),
                "plane_rmse_mm": source.get("coarse_plane_rmse_mm"),
                "valid_frames": source.get("coarse_valid_frames"),
                "total_frames": source.get("coarse_total_frames"),
                "center_scatter_p95_px": source.get("coarse_center_scatter_p95_px"),
                "tracking_distance_p95_px": source.get("coarse_tracking_distance_p95_px"),
                "ring_points_median": source.get("coarse_ring_points_median"),
                "surface_model": source.get("coarse_surface_model"),
                "surface_selection_policy": source.get("coarse_surface_selection_policy"),
                "front_surface_z_mm": source.get("coarse_front_surface_z_mm"),
                "ring_points_raw_median": source.get("coarse_ring_points_raw_median"),
                "surface_points_selected_median": source.get(
                    "coarse_surface_points_selected_median"
                ),
                "sphere_center_camera_mm": source.get("coarse_sphere_center_camera_mm"),
                "sphere_radius_mm": source.get("coarse_sphere_radius_mm"),
                "coarse_captures": [dict(seed_capture)],
                "pointcloud_image_path": None,
                "batch_pointcloud_archive_path": None,
                "observations": [],
                "batch_observed_center_base_mm": center_base.tolist(),
                "batch_observed_plane_point_base_mm": plane_base.tolist(),
                "batch_observed_normal_base": normal_base.tolist(),
                "batch_coarse_source": "coarse_map",
            }
        report["stages"]["batch_coarse_plan"] = {
            "enabled": True,
            "source": "coarse_map",
            "capture_performed": False,
            "target_height_mm": float(cfg.coarse_height_mm),
            "hole_count": len(initial_holes),
            "hole_order": order_ids,
            "group_count": 0,
            "groups": [],
            "policy": "reuse_persisted_coarse_geometry_as_seed_then_fresh_fine",
        }
        report["stages"]["batch_coarse_results"] = {
            "success_count": len(initial_holes),
            "total_count": len(initial_holes),
            "failed_holes": [],
            "source": "coarse_map",
            "capture_performed": False,
            "results": {
                int(hole_id): {
                    "success": True,
                    "source": "coarse_map",
                    "coarse_captures": result.get("coarse_captures", []),
                }
                for hole_id, result in batch_coarse_results.items()
            },
        }
        ctx.current_tcp = current_tcp
        ctx.batch_coarse_results = batch_coarse_results
        ctx.cache_entries = cache_entries
        ctx.cache_sources = cache_sources
        ctx.cache_source_ids = cache_source_ids
        ctx.persistent_entries = persistent_entries
        ctx.cache_built_ids = cache_built_ids
        return

    if batch_coarse_for_cache and (len(initial_holes) > 1 or direct_strategy):
        print(
            f"[BATCH_COARSE] 启用批量粗定位模式: {len(initial_holes)}个孔",
            flush=True,
        )

        rgbd_pipeline, align, chain = ensure_rgbd_pipeline()

        if direct_strategy:
            # 分组搜索本身可能消耗数秒；先落一个活跃阶段标记，避免搜索
            # 期间 progress.json 看起来像流程已经停住。
            planning_stage = {
                "stage": "batch_coarse",
                "phase": "planning_groups",
                "group_count": None,
                "hole_count": len(initial_holes),
                "updated_at_unix_s": float(time.time()),
            }
            report.setdefault("stages", {})["active_stage"] = planning_stage
            try:
                timing.mark(
                    "batch_coarse/planning_groups_start",
                    hole_count=len(initial_holes),
                )
                writer = globals().get("_write_progress_checkpoint")
                if callable(writer):
                    writer(run_dir, report, timing=timing)
            except Exception:
                pass

        coarse_group_splitter = (
            globals().get("_split_batch_localization_groups_edge_first")
            if direct_strategy else _split_batch_localization_groups
        )
        if not callable(coarse_group_splitter):
            raise RuntimeError("策略四外围优先分组函数未安装")
        grouping_result = coarse_group_splitter(
            initial_holes, current_tcp, handeye, fixed_rz_rad, initial_intrinsics,
            cfg.coarse_height_mm, cfg.batch_coarse_view_margin_px,
            cfg.batch_coarse_max_view_span_ratio,
            max_group_size=cfg.batch_coarse_max_group_size,
            max_aspect_ratio=cfg.batch_coarse_group_max_aspect_ratio,
            adjacency_distance_factor=cfg.batch_coarse_group_adjacency_factor,
            max_normal_spread_deg=cfg.batch_coarse_group_max_normal_spread_deg,
            max_xy_diameter_mm=cfg.batch_coarse_group_max_xy_diameter_mm,
            preferred_min_group_size=(3 if direct_strategy else None),
            search_timeout_s=(
                float(getattr(cfg, "coarse_direct_final_group_planning_timeout_s", 10.0))
                if direct_strategy else None
            ),
            return_diagnostics=True,
            return_metadata=True,
        )
        batch_groups, batch_group_diagnostics, batch_grouping_metadata = (
            _unpack_batch_grouping_result(grouping_result)
        )
        if direct_strategy:
            planning_elapsed_s = (
                batch_grouping_metadata.get("elapsed_s")
                if isinstance(batch_grouping_metadata, dict) else None
            )
            report.setdefault("stages", {})["active_stage"] = {
                "stage": "batch_coarse",
                "phase": "groups_planned",
                "group_count": len(batch_groups),
                "hole_count": len(initial_holes),
                "planning_elapsed_s": planning_elapsed_s,
                "updated_at_unix_s": float(time.time()),
            }
            try:
                timing.mark(
                    "batch_coarse/planning_groups_complete",
                    group_count=len(batch_groups),
                    hole_count=len(initial_holes),
                    planning_elapsed_s=planning_elapsed_s,
                )
            except Exception:
                pass
            writer = globals().get("_write_progress_checkpoint")
            if callable(writer):
                try:
                    writer(run_dir, report, timing=timing)
                except Exception:
                    pass
            # 把分类审计字段同步回上下文中的原始孔记录，后续逐孔报告、
            # 缓存和结果摘要都能看到同一份外围/内部判定依据。
            original_by_id = {
                int(hole["hole_id"]): hole for hole in initial_holes
                if hole.get("hole_id") is not None
            }
            for grouped_hole in (
                hole for group in batch_groups for hole in group
            ):
                original = original_by_id.get(int(grouped_hole["hole_id"]))
                if original is None:
                    continue
                for key in (
                    "boundary_class", "boundary_layer", "boundary_component_index",
                    "boundary_edge_score_deg", "boundary_local_degree",
                    "boundary_distance", "boundary_distance_mm", "boundary_distance_px",
                    "boundary_distance_unit",
                    "boundary_coordinate_source", "boundary_classification_reason",
                    "boundary_override", "boundary_override_source",
                    "group_phase", "group_boundary_class", "edge_first_group_index",
                    "edge_first_phase_group_index",
                ):
                    if key in grouped_hole:
                        original[key] = grouped_hole[key]
            direct_group_limit = int(cfg.batch_coarse_max_group_size)
            if direct_group_limit < 1 or direct_group_limit > 5:
                raise ValueError(
                    "第四策略每组最多孔数必须在1到5之间："
                    f"{direct_group_limit}"
                )
            expected_ids = [int(hole["hole_id"]) for hole in initial_holes]
            flattened_ids = [
                int(hole["hole_id"])
                for group in batch_groups
                for hole in group
            ]
            if (
                len(flattened_ids) != len(set(flattened_ids))
                or sorted(flattened_ids) != sorted(expected_ids)
            ):
                raise RuntimeError(
                    "第四策略分组未完整覆盖所选孔或存在重复分配："
                    f"expected={sorted(expected_ids)}, actual={sorted(flattened_ids)}"
                )
            if any(
                not group or len(group) > direct_group_limit
                for group in batch_groups
            ):
                raise RuntimeError(
                    "第四策略分组大小必须在1到配置上限之间："
                    f"sizes={[len(group) for group in batch_groups]}, "
                    f"limit={direct_group_limit}"
                )
            phase_state = "edge_first"
            for group in batch_groups:
                classes = {
                    str(hole.get("group_boundary_class", hole.get("boundary_class", "edge")))
                    for hole in group
                }
                if len(classes) != 1:
                    raise RuntimeError(
                        "第四策略单组不能混合外围孔和内部孔："
                        f"hole_ids={[int(hole['hole_id']) for hole in group]}, classes={sorted(classes)}"
                    )
                component_indices = {
                    int(hole["boundary_component_index"])
                    for hole in group
                    if hole.get("boundary_component_index") is not None
                }
                if len(component_indices) > 1:
                    raise RuntimeError(
                        "第四策略单组不能跨越断开的选区："
                        f"hole_ids={[int(hole['hole_id']) for hole in group]}, "
                        f"components={sorted(component_indices)}"
                    )
                group_class = next(iter(classes))
                if group_class not in {"edge", "interior"}:
                    raise RuntimeError(
                        "第四策略分组分类值无效："
                        f"hole_ids={[int(hole['hole_id']) for hole in group]}, class={group_class}"
                    )
                if group_class == "interior":
                    phase_state = "interior_after_edge"
                elif phase_state == "interior_after_edge":
                    raise RuntimeError(
                        "第四策略必须先完成外围孔分组，再处理内部孔分组"
                    )
                if group_class == "edge" and len(group) > min(3, direct_group_limit):
                    raise RuntimeError(
                        "第四策略外围孔每组最多三个孔："
                        f"hole_ids={[int(hole['hole_id']) for hole in group]}, size={len(group)}"
                    )
            infeasible_groups = [
                {
                    "hole_ids": item.get("hole_ids", []),
                    "reason": item.get("planner_reason")
                    or item.get("geometry_reason")
                    or "planner_rejected",
                }
                for item in batch_group_diagnostics
                if isinstance(item, dict) and item.get("planner_ok") is False
            ]
            if infeasible_groups:
                raise RuntimeError(
                    "第四策略存在无法满足共同视野/几何约束的孔组，未下发运动："
                    f"{infeasible_groups}"
                )
        batch_plan: dict[str, Any] = {
            "enabled": True,
            "hole_count": len(initial_holes),
            "hole_order": order_ids,
            "target_height_mm": float(cfg.coarse_height_mm),
            "group_count": len(batch_groups),
            "all_selected_holes_single_group": len(batch_groups) == 1,
            "grouping_policy": (
                "edge_first_two_phase_boundary_exact_cover_v1"
                if direct_strategy else
                "minimum_feasible_exact_cover_compact_v2_spatial_groups"
            ),
            "max_view_span_ratio": float(cfg.batch_coarse_max_view_span_ratio),
            "max_group_size": int(cfg.batch_coarse_max_group_size),
            "group_max_aspect_ratio": float(cfg.batch_coarse_group_max_aspect_ratio),
            "group_adjacency_factor": float(cfg.batch_coarse_group_adjacency_factor),
            "group_max_xy_diameter_mm": float(
                cfg.batch_coarse_group_max_xy_diameter_mm
            ),
            "group_max_normal_spread_deg": float(
                cfg.batch_coarse_group_max_normal_spread_deg
            ),
            "grouping_order_policy": (
                "edge_first_boundary_then_home_image_v_then_u_row_major"
                if direct_strategy else "home_image_v_then_u_row_major"
            ),
            "edge_first_boundary_enabled": bool(direct_strategy),
            "edge_first_grouping_phase_order": (
                ["edge_first", "interior_after_edge"] if direct_strategy else None
            ),
            "edge_max_group_size": (
                min(3, int(cfg.batch_coarse_max_group_size))
                if direct_strategy else None
            ),
            "edge_hole_count": (
                batch_grouping_metadata.get("edge_hole_count")
                if direct_strategy else None
            ),
            "interior_hole_count": (
                batch_grouping_metadata.get("interior_hole_count")
                if direct_strategy else None
            ),
            "edge_group_count": (
                batch_grouping_metadata.get("phases", {})
                .get("edge_first", {})
                .get("group_count")
                if direct_strategy else None
            ),
            "interior_group_count": (
                batch_grouping_metadata.get("phases", {})
                .get("interior_after_edge", {})
                .get("group_count")
                if direct_strategy else None
            ),
            "group_size_policy": (
                "edge_first_max3_then_interior_max5_prefer3_to5_allow1_to2"
                if direct_strategy else None
            ),
            "preferred_min_group_size": (
                min(3, int(cfg.batch_coarse_max_group_size))
                if direct_strategy else None
            ),
            "settle_delay_s": (
                float(getattr(cfg, "coarse_direct_final_settle_delay_s", 1.0))
                if direct_strategy else float(cfg.coarse_settle_delay_s)
            ),
            "steady_timeout_s": (
                float(getattr(cfg, "coarse_direct_final_steady_timeout_s", 45.0))
                if direct_strategy else None
            ),
            "settle_discard_timeout_s": (
                float(getattr(cfg, "coarse_direct_final_settle_discard_timeout_s", 10.0))
                if direct_strategy else None
            ),
            "capture_timeout_s": (
                float(getattr(cfg, "coarse_direct_final_capture_timeout_s", 60.0))
                if direct_strategy else None
            ),
            "max_consecutive_frame_failures": (
                int(getattr(cfg, "coarse_direct_final_max_consecutive_frame_failures", 3))
                if direct_strategy else None
            ),
            "grouping_search": batch_grouping_metadata,
            "compactness_policy": (
                batch_grouping_metadata.get("compactness_policy")
                if direct_strategy else None
            ),
            "edge_aspect_ratio_gate": (
                batch_grouping_metadata.get("edge_aspect_ratio_gate")
                if direct_strategy else None
            ),
            "boundary_classification": (
                batch_grouping_metadata.get("boundary_classification")
                if direct_strategy else None
            ),
            "grouping_diagnostics": batch_group_diagnostics,
            "capture_frames": int(cfg.batch_coarse_frames),
            "early_stop_extra_frames": int(
                cfg.batch_coarse_early_stop_extra_frames
            ),
            "save_all_capture_overlays": bool(cfg.save_all_capture_overlays),
            "combined_position_policy": "projected_bbox_center_above_all_selected_holes",
            "motion_policy": (
                "map_build_pure_z_lift_configured_safe_margin_horizontal_then_"
                "separate_attitude_then_pure_z_descent_guard10_then_steady_capture"
                if bool(getattr(args, "map_build_coarse_only", False)) else
                "shared_vertical_lift_min10_safe_horizontal_descent_guard10_then_"
                "pure_descent10_then_steady_capture"
            ),
            "strategy": (
                "coarse_direct_pointcloud_only" if direct_strategy
                else "shared_coarse"
            ),
            "capture_height_label": f"{coarse_height_label}mm",
            "direct_capture_only": bool(
                getattr(args, "coarse_direct_final_capture_only", False)
            ) if direct_strategy else False,
            "safe_z_margin_mm": float(
                getattr(
                    args,
                    "map_build_safe_z_margin_mm",
                    THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM,
                )
                if bool(getattr(args, "map_build_coarse_only", False))
                else THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM
            ),
            "pose_refinement_policy": {
                "enabled": bool(coarse_pose_cfg.batch_coarse_group_pose_refinement),
                "frames": int(cfg.batch_coarse_pose_refine_frames),
                "min_valid_frames": int(cfg.batch_coarse_pose_refine_min_valid_frames),
                "max_iterations": int(cfg.batch_coarse_pose_refine_max_iterations),
                "max_step_translation_mm": float(
                    cfg.batch_coarse_pose_refine_max_correction_mm
                ),
                "max_step_rotation_deg": float(
                    cfg.batch_coarse_pose_refine_max_correction_rotation_deg
                ),
                "max_total_translation_mm": float(
                    cfg.batch_coarse_pose_refine_max_total_correction_mm
                ),
                "max_total_rotation_deg": float(
                    cfg.batch_coarse_pose_refine_max_total_correction_rotation_deg
                ),
                "uses_live_group_geometry": True,
                "nominal_pose_source": (
                    "initial_selection_pointcloud_geometry_plus_handeye"
                ),
                "first_340_capture_policy": (
                    "capture_after_nominal_navigation_then_replan_or_use_actual_tcp"
                ),
                "formal_capture_requires_its_own_quality_gate": True,
                "map_build_requires_accepted_refined_pose": bool(
                    getattr(args, "map_build_coarse_only", False)
                ),
                "map_build_accepts_safe_singleton_without_refinement": bool(
                    getattr(args, "map_build_coarse_only", False)
                ),
            },
            "view_margin_px": float(cfg.batch_coarse_view_margin_px),
            "groups": [],
        }
        batch_plan["grouping_visualization"] = _save_grouping_plan_visualization(
            run_dir / "01_home_selected.png",
            run_dir / "batch_coarse_grouping_plan",
            initial_holes,
            batch_groups,
            f"{coarse_height_label}mm "
            + (
                "POINTCLOUD-ONLY GROUP PLAN"
                if direct_strategy else "SHARED COARSE GROUP PLAN"
            ),
            batch_group_diagnostics,
            timing=timing,
        )
        batch_capture_metadata: list[dict[str, Any]] = []

        def update_direct_progress(
            group_index: int,
            hole_ids: list[int],
            phase: str,
            *,
            force_write: bool = True,
            **details: Any,
        ) -> None:
            """Publish a small, resumable marker for the active direct group."""
            if not direct_strategy:
                return
            active = {
                "stage": "batch_coarse",
                "group_index": int(group_index),
                "group_count": len(batch_groups),
                "hole_ids": list(hole_ids),
                "phase": str(phase),
                "updated_at_unix_s": float(time.time()),
                **details,
            }
            report.setdefault("stages", {})["active_stage"] = active
            print(
                f"[DIRECT_PROGRESS] group={int(group_index)}/{len(batch_groups)} "
                f"holes={list(hole_ids)} phase={str(phase)} "
                + " ".join(
                    f"{key}={value}" for key, value in details.items()
                    if key in {
                        "frame_index", "captured_frame_count", "valid_hole_count",
                        "consecutive_frame_failures", "accepted_holes",
                        "fallback_holes", "capture_stop_reason",
                    }
                ),
                flush=True,
            )
            try:
                timing.mark(
                    f"batch_coarse/group_{int(group_index):02d}/phase_{str(phase)}",
                    group_index=int(group_index),
                    hole_ids=list(hole_ids),
                    **details,
                )
            except Exception:
                pass
            if not force_write:
                return
            writer = globals().get("_write_progress_checkpoint")
            if callable(writer):
                try:
                    writer(run_dir, report, timing=timing)
                except Exception:
                    # A diagnostic write failure must not interrupt robot/camera work.
                    pass

        # 每个批量组有自己的 TCP/相机变换；缓存点云必须使用产生该
        # 组观测的变换，不能在循环结束后统一使用最后一组的位姿。
        batch_cache_transforms: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        quality_retry_attempted: set[int] = set()
        quality_retry_enabled = bool(cfg.batch_coarse_quality_singleton_retry) and not direct_strategy
        for group_index, group in enumerate(batch_groups, start=1):
            group_ids = [int(hole["hole_id"]) for hole in group]
            group_boundary_class = (
                str(group[0].get("group_boundary_class", group[0].get("boundary_class", "interior")))
                if group else "interior"
            )
            group_phase = (
                str(group[0].get("group_phase", "interior_after_edge"))
                if group else "interior_after_edge"
            )
            group_boundary_components = sorted({
                int(hole["boundary_component_index"])
                for hole in group
                if hole.get("boundary_component_index") is not None
            })
            group_boundary_reasons = sorted({
                str(hole.get("boundary_classification_reason"))
                for hole in group
                if hole.get("boundary_classification_reason")
            })
            if direct_strategy and group_boundary_class == "edge":
                group_size_reason = (
                    "edge_preferred_3_satisfied" if len(group) >= 3
                    else "edge_remainder_or_geometry_constraint"
                )
            elif direct_strategy:
                group_size_reason = (
                    "preferred_3_to_5_satisfied" if len(group) >= 3
                    else (
                        "selected_hole_count_below_preferred"
                        if len(initial_holes) < 3 else
                        "configured_max_group_size_below_preferred"
                        if int(cfg.batch_coarse_max_group_size) < 3 else
                        "remainder_or_geometry_constraint"
                    )
                )
            else:
                group_size_reason = None
            group_report: dict[str, Any] = {
                "coarse_quality_retry": any(h.get("coarse_quality_retry") for h in group),
                "coarse_quality_retry_parent_group": group[0].get("coarse_quality_retry_parent_group"),
                "group_index": group_index,
                "hole_ids": group_ids,
                "hole_count": len(group),
                "target_height_mm": float(cfg.coarse_height_mm),
                "boundary_class": group_boundary_class if direct_strategy else None,
                "group_phase": group_phase if direct_strategy else None,
                # 外围组也必须保持紧凑；不再为了凑满三孔关闭长宽比门限。
                "aspect_ratio_gate_disabled_for_edge": False,
                "aspect_ratio_gate_applied_for_edge": bool(
                    direct_strategy
                    and group_boundary_class == "edge"
                    and cfg.batch_coarse_group_max_aspect_ratio is not None
                ),
                "boundary_component_indices": group_boundary_components if direct_strategy else [],
                "boundary_classification_reasons": group_boundary_reasons if direct_strategy else [],
                "boundary_holes": ([
                    {
                        "hole_id": int(hole["hole_id"]),
                        "boundary_class": str(hole.get("boundary_class", group_boundary_class)),
                        "boundary_component_index": hole.get("boundary_component_index"),
                        "boundary_distance": hole.get("boundary_distance"),
                        "boundary_distance_mm": hole.get("boundary_distance_mm"),
                        "boundary_distance_px": hole.get("boundary_distance_px"),
                        "boundary_distance_unit": hole.get("boundary_distance_unit"),
                        "boundary_edge_score_deg": hole.get("boundary_edge_score_deg"),
                        "boundary_local_degree": hole.get("boundary_local_degree"),
                        "boundary_override": hole.get("boundary_override", False),
                        "boundary_override_source": hole.get("boundary_override_source"),
                        "reason": hole.get("boundary_classification_reason"),
                    }
                    for hole in group
                ] if direct_strategy else []),
                "accepted_holes": [],
                "fallback_holes": list(group_ids),
                "group_size_policy": (
                    "edge_first_max3_then_interior_max5_prefer3_to5_allow1_to2"
                    if direct_strategy else None
                ),
                "group_size_reason": group_size_reason,
                "settle_delay_s": (
                    float(getattr(cfg, "coarse_direct_final_settle_delay_s", 1.0))
                    if direct_strategy else float(cfg.coarse_settle_delay_s)
                ),
                "grouping_diagnostics": (
                    batch_group_diagnostics[group_index - 1]
                    if group_index - 1 < len(batch_group_diagnostics) else None
                ),
            }
            try:
                update_direct_progress(
                    group_index, group_ids, "planning", hole_count=len(group),
                )
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
                    update_direct_progress(
                        group_index, group_ids, "navigating", hole_count=len(group),
                    )
                    current_tcp = _move_to_shared_coarse_pose(
                        group_index,
                        len(batch_groups),
                        current_tcp,
                        batch_target,
                        args,
                        motion_session,
                        pose_session,
                        target_height_mm=float(cfg.coarse_height_mm),
                        steady_timeout_s=(
                            float(getattr(
                                cfg,
                                "coarse_direct_final_steady_timeout_s",
                                45.0,
                            ))
                            if direct_strategy else None
                        ),
                    )

                if direct_strategy:
                    # 第四策略不调用现场复核，也不根据复核结果修正共同位姿；
                    # 当前共同位姿只负责把相机送到本策略的点云采集高度。
                    refinement = {
                        "group": group,
                        "current_tcp": current_tcp,
                        "target": batch_target,
                        "geometry": batch_geometry,
                        "report": {
                            "enabled": False,
                            "group_index": int(group_index),
                            "group_count": len(batch_groups),
                            "iterations": [],
                            "accepted": False,
                            "reason": "disabled_for_coarse_direct_pointcloud_only",
                            "formal_capture_policy": "nominal_group_pose",
                        },
                    }
                else:
                    try:
                        refinement = _refine_shared_coarse_group_pose(
                            group_index,
                            len(batch_groups),
                            group,
                            current_tcp,
                            batch_target,
                            batch_geometry,
                            handeye,
                            rgbd_pipeline,
                            align,
                            chain,
                            model,
                            args.confidence,
                            coarse_pose_cfg,
                            initial_intrinsics,
                            run_dir,
                            timing,
                            rows,
                            args,
                            motion_session,
                            pose_session,
                            fixed_rz_rad,
                        )
                    except LocalizationHardwareError:
                        raise
                    except Exception as refinement_exc:
                        # 现场复核是共享粗定位的增强层；任何未预期的复核
                        # 异常都必须退回已验证的名义共同位姿，而不是把新功能
                        # 的异常扩大成整组粗定位失败。
                        refinement = {
                            "group": group,
                            "current_tcp": current_tcp,
                            "target": batch_target,
                            "geometry": batch_geometry,
                            "report": {
                                "enabled": bool(cfg.batch_coarse_group_pose_refinement),
                                "group_index": int(group_index),
                                "group_count": len(batch_groups),
                                "iterations": [],
                                "accepted": False,
                                "reason": (
                                    f"unexpected_refinement_error:{type(refinement_exc).__name__}:"
                                    f"{refinement_exc}"
                                ),
                                "formal_capture_policy": "nominal_group_pose",
                            },
                        }
                group = refinement["group"]
                current_tcp = np.asarray(refinement["current_tcp"], dtype=np.float64).copy()
                batch_target = np.asarray(refinement["target"], dtype=np.float64).copy()
                batch_geometry = refinement["geometry"]
                group_report["pose_refinement"] = refinement["report"]
                if (
                    bool(getattr(args, "map_build_coarse_only", False))
                    and not direct_strategy
                    and not bool((refinement.get("report") or {}).get("accepted", False))
                    and not bool((refinement.get("report") or {}).get("map_build_safe", False))
                ):
                    # 建图阶段不能把未收敛的名义组位姿写成粗定位地图。
                    # 这里在正式点云采集前中止当前建图，保留完整纠偏诊断，
                    # 由上层报告明确告诉操作者是哪个组没有收敛。
                    refine_reason = str(
                        (refinement.get("report") or {}).get(
                            "reason", "pose_refinement_not_accepted"
                        )
                    )
                    group_report["map_build_blocked"] = True
                    group_report["map_build_block_reason"] = refine_reason
                    retryable_refinement_reasons = {
                        "not_all_holes_have_stable_geometry",
                        "pose_correction_exceeds_motion_gate",
                        "total_pose_correction_exceeds_motion_gate",
                        "pose_gate_failed_after_max_iterations",
                        "pose_gate_failed_without_actionable_correction",
                    }
                    if quality_retry_enabled and refine_reason in retryable_refinement_reasons:
                        queued = queue_singleton_recaptures(batch_groups, group, group_ids,
                                                           quality_retry_attempted, group_index)
                        if queued:
                            group_report["quality_retry_holes"] = queued
                            batch_plan["groups"].append(group_report)
                            report["stages"]["batch_coarse_plan"] = batch_plan
                            print(f"[COARSE_QUALITY_RETRY] group={group_index} holes={queued} reason={refine_reason}", flush=True)
                            continue
                    raise RuntimeError(
                        f"建图粗定位第{int(group_index)}组位姿纠偏未收敛，"
                        f"拒绝保存未纠偏粗坐标：{refine_reason}"
                    )
                group_report.update({
                    "target_tcp_pose_m_rad": batch_geometry.get(
                        "target_tcp_pose_m_rad",
                        transform_to_sdk_pose_m_rad(batch_target),
                    ),
                    "combined_point_base_mm": batch_geometry.get("group_point_base_mm"),
                    "combined_normal_base": batch_geometry.get(
                        "group_normal_toward_camera_base"
                    ),
                    "projected_holes_px": {
                        str(key): np.asarray(value).tolist()
                        for key, value in batch_geometry.get(
                            "projected_holes_px", {}
                        ).items()
                    },
                    "group_bbox_px": batch_geometry.get("group_bbox_px"),
                    "group_center_px": np.asarray(
                        batch_geometry.get("group_center_px", [0.0, 0.0]),
                        dtype=np.float64,
                    ).tolist(),
                })

                if direct_strategy:
                    direct_settle_delay_s = max(
                        0.0, float(getattr(cfg, "coarse_direct_final_settle_delay_s", 1.0))
                    )
                    direct_steady_timeout_s = float(
                        getattr(cfg, "coarse_direct_final_steady_timeout_s", 45.0)
                    )
                    update_direct_progress(
                        group_index,
                        group_ids,
                        "waiting_for_steady",
                        settle_delay_s=direct_settle_delay_s,
                        timeout_s=direct_steady_timeout_s,
                    )
                    settle_waiter = globals().get(
                        "_wait_robot_steady_before_initial_capture"
                    )
                    read_snapshot = getattr(pose_session, "read_pose_snapshot", None)
                    if callable(settle_waiter) and callable(read_snapshot):
                        with timing.measure(
                            f"batch_coarse/group_{group_index:02d}/steady_settle",
                            delay_s=direct_settle_delay_s,
                            timeout_s=direct_steady_timeout_s,
                            group_index=group_index,
                        ):
                            try:
                                current_tcp = np.asarray(
                                    settle_waiter(
                                        pose_session,
                                        settle_delay_s=direct_settle_delay_s,
                                        timeout_s=direct_steady_timeout_s,
                                    ),
                                    dtype=np.float64,
                                ).reshape(4, 4).copy()
                            except TypeError as exc:
                                # Keep compatibility with test/legacy stand-ins
                                # that predate the timeout keyword.
                                if "timeout_s" not in str(exc):
                                    raise
                                current_tcp = np.asarray(
                                    settle_waiter(
                                        pose_session,
                                        settle_delay_s=direct_settle_delay_s,
                                    ),
                                    dtype=np.float64,
                                ).reshape(4, 4).copy()
                    elif direct_settle_delay_s > 0.0:
                        # Offline substitutes may not expose a pose snapshot;
                        # retain the bounded delay without pretending to verify
                        # a robot state that cannot be observed.
                        with timing.measure(
                            f"batch_coarse/group_{group_index:02d}/settle_delay",
                            delay_s=direct_settle_delay_s,
                            group_index=group_index,
                        ):
                            time.sleep(direct_settle_delay_s)
                    group_report["settle"] = {
                        "delay_s": direct_settle_delay_s,
                        "steady_timeout_s": direct_steady_timeout_s,
                        "steady_verified": bool(
                            callable(settle_waiter) and callable(read_snapshot)
                        ),
                    }
                else:
                    coarse_settle_delay_s = max(0.0, float(cfg.coarse_settle_delay_s))
                    if coarse_settle_delay_s > 0.0:
                        with timing.measure(
                            f"batch_coarse/group_{group_index:02d}/settle_delay",
                            delay_s=coarse_settle_delay_s, group_index=group_index,
                        ):
                            time.sleep(coarse_settle_delay_s)

                update_direct_progress(
                    group_index,
                    group_ids,
                    "capturing",
                    hole_count=len(group),
                    target_height_mm=float(cfg.coarse_height_mm),
                )
                capture_kwargs = {
                    "artifact_prefix": f"batch_coarse_group_{group_index:02d}",
                    "use_projected_anchor_correction": not direct_strategy,
                }
                if direct_strategy:
                    capture_kwargs["progress_callback"] = (
                        lambda payload, _group_index=group_index, _group_ids=list(group_ids):
                        update_direct_progress(
                            _group_index,
                            _group_ids,
                            str((payload or {}).get("phase", "capturing"))
                            if isinstance(payload, dict) else "capturing",
                            force_write=True,
                            **(
                                {
                                    str(key): value
                                    for key, value in (payload or {}).items()
                                    if key != "phase"
                                }
                                if isinstance(payload, dict) else {}
                            ),
                        )
                    )
                try:
                    group_results = _batch_coarse_localization_at_340mm(
                        group, current_tcp, handeye, rgbd_pipeline, align, chain,
                        model, args.confidence, cfg, initial_intrinsics, run_dir,
                        timing, rows, **capture_kwargs,
                    )
                except TypeError as exc:
                    if not direct_strategy or "progress_callback" not in str(exc):
                        raise
                    capture_kwargs.pop("progress_callback", None)
                    group_results = _batch_coarse_localization_at_340mm(
                        group, current_tcp, handeye, rgbd_pipeline, align, chain,
                        model, args.confidence, cfg, initial_intrinsics, run_dir,
                        timing, rows, **capture_kwargs,
                    )
                group_metadata = group_results.pop("_batch_metadata", {})
                update_direct_progress(
                    group_index,
                    group_ids,
                    "fusion",
                    captured_frame_count=group_metadata.get("captured_frame_count"),
                    capture_stop_reason=group_metadata.get("capture_stop_reason"),
                    force_write=True,
                )
                group_tcp = np.asarray(current_tcp, dtype=np.float64).copy()
                group_T_base_camera = camera_transform(
                    group_tcp, handeye.T_tcp_rgb_camera,
                )
                pose_refinement_report = group_report.get("pose_refinement") or {}
                batch_capture_metadata.append({
                    "group_index": group_index, "hole_ids": group_ids,
                    "capture": group_metadata,
                    "pose_refinement_accepted": bool(
                        pose_refinement_report.get("accepted", False)
                    ),
                    "pose_refinement_map_build_safe": bool(
                        pose_refinement_report.get("map_build_safe", False)
                    ),
                    "pose_refinement_reason": pose_refinement_report.get("reason"),
                })
                batch_coarse_results.update({
                    int(hole_id): result
                    for hole_id, result in group_results.items()
                    if isinstance(hole_id, int)
                })
                for hole_id in group_results:
                    if isinstance(hole_id, int):
                        group_result = group_results[hole_id]
                        group_result["coarse_quality_retry"] = bool(group_report["coarse_quality_retry"])
                        group_result["batch_coarse_group_index"] = int(group_index)
                        group_result["batch_coarse_group_hole_ids"] = list(group_ids)
                        group_result["batch_coarse_pose_refinement_accepted"] = bool(
                            pose_refinement_report.get("accepted", False)
                        )
                        group_result["batch_coarse_pose_refinement_map_build_safe"] = bool(
                            pose_refinement_report.get("map_build_safe", False)
                        )
                        group_result["batch_coarse_pose_refinement_reason"] = (
                            pose_refinement_report.get("reason")
                        )
                        group_result["batch_coarse_pose_refinement_iterations"] = int(
                            len(pose_refinement_report.get("iterations") or [])
                        )
                        if pose_refinement_report.get("final_metrics") is not None:
                            group_result["batch_coarse_pose_refinement_final_metrics"] = (
                                pose_refinement_report.get("final_metrics")
                            )
                        if direct_strategy:
                            source_hole = next(
                                (hole for hole in group if int(hole["hole_id"]) == int(hole_id)),
                                None,
                            )
                            if source_hole is not None:
                                for key in (
                                    "boundary_class", "boundary_layer",
                                    "boundary_component_index", "boundary_edge_score_deg",
                                    "boundary_local_degree", "boundary_distance",
                                    "boundary_distance_mm", "boundary_distance_px",
                                    "boundary_distance_unit", "boundary_coordinate_source",
                                    "boundary_classification_reason", "boundary_override",
                                    "boundary_override_source", "group_phase",
                                    "group_boundary_class",
                                ):
                                    if key in source_hole:
                                        group_result[key] = source_hole[key]
                        for capture in group_result.get("coarse_captures", []):
                            if isinstance(capture, dict):
                                capture.setdefault("group_index", int(group_index))
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
                # Keep complete measured coordinates even if a later group aborts.
                group_report["results"] = {
                    int(hid): {key: value for key, value in result.items()
                               if key != "observations"}
                    for hid, result in group_results.items() if isinstance(hid, int)
                }
                group_report["capture_grouping_visualization"] = (
                    _save_group_capture_visualization(
                        group_metadata.get("latest_overlay_path"),
                        run_dir / f"batch_coarse_group_{group_index:02d}_grouping_capture",
                        group_index,
                        group,
                        batch_geometry.get("projected_holes_px"),
                        batch_geometry.get("group_bbox_px"),
                        f"{coarse_height_label}mm "
                        + (
                            "POINTCLOUD-ONLY CAPTURE GROUP"
                            if direct_strategy else "SHARED COARSE CAPTURE GROUP"
                        ),
                        timing=timing,
                    )
                )
                if group_report["fallback_holes"]:
                    if quality_retry_enabled:
                        group_report["quality_retry_holes"] = queue_singleton_recaptures(
                            batch_groups, group, group_report["fallback_holes"],
                            quality_retry_attempted, group_index,
                        )
                    print(
                        f"[BATCH_COARSE] 第{group_index}组存在失败孔，其他共享组继续: "
                        f"{group_report['fallback_holes']}", flush=True,
                    )
                update_direct_progress(
                    group_index,
                    group_ids,
                    "group_complete",
                    accepted_holes=list(accepted),
                    fallback_holes=list(group_report["fallback_holes"]),
                    force_write=True,
                )
            except LocalizationHardwareError as exc:
                group_report["error"] = str(exc)
                group_report["failure_type"] = exc.failure_type
                update_direct_progress(
                    group_index,
                    group_ids,
                    "error",
                    error=str(exc),
                    failure_type=exc.failure_type,
                    force_write=True,
                )
                batch_plan["groups"].append(group_report)
                report["stages"]["batch_coarse_plan"] = batch_plan
                raise
            except Exception as exc:
                # 规划/移动/采集只影响当前视野组；其余组已经得到的结果和缓存
                # 继续保留，当前组的孔走后面的逐孔压轴兜底路径。
                group_report["error"] = f"{type(exc).__name__}:{exc}"
                group_report["fallback_reason"] = "group_failed"
                update_direct_progress(
                    group_index,
                    group_ids,
                    "error",
                    error=group_report["error"],
                    force_write=True,
                )
                if direct_strategy or bool(getattr(args, "map_build_coarse_only", False)):
                    # 第四策略和建图阶段都不允许在运动/停稳/相机阶段异常后
                    # 继续尝试后续组，避免机器人或工装状态未知时再次下发运动。
                    group_report["abort_after_group_error"] = True
                    if bool(getattr(args, "map_build_coarse_only", False)):
                        group_report["map_build_abort"] = True
                    batch_plan["groups"].append(group_report)
                    report["stages"]["batch_coarse_plan"] = batch_plan
                    raise
                print(
                    f"[BATCH_COARSE] 第{group_index}组失败；"
                    + "该组孔延后，其他共享组继续"
                    + f": {exc}",
                    flush=True,
                )
            batch_plan["groups"].append(group_report)

        batch_plan["group_count"] = len(batch_groups)
        batch_plan["quality_retry_holes"] = sorted(quality_retry_attempted)

        if direct_strategy:
            report.setdefault("stages", {})["active_stage"] = {
                "stage": "batch_coarse",
                "phase": "complete",
                "group_count": len(batch_groups),
                "completed_group_count": len(batch_plan["groups"]),
                "updated_at_unix_s": float(time.time()),
            }
            writer = globals().get("_write_progress_checkpoint")
            if callable(writer):
                try:
                    writer(run_dir, report, timing=timing)
                except Exception:
                    pass

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
                    "capture_height_mm": result.get("capture_height_mm"),
                    "batch_coarse_group_index": result.get(
                        "batch_coarse_group_index"
                    ),
                    "batch_coarse_group_hole_ids": result.get(
                        "batch_coarse_group_hole_ids"
                    ),
                    "boundary_class": result.get("boundary_class"),
                    "boundary_component_index": result.get("boundary_component_index"),
                    "boundary_distance": result.get("boundary_distance"),
                    "boundary_distance_unit": result.get("boundary_distance_unit"),
                    "boundary_classification_reason": result.get(
                        "boundary_classification_reason"
                    ),
                    "group_phase": result.get("group_phase"),
                    "valid_frames": result.get("valid_frames", 0),
                    "error": result.get("error"),
                    "error_counts": result.get("error_counts", {}),
                    "center_base_mm": result.get("center_base_mm"),
                    "normal_base": result.get("normal_base"),
                    "plane_rmse_mm": result.get("plane_rmse_mm"),
                    "ring_coverage_min_ratio": result.get("ring_coverage_min_ratio"),
                    "ring_max_gap_deg": result.get("ring_max_gap_deg"),
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
        # 批量点云采集完成后、逐孔精定位之前；逐孔回退不会覆盖这些条目。
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
                    with timing.measure_artifact(
                        "coarse_cache/write_current",
                        artifact_kind="coarse_cache_manifest",
                        paths=[str(coarse_cache_dir / "manifest.json")],
                    ):
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
                        with timing.measure_artifact(
                            "coarse_cache/write_persistent",
                            artifact_kind="persistent_coarse_cache_manifest",
                            paths=[str(persistent_cache_dir / "manifest.json")],
                        ):
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
    ctx.current_tcp = current_tcp
    ctx.batch_coarse_results = batch_coarse_results
    ctx.cache_entries = cache_entries
    ctx.cache_sources = cache_sources
    ctx.cache_source_ids = cache_source_ids
    ctx.persistent_entries = persistent_entries
    ctx.cache_built_ids = cache_built_ids

def run_shared_cache_validation_stage(ctx: Any) -> None:
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

    if bool(getattr(args, "coarse_direct_final", False)):
        report["stages"]["shared_cache_validation"] = {
            "requested": False,
            "enabled": False,
            "reason": "disabled_for_coarse_direct_pointcloud_only",
            "groups": [],
            "holes": {},
        }
        ctx.shared_cache_results = {}
        ctx.shared_cache_failed_ids = set()
        ctx.invalidated_cache_ids = set()
        return

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
            except LocalizationHardwareError:
                raise
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
    ctx.current_tcp = current_tcp
    ctx.shared_cache_results = shared_cache_results
    ctx.shared_cache_failed_ids = shared_cache_failed_ids
    ctx.invalidated_cache_ids = invalidated_cache_ids

def _capture_failed_holes_before_next_group(
    ctx: Any,
    group_index: int,
    fallback_holes: list[int],
    planning_holes_by_id: dict[int, dict[str, Any]],
    current_tcp: Any,
    group_report: dict[str, Any],
) -> Any:
    """Capture single-hole fallbacks now; defer final points and motion to hole order."""
    originals = {int(hole["hole_id"]): hole for hole in ctx.initial_holes}
    records: list[dict[str, Any]] = []
    group_report["immediate_per_hole_fallbacks"] = records
    for hole_id in fallback_holes:
        planning_hole = planning_holes_by_id.get(hole_id)
        hole = originals.get(hole_id)
        if planning_hole is None or hole is None:
            records.append({"hole_id": hole_id, "status": "missing_coarse_geometry"})
            continue
        try:
            target, _ = _plan_hole_tcp_pose_fixed_rz(
                np.asarray(planning_hole["coarse_center_base_mm"], dtype=np.float64),
                np.asarray(planning_hole["coarse_normal_toward_camera_base"], dtype=np.float64),
                current_tcp,
                ctx.handeye.T_tcp_rgb_camera,
                fixed_rz_rad=ctx.fixed_rz_rad,
                camera_height_mm=ctx.cfg.fine_height_mm,
            )
        except (ValueError, RuntimeError) as exc:
            # No motion was issued. The original per-hole path remains available.
            records.append({
                "hole_id": hole_id,
                "status": "planning_failed_deferred_to_per_hole_stage",
                "error": f"{type(exc).__name__}:{exc}",
            })
            continue
        with ctx.timing.measure(
            f"hole_{hole_id:02d}/in_group_move_to_fine_pose",
            hole_id=hole_id, group_index=group_index,
        ):
            current_tcp = _move_to_fine_pose(
                str(hole_id), current_tcp, target,
                ctx.args, ctx.motion_session, ctx.pose_session,
                target_height_mm=ctx.cfg.fine_height_mm,
                target_stage="本组失败孔单孔精定位",
                safe_margin_mm=ctx.cfg.per_hole_fine_safe_z_margin_mm,
                descent_guard_mm=(
                    SHARED_OBSERVATION_MIN_DESCENT_MM
                    if ctx.all_selected_two_capture_mode else 0.0
                ),
            )
        plane_base = np.asarray(
            planning_hole["coarse_plane_point_base_mm"], dtype=np.float64,
        ).reshape(3)
        for correction_index in range(ctx.cfg.max_z_corrections):
            _, measured_tcp = _require_safe_snapshot(ctx.pose_session)
            height = camera_height_to_plane_mm(
                measured_tcp, ctx.handeye.T_tcp_rgb_camera, plane_base,
            )
            current_tcp = measured_tcp
            if abs(height - ctx.cfg.fine_height_mm) <= ctx.cfg.height_tolerance_mm:
                break
            z_target, _ = base_z_target_for_camera_height(
                measured_tcp, ctx.handeye.T_tcp_rgb_camera,
                plane_base, ctx.cfg.fine_height_mm,
            )
            with ctx.timing.measure(
                f"hole_{hole_id:02d}/in_group_fine_height_correction_{correction_index + 1}",
                hole_id=hole_id, group_index=group_index,
            ):
                current_tcp = _confirm_and_move_line(
                    f"孔{hole_id}组内补拍纯Z高度修正",
                    measured_tcp, z_target, ctx.args,
                    ctx.motion_session, ctx.pose_session,
                    "只修正基坐标Z；XY和姿态不变",
                    require_confirmation=False, motion_profile="approach",
                )
        _, current_tcp = _require_safe_snapshot(ctx.pose_session)
        height = camera_height_to_plane_mm(
            current_tcp, ctx.handeye.T_tcp_rgb_camera, plane_base,
        )
        if abs(height - ctx.cfg.fine_height_mm) > ctx.cfg.height_tolerance_mm:
            raise RuntimeError(
                f"孔{hole_id}组内补拍未达到精拍高度：{height:.2f} mm"
            )
        capture_tcp = np.asarray(current_tcp, dtype=np.float64).copy()
        camera = camera_transform(capture_tcp, ctx.handeye.T_tcp_rgb_camera)
        expected_anchor = _project_base_point_to_pixel(
            np.asarray(planning_hole["coarse_center_base_mm"], dtype=np.float64),
            camera, ctx.initial_intrinsics,
        )
        pipeline, _, _ = ctx.ensure_rgbd_pipeline()
        recovery = _capture_fine_with_recovery(
            pipeline, ctx.model, ctx.args.confidence,
            hole["initial_detection"], ctx.cfg, ctx.run_dir,
            hole, hole_id,
            int(hole.get("initial_selection_order") or hole_id),
            expected_anchor, ctx.timing, ctx.rows,
        )
        ctx.in_group_fine_recoveries[hole_id] = {
            "capture_tcp": capture_tcp,
            "expected_anchor_px": expected_anchor,
            "height_mm": height,
            "recovery": recovery,
            "group_index": group_index,
        }
        records.append({
            "hole_id": hole_id,
            "status": "captured" if recovery["success"] else "quality_failed",
            "fine_quality_status": recovery.get("status"),
            "error": recovery.get("error"),
            "capture_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(capture_tcp),
        })
        print(
            f"[BATCH_FINE] 第{group_index}组孔{hole_id}已立即完成逐孔补拍；"
            f"quality={records[-1]['status']}",
            flush=True,
        )
    return current_tcp


def run_shared_fine_stage(ctx: Any) -> None:
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

    if bool(getattr(args, "coarse_direct_final", False)):
        batch_fine_plan = {
            "enabled": False,
            "requested": False,
            "hole_count": len(initial_holes),
            "groups": [],
            "fallback_holes": [],
            "per_hole_fallback_enabled": False,
            "pointcloud_xy_fusion_enabled": False,
            "reason": "disabled_for_coarse_direct_pointcloud_only",
        }
        report["stages"]["batch_fine_plan"] = batch_fine_plan
        ctx.batch_fine_results = {}
        ctx.batch_fine_plan = batch_fine_plan
        return

    batch_fine_results: dict[int, dict[str, Any]] = {}
    batch_fine_plan: dict[str, Any] = {
        "enabled": False,
        "requested": bool(cfg.batch_fine_localization and len(initial_holes) > 1),
        "hole_count": len(initial_holes),
        "groups": [],
        "fallback_holes": [],
        "initial_pointcloud_reused_holes": initial_pointcloud_reused_holes,
        "coarse_geometry_policy": "fresh_shared_340mm_capture_required",
        "failure_policy": (
            "bounded_in_group_pose_adjustment_then_immediate_per_hole_capture"
            if cfg.batch_fine_per_hole_fallback else
            "bounded_in_group_pose_adjustment_then_defer"
        ),
        "joint_failure_policy": (
            "capture_failed_hole_before_next_group_with_charuco_final_later"
        ),
        "per_hole_fallback_timing": "after_own_group_before_next_group",
        "combined_position_policy": "projected_bbox_center_from_all_coarse_holes",
        "motion_policy": "shared_vertical_lift_min10_safe_horizontal_descent_guard10_then_pure_descent10_then_steady_capture",
        "final_pose_policy": (
            "per_hole_coarse_z_and_orientation_with_joint_batch_fine_xy_and_charuco_"
            "then_direct_safe_final_tcp_when_final_motion_enabled"
        ),
        "final_motion_policy": (
            "batch_fine_direct_safe_tcp: pure_z_lift_then_safe_horizontal_to_final_"
            "xy_pose_then_pure_z_guard10_then_pure_z_final10;"
            "legacy_path_kept_for_non_batch_fine_modes"
        ),
        "fine_output_components": ["base_x", "base_y"],
        "coarse_output_components": ["base_z", "rx", "ry", "rz"],
        "supplement_rounds": int(cfg.batch_fine_supplement_rounds),
        "supplement_policy": (
            "one_bounded_direct_xy_z_rx_ry_adjustment_for_failed_hole_subset"
            if cfg.batch_fine_in_group_pose_adjustment else
            "spatially_cluster_failed_holes_for_distinct_260mm_views"
        ),
        "supplement_max_view_span_ratio": float(
            cfg.batch_fine_supplement_max_view_span_ratio
        ),
        "in_group_pose_adjustment_enabled": bool(
            cfg.batch_fine_in_group_pose_adjustment
        ),
        "in_group_max_adjustments": int(
            cfg.batch_fine_in_group_max_adjustments
        ),
        "in_group_max_xy_mm": float(cfg.batch_fine_in_group_max_xy_mm),
        "in_group_max_z_mm": float(cfg.batch_fine_in_group_max_z_mm),
        "in_group_max_rotation_deg": float(
            cfg.batch_fine_in_group_max_rotation_deg
        ),
        "in_group_min_normal_holes": int(
            cfg.batch_fine_in_group_min_normal_holes
        ),
        "in_group_max_normal_spread_deg": float(
            cfg.batch_fine_in_group_max_normal_spread_deg
        ),
        "inplace_recovery_frames": int(cfg.batch_fine_inplace_recovery_frames),
        "inplace_recovery_policy": (
            "continue_at_current_260mm_pose_for_unlocked_holes_before_moving"
            if int(cfg.batch_fine_inplace_recovery_frames) > 0 else
            "disabled"
        ),
        "max_group_size": int(cfg.batch_fine_max_group_size),
        "group_max_aspect_ratio": float(cfg.batch_fine_group_max_aspect_ratio),
        "group_adjacency_factor": float(cfg.batch_fine_group_adjacency_factor),
        "group_max_normal_spread_deg": float(
            cfg.batch_fine_group_max_normal_spread_deg
        ),
        "per_hole_fallback_enabled": bool(cfg.batch_fine_per_hole_fallback),
        "geometric_anchor_distance_gate_px": float(
            cfg.batch_fine_max_geometric_anchor_distance_px
        ),
        "fallback_coarse_to_fine_xy_gate_mm": float(
            cfg.batch_fine_fallback_max_coarse_to_fine_xy_mm
        ),
        "joint_localization_enabled": bool(cfg.batch_fine_joint_localization),
        "joint_method": (
            "per_hole_temporal_robust_fusion_then_group_xy_transform"
        ),
        "joint_min_holes": int(cfg.batch_fine_joint_min_holes),
        "joint_min_valid_frames": int(cfg.batch_fine_joint_min_valid_frames),
        "joint_max_residual_mm": float(cfg.batch_fine_joint_max_residual_mm),
        "joint_max_translation_mm": float(cfg.batch_fine_joint_max_translation_mm),
        "joint_max_yaw_deg": float(cfg.batch_fine_joint_max_yaw_deg),
        "joint_stable_translation_mm": float(
            cfg.batch_fine_joint_stable_translation_mm
        ),
        "joint_stable_yaw_deg": float(cfg.batch_fine_joint_stable_yaw_deg),
        "joint_local_residual_weight": float(
            cfg.batch_fine_joint_local_residual_weight
        ),
        "joint_local_residual_limit_mm": float(
            cfg.batch_fine_joint_local_residual_limit_mm
        ),
        "pointcloud_xy_fusion_enabled": bool(
            cfg.batch_fine_pointcloud_xy_fusion
        ),
        "pointcloud_xy_weight": float(cfg.batch_fine_pointcloud_xy_weight),
        "pointcloud_xy_weight_policy": (
            "joint_residual_adaptive_0.4_minimum_below_0.3_"
            "full_weight_above_0.5"
        ),
        "pointcloud_xy_max_correction_mm": float(
            cfg.batch_fine_pointcloud_xy_max_correction_mm
        ),
        "pointcloud_xy_agreement_gate_mm": float(
            cfg.batch_fine_pointcloud_xy_agreement_gate_mm
        ),
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
            if all_selected_two_capture_mode:
                # 双共享模式的260 mm联合反投影与中心/法向一样，必须
                # 完整使用本轮同一拍340 mm粗定位的平面，禁止旧字段混入。
                plane_value = (
                    coarse_result.get("plane_point_base_mm")
                    if coarse_result.get("success") else None
                )
            else:
                plane_value = hole.get("coarse_plane_point_base_mm")
                if plane_value is None and coarse_result.get("success"):
                    plane_value = coarse_result.get("plane_point_base_mm")
            if plane_value is None:
                # 粗定位结果正常时通常必然存在平面点；仅保留中心作为
                # 最后兼容回退，避免改变旧缓存/测试数据的进入条件。
                plane_value = point_value
            fine_planning_holes.append({
                **hole,
                # 共同260mm位姿规划器复用已有的批量视野算法；这里把
                # 粗定位后的可信几何映射到规划器要求的通用字段。
                "initial_center_base_mm": np.asarray(point_value, dtype=np.float64).copy(),
                "initial_plane_normal_base": np.asarray(normal_value, dtype=np.float64).copy(),
                "planning_normal_base": np.asarray(normal_value, dtype=np.float64).copy(),
                # 联合精定位要在每个孔自己的粗定位平面上反投影，不能
                # 使用共享260mm综合位姿的平面替代孔级粗几何。
                "coarse_plane_point_base_mm": np.asarray(
                    plane_value, dtype=np.float64,
                ).copy(),
                # 显式保留本轮粗拍点云支撑中心。联合变换和最终点云先验
                # 使用同一个源，避免 ``initial_center`` 名字造成数据含义混淆。
                "coarse_center_base_mm": np.asarray(
                    point_value, dtype=np.float64,
                ).copy(),
                "coarse_normal_toward_camera_base": np.asarray(
                    normal_value, dtype=np.float64,
                ).copy(),
            })
        fine_planning_holes_by_id = {
            int(hole["hole_id"]): hole for hole in fine_planning_holes
        }

        batch_fine_plan.update({
            "hole_count_with_coarse_geometry": len(fine_planning_holes),
            "missing_coarse_geometry_holes": missing_fine_geometry,
            "view_margin_px": float(cfg.batch_fine_view_margin_px),
            "target_height_mm": float(cfg.fine_height_mm),
            "frames": int(cfg.batch_fine_frames),
            "min_valid_frames": int(cfg.batch_fine_min_valid),
            "stable_min_frames": int(cfg.batch_fine_stable_min_frames),
            "settle_discard_frames": int(cfg.batch_fine_settle_discard_frames),
            "save_all_capture_overlays": bool(cfg.save_all_capture_overlays),
            "supplement_rounds": int(cfg.batch_fine_supplement_rounds),
            "in_group_pose_adjustment_enabled": bool(
                cfg.batch_fine_in_group_pose_adjustment
            ),
            "max_view_span_ratio": float(cfg.batch_fine_max_view_span_ratio),
            "max_group_size": int(cfg.batch_fine_max_group_size),
            "group_max_aspect_ratio": float(cfg.batch_fine_group_max_aspect_ratio),
            "group_adjacency_factor": float(cfg.batch_fine_group_adjacency_factor),
            "group_max_normal_spread_deg": float(
                cfg.batch_fine_group_max_normal_spread_deg
            ),
            "grouping_order_policy": "home_image_v_then_u_row_major",
            "per_hole_fallback_enabled": bool(cfg.batch_fine_per_hole_fallback),
        })
        if fine_planning_holes:
            fine_grouping_result = _split_batch_localization_groups(
                fine_planning_holes, current_tcp, handeye, fixed_rz_rad,
                initial_intrinsics, cfg.fine_height_mm,
                cfg.batch_fine_view_margin_px,
                cfg.batch_fine_max_view_span_ratio,
                max_group_size=cfg.batch_fine_max_group_size,
                max_aspect_ratio=cfg.batch_fine_group_max_aspect_ratio,
                adjacency_distance_factor=cfg.batch_fine_group_adjacency_factor,
                max_normal_spread_deg=cfg.batch_fine_group_max_normal_spread_deg,
                return_diagnostics=True,
                return_metadata=True,
            )
            (
                batch_fine_groups,
                batch_fine_group_diagnostics,
                batch_fine_grouping_metadata,
            ) = _unpack_batch_grouping_result(fine_grouping_result)
            batch_fine_plan.update({
                "enabled": True,
                "group_count": len(batch_fine_groups),
                "all_selected_holes_single_group": len(batch_fine_groups) == 1,
                "grouping_policy": (
                    "minimum_feasible_exact_cover_compact_v2_spatial_groups"
                ),
                "grouping_search": batch_fine_grouping_metadata,
                "grouping_diagnostics": batch_fine_group_diagnostics,
            })
            batch_fine_plan["grouping_visualization"] = _save_grouping_plan_visualization(
                run_dir / "01_home_selected.png",
                run_dir / "batch_fine_grouping_plan",
                initial_holes,
                batch_fine_groups,
                "260mm SHARED FINE GROUP PLAN",
                batch_fine_group_diagnostics,
                timing=timing,
            )
            rgbd_pipeline, align, chain = ensure_rgbd_pipeline()
            fine_hole_groups_by_id: dict[int, int] = {}
            seed_correction_model = getattr(
                args, "_coarse_map_seed_correction_model", None,
            )
            live_anchor_xy_by_hole: dict[int, np.ndarray] = {}
            live_anchor_group_by_hole: dict[int, int] = {}
            live_seed_alignment: dict[str, Any] = {
                "success": False,
                "reason": (
                    "waiting_for_live_anchors"
                    if isinstance(seed_correction_model, dict) else
                    "seed_correction_model_not_loaded"
                ),
            }
            seed_correction_report: dict[str, Any] = {
                "enabled": isinstance(seed_correction_model, dict),
                "policy": (
                    "historical_reference_requires_current_run_alignment_before_use"
                ),
                "static_reference_applied": False,
                "reference_run_id": (
                    (seed_correction_model.get("reference") or {}).get("run_id")
                    if isinstance(seed_correction_model, dict) else None
                ),
                "alignment_history": [],
                "latest_alignment": live_seed_alignment,
                "applied_capture_groups": [],
                "applied_supplement_groups": [],
            }
            batch_fine_plan["seed_correction"] = seed_correction_report

            def _runtime_corrected_group(
                raw_group: list[dict[str, Any]],
            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
                if not isinstance(seed_correction_model, dict):
                    return list(raw_group), []
                corrected: list[dict[str, Any]] = []
                details: list[dict[str, Any]] = []
                for raw_hole in raw_group:
                    planned_hole, detail = apply_runtime_aligned_seed_correction(
                        raw_hole, seed_correction_model, live_seed_alignment,
                    )
                    corrected.append(planned_hole)
                    details.append(detail)
                return corrected, details

            def _update_live_seed_alignment(
                values: dict[Any, dict[str, Any]],
                capture_group_index: int,
                capture_label: str,
            ) -> None:
                nonlocal live_seed_alignment
                if not isinstance(seed_correction_model, dict):
                    return
                accepted_anchor_ids: list[int] = []
                for raw_hole_id, result in values.items():
                    if not isinstance(raw_hole_id, int) or not isinstance(result, dict):
                        continue
                    fine = result.get("fine") or {}
                    direct_point = result.get("batch_fine_direct_point_base_mm")
                    if (
                        direct_point is None
                        or str(fine.get("fine_quality_status")) != "strict"
                        or int(fine.get("valid_frames", 0)) < int(cfg.batch_fine_min_valid)
                    ):
                        continue
                    point = np.asarray(direct_point, dtype=np.float64).reshape(3)
                    if not np.isfinite(point).all():
                        continue
                    hole_id = int(raw_hole_id)
                    live_anchor_xy_by_hole[hole_id] = point[:2].copy()
                    live_anchor_group_by_hole[hole_id] = int(capture_group_index)
                    accepted_anchor_ids.append(hole_id)
                candidate = fit_runtime_seed_alignment(
                    seed_correction_model,
                    live_anchor_xy_by_hole,
                    live_anchor_group_by_hole,
                )
                history_item = {
                    "capture": capture_label,
                    "new_anchor_hole_ids": sorted(accepted_anchor_ids),
                    **candidate,
                }
                seed_correction_report["alignment_history"].append(history_item)
                if bool(candidate.get("success", False)):
                    live_seed_alignment = candidate
                seed_correction_report["latest_alignment"] = live_seed_alignment

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
                        "joint_success": bool(
                            result.get("batch_fine_joint_success", False)
                        ),
                        "joint_translation_mm": (
                            (result.get("batch_fine_joint_summary") or {}).get(
                                "translation_mm"
                            )
                        ),
                        "joint_yaw_deg": (
                            (result.get("batch_fine_joint_summary") or {}).get(
                                "yaw_deg"
                            )
                        ),
                        "joint_residual_p95_mm": (
                            (result.get("batch_fine_joint_summary") or {}).get(
                                "residual_p95_mm",
                                (result.get("batch_fine_joint_summary") or {}).get(
                                    "frame_residual_p95_mm"
                                ),
                            )
                        ),
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
                    "grouping_diagnostics": (
                        batch_fine_group_diagnostics[group_index - 1]
                        if group_index - 1 < len(batch_fine_group_diagnostics) else None
                    ),
                }
                for hole_id in group_ids:
                    fine_hole_groups_by_id[hole_id] = group_index
                try:
                    planning_group, planning_corrections = _runtime_corrected_group(
                        group
                    )
                    applied_planning_corrections = [
                        item for item in planning_corrections
                        if bool(item.get("applied", False))
                    ]
                    group_report["seed_correction"] = {
                        "live_alignment_ready": bool(
                            live_seed_alignment.get("success", False)
                        ),
                        "holes": planning_corrections,
                        "applied_hole_ids": [
                            int(item["hole_id"])
                            for item in applied_planning_corrections
                        ],
                    }
                    if applied_planning_corrections:
                        seed_correction_report["applied_capture_groups"].append({
                            "group_index": group_index,
                            "hole_ids": [
                                int(item["hole_id"])
                                for item in applied_planning_corrections
                            ],
                            "max_correction_mm": max(
                                float(item["correction_norm_mm"])
                                for item in applied_planning_corrections
                            ),
                        })
                    batch_target, batch_geometry = _plan_batch_coarse_group_pose(
                        planning_group, current_tcp, handeye, fixed_rz_rad,
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
                                planning_group, current_tcp, handeye, rgbd_pipeline, model,
                                args.confidence, cfg, initial_intrinsics, run_dir,
                                timing, rows,
                                artifact_prefix=f"batch_fine_group_{group_index:02d}",
                                additional_capture_frames=(
                                    cfg.batch_fine_inplace_recovery_frames
                                ),
                            )
                    except LocalizationHardwareError:
                        raise
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
                    _update_live_seed_alignment(
                        group_results,
                        group_index,
                        f"group_{group_index:02d}_initial",
                    )
                    if initial_capture_error is not None and not group_results:
                        for hole_id in group_ids:
                            batch_fine_results[hole_id] = {
                                "success": False,
                                "error": initial_capture_error,
                                "batch_fine_source": "batch_fine_at_260mm",
                                "batch_fine_capture_round": 0,
                            }
                    group_report["capture"] = group_metadata
                    group_report["capture_grouping_visualization"] = (
                        _save_group_capture_visualization(
                            group_metadata.get("latest_overlay_path"),
                            run_dir / f"batch_fine_group_{group_index:02d}_grouping_capture",
                            group_index,
                            planning_group,
                            batch_geometry.get("projected_holes_px"),
                            batch_geometry.get("group_bbox_px"),
                            "260mm SHARED FINE CAPTURE GROUP",
                            timing=timing,
                        )
                    )
                    group_report["initial_results"] = _summarize_batch_fine_results(
                        group_results
                    )

                    # 首拍中只有质量门失败的孔进入补拍。默认把失败孔保持为
                    # 原组的一个子集，在当前260mm附近执行一次受限位姿调整；
                    # 关闭该策略时才恢复旧的空间拆组和安全高度补拍路径。
                    pending_holes = [
                        hole_id for hole_id in group_ids
                        if not batch_fine_results.get(hole_id, {}).get("success", False)
                    ]
                    supplement_round_limit = max(
                        0, int(cfg.batch_fine_supplement_rounds),
                    )
                    if cfg.batch_fine_in_group_pose_adjustment:
                        supplement_round_limit = min(
                            supplement_round_limit,
                            max(0, int(cfg.batch_fine_in_group_max_adjustments)),
                        )
                    continue_in_group_adjustment = True
                    for supplement_round in range(1, supplement_round_limit + 1):
                        if not pending_holes:
                            break
                        if (
                            cfg.batch_fine_in_group_pose_adjustment
                            and supplement_round > 1
                            and not continue_in_group_adjustment
                        ):
                            break
                        round_made_progress = False
                        round_had_clipped_motion = False
                        pending_group_raw = [
                            fine_planning_holes_by_id[hole_id]
                            for hole_id in pending_holes
                            if hole_id in fine_planning_holes_by_id
                        ]
                        if not pending_group_raw:
                            group_report["supplement_captures"].append({
                                "round": supplement_round,
                                "hole_ids": list(pending_holes),
                                "hole_count": 0,
                                "accepted_holes": [],
                                "fallback_holes": list(pending_holes),
                                "error": (
                                "补拍缺少可用粗定位几何，无法规划共享260mm补拍位"
                                ),
                            })
                            break
                        pending_group, pending_corrections = _runtime_corrected_group(
                            pending_group_raw
                        )
                        pending_correction_by_hole = {
                            int(item["hole_id"]): item
                            for item in pending_corrections
                        }
                        if cfg.batch_fine_in_group_pose_adjustment:
                            # 原始组已经通过视野/紧凑度门，失败孔子集不会比原组
                            # 更大。保持为一个子组，避免再次拆组产生多次机械臂
                            # 运动；只有首步被限幅或缩小了失败子集时才允许第二步。
                            supplement_groups = [pending_group]
                            supplement_group_diagnostics = [{
                                "policy": "original_group_failed_subset_kept_together",
                                "hole_ids": [
                                    int(hole["hole_id"]) for hole in pending_group
                                ],
                            }]
                        else:
                            supplement_groups, supplement_group_diagnostics = _split_batch_fine_supplement_groups(
                                pending_group, current_tcp, handeye, fixed_rz_rad,
                                initial_intrinsics, cfg.fine_height_mm,
                                cfg.batch_fine_view_margin_px,
                                cfg.batch_fine_supplement_max_view_span_ratio,
                                max_group_size=cfg.batch_fine_max_group_size,
                                max_aspect_ratio=cfg.batch_fine_group_max_aspect_ratio,
                                adjacency_distance_factor=cfg.batch_fine_group_adjacency_factor,
                                max_normal_spread_deg=cfg.batch_fine_group_max_normal_spread_deg,
                                return_diagnostics=True,
                            )
                        for cluster_index, supplement_group in enumerate(
                            supplement_groups, start=1,
                        ):
                            cluster_ids = [
                                int(hole["hole_id"]) for hole in supplement_group
                            ]
                            supplement_report: dict[str, Any] = {
                                "round": supplement_round,
                                "cluster_index": cluster_index,
                                "cluster_count": len(supplement_groups),
                                "hole_ids": cluster_ids,
                                "hole_count": len(supplement_group),
                                "accepted_holes": [],
                                "fallback_holes": list(cluster_ids),
                                "max_view_span_ratio": float(
                                    cfg.batch_fine_supplement_max_view_span_ratio
                                ),
                                "grouping_policy": (
                                    "original_group_failed_subset_kept_together"
                                    if cfg.batch_fine_in_group_pose_adjustment else
                                    "minimum_feasible_exact_cover_compact_v2_spatial_groups"
                                ),
                                "grouping_diagnostics": (
                                    supplement_group_diagnostics[cluster_index - 1]
                                    if cluster_index - 1 < len(supplement_group_diagnostics)
                                    else None
                                ),
                                "seed_correction": {
                                    "live_alignment_ready": bool(
                                        live_seed_alignment.get("success", False)
                                    ),
                                    "holes": [
                                        pending_correction_by_hole[hole_id]
                                        for hole_id in cluster_ids
                                        if hole_id in pending_correction_by_hole
                                    ],
                                },
                            }
                            applied_supplement_corrections = [
                                pending_correction_by_hole[hole_id]
                                for hole_id in cluster_ids
                                if hole_id in pending_correction_by_hole
                                and bool(pending_correction_by_hole[hole_id].get(
                                    "applied", False
                                ))
                            ]
                            if applied_supplement_corrections:
                                seed_correction_report[
                                    "applied_supplement_groups"
                                ].append({
                                    "group_index": group_index,
                                    "round": supplement_round,
                                    "cluster_index": cluster_index,
                                    "hole_ids": [
                                        int(item["hole_id"])
                                        for item in applied_supplement_corrections
                                    ],
                                    "max_correction_mm": max(
                                        float(item["correction_norm_mm"])
                                        for item in applied_supplement_corrections
                                    ),
                                })
                            artifact_suffix = (
                                f"supplement_{supplement_round:02d}"
                                f"_cluster_{cluster_index:02d}"
                            )
                            try:
                                supplement_target, supplement_geometry = (
                                    _plan_batch_coarse_group_pose(
                                        supplement_group, current_tcp, handeye,
                                        fixed_rz_rad, initial_intrinsics,
                                        cfg.fine_height_mm,
                                        cfg.batch_fine_view_margin_px,
                                    )
                                )
                                supplement_report.update({
                                    "planned_target_tcp_pose_m_rad": supplement_geometry[
                                        "target_tcp_pose_m_rad"
                                    ],
                                    "combined_point_base_mm": supplement_geometry.get(
                                        "group_point_base_mm"
                                    ),
                                    "combined_normal_base": supplement_geometry.get(
                                        "group_normal_toward_camera_base"
                                    ),
                                    "normal_spread_deg": supplement_geometry.get(
                                        "normal_spread_deg"
                                    ),
                                    "pose_policy": (
                                        "bounded_direct_failed_subset_pose_adjustment"
                                        if cfg.batch_fine_in_group_pose_adjustment else
                                        "above_compact_failed_hole_cluster_bbox_center"
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
                                executed_target = supplement_target
                                rotation_allowed = False
                                if cfg.batch_fine_in_group_pose_adjustment:
                                    normal_spread = supplement_geometry.get(
                                        "normal_spread_deg"
                                    )
                                    rotation_allowed = bool(
                                        len(supplement_group) >= int(
                                            cfg.batch_fine_in_group_min_normal_holes
                                        )
                                        and normal_spread is not None
                                        and np.isfinite(float(normal_spread))
                                        and float(normal_spread) <= float(
                                            cfg.batch_fine_in_group_max_normal_spread_deg
                                        )
                                    )
                                    executed_target, adjustment_details = (
                                        _bounded_shared_fine_pose_adjustment(
                                            current_tcp,
                                            supplement_target,
                                            max_xy_mm=(
                                                cfg.batch_fine_in_group_max_xy_mm
                                            ),
                                            max_z_mm=(
                                                cfg.batch_fine_in_group_max_z_mm
                                            ),
                                            max_rotation_deg=(
                                                cfg.batch_fine_in_group_max_rotation_deg
                                                if rotation_allowed else 0.0
                                            ),
                                        )
                                    )
                                    adjustment_details.update({
                                        "rotation_allowed": rotation_allowed,
                                        "normal_hole_count": len(supplement_group),
                                        "normal_spread_deg": normal_spread,
                                        "min_normal_holes": int(
                                            cfg.batch_fine_in_group_min_normal_holes
                                        ),
                                        "max_normal_spread_deg": float(
                                            cfg.batch_fine_in_group_max_normal_spread_deg
                                        ),
                                    })
                                    effective_motion = bool(
                                        float(adjustment_details["applied_xy_mm"]) > 0.05
                                        or abs(float(adjustment_details["applied_z_mm"])) > 0.05
                                        or float(adjustment_details["applied_rotation_deg"]) > 0.05
                                    )
                                    adjustment_details["effective_motion"] = effective_motion
                                    adjustment_details["followup_adjustment_candidate"] = bool(
                                        effective_motion
                                        and (
                                            adjustment_details["translation_clipped"]
                                            or adjustment_details["rotation_clipped"]
                                        )
                                    )
                                    round_had_clipped_motion = bool(
                                        round_had_clipped_motion
                                        or adjustment_details[
                                            "followup_adjustment_candidate"
                                        ]
                                    )
                                    supplement_report[
                                        "in_group_pose_adjustment"
                                    ] = adjustment_details
                                supplement_report["target_tcp_pose_m_rad"] = (
                                    transform_to_sdk_pose_m_rad(executed_target)
                                )
                                capture_projected_holes = dict(
                                    supplement_geometry["projected_holes_px"]
                                )
                                capture_group_bbox = list(
                                    supplement_geometry["group_bbox_px"]
                                )
                                if cfg.batch_fine_in_group_pose_adjustment:
                                    executed_view_rejection: str | None = None
                                    try:
                                        executed_camera = camera_transform(
                                            executed_target,
                                            handeye.T_tcp_rgb_camera,
                                        )
                                        capture_projected_holes = {
                                            int(hole["hole_id"]): _project_base_point_to_pixel(
                                                np.asarray(
                                                    hole["initial_center_base_mm"],
                                                    dtype=np.float64,
                                                ),
                                                executed_camera,
                                                initial_intrinsics,
                                            )
                                            for hole in supplement_group
                                        }
                                        projected_values = np.asarray(
                                            list(capture_projected_holes.values()),
                                            dtype=np.float64,
                                        )
                                        bbox_min = np.min(projected_values, axis=0)
                                        bbox_max = np.max(projected_values, axis=0)
                                        capture_group_bbox = [
                                            float(bbox_min[0]), float(bbox_min[1]),
                                            float(bbox_max[0]), float(bbox_max[1]),
                                        ]
                                        supplement_report.update({
                                            "executed_projected_holes_px": {
                                                str(key): value.tolist()
                                                for key, value in (
                                                    capture_projected_holes.items()
                                                )
                                            },
                                            "executed_group_bbox_px": capture_group_bbox,
                                        })
                                        image_width = float(getattr(
                                            initial_intrinsics, "width", 0.0,
                                        ))
                                        image_height = float(getattr(
                                            initial_intrinsics, "height", 0.0,
                                        ))
                                        view_margin = float(
                                            cfg.batch_fine_view_margin_px
                                        )
                                        out_of_view_holes = [
                                            int(hole_id)
                                            for hole_id, point in (
                                                capture_projected_holes.items()
                                            )
                                            if (
                                                not np.isfinite(point).all()
                                                or float(point[0]) < view_margin
                                                or float(point[0]) > image_width - view_margin
                                                or float(point[1]) < view_margin
                                                or float(point[1]) > image_height - view_margin
                                            )
                                        ]
                                        supplement_report[
                                            "executed_out_of_view_holes"
                                        ] = out_of_view_holes
                                        if image_width <= 0.0 or image_height <= 0.0:
                                            executed_view_rejection = (
                                                "组内位姿微调缺少有效图像尺寸，拒绝运动"
                                            )
                                        elif out_of_view_holes:
                                            executed_view_rejection = (
                                                "组内受限位姿微调会使失败孔离开安全视野："
                                                f"{out_of_view_holes}"
                                            )
                                    except Exception as projection_exc:
                                        supplement_report[
                                            "executed_projection_error"
                                        ] = (
                                            f"{type(projection_exc).__name__}:"
                                            f"{projection_exc}"
                                        )
                                        supplement_report[
                                            "executed_view_validation"
                                        ] = (
                                            "projection_unavailable_using_planner_"
                                            "validated_target_view"
                                        )
                                    if executed_view_rejection is not None:
                                        supplement_report[
                                            "executed_view_validation"
                                        ] = "rejected"
                                        raise RuntimeError(executed_view_rejection)
                                with timing.measure(
                                    f"batch_fine/group_{group_index:02d}/{artifact_suffix}/navigate_to_260mm",
                                    hole_count=len(supplement_group),
                                    group_index=group_index,
                                    supplement_round=supplement_round,
                                    supplement_cluster=cluster_index,
                                ):
                                    if cfg.batch_fine_in_group_pose_adjustment:
                                        current_tcp = _move_shared_fine_group_adjustment(
                                            "共享精定位组内失败孔位姿微调",
                                            current_tcp,
                                            executed_target,
                                            args,
                                            motion_session,
                                            pose_session,
                                            max_xy_mm=cfg.batch_fine_in_group_max_xy_mm,
                                            max_z_mm=cfg.batch_fine_in_group_max_z_mm,
                                            max_rotation_deg=(
                                                cfg.batch_fine_in_group_max_rotation_deg
                                                if rotation_allowed else 0.0
                                            ),
                                        )
                                    else:
                                        current_tcp = _move_to_shared_fine_pose(
                                            current_tcp, executed_target,
                                            args, motion_session, pose_session,
                                            target_height_mm=cfg.fine_height_mm,
                                            target_stage="共享精定位分组补拍",
                                        )
                                with timing.measure(
                                    f"batch_fine/group_{group_index:02d}/{artifact_suffix}/capture_and_localize",
                                    hole_count=len(supplement_group),
                                    group_index=group_index,
                                    supplement_round=supplement_round,
                                    supplement_cluster=cluster_index,
                                ):
                                    supplement_results = _batch_fine_localization_at_260mm(
                                        supplement_group, current_tcp, handeye,
                                        rgbd_pipeline, model, args.confidence, cfg,
                                        initial_intrinsics, run_dir, timing, rows,
                                        artifact_prefix=(
                                            f"batch_fine_group_{group_index:02d}_"
                                            f"{artifact_suffix}"
                                        ),
                                        additional_capture_frames=(
                                            cfg.batch_fine_inplace_recovery_frames
                                        ),
                                    )
                                supplement_metadata = supplement_results.pop(
                                    "_batch_metadata", {}
                                )
                                for hole_id, result in supplement_results.items():
                                    if not isinstance(hole_id, int):
                                        continue
                                    result["batch_fine_source"] = (
                                        "batch_fine_in_group_pose_adjustment_at_260mm"
                                        if cfg.batch_fine_in_group_pose_adjustment else
                                        "batch_fine_clustered_supplement_at_260mm"
                                    )
                                    result["batch_fine_capture_round"] = supplement_round
                                    result["batch_fine_supplement_cluster"] = cluster_index
                                    batch_fine_results[int(hole_id)] = result
                                _update_live_seed_alignment(
                                    supplement_results,
                                    group_index,
                                    (
                                        f"group_{group_index:02d}_supplement_"
                                        f"{supplement_round:02d}_cluster_"
                                        f"{cluster_index:02d}"
                                    ),
                                )
                                supplement_report["capture"] = supplement_metadata
                                supplement_report["capture_grouping_visualization"] = (
                                    _save_group_capture_visualization(
                                        supplement_metadata.get("latest_overlay_path"),
                                        run_dir / (
                                            f"batch_fine_group_{group_index:02d}_"
                                            f"{artifact_suffix}_grouping_capture"
                                        ),
                                        cluster_index,
                                        supplement_group,
                                        capture_projected_holes,
                                        capture_group_bbox,
                                        "260mm SHARED FINE SUPPLEMENT GROUP",
                                        timing=timing,
                                    )
                                )
                                supplement_report["results"] = (
                                    _summarize_batch_fine_results(supplement_results)
                                )
                                supplement_report["accepted_holes"] = [
                                    int(hole_id)
                                    for hole_id, result in supplement_results.items()
                                    if isinstance(hole_id, int)
                                    and result.get("success", False)
                                ]
                                round_made_progress = bool(
                                    round_made_progress
                                    or supplement_report["accepted_holes"]
                                )
                                supplement_report["fallback_holes"] = [
                                    hole_id for hole_id in cluster_ids
                                    if not batch_fine_results.get(hole_id, {}).get(
                                        "success", False
                                    )
                                ]
                                print(
                                    f"[BATCH_FINE] 第{group_index}组第{supplement_round}轮"
                                    f"第{cluster_index}/{len(supplement_groups)}补拍完成；"
                                    f"accepted={supplement_report['accepted_holes']} "
                                    f"fallback={supplement_report['fallback_holes']}",
                                    flush=True,
                                )
                            except LocalizationHardwareError:
                                raise
                            except Exception as exc:
                                supplement_report["error"] = (
                                    f"{type(exc).__name__}:{exc}"
                                )
                                supplement_report["fallback_reason"] = (
                                    "batch_fine_in_group_pose_adjustment_failed"
                                    if cfg.batch_fine_in_group_pose_adjustment else
                                    "batch_fine_clustered_supplement_failed"
                                )
                                for hole_id in cluster_ids:
                                    previous = dict(batch_fine_results.get(hole_id, {}))
                                    previous.update({
                                        "success": False,
                                        "error": supplement_report["error"],
                                        "batch_fine_source": (
                                            "batch_fine_in_group_pose_adjustment_at_260mm"
                                            if cfg.batch_fine_in_group_pose_adjustment else
                                            "batch_fine_clustered_supplement_at_260mm"
                                        ),
                                        "batch_fine_capture_round": supplement_round,
                                        "batch_fine_supplement_cluster": cluster_index,
                                    })
                                    batch_fine_results[hole_id] = previous
                                print(
                                    f"[BATCH_FINE] 第{group_index}组第{supplement_round}轮"
                                    f"第{cluster_index}组补拍失败: "
                                    f"{supplement_report['error']}",
                                    flush=True,
                                )
                            group_report["supplement_captures"].append(
                                supplement_report
                            )
                        pending_holes = [
                            hole_id for hole_id in group_ids
                            if not batch_fine_results.get(hole_id, {}).get(
                                "success", False
                            )
                        ]
                        # A second direct adjustment is useful only when the
                        # first step was clipped, or when that capture accepted
                        # part of the group and left a smaller failed subset to
                        # re-centre. A no-op full-group recapture must not turn
                        # into another identical robot motion/capture cycle.
                        continue_in_group_adjustment = bool(
                            pending_holes
                            and (round_had_clipped_motion or round_made_progress)
                        )

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
                except LocalizationHardwareError as exc:
                    group_report["error"] = str(exc)
                    group_report["failure_type"] = exc.failure_type
                    batch_fine_plan["groups"].append(group_report)
                    report["stages"]["batch_fine_plan"] = batch_fine_plan
                    raise
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
                if (
                    cfg.batch_fine_per_hole_fallback
                    and group_report.get("fallback_holes")
                    and not group_report.get("error")
                ):
                    # Stay in this group's vicinity. The later per-hole stage
                    # consumes these RGB results but retains its original final
                    # motion and operator-confirmation order.
                    current_tcp = _capture_failed_holes_before_next_group(
                        ctx, group_index, group_report["fallback_holes"],
                        fine_planning_holes_by_id, current_tcp, group_report,
                    )

            batch_fine_plan["fallback_holes"] = sorted({
                *missing_fine_geometry,
                *[
                    hole_id for group in batch_fine_plan["groups"]
                    for hole_id in group.get("fallback_holes", [])
                ],
            })
            batch_fine_plan["per_hole_fallback_holes"] = (
                list(batch_fine_plan["fallback_holes"])
                if cfg.batch_fine_per_hole_fallback else []
            )
            batch_fine_plan["hole_group_indices"] = {
                str(hole_id): group_index
                for hole_id, group_index in fine_hole_groups_by_id.items()
            }
        else:
            batch_fine_plan["fallback_holes"] = missing_fine_geometry
            batch_fine_plan["per_hole_fallback_holes"] = (
                list(missing_fine_geometry)
                if cfg.batch_fine_per_hole_fallback else []
            )
            batch_fine_plan["reason"] = "no_hole_has_reliable_coarse_geometry"
    elif len(initial_holes) <= 1:
        batch_fine_plan["reason"] = "single_hole"
    else:
        batch_fine_plan["reason"] = "batch_fine_disabled"
    ctx.current_tcp = current_tcp
    ctx.batch_fine_results = batch_fine_results
    ctx.batch_fine_plan = batch_fine_plan

def optimize_hole_order_stage(ctx: Any) -> None:
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

    report["stages"]["batch_fine_plan"] = batch_fine_plan
    report["stages"]["batch_fine_results"] = {
        "success_count": sum(
            1 for result in batch_fine_results.values() if result.get("success", False)
        ),
        "total_count": len(initial_holes),
        "fallback_holes": batch_fine_plan.get("fallback_holes", []),
        "per_hole_fallback_holes": batch_fine_plan.get(
            "per_hole_fallback_holes", []
        ),
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
    ctx.initial_holes = initial_holes
    ctx.order_ids = order_ids
