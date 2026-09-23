"""Sequential multi-hole orchestration.

The public runner remains the compatibility facade. It installs the facade's
runtime symbols here before dispatch so existing tests, GUI integrations, and
site-specific dependency patches continue to affect the workflow.
"""

from __future__ import annotations

from aubo_workbench import sequential_workflow_stages as _workflow_stages


_RUNTIME_DEPENDENCIES = {
    "CHARUCO_XY_MODEL_BIAS_MM",
    "CHARUCO_XY_MODEL_MATRIX",
    "CHARUCO_XY_MODEL_SOURCE",
    "COARSE_SURFACE_MODEL",
    "COARSE_SURFACE_SELECTION_POLICY",
    "CacheValidationGates",
    "FINAL_BASE_Y_AFTER_Z_MM",
    "FINAL_TOOL_Y_AFTER_Z_MM",
    "HOLE_DIAMETERS_MM",
    "SHARED_OBSERVATION_MIN_DESCENT_MM",
    "THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM",
    "_angle_deg",
    "_apply_cached_geometry_to_hole",
    "_apply_coarse_geometry_to_hole",
    "_batch_coarse_localization_at_340mm",
    "_batch_fine_localization_at_260mm",
    "_build_comparison_hole_diagnostics",
    "_build_comparison_run_diagnostics",
    "_cache_entry_from_observations",
    "_cache_measurements_from_observations",
    "_capture_coarse_burst",
    "_capture_fine_with_recovery",
    "_compose_batch_fine_joint_xy_with_tilt",
    "_confirm_and_move_line",
    "_matrix_to_rpy_zyx",
    "_move_to_batch_final_tcp_direct",
    "_move_to_fine_pose",
    "_move_to_sequential_coarse_pose",
    "_move_to_shared_coarse_pose",
    "_move_to_shared_fine_pose",
    "_observation_rows",
    "_plan_batch_coarse_group_pose",
    "_plan_hole_tcp_pose_fixed_rz",
    "_project_base_point_to_pixel",
    "_record_hole_tracking_event",
    "_refine_shared_coarse_group_pose",
    "_request_next_hole_confirmation",
    "_require_safe_snapshot",
    "_reuse_initial_pointcloud_geometry_for_batch_fine",
    "_save_batch_fine_final_result_overlay",
    "_save_coarse_pointcloud_image",
    "_save_group_capture_visualization",
    "_save_grouping_plan_visualization",
    "_settle_and_discard_coarse_recapture_frames",
    "_split_batch_fine_supplement_groups",
    "_split_batch_localization_groups",
    "_split_shared_cache_validation_groups",
    "_unit",
    "_unpack_batch_grouping_result",
    "_upsert_persistent_cache_entry",
    "_validate_coarse_cache_at_current_pose",
    "_write_progress_checkpoint",
    "_write_report",
    "base_z_target_for_camera_height",
    "camera_height_to_plane_mm",
    "camera_transform",
    "compose_batch_fine_xy_with_coarse_z",
    "correct_projected_circle_center",
    "fuse_batch_fine_xy_with_pointcloud_prior",
    "init_pipeline",
    "load_cache_entries",
    "load_persistent_cache_entries",
    "math",
    "np",
    "optimize_hole_order",
    "pixel_to_base_plane",
    "plan_final_tcp_base_z",
    "plan_final_tcp_combined_y_trim",
    "plan_final_tcp_xy",
    "rekey_cache_entry",
    "replace",
    "save_cache_entries",
    "save_persistent_cache_entries",
    "time",
    "transform_cached_points_to_camera",
    "transform_to_sdk_pose_m_rad",
    "validate_cache_entry",
}


def install_runtime(symbols: dict[str, object]) -> None:
    for name in _RUNTIME_DEPENDENCIES:
        if name in symbols:
            globals()[name] = symbols[name]
    # Stage helpers live in this compatibility module, while camera/robot
    # symbols arrive from the legacy runner. Merge both namespaces so extracted
    # stages keep the same patch points as the old monolithic function.
    _workflow_stages.install_runtime({**symbols, **globals()})


def _append_deferred_hole_result(
    hole: dict[str, Any],
    result: dict[str, Any],
    results: list[dict[str, Any]],
    report: dict[str, Any],
    run_dir: Path,
    timing: TimingRecorder,
) -> None:
    """统一记录 deferred 结果，保证各失败分支写入相同的进度状态。"""
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
    results.append(result)
    hole_id = int(result["hole_id"])
    report["stages"][f"hole_{hole_id}"] = result
    report.setdefault("deferred_holes", []).append(hole_id)
    report["stages"]["processed_holes"] = {
        "completed_count": sum(item.get("status") == "completed" for item in results),
        "deferred_count": sum(item.get("status") != "completed" for item in results),
        "total_count": len(results),
        "hole_order": [int(item["hole_id"]) for item in results],
        "holes": results,
    }
    _write_progress_checkpoint(run_dir, report, timing=timing)


def _confirm_next_hole_if_needed(
    order: int,
    initial_holes: list[dict[str, Any]],
    hole_id: int,
    timing: TimingRecorder,
    *,
    enabled: bool = True,
    stop_message: str = "完成",
) -> None:
    """等待操作者确认进入下一个孔，统一各分支的停止语义。"""
    if not enabled or order >= len(initial_holes):
        return
    next_hole_id = int(initial_holes[order]["hole_id"])
    with timing.measure(
        f"hole_{hole_id:02d}/wait_next_hole_confirmation",
        category="operator_wait",
        level="leaf",
        hole_id=hole_id,
        next_hole_id=next_hole_id,
    ):
        command = _request_next_hole_confirmation(hole_id, next_hole_id)
    if command != "m":
        raise RuntimeError(f"用户在孔{hole_id}{stop_message}后停止流程")


def _finalize_sequential_results(
    results: list[dict[str, Any]],
    order_ids: list[int],
    all_selected_two_capture_mode: bool,
    fixed_rz_rad: float,
    batch_fine_results: dict[int, dict[str, Any]],
    handeye: Any,
    cfg: TwoStageConfig,
    current_tcp: np.ndarray,
    runtime: dict[str, Any],
    report: dict[str, Any],
    run_dir: Path,
    rows: list[dict[str, Any]],
    timing: TimingRecorder,
) -> int:
    """写入本轮最终摘要并返回流程退出码。"""
    completed_results = [item for item in results if item.get("status") == "completed"]
    capture_only_results = [item for item in results if item.get("status") == "capture_only"]
    deferred_results = [
        item for item in results if item.get("status", "").startswith("deferred_")
    ]
    coarse_direct_results = [
        item for item in results
        if str(item.get("localization_path", "")).startswith("coarse_")
        and item.get("coarse_direct_decision") in {
            "coarse_direct", "pointcloud_direct", "capture_only",
        }
    ]
    coarse_direct_deferred_results = [
        item for item in results
        if item.get("status") == "deferred_coarse_pointcloud"
    ]
    coarse_first_strategy = bool(
        report.get("configuration", {}).get("coarse_direct_final", False)
    )
    report["stages"]["sequential_holes"] = {
        "mode": (
            f"coarse_{float(cfg.coarse_height_mm):g}_pointcloud_center_only_no_fallback_edge_first"
            if coarse_first_strategy else
            "grouped_batch_fine_captures_then_compute_all_holes"
            if all_selected_two_capture_mode else
            "one_hole_complete_then_next"
        ),
        "hole_count": len(results),
        "completed_count": len(completed_results),
        "capture_only_count": len(capture_only_results),
        "deferred_count": len(deferred_results),
        "failed_holes": [int(item["hole_id"]) for item in deferred_results],
        "hole_order": order_ids,
        "fixed_rz_rad": fixed_rz_rad,
        "tracking_identity_source": "initial_selection_order_and_locked_projected_anchor",
        "holes": results,
        "coarse_direct_count": len(coarse_direct_results),
        "coarse_direct_pointcloud_deferred_count": len(
            coarse_direct_deferred_results
        ),
    }
    timing.mark(
        "cycle/complete",
        cycle_index=int(report.get("cycle_index", 0)),
        completed_count=len(completed_results),
        deferred_count=len(deferred_results),
    )
    if coarse_first_strategy:
        final_overlay = {
            "status": "skipped",
            "reason": "coarse_direct_pointcloud_only",
        }
    else:
        try:
            final_overlay = _save_batch_fine_final_result_overlay(
                run_dir, batch_fine_results, results, handeye, timing=timing,
            )
        except Exception as exc:
            final_overlay = {"error": f"{type(exc).__name__}:{exc}"}
            print(f"[FINAL_RESULT_OVERLAY_WARNING] {exc}", flush=True)
    report["stages"]["batch_fine_final_result_overlay"] = final_overlay
    if final_overlay and final_overlay.get("image_path"):
        for item in results:
            if str(item.get("batch_fine_source", "")).startswith("batch_fine"):
                item["final_result_overlay_path"] = final_overlay["image_path"]
    report["final_result"] = {
        "hole_count": len(results),
        "completed_count": len(completed_results),
        "capture_only_count": len(capture_only_results),
        "deferred_count": len(deferred_results),
        "failed_holes": [int(item["hole_id"]) for item in deferred_results],
        "hole_order": order_ids,
        "fixed_rz_rad": fixed_rz_rad,
        "holes": results,
        "final_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
        "coarse_direct_count": len(coarse_direct_results),
        "coarse_direct_pointcloud_deferred_count": len(
            coarse_direct_deferred_results
        ),
    }
    report["fine_stage_skipped"] = bool(coarse_first_strategy)
    report["comparison_diagnostics"] = _build_comparison_run_diagnostics(results, report, cfg)
    runtime["current_tcp"] = np.asarray(current_tcp, dtype=np.float64).copy()
    if capture_only_results and not deferred_results and not completed_results:
        report["status"] = "coarse_direct_capture_only_complete"
    elif deferred_results:
        report["status"] = (
            "completed_with_deferred_holes_experimental_handeye"
            if not handeye.validated_for_motion else "completed_with_deferred_holes"
        )
    else:
        report["status"] = (
            "completed_experimental_handeye"
            if not handeye.validated_for_motion else "completed"
        )
    _write_report(run_dir, report, rows, timing=timing)
    if capture_only_results and not deferred_results and not completed_results:
        print(
            f"\n[顺序孔定位结果] 仅采集评估完成："
            f"采集={len(capture_only_results)}，未执行最终XY/Z动作。",
            flush=True,
        )
    elif deferred_results:
        print(
            f"\n[顺序孔定位结果] 流程已继续完成：成功={len(completed_results)}，"
            f"deferred={len(deferred_results)}；deferred孔未执行最终目标点运动。",
            flush=True,
        )
    else:
        if coarse_first_strategy:
            print(
                f"\n[顺序孔定位结果] 已完成{float(cfg.coarse_height_mm):g} mm点云中心直达流程："
                f"粗中心直达={len(coarse_direct_results)}，"
                f"点云中心不可用={len(coarse_direct_deferred_results)}。"
            )
        else:
            print("\n[顺序孔定位结果] 已按初始孔号逐个完成粗定位、精定位和目标点运动。")
    print(
        f"[DONE] 结果目录: {run_dir}；"
        f"成功={len(completed_results)}，仅采集={len(capture_only_results)}，"
        f"延后/失败={len(deferred_results)}；"
        "详细结果请查看 report.json，简明结果请查看 result_summary.txt。",
        flush=True,
    )
    return 0


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
    persistent_cache_entries: dict[int, CoarseCacheEntry] | None = None,
    persistent_cache_dir: Path | None = None,
) -> int:
    """Run all sequential stages through one explicit workflow context."""
    if not initial_holes:
        raise RuntimeError("没有初始选定孔，无法执行顺序定位")

    direct_strategy = bool(getattr(args, "coarse_direct_final", False))
    if direct_strategy:
        # 第四策略的唯一输入是本轮独立高度点云中心；关闭共享精定位、缓存和
        # 逐孔回退，避免旧参数把流程重新带到260mm或历史几何。
        direct_height_mm = float(getattr(
            cfg, "coarse_direct_final_height_mm", cfg.coarse_height_mm,
        ))
        direct_group_size = int(getattr(
            cfg, "coarse_direct_final_max_group_size", 5,
        ))
        direct_extra_frames = int(getattr(
            cfg, "coarse_direct_final_early_stop_extra_frames", 5,
        ))
        cfg = replace(
            cfg,
            coarse_direct_final=True,
            coarse_height_mm=direct_height_mm,
            batch_coarse_max_group_size=direct_group_size,
            batch_coarse_early_stop_extra_frames=direct_extra_frames,
            batch_coarse_localization=True,
            batch_fine_localization=False,
            batch_fine_joint_localization=False,
            batch_fine_pointcloud_xy_fusion=False,
            batch_coarse_group_pose_refinement=False,
            shared_cache_validation=False,
            batch_fine_per_hole_fallback=False,
        )
        args.batch_coarse_localization = True
        args.batch_fine_localization = False
        args.batch_fine_joint_localization = False
        args.batch_fine_pointcloud_xy_fusion = False
        args.shared_cache_validation = False
        args.reuse_coarse_cache = False
        args.reuse_persistent_coarse_cache = False
        if bool(getattr(args, "coarse_direct_final_capture_only", False)):
            args.move_final_xy = False

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
    cache_enabled = (
        not direct_strategy
        and bool(getattr(args, "reuse_coarse_cache", True))
        and coarse_cache_dir is not None
    )
    persistent_enabled = bool(
        cache_enabled
        and getattr(args, "reuse_persistent_coarse_cache", True)
        and persistent_cache_dir is not None
    )
    persistent_entries = persistent_cache_entries if persistent_cache_entries is not None else {}
    cache_built_ids: set[int] = set(
        int(value) for value in report.get("coarse_cache", {}).get("cache_built", [])
    )
    for key in (
        "cache_reused", "persistent_cache_loaded", "persistent_cache_reused",
        "cache_validation_skipped", "cache_validation_failed", "cache_invalidated",
        "full_coarse_fallback",
    ):
        report.setdefault("coarse_cache", {}).setdefault(key, [])
    report["stages"]["sequential_plan"] = {
        "mode": (
            f"coarse_{float(cfg.coarse_height_mm):g}_pointcloud_center_only_no_fallback_edge_first"
            if direct_strategy else
            "per_hole_same_capture_340_coarse_and_strict_fine_with_quality_gated_pointcloud_fallback_reference"
            if bool(getattr(args, "map_build_same_capture_340", False)) else
            "per_hole_coarse_340_then_fine_260_reference_only"
            if bool(getattr(args, "map_build_per_hole_reference", False)) else
            "grouped_shared_coarse_then_grouped_shared_fine_with_per_hole_fallback"
        ),
        "hole_count": len(initial_holes),
        "hole_order": order_ids,
        "fixed_rz_rad": fixed_rz_rad,
        "coarse_settle_buffer_s": float(
            cfg.coarse_direct_final_settle_delay_s
            if direct_strategy else cfg.coarse_settle_delay_s
        ),
        "tracking_identity_source": "initial_selection_order_then_shared_group_live_refined_3d_projection",
        "confirmation_policy": "no_per_hole_pause_after_batch_fine_capture",
        "camera_pipeline_policy": "reuse_single_rgbd_pipeline_for_coarse_and_rgb_fine",
        "initial_pointcloud_reused_holes": initial_pointcloud_reused_holes,
        "capture_policy": (
            f"{float(cfg.coarse_height_mm):g}mm_pointcloud_center_only_direct_no_fallback_edge_first"
            if direct_strategy else
            "each_selected_hole_fresh_340mm_rgbd_coarse_and_strict_rgb_fine_same_capture;"
            "failed_fine_uses_quality_gated_pointcloud_center_or_defers_without_final_motion"
            if bool(getattr(args, "map_build_same_capture_340", False)) else
            "each_selected_hole_fresh_340mm_coarse_then_260mm_fine;"
            "fine_reference_persisted_without_final_motion"
            if bool(getattr(args, "map_build_per_hole_reference", False)) else
            "spatially_group_all_selected_holes_at_each_stage_then_failed_holes"
            "_in_compact_shared_supplement_groups_then_per_hole_fine_fallback"
        ),
        "group_limits": {
            "coarse_max_holes": int(cfg.batch_coarse_max_group_size),
            "fine_max_holes": int(cfg.batch_fine_max_group_size),
            "coarse_max_aspect_ratio": float(cfg.batch_coarse_group_max_aspect_ratio),
            "fine_max_aspect_ratio": float(cfg.batch_fine_group_max_aspect_ratio),
            "coarse_max_xy_diameter_mm": float(cfg.batch_coarse_group_max_xy_diameter_mm),
            "coarse_max_view_span_ratio": float(cfg.batch_coarse_max_view_span_ratio),
            "coarse_max_normal_spread_deg": float(
                cfg.batch_coarse_group_max_normal_spread_deg
            ),
            "coarse_adjacency_factor": float(cfg.batch_coarse_group_adjacency_factor),
        },
        "capture_height_mm": float(cfg.coarse_height_mm),
        "coarse_direct_final_max_group_size": (
            int(cfg.batch_coarse_max_group_size) if direct_strategy else None
        ),
        "coarse_direct_final_settle_delay_s": (
            float(cfg.coarse_direct_final_settle_delay_s)
            if direct_strategy else None
        ),
        "coarse_direct_final_group_planning_timeout_s": (
            float(cfg.coarse_direct_final_group_planning_timeout_s)
            if direct_strategy else None
        ),
        "coarse_direct_final_settle_discard_timeout_s": (
            float(cfg.coarse_direct_final_settle_discard_timeout_s)
            if direct_strategy else None
        ),
        "coarse_direct_final_capture_timeout_s": (
            float(cfg.coarse_direct_final_capture_timeout_s)
            if direct_strategy else None
        ),
        "coarse_direct_final_steady_timeout_s": (
            float(cfg.coarse_direct_final_steady_timeout_s)
            if direct_strategy else None
        ),
        "coarse_direct_final_max_consecutive_frame_failures": (
            int(cfg.coarse_direct_final_max_consecutive_frame_failures)
            if direct_strategy else None
        ),
        "coarse_direct_final_group_size_policy": (
            "edge_first_max3_then_interior_max5_prefer3_to5_allow1_to2"
            if direct_strategy else None
        ),
        "coarse_direct_final_compactness_policy": (
            "same_aspect_ratio_xy_diameter_and_adjacency_gates_for_edge_and_interior"
            if direct_strategy else None
        ),
        "coarse_direct_final_edge_aspect_ratio_gate": (
            {
                "enabled": True,
                "max_aspect_ratio": float(cfg.batch_coarse_group_max_aspect_ratio),
            }
            if direct_strategy else None
        ),
        "coarse_direct_final_boundary_policy": (
            "selected_region_outer_layer_first_local_boundary_classification"
            if direct_strategy else None
        ),
        "coarse_direct_final_edge_first": bool(direct_strategy),
        "coarse_direct_final_edge_max_group_size": (
            min(3, int(cfg.batch_coarse_max_group_size))
            if direct_strategy else None
        ),
        "coarse_direct_final_capture_only": (
            bool(getattr(args, "coarse_direct_final_capture_only", False))
            if direct_strategy else False
        ),
        "per_hole_fine_fallback_enabled": bool(
            cfg.batch_fine_per_hole_fallback
            and not direct_strategy
        ),
        "coarse_direct_final": bool(getattr(args, "coarse_direct_final", False)),
        "fine_stage_policy": (
            "disabled_for_coarse_direct_pointcloud_only"
            if direct_strategy else
            "configured_shared_or_per_hole_fine"
        ),
        "fine_stage_skipped": None,
    }

    ctx = _workflow_stages.SequentialWorkflowContext(
        args=args, handeye=handeye, model=model, cfg=cfg, run_dir=run_dir,
        report=report, timing=timing, rows=rows, runtime=runtime,
        pose_session=pose_session, motion_session=motion_session,
        current_tcp=current_tcp, initial_holes=initial_holes,
        initial_intrinsics=initial_intrinsics, fixed_rz_rad=fixed_rz_rad,
        results=results, order_ids=order_ids,
        initial_pointcloud_reused_holes=initial_pointcloud_reused_holes,
        all_selected_two_capture_mode=all_selected_two_capture_mode,
        cache_entries=cache_entries, cache_sources=cache_sources,
        cache_source_ids=cache_source_ids, cache_gates=cache_gates,
        cache_enabled=cache_enabled, persistent_enabled=persistent_enabled,
        persistent_entries=persistent_entries, coarse_cache_dir=coarse_cache_dir,
        persistent_cache_dir=persistent_cache_dir, cache_built_ids=cache_built_ids,
        batch_coarse_results={},
        batch_coarse_for_cache=bool(
            direct_strategy
            or all_selected_two_capture_mode
            or cfg.batch_coarse_localization
            or (cache_enabled and not cache_entries)
        ),
        shared_cache_results={}, shared_cache_failed_ids=set(),
        invalidated_cache_ids=set(), batch_fine_results={}, batch_fine_plan={},
    )

    preview_result = _workflow_stages.run_preview_stage(ctx)
    if preview_result is not None:
        return int(preview_result)
    _workflow_stages.run_shared_coarse_stage(ctx)
    _workflow_stages.run_shared_cache_validation_stage(ctx)
    _workflow_stages.run_shared_fine_stage(ctx)
    _workflow_stages.optimize_hole_order_stage(ctx)
    _workflow_stages.run_per_hole_stage(ctx)
    return _finalize_sequential_results(
        ctx.results, ctx.order_ids, ctx.all_selected_two_capture_mode,
        ctx.fixed_rz_rad, ctx.batch_fine_results, ctx.handeye, ctx.cfg,
        ctx.current_tcp, ctx.runtime, ctx.report, ctx.run_dir, ctx.rows, ctx.timing,
    )
