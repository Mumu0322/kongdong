"""Sequential multi-hole orchestration.

The public runner remains the compatibility facade. It installs the facade's
runtime symbols here before dispatch so existing tests, GUI integrations, and
site-specific dependency patches continue to affect the workflow.
"""

from __future__ import annotations


_RUNTIME_DEPENDENCIES = {
    "CHARUCO_XY_MODEL_BIAS_MM",
    "CHARUCO_XY_MODEL_MATRIX",
    "CHARUCO_XY_MODEL_SOURCE",
    "COARSE_SURFACE_MODEL",
    "COARSE_SURFACE_SELECTION_POLICY",
    "CacheValidationGates",
    "DEFAULT_FINAL_TARGET_MODE",
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
    "apply_final_point_base_offsets",
    "base_z_target_for_camera_height",
    "camera_height_to_plane_mm",
    "camera_transform",
    "compose_batch_fine_xy_with_coarse_z",
    "correct_projected_circle_center",
    "final_point_offsets_for_mode",
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
        "mode": "grouped_shared_coarse_then_grouped_shared_fine_with_per_hole_fallback",
        "hole_count": len(initial_holes),
        "hole_order": order_ids,
        "fixed_rz_rad": fixed_rz_rad,
        "coarse_settle_buffer_s": float(cfg.coarse_settle_delay_s),
        "tracking_identity_source": "initial_selection_order_then_shared_group_live_refined_3d_projection",
        "confirmation_policy": "no_per_hole_pause_after_batch_fine_capture",
        "camera_pipeline_policy": "reuse_single_rgbd_pipeline_for_coarse_and_rgb_fine",
        "initial_pointcloud_reused_holes": initial_pointcloud_reused_holes,
        "capture_policy": (
            "spatially_group_all_selected_holes_at_each_stage_then_failed_holes"
            "_in_compact_shared_supplement_groups_then_per_hole_fine_fallback"
        ),
        "group_limits": {
            "coarse_max_holes": int(cfg.batch_coarse_max_group_size),
            "fine_max_holes": int(cfg.batch_fine_max_group_size),
            "coarse_max_aspect_ratio": float(cfg.batch_coarse_group_max_aspect_ratio),
            "fine_max_aspect_ratio": float(cfg.batch_fine_group_max_aspect_ratio),
            "coarse_max_xy_diameter_mm": float(
                cfg.batch_coarse_group_max_xy_diameter_mm
            ),
        },
        "per_hole_fine_fallback_enabled": bool(cfg.batch_fine_per_hole_fallback),
    }

    def ensure_rgbd_pipeline() -> tuple[Any, Any, Any]:
        if runtime.get("rgbd_pipeline") is None:
            with timing.measure("camera/restart_rgbd_pipeline"):
                pipeline, align, chain = init_pipeline()
            runtime["rgbd_pipeline"] = pipeline
            runtime["align"] = align
            runtime["chain"] = chain
        return runtime["rgbd_pipeline"], runtime["align"], runtime["chain"]

    # 正式多孔流程先按340 mm可靠视野自动分组，每组共享拍摄。初始RGB-D
    # 点云只负责规划这些共同粗定位位姿，不能代替正式粗定位拍摄。
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
        if len(initial_holes) > 1:
            # 预览模式也生成与正式执行完全相同的空间分组图；这里使用
            # 初始点云几何估算精定位分组，只用于检查组边界和排序，不下发运动。
            try:
                preview_coarse_grouping_result = _split_batch_localization_groups(
                    initial_holes, current_tcp, handeye, fixed_rz_rad,
                    initial_intrinsics, cfg.coarse_height_mm,
                    cfg.batch_coarse_view_margin_px,
                    cfg.batch_coarse_max_view_span_ratio,
                    max_group_size=cfg.batch_coarse_max_group_size,
                    max_aspect_ratio=cfg.batch_coarse_group_max_aspect_ratio,
                    adjacency_distance_factor=cfg.batch_coarse_group_adjacency_factor,
                    max_normal_spread_deg=cfg.batch_coarse_group_max_normal_spread_deg,
                    max_xy_diameter_mm=cfg.batch_coarse_group_max_xy_diameter_mm,
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
                    "group_count": len(preview_coarse_groups),
                    "grouping_policy": (
                        "minimum_feasible_exact_cover_compact_v2_spatial_groups"
                    ),
                    "grouping_search": preview_coarse_grouping_metadata,
                    "max_group_size": int(cfg.batch_coarse_max_group_size),
                    "max_view_span_ratio": float(cfg.batch_coarse_max_view_span_ratio),
                    "group_max_xy_diameter_mm": float(
                        cfg.batch_coarse_group_max_xy_diameter_mm
                    ),
                    "groups": [
                        {
                            "group_index": index,
                            "hole_ids": [int(hole["hole_id"]) for hole in group],
                            "hole_count": len(group),
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
                        "340mm SHARED COARSE GROUP PREVIEW",
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

    # 执行批量粗定位（如果启用且有多个孔）
    if batch_coarse_for_cache and len(initial_holes) > 1:
        print(
            f"[BATCH_COARSE] 启用批量粗定位模式: {len(initial_holes)}个孔",
            flush=True,
        )

        rgbd_pipeline, align, chain = ensure_rgbd_pipeline()

        grouping_result = _split_batch_localization_groups(
            initial_holes, current_tcp, handeye, fixed_rz_rad, initial_intrinsics,
            cfg.coarse_height_mm, cfg.batch_coarse_view_margin_px,
            cfg.batch_coarse_max_view_span_ratio,
            max_group_size=cfg.batch_coarse_max_group_size,
            max_aspect_ratio=cfg.batch_coarse_group_max_aspect_ratio,
            adjacency_distance_factor=cfg.batch_coarse_group_adjacency_factor,
            max_normal_spread_deg=cfg.batch_coarse_group_max_normal_spread_deg,
            max_xy_diameter_mm=cfg.batch_coarse_group_max_xy_diameter_mm,
            return_diagnostics=True,
            return_metadata=True,
        )
        batch_groups, batch_group_diagnostics, batch_grouping_metadata = (
            _unpack_batch_grouping_result(grouping_result)
        )
        batch_plan: dict[str, Any] = {
            "enabled": True,
            "hole_count": len(initial_holes),
            "hole_order": order_ids,
            "group_count": len(batch_groups),
            "all_selected_holes_single_group": len(batch_groups) == 1,
            "grouping_policy": (
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
            "grouping_order_policy": "home_image_v_then_u_row_major",
            "grouping_search": batch_grouping_metadata,
            "grouping_diagnostics": batch_group_diagnostics,
            "capture_frames": int(cfg.batch_coarse_frames),
            "early_stop_extra_frames": int(
                cfg.batch_coarse_early_stop_extra_frames
            ),
            "save_all_capture_overlays": bool(cfg.save_all_capture_overlays),
            "combined_position_policy": "projected_bbox_center_above_all_selected_holes",
            "motion_policy": "shared_vertical_lift_min10_safe_horizontal_descent_guard10_then_pure_descent10_then_steady_capture",
            "pose_refinement_policy": {
                "enabled": bool(cfg.batch_coarse_group_pose_refinement),
                "frames": int(cfg.batch_coarse_pose_refine_frames),
                "min_valid_frames": int(cfg.batch_coarse_pose_refine_min_valid_frames),
                "max_iterations": int(cfg.batch_coarse_pose_refine_max_iterations),
                "uses_live_group_geometry": True,
                "formal_capture_requires_its_own_quality_gate": True,
            },
            "view_margin_px": float(cfg.batch_coarse_view_margin_px),
            "groups": [],
        }
        batch_plan["grouping_visualization"] = _save_grouping_plan_visualization(
            run_dir / "01_home_selected.png",
            run_dir / "batch_coarse_grouping_plan",
            initial_holes,
            batch_groups,
            "340mm SHARED COARSE GROUP PLAN",
            batch_group_diagnostics,
            timing=timing,
        )
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
                "grouping_diagnostics": (
                    batch_group_diagnostics[group_index - 1]
                    if group_index - 1 < len(batch_group_diagnostics) else None
                ),
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
                        cfg,
                        initial_intrinsics,
                        run_dir,
                        timing,
                        rows,
                        args,
                        motion_session,
                        pose_session,
                        fixed_rz_rad,
                    )
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
                group_report["capture_grouping_visualization"] = (
                    _save_group_capture_visualization(
                        group_metadata.get("latest_overlay_path"),
                        run_dir / f"batch_coarse_group_{group_index:02d}_grouping_capture",
                        group_index,
                        group,
                        batch_geometry.get("projected_holes_px"),
                        batch_geometry.get("group_bbox_px"),
                        "340mm SHARED COARSE CAPTURE GROUP",
                        timing=timing,
                    )
                )
                if group_report["fallback_holes"]:
                    print(
                        f"[BATCH_COARSE] 第{group_index}组存在失败孔，其他共享组继续: "
                        f"{group_report['fallback_holes']}", flush=True,
                    )
            except Exception as exc:
                # 规划/移动/采集只影响当前视野组；其余组已经得到的结果和缓存
                # 继续保留，当前组的孔走后面的逐孔压轴兜底路径。
                group_report["error"] = f"{type(exc).__name__}:{exc}"
                group_report["fallback_reason"] = "group_failed"
                print(
                    f"[BATCH_COARSE] 第{group_index}组失败；"
                    + "该组孔延后，其他共享组继续"
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
        "failure_policy": (
            "move_to_shared_260mm_supplement_capture_then_per_hole_fine_fallback"
            if cfg.batch_fine_per_hole_fallback else
            "move_to_shared_260mm_supplement_capture_then_defer"
        ),
        "joint_failure_policy": (
            "move_to_per_hole_fine_capture_with_charuco_when_joint_gate_fails"
        ),
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
        "supplement_policy": "spatially_cluster_failed_holes_for_distinct_260mm_views",
        "supplement_max_view_span_ratio": float(
            cfg.batch_fine_supplement_max_view_span_ratio
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
                                additional_capture_frames=(
                                    cfg.batch_fine_inplace_recovery_frames
                                ),
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
                    group_report["capture_grouping_visualization"] = (
                        _save_group_capture_visualization(
                            group_metadata.get("latest_overlay_path"),
                            run_dir / f"batch_fine_group_{group_index:02d}_grouping_capture",
                            group_index,
                            group,
                            batch_geometry.get("projected_holes_px"),
                            batch_geometry.get("group_bbox_px"),
                            "260mm SHARED FINE CAPTURE GROUP",
                            timing=timing,
                        )
                    )
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
                        pending_group = [
                            fine_planning_holes_by_id[hole_id]
                            for hole_id in pending_holes
                            if hole_id in fine_planning_holes_by_id
                        ]
                        if not pending_group:
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
                                    "minimum_feasible_exact_cover_compact_v2_spatial_groups"
                                ),
                                "grouping_diagnostics": (
                                    supplement_group_diagnostics[cluster_index - 1]
                                    if cluster_index - 1 < len(supplement_group_diagnostics)
                                    else None
                                ),
                            }
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
                                with timing.measure(
                                    f"batch_fine/group_{group_index:02d}/{artifact_suffix}/navigate_to_260mm",
                                    hole_count=len(supplement_group),
                                    group_index=group_index,
                                    supplement_round=supplement_round,
                                    supplement_cluster=cluster_index,
                                ):
                                    current_tcp = _move_to_shared_fine_pose(
                                        current_tcp, supplement_target,
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
                                        "batch_fine_clustered_supplement_at_260mm"
                                    )
                                    result["batch_fine_capture_round"] = supplement_round
                                    result["batch_fine_supplement_cluster"] = cluster_index
                                    batch_fine_results[int(hole_id)] = result
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
                                        supplement_geometry.get("projected_holes_px"),
                                        supplement_geometry.get("group_bbox_px"),
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
                            except Exception as exc:
                                supplement_report["error"] = (
                                    f"{type(exc).__name__}:{exc}"
                                )
                                supplement_report["fallback_reason"] = (
                                    "batch_fine_clustered_supplement_failed"
                                )
                                for hole_id in cluster_ids:
                                    previous = dict(batch_fine_results.get(hole_id, {}))
                                    previous.update({
                                        "success": False,
                                        "error": supplement_report["error"],
                                        "batch_fine_source": (
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

        # 批量模式的正式RGB图像已经在进入逐孔结果计算前统一拍完；首拍失败
        # 的孔会在此前的共享精拍阶段尝试移动到补拍位。共享补拍仍失败时，
        # 默认进入逐孔精拍兜底；只有显式关闭该开关才保留旧的 deferred 行为。
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
            _write_progress_checkpoint(run_dir, report, timing=timing)
            print(
                f"[BATCH_FINE] hole={hole_id} status=deferred_batch_fine；"
                f"共享补拍失败且已关闭逐孔补拍：{failure_reason}",
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
                        current_tcp,
                        batch_fine_target,
                        args,
                        motion_session,
                        pose_session,
                        target_height_mm=cfg.fine_height_mm,
                        target_stage="批量粗定位后的单孔精定位兜底",
                        descent_guard_mm=(
                            SHARED_OBSERVATION_MIN_DESCENT_MM
                            if all_selected_two_capture_mode else 0.0
                        ),
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
                        str(hole_id), current_tcp, fine_target, args,
                        motion_session, pose_session, target_height_mm=cfg.fine_height_mm,
                        target_stage="共享缓存验证后的单孔精定位兜底",
                        descent_guard_mm=(
                            SHARED_OBSERVATION_MIN_DESCENT_MM
                            if all_selected_two_capture_mode else 0.0
                        ),
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
            if not batch_fine_available:
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
                        descent_guard_mm=(
                            SHARED_OBSERVATION_MIN_DESCENT_MM
                            if all_selected_two_capture_mode else 0.0
                        ),
                    )
            else:
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
                    timing=timing,
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
            _write_progress_checkpoint(run_dir, report, timing=timing)
            print(
                f"[SEQUENTIAL_HOLE] order={order} hole={hole_id} "
                f"status=deferred_coarse_quality；继续后续孔",
                flush=True,
            )
            if order < len(initial_holes):
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
            _write_progress_checkpoint(run_dir, report, timing=timing)
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
                    category="operator_wait",
                    level="leaf",
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
        final_motion_direct: dict[str, Any] | None = None
        if args.move_final_xy and batch_fine_available:
            assert coarse_fine_reference_tcp is not None
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
            )
            correction_xy = xy_target[:2, 3] - final_target_point_base[:2]
            z_target = plan_final_tcp_base_z(xy_target, final_target_point_base)
            # 固定的基坐标Y和工具系+Y补偿与最终Z移动正交，提前合并到
            # 高位最终TCP；因此不再在工件低位额外执行一次微调。
            direct_target = plan_final_tcp_combined_y_trim(z_target)
            direct_safe_z = max(
                float(motion_start_tcp[2, 3]), float(direct_target[2, 3]),
            ) + THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM
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
                if fixed_offset is None else f"显式固定补偿={list(fixed_offset)} mm"
            )
            final_xy_motion = {
                "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(direct_target),
                "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "tcp_position_before_mm": motion_start_tcp[:3, 3].copy(),
                "hole_center_base_mm": final_point_base,
                "pose_point_base_mm": pose_point_base,
                "target_point_base_mm": final_target_point_base,
                "batch_fine_joint_applied": batch_fine_joint_applied,
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
                "final_point_mode": final_target_mode,
                "final_point_offset_base_mm": [final_x_offset_mm, 0.0, final_z_offset_mm],
                "delta_base_z_mm": float(direct_target[2, 3] - reference_tcp[2, 3]),
                "motion_frame": "base_z_only_guard10_then_precision10",
                "descent_guard_mm": SHARED_OBSERVATION_MIN_DESCENT_MM,
            }
            final_y_trim_motion = {
                "planned_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(direct_target),
                "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                "delta_base_y_mm": FINAL_BASE_Y_AFTER_Z_MM,
                "delta_tool_y_mm": FINAL_TOOL_Y_AFTER_Z_MM,
                "motion_frame": "base_y_plus_tool_y_combined_at_safe_height",
                "applied_at_safe_height": True,
                "separate_motion_executed": False,
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
                "y_trim_applied_at_safe_height": True,
                "actual_final_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
            }
        if args.move_final_xy and not batch_fine_available:
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
                "batch_fine_joint_applied": batch_fine_joint_applied,
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
            "batch_fine_fallback_from_shared": bool(batch_fine_fallback_active),
            "batch_fine_fallback_reason": hole.get("batch_fine_fallback_reason"),
            "batch_fine_group_index": batch_fine_plan.get("hole_group_indices", {}).get(str(hole_id)),
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
                pose_point_base.copy() if batch_fine_joint_applied else None
            ),
            "batch_fine_joint_details": batch_fine_joint_details,
            "batch_fine_joint_summary": batch_fine_joint_summary,
            "batch_fine_direct_point_base_mm": batch_fine_result.get(
                "batch_fine_direct_point_base_mm"
            ),
            "coarse_source": coarse_source,
            "fine_center_source": fine.get("center_source", "unknown"),
            "fine_center_source_counts": fine.get("center_source_counts", {}),
            "fine_quality_status": fine.get("fine_quality_status", "strict"),
            "fine_quality_note": fine.get("fine_quality_note"),
            "fine_recovery_attempts": fine.get("fine_recovery_attempts", []),
            "hole_center_base_mm": final_point_base,
            "hole_result_type": "base_frame_3d_point",
            "final_pose_source": (
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
                category="operator_wait",
                level="leaf",
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
            "grouped_batch_fine_captures_then_compute_all_holes"
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
    timing.mark(
        "cycle/complete",
        cycle_index=int(report.get("cycle_index", 0)),
        completed_count=len(completed_results),
        deferred_count=len(deferred_results),
    )
    try:
        final_overlay = _save_batch_fine_final_result_overlay(
            run_dir, batch_fine_results, results, handeye, timing=timing,
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
    _write_report(run_dir, report, rows, timing=timing)
    if deferred_results:
        print(
            f"\n[顺序孔定位结果] 流程已继续完成：成功={len(completed_results)}，"
            f"deferred={len(deferred_results)}；deferred孔未执行最终目标点运动。",
            flush=True,
        )
    else:
        print("\n[顺序孔定位结果] 已按初始孔号逐个完成粗定位、精定位和目标点运动。")
    print(
        f"[DONE] 结果目录: {run_dir}；"
        f"成功={len(completed_results)}，延后/失败={len(deferred_results)}；"
        "详细结果请查看 report.json，简明结果请查看 result_summary.txt。",
        flush=True,
    )
    return 0
