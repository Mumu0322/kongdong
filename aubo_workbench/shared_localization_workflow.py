"""Shared coarse/fine capture workflows.

This module is invoked through the legacy runner facade. The facade installs its
runtime symbols before each call so existing dependency injection and field
patches keep working during the gradual modularization.
"""

from __future__ import annotations

from aubo_workbench.hole_localization_models import artifact_measure


_RUNTIME_DEPENDENCIES = {
    "COARSE_SURFACE_SELECTION_POLICY",
    "Observation",
    "_assign_detections_to_projection",
    "_coarse_expected_point_base",
    "_ellipse_ok",
    "_fine_burst_stable",
    "_fit_batch_fine_joint_transform",
    "_fit_batch_projected_anchor_correction",
    "_fuse_coarse",
    "_fuse_fine",
    "_observation_rows",
    "_plane_estimate_from_info",
    "_project_base_point_to_pixel",
    "_save_coarse_pointcloud_image",
    "_unit",
    "camera_transform",
    "cv2",
    "detect",
    "fit_hole_ellipse",
    "get_aligned_frame_bundle",
    "get_rgb_frame_bundle",
    "hole_camera_point",
    "np",
    "pixel_to_base_plane",
    "replace",
    "transform_to_sdk_pose_m_rad",
    "undistort_pixels",
}


def install_runtime(symbols: dict[str, object]) -> None:
    for name in _RUNTIME_DEPENDENCIES:
        if name in symbols:
            globals()[name] = symbols[name]


def _batch_coarse_capture_is_stable(
    selected_holes: list[dict[str, Any]],
    hole_observations: dict[int, list[Observation]],
    frame_records: list[dict[str, Any]],
    min_holes_per_frame: int,
    cfg: TwoStageConfig,
) -> bool:
    """判断当前共享粗拍是否已经满足正式融合门限。

    这里复用正式的 ``_fuse_coarse`` 质量门，而不是只按“读够了几帧”
    提前结束。默认要求整组孔在同一批有效帧中达到稳定帧数；因此提前
    结束不会改变最终融合规则，只会跳过已经没有信息增益的尾部帧。
    """
    required = max(1, int(cfg.batch_coarse_min_valid))
    valid_group_frame_indices = {
        int(record["frame_index"])
        for record in frame_records
        if bool(record.get("valid", False))
    }
    if len(valid_group_frame_indices) < required:
        return False

    require_same_group_frames = min_holes_per_frame >= len(selected_holes)
    for hole in selected_holes:
        hole_id = int(hole["hole_id"])
        observations = hole_observations.get(hole_id, [])
        if require_same_group_frames:
            observations = [
                item for item in observations
                if int(item.frame_index) in valid_group_frame_indices
            ]
        try:
            _fuse_coarse(
                observations,
                cfg,
                min_valid_frames=required,
                max_center_scatter_p95_px=cfg.max_coarse_center_scatter_p95_px,
                max_tracking_distance_p95_px=cfg.max_coarse_tracking_distance_p95_px,
            )
        except Exception:
            return False
    return True


def _batch_coarse_localization_at_340mm(
    selected_holes: list[dict[str, Any]],
    current_tcp: np.ndarray,
    handeye: Any,
    pipeline: Any,
    align: Any,
    chain: Any,
    model: Any,
    confidence: float,
    cfg: TwoStageConfig,
    intrinsics: Any,
    run_dir: Path,
    timing: TimingRecorder,
    rows: list[dict[str, Any]],
    *,
    artifact_prefix: str | None = None,
) -> dict[Any, dict[str, Any]]:
    """在340mm共同位姿一次性检测所有孔的位姿和深度。

    返回: {hole_id: {
        "center_base_mm": [x, y, z],
        "normal_base": [nx, ny, nz],
        "plane_point_base_mm": [x, y, z],
        "depth_mm": float,
        "valid_frames": int,
        "center_scatter_p95_px": float,
        "plane_rmse_mm": float,
        "coarse_captures": [...],
        "coarse_center_camera_mm": [x, y, z],
        "coarse_plane_point_camera_mm": [x, y, z],
        "coarse_normal_camera": [nx, ny, nz],
        其他字段...
    }}
    """
    if not selected_holes:
        raise RuntimeError("批量粗定位：没有选中孔")

    T_base_camera = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
    expected_anchors = {
        int(hole["hole_id"]): _project_base_point_to_pixel(
            _coarse_expected_point_base(hole),
            T_base_camera,
            intrinsics,
        )
        for hole in selected_holes
    }
    holes_by_id = {int(hole["hole_id"]): hole for hole in selected_holes}
    batch_frames = max(1, int(cfg.batch_coarse_frames))
    min_valid = int(cfg.batch_coarse_min_valid)
    early_stop_min_frames = min(
        batch_frames,
        max(min_valid, min_valid + max(
            0, int(getattr(cfg, "batch_coarse_early_stop_extra_frames", 0)),
        )),
    )
    min_holes_per_frame = cfg.batch_coarse_min_holes_per_frame
    if min_holes_per_frame is None:
        # 批量模式的目标是一次获得整组孔的几何；单孔缺失时交给外层逐孔流程回退。
        min_holes_per_frame = len(selected_holes)
    min_holes_per_frame = max(1, min(int(min_holes_per_frame), len(selected_holes)))
    settle_discard_frames = max(0, int(cfg.batch_coarse_settle_discard_frames))
    hole_observations: dict[int, list[Observation]] = {
        hole_id: [] for hole_id in holes_by_id
    }
    frame_records: list[dict[str, Any]] = []
    latest_overlay_path: Path | None = None
    latest_view: Any = None
    artifact_tag = "" if not artifact_prefix else f"{str(artifact_prefix).strip()}_"
    last_intrinsics = intrinsics
    anchor_correction: dict[str, np.ndarray] | None = None
    anchor_correction_info: dict[str, Any] | None = None
    anchor_correction_attempted = False
    capture_stop_reason = "max_frames_reached"

    # moveLine 已等待控制器的 steady 标志，但相机管线里仍可能有运动期间的
    # RGB-D 帧。先清空这段队列，避免把不同TCP位姿下的点云融合到同一个变换。
    discarded_frame_count = 0
    if settle_discard_frames > 0:
        with timing.measure(
            "batch_coarse/discard_settle_frames",
            target_frames=settle_discard_frames,
        ):
            for _ in range(settle_discard_frames):
                bundle = get_aligned_frame_bundle(pipeline, align, chain)
                if bundle is not None:
                    discarded_frame_count += 1

    with timing.measure(
        "batch_coarse/capture_all_holes",
        hole_count=len(selected_holes),
        target_frames=batch_frames,
    ):
        for frame_index in range(batch_frames):
            bundle = get_aligned_frame_bundle(pipeline, align, chain)
            if bundle is None or bundle.intrinsics is None:
                reason = "rgbd_frame_missing"
                for hole_id, anchor in expected_anchors.items():
                    hole_observations[hole_id].append(Observation(
                        "batch_coarse", frame_index, anchor,
                        error=reason,
                    ))
                frame_records.append({
                    "frame_index": frame_index,
                    "valid": False,
                    "valid_hole_count": 0,
                    "reason": reason,
                })
                continue

            last_intrinsics = bundle.intrinsics
            raw_anchors = {
                str(hole_id): _project_base_point_to_pixel(
                    _coarse_expected_point_base(hole),
                    T_base_camera,
                    bundle.intrinsics,
                )
                for hole_id, hole in holes_by_id.items()
            }
            try:
                detections = detect(model, bundle.color_bgr, confidence)
                if not anchor_correction_attempted:
                    anchor_correction_attempted = True
                    try:
                        anchor_correction, anchor_correction_info = (
                            _fit_batch_projected_anchor_correction(
                                detections,
                                raw_anchors,
                                cfg.multi_coarse_tracking_tolerance_px,
                                min_matches=max(4, min(6, len(selected_holes))),
                            )
                        )
                    except Exception as correction_exc:
                        # 校正只解决公共位姿的系统性投影偏差；校正本身不可靠时，
                        # 保留原始锚点并让后面的严格跟踪门和外层回退机制接管。
                        anchor_correction = None
                        anchor_correction_info = {
                            "enabled": False,
                            "error": f"{type(correction_exc).__name__}:{correction_exc}",
                        }
                current_anchors = (
                    anchor_correction
                    if anchor_correction is not None else raw_anchors
                )
                assignments = _assign_detections_to_projection(
                    detections,
                    current_anchors,
                    cfg.multi_coarse_tracking_tolerance_px,
                )
            except Exception as exc:
                reason = f"detection_failed:{type(exc).__name__}:{exc}"
                for hole_id, anchor in expected_anchors.items():
                    hole_observations[hole_id].append(Observation(
                        "batch_coarse", frame_index, anchor,
                        timestamp_ns=bundle.host_timestamp_ns,
                        error=reason,
                    ))
                frame_records.append({
                    "frame_index": frame_index,
                    "valid": False,
                    "valid_hole_count": 0,
                    "reason": reason,
                })
                continue

            view = bundle.color_bgr.copy()
            valid_hole_count = 0
            hole_records: list[dict[str, Any]] = []
            for hole_id, hole in holes_by_id.items():
                anchor = np.asarray(current_anchors[str(hole_id)], dtype=np.float64)
                assigned = assignments.get(str(hole_id))
                if assigned is None:
                    reason = "yolo_missing_or_far"
                    hole_observations[hole_id].append(Observation(
                        "batch_coarse", frame_index, anchor,
                        timestamp_ns=bundle.host_timestamp_ns,
                        error=reason,
                    ))
                    hole_records.append({"hole_id": hole_id, "valid": False, "reason": reason})
                    continue

                detection = assigned["detection"]
                center = np.asarray(detection["center"], dtype=np.float64).reshape(2)
                box = np.asarray(detection["box"], dtype=np.float64).reshape(4)
                radius = max(float(box[2] - box[0]), float(box[3] - box[1])) / 2.0
                try:
                    _, plane_info = hole_camera_point(
                        tuple(center.tolist()),
                        bundle.xyz_map_mm,
                        bundle.intrinsics,
                        radius,
                        ray_center_xy=center,
                        ray_center_is_undistorted=False,
                        include_points=True,
                        surface_selection_policy=COARSE_SURFACE_SELECTION_POLICY,
                    )
                    plane = _plane_estimate_from_info(
                        plane_info, f"batch coarse hole {hole_id} normal",
                    )
                    error = (
                        None if plane.rmse_mm <= float(cfg.max_plane_rmse_mm)
                        else f"plane_quality:{plane.rmse_mm:.3f}mm"
                    )
                    observation = Observation(
                        "batch_coarse", frame_index, center,
                        plane=plane,
                        timestamp_ns=bundle.host_timestamp_ns,
                        error=error,
                        tracking_distance_px=float(assigned["distance_px"]),
                    )
                    hole_observations[hole_id].append(observation)
                    valid = error is None
                    if valid:
                        valid_hole_count += 1
                    hole_records.append({
                        "hole_id": hole_id,
                        "valid": valid,
                        "distance_px": float(assigned["distance_px"]),
                        "plane_rmse_mm": float(plane.rmse_mm),
                        "ring_points": int(plane.ring_points),
                        "reason": error,
                    })
                except Exception as exc:
                    reason = f"pointcloud_failed:{type(exc).__name__}:{exc}"
                    hole_observations[hole_id].append(Observation(
                        "batch_coarse", frame_index, center,
                        timestamp_ns=bundle.host_timestamp_ns,
                        error=reason,
                        tracking_distance_px=float(assigned["distance_px"]),
                    ))
                    hole_records.append({
                        "hole_id": hole_id,
                        "valid": False,
                        "distance_px": float(assigned["distance_px"]),
                        "reason": reason,
                    })

            for hole_id, expected in current_anchors.items():
                point = tuple(np.rint(expected).astype(int))
                record = next(item for item in hole_records if str(item["hole_id"]) == hole_id)
                color = (0, 255, 0) if record["valid"] else (0, 165, 255)
                cv2.circle(view, point, 18, color, 2, cv2.LINE_AA)
                cv2.putText(
                    view, f"H{hole_id}", (point[0] + 10, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )
                assigned = assignments.get(hole_id)
                if assigned is not None:
                    detected = tuple(np.rint(assigned["detection"]["center"]).astype(int))
                    cv2.drawMarker(
                        view, detected, (255, 0, 255), cv2.MARKER_TILTED_CROSS,
                        12, 2, cv2.LINE_AA,
                    )
            cv2.putText(
                view,
                f"Batch RGB-D 340mm frame={frame_index} valid={valid_hole_count}/{len(selected_holes)}",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
            latest_view = view
            frame_overlay_path = None
            if bool(getattr(cfg, "save_all_capture_overlays", False)):
                frame_overlay_path = run_dir / (
                    f"{artifact_tag}batch_coarse_340_frame_{frame_index:02d}.png"
                )
                with artifact_measure(
                    timing,
                    "batch_coarse/write_frame_overlay",
                    artifact_kind="coarse_frame_overlay",
                    paths=[str(frame_overlay_path)],
                    frame_index=int(frame_index),
                ):
                    cv2.imwrite(str(frame_overlay_path), view)
                latest_overlay_path = frame_overlay_path
            frame_records.append({
                "frame_index": frame_index,
                "valid": valid_hole_count >= min_holes_per_frame,
                "valid_hole_count": valid_hole_count,
                "required_hole_count": min_holes_per_frame,
                "detection_count": len(detections),
                "holes": hole_records,
                "overlay_path": (
                    None if frame_overlay_path is None else str(frame_overlay_path)
                ),
            })

            # 先达到配置的最少有效帧，再额外保留少量确认帧；只有整组每个孔
            # 通过与最终融合相同的稳定性/跟踪门限时才提前结束。
            if frame_index + 1 >= early_stop_min_frames:
                if _batch_coarse_capture_is_stable(
                    selected_holes,
                    hole_observations,
                    frame_records,
                    min_holes_per_frame,
                    cfg,
                ):
                    capture_stop_reason = "all_holes_stable_early_stop"
                    break

    if latest_view is not None and not bool(
        getattr(cfg, "save_all_capture_overlays", False)
    ):
        latest_overlay_path = run_dir / (
            f"{artifact_tag}batch_coarse_340_last_frame.png"
        )
        with artifact_measure(
            timing,
            "batch_coarse/write_last_frame_overlay",
            artifact_kind="coarse_last_frame_overlay",
            paths=[str(latest_overlay_path)],
        ):
            cv2.imwrite(str(latest_overlay_path), latest_view)
        if frame_records:
            frame_records[-1]["overlay_path"] = str(latest_overlay_path)

    # 融合每个孔的多帧观测
    batch_results: dict[int, dict[str, Any]] = {}

    for hole in selected_holes:
        hole_id = int(hole["hole_id"])
        observations = hole_observations[hole_id]
        valid_group_frame_indices = {
            int(record["frame_index"])
            for record in frame_records
            if bool(record.get("valid", False))
        }
        # 默认要求整组孔同帧有效；否则不同孔会来自不同的运动/曝光阶段，
        # 虽然每个孔单独看似有足够帧，批量共同位姿却没有一致的数据基础。
        fusion_observations = observations
        if min_holes_per_frame >= len(selected_holes):
            fusion_observations = [
                item for item in observations
                if int(item.frame_index) in valid_group_frame_indices
            ]
        valid_observations = [
            item for item in fusion_observations
            if item.error is None and item.plane is not None
        ]
        error_counts: dict[str, int] = {}
        for item in observations:
            if item.error:
                key = str(item.error).split(":", 1)[0]
                error_counts[key] = error_counts.get(key, 0) + 1

        # 记录全部观测数据；其中未进入融合的帧也要保留，便于诊断运动/队列问题。
        rows.extend(_observation_rows(observations))

        if len(valid_observations) < min_valid:
            # 当前孔的有效帧数不足，标记为失败
            batch_results[hole_id] = {
                "success": False,
                "valid_frames": len(valid_observations),
                "total_frames": len(observations),
                "min_valid_frames": min_valid,
                "error": (
                    f"整组有效帧不足: {len(valid_observations)} < {min_valid}"
                    if min_holes_per_frame >= len(selected_holes)
                    else f"有效帧数不足: {len(valid_observations)} < {min_valid}"
                ),
                "error_counts": error_counts,
                "observations": observations,
            }
            continue

        # 融合几何信息
        try:
            summary = _fuse_coarse(
                fusion_observations,
                cfg,
                min_valid_frames=min_valid,
                max_center_scatter_p95_px=cfg.max_coarse_center_scatter_p95_px,
                max_tracking_distance_p95_px=cfg.max_coarse_tracking_distance_p95_px,
            )

            # 转换到base坐标系
            R_base_camera = T_base_camera[:3, :3]
            t_base_camera = T_base_camera[:3, 3]
            camera_origin = t_base_camera

            point_camera = np.asarray(summary["plane_point_camera_mm"], dtype=np.float64).reshape(3)
            point_base = R_base_camera @ point_camera + t_base_camera

            normal_camera = _unit(
                np.asarray(summary["plane_normal_camera"], dtype=np.float64),
                f"batch coarse hole {hole_id} camera normal",
            )
            normal_base = _unit(
                R_base_camera @ normal_camera,
                f"batch coarse hole {hole_id} base normal",
            )

            # 确保法向指向相机
            if float(normal_base @ (camera_origin - point_base)) < 0.0:
                normal_base = -normal_base
                normal_camera = -normal_camera

            local_plane_point_camera = summary.get("surface_plane_point_camera_mm")
            local_plane_point_base = (
                R_base_camera @ np.asarray(local_plane_point_camera, dtype=np.float64) + t_base_camera
                if local_plane_point_camera is not None else None
            )

            batch_results[hole_id] = {
                "success": True,
                "center_base_mm": point_base.tolist(),
                "center_camera_mm": point_camera.tolist(),
                "normal_base": normal_base.tolist(),
                "normal_camera": normal_camera.tolist(),
                "plane_point_base_mm": (
                    local_plane_point_base.tolist()
                    if local_plane_point_base is not None
                    else point_base.tolist()
                ),
                "plane_point_camera_mm": (
                    point_camera.tolist()
                    if local_plane_point_camera is None
                    else np.asarray(local_plane_point_camera, dtype=np.float64).tolist()
                ),
                "depth_mm": float(point_camera[2]),
                "valid_frames": summary["valid_frames"],
                "total_frames": summary["total_frames"],
                "center_scatter_p95_px": summary["center_scatter_p95_px"],
                "tracking_distance_p95_px": summary["tracking_distance_p95_px"],
                "plane_rmse_mm": summary["plane_rmse_median_mm"],
                "ring_points_median": summary.get("ring_points_median"),
                "center_px": summary["center_px"],
                "surface_model": summary["surface_model"],
                "surface_selection_policy": summary.get("surface_selection_policy"),
                "front_surface_z_mm": summary.get("front_surface_z_median_mm"),
                "ring_points_raw_median": summary.get("ring_points_raw_median"),
                "surface_points_selected_median": summary.get("surface_points_selected_median"),
                "sphere_center_camera_mm": summary.get("sphere_center_camera_mm"),
                "sphere_radius_mm": summary.get("sphere_radius_mm"),
                "coarse_captures": [{
                    "capture_index": 0,
                    "mode": "batch_coarse_at_340mm",
                    "tcp_pose_m_rad": transform_to_sdk_pose_m_rad(current_tcp),
                    "summary": summary,
                }],
                "observations": observations,
                "error_counts": error_counts,
            }

        except Exception as exc:
            batch_results[hole_id] = {
                "success": False,
                "valid_frames": len(valid_observations),
                "total_frames": len(observations),
                "error": f"几何融合失败: {type(exc).__name__}: {exc}",
                "error_counts": error_counts,
                "observations": observations,
            }

    artifact_warnings: list[str] = []
    successful = {
        hole_id: result for hole_id, result in batch_results.items()
        if result.get("success")
    }
    if successful:
        point_chunks: list[np.ndarray] = []
        hole_labels: list[np.ndarray] = []
        frame_labels: list[np.ndarray] = []
        for hole_id, result in successful.items():
            for item in result["observations"]:
                if (
                    item.error is not None or item.plane is None
                    or item.plane.points_camera_mm is None
                ):
                    continue
                points = np.asarray(item.plane.points_camera_mm, dtype=np.float32).reshape(-1, 3)
                point_chunks.append(points)
                hole_labels.append(np.full(len(points), hole_id, dtype=np.int32))
                frame_labels.append(np.full(len(points), item.frame_index, dtype=np.int32))
        if point_chunks:
            archive_path = run_dir / f"{artifact_tag}batch_coarse_340_all_holes_pointcloud.npz"
            try:
                with artifact_measure(
                    timing,
                    "batch_coarse/write_pointcloud_archive",
                    artifact_kind="coarse_pointcloud_npz",
                    paths=[str(archive_path)],
                    hole_count=len(successful),
                ):
                    np.savez_compressed(
                        archive_path,
                        points_camera_mm=np.concatenate(point_chunks, axis=0),
                        point_hole_ids=np.concatenate(hole_labels),
                        point_frame_indices=np.concatenate(frame_labels),
                        hole_ids=np.asarray(sorted(successful), dtype=np.int32),
                        T_base_camera=np.asarray(T_base_camera, dtype=np.float64),
                    )
            except Exception as exc:
                artifact_warnings.append(f"npz:{type(exc).__name__}:{exc}")
                archive_path = None
        else:
            archive_path = None

        for hole_id, result in successful.items():
            pointcloud_path = _save_coarse_pointcloud_image(
                {
                    "coarse_center_camera_mm": result["center_camera_mm"],
                    "coarse_plane_point_camera_mm": result["plane_point_camera_mm"],
                    "coarse_normal_camera": result["normal_camera"],
                    "coarse_plane_rmse_mm": result["plane_rmse_mm"],
                    "coarse_center_scatter_p95_px": result["center_scatter_p95_px"],
                },
                hole_id,
                run_dir,
                observations=result["observations"],
                cache_entry=None,
                T_base_camera=T_base_camera,
                intrinsics=last_intrinsics,
                rgb_path=latest_overlay_path,
                source_label="batch_coarse_340mm",
                timing=timing,
            )
            result["pointcloud_image_path"] = (
                None if pointcloud_path is None else str(pointcloud_path)
            )
            result["batch_pointcloud_archive_path"] = (
                None if archive_path is None else str(archive_path)
            )

    batch_results["_batch_metadata"] = {
        "frame_records": frame_records,
        "anchor_correction": anchor_correction_info,
        "required_holes_per_frame": min_holes_per_frame,
        "settle_discard_frames": settle_discard_frames,
        "discarded_frame_count": discarded_frame_count,
        "captured_frame_count": len(frame_records),
        "early_stop_min_frames": early_stop_min_frames,
        "capture_stop_reason": capture_stop_reason,
        "max_center_scatter_p95_px": float(cfg.max_coarse_center_scatter_p95_px),
        "max_tracking_distance_p95_px": float(cfg.max_coarse_tracking_distance_p95_px),
        "latest_overlay_path": None if latest_overlay_path is None else str(latest_overlay_path),
        "artifact_warnings": artifact_warnings,
    }
    return batch_results


def _flush_rgb_queue_until_fresh(
    pipeline: Any,
    minimum_discard_frames: int,
    *,
    maximum_extra_frames: int = 20,
    fresh_host_interval_ms: float = 10.0,
    required_fresh_intervals: int = 2,
) -> dict[str, Any]:
    """丢弃运动过程RGB帧，直到确认pipeline已经返回实时新帧。

    Orbbec队列中的旧帧会在几毫秒内连续返回；队列清空后，wait_for_frames
    必须等待下一个相机周期。RgbFrameBundle的host_timestamp_ns记录在
    wait_for_frames返回后，因此连续两个足够长的主机时间间隔可以作为
    “已追上实时流”的证据。测试替身或旧调用方没有设备帧元数据时，保持
    原有行为，只执行配置的最少丢帧数。
    """
    minimum = max(0, int(minimum_discard_frames))
    maximum = minimum + max(0, int(maximum_extra_frames))
    required = max(1, int(required_fresh_intervals))
    threshold_ns = max(0, int(float(fresh_host_interval_ms) * 1_000_000.0))
    if minimum == 0:
        return {
            "discarded_frame_count": 0,
            "fresh_frame_confirmed": False,
            "reason": "disabled",
            "records": [],
        }

    discarded = 0
    attempts = 0
    fresh_streak = 0
    previous_host_timestamp_ns: int | None = None
    metadata_available = False
    records: list[dict[str, Any]] = []
    max_attempts = max(maximum + 5, minimum)
    while discarded < maximum and attempts < max_attempts:
        attempts += 1
        bundle = get_rgb_frame_bundle(pipeline)
        if bundle is None:
            records.append({"attempt": attempts, "valid": False})
            continue
        discarded += 1
        has_frame_metadata = bool(
            hasattr(bundle, "color_frame_index")
            or hasattr(bundle, "color_timestamp_us")
        )
        metadata_available = metadata_available or has_frame_metadata
        host_timestamp_ns = int(bundle.host_timestamp_ns)
        host_interval_ms = None
        if previous_host_timestamp_ns is not None:
            interval_ns = host_timestamp_ns - previous_host_timestamp_ns
            host_interval_ms = float(interval_ns / 1_000_000.0)
            if has_frame_metadata and interval_ns >= threshold_ns:
                fresh_streak += 1
            elif has_frame_metadata:
                fresh_streak = 0
        previous_host_timestamp_ns = host_timestamp_ns
        records.append({
            "attempt": attempts,
            "discard_index": discarded,
            "valid": True,
            "host_timestamp_ns": host_timestamp_ns,
            "host_interval_ms": host_interval_ms,
            "color_timestamp_us": getattr(bundle, "color_timestamp_us", None),
            "color_frame_index": getattr(bundle, "color_frame_index", None),
            "fresh_interval_streak": fresh_streak,
        })
        if discarded < minimum:
            continue
        if not metadata_available:
            return {
                "discarded_frame_count": discarded,
                "attempt_count": attempts,
                "fresh_frame_confirmed": False,
                "reason": "minimum_reached_without_frame_metadata",
                "fresh_host_interval_ms": float(fresh_host_interval_ms),
                "required_fresh_intervals": required,
                "records": records,
            }
        if fresh_streak >= required:
            return {
                "discarded_frame_count": discarded,
                "attempt_count": attempts,
                "fresh_frame_confirmed": True,
                "reason": "minimum_and_fresh_intervals_reached",
                "fresh_host_interval_ms": float(fresh_host_interval_ms),
                "required_fresh_intervals": required,
                "records": records,
            }

    return {
        "discarded_frame_count": discarded,
        "attempt_count": attempts,
        "fresh_frame_confirmed": bool(metadata_available and fresh_streak >= required),
        "reason": "maximum_discard_reached_before_fresh_confirmation",
        "fresh_host_interval_ms": float(fresh_host_interval_ms),
        "required_fresh_intervals": required,
        "records": records,
    }


def _batch_fine_localization_at_260mm(
    selected_holes: list[dict[str, Any]],
    current_tcp: np.ndarray,
    handeye: Any,
    pipeline: Any,
    model: Any,
    confidence: float,
    cfg: TwoStageConfig,
    intrinsics: Any,
    run_dir: Path,
    timing: TimingRecorder,
    rows: list[dict[str, Any]],
    *,
    artifact_prefix: str | None = None,
    additional_capture_frames: int = 0,
) -> dict[Any, dict[str, Any]]:
    """在一个260mm共同位姿同时精定位当前视野内的全部选中孔。

    机器人在本函数调用前已经移动到共同260mm位姿。本函数只采集RGB帧，
    每帧运行一次YOLO并把检测框一对一分配给所有目标孔；椭圆质量门和
    融合/验收沿用单孔精定位逻辑。启用联合开关时，先分别对每个孔的
    多帧圆心做时间方向稳健融合，再用这些“每孔一个”的基坐标XY拟合
    一个组级变换：三孔及以上拟合平面刚体变换，两孔只拟合共同平移，
    避免用两孔噪声虚构旋转。移动到另一个共享位姿的补拍由外层工作流
    负责，本函数本身不移动机器人。
    """
    if not selected_holes:
        raise RuntimeError("批量精定位：没有选中孔")

    T_base_camera = camera_transform(current_tcp, handeye.T_tcp_rgb_camera)
    projected_holes = {
        str(int(hole["hole_id"])): _project_base_point_to_pixel(
            np.asarray(hole["initial_center_base_mm"], dtype=np.float64).reshape(3),
            T_base_camera, intrinsics,
        )
        for hole in selected_holes
    }
    expected_metadata = {
        str(int(hole["hole_id"])): {
            "class_id": int(hole["initial_detection"].get("class_id", -1)),
            "box": hole["initial_detection"].get("box"),
        }
        for hole in selected_holes
    }
    hole_ids = [int(hole["hole_id"]) for hole in selected_holes]
    observations_by_hole: dict[int, list[Observation]] = {
        hole_id: [] for hole_id in hole_ids
    }
    frame_records: list[dict[str, Any]] = []
    latest_bundle: Any = None
    latest_view: Any = None
    artifact_tag = "" if not artifact_prefix else f"{str(artifact_prefix).strip()}_"
    anchor_correction: dict[str, np.ndarray] | None = None
    anchor_correction_info: dict[str, Any] | None = None
    anchor_correction_attempted = False
    discarded_frame_count = 0
    settle_flush: dict[str, Any] = {
        "discarded_frame_count": 0,
        "fresh_frame_confirmed": False,
        "reason": "disabled",
        "records": [],
    }
    batch_cfg = replace(
        cfg,
        fine_frames=int(cfg.batch_fine_frames),
        min_fine_valid=int(cfg.batch_fine_min_valid),
        fine_stable_min_frames=int(cfg.batch_fine_stable_min_frames),
        fine_settle_discard_frames=int(cfg.batch_fine_settle_discard_frames),
    )
    settle_discard_frames = max(0, int(batch_cfg.fine_settle_discard_frames))
    if settle_discard_frames > 0:
        with timing.measure(
            "batch_fine/discard_settle_frames",
            target_frames=settle_discard_frames,
        ):
            settle_flush = _flush_rgb_queue_until_fresh(
                pipeline, settle_discard_frames,
            )
            discarded_frame_count = int(settle_flush["discarded_frame_count"])

    max_frames = max(1, int(batch_cfg.fine_frames))
    # 首拍仍以原有正式帧数为主；只有某些孔未通过质量门时，才利用
    # 当前位置追加少量帧。已达到稳定门的孔会被锁定，不会因为追加帧
    # 被重新融合。这样“只差一两帧”的情况不需要先移动机械臂。
    inplace_recovery_frames = max(0, int(additional_capture_frames))
    # batch_fine_frames定义同一共同位姿下短连拍的正式帧数上限；达到
    # 当前孔集合的有效帧数和稳定门即可提前结束。
    max_attempts = max_frames + inplace_recovery_frames
    stable_gate_frames = max(
        1, int(batch_cfg.fine_stable_min_frames), int(batch_cfg.min_fine_valid),
    )
    locked_holes: set[int] = set()
    joint_frame_transforms: list[dict[str, Any]] = []
    joint_frame_records: list[dict[str, Any]] = []
    source_xy_by_hole = {
        int(hole["hole_id"]): np.asarray(
            hole["initial_center_base_mm"], dtype=np.float64,
        ).reshape(3)[:2].copy()
        for hole in selected_holes
    }
    # 联合路径只在每个选中孔都带有可用粗定位法向时启用。旧调用方
    # 可能只提供中心和检测框，此时保持原共享精拍的采集/锁定节奏，
    # 联合结果自然回退，不让新功能改变旧模式行为。
    joint_geometry_ready = all(
        hole.get("coarse_normal_toward_camera_base") is not None
        or hole.get("initial_plane_normal_base") is not None
        for hole in selected_holes
    )
    joint_enabled = bool(
        batch_cfg.batch_fine_joint_localization
        and len(hole_ids) >= max(2, int(batch_cfg.batch_fine_joint_min_holes))
        and joint_geometry_ready
    )
    with timing.measure(
        "batch_fine/capture_all_holes",
        hole_count=len(selected_holes), target_frames=max_frames,
        inplace_recovery_frames=inplace_recovery_frames,
        max_attempts=max_attempts,
    ):
        for frame_index in range(max_attempts):
            if len(locked_holes) == len(hole_ids):
                break

            bundle = get_rgb_frame_bundle(pipeline)
            if bundle is None or bundle.intrinsics is None:
                for hole_id, anchor in projected_holes.items():
                    if (
                        int(hole_id) in locked_holes
                        and not joint_enabled
                    ):
                        continue
                    observations_by_hole[int(hole_id)].append(Observation(
                        "batch_fine", frame_index,
                        np.asarray(anchor, dtype=np.float64),
                        timestamp_ns=None, error="rgb_frame_missing",
                    ))
                frame_records.append({
                    "frame_index": frame_index, "valid": False,
                    "valid_hole_count": 0, "reason": "rgb_frame_missing",
                })
                continue

            latest_bundle = bundle
            detections = detect(model, bundle.color_bgr, confidence)
            raw_anchors = {
                hole_id: _project_base_point_to_pixel(
                    np.asarray(next(
                        hole["initial_center_base_mm"]
                        for hole in selected_holes
                        if str(int(hole["hole_id"])) == hole_id
                    ), dtype=np.float64).reshape(3),
                    camera_transform(current_tcp, handeye.T_tcp_rgb_camera),
                    bundle.intrinsics,
                )
                for hole_id in projected_holes
            }
            if not anchor_correction_attempted:
                anchor_correction_attempted = True
                try:
                    anchor_correction, anchor_correction_info = (
                        _fit_batch_projected_anchor_correction(
                            detections,
                            raw_anchors,
                            cfg.multi_coarse_tracking_tolerance_px,
                            min_matches=max(3, min(6, len(selected_holes))),
                        )
                    )
                except Exception as correction_exc:
                    anchor_correction = None
                    anchor_correction_info = {
                        "enabled": False,
                        "error": f"{type(correction_exc).__name__}:{correction_exc}",
                    }
            current_anchors = anchor_correction if anchor_correction is not None else raw_anchors
            assignments = _assign_detections_to_projection(
                detections,
                current_anchors,
                batch_cfg.fine_pointcloud_anchor_tolerance_px,
                expected_metadata=expected_metadata,
            )
            valid_hole_count = 0
            hole_records: list[dict[str, Any]] = []
            for hole in selected_holes:
                hole_id = int(hole["hole_id"])
                key = str(hole_id)
                # 新联合算法按孔分别做时间融合，不要求所有孔使用同一帧
                # 子集；孔一旦达到自己的稳定门，就不再把后续可能变差的
                # 帧混入该孔的最终统计。
                if hole_id in locked_holes:
                    hole_records.append({
                        "hole_id": hole_id, "valid": True, "locked": True,
                    })
                    continue
                anchor = np.asarray(current_anchors[key], dtype=np.float64)
                assigned = assignments.get(key)
                if assigned is None:
                    observations_by_hole[hole_id].append(Observation(
                        "batch_fine", frame_index, anchor,
                        timestamp_ns=bundle.host_timestamp_ns,
                        error="yolo_missing_or_far",
                    ))
                    hole_records.append({
                        "hole_id": hole_id, "valid": False,
                        "reason": "yolo_missing_or_far",
                    })
                    continue

                detection = assigned["detection"]
                detection_center = np.asarray(
                    detection["center"], dtype=np.float64,
                ).reshape(2)
                ellipse = fit_hole_ellipse(bundle.color_bgr, detection, bundle.intrinsics)
                strict_ok = _ellipse_ok(ellipse, cfg)
                if not strict_ok:
                    rejected_center = (
                        np.asarray(ellipse["center_px"], dtype=np.float64).reshape(2)
                        if ellipse is not None else
                        undistort_pixels(
                            bundle.intrinsics, detection_center.reshape(1, 2),
                            pixel_output=True,
                        )[0]
                    )
                    observations_by_hole[hole_id].append(Observation(
                        "batch_fine", frame_index, rejected_center, ellipse,
                        timestamp_ns=bundle.host_timestamp_ns,
                        error="ellipse_quality", center_source="rejected",
                        quality_note="strict_geometric_center_gate_failed_no_yolo_fallback",
                        tracking_distance_px=float(assigned["distance_px"]),
                    ))
                    hole_records.append({
                        "hole_id": hole_id, "valid": False,
                        "distance_px": float(assigned["distance_px"]),
                        "reason": "ellipse_quality",
                    })
                    continue

                center = np.asarray(ellipse["center_px"], dtype=np.float64).reshape(2)
                center_distorted = np.asarray(
                    ellipse.get("center_px_distorted", center), dtype=np.float64,
                ).reshape(2)
                center_source = str(ellipse.get("fit_method") or "ellipse")
                geometric_anchor_distance = float(np.linalg.norm(
                    center_distorted - anchor
                ))
                if geometric_anchor_distance > float(
                    batch_cfg.batch_fine_max_geometric_anchor_distance_px
                ):
                    reason = (
                        "geometric_anchor_distance:"
                        f"{geometric_anchor_distance:.3f}px>"
                        f"{float(batch_cfg.batch_fine_max_geometric_anchor_distance_px):.3f}px"
                    )
                    observations_by_hole[hole_id].append(Observation(
                        "batch_fine", frame_index, center, ellipse,
                        timestamp_ns=bundle.host_timestamp_ns,
                        error=reason, center_source=center_source,
                        quality_note="geometric_center_far_from_coarse_projection_anchor",
                        tracking_distance_px=float(assigned["distance_px"]),
                        geometric_anchor_distance_px=geometric_anchor_distance,
                    ))
                    hole_records.append({
                        "hole_id": hole_id, "valid": False,
                        "distance_px": float(assigned["distance_px"]),
                        "geometric_anchor_distance_px": geometric_anchor_distance,
                        "reason": reason,
                    })
                    continue
                observations_by_hole[hole_id].append(Observation(
                    "batch_fine", frame_index, center, ellipse,
                    timestamp_ns=bundle.host_timestamp_ns,
                    center_source=center_source,
                    quality_note="strict_geometric_center",
                    tracking_distance_px=float(assigned["distance_px"]),
                    geometric_anchor_distance_px=geometric_anchor_distance,
                ))
                valid_hole_count += 1
                hole_records.append({
                    "hole_id": hole_id, "valid": True,
                    "distance_px": float(assigned["distance_px"]),
                    "geometric_anchor_distance_px": geometric_anchor_distance,
                    "center_source": center_source,
                    "geometric_center_px": center.tolist(),
                    "geometric_center_px_distorted": center_distorted.tolist(),
                    "ellipse_residual_px": float(ellipse["residual_px"]),
                })

            for hole_id in hole_ids:
                if hole_id in locked_holes:
                    continue
                if _fine_burst_stable(
                    {hole_id: observations_by_hole[hole_id]},
                    stable_gate_frames,
                    float(batch_cfg.fine_stable_center_scatter_p95_px),
                ):
                    locked_holes.add(hole_id)

            joint_frame_transform: dict[str, Any] | None = None
            joint_frame_error: str | None = None
            joint_frame_record: dict[str, Any] | None = None
            if joint_enabled:
                frame_target_xy_by_hole: dict[int, np.ndarray] = {}
                frame_weights_by_hole: dict[int, float] = {}
                for hole in selected_holes:
                    hole_id = int(hole["hole_id"])
                    observation = next(
                        (
                            item for item in reversed(observations_by_hole[hole_id])
                            if item.frame_index == frame_index
                            and item.error is None
                            and item.ellipse is not None
                        ),
                        None,
                    )
                    if observation is None:
                        continue
                    plane_point_value = hole.get("coarse_plane_point_base_mm")
                    if plane_point_value is None:
                        plane_point_value = hole.get("initial_center_base_mm")
                    normal_value = hole.get("coarse_normal_toward_camera_base")
                    if normal_value is None:
                        normal_value = hole.get("initial_plane_normal_base")
                    if plane_point_value is None or normal_value is None:
                        continue
                    try:
                        target_point = pixel_to_base_plane(
                            observation.center_px,
                            bundle.intrinsics,
                            current_tcp,
                            handeye.T_tcp_rgb_camera,
                            np.asarray(plane_point_value, dtype=np.float64).reshape(3),
                            _unit(
                                np.asarray(normal_value, dtype=np.float64).reshape(3),
                                f"batch fine joint hole {hole_id} normal",
                            ),
                            center_is_undistorted=True,
                        )
                    except Exception:
                        continue
                    frame_target_xy_by_hole[hole_id] = np.asarray(
                        target_point, dtype=np.float64,
                    ).reshape(3)[:2]
                    ellipse = observation.ellipse or {}
                    coverage = float(ellipse.get("coverage_deg", 0.0))
                    residual = float(ellipse.get("residual_px", 10.0))
                    roundness = float(ellipse.get("roundness", 0.0))
                    frame_weights_by_hole[hole_id] = max(
                        1.0e-3,
                        max(0.0, min(360.0, coverage)) / 360.0
                        * max(0.1, min(1.0, roundness))
                        / max(0.25, residual),
                    )
                try:
                    joint_frame_transform = _fit_batch_fine_joint_transform(
                        hole_ids,
                        source_xy_by_hole,
                        frame_target_xy_by_hole,
                        frame_weights_by_hole,
                        min_holes=batch_cfg.batch_fine_joint_min_holes,
                        max_residual_mm=batch_cfg.batch_fine_joint_max_residual_mm,
                        max_translation_mm=batch_cfg.batch_fine_joint_max_translation_mm,
                        max_yaw_deg=batch_cfg.batch_fine_joint_max_yaw_deg,
                        estimate_rotation=len(frame_target_xy_by_hole) >= 3,
                    )
                    joint_frame_transform["frame_index"] = int(frame_index)
                    joint_frame_transforms.append(joint_frame_transform)
                except Exception as joint_exc:
                    joint_frame_error = f"{type(joint_exc).__name__}:{joint_exc}"
                joint_frame_record = {
                    "frame_index": int(frame_index),
                    "success": joint_frame_transform is not None,
                    "fit_mode": (
                        None if joint_frame_transform is None else
                        joint_frame_transform.get("fit_mode")
                    ),
                    "hole_count": len(frame_target_xy_by_hole),
                    "hole_ids": sorted(frame_target_xy_by_hole),
                    "inlier_hole_ids": (
                        [] if joint_frame_transform is None else
                        list(joint_frame_transform["inlier_hole_ids"])
                    ),
                    "translation_mm": (
                        None if joint_frame_transform is None else
                        np.asarray(
                            joint_frame_transform["translation_mm"],
                            dtype=np.float64,
                        ).tolist()
                    ),
                    "yaw_deg": (
                        None if joint_frame_transform is None else
                        float(joint_frame_transform["yaw_deg"])
                    ),
                    "residual_p95_mm": (
                        None if joint_frame_transform is None else
                        float(joint_frame_transform["residual_p95_mm"])
                    ),
                    "error": joint_frame_error,
                }
                joint_frame_records.append(joint_frame_record)

            geometric_centers = {
                str(record["hole_id"]): np.asarray(
                    record["geometric_center_px_distorted"], dtype=np.float64,
                )
                for record in hole_records
                if record.get("geometric_center_px_distorted") is not None
            }
            view = bundle.color_bgr.copy()
            for detection in detections:
                box = tuple(np.rint(np.asarray(detection["box"], dtype=np.float64)).astype(int))
                cv2.rectangle(view, (box[0], box[1]), (box[2], box[3]), (255, 180, 0), 1)
            for hole_id, anchor in current_anchors.items():
                point = tuple(np.rint(np.asarray(anchor, dtype=np.float64)).astype(int))
                assigned = assignments.get(hole_id)
                color = (0, 255, 0) if assigned is not None else (0, 165, 255)
                cv2.circle(view, point, 14, color, 2, cv2.LINE_AA)
                cv2.putText(
                    view, f"H{hole_id}", (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )
                if assigned is not None:
                    detected_point = tuple(np.rint(np.asarray(
                        assigned["detection"]["center"], dtype=np.float64,
                    )).astype(int))
                    cv2.drawMarker(
                        view, detected_point, (255, 0, 255),
                        cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA,
                    )
                geometric_center = geometric_centers.get(str(hole_id))
                if geometric_center is not None:
                    geometric_point = tuple(np.rint(geometric_center).astype(int))
                    cv2.drawMarker(
                        view, geometric_point, (0, 0, 255),
                        cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA,
                    )
            cv2.putText(
                view,
                f"Batch RGB 260mm frame={frame_index} valid={valid_hole_count}/{len(selected_holes)}",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA,
            )
            latest_view = view
            overlay_path = None
            if bool(getattr(cfg, "save_all_capture_overlays", False)):
                overlay_path = run_dir / (
                    f"{artifact_tag}batch_fine_260_frame_{frame_index:02d}.png"
                )
                with artifact_measure(
                    timing,
                    "batch_fine/write_frame_overlay",
                    artifact_kind="fine_frame_overlay",
                    paths=[str(overlay_path)],
                    frame_index=int(frame_index),
                ):
                    cv2.imwrite(str(overlay_path), view)
            frame_records.append({
                "frame_index": frame_index,
                "valid": valid_hole_count > 0,
                "valid_hole_count": valid_hole_count,
                "detection_count": len(detections),
                "holes": hole_records,
                "locked_holes": sorted(locked_holes),
                "overlay_path": None if overlay_path is None else str(overlay_path),
                "joint_fine": joint_frame_record,
            })

    if latest_view is not None and not bool(
        getattr(cfg, "save_all_capture_overlays", False)
    ):
        overlay_path = run_dir / f"{artifact_tag}batch_fine_260_last_frame.png"
        with artifact_measure(
            timing,
            "batch_fine/write_last_frame_overlay",
            artifact_kind="fine_last_frame_overlay",
            paths=[str(overlay_path)],
        ):
            cv2.imwrite(str(overlay_path), latest_view)
        if frame_records:
            frame_records[-1]["overlay_path"] = str(overlay_path)

    if latest_bundle is None:
        raise RuntimeError("批量精定位期间未获得RGB帧")

    holes_by_id = {
        int(hole["hole_id"]): hole for hole in selected_holes
    }
    expected_anchor_by_hole: dict[int, np.ndarray] = {}
    fused_summaries: dict[int, dict[str, Any]] = {}
    fused_errors: dict[int, str] = {}
    fused_target_xy_by_hole: dict[int, np.ndarray] = {}
    fused_target_point_by_hole: dict[int, np.ndarray] = {}
    fused_weights_by_hole: dict[int, float] = {}
    fused_quality_report: dict[str, dict[str, Any]] = {}

    # 先分别在每个孔自己的跨帧观测上做严格融合。这里故意不再把
    # “每一帧各孔反投影后拟合出的联合变换”作为最终数据源：不同孔的
    # 有效帧集合可能不同，逐帧联合变换会把某一孔的瞬时偏差放大成整组
    # 失败。联合变换只消费下面已经通过单孔质量门的融合圆心。
    for hole_id in hole_ids:
        observations = observations_by_hole[hole_id]
        expected_anchor_for_hole = np.asarray(
            (
                anchor_correction.get(
                    str(hole_id), projected_holes[str(hole_id)]
                )
                if anchor_correction is not None else
                projected_holes[str(hole_id)]
            ),
            dtype=np.float64,
        ).reshape(2)
        expected_anchor_by_hole[hole_id] = expected_anchor_for_hole.copy()
        try:
            summary = _fuse_fine(
                observations, batch_cfg,
                max_center_scatter_p95_px=batch_cfg.max_fine_center_scatter_p95_px,
                reject_multimodal=True,
            )
            fused_anchor_distance = float(np.linalg.norm(
                np.asarray(summary["center_px_distorted"], dtype=np.float64).reshape(2)
                - expected_anchor_for_hole
            ))
            summary.update({
                "expected_anchor_px": expected_anchor_for_hole.copy(),
                "geometric_anchor_distance_px": fused_anchor_distance,
                "geometric_anchor_distance_gate_px": float(
                    batch_cfg.batch_fine_max_geometric_anchor_distance_px
                ),
                "fine_quality_status": "strict",
                "fine_quality_note": None,
                "fine_recovery_attempts": [{
                    "attempt": 1,
                    "name": "batch_fine",
                    "capture_status": "completed",
                    "total_frames": len(observations),
                    "valid_frames": int(summary["valid_frames"]),
                    "mode": "batch_fine_at_260mm",
                }],
            })
            if fused_anchor_distance > float(
                batch_cfg.batch_fine_max_geometric_anchor_distance_px
            ):
                raise RuntimeError(
                    "共享精定位融合圆心偏离粗定位投影锚点："
                    f"{fused_anchor_distance:.3f}px > "
                    f"{float(batch_cfg.batch_fine_max_geometric_anchor_distance_px):.3f}px"
                )

            target_point = None
            target_error = None
            hole = holes_by_id[hole_id]
            plane_point_value = hole.get("coarse_plane_point_base_mm")
            if plane_point_value is None:
                plane_point_value = hole.get("initial_center_base_mm")
            normal_value = hole.get("coarse_normal_toward_camera_base")
            if normal_value is None:
                normal_value = hole.get("initial_plane_normal_base")
            if plane_point_value is not None and normal_value is not None:
                try:
                    target_point = pixel_to_base_plane(
                        summary["center_px"],
                        latest_bundle.intrinsics,
                        current_tcp,
                        handeye.T_tcp_rgb_camera,
                        np.asarray(plane_point_value, dtype=np.float64).reshape(3),
                        _unit(
                            np.asarray(normal_value, dtype=np.float64).reshape(3),
                            f"batch fine fused hole {hole_id} normal",
                        ),
                        center_is_undistorted=True,
                    )
                    target_point = np.asarray(
                        target_point, dtype=np.float64,
                    ).reshape(3)
                    if not np.isfinite(target_point).all():
                        raise ValueError("融合圆心反投影结果包含非有限坐标")
                    fused_target_point_by_hole[hole_id] = target_point.copy()
                    fused_target_xy_by_hole[hole_id] = target_point[:2].copy()
                except Exception as target_exc:
                    target_error = f"{type(target_exc).__name__}:{target_exc}"
            if target_error is not None:
                summary["fused_target_error"] = target_error

            scatter = max(
                0.05, float(summary.get("center_scatter_p95_px", 1.0))
            )
            residual = max(
                0.25, float(summary.get("ellipse_residual_median_px", 1.0))
            )
            roundness = max(
                0.1, min(1.0, float(summary.get("ellipse_roundness_median", 0.1)))
            )
            fused_weights_by_hole[hole_id] = roundness / (scatter * residual)
            fused_summaries[hole_id] = summary
            fused_quality_report[str(hole_id)] = {
                "success": True,
                "valid_frames": int(summary["valid_frames"]),
                "total_frames": int(summary["total_frames"]),
                "center_scatter_p95_px": float(summary["center_scatter_p95_px"]),
                "geometric_anchor_distance_px": fused_anchor_distance,
                "target_point_base_mm": (
                    None if target_point is None else target_point.copy()
                ),
                "target_error": target_error,
                "joint_weight": float(fused_weights_by_hole[hole_id]),
            }
        except Exception as fused_exc:
            fused_errors[hole_id] = (
                f"{type(fused_exc).__name__}:{fused_exc}"
            )
            fused_quality_report[str(hole_id)] = {
                "success": False,
                "valid_frames": 0,
                "total_frames": len(observations),
                "error": fused_errors[hole_id],
            }

    joint_summary: dict[str, Any] = {
        "enabled": bool(batch_cfg.batch_fine_joint_localization),
        "geometry_ready": joint_geometry_ready,
        "active": joint_enabled,
        "success": False,
        "method": "per_hole_temporal_robust_fusion_then_group_xy_transform",
        "frame_records": joint_frame_records,
        # 保留逐帧联合结果仅用于诊断兼容，不再用它作为最终验收门。
        "valid_frame_count": len(joint_frame_transforms),
        "diagnostic_frame_count": len(joint_frame_transforms),
        "per_hole_fusion": fused_quality_report,
        "joint_frame_stability_gate_applied": False,
        "joint_frame_stability_gate_reason": (
            "diagnostic_only_after_per_hole_temporal_fusion"
        ),
        "min_valid_frames": int(batch_cfg.batch_fine_joint_min_valid_frames),
        "min_holes": int(batch_cfg.batch_fine_joint_min_holes),
        "max_residual_mm": float(batch_cfg.batch_fine_joint_max_residual_mm),
        "max_translation_mm": float(batch_cfg.batch_fine_joint_max_translation_mm),
        "max_yaw_deg": float(batch_cfg.batch_fine_joint_max_yaw_deg),
        "stable_translation_mm": float(
            batch_cfg.batch_fine_joint_stable_translation_mm
        ),
        "stable_yaw_deg": float(batch_cfg.batch_fine_joint_stable_yaw_deg),
    }
    if joint_enabled:
        try:
            joint_input_hole_ids = [
                hole_id for hole_id in hole_ids
                if hole_id in source_xy_by_hole
                and hole_id in fused_target_xy_by_hole
            ]
            required_holes = max(2, int(batch_cfg.batch_fine_joint_min_holes))
            if len(joint_input_hole_ids) < required_holes:
                raise ValueError(
                    "共享精定位联合可用孔数不足："
                    f"{len(joint_input_hole_ids)}/{required_holes}"
                )
            estimate_rotation = len(joint_input_hole_ids) >= 3
            joint_summary.update(
                _fit_batch_fine_joint_transform(
                    joint_input_hole_ids,
                    source_xy_by_hole,
                    fused_target_xy_by_hole,
                    fused_weights_by_hole,
                    min_holes=required_holes,
                    max_residual_mm=batch_cfg.batch_fine_joint_max_residual_mm,
                    max_translation_mm=batch_cfg.batch_fine_joint_max_translation_mm,
                    max_yaw_deg=batch_cfg.batch_fine_joint_max_yaw_deg,
                    estimate_rotation=estimate_rotation,
                )
            )
            joint_summary["joint_input_hole_ids"] = joint_input_hole_ids
            joint_summary["accepted_hole_ids"] = list(
                joint_summary.get("inlier_hole_ids", [])
            )
            joint_summary["rotation_policy"] = (
                "planar_rigid_3_or_more_holes"
                if estimate_rotation else "translation_only_2_holes"
            )
            joint_summary["fused_frame_count"] = min(
                int(fused_summaries[hole_id]["valid_frames"])
                for hole_id in joint_input_hole_ids
            )
            # 当前联合结果来自每孔各自的稳健帧子集，不再伪装成同一组的
            # 公共帧窗口；保留字段是为了让旧报告读取器不崩溃。
            joint_summary["fused_frame_indices"] = []
            joint_summary["fused_frame_indices_policy"] = (
                "per_hole_temporal_fusion_no_common_frame_subset"
            )
            joint_summary["source_xy_by_hole"] = {
                hole_id: np.asarray(source_xy_by_hole[hole_id], dtype=np.float64)
                .reshape(2).copy()
                for hole_id in joint_input_hole_ids
            }
        except Exception as joint_exc:
            joint_summary["error"] = (
                f"{type(joint_exc).__name__}:{joint_exc}"
            )
    elif batch_cfg.batch_fine_joint_localization:
        joint_summary["error"] = (
            "shared_joint_geometry_unavailable_or_hole_count_below_minimum"
        )

    batch_results: dict[int, dict[str, Any]] = {}
    for hole_id in hole_ids:
        observations = observations_by_hole[hole_id]
        rows.extend(_observation_rows(observations))
        expected_anchor_for_hole = expected_anchor_by_hole[hole_id]
        summary: dict[str, Any] | None = None
        direct_point: np.ndarray | None = fused_target_point_by_hole.get(hole_id)
        joint_point: np.ndarray | None = None
        try:
            fused_error = fused_errors.get(hole_id)
            if fused_error is not None:
                raise RuntimeError(
                    "共享精定位该孔单孔跨帧融合未通过质量门：" + fused_error
                )
            summary = dict(fused_summaries[hole_id])
            if direct_point is not None:
                coarse_xy = np.asarray(
                    holes_by_id[hole_id]["initial_center_base_mm"],
                    dtype=np.float64,
                ).reshape(3)[:2]
                coarse_to_fine_xy_mm = float(np.linalg.norm(
                    direct_point[:2] - coarse_xy
                ))
                summary["coarse_to_fine_xy_mm"] = coarse_to_fine_xy_mm
                summary["coarse_to_fine_xy_gate_mm"] = float(
                    batch_cfg.batch_fine_fallback_max_coarse_to_fine_xy_mm
                )

            if joint_enabled:
                if not joint_summary.get("success", False):
                    detail = joint_summary.get(
                        "error", "共享联合变换未生成"
                    )
                    raise RuntimeError(
                        "共享联合精定位未通过质量门，转逐孔精定位："
                        f"{detail}"
                    )
                accepted_hole_ids = {
                    int(value) for value in joint_summary.get(
                        "inlier_hole_ids", []
                    )
                }
                if hole_id not in accepted_hole_ids:
                    residual = (
                        joint_summary.get("residual_mm_by_hole") or {}
                    ).get(hole_id)
                    if residual is None:
                        residual = (
                            joint_summary.get("residual_mm_by_hole") or {}
                        ).get(str(hole_id))
                    detail = (
                        "未进入组级联合内点"
                        if residual is None else
                        f"组级联合残差{float(residual):.3f}mm超出内点门"
                    )
                    raise RuntimeError(
                        "共享联合精定位该孔未通过质量门，转逐孔精定位："
                        + detail
                    )
                joint_predictions = joint_summary.get("predicted_xy_by_hole") or {}
                joint_xy = joint_predictions.get(
                    hole_id, joint_predictions.get(str(hole_id))
                )
                if joint_xy is None:
                    raise RuntimeError(
                        "共享联合精定位缺少该孔预测XY，转逐孔精定位"
                    )
                coarse_point = np.asarray(
                    holes_by_id[hole_id].get(
                        "initial_center_base_mm", [0.0, 0.0, 0.0]
                    ),
                    dtype=np.float64,
                ).reshape(3)
                joint_point = np.asarray(
                    [
                        float(np.asarray(joint_xy).reshape(2)[0]),
                        float(np.asarray(joint_xy).reshape(2)[1]),
                        float(coarse_point[2]),
                    ],
                    dtype=np.float64,
                )

            batch_results[hole_id] = {
                "success": True,
                "fine": summary,
                "observations": observations,
                "intrinsics": latest_bundle.intrinsics,
                "capture_tcp": np.asarray(current_tcp, dtype=np.float64).copy(),
                "expected_anchor_px": np.asarray(
                    expected_anchor_for_hole, dtype=np.float64,
                ).copy(),
                "fine_capture_overlay_path": frame_records[-1].get("overlay_path")
                if frame_records else None,
                "batch_fine_joint_success": bool(
                    joint_point is not None and joint_summary.get("success", False)
                ),
                "batch_fine_joint_xy_base_mm": (
                    None if joint_point is None else joint_point[:2].copy()
                ),
                "batch_fine_joint_predicted_point_base_mm": joint_point,
                "batch_fine_direct_point_base_mm": (
                    None if direct_point is None else direct_point.copy()
                ),
                "batch_fine_joint_summary": joint_summary,
            }
        except Exception as exc:
            batch_results[hole_id] = {
                "success": False,
                "fine": summary,
                "observations": observations,
                "intrinsics": latest_bundle.intrinsics,
                "capture_tcp": np.asarray(current_tcp, dtype=np.float64).copy(),
                "expected_anchor_px": np.asarray(
                    expected_anchor_for_hole, dtype=np.float64,
                ).copy(),
                "error": f"{type(exc).__name__}:{exc}",
                "batch_fine_joint_success": False,
                "batch_fine_joint_predicted_point_base_mm": joint_point,
                "batch_fine_direct_point_base_mm": (
                    None if direct_point is None else direct_point.copy()
                ),
                "batch_fine_joint_summary": joint_summary,
            }

    batch_results["_batch_metadata"] = {
        "frame_records": frame_records,
        "anchor_correction": anchor_correction_info,
        "settle_discard_frames": settle_discard_frames,
        "discarded_frame_count": discarded_frame_count,
        "settle_flush": settle_flush,
        "max_frames": max_frames,
        "max_attempts": max_attempts,
        "inplace_recovery_frames": inplace_recovery_frames,
        "inplace_recovery_policy": (
            "continue_at_current_260mm_pose_for_unlocked_holes_before_moving"
            if inplace_recovery_frames > 0 else "disabled"
        ),
        "locked_holes": sorted(locked_holes),
        "joint_fine": joint_summary,
        "max_tracking_distance_px": float(batch_cfg.fine_pointcloud_anchor_tolerance_px),
        "max_geometric_anchor_distance_px": float(
            batch_cfg.batch_fine_max_geometric_anchor_distance_px
        ),
        "fallback_max_coarse_to_fine_xy_mm": float(
            batch_cfg.batch_fine_fallback_max_coarse_to_fine_xy_mm
        ),
        "latest_overlay_path": (
            frame_records[-1].get("overlay_path") if frame_records else None
        ),
    }
    return batch_results
