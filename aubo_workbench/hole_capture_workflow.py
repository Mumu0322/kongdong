"""Initial selection and stable coarse/fine frame capture workflows."""

from __future__ import annotations
from aubo_workbench.coarse_surface import save_surface_diagnostic

from contextlib import nullcontext
import json

from aubo_workbench.localization_errors import LocalizationHardwareError
from aubo_workbench.auto_sector_selection import (
    AutoSectorConfig,
    AutoSectorSelectionResult,
    row_major_layout,
    render_auto_sector_overlay,
    select_auto_holes,
    validate_selection_image,
)
from aubo_workbench.hole_localization_models import artifact_measure, visual_measure
from aubo_workbench.io_utils import atomic_write_json, jsonable
from aubo_workbench.paths import HOLE_LOCALIZATION_SECTOR_INFO_DIR
from aubo_workbench.sector_info import write_sector_info_snapshots


def _capture_initial_multi_hole_selection(
    pipeline: Any, align: Any, chain: Any, model: Any, confidence: float,
    run_dir: Path, max_plane_rmse_mm: float, count: int | None = None,
    *, timing: Any | None = None,
    auto_sector_config: AutoSectorConfig | None = None,
    map_hole_selection_mode: str | None = None,
) -> tuple[Any, list[dict[str, Any]], np.ndarray, PlaneEstimate, Any]:
    """初始画面选择任意数量的孔，并允许单孔深度异常降级到共享平面导航。

    ``auto_sector_config`` 只控制初始画面的候选筛选与扇区审计。初始帧的
    点云几何仍然只作为安全导航种子，340 mm 粗定位和后续精定位流程不变，
    这里不会写入最终孔位地图。
    """
    required = None if count is None else max(1, int(count))
    bundle = None
    for attempt in range(10):
        with visual_measure(
            timing,
            "initial_selection/frame_acquisition",
            "frame_acquisition",
            attempt=int(attempt + 1),
        ):
            bundle = get_aligned_frame_bundle(pipeline, align, chain)
        if bundle is not None and bundle.intrinsics is not None:
            break
    if bundle is None or bundle.intrinsics is None:
        raise RuntimeError("无法获得带RGB内参的初始多孔 RGB-D 帧")
    with visual_measure(
        timing,
        "initial_selection/yolo_inference",
        "yolo_inference",
        frame_count=1,
    ):
        detections = detect(model, bundle.color_bgr, confidence)
    if required is not None and len(detections) < required:
        raise RuntimeError(f"初始画面仅检测到 {len(detections)} 个孔，无法选择 {required} 个")
    auto_selection: AutoSectorSelectionResult | None = None
    map_selection_mode = (
        None
        if map_hole_selection_mode is None
        else str(map_hole_selection_mode).strip().lower()
    )
    if map_selection_mode not in {None, "auto", "manual"}:
        raise ValueError("map_hole_selection_mode必须是auto、manual或未指定")
    if map_selection_mode == "auto" and auto_sector_config is None:
        raise ValueError("建图auto选孔模式必须提供自动扇区配置")
    if auto_sector_config is not None:
        validate_selection_image(bundle.color_bgr, auto_sector_config)
        with visual_measure(
            timing,
            "initial_selection/auto_sector_selection",
            "visual_pipeline_other",
            candidate_count=len(detections),
            sector_count=int(auto_sector_config.sector_count),
        ):
            auto_selection = select_auto_holes(detections, auto_sector_config)
        selected_indices = list(auto_selection.selected_indices)
        auto_candidate_indices = list(selected_indices)
        # 建图时允许在自动扇区候选的范围内改为人工选孔。候选之外的
        # 检测不会传给选择器，避免误点到区域外的孔并写入当前扇区。
        if map_selection_mode == "manual":
            eligible_indices = list(selected_indices)
            if eligible_indices:
                eligible_detections = [
                    dict(detections[index], _original_detection_index=int(index))
                    for index in eligible_indices
                ]
                wait_context = (
                    timing.measure(
                        "operator/map_manual_selection_wait",
                        category="operator_wait",
                        level="leaf",
                        candidate_count=len(eligible_detections),
                    )
                    if timing is not None else nullcontext()
                )
                with wait_context:
                    local_selection = choose_boxes(
                        bundle.color_bgr,
                        eligible_detections,
                        count=required,
                        return_clicks=True,
                    )
                if local_selection is None:
                    raise TwoStageSelectionCancelled("用户取消建图人工选孔")
                local_indices, local_clicks = local_selection
                selected_indices = [eligible_indices[int(index)] for index in local_indices]
                selection_clicks = {
                    eligible_indices[int(index)]: list(map(float, click))
                    for index, click in local_clicks.items()
                    if 0 <= int(index) < len(eligible_indices)
                }
                # 后续扇区快照和运行报告应反映操作者最终确认的候选子集，
                # 同时保留自动筛选的原始索引供审计。
                selected_set = {int(index) for index in selected_indices}
                auto_selection.selected_indices = [int(index) for index in selected_indices]
                auto_candidate_set = set(auto_candidate_indices)
                for candidate in auto_selection.candidates:
                    if candidate.detection_index in selected_set:
                        candidate.selected = True
                        candidate.selection_reason = "operator_selected"
                    elif candidate.detection_index in auto_candidate_set:
                        candidate.selected = False
                        candidate.selection_reason = "operator_deselected"
            else:
                selection_clicks = {}
        else:
            # 自动模式没有人工点击，但下游仍需要一个稳定的像素锚点字典。
            selection_clicks = {
                int(index): list(map(float, detections[index]["center"]))
                for index in selected_indices
            }
        # 无论筛选结果是否为空，先落盘叠加图和JSON，便于定位ROI、边界
        # 和置信度配置问题；不能在这里提前抛错导致现场没有诊断证据。
        auto_report_path = run_dir / "auto_sector_selection.json"
        auto_overlay_path = run_dir / "01_home_auto_sector_selection.png"
        auto_overlay = render_auto_sector_overlay(
            bundle.color_bgr, detections, auto_selection,
        )
        with artifact_measure(
            timing,
            "initial_selection/write_auto_sector_overlay",
            artifact_kind="auto_sector_selection_overlay",
            paths=[str(auto_overlay_path)],
        ):
            if not cv2.imwrite(
                str(auto_overlay_path),
                auto_overlay,
            ):
                raise RuntimeError(f"自动分区叠加图写入失败：{auto_overlay_path}")
        with artifact_measure(
            timing,
            "initial_selection/write_auto_sector_report",
            artifact_kind="auto_sector_selection_report",
            paths=[str(auto_report_path)],
        ):
            auto_report = auto_selection.to_dict()
            auto_report.update({
                "map_hole_selection_mode": map_selection_mode,
                "auto_selected_detection_indices": [int(v) for v in auto_candidate_indices],
                "operator_selected_detection_indices": [int(v) for v in selected_indices]
                if map_selection_mode == "manual" else None,
            })
            atomic_write_json(auto_report_path, jsonable(auto_report))
        with artifact_measure(
            timing,
            "initial_selection/write_sector_info",
            artifact_kind="sector_info_snapshot",
            paths=[str(HOLE_LOCALIZATION_SECTOR_INFO_DIR)],
        ):
            write_sector_info_snapshots(
                auto_selection,
                HOLE_LOCALIZATION_SECTOR_INFO_DIR,
                source_run_dir=run_dir,
                overlay=auto_overlay,
            )
        if required is not None and len(selected_indices) != required:
            raise RuntimeError(
                f"自动分区筛选出 {len(selected_indices)} 个可执行孔，"
                f"与要求的 {required} 个不一致；边界候选需先经粗定位或补拍确认；"
                "请调整ROI/置信度/扇区配置"
            )
        if not selected_indices:
            if auto_sector_config.partition_mode == "polygons":
                raise RuntimeError(
                    "所画区域内没有可执行孔；请查看自动选孔叠加图，检查区域范围、活动扇区和检测置信度"
                )
            raise RuntimeError(
                "自动分区没有筛选出可执行孔；边界候选不能直接执行，"
                "请检查 origin_px、ROI、活动扇区和YOLO置信度，或先完成340 mm确认"
            )
        selection = (selected_indices, selection_clicks)
    elif timing is None:
        selection = choose_boxes(
            bundle.color_bgr, detections, count=required, return_clicks=True,
        )
    else:
        # 把人工点击/按 Enter 的等待单独记录，避免它被误认为视觉算法耗时。
        with timing.measure(
            "operator/initial_selection_wait",
            category="operator_wait",
            level="leaf",
        ):
            selection = choose_boxes(
                bundle.color_bgr, detections, count=required, return_clicks=True,
            )
    if selection is None:
        raise TwoStageSelectionCancelled("用户取消初始选孔")
    selected_indices, selection_clicks = selection
    if not selected_indices:
        raise RuntimeError("人工选孔未选择任何孔，无法建立地图")
    # 地图中的孔号采用图像坐标的行优先顺序。普通两阶段定位保留原有
    # 执行/点击顺序，避免改变已有缓存和单次定位行为。
    map_numbering_enabled = map_selection_mode in {"auto", "manual"}
    layout_by_detection_index: dict[int, dict[str, Any]] = {}
    original_selection_order = [int(index) for index in selected_indices]
    if map_numbering_enabled:
        layout = row_major_layout([
            {
                "detection_index": int(index),
                "center": detections[index]["center"],
                "box": detections[index].get("box"),
            }
            for index in selected_indices
        ])
        selected_indices = [
            int(entry["item"]["detection_index"]) for entry in layout
        ]
        layout_by_detection_index = {
            int(entry["item"]["detection_index"]): entry for entry in layout
        }
        if auto_selection is not None and map_selection_mode == "manual":
            # 人工点击顺序只用于审计；自动分区报告的正式 selected 索引与
            # 地图孔号保持同一行优先顺序，避免报告和 Hxx 对不上。
            auto_selection.selected_indices = [int(index) for index in selected_indices]
            auto_report_path = run_dir / "auto_sector_selection.json"
            if auto_report_path.is_file():
                try:
                    auto_report = json.loads(auto_report_path.read_text(encoding="utf-8"))
                    refreshed = auto_selection.to_dict()
                    for key in (
                        "selected_count", "selected_detection_indices", "selected_global_hole_keys",
                        "candidates", "coverage", "warnings",
                    ):
                        auto_report[key] = refreshed[key]
                    auto_report["map_ordered_detection_indices"] = [int(index) for index in selected_indices]
                    atomic_write_json(auto_report_path, jsonable(auto_report))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    # 初始报告已经写入；排序审计补写失败不应阻断几何采集。
                    pass
    selected = [detections[index] for index in selected_indices]
    auto_candidates = {
        int(item.detection_index): item
        for item in auto_selection.candidates
    } if auto_selection is not None else {}
    class_ids = {int(item.get("class_id", -1)) for item in selected}
    if len(class_ids) != 1:
        raise RuntimeError("初始选择的目标不是同一个YOLO孔类别，无法进行稳定身份关联")

    holes: list[dict[str, Any]] = []
    points_camera: list[np.ndarray] = []
    normals_camera: list[np.ndarray] = []
    overlays: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    geometries: list[dict[str, Any] | None] = []
    geometry_errors: list[str | None] = []

    for hole_id, (detection_index, detection) in enumerate(
        zip(selected_indices, selected), start=1,
    ):
        selection_click = np.asarray(
            selection_clicks[detection_index], dtype=np.float64,
        ).reshape(2)
        radius = max(
            float(detection["box"][2] - detection["box"][0]),
            float(detection["box"][3] - detection["box"][1]),
        ) / 2.0
        # 初始选孔只把点击用于确定目标身份；几何中心统一采用YOLO框中心。
        yolo_center = np.asarray(detection["center"], dtype=np.float64).reshape(2)
        display_detection = dict(detection)
        display_detection["_hole_id"] = hole_id
        overlays.append((display_detection, None))
        try:
            with visual_measure(
                timing,
                f"initial_selection/pointcloud_geometry_{hole_id:02d}",
                "pointcloud_processing",
                hole_id=int(hole_id),
            ):
                point, info = hole_camera_point(
                    tuple(yolo_center.tolist()),
                    bundle.xyz_map_mm,
                    bundle.intrinsics,
                    radius,
                    ray_center_xy=yolo_center,
                    ray_center_is_undistorted=False,
                    surface_selection_policy=COARSE_SURFACE_SELECTION_POLICY,
                )
            plane = _plane_estimate_from_info(
                info, f"initial multi-hole {hole_id} plane normal",
            )
            if plane.rmse_mm > float(max_plane_rmse_mm):
                raise ValueError(
                    f"深度拟合RMSE过大：{plane.rmse_mm:.3f} mm "
                    f"> {float(max_plane_rmse_mm):.3f} mm"
                )
            geometries.append({
                "point": np.asarray(point, dtype=np.float64),
                "plane": plane,
                "fallback": False,
            })
            geometry_errors.append(None)
            points_camera.append(np.asarray(point, dtype=np.float64))
            normals_camera.append(np.asarray(plane.normal_camera, dtype=np.float64))
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            geometries.append(None)
            geometry_errors.append(reason)
            print(
                f"[INITIAL_GEOMETRY_FALLBACK] hole={hole_id} reason={reason}",
                flush=True,
            )

    if not points_camera:
        reasons = "; ".join(
            f"hole_{index + 1}:{reason}"
            for index, reason in enumerate(geometry_errors)
            if reason
        )
        raise RuntimeError(f"初始选孔没有任何可用局部平面：{reasons}")

    # 共享平面只用于异常孔的安全340 mm导航和缓存匹配兜底；正常孔仍使用自己的局部平面。
    group_center_camera = np.mean(np.asarray(points_camera, dtype=np.float64), axis=0)
    group_normal_camera = _fuse_normals(normals_camera)
    successful_geometries = [
        geometry for geometry in geometries if geometry is not None
    ]
    group_rmse_mm = float(np.max([
        float(geometry["plane"].rmse_mm) for geometry in successful_geometries
    ]))
    group_ring_points = int(sum(
        int(geometry["plane"].ring_points) for geometry in successful_geometries
    ))
    group_plane = PlaneEstimate(
        point_camera_mm=group_center_camera,
        normal_camera=group_normal_camera,
        rmse_mm=group_rmse_mm,
        ring_points=group_ring_points,
        surface_model="initial_multi_hole_group",
    )

    for hole_id, (detection_index, detection), geometry, geometry_error in zip(
        range(1, len(selected) + 1),
        zip(selected_indices, selected),
        geometries,
        geometry_errors,
    ):
        selection_click = np.asarray(
            selection_clicks[detection_index], dtype=np.float64,
        ).reshape(2)
        yolo_center = np.asarray(detection["center"], dtype=np.float64).reshape(2)
        fallback = geometry is None
        if fallback:
            # 该孔的当前环带深度不可信时，只用共享平面求一个保守导航点。
            # 真正进入340 mm后仍优先直接调用缓存；无缓存则走完整粗定位。
            fallback_point = ray_plane_intersection(
                np.zeros(3, dtype=np.float64),
                camera_ray(bundle.intrinsics, yolo_center, already_undistorted=False),
                group_center_camera,
                group_normal_camera,
            )
            point = fallback_point
            plane = PlaneEstimate(
                point_camera_mm=fallback_point,
                normal_camera=group_normal_camera,
                rmse_mm=group_rmse_mm,
                ring_points=group_ring_points,
                surface_model="initial_group_plane_fallback",
                surface_plane_point_camera_mm=group_center_camera,
                surface_selection_policy=COARSE_SURFACE_SELECTION_POLICY,
            )
        else:
            point = np.asarray(geometry["point"], dtype=np.float64)
            plane = geometry["plane"]

        hole_payload = {
            "hole_id": hole_id,
            "initial_selection_order": hole_id,
            "operator_selection_order": int(
                original_selection_order.index(int(detection_index)) + 1
            ),
            "selection_source": (
                "auto"
                if map_numbering_enabled and auto_selection is not None
                and map_selection_mode != "manual" else
                "manual"
                if map_numbering_enabled and map_selection_mode == "manual"
                else "auto_sector_selection"
                if auto_selection is not None
                else "manual_selection"
            ),
            "class_id": int(detection["class_id"]),
            "initial_detection": dict(detection),
            "initial_center_px": list(map(float, detection["center"])),
            "initial_box": list(map(float, detection["box"])),
            "initial_selection_click_px": selection_click.tolist(),
            "tracking_anchor_px": list(map(float, detection["center"])),
            "tracking_roi_size_px": [
                float(detection["box"][2] - detection["box"][0]),
                float(detection["box"][3] - detection["box"][1]),
            ],
            "initial_point_camera_mm": np.asarray(point, dtype=np.float64).tolist(),
            "initial_plane_point_camera_mm": plane.point_camera_mm.tolist(),
            "initial_plane_normal_camera": plane.normal_camera.tolist(),
            "initial_plane_rmse_mm": float(plane.rmse_mm),
            "initial_ring_points": int(plane.ring_points),
            "initial_surface_model": plane.surface_model,
            "initial_surface_selection_policy": plane.surface_selection_policy,
            "initial_front_surface_z_mm": plane.front_surface_z_mm,
            "initial_ring_points_raw": plane.ring_points_raw,
            "initial_surface_points_selected": plane.surface_points_selected,
            "initial_geometry_fallback": bool(fallback),
            "initial_geometry_fallback_reason": geometry_error,
            "pointcloud_segmentation": f"yolo_box_{COARSE_SURFACE_SELECTION_POLICY}",
        }
        if map_numbering_enabled:
            layout_entry = layout_by_detection_index.get(int(detection_index))
            if layout_entry is not None:
                hole_payload.update({
                    "numbering_policy": "image_row_major_v1",
                    "map_order": int(hole_id),
                    "layout_row": int(layout_entry["row"]),
                    "layout_column": int(layout_entry["column"]),
                    "numbering_center_px": list(layout_entry["center_px"]),
                    "numbering_row_tolerance_px": float(layout_entry["row_tolerance_px"]),
                })
        if auto_selection is not None:
            auto_candidate = auto_candidates.get(int(detection_index))
            if auto_candidate is not None:
                hole_payload.update({
                    "selection_mode": "auto_sector_selection",
                    "sector_id": int(auto_candidate.sector_id)
                    if auto_candidate.sector_id is not None else None,
                    "sector_angle_deg": float(auto_candidate.angle_deg)
                    if auto_candidate.angle_deg is not None else None,
                    "sector_assignment_status": auto_candidate.assignment_status,
                    "global_hole_key": auto_candidate.global_hole_key,
                    "map_hole_key": (
                        f"S{int(auto_candidate.sector_id):02d}-H{int(hole_id):02d}"
                        if map_numbering_enabled and auto_candidate.sector_id is not None
                        else None
                    ),
                    "hole_identity_status": auto_candidate.identity_status,
                    "auto_selection_reason": auto_candidate.selection_reason,
                })
        holes.append(hole_payload)

    output_path = run_dir / "01_home_selected.png"
    with artifact_measure(
        timing,
        "initial_selection/write_overlay",
        artifact_kind="initial_selection_overlay",
        paths=[str(output_path)],
    ):
        if not cv2.imwrite(
            str(output_path),
            _overlay_multi(bundle.color_bgr, overlays, "initial multi-hole group selection"),
        ):
            raise RuntimeError(f"初始选孔图像写入失败：{output_path}")
    return bundle, holes, group_center_camera, group_plane, bundle.intrinsics


def _center_scatter_p95(observations: list[Observation]) -> float:
    """返回一组有效观测中心相对中位数的P95散布。"""
    centers = np.asarray([
        item.center_px for item in observations if item.error is None
    ], dtype=np.float64)
    if len(centers) < 2:
        return math.inf
    median = np.median(centers, axis=0)
    return float(np.percentile(np.linalg.norm(centers - median, axis=1), 95.0))


def _coarse_burst_stable(
    observations_by_hole: dict[int, list[Observation]],
    min_valid: int,
    max_scatter_p95_px: float,
) -> bool:
    """判断粗定位是否已达到可以提前结束的稳定状态。"""
    for observations in observations_by_hole.values():
        valid = [item for item in observations if item.error is None and item.plane is not None]
        if len(valid) < int(min_valid):
            return False
        if _center_scatter_p95(valid) > float(max_scatter_p95_px):
            return False
    return True


def _fine_burst_stable(
    observations_by_hole: dict[int, list[Observation]],
    min_valid: int,
    max_scatter_p95_px: float,
) -> bool:
    """判断RGB精定位多孔观测是否已足够稳定，可以停止继续补帧。"""
    for observations in observations_by_hole.values():
        valid = [item for item in observations if item.error is None and item.ellipse is not None]
        if len(valid) < int(min_valid):
            return False
        if _center_scatter_p95(valid) > float(max_scatter_p95_px):
            return False
    return True


def _capture_coarse_burst(pipeline: Any, align: Any, chain: Any, model: Any, confidence: float,
                          chosen: dict[str, Any], cfg: TwoStageConfig, run_dir: Path,
                          name: str,
                          initial_anchor_px: np.ndarray | None = None,
                          tracking_tolerance_px: float | None = None,
                          lock_anchor: bool = False,
                          stop_when_stable: bool = True,
                          include_points: bool = False,
                          timing: Any | None = None,
                          ) -> tuple[list[Observation], np.ndarray]:
    observations: list[Observation] = []
    latest_image: np.ndarray | None = None
    latest_detection: dict[str, Any] | None = None
    anchor = None if initial_anchor_px is None else np.asarray(initial_anchor_px, dtype=np.float64).reshape(2)
    locked_anchor = anchor.copy() if anchor is not None and lock_anchor else None
    with visual_measure(
        timing,
        f"{name}/frame_acquisition_settle",
        "frame_acquisition",
        target_frames=int(cfg.coarse_settle_frames),
    ):
        for _ in range(cfg.coarse_settle_frames):
            get_aligned_frame_bundle(pipeline, align, chain)
    for attempt in range(cfg.coarse_frames * cfg.coarse_max_attempt_multiplier):
        valid_count = len([item for item in observations if item.error is None and item.plane is not None])
        if valid_count >= cfg.coarse_frames or (
            stop_when_stable and (
                valid_count >= cfg.min_coarse_valid
                and _coarse_burst_stable(
                    {1: observations}, cfg.min_coarse_valid,
                    cfg.max_coarse_center_scatter_p95_px,
                )
            )
        ):
            break
        with visual_measure(
            timing,
            f"{name}/frame_acquisition",
            "frame_acquisition",
            attempt=int(attempt),
        ):
            bundle = get_aligned_frame_bundle(pipeline, align, chain)
        if bundle is None or bundle.intrinsics is None:
            continue
        latest_image = bundle.color_bgr
        if anchor is None:
            anchor = np.array([bundle.color_bgr.shape[1] / 2.0, bundle.color_bgr.shape[0] / 2.0])
            if lock_anchor:
                locked_anchor = anchor.copy()
        search_anchor = locked_anchor if lock_anchor and locked_anchor is not None else anchor
        with visual_measure(
            timing,
            f"{name}/yolo_inference",
            "yolo_inference",
            attempt=int(attempt),
        ):
            detection = _nearest_detection(
                detect(model, bundle.color_bgr, confidence), search_anchor, chosen["class_id"],
            )
        if detection is None:
            observations.append(Observation(
                name, attempt, search_anchor, timestamp_ns=bundle.host_timestamp_ns, error="yolo_missing",
            ))
            continue
        detection_center = np.asarray(detection["center"], dtype=np.float64)
        tracking_distance = float(np.linalg.norm(detection_center - search_anchor))
        if tracking_tolerance_px is not None and tracking_distance > float(tracking_tolerance_px):
            observations.append(Observation(
                name, attempt, search_anchor, timestamp_ns=bundle.host_timestamp_ns,
                error="tracking_distance", tracking_distance_px=tracking_distance,
            ))
            continue
        # 粗定位只使用YOLO框中心和点云环带，测量孔中心及局部法向。
        # 椭圆拟合只用于旧诊断叠加图，不参与粗定位结果，因此不在这里计算。
        center = detection_center
        radius = max(detection["box"][2] - detection["box"][0], detection["box"][3] - detection["box"][1]) / 2.0
        try:
            ring_center = tuple(detection["center"])
            with visual_measure(
                timing,
                f"{name}/pointcloud_geometry",
                "pointcloud_processing",
                attempt=int(attempt),
            ):
                _, plane_info = hole_camera_point(
                    ring_center, bundle.xyz_map_mm, bundle.intrinsics, radius,
                    ray_center_xy=center, ray_center_is_undistorted=False,
                    include_points=include_points,
                    surface_selection_policy=COARSE_SURFACE_SELECTION_POLICY,
                )
                plane = _plane_estimate_from_info(plane_info, "coarse plane normal")
            valid = plane.rmse_mm <= cfg.max_plane_rmse_mm
            observations.append(Observation(
                name, attempt, center, None, plane, bundle.host_timestamp_ns,
                None if valid else "plane_quality", tracking_distance_px=tracking_distance,
            ))
        except Exception as exc:
            observations.append(Observation(name, attempt, center, None, timestamp_ns=bundle.host_timestamp_ns,
                                            error=f"plane_error:{exc}",
                                            tracking_distance_px=tracking_distance))
        if not lock_anchor:
            anchor = detection_center
        latest_detection = detection
    if latest_image is None:
        raise RuntimeError("粗定位期间未获得相机帧")
    diagnostic = next((item for item in reversed(observations) if item.plane is not None), None)
    if diagnostic is not None:
        diagnostic_path = run_dir / f"{name}_surface_diagnostic.npz"
        with artifact_measure(timing, "per_hole_coarse/write_surface_diagnostic",
                              artifact_kind="surface_diagnostic", paths=[str(diagnostic_path)]):
            save_surface_diagnostic(diagnostic_path, diagnostic.plane,
                                    frame_index=diagnostic.frame_index)
    output_path = run_dir / f"{name}_overlay.png"
    with artifact_measure(
        timing,
        "per_hole_coarse/write_overlay",
        artifact_kind="per_hole_coarse_overlay",
        paths=[str(output_path)],
        capture_name=name,
    ):
        if not cv2.imwrite(
            str(output_path), _overlay(latest_image, latest_detection, None, name),
        ):
            raise RuntimeError(f"粗定位叠加图写入失败：{output_path}")
    return observations, latest_image


def _capture_fine_burst(
    pipeline: Any, model: Any, confidence: float, chosen: dict[str, Any],
    cfg: TwoStageConfig, run_dir: Path,
    initial_anchor_px: np.ndarray | None = None,
    tracking_tolerance_px: float | None = None,
    lock_anchor: bool = False,
    name: str = "04_fine",
    stable_scatter_p95_px: float | None = None,
    allow_yolo_fallback: bool = False,
    fallback_min_coverage_deg: float | None = None,
    fallback_max_residual_px: float | None = None,
    settle_discard_frames: int = 0,
    timing: Any | None = None,
) -> tuple[list[Observation], Any, np.ndarray, dict[str, Any] | None]:
    observations: list[Observation] = []
    latest_bundle = None
    latest_detection = latest_ellipse = None
    latest_center_source = "none"
    anchor = None if initial_anchor_px is None else np.asarray(initial_anchor_px, dtype=np.float64).reshape(2)
    locked_anchor = anchor.copy() if anchor is not None and lock_anchor else None
    fallback_coverage_gate = (
        cfg.fine_yolo_fallback_min_ellipse_coverage_deg
        if fallback_min_coverage_deg is None else float(fallback_min_coverage_deg)
    )
    fallback_residual_gate = (
        cfg.fine_yolo_fallback_max_ellipse_residual_px
        if fallback_max_residual_px is None else float(fallback_max_residual_px)
    )
    with visual_measure(
        timing,
        f"{name}/frame_acquisition_settle",
        "frame_acquisition",
        target_frames=max(0, int(settle_discard_frames)),
    ):
        for _ in range(max(0, int(settle_discard_frames))):
            # 只取RGB帧而不运行YOLO；这些帧用于跨越机器人到位后的相机/末端稳定期。
            get_rgb_frame_bundle(pipeline)
    for attempt in range(cfg.fine_frames * 3):
        valid_count = len([item for item in observations if item.error is None and item.ellipse is not None])
        if (
            valid_count >= cfg.fine_frames
            or (
                valid_count >= cfg.fine_stable_min_frames
                and _fine_burst_stable(
                    {1: observations}, cfg.fine_stable_min_frames,
                    cfg.fine_stable_center_scatter_p95_px
                    if stable_scatter_p95_px is None else float(stable_scatter_p95_px),
                )
            )
        ):
            break
        with visual_measure(
            timing,
            f"{name}/frame_acquisition",
            "frame_acquisition",
            attempt=int(attempt),
        ):
            bundle = get_rgb_frame_bundle(pipeline)
        if bundle is None or bundle.intrinsics is None:
            continue
        latest_bundle = bundle
        if anchor is None:
            anchor = np.array([bundle.color_bgr.shape[1] / 2.0, bundle.color_bgr.shape[0] / 2.0])
            if lock_anchor:
                locked_anchor = anchor.copy()
        search_anchor = locked_anchor if lock_anchor and locked_anchor is not None else anchor
        with visual_measure(
            timing,
            f"{name}/yolo_inference",
            "yolo_inference",
            attempt=int(attempt),
        ):
            detection = _nearest_detection(
                detect(model, bundle.color_bgr, confidence), search_anchor, chosen["class_id"],
            )
        if detection is None:
            observations.append(Observation(
                name, attempt, search_anchor, timestamp_ns=bundle.host_timestamp_ns, error="yolo_missing",
            ))
            continue
        detection_center = np.asarray(detection["center"], dtype=np.float64)
        tracking_distance = float(np.linalg.norm(detection_center - search_anchor))
        if tracking_tolerance_px is not None and tracking_distance > float(tracking_tolerance_px):
            observations.append(Observation(
                name, attempt, search_anchor, timestamp_ns=bundle.host_timestamp_ns,
                error="tracking_distance",
            ))
            continue
        with visual_measure(
            timing,
            f"{name}/ellipse_fitting",
            "ellipse_fitting",
            attempt=int(attempt),
        ):
            ellipse = fit_hole_ellipse(bundle.color_bgr, detection, bundle.intrinsics)
        # 即使严格椭圆门失败，也保存当前检测用于最后的诊断叠加图；
        # 否则“检测到了但拟合没通过”会被误看成“YOLO没有找到孔”。
        latest_detection, latest_ellipse = detection, ellipse
        strict_ellipse_ok = _ellipse_ok(ellipse, cfg)
        relaxed_ellipse_ok = _ellipse_ok(
            ellipse, cfg,
            min_coverage_deg=fallback_coverage_gate,
            max_residual_px=fallback_residual_gate,
        )
        if not strict_ellipse_ok and not (
            allow_yolo_fallback and relaxed_ellipse_ok
        ):
            observations.append(Observation(name, attempt, detection_center, ellipse,
                                            timestamp_ns=bundle.host_timestamp_ns,
                                            error="ellipse_quality",
                                            center_source="rejected",
                                            quality_note=(
                                                "strict_ellipse_gate_failed;"
                                                f"fallback_gate={fallback_residual_gate:.3f}px/"
                                                f"{fallback_coverage_gate:.1f}deg"
                                            )))
            latest_center_source = "rejected_ellipse_quality"
            continue

        # 精定位中心统一来自 YOLO 检测框；椭圆拟合只负责质量门和诊断，
        # 不再把 ellipse["center_px"] 混入中心融合，避免两种中心在帧间跳变。
        # YOLO 中心属于原始畸变像素域；下游 pixel_to_base_plane 要求去畸变域，
        # 因此统一先用当前 RGB 内参转换到去畸变像素坐标。
        with visual_measure(
            timing,
            f"{name}/coordinate_transform",
            "coordinate_transform",
            attempt=int(attempt),
        ):
            center = undistort_pixels(
                bundle.intrinsics, detection_center.reshape(1, 2), pixel_output=True,
            )[0]
        center_source = "yolo"
        quality_note = (
            None if strict_ellipse_ok
            else "strict_ellipse_rejected;accepted_by_pointcloud_anchor_and_relaxed_ellipse"
        )
        observations.append(Observation(
            name, attempt, center, ellipse,
            timestamp_ns=bundle.host_timestamp_ns,
            center_source=center_source,
            quality_note=quality_note,
        ))
        latest_center_source = center_source
        if not lock_anchor:
            # detection["center"] 属于原始像素域；不能用去畸变椭圆中心更新跟踪锚点。
            anchor = detection_center
    if latest_bundle is None:
        raise RuntimeError("精定位期间未获得 RGB 帧")
    reference_anchor = locked_anchor if lock_anchor and locked_anchor is not None else anchor
    output_path = run_dir / f"{name}_overlay.png"
    with artifact_measure(
        timing,
        "per_hole_fine/write_overlay",
        artifact_kind="per_hole_fine_overlay",
        paths=[str(output_path)],
        capture_name=name,
    ):
        if not cv2.imwrite(
            str(output_path),
            _overlay(
                latest_bundle.color_bgr, latest_detection, latest_ellipse,
                f"{name} RGB center={latest_center_source}",
                reference_center_px=reference_anchor,
            ),
        ):
            raise RuntimeError(f"精定位叠加图写入失败：{output_path}")
    return observations, latest_bundle.intrinsics, latest_bundle.color_bgr, latest_ellipse


def _capture_fine_with_recovery(
    pipeline: Any, model: Any, confidence: float, chosen: dict[str, Any],
    cfg: TwoStageConfig, run_dir: Path, hole: dict[str, Any], hole_id: int,
    processing_order: int, expected_anchor_px: np.ndarray,
    timing: TimingRecorder, rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """单孔精定位恢复流程：预热丢帧 -> 严格重拍 -> 稳定降级。"""
    attempts: list[dict[str, Any]] = []
    last_observations: list[Observation] = []
    last_intrinsics: Any = None
    retry_count = max(0, int(cfg.fine_retry_count))

    for attempt_index in range(retry_count + 1):
        attempt_number = attempt_index + 1
        fine_name = (
            f"hole_{hole_id:02d}_fine"
            if attempt_number == 1
            else f"hole_{hole_id:02d}_fine_retry_{attempt_number - 1}"
        )
        attempt_record: dict[str, Any] = {
            "attempt": attempt_number,
            "name": fine_name,
            "settle_discard_frames": int(cfg.fine_settle_discard_frames),
        }
        try:
            with timing.measure(
                f"hole_{hole_id:02d}/fine_capture_attempt_{attempt_number}",
                hole_id=hole_id,
                processing_order=processing_order,
                attempt=attempt_number,
            ):
                observations, fine_intrinsics, _, _ = _capture_fine_burst(
                    pipeline, model, confidence, chosen, cfg, run_dir,
                    initial_anchor_px=expected_anchor_px,
                    tracking_tolerance_px=cfg.fine_pointcloud_anchor_tolerance_px,
                    lock_anchor=True,
                    name=fine_name,
                    stable_scatter_p95_px=cfg.max_fine_center_scatter_p95_px,
                    allow_yolo_fallback=True,
                    fallback_min_coverage_deg=cfg.fine_yolo_fallback_min_ellipse_coverage_deg,
                    fallback_max_residual_px=cfg.fine_yolo_fallback_max_ellipse_residual_px,
                    settle_discard_frames=cfg.fine_settle_discard_frames,
                    timing=timing,
                )
            rows.extend(_observation_rows(observations))
            _record_hole_tracking_event(
                hole, fine_name, expected_anchor_px, observations,
                fine_intrinsics, "undistorted_pixel_yolo_center",
            )
            last_observations = observations
            last_intrinsics = fine_intrinsics
            attempt_record.update({
                "capture_status": "completed",
                "total_frames": len(observations),
                "valid_frames": len([
                    item for item in observations
                    if item.error is None and item.ellipse is not None
                ]),
            })
        except LocalizationHardwareError:
            raise
        except Exception as exc:
            attempt_record.update({
                "capture_status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            attempts.append(attempt_record)
            if attempt_index < retry_count:
                print(
                    f"[FINE_RETRY] 孔{hole_id}第{attempt_number}次采集失败，"
                    f"准备重拍：{attempt_record['error']}",
                    flush=True,
                )
                continue
            break

        try:
            with timing.measure(
                f"hole_{hole_id:02d}/fine_fusion_attempt_{attempt_number}",
                hole_id=hole_id,
                processing_order=processing_order,
                attempt=attempt_number,
                quality_mode="strict",
            ):
                fine = _fuse_fine(last_observations, cfg)
            fine["fine_quality_status"] = "strict"
            fine["fine_quality_note"] = None
            fine["fine_recovery_attempts"] = list(attempts) + [attempt_record]
            attempt_record.update({
                "fusion_status": "strict_pass",
                "center_scatter_p95_px": float(fine["center_scatter_p95_px"]),
            })
            attempts.append(attempt_record)
            fine["fine_recovery_attempts"] = list(attempts)
            return {
                "success": True,
                "status": "strict",
                "fine": fine,
                "observations": last_observations,
                "intrinsics": last_intrinsics,
                "attempts": attempts,
                "error": None,
            }
        except Exception as exc:
            attempt_record.update({
                "fusion_status": "strict_failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            attempts.append(attempt_record)
            if attempt_index < retry_count:
                print(
                    f"[FINE_RETRY] 孔{hole_id}第{attempt_number}次精定位未通过，"
                    f"准备预热后重拍：{attempt_record['error']}",
                    flush=True,
                )

    degraded_error: str | None = None
    valid_count = len([
        item for item in last_observations
        if item.error is None and item.ellipse is not None
    ])
    if last_intrinsics is not None and valid_count >= int(cfg.fine_stable_min_frames):
        try:
            with timing.measure(
                f"hole_{hole_id:02d}/fine_fusion_degraded",
                hole_id=hole_id,
                processing_order=processing_order,
                quality_mode="degraded",
            ):
                fine = _fuse_fine(
                    last_observations, cfg,
                    max_center_scatter_p95_px=cfg.fine_degraded_max_center_scatter_p95_px,
                )
            if int(fine["valid_frames"]) < int(cfg.fine_stable_min_frames):
                raise RuntimeError(
                    f"降级融合有效帧不足：{fine['valid_frames']}/{cfg.fine_stable_min_frames}"
                )
            fine["fine_quality_status"] = "degraded_fine"
            fine["fine_quality_note"] = (
                f"严格精定位门未通过；稳定P95={float(fine['center_scatter_p95_px']):.3f}px，"
                f"降级门={float(cfg.fine_degraded_max_center_scatter_p95_px):.3f}px"
            )
            fine["fine_recovery_attempts"] = list(attempts)
            attempts.append({
                "attempt": len(attempts) + 1,
                "name": "degraded_fusion",
                "fusion_status": "degraded_pass",
                "center_scatter_p95_px": float(fine["center_scatter_p95_px"]),
            })
            fine["fine_recovery_attempts"] = list(attempts)
            print(
                f"[FINE_DEGRADED] 孔{hole_id}采用稳定降级精定位，"
                f"P95={float(fine['center_scatter_p95_px']):.3f}px",
                flush=True,
            )
            return {
                "success": True,
                "status": "degraded_fine",
                "fine": fine,
                "observations": last_observations,
                "intrinsics": last_intrinsics,
                "attempts": attempts,
                "error": None,
            }
        except Exception as exc:
            degraded_error = f"{type(exc).__name__}: {exc}"

    if degraded_error is None:
        degraded_error = (
            f"降级融合前有效帧不足：{valid_count}/{int(cfg.fine_stable_min_frames)}"
        )
    attempts.append({
        "attempt": len(attempts) + 1,
        "name": "degraded_fusion",
        "fusion_status": "degraded_failed",
        "error": degraded_error,
    })
    print(
        f"[FINE_DEFERRED] 孔{hole_id}多次精定位仍未获得可靠中心，"
        f"当前孔标记deferred并继续后续孔：{degraded_error}",
        flush=True,
    )
    return {
        "success": False,
        "status": "deferred_fine_quality",
        "fine": None,
        "observations": last_observations,
        "intrinsics": last_intrinsics,
        "attempts": attempts,
        "error": degraded_error,
    }


_RUNTIME_DEPENDENCIES = {
    "COARSE_SURFACE_SELECTION_POLICY",
    "Observation",
    "PlaneEstimate",
    "TwoStageSelectionCancelled",
    "_ellipse_ok",
    "_fuse_fine",
    "_fuse_normals",
    "_nearest_detection",
    "_observation_rows",
    "_overlay",
    "_overlay_multi",
    "_plane_estimate_from_info",
    "_record_hole_tracking_event",
    "camera_ray",
    "choose_boxes",
    "cv2",
    "detect",
    "fit_hole_ellipse",
    "get_aligned_frame_bundle",
    "get_rgb_frame_bundle",
    "hole_camera_point",
    "math",
    "np",
    "ray_plane_intersection",
    "undistort_pixels",
}
_PATCHABLE_FUNCTIONS = {
    "_capture_initial_multi_hole_selection",
    "_center_scatter_p95",
    "_coarse_burst_stable",
    "_fine_burst_stable",
    "_capture_coarse_burst",
    "_capture_fine_burst",
    "_capture_fine_with_recovery",
}
_ORIGINAL_FUNCTIONS = {name: globals()[name] for name in _PATCHABLE_FUNCTIONS}
def install_runtime(symbols: dict[str, object]) -> None:
    for name in _RUNTIME_DEPENDENCIES:
        if name in symbols:
            globals()[name] = symbols[name]
    for name in _PATCHABLE_FUNCTIONS:
        value = symbols.get(name)
        if value is None:
            continue
        globals()[name] = (
            _ORIGINAL_FUNCTIONS[name]
            if getattr(value, "_runtime_facade_target", None) == name
            else value
        )
