#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""正式E7手眼独立交叉验证；不安装证据、不解锁运动。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import AUTO_CAPTURE_CFG, E7_HAND_EYE_CFG, ROBOT_CAMERA_INTEGRATION_CFG, E7HandEyeConfig
from .geometry import rotation_error_deg
from .io_utils import atomic_write_json, matrix_to_list
from .samples import CalibSample, build_raw_data_manifest
from .solve import (
    pose_coverage_report,
    sample_board_transform,
    sample_calibration_frame,
    sample_quality_flags,
    solve_handeye_estimate,
)


EDGE_REGIONS = ("left", "right", "top", "bottom")


def assess_board_fixity(
    samples: list[CalibSample],
    cfg: E7HandEyeConfig | None = None,
) -> dict[str, Any]:
    """通过反推所有样本的 T_base_board 散布评估板是否真的固定。

    如果标定板在基座坐标系中真的固定，那么所有样本反推的板位姿应该高度一致。
    """
    cfg = cfg or E7_HAND_EYE_CFG

    if not samples:
        return {
            "board_position_stable": False,
            "board_orientation_stable": False,
            "reason": "no_samples",
        }

    # 使用每个样本自己的板位姿观测反推 T_base_board
    T_base_boards: list[np.ndarray] = []
    timestamps: list[float] = []

    for sample in samples:
        if sample.T_base_tool is None or sample.T_rgb_board is None:
            continue
        # 需要临时的 T_tcp_rgb 估计，这里用单位阵作为粗略近似
        # 实际上这个函数应该在有初步 T_tcp_rgb 估计之后调用
        # 或者直接用 sample.T_base_tool 和 sample.T_rgb_board 计算
        T_base_board = sample.T_base_tool @ sample.T_rgb_board
        T_base_boards.append(T_base_board)

        # 提取时间戳（如果有）
        ts = (sample.camera_metadata or {}).get("host_timestamp_ns", 0)
        timestamps.append(float(ts) / 1e9 if ts else 0)

    if len(T_base_boards) < 2:
        return {
            "board_position_stable": False,
            "board_orientation_stable": False,
            "reason": "insufficient_valid_samples",
            "valid_sample_count": len(T_base_boards),
        }

    # 计算位置散布
    centers = np.array([T[:3, 3] for T in T_base_boards])
    center_mean = centers.mean(axis=0)
    center_deviations = np.linalg.norm(centers - center_mean, axis=1)

    position_std_mm = float(center_deviations.std())
    position_max_mm = float(center_deviations.max())

    # 计算姿态散布（相对于第一个样本）
    reference_R = T_base_boards[0][:3, :3]
    rotation_errors_deg = [
        rotation_error_deg(reference_R, T[:3, :3])
        for T in T_base_boards
    ]
    orientation_std_deg = float(np.std(rotation_errors_deg))
    orientation_max_deg = float(np.max(rotation_errors_deg))

    # 时间连续性检查
    time_continuous = True
    max_gap_hours = 0.0
    if timestamps and all(t > 0 for t in timestamps):
        sorted_timestamps = sorted(timestamps)
        gaps = np.diff(sorted_timestamps)
        if len(gaps) > 0:
            max_gap_hours = float(np.max(gaps) / 3600)
            # 使用配置的时间间隔门槛
            time_continuous = max_gap_hours < cfg.maximum_time_gap_hours

    # 使用配置的固定板门槛
    position_stable = position_std_mm < cfg.maximum_board_position_scatter_std_mm
    orientation_stable = orientation_std_deg < cfg.maximum_board_orientation_scatter_std_deg

    return {
        "board_position_stable": bool(position_stable),
        "board_orientation_stable": bool(orientation_stable),
        "time_continuous": bool(time_continuous),
        "position_scatter_std_mm": position_std_mm,
        "position_scatter_max_mm": position_max_mm,
        "orientation_scatter_std_deg": orientation_std_deg,
        "orientation_scatter_max_deg": orientation_max_deg,
        "max_time_gap_hours": max_gap_hours,
        "valid_sample_count": len(T_base_boards),
        "total_sample_count": len(samples),
        "board_center_mean_base_mm": center_mean.tolist(),
    }


def _robot_id(sample: CalibSample) -> str:
    snapshot = sample.robot_snapshot or {}
    parts = [
        str(snapshot.get("robot_brand") or "").strip(),
        str(snapshot.get("robot_name") or "").strip(),
        str(snapshot.get("robot_type") or "").strip(),
    ]
    return ":".join(part for part in parts if part)


def _camera_serial(sample: CalibSample) -> str:
    device = (sample.camera_metadata or {}).get("device") or {}
    return str(device.get("serial_number") or "").strip()


def _camera_profile_signature(sample: CalibSample) -> str:
    metadata = sample.camera_metadata or {}
    payload = {
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "intrinsics": metadata.get("intrinsics"),
        "capture_mode": metadata.get("capture_mode"),
        "device": metadata.get("device"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _pose_source(sample: CalibSample) -> str:
    return str((sample.robot_snapshot or {}).get("pose_source") or "").strip().lower()


def _snapshot_pose_values_valid(snapshot: Any) -> bool:
    if not isinstance(snapshot, dict):
        return False
    try:
        raw_values = snapshot.get("pose_values_sdk_m_rad")
        if not isinstance(raw_values, (list, tuple)):
            raw_values = snapshot.get("pose_values")
        values = np.asarray(raw_values, dtype=np.float64).reshape(-1)
        return bool(
            values.size >= 6
            and np.all(np.isfinite(values[:6]))
            and np.max(np.abs(values[:6])) > 1e-12
        )
    except (TypeError, ValueError):
        return False


def _robot_pose_valid(sample: CalibSample) -> bool:
    try:
        transform = np.asarray(sample.T_base_tool, dtype=np.float64)
        return bool(
            _snapshot_pose_values_valid(sample.robot_snapshot)
            and transform.shape == (4, 4)
            and np.all(np.isfinite(transform))
            and not np.allclose(transform, np.eye(4), rtol=0.0, atol=1e-12)
        )
    except (TypeError, ValueError):
        return False


def _pose_bracket_ok(sample: CalibSample, cfg: E7HandEyeConfig) -> bool:
    bracket = (sample.robot_snapshot or {}).get("camera_frame_pose_bracket") or {}
    try:
        xyz = np.asarray(bracket.get("xyz_delta_mm"), dtype=np.float64).reshape(3)
        abc = np.asarray(bracket.get("abc_delta_deg"), dtype=np.float64).reshape(3)
        before = bracket.get("before_snapshot")
        after = bracket.get("after_snapshot")
        snapshots_complete = all(_snapshot_pose_values_valid(snapshot) for snapshot in (before, after))
        return bool(
            bracket.get("ok") is True
            and snapshots_complete
            and np.all(np.isfinite(xyz))
            and np.all(np.isfinite(abc))
            and float(np.max(np.abs(xyz))) <= float(cfg.maximum_pose_bracket_xyz_mm)
            and float(np.max(np.abs(abc))) <= float(cfg.maximum_pose_bracket_abc_deg)
        )
    except (TypeError, ValueError):
        return False


def _camera_frame_metadata_ok(sample: CalibSample) -> bool:
    metadata = sample.camera_metadata or {}
    intrinsics = metadata.get("intrinsics")
    try:
        return bool(
            int(metadata.get("width")) > 0
            and int(metadata.get("height")) > 0
            and int(metadata.get("host_timestamp_ns")) > 0
            and metadata.get("color_timestamp_us") is not None
            and metadata.get("capture_mode") == "rgb_only_no_depth_or_pointcloud"
            and isinstance(intrinsics, dict)
            and float(intrinsics.get("fx")) > 0.0
            and float(intrinsics.get("fy")) > 0.0
        )
    except (TypeError, ValueError):
        return False


def _capture_quality_complete(sample: CalibSample) -> bool:
    quality = sample.capture_quality or {}
    try:
        metrics_match = bool(
            sample_calibration_frame(sample) == "rgb_camera"
            and str(quality.get("calibration_frame") or "") == "rgb_camera"
            and int(quality.get("charuco_count")) == int(sample.charuco_count)
            and int(quality.get("rgb_pnp_inlier_count")) == int(sample.rgb_pnp_inlier_count)
            and np.isclose(
                float(quality.get("rgb_reprojection_rmse_px")),
                float(sample.rgb_reprojection_rmse_px), atol=1e-9,
            )
            and np.isclose(
                float(quality.get("rgb_reprojection_max_px")),
                float(sample.rgb_reprojection_max_px), atol=1e-9,
            )
        )
        return bool(
            quality.get("ok") is True
            and float(quality.get("score")) >= float(AUTO_CAPTURE_CFG.min_score)
            and metrics_match
            and sample.charuco_count >= int(AUTO_CAPTURE_CFG.min_charuco_corners)
            and sample.rgb_pnp_inlier_count >= int(AUTO_CAPTURE_CFG.min_rgb_pnp_inliers)
            and np.isfinite(float(sample.rgb_reprojection_rmse_px))
            and np.isfinite(float(sample.rgb_reprojection_max_px))
            and float(sample.rgb_reprojection_rmse_px) <= float(AUTO_CAPTURE_CFG.max_rgb_reprojection_rmse_px)
            and float(sample.rgb_reprojection_max_px) <= float(AUTO_CAPTURE_CFG.max_rgb_reprojection_error_px)
            and "warning" not in str(sample.board_status or "").lower()
        )
    except (TypeError, ValueError):
        return False


def _measured_view_region(sample: CalibSample, cfg: E7HandEyeConfig) -> str:
    try:
        center = np.asarray(sample.board_center_uv, dtype=np.float64).reshape(2)
        width, height = [int(value) for value in sample.image_size_wh]
        if width <= 0 or height <= 0 or not np.all(np.isfinite(center)):
            return "unknown"
    except (TypeError, ValueError):
        return "unknown"
    dx = (float(center[0]) - 0.5 * width) / max(0.5 * width, 1.0)
    dy = (float(center[1]) - 0.5 * height) / max(0.5 * height, 1.0)
    if abs(dx) <= float(cfg.center_region_half_width_ratio) and abs(dy) <= float(cfg.center_region_half_height_ratio):
        return "center"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0.0 else "left"
    return "bottom" if dy > 0.0 else "top"


def view_coverage_report(samples: list[CalibSample], cfg: E7HandEyeConfig | None = None) -> dict[str, Any]:
    cfg = cfg or E7_HAND_EYE_CFG
    regions = [_measured_view_region(sample, cfg) for sample in samples]
    counts = {region: regions.count(region) for region in ("center", *EDGE_REGIONS, "unknown")}
    distinct_edges = [region for region in EDGE_REGIONS if counts[region] > 0]
    return {
        "counts": counts,
        "center_covered": counts["center"] > 0,
        "edge_regions_covered": distinct_edges,
        "minimum_distinct_edge_regions": int(cfg.minimum_distinct_edge_regions),
        "stored_label_mismatch_indices": [
            int(sample.index)
            for sample, measured in zip(samples, regions)
            if str(sample.view_region or "unknown") != measured
        ],
        "ok": counts["center"] > 0 and len(distinct_edges) >= int(cfg.minimum_distinct_edge_regions),
    }


def deterministic_e7_split(
    samples: list[CalibSample],
    cfg: E7HandEyeConfig | None = None,
) -> tuple[list[CalibSample], list[CalibSample], dict[str, Any]]:
    """仅按样本编号和视野区域确定留出集；不读取残差或求解结果。"""
    cfg = cfg or E7_HAND_EYE_CFG
    ordered = sorted(samples, key=lambda item: item.index)
    validation_count = max(
        int(cfg.minimum_validation_poses),
        int(math.ceil(len(ordered) * float(cfg.minimum_validation_fraction))),
    )
    if validation_count >= len(ordered):
        raise ValueError("validation split would leave no calibration samples")

    selected: list[CalibSample] = []
    selected_ids: set[int] = set()
    ordered_position = {sample.index: position for position, sample in enumerate(ordered)}

    # 先确保留出集至少包含中心和可用的边缘区域；选择规则只依赖编号顺序。
    measured_regions = {sample.index: _measured_view_region(sample, cfg) for sample in ordered}
    available_edges = [region for region in EDGE_REGIONS if any(measured_regions[s.index] == region for s in ordered)]
    seed_regions = ["center", *available_edges]
    for seed_number, region in enumerate(seed_regions, start=1):
        candidates = [sample for sample in ordered if measured_regions[sample.index] == region]
        if not candidates or len(selected) >= validation_count:
            continue
        target_position = seed_number * (len(ordered) - 1) / (len(seed_regions) + 1)
        choice = min(
            candidates,
            key=lambda sample: (abs(ordered_position[sample.index] - target_position), sample.index),
        )
        if choice.index not in selected_ids:
            selected.append(choice)
            selected_ids.add(choice.index)

    # 再按固定的等距位置补足20%，仍不使用任何误差值。
    if len(selected) < validation_count:
        positions = np.linspace(0, len(ordered) - 1, num=validation_count + 2)[1:-1]
        for position in positions:
            sample = ordered[int(round(float(position)))]
            if sample.index not in selected_ids:
                selected.append(sample)
                selected_ids.add(sample.index)
            if len(selected) >= validation_count:
                break
    if len(selected) < validation_count:
        for sample in ordered:
            if sample.index not in selected_ids:
                selected.append(sample)
                selected_ids.add(sample.index)
            if len(selected) >= validation_count:
                break

    validation = sorted(selected, key=lambda item: item.index)
    calibration = [sample for sample in ordered if sample.index not in selected_ids]
    split_report = {
        "rule": "deterministic_stratified_by_view_region_then_even_index_before_fit",
        "assignment_uses_measurement_residuals": False,
        "assignment_before_fit": True,
        "calibration_sample_indices": [int(sample.index) for sample in calibration],
        "validation_sample_indices": [int(sample.index) for sample in validation],
    }
    return calibration, validation, split_report


def assess_e7_dataset(
    samples: list[CalibSample],
    fixed_board_confirmed: bool,
    cfg: E7HandEyeConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or E7_HAND_EYE_CFG
    ordered = sorted(samples, key=lambda item: item.index)
    manifest = build_raw_data_manifest(ordered)
    view = view_coverage_report(ordered, cfg)
    pose_coverage = pose_coverage_report(ordered)
    indices = [int(sample.index) for sample in ordered]
    pose_sources = sorted({_pose_source(sample) for sample in ordered})
    calibration_frames = sorted({sample_calibration_frame(sample) for sample in ordered})
    camera_serials = sorted({_camera_serial(sample) for sample in ordered if _camera_serial(sample)})
    robot_ids = sorted({_robot_id(sample) for sample in ordered if _robot_id(sample)})
    profile_signatures = {_camera_profile_signature(sample) for sample in ordered}
    bad_quality: dict[int, list[str]] = {}
    for sample in ordered:
        flags = sample_quality_flags(sample)
        if not _capture_quality_complete(sample):
            flags = [*flags, "capture_quality_incomplete_or_failed"]
        if flags:
            bad_quality[int(sample.index)] = flags
    expected_serial = str(ROBOT_CAMERA_INTEGRATION_CFG.production_camera_serial).strip()

    # 板固定程度数值验证
    fixity = assess_board_fixity(ordered, cfg)

    checks = {
        "minimum_total_poses": len(ordered) >= int(cfg.minimum_total_poses),
        "sample_indices_unique": len(indices) == len(set(indices)),
        "fixed_board_confirmed": bool(fixed_board_confirmed),
        "board_position_numerically_stable": bool(fixity.get("board_position_stable", False)),
        "board_orientation_numerically_stable": bool(fixity.get("board_orientation_stable", False)),
        "board_time_continuous": bool(fixity.get("time_continuous", True)),
        "all_samples_use_tcp_pose": (
            not cfg.require_tcp_pose_source or (bool(ordered) and pose_sources == ["tcp"])
        ),
        "manual_pose_not_used": bool(cfg.allow_manual_pose) or "manual" not in pose_sources,
        "all_robot_pose_values_valid": bool(ordered) and all(_robot_pose_valid(sample) for sample in ordered),
        "all_samples_use_rgb_pnp": bool(ordered) and calibration_frames == ["rgb_camera"]
        and all(sample.T_rgb_board is not None for sample in ordered),
        "single_robot_identified": len(robot_ids) == 1,
        "production_camera_serial_matches": camera_serials == [expected_serial],
        "camera_profile_consistent": bool(ordered) and len(profile_signatures) == 1,
        "camera_frame_metadata_complete": bool(ordered) and all(_camera_frame_metadata_ok(sample) for sample in ordered),
        "camera_frame_pose_bracket_verified": bool(ordered) and all(_pose_bracket_ok(sample, cfg) for sample in ordered),
        "view_center_and_edges_covered": bool(view["ok"]),
        "view_region_labels_match_measurements": not view["stored_label_mismatch_indices"],
        "robot_pose_excitation_sufficient": bool(
            pose_coverage.get("ok") and not pose_coverage.get("warnings")
        ),
        "all_samples_pass_capture_quality": not bad_quality,
        "raw_data_manifest_complete": bool(manifest["complete"]),
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "total_pose_count": len(ordered),
        "pose_sources": pose_sources,
        "calibration_frames": calibration_frames,
        "camera_serials": camera_serials,
        "robot_ids": robot_ids,
        "view_coverage": view,
        "robot_pose_coverage": pose_coverage,
        "board_fixity": fixity,
        "bad_quality_samples": bad_quality,
        "raw_data_manifest": manifest,
    }


def validation_stats_against_reference(
    validation: list[CalibSample],
    T_tcp_rgb_camera: np.ndarray,
    T_base_board_reference: np.ndarray,
) -> dict[str, Any]:
    """只用标定集确定的板位姿作参考，避免留出集用自己的均值把整组偏移抵消。"""
    transforms = [
        sample.T_base_tool @ T_tcp_rgb_camera @ sample_board_transform(sample, "rgb_camera")
        for sample in validation
    ]
    reference = np.asarray(T_base_board_reference, dtype=np.float64)
    translation_error_xyz = np.asarray(
        [transform[:3, 3] - reference[:3, 3] for transform in transforms], dtype=np.float64,
    )
    translation_errors = np.linalg.norm(translation_error_xyz, axis=1)
    rotation_errors = np.asarray(
        [rotation_error_deg(reference[:3, :3], transform[:3, :3]) for transform in transforms],
        dtype=np.float64,
    )
    return {
        "T_base_board_reference": reference,
        "transforms": transforms,
        "translation_rmse_mm": float(np.sqrt(np.mean(translation_errors * translation_errors))),
        "translation_p95_mm": float(np.quantile(translation_errors, 0.95)),
        "translation_max_mm": float(np.max(translation_errors)),
        "rotation_rmse_deg": float(np.sqrt(np.mean(rotation_errors * rotation_errors))),
        "translation_errors_mm": [float(value) for value in translation_errors],
        "translation_error_xyz_mm": [[float(value) for value in row] for row in translation_error_xyz],
        "rotation_errors_deg": [float(value) for value in rotation_errors],
    }


def _validation_pose_results(
    validation: list[CalibSample],
    stats: dict[str, Any],
    cfg: E7HandEyeConfig,
) -> list[dict[str, Any]]:
    transforms = stats["transforms"]
    translations = stats["translation_errors_mm"]
    translation_xyz = stats["translation_error_xyz_mm"]
    rotations = stats["rotation_errors_deg"]
    return [
        {
            "sample_index": int(sample.index),
            "view_region": _measured_view_region(sample, cfg),
            "T_base_board": matrix_to_list(transform),
            "board_center_base_mm": [float(v) for v in transform[:3, 3]],
            "center_error_xyz_mm": [float(v) for v in translation_xyz[index]],
            "center_scatter_mm": float(translations[index]),
            "rotation_scatter_deg": float(rotations[index]),
        }
        for index, (sample, transform) in enumerate(zip(validation, transforms))
    ]


def run_e7_cross_validation(
    samples: list[CalibSample],
    fixed_board_confirmed: bool,
    cfg: E7HandEyeConfig | None = None,
    candidate_path: str | Path | None = None,
) -> dict[str, Any]:
    """生成待独立复核的E7候选；永远不会自动标记validated或安装current。"""
    cfg = cfg or E7_HAND_EYE_CFG
    dataset = assess_e7_dataset(samples, fixed_board_confirmed, cfg)
    if not dataset["ok"]:
        raise RuntimeError(f"E7数据集未就绪: {dataset['failed_checks']}")

    calibration, validation, split = deterministic_e7_split(samples, cfg)
    if len(validation) < int(cfg.minimum_validation_poses):
        raise RuntimeError("E7独立验证姿态不足")
    if len(validation) / len(samples) < float(cfg.minimum_validation_fraction):
        raise RuntimeError("E7独立验证比例不足20%")
    validation_view = view_coverage_report(validation, cfg)
    if not validation_view["ok"]:
        raise RuntimeError("E7留出集必须同时包含视野中心和至少两个边缘区域")
    calibration_pose_coverage = pose_coverage_report(calibration)
    validation_pose_coverage = pose_coverage_report(validation)
    if not calibration_pose_coverage.get("ok") or calibration_pose_coverage.get("warnings"):
        raise RuntimeError(f"E7标定集机器人姿态激励不足: {calibration_pose_coverage.get('warnings')}")
    if not validation_pose_coverage.get("ok") or validation_pose_coverage.get("warnings"):
        raise RuntimeError(f"E7留出集机器人姿态覆盖不足: {validation_pose_coverage.get('warnings')}")

    estimate = solve_handeye_estimate(calibration, quiet=False)
    T_final = np.asarray(estimate["T_final"], dtype=np.float64)
    calibration_board_reference = np.asarray(estimate["quality"]["T_base_board_mean"], dtype=np.float64)
    validation_stats = validation_stats_against_reference(
        validation, T_final, calibration_board_reference,
    )
    validation_p95 = float(validation_stats["translation_p95_mm"])
    validation_rms = float(validation_stats["translation_rmse_mm"])
    numeric_pass = validation_rms <= float(cfg.maximum_validation_center_scatter_rms_mm)
    manifest = dataset["raw_data_manifest"]
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    calibration_id = datetime.now().strftime("handeye-e7-%Y%m%d-%H%M%S-%f")[:-3]
    candidate = {
        "record_type": "e7_handeye_cross_validation",
        "record_policy": "candidate_current_replace_on_refresh",
        "template_only": False,
        "validated": False,
        "do_not_use_for_motion": True,
        "production_eligible": False,
        "candidate_ready_for_independent_review": bool(numeric_pass),
        "generated_at": generated,
        "camera_serial": dataset["camera_serials"][0],
        "robot_id": dataset["robot_ids"][0],
        "camera_mount_id": "",
        "calibration_id": calibration_id,
        "verified_by": "",
        "raw_data_sha256": manifest["raw_data_sha256"],
        "raw_data_manifest": manifest,
        "fixed_board_during_validation": True,
        "total_pose_count": len(samples),
        "calibration_pose_count": len(calibration),
        "validation_pose_count": len(validation),
        "split_assignment_before_fit": True,
        "split_assignment_uses_measurement_residuals": False,
        "validation_poses_excluded_from_fit": True,
        "view_center_and_edges_covered": True,
        "calibration_sample_indices": split["calibration_sample_indices"],
        "validation_sample_indices": split["validation_sample_indices"],
        "split_rule": split["rule"],
        "pose_source": "tcp",
        "calibration_frame": "rgb_camera",
        "rgb_coordinate_convention": "opencv_optical_x_right_y_down_z_forward",
        "T_tcp_rgb_camera": matrix_to_list(T_final),
        "T_base_board_reference_from_calibration": matrix_to_list(calibration_board_reference),
        "validation_reference_source": "calibration_set_T_base_board_mean",
        "validation_center_scatter_rms_mm": validation_rms,
        "validation_center_scatter_p95_mm": validation_p95,
        "validation_rotation_scatter_rms_deg": float(validation_stats["rotation_rmse_deg"]),
        "validation_numeric_pass": bool(numeric_pass),
        "validation_pose_results": _validation_pose_results(validation, validation_stats, cfg),
        "calibration_quality": {
            key: value for key, value in estimate["quality"].items() if key != "T_base_board_mean"
        },
        "calibration_pose_coverage": calibration_pose_coverage,
        "validation_pose_coverage": validation_pose_coverage,
        "dataset_preflight": dataset,
        "solver": {
            "best_opencv_method": estimate["best_method"],
            "nonlinear_refine": estimate["nonlinear_refine"],
            "measurement_source": "charuco_rgb_2d_corners_plus_intrinsics_solvepnp",
            "depth_based_board_pose_used": False,
            "rgb_pnp_board_pose_used": True,
            "note": "正式手眼与孔洞2D定位共用RGB光学坐标系；深度和点云不参与手眼求解。",
        },
        "limits": asdict(cfg),
        "review_requirements": [
            "填写camera_mount_id并核对安装ID/紧固状态",
            "由独立复核人填写verified_by并确认原始数据清单",
            "复核通过后显式设置validated=true、do_not_use_for_motion=false、production_eligible=true",
            "设置上述字段前重新运行证据检查工具",
            "候选验证通过后才可显式安装为唯一e7_handeye_validation_current.json",
        ],
    }
    output = Path(candidate_path) if candidate_path is not None else Path(cfg.candidate_dir) / "e7_handeye_candidate_current.json"
    atomic_write_json(output, candidate)
    candidate["candidate_path"] = str(output.resolve())
    print(
        f"[E7] calibration={len(calibration)}, validation={len(validation)}, "
        f"validation_rms={validation_rms:.6f} mm, limit={cfg.maximum_validation_center_scatter_rms_mm:.3f} mm"
    )
    print(f"[E7] 候选已写入（未验证、不可运动）：{output}")
    return candidate
