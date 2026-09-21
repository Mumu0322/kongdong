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

from .config import (
    AUTO_CAPTURE_CFG,
    CAMERA_CFG,
    E7_HAND_EYE_CFG,
    ROBOT_CAMERA_INTEGRATION_CFG,
    E7HandEyeConfig,
)
from .geometry import average_transforms, circular_angle_abs_diff_deg, rotation_error_deg
from .io_utils import atomic_write_json, matrix_to_list
from .samples import (
    CalibSample,
    active_sample_data_root,
    build_raw_data_manifest,
    sample_content_identity,
)
from .solve import (
    pose_coverage_report,
    sample_board_transform,
    sample_calibration_frame,
    sample_quality_flags,
    solve_handeye_estimate,
)


EDGE_REGIONS = ("left", "right", "top", "bottom")


def _sample_data_root(samples: list[CalibSample]) -> Path:
    """返回样本所属的数据目录；测试数据和GUI运行目录都能独立保存分组文件。"""
    return active_sample_data_root(samples)


def fixed_validation_split_path(samples: list[CalibSample]) -> Path:
    """固定留出集的记录路径，不把它混入原始样本清单。"""
    return _sample_data_root(samples) / "e7_validation_split_current.json"


def _split_report(
    ordered: list[CalibSample],
    validation_indices: set[int],
    path: Path,
    source: str,
    rule: str,
) -> tuple[list[CalibSample], list[CalibSample], dict[str, Any]]:
    current_indices = {int(sample.index) for sample in ordered}
    missing = sorted(validation_indices - current_indices)
    if missing:
        raise RuntimeError(
            f"固定验证集缺少样本 {missing}；不能把验证样本替换成新样本，请新建一次标定会话。"
        )
    validation = [sample for sample in ordered if int(sample.index) in validation_indices]
    calibration = [sample for sample in ordered if int(sample.index) not in validation_indices]
    report = {
        "rule": rule,
        "assignment_uses_measurement_residuals": False,
        "assignment_before_fit": True,
        "split_source": source,
        "split_path": str(path.resolve()),
        "calibration_sample_indices": [int(sample.index) for sample in calibration],
        "validation_sample_indices": [int(sample.index) for sample in validation],
    }
    return calibration, validation, report


def _validation_sample_records(
    validation: list[CalibSample],
    data_root: Path,
) -> list[dict[str, Any]]:
    records = [sample_content_identity(sample, data_root) for sample in validation]
    if any(record.get("missing_files") for record in records):
        missing = {
            int(record["index"]): list(record["missing_files"])
            for record in records
            if record.get("missing_files")
        }
        raise RuntimeError(f"固定验证集样本原始文件缺失：{missing}")
    return records


def _identity_comparison_key(record: Any) -> tuple[Any, ...] | None:
    if not isinstance(record, dict):
        return None
    try:
        index = int(record["index"])
        timestamp = str(record["timestamp"])
        content_sha256 = str(record["content_sha256"])
        raw_files = record["raw_files"]
        session = record["session"]
    except (KeyError, TypeError, ValueError):
        return None
    if len(content_sha256) != 64 or not isinstance(raw_files, list) or not isinstance(session, dict):
        return None
    normalized_files: list[tuple[str, int, str]] = []
    for item in raw_files:
        if not isinstance(item, dict):
            return None
        try:
            normalized_files.append((str(item["kind"]), int(item["size_bytes"]), str(item["sha256"])))
        except (KeyError, TypeError, ValueError):
            return None
    return index, timestamp, content_sha256, tuple(sorted(normalized_files)), json.dumps(
        session, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


def get_or_create_fixed_e7_split(
    samples: list[CalibSample],
    cfg: E7HandEyeConfig | None = None,
    persist: bool = True,
) -> tuple[list[CalibSample], list[CalibSample], dict[str, Any]]:
    """建立或读取固定留出集；残差筛选和求解都不能改变它。"""
    cfg = cfg or E7_HAND_EYE_CFG
    ordered = sorted(samples, key=lambda item: item.index)
    if len(ordered) < int(cfg.minimum_total_poses):
        raise ValueError(
            f"固定E7验证集至少需要 {cfg.minimum_total_poses} 组样本，当前 {len(ordered)} 组"
        )
    current_indices = [int(sample.index) for sample in ordered]
    if len(current_indices) != len(set(current_indices)):
        raise RuntimeError("固定E7验证集当前样本存在重复index，不能建立或复用split")
    path = fixed_validation_split_path(ordered)
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_indices = payload.get("validation_sample_indices")
            if not isinstance(raw_indices, list):
                raise ValueError("validation_sample_indices不是列表")
            parsed_indices = [int(value) for value in raw_indices]
            if len(parsed_indices) != len(set(parsed_indices)):
                raise ValueError("validation_sample_indices存在重复index")
            validation_indices = set(parsed_indices)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"固定验证集记录不可读取：{path} ({exc})") from exc
        if not validation_indices:
            raise RuntimeError(f"固定验证集记录为空：{path}；请新建一次标定会话。")
        calibration, validation, report = _split_report(
            ordered,
            validation_indices,
            path,
            "persisted",
            str(payload.get("rule") or "persisted_validation_set_before_residual_filter"),
        )
        persisted_records = payload.get("validation_sample_records")
        if not isinstance(persisted_records, list):
            raise RuntimeError(
                f"固定验证集记录缺少样本内容身份：{path}；旧的仅index记录不能静默信任，请新建一次标定会话。"
            )
        data_root = _sample_data_root(ordered)
        current_records = _validation_sample_records(validation, data_root)
        persisted_by_index: dict[int, tuple[Any, ...]] = {}
        for record in persisted_records:
            key = _identity_comparison_key(record)
            if key is None:
                raise RuntimeError(f"固定验证集记录包含无效样本内容身份：{path}")
            index = int(key[0])
            if index in persisted_by_index:
                raise RuntimeError(f"固定验证集记录存在重复样本身份 index={index}：{path}")
            persisted_by_index[index] = key
        if set(persisted_by_index) != validation_indices:
            raise RuntimeError(f"固定验证集记录的样本身份与validation index不一致：{path}")
        for record in current_records:
            current_key = _identity_comparison_key(record)
            expected_key = persisted_by_index.get(int(record["index"]))
            if current_key is None or expected_key != current_key:
                raise RuntimeError(
                    f"固定验证样本内容已变化或被替换 index={record['index']}；"
                    "不能复用旧holdout，请新建一次标定会话。"
                )
        report["validation_sample_records"] = current_records
    else:
        calibration, validation, report = deterministic_e7_split(ordered, cfg)
        report["split_source"] = "created"
        report["split_path"] = str(path.resolve())
        data_root = _sample_data_root(ordered)
        validation_records = _validation_sample_records(validation, data_root)
        report["validation_sample_records"] = validation_records
        if persist:
            payload = {
                "record_type": "e7_handeye_fixed_validation_split",
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "rule": report["rule"],
                "assignment_before_fit": True,
                "assignment_uses_measurement_residuals": False,
                "dataset_sample_indices_at_creation": [int(sample.index) for sample in ordered],
                "calibration_sample_indices_at_creation": report["calibration_sample_indices"],
                "validation_sample_indices": report["validation_sample_indices"],
                "validation_sample_records": validation_records,
            }
            atomic_write_json(path, payload)

    validation_count = len(validation)
    if validation_count < int(cfg.minimum_validation_poses):
        raise RuntimeError(
            f"固定验证集只有 {validation_count} 组，至少需要 {cfg.minimum_validation_poses} 组"
        )
    if validation_count / len(ordered) < float(cfg.minimum_validation_fraction):
        raise RuntimeError(
            f"固定验证集占比 {validation_count / len(ordered):.1%}，低于 {cfg.minimum_validation_fraction:.1%}"
        )
    return calibration, validation, report


def assess_board_fixity(
    samples: list[CalibSample],
    cfg: E7HandEyeConfig | None = None,
    T_pose_source_sensor: np.ndarray | None = None,
) -> dict[str, Any]:
    """通过反推所有样本的 T_base_board 散布评估板是否真的固定。

    如果标定板在基座坐标系中真的固定，那么所有样本反推的板位姿应该高度一致。

    ``T_pose_source_sensor`` 是从机器人位姿源到相机的变换。如果调用方没有
    提供它，这里会先用当前样本做一次临时手眼估计；不能把这个变换当成单位阵，
    否则相机与 TCP 之间的实际安装偏置会被错误地判定为标定板移动。
    """
    cfg = cfg or E7_HAND_EYE_CFG

    if not samples:
        return {
            "board_position_stable": False,
            "board_orientation_stable": False,
            "reason": "no_samples",
        }

    calibration_frames = sorted({sample_calibration_frame(sample) for sample in samples})
    if calibration_frames != ["rgb_camera"]:
        return {
            "board_position_stable": False,
            "board_orientation_stable": False,
            "reason": "board_fixity_requires_rgb_camera_samples",
            "calibration_frames": calibration_frames,
        }

    estimate_source = "provided"
    if T_pose_source_sensor is None:
        try:
            estimate = solve_handeye_estimate(samples, quiet=True)
            T_pose_source_sensor = np.asarray(estimate["T_final"], dtype=np.float64)
            estimate_source = "preliminary_handeye_estimate"
        except Exception as exc:
            return {
                "board_position_stable": False,
                "board_orientation_stable": False,
                "reason": "preliminary_handeye_estimate_failed",
                "error": str(exc),
            }

    T_pose_source_sensor = np.asarray(T_pose_source_sensor, dtype=np.float64)
    if T_pose_source_sensor.shape != (4, 4) or not np.all(np.isfinite(T_pose_source_sensor)):
        return {
            "board_position_stable": False,
            "board_orientation_stable": False,
            "reason": "invalid_pose_source_sensor_transform",
        }

    # 使用统一的 ^baseT_pose_source @ ^pose_sourceT_sensor @ ^sensorT_board
    # 链路反推每个样本的 T_base_board。
    T_base_boards: list[np.ndarray] = []
    timestamps: list[float] = []

    for sample in samples:
        if sample.T_base_tool is None or sample.T_rgb_board is None:
            continue
        T_base_board = sample.T_base_tool @ T_pose_source_sensor @ sample_board_transform(
            sample, "rgb_camera"
        )
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

    position_rms_mm = float(np.sqrt(np.mean(center_deviations ** 2)))
    position_max_mm = float(center_deviations.max())

    # 平移和旋转均相对于整组均值，避免结果依赖样本顺序。
    reference_R = average_transforms(T_base_boards)[:3, :3]
    rotation_errors_deg = [
        rotation_error_deg(reference_R, T[:3, :3])
        for T in T_base_boards
    ]
    orientation_rms_deg = float(np.sqrt(np.mean(np.square(rotation_errors_deg))))
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
    position_stable = (
        position_rms_mm <= cfg.maximum_board_position_scatter_rms_mm
        and position_max_mm <= cfg.maximum_board_position_scatter_max_mm
    )
    orientation_stable = (
        orientation_rms_deg <= cfg.maximum_board_orientation_scatter_rms_deg
        and orientation_max_deg <= cfg.maximum_board_orientation_scatter_max_deg
    )

    return {
        "board_position_stable": bool(position_stable),
        "board_orientation_stable": bool(orientation_stable),
        "time_continuous": bool(time_continuous),
        "position_scatter_rms_mm": position_rms_mm,
        "position_scatter_max_mm": position_max_mm,
        "orientation_scatter_rms_deg": orientation_rms_deg,
        "orientation_scatter_max_deg": orientation_max_deg,
        "max_time_gap_hours": max_gap_hours,
        "valid_sample_count": len(T_base_boards),
        "total_sample_count": len(samples),
        "board_center_mean_base_mm": center_mean.tolist(),
        "pose_source_sensor_estimate_source": estimate_source,
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


def _offset_vector(snapshot: Any, key: str) -> np.ndarray | None:
    if not isinstance(snapshot, dict):
        return None
    try:
        values = np.asarray(snapshot.get(key), dtype=np.float64).reshape(-1)
        if values.size < 6 or not np.all(np.isfinite(values[:6])):
            return None
        return values[:6]
    except (TypeError, ValueError):
        return None


def _tcp_offset_consistency(samples: list[CalibSample], cfg: E7HandEyeConfig) -> dict[str, Any]:
    """检查一个采集会话是否混用了不同TCP，且实际/配置TCP是否一致。"""
    actual_rows = [_offset_vector(sample.robot_snapshot, "actual_tcp_offset_sdk_m_rad") for sample in samples]
    configured_rows = [_offset_vector(sample.robot_snapshot, "configured_tcp_offset_sdk_m_rad") for sample in samples]
    actual_complete = all(row is not None for row in actual_rows)
    configured_complete = all(row is not None for row in configured_rows)
    result: dict[str, Any] = {
        "actual_metadata_complete": bool(actual_complete),
        "configured_metadata_complete": bool(configured_complete),
        "actual_offset_consistent": False,
        "actual_configured_offsets_match": False,
        "actual_offset_reference_sdk_m_rad": None,
        "actual_offset_max_xyz_delta_mm": None,
        "actual_offset_max_rpy_delta_deg": None,
        "actual_configured_max_xyz_delta_mm": None,
        "actual_configured_max_rpy_delta_deg": None,
    }
    if not actual_complete:
        return result

    actual = np.asarray(actual_rows, dtype=np.float64)
    reference = actual[0]
    xyz_delta = np.abs(actual[:, :3] - reference[:3]) * 1000.0
    rpy_delta = np.asarray(
        [
            [circular_angle_abs_diff_deg(np.degrees(row[i]), np.degrees(reference[i])) for i in range(3, 6)]
            for row in actual
        ],
        dtype=np.float64,
    )
    result.update({
        "actual_offset_reference_sdk_m_rad": [float(value) for value in reference],
        "actual_offset_max_xyz_delta_mm": float(np.max(xyz_delta)),
        "actual_offset_max_rpy_delta_deg": float(np.max(rpy_delta)),
        "actual_offset_consistent": bool(
            np.max(xyz_delta) <= float(cfg.maximum_tcp_offset_xyz_delta_mm)
            and np.max(rpy_delta) <= float(cfg.maximum_tcp_offset_rpy_delta_deg)
        ),
    })

    if configured_complete:
        configured = np.asarray(configured_rows, dtype=np.float64)
        paired_xyz_delta = np.abs(actual[:, :3] - configured[:, :3]) * 1000.0
        paired_rpy_delta = np.asarray(
            [
                [
                    circular_angle_abs_diff_deg(np.degrees(actual[row_index, i]), np.degrees(configured[row_index, i]))
                    for i in range(3, 6)
                ]
                for row_index in range(len(actual))
            ],
            dtype=np.float64,
        )
        result.update({
            "actual_configured_max_xyz_delta_mm": float(np.max(paired_xyz_delta)),
            "actual_configured_max_rpy_delta_deg": float(np.max(paired_rpy_delta)),
            "actual_configured_offsets_match": bool(
                np.max(paired_xyz_delta) <= float(cfg.maximum_tcp_offset_xyz_delta_mm)
                and np.max(paired_rpy_delta) <= float(cfg.maximum_tcp_offset_rpy_delta_deg)
            ),
        })
    return result


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
    indices = [int(sample.index) for sample in ordered]
    if len(indices) != len(set(indices)):
        raise ValueError("deterministic E7 split不接受重复index")
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
    fixed_split_path = fixed_validation_split_path(ordered)
    if fixed_split_path.is_file():
        preliminary_fit_samples, _, _ = get_or_create_fixed_e7_split(
            ordered, cfg, persist=False,
        )
        preliminary_fit_scope = "persisted_fixed_calibration"
    elif len(ordered) >= int(cfg.minimum_total_poses) and len({int(sample.index) for sample in ordered}) == len(ordered):
        preliminary_fit_samples, _, _ = deterministic_e7_split(ordered, cfg)
        preliminary_fit_scope = "deterministic_calibration_preview"
    elif len(ordered) >= int(cfg.minimum_total_poses):
        preliminary_fit_samples = []
        preliminary_fit_scope = "not_run_duplicate_indices"
    else:
        preliminary_fit_samples = ordered
        preliminary_fit_scope = "all_samples_below_fixed_split_threshold"
    manifest = build_raw_data_manifest(ordered)
    view = view_coverage_report(ordered, cfg)
    pose_coverage = pose_coverage_report(ordered)
    indices = [int(sample.index) for sample in ordered]
    pose_sources = sorted({_pose_source(sample) for sample in ordered})
    calibration_frames = sorted({sample_calibration_frame(sample) for sample in ordered})
    camera_serials = sorted({_camera_serial(sample) for sample in ordered if _camera_serial(sample)})
    robot_ids = sorted({_robot_id(sample) for sample in ordered if _robot_id(sample)})
    profile_signatures = {_camera_profile_signature(sample) for sample in ordered}
    tcp_offsets = _tcp_offset_consistency(ordered, cfg)
    bad_quality: dict[int, list[str]] = {}
    for sample in ordered:
        flags = sample_quality_flags(sample)
        if not _capture_quality_complete(sample):
            flags = [*flags, "capture_quality_incomplete_or_failed"]
        if flags:
            bad_quality[int(sample.index)] = flags
    expected_serial = str(ROBOT_CAMERA_INTEGRATION_CFG.production_camera_serial).strip()

    # 该初步拟合只服务于板固定性预检，不能参与留出集选择、调参或最终拟合。
    fixity = assess_board_fixity(preliminary_fit_samples, cfg)

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
        "camera_serial_recorded": (
            not cfg.require_camera_serial_metadata
            or (bool(ordered) and all(_camera_serial(sample) for sample in ordered) and len(camera_serials) == 1)
        ),
        "tcp_offset_metadata_complete": (
            not cfg.require_tcp_offset_metadata or bool(tcp_offsets["actual_metadata_complete"])
        ),
        "tcp_offset_consistent": bool(tcp_offsets["actual_offset_consistent"]),
        "actual_configured_tcp_offset_match": bool(
            not tcp_offsets["configured_metadata_complete"]
            or tcp_offsets["actual_configured_offsets_match"]
        ),
        # 候选求解可以在尚未绑定现场设备序列号时进行；最终生产证据检查
        # 仍会要求配置非空且与采集记录一致。
        "production_camera_serial_matches": (
            not expected_serial or camera_serials == [expected_serial]
        ),
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
        "tcp_offset_consistency": tcp_offsets,
        "view_coverage": view,
        "robot_pose_coverage": pose_coverage,
        "board_fixity": fixity,
        "board_fixity_preliminary_fit_scope": preliminary_fit_scope,
        "board_fixity_preliminary_sample_indices": [
            int(sample.index) for sample in preliminary_fit_samples
        ],
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
    # 先建立或核验固定身份，再让预检和正式拟合复用同一 calibration 子集。
    calibration, validation, split = get_or_create_fixed_e7_split(samples, cfg, persist=True)
    dataset = assess_e7_dataset(samples, fixed_board_confirmed, cfg)
    if not dataset["ok"]:
        fixity = dataset["board_fixity"]
        details = []
        failed = set(dataset["failed_checks"])
        if "board_position_numerically_stable" in failed:
            details.append(
                f"板位姿平移不一致：RMS {fixity.get('position_scatter_rms_mm', float('nan')):.3f} mm，"
                f"最大 {fixity.get('position_scatter_max_mm', float('nan')):.3f} mm"
            )
            failed.remove("board_position_numerically_stable")
        if "board_orientation_numerically_stable" in failed:
            details.append(
                f"板位姿旋转不一致：RMS {fixity.get('orientation_scatter_rms_deg', float('nan')):.3f}°，"
                f"最大 {fixity.get('orientation_scatter_max_deg', float('nan')):.3f}°"
            )
            failed.remove("board_orientation_numerically_stable")
        if failed:
            details.append("采集条件未通过：" + ", ".join(sorted(failed)))
        raise RuntimeError("；".join(details))

    if len(validation) < int(cfg.minimum_validation_poses):
        raise RuntimeError("E7独立验证姿态不足")
    if len(validation) / len(samples) < float(cfg.minimum_validation_fraction):
        raise RuntimeError(
            f"E7独立验证比例不足{float(cfg.minimum_validation_fraction):.1%}"
        )
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
    validation_max = float(validation_stats["translation_max_mm"])
    validation_rms = float(validation_stats["translation_rmse_mm"])
    numeric_pass = bool(
        validation_rms <= float(cfg.maximum_validation_center_scatter_rms_mm)
        and validation_max <= float(cfg.maximum_validation_center_scatter_max_mm)
    )
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
            # 副本允许在尚未绑定现场相机序列号时生成候选；最终生产验收
            # 仍会在 calibration_readiness 中强制要求非空且一致。
            "camera_serial": dataset["camera_serials"][0] if dataset["camera_serials"] else "",
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
        "view_center_and_edges_covered": bool(validation_view["ok"]),
        "calibration_sample_indices": split["calibration_sample_indices"],
        "validation_sample_indices": split["validation_sample_indices"],
        "split_rule": split["rule"],
        "fixed_validation_split_path": split["split_path"],
        "fixed_validation_split_source": split["split_source"],
        "validation_view_coverage": validation_view,
        "pose_source": "tcp",
        "calibration_frame": "rgb_camera",
        "rgb_coordinate_convention": "opencv_optical_x_right_y_down_z_forward",
        "T_tcp_rgb_camera": matrix_to_list(T_final),
        "T_base_board_reference_from_calibration": matrix_to_list(calibration_board_reference),
        "validation_reference_source": "calibration_set_T_base_board_mean",
        "validation_center_scatter_rms_mm": validation_rms,
        "validation_center_scatter_p95_mm": validation_p95,
        "validation_center_scatter_max_mm": validation_max,
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
