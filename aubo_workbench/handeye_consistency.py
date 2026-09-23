"""采集和诊断共用的相机、TCP 一致性检查。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .config import AUTO_CAPTURE_CFG
from .geometry import circular_angle_abs_diff_deg
from .samples import CalibSample


def camera_serial(sample: CalibSample) -> str:
    device = (sample.camera_metadata or {}).get("device") or {}
    return str(device.get("serial_number") or "").strip()


def camera_profile_signature(sample: CalibSample) -> str:
    metadata = sample.camera_metadata or {}
    payload = {
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "intrinsics": metadata.get("intrinsics"),
        "capture_mode": metadata.get("capture_mode"),
        "device": metadata.get("device"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def tcp_offset_consistency(samples: list[CalibSample]) -> dict[str, Any]:
    """报告样本间的实际 TCP 偏置，以及实际与控制器配置的偏差。"""
    actual_rows = [_offset_vector(sample.robot_snapshot, "actual_tcp_offset_sdk_m_rad") for sample in samples]
    configured_rows = [_offset_vector(sample.robot_snapshot, "configured_tcp_offset_sdk_m_rad") for sample in samples]
    actual_complete = bool(samples) and all(row is not None for row in actual_rows)
    configured_complete = bool(samples) and all(row is not None for row in configured_rows)
    result: dict[str, Any] = {
        "actual_metadata_complete": actual_complete,
        "configured_metadata_complete": configured_complete,
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
    rpy_delta = np.asarray([
        [circular_angle_abs_diff_deg(np.degrees(row[i]), np.degrees(reference[i])) for i in range(3, 6)]
        for row in actual
    ], dtype=np.float64)
    xyz_max = float(np.max(xyz_delta))
    rpy_max = float(np.max(rpy_delta))
    result.update({
        "actual_offset_reference_sdk_m_rad": [float(value) for value in reference],
        "actual_offset_max_xyz_delta_mm": xyz_max,
        "actual_offset_max_rpy_delta_deg": rpy_max,
        "actual_offset_consistent": (
            xyz_max <= AUTO_CAPTURE_CFG.tcp_offset_tolerance_xyz_mm
            and rpy_max <= AUTO_CAPTURE_CFG.tcp_offset_tolerance_rpy_deg
        ),
    })

    if configured_complete:
        configured = np.asarray(configured_rows, dtype=np.float64)
        paired_xyz = np.abs(actual[:, :3] - configured[:, :3]) * 1000.0
        paired_rpy = np.asarray([
            [circular_angle_abs_diff_deg(np.degrees(actual[row, i]), np.degrees(configured[row, i])) for i in range(3, 6)]
            for row in range(len(actual))
        ], dtype=np.float64)
        xyz_max = float(np.max(paired_xyz))
        rpy_max = float(np.max(paired_rpy))
        result.update({
            "actual_configured_max_xyz_delta_mm": xyz_max,
            "actual_configured_max_rpy_delta_deg": rpy_max,
            "actual_configured_offsets_match": (
                xyz_max <= AUTO_CAPTURE_CFG.tcp_offset_tolerance_xyz_mm
                and rpy_max <= AUTO_CAPTURE_CFG.tcp_offset_tolerance_rpy_deg
            ),
        })
    return result
