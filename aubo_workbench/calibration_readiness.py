#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5 E7手眼交叉验证与E8 TCP/负载/夹持证据硬门。"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from .config import E7_HAND_EYE_CFG, ROBOT_CAMERA_INTEGRATION_CFG, RobotCameraIntegrationConfig


def _valid_rigid_transform(value) -> bool:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return False
    rotation = matrix[:3, :3]
    return bool(
        np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)
    )


def _read_json(path: Path) -> tuple[dict | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _at_least(value, minimum: int) -> bool:
    try:
        return int(value) >= int(minimum)
    except (TypeError, ValueError):
        return False


def _at_most(value, maximum: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and 0.0 <= numeric <= float(maximum)


def _string(data: dict, key: str) -> bool:
    return bool(str(data.get(key) or "").strip())


def _int_list(value) -> list[int] | None:
    if not isinstance(value, list):
        return None
    try:
        result = [int(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if len(result) == len(set(result)) else None


def _sha256(value) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(value or "").strip()))


def _pose_coverage_valid(report, pose_count: int) -> bool:
    try:
        return bool(
            isinstance(report, dict)
            and report.get("ok") is True
            and report.get("warnings") == []
            and float(report["pose_ranges"]["A"]["range"]) >= 10.0
            and float(report["angle_circular_span_deg"]["C"]) >= 50.0
            and int(report["relative_rotation_deg"]["pair_count"]) == int(pose_count) * (int(pose_count) - 1) // 2
            and float(report["relative_rotation_deg"]["mean"]) >= 15.0
        )
    except (KeyError, TypeError, ValueError):
        return False


def _evidence_result(
    checks: dict,
    status: str,
    evidence_path: Path,
    evidence_error: str | None,
    scope_note: str,
) -> dict:
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "ok": not failures,
        "status": "ok" if not failures else status,
        "checks": checks,
        "failed_checks": failures,
        "evidence_path": str(evidence_path),
        "evidence_error": evidence_error,
        "scope_note": scope_note,
    }


def assess_handeye_cross_validation(
    payload: dict | None = None,
    cfg: RobotCameraIntegrationConfig | None = None,
) -> dict:
    cfg = cfg or ROBOT_CAMERA_INTEGRATION_CFG
    path = Path(cfg.handeye_validation_evidence_path).resolve()
    error = None
    if payload is None:
        payload, error = _read_json(path)
    data = payload or {}
    total_poses = data.get("total_pose_count")
    calibration_poses = data.get("calibration_pose_count")
    validation_poses = data.get("validation_pose_count")
    calibration_indices = _int_list(data.get("calibration_sample_indices"))
    validation_indices = _int_list(data.get("validation_sample_indices"))
    indices_disjoint = bool(
        calibration_indices is not None
        and validation_indices is not None
        and not (set(calibration_indices) & set(validation_indices))
    )
    result_rows = data.get("validation_pose_results")
    result_indices = None
    result_rows_valid = False
    result_rows_rms_matches = False
    validation_regions: set[str] = set()
    if isinstance(result_rows, list):
        try:
            result_indices = [int(item["sample_index"]) for item in result_rows]
            row_scatter = [float(item["center_scatter_mm"]) for item in result_rows]
            validation_regions = {str(item.get("view_region") or "").strip().lower() for item in result_rows}
            result_rows_valid = all(
                _at_most(item.get("center_scatter_mm"), 10000.0)
                and isinstance(item.get("board_center_base_mm"), list)
                and len(item["board_center_base_mm"]) == 3
                and all(math.isfinite(float(v)) for v in item["board_center_base_mm"])
                for item in result_rows
            )
            row_rms = math.sqrt(sum(value * value for value in row_scatter) / len(row_scatter))
            result_rows_rms_matches = math.isclose(
                row_rms,
                float(data.get("validation_center_scatter_rms_mm")),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            result_indices = None
            result_rows_valid = False
            result_rows_rms_matches = False
    manifest = data.get("raw_data_manifest")
    manifest_files = manifest.get("files") if isinstance(manifest, dict) else None
    manifest_complete = bool(
        isinstance(manifest, dict)
        and manifest.get("complete") is True
        and manifest.get("missing_files") == []
        and manifest.get("raw_data_sha256") == data.get("raw_data_sha256")
        and isinstance(manifest_files, list)
        and manifest_files
        and all(
            isinstance(item, dict)
            and _sha256(item.get("sha256"))
            and _string(item, "path")
            and _at_least(item.get("size_bytes"), 1)
            for item in manifest_files
        )
    )
    try:
        split_valid = (
            int(total_poses) == int(calibration_poses) + int(validation_poses)
            and int(validation_poses) / int(total_poses) >= 0.20
        )
        calibration_count_int = int(calibration_poses)
        validation_count_int = int(validation_poses)
    except (TypeError, ValueError, ZeroDivisionError):
        split_valid = False
        calibration_count_int = -1
        validation_count_int = -1
    checks = {
        "evidence_available": payload is not None,
        "record_type_valid": data.get("record_type") == "e7_handeye_cross_validation",
        "not_template": data.get("template_only") is False,
        "marked_validated": data.get("validated") is True,
        "candidate_motion_lock_explicitly_cleared": data.get("do_not_use_for_motion") is False,
        "production_eligible_explicitly_confirmed": data.get("production_eligible") is True,
        "production_camera_serial_matches": (
            str(data.get("camera_serial") or "").strip()
            == str(cfg.production_camera_serial).strip()
        ),
        "robot_identified": _string(data, "robot_id"),
        "camera_mount_identified": _string(data, "camera_mount_id"),
        "calibration_identified": _string(data, "calibration_id"),
        "independent_verifier_identified": _string(data, "verified_by"),
        "raw_data_integrity_recorded": _sha256(data.get("raw_data_sha256")),
        "raw_data_manifest_complete": manifest_complete,
        "fixed_board_confirmed": data.get("fixed_board_during_validation") is True,
        "minimum_total_poses": _at_least(total_poses, E7_HAND_EYE_CFG.minimum_total_poses),
        "calibration_validation_split_valid": split_valid,
        "minimum_validation_poses": _at_least(
            validation_poses, E7_HAND_EYE_CFG.minimum_validation_poses,
        ),
        "view_center_and_edges_covered": data.get("view_center_and_edges_covered") is True,
        "independent_validation_set": data.get("validation_poses_excluded_from_fit") is True,
        "split_assigned_before_fit": data.get("split_assignment_before_fit") is True,
        "split_does_not_use_measurement_residuals": (
            data.get("split_assignment_uses_measurement_residuals") is False
        ),
        "calibration_validation_indices_disjoint": indices_disjoint,
        "split_index_counts_match": bool(
            calibration_indices is not None
            and validation_indices is not None
            and len(calibration_indices) == calibration_count_int
            and len(validation_indices) == validation_count_int
        ),
        "calibration_pose_excitation_sufficient": _pose_coverage_valid(
            data.get("calibration_pose_coverage"), calibration_count_int,
        ),
        "validation_pose_excitation_sufficient": _pose_coverage_valid(
            data.get("validation_pose_coverage"), validation_count_int,
        ),
        "validation_pose_results_traceable": bool(
            result_rows_valid
            and result_indices is not None
            and validation_indices is not None
            and sorted(result_indices) == sorted(validation_indices)
        ),
        "validation_result_rows_match_reported_rms": bool(result_rows_valid and result_rows_rms_matches),
        "validation_rows_cover_center_and_two_edges": bool(
            "center" in validation_regions
            and len(validation_regions & {"left", "right", "top", "bottom"}) >= 2
        ),
        "tcp_pose_source_used": str(data.get("pose_source") or "").strip().lower() == "tcp",
        "rgb_camera_calibration_frame_used": str(data.get("calibration_frame") or "").strip().lower()
        == "rgb_camera",
        "rgb_coordinate_convention_explicit": data.get("rgb_coordinate_convention")
        == "opencv_optical_x_right_y_down_z_forward",
        "rgb_handeye_transform_valid": _valid_rigid_transform(data.get("T_tcp_rgb_camera")),
        "rgb_pnp_measurement_source_confirmed": bool(
            isinstance(data.get("solver"), dict)
            and data["solver"].get("rgb_pnp_board_pose_used") is True
            and data["solver"].get("depth_based_board_pose_used") is False
        ),
        "validation_reference_is_from_calibration_set": bool(
            data.get("validation_reference_source") == "calibration_set_T_base_board_mean"
            and _valid_rigid_transform(data.get("T_base_board_reference_from_calibration"))
        ),
        "validation_scatter_rms_within_0_10_mm": _at_most(
            data.get("validation_center_scatter_rms_mm"), 0.10,
        ),
    }
    return _evidence_result(
        checks, "handeye_cross_validation_not_ready", path, error,
        "求解器内部残差或毫米级一致性不能替代固定板20%独立姿态交叉验证。",
    )
