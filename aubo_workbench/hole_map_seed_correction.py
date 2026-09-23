"""Map-seed XY correction derived from confirmed per-hole fine localization.

The coarse navigation geometry remains immutable.  A per-hole map build may
also keep fine references for traceability, while this module stores validated
fine-localization calibration in a separate, map-bound artifact
and applies it to an in-memory copy of a map hole before shared-fine planning.
It deliberately corrects XY only: Z and the surface normal still come from the
coarse map and every execution still performs fresh fine localization.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import atomic_write_json, jsonable
from .rotary_sector_map import sector_key, validate_sector_id


SEED_CORRECTION_SCHEMA_VERSION = 1
SEED_CORRECTION_KIND = "hole_map_seed_xy_correction"
DEFAULT_SEED_CORRECTION_FILENAME = "hole_seed_correction.json"


def _file_sha256(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_xy(value: Any, field: str) -> np.ndarray:
    try:
        point = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 不是有效坐标") from exc
    if point.size < 2 or not np.isfinite(point[:2]).all():
        raise ValueError(f"{field} 必须至少包含两个有限数字")
    return point[:2].copy()


def _map_holes(
    map_payload: dict[str, Any], sector_id: int | None,
) -> list[dict[str, Any]]:
    version = int(map_payload.get("schema_version", -1))
    if version == 4:
        if sector_id is None:
            raise ValueError("旋转扇区地图生成种子纠正模型时必须指定 sector_id")
        sector_number = validate_sector_id(int(sector_id))
        sector = (map_payload.get("sectors") or {}).get(sector_key(sector_number))
        if not isinstance(sector, dict):
            raise ValueError(f"地图缺少扇区 {sector_key(sector_number)}")
        raw_holes = (sector.get("holes") or {}).values()
    elif version == 2:
        raw_holes = (map_payload.get("holes") or {}).values()
    else:
        raise ValueError(f"种子纠正模型不支持孔位地图 v{version}")
    holes = [
        hole for hole in raw_holes
        if isinstance(hole, dict)
        and str(hole.get("status", "ready")) in {"ready", "completed"}
    ]
    return sorted(holes, key=lambda hole: int(hole["hole_id"]))


def _reference_holes(reference_report: dict[str, Any]) -> list[dict[str, Any]]:
    raw = (reference_report.get("final_result") or {}).get("holes")
    if not isinstance(raw, list):
        raw = ((reference_report.get("stages") or {}).get("processed_holes") or {}).get(
            "holes"
        )
    if not isinstance(raw, list):
        raise ValueError("逐孔参考报告中没有最终孔结果")
    holes: list[dict[str, Any]] = []
    for hole in raw:
        if not isinstance(hole, dict) or str(hole.get("status")) != "completed":
            continue
        if bool(hole.get("pointcloud_center_fallback")) or str(
            hole.get("fine_quality_status", "")
        ).lower() == "pointcloud_center_fallback":
            # The fallback is sufficient to keep the hole visible in a coarse
            # map, but it is not an RGB-confirmed center for fitting a map
            # seed-correction model.
            continue
        if hole.get("hole_center_base_mm") is None:
            continue
        _finite_xy(hole["hole_center_base_mm"], "逐孔参考孔心")
        holes.append(hole)
    if not holes:
        raise ValueError("逐孔参考报告中没有成功且坐标有效的孔")
    return holes


def _greedy_unique_matches(
    map_xy: np.ndarray,
    reference_xy: np.ndarray,
    *,
    gate_mm: float,
) -> list[tuple[int, int, float]]:
    """Return deterministic one-to-one matches for already coarsely aligned sets."""
    distances = np.linalg.norm(
        map_xy[:, np.newaxis, :] - reference_xy[np.newaxis, :, :], axis=2,
    )
    matches: list[tuple[int, int, float]] = []
    used_map: set[int] = set()
    used_reference: set[int] = set()
    for flat_index in np.argsort(distances, axis=None):
        map_index, reference_index = np.unravel_index(flat_index, distances.shape)
        distance = float(distances[map_index, reference_index])
        if distance > float(gate_mm):
            break
        if map_index in used_map or reference_index in used_reference:
            continue
        matches.append((int(map_index), int(reference_index), distance))
        used_map.add(int(map_index))
        used_reference.add(int(reference_index))
    return sorted(matches)


def _fit_rigid_xy(source_xy: np.ndarray, target_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if source_xy.shape != target_xy.shape or source_xy.ndim != 2 or source_xy.shape[1] != 2:
        raise ValueError("二维刚体拟合输入形状不一致")
    if len(source_xy) < 2:
        raise ValueError("二维刚体拟合至少需要两个匹配孔")
    source_center = np.mean(source_xy, axis=0)
    target_center = np.mean(target_xy, axis=0)
    covariance = (source_xy - source_center).T @ (target_xy - target_center)
    u, _singular, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if float(np.linalg.det(rotation)) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def build_seed_correction_model(
    map_payload: dict[str, Any],
    reference_report: dict[str, Any],
    *,
    map_path: str | Path,
    reference_report_path: str | Path,
    sector_id: int | None,
    match_gate_mm: float = 10.0,
    max_correction_mm: float = 5.0,
) -> dict[str, Any]:
    """Build a map-bound XY seed correction model from confirmed hole centers."""
    if not math.isfinite(match_gate_mm) or match_gate_mm <= 0.0:
        raise ValueError("孔匹配门限必须为正数")
    if not math.isfinite(max_correction_mm) or max_correction_mm <= 0.0:
        raise ValueError("种子最大纠正量必须为正数")
    map_file = Path(map_path).expanduser().resolve()
    reference_file = Path(reference_report_path).expanduser().resolve()
    map_holes = _map_holes(map_payload, sector_id)
    reference_holes = _reference_holes(reference_report)
    map_xy = np.asarray([
        _finite_xy(hole.get("coarse_center_base_mm"), "地图粗孔心")
        for hole in map_holes
    ])
    reference_xy = np.asarray([
        _finite_xy(hole.get("hole_center_base_mm"), "逐孔成功孔心")
        for hole in reference_holes
    ])
    matches = _greedy_unique_matches(map_xy, reference_xy, gate_mm=match_gate_mm)
    if len(matches) < 3:
        raise ValueError(
            f"逐孔参考结果与地图仅匹配 {len(matches)} 个孔，至少需要3个；"
            "请确认它们属于同一工件位置"
        )

    matched_map_xy = np.asarray([map_xy[i] for i, _j, _d in matches])
    matched_reference_xy = np.asarray([reference_xy[j] for _i, j, _d in matches])
    rotation, translation = _fit_rigid_xy(matched_map_xy, matched_reference_xy)
    global_predictions = (rotation @ matched_map_xy.T).T + translation
    global_residuals = matched_reference_xy - global_predictions

    correction_entries: dict[str, Any] = {}
    rejected_large: list[int] = []
    match_distances: list[float] = []
    global_residual_norms: list[float] = []
    for pair_index, (map_index, reference_index, match_distance) in enumerate(matches):
        map_hole = map_holes[map_index]
        reference_hole = reference_holes[reference_index]
        correction = reference_xy[reference_index] - map_xy[map_index]
        correction_norm = float(np.linalg.norm(correction))
        hole_id = int(map_hole["hole_id"])
        if correction_norm > float(max_correction_mm):
            rejected_large.append(hole_id)
            continue
        residual = global_residuals[pair_index]
        correction_entries[f"H{hole_id:02d}"] = {
            "hole_id": hole_id,
            "reference_hole_id": int(reference_hole["hole_id"]),
            "map_xy_base_mm": map_xy[map_index].tolist(),
            "reference_xy_base_mm": reference_xy[reference_index].tolist(),
            "correction_xy_base_mm": correction.tolist(),
            "correction_norm_mm": correction_norm,
            "global_residual_xy_mm": residual.tolist(),
            "global_residual_norm_mm": float(np.linalg.norm(residual)),
            "match_distance_mm": float(match_distance),
            "reference_quality": {
                "fine_quality_status": reference_hole.get("fine_quality_status"),
                "valid_frames": reference_hole.get("valid_frames"),
                "center_scatter_p95_px": reference_hole.get("center_scatter_p95_px"),
                "ellipse_residual_median_px": reference_hole.get(
                    "ellipse_residual_median_px"
                ),
            },
        }
        match_distances.append(float(match_distance))
        global_residual_norms.append(float(np.linalg.norm(residual)))

    if len(correction_entries) < 3:
        raise ValueError("通过最大纠正量门限的匹配孔少于3个，无法生成模型")
    yaw_deg = math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
    unmatched_map_ids = sorted(
        int(hole["hole_id"])
        for index, hole in enumerate(map_holes)
        if f"H{int(hole['hole_id']):02d}" not in correction_entries
    )
    matched_reference_indices = {reference_index for _map_index, reference_index, _d in matches}
    unmatched_reference_ids = sorted(
        int(hole["hole_id"])
        for index, hole in enumerate(reference_holes)
        if index not in matched_reference_indices
    )
    return jsonable({
        "schema_version": SEED_CORRECTION_SCHEMA_VERSION,
        "kind": SEED_CORRECTION_KIND,
        "status": "provisional_single_reference_run",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "coordinate_frame": "robot_base_mm",
        "correction_axes": "xy_only",
        "application_stage": "coarse_map_seed_before_shared_fine_grouping",
        "map": {
            "map_id": map_payload.get("map_id"),
            "map_path": str(map_file),
            "map_sha256": _file_sha256(map_file),
            "sector_id": None if sector_id is None else int(sector_id),
        },
        "reference": {
            "report_path": str(reference_file),
            "run_id": reference_file.parent.name,
            "report_sha256": _file_sha256(reference_file),
            "policy": "completed_per_hole_fine_hole_center_base_xy",
        },
        "matching": {
            "identity_policy": "unique_spatial_match_not_numeric_hole_id",
            "gate_mm": float(match_gate_mm),
            "matched_count": len(correction_entries),
            "map_hole_count": len(map_holes),
            "reference_hole_count": len(reference_holes),
            "coverage_ratio": len(correction_entries) / max(1, len(map_holes)),
            "distance_median_mm": float(np.median(match_distances)),
            "distance_max_mm": float(np.max(match_distances)),
            "unmatched_map_hole_ids": unmatched_map_ids,
            "unmatched_reference_hole_ids": unmatched_reference_ids,
            "rejected_large_correction_hole_ids": sorted(rejected_large),
        },
        "global_rigid_xy": {
            "rotation_2x2": rotation.tolist(),
            "translation_xy_mm": translation.tolist(),
            "yaw_deg": float(yaw_deg),
            "residual_median_mm": float(np.median(global_residual_norms)),
            "residual_max_mm": float(np.max(global_residual_norms)),
        },
        "safety": {
            "max_correction_mm": float(max_correction_mm),
            "runtime_min_anchor_holes": 4,
            "runtime_min_anchor_groups": 2,
            "runtime_inlier_residual_mm": 1.5,
            "runtime_max_center_shift_mm": 10.0,
            "runtime_max_yaw_deg": 2.0,
            "fresh_fine_required": True,
            "final_position_replacement_allowed": False,
        },
        "holes": correction_entries,
    })


def save_seed_correction_model(payload: dict[str, Any], path: str | Path) -> Path:
    validate_seed_correction_model(payload)
    return atomic_write_json(path, jsonable(payload))


def load_seed_correction_model(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"地图种子纠正模型无法读取：{target}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"地图种子纠正模型根节点必须是对象：{target}")
    validate_seed_correction_model(payload)
    return payload


def validate_seed_correction_model(payload: dict[str, Any]) -> None:
    if int(payload.get("schema_version", -1)) != SEED_CORRECTION_SCHEMA_VERSION:
        raise ValueError("地图种子纠正模型版本不兼容")
    if payload.get("kind") != SEED_CORRECTION_KIND:
        raise ValueError("文件不是地图种子纠正模型")
    if payload.get("coordinate_frame") != "robot_base_mm":
        raise ValueError("地图种子纠正模型必须使用 robot_base_mm 坐标系")
    holes = payload.get("holes")
    if not isinstance(holes, dict) or not holes:
        raise ValueError("地图种子纠正模型没有孔级纠正记录")
    max_correction = float((payload.get("safety") or {}).get("max_correction_mm", 0.0))
    if not math.isfinite(max_correction) or max_correction <= 0.0:
        raise ValueError("地图种子纠正模型最大纠正量无效")
    for key, item in holes.items():
        if not isinstance(item, dict):
            raise ValueError(f"地图种子纠正记录 {key} 无效")
        hole_id = int(item.get("hole_id", -1))
        if key != f"H{hole_id:02d}":
            raise ValueError(f"地图种子纠正记录 {key} 的孔号不一致")
        correction = _finite_xy(item.get("correction_xy_base_mm"), f"{key}.correction")
        if float(np.linalg.norm(correction)) > max_correction + 1e-9:
            raise ValueError(f"地图种子纠正记录 {key} 超过模型安全上限")


def validate_model_binding(
    payload: dict[str, Any],
    *,
    map_payload: dict[str, Any],
    map_path: str | Path,
    sector_id: int | None,
) -> None:
    validate_seed_correction_model(payload)
    binding = payload.get("map") or {}
    if binding.get("map_id") != map_payload.get("map_id"):
        raise ValueError("地图种子纠正模型与当前 map_id 不一致")
    model_sector = binding.get("sector_id")
    current_sector = None if sector_id is None else int(sector_id)
    if model_sector != current_sector:
        raise ValueError("地图种子纠正模型与当前扇区不一致")
    expected_hash = str(binding.get("map_sha256") or "").strip()
    if expected_hash and expected_hash != _file_sha256(Path(map_path).resolve()):
        raise ValueError("地图文件已变化，现有种子纠正模型失效，请重新生成")


def apply_seed_correction_to_hole(
    hole: dict[str, Any],
    model: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return a corrected runtime copy of one coarse-map hole."""
    result = dict(hole)
    hole_id = int(result["hole_id"])
    entry = (model.get("holes") or {}).get(f"H{hole_id:02d}")
    if not isinstance(entry, dict):
        return result, None
    correction = _finite_xy(entry.get("correction_xy_base_mm"), "种子纠正量")
    max_correction = float((model.get("safety") or {}).get("max_correction_mm"))
    correction_norm = float(np.linalg.norm(correction))
    if correction_norm > max_correction + 1e-9:
        raise ValueError(f"孔 H{hole_id:02d} 的种子纠正量超过安全上限")

    original_center = np.asarray(result["coarse_center_base_mm"], dtype=np.float64).reshape(3)
    original_plane = np.asarray(
        result.get("coarse_plane_point_base_mm", original_center), dtype=np.float64,
    ).reshape(3)
    corrected_center = original_center.copy()
    corrected_plane = original_plane.copy()
    corrected_center[:2] += correction
    corrected_plane[:2] += correction
    result["coarse_center_base_mm"] = corrected_center.tolist()
    result["coarse_plane_point_base_mm"] = corrected_plane.tolist()
    metadata = {
        "applied": True,
        "hole_id": hole_id,
        "reference_hole_id": entry.get("reference_hole_id"),
        "correction_xy_base_mm": correction.tolist(),
        "correction_norm_mm": correction_norm,
        "original_center_base_mm": original_center.tolist(),
        "corrected_center_base_mm": corrected_center.tolist(),
        "z_and_normal_policy": "preserve_coarse_map",
    }
    result["coarse_seed_correction"] = metadata
    return result, metadata


def fit_runtime_seed_alignment(
    model: dict[str, Any],
    observed_xy_by_hole: dict[int, Any],
    anchor_group_by_hole: dict[int, int],
) -> dict[str, Any]:
    """Robustly align reference fine centers to the current live run.

    A correction built from one historical run is never applied directly.  At
    least two capture groups must first provide strict per-hole fine centers.
    Pair hypotheses provide a small deterministic RANSAC suitable for the
    typical 4--30 anchors without adding a SciPy dependency.
    """
    validate_seed_correction_model(model)
    safety = model.get("safety") or {}
    min_holes = max(3, int(safety.get("runtime_min_anchor_holes", 4)))
    min_groups = max(2, int(safety.get("runtime_min_anchor_groups", 2)))
    residual_gate = float(safety.get("runtime_inlier_residual_mm", 1.5))
    max_center_shift = float(safety.get("runtime_max_center_shift_mm", 10.0))
    max_yaw_deg = float(safety.get("runtime_max_yaw_deg", 2.0))
    entries = model.get("holes") or {}
    hole_ids = sorted(
        int(hole_id) for hole_id in observed_xy_by_hole
        if f"H{int(hole_id):02d}" in entries
        and int(hole_id) in anchor_group_by_hole
    )
    groups = sorted({int(anchor_group_by_hole[hole_id]) for hole_id in hole_ids})
    pending = {
        "success": False,
        "anchor_hole_ids": hole_ids,
        "anchor_group_indices": groups,
        "min_anchor_holes": min_holes,
        "min_anchor_groups": min_groups,
        "inlier_residual_gate_mm": residual_gate,
    }
    if len(hole_ids) < min_holes or len(groups) < min_groups:
        pending["reason"] = "insufficient_live_anchors"
        return pending

    reference_xy = np.asarray([
        _finite_xy(entries[f"H{hole_id:02d}"]["reference_xy_base_mm"], "参考孔心")
        for hole_id in hole_ids
    ])
    observed_xy = np.asarray([
        _finite_xy(observed_xy_by_hole[hole_id], "本轮锚点孔心")
        for hole_id in hole_ids
    ])
    best: tuple[int, float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for first in range(len(hole_ids)):
        for second in range(first + 1, len(hole_ids)):
            if float(np.linalg.norm(reference_xy[first] - reference_xy[second])) < 20.0:
                continue
            rotation, translation = _fit_rigid_xy(
                reference_xy[[first, second]], observed_xy[[first, second]],
            )
            predicted = (rotation @ reference_xy.T).T + translation
            residuals = np.linalg.norm(observed_xy - predicted, axis=1)
            inliers = residuals <= residual_gate
            count = int(np.sum(inliers))
            median = float(np.median(residuals[inliers])) if count else float("inf")
            candidate = (count, -median, rotation, translation, inliers)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None or best[0] < min_holes:
        pending["reason"] = "no_consistent_live_anchor_transform"
        return pending

    inliers = best[4]
    inlier_groups = {
        int(anchor_group_by_hole[hole_ids[index]])
        for index in range(len(hole_ids)) if bool(inliers[index])
    }
    if len(inlier_groups) < min_groups:
        pending["reason"] = "live_anchor_inliers_from_one_group_only"
        return pending
    rotation, translation = _fit_rigid_xy(reference_xy[inliers], observed_xy[inliers])
    predicted = (rotation @ reference_xy.T).T + translation
    residuals = np.linalg.norm(observed_xy - predicted, axis=1)
    inliers = residuals <= residual_gate
    if int(np.sum(inliers)) < min_holes:
        pending["reason"] = "refit_lost_required_live_anchors"
        return pending
    final_inlier_groups = {
        int(anchor_group_by_hole[hole_ids[index]])
        for index in range(len(hole_ids)) if bool(inliers[index])
    }
    if len(final_inlier_groups) < min_groups:
        pending["reason"] = "refit_live_anchor_inliers_from_one_group_only"
        return pending
    reference_center = np.mean(reference_xy[inliers], axis=0)
    center_shift = (rotation @ reference_center + translation) - reference_center
    center_shift_mm = float(np.linalg.norm(center_shift))
    yaw_deg = math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
    if center_shift_mm > max_center_shift:
        pending.update({
            "reason": "live_alignment_center_shift_exceeds_gate",
            "center_shift_mm": center_shift_mm,
        })
        return pending
    if abs(yaw_deg) > max_yaw_deg:
        pending.update({
            "reason": "live_alignment_yaw_exceeds_gate",
            "yaw_deg": yaw_deg,
        })
        return pending
    inlier_ids = [
        hole_ids[index] for index in range(len(hole_ids)) if bool(inliers[index])
    ]
    return {
        **pending,
        "success": True,
        "reason": "live_reference_alignment_ready",
        "rotation_2x2": rotation.tolist(),
        "translation_xy_mm": translation.tolist(),
        "yaw_deg": float(yaw_deg),
        "center_shift_xy_mm": center_shift.tolist(),
        "center_shift_mm": center_shift_mm,
        "inlier_hole_ids": inlier_ids,
        "outlier_hole_ids": sorted(set(hole_ids) - set(inlier_ids)),
        "inlier_group_indices": sorted({
            int(anchor_group_by_hole[hole_id]) for hole_id in inlier_ids
        }),
        "residual_mm_by_hole": {
            str(hole_id): float(residuals[index])
            for index, hole_id in enumerate(hole_ids)
        },
        "residual_p95_mm": float(np.percentile(residuals[inliers], 95)),
    }


def apply_runtime_aligned_seed_correction(
    hole: dict[str, Any],
    model: dict[str, Any],
    alignment: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Correct a runtime planning copy after live alignment is available."""
    result = dict(hole)
    hole_id = int(result["hole_id"])
    if not bool(alignment.get("success", False)):
        return result, {
            "applied": False,
            "hole_id": hole_id,
            "reason": "live_alignment_not_ready",
        }
    entry = (model.get("holes") or {}).get(f"H{hole_id:02d}")
    if not isinstance(entry, dict):
        return result, {
            "applied": False,
            "hole_id": hole_id,
            "reason": "hole_not_calibrated",
        }
    original_center_value = result.get("initial_center_base_mm")
    if original_center_value is None:
        original_center_value = result.get("coarse_center_base_mm")
    original_center = np.asarray(original_center_value, dtype=np.float64).reshape(3)
    plane_value = result.get("coarse_plane_point_base_mm", original_center)
    original_plane = np.asarray(plane_value, dtype=np.float64).reshape(3)
    reference_xy = _finite_xy(entry.get("reference_xy_base_mm"), "参考孔心")
    rotation = np.asarray(alignment["rotation_2x2"], dtype=np.float64).reshape(2, 2)
    translation = np.asarray(alignment["translation_xy_mm"], dtype=np.float64).reshape(2)
    predicted_xy = rotation @ reference_xy + translation
    correction = predicted_xy - original_center[:2]
    correction_norm = float(np.linalg.norm(correction))
    max_correction = float((model.get("safety") or {}).get("max_correction_mm", 5.0))
    if correction_norm > max_correction:
        return result, {
            "applied": False,
            "hole_id": hole_id,
            "reason": "runtime_correction_exceeds_gate",
            "correction_xy_base_mm": correction.tolist(),
            "correction_norm_mm": correction_norm,
            "max_correction_mm": max_correction,
        }
    corrected_center = original_center.copy()
    corrected_plane = original_plane.copy()
    corrected_center[:2] = predicted_xy
    corrected_plane[:2] += correction
    for field in ("initial_center_base_mm", "coarse_center_base_mm"):
        if result.get(field) is not None:
            result[field] = corrected_center.copy()
    result["coarse_plane_point_base_mm"] = corrected_plane.copy()
    metadata = {
        "applied": True,
        "hole_id": hole_id,
        "reference_hole_id": entry.get("reference_hole_id"),
        "correction_xy_base_mm": correction.tolist(),
        "correction_norm_mm": correction_norm,
        "original_center_base_mm": original_center.tolist(),
        "corrected_center_base_mm": corrected_center.tolist(),
        "alignment_inlier_hole_ids": alignment.get("inlier_hole_ids", []),
        "z_and_normal_policy": "preserve_current_coarse_geometry",
    }
    result["coarse_seed_correction"] = metadata
    return result, metadata
