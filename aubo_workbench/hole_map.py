#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多孔一次建图、按孔号调用所需的纯文件/数据层。

``coarse_cache`` 只表示粗定位几何是否可以复用；本模块的 HoleMap 表示
一整批孔已经完成粗、精定位，可以在同一工件未移动的前提下直接执行。
模块不依赖相机或机器人，便于离线验证地图完整性和兼容性。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io_utils import atomic_write_json, jsonable


HOLE_MAP_SCHEMA_VERSION = 1
HOLE_MAP_READY_STATUSES = frozenset({"ready", "completed"})
CURRENT_HOLE_MAP_POINTER_KIND = "current_hole_map_pointer"
CURRENT_HOLE_MAP_POINTER_SCHEMA_VERSION = 1


def file_sha256(path: str | Path) -> str | None:
    """返回文件指纹；文件不存在或读取失败时返回 ``None``。"""
    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _finite_vector(value: Any, length: int, field: str) -> list[float]:
    try:
        values = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"地图字段 {field} 不是有效向量") from exc
    if values.size != length or not np.isfinite(values).all():
        raise ValueError(f"地图字段 {field} 必须是 {length} 个有限数字")
    return [float(item) for item in values]


def _optional_vector(value: Any, length: int, field: str) -> list[float] | None:
    if value is None:
        return None
    return _finite_vector(value, length, field)


def _required_result_vector(result: dict[str, Any], keys: Iterable[str], length: int, label: str) -> list[float]:
    for key in keys:
        if result.get(key) is not None:
            return _finite_vector(result[key], length, label)
    raise ValueError(f"孔 {result.get('hole_id', '?')} 缺少 {label}")


def _reference_tcp_pose(result: dict[str, Any]) -> list[float]:
    """选出执行时用来保留该孔姿态的 TCP 参考位姿。"""
    return _required_result_vector(
        result,
        (
            "coarse_fine_reference_tcp_pose_m_rad",
            "fine_tcp_pose_m_rad",
            "final_tcp_pose_m_rad",
        ),
        6,
        "执行参考TCP位姿",
    )


def build_hole_map_payload(
    report: dict[str, Any],
    *,
    map_id: str,
    source_run_dir: str | Path,
    handeye_path: str | Path | None,
    camera_identity: dict[str, Any] | None,
    charuco_model_source: str | Path | None,
    final_target_mode: str,
    final_point_offset_base_mm: Any,
    tcp_xy_offset_mm: Any = None,
    charuco_model_matrix: Any = None,
    charuco_model_bias_mm: Any = None,
) -> dict[str, Any]:
    """从两阶段运行报告生成可执行 HoleMap。

    地图只收录已通过粗/精定位质量门的孔。失败孔不会污染已就绪孔，
    但会被写入 ``deferred_holes`` 供界面显示。
    """
    final_result = report.get("final_result") or {}
    raw_results = final_result.get("holes")
    if not isinstance(raw_results, list):
        raw_results = (report.get("stages") or {}).get("processed_holes", {}).get("holes") or []
    if not isinstance(raw_results, list):
        raise ValueError("运行报告中没有孔定位结果")

    holes: dict[str, dict[str, Any]] = {}
    deferred: list[dict[str, Any]] = []
    errors: list[str] = []
    for result in raw_results:
        if not isinstance(result, dict):
            continue
        try:
            hole_id = int(result["hole_id"])
        except (KeyError, TypeError, ValueError):
            errors.append("发现没有有效 hole_id 的结果")
            continue
        if str(result.get("status", "")).lower() not in {"completed", "ready"}:
            deferred.append({
                "hole_id": hole_id,
                "status": result.get("status"),
                "error": result.get("error") or result.get("fine_quality_note"),
            })
            continue
        try:
            visual_center = _required_result_vector(
                result,
                ("hole_center_base_mm", "hole_center_base_naive_mm"),
                3,
                "最终孔中心",
            )
            coarse_center = _required_result_vector(
                result, ("coarse_center_base_mm",), 3, "粗定位孔中心",
            )
            coarse_plane = _required_result_vector(
                result,
                ("coarse_plane_point_base_mm",),
                3,
                "粗定位平面点",
            )
            coarse_normal = _required_result_vector(
                result,
                ("coarse_normal_toward_camera_base", "plane_normal_toward_camera_base"),
                3,
                "粗定位法向",
            )
            reference_tcp = _reference_tcp_pose(result)
        except ValueError as exc:
            errors.append(str(exc))
            deferred.append({"hole_id": hole_id, "status": "invalid_map_record", "error": str(exc)})
            continue

        if str(hole_id) in holes:
            errors.append(f"孔 {hole_id} 重复出现在运行报告中")
            continue
        holes[f"H{hole_id:02d}"] = {
            "hole_id": hole_id,
            "status": "ready",
            "tracking_identity": result.get("tracking_identity"),
            "initial_selection_order": result.get("initial_selection_order"),
            # final_point 已包含联合XY/逐孔倾斜纠偏，但尚未套用最终点模式的
            # X/Z偏移和ChArUco TCP补偿；执行时仍统一重新计算。
            "visual_center_base_mm": visual_center,
            "fine_xy_base_mm": visual_center[:2],
            "coarse_center_base_mm": coarse_center,
            "coarse_z_base_mm": coarse_center[2],
            "coarse_plane_point_base_mm": coarse_plane,
            "coarse_normal_toward_camera_base": coarse_normal,
            "plane_normal_toward_camera_base": _optional_vector(
                result.get("plane_normal_toward_camera_base"), 3,
                f"孔{hole_id}最终法向",
            ),
            "execution_reference_tcp_pose_m_rad": reference_tcp,
            "diameter_estimate_mm": result.get("diameter_estimate_mm"),
            "matched_diameter_mm": result.get("matched_diameter_mm"),
            "final_point_mode_at_build": result.get("final_point_mode", final_target_mode),
            "fine_xy_source": result.get("fine_xy_source"),
            "fine_z_source": result.get("fine_z_source"),
            "batch_fine_source": result.get("batch_fine_source"),
            "batch_fine_capture_round": result.get("batch_fine_capture_round"),
            "batch_fine_fallback_from_shared": bool(
                result.get("batch_fine_fallback_from_shared", False)
            ),
            "batch_fine_fallback_reason": result.get("batch_fine_fallback_reason"),
            "batch_fine_joint_applied": bool(result.get("batch_fine_joint_applied", False)),
            "quality": {
                "coarse_valid_frames": result.get("coarse_valid_frames"),
                "coarse_center_scatter_p95_px": result.get("coarse_center_scatter_p95_px"),
                "coarse_plane_rmse_mm": result.get("coarse_plane_rmse_mm"),
                "fine_quality_status": result.get("fine_quality_status"),
                "fine_valid_frames": result.get("valid_frames"),
                "fine_center_scatter_p95_px": result.get("center_scatter_p95_px"),
                "joint_summary": result.get("batch_fine_joint_summary"),
            },
            "source_result_keys": {
                "hole_result_type": result.get("hole_result_type"),
                "final_pose_source": result.get("final_pose_source"),
            },
        }

    if not holes:
        detail = "; ".join(errors or ["没有可执行的已完成孔"])
        raise ValueError(f"无法建立孔位地图：{detail}")

    source_path = Path(source_run_dir)
    target_offset = _finite_vector(final_point_offset_base_mm, 3, "最终点基坐标偏移")
    tcp_offset = None if tcp_xy_offset_mm is None else _finite_vector(tcp_xy_offset_mm, 2, "TCP XY偏移")
    camera = dict(camera_identity or {})
    charuco_path = None if charuco_model_source is None else str(charuco_model_source)
    payload: dict[str, Any] = {
        "schema_version": HOLE_MAP_SCHEMA_VERSION,
        "map_id": str(map_id),
        "status": "valid" if not deferred and not errors else "partial",
        "scope": "current_robot_cycle",
        "coordinate_frame": "robot_base_mm",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_run_id": source_path.name,
        "source_report_path": str(source_path / "report.json"),
        "final_target_mode_at_build": str(final_target_mode),
        "final_point_offset_base_mm": target_offset,
        "tcp_xy_offset_at_build_mm": tcp_offset,
        "environment": {
            "robot_id": report.get("robot_id") or report.get("robot_name"),
            "robot_initial_tcp_pose_m_rad": report.get("robot_initial_tcp_pose_m_rad"),
            "home_point": report.get("home_point"),
            "camera_serial": camera.get("serial_number") or camera.get("serial"),
            "camera_identity": camera,
            "handeye_path": None if handeye_path is None else str(handeye_path),
            "handeye_sha256": None if handeye_path is None else file_sha256(handeye_path),
            "charuco_model_source": charuco_path,
            "charuco_model_sha256": None if charuco_path is None else file_sha256(charuco_path),
            "charuco_model_matrix_2x2": None if charuco_model_matrix is None else jsonable(charuco_model_matrix),
            "charuco_model_bias_mm": None if charuco_model_bias_mm is None else jsonable(charuco_model_bias_mm),
        },
        "quality_summary": {
            "ready_holes": len(holes),
            "deferred_holes": deferred,
            "map_errors": errors,
            "batch_coarse_requested": bool((report.get("configuration") or {}).get("batch_coarse_localization", False)),
            "batch_fine_requested": bool((report.get("configuration") or {}).get("batch_fine_localization", False)),
            "batch_fine_joint_requested": bool((report.get("configuration") or {}).get("batch_fine_joint_localization", False)),
        },
        "safety": {
            "requires_same_workpiece_pose": True,
            "requires_same_robot_tcp_and_calibration": True,
            "minimum_lift_mm": 10.0,
            "minimum_final_descent_guard_mm": 10.0,
        },
        "holes": holes,
    }
    return jsonable(payload)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"孔位地图无法读取：{path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"孔位地图根节点必须是对象：{path}")
    return raw


def resolve_hole_map_path(path: str | Path) -> Path:
    """解析实际地图文件；``current.json`` 可以是自动更新的指针文件。

    指针只允许指向 ``current.json`` 所在地图目录的子目录，避免一个误写的
    JSON 把调用流程带到粗定位缓存或其它任意文件。
    """
    target = Path(path).expanduser()
    if target.name.lower() != "current.json":
        return target
    if not target.is_file():
        return target
    raw = _read_json_object(target)
    if raw.get("kind") != CURRENT_HOLE_MAP_POINTER_KIND:
        # 兼容极早期把 current.json 直接当地图正文保存的情况。
        return target
    if int(raw.get("schema_version", -1)) != CURRENT_HOLE_MAP_POINTER_SCHEMA_VERSION:
        raise ValueError(f"当前孔位地图指针版本不兼容：{target}")
    raw_path = str(raw.get("map_path", "")).strip()
    if not raw_path:
        raise ValueError(f"当前孔位地图指针缺少 map_path：{target}")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = target.parent / candidate
    candidate = candidate.resolve()
    root = target.parent.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"当前孔位地图指针越过地图目录：{candidate}") from exc
    if (
        candidate.name.lower() != "hole_map.json"
        or len(candidate.relative_to(root).parts) != 2
        or candidate == target.resolve()
    ):
        raise ValueError(f"当前孔位地图指针不是有效地图文件：{candidate}")
    return candidate


def publish_current_hole_map(
    payload: dict[str, Any],
    map_path: str | Path,
    current_path: str | Path,
) -> Path | None:
    """把完整地图发布为当前地图；部分地图只保留版本，不替换当前地图。

    ``current.json`` 采用指针而不是地图副本，因此地图内的相对点云产物路径
    始终相对于版本目录解析，不会因当前入口位置不同而失效。
    """
    validate_hole_map(payload)
    if payload.get("status") != "valid":
        return None

    target = Path(map_path).expanduser().resolve()
    current = Path(current_path).expanduser().resolve()
    root = current.parent
    if current.name.lower() != "current.json":
        raise ValueError(f"当前地图入口必须命名为 current.json：{current}")
    try:
        relative_map = target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"地图文件必须位于地图目录内：{target}") from exc
    if target.name.lower() != "hole_map.json" or len(relative_map.parts) != 2:
        raise ValueError(f"地图文件必须是版本目录下的 hole_map.json：{target}")
    pointer = {
        "kind": CURRENT_HOLE_MAP_POINTER_KIND,
        "schema_version": CURRENT_HOLE_MAP_POINTER_SCHEMA_VERSION,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "map_id": payload.get("map_id"),
        "map_status": payload.get("status"),
        "map_path": relative_map.as_posix(),
        "source_run_id": payload.get("source_run_id"),
    }
    return atomic_write_json(current, pointer)


def save_hole_map(payload: dict[str, Any], path: str | Path) -> Path:
    """校验并原子保存地图。"""
    validate_hole_map(payload)
    return atomic_write_json(path, jsonable(payload))


def load_hole_map(path: str | Path) -> dict[str, Any]:
    target = resolve_hole_map_path(path)
    if not target.is_file():
        raise FileNotFoundError(f"孔位地图不存在：{target}")
    payload = _read_json_object(target)
    validate_hole_map(payload)
    return payload


def validate_hole_map(
    payload: dict[str, Any],
    *,
    requested_hole_ids: Iterable[int] | None = None,
    expected_target_mode: str | None = None,
    expected_tcp_xy_offset_mm: Any = None,
) -> list[int]:
    """校验地图结构并返回可执行孔号。"""
    if not isinstance(payload, dict):
        raise ValueError("孔位地图根节点必须是对象")
    if int(payload.get("schema_version", -1)) != HOLE_MAP_SCHEMA_VERSION:
        raise ValueError(f"不支持的孔位地图版本：{payload.get('schema_version')}")
    if payload.get("scope") != "current_robot_cycle":
        raise ValueError("当前只允许调用同一机器人循环内建立的孔位地图")
    if payload.get("coordinate_frame") != "robot_base_mm":
        raise ValueError("孔位地图坐标系不是 robot_base_mm，禁止直接执行")
    if payload.get("status") not in {"valid", "partial"}:
        raise ValueError(f"孔位地图状态不可执行：{payload.get('status')}")
    if expected_target_mode is not None and str(payload.get("final_target_mode_at_build")) != str(expected_target_mode):
        raise ValueError(
            "最终点模式与建图时不一致："
            f"map={payload.get('final_target_mode_at_build')} requested={expected_target_mode}"
        )
    if expected_tcp_xy_offset_mm is not None:
        expected_offset = _finite_vector(expected_tcp_xy_offset_mm, 2, "调用TCP XY偏移")
        stored_offset = payload.get("tcp_xy_offset_at_build_mm")
        if stored_offset is None or not np.allclose(
            np.asarray(stored_offset, dtype=np.float64),
            np.asarray(expected_offset, dtype=np.float64),
            atol=1e-9,
        ):
            raise ValueError(
                "TCP XY补偿与建图时不一致："
                f"map={stored_offset} requested={expected_offset}"
            )
    elif payload.get("tcp_xy_offset_at_build_mm") is not None:
        raise ValueError(
            "该地图建立时使用了固定TCP XY补偿，调用时必须提供相同的 --tcp-xy-offset-mm"
        )
    holes = payload.get("holes")
    if not isinstance(holes, dict) or not holes:
        raise ValueError("孔位地图没有 holes")
    ready_ids: list[int] = []
    for key, hole in holes.items():
        if not isinstance(hole, dict):
            raise ValueError(f"孔位地图条目 {key} 不是对象")
        try:
            hole_id = int(hole["hole_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"孔位地图条目 {key} 的 hole_id 无效") from exc
        if str(key) != f"H{hole_id:02d}":
            raise ValueError(f"孔位地图键 {key} 与 hole_id={hole_id} 不一致")
        _finite_vector(hole.get("visual_center_base_mm"), 3, f"{key}.visual_center_base_mm")
        _finite_vector(hole.get("fine_xy_base_mm"), 2, f"{key}.fine_xy_base_mm")
        _finite_vector(hole.get("coarse_center_base_mm"), 3, f"{key}.coarse_center_base_mm")
        _finite_vector(hole.get("coarse_plane_point_base_mm"), 3, f"{key}.coarse_plane_point_base_mm")
        _finite_vector(hole.get("coarse_normal_toward_camera_base"), 3, f"{key}.coarse_normal_toward_camera_base")
        _finite_vector(hole.get("execution_reference_tcp_pose_m_rad"), 6, f"{key}.execution_reference_tcp_pose_m_rad")
        if str(hole.get("status", "ready")) in HOLE_MAP_READY_STATUSES:
            ready_ids.append(hole_id)
    requested = None if requested_hole_ids is None else [int(item) for item in requested_hole_ids]
    if requested is not None:
        missing = [item for item in requested if item not in ready_ids]
        if missing:
            raise ValueError(f"请求调用的孔不在有效地图中：{missing}")
        return requested
    return sorted(ready_ids)


def get_hole(payload: dict[str, Any], hole_id: int) -> dict[str, Any]:
    key = f"H{int(hole_id):02d}"
    hole = (payload.get("holes") or {}).get(key)
    if not isinstance(hole, dict) or int(hole.get("hole_id", -1)) != int(hole_id):
        raise KeyError(f"孔位地图中不存在有效孔 H{int(hole_id):02d}")
    return hole
