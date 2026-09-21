"""扇区信息的独立归档。

自动分区报告本身仍保存在每次定位运行目录中，便于完整复盘。本模块再
把同一结果按 S01、S02 ... 分开写入稳定的扇区目录，供后续建图、复核和
现场参数维护使用；它不参与机器人运动和孔位坐标计算。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

from aubo_workbench.auto_sector_selection import (
    AutoSectorConfig,
    AutoSectorSelectionResult,
    normalize_angle_deg,
)
from aubo_workbench.io_utils import atomic_write_json, jsonable


SECTOR_INFO_SCHEMA_VERSION = 1


def sector_info_key(sector_id: int) -> str:
    return f"S{int(sector_id):02d}"


def _safe_snapshot_name(source_run_dir: str | Path | None) -> str:
    raw = "" if source_run_dir is None else Path(source_run_dir).name
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    if raw:
        return raw
    return datetime.now().strftime("snapshot-%Y%m%d_%H%M%S_%f")[:-3]


def sector_definition(config: AutoSectorConfig, sector_id: int) -> dict[str, Any]:
    """返回一个扇区的角度范围和当前静态图像基准。"""
    config.validate()
    sector_number = int(sector_id)
    if not 1 <= sector_number <= int(config.sector_count):
        raise ValueError(
            f"扇区编号必须在1..{int(config.sector_count)}范围内：{sector_id!r}"
        )
    width = 360.0 / float(config.sector_count)
    pose_records = [
        dict(record)
        for record in config.sector_pose_records
        if record.get("sector_id") in (None, sector_number)
    ]
    if config.partition_mode == "polygons":
        region = next((r for r in config.regions if int(r["sector_id"]) == sector_number), None)
        return {
            "sector_id": sector_number,
            "sector_key": sector_info_key(sector_number),
            "partition_mode": "polygons",
            "defined": region is not None,
            "polygon_px": region["polygon_px"] if region else None,
            "pose_record_id": region.get("pose_record_id") if region else None,
            "frame_image_path": region.get("frame_image_path") if region else None,
            "image_size_px": list(config.image_size_px),
            "overlap_policy": "lowest_sector_id",
            "boundary_policy": "included",
            "pose_records": pose_records,
        }
    start = normalize_angle_deg(
        float(config.zero_angle_deg) + (sector_number - 1) * width
    )
    end = normalize_angle_deg(start + width)
    return {
        "sector_id": sector_number,
        "sector_key": sector_info_key(sector_number),
        "sector_count": int(config.sector_count),
        "angle_start_deg": float(start),
        "angle_end_deg": float(end),
        "angle_interval": "left_closed_right_open",
        "origin_px": list(config.origin_px) if config.origin_px is not None else None,
        "zero_angle_deg": float(config.zero_angle_deg),
        "boundary_margin_deg": float(config.boundary_margin_deg),
        "roi_polygon_px": (
            [list(point) for point in config.roi_polygon_px]
            if config.roi_polygon_px is not None else None
        ),
        "pose_records": pose_records,
    }


def build_sector_info_payload(
    result: AutoSectorSelectionResult,
    sector_id: int,
    *,
    source_run_dir: str | Path | None = None,
    generated_at: str | None = None,
    global_report_path: str | Path | None = None,
    global_overlay_path: str | Path | None = None,
) -> dict[str, Any]:
    """提取单个扇区的候选、选孔和审计信息。"""
    definition = sector_definition(result.config, sector_id)
    sector_number = int(sector_id)
    selected_set = {int(value) for value in result.selected_indices}
    candidates = [
        item.to_dict()
        for item in result.candidates
        if int(item.sector_id or -1) == sector_number
    ]
    selected_indices = [
        int(item.detection_index)
        for item in result.candidates
        if int(item.sector_id or -1) == sector_number
        and int(item.detection_index) in selected_set
    ]
    pending_indices = [
        int(item.detection_index)
        for item in result.candidates
        if int(item.sector_id or -1) == sector_number
        and item.assignment_status == "pending_boundary"
    ]
    rejected = [
        dict(item)
        for item in result.rejected
        if item.get("sector_id") is not None
        and int(item.get("sector_id")) == sector_number
    ]
    payload = {
        "schema_version": SECTOR_INFO_SCHEMA_VERSION,
        "mode": "static_image_sector_info",
        "generated_at": generated_at or datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": None if source_run_dir is None else str(source_run_dir),
        "global_report_path": None if global_report_path is None else str(global_report_path),
        "global_overlay_path": None if global_overlay_path is None else str(global_overlay_path),
        "definition": definition,
        "config": result.config.to_dict(),
        "candidate_count": len(candidates),
        "selected_detection_indices": selected_indices,
        "pending_boundary_detection_indices": pending_indices,
        "candidates": candidates,
        "rejected": rejected,
        "coverage": dict(result.coverage),
        "warnings": list(result.warnings),
    }
    return jsonable(payload)


def write_sector_info_snapshots(
    result: AutoSectorSelectionResult,
    root: str | Path,
    *,
    source_run_dir: str | Path | None = None,
    overlay: np.ndarray | None = None,
    snapshot_name: str | None = None,
) -> dict[str, Any]:
    """把一次自动分区结果写入独立的扇区目录。

    目录结构为 ``<root>/Sxx/sector_definition.json``、
    ``<root>/Sxx/latest.json`` 和 ``<root>/runs/<snapshot>/``。六个扇区
    即使本次没有检测到孔也会建立目录，避免把“没有候选”和“没有该扇区”
    混淆。
    """
    result.config.validate()
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_name or _safe_snapshot_name(source_run_dir)
    snapshot = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(snapshot)).strip("._")
    if not snapshot:
        snapshot = _safe_snapshot_name(None)
    global_run_dir = root_path / "runs" / snapshot
    if global_run_dir.exists():
        snapshot = f"{snapshot}-{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
        global_run_dir = root_path / "runs" / snapshot
    global_run_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().isoformat(timespec="seconds")
    global_report_path = global_run_dir / "auto_sector_selection.json"
    global_overlay_path = global_run_dir / "auto_sector_selection_overlay.png"
    atomic_write_json(global_report_path, result.to_dict())
    if overlay is not None:
        if not isinstance(overlay, np.ndarray):
            raise ValueError("overlay必须是numpy数组")
        if not cv2.imwrite(str(global_overlay_path), overlay):
            raise RuntimeError(f"扇区叠加图写入失败：{global_overlay_path}")

    sector_entries: dict[str, Any] = {}
    for sector_id in range(1, int(result.config.sector_count) + 1):
        key = sector_info_key(sector_id)
        sector_dir = root_path / key
        snapshot_dir = sector_dir / "runs" / snapshot
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        definition = sector_definition(result.config, sector_id)
        definition_path = sector_dir / "sector_definition.json"
        atomic_write_json(definition_path, definition)
        payload = build_sector_info_payload(
            result,
            sector_id,
            source_run_dir=source_run_dir,
            generated_at=generated_at,
            global_report_path=global_report_path,
            global_overlay_path=(global_overlay_path if overlay is not None else None),
        )
        snapshot_path = snapshot_dir / "sector_info.json"
        atomic_write_json(snapshot_path, payload)
        latest_path = sector_dir / "latest.json"
        atomic_write_json(latest_path, payload)
        sector_entries[key] = {
            "sector_id": int(sector_id),
            "sector_info_path": str(snapshot_path),
            "latest_path": str(latest_path),
            "candidate_count": int(payload["candidate_count"]),
            "selected_count": len(payload["selected_detection_indices"]),
            "pending_boundary_count": len(payload["pending_boundary_detection_indices"]),
        }

    manifest = {
        "schema_version": SECTOR_INFO_SCHEMA_VERSION,
        "mode": "static_image_sector_info_index",
        "updated_at": generated_at,
        "source_run_dir": None if source_run_dir is None else str(source_run_dir),
        "snapshot": snapshot,
        "sector_count": int(result.config.sector_count),
        "sectors": sector_entries,
        "global_report_path": str(global_report_path),
        "global_overlay_path": str(global_overlay_path) if overlay is not None else None,
    }
    atomic_write_json(root_path / "index.json", manifest)
    return jsonable(manifest)


__all__ = [
    "SECTOR_INFO_SCHEMA_VERSION",
    "build_sector_info_payload",
    "sector_definition",
    "sector_info_key",
    "write_sector_info_snapshots",
]
