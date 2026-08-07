#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CalibSample 的定义、JSON/CSV 读写、磁盘文件归档。"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .camera import depth_to_vis
from .config import CAMERA_CFG
from .io_utils import (
    atomic_write_json,
    list_to_matrix,
    make_dir,
    matrix_to_list,
    move_path_to_dir,
    timestamp_str,
    unique_existing_paths,
)


@dataclass
class CalibSample:
    index: int
    timestamp: str
    T_base_tool: np.ndarray
    T_pointcloud_board: np.ndarray | None
    robot_snapshot: dict[str, Any]
    board_status: str
    charuco_count: int
    valid_3d_count: int
    corner_rmse_mm: float
    corner_max_error_mm: float
    plane_rmse_mm: float
    plane_inlier_count: int
    board_mask_point_count: int
    rgb_path: str = ""
    overlay_path: str = ""
    depth_vis_path: str = ""
    sample_json_path: str = ""
    camera_metadata: dict[str, Any] = field(default_factory=dict)
    capture_quality: dict[str, Any] = field(default_factory=dict)
    board_center_uv: list[float] = field(default_factory=list)
    image_size_wh: list[int] = field(default_factory=list)
    view_region: str = "unknown"
    calibration_frame: str = "pointcloud"
    T_rgb_board: np.ndarray | None = None
    rgb_pnp_inlier_count: int = 0
    rgb_reprojection_rmse_px: float = float("nan")
    rgb_reprojection_max_px: float = float("nan")


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _float_or_nan(value: Any) -> float:
    numeric = _finite_or_none(value)
    return float("nan") if numeric is None else numeric


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return _finite_or_none(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def sample_to_json_dict(sample: CalibSample) -> dict[str, Any]:
    return {
        "index": sample.index,
        "timestamp": sample.timestamp,
        "T_base_tool": matrix_to_list(sample.T_base_tool),
        "T_pointcloud_board": (
            None if sample.T_pointcloud_board is None else matrix_to_list(sample.T_pointcloud_board)
        ),
        "calibration_frame": sample.calibration_frame,
        "T_rgb_board": None if sample.T_rgb_board is None else matrix_to_list(sample.T_rgb_board),
        "rgb_pnp_inlier_count": int(sample.rgb_pnp_inlier_count),
        "rgb_reprojection_rmse_px": _finite_or_none(sample.rgb_reprojection_rmse_px),
        "rgb_reprojection_max_px": _finite_or_none(sample.rgb_reprojection_max_px),
        "robot_snapshot": sample.robot_snapshot,
        "board_status": sample.board_status,
        "charuco_count": sample.charuco_count,
        "valid_3d_count": sample.valid_3d_count,
        "corner_rmse_mm": _finite_or_none(sample.corner_rmse_mm),
        "corner_max_error_mm": _finite_or_none(sample.corner_max_error_mm),
        "plane_rmse_mm": _finite_or_none(sample.plane_rmse_mm),
        "plane_inlier_count": int(sample.plane_inlier_count),
        "board_mask_point_count": int(sample.board_mask_point_count),
        "rgb_path": sample.rgb_path,
        "overlay_path": sample.overlay_path,
        "depth_vis_path": sample.depth_vis_path,
        "sample_json_path": sample.sample_json_path,
        "camera_metadata": sample.camera_metadata,
        "capture_quality": _json_safe(sample.capture_quality),
        "board_center_uv": [float(v) for v in sample.board_center_uv],
        "image_size_wh": [int(v) for v in sample.image_size_wh],
        "view_region": sample.view_region,
    }


def sample_from_json_dict(data: dict[str, Any]) -> CalibSample:
    return CalibSample(
        index=int(data["index"]),
        timestamp=str(data["timestamp"]),
        T_base_tool=list_to_matrix(data["T_base_tool"]),
        T_pointcloud_board=(
            None if data.get("T_pointcloud_board") is None else list_to_matrix(data["T_pointcloud_board"])
        ),
        robot_snapshot=dict(data.get("robot_snapshot", {})),
        board_status=str(data.get("board_status", "")),
        charuco_count=int(data.get("charuco_count", 0)),
        valid_3d_count=int(data.get("valid_3d_count", 0)),
        corner_rmse_mm=_float_or_nan(data.get("corner_rmse_mm")),
        corner_max_error_mm=_float_or_nan(data.get("corner_max_error_mm")),
        plane_rmse_mm=_float_or_nan(data.get("plane_rmse_mm")),
        plane_inlier_count=int(data.get("plane_inlier_count", 0)),
        board_mask_point_count=int(data.get("board_mask_point_count", 0)),
        rgb_path=str(data.get("rgb_path", "")),
        overlay_path=str(data.get("overlay_path", "")),
        depth_vis_path=str(data.get("depth_vis_path", "")),
        sample_json_path=str(data.get("sample_json_path", "")),
        camera_metadata=dict(data.get("camera_metadata", {})),
        capture_quality=dict(data.get("capture_quality", {})),
        board_center_uv=[float(v) for v in data.get("board_center_uv", [])],
        image_size_wh=[int(v) for v in data.get("image_size_wh", [])],
        view_region=str(data.get("view_region", "unknown")),
        calibration_frame=str(data.get("calibration_frame", "pointcloud")),
        T_rgb_board=(
            None if data.get("T_rgb_board") is None else list_to_matrix(data["T_rgb_board"])
        ),
        rgb_pnp_inlier_count=int(data.get("rgb_pnp_inlier_count", 0)),
        rgb_reprojection_rmse_px=_float_or_nan(data.get("rgb_reprojection_rmse_px")),
        rgb_reprojection_max_px=_float_or_nan(data.get("rgb_reprojection_max_px")),
    )


def save_sample(
    sample: CalibSample,
    color_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    depth_mm: np.ndarray | None,
) -> CalibSample:
    base_dir = Path(CAMERA_CFG.save_dir)
    img_dir = base_dir / "images"
    sample_dir = base_dir / "samples"
    make_dir(img_dir)
    make_dir(sample_dir)

    name = f"sample_{sample.index:03d}_{sample.timestamp}"
    rgb_path = img_dir / f"{name}_rgb.png"
    overlay_path = img_dir / f"{name}_overlay.png"
    depth_vis_path = img_dir / f"{name}_depth_vis.png"
    json_path = sample_dir / f"{name}.json"

    sample.rgb_path = str(rgb_path)
    sample.overlay_path = str(overlay_path)
    sample.depth_vis_path = str(depth_vis_path) if depth_mm is not None else ""
    sample.sample_json_path = str(json_path)

    written: list[Path] = []
    try:
        images_to_write = [(rgb_path, color_bgr), (overlay_path, overlay_bgr)]
        if depth_mm is not None:
            images_to_write.append((depth_vis_path, depth_to_vis(depth_mm)))
        for path, image in images_to_write:
            if not cv2.imwrite(str(path), image):
                raise OSError(f"OpenCV写图失败: {path}")
            written.append(path)
        atomic_write_json(json_path, sample_to_json_dict(sample))
        written.append(json_path)
        append_sample_csv(sample)
    except Exception:
        for path in written:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    print(f"[SAVE] 已保存样本 {sample.index}: {json_path}")
    return sample


_CSV_FIELDS = [
    "index", "timestamp", "board_status", "charuco_count", "valid_3d_count",
    "corner_rmse_mm", "corner_max_error_mm", "plane_rmse_mm", "plane_inlier_count", "board_mask_point_count",
    "calibration_frame", "rgb_pnp_inlier_count", "rgb_reprojection_rmse_px", "rgb_reprojection_max_px",
    "robot_brand", "robot_name", "robot_type", "robot_pose_source",
    "robot_pose_values_mm_deg", "robot_pose_values_sdk_m_rad",
    "rgb_path", "overlay_path", "depth_vis_path", "sample_json_path",
]


def sample_csv_row(sample: CalibSample) -> dict[str, Any]:
    snap = sample.robot_snapshot or {}
    return {
        "index": sample.index,
        "timestamp": sample.timestamp,
        "board_status": sample.board_status,
        "charuco_count": sample.charuco_count,
        "valid_3d_count": sample.valid_3d_count,
        "corner_rmse_mm": f"{sample.corner_rmse_mm:.6f}" if np.isfinite(sample.corner_rmse_mm) else "",
        "corner_max_error_mm": (
            f"{sample.corner_max_error_mm:.6f}" if np.isfinite(sample.corner_max_error_mm) else ""
        ),
        "plane_rmse_mm": f"{sample.plane_rmse_mm:.6f}" if np.isfinite(sample.plane_rmse_mm) else "",
        "plane_inlier_count": sample.plane_inlier_count,
        "board_mask_point_count": sample.board_mask_point_count,
        "calibration_frame": sample.calibration_frame,
        "rgb_pnp_inlier_count": sample.rgb_pnp_inlier_count,
        "rgb_reprojection_rmse_px": (
            f"{sample.rgb_reprojection_rmse_px:.6f}" if np.isfinite(sample.rgb_reprojection_rmse_px) else ""
        ),
        "rgb_reprojection_max_px": (
            f"{sample.rgb_reprojection_max_px:.6f}" if np.isfinite(sample.rgb_reprojection_max_px) else ""
        ),
        "robot_brand": snap.get("robot_brand", "AUBO"),
        "robot_name": snap.get("robot_name", ""),
        "robot_type": snap.get("robot_type", ""),
        "robot_pose_source": snap.get("pose_source", ""),
        "robot_pose_values_mm_deg": json.dumps(snap.get("pose_values", []), ensure_ascii=False),
        "robot_pose_values_sdk_m_rad": json.dumps(snap.get("pose_values_sdk_m_rad", []), ensure_ascii=False),
        "rgb_path": sample.rgb_path,
        "overlay_path": sample.overlay_path,
        "depth_vis_path": sample.depth_vis_path,
        "sample_json_path": sample.sample_json_path,
    }


def append_sample_csv(sample: CalibSample) -> None:
    csv_path = Path(CAMERA_CFG.save_dir) / "charuco_pointcloud_samples.csv"
    make_dir(csv_path.parent)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(sample_csv_row(sample))


def rewrite_sample_csv(samples: list[CalibSample]) -> None:
    csv_path = Path(CAMERA_CFG.save_dir) / "charuco_pointcloud_samples.csv"
    make_dir(csv_path.parent)
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            for sample in samples:
                writer.writerow(sample_csv_row(sample))
        temporary.replace(csv_path)
    finally:
        temporary.unlink(missing_ok=True)


def load_existing_samples() -> list[CalibSample]:
    sample_dir = Path(CAMERA_CFG.save_dir) / "samples"
    if not sample_dir.exists():
        return []
    samples: list[CalibSample] = []
    for path in sorted(sample_dir.glob("sample_*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            samples.append(sample_from_json_dict(data))
        except Exception as exc:
            print(f"[WARN] 读取样本失败 {path}: {exc}")
    samples.sort(key=lambda s: s.index)
    print(f"[INFO] 已加载历史样本 {len(samples)} 个")
    return samples


def sample_disk_paths(sample: CalibSample) -> list[Path]:
    base_dir = Path(CAMERA_CFG.save_dir).resolve()
    paths = unique_existing_paths([
        sample.sample_json_path, sample.rgb_path, sample.overlay_path, sample.depth_vis_path,
    ])
    prefix = f"sample_{sample.index:03d}_{sample.timestamp}"
    for folder in (base_dir / "samples", base_dir / "images"):
        if folder.exists():
            paths.extend(unique_existing_paths(list(folder.glob(prefix + "*"))))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        try:
            within_active_dir = path.resolve().is_relative_to(base_dir)
        except (OSError, ValueError):
            within_active_dir = False
        if not within_active_dir:
            print(f"[WARN] 跳过活动采集目录外的样本路径，不移动：{path}")
            continue
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def archive_removed_samples(removed: list[dict[str, Any]], report: dict[str, Any], timestamp_str: str) -> Path:
    """把自动剔除的坏样本挪到归档目录，附带剔除报告。"""
    archive_dir = Path(CAMERA_CFG.save_dir) / f"archived_auto_prune_{timestamp_str}"
    sample_archive_dir = archive_dir / "samples"
    image_archive_dir = archive_dir / "images"
    make_dir(sample_archive_dir)
    make_dir(image_archive_dir)

    csv_path = Path(CAMERA_CFG.save_dir) / "charuco_pointcloud_samples.csv"
    if csv_path.exists():
        import shutil

        shutil.copy2(str(csv_path), str(archive_dir / "charuco_pointcloud_samples_before_prune.csv"))

    moved: list[dict[str, Any]] = []
    for item in removed:
        sample: CalibSample = item["sample"]
        moved_paths: list[str] = []
        for path in sample_disk_paths(sample):
            target_dir = sample_archive_dir if path.suffix.lower() == ".json" else image_archive_dir
            moved_paths.append(move_path_to_dir(path, target_dir))
        item_report = {k: v for k, v in item.items() if k != "sample"}
        item_report["moved_paths"] = moved_paths
        moved.append(item_report)

    report = dict(report)
    report["removed"] = moved
    report_path = archive_dir / "auto_prune_report.json"
    atomic_write_json(report_path, report)
    return archive_dir


def archive_samples(
    samples_to_archive: list[CalibSample],
    remaining_samples: list[CalibSample],
    reason: str,
    archive_prefix: str = "archived_user_removed",
) -> Path | None:
    """把不再参与求解的样本移出活动目录，并同步重写唯一CSV。"""
    if not samples_to_archive:
        return None
    archive_dir = Path(CAMERA_CFG.save_dir) / f"{archive_prefix}_{timestamp_str()}"
    sample_archive_dir = archive_dir / "samples"
    image_archive_dir = archive_dir / "images"
    make_dir(sample_archive_dir)
    make_dir(image_archive_dir)

    moved: list[dict[str, Any]] = []
    for sample in samples_to_archive:
        moved_paths: list[str] = []
        for path in sample_disk_paths(sample):
            target_dir = sample_archive_dir if path.suffix.lower() == ".json" else image_archive_dir
            moved_paths.append(move_path_to_dir(path, target_dir))
        moved.append({
            "index": int(sample.index),
            "timestamp": sample.timestamp,
            "reason": str(reason),
            "moved_paths": moved_paths,
        })

    report_path = archive_dir / "archive_report.json"
    atomic_write_json(report_path, {
        "record_type": "handeye_sample_archive",
        "reason": str(reason),
        "archived_samples": moved,
        "remaining_sample_indices": [int(item.index) for item in remaining_samples],
    })
    rewrite_sample_csv(remaining_samples)
    return archive_dir


def build_raw_data_manifest(samples: list[CalibSample]) -> dict[str, Any]:
    """计算活动样本文件清单及组合SHA256，供E7候选证据追溯。"""
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    for sample in sorted(samples, key=lambda item: item.index):
        fields = ["sample_json_path", "rgb_path", "overlay_path"]
        if sample.calibration_frame != "rgb_camera":
            fields.append("depth_vis_path")
        for field_name in fields:
            raw_path = str(getattr(sample, field_name, "") or "").strip()
            if not raw_path:
                missing.append(f"sample={sample.index}:{field_name}:empty")
                continue
            path = Path(raw_path)
            if not path.is_file():
                missing.append(f"sample={sample.index}:{field_name}:{path}")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append({
                "sample_index": int(sample.index),
                "kind": field_name,
                "path": str(path.resolve()),
                "size_bytes": int(path.stat().st_size),
                "sha256": digest,
            })
    canonical = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "algorithm": "sha256",
        "raw_data_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "files": entries,
        "missing_files": missing,
        "complete": bool(entries) and not missing,
    }
