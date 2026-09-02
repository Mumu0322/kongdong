#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""孔位地图的点云归档和可视化产物。

地图正文只保存孔位几何；本模块把共享340 mm粗定位产生的点云另存为：

* ``pointcloud_raw.npz``：带相机坐标、基坐标和帧/孔标签的可复算数据；
* ``pointcloud_base.ply``：可直接用 Open3D、CloudCompare 等工具打开的三维点云；
* ``hole_centers_base.ply``：粗定位中心和最终孔中心标记；
* ``pointcloud_preview.jpg``：无需三维软件即可查看的基坐标 XY/XZ 投影图。

可视化属于诊断产物，不能反过来改变已经通过质量门的定位结果。任何导出
失败都会返回 warning，由调用方写入地图报告，但不会把精定位结果改成成功或失败。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (220, 80, 60),
    (60, 150, 220),
    (80, 180, 90),
    (180, 80, 180),
    (40, 180, 180),
    (220, 150, 50),
    (120, 80, 220),
    (80, 120, 180),
    (180, 180, 60),
    (60, 180, 120),
)


def _colour_for_hole(hole_id: int) -> tuple[int, int, int]:
    return _PALETTE_BGR[(int(hole_id) - 1) % len(_PALETTE_BGR)]


def _atomic_save_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_batch_archive(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = (
            "points_camera_mm",
            "point_hole_ids",
            "point_frame_indices",
            "T_base_camera",
        )
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"共享粗定位点云NPZ缺少字段：{missing}")
        points_camera = np.asarray(archive["points_camera_mm"], dtype=np.float64).reshape(-1, 3)
        hole_ids = np.asarray(archive["point_hole_ids"], dtype=np.int32).reshape(-1)
        frame_indices = np.asarray(archive["point_frame_indices"], dtype=np.int32).reshape(-1)
        transform = np.asarray(archive["T_base_camera"], dtype=np.float64).reshape(4, 4)
    if len(points_camera) == 0:
        raise ValueError("共享粗定位点云为空")
    if len(hole_ids) != len(points_camera) or len(frame_indices) != len(points_camera):
        raise ValueError("共享粗定位点云的孔标签/帧标签长度不一致")
    if not np.isfinite(points_camera).all() or not np.isfinite(transform).all():
        raise ValueError("共享粗定位点云包含非有限值")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points_base = points_camera @ rotation.T + translation
    return (
        points_camera.astype(np.float32),
        points_base.astype(np.float32),
        hole_ids,
        frame_indices,
    )


def _write_point_ply(
    path: Path,
    points_base: np.ndarray,
    hole_ids: np.ndarray,
) -> None:
    points = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    labels = np.asarray(hole_ids, dtype=np.int32).reshape(-1)
    if len(points) != len(labels):
        raise ValueError("PLY点和孔标签长度不一致")
    temporary = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="ascii", newline="\n") as handle:
            handle.write("ply\nformat ascii 1.0\n")
            handle.write(f"element vertex {len(points)}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            handle.write("property int hole_id\nend_header\n")
            for point, hole_id in zip(points, labels):
                blue, green, red = _colour_for_hole(int(hole_id))
                handle.write(
                    f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f} "
                    f"{red} {green} {blue} {int(hole_id)}\n"
                )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_center_ply(path: Path, payload: dict[str, Any]) -> int:
    records: list[tuple[np.ndarray, int, int]] = []
    for raw in (payload.get("holes") or {}).values():
        if not isinstance(raw, dict):
            continue
        try:
            hole_id = int(raw["hole_id"])
            coarse = np.asarray(raw["coarse_center_base_mm"], dtype=np.float64).reshape(3)
            final = np.asarray(raw["visual_center_base_mm"], dtype=np.float64).reshape(3)
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(coarse).all() or not np.isfinite(final).all():
            continue
        records.append((coarse, hole_id, 0))
        records.append((final, hole_id, 1))
    temporary = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="ascii", newline="\n") as handle:
            handle.write("ply\nformat ascii 1.0\n")
            handle.write(f"element vertex {len(records)}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            handle.write("property int hole_id\nproperty uchar point_kind\nend_header\n")
            for point, hole_id, point_kind in records:
                if point_kind == 1:
                    red, green, blue = 0, 0, 0
                else:
                    red, green, blue = 150, 150, 150
                handle.write(
                    f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f} "
                    f"{red} {green} {blue} {hole_id} {point_kind}\n"
                )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(records) // 2


def _balanced_downsample(labels: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(labels) <= max_points:
        return np.arange(len(labels), dtype=np.int64)
    unique = sorted(set(int(value) for value in labels))
    quota = max(1, int(max_points) // max(1, len(unique)))
    selected: list[np.ndarray] = []
    for hole_id in unique:
        indices = np.flatnonzero(labels == int(hole_id))
        if len(indices) > quota:
            indices = indices[np.linspace(0, len(indices) - 1, quota, dtype=np.int64)]
        selected.append(indices)
    result = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
    if len(result) > max_points:
        result = result[np.linspace(0, len(result) - 1, max_points, dtype=np.int64)]
    return result


def _panel_projection(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    points: np.ndarray,
    labels: np.ndarray,
    centers: list[tuple[int, np.ndarray, np.ndarray]],
    horizontal_axis: int,
    vertical_axis: int,
    title: str,
) -> None:
    left, top, right, bottom = rect
    cv2.rectangle(image, (left, top), (right, bottom), (35, 35, 35), 2, cv2.LINE_AA)
    all_values = [points[:, [horizontal_axis, vertical_axis]]]
    for _, coarse, final in centers:
        all_values.append(np.asarray([coarse[[horizontal_axis, vertical_axis]], final[[horizontal_axis, vertical_axis]]]))
    combined = np.concatenate(all_values, axis=0)
    lower = np.min(combined, axis=0)
    upper = np.max(combined, axis=0)
    span = max(float(np.max(upper - lower)), 1.0)
    padding = max(2.0, span * 0.06)
    lower -= padding
    upper += padding
    width = max(1, right - left)
    height = max(1, bottom - top)

    def pixel(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).reshape(-1, 2)
        x = left + np.rint((values[:, 0] - lower[0]) / (upper[0] - lower[0]) * width).astype(np.int32)
        y = bottom - np.rint((values[:, 1] - lower[1]) / (upper[1] - lower[1]) * height).astype(np.int32)
        return np.column_stack((x, y))

    sample = pixel(points[:, [horizontal_axis, vertical_axis]])
    for point, hole_id in zip(sample, labels):
        x, y = int(point[0]), int(point[1])
        if left <= x <= right and top <= y <= bottom:
            cv2.circle(image, (x, y), 1, _colour_for_hole(int(hole_id)), -1, cv2.LINE_AA)
    marker_points = np.asarray([
        np.asarray(final)[[horizontal_axis, vertical_axis]]
        for _, _, final in centers
    ], dtype=np.float64)
    if len(marker_points):
        marker_pixels = pixel(marker_points)
        for (hole_id, coarse, final), marker in zip(centers, marker_pixels):
            coarse_pixel = pixel(np.asarray(coarse)[[horizontal_axis, vertical_axis]])[0]
            cx, cy = int(marker[0]), int(marker[1])
            gx, gy = int(coarse_pixel[0]), int(coarse_pixel[1])
            cv2.drawMarker(image, (gx, gy), (120, 120, 120), cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
            cv2.circle(image, (cx, cy), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image, (cx, cy), 6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"H{int(hole_id):02d}", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, title, (left + 12, top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f"{lower[0]:.1f} .. {upper[0]:.1f} mm", (left + 12, bottom - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(image, f"{lower[1]:.1f} .. {upper[1]:.1f} mm", (right - 150, top + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)


def _write_preview(
    path: Path,
    points_base: np.ndarray,
    labels: np.ndarray,
    payload: dict[str, Any],
    *,
    max_points: int = 120_000,
) -> None:
    centers: list[tuple[int, np.ndarray, np.ndarray]] = []
    for raw in (payload.get("holes") or {}).values():
        if not isinstance(raw, dict):
            continue
        try:
            hole_id = int(raw["hole_id"])
            coarse = np.asarray(raw["coarse_center_base_mm"], dtype=np.float64).reshape(3)
            final = np.asarray(raw["visual_center_base_mm"], dtype=np.float64).reshape(3)
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(coarse).all() and np.isfinite(final).all():
            centers.append((hole_id, coarse, final))
    indices = _balanced_downsample(labels, max_points)
    points = np.asarray(points_base, dtype=np.float64)[indices]
    point_labels = np.asarray(labels, dtype=np.int32)[indices]
    canvas = np.full((1000, 1800, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "Hole map point cloud | robot base frame (mm)", (45, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"points={len(points_base):,}  displayed={len(points):,}  holes={len(centers)}", (48, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (70, 70, 70), 1, cv2.LINE_AA)
    _panel_projection(canvas, (45, 110, 865, 940), points, point_labels, centers, 0, 1, "XY top view")
    _panel_projection(canvas, (935, 110, 1755, 940), points, point_labels, centers, 0, 2, "XZ side view")
    cv2.putText(canvas, "gray cross=coarse XYZ   black circle=final visual XY/Z point", (48, 975), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (60, 60, 60), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("点云预览JPG编码失败")
    _atomic_write_bytes(path, encoded.tobytes())


def export_hole_map_artifacts(
    source_npz: str | Path | None,
    map_dir: str | Path,
    payload: dict[str, Any],
    *,
    overlay_path: str | Path | None = None,
) -> dict[str, Any]:
    """导出地图点云和预览；返回的路径均相对于地图版本目录。"""
    result: dict[str, Any] = {
        "status": "unavailable",
        "coordinate_frame": "robot_base_mm",
    }
    if source_npz is None:
        result["reason"] = "run中没有共享粗定位点云NPZ"
        return result
    source = Path(source_npz).expanduser()
    if not source.is_file():
        result["reason"] = f"共享粗定位点云NPZ不存在：{source}"
        return result
    directory = Path(map_dir).expanduser()
    try:
        points_camera, points_base, labels, frame_indices = _load_batch_archive(source)
        raw_name = "pointcloud_raw.npz"
        _atomic_save_npz(
            directory / raw_name,
            points_camera_mm=points_camera,
            points_base_mm=points_base,
            point_hole_ids=labels,
            point_frame_indices=frame_indices,
            source_npz_path=np.asarray(str(source)),
            coordinate_frame=np.asarray("robot_base_mm"),
        )
        _write_point_ply(directory / "pointcloud_base.ply", points_base, labels)
        center_count = _write_center_ply(directory / "hole_centers_base.ply", payload)
        _write_preview(directory / "pointcloud_preview.jpg", points_base, labels, payload)
        artifacts: dict[str, Any] = {
            "status": "ready",
            "coordinate_frame": "robot_base_mm",
            "raw_npz": raw_name,
            "ply": "pointcloud_base.ply",
            "centers_ply": "hole_centers_base.ply",
            "preview_jpg": "pointcloud_preview.jpg",
            "point_count": int(len(points_base)),
            "frame_count": int(len(set(int(value) for value in frame_indices))),
            "hole_ids": sorted(set(int(value) for value in labels)),
            "center_hole_count": int(center_count),
            "source_npz": str(source),
        }
        if overlay_path is not None and Path(overlay_path).is_file():
            overlay_target = directory / "map_overlay.jpg"
            overlay_image = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
            if overlay_image is not None:
                ok, encoded = cv2.imencode(
                    ".jpg", overlay_image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                )
                if not ok:
                    raise RuntimeError("最终结果叠加图JPG编码失败")
                _atomic_write_bytes(overlay_target, encoded.tobytes())
                artifacts["overlay_jpg"] = "map_overlay.jpg"
        return artifacts
    except Exception as exc:
        result.update({
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
        })
        return result
