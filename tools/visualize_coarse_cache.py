#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""可视化旧两阶段粗定位缓存中的 NPZ 局部点云。

当前缓存的 NPZ 由 manifest.json 提供几何摘要和相机内参，由 NPZ 本身提供
每帧局部点云。这个工具不依赖 PLY，也不修改缓存文件。

示例：

    python tools/visualize_coarse_cache.py ^
        --cache-dir C:\MM\aubo_tools\data\hole_localization_coarse_cache ^
        --hole-id 3 --show

    python tools/visualize_coarse_cache.py ^
        --cache-dir C:\MM\aubo_tools\data\hole_localization_coarse_cache ^
        --hole-id 3 ^
        --rgb C:\MM\aubo_tools\data\hole_localization_runs\...\01_home_selected.png ^
        --output C:\MM\cache_hole_03.png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aubo_workbench.coarse_cache import (  # noqa: E402
    CoarseCacheEntry,
    load_cache_entries,
    load_persistent_cache_entries,
)
from aubo_workbench.paths import HOLE_LOCALIZATION_COARSE_CACHE_DIR  # noqa: E402


@dataclass(frozen=True)
class CacheCloud:
    """可视化所需的单孔缓存数据。"""

    hole_id: int
    points_camera_mm_by_frame: tuple[np.ndarray, ...]
    frame_indices: np.ndarray
    frame_centers_px: np.ndarray
    frame_plane_rmse_mm: np.ndarray
    intrinsics: dict[str, Any] | None
    T_base_camera_build: np.ndarray | None
    point_camera_mm: np.ndarray | None
    plane_point_camera_mm: np.ndarray | None
    normal_camera: np.ndarray | None
    point_base_mm: np.ndarray | None
    plane_point_base_mm: np.ndarray | None
    normal_base: np.ndarray | None
    plane_rmse_mm: float | None
    center_scatter_p95_px: float | None
    source_path: Path
    cache_scope: str | None


def _validate_npz_arrays(
    points: Any,
    offsets: Any,
    frame_indices: Any,
    frame_centers_px: Any,
    frame_plane_rmse_mm: Any,
    source: Path,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray, np.ndarray]:
    points_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    offsets_array = np.asarray(offsets, dtype=np.int64).reshape(-1)
    indices_array = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    centers_array = np.asarray(frame_centers_px, dtype=np.float64).reshape(-1, 2)
    rmse_array = np.asarray(frame_plane_rmse_mm, dtype=np.float64).reshape(-1)

    if not np.isfinite(points_array).all():
        raise ValueError(f"NPZ点云包含非有限值：{source}")
    if (
        len(indices_array) < 1
        or len(offsets_array) != len(indices_array) + 1
        or offsets_array[0] != 0
        or np.any(np.diff(offsets_array) <= 0)
        or int(offsets_array[-1]) != len(points_array)
    ):
        raise ValueError(f"NPZ帧偏移与点数不一致：{source}")
    if (
        len(centers_array) != len(indices_array)
        or len(rmse_array) != len(indices_array)
        or not np.isfinite(centers_array).all()
        or not np.isfinite(rmse_array).all()
    ):
        raise ValueError(f"NPZ帧审计数组不一致：{source}")

    points_by_frame = tuple(
        points_array[int(start):int(end)]
        for start, end in zip(offsets_array[:-1], offsets_array[1:])
    )
    return points_by_frame, indices_array, centers_array, rmse_array


def _cloud_from_entry(
    entry: CoarseCacheEntry,
    source_path: Path,
    cache_scope: str | None,
) -> CacheCloud:
    return CacheCloud(
        hole_id=int(entry.hole_id),
        points_camera_mm_by_frame=tuple(
            np.asarray(item, dtype=np.float32).reshape(-1, 3)
            for item in entry.points_camera_mm_by_frame
        ),
        frame_indices=np.asarray(entry.frame_indices, dtype=np.int64),
        frame_centers_px=np.asarray(entry.frame_centers_px, dtype=np.float64),
        frame_plane_rmse_mm=np.asarray(entry.frame_plane_rmse_mm, dtype=np.float64),
        intrinsics=dict(entry.intrinsics),
        T_base_camera_build=np.asarray(entry.T_base_camera_build, dtype=np.float64),
        point_camera_mm=np.asarray(entry.point_camera_mm, dtype=np.float64),
        plane_point_camera_mm=np.asarray(entry.plane_point_camera_mm, dtype=np.float64),
        normal_camera=np.asarray(entry.normal_camera, dtype=np.float64),
        point_base_mm=np.asarray(entry.point_base_mm, dtype=np.float64),
        plane_point_base_mm=np.asarray(entry.plane_point_base_mm, dtype=np.float64),
        normal_base=np.asarray(entry.normal_base, dtype=np.float64),
        plane_rmse_mm=float(entry.plane_rmse_mm),
        center_scatter_p95_px=float(entry.center_scatter_p95_px),
        source_path=source_path,
        cache_scope=cache_scope,
    )


def _cloud_from_npz(
    npz_path: Path,
    hole_id: int,
    *,
    intrinsics: dict[str, Any] | None = None,
    cache_scope: str | None = None,
    manifest_entry: dict[str, Any] | None = None,
) -> CacheCloud:
    with np.load(npz_path, allow_pickle=False) as payload:
        required = (
            "points_camera_mm",
            "frame_offsets",
            "frame_indices",
            "frame_centers_px",
            "frame_plane_rmse_mm",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"NPZ缺少字段 {missing}：{npz_path}")
        points_by_frame, frame_indices, frame_centers, frame_rmse = _validate_npz_arrays(
            payload["points_camera_mm"],
            payload["frame_offsets"],
            payload["frame_indices"],
            payload["frame_centers_px"],
            payload["frame_plane_rmse_mm"],
            npz_path,
        )

    raw = manifest_entry or {}

    def optional_vector(name: str) -> np.ndarray | None:
        value = raw.get(name)
        return None if value is None else np.asarray(value, dtype=np.float64).reshape(3)

    def optional_matrix(name: str) -> np.ndarray | None:
        value = raw.get(name)
        return None if value is None else np.asarray(value, dtype=np.float64).reshape(4, 4)

    return CacheCloud(
        hole_id=int(hole_id),
        points_camera_mm_by_frame=points_by_frame,
        frame_indices=frame_indices,
        frame_centers_px=frame_centers,
        frame_plane_rmse_mm=frame_rmse,
        intrinsics=None if intrinsics is None else dict(intrinsics),
        T_base_camera_build=optional_matrix("T_base_camera_build"),
        point_camera_mm=optional_vector("point_camera_mm"),
        plane_point_camera_mm=optional_vector("plane_point_camera_mm"),
        normal_camera=optional_vector("normal_camera"),
        point_base_mm=optional_vector("point_base_mm"),
        plane_point_base_mm=optional_vector("plane_point_base_mm"),
        normal_base=optional_vector("normal_base"),
        plane_rmse_mm=(
            None if raw.get("plane_rmse_mm") is None else float(raw["plane_rmse_mm"])
        ),
        center_scatter_p95_px=(
            None
            if raw.get("center_scatter_p95_px") is None
            else float(raw["center_scatter_p95_px"])
        ),
        source_path=npz_path,
        cache_scope=cache_scope,
    )


def resolve_cache_directory(cache_path: str | Path) -> tuple[Path, Path | None]:
    """把目录、manifest 或单个 NPZ 统一解析为缓存目录和可选NPZ路径。"""
    path = Path(cache_path).expanduser()
    if path.is_file() and path.suffix.lower() == ".npz":
        return path.parent, path
    if path.is_file() and path.name.lower() == "manifest.json":
        return path.parent, None
    return path, None


def list_hole_ids(cache_path: str | Path) -> list[int]:
    cache_dir, npz_path = resolve_cache_directory(cache_path)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.is_file():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = raw.get("entries", {})
        if not isinstance(entries, dict):
            raise ValueError(f"缓存manifest的entries无效：{manifest_path}")
        return sorted(int(key) for key in entries)
    if npz_path is not None:
        match = re.search(r"hole_(\d+)", npz_path.stem, re.IGNORECASE)
        return [] if match is None else [int(match.group(1))]
    return sorted(
        int(match.group(1))
        for path in cache_dir.glob("hole_*.npz")
        if (match := re.search(r"hole_(\d+)", path.stem, re.IGNORECASE)) is not None
    )


def load_cache_cloud(cache_path: str | Path, hole_id: int | None = None) -> CacheCloud:
    """读取一个孔的NPZ及manifest；允许直接传入hole_XX.npz。"""
    cache_dir, direct_npz = resolve_cache_directory(cache_path)
    manifest_path = cache_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    available = list_hole_ids(cache_path)
    if hole_id is None:
        if len(available) != 1:
            choices = ", ".join(str(value) for value in available) or "(none)"
            raise ValueError(f"请用--hole-id指定孔号，可选：{choices}")
        hole_id = available[0]
    hole_id = int(hole_id)

    raw_entry = None
    scope = manifest.get("cache_scope")
    if manifest:
        raw_entries = manifest.get("entries", {})
        raw_entry = raw_entries.get(str(hole_id))
        if not isinstance(raw_entry, dict):
            raise KeyError(f"manifest中不存在孔{hole_id}")
        npz_name = str(raw_entry.get("npz", "")).strip()
        if not npz_name or Path(npz_name).name != npz_name:
            raise ValueError(f"manifest中孔{hole_id}的NPZ文件名无效")
        direct_npz = cache_dir / npz_name
        if scope == "base_frame_persistent":
            entries = load_persistent_cache_entries(cache_dir, [hole_id])
        else:
            entries = load_cache_entries(cache_dir, [hole_id])
        entry = entries.get(hole_id)
        if entry is None:
            raise ValueError(f"孔{hole_id}缓存无法读取：{cache_dir}")
        return _cloud_from_entry(entry, direct_npz, str(scope) if scope else None)

    if direct_npz is None:
        direct_npz = cache_dir / f"hole_{hole_id:02d}.npz"
    if not direct_npz.is_file():
        raise FileNotFoundError(f"找不到孔{hole_id}的NPZ：{direct_npz}")
    return _cloud_from_npz(direct_npz, hole_id)


def select_frame_points(
    cloud: CacheCloud,
    frame_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """返回选定帧的点和帧标签；frame_index为None时返回全部帧。"""
    if frame_index is None:
        points = np.concatenate(cloud.points_camera_mm_by_frame, axis=0)
        labels = np.concatenate([
            np.full(len(points_in_frame), int(frame_id), dtype=np.int64)
            for points_in_frame, frame_id in zip(
                cloud.points_camera_mm_by_frame, cloud.frame_indices,
            )
        ])
        return points, labels
    matches = np.flatnonzero(cloud.frame_indices == int(frame_index))
    if len(matches) != 1:
        available = ", ".join(str(int(value)) for value in cloud.frame_indices)
        raise ValueError(f"缓存没有帧{frame_index}，可选：{available}")
    index = int(matches[0])
    return (
        np.asarray(cloud.points_camera_mm_by_frame[index], dtype=np.float32),
        np.full(len(cloud.points_camera_mm_by_frame[index]), int(frame_index), dtype=np.int64),
    )


def transform_points_to_base(points_camera_mm: Any, T_base_camera: Any) -> np.ndarray:
    points = np.asarray(points_camera_mm, dtype=np.float64).reshape(-1, 3)
    transform = np.asarray(T_base_camera, dtype=np.float64).reshape(4, 4)
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_camera_points(
    points_camera_mm: Any,
    intrinsics: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """把相机坐标点投影到RGB图，返回像素和有效深度掩码。"""
    points = np.asarray(points_camera_mm, dtype=np.float64).reshape(-1, 3)
    required = ("fx", "fy", "cx", "cy")
    if any(key not in intrinsics for key in required):
        raise ValueError(f"内参缺少字段：{required}")
    camera_matrix = np.array([
        [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
        [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    distortion_values = intrinsics.get("distortion", intrinsics.get("dist_coeffs", ()))
    distortion = np.asarray(distortion_values, dtype=np.float64).reshape(-1, 1)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)
    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    if np.any(valid):
        projected[valid] = cv2.projectPoints(
            points[valid],
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            camera_matrix,
            distortion if len(distortion) else None,
        )[0].reshape(-1, 2)
    return projected, valid


def _downsample(
    points: np.ndarray,
    labels: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= int(max_points):
        return points, labels
    indices = np.linspace(0, len(points) - 1, int(max_points), dtype=np.int64)
    return points[indices], labels[indices]


def _set_equal_3d(ax: Any, points: np.ndarray) -> None:
    if len(points) == 0:
        return
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) / 2.0, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _frame_colors(labels: np.ndarray, cmap: Any) -> np.ndarray:
    unique = sorted(set(int(value) for value in labels))
    positions = {value: index for index, value in enumerate(unique)}
    denominator = max(1, len(unique) - 1)
    return np.asarray([
        cmap(float(positions[int(value)]) / denominator)
        for value in labels
    ])


def _coordinate_data(
    cloud: CacheCloud,
    points_camera: np.ndarray,
    coordinate_frame: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    mode = str(coordinate_frame).lower()
    if mode == "camera":
        return (
            points_camera,
            cloud.point_camera_mm,
            cloud.normal_camera,
        )
    if mode == "base":
        if cloud.T_base_camera_build is None:
            raise ValueError("manifest没有T_base_camera_build，无法显示基坐标")
        point = (
            None if cloud.point_base_mm is None
            else np.asarray(cloud.point_base_mm, dtype=np.float64)
        )
        normal = (
            None if cloud.normal_base is None
            else np.asarray(cloud.normal_base, dtype=np.float64)
        )
        return transform_points_to_base(points_camera, cloud.T_base_camera_build), point, normal
    raise ValueError(f"未知坐标系：{coordinate_frame!r}")


def render_cache_cloud(
    cloud: CacheCloud,
    *,
    rgb_path: str | Path | None = None,
    frame_index: int | None = None,
    coordinate_frame: str = "camera",
    max_points: int = 25000,
    output_path: str | Path | None = None,
    show: bool = False,
) -> Path | None:
    """绘制三维点云，必要时在右侧叠加RGB投影。"""
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points_camera, labels = select_frame_points(cloud, frame_index)
    points_camera, labels = _downsample(points_camera, labels, int(max_points))
    points_display, anchor, normal = _coordinate_data(
        cloud, points_camera, coordinate_frame,
    )

    image_bgr = None
    if rgb_path is not None:
        image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"RGB图片无法读取：{rgb_path}")

    columns = 2 if image_bgr is not None else 1
    figure = plt.figure(figsize=(14.0 if columns == 2 else 8.5, 7.5))
    axis_3d = figure.add_subplot(1, columns, 1, projection="3d")
    cmap = plt.get_cmap("viridis")
    colors = _frame_colors(labels, cmap)
    axis_3d.scatter(
        points_display[:, 0],
        points_display[:, 1],
        points_display[:, 2],
        s=3.0,
        c=colors,
        alpha=0.75,
        depthshade=False,
    )
    if anchor is not None:
        axis_3d.scatter(
            [anchor[0]], [anchor[1]], [anchor[2]],
            s=50.0, c="red", marker="x", label="cached center",
        )
        if normal is not None:
            arrow_length = max(float(np.ptp(points_display, axis=0).max()) * 0.35, 10.0)
            axis_3d.quiver(
                anchor[0], anchor[1], anchor[2],
                normal[0], normal[1], normal[2],
                length=arrow_length, color="red", linewidth=2.0,
                label="cached normal",
            )
    _set_equal_3d(axis_3d, points_display)
    axis_3d.set_xlabel("X (mm)")
    axis_3d.set_ylabel("Y (mm)")
    axis_3d.set_zlabel("Z (mm)")
    axis_3d.set_title(
        f"Hole {cloud.hole_id:02d} | {coordinate_frame} frame | "
        f"{len(points_camera):,} points",
    )
    axis_3d.view_init(elev=24.0, azim=-62.0)
    axis_3d.legend(loc="best")

    if image_bgr is not None:
        axis_rgb = figure.add_subplot(1, columns, 2)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        axis_rgb.imshow(image_rgb)
        projected, valid = project_camera_points(points_camera, cloud.intrinsics or {})
        height, width = image_rgb.shape[:2]
        visible = (
            valid
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] < float(width))
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < float(height))
        )
        if np.any(visible):
            axis_rgb.scatter(
                projected[visible, 0],
                projected[visible, 1],
                s=3.0,
                c=colors[visible],
                alpha=0.75,
                linewidths=0.0,
            )
        if cloud.frame_centers_px.size:
            centers = np.asarray(cloud.frame_centers_px, dtype=np.float64)
            if frame_index is not None:
                center_rows = np.flatnonzero(cloud.frame_indices == int(frame_index))
                centers = centers[center_rows]
            if len(centers):
                axis_rgb.scatter(
                    centers[:, 0], centers[:, 1],
                    s=42.0, facecolors="none", edgecolors="red",
                    linewidths=1.2, label="cached RGB centers",
                )
        axis_rgb.set_xlim(0, width)
        axis_rgb.set_ylim(height, 0)
        axis_rgb.set_aspect("equal", adjustable="box")
        axis_rgb.set_xlabel("u (pixel)")
        axis_rgb.set_ylabel("v (pixel)")
        axis_rgb.set_title("RGB + projected cached points")
        axis_rgb.legend(loc="best")

    frame_text = "all frames" if frame_index is None else f"frame {frame_index}"
    stats = [
        f"source: {cloud.source_path}",
        f"frames: {len(cloud.frame_indices)} ({frame_text})",
        f"RMSE: {cloud.plane_rmse_mm:.3f} mm"
        if cloud.plane_rmse_mm is not None else "RMSE: n/a",
        f"scatter P95: {cloud.center_scatter_p95_px:.3f} px"
        if cloud.center_scatter_p95_px is not None else "scatter P95: n/a",
    ]
    figure.suptitle(" | ".join(stats), fontsize=9)
    figure.tight_layout()

    saved = None
    if output_path is not None:
        saved = Path(output_path).expanduser()
        saved.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(saved, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    return saved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="查看粗定位缓存NPZ，不需要转换为PLY。",
    )
    parser.add_argument(
        "--cache-dir", "--cache",
        default=str(HOLE_LOCALIZATION_COARSE_CACHE_DIR),
        help="缓存目录、manifest.json或单个hole_XX.npz",
    )
    parser.add_argument("--hole-id", type=int, help="孔号；缓存有多个孔时必须指定")
    parser.add_argument(
        "--rgb",
        help="与缓存采集位姿相同的RGB图片；未指定时自动尝试缓存目录中的初始RGB快照",
    )
    parser.add_argument(
        "--frame",
        type=int,
        help="只显示指定缓存帧；默认显示全部有效帧",
    )
    parser.add_argument(
        "--coordinate-frame",
        choices=("camera", "base"),
        default="camera",
        help="三维视图坐标系，默认camera",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=25000,
        help="最多绘制点数，默认25000；仅影响显示，不修改NPZ",
    )
    parser.add_argument("--output", type=Path, help="保存PNG路径")
    parser.add_argument(
        "--show",
        action="store_true",
        help="打开Matplotlib窗口；不指定时只保存--output",
    )
    parser.add_argument(
        "--list-holes",
        action="store_true",
        help="只列出缓存中的孔号",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_holes:
        for hole_id in list_hole_ids(args.cache_dir):
            print(hole_id)
        return 0
    if not args.show and args.output is None:
        args.show = True
    if args.rgb is None:
        cache_dir, _ = resolve_cache_directory(args.cache_dir)
        snapshots = sorted(cache_dir.glob("initial_cache_rgb_frame_*.png"))
        if snapshots:
            args.rgb = str(snapshots[0])
    cloud = load_cache_cloud(args.cache_dir, args.hole_id)
    saved = render_cache_cloud(
        cloud,
        rgb_path=args.rgb,
        frame_index=args.frame,
        coordinate_frame=args.coordinate_frame,
        max_points=args.max_points,
        output_path=args.output,
        show=args.show,
    )
    if saved is not None:
        print(f"[VISUALIZATION_SAVED] {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
