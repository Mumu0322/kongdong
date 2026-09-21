#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""旧两阶段流程的当前运行局部点云缓存。

缓存只服务于一次连续运行：JSON 保存可审计的几何摘要，NPZ 保存每帧真正
参与局部平面拟合的点。这个模块不依赖机器人、相机或 GUI，便于单元测试。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from aubo_workbench.io_utils import atomic_write_json, make_dir


CACHE_SCHEMA_VERSION = 1
CURRENT_RUN_CACHE_SCOPE = "current_run_only"
PERSISTENT_CACHE_SCOPE = "base_frame_persistent"


@dataclass(frozen=True)
class CacheValidationGates:
    """缓存建立和复用使用的质量门。

    tracking/center/plane 门沿用两阶段流程现有门限；验证固定采集5帧（优化后），
    至少4帧有效。法向门用于比较缓存几何与现场观测的物理一致性。
    """

    validation_frames: int = 5
    min_valid_frames: int = 4
    max_tracking_distance_px: float = 70.0
    max_center_offset_px: float = 5.0
    max_center_scatter_p95_px: float = 0.8
    max_plane_rmse_mm: float = 3.5
    max_normal_error_deg: float = 2.0
    # 新增：自适应匹配距离相关参数
    adaptive_match_distance: bool = True
    min_match_distance_mm: float = 10.0
    max_match_distance_mm: float = 50.0
    match_distance_ratio: float = 0.25  # 最小孔间距的25%

    def __post_init__(self) -> None:
        if int(self.validation_frames) < 1:
            raise ValueError("缓存验证帧数必须大于0")
        if int(self.min_valid_frames) < 1 or int(self.min_valid_frames) > int(self.validation_frames):
            raise ValueError("缓存最少有效帧数必须位于验证帧数范围内")
        if float(self.max_tracking_distance_px) <= 0.0:
            raise ValueError("缓存跟踪距离门限必须大于0")
        if float(self.max_center_offset_px) <= 0.0:
            raise ValueError("缓存中心偏移门限必须大于0")
        if float(self.max_center_scatter_p95_px) <= 0.0:
            raise ValueError("缓存中心散布门限必须大于0")
        if float(self.max_plane_rmse_mm) <= 0.0:
            raise ValueError("缓存平面RMSE门限必须大于0")
        if float(self.max_normal_error_deg) <= 0.0:
            raise ValueError("缓存法向门限必须大于0")
        if float(self.min_match_distance_mm) <= 0.0:
            raise ValueError("缓存最小匹配距离必须大于0")
        if float(self.max_match_distance_mm) < float(self.min_match_distance_mm):
            raise ValueError("缓存最大匹配距离必须不小于最小匹配距离")
        if float(self.match_distance_ratio) <= 0.0:
            raise ValueError("缓存匹配距离比例必须大于0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_frames": int(self.validation_frames),
            "min_valid_frames": int(self.min_valid_frames),
            "max_tracking_distance_px": float(self.max_tracking_distance_px),
            "max_center_offset_px": float(self.max_center_offset_px),
            "max_center_scatter_p95_px": float(self.max_center_scatter_p95_px),
            "max_plane_rmse_mm": float(self.max_plane_rmse_mm),
            "max_normal_error_deg": float(self.max_normal_error_deg),
            "adaptive_match_distance": bool(self.adaptive_match_distance),
            "min_match_distance_mm": float(self.min_match_distance_mm),
            "max_match_distance_mm": float(self.max_match_distance_mm),
            "match_distance_ratio": float(self.match_distance_ratio),
        }


@dataclass(frozen=True)
class CacheExpiryPolicy:
    """缓存过期策略配置"""

    max_age_hours: float = 72.0      # 3天后自动失效
    warn_age_hours: float = 24.0     # 超过1天给警告
    enabled: bool = True              # 是否启用过期检查

    def __post_init__(self) -> None:
        if float(self.max_age_hours) <= 0.0:
            raise ValueError("缓存最大年龄必须大于0")
        if float(self.warn_age_hours) < 0.0:
            raise ValueError("缓存警告年龄不能为负")
        if float(self.warn_age_hours) > float(self.max_age_hours):
            raise ValueError("缓存警告年龄不能超过最大年龄")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_age_hours": float(self.max_age_hours),
            "warn_age_hours": float(self.warn_age_hours),
            "enabled": bool(self.enabled),
        }


@dataclass
class CoarseCacheEntry:
    """一个孔的局部点云缓存条目。所有三维量单位均为 mm。"""

    hole_id: int
    T_base_camera_build: np.ndarray
    T_tcp_camera: np.ndarray
    tcp_pose_m_rad: list[float]
    camera_serial: str
    handeye_path: str
    intrinsics: dict[str, Any]
    center_px: np.ndarray
    point_camera_mm: np.ndarray
    plane_point_camera_mm: np.ndarray
    normal_camera: np.ndarray
    point_base_mm: np.ndarray
    plane_point_base_mm: np.ndarray
    normal_base: np.ndarray
    plane_rmse_mm: float
    valid_frames: int
    total_frames: int
    center_scatter_p95_px: float
    ring_points_median: float
    surface_model: str
    frame_indices: np.ndarray
    frame_centers_px: np.ndarray
    frame_plane_rmse_mm: np.ndarray
    points_camera_mm_by_frame: tuple[np.ndarray, ...]
    created_at: str = ""
    last_validated_at: str = ""  # 新增：最后验证时间
    validation_count: int = 0     # 新增：成功验证次数
    # 缓存来源。正式可复用缓存必须来自340 mm一拍多批量粗定位；
    # 旧缓存没有该字段时按unknown处理，由运行层拒绝复用。
    cache_source: str = "unknown"

    def __post_init__(self) -> None:
        self.hole_id = int(self.hole_id)
        self.T_base_camera_build = _finite_matrix(self.T_base_camera_build, "T_base_camera_build")
        self.T_tcp_camera = _finite_matrix(self.T_tcp_camera, "T_tcp_camera")
        self.center_px = _finite_vector(self.center_px, 2, "center_px")
        self.point_camera_mm = _finite_vector(self.point_camera_mm, 3, "point_camera_mm")
        self.plane_point_camera_mm = _finite_vector(
            self.plane_point_camera_mm, 3, "plane_point_camera_mm",
        )
        self.normal_camera = _unit(self.normal_camera, "normal_camera")
        self.point_base_mm = _finite_vector(self.point_base_mm, 3, "point_base_mm")
        self.plane_point_base_mm = _finite_vector(
            self.plane_point_base_mm, 3, "plane_point_base_mm",
        )
        self.normal_base = _unit(self.normal_base, "normal_base")
        self.frame_indices = np.asarray(self.frame_indices, dtype=np.int64).reshape(-1)
        self.frame_centers_px = np.asarray(self.frame_centers_px, dtype=np.float64).reshape(-1, 2)
        self.frame_plane_rmse_mm = np.asarray(self.frame_plane_rmse_mm, dtype=np.float64).reshape(-1)
        if not np.isfinite(self.frame_centers_px).all():
            raise ValueError("缓存帧中心包含非有限数值")
        if not np.isfinite(self.frame_plane_rmse_mm).all() or np.any(self.frame_plane_rmse_mm < 0.0):
            raise ValueError("缓存帧RMSE包含无效数值")
        if len(self.frame_indices) != len(self.frame_centers_px):
            raise ValueError("缓存帧索引与帧中心数量不一致")
        if len(self.frame_indices) != len(self.frame_plane_rmse_mm):
            raise ValueError("缓存帧索引与帧RMSE数量不一致")
        if len(self.points_camera_mm_by_frame) != len(self.frame_indices):
            raise ValueError("缓存帧索引与点云帧数量不一致")
        points: list[np.ndarray] = []
        for index, values in enumerate(self.points_camera_mm_by_frame):
            item = np.asarray(values, dtype=np.float32).reshape(-1, 3)
            if len(item) < 1 or not np.isfinite(item).all():
                raise ValueError(f"缓存第{index}帧点云无效")
            points.append(item)
        self.points_camera_mm_by_frame = tuple(points)
        self.tcp_pose_m_rad = [float(value) for value in self.tcp_pose_m_rad]
        if len(self.tcp_pose_m_rad) != 6 or not np.isfinite(self.tcp_pose_m_rad).all():
            raise ValueError("缓存TCP位姿必须是6个有限数值")
        if int(self.valid_frames) != len(self.frame_indices):
            raise ValueError("缓存有效帧数与帧数据数量不一致")
        if int(self.valid_frames) < 1 or int(self.total_frames) < int(self.valid_frames):
            raise ValueError("缓存帧统计无效")
        if not isinstance(self.intrinsics, Mapping):
            raise ValueError("缓存内参必须是JSON对象")
        for key in ("width", "height", "fx", "fy", "cx", "cy"):
            if key not in self.intrinsics or not math.isfinite(float(self.intrinsics[key])):
                raise ValueError(f"缓存内参缺少有效字段：{key}")
        for name, value in (
            ("plane_rmse_mm", self.plane_rmse_mm),
            ("center_scatter_p95_px", self.center_scatter_p95_px),
            ("ring_points_median", self.ring_points_median),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"缓存{name}无效")
        if not self.created_at:
            self.created_at = datetime.now().isoformat(timespec="seconds")
        if int(self.validation_count) < 0:
            raise ValueError("缓存验证计数不能为负")

    def manifest_dict(self, npz_name: str) -> dict[str, Any]:
        return {
            "hole_id": self.hole_id,
            "npz": npz_name,
            "created_at": self.created_at,
            "last_validated_at": self.last_validated_at,
            "validation_count": int(self.validation_count),
            "cache_source": str(self.cache_source),
            "camera_serial": self.camera_serial,
            "handeye_path": self.handeye_path,
            "intrinsics": dict(self.intrinsics),
            "T_base_camera_build": self.T_base_camera_build.tolist(),
            "T_tcp_camera": self.T_tcp_camera.tolist(),
            "tcp_pose_m_rad": list(self.tcp_pose_m_rad),
            "center_px": self.center_px.tolist(),
            "point_camera_mm": self.point_camera_mm.tolist(),
            "plane_point_camera_mm": self.plane_point_camera_mm.tolist(),
            "normal_camera": self.normal_camera.tolist(),
            "point_base_mm": self.point_base_mm.tolist(),
            "plane_point_base_mm": self.plane_point_base_mm.tolist(),
            "normal_base": self.normal_base.tolist(),
            "plane_rmse_mm": float(self.plane_rmse_mm),
            "valid_frames": int(self.valid_frames),
            "total_frames": int(self.total_frames),
            "center_scatter_p95_px": float(self.center_scatter_p95_px),
            "ring_points_median": float(self.ring_points_median),
            "surface_model": self.surface_model,
            "frame_indices": self.frame_indices.tolist(),
            "frame_centers_px": self.frame_centers_px.tolist(),
            "frame_plane_rmse_mm": self.frame_plane_rmse_mm.tolist(),
        }


@dataclass(frozen=True)
class CacheValidationResult:
    accepted: bool
    reason: str
    valid_frames: int
    total_frames: int
    max_tracking_distance_px: float | None
    center_offset_px: float | None
    center_scatter_p95_px: float | None
    max_plane_rmse_mm: float | None
    max_normal_error_deg: float | None
    point_delta_mm: float | None
    current_point_base_mm: np.ndarray | None
    current_plane_point_base_mm: np.ndarray | None
    current_normal_base: np.ndarray | None
    current_center_px: np.ndarray | None
    current_point_camera_mm: np.ndarray | None
    current_plane_point_camera_mm: np.ndarray | None
    current_normal_camera: np.ndarray | None
    freshness_status: str = "unknown"  # 新增：fresh/stale/expired
    cache_age_hours: float | None = None  # 新增：缓存年龄

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "valid_frames": int(self.valid_frames),
            "total_frames": int(self.total_frames),
            "max_tracking_distance_px": self.max_tracking_distance_px,
            "center_offset_px": self.center_offset_px,
            "center_scatter_p95_px": self.center_scatter_p95_px,
            "max_plane_rmse_mm": self.max_plane_rmse_mm,
            "max_normal_error_deg": self.max_normal_error_deg,
            "point_delta_mm": self.point_delta_mm,
            "current_point_base_mm": _list_or_none(self.current_point_base_mm),
            "current_plane_point_base_mm": _list_or_none(self.current_plane_point_base_mm),
            "current_normal_base": _list_or_none(self.current_normal_base),
            "current_center_px": _list_or_none(self.current_center_px),
            "current_point_camera_mm": _list_or_none(self.current_point_camera_mm),
            "current_plane_point_camera_mm": _list_or_none(self.current_plane_point_camera_mm),
            "current_normal_camera": _list_or_none(self.current_normal_camera),
            "freshness_status": self.freshness_status,
            "cache_age_hours": self.cache_age_hours,
        }


def save_cache_entries(
    cache_dir: str | Path,
    entries: Mapping[int, CoarseCacheEntry],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """保存当前运行的全部缓存条目，并原子替换 manifest。"""

    return _save_cache_entries(
        cache_dir, entries, metadata=metadata, cache_scope=CURRENT_RUN_CACHE_SCOPE,
    )


def save_persistent_cache_entries(
    cache_dir: str | Path,
    entries: Mapping[int, CoarseCacheEntry],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """保存可跨运行复用的 base 坐标缓存。"""

    return _save_cache_entries(
        cache_dir, entries, metadata=metadata, cache_scope=PERSISTENT_CACHE_SCOPE,
    )


def _save_cache_entries(
    cache_dir: str | Path,
    entries: Mapping[int, CoarseCacheEntry],
    *,
    metadata: Mapping[str, Any] | None,
    cache_scope: str,
) -> Path:
    directory = Path(cache_dir)
    make_dir(directory)
    manifest_entries: dict[str, Any] = {}
    for hole_id, entry in sorted(entries.items(), key=lambda item: int(item[0])):
        if int(hole_id) != int(entry.hole_id):
            raise ValueError("缓存字典键与孔号不一致")
        npz_name = f"hole_{int(entry.hole_id):02d}.npz"
        _write_npz_atomically(directory / npz_name, entry)
        manifest_entries[str(int(entry.hole_id))] = entry.manifest_dict(npz_name)
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_scope": str(cache_scope),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": dict(metadata or {}),
        "entries": manifest_entries,
    }
    return atomic_write_json(directory / "manifest.json", manifest)


def load_cache_entries(
    cache_dir: str | Path,
    hole_ids: Sequence[int] | None = None,
) -> dict[int, CoarseCacheEntry]:
    """读取一个当前运行目录的缓存；任一选中的条目损坏时显式失败。

    ``hole_ids`` 允许逐孔读取，避免一个孔的坏文件影响其它孔的自动回退。
    未传入时仍校验并读取 manifest 中的全部条目。
    """

    return _load_cache_entries(
        cache_dir, hole_ids, expected_scope=CURRENT_RUN_CACHE_SCOPE,
    )


def load_persistent_cache_entries(
    cache_dir: str | Path,
    hole_ids: Sequence[int] | None = None,
    *,
    errors: dict[int, str] | None = None,
) -> dict[int, CoarseCacheEntry]:
    """读取跨运行缓存；单个损坏条目只被跳过，其它孔仍可继续验证。"""

    return _load_cache_entries(
        cache_dir, hole_ids, expected_scope=PERSISTENT_CACHE_SCOPE,
        skip_invalid=True, errors=errors,
    )


def _load_cache_entries(
    cache_dir: str | Path,
    hole_ids: Sequence[int] | None,
    *,
    expected_scope: str,
    skip_invalid: bool = False,
    errors: dict[int, str] | None = None,
) -> dict[int, CoarseCacheEntry]:
    directory = Path(cache_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缓存manifest不存在：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise ValueError("缓存schema版本不兼容")
    if manifest.get("cache_scope") != str(expected_scope):
        raise ValueError(f"缓存范围不是{expected_scope}")
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, Mapping):
        raise ValueError("缓存manifest缺少entries")
    selected_keys = None if hole_ids is None else {str(int(value)) for value in hole_ids}
    result: dict[int, CoarseCacheEntry] = {}
    for raw_hole_id, raw in raw_entries.items():
        if selected_keys is not None and str(raw_hole_id) not in selected_keys:
            continue
        try:
            if not isinstance(raw, Mapping):
                raise ValueError(f"孔{raw_hole_id}缓存摘要无效")
            hole_id = int(raw.get("hole_id", raw_hole_id))
            if hole_id != int(raw_hole_id):
                raise ValueError(f"缓存manifest孔号不一致：key={raw_hole_id}, hole_id={hole_id}")
            npz_name = str(raw.get("npz", "")).strip()
            if not npz_name or Path(npz_name).name != npz_name:
                raise ValueError(f"孔{hole_id}缓存NPZ路径无效")
            result[hole_id] = _entry_from_manifest(raw, directory / npz_name)
        except Exception as exc:
            if not skip_invalid:
                raise
            try:
                error_hole_id = int(raw.get("hole_id", raw_hole_id)) if isinstance(raw, Mapping) else int(raw_hole_id)
            except (TypeError, ValueError):
                error_hole_id = -1
            if errors is not None:
                errors[error_hole_id] = f"{type(exc).__name__}:{exc}"
    return result


def cache_entry_compatibility_reasons(
    entry: CoarseCacheEntry,
    *,
    camera_serial: str,
    T_tcp_camera: np.ndarray,
    intrinsics: Any,
    handeye_path: str | None = None,
    max_handeye_translation_delta_mm: float = 0.5,
    max_handeye_rotation_delta_deg: float = 0.10,
    max_intrinsics_delta_px: float = 0.5,
) -> list[str]:
    """检查历史缓存是否仍属于当前相机、手眼和内参。"""

    reasons: list[str] = []
    stored_serial = str(entry.camera_serial).strip()
    current_serial = str(camera_serial).strip()
    if not stored_serial or not current_serial:
        reasons.append("camera_serial_unverified")
    elif stored_serial != current_serial:
        reasons.append("camera_serial_mismatch")

    stored_handeye_path = _canonical_path(entry.handeye_path)
    current_handeye_path = _canonical_path(handeye_path or "")
    if not stored_handeye_path or not current_handeye_path:
        reasons.append("handeye_path_unverified")
    elif stored_handeye_path != current_handeye_path:
        reasons.append("handeye_path_mismatch")

    stored_T = np.asarray(entry.T_tcp_camera, dtype=np.float64).reshape(4, 4)
    current_T = _finite_matrix(T_tcp_camera, "current T_tcp_camera")
    translation_delta = float(np.linalg.norm(stored_T[:3, 3] - current_T[:3, 3]))
    rotation_delta = _rotation_delta_deg(stored_T[:3, :3], current_T[:3, :3])
    if translation_delta > float(max_handeye_translation_delta_mm):
        reasons.append("handeye_translation_mismatch")
    if rotation_delta > float(max_handeye_rotation_delta_deg):
        reasons.append("handeye_rotation_mismatch")

    current_intrinsics = _intrinsics_dict(intrinsics)
    for key in ("width", "height"):
        if int(entry.intrinsics.get(key, -1)) != int(current_intrinsics.get(key, -2)):
            reasons.append(f"intrinsics_{key}_mismatch")
    for key in ("fx", "fy", "cx", "cy"):
        try:
            delta = abs(float(entry.intrinsics[key]) - float(current_intrinsics[key]))
        except (KeyError, TypeError, ValueError):
            delta = math.inf
        if delta > float(max_intrinsics_delta_px):
            reasons.append(f"intrinsics_{key}_mismatch")
    stored_distortion = np.asarray(entry.intrinsics.get("distortion", []), dtype=np.float64).reshape(-1)
    current_distortion = np.asarray(current_intrinsics.get("distortion", []), dtype=np.float64).reshape(-1)
    if stored_distortion.shape != current_distortion.shape:
        reasons.append("intrinsics_distortion_mismatch")
    elif len(stored_distortion) and float(np.max(np.abs(stored_distortion - current_distortion))) > 1.0e-6:
        reasons.append("intrinsics_distortion_mismatch")
    return reasons


def rekey_cache_entry(entry: CoarseCacheEntry, hole_id: int) -> CoarseCacheEntry:
    """复制缓存条目并替换运行内孔号，保留其 base 坐标几何。"""

    from dataclasses import replace as dataclass_replace

    return dataclass_replace(entry, hole_id=int(hole_id))


def transform_cached_points_to_camera(
    entry: CoarseCacheEntry,
    current_T_base_camera: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """把缓存采集时的每帧局部点云从 build 相机变换到当前相机坐标。"""

    T_build = np.asarray(entry.T_base_camera_build, dtype=np.float64).reshape(4, 4)
    T_current = _finite_matrix(current_T_base_camera, "current_T_base_camera")
    T_camera_current_base = np.linalg.inv(T_current)
    transformed: list[np.ndarray] = []
    for points_camera in entry.points_camera_mm_by_frame:
        points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
        homogeneous = np.concatenate(
            [points, np.ones((len(points), 1), dtype=np.float64)], axis=1,
        )
        points_base = (T_build @ homogeneous.T).T[:, :3]
        points_current = (T_camera_current_base @ np.concatenate(
            [points_base, np.ones((len(points_base), 1), dtype=np.float64)], axis=1,
        ).T).T[:, :3]
        if not np.isfinite(points_current).all():
            raise ValueError("缓存点云变换后包含非有限数值")
        transformed.append(points_current.astype(np.float32))
    return tuple(transformed)


def compute_adaptive_match_distance(
    target_points_base_mm: Mapping[int, Any],
    gates: CacheValidationGates,
) -> float:
    """根据孔阵列密度自适应计算匹配距离。

    对于密集孔阵列，使用更小的匹配距离避免误匹配；
    对于稀疏孔阵列，使用更大的匹配距离容忍微小偏移。
    """
    if not gates.adaptive_match_distance or len(target_points_base_mm) < 2:
        return float(gates.max_match_distance_mm)

    points = np.array([
        _finite_vector(point, 3, f"hole {hole_id} base point")[:2]
        for hole_id, point in target_points_base_mm.items()
    ])

    # 计算所有点对的XY平面距离
    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = float(np.linalg.norm(points[i] - points[j]))
            distances.append(dist)

    if not distances:
        return float(gates.max_match_distance_mm)

    min_spacing = float(np.min(distances))

    # 匹配距离 = 最小孔间距 × 比例系数，限制在[min, max]范围内
    adaptive_distance = min_spacing * float(gates.match_distance_ratio)
    result = np.clip(
        adaptive_distance,
        float(gates.min_match_distance_mm),
        float(gates.max_match_distance_mm),
    )
    return float(result)


def check_cache_freshness(
    entry: CoarseCacheEntry,
    policy: CacheExpiryPolicy | None = None,
) -> tuple[str, float]:
    """检查缓存新鲜度。

    返回: (状态, 年龄小时数)
    - "fresh": 在警告期内
    - "stale": 超过警告期但未过期
    - "expired": 已过期
    - "unknown": 无法解析时间戳
    """
    policy = policy or CacheExpiryPolicy()

    if not policy.enabled:
        return "fresh", 0.0

    if not entry.created_at:
        return "unknown", 0.0

    try:
        created = datetime.fromisoformat(entry.created_at)
        age_hours = (datetime.now() - created).total_seconds() / 3600.0

        if age_hours > float(policy.max_age_hours):
            return "expired", age_hours
        elif age_hours > float(policy.warn_age_hours):
            return "stale", age_hours
        else:
            return "fresh", age_hours
    except (ValueError, TypeError):
        return "unknown", 0.0


def match_entries_by_base_point(
    target_points_base_mm: Mapping[int, Any],
    entries: Mapping[int, CoarseCacheEntry],
    *,
    max_match_distance_mm: float = 30.0,
    min_match_margin_mm: float = 5.0,
    gates: CacheValidationGates | None = None,
    expiry_policy: CacheExpiryPolicy | None = None,
) -> tuple[dict[int, CoarseCacheEntry], dict[int, int], dict[int, dict[str, Any]]]:
    """按 base 坐标 XY 将本次选孔与历史缓存做一对一关联。

    新增：支持自适应匹配距离和过期检查。
    """
    # 使用自适应匹配距离（如果启用）
    if gates is not None and gates.adaptive_match_distance:
        effective_match_distance = compute_adaptive_match_distance(target_points_base_mm, gates)
    else:
        effective_match_distance = float(max_match_distance_mm)

    if float(effective_match_distance) <= 0.0 or float(min_match_margin_mm) < 0.0:
        raise ValueError("缓存世界坐标匹配门限无效")

    # 过滤过期条目
    valid_entries: dict[int, CoarseCacheEntry] = {}
    expired_keys: list[int] = []
    for key, entry in entries.items():
        freshness, age_hours = check_cache_freshness(entry, expiry_policy)
        if freshness == "expired":
            expired_keys.append(int(key))
        else:
            valid_entries[int(key)] = entry

    normalized_targets = {
        int(hole_id): _finite_vector(point, 3, f"hole {hole_id} base point")
        for hole_id, point in target_points_base_mm.items()
    }
    normalized_entries = {
        int(key): entry for key, entry in valid_entries.items()
    }
    distances: dict[int, list[tuple[float, int]]] = {}
    for hole_id, point in normalized_targets.items():
        candidates = sorted(
            (
                float(np.linalg.norm(point[:2] - np.asarray(entry.point_base_mm, dtype=np.float64)[:2])),
                int(key),
            )
            for key, entry in normalized_entries.items()
        )
        distances[hole_id] = candidates

    # 先处理候选最少、最近距离最小的孔，减少密集孔阵列的贪心冲突。
    order = sorted(
        normalized_targets,
        key=lambda hole_id: (
            len([item for item in distances[hole_id] if item[0] <= float(effective_match_distance)]),
            distances[hole_id][0][0] if distances[hole_id] else math.inf,
            hole_id,
        ),
    )
    matched: dict[int, CoarseCacheEntry] = {}
    source_ids: dict[int, int] = {}
    audit: dict[int, dict[str, Any]] = {}
    used: set[int] = set()
    for hole_id in order:
        candidates = distances[hole_id]
        if not candidates:
            audit[hole_id] = {
                "matched": False,
                "reason": "no_persistent_entries",
                "adaptive_match_distance_mm": effective_match_distance,
            }
            continue
        nearest_distance, _ = candidates[0]
        available = [item for item in candidates if item[1] not in used]
        in_range = [item for item in available if item[0] <= float(effective_match_distance)]
        if not available:
            audit[hole_id] = {
                "matched": False, "reason": "persistent_entry_already_matched",
                "nearest_distance_xy_mm": nearest_distance,
                "adaptive_match_distance_mm": effective_match_distance,
            }
            continue
        nearest_distance, _ = available[0]
        if nearest_distance > float(effective_match_distance):
            audit[hole_id] = {
                "matched": False, "reason": "world_distance",
                "nearest_distance_xy_mm": nearest_distance,
                "max_match_distance_xy_mm": float(effective_match_distance),
                "adaptive_match_distance_mm": effective_match_distance,
            }
            continue
        if len(in_range) > 1 and in_range[1][0] - nearest_distance < float(min_match_margin_mm):
            audit[hole_id] = {
                "matched": False, "reason": "ambiguous_world_match",
                "nearest_distance_xy_mm": nearest_distance,
                "second_distance_xy_mm": in_range[1][0],
                "min_match_margin_mm": float(min_match_margin_mm),
                "adaptive_match_distance_mm": effective_match_distance,
            }
            continue
        selected_distance, selected_key = available[0]
        selected_entry = normalized_entries[int(selected_key)]
        freshness, age_hours = check_cache_freshness(selected_entry, expiry_policy)
        used.add(int(selected_key))
        matched[hole_id] = selected_entry
        source_ids[hole_id] = int(selected_key)
        audit[hole_id] = {
            "matched": True,
            "persistent_hole_id": int(selected_key),
            "distance_xy_mm": float(selected_distance),
            "max_match_distance_xy_mm": float(effective_match_distance),
            "adaptive_match_distance_mm": effective_match_distance,
            "min_match_margin_mm": float(min_match_margin_mm),
            "cache_freshness": freshness,
            "cache_age_hours": age_hours,
        }

    # 记录被过滤的过期条目
    if expired_keys:
        for key in expired_keys:
            audit[-(int(key) + 1000)] = {  # 使用负数ID避免冲突
                "matched": False,
                "reason": "cache_expired",
                "persistent_hole_id": int(key),
            }

    return matched, source_ids, audit


def validate_cache_entry(
    entry: CoarseCacheEntry,
    measurements: Sequence[Mapping[str, Any]],
    *,
    current_T_base_camera: np.ndarray,
    intrinsics: Any,
    gates: CacheValidationGates,
    expiry_policy: CacheExpiryPolicy | None = None,
) -> CacheValidationResult:
    """用当前340mm少量深度观测验证缓存。

    measurement 至少包含 center_px、plane_point_camera_mm、normal_camera、
    plane_rmse_mm、tracking_distance_px 和 error；surface_plane_point_camera_mm
    可选，缺失时退回孔中心平面交点。
    """
    # 检查缓存新鲜度
    freshness_status, cache_age_hours = check_cache_freshness(entry, expiry_policy)

    total = len(measurements)
    valid: list[Mapping[str, Any]] = []
    tracking_values: list[float] = []
    rejected_reasons: list[str] = []
    for item in measurements:
        error = item.get("error")
        tracking = _finite_scalar(item.get("tracking_distance_px"))
        if tracking is not None:
            tracking_values.append(tracking)
        if error:
            rejected_reasons.append(str(error))
            continue
        if tracking is None or tracking > float(gates.max_tracking_distance_px):
            rejected_reasons.append("tracking_distance")
            continue
        rmse = _finite_scalar(item.get("plane_rmse_mm"))
        if rmse is None or rmse > float(gates.max_plane_rmse_mm):
            rejected_reasons.append("plane_quality")
            continue
        try:
            center = _finite_vector(item.get("center_px"), 2, "measurement center_px")
            point_camera = _finite_vector(
                item.get("plane_point_camera_mm"), 3, "measurement point_camera_mm",
            )
            normal_camera = _unit(item.get("normal_camera"), "measurement normal_camera")
        except (TypeError, ValueError) as exc:
            rejected_reasons.append(f"measurement_invalid:{exc}")
            continue
        valid.append({
            **item,
            "center_px": center,
            "plane_point_camera_mm": point_camera,
            "normal_camera": normal_camera,
            "plane_rmse_mm": rmse,
        })

    if total != int(gates.validation_frames):
        return _rejected(
            f"validation_frames:{total}/{int(gates.validation_frames)}",
            total, valid, tracking_values, freshness_status, cache_age_hours,
        )
    if len(valid) < int(gates.min_valid_frames):
        reason = f"valid_frames:{len(valid)}/{int(gates.min_valid_frames)}"
        if rejected_reasons:
            reason += ";" + ";".join(sorted(set(rejected_reasons)))
        return _rejected(
            reason, total, valid, tracking_values, freshness_status, cache_age_hours,
        )

    centers = np.asarray([item["center_px"] for item in valid], dtype=np.float64)
    center = np.median(centers, axis=0)
    scatter = np.linalg.norm(centers - center, axis=1)
    center_scatter = float(np.percentile(scatter, 95.0))
    center_offset = float(np.linalg.norm(
        center - np.asarray([float(intrinsics.cx), float(intrinsics.cy)], dtype=np.float64),
    ))
    rmse_values = [float(item["plane_rmse_mm"]) for item in valid]
    max_rmse = max(rmse_values)

    R_current = np.asarray(current_T_base_camera, dtype=np.float64).reshape(4, 4)[:3, :3]
    t_current = np.asarray(current_T_base_camera, dtype=np.float64).reshape(4, 4)[:3, 3]
    camera_origin = t_current.copy()
    current_points_base = [R_current @ item["plane_point_camera_mm"] + t_current for item in valid]
    current_point_base = np.median(np.asarray(current_points_base), axis=0)
    surface_points_camera = [
        _finite_vector(
            item.get("surface_plane_point_camera_mm", item["plane_point_camera_mm"]),
            3,
            "surface_plane_point_camera_mm",
        )
        for item in valid
    ]
    current_surface_points_base = [R_current @ item + t_current for item in surface_points_camera]
    current_plane_point_base = np.median(np.asarray(current_surface_points_base), axis=0)

    current_normals_base: list[np.ndarray] = []
    normal_errors: list[float] = []
    for item, point_base in zip(valid, current_points_base):
        normal_base = _unit(R_current @ item["normal_camera"], "current normal_base")
        if float(normal_base @ (camera_origin - point_base)) < 0.0:
            normal_base = -normal_base
        current_normals_base.append(normal_base)
        normal_errors.append(_angle_deg(normal_base, entry.normal_base))
    current_normal_base = _fuse_normals_robust(current_normals_base)
    current_normal_camera = _fuse_normals_robust([item["normal_camera"] for item in valid])
    max_normal_error = max(normal_errors)
    point_delta = float(np.linalg.norm(current_point_base - entry.point_base_mm))

    failure_reasons: list[str] = []
    valid_tracking_values = [
        float(item["tracking_distance_px"])
        for item in valid
        if _finite_scalar(item.get("tracking_distance_px")) is not None
    ]
    max_tracking = max(valid_tracking_values) if valid_tracking_values else None
    if max_tracking is None or max_tracking > float(gates.max_tracking_distance_px):
        failure_reasons.append("tracking_distance")
    if center_offset > float(gates.max_center_offset_px):
        failure_reasons.append("center_offset")
    if center_scatter > float(gates.max_center_scatter_p95_px):
        failure_reasons.append("center_scatter")
    if max_rmse > float(gates.max_plane_rmse_mm):
        failure_reasons.append("plane_quality")
    if max_normal_error > float(gates.max_normal_error_deg):
        failure_reasons.append("normal_error")

    # 过期缓存自动拒绝
    if freshness_status == "expired":
        failure_reasons.append("cache_expired")

    return CacheValidationResult(
        accepted=not failure_reasons,
        reason="ok" if not failure_reasons else ";".join(failure_reasons),
        valid_frames=len(valid),
        total_frames=total,
        max_tracking_distance_px=max_tracking,
        center_offset_px=center_offset,
        center_scatter_p95_px=center_scatter,
        max_plane_rmse_mm=max_rmse,
        max_normal_error_deg=max_normal_error,
        point_delta_mm=point_delta,
        current_point_base_mm=current_point_base,
        current_plane_point_base_mm=current_plane_point_base,
        current_normal_base=current_normal_base,
        current_center_px=center,
        current_point_camera_mm=np.median(
            np.asarray([item["plane_point_camera_mm"] for item in valid]), axis=0,
        ),
        current_plane_point_camera_mm=np.median(np.asarray(surface_points_camera), axis=0),
        current_normal_camera=current_normal_camera,
        freshness_status=freshness_status,
        cache_age_hours=cache_age_hours,
    )


def replace_base_z(cached_point_base_mm: Any, current_point_base_mm: Any) -> np.ndarray:
    """保留缓存XY，仅用现场测量更新基坐标Z。"""

    cached = _finite_vector(cached_point_base_mm, 3, "cached_point_base_mm")
    current = _finite_vector(current_point_base_mm, 3, "current_point_base_mm")
    result = cached.copy()
    result[2] = current[2]
    return result


def _write_npz_atomically(path: Path, entry: CoarseCacheEntry) -> None:
    points: list[np.ndarray] = []
    offsets = [0]
    for values in entry.points_camera_mm_by_frame:
        item = np.asarray(values, dtype=np.float32).reshape(-1, 3)
        points.append(item)
        offsets.append(offsets[-1] + len(item))
    concatenated = np.concatenate(points, axis=0)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                points_camera_mm=concatenated,
                frame_offsets=np.asarray(offsets, dtype=np.int64),
                frame_indices=entry.frame_indices,
                frame_centers_px=entry.frame_centers_px,
                frame_plane_rmse_mm=entry.frame_plane_rmse_mm,
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _entry_from_manifest(raw: Mapping[str, Any], npz_path: Path) -> CoarseCacheEntry:
    if not npz_path.is_file():
        raise FileNotFoundError(f"缓存NPZ不存在：{npz_path}")
    with np.load(npz_path, allow_pickle=False) as payload:
        points = np.asarray(payload["points_camera_mm"], dtype=np.float32).reshape(-1, 3)
        offsets = np.asarray(payload["frame_offsets"], dtype=np.int64).reshape(-1)
        frame_indices = np.asarray(payload["frame_indices"], dtype=np.int64).reshape(-1)
        frame_centers = np.asarray(payload["frame_centers_px"], dtype=np.float64).reshape(-1, 2)
        frame_rmse = np.asarray(payload["frame_plane_rmse_mm"], dtype=np.float64).reshape(-1)
    if (
        len(offsets) != len(frame_indices) + 1
        or len(offsets) < 2
        or offsets[0] != 0
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError(f"缓存NPZ帧偏移无效：{npz_path}")
    if int(offsets[-1]) != len(points):
        raise ValueError(f"缓存NPZ点数与帧偏移不一致：{npz_path}")
    manifest_indices = np.asarray(raw.get("frame_indices", []), dtype=np.int64).reshape(-1)
    manifest_centers_raw = np.asarray(raw.get("frame_centers_px", []), dtype=np.float64)
    manifest_rmse = np.asarray(raw.get("frame_plane_rmse_mm", []), dtype=np.float64).reshape(-1)
    if manifest_centers_raw.size % 2 != 0:
        raise ValueError(f"缓存manifest帧中心维度无效：{npz_path}")
    manifest_centers = manifest_centers_raw.reshape(-1, 2)
    if not (
        np.array_equal(manifest_indices, frame_indices)
        and np.allclose(manifest_centers, frame_centers, rtol=0.0, atol=1e-9)
        and np.allclose(manifest_rmse, frame_rmse, rtol=0.0, atol=1e-9)
    ):
        raise ValueError(f"缓存manifest与NPZ帧审计数据不一致：{npz_path}")
    points_by_frame = tuple(points[int(start):int(end)] for start, end in zip(offsets[:-1], offsets[1:]))
    return CoarseCacheEntry(
        hole_id=int(raw["hole_id"]),
        T_base_camera_build=np.asarray(raw["T_base_camera_build"], dtype=np.float64),
        T_tcp_camera=np.asarray(raw["T_tcp_camera"], dtype=np.float64),
        tcp_pose_m_rad=list(raw["tcp_pose_m_rad"]),
        camera_serial=str(raw.get("camera_serial", "")),
        handeye_path=str(raw.get("handeye_path", "")),
        intrinsics=dict(raw.get("intrinsics", {})),
        center_px=np.asarray(raw["center_px"], dtype=np.float64),
        point_camera_mm=np.asarray(raw["point_camera_mm"], dtype=np.float64),
        plane_point_camera_mm=np.asarray(raw["plane_point_camera_mm"], dtype=np.float64),
        normal_camera=np.asarray(raw["normal_camera"], dtype=np.float64),
        point_base_mm=np.asarray(raw["point_base_mm"], dtype=np.float64),
        plane_point_base_mm=np.asarray(raw["plane_point_base_mm"], dtype=np.float64),
        normal_base=np.asarray(raw["normal_base"], dtype=np.float64),
        plane_rmse_mm=float(raw["plane_rmse_mm"]),
        valid_frames=int(raw["valid_frames"]),
        total_frames=int(raw["total_frames"]),
        center_scatter_p95_px=float(raw["center_scatter_p95_px"]),
        ring_points_median=float(raw["ring_points_median"]),
        surface_model=str(raw.get("surface_model", "local_tangent_plane")),
        frame_indices=frame_indices,
        frame_centers_px=frame_centers,
        frame_plane_rmse_mm=frame_rmse,
        points_camera_mm_by_frame=points_by_frame,
        created_at=str(raw.get("created_at", "")),
        last_validated_at=str(raw.get("last_validated_at", "")),
        validation_count=int(raw.get("validation_count", 0)),
        cache_source=str(raw.get("cache_source", "unknown")),
    )


def _rejected(
    reason: str,
    total: int,
    valid: Sequence[Mapping[str, Any]],
    tracking_values: Sequence[float],
    freshness_status: str = "unknown",
    cache_age_hours: float | None = None,
) -> CacheValidationResult:
    return CacheValidationResult(
        accepted=False,
        reason=reason,
        valid_frames=len(valid),
        total_frames=int(total),
        max_tracking_distance_px=max(tracking_values) if tracking_values else None,
        center_offset_px=None,
        center_scatter_p95_px=None,
        max_plane_rmse_mm=None,
        max_normal_error_deg=None,
        point_delta_mm=None,
        current_point_base_mm=None,
        current_plane_point_base_mm=None,
        current_normal_base=None,
        current_center_px=None,
        current_point_camera_mm=None,
        current_plane_point_camera_mm=None,
        current_normal_camera=None,
        freshness_status=freshness_status,
        cache_age_hours=cache_age_hours,
    )


def _fuse_normals_robust(normals: Sequence[Any]) -> np.ndarray:
    """基于主成分的法向融合，对异常值更鲁棒。"""
    if not normals:
        raise ValueError("法向融合至少需要一个输入")

    reference = _unit(normals[0], "normal")
    aligned = []
    for n in normals:
        unit_n = _unit(n, "normal")
        # 确保所有法向方向一致（同半球）
        aligned.append(unit_n if float(unit_n @ reference) >= 0.0 else -unit_n)

    aligned_array = np.array(aligned, dtype=np.float64)

    # 如果只有1-2个法向，直接用中位数
    if len(aligned) <= 2:
        fused = np.median(aligned_array, axis=0)
        return _unit(fused, "fused normal")

    # 使用PCA提取主方向（更鲁棒）
    # 计算协方差矩阵的最大特征向量
    cov_matrix = aligned_array.T @ aligned_array
    _, eigenvectors = np.linalg.eigh(cov_matrix)
    # 最大特征值对应的特征向量
    principal_direction = eigenvectors[:, -1]

    # 确保方向与参考一致
    if float(principal_direction @ reference) < 0.0:
        principal_direction = -principal_direction

    return _unit(principal_direction, "fused normal")


def _finite_matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise ValueError(f"{name}必须是有限4x4矩阵")
    return array


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != int(size) or not np.isfinite(array).all():
        raise ValueError(f"{name}必须是有限{size}维向量")
    return array


def _unit(value: Any, name: str) -> np.ndarray:
    array = _finite_vector(value, 3, name)
    length = float(np.linalg.norm(array))
    if length < 1e-9:
        raise ValueError(f"{name}无法归一化")
    return array / length


def _fuse_normals(normals: Sequence[np.ndarray]) -> np.ndarray:
    if not normals:
        raise ValueError("没有可融合的法向")
    reference = _unit(normals[0], "normal")
    aligned = []
    for value in normals:
        item = _unit(value, "normal")
        aligned.append(item if float(item @ reference) >= 0.0 else -item)
    return _unit(np.median(np.asarray(aligned), axis=0), "fused normal")


def _angle_deg(a: Any, b: Any) -> float:
    left = _unit(a, "angle left")
    right = _unit(b, "angle right")
    return float(math.degrees(math.acos(np.clip(float(left @ right), -1.0, 1.0))))


def _finite_scalar(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _list_or_none(value: Any) -> list[float] | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def _intrinsics_dict(intrinsics: Any) -> dict[str, Any]:
    if hasattr(intrinsics, "as_dict") and callable(intrinsics.as_dict):
        return dict(intrinsics.as_dict())
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.cx),
        "cy": float(intrinsics.cy),
        "distortion": [float(value) for value in getattr(intrinsics, "distortion", ())],
    }


def _canonical_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).expanduser().resolve(strict=False)).casefold()
    except OSError:
        return text.casefold()


def _rotation_delta_deg(left: Any, right: Any) -> float:
    R_left = np.asarray(left, dtype=np.float64).reshape(3, 3)
    R_right = np.asarray(right, dtype=np.float64).reshape(3, 3)
    relative = R_left.T @ R_right
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))
