"""静止伞架实验用的自动扇区归属、候选孔筛选和去重。

这个模块只负责初始观察阶段的图像级候选管理。它不生成 340 mm 粗定位
地图，也不计算最终机器人目标。当前实验使用固定观察位的图像坐标系；
后续相机移动或转台工作流应改用工件坐标系变换后再调用同一套归属规则。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


AUTO_SECTOR_SCHEMA_VERSION = 1
DEFAULT_SECTOR_COUNT = 6


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label}必须是有限数字")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是有限数字") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label}必须是有限数字")
    return result


def _point2(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label}必须包含两个数字")
    return (_finite_float(value[0], f"{label}[0]"), _finite_float(value[1], f"{label}[1]"))


def _sector_id(value: Any, sector_count: int) -> int:
    if isinstance(value, bool):
        raise ValueError("扇区编号必须是整数")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("扇区编号必须是整数") from exc
    if result < 1 or result > int(sector_count):
        raise ValueError(f"扇区编号必须在 1..{int(sector_count)} 范围内")
    return result


@dataclass(frozen=True)
class AutoSectorConfig:
    """静止单相机观察位的自动分区配置。

    ``origin_px`` 是一次人工确认的伞架中心投影；它只适用于相机观察位
    不变的当前实验。``reference_holes`` 可选，用于跨次采集稳定孔号和
    完整性核对；没有参考孔表时，结果会明确标成 provisional。边界候选
    会保留在审计结果中，但无论该开关取值如何都不会直接进入执行列表。
    """

    origin_px: tuple[float, float] | None = None
    zero_angle_deg: float = 0.0
    sector_count: int = DEFAULT_SECTOR_COUNT
    boundary_margin_deg: float = 2.0
    dedup_distance_px: float = 25.0
    dedup_iou_threshold: float = 0.5
    dedup_min_size_ratio: float = 0.5
    min_confidence: float = 0.35
    active_sector_ids: tuple[int, ...] | None = None
    roi_polygon_px: tuple[tuple[float, float], ...] | None = None
    include_boundary_candidates: bool = True
    reference_match_distance_px: float = 40.0
    reference_holes: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    partition_mode: str = "radial"
    regions: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    image_size_px: tuple[int, int] | None = None
    # 鼠标画区时记录的“当前画面 + 机械臂位姿”。这些记录只用于复现
    # 观察位和审计，不参与孔位坐标计算，也不会触发机器人运动。
    sector_pose_records: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if self.partition_mode not in {"radial", "polygons"}:
            raise ValueError("partition_mode必须为radial或polygons")
        if self.partition_mode == "radial" and self.origin_px is None:
            raise ValueError(
                "静止伞架自动分区需要配置 origin_px；请先在固定观察位确认伞架中心投影"
            )
        if self.origin_px is not None:
            _point2(self.origin_px, "origin_px")
        if self.partition_mode == "polygons":
            if not self.regions:
                raise ValueError("请先用鼠标绘制至少一个工作区域")
            ids = []
            for region in self.regions:
                ids.append(_sector_id(region["sector_id"], self.sector_count))
                validate_region_polygon(region["polygon_px"])
            if len(ids) != len(set(ids)):
                raise ValueError("每个区域编号只能对应一个多边形")
            if self.image_size_px is None or any(v <= 0 for v in self.image_size_px):
                raise ValueError("画区配置必须记录原图宽高image_size_px")
        if isinstance(self.sector_count, bool) or int(self.sector_count) < 1:
            raise ValueError("sector_count必须是正整数")
        _finite_float(self.zero_angle_deg, "zero_angle_deg")
        margin = _finite_float(self.boundary_margin_deg, "boundary_margin_deg")
        if margin < 0.0 or margin >= 180.0 / int(self.sector_count):
            raise ValueError("boundary_margin_deg必须在[0, 每扇区角度的一半)内")
        dedup = _finite_float(self.dedup_distance_px, "dedup_distance_px")
        if dedup < 0.0:
            raise ValueError("dedup_distance_px不能为负数")
        dedup_iou = _finite_float(self.dedup_iou_threshold, "dedup_iou_threshold")
        if not 0.0 <= dedup_iou <= 1.0:
            raise ValueError("dedup_iou_threshold必须在[0,1]内")
        dedup_size_ratio = _finite_float(
            self.dedup_min_size_ratio, "dedup_min_size_ratio"
        )
        if not 0.0 < dedup_size_ratio <= 1.0:
            raise ValueError("dedup_min_size_ratio必须在(0,1]内")
        confidence = _finite_float(self.min_confidence, "min_confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("min_confidence必须在[0,1]内")
        reference_distance = _finite_float(
            self.reference_match_distance_px, "reference_match_distance_px"
        )
        if reference_distance <= 0.0:
            raise ValueError("reference_match_distance_px必须大于0")
        if self.active_sector_ids is not None:
            if not self.active_sector_ids:
                raise ValueError("active_sector_ids不能是空列表；不限制扇区时使用null")
            for value in self.active_sector_ids:
                _sector_id(value, int(self.sector_count))
            if len(set(self.active_sector_ids)) != len(self.active_sector_ids):
                raise ValueError("active_sector_ids不能包含重复扇区")
        if self.roi_polygon_px is not None and len(self.roi_polygon_px) < 3:
            raise ValueError("roi_polygon_px至少需要三个点")
        for point in self.roi_polygon_px or ():
            _point2(point, "roi_polygon_px点")
        reference_keys: set[str] = set()
        for index, reference in enumerate(self.reference_holes):
            if not isinstance(reference, dict):
                raise ValueError(f"reference_holes[{index}]必须是对象")
            if "center_px" not in reference:
                raise ValueError(f"reference_holes[{index}]缺少center_px")
            _point2(reference["center_px"], f"reference_holes[{index}].center_px")
            if "sector_id" in reference and reference["sector_id"] is not None:
                _sector_id(reference["sector_id"], int(self.sector_count))
            key = reference.get("global_hole_key", reference.get("hole_id"))
            if key is not None and not str(key).strip():
                raise ValueError(f"reference_holes[{index}]的global_hole_key不能为空")
            if key is not None:
                key_text = str(key)
                if key_text in reference_keys:
                    raise ValueError(f"reference_holes不能包含重复孔号：{key_text}")
                reference_keys.add(key_text)
        for index, record in enumerate(self.sector_pose_records):
            if not isinstance(record, dict):
                raise ValueError(f"sector_pose_records[{index}]必须是对象")
            if "sector_id" in record and record["sector_id"] is not None:
                _sector_id(record["sector_id"], int(self.sector_count))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AutoSectorConfig":
        if not isinstance(raw, dict):
            raise ValueError("自动分区配置必须是JSON对象")
        schema_version = raw.get("schema_version", AUTO_SECTOR_SCHEMA_VERSION)
        try:
            schema_version_int = int(schema_version)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"schema_version必须是整数：{schema_version!r}") from exc
        if schema_version_int != AUTO_SECTOR_SCHEMA_VERSION:
            raise ValueError(f"不支持的自动分区配置版本：{schema_version!r}")
        origin = raw.get("origin_px")
        active = raw.get("active_sector_ids")
        roi = raw.get("roi_polygon_px")
        if active is not None and not isinstance(active, (list, tuple)):
            raise ValueError("active_sector_ids必须是整数列表或null")
        if roi is not None and not isinstance(roi, (list, tuple)):
            raise ValueError("roi_polygon_px必须是点列表或null")
        references_raw = raw.get("reference_holes", [])
        if not isinstance(references_raw, (list, tuple)):
            raise ValueError("reference_holes必须是对象列表")
        pose_records_raw = raw.get("sector_pose_records", [])
        if pose_records_raw is None:
            pose_records_raw = []
        if not isinstance(pose_records_raw, (list, tuple)):
            raise ValueError("sector_pose_records必须是对象列表")
        try:
            sector_count = int(raw.get("sector_count", DEFAULT_SECTOR_COUNT))
        except (TypeError, ValueError) as exc:
            raise ValueError("sector_count必须是正整数") from exc
        config = cls(
            partition_mode=str(raw.get("partition_mode", "radial")),
            regions=tuple(raw.get("regions", [])),
            image_size_px=(None if raw.get("image_size_px") is None else
                           _point2(raw["image_size_px"], "image_size_px")),
            origin_px=None if origin is None else _point2(origin, "origin_px"),
            zero_angle_deg=_finite_float(raw.get("zero_angle_deg", 0.0), "zero_angle_deg"),
            sector_count=sector_count,
            boundary_margin_deg=_finite_float(
                raw.get("boundary_margin_deg", 2.0), "boundary_margin_deg"
            ),
            dedup_distance_px=_finite_float(
                raw.get("dedup_distance_px", 25.0), "dedup_distance_px"
            ),
            dedup_iou_threshold=_finite_float(
                raw.get("dedup_iou_threshold", 0.5), "dedup_iou_threshold"
            ),
            dedup_min_size_ratio=_finite_float(
                raw.get("dedup_min_size_ratio", 0.5), "dedup_min_size_ratio"
            ),
            min_confidence=_finite_float(raw.get("min_confidence", 0.35), "min_confidence"),
            active_sector_ids=(
                None if active is None else tuple(_sector_id(v, sector_count) for v in active)
            ),
            roi_polygon_px=(
                None
                if roi is None
                else tuple(_point2(point, "roi_polygon_px点") for point in roi)
            ),
            include_boundary_candidates=bool(raw.get("include_boundary_candidates", True)),
            reference_match_distance_px=_finite_float(
                raw.get("reference_match_distance_px", 40.0),
                "reference_match_distance_px",
            ),
            reference_holes=tuple(dict(item) for item in references_raw),
            sector_pose_records=tuple(dict(item) for item in pose_records_raw),
        )
        config.validate()
        return config

    @classmethod
    def from_path(cls, path: str | Path) -> "AutoSectorConfig":
        config_path = Path(path).expanduser()
        if not config_path.is_file():
            raise FileNotFoundError(f"自动分区配置不存在：{config_path}")
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取自动分区配置：{config_path}: {exc}") from exc
        try:
            return cls.from_dict(raw)
        except ValueError as exc:
            raise ValueError(f"自动分区配置无效：{config_path}: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AUTO_SECTOR_SCHEMA_VERSION,
            "coordinate_mode": "static_image",
            "partition_mode": self.partition_mode,
            "regions": list(self.regions),
            "image_size_px": list(self.image_size_px) if self.image_size_px else None,
            "origin_px": list(self.origin_px) if self.origin_px is not None else None,
            "zero_angle_deg": float(self.zero_angle_deg),
            "sector_count": int(self.sector_count),
            "boundary_margin_deg": float(self.boundary_margin_deg),
            "dedup_distance_px": float(self.dedup_distance_px),
            "dedup_iou_threshold": float(self.dedup_iou_threshold),
            "dedup_min_size_ratio": float(self.dedup_min_size_ratio),
            "min_confidence": float(self.min_confidence),
            "active_sector_ids": (
                list(self.active_sector_ids) if self.active_sector_ids is not None else None
            ),
            "roi_polygon_px": (
                [list(point) for point in self.roi_polygon_px]
                if self.roi_polygon_px is not None else None
            ),
            "include_boundary_candidates": bool(self.include_boundary_candidates),
            "reference_match_distance_px": float(self.reference_match_distance_px),
            "reference_holes": [dict(item) for item in self.reference_holes],
            "sector_pose_records": [dict(item) for item in self.sector_pose_records],
        }


@dataclass
class AutoSectorCandidate:
    detection_index: int
    center_px: tuple[float, float]
    box: list[float]
    confidence: float
    sector_id: int | None
    angle_deg: float | None
    radial_distance_px: float | None
    assignment_status: str
    selected: bool = False
    selection_reason: str | None = None
    global_hole_key: str | None = None
    identity_status: str = "provisional"
    duplicate_of_detection_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_index": int(self.detection_index),
            "center_px": [float(v) for v in self.center_px],
            "box": [float(v) for v in self.box],
            "confidence": float(self.confidence),
            "sector_id": int(self.sector_id) if self.sector_id is not None else None,
            "angle_deg": float(self.angle_deg) if self.angle_deg is not None else None,
            "radial_distance_px": (
                float(self.radial_distance_px) if self.radial_distance_px is not None else None
            ),
            "assignment_status": self.assignment_status,
            "selected": bool(self.selected),
            "selection_reason": self.selection_reason,
            "global_hole_key": self.global_hole_key,
            "identity_status": self.identity_status,
            "duplicate_of_detection_index": self.duplicate_of_detection_index,
        }


def _row_sort_position(item: Any, fallback_index: int) -> tuple[float, float, float, int]:
    """Return ``(x, y, height, stable_index)`` for a detection-like item.

    The automatic sector selector deliberately keeps its historical execution
    order (sector/radius order).  Map numbering needs a separate, deterministic
    image-space order, so this helper accepts both ``AutoSectorCandidate`` and
    the raw detection dictionaries used by the initial selection workflow.
    """
    if isinstance(item, AutoSectorCandidate):
        center = item.center_px
        box = item.box
        stable_index = int(item.detection_index)
    elif isinstance(item, dict):
        center = item.get("center_px")
        if center is None:
            center = item.get("center")
        box = item.get("box") or []
        stable_index = int(item.get("detection_index", fallback_index))
    else:
        raise TypeError("孔位排序项必须是AutoSectorCandidate或检测对象")
    try:
        center_values = np.asarray(center, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("孔位排序项缺少两个元素的center/center_px") from exc
    if center_values.size != 2:
        raise ValueError("孔位排序项缺少两个元素的center/center_px")
    x, y = float(center_values[0]), float(center_values[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("孔位排序坐标必须是有限数字")
    height = 0.0
    try:
        box_values = np.asarray(box, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        box_values = np.asarray([], dtype=np.float64)
    if box_values.size == 4:
        try:
            height = abs(float(box_values[3]) - float(box_values[1]))
        except (TypeError, ValueError):
            height = 0.0
        if not math.isfinite(height):
            height = 0.0
    return x, y, height, stable_index


def row_major_layout(
    items: Sequence[Any], *, row_tolerance_px: float | None = None,
) -> list[dict[str, Any]]:
    """Assign deterministic top-to-bottom/left-to-right image rows.

    A row is formed from centres whose vertical distance is within the median
    detection height based tolerance.  The input order has no effect on the
    result; ``detection_index`` is used only as a final tie-breaker.  Returned
    rows and columns are one-based and are suitable for persisting as map
    numbering metadata.
    """
    records: list[tuple[int, Any, float, float, float, int]] = []
    heights: list[float] = []
    for fallback_index, item in enumerate(items):
        x, y, height, stable_index = _row_sort_position(item, fallback_index)
        records.append((fallback_index, item, x, y, height, stable_index))
        if height > 0.0:
            heights.append(height)
    if not records:
        return []
    if row_tolerance_px is None:
        median_height = float(np.median(np.asarray(heights, dtype=np.float64))) if heights else 16.0
        # 0.70×框高允许轻微透视倾斜和检测抖动，同时以30px封顶，
        # 避免相邻真实行在低分辨率画面中被吞并。
        tolerance = max(4.0, min(0.70 * median_height, 30.0))
    else:
        tolerance = float(row_tolerance_px)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("row_tolerance_px必须是大于0的有限数字")

    # First sort by position so row creation is independent of YOLO confidence
    # order or the order in which a manual picker returns its clicks.
    records.sort(key=lambda value: (value[3], value[2], value[5], value[0]))
    rows: list[dict[str, Any]] = []
    for record in records:
        _, item, x, y, _, _ = record
        matches = [
            (abs(y - float(row["center_y"])), row_index, row)
            for row_index, row in enumerate(rows)
            if abs(y - float(row["center_y"])) <= tolerance
        ]
        if matches:
            _, _, row = min(matches, key=lambda value: (value[0], value[1]))
            row["items"].append(record)
            row["center_y"] = float(np.median([entry[3] for entry in row["items"]]))
        else:
            rows.append({"center_y": y, "items": [record]})

    rows.sort(key=lambda row: (float(row["center_y"]), min(entry[3] for entry in row["items"])))
    output: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        row_items = sorted(
            row["items"],
            key=lambda value: (value[2], value[3], value[5], value[0]),
        )
        for column_number, record in enumerate(row_items, start=1):
            _, item, x, y, _, stable_index = record
            output.append({
                "item": item,
                "row": int(row_number),
                "column": int(column_number),
                "center_px": [float(x), float(y)],
                "stable_index": int(stable_index),
                "row_tolerance_px": float(tolerance),
            })
    return output


def order_holes_row_major(
    items: Sequence[Any], *, row_tolerance_px: float | None = None,
) -> list[Any]:
    """Return hole-like items numbered top-to-bottom, then left-to-right."""
    return [entry["item"] for entry in row_major_layout(items, row_tolerance_px=row_tolerance_px)]


@dataclass
class AutoSectorSelectionResult:
    config: AutoSectorConfig
    candidates: list[AutoSectorCandidate]
    selected_indices: list[int]
    rejected: list[dict[str, Any]]
    duplicate_groups: list[list[int]]
    coverage: dict[str, Any]
    warnings: list[str]
    pending_boundary_indices: list[int] = field(default_factory=list)

    @property
    def selected_global_hole_keys(self) -> list[str]:
        """返回按执行顺序排列的已选孔号，便于调用方直接写入报告。"""
        by_index = {int(item.detection_index): item for item in self.candidates}
        return [
            str(by_index[index].global_hole_key)
            for index in self.selected_indices
            if index in by_index and by_index[index].global_hole_key
        ]

    def candidate_for_detection(self, detection_index: int) -> AutoSectorCandidate | None:
        return next(
            (item for item in self.candidates if item.detection_index == int(detection_index)),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for item in self.candidates:
            key = item.assignment_status
            counts[key] = counts.get(key, 0) + 1
        return {
            "schema_version": AUTO_SECTOR_SCHEMA_VERSION,
            "mode": "static_image",
            "config": self.config.to_dict(),
            "candidate_count": len(self.candidates),
            "selected_count": len(self.selected_indices),
            "selected_detection_indices": [int(v) for v in self.selected_indices],
            "assignment_status_counts": counts,
            "candidates": [item.to_dict() for item in self.candidates],
            "rejected": list(self.rejected),
            "duplicate_groups": [[int(v) for v in group] for group in self.duplicate_groups],
            "coverage": dict(self.coverage),
            "warnings": list(self.warnings),
            "pending_boundary_detection_indices": [
                int(v) for v in self.pending_boundary_indices
            ],
            "selected_global_hole_keys": self.selected_global_hole_keys,
        }


def normalize_angle_deg(angle_deg: float) -> float:
    """归一化到[0,360)，避免360度边界生成第七扇区。"""
    value = math.fmod(float(angle_deg), 360.0)
    if value < 0.0:
        value += 360.0
    if value >= 360.0 - 1e-9:
        return 0.0
    return value


def sector_angle_and_id(
    center_px: Sequence[float], config: AutoSectorConfig
) -> tuple[float, int, float, str]:
    """按图像坐标计算孔相对零位的角度、扇区和边界状态。"""
    config.validate()
    x, y = _point2(center_px, "center_px")
    assert config.origin_px is not None
    ox, oy = config.origin_px
    dx, dy = x - ox, y - oy
    # 图像y轴向下，因此取-dy使角度方向与常规平面坐标一致。
    raw_angle = math.degrees(math.atan2(-dy, dx))
    angle = normalize_angle_deg(raw_angle - float(config.zero_angle_deg))
    width = 360.0 / float(config.sector_count)
    sector = min(int(math.floor(angle / width)) + 1, int(config.sector_count))
    within_boundary = min(angle % width, width - (angle % width)) <= float(
        config.boundary_margin_deg
    )
    status = "pending_boundary" if within_boundary else "confirmed"
    return angle, sector, math.hypot(dx, dy), status


def _inside_roi(center: tuple[float, float], polygon: tuple[tuple[float, float], ...] | None) -> bool:
    if polygon is None:
        return True
    array = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(array, center, False) >= 0


def _reference_key(reference: dict[str, Any], index: int) -> str:
    value = reference.get("global_hole_key", reference.get("hole_id"))
    return str(value) if value is not None else f"REF-{index + 1:03d}"


def validate_region_polygon(points: Sequence) -> None:
    if len(points) < 3:
        raise ValueError("区域至少需要三个顶点")
    polygon = np.asarray([_point2(p, "区域顶点") for p in points], dtype=np.float32)
    if abs(cv2.contourArea(polygon)) < 1.0:
        raise ValueError("区域面积太小或顶点共线")
    # 拒绝交叉边和重复顶点，避免复杂多边形产生意外覆盖。
    if len(set(map(tuple, polygon))) != len(polygon):
        raise ValueError("区域顶点不能重复")
    def cross(a, b, c):
        return float((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i+1) % len(polygon)]
        for j in range(i+1, len(polygon)):
            if j == i+1 or (i == 0 and j == len(polygon)-1):
                continue
            c, d = polygon[j], polygon[(j+1) % len(polygon)]
            if (max(min(a[0], b[0]), min(c[0], d[0])) <= min(max(a[0], b[0]), max(c[0], d[0]))
                and max(min(a[1], b[1]), min(c[1], d[1])) <= min(max(a[1], b[1]), max(c[1], d[1]))
                and cross(a,b,c)*cross(a,b,d) <= 0 and cross(c,d,a)*cross(c,d,b) <= 0):
                raise ValueError("区域边线不能自相交，请重新画区")


def polygon_sector_id(center: Sequence[float], config: AutoSectorConfig) -> int | None:
    # 固定按编号从小到大归属；边界算在区域内，活动扇区筛选不能改变归属。
    for region in sorted(config.regions, key=lambda r: int(r["sector_id"])):
        if _inside_roi(tuple(map(float, center)), region["polygon_px"]):
            return int(region["sector_id"])
    return None


def validate_selection_image(image: np.ndarray, config: AutoSectorConfig) -> None:
    if config.partition_mode == "polygons" and tuple(config.image_size_px or ()) != (image.shape[1], image.shape[0]):
        raise ValueError("当前图像分辨率与画区图像不同，请在当前固定观察位重新画区")


def _normalized_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    """将检测框规范化为左上、右下坐标，兼容反向框。"""
    if len(box) != 4:
        raise ValueError("box必须包含四个数字")
    x1, y1, x2, y2 = (float(value) for value in box)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """计算两个检测框的IoU；零面积框的IoU为0。"""
    ax1, ay1, ax2, ay2 = _normalized_box(box_a)
    bx1, by1, bx2, by2 = _normalized_box(box_b)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _box_size_ratio(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """返回检测框短边尺寸比，用来避免不同尺度目标被误去重。"""
    ax1, ay1, ax2, ay2 = _normalized_box(box_a)
    bx1, by1, bx2, by2 = _normalized_box(box_b)
    short_a = min(max(0.0, ax2 - ax1), max(0.0, ay2 - ay1))
    short_b = min(max(0.0, bx2 - bx1), max(0.0, by2 - by1))
    long_side = max(short_a, short_b)
    if long_side <= 0.0:
        return 1.0 if short_a == short_b else 0.0
    return float(min(short_a, short_b) / long_side)


def _is_duplicate_pair(
    first: AutoSectorCandidate,
    second: AutoSectorCandidate,
    config: AutoSectorConfig,
) -> bool:
    """判断两个检测是否是同一孔的重复框。

    中心距离只作为上限约束，真正的重复判定还要求框有足够重叠且尺度
    相近。这样相邻孔即使中心距离小，也不会仅凭距离被合并。
    """
    distance = float(
        np.linalg.norm(np.asarray(first.center_px) - np.asarray(second.center_px))
    )
    if distance > float(config.dedup_distance_px):
        return False
    if _box_size_ratio(first.box, second.box) < float(config.dedup_min_size_ratio):
        return False
    return _box_iou(first.box, second.box) >= float(config.dedup_iou_threshold)


def _reference_sector_id(reference: dict[str, Any], config: AutoSectorConfig) -> int | None:
    if config.partition_mode == "polygons":
        return polygon_sector_id(reference["center_px"], config)
    value = reference.get("sector_id")
    if value is not None:
        return int(value)
    try:
        return int(sector_angle_and_id(reference["center_px"], config)[1])
    except (KeyError, TypeError, ValueError):
        return None


def _reference_is_in_scope(reference: dict[str, Any], config: AutoSectorConfig) -> bool:
    if config.partition_mode == "polygons" and polygon_sector_id(reference["center_px"], config) is None:
        return False
    allowed = set(config.active_sector_ids or ())
    if not allowed:
        return True
    sector_id = _reference_sector_id(reference, config)
    return sector_id in allowed


def _assign_reference_ids(
    selected: list[AutoSectorCandidate], config: AutoSectorConfig
) -> None:
    references = list(config.reference_holes)
    if not references:
        by_sector: dict[int, list[AutoSectorCandidate]] = {}
        for item in selected:
            if item.sector_id is not None:
                by_sector.setdefault(int(item.sector_id), []).append(item)
        for sector_id, items in by_sector.items():
            items.sort(
                key=lambda item: (
                    float(item.radial_distance_px or 0.0),
                    float(item.angle_deg or 0.0),
                    int(item.detection_index),
                )
            )
            for rank, item in enumerate(items, start=1):
                item.global_hole_key = f"S{sector_id:02d}-P{rank:03d}"
                item.identity_status = "provisional"
        return

    # 用增广路径做最大基数的一对一匹配。检测按置信度处理，但当一个
    # 参考孔已被占用时允许回溯换孔，避免“先到先得”造成假漏孔。
    ordered = sorted(
        selected,
        key=lambda candidate: (-float(candidate.confidence), int(candidate.detection_index)),
    )
    feasible: list[list[tuple[float, int]]] = []
    for item in ordered:
        options: list[tuple[float, int]] = []
        for index, reference in enumerate(references):
            if not _reference_is_in_scope(reference, config):
                continue
            reference_sector = _reference_sector_id(reference, config)
            if (
                reference_sector is not None
                and int(reference_sector) != int(item.sector_id or -1)
            ):
                continue
            ref_center = np.asarray(_point2(reference["center_px"], "reference center"))
            distance = float(np.linalg.norm(np.asarray(item.center_px) - ref_center))
            if distance <= float(config.reference_match_distance_px):
                options.append((distance, index))
        options.sort(key=lambda value: (value[0], value[1]))
        feasible.append(options)

    ref_to_position: dict[int, int] = {}
    position_to_ref: dict[int, int] = {}

    def augment(position: int, visited_refs: set[int]) -> bool:
        for _, reference_index in feasible[position]:
            if reference_index in visited_refs:
                continue
            visited_refs.add(reference_index)
            previous_position = ref_to_position.get(reference_index)
            if previous_position is None or augment(previous_position, visited_refs):
                ref_to_position[reference_index] = position
                position_to_ref[position] = reference_index
                return True
        return False

    for position in range(len(ordered)):
        augment(position, set())

    for position, item in enumerate(ordered):
        reference_index = position_to_ref.get(position)
        if reference_index is None:
            item.global_hole_key = None
            item.identity_status = "unmatched_reference"
        else:
            item.global_hole_key = _reference_key(references[reference_index], reference_index)
            item.identity_status = "confirmed_reference"


def _coverage_against_references(
    selected: Iterable[AutoSectorCandidate],
    config: AutoSectorConfig,
    pending_boundary_indices: Iterable[int] = (),
) -> dict[str, Any]:
    references = list(config.reference_holes)
    pending_indices = sorted({int(value) for value in pending_boundary_indices})
    if not references:
        return {
            "status": "unverified_no_reference_catalogue",
            "reference_count": 0,
            "total_reference_count": 0,
            "matched_reference_count": 0,
            "missing_reference_keys": [],
            "unmatched_candidate_detection_indices": [],
            "pending_boundary_detection_indices": pending_indices,
        }
    selected_items = list(selected)
    scoped_references = [
        (index, item)
        for index, item in enumerate(references)
        if _reference_is_in_scope(item, config)
    ]
    ref_keys = {_reference_key(item, index) for index, item in scoped_references}
    matched_keys = {
        str(item.global_hole_key)
        for item in selected_items
        if item.identity_status == "confirmed_reference" and item.global_hole_key
    }
    unmatched_candidates = [
        int(item.detection_index)
        for item in selected_items
        if item.identity_status != "confirmed_reference"
    ]
    missing = sorted(ref_keys - matched_keys)
    if not scoped_references:
        status = "unverified_no_in_scope_reference_catalogue"
    else:
        status = (
            "complete"
            if not missing and not unmatched_candidates and not pending_indices
            else "incomplete"
        )
    return {
        "status": status,
        "reference_count": len(scoped_references),
        "total_reference_count": len(references),
        "out_of_scope_reference_count": len(references) - len(scoped_references),
        "matched_reference_count": len(matched_keys),
        "missing_reference_keys": missing,
        "unmatched_candidate_detection_indices": unmatched_candidates,
        "pending_boundary_detection_indices": pending_indices,
    }


def select_auto_holes(
    detections: Sequence[dict[str, Any]], config: AutoSectorConfig
) -> AutoSectorSelectionResult:
    """把YOLO检测转换为自动选孔结果。

    该函数不假设固定孔数，也不把低质量/边界候选静默删除；所有被排除
    的检测都会出现在 ``rejected`` 或候选状态里，供报告和人工复核使用。
    """
    config.validate()
    candidates: list[AutoSectorCandidate] = []
    rejected: list[dict[str, Any]] = []
    allowed = set(config.active_sector_ids or ())
    for detection_index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            rejected.append({
                "detection_index": int(detection_index),
                "reason": "malformed_detection",
                "detail": "detection必须是对象",
            })
            continue
        try:
            center = _point2(detection.get("center"), f"detection[{detection_index}].center")
            box_raw = detection.get("box")
            if not isinstance(box_raw, (list, tuple)) or len(box_raw) != 4:
                raise ValueError("box必须包含四个数字")
            box = [_finite_float(value, f"detection[{detection_index}].box") for value in box_raw]
            confidence = _finite_float(
                detection.get("confidence", 1.0), f"detection[{detection_index}].confidence"
            )
        except (AttributeError, TypeError, ValueError) as exc:
            rejected.append({
                "detection_index": int(detection_index),
                "reason": "malformed_detection",
                "detail": str(exc),
            })
            continue
        if confidence < float(config.min_confidence):
            rejected.append({
                "detection_index": int(detection_index),
                "reason": "confidence_below_threshold",
                "confidence": confidence,
                "threshold": float(config.min_confidence),
            })
            continue
        if not _inside_roi(center, config.roi_polygon_px):
            rejected.append({
                "detection_index": int(detection_index),
                "reason": "outside_experiment_roi",
            })
            continue
        if config.partition_mode == "polygons":
            sector_id = polygon_sector_id(center, config)
            if sector_id is None:
                rejected.append({"detection_index": detection_index, "reason": "outside_drawn_regions"})
                continue
            angle, radius, assignment_status = None, None, "confirmed"
        else:
            angle, sector_id, radius, assignment_status = sector_angle_and_id(center, config)
        if allowed and sector_id not in allowed:
            rejected.append({
                "detection_index": int(detection_index),
                "reason": "outside_active_sectors",
                "sector_id": int(sector_id),
            })
            continue
        # 边界候选始终保留在审计结果中，但不进入可执行列表。边界归属
        # 必须经过粗定位或补拍确认，不能因为配置为“保留候选”就
        # 直接驱动机器人。
        selected = assignment_status == "confirmed"
        if assignment_status == "pending_boundary":
            selection_reason = (
                "boundary_requires_confirmation"
                if config.include_boundary_candidates
                else "boundary_excluded_by_config"
            )
        else:
            selection_reason = "selected"
        candidates.append(AutoSectorCandidate(
            detection_index=int(detection_index),
            center_px=center,
            box=box,
            confidence=confidence,
            sector_id=sector_id,
            angle_deg=angle,
            radial_distance_px=radius,
            assignment_status=assignment_status,
            selected=selected,
            selection_reason=selection_reason,
        ))

    # 同一固定观察位的重复检测只保留置信度最高的一个，保留所有索引供诊断。
    duplicate_groups_by_winner: dict[int, set[int]] = {}
    winners: list[AutoSectorCandidate] = []
    selected_candidates = [
        item for item in candidates
        if item.selected and item.assignment_status == "confirmed"
    ]
    if float(config.dedup_distance_px) <= 0.0:
        winners = list(selected_candidates)
    else:
        # 用连通分量而不是只和当前winner比较，避免 A-B、B-C 相近但
        # A-C稍远时把同一检测簇错误拆成两个正式孔。
        remaining = set(range(len(selected_candidates)))
        while remaining:
            seed_index = min(remaining)
            remaining.remove(seed_index)
            component = [seed_index]
            frontier = [seed_index]
            while frontier:
                current_index = frontier.pop()
                current = selected_candidates[current_index]
                neighbours = [
                    index for index in remaining
                    if _is_duplicate_pair(current, selected_candidates[index], config)
                ]
                for index in neighbours:
                    remaining.remove(index)
                    component.append(index)
                    frontier.append(index)
            members = [selected_candidates[index] for index in component]
            winner = min(members, key=lambda item: (-item.confidence, item.detection_index))
            winners.append(winner)
            if len(members) > 1:
                duplicate_groups_by_winner[int(winner.detection_index)] = {
                    int(item.detection_index) for item in members
                }
                for item in members:
                    if item is winner:
                        continue
                    item.selected = False
                    item.selection_reason = (
                        f"duplicate_of_detection_{winner.detection_index}"
                    )
                    item.duplicate_of_detection_index = winner.detection_index

    _assign_reference_ids(winners, config)
    winners.sort(
        key=lambda item: (
            int(item.sector_id or 0),
            float(item.radial_distance_px or 0.0),
            float(item.angle_deg or 0.0),
            int(item.detection_index),
        )
    )
    selected_indices = [int(item.detection_index) for item in winners]
    warnings: list[str] = []
    if config.roi_polygon_px is None and config.partition_mode != "polygons":
        warnings.append("未配置实验ROI；当前按整幅图筛选，建议在固定伞架图像上填写ROI")
    if not config.reference_holes:
        warnings.append("未配置参考孔位表；全局孔号为provisional，不能据此证明跨次采集无漏孔")
    if any(item.assignment_status == "pending_boundary" for item in candidates):
        warnings.append("存在扇区边界候选；需在粗定位或补拍后确认归属")
    pending_boundary_indices = sorted(
        int(item.detection_index)
        for item in candidates
        if item.assignment_status == "pending_boundary"
    )
    coverage = _coverage_against_references(
        winners,
        config,
        pending_boundary_indices,
    )
    sector_counts = {
        str(sector_id): sum(
            1 for item in winners if int(item.sector_id or -1) == sector_id
        )
        for sector_id in range(1, int(config.sector_count) + 1)
    }
    active_sector_ids = sorted(
        int(value) for value in (config.active_sector_ids or (
            tuple(int(r["sector_id"]) for r in config.regions)
            if config.partition_mode == "polygons"
            else tuple(range(1, int(config.sector_count) + 1))
        ))
    )
    coverage.update({
        "active_sector_ids": active_sector_ids,
        "selected_sector_counts": sector_counts,
        "unobserved_active_sector_ids": [
            sector_id for sector_id in active_sector_ids if sector_counts[str(sector_id)] == 0
        ],
        "pending_boundary_candidate_count": sum(
            1 for item in candidates if item.assignment_status == "pending_boundary"
        ),
    })
    return AutoSectorSelectionResult(
        config=config,
        candidates=candidates,
        selected_indices=selected_indices,
        rejected=rejected,
        duplicate_groups=[
            sorted(group) for group in duplicate_groups_by_winner.values()
        ],
        coverage=coverage,
        warnings=warnings,
        pending_boundary_indices=pending_boundary_indices,
    )


def render_auto_sector_overlay(
    image: np.ndarray,
    detections: Sequence[dict[str, Any]],
    result: AutoSectorSelectionResult,
) -> np.ndarray:
    """生成可审查的扇区/自动选孔叠加图。"""
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("image必须是二维或三维numpy数组")
    view = image.copy()
    if view.ndim == 2:
        view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    height, width = view.shape[:2]
    for region in result.config.regions if result.config.partition_mode == "polygons" else ():
        polygon = np.rint(region["polygon_px"]).astype(np.int32)
        cv2.polylines(view, [polygon], True, (255, 180, 40), 2)
        cv2.putText(view, f'S{int(region["sector_id"]):02d}', tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 40), 2)
    if result.config.partition_mode == "radial" and result.config.origin_px is not None:
        origin = np.asarray(result.config.origin_px, dtype=np.float64)
        radius = math.hypot(float(width), float(height)) * 1.2
        for index in range(int(result.config.sector_count)):
            angle = math.radians(float(result.config.zero_angle_deg) + index * 360.0 / result.config.sector_count)
            endpoint = (
                int(round(origin[0] + radius * math.cos(angle))),
                int(round(origin[1] - radius * math.sin(angle))),
            )
            cv2.line(view, tuple(np.rint(origin).astype(int)), endpoint, (80, 80, 180), 1, cv2.LINE_AA)
        cv2.drawMarker(view, tuple(np.rint(origin).astype(int)), (255, 0, 0), cv2.MARKER_CROSS, 26, 2)
        cv2.putText(view, "origin", tuple(np.rint(origin + [10, -10]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
    by_index = {item.detection_index: item for item in result.candidates}
    rejected_reasons = {
        int(item.get("detection_index")): str(item.get("reason", "rejected"))
        for item in result.rejected
        if item.get("detection_index") is not None
    }
    selected_set = set(result.selected_indices)
    for index, detection in enumerate(detections):
        candidate = by_index.get(index)
        box = detection.get("box") or []
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        if candidate is None:
            color = (0, 0, 180)
            label = f"D{index} {rejected_reasons.get(index, 'rejected')}"
        elif index in selected_set and candidate.assignment_status == "confirmed":
            color = (0, 210, 0)
            label = f"{candidate.global_hole_key or 'provisional'} S{candidate.sector_id:02d}"
        elif candidate.assignment_status == "pending_boundary":
            color = (0, 210, 255)
            label = (
                f"{candidate.global_hole_key or 'boundary'} "
                f"S{candidate.sector_id:02d}?"
            )
        else:
            color = (0, 150, 255)
            label = f"D{index} {candidate.selection_reason or 'excluded'}"
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        center = tuple(np.rint(np.asarray(detection.get("center", [x1, y1]), dtype=np.float64)).astype(int))
        cv2.circle(view, center, 5, color, -1)
        cv2.putText(view, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
    title = f"auto sector selection: {len(result.selected_indices)} selected"
    cv2.putText(view, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return view


def load_auto_sector_config(
    path: str | Path,
    *,
    active_sector_ids: Sequence[int] | None = None,
    include_boundary_candidates: bool | None = None,
) -> AutoSectorConfig:
    config = AutoSectorConfig.from_path(path)
    changes: dict[str, Any] = {}
    if active_sector_ids is not None:
        changes["active_sector_ids"] = tuple(int(value) for value in active_sector_ids)
    if include_boundary_candidates is not None:
        changes["include_boundary_candidates"] = bool(include_boundary_candidates)
    if not changes:
        return config
    updated = AutoSectorConfig(**{**config.__dict__, **changes})
    updated.validate()
    return updated
