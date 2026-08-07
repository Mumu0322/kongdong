#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAD 孔位模型加载、校验和 STEP 几何提取。

CAD 配准只依赖这层提供的毫米制、CAD 坐标系数据，不在这里连接相机或
机械臂。STEP 解析使用可选的 OCP；运行配准时推荐先把 STEP 固化成 JSON，
这样现场运行不会重复解析 CAD 文件。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


CAD_SCHEMA_VERSION = "cad_hole_model_v1"
DEFAULT_TOP_Z_MM = 6.0
DEFAULT_TOP_NORMAL_CAD = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须是有限的 {size} 维数值")
    return array.copy()


def _unit(value: Any, name: str) -> np.ndarray:
    vector = _finite_vector(value, 3, name)
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError(f"{name} 不能是零向量")
    return vector / length


def _first(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


@dataclass(frozen=True)
class CadHole:
    """一个 CAD 顶面孔口，所有长度均为 mm。"""

    hole_id: str
    center_cad_mm: np.ndarray
    diameter_mm: float
    top_z_mm: float
    normal_cad: np.ndarray = field(default_factory=lambda: DEFAULT_TOP_NORMAL_CAD.copy())
    row: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        center = _finite_vector(self.center_cad_mm, 3, f"{self.hole_id}.center_cad_mm")
        normal = _unit(self.normal_cad, f"{self.hole_id}.normal_cad")
        diameter = float(self.diameter_mm)
        top_z = float(self.top_z_mm)
        if not math.isfinite(diameter) or diameter <= 0.0:
            raise ValueError(f"{self.hole_id}.diameter_mm 必须为正数")
        if not math.isfinite(top_z):
            raise ValueError(f"{self.hole_id}.top_z_mm 必须有限")
        if abs(float(center[2]) - top_z) > 1e-5:
            raise ValueError(f"{self.hole_id} 中心 Z 与 top_z_mm 不一致")
        object.__setattr__(self, "center_cad_mm", center)
        object.__setattr__(self, "normal_cad", normal)
        object.__setattr__(self, "diameter_mm", diameter)
        object.__setattr__(self, "top_z_mm", top_z)

    @property
    def radius_mm(self) -> float:
        return float(self.diameter_mm) * 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.hole_id,
            "center_cad_mm": self.center_cad_mm.tolist(),
            "diameter_mm": self.diameter_mm,
            "radius_mm": self.radius_mm,
            "top_z_mm": self.top_z_mm,
            "normal_cad": self.normal_cad.tolist(),
            "row": self.row,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class CadModel:
    """可用于配准的 CAD 孔位表和坐标系元数据。"""

    source_path: str
    unit: str
    coordinate_system: str
    origin_description: str
    top_surface_z_mm: float
    top_normal_cad: np.ndarray
    holes: tuple[CadHole, ...]
    bounds_min_mm: np.ndarray | None = None
    bounds_max_mm: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unit = str(self.unit).strip().lower()
        if unit not in {"mm", "millimeter", "millimeters"}:
            raise ValueError(f"CAD 模型单位必须是 mm，当前为 {self.unit!r}")
        top_z = float(self.top_surface_z_mm)
        if not math.isfinite(top_z):
            raise ValueError("top_surface_z_mm 必须有限")
        normal = _unit(self.top_normal_cad, "top_normal_cad")
        if float(normal[2]) < 0.5:
            raise ValueError("当前实现要求 CAD 顶面法向为 +Z 方向")
        holes = tuple(self.holes)
        if len(holes) < 4:
            raise ValueError(f"CAD 孔数量至少需要 4 个，当前为 {len(holes)}")
        ids = [hole.hole_id for hole in holes]
        if len(ids) != len(set(ids)):
            raise ValueError("CAD 孔号必须唯一")
        for hole in holes:
            if abs(hole.top_z_mm - top_z) > 1e-3:
                raise ValueError(
                    f"{hole.hole_id} 顶面 Z={hole.top_z_mm:g} 与模型顶面 "
                    f"Z={top_z:g} 不一致"
                )
        points = np.asarray([hole.center_cad_mm for hole in holes], dtype=np.float64)
        bounds_min = (
            np.min(points, axis=0)
            if self.bounds_min_mm is None
            else _finite_vector(self.bounds_min_mm, 3, "bounds_min_mm")
        )
        bounds_max = (
            np.max(points, axis=0)
            if self.bounds_max_mm is None
            else _finite_vector(self.bounds_max_mm, 3, "bounds_max_mm")
        )
        if np.any(bounds_min > bounds_max):
            raise ValueError("CAD 包围盒上下限不合法")
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "top_surface_z_mm", top_z)
        object.__setattr__(self, "top_normal_cad", normal)
        object.__setattr__(self, "holes", holes)
        object.__setattr__(self, "bounds_min_mm", bounds_min)
        object.__setattr__(self, "bounds_max_mm", bounds_max)

    @property
    def hole_count(self) -> int:
        return len(self.holes)

    @property
    def hole_by_id(self) -> dict[str, CadHole]:
        return {hole.hole_id: hole for hole in self.holes}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CAD_SCHEMA_VERSION,
            "source_path": self.source_path,
            "unit": "mm",
            "coordinate_system": self.coordinate_system,
            "origin_description": self.origin_description,
            "top_surface_z_mm": self.top_surface_z_mm,
            "top_normal_cad": self.top_normal_cad.tolist(),
            "bounds_min_mm": self.bounds_min_mm.tolist(),
            "bounds_max_mm": self.bounds_max_mm.tolist(),
            "hole_count": self.hole_count,
            "holes": [hole.to_dict() for hole in self.holes],
            "metadata": self.metadata,
        }

    def save_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source_path: str = "") -> "CadModel":
        if not isinstance(data, dict):
            raise ValueError("CAD 模型 JSON 根节点必须是对象")
        unit = str(_first(data, "unit", "units", default="mm")).strip().lower()
        top_z = float(_first(data, "top_surface_z_mm", "top_z_mm", "top_z", default=DEFAULT_TOP_Z_MM))
        top_normal = _unit(
            _first(data, "top_normal_cad", "top_normal", "normal_cad", default=DEFAULT_TOP_NORMAL_CAD),
            "top_normal_cad",
        )
        coordinate_system = str(
            _first(data, "coordinate_system", "cad_coordinate_system", default="CAD")
        )
        origin_description = str(
            _first(
                data,
                "origin_description",
                "cad_origin_description",
                default="未提供，必须由 CAD/安装基准确认",
            )
        )
        raw_holes = _first(data, "holes", "cad_holes", "hole_table", default=None)
        if not isinstance(raw_holes, list):
            raise ValueError("CAD 模型 JSON 缺少 holes 数组")

        holes: list[CadHole] = []
        for index, raw in enumerate(raw_holes, start=1):
            if not isinstance(raw, dict):
                raise ValueError(f"holes[{index}] 必须是对象")
            hole_id = str(_first(raw, "id", "hole_id", "name", default=f"CAD-{index:02d}"))
            center_value = _first(
                raw,
                "center_cad_mm",
                "center_mm",
                "cad_center_mm",
                "center",
                default=None,
            )
            if center_value is None and "x_mm" in raw and "y_mm" in raw:
                center_value = [raw["x_mm"], raw["y_mm"], raw.get("z_mm", top_z)]
            if center_value is None:
                raise ValueError(f"{hole_id} 缺少 center_cad_mm")
            center = _finite_vector(center_value, 3, f"{hole_id}.center_cad_mm")
            diameter_value = _first(
                raw,
                "diameter_mm",
                "top_diameter_mm",
                "diameter",
                default=None,
            )
            if diameter_value is None and _first(raw, "radius_mm", "radius", default=None) is not None:
                diameter_value = 2.0 * float(_first(raw, "radius_mm", "radius"))
            if diameter_value is None:
                raise ValueError(f"{hole_id} 缺少 diameter_mm/radius_mm")
            hole_top_z = float(_first(raw, "top_z_mm", "z_mm", default=center[2]))
            normal = _unit(
                _first(raw, "normal_cad", "normal", default=top_normal),
                f"{hole_id}.normal_cad",
            )
            holes.append(
                CadHole(
                    hole_id=hole_id,
                    center_cad_mm=center,
                    diameter_mm=float(diameter_value),
                    top_z_mm=hole_top_z,
                    normal_cad=normal,
                    row=str(raw.get("row", "")),
                    metadata=dict(raw.get("metadata", {})),
                )
            )

        return cls(
            source_path=str(source_path or data.get("source_path", "")),
            unit=unit,
            coordinate_system=coordinate_system,
            origin_description=origin_description,
            top_surface_z_mm=top_z,
            top_normal_cad=top_normal,
            holes=tuple(holes),
            bounds_min_mm=_first(data, "bounds_min_mm", "bounds_min", default=None),
            bounds_max_mm=_first(data, "bounds_max_mm", "bounds_max", default=None),
            metadata=dict(data.get("metadata", {})),
        )


def load_cad_model_json(path: str | Path) -> CadModel:
    """读取并严格校验 CAD 孔位 JSON。"""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"CAD 孔位 JSON 不存在: {source}")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"CAD 孔位 JSON 解析失败: {source}: {exc}") from exc
    return CadModel.from_dict(data, source_path=str(source))


def _step_bounds(shape: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        from OCP.Bnd import Bnd_Box  # type: ignore
        from OCP.BRepBndLib import BRepBndLib  # type: ignore

        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box)
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        return (
            np.array([xmin, ymin, zmin], dtype=np.float64),
            np.array([xmax, ymax, zmax], dtype=np.float64),
        )
    except Exception:
        return None, None


def parse_step_model(
    path: str | Path,
    *,
    top_z_tolerance_mm: float = 0.01,
    min_radius_mm: float = 25.0,
    expected_hole_count: int | None = 11,
) -> CadModel:
    """用 OCP 从 STEP 提取顶面大孔口圆。

    该函数只遍历顶面圆形边，不把孔内部的台阶圆误当成相机可见孔口。
    OCP 是可选依赖；缺失时明确报错，不静默生成假 CAD 数据。
    """

    step_path = Path(path)
    if not step_path.is_file():
        raise FileNotFoundError(f"STEP 文件不存在: {step_path}")
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface  # type: ignore
        from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Plane  # type: ignore
        from OCP.IFSelect import IFSelect_RetDone  # type: ignore
        from OCP.STEPControl import STEPControl_Reader  # type: ignore
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE  # type: ignore
        from OCP.TopExp import TopExp_Explorer  # type: ignore
        from OCP.TopoDS import TopoDS  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "STEP 解析需要 OCP（cadquery-ocp）。请先在 lip_env310 安装后再执行解析。"
        ) from exc

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"OCP 读取 STEP 失败: {step_path}，状态={status}")
    reader.TransferRoots()
    shape = reader.OneShape()

    planes: list[tuple[float, float]] = []
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = TopoDS.Face_s(face_explorer.Current())
        try:
            surface = BRepAdaptor_Surface(face, True)
            if surface.GetType() == GeomAbs_Plane:
                plane = surface.Plane()
                location = plane.Location()
                direction = plane.Axis().Direction()
                planes.append((float(location.Z()), float(direction.Z())))
        except Exception:
            pass
        face_explorer.Next()
    if not planes:
        raise RuntimeError("STEP 中没有找到可用于确认顶面高度的平面")

    top_z = max(z for z, _ in planes)
    top_normal_z = max(planes, key=lambda item: item[0])[1]
    if top_normal_z < 0.0:
        top_normal = -DEFAULT_TOP_NORMAL_CAD
    else:
        top_normal = DEFAULT_TOP_NORMAL_CAD.copy()

    circles: list[tuple[float, float, float, float]] = []
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_explorer.More():
        edge = TopoDS.Edge_s(edge_explorer.Current())
        try:
            curve = BRepAdaptor_Curve(edge)
            if curve.GetType() == GeomAbs_Circle:
                circle = curve.Circle()
                location = circle.Location()
                radius = float(circle.Radius())
                if radius > float(min_radius_mm) and abs(float(location.Z()) - top_z) <= float(top_z_tolerance_mm):
                    circles.append((float(location.X()), float(location.Y()), top_z, radius))
        except Exception:
            pass
        edge_explorer.Next()

    if not circles:
        raise RuntimeError("STEP 顶面没有找到半径满足条件的大孔口圆")

    # 同一圆可能因拓扑边重复出现；按中心聚类，保留最大半径。
    clusters: list[tuple[float, float, float, float]] = []
    for item in circles:
        x, y, z, radius = item
        matched_index = None
        for index, current in enumerate(clusters):
            if math.hypot(x - current[0], y - current[1]) <= 1e-3:
                matched_index = index
                break
        if matched_index is None:
            clusters.append(item)
        elif radius > clusters[matched_index][3]:
            clusters[matched_index] = item

    # STEP 浮点拓扑常带 1e-14 级噪声；先按毫米级近似值排序，避免同一
    # CAD 行内的孔号被拓扑遍历顺序打乱。
    clusters.sort(key=lambda item: (round(item[1], 3), round(item[0], 3)))
    if expected_hole_count is not None and len(clusters) != int(expected_hole_count):
        raise RuntimeError(
            f"STEP 顶面大孔数量为 {len(clusters)}，期望 {expected_hole_count}；"
            "请检查半径/顶面过滤条件，不自动截断或任选孔位。"
        )

    bounds_min, bounds_max = _step_bounds(shape)
    holes = tuple(
        CadHole(
            hole_id=f"CAD-{index:02d}",
            center_cad_mm=np.array([x, y, z], dtype=np.float64),
            diameter_mm=2.0 * radius,
            top_z_mm=z,
            normal_cad=top_normal,
            metadata={"source": "step_top_circular_edge", "radius_mm": radius},
        )
        for index, (x, y, z, radius) in enumerate(clusters, start=1)
    )
    return CadModel(
        source_path=str(step_path),
        unit="mm",
        coordinate_system="CAD",
        origin_description="由 STEP 模型定义；请结合安装基准确认与 Base 的方向关系",
        top_surface_z_mm=top_z,
        top_normal_cad=top_normal,
        holes=holes,
        bounds_min_mm=bounds_min,
        bounds_max_mm=bounds_max,
        metadata={
            "parser": "OCP STEPControl_Reader + top circular edges",
            "top_z_tolerance_mm": top_z_tolerance_mm,
            "min_radius_mm": min_radius_mm,
        },
    )
