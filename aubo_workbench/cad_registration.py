#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAD 顶面孔位与 RGB 图像的离线配准。

本模块只做数学、图像和报告处理，不导入 AUBO 运动会话，也不读取深度。
Stage 0 的输出是 ``T_camera_cad``、可选的 ``T_base_cad``、11 个孔的基坐标
中心/法向、叠加图和质量报告。正式运动接入必须在更高阶段单独实现。
"""

from __future__ import annotations

import itertools
import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .cad_model import CadHole, CadModel
from .geometry import make_transform, unit_vector
from .io_utils import jsonable

# 实现已统一到 io_utils.jsonable；保留原名供本模块内既有调用点使用。
_jsonable = jsonable


def _unit(value: Any, name: str) -> np.ndarray:
    """保留"含有非有限值"这条更具体的报错，归一化本身交给 geometry.unit_vector。"""
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} 含有非有限值")
    return unit_vector(vector, name)


def _validate_transform(value: Any, name: str) -> np.ndarray:
    T = np.asarray(value, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(T).all():
        raise ValueError(f"{name} 含有非有限值")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} 不是合法齐次变换")
    if not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-3):
        raise ValueError(f"{name} 的旋转矩阵不正交")
    if np.linalg.det(T[:3, :3]) <= 0.0:
        raise ValueError(f"{name} 的旋转矩阵行列式必须为正")
    return T.copy()


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))

    def __post_init__(self) -> None:
        values = [self.fx, self.fy, self.cx, self.cy]
        if int(self.width) <= 0 or int(self.height) <= 0 or not all(math.isfinite(float(v)) for v in values):
            raise ValueError("相机内参尺寸或焦距不合法")
        if float(self.fx) <= 0.0 or float(self.fy) <= 0.0:
            raise ValueError("fx/fy 必须为正数")
        dist = np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1)
        if dist.size not in {4, 5, 8, 12, 14} or not np.isfinite(dist).all():
            raise ValueError("畸变参数数量必须为 OpenCV 支持的 4/5/8/12/14 项")
        object.__setattr__(self, "dist_coeffs", dist.copy())

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
        }


def load_intrinsics_json(path: str | Path) -> CameraIntrinsics:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"RGB 内参文件不存在: {source}")
    data = json.loads(source.read_text(encoding="utf-8"))
    root = data.get("intrinsics", data) if isinstance(data, dict) else data
    if not isinstance(root, dict):
        raise ValueError("内参 JSON 根节点必须是对象")
    opencv = data.get("opencv", {}) if isinstance(data, dict) else {}
    matrix = root.get("camera_matrix") or opencv.get("camera_matrix")
    if matrix is None:
        matrix = data.get("K") if isinstance(data, dict) else None
    if matrix is not None:
        K = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    else:
        fx = float(root["fx"])
        fy = float(root["fy"])
        cx = float(root["cx"])
        cy = float(root["cy"])
    dist = (
        root.get("distortion")
        or root.get("dist_coeffs")
        or opencv.get("dist_coeffs")
        or data.get("distortion")
        or [0.0] * 5
    )
    return CameraIntrinsics(
        width=int(root.get("width", data.get("width", 1280))),
        height=int(root.get("height", data.get("height", 800))),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        dist_coeffs=np.asarray(dist, dtype=np.float64),
    )


def load_transform_json(path: str | Path, *keys: str) -> np.ndarray:
    """读取 4x4 矩阵文件，支持直接数组或带常见字段的对象。"""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"变换文件不存在: {source}")
    data = json.loads(source.read_text(encoding="utf-8"))
    value: Any = data
    if isinstance(data, dict):
        candidate_keys = list(keys) + [
            "matrix",
            "transform",
            "T_base_camera",
            "T_camera_cad",
            "T_base_cad",
        ]
        for key in candidate_keys:
            if key in data:
                value = data[key]
                break
    return _validate_transform(value, str(source))


def transform_points(T: np.ndarray, points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    matrix = _validate_transform(T, "T")
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (matrix[:3, :3] @ values.T + matrix[:3, 3:4]).T


def _rvec_from_rotation(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64).reshape(3, 3))
    return rvec.reshape(3, 1)


def _rotation_distance_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    relative = np.asarray(R_a, dtype=np.float64).reshape(3, 3).T @ np.asarray(R_b, dtype=np.float64).reshape(3, 3)
    cosine = np.clip((float(np.trace(relative)) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.degrees(math.acos(float(cosine))))


def project_points(
    points_cad_mm: np.ndarray | Sequence[Sequence[float]],
    T_camera_cad: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    distorted: bool = False,
) -> np.ndarray:
    T = _validate_transform(T_camera_cad, "T_camera_cad")
    points = np.asarray(points_cad_mm, dtype=np.float64).reshape(-1, 3)
    rvec = _rvec_from_rotation(T[:3, :3])
    dist = intrinsics.dist_coeffs if distorted else np.zeros(5, dtype=np.float64)
    projected, _ = cv2.projectPoints(
        points,
        rvec,
        T[:3, 3].reshape(3, 1),
        intrinsics.camera_matrix,
        dist,
    )
    return projected.reshape(-1, 2)


def undistort_pixels(points_px: np.ndarray | Sequence[Sequence[float]], intrinsics: CameraIntrinsics) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
    result = cv2.undistortPoints(
        points,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        P=intrinsics.camera_matrix,
    )
    return result.reshape(-1, 2)


@dataclass
class CadDetection:
    detection_id: int
    box_xyxy: np.ndarray
    center_px_distorted: np.ndarray
    center_px: np.ndarray
    confidence: float = 0.0
    class_id: int = 0
    ellipse: dict[str, Any] | None = None
    source: str = "yolo"

    def __post_init__(self) -> None:
        self.box_xyxy = np.asarray(self.box_xyxy, dtype=np.float64).reshape(4)
        self.center_px_distorted = np.asarray(self.center_px_distorted, dtype=np.float64).reshape(2)
        self.center_px = np.asarray(self.center_px, dtype=np.float64).reshape(2)
        if not np.isfinite(self.box_xyxy).all() or not np.isfinite(self.center_px).all():
            raise ValueError("检测框/中心必须有限")

    @property
    def diameter_px(self) -> float | None:
        if not self.ellipse:
            return None
        axes = self.ellipse.get("axes_px", self.ellipse.get("axes_px_distorted"))
        if axes is None:
            return None
        values = np.asarray(axes, dtype=np.float64).reshape(-1)
        if values.size < 2 or not np.isfinite(values[:2]).all():
            return None
        return float(np.mean(values[:2]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "box_xyxy": self.box_xyxy.tolist(),
            "center_px_distorted": self.center_px_distorted.tolist(),
            "center_px": self.center_px.tolist(),
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "ellipse": _jsonable(self.ellipse),
            "source": self.source,
            "diameter_px": self.diameter_px,
        }


def detection_from_dict(
    data: dict[str, Any],
    intrinsics: CameraIntrinsics,
    detection_id: int,
) -> CadDetection:
    box = data.get("box_xyxy", data.get("box"))
    if box is None:
        raise ValueError(f"检测 {detection_id} 缺少 box/box_xyxy")
    box_array = np.asarray(box, dtype=np.float64).reshape(4)
    center_distorted = np.asarray(
        data.get("center_px_distorted", data.get("center", data.get("center_px"))),
        dtype=np.float64,
    ).reshape(2)
    center = data.get("center_px")
    ellipse = data.get("ellipse")
    if center is None:
        center_undistorted = undistort_pixels(center_distorted.reshape(1, 2), intrinsics)[0]
    else:
        center_undistorted = np.asarray(center, dtype=np.float64).reshape(2)
    return CadDetection(
        detection_id=detection_id,
        box_xyxy=box_array,
        center_px_distorted=center_distorted,
        center_px=center_undistorted,
        confidence=float(data.get("confidence", data.get("score", 0.0))),
        class_id=int(data.get("class_id", 0)),
        ellipse=ellipse,
        source=str(data.get("source", "detections_json")),
    )


def load_detections_json(path: str | Path, intrinsics: CameraIntrinsics) -> list[CadDetection]:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("detections", data.get("boxes", []))
    if not isinstance(data, list):
        raise ValueError("检测 JSON 必须是数组或包含 detections 数组的对象")
    return [detection_from_dict(item, intrinsics, index) for index, item in enumerate(data)]


def detect_yolo_holes(
    image_bgr: np.ndarray,
    model_path: str | Path,
    intrinsics: CameraIntrinsics,
    *,
    confidence: float = 0.35,
    yolo_model: Any | None = None,
) -> list[CadDetection]:
    """调用现有 YOLO 权重，并尽量复用主流程的椭圆拟合。"""

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise RuntimeError("YOLO 预览需要 ultralytics，请使用 lip_env310") from exc
    model = yolo_model if yolo_model is not None else YOLO(str(model_path))
    results = model(image_bgr, conf=float(confidence), verbose=False)
    detections: list[CadDetection] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy() if getattr(boxes, "conf", None) is not None else np.ones(len(xyxy))
        classes = boxes.cls.detach().cpu().numpy() if getattr(boxes, "cls", None) is not None else np.zeros(len(xyxy))
        for box, score, class_id in zip(xyxy, confs, classes):
            box = np.asarray(box, dtype=np.float64).reshape(4)
            center_distorted = np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float64)
            raw = {
                "box": box.tolist(),
                "center": center_distorted.tolist(),
                "confidence": float(score),
                "class_id": int(class_id),
            }
            ellipse: dict[str, Any] | None = None
            try:
                # 主脚本的导入是惰性的；没有 SDK/主脚本时仍可用框中心做诊断输入。
                from run_yolo_eye_in_hand_optimized import fit_hole_ellipse  # type: ignore

                ellipse = fit_hole_ellipse(image_bgr, raw, intrinsics)
            except Exception:
                ellipse = None
            center = (
                np.asarray(ellipse["center_px"], dtype=np.float64)
                if ellipse is not None and ellipse.get("center_px") is not None
                else undistort_pixels(center_distorted.reshape(1, 2), intrinsics)[0]
            )
            detections.append(
                CadDetection(
                    detection_id=len(detections),
                    box_xyxy=box,
                    center_px_distorted=center_distorted,
                    center_px=center,
                    confidence=float(score),
                    class_id=int(class_id),
                    ellipse=ellipse,
                    source="yolo+fit_hole_ellipse" if ellipse is not None else "yolo_box_center",
                )
            )
    return detections


@dataclass(frozen=True)
class CadRegistrationConfig:
    min_matches: int = 4
    match_distance_px: float = 80.0
    diameter_ratio_min: float = 0.75
    diameter_ratio_max: float = 1.25
    require_ellipse: bool = False
    min_triangle_area_px2: float = 200.0
    min_triangle_area_fraction: float = 0.10
    reprojection_rmse_px: float = 2.0
    reprojection_max_px: float = 4.0
    pnp_ambiguity_score_margin: float = 0.08
    positive_depth_min_mm: float = 1.0
    camera_cad_z_min_mm: float = 400.0
    camera_cad_z_max_mm: float = 650.0
    frame_count: int = 5
    min_valid_frames: int = 3
    cross_frame_center_p95_mm: float = 1.5
    max_cad_to_fine_error_mm: float = 20.0
    allow_single_frame_preview: bool = True
    automatic_triangle_ratio_tolerance: float = 0.18
    automatic_initial_distance_px: float = 100.0
    automatic_inlier_distance_px: float = 14.0
    automatic_max_hypotheses: int = 600
    automatic_unmatched_penalty_px: float = 10.0
    automatic_ambiguity_score_margin_px: float = 1.0
    automatic_diameter_consistency_weight: float = 75.0
    automatic_max_pnp_candidates: int = 128

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


@dataclass
class CadMatch:
    cad_hole_id: str
    detection_id: int
    predicted_center_px: np.ndarray
    observed_center_px: np.ndarray
    distance_px: float
    predicted_diameter_px: float | None = None
    observed_diameter_px: float | None = None
    diameter_ratio: float | None = None
    mapping_source: str = "prior_mutual_nearest"

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self.__dict__)


@dataclass
class PnPCandidate:
    T_camera_cad: np.ndarray
    positive_depth: bool
    normal_faces_camera: bool
    reprojection_rmse_px: float
    reprojection_max_px: float
    prior_translation_error_mm: float | None
    prior_rotation_error_deg: float | None
    selection_score: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self.__dict__)


@dataclass
class CadFrameResult:
    frame_index: int
    success: bool = False
    matches: list[CadMatch] = field(default_factory=list)
    unmatched_cad_ids: list[str] = field(default_factory=list)
    unmatched_detection_ids: list[int] = field(default_factory=list)
    pnp_candidates: list[PnPCandidate] = field(default_factory=list)
    selected_candidate_index: int | None = None
    T_camera_cad: np.ndarray | None = None
    T_base_cad: np.ndarray | None = None
    projected_centers_px: dict[str, np.ndarray] = field(default_factory=dict)
    reprojection_errors_px: dict[str, float] = field(default_factory=dict)
    base_holes: list[dict[str, Any]] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self.__dict__)


@dataclass
class CadRegistrationResult:
    success: bool
    stage: str
    frames: list[CadFrameResult]
    valid_frame_indices: list[int]
    T_camera_cad: np.ndarray | None = None
    T_base_cad: np.ndarray | None = None
    projected_centers_px: dict[str, np.ndarray] = field(default_factory=dict)
    reprojection_errors_px: dict[str, float] = field(default_factory=dict)
    final_holes_base: list[dict[str, Any]] = field(default_factory=list)
    cross_frame_center_p95_mm: float | None = None
    failure_reasons: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    motion_allowed: bool = False
    multi_frame_gate_pass: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self.__dict__)


@dataclass
class CadMatchingResult:
    """一帧中 CAD 孔号与 YOLO 检测的明确映射结果。"""

    matches: list[CadMatch] = field(default_factory=list)
    unmatched_cad_ids: list[str] = field(default_factory=list)
    unmatched_detection_ids: list[int] = field(default_factory=list)
    ambiguous: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self.__dict__)


def _triangle_signature(points: np.ndarray) -> np.ndarray | None:
    """返回三点三条边的归一化长度签名，用于无先验几何候选生成。"""

    values = np.asarray(points, dtype=np.float64).reshape(3, 2)
    edges = np.asarray(
        [
            np.linalg.norm(values[0] - values[1]),
            np.linalg.norm(values[0] - values[2]),
            np.linalg.norm(values[1] - values[2]),
        ],
        dtype=np.float64,
    )
    scale = float(np.max(edges))
    vector_b = values[1] - values[0]
    vector_c = values[2] - values[0]
    area = abs(float(vector_b[0] * vector_c[1] - vector_b[1] * vector_c[0])) * 0.5
    if scale <= 1e-9 or area <= 1e-9:
        return None
    return np.sort(edges / scale)


def _fit_affine_homography(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray | None:
    """由三个非共线 2D 对拟合初始仿射变换。"""

    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 2)
    if len(source) != 3 or len(target) != 3:
        return None
    design = np.column_stack((source, np.ones(len(source), dtype=np.float64)))
    if np.linalg.matrix_rank(design) < 3:
        return None
    try:
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    H = np.array(
        [
            [coefficients[0, 0], coefficients[1, 0], coefficients[2, 0]],
            [coefficients[0, 1], coefficients[1, 1], coefficients[2, 1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return H if np.isfinite(H).all() else None


def _project_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """投影平面 2D 点；仅用于匹配候选，不替代带内参的 PnP。"""

    matrix = np.asarray(H, dtype=np.float64).reshape(3, 3)
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((values, np.ones(len(values), dtype=np.float64))) @ matrix.T
    denominator = homogeneous[:, 2:3]
    if np.any(np.abs(denominator) <= 1e-12):
        raise ValueError("平面候选投影出现无效齐次坐标")
    return homogeneous[:, :2] / denominator


def _assign_unique_projected_points(
    projected_points: np.ndarray,
    observed_points: np.ndarray,
    max_distance_px: float,
    *,
    seed_pairs: Sequence[tuple[int, int]] = (),
) -> list[tuple[int, int, float]]:
    """按距离生成一对一关联；种子对应关系会被强制保留。"""

    projected = np.asarray(projected_points, dtype=np.float64).reshape(-1, 2)
    observed = np.asarray(observed_points, dtype=np.float64).reshape(-1, 2)
    if not len(projected) or not len(observed):
        return []
    limit = float(max_distance_px)
    matches: list[tuple[int, int, float]] = []
    used_projected: set[int] = set()
    used_observed: set[int] = set()
    for projected_index, observed_index in seed_pairs:
        if projected_index in used_projected or observed_index in used_observed:
            return []
        distance = float(np.linalg.norm(projected[projected_index] - observed[observed_index]))
        if not np.isfinite(distance) or distance > limit:
            return []
        matches.append((int(projected_index), int(observed_index), distance))
        used_projected.add(int(projected_index))
        used_observed.add(int(observed_index))

    pairs: list[tuple[float, int, int]] = []
    for projected_index in range(len(projected)):
        if projected_index in used_projected:
            continue
        for observed_index in range(len(observed)):
            if observed_index in used_observed:
                continue
            distance = float(np.linalg.norm(projected[projected_index] - observed[observed_index]))
            if np.isfinite(distance) and distance <= limit:
                pairs.append((distance, projected_index, observed_index))
    for distance, projected_index, observed_index in sorted(pairs):
        if projected_index in used_projected or observed_index in used_observed:
            continue
        matches.append((projected_index, observed_index, distance))
        used_projected.add(projected_index)
        used_observed.add(observed_index)
    return sorted(matches, key=lambda item: item[0])


def _refine_geometry_mapping_candidate(
    cad_xy: np.ndarray,
    observed_xy: np.ndarray,
    cad_triangle: tuple[int, int, int],
    detection_triangle: tuple[int, int, int],
    config: CadRegistrationConfig,
) -> dict[str, Any] | None:
    """由一个三点几何假设扩展到全孔并用单应性精化。"""

    seed_pairs = list(zip(cad_triangle, detection_triangle))
    H = _fit_affine_homography(
        cad_xy[list(cad_triangle)], observed_xy[list(detection_triangle)],
    )
    if H is None:
        return None
    try:
        projected = _project_homography(H, cad_xy)
    except ValueError:
        return None
    pairs = _assign_unique_projected_points(
        projected,
        observed_xy,
        float(config.automatic_initial_distance_px),
        seed_pairs=seed_pairs,
    )
    if len(pairs) < int(config.min_matches):
        return None

    for _ in range(3):
        source = cad_xy[[item[0] for item in pairs]]
        target = observed_xy[[item[1] for item in pairs]]
        try:
            refined_H, _ = cv2.findHomography(source, target, method=0)
        except cv2.error:
            return None
        if refined_H is None or not np.isfinite(refined_H).all():
            return None
        H = np.asarray(refined_H, dtype=np.float64)
        try:
            projected = _project_homography(H, cad_xy)
        except ValueError:
            return None
        pairs = _assign_unique_projected_points(
            projected,
            observed_xy,
            float(config.automatic_inlier_distance_px),
        )
        if len(pairs) < int(config.min_matches):
            return None

    errors = np.asarray([item[2] for item in pairs], dtype=np.float64)
    if not len(errors) or not np.isfinite(errors).all():
        return None
    rmse = float(np.sqrt(np.mean(errors**2)))
    maximum = float(np.max(errors))
    inlier_count = len(pairs)
    score = (
        rmse
        + 0.1 * maximum
        + float(config.automatic_unmatched_penalty_px) * max(0, len(cad_xy) - inlier_count)
    )
    return {
        "H": H,
        "pairs": pairs,
        "inlier_count": inlier_count,
        "rmse_px": rmse,
        "max_error_px": maximum,
        "score": float(score),
    }


def _triangle_diameter_scatter(
    cad_holes: Sequence[CadHole],
    cad_indices: Sequence[int],
    detections: Sequence[CadDetection],
    detection_indices: Sequence[int],
) -> float | None:
    """计算三点 CAD 直径与图像直径比例的离散度。

    绝对比例由相机距离和姿态决定，因此这里只比较三点比例是否一致。
    没有足够的椭圆直径时返回 ``None``，让调用方退回中心几何排序。
    """

    ratios: list[float] = []
    for cad_index, detection_index in zip(cad_indices, detection_indices):
        cad_diameter = float(cad_holes[int(cad_index)].diameter_mm)
        observed_diameter = detections[int(detection_index)].diameter_px
        if observed_diameter is None or cad_diameter <= 1e-9:
            return None
        ratio = float(observed_diameter) / cad_diameter
        if not np.isfinite(ratio) or ratio <= 0.0:
            return None
        ratios.append(ratio)
    if len(ratios) < 3:
        return None
    log_ratios = np.log(np.asarray(ratios, dtype=np.float64))
    return float(np.sqrt(np.mean((log_ratios - np.median(log_ratios)) ** 2)))


def _automatic_geometry_mapping(
    model: CadModel,
    detections: Sequence[CadDetection],
    intrinsics: CameraIntrinsics,
    config: CadRegistrationConfig,
) -> CadMatchingResult:
    """不依赖旧位姿，按孔间几何关系自动寻找 CAD↔YOLO 映射。"""

    detection_values = list(detections)
    cad_xy = np.asarray([hole.center_cad_mm[:2] for hole in model.holes], dtype=np.float64)
    observed_xy = np.asarray([item.center_px for item in detection_values], dtype=np.float64)
    cad_ids = [hole.hole_id for hole in model.holes]
    diagnostics: dict[str, Any] = {
        "matching_method": "triangle_geometric_hash + homography + one_to_one_assignment",
        "automatic_matching": True,
        "detection_count": len(detection_values),
        "cad_count": len(cad_ids),
        "triangle_ratio_tolerance": float(config.automatic_triangle_ratio_tolerance),
        "initial_distance_px": float(config.automatic_initial_distance_px),
        "inlier_distance_px": float(config.automatic_inlier_distance_px),
    }
    if len(detection_values) < int(config.min_matches):
        return CadMatchingResult(
            unmatched_cad_ids=list(cad_ids),
            unmatched_detection_ids=[int(item.detection_id) for item in detection_values],
            failure_reasons=[
                f"自动几何匹配检测孔数 {len(detection_values)} 小于最少 {config.min_matches} 个",
            ],
            diagnostics=diagnostics,
        )

    cad_triangles: list[tuple[tuple[int, int, int], np.ndarray]] = []
    for indices in itertools.combinations(range(len(cad_xy)), 3):
        signature = _triangle_signature(cad_xy[list(indices)])
        if signature is not None:
            cad_triangles.append((indices, signature))
    detection_triangles: list[tuple[tuple[int, int, int], np.ndarray]] = []
    for indices in itertools.combinations(range(len(observed_xy)), 3):
        signature = _triangle_signature(observed_xy[list(indices)])
        if signature is not None:
            detection_triangles.append((indices, signature))
    if not cad_triangles or not detection_triangles:
        return CadMatchingResult(
            unmatched_cad_ids=list(cad_ids),
            unmatched_detection_ids=[int(item.detection_id) for item in detection_values],
            failure_reasons=["自动几何匹配没有找到非共线三角形"],
            diagnostics=diagnostics,
        )

    compatible_triangles: list[tuple[float, tuple[int, int, int], tuple[int, int, int]]] = []
    tolerance = float(config.automatic_triangle_ratio_tolerance)
    for cad_indices, cad_signature in cad_triangles:
        for detection_indices, detection_signature in detection_triangles:
            signature_error = float(np.max(np.abs(cad_signature - detection_signature)))
            if signature_error <= tolerance:
                compatible_triangles.append((signature_error, cad_indices, detection_indices))
    compatible_triangles.sort(key=lambda item: item[0])
    diagnostics["cad_triangle_count"] = len(cad_triangles)
    diagnostics["detection_triangle_count"] = len(detection_triangles)
    diagnostics["compatible_triangle_count"] = len(compatible_triangles)

    # 三点边长签名在规则孔阵列中会产生大量对称候选。若检测提供了
    # 椭圆直径，先按“图像直径 / CAD 直径”的一致性对具体排列排序，
    # 再截断候选；这样正确排列不会因中心几何对称而被过早丢弃。
    geometry_hypotheses: list[
        tuple[float, float, tuple[int, int, int], tuple[int, int, int]]
    ] = []
    for signature_error, cad_indices, detection_indices in compatible_triangles:
        for permutation in itertools.permutations(detection_indices):
            diameter_scatter = _triangle_diameter_scatter(
                model.holes,
                cad_indices,
                detection_values,
                permutation,
            )
            diameter_sort_error = 0.0 if diameter_scatter is None else float(diameter_scatter)
            sort_score = diameter_sort_error + 0.5 * float(signature_error)
            geometry_hypotheses.append(
                (
                    float(sort_score),
                    float(signature_error),
                    cad_indices,
                    tuple(int(item) for item in permutation),
                )
            )
    geometry_hypotheses.sort(key=lambda item: (item[0], item[1]))
    max_hypotheses = max(6, int(config.automatic_max_hypotheses))
    geometry_hypotheses = geometry_hypotheses[:max_hypotheses]
    diagnostics["compatible_hypothesis_count"] = len(geometry_hypotheses)

    candidates_by_mapping: dict[tuple[tuple[int, int], ...], dict[str, Any]] = {}
    for _, _, cad_indices, detection_indices in geometry_hypotheses:
        candidate = _refine_geometry_mapping_candidate(
            cad_xy,
            observed_xy,
            cad_indices,
            detection_indices,
            config,
        )
        if candidate is None:
            continue
        mapping_key = tuple(sorted((int(item[0]), int(item[1])) for item in candidate["pairs"]))
        previous = candidates_by_mapping.get(mapping_key)
        if previous is None or float(candidate["score"]) < float(previous["score"]):
            candidate["mapping_key"] = mapping_key
            candidates_by_mapping[mapping_key] = candidate

    # 单应性只能说明平面点集能对齐；还要把每个候选送入带内参的平面 PnP，
    # 用正深度、顶面法向、相机距离和像素误差排除数学上能拟合但物理上不可能的映射。
    cad_by_index = {index: hole for index, hole in enumerate(model.holes)}
    pnp_inputs = sorted(
        candidates_by_mapping.values(),
        key=lambda item: (-int(item["inlier_count"]), float(item["score"])),
    )[: max(1, int(config.automatic_max_pnp_candidates))]
    pnp_valid_candidates: list[dict[str, Any]] = []
    pnp_rejected_count = 0
    for candidate in pnp_inputs:
        pairs = candidate["pairs"]
        object_points = np.asarray(
            [cad_by_index[item[0]].center_cad_mm for item in pairs], dtype=np.float64,
        )
        image_points = np.asarray(
            [observed_xy[item[1]] for item in pairs], dtype=np.float64,
        )
        normals = np.asarray(
            [cad_by_index[item[0]].normal_cad for item in pairs], dtype=np.float64,
        )
        raw_pnp = _solve_pnp_candidates(object_points, image_points, intrinsics)
        pnp_metrics: list[PnPCandidate] = []
        for rvec, tvec, source in raw_pnp:
            try:
                pnp_metrics.append(
                    _candidate_metrics(
                        rvec,
                        tvec,
                        object_points,
                        image_points,
                        normals,
                        intrinsics,
                        None,
                        config,
                        source,
                    )
                )
            except (ValueError, FloatingPointError, cv2.error):
                continue
        pnp_candidates = _deduplicate_pnp_candidates(pnp_metrics)
        selected_pnp_index, _ = _select_pnp_candidate(pnp_candidates, config)
        if selected_pnp_index is None:
            pnp_rejected_count += 1
            continue
        selected_pnp = pnp_candidates[selected_pnp_index]
        candidate["pnp_candidate"] = selected_pnp
        diameter_ratios: list[float] = []
        for cad_index, detection_index, _ in candidate["pairs"]:
            observed_diameter = detection_values[detection_index].diameter_px
            if observed_diameter is None:
                continue
            predicted_circle = project_cad_circle(
                cad_by_index[cad_index],
                selected_pnp.T_camera_cad,
                intrinsics,
            )
            predicted_diameter = _projected_circle_diameter_px(predicted_circle)
            if predicted_diameter is None or predicted_diameter <= 1e-9:
                continue
            ratio = float(observed_diameter / predicted_diameter)
            if np.isfinite(ratio) and ratio > 0.0:
                diameter_ratios.append(ratio)
        diameter_scatter = 0.0
        if len(diameter_ratios) >= 3:
            log_ratios = np.log(np.asarray(diameter_ratios, dtype=np.float64))
            diameter_scatter = float(np.sqrt(np.mean((log_ratios - np.median(log_ratios)) ** 2)))
            candidate["diameter_ratio_median"] = float(np.median(diameter_ratios))
            candidate["diameter_ratio_scatter"] = diameter_scatter
        else:
            candidate["diameter_ratio_median"] = None
            candidate["diameter_ratio_scatter"] = None
        candidate["diameter_observation_count"] = len(diameter_ratios)
        candidate["score"] += (
            0.25 * float(selected_pnp.reprojection_rmse_px)
            + 0.1 * float(selected_pnp.reprojection_max_px)
            + float(config.automatic_diameter_consistency_weight) * diameter_scatter
        )
        pnp_valid_candidates.append(candidate)

    candidates = sorted(
        pnp_valid_candidates,
        key=lambda item: (-int(item["inlier_count"]), float(item["score"])),
    )
    diagnostics["hypothesis_count"] = len(candidates_by_mapping)
    diagnostics["pnp_evaluated_hypothesis_count"] = len(pnp_inputs)
    diagnostics["pnp_rejected_hypothesis_count"] = pnp_rejected_count
    diagnostics["top_candidates"] = [
        {
            "inlier_count": int(item["inlier_count"]),
            "rmse_px": float(item["rmse_px"]),
            "max_error_px": float(item["max_error_px"]),
            "score": float(item["score"]),
            "diameter_observation_count": int(item.get("diameter_observation_count", 0)),
            "diameter_ratio_median": item.get("diameter_ratio_median"),
            "diameter_ratio_scatter": item.get("diameter_ratio_scatter"),
            "mapping": [
                [cad_ids[cad_index], int(detection_values[detection_index].detection_id)]
                for cad_index, detection_index in item["mapping_key"]
            ],
        }
        for item in candidates[:5]
    ]
    if not candidates:
        return CadMatchingResult(
            unmatched_cad_ids=list(cad_ids),
            unmatched_detection_ids=[int(item.detection_id) for item in detection_values],
            failure_reasons=[
                "自动几何匹配没有形成至少四孔的一致候选；请检查 YOLO 检测或进入人工映射",
            ],
            diagnostics=diagnostics,
        )

    best = candidates[0]
    ambiguous = False
    ambiguity_reason = None
    if len(candidates) > 1:
        second = candidates[1]
        same_support = int(second["inlier_count"]) == int(best["inlier_count"])
        score_gap = float(second["score"]) - float(best["score"])
        if same_support and score_gap <= float(config.automatic_ambiguity_score_margin_px):
            ambiguous = True
            ambiguity_reason = (
                "自动几何匹配存在两个近似等分候选；禁止自动选择，"
                f"best={float(best['score']):.3f}, second={float(second['score']):.3f}"
            )
    diagnostics["ambiguous"] = ambiguous
    diagnostics["best_inlier_count"] = int(best["inlier_count"])
    diagnostics["best_score"] = float(best["score"])
    if len(candidates) > 1:
        diagnostics["second_score"] = float(candidates[1]["score"])

    matches: list[CadMatch] = []
    projected = _project_homography(best["H"], cad_xy)
    selected_pnp = best["pnp_candidate"]
    used_cad: set[str] = set()
    used_detection: set[int] = set()
    for cad_index, detection_index, distance in best["pairs"]:
        cad_id = cad_ids[cad_index]
        detection = detection_values[detection_index]
        predicted_diameter = _projected_circle_diameter_px(
            project_cad_circle(cad_by_index[cad_index], selected_pnp.T_camera_cad, intrinsics)
        )
        diameter_ratio = None
        if predicted_diameter is not None and detection.diameter_px is not None:
            diameter_ratio = float(detection.diameter_px / predicted_diameter)
        matches.append(
            CadMatch(
                cad_hole_id=cad_id,
                detection_id=int(detection.detection_id),
                predicted_center_px=projected[cad_index].copy(),
                observed_center_px=detection.center_px.copy(),
                distance_px=float(distance),
                predicted_diameter_px=predicted_diameter,
                observed_diameter_px=detection.diameter_px,
                diameter_ratio=diameter_ratio,
                mapping_source="automatic_geometry_ransac",
            )
        )
        used_cad.add(cad_id)
        used_detection.add(int(detection.detection_id))
    reasons = [] if ambiguity_reason is None else [ambiguity_reason]
    return CadMatchingResult(
        matches=matches,
        unmatched_cad_ids=[cad_id for cad_id in cad_ids if cad_id not in used_cad],
        unmatched_detection_ids=[
            int(item.detection_id) for item in detection_values
            if int(item.detection_id) not in used_detection
        ],
        ambiguous=ambiguous,
        failure_reasons=reasons,
        diagnostics=diagnostics,
    )


def project_cad_circle(
    hole: CadHole,
    T_camera_cad: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    sample_count: int = 64,
) -> np.ndarray:
    """把 CAD 圆周投影到无畸变 RGB 像素坐标。"""

    normal = _unit(hole.normal_cad, f"{hole.hole_id}.normal_cad")
    reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(reference, normal))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    basis_u = _unit(np.cross(normal, reference), "CAD circle basis_u")
    basis_v = _unit(np.cross(normal, basis_u), "CAD circle basis_v")
    angles = np.linspace(0.0, 2.0 * math.pi, max(16, int(sample_count)), endpoint=False)
    points = np.asarray(
        [
            hole.center_cad_mm
            + hole.radius_mm * (math.cos(angle) * basis_u + math.sin(angle) * basis_v)
            for angle in angles
        ],
        dtype=np.float64,
    )
    return project_points(points, T_camera_cad, intrinsics, distorted=False)


def _projected_circle_diameter_px(points_px: np.ndarray) -> float | None:
    values = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    if len(values) < 4 or not np.isfinite(values).all():
        return None
    half = len(values) // 2
    opposite = np.linalg.norm(values[:half] - values[half : half * 2], axis=1)
    if opposite.size == 0 or not np.isfinite(opposite).all():
        return None
    return float(np.median(opposite))


def predict_cad_holes_px(
    model: CadModel,
    T_camera_cad: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> dict[str, dict[str, Any]]:
    """用一个近似 CAD→相机位姿生成匹配所需的孔中心/直径先验。"""

    T = _validate_transform(T_camera_cad, "T_camera_cad")
    centers = project_points(
        np.asarray([hole.center_cad_mm for hole in model.holes], dtype=np.float64),
        T,
        intrinsics,
        distorted=False,
    )
    result: dict[str, dict[str, Any]] = {}
    for index, hole in enumerate(model.holes):
        circle = project_cad_circle(hole, T, intrinsics)
        result[hole.hole_id] = {
            "center_px": centers[index],
            "diameter_px": _projected_circle_diameter_px(circle),
            "circle_px": circle,
        }
    return result


def _manual_mapping_pairs(mapping: Any) -> list[tuple[int, str]]:
    """规范化人工映射，约定主格式为 {"detection_id": "CAD-01"}。"""

    if mapping is None:
        return []
    if isinstance(mapping, (str, Path)):
        mapping = json.loads(Path(mapping).read_text(encoding="utf-8"))
    if isinstance(mapping, Mapping) and "mapping" in mapping:
        mapping = mapping["mapping"]

    pairs: list[tuple[int, str]] = []

    def add_pair(detection_value: Any, cad_value: Any) -> None:
        detection_text = str(detection_value).strip()
        if detection_text.lower().startswith("detection-"):
            detection_text = detection_text.split("-", 1)[1]
        if detection_text.lower().startswith("det-"):
            detection_text = detection_text.split("-", 1)[1]
        detection_id = int(detection_text)
        cad_id = str(cad_value).strip()
        if not cad_id:
            raise ValueError("人工映射中的 CAD 孔号不能为空")
        pairs.append((detection_id, cad_id))

    if isinstance(mapping, Mapping):
        for key, value in mapping.items():
            key_text = str(key).strip()
            value_text = str(value).strip()
            if key_text.upper().startswith("CAD-"):
                add_pair(value_text, key_text)
            else:
                add_pair(key_text, value_text)
    elif isinstance(mapping, Sequence) and not isinstance(mapping, (str, bytes)):
        for item in mapping:
            if isinstance(item, Mapping):
                detection_value = item.get("detection_id", item.get("detection", item.get("det_id")))
                cad_value = item.get("cad_hole_id", item.get("cad_id", item.get("hole_id")))
                if detection_value is None or cad_value is None:
                    raise ValueError("人工映射数组项需要 detection_id 和 cad_hole_id")
                add_pair(detection_value, cad_value)
            elif isinstance(item, Sequence) and len(item) == 2:
                add_pair(item[0], item[1])
            else:
                raise ValueError("人工映射数组项格式不支持")
    else:
        raise ValueError("人工映射必须是对象、数组或 JSON 文件")
    return pairs


def match_detections_to_cad(
    model: CadModel,
    detections: Sequence[CadDetection],
    prior_T_camera_cad: np.ndarray | None,
    intrinsics: CameraIntrinsics,
    config: CadRegistrationConfig | None = None,
    *,
    manual_mapping: Any = None,
) -> CadMatchingResult:
    """生成可审计的 CAD↔YOLO 映射。

    有先验位姿时使用投影与互为最近邻；没有先验时使用孔间几何、
    单应性、一对一分配和孔径比例排序。自动匹配不会在对称解之间
    任意挑选；无法得到唯一的至少四孔映射时，返回失败原因，调用方
    应提供人工映射。
    """

    cfg = config or CadRegistrationConfig()
    detections = list(detections)
    cad_ids = [hole.hole_id for hole in model.holes]
    detection_by_id = {int(item.detection_id): item for item in detections}
    unmatched_cad = list(cad_ids)
    unmatched_detection = sorted(detection_by_id)
    diagnostics: dict[str, Any] = {
        "automatic_matching": manual_mapping is None,
        "detection_count": len(detections),
        "cad_count": len(cad_ids),
    }

    predictions: dict[str, dict[str, Any]] = {}
    if prior_T_camera_cad is not None:
        predictions = predict_cad_holes_px(model, prior_T_camera_cad, intrinsics)

    if manual_mapping is not None:
        pairs = _manual_mapping_pairs(manual_mapping)
        seen_detections: set[int] = set()
        seen_cad: set[str] = set()
        matches: list[CadMatch] = []
        errors: list[str] = []
        for detection_id, cad_id in pairs:
            if detection_id in seen_detections:
                errors.append(f"检测 {detection_id} 被人工映射了多次")
                continue
            if cad_id in seen_cad:
                errors.append(f"CAD 孔 {cad_id} 被人工映射了多次")
                continue
            if detection_id not in detection_by_id:
                errors.append(f"人工映射引用了不存在的检测 {detection_id}")
                continue
            if cad_id not in model.hole_by_id:
                errors.append(f"人工映射引用了不存在的 CAD 孔 {cad_id}")
                continue
            detection = detection_by_id[detection_id]
            predicted = predictions.get(cad_id)
            predicted_center = (
                np.asarray(predicted["center_px"], dtype=np.float64)
                if predicted is not None
                else detection.center_px.copy()
            )
            predicted_diameter = None if predicted is None else predicted.get("diameter_px")
            observed_diameter = detection.diameter_px
            ratio = None
            if predicted_diameter and observed_diameter:
                ratio = float(observed_diameter / predicted_diameter)
                if not (cfg.diameter_ratio_min <= ratio <= cfg.diameter_ratio_max):
                    errors.append(
                        f"人工映射 {detection_id}->{cad_id} 的直径比 {ratio:.3f} 超出 "
                        f"[{cfg.diameter_ratio_min:.2f}, {cfg.diameter_ratio_max:.2f}]"
                    )
            if cfg.require_ellipse and observed_diameter is None:
                errors.append(f"人工映射 {detection_id}->{cad_id} 缺少椭圆直径")
            matches.append(
                CadMatch(
                    cad_hole_id=cad_id,
                    detection_id=detection_id,
                    predicted_center_px=predicted_center,
                    observed_center_px=detection.center_px.copy(),
                    distance_px=float(np.linalg.norm(predicted_center - detection.center_px)),
                    predicted_diameter_px=None if predicted_diameter is None else float(predicted_diameter),
                    observed_diameter_px=observed_diameter,
                    diameter_ratio=ratio,
                    mapping_source="manual_confirmed",
                )
            )
            seen_detections.add(detection_id)
            seen_cad.add(cad_id)
        unmatched_cad = [cad_id for cad_id in cad_ids if cad_id not in seen_cad]
        unmatched_detection = [det_id for det_id in sorted(detection_by_id) if det_id not in seen_detections]
        if len(matches) < cfg.min_matches:
            errors.append(f"人工映射有效孔数 {len(matches)} 小于最少 {cfg.min_matches} 个")
        diagnostics["mapping_pairs"] = pairs
        return CadMatchingResult(
            matches=matches,
            unmatched_cad_ids=unmatched_cad,
            unmatched_detection_ids=unmatched_detection,
            ambiguous=False,
            failure_reasons=errors,
            diagnostics=diagnostics,
        )

    if prior_T_camera_cad is None:
        automatic = _automatic_geometry_mapping(model, detections, intrinsics, cfg)
        automatic.diagnostics = {**diagnostics, **automatic.diagnostics}
        return automatic
    if not detections:
        return CadMatchingResult(
            unmatched_cad_ids=unmatched_cad,
            unmatched_detection_ids=unmatched_detection,
            failure_reasons=["YOLO 没有检测到大孔"],
            diagnostics=diagnostics,
        )

    detection_ids = [int(item.detection_id) for item in detections]
    distances = np.full((len(cad_ids), len(detections)), np.inf, dtype=np.float64)
    ratio_matrix = np.full_like(distances, np.nan)
    allowed = np.zeros_like(distances, dtype=bool)
    for cad_index_value, cad_id in enumerate(cad_ids):
        prediction = predictions[cad_id]
        predicted_center = np.asarray(prediction["center_px"], dtype=np.float64)
        predicted_diameter = prediction.get("diameter_px")
        for detection_index, detection in enumerate(detections):
            distance = float(np.linalg.norm(predicted_center - detection.center_px))
            distances[cad_index_value, detection_index] = distance
            observed_diameter = detection.diameter_px
            ratio = None
            if predicted_diameter and observed_diameter:
                ratio = float(observed_diameter / predicted_diameter)
                ratio_matrix[cad_index_value, detection_index] = ratio
            diameter_ok = True
            if cfg.require_ellipse and observed_diameter is None:
                diameter_ok = False
            if ratio is not None and not (cfg.diameter_ratio_min <= ratio <= cfg.diameter_ratio_max):
                diameter_ok = False
            allowed[cad_index_value, detection_index] = (
                distance <= float(cfg.match_distance_px) and diameter_ok
            )

    cad_nearest: dict[int, int] = {}
    for row in range(len(cad_ids)):
        candidates = np.where(allowed[row])[0]
        if len(candidates):
            cad_nearest[row] = int(candidates[np.argmin(distances[row, candidates])])
    detection_nearest: dict[int, int] = {}
    for col in range(len(detections)):
        candidates = np.where(allowed[:, col])[0]
        if len(candidates):
            detection_nearest[col] = int(candidates[np.argmin(distances[candidates, col])])

    candidate_pairs: list[tuple[float, int, int]] = []
    for cad_row, detection_col in cad_nearest.items():
        if detection_nearest.get(detection_col) == cad_row:
            candidate_pairs.append((float(distances[cad_row, detection_col]), cad_row, detection_col))
    candidate_pairs.sort(key=lambda item: item[0])
    matches = []
    used_cad: set[str] = set()
    used_detection: set[int] = set()
    for distance, cad_row, detection_col in candidate_pairs:
        cad_id = cad_ids[cad_row]
        detection = detections[detection_col]
        predicted = predictions[cad_id]
        predicted_diameter = predicted.get("diameter_px")
        observed_diameter = detection.diameter_px
        ratio = ratio_matrix[cad_row, detection_col]
        matches.append(
            CadMatch(
                cad_hole_id=cad_id,
                detection_id=int(detection.detection_id),
                predicted_center_px=np.asarray(predicted["center_px"], dtype=np.float64),
                observed_center_px=detection.center_px.copy(),
                distance_px=distance,
                predicted_diameter_px=None if predicted_diameter is None else float(predicted_diameter),
                observed_diameter_px=observed_diameter,
                diameter_ratio=None if not np.isfinite(ratio) else float(ratio),
                mapping_source="prior_mutual_nearest",
            )
        )
        used_cad.add(cad_id)
        used_detection.add(int(detection.detection_id))
    unmatched_cad = [cad_id for cad_id in cad_ids if cad_id not in used_cad]
    unmatched_detection = [det_id for det_id in detection_ids if det_id not in used_detection]
    diagnostics.update(
        {
            "candidate_pair_count": len(candidate_pairs),
            "allowed_edge_count": int(np.count_nonzero(allowed)),
            "match_distance_px": float(cfg.match_distance_px),
            "diameter_ratio_matrix": ratio_matrix,
        }
    )
    reasons: list[str] = []
    if len(matches) < cfg.min_matches:
        reasons.append(
            f"自动匹配唯一互为最近邻孔数 {len(matches)} 小于最少 {cfg.min_matches} 个；"
            "请核对 approximate T_camera_cad 或提供人工映射"
        )
    return CadMatchingResult(
        matches=matches,
        unmatched_cad_ids=unmatched_cad,
        unmatched_detection_ids=unmatched_detection,
        ambiguous=False,
        failure_reasons=reasons,
        diagnostics=diagnostics,
    )


def _triangle_area(points: np.ndarray) -> float:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(values) < 3:
        return 0.0
    maximum = 0.0
    for a, b, c in itertools.combinations(values, 3):
        vector_b = b - a
        vector_c = c - a
        determinant = float(vector_b[0] * vector_c[1] - vector_b[1] * vector_c[0])
        maximum = max(maximum, abs(determinant) * 0.5)
    return float(maximum)


def _check_non_collinear(
    cad_points: np.ndarray,
    image_points: np.ndarray,
    config: CadRegistrationConfig,
) -> tuple[bool, str | None, dict[str, float]]:
    cad_values = np.asarray(cad_points, dtype=np.float64).reshape(-1, 2)
    image_values = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    cad_span = np.ptp(cad_values, axis=0)
    image_span = np.ptp(image_values, axis=0)
    cad_area = _triangle_area(cad_values)
    image_area = _triangle_area(image_values)
    cad_box_area = max(1.0, float(cad_span[0] * cad_span[1]))
    image_box_area = max(1.0, float(image_span[0] * image_span[1]))
    if cad_area <= 1e-6 * cad_box_area:
        return False, "选中的 CAD 孔中心近似共线，无法进行平面 PnP", {
            "cad_max_triangle_area": cad_area,
            "image_max_triangle_area": image_area,
        }
    image_threshold = max(
        float(config.min_triangle_area_px2),
        float(config.min_triangle_area_fraction) * image_box_area,
    )
    if image_area < image_threshold:
        return False, (
            f"选中的图像孔中心近似共线：最大三角形面积 {image_area:.3f}px² "
            f"< 门限 {image_threshold:.3f}px²"
        ), {
            "cad_max_triangle_area": cad_area,
            "image_max_triangle_area": image_area,
            "image_triangle_area_threshold": image_threshold,
        }
    return True, None, {
        "cad_max_triangle_area": cad_area,
        "image_max_triangle_area": image_area,
        "image_triangle_area_threshold": image_threshold,
    }


def _select_well_spread_indices(cad_points: np.ndarray, image_points: np.ndarray) -> np.ndarray:
    cad_values = np.asarray(cad_points, dtype=np.float64).reshape(-1, 3)
    image_values = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if len(cad_values) <= 4:
        return np.arange(len(cad_values), dtype=int)
    cad_xy = cad_values[:, :2]
    cad_span = np.maximum(np.ptp(cad_xy, axis=0), 1e-9)
    image_span = np.maximum(np.ptp(image_values, axis=0), 1e-9)
    best_score = -np.inf
    best = None
    for indices in itertools.combinations(range(len(cad_values)), 4):
        selected_cad = cad_xy[list(indices)]
        selected_image = image_values[list(indices)]
        cad_area = _triangle_area(selected_cad) / float(np.prod(cad_span))
        image_area = _triangle_area(selected_image) / float(np.prod(image_span))
        score = cad_area + image_area
        if score > best_score:
            best_score = score
            best = indices
    return np.asarray(best if best is not None else tuple(range(4)), dtype=int)


def _solve_pnp_generic_ippe(object_points: np.ndarray, image_points: np.ndarray, intrinsics: CameraIntrinsics) -> list[tuple[np.ndarray, np.ndarray, str]]:
    object_values = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    image_values = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    zero_distortion = np.zeros(5, dtype=np.float64)
    try:
        output = cv2.solvePnPGeneric(
            object_values,
            image_values,
            intrinsics.camera_matrix,
            zero_distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not output[0]:
            return []
        rvecs = output[1]
        tvecs = output[2]
        return [
            (np.asarray(rvec, dtype=np.float64).reshape(3, 1), np.asarray(tvec, dtype=np.float64).reshape(3, 1), "IPPE")
            for rvec, tvec in zip(rvecs, tvecs)
        ]
    except (cv2.error, TypeError, ValueError):
        return []


def _solve_pnp_candidates(
    object_points: np.ndarray,
    image_points: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    object_values = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_values = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    candidates: list[tuple[np.ndarray, np.ndarray, str]] = []
    seed_indices = _select_well_spread_indices(object_values, image_values)
    candidates.extend(
        _solve_pnp_generic_ippe(object_values[seed_indices], image_values[seed_indices], intrinsics)
    )
    zero_distortion = np.zeros(5, dtype=np.float64)
    for rvec, tvec, source in list(candidates):
        try:
            ok, refined_rvec, refined_tvec = cv2.solvePnP(
                object_values.reshape(-1, 1, 3),
                image_values.reshape(-1, 1, 2),
                intrinsics.camera_matrix,
                zero_distortion,
                rvec.copy(),
                tvec.copy(),
                True,
                cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                candidates.append((refined_rvec, refined_tvec, f"{source}+ITERATIVE"))
        except cv2.error:
            pass
    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_values.reshape(-1, 1, 3),
            image_values.reshape(-1, 1, 2),
            intrinsics.camera_matrix,
            zero_distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok:
            candidates.append((rvec, tvec, "ITERATIVE"))
    except cv2.error:
        pass
    if not candidates:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_values.reshape(-1, 1, 3),
                image_values.reshape(-1, 1, 2),
                intrinsics.camera_matrix,
                zero_distortion,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if ok:
                candidates.append((rvec, tvec, "EPNP"))
        except cv2.error:
            pass
    return candidates


def _candidate_metrics(
    rvec: np.ndarray,
    tvec: np.ndarray,
    object_points: np.ndarray,
    image_points: np.ndarray,
    normals_cad: np.ndarray,
    intrinsics: CameraIntrinsics,
    prior_T_camera_cad: np.ndarray | None,
    config: CadRegistrationConfig,
    source: str,
) -> PnPCandidate:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = make_transform(rotation, np.asarray(tvec, dtype=np.float64).reshape(3))
    camera_points = transform_points(T, object_points)
    projected = project_points(object_points, T, intrinsics, distorted=False)
    residuals = np.linalg.norm(projected - np.asarray(image_points, dtype=np.float64).reshape(-1, 2), axis=1)
    positive_depth = bool(np.all(camera_points[:, 2] > float(config.positive_depth_min_mm)))
    normals_camera = (rotation @ np.asarray(normals_cad, dtype=np.float64).reshape(-1, 3).T).T
    normal_faces_camera = bool(np.all(np.sum(normals_camera * (-camera_points), axis=1) > 0.0))
    rmse = float(np.sqrt(np.mean(residuals**2))) if residuals.size else float("inf")
    maximum = float(np.max(residuals)) if residuals.size else float("inf")
    prior_translation = None
    prior_rotation = None
    if prior_T_camera_cad is not None:
        prior = _validate_transform(prior_T_camera_cad, "prior_T_camera_cad")
        prior_translation = float(np.linalg.norm(T[:3, 3] - prior[:3, 3]))
        prior_rotation = _rotation_distance_deg(prior[:3, :3], T[:3, :3])
    score = rmse + 0.1 * maximum
    if prior_translation is not None:
        score += 0.01 * prior_translation + 0.02 * float(prior_rotation)
    return PnPCandidate(
        T_camera_cad=T,
        positive_depth=positive_depth,
        normal_faces_camera=normal_faces_camera,
        reprojection_rmse_px=rmse,
        reprojection_max_px=maximum,
        prior_translation_error_mm=prior_translation,
        prior_rotation_error_deg=prior_rotation,
        selection_score=float(score),
        source=source,
    )


def _deduplicate_pnp_candidates(candidates: Sequence[PnPCandidate]) -> list[PnPCandidate]:
    """合并同一 IPPE 解的 seed/refine 结果，保留评分更好的记录。"""

    unique: list[PnPCandidate] = []
    for candidate in candidates:
        duplicate = None
        for index, previous in enumerate(unique):
            if (
                np.linalg.norm(candidate.T_camera_cad[:3, 3] - previous.T_camera_cad[:3, 3]) < 1.0
                and _rotation_distance_deg(candidate.T_camera_cad[:3, :3], previous.T_camera_cad[:3, :3]) < 0.5
            ):
                duplicate = index
                break
        if duplicate is None:
            unique.append(candidate)
        elif candidate.selection_score < unique[duplicate].selection_score:
            unique[duplicate] = candidate
    return unique


def _select_pnp_candidate(
    candidates: Sequence[PnPCandidate],
    config: CadRegistrationConfig,
) -> tuple[int | None, list[str]]:
    valid_indices = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.positive_depth
        and candidate.normal_faces_camera
        and float(config.camera_cad_z_min_mm) <= float(candidate.T_camera_cad[2, 3]) <= float(config.camera_cad_z_max_mm)
        and candidate.reprojection_rmse_px <= float(config.reprojection_rmse_px)
        and candidate.reprojection_max_px <= float(config.reprojection_max_px)
    ]
    if not valid_indices:
        return None, [
            "PnP 候选没有同时通过正深度、顶面法向、相机距离和重投影质量门",
        ]
    ordered = sorted(valid_indices, key=lambda index: candidates[index].selection_score)
    best_index = ordered[0]
    if len(ordered) > 1:
        best_score = float(candidates[best_index].selection_score)
        second_score = float(candidates[ordered[1]].selection_score)
        margin = max(float(config.pnp_ambiguity_score_margin), 0.08 * max(1.0, abs(best_score)))
        if second_score - best_score <= margin:
            return None, [
                "PnP 存在无法由先验/质量唯一排除的正面解；禁止自动选择，请人工确认孔号映射或补充更可靠先验",
            ]
    return best_index, []


def orient_base_hole_points(
    model: CadModel,
    T_base_cad: np.ndarray,
    T_base_camera: np.ndarray,
) -> list[dict[str, Any]]:
    """输出后续运动可消费的 CAD 孔中心和朝向，但本阶段不执行运动。"""

    base_cad = _validate_transform(T_base_cad, "T_base_cad")
    base_camera = _validate_transform(T_base_camera, "T_base_camera")
    camera_origin_base = base_camera[:3, 3]
    result: list[dict[str, Any]] = []
    for hole in model.holes:
        point_base = transform_points(base_cad, hole.center_cad_mm.reshape(1, 3))[0]
        normal_base = _unit(base_cad[:3, :3] @ hole.normal_cad, f"{hole.hole_id}.normal_base")
        normal_toward_camera = normal_base.copy()
        flipped = False
        if float(np.dot(normal_toward_camera, camera_origin_base - point_base)) < 0.0:
            normal_toward_camera = -normal_toward_camera
            flipped = True
        result.append(
            {
                "hole_id": hole.hole_id,
                "point_cad_mm": hole.center_cad_mm.copy(),
                "point_base_mm": point_base,
                "normal_cad": hole.normal_cad.copy(),
                "normal_base": normal_base,
                "normal_toward_camera_base": normal_toward_camera,
                "normal_flipped_for_camera": flipped,
                "diameter_mm": hole.diameter_mm,
                "top_z_mm": hole.top_z_mm,
            }
        )
    return result


def register_cad_frame(
    frame_index: int,
    detections: Sequence[CadDetection],
    model: CadModel,
    intrinsics: CameraIntrinsics,
    prior_T_camera_cad: np.ndarray | None,
    T_base_camera: np.ndarray | None,
    config: CadRegistrationConfig | None = None,
    *,
    manual_mapping: Any = None,
) -> CadFrameResult:
    """注册单帧；失败只返回报告，不会进入机器人流程。"""

    cfg = config or CadRegistrationConfig()
    result = CadFrameResult(frame_index=int(frame_index))
    try:
        matching = match_detections_to_cad(
            model,
            detections,
            prior_T_camera_cad,
            intrinsics,
            cfg,
            manual_mapping=manual_mapping,
        )
    except Exception as exc:
        result.failure_reasons.append(f"CAD↔YOLO 匹配异常：{type(exc).__name__}: {exc}")
        return result
    result.matches = matching.matches
    result.unmatched_cad_ids = matching.unmatched_cad_ids
    result.unmatched_detection_ids = matching.unmatched_detection_ids
    result.quality["matching"] = matching.diagnostics
    if matching.failure_reasons:
        result.failure_reasons.extend(matching.failure_reasons)
    if matching.ambiguous:
        return result
    if len(result.matches) < int(cfg.min_matches):
        return result

    detection_by_id = {int(item.detection_id): item for item in detections}
    cad_by_id = model.hole_by_id
    ordered_matches = sorted(result.matches, key=lambda item: item.cad_hole_id)
    cad_points = np.asarray([cad_by_id[item.cad_hole_id].center_cad_mm for item in ordered_matches], dtype=np.float64)
    image_points = np.asarray([detection_by_id[item.detection_id].center_px for item in ordered_matches], dtype=np.float64)
    normals = np.asarray([cad_by_id[item.cad_hole_id].normal_cad for item in ordered_matches], dtype=np.float64)
    spread_ok, spread_reason, spread_metrics = _check_non_collinear(cad_points[:, :2], image_points, cfg)
    result.quality.update(spread_metrics)
    if not spread_ok:
        result.failure_reasons.append(spread_reason or "孔中心分布不满足平面 PnP")
        return result

    raw_candidates = _solve_pnp_candidates(cad_points, image_points, intrinsics)
    result.pnp_candidates = [
        _candidate_metrics(
            rvec,
            tvec,
            cad_points,
            image_points,
            normals,
            intrinsics,
            prior_T_camera_cad,
            cfg,
            source,
        )
        for rvec, tvec, source in raw_candidates
    ]
    # IPPE seed 和 ITERATIVE 精化通常会返回同一个位姿；去重后再判断正反面
    # 解是否真正含糊，避免把同一解重复计为多解。
    result.pnp_candidates = _deduplicate_pnp_candidates(result.pnp_candidates)
    selected_index, selection_reasons = _select_pnp_candidate(result.pnp_candidates, cfg)
    result.selected_candidate_index = selected_index
    if selection_reasons:
        result.failure_reasons.extend(selection_reasons)
        return result
    assert selected_index is not None
    selected = result.pnp_candidates[selected_index]
    result.T_camera_cad = selected.T_camera_cad.copy()
    if T_base_camera is not None:
        base_camera = _validate_transform(T_base_camera, "T_base_camera")
        # T_camera_cad 是 CAD→相机，因此基坐标变换为 T_base_camera @ T_camera_cad。
        result.T_base_cad = base_camera @ result.T_camera_cad
        result.base_holes = orient_base_hole_points(model, result.T_base_cad, base_camera)

    projected_centers = project_points(
        np.asarray([hole.center_cad_mm for hole in model.holes], dtype=np.float64),
        result.T_camera_cad,
        intrinsics,
        distorted=False,
    )
    result.projected_centers_px = {
        hole.hole_id: projected_centers[index].copy() for index, hole in enumerate(model.holes)
    }
    result.reprojection_errors_px = {
        item.cad_hole_id: float(
            np.linalg.norm(result.projected_centers_px[item.cad_hole_id] - item.observed_center_px)
        )
        for item in result.matches
    }
    prior_translation = selected.prior_translation_error_mm
    gate_values = {
        "match_count": len(result.matches),
        "rmse_px": selected.reprojection_rmse_px,
        "max_error_px": selected.reprojection_max_px,
        "camera_cad_z_mm": float(result.T_camera_cad[2, 3]),
        "positive_depth": selected.positive_depth,
        "normal_faces_camera": selected.normal_faces_camera,
        "prior_translation_error_mm": prior_translation,
        "prior_rotation_error_deg": selected.prior_rotation_error_deg,
        "cad_to_fine_error_mm": prior_translation,
        "all_cad_holes_projected": len(result.projected_centers_px) == model.hole_count,
    }
    result.quality["gates"] = gate_values
    if prior_translation is not None and prior_translation > float(cfg.max_cad_to_fine_error_mm):
        result.failure_reasons.append(
            f"CAD 初始到配准位姿的平移差 {prior_translation:.3f}mm 超过 "
            f"{cfg.max_cad_to_fine_error_mm:.3f}mm 硬门"
        )
    result.success = not result.failure_reasons
    return result


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale],
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(max(1e-15, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            quaternion = np.array([(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale], dtype=np.float64)
        elif index == 1:
            scale = math.sqrt(max(1e-15, 1.0 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2])) * 2.0
            quaternion = np.array([(matrix[0, 2] + matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale], dtype=np.float64)
        else:
            scale = math.sqrt(max(1e-15, 1.0 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2])) * 2.0
            quaternion = np.array([(matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale], dtype=np.float64)
    return quaternion / max(float(np.linalg.norm(quaternion)), 1e-15)


def _quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    q = q / max(float(np.linalg.norm(q)), 1e-15)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _average_rotations(rotations: Sequence[np.ndarray]) -> np.ndarray:
    if not rotations:
        return np.eye(3, dtype=np.float64)
    reference = _rotation_to_quaternion(rotations[0])
    accumulator = np.zeros((4, 4), dtype=np.float64)
    for rotation in rotations:
        quaternion = _rotation_to_quaternion(rotation)
        if float(np.dot(quaternion, reference)) < 0.0:
            quaternion = -quaternion
        accumulator += np.outer(quaternion, quaternion)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    quaternion = eigenvectors[:, int(np.argmax(eigenvalues))]
    if float(np.dot(quaternion, reference)) < 0.0:
        quaternion = -quaternion
    return _quaternion_to_rotation(quaternion)


def fuse_frame_results(
    valid_frames: Sequence[CadFrameResult],
    model: CadModel,
    T_base_camera: np.ndarray | None,
    *,
    intrinsics: CameraIntrinsics | None = None,
    config: CadRegistrationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[dict[str, Any]], float | None]:
    """按跨帧孔中心中位数重新做一次平面 PnP，再输出融合结果。"""

    frames = [frame for frame in valid_frames if frame.T_camera_cad is not None]
    if not frames:
        raise ValueError("没有可融合的 CAD 配准帧")
    cfg = config or CadRegistrationConfig()
    translations = np.asarray([frame.T_camera_cad[:3, 3] for frame in frames], dtype=np.float64)
    translation = np.median(translations, axis=0)
    rotation = _average_rotations([frame.T_camera_cad[:3, :3] for frame in frames])
    initial_T_camera_cad = make_transform(rotation, translation)

    T_camera_cad = initial_T_camera_cad
    if intrinsics is not None:
        # 同一 CAD 孔在通过帧中的观测中心按孔号取中位数；至少半数有效帧
        # 需要看到该孔，避免把单帧偶然误检直接带入最终解。
        observations_by_id: dict[str, list[np.ndarray]] = {hole.hole_id: [] for hole in model.holes}
        for frame in frames:
            for match in frame.matches:
                observations_by_id.setdefault(match.cad_hole_id, []).append(match.observed_center_px)
        minimum_observations = max(1, int(math.ceil(len(frames) / 2.0)))
        median_ids = [
            hole.hole_id
            for hole in model.holes
            if len(observations_by_id.get(hole.hole_id, [])) >= minimum_observations
        ]
        if len(median_ids) < int(cfg.min_matches):
            raise RuntimeError(
                f"跨帧孔中心中位数可用孔数 {len(median_ids)} 小于最少 {cfg.min_matches} 个"
            )
        cad_by_id = model.hole_by_id
        median_cad = np.asarray([cad_by_id[hole_id].center_cad_mm for hole_id in median_ids], dtype=np.float64)
        median_image = np.asarray(
            [np.median(np.asarray(observations_by_id[hole_id]), axis=0) for hole_id in median_ids],
            dtype=np.float64,
        )
        spread_ok, spread_reason, _ = _check_non_collinear(median_cad[:, :2], median_image, cfg)
        if not spread_ok:
            raise RuntimeError(spread_reason or "跨帧中位数孔中心近似共线")
        median_normals = np.asarray([cad_by_id[hole_id].normal_cad for hole_id in median_ids], dtype=np.float64)
        raw_candidates = _solve_pnp_candidates(median_cad, median_image, intrinsics)
        median_candidates = _deduplicate_pnp_candidates(
            [
                _candidate_metrics(
                    rvec,
                    tvec,
                    median_cad,
                    median_image,
                    median_normals,
                    intrinsics,
                    initial_T_camera_cad,
                    cfg,
                    f"median_{source}",
                )
                for rvec, tvec, source in raw_candidates
            ]
        )
        selected_index, reasons = _select_pnp_candidate(median_candidates, cfg)
        if reasons or selected_index is None:
            raise RuntimeError("跨帧中位数 PnP 失败：" + "；".join(reasons))
        T_camera_cad = median_candidates[selected_index].T_camera_cad.copy()

    T_base_cad = None
    base_holes: list[dict[str, Any]] = []
    if T_base_camera is not None:
        base_camera = _validate_transform(T_base_camera, "T_base_camera")
        T_base_cad = base_camera @ T_camera_cad
        base_holes = orient_base_hole_points(model, T_base_cad, base_camera)

    centers_by_hole: list[np.ndarray] = []
    for frame in frames:
        frame_transform = frame.T_base_cad if T_base_cad is not None else frame.T_camera_cad
        points = transform_points(
            frame_transform,
            np.asarray([hole.center_cad_mm for hole in model.holes], dtype=np.float64),
        )
        centers_by_hole.append(points)
    centers = np.asarray(centers_by_hole, dtype=np.float64)
    if len(centers) <= 1:
        cross_frame_p95 = 0.0
    else:
        median_centers = np.median(centers, axis=0)
        deviations = np.linalg.norm(centers - median_centers[None, :, :], axis=2).reshape(-1)
        cross_frame_p95 = float(np.percentile(deviations, 95.0))
    return T_camera_cad, T_base_cad, base_holes, cross_frame_p95


def register_detections_frames(
    frame_detections: Sequence[Sequence[CadDetection]],
    model: CadModel,
    intrinsics: CameraIntrinsics,
    prior_T_camera_cad: np.ndarray | None,
    T_base_camera: np.ndarray | None,
    config: CadRegistrationConfig | None = None,
    *,
    manual_mapping: Any = None,
    sequential_prior: bool = False,
) -> CadRegistrationResult:
    """执行 Stage 0 单帧/5 帧注册质量门；始终禁止运动权限。

    ``sequential_prior=True`` 用于现场首次采集：第一帧可以使用人工确认的
    detection→CAD 映射，在第一帧 PnP 成功后，后续帧自动使用上一帧的
    CAD→相机结果进行匹配，不需要重复输入孔号映射。
    """

    cfg = config or CadRegistrationConfig()
    working_prior = None if prior_T_camera_cad is None else _validate_transform(
        prior_T_camera_cad, "prior_T_camera_cad"
    )
    frames: list[CadFrameResult] = []
    for index, detections in enumerate(frame_detections):
        mapping_for_frame = manual_mapping
        if sequential_prior and index > 0:
            mapping_for_frame = None
        frame = register_cad_frame(
            frame_index=index,
            detections=detections,
            model=model,
            intrinsics=intrinsics,
            prior_T_camera_cad=working_prior,
            T_base_camera=T_base_camera,
            config=cfg,
            manual_mapping=mapping_for_frame,
        )
        frames.append(frame)
        if sequential_prior and frame.success and frame.T_camera_cad is not None:
            working_prior = frame.T_camera_cad.copy()
    valid_frames = [frame for frame in frames if frame.success]
    valid_indices = [frame.frame_index for frame in valid_frames]
    result = CadRegistrationResult(
        success=False,
        stage="stage0_cad_registration_preview",
        frames=frames,
        valid_frame_indices=valid_indices,
        motion_allowed=False,
        multi_frame_gate_pass=False,
    )

    if len(frames) == 1 and cfg.allow_single_frame_preview:
        frame = frames[0]
        result.T_camera_cad = None if frame.T_camera_cad is None else frame.T_camera_cad.copy()
        result.T_base_cad = None if frame.T_base_cad is None else frame.T_base_cad.copy()
        result.projected_centers_px = dict(frame.projected_centers_px)
        result.reprojection_errors_px = dict(frame.reprojection_errors_px)
        result.final_holes_base = list(frame.base_holes)
        result.success = bool(frame.success)
        result.failure_reasons.extend(frame.failure_reasons)
        result.failure_reasons.append("单帧预览尚未通过 5 帧稳定门；motion_allowed=false")
        return result

    if len(frames) != int(cfg.frame_count):
        result.failure_reasons.append(
            f"多帧质量门要求 {cfg.frame_count} 帧，当前提供 {len(frames)} 帧"
        )
    if len(valid_frames) < int(cfg.min_valid_frames):
        result.failure_reasons.append(
            f"通过质量门的帧数 {len(valid_frames)} 小于最少 {cfg.min_valid_frames} 帧"
        )
    if not valid_frames:
        for frame in frames:
            result.failure_reasons.extend(
                f"frame_{frame.frame_index:03d}: {reason}" for reason in frame.failure_reasons
            )
        return result

    result.diagnostics["invalid_frame_reasons"] = {
        str(frame.frame_index): list(frame.failure_reasons)
        for frame in frames
        if not frame.success
    }

    try:
        T_camera_cad, T_base_cad, base_holes, cross_p95 = fuse_frame_results(
            valid_frames,
            model,
            T_base_camera,
            intrinsics=intrinsics,
            config=cfg,
        )
    except Exception as exc:
        result.failure_reasons.append(f"跨帧融合失败：{type(exc).__name__}: {exc}")
        return result
    result.T_camera_cad = T_camera_cad
    result.T_base_cad = T_base_cad
    final_projected = project_points(
        np.asarray([hole.center_cad_mm for hole in model.holes], dtype=np.float64),
        T_camera_cad,
        intrinsics,
        distorted=False,
    )
    result.projected_centers_px = {
        hole.hole_id: final_projected[index].copy()
        for index, hole in enumerate(model.holes)
    }
    observed_by_id: dict[str, list[np.ndarray]] = {hole.hole_id: [] for hole in model.holes}
    for frame in valid_frames:
        for match in frame.matches:
            observed_by_id.setdefault(match.cad_hole_id, []).append(match.observed_center_px)
    result.reprojection_errors_px = {
        hole_id: float(
            np.linalg.norm(
                result.projected_centers_px[hole_id]
                - np.median(np.asarray(observations), axis=0)
            )
        )
        for hole_id, observations in observed_by_id.items()
        if observations
    }
    result.final_holes_base = base_holes
    result.cross_frame_center_p95_mm = cross_p95
    result.diagnostics["fusion"] = {
        "method": "per_hole_observed_center_median_then_planar_pnp",
        "valid_frame_count": len(valid_frames),
        "configured_frame_count": int(cfg.frame_count),
    }
    cross_gate = cross_p95 is not None and cross_p95 <= float(cfg.cross_frame_center_p95_mm)
    result.multi_frame_gate_pass = (
        len(frames) == int(cfg.frame_count)
        and len(valid_frames) >= int(cfg.min_valid_frames)
        and cross_gate
        and all(frame.success for frame in valid_frames)
    )
    if not cross_gate:
        result.failure_reasons.append(
            f"跨帧孔中心 P95={cross_p95 if cross_p95 is not None else float('nan'):.3f}mm "
            f"> 门限 {cfg.cross_frame_center_p95_mm:.3f}mm"
        )
    result.success = result.multi_frame_gate_pass and not result.failure_reasons
    # 这里显式保持 False；Stage 0 报告不能授权运动。
    result.motion_allowed = False
    return result


def draw_cad_overlay(
    image_bgr: np.ndarray,
    model: CadModel,
    intrinsics: CameraIntrinsics,
    frame: CadFrameResult,
    *,
    title: str | None = None,
) -> np.ndarray:
    """绘制无畸变 RGB、CAD 圆、YOLO 框、孔号、残差箭头和统计信息。"""

    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("overlay 输入必须是 BGR 三通道图像")
    canvas = cv2.undistort(image, intrinsics.camera_matrix, intrinsics.dist_coeffs)
    if frame.T_camera_cad is not None:
        matched_cad = {match.cad_hole_id for match in frame.matches}
        for hole in model.holes:
            circle = project_cad_circle(hole, frame.T_camera_cad, intrinsics)
            integer_points = np.round(circle).astype(np.int32).reshape(-1, 1, 2)
            color = (40, 210, 40) if hole.hole_id in matched_cad else (150, 150, 0)
            if hole.hole_id not in matched_cad:
                for segment_index in range(len(integer_points)):
                    if segment_index % 2 == 0:
                        cv2.line(
                            canvas,
                            tuple(integer_points[segment_index, 0]),
                            tuple(integer_points[(segment_index + 1) % len(integer_points), 0]),
                            color,
                            2,
                            cv2.LINE_AA,
                        )
            else:
                cv2.polylines(canvas, [integer_points], True, color, 2, cv2.LINE_AA)
            center = frame.projected_centers_px.get(hole.hole_id)
            if center is not None:
                point = tuple(np.round(center).astype(int))
                cv2.putText(canvas, hole.hole_id, (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # 用匹配中保存的观测中心画 YOLO 中心；框由单独的 detection overlay helper 补充。
    for match in frame.matches:
        observed = tuple(np.round(match.observed_center_px).astype(int))
        predicted = frame.projected_centers_px.get(match.cad_hole_id)
        cv2.circle(canvas, observed, 5, (255, 80, 0), -1, cv2.LINE_AA)
        if predicted is not None:
            predicted_point = tuple(np.round(predicted).astype(int))
            cv2.arrowedLine(canvas, predicted_point, observed, (0, 80, 255), 2, cv2.LINE_AA, tipLength=0.25)
            cv2.putText(
                canvas,
                f"d{match.detection_id} e={frame.reprojection_errors_px.get(match.cad_hole_id, float('nan')):.2f}px",
                (observed[0] + 6, observed[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 80, 255),
                1,
                cv2.LINE_AA,
            )

    quality = frame.quality.get("gates", {}) if isinstance(frame.quality, dict) else {}
    rmse = quality.get("rmse_px", float("nan"))
    maximum = quality.get("max_error_px", float("nan"))
    text_lines = [
        title or f"CAD registration frame {frame.frame_index}",
        f"matches={len(frame.matches)}  RMSE={rmse:.3f}px  max={maximum:.3f}px",
        f"status={'PASS' if frame.success else 'FAIL'}  z={quality.get('camera_cad_z_mm', float('nan')):.1f}mm",
    ]
    if frame.failure_reasons:
        text_lines.append(frame.failure_reasons[0][:100])
    for line_index, line in enumerate(text_lines):
        cv2.putText(canvas, line, (20, 30 + 24 * line_index), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (20, 30 + 24 * line_index), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def draw_detection_boxes(
    overlay_bgr: np.ndarray,
    detections: Sequence[CadDetection],
    frame: CadFrameResult,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """在已经无畸变的 overlay 上绘制经畸变校正的 YOLO 框。"""

    canvas = overlay_bgr.copy()
    mapping_by_detection = {int(match.detection_id): match for match in frame.matches}
    for detection in detections:
        corners = np.asarray(
            [[detection.box_xyxy[0], detection.box_xyxy[1]], [detection.box_xyxy[2], detection.box_xyxy[3]]],
            dtype=np.float64,
        )
        undistorted_corners = undistort_pixels(corners, intrinsics)
        x1, y1 = np.round(undistorted_corners[0]).astype(int)
        x2, y2 = np.round(undistorted_corners[1]).astype(int)
        match = mapping_by_detection.get(int(detection.detection_id))
        color = (255, 160, 0) if match is not None else (0, 0, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"det-{detection.detection_id}"
        if match is not None:
            label += f" -> {match.cad_hole_id}"
        cv2.putText(canvas, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
        if match is None:
            center = tuple(np.round(detection.center_px).astype(int))
            cv2.drawMarker(canvas, center, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
    return canvas


def save_mapping_table(output_dir: str | Path, result: CadRegistrationResult) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for frame in result.frames:
        for match in frame.matches:
            rows.append(
                {
                    "frame_index": frame.frame_index,
                    "detection_id": match.detection_id,
                    "cad_hole_id": match.cad_hole_id,
                    "mapping_source": match.mapping_source,
                    "prior_distance_px": match.distance_px,
                    "reprojection_error_px": frame.reprojection_errors_px.get(match.cad_hole_id),
                }
            )
    json_path = output / "cad_registration_mapping.json"
    json_path.write_text(json.dumps(_jsonable(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output / "cad_registration_mapping.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["frame_index", "detection_id", "cad_hole_id", "mapping_source", "prior_distance_px", "reprojection_error_px"])
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def save_registration_report(
    output_dir: str | Path,
    result: CadRegistrationResult,
    *,
    model: CadModel | None = None,
    intrinsics: CameraIntrinsics | None = None,
    inputs: Mapping[str, Any] | None = None,
    detections_by_frame: Sequence[Sequence[CadDetection]] | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mapping_json, mapping_csv = save_mapping_table(output, result)
    payload: dict[str, Any] = {
        "schema_version": "cad_registration_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": result.stage,
        "motion_allowed": False,
        "inputs": _jsonable(dict(inputs or {})),
        "cad_model": None if model is None else model.to_dict(),
        "intrinsics": None if intrinsics is None else intrinsics.to_dict(),
        "result": result.to_dict(),
        "mapping_files": {
            "json": str(mapping_json),
            "csv": str(mapping_csv),
        },
    }
    if detections_by_frame is not None:
        payload["detections_by_frame"] = [
            [detection.to_dict() for detection in detections] for detections in detections_by_frame
        ]
    report_path = output / "cad_registration_report.json"
    report_path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path
