"""Assignment and joint-transform math for shared hole localization."""

from __future__ import annotations

from itertools import combinations
import math
from typing import Any

import numpy as np

from aubo_workbench.hole_localization_models import TwoStageConfig


def _assign_detections_to_projection(
    detections: list[dict[str, Any]],
    projected_holes_px: dict[str, np.ndarray],
    max_distance_px: float,
    expected_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """将检测框与预计孔位做带门限的一对一全局匹配。

    代价以预计像素位置为主，并可选叠加类别、框尺寸和置信度信息。
    优先使用 scipy 的匈牙利算法；现场环境没有 scipy 时回退到确定性的
    距离排序算法，仍保证一个检测框只会分配给一个孔。
    """

    hole_ids = [str(value) for value in projected_holes_px]
    if not hole_ids or not detections:
        return {}
    metadata = expected_metadata or {}
    max_distance = float(max_distance_px)
    cost = np.full((len(hole_ids), len(detections)), np.inf, dtype=np.float64)
    for hole_id, expected in projected_holes_px.items():
        expected_value = np.asarray(expected, dtype=np.float64).reshape(2)
        for detection_index, detection in enumerate(detections):
            center = np.asarray(detection["center"], dtype=np.float64).reshape(2)
            distance = float(np.linalg.norm(center - expected_value))
            if distance > max_distance:
                continue
            hole_metadata = metadata.get(str(hole_id), {})
            expected_class = hole_metadata.get("class_id")
            if expected_class is not None and int(detection.get("class_id", -1)) != int(expected_class):
                continue
            value = distance
            expected_box = hole_metadata.get("box")
            detection_box = detection.get("box")
            if expected_box is not None and detection_box is not None:
                expected_size = max(
                    float(expected_box[2]) - float(expected_box[0]),
                    float(expected_box[3]) - float(expected_box[1]),
                    1.0,
                )
                detected_size = max(
                    float(detection_box[2]) - float(detection_box[0]),
                    float(detection_box[3]) - float(detection_box[1]),
                    1.0,
                )
                value += 4.0 * abs(math.log(detected_size / expected_size))
            value -= 2.0 * float(detection.get("confidence", 0.0))
            cost[hole_ids.index(str(hole_id)), detection_index] = value

    assignments: dict[str, dict[str, Any]] = {}
    finite_pairs = np.argwhere(np.isfinite(cost))
    if finite_pairs.size == 0:
        return assignments

    try:
        from scipy.optimize import linear_sum_assignment

        safe_cost = np.where(np.isfinite(cost), cost, 1.0e9)
        row_indices, column_indices = linear_sum_assignment(safe_cost)
        selected_pairs = [
            (int(row), int(column))
            for row, column in zip(row_indices, column_indices)
            if np.isfinite(cost[row, column])
        ]
    except Exception:
        # 机器人现场可能只部署最小依赖；无 scipy 时使用纯 Python/Numpy
        # 的最大匹配、最小代价回退，不能退化为简单的逐边贪心。
        selected_pairs = _minimum_cost_maximum_assignment_without_scipy(cost)

    used_rows: set[int] = set()
    assigned: dict[str, dict[str, Any]] = {}
    used_detection_indices: set[int] = set()
    for row, detection_index in selected_pairs:
        if row in used_rows or detection_index in used_detection_indices:
            continue
        hole_id = hole_ids[row]
        expected_value = np.asarray(projected_holes_px[hole_id], dtype=np.float64).reshape(2)
        distance = float(np.linalg.norm(
            np.asarray(detections[detection_index]["center"], dtype=np.float64).reshape(2)
            - expected_value
        ))
        assigned[hole_id] = {
            "detection": detections[detection_index],
            "detection_index": detection_index,
            "distance_px": distance,
        }
        used_rows.add(row)
        used_detection_indices.add(detection_index)
    return assigned


def _minimum_cost_maximum_assignment_without_scipy(
    cost: np.ndarray,
) -> list[tuple[int, int]]:
    """无 scipy 时计算最大基数、最小代价的一对一匹配。

    通过虚拟未匹配行/列把“最大匹配数量”编码为主目标，再用方阵
    Hungarian 算法优化有限边代价。有限边代价按当前匹配代价范围设置，
    因此任何合法匹配都优先于把孔位留空，非法边永远不会被选中。
    """
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        return []
    row_count, column_count = matrix.shape
    finite_values = matrix[np.isfinite(matrix)]
    if finite_values.size == 0:
        return []

    unmatched_cost = max(1.0, float(np.max(np.abs(finite_values))) + 1.0)
    forbidden_cost = unmatched_cost * 3.0
    size = row_count + column_count
    padded = np.zeros((size, size), dtype=np.float64)
    padded[:row_count, :column_count] = np.where(
        np.isfinite(matrix), matrix, forbidden_cost,
    )
    # 真实孔位匹配虚拟列表示“不匹配”；虚拟行匹配真实检测框表示
    # “该检测框未被选中”。两者的配对代价分别是 unmatched_cost 和0。
    padded[:row_count, column_count:] = unmatched_cost

    # 方阵 Hungarian 最小化实现，兼容负代价。
    u = np.zeros(size + 1, dtype=np.float64)
    v = np.zeros(size + 1, dtype=np.float64)
    p = np.zeros(size + 1, dtype=np.int32)
    way = np.zeros(size + 1, dtype=np.int32)
    for row in range(1, size + 1):
        p[0] = row
        min_value = np.full(size + 1, np.inf, dtype=np.float64)
        used = np.zeros(size + 1, dtype=bool)
        column0 = 0
        while True:
            used[column0] = True
            row0 = int(p[column0])
            delta = math.inf
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = padded[row0 - 1, column - 1] - u[row0] - v[column]
                if current < min_value[column]:
                    min_value[column] = current
                    way[column] = column0
                if min_value[column] < delta:
                    delta = float(min_value[column])
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[int(p[column])] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            previous = int(way[column0])
            p[column0] = p[previous]
            column0 = previous
            if column0 == 0:
                break

    assigned_columns = np.full(size, -1, dtype=np.int32)
    for column in range(1, size + 1):
        if p[column] > 0:
            assigned_columns[int(p[column]) - 1] = column - 1
    return [
        (row, int(column))
        for row, column in enumerate(assigned_columns[:row_count])
        if 0 <= int(column) < column_count and np.isfinite(matrix[row, int(column)])
    ]


def _fit_batch_projected_anchor_correction(
    detections: list[dict[str, Any]],
    projected_holes_px: dict[str, np.ndarray],
    max_distance_px: float,
    min_matches: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """用首帧的宽匹配结果校正批量投影锚点的整体成像误差。

    公共观察位姿下，手眼/内参与实际到位误差通常表现为小的平移、缩放
    或剪切。先用宽门限做一次一一匹配，再拟合二维仿射映射；后续帧仍由
    原有严格跟踪门限和融合质量门限验收，避免单纯放宽匹配距离。
    """
    broad = _assign_detections_to_projection(
        detections, projected_holes_px, float(max_distance_px),
    )
    required = max(3, min(int(min_matches), len(projected_holes_px)))
    if len(broad) < required:
        raise RuntimeError(
            f"批量锚点校正有效匹配不足：{len(broad)}/{required}"
        )

    source = np.asarray(
        [np.asarray(projected_holes_px[hole_id], dtype=np.float64).reshape(2)
         for hole_id in broad],
        dtype=np.float64,
    )
    target = np.asarray(
        [np.asarray(item["detection"]["center"], dtype=np.float64).reshape(2)
         for item in broad.values()],
        dtype=np.float64,
    )
    design = np.column_stack((source, np.ones(len(source), dtype=np.float64)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if int(rank) < 3 or not np.isfinite(coefficients).all():
        raise RuntimeError("批量锚点校正仿射模型退化")

    linear = np.asarray(coefficients[:2, :], dtype=np.float64).T
    singular_values = np.linalg.svd(linear, compute_uv=False)
    translation_norm = float(np.linalg.norm(coefficients[2, :]))
    if (
        not np.isfinite(singular_values).all()
        or float(np.min(singular_values)) < 0.85
        or float(np.max(singular_values)) > 1.15
        or translation_norm > 50.0
    ):
        raise RuntimeError(
            "批量锚点校正模型偏离单位变换："
            f"scale={singular_values.tolist()}，平移={translation_norm:.2f}px"
        )

    corrected_values = np.column_stack((
        np.asarray(list(projected_holes_px.values()), dtype=np.float64),
        np.ones(len(projected_holes_px), dtype=np.float64),
    )) @ coefficients
    corrected = {
        hole_id: corrected_values[index]
        for index, hole_id in enumerate(projected_holes_px)
    }
    fitted = design @ coefficients
    residuals = np.linalg.norm(fitted - target, axis=1)
    residual_p95 = float(np.percentile(residuals, 95))
    residual_max = float(np.max(residuals))
    if residual_p95 > 5.0 or residual_max > 8.0:
        raise RuntimeError(
            "批量锚点校正残差过大："
            f"P95={residual_p95:.2f}px，最大={residual_max:.2f}px"
        )

    return corrected, {
        "enabled": True,
        "model": "affine_expected_to_detected",
        "match_count": len(broad),
        "match_hole_ids": [str(hole_id) for hole_id in broad],
        "calibration_max_distance_px": float(max_distance_px),
        "residual_p95_px": residual_p95,
        "residual_max_px": residual_max,
        "linear_singular_values": singular_values.tolist(),
        "translation_norm_px": translation_norm,
        "matrix": coefficients.tolist(),
        "raw_anchors_px": {
            str(hole_id): np.asarray(point, dtype=np.float64).tolist()
            for hole_id, point in projected_holes_px.items()
        },
        "corrected_anchors_px": {
            str(hole_id): np.asarray(point, dtype=np.float64).tolist()
            for hole_id, point in corrected.items()
        },
    }




def _fit_planar_rigid_transform(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit a weighted 2-D rigid transform without allowing scale or shear."""
    source = np.asarray(source_xy, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_xy, dtype=np.float64).reshape(-1, 2)
    if source.shape != target.shape or len(source) < 2:
        raise ValueError("平面刚体变换至少需要两个一一对应的二维点")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("平面刚体变换输入包含非有限坐标")

    if weights is None:
        weight_values = np.ones(len(source), dtype=np.float64)
    else:
        weight_values = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(weight_values) != len(source):
            raise ValueError("平面刚体变换权重数量与点数量不一致")
        weight_values = np.maximum(weight_values, 1.0e-6)
    if not np.isfinite(weight_values).all() or float(np.sum(weight_values)) <= 0.0:
        raise ValueError("平面刚体变换权重无效")

    weight_values = weight_values / float(np.sum(weight_values))
    source_centroid = np.sum(source * weight_values[:, None], axis=0)
    target_centroid = np.sum(target * weight_values[:, None], axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    source_spread = float(np.max(np.linalg.norm(source_centered, axis=1)))
    if source_spread < 1.0e-6:
        raise ValueError("平面刚体变换源点几何退化")

    covariance = source_centered.T @ (weight_values[:, None] * target_centered)
    try:
        left, _, right_transposed = np.linalg.svd(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("平面刚体变换SVD失败") from exc
    rotation = right_transposed.T @ left.T
    if float(np.linalg.det(rotation)) < 0.0:
        right_transposed[-1, :] *= -1.0
        rotation = right_transposed.T @ left.T
    if not np.isfinite(rotation).all() or float(np.linalg.det(rotation)) <= 0.0:
        raise ValueError("平面刚体变换旋转矩阵无效")

    translation = target_centroid - rotation @ source_centroid
    predicted = (rotation @ source.T).T + translation
    residuals = np.linalg.norm(predicted - target, axis=1)
    yaw_rad = float(math.atan2(rotation[1, 0], rotation[0, 0]))
    return {
        "rotation": rotation,
        "translation": translation,
        "yaw_rad": yaw_rad,
        "predicted": predicted,
        "residuals_mm": residuals,
        "weighted_rmse_mm": float(
            math.sqrt(np.sum(weight_values * residuals * residuals))
        ),
    }


def _fit_batch_fine_joint_transform(
    hole_ids: list[int],
    source_xy_by_hole: dict[int, np.ndarray],
    target_xy_by_hole: dict[int, np.ndarray],
    weights_by_hole: dict[int, float],
    *,
    min_holes: int,
    max_residual_mm: float,
    max_translation_mm: float,
    max_yaw_deg: float,
    estimate_rotation: bool = True,
) -> dict[str, Any]:
    """Fit one robust shared XY correction from fused hole centers.

    The source points are the 340 mm coarse hole centers and the target points
    are the 260 mm RGB centers intersected with each hole's frozen coarse plane.
    For three or more holes a planar rigid transform is used. For a two-hole
    group callers can disable rotation and require the two measured offsets to
    agree on one translation. The existing ChArUco affine model remains the
    final TCP compensation and is not replaced by this observation-level fit.
    """
    ordered_ids = [int(value) for value in hole_ids]
    available = [
        hole_id for hole_id in ordered_ids
        if hole_id in source_xy_by_hole and hole_id in target_xy_by_hole
    ]
    required = max(2, int(min_holes))
    if len(available) < required:
        raise ValueError(f"共享精定位联合孔数不足：{len(available)}/{required}")

    source = np.asarray(
        [source_xy_by_hole[hole_id] for hole_id in available], dtype=np.float64,
    )
    target = np.asarray(
        [target_xy_by_hole[hole_id] for hole_id in available], dtype=np.float64,
    )
    weights = np.asarray([
        max(1.0e-3, float(weights_by_hole.get(hole_id, 1.0)))
        for hole_id in available
    ], dtype=np.float64)
    max_residual = max(1.0e-6, float(max_residual_mm))

    if not estimate_rotation:
        # A two-point rigid fit can turn a small independent center error into
        # a large yaw change. Estimate only the common XY translation and use
        # the per-hole offset residual as the consistency gate.
        offsets = target - source
        translation = np.average(offsets, axis=0, weights=weights)
        residuals = np.linalg.norm(offsets - translation, axis=1)
        inlier_mask = residuals <= max_residual
        if int(np.count_nonzero(inlier_mask)) < required:
            raise ValueError(
                f"共享精定位联合平移残差过大：没有达到{required}个孔的"
                f"{max_residual:.3f}mm内点门槛"
            )
        translation = np.average(
            offsets[inlier_mask], axis=0, weights=weights[inlier_mask],
        )
        residuals = np.linalg.norm(offsets - translation, axis=1)
        inlier_mask = residuals <= max_residual
        if int(np.count_nonzero(inlier_mask)) < required:
            raise ValueError(
                f"共享精定位联合平移内点不足："
                f"{int(np.count_nonzero(inlier_mask))}/{required}"
            )
        translation_norm = float(np.linalg.norm(translation))
        if translation_norm > float(max_translation_mm):
            raise ValueError(
                f"共享精定位联合平移超限：{translation_norm:.3f}mm > "
                f"{float(max_translation_mm):.3f}mm"
            )
        predicted_points = source + translation
        inlier_residuals = residuals[inlier_mask]
        weighted_rmse = float(math.sqrt(
            np.sum(weights[inlier_mask] * inlier_residuals * inlier_residuals)
            / max(float(np.sum(weights[inlier_mask])), 1.0e-6)
        ))
        return {
            "success": True,
            "fit_mode": "translation_only",
            "hole_ids": available,
            "inlier_hole_ids": [
                hole_id for hole_id, keep in zip(available, inlier_mask)
                if bool(keep)
            ],
            "rotation": np.eye(2, dtype=np.float64),
            "translation_mm": np.asarray(translation, dtype=np.float64),
            "yaw_rad": 0.0,
            "yaw_deg": 0.0,
            "translation_norm_mm": translation_norm,
            "predicted_xy_by_hole": {
                hole_id: np.asarray(predicted_point, dtype=np.float64)
                for hole_id, predicted_point in zip(available, predicted_points)
            },
            "target_xy_by_hole": {
                hole_id: np.asarray(point, dtype=np.float64)
                for hole_id, point in zip(available, target)
            },
            "residual_mm_by_hole": {
                hole_id: float(residual)
                for hole_id, residual in zip(available, residuals)
            },
            "residual_p95_mm": float(np.percentile(inlier_residuals, 95.0)),
            "residual_max_mm": float(np.max(inlier_residuals)),
            "weighted_rmse_mm": weighted_rmse,
            "max_residual_mm": max_residual,
        }

    # Fitting all available points can let one bad center pull the solution.
    # Enumerate deterministic 2-/3-point rigid hypotheses as a small
    # RANSAC-like guard, then retain all compatible inliers for the final
    # weighted fit. The 2-point hypotheses also let a 3-hole group keep two
    # good holes while sending only the outlier to the per-hole fallback.
    candidate_indices: list[tuple[int, ...]] = [tuple(range(len(available)))]
    for subset_size in (2, 3):
        if len(available) >= subset_size:
            candidate_indices.extend(
                combinations(range(len(available)), subset_size)
            )

    best: tuple[tuple[int, float, float], dict[str, Any], np.ndarray] | None = None
    for indices in candidate_indices:
        index_array = np.asarray(indices, dtype=np.int64)
        try:
            candidate = _fit_planar_rigid_transform(
                source[index_array], target[index_array], weights[index_array],
            )
        except ValueError:
            continue
        candidate_predicted = (
            candidate["rotation"] @ source.T
        ).T + candidate["translation"]
        residuals = np.linalg.norm(candidate_predicted - target, axis=1)
        inliers = residuals <= max_residual
        if int(np.count_nonzero(inliers)) < required:
            continue
        weighted_rmse = float(math.sqrt(
            np.sum(weights * np.minimum(residuals, max_residual) ** 2)
            / max(float(np.sum(weights)), 1.0e-6)
        ))
        key = (
            int(np.count_nonzero(inliers)),
            -weighted_rmse,
            -float(np.max(residuals[inliers])) if np.any(inliers) else -math.inf,
        )
        if best is None or key > best[0]:
            best = (key, candidate, inliers)

    if best is None:
        raise ValueError(
            f"共享精定位联合变换残差过大：没有达到{required}个孔的"
            f"{max_residual:.3f}mm内点门槛"
        )

    _, _, inlier_mask = best
    refined = _fit_planar_rigid_transform(
        source[inlier_mask], target[inlier_mask], weights[inlier_mask],
    )
    residuals = np.linalg.norm(
        (refined["rotation"] @ source.T).T + refined["translation"] - target,
        axis=1,
    )
    inlier_mask = residuals <= max_residual
    if int(np.count_nonzero(inlier_mask)) >= required and not np.all(inlier_mask):
        refined = _fit_planar_rigid_transform(
            source[inlier_mask], target[inlier_mask], weights[inlier_mask],
        )
        residuals = np.linalg.norm(
            (refined["rotation"] @ source.T).T + refined["translation"] - target,
            axis=1,
        )

    translation_norm = float(np.linalg.norm(refined["translation"]))
    yaw_deg = abs(float(math.degrees(refined["yaw_rad"])))
    if translation_norm > float(max_translation_mm):
        raise ValueError(
            f"共享精定位联合平移超限：{translation_norm:.3f}mm > "
            f"{float(max_translation_mm):.3f}mm"
        )
    if yaw_deg > float(max_yaw_deg):
        raise ValueError(
            f"共享精定位联合旋转超限：{yaw_deg:.3f}deg > "
            f"{float(max_yaw_deg):.3f}deg"
        )
    if int(np.count_nonzero(inlier_mask)) < required:
        raise ValueError(
            f"共享精定位联合内点不足：{int(np.count_nonzero(inlier_mask))}/{required}"
        )

    predicted_points = (refined["rotation"] @ source.T).T + refined["translation"]
    residual_p95 = float(np.percentile(residuals[inlier_mask], 95.0))
    return {
        "success": True,
        "fit_mode": "planar_rigid",
        "hole_ids": available,
        "inlier_hole_ids": [
            hole_id for hole_id, keep in zip(available, inlier_mask) if bool(keep)
        ],
        "rotation": np.asarray(refined["rotation"], dtype=np.float64),
        "translation_mm": np.asarray(refined["translation"], dtype=np.float64),
        "yaw_rad": float(refined["yaw_rad"]),
        "yaw_deg": float(math.degrees(refined["yaw_rad"])),
        "translation_norm_mm": translation_norm,
        "predicted_xy_by_hole": {
            hole_id: np.asarray(predicted_point, dtype=np.float64)
            for hole_id, predicted_point in zip(available, predicted_points)
        },
        "target_xy_by_hole": {
            hole_id: np.asarray(point, dtype=np.float64)
            for hole_id, point in zip(available, target)
        },
        "residual_mm_by_hole": {
            hole_id: float(residual)
            for hole_id, residual in zip(available, residuals)
        },
        "residual_p95_mm": residual_p95,
        "residual_max_mm": float(np.max(residuals[inlier_mask])),
        "weighted_rmse_mm": float(refined["weighted_rmse_mm"]),
        "max_residual_mm": max_residual,
    }


def _batch_fine_joint_transform_stable(
    transforms: list[dict[str, Any]],
    min_valid_frames: int,
    max_translation_scatter_mm: float,
    max_yaw_scatter_deg: float,
) -> bool:
    """Check whether the latest shared transforms form a stable burst."""
    required = max(1, int(min_valid_frames))
    if len(transforms) < required:
        return False
    recent = transforms[-required:]
    translations = np.asarray([
        np.asarray(item["translation_mm"], dtype=np.float64).reshape(2)
        for item in recent
    ])
    yaws = np.unwrap(np.asarray([
        float(item["yaw_rad"]) for item in recent
    ], dtype=np.float64))
    translation_median = np.median(translations, axis=0)
    yaw_median = float(np.median(yaws))
    translation_scatter = np.linalg.norm(translations - translation_median, axis=1)
    yaw_scatter_deg = np.degrees(np.abs(yaws - yaw_median))
    return bool(
        float(np.percentile(translation_scatter, 95.0))
        <= float(max_translation_scatter_mm)
        and float(np.percentile(yaw_scatter_deg, 95.0))
        <= float(max_yaw_scatter_deg)
    )
