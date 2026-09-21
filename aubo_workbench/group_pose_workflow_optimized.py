"""优化后的位姿规划模块 - 改进法向融合和位姿计算

主要改进：
1. 使用PCA进行鲁棒的法向融合，替代简单中位数
2. 增加法向质量检查和异常值过滤
3. 优化组中心计算，考虑深度权重
"""

from __future__ import annotations
from typing import Any
import numpy as np


def _fuse_normals_robust(normals: list[np.ndarray]) -> np.ndarray:
    """基于PCA的鲁棒法向融合，对异常值更具抵抗力。

    Args:
        normals: 法向量列表，每个都应该是归一化的3D向量

    Returns:
        融合后的单位法向量

    算法说明：
    - 1-2个法向：使用中位数
    - 3+个法向：使用PCA提取主方向，更鲁棒
    """
    if not normals:
        raise ValueError("法向融合至少需要一个输入")

    if len(normals) == 1:
        return _unit(normals[0], "single normal")

    # 确保所有法向在同一半球（与第一个法向夹角<90度）
    reference = _unit(normals[0], "reference normal")
    aligned = []
    for normal in normals:
        unit_n = _unit(normal, "normal")
        aligned.append(unit_n if float(unit_n @ reference) >= 0.0 else -unit_n)

    aligned_array = np.array(aligned, dtype=np.float64)

    # 如果只有2个法向，直接用中位数
    if len(aligned) == 2:
        fused = np.median(aligned_array, axis=0)
        return _unit(fused, "fused normal")

    # 3+个法向：使用PCA提取主方向
    # 计算协方差矩阵的最大特征向量作为主方向
    cov_matrix = aligned_array.T @ aligned_array
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 最大特征值对应的特征向量就是主方向
    principal_direction = eigenvectors[:, -1]

    # 确保方向与参考一致
    if float(principal_direction @ reference) < 0.0:
        principal_direction = -principal_direction

    return _unit(principal_direction, "pca fused normal")


def _filter_outlier_normals(
    normals: list[np.ndarray],
    max_angle_deg: float = 15.0,
) -> tuple[list[np.ndarray], list[int]]:
    """过滤明显偏离的异常法向。

    Args:
        normals: 法向量列表
        max_angle_deg: 与中位数法向的最大允许夹角

    Returns:
        (过滤后的法向列表, 被移除的索引列表)
    """
    if len(normals) <= 2:
        return list(normals), []

    # 先用中位数估计大致方向
    reference = _unit(normals[0], "reference")
    aligned = [
        n if float(_unit(n, "n") @ reference) >= 0.0 else -_unit(n, "n")
        for n in normals
    ]
    median_normal = _unit(np.median(np.array(aligned), axis=0), "median normal")

    # 计算每个法向与中位数的夹角
    angles = []
    for normal in normals:
        unit_n = _unit(normal, "normal")
        # 确保同半球
        if float(unit_n @ median_normal) < 0.0:
            unit_n = -unit_n
        angle = np.degrees(np.arccos(np.clip(float(unit_n @ median_normal), -1.0, 1.0)))
        angles.append(angle)

    # 过滤超出阈值的法向
    max_angle = float(max_angle_deg)
    filtered = []
    removed_indices = []

    for i, (normal, angle) in enumerate(zip(normals, angles)):
        if angle <= max_angle:
            filtered.append(normal)
        else:
            removed_indices.append(i)

    # 如果过滤掉太多，保留全部（可能是阈值设置问题）
    if len(filtered) < max(2, len(normals) // 2):
        return list(normals), []

    return filtered, removed_indices


def _plan_batch_coarse_group_pose_optimized(
    selected_holes: list[dict[str, Any]],
    current_tcp: np.ndarray,
    handeye: Any,
    fixed_rz_rad: float,
    intrinsics: Any,
    target_height_mm: float,
    view_margin_px: float,
    *,
    use_robust_normal_fusion: bool = True,
    filter_outlier_normals: bool = True,
    normal_outlier_threshold_deg: float = 15.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """优化后的批量粗定位位姿规划。

    相比原版的改进：
    1. 可选的异常法向过滤
    2. 可选的鲁棒PCA法向融合
    3. 更详细的诊断信息
    """
    if not selected_holes:
        raise RuntimeError("批量粗定位规划缺少选中孔")

    # 收集所有选中孔的初始位置和法向
    points_base = []
    normals_base = []
    hole_ids = []

    for hole in selected_holes:
        hole_ids.append(int(hole["hole_id"]))
        point = np.asarray(hole["initial_center_base_mm"], dtype=np.float64).reshape(3)
        normal = np.asarray(
            hole.get(
                "planning_normal_base",
                hole.get(
                    "initial_shared_plane_normal_base",
                    hole["initial_plane_normal_base"],
                ),
            ),
            dtype=np.float64,
        ).reshape(3)
        points_base.append(point)
        normals_base.append(normal)

    points_base = np.asarray(points_base, dtype=np.float64)
    normals_base_list = [_unit(n, f"hole {hole_id} normal") for n, hole_id in zip(normals_base, hole_ids)]

    # 使用第一个孔的法向作为参考，对齐所有法向到同一半球
    reference_normal = normals_base_list[0]
    aligned_normals = []
    for normal in normals_base_list:
        aligned_normals.append(
            normal if float(normal @ reference_normal) >= 0.0 else -normal
        )

    # 可选：过滤异常法向
    outlier_info = {"enabled": False}
    if filter_outlier_normals and len(aligned_normals) > 2:
        filtered_normals, removed_indices = _filter_outlier_normals(
            aligned_normals, max_angle_deg=normal_outlier_threshold_deg
        )
        outlier_info = {
            "enabled": True,
            "original_count": len(aligned_normals),
            "filtered_count": len(filtered_normals),
            "removed_indices": removed_indices,
            "removed_hole_ids": [hole_ids[i] for i in removed_indices] if removed_indices else [],
            "threshold_deg": float(normal_outlier_threshold_deg),
        }
        if filtered_normals:
            aligned_normals = filtered_normals

    # 计算组平均法向
    if use_robust_normal_fusion and len(aligned_normals) >= 3:
        group_normal = _fuse_normals_robust(aligned_normals)
        fusion_method = "pca_robust"
    else:
        group_normal = _unit(np.median(np.array(aligned_normals), axis=0), "batch coarse group normal")
        fusion_method = "median"

    # 计算每个法向与组法向的夹角，用于诊断
    normal_deviations = []
    for i, normal in enumerate(aligned_normals):
        angle = np.degrees(np.arccos(np.clip(float(normal @ group_normal), -1.0, 1.0)))
        normal_deviations.append({
            "hole_id": hole_ids[i],
            "deviation_deg": float(angle),
        })

    # 计算目标相机姿态（固定RZ）
    R_tcp, rotation_info = _tcp_rotation_with_fixed_rz_for_camera_axis(
        -group_normal,
        current_tcp,
        handeye.T_tcp_rgb_camera,
        fixed_rz_rad,
    )
    R_base_camera = R_tcp @ np.asarray(handeye.T_tcp_rgb_camera, dtype=np.float64)[:3, :3]

    # 在目标相机坐标系下计算选中孔的包围盒中心
    points_camera_origin = (R_base_camera.T @ points_base.T).T

    # 优化：使用加权中心，深度（Z）较小的孔权重更高
    # 因为它们更可能是主要特征
    z_values = points_camera_origin[:, 2]
    z_median = np.median(z_values)
    weights = np.exp(-0.5 * ((z_values - z_median) / (np.std(z_values) + 1.0))**2)
    weights = weights / np.sum(weights)

    weighted_center_camera = np.sum(points_camera_origin * weights[:, np.newaxis], axis=0)

    # 也计算简单的包围盒中心用于对比
    bbox_center_camera = np.array([
        0.5 * (float(np.min(points_camera_origin[:, 0])) + float(np.max(points_camera_origin[:, 0]))),
        0.5 * (float(np.min(points_camera_origin[:, 1])) + float(np.max(points_camera_origin[:, 1]))),
        float(np.median(points_camera_origin[:, 2])),
    ])

    # 使用加权中心作为主要方案
    group_camera_point = weighted_center_camera
    group_point_base = R_base_camera @ group_camera_point

    # 规划共同观察位姿
    target, pose_geometry = _plan_hole_tcp_pose_fixed_rz(
        group_point_base,
        group_normal,
        current_tcp,
        handeye.T_tcp_rgb_camera,
        fixed_rz_rad=fixed_rz_rad,
        camera_height_mm=float(target_height_mm),
    )

    T_base_camera_target = camera_transform(target, handeye.T_tcp_rgb_camera)

    # 验证所有孔是否在视野内
    width = float(getattr(intrinsics, "width", 0))
    height = float(getattr(intrinsics, "height", 0))
    if width <= 0.0 or height <= 0.0:
        raise RuntimeError("RGB内参缺少有效图像宽高，无法检查批量粗定位共同位姿的视野范围")

    projected_holes = {}
    out_of_view = []
    margin = float(view_margin_px)
    projection_distances = []

    for hole in selected_holes:
        hole_id = int(hole["hole_id"])
        point_base = np.asarray(hole["initial_center_base_mm"], dtype=np.float64).reshape(3)
        try:
            projected = _project_base_point_to_pixel(
                point_base,
                T_base_camera_target,
                intrinsics,
            )
            projected_holes[hole_id] = projected

            # 计算到画面中心的距离
            center_px = np.array([width / 2.0, height / 2.0])
            distance = float(np.linalg.norm(projected - center_px))
            projection_distances.append({
                "hole_id": hole_id,
                "distance_from_center_px": distance,
            })

            # 检查是否在视野内（包含边缘余量）
            if (not np.isfinite(projected).all()
                or float(projected[0]) < margin
                or float(projected[0]) > width - margin
                or float(projected[1]) < margin
                or float(projected[1]) > height - margin):
                out_of_view.append(hole_id)
        except Exception:
            out_of_view.append(hole_id)

    if out_of_view:
        raise RuntimeError(
            f"批量粗定位：选中孔无法在共同{float(target_height_mm):g}mm位姿下全部保持在视野内："
            f"{out_of_view}；"
            "请减少数量或选择更紧凑的一组孔"
        )

    # 计算投影包围盒
    projected_values = np.asarray(list(projected_holes.values()), dtype=np.float64)
    bbox_min = np.min(projected_values, axis=0)
    bbox_max = np.max(projected_values, axis=0)
    group_center_px = 0.5 * (bbox_min + bbox_max)

    # 计算投影范围占画面的比例
    projection_span_x = float(bbox_max[0] - bbox_min[0])
    projection_span_y = float(bbox_max[1] - bbox_min[1])
    projection_span_ratio_x = projection_span_x / width
    projection_span_ratio_y = projection_span_y / height

    return target, {
        "hole_order": [int(hole["hole_id"]) for hole in selected_holes],
        "group_point_base_mm": group_point_base,
        "group_normal_toward_camera_base": group_normal,
        "projected_holes_px": projected_holes,
        "group_bbox_px": [bbox_min[0], bbox_min[1], bbox_max[0], bbox_max[1]],
        "group_center_px": group_center_px,
        "view_margin_px": margin,
        "target_height_mm": float(target_height_mm),
        "target_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(target),
        "target_tcp_transform_mm": target,
        "pose_geometry": pose_geometry,
        "rotation_info": rotation_info,
        # 新增的诊断信息
        "optimization_info": {
            "normal_fusion_method": fusion_method,
            "outlier_filtering": outlier_info,
            "normal_deviations": normal_deviations,
            "max_normal_deviation_deg": float(max(d["deviation_deg"] for d in normal_deviations)) if normal_deviations else 0.0,
            "mean_normal_deviation_deg": float(np.mean([d["deviation_deg"] for d in normal_deviations])) if normal_deviations else 0.0,
            "projection_distances": projection_distances,
            "projection_span_ratio": {
                "x": float(projection_span_ratio_x),
                "y": float(projection_span_ratio_y),
                "max": float(max(projection_span_ratio_x, projection_span_ratio_y)),
            },
            "weighted_vs_bbox_center_delta_mm": float(np.linalg.norm(
                (R_base_camera @ weighted_center_camera) - (R_base_camera @ bbox_center_camera)
            )),
        },
    }


def _unit(value: Any, name: str) -> np.ndarray:
    """归一化向量（需要从其他模块导入或定义）"""
    array = np.asarray(value, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(array))
    if length < 1e-9:
        raise ValueError(f"{name}无法归一化")
    return array / length


# 需要从其他模块导入的函数（示意）
# from aubo_workbench.hole_localization_planning import _tcp_rotation_with_fixed_rz_for_camera_axis
# from aubo_workbench.hole_localization_planning import _plan_hole_tcp_pose_fixed_rz
# from aubo_workbench.hole_localization_planning import _project_base_point_to_pixel
# from aubo_workbench.robot import camera_transform, transform_to_sdk_pose_m_rad
