"""Final target planning for hole localization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aubo_workbench.paths import TCP_XY_MODEL_PATH


# 旧版最终点到位后还会追加基坐标Y和工具系Y微调。该补偿已经取消；
# 保留零值常量和规划函数仅用于兼容历史脚本/报告，任何调用都不会再改变位姿。
FINAL_BASE_Y_AFTER_Z_MM = 0.0
FINAL_TOOL_Y_AFTER_Z_MM = 0.0
# 三孔逐孔安放时，所有低位横移前先抬到该安全余量。
THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM = 60.0

# 新副本默认没有历史补偿。只有通过环境变量或 current.json 显式提供的
# report.json 才会加载矩阵；文件不存在、损坏或模型字段不完整时保留单位
# 变换和零偏置用于离线规划，但不会被允许运动流程当成有效补偿。
CHARUCO_XY_MODEL_MATRIX = np.eye(2, dtype=np.float64)
CHARUCO_XY_MODEL_BIAS_MM = np.zeros(2, dtype=np.float64)
CHARUCO_XY_MODEL_SOURCE = Path(TCP_XY_MODEL_PATH)


def _load_tcp_xy_model() -> tuple[np.ndarray, np.ndarray, bool]:
    path = Path(TCP_XY_MODEL_PATH)
    if not path.is_file():
        return CHARUCO_XY_MODEL_MATRIX, CHARUCO_XY_MODEL_BIAS_MM, False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = payload.get("model", payload)
        matrix = np.asarray(model["matrix_2x2"], dtype=np.float64)
        bias = np.asarray(model["bias_mm"], dtype=np.float64).reshape(2)
        if matrix.shape != (2, 2) or not np.isfinite(matrix).all() or not np.isfinite(bias).all():
            raise ValueError("TCP-XY 模型数值无效")
        return matrix, bias, True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return CHARUCO_XY_MODEL_MATRIX, CHARUCO_XY_MODEL_BIAS_MM, False


CHARUCO_XY_MODEL_MATRIX, CHARUCO_XY_MODEL_BIAS_MM, CHARUCO_XY_MODEL_READY = _load_tcp_xy_model()


def plan_final_tcp_xy(T_base_tcp: np.ndarray, hole_center_base: np.ndarray,
                      xy_offset_mm: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """只调整当前 TCP 的基坐标 XY，姿态（含 yaw）和 Z 保持不变。

    默认使用 ChArUco 9 点仿射模型；显式提供固定偏置时覆盖该模型。
    """
    target = np.asarray(T_base_tcp, dtype=np.float64).copy()
    visual_xy = np.asarray(hole_center_base, dtype=np.float64)[:2]
    if xy_offset_mm is None:
        target[:2, 3] = CHARUCO_XY_MODEL_MATRIX @ visual_xy + CHARUCO_XY_MODEL_BIAS_MM
    else:
        target[:2, 3] = visual_xy + np.asarray(xy_offset_mm, dtype=np.float64)
    return target, np.asarray(T_base_tcp, dtype=np.float64)[:3, 3].copy()


def plan_final_tcp_base_z(T_base_tcp: np.ndarray, hole_center_base: np.ndarray) -> np.ndarray:
    """保持当前 TCP 的 XY 和姿态，仅令基坐标 Z 等于孔中心基坐标 Z。"""
    target = np.asarray(T_base_tcp, dtype=np.float64).copy()
    target[2, 3] = float(np.asarray(hole_center_base, dtype=np.float64).reshape(3)[2])
    return target


def compose_batch_fine_xy_with_coarse_z(
    fine_point_base: np.ndarray,
    coarse_point_base: np.ndarray,
) -> np.ndarray:
    """共享精定位只提供XY；最终点Z继续采用该孔的粗定位结果。"""
    target = np.asarray(fine_point_base, dtype=np.float64).reshape(3).copy()
    target[2] = float(np.asarray(coarse_point_base, dtype=np.float64).reshape(3)[2])
    return target


def _compose_batch_fine_joint_xy_with_tilt(
    naive_point_base: np.ndarray,
    tilted_point_base: np.ndarray,
    joint_point_base: np.ndarray,
    *,
    local_residual_weight: float,
    local_residual_limit_mm: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """将共享联合XY作为主结果，并保留受限的逐孔残差和倾斜纠偏。

    ``naive_point_base`` 是当前孔单独由精拍圆心反投影得到的点，
    ``tilted_point_base`` 是经过既有倾斜圆心纠偏后的点，
    ``joint_point_base`` 是同一共享260 mm视野内多孔联合求解的XY点。
    联合结果只负责XY；Z由调用方继续替换为该孔粗定位Z。这里不涉及
    ChArUco，ChArUco仍在最终 ``plan_final_tcp_xy`` 中统一执行一次。
    """
    naive = np.asarray(naive_point_base, dtype=np.float64).reshape(3).copy()
    tilted = np.asarray(tilted_point_base, dtype=np.float64).reshape(3).copy()
    joint = np.asarray(joint_point_base, dtype=np.float64).reshape(3).copy()
    if (
        not np.isfinite(naive).all()
        or not np.isfinite(tilted).all()
        or not np.isfinite(joint).all()
    ):
        raise ValueError("共享联合XY输入包含非有限坐标")

    weight = min(1.0, max(0.0, float(local_residual_weight)))
    limit = max(0.0, float(local_residual_limit_mm))
    local_residual = naive[:2] - joint[:2]
    residual_norm = float(np.linalg.norm(local_residual))
    if residual_norm > limit > 0.0:
        local_residual_clipped = local_residual * (limit / residual_norm)
    elif limit <= 0.0:
        local_residual_clipped = np.zeros(2, dtype=np.float64)
    else:
        local_residual_clipped = local_residual.copy()
    local_residual_used = weight * local_residual_clipped
    tilt_delta = tilted[:2] - naive[:2]

    result = tilted.copy()
    result[:2] = joint[:2] + local_residual_used + tilt_delta
    return result, {
        "joint_xy_base_mm": joint[:2].copy(),
        "local_residual_base_mm": local_residual.copy(),
        "local_residual_norm_mm": residual_norm,
        "local_residual_clipped_base_mm": local_residual_clipped.copy(),
        "local_residual_used_base_mm": local_residual_used.copy(),
        "local_residual_weight": weight,
        "local_residual_limit_mm": limit,
        "tilt_delta_base_mm": tilt_delta.copy(),
    }


def fuse_batch_fine_xy_with_pointcloud_prior(
    visual_point_base: np.ndarray,
    pointcloud_point_base: np.ndarray,
    *,
    pointcloud_weight: float,
    max_correction_mm: float,
    agreement_gate_mm: float,
    joint_residual_mm: float | None = None,
    max_joint_residual_mm: float | None = None,
    adaptive_weight: bool = False,
    minimum_weight_ratio: float = 0.4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """用粗拍点云支撑中心对共享精定位XY做有门控的有限融合。

    这里的 ``pointcloud_point_base`` 是粗拍圆心射线与多帧局部点云融合
    平面的交点，不是对不完整点云点集直接求质心。只有粗、精两个中心的
    XY差异不超过一致性门限时才施加修正；修正量同时受权重和绝对限幅
    约束。启用自适应权重时，孔级联合残差越小表示视觉联合越稳定，点云
    权重越低；没有联合残差的直接视觉结果也使用保守下限权重。发生冲突
    时保留精定位结果，让报告显式暴露冲突而不是静默平均。
    """
    visual = np.asarray(visual_point_base, dtype=np.float64).reshape(3).copy()
    pointcloud = np.asarray(pointcloud_point_base, dtype=np.float64).reshape(3).copy()
    if not np.isfinite(visual).all() or not np.isfinite(pointcloud).all():
        raise ValueError("共享精定位点云融合输入包含非有限坐标")

    configured_weight = min(1.0, max(0.0, float(pointcloud_weight)))
    min_weight_ratio = min(1.0, max(0.0, float(minimum_weight_ratio)))
    adaptive_weight_ratio = 1.0
    adaptive_weight_reason = "disabled"
    minimum_weight_joint_residual_ratio = 0.30
    full_weight_joint_residual_ratio = 0.50
    if adaptive_weight:
        if joint_residual_mm is not None and max_joint_residual_mm is not None:
            max_joint_residual = float(max_joint_residual_mm)
            joint_residual = float(joint_residual_mm)
            if np.isfinite(max_joint_residual) and max_joint_residual > 0.0:
                normalized_joint_residual = min(
                    1.0, max(0.0, joint_residual / max_joint_residual),
                )
                ramp = (
                    (normalized_joint_residual - minimum_weight_joint_residual_ratio)
                    / (
                        full_weight_joint_residual_ratio
                        - minimum_weight_joint_residual_ratio
                    )
                )
                ramp = min(1.0, max(0.0, ramp))
                adaptive_weight_ratio = (
                    min_weight_ratio + (1.0 - min_weight_ratio) * ramp
                )
                adaptive_weight_reason = "scaled_by_joint_residual"
            else:
                adaptive_weight_ratio = min_weight_ratio
                adaptive_weight_reason = "invalid_joint_gate_uses_minimum"
        else:
            adaptive_weight_ratio = min_weight_ratio
            adaptive_weight_reason = "missing_joint_residual_uses_minimum"
    weight = configured_weight * adaptive_weight_ratio
    correction_limit = max(0.0, float(max_correction_mm))
    agreement_gate = max(0.0, float(agreement_gate_mm))
    delta = pointcloud[:2] - visual[:2]
    disagreement = float(np.linalg.norm(delta))
    details: dict[str, Any] = {
        "method": "guarded_weighted_pointcloud_prior",
        "visual_xy_base_mm": visual[:2].copy(),
        "pointcloud_xy_base_mm": pointcloud[:2].copy(),
        "pointcloud_definition": (
            "coarse_yolo_center_ray_intersection_with_fused_local_pointcloud_plane"
        ),
        "pointcloud_minus_visual_xy_mm": delta.copy(),
        "disagreement_mm": disagreement,
        "agreement_gate_mm": agreement_gate,
        "configured_pointcloud_weight": configured_weight,
        "pointcloud_weight": weight,
        "adaptive_weight_enabled": bool(adaptive_weight),
        "adaptive_weight_ratio": adaptive_weight_ratio,
        "adaptive_weight_reason": adaptive_weight_reason,
        "minimum_weight_ratio": min_weight_ratio,
        "minimum_weight_joint_residual_ratio": (
            minimum_weight_joint_residual_ratio
        ),
        "full_weight_joint_residual_ratio": full_weight_joint_residual_ratio,
        "max_correction_mm": correction_limit,
        "joint_residual_mm": (
            None if joint_residual_mm is None else float(joint_residual_mm)
        ),
        "max_joint_residual_mm": (
            None if max_joint_residual_mm is None else float(max_joint_residual_mm)
        ),
        "applied": False,
    }

    # 联合结果存在时，点云融合必须继承孔级联合残差门。否则可能出现
    # “联合结果已经处于门限边缘，点云又继续把最终点拉向粗中心”的二次偏移。
    if max_joint_residual_mm is not None:
        max_joint_residual = float(max_joint_residual_mm)
        if not np.isfinite(max_joint_residual) or max_joint_residual <= 0.0:
            raise ValueError("共享精定位点云融合的联合残差门限无效")
        if joint_residual_mm is None:
            details.update({
                "status": "rejected_missing_joint_residual",
                "reason": "joint_result_has_no_per_hole_residual",
                "correction_xy_mm": np.zeros(2, dtype=np.float64),
                "correction_norm_mm": 0.0,
                "result_xy_base_mm": visual[:2].copy(),
            })
            return visual, details
        joint_residual = float(joint_residual_mm)
        if not np.isfinite(joint_residual) or joint_residual > max_joint_residual:
            details.update({
                "status": "rejected_joint_residual",
                "reason": (
                    f"joint_residual_{joint_residual:.3f}mm_"
                    f"exceeds_{max_joint_residual:.3f}mm"
                ),
                "correction_xy_mm": np.zeros(2, dtype=np.float64),
                "correction_norm_mm": 0.0,
                "result_xy_base_mm": visual[:2].copy(),
            })
            return visual, details

    if disagreement > agreement_gate:
        details.update({
            "status": "rejected_disagreement",
            "reason": (
                f"coarse_fine_xy_disagreement_{disagreement:.3f}mm_"
                f"exceeds_{agreement_gate:.3f}mm"
            ),
            "correction_xy_mm": np.zeros(2, dtype=np.float64),
            "correction_norm_mm": 0.0,
            "result_xy_base_mm": visual[:2].copy(),
        })
        return visual, details

    correction = weight * delta
    correction_norm = float(np.linalg.norm(correction))
    if correction_norm > correction_limit > 0.0:
        correction = correction * (correction_limit / correction_norm)
        correction_norm = correction_limit
    elif correction_limit <= 0.0:
        correction = np.zeros(2, dtype=np.float64)
        correction_norm = 0.0

    result = visual.copy()
    result[:2] += correction
    applied = bool(correction_norm > 1.0e-12)
    details.update({
        "status": "applied" if applied else "accepted_no_correction",
        "reason": None if applied else "zero_weight_or_zero_correction_limit",
        "applied": applied,
        "correction_xy_mm": correction.copy(),
        "correction_norm_mm": correction_norm,
        "correction_limited": bool(
            weight * disagreement > correction_limit > 0.0
        ),
        "result_xy_base_mm": result[:2].copy(),
    })
    return result, details


def plan_final_tcp_base_y_trim(T_base_tcp: np.ndarray, delta_y_mm: float = FINAL_BASE_Y_AFTER_Z_MM) -> np.ndarray:
    """兼容旧接口；默认不再施加最终基坐标 Y 微调。"""
    target = np.asarray(T_base_tcp, dtype=np.float64).copy()
    target[1, 3] += float(delta_y_mm)
    return target


def plan_final_tcp_combined_y_trim(
    T_base_tcp: np.ndarray,
    base_delta_y_mm: float = FINAL_BASE_Y_AFTER_Z_MM,
    tool_delta_y_mm: float = FINAL_TOOL_Y_AFTER_Z_MM,
) -> np.ndarray:
    """兼容旧接口；默认不再施加最终 Y 微调。

    只有旧调用者显式传入非零参数时才会生成对应位移。
    """
    T_base_tcp = np.asarray(T_base_tcp, dtype=np.float64)
    target = T_base_tcp.copy()
    tool_y_axis_base = T_base_tcp[:3, 1]
    delta_base_mm = np.array([0.0, float(base_delta_y_mm), 0.0]) + tool_y_axis_base * float(tool_delta_y_mm)
    target[:3, 3] += delta_base_mm
    return target
