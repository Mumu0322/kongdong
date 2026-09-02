"""Final target planning for hole localization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aubo_workbench.paths import TCP_ABSOLUTE_XY_MODEL_DIR


FINAL_TARGET_MODE_GRIPPER = "gripper"
FINAL_TARGET_MODE_NORMAL = "normal"
DEFAULT_FINAL_TARGET_MODE = FINAL_TARGET_MODE_GRIPPER
GRIPPER_BASE_X_OFFSET_MM = 64.0
GRIPPER_BASE_Z_OFFSET_MM = 50.0
FINAL_BASE_Y_AFTER_Z_MM = 0.2
# 与上面的基坐标Y微调融合为一次移动执行，避免最终Z到位后再做第二次单独的+Y移动。
FINAL_TOOL_Y_AFTER_Z_MM = 1.0
# 三孔逐孔安放时，所有低位横移前先抬到该安全余量。
THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM = 60.0

# 当前使用的 ChArUco 9 点 XY 仿射模型；不启用工作区限制。
CHARUCO_XY_MODEL_MATRIX = np.array([
    [1.0008588303603987, -0.0011632075996384716],
    [-0.0013284394231714038, 0.999581433830417],
], dtype=np.float64)
CHARUCO_XY_MODEL_BIAS_MM = np.array([0.05229713949213546, 3.1959138367376676], dtype=np.float64)
CHARUCO_XY_MODEL_SOURCE = Path(
    TCP_ABSOLUTE_XY_MODEL_DIR / "charuco-tcp-xy-20260727_174559" / "report.json"
)


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


def apply_final_point_base_offsets(
    point_base: np.ndarray,
    delta_x_mm: float = GRIPPER_BASE_X_OFFSET_MM,
    delta_z_mm: float = GRIPPER_BASE_Z_OFFSET_MM,
) -> np.ndarray:
    """在规划最终运动前，先对最终点施加基坐标 X/Z 偏移。"""
    target = np.asarray(point_base, dtype=np.float64).reshape(3).copy()
    target[0] += float(delta_x_mm)
    target[2] += float(delta_z_mm)
    return target


def final_point_offsets_for_mode(mode: str) -> tuple[float, float]:
    """返回最终点在基坐标 X/Z 方向的偏置，单位 mm。"""
    normalized = str(mode).strip().lower()
    if normalized == FINAL_TARGET_MODE_GRIPPER:
        return GRIPPER_BASE_X_OFFSET_MM, GRIPPER_BASE_Z_OFFSET_MM
    if normalized == FINAL_TARGET_MODE_NORMAL:
        return 0.0, 0.0
    raise ValueError(
        f"未知最终点模式：{mode!r}；可选模式为 "
        f"{FINAL_TARGET_MODE_GRIPPER!r} 或 {FINAL_TARGET_MODE_NORMAL!r}"
    )


def plan_final_tcp_base_y_trim(T_base_tcp: np.ndarray, delta_y_mm: float = FINAL_BASE_Y_AFTER_Z_MM) -> np.ndarray:
    """保持当前 TCP 的 X、Z 和姿态，仅沿基坐标 Y 做最终微调。"""
    target = np.asarray(T_base_tcp, dtype=np.float64).copy()
    target[1, 3] += float(delta_y_mm)
    return target


def plan_final_tcp_combined_y_trim(
    T_base_tcp: np.ndarray,
    base_delta_y_mm: float = FINAL_BASE_Y_AFTER_Z_MM,
    tool_delta_y_mm: float = FINAL_TOOL_Y_AFTER_Z_MM,
) -> np.ndarray:
    """保持当前 TCP 姿态，把基坐标+Y微调和工具系+Y微调合并成一次平移执行。

    两次平移都不改变姿态（旋转矩阵不变），所以可以直接把基坐标Y分量和
    工具系Y轴（当前旋转矩阵第2列，在base系下的方向）分量相加成一个
    位移向量，一次性移动到位，避免拆成两次串行移动。
    """
    T_base_tcp = np.asarray(T_base_tcp, dtype=np.float64)
    target = T_base_tcp.copy()
    tool_y_axis_base = T_base_tcp[:3, 1]
    delta_base_mm = np.array([0.0, float(base_delta_y_mm), 0.0]) + tool_y_axis_base * float(tool_delta_y_mm)
    target[:3, 3] += delta_base_mm
    return target
