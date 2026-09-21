"""Robot motion and live pose refinement for grouped coarse localization."""

from __future__ import annotations
import numpy as np
from aubo_workbench.coarse_surface import save_surface_diagnostic
from aubo_workbench.geometry import matrix_to_rpy_zyx, rotx, roty, rotz

from aubo_workbench.localization_errors import LocalizationHardwareError
from aubo_workbench.hole_localization_models import artifact_measure, visual_measure

def _move_to_sequential_coarse_pose(
    hole_id: int, order: int, current_tcp: np.ndarray, target: np.ndarray,
    args: Any, motion_session: Any, pose_session: Any,
) -> np.ndarray:
    """将当前TCP安全移动到指定孔的340 mm粗定位位。"""
    if order == 1:
        return _confirm_and_move_line(
            f"移动到孔{hole_id}上方340 mm粗定位位",
            current_tcp, target, args, motion_session, pose_session,
            "初始选定孔的三维点投影导航；光轴对准该孔初始法向",
            require_confirmation=False,
            motion_profile="transit",
        )

    # 当前已经高于目标安全高度时保持当前Z，避免异常重试时不断抬升。
    safe_z = max(
        float(current_tcp[2, 3]),
        float(target[2, 3]) + THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM,
    )
    actual = np.asarray(current_tcp, dtype=np.float64).copy()
    lift = actual.copy()
    lift[2, 3] = safe_z
    if abs(float(lift[2, 3] - actual[2, 3])) > 0.2:
        actual = _confirm_and_move_line(
            f"孔{hole_id}粗定位前抬升",
            actual, lift, args, motion_session, pose_session,
            "仅修改基坐标Z；离开上一个孔的目标点",
            require_confirmation=False,
            motion_profile="transit",
        )
    high_target = np.asarray(target, dtype=np.float64).copy()
    high_target[2, 3] = safe_z
    actual = _confirm_and_move_line(
        f"移动到孔{hole_id}上方安全高度",
        actual, high_target, args, motion_session, pose_session,
        "安全高度横移并调整到该孔的固定RZ姿态；不下降",
        require_confirmation=False,
        motion_profile="transit",
    )
    return _confirm_and_move_line(
        f"下降到孔{hole_id}上方340 mm粗定位位",
        actual, target, args, motion_session, pose_session,
        "仅基坐标Z下降；XY和孔法向姿态锁定",
        require_confirmation=False,
        motion_profile="transit",
    )


def _move_to_shared_observation_pose(
    label: str,
    current_tcp: np.ndarray,
    target: np.ndarray,
    args: Any,
    motion_session: Any,
    pose_session: Any,
    *,
    target_height_mm: float,
    descent_profile: str,
    steady_timeout_s: float | None = None,
) -> np.ndarray:
    """共享粗/精定位共同观察位的安全移动路径。

    共享模式会先确保TCP处于目标上方的安全高度，再允许XY和姿态变化；
    当前位姿如果已经具有足够净空则保持当前Z，不在异常重试时继续抬升。
    从较低位置进入安全高度时使用纯基坐标Z抬升，并保留配置的安全高度
    余量（建图默认100 mm，普通流程沿用60 mm）；横移完成后，最后至少保留
    ``SHARED_OBSERVATION_MIN_DESCENT_MM`` 的纯基坐标Z下降段。正式采集前
    的steady检查和相机残帧清理仍由调用方保留。
    """
    actual = np.asarray(current_tcp, dtype=np.float64).reshape(4, 4).copy()
    desired = np.asarray(target, dtype=np.float64).reshape(4, 4).copy()
    position_error = float(np.linalg.norm(actual[:3, 3] - desired[:3, 3]))
    rotation_error = _rotation_distance_deg(actual[:3, :3], desired[:3, :3])
    if position_error <= 0.5 and rotation_error <= 0.5:
        return actual

    # 建图阶段使用更保守的可配置Z余量；若当前位姿本身更高，直接保持
    # 当前高度，避免中途退出后重试不断抬升。
    safe_margin_mm = float(
        getattr(args, "map_build_safe_z_margin_mm", THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM)
        if bool(getattr(args, "map_build_coarse_only", False))
        else THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM
    )
    if not np.isfinite(safe_margin_mm) or safe_margin_mm <= 0.0:
        raise ValueError(f"安全横移高度余量必须是正数：{safe_margin_mm}")
    safe_z = max(
        float(actual[2, 3]),
        float(desired[2, 3]) + safe_margin_mm,
    )
    lift_z = safe_z
    lift = actual.copy()
    lift[2, 3] = lift_z
    actual = _confirm_and_move_line(
        f"{label}：先纯Z抬升{SHARED_OBSERVATION_MIN_LIFT_MM:.0f}mm以上",
        actual,
        lift,
        args,
        motion_session,
        pose_session,
        "仅修改基坐标Z；XY和姿态保持不变",
        require_confirmation=False,
        motion_profile="transit",
        steady_timeout_s=steady_timeout_s,
    )

    high_target = desired.copy()
    high_target[2, 3] = lift_z
    # 建图先保持当前姿态完成高位水平移动，再在同一高位单独改变
    # 姿态。这样不会在一次斜向运动里同时叠加XY和RZ变化。
    map_build_path = bool(getattr(args, "map_build_coarse_only", False))
    if map_build_path:
        high_target[:3, :3] = actual[:3, :3]
    actual = _confirm_and_move_line(
        f"{label}：安全高度横移"
        if map_build_path else
        f"{label}：安全高度横移并调整共同视野姿态",
        actual,
        high_target,
        args,
        motion_session,
        pose_session,
        "已完成纯Z抬升；此段只做高位水平横移，不下降",
        require_confirmation=False,
        motion_profile="transit",
        steady_timeout_s=steady_timeout_s,
    )
    if map_build_path and rotation_error > 0.5:
        high_attitude_target = desired.copy()
        high_attitude_target[2, 3] = lift_z
        actual = _confirm_and_move_line(
            f"{label}：高位单独调整共同视野姿态",
            actual,
            high_attitude_target,
            args,
            motion_session,
            pose_session,
            "保持安全横移高度；姿态变化与水平位移分段执行",
            require_confirmation=False,
            motion_profile="transit",
            steady_timeout_s=steady_timeout_s,
        )
    descent_clearance_mm = float(lift_z - desired[2, 3])
    if descent_clearance_mm < SHARED_OBSERVATION_MIN_DESCENT_MM:
        raise RuntimeError(
            f"{label}纯Z下降安全余量不足：{descent_clearance_mm:.3f}mm < "
            f"{SHARED_OBSERVATION_MIN_DESCENT_MM:.3f}mm"
        )
    descent_guard = desired.copy()
    descent_guard[2, 3] = float(desired[2, 3]) + SHARED_OBSERVATION_MIN_DESCENT_MM
    actual = _confirm_and_move_line(
        f"{label}：纯Z下降到目标上方{SHARED_OBSERVATION_MIN_DESCENT_MM:.0f}mm安全位",
        actual,
        descent_guard,
        args,
        motion_session,
        pose_session,
        f"保持XY和姿态不变；先纯Z下降到最终观察位上方"
        f"{SHARED_OBSERVATION_MIN_DESCENT_MM:.0f}mm，禁止斜向接近",
        require_confirmation=False,
        motion_profile=descent_profile,
        steady_timeout_s=steady_timeout_s,
    )
    return _confirm_and_move_line(
        f"{label}：再纯Z下降{SHARED_OBSERVATION_MIN_DESCENT_MM:.0f}mm到"
        f"{float(target_height_mm):.0f}mm共同观察位",
        actual,
        desired,
        args,
        motion_session,
        pose_session,
        f"共同视野孔集合已冻结；保持XY和姿态不变，最后"
        f"纯Z下降{SHARED_OBSERVATION_MIN_DESCENT_MM:.0f}mm到指定RGB相机高度"
        f"{float(target_height_mm):.0f}mm",
        require_confirmation=False,
        motion_profile=descent_profile,
        steady_timeout_s=steady_timeout_s,
    )


def _move_to_shared_coarse_pose(
    group_index: int,
    group_count: int,
    current_tcp: np.ndarray,
    target: np.ndarray,
    args: Any,
    motion_session: Any,
    pose_session: Any,
    *,
    target_height_mm: float = 340.0,
    steady_timeout_s: float | None = None,
) -> np.ndarray:
    """移动到共享粗定位配置的共同观察位。"""
    return _move_to_shared_observation_pose(
        f"共享粗定位：第{int(group_index)}组/{int(group_count)}组",
        current_tcp,
        target,
        args,
        motion_session,
        pose_session,
        target_height_mm=float(target_height_mm),
        descent_profile="transit",
        steady_timeout_s=steady_timeout_s,
    )


def _move_shared_coarse_pose_correction(
    group_index: int,
    group_count: int,
    current_tcp: np.ndarray,
    target: np.ndarray,
    args: Any,
    motion_session: Any,
    pose_session: Any,
    cfg: TwoStageConfig,
    *,
    target_height_mm: float = 340.0,
    steady_timeout_s: float | None = None,
) -> tuple[np.ndarray, str]:
    """执行同一共享粗定位孔组内的受限闭环纠偏。

    调用方已经处于该孔组的观察位，且每次纠偏后都会重新拍摄。因此，在
    单步平移/旋转均通过硬门限时直接从当前位置MoveL，避免每轮纠偏都重复
    抬升和下降。关闭直移开关或有限值超过门限时，回退到原有安全高度路径。
    """
    actual = np.asarray(current_tcp, dtype=np.float64).reshape(4, 4).copy()
    desired = np.asarray(target, dtype=np.float64).reshape(4, 4).copy()
    translation_mm = float(np.linalg.norm(desired[:3, 3] - actual[:3, 3]))
    rotation_deg = float(_rotation_distance_deg(
        actual[:3, :3], desired[:3, :3],
    ))
    correction_allowed, gate_reason = _pose_correction_motion_gate(
        translation_mm, rotation_deg, cfg,
    )
    if gate_reason == "non_finite_pose_correction":
        raise ValueError("共享粗定位同组纠偏目标包含非有限位姿差")

    direct_enabled = bool(cfg.batch_coarse_pose_refine_direct_motion)
    if not direct_enabled or not correction_allowed:
        moved = _move_to_shared_coarse_pose(
            group_index,
            group_count,
            actual,
            desired,
            args,
            motion_session,
            pose_session,
            target_height_mm=float(target_height_mm),
            steady_timeout_s=steady_timeout_s,
        )
        policy = (
            "safe_lift_horizontal_descent_direct_motion_disabled"
            if not direct_enabled
            else f"safe_lift_horizontal_descent_{gate_reason}"
        )
        return moved, policy

    moved = _confirm_and_move_line(
        f"共享粗定位：第{int(group_index)}组/{int(group_count)}组原位小步纠偏",
        actual,
        desired,
        args,
        motion_session,
        pose_session,
        (
            "同一孔组340mm观察位闭环纠偏；"
            f"单步平移{translation_mm:.3f}mm、旋转{rotation_deg:.3f}°；"
            "不抬升，移动后重新拍摄复核"
        ),
        require_confirmation=False,
        motion_profile="approach",
        steady_timeout_s=steady_timeout_s,
    )
    return moved, "direct_same_group_bounded_move_line"


def _plan_batch_coarse_group_pose(
    selected_holes: list[dict[str, Any]],
    current_tcp: np.ndarray,
    handeye: Any,
    fixed_rz_rad: float,
    intrinsics: Any,
    target_height_mm: float,
    view_margin_px: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """为批量粗定位规划能同时看到所有选中孔的配置高度共同位姿。

    用于两阶段流程批量粗定位模式的共同进入位姿规划。
    使用选中孔的初始位置计算组中心和共同观察位姿。
    """
    if not selected_holes:
        raise RuntimeError("批量粗定位规划缺少选中孔")

    # 收集所有选中孔的初始位置和法向
    points_base = []
    normals_base = []
    for hole in selected_holes:
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
    normals_base = np.asarray(normals_base, dtype=np.float64)

    # 使用第一个孔的法向作为参考，对齐所有法向
    reference_normal = _unit(normals_base[0], "batch coarse reference normal")
    aligned_normals = []
    for normal in normals_base:
        normal_unit = _unit(normal, "batch coarse normal")
        # 如果法向与参考法向夹角大于90度，则翻转
        aligned_normals.append(
            normal_unit if float(normal_unit @ reference_normal) >= 0.0 else -normal_unit
        )
    aligned_normals = np.asarray(aligned_normals, dtype=np.float64)

    # 计算组平均法向
    group_normal = _unit(np.median(aligned_normals, axis=0), "batch coarse group normal")
    normal_spread_deg = float(max(
        np.degrees(np.arccos(np.clip(float(normal @ group_normal), -1.0, 1.0)))
        for normal in aligned_normals
    ))

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
    group_camera_point = np.asarray([
        0.5 * (float(np.min(points_camera_origin[:, 0])) + float(np.max(points_camera_origin[:, 0]))),
        0.5 * (float(np.min(points_camera_origin[:, 1])) + float(np.max(points_camera_origin[:, 1]))),
        float(np.median(points_camera_origin[:, 2])),
    ])
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

    return target, {
        "hole_order": [int(hole["hole_id"]) for hole in selected_holes],
        "group_point_base_mm": group_point_base,
        "group_normal_toward_camera_base": group_normal,
        "normal_spread_deg": normal_spread_deg,
        "projected_holes_px": projected_holes,
        "group_bbox_px": [bbox_min[0], bbox_min[1], bbox_max[0], bbox_max[1]],
        "group_center_px": group_center_px,
        "view_margin_px": margin,
        "target_height_mm": float(target_height_mm),
        "target_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(target),
        "target_tcp_transform_mm": target,
        "pose_geometry": pose_geometry,
        "rotation_info": rotation_info,
    }


def _pose_correction_motion_gate(
    translation_mm: float,
    rotation_deg: float,
    cfg: TwoStageConfig,
) -> tuple[bool, str | None]:
    """Check whether one live pose-correction step is small enough to execute.

    The full correction is deliberately handled by ``_bounded_pose_step``.
    This gate remains a hard per-step safety limit so a large correction can
    converge only through repeated fresh observations.
    """
    translation = float(translation_mm)
    rotation = float(rotation_deg)
    if not (np.isfinite(translation) and np.isfinite(rotation)):
        return False, "non_finite_pose_correction"
    if translation > float(cfg.batch_coarse_pose_refine_max_correction_mm) + 1e-9:
        return False, "translation_exceeds_motion_gate"
    if rotation > float(cfg.batch_coarse_pose_refine_max_correction_rotation_deg) + 1e-9:
        return False, "rotation_exceeds_motion_gate"
    return True, None


def _interpolate_pose(
    start: np.ndarray,
    target: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """在两个TCP变换之间做平移线性插值和旋转轴角插值。

    共享粗定位纠偏不能把18 mm/5°的修正一次性下发。这个纯数学辅助函数
    不读取机器人状态，只生成一个受限的中间目标；真正运动由同组纠偏
    执行器在门限内原位MoveL，关闭直移开关或防御性复核超限时走安全高度路径。
    """
    left = np.asarray(start, dtype=np.float64).reshape(4, 4)
    right = np.asarray(target, dtype=np.float64).reshape(4, 4)
    alpha = float(np.clip(float(fraction), 0.0, 1.0))
    result = left.copy()
    result[:3, 3] = left[:3, 3] + alpha * (right[:3, 3] - left[:3, 3])

    relative = left[:3, :3].T @ right[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    if theta <= 1.0e-10:
        result[:3, :3] = left[:3, :3]
        return result

    sine = float(np.sin(theta))
    if abs(sine) > 1.0e-7:
        skew = (relative - relative.T) / (2.0 * sine)
        axis = np.asarray((skew[2, 1], skew[0, 2], skew[1, 0]), dtype=np.float64)
    else:
        # 180°附近用特征向量取旋转轴，避免除以接近0的sin(theta)。
        eigenvalues, eigenvectors = np.linalg.eig(relative)
        index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        axis = np.real(eigenvectors[:, index]).astype(np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis_norm) or axis_norm <= 1.0e-10:
        raise ValueError("无法从共享粗定位纠偏目标提取旋转轴")
    axis = axis / axis_norm
    K = np.asarray([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ], dtype=np.float64)
    partial_theta = theta * alpha
    step_rotation = (
        np.eye(3, dtype=np.float64)
        + np.sin(partial_theta) * K
        + (1.0 - np.cos(partial_theta)) * (K @ K)
    )
    result[:3, :3] = left[:3, :3] @ step_rotation
    return result


def _bounded_pose_step(
    current: np.ndarray,
    target: np.ndarray,
    cfg: TwoStageConfig,
) -> tuple[np.ndarray, float, float, float]:
    """把一次整体纠偏限制到当前配置允许的单步范围内。

    返回 ``(step_target, fraction, step_translation_mm, step_rotation_deg)``。
    fraction 同时受平移和旋转两个门限约束，保证一个运动周期不会因为
    其中一个维度超限而留下不可解释的大动作。
    """
    current_value = np.asarray(current, dtype=np.float64).reshape(4, 4)
    target_value = np.asarray(target, dtype=np.float64).reshape(4, 4)
    translation = float(np.linalg.norm(target_value[:3, 3] - current_value[:3, 3]))
    rotation = float(_rotation_distance_deg(
        current_value[:3, :3], target_value[:3, :3],
    ))
    max_translation = float(cfg.batch_coarse_pose_refine_max_correction_mm)
    max_rotation = float(cfg.batch_coarse_pose_refine_max_correction_rotation_deg)
    fractions = [1.0]
    if translation > max_translation:
        fractions.append(max_translation / translation)
    if rotation > max_rotation:
        fractions.append(max_rotation / rotation)
    fraction = float(np.clip(min(fractions), 0.0, 1.0))
    step_target = _interpolate_pose(current_value, target_value, fraction)
    step_translation = float(np.linalg.norm(
        step_target[:3, 3] - current_value[:3, 3],
    ))
    step_rotation = float(_rotation_distance_deg(
        current_value[:3, :3], step_target[:3, :3],
    ))
    return step_target, fraction, step_translation, step_rotation


def _bounded_shared_fine_pose_adjustment(
    current: np.ndarray,
    planned_target: np.ndarray,
    *,
    max_xy_mm: float,
    max_z_mm: float,
    max_rotation_deg: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """生成共享精定位组内直接调整的受限目标位姿。

    平移分别限制XY和Z；旋转只插值AUBO ZYX欧拉角中的RX/RY，并把RZ
    锁定为当前值。RX/RY使用二分搜索限制实际SO(3)旋转距离，避免欧拉角
    分量虽小但合成旋转超过安全门限。
    """
    actual = np.asarray(current, dtype=np.float64).reshape(4, 4).copy()
    planned = np.asarray(planned_target, dtype=np.float64).reshape(4, 4).copy()
    if not np.isfinite(actual).all() or not np.isfinite(planned).all():
        raise ValueError("共享精定位组内调整包含非有限目标位姿")

    xy_limit = max(0.0, float(max_xy_mm))
    z_limit = max(0.0, float(max_z_mm))
    rotation_limit = max(0.0, float(max_rotation_deg))
    raw_delta = planned[:3, 3] - actual[:3, 3]
    raw_xy_mm = float(np.linalg.norm(raw_delta[:2]))
    xy_scale = 1.0
    if raw_xy_mm > xy_limit > 0.0:
        xy_scale = xy_limit / raw_xy_mm
    elif xy_limit <= 0.0:
        xy_scale = 0.0

    target = actual.copy()
    target[:2, 3] = actual[:2, 3] + raw_delta[:2] * xy_scale
    target[2, 3] = actual[2, 3] + float(np.clip(raw_delta[2], -z_limit, z_limit))

    current_rpy = matrix_to_rpy_zyx(actual[:3, :3])
    planned_rpy = matrix_to_rpy_zyx(planned[:3, :3])

    def wrapped_delta(value: float) -> float:
        return float((value + np.pi) % (2.0 * np.pi) - np.pi)

    tilt_delta = np.asarray([
        wrapped_delta(float(planned_rpy[0] - current_rpy[0])),
        wrapped_delta(float(planned_rpy[1] - current_rpy[1])),
    ], dtype=np.float64)

    def rotation_at(fraction: float) -> np.ndarray:
        rpy = current_rpy.copy()
        rpy[:2] += float(fraction) * tilt_delta
        rpy[2] = current_rpy[2]
        return rotz(float(rpy[2])) @ roty(float(rpy[1])) @ rotx(float(rpy[0]))

    raw_tilt_target = rotation_at(1.0)
    raw_rotation_deg = float(_rotation_distance_deg(
        actual[:3, :3], raw_tilt_target,
    ))
    rotation_fraction = 1.0
    if rotation_limit <= 0.0:
        rotation_fraction = 0.0
    elif raw_rotation_deg > rotation_limit:
        low, high = 0.0, 1.0
        for _ in range(32):
            middle = 0.5 * (low + high)
            middle_rotation = rotation_at(middle)
            distance = float(_rotation_distance_deg(
                actual[:3, :3], middle_rotation,
            ))
            if distance <= rotation_limit:
                low = middle
            else:
                high = middle
        rotation_fraction = low
    target[:3, :3] = rotation_at(rotation_fraction)

    applied_xy_mm = float(np.linalg.norm(target[:2, 3] - actual[:2, 3]))
    applied_z_mm = float(target[2, 3] - actual[2, 3])
    applied_rotation_deg = float(_rotation_distance_deg(
        actual[:3, :3], target[:3, :3],
    ))
    result_rpy = matrix_to_rpy_zyx(target[:3, :3])
    return target, {
        "policy": "bounded_direct_xy_z_rx_ry_with_locked_rz",
        "planned_translation_delta_mm": raw_delta.copy(),
        "planned_xy_mm": raw_xy_mm,
        "planned_z_mm": float(raw_delta[2]),
        "planned_rotation_deg": raw_rotation_deg,
        "max_xy_mm": xy_limit,
        "max_z_mm": z_limit,
        "max_rotation_deg": rotation_limit,
        "xy_scale": xy_scale,
        "rotation_fraction": rotation_fraction,
        "applied_xy_mm": applied_xy_mm,
        "applied_z_mm": applied_z_mm,
        "applied_rotation_deg": applied_rotation_deg,
        "current_rz_rad": float(current_rpy[2]),
        "result_rz_rad": float(result_rpy[2]),
        "rz_locked": bool(abs(wrapped_delta(float(result_rpy[2] - current_rpy[2]))) <= 1.0e-8),
        "translation_clipped": bool(
            raw_xy_mm > xy_limit + 1.0e-9
            or abs(float(raw_delta[2])) > z_limit + 1.0e-9
        ),
        "rotation_clipped": bool(raw_rotation_deg > rotation_limit + 1.0e-9),
    }


def _move_shared_fine_group_adjustment(
    label: str,
    current_tcp: np.ndarray,
    target_tcp: np.ndarray,
    args: Any,
    motion_session: Any,
    pose_session: Any,
    *,
    max_xy_mm: float,
    max_z_mm: float,
    max_rotation_deg: float,
) -> np.ndarray:
    """在当前共享精拍高度附近执行一次受限MoveL，不走抬升/下降路径。"""
    actual = np.asarray(current_tcp, dtype=np.float64).reshape(4, 4).copy()
    desired = np.asarray(target_tcp, dtype=np.float64).reshape(4, 4).copy()
    xy_mm = float(np.linalg.norm(desired[:2, 3] - actual[:2, 3]))
    z_mm = abs(float(desired[2, 3] - actual[2, 3]))
    rotation_deg = float(_rotation_distance_deg(
        actual[:3, :3], desired[:3, :3],
    ))
    if xy_mm > float(max_xy_mm) + 1.0e-6:
        raise ValueError(f"共享精定位组内XY调整超限：{xy_mm:.3f}mm")
    if z_mm > float(max_z_mm) + 1.0e-6:
        raise ValueError(f"共享精定位组内Z调整超限：{z_mm:.3f}mm")
    if rotation_deg > float(max_rotation_deg) + 1.0e-6:
        raise ValueError(f"共享精定位组内姿态调整超限：{rotation_deg:.3f}deg")
    if xy_mm <= 0.05 and z_mm <= 0.05 and rotation_deg <= 0.05:
        return actual
    return _confirm_and_move_line(
        label,
        actual,
        desired,
        args,
        motion_session,
        pose_session,
        (
            "同一共享精定位组内直接重观察；"
            f"XY={xy_mm:.3f}mm，Z={z_mm:.3f}mm，"
            f"RX/RY合成旋转={rotation_deg:.3f}°，RZ锁定；不抬升"
        ),
        require_confirmation=False,
        motion_profile="approach",
    )


def _measured_340_pose_is_map_safe(
    metrics: dict[str, Any] | None,
    observed_hole_count: int,
    expected_hole_count: int,
    cfg: TwoStageConfig,
) -> bool:
    """判断名义340 mm位姿上的实拍结果能否直接作为建图采集位姿。

    初始共同位姿只能由初始选孔几何规划出来；机器人到达该位姿后，
    340 mm现场帧才是更接近正式粗定位的数据。如果这批帧已经覆盖整组孔，
    高度、中心和重投影质量都通过，就没有必要为了追逐一个偏差很大的
    理想姿态而强行移动。该门只允许“原位使用实测340 mm数据”，不会放宽
    后续正式采集自己的质量门，也不会允许一次大幅运动。
    """
    if metrics is None or int(observed_hole_count) < int(expected_hole_count):
        return False
    try:
        height_error = abs(float(metrics["pose_height_error_mm"]))
        current_center_error = float(metrics["current_group_center_error_px"])
        target_center_error = float(metrics["target_group_center_error_px"])
        reprojection_error = float(metrics["reprojection_error_p95_px"])
    except (KeyError, TypeError, ValueError):
        return False
    values = (
        height_error,
        current_center_error,
        target_center_error,
        reprojection_error,
    )
    if not all(np.isfinite(value) for value in values):
        return False
    return bool(
        height_error <= float(cfg.batch_coarse_pose_refine_pose_tolerance_mm)
        and current_center_error <= float(cfg.center_tolerance_px)
        and target_center_error <= float(cfg.center_tolerance_px)
        and reprojection_error <= float(
            cfg.batch_coarse_pose_refine_max_reprojection_error_px
        )
    )


def _save_group_pose_refinement_visualization(
    source_image_path: str | Path | None,
    output_base: Path,
    group_index: int,
    iteration: int,
    group: list[dict[str, Any]],
    expected_before_px: dict[Any, Any],
    observed_px: dict[Any, Any],
    expected_after_px: dict[Any, Any] | None,
    *,
    accepted: bool,
    note: str,
    timing: Any | None = None,
) -> dict[str, str] | None:
    """在现场复核帧上显示名义锚点、实测锚点和纠偏后投影。

    这张图专门用于解释共享组位姿是怎样来的：圆圈是复核开始时的
    投影，叉号是多帧融合后的实测检测中心，方框是按实测点/法向重新
    规划后的预测位置。输出同时保存PNG和高质量JPG，便于直接查看。
    """
    if source_image_path is None:
        return None
    view = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
    if view is None:
        return None
    view = cv2.cvtColor(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    height, width = view.shape[:2]
    expected_after_px = expected_after_px or {}

    def value_for(mapping: dict[Any, Any], hole_id: int) -> np.ndarray | None:
        value = mapping.get(hole_id, mapping.get(str(hole_id)))
        if value is None:
            return None
        try:
            point = np.asarray(value, dtype=np.float64).reshape(2)
        except (TypeError, ValueError):
            return None
        return point if np.isfinite(point).all() else None

    def draw_arrow(left: np.ndarray, right: np.ndarray) -> None:
        left_i = tuple(np.rint(left).astype(int))
        right_i = tuple(np.rint(right).astype(int))
        cv2.arrowedLine(view, left_i, right_i, (255, 255, 255), 6, cv2.LINE_AA, tipLength=0.14)
        cv2.arrowedLine(view, left_i, right_i, (0, 0, 0), 2, cv2.LINE_AA, tipLength=0.14)

    for hole in group:
        hole_id = int(hole["hole_id"])
        before = value_for(expected_before_px, hole_id)
        observed = value_for(observed_px, hole_id)
        after = value_for(expected_after_px, hole_id)
        if before is not None:
            before_i = tuple(np.rint(before).astype(int))
            cv2.circle(view, before_i, 18, (255, 255, 255), 6, cv2.LINE_AA)
            cv2.circle(view, before_i, 18, (0, 0, 0), 2, cv2.LINE_AA)
        if observed is not None:
            observed_i = tuple(np.rint(observed).astype(int))
            cv2.drawMarker(view, observed_i, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 28, 6, cv2.LINE_AA)
            cv2.drawMarker(view, observed_i, (0, 0, 0), cv2.MARKER_TILTED_CROSS, 22, 2, cv2.LINE_AA)
        if after is not None:
            after_i = tuple(np.rint(after).astype(int))
            cv2.rectangle(
                view, (after_i[0] - 13, after_i[1] - 13),
                (after_i[0] + 13, after_i[1] + 13),
                (255, 255, 255), 6, cv2.LINE_AA,
            )
            cv2.rectangle(
                view, (after_i[0] - 13, after_i[1] - 13),
                (after_i[0] + 13, after_i[1] + 13),
                (0, 0, 0), 2, cv2.LINE_AA,
            )
        if before is not None and after is not None:
            draw_arrow(before, after)
        label_point = after if after is not None else observed if observed is not None else before
        if label_point is not None:
            label = f"H{hole_id}"
            point_i = tuple(np.rint(label_point).astype(int))
            _draw_black_text(
                view, label,
                (max(5, min(width - 80, point_i[0] + 18)), max(25, min(height - 8, point_i[1] - 8))),
                scale=0.52, thickness=1,
            )

    panel_height = min(120, max(76, height // 6))
    cv2.rectangle(view, (0, 0), (width - 1, panel_height), (255, 255, 255), -1)
    _draw_black_text(
        view,
        f"SHARED COARSE POSE REFINE G{int(group_index):02d} ITER={int(iteration):02d}",
        (15, 25), scale=0.62, thickness=2,
    )
    _draw_black_text(
        view,
        "CIRCLE=NOMINAL  X=OBSERVED FUSED  BOX=REPLANNED  ARROW=POSE CORRECTION",
        (15, 51), scale=0.43, thickness=1,
    )
    _draw_black_text(
        view,
        f"STATUS={'ACCEPTED' if accepted else 'CHECK'}  {note[:100]}",
        (15, 76), scale=0.45, thickness=1,
    )
    try:
        return _write_visualization_pair(
            view,
            output_base,
            timing=timing,
            artifact_name=f"coarse_pose_refine/{output_base.name}",
        )
    except Exception:
        return None


def _refine_shared_coarse_group_pose(
    group_index: int,
    group_count: int,
    selected_holes: list[dict[str, Any]],
    current_tcp: np.ndarray,
    nominal_target: np.ndarray,
    nominal_geometry: dict[str, Any],
    handeye: Any,
    pipeline: Any,
    align: Any,
    chain: Any,
    model: Any,
    confidence: float,
    cfg: TwoStageConfig,
    intrinsics: Any,
    run_dir: Path,
    timing: TimingRecorder,
    rows: list[dict[str, Any]],
    args: Any,
    motion_session: Any,
    pose_session: Any,
    fixed_rz_rad: float,
) -> dict[str, Any]:
    """复核共享粗定位组的实测几何，并在必要时安全纠偏后再正式采集。

    该函数只由共享粗定位主路径调用。每一轮都在机器人 steady 后丢弃
    预热RGB-D帧，再采集固定数量的稳定帧；只有整组每个孔都通过稳定性、
    点云重投影和规划器门限，才允许把实测几何写入正式粗定位的关联锚点。
    """
    original_tcp = np.asarray(current_tcp, dtype=np.float64).reshape(4, 4).copy()
    original_group = list(selected_holes)
    report: dict[str, Any] = {
        "enabled": bool(cfg.batch_coarse_group_pose_refinement),
        "group_index": int(group_index),
        "group_count": int(group_count),
        "iterations": [],
        "accepted": False,
        "formal_capture_policy": (
            "accepted_refined_pose_only"
            if bool(getattr(args, "map_build_coarse_only", False))
            else "nominal_group_pose_until_refined"
        ),
        "nominal_pose_source": (
            "initial_selection_pointcloud_geometry_plus_handeye"
        ),
        "first_340_capture_role": (
            "post_navigation_live_replan_and_map_candidate"
        ),
        "motion_policy": "bounded_closed_loop_direct_same_group_pose_correction",
        "direct_same_group_motion": bool(
            cfg.batch_coarse_pose_refine_direct_motion
        ),
        "max_step_translation_mm": float(
            cfg.batch_coarse_pose_refine_max_correction_mm
        ),
        "max_step_rotation_deg": float(
            cfg.batch_coarse_pose_refine_max_correction_rotation_deg
        ),
        "max_total_translation_mm": float(
            cfg.batch_coarse_pose_refine_max_total_correction_mm
        ),
        "max_total_rotation_deg": float(
            cfg.batch_coarse_pose_refine_max_total_correction_rotation_deg
        ),
    }
    if not cfg.batch_coarse_group_pose_refinement:
        report["reason"] = "disabled_by_config"
        return {
            "group": original_group,
            "current_tcp": original_tcp,
            "target": nominal_target,
            "geometry": nominal_geometry,
            "report": report,
        }
    if len(original_group) < 2 and not any(h.get("coarse_quality_retry") for h in original_group):
        report["reason"] = "singleton_group_not_shared"
        # 单孔没有可供估计的“共享组位姿偏差”，但它仍然可以在规划位姿
        # 直接作为正式粗定位样本。把这个状态单独标记出来，避免收紧
        # 分组后出现一个合法余组，却被地图写入防线误判成纠偏失败。
        report["map_build_safe"] = True
        report["formal_capture_policy"] = "nominal_singleton_pose_no_shared_refinement"
        return {
            "group": original_group,
            "current_tcp": original_tcp,
            "target": nominal_target,
            "geometry": nominal_geometry,
            "report": report,
        }
    # Offline unit tests and old integrations may pass opaque camera/pose
    # substitutes.  The new closed-loop feature is isolated in that case so it
    # cannot change the old shared-capture behavior.
    if not callable(getattr(pose_session, "read_pose_snapshot", None)):
        report.update({
            "enabled": False,
            "reason": "pose_session_has_no_steady_snapshot_interface",
        })
        return {
            "group": original_group,
            "current_tcp": original_tcp,
            "target": nominal_target,
            "geometry": nominal_geometry,
            "report": report,
        }

    working_group = [dict(hole) for hole in original_group]
    capture_tcp = original_tcp.copy()
    last_refined_group = original_group
    last_target = np.asarray(nominal_target, dtype=np.float64).copy()
    last_geometry = nominal_geometry
    max_iterations = max(1, int(cfg.batch_coarse_pose_refine_max_iterations))
    min_valid = max(1, int(cfg.batch_coarse_pose_refine_min_valid_frames))
    frame_count = max(1, int(cfg.batch_coarse_pose_refine_frames))
    last_live_values: dict[int, dict[str, Any]] = {}
    last_metrics: dict[str, Any] | None = None

    def prepare_stable_capture(iteration: int) -> tuple[np.ndarray, int]:
        with timing.measure(
            f"batch_coarse/group_{int(group_index):02d}/pose_refinement_{int(iteration):02d}/wait_steady",
            group_index=int(group_index), iteration=int(iteration),
        ):
            actual = _wait_robot_steady_before_initial_capture(
                pose_session,
                settle_delay_s=max(0.0, float(cfg.coarse_settle_delay_s)),
            )
        discarded = 0
        discard_count = max(0, int(cfg.batch_coarse_pose_refine_settle_discard_frames))
        with timing.measure(
            f"batch_coarse/group_{int(group_index):02d}/pose_refinement_{int(iteration):02d}/discard_settle_frames",
            target_frames=discard_count, group_index=int(group_index), iteration=int(iteration),
        ):
            with visual_measure(
                timing,
                f"batch_coarse/group_{int(group_index):02d}/pose_refinement_{int(iteration):02d}/frame_acquisition_settle",
                "frame_acquisition",
                target_frames=int(discard_count),
                group_index=int(group_index),
                iteration=int(iteration),
            ):
                for _ in range(discard_count):
                    if get_aligned_frame_bundle(pipeline, align, chain) is not None:
                        discarded += 1
        return np.asarray(actual, dtype=np.float64).reshape(4, 4).copy(), discarded

    for iteration in range(1, max_iterations + 1):
        iteration_report: dict[str, Any] = {
            "iteration": int(iteration),
            "frame_count_requested": int(frame_count),
            "min_valid_frames": int(min_valid),
            "discarded_frame_count": 0,
            "frame_records": [],
            "holes": {},
        }
        try:
            capture_tcp, discarded = prepare_stable_capture(iteration)
            iteration_report["capture_tcp_pose_m_rad"] = transform_to_sdk_pose_m_rad(capture_tcp)
            iteration_report["discarded_frame_count"] = int(discarded)
        except LocalizationHardwareError:
            raise
        except Exception as exc:
            iteration_report["error"] = f"steady_or_discard_failed:{type(exc).__name__}:{exc}"
            report["iterations"].append(iteration_report)
            break

        T_base_camera = camera_transform(capture_tcp, handeye.T_tcp_rgb_camera)
        last_intrinsics = intrinsics
        expected_before = {
            int(hole["hole_id"]): _project_base_point_to_pixel(
                _coarse_expected_point_base(hole), T_base_camera, intrinsics,
            )
            for hole in working_group
        }
        hole_observations: dict[int, list[Observation]] = {
            int(hole["hole_id"]): [] for hole in working_group
        }
        latest_overlay_path: Path | None = None
        latest_view: Any = None
        holes_by_id = {int(hole["hole_id"]): hole for hole in working_group}

        with timing.measure(
            f"batch_coarse/group_{int(group_index):02d}/pose_refinement_{int(iteration):02d}/capture",
            hole_count=len(working_group), target_frames=frame_count,
            group_index=int(group_index), iteration=int(iteration),
        ):
            for frame_index in range(frame_count):
                with visual_measure(
                    timing,
                    f"batch_coarse/group_{int(group_index):02d}/pose_refinement_{int(iteration):02d}/frame_acquisition",
                    "frame_acquisition",
                    frame_index=int(frame_index),
                    group_index=int(group_index),
                    iteration=int(iteration),
                ):
                    bundle = get_aligned_frame_bundle(pipeline, align, chain)
                if bundle is None or bundle.intrinsics is None:
                    for hole_id, anchor in expected_before.items():
                        hole_observations[hole_id].append(Observation(
                            "batch_coarse_pose_refine", frame_index,
                            np.asarray(anchor, dtype=np.float64),
                            error="rgbd_frame_missing",
                        ))
                    iteration_report["frame_records"].append({
                        "frame_index": int(frame_index),
                        "valid": False,
                        "valid_hole_count": 0,
                        "reason": "rgbd_frame_missing",
                    })
                    continue
                last_intrinsics = bundle.intrinsics
                raw_anchors = {
                    str(hole_id): _project_base_point_to_pixel(
                        _coarse_expected_point_base(hole),
                        T_base_camera,
                        bundle.intrinsics,
                    )
                    for hole_id, hole in holes_by_id.items()
                }
                view = bundle.color_bgr.copy()
                valid_hole_count = 0
                hole_records: list[dict[str, Any]] = []
                try:
                    with visual_measure(
                        timing,
                        f"batch_coarse/group_{int(group_index):02d}/pose_refinement_{int(iteration):02d}/yolo_inference",
                        "yolo_inference",
                        frame_index=int(frame_index),
                        group_index=int(group_index),
                        iteration=int(iteration),
                    ):
                        detections = detect(model, bundle.color_bgr, confidence)
                    assignments = _assign_detections_to_projection(
                        detections,
                        raw_anchors,
                        cfg.batch_coarse_pose_refine_tracking_tolerance_px,
                    )
                    detect_error = None
                except Exception as exc:
                    detections = []
                    assignments = {}
                    detect_error = f"detection_failed:{type(exc).__name__}:{exc}"

                for hole_id, hole in holes_by_id.items():
                    anchor = np.asarray(raw_anchors[str(hole_id)], dtype=np.float64)
                    assigned = assignments.get(str(hole_id))
                    if assigned is None:
                        reason = detect_error or "yolo_missing_or_far"
                        hole_observations[hole_id].append(Observation(
                            "batch_coarse_pose_refine", frame_index, anchor,
                            timestamp_ns=bundle.host_timestamp_ns, error=reason,
                        ))
                        hole_records.append({
                            "hole_id": hole_id, "valid": False, "reason": reason,
                        })
                        continue
                    detection = assigned["detection"]
                    center = np.asarray(detection["center"], dtype=np.float64).reshape(2)
                    box = np.asarray(detection["box"], dtype=np.float64).reshape(4)
                    radius = max(float(box[2] - box[0]), float(box[3] - box[1])) / 2.0
                    try:
                        with visual_measure(
                            timing,
                            f"batch_coarse/group_{int(group_index):02d}/pose_refinement_{int(iteration):02d}/pointcloud_geometry_hole_{hole_id:02d}",
                            "pointcloud_processing",
                            frame_index=int(frame_index),
                            group_index=int(group_index),
                            iteration=int(iteration),
                            hole_id=int(hole_id),
                        ):
                            _, plane_info = hole_camera_point(
                                tuple(center.tolist()), bundle.xyz_map_mm,
                                bundle.intrinsics, radius,
                                ray_center_xy=center,
                                ray_center_is_undistorted=False,
                                include_points=True,
                                surface_selection_policy=COARSE_SURFACE_SELECTION_POLICY,
                            )
                            plane = _plane_estimate_from_info(
                                plane_info, f"batch coarse pose refine hole {hole_id} normal",
                            )
                        reasons: list[str] = []
                        if plane.rmse_mm > float(cfg.max_plane_rmse_mm):
                            reasons.append(f"plane_quality:{plane.rmse_mm:.3f}mm")
                        tracking_distance = float(assigned["distance_px"])
                        if tracking_distance > float(
                            cfg.batch_coarse_pose_refine_max_tracking_distance_p95_px
                        ):
                            reasons.append(f"tracking_distance:{tracking_distance:.3f}px")
                        error = ";".join(reasons) if reasons else None
                        hole_observations[hole_id].append(Observation(
                            "batch_coarse_pose_refine", frame_index, center,
                            plane=plane, timestamp_ns=bundle.host_timestamp_ns,
                            error=error, tracking_distance_px=tracking_distance,
                        ))
                        valid = error is None
                        valid_hole_count += int(valid)
                        hole_records.append({
                            "hole_id": hole_id, "valid": valid,
                            "distance_px": tracking_distance,
                            "plane_rmse_mm": float(plane.rmse_mm),
                            "ring_points": int(plane.ring_points),
                            "reason": error,
                        })
                    except Exception as exc:
                        reason = f"pointcloud_failed:{type(exc).__name__}:{exc}"
                        hole_observations[hole_id].append(Observation(
                            "batch_coarse_pose_refine", frame_index, center,
                            timestamp_ns=bundle.host_timestamp_ns,
                            error=reason, tracking_distance_px=float(assigned["distance_px"]),
                        ))
                        hole_records.append({
                            "hole_id": hole_id, "valid": False,
                            "distance_px": float(assigned["distance_px"]),
                            "reason": reason,
                        })

                for hole_id, anchor in raw_anchors.items():
                    anchor_i = tuple(np.rint(np.asarray(anchor)).astype(int))
                    cv2.circle(view, anchor_i, 16, (255, 255, 255), 5, cv2.LINE_AA)
                    cv2.circle(view, anchor_i, 16, (0, 0, 0), 2, cv2.LINE_AA)
                    assigned = assignments.get(hole_id)
                    if assigned is not None:
                        detected_center = tuple(
                            np.rint(np.asarray(assigned["detection"]["center"])).astype(int)
                        )
                        cv2.drawMarker(
                            view, detected_center, (255, 255, 255),
                            cv2.MARKER_TILTED_CROSS, 23, 5, cv2.LINE_AA,
                        )
                        cv2.drawMarker(
                            view, detected_center, (0, 0, 0),
                            cv2.MARKER_TILTED_CROSS, 17, 2, cv2.LINE_AA,
                        )
                    _draw_black_text(
                        view, f"H{hole_id}",
                        (anchor_i[0] + 10, anchor_i[1] - 8), scale=0.48, thickness=1,
                    )
                _draw_black_text(
                    view,
                    f"POSE REFINE G{int(group_index):02d} I{int(iteration):02d} F{int(frame_index):02d} VALID={valid_hole_count}/{len(working_group)}",
                    (15, 28), scale=0.52, thickness=1,
                )
                latest_view = view
                frame_overlay_path = None
                if bool(getattr(cfg, "save_all_capture_overlays", False)):
                    frame_overlay_path = run_dir / (
                        f"batch_coarse_group_{int(group_index):02d}_pose_refine_"
                        f"iter_{int(iteration):02d}_frame_{int(frame_index):02d}.png"
                    )
                    with artifact_measure(
                        timing,
                        "coarse_pose_refine/write_frame_overlay",
                        artifact_kind="coarse_pose_refine_frame_overlay",
                        paths=[str(frame_overlay_path)],
                        group_index=int(group_index),
                        iteration=int(iteration),
                        frame_index=int(frame_index),
                    ):
                        if not cv2.imwrite(str(frame_overlay_path), view):
                            raise RuntimeError(f"粗定位姿态修正叠加图写入失败：{frame_overlay_path}")
                    latest_overlay_path = frame_overlay_path
                iteration_report["frame_records"].append({
                    "frame_index": int(frame_index),
                    "valid": valid_hole_count == len(working_group),
                    "valid_hole_count": int(valid_hole_count),
                    "required_hole_count": len(working_group),
                    "detection_count": len(detections),
                    "holes": hole_records,
                    "overlay_path": (
                        None if frame_overlay_path is None else str(frame_overlay_path)
                    ),
                })

        if latest_view is not None and not bool(
            getattr(cfg, "save_all_capture_overlays", False)
        ):
            latest_overlay_path = run_dir / (
                f"batch_coarse_group_{int(group_index):02d}_pose_refine_"
                f"iter_{int(iteration):02d}_last_frame.png"
            )
            with artifact_measure(
                timing,
                "coarse_pose_refine/write_last_frame_overlay",
                artifact_kind="coarse_pose_refine_last_frame_overlay",
                paths=[str(latest_overlay_path)],
                group_index=int(group_index),
                iteration=int(iteration),
            ):
                if not cv2.imwrite(str(latest_overlay_path), latest_view):
                    raise RuntimeError(f"粗定位姿态修正末帧叠加图写入失败：{latest_overlay_path}")
            if iteration_report["frame_records"]:
                iteration_report["frame_records"][-1]["overlay_path"] = str(
                    latest_overlay_path
                )

        for hid, values in hole_observations.items():
            rows.extend(dict(row, hole_id=int(hid), group_index=int(group_index),
                             refine_iteration=int(iteration))
                        for row in _observation_rows(values))

        live_values: dict[int, dict[str, Any]] = {}
        hole_failures: dict[int, str] = {}
        for hole_id, observations in hole_observations.items():
            diagnostic = next((item for item in reversed(observations) if item.plane is not None), None)
            if diagnostic is not None:
                diagnostic_path = run_dir / f"batch_coarse_group_{group_index:02d}_refine_{iteration:02d}_hole_{hole_id:02d}_surface_diagnostic.npz"
                with artifact_measure(timing, "coarse_pose_refine/write_surface_diagnostic",
                                      artifact_kind="surface_diagnostic", paths=[str(diagnostic_path)]):
                    save_surface_diagnostic(diagnostic_path, diagnostic.plane,
                                            hole_id=hole_id, frame_index=diagnostic.frame_index)
            valid_observations = [
                item for item in observations
                if item.error is None and item.plane is not None
            ]
            try:
                summary = _fuse_coarse(
                    observations,
                    cfg,
                    min_valid_frames=min_valid,
                    max_center_scatter_p95_px=(
                        cfg.batch_coarse_pose_refine_max_center_scatter_p95_px
                    ),
                    max_tracking_distance_p95_px=(
                        cfg.batch_coarse_pose_refine_max_tracking_distance_p95_px
                    ),
                )
                R_base_camera = T_base_camera[:3, :3]
                t_base_camera = T_base_camera[:3, 3]
                point_camera = np.asarray(
                    summary["plane_point_camera_mm"], dtype=np.float64,
                ).reshape(3)
                point_base = R_base_camera @ point_camera + t_base_camera
                normal_camera = _unit(
                    np.asarray(summary["plane_normal_camera"], dtype=np.float64),
                    f"batch coarse pose refine hole {hole_id} camera normal",
                )
                normal_base = _unit(
                    R_base_camera @ normal_camera,
                    f"batch coarse pose refine hole {hole_id} base normal",
                )
                if float(normal_base @ (t_base_camera - point_base)) < 0.0:
                    normal_base = -normal_base
                    normal_camera = -normal_camera
                reprojection = _project_base_point_to_pixel(
                    point_base, T_base_camera, last_intrinsics,
                )
                reprojection_error_px = float(np.linalg.norm(
                    reprojection - np.asarray(summary["center_px"], dtype=np.float64).reshape(2)
                ))
                if reprojection_error_px > float(
                    cfg.batch_coarse_pose_refine_max_reprojection_error_px
                ):
                    raise RuntimeError(
                        f"reprojection_error={reprojection_error_px:.3f}px > "
                        f"{float(cfg.batch_coarse_pose_refine_max_reprojection_error_px):.3f}px"
                    )
                live_values[hole_id] = {
                    "point_base_mm": point_base,
                    "normal_base": normal_base,
                    "normal_camera": normal_camera,
                    "center_px": np.asarray(summary["center_px"], dtype=np.float64),
                    "summary": summary,
                    "reprojection_px": reprojection,
                    "reprojection_error_px": reprojection_error_px,
                }
                iteration_report["holes"][str(hole_id)] = {
                    "success": True,
                    "valid_frames": int(summary["valid_frames"]),
                    "center_px": summary["center_px"],
                    "center_scatter_p95_px": summary["center_scatter_p95_px"],
                    "tracking_distance_p95_px": summary.get("tracking_distance_p95_px"),
                    "plane_rmse_median_mm": summary["plane_rmse_median_mm"],
                    "reprojection_error_px": reprojection_error_px,
                }
            except Exception as exc:
                hole_failures[hole_id] = f"{type(exc).__name__}:{exc}"
                iteration_report["holes"][str(hole_id)] = {
                    "success": False,
                    "valid_frames": len(valid_observations),
                    "total_frames": len(observations),
                    "error": hole_failures[hole_id],
                }

        iteration_report["latest_overlay_path"] = (
            None if latest_overlay_path is None else str(latest_overlay_path)
        )
        iteration_report["success_hole_count"] = len(live_values)
        iteration_report["failed_holes"] = sorted(hole_failures)
        expected_after: dict[int, np.ndarray] = {}
        observed_px = {
            hole_id: value["center_px"] for hole_id, value in live_values.items()
        }
        updated_group: list[dict[str, Any]] | None = None
        planned_target: np.ndarray | None = None
        planned_geometry: dict[str, Any] | None = None
        plan_error: str | None = None
        if not hole_failures and len(live_values) == len(working_group):
            updated_group = []
            for hole in working_group:
                hole_id = int(hole["hole_id"])
                value = live_values[hole_id]
                updated = dict(hole)
                point_base = np.asarray(value["point_base_mm"], dtype=np.float64).reshape(3)
                normal_base = np.asarray(value["normal_base"], dtype=np.float64).reshape(3)
                updated["initial_center_base_mm"] = point_base.copy()
                updated["initial_plane_normal_base"] = normal_base.copy()
                updated["planning_normal_base"] = normal_base.copy()
                updated["coarse_pose_refined_center_base_mm"] = point_base.copy()
                updated["coarse_pose_refined_normal_base"] = normal_base.copy()
                updated_group.append(updated)
            try:
                planned_target, planned_geometry = _plan_batch_coarse_group_pose(
                    updated_group, capture_tcp, handeye, fixed_rz_rad,
                    last_intrinsics, cfg.coarse_height_mm,
                    cfg.batch_coarse_view_margin_px,
                )
                expected_after = {
                    int(key): np.asarray(value, dtype=np.float64).reshape(2)
                    for key, value in planned_geometry["projected_holes_px"].items()
                }
            except Exception as exc:
                plan_error = f"replan_failed:{type(exc).__name__}:{exc}"

        current_projection_values = [
            np.asarray(value["reprojection_px"], dtype=np.float64).reshape(2)
            for value in live_values.values()
        ]
        observed_values = [
            np.asarray(value["center_px"], dtype=np.float64).reshape(2)
            for value in live_values.values()
        ]
        metrics: dict[str, Any] = {
            "target_position_delta_mm": None,
            "target_rotation_delta_deg": None,
            "target_xy_delta_mm": None,
            "target_z_delta_mm": None,
            "current_group_center_px": None,
            "observed_group_center_px": (
                np.mean(np.asarray(observed_values), axis=0)
                if observed_values else None
            ),
            "current_group_center_error_px": None,
            "target_group_center_px": None,
            "target_group_center_error_px": None,
            "pose_height_error_mm": None,
            "pose_normal_error_deg": None,
            "reprojection_error_p95_px": None,
        }
        if current_projection_values:
            current_array = np.asarray(current_projection_values, dtype=np.float64)
            current_center = 0.5 * (
                np.min(current_array, axis=0) + np.max(current_array, axis=0)
            )
            metrics["current_group_center_px"] = current_center
            metrics["current_group_center_error_px"] = float(np.linalg.norm(
                current_center - np.asarray((last_intrinsics.cx, last_intrinsics.cy), dtype=np.float64)
            ))
        if live_values:
            metrics["reprojection_error_p95_px"] = float(np.percentile([
                float(value["reprojection_error_px"]) for value in live_values.values()
            ], 95.0))
        if planned_target is not None and planned_geometry is not None and updated_group is not None:
            position_delta = np.asarray(planned_target[:3, 3] - capture_tcp[:3, 3], dtype=np.float64)
            metrics.update({
                "target_position_delta_mm": float(np.linalg.norm(position_delta)),
                "target_rotation_delta_deg": _rotation_distance_deg(
                    capture_tcp[:3, :3], planned_target[:3, :3],
                ),
                "target_xy_delta_mm": float(np.linalg.norm(position_delta[:2])),
                "target_z_delta_mm": float(abs(position_delta[2])),
                "target_group_center_px": planned_geometry.get("group_center_px"),
                "target_group_center_error_px": float(np.linalg.norm(
                    np.asarray(planned_geometry.get("group_center_px"), dtype=np.float64).reshape(2)
                    - np.asarray((last_intrinsics.cx, last_intrinsics.cy), dtype=np.float64)
                )),
                "pose_height_error_mm": float(np.median([
                    camera_height_to_plane_mm(
                        capture_tcp, handeye.T_tcp_rgb_camera,
                        np.asarray(value["point_base_mm"], dtype=np.float64),
                    ) - float(cfg.coarse_height_mm)
                    for value in live_values.values()
                ])),
                "pose_normal_error_deg": _angle_deg(
                    camera_transform(capture_tcp, handeye.T_tcp_rgb_camera)[:3, 2],
                    -np.asarray(planned_geometry["group_normal_toward_camera_base"], dtype=np.float64),
                ),
            })
        iteration_report["metrics"] = metrics
        iteration_report["plan_error"] = plan_error
        iteration_report["visualization"] = _save_group_pose_refinement_visualization(
            latest_overlay_path,
            run_dir / (
                f"batch_coarse_group_{int(group_index):02d}_pose_refine_"
                f"iter_{int(iteration):02d}"
            ),
            group_index,
            iteration,
            working_group,
            expected_before,
            observed_px,
            expected_after,
            accepted=False,
            note=(
                plan_error or
                (";".join(f"H{hole_id}:{reason}" for hole_id, reason in hole_failures.items())
                 if hole_failures else "replan_check")
            ),
            timing=timing,
        )
        report["iterations"].append(iteration_report)

        if plan_error is not None or updated_group is None or planned_target is None or planned_geometry is None:
            report["reason"] = plan_error or "not_all_holes_have_stable_geometry"
            break

        last_live_values = live_values
        last_refined_group = updated_group
        last_target = np.asarray(planned_target, dtype=np.float64).copy()
        last_geometry = planned_geometry
        last_metrics = metrics
        pose_ok = bool(
            float(metrics["target_position_delta_mm"]) <= float(cfg.batch_coarse_pose_refine_pose_tolerance_mm)
            and float(metrics["target_rotation_delta_deg"]) <= float(cfg.batch_coarse_pose_refine_rotation_tolerance_deg)
            and float(metrics["reprojection_error_p95_px"]) <= float(cfg.batch_coarse_pose_refine_max_reprojection_error_px)
            and float(metrics["target_group_center_error_px"]) <= float(cfg.center_tolerance_px)
        )
        iteration_report["pose_gate_passed"] = pose_ok
        if pose_ok:
            # Only now publish the live anchors to the formal shared capture.
            for hole in original_group:
                hole_id = int(hole["hole_id"])
                value = live_values[hole_id]
                hole["coarse_pose_refined_center_base_mm"] = np.asarray(
                    value["point_base_mm"], dtype=np.float64,
                ).copy()
                hole["coarse_pose_refined_normal_base"] = np.asarray(
                    value["normal_base"], dtype=np.float64,
                ).copy()
            report.update({
                "accepted": True,
                "accepted_iteration": int(iteration),
                "reason": "stable_live_geometry_replanned_pose_within_gate",
                "formal_capture_policy": "refined_live_anchor_and_pose",
                "final_target_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(last_target),
                "final_target_tcp_transform_mm": last_target,
                "final_capture_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(capture_tcp),
                "final_metrics": metrics,
            })
            # Mark the final visualization as accepted in its own artifact when
            # the iteration already has a frame; failure to write this optional
            # audit image must not invalidate localization.
            if latest_overlay_path is not None:
                accepted_visualization = _save_group_pose_refinement_visualization(
                    latest_overlay_path,
                    run_dir / (
                        f"batch_coarse_group_{int(group_index):02d}_pose_refine_"
                        f"iter_{int(iteration):02d}_accepted"
                    ),
                    group_index, iteration, working_group,
                    expected_before, observed_px, expected_after,
                    accepted=True,
                    note="stable multi-frame live geometry accepted",
                    timing=timing,
                )
                iteration_report["accepted_visualization"] = accepted_visualization
            return {
                "group": original_group,
                "current_tcp": capture_tcp,
                "target": last_target,
                "geometry": last_geometry,
                "report": report,
            }

        if iteration >= max_iterations:
            report["reason"] = "pose_gate_failed_after_max_iterations"
            break

        correction_distance = float(metrics["target_position_delta_mm"])
        correction_rotation = float(metrics["target_rotation_delta_deg"])
        if (
            correction_distance < float(cfg.batch_coarse_pose_refine_min_move_mm)
            and correction_rotation < float(cfg.batch_coarse_pose_refine_rotation_tolerance_deg)
        ):
            report["reason"] = "pose_gate_failed_without_actionable_correction"
            break

        # 总量门限保护规划异常；允许在总量范围内拆成多次小步闭环移动。
        total_translation = float(np.linalg.norm(
            last_target[:3, 3] - original_tcp[:3, 3],
        ))
        total_rotation = float(_rotation_distance_deg(
            original_tcp[:3, :3], last_target[:3, :3],
        ))
        if total_translation > float(cfg.batch_coarse_pose_refine_max_total_correction_mm) + 1e-9:
            iteration_report["correction_move_rejected"] = {
                "translation_mm": correction_distance,
                "rotation_deg": correction_rotation,
                "total_translation_mm": total_translation,
                "total_rotation_deg": total_rotation,
                "max_total_translation_mm": float(
                    cfg.batch_coarse_pose_refine_max_total_correction_mm
                ),
                "max_total_rotation_deg": float(
                    cfg.batch_coarse_pose_refine_max_total_correction_rotation_deg
                ),
                "reason": "total_translation_exceeds_motion_gate",
            }
            report["reason"] = "total_pose_correction_exceeds_motion_gate"
            break
        if total_rotation > float(cfg.batch_coarse_pose_refine_max_total_correction_rotation_deg) + 1e-9:
            iteration_report["correction_move_rejected"] = {
                "translation_mm": correction_distance,
                "rotation_deg": correction_rotation,
                "total_translation_mm": total_translation,
                "total_rotation_deg": total_rotation,
                "max_total_translation_mm": float(
                    cfg.batch_coarse_pose_refine_max_total_correction_mm
                ),
                "max_total_rotation_deg": float(
                    cfg.batch_coarse_pose_refine_max_total_correction_rotation_deg
                ),
                "reason": "total_rotation_exceeds_motion_gate",
            }
            report["reason"] = "total_pose_correction_exceeds_motion_gate"
            break

        step_target, step_fraction, step_distance, step_rotation = _bounded_pose_step(
            capture_tcp, last_target, cfg,
        )
        correction_allowed, correction_gate_reason = _pose_correction_motion_gate(
            step_distance, step_rotation, cfg,
        )
        if not correction_allowed:
            iteration_report["correction_move_rejected"] = {
                "translation_mm": correction_distance,
                "rotation_deg": correction_rotation,
                "step_translation_mm": step_distance,
                "step_rotation_deg": step_rotation,
                "step_fraction": step_fraction,
                "max_translation_mm": float(cfg.batch_coarse_pose_refine_max_correction_mm),
                "max_rotation_deg": float(
                    cfg.batch_coarse_pose_refine_max_correction_rotation_deg
                ),
                "reason": correction_gate_reason,
            }
            report["reason"] = "pose_correction_exceeds_motion_gate"
            break
        try:
            with timing.measure(
                f"batch_coarse/group_{int(group_index):02d}/pose_refinement_{int(iteration):02d}/safe_correction_move",
                group_index=int(group_index), iteration=int(iteration),
            ):
                capture_tcp, correction_path_policy = _move_shared_coarse_pose_correction(
                    group_index, group_count, capture_tcp, step_target,
                    args, motion_session, pose_session, cfg,
                    target_height_mm=float(cfg.coarse_height_mm),
                )
            iteration_report["correction_move"] = {
                "requested_target_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(last_target),
                "step_target_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(step_target),
                "actual_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(capture_tcp),
                "translation_mm": correction_distance,
                "rotation_deg": correction_rotation,
                "step_fraction": step_fraction,
                "step_translation_mm": step_distance,
                "step_rotation_deg": step_rotation,
                "total_translation_mm": total_translation,
                "total_rotation_deg": total_rotation,
                "path_policy": correction_path_policy,
            }
            working_group = last_refined_group
        except LocalizationHardwareError:
            raise
        except Exception as exc:
            iteration_report["correction_move_error"] = f"{type(exc).__name__}:{exc}"
            report["reason"] = "safe_correction_move_failed"
            break

    # If one correction was physically executed but the post-check did not
    # pass, keep its measured anchors paired with the actual TCP.  The formal
    # 340 mm capture still has its own strict 15-frame quality gate; this avoids
    # using stale nominal anchors at a corrected pose and lets the existing
    # per-hole fallback decide the final outcome.
    if last_live_values:
        for hole in last_refined_group:
            hole_id = int(hole["hole_id"])
            for original in original_group:
                if int(original["hole_id"]) == hole_id:
                    original["coarse_pose_refined_center_base_mm"] = np.asarray(
                        hole["initial_center_base_mm"], dtype=np.float64,
                    ).copy()
                    original["coarse_pose_refined_normal_base"] = np.asarray(
                        hole.get("planning_normal_base"), dtype=np.float64,
                    ).copy()
                    break
        if (
            bool(getattr(args, "map_build_coarse_only", False))
            and last_metrics is not None
            and (not any(h.get("coarse_quality_retry") for h in original_group)
                 or float(last_metrics.get("pose_normal_error_deg") or 0.0) <= cfg.normal_tolerance_deg)
            and _measured_340_pose_is_map_safe(
                last_metrics,
                len(last_live_values),
                len(original_group),
                cfg,
            )
        ):
            # 340 mm现场帧已经给出了整组孔的稳定实测几何；此时继续追逐
            # 大幅重规划姿态只会把初始手眼/种子误差转化成额外机器人运动。
            # 建图直接使用当前实际TCP，正式采集仍会再次执行自己的质量门。
            actual_capture_geometry = dict(last_geometry)
            actual_capture_geometry["target_tcp_pose_m_rad"] = (
                transform_to_sdk_pose_m_rad(capture_tcp)
            )
            actual_capture_geometry["target_tcp_transform_mm"] = capture_tcp.copy()
            report.update({
                "accepted": True,
                "accepted_iteration": int(
                    (report["iterations"][-1] or {}).get("iteration", 0)
                ),
                "reason": "stable_340_capture_accepted_without_reposition",
                "map_build_safe": True,
                "formal_capture_policy": (
                    "actual_340_capture_pose_with_formal_quality_gate"
                ),
                "deferred_pose_correction": {
                    "translation_mm": float(
                        last_metrics["target_position_delta_mm"]
                    ),
                    "rotation_deg": float(
                        last_metrics["target_rotation_delta_deg"]
                    ),
                    "target_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(last_target),
                    "reason": "replanned_target_exceeds_motion_gate",
                },
                "final_target_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(
                    capture_tcp
                ),
                "final_target_tcp_transform_mm": capture_tcp.copy(),
                "final_capture_tcp_pose_m_rad": transform_to_sdk_pose_m_rad(
                    capture_tcp
                ),
                "final_metrics": last_metrics,
            })
            return {
                "group": original_group,
                "current_tcp": capture_tcp,
                "target": capture_tcp.copy(),
                "geometry": actual_capture_geometry,
                "report": report,
            }
        report["formal_capture_policy"] = (
            "accepted_refined_pose_only"
            if bool(getattr(args, "map_build_coarse_only", False))
            else "last_refined_pose_with_formal_quality_gate"
        )
        return {
            "group": original_group,
            "current_tcp": capture_tcp,
            "target": last_target,
            "geometry": last_geometry,
            "report": report,
        }
    return {
        "group": original_group,
        "current_tcp": original_tcp,
        "target": nominal_target,
        "geometry": nominal_geometry,
        "report": report,
    }

_RUNTIME_DEPENDENCIES = {
    "COARSE_SURFACE_SELECTION_POLICY",
    "Observation",
    "SHARED_OBSERVATION_MIN_DESCENT_MM",
    "SHARED_OBSERVATION_MIN_LIFT_MM",
    "THREE_HOLE_PLACE_SAFE_Z_MARGIN_MM",
    "_angle_deg",
    "_assign_detections_to_projection",
    "_coarse_expected_point_base",
    "_confirm_and_move_line",
    "_draw_black_text",
    "_fuse_coarse",
    "_observation_rows",
    "_plan_hole_tcp_pose_fixed_rz",
    "_plane_estimate_from_info",
    "_project_base_point_to_pixel",
    "_rotation_distance_deg",
    "_tcp_rotation_with_fixed_rz_for_camera_axis",
    "_unit",
    "_wait_robot_steady_before_initial_capture",
    "_write_visualization_pair",
    "camera_height_to_plane_mm",
    "camera_transform",
    "cv2",
    "detect",
    "get_aligned_frame_bundle",
    "hole_camera_point",
    "np",
    "transform_to_sdk_pose_m_rad",
}
_PATCHABLE_FUNCTIONS = {
    "_move_to_sequential_coarse_pose",
    "_move_to_shared_observation_pose",
    "_move_to_shared_coarse_pose",
    "_measured_340_pose_is_map_safe",
    "_plan_batch_coarse_group_pose",
    "_save_group_pose_refinement_visualization",
    "_refine_shared_coarse_group_pose",
}
_ORIGINAL_FUNCTIONS = {name: globals()[name] for name in _PATCHABLE_FUNCTIONS}


def install_runtime(symbols: dict[str, object]) -> None:
    for name in _RUNTIME_DEPENDENCIES:
        if name in symbols:
            globals()[name] = symbols[name]
    for name in _PATCHABLE_FUNCTIONS:
        value = symbols.get(name)
        if value is None:
            continue
        globals()[name] = (
            _ORIGINAL_FUNCTIONS[name]
            if getattr(value, "_runtime_facade_target", None) == name
            else value
        )
