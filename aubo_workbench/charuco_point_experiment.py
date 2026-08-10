# -*- coding: utf-8 -*-
"""ChArUco 角点的手眼结果实验工具（只读、绝不控制机械臂运动）。

固定标定板后，在不同机器人姿态记录同一个 ChArUco 角点。程序把该点从
标定板系依次换算到 RGB 相机系、TCP 系和机器人基坐标系，并统计基坐标系
结果的跨姿态散布。该散布可用于实验评估眼在手上手眼标定的一致性。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import BOARD_CFG, E7_HAND_EYE_CFG, ROBOT_CAMERA_INTEGRATION_CFG
from .geometry import average_transforms, make_transform, rotation_error_deg, rotx, roty, rotz
from .io_utils import atomic_write_json, matrix_to_list, timestamp_str
from .paths import CHARUCO_POINT_EXPERIMENTS_DIR


DEFAULT_CANDIDATE_PATH = Path(E7_HAND_EYE_CFG.candidate_dir) / "e7_handeye_candidate_current.json"
DEFAULT_OUTPUT_DIR = CHARUCO_POINT_EXPERIMENTS_DIR
WINDOW_NAME = "ChArUco selected point hand-eye experiment (read only)"
DEFAULT_FIXED_BASE_RZ_RAD = 1.735


@dataclass(frozen=True)
class HandEyeExperimentResult:
    path: Path
    payload: dict[str, Any]
    T_tcp_rgb_camera: np.ndarray
    validated_for_motion: bool
    experimental_only: bool
    warning: str


def _rigid_transform(values: Any, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} 必须是有限数值组成的 4x4 矩阵")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} 最后一行不是 [0,0,0,1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} 旋转部分不是正交矩阵")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise ValueError(f"{name} 旋转部分行列式不是 +1")
    return matrix


def choose_default_handeye_path() -> Path:
    authoritative = Path(ROBOT_CAMERA_INTEGRATION_CFG.handeye_validation_evidence_path)
    if authoritative.exists():
        return authoritative
    return DEFAULT_CANDIDATE_PATH


def load_handeye_experiment_result(path: str | Path) -> HandEyeExperimentResult:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"手眼结果不存在: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    matrix_value = payload.get("T_tcp_rgb_camera")
    if matrix_value is None:
        matrix_value = payload.get("T_pose_source_sensor")
    if matrix_value is None:
        raise ValueError("手眼结果缺少 T_tcp_rgb_camera")
    matrix = _rigid_transform(matrix_value, "T_tcp_rgb_camera")

    pose_source = str(payload.get("pose_source", "tcp")).strip().lower()
    if pose_source not in {"", "tcp"}:
        raise ValueError(f"当前实验只支持 TCP 手眼结果，文件 pose_source={pose_source!r}")
    frame = str(payload.get("calibration_frame", "rgb_camera")).strip().lower()
    if frame not in {"", "rgb_camera"}:
        raise ValueError(f"当前实验只支持 RGB-PnP 手眼结果，文件 calibration_frame={frame!r}")

    validated = bool(payload.get("validated", False))
    motion_allowed = (
        validated
        and not bool(payload.get("do_not_use_for_motion", True))
        and bool(payload.get("production_eligible", False))
    )
    warning = ""
    if not motion_allowed:
        warning = "该手眼结果未通过生产验证，仅允许只读误差实验，禁止用于机械臂运动。"
    return HandEyeExperimentResult(
        path=source.resolve(), payload=payload, T_tcp_rgb_camera=matrix,
        validated_for_motion=motion_allowed, experimental_only=not motion_allowed,
        warning=warning,
    )


def xyz_rpy_pose(T: np.ndarray) -> dict[str, Any]:
    """输出 ZYX 欧拉角：[Rx(roll), Ry(pitch), Rz(yaw)]，角度单位 rad。"""
    matrix = _rigid_transform(T, "pose")
    rotation = matrix[:3, :3]
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    cp = math.cos(pitch)
    if abs(cp) > 1e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
    return {
        "matrix_4x4": matrix_to_list(matrix),
        "xyz_mm": matrix[:3, 3].astype(float).tolist(),
        "rpy_rad_zyx": [float(roll), float(pitch), float(yaw)],
        "rpy_order_note": "R = Rz(yaw) @ Ry(pitch) @ Rx(roll)",
        "angle_unit": "rad",
    }


def target_pose_with_fixed_rz(T_base_measured: np.ndarray, fixed_rz_rad: float) -> dict[str, Any]:
    """保持测量 XYZ/Rx/Ry，仅将基坐标目标姿态的 Rz 锁定为给定弧度值。"""
    measured = xyz_rpy_pose(T_base_measured)
    rx, ry, _ = [float(v) for v in measured["rpy_rad_zyx"]]
    rz = float(fixed_rz_rad)
    if not math.isfinite(rz):
        raise ValueError("fixed_rz_rad 必须是有限数值")
    rotation = rotz(rz) @ roty(ry) @ rotx(rx)
    target = make_transform(rotation, np.asarray(measured["xyz_mm"], dtype=np.float64))
    result = xyz_rpy_pose(target)
    # 欧拉角存在等价表示；这里显式保存用户要求的命令值，保证 Rz 精确为配置值。
    result["rpy_rad_zyx"] = [rx, ry, rz]
    result["fixed_rz_rad"] = rz
    result["orientation_policy"] = "keep measured Rx/Ry and lock base-frame target Rz"
    return result


def selected_corner_transforms(
    T_base_tcp: np.ndarray,
    T_tcp_rgb_camera: np.ndarray,
    T_rgb_board: np.ndarray,
    corner_board_mm: np.ndarray,
) -> dict[str, np.ndarray]:
    """构造所选角点坐标系；其姿态轴与 ChArUco 标定板坐标轴相同。"""
    T_board_corner = make_transform(np.eye(3), np.asarray(corner_board_mm, dtype=np.float64))
    T_rgb_corner = _rigid_transform(T_rgb_board, "T_rgb_board") @ T_board_corner
    T_tcp_corner = _rigid_transform(T_tcp_rgb_camera, "T_tcp_rgb_camera") @ T_rgb_corner
    T_base_corner = _rigid_transform(T_base_tcp, "T_base_tcp") @ T_tcp_corner
    return {
        "board": T_board_corner,
        "rgb_camera": T_rgb_corner,
        "tcp": T_tcp_corner,
        "base": T_base_corner,
    }


def nearest_detected_corner(
    click_xy: tuple[float, float],
    corner_ids: list[int] | None,
    image_points: list[list[float]] | None,
    maximum_distance_px: float = 35.0,
) -> int | None:
    if not corner_ids or not image_points or len(corner_ids) != len(image_points):
        return None
    points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    distances = np.linalg.norm(points - np.asarray(click_xy, dtype=np.float64), axis=1)
    index = int(np.argmin(distances))
    if float(distances[index]) > float(maximum_distance_px):
        return None
    return int(corner_ids[index])


def compute_experiment_statistics(
    records: list[dict[str, Any]],
    reference_base_xyz_mm: list[float] | tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    if not records:
        return {"sample_count": 0, "status": "no_samples"}
    points = np.asarray([item["frames"]["base"]["xyz_mm"] for item in records], dtype=np.float64)
    transforms = [np.asarray(item["frames"]["base"]["matrix_4x4"], dtype=np.float64) for item in records]
    mean_point = np.mean(points, axis=0)
    residual_xyz = points - mean_point
    distances = np.linalg.norm(residual_xyz, axis=1)
    mean_transform = average_transforms(transforms)
    rotation_errors_rad = np.deg2rad(np.asarray(
        [rotation_error_deg(mean_transform[:3, :3], T[:3, :3]) for T in transforms],
        dtype=np.float64,
    ))
    tcp_poses = np.asarray(
        [item.get("robot", {}).get("pose_values_mm_rad", [np.nan] * 6) for item in records],
        dtype=np.float64,
    )
    result = {
        "sample_count": len(records),
        "meaning": "固定标定板、跨机器人姿态观测同一角点时，该点在机器人基坐标系下的散布",
        "base_xyz_mean_mm": mean_point.astype(float).tolist(),
        "base_xyz_axis_std_mm": np.std(points, axis=0).astype(float).tolist(),
        "base_xyz_axis_range_mm": np.ptp(points, axis=0).astype(float).tolist(),
        "base_point_scatter_rms_mm": float(np.sqrt(np.mean(distances * distances))),
        "base_point_scatter_p95_mm": float(np.percentile(distances, 95)),
        "base_point_scatter_max_mm": float(np.max(distances)),
        "base_rotation_scatter_rms_rad": float(
            np.sqrt(np.mean(rotation_errors_rad * rotation_errors_rad))
        ),
        "base_rotation_scatter_max_rad": float(np.max(rotation_errors_rad)),
        "robot_tcp_xyz_range_mm": (
            np.ptp(tcp_poses[:, :3], axis=0).astype(float).tolist()
            if np.isfinite(tcp_poses[:, :3]).all() else None
        ),
        "robot_tcp_rpy_naive_range_rad": (
            np.ptp(tcp_poses[:, 3:6], axis=0).astype(float).tolist()
            if np.isfinite(tcp_poses[:, 3:6]).all() else None
        ),
        "per_sample_base_residual_xyz_mm": residual_xyz.astype(float).tolist(),
        "per_sample_base_distance_mm": distances.astype(float).tolist(),
        "interpretation_warning": (
            "未提供基坐标真值时，散布反映跨姿态一致性而非绝对准确度。至少应在明显不同的机器人位置和姿态"
            "记录同一固定角点；同一姿态重复采样只能反映检测重复性。"
        ),
    }
    if reference_base_xyz_mm is not None:
        reference = np.asarray(reference_base_xyz_mm, dtype=np.float64).reshape(3)
        if not np.isfinite(reference).all():
            raise ValueError("reference_base_xyz_mm 包含非有限数值")
        absolute_vectors = points - reference
        absolute_distances = np.linalg.norm(absolute_vectors, axis=1)
        result["reference_base_xyz_mm"] = reference.astype(float).tolist()
        result["absolute_error_mean_xyz_mm"] = (mean_point - reference).astype(float).tolist()
        result["absolute_error_of_mean_mm"] = float(np.linalg.norm(mean_point - reference))
        result["absolute_error_rms_mm"] = float(
            np.sqrt(np.mean(absolute_distances * absolute_distances))
        )
        result["absolute_error_max_mm"] = float(np.max(absolute_distances))
        result["per_sample_absolute_error_xyz_mm"] = absolute_vectors.astype(float).tolist()
        result["per_sample_absolute_distance_mm"] = absolute_distances.astype(float).tolist()
    return result


def _pose_bracket(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    a = (before or {}).get("pose_values")
    b = (after or {}).get("pose_values")
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) < 6 or len(b) < 6:
        return {"ok": False, "reason": "pose_values_missing"}
    av = np.asarray(a[:6], dtype=np.float64)
    bv = np.asarray(b[:6], dtype=np.float64)
    xyz_delta = np.abs(bv[:3] - av[:3])
    abc_delta = np.abs((bv[3:6] - av[3:6] + 180.0) % 360.0 - 180.0)
    xyz_limit = float(E7_HAND_EYE_CFG.maximum_pose_bracket_xyz_mm)
    abc_limit = float(E7_HAND_EYE_CFG.maximum_pose_bracket_abc_deg)
    return {
        "ok": bool(np.max(xyz_delta) <= xyz_limit and np.max(abc_delta) <= abc_limit),
        "xyz_delta_mm": xyz_delta.astype(float).tolist(),
        "rpy_delta_deg": abc_delta.astype(float).tolist(),
        "xyz_limit_mm": xyz_limit,
        "rpy_limit_deg": abc_limit,
        "before_timestamp": (before or {}).get("timestamp"),
        "after_timestamp": (after or {}).get("timestamp"),
    }


def _record_payload(
    index: int,
    selected_id: int,
    pose_result: Any,
    T_base_tcp: np.ndarray,
    T_tcp_rgb_camera: np.ndarray,
    snapshot: dict[str, Any],
    bracket: dict[str, Any],
    camera_metadata: dict[str, Any],
    fixed_base_rz_rad: float,
) -> dict[str, Any]:
    ids = [int(v) for v in (pose_result.used_corner_ids or [])]
    if selected_id not in ids:
        raise ValueError(f"当前帧没有检测到所选角点 ID={selected_id}")
    array_index = ids.index(selected_id)
    object_point = np.asarray(pose_result.object_points_mm[array_index], dtype=np.float64)
    image_point = np.asarray(pose_result.image_points[array_index], dtype=np.float64)
    transforms = selected_corner_transforms(
        T_base_tcp, T_tcp_rgb_camera, pose_result.T_rgb_board, object_point,
    )
    sdk_pose = snapshot.get("pose_values_sdk_m_rad")
    pose_values_mm_rad = None
    if isinstance(sdk_pose, (list, tuple)) and len(sdk_pose) >= 6:
        values = [float(v) for v in sdk_pose[:6]]
        pose_values_mm_rad = [values[0] * 1000.0, values[1] * 1000.0, values[2] * 1000.0, *values[3:6]]
    return {
        "index": int(index),
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "selected_corner_id": int(selected_id),
        "selected_corner_image_uv_px": image_point.astype(float).tolist(),
        "selected_corner_board_xyz_mm": object_point.astype(float).tolist(),
        "selected_corner_pose_definition": "原点为所选角点，XYZ轴方向与ChArUco标定板坐标系相同",
        "frames": {name: xyz_rpy_pose(T) for name, T in transforms.items()},
        "target_pose_base": target_pose_with_fixed_rz(
            transforms["base"], fixed_base_rz_rad,
        ),
        "board_pose": {
            "rgb_camera": xyz_rpy_pose(pose_result.T_rgb_board),
            "tcp": xyz_rpy_pose(T_tcp_rgb_camera @ pose_result.T_rgb_board),
            "base": xyz_rpy_pose(T_base_tcp @ T_tcp_rgb_camera @ pose_result.T_rgb_board),
        },
        "pnp_quality": {
            "charuco_count": int(pose_result.charuco_count),
            "pnp_inlier_count": int(pose_result.rgb_pnp_inlier_count),
            "reprojection_rmse_px": float(pose_result.rgb_reprojection_rmse_px),
            "reprojection_max_px": float(pose_result.rgb_reprojection_max_px),
            "status": str(pose_result.status),
        },
        "robot": {
            "pose_source": snapshot.get("pose_source"),
            "pose_values_mm_rad": pose_values_mm_rad,
            "pose_values_sdk_m_rad": sdk_pose,
            "tcp_pose_sdk_m_rad": snapshot.get("tcp_pose_sdk_m_rad"),
            "actual_tcp_offset_sdk_m_rad": snapshot.get("actual_tcp_offset_sdk_m_rad"),
            "configured_tcp_offset_sdk_m_rad": snapshot.get("configured_tcp_offset_sdk_m_rad"),
            "T_base_tcp": matrix_to_list(T_base_tcp),
            "snapshot": snapshot,
            "camera_frame_pose_bracket": bracket,
        },
        "camera_metadata": camera_metadata,
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index", "timestamp", "corner_id", "u_px", "v_px",
        "board_x_mm", "board_y_mm", "board_z_mm",
        "camera_x_mm", "camera_y_mm", "camera_z_mm",
        "tcp_x_mm", "tcp_y_mm", "tcp_z_mm",
        "base_x_mm", "base_y_mm", "base_z_mm",
        "measured_base_rx_rad", "measured_base_ry_rad", "measured_base_rz_rad",
        "target_base_rx_rad", "target_base_ry_rad", "target_base_rz_rad",
        "pnp_rmse_px", "pnp_max_px",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            uv = item["selected_corner_image_uv_px"]
            board = item["frames"]["board"]["xyz_mm"]
            camera = item["frames"]["rgb_camera"]["xyz_mm"]
            tcp = item["frames"]["tcp"]["xyz_mm"]
            base = item["frames"]["base"]["xyz_mm"]
            measured_rpy = item["frames"]["base"]["rpy_rad_zyx"]
            target_rpy = item["target_pose_base"]["rpy_rad_zyx"]
            writer.writerow({
                "index": item["index"], "timestamp": item["timestamp"],
                "corner_id": item["selected_corner_id"], "u_px": uv[0], "v_px": uv[1],
                "board_x_mm": board[0], "board_y_mm": board[1], "board_z_mm": board[2],
                "camera_x_mm": camera[0], "camera_y_mm": camera[1], "camera_z_mm": camera[2],
                "tcp_x_mm": tcp[0], "tcp_y_mm": tcp[1], "tcp_z_mm": tcp[2],
                "base_x_mm": base[0], "base_y_mm": base[1], "base_z_mm": base[2],
                "measured_base_rx_rad": measured_rpy[0],
                "measured_base_ry_rad": measured_rpy[1],
                "measured_base_rz_rad": measured_rpy[2],
                "target_base_rx_rad": target_rpy[0],
                "target_base_ry_rad": target_rpy[1],
                "target_base_rz_rad": target_rpy[2],
                "pnp_rmse_px": item["pnp_quality"]["reprojection_rmse_px"],
                "pnp_max_px": item["pnp_quality"]["reprojection_max_px"],
            })


def _write_outputs(
    json_path: Path,
    csv_path: Path,
    records: list[dict[str, Any]],
    handeye: HandEyeExperimentResult,
    selected_id: int | None,
    reference_base_xyz_mm: list[float] | None = None,
    fixed_base_rz_rad: float = DEFAULT_FIXED_BASE_RZ_RAD,
) -> dict[str, Any]:
    payload = {
        "record_type": "charuco_selected_corner_handeye_experiment",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "read_only_no_robot_motion": True,
        "experimental_only": bool(handeye.experimental_only),
        "warning": handeye.warning,
        "handeye_source_path": str(handeye.path),
        "handeye_calibration_id": handeye.payload.get("calibration_id"),
        "handeye_generated_at": handeye.payload.get("generated_at"),
        "handeye_validated_for_motion": bool(handeye.validated_for_motion),
        "handeye_validation_center_scatter_rms_mm": handeye.payload.get(
            "validation_center_scatter_rms_mm"
        ),
        "camera_serial_expected": handeye.payload.get("camera_serial"),
        "T_tcp_rgb_camera": matrix_to_list(handeye.T_tcp_rgb_camera),
        "selected_corner_id": selected_id,
        "target_pose_policy": {
            "frame": "robot_base",
            "fixed_rz_rad": float(fixed_base_rz_rad),
            "rule": "XYZ from hand-eye; Rx/Ry from measured board-aligned point pose; Rz fixed",
        },
        "board": {
            "squares_x": BOARD_CFG.squares_x, "squares_y": BOARD_CFG.squares_y,
            "square_length_mm": BOARD_CFG.square_length_mm,
            "marker_length_mm": BOARD_CFG.marker_length_mm,
            "aruco_dict_name": BOARD_CFG.aruco_dict_name,
        },
        "statistics": compute_experiment_statistics(records, reference_base_xyz_mm),
        "records": records,
    }
    atomic_write_json(json_path, payload)
    _write_csv(csv_path, records)
    return payload


def _draw_ascii_lines(image: np.ndarray, lines: list[str], warning: bool = False) -> None:
    y = 26
    for line in lines:
        cv2.putText(
            image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            (0, 80, 255) if warning else (40, 255, 40), 2, cv2.LINE_AA,
        )
        y += 27


def _annotate_corners(image: np.ndarray, pose_result: Any, selected_id: int | None) -> None:
    for corner_id, point in zip(pose_result.used_corner_ids or [], pose_result.image_points or []):
        center = tuple(np.round(np.asarray(point)).astype(int))
        is_selected = int(corner_id) == selected_id
        cv2.circle(image, center, 9 if is_selected else 4, (0, 255, 255) if is_selected else (255, 255, 0), 2)
        cv2.putText(
            image, str(int(corner_id)), (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255) if is_selected else (255, 255, 0), 1,
            cv2.LINE_AA,
        )


def _print_record(record: dict[str, Any], statistics: dict[str, Any]) -> None:
    print(f"\n[MEASURE] #{record['index']} ChArUco corner ID={record['selected_corner_id']}")
    for label, key in (("RGB相机系", "rgb_camera"), ("TCP系", "tcp"), ("机器人基坐标系", "base")):
        pose = record["frames"][key]
        xyz = pose["xyz_mm"]
        rpy = pose["rpy_rad_zyx"]
        print(
            f"[MEASURE] {label}: XYZ=({xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f}) mm | "
            f"RPY=({rpy[0]:.9f}, {rpy[1]:.9f}, {rpy[2]:.9f}) rad"
        )
    print(
        f"[MEASURE] PnP: RMSE={record['pnp_quality']['reprojection_rmse_px']:.4f}px, "
        f"max={record['pnp_quality']['reprojection_max_px']:.4f}px"
    )
    target = record["target_pose_base"]
    target_xyz = target["xyz_mm"]
    target_rpy = target["rpy_rad_zyx"]
    print(
        f"[TARGET] BASE XYZ=({target_xyz[0]:.6f}, {target_xyz[1]:.6f}, {target_xyz[2]:.6f}) mm | "
        f"RPY=({target_rpy[0]:.9f}, {target_rpy[1]:.9f}, {target_rpy[2]:.9f}) rad | "
        f"Rz locked={target['fixed_rz_rad']:.3f} rad"
    )
    if int(statistics.get("sample_count", 0)) >= 2:
        ranges = statistics["base_xyz_axis_range_mm"]
        print(
            f"[STATS] n={statistics['sample_count']} | base scatter RMS="
            f"{statistics['base_point_scatter_rms_mm']:.6f} mm | "
            f"max={statistics['base_point_scatter_max_mm']:.6f} mm | "
            f"XYZ range=({ranges[0]:.6f}, {ranges[1]:.6f}, {ranges[2]:.6f}) mm"
        )


def _default_corner_id(board: Any) -> int:
    from .charuco_detect import get_board_chessboard_corners

    points = get_board_chessboard_corners(board)
    target = np.asarray([BOARD_CFG.pattern_width_mm / 2.0, BOARD_CFG.pattern_height_mm / 2.0, 0.0])
    return int(np.argmin(np.linalg.norm(points - target, axis=1)))


def run_live(args: argparse.Namespace) -> int:
    # 硬件模块延迟导入，确保核心坐标与统计函数可以在无相机/机械臂环境中单测。
    from .camera import (
        get_device_identity, get_rgb_frame_bundle, init_rgb_handeye_pipeline,
    )
    from .charuco_detect import create_charuco_board, estimate_rgb_board_pose
    from .robot import close_aubo_session, get_capture_pose_transform

    handeye = load_handeye_experiment_result(args.handeye)
    print(f"[INFO] 手眼结果: {handeye.path}")
    if handeye.warning:
        print(f"[EXPERIMENT-WARNING] {handeye.warning}")

    session_id = timestamp_str()
    output_dir = Path(args.output_dir)
    json_path = output_dir / f"charuco_point_experiment_{session_id}.json"
    csv_path = output_dir / f"charuco_point_experiment_{session_id}.csv"
    image_dir = output_dir / f"charuco_point_experiment_{session_id}_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    pipeline = None
    records: list[dict[str, Any]] = []
    mouse_click: list[tuple[int, int] | None] = [None]
    board, dictionary = create_charuco_board()
    selected_id = int(args.corner_id) if args.corner_id is not None else _default_corner_id(board)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_click[0] = (x, y)

    try:
        pipeline = init_rgb_handeye_pipeline()
        camera_identity = get_device_identity(pipeline)
        actual_serial = str(camera_identity.get("serial_number", "")).strip()
        expected_serial = str(handeye.payload.get("camera_serial", "")).strip()
        if expected_serial and actual_serial != expected_serial and not args.allow_camera_mismatch:
            raise RuntimeError(
                f"相机序列号不匹配：手眼结果={expected_serial}, 当前相机={actual_serial or 'unknown'}。"
                "如确实只做受控实验，可显式添加 --allow-camera-mismatch。"
            )
        print(f"[INFO] 当前相机序列号: {actual_serial or 'unknown'}")
        print(f"[INFO] 默认/当前角点 ID={selected_id}；鼠标左键可选择其它已检测角点。")
        print("[INFO] s/SPACE=记录当前点，q/ESC=保存并退出。固定标定板后请移动机器人到不同姿态逐次记录。")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)

        while True:
            bundle = get_rgb_frame_bundle(pipeline)
            if bundle is None:
                continue
            result = estimate_rgb_board_pose(bundle.color_bgr, bundle.intrinsics, board, dictionary)
            if mouse_click[0] is not None:
                chosen = nearest_detected_corner(
                    mouse_click[0], result.used_corner_ids, result.image_points,
                )
                if chosen is not None and records and chosen != selected_id:
                    print(
                        f"[SELECT] 本次实验已经锁定角点 ID={selected_id}。不同角点不能混合统计；"
                        "请退出后重新启动一次实验。"
                    )
                elif chosen is not None:
                    selected_id = chosen
                    print(f"[SELECT] 已选择 ChArUco 角点 ID={selected_id}")
                else:
                    print("[SELECT] 点击位置附近没有已检测角点，请点击角点编号附近。")
                mouse_click[0] = None

            display = result.rgb_overlay.copy()
            _annotate_corners(display, result, selected_id)
            lines = [
                f"Selected corner ID={selected_id} | records={len(records)}",
                f"Target BASE Rz locked={args.fixed_base_rz:.3f} rad",
                "Mouse: select corner | S/SPACE: measure | Q/ESC: save & quit",
            ]
            if handeye.experimental_only:
                lines.insert(0, "UNVALIDATED HAND-EYE: EXPERIMENT ONLY / NO MOTION")
            if result.ok and selected_id in (result.used_corner_ids or []):
                idx = list(result.used_corner_ids).index(selected_id)
                p = result.object_points_mm[idx]
                lines.append(f"Board XYZ=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) mm")
            _draw_ascii_lines(display, lines, warning=handeye.experimental_only)
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key not in (ord("s"), ord("S"), 32):
                continue

            T_before, snapshot_before, status_before = get_capture_pose_transform()
            if T_before is None:
                print(f"[REJECT] 取帧前 AUBO TCP 位姿不可用: {status_before}")
                continue
            measurement_bundle = get_rgb_frame_bundle(pipeline)
            if measurement_bundle is None:
                print("[REJECT] 相机取帧失败")
                continue
            measurement_result = estimate_rgb_board_pose(
                measurement_bundle.color_bgr, measurement_bundle.intrinsics, board, dictionary,
            )
            T_after, snapshot_after, status_after = get_capture_pose_transform()
            if T_after is None:
                print(f"[REJECT] 取帧后 AUBO TCP 位姿不可用: {status_after}")
                continue
            bracket = _pose_bracket(snapshot_before, snapshot_after)
            if not bracket.get("ok", False):
                print(f"[REJECT] 取帧前后 TCP 不稳定: {bracket}")
                continue
            if not measurement_result.ok or measurement_result.T_rgb_board is None:
                print(f"[REJECT] 当前 ChArUco RGB-PnP 不合格: {measurement_result.status}")
                continue
            if selected_id not in (measurement_result.used_corner_ids or []):
                print(f"[REJECT] 当前测量帧未检测到角点 ID={selected_id}")
                continue
            if selected_id not in (measurement_result.valid_corner_ids or []):
                print(f"[REJECT] 角点 ID={selected_id} 不是当前 PnP 内点，请调整视角/光照后重试")
                continue

            snapshot = dict(snapshot_after or {})
            snapshot["pose_status"] = status_after
            camera_metadata = measurement_bundle.metadata_dict()
            camera_metadata["device"] = camera_identity
            record = _record_payload(
                len(records) + 1, selected_id, measurement_result, T_after,
                handeye.T_tcp_rgb_camera, snapshot, bracket, camera_metadata,
                args.fixed_base_rz,
            )
            records.append(record)
            overlay = measurement_result.rgb_overlay.copy()
            _annotate_corners(overlay, measurement_result, selected_id)
            cv2.imwrite(str(image_dir / f"measure_{len(records):03d}_rgb.png"), measurement_bundle.color_bgr)
            cv2.imwrite(str(image_dir / f"measure_{len(records):03d}_overlay.png"), overlay)
            payload = _write_outputs(
                json_path, csv_path, records, handeye, selected_id, args.reference_base_xyz,
                args.fixed_base_rz,
            )
            _print_record(record, payload["statistics"])
            print(f"[SAVE] {json_path}")

        final_payload = _write_outputs(
            json_path, csv_path, records, handeye, selected_id, args.reference_base_xyz,
            args.fixed_base_rz,
        )
        print(f"[DONE] JSON: {json_path}")
        print(f"[DONE] CSV:  {csv_path}")
        stats = final_payload["statistics"]
        if int(stats.get("sample_count", 0)) >= 2:
            print(
                f"[RESULT] 基坐标点散布 RMS={stats['base_point_scatter_rms_mm']:.6f} mm, "
                f"P95={stats['base_point_scatter_p95_mm']:.6f} mm, "
                f"max={stats['base_point_scatter_max_mm']:.6f} mm"
            )
            if "absolute_error_of_mean_mm" in stats:
                print(
                    f"[RESULT] 相对输入真值的均值绝对误差="
                    f"{stats['absolute_error_of_mean_mm']:.6f} mm"
                )
        else:
            print("[RESULT] 至少需要 2 个姿态才能统计散布，建议采集 8~12 个明显不同的姿态。")
        return 0
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        close_aubo_session()
        cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="识别 ChArUco 角点并用眼在手上手眼结果输出相机/TCP/基坐标 XYZ 与位姿。"
    )
    parser.add_argument(
        "--handeye", type=Path, default=choose_default_handeye_path(),
        help="手眼 JSON；优先使用已安装 E7，若不存在则默认当前 E7 candidate（只做实验）。",
    )
    parser.add_argument("--corner-id", type=int, default=None, help="初始 ChArUco 角点 ID；也可在窗口中点击选择。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="JSON/CSV/图片输出目录。")
    parser.add_argument(
        "--reference-base-xyz", type=float, nargs=3, metavar=("X_MM", "Y_MM", "Z_MM"), default=None,
        help="可选：所选角点经外部测量得到的基坐标真值，用于计算绝对位置误差。",
    )
    parser.add_argument(
        "--fixed-base-rz", type=float, default=DEFAULT_FIXED_BASE_RZ_RAD, metavar="RAD",
        help="机器人基坐标系目标位姿的固定 Rz，单位 rad；默认 1.735。",
    )
    parser.add_argument(
        "--allow-camera-mismatch", action="store_true",
        help="仅受控实验时允许当前相机序列号与手眼文件不一致。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_live(args)
