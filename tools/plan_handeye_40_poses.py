#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根据已采集RGB手眼样本，离线生成总计40个位姿的候选采集计划。

本工具只读样本和诊断手眼矩阵，只写 JSON/CSV；不会连接机器人或下发运动。
新增位姿围绕标定板中心生成，并用当前RGB内参做无畸变近似投影边界检查。
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from aubo_workbench.paths import (
    CHARUCO_CALIBRATION_DIR,
    HANDEYE_40_POSE_CSV_PATH,
    HANDEYE_40_POSE_PLAN_PATH,
    HANDEYE_CANDIDATE_PATH,
)

DEFAULT_SAMPLE_DIR = CHARUCO_CALIBRATION_DIR / "samples"
DEFAULT_HANDEYE = HANDEYE_CANDIDATE_PATH
DEFAULT_JSON = HANDEYE_40_POSE_PLAN_PATH
DEFAULT_CSV = HANDEYE_40_POSE_CSV_PATH
BOARD_CENTER_Q_MM = np.array([180.0, 135.0, 0.0], dtype=np.float64)
BOARD_OUTER_CORNERS_Q_MM = np.array(
    [[0.0, 0.0, 0.0], [360.0, 0.0, 0.0], [360.0, 270.0, 0.0], [0.0, 270.0, 0.0]],
    dtype=np.float64,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    result[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return result


def _rpy_zyx_deg(rx: float, ry: float, rz: float) -> np.ndarray:
    ax, ay, az = (math.radians(float(value)) for value in (rx, ry, rz))
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rx_matrix = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry_matrix = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz_matrix = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz_matrix @ ry_matrix @ rx_matrix


def _average_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def _project_board(
    T_base_tcp: np.ndarray,
    T_tcp_camera: np.ndarray,
    T_base_board: np.ndarray,
    intrinsics: dict[str, Any],
) -> dict[str, Any]:
    T_base_camera = T_base_tcp @ T_tcp_camera
    rotation = T_base_camera[:3, :3]
    translation = T_base_camera[:3, 3]
    corners_base = (
        T_base_board[:3, :3] @ BOARD_OUTER_CORNERS_Q_MM.T
    ).T + T_base_board[:3, 3]
    corners_camera = (rotation.T @ (corners_base - translation).T).T
    z = corners_camera[:, 2]
    if np.any(z <= 1.0):
        return {"ok": False, "reason": "board_behind_camera"}
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    uv = np.column_stack((fx * corners_camera[:, 0] / z + cx, fy * corners_camera[:, 1] / z + cy))
    width, height = int(intrinsics["width"]), int(intrinsics["height"])
    margins = np.column_stack((uv[:, 0], width - uv[:, 0], uv[:, 1], height - uv[:, 1]))
    minimum_margin = float(np.min(margins))
    center = np.mean(uv, axis=0)
    return {
        "ok": minimum_margin >= 25.0,
        "reason": "ok" if minimum_margin >= 25.0 else "projected_board_margin_too_small",
        "board_outer_uv": [[float(value) for value in row] for row in uv],
        "board_center_uv_approx": [float(value) for value in center],
        "minimum_image_margin_px_approx": minimum_margin,
        "camera_board_corner_z_mm": [float(value) for value in z],
    }


def _new_orientation_schedule() -> list[tuple[str, float, float, float]]:
    # 共29个：远层10、中层10、近层9。顺序为 layer, Rx, Ry, Rz（deg）。
    return [
        ("far", 18, 0, 0), ("far", -18, 0, 0),
        ("far", 0, 18, 0), ("far", 0, -18, 0),
        ("far", 14, 14, 25), ("far", 14, -14, -25),
        ("far", -14, 14, -25), ("far", -14, -14, 25),
        ("far", 10, 0, 45), ("far", -10, 0, -45),
        ("mid", 20, 5, 20), ("mid", -20, -5, -20),
        ("mid", 5, 20, -20), ("mid", -5, -20, 20),
        ("mid", 15, 15, 40), ("mid", 15, -15, -40),
        ("mid", -15, 15, -40), ("mid", -15, -15, 40),
        ("mid", 10, -5, 55), ("mid", -10, 5, -55),
        ("near", 12, 0, 20), ("near", -12, 0, -20),
        ("near", 0, 12, -20), ("near", 0, -12, 20),
        ("near", 10, 10, 0), ("near", 10, -10, 0),
        ("near", -10, 10, 0), ("near", -10, -10, 0),
        ("near", 6, -6, 15),
    ]


def _circular_span_deg(values: list[float]) -> float:
    wrapped = np.sort((np.asarray(values, dtype=np.float64) + 180.0) % 360.0 - 180.0)
    if wrapped.size < 2:
        return 0.0
    gaps = np.diff(np.r_[wrapped, wrapped[0] + 360.0])
    return float(360.0 - np.max(gaps))


def _orientation_coverage(poses: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([pose["pose_mm_deg_rxryrz"] for pose in poses], dtype=np.float64)
    rotations = [_rpy_zyx_deg(*row[3:6]) for row in values]
    pair_angles = []
    for first in range(len(rotations)):
        for second in range(first + 1, len(rotations)):
            relative = rotations[first].T @ rotations[second]
            cosine = max(-1.0, min(1.0, (float(np.trace(relative)) - 1.0) * 0.5))
            pair_angles.append(math.degrees(math.acos(cosine)))
    return {
        "rx_range_deg": float(np.ptp(values[:, 3])),
        "ry_range_deg": float(np.ptp(values[:, 4])),
        "rz_circular_span_deg": _circular_span_deg(values[:, 5].tolist()),
        "relative_rotation_mean_deg": float(np.mean(pair_angles)) if pair_angles else 0.0,
        "relative_rotation_max_deg": float(np.max(pair_angles)) if pair_angles else 0.0,
    }


def build_plan(sample_dir: Path, handeye_path: Path) -> dict[str, Any]:
    sample_paths = sorted(sample_dir.glob("sample_*.json"))
    if not sample_paths:
        raise RuntimeError(f"no samples found in {sample_dir}")
    samples = [_load_json(path) for path in sample_paths]
    handeye = _load_json(handeye_path)
    T_tcp_camera = np.asarray(handeye["T_tcp_rgb_camera"], dtype=np.float64).reshape(4, 4)
    handeye_translation_norm_mm = float(np.linalg.norm(T_tcp_camera[:3, 3]))
    if not np.isfinite(T_tcp_camera).all() or handeye_translation_norm_mm > 500.0:
        raise RuntimeError(
            "handeye matrix is physically implausible for capture planning: "
            f"translation_norm={handeye_translation_norm_mm:.3f} mm"
        )

    board_transforms: list[np.ndarray] = []
    board_centers: list[np.ndarray] = []
    distances: list[float] = []
    for sample in samples:
        T_base_tcp = np.asarray(sample["T_base_tool"], dtype=np.float64).reshape(4, 4)
        T_camera_board = np.asarray(sample["T_rgb_board"], dtype=np.float64).reshape(4, 4)
        T_base_board = T_base_tcp @ T_tcp_camera @ T_camera_board
        board_transforms.append(T_base_board)
        board_centers.append(T_base_board[:3, :3] @ BOARD_CENTER_Q_MM + T_base_board[:3, 3])
        distances.append(float(T_camera_board[2, 3]))

    T_base_board = _transform(
        _average_rotation([value[:3, :3] for value in board_transforms]),
        np.mean([value[:3, 3] for value in board_transforms], axis=0),
    )
    board_center_base = np.mean(board_centers, axis=0)
    intrinsics = dict(samples[0]["camera_metadata"]["intrinsics"])

    # 按TCP Z自动分成近、中、远三层，并使用每层实测相机到板距离。
    sorted_z = sorted(float(sample["robot_snapshot"]["pose_values"][2]) for sample in samples)
    layer_z_centers = {
        "near": float(np.mean(sorted_z[: max(1, len(sorted_z) // 3)])),
        "mid": float(np.mean(sorted_z[len(sorted_z) // 3: 2 * len(sorted_z) // 3])),
        "far": float(np.mean(sorted_z[2 * len(sorted_z) // 3:])),
    }
    layer_distance: dict[str, float] = {}
    for layer, z_center in layer_z_centers.items():
        selected = [
            distances[index] for index, sample in enumerate(samples)
            if abs(float(sample["robot_snapshot"]["pose_values"][2]) - z_center) < 25.0
        ]
        layer_distance[layer] = float(np.mean(selected)) if selected else float(np.mean(distances))

    poses: list[dict[str, Any]] = []
    for sample in samples:
        values = [float(value) for value in sample["robot_snapshot"]["pose_values"][:6]]
        poses.append({
            "plan_index": len(poses) + 1,
            "source": "captured_existing",
            "sample_index": int(sample["index"]),
            "layer": min(layer_z_centers, key=lambda name: abs(values[2] - layer_z_centers[name])),
            "pose_mm_deg_rxryrz": values,
            "pose_m_rad_rxryrz": [
                values[0] / 1000.0, values[1] / 1000.0, values[2] / 1000.0,
                math.radians(values[3]), math.radians(values[4]), math.radians(values[5]),
            ],
            "pose_mm_rad_rxryrz": [
                values[0], values[1], values[2],
                math.radians(values[3]), math.radians(values[4]), math.radians(values[5]),
            ],
            "capture_status": "already_captured",
            "projection_check": {"ok": True, "reason": "actual_image_verified"},
        })

    t_tcp_camera = T_tcp_camera[:3, 3]
    for layer, rx, ry, rz in _new_orientation_schedule():
        R_base_tcp = _rpy_zyx_deg(rx, ry, rz)
        R_base_camera = R_base_tcp @ T_tcp_camera[:3, :3]
        distance = layer_distance[layer]
        camera_origin_base = board_center_base - distance * R_base_camera[:, 2]
        tcp_origin_base = camera_origin_base - R_base_tcp @ t_tcp_camera
        # 用户已验证的三层TCP高度保持不变；只补充XY与姿态激励，避免离线规划
        # 因相机偏置补偿把TCP降到已采集最低高度以下。
        tcp_origin_base[2] = layer_z_centers[layer]
        T_base_tcp = _transform(R_base_tcp, tcp_origin_base)
        projection = _project_board(T_base_tcp, T_tcp_camera, T_base_board, intrinsics)
        pose_mm_deg = [float(value) for value in tcp_origin_base] + [float(rx), float(ry), float(rz)]
        poses.append({
            "plan_index": len(poses) + 1,
            "source": "planned_new",
            "sample_index": None,
            "layer": layer,
            "pose_mm_deg_rxryrz": pose_mm_deg,
            "pose_m_rad_rxryrz": [
                pose_mm_deg[0] / 1000.0, pose_mm_deg[1] / 1000.0, pose_mm_deg[2] / 1000.0,
                math.radians(rx), math.radians(ry), math.radians(rz),
            ],
            "pose_mm_rad_rxryrz": [
                pose_mm_deg[0], pose_mm_deg[1], pose_mm_deg[2],
                math.radians(rx), math.radians(ry), math.radians(rz),
            ],
            "capture_status": "not_captured",
            "projection_check": projection,
        })

    planned = [pose for pose in poses if pose["source"] == "planned_new"]
    planned_values = np.asarray([pose["pose_mm_deg_rxryrz"] for pose in planned], dtype=np.float64)
    minimum_projection_margin = min(
        float(pose["projection_check"]["minimum_image_margin_px_approx"])
        for pose in planned
    )
    return {
        "record_type": "offline_handeye_40_pose_capture_plan",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_pose_count": len(poses),
        "existing_pose_count": len(samples),
        "new_pose_count": len(planned),
        "robot_connection_attempted": False,
        "robot_motion_command_sent": False,
        "motion_authorized": False,
        "robot_command_pose_field": "pose_mm_rad_rxryrz",
        "robot_command_units": {"translation": "mm", "rotation": "rad", "value_order": "x,y,z,rx,ry,rz"},
        "planning_review_units": {"translation": "mm", "rotation": "deg", "rotation_order": "Rz*Ry*Rx"},
        "source_sample_dir": str(sample_dir.resolve()),
        "source_handeye_path": str(handeye_path.resolve()),
        "source_handeye_validated": bool(handeye.get("validated", False)),
        "source_handeye_translation_norm_mm": handeye_translation_norm_mm,
        "board_center_base_mm_estimate": [float(value) for value in board_center_base],
        "layer_tcp_z_centers_mm": layer_z_centers,
        "layer_camera_board_distance_mm": layer_distance,
        "all_new_projection_checks_pass": all(pose["projection_check"].get("ok") for pose in planned),
        "minimum_new_pose_image_margin_px_approx": minimum_projection_margin,
        "all_pose_orientation_coverage": _orientation_coverage(poses),
        "new_pose_translation_envelope_mm": {
            "min_xyz": [float(value) for value in np.min(planned_values[:, :3], axis=0)],
            "max_xyz": [float(value) for value in np.max(planned_values[:, :3], axis=0)],
        },
        "manual_checks_required": [
            "逐点使用示教器低速预览并确认机器人可达",
            "确认相机、末端、标定板、桌面和周边设备无碰撞",
            "每次到位后等待机器人steady，再由手眼采集界面保存样本",
            "若完整标定板不在画面内，不采集该点；先小步退回再重新规划",
            "本计划使用未通过E7的历史候选矩阵只做取景估算，不能用于自动运动",
        ],
        "poses": poses,
    }


def _write_csv(path: Path, plan: dict[str, Any]) -> None:
    fields = [
        "plan_index", "source", "sample_index", "layer", "x_mm", "y_mm", "z_mm",
        "rx_rad", "ry_rad", "rz_rad", "projection_ok", "min_image_margin_px_approx", "capture_status",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pose in plan["poses"]:
            values = pose["pose_mm_rad_rxryrz"]
            projection = pose["projection_check"]
            writer.writerow({
                "plan_index": pose["plan_index"], "source": pose["source"],
                "sample_index": pose["sample_index"], "layer": pose["layer"],
                "x_mm": f"{values[0]:.6f}", "y_mm": f"{values[1]:.6f}", "z_mm": f"{values[2]:.6f}",
                "rx_rad": f"{values[3]:.9f}", "ry_rad": f"{values[4]:.9f}", "rz_rad": f"{values[5]:.9f}",
                "projection_ok": bool(projection.get("ok")),
                "min_image_margin_px_approx": projection.get("minimum_image_margin_px_approx", ""),
                "capture_status": pose["capture_status"],
            })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--handeye", type=Path, default=DEFAULT_HANDEYE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    resolved_sample_dir = args.sample_dir.resolve()
    if any(resolved_sample_dir.glob("sample_*.json")):
        plan = build_plan(resolved_sample_dir, args.handeye.resolve())
    elif args.output_json.exists():
        # 采集界面可能在规划生成后归档/清空活动样本；保留已生成的40点快照，
        # 只重建单位格式，避免因缺少原始样本把有效离线计划覆盖掉。
        plan = _load_json(args.output_json)
        if int(plan.get("total_pose_count", 0)) != 40 or len(plan.get("poses", [])) != 40:
            raise RuntimeError("existing plan is incomplete; refusing snapshot-only CSV rebuild")
        plan.pop("coordinate_units", None)
        for pose in plan["poses"]:
            pose_m_rad = [float(value) for value in pose["pose_m_rad_rxryrz"]]
            pose["pose_mm_rad_rxryrz"] = [
                pose_m_rad[0] * 1000.0, pose_m_rad[1] * 1000.0, pose_m_rad[2] * 1000.0,
                pose_m_rad[3], pose_m_rad[4], pose_m_rad[5],
            ]
        plan["robot_command_pose_field"] = "pose_mm_rad_rxryrz"
        plan["robot_command_units"] = {
            "translation": "mm", "rotation": "rad", "value_order": "x,y,z,rx,ry,rz",
        }
        plan["planning_review_units"] = {
            "translation": "mm", "rotation": "deg", "rotation_order": "Rz*Ry*Rx",
        }
        plan["rebuilt_from_existing_plan_snapshot"] = True
    else:
        plan = build_plan(resolved_sample_dir, args.handeye.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.output_csv, plan)
    print(json.dumps({
        "json": str(args.output_json.resolve()),
        "csv": str(args.output_csv.resolve()),
        "total": plan["total_pose_count"],
        "existing": plan["existing_pose_count"],
        "new": plan["new_pose_count"],
        "projection_checks_pass": plan["all_new_projection_checks_pass"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
