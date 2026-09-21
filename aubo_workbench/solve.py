#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按样本坐标源求手眼；正式链路为RGB-PnP，旧点云样本仅保留诊断。"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import AUTO_CAPTURE_CFG, BOARD_CFG, CAMERA_CFG, E7_HAND_EYE_CFG, ROBOT_CFG, SOLVE_CFG
from .geometry import (
    angle_span_deg,
    average_transforms,
    invert_transform,
    make_transform,
    rotation_angle_deg,
    rotation_error_deg,
    transform_to_pose6_rzryrx,
    transform_to_vec6,
    valid_rigid_transform as _valid_rigid_transform,
    vec6_to_transform,
)
from .io_utils import atomic_write_json, matrix_to_list
from .samples import CalibSample


def sample_calibration_frame(sample: CalibSample) -> str:
    frame = str(sample.calibration_frame or "pointcloud").strip().lower()
    return frame


def calibration_frame_for_samples(samples: list[CalibSample]) -> str:
    frames = sorted({sample_calibration_frame(sample) for sample in samples})
    if len(frames) != 1:
        raise RuntimeError(f"不能混合求解RGB与点云样本，当前坐标源={frames}")
    frame = frames[0]
    if frame not in {"rgb_camera", "pointcloud"}:
        raise RuntimeError(f"不支持的手眼样本坐标源：{frame}")
    missing = [
        int(sample.index) for sample in samples
        if (sample.T_rgb_board is None if frame == "rgb_camera" else sample.T_pointcloud_board is None)
    ]
    if missing:
        raise RuntimeError(f"{frame}样本缺少板位姿矩阵，indices={missing}")
    return frame


def sample_board_transform(sample: CalibSample, calibration_frame: str) -> np.ndarray:
    transform = sample.T_rgb_board if calibration_frame == "rgb_camera" else sample.T_pointcloud_board
    if transform is None:
        raise RuntimeError(f"sample {sample.index} missing {calibration_frame} board transform")
    return np.asarray(transform, dtype=np.float64)


def compute_board_base_stats(
    samples: list[CalibSample],
    T_pose_source_sensor: np.ndarray,
    calibration_frame: str | None = None,
) -> dict[str, Any]:
    frame = calibration_frame or calibration_frame_for_samples(samples)
    T_base_board_list = [
        sample.T_base_tool @ T_pose_source_sensor @ sample_board_transform(sample, frame)
        for sample in samples
    ]
    T_mean = average_transforms(T_base_board_list)
    trans_error_xyz = np.asarray([T[:3, 3] - T_mean[:3, 3] for T in T_base_board_list], dtype=np.float64)
    trans_errors = np.asarray([np.linalg.norm(v) for v in trans_error_xyz], dtype=np.float64)
    rot_errors = np.asarray([rotation_error_deg(T_mean[:3, :3], T[:3, :3]) for T in T_base_board_list], dtype=np.float64)
    return {
        "T_base_board_mean": T_mean,
        "translation_mean_mm": float(np.mean(trans_errors)),
        "translation_rmse_mm": float(np.sqrt(np.mean(trans_errors * trans_errors))),
        "translation_max_mm": float(np.max(trans_errors)),
        "rotation_mean_deg": float(np.mean(rot_errors)),
        "rotation_rmse_deg": float(np.sqrt(np.mean(rot_errors * rot_errors))),
        "rotation_max_deg": float(np.max(rot_errors)),
        "per_sample_translation_error_mm": [float(v) for v in trans_errors],
        "per_sample_translation_error_xyz_mm": [[float(x) for x in v] for v in trans_error_xyz],
        "per_sample_rotation_error_deg": [float(v) for v in rot_errors],
    }


def solve_handeye_opencv(samples: list[CalibSample]) -> tuple[np.ndarray, str, dict[str, Any]]:
    if len(samples) < SOLVE_CFG.min_samples_for_solve:
        raise RuntimeError(f"样本数量不足：{len(samples)} < {SOLVE_CFG.min_samples_for_solve}")

    calibration_frame = calibration_frame_for_samples(samples)
    R_gripper2base, t_gripper2base, R_target2cam, t_target2cam = [], [], [], []
    for s in samples:
        T_sensor_board = sample_board_transform(s, calibration_frame)
        R_gripper2base.append(s.T_base_tool[:3, :3].astype(np.float64))
        t_gripper2base.append(s.T_base_tool[:3, 3].reshape(3, 1).astype(np.float64))
        R_target2cam.append(T_sensor_board[:3, :3].astype(np.float64))
        t_target2cam.append(T_sensor_board[:3, 3].reshape(3, 1).astype(np.float64))

    methods = [
        ("TSAI", cv2.CALIB_HAND_EYE_TSAI),
        ("PARK", cv2.CALIB_HAND_EYE_PARK),
        ("HORAUD", cv2.CALIB_HAND_EYE_HORAUD),
        ("ANDREFF", cv2.CALIB_HAND_EYE_ANDREFF),
        ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS),
    ]
    results: dict[str, Any] = {}
    best_score = float("inf")
    best_name = ""
    best_T = None

    for name, method in methods:
        try:
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=method,
            )
            T_pose_source_sensor = make_transform(R_cam2gripper, np.asarray(t_cam2gripper).reshape(3))
            if not _valid_rigid_transform(T_pose_source_sensor):
                raise ValueError("calibrateHandEye returned an invalid rigid transform")
            stats = compute_board_base_stats(samples, T_pose_source_sensor, calibration_frame)
            if not all(
                np.isfinite(float(stats[key]))
                for key in (
                    "translation_mean_mm", "translation_rmse_mm", "translation_max_mm",
                    "rotation_mean_deg", "rotation_rmse_deg", "rotation_max_deg",
                )
            ):
                raise ValueError("hand-eye quality contains NaN or Inf")
            score = stats["translation_rmse_mm"] + 10.0 * stats["rotation_rmse_deg"]
            results[name] = {
                "ok": True,
                "score": float(score),
                "calibration_frame": calibration_frame,
                "T_pose_source_sensor": matrix_to_list(T_pose_source_sensor),
                "stats": {k: v for k, v in stats.items() if k != "T_base_board_mean"},
            }
            print(
                f"[SOLVE] {name}: trans_rmse={stats['translation_rmse_mm']:.4f} mm, "
                f"rot_rmse={stats['rotation_rmse_deg']:.5f} deg, score={score:.4f}"
            )
            if score < best_score:
                best_score = score
                best_name = name
                best_T = T_pose_source_sensor
        except Exception as exc:
            print(f"[WARN] calibrateHandEye {name} 失败: {exc}")
            results[name] = {"ok": False, "error": str(exc)}

    if best_T is None:
        raise RuntimeError("所有 OpenCV hand-eye 方法均失败")
    return best_T, best_name, results


def refine_handeye_nonlinear(samples: list[CalibSample], T_init: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if not SOLVE_CFG.enable_nonlinear_refine:
        return T_init, {"enabled": False}
    try:
        from scipy.optimize import least_squares  # type: ignore
    except Exception as exc:
        print("[WARN] scipy 不可用，跳过非线性优化:", exc)
        return T_init, {"enabled": False, "reason": "scipy_not_available"}

    T_base_board_init = compute_board_base_stats(samples, T_init)["T_base_board_mean"]
    x0 = np.r_[transform_to_vec6(T_init), transform_to_vec6(T_base_board_init)]
    rot_weight = float(SOLVE_CFG.nonlinear_rotation_weight_mm)

    def residual_func(x: np.ndarray) -> np.ndarray:
        T_tool_pc = vec6_to_transform(x[:6])
        T_base_board = vec6_to_transform(x[6:12])
        inv_base_board = invert_transform(T_base_board)
        residuals = []
        calibration_frame = calibration_frame_for_samples(samples)
        for s in samples:
            pred = s.T_base_tool @ T_tool_pc @ sample_board_transform(s, calibration_frame)
            err = inv_base_board @ pred
            rvec = transform_to_vec6(err)[:3]
            t = err[:3, 3]
            residuals.extend(t.tolist())
            residuals.extend((rot_weight * rvec).tolist())
        return np.asarray(residuals, dtype=np.float64)

    before = residual_func(x0)
    result = least_squares(
        residual_func, x0, method="trf", loss="soft_l1", f_scale=1.0, max_nfev=int(SOLVE_CFG.nonlinear_max_nfev),
    )
    T_refined = vec6_to_transform(result.x[:6])
    after = residual_func(result.x)
    info = {
        "enabled": True,
        "success": bool(result.success),
        "message": str(result.message),
        "cost_before": float(0.5 * np.sum(before * before)),
        "cost_after": float(0.5 * np.sum(after * after)),
        "nfev": int(result.nfev),
        "rotation_weight_mm": rot_weight,
    }
    print(
        f"[REFINE] nonlinear success={result.success}, "
        f"cost {info['cost_before']:.6f} -> {info['cost_after']:.6f}, nfev={result.nfev}"
    )
    return T_refined, info


def choose_refined_result(
    T_init: np.ndarray,
    init_stats: dict[str, Any],
    T_candidate: np.ndarray,
    candidate_stats: dict[str, Any],
    refine_info: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """非线性精修可能把解拉飞；只有明确变好才采纳，否则回退到 OpenCV 初解。"""
    info = dict(refine_info)
    if not info.get("enabled", False):
        info["accepted"] = False
        return T_init, init_stats, info

    reasons: list[str] = []
    if not info.get("success", False):
        reasons.append("nonlinear_solver_not_successful")
    if not _valid_rigid_transform(T_candidate):
        reasons.append("invalid_refined_transform")
    if not all(
        np.isfinite(float(candidate_stats[key]))
        for key in (
            "translation_mean_mm", "translation_rmse_mm", "translation_max_mm",
            "rotation_mean_deg", "rotation_rmse_deg", "rotation_max_deg",
        )
    ):
        reasons.append("refined_quality_not_finite")
    if candidate_stats["translation_mean_mm"] > init_stats["translation_mean_mm"] + 0.03:
        reasons.append("translation_mean_worse")
    if candidate_stats["translation_rmse_mm"] > init_stats["translation_rmse_mm"] + 0.03:
        reasons.append("translation_rmse_worse")
    if candidate_stats["translation_max_mm"] > init_stats["translation_max_mm"] + 0.20:
        reasons.append("translation_max_worse")

    rot_mean_limit = max(float(init_stats["rotation_mean_deg"]) + 0.50, 2.00)
    rot_max_limit = max(float(init_stats["rotation_max_deg"]) + 1.00, 3.00)
    if candidate_stats["rotation_mean_deg"] > rot_mean_limit:
        reasons.append("rotation_mean_exploded")
    if candidate_stats["rotation_max_deg"] > rot_max_limit:
        reasons.append("rotation_max_exploded")

    if reasons:
        info["accepted"] = False
        info["reject_reasons"] = reasons
        print(f"[REFINE] nonlinear result rejected; use initial OpenCV hand-eye. reasons={reasons}")
        return T_init, init_stats, info

    info["accepted"] = True
    return T_candidate, candidate_stats, info


def solve_handeye_estimate(samples: list[CalibSample], quiet: bool = False) -> dict[str, Any]:
    def _run() -> dict[str, Any]:
        T_init, best_method, all_method_results = solve_handeye_opencv(samples)
        init_stats = compute_board_base_stats(samples, T_init)
        T_candidate, refine_info = refine_handeye_nonlinear(samples, T_init)
        candidate_stats = compute_board_base_stats(samples, T_candidate)
        T_final, final_stats, refine_info = choose_refined_result(
            T_init, init_stats, T_candidate, candidate_stats, refine_info
        )
        if not _valid_rigid_transform(T_final):
            raise RuntimeError("最终手眼变换不是合法的刚体变换")
        return {
            "best_method": best_method,
            "all_method_results": all_method_results,
            "T_init": T_init,
            "T_final": T_final,
            "initial_quality": init_stats,
            "quality": final_stats,
            "nonlinear_refine": refine_info,
        }

    if not quiet:
        return _run()

    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return _run()


def sample_quality_flags(sample: CalibSample) -> list[str]:
    flags: list[str] = []
    if sample_calibration_frame(sample) == "rgb_camera":
        if sample.rgb_pnp_inlier_count < AUTO_CAPTURE_CFG.min_rgb_pnp_inliers:
            flags.append(f"rgb_pnp_inliers<{AUTO_CAPTURE_CFG.min_rgb_pnp_inliers}")
        if (
            not np.isfinite(sample.rgb_reprojection_rmse_px)
            or sample.rgb_reprojection_rmse_px > AUTO_CAPTURE_CFG.max_rgb_reprojection_rmse_px
        ):
            flags.append(f"rgb_reprojection_rmse>{AUTO_CAPTURE_CFG.max_rgb_reprojection_rmse_px}")
        if (
            not np.isfinite(sample.rgb_reprojection_max_px)
            or sample.rgb_reprojection_max_px > AUTO_CAPTURE_CFG.max_rgb_reprojection_error_px
        ):
            flags.append(f"rgb_reprojection_max>{AUTO_CAPTURE_CFG.max_rgb_reprojection_error_px}")
        return flags
    if sample.valid_3d_count < AUTO_CAPTURE_CFG.min_valid_3d_corners:
        flags.append(f"valid3d<{AUTO_CAPTURE_CFG.min_valid_3d_corners}")
    if np.isfinite(sample.corner_rmse_mm) and sample.corner_rmse_mm > AUTO_CAPTURE_CFG.max_corner_rmse_mm:
        flags.append(f"corner_rmse>{AUTO_CAPTURE_CFG.max_corner_rmse_mm}")
    if np.isfinite(sample.plane_rmse_mm) and sample.plane_rmse_mm > AUTO_CAPTURE_CFG.max_plane_rmse_mm:
        flags.append(f"plane_rmse>{AUTO_CAPTURE_CFG.max_plane_rmse_mm}")
    return flags


def sample_pose_values(sample: CalibSample) -> list[float]:
    snap = sample.robot_snapshot or {}
    values = snap.get("pose_values", [])
    if isinstance(values, (list, tuple)):
        return [float(v) for v in values[:6]]
    return []


def _relative_rotation_axis_report(rotations: list[np.ndarray]) -> dict[str, Any]:
    """判断相对旋转是否几乎都绕同一根轴，避免只看欧拉角跨度。"""
    axes: list[np.ndarray] = []
    for i in range(len(rotations)):
        for j in range(i + 1, len(rotations)):
            rvec, _ = cv2.Rodrigues(rotations[i].T @ rotations[j])
            vector = rvec.reshape(3).astype(np.float64)
            angle_deg = float(np.linalg.norm(vector) * 180.0 / np.pi)
            if angle_deg < 5.0:
                continue
            norm = float(np.linalg.norm(vector))
            if norm > 1e-12:
                axes.append(vector / norm)
    if len(axes) < 3:
        return {
            "usable_pair_count": len(axes),
            "singular_values": [],
            "secondary_to_primary_ratio": 0.0,
            "axes_nearly_collinear": len(axes) > 0,
        }
    singular_values = np.linalg.svd(np.asarray(axes, dtype=np.float64), compute_uv=False)
    ratio = float(singular_values[1] / max(singular_values[0], 1e-12))
    return {
        "usable_pair_count": len(axes),
        "singular_values": [float(value) for value in singular_values],
        "secondary_to_primary_ratio": ratio,
        "axes_nearly_collinear": bool(ratio < 0.15),
    }


def pose_coverage_report(samples: list[CalibSample]) -> dict[str, Any]:
    poses = [p for p in (sample_pose_values(s) for s in samples) if len(p) >= 6]
    if not poses:
        return {"ok": False, "reason": "no_robot_pose_values"}

    arr = np.asarray(poses, dtype=np.float64)
    linear: dict[str, Any] = {}
    for i, name in enumerate(("X", "Y", "Z", "A", "B", "C")):
        col = arr[:, i]
        linear[name] = {
            "min": float(np.min(col)), "max": float(np.max(col)),
            "range": float(np.max(col) - np.min(col)), "std": float(np.std(col)),
        }

    rotations = [s.T_base_tool[:3, :3].astype(np.float64) for s in samples]
    axis_report = _relative_rotation_axis_report(rotations)
    rel_angles = [
        rotation_angle_deg(rotations[i].T @ rotations[j])
        for i in range(len(rotations)) for j in range(i + 1, len(rotations))
    ]
    rel = np.asarray(rel_angles, dtype=np.float64)

    warnings: list[str] = []
    if linear["A"]["range"] < 10.0:
        warnings.append("A_range_lt_10deg")
    if angle_span_deg(arr[:, 5]) < 50.0:
        warnings.append("C_circular_span_lt_50deg")
    if rel.size and float(np.mean(rel)) < 15.0:
        warnings.append("relative_rotation_mean_lt_15deg")
    if axis_report["axes_nearly_collinear"]:
        warnings.append("relative_rotation_axes_nearly_collinear")

    return {
        "ok": True,
        "pose_ranges": linear,
        "angle_circular_span_deg": {
            "A": angle_span_deg(arr[:, 3]), "B": angle_span_deg(arr[:, 4]), "C": angle_span_deg(arr[:, 5]),
        },
        "relative_rotation_deg": {
            "min": float(np.min(rel)) if rel.size else 0.0,
            "mean": float(np.mean(rel)) if rel.size else 0.0,
            "max": float(np.max(rel)) if rel.size else 0.0,
            "std": float(np.std(rel)) if rel.size else 0.0,
            "pair_count": int(rel.size),
            "pair_count_lt_5deg": int(np.sum(rel < 5.0)) if rel.size else 0,
            "pair_count_lt_10deg": int(np.sum(rel < 10.0)) if rel.size else 0,
        },
        "relative_rotation_axis_coverage": axis_report,
        "warnings": warnings,
    }


def solve_and_save(
    samples: list[CalibSample],
    *,
    validation_samples: list[CalibSample] | None = None,
    fixed_validation_split: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    fit_samples = list(samples)
    held_out_samples = list(validation_samples or [])
    active_samples = sorted(
        [*fit_samples, *held_out_samples], key=lambda item: int(item.index),
    )

    # 先检查整批样本的位姿源，再尝试复用固定 E7 留出集。否则当工作区残留
    # 旧 split 文件、当前诊断样本只有 8 组时，错误会被“至少需要 11 组”遮住，
    # 现场会看不到真正的 tcp/tool 混用原因。
    if active_samples:
        active_pose_sources = sorted({
            str((sample.robot_snapshot or {}).get("pose_source") or "").strip().lower()
            for sample in active_samples
        })
        if len(active_pose_sources) != 1 or not active_pose_sources[0]:
            raise RuntimeError(
                f"样本位姿源必须唯一且显式，当前={active_pose_sources}"
            )

    if fixed_validation_split is None and held_out_samples == []:
        from .e7_handeye import fixed_validation_split_path, get_or_create_fixed_e7_split

        split_path = fixed_validation_split_path(active_samples)
        if len(active_samples) < SOLVE_CFG.min_samples_for_solve:
            if split_path.is_file():
                get_or_create_fixed_e7_split(active_samples, E7_HAND_EYE_CFG, persist=True)
            print(
                f"[WARN] 样本不足，至少需要 {SOLVE_CFG.min_samples_for_solve} 个，"
                f"当前 {len(active_samples)} 个"
            )
            return None
        if len(active_samples) >= int(E7_HAND_EYE_CFG.minimum_total_poses) or split_path.is_file():
            fit_samples, held_out_samples, fixed_validation_split = get_or_create_fixed_e7_split(
                active_samples, E7_HAND_EYE_CFG, persist=True,
            )
            active_samples = sorted(
                [*fit_samples, *held_out_samples], key=lambda item: int(item.index),
            )
    if len(fit_samples) < SOLVE_CFG.min_samples_for_solve:
        print(
            f"[WARN] 固定验证集占用后标定样本不足，至少需要 {SOLVE_CFG.min_samples_for_solve} 个，"
            f"当前 {len(fit_samples)} 个"
        )
        return None

    pose_sources = sorted({
        str((sample.robot_snapshot or {}).get("pose_source") or "").strip().lower()
        for sample in fit_samples
    })
    if len(pose_sources) != 1 or not pose_sources[0]:
        raise RuntimeError(f"样本位姿源必须唯一且显式，当前={pose_sources}")
    pose_source = pose_sources[0]
    calibration_frame = calibration_frame_for_samples(fit_samples)
    is_rgb = calibration_frame == "rgb_camera"
    sensor_frame_name = "gemini435le_rgb_optical_frame" if is_rgb else "gemini435le_pointcloud_xyz_map_frame"
    method_name = "charuco_rgb_pnp_handeye" if is_rgb else "legacy_charuco_pointcloud_handeye_diagnostic"
    reference_frame = {
        "tcp": "tool_tcp",
        "tool": "robot_tool_or_flange",
        "manual": "manual_pose_reference",
    }.get(pose_source, f"robot_pose_source:{pose_source}")

    estimate = solve_handeye_estimate(fit_samples, quiet=True)
    T_final = estimate["T_final"]
    quality = {k: v for k, v in estimate["quality"].items() if k != "T_base_board_mean"}
    validation_quality = None
    if held_out_samples:
        from .e7_handeye import validation_stats_against_reference

        validation_quality = validation_stats_against_reference(
            held_out_samples, T_final, estimate["quality"]["T_base_board_mean"],
        )
        validation_quality = {
            key: value for key, value in validation_quality.items()
            if key not in {"T_base_board_reference", "transforms"}
        }
    rms_limit = float(E7_HAND_EYE_CFG.maximum_validation_center_scatter_rms_mm)
    max_limit = float(E7_HAND_EYE_CFG.maximum_validation_center_scatter_max_mm)
    measured = [quality] + ([validation_quality] if validation_quality is not None else [])
    numeric_pass = all(
        np.isfinite(q["translation_rmse_mm"]) and np.isfinite(q["translation_max_mm"])
        and q["translation_rmse_mm"] <= rms_limit and q["translation_max_mm"] <= max_limit
        for q in measured
    )
    enough_validation = (
        len(active_samples) >= E7_HAND_EYE_CFG.minimum_total_poses
        and len(held_out_samples) >= E7_HAND_EYE_CFG.minimum_validation_poses
        and len(held_out_samples) / len(active_samples) >= E7_HAND_EYE_CFG.minimum_validation_fraction
    )
    if not numeric_pass:
        conclusion = "未达标"
        next_action = "检查图像与机器人位姿对应、相机安装和标定板尺寸，补采重复姿态核对。"
    elif not enough_validation:
        conclusion = "待验证"
        next_action = "独立验证样本不足，继续采集不同姿态。"
    else:
        conclusion = "数值达标，待完整验证"
        next_action = "执行独立验证，核对采集条件与姿态覆盖。"

    session_consistency = None
    if is_rgb:
        from .e7_handeye import _tcp_offset_consistency

        session_consistency = _tcp_offset_consistency(active_samples, E7_HAND_EYE_CFG)
    output = {
        "record_type": "handeye_rgb_diagnostic_result" if is_rgb else "handeye_pointcloud_legacy_diagnostic_result",
        "record_policy": "diagnostic_current_replace_on_refresh",
        "validated": False,
        "do_not_use_for_motion": True,
        "production_eligible": False,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "conclusion": conclusion,
        "next_action": next_action,
        "precision_target": {"translation_rmse_mm_max": rms_limit, "translation_max_mm_max": max_limit},
        "method": method_name,
        "calibration_frame": calibration_frame,
        "rgb_coordinate_convention": "opencv_optical_x_right_y_down_z_forward" if is_rgb else None,
        "frame_definition": {
            "B": "robot_base_or_world", "T": reference_frame, "S": sensor_frame_name,
            "Q": "charuco_board", "T_pose_source_sensor": "^T T_S",
            "runtime_formula": "P_base = T_base_pose_source @ T_pose_source_sensor @ P_sensor",
        },
        "pose_source": pose_source,
        "raw_sample_count": len(active_samples),
        "sample_count": len(fit_samples),
        "used_sample_indices": [int(s.index) for s in fit_samples],
        "calibration_sample_indices": [int(s.index) for s in fit_samples],
        "validation_sample_indices": [int(s.index) for s in held_out_samples],
        "validation_poses_excluded_from_fit": bool(held_out_samples),
        "fixed_validation_split": fixed_validation_split,
        "best_opencv_method": estimate["best_method"],
        "T_pose_source_sensor": matrix_to_list(T_final),
        "T_pose_source_sensor_pose6_rzryrx_mm_deg": list(transform_to_pose6_rzryrx(T_final)),
        "T_pose_source_sensor_initial_opencv": matrix_to_list(estimate["T_init"]),
        "quality": quality,
        "validation_quality": validation_quality,
        "pose_coverage": pose_coverage_report(fit_samples),
        "T_base_board_mean": matrix_to_list(estimate["quality"]["T_base_board_mean"]),
        "nonlinear_refine": estimate["nonlinear_refine"],
        "opencv_method_results": estimate["all_method_results"],
        "board": asdict(BOARD_CFG),
        "camera": asdict(CAMERA_CFG),
        "capture": asdict(AUTO_CAPTURE_CFG),
        "session_consistency": session_consistency,
        "robot": {
            "robot_brand": "AUBO", "ip": ROBOT_CFG.ip, "rpc_port": ROBOT_CFG.rpc_port,
            "pose_source": pose_source, "sdk_pose_units": {"xyz": "m", "rpy": "rad"},
            "solver_pose_units": {"xyz": "mm", "rotation": "matrix"},
        },
    }
    if pose_source == "tcp":
        output["T_tcp_rgb_camera" if is_rgb else "T_tcp_pointcloud"] = output["T_pose_source_sensor"]
    elif pose_source == "tool":
        output["T_robot_tool_rgb_camera" if is_rgb else "T_robot_tool_pointcloud"] = output["T_pose_source_sensor"]
    if SOLVE_CFG.also_write_compatible_key_t_tooltcp_cam and pose_source == "tcp" and is_rgb:
        output["t_tooltcp_cam"] = output["T_tcp_rgb_camera"]

    out_path = atomic_write_json(Path(SOLVE_CFG.output_json), output)
    print(format_handeye_summary(output))
    print(f"[结果文件] {out_path}")
    return output


def format_handeye_summary(result: dict[str, Any]) -> str:
    """界面和日志共用同一份结论，拟合误差与独立验证误差分开显示。"""
    quality = result["quality"]
    target = result["precision_target"]
    lines = [
        f"结论：{result['conclusion']}",
        f"拟合 {result['sample_count']} 组：平移 RMS {quality['translation_rmse_mm']:.3f} mm，"
        f"最大 {quality['translation_max_mm']:.3f} mm",
    ]
    validation = result.get("validation_quality")
    if validation is None:
        lines.append("独立验证：样本不足")
    else:
        lines.append(
            f"验证 {len(result['validation_sample_indices'])} 组：平移 RMS "
            f"{validation['translation_rmse_mm']:.3f} mm，最大 {validation['translation_max_mm']:.3f} mm"
        )
    lines.extend([
        f"目标：RMS ≤ {target['translation_rmse_mm_max']:.2f} mm，"
        f"最大 ≤ {target['translation_max_mm_max']:.2f} mm",
        result["next_action"],
    ])
    return "\n".join(lines)
