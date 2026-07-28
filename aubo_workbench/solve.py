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

from .config import AUTO_CAPTURE_CFG, AUTO_PRUNE_CFG, BOARD_CFG, CAMERA_CFG, CONFLICT_DIAG_CFG, E7_HAND_EYE_CFG, ROBOT_CFG, SOLVE_CFG
from .geometry import (
    angle_span_deg,
    average_transforms,
    fmt_vec,
    invert_transform,
    make_transform,
    rotation_angle_deg,
    rotation_error_deg,
    transform_to_pose6_rzryrx,
    transform_to_vec6,
    vec6_to_transform,
)
from .io_utils import atomic_write_json, matrix_to_list, timestamp_str
from .samples import CalibSample, archive_removed_samples, rewrite_sample_csv


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
            stats = compute_board_base_stats(samples, T_pose_source_sensor, calibration_frame)
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


def _quality_summary(estimate: dict[str, Any], sample_count: int, removed_indices: list[int] | None = None) -> dict[str, Any]:
    q = estimate["quality"]
    return {
        "sample_count": int(sample_count),
        "best_method": estimate["best_method"],
        "translation_mean_mm": float(q["translation_mean_mm"]),
        "translation_rmse_mm": float(q["translation_rmse_mm"]),
        "translation_max_mm": float(q["translation_max_mm"]),
        "rotation_mean_deg": float(q["rotation_mean_deg"]),
        "rotation_rmse_deg": float(q["rotation_rmse_deg"]),
        "rotation_max_deg": float(q["rotation_max_deg"]),
        "removed_indices": [int(v) for v in (removed_indices or [])],
    }


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
    if sample.valid_3d_count < AUTO_PRUNE_CFG.min_valid_3d_count:
        flags.append(f"valid3d<{AUTO_PRUNE_CFG.min_valid_3d_count}")
    if np.isfinite(sample.corner_rmse_mm) and sample.corner_rmse_mm > AUTO_PRUNE_CFG.max_corner_rmse_mm:
        flags.append(f"corner_rmse>{AUTO_PRUNE_CFG.max_corner_rmse_mm}")
    if np.isfinite(sample.plane_rmse_mm) and sample.plane_rmse_mm > AUTO_PRUNE_CFG.max_plane_rmse_mm:
        flags.append(f"plane_rmse>{AUTO_PRUNE_CFG.max_plane_rmse_mm}")
    return flags


def sample_pose_values(sample: CalibSample) -> list[float]:
    snap = sample.robot_snapshot or {}
    values = snap.get("pose_values", [])
    if isinstance(values, (list, tuple)):
        return [float(v) for v in values[:6]]
    return []


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
        "warnings": warnings,
    }


def sample_residual_report(samples: list[CalibSample], quality: dict[str, Any]) -> list[dict[str, Any]]:
    trans = quality.get("per_sample_translation_error_mm", [])
    trans_xyz = quality.get("per_sample_translation_error_xyz_mm", [])
    rots = quality.get("per_sample_rotation_error_deg", [])
    rows: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        residual = float(trans[i]) if i < len(trans) else float("nan")
        residual_xyz = trans_xyz[i] if i < len(trans_xyz) else [float("nan")] * 3
        rot = float(rots[i]) if i < len(rots) else float("nan")
        severity = "ok"
        if np.isfinite(residual) and residual >= CONFLICT_DIAG_CFG.bad_residual_mm:
            severity = "bad"
        elif np.isfinite(residual) and residual >= CONFLICT_DIAG_CFG.suspect_residual_mm:
            severity = "suspect"
        rows.append({
            "index": int(sample.index), "timestamp": sample.timestamp, "residual_mm": residual,
            "residual_xyz_mm": [float(v) for v in residual_xyz], "rotation_error_deg": rot,
            "severity": severity, "pose_values": sample_pose_values(sample),
            "quality_flags": sample_quality_flags(sample), "board_status": sample.board_status,
            "calibration_frame": sample_calibration_frame(sample),
            "charuco_count": int(sample.charuco_count), "valid_3d_count": int(sample.valid_3d_count),
            "corner_rmse_mm": float(sample.corner_rmse_mm), "plane_rmse_mm": float(sample.plane_rmse_mm),
            "rgb_pnp_inlier_count": int(sample.rgb_pnp_inlier_count),
            "rgb_reprojection_rmse_px": float(sample.rgb_reprojection_rmse_px),
            "rgb_reprojection_max_px": float(sample.rgb_reprojection_max_px),
        })
    return sorted(rows, key=lambda item: item["residual_mm"], reverse=True)


def diagnose_sample_conflicts(samples: list[CalibSample], baseline_estimate: dict[str, Any]) -> dict[str, Any]:
    cfg = CONFLICT_DIAG_CFG
    report: dict[str, Any] = {"enabled": bool(cfg.enable_on_solve), "config": asdict(cfg)}
    if not cfg.enable_on_solve:
        return report
    if len(samples) < SOLVE_CFG.min_samples_for_solve:
        report["status"] = "not_enough_samples"
        return report

    remaining = list(samples)
    removed: list[dict[str, Any]] = []
    current = baseline_estimate
    path: list[dict[str, Any]] = [_quality_summary(current, len(remaining), [])]

    max_remove = min(int(cfg.max_remove_for_report), max(0, len(samples) - int(cfg.min_report_samples)))
    for _ in range(max_remove):
        if len(remaining) <= max(SOLVE_CFG.min_samples_for_solve, int(cfg.min_report_samples)):
            break

        current_quality = current["quality"]
        current_errors = current_quality.get("per_sample_translation_error_mm", [])
        best: dict[str, Any] | None = None
        for i, sample in enumerate(remaining):
            test_samples = remaining[:i] + remaining[i + 1:]
            if len(test_samples) < SOLVE_CFG.min_samples_for_solve:
                continue
            try:
                estimate = solve_handeye_estimate(test_samples, quiet=True)
            except Exception:
                continue
            q = estimate["quality"]
            mean_improve = float(current_quality["translation_mean_mm"] - q["translation_mean_mm"])
            rmse_improve = float(current_quality["translation_rmse_mm"] - q["translation_rmse_mm"])
            max_improve = float(current_quality["translation_max_mm"] - q["translation_max_mm"])
            key = (
                float(q["translation_mean_mm"]), float(q["translation_rmse_mm"]),
                float(q["translation_max_mm"]), float(q["rotation_mean_deg"]),
            )
            item = {
                "key": key, "sample": sample, "sample_pos": i, "estimate": estimate,
                "residual_before_remove_mm": float(current_errors[i]) if i < len(current_errors) else float("nan"),
                "mean_improve_mm": mean_improve, "rmse_improve_mm": rmse_improve, "max_improve_mm": max_improve,
            }
            if best is None or item["key"] < best["key"]:
                best = item

        if best is None:
            break
        if best["mean_improve_mm"] < 0.005 and best["rmse_improve_mm"] < 0.005 and best["max_improve_mm"] < 0.05:
            break

        sample = best["sample"]
        removed.append({
            "index": int(sample.index), "timestamp": sample.timestamp,
            "residual_before_remove_mm": float(best["residual_before_remove_mm"]),
            "mean_improve_mm": float(best["mean_improve_mm"]), "rmse_improve_mm": float(best["rmse_improve_mm"]),
            "max_improve_mm": float(best["max_improve_mm"]), "pose_values": sample_pose_values(sample),
            "quality_flags": sample_quality_flags(sample),
        })
        remaining.pop(int(best["sample_pos"]))
        current = best["estimate"]
        path.append(_quality_summary(current, len(remaining), [item["index"] for item in removed]))

    initial_rows = sample_residual_report(samples, baseline_estimate["quality"])
    accepted_entries = [
        item for item in path
        if item["sample_count"] >= cfg.min_accept_samples
        and item["translation_mean_mm"] <= cfg.target_translation_mean_mm
        and item["translation_rmse_mm"] <= cfg.target_translation_rmse_mm
        and item["translation_max_mm"] <= cfg.target_translation_max_mm
    ]
    best_with_enough_samples = min(
        (item for item in path if item["sample_count"] >= cfg.min_accept_samples),
        key=lambda item: (item["translation_mean_mm"], item["translation_rmse_mm"], item["translation_max_mm"]),
        default=None,
    )
    best_any = min(path, key=lambda item: (item["translation_mean_mm"], item["translation_rmse_mm"], item["translation_max_mm"]))

    if accepted_entries:
        status = "accepted_consensus_found"
        recommended = accepted_entries[0]
    elif (
        best_any["translation_mean_mm"] <= cfg.target_translation_mean_mm
        and best_any["translation_rmse_mm"] <= cfg.target_translation_rmse_mm
        and best_any["translation_max_mm"] <= cfg.target_translation_max_mm
    ):
        status = "only_small_subset_meets_target"
        recommended = best_any
    else:
        status = "no_consensus_meets_target"
        recommended = best_with_enough_samples or best_any

    report.update({
        "status": status,
        "baseline": path[0],
        "pose_coverage": pose_coverage_report(samples),
        "initial_residuals_sorted": initial_rows,
        "initial_suspect_indices": [int(item["index"]) for item in initial_rows if item["severity"] in ("suspect", "bad")],
        "initial_bad_indices": [int(item["index"]) for item in initial_rows if item["severity"] == "bad"],
        "greedy_removed_order": removed,
        "greedy_path": path,
        "best_with_enough_samples": best_with_enough_samples,
        "best_any_subset": best_any,
        "recommended_entry": recommended,
        "accepted_for_replacing_final_result": bool(status == "accepted_consensus_found"),
        "note": (
            "该诊断只用于识别冲突样本；如果达到 1 mm 的子集样本数过少，"
            "应补拍同类姿态的新样本，而不是直接把少量点作为通用手眼结果。"
        ),
    })
    return report


def auto_prune_samples(samples: list[CalibSample]) -> tuple[list[CalibSample], list[dict[str, Any]], dict[str, Any]]:
    if len(samples) < SOLVE_CFG.min_samples_for_solve:
        print(f"[WARN] 样本不足，不能自动剔除：{len(samples)} < {SOLVE_CFG.min_samples_for_solve}")
        return samples, [], {}

    remaining = list(samples)
    removed: list[dict[str, Any]] = []
    minimum_remaining = max(
        SOLVE_CFG.min_samples_for_solve,
        E7_HAND_EYE_CFG.minimum_total_poses,
        AUTO_PRUNE_CFG.min_remaining_samples,
    )
    before_estimate = solve_handeye_estimate(remaining, quiet=False)
    before_quality = before_estimate["quality"]

    print(
        f"[AUTO] 剔除前: n={len(remaining)}, mean={before_quality['translation_mean_mm']:.4f} mm, "
        f"rmse={before_quality['translation_rmse_mm']:.4f} mm, max={before_quality['translation_max_mm']:.4f} mm, "
        f"rot_mean={before_quality['rotation_mean_deg']:.4f} deg"
    )

    current_estimate = before_estimate
    for _ in range(AUTO_PRUNE_CFG.max_remove_per_run):
        if len(remaining) <= minimum_remaining:
            print(f"[AUTO] 已达到最低保留样本数 {minimum_remaining}，停止剔除。")
            break

        current_quality = current_estimate["quality"]
        if (
            current_quality["translation_mean_mm"] <= AUTO_PRUNE_CFG.target_translation_mean_mm
            and current_quality["translation_max_mm"] <= AUTO_PRUNE_CFG.target_translation_max_mm
        ):
            break

        trans_errors = current_quality["per_sample_translation_error_mm"]
        candidates: list[dict[str, Any]] = []

        for i, sample in enumerate(remaining):
            residual = float(trans_errors[i])
            flags = sample_quality_flags(sample)
            is_candidate = (
                residual >= AUTO_PRUNE_CFG.max_translation_error_mm
                or flags
                or (
                    current_quality["translation_max_mm"] > AUTO_PRUNE_CFG.target_translation_max_mm
                    and residual >= AUTO_PRUNE_CFG.target_translation_max_mm
                )
            )
            if not is_candidate:
                continue

            test_samples = remaining[:i] + remaining[i + 1:]
            if len(test_samples) < minimum_remaining:
                continue
            try:
                test_estimate = solve_handeye_estimate(test_samples, quiet=True)
            except Exception as exc:
                print(f"[WARN] 自动剔除评估失败 index={sample.index}: {exc}")
                continue

            test_quality = test_estimate["quality"]
            mean_improve = current_quality["translation_mean_mm"] - test_quality["translation_mean_mm"]
            rmse_improve = current_quality["translation_rmse_mm"] - test_quality["translation_rmse_mm"]
            max_improve = current_quality["translation_max_mm"] - test_quality["translation_max_mm"]
            rot_improve = current_quality["rotation_mean_deg"] - test_quality["rotation_mean_deg"]
            accepted = (
                mean_improve >= AUTO_PRUNE_CFG.min_mean_improvement_mm
                or max_improve >= AUTO_PRUNE_CFG.min_max_improvement_mm
                or (flags and mean_improve >= -0.02 and max_improve >= -0.10)
            )
            if not accepted:
                continue

            score = mean_improve * 3.0 + rmse_improve * 1.5 + max_improve * 0.4 + max(rot_improve, 0.0) * 0.02
            candidates.append({
                "sample": sample, "index": sample.index, "timestamp": sample.timestamp,
                "pose_values": sample_pose_values(sample), "residual_mm": residual, "quality_flags": flags,
                "mean_improve_mm": float(mean_improve), "rmse_improve_mm": float(rmse_improve),
                "max_improve_mm": float(max_improve), "rotation_mean_improve_deg": float(rot_improve),
                "score": float(score), "test_estimate": test_estimate,
            })

        if not candidates:
            break

        best = max(candidates, key=lambda item: item["score"])
        sample = best["sample"]
        remaining = [s for s in remaining if s is not sample]
        current_estimate = best["test_estimate"]
        removed.append({k: v for k, v in best.items() if k != "test_estimate"})
        print(
            f"[AUTO] 剔除 index={sample.index}, residual={best['residual_mm']:.4f} mm, "
            f"d_mean={best['mean_improve_mm']:.4f}, d_max={best['max_improve_mm']:.4f}, flags={best['quality_flags']}"
        )

    after_estimate = solve_handeye_estimate(remaining, quiet=True)
    after_quality = after_estimate["quality"]
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": asdict(AUTO_PRUNE_CFG),
        "before": {
            "sample_count": len(samples), "best_method": before_estimate["best_method"],
            "translation_mean_mm": before_quality["translation_mean_mm"],
            "translation_rmse_mm": before_quality["translation_rmse_mm"],
            "translation_max_mm": before_quality["translation_max_mm"],
            "rotation_mean_deg": before_quality["rotation_mean_deg"],
        },
        "after": {
            "sample_count": len(remaining), "best_method": after_estimate["best_method"],
            "translation_mean_mm": after_quality["translation_mean_mm"],
            "translation_rmse_mm": after_quality["translation_rmse_mm"],
            "translation_max_mm": after_quality["translation_max_mm"],
            "rotation_mean_deg": after_quality["rotation_mean_deg"],
        },
    }
    return remaining, removed, report


def auto_prune_and_save(samples: list[CalibSample]) -> list[CalibSample]:
    remaining, removed, report = auto_prune_samples(samples)
    if removed:
        archive_dir = archive_removed_samples(removed, report, timestamp_str())
        rewrite_sample_csv(remaining)
        print(f"[AUTO] 已归档坏样本 {len(removed)} 个: {archive_dir}")
        print(f"[AUTO] CSV 已重写，剩余有效样本 {len(remaining)} 个")
    else:
        print("[AUTO] 未发现满足自动剔除条件的坏样本")
    solve_and_save(remaining)
    return remaining


def solve_and_save(samples: list[CalibSample]) -> dict[str, Any] | None:
    if len(samples) < SOLVE_CFG.min_samples_for_solve:
        print(f"[WARN] 样本不足，至少需要 {SOLVE_CFG.min_samples_for_solve} 个，当前 {len(samples)} 个")
        return None

    pose_sources = sorted({
        str((sample.robot_snapshot or {}).get("pose_source") or "").strip().lower()
        for sample in samples
    })
    if len(pose_sources) != 1 or not pose_sources[0]:
        raise RuntimeError(f"样本位姿源必须唯一且显式，当前={pose_sources}")
    pose_source = pose_sources[0]
    calibration_frame = calibration_frame_for_samples(samples)
    is_rgb = calibration_frame == "rgb_camera"
    sensor_frame_name = "gemini435le_rgb_optical_frame" if is_rgb else "gemini435le_pointcloud_xyz_map_frame"
    method_name = "charuco_rgb_pnp_handeye" if is_rgb else "legacy_charuco_pointcloud_handeye_diagnostic"
    reference_frame = {
        "tcp": "tool_tcp",
        "tool": "robot_tool_or_flange",
        "manual": "manual_pose_reference",
    }.get(pose_source, f"robot_pose_source:{pose_source}")

    print("=" * 70)
    print(
        f"[SOLVE] 开始诊断求 T_pose_source_sensor，pose_source={pose_source}，"
        f"calibration_frame={calibration_frame}，样本数={len(samples)}"
    )
    T_init, best_method, all_method_results = solve_handeye_opencv(samples)
    init_stats = compute_board_base_stats(samples, T_init)

    T_candidate, refine_info = refine_handeye_nonlinear(samples, T_init)
    candidate_stats = compute_board_base_stats(samples, T_candidate)
    T_final, final_stats, refine_info = choose_refined_result(T_init, init_stats, T_candidate, candidate_stats, refine_info)
    pose6 = transform_to_pose6_rzryrx(T_final)
    raw_sample_count = len(samples)
    used_samples = list(samples)
    all_sample_quality = {k: v for k, v in final_stats.items() if k != "T_base_board_mean"}
    all_sample_estimate = {
        "best_method": best_method, "all_method_results": all_method_results, "T_init": T_init, "T_final": T_final,
        "initial_quality": init_stats, "quality": final_stats, "nonlinear_refine": refine_info,
    }
    conflict_report = diagnose_sample_conflicts(samples, all_sample_estimate)
    conflict_report["final_result_uses_consensus_subset"] = False

    if conflict_report.get("accepted_for_replacing_final_result"):
        recommended = conflict_report.get("recommended_entry") or {}
        excluded_indices = {int(v) for v in recommended.get("removed_indices", [])}
        robust_samples = [s for s in samples if s.index not in excluded_indices]
        if len(robust_samples) >= CONFLICT_DIAG_CFG.min_accept_samples and len(robust_samples) < len(samples):
            print(
                f"[ROBUST] 找到达标一致子集：保留 {len(robust_samples)}/{len(samples)} 个样本，"
                f"排除 {sorted(excluded_indices)}，使用该子集输出最终矩阵。"
            )
            robust_estimate = solve_handeye_estimate(robust_samples, quiet=True)
            used_samples = robust_samples
            best_method = robust_estimate["best_method"]
            all_method_results = robust_estimate["all_method_results"]
            T_init = robust_estimate["T_init"]
            T_final = robust_estimate["T_final"]
            init_stats = robust_estimate["initial_quality"]
            final_stats = robust_estimate["quality"]
            refine_info = robust_estimate["nonlinear_refine"]
            pose6 = transform_to_pose6_rzryrx(T_final)
            conflict_report["final_result_uses_consensus_subset"] = True

    output = {
        "record_type": "handeye_rgb_diagnostic_result" if is_rgb else "handeye_pointcloud_legacy_diagnostic_result",
        "record_policy": "diagnostic_current_replace_on_refresh",
        "validated": False,
        "do_not_use_for_motion": True,
        "production_eligible": False,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": method_name,
        "calibration_frame": calibration_frame,
        "rgb_coordinate_convention": (
            "opencv_optical_x_right_y_down_z_forward" if is_rgb else None
        ),
        "frame_definition": {
            "B": "robot_base_or_world", "T": reference_frame, "S": sensor_frame_name,
            "Q": "charuco_board", "T_pose_source_sensor": "^T T_S",
            "runtime_formula": "P_base = T_base_pose_source @ T_pose_source_sensor @ P_sensor",
        },
        "pose_source": pose_source,
        "raw_sample_count": raw_sample_count,
        "sample_count": len(used_samples),
        "used_sample_indices": [int(s.index) for s in used_samples],
        "excluded_sample_indices": [int(s.index) for s in samples if s.index not in {u.index for u in used_samples}],
        "best_opencv_method": best_method,
        "T_pose_source_sensor": matrix_to_list(T_final),
        "T_pose_source_sensor_pose6_rzryrx_mm_deg": [float(v) for v in pose6],
        "T_pose_source_sensor_pose_note": (
            "该字段只是把矩阵近似展开为 [x, y, z, rz, ry, rx] 便于人工查看；"
            "该文件仅用于诊断，不能直接用于运动。"
        ),
        "T_pose_source_sensor_initial_opencv": matrix_to_list(T_init),
        "initial_quality": {k: v for k, v in init_stats.items() if k != "T_base_board_mean"},
        "quality": {k: v for k, v in final_stats.items() if k != "T_base_board_mean"},
        "all_sample_quality_before_conflict_filter": all_sample_quality,
        "sample_conflict_diagnosis": conflict_report,
        "diagnostic_consistency_gate_1mm": {
            "production_acceptance": False,
            "translation_mean_mm_max": AUTO_CAPTURE_CFG.diagnostic_result_mean_mm,
            "translation_rmse_mm_max": AUTO_CAPTURE_CFG.diagnostic_result_rmse_mm,
            "translation_max_mm_max": AUTO_CAPTURE_CFG.diagnostic_result_max_mm,
            "meets_diagnostic_gate": (
                final_stats["translation_mean_mm"] <= AUTO_CAPTURE_CFG.diagnostic_result_mean_mm
                and final_stats["translation_rmse_mm"] <= AUTO_CAPTURE_CFG.diagnostic_result_rmse_mm
                and final_stats["translation_max_mm"] <= AUTO_CAPTURE_CFG.diagnostic_result_max_mm
            ),
            "note": "仅用于发现冲突样本，不能解锁运动。",
        },
        "production_e7_requirement": {
            "independent_cross_validation_required": True,
            "validation_pose_scatter_rms_mm_max": E7_HAND_EYE_CFG.maximum_validation_center_scatter_rms_mm,
            "minimum_total_poses": E7_HAND_EYE_CFG.minimum_total_poses,
            "minimum_validation_fraction": E7_HAND_EYE_CFG.minimum_validation_fraction,
            "minimum_validation_poses": E7_HAND_EYE_CFG.minimum_validation_poses,
            "production_eligible_from_this_solver_output_alone": False,
        },
        "T_base_board_mean": matrix_to_list(final_stats["T_base_board_mean"]),
        "nonlinear_refine": refine_info,
        "opencv_method_results": all_method_results,
        "board": asdict(BOARD_CFG),
        "camera": asdict(CAMERA_CFG),
        "capture": {
            "manual_burst_frames": AUTO_CAPTURE_CFG.manual_burst_frames,
            "manual_burst_max_attempts": AUTO_CAPTURE_CFG.manual_burst_max_attempts,
            "burst_interval_s": AUTO_CAPTURE_CFG.burst_interval_s,
            "burst_pose_stability_xyz_mm": AUTO_CAPTURE_CFG.burst_pose_stability_xyz_mm,
            "burst_pose_stability_abc_deg": AUTO_CAPTURE_CFG.burst_pose_stability_abc_deg,
            "reject_unstable_burst_pose": AUTO_CAPTURE_CFG.reject_unstable_burst_pose,
            "burst_pose_cluster_xyz_mm": AUTO_CAPTURE_CFG.burst_pose_cluster_xyz_mm,
            "burst_pose_cluster_abc_deg": AUTO_CAPTURE_CFG.burst_pose_cluster_abc_deg,
            "burst_min_pose_cluster_frames": AUTO_CAPTURE_CFG.burst_min_pose_cluster_frames,
            "require_quality_ok_for_burst": AUTO_CAPTURE_CFG.require_quality_ok_for_burst,
            "min_rgb_pnp_inliers": AUTO_CAPTURE_CFG.min_rgb_pnp_inliers,
            "max_rgb_reprojection_rmse_px": AUTO_CAPTURE_CFG.max_rgb_reprojection_rmse_px,
            "max_rgb_reprojection_error_px": AUTO_CAPTURE_CFG.max_rgb_reprojection_error_px,
        },
        "robot": {
            "robot_brand": "AUBO", "ip": ROBOT_CFG.ip, "rpc_port": ROBOT_CFG.rpc_port,
            "pose_source": pose_source, "sdk_pose_units": {"xyz": "m", "rpy": "rad"},
            "solver_pose_units": {"xyz": "mm", "rotation": "matrix"},
            "read_api": "RobotState.getTcpPose()" if pose_source == "tcp" else "RobotState.getToolPose()",
            "request_timeout_ms": ROBOT_CFG.request_timeout_ms,
            "require_power_on": ROBOT_CFG.require_power_on, "require_steady": ROBOT_CFG.require_steady,
            "reject_collision": ROBOT_CFG.reject_collision,
        },
    }

    if pose_source == "tcp" and is_rgb:
        output["T_tcp_rgb_camera"] = output["T_pose_source_sensor"]
    elif pose_source == "tcp":
        output["T_tcp_pointcloud"] = output["T_pose_source_sensor"]
    elif pose_source == "tool" and is_rgb:
        output["T_robot_tool_rgb_camera"] = output["T_pose_source_sensor"]
    elif pose_source == "tool":
        output["T_robot_tool_pointcloud"] = output["T_pose_source_sensor"]

    if SOLVE_CFG.also_write_compatible_key_t_tooltcp_cam and pose_source == "tcp" and is_rgb:
        output["t_tooltcp_cam"] = output["T_tcp_rgb_camera"]
        output["compatible_note"] = (
            "t_tooltcp_cam is kept only for old runtime compatibility; "
            "it means T_tcp_rgb_camera (^T T_Crgb)."
        )

    out_path = atomic_write_json(Path(SOLVE_CFG.output_json), output)

    print("=" * 70)
    print("[RESULT] 已原子更新唯一诊断结果:", out_path)
    print(
        f"[RESULT] T_pose_source_sensor, pose_source={pose_source}, "
        f"sensor={sensor_frame_name}, reference={reference_frame}"
    )
    print(np.asarray(T_final))
    print("[RESULT] pose6 approx mm/deg:", fmt_vec(pose6, 6))
    print(
        f"[QUALITY] trans_mean={final_stats['translation_mean_mm']:.4f} mm, "
        f"trans_rmse={final_stats['translation_rmse_mm']:.4f} mm, trans_max={final_stats['translation_max_mm']:.4f} mm"
    )
    print(
        f"[QUALITY] rot_mean={final_stats['rotation_mean_deg']:.6f} deg, "
        f"rot_rmse={final_stats['rotation_rmse_deg']:.6f} deg, rot_max={final_stats['rotation_max_deg']:.6f} deg"
    )
    diag_status = conflict_report.get("status")
    bad_indices = conflict_report.get("initial_bad_indices") or []
    suspect_indices = conflict_report.get("initial_suspect_indices") or []
    recommended_entry = conflict_report.get("recommended_entry") or {}
    if diag_status:
        print(f"[CONFLICT] status={diag_status}, bad={bad_indices[:12]}, suspect={suspect_indices[:12]}")
        if recommended_entry:
            print(
                f"[CONFLICT] best_preview: n={recommended_entry.get('sample_count')}, "
                f"mean={recommended_entry.get('translation_mean_mm'):.4f} mm, "
                f"rmse={recommended_entry.get('translation_rmse_mm'):.4f} mm, "
                f"max={recommended_entry.get('translation_max_mm'):.4f} mm, "
                f"removed={recommended_entry.get('removed_indices')}"
            )
        if diag_status == "only_small_subset_meets_target":
            print("[CONFLICT] 少量样本能凑到 1 mm，但样本数不足，不建议作为通用手眼结果；请补拍同类姿态的新样本。")
        elif diag_status == "no_consensus_meets_target":
            print("[CONFLICT] 当前样本集合没有足够稳定的一致子集；请先处理 bad/suspect 点位并补拍。")
    diagnostic_ok = (
        final_stats["translation_mean_mm"] <= AUTO_CAPTURE_CFG.diagnostic_result_mean_mm
        and final_stats["translation_rmse_mm"] <= AUTO_CAPTURE_CFG.diagnostic_result_rmse_mm
        and final_stats["translation_max_mm"] <= AUTO_CAPTURE_CFG.diagnostic_result_max_mm
    )
    if diagnostic_ok:
        print(
            f"[DIAGNOSTIC] 通过毫米级一致性门："
            f"mean<={AUTO_CAPTURE_CFG.diagnostic_result_mean_mm:.2f}, "
            f"rmse<={AUTO_CAPTURE_CFG.diagnostic_result_rmse_mm:.2f}, "
            f"max<={AUTO_CAPTURE_CFG.diagnostic_result_max_mm:.2f} mm。"
        )
        print("[LOCK] 该结果仍不能解锁运动；必须另做E7独立交叉验证，验证姿态RMS<=0.10 mm。")
    else:
        print(
            f"[DIAGNOSTIC] 未通过毫米级一致性门；建议按 a 自动剔除坏样本并补采。门槛："
            f"mean<={AUTO_CAPTURE_CFG.diagnostic_result_mean_mm:.2f}, "
            f"rmse<={AUTO_CAPTURE_CFG.diagnostic_result_rmse_mm:.2f}, "
            f"max<={AUTO_CAPTURE_CFG.diagnostic_result_max_mm:.2f} mm。"
        )
    print("=" * 70)
    return output
