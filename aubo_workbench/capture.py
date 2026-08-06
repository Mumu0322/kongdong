#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""采集一个手眼样本：单帧采集 + “连续5帧选1帧”批量采集。"""

from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any, Callable

import cv2
import numpy as np

from .charuco_detect import BoardPoseResult, estimate_rgb_board_pose
from .config import AUTO_CAPTURE_CFG, E7_HAND_EYE_CFG, SOLVE_CFG
from .drawing import draw_unicode_text
from .geometry import circular_angle_abs_diff_deg, angle_span_deg as circular_angle_span_deg
from .quality import evaluate_image_quality
from .robot import get_capture_pose_transform
from .samples import CalibSample, save_sample
from .visualization import compose_display
from .io_utils import timestamp_str


def board_view_metadata(pose_result: BoardPoseResult, image_shape: tuple[int, ...]) -> dict[str, Any]:
    """从实际角点自动计算标定板中心和视野区域，供E7覆盖检查。"""
    height, width = int(image_shape[0]), int(image_shape[1])
    raw_points = pose_result.image_points
    points = (
        np.empty((0, 2), dtype=np.float64)
        if raw_points is None
        else np.asarray(raw_points, dtype=np.float64).reshape(-1, 2)
    )
    if points.size == 0 or width <= 0 or height <= 0:
        return {"board_center_uv": [], "image_size_wh": [width, height], "view_region": "unknown"}
    center = np.mean(points, axis=0)
    dx = (float(center[0]) - 0.5 * width) / max(0.5 * width, 1.0)
    dy = (float(center[1]) - 0.5 * height) / max(0.5 * height, 1.0)
    if (
        abs(dx) <= float(E7_HAND_EYE_CFG.center_region_half_width_ratio)
        and abs(dy) <= float(E7_HAND_EYE_CFG.center_region_half_height_ratio)
    ):
        region = "center"
    elif abs(dx) >= abs(dy):
        region = "right" if dx > 0.0 else "left"
    else:
        region = "bottom" if dy > 0.0 else "top"
    return {
        "board_center_uv": [float(center[0]), float(center[1])],
        "image_size_wh": [width, height],
        "view_region": region,
    }


def pose_bracket_report(
    before_snapshot: dict[str, Any] | None,
    after_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    """检查相机取帧前后TCP读数，防止把运动中的帧作为静态标定样本。"""
    before = snapshot_pose_array(before_snapshot)
    after = snapshot_pose_array(after_snapshot)
    if before is None or after is None:
        return {"ok": False, "reason": "pose_values_missing"}
    xyz_delta = np.abs(after[:3] - before[:3])
    abc_delta = np.asarray(
        [circular_angle_abs_diff_deg(before[i], after[i]) for i in range(3, 6)],
        dtype=np.float64,
    )
    xyz_limit = float(E7_HAND_EYE_CFG.maximum_pose_bracket_xyz_mm)
    abc_limit = float(E7_HAND_EYE_CFG.maximum_pose_bracket_abc_deg)
    ok = float(np.max(xyz_delta)) <= xyz_limit and float(np.max(abc_delta)) <= abc_limit
    return {
        "ok": bool(ok),
        "xyz_delta_mm": xyz_delta.tolist(),
        "abc_delta_deg": abc_delta.tolist(),
        "xyz_limit_mm": xyz_limit,
        "abc_limit_deg": abc_limit,
        "before_timestamp": (before_snapshot or {}).get("timestamp"),
        "after_timestamp": (after_snapshot or {}).get("timestamp"),
    }


def snapshot_pose_array(robot_snapshot: dict[str, Any] | None) -> np.ndarray | None:
    values = (robot_snapshot or {}).get("pose_values")
    if not isinstance(values, (list, tuple)) or len(values) < 6:
        return None
    try:
        return np.asarray([float(v) for v in values[:6]], dtype=np.float64)
    except Exception:
        return None


def capture_quality_score(item: dict[str, Any]) -> float:
    pose_result: BoardPoseResult = item["pose_result"]
    if pose_result.calibration_frame == "rgb_camera":
        reprojection = (
            pose_result.rgb_reprojection_rmse_px
            if np.isfinite(pose_result.rgb_reprojection_rmse_px) else 999.0
        )
        maximum = (
            pose_result.rgb_reprojection_max_px
            if np.isfinite(pose_result.rgb_reprojection_max_px) else 999.0
        )
        return float(reprojection + 0.1 * maximum)
    corner = pose_result.corner_rmse_mm if np.isfinite(pose_result.corner_rmse_mm) else 999.0
    plane = pose_result.plane_rmse_mm if np.isfinite(pose_result.plane_rmse_mm) else 999.0
    return float(corner + plane)


def pose_is_near(seed: np.ndarray, pose: np.ndarray, xyz_tol_mm: float, abc_tol_deg: float) -> bool:
    xyz_ok = bool(np.max(np.abs(seed[:3] - pose[:3])) <= xyz_tol_mm)
    abc_diffs = [circular_angle_abs_diff_deg(seed[i], pose[i]) for i in range(3, 6)]
    abc_ok = bool(max(abc_diffs, default=0.0) <= abc_tol_deg)
    return xyz_ok and abc_ok


def select_burst_representative_frame(captures: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """在 5 帧里找到彼此接近（同一次静止姿态）的最大簇，再选簇里质量最好的 1 帧。"""
    pose_entries: list[tuple[int, np.ndarray, dict[str, Any]]] = []
    for frame_no, item in enumerate(captures, start=1):
        pose = snapshot_pose_array(item.get("robot_snapshot"))
        if pose is not None:
            pose_entries.append((frame_no, pose, item))

    if not pose_entries:
        return None, {"selected": False, "reason": "no_robot_pose_values", "pose_cluster_frame_numbers": []}

    xyz_tol = float(AUTO_CAPTURE_CFG.burst_pose_cluster_xyz_mm)
    abc_tol = float(AUTO_CAPTURE_CFG.burst_pose_cluster_abc_deg)
    min_cluster = int(AUTO_CAPTURE_CFG.burst_min_pose_cluster_frames)

    best_cluster: list[tuple[int, np.ndarray, dict[str, Any]]] = []
    for _, seed_pose, _ in pose_entries:
        cluster = [entry for entry in pose_entries if pose_is_near(seed_pose, entry[1], xyz_tol, abc_tol)]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
        elif len(cluster) == len(best_cluster) and cluster:
            cluster_best = min(capture_quality_score(entry[2]) for entry in cluster)
            current_best = min(capture_quality_score(entry[2]) for entry in best_cluster)
            if cluster_best < current_best:
                best_cluster = cluster

    if len(best_cluster) < min_cluster:
        return None, {
            "selected": False, "reason": "no_repeated_stable_pose_cluster",
            "pose_cluster_frame_numbers": [entry[0] for entry in best_cluster],
            "pose_cluster_size": len(best_cluster), "required_pose_cluster_size": min_cluster,
            "pose_cluster_xyz_tolerance_mm": xyz_tol, "pose_cluster_abc_tolerance_deg": abc_tol,
        }

    selected_frame_no, selected_pose, selected_item = min(best_cluster, key=lambda entry: capture_quality_score(entry[2]))
    return selected_item, {
        "selected": True, "source": "manual_5_frame_best_single_from_pose_cluster",
        "selected_frame_number": selected_frame_no, "selected_pose_values": selected_pose.tolist(),
        "pose_cluster_frame_numbers": [entry[0] for entry in best_cluster], "pose_cluster_size": len(best_cluster),
        "pose_cluster_xyz_tolerance_mm": xyz_tol, "pose_cluster_abc_tolerance_deg": abc_tol,
        "selected_quality_score": capture_quality_score(selected_item),
    }


def burst_pose_range_pose6(captures: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray] | None:
    poses = [p for p in (snapshot_pose_array(item.get("robot_snapshot")) for item in captures) if p is not None]
    if len(poses) < 2:
        return None
    arr = np.vstack(poses)
    xyz_range = np.ptp(arr[:, :3], axis=0)
    abc_range = np.asarray([circular_angle_span_deg(arr[:, i]) for i in range(3, 6)], dtype=np.float64)
    return xyz_range, abc_range


def burst_pose_is_stable(captures: list[dict[str, Any]]) -> tuple[bool, str, np.ndarray | None, np.ndarray | None]:
    ranges = burst_pose_range_pose6(captures)
    if ranges is None:
        return False, "机器人 TCP 位姿数量不足，无法确认五帧稳定性", None, None
    xyz_range, abc_range = ranges
    xyz_limit = float(AUTO_CAPTURE_CFG.burst_pose_stability_xyz_mm)
    abc_limit = float(AUTO_CAPTURE_CFG.burst_pose_stability_abc_deg)
    ok = float(np.max(xyz_range)) <= xyz_limit and float(np.max(abc_range)) <= abc_limit
    msg = (
        f"rangeXYZ=({xyz_range[0]:.3f},{xyz_range[1]:.3f},{xyz_range[2]:.3f}) mm, "
        f"rangeABC=({abc_range[0]:.3f},{abc_range[1]:.3f},{abc_range[2]:.3f}) deg, "
        f"limit=({xyz_limit:.3f} mm, {abc_limit:.3f} deg)"
    )
    return ok, msg, xyz_range, abc_range


def save_burst_single_frame_sample(
    captures: list[dict[str, Any]], next_index: int, samples: list[CalibSample],
) -> tuple[CalibSample | None, int]:
    if not captures:
        return None, next_index

    ok, stability_msg, xyz_range, abc_range = burst_pose_is_stable(captures)
    reject_unstable = bool(AUTO_CAPTURE_CFG.reject_unstable_burst_pose)
    if not ok and reject_unstable:
        print(f"[BURST-REJECT] 五帧期间 TCP 未稳定，本次不保存。{stability_msg}")
        print("[BURST-REJECT] 请确认机器人完全停止后等待 1~2 秒，再按 c 采集。")
        return None, next_index
    if not ok:
        print(f"[BURST-WARN] 五帧期间 TCP 波动超过建议阈值，但当前配置允许保存。{stability_msg}")
        print("[BURST-WARN] 该样本会写入 warning 标记，后续求解时如残差偏大再剔除。")

    best_item, selection_info = select_burst_representative_frame(captures)
    if best_item is None:
        reason = selection_info.get("reason", "unknown")
        print(f"[BURST-REJECT] 五帧内没有找到重复稳定的 TCP 姿态簇，本次不保存。reason={reason}")
        print("[BURST-REJECT] 请确认机器人完全停止后等待 1~2 秒，再按 c 采集；或检查控制器当前 TCP 读数是否在多个点之间跳变。")
        return None, next_index

    selected_pose_result: BoardPoseResult = best_item["pose_result"]
    raw_pose_values = [p.tolist() for p in (snapshot_pose_array(item.get("robot_snapshot")) for item in captures) if p is not None]

    robot_snapshot = dict(best_item.get("robot_snapshot") or {})
    robot_snapshot["burst_aggregate"] = {
        "frame_count": len(captures), "source": "manual_5_frame_best_single_from_pose_cluster",
        "raw_pose_values": raw_pose_values,
        "pose_range_xyz_mm": xyz_range.tolist() if xyz_range is not None else [],
        "pose_range_abc_deg": abc_range.tolist() if abc_range is not None else [],
        "pose_stable": bool(ok), "stability_message": stability_msg, "selection": selection_info,
        "note": "Five valid frames were used only for screening; the saved calibration sample uses one real frame, not averaged TCP.",
    }

    ts = timestamp_str()
    view = board_view_metadata(selected_pose_result, best_item["color_bgr"].shape)
    sample = CalibSample(
        index=next_index, timestamp=ts,
        T_base_tool=np.asarray(best_item["T_base_tool"], dtype=np.float64),
        T_pointcloud_board=(
            None if selected_pose_result.T_pointcloud_board is None
            else np.asarray(selected_pose_result.T_pointcloud_board, dtype=np.float64)
        ),
        robot_snapshot=robot_snapshot,
        board_status="burst_single_frame_ok" if ok else "burst_single_frame_unstable_tcp_warning",
        charuco_count=int(selected_pose_result.charuco_count), valid_3d_count=int(selected_pose_result.valid_3d_count),
        corner_rmse_mm=float(selected_pose_result.corner_rmse_mm),
        corner_max_error_mm=float(selected_pose_result.corner_max_error_mm),
        plane_rmse_mm=float(selected_pose_result.plane_rmse_mm),
        plane_inlier_count=int(selected_pose_result.plane_inlier_count),
        board_mask_point_count=int(selected_pose_result.board_mask_point_count),
        camera_metadata=dict(best_item.get("camera_metadata") or {}),
        capture_quality=asdict(best_item["quality"]),
        calibration_frame=selected_pose_result.calibration_frame,
        T_rgb_board=(
            None if selected_pose_result.T_rgb_board is None
            else np.asarray(selected_pose_result.T_rgb_board, dtype=np.float64)
        ),
        rgb_pnp_inlier_count=int(selected_pose_result.rgb_pnp_inlier_count),
        rgb_reprojection_rmse_px=float(selected_pose_result.rgb_reprojection_rmse_px),
        rgb_reprojection_max_px=float(selected_pose_result.rgb_reprojection_max_px),
        board_center_uv=view["board_center_uv"], image_size_wh=view["image_size_wh"],
        view_region=view["view_region"],
    )
    sample = save_sample(sample, best_item["color_bgr"], best_item["overlay_bgr"], best_item["depth_mm"])
    samples.append(sample)
    if sample.calibration_frame == "rgb_camera":
        quality_text = (
            f"rgb_inliers={sample.rgb_pnp_inlier_count} | "
            f"reproj_rmse={sample.rgb_reprojection_rmse_px:.4f}px | "
            f"reproj_max={sample.rgb_reprojection_max_px:.4f}px"
        )
    else:
        quality_text = (
            f"corner_rmse={sample.corner_rmse_mm:.4f}mm | "
            f"plane_rmse={sample.plane_rmse_mm:.4f}mm"
        )
    print(
        f"[CAPTURE-MANUAL5-SINGLE] 样本 {sample.index} OK | 五帧筛选，保存第 {selection_info.get('selected_frame_number')} 帧 | "
        f"cluster={selection_info.get('pose_cluster_size')}/{len(captures)} | "
        f"{quality_text} | {stability_msg} | samples={len(samples)}"
    )
    next_index += 1
    if len(samples) >= SOLVE_CFG.min_samples_for_solve:
        print(
            f"[INFO] 样本数已满足诊断求解条件，可按 h；"
            f"E7正式验证仍需至少{E7_HAND_EYE_CFG.minimum_total_poses}组及完整证据。"
        )
    return sample, next_index


def _run_burst_loop(
    pipeline, align_filter, point_cloud_filter, board, dictionary,
    samples: list[CalibSample], on_frame: Callable[[np.ndarray, int, int, int], None] | None,
    stop_check: Callable[[], bool] | None,
    log_prefix: str,
) -> list[dict[str, Any]]:
    """5 帧采集的共享主循环；OpenCV 窗口版和 GUI 版都复用这段逻辑，只是展示方式不同。"""
    from .camera import get_device_identity, get_rgb_frame_bundle

    target_count = int(AUTO_CAPTURE_CFG.manual_burst_frames)
    max_attempts = max(target_count, int(AUTO_CAPTURE_CFG.manual_burst_max_attempts))
    captures: list[dict[str, Any]] = []
    device_identity = get_device_identity(pipeline)
    print(f"[{log_prefix}] 开始采集：目标 {target_count} 帧合格观测，最多尝试 {max_attempts} 帧。")

    for attempt in range(1, max_attempts + 1):
        if stop_check is not None and stop_check():
            print(f"[{log_prefix}] 已收到停止信号，取消本次采集。")
            return []
        if len(captures) >= target_count:
            break

        T_before, snapshot_before, status_before = get_capture_pose_transform()
        if T_before is None:
            print(f"[{log_prefix}-SKIP] 第 {attempt}/{max_attempts} 次取帧前机器人位姿不可用，status={status_before}")
            time.sleep(float(AUTO_CAPTURE_CFG.burst_interval_s))
            continue

        bundle = get_rgb_frame_bundle(pipeline)
        if bundle is None:
            print(f"[{log_prefix}] 第 {attempt}/{max_attempts} 次取帧失败，跳过。")
            continue

        T_after, snapshot_after, status_after = get_capture_pose_transform()
        if T_after is None:
            print(f"[{log_prefix}-SKIP] 第 {attempt}/{max_attempts} 次取帧后机器人位姿不可用，status={status_after}")
            time.sleep(float(AUTO_CAPTURE_CFG.burst_interval_s))
            continue
        bracket = pose_bracket_report(snapshot_before, snapshot_after)
        if not bracket.get("ok", False):
            print(f"[{log_prefix}-SKIP] 第 {attempt}/{max_attempts} 帧取帧前后TCP不稳定：{bracket}")
            time.sleep(float(AUTO_CAPTURE_CFG.burst_interval_s))
            continue

        color_bgr = bundle.color_bgr
        depth_for_display = np.zeros(color_bgr.shape[:2], dtype=np.float32)
        pose_result = estimate_rgb_board_pose(color_bgr, bundle.intrinsics, board, dictionary)
        quality = evaluate_image_quality(color_bgr, pose_result)

        if on_frame is not None:
            display = compose_display(
                pose_result.rgb_overlay.copy(), depth_for_display, len(samples), quality=quality,
                good_frame_count=len(captures), auto_enabled=False, cooldown_left_s=0.0,
            )
            draw_unicode_text(
                display, f"批量采集：第 {attempt}/{max_attempts} 次尝试，已取得 {len(captures)}/{target_count}",
                (24, 36), (0, 220, 255), 22, 1,
            )
            on_frame(display, attempt, max_attempts, len(captures))

        if AUTO_CAPTURE_CFG.require_quality_ok_for_burst and not quality.ok:
            reason_text = "；".join(quality.reasons[:2]) if quality.reasons else "质量评分未达标"
            advice_text = "；".join(quality.advice[:2]) if quality.advice else "请调整标定板位置和光照"
            print(f"[{log_prefix}-SKIP] 第 {attempt}/{max_attempts} 帧未保存 | score={quality.score:.1f} | 原因：{reason_text} | 建议：{advice_text}")
            time.sleep(float(AUTO_CAPTURE_CFG.burst_interval_s))
            continue

        if not pose_result.ok or pose_result.T_rgb_board is None:
            print(f"[{log_prefix}-SKIP] 第 {attempt}/{max_attempts} 帧未保存 | ChArUco RGB-PnP板位姿不可用，status={pose_result.status}")
            time.sleep(float(AUTO_CAPTURE_CFG.burst_interval_s))
            continue

        robot_snapshot = dict(snapshot_after or {})
        robot_snapshot["pose_status"] = status_after
        robot_snapshot["camera_frame_pose_bracket"] = {
            **bracket,
            "before_snapshot": snapshot_before,
            "after_snapshot": snapshot_after,
        }
        camera_metadata = bundle.metadata_dict()
        camera_metadata["device"] = device_identity
        overlay_for_save = pose_result.rgb_overlay.copy()
        captures.append({
            "pose_result": pose_result, "color_bgr": color_bgr, "depth_mm": None,
            "overlay_bgr": overlay_for_save, "T_base_tool": T_after,
            "robot_snapshot": robot_snapshot, "pose_status": status_after,
            "camera_metadata": camera_metadata,
            "quality": quality,
        })
        time.sleep(float(AUTO_CAPTURE_CFG.burst_interval_s))

    return captures


def capture_burst_samples(
    pipeline, align_filter, point_cloud_filter, board, dictionary,
    next_index: int, samples: list[CalibSample], window_name: str,
) -> int:
    """OpenCV 窗口版：按一次 c 后连续抓取 5 帧，并从稳定姿态簇中选 1 帧保存。"""
    target_count = int(AUTO_CAPTURE_CFG.manual_burst_frames)

    def on_frame(display: np.ndarray, attempt: int, max_attempts: int, saved: int) -> None:
        cv2.imshow(window_name, display)
        cv2.waitKey(1)

    captures = _run_burst_loop(
        pipeline, align_filter, point_cloud_filter, board, dictionary, samples, on_frame, None, "BURST",
    )
    if len(captures) == target_count:
        print(f"[BURST] 手动5帧采集完成：成功取得 {len(captures)}/{target_count} 帧，准备选择 1 帧保存为样本。")
        _, next_index = save_burst_single_frame_sample(captures, next_index, samples)
    else:
        print(f"[BURST] 手动5帧采集未满：成功取得 {len(captures)}/{target_count} 帧。请按画面建议调整后再按 c 重试。")
    return next_index


def capture_burst_samples_gui(
    pipeline, align_filter, point_cloud_filter, board, dictionary,
    next_index: int, samples: list[CalibSample],
    publish_display: Callable[[np.ndarray], None] | None = None,
    stop_event: Any | None = None,
) -> int:
    """GUI 按钮触发的 5 帧采集；算法与 OpenCV 窗口版一致，只是展示方式换成回调。"""
    target_count = int(AUTO_CAPTURE_CFG.manual_burst_frames)

    def on_frame(display: np.ndarray, attempt: int, max_attempts: int, saved: int) -> None:
        if publish_display is not None:
            publish_display(display)

    def stop_check() -> bool:
        return bool(stop_event is not None and stop_event.is_set())

    captures = _run_burst_loop(
        pipeline, align_filter, point_cloud_filter, board, dictionary, samples, on_frame, stop_check, "GUI-BURST",
    )
    if len(captures) == target_count:
        print(f"[GUI-BURST] 已取得 {len(captures)}/{target_count} 帧，开始选择 1 帧保存为样本。")
        _, next_index = save_burst_single_frame_sample(captures, next_index, samples)
    elif captures:
        print(f"[GUI-BURST] 采集未满：成功取得 {len(captures)}/{target_count} 帧。请按画面建议调整后重试。")
    return next_index
