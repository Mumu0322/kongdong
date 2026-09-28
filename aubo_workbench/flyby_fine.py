"""Capture-only RGB fine localization during a blended moving scan."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class PoseSample:
    timestamp_ns: int
    tcp: np.ndarray
    read_duration_ns: int


def rows_from_map(seeds: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = {}
    for seed in seeds:
        source = seed.get("coarse_map_source") or {}
        row = source.get("layout_row")
        column = source.get("layout_column")
        if not isinstance(row, int) or not isinstance(column, int):
            raise ValueError("运动取帧要求地图中每个孔都有排号和列号")
        rows.setdefault(row, []).append(seed)
    if not rows:
        raise ValueError("地图中没有可扫描孔")
    result = []
    for index, row in enumerate(sorted(rows)):
        ordered = sorted(
            rows[row], key=lambda item: int(item["coarse_map_source"]["layout_column"]),
            reverse=bool(index % 2),
        )
        if len(ordered) < 2:
            raise ValueError(f"第 {row} 排只有一个孔，无法建立匀速扫描段")
        result.append(ordered)
    return result


def scan_endpoints(
    row: list[dict[str, Any]], reference_tcp: np.ndarray,
    tcp_to_camera: np.ndarray, height_mm: float, lead_mm: float = 15.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Keep a constant camera orientation and clear the highest mapped surface."""
    points = np.asarray([item["initial_center_base_mm"] for item in row], dtype=float)
    if not np.isfinite(points).all():
        raise ValueError("地图孔位包含非有限坐标")
    center_xy = np.mean(points[:, :2], axis=0)
    _, _, axes = np.linalg.svd(points[:, :2] - center_xy, full_matrices=False)
    direction = axes[0]
    if float(direction @ (points[-1, :2] - points[0, :2])) < 0:
        direction = -direction
    along = (points[:, :2] - center_xy) @ direction
    if np.any(np.diff(along) <= 0):
        raise ValueError("地图列号与孔位沿扫描方向的顺序不一致")
    length = float(along[-1] - along[0])
    if length < 10.0:
        raise ValueError("扫描排首尾孔距离不足 10 mm")
    fitted_offsets = np.abs(
        direction[0] * (points[:, 1] - center_xy[1])
        - direction[1] * (points[:, 0] - center_xy[0])
    )
    if float(np.max(fitted_offsets)) > 30.0:
        raise ValueError("同一排孔偏离拟合扫描直线超过 30 mm")
    camera_axis = (reference_tcp @ tcp_to_camera)[:3, 2]
    if camera_axis[2] > -0.8:
        raise ValueError("相机光轴未充分朝向工件，拒绝连续扫描")
    surface_z = float(np.max(points[:, 2]))
    camera_z = surface_z - float(height_mm) * float(camera_axis[2])
    starts = (
        center_xy + (along[0] - lead_mm) * direction - height_mm * camera_axis[:2],
        center_xy + (along[-1] + lead_mm) * direction - height_mm * camera_axis[:2],
    )
    targets = []
    for xy in starts:
        camera = np.eye(4)
        camera[:3, :3] = (reference_tcp @ tcp_to_camera)[:3, :3]
        camera[:3, 3] = [float(xy[0]), float(xy[1]), camera_z]
        targets.append(camera @ np.linalg.inv(tcp_to_camera))
    return targets[0], targets[1], {
        "path_length_mm": length + 2.0 * lead_mm,
        "surface_z_spread_mm": float(np.ptp(points[:, 2])),
        "minimum_camera_height_mm": float(height_mm),
        "maximum_row_deviation_mm": float(np.max(fitted_offsets)),
    }


def scan_pose_waypoints(
    hole_poses: list[np.ndarray], *, lead_mm: float = 15.0,
    max_rotation_step_deg: float = 20.0,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, float]]:
    """Add approach and exit poses around each hole's own mapped view pose."""
    poses = [np.asarray(pose, dtype=float).reshape(4, 4) for pose in hole_poses]
    if len(poses) < 2 or not all(np.isfinite(pose).all() for pose in poses):
        raise ValueError("连续变姿态扫描至少需要两个有效孔位姿")
    steps = np.diff(np.asarray([pose[:3, 3] for pose in poses]), axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    if np.any(lengths < 10.0):
        raise ValueError("相邻孔拍摄位姿距离不足 10 mm")
    rotations = [
        float(np.degrees((
            Rotation.from_matrix(poses[index][:3, :3]).inv()
            * Rotation.from_matrix(poses[index + 1][:3, :3])
        ).magnitude()))
        for index in range(len(poses) - 1)
    ]
    if max(rotations) > max_rotation_step_deg:
        raise ValueError(
            f"相邻孔姿态变化 {max(rotations):.1f}° 超过 {max_rotation_step_deg:g}°"
        )
    start = poses[0].copy()
    start[:3, 3] -= lead_mm * steps[0] / lengths[0]
    exit_pose = poses[-1].copy()
    exit_pose[:3, 3] += lead_mm * steps[-1] / lengths[-1]
    waypoints = [*poses, exit_pose]
    path_mm = float(np.sum(lengths) + 2.0 * lead_mm)
    return start, waypoints, {
        "path_length_mm": path_mm,
        "maximum_pose_rotation_step_deg": max(rotations),
        "pose_policy": "per_hole_coarse_map_normal_fixed_rz",
    }


def continuous_scan_waypoints(
    rows: list[list[np.ndarray]], *, transfer_clearance_mm: float = 50.0,
    blend_radius_mm: float = 5.0,
    tcp_to_camera: np.ndarray | None = None,
    mapped_points_mm: np.ndarray | None = None,
    min_mapped_clearance_mm: float = 200.0,
    initial_tcp: np.ndarray | None = None,
    approach_position_step_mm: float = 60.0,
    approach_rotation_step_deg: float = 8.0,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    """Join mapped rows and, optionally, the current TCP in one blended path.

    When ``initial_tcp`` is provided, the approach to the first hole is part of
    the same Cartesian queue.  The orientation is changed gradually while the
    TCP moves toward the first hole and descends, so an unreachable high-height
    attitude target is never sent as a separate ``moveJoint`` command.
    """
    if not rows:
        raise ValueError("连续扫描没有可用排")
    planned = [scan_pose_waypoints(row) for row in rows]
    safe_z = max(float(pose[2, 3]) for row in rows for pose in row) + transfer_clearance_mm
    if initial_tcp is not None:
        initial_candidate = np.asarray(initial_tcp, dtype=float).reshape(4, 4)
        if not np.isfinite(initial_candidate).all():
            raise ValueError("连续路径起始TCP包含非有限值")
    scan_start = planned[0][0]
    start = scan_start
    path: list[np.ndarray] = []
    transit_flags: list[bool] = []
    omitted_exit_waypoints = 0
    for index, (_, row_path, _) in enumerate(planned):
        # The per-row exit lead is useful when a row is captured in isolation,
        # but it is redundant in the global path: the next diagonal transfer
        # already leaves the last hole.  Removing it avoids a short trailing
        # segment that AUBO may answer with AUBO_REQUEST_IGNORE when its queue
        # is under pressure.  The last row keeps it: without it the path ends
        # with a deceleration onto the last hole, which then yields only
        # stationary (rejected) frames.
        is_last_row = index == len(planned) - 1
        row_scan_path = row_path if is_last_row else row_path[:-1]
        omitted_exit_waypoints += len(row_path) - len(row_scan_path)
        path.extend(row_scan_path)
        transit_flags.extend([False] * len(row_scan_path))
        if is_last_row:
            break
        next_start = planned[index + 1][0]
        midpoint = row_scan_path[-1].copy()
        midpoint[:3, 3] = (
            row_scan_path[-1][:3, 3] + next_start[:3, 3]
        ) / 2.0
        midpoint[2, 3] = safe_z
        rotations = Rotation.from_matrix(np.stack([
            row_scan_path[-1][:3, :3], next_start[:3, :3],
        ]))
        midpoint[:3, :3] = Slerp([0.0, 1.0], rotations)([0.5]).as_matrix()[0]
        path.extend([midpoint, next_start])
        transit_flags.extend([True, True])

    approach_waypoints: list[np.ndarray] = []
    if initial_tcp is not None:
        initial = np.asarray(initial_tcp, dtype=float).reshape(4, 4).copy()
        if not np.isfinite(initial).all():
            raise ValueError("连续路径起始TCP包含非有限值")
        if not math.isfinite(float(approach_position_step_mm)) or approach_position_step_mm <= 0:
            raise ValueError("连续路径起始位置步长必须是正数")
        if not math.isfinite(float(approach_rotation_step_deg)) or approach_rotation_step_deg <= 0:
            raise ValueError("连续路径起始姿态步长必须是正数")

        # Keep the transfer at the mapped surface's safety height.  If the
        # current TCP is higher, the first queued segment descends to that
        # height instead of carrying the unnecessary current Z through the
        # whole transfer.  The orientation is
        # interpolated during the horizontal move and the descent starts only
        # after most of that transfer, avoiding a standalone high-Z attitude IK.
        transition = initial.copy()
        if abs(float(safe_z - transition[2, 3])) > 0.5:
            transition[2, 3] = safe_z
            approach_waypoints.append(transition.copy())
        else:
            transition[2, 3] = safe_z

        rotation = Rotation.from_matrix(np.stack([
            transition[:3, :3], scan_start[:3, :3],
        ]))
        rotation_deg = float(np.degrees(
            (rotation[0].inv() * rotation[1]).magnitude()
        ))
        distance_xy = float(np.linalg.norm(
            scan_start[:2, 3] - transition[:2, 3],
        ))
        steps = max(
            2,
            math.ceil(distance_xy / float(approach_position_step_mm)),
            math.ceil(rotation_deg / float(approach_rotation_step_deg)),
        )
        descent_fraction = 0.65
        for index in range(1, steps + 1):
            fraction = index / steps
            pose = transition.copy()
            pose[:3, 3] = (
                (1.0 - fraction) * transition[:3, 3]
                + fraction * scan_start[:3, 3]
            )
            # Do not hold the final scan attitude at the high transfer height.
            # The last part of the same segment changes attitude while descending.
            descent = max(0.0, (fraction - descent_fraction) / (1.0 - descent_fraction))
            pose[2, 3] = safe_z + descent * (scan_start[2, 3] - safe_z)
            pose[:3, :3] = Slerp([0.0, 1.0], rotation)([fraction]).as_matrix()[0]
            approach_waypoints.append(pose)
        start = initial
        path = approach_waypoints + path
        transit_flags = [True] * len(approach_waypoints) + transit_flags
    points = [start, *path]
    lengths = [
        float(np.linalg.norm(right[:3, 3] - left[:3, 3]))
        for left, right in zip(points, points[1:])
    ]
    if min(lengths) <= 2.0 * blend_radius_mm:
        raise ValueError("连续路径短段不足以容纳交融半径")
    diagnostics: dict[str, Any] = {
        "path_length_mm": sum(lengths),
        "transfer_safe_tcp_z_mm": safe_z,
        "transfer_count": len(rows) - 1,
        "waypoint_count": len(path),
        "continuous_from_initial_tcp": initial_tcp is not None,
        "approach_waypoint_count": len(approach_waypoints),
        "omitted_exit_waypoint_count": omitted_exit_waypoints,
        "transit_segment_flags": transit_flags,
    }
    if tcp_to_camera is not None and mapped_points_mm is not None:
        camera_from_tcp = np.asarray(tcp_to_camera, dtype=float).reshape(4, 4)
        map_points = np.asarray(mapped_points_mm, dtype=float).reshape(-1, 3)
        if not len(map_points) or not np.isfinite(map_points).all():
            raise ValueError("连续路径缺少有效地图点")
        minimum = float("inf")
        for left, right, length in zip(points, points[1:], lengths):
            rotations = Rotation.from_matrix(np.stack([
                left[:3, :3], right[:3, :3],
            ]))
            for fraction in np.linspace(0.0, 1.0, max(2, math.ceil(length / 10.0) + 1)):
                orientation = Slerp([0.0, 1.0], rotations)([fraction]).as_matrix()[0]
                position = (1.0 - fraction) * left[:3, 3] + fraction * right[:3, 3]
                camera_position = position + orientation @ camera_from_tcp[:3, 3]
                minimum = min(minimum, float(np.min(np.linalg.norm(
                    map_points - camera_position, axis=1,
                ))))
        diagnostics["minimum_camera_to_mapped_point_mm"] = minimum
        if minimum < min_mapped_clearance_mm:
            raise ValueError(
                f"连续路径相机距地图点仅 {minimum:.1f} mm，低于"
                f" {min_mapped_clearance_mm:.1f} mm"
            )
    return start, path, diagnostics


def retain_nearest_frame(
    selected: list[tuple[float, int, Any]], *, score: float,
    timestamp_ns: int, bundle: Any, max_frames: int = 6,
    min_spacing_ns: int = 50_000_000,
) -> None:
    """Keep bounded, separated frames closest to the mapped hole view pose."""
    if not math.isfinite(score) or score < 0:
        return
    nearby = next((
        index for index, item in enumerate(selected)
        if abs(item[1] - timestamp_ns) < min_spacing_ns
    ), None)
    candidate = (score, timestamp_ns, bundle)
    if nearby is not None:
        if score < selected[nearby][0]:
            selected[nearby] = candidate
        return
    if len(selected) < max_frames:
        selected.append(candidate)
        return
    worst = max(range(len(selected)), key=lambda index: selected[index][0])
    if score < selected[worst][0]:
        selected[worst] = candidate


def interpolate_pose(
    samples: list[PoseSample], timestamp_ns: int, *,
    max_gap_ns: int = 30_000_000, max_read_ns: int = 8_000_000,
) -> np.ndarray | None:
    """Reject extrapolation and slow RPC reads before using a moving RGB frame."""
    if len(samples) < 2:
        return None
    lo, hi = 0, len(samples)
    while lo < hi:
        mid = (lo + hi) // 2
        if samples[mid].timestamp_ns < timestamp_ns:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0 or lo == len(samples):
        return None
    left, right = samples[lo - 1], samples[lo]
    gap = right.timestamp_ns - left.timestamp_ns
    if (
        gap <= 0 or gap > max_gap_ns
        or left.read_duration_ns > max_read_ns
        or right.read_duration_ns > max_read_ns
    ):
        return None
    fraction = (timestamp_ns - left.timestamp_ns) / gap
    result = np.eye(4)
    result[:3, 3] = (1.0 - fraction) * left.tcp[:3, 3] + fraction * right.tcp[:3, 3]
    rotations = Rotation.from_matrix(
        np.stack([left.tcp[:3, :3], right.tcp[:3, :3]]),
    )
    result[:3, :3] = Slerp([0.0, 1.0], rotations)([fraction]).as_matrix()[0]
    return result


def translation_speed_mm_s(
    samples: list[PoseSample], timestamp_ns: int, *,
    max_gap_ns: int = 30_000_000, max_read_ns: int = 8_000_000,
) -> float | None:
    if len(samples) < 2:
        return None
    lo, hi = 0, len(samples)
    while lo < hi:
        mid = (lo + hi) // 2
        if samples[mid].timestamp_ns < timestamp_ns:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0 or lo == len(samples):
        return None
    left, right = samples[lo - 1], samples[lo]
    gap = right.timestamp_ns - left.timestamp_ns
    if (
        gap <= 0 or gap > max_gap_ns
        or left.read_duration_ns > max_read_ns
        or right.read_duration_ns > max_read_ns
    ):
        return None
    return float(np.linalg.norm(
        right.tcp[:3, 3] - left.tcp[:3, 3]
    ) * 1e9 / gap)


def frame_time_ns(
    bundle: Any, *, clock_offset_ns: int = 0, max_arrival_age_ms: float = 80.0,
    max_future_ms: float = 2.0,
) -> int | None:
    system_us = getattr(bundle, "color_system_timestamp_us", None)
    host_ns = getattr(bundle, "host_timestamp_ns", None)
    if system_us is None or host_ns is None:
        return None
    frame_ns = int(system_us) * 1000 + int(clock_offset_ns)
    age_ns = int(host_ns) - frame_ns
    if age_ns < -max_future_ms * 1_000_000 or age_ns > max_arrival_age_ms * 1_000_000:
        return None
    return frame_ns


def validate_camera_preflight(
    bundles: list[Any], *, min_fps: float = 45.0, max_exposure_us: int = 10_000,
) -> dict[str, float]:
    """Require a shared host clock and short exposure before any scan motion."""
    if len(bundles) < 10:
        raise RuntimeError("运动取帧预检至少需要 10 张 RGB 帧")
    system_times = [getattr(bundle, "color_system_timestamp_us", None) for bundle in bundles]
    if any(value is None for value in system_times):
        raise RuntimeError("RGB 系统时间戳缺失")
    offsets = np.asarray([
        int(bundle.host_timestamp_ns) - int(bundle.color_system_timestamp_us) * 1000
        for bundle in bundles
    ], dtype=np.int64)
    # This SDK's system timestamp uses the host epoch. A median offset built from
    # arrival times can move genuine frames into the future and bias moving poses.
    clock_offset_ns = 0
    if float(np.percentile(np.abs(offsets), 95)) > 20_000_000:
        raise RuntimeError("RGB 系统时间戳与主机时钟偏差超过 20 ms")
    timestamps = [frame_time_ns(bundle) for bundle in bundles]
    if any(value is None for value in timestamps):
        raise RuntimeError("RGB 系统时间戳与主机时钟无法对齐")
    times = np.asarray(timestamps, dtype=np.int64)
    intervals = np.diff(times)
    if np.any(intervals <= 0):
        raise RuntimeError("RGB 系统时间戳不递增")
    fps = 1_000_000_000.0 / float(np.median(intervals))
    if fps < min_fps:
        raise RuntimeError(f"RGB 实测帧率 {fps:.1f} fps 低于 {min_fps:g} fps")
    exposures = [getattr(bundle, "color_exposure_us", None) for bundle in bundles]
    if any(value is None or value <= 0 for value in exposures):
        raise RuntimeError("RGB 曝光元数据不可用，无法控制运动模糊")
    exposure = max(int(value) for value in exposures)
    if exposure > max_exposure_us:
        raise RuntimeError(f"RGB 曝光 {exposure} us 超过 {max_exposure_us} us")
    return {
        "measured_fps": round(fps, 2),
        "maximum_exposure_us": exposure,
        "clock_offset_ns": clock_offset_ns,
        "host_minus_frame_p95_ms": round(float(np.percentile(offsets, 95)) / 1e6, 3),
        "host_minus_frame_p05_ms": round(float(np.percentile(offsets, 5)) / 1e6, 3),
    }


def robust_xy(points: list[np.ndarray], *, min_frames: int = 5) -> dict[str, Any]:
    if len(points) < min_frames:
        return {"success": False, "reason": "insufficient_valid_frames", "valid_frames": len(points)}
    array = np.asarray(points, dtype=float).reshape(-1, 2)
    median = np.median(array, axis=0)
    distances = np.linalg.norm(array - median, axis=1)
    keep = distances <= max(0.3, 3.0 * float(np.median(distances)))
    accepted = array[keep]
    if len(accepted) < min_frames:
        return {"success": False, "reason": "insufficient_inlier_frames", "valid_frames": len(accepted)}
    center = np.median(accepted, axis=0)
    scatter = float(np.percentile(np.linalg.norm(accepted - center, axis=1), 95))
    return {
        "success": bool(scatter <= 0.35),
        "reason": None if scatter <= 0.35 else "moving_xy_scatter_over_0.35mm",
        "xy_base_mm": center.tolist(),
        "scatter_p95_mm": scatter,
        "valid_frames": int(len(accepted)),
        "rejected_frames": int(len(array) - len(accepted)),
    }


def sample_tcp(pose_session: Any) -> PoseSample:
    before = time.time_ns()
    pose = list(pose_session.state.getTcpPose())
    after = time.time_ns()
    if len(pose) != 6 or not all(math.isfinite(float(value)) for value in pose):
        raise RuntimeError("运动中 TCP 读数无效")
    return PoseSample(
        (before + after) // 2,
        pose_session.pose_sdk_to_transform_mm(pose),
        after - before,
    )


def _put_label(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.6) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)


def draw_flyby_overlay(
    image_bgr: np.ndarray, *, header_lines: list[str],
    holes: list[dict[str, Any]], detections: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Annotate one moving frame: map anchor, YOLO boxes, fitted hole and status.

    ``holes`` items use raw (distorted) image pixels: ``hole_id``, ``anchor_px``,
    optional ``ellipse_center_px``/``ellipse_axes_px``/``ellipse_angle_deg`` and
    ``status`` (``accepted`` or a short rejection reason).
    """
    view = np.ascontiguousarray(image_bgr).copy()
    for detection in detections or []:
        x1, y1, x2, y2 = (int(round(float(value))) for value in detection["box"])
        cv2.rectangle(view, (x1, y1), (x2, y2), (200, 200, 0), 1, cv2.LINE_AA)
    for hole in holes:
        accepted = hole.get("status") == "accepted"
        color = (0, 200, 0) if accepted else (0, 0, 255)
        anchor = hole.get("anchor_px")
        if anchor is not None and np.isfinite(anchor).all():
            u, v = (int(round(float(value))) for value in anchor)
            cv2.drawMarker(view, (u, v), (0, 220, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
            _put_label(view, f"H{hole['hole_id']} {hole.get('status', '-')}", (u + 12, v - 12))
        center = hole.get("ellipse_center_px")
        axes = hole.get("ellipse_axes_px")
        if center is not None and axes is not None:
            cv2.ellipse(
                view,
                (int(round(float(center[0]))), int(round(float(center[1])))),
                (max(1, int(round(float(axes[0]) / 2.0))), max(1, int(round(float(axes[1]) / 2.0)))),
                float(hole.get("ellipse_angle_deg", 0.0)), 0, 360, color, 2, cv2.LINE_AA,
            )
            cv2.circle(
                view, (int(round(float(center[0]))), int(round(float(center[1])))),
                3, color, -1, cv2.LINE_AA,
            )
    for index, line in enumerate(header_lines):
        _put_label(view, line, (10, 26 + 24 * index))
    return view


def build_contact_sheet(
    tiles: list[tuple[np.ndarray, str]], *, tile_width: int = 480, columns: int = 4,
) -> np.ndarray:
    """Tile annotated frames into one overview image for quick review."""
    if not tiles:
        raise ValueError("没有可拼接的飞拍图像")
    resized = []
    for image, label in tiles:
        height, width = image.shape[:2]
        scale = tile_width / float(width)
        tile = cv2.resize(image, (tile_width, max(1, int(round(height * scale)))),
                          interpolation=cv2.INTER_AREA)
        _put_label(tile, label, (8, tile.shape[0] - 10), 0.55)
        resized.append(tile)
    tile_height = max(tile.shape[0] for tile in resized)
    columns = max(1, min(columns, len(resized)))
    rows = math.ceil(len(resized) / columns)
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 40, dtype=np.uint8)
    for index, tile in enumerate(resized):
        row, column = divmod(index, columns)
        sheet[row * tile_height:row * tile_height + tile.shape[0],
              column * tile_width:(column + 1) * tile_width] = tile
    return sheet


def thumbnail(image: np.ndarray, width: int = 480) -> np.ndarray:
    """Downscale before buffering overview tiles so long scans stay small in memory."""
    height = max(1, int(round(image.shape[0] * width / float(image.shape[1]))))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
