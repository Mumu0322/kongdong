"""Image artifacts for grouping and shared fine-localization results."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from aubo_workbench.geometry import invert_transform
from aubo_workbench.hole_localization_models import artifact_measure
from aubo_workbench.optics import camera_matrix, camera_transform, distortion_coeffs


_camera_matrix = camera_matrix
_distortion = distortion_coeffs


def _group_visual_point_px(hole: dict[str, Any]) -> np.ndarray | None:
    """读取分组可视化使用的初始图像孔中心。"""
    for value in (
        hole.get("initial_center_px"),
        hole.get("home_center_px"),
        (hole.get("initial_detection") or {}).get("center"),
    ):
        if value is None:
            continue
        try:
            point = np.asarray(value, dtype=np.float64).reshape(2)
        except (TypeError, ValueError):
            continue
        if np.isfinite(point).all():
            return point
    return None


def _write_visualization_pair(
    view: np.ndarray,
    output_base: Path,
    *,
    timing: Any | None = None,
    artifact_name: str | None = None,
) -> dict[str, str]:
    """保存PNG原图和高质量JPG，返回报告可直接写入的路径。"""
    png_path = output_base.with_suffix(".png")
    jpg_path = output_base.with_suffix(".jpg")
    event_name = artifact_name or f"visualization/{output_base.name}"
    with artifact_measure(
        timing,
        event_name,
        artifact_kind="visualization_pair",
        paths=[str(png_path), str(jpg_path)],
    ):
        output_base.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(png_path), view):
            raise RuntimeError(f"可视化PNG写入失败：{png_path}")
        if not cv2.imwrite(
            str(jpg_path), view,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        ):
            raise RuntimeError(f"可视化JPG写入失败：{jpg_path}")
    return {"png": str(png_path), "jpg": str(jpg_path)}


def _draw_black_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    """用白色描边保证黑白标签在任意背景上都可读。"""
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
        (255, 255, 255), thickness + 2, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
        (0, 0, 0), thickness, cv2.LINE_AA,
    )


def _save_grouping_plan_visualization(
    source_image_path: Path,
    output_base: Path,
    holes: list[dict[str, Any]],
    groups: list[list[dict[str, Any]]],
    title: str,
    diagnostics: list[dict[str, Any]] | None = None,
    *,
    timing: Any | None = None,
) -> dict[str, str] | None:
    """在初始选择图上显示组边界、组内连线、孔号和扫描顺序。"""
    view = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
    if view is None:
        return None
    # 分组审计图统一使用灰度底图；组边界、箭头和文字仍只使用黑白/灰度，
    # 这样现场打印或压缩成JPG后不会依赖颜色区分组别。
    view = cv2.cvtColor(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    diagnostics = diagnostics or []
    h, w = view.shape[:2]
    hole_by_id = {
        int(hole["hole_id"]): hole for hole in holes
        if hole.get("hole_id") is not None
    }
    point_by_id = {
        hole_id: _group_visual_point_px(hole)
        for hole_id, hole in hole_by_id.items()
    }

    # 顶部信息栏只使用黑白，便于现场打印和快速核对。
    panel_height = min(132, max(76, h // 5))
    cv2.rectangle(view, (0, 0), (w - 1, panel_height), (255, 255, 255), -1)
    _draw_black_text(
        view, title, (15, 25), scale=0.68, thickness=2,
    )
    _draw_black_text(
        view,
        "ORDER=HOME IMAGE v THEN u; GATE=ADJACENCY+COMPACTNESS+VIEW",
        (15, 51), scale=0.48, thickness=1,
    )
    summary = "  ".join(
        f"G{index:02d}:{len(group)} HOLES"
        for index, group in enumerate(groups, start=1)
    )
    _draw_black_text(view, summary or "NO GROUP", (15, 76), scale=0.50, thickness=1)
    _draw_black_text(
        view, "LABEL=GROUP/HOLE/RANK; ARROW=SPATIAL LINK; GRAY=GROUP",
        (15, 101), scale=0.46, thickness=1,
    )

    # 六档灰度足以区分常见的分组数量，同时保持黑白输出。
    gray_levels = (25, 65, 105, 145, 185, 225)
    for group_index, group in enumerate(groups, start=1):
        gray = gray_levels[(group_index - 1) % len(gray_levels)]
        color = (gray, gray, gray)
        ids = [int(hole["hole_id"]) for hole in group]
        points = [point_by_id.get(hole_id) for hole_id in ids]
        valid_points = [point for point in points if point is not None]
        integer_points = [tuple(np.rint(point).astype(int)) for point in valid_points]

        diagnostic = diagnostics[group_index - 1] if group_index - 1 < len(diagnostics) else {}
        diagnostic_edges = diagnostic.get("adjacency_edges") or []
        drawn_edges = 0
        for edge in diagnostic_edges:
            left = point_by_id.get(int(edge["from_hole_id"]))
            right = point_by_id.get(int(edge["to_hole_id"]))
            if left is None or right is None:
                continue
            cv2.line(
                view,
                tuple(np.rint(left).astype(int)),
                tuple(np.rint(right).astype(int)),
                color, 2, cv2.LINE_AA,
            )
            drawn_edges += 1
        if drawn_edges == 0 and len(integer_points) > 1:
            for left, right in zip(integer_points, integer_points[1:]):
                cv2.arrowedLine(view, left, right, color, 2, cv2.LINE_AA, tipLength=0.12)

        if len(integer_points) >= 3:
            hull = cv2.convexHull(np.asarray(integer_points, dtype=np.int32))
            cv2.polylines(view, [hull], True, (255, 255, 255), 8, cv2.LINE_AA)
            cv2.polylines(view, [hull], True, color, 3, cv2.LINE_AA)
        elif len(integer_points) == 2:
            cv2.line(view, integer_points[0], integer_points[1], (255, 255, 255), 9, cv2.LINE_AA)
            cv2.line(view, integer_points[0], integer_points[1], color, 4, cv2.LINE_AA)

        if valid_points:
            label_point = tuple(np.rint(np.min(np.asarray(valid_points), axis=0)).astype(int))
            aspect_value = diagnostic.get("aspect_ratio")
            aspect_label = (
                f" AR={float(aspect_value):.2f}"
                if isinstance(aspect_value, (int, float)) and math.isfinite(float(aspect_value))
                else ""
            )
            diameter_value = diagnostic.get("xy_diameter_mm")
            diameter_label = (
                f" XYD={float(diameter_value):.0f}mm"
                if isinstance(diameter_value, (int, float))
                and math.isfinite(float(diameter_value))
                else ""
            )
            group_label = (
                f"G{group_index:02d} ({len(group)} HOLES"
                f"{aspect_label}{diameter_label})"
            )
            label_origin = (
                max(5, min(w - 170, label_point[0] + 8)),
                max(panel_height + 22, min(h - 8, label_point[1] - 12)),
            )
            _draw_black_text(view, group_label, label_origin, scale=0.58, thickness=2)

        for rank, hole in enumerate(group, start=1):
            hole_id = int(hole["hole_id"])
            point = point_by_id.get(hole_id)
            if point is None:
                continue
            center = tuple(np.rint(point).astype(int))
            cv2.circle(view, center, 13, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(view, center, 13, color, 2, cv2.LINE_AA)
            _draw_black_text(
                view, f"G{group_index:02d}/H{hole_id} #{rank}",
                (center[0] + 16, center[1] + 5), scale=0.48, thickness=1,
            )

    grouped_ids = {
        int(hole["hole_id"])
        for group in groups
        for hole in group
        if hole.get("hole_id") is not None
    }
    ungrouped_ids: list[int] = []
    for hole_id, point in point_by_id.items():
        if hole_id in grouped_ids or point is None:
            continue
        ungrouped_ids.append(hole_id)
        center = tuple(np.rint(point).astype(int))
        cv2.drawMarker(view, center, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 30, 5, cv2.LINE_AA)
        cv2.drawMarker(view, center, (0, 0, 0), cv2.MARKER_TILTED_CROSS, 22, 2, cv2.LINE_AA)
        _draw_black_text(
            view, f"UNASSIGNED/H{hole_id}",
            (center[0] + 16, center[1] + 5), scale=0.48, thickness=1,
        )

    # 对未能读取初始坐标的孔仍显示一行报告提示，不默默丢掉。
    missing_ids = [
        int(hole["hole_id"]) for hole in holes
        if _group_visual_point_px(hole) is None
    ]
    if missing_ids or ungrouped_ids:
        _draw_black_text(
            view,
            f"UNMARKED={missing_ids}  UNASSIGNED={sorted(ungrouped_ids)}",
            (15, h - 18), scale=0.50, thickness=1,
        )
    try:
        return _write_visualization_pair(
            view,
            output_base,
            timing=timing,
            artifact_name=f"grouping_plan/{output_base.name}",
        )
    except Exception:
        # 可视化是审计产物，不能因为磁盘/图像编码问题影响定位主流程。
        return None


def _save_group_capture_visualization(
    source_image_path: str | Path | None,
    output_base: Path,
    group_index: int,
    group: list[dict[str, Any]],
    projected_holes_px: dict[Any, Any] | None,
    group_bbox_px: list[float] | None,
    title: str,
    *,
    timing: Any | None = None,
) -> dict[str, str] | None:
    """在共享拍摄帧上标出该组规划的目标孔投影和共同视野框。"""
    if source_image_path is None:
        return None
    view = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
    if view is None:
        return None
    view = cv2.cvtColor(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    h, w = view.shape[:2]
    projected_holes_px = projected_holes_px or {}
    points: list[tuple[int, int, int]] = []
    for rank, hole in enumerate(group, start=1):
        hole_id = int(hole["hole_id"])
        value = projected_holes_px.get(hole_id, projected_holes_px.get(str(hole_id)))
        if value is None:
            continue
        try:
            point = np.asarray(value, dtype=np.float64).reshape(2)
        except (TypeError, ValueError):
            continue
        if np.isfinite(point).all():
            points.append((hole_id, rank, tuple(np.rint(point).astype(int))))

    if group_bbox_px is not None:
        try:
            bbox = np.rint(np.asarray(group_bbox_px, dtype=np.float64).reshape(4)).astype(int)
            cv2.rectangle(
                view, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                (255, 255, 255), 7, cv2.LINE_AA,
            )
            cv2.rectangle(
                view, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                (0, 0, 0), 3, cv2.LINE_AA,
            )
        except (TypeError, ValueError):
            pass

    for (_, _, left), (_, _, right) in zip(points, points[1:]):
        cv2.arrowedLine(view, left, right, (0, 0, 0), 2, cv2.LINE_AA, tipLength=0.12)
    for hole_id, rank, point in points:
        cv2.drawMarker(view, point, (255, 255, 255), cv2.MARKER_CROSS, 25, 5, cv2.LINE_AA)
        cv2.drawMarker(view, point, (0, 0, 0), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        _draw_black_text(
            view, f"G{int(group_index):02d}/H{hole_id} #{rank}",
            (point[0] + 10, point[1] - 10), scale=0.50, thickness=1,
        )

    panel_height = min(88, max(52, h // 8))
    cv2.rectangle(view, (0, 0), (w - 1, panel_height), (255, 255, 255), -1)
    _draw_black_text(view, title, (15, 25), scale=0.64, thickness=2)
    _draw_black_text(
        view,
        f"G{int(group_index):02d}  HOLES={len(group)}  BOX=SHARED VIEW  ARROW=ORDER",
        (15, 52), scale=0.48, thickness=1,
    )
    try:
        return _write_visualization_pair(
            view,
            output_base,
            timing=timing,
            artifact_name=f"group_capture/{output_base.name}",
        )
    except Exception:
        # 可视化是审计产物，不能因为磁盘/图像编码问题影响定位主流程。
        return None


def _project_base_point_to_pixel(
    point_base_mm: np.ndarray, T_base_camera: np.ndarray, intrinsics: Any,
) -> np.ndarray:
    """将基坐标三维点投影到当前RGB原始像素坐标（包含镜头畸变）。"""
    T_camera_base = invert_transform(T_base_camera)
    point_camera = T_camera_base[:3, :3] @ np.asarray(point_base_mm, dtype=np.float64) + T_camera_base[:3, 3]
    if point_camera[2] <= 1e-6:
        raise ValueError("目标孔位于当前相机后方")
    projected, _ = cv2.projectPoints(
        point_camera.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
        _camera_matrix(intrinsics), _distortion(intrinsics),
    )
    return np.asarray(projected, dtype=np.float64).reshape(2)


def _save_batch_fine_final_result_overlay(
    run_dir: Path,
    batch_fine_results: dict[int, dict[str, Any]],
    final_results: list[dict[str, Any]],
    handeye: Any,
    *,
    timing: Any | None = None,
) -> dict[str, Any] | None:
    """把共享精拍测量点、孔级目标和最终TCP画回同一张260图像。"""
    capture = next((
        item for item in batch_fine_results.values()
        if item.get("success")
        and item.get("fine_capture_overlay_path")
        and item.get("capture_tcp") is not None
        and item.get("intrinsics") is not None
    ), None)
    if capture is None:
        return None
    source_path = Path(str(capture["fine_capture_overlay_path"]))
    view = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if view is None:
        return None

    T_base_camera = camera_transform(
        np.asarray(capture["capture_tcp"], dtype=np.float64).reshape(4, 4),
        handeye.T_tcp_rgb_camera,
    )
    intrinsics = capture["intrinsics"]
    records: list[dict[str, Any]] = []

    def projected(point: Any) -> np.ndarray | None:
        if point is None:
            return None
        try:
            value = np.asarray(point, dtype=np.float64).reshape(3)
            if not np.isfinite(value).all():
                return None
            return _project_base_point_to_pixel(value, T_base_camera, intrinsics)
        except Exception:
            return None

    for result in final_results:
        if (
            result.get("status") != "completed"
            or not str(result.get("batch_fine_source", "")).startswith("batch_fine")
        ):
            continue
        hole_id = int(result["hole_id"])
        fine_point = np.asarray(
            result.get("hole_center_base_naive_mm", result["hole_center_base_mm"]),
            dtype=np.float64,
        ).reshape(3)
        joint_point_value = result.get("batch_fine_joint_visual_point_base_mm")
        joint_point = (
            None if joint_point_value is None else
            np.asarray(joint_point_value, dtype=np.float64).reshape(3)
        )
        target_point = np.asarray(result["target_point_base_mm"], dtype=np.float64).reshape(3)
        final_pose = result.get("final_tcp_pose_m_rad")
        final_tcp_point = (
            np.asarray(final_pose[:3], dtype=np.float64) * 1000.0
            if final_pose is not None and len(final_pose) >= 3
            and result.get("final_xy_motion") is not None else None
        )
        fine_px = projected(fine_point)
        joint_px = projected(joint_point)
        target_px = projected(target_point)
        final_tcp_px = projected(final_tcp_point)

        if fine_px is not None:
            p = tuple(np.rint(fine_px).astype(int))
            cv2.drawMarker(view, p, (0, 0, 255), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
            cv2.putText(view, f"H{hole_id} FINE", (p[0] + 8, p[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1, cv2.LINE_AA)
        if joint_px is not None:
            p = tuple(np.rint(joint_px).astype(int))
            cv2.drawMarker(view, p, (0, 255, 0), cv2.MARKER_DIAMOND, 22, 2, cv2.LINE_AA)
            cv2.putText(view, f"H{hole_id} JOINT", (p[0] + 8, p[1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)
        if target_px is not None:
            p = tuple(np.rint(target_px).astype(int))
            cv2.drawMarker(view, p, (255, 255, 0), cv2.MARKER_DIAMOND, 22, 2, cv2.LINE_AA)
            cv2.putText(view, f"H{hole_id} TARGET", (p[0] + 8, p[1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1, cv2.LINE_AA)
        if final_tcp_px is not None:
            p = tuple(np.rint(final_tcp_px).astype(int))
            cv2.drawMarker(view, p, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 22, 2, cv2.LINE_AA)
            cv2.putText(view, f"H{hole_id} TCP", (p[0] + 8, p[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
            compensation_start_px = joint_px if joint_px is not None else fine_px
            if compensation_start_px is not None:
                cv2.arrowedLine(
                    view, tuple(np.rint(compensation_start_px).astype(int)), p,
                    (0, 165, 255), 2, cv2.LINE_AA, tipLength=0.15,
                )

        records.append({
            "hole_id": hole_id,
            "fine_point_base_mm": fine_point,
            "fine_point_px": fine_px,
            "joint_point_base_mm": joint_point,
            "joint_point_px": joint_px,
            "target_point_base_mm": target_point,
            "target_point_px": target_px,
            "final_tcp_base_mm": final_tcp_point,
            "final_tcp_px": final_tcp_px,
            "fine_to_final_tcp_xy_mm": (
                None if final_tcp_point is None else
                np.asarray(
                    final_tcp_point[:2] - (
                        joint_point[:2] if joint_point is not None else fine_point[:2]
                    ),
                    dtype=np.float64,
                )
            ),
        })

    if not records:
        return None
    cv2.rectangle(view, (8, 8), (700, 72), (0, 0, 0), -1)
    cv2.putText(view, "FINAL RESULT ON SHARED 260mm IMAGE", (18, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        view, "red+=RAW FINE  green diamond=JOINT XY  cyan diamond=TARGET  yellow x=TCP",
        (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA,
    )
    output_path = run_dir / "batch_fine_260_final_result_overlay.png"
    with artifact_measure(
        timing,
        "final_result_overlay/write",
        artifact_kind="final_result_overlay",
        paths=[str(output_path)],
    ):
        if not cv2.imwrite(str(output_path), view):
            return None
    return {
        "image_path": str(output_path),
        "source_image_path": str(source_path),
        "coordinate_frame": "shared_260mm_capture_rgb",
        "legend": {
            "red_cross": "raw_per_hole_fine_3d_center_reprojected",
            "green_diamond": "shared_joint_xy_with_limited_local_residual_and_tilt_correction",
            "cyan_diamond": "final_hole_target_fine_xy_with_coarse_z",
            "yellow_tilted_cross": "actual_final_tcp_after_xy_compensation_and_y_trim",
            "orange_arrow": "joint_or_raw_visual_point_to_actual_final_tcp",
        },
        "holes": records,
    }
