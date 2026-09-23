"""Compact, non-ambiguous result reports for two-stage hole localization.

The existing ``report.json`` intentionally contains a large amount of diagnostic
data.  This module builds a small operator-facing summary from that report and
from the timing intervals.  Timing is calculated from the deepest active event
on a timeline, so a parent event and its nested capture event are not counted
twice.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from aubo_workbench.hole_localization_models import (
    artifact_measure,
    infer_timing_category,
    infer_visual_timing_stage,
    VISUAL_TIMING_STAGES,
)
from aubo_workbench.io_utils import atomic_write_json, jsonable


_TIMING_CATEGORIES = (
    "robot_motion",
    "vision_compute",
    "safety_wait",
    "operator_wait",
    "artifact_io",
    "other",
)

_VISUAL_STAGE_LABELS = {
    "camera_startup": "相机启动/管线准备",
    "frame_acquisition": "RGB/RGB-D取帧",
    "yolo_inference": "YOLO检测与孔位关联",
    "pointcloud_processing": "点云/平面几何",
    "ellipse_fitting": "椭圆/几何中心拟合",
    "coordinate_transform": "像素到基坐标变换",
    "fusion_quality": "多帧融合与质量判断",
    "final_point_calculation": "最终点计算",
    "visual_pipeline_other": "其它视觉流水线",
}


def _round(value: Any, digits: int = 3) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return round(number, digits)


def _category(event: dict[str, Any]) -> str:
    category = str(event.get("category", "")).strip()
    return category if category in _TIMING_CATEGORIES else infer_timing_category(
        str(event.get("name", "")),
    )


def _is_artifact(event: dict[str, Any]) -> bool:
    return _category(event) == "artifact_io" or bool(
        event.get("exclude_from_effective", False),
    )


def _visual_stage(event: dict[str, Any]) -> str | None:
    """返回事件所属的视觉子阶段；非视觉事件返回 ``None``。

    优先读取记录时写入的显式阶段，旧报告再走名称推断。安全/人工/机械臂
    事件即使名称里包含 ``capture`` 也不能被误计入视觉时间。
    """
    category = _category(event)
    if category in {"robot_motion", "safety_wait", "operator_wait", "artifact_io"}:
        # 丢弃相机预热帧本身仍是视觉取帧；其它 safety wait 是机器人稳定等待。
        name = str(event.get("name", "")).strip().lower().replace("\\", "/")
        if "discard_settle_frames" not in name:
            return None
    stage = str(event.get("visual_stage", "")).strip()
    if stage in VISUAL_TIMING_STAGES:
        return stage
    return infer_visual_timing_stage(str(event.get("name", "")))


def _interval(event: dict[str, Any]) -> tuple[float, float] | None:
    try:
        start = float(event["start_offset_s"])
        end = float(event["end_offset_s"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end):
        return None
    if end < start:
        start, end = end, start
    return max(0.0, start), max(0.0, end)


def _timeline_segments(
    events: Iterable[dict[str, Any]],
    start_s: float,
    end_s: float,
) -> list[tuple[float, float, str]]:
    """把嵌套事件展开成互不重叠的时间片。"""
    start_s = max(0.0, float(start_s))
    end_s = max(start_s, float(end_s))
    prepared: list[tuple[float, float, int, int, str]] = []
    for index, event in enumerate(events):
        interval = _interval(event)
        if interval is None:
            continue
        event_start, event_end = interval
        clipped_start = max(start_s, event_start)
        clipped_end = min(end_s, event_end)
        if clipped_end <= clipped_start:
            continue
        try:
            depth = int(event.get("depth", 0))
        except (TypeError, ValueError):
            depth = 0
        # 同一深度同时存在时，让产物和等待优先显示为自己的类别。
        category_priority = {
            "artifact_io": 4,
            "operator_wait": 3,
            "safety_wait": 2,
            "vision_compute": 1,
            "robot_motion": 1,
            "other": 0,
        }.get(_category(event), 0)
        prepared.append((clipped_start, clipped_end, depth, category_priority, _category(event)))
    if not prepared or end_s <= start_s:
        return []

    boundaries = {start_s, end_s}
    for event_start, event_end, *_ in prepared:
        boundaries.add(event_start)
        boundaries.add(event_end)
    sorted_boundaries = sorted(boundaries)
    segments: list[tuple[float, float, str]] = []
    for left, right in zip(sorted_boundaries, sorted_boundaries[1:]):
        if right <= left:
            continue
        active = [
            item for item in prepared
            if item[0] <= left + 1e-9 and item[1] >= right - 1e-9
        ]
        if not active:
            continue
        _, _, _, _, category = max(
            active,
            key=lambda item: (item[2], item[3]),
        )
        if segments and segments[-1][2] == category and abs(segments[-1][1] - left) < 1e-9:
            segments[-1] = (segments[-1][0], right, category)
        else:
            segments.append((left, right, category))
    return segments


def _visual_timeline_segments(
    events: Iterable[dict[str, Any]],
    start_s: float,
    end_s: float,
) -> list[tuple[float, float, str]]:
    """把视觉事件展开成互不重叠的子阶段时间片。

    视觉父事件通常包含人工选择、文件写入或更细的视觉子事件。这里把
    所有事件放到同一时间轴，使用深度优先级选择最具体的事件；非视觉子
    事件会遮住视觉父事件，从而不会把等待/写盘时间偷偷算回去。
    """
    start_s = max(0.0, float(start_s))
    end_s = max(start_s, float(end_s))
    prepared: list[tuple[float, float, int, int, str | None]] = []
    for index, event in enumerate(events):
        interval = _interval(event)
        if interval is None:
            continue
        event_start, event_end = interval
        clipped_start = max(start_s, event_start)
        clipped_end = min(end_s, event_end)
        if clipped_end <= clipped_start:
            continue
        try:
            depth = int(event.get("depth", 0))
        except (TypeError, ValueError):
            depth = 0
        category = _category(event)
        stage = _visual_stage(event)
        # 同一深度下，人工/安全/写盘事件必须遮住其父级视觉容器；
        # 普通非视觉事件也应优先于容器，避免“未知代码”被冒充视觉算法。
        priority = {
            "artifact_io": 9,
            "operator_wait": 8,
            "safety_wait": 8,
            "robot_motion": 5,
            "other": 4,
            "vision_compute": 5,
        }.get(category, 1)
        if stage is not None:
            priority += 1
        prepared.append((clipped_start, clipped_end, depth, priority, stage))
    if not prepared or end_s <= start_s:
        return []

    boundaries = {start_s, end_s}
    for event_start, event_end, *_ in prepared:
        boundaries.add(event_start)
        boundaries.add(event_end)
    sorted_boundaries = sorted(boundaries)
    segments: list[tuple[float, float, str]] = []
    for left, right in zip(sorted_boundaries, sorted_boundaries[1:]):
        if right <= left:
            continue
        active = [
            item for item in prepared
            if item[0] <= left + 1e-9 and item[1] >= right - 1e-9
        ]
        if not active:
            continue
        selected = max(active, key=lambda item: (item[2], item[3]))
        stage = selected[4]
        if stage is None:
            continue
        if segments and segments[-1][2] == stage and abs(segments[-1][1] - left) < 1e-9:
            segments[-1] = (segments[-1][0], right, stage)
        else:
            segments.append((left, right, stage))
    return segments


def _empty_visual_timing(*, interval_accounting: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "total_visual_s": 0.0,
        "visual_compute_s": 0.0,
        "visual_wall_elapsed_s": 0.0,
        "visual_measured_timeline_s": 0.0,
        "visual_unmeasured_gap_s": 0.0,
        "visual_event_count": 0,
        "interval_accounting": bool(interval_accounting),
        "stage_breakdown": {},
    }
    for stage in VISUAL_TIMING_STAGES:
        result[f"{stage}_s"] = 0.0
    return result


def _legacy_visual_timing_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    """兼容旧报告：按有效事件累加视觉阶段（旧报告无法消除父子重叠）。"""
    result = _empty_visual_timing(interval_accounting=False)
    stage_totals = {stage: 0.0 for stage in VISUAL_TIMING_STAGES}
    for event in events:
        stage = _visual_stage(event)
        if stage is None:
            continue
        try:
            elapsed = float(event.get("effective_elapsed_s", event.get("elapsed_s", 0.0)))
        except (TypeError, ValueError):
            elapsed = 0.0
        if math.isfinite(elapsed) and elapsed > 0.0:
            stage_totals[stage] += elapsed
            result["visual_event_count"] += 1
    total = sum(stage_totals.values())
    result["total_visual_s"] = round(total, 6)
    result["visual_compute_s"] = result["total_visual_s"]
    result["visual_wall_elapsed_s"] = result["total_visual_s"]
    result["visual_measured_timeline_s"] = result["total_visual_s"]
    result["stage_breakdown"] = {
        stage: round(value, 6) for stage, value in stage_totals.items()
        if value > 0.0
    }
    for stage, value in stage_totals.items():
        result[f"{stage}_s"] = round(value, 6)
    return result


def summarize_visual_timing_window(
    events: list[dict[str, Any]],
    *,
    start_s: float | None = None,
    end_s: float | None = None,
) -> dict[str, Any]:
    """只汇总视觉时间，并拆分为相机、检测、点云、几何和融合阶段。"""
    intervals = [_interval(event) for event in events]
    intervals = [item for item in intervals if item is not None]
    if not intervals:
        return _legacy_visual_timing_summary(events)
    inferred_start = min(item[0] for item in intervals)
    inferred_end = max(item[1] for item in intervals)
    window_start = inferred_start if start_s is None else max(0.0, float(start_s))
    window_end = inferred_end if end_s is None else max(window_start, float(end_s))
    if window_end <= window_start:
        return _legacy_visual_timing_summary(events)

    segments = _visual_timeline_segments(events, window_start, window_end)
    stage_totals = {stage: 0.0 for stage in VISUAL_TIMING_STAGES}
    for left, right, stage in segments:
        stage_totals[stage] = stage_totals.get(stage, 0.0) + right - left
    measured = sum(stage_totals.values())
    visual_event_count = sum(
        1 for event in events
        if _visual_stage(event) is not None and _interval(event) is not None
        and not _is_artifact(event)
    )
    result = _empty_visual_timing(interval_accounting=True)
    result.update({
        "total_visual_s": round(measured, 6),
        "visual_compute_s": round(measured, 6),
        "visual_wall_elapsed_s": round(window_end - window_start, 6),
        "visual_measured_timeline_s": round(measured, 6),
        "visual_unmeasured_gap_s": round(
            max(0.0, (window_end - window_start) - measured), 6,
        ),
        "visual_event_count": visual_event_count,
        "window_start_offset_s": round(window_start, 6),
        "window_end_offset_s": round(window_end, 6),
        "stage_breakdown": {
            stage: round(value, 6)
            for stage, value in stage_totals.items()
            if value > 0.0
        },
    })
    for stage, value in stage_totals.items():
        result[f"{stage}_s"] = round(value, 6)
    return result


def summarize_visual_timing_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_visual_timing_window(events)


def _legacy_timing_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    """兼容没有时间区间的旧报告，仍保持旧字段含义。"""
    totals = {category: 0.0 for category in _TIMING_CATEGORIES}
    measured_event_sum_s = 0.0
    artifact_event_count = 0
    for event in events:
        elapsed_s = _round(event.get("elapsed_s", 0.0), 6)
        measured_event_sum_s += elapsed_s
        category = _category(event)
        if category == "artifact_io":
            artifact_event_count += 1
        totals[category] += _round(
            event.get("effective_elapsed_s", elapsed_s), 6,
        )
    effective_total_s = sum(
        totals[category]
        for category in ("robot_motion", "vision_compute", "other")
    )
    production_s = effective_total_s + totals["safety_wait"]
    result = {
        "wall_elapsed_s": round(measured_event_sum_s, 6),
        "measured_timeline_s": round(measured_event_sum_s, 6),
        "unmeasured_gap_s": 0.0,
        "robot_motion_s": round(totals["robot_motion"], 6),
        "vision_compute_s": round(totals["vision_compute"], 6),
        "safety_wait_s": round(totals["safety_wait"], 6),
        "operator_wait_s": round(totals["operator_wait"], 6),
        "artifact_io_s": round(totals["artifact_io"], 6),
        "other_s": round(totals["other"], 6),
        "effective_total_s": round(effective_total_s, 6),
        "production_cycle_s": round(production_s, 6),
        "pure_execution_s": round(effective_total_s, 6),
        "measured_event_sum_s": round(measured_event_sum_s, 6),
        "artifact_event_count": artifact_event_count,
        "interval_accounting": False,
    }
    result.update({
        "motion_s": result["robot_motion_s"],
        "excluded_wait_s": round(
            result["safety_wait_s"] + result["operator_wait_s"], 6,
        ),
    })
    return result


def summarize_timing_window(
    events: list[dict[str, Any]],
    *,
    start_s: float | None = None,
    end_s: float | None = None,
) -> dict[str, Any]:
    """返回一个不重复计算的时间窗口汇总。

    ``production_cycle_s`` 包含必要安全等待，但排除人工等待和文件产物；
    ``pure_execution_s`` 进一步排除安全等待，仅用于性能分析。
    """
    intervals = [_interval(event) for event in events]
    intervals = [item for item in intervals if item is not None]
    if not intervals:
        return _legacy_timing_summary(events)
    inferred_start = min(item[0] for item in intervals)
    inferred_end = max(item[1] for item in intervals)
    window_start = inferred_start if start_s is None else max(0.0, float(start_s))
    window_end = inferred_end if end_s is None else max(window_start, float(end_s))
    if window_end <= window_start:
        return _legacy_timing_summary(events)

    segments = _timeline_segments(events, window_start, window_end)
    totals = {category: 0.0 for category in _TIMING_CATEGORIES}
    for left, right, category in segments:
        totals[category] += right - left
    measured_timeline_s = sum(right - left for left, right, _ in segments)
    measured_event_sum_s = sum(
        max(0.0, min(window_end, _interval(event)[1]) - max(window_start, _interval(event)[0]))
        for event in events
        if _interval(event) is not None
    )
    effective_total_s = sum(
        totals[category]
        for category in ("robot_motion", "vision_compute", "other")
    )
    production_s = sum(
        totals[category]
        for category in ("robot_motion", "vision_compute", "safety_wait", "other")
    )
    artifact_event_count = sum(
        1 for event in events
        if _is_artifact(event) and _interval(event) is not None
    )
    result = {
        "wall_elapsed_s": round(window_end - window_start, 6),
        "measured_timeline_s": round(measured_timeline_s, 6),
        "unmeasured_gap_s": round(
            max(0.0, (window_end - window_start) - measured_timeline_s), 6,
        ),
        "robot_motion_s": round(totals["robot_motion"], 6),
        "vision_compute_s": round(totals["vision_compute"], 6),
        "safety_wait_s": round(totals["safety_wait"], 6),
        "operator_wait_s": round(totals["operator_wait"], 6),
        "artifact_io_s": round(totals["artifact_io"], 6),
        "other_s": round(totals["other"], 6),
        "effective_total_s": round(effective_total_s, 6),
        "production_cycle_s": round(production_s, 6),
        "pure_execution_s": round(
            max(0.0, production_s - totals["safety_wait"]), 6,
        ),
        "measured_event_sum_s": round(measured_event_sum_s, 6),
        "artifact_event_count": artifact_event_count,
        "interval_accounting": True,
        "window_start_offset_s": round(window_start, 6),
        "window_end_offset_s": round(window_end, 6),
    }
    result.update({
        "motion_s": result["robot_motion_s"],
        "excluded_wait_s": round(
            result["safety_wait_s"] + result["operator_wait_s"], 6,
        ),
    })
    return result


def summarize_timing_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总完整事件列表；保留一个便于诊断模块调用的明确入口。"""
    return summarize_timing_window(events)


def _motion_xyz(value: Any) -> list[float] | None:
    """从 3/4/16 元素位姿值中提取 XYZ(mm)。"""
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if len(values) >= 16:
        # 兼容扁平化的 4x4 TCP 变换矩阵（行主序）。
        values = [values[3], values[7], values[11]]
    elif len(values) >= 3:
        values = values[:3]
    else:
        return None
    if not all(math.isfinite(item) for item in values):
        return None
    return values


def summarize_motion_distance(
    events: Iterable[dict[str, Any]],
    *,
    include_segments: bool = False,
) -> dict[str, Any]:
    """汇总机器人 TCP 运动距离，并拆分水平、上升和下降。

    距离来自每个已完成运动的规划 TCP 起点/终点差值。对于 moveLine，
    这是该段规划直线长度；对于回原点 moveJoint，这是 TCP 端点位移，
    不能等同于控制器内部的真实关节轨迹长度，因此结果明确标注为
    ``planned_tcp_waypoint_delta``。
    """
    source = [event for event in events if isinstance(event, dict)]
    total_path = 0.0
    total_horizontal = 0.0
    total_up = 0.0
    total_down = 0.0
    completed_count = 0
    by_motion_type: dict[str, float] = {}
    by_profile: dict[str, float] = {}
    by_hole: dict[str, dict[str, float]] = {}
    segments: list[dict[str, Any]] = []

    for event in source:
        status = str(event.get("status", "completed")).strip().lower()
        if status not in {"", "completed", "success", "succeeded"}:
            continue
        start = _motion_xyz(event.get("start_xyz_mm"))
        target = _motion_xyz(event.get("target_xyz_mm"))
        delta = _motion_xyz(event.get("delta_xyz_mm"))
        if start is not None and target is not None:
            dx = target[0] - start[0]
            dy = target[1] - start[1]
            dz = target[2] - start[2]
        elif delta is not None:
            dx, dy, dz = delta
            start = None
            target = None
        else:
            # 兼容未来/手工写入的聚合字段；没有端点时不能反推出方向，
            # 只接受同时提供了完整的水平、上升、下降字段的记录。
            try:
                path = float(event["path_distance_mm"])
                horizontal = float(event["horizontal_distance_mm"])
                up = float(event["vertical_up_mm"])
                down = float(event["vertical_down_mm"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(item) for item in (path, horizontal, up, down)):
                continue
            dx = dy = dz = 0.0
        try:
            horizontal = float(event.get("horizontal_distance_mm", math.hypot(dx, dy)))
            path = float(event.get("path_distance_mm", math.sqrt(dx * dx + dy * dy + dz * dz)))
            up = float(event.get("vertical_up_mm", max(0.0, dz)))
            down = float(event.get("vertical_down_mm", max(0.0, -dz)))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(item) for item in (path, horizontal, up, down)):
            continue
        path = max(0.0, path)
        horizontal = max(0.0, horizontal)
        up = max(0.0, up)
        down = max(0.0, down)
        total_path += path
        total_horizontal += horizontal
        total_up += up
        total_down += down
        completed_count += 1

        motion_type = str(event.get("motion_type", "unknown"))
        by_motion_type[motion_type] = by_motion_type.get(motion_type, 0.0) + path
        profile = str(event.get("motion_profile", "unknown"))
        by_profile[profile] = by_profile.get(profile, 0.0) + path
        match = re.search(
            r"(?:孔\s*|hole[_ -]*|(?:^|[-_/])h)(\d+)",
            str(event.get("name", "")),
            re.IGNORECASE,
        )
        if match:
            hole_key = str(int(match.group(1)))
            hole = by_hole.setdefault(
                hole_key,
                {
                    "path_distance_mm": 0.0,
                    "horizontal_distance_mm": 0.0,
                    "vertical_up_mm": 0.0,
                    "vertical_down_mm": 0.0,
                    "segment_count": 0.0,
                },
            )
            hole["path_distance_mm"] += path
            hole["horizontal_distance_mm"] += horizontal
            hole["vertical_up_mm"] += up
            hole["vertical_down_mm"] += down
            hole["segment_count"] += 1.0
        if include_segments:
            item = dict(event)
            item.update({
                "path_distance_mm": _round(path, 6),
                "horizontal_distance_mm": _round(horizontal, 6),
                "vertical_up_mm": _round(up, 6),
                "vertical_down_mm": _round(down, 6),
            })
            segments.append(item)

    result: dict[str, Any] = {
        "schema_version": 1,
        "coordinate_frame": "robot_base_mm",
        "distance_basis": "planned_tcp_waypoint_delta",
        "segment_count": completed_count,
        "total_path_mm": _round(total_path, 3),
        "horizontal_distance_mm": _round(total_horizontal, 3),
        "vertical_up_mm": _round(total_up, 3),
        "vertical_down_mm": _round(total_down, 3),
        "vertical_net_mm": _round(total_up - total_down, 3),
        "by_motion_type_mm": {
            key: _round(value, 3) for key, value in sorted(by_motion_type.items())
        },
        "by_motion_profile_mm": {
            key: _round(value, 3) for key, value in sorted(by_profile.items())
        },
        "by_hole": {
            key: {
                "path_distance_mm": _round(value["path_distance_mm"], 3),
                "horizontal_distance_mm": _round(value["horizontal_distance_mm"], 3),
                "vertical_up_mm": _round(value["vertical_up_mm"], 3),
                "vertical_down_mm": _round(value["vertical_down_mm"], 3),
                "vertical_net_mm": _round(
                    value["vertical_up_mm"] - value["vertical_down_mm"], 3,
                ),
                "segment_count": int(value["segment_count"]),
            }
            for key, value in sorted(by_hole.items(), key=lambda item: int(item[0]))
        },
    }
    if include_segments:
        result["segments"] = segments
    return result


def _marker_offset(timing: dict[str, Any], name: str) -> float | None:
    for marker in reversed(list(timing.get("markers") or [])):
        if str(marker.get("name", "")) == name:
            try:
                value = float(marker["offset_s"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                return max(0.0, value)
    return None


def _events_for_prefix(events: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    return [event for event in events if str(event.get("name", "")).startswith(prefix)]


def _window_for_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return summarize_timing_window([])
    return summarize_timing_window(events)


def _safe_hole_id(value: Any) -> int | None:
    try:
        hole_id = int(value)
    except (TypeError, ValueError):
        return None
    return hole_id if hole_id > 0 else None


def _planned_hole_ids(report: dict[str, Any]) -> list[int]:
    """读取本轮计划顺序，供中断报告区分未处理孔和失败孔。"""
    stages = report.get("stages") or {}
    plan = stages.get("sequential_plan") or {}
    candidates = plan.get("hole_order") if isinstance(plan, dict) else None
    if not isinstance(candidates, list):
        home = stages.get("home_selection") or {}
        candidates = home.get("selected_holes") if isinstance(home, dict) else None

    result: list[int] = []
    seen: set[int] = set()
    for item in candidates or []:
        value = item.get("hole_id") if isinstance(item, dict) else item
        hole_id = _safe_hole_id(value)
        if hole_id is not None and hole_id not in seen:
            seen.add(hole_id)
            result.append(hole_id)
    return result


def _hole_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    final_result = report.get("final_result") or {}
    holes = final_result.get("holes") if isinstance(final_result, dict) else None
    stages = report.get("stages") or {}
    if not isinstance(holes, list) or not holes:
        holes = None
        # processed_holes 是逐孔阶段每完成/延后一个孔就更新的中间检查点；
        # sequential_holes 只有整个逐孔阶段正常收尾后才会生成。
        for stage_name in ("processed_holes", "sequential_holes"):
            stage = stages.get(stage_name) or {}
            candidate = stage.get("holes") if isinstance(stage, dict) else None
            if isinstance(candidate, list):
                holes = candidate
                break
    if not isinstance(holes, list) or not holes:
        holes = []
        for key, value in stages.items():
            if str(key).startswith("hole_") and isinstance(value, dict):
                holes.append(value)

    filtered: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in holes:
        if not isinstance(item, dict):
            continue
        hole_id = _safe_hole_id(item.get("hole_id"))
        if hole_id is None or hole_id in seen:
            continue
        seen.add(hole_id)
        filtered.append(item)
    return filtered


def _point(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        return None
    return [_round(item, 3) for item in values]


def _hole_source(result: dict[str, Any]) -> str:
    if result.get("pointcloud_center_fallback"):
        return "pointcloud_center_fallback"
    if result.get("status") == "capture_only":
        return "coarse_direct_capture_only"
    if result.get("coarse_direct_decision") in {
        "coarse_direct", "pointcloud_direct",
    }:
        return "coarse_direct_pointcloud"
    fine_source = str(result.get("batch_fine_source", ""))
    if fine_source.startswith((
        "batch_fine_clustered_supplement",
        "batch_fine_in_group_pose_adjustment",
    )):
        return "shared_supplement"
    if fine_source.startswith("batch_fine"):
        return "shared_first_capture"
    if fine_source == "per_hole_fine" or result.get("batch_fine_fallback_from_shared"):
        return "per_hole_fallback"
    return fine_source or str(result.get("coarse_source", "unknown"))


def _group_specs(report: dict[str, Any], stage_key: str) -> list[dict[str, Any]]:
    stage = (report.get("stages") or {}).get(stage_key) or {}
    groups = stage.get("groups") if isinstance(stage, dict) else None
    return [item for item in groups or [] if isinstance(item, dict)]


def _group_time(
    events: list[dict[str, Any]],
    prefix: str,
    *,
    exclude_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    selected = [
        event for event in events
        if str(event.get("name", "")).startswith(prefix)
        and not any(str(event.get("name", "")).startswith(item) for item in exclude_prefixes)
    ]
    return _window_for_events(selected)


def _shared_allocations(
    report: dict[str, Any],
    events: list[dict[str, Any]],
) -> tuple[dict[int, float], list[dict[str, Any]]]:
    allocations: dict[int, float] = {}
    groups: list[dict[str, Any]] = []

    def allocate(stage: str, group_index: int, hole_ids: list[int], timing: dict[str, Any], label: str) -> None:
        valid_ids = [int(item) for item in hole_ids]
        if not valid_ids:
            return
        visual_total = float(timing.get("total_visual_s", 0.0))
        # 旧报告没有 visual_timing 时，用原视觉分类字段回退；绝不回退到
        # production_cycle_s，因为其中包含机械臂运动和安全等待。
        if visual_total <= 0.0:
            visual_total = float(timing.get("vision_compute_s", 0.0))
        share = visual_total / len(valid_ids)
        for hole_id in valid_ids:
            allocations[hole_id] = allocations.get(hole_id, 0.0) + share
        groups.append({
            "stage": stage,
            "group_index": int(group_index),
            "label": label,
            "hole_ids": valid_ids,
            "production_cycle_s": _round(timing.get("production_cycle_s", 0.0)),
            "robot_motion_s": _round(timing.get("robot_motion_s", 0.0)),
            "vision_compute_s": _round(timing.get("vision_compute_s", 0.0)),
            "visual_total_s": _round(visual_total),
            "safety_wait_s": _round(timing.get("safety_wait_s", 0.0)),
            "allocated_per_hole_s": _round(share),
            "allocated_per_hole_visual_s": _round(share),
        })

    for group in _group_specs(report, "batch_coarse_plan"):
        index = int(group.get("group_index", len(groups) + 1))
        prefix = f"batch_coarse/group_{index:02d}/"
        timing = _group_time(events, prefix)
        timing["visual_timing"] = summarize_visual_timing_events(
            list(timing.get("events") or [])
        )
        timing.update(timing["visual_timing"])
        allocate("coarse", index, list(group.get("hole_ids") or []), timing, "coarse_shared")

    fine_groups = _group_specs(report, "batch_fine_plan")
    fine_results = _group_specs(report, "batch_fine_results")
    result_by_index = {
        int(item.get("group_index")): item for item in fine_results
        if item.get("group_index") is not None
    }
    for group in fine_groups:
        index = int(group.get("group_index", len(groups) + 1))
        prefix = f"batch_fine/group_{index:02d}/"
        supplement_prefix = f"{prefix}supplement_"
        timing = _group_time(events, prefix, exclude_prefixes=(supplement_prefix,))
        timing["visual_timing"] = summarize_visual_timing_events(
            list(timing.get("events") or [])
        )
        timing.update(timing["visual_timing"])
        allocate("fine", index, list(group.get("hole_ids") or []), timing, "fine_shared_first")
        result_group = result_by_index.get(index, group)
        for supplement in result_group.get("supplement_captures") or []:
            if not isinstance(supplement, dict):
                continue
            round_index = int(supplement.get("round", 1))
            cluster_index = int(supplement.get("cluster_index", 1))
            supplement_path = (
                f"{prefix}supplement_{round_index:02d}_cluster_{cluster_index:02d}/"
            )
            timing = _group_time(events, supplement_path)
            timing["visual_timing"] = summarize_visual_timing_events(
                list(timing.get("events") or [])
            )
            timing.update(timing["visual_timing"])
            allocate(
                "fine",
                index,
                list(supplement.get("hole_ids") or []),
                timing,
                f"fine_shared_supplement_{round_index:02d}_{cluster_index:02d}",
            )
    return allocations, groups


def build_result_summary(report: dict[str, Any]) -> dict[str, Any]:
    """从完整报告生成小而直接的结果摘要。"""
    timing = report.get("timing") or {}
    events = list(timing.get("events") or [])
    total_elapsed_s = _round(timing.get("total_elapsed_s", 0.0), 6)
    task_start = _marker_offset(timing, "cycle/task_start")
    selection_confirmed = _marker_offset(timing, "cycle/selection_confirmed")
    execution_start = _marker_offset(timing, "cycle/automatic_execution_start")
    cycle_end = _marker_offset(timing, "cycle/complete")
    cycle_completed_marker = cycle_end is not None
    if task_start is None:
        task_start = 0.0
    if cycle_end is None:
        interval_values = [_interval(event) for event in events]
        interval_values = [item for item in interval_values if item is not None]
        cycle_end = max((item[1] for item in interval_values), default=total_elapsed_s)
    if selection_confirmed is None:
        selection_confirmed = task_start
    if execution_start is None:
        execution_start = selection_confirmed

    task_timing = summarize_timing_window(events, start_s=task_start, end_s=cycle_end)
    execution_timing = summarize_timing_window(events, start_s=execution_start, end_s=cycle_end)
    session_timing = summarize_timing_window(events, start_s=0.0, end_s=total_elapsed_s)
    visual_task_timing = summarize_visual_timing_window(
        events, start_s=task_start, end_s=cycle_end,
    )
    visual_execution_timing = summarize_visual_timing_window(
        events, start_s=execution_start, end_s=cycle_end,
    )
    visual_session_timing = summarize_visual_timing_window(
        events, start_s=0.0, end_s=total_elapsed_s,
    )
    motion_events = list(timing.get("motion_events") or [])
    if not motion_events:
        # 兼容后续可能把距离字段直接写进 timing.events 的报告格式。
        motion_events = [
            event for event in events
            if isinstance(event, dict)
            and (
                "path_distance_mm" in event
                or "start_xyz_mm" in event
                or str(event.get("motion_type", "")).strip()
            )
        ]
    existing_motion_distance = timing.get("motion_distance")
    motion_distance = (
        dict(existing_motion_distance)
        if not motion_events and isinstance(existing_motion_distance, dict)
        else summarize_motion_distance(motion_events)
    )

    holes = _hole_results(report)
    planned_hole_ids = _planned_hole_ids(report)
    shared_allocations, groups = _shared_allocations(report, events)
    hole_rows: list[dict[str, Any]] = []
    for result in sorted(holes, key=lambda item: int(item.get("hole_id", 0))):
        hole_id = int(result.get("hole_id", 0))
        hole_events = list((result.get("timing") or {}).get("events") or [])
        hole_motion_events = list((result.get("timing") or {}).get("motion_events") or [])
        hole_timing = _window_for_events(hole_events)
        hole_visual_timing = summarize_visual_timing_events(hole_events)
        hole_motion_distance = summarize_motion_distance(hole_motion_events)
        exclusive_visual_s = float(hole_visual_timing.get("total_visual_s", 0.0))
        allocated_s = float(shared_allocations.get(hole_id, 0.0))
        final_point = (
            _point(result.get("target_point_base_mm"))
            or _point(result.get("hole_center_base_mm"))
            or _point(result.get("final_point_base_mm"))
        )
        planned_final = result.get("planned_final_point") or {}
        hole_rows.append({
            "hole_id": hole_id,
            "status": str(result.get("status", "unknown")),
            "source": _hole_source(result),
            "fine_quality_status": result.get("fine_quality_status"),
            "final_point_base_mm": final_point,
            "planned_final_tcp_xyz_mm": _point(
                planned_final.get("planned_final_tcp_xyz_mm")
            ),
            "planned_final_tcp_pose_m_rad": planned_final.get(
                "planned_final_tcp_pose_m_rad"
            ),
            "exclusive_visual_s": _round(exclusive_visual_s),
            "shared_visual_s": _round(allocated_s),
            "attributed_visual_s": _round(exclusive_visual_s + allocated_s),
            # 兼容旧 GUI/脚本字段，但语义改为“视觉耗时”，不再包含机械臂运动。
            "exclusive_time_s": _round(exclusive_visual_s),
            "shared_allocated_time_s": _round(allocated_s),
            "attributed_total_s": _round(exclusive_visual_s + allocated_s),
            "visual_timing": hole_visual_timing,
            "motion_distance": hole_motion_distance,
            "batch_fine_group_index": result.get("batch_fine_group_index"),
            "fallback_reason": result.get("batch_fine_fallback_reason"),
            "valid_frames": result.get("valid_frames"),
            "error": result.get("error"),
        })

    completed = [item for item in hole_rows if item["status"] == "completed"]
    capture_only = [item for item in hole_rows if item["status"] == "capture_only"]
    fallback = [item for item in hole_rows if item["source"] == "per_hole_fallback"]
    processed_hole_ids = [int(item["hole_id"]) for item in hole_rows]
    processed_hole_id_set = set(processed_hole_ids)
    if not planned_hole_ids:
        planned_hole_ids = processed_hole_ids.copy()
    try:
        selected_hole_count = int(report.get("hole_count") or 0)
    except (TypeError, ValueError):
        selected_hole_count = 0
    selected_hole_count = max(
        selected_hole_count,
        len(planned_hole_ids),
        len(processed_hole_ids),
    )
    unprocessed_hole_ids = [
        hole_id for hole_id in planned_hole_ids
        if hole_id not in processed_hole_id_set
    ]
    failed_hole_ids = [
        int(item["hole_id"])
        for item in hole_rows
        if item["status"] not in {"completed", "capture_only"}
    ]
    summary_status = str(report.get("status", "unknown"))
    measured_holes = len(completed) + len(capture_only)
    cycle_status = (
        "capture_only"
        if capture_only and not completed and not failed_hole_ids
        else "completed"
        if summary_status != "failed"
        and (
            cycle_completed_marker
            or (bool(holes) and measured_holes == len(hole_rows))
        )
        else summary_status
    )
    return {
        "schema_version": 1,
        "run_dir": str(report.get("run_dir", "")),
        "created_at": report.get("created_at"),
        "cycle_index": report.get("cycle_index"),
        "status": {
            "cycle_status": cycle_status,
            "session_status": summary_status,
            "session_end_reason": report.get("session_end_reason"),
            "selected_holes": selected_hole_count,
            "selected_hole_ids": planned_hole_ids,
            "processed_holes": len(processed_hole_ids),
            "completed_holes": len(completed),
            "capture_only_holes": len(capture_only),
            "deferred_holes": max(0, selected_hole_count - measured_holes),
            "unprocessed_holes": unprocessed_hole_ids,
            "failed_holes": failed_hole_ids,
            "shared_fine_holes": sum(
                item["source"] in {"shared_first_capture", "shared_supplement"}
                for item in hole_rows
            ),
            "per_hole_fallback_holes": [item["hole_id"] for item in fallback],
        },
        "timing": {
            "scope": "vision_only",
            "excluded_from_display": [
                "robot_motion", "safety_wait", "operator_wait", "artifact_io",
            ],
            "session_wall_s": _round(total_elapsed_s),
            "task_wall_s": _round(task_timing.get("wall_elapsed_s")),
            "production_cycle_s": _round(task_timing.get("production_cycle_s")),
            "pure_execution_s": _round(task_timing.get("pure_execution_s")),
            "automatic_execution_wall_s": _round(execution_timing.get("wall_elapsed_s")),
            "operator_wait_s": _round(task_timing.get("operator_wait_s")),
            "safety_wait_s": _round(task_timing.get("safety_wait_s")),
            "artifact_io_s": _round(task_timing.get("artifact_io_s")),
            "unmeasured_gap_s": _round(task_timing.get("unmeasured_gap_s")),
            "post_cycle_wall_s": _round(max(0.0, total_elapsed_s - float(cycle_end))),
            "session_breakdown": session_timing,
            "task_breakdown": task_timing,
            "execution_breakdown": execution_timing,
            "visual_total_s": _round(visual_task_timing.get("total_visual_s")),
            "visual_execution_s": _round(visual_execution_timing.get("total_visual_s")),
            "visual_session_s": _round(visual_session_timing.get("total_visual_s")),
            "visual_timing": visual_task_timing,
            "visual_execution_timing": visual_execution_timing,
            "visual_session_timing": visual_session_timing,
            "motion_distance": motion_distance,
        },
        "motion_distance": motion_distance,
        "groups": groups,
        "holes": hole_rows,
    }


def _format_seconds(value: Any) -> str:
    return f"{_round(value, 2):.2f} s"


def _format_distance_mm(value: Any) -> str:
    distance_mm = _round(value, 1)
    if abs(distance_mm) >= 1000.0:
        return f"{distance_mm:.1f} mm ({distance_mm / 1000.0:.3f} m)"
    return f"{distance_mm:.1f} mm"


def render_result_summary_text(summary: dict[str, Any]) -> str:
    status = summary.get("status") or {}
    timing = summary.get("timing") or {}
    visual_timing = timing.get("visual_timing") or {}
    if not visual_timing:
        # 兼容旧版 result_summary.json；vision_compute 已经排除了人工/安全
        # 等待，但这里不再显示 production_cycle 或 robot_motion。
        visual_timing = {
            "total_visual_s": timing.get(
                "visual_total_s",
                (timing.get("task_breakdown") or {}).get("vision_compute_s", 0.0),
            ),
            "stage_breakdown": {},
        }
    visual_total_s = float(visual_timing.get("total_visual_s", 0.0))
    visual_session_timing = timing.get("visual_session_timing") or {}
    visual_session_s = float(
        timing.get("visual_session_s", visual_session_timing.get("total_visual_s", 0.0))
    )
    motion_distance = timing.get("motion_distance") or summary.get("motion_distance") or {}
    motion_basis = str(
        motion_distance.get("distance_basis", "planned_tcp_waypoint_delta")
    )
    average_cycle_s = visual_total_s / max(
        1, int(status.get("selected_holes", 0))
    )
    stage_lines = []
    for stage in VISUAL_TIMING_STAGES:
        value = float(visual_timing.get(f"{stage}_s", 0.0))
        if value <= 0.0:
            value = float((visual_timing.get("stage_breakdown") or {}).get(stage, 0.0))
        if value > 0.0:
            stage_lines.append(
                f"  {_VISUAL_STAGE_LABELS.get(stage, stage)}：{_format_seconds(value)}"
            )
    lines = [
        "孔洞定位结果",
        f"本轮状态：{status.get('cycle_status', 'unknown')}",
        f"会话结束：{status.get('session_end_reason') or status.get('session_status') or 'unknown'}",
        (
            f"孔数：{status.get('selected_holes', 0)} | "
            f"成功：{status.get('completed_holes', 0)} | "
            f"仅采集：{status.get('capture_only_holes', 0)} | "
            f"延后/失败：{status.get('deferred_holes', 0)}"
        ),
        f"共享精定位：{status.get('shared_fine_holes', 0)}孔",
        f"逐孔回退：{status.get('per_hole_fallback_holes') or '无'}",
        "",
        f"视觉生产节拍（不含机械臂运动/等待）：{_format_seconds(visual_total_s)}",
        f"视觉会话总耗时（含相机启动）：{_format_seconds(visual_session_s)}",
        f"平均视觉耗时：{_format_seconds(average_cycle_s)}/孔",
        "视觉阶段拆分：",
        *stage_lines,
        "",
        "孔位视觉耗时（孔自身 + 共享组分摊）：",
        "孔号  状态      来源                  孔自身视觉(s)  共享视觉(s)  合计(s)",
    ]
    if status.get("unprocessed_holes"):
        lines.insert(6, f"未处理孔号：{status['unprocessed_holes']}")
    if status.get("failed_holes"):
        lines.insert(7, f"异常/延后孔号：{status['failed_holes']}")
    motion_count = int(motion_distance.get("segment_count", 0) or 0)
    if motion_count > 0:
        motion_lines = [
            "机械臂运动距离（不计入视觉耗时）：",
            f"  总路径：{_format_distance_mm(motion_distance.get('total_path_mm', 0.0))}    "
            f"水平：{_format_distance_mm(motion_distance.get('horizontal_distance_mm', 0.0))}",
            f"  上升：{_format_distance_mm(motion_distance.get('vertical_up_mm', 0.0))}    "
            f"下降：{_format_distance_mm(motion_distance.get('vertical_down_mm', 0.0))}    "
            f"净Z：{_format_distance_mm(motion_distance.get('vertical_net_mm', 0.0))}",
            f"  统计段数：{motion_count}    口径：{motion_basis}",
            "",
        ]
        marker_index = lines.index("孔位视觉耗时（孔自身 + 共享组分摊）：")
        lines[marker_index:marker_index] = motion_lines
    else:
        marker = "孔位视觉耗时（孔自身 + 共享组分摊）："
        lines.insert(lines.index(marker), "机械臂运动距离：暂无可用的运动端点记录（旧报告可能未启用此统计）。")
        lines.insert(lines.index(marker), "")
    for hole in summary.get("holes") or []:
        lines.append(
            f"H{int(hole.get('hole_id', 0)):02d}   "
            f"{str(hole.get('status', 'unknown')):<9} "
            f"{str(hole.get('source', 'unknown')):<21} "
            f"{float(hole.get('exclusive_visual_s', hole.get('exclusive_time_s', 0.0))):>11.2f}  "
            f"{float(hole.get('shared_visual_s', hole.get('shared_allocated_time_s', 0.0))):>9.2f}  "
            f"{float(hole.get('attributed_visual_s', hole.get('attributed_total_s', 0.0))):>7.2f}"
        )
    planned_holes = [
        hole for hole in summary.get("holes") or []
        if hole.get("planned_final_tcp_xyz_mm") is not None
    ]
    if planned_holes:
        lines.extend(["", "计划最终点（基坐标 mm；计划值不代表实际到位）："])
        for hole in planned_holes:
            visual = hole.get("final_point_base_mm")
            tcp = hole["planned_final_tcp_xyz_mm"]
            lines.append(
                f"H{int(hole.get('hole_id', 0)):02d} "
                f"孔中心={visual}，计划TCP={tcp}，状态={hole.get('status', 'unknown')}"
            )
    return "\n".join(lines) + "\n"


def build_progress_checkpoint(report: dict[str, Any]) -> dict[str, Any]:
    stages = report.get("stages") or {}
    holes: list[dict[str, Any]] = []
    for stage_name in ("processed_holes", "sequential_holes"):
        stage = stages.get(stage_name) or {}
        candidate = stage.get("holes") if isinstance(stage, dict) else None
        if isinstance(candidate, list):
            holes = [
                item for item in candidate
                if isinstance(item, dict) and _safe_hole_id(item.get("hole_id")) is not None
            ]
            break
    if not holes:
        final_result = report.get("final_result") or {}
        candidate = final_result.get("holes") if isinstance(final_result, dict) else None
        if isinstance(candidate, list):
            holes = [
                item for item in candidate
                if isinstance(item, dict) and _safe_hole_id(item.get("hole_id")) is not None
            ]

    processed_hole_ids: list[int] = []
    for item in holes:
        hole_id = _safe_hole_id(item.get("hole_id"))
        if hole_id is not None and hole_id not in processed_hole_ids:
            processed_hole_ids.append(hole_id)
    planned_hole_ids = _planned_hole_ids(report)
    if not planned_hole_ids:
        planned_hole_ids = processed_hole_ids.copy()
    try:
        selected_hole_count = int(report.get("hole_count") or 0)
    except (TypeError, ValueError):
        selected_hole_count = 0
    selected_hole_count = max(
        selected_hole_count,
        len(planned_hole_ids),
        len(processed_hole_ids),
    )
    processed_set = set(processed_hole_ids)
    active_hole = stages.get("active_hole") or {}
    active_hole_id = (
        _safe_hole_id(active_hole.get("hole_id"))
        if isinstance(active_hole, dict) else None
    )
    completed = [
        item for item in holes
        if item.get("status") == "completed"
    ]
    capture_only = [
        item for item in holes
        if item.get("status") == "capture_only"
    ]
    failed_hole_ids = [
        hole_id for item in holes
        if item.get("status") not in {"completed", "capture_only"}
        for hole_id in [_safe_hole_id(item.get("hole_id"))]
        if hole_id is not None
    ]
    active_stage = stages.get("active_stage")
    return {
        "schema_version": 1,
        "run_dir": str(report.get("run_dir", "")),
        "cycle_index": report.get("cycle_index"),
        "status": report.get("status"),
        "hole_count": selected_hole_count or report.get("hole_count"),
        "planned_hole_ids": planned_hole_ids,
        "processed_count": len(processed_hole_ids),
        "completed_count": len(completed),
        "capture_only_count": len(capture_only),
        "failed_hole_ids": failed_hole_ids,
        "processed_hole_ids": processed_hole_ids,
        "remaining_hole_ids": [
            hole_id for hole_id in planned_hole_ids
            if hole_id not in processed_set
        ],
        "current_hole_id": active_hole_id,
        "active_stage": (
            dict(active_stage) if isinstance(active_stage, dict) else None
        ),
        "last_hole_id": processed_hole_ids[-1] if processed_hole_ids else None,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_progress_checkpoint(
    run_dir: Path,
    report: dict[str, Any],
    *,
    timing: Any | None = None,
) -> Path:
    path = Path(run_dir) / "progress.json"
    payload = build_progress_checkpoint(report)
    with artifact_measure(
        timing,
        "report/write_progress_json",
        artifact_kind="progress_json",
        paths=[str(path)],
    ):
        atomic_write_json(path, jsonable(payload))
    return path


def write_result_summary(
    run_dir: Path,
    report: dict[str, Any],
    *,
    timing: Any | None = None,
) -> dict[str, str]:
    """在一轮结束时写入轻量 JSON/TXT 摘要。"""
    summary = build_result_summary(report)
    json_path = Path(run_dir) / "result_summary.json"
    text_path = Path(run_dir) / "result_summary.txt"
    with artifact_measure(
        timing,
        "report/write_result_summary_json",
        artifact_kind="result_summary_json",
        paths=[str(json_path)],
    ):
        atomic_write_json(json_path, jsonable(summary))
    with artifact_measure(
        timing,
        "report/write_result_summary_text",
        artifact_kind="result_summary_text",
        paths=[str(text_path)],
    ):
        _atomic_write_text(text_path, render_result_summary_text(summary))
    return {"json": str(json_path), "text": str(text_path)}
