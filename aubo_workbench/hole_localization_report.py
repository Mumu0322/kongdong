"""Compact, non-ambiguous result reports for two-stage hole localization.

The existing ``report.json`` intentionally contains a large amount of diagnostic
data.  This module builds a small operator-facing summary from that report and
from the timing intervals.  Timing is calculated from the deepest active event
on a timeline, so a parent event and its nested capture event are not counted
twice.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from aubo_workbench.hole_localization_models import (
    artifact_measure,
    infer_timing_category,
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


def _hole_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    final_result = report.get("final_result") or {}
    holes = final_result.get("holes") if isinstance(final_result, dict) else None
    if not isinstance(holes, list):
        holes = (report.get("stages") or {}).get("sequential_holes", {}).get("holes")
    if not isinstance(holes, list):
        holes = []
        for key, value in (report.get("stages") or {}).items():
            if str(key).startswith("hole_") and isinstance(value, dict):
                holes.append(value)
    return [item for item in holes if isinstance(item, dict)]


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
    fine_source = str(result.get("batch_fine_source", ""))
    if fine_source.startswith("batch_fine_clustered_supplement"):
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
        share = float(timing.get("production_cycle_s", 0.0)) / len(valid_ids)
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
            "safety_wait_s": _round(timing.get("safety_wait_s", 0.0)),
            "allocated_per_hole_s": _round(share),
        })

    for group in _group_specs(report, "batch_coarse_plan"):
        index = int(group.get("group_index", len(groups) + 1))
        prefix = f"batch_coarse/group_{index:02d}/"
        timing = _group_time(events, prefix)
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

    holes = _hole_results(report)
    shared_allocations, groups = _shared_allocations(report, events)
    hole_rows: list[dict[str, Any]] = []
    for result in sorted(holes, key=lambda item: int(item.get("hole_id", 0))):
        hole_id = int(result.get("hole_id", 0))
        hole_events = list((result.get("timing") or {}).get("events") or [])
        hole_timing = _window_for_events(hole_events)
        exclusive_s = float(hole_timing.get("production_cycle_s", 0.0))
        allocated_s = float(shared_allocations.get(hole_id, 0.0))
        final_point = (
            _point(result.get("target_point_base_mm"))
            or _point(result.get("hole_center_base_mm"))
            or _point(result.get("final_point_base_mm"))
        )
        hole_rows.append({
            "hole_id": hole_id,
            "status": str(result.get("status", "unknown")),
            "source": _hole_source(result),
            "fine_quality_status": result.get("fine_quality_status"),
            "final_point_base_mm": final_point,
            "exclusive_time_s": _round(exclusive_s),
            "shared_allocated_time_s": _round(allocated_s),
            "attributed_total_s": _round(exclusive_s + allocated_s),
            "batch_fine_group_index": result.get("batch_fine_group_index"),
            "fallback_reason": result.get("batch_fine_fallback_reason"),
            "valid_frames": result.get("valid_frames"),
        })

    completed = [item for item in hole_rows if item["status"] == "completed"]
    fallback = [item for item in hole_rows if item["source"] == "per_hole_fallback"]
    summary_status = str(report.get("status", "unknown"))
    cycle_status = (
        "completed"
        if cycle_completed_marker
        or (bool(holes) and len(completed) == len(hole_rows) and summary_status != "failed")
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
            "selected_holes": len(hole_rows),
            "completed_holes": len(completed),
            "deferred_holes": len(hole_rows) - len(completed),
            "shared_fine_holes": sum(
                item["source"] in {"shared_first_capture", "shared_supplement"}
                for item in hole_rows
            ),
            "per_hole_fallback_holes": [item["hole_id"] for item in fallback],
        },
        "timing": {
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
        },
        "groups": groups,
        "holes": hole_rows,
    }


def _format_seconds(value: Any) -> str:
    return f"{_round(value, 2):.2f} s"


def render_result_summary_text(summary: dict[str, Any]) -> str:
    status = summary.get("status") or {}
    timing = summary.get("timing") or {}
    lines = [
        "孔洞定位结果",
        f"本轮状态：{status.get('cycle_status', 'unknown')}",
        f"会话结束：{status.get('session_end_reason') or status.get('session_status') or 'unknown'}",
        (
            f"孔数：{status.get('selected_holes', 0)} | "
            f"成功：{status.get('completed_holes', 0)} | "
            f"延后/失败：{status.get('deferred_holes', 0)}"
        ),
        f"共享精定位：{status.get('shared_fine_holes', 0)}孔",
        f"逐孔回退：{status.get('per_hole_fallback_holes') or '无'}",
        "",
        f"生产节拍：{_format_seconds(timing.get('production_cycle_s'))}",
        f"纯执行时间：{_format_seconds(timing.get('pure_execution_s'))}",
        f"平均生产节拍：{_format_seconds(
            float(timing.get('production_cycle_s', 0.0)) /
            max(1, int(status.get('selected_holes', 0)))
        )}/孔",
        f"机械臂运动：{_format_seconds((timing.get('task_breakdown') or {}).get('robot_motion_s'))}",
        f"视觉计算：{_format_seconds((timing.get('task_breakdown') or {}).get('vision_compute_s'))}",
        f"安全等待：{_format_seconds(timing.get('safety_wait_s'))}",
        f"人工等待：{_format_seconds(timing.get('operator_wait_s'))}",
        f"文件生成：{_format_seconds(timing.get('artifact_io_s'))}",
        "",
        "孔位耗时（孔自身 + 共享组分摊）：",
        "孔号  状态      来源                  孔自身(s)  共享分摊(s)  合计(s)",
    ]
    for hole in summary.get("holes") or []:
        lines.append(
            f"H{int(hole.get('hole_id', 0)):02d}   "
            f"{str(hole.get('status', 'unknown')):<9} "
            f"{str(hole.get('source', 'unknown')):<21} "
            f"{float(hole.get('exclusive_time_s', 0.0)):>8.2f}  "
            f"{float(hole.get('shared_allocated_time_s', 0.0)):>9.2f}  "
            f"{float(hole.get('attributed_total_s', 0.0)):>7.2f}"
        )
    return "\n".join(lines) + "\n"


def build_progress_checkpoint(report: dict[str, Any]) -> dict[str, Any]:
    stages = report.get("stages") or {}
    sequential = stages.get("sequential_holes") or {}
    holes = sequential.get("holes") if isinstance(sequential, dict) else []
    if not isinstance(holes, list):
        holes = []
    completed = [item for item in holes if isinstance(item, dict) and item.get("status") == "completed"]
    return {
        "schema_version": 1,
        "run_dir": str(report.get("run_dir", "")),
        "cycle_index": report.get("cycle_index"),
        "status": report.get("status"),
        "hole_count": report.get("hole_count"),
        "processed_count": len(holes),
        "completed_count": len(completed),
        "processed_hole_ids": [int(item.get("hole_id", 0)) for item in holes],
        "last_hole_id": int(holes[-1].get("hole_id", 0)) if holes else None,
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
