#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""汇总多轮第四策略点云实验报告。

输入可以是一个或多个 ``report.json``，也可以是包含该文件的运行目录。
工具只读取报告，输出明细、分组明细和配置对照摘要，不读取或修改孔位地图。

示例（PowerShell）：

    python tools/summarize_coarse_direct_experiment.py `
        --label A=C:\runs\A_340mm_3holes `
        --label B=C:\runs\B_320mm_3holes `
        --label C=C:\runs\C_320mm_1hole `
        --output-dir C:\runs\summary

也可以直接传运行目录；没有显式标签时，工具按报告中的高度、分组上限和
策略自动生成标签，例如 ``direct_320mm_3holes``。
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _vector3(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    values = [_number(item) for item in value[:3]]
    if any(item is None for item in values):
        return None
    return [float(item) for item in values if item is not None]


def _vector2(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    values = [_number(item) for item in value[:2]]
    if any(item is None for item in values):
        return None
    return [float(item) for item in values if item is not None]


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _stats(values: Iterable[Any]) -> dict[str, Any]:
    numbers = sorted(
        number for value in values
        if (number := _number(value)) is not None
    )
    if not numbers:
        return {
            "count": 0, "mean": None, "std": None, "min": None,
            "p50": None, "p95": None, "max": None,
        }
    mean = sum(numbers) / len(numbers)
    variance = sum((value - mean) ** 2 for value in numbers) / len(numbers)
    return {
        "count": len(numbers),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": numbers[0],
        "p50": numbers[(len(numbers) - 1) // 2],
        "p95": _p95(numbers),
        "max": numbers[-1],
    }


def _resolve_report_paths(path: Path) -> list[Path]:
    path = path.expanduser()
    if path.is_file():
        return [path.resolve()]
    if not path.is_dir():
        raise FileNotFoundError(f"报告路径不存在：{path}")
    direct = path / "report.json"
    if direct.is_file():
        return [direct.resolve()]
    reports = sorted(item.resolve() for item in path.rglob("report.json"))
    if not reports:
        raise FileNotFoundError(f"目录中没有找到 report.json：{path}")
    return reports


def _load_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"报告根节点必须是对象：{path}")
    return payload


def _first_list(report: dict[str, Any], candidates: list[tuple[str, ...]]) -> list[Any]:
    for path in candidates:
        current: Any = report
        for key in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if isinstance(current, list):
            return current
    return []


def _hole_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    raw = _first_list(report, [
        ("final_result", "holes"),
        ("stages", "sequential_holes", "holes"),
        ("stages", "processed_holes", "holes"),
    ])
    return [item for item in raw if isinstance(item, dict)]


def _group_rows(report: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    stages = report.get("stages") or {}
    plan = stages.get("batch_coarse_plan") or {}
    groups = plan.get("groups")
    if not isinstance(groups, list):
        groups = (stages.get("batch_coarse_results") or {}).get("groups") or []
    if not isinstance(groups, list):
        groups = []

    rows: list[dict[str, Any]] = []
    by_hole: dict[int, dict[str, Any]] = {}
    for fallback_index, group in enumerate(groups, start=1):
        if not isinstance(group, dict):
            continue
        group_index = _integer(group.get("group_index")) or fallback_index
        hole_ids: list[int] = []
        for value in group.get("hole_ids") or []:
            hole_id = _integer(value)
            if hole_id is not None:
                hole_ids.append(hole_id)
        capture = group.get("capture") or {}
        if not isinstance(capture, dict):
            capture = {}
        row = {
            "group_index": group_index,
            "hole_ids": hole_ids,
            "hole_count": _integer(group.get("hole_count")) or len(hole_ids),
            "boundary_class": group.get("boundary_class"),
            "group_phase": group.get("group_phase"),
            "aspect_ratio_gate_disabled_for_edge": bool(
                group.get("aspect_ratio_gate_disabled_for_edge", False)
            ),
            "aspect_ratio_gate_applied_for_edge": bool(
                group.get("aspect_ratio_gate_applied_for_edge", False)
            ),
            "boundary_component_indices": group.get("boundary_component_indices") or [],
            "boundary_holes": group.get("boundary_holes") or [],
            "target_height_mm": _number(
                group.get("target_height_mm", plan.get("target_height_mm"))
            ),
            "accepted_holes": [
                value for value in (
                    _integer(item) for item in group.get("accepted_holes") or []
                ) if value is not None
            ],
            "fallback_holes": [
                value for value in (
                    _integer(item) for item in group.get("fallback_holes") or []
                ) if value is not None
            ],
            "captured_frame_count": _integer(capture.get("captured_frame_count")),
            "early_stop_min_frames": _integer(capture.get("early_stop_min_frames")),
            "capture_stop_reason": capture.get("capture_stop_reason"),
            "failure_type": group.get("failure_type"),
            "error": group.get("error"),
        }
        rows.append(row)
        for position, hole_id in enumerate(hole_ids, start=1):
            by_hole.setdefault(hole_id, {
                "group_index": group_index,
                "group_hole_count": len(hole_ids),
                "group_position": position,
                "group_hole_ids": list(hole_ids),
            })
    return rows, by_hole


def _config_info(
    report: dict[str, Any],
    holes: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    label_override: str | None,
    report_path: Path,
) -> dict[str, Any]:
    configuration = report.get("configuration") or {}
    if not isinstance(configuration, dict):
        configuration = {}
    stages = report.get("stages") or {}
    plan = stages.get("batch_coarse_plan") or {}
    direct = bool(
        configuration.get("coarse_direct_final", False)
        or str(report.get("localization_strategy", "")).startswith("coarse_")
        or plan.get("strategy") == "coarse_direct_pointcloud_only"
    )
    height = (
        configuration.get("coarse_direct_final_height_mm") if direct else None
    )
    if _number(height) is None:
        height = configuration.get("coarse_height_mm")
    if _number(height) is None:
        height = plan.get("target_height_mm")
    if _number(height) is None:
        height = next((item.get("coarse_capture_height_mm") for item in holes), None)
    height = _number(height)

    max_group = (
        configuration.get("coarse_direct_final_max_group_size") if direct else None
    )
    if _integer(max_group) is None:
        max_group = configuration.get("batch_coarse_max_group_size")
    if _integer(max_group) is None:
        max_group = plan.get("max_group_size")
    max_group = _integer(max_group)

    derived_label = (
        f"{'direct' if direct else 'shared'}_"
        f"{height:g}mm_{max_group or '?'}holes"
        if height is not None else
        str(report.get("localization_strategy") or "unknown")
    )
    label = str(label_override or derived_label)
    selected_count = (
        _integer((stages.get("sequential_plan") or {}).get("hole_count"))
        or _integer(report.get("hole_count"))
        or len(holes)
    )
    return {
        "label": label,
        "report_path": str(report_path),
        "report_status": report.get("status"),
        "direct_strategy": direct,
        "height_mm": height,
        "max_group_size": max_group,
        "selected_holes": selected_count,
        "group_count": len(groups),
    }


def _hole_row(
    report: dict[str, Any],
    config: dict[str, Any],
    hole: dict[str, Any],
    group_by_hole: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    hole_id = _integer(hole.get("hole_id"))
    group = group_by_hole.get(hole_id or -1, {})
    point = _vector3(
        hole.get("pointcloud_center_base_mm")
        or hole.get("coarse_center_base_mm")
        or hole.get("hole_center_base_mm")
    )
    target = _vector3(hole.get("target_point_base_mm"))
    correction = _vector2(hole.get("charuco_xy_correction_mm"))
    status = str(hole.get("status") or "unknown")
    measured = status in {"completed", "capture_only"}
    return {
        "config": config["label"],
        "report_path": config["report_path"],
        "report_status": config["report_status"],
        "hole_id": hole_id,
        "status": status,
        "measured": measured,
        "capture_only": status == "capture_only",
        "group_index": _integer(hole.get("batch_coarse_group_index"))
        or group.get("group_index"),
        "group_hole_count": group.get("group_hole_count"),
        "group_position": group.get("group_position"),
        "group_hole_ids": hole.get("batch_coarse_group_hole_ids")
        or group.get("group_hole_ids", []),
        "boundary_class": hole.get("boundary_class") or group.get("boundary_class"),
        "boundary_layer": _integer(hole.get("boundary_layer")),
        "boundary_component_index": _integer(hole.get("boundary_component_index")),
        "boundary_distance": _number(hole.get("boundary_distance")),
        "boundary_distance_mm": _number(hole.get("boundary_distance_mm")),
        "boundary_distance_px": _number(hole.get("boundary_distance_px")),
        "boundary_distance_unit": hole.get("boundary_distance_unit"),
        "boundary_edge_score_deg": _number(hole.get("boundary_edge_score_deg")),
        "boundary_local_degree": _integer(hole.get("boundary_local_degree")),
        "boundary_classification_reason": hole.get("boundary_classification_reason"),
        "boundary_override": bool(hole.get("boundary_override", False)),
        "boundary_override_source": hole.get("boundary_override_source"),
        "group_phase": hole.get("group_phase") or group.get("group_phase"),
        "capture_height_mm": _number(
            hole.get("coarse_capture_height_mm") or config.get("height_mm")
        ),
        "valid_frames": _integer(
            hole.get("coarse_valid_frames") or hole.get("valid_frames")
        ),
        "total_frames": _integer(
            hole.get("coarse_total_frames") or hole.get("total_frames")
        ),
        "plane_rmse_mm": _number(hole.get("coarse_plane_rmse_mm")),
        "center_scatter_p95_px": _number(hole.get("coarse_center_scatter_p95_px")),
        "tracking_distance_p95_px": _number(
            hole.get("coarse_tracking_distance_p95_px")
        ),
        "center_source": hole.get("coarse_center_source"),
        "coarse_source": hole.get("coarse_source"),
        "coarse_direct_decision": hole.get("coarse_direct_decision"),
        "pointcloud_center_base_mm": point,
        "target_point_base_mm": target,
        "charuco_compensation_applied": bool(
            hole.get("charuco_compensation_applied", False)
        ),
        "charuco_xy_correction_mm": correction,
        "estimated_height_mm": _number(hole.get("estimated_height_mm")),
        "failure_type": hole.get("failure_type"),
        "error": hole.get("error") or hole.get("deferred_reason"),
    }


def _repeatability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vectors = [row["pointcloud_center_base_mm"] for row in rows]
    vectors = [item for item in vectors if item is not None]
    if not vectors:
        return {"sample_count": 0, "mean_base_mm": None, "std_base_mm": None, "radial_p95_mm": None}
    mean = [sum(item[index] for item in vectors) / len(vectors) for index in range(3)]
    std = [
        math.sqrt(sum((item[index] - mean[index]) ** 2 for item in vectors) / len(vectors))
        for index in range(3)
    ]
    radial = [math.sqrt(sum((item[index] - mean[index]) ** 2 for index in range(3))) for item in vectors]
    return {
        "sample_count": len(vectors),
        "mean_base_mm": mean,
        "std_base_mm": std,
        "radial_p95_mm": _p95(radial),
    }


def _comparison(rows: list[dict[str, Any]], reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_config[str(row["config"])].append(row)
    output: list[dict[str, Any]] = []
    for label in sorted(by_config):
        config_rows = by_config[label]
        report_rows = [item for item in reports if item["label"] == label]
        selected = sum(int(item.get("selected_holes") or 0) for item in report_rows)
        measured = sum(bool(item["measured"]) for item in config_rows)
        completed = sum(item["status"] == "completed" for item in config_rows)
        capture_only = sum(item["capture_only"] for item in config_rows)
        deferred = max(0, selected - measured)
        per_hole: list[dict[str, Any]] = []
        by_hole: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in config_rows:
            if row["hole_id"] is not None:
                by_hole[int(row["hole_id"])].append(row)
        for hole_id in sorted(by_hole):
            samples = by_hole[hole_id]
            per_hole.append({
                "hole_id": hole_id,
                "sample_count": len(samples),
                "measured_count": sum(bool(item["measured"]) for item in samples),
                "group_indices": sorted({
                    int(item["group_index"])
                    for item in samples if item.get("group_index") is not None
                }),
                "repeatability": _repeatability(samples),
            })
        output.append({
            "config": label,
            "report_count": len(report_rows),
            "selected_holes": selected,
            "processed_holes": len(config_rows),
            "measured_holes": measured,
            "completed_holes": completed,
            "capture_only_holes": capture_only,
            "deferred_holes": deferred,
            "unprocessed_holes": max(0, selected - len(config_rows)),
            "measured_rate": measured / selected if selected else None,
            "completed_rate": completed / selected if selected else None,
            "valid_frames": _stats(item["valid_frames"] for item in config_rows),
            "total_frames": _stats(item["total_frames"] for item in config_rows),
            "plane_rmse_mm": _stats(item["plane_rmse_mm"] for item in config_rows),
            "center_scatter_p95_px": _stats(
                item["center_scatter_p95_px"] for item in config_rows
            ),
            "tracking_distance_p95_px": _stats(
                item["tracking_distance_p95_px"] for item in config_rows
            ),
            "group_count": len({
                (item["report_path"], item["group_index"])
                for item in config_rows if item.get("group_index") is not None
            }),
            "edge_holes": sum(item.get("boundary_class") == "edge" for item in config_rows),
            "interior_holes": sum(item.get("boundary_class") == "interior" for item in config_rows),
            "per_hole": per_hole,
        })
    return output


def summarize_reports(
    report_specs: Iterable[tuple[Path, str | None]],
) -> dict[str, Any]:
    """读取报告并生成可序列化的明细与对照数据。"""
    hole_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for input_path, label_override in report_specs:
        for report_path in _resolve_report_paths(Path(input_path)):
            if report_path in seen:
                continue
            seen.add(report_path)
            report = _load_report(report_path)
            holes = _hole_results(report)
            groups, group_by_hole = _group_rows(report)
            config = _config_info(
                report, holes, groups, label_override, report_path,
            )
            report_rows.append(config)
            for group in groups:
                group_rows.append({"config": config["label"], **group})
            hole_rows.extend(
                _hole_row(report, config, hole, group_by_hole) for hole in holes
            )

    return {
        "schema_version": 1,
        "tool": "summarize_coarse_direct_experiment",
        "source_report_count": len(report_rows),
        "reports": report_rows,
        "holes": hole_rows,
        "groups": group_rows,
        "comparison": _comparison(hole_rows, report_rows),
    }


HOLE_FIELDS = [
    "config", "report_path", "report_status", "hole_id", "status", "measured",
    "capture_only", "group_index", "group_hole_count", "group_position",
    "group_hole_ids", "capture_height_mm", "valid_frames", "total_frames",
    "plane_rmse_mm", "center_scatter_p95_px", "tracking_distance_p95_px",
    "boundary_class", "boundary_layer", "boundary_component_index",
    "boundary_distance", "boundary_distance_mm", "boundary_distance_px",
    "boundary_distance_unit", "boundary_edge_score_deg",
    "boundary_local_degree", "boundary_classification_reason", "group_phase",
    "boundary_override", "boundary_override_source",
    "center_source", "coarse_source", "coarse_direct_decision",
    "pointcloud_center_base_mm", "target_point_base_mm",
    "charuco_compensation_applied", "charuco_xy_correction_mm",
    "estimated_height_mm", "failure_type", "error",
]
GROUP_FIELDS = [
    "config", "group_index", "hole_ids", "hole_count", "boundary_class",
    "group_phase", "aspect_ratio_gate_disabled_for_edge",
    "aspect_ratio_gate_applied_for_edge",
    "boundary_component_indices", "boundary_holes", "target_height_mm",
    "accepted_holes", "fallback_holes", "captured_frame_count",
    "early_stop_min_frames", "capture_stop_reason", "failure_type", "error",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            materialized = {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(materialized)
    return path


def write_summary(summary: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "coarse_experiment_summary.json"
    holes_path = output_dir / "coarse_experiment_holes.csv"
    groups_path = output_dir / "coarse_experiment_groups.csv"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(holes_path, list(summary.get("holes") or []), HOLE_FIELDS)
    _write_csv(groups_path, list(summary.get("groups") or []), GROUP_FIELDS)
    return {
        "json": str(json_path.resolve()),
        "holes_csv": str(holes_path.resolve()),
        "groups_csv": str(groups_path.resolve()),
    }


def _parse_label(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--label 必须使用 NAME=PATH 格式：{value}")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise ValueError(f"--label 必须同时提供名称和路径：{value}")
    return name, Path(raw_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports", nargs="*", type=Path,
        help="report.json或运行目录；目录没有直接报告时会递归查找",
    )
    parser.add_argument(
        "--label", action="append", default=[], metavar="NAME=PATH",
        help="给报告或运行目录指定A/B/C配置标签，可重复使用",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path.cwd() / "coarse_experiment_summary",
        help="汇总输出目录，默认当前目录/coarse_experiment_summary",
    )
    args = parser.parse_args(argv)
    specs: list[tuple[Path, str | None]] = [(path, None) for path in args.reports]
    for raw_label in args.label:
        try:
            label, path = _parse_label(raw_label)
        except ValueError as exc:
            parser.error(str(exc))
        specs.append((path, label))
    if not specs:
        parser.error("至少提供一个报告路径或--label NAME=PATH")
    try:
        summary = summarize_reports(specs)
        paths = write_summary(summary, args.output_dir)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
