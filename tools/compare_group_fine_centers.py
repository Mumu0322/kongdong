"""Compare paired group-fine center diagnostics against a per-hole run.

This is an offline diagnostic. It does not alter the robot target or imply that
two runs used the same fixture registration; verify that before interpreting the
coordinate differences as localization error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


FIELDS = (
    "group", "round", "hole_id", "capture_frames", "selected_by_runtime",
    "reference_x_mm", "reference_y_mm",
    "yolo_x_mm", "yolo_y_mm", "geometric_x_mm", "geometric_y_mm",
    "runtime_final_x_mm", "runtime_final_y_mm",
    "yolo_minus_reference_xy_mm", "geometric_minus_reference_xy_mm",
    "runtime_final_minus_reference_xy_mm",
    "yolo_minus_geometric_xy_mm",
)


def _point_xy(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError("Missing XY point")
    xy = float(value[0]), float(value[1])
    if not all(math.isfinite(item) for item in xy):
        raise ValueError("Non-finite XY point")
    return xy


def build_rows(candidate: dict[str, Any], reference: dict[str, Any]) -> list[dict[str, Any]]:
    reference_holes = {
        int(hole["hole_id"]): _point_xy(hole["hole_center_base_mm"])
        for hole in reference["final_result"]["holes"]
    }
    runtime_holes = {
        int(hole["hole_id"]): _point_xy(hole["hole_center_base_mm"])
        for hole in candidate.get("final_result", {}).get("holes", [])
        if hole.get("hole_center_base_mm") is not None
    }
    rows: list[dict[str, Any]] = []
    groups = candidate["stages"]["batch_fine_results"]["groups"]
    for group in groups:
        captures = [(0, group)] + [
            (int(item["round"]), item)
            for item in group.get("supplement_captures", [])
        ]
        for capture_round, capture in captures:
            paired: dict[int, list[dict[str, Any]]] = {}
            for frame in capture.get("capture", {}).get("frame_records", []):
                for hole in frame.get("holes", []):
                    if (
                        hole.get("valid") is True
                        and hole.get("paired_center_diagnostic_status") == "complete"
                    ):
                        paired.setdefault(int(hole["hole_id"]), []).append(hole)
            for hole_id, observations in sorted(paired.items()):
                if hole_id not in reference_holes:
                    continue
                yolo = tuple(median(_point_xy(item["yolo_point_base_mm"])[axis]
                                    for item in observations) for axis in (0, 1))
                geometric = tuple(median(_point_xy(item["geometric_point_base_mm"])[axis]
                                         for item in observations) for axis in (0, 1))
                ref = reference_holes[hole_id]
                runtime_final = runtime_holes.get(hole_id)
                final_result = (group.get("results") or {}).get(str(hole_id), {})
                selected = bool(
                    final_result.get("success")
                    and int(final_result.get("capture_round", -1)) == capture_round
                    and hole_id in [int(value) for value in capture.get("hole_ids", [])]
                )
                rows.append({
                    "group": int(group["group_index"]),
                    "round": capture_round,
                    "hole_id": hole_id,
                    "capture_frames": len(observations),
                    "selected_by_runtime": selected,
                    "reference_x_mm": ref[0], "reference_y_mm": ref[1],
                    "yolo_x_mm": yolo[0], "yolo_y_mm": yolo[1],
                    "geometric_x_mm": geometric[0], "geometric_y_mm": geometric[1],
                    "runtime_final_x_mm": (
                        None if runtime_final is None else runtime_final[0]
                    ),
                    "runtime_final_y_mm": (
                        None if runtime_final is None else runtime_final[1]
                    ),
                    "yolo_minus_reference_xy_mm": math.dist(yolo, ref),
                    "geometric_minus_reference_xy_mm": math.dist(geometric, ref),
                    "runtime_final_minus_reference_xy_mm": (
                        None if runtime_final is None else math.dist(runtime_final, ref)
                    ),
                    "yolo_minus_geometric_xy_mm": math.dist(yolo, geometric),
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_report", type=Path)
    parser.add_argument("reference_report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    candidate = json.loads(args.candidate_report.read_text(encoding="utf-8"))
    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    rows = build_rows(candidate, reference)
    if not rows:
        raise SystemExit(
            "No paired group-fine diagnostics found. Capture a new run with the updated code."
        )
    output = args.output or args.candidate_report.parent / "paired_center_comparison.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    selected = [row for row in rows if row["selected_by_runtime"]]
    print(f"Wrote {len(rows)} capture/hole comparisons to {output}")
    print(f"Runtime-selected group captures: {len(selected)}")
    if selected:
        for source in ("yolo", "geometric"):
            values = [row[f"{source}_minus_reference_xy_mm"] for row in selected]
            print(
                f"{source}: median={median(values):.3f} mm, "
                f"max={max(values):.3f} mm, "
                f"within_0.25mm={sum(value <= 0.25 for value in values)}/{len(values)}"
            )
        runtime_values = [
            row["runtime_final_minus_reference_xy_mm"] for row in selected
            if row["runtime_final_minus_reference_xy_mm"] is not None
        ]
        if runtime_values:
            print(
                f"runtime final: median={median(runtime_values):.3f} mm, "
                f"max={max(runtime_values):.3f} mm, "
                f"within_0.25mm={sum(value <= 0.25 for value in runtime_values)}"
                f"/{len(runtime_values)}"
            )
    print("Diagnostic projection only; not the joint-fused or compensated final target.")


if __name__ == "__main__":
    main()
