from __future__ import annotations

import json
from pathlib import Path

from tools.summarize_coarse_direct_experiment import (
    summarize_reports,
    write_summary,
)


def _hole(hole_id: int, status: str, x: float) -> dict[str, object]:
    row: dict[str, object] = {
        "hole_id": hole_id,
        "status": status,
        "coarse_capture_height_mm": 320.0,
        "coarse_valid_frames": 10,
        "coarse_total_frames": 15,
        "coarse_plane_rmse_mm": 0.4,
        "coarse_center_scatter_p95_px": 1.2,
        "coarse_tracking_distance_p95_px": 2.0,
        "pointcloud_center_base_mm": [x, 20.0, 100.0],
        "coarse_direct_decision": "capture_only" if status == "capture_only" else "pointcloud_direct",
        "charuco_compensation_applied": status == "completed",
        "charuco_xy_correction_mm": [0.2, -0.1] if status == "completed" else None,
    }
    if status.startswith("deferred"):
        row.update({"failure_type": "pointcloud_center_unavailable", "error": "no valid cloud"})
    return row


def _write_report(directory: Path, x_offset: float = 0.0) -> Path:
    directory.mkdir()
    report = {
        "status": "completed",
        "localization_strategy": "coarse_320_pointcloud_center_only",
        "configuration": {
            "coarse_direct_final": True,
            "coarse_height_mm": 320.0,
            "coarse_direct_final_height_mm": 320.0,
            "coarse_direct_final_max_group_size": 3,
        },
        "stages": {
            "sequential_plan": {"hole_count": 3},
            "batch_coarse_plan": {
                "target_height_mm": 320.0,
                "max_group_size": 3,
                "groups": [{
                    "group_index": 1,
                    "hole_ids": [1, 2, 3],
                    "hole_count": 3,
                    "target_height_mm": 320.0,
                    "accepted_holes": [1, 2],
                    "fallback_holes": [3],
                    "capture": {
                        "captured_frame_count": 15,
                        "early_stop_min_frames": 15,
                        "capture_stop_reason": "max_frames_reached",
                    },
                }],
            },
            "sequential_holes": {
                "holes": [
                    _hole(1, "capture_only", 10.0 + x_offset),
                    _hole(2, "completed", 20.0 + x_offset),
                    _hole(3, "deferred_coarse_pointcloud", 30.0 + x_offset),
                ],
            },
        },
    }
    path = directory / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_summary_keeps_group_position_and_capture_only_as_measured(tmp_path: Path) -> None:
    report_path = _write_report(tmp_path / "run_a")

    summary = summarize_reports([(report_path.parent, "B")])

    assert summary["source_report_count"] == 1
    assert [row["group_position"] for row in summary["holes"]] == [1, 2, 3]
    comparison = summary["comparison"][0]
    assert comparison["config"] == "B"
    assert comparison["measured_holes"] == 2
    assert comparison["capture_only_holes"] == 1
    assert comparison["deferred_holes"] == 1
    assert comparison["measured_rate"] == 2 / 3
    assert summary["holes"][1]["charuco_xy_correction_mm"] == [0.2, -0.1]


def test_summary_writes_json_and_two_csv_files_without_touching_map(tmp_path: Path) -> None:
    report_a = _write_report(tmp_path / "run_a")
    report_b = _write_report(tmp_path / "run_b", x_offset=0.5)
    output_dir = tmp_path / "summary"

    summary = summarize_reports([(report_a, "B"), (report_b, "B")])
    paths = write_summary(summary, output_dir)

    assert Path(paths["json"]).is_file()
    assert Path(paths["holes_csv"]).is_file()
    assert Path(paths["groups_csv"]).is_file()
    saved = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    per_hole = {
        item["hole_id"]: item
        for item in saved["comparison"][0]["per_hole"]
    }
    assert per_hole[1]["repeatability"]["sample_count"] == 2
    assert not (tmp_path / "map.json").exists()
