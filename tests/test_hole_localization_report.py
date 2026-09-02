from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from aubo_workbench.hole_localization_report import (
    build_result_summary,
    render_result_summary_text,
    summarize_timing_window,
    write_result_summary,
)
from aubo_workbench.hole_localization_models import TimingRecorder


class HoleLocalizationReportTests(unittest.TestCase):
    def test_timing_recorder_persists_intervals_and_markers(self) -> None:
        timer = TimingRecorder()
        timer.mark("cycle/task_start")
        with timer.measure("robot/move", category="robot_motion"):
            time.sleep(0.001)
        timer.mark("cycle/complete")

        snapshot = timer.snapshot()
        event = snapshot["events"][0]

        self.assertEqual(snapshot["schema_version"], 2)
        self.assertEqual([item["name"] for item in snapshot["markers"]], [
            "cycle/task_start", "cycle/complete",
        ])
        self.assertGreaterEqual(event["end_offset_s"], event["start_offset_s"])
        self.assertEqual(event["category"], "robot_motion")
        self.assertEqual(event["depth"], 0)

    def test_nested_events_are_counted_once_on_the_timeline(self) -> None:
        events = [
            {
                "name": "group/capture_and_localize",
                "elapsed_s": 10.0,
                "start_offset_s": 0.0,
                "end_offset_s": 10.0,
                "depth": 0,
                "category": "robot_motion",
            },
            {
                "name": "group/capture",
                "elapsed_s": 4.0,
                "start_offset_s": 3.0,
                "end_offset_s": 7.0,
                "depth": 1,
                "category": "vision_compute",
            },
            {
                "name": "artifact_io/write",
                "elapsed_s": 1.0,
                "start_offset_s": 5.0,
                "end_offset_s": 6.0,
                "depth": 2,
                "category": "artifact_io",
                "exclude_from_effective": True,
            },
        ]

        summary = summarize_timing_window(events, start_s=0.0, end_s=10.0)

        self.assertTrue(summary["interval_accounting"])
        self.assertEqual(summary["wall_elapsed_s"], 10.0)
        self.assertEqual(summary["robot_motion_s"], 6.0)
        self.assertEqual(summary["vision_compute_s"], 3.0)
        self.assertEqual(summary["artifact_io_s"], 1.0)
        self.assertEqual(summary["production_cycle_s"], 9.0)
        self.assertEqual(summary["pure_execution_s"], 9.0)

    def test_summary_separates_cycle_from_post_cycle_session_wait(self) -> None:
        timing = {
            "total_elapsed_s": 20.0,
            "events": [
                {
                    "name": "robot/final_motion",
                    "elapsed_s": 10.0,
                    "start_offset_s": 0.0,
                    "end_offset_s": 10.0,
                    "depth": 0,
                    "category": "robot_motion",
                },
                {
                    "name": "operator/next_cycle_confirmation",
                    "elapsed_s": 10.0,
                    "start_offset_s": 10.0,
                    "end_offset_s": 20.0,
                    "depth": 0,
                    "category": "operator_wait",
                },
            ],
            "markers": [
                {"name": "cycle/task_start", "offset_s": 0.0},
                {"name": "cycle/selection_confirmed", "offset_s": 0.0},
                {"name": "cycle/automatic_execution_start", "offset_s": 0.0},
                {"name": "cycle/complete", "offset_s": 10.0},
            ],
        }
        report = {
            "status": "stopped_by_user",
            "session_end_reason": "user_exit_after_cycle",
            "run_dir": "run",
            "created_at": "2026-09-02T00:00:00",
            "cycle_index": 1,
            "timing": timing,
            "final_result": {
                "holes": [
                    {
                        "hole_id": 1,
                        "status": "completed",
                        "batch_fine_source": "per_hole_fine",
                        "target_point_base_mm": [1.0, 2.0, 3.0],
                        "timing": {"events": [timing["events"][0]]},
                    },
                ],
            },
        }

        summary = build_result_summary(report)

        self.assertEqual(summary["status"]["cycle_status"], "completed")
        self.assertEqual(summary["timing"]["session_wall_s"], 20.0)
        self.assertEqual(summary["timing"]["task_wall_s"], 10.0)
        self.assertEqual(summary["timing"]["production_cycle_s"], 10.0)
        self.assertEqual(summary["timing"]["post_cycle_wall_s"], 10.0)
        self.assertEqual(summary["status"]["session_end_reason"], "user_exit_after_cycle")

    def test_result_summary_writes_small_json_and_text_files(self) -> None:
        report = {
            "status": "completed",
            "session_end_reason": "completed",
            "run_dir": "run",
            "created_at": "2026-09-02T00:00:00",
            "cycle_index": 1,
            "timing": {"total_elapsed_s": 1.0, "events": []},
            "final_result": {"holes": []},
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = write_result_summary(Path(directory), report)
            json_path = Path(paths["json"])
            text_path = Path(paths["text"])
            self.assertTrue(json_path.is_file())
            self.assertTrue(text_path.is_file())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertIn("生产节拍", render_result_summary_text(payload))
            self.assertLess(json_path.stat().st_size, 20_000)


if __name__ == "__main__":
    unittest.main()
