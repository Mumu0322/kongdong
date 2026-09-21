from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from aubo_workbench.hole_localization_report import (
    build_result_summary,
    build_progress_checkpoint,
    render_result_summary_text,
    summarize_motion_distance,
    summarize_timing_window,
    summarize_visual_timing_window,
    write_result_summary,
)
from aubo_workbench.hole_localization_models import TimingRecorder


class HoleLocalizationReportTests(unittest.TestCase):
    def test_failed_hole_summary_keeps_planned_tcp_distinct_from_actual(self) -> None:
        report = {
            "status": "failed",
            "hole_count": 1,
            "timing": {"total_elapsed_s": 1.0, "events": []},
            "stages": {"processed_holes": {"holes": [{
                "hole_id": 1,
                "status": "deferred_unexpected_error",
                "target_point_base_mm": [1.0, 2.0, 3.0],
                "planned_final_point": {
                    "planned_final_tcp_xyz_mm": [10.0, 20.0, 30.0],
                    "planned_final_tcp_pose_m_rad": [0.01, 0.02, 0.03, 0.0, 0.0, 0.0],
                },
            }]}}
        }
        summary = build_result_summary(report)
        hole = summary["holes"][0]
        self.assertEqual(hole["final_point_base_mm"], [1.0, 2.0, 3.0])
        self.assertEqual(hole["planned_final_tcp_xyz_mm"], [10.0, 20.0, 30.0])
        self.assertIn("计划值不代表实际到位", render_result_summary_text(summary))

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

    def test_visual_timing_excludes_robot_wait_and_splits_leaf_stages(self) -> None:
        events = [
            {
                "name": "capture_all_holes",
                "elapsed_s": 10.0,
                "start_offset_s": 0.0,
                "end_offset_s": 10.0,
                "depth": 0,
                "category": "vision_compute",
                "visual_stage": "visual_pipeline_other",
            },
            {
                "name": "robot/move",
                "elapsed_s": 2.0,
                "start_offset_s": 0.0,
                "end_offset_s": 2.0,
                "depth": 1,
                "category": "robot_motion",
            },
            {
                "name": "operator/wait",
                "elapsed_s": 2.0,
                "start_offset_s": 2.0,
                "end_offset_s": 4.0,
                "depth": 1,
                "category": "operator_wait",
            },
            {
                "name": "capture/frame",
                "elapsed_s": 1.0,
                "start_offset_s": 4.0,
                "end_offset_s": 5.0,
                "depth": 1,
                "category": "vision_compute",
                "visual_stage": "frame_acquisition",
            },
            {
                "name": "capture/yolo",
                "elapsed_s": 1.0,
                "start_offset_s": 5.0,
                "end_offset_s": 6.0,
                "depth": 1,
                "category": "vision_compute",
                "visual_stage": "yolo_inference",
            },
        ]
        summary = summarize_visual_timing_window(events, start_s=0.0, end_s=10.0)

        self.assertEqual(summary["total_visual_s"], 6.0)
        self.assertEqual(summary["frame_acquisition_s"], 1.0)
        self.assertEqual(summary["yolo_inference_s"], 1.0)
        self.assertEqual(summary["visual_pipeline_other_s"], 4.0)

        report = {
            "status": "completed",
            "session_end_reason": "completed",
            "run_dir": "run",
            "created_at": "2026-09-02T00:00:00",
            "cycle_index": 1,
            "timing": {"total_elapsed_s": 10.0, "events": events},
            "final_result": {"holes": []},
        }
        text = render_result_summary_text(build_result_summary(report))
        self.assertIn("视觉生产节拍", text)
        self.assertNotIn("机械臂运动：", text)

    def test_motion_distance_splits_horizontal_up_and_down(self) -> None:
        events = [
            {
                "name": "孔1安全高度平移",
                "status": "completed",
                "motion_type": "move_line",
                "motion_profile": "transit",
                "start_xyz_mm": [0.0, 0.0, 10.0],
                "target_xyz_mm": [3.0, 4.0, 20.0],
            },
            {
                "name": "孔1纯Z下降",
                "status": "completed",
                "motion_type": "move_line",
                "motion_profile": "precision",
                "start_xyz_mm": [3.0, 4.0, 20.0],
                "target_xyz_mm": [3.0, 4.0, 5.0],
            },
            {
                "name": "cancelled",
                "status": "failed",
                "start_xyz_mm": [0.0, 0.0, 0.0],
                "target_xyz_mm": [100.0, 0.0, 0.0],
            },
        ]
        summary = summarize_motion_distance(events)
        self.assertEqual(summary["segment_count"], 2)
        self.assertAlmostEqual(summary["total_path_mm"], 26.18, places=2)
        self.assertAlmostEqual(summary["horizontal_distance_mm"], 5.0, places=3)
        self.assertAlmostEqual(summary["vertical_up_mm"], 10.0, places=3)
        self.assertAlmostEqual(summary["vertical_down_mm"], 15.0, places=3)
        self.assertAlmostEqual(summary["vertical_net_mm"], -5.0, places=3)
        self.assertEqual(summary["by_hole"]["1"]["segment_count"], 2)

    def test_motion_distance_is_written_to_summary_and_text(self) -> None:
        report = {
            "status": "completed",
            "session_end_reason": "completed",
            "run_dir": "run",
            "created_at": "2026-09-02T00:00:00",
            "cycle_index": 1,
            "timing": {
                "total_elapsed_s": 1.0,
                "events": [],
                "motion_events": [
                    {
                        "name": "孔1下降",
                        "status": "completed",
                        "motion_type": "move_line",
                        "start_xyz_mm": [0.0, 0.0, 20.0],
                        "target_xyz_mm": [0.0, 0.0, 10.0],
                    },
                ],
            },
            "final_result": {"holes": []},
        }
        summary = build_result_summary(report)
        motion = summary["timing"]["motion_distance"]
        self.assertEqual(motion["segment_count"], 1)
        self.assertEqual(motion["vertical_down_mm"], 10.0)
        text = render_result_summary_text(summary)
        self.assertIn("机械臂运动距离", text)
        self.assertIn("下降：10.0 mm", text)

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

    def test_capture_only_holes_are_measured_without_being_reported_as_failures(self) -> None:
        report = {
            "status": "coarse_direct_capture_only_complete",
            "session_end_reason": "completed",
            "run_dir": "run",
            "created_at": "2026-09-15T00:00:00",
            "cycle_index": 1,
            "hole_count": 2,
            "stages": {
                "sequential_plan": {
                    "hole_order": [1, 2],
                    "capture_height_mm": 320.0,
                },
            },
            "timing": {"total_elapsed_s": 1.0, "events": []},
            "final_result": {
                "holes": [
                    {
                        "hole_id": 1,
                        "status": "capture_only",
                        "localization_path": "coarse_320_direct",
                        "target_point_base_mm": [1.0, 2.0, 3.0],
                        "coarse_capture_height_mm": 320.0,
                        "fine_quality_status": "coarse_direct_capture_only",
                        "timing": {"events": []},
                    },
                    {
                        "hole_id": 2,
                        "status": "deferred_coarse_pointcloud",
                        "error": "empty point cloud",
                        "timing": {"events": []},
                    },
                ],
            },
        }

        summary = build_result_summary(report)

        self.assertEqual(summary["status"]["capture_only_holes"], 1)
        self.assertEqual(summary["status"]["completed_holes"], 0)
        self.assertEqual(summary["status"]["deferred_holes"], 1)
        self.assertEqual(summary["status"]["failed_holes"], [2])
        self.assertEqual(summary["status"]["cycle_status"], "coarse_direct_capture_only_complete")
        self.assertEqual(summary["holes"][0]["source"], "coarse_direct_capture_only")
        self.assertIn("仅采集", render_result_summary_text(summary))

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

    def test_interrupted_report_uses_processed_holes_and_keeps_remaining_ids(self) -> None:
        processed = [
            {
                "hole_id": hole_id,
                "status": "completed",
                "batch_fine_source": "batch_fine_at_260mm",
                "timing": {"events": []},
            }
            for hole_id in range(1, 34)
        ]
        processed.append({
            "hole_id": 34,
            "status": "deferred_unexpected_error",
            "coarse_source": "batch_coarse_localization",
            "error": "UnboundLocalError: local variable 'rgbd_pipeline' referenced before assignment",
            "timing": {"events": []},
        })
        report = {
            "status": "failed",
            "session_end_reason": "error",
            "hole_count": 42,
            "run_dir": "run",
            "stages": {
                "sequential_plan": {"hole_order": list(range(1, 43))},
                "processed_holes": {"holes": processed},
            },
            "timing": {"total_elapsed_s": 10.0, "events": []},
        }

        summary = build_result_summary(report)
        progress = build_progress_checkpoint(report)

        self.assertEqual(summary["status"]["selected_holes"], 42)
        self.assertEqual(summary["status"]["processed_holes"], 34)
        self.assertEqual(summary["status"]["completed_holes"], 33)
        self.assertEqual(summary["status"]["deferred_holes"], 9)
        self.assertEqual(summary["status"]["failed_holes"], [34])
        self.assertEqual(summary["status"]["unprocessed_holes"], list(range(35, 43)))
        self.assertEqual(progress["processed_count"], 34)
        self.assertEqual(progress["completed_count"], 33)
        self.assertEqual(progress["remaining_hole_ids"], list(range(35, 43)))
        self.assertNotIn(0, [item["hole_id"] for item in summary["holes"]])
        self.assertIn("未处理孔号", render_result_summary_text(summary))


if __name__ == "__main__":
    unittest.main()
