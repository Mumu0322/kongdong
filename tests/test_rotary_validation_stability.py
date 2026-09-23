from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import run_yolo_eye_in_hand_optimized as module


class RotaryValidationStabilityTests(unittest.TestCase):
    def test_rotary_rebuild_forces_coarse_only_build(self) -> None:
        args = SimpleNamespace(
            hole_map_mode="execute",
            batch_coarse_localization=False,
            batch_fine_localization=False,
            batch_fine_joint_localization=False,
            move_final_xy=True,
            reuse_coarse_cache=True,
            reuse_persistent_coarse_cache=True,
            shared_cache_validation=True,
        )

        result = module._prepare_rotary_sector_rebuild_args(args)

        self.assertIs(result, args)
        self.assertEqual(args.hole_map_mode, "build")
        self.assertTrue(args.batch_coarse_localization)
        self.assertFalse(args.batch_fine_localization)
        self.assertFalse(args.batch_fine_joint_localization)
        self.assertFalse(args.batch_fine_pointcloud_xy_fusion)
        self.assertFalse(args.move_final_xy)
        self.assertFalse(args.reuse_coarse_cache)
        self.assertFalse(args.reuse_persistent_coarse_cache)
        self.assertFalse(args.shared_cache_validation)

    def test_only_v4_coarse_sector_map_is_call_ready(self) -> None:
        args = SimpleNamespace(sector_id=1, hole_map_path="current.json")
        payload = {
            "schema_version": 4,
            "kind": "coarse_sector_hole_map",
            "target_policy": "coarse_map_requires_fresh_fine",
            "requires_fresh_fine": True,
            "sectors": {
                "S01": {
                    "status": "valid",
                    "holes": {
                        "H01": {
                            "status": "ready",
                            "hole_id": 1,
                            "coarse_center_base_mm": [1.0, 2.0, 3.0],
                            "coarse_plane_point_base_mm": [1.0, 2.0, 3.0],
                            "coarse_normal_toward_camera_base": [0.0, 0.0, 1.0],
                        },
                    },
                },
            },
        }
        with patch.object(
            module, "_resolve_hole_map_path", return_value=(Path("map.json"), "map")
        ), patch.object(module, "load_hole_map", return_value=payload):
            self.assertTrue(module._rotary_sector_map_ready_for_auto_call(args))

        legacy = dict(payload, schema_version=3)
        with patch.object(
            module, "_resolve_hole_map_path", return_value=(Path("map.json"), "map")
        ), patch.object(module, "load_hole_map", return_value=legacy):
            self.assertFalse(module._rotary_sector_map_ready_for_auto_call(args))

    def test_map_execution_wraps_live_two_stage_fine_localization(self) -> None:
        coarse_hole = {
            "hole_id": 1,
            "status": "ready",
            "class_id": 0,
            "class_name": "hole",
            "coarse_center_base_mm": [10.0, 20.0, 30.0],
            "coarse_plane_point_base_mm": [10.0, 20.0, 30.0],
            "coarse_normal_toward_camera_base": [0.0, 0.0, 1.0],
        }
        payload = {
            "schema_version": 2,
            "kind": "coarse_hole_map",
            "status": "valid",
            "scope": "current_robot_cycle",
            "target_policy": "coarse_map_requires_fresh_fine",
            "requires_fresh_fine": True,
            "environment": {},
            "holes": {"H01": coarse_hole},
        }
        args = SimpleNamespace(
            hole_map_path="map.json",
            hole_ids=[1],
            sector_id=None,
            handeye="handeye.json",
            model="model.pt",
            move_final_xy=True,
            execute=True,
        )
        captured: dict[str, object] = {}

        def fake_run(run_args, _handeye, _model, *, initial_holes_override):
            captured["args"] = run_args
            captured["seeds"] = initial_holes_override
            return 0

        with patch.object(module, "_resolve_hole_map_path", return_value=(Path("map.json"), "map")), \
                patch.object(module, "load_hole_map", return_value=payload), \
                patch.object(module, "validate_hole_map", return_value=[1]), \
                patch.object(module, "get_hole", return_value=coarse_hole), \
                patch.object(module, "load_handeye_experiment_result", return_value=object()), \
                patch.object(module, "load_yolo", return_value=object()), \
                patch.object(module, "file_sha256", return_value=None), \
                patch.object(module, "run_two_stage_hole_localization", side_effect=fake_run):
            result = module.run_hole_map_execution(args)

        self.assertEqual(result, 0)
        self.assertIs(captured["args"], args)
        self.assertTrue(args.batch_fine_localization)
        self.assertTrue(args.batch_coarse_localization)
        self.assertTrue(args.move_final_xy)
        self.assertEqual(args.hole_map_mode, "execute")
        seeds = captured["seeds"]
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]["initial_center_base_mm"], [10.0, 20.0, 30.0])

    def test_map_seed_correction_waits_for_live_alignment(self) -> None:
        coarse_hole = {
            "hole_id": 1,
            "status": "ready",
            "coarse_center_base_mm": [10.0, 20.0, 30.0],
            "coarse_plane_point_base_mm": [10.0, 20.0, 30.0],
            "coarse_normal_toward_camera_base": [0.0, 0.0, 1.0],
        }
        payload = {
            "schema_version": 2,
            "kind": "coarse_hole_map",
            "map_id": "map-with-correction",
            "status": "valid",
            "target_policy": "coarse_map_requires_fresh_fine",
            "requires_fresh_fine": True,
            "environment": {},
            "holes": {"H01": coarse_hole},
        }
        correction_model = {
            "status": "provisional_single_reference_run",
            "reference": {"run_id": "reference-run"},
            "matching": {"matched_count": 1},
            "holes": {
                "H01": {
                    "hole_id": 1,
                    "correction_xy_base_mm": [2.0, -1.0],
                }
            },
        }
        args = SimpleNamespace(
            hole_map_path="map.json",
            hole_ids=[1],
            sector_id=None,
            handeye="handeye.json",
            model="model.pt",
            move_final_xy=True,
            execute=True,
            hole_map_seed_correction=True,
            hole_map_seed_correction_path=Path(__file__),
        )
        captured: dict[str, object] = {}

        def fake_run(run_args, _handeye, _model, *, initial_holes_override):
            captured["seeds"] = initial_holes_override
            return 0

        with patch.object(module, "_resolve_hole_map_path", return_value=(Path("map.json"), "map")), \
                patch.object(module, "load_hole_map", return_value=payload), \
                patch.object(module, "validate_hole_map", return_value=[1]), \
                patch.object(module, "get_hole", return_value=coarse_hole), \
                patch.object(module, "load_handeye_experiment_result", return_value=object()), \
                patch.object(module, "load_yolo", return_value=object()), \
                patch.object(module, "file_sha256", return_value=None), \
                patch.object(module, "load_seed_correction_model", return_value=correction_model), \
                patch.object(module, "validate_model_binding"), \
                patch.object(module, "run_two_stage_hole_localization", side_effect=fake_run):
            result = module.run_hole_map_execution(args)

        self.assertEqual(result, 0)
        self.assertEqual(captured["seeds"][0]["initial_center_base_mm"], [10.0, 20.0, 30.0])
        self.assertIs(args._coarse_map_seed_correction_model, correction_model)
        summary = args._coarse_map_seed_correction_summary
        self.assertTrue(summary["loaded"])
        self.assertFalse(summary["static_reference_applied_before_live_alignment"])

    def test_ignored_move_line_is_accepted_only_when_tcp_is_already_at_target(self) -> None:
        args = SimpleNamespace(speed_m_s=0.08, acc_m_s2=0.25)
        motion = SimpleNamespace(move_line=lambda *_args: [0, 13])
        target = np.eye(4, dtype=np.float64)

        with patch.object(module, "_wait_robot_steady", return_value=({}, target.copy())):
            actual = module._confirm_and_move_line(
                "already there",
                np.eye(4, dtype=np.float64),
                target,
                args,
                motion,
                object(),
                require_confirmation=False,
            )

        self.assertTrue(np.allclose(actual, target))

    def test_ignored_move_line_still_fails_when_tcp_is_not_at_target(self) -> None:
        args = SimpleNamespace(speed_m_s=0.08, acc_m_s2=0.25)
        motion = SimpleNamespace(move_line=lambda *_args: [0, 13])
        target = np.eye(4, dtype=np.float64)
        target[0, 3] = 10.0

        with patch.object(module, "_wait_robot_steady", return_value=({}, np.eye(4))):
            with self.assertRaisesRegex(RuntimeError, "AUBO_REQUEST_IGNORE"):
                module._confirm_and_move_line(
                    "not there",
                    np.eye(4, dtype=np.float64),
                    target,
                    args,
                    motion,
                    object(),
                    require_confirmation=False,
                )

    def test_successful_move_waits_for_fresh_target_pose_not_stale_steady_flag(self) -> None:
        class FakePoseSession:
            def __init__(self) -> None:
                self.snapshots = [
                    {"power_on": True, "steady": True, "collision": False,
                     "pose_values_sdk_m_rad": [0.0] * 6},
                    {"power_on": True, "steady": True, "collision": False,
                     "pose_values_sdk_m_rad": [0.010, 0.0, 0.0, 0.0, 0.0, 0.0]},
                ]

            def read_pose_snapshot(self):
                return self.snapshots.pop(0) if self.snapshots else {
                    "power_on": True, "steady": True, "collision": False,
                    "pose_values_sdk_m_rad": [0.010, 0.0, 0.0, 0.0, 0.0, 0.0],
                }

            def pose_sdk_to_transform_mm(self, pose):
                transform = np.eye(4, dtype=np.float64)
                transform[:3, 3] = np.asarray(pose[:3], dtype=np.float64) * 1000.0
                return transform

        args = SimpleNamespace(speed_m_s=0.08, acc_m_s2=0.25)
        motion = SimpleNamespace(move_line=lambda *_args: [0, 0])
        target = np.eye(4, dtype=np.float64)
        target[0, 3] = 10.0
        pose = FakePoseSession()
        with patch.object(module.time, "sleep"):
            actual = module._confirm_and_move_line(
                "fresh target", np.eye(4, dtype=np.float64), target,
                args, motion, pose, require_confirmation=False,
            )
        self.assertTrue(np.allclose(actual, target))

    def test_reached_timeout_reports_accepted_command_without_tcp_motion(self) -> None:
        class StationaryPoseSession:
            def read_pose_snapshot(self):
                return {
                    "power_on": True, "steady": True, "collision": False,
                    "robot_mode": "RobotModeType.Running",
                    "safety_mode": "SafetyModeType.Normal",
                    "pose_values_sdk_m_rad": [0.0] * 6,
                }

            def pose_sdk_to_transform_mm(self, _pose):
                return np.eye(4, dtype=np.float64)

        target = np.eye(4, dtype=np.float64)
        target[0, 3] = 340.0
        with patch.object(module, "ROBOT_STEADY_POLL_INTERVAL_S", 0.001):
            with self.assertRaisesRegex(
                module.MotionExecutionError, "下发后TCP未见明显运动",
            ) as caught:
                module._wait_robot_reached(
                    StationaryPoseSession(), target, timeout_s=0.02,
                    command_start_pose=np.eye(4),
                )
        self.assertIn("RobotModeType.Running", str(caught.exception))
        self.assertIn("SafetyModeType.Normal", str(caught.exception))

    def test_move_line_timeout_keeps_sdk_response_in_error(self) -> None:
        args = SimpleNamespace(speed_m_s=0.08, acc_m_s2=0.25)
        motion = SimpleNamespace(move_line=lambda *_args: [0, 0])
        target = np.eye(4, dtype=np.float64)
        target[0, 3] = 340.0
        with patch.object(module, "_wait_motion_session_steady"), \
                patch.object(module, "_require_safe_snapshot", return_value=({}, np.eye(4))), \
                patch.object(
                    module, "_wait_robot_reached",
                    side_effect=module.MotionExecutionError("timeout"),
                ):
            with self.assertRaisesRegex(module.MotionExecutionError, r"\[0, 0\]"):
                module._confirm_and_move_line(
                    "first coarse", np.eye(4), target, args, motion, object(),
                    require_confirmation=False,
                )

    def test_home_motion_uses_joint_target_when_saved_tcp_is_stale(self) -> None:
        class FakePoseSession:
            def __init__(self) -> None:
                self.snapshots = [
                    {"power_on": True, "steady": True, "collision": False,
                     "joints_rad": [0.5] * 6,
                     "pose_values_sdk_m_rad": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]},
                    {"power_on": True, "steady": True, "collision": False,
                     "joints_rad": [0.0] * 6,
                     "pose_values_sdk_m_rad": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]},
                ]

            def read_pose_snapshot(self):
                return self.snapshots.pop(0) if self.snapshots else {
                    "power_on": True, "steady": True, "collision": False,
                    "joints_rad": [0.0] * 6,
                    "pose_values_sdk_m_rad": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                }

            def pose_sdk_to_transform_mm(self, pose):
                transform = np.eye(4, dtype=np.float64)
                transform[:3, 3] = np.asarray(pose[:3], dtype=np.float64) * 1000.0
                return transform

        home = SimpleNamespace(
            joints_rad=[0.0] * 6,
            tcp_pose_m_rad=[0.0] * 6,
            name="原始点",
            created_at="test",
        )
        pose = FakePoseSession()
        motion = SimpleNamespace(move_joint=lambda *_args: [0, 0])
        with patch.object(module.time, "sleep"):
            actual = module._confirm_and_move_home(home, motion, pose)
        self.assertTrue(np.allclose(actual[:3, 3], [200.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
