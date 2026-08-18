from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import run_yolo_eye_in_hand_optimized as localization
from aubo_workbench.camera import CameraIntrinsics
from aubo_workbench.coarse_cache import (
    CacheValidationGates,
    CoarseCacheEntry,
    cache_entry_compatibility_reasons,
    load_cache_entries,
    load_persistent_cache_entries,
    match_entries_by_base_point,
    replace_base_z,
    save_persistent_cache_entries,
    save_cache_entries,
    transform_cached_points_to_camera,
    validate_cache_entry,
)
from aubo_workbench.gui_hole_localization import HoleLocalizationPanel


class _Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def _entry() -> CoarseCacheEntry:
    return CoarseCacheEntry(
        hole_id=1,
        T_base_camera_build=np.eye(4),
        T_tcp_camera=np.eye(4),
        tcp_pose_m_rad=[0.0] * 6,
        camera_serial="camera-1",
        handeye_path="handeye.json",
        intrinsics=CameraIntrinsics(1280, 720, 800.0, 800.0, 640.0, 360.0, ()).as_dict(),
        center_px=np.array([640.0, 360.0]),
        point_camera_mm=np.array([10.0, 20.0, 340.0]),
        plane_point_camera_mm=np.array([10.0, 20.0, 338.0]),
        normal_camera=np.array([0.0, 0.0, -1.0]),
        point_base_mm=np.array([10.0, 20.0, 340.0]),
        plane_point_base_mm=np.array([10.0, 20.0, 338.0]),
        normal_base=np.array([0.0, 0.0, -1.0]),
        plane_rmse_mm=1.0,
        valid_frames=3,
        total_frames=3,
        center_scatter_p95_px=0.2,
        ring_points_median=120.0,
        surface_model="local_tangent_plane",
        frame_indices=np.array([0, 1, 2]),
        frame_centers_px=np.array([[640.0, 360.0], [640.1, 360.0], [639.9, 360.1]]),
        frame_plane_rmse_mm=np.array([1.0, 1.1, 0.9]),
        points_camera_mm_by_frame=(
            np.array([[0.0, 0.0, 340.0], [1.0, 0.0, 340.0]]),
            np.array([[0.0, 1.0, 340.0], [1.0, 1.0, 340.0]]),
            np.array([[0.0, 0.0, 339.9], [1.0, 0.0, 340.1]]),
        ),
    )


class CoarseCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gates = CacheValidationGates()
        self.intrinsics = CameraIntrinsics(1280, 720, 800.0, 800.0, 640.0, 360.0, ())

    def _measurements(self, *, tracking=5.0, normal=(0.0, 0.0, -1.0)) -> list[dict[str, object]]:
        return [
            {
                "center_px": np.array([640.0 + offset, 360.0]),
                "tracking_distance_px": tracking,
                "plane_point_camera_mm": np.array([10.0, 20.0, 340.0 + offset * 0.1]),
                "surface_plane_point_camera_mm": np.array([10.0, 20.0, 338.0 + offset * 0.1]),
                "normal_camera": np.array(normal),
                "plane_rmse_mm": 1.0,
                "error": None,
            }
            for offset in (0.0, 0.1, -0.1)
        ]

    def test_save_and_load_npz_json_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = save_cache_entries(directory, {1: _entry()}, metadata={"test": True})
            self.assertEqual(path, Path(directory) / "manifest.json")
            loaded = load_cache_entries(directory)
            self.assertEqual(sorted(loaded), [1])
            np.testing.assert_allclose(loaded[1].point_base_mm, [10.0, 20.0, 340.0])
            self.assertEqual(len(loaded[1].points_camera_mm_by_frame), 3)
            self.assertTrue((Path(directory) / "hole_01.npz").is_file())

    def test_save_and_load_persistent_base_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            save_persistent_cache_entries(directory, {1: _entry()})
            errors: dict[int, str] = {}
            loaded = load_persistent_cache_entries(directory, errors=errors)
            self.assertEqual(sorted(loaded), [1])
            self.assertEqual(errors, {})
            manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["cache_scope"], "base_frame_persistent")

    def test_base_point_matching_is_one_to_one_and_reports_ambiguity(self) -> None:
        first = _entry()
        second = _entry()
        second.hole_id = 2
        second.point_base_mm = np.array([100.0, 0.0, 340.0])
        matched, source_ids, audit = match_entries_by_base_point(
            {10: [10.0, 20.0, 0.0], 11: [100.5, 0.0, 0.0]},
            {1: first, 2: second},
            max_match_distance_mm=30.0,
            min_match_margin_mm=5.0,
        )
        self.assertEqual(source_ids, {10: 1, 11: 2})
        self.assertEqual(sorted(matched), [10, 11])
        self.assertTrue(audit[10]["matched"])

        _, _, ambiguous = match_entries_by_base_point(
            {10: [50.0, 0.0, 0.0]},
            {1: first, 2: second},
            max_match_distance_mm=60.0,
            min_match_margin_mm=60.0,
        )
        self.assertEqual(ambiguous[10]["reason"], "ambiguous_world_match")

    def test_persistent_compatibility_rejects_changed_camera_or_handeye(self) -> None:
        entry = _entry()
        reasons = cache_entry_compatibility_reasons(
            entry,
            camera_serial="camera-2",
            T_tcp_camera=np.eye(4),
            intrinsics=self.intrinsics,
            handeye_path="different-handeye.json",
        )
        self.assertIn("camera_serial_mismatch", reasons)
        self.assertIn("handeye_path_mismatch", reasons)

    def test_load_rejects_manifest_npz_frame_audit_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            save_cache_entries(directory, {1: _entry()})
            manifest_path = Path(directory) / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["entries"]["1"]["frame_indices"] = [10, 11, 12]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cache_entries(directory)

    def test_replace_base_z_keeps_cached_xy(self) -> None:
        np.testing.assert_allclose(replace_base_z([1.0, 2.0, 3.0], [9.0, 8.0, 7.0]), [1.0, 2.0, 7.0])

    def test_transform_cached_points_from_build_camera_to_current_camera(self) -> None:
        entry = _entry()
        current = np.eye(4)
        current[:3, 3] = [10.0, 20.0, 30.0]
        transformed = transform_cached_points_to_camera(entry, current)
        np.testing.assert_allclose(
            transformed[0][0], [-10.0, -20.0, 310.0], atol=1e-6,
        )

    def test_validation_transforms_live_point_and_replaces_only_z(self) -> None:
        current_transform = np.eye(4)
        current_transform[:3, 3] = [0.0, 0.0, 10.0]
        result = validate_cache_entry(
            _entry(), self._measurements(),
            current_T_base_camera=current_transform,
            intrinsics=self.intrinsics,
            gates=self.gates,
        )
        self.assertTrue(result.accepted)
        np.testing.assert_allclose(result.current_point_base_mm, [10.0, 20.0, 350.0])
        np.testing.assert_allclose(result.current_plane_point_base_mm, [10.0, 20.0, 348.0])
        np.testing.assert_allclose(
            replace_base_z(_entry().point_base_mm, result.current_point_base_mm),
            [10.0, 20.0, 350.0],
        )

    def test_validation_requires_all_three_measurement_frames(self) -> None:
        result = validate_cache_entry(
            _entry(), self._measurements()[:2],
            current_T_base_camera=np.eye(4),
            intrinsics=self.intrinsics,
            gates=self.gates,
        )
        self.assertFalse(result.accepted)
        self.assertIn("validation_frames", result.reason)

    def test_two_of_three_valid_measurements_are_accepted(self) -> None:
        measurements = self._measurements()
        measurements[2]["error"] = "yolo_missing"
        result = validate_cache_entry(
            _entry(), measurements, current_T_base_camera=np.eye(4),
            intrinsics=self.intrinsics, gates=self.gates,
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.valid_frames, 2)

    def test_tracking_distance_rejects_cache(self) -> None:
        result = validate_cache_entry(
            _entry(), self._measurements(tracking=71.0), current_T_base_camera=np.eye(4),
            intrinsics=self.intrinsics, gates=self.gates,
        )
        self.assertFalse(result.accepted)
        self.assertIn("tracking_distance", result.reason)

    def test_center_scatter_rejects_cache(self) -> None:
        measurements = self._measurements()
        measurements[2]["center_px"] = np.array([642.0, 360.0])
        result = validate_cache_entry(
            _entry(), measurements, current_T_base_camera=np.eye(4),
            intrinsics=self.intrinsics, gates=self.gates,
        )
        self.assertFalse(result.accepted)
        self.assertIn("center_scatter", result.reason)

    def test_center_offset_rejects_cache(self) -> None:
        measurements = self._measurements()
        for item in measurements:
            item["center_px"] = np.array([646.0, 360.0])
        result = validate_cache_entry(
            _entry(), measurements,
            current_T_base_camera=np.eye(4),
            intrinsics=self.intrinsics,
            gates=self.gates,
        )
        self.assertFalse(result.accepted)
        self.assertIn("center_offset", result.reason)

    def test_plane_rmse_rejects_cache(self) -> None:
        measurements = self._measurements()
        for item in measurements:
            item["plane_rmse_mm"] = 3.6
        result = validate_cache_entry(
            _entry(), measurements,
            current_T_base_camera=np.eye(4),
            intrinsics=self.intrinsics,
            gates=self.gates,
        )
        self.assertFalse(result.accepted)
        self.assertIn("plane_quality", result.reason)

    def test_normal_rejects_cache(self) -> None:
        result = validate_cache_entry(
            _entry(), self._measurements(normal=(0.1, 0.0, -0.995)),
            current_T_base_camera=np.eye(4), intrinsics=self.intrinsics, gates=self.gates,
        )
        self.assertFalse(result.accepted)
        self.assertIn("normal_error", result.reason)

    def test_cli_cache_switch_defaults_on_and_can_be_disabled(self) -> None:
        self.assertTrue(localization.build_parser().parse_args([]).reuse_coarse_cache)
        self.assertTrue(localization.build_parser().parse_args([]).reuse_persistent_coarse_cache)
        self.assertFalse(
            localization.build_parser().parse_args(["--no-reuse-coarse-cache"]).reuse_coarse_cache
        )
        self.assertTrue(
            localization.build_parser().parse_args(["--reuse-coarse-cache"]).reuse_coarse_cache
        )
        self.assertFalse(
            localization.build_parser().parse_args(
                ["--no-reuse-persistent-coarse-cache"]
            ).reuse_persistent_coarse_cache
        )

    @staticmethod
    def _panel_for_command(reuse: bool) -> HoleLocalizationPanel:
        panel = HoleLocalizationPanel.__new__(HoleLocalizationPanel)
        source_file = str(Path(__file__))
        panel.model_var = _Value(source_file)
        panel.handeye_var = _Value(source_file)
        panel.cad_model_var = _Value(source_file)
        panel.confidence_var = _Value("0.35")
        panel.coarse_height_var = _Value("340")
        panel.fine_height_var = _Value("260")
        panel.coarse_frames_var = _Value("10")
        panel.fine_frames_var = _Value("20")
        panel.speed_var = _Value("0.08")
        panel.acc_var = _Value("0.25")
        panel.transit_speed_var = _Value("0.15")
        panel.transit_acc_var = _Value("0.45")
        panel.approach_speed_var = _Value("0.12")
        panel.approach_acc_var = _Value("0.35")
        panel.execute_var = _Value(False)
        panel.experimental_var = _Value(False)
        panel.final_target_mode_var = _Value("机械爪模式")
        panel.final_xy_var = _Value(False)
        panel.reuse_coarse_cache_var = _Value(reuse)
        panel.reuse_persistent_coarse_cache_var = _Value(reuse)
        panel.include_final_motion_var = _Value(False)
        panel.connection_provider = lambda: {
            "ip": "127.0.0.1", "port": 30004, "user": "a",
            "password": "b", "timeout_ms": 1000,
        }
        return panel

    def test_gui_cache_switch_is_only_emitted_for_old_two_stage(self) -> None:
        panel = self._panel_for_command(True)
        self.assertIn("--reuse-coarse-cache", panel._build_command("two_stage"))
        panel.reuse_coarse_cache_var = _Value(False)
        self.assertIn("--no-reuse-coarse-cache", panel._build_command("two_stage"))
        panel.reuse_persistent_coarse_cache_var = _Value(False)
        self.assertIn(
            "--no-reuse-persistent-coarse-cache",
            panel._build_command("two_stage"),
        )

        cad_command = self._panel_for_command(True)._build_command("cad_motion")
        self.assertNotIn("--reuse-coarse-cache", cad_command)
        self.assertNotIn("--no-reuse-coarse-cache", cad_command)
        self.assertNotIn("--reuse-persistent-coarse-cache", cad_command)
        self.assertNotIn("--no-reuse-persistent-coarse-cache", cad_command)

    def _workflow_args(self, *, reuse: bool = True) -> object:
        return type("Args", (), {
            "execute": True,
            "reuse_coarse_cache": reuse,
            "reuse_persistent_coarse_cache": reuse,
            "confidence": 0.5,
            "speed_m_s": 0.08,
            "acc_m_s2": 0.25,
            "transit_speed_m_s": 0.15,
            "transit_acc_m_s2": 0.45,
            "approach_speed_m_s": 0.12,
            "approach_acc_m_s2": 0.35,
            "move_final_xy": False,
            "tcp_xy_offset_mm": None,
            "final_target_mode": "normal",
            "handeye": "handeye.json",
        })()

    @staticmethod
    def _workflow_hole() -> dict[str, object]:
        return {
            "hole_id": 1,
            "initial_detection": {
                "class_id": 0,
                "center": [640.0, 360.0],
                "box": [600.0, 320.0, 680.0, 400.0],
            },
            "initial_center_px": [640.0, 360.0],
            "initial_center_base_mm": [10.0, 20.0, 340.0],
            "initial_plane_normal_base": [0.0, 0.0, -1.0],
            "pointcloud_segmentation": "yolo_box_annular_depth_ring",
        }

    def _workflow_cfg(self):
        return localization.TwoStageConfig(
            coarse_frames=10,
            min_coarse_valid=8,
            fine_frames=1,
            min_fine_valid=1,
            fine_stable_min_frames=1,
            fine_retry_count=0,
            enable_tilt_center_correction=False,
        )

    @staticmethod
    def _fine_recovery(intrinsics: CameraIntrinsics) -> dict[str, object]:
        return {
            "success": True,
            "fine": {
                "center_px": np.array([640.0, 360.0]),
                "axes_px_median": np.array([100.0, 100.0]),
                "center_source": "yolo",
                "center_source_counts": {"yolo": 1},
                "fine_quality_status": "strict",
                "fine_quality_note": None,
                "fine_recovery_attempts": [],
            },
            "intrinsics": intrinsics,
            "observations": [],
            "attempts": [],
            "error": None,
        }

    def test_cache_acceptance_skips_full_coarse_burst(self) -> None:
        entry = _entry()
        validation = validate_cache_entry(
            entry, self._measurements(), current_T_base_camera=np.eye(4),
            intrinsics=self.intrinsics, gates=self.gates,
        )
        cache_observations = [
            localization.Observation("cache_validation", index, np.array([640.0, 360.0]))
            for index in range(3)
        ]
        report = {
            "camera": {},
            "stages": {},
            "coarse_cache": {
                "cache_built": [1], "cache_reused": [],
                "cache_validation_failed": [], "full_coarse_fallback": [],
            },
        }
        handeye = type(
            "Handeye",
            (),
            {"T_tcp_rgb_camera": np.eye(4), "validated_for_motion": False},
        )()
        fine_recovery = self._fine_recovery(self.intrinsics)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(localization, "_move_to_sequential_coarse_pose", return_value=np.eye(4)), \
                patch.object(localization, "load_cache_entries", return_value={1: entry}), \
                patch.object(
                    localization, "_validate_coarse_cache_at_current_pose",
                    return_value=(validation, cache_observations, np.array([640.0, 360.0])),
                ) as live_validation, \
                patch.object(localization, "_capture_coarse_burst") as full_coarse, \
                patch.object(
                    localization, "_capture_fine_with_recovery",
                    return_value=fine_recovery,
                ), \
                patch.object(
                    localization, "_require_safe_snapshot",
                    return_value=(
                        {"power_on": True, "steady": True, "collision": False},
                        np.eye(4),
                    ),
                ), \
                patch.object(localization, "camera_height_to_plane_mm", return_value=260.0):
            result = localization._run_sequential_hole_workflow(
                self._workflow_args(), handeye, object(), self._workflow_cfg(),
                Path(directory), report, localization.TimingRecorder(), [],
                {"rgbd_pipeline": object(), "align": object(), "chain": object()},
                object(), None, np.eye(4), [self._workflow_hole()], self.intrinsics,
                coarse_cache_entries={1: entry},
                coarse_cache_dir=Path(directory) / "coarse_cache",
                coarse_cache_gates=self.gates,
                coarse_cache_metadata={},
            )

        self.assertEqual(result, 0)
        full_coarse.assert_not_called()
        live_validation.assert_not_called()
        self.assertEqual(report["coarse_cache"]["cache_reused"], [1])
        self.assertTrue(
            report["stages"]["hole_1"]["coarse_cache_event"]["cache_validation_skipped"]
        )

    def test_cache_rejection_runs_full_coarse_and_refreshes_entry(self) -> None:
        entry = _entry()
        plane = localization.PlaneEstimate(
            point_camera_mm=np.array([10.0, 20.0, 340.0]),
            normal_camera=np.array([0.0, 0.0, -1.0]),
            rmse_mm=1.0,
            ring_points=100,
            surface_plane_point_camera_mm=np.array([10.0, 20.0, 338.0]),
            points_camera_mm=np.array([[0.0, 0.0, 340.0], [1.0, 0.0, 340.0]]),
        )
        coarse_observations = [
            localization.Observation(
                "full_coarse", index, np.array([640.0, 360.0]),
                plane=plane, tracking_distance_px=5.0,
            )
            for index in range(8)
        ]
        refreshed = _entry()
        report = {
            "camera": {},
            "stages": {},
            "coarse_cache": {
                "cache_built": [1], "cache_reused": [],
                "cache_validation_failed": [], "full_coarse_fallback": [],
            },
        }
        handeye = type(
            "Handeye",
            (),
            {"T_tcp_rgb_camera": np.eye(4), "validated_for_motion": False},
        )()
        fine_recovery = self._fine_recovery(self.intrinsics)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(localization, "_move_to_sequential_coarse_pose", return_value=np.eye(4)), \
                patch.object(localization, "load_cache_entries", return_value={}), \
                patch.object(
                    localization, "_capture_coarse_burst",
                    return_value=(coarse_observations, np.zeros((8, 8, 3), dtype=np.uint8)),
                ) as full_coarse, \
                patch.object(
                    localization, "_cache_entry_from_observations", return_value=refreshed,
                ), \
                patch.object(localization, "save_cache_entries") as save_cache, \
                patch.object(
                    localization, "_capture_fine_with_recovery",
                    return_value=fine_recovery,
                ), \
                patch.object(
                    localization, "_require_safe_snapshot",
                    return_value=(
                        {"power_on": True, "steady": True, "collision": False},
                        np.eye(4),
                    ),
                ), \
                patch.object(localization, "camera_height_to_plane_mm", return_value=260.0):
            result = localization._run_sequential_hole_workflow(
                self._workflow_args(), handeye, object(), self._workflow_cfg(),
                Path(directory), report, localization.TimingRecorder(), [],
                {"rgbd_pipeline": object(), "align": object(), "chain": object()},
                object(), None, np.eye(4), [self._workflow_hole()], self.intrinsics,
                coarse_cache_entries={1: entry},
                coarse_cache_dir=Path(directory) / "coarse_cache",
                coarse_cache_gates=self.gates,
                coarse_cache_metadata={},
            )

        self.assertEqual(result, 0)
        full_coarse.assert_called_once()
        save_cache.assert_called_once()
        self.assertEqual(report["coarse_cache"]["cache_validation_failed"][0]["hole_id"], 1)
        self.assertEqual(report["coarse_cache"]["full_coarse_fallback"][0]["hole_id"], 1)

    def test_persistent_cache_acceptance_skips_full_coarse_burst(self) -> None:
        entry = _entry()
        validation = validate_cache_entry(
            entry, self._measurements(), current_T_base_camera=np.eye(4),
            intrinsics=self.intrinsics, gates=self.gates,
        )
        cache_observations = [
            localization.Observation("persistent_validation", index, np.array([640.0, 360.0]))
            for index in range(3)
        ]
        report = {
            "camera": {"serial_number": "camera-1"},
            "stages": {},
            "coarse_cache": {
                "cache_built": [1], "cache_reused": [],
                "persistent_cache_reused": [],
                "cache_validation_failed": [], "full_coarse_fallback": [],
            },
        }
        handeye = type(
            "Handeye",
            (),
            {"T_tcp_rgb_camera": np.eye(4), "validated_for_motion": False},
        )()
        fine_recovery = self._fine_recovery(self.intrinsics)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(localization, "_move_to_sequential_coarse_pose", return_value=np.eye(4)), \
                patch.object(localization, "load_persistent_cache_entries", return_value={1: entry}), \
                patch.object(localization, "load_cache_entries", side_effect=AssertionError("must not read current cache")), \
                patch.object(
                    localization, "_validate_coarse_cache_at_current_pose",
                    return_value=(validation, cache_observations, np.array([640.0, 360.0])),
                ), \
                patch.object(localization, "_capture_coarse_burst") as full_coarse, \
                patch.object(
                    localization, "_capture_fine_with_recovery",
                    return_value=fine_recovery,
                ), \
                patch.object(
                    localization, "_require_safe_snapshot",
                    return_value=(
                        {"power_on": True, "steady": True, "collision": False},
                        np.eye(4),
                    ),
                ), \
                patch.object(localization, "camera_height_to_plane_mm", return_value=260.0):
            result = localization._run_sequential_hole_workflow(
                self._workflow_args(), handeye, object(), self._workflow_cfg(),
                Path(directory), report, localization.TimingRecorder(), [],
                {"rgbd_pipeline": object(), "align": object(), "chain": object()},
                object(), None, np.eye(4), [self._workflow_hole()], self.intrinsics,
                coarse_cache_entries={1: entry},
                coarse_cache_sources={1: "persistent_base_cache"},
                coarse_cache_source_ids={1: 1},
                coarse_cache_dir=Path(directory) / "coarse_cache",
                coarse_cache_gates=self.gates,
                coarse_cache_metadata={},
                persistent_cache_entries={1: entry},
                persistent_cache_dir=Path(directory) / "persistent_cache",
                persistent_cache_metadata={},
            )

        self.assertEqual(result, 0)
        full_coarse.assert_not_called()
        self.assertEqual(report["coarse_cache"]["persistent_cache_reused"], [1])
        self.assertEqual(
            report["stages"]["hole_1"]["coarse_cache_event"]["cache_source"],
            "persistent_base_cache",
        )

    def test_legacy_current_run_cache_can_seed_persistent_cache(self) -> None:
        entry = _entry()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_dir = root / "runs" / "two-stage-legacy" / "coarse_cache"
            save_cache_entries(legacy_dir, {1: entry})
            with patch.object(localization, "RUNS_DIR", root / "runs"):
                loaded, errors, audit = localization._load_persistent_coarse_cache_for_run(
                    run_dir=root / "runs" / "two-stage-current",
                    camera_serial="camera-1",
                    handeye=type("Handeye", (), {"T_tcp_rgb_camera": np.eye(4)})(),
                    handeye_path="handeye.json",
                    intrinsics=self.intrinsics,
                    persistent_cache_dir=root / "persistent",
                )
        self.assertEqual(sorted(loaded), [1])
        self.assertEqual(errors, {})
        self.assertEqual(audit["bootstrap_source"], str(legacy_dir))

    def test_loader_rejects_cache_from_previous_surface_selection_policy(self) -> None:
        entry = _entry()
        with tempfile.TemporaryDirectory() as directory:
            persistent_dir = Path(directory) / "persistent"
            save_persistent_cache_entries(persistent_dir, {1: entry}, metadata={})
            loaded, errors, audit = localization._load_persistent_coarse_cache_for_run(
                run_dir=Path(directory) / "current",
                camera_serial="camera-1",
                handeye=type("Handeye", (), {"T_tcp_rgb_camera": np.eye(4)})(),
                handeye_path="handeye.json",
                intrinsics=self.intrinsics,
                persistent_cache_dir=persistent_dir,
                required_surface_model=localization.COARSE_SURFACE_MODEL,
            )

        self.assertEqual(loaded, {})
        self.assertEqual(errors, {})
        self.assertIn("1", audit["compatibility_rejected"])
        self.assertIn(
            "surface_selection_policy_mismatch",
            audit["compatibility_rejected"]["1"],
        )


if __name__ == "__main__":
    unittest.main()
