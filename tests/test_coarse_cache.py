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

    def set(self, value):
        self.value = value


class _FakeWidget:
    def __init__(self):
        self.configured = []

    def configure(self, **kwargs):
        self.configured.append(kwargs)


class _FakeStdin:
    def __init__(self):
        self.writes = []

    def write(self, value):
        self.writes.append(value)

    def flush(self):
        pass


class _FakeProcess:
    def __init__(self):
        self.stdin = _FakeStdin()

    def poll(self):
        return None


def _entry(*, hole_id: int = 1, x_mm: float = 10.0) -> CoarseCacheEntry:
    return CoarseCacheEntry(
        hole_id=hole_id,
        T_base_camera_build=np.eye(4),
        T_tcp_camera=np.eye(4),
        tcp_pose_m_rad=[0.0] * 6,
        camera_serial="camera-1",
        handeye_path="handeye.json",
        intrinsics=CameraIntrinsics(1280, 720, 800.0, 800.0, 640.0, 360.0, ()).as_dict(),
        center_px=np.array([640.0, 360.0]),
        point_camera_mm=np.array([x_mm, 20.0, 340.0]),
        plane_point_camera_mm=np.array([x_mm, 20.0, 338.0]),
        normal_camera=np.array([0.0, 0.0, -1.0]),
        point_base_mm=np.array([x_mm, 20.0, 340.0]),
        plane_point_base_mm=np.array([x_mm, 20.0, 338.0]),
        normal_base=np.array([0.0, 0.0, -1.0]),
        plane_rmse_mm=1.0,
        cache_source="batch_coarse_source",
        valid_frames=5,
        total_frames=5,
        center_scatter_p95_px=0.2,
        ring_points_median=120.0,
        surface_model="local_tangent_plane",
        frame_indices=np.array([0, 1, 2, 3, 4]),
        frame_centers_px=np.array([
            [640.0, 360.0], [640.1, 360.0], [639.9, 360.1],
            [640.0, 359.9], [640.1, 360.1],
        ]),
        frame_plane_rmse_mm=np.array([1.0, 1.1, 0.9, 1.0, 1.05]),
        points_camera_mm_by_frame=(
            np.array([[0.0, 0.0, 340.0], [1.0, 0.0, 340.0]]),
            np.array([[0.0, 1.0, 340.0], [1.0, 1.0, 340.0]]),
            np.array([[0.0, 0.0, 339.9], [1.0, 0.0, 340.1]]),
            np.array([[0.0, 0.0, 340.0], [1.0, 0.0, 340.0]]),
            np.array([[0.0, 1.0, 340.0], [1.0, 1.0, 340.0]]),
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
            for offset in (0.0, 0.1, -0.1, 0.05, -0.05)
        ]

    def test_save_and_load_npz_json_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = save_cache_entries(directory, {1: _entry()}, metadata={"test": True})
            self.assertEqual(path, Path(directory) / "manifest.json")
            loaded = load_cache_entries(directory)
            self.assertEqual(sorted(loaded), [1])
            np.testing.assert_allclose(loaded[1].point_base_mm, [10.0, 20.0, 340.0])
            self.assertEqual(len(loaded[1].points_camera_mm_by_frame), 5)
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

    def test_batch_source_persistent_cache_survives_a_new_run(self) -> None:
        """正式一拍多缓存应能被下一轮按 base 坐标加载。"""
        with tempfile.TemporaryDirectory() as directory:
            persistent_dir = Path(directory) / "persistent"
            source_entry = _entry(hole_id=17, x_mm=123.0)
            save_persistent_cache_entries(
                persistent_dir,
                {17: source_entry},
                metadata={"source": "batch_coarse_source"},
            )
            loaded, errors, audit = localization._load_persistent_coarse_cache_for_run(
                run_dir=Path(directory) / "run-02",
                camera_serial="camera-1",
                handeye=type("Handeye", (), {"T_tcp_rgb_camera": np.eye(4)})(),
                handeye_path="handeye.json",
                intrinsics=self.intrinsics,
                persistent_cache_dir=persistent_dir,
                required_surface_model=None,
            )

        self.assertEqual(errors, {})
        self.assertEqual(sorted(loaded), [17])
        self.assertEqual(loaded[17].cache_source, "batch_coarse_source")
        self.assertEqual(audit["loaded_count"], 1)

    def test_batch_view_groups_split_when_selection_exceeds_one_view(self) -> None:
        holes = [
            {
                "hole_id": hole_id,
                "initial_center_base_mm": np.array([x_mm, 0.0, 340.0]),
                "initial_plane_normal_base": np.array([0.0, 0.0, 1.0]),
            }
            for hole_id, x_mm in ((1, 0.0), (2, 40.0), (3, 220.0), (4, 260.0))
        ]

        def plan(group, *_args):
            x_values = [float(item["initial_center_base_mm"][0]) for item in group]
            if max(x_values) - min(x_values) > 100.0:
                raise RuntimeError("out_of_view")
            return np.eye(4), {}

        with patch.object(localization, "_plan_batch_coarse_group_pose", side_effect=plan):
            groups = localization._split_shared_cache_validation_groups(
                holes,
                np.eye(4),
                type("Handeye", (), {"T_tcp_rgb_camera": np.eye(4)})(),
                0.0,
                self.intrinsics,
                340.0,
                50.0,
            )

        self.assertEqual([[item["hole_id"] for item in group] for group in groups], [[1, 2], [3, 4]])

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
            manifest["entries"]["1"]["frame_indices"] = [10, 11, 12, 13, 14]
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

    def test_validation_requires_all_five_measurement_frames(self) -> None:
        result = validate_cache_entry(
            _entry(), self._measurements()[:4],
            current_T_base_camera=np.eye(4),
            intrinsics=self.intrinsics,
            gates=self.gates,
        )
        self.assertFalse(result.accepted)
        self.assertIn("validation_frames", result.reason)

    def test_four_of_five_valid_measurements_are_accepted(self) -> None:
        measurements = self._measurements()
        measurements[4]["error"] = "yolo_missing"
        result = validate_cache_entry(
            _entry(), measurements, current_T_base_camera=np.eye(4),
            intrinsics=self.intrinsics, gates=self.gates,
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.valid_frames, 4)

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

    def test_shared_cache_cli_rejects_disabled_cache_reuse(self) -> None:
        args = localization.build_parser().parse_args([
            "--shared-cache-validation", "--no-reuse-coarse-cache",
        ])

        with self.assertRaisesRegex(ValueError, "共享缓存快速验证要求启用粗定位缓存复用"):
            localization.run_two_stage_hole_localization(args, object(), object())

    @staticmethod
    def _panel_for_command(reuse: bool) -> HoleLocalizationPanel:
        panel = HoleLocalizationPanel.__new__(HoleLocalizationPanel)
        source_file = str(Path(__file__))
        panel.model_var = _Value(source_file)
        panel.handeye_var = _Value(source_file)
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
        panel.offset_radii_var = _Value("0 5")
        panel.offset_angles_var = _Value("0 90")
        panel.batch_coarse_localization_var = _Value(False)
        panel.batch_coarse_frames_var = _Value("15")
        panel.batch_coarse_min_valid_var = _Value("10")
        panel.batch_coarse_min_holes_var = _Value("")
        panel.batch_coarse_view_margin_var = _Value("50.0")
        panel.optimize_hole_order_var = _Value(False)
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

    def test_gui_cache_strategy_forces_cross_run_persistent_reuse(self) -> None:
        panel = self._panel_for_command(False)
        panel.strategy_var = _Value("cache")
        command = panel._build_command("two_stage")
        self.assertIn("--reuse-coarse-cache", command)
        self.assertIn("--reuse-persistent-coarse-cache", command)
        self.assertNotIn("--no-reuse-persistent-coarse-cache", command)

    @staticmethod
    def _panel_for_next_hole_confirmation(auto: bool, mode: str = "two_stage") -> HoleLocalizationPanel:
        panel = HoleLocalizationPanel.__new__(HoleLocalizationPanel)
        panel.process_mode = mode
        panel.auto_next_hole_var = _Value(auto)
        panel.waiting_confirmation = False
        panel.confirmation_kind = ""
        panel.process = _FakeProcess()
        panel.confirm_btn = _FakeWidget()
        panel.mark_error_btn = _FakeWidget()
        panel.cancel_btn = _FakeWidget()
        panel.motion_var = _Value("")
        panel.status_var = _Value("")
        panel.logs = []
        panel._append_log = panel.logs.append
        return panel

    def test_gui_auto_next_hole_sends_one_continue_command(self) -> None:
        panel = self._panel_for_next_hole_confirmation(True)

        panel._handle_next_hole_confirmation("[NEXT_HOLE_CONFIRM_REQUIRED] 当前孔=1，下一个检测孔=2")

        self.assertEqual(panel.process.stdin.writes, ["m\n"])
        self.assertFalse(panel.waiting_confirmation)
        self.assertEqual(panel.confirmation_kind, "")
        self.assertIn("自动检测下一孔", "".join(panel.logs))

    def test_gui_manual_next_hole_keeps_confirmation_waiting(self) -> None:
        panel = self._panel_for_next_hole_confirmation(False)

        panel._handle_next_hole_confirmation("[NEXT_HOLE_CONFIRM_REQUIRED] 当前孔=1，下一个检测孔=2")

        self.assertEqual(panel.process.stdin.writes, [])
        self.assertTrue(panel.waiting_confirmation)
        self.assertEqual(panel.confirmation_kind, "hole")
        self.assertEqual(panel.confirm_btn.configured[-1]["state"], "normal")

    def test_gui_auto_next_hole_does_not_affect_offset_process(self) -> None:
        panel = self._panel_for_next_hole_confirmation(True, mode="offset")

        panel._handle_next_hole_confirmation("unexpected marker")

        self.assertEqual(panel.process.stdin.writes, [])
        self.assertFalse(panel.waiting_confirmation)
        self.assertIn("忽略非孔洞检测流程", "".join(panel.logs))

    def test_gui_batch_validation_only_applies_when_batch_is_enabled(self) -> None:
        panel = self._panel_for_command(True)
        panel.batch_coarse_frames_var = _Value("invalid")
        panel.batch_coarse_min_valid_var = _Value("invalid")
        panel.batch_coarse_min_holes_var = _Value("invalid")
        panel.batch_coarse_view_margin_var = _Value("invalid")
        for mode in ("two_stage", "offset"):
            self.assertIsInstance(panel._build_command(mode), list)

        panel.batch_coarse_localization_var = _Value(True)
        with self.assertRaisesRegex(ValueError, "批量粗定位参数"):
            panel._build_command("two_stage")

        invalid_cases = [
            ("0", "1", "", "50", "必须大于 0"),
            ("5", "6", "", "50", "必须不少于"),
            ("5", "4", "0", "50", "每帧最少孔数"),
            ("5", "4", "", "nan", "视野边缘余量"),
        ]
        for frames, min_valid, min_holes, margin, message in invalid_cases:
            panel.batch_coarse_frames_var = _Value(frames)
            panel.batch_coarse_min_valid_var = _Value(min_valid)
            panel.batch_coarse_min_holes_var = _Value(min_holes)
            panel.batch_coarse_view_margin_var = _Value(margin)
            with self.subTest(frames=frames, min_valid=min_valid, min_holes=min_holes, margin=margin):
                with self.assertRaisesRegex(ValueError, message):
                    panel._build_command("two_stage")

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
    def _workflow_hole(*, hole_id: int = 1, x_mm: float = 10.0) -> dict[str, object]:
        return {
            "hole_id": hole_id,
            "initial_detection": {
                "class_id": 0,
                "center": [640.0, 360.0],
                "box": [600.0, 320.0, 680.0, 400.0],
            },
            "initial_center_px": [640.0, 360.0],
            "initial_center_base_mm": [x_mm, 20.0, 340.0],
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
        args = self._workflow_args()
        args.batch_coarse_localization = True
        cfg = localization.replace(self._workflow_cfg(), batch_coarse_localization=True)
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
                    return_value=(validation, cache_observations),
                ) as live_validation, \
                patch.object(localization, "_capture_coarse_burst") as full_coarse, \
                patch.object(localization, "render_cache_cloud", return_value=None), \
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
                args, handeye, object(), cfg,
                Path(directory), report, localization.TimingRecorder(), [],
                {"rgbd_pipeline": object(), "align": object(), "chain": object()},
                object(), None, np.eye(4), [self._workflow_hole()], self.intrinsics,
                coarse_cache_entries={1: entry},
                coarse_cache_dir=Path(directory) / "coarse_cache",
                coarse_cache_gates=self.gates,
            )

        self.assertEqual(result, 0)
        full_coarse.assert_not_called()
        live_validation.assert_called_once()
        self.assertEqual(report["coarse_cache"]["cache_reused"], [1])
        self.assertFalse(
            report["stages"]["hole_1"]["coarse_cache_event"]["cache_validation_skipped"]
        )
        self.assertIn(
            "cache_validation_result", report["stages"]["hole_1"]["coarse_cache_event"]
        )
        self.assertTrue(
            report["stages"]["hole_1"]["coarse_cache_event"]["cache_validation_result"]["accepted"]
        )

    def test_shared_cache_acceptance_skips_per_hole_340mm_validation(self) -> None:
        entry = _entry()
        plane = localization.PlaneEstimate(
            point_camera_mm=np.array([10.0, 20.0, 340.0]),
            normal_camera=np.array([0.0, 0.0, -1.0]), rmse_mm=1.0,
            ring_points=100, surface_plane_point_camera_mm=np.array([10.0, 20.0, 338.0]),
            points_camera_mm=np.array([[0.0, 0.0, 340.0]]),
        )
        observations = [
            localization.Observation("shared", index, np.array([640.0, 360.0]), plane=plane,
                                     tracking_distance_px=5.0)
            for index in range(3)
        ]
        args = self._workflow_args()
        args.shared_cache_validation = True
        args.batch_coarse_localization = False
        cfg = localization.replace(
            self._workflow_cfg(), shared_cache_validation=True,
            shared_cache_validation_frames=3, shared_cache_validation_min_valid=2,
        )
        gates = CacheValidationGates(validation_frames=3, min_valid_frames=2)
        report = {"camera": {}, "stages": {}, "coarse_cache": {
            "cache_reused": [], "cache_validation_failed": [], "full_coarse_fallback": [],
        }}
        handeye = type("Handeye", (), {"T_tcp_rgb_camera": np.eye(4), "validated_for_motion": False})()
        batch_result = {1: {"observations": observations}, "_batch_metadata": {}}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(
                    localization, "_confirm_and_move_line",
                    side_effect=lambda _label, _current, target, *_args, **_kwargs: target,
                ), \
                patch.object(localization, "_batch_coarse_localization_at_340mm", return_value=batch_result), \
                patch.object(localization, "_move_to_sequential_coarse_pose", return_value=np.eye(4)) as per_hole_move, \
                patch.object(localization, "_validate_coarse_cache_at_current_pose") as per_hole_validate, \
                patch.object(localization, "_move_to_fine_pose", return_value=np.eye(4)), \
                patch.object(localization, "_capture_fine_with_recovery", return_value=self._fine_recovery(self.intrinsics)), \
                patch.object(localization, "_require_safe_snapshot", return_value=({"power_on": True, "steady": True, "collision": False}, np.eye(4))), \
                patch.object(localization, "camera_height_to_plane_mm", return_value=260.0):
            result = localization._run_sequential_hole_workflow(
                args, handeye, object(), cfg, Path(directory), report, localization.TimingRecorder(), [],
                {"rgbd_pipeline": object(), "align": object(), "chain": object()}, object(), None,
                np.eye(4), [self._workflow_hole()], self.intrinsics,
                coarse_cache_entries={1: entry}, coarse_cache_dir=Path(directory) / "coarse_cache",
                coarse_cache_gates=gates,
            )
        self.assertEqual(result, 0)
        per_hole_move.assert_not_called()
        per_hole_validate.assert_not_called()
        self.assertTrue(report["stages"]["hole_1"]["coarse_cache_event"]["shared_cache_validation"])

    def test_shared_multi_hole_validation_keeps_off_axis_hole_and_falls_back_only_failed_hole(self) -> None:
        entries = {1: _entry(hole_id=1, x_mm=10.0), 2: _entry(hole_id=2, x_mm=60.0)}

        def observations(x_mm: float, center_x: float, *, valid: bool) -> list[localization.Observation]:
            plane = localization.PlaneEstimate(
                point_camera_mm=np.array([x_mm, 20.0, 340.0]),
                normal_camera=np.array([0.0, 0.0, -1.0]), rmse_mm=1.0,
                ring_points=100,
                surface_plane_point_camera_mm=np.array([x_mm, 20.0, 338.0]),
                points_camera_mm=np.array([[x_mm, 20.0, 340.0]]),
            )
            return [
                localization.Observation(
                    "shared", index, np.array([center_x, 360.0]), plane=plane,
                    tracking_distance_px=5.0, error=None if valid else "yolo_missing",
                )
                for index in range(3)
            ]

        args = self._workflow_args()
        args.shared_cache_validation = True
        args.batch_coarse_localization = False
        cfg = localization.replace(
            self._workflow_cfg(), coarse_settle_delay_s=0.0,
            shared_cache_validation=True, shared_cache_validation_frames=3,
            shared_cache_validation_min_valid=2,
        )
        report = {"camera": {}, "stages": {}, "coarse_cache": {
            "cache_reused": [], "cache_validation_failed": [], "full_coarse_fallback": [],
        }}
        batch_result = {
            1: {"observations": observations(10.0, 580.0, valid=True)},
            2: {"observations": observations(60.0, 700.0, valid=False)},
            "_batch_metadata": {},
        }
        geometry = {"target_tcp_pose_m_rad": [0.0] * 6, "projected_holes_px": {
            1: np.array([580.0, 360.0]), 2: np.array([700.0, 360.0]),
        }}
        handeye = type("Handeye", (), {"T_tcp_rgb_camera": np.eye(4), "validated_for_motion": False})()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(localization, "_split_shared_cache_validation_groups", side_effect=lambda holes, *_args: [holes]), \
                patch.object(localization, "_plan_batch_coarse_group_pose", return_value=(np.eye(4), geometry)), \
                patch.object(localization, "_confirm_and_move_line", return_value=np.eye(4)), \
                patch.object(localization, "_batch_coarse_localization_at_340mm", return_value=batch_result), \
                patch.object(localization, "_project_base_point_to_pixel", side_effect=lambda point, *_args: np.array([580.0 if point[0] < 30.0 else 700.0, 360.0])), \
                patch.object(localization, "_move_to_fine_pose", return_value=np.eye(4)), \
                patch.object(localization, "_capture_fine_with_recovery", return_value=self._fine_recovery(self.intrinsics)), \
                patch.object(localization, "_require_safe_snapshot", return_value=({}, np.eye(4))), \
                patch.object(localization, "camera_height_to_plane_mm", return_value=260.0), \
                patch.object(localization, "_request_next_hole_confirmation", return_value="m"), \
                patch.object(localization, "_move_to_sequential_coarse_pose", side_effect=RuntimeError("failed-hole-fallback")) as per_hole_move:
            with self.assertRaisesRegex(RuntimeError, "failed-hole-fallback"):
                localization._run_sequential_hole_workflow(
                    args, handeye, object(), cfg, Path(directory), report, localization.TimingRecorder(), [],
                    {"rgbd_pipeline": object(), "align": object(), "chain": object()}, object(), None,
                    np.eye(4), [self._workflow_hole(hole_id=1, x_mm=10.0), self._workflow_hole(hole_id=2, x_mm=60.0)], self.intrinsics,
                    coarse_cache_entries=entries, coarse_cache_dir=Path(directory) / "coarse_cache",
                    coarse_cache_gates=CacheValidationGates(validation_frames=3, min_valid_frames=2),
                )

        self.assertEqual(per_hole_move.call_args.args[0], 2)
        self.assertTrue(report["stages"]["shared_cache_validation"]["holes"]["1"]["accepted"])
        self.assertFalse(report["stages"]["shared_cache_validation"]["holes"]["2"]["accepted"])

    def test_cache_rejection_runs_full_coarse_without_refreshing_entry(self) -> None:
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
            for index in range(10)
        ]
        rejected_validation = validate_cache_entry(
            entry, self._measurements(tracking=71.0),
            current_T_base_camera=np.eye(4),
            intrinsics=self.intrinsics,
            gates=self.gates,
        )
        args = self._workflow_args()
        args.batch_coarse_localization = True
        cfg = localization.replace(self._workflow_cfg(), batch_coarse_localization=True)
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
                    return_value=(
                        rejected_validation,
                        [
                            localization.Observation(
                                "cache_validation", index,
                                np.array([640.0, 360.0]),
                                tracking_distance_px=71.0,
                                error="tracking_distance",
                            )
                            for index in range(5)
                        ],
                    ),
                ) as live_validation, \
                patch.object(
                    localization, "_capture_coarse_burst",
                    return_value=(coarse_observations, np.zeros((8, 8, 3), dtype=np.uint8)),
                ) as full_coarse, \
                patch.object(localization, "save_cache_entries") as save_cache, \
                patch.object(localization, "render_cache_cloud", return_value=None), \
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
                args, handeye, object(), cfg,
                Path(directory), report, localization.TimingRecorder(), [],
                {"rgbd_pipeline": object(), "align": object(), "chain": object()},
                object(), None, np.eye(4), [self._workflow_hole()], self.intrinsics,
                coarse_cache_entries={1: entry},
                coarse_cache_dir=Path(directory) / "coarse_cache",
                coarse_cache_gates=self.gates,
            )

        self.assertEqual(result, 0)
        full_coarse.assert_called_once()
        live_validation.assert_called_once()
        save_cache.assert_not_called()
        self.assertEqual(report["coarse_cache"]["cache_validation_failed"][0]["hole_id"], 1)
        self.assertEqual(report["coarse_cache"]["full_coarse_fallback"][0]["hole_id"], 1)
        self.assertEqual(
            report["stages"]["hole_1"]["coarse_cache_event"]["cache_refresh_skipped"],
            "per_hole_fallback_not_cache_source",
        )

    def test_persistent_cache_acceptance_skips_full_coarse_burst(self) -> None:
        entry = _entry()
        args = self._workflow_args()
        args.batch_coarse_localization = True
        cfg = localization.replace(self._workflow_cfg(), batch_coarse_localization=True)
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
                    return_value=(validation, cache_observations),
                ) as live_validation, \
                patch.object(localization, "_capture_coarse_burst") as full_coarse, \
                patch.object(localization, "render_cache_cloud", return_value=None), \
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
                args, handeye, object(), cfg,
                Path(directory), report, localization.TimingRecorder(), [],
                {"rgbd_pipeline": object(), "align": object(), "chain": object()},
                object(), None, np.eye(4), [self._workflow_hole()], self.intrinsics,
                coarse_cache_entries={1: entry},
                coarse_cache_sources={1: "persistent_base_cache"},
                coarse_cache_source_ids={1: 1},
                coarse_cache_dir=Path(directory) / "coarse_cache",
                coarse_cache_gates=self.gates,
                persistent_cache_entries={1: entry},
                persistent_cache_dir=Path(directory) / "persistent_cache",
            )

        self.assertEqual(result, 0)
        full_coarse.assert_not_called()
        live_validation.assert_called_once()
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
