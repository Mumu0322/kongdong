from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aubo_workbench.gui_hole_localization import HoleLocalizationPanel


class _Value:
    def __init__(self, value: object) -> None:
        self.value = value

    def get(self) -> object:
        return self.value


class HoleLocalizationCommandTests(unittest.TestCase):
    def _map_execution_command(
        self, *, allow_experimental: bool, final_xy_checked: bool = True,
        hole_ids: str = "", strategy: str | None = None,
        mode: str = "hole_map_execute", map_build_mode: str | None = None,
        direct_height: str = "340", direct_group_size: str = "5",
        direct_extra_frames: str = "5", direct_settle_delay: str = "1.0",
        direct_capture_only: bool = False,
        in_group_pose: bool = True, in_group_adjustments: str = "1",
        in_group_xy: str = "8.0", in_group_z: str = "3.0",
        in_group_rotation: str = "2.0", in_group_normal_holes: str = "2",
        in_group_normal_spread: str = "3.0",
        per_hole_fine_safe_z_margin: float = 20.0,
    ) -> list[str]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        model = root / "model.pt"
        handeye = root / "handeye.json"
        hole_map = root / "hole_map.json"
        model.touch()
        handeye.write_text("{}", encoding="utf-8")
        hole_map.write_text("{}", encoding="utf-8")

        panel = HoleLocalizationPanel.__new__(HoleLocalizationPanel)
        panel.model_var = _Value(str(model))
        panel.handeye_var = _Value(str(handeye))
        panel.execute_var = _Value(True)
        panel.experimental_var = _Value(allow_experimental)
        panel.final_xy_var = _Value(final_xy_checked)
        panel.charuco_xy_var = _Value(True)
        panel.hole_map_path_var = _Value(str(hole_map))
        panel.sector_id_var = _Value("1")
        panel.hole_map_ids_var = _Value(hole_ids)
        panel.coarse_direct_final_height_var = _Value(direct_height)
        panel.coarse_direct_final_max_group_size_var = _Value(direct_group_size)
        panel.coarse_direct_final_early_stop_extra_frames_var = _Value(direct_extra_frames)
        panel.coarse_direct_final_settle_delay_var = _Value(direct_settle_delay)
        panel.coarse_direct_final_capture_only_var = _Value(direct_capture_only)
        panel.batch_fine_in_group_pose_adjustment_var = _Value(in_group_pose)
        panel.batch_fine_in_group_max_adjustments_var = _Value(in_group_adjustments)
        panel.batch_fine_in_group_max_xy_var = _Value(in_group_xy)
        panel.batch_fine_in_group_max_z_var = _Value(in_group_z)
        panel.batch_fine_in_group_max_rotation_var = _Value(in_group_rotation)
        panel.batch_fine_in_group_min_normal_holes_var = _Value(
            in_group_normal_holes
        )
        panel.batch_fine_in_group_max_normal_spread_var = _Value(
            in_group_normal_spread
        )
        if map_build_mode is not None:
            panel.map_build_localization_mode_var = _Value(map_build_mode)
        if strategy is not None:
            panel.strategy_var = _Value(strategy)
        panel.connection_provider = lambda: {
            "ip": "127.0.0.1",
            "port": 30004,
            "user": "AUBO",
            "password": "",
            "timeout_ms": 3000,
        }
        panel._numbers = lambda: {
            "confidence": 0.35,
            "coarse_height": 340.0,
            "fine_height": 260.0,
            "per_hole_fine_safe_z_margin": per_hole_fine_safe_z_margin,
            "coarse_frames": 15,
            "fine_frames": 30,
            "speed": 0.08,
            "acc": 0.25,
            "transit_speed": 0.15,
            "transit_acc": 0.45,
            "approach_speed": 0.12,
            "approach_acc": 0.35,
        }
        return panel._build_command(mode)

    def test_map_execution_passes_handeye_motion_policy(self) -> None:
        command = self._map_execution_command(allow_experimental=True)

        self.assertIn("--allow-experimental-handeye", command)
        self.assertIn("--move-final-xy", command)
        self.assertIn("--use-charuco-xy-correction", command)
        self.assertNotIn("--final-target-mode", command)
        margin_index = command.index("--per-hole-fine-safe-z-margin-mm")
        self.assertEqual(command[margin_index + 1], "20.0")

    def test_shared_fine_in_group_pose_settings_are_passed(self) -> None:
        command = self._map_execution_command(
            allow_experimental=True,
            mode="two_stage",
            strategy="cache",
            in_group_adjustments="1",
            in_group_xy="7.5",
            in_group_z="2.5",
            in_group_rotation="1.5",
            in_group_normal_holes="3",
            in_group_normal_spread="2.5",
        )

        def value_after(flag: str) -> str:
            return command[command.index(flag) + 1]

        self.assertIn("--batch-fine-in-group-pose-adjustment", command)
        self.assertEqual(value_after("--batch-fine-in-group-max-adjustments"), "1")
        self.assertEqual(value_after("--batch-fine-in-group-max-xy-mm"), "7.5")
        self.assertEqual(value_after("--batch-fine-in-group-max-z-mm"), "2.5")
        self.assertEqual(
            value_after("--batch-fine-in-group-max-rotation-deg"), "1.5",
        )
        self.assertEqual(
            value_after("--batch-fine-in-group-min-normal-holes"), "3",
        )
        self.assertEqual(
            value_after("--batch-fine-in-group-max-normal-spread-deg"), "2.5",
        )

    def test_shared_fine_in_group_pose_can_be_disabled(self) -> None:
        command = self._map_execution_command(
            allow_experimental=True,
            mode="two_stage",
            strategy="cache",
            in_group_pose=False,
        )

        self.assertIn("--no-batch-fine-in-group-pose-adjustment", command)
        self.assertNotIn("--batch-fine-in-group-pose-adjustment", command)

    def test_map_build_per_hole_mode_disables_shared_captures(self) -> None:
        command = self._map_execution_command(
            allow_experimental=True,
            mode="hole_map_build",
            map_build_mode="per_hole",
        )
        self.assertEqual(
            command[command.index("--map-build-localization-mode") + 1],
            "per_hole",
        )
        self.assertIn("--no-batch-coarse-localization", command)
        self.assertIn("--no-batch-fine-localization", command)
        self.assertIn("--no-batch-fine-joint-localization", command)
        self.assertIn("--no-batch-fine-pointcloud-xy-fusion", command)
        self.assertIn("--no-move-final-xy", command)

    def test_map_build_same_capture_340_is_separate_mode(self) -> None:
        command = self._map_execution_command(
            allow_experimental=True,
            mode="hole_map_build",
            map_build_mode="same_capture_340",
        )
        self.assertEqual(
            command[command.index("--map-build-localization-mode") + 1],
            "same_capture_340",
        )
        self.assertIn("--no-batch-coarse-localization", command)
        self.assertIn("--no-batch-fine-localization", command)
        self.assertIn("--no-move-final-xy", command)

    def test_map_execution_always_moves_successful_fine_result_to_final_point(self) -> None:
        command = self._map_execution_command(
            allow_experimental=True, final_xy_checked=False,
        )

        self.assertIn("--move-final-xy", command)
        self.assertNotIn("--no-move-final-xy", command)

    def test_map_execution_requires_explicit_experimental_handeye(self) -> None:
        command = self._map_execution_command(allow_experimental=False)

        self.assertNotIn("--allow-experimental-handeye", command)

    def test_visual_selection_order_is_passed_to_map_execution(self) -> None:
        command = self._map_execution_command(
            allow_experimental=True, hole_ids="12 3 27",
        )

        index = command.index("--hole-ids")
        self.assertEqual(command[index + 1:index + 4], ["12", "3", "27"])

    def test_map_execution_coarse_direct_strategy_uses_pointcloud_only(self) -> None:
        command = self._map_execution_command(
            allow_experimental=True, strategy="coarse_direct",
        )

        self.assertIn("--coarse-direct-final", command)
        self.assertIn("--no-batch-fine-localization", command)
        self.assertIn("--no-batch-fine-joint-localization", command)
        self.assertIn("--no-batch-fine-pointcloud-xy-fusion", command)
        for obsolete_flag in (
            "--coarse-direct-min-valid-frames",
            "--coarse-direct-max-center-scatter-p95-px",
            "--coarse-direct-max-tracking-distance-p95-px",
            "--coarse-direct-max-plane-rmse-mm",
            "--coarse-direct-same-pose-fusion",
            "--coarse-direct-same-pose-min-geometric-frames",
            "--coarse-direct-same-pose-max-geometric-scatter-p95-px",
            "--no-coarse-direct-center-recheck",
            "--coarse-direct-max-reference-xy-delta-mm",
        ):
            self.assertNotIn(obsolete_flag, command)

    def test_map_execution_coarse_direct_strategy_passes_configured_height_group_and_settle_delay(self) -> None:
        command = self._map_execution_command(
            allow_experimental=True,
            strategy="coarse_direct",
            direct_height="320",
            direct_group_size="3",
            direct_extra_frames="5",
            direct_settle_delay="2.5",
            direct_capture_only=True,
        )

        def value_after(flag: str) -> str:
            return command[command.index(flag) + 1]

        self.assertEqual(value_after("--coarse-direct-final-height-mm"), "320.0")
        self.assertEqual(value_after("--coarse-direct-final-max-group-size"), "3")
        self.assertEqual(
            value_after("--coarse-direct-final-early-stop-extra-frames"), "5",
        )
        self.assertEqual(value_after("--coarse-direct-final-settle-delay-s"), "2.5")
        self.assertIn("--coarse-direct-final-capture-only", command)
        self.assertIn("--no-move-final-xy", command)

    def test_map_repair_forces_single_hole_safe_flags(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        model = root / "model.pt"
        handeye = root / "handeye.json"
        hole_map = root / "hole_map.json"
        model.touch()
        handeye.write_text("{}", encoding="utf-8")
        hole_map.write_text("{}", encoding="utf-8")
        panel = HoleLocalizationPanel.__new__(HoleLocalizationPanel)
        panel.model_var = _Value(str(model))
        panel.handeye_var = _Value(str(handeye))
        panel.execute_var = _Value(True)
        panel.experimental_var = _Value(False)
        panel.hole_map_path_var = _Value(str(hole_map))
        panel.sector_id_var = _Value("2")
        panel.repair_hole_id_var = _Value("19")
        panel.strategy_var = _Value("batch")
        panel.final_xy_var = _Value(True)
        panel.offset_radii_var = _Value("0")
        panel.offset_angles_var = _Value("0")
        panel.connection_provider = lambda: {
            "ip": "127.0.0.1", "port": 30004, "user": "AUBO",
            "password": "", "timeout_ms": 3000,
        }
        panel._numbers = lambda: {
            "confidence": 0.35, "coarse_height": 340.0, "fine_height": 260.0,
            "coarse_frames": 15, "fine_frames": 30, "speed": 0.08,
            "acc": 0.25, "transit_speed": 0.15, "transit_acc": 0.45,
            "approach_speed": 0.12, "approach_acc": 0.35,
        }
        command = panel._build_command("hole_map_repair")
        self.assertIn("--hole-map-mode", command)
        self.assertIn("repair", command)
        self.assertIn("--repair-hole-id", command)
        self.assertIn("19", command)
        for flag in (
            "--no-batch-fine-localization",
            "--no-batch-fine-joint-localization",
            "--no-batch-fine-pointcloud-xy-fusion",
            "--no-reuse-coarse-cache",
            "--no-reuse-persistent-coarse-cache",
            "--no-move-final-xy",
        ):
            self.assertIn(flag, command)


if __name__ == "__main__":
    unittest.main()
