import unittest

import numpy as np

import run_coarse_to_fine_offset_test as offset_test
from aubo_workbench.gui_hole_localization import HoleLocalizationPanel


class _Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class CoarseToFineOffsetTests(unittest.TestCase):
    def test_default_plan_has_one_center_and_eight_directions_per_ring(self):
        plan = offset_test.build_offset_plan()
        self.assertEqual(len(plan), 33)
        counts = {}
        for item in plan:
            counts[item["radius_mm"]] = counts.get(item["radius_mm"], 0) + 1
        self.assertEqual(counts, {0.0: 1, 5.0: 8, 10.0: 8, 15.0: 8, 20.0: 8})
        self.assertEqual([item["sample_id"] for item in plan], list(range(1, 34)))

    def test_camera_offset_moves_tcp_in_the_opposite_direction(self):
        reference = np.eye(4)
        target, delta = offset_test.plan_camera_plane_offset_tcp(
            reference, np.eye(4), [5.0, -3.0],
        )
        np.testing.assert_allclose(delta, [-5.0, 3.0, 0.0])
        np.testing.assert_allclose(target[:3, 3], [-5.0, 3.0, 0.0])
        np.testing.assert_allclose(target[:3, :3], reference[:3, :3])

    def test_camera_offset_respects_reference_tcp_orientation(self):
        reference = np.eye(4)
        reference[:3, :3] = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        target, delta = offset_test.plan_camera_plane_offset_tcp(
            reference, np.eye(4), [5.0, 0.0],
        )
        np.testing.assert_allclose(delta, [0.0, -5.0, 0.0])
        np.testing.assert_allclose(target[:3, :3], reference[:3, :3])

    def test_summary_requires_all_directions_at_a_radius(self):
        results = [
            {"radius_mm": 0.0, "accepted_pass": True, "strict_pass": True},
            {"radius_mm": 5.0, "accepted_pass": True, "strict_pass": True},
            {"radius_mm": 5.0, "accepted_pass": True, "strict_pass": False},
            {"radius_mm": 10.0, "accepted_pass": False, "strict_pass": False},
        ]
        summary = offset_test._summarize_radii(results)
        self.assertEqual(summary["max_supported_radius_mm"], 5.0)
        self.assertEqual(summary["max_strict_radius_mm"], 0.0)
        self.assertEqual(summary["by_radius"][1]["accepted_count"], 2)

    def test_final_motion_summary_reports_max_and_mean_pose_error(self):
        results = [
            {
                "final_motion": {
                    "final_xy": {"pose_error": {
                        "translation_error_norm_mm": 0.2, "rotation_error_deg": 0.01,
                    }},
                    "final_z": {"pose_error": {
                        "translation_error_norm_mm": 0.3, "rotation_error_deg": 0.02,
                    }},
                    "final_y_plus_0_3": {"pose_error": {
                        "translation_error_norm_mm": 0.1, "rotation_error_deg": 0.03,
                    }},
                },
            },
            {
                "final_motion": {
                    "final_xy": {"pose_error": {
                        "translation_error_norm_mm": 0.4, "rotation_error_deg": 0.04,
                    }},
                    "final_z": {"pose_error": {
                        "translation_error_norm_mm": 0.2, "rotation_error_deg": 0.01,
                    }},
                    "final_y_plus_0_3": {"pose_error": {
                        "translation_error_norm_mm": 0.1, "rotation_error_deg": 0.02,
                    }},
                },
            },
        ]
        summary = offset_test._summarize_final_motion(results)
        self.assertTrue(summary["enabled"])
        self.assertEqual(summary["sample_count"], 2)
        self.assertAlmostEqual(summary["steps"]["final_xy"]["max_translation_error_mm"], 0.4)
        self.assertAlmostEqual(summary["steps"]["final_z"]["mean_translation_error_mm"], 0.25)

    def test_final_target_confirmation_is_enabled_by_default(self):
        args = offset_test.build_parser().parse_args(["--no-execute"])
        self.assertTrue(args.confirm_each_offset)
        args = offset_test.build_parser().parse_args([
            "--no-execute", "--auto-continue-offset",
        ])
        self.assertFalse(args.confirm_each_offset)

    def test_manual_error_mark_is_preserved_in_radius_summary(self):
        summary = offset_test._summarize_radii([
            {"sample_id": 1, "radius_mm": 0.0, "accepted_pass": True,
             "strict_pass": True, "manual_error_marked": False},
            {"sample_id": 2, "radius_mm": 5.0, "accepted_pass": True,
             "strict_pass": True, "manual_error_marked": True},
        ])
        self.assertEqual(summary["manual_error_count"], 1)
        self.assertEqual(summary["manual_error_sample_ids"], [2])
        self.assertEqual(summary["by_radius"][1]["manual_error_count"], 1)
        self.assertEqual(summary["max_supported_radius_mm"], 0.0)

    def test_gui_builds_offset_test_command_without_final_motion_flags(self):
        panel = HoleLocalizationPanel.__new__(HoleLocalizationPanel)
        panel.model_var = _Value("model.pt")
        panel.handeye_var = _Value("handeye.json")
        panel.confidence_var = _Value("0.35")
        panel.coarse_height_var = _Value("340")
        panel.fine_height_var = _Value("260")
        panel.coarse_frames_var = _Value("15")
        panel.fine_frames_var = _Value("30")
        panel.speed_var = _Value("0.08")
        panel.acc_var = _Value("0.25")
        panel.transit_speed_var = _Value("0.15")
        panel.transit_acc_var = _Value("0.45")
        panel.approach_speed_var = _Value("0.12")
        panel.approach_acc_var = _Value("0.35")
        panel.offset_radii_var = _Value("0 5 10")
        panel.offset_angles_var = _Value("0 90 180")
        panel.execute_var = _Value(False)
        panel.experimental_var = _Value(True)
        panel.final_xy_var = _Value(True)
        panel.include_final_motion_var = _Value(False)
        panel.connection_provider = lambda: {
            "ip": "127.0.0.1", "port": 30004, "user": "a",
            "password": "b", "timeout_ms": 1000,
        }
        command = panel._build_command("offset")
        self.assertIn("run_coarse_to_fine_offset_test.py", command[2])
        self.assertIn("--radii-mm", command)
        self.assertIn("--angles-deg", command)
        self.assertNotIn("--move-final-xy", command)
        self.assertNotIn("--start-confirmed", command)


if __name__ == "__main__":
    unittest.main()
