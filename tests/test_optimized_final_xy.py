import unittest

import numpy as np

from run_yolo_eye_in_hand_optimized import (
    Observation,
    TwoStageConfig,
    _fuse_fine,
    build_parser,
    fit_sphere,
    _match_selected_holes_at_fine,
    _ray_sphere_intersection_base,
    _hole_surface_pose,
    plan_final_tcp_base_z,
    plan_final_tcp_base_y_trim,
    plan_final_tcp_xy,
)


class FinalXyPlanningTests(unittest.TestCase):
    def test_manual_three_hole_matching_uses_reference_and_relative_geometry(self):
        class Intrinsics:
            cx = 640.0
            cy = 400.0

        tracked = [
            {"hole_id": 1, "initial_center_px": [556.0, 387.0], "initial_box": [496.0, 327.0, 616.0, 447.0]},
            {"hole_id": 2, "initial_center_px": [422.0, 353.0], "initial_box": [362.0, 293.0, 482.0, 413.0]},
            {"hole_id": 3, "initial_center_px": [683.0, 340.0], "initial_box": [623.0, 280.0, 743.0, 400.0]},
        ]
        def detection(center):
            x, y = center
            return {"center": [x, y], "box": [x - 75.0, y - 75.0, x + 75.0, y + 75.0]}
        reference = detection((650.0, 400.0))
        left_selected = detection((482.5, 357.5))
        right_selected = detection((808.75, 341.25))
        decoy = detection((320.0, 278.0))
        matches = _match_selected_holes_at_fine(
            [reference, left_selected, right_selected, decoy], tracked, Intrinsics(),
        )
        self.assertIs(matches[1], reference)
        self.assertIs(matches[2], left_selected)
        self.assertIs(matches[3], right_selected)

    def test_three_hole_parser_and_sphere_ray_intersection(self):
        args = build_parser().parse_args(["--hole-count", "3"])
        self.assertEqual(args.hole_count, 3)

        class Intrinsics:
            fx = 1000.0
            fy = 1000.0
            cx = 320.0
            cy = 240.0
            distortion = ()

        point, normal, mode = _ray_sphere_intersection_base(
            np.array([320.0, 240.0]), Intrinsics(), np.eye(4),
            np.array([0.0, 0.0, 1000.0]), 100.0,
        )
        np.testing.assert_allclose(point, np.array([0.0, 0.0, 900.0]), atol=1e-6)
        np.testing.assert_allclose(normal, np.array([0.0, 0.0, -1.0]), atol=1e-6)
        self.assertEqual(mode, "sphere")
        pose = _hole_surface_pose(point, normal, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(pose[:3, 3], point, atol=1e-6)
        np.testing.assert_allclose(pose[:3, 2], normal, atol=1e-6)

    def test_final_xy_motion_is_enabled_by_default(self):
        args = build_parser().parse_args([])
        self.assertTrue(args.move_final_xy)
        self.assertTrue(args.two_stage_hole_localization)
        self.assertTrue(args.execute)
        self.assertTrue(args.allow_experimental_handeye)

    def test_coarse_normal_gate_matches_measured_depth_repeatability(self):
        self.assertEqual(TwoStageConfig().normal_tolerance_deg, 2.0)
        self.assertEqual(TwoStageConfig().initial_max_plane_rmse_mm, 3.5)
        self.assertEqual(TwoStageConfig().max_plane_rmse_mm, 3.5)
        self.assertEqual(TwoStageConfig().coarse_frames, 10)
        self.assertEqual(TwoStageConfig().fine_frames, 20)
        self.assertEqual(TwoStageConfig().preliminary_coarse_frames, 6)
        self.assertEqual(TwoStageConfig().multi_fine_extra_frames, 4)
        self.assertEqual(TwoStageConfig().coarse_settle_frames, 5)
        self.assertEqual(TwoStageConfig().coarse_max_attempt_multiplier, 4)
        self.assertEqual(TwoStageConfig().min_coarse_ellipse_coverage_deg, 120.0)

    def test_default_charuco_model_preserves_z_and_orientation(self):
        tcp = np.eye(4)
        tcp[:3, :3] = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        tcp[:3, 3] = np.array([100.0, 200.0, 300.0])
        target, before = plan_final_tcp_xy(tcp, np.array([443.0, -165.0, -99.0]))
        np.testing.assert_allclose(target[:2, 3], np.array([443.62468824, -162.32352141]))
        self.assertEqual(target[2, 3], 300.0)
        np.testing.assert_allclose(target[:3, :3], tcp[:3, :3])
        np.testing.assert_allclose(before, tcp[:3, 3])

    def test_explicit_offset_overrides_charuco_model(self):
        tcp = np.eye(4)
        target, _ = plan_final_tcp_xy(tcp, np.array([443.0, -165.0, -99.0]), (0.0, 7.0))
        np.testing.assert_allclose(target[:2, 3], np.array([443.0, -158.0]))

    def test_final_base_z_uses_hole_center_and_preserves_xy_orientation(self):
        tcp = np.eye(4)
        tcp[:3, :3] = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        tcp[:3, 3] = np.array([631.7, -108.3, 148.2])
        target = plan_final_tcp_base_z(tcp, np.array([631.0, -110.7, 56.7]))
        np.testing.assert_allclose(target[:2, 3], tcp[:2, 3])
        self.assertAlmostEqual(target[2, 3], 56.7)
        np.testing.assert_allclose(target[:3, :3], tcp[:3, :3])

    def test_final_base_y_trim_preserves_x_z_and_orientation(self):
        tcp = np.eye(4)
        tcp[:3, 3] = np.array([631.7, -108.3, 56.7])
        target = plan_final_tcp_base_y_trim(tcp)
        np.testing.assert_allclose(target[:3, 3], np.array([631.7, -108.1, 56.7]))
        np.testing.assert_allclose(target[:3, :3], tcp[:3, :3])

    def test_fine_fusion_does_not_make_p95_worse_after_outlier_filtering(self):
        # 回归样本：原始P95小于1 px，但旧算法剔除1帧后中位数偏向一簇，
        # 反而把P95放大到1 px以上。
        center_v = [
            332.07994531419695, 331.95069967419954, 330.61083376422056,
            330.8037100901098, 331.1300572625171, 330.67781485907216,
            330.69735710687746, 331.8939782737084, 330.6362278090009,
            330.7608659185484, 330.69968753078626, 330.73853579578883,
            331.9915950528126, 331.096449739288, 331.9819165744354,
            330.75230655823776, 332.03853054954936, 330.8726724066489,
            332.1673947147796, 330.71694271113415, 330.9497804532087,
            332.0092691189518, 332.06311867240703, 332.03395852791436,
            331.9474284131484,
        ]
        ellipse = {
            "residual_px": 0.5,
            "roundness": 0.99,
            "axes_px": [150.0, 150.0],
        }
        observations = [
            Observation(
                stage="fine_hole_3",
                frame_index=index,
                center_px=np.array([478.6, value]),
                ellipse=ellipse,
            )
            for index, value in enumerate(center_v)
        ]
        summary = _fuse_fine(
            observations,
            TwoStageConfig(),
            max_center_scatter_p95_px=1.0,
        )
        self.assertEqual(summary["fusion_selection"], "raw_lower_or_equal_p95")
        self.assertEqual(summary["valid_frames"], len(observations))
        self.assertLessEqual(summary["center_scatter_p95_px"], 1.0)
        self.assertLessEqual(
            summary["center_scatter_p95_px"],
            summary["mad_filtered_p95_px"],
        )

    def test_unknown_radius_sphere_fit_with_outliers(self):
        rng = np.random.default_rng(7)
        center = np.array([20.0, -15.0, 340.0])
        radius = 520.0
        azimuth = rng.uniform(0.0, 2.0 * np.pi, 300)
        elevation = rng.uniform(-0.18, 0.18, 300)
        points = center + radius * np.column_stack((
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ))
        points += rng.normal(0.0, 0.15, points.shape)
        points[:20] += np.array([0.0, 0.0, 30.0])
        estimated_center, estimated_radius, rmse = fit_sphere(points)
        np.testing.assert_allclose(estimated_center, center, atol=1.0)
        self.assertAlmostEqual(estimated_radius, radius, delta=1.0)
        self.assertLess(rmse, 1.0)

if __name__ == "__main__":
    unittest.main()
