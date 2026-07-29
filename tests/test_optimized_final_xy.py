import unittest

import numpy as np

from run_yolo_eye_in_hand_optimized import (
    TwoStageConfig,
    build_parser,
    fit_sphere,
    _ray_sphere_intersection_base,
    _hole_surface_pose,
    plan_final_tcp_base_z,
    plan_final_tcp_base_y_trim,
    plan_final_tcp_xy,
)


class FinalXyPlanningTests(unittest.TestCase):
    def test_three_hole_parser_and_sphere_ray_intersection(self):
        args = build_parser().parse_args(["--hole-count", "3"])
        self.assertEqual(args.hole_count, 3)

        class Intrinsics:
            fx = 1000.0
            fy = 1000.0
            cx = 320.0
            cy = 240.0
            distortion = ()

        point, normal = _ray_sphere_intersection_base(
            np.array([320.0, 240.0]), Intrinsics(), np.eye(4),
            np.array([0.0, 0.0, 1000.0]), 100.0,
        )
        np.testing.assert_allclose(point, np.array([0.0, 0.0, 900.0]), atol=1e-6)
        np.testing.assert_allclose(normal, np.array([0.0, 0.0, -1.0]), atol=1e-6)
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
        self.assertEqual(TwoStageConfig().coarse_settle_frames, 10)
        self.assertEqual(TwoStageConfig().coarse_max_attempt_multiplier, 6)
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
