import unittest

import numpy as np

import diagnose_hole_precision_pycharm as diagnostic


class _Intrinsics:
    fx = 610.0
    fy = 610.0


class AutomaticMultiViewPlanTests(unittest.TestCase):
    def test_default_views_obey_hard_motion_limits(self):
        views = diagnostic.automatic_view_sequence()
        diagnostic.validate_automatic_views(views)
        self.assertEqual(len(views), 12)
        for view in views:
            self.assertLessEqual(np.linalg.norm(view.translation_mm), 50.0)
            rotation = diagnostic.rotation_matrix_rxyz_deg(*view.rotation_deg_rxyz)
            self.assertLessEqual(diagnostic.relative_rotation_deg(np.eye(3), rotation), 40.0)

    def test_relative_translation_uses_reference_tcp_axes(self):
        reference = np.eye(4)
        # 参考 TCP 绕 Z 旋转 90°：其本地 X 正方向是基坐标 Y 正方向。
        reference[:3, :3] = diagnostic.rotation_matrix_rxyz_deg(0.0, 0.0, 90.0)
        view = diagnostic.AutoView("local_x", np.array([40.0, 0.0, 0.0]), np.zeros(3))
        target = diagnostic.automatic_target_pose(reference, view)
        np.testing.assert_allclose(target[:3, 3], np.array([0.0, 40.0, 0.0]), atol=1e-9)
        self.assertAlmostEqual(diagnostic.relative_rotation_deg(reference[:3, :3], target[:3, :3]), 0.0)

    def test_hard_limit_rejects_unsafe_view(self):
        unsafe = [diagnostic.AutoView("too_far", np.array([50.1, 0.0, 0.0]), np.zeros(3))]
        with self.assertRaises(ValueError):
            diagnostic.validate_automatic_views(unsafe)

    def test_absolute_error_separates_fixed_bias_from_view_scatter(self):
        truth = np.array([533.21, -108.40, -136.75])
        bias = np.array([-0.58, -6.72, 35.66])
        records = [
            {"hole_point_base_corrected_mm": truth + bias + np.array([0.10, 0.00, 0.00])},
            {"hole_point_base_corrected_mm": truth + bias + np.array([-0.10, 0.00, 0.00])},
        ]
        stats = diagnostic.summarize_absolute_errors(records, "hole_point_base_corrected_mm", truth)
        diagnosis = diagnostic.diagnose_absolute_bias(stats)
        np.testing.assert_allclose(stats["median_error_estimate_minus_truth_mm"], bias, atol=1e-9)
        self.assertAlmostEqual(diagnosis["fixed_bias_norm_mm"], float(np.linalg.norm(bias)))
        self.assertEqual(diagnosis["dominant_bias_axis"], "Z")
        self.assertAlmostEqual(diagnosis["relative_scatter_after_removing_fixed_bias_rms_mm"], 0.1)

    def test_known_circle_rgb_range_is_independent_of_depth(self):
        # 70 mm 圆孔在 470 mm 正视距离下的理论像素直径。
        expected_range = 470.0
        diameter_px = _Intrinsics.fx * 70.0 / expected_range
        estimated = diagnostic.estimate_circle_range_from_rgb_mm(
            np.array([diameter_px, diameter_px]), _Intrinsics(), 70.0,
        )
        self.assertAlmostEqual(estimated, expected_range)

    def test_rgb_depth_diameter_estimates_65_mm(self):
        depth_mm = 434.0
        diameter_mm = 65.0
        diameter_px = _Intrinsics.fx * diameter_mm / depth_mm
        estimated = diagnostic.estimate_circle_diameter_from_rgb_depth_mm(
            np.array([diameter_px, diameter_px]), _Intrinsics(), depth_mm,
        )
        self.assertAlmostEqual(estimated, diameter_mm)

    def test_multiview_summary_ignores_skipped_view_values(self):
        records = [
            {"hole": np.array([1.0, 2.0, 3.0])},
            {"hole": None},
            {"hole": np.array([1.1, 2.0, 3.0])},
        ]
        stats = diagnostic.summarize_multiview(records, "hole")
        self.assertIsNotNone(stats)
        self.assertEqual(stats["count"], 2)

    def test_target_not_visible_is_a_distinct_recoverable_error(self):
        self.assertTrue(issubclass(diagnostic.TargetNotVisibleError, RuntimeError))

    def test_return_is_not_requested_when_already_at_reference(self):
        reference = np.eye(4)
        current = reference.copy()
        current[:3, 3] = np.array([0.02, 0.0, 0.0])
        self.assertFalse(diagnostic.needs_reference_return(current, reference))
        current[:3, 3] = np.array([0.11, 0.0, 0.0])
        self.assertTrue(diagnostic.needs_reference_return(current, reference))


if __name__ == "__main__":
    unittest.main()
