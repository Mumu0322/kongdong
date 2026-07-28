# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aubo_workbench.charuco_point_experiment import (
    compute_experiment_statistics,
    load_handeye_experiment_result,
    nearest_detected_corner,
    selected_corner_transforms,
    target_pose_with_fixed_rz,
    xyz_rpy_pose,
)
from aubo_workbench.geometry import make_transform, rotz


class CharucoPointExperimentTests(unittest.TestCase):
    def test_selected_corner_transform_chain(self):
        T_base_tcp = make_transform(np.eye(3), [100.0, 200.0, 300.0])
        T_tcp_camera = make_transform(np.eye(3), [10.0, 20.0, 30.0])
        T_camera_board = make_transform(np.eye(3), [1.0, 2.0, 500.0])
        frames = selected_corner_transforms(
            T_base_tcp, T_tcp_camera, T_camera_board, np.array([30.0, 60.0, 0.0]),
        )
        np.testing.assert_allclose(frames["rgb_camera"][:3, 3], [31.0, 62.0, 500.0])
        np.testing.assert_allclose(frames["tcp"][:3, 3], [41.0, 82.0, 530.0])
        np.testing.assert_allclose(frames["base"][:3, 3], [141.0, 282.0, 830.0])

    def test_pose_reports_rpy_radians_in_rx_ry_rz_order(self):
        pose = xyz_rpy_pose(make_transform(rotz(np.deg2rad(30.0)), [1.0, 2.0, 3.0]))
        np.testing.assert_allclose(pose["xyz_mm"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(pose["rpy_rad_zyx"], [0.0, 0.0, np.pi / 6.0], atol=1e-8)
        self.assertEqual(pose["angle_unit"], "rad")

    def test_target_pose_keeps_xyz_and_locks_rz(self):
        measured = make_transform(rotz(0.25), [11.0, 22.0, 33.0])
        target = target_pose_with_fixed_rz(measured, 1.735)
        np.testing.assert_allclose(target["xyz_mm"], [11.0, 22.0, 33.0])
        self.assertAlmostEqual(target["rpy_rad_zyx"][2], 1.735, places=12)
        self.assertAlmostEqual(target["fixed_rz_rad"], 1.735, places=12)

    def test_nearest_detected_corner(self):
        self.assertEqual(nearest_detected_corner((12, 9), [3, 8], [[10, 10], [100, 100]]), 3)
        self.assertIsNone(nearest_detected_corner((60, 60), [3, 8], [[10, 10], [100, 100]], 20))

    def test_statistics_use_base_point_scatter(self):
        records = []
        for index, x in enumerate([0.0, 2.0], start=1):
            T = make_transform(np.eye(3), [x, 0.0, 0.0])
            records.append({
                "frames": {"base": xyz_rpy_pose(T)},
                "robot": {"pose_values_mm_rad": [x, 0, 0, 0, 0, 0]},
            })
        stats = compute_experiment_statistics(records, reference_base_xyz_mm=[0.0, 0.0, 0.0])
        self.assertEqual(stats["sample_count"], 2)
        self.assertAlmostEqual(stats["base_point_scatter_rms_mm"], 1.0)
        self.assertAlmostEqual(stats["base_point_scatter_max_mm"], 1.0)
        np.testing.assert_allclose(stats["base_xyz_axis_range_mm"], [2.0, 0.0, 0.0])
        self.assertAlmostEqual(stats["absolute_error_of_mean_mm"], 1.0)
        self.assertAlmostEqual(stats["absolute_error_rms_mm"], np.sqrt(2.0))

    def test_unvalidated_candidate_is_explicitly_experimental(self):
        payload = {
            "validated": False,
            "do_not_use_for_motion": True,
            "production_eligible": False,
            "pose_source": "tcp",
            "calibration_frame": "rgb_camera",
            "T_tcp_rgb_camera": np.eye(4).tolist(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_handeye_experiment_result(path)
        self.assertTrue(loaded.experimental_only)
        self.assertFalse(loaded.validated_for_motion)
        self.assertIn("禁止用于机械臂运动", loaded.warning)


if __name__ == "__main__":
    unittest.main()
