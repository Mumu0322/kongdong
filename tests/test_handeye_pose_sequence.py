#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import unittest

from run_handeye_pose_sequence import (
    DEFAULT_PLAN,
    angular_delta_rad,
    load_plan,
    pose_error,
    pose_mm_rad_to_sdk_m_rad,
)


class HandEyePoseSequenceTests(unittest.TestCase):
    def test_current_plan_has_40_valid_mm_rad_poses(self) -> None:
        poses = load_plan(DEFAULT_PLAN)
        self.assertEqual(len(poses), 40)
        self.assertEqual([pose["plan_index"] for pose in poses], list(range(1, 41)))
        self.assertAlmostEqual(poses[11]["pose_mm_rad_rxryrz"][3], math.pi / 10.0, places=8)

    def test_mm_rad_to_sdk_m_rad_only_scales_xyz(self) -> None:
        result = pose_mm_rad_to_sdk_m_rad([411.0, -262.0, 225.0, 0.31, -0.22, 0.77])
        self.assertEqual(result[:3], [0.411, -0.262, 0.225])
        self.assertEqual(result[3:], [0.31, -0.22, 0.77])

    def test_pose_error_wraps_rotation_at_pi(self) -> None:
        target = [0.0, 0.0, 0.0, math.pi - 0.01, 0.0, 0.0]
        current = [0.0, 0.0, 0.0, -math.pi + 0.01, 0.0, 0.0]
        xyz_mm, rotation_rad = pose_error(target, current)
        self.assertAlmostEqual(xyz_mm, 0.0)
        self.assertAlmostEqual(rotation_rad, 0.02, places=8)
        self.assertAlmostEqual(angular_delta_rad(target[3], current[3]), -0.02, places=8)


if __name__ == "__main__":
    unittest.main()
