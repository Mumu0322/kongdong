#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aubo_workbench.cad_model import CadHole, CadModel  # noqa: E402
from aubo_workbench.cad_registration import (  # noqa: E402
    CameraIntrinsics,
    CadRegistrationConfig,
    CadDetection,
    draw_cad_overlay,
    draw_detection_boxes,
    match_detections_to_cad,
    project_cad_circle,
    project_points,
    register_cad_frame,
    register_detections_frames,
    save_registration_report,
    undistort_pixels,
    _projected_circle_diameter_px,
)
from aubo_workbench.geometry import make_transform, rotx, rotz  # noqa: E402


class CadRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        table = [
            (-150, -90, 68.5),
            (-50, -90, 78.5),
            (50, -90, 78.5),
            (150, -90, 68.5),
            (-100, 0, 78.5),
            (0, 0, 73.5),
            (100, 0, 78.5),
            (-150, 90, 68.5),
            (-50, 90, 73.5),
            (50, 90, 73.5),
            (150, 90, 68.5),
        ]
        cls.model = CadModel(
            source_path="synthetic",
            unit="mm",
            coordinate_system="CAD",
            origin_description="test",
            top_surface_z_mm=6.0,
            top_normal_cad=np.array([0.0, 0.0, 1.0]),
            holes=tuple(
                CadHole(
                    hole_id=f"CAD-{index:02d}",
                    center_cad_mm=np.array([x, y, 6.0]),
                    diameter_mm=diameter,
                    top_z_mm=6.0,
                    normal_cad=np.array([0.0, 0.0, 1.0]),
                )
                for index, (x, y, diameter) in enumerate(table, start=1)
            ),
        )
        cls.intrinsics = CameraIntrinsics(
            width=1280,
            height=800,
            fx=610.0485,
            fy=610.2708,
            cx=648.2422,
            cy=406.7495,
            dist_coeffs=np.array(
                [-0.0335317552, 0.0374677218, 0.0002021208, -0.0000999125, -0.0132375685, 0, 0, 0]
            ),
        )
        cls.T_true = make_transform(
            rotx(math.pi) @ rotz(math.radians(4.0)),
            np.array([20.0, -30.0, 520.0]),
        )
        cls.T_prior = make_transform(
            cls.T_true[:3, :3],
            cls.T_true[:3, 3] + np.array([0.5, -0.3, 0.5]),
        )

    def detections(self, indices, noise_px=0.0, seed=7):
        rng = np.random.default_rng(seed)
        points = np.asarray([self.model.holes[index].center_cad_mm for index in indices])
        undistorted = project_points(points, self.T_true, self.intrinsics, distorted=False)
        distorted = project_points(points, self.T_true, self.intrinsics, distorted=True)
        detections = []
        for detection_id, (center, center_distorted) in enumerate(zip(undistorted, distorted)):
            noisy = center + rng.normal(0.0, noise_px, 2)
            box_center = center_distorted
            box = np.r_[box_center - [28.0, 28.0], box_center + [28.0, 28.0]]
            detections.append(
                CadDetection(
                    detection_id=detection_id,
                    box_xyxy=box,
                    center_px_distorted=center_distorted,
                    center_px=noisy,
                    confidence=0.95,
                )
            )
        return detections

    def test_four_visible_holes_ippe_and_transform_direction(self):
        detections = self.detections([0, 4, 6, 10])
        result = register_cad_frame(
            0,
            detections,
            self.model,
            self.intrinsics,
            self.T_prior,
            np.eye(4),
        )
        self.assertTrue(result.success, result.failure_reasons)
        np.testing.assert_allclose(result.T_camera_cad[:3, 3], self.T_true[:3, 3], atol=1e-2)
        self.assertEqual(len(result.projected_centers_px), 11)
        self.assertEqual(len(result.base_holes), 11)
        camera_to_hole = result.base_holes[0]["point_base_mm"] - np.array([0.0, 0.0, 0.0])
        self.assertGreater(
            float(np.dot(result.base_holes[0]["normal_toward_camera_base"], -camera_to_hole)),
            0.0,
        )

    def test_more_than_four_holes_and_all_reprojection_errors(self):
        detections = self.detections(list(range(8)))
        result = register_cad_frame(0, detections, self.model, self.intrinsics, self.T_prior, np.eye(4))
        self.assertTrue(result.success, result.failure_reasons)
        self.assertLess(result.quality["gates"]["rmse_px"], 1e-5)
        self.assertEqual(set(result.reprojection_errors_px), {f"CAD-{index:02d}" for index in range(1, 9)})

    def test_distortion_is_removed_before_pnp(self):
        undistorted = project_points(
            np.asarray([hole.center_cad_mm for hole in self.model.holes[:4]]),
            self.T_true,
            self.intrinsics,
            distorted=False,
        )
        distorted = project_points(
            np.asarray([hole.center_cad_mm for hole in self.model.holes[:4]]),
            self.T_true,
            self.intrinsics,
            distorted=True,
        )
        corrected = undistort_pixels(distorted, self.intrinsics)
        np.testing.assert_allclose(corrected, undistorted, atol=1e-5)

    def test_missing_or_false_detection_cannot_pass_four_hole_gate(self):
        detections = self.detections([0, 4, 6])
        false = CadDetection(99, [30, 30, 80, 80], [55, 55], [55, 55], 0.99)
        result = register_cad_frame(0, detections + [false], self.model, self.intrinsics, self.T_prior, np.eye(4))
        self.assertFalse(result.success)
        self.assertTrue(any("小于最少" in reason for reason in result.failure_reasons))

    def test_collinear_visible_holes_are_rejected(self):
        detections = self.detections([0, 1, 2, 3])
        result = register_cad_frame(0, detections, self.model, self.intrinsics, self.T_prior, np.eye(4))
        self.assertFalse(result.success)
        self.assertTrue(any("共线" in reason for reason in result.failure_reasons))

    def test_no_prior_center_only_mapping_rejects_symmetric_ambiguity(self):
        detections = self.detections([0, 4, 6, 10])
        automatic = match_detections_to_cad(self.model, detections, None, self.intrinsics)
        self.assertTrue(automatic.ambiguous)
        self.assertTrue(any("自动几何匹配" in reason for reason in automatic.failure_reasons))
        self.assertEqual(
            automatic.diagnostics["matching_method"],
            "triangle_geometric_hash + homography + one_to_one_assignment",
        )
        manual = {str(index): f"CAD-{cad_index + 1:02d}" for index, cad_index in enumerate([0, 4, 6, 10])}
        result = register_cad_frame(
            0,
            detections,
            self.model,
            self.intrinsics,
            None,
            np.eye(4),
            manual_mapping=manual,
        )
        self.assertTrue(result.success, result.failure_reasons)

    def test_no_prior_geometry_matching_uses_diameter_to_select_full_layout(self):
        detections = self.detections(list(range(11)))
        for detection, hole in zip(detections, self.model.holes):
            diameter = _projected_circle_diameter_px(
                project_cad_circle(hole, self.T_true, self.intrinsics)
            )
            detection.ellipse = {"axes_px": [diameter, diameter]}

        result = register_cad_frame(
            0,
            detections,
            self.model,
            self.intrinsics,
            None,
            np.eye(4),
        )
        self.assertTrue(result.success, result.failure_reasons)
        self.assertEqual(
            [match.cad_hole_id for match in sorted(result.matches, key=lambda item: item.detection_id)],
            [f"CAD-{index:02d}" for index in range(1, 12)],
        )
        self.assertTrue(all(
            match.mapping_source == "automatic_geometry_ransac"
            for match in result.matches
        ))

    def test_single_frame_is_preview_only(self):
        result = register_detections_frames(
            [self.detections(list(range(8)))],
            self.model,
            self.intrinsics,
            self.T_prior,
            np.eye(4),
        )
        self.assertTrue(result.success, result.failure_reasons)
        self.assertFalse(result.multi_frame_gate_pass)
        self.assertFalse(result.motion_allowed)

    def test_five_frames_three_valid_is_allowed_by_frame_count_gate(self):
        valid = self.detections(list(range(11)), noise_px=0.05)
        invalid = self.detections([0, 1, 2])
        result = register_detections_frames(
            [valid, valid, valid, invalid, invalid],
            self.model,
            self.intrinsics,
            self.T_prior,
            np.eye(4),
        )
        self.assertTrue(result.success, result.failure_reasons)
        self.assertTrue(result.multi_frame_gate_pass)
        self.assertEqual(result.valid_frame_indices, [0, 1, 2])
        self.assertFalse(result.motion_allowed)
        self.assertLessEqual(result.cross_frame_center_p95_mm, 1.5)
        self.assertEqual(len(result.projected_centers_px), 11)

    def test_first_manual_mapping_bootstraps_following_frames(self):
        frames = [
            self.detections(list(range(11)), noise_px=0.05, seed=index)
            for index in range(5)
        ]
        mapping = {str(index): f"CAD-{index + 1:02d}" for index in range(6)}
        result = register_detections_frames(
            frames,
            self.model,
            self.intrinsics,
            None,
            np.eye(4),
            manual_mapping=mapping,
            sequential_prior=True,
        )
        self.assertTrue(result.success, result.failure_reasons)
        self.assertTrue(result.multi_frame_gate_pass)
        self.assertEqual(result.valid_frame_indices, [0, 1, 2, 3, 4])

    def test_bad_reprojection_fails_quality_gate(self):
        detections = self.detections(list(range(11)), noise_px=4.0)
        result = register_cad_frame(0, detections, self.model, self.intrinsics, self.T_prior, np.eye(4))
        self.assertFalse(result.success)
        self.assertTrue(result.failure_reasons)

    def test_overlay_and_report_contain_mapping_and_motion_lock(self):
        detections = self.detections(list(range(8)))
        result = register_detections_frames(
            [detections], self.model, self.intrinsics, self.T_prior, np.eye(4)
        )
        image = np.zeros((800, 1280, 3), dtype=np.uint8)
        overlay = draw_cad_overlay(image, self.model, self.intrinsics, result.frames[0])
        overlay = draw_detection_boxes(overlay, detections, result.frames[0], self.intrinsics)
        self.assertEqual(overlay.shape, image.shape)
        with tempfile.TemporaryDirectory() as directory:
            report = save_registration_report(
                directory,
                result,
                model=self.model,
                intrinsics=self.intrinsics,
                inputs={"test": True},
                detections_by_frame=[detections],
            )
            payload = json.loads(Path(report).read_text(encoding="utf-8"))
            self.assertFalse(payload["motion_allowed"])
            self.assertTrue(Path(payload["mapping_files"]["csv"]).is_file())
            self.assertEqual(len(payload["result"]["frames"][0]["projected_centers_px"]), 11)


if __name__ == "__main__":
    unittest.main()
