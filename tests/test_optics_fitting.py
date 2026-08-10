#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""aubo_workbench.optics / fitting / geometry 新增纯函数的直接测试。"""

import math
import unittest

import cv2
import numpy as np

from aubo_workbench.camera import CameraIntrinsics as RgbIntrinsics
from aubo_workbench.cad_registration import CameraIntrinsics as CadIntrinsics
from aubo_workbench.fitting import fit_plane, fit_sphere
from aubo_workbench.geometry import (
    angle_between_deg,
    make_transform,
    matrix_to_rpy_zyx,
    rotx,
    roty,
    rotz,
    transform_to_sdk_pose_m_rad,
    unit_vector,
)
from aubo_workbench.optics import (
    base_z_target_for_camera_height,
    camera_height_to_plane_mm,
    correct_projected_circle_center,
    distortion_coeffs,
    pixel_to_base_plane,
    plane_basis,
    project_undistorted_pixels,
    ray_plane_intersection,
    undistort_pixels,
)

IDENTITY = np.eye(4, dtype=np.float64)


def _rgb_intrinsics(distortion=()):
    return RgbIntrinsics(1280, 720, 1000.0, 1000.0, 640.0, 360.0, tuple(distortion))


class GeometryHelperTests(unittest.TestCase):
    def test_unit_vector_normalizes(self):
        np.testing.assert_allclose(unit_vector([0.0, 0.0, 5.0]), [0.0, 0.0, 1.0])

    def test_unit_vector_rejects_zero_and_non_finite(self):
        for bad in ([0.0, 0.0, 0.0], [np.nan, 0.0, 1.0]):
            with self.assertRaisesRegex(ValueError, "无法归一化"):
                unit_vector(bad, "法向")

    def test_angle_between_deg(self):
        self.assertAlmostEqual(angle_between_deg([1, 0, 0], [0, 1, 0]), 90.0, places=9)
        self.assertAlmostEqual(angle_between_deg([1, 0, 0], [2, 0, 0]), 0.0, places=9)

    def test_matrix_to_rpy_zyx_roundtrip(self):
        rx, ry, rz = 0.21, -0.34, 1.05
        R = rotz(rz) @ roty(ry) @ rotx(rx)
        np.testing.assert_allclose(matrix_to_rpy_zyx(R), [rx, ry, rz], atol=1e-9)

    def test_matrix_to_rpy_zyx_gimbal_lock_stays_finite(self):
        # ry = -90° 时 cos(ry)=0，退化分支必须给出有限解而不是 nan。
        R = rotz(0.4) @ roty(-math.pi / 2) @ rotx(0.3)
        rpy = matrix_to_rpy_zyx(R)
        self.assertTrue(np.isfinite(rpy).all())
        self.assertAlmostEqual(float(rpy[1]), -math.pi / 2, places=7)

    def test_transform_to_sdk_pose_converts_mm_to_m(self):
        T = make_transform(rotz(0.5), [1200.0, -300.0, 450.0])
        pose = transform_to_sdk_pose_m_rad(T)
        np.testing.assert_allclose(pose[:3], [1.2, -0.3, 0.45], atol=1e-12)
        np.testing.assert_allclose(pose[3:], matrix_to_rpy_zyx(rotz(0.5)), atol=1e-12)


class DistortionCoeffsTests(unittest.TestCase):
    def test_reads_rgb_intrinsics_distortion_field(self):
        coeffs = distortion_coeffs(_rgb_intrinsics((0.1, -0.2, 0.0, 0.0, 0.05)))
        np.testing.assert_allclose(coeffs, [0.1, -0.2, 0.0, 0.0, 0.05])

    def test_reads_cad_intrinsics_dist_coeffs_field(self):
        cad = CadIntrinsics(1280, 720, 1000.0, 1000.0, 640.0, 360.0,
                            dist_coeffs=np.array([0.3, 0.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(distortion_coeffs(cad), [0.3, 0.0, 0.0, 0.0, 0.0])

    def test_empty_distortion_means_no_distortion(self):
        self.assertEqual(distortion_coeffs(_rgb_intrinsics(())).size, 0)

    def test_missing_both_fields_raises_instead_of_silently_undistorting(self):
        class Bare:
            fx = fy = 1000.0
            cx, cy = 640.0, 360.0

        with self.assertRaisesRegex(AttributeError, "dist_coeffs"):
            distortion_coeffs(Bare())

    def test_non_finite_distortion_raises_instead_of_falling_back_to_zero(self):
        # 字段缺失会响亮抛错，但字段存在却含 NaN/inf 曾被静默当成零畸变，
        # 属于同一类静默失败：标定坏了却照常算出一个看似正常的结果。
        for bad in ((0.1, np.nan, 0.0, 0.0, 0.0), (np.inf, 0.0, 0.0, 0.0, 0.0)):
            with self.assertRaisesRegex(ValueError, "非有限值"):
                distortion_coeffs(_rgb_intrinsics(bad))

    def test_non_finite_distortion_also_blocks_undistort_pixels(self):
        # 保证抛错发生在真正用到系数的调用路径上，而不只是取值函数里。
        intr = _rgb_intrinsics((0.1, np.nan, 0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "非有限值"):
            undistort_pixels(intr, np.asarray([[640.0, 360.0]]))

    def test_swapped_arguments_fail_loudly(self):
        # cad_registration.undistort_pixels 的参数顺序与本模块相反。
        # 传错顺序必须立刻抛错，而不是把畸变静默当成 0 —— 后者会让结果
        # 偏移零点几毫米却看不出任何异常。
        with self.assertRaises((TypeError, AttributeError)):
            undistort_pixels(np.asarray([[640.0, 360.0]]), _rgb_intrinsics((0.2, 0.0, 0.0, 0.0, 0.0)))


class UndistortTests(unittest.TestCase):
    def test_zero_distortion_is_identity_in_pixel_mode(self):
        pts = np.asarray([[100.0, 200.0], [640.0, 360.0]])
        np.testing.assert_allclose(undistort_pixels(_rgb_intrinsics(), pts), pts)

    def test_zero_distortion_normalized_output(self):
        out = undistort_pixels(_rgb_intrinsics(), np.asarray([[1640.0, 1360.0]]), pixel_output=False)
        np.testing.assert_allclose(out, [[1.0, 1.0]], atol=1e-12)

    def test_matches_opencv_when_distortion_present(self):
        intr = _rgb_intrinsics((0.12, -0.05, 0.001, 0.002, 0.01))
        pts = np.asarray([[300.0, 250.0], [900.0, 500.0]])
        K = np.asarray([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
        expected = cv2.undistortPoints(
            pts.reshape(-1, 1, 2), K,
            np.asarray(intr.distortion, dtype=np.float64).reshape(1, -1), P=K,
        ).reshape(-1, 2)
        np.testing.assert_allclose(undistort_pixels(intr, pts), expected, atol=1e-9)


class RayPlaneTests(unittest.TestCase):
    def test_intersection_with_known_plane(self):
        hit = ray_plane_intersection([0, 0, 0], [0, 0, 1], [0, 0, 500], [0, 0, -1])
        np.testing.assert_allclose(hit, [0, 0, 500], atol=1e-9)

    def test_parallel_ray_raises(self):
        with self.assertRaisesRegex(ValueError, "近乎平行"):
            ray_plane_intersection([0, 0, 0], [1, 0, 0], [0, 0, 500], [0, 0, -1])

    def test_plane_behind_camera_raises(self):
        with self.assertRaisesRegex(ValueError, "反向"):
            ray_plane_intersection([0, 0, 0], [0, 0, 1], [0, 0, -500], [0, 0, -1])

    def test_pixel_to_base_plane_centre_pixel_hits_optical_axis(self):
        point = pixel_to_base_plane(
            np.asarray([640.0, 360.0]), _rgb_intrinsics(), IDENTITY, IDENTITY,
            np.asarray([0.0, 0.0, 500.0]), np.asarray([0.0, 0.0, -1.0]),
        )
        np.testing.assert_allclose(point, [0.0, 0.0, 500.0], atol=1e-9)

    def test_plane_basis_is_orthonormal_and_spans_plane(self):
        normal = unit_vector([0.3, -0.2, -1.0])
        ax, ay = plane_basis(normal)
        self.assertAlmostEqual(float(ax @ ay), 0.0, places=12)
        self.assertAlmostEqual(float(ax @ normal), 0.0, places=12)
        self.assertAlmostEqual(float(ay @ normal), 0.0, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(ax)), 1.0, places=12)


class CameraHeightTests(unittest.TestCase):
    def test_height_along_optical_axis(self):
        height = camera_height_to_plane_mm(IDENTITY, IDENTITY, np.asarray([0.0, 0.0, 480.0]))
        self.assertAlmostEqual(height, 480.0, places=9)

    def test_base_z_target_reaches_requested_height(self):
        target, current = base_z_target_for_camera_height(
            IDENTITY, IDENTITY, np.asarray([0.0, 0.0, 480.0]), 300.0,
        )
        self.assertAlmostEqual(current, 480.0, places=9)
        reached = camera_height_to_plane_mm(target, IDENTITY, np.asarray([0.0, 0.0, 480.0]))
        self.assertAlmostEqual(reached, 300.0, places=6)

    def test_near_horizontal_optical_axis_refuses(self):
        T = make_transform(roty(math.pi / 2), [0.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "水平面"):
            base_z_target_for_camera_height(T, IDENTITY, np.asarray([500.0, 0.0, 0.0]), 300.0)


class ProjectedCircleCenterTests(unittest.TestCase):
    """倾斜圆的投影椭圆中心并不是真实圆心的投影，修正必须收敛到真值。"""

    def _scenario(self, diameter_mm=30.0, normal=(0.30, 0.0, -1.0), xy=(20.0, -10.0)):
        intr = _rgb_intrinsics()
        n = unit_vector(np.asarray(normal, dtype=np.float64))
        plane_point = np.asarray([0.0, 0.0, 500.0])
        x, y = xy
        # 由平面方程 n·(p - plane_point) = 0 解出 z
        z = float(plane_point[2] + (n[0] * (plane_point[0] - x) + n[1] * (plane_point[1] - y)) / n[2])
        true_center = np.asarray([x, y, z], dtype=np.float64)
        ax, ay = plane_basis(n)
        theta = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
        radius = diameter_mm / 2.0
        circle = (true_center[None, :]
                  + radius * np.cos(theta)[:, None] * ax[None, :]
                  + radius * np.sin(theta)[:, None] * ay[None, :])
        projected = project_undistorted_pixels(circle, intr)
        ellipse = cv2.fitEllipse(projected.astype(np.float32).reshape(-1, 1, 2))
        observed_center_px = np.asarray(ellipse[0], dtype=np.float64)
        return intr, n, plane_point, true_center, observed_center_px

    def test_correction_beats_naive_ray_intersection(self):
        intr, n, plane_point, true_center, observed_px = self._scenario()
        naive = pixel_to_base_plane(observed_px, intr, IDENTITY, IDENTITY,
                                    plane_point, n, center_is_undistorted=True)
        corrected, info = correct_projected_circle_center(
            observed_px, intr, IDENTITY, IDENTITY, plane_point, n, 30.0,
        )
        naive_err = float(np.linalg.norm(naive - true_center))
        corrected_err = float(np.linalg.norm(corrected - true_center))
        self.assertLess(corrected_err, naive_err)
        self.assertLess(corrected_err, 0.05)
        self.assertGreater(info["tilt_deg"], 15.0)
        self.assertEqual(info["method"], "iterative_projected_circle_center")

    def test_rejects_non_positive_diameter(self):
        intr, n, plane_point, _, observed_px = self._scenario()
        with self.assertRaisesRegex(ValueError, "孔径"):
            correct_projected_circle_center(
                observed_px, intr, IDENTITY, IDENTITY, plane_point, n, 0.0,
            )

    def test_rejects_too_few_samples(self):
        intr, n, plane_point, _, observed_px = self._scenario()
        with self.assertRaisesRegex(ValueError, "samples"):
            correct_projected_circle_center(
                observed_px, intr, IDENTITY, IDENTITY, plane_point, n, 30.0, samples=8,
            )

    def test_oversize_correction_is_refused(self):
        intr, n, plane_point, _, observed_px = self._scenario()
        with self.assertRaisesRegex(RuntimeError, "修正过大"):
            correct_projected_circle_center(
                observed_px, intr, IDENTITY, IDENTITY, plane_point, n, 30.0,
                max_correction_mm=1e-6,
            )

    def test_projection_rejects_points_behind_camera(self):
        with self.assertRaisesRegex(ValueError, "相机后方"):
            project_undistorted_pixels(np.asarray([[0.0, 0.0, -10.0]]), _rgb_intrinsics())


class FitPlaneTests(unittest.TestCase):
    def _grid(self, z_fn, n=8, span=20.0):
        xs, ys = np.meshgrid(np.linspace(-span, span, n), np.linspace(-span, span, n))
        xs, ys = xs.ravel(), ys.ravel()
        return np.column_stack((xs, ys, z_fn(xs, ys)))

    def test_recovers_horizontal_plane(self):
        points = self._grid(lambda x, y: np.full_like(x, 500.0))
        normal, rmse = fit_plane(points)
        np.testing.assert_allclose(np.abs(normal), [0.0, 0.0, 1.0], atol=1e-9)
        self.assertLess(rmse, 1e-9)

    def test_normal_points_toward_camera(self):
        normal, _ = fit_plane(self._grid(lambda x, y: np.full_like(x, 500.0)))
        self.assertLess(float(normal[2]), 0.0)

    def test_recovers_tilted_plane(self):
        normal, rmse = fit_plane(self._grid(lambda x, y: 500.0 + 0.25 * x))
        expected = unit_vector([0.25, 0.0, -1.0])
        np.testing.assert_allclose(normal, expected, atol=1e-9)
        self.assertLess(rmse, 1e-9)

    def test_rmse_matches_returned_plane_after_outlier_rejection(self):
        points = self._grid(lambda x, y: np.full_like(x, 500.0))
        points[0, 2] = 560.0  # 单个粗大外点
        normal, rmse = fit_plane(points)
        # rmse 必须描述最终返回的平面，而不是含外点的第一轮平面。
        self.assertLess(rmse, 1.0)
        np.testing.assert_allclose(np.abs(normal), [0.0, 0.0, 1.0], atol=1e-6)

    def test_too_few_points_raises(self):
        with self.assertRaisesRegex(ValueError, "无法拟合平面"):
            fit_plane(np.zeros((5, 3)))

    def test_non_finite_points_are_dropped(self):
        points = self._grid(lambda x, y: np.full_like(x, 500.0))
        points[:4, 2] = np.nan
        normal, _ = fit_plane(points)
        np.testing.assert_allclose(np.abs(normal), [0.0, 0.0, 1.0], atol=1e-9)


class FitSphereTests(unittest.TestCase):
    def _sphere_points(self, center, radius, count=200):
        rng = np.random.default_rng(7)
        directions = rng.normal(size=(count, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return np.asarray(center, dtype=np.float64) + radius * directions

    def test_recovers_center_and_radius(self):
        center, radius, rmse = fit_sphere(self._sphere_points([10.0, -5.0, 400.0], 25.0))
        np.testing.assert_allclose(center, [10.0, -5.0, 400.0], atol=1e-6)
        self.assertAlmostEqual(radius, 25.0, places=6)
        self.assertLess(rmse, 1e-6)

    def test_too_few_points_raises(self):
        with self.assertRaisesRegex(ValueError, "球面拟合有效深度点不足"):
            fit_sphere(np.zeros((10, 3)))


if __name__ == "__main__":
    unittest.main()
