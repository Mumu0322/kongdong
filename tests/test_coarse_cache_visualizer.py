from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from aubo_workbench.camera import CameraIntrinsics
from aubo_workbench.coarse_cache import CoarseCacheEntry, save_cache_entries
from tools.visualize_coarse_cache import (
    load_cache_cloud,
    project_camera_points,
    render_cache_cloud,
    select_frame_points,
    transform_points_to_base,
)


def _entry() -> CoarseCacheEntry:
    return CoarseCacheEntry(
        hole_id=1,
        T_base_camera_build=np.eye(4),
        T_tcp_camera=np.eye(4),
        tcp_pose_m_rad=[0.0] * 6,
        camera_serial="camera-1",
        handeye_path="handeye.json",
        intrinsics=CameraIntrinsics(
            1280, 720, 800.0, 800.0, 640.0, 360.0, (),
        ).as_dict(),
        center_px=np.array([640.0, 360.0]),
        point_camera_mm=np.array([0.0, 0.0, 340.0]),
        plane_point_camera_mm=np.array([0.0, 0.0, 338.0]),
        normal_camera=np.array([0.0, 0.0, -1.0]),
        point_base_mm=np.array([0.0, 0.0, 340.0]),
        plane_point_base_mm=np.array([0.0, 0.0, 338.0]),
        normal_base=np.array([0.0, 0.0, -1.0]),
        plane_rmse_mm=1.0,
        valid_frames=2,
        total_frames=2,
        center_scatter_p95_px=0.2,
        ring_points_median=50.0,
        surface_model="local_tangent_plane",
        frame_indices=np.array([0, 1]),
        frame_centers_px=np.array([[640.0, 360.0], [640.2, 360.1]]),
        frame_plane_rmse_mm=np.array([1.0, 1.1]),
        points_camera_mm_by_frame=(
            np.array([[0.0, 0.0, 340.0], [1.0, 0.0, 340.0]]),
            np.array([[0.0, 1.0, 340.0], [1.0, 1.0, 340.0]]),
        ),
    )


class CoarseCacheVisualizerTests(unittest.TestCase):
    def test_load_npz_and_select_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            save_cache_entries(directory, {1: _entry()})
            cloud = load_cache_cloud(Path(directory) / "hole_01.npz", 1)

        self.assertEqual(cloud.hole_id, 1)
        self.assertEqual(len(cloud.points_camera_mm_by_frame), 2)
        points, labels = select_frame_points(cloud)
        self.assertEqual(points.shape, (4, 3))
        self.assertEqual(labels.tolist(), [0, 0, 1, 1])
        frame_points, frame_labels = select_frame_points(cloud, 1)
        self.assertEqual(frame_points.shape, (2, 3))
        self.assertEqual(frame_labels.tolist(), [1, 1])

    def test_projection_uses_cached_rgb_intrinsics(self) -> None:
        intrinsics = CameraIntrinsics(
            1280, 720, 800.0, 800.0, 640.0, 360.0, (),
        ).as_dict()
        pixels, valid = project_camera_points(
            np.array([[0.0, 0.0, 340.0], [85.0, 0.0, 340.0], [0.0, 0.0, -1.0]]),
            intrinsics,
        )
        self.assertEqual(valid.tolist(), [True, True, False])
        np.testing.assert_allclose(pixels[0], [640.0, 360.0])
        np.testing.assert_allclose(pixels[1], [840.0, 360.0])
        self.assertTrue(np.isnan(pixels[2]).all())

    def test_transform_points_to_base_applies_rotation_and_translation(self) -> None:
        transform = np.eye(4)
        transform[:3, 3] = [10.0, 20.0, 30.0]
        points = transform_points_to_base([[1.0, 2.0, 3.0]], transform)
        np.testing.assert_allclose(points, [[11.0, 22.0, 33.0]])

    def test_render_can_overlay_projected_points_on_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_cache_entries(root / "cache", {1: _entry()})
            rgb_path = root / "rgb.png"
            cv2.imwrite(str(rgb_path), np.zeros((720, 1280, 3), dtype=np.uint8))
            output_path = root / "preview.png"
            cloud = load_cache_cloud(root / "cache", 1)
            saved = render_cache_cloud(
                cloud,
                rgb_path=rgb_path,
                output_path=output_path,
                show=False,
            )
            self.assertEqual(saved, output_path)
            self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
