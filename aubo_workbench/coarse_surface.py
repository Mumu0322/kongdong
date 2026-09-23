"""Offline geometry and diagnostics for coarse hole surface sampling."""
from __future__ import annotations

import numpy as np

from .fitting import fit_plane_model


def ring_coverage(pixels: np.ndarray, center: tuple[float, float]) -> dict:
    """Count supported 10-degree sectors; isolated pixels do not fill gaps."""
    delta = np.asarray(pixels, dtype=float).reshape(-1, 2) - np.asarray(center)
    angle = np.mod(np.arctan2(delta[:, 1], delta[:, 0]), 2 * np.pi)
    bins = np.minimum(35, (angle * 36 / (2 * np.pi)).astype(int))
    counts = np.bincount(bins, minlength=36)
    supported = counts >= 3
    longest = current = 0
    for present in np.tile(supported, 2):
        current = 0 if present else current + 1
        longest = max(longest, current)
    return {"coverage_ratio": float(np.mean(supported)),
            "max_gap_deg": float(min(36, longest) * 10),
            "sector_counts": counts.tolist()}


def select_front_surface(points: np.ndarray, pixels: np.ndarray, center: tuple[float, float]):
    """Seed the near surface, then grow by plane distance, not camera Z.

    The depth seed prevents selecting a dominant, more distant hole bottom.
    Each refit samples sectors equally so one dense arc cannot dominate.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    front_z = float(np.percentile(points[:, 2], 5))
    seed = points[:, 2] <= front_z + 8.0
    minimum = max(80, int(len(points) * 0.03))
    if seed.sum() < minimum:
        raise ValueError("孔口前表面有效点不足")
    delta = np.asarray(pixels) - np.asarray(center)
    bins = np.minimum(35, (np.mod(np.arctan2(delta[:, 1], delta[:, 0]), 2*np.pi) * 36/(2*np.pi)).astype(int))

    def balanced_fit(mask):
        indices = []
        for sector in range(36):
            ix = np.flatnonzero(mask & (bins == sector))
            if len(ix):
                indices.extend(ix[np.linspace(0, len(ix)-1, min(64, len(ix))).astype(int)])
        return fit_plane_model(points[indices])[:3]

    active = seed.copy()
    for _ in range(5):
        normal, anchor, rmse = balanced_fit(active)
        if abs(normal[2]) < 0.5:
            raise ValueError("孔口平面倾斜过大，拒绝用不可靠法向规划运动")
        tolerance = max(2.0, min(4.0, 2.5 * rmse))
        residual = np.abs((points-anchor) @ normal)
        updated = residual <= tolerance
        if updated.sum() < minimum:
            raise ValueError("孔口平面内点不足")
        if np.array_equal(updated, active):
            break
        active = updated
    normal, anchor, _ = balanced_fit(active)
    rmse = float(np.sqrt(np.mean(((points[active]-anchor) @ normal)**2)))
    return normal, anchor, rmse, active, front_z


def save_surface_diagnostic(path, plane, *, hole_id=None, frame_index=None,
                            T_base_camera=None):
    """Persist the last raw annulus and mask, including quality-rejected holes."""
    if plane.raw_points_camera_mm is None:
        return None
    arrays = {
        "raw_points_camera_mm": plane.raw_points_camera_mm,
        "raw_pixels": plane.raw_pixels,
        "selected_mask": plane.surface_selected_mask,
        "ring_pixels": plane.ring_pixels if plane.ring_pixels is not None else plane.raw_pixels,
        "plane_normal_camera": plane.normal_camera,
        "plane_anchor_camera_mm": plane.surface_plane_point_camera_mm,
        "hole_id": -1 if hole_id is None else hole_id,
        "frame_index": -1 if frame_index is None else frame_index,
    }
    if T_base_camera is not None:
        arrays["T_base_camera"] = np.asarray(T_base_camera, dtype=np.float64).reshape(4, 4)
    np.savez_compressed(path, **arrays)
    return str(path)
