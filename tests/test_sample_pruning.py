from __future__ import annotations

import json

import numpy as np
import pytest

from aubo_workbench.config import CAMERA_CFG
from aubo_workbench.sample_pruning import (
    PruneThresholds, delete_high_error_samples, plan_high_error_samples,
)
from aubo_workbench.samples import CalibSample, next_available_sample_index


def _sample(root, index: int, rmse: float) -> CalibSample:
    name = f"sample_{index:03d}_20260923_120000_000"
    image_dir = root / "images"
    sample_dir = root / "samples"
    image_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    rgb = image_dir / f"{name}_rgb.png"
    overlay = image_dir / f"{name}_overlay.png"
    record = sample_dir / f"{name}.json"
    for path in (rgb, overlay, record):
        path.write_bytes(b"test")
    return CalibSample(
        index=index, timestamp="20260923_120000_000", T_base_tool=np.eye(4),
        T_pointcloud_board=None, robot_snapshot={}, board_status="ok",
        charuco_count=88, valid_3d_count=0, corner_rmse_mm=float("nan"),
        corner_max_error_mm=float("nan"), plane_rmse_mm=float("nan"),
        plane_inlier_count=0, board_mask_point_count=0, rgb_path=str(rgb),
        overlay_path=str(overlay), sample_json_path=str(record),
        calibration_frame="rgb_camera", T_rgb_board=np.eye(4),
        rgb_pnp_inlier_count=88, rgb_reprojection_rmse_px=rmse,
        rgb_reprojection_max_px=0.4,
    )


def test_prune_deletes_only_matching_sample_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(CAMERA_CFG, "save_dir", str(tmp_path))
    samples = [_sample(tmp_path, i, 0.4 if i == 9 else 0.2) for i in range(1, 10)]
    result = {
        "used_sample_indices": list(range(1, 10)),
        "quality": {
            "per_sample_translation_error_mm": [0.1] * 9,
            "per_sample_rotation_error_deg": [0.01] * 9,
        },
    }
    thresholds = PruneThresholds(0.35, 1.0)
    decisions = plan_high_error_samples(samples, result, thresholds)
    assert [item["index"] for item in decisions] == [9]
    remaining, report = delete_high_error_samples(samples, decisions, thresholds)
    assert len(remaining) == 8
    assert report is not None
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "complete"
    assert not (tmp_path / "images" / "sample_009_20260923_120000_000_rgb.png").exists()
    assert (tmp_path / "images" / "sample_008_20260923_120000_000_rgb.png").exists()
    assert next_available_sample_index(remaining) == 10


def test_prune_rejects_stale_solve_and_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(CAMERA_CFG, "save_dir", str(tmp_path))
    samples = [_sample(tmp_path, i, 0.4 if i == 9 else 0.2) for i in range(1, 10)]
    thresholds = PruneThresholds(0.35, 1.0)
    with pytest.raises(ValueError, match="不一致"):
        plan_high_error_samples(samples, {"used_sample_indices": list(range(1, 9))}, thresholds)
    result = {"used_sample_indices": list(range(1, 10)), "quality": {}}
    decisions = plan_high_error_samples(samples, result, thresholds)
    (tmp_path / "images" / "sample_009_20260923_120000_000_overlay.png").unlink()
    with pytest.raises(ValueError, match="缺失"):
        delete_high_error_samples(samples, decisions, thresholds)
    assert (tmp_path / "images" / "sample_009_20260923_120000_000_rgb.png").exists()


def test_handeye_residual_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.setattr(CAMERA_CFG, "save_dir", str(tmp_path))
    samples = [_sample(tmp_path, i, 0.2) for i in range(1, 10)]
    result = {
        "used_sample_indices": list(range(1, 10)),
        "quality": {
            "per_sample_translation_error_mm": [0.2] * 8 + [0.6],
            "per_sample_rotation_error_deg": [0.02] * 9,
        },
    }
    assert plan_high_error_samples(samples, result, PruneThresholds(0.35, 1.0)) == []
    decisions = plan_high_error_samples(samples, result, PruneThresholds(0.35, 1.0, True, 0.5, 0.1))
    assert [item["index"] for item in decisions] == [9]
    assert decisions[0]["reasons"] == ["handeye_translation_error_mm"]
