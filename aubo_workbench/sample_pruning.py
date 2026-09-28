"""Permanently remove RGB handeye samples that exceed explicit error gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import CAMERA_CFG, SOLVE_CFG
from .io_utils import atomic_write_json, timestamp_str
from .samples import CalibSample, resolve_sample_data_path, rewrite_sample_csv


@dataclass(frozen=True)
class PruneThresholds:
    reprojection_rmse_px: float
    reprojection_max_px: float
    use_handeye_residual: bool = False
    handeye_translation_mm: float = 0.5
    handeye_rotation_deg: float = 0.1

    def validate(self) -> None:
        values = (self.reprojection_rmse_px, self.reprojection_max_px)
        if self.use_handeye_residual:
            values += (self.handeye_translation_mm, self.handeye_rotation_deg)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("误差阈值必须是大于 0 的有限数值")


def plan_high_error_samples(
    samples: list[CalibSample], result: dict[str, Any], thresholds: PruneThresholds,
) -> list[dict[str, Any]]:
    """Return sample indexed decisions from the *same* solve, never from a stale file."""
    thresholds.validate()
    indices = [int(sample.index) for sample in samples]
    result_indices = [int(value) for value in result.get("used_sample_indices", [])]
    if len(indices) != len(set(indices)) or indices != result_indices:
        raise ValueError("求解结果与当前样本编号不一致，拒绝删除")
    quality = result.get("quality", {})
    translations = quality.get("per_sample_translation_error_mm", [])
    rotations = quality.get("per_sample_rotation_error_deg", [])
    if thresholds.use_handeye_residual and (
        len(translations) != len(samples) or len(rotations) != len(samples)
    ):
        raise ValueError("求解结果缺少逐样本手眼残差，拒绝删除")
    decisions: list[dict[str, Any]] = []
    for pos, sample in enumerate(samples):
        if sample.calibration_frame != "rgb_camera":
            raise ValueError(f"样本 {sample.index} 不是 RGB 手眼样本")
        reasons: list[str] = []
        rmse = float(sample.rgb_reprojection_rmse_px)
        maximum = float(sample.rgb_reprojection_max_px)
        if not math.isfinite(rmse) or rmse > thresholds.reprojection_rmse_px:
            reasons.append("rgb_reprojection_rmse_px")
        if not math.isfinite(maximum) or maximum > thresholds.reprojection_max_px:
            reasons.append("rgb_reprojection_max_px")
        translation = float(translations[pos]) if thresholds.use_handeye_residual else None
        rotation = float(rotations[pos]) if thresholds.use_handeye_residual else None
        if translation is not None and (
            not math.isfinite(translation) or translation > thresholds.handeye_translation_mm
        ):
            reasons.append("handeye_translation_error_mm")
        if rotation is not None and (
            not math.isfinite(rotation) or rotation > thresholds.handeye_rotation_deg
        ):
            reasons.append("handeye_rotation_error_deg")
        if reasons:
            decisions.append({
                "index": int(sample.index), "timestamp": sample.timestamp,
                "reasons": reasons, "rgb_reprojection_rmse_px": rmse,
                "rgb_reprojection_max_px": maximum,
                "handeye_translation_error_mm": translation,
                "handeye_rotation_error_deg": rotation,
            })
    if len(samples) - len(decisions) < SOLVE_CFG.min_samples_for_solve:
        raise ValueError("超阈值样本过多，删除后不足最低求解样本数；请调整阈值或补采")
    return decisions


def delete_high_error_samples(
    samples: list[CalibSample], decisions: list[dict[str, Any]], thresholds: PruneThresholds,
) -> tuple[list[CalibSample], Path | None]:
    """Delete only fully matched sample artifacts under the active capture root."""
    if not decisions:
        return samples, None
    root = Path(CAMERA_CFG.save_dir).expanduser().resolve()
    by_index = {int(sample.index): sample for sample in samples}
    delete_indices = [int(item["index"]) for item in decisions]
    if len(delete_indices) != len(set(delete_indices)) or any(index not in by_index for index in delete_indices):
        raise ValueError("删除清单含重复或未知样本编号")
    remaining = [sample for sample in samples if int(sample.index) not in set(delete_indices)]
    if len(remaining) < SOLVE_CFG.min_samples_for_solve:
        raise ValueError("删除后不足最低求解样本数")

    files: list[Path] = []
    seen: set[Path] = set()
    for index in delete_indices:
        sample = by_index[index]
        prefix = f"sample_{sample.index:03d}_{sample.timestamp}"
        for field_name, folder, suffix in (
            ("rgb_path", "images", "_rgb.png"),
            ("overlay_path", "images", "_overlay.png"),
            ("sample_json_path", "samples", ".json"),
        ):
            path = resolve_sample_data_path(sample, field_name, root)
            expected = (root / folder / f"{prefix}{suffix}").resolve()
            if path != expected or not expected.is_file() or not expected.is_relative_to(root):
                raise ValueError(f"样本 {index} 的 {field_name} 缺失或路径不匹配，拒绝删除")
            if path in seen:
                raise ValueError(f"重复文件路径，拒绝删除: {path}")
            seen.add(path)
            files.append(path)

    report = root / "deletion_reports" / f"high_error_{timestamp_str()}.json"
    payload = {
        "record_type": "rgb_handeye_permanent_deletion",
        "status": "planned",
        "thresholds": thresholds.__dict__,
        "deleted_samples": decisions,
        "deleted_files": [str(path) for path in files],
        "remaining_sample_indices": [int(sample.index) for sample in remaining],
    }
    atomic_write_json(report, payload)
    try:
        for path in files:
            path.unlink()
        rewrite_sample_csv(remaining)
    except Exception as exc:
        payload["status"] = "incomplete"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(report, payload)
        raise
    payload["status"] = "complete"
    atomic_write_json(report, payload)
    return remaining, report
