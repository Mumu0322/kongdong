#!/usr/bin/env python3
"""Build an XY seed-correction artifact from a confirmed per-hole run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aubo_workbench.hole_map import load_hole_map, resolve_hole_map_path
from aubo_workbench.hole_map_seed_correction import (
    DEFAULT_SEED_CORRECTION_FILENAME,
    build_seed_correction_model,
    save_seed_correction_model,
)


def _reference_report_path(value: Path) -> Path:
    candidate = value.expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "report.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"逐孔参考报告不存在：{candidate}")
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用已确认成功的逐孔精定位结果生成地图种子XY纠正模型",
    )
    parser.add_argument("--map", dest="map_path", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--sector-id", type=int, default=None)
    parser.add_argument("--match-gate-mm", type=float, default=10.0)
    parser.add_argument("--max-correction-mm", type=float, default=5.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="默认写入地图目录下的 hole_seed_correction.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    map_path = resolve_hole_map_path(args.map_path).resolve()
    reference_path = _reference_report_path(args.reference_run)
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else map_path.parent / DEFAULT_SEED_CORRECTION_FILENAME
    )
    map_payload = load_hole_map(map_path)
    reference_report = json.loads(reference_path.read_text(encoding="utf-8"))
    model = build_seed_correction_model(
        map_payload,
        reference_report,
        map_path=map_path,
        reference_report_path=reference_path,
        sector_id=args.sector_id,
        match_gate_mm=args.match_gate_mm,
        max_correction_mm=args.max_correction_mm,
    )
    save_seed_correction_model(model, output)
    matching = model["matching"]
    rigid = model["global_rigid_xy"]
    print(f"已生成地图种子纠正模型：{output}")
    print(
        f"匹配 {matching['matched_count']}/{matching['map_hole_count']} 个地图孔；"
        f"匹配距离中位数={matching['distance_median_mm']:.3f} mm，"
        f"最大={matching['distance_max_mm']:.3f} mm"
    )
    print(
        f"参考整体偏移 yaw={rigid['yaw_deg']:.4f} deg，"
        f"局部残差中位数={rigid['residual_median_mm']:.3f} mm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
