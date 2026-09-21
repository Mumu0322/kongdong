#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""静止伞架单帧自动分区/选孔离线预览。

这个入口只运行图像检测和自动分区，不连接机器人、不读取深度，也不生成
340 mm 粗定位地图。它用于先用现有照片或保存的RGB帧检查 ``origin_px``、
ROI、扇区边界和去重参数，再把同一配置交给两阶段现场入口。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aubo_workbench.auto_sector_selection import (  # noqa: E402
    load_auto_sector_config,
    render_auto_sector_overlay,
    select_auto_holes,
    validate_selection_image,
)
from aubo_workbench.hole_localization_vision import detect, load_yolo  # noqa: E402
from aubo_workbench.io_utils import atomic_write_json, jsonable  # noqa: E402
from aubo_workbench.paths import HOLE_LOCALIZATION_SECTOR_INFO_DIR  # noqa: E402
from aubo_workbench.sector_info import write_sector_info_snapshots  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="静止伞架单帧自动扇区划分与选孔离线预览"
    )
    parser.add_argument("--image", type=Path, required=True, help="待分析的RGB图像")
    parser.add_argument("--model", type=Path, required=True, help="YOLO模型路径")
    parser.add_argument(
        "--auto-sector-config", type=Path, required=True,
        help="静态自动分区配置JSON（必须包含origin_px）",
    )
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument(
        "--auto-sector-ids", type=int, nargs="+", default=None, metavar="SECTOR",
        help="只分析指定扇区；默认使用配置中的全部扇区",
    )
    parser.add_argument(
        "--auto-exclude-boundary-candidates", action="store_true",
        help="将边界候选标记为配置排除；边界候选始终不会直接执行",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="输出目录；默认写到图像同目录的auto_sector_selection_preview",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图像：{args.image}")
    config = load_auto_sector_config(
        args.auto_sector_config,
        active_sector_ids=args.auto_sector_ids,
        include_boundary_candidates=(False if args.auto_exclude_boundary_candidates else None),
    )
    validate_selection_image(image, config)
    model = load_yolo(args.model)
    detections = detect(model, image, float(args.confidence))
    result = select_auto_holes(detections, config)
    output_dir = args.output_dir or args.image.parent / "auto_sector_selection_preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "auto_sector_selection_overlay.png"
    report_path = output_dir / "auto_sector_selection.json"
    overlay = render_auto_sector_overlay(image, detections, result)
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"自动分区叠加图写入失败：{overlay_path}")
    report_payload = result.to_dict()
    report_payload["input"] = {
        "image": str(args.image),
        "model": str(args.model),
        "confidence": float(args.confidence),
    }
    atomic_write_json(report_path, jsonable(report_payload))
    sector_manifest = write_sector_info_snapshots(
        result,
        HOLE_LOCALIZATION_SECTOR_INFO_DIR,
        source_run_dir=output_dir,
        overlay=overlay,
        snapshot_name=f"preview-{args.image.stem}",
    )
    print(
        f"检测候选={len(detections)}，可执行选孔={len(result.selected_indices)}，"
        f"排除={len(result.rejected)}，重复组={len(result.duplicate_groups)}"
    )
    print(f"叠加图：{overlay_path}")
    print(f"报告：{report_path}")
    print(f"扇区信息目录：{HOLE_LOCALIZATION_SECTOR_INFO_DIR}")
    for warning in result.warnings:
        print(f"[WARNING] {warning}")
    return 0 if result.selected_indices else 2


if __name__ == "__main__":
    raise SystemExit(main())
