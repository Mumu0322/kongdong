#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyCharm 直接运行入口。

与 run_yolo_eye_in_hand_optimized.py 放在同一目录。
修改下方配置后直接 Run，不需要填写命令行。

默认进入两阶段孔洞定位预览，不连接运动控制。
启用真实运动前还必须显式确认工作空间安全。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from aubo_workbench.paths import HANDEYE_CANDIDATE_PATH, MODEL_PATH as DEFAULT_MODEL_PATH

# ============================================================================
# 用户配置区
# ============================================================================

TARGET_SCRIPT = Path(__file__).with_name("run_yolo_eye_in_hand_optimized.py")
MODEL_PATH = DEFAULT_MODEL_PATH
HANDEYE_PATH = HANDEYE_CANDIDATE_PATH

# 两阶段定位：原点选孔 -> 粗定位 -> 精定位。
TWO_STAGE_MODE = True
REUSE_COARSE_CACHE = True
REUSE_PERSISTENT_COARSE_CACHE = True
COARSE_HEIGHT_MM = 340.0
FINE_HEIGHT_MM = 260.0
COARSE_FRAMES = 15
FINE_FRAMES = 30
BATCH_COARSE_LOCALIZATION = True
BATCH_COARSE_FRAMES = 15
BATCH_COARSE_MIN_VALID = 10
BATCH_FINE_LOCALIZATION = True
BATCH_FINE_FRAMES = 8
BATCH_FINE_MIN_VALID = 5
BATCH_FINE_STABLE_MIN_FRAMES = 5
BATCH_FINE_SETTLE_DISCARD_FRAMES = 10
BATCH_FINE_SUPPLEMENT_ROUNDS = 1
BATCH_FINE_VIEW_MARGIN_PX = 50.0
YOLO_CONFIDENCE = 0.35

# 运动配置。默认关闭。
EXECUTE_MOTION = False
ALLOW_EXPERIMENTAL_HANDEYE = False
MOVE_FINAL_XY = False

# 只有准备真实运动时才改为 True。
I_HAVE_CHECKED_ROBOT_PATH_AND_WORKSPACE = False

SPEED_M_S = 0.02
ACC_M_S2 = 0.06

# 临时诊断偏置；None 表示使用正式 ChArUco XY 模型。
# 工具尖端不等于TCP时，应标定T_tcp_tool，不应长期依赖这里。
TCP_XY_OFFSET_MM: tuple[float, float] | None = None


# ============================================================================
# 加载并运行主脚本
# ============================================================================


def load_module():
    if not TARGET_SCRIPT.is_file():
        raise FileNotFoundError(f"找不到主脚本：{TARGET_SCRIPT}")
    spec = importlib.util.spec_from_file_location("run_yolo_eye_in_hand_optimized", TARGET_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载：{TARGET_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_arguments() -> list[str]:
    args = [
        "--model", str(MODEL_PATH),
        "--handeye", str(HANDEYE_PATH),
        "--confidence", str(YOLO_CONFIDENCE),
        "--speed-m-s", str(SPEED_M_S),
        "--acc-m-s2", str(ACC_M_S2),
        "--two-stage-hole-localization" if TWO_STAGE_MODE else "--single-stage",
        "--execute" if EXECUTE_MOTION else "--no-execute",
        "--allow-experimental-handeye"
        if ALLOW_EXPERIMENTAL_HANDEYE else "--require-validated-handeye",
        "--move-final-xy" if MOVE_FINAL_XY else "--no-move-final-xy",
    ]

    if TCP_XY_OFFSET_MM is not None:
        args.extend([
            "--tcp-xy-offset-mm", str(TCP_XY_OFFSET_MM[0]), str(TCP_XY_OFFSET_MM[1]),
        ])

    if TWO_STAGE_MODE:
        args.extend([
            "--reuse-coarse-cache" if REUSE_COARSE_CACHE else "--no-reuse-coarse-cache",
            "--reuse-persistent-coarse-cache"
            if REUSE_PERSISTENT_COARSE_CACHE else "--no-reuse-persistent-coarse-cache",
            "--coarse-height-mm", str(COARSE_HEIGHT_MM),
            "--fine-height-mm", str(FINE_HEIGHT_MM),
            "--coarse-frames", str(COARSE_FRAMES),
            "--fine-frames", str(FINE_FRAMES),
            "--batch-coarse-frames", str(BATCH_COARSE_FRAMES),
            "--batch-coarse-min-valid", str(BATCH_COARSE_MIN_VALID),
            "--batch-fine-localization"
            if BATCH_FINE_LOCALIZATION else "--no-batch-fine-localization",
            "--batch-fine-view-margin-px", str(BATCH_FINE_VIEW_MARGIN_PX),
            "--batch-fine-frames", str(BATCH_FINE_FRAMES),
            "--batch-fine-min-valid", str(BATCH_FINE_MIN_VALID),
            "--batch-fine-stable-min-frames", str(BATCH_FINE_STABLE_MIN_FRAMES),
            "--batch-fine-settle-discard-frames", str(BATCH_FINE_SETTLE_DISCARD_FRAMES),
            "--batch-fine-supplement-rounds", str(BATCH_FINE_SUPPLEMENT_ROUNDS),
        ])
        if BATCH_COARSE_LOCALIZATION:
            args.append("--batch-coarse-localization")

    if EXECUTE_MOTION and not I_HAVE_CHECKED_ROBOT_PATH_AND_WORKSPACE:
        raise RuntimeError(
            "EXECUTE_MOTION=True，但尚未将 "
            "I_HAVE_CHECKED_ROBOT_PATH_AND_WORKSPACE 设为 True"
        )

    if MOVE_FINAL_XY and not EXECUTE_MOTION:
        raise RuntimeError("MOVE_FINAL_XY=True 时必须同时启用 EXECUTE_MOTION")

    return args


def main() -> int:
    module = load_module()
    args = build_arguments()
    print("[PYCHARM] 调用参数：", args)
    return int(module.main(args))


if __name__ == "__main__":
    raise SystemExit(main())
