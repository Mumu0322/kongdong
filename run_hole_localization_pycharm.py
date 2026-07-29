#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyCharm 直接运行入口。

与 run_yolo_eye_in_hand_optimized.py 放在同一目录。
修改下方配置后直接 Run，不需要填写命令行。

默认 EXECUTE_MOTION=False，只生成运动预览。
即使打开运动，主脚本仍会逐步要求输入 m 确认。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


# ============================================================================
# 用户配置区
# ============================================================================

TARGET_SCRIPT = Path(__file__).with_name("run_yolo_eye_in_hand_optimized.py")
MODEL_PATH = Path(r"C:\MM\models\small_silu.pt")
HANDEYE_PATH = Path(r"C:\MM\aubo_tools\data\e7_candidates\e7_handeye_candidate_current.json")

# 两阶段定位：原点选孔 -> 粗定位 -> 精定位。
TWO_STAGE_MODE = True
COARSE_HEIGHT_MM = 340.0
FINE_HEIGHT_MM = 260.0
COARSE_FRAMES = 15
FINE_FRAMES = 30
HOLE_COUNT = 3
YOLO_CONFIDENCE = 0.35

# 运动配置。默认关闭。
EXECUTE_MOTION = False
ALLOW_EXPERIMENTAL_HANDEYE = False
MOVE_FINAL_XY = False

# 只有准备真实运动时才改为 True。
I_HAVE_CHECKED_ROBOT_PATH_AND_WORKSPACE = False

SPEED_M_S = 0.02
ACC_M_S2 = 0.06

# 临时诊断偏置，正常应保持0,0。
# 工具尖端不等于TCP时，应标定T_tcp_tool，不应长期依赖这里。
TCP_XY_OFFSET_MM = (0.0, 0.0)


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
        "--tcp-xy-offset-mm", str(TCP_XY_OFFSET_MM[0]), str(TCP_XY_OFFSET_MM[1]),
    ]

    if TWO_STAGE_MODE:
        args.extend([
            "--two-stage-hole-localization",
            "--coarse-height-mm", str(COARSE_HEIGHT_MM),
            "--fine-height-mm", str(FINE_HEIGHT_MM),
            "--coarse-frames", str(COARSE_FRAMES),
            "--fine-frames", str(FINE_FRAMES),
            "--hole-count", str(HOLE_COUNT),
        ])

    if EXECUTE_MOTION:
        if not I_HAVE_CHECKED_ROBOT_PATH_AND_WORKSPACE:
            raise RuntimeError(
                "EXECUTE_MOTION=True，但尚未将 "
                "I_HAVE_CHECKED_ROBOT_PATH_AND_WORKSPACE 设为 True"
            )
        args.append("--execute")

    if ALLOW_EXPERIMENTAL_HANDEYE:
        args.append("--allow-experimental-handeye")

    if MOVE_FINAL_XY:
        if not EXECUTE_MOTION:
            raise RuntimeError("MOVE_FINAL_XY=True 时必须同时启用 EXECUTE_MOTION")
        args.append("--move-final-xy")

    return args


def main() -> int:
    module = load_module()
    args = build_arguments()
    print("[PYCHARM] 调用参数：", args)
    return int(module.main(args))


if __name__ == "__main__":
    raise SystemExit(main())
