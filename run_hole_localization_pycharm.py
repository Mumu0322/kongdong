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

from aubo_workbench.paths import HANDEYE_DIAGNOSTIC_PATH, MODEL_PATH as DEFAULT_MODEL_PATH

# ============================================================================
# 用户配置区
# ============================================================================

TARGET_SCRIPT = Path(__file__).with_name("run_yolo_eye_in_hand_optimized.py")
MODEL_PATH = DEFAULT_MODEL_PATH
HANDEYE_PATH = HANDEYE_DIAGNOSTIC_PATH

# 两阶段定位：原点选孔 -> 粗定位 -> 精定位。
TWO_STAGE_MODE = True
REUSE_COARSE_CACHE = True
REUSE_PERSISTENT_COARSE_CACHE = True
COARSE_HEIGHT_MM = 340.0
FINE_HEIGHT_MM = 260.0
# 仅用于共享精拍失败后的逐孔回退；最终低位落位仍保留60 mm。
PER_HOLE_FINE_SAFE_Z_MARGIN_MM = 20.0
COARSE_FRAMES = 15
FINE_FRAMES = 30
BATCH_COARSE_LOCALIZATION = True

# ============================================================================
# 共享粗定位现场参数
# ============================================================================
# 优化1: 增加采集帧数，提高稳定性（从15->18, 从10->12）
BATCH_COARSE_FRAMES = 18  # 原值：15
BATCH_COARSE_MIN_VALID = 12  # 原值：10

# 优化2: 启用位姿纠偏（确保启用）
BATCH_COARSE_GROUP_POSE_REFINEMENT = True

# 现场纠偏采用“小步闭环”：大偏差拆成多次移动，每步重新拍摄确认。
BATCH_COARSE_POSE_REFINE_MAX_ITERATIONS = 8
BATCH_COARSE_POSE_REFINE_FRAMES = 7  # 原值：5（默认）
BATCH_COARSE_POSE_REFINE_MIN_VALID_FRAMES = 5  # 原值：3（默认）
BATCH_COARSE_POSE_REFINE_MAX_CORRECTION_MM = 5.0
BATCH_COARSE_POSE_REFINE_MAX_CORRECTION_ROTATION_DEG = 2.0
BATCH_COARSE_POSE_REFINE_MAX_TOTAL_CORRECTION_MM = 25.0
BATCH_COARSE_POSE_REFINE_MAX_TOTAL_CORRECTION_ROTATION_DEG = 7.0

# 建图安全余量保持100mm；不要为了点云密度擅自降低机器人横移高度。
MAP_BUILD_SAFE_Z_MARGIN_MM = 100.0

# 优化5: 增加稳定等待时间
COARSE_SETTLE_DELAY_S = 0.8  # 原值：0.6（默认）

# === 分组策略优化 - 解决分组太散、边缘孔点云不完整问题 ===
# 收紧每组最大孔数，让共同位姿下的视野跨度更小。
BATCH_COARSE_MAX_GROUP_SIZE = 5

# 单组XY最大直径。
BATCH_COARSE_GROUP_MAX_XY_DIAMETER_MM = 150.0

# 收紧视野跨度，让孔更集中在画面中心。
BATCH_COARSE_MAX_VIEW_SPAN_RATIO = 0.35

# 邻接倍数沿用默认值；由视野、XY直径和法向离散共同限制组跨度。
BATCH_COARSE_GROUP_ADJACENCY_FACTOR = 1.8
BATCH_COARSE_GROUP_MAX_NORMAL_SPREAD_DEG = 3.0

# ============================================================================
# 🎯 共享精定位优化 - 提高准确性，减少回退
# ============================================================================

# === 策略1: 提高采集质量，一次成功 ===
# 优化10: 增加采集帧数，提供更多冗余数据
BATCH_FINE_FRAMES = 12  # 原值：8，增加50%数据量

# 优化11: 保持有效帧要求（避免过于宽松）
BATCH_FINE_MIN_VALID = 5  # 保持原值

# 优化12: 提高稳定性要求，确保数据质量
BATCH_FINE_STABLE_MIN_FRAMES = 7  # 原值：5

# 优化13: 增加预热丢弃帧，更充分稳定
BATCH_FINE_SETTLE_DISCARD_FRAMES = 15  # 原值：10

# === 策略2: 放宽容差，减少不必要的拒绝 ===
# 几何锚点门保持5 px，避免接受“身份匹配正确但圆心错误”的稳定结果。
BATCH_FINE_MAX_GEOMETRIC_ANCHOR_DISTANCE_PX = 5.0

# 联合定位失败时的逐孔安全回退门保持3 mm。
BATCH_FINE_FALLBACK_MAX_COARSE_TO_FINE_XY_MM = 3.0

# === 策略3: 优化恢复策略，避免无效重试 ===
# 优化16: 增加原位补帧次数（最省时的恢复方式）
BATCH_FINE_INPLACE_RECOVERY_FRAMES = 6  # 原值：4

# 每组最多一次组内重观察，避免重复运动拖慢节拍。
BATCH_FINE_SUPPLEMENT_ROUNDS = 2

# === 策略4: 大组首拍 + 一次组内受限位姿微调 ===
BATCH_FINE_MAX_VIEW_SPAN_RATIO = 0.55
BATCH_FINE_MAX_GROUP_SIZE = 4
BATCH_FINE_SUPPLEMENT_MAX_VIEW_SPAN_RATIO = 0.55
BATCH_FINE_IN_GROUP_POSE_ADJUSTMENT = True
BATCH_FINE_IN_GROUP_MAX_ADJUSTMENTS = 2
BATCH_FINE_IN_GROUP_MAX_XY_MM = 8.0
BATCH_FINE_IN_GROUP_MAX_Z_MM = 3.0
BATCH_FINE_IN_GROUP_MAX_ROTATION_DEG = 2.0
BATCH_FINE_IN_GROUP_MIN_NORMAL_HOLES = 2
BATCH_FINE_IN_GROUP_MAX_NORMAL_SPREAD_DEG = 3.0

# ============================================================================

BATCH_FINE_LOCALIZATION = True
BATCH_FINE_VIEW_MARGIN_PX = 50.0
YOLO_CONFIDENCE = 0.35

# 运动配置。默认关闭。
EXECUTE_MOTION = False
ALLOW_EXPERIMENTAL_HANDEYE = True
MOVE_FINAL_XY = True

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
            "--per-hole-fine-safe-z-margin-mm", str(PER_HOLE_FINE_SAFE_Z_MARGIN_MM),
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
            "--batch-fine-max-view-span-ratio", str(BATCH_FINE_MAX_VIEW_SPAN_RATIO),
            "--batch-fine-max-group-size", str(BATCH_FINE_MAX_GROUP_SIZE),
            "--batch-fine-supplement-max-view-span-ratio",
            str(BATCH_FINE_SUPPLEMENT_MAX_VIEW_SPAN_RATIO),
            "--batch-fine-in-group-pose-adjustment"
            if BATCH_FINE_IN_GROUP_POSE_ADJUSTMENT
            else "--no-batch-fine-in-group-pose-adjustment",
            "--batch-fine-in-group-max-adjustments",
            str(BATCH_FINE_IN_GROUP_MAX_ADJUSTMENTS),
            "--batch-fine-in-group-max-xy-mm", str(BATCH_FINE_IN_GROUP_MAX_XY_MM),
            "--batch-fine-in-group-max-z-mm", str(BATCH_FINE_IN_GROUP_MAX_Z_MM),
            "--batch-fine-in-group-max-rotation-deg",
            str(BATCH_FINE_IN_GROUP_MAX_ROTATION_DEG),
            "--batch-fine-in-group-min-normal-holes",
            str(BATCH_FINE_IN_GROUP_MIN_NORMAL_HOLES),
            "--batch-fine-in-group-max-normal-spread-deg",
            str(BATCH_FINE_IN_GROUP_MAX_NORMAL_SPREAD_DEG),
        ])
        if BATCH_COARSE_LOCALIZATION:
            args.append("--batch-coarse-localization")

        # 添加优化参数（如果定义了这些变量）
        if "BATCH_COARSE_GROUP_POSE_REFINEMENT" in globals():
            if BATCH_COARSE_GROUP_POSE_REFINEMENT:
                args.append("--batch-coarse-group-pose-refinement")
        if "BATCH_COARSE_POSE_REFINE_MAX_ITERATIONS" in globals():
            args.extend(["--batch-coarse-pose-refine-max-iterations",
                        str(BATCH_COARSE_POSE_REFINE_MAX_ITERATIONS)])
        if "BATCH_COARSE_POSE_REFINE_FRAMES" in globals():
            args.extend(["--batch-coarse-pose-refine-frames",
                        str(BATCH_COARSE_POSE_REFINE_FRAMES)])
        if "BATCH_COARSE_POSE_REFINE_MIN_VALID_FRAMES" in globals():
            args.extend(["--batch-coarse-pose-refine-min-valid-frames",
                        str(BATCH_COARSE_POSE_REFINE_MIN_VALID_FRAMES)])
        if "BATCH_COARSE_POSE_REFINE_MAX_CORRECTION_MM" in globals():
            args.extend(["--batch-coarse-pose-refine-max-correction-mm",
                        str(BATCH_COARSE_POSE_REFINE_MAX_CORRECTION_MM)])
        if "BATCH_COARSE_POSE_REFINE_MAX_CORRECTION_ROTATION_DEG" in globals():
            args.extend(["--batch-coarse-pose-refine-max-correction-rotation-deg",
                        str(BATCH_COARSE_POSE_REFINE_MAX_CORRECTION_ROTATION_DEG)])
        if "BATCH_COARSE_POSE_REFINE_MAX_TOTAL_CORRECTION_MM" in globals():
            args.extend(["--batch-coarse-pose-refine-max-total-correction-mm",
                        str(BATCH_COARSE_POSE_REFINE_MAX_TOTAL_CORRECTION_MM)])
        if "BATCH_COARSE_POSE_REFINE_MAX_TOTAL_CORRECTION_ROTATION_DEG" in globals():
            args.extend(["--batch-coarse-pose-refine-max-total-correction-rotation-deg",
                        str(BATCH_COARSE_POSE_REFINE_MAX_TOTAL_CORRECTION_ROTATION_DEG)])
        if "MAP_BUILD_SAFE_Z_MARGIN_MM" in globals():
            args.extend(["--map-build-safe-z-margin-mm",
                        str(MAP_BUILD_SAFE_Z_MARGIN_MM)])
        if "COARSE_SETTLE_DELAY_S" in globals():
            args.extend(["--coarse-settle-delay-s",
                        str(COARSE_SETTLE_DELAY_S)])

        # 添加分组策略优化参数
        if "BATCH_COARSE_MAX_GROUP_SIZE" in globals():
            args.extend(["--batch-coarse-max-group-size",
                        str(BATCH_COARSE_MAX_GROUP_SIZE)])
        if "BATCH_COARSE_GROUP_MAX_XY_DIAMETER_MM" in globals():
            args.extend(["--batch-coarse-group-max-xy-diameter-mm",
                        str(BATCH_COARSE_GROUP_MAX_XY_DIAMETER_MM)])
        if "BATCH_COARSE_MAX_VIEW_SPAN_RATIO" in globals():
            args.extend(["--batch-coarse-max-view-span-ratio",
                        str(BATCH_COARSE_MAX_VIEW_SPAN_RATIO)])
        if "BATCH_COARSE_GROUP_ADJACENCY_FACTOR" in globals():
            args.extend(["--batch-coarse-group-adjacency-factor",
                        str(BATCH_COARSE_GROUP_ADJACENCY_FACTOR)])
        if "BATCH_COARSE_GROUP_MAX_NORMAL_SPREAD_DEG" in globals():
            args.extend(["--batch-coarse-group-max-normal-spread-deg",
                        str(BATCH_COARSE_GROUP_MAX_NORMAL_SPREAD_DEG)])

        # 添加精定位优化参数
        if "BATCH_FINE_MAX_GEOMETRIC_ANCHOR_DISTANCE_PX" in globals():
            args.extend(["--batch-fine-max-geometric-anchor-distance-px",
                        str(BATCH_FINE_MAX_GEOMETRIC_ANCHOR_DISTANCE_PX)])
        if "BATCH_FINE_FALLBACK_MAX_COARSE_TO_FINE_XY_MM" in globals():
            args.extend(["--batch-fine-fallback-max-coarse-to-fine-xy-mm",
                        str(BATCH_FINE_FALLBACK_MAX_COARSE_TO_FINE_XY_MM)])
        if "BATCH_FINE_INPLACE_RECOVERY_FRAMES" in globals():
            args.extend(["--batch-fine-inplace-recovery-frames",
                        str(BATCH_FINE_INPLACE_RECOVERY_FRAMES)])

    if EXECUTE_MOTION and not I_HAVE_CHECKED_ROBOT_PATH_AND_WORKSPACE:
        raise RuntimeError(
            "EXECUTE_MOTION=True，但尚未将 "
            "I_HAVE_CHECKED_ROBOT_PATH_AND_WORKSPACE 设为 True"
        )

    return args


def main() -> int:
    module = load_module()
    args = build_arguments()
    print("[PYCHARM] 调用参数：", args)
    return int(module.main(args))


if __name__ == "__main__":
    raise SystemExit(main())
