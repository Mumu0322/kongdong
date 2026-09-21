#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主项目的路径约定。

业务代码只从这里取得项目、数据和模型路径。这样目录整理或
更换运行机器时只需要调整环境变量，不必在十几个入口脚本里逐处修改绝对路径。

这里仅描述依赖位置，不移动或修改任何厂商 SDK/驱动文件。
"""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
AUBO_TOOLS_DIR = PROJECT_DIR.parent
WORKSPACE_DIR = AUBO_TOOLS_DIR.parent

DEFAULT_ROBOT_IP = "192.168.50.200"
DEFAULT_ROBOT_PORT = 30004
DEFAULT_ROBOT_USER = "AUBO"
# 密码不再写入业务源码；现场可通过环境变量或 GUI/命令行输入。
DEFAULT_ROBOT_PASSWORD = os.environ.get("AUBO_PASSWORD", "")
DEFAULT_ROBOT_TIMEOUT_MS = 3000


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else default


# 第三方运行组件随副本部署；环境变量允许现场把它们放到独立安装目录。
# 这里不再回退到原 MM 工作区，避免新副本悄悄加载旧 SDK。
AUBO_SDK_DIR = _path_from_env(
    "AUBO_WORKBENCH_AUBO_SDK_DIR",
    WORKSPACE_DIR / "third_party" / "aubo_sdk",
)
ORBBEC_RUNTIME_DIR = _path_from_env(
    "AUBO_WORKBENCH_ORBBEC_RUNTIME_DIR",
    WORKSPACE_DIR / "third_party" / "orbbec_runtime",
)


# 运行数据留在 aubo_tools/data，便于和源码、三方依赖分开，也兼容已有实验报告路径。
DATA_DIR = _path_from_env("AUBO_WORKBENCH_DATA_DIR", AUBO_TOOLS_DIR / "data")

# 当前副本从空白标定目录开始；现场样本由本副本重新采集。
CHARUCO_CALIBRATION_DIR = _path_from_env(
    "AUBO_WORKBENCH_CHARUCO_CALIBRATION_DIR",
    AUBO_TOOLS_DIR / "charuco_pointcloud_calib",
)

# 资源文件默认保持现有位置；环境变量用于在另一台机器上复用同一套源码。
MODEL_PATH = _path_from_env("AUBO_WORKBENCH_MODEL", WORKSPACE_DIR / "models" / "small_silu.pt")
CAMERA_CALIBRATION_PATH = DATA_DIR / "camera_calibration" / "current_rgb_intrinsics.json"
HANDEYE_CANDIDATE_PATH = DATA_DIR / "e7_candidates" / "e7_handeye_candidate_current.json"
HANDEYE_DIAGNOSTIC_PATH = DATA_DIR / "handeye_diagnostic_current.json"
HANDEYE_VALIDATION_PATH = DATA_DIR / "e7_handeye_validation_current.json"

HOLE_LOCALIZATION_RUNS_DIR = DATA_DIR / "hole_localization_runs"
# 自动扇区的定义、按扇区拆分的候选和审计快照单独保存，避免与一次运行
# 的临时文件、孔位地图正文混在一起。可用环境变量迁移到现场指定目录。
HOLE_LOCALIZATION_SECTOR_INFO_DIR = _path_from_env(
    "AUBO_WORKBENCH_SECTOR_INFO_DIR",
    DATA_DIR / "hole_localization_sector_info",
)
# 340 mm 粗定位地图。地图只保存粗中心、平面、法向和质量；260 mm 精定位
# 与最终 TCP 目标在每次调用的当前运行中重新计算。旧粗缓存目录独立保留。
HOLE_LOCALIZATION_MAPS_DIR = DATA_DIR / "hole_localization_maps"
# 当前可执行地图的稳定入口。它是一个小型指针文件，指向
# HOLE_LOCALIZATION_MAPS_DIR 下最新一次完整建图结果，不复制/覆盖地图正文。
HOLE_LOCALIZATION_CURRENT_MAP_PATH = HOLE_LOCALIZATION_MAPS_DIR / "current.json"
# 旧两阶段的跨运行局部点云缓存。缓存中的几何均落在机器人 base 坐标系，
# 每次复用前仍必须在340 mm现场验证。
HOLE_LOCALIZATION_COARSE_CACHE_DIR = DATA_DIR / "hole_localization_coarse_cache"
CHARUCO_HEIGHT_ERROR_DIR = DATA_DIR / "charuco_height_error"
CHARUCO_POINT_EXPERIMENTS_DIR = DATA_DIR / "charuco_point_experiments"
TCP_ABSOLUTE_XY_MODEL_DIR = DATA_DIR / "tcp_absolute_xy_model"
# 新副本不携带旧补偿模型。完成本次 TCP-XY 实验后，可通过环境变量显式
# 指定经复核的 report.json；默认 current.json 由现场流程产生或由操作者指定。
TCP_XY_MODEL_PATH = _path_from_env(
    "AUBO_WORKBENCH_TCP_XY_MODEL",
    TCP_ABSOLUTE_XY_MODEL_DIR / "current.json",
)
HANDEYE_CANDIDATE_DIR = DATA_DIR / "e7_candidates"

HANDEYE_40_POSE_PLAN_PATH = DATA_DIR / "handeye_40_pose_plan_current.json"
HANDEYE_40_POSE_PROGRESS_PATH = DATA_DIR / "handeye_40_pose_progress_current.json"
HANDEYE_40_POSE_CSV_PATH = DATA_DIR / "handeye_40_pose_plan_current.csv"
