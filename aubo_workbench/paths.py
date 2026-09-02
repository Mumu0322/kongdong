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


# 运行数据留在 aubo_tools/data，便于和源码、三方依赖分开，也兼容已有实验报告路径。
DATA_DIR = _path_from_env("AUBO_WORKBENCH_DATA_DIR", AUBO_TOOLS_DIR / "data")

# 这批历史标定样本仍有追溯价值，暂不移动；通过路径常量统一引用。
CHARUCO_CALIBRATION_DIR = _path_from_env(
    "AUBO_WORKBENCH_CHARUCO_CALIBRATION_DIR",
    AUBO_TOOLS_DIR / "charuco_pointcloud_calib",
)

# 资源文件默认保持现有位置；环境变量用于在另一台机器上复用同一套源码。
MODEL_PATH = _path_from_env("AUBO_WORKBENCH_MODEL", WORKSPACE_DIR / "models" / "small_silu.pt")
CAMERA_CALIBRATION_PATH = DATA_DIR / "camera_calibration" / "current_rgb_intrinsics.json"
HANDEYE_CANDIDATE_PATH = DATA_DIR / "e7_candidates" / "e7_handeye_candidate_current.json"
HANDEYE_VALIDATION_PATH = DATA_DIR / "e7_handeye_validation_current.json"

HOLE_LOCALIZATION_RUNS_DIR = DATA_DIR / "hole_localization_runs"
# 多孔一次定位后可直接调用的最终孔位地图。它与旧的粗定位缓存分开，
# 地图保存的是已经融合完成的孔位几何和执行所需的环境指纹。
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
HANDEYE_CANDIDATE_DIR = DATA_DIR / "e7_candidates"

HANDEYE_40_POSE_PLAN_PATH = DATA_DIR / "handeye_40_pose_plan_current.json"
HANDEYE_40_POSE_PROGRESS_PATH = DATA_DIR / "handeye_40_pose_progress_current.json"
HANDEYE_40_POSE_CSV_PATH = DATA_DIR / "handeye_40_pose_plan_current.csv"
