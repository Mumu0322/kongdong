#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动 ChArUco 指定角点的眼在手上坐标/误差实验（只读，不控制运动）。"""

from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from aubo_workbench.charuco_point_experiment import main


if __name__ == "__main__":
    raise SystemExit(main())

