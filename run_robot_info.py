#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""命令行只读查询 AUBO 机械臂信息。

用法：
    python run_robot_info.py --ip 192.168.50.200 --port 30004
"""

from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from aubo_workbench.robot_info import main

if __name__ == "__main__":
    raise SystemExit(main())
