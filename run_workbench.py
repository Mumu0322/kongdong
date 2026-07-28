#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""默认入口：启动“机械臂信息 / TCP示教 / 眼在手标定”三合一工作台。

用法：
    python run_workbench.py
"""

from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from aubo_workbench.workbench import main

if __name__ == "__main__":
    main()
