#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只启动手眼标定这一个功能（不经过工作台导航）。

用法：
    python run_handeye.py            # Tk GUI（默认）
    python run_handeye.py --opencv-ui # 老版 OpenCV 窗口 + 键盘操作
"""

from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from aubo_workbench.gui_handeye import main

if __name__ == "__main__":
    main()
