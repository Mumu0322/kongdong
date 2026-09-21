#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pytest 根配置。

把仓库根加入 ``sys.path``，使 ``aubo_workbench`` 包和根目录的入口脚本模块
在未安装（未执行 ``pip install -e .``）时也能被 tests/ 直接导入。

有了这个文件，各测试文件顶部就不需要再各写一遍
``sys.path.insert(0, PROJECT_ROOT)``。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
