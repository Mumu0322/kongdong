#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 之间共享的小工具：把 print 重定向到 Tk 日志框、解析手动位姿输入。"""

from __future__ import annotations

import queue
from typing import Any


class GuiLogWriter:
    """实现 write/flush，可以整体赋值给 sys.stdout/sys.stderr，把打印导入 Tk 队列。"""

    def __init__(self, log_queue: "queue.Queue[str]", original: Any | None = None) -> None:
        self.log_queue = log_queue
        self.original = original

    def write(self, text: str) -> int:
        if self.original is not None:
            try:
                self.original.write(text)
            except Exception:
                pass
        if text:
            self.log_queue.put(text)
        return len(text)

    def flush(self) -> None:
        if self.original is not None:
            try:
                self.original.flush()
            except Exception:
                pass


def parse_manual_pose_text(text: str) -> tuple[float, float, float, float, float, float]:
    parts = text.replace(",", " ").replace("，", " ").split()
    if len(parts) != 6:
        raise ValueError("请输入 6 个数：x y z rx ry rz，单位 m / rad")
    try:
        return tuple(float(v) for v in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError("手动位姿包含非数字内容") from exc
