#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""与具体业务无关的文件系统小工具：建目录、时间戳、矩阵<->list、路径去重/移动。"""

from __future__ import annotations

import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def make_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def matrix_to_list(T: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in np.asarray(T, dtype=np.float64).reshape(4, 4)]


def list_to_matrix(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError("expected 4x4 matrix")
    return arr


def unique_existing_paths(paths: list[str | Path]) -> list[Path]:
    """去重并只保留磁盘上真实存在的路径，保持输入顺序。"""
    seen: set[Path] = set()
    result: list[Path] = []
    for raw in paths:
        if not raw:
            continue
        path = Path(raw)
        try:
            path = path.resolve()
        except Exception:
            pass
        if path.exists() and path not in seen:
            seen.add(path)
            result.append(path)
    return result


def move_path_to_dir(path: Path, dest_dir: Path) -> str:
    """把文件移动到目标目录；若同名文件已存在，自动加数字后缀避免覆盖。"""
    make_dir(dest_dir)
    dest = dest_dir / path.name
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        for i in range(1, 1000):
            candidate = dest_dir / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                dest = candidate
                break
    shutil.move(str(path), str(dest))
    return str(dest)


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """以唯一临时文件原子替换JSON，不保留重复current副本。"""
    target = Path(path)
    make_dir(target.parent)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
