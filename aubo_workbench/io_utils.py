#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""与具体业务无关的文件系统小工具：建目录、时间戳、矩阵<->list、路径去重/移动、
JSON 归一化、CSV 落盘。"""

from __future__ import annotations

import csv
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

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


def jsonable(value: Any) -> Any:
    """把 numpy 标量/数组、Path、带 to_dict 的对象递归转成可 json 序列化的值。

    原来 ``run_yolo_eye_in_hand_optimized``、``record_tcp_absolute_xy_model`` 和
    ``cad_registration`` 各有一份，此处取三者的并集作为唯一实现。
    """
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return jsonable(value.to_dict())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_dict_rows(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    fields: Sequence[str] | None = None,
    fallback_fields: Sequence[str] | None = None,
) -> Path:
    """把一组 dict 写成 CSV，统一用 utf-8-sig 以便 Excel 正确识别中文。

    ``fields`` 给定时按该顺序输出；否则取所有行键的并集排序。``rows`` 为空且
    未给 ``fields`` 时，用 ``fallback_fields`` 至少写出表头，避免产生无表头的空文件。
    """
    target = Path(path)
    make_dir(target.parent)
    materialized = list(rows)
    if fields is not None:
        fieldnames = list(fields)
    elif materialized:
        fieldnames = sorted({key for row in materialized for key in row})
    else:
        fieldnames = list(fallback_fields or [])
    with target.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)
    return target
