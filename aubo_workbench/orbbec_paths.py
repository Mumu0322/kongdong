#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Locate the bundled Orbbec Python extension and native runtime."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .paths import ORBBEC_RUNTIME_DIR


_DLL_HANDLES: list[object] = []
_ADDED_PATHS: set[str] = set()


def _runtime_paths() -> list[Path]:
    root = Path(ORBBEC_RUNTIME_DIR)
    return [
        root,
        root / "extensions" / "depthengine",
        root / "extensions" / "filters",
        root / "extensions" / "firmwareupdater",
        root / "extensions" / "frameprocessor",
    ]


def add_orbbec_runtime_path() -> Path:
    """Make the bundled Gemini 338Le extension and DLL directories importable."""

    root = Path(ORBBEC_RUNTIME_DIR)
    if not root.is_dir():
        raise FileNotFoundError(f"Gemini 338Le SDK 目录不存在: {root}")

    for path in _runtime_paths():
        if not path.exists():
            continue
        key = str(path.resolve()).casefold()
        if key not in _ADDED_PATHS:
            sys.path.insert(0, str(path))
            _ADDED_PATHS.add(key)
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None:
            # Keep handles alive for the whole process; otherwise Windows may
            # remove the DLL search path while the SDK is still in use.
            _DLL_HANDLES.append(add_dll_directory(str(path)))
    return root
