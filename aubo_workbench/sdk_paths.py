#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve local third-party SDK paths for the workbench package."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent


def find_aubo_sdk_dir() -> Path | None:
    """Return the first existing local AUBO SDK directory, if present."""
    candidates = [
        PROJECT_DIR / "third_party" / "aubo_sdk",
        PROJECT_DIR.parent / "third_party" / "aubo_sdk",
        PROJECT_DIR.parent.parent / "third_party" / "aubo_sdk",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def add_aubo_sdk_to_path() -> Path | None:
    """Add the local AUBO SDK directory to sys.path and return it."""
    sdk_dir = find_aubo_sdk_dir()
    if sdk_dir is not None and str(sdk_dir) not in sys.path:
        sys.path.insert(0, str(sdk_dir))
    return sdk_dir


def aubo_sdk_hint() -> str:
    candidates = [
        PROJECT_DIR / "third_party" / "aubo_sdk",
        PROJECT_DIR.parent / "third_party" / "aubo_sdk",
        PROJECT_DIR.parent.parent / "third_party" / "aubo_sdk",
    ]
    return "、".join(str(path) for path in candidates)
