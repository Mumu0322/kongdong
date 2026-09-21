#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""六扇区地图的编号与标称角度工具。"""

from __future__ import annotations


ROTARY_SECTOR_COUNT = 6


def validate_sector_id(sector_id: int, *, sector_count: int = ROTARY_SECTOR_COUNT) -> int:
    value = int(sector_id)
    if not 1 <= value <= int(sector_count):
        raise ValueError(
            f"扇区编号必须在 1..{int(sector_count)} 范围内，实际为 {sector_id!r}"
        )
    return value


def sector_key(sector_id: int, *, sector_count: int = ROTARY_SECTOR_COUNT) -> str:
    return f"S{validate_sector_id(sector_id, sector_count=sector_count):02d}"


def nominal_sector_angle_deg(
    sector_id: int,
    *,
    sector_count: int = ROTARY_SECTOR_COUNT,
) -> float:
    value = validate_sector_id(sector_id, sector_count=sector_count)
    return float((value - 1) * 360.0 / float(sector_count))
