#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenCV 画面上的中文文字、面板、进度条等绘制工具。

纯绘图逻辑，不依赖相机/机械臂，方便单独测试和复用。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
except Exception:  # Pillow 不可用时，中文会自动退回 OpenCV 英文/ASCII 显示
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore

_FONT_CACHE: dict[tuple[int, str], Any] = {}


def get_chinese_font(font_size: int):
    """优先使用 Windows 常见中文字体；找不到时退回 PIL 默认字体或 None。"""
    if ImageFont is None:
        return None
    font_size = int(max(12, font_size))
    font_candidates = [
        os.environ.get("OPENCV_CHINESE_FONT", ""),
        r"C:\Windows\Fonts\msyh.ttc",       # 微软雅黑
        r"C:\Windows\Fonts\simhei.ttf",     # 黑体
        r"C:\Windows\Fonts\simsun.ttc",     # 宋体
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for font_path in font_candidates:
        if not font_path:
            continue
        key = (font_size, font_path)
        if key in _FONT_CACHE:
            return _FONT_CACHE[key]
        if Path(font_path).exists():
            try:
                font = ImageFont.truetype(font_path, font_size)
                _FONT_CACHE[key] = font
                return font
            except Exception:
                continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def draw_unicode_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    font_size: int = 22,
    thickness: int = 1,
) -> None:
    """在 OpenCV 图像上写中文。没有 Pillow/中文字体时尽量不报错。

    为了避免实时画面卡顿，中文绘制只转换文字附近的小 ROI，
    不会每次都把整张图转成 PIL。
    """
    # 纯 ASCII 仍用 OpenCV，速度更快。
    if all(ord(ch) < 128 for ch in text):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, max(0.35, font_size / 34.0), color, thickness)
        return
    if Image is None or ImageDraw is None or ImageFont is None:
        safe_text = text.encode("ascii", "ignore").decode("ascii") or "CN text"
        cv2.putText(img, safe_text, org, cv2.FONT_HERSHEY_SIMPLEX, max(0.35, font_size / 34.0), color, thickness)
        return
    font = get_chinese_font(font_size)
    if font is None:
        safe_text = text.encode("ascii", "ignore").decode("ascii") or "CN text"
        cv2.putText(img, safe_text, org, cv2.FONT_HERSHEY_SIMPLEX, max(0.35, font_size / 34.0), color, thickness)
        return

    h, w = img.shape[:2]
    x, y = int(org[0]), int(org[1])
    text_x = x
    text_y = int(y - font_size * 0.90)

    try:
        dummy = Image.new("RGB", (4, 4))
        dummy_draw = ImageDraw.Draw(dummy)
        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        text_w = max(1, int(bbox[2] - bbox[0]))
        text_h = max(1, int(bbox[3] - bbox[1]))
    except Exception:
        text_w = int(len(text) * font_size)
        text_h = int(font_size * 1.3)

    pad = 4
    x0 = max(0, text_x - pad)
    y0 = max(0, text_y - pad)
    x1 = min(w, text_x + text_w + pad)
    y1 = min(h, text_y + text_h + pad + 6)
    if x1 <= x0 or y1 <= y0:
        return

    roi = img[y0:y1, x0:x1]
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    b, g, r = color
    draw.text((text_x - x0, text_y - y0), text, fill=(int(r), int(g), int(b)), font=font)
    img[y0:y1, x0:x1] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def draw_text_panel(img: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]], x: int = 16, y: int = 28) -> None:
    line_h = 28
    w = 860
    h = max(46, line_h * len(lines) + 16)
    panel = img.copy()
    cv2.rectangle(panel, (x - 8, y - 24), (x + w, y - 24 + h), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.45, img, 0.55, 0, dst=img)
    for i, (text, color) in enumerate(lines):
        draw_unicode_text(img, text, (x, y + i * line_h), color, 22, 1)


def clamp_score(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 100.0))


def ellipsis_text(text: str, max_chars: int) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)] + "…"


def draw_panel_box(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str | None = None,
    fill: tuple[int, int, int] = (28, 31, 36),
    border: tuple[int, int, int] = (74, 82, 92),
) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), border, 1)
    if title:
        draw_unicode_text(img, title, (x + 14, y + 28), (255, 255, 255), 20, 1)


def draw_status_pill(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    w: int = 118,
    h: int = 28,
) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), (42, 46, 52), -1)
    cv2.rectangle(img, (x, y), (x + 5, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 96, 105), 1)
    draw_unicode_text(img, text, (x + 12, y + 20), color, 16, 1)


def draw_help_line(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int] = (220, 225, 230)) -> None:
    draw_unicode_text(img, text, (x, y), color, 17, 1)
