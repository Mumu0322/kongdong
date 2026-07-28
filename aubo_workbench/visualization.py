#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 RGB 检测叠加图、深度图、质量看板拼成一整块可显示画面。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import AUTO_CAPTURE_CFG, BOARD_CFG, E7_HAND_EYE_CFG, ROBOT_CFG, SOLVE_CFG
from .drawing import (
    clamp_score,
    draw_help_line,
    draw_metric_bar,
    draw_panel_box,
    draw_status_pill,
    draw_unicode_text,
    ellipsis_text,
)
from .quality import (
    ImageQualityResult,
    score_at_least,
    score_at_most,
    score_between,
)


def quality_status_color(quality: ImageQualityResult) -> tuple[int, int, int]:
    if quality.ok:
        return (0, 210, 105)
    if quality.score >= 70.0:
        return (0, 185, 255)
    return (0, 80, 255)


def build_ui_lines(quality: ImageQualityResult, sample_count: int) -> dict[str, list[str]]:
    target = AUTO_CAPTURE_CFG
    status = [
        f"AUBO：{ROBOT_CFG.ip}:{ROBOT_CFG.rpc_port}",
        f"位姿源：{ROBOT_CFG.pose_source}  只读",
        f"样本：{sample_count}（诊断≥{SOLVE_CFG.min_samples_for_solve}，E7≥{E7_HAND_EYE_CFG.minimum_total_poses}）",
        f"诊断：{'可按 h' if sample_count >= SOLVE_CFG.min_samples_for_solve else f'还需 {SOLVE_CFG.min_samples_for_solve - sample_count} 组'}",
    ]
    if quality.calibration_frame == "rgb_camera":
        metrics = [
            f"ChArUco角点：{quality.charuco_count} / {target.min_charuco_corners}",
            f"RGB PnP内点：{quality.rgb_pnp_inlier_count} / {target.min_rgb_pnp_inliers}",
            f"重投影RMSE：{quality.rgb_reprojection_rmse_px:.3f} px",
            f"最大重投影：{quality.rgb_reprojection_max_px:.3f} px",
            f"亮度/对比/清晰：{quality.brightness:.0f} / {quality.contrast:.0f} / {quality.sharpness:.0f}",
            f"画面占比/中心偏移：{quality.board_area_ratio:.3f} / {quality.center_offset_ratio:.3f}",
        ]
    else:
        metrics = [
            f"ChArUco角点：{quality.charuco_count} / {target.min_charuco_corners}",
            f"有效3D角点：{quality.valid_3d_count} / {target.min_valid_3d_corners}",
            f"角点RMSE：{quality.corner_rmse_mm:.3f} mm",
            f"平面RMSE：{quality.plane_rmse_mm:.3f} mm",
            f"亮度/对比/清晰：{quality.brightness:.0f} / {quality.contrast:.0f} / {quality.sharpness:.0f}",
            f"画面占比/中心偏移：{quality.board_area_ratio:.3f} / {quality.center_offset_ratio:.3f}",
        ]
    reasons = quality.reasons[:3] if quality.reasons else ["当前画面满足严格采集门槛。"]
    advice = quality.advice[:3]
    return {"status": status, "metrics": metrics, "reasons": reasons, "advice": advice}


def draw_quality_dashboard(
    img: np.ndarray, quality: ImageQualityResult, burst_saved_count: int,
    auto_enabled: bool, cooldown_left_s: float, sample_count: int,
) -> None:
    """轻量叠加版看板，用于保存到磁盘的样本预览图。"""
    del auto_enabled, cooldown_left_s
    h, w = img.shape[:2]
    panel_w = min(760, w - 24)
    panel_h = 132
    x0 = 12
    y0 = max(12, h - panel_h - 14)

    panel = img.copy()
    cv2.rectangle(panel, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 24, 28), -1)
    cv2.addWeighted(panel, 0.68, img, 0.32, 0, dst=img)
    cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (90, 96, 105), 1)

    status_color = quality_status_color(quality)
    draw_unicode_text(img, f"质量 {quality.score:5.1f}/100 | {quality.label} | 样本 {sample_count}", (x0 + 14, y0 + 30), status_color, 22, 1)
    draw_unicode_text(
        img, f"采集进度 {burst_saved_count}/{AUTO_CAPTURE_CFG.manual_burst_frames} | c 采集 | h 诊断 | v E7验证 | q 退出",
        (x0 + 14, y0 + 58), (255, 255, 255), 17, 1,
    )

    bx, by, bar_w, bar_h = x0 + 16, y0 + 90, 112, 13
    draw_metric_bar(img, bx, by, bar_w, bar_h, "角点数", f"{quality.charuco_count}",
                     score_at_least(quality.charuco_count, AUTO_CAPTURE_CFG.min_charuco_corners, BOARD_CFG.min_charuco_corners))
    if quality.calibration_frame == "rgb_camera":
        draw_metric_bar(img, bx + 240, by, bar_w, bar_h, "PnP内点", f"{quality.rgb_pnp_inlier_count}",
                         score_at_least(quality.rgb_pnp_inlier_count, AUTO_CAPTURE_CFG.min_rgb_pnp_inliers, BOARD_CFG.min_charuco_corners))
        draw_metric_bar(img, bx + 480, by, bar_w, bar_h, "重投影", f"{quality.rgb_reprojection_rmse_px:.2f}px",
                         score_at_most(quality.rgb_reprojection_rmse_px, AUTO_CAPTURE_CFG.max_rgb_reprojection_rmse_px, AUTO_CAPTURE_CFG.max_rgb_reprojection_rmse_px * 2.5))
    else:
        draw_metric_bar(img, bx + 240, by, bar_w, bar_h, "有效3D", f"{quality.valid_3d_count}",
                         score_at_least(quality.valid_3d_count, AUTO_CAPTURE_CFG.min_valid_3d_corners, BOARD_CFG.min_valid_3d_corners))
        draw_metric_bar(img, bx + 480, by, bar_w, bar_h, "平面", f"{quality.plane_rmse_mm:.2f}mm",
                         score_at_most(quality.plane_rmse_mm, AUTO_CAPTURE_CFG.max_plane_rmse_mm, BOARD_CFG.max_plane_rmse_mm))


def draw_side_panel(
    canvas: np.ndarray, x: int, y: int, w: int, h: int,
    quality: ImageQualityResult, sample_count: int, burst_count: int,
) -> None:
    draw_panel_box(canvas, x, y, w, h, "采集状态")
    accent = quality_status_color(quality)
    draw_status_pill(canvas, "合格" if quality.ok else "待调整", x + w - 130, y + 12, accent, 112, 28)
    cv2.rectangle(canvas, (x + 16, y + 46), (x + w - 16, y + 50), (55, 60, 68), -1)
    cv2.rectangle(canvas, (x + 16, y + 46), (x + 16 + int((w - 32) * clamp_score(quality.score) / 100.0), y + 50), accent, -1)
    draw_unicode_text(canvas, f"质量评分 {quality.score:.1f}/100", (x + 16, y + 76), accent, 22, 1)

    lines = build_ui_lines(quality, sample_count)
    cy = y + 112
    for line in lines["status"]:
        draw_unicode_text(canvas, ellipsis_text(line, 28), (x + 18, cy), (235, 238, 242), 17, 1)
        cy += 25

    metric_y = y + 190
    draw_panel_box(canvas, x + 12, metric_y, w - 24, 136, "质量指标", fill=(34, 38, 44), border=(70, 78, 88))
    my = metric_y + 42
    if quality.calibration_frame == "rgb_camera":
        metric_scores = [
            score_at_least(quality.charuco_count, AUTO_CAPTURE_CFG.min_charuco_corners, BOARD_CFG.min_charuco_corners),
            score_at_least(quality.rgb_pnp_inlier_count, AUTO_CAPTURE_CFG.min_rgb_pnp_inliers, BOARD_CFG.min_charuco_corners),
            score_at_most(quality.rgb_reprojection_rmse_px, AUTO_CAPTURE_CFG.max_rgb_reprojection_rmse_px, AUTO_CAPTURE_CFG.max_rgb_reprojection_rmse_px * 2.5),
            score_at_most(quality.rgb_reprojection_max_px, AUTO_CAPTURE_CFG.max_rgb_reprojection_error_px, AUTO_CAPTURE_CFG.max_rgb_reprojection_error_px * 2.0),
            score_between(quality.brightness, AUTO_CAPTURE_CFG.min_brightness, AUTO_CAPTURE_CFG.max_brightness, 45.0),
            score_at_most(quality.center_offset_ratio, AUTO_CAPTURE_CFG.max_rgb_center_offset_ratio, 1.0),
        ]
    else:
        metric_scores = [
            score_at_least(quality.charuco_count, AUTO_CAPTURE_CFG.min_charuco_corners, BOARD_CFG.min_charuco_corners),
            score_at_least(quality.valid_3d_count, AUTO_CAPTURE_CFG.min_valid_3d_corners, BOARD_CFG.min_valid_3d_corners),
            score_at_most(quality.corner_rmse_mm, AUTO_CAPTURE_CFG.max_corner_rmse_mm, BOARD_CFG.max_corner_3d_rmse_mm),
            score_at_most(quality.plane_rmse_mm, AUTO_CAPTURE_CFG.max_plane_rmse_mm, BOARD_CFG.max_plane_rmse_mm),
            score_between(quality.brightness, AUTO_CAPTURE_CFG.min_brightness, AUTO_CAPTURE_CFG.max_brightness, 45.0),
            score_at_most(quality.center_offset_ratio, AUTO_CAPTURE_CFG.max_center_offset_ratio, AUTO_CAPTURE_CFG.max_center_offset_ratio * 1.8),
        ]
    for line, score in zip(lines["metrics"], metric_scores):
        color = (0, 210, 105) if score >= 85 else (0, 185, 255) if score >= 65 else (0, 80, 255)
        cv2.circle(canvas, (x + 28, my - 6), 4, color, -1)
        draw_unicode_text(canvas, ellipsis_text(line, 32), (x + 40, my), (235, 238, 242), 16, 1)
        my += 19

    issue_y = metric_y + 142
    draw_panel_box(canvas, x + 12, issue_y, w - 24, 78, "原因 / 建议", fill=(34, 38, 44), border=(70, 78, 88))
    iy = issue_y + 38
    for line in lines["reasons"][:1]:
        draw_unicode_text(canvas, ellipsis_text(line, 34), (x + 24, iy), (0, 220, 255) if not quality.ok else (0, 210, 105), 16, 1)
        iy += 20
    for line in lines["advice"][:1]:
        draw_unicode_text(canvas, ellipsis_text("建议：" + line, 34), (x + 24, iy), (235, 238, 242), 16, 1)
        iy += 20

    help_y = y + h - 118
    draw_panel_box(canvas, x + 12, help_y, w - 24, 102, "快捷键", fill=(34, 38, 44), border=(70, 78, 88))
    draw_help_line(canvas, f"c  采集{AUTO_CAPTURE_CFG.manual_burst_frames}帧选1帧  当前 {burst_count}/{AUTO_CAPTURE_CFG.manual_burst_frames}", x + 24, help_y + 40)
    draw_help_line(canvas, "h 诊断  v E7验证  a 剔除诊断  d 归档最后样本", x + 24, help_y + 63)
    draw_help_line(canvas, "m 手动TCP  q/ESC 退出并断开 AUBO", x + 24, help_y + 86, (180, 205, 255))


def compose_display(
    overlay_bgr: np.ndarray, depth_mm: np.ndarray, sample_count: int,
    quality: ImageQualityResult | None = None, good_frame_count: int = 0,
    auto_enabled: bool = True, cooldown_left_s: float = 0.0,
) -> np.ndarray:
    del auto_enabled, cooldown_left_s, depth_mm
    canvas_w, canvas_h = 1600, 900
    top_h = 56
    margin = 12
    left_w = 1044
    right_w = canvas_w - left_w - margin * 3
    content_y = top_h + margin
    content_h = canvas_h - content_y - margin
    depth_h = 268
    panel_h = content_h - depth_h - margin

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:, :] = (18, 21, 25)

    cv2.rectangle(canvas, (0, 0), (canvas_w, top_h), (31, 35, 41), -1)
    cv2.rectangle(canvas, (0, top_h - 1), (canvas_w, top_h), (72, 80, 90), -1)
    title_color = quality_status_color(quality) if quality is not None else (0, 185, 255)
    draw_unicode_text(canvas, "AUBO 眼在手 RGB-PnP 手眼标定", (18, 36), title_color, 25, 1)
    draw_unicode_text(
        canvas,
        f"机器人 {ROBOT_CFG.ip}:{ROBOT_CFG.rpc_port} | 位姿源 {ROBOT_CFG.pose_source} | 输出 {Path(SOLVE_CFG.output_json).name}",
        (390, 35), (232, 236, 241), 18, 1,
    )

    left_x, left_y = margin, content_y
    draw_panel_box(canvas, left_x, left_y, left_w, content_h, "RGB + ChArUco + PnP")
    left_img = cv2.resize(overlay_bgr, (left_w - 24, content_h - 52))
    canvas[left_y + 40:left_y + 40 + left_img.shape[0], left_x + 12:left_x + 12 + left_img.shape[1]] = left_img

    right_x = left_x + left_w + margin
    depth_y = content_y
    draw_panel_box(canvas, right_x, depth_y, right_w, depth_h, "正式手眼不启用Depth/PointCloud")
    cv2.rectangle(
        canvas,
        (right_x + 12, depth_y + 40),
        (right_x + right_w - 12, depth_y + depth_h - 12),
        (27, 31, 36),
        -1,
    )
    draw_unicode_text(canvas, "RGB角点 + 内参 + solvePnP", (right_x + 36, depth_y + 116), (0, 210, 255), 22, 1)
    draw_unicode_text(canvas, "深度失败不会影响手眼采集", (right_x + 36, depth_y + 154), (235, 238, 242), 18, 1)

    panel_y = depth_y + depth_h + margin
    if quality is None:
        draw_panel_box(canvas, right_x, panel_y, right_w, panel_h, "采集状态")
        draw_unicode_text(canvas, "正在采集批量帧，请保持机械臂和标定板静止。", (right_x + 18, panel_y + 64), (0, 220, 255), 18, 1)
        draw_unicode_text(canvas, f"本次已取得 {good_frame_count}/{AUTO_CAPTURE_CFG.manual_burst_frames} 帧合格观测。", (right_x + 18, panel_y + 96), (255, 255, 255), 18, 1)
        draw_help_line(canvas, "q/ESC 可退出；采集完成后会自动选择 1 帧保存。", right_x + 18, panel_y + 134, (180, 205, 255))
    else:
        draw_side_panel(canvas, right_x, panel_y, right_w, panel_h, quality, sample_count, good_frame_count)

    return canvas
