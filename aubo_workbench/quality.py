#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对当前画面/ChArUco 检测结果打分，判断是否适合作为标定采样帧。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .charuco_detect import BoardPoseResult
from .config import AUTO_CAPTURE_CFG, BOARD_CFG


@dataclass
class ImageQualityResult:
    ok: bool
    score: float
    label: str
    reasons: list[str]
    advice: list[str]
    brightness: float
    contrast: float
    sharpness: float
    board_area_ratio: float
    center_offset_ratio: float
    charuco_count: int
    valid_3d_count: int
    corner_rmse_mm: float
    corner_max_error_mm: float
    plane_rmse_mm: float
    calibration_frame: str
    rgb_pnp_inlier_count: int
    rgb_reprojection_rmse_px: float
    rgb_reprojection_max_px: float


def clamp_score(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 100.0))


def score_at_least(value: float, good: float, bad: float = 0.0) -> float:
    """value >= good 得满分；value <= bad 得 0 分。"""
    if not np.isfinite(value):
        return 0.0
    if good <= bad:
        return 100.0 if value >= good else 0.0
    return clamp_score((value - bad) / (good - bad) * 100.0)


def score_at_most(value: float, good: float, bad: float) -> float:
    """value <= good 得满分；value >= bad 得 0 分。"""
    if not np.isfinite(value):
        return 0.0
    if bad <= good:
        return 100.0 if value <= good else 0.0
    return clamp_score((bad - value) / (bad - good) * 100.0)


def score_between(value: float, low: float, high: float, soft_margin: float) -> float:
    """value 在 [low, high] 内得满分，越界后按 soft_margin 线性扣分。"""
    if not np.isfinite(value):
        return 0.0
    if low <= value <= high:
        return 100.0
    if value < low:
        return clamp_score((value - (low - soft_margin)) / soft_margin * 100.0)
    return clamp_score(((high + soft_margin) - value) / soft_margin * 100.0)


def get_quality_roi(color_bgr: np.ndarray, pose_result: BoardPoseResult) -> tuple[np.ndarray, float, float]:
    """返回用于评价亮度/清晰度的 ROI，以及标定板面积占比、中心偏移比例。"""
    h, w = color_bgr.shape[:2]
    image_area = float(max(1, h * w))
    points = pose_result.image_points or []
    if len(points) >= 4:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] >= 4:
            x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
            pad = int(max(12, 0.08 * max(bw, bh)))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w, x + bw + pad)
            y1 = min(h, y + bh + pad)
            if x1 > x0 and y1 > y0:
                roi = color_bgr[y0:y1, x0:x1]
                area_ratio = float((x1 - x0) * (y1 - y0) / image_area)
                cx = x0 + (x1 - x0) * 0.5
                cy = y0 + (y1 - y0) * 0.5
                center_offset_ratio = float(
                    math.sqrt(((cx - w * 0.5) / max(1.0, w * 0.5)) ** 2 + ((cy - h * 0.5) / max(1.0, h * 0.5)) ** 2)
                )
                return roi, area_ratio, center_offset_ratio
    return color_bgr, 1.0, 0.0


def evaluate_image_quality(color_bgr: np.ndarray, pose_result: BoardPoseResult) -> ImageQualityResult:
    roi, board_area_ratio, center_offset_ratio = get_quality_roi(color_bgr, pose_result)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray)) if gray.size else 0.0
    contrast = float(np.std(gray)) if gray.size else 0.0
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray.size else 0.0

    cfg = AUTO_CAPTURE_CFG
    is_rgb = pose_result.calibration_frame == "rgb_camera"
    reasons: list[str] = []
    advice: list[str] = []

    def add_advice(text: str) -> None:
        if text not in advice:
            advice.append(text)

    pose_transform_available = (
        pose_result.T_rgb_board is not None if is_rgb else pose_result.T_pointcloud_board is not None
    )
    if not pose_result.ok or not pose_transform_available:
        reasons.append(f"标定板位姿无效：{pose_result.status}")
        add_advice("请让整张 ChArUco 板完整出现在画面内，避免边缘被裁切和过大倾斜。")
    if pose_result.charuco_count < cfg.min_charuco_corners:
        reasons.append(f"角点数量不足：{pose_result.charuco_count}<{cfg.min_charuco_corners}")
        add_advice("相机适当靠近，或减小标定板倾斜角，让更多 ChArUco 角点被检测到。")
    if is_rgb:
        if pose_result.rgb_pnp_inlier_count < cfg.min_rgb_pnp_inliers:
            reasons.append(f"RGB PnP内点不足：{pose_result.rgb_pnp_inlier_count}<{cfg.min_rgb_pnp_inliers}")
            add_advice("保证标定板完整可见并提高清晰度，让绝大多数ChArUco角点进入PnP。")
        if (
            not np.isfinite(pose_result.rgb_reprojection_rmse_px)
            or pose_result.rgb_reprojection_rmse_px > cfg.max_rgb_reprojection_rmse_px
        ):
            reasons.append(
                f"RGB重投影RMSE偏大：{pose_result.rgb_reprojection_rmse_px:.3f}>"
                f"{cfg.max_rgb_reprojection_rmse_px:.3f}px"
            )
            add_advice("保持机器人静止，改善对焦和光照，并复核RGB内参与当前分辨率。")
        if (
            not np.isfinite(pose_result.rgb_reprojection_max_px)
            or pose_result.rgb_reprojection_max_px > cfg.max_rgb_reprojection_error_px
        ):
            reasons.append(
                f"RGB最大重投影误差偏大：{pose_result.rgb_reprojection_max_px:.3f}>"
                f"{cfg.max_rgb_reprojection_error_px:.3f}px"
            )
            add_advice("检查局部角点误检、板面翘曲或错误的相机畸变参数。")
    else:
        if pose_result.valid_3d_count < cfg.min_valid_3d_corners:
            reasons.append(f"有效3D角点不足：{pose_result.valid_3d_count}<{cfg.min_valid_3d_corners}")
            add_advice("把标定板移到画面中心和有效深度范围内，避开反光区域和黑边区域。")
        if np.isfinite(pose_result.corner_rmse_mm) and pose_result.corner_rmse_mm > cfg.max_corner_rmse_mm:
            reasons.append(f"角点RMSE偏大：{pose_result.corner_rmse_mm:.2f}>{cfg.max_corner_rmse_mm:.2f}mm")
            add_advice("保持机器人静止，改善对焦和光照，并保证标定板平整不弯曲。")
        if np.isfinite(pose_result.corner_max_error_mm) and pose_result.corner_max_error_mm > cfg.max_corner_max_error_mm:
            reasons.append(f"角点最大误差偏大：{pose_result.corner_max_error_mm:.2f}>{cfg.max_corner_max_error_mm:.2f}mm")
            add_advice("检查局部深度空洞，避免反光，不要使用破损或翘边的标定板角点。")
        if np.isfinite(pose_result.plane_rmse_mm) and pose_result.plane_rmse_mm > cfg.max_plane_rmse_mm:
            reasons.append(f"平面RMSE偏大：{pose_result.plane_rmse_mm:.2f}>{cfg.max_plane_rmse_mm:.2f}mm")
            add_advice("减小标定板倾斜和反光，确认打印板平整、没有翘曲。")
    if not (cfg.min_board_area_ratio <= board_area_ratio <= cfg.max_board_area_ratio):
        reasons.append(f"标定板画面占比不合适：{board_area_ratio:.2f}")
        if board_area_ratio < cfg.min_board_area_ratio:
            add_advice("相机适当靠近；标定板太小，不利于稳定 1 mm 标定。")
        else:
            add_advice("相机适当远离；必须保证整张标定板完整可见。")
    center_limit = cfg.max_rgb_center_offset_ratio if is_rgb else cfg.max_center_offset_ratio
    if center_offset_ratio > center_limit:
        reasons.append(f"标定板偏离画面过多：{center_offset_ratio:.2f}>{center_limit:.2f}")
        add_advice("保证标定板完整可见；E7需要中心和边缘视野，但不能发生裁切。")
    if not (cfg.min_brightness <= brightness <= cfg.max_brightness):
        reasons.append(f"亮度不合适：{brightness:.0f}")
        if brightness < cfg.min_brightness:
            add_advice("增加均匀补光，避免标定板上出现阴影。")
        else:
            add_advice("降低曝光或光照，避免白色格子过曝。")
    if contrast < cfg.min_contrast:
        reasons.append(f"对比度不足：{contrast:.0f}<{cfg.min_contrast:.0f}")
        add_advice("提高黑白对比度，避免模糊、反光和板面污渍。")
    if sharpness < cfg.min_sharpness:
        reasons.append(f"清晰度不足：{sharpness:.0f}<{cfg.min_sharpness:.0f}")
        add_advice("采集前停止机器人运动，重新对焦，或缩短曝光时间。")

    if not advice:
        advice.append("已合格：可以采集，但要换不同姿态，不要连续采很多几乎相同的帧。")

    if is_rgb:
        scores = [
            score_at_least(pose_result.charuco_count, cfg.min_charuco_corners, BOARD_CFG.min_charuco_corners),
            score_at_least(pose_result.rgb_pnp_inlier_count, cfg.min_rgb_pnp_inliers, BOARD_CFG.min_charuco_corners),
            score_at_most(
                pose_result.rgb_reprojection_rmse_px,
                cfg.max_rgb_reprojection_rmse_px,
                cfg.max_rgb_reprojection_rmse_px * 2.5,
            ),
            score_at_most(
                pose_result.rgb_reprojection_max_px,
                cfg.max_rgb_reprojection_error_px,
                cfg.max_rgb_reprojection_error_px * 2.0,
            ),
            score_between(board_area_ratio, cfg.min_board_area_ratio, cfg.max_board_area_ratio, 0.08),
            score_at_most(center_offset_ratio, center_limit, 1.0),
            score_between(brightness, cfg.min_brightness, cfg.max_brightness, 45.0),
            score_at_least(contrast, cfg.min_contrast, 5.0),
            score_at_least(sharpness, cfg.min_sharpness, 5.0),
        ]
    else:
        scores = [
            score_at_least(pose_result.charuco_count, cfg.min_charuco_corners, BOARD_CFG.min_charuco_corners),
            score_at_least(pose_result.valid_3d_count, cfg.min_valid_3d_corners, BOARD_CFG.min_valid_3d_corners),
            score_at_most(pose_result.corner_rmse_mm, cfg.max_corner_rmse_mm, BOARD_CFG.max_corner_3d_rmse_mm),
            score_at_most(pose_result.corner_max_error_mm, cfg.max_corner_max_error_mm, cfg.max_corner_max_error_mm * 2.0),
            score_at_most(pose_result.plane_rmse_mm, cfg.max_plane_rmse_mm, BOARD_CFG.max_plane_rmse_mm),
            score_between(board_area_ratio, cfg.min_board_area_ratio, cfg.max_board_area_ratio, 0.08),
            score_at_most(center_offset_ratio, center_limit, center_limit * 1.8),
            score_between(brightness, cfg.min_brightness, cfg.max_brightness, 45.0),
            score_at_least(contrast, cfg.min_contrast, 5.0),
            score_at_least(sharpness, cfg.min_sharpness, 5.0),
        ]
    score = float(np.mean(scores)) if scores else 0.0
    ok = score >= cfg.min_score and not reasons
    if ok:
        label = "合格 - 可按 c 抓取5帧并选1帧保存"
    elif score >= 70.0:
        label = "接近合格 - 轻微调整"
    else:
        label = "不合格 - 暂不采集"
    return ImageQualityResult(
        ok=ok,
        score=score,
        label=label,
        reasons=reasons[:4],
        advice=advice[:4],
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        board_area_ratio=board_area_ratio,
        center_offset_ratio=center_offset_ratio,
        charuco_count=pose_result.charuco_count,
        valid_3d_count=pose_result.valid_3d_count,
        corner_rmse_mm=pose_result.corner_rmse_mm,
        corner_max_error_mm=pose_result.corner_max_error_mm,
        plane_rmse_mm=pose_result.plane_rmse_mm,
        calibration_frame=pose_result.calibration_frame,
        rgb_pnp_inlier_count=pose_result.rgb_pnp_inlier_count,
        rgb_reprojection_rmse_px=pose_result.rgb_reprojection_rmse_px,
        rgb_reprojection_max_px=pose_result.rgb_reprojection_max_px,
    )
