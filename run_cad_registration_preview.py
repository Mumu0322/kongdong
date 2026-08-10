#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAD 配准 Stage 0 预览入口。

本脚本支持两种模式：传入 --image 做离线回放；不传 --image 时连接现有
AUBO SDK 的只读 AuboPoseSession，启动 Gemini Color 流，用户按键采集 RGB。
脚本不导入 AuboMotionSession，不发送运动指令，不读取 Depth 或 PointCloud。
通过质量门后也只生成报告，motion_allowed 固定为 False。

常用离线调用（5 张图）：
    python run_cad_registration_preview.py `
        --step "C:\\MM\\aubo_tools\\aubo_workbench_project\\孔位板_JXDZ26-KWB-001.STEP" `
        --image frame1.png --image frame2.png --image frame3.png --image frame4.png --image frame5.png `
        --intrinsics "C:\\MM\\aubo_tools\\data\\camera_calibration\\current_rgb_intrinsics.json" `
        --prior-camera-cad approximate_T_camera_cad.json `
        --base-camera approximate_T_base_camera.json

现场采集模式不传 --image；启动后按 c/空格拍摄，按 s/回车/c 保存复核帧，按 r 重拍，
按 q 退出。默认采集 5 帧。若没有先验或映射，上一轮有有效 T_base_cad 时可按 P 复用，
也可按 A 直接使用孔间几何自动匹配，或按 M 进入鼠标映射：先点击左侧 YOLO 孔，
再点击右侧 CAD 孔，至少配 4 对后按 S/回车完成，R 重置，Z 撤销。

如果自动匹配不唯一，准备一个 JSON，例如：
    {"0": "CAD-01", "1": "CAD-05", "2": "CAD-06", "3": "CAD-11"}
然后加上 --mapping-json。此映射必须由图像编号与 CAD 孔号表人工确认。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aubo_workbench.cad_model import load_cad_model_json, parse_step_model  # noqa: E402
from aubo_workbench.config import apply_robot_connection_overrides  # noqa: E402
from aubo_workbench.cad_registration import (  # noqa: E402
    CadFrameResult,
    CadRegistrationConfig,
    detect_yolo_holes,
    draw_cad_overlay,
    draw_detection_boxes,
    load_detections_json,
    load_intrinsics_json,
    load_transform_json,
    register_cad_frame,
    register_detections_frames,
    save_registration_report,
    undistort_pixels,
)
from aubo_workbench.geometry import invert_transform  # noqa: E402
from aubo_workbench.paths import (  # noqa: E402
    CAD_MODEL_PATH,
    CAD_REGISTRATION_RUNS_DIR,
    CAMERA_CALIBRATION_PATH,
    HANDEYE_CANDIDATE_PATH,
    MODEL_PATH,
    STEP_MODEL_PATH,
)


DEFAULT_INTRINSICS = CAMERA_CALIBRATION_PATH
DEFAULT_STEP = STEP_MODEL_PATH
DEFAULT_CAD_MODEL = CAD_MODEL_PATH
DEFAULT_YOLO = MODEL_PATH
DEFAULT_HANDEYE = HANDEYE_CANDIDATE_PATH
DEFAULT_RUN_ROOT = CAD_REGISTRATION_RUNS_DIR
CAD_PANEL_WIDTH = 330


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 0：CAD 顶面孔与 RGB/YOLO 检测预览；现场模式只读连接 AUBO，不运动。"
    )
    parser.add_argument("--image", action="append", default=[], help="离线 RGB 图片；可重复 1 或 5 次")
    parser.add_argument("--image-dir", help="读取目录内按文件名排序的 png/jpg/jpeg/bmp 图片")
    parser.add_argument("--frames", type=int, default=5, help="无 --image 时采集的 RGB 帧数，默认 5")
    parser.add_argument("--intrinsics", default=str(DEFAULT_INTRINSICS), help="RGB 内参 JSON")
    parser.add_argument("--step", default=None, help="STEP 文件；指定后用 OCP 解析顶面大孔")
    parser.add_argument("--cad-model", default=str(DEFAULT_CAD_MODEL), help="已解析的 cad_hole_model.json")
    parser.add_argument("--save-cad-model", default=None, help="解析 STEP 后保存的 CAD JSON 路径")
    parser.add_argument("--model", default=str(DEFAULT_YOLO), help="现有 YOLO 权重路径")
    parser.add_argument("--handeye", default=str(DEFAULT_HANDEYE), help="TCP→RGB 相机手眼 JSON；现场模式只读使用")
    parser.add_argument("--confidence", type=float, default=0.35, help="YOLO 置信度门限")
    parser.add_argument("--detections-json", action="append", default=[], help="离线检测 JSON；可重复或单个复用到所有帧")
    parser.add_argument("--mapping-json", help="人工确认的 detection_id→CAD-xx 映射 JSON；提供后跳过自动匹配")
    parser.add_argument("--prior-camera-cad", help="近似 CAD→RGB 相机 4x4 变换 JSON")
    parser.add_argument("--prior-base-cad", help="近似 CAD→Base 4x4 变换 JSON；需同时提供 --base-camera")
    parser.add_argument("--base-camera", help="RGB 相机→Base 的 4x4 变换 JSON；缺失时不输出基坐标孔位")
    parser.add_argument("--output-dir", help="结果目录；默认 data/cad_registration_runs/<timestamp>")
    parser.add_argument("--match-distance-px", type=float, default=80.0)
    parser.add_argument("--rmse-px", type=float, default=2.0)
    parser.add_argument("--max-error-px", type=float, default=4.0)
    parser.add_argument("--cross-frame-p95-mm", type=float, default=1.5)
    parser.add_argument("--min-valid-frames", type=int, default=3)
    parser.add_argument("--auto-triangle-tolerance", type=float, default=0.18,
                        help="自动几何匹配三角形边长签名容差")
    parser.add_argument("--auto-initial-distance-px", type=float, default=100.0,
                        help="自动几何候选初始一对一匹配距离门限(px)")
    parser.add_argument("--auto-inlier-distance-px", type=float, default=14.0,
                        help="自动几何候选单应性精化内点距离门限(px)")
    parser.add_argument("--auto-max-hypotheses", type=int, default=600,
                        help="自动几何匹配最多保留的三点假设数")
    parser.add_argument("--auto-ambiguity-margin-px", type=float, default=1.0,
                        help="自动匹配最佳/次佳分数差小于该值时转人工确认(px)")
    parser.add_argument("--auto-diameter-weight", type=float, default=75.0,
                        help="自动匹配中孔径一致性评分权重")
    parser.add_argument("--auto-max-pnp-candidates", type=int, default=128,
                        help="自动匹配最多进行 PnP 物理质量检查的候选数")
    # 工作台 GUI 传入的只读 AUBO 连接参数；不改变本脚本只读、不运动的性质。
    parser.add_argument("--robot-ip", type=str, help="AUBO RPC IP（工作台传入）")
    parser.add_argument("--robot-port", type=int, help="AUBO RPC 端口（工作台传入）")
    parser.add_argument("--robot-user", type=str, help="AUBO 用户名（工作台传入）")
    parser.add_argument("--robot-password", type=str, help="AUBO 密码（工作台传入）")
    parser.add_argument("--robot-timeout-ms", type=int, help="AUBO 请求超时毫秒（工作台传入）")
    return parser


# 实现已统一到 aubo_workbench.config.apply_robot_connection_overrides。
_apply_robot_connection_overrides = apply_robot_connection_overrides


def _collect_image_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(item) for item in args.image]
    if args.image_dir:
        directory = Path(args.image_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"图片目录不存在: {directory}")
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        paths.extend(sorted(path for path in directory.iterdir() if path.suffix.lower() in extensions))
    if len(paths) > 5:
        raise ValueError(f"最多接收 5 帧 RGB，当前 {len(paths)} 帧")
    return paths


def _load_yolo_model(args: argparse.Namespace):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "YOLO 现场预览需要 ultralytics；请使用 lip_env310，或提供 --detections-json"
        ) from exc
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO 权重不存在: {model_path}")
    return model_path, YOLO(str(model_path))


def _load_live_handeye(args: argparse.Namespace):
    from aubo_workbench.charuco_point_experiment import load_handeye_experiment_result

    handeye_path = Path(args.handeye)
    handeye = load_handeye_experiment_result(handeye_path)
    if not handeye.validated_for_motion:
        print(f"[WARN] 当前手眼文件仅用于只读预览，不得用于运动: {handeye.warning}")
    print(f"[INFO] 现场 RGB 预览使用手眼文件: {handeye.path}")
    return handeye_path, handeye


def _read_robot_base_camera(pose_session: Any, handeye: Any) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    snapshot = pose_session.read_pose_snapshot()
    values = snapshot.get("pose_values_sdk_m_rad")
    if not isinstance(values, (list, tuple)) or len(values) < 6:
        raise RuntimeError("AUBO SDK 没有返回完整 TCP 位姿")
    T_base_tcp = pose_session.pose_sdk_to_transform_mm([float(value) for value in values[:6]])
    T_base_camera = T_base_tcp @ np.asarray(handeye.T_tcp_rgb_camera, dtype=np.float64)
    return snapshot, T_base_tcp, T_base_camera


def _append_cad_layout_panel(image_bgr: np.ndarray, model: Any) -> np.ndarray:
    """在 RGB 画面右侧追加 CAD 顶视孔位布局，供人工确认孔号和坐标方向。"""

    image = np.asarray(image_bgr)
    height = int(image.shape[0])
    panel_width = CAD_PANEL_WIDTH
    panel = np.full((height, panel_width, 3), (28, 28, 28), dtype=np.uint8)
    cv2.putText(panel, "CAD TOP VIEW", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "+X -> forward/up   +Y -> left", (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1, cv2.LINE_AA)

    centers = np.asarray([hole.center_cad_mm[:2] for hole in model.holes], dtype=np.float64)
    x_min, y_min = np.min(centers, axis=0)
    x_max, y_max = np.max(centers, axis=0)
    plot_left, plot_right = 28, panel_width - 28
    plot_top = 86
    plot_bottom = max(plot_top + 80, height - 54)
    scale = min(
        (plot_right - plot_left) / max(1.0, float(x_max - x_min) + 100.0),
        (plot_bottom - plot_top) / max(1.0, float(y_max - y_min) + 100.0),
    )
    plot_center = np.array([(plot_left + plot_right) * 0.5, (plot_top + plot_bottom) * 0.5], dtype=np.float64)
    cad_center = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=np.float64)

    cv2.rectangle(panel, (plot_left, plot_top), (plot_right, plot_bottom), (80, 80, 80), 1, cv2.LINE_AA)
    axis_origin = np.round(plot_center).astype(int)
    # 按现场定义：+X 朝前（俯视图上方），+Y 朝左（俯视图左方）。
    cv2.arrowedLine(panel, tuple(axis_origin), (plot_left + 8, axis_origin[1]), (120, 120, 120), 1, cv2.LINE_AA, tipLength=0.08)
    cv2.arrowedLine(panel, tuple(axis_origin), (axis_origin[0], plot_top + 8), (120, 120, 120), 1, cv2.LINE_AA, tipLength=0.08)
    cv2.putText(panel, "+Y", (plot_left + 8, axis_origin[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(panel, "+X", (axis_origin[0] + 8, plot_top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    for hole in model.holes:
        cad_xy = np.asarray(hole.center_cad_mm[:2], dtype=np.float64)
        # 图像坐标 y 向下，所以现场定义的 CAD +X（向前）映射到面板上方，
        # CAD +Y（向左）映射到面板左方。
        point_float = plot_center + np.array(
            [-(cad_xy[1] - cad_center[1]) * scale, -(cad_xy[0] - cad_center[0]) * scale],
            dtype=np.float64,
        )
        point = tuple(np.round(point_float).astype(int))
        radius = max(7, int(round(float(hole.radius_mm) * scale)))
        cv2.circle(panel, point, radius, (60, 205, 60), 2, cv2.LINE_AA)
        cv2.circle(panel, point, 3, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.putText(panel, hole.hole_id, (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 255, 100), 1, cv2.LINE_AA)

    cv2.putText(panel, "green circles = CAD holes", (18, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1, cv2.LINE_AA)
    return np.hstack((image, panel))


def _cad_layout_points(model: Any, image_height: int) -> dict[str, tuple[int, int]]:
    """返回 CAD 侧栏中各孔中心的像素位置，供鼠标点选使用。"""

    centers = np.asarray([hole.center_cad_mm[:2] for hole in model.holes], dtype=np.float64)
    x_min, y_min = np.min(centers, axis=0)
    x_max, y_max = np.max(centers, axis=0)
    plot_left, plot_right = 28, CAD_PANEL_WIDTH - 28
    plot_top = 86
    plot_bottom = max(plot_top + 80, int(image_height) - 54)
    scale = min(
        (plot_right - plot_left) / max(1.0, float(x_max - x_min) + 100.0),
        (plot_bottom - plot_top) / max(1.0, float(y_max - y_min) + 100.0),
    )
    plot_center = np.array([(plot_left + plot_right) * 0.5, (plot_top + plot_bottom) * 0.5], dtype=np.float64)
    cad_center = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=np.float64)
    points: dict[str, tuple[int, int]] = {}
    for hole in model.holes:
        cad_xy = np.asarray(hole.center_cad_mm[:2], dtype=np.float64)
        point = plot_center + np.array(
            [-(cad_xy[1] - cad_center[1]) * scale, -(cad_xy[0] - cad_center[0]) * scale],
            dtype=np.float64,
        )
        points[hole.hole_id] = tuple(np.round(point).astype(int))
    return points


def _draw_capture_detection_preview(
    image_bgr: np.ndarray,
    detections: list[Any],
    intrinsics: Any,
    title: str,
    expected_hole_count: int,
    cad_model: Any | None = None,
) -> np.ndarray:
    """绘制 YOLO 复核画面。

    这里故意只使用 ASCII 文本：OpenCV 的 Hershey 字体无法渲染中文，
    会把中文显示成 ``???``，容易让现场人员误以为程序异常。
    """

    height, width = image_bgr.shape[:2]
    edge_ids: list[str] = []
    for detection in detections:
        x1, y1, x2, y2 = [float(value) for value in detection.box_xyxy]
        if x1 <= 2.0 or y1 <= 2.0 or x2 >= width - 2.0 or y2 >= height - 2.0:
            edge_ids.append(f"det-{int(detection.detection_id)}")

    count = len(detections)
    extra_count = max(0, count - int(expected_hole_count))
    missing_count = max(0, int(expected_hole_count) - count)
    if count < 4:
        status = "BLOCK: fewer than 4 candidates"
        status_color = (0, 0, 255)
    elif edge_ids or extra_count:
        status = "REVIEW: check extra/edge candidates before SAVE"
        status_color = (0, 165, 255)
    else:
        status = "READY: press S/Enter/C to SAVE"
        status_color = (0, 180, 0)

    feedback_lines = [
        title,
        f"YOLO candidates={count} | CAD holes={int(expected_hole_count)} | extra={extra_count} | missing={missing_count}",
        f"edge/partial candidates={len(edge_ids)}" + (f" ({', '.join(edge_ids)})" if edge_ids else ""),
        status,
        "S/Enter/C=SAVE   R=RETAKE   Q=QUIT",
    ]
    canvas = cv2.undistort(image_bgr, intrinsics.camera_matrix, intrinsics.dist_coeffs)
    canvas = draw_detection_boxes(canvas, detections, CadFrameResult(frame_index=0), intrinsics)
    # 深色半透明背景让高亮反馈在白色工件和黑色背景上都清楚可读。
    panel_height = 30 + 26 * len(feedback_lines)
    panel_width = min(width - 10, max(760, int(width * 0.56)))
    panel = canvas[:panel_height, :panel_width].copy()
    panel[:] = (25, 25, 25)
    canvas[:panel_height, :panel_width] = cv2.addWeighted(
        panel, 0.72, canvas[:panel_height, :panel_width], 0.28, 0.0
    )
    for line_index, line in enumerate(feedback_lines):
        color = status_color if line == status else (255, 255, 255)
        origin = (20, 28 + 26 * line_index)
        cv2.putText(canvas, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)

    # 在画面边缘的候选框旁加黄色 EDGE 标记，提示它们通常是局部/误检。
    for detection in detections:
        x1, y1, x2, y2 = [float(value) for value in detection.box_xyxy]
        if not (x1 <= 2.0 or y1 <= 2.0 or x2 >= width - 2.0 or y2 >= height - 2.0):
            continue
        center = tuple(np.round(detection.center_px).astype(int))
        cv2.drawMarker(canvas, center, (0, 255, 255), cv2.MARKER_DIAMOND, 18, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "EDGE",
            (max(4, center[0] - 26), max(panel_height + 18, center[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "EDGE",
            (max(4, center[0] - 26), max(panel_height + 18, center[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if cad_model is not None:
        canvas = _append_cad_layout_panel(canvas, cad_model)
    return canvas


def _capture_live_robot(
    args: argparse.Namespace,
    intrinsics: Any,
    output_dir: Path,
    expected_hole_count: int,
    cad_model: Any,
) -> tuple[list[np.ndarray], list[str], list[list[Any]], np.ndarray, dict[str, Any]]:
    """现场采集：RGB Color + 只读 TCP，不发送机器人运动。"""

    if args.frames not in {1, 5}:
        raise ValueError("现场采集的 --frames 只能是 1 或 5")
    handeye_path, handeye = _load_live_handeye(args)
    try:
        from aubo_workbench.camera import get_rgb_frame_bundle, init_rgb_handeye_pipeline
        from aubo_workbench.robot import AuboPoseSession
    except Exception as exc:
        raise RuntimeError(f"无法加载现场 RGB/只读 AUBO SDK 接口: {type(exc).__name__}: {exc}") from exc

    model_path, yolo_model = _load_yolo_model(args)
    pose_session = AuboPoseSession()
    pipeline = None
    images: list[np.ndarray] = []
    labels: list[str] = []
    detections_by_frame: list[list[Any]] = []
    snapshots: list[dict[str, Any]] = []
    first_T_base_camera: np.ndarray | None = None
    window_name = "CAD Registration Preview - press C to capture"
    try:
        pose_session.connect()
        pipeline = init_rgb_handeye_pipeline()
        print("[INFO] 现场采集已启动：机械臂保持静止，按 C/空格拍摄，按 Q 退出。")
        while len(images) < int(args.frames):
            bundle = get_rgb_frame_bundle(pipeline)
            if bundle is None:
                continue
            live = cv2.undistort(bundle.color_bgr, intrinsics.camera_matrix, intrinsics.dist_coeffs)
            live = live.copy()
            message = (
                f"Saved {len(images)}/{args.frames} | CAD holes={int(expected_hole_count)} | "
                "C/Space=capture | Q=quit"
            )
            cv2.putText(live, message, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(live, message, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            live = _append_cad_layout_panel(live, cad_model)
            cv2.imshow(window_name, live)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), ord("Q"), 27}:
                raise RuntimeError("用户退出现场采集，未完成 5 帧采集")
            if key not in {ord("c"), ord("C"), ord(" ")}:  # C 或空格
                continue

            snapshot, _, T_base_camera = _read_robot_base_camera(pose_session, handeye)
            detections = detect_yolo_holes(
                bundle.color_bgr,
                model_path,
                intrinsics,
                confidence=args.confidence,
                yolo_model=yolo_model,
            )
            review = _draw_capture_detection_preview(
                bundle.color_bgr,
                detections,
                intrinsics,
                f"frame {len(images)}: YOLO detections={len(detections)}",
                expected_hole_count,
                cad_model,
            )
            cv2.imshow(window_name, review)
            edge_ids = [
                f"det-{int(detection.detection_id)}"
                for detection in detections
                if float(detection.box_xyxy[0]) <= 2.0
                or float(detection.box_xyxy[1]) <= 2.0
                or float(detection.box_xyxy[2]) >= bundle.color_bgr.shape[1] - 2.0
                or float(detection.box_xyxy[3]) >= bundle.color_bgr.shape[0] - 2.0
            ]
            extra_count = max(0, len(detections) - int(expected_hole_count))
            print(
                f"[CAPTURE] frame_{len(images):03d}: detections={len(detections)}, "
                f"CAD holes={int(expected_hole_count)}, extra={extra_count}, "
                f"edge/partial={len(edge_ids)} ({', '.join(edge_ids) if edge_ids else 'none'}); "
                "S/Enter/C=save, R=retake, Q=quit"
            )
            review_key = cv2.waitKey(0) & 0xFF
            if review_key in {ord("q"), ord("Q"), 27}:
                raise RuntimeError("用户退出现场采集，已保存的帧仍保留在结果目录")
            if review_key not in {ord("s"), ord("S"), ord("c"), ord("C"), 13, 10, ord(" ")}:  # S/Enter/C/空格
                print("[CAPTURE] 当前帧未保存，准备重拍")
                continue

            if first_T_base_camera is None:
                first_T_base_camera = T_base_camera.copy()
            else:
                delta_mm = float(np.linalg.norm(T_base_camera[:3, 3] - first_T_base_camera[:3, 3]))
                if delta_mm > 1.0:
                    raise RuntimeError(
                        f"拍摄期间 AUBO TCP 平移变化 {delta_mm:.3f}mm；请保持机器人静止后重新采集"
                    )
            index = len(images)
            images.append(bundle.color_bgr.copy())
            labels.append(f"gemini_rgb_frame_{index:03d}")
            detections_by_frame.append(detections)
            snapshots.append(snapshot)
            cv2.imwrite(str(output_dir / f"rgb_original_{index:03d}.png"), bundle.color_bgr)
            (output_dir / f"detections_{index:03d}.json").write_text(
                json.dumps([detection.to_dict() for detection in detections], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (output_dir / f"robot_pose_snapshot_{index:03d}.json").write_text(
                json.dumps(
                    {
                        "snapshot": snapshot,
                        "T_base_camera": T_base_camera.tolist(),
                        "T_tcp_rgb_camera": np.asarray(handeye.T_tcp_rgb_camera).tolist(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[CAPTURE] 已保存 frame_{index:03d}")
    finally:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        pose_session.disconnect()

    if first_T_base_camera is None or not images:
        raise RuntimeError("没有成功保存 RGB 帧")
    robot_meta = {
        "mode": "live_rgb_only_with_read_only_aubo_pose",
        "handeye_path": str(handeye_path),
        "handeye_validated_for_motion": bool(handeye.validated_for_motion),
        "handeye_warning": handeye.warning,
        "yolo_model": str(model_path),
        "robot_pose_snapshot_count": len(snapshots),
        "robot_pose_snapshots": snapshots,
        "T_base_camera": first_T_base_camera.tolist(),
        "T_tcp_rgb_camera": np.asarray(handeye.T_tcp_rgb_camera).tolist(),
    }
    (output_dir / "live_robot_capture.json").write_text(
        json.dumps(robot_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return images, labels, detections_by_frame, first_T_base_camera, robot_meta


def _load_model(args: argparse.Namespace):
    cad_json = Path(args.cad_model)
    if args.step:
        step_path = Path(args.step)
        model = parse_step_model(step_path, expected_hole_count=11)
        save_path = Path(args.save_cad_model) if args.save_cad_model else cad_json
        model.save_json(save_path)
        print(f"[INFO] STEP 顶面孔已解析并保存: {save_path}")
        return model, str(step_path), str(save_path)
    if not cad_json.is_file():
        raise FileNotFoundError(
            f"CAD JSON 不存在: {cad_json}；请使用 --step \"{DEFAULT_STEP}\" 先解析 STEP"
        )
    model = load_cad_model_json(cad_json)
    return model, str(cad_json), str(cad_json)


def _load_detection_frames(args: argparse.Namespace, images: list[Any], intrinsics) -> list[list[Any]]:
    if not args.detections_json:
        model_path, yolo_model = _load_yolo_model(args)
        return [
            detect_yolo_holes(
                image,
                model_path,
                intrinsics,
                confidence=args.confidence,
                yolo_model=yolo_model,
            )
            for image in images
        ]

    paths = [Path(item) for item in args.detections_json]
    if len(paths) not in {1, len(images)}:
        raise ValueError(
            f"--detections-json 应提供 1 个（复用）或与图片数相同的文件，当前 {len(paths)} 个/{len(images)} 帧"
        )
    if len(paths) == 1:
        paths = paths * len(images)
    return [load_detections_json(path, intrinsics) for path in paths]


def _load_mapping(args: argparse.Namespace) -> Any:
    if not args.mapping_json:
        return None
    path = Path(args.mapping_json)
    if not path.is_file():
        raise FileNotFoundError(f"人工映射 JSON 不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_and_show_first_cad_projection(
    image: np.ndarray,
    detections: list[Any],
    model: Any,
    intrinsics: Any,
    frame: CadFrameResult,
    output_dir: Path,
) -> None:
    """保存并在现场模式显示人工映射后的第一帧 CAD 投影。"""

    overlay = draw_cad_overlay(
        image,
        model,
        intrinsics,
        frame,
        title="CAD projection preview after mapping",
    )
    overlay = draw_detection_boxes(overlay, detections, frame, intrinsics)
    preview_path = output_dir / "first_frame_cad_projection.png"
    if not cv2.imwrite(str(preview_path), overlay):
        raise RuntimeError(f"第一帧 CAD 投影图保存失败: {preview_path}")
    print(f"[PREVIEW] 第一帧 CAD 投影图已保存: {preview_path}")
    if frame.T_camera_cad is None:
        print("[PREVIEW][WARN] 当前人工映射没有得到有效 T_camera_cad，画面中不会有 CAD 圆。")
        return

    try:
        cv2.imshow("CAD Projection Preview", overlay)
        print("[PREVIEW] 已显示 CAD 圆、孔号、YOLO 中心和重投影误差；按任意键继续，Q 退出。")
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("CAD Projection Preview")
        if key in {ord("q"), ord("Q"), 27}:
            raise RuntimeError("用户在 CAD 投影复核窗口中退出")
    except cv2.error:
        # 离线/无桌面环境仍保留 PNG，不因无法弹窗阻断配准报告生成。
        print("[PREVIEW][WARN] 当前环境无法弹出 CAD 投影窗口，请直接查看保存的 PNG。")


def _parse_mapping_text(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in re.split(r"[,;，；\s]+", text):
        token = item.strip()
        if not token:
            continue
        match = re.match(r"^(?:detection[-_])?(\d+)\s*[:=、→-]\s*(CAD[-_]\d+)$", token, re.IGNORECASE)
        if match is None:
            raise ValueError(f"无法解析人工映射项: {token!r}；格式应为 0=CAD-01")
        detection_id, cad_id = match.groups()
        mapping[detection_id] = cad_id.upper().replace("_", "-")
    return mapping


def _render_mouse_mapping_preview(
    preview: np.ndarray,
    image_width: int,
    detections: list[Any],
    intrinsics: Any,
    model: Any,
    mapping: dict[str, str],
    selected_detection_id: str | None,
    message: str,
) -> np.ndarray:
    canvas = preview.copy()
    centers = undistort_pixels(
        np.asarray([detection.center_px for detection in detections], dtype=np.float64),
        intrinsics,
    )
    detection_points = {
        str(detection.detection_id): tuple(np.round(point).astype(int))
        for detection, point in zip(detections, centers)
    }
    for detection in detections:
        detection_id = str(detection.detection_id)
        point = detection_points[detection_id]
        if detection_id == selected_detection_id:
            color, radius = (0, 255, 255), 18
        elif detection_id in mapping:
            color, radius = (0, 220, 0), 14
        else:
            color, radius = (0, 0, 255), 11
        cv2.circle(canvas, point, radius, color, 2, cv2.LINE_AA)

    cad_points = _cad_layout_points(model, preview.shape[0])
    for cad_id, local_point in cad_points.items():
        if cad_id in mapping.values():
            point = (image_width + local_point[0], local_point[1])
            cv2.circle(canvas, point, 12, (255, 180, 0), 2, cv2.LINE_AA)

    status = f"CLICK YOLO -> CLICK CAD | pairs={len(mapping)}/4 | {message}"
    if len(status) > 145:
        status = status[:142] + "..."
    origin = (20, max(28, preview.shape[0] - 18))
    cv2.putText(canvas, status, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, status, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _prompt_mouse_mapping(
    image: np.ndarray,
    detections: list[Any],
    model: Any,
    intrinsics: Any,
    preview: np.ndarray,
) -> dict[str, str]:
    """鼠标点选：先点 RGB 中的 YOLO 孔，再点右侧 CAD 孔。"""

    image_width = int(image.shape[1])
    centers = undistort_pixels(
        np.asarray([detection.center_px for detection in detections], dtype=np.float64),
        intrinsics,
    )
    detection_points = {
        str(detection.detection_id): tuple(np.round(point).astype(int))
        for detection, point in zip(detections, centers)
    }
    cad_points = _cad_layout_points(model, preview.shape[0])
    state: dict[str, Any] = {
        "mapping": {},
        "selected_detection_id": None,
        "history": [],
        "message": "click a YOLO center first",
    }

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if x < image_width:
            candidates = [
                (float(np.hypot(x - point[0], y - point[1])), detection_id)
                for detection_id, point in detection_points.items()
            ]
            distance, detection_id = min(candidates, default=(float("inf"), None))
            if detection_id is None or distance > 45.0:
                state["message"] = "click closer to a YOLO center"
                return
            old_cad_id = state["mapping"].pop(detection_id, None)
            if old_cad_id is not None:
                state["history"] = [item for item in state["history"] if item[0] != detection_id]
            state["selected_detection_id"] = detection_id
            state["message"] = f"det-{detection_id} selected; click its CAD hole"
            return

        local_x = x - image_width
        candidates = [
            (float(np.hypot(local_x - point[0], y - point[1])), cad_id)
            for cad_id, point in cad_points.items()
        ]
        distance, cad_id = min(candidates, default=(float("inf"), None))
        selected_detection_id = state["selected_detection_id"]
        if cad_id is None or distance > 45.0:
            state["message"] = "click closer to a CAD center"
            return
        if selected_detection_id is None:
            state["message"] = "select a YOLO center first"
            return
        if cad_id in state["mapping"].values():
            state["message"] = f"{cad_id} is already assigned"
            return
        state["mapping"][selected_detection_id] = cad_id
        state["history"].append((selected_detection_id, cad_id))
        state["selected_detection_id"] = None
        state["message"] = f"saved det-{selected_detection_id} -> {cad_id}"

    window_name = "CAD Mapping - click YOLO then CAD"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)
        while True:
            display = _render_mouse_mapping_preview(
                preview,
                image_width,
                detections,
                intrinsics,
                model,
                state["mapping"],
                state["selected_detection_id"],
                state["message"],
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF
            if key in {ord("q"), ord("Q"), 27}:
                raise RuntimeError("用户退出鼠标映射")
            if key in {ord("r"), ord("R")}:
                state["mapping"].clear()
                state["selected_detection_id"] = None
                state["history"].clear()
                state["message"] = "mapping reset"
            elif key in {ord("z"), ord("Z"), 8} and state["history"]:
                detection_id, cad_id = state["history"].pop()
                if state["mapping"].get(detection_id) == cad_id:
                    state["mapping"].pop(detection_id, None)
                state["selected_detection_id"] = None
                state["message"] = f"undid det-{detection_id} -> {cad_id}"
            elif key in {ord("s"), ord("S"), 13, 10, ord(" ")}:
                if len(state["mapping"]) >= 4:
                    return dict(state["mapping"])
                state["message"] = "need at least 4 pairs"
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


def _prompt_initial_mapping(
    image: np.ndarray,
    detections: list[Any],
    model: Any,
    intrinsics: Any,
    output_dir: Path,
) -> dict[str, str]:
    """现场第一帧用鼠标确认 detection→CAD 映射；无 GUI 时回退到文本输入。"""

    preview = _draw_capture_detection_preview(
        image,
        detections,
        intrinsics,
        "FIRST FRAME: confirm detection_id -> CAD hole_id",
        expected_hole_count=len(model.holes),
        cad_model=model,
    )
    preview_path = output_dir / "first_frame_detection_ids.png"
    cv2.imwrite(str(preview_path), preview)
    print(f"[MAPPING] 第一帧检测编号图已保存: {preview_path}")
    print("[MAPPING] 鼠标操作：先点击左侧 YOLO 孔中心，再点击右侧 CAD 孔；S/Enter 完成，R 重置，Z 撤销，Q 退出。")
    print(
        "[MAPPING][WARN] 当前仅使用孔中心/直径时，左右镜像可能无法由图像唯一确定；"
        "请根据 CAD +X 方向或工件实体基准确认，不要任选一个对称解。"
    )
    try:
        mapping = _prompt_mouse_mapping(image, detections, model, intrinsics, preview)
    except cv2.error:
        print("[MAPPING][WARN] 当前环境无法使用鼠标窗口，回退到终端输入。")
        text = input("[MAPPING] 请输入至少 4 组映射，例如 0=CAD-01,1=CAD-05：\n> ").strip()
        if not text:
            raise RuntimeError("没有输入第一帧人工映射，无法在没有 CAD 先验时进行 PnP")
        mapping = _parse_mapping_text(text)
    if len(mapping) < 4:
        raise ValueError(f"第一帧人工映射只有 {len(mapping)} 组，至少需要 4 组")
    (output_dir / "manual_mapping_input.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return mapping


def _find_latest_reusable_base_cad(cad_json: str | Path) -> tuple[Path, np.ndarray] | None:
    """寻找上一轮通过 Stage 0 的 T_base_cad；只用于预览复用，不授权运动。"""

    current_cad = Path(cad_json).resolve()
    reports = sorted(
        DEFAULT_RUN_ROOT.glob("*/cad_registration_report.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for report_path in reports:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            result = payload.get("result", {})
            if not bool(result.get("success")):
                continue
            previous_cad = payload.get("inputs", {}).get("cad_json")
            if previous_cad and Path(previous_cad).resolve() != current_cad:
                continue
            transform = np.asarray(result.get("T_base_cad"), dtype=np.float64)
            if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
                continue
            return report_path, transform
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return None


def _choose_prior_reuse_or_auto_or_mouse(
    image: np.ndarray,
    detections: list[Any],
    model: Any,
    intrinsics: Any,
    preview: np.ndarray,
    previous_report: Path,
) -> str:
    """在复用旧位姿、自动几何匹配和人工映射之间选择。"""

    canvas = preview.copy()
    message = "A=AUTO geometry | P=REUSE previous T_base_cad | M=MOUSE mapping | Q=QUIT"
    origin = (20, max(28, image.shape[0] - 18))
    cv2.putText(canvas, message, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, message, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    print(f"[REUSE] 找到上一轮可复用结果: {previous_report}")
    print("[REUSE][WARN] 只有工件相对机器人基座未移动时才能复用。A=自动匹配，P=复用，M=鼠标映射，Q=退出。")
    window_name = "CAD Mapping Mode"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in {ord("p"), ord("P"), ord("u"), ord("U")}:
                return "reuse"
            if key in {ord("a"), ord("A")}:
                return "auto"
            if key in {ord("m"), ord("M")}:
                return "manual"
            if key in {ord("q"), ord("Q"), 27}:
                raise RuntimeError("用户退出 CAD 映射模式选择")
    except cv2.error:
        text = input("[REUSE] 输入 A 自动匹配，P 复用旧位姿，M 进入鼠标映射：\n> ").strip().lower()
        if text in {"p", "u"}:
            return "reuse"
        if text == "a":
            return "auto"
        if text == "m":
            return "manual"
        raise RuntimeError("未选择 CAD 映射方式")
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


def _load_transforms(args: argparse.Namespace, base_camera_override: np.ndarray | None = None):
    T_base_camera = (
        load_transform_json(args.base_camera, "T_base_camera")
        if args.base_camera
        else None if base_camera_override is None else np.asarray(base_camera_override, dtype=np.float64)
    )
    if args.prior_camera_cad and args.prior_base_cad:
        raise ValueError("--prior-camera-cad 与 --prior-base-cad 只能二选一")
    if args.prior_base_cad:
        if T_base_camera is None:
            raise ValueError("--prior-base-cad 必须同时提供 --base-camera")
        prior_base_cad = load_transform_json(args.prior_base_cad, "T_base_cad")
        prior_camera_cad = invert_transform(T_base_camera) @ prior_base_cad
        prior_source = str(args.prior_base_cad)
    elif args.prior_camera_cad:
        prior_camera_cad = load_transform_json(args.prior_camera_cad, "T_camera_cad")
        prior_source = str(args.prior_camera_cad)
    else:
        prior_camera_cad = None
        prior_source = None
    return prior_camera_cad, T_base_camera, prior_source


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _apply_robot_connection_overrides(args)
    if args.frames != 1 and not args.image and not args.image_dir and args.frames != 5:
        raise ValueError("无 --image 时建议采集 5 帧；如只做预览可显式使用 --frames 1")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_RUN_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model, cad_source, cad_json = _load_model(args)
    intrinsics = load_intrinsics_json(args.intrinsics)

    image_paths = _collect_image_paths(args)
    live_mode = not bool(image_paths)
    live_robot_meta: dict[str, Any] = {}
    if live_mode:
        if args.detections_json:
            raise ValueError("现场 RGB 采集模式直接运行 YOLO，不使用 --detections-json")
        images, image_sources, detections_by_frame, live_base_camera, live_robot_meta = _capture_live_robot(
            args,
            intrinsics,
            output_dir,
            expected_hole_count=len(model.holes),
            cad_model=model,
        )
        prior_camera_cad, T_base_camera, prior_source = _load_transforms(
            args, base_camera_override=live_base_camera
        )
    else:
        images = []
        image_sources = []
        for path in image_paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"无法读取 RGB 图片: {path}")
            images.append(image)
            image_sources.append(str(path))
        prior_camera_cad, T_base_camera, prior_source = _load_transforms(args)
        detections_by_frame = _load_detection_frames(args, images, intrinsics)

    mapping = _load_mapping(args)
    if prior_camera_cad is None and mapping is None:
        if not live_mode:
            print("[AUTO] 离线模式没有提供先验，将使用孔间几何自动匹配。")
        else:
            reusable = _find_latest_reusable_base_cad(cad_json)
            if reusable is not None and T_base_camera is not None:
                previous_report, previous_T_base_cad = reusable
                choice_preview = _draw_capture_detection_preview(
                    images[0],
                    detections_by_frame[0],
                    intrinsics,
                    "MAPPING MODE: A auto | P reuse | M mouse",
                    expected_hole_count=len(model.holes),
                    cad_model=model,
                )
                choice = _choose_prior_reuse_or_auto_or_mouse(
                    images[0],
                    detections_by_frame[0],
                    model,
                    intrinsics,
                    choice_preview,
                    previous_report,
                )
                if choice == "reuse":
                    prior_camera_cad = invert_transform(T_base_camera) @ previous_T_base_cad
                    prior_source = f"reused_latest_T_base_cad:{previous_report}"
                    live_robot_meta["reused_previous_report"] = str(previous_report)
                    live_robot_meta["reused_previous_transform"] = "T_base_cad"
                    print("[REUSE] 已复用上一轮 T_base_cad，并按当前 T_base_camera 换算 T_camera_cad。")
                elif choice == "manual":
                    mapping = _prompt_initial_mapping(
                        images[0], detections_by_frame[0], model, intrinsics, output_dir
                    )
                else:
                    print("[AUTO] 将使用孔间几何自动匹配，不复用上一轮 CAD 位姿。")
            else:
                print("[AUTO] 未找到可安全复用的旧位姿，将使用孔间几何自动匹配。")

    for index, image in enumerate(images):
        cv2.imwrite(str(output_dir / f"rgb_original_{index:03d}.png"), image)
        if image.shape[1] != intrinsics.width or image.shape[0] != intrinsics.height:
            print(
                f"[WARN] frame {index} 尺寸 {image.shape[1]}x{image.shape[0]} 与内参 "
                f"{intrinsics.width}x{intrinsics.height} 不一致"
            )
    config = CadRegistrationConfig(
        match_distance_px=args.match_distance_px,
        reprojection_rmse_px=args.rmse_px,
        reprojection_max_px=args.max_error_px,
        cross_frame_center_p95_mm=args.cross_frame_p95_mm,
        min_valid_frames=args.min_valid_frames,
        automatic_triangle_ratio_tolerance=args.auto_triangle_tolerance,
        automatic_initial_distance_px=args.auto_initial_distance_px,
        automatic_inlier_distance_px=args.auto_inlier_distance_px,
        automatic_max_hypotheses=args.auto_max_hypotheses,
        automatic_ambiguity_score_margin_px=args.auto_ambiguity_margin_px,
        automatic_diameter_consistency_weight=args.auto_diameter_weight,
        automatic_max_pnp_candidates=args.auto_max_pnp_candidates,
    )
    # 人工映射输入后先单独求第一帧 PnP，立即把 CAD 圆投影回图像。
    # 这样现场人员可以在继续处理 5 帧前确认孔号和 CAD 坐标方向没有错。
    first_preview_frame = register_cad_frame(
        frame_index=0,
        detections=detections_by_frame[0],
        model=model,
        intrinsics=intrinsics,
        prior_T_camera_cad=prior_camera_cad,
        T_base_camera=T_base_camera,
        config=config,
        manual_mapping=mapping,
    )
    if mapping is None and live_mode and not first_preview_frame.success:
        print("[AUTO][WARN] 第一帧自动匹配未通过，进入人工映射复核。")
        for reason in first_preview_frame.failure_reasons:
            print(f"[AUTO][WARN] {reason}")
        mapping = _prompt_initial_mapping(
            images[0], detections_by_frame[0], model, intrinsics, output_dir
        )
        first_preview_frame = register_cad_frame(
            frame_index=0,
            detections=detections_by_frame[0],
            model=model,
            intrinsics=intrinsics,
            prior_T_camera_cad=prior_camera_cad,
            T_base_camera=T_base_camera,
            config=config,
            manual_mapping=mapping,
        )
    if live_mode:
        _save_and_show_first_cad_projection(
            images[0],
            detections_by_frame[0],
            model,
            intrinsics,
            first_preview_frame,
            output_dir,
        )
    else:
        preview_overlay = draw_cad_overlay(
            images[0],
            model,
            intrinsics,
            first_preview_frame,
            title="CAD projection preview after mapping",
        )
        preview_overlay = draw_detection_boxes(
            preview_overlay,
            detections_by_frame[0],
            first_preview_frame,
            intrinsics,
        )
        cv2.imwrite(str(output_dir / "first_frame_cad_projection.png"), preview_overlay)

    matching_quality = first_preview_frame.quality.get("matching", {})
    if matching_quality.get("matching_method"):
        print(
            "[MAPPING] method="
            f"{matching_quality['matching_method']} | "
            f"matches={len(first_preview_frame.matches)} | "
            f"ambiguous={bool(matching_quality.get('ambiguous', False))}"
        )

    result = register_detections_frames(
        detections_by_frame,
        model,
        intrinsics,
        prior_camera_cad,
        T_base_camera,
        config,
        manual_mapping=mapping,
        sequential_prior=live_mode,
    )
    for index, (image, detections) in enumerate(zip(images, detections_by_frame)):
        frame = result.frames[index]
        overlay = draw_cad_overlay(image, model, intrinsics, frame, title=f"CAD registration frame {index}")
        overlay = draw_detection_boxes(overlay, detections, frame, intrinsics)
        cv2.imwrite(str(output_dir / f"frame_{index:03d}_overlay.png"), overlay)

    report = save_registration_report(
        output_dir,
        result,
        model=model,
        intrinsics=intrinsics,
        inputs={
            "cad_source": cad_source,
            "cad_json": cad_json,
            "intrinsics": str(args.intrinsics),
            "images": image_sources,
            "prior_transform": prior_source,
            "base_camera": args.base_camera or ("robot_sdk_read_only" if live_mode else None),
            "handeye": live_robot_meta.get("handeye_path"),
            "live_robot_capture": live_mode,
            "live_robot_metadata": live_robot_meta,
            "mapping_json": args.mapping_json,
            "manual_mapping_used": mapping,
            "mapping_mode": (
                "manual" if mapping is not None else
                "prior_projection" if prior_camera_cad is not None else
                "automatic_geometry"
            ),
            "first_frame_mapping_method": matching_quality.get("matching_method"),
            "yolo_model": args.model if not args.detections_json else None,
            "stage0_only": True,
        },
        detections_by_frame=detections_by_frame,
    )
    status = "PASS" if result.success else "FAIL"
    print(f"[{status}] CAD 配准预览完成: {output_dir}")
    print(f"[INFO] 报告: {report}")
    print(f"[INFO] motion_allowed={result.motion_allowed}; multi_frame_gate_pass={result.multi_frame_gate_pass}")
    if result.failure_reasons:
        for reason in result.failure_reasons:
            print(f"[WARN] {reason}")
    return 0 if result.success else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
