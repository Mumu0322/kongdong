#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ChArUco检测；正式RGB-PnP板位姿与旧点云诊断板位姿相互隔离。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import BOARD_CFG, CAMERA_CFG
from .drawing import draw_text_panel
from .geometry import make_transform
from .camera import CameraIntrinsics


@dataclass
class BoardPoseResult:
    ok: bool
    status: str
    T_pointcloud_board: np.ndarray | None
    rgb_overlay: np.ndarray
    charuco_count: int = 0
    valid_3d_count: int = 0
    corner_rmse_mm: float = float("nan")
    corner_max_error_mm: float = float("nan")
    plane_rmse_mm: float = float("nan")
    plane_inlier_count: int = 0
    board_mask_point_count: int = 0
    marker_count: int = 0
    used_corner_ids: list[int] | None = None
    valid_corner_ids: list[int] | None = None
    image_points: list[list[float]] | None = None
    object_points_mm: list[list[float]] | None = None
    pointcloud_points_mm: list[list[float]] | None = None
    calibration_frame: str = "pointcloud"
    T_rgb_board: np.ndarray | None = None
    rgb_pnp_inlier_count: int = 0
    rgb_reprojection_rmse_px: float = float("nan")
    rgb_reprojection_max_px: float = float("nan")


def get_aruco_dictionary():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("当前 OpenCV 不包含 aruco 模块，请安装 opencv-contrib-python")
    aruco = cv2.aruco
    if not hasattr(aruco, BOARD_CFG.aruco_dict_name):
        raise RuntimeError(f"cv2.aruco 不包含字典 {BOARD_CFG.aruco_dict_name}")
    return aruco.getPredefinedDictionary(getattr(aruco, BOARD_CFG.aruco_dict_name))


def create_charuco_board():
    aruco = cv2.aruco
    dictionary = get_aruco_dictionary()
    try:
        # OpenCV 4.7+
        board = aruco.CharucoBoard(
            (BOARD_CFG.squares_x, BOARD_CFG.squares_y),
            float(BOARD_CFG.square_length_mm),
            float(BOARD_CFG.marker_length_mm),
            dictionary,
        )
    except Exception:
        # OpenCV legacy
        board = aruco.CharucoBoard_create(
            int(BOARD_CFG.squares_x),
            int(BOARD_CFG.squares_y),
            float(BOARD_CFG.square_length_mm),
            float(BOARD_CFG.marker_length_mm),
            dictionary,
        )
    return board, dictionary


def get_board_chessboard_corners(board) -> np.ndarray:
    if hasattr(board, "getChessboardCorners"):
        pts = board.getChessboardCorners()
    else:
        pts = board.chessboardCorners
    return np.asarray(pts, dtype=np.float64).reshape(-1, 3)


def create_detector_parameters():
    aruco = cv2.aruco
    try:
        return aruco.DetectorParameters()
    except Exception:
        return aruco.DetectorParameters_create()


def detect_charuco(color_bgr: np.ndarray, board, dictionary):
    aruco = cv2.aruco
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    params = create_detector_parameters()

    if hasattr(aruco, "CharucoDetector"):
        try:
            detector = aruco.CharucoDetector(board)
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
            marker_count = 0 if marker_ids is None else int(len(marker_ids))
            return marker_corners, marker_ids, charuco_corners, charuco_ids, marker_count
        except Exception as exc:
            print("[WARN] CharucoDetector.detectBoard 失败，尝试旧接口:", exc)

    if not hasattr(aruco, "detectMarkers"):
        raise RuntimeError("当前 OpenCV aruco 模块既没有 CharucoDetector.detectBoard，也没有旧版 detectMarkers")

    marker_corners, marker_ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=params)
    marker_count = 0 if marker_ids is None else int(len(marker_ids))
    charuco_corners = None
    charuco_ids = None

    if marker_ids is not None and len(marker_ids) > 0:
        try:
            aruco.refineDetectedMarkers(gray, board, marker_corners, marker_ids, rejected)
        except Exception:
            pass
        if hasattr(aruco, "interpolateCornersCharuco"):
            try:
                _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
            except Exception as exc:
                print("[WARN] interpolateCornersCharuco 失败:", exc)

    return marker_corners, marker_ids, charuco_corners, charuco_ids, marker_count


def _charuco_correspondences(board, charuco_corners, charuco_ids) -> tuple[np.ndarray, np.ndarray, list[int]]:
    chessboard_corners = get_board_chessboard_corners(board)
    corners = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
    ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_ids: list[int] = []
    for uv, corner_id in zip(corners, ids):
        index = int(corner_id)
        if index < 0 or index >= chessboard_corners.shape[0] or not np.isfinite(uv).all():
            continue
        object_points.append(chessboard_corners[index])
        image_points.append(uv)
        used_ids.append(index)
    return (
        np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
        np.asarray(image_points, dtype=np.float64).reshape(-1, 2),
        used_ids,
    )


def estimate_rgb_board_pose(
    color_bgr: np.ndarray,
    intrinsics: CameraIntrinsics | None,
    board,
    dictionary,
) -> BoardPoseResult:
    """由RGB二维角点和已知板尺寸求 ^C_rgb T_Q；不读取深度或点云。"""
    overlay = color_bgr.copy()
    marker_corners, marker_ids, charuco_corners, charuco_ids, marker_count = detect_charuco(
        color_bgr, board, dictionary,
    )
    if marker_ids is not None and len(marker_ids) > 0:
        try:
            cv2.aruco.drawDetectedMarkers(overlay, marker_corners, marker_ids)
        except Exception:
            pass
    if charuco_corners is None or charuco_ids is None:
        draw_text_panel(overlay, [("未检测到 ChArUco 标定板", (0, 0, 255))])
        return BoardPoseResult(
            False, "charuco_not_detected", None, overlay,
            marker_count=marker_count, calibration_frame="rgb_camera",
        )
    try:
        cv2.aruco.drawDetectedCornersCharuco(overlay, charuco_corners, charuco_ids, (255, 0, 255))
    except Exception:
        pass

    object_points, image_points, used_ids = _charuco_correspondences(
        board, charuco_corners, charuco_ids,
    )
    charuco_count = int(object_points.shape[0])
    if intrinsics is None:
        draw_text_panel(overlay, [("RGB内参不可用，不能进行PnP手眼采集", (0, 0, 255))])
        return BoardPoseResult(
            False, "rgb_intrinsics_missing", None, overlay,
            charuco_count=charuco_count, marker_count=marker_count,
            used_corner_ids=used_ids, image_points=image_points.tolist(),
            object_points_mm=object_points.tolist(), calibration_frame="rgb_camera",
        )
    height, width = color_bgr.shape[:2]
    if (int(intrinsics.width), int(intrinsics.height)) != (width, height):
        draw_text_panel(overlay, [("RGB内参与当前图像尺寸不一致", (0, 0, 255))])
        return BoardPoseResult(
            False, "rgb_intrinsics_size_mismatch", None, overlay,
            charuco_count=charuco_count, marker_count=marker_count,
            used_corner_ids=used_ids, image_points=image_points.tolist(),
            object_points_mm=object_points.tolist(), calibration_frame="rgb_camera",
        )
    if charuco_count < int(BOARD_CFG.min_charuco_corners):
        draw_text_panel(overlay, [(f"ChArUco角点过少：{charuco_count}", (0, 0, 255))])
        return BoardPoseResult(
            False, "charuco_corners_too_few", None, overlay,
            charuco_count=charuco_count, marker_count=marker_count,
            used_corner_ids=used_ids, image_points=image_points.tolist(),
            object_points_mm=object_points.tolist(), calibration_frame="rgb_camera",
        )

    camera_matrix = intrinsics.camera_matrix()
    distortion = np.asarray(intrinsics.distortion, dtype=np.float64).reshape(-1, 1)
    distortion_arg = distortion if distortion.size else None
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            distortion_arg,
            iterationsCount=int(BOARD_CFG.rgb_pnp_ransac_iterations),
            reprojectionError=float(BOARD_CFG.rgb_pnp_ransac_reprojection_px),
            confidence=float(BOARD_CFG.rgb_pnp_ransac_confidence),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error as exc:
        draw_text_panel(overlay, [(f"RGB PnP失败：{exc}", (0, 0, 255))])
        return BoardPoseResult(
            False, "rgb_pnp_exception", None, overlay,
            charuco_count=charuco_count, marker_count=marker_count,
            used_corner_ids=used_ids, image_points=image_points.tolist(),
            object_points_mm=object_points.tolist(), calibration_frame="rgb_camera",
        )
    if not success or rvec is None or tvec is None or inliers is None:
        draw_text_panel(overlay, [("RGB PnP未得到有效解", (0, 0, 255))])
        return BoardPoseResult(
            False, "rgb_pnp_failed", None, overlay,
            charuco_count=charuco_count, marker_count=marker_count,
            used_corner_ids=used_ids, image_points=image_points.tolist(),
            object_points_mm=object_points.tolist(), calibration_frame="rgb_camera",
        )

    inlier_indices = np.asarray(inliers, dtype=np.int32).reshape(-1)
    inlier_object = object_points[inlier_indices]
    inlier_image = image_points[inlier_indices]
    if hasattr(cv2, "solvePnPRefineLM") and inlier_indices.size >= 4:
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                inlier_object, inlier_image, camera_matrix, distortion_arg, rvec, tvec,
            )
        except cv2.error:
            pass
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion_arg)
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    rmse = float(np.sqrt(np.mean(errors * errors)))
    max_error = float(np.max(errors))
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    camera_points_z = ((rotation @ object_points.T).T + translation)[:, 2]
    positive_depth = bool(translation[2] > 0.0 and np.all(camera_points_z > 0.0))
    ok = bool(
        positive_depth
        and inlier_indices.size >= int(BOARD_CFG.min_charuco_corners)
        and rmse <= float(BOARD_CFG.max_rgb_reprojection_rmse_px)
        and max_error <= float(BOARD_CFG.max_rgb_reprojection_error_px)
    )
    status_parts: list[str] = []
    if not positive_depth:
        status_parts.append("rgb_pnp_board_behind_camera")
    if inlier_indices.size < int(BOARD_CFG.min_charuco_corners):
        status_parts.append("rgb_pnp_inliers_too_few")
    if rmse > float(BOARD_CFG.max_rgb_reprojection_rmse_px):
        status_parts.append(f"rgb_reprojection_rmse_high_{rmse:.3f}")
    if max_error > float(BOARD_CFG.max_rgb_reprojection_error_px):
        status_parts.append(f"rgb_reprojection_max_high_{max_error:.3f}")
    if not status_parts:
        status_parts.append("rgb_pnp_ok")

    # OpenCV 在坐标轴端点出画时会逐帧输出 warning；这不影响 PnP，
    # 但叠加图本身也不再可靠，因此仅在全部轴端点位于画面内时绘制。
    try:
        axis_length = float(BOARD_CFG.square_length_mm) * 3.0
        axis_points = np.array(
            [[0.0, 0.0, 0.0], [axis_length, 0.0, 0.0],
             [0.0, axis_length, 0.0], [0.0, 0.0, axis_length]],
            dtype=np.float64,
        )
        axis_uv, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, distortion_arg)
        axis_uv = axis_uv.reshape(-1, 2)
        inside = (
            np.all(axis_uv[:, 0] >= 0.0) and np.all(axis_uv[:, 0] < float(width))
            and np.all(axis_uv[:, 1] >= 0.0) and np.all(axis_uv[:, 1] < float(height))
        )
        if inside:
            cv2.drawFrameAxes(overlay, camera_matrix, distortion_arg, rvec, tvec, axis_length, 3)
    except cv2.error:
        pass
    text_color = (0, 255, 0) if ok else (0, 165, 255)
    draw_text_panel(overlay, [
        (f"RGB ChArUco角点={charuco_count}，PnP内点={inlier_indices.size}", text_color),
        (f"RGB重投影 RMSE={rmse:.3f}px，最大={max_error:.3f}px", text_color),
        ("正式链路：RGB 2D角点 + 内参 + PnP；深度不参与手眼求解", (255, 255, 255)),
        ("c: 采集5帧选1帧 | h: RGB诊断 | v: E7验证 | d: 归档最后样本", (255, 255, 255)),
    ])
    inlier_ids = [used_ids[int(index)] for index in inlier_indices]
    return BoardPoseResult(
        ok=ok,
        status="|".join(status_parts),
        T_pointcloud_board=None,
        rgb_overlay=overlay,
        charuco_count=charuco_count,
        marker_count=marker_count,
        used_corner_ids=used_ids,
        valid_corner_ids=inlier_ids,
        image_points=image_points.tolist(),
        object_points_mm=object_points.tolist(),
        calibration_frame="rgb_camera",
        T_rgb_board=make_transform(rotation, translation),
        rgb_pnp_inlier_count=int(inlier_indices.size),
        rgb_reprojection_rmse_px=rmse,
        rgb_reprojection_max_px=max_error,
    )


def valid_xyz_mask(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    return (
        np.isfinite(pts).all(axis=-1)
        & (pts[..., 2] > CAMERA_CFG.min_valid_z_mm)
        & (pts[..., 2] < CAMERA_CFG.max_valid_z_mm)
    )


def get_xyz_window_median(xyz_map: np.ndarray, u: float, v: float, radius: int) -> np.ndarray | None:
    h, w = xyz_map.shape[:2]
    ui = int(round(float(u)))
    vi = int(round(float(v)))
    if ui < 0 or ui >= w or vi < 0 or vi >= h:
        return None
    x0 = max(0, ui - radius)
    x1 = min(w, ui + radius + 1)
    y0 = max(0, vi - radius)
    y1 = min(h, vi + radius + 1)
    patch = xyz_map[y0:y1, x0:x1, :].reshape(-1, 3).astype(np.float64)
    mask = valid_xyz_mask(patch)
    pts = patch[mask]
    if pts.shape[0] < max(4, radius + 2):
        return None
    return np.median(pts, axis=0)


def fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    center = np.mean(pts, axis=0)
    _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
    normal = vh[-1]
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    distances = (pts - center) @ normal
    rmse = float(np.sqrt(np.mean(distances * distances)))
    return normal, rmse, center


def fit_plane_ransac(points: np.ndarray, tol_mm: float, iterations: int):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 30:
        return None, np.zeros((pts.shape[0],), dtype=bool), float("nan"), None
    best_mask = np.zeros((pts.shape[0],), dtype=bool)
    best_count = 0
    rng = np.random.default_rng()
    for _ in range(iterations):
        idx = rng.choice(pts.shape[0], size=3, replace=False)
        p0, p1, p2 = pts[idx]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = np.abs((pts - p0) @ n)
        mask = d < tol_mm
        count = int(np.sum(mask))
        if count > best_count:
            best_count = count
            best_mask = mask
    if best_count < 20:
        return None, best_mask, float("nan"), None
    normal, rmse, center = fit_plane_svd(pts[best_mask])
    return normal, best_mask, rmse, center


def project_points_to_plane(points: np.ndarray, normal: np.ndarray, center: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / max(np.linalg.norm(n), 1e-12)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    signed = (pts - c) @ n
    return pts - signed[:, None] * n[None, :]


def fit_rigid_transform_kabsch(src: np.ndarray, dst: np.ndarray):
    """求 R,t，使 dst ≈ R @ src + t。返回 R, t, per-point residual。"""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.shape[0] != dst.shape[0] or src.shape[0] < 3:
        raise ValueError("Kabsch requires at least 3 point pairs")
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src0 = src - src_mean
    dst0 = dst - dst_mean
    H = src0.T @ dst0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = dst_mean - R @ src_mean
    residuals = np.linalg.norm((R @ src.T).T + t - dst, axis=1)
    return R, t, residuals


def robust_board_to_pointcloud_transform(board_pts: np.ndarray, pc_pts: np.ndarray):
    """带 MAD 稳健剔除的 3 轮迭代刚体拟合，返回 R,t,rmse,max_err,inlier_mask。"""
    board_pts = np.asarray(board_pts, dtype=np.float64).reshape(-1, 3)
    pc_pts = np.asarray(pc_pts, dtype=np.float64).reshape(-1, 3)
    mask = np.ones((board_pts.shape[0],), dtype=bool)
    for _ in range(3):
        R, t, residuals = fit_rigid_transform_kabsch(board_pts[mask], pc_pts[mask])
        med = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - med)))
        thresh = max(1.5, med + 3.0 * 1.4826 * mad)
        new_local = residuals <= thresh
        new_mask = mask.copy()
        new_mask[np.where(mask)[0]] = new_local
        if int(np.sum(new_mask)) < 6 or np.array_equal(new_mask, mask):
            break
        mask = new_mask
    R, t, residuals = fit_rigid_transform_kabsch(board_pts[mask], pc_pts[mask])
    rmse = float(np.sqrt(np.mean(residuals * residuals)))
    max_err = float(np.max(residuals)) if residuals.size else float("nan")
    return R, t, rmse, max_err, mask


def make_board_mask(image_shape: tuple[int, int], marker_corners, charuco_corners) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts_list = []
    if marker_corners is not None:
        for c in marker_corners:
            pts_list.append(np.asarray(c, dtype=np.float32).reshape(-1, 2))
    if charuco_corners is not None:
        pts_list.append(np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2))
    if not pts_list:
        return mask
    pts = np.vstack(pts_list)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 4:
        return mask
    hull = cv2.convexHull(pts.astype(np.float32)).astype(np.int32)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def extract_board_cloud_from_mask(xyz_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    step = max(1, int(CAMERA_CFG.board_mask_sample_step_px))
    sampled_mask = mask[::step, ::step] > 0
    sampled_xyz = xyz_map[::step, ::step, :].astype(np.float64)
    pts = sampled_xyz[sampled_mask]
    pts = pts[valid_xyz_mask(pts)]
    # 防止点太多导致 RANSAC 变慢，随机限幅。
    if pts.shape[0] > 12000:
        idx = np.random.default_rng().choice(pts.shape[0], size=12000, replace=False)
        pts = pts[idx]
    return pts


def draw_board_axes_on_image(img: np.ndarray, board_pts: np.ndarray, img_pts: np.ndarray) -> None:
    if board_pts.shape[0] < 8:
        return
    src = board_pts[:, :2].astype(np.float32)
    dst = img_pts.astype(np.float32)
    H, _inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        return
    axis = np.array([[[0, 0]], [[90, 0]], [[0, 90]]], dtype=np.float32)
    proj = cv2.perspectiveTransform(axis, H).reshape(-1, 2)
    o = tuple(np.round(proj[0]).astype(int))
    x = tuple(np.round(proj[1]).astype(int))
    y = tuple(np.round(proj[2]).astype(int))
    cv2.arrowedLine(img, o, x, (0, 0, 255), 3, tipLength=0.18)
    cv2.arrowedLine(img, o, y, (0, 255, 0), 3, tipLength=0.18)
    cv2.putText(img, "Board X", (x[0] + 8, x[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img, "Board Y", (y[0] + 8, y[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    rect = np.array(
        [[
            [0, 0],
            [BOARD_CFG.pattern_width_mm, 0],
            [BOARD_CFG.pattern_width_mm, BOARD_CFG.pattern_height_mm],
            [0, BOARD_CFG.pattern_height_mm],
        ]],
        dtype=np.float32,
    )
    rect_proj = cv2.perspectiveTransform(rect, H).reshape(-1, 2).astype(np.int32)
    cv2.polylines(img, [rect_proj], isClosed=True, color=(255, 255, 0), thickness=2)


def estimate_pointcloud_board_pose(color_bgr: np.ndarray, depth_mm: np.ndarray, xyz_map: np.ndarray, board, dictionary) -> BoardPoseResult:
    overlay = color_bgr.copy()
    marker_corners, marker_ids, charuco_corners, charuco_ids, marker_count = detect_charuco(color_bgr, board, dictionary)

    if marker_ids is not None and len(marker_ids) > 0:
        try:
            cv2.aruco.drawDetectedMarkers(overlay, marker_corners, marker_ids)
        except Exception:
            pass

    if charuco_corners is None or charuco_ids is None:
        draw_text_panel(overlay, [("未检测到 ChArUco 标定板", (0, 0, 255))])
        return BoardPoseResult(False, "charuco_not_detected", None, overlay, marker_count=marker_count)

    charuco_corners_np = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
    charuco_ids_np = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    charuco_count = int(charuco_ids_np.size)

    try:
        cv2.aruco.drawDetectedCornersCharuco(overlay, charuco_corners, charuco_ids, (255, 0, 255))
    except Exception:
        pass

    chessboard_corners = get_board_chessboard_corners(board)
    board_pts, img_pts, pc_pts_raw, ids_used, valid_ids = [], [], [], [], []

    for uv, cid in zip(charuco_corners_np, charuco_ids_np):
        cid_int = int(cid)
        if cid_int < 0 or cid_int >= chessboard_corners.shape[0]:
            continue
        board_pts.append(chessboard_corners[cid_int])
        img_pts.append(uv)
        ids_used.append(cid_int)
        xyz = get_xyz_window_median(xyz_map, float(uv[0]), float(uv[1]), CAMERA_CFG.xyz_window_radius_px)
        if xyz is None:
            pc_pts_raw.append([np.nan, np.nan, np.nan])
        else:
            pc_pts_raw.append(xyz)
            valid_ids.append(cid_int)

    if len(board_pts) < BOARD_CFG.min_charuco_corners:
        draw_text_panel(overlay, [(f"ChArUco 角点过少：{len(board_pts)}", (0, 0, 255))])
        return BoardPoseResult(
            False, "charuco_corners_too_few", None, overlay,
            charuco_count=charuco_count, marker_count=marker_count, used_corner_ids=ids_used,
        )

    board_pts_np = np.asarray(board_pts, dtype=np.float64).reshape(-1, 3)
    img_pts_np = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    pc_pts_raw_np = np.asarray(pc_pts_raw, dtype=np.float64).reshape(-1, 3)
    valid_mask = valid_xyz_mask(pc_pts_raw_np)
    valid_3d_count = int(np.sum(valid_mask))

    for uv, ok in zip(img_pts_np, valid_mask):
        color = (0, 255, 0) if ok else (0, 0, 255)
        cv2.circle(overlay, tuple(np.round(uv).astype(int)), 4, color, -1)

    if valid_3d_count < BOARD_CFG.min_valid_3d_corners:
        draw_text_panel(
            overlay,
            [
                (f"有效 3D 角点过少：{valid_3d_count}/{len(board_pts_np)}", (0, 0, 255)),
                ("请将标定板移到有效深度范围和画面中心区域", (0, 255, 255)),
            ],
        )
        return BoardPoseResult(
            False, "valid_3d_corners_too_few", None, overlay,
            charuco_count=charuco_count, valid_3d_count=valid_3d_count, marker_count=marker_count,
            used_corner_ids=ids_used, valid_corner_ids=valid_ids,
        )

    board_mask = make_board_mask(depth_mm.shape[:2], marker_corners, charuco_corners)
    board_cloud = extract_board_cloud_from_mask(xyz_map, board_mask)
    board_mask_point_count = int(board_cloud.shape[0])
    normal, plane_rmse, plane_inlier_count, plane_center = None, float("nan"), 0, None
    if board_cloud.shape[0] >= 50:
        normal, inlier_mask, plane_rmse, plane_center = fit_plane_ransac(
            board_cloud, CAMERA_CFG.board_plane_ransac_tol_mm, CAMERA_CFG.board_plane_ransac_iter,
        )
        plane_inlier_count = int(np.sum(inlier_mask))

    board_valid = board_pts_np[valid_mask]
    pc_valid = pc_pts_raw_np[valid_mask]
    pc_fit = project_points_to_plane(pc_valid, normal, plane_center) if normal is not None and plane_center is not None else pc_valid

    try:
        R_pq, t_pq, rmse, max_err, inlier_pair_mask = robust_board_to_pointcloud_transform(board_valid, pc_fit)
    except Exception as exc:
        draw_text_panel(overlay, [(f"3D rigid fit failed: {exc}", (0, 0, 255))])
        return BoardPoseResult(
            False, "rigid_fit_failed", None, overlay,
            charuco_count=charuco_count, valid_3d_count=valid_3d_count, marker_count=marker_count,
            plane_rmse_mm=plane_rmse, plane_inlier_count=plane_inlier_count,
        )

    T_pointcloud_board = make_transform(R_pq, t_pq)
    pair_valid_ids = np.asarray(valid_ids, dtype=np.int32)
    final_ids = pair_valid_ids[inlier_pair_mask].tolist() if pair_valid_ids.shape[0] == inlier_pair_mask.shape[0] else valid_ids

    mask_color = np.zeros_like(overlay)
    mask_color[board_mask > 0] = (255, 255, 0)
    overlay = cv2.addWeighted(overlay, 1.0, mask_color, CAMERA_CFG.overlay_alpha, 0)
    draw_board_axes_on_image(overlay, board_valid[inlier_pair_mask], img_pts_np[valid_mask][inlier_pair_mask])

    ok = True
    status_parts = []
    if rmse > BOARD_CFG.max_corner_3d_rmse_mm:
        ok = False
        status_parts.append(f"corner_rmse_high_{rmse:.3f}")
    if np.isfinite(plane_rmse) and plane_rmse > BOARD_CFG.max_plane_rmse_mm:
        ok = False
        status_parts.append(f"plane_rmse_high_{plane_rmse:.3f}")
    if not status_parts:
        status_parts.append("ok")

    text_color = (0, 255, 0) if ok else (0, 165, 255)
    lines = [
        (f"ChArUco 标记={marker_count}，角点={charuco_count}，有效3D={valid_3d_count}", text_color),
        (f"点云-标定板拟合 RMSE={rmse:.3f} mm，最大误差={max_err:.3f} mm", text_color),
        (f"板面点数={board_mask_point_count}，内点={plane_inlier_count}，平面RMSE={plane_rmse:.3f} mm", text_color),
        ("c: 采集5帧选1帧 | h: 诊断 | v: E7验证 | a: 剔除并诊断 | d: 归档最后样本 | m: 手动位姿 | q/ESC: 退出", (255, 255, 255)),
    ]
    draw_text_panel(overlay, lines)

    return BoardPoseResult(
        ok=ok,
        status="|".join(status_parts),
        T_pointcloud_board=T_pointcloud_board,
        rgb_overlay=overlay,
        charuco_count=charuco_count,
        valid_3d_count=valid_3d_count,
        corner_rmse_mm=rmse,
        corner_max_error_mm=max_err,
        plane_rmse_mm=plane_rmse,
        plane_inlier_count=plane_inlier_count,
        board_mask_point_count=board_mask_point_count,
        marker_count=marker_count,
        used_corner_ids=ids_used,
        valid_corner_ids=final_ids,
        image_points=img_pts_np[valid_mask][inlier_pair_mask].tolist(),
        object_points_mm=board_valid[inlier_pair_mask].tolist(),
        pointcloud_points_mm=pc_fit[inlier_pair_mask].tolist(),
    )
