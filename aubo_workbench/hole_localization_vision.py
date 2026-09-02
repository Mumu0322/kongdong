"""Vision primitives for hole detection, selection, depth geometry, and ellipse fitting."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from aubo_workbench.fitting import fit_plane, fit_sphere
from aubo_workbench.optics import camera_ray, undistort_pixels


WINDOW = "YOLO eye-in-hand hole selection (click hole, Enter=confirm, Esc=quit)"
HOLE_DIAMETERS_MM = (65.0, 70.0, 75.0)
COARSE_SURFACE_SELECTION_POLICY = "front_surface_outer_ring_v2"
COARSE_SURFACE_MODEL = "local_tangent_plane_front_surface_outer_ring_v2"


def load_yolo(model_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("缺少 ultralytics，请安装：pip install ultralytics") from exc
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO 模型不存在: {model_path}")
    return YOLO(str(model_path))


def detect(model: Any, image: np.ndarray, confidence: float) -> list[dict[str, Any]]:
    result = model.predict(source=image, conf=confidence, verbose=False)[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    names = getattr(result, "names", {})
    output = []
    for box, score, cls in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = [float(v) for v in box]
        output.append({
            "box": [x1, y1, x2, y2], "confidence": float(score), "class_id": int(cls),
            "class_name": str(names.get(int(cls), cls)) if isinstance(names, dict) else str(cls),
            "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
        })
    return output




def hole_camera_point(
    ring_center_xy: tuple[float, float],
    xyz_map_mm: np.ndarray,
    intrinsics: Any,
    radius_px: float,
    *,
    ray_center_xy: tuple[float, float] | np.ndarray | None = None,
    ray_center_is_undistorted: bool = False,
    include_points: bool = False,
    surface_selection_policy: str = "legacy",
) -> tuple[np.ndarray, dict[str, Any]]:
    """用孔外环带拟合平面，再将孔中心射线与平面求交。

    ring_center_xy 必须是原始图像坐标，用于在 aligned xyz_map 中选择深度环带。
    ray_center_xy 可以是去畸变轮廓拟合得到的中心；这样深度采样坐标和几何射线
    分别使用各自正确的坐标域，避免把去畸变像素直接拿去索引原始深度图。
    """
    ring_u, ring_v = map(float, ring_center_xy)
    h, w = xyz_map_mm.shape[:2]
    policy = str(surface_selection_policy).strip().lower()
    if policy == COARSE_SURFACE_SELECTION_POLICY:
        # 当前工件孔距较密，旧1.25R～1.50R环带会越过孔口附近曲面，
        # 接近相邻孔边和凸起结构，使局部Z/法向变成大范围曲面平均值。
        # 收回到孔口外侧的窄环；仍保留0.10R安全距离，并继续由前景深度
        # 簇过滤孔壁/孔底，不依靠把环带无限外移来规避无效深度。
        ring_inner_factor = 1.10
        ring_outer_factor = 1.30
        front_percentile = 5.0
        surface_band_mm = 8.0
        min_surface_fraction = 0.03
        surface_model = COARSE_SURFACE_MODEL
    elif policy == "legacy":
        ring_inner_factor = 1.08
        ring_outer_factor = 1.35
        front_percentile = 20.0
        surface_band_mm = 8.0
        min_surface_fraction = 0.10
        surface_model = "local_tangent_plane"
    else:
        raise ValueError(f"未知点云表面筛选策略：{surface_selection_policy!r}")

    # 只在孔附近建立环带网格，避免每个粗定位帧都为整幅RGB-D图创建
    # 1280x800级别的坐标矩阵；环带定义按策略选择。
    ring_outer_px = max(8.0, radius_px * ring_outer_factor)
    x0 = max(0, int(math.floor(ring_u - ring_outer_px - 1.0)))
    x1 = min(w, int(math.ceil(ring_u + ring_outer_px + 2.0)))
    y0 = max(0, int(math.floor(ring_v - ring_outer_px - 1.0)))
    y1 = min(h, int(math.ceil(ring_v + ring_outer_px + 2.0)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("孔环带超出深度图范围，无法提取点云")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - ring_u, yy - ring_v)
    ring = (
        (rr >= max(4.0, radius_px * ring_inner_factor))
        & (rr <= max(8.0, radius_px * ring_outer_factor))
    )
    points = xyz_map_mm[y0:y1, x0:x1][ring]
    points = points[np.isfinite(points).all(axis=1)]
    points = points[(points[:, 2] > 100.0) & (points[:, 2] < 3000.0)]
    raw_ring_points = int(len(points))
    if len(points) < 12:
        raise ValueError("孔周围有效深度点不足，无法拟合局部切平面")

    # 镀膜伞具的孔壁/孔内深度会落入环带；它们通常比外表面离相机更远，
    # 混入后会把局部平面RMSE拉到二十多毫米。按前景深度簇筛选外表面，
    # 同时保留足够带宽覆盖孔面倾斜与局部曲率。
    front_surface_z = float(np.percentile(points[:, 2], front_percentile))
    # 相机坐标Z沿视线向远处增加，因此孔口前表面必须取最小深度簇，
    # 不能用“离第20百分位绝对值相近”把更远的孔底簇带回来。
    surface_points = points[points[:, 2] <= front_surface_z + surface_band_mm]
    min_surface_points = max(80, int(len(points) * min_surface_fraction))
    if len(surface_points) < min_surface_points:
        if policy == COARSE_SURFACE_SELECTION_POLICY:
            raise ValueError(
                "孔口前表面有效点不足，拒绝把孔底/孔壁混入平面："
                f"{len(surface_points)}/{min_surface_points}"
            )
    else:
        points = surface_points
    # 粗拍姿态只需要选中孔处的局部切平面。对单孔窄环带拟合整球半径
    # 数值病态，且镀膜反光会使球心/法向跳变；不能用它直接规划TCP姿态。
    normal, plane_rmse = fit_plane(points)
    plane_point = np.median(points, axis=0)
    sphere_center = sphere_radius = None
    sphere_rmse = None
    try:
        # 仍记录球面参考，供显式三孔模式的后续坐标推算使用；失败不影响
        # 单孔粗定位和TCP垂直孔面的姿态规划。
        sphere_center, sphere_radius, sphere_rmse = fit_sphere(points)
    except ValueError:
        pass

    ray_center = np.asarray(
        ring_center_xy if ray_center_xy is None else ray_center_xy,
        dtype=np.float64,
    ).reshape(2)
    ray = camera_ray(intrinsics, ray_center, already_undistorted=ray_center_is_undistorted)
    denominator = float(normal @ ray)
    if abs(denominator) < 1e-6:
        raise ValueError("孔中心视线与局部切平面近似平行")
    scale = float(normal @ plane_point) / denominator
    if scale <= 0.0 or not math.isfinite(scale):
        raise ValueError("孔中心局部切平面交点位于相机反向")
    point = ray * scale
    if not np.isfinite(point).all() or point[2] <= 0:
        raise ValueError("孔中心反投影得到无效深度")
    result = {
        "ring_points": int(len(points)),
        "ring_points_raw": raw_ring_points,
        "front_surface_z_mm": front_surface_z,
        "plane_rmse_mm": plane_rmse,
        "plane_normal_camera": normal.tolist(),
        # 与孔中心 point_camera_mm 分开记录环带拟合平面上的代表点；
        # 当前中心仍由中心射线与该局部平面求交得到。
        "local_plane_point_camera_mm": plane_point.tolist(),
        "plane_point_camera_mm": point.tolist(),
        "surface_model": surface_model,
        "surface_selection_policy": policy,
        "ring_inner_factor": float(ring_inner_factor),
        "ring_outer_factor": float(ring_outer_factor),
        "front_surface_percentile": float(front_percentile),
        "surface_band_mm": float(surface_band_mm),
        "surface_points_selected": int(len(points)),
        "sphere_center_camera_mm": sphere_center.tolist() if sphere_center is not None else None,
        "sphere_radius_mm": sphere_radius,
        "sphere_rmse_mm": sphere_rmse,
        "ring_center_px_distorted": [ring_u, ring_v],
        "ray_center_px": ray_center.tolist(),
        "ray_center_is_undistorted": bool(ray_center_is_undistorted),
    }
    if include_points:
        result["points_camera_mm"] = np.asarray(points, dtype=np.float32)
    return point, result


def choose_boxes(image: np.ndarray, detections: list[dict[str, Any]], count: int | None = None,
                 preselected_indices: list[int] | None = None,
                 locked_indices: list[int] | None = None,
                 return_clicks: bool = False) -> list[int] | tuple[list[int], dict[int, list[float]]] | None:
    """手动选择孔；count=None 时由用户点击任意数量并按 Enter 结束。"""
    required = None if count is None else max(1, int(count))
    if required is not None and len(detections) < required:
        raise ValueError(f"当前画面仅检测到 {len(detections)} 个孔，无法选择 {required} 个")
    if not detections:
        raise ValueError("当前画面未检测到孔，无法选择")
    selected = list(dict.fromkeys(preselected_indices or []))
    selected = [index for index in selected if 0 <= index < len(detections)]
    if required is not None:
        selected = selected[:required]
    locked = set(locked_indices or []) & set(selected)
    # 未手选的锁定参考孔仍以检测中心作锚点；输出孔以用户点击的真实孔中心作锚点。
    clicks: dict[int, list[float]] = {
        index: list(map(float, detections[index]["center"])) for index in locked
    }

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        distances = [np.hypot(x - d["center"][0], y - d["center"][1]) for d in detections]
        if not distances:
            return
        index = int(np.argmin(distances))
        if index in selected:
            if index not in locked:
                selected.remove(index)  # 再点一次即可取消该孔。
                clicks.pop(index, None)
        elif required is None or len(selected) < required:
            selected.append(index)
            clicks[index] = [float(x), float(y)]

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        view = image.copy()
        for i, d in enumerate(detections):
            x1, y1, x2, y2 = map(int, d["box"])
            if i in selected:
                order = selected.index(i) + 1
                color = (0, 255, 0) if order == 1 else (255, 220, 0)
            else:
                order = None
                color = (0, 180, 255)
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
            cv2.circle(view, tuple(map(int, d["center"])), 4, color, -1)
            cv2.putText(view, f"{i}: {d['class_name']} {d['confidence']:.2f}",
                        (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if order is not None:
                label = f"{order}: reference" if i in locked or order == 1 else f"{order}: output"
                cv2.putText(view, label, (x1, min(view.shape[0] - 10, y2 + 24)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
                if i in clicks:
                    cx, cy = map(int, clicks[i])
                    cv2.drawMarker(view, (cx, cy), color, cv2.MARKER_CROSS, 16, 2)
        if required is None:
            prompt = f"click hole center, select any number: {len(selected)}; Enter=confirm; Esc=quit"
        else:
            prompt = f"click hole center, select {required}: {len(selected)}/{required}; Enter=confirm; Esc=quit"
        cv2.putText(view, prompt,
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):
            if len(selected) >= 1 and (required is None or len(selected) == required):
                if return_clicks:
                    return selected, {index: clicks.get(index, list(map(float, detections[index]["center"])))
                                      for index in selected}
                return selected
        if key == 27:
            return None


def choose_box(image: np.ndarray, detections: list[dict[str, Any]]) -> int | None:
    """保留旧调用接口，单孔选择。"""
    selected = choose_boxes(image, detections, count=1)
    return selected[0] if selected else None


def _load_intrinsics(path: Path) -> Any:
    from aubo_workbench.camera import CameraIntrinsics
    data = json.loads(path.read_text(encoding="utf-8"))
    return CameraIntrinsics(int(data["width"]), int(data["height"]), float(data["fx"]),
                            float(data["fy"]), float(data["cx"]), float(data["cy"]),
                            tuple(data.get("distortion", [])))


def fit_hole_ellipse(
    image_bgr: np.ndarray,
    detection: dict[str, Any],
    intrinsics: Any | None = None,
    expected_center_px: np.ndarray | list[float] | tuple[float, float] | None = None,
) -> dict[str, Any] | None:
    """YOLO仅给ROI；将轮廓点去畸变后再拟合椭圆。

    center_px / axes_px / angle_deg 位于去畸变像素域，用于几何计算。
    *_distorted 字段位于原始图像域，仅用于深度环带索引和显示。
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in detection["box"]]
    pad = 0.18 * max(x2 - x1, y2 - y1)
    xa, ya = max(0, int(math.floor(x1 - pad))), max(0, int(math.floor(y1 - pad)))
    xb, yb = min(w, int(math.ceil(x2 + pad))), min(h, int(math.ceil(y2 + pad)))
    if xb - xa < 24 or yb - ya < 24:
        return None

    gray = cv2.cvtColor(image_bgr[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    lo, hi = int(max(5.0, 0.66 * median)), int(min(250.0, 1.33 * median + 20.0))
    edges = cv2.Canny(gray, lo, max(lo + 10, hi))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    expected_distorted = np.asarray(
        detection["center"] if expected_center_px is None else expected_center_px,
        dtype=np.float64,
    ).reshape(2)
    box_width = float(x2 - x1)
    box_height = float(y2 - y1)
    # 密集孔阵列中，一个YOLO框的扩展ROI会同时包含相邻孔。椭圆轮廓必须
    # 仍属于该检测框附近；否则看似“残差很小”的相邻孔会被错误地拿来使用。
    # 该门槛只用于身份关联，不是降低几何质量要求。
    max_raw_center_offset_px = max(12.0, 0.45 * min(box_width, box_height))
    expected_undistorted = (
        undistort_pixels(intrinsics, expected_distorted.reshape(1, 2), pixel_output=True)[0]
        if intrinsics is not None else expected_distorted
    )
    best: dict[str, Any] | None = None

    for contour in contours:
        if len(contour) < 30:
            continue
        local_points = contour.reshape(-1, 2).astype(np.float64)
        global_distorted = local_points + np.array([xa, ya], dtype=np.float64)
        try:
            raw_ellipse = cv2.fitEllipse(global_distorted.astype(np.float32).reshape(-1, 1, 2))
        except cv2.error:
            continue
        raw_center = np.asarray(raw_ellipse[0], dtype=np.float64)
        raw_center_offset = float(np.linalg.norm(raw_center - expected_distorted))
        if raw_center_offset > max_raw_center_offset_px:
            continue
        raw_axes = np.asarray(raw_ellipse[1], dtype=np.float64)
        raw_width, raw_height = float(raw_axes[0]), float(raw_axes[1])
        raw_major, raw_minor = max(raw_width, raw_height), min(raw_width, raw_height)
        if raw_minor < 12.0 or raw_major / raw_minor > 1.60:
            continue

        undistorted_points = (
            undistort_pixels(intrinsics, global_distorted, pixel_output=True)
            if intrinsics is not None else global_distorted
        )
        try:
            (center_tuple, axes_tuple, angle) = cv2.fitEllipse(
                undistorted_points.astype(np.float32).reshape(-1, 1, 2)
            )
        except cv2.error:
            continue
        center = np.asarray(center_tuple, dtype=np.float64)
        axes = np.asarray(axes_tuple, dtype=np.float64)
        fit_width, fit_height = float(axes[0]), float(axes[1])
        major, minor = max(fit_width, fit_height), min(fit_width, fit_height)
        if minor < 12.0 or major / minor > 1.45:
            continue

        theta = math.radians(float(angle))
        c, s = math.cos(theta), math.sin(theta)
        dx = undistorted_points[:, 0] - center[0]
        dy = undistorted_points[:, 1] - center[1]
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        # 使用OpenCV返回的轴顺序和角度计算残差，避免把长短轴排序后与angle错配。
        radial = np.sqrt((local_x / (fit_width / 2.0)) ** 2 + (local_y / (fit_height / 2.0)) ** 2)
        residual = float(np.median(np.abs(radial - 1.0)) * (fit_width + fit_height) / 4.0)
        angles = np.mod(
            np.degrees(np.arctan2(local_y / (fit_height / 2.0), local_x / (fit_width / 2.0))),
            360.0,
        )
        bins = np.unique(np.floor(angles / 10.0).astype(int))
        coverage = float(len(bins) * 10.0)
        center_distance = float(np.linalg.norm(center - expected_undistorted))
        score = center_distance / max(major, 1.0) + residual / 2.0 + abs(1.0 - minor / major)

        legacy_center_undistorted = (
            undistort_pixels(intrinsics, raw_center.reshape(1, 2), pixel_output=True)[0]
            if intrinsics is not None else raw_center
        )
        item = {
            "center_px": center.tolist(),
            "axes_px": [major, minor],
            "angle_deg": float(angle),
            "center_px_distorted": raw_center.tolist(),
            "axes_px_distorted": [raw_width, raw_height],
            "axes_px_distorted_major_minor": [raw_major, raw_minor],
            "angle_deg_distorted": float(raw_ellipse[2]),
            "legacy_center_px_undistorted": legacy_center_undistorted.tolist(),
            "contour_undistortion_shift_px": (center - legacy_center_undistorted).tolist(),
            "residual_px": residual,
            "coverage_deg": coverage,
            "roundness": float(minor / major),
            "contour_points": int(len(contour)),
            "detection_center_offset_px": raw_center_offset,
            "max_detection_center_offset_px": max_raw_center_offset_px,
            "score": score,
            "center_coordinate_domain": "undistorted_pixel",
        }
        if best is None or score < best["score"]:
            best = item

    # 镀膜内壁会让Canny轮廓在上、下两段之间跳变；对于当前这类近圆孔，
    # 固定局部ROI内的霍夫圆心通常更稳定。它仍受同一身份锚点约束，绝不会
    # 在整张图中搜索到邻孔。椭圆拟合保留在上方，供非圆或倾斜明显的情况使用。
    min_radius = max(12, int(0.26 * min(box_width, box_height)))
    max_radius = max(min_radius + 2, int(0.72 * max(box_width, box_height)))
    circles = cv2.HoughCircles(
        cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(24.0, 0.70 * min(box_width, box_height)),
        param1=max(50.0, float(hi)), param2=20.0,
        minRadius=min_radius, maxRadius=max_radius,
    )
    hough_best: dict[str, Any] | None = None
    if circles is not None:
        edge_y, edge_x = np.nonzero(edges)
        raw_edge_points = np.column_stack((edge_x + xa, edge_y + ya)).astype(np.float64)
        for local_u, local_v, radius in np.asarray(circles[0], dtype=np.float64):
            raw_center = np.array([local_u + xa, local_v + ya], dtype=np.float64)
            raw_center_offset = float(np.linalg.norm(raw_center - expected_distorted))
            if raw_center_offset > max_raw_center_offset_px:
                continue
            # HoughCircles 的 dp=1.2 输出天然量化到约1.2 px，不能直接用于
            # 1 px级精定位。用其作为初值，在同一环带边缘做两次亚像素圆最小二乘。
            refined_center = raw_center.copy()
            refined_radius = float(radius)
            for _ in range(2):
                radial_distance = np.linalg.norm(raw_edge_points - refined_center, axis=1)
                annulus = np.abs(radial_distance - refined_radius) <= max(3.0, 0.08 * refined_radius)
                fit_points = raw_edge_points[annulus]
                if len(fit_points) < 30:
                    break
                local_fit = fit_points - np.array([xa, ya], dtype=np.float64)
                A = np.column_stack((2.0 * local_fit[:, 0], 2.0 * local_fit[:, 1], np.ones(len(local_fit))))
                b = np.sum(local_fit * local_fit, axis=1)
                try:
                    solution, *_ = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    break
                local_center = solution[:2]
                radius_sq = float(solution[2] + local_center @ local_center)
                if not math.isfinite(radius_sq) or radius_sq <= 0.0:
                    break
                refined_center = local_center + np.array([xa, ya], dtype=np.float64)
                refined_radius = math.sqrt(radius_sq)
            # 最小二乘初值之后再做确定性的鲁棒圆拟合。反光横线会贡献
            # 大量非圆边缘；只用与同一圆一致的径向内点重新求解亚像素圆心。
            refined_center, refined_radius = _robust_refine_circle_from_edges(
                raw_edge_points, refined_center, refined_radius,
            )
            raw_center = refined_center
            radius = refined_radius
            raw_center_offset = float(np.linalg.norm(raw_center - expected_distorted))
            if raw_center_offset > max_raw_center_offset_px:
                continue
            radial_distance = np.linalg.norm(raw_edge_points - raw_center, axis=1)
            annulus = np.abs(radial_distance - radius) <= max(2.5, 0.06 * radius)
            if not np.any(annulus):
                continue
            annulus_points = raw_edge_points[annulus]
            residual = float(np.median(np.abs(radial_distance[annulus] - radius)))
            angles = np.mod(
                np.degrees(np.arctan2(annulus_points[:, 1] - raw_center[1], annulus_points[:, 0] - raw_center[0])),
                360.0,
            )
            coverage = float(len(np.unique(np.floor(angles / 10.0).astype(int))) * 10.0)
            center = (
                undistort_pixels(intrinsics, raw_center.reshape(1, 2), pixel_output=True)[0]
                if intrinsics is not None else raw_center
            )
            # 相机畸变在本工作区较小；圆轴用于圆心定位，姿态仍来自冻结深度局部面。
            item = {
                "center_px": center.tolist(),
                "axes_px": [float(2.0 * radius), float(2.0 * radius)],
                "angle_deg": 0.0,
                "center_px_distorted": raw_center.tolist(),
                "axes_px_distorted": [float(2.0 * radius), float(2.0 * radius)],
                "axes_px_distorted_major_minor": [float(2.0 * radius), float(2.0 * radius)],
                "angle_deg_distorted": 0.0,
                "legacy_center_px_undistorted": center.tolist(),
                "contour_undistortion_shift_px": [0.0, 0.0],
                "residual_px": residual,
                "coverage_deg": coverage,
                "roundness": 1.0,
                "contour_points": int(len(annulus_points)),
                "detection_center_offset_px": raw_center_offset,
                "max_detection_center_offset_px": max_raw_center_offset_px,
                "score": raw_center_offset / max(radius, 1.0) + residual / 2.0,
                "center_coordinate_domain": "undistorted_pixel",
                "fit_method": "hough_circle",
            }
            if hough_best is None or item["score"] < hough_best["score"]:
                hough_best = item
    # 霍夫圆只有达到与正式精定位一致的严格几何质量时才优先。旧的
    # 80deg/2.5px放宽门会让反光横线污染的圆覆盖掉更可靠的轮廓椭圆。
    if (
        hough_best is not None
        and hough_best["coverage_deg"] >= 200.0
        and hough_best["residual_px"] <= 0.9
    ):
        return hough_best
    return best


def _robust_refine_circle_from_edges(
    edge_points: np.ndarray,
    initial_center: np.ndarray,
    initial_radius: float,
    *,
    iterations: int = 180,
    inlier_threshold_px: float = 1.5,
) -> tuple[np.ndarray, float]:
    """在霍夫圆附近用RANSAC径向内点抑制横线、反光和缺口。"""
    points = np.asarray(edge_points, dtype=np.float64).reshape(-1, 2)
    center0 = np.asarray(initial_center, dtype=np.float64).reshape(2)
    radius0 = float(initial_radius)
    if len(points) < 30 or not np.isfinite(center0).all() or radius0 <= 0.0:
        return center0.copy(), radius0

    radial0 = np.linalg.norm(points - center0, axis=1)
    candidate = points[np.abs(radial0 - radius0) <= max(4.0, 0.10 * radius0)]
    if len(candidate) < 30:
        return center0.copy(), radius0

    def circle_from_three(sample: np.ndarray) -> tuple[np.ndarray, float] | None:
        a = 2.0 * (sample[1:] - sample[0])
        b = np.sum(sample[1:] * sample[1:], axis=1) - float(sample[0] @ sample[0])
        if abs(float(np.linalg.det(a))) < 1.0e-6:
            return None
        try:
            center = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            return None
        radius = float(np.linalg.norm(sample[0] - center))
        if not np.isfinite(center).all() or not math.isfinite(radius) or radius <= 0.0:
            return None
        return center, radius

    rng = np.random.default_rng(0)
    best_mask: np.ndarray | None = None
    best_key: tuple[int, int, float] | None = None
    for _ in range(max(1, int(iterations))):
        sample = candidate[rng.choice(len(candidate), size=3, replace=False)]
        fitted = circle_from_three(sample)
        if fitted is None:
            continue
        center, radius = fitted
        if (
            np.linalg.norm(center - center0) > max(6.0, 0.08 * radius0)
            or abs(radius - radius0) > max(6.0, 0.12 * radius0)
        ):
            continue
        residual = np.abs(np.linalg.norm(candidate - center, axis=1) - radius)
        mask = residual <= float(inlier_threshold_px)
        if int(np.count_nonzero(mask)) < 24:
            continue
        inlier_points = candidate[mask]
        angles = np.mod(
            np.degrees(np.arctan2(
                inlier_points[:, 1] - center[1], inlier_points[:, 0] - center[0],
            )),
            360.0,
        )
        coverage_bins = int(len(np.unique(np.floor(angles / 10.0).astype(int))))
        key = (
            coverage_bins,
            int(np.count_nonzero(mask)),
            -float(np.median(residual[mask])),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_mask = mask

    if best_mask is None:
        return center0.copy(), radius0

    inliers = candidate[best_mask]
    center = center0.copy()
    radius = radius0
    for _ in range(3):
        A = np.column_stack((2.0 * inliers[:, 0], 2.0 * inliers[:, 1], np.ones(len(inliers))))
        b = np.sum(inliers * inliers, axis=1)
        try:
            solution, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            break
        center = np.asarray(solution[:2], dtype=np.float64)
        radius_sq = float(solution[2] + center @ center)
        if not math.isfinite(radius_sq) or radius_sq <= 0.0:
            return center0.copy(), radius0
        radius = math.sqrt(radius_sq)
        residual = np.abs(np.linalg.norm(candidate - center, axis=1) - radius)
        next_inliers = candidate[residual <= float(inlier_threshold_px)]
        if len(next_inliers) < 24:
            break
        inliers = next_inliers
    return center, radius
