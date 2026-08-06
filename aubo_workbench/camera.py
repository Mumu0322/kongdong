#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gemini 435Le 相机初始化、取帧、RGB/Depth/点云格式转换。

所有 pyorbbecsdk 相关的细节都封装在这里；其余模块只应该调用
`init_pipeline()` / `get_aligned_rgb_depth_xyz()` / `print_device_info()`，
不直接 import pyorbbecsdk。
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import time

import cv2
import numpy as np

from .config import CAMERA_CFG
_ORBBEC_IMPORT_ERROR: Exception | None = None
try:
    from pyorbbecsdk import *  # type: ignore  # noqa: F401,F403
except Exception as exc:  # pragma: no cover - 运行环境没有相机 SDK 时只用于提示
    _ORBBEC_IMPORT_ERROR = exc


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...] = ()

    def camera_matrix(self) -> np.ndarray:
        return np.asarray(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def as_dict(self) -> dict:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "distortion": [float(value) for value in self.distortion],
        }


@dataclass(frozen=True)
class VideoProfileInfo:
    index: int
    width: int
    height: int
    fps: int
    format: str
    stream_type: str

    def as_dict(self) -> dict:
        return {
            "index": int(self.index), "width": int(self.width), "height": int(self.height),
            "fps": int(self.fps), "format": self.format, "stream_type": self.stream_type,
        }


@dataclass
class CameraFrameBundle:
    color_bgr: np.ndarray
    depth_mm: np.ndarray
    xyz_map_mm: np.ndarray
    color_timestamp_us: int | None
    depth_timestamp_us: int | None
    color_frame_index: int | None
    depth_frame_index: int | None
    host_timestamp_ns: int
    intrinsics: CameraIntrinsics | None = None
    depth_processing: dict | None = None

    def metadata_dict(self) -> dict:
        delta = None
        if self.color_timestamp_us is not None and self.depth_timestamp_us is not None:
            delta = int(self.color_timestamp_us - self.depth_timestamp_us)
        return {
            "color_timestamp_us": self.color_timestamp_us,
            "depth_timestamp_us": self.depth_timestamp_us,
            "rgb_depth_timestamp_delta_us": delta,
            "color_frame_index": self.color_frame_index,
            "depth_frame_index": self.depth_frame_index,
            "host_timestamp_ns": int(self.host_timestamp_ns),
            "width": int(self.color_bgr.shape[1]),
            "height": int(self.color_bgr.shape[0]),
            "intrinsics": None if self.intrinsics is None else self.intrinsics.as_dict(),
            "depth_processing": self.depth_processing,
        }


@dataclass
class RgbFrameBundle:
    """RGB手眼专用帧；不获取、不对齐、不生成深度点云。"""

    color_bgr: np.ndarray
    color_timestamp_us: int | None
    color_frame_index: int | None
    host_timestamp_ns: int
    intrinsics: CameraIntrinsics | None = None

    def metadata_dict(self) -> dict:
        return {
            "color_timestamp_us": self.color_timestamp_us,
            "color_frame_index": self.color_frame_index,
            "host_timestamp_ns": int(self.host_timestamp_ns),
            "width": int(self.color_bgr.shape[1]),
            "height": int(self.color_bgr.shape[0]),
            "intrinsics": None if self.intrinsics is None else self.intrinsics.as_dict(),
            "capture_mode": "rgb_only_no_depth_or_pointcloud",
        }


@dataclass
class DepthProcessingChain:
    """对齐深度的可追溯后处理链；点云必须由同一张处理后深度生成。"""

    point_cloud_filter: object
    mode: str = "none"
    temporal_filter: object | None = None
    temporal_weight: float | None = None
    temporal_diff_scale: float | None = None

    def process_depth(self, depth_frame):
        if self.mode == "none":
            return depth_frame
        if self.mode != "temporal" or self.temporal_filter is None:
            raise RuntimeError(f"unknown or incomplete depth processing mode: {self.mode}")
        filtered = self.temporal_filter.process(depth_frame)
        if filtered is None:
            raise RuntimeError("TemporalFilter returned no frame")
        try:
            return filtered.as_depth_frame()
        except Exception:
            return filtered

    def process_point_cloud(self, depth_frame):
        return self.point_cloud_filter.process(depth_frame)

    def as_dict(self) -> dict:
        try:
            sdk_version = importlib.metadata.version("pyorbbecsdk2")
        except importlib.metadata.PackageNotFoundError:
            sdk_version = None
        return {
            "mode": self.mode,
            "order": ["align_depth_to_color"] + (["temporal"] if self.mode == "temporal" else [])
            + ["point_cloud_from_processed_depth"],
            "temporal_weight": self.temporal_weight,
            "temporal_diff_scale": self.temporal_diff_scale,
            "pyorbbecsdk2_version": sdk_version,
        }


def format_orbbec_error_hint(exc: Exception) -> str:
    text = str(exc)
    hints = [f"[ERROR] Orbbec/Gemini 相机 SDK 错误：{text}"]
    if "VendorTCPClient" in text or "192.168.1.10" in text or "port=8090" in text:
        hints.extend([
            "[ERROR] SDK 正在按以太网相机访问 192.168.1.10:8090，但该地址/端口没有响应。",
            "[CHECK] 这不是 AUBO 机械臂 192.168.50.200 的错误，而是 Gemini 435Le 相机网络错误。",
            "[CHECK] 如果相机走网口，请确认本机连接相机的网卡在 192.168.1.x/24 网段，且能 ping 通 192.168.1.10。",
            "[CHECK] 如果相机走 USB，请检查 Orbbec SDK/设备配置里是否误把相机切到了 ethernet 模式。",
        ])
    return "\n".join(hints)


def print_device_info() -> bool:
    if _ORBBEC_IMPORT_ERROR is not None:
        print("[ERROR] 未能导入 pyorbbecsdk。请在 Gemini 435Le 相机运行环境中执行本脚本。")
        print("[ERROR] import error:", _ORBBEC_IMPORT_ERROR)
        return False
    try:
        ctx = Context()
        dev_list = ctx.query_devices()
        count = dev_list.get_count()
    except Exception as exc:
        print(format_orbbec_error_hint(exc))
        return False

    print("[INFO] 检测到设备数量:", count)
    if count == 0:
        print("[ERROR] 没有检测到 Orbbec/Gemini 设备")
        return False
    for i in range(count):
        try:
            dev = dev_list[i]
            info = dev.get_device_info()
        except Exception as exc:
            print(f"[ERROR] 读取第 {i} 个 Orbbec/Gemini 设备信息失败。")
            print(format_orbbec_error_hint(exc))
            return False
        print("=" * 60)
        print("设备索引:", i)
        print("设备名称:", info.get_name())
        print("序列号:", info.get_serial_number())
        print("PID:", info.get_pid())
        print("连接类型:", info.get_connection_type())
    print("=" * 60)
    return True


def get_depth_profile(pipeline: "Pipeline"):
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profiles.get_default_video_stream_profile()
    print("[INFO] 使用 Depth Profile:", depth_profile)
    return depth_profile


def _enum_name(value) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name).removeprefix("OB_FORMAT_")
    text = str(value)
    return text.rsplit(".", 1)[-1].removeprefix("OB_FORMAT_")


def describe_video_profiles(profile_list) -> list[VideoProfileInfo]:
    result = []
    for index in range(int(profile_list.get_count())):
        profile = profile_list.get_stream_profile_by_index(index)
        try:
            video = profile.as_video_stream_profile()
        except Exception:
            video = profile
        try:
            result.append(VideoProfileInfo(
                index=index,
                width=int(video.get_width()),
                height=int(video.get_height()),
                fps=int(video.get_fps()),
                format=_enum_name(video.get_format()),
                stream_type=_enum_name(video.get_type()),
            ))
        except Exception as exc:
            print(f"[WARN] 无法读取 stream profile index={index}: {exc}")
    return result


def choose_color_profile_index(profiles: list[VideoProfileInfo]) -> int:
    """纯函数：显式按配置选择彩色流，便于离线单测。"""
    if not profiles:
        raise RuntimeError("相机没有可用的彩色流 profile")
    preferred_formats = [str(x).upper().removeprefix("OB_FORMAT_") for x in CAMERA_CFG.preferred_color_formats]

    def format_rank(name: str) -> int:
        normalized = str(name).upper().removeprefix("OB_FORMAT_")
        try:
            return len(preferred_formats) - preferred_formats.index(normalized)
        except ValueError:
            return 0

    def key(info: VideoProfileInfo):
        width_match = CAMERA_CFG.preferred_color_width <= 0 or info.width == CAMERA_CFG.preferred_color_width
        height_match = CAMERA_CFG.preferred_color_height <= 0 or info.height == CAMERA_CFG.preferred_color_height
        dimension_match = int(width_match and height_match)
        area = int(info.width * info.height)
        fps_distance = abs(int(info.fps) - int(CAMERA_CFG.preferred_color_fps))
        return dimension_match, area, format_rank(info.format), -fps_distance, info.fps

    return int(max(profiles, key=key).index)


def print_video_profiles(title: str, profiles: list[VideoProfileInfo], selected_index: int | None = None) -> None:
    print(f"[INFO] {title} profiles ({len(profiles)}):")
    for info in profiles:
        marker = "*" if selected_index == info.index else " "
        print(
            f"[INFO] {marker} index={info.index:2d} {info.width}x{info.height} "
            f"@ {info.fps} FPS format={info.format} type={info.stream_type}"
        )


def get_color_profile(pipeline: "Pipeline"):
    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    descriptions = describe_video_profiles(color_profiles)
    selected_index = choose_color_profile_index(descriptions)
    print_video_profiles("Color", descriptions, selected_index)
    color_profile = color_profiles.get_stream_profile_by_index(selected_index)
    print(f"[INFO] 显式选择 Color Profile index={selected_index}:", color_profile)
    return color_profile


def get_camera_intrinsics(pipeline) -> CameraIntrinsics | None:
    try:
        camera_param = pipeline.get_camera_param()
        intr = camera_param.rgb_intrinsic
        distortion = camera_param.rgb_distortion
        coeffs = (
            distortion.k1, distortion.k2, distortion.p1, distortion.p2,
            distortion.k3, distortion.k4, distortion.k5, distortion.k6,
        )
        return CameraIntrinsics(
            width=int(intr.width), height=int(intr.height),
            fx=float(intr.fx), fy=float(intr.fy), cx=float(intr.cx), cy=float(intr.cy),
            distortion=tuple(float(x) for x in coeffs),
        )
    except Exception as exc:
        print("[WARN] 读取 RGB 内参失败:", exc)
        return None


def get_device_identity(pipeline) -> dict:
    try:
        info = pipeline.get_device().get_device_info()
        result = {
            "name": str(info.get_name()),
            "serial_number": str(info.get_serial_number()),
            "pid": str(info.get_pid()),
            "connection_type": str(info.get_connection_type()),
        }
        for key, method_name in (
            ("firmware_version", "get_firmware_version"),
            ("hardware_version", "get_hardware_version"),
            ("supported_min_sdk_version", "get_supported_min_sdk_version"),
            ("device_ip_address", "get_device_ip_address"),
        ):
            try:
                result[key] = str(getattr(info, method_name)())
            except Exception:
                result[key] = None
        return result
    except Exception as exc:
        return {"error": str(exc)}


def configure_color_controls(
    pipeline,
    lock_auto_exposure: bool,
    exposure: int | None = None,
    gain: int | None = None,
) -> dict:
    """读取/设置曝光状态。所有不支持项都写入结果，不把异常吞成“已成功”。"""
    result: dict = {"requested": {
        "lock_auto_exposure": bool(lock_auto_exposure), "exposure": exposure, "gain": gain,
    }, "errors": []}
    if _ORBBEC_IMPORT_ERROR is not None:
        result["errors"].append(f"pyorbbecsdk import failed: {_ORBBEC_IMPORT_ERROR}")
        result["ok"] = False
        return result
    try:
        device = pipeline.get_device()
        prop_auto = OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL
        prop_exposure = OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT
        prop_gain = OBPropertyID.OB_PROP_COLOR_GAIN_INT
        if lock_auto_exposure:
            device.set_bool_property(prop_auto, False)
        if exposure is not None:
            device.set_int_property(prop_exposure, int(exposure))
        if gain is not None:
            device.set_int_property(prop_gain, int(gain))
        for name, prop, getter in (
            ("auto_exposure", prop_auto, device.get_bool_property),
            ("exposure", prop_exposure, device.get_int_property),
            ("gain", prop_gain, device.get_int_property),
        ):
            try:
                result[name] = getter(prop)
            except Exception as exc:
                result["errors"].append(f"read {name}: {exc}")
    except Exception as exc:
        result["errors"].append(str(exc))
    result["ok"] = not result["errors"]
    return result


def color_frame_to_bgr(color_frame) -> np.ndarray | None:
    width = color_frame.get_width()
    height = color_frame.get_height()
    fmt = color_frame.get_format()
    data = np.asanyarray(color_frame.get_data())

    if fmt == OBFormat.RGB:
        image = data.reshape((height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return data.reshape((height, width, 3))
    if fmt == OBFormat.BGRA:
        image = data.reshape((height, width, 4))
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.I420:
        image = data.reshape((height * 3 // 2, width))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_I420)
    if fmt == OBFormat.YUYV:
        image = data.reshape((height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    if fmt == OBFormat.UYVY:
        image = data.reshape((height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    print("[WARN] 不支持的彩色格式:", fmt)
    return None


def depth_frame_to_mm(depth_frame) -> np.ndarray:
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()
    depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
    return (depth_raw.astype(np.float32) * scale).astype(np.float32)


def depth_to_vis(depth_mm: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_mm, dtype=np.float32)
    valid = depth.copy()
    valid[(valid < CAMERA_CFG.depth_vis_min_mm) | (valid > CAMERA_CFG.depth_vis_max_mm)] = 0
    depth_vis = cv2.normalize(valid, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)


def pointcloud_frame_to_xyz_map(point_cloud_frame, width: int, height: int) -> np.ndarray:
    data = np.frombuffer(point_cloud_frame.get_data(), dtype=np.float32)
    if data.size % 3 != 0:
        raise ValueError(f"点云数据长度异常，无法按 x,y,z 解析，data.size={data.size}")
    points = data.reshape(-1, 3)
    expected = width * height
    if points.shape[0] != expected:
        print(
            "[WARN] 点云数量和图像尺寸不一致:",
            "points=", points.shape[0], "image=", width, "x", height, "expected=", expected,
        )
        valid_count = min(points.shape[0], expected)
        xyz = np.zeros((height * width, 3), dtype=np.float32)
        xyz[:valid_count, :] = points[:valid_count]
        return xyz.reshape((height, width, 3))
    return points.reshape((height, width, 3))


def init_pipeline(depth_filter_mode: str = "none"):
    """启动pipeline，返回(pipeline, align_filter, depth_processing_chain)。"""
    if depth_filter_mode not in {"none", "temporal"}:
        raise ValueError(f"unsupported depth_filter_mode: {depth_filter_mode}")
    pipeline = Pipeline()
    config = Config()
    config.enable_stream(get_depth_profile(pipeline))
    config.enable_stream(get_color_profile(pipeline))
    try:
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        print("[INFO] 已设置 FULL_FRAME_REQUIRE")
    except Exception as exc:
        print("[WARN] 设置 FULL_FRAME_REQUIRE 失败，继续运行:", exc)
    pipeline.start(config)
    print("[INFO] pipeline 启动成功")

    if CAMERA_CFG.lock_color_auto_exposure or CAMERA_CFG.color_exposure is not None or CAMERA_CFG.color_gain is not None:
        controls = configure_color_controls(
            pipeline,
            lock_auto_exposure=CAMERA_CFG.lock_color_auto_exposure,
            exposure=CAMERA_CFG.color_exposure,
            gain=CAMERA_CFG.color_gain,
        )
        print("[INFO] Color controls:", controls)

    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    print("[INFO] 对齐模式: Depth -> RGB / Color")

    point_cloud_filter = PointCloudFilter()
    point_cloud_filter.set_create_point_format(OBFormat.POINT)
    try:
        point_cloud_filter.set_camera_param(pipeline.get_camera_param())
        print("[INFO] 已为 PointCloudFilter 设置 camera_param")
    except Exception as exc:
        print("[WARN] 设置 PointCloudFilter camera_param 失败，继续运行:", exc)
    temporal_filter = None
    try:
        if depth_filter_mode == "temporal":
            # 当前Windows绑定要求在Pipeline启动后构造，否则原生层会触发访问冲突。
            temporal_filter = TemporalFilter()
            temporal_filter.set_weight(float(CAMERA_CFG.depth_temporal_weight))
            temporal_filter.set_diff_scale(float(CAMERA_CFG.depth_temporal_diff_scale))
            print(
                "[INFO] Depth TemporalFilter:",
                f"weight={CAMERA_CFG.depth_temporal_weight}",
                f"diff_scale={CAMERA_CFG.depth_temporal_diff_scale}",
            )
    except Exception:
        pipeline.stop()
        raise
    chain = DepthProcessingChain(
        point_cloud_filter=point_cloud_filter,
        mode=depth_filter_mode,
        temporal_filter=temporal_filter,
        temporal_weight=(
            float(CAMERA_CFG.depth_temporal_weight) if temporal_filter is not None else None
        ),
        temporal_diff_scale=(
            float(CAMERA_CFG.depth_temporal_diff_scale) if temporal_filter is not None else None
        ),
    )
    return pipeline, align_filter, chain


def init_rgb_handeye_pipeline():
    """启动仅彩色流的手眼pipeline；正式RGB-PnP链路不依赖Depth设备帧。"""
    pipeline = Pipeline()
    config = Config()
    config.enable_stream(get_color_profile(pipeline))
    pipeline.start(config)
    print("[INFO] RGB-PnP手眼pipeline启动成功：仅启用Color流，不启用Depth/PointCloud")
    if CAMERA_CFG.lock_color_auto_exposure or CAMERA_CFG.color_exposure is not None or CAMERA_CFG.color_gain is not None:
        controls = configure_color_controls(
            pipeline,
            lock_auto_exposure=CAMERA_CFG.lock_color_auto_exposure,
            exposure=CAMERA_CFG.color_exposure,
            gain=CAMERA_CFG.color_gain,
        )
        print("[INFO] Color controls:", controls)
    return pipeline


def _frame_optional_int(frame, method_name: str) -> int | None:
    try:
        return int(getattr(frame, method_name)())
    except Exception:
        return None


def get_aligned_frame_bundle(pipeline, align_filter, depth_processor) -> CameraFrameBundle | None:
    """取一帧带时间戳/内参的对齐 RGB、Depth 和点云；失败返回 None。"""
    frames = pipeline.wait_for_frames(3000)
    if frames is None:
        return None
    # 尽量靠近取帧完成时记录主机时钟；不能在点云处理结束后才记时间。
    host_timestamp_ns = time.time_ns()
    aligned_frames = align_filter.process(frames)
    if aligned_frames is None:
        return None
    aligned_frames = aligned_frames.as_frame_set()
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if depth_frame is None or color_frame is None:
        return None

    color_bgr = color_frame_to_bgr(color_frame)
    if color_bgr is None:
        return None
    try:
        processed_depth_frame = depth_processor.process_depth(depth_frame)
    except RuntimeError as exc:
        print("[WARN] 深度后处理失败:", exc)
        return None
    depth_mm = depth_frame_to_mm(processed_depth_frame)
    h, w = depth_mm.shape[:2]
    if color_bgr.shape[:2] != (h, w):
        print(f"[WARN] 对齐后尺寸不一致: RGB={color_bgr.shape[1]}x{color_bgr.shape[0]}, Depth={w}x{h}")
        return None

    pc_frame = depth_processor.process_point_cloud(processed_depth_frame)
    if pc_frame is None:
        print("[WARN] point_cloud_frame 为空")
        return None
    xyz_map = pointcloud_frame_to_xyz_map(pc_frame, w, h)
    intrinsics = get_camera_intrinsics(pipeline)
    if intrinsics is not None and (intrinsics.width, intrinsics.height) != (w, h):
        print(
            f"[WARN] RGB 内参尺寸与对齐帧不一致: K={intrinsics.width}x{intrinsics.height}, frame={w}x{h}"
        )
    return CameraFrameBundle(
        color_bgr=color_bgr,
        depth_mm=depth_mm,
        xyz_map_mm=xyz_map,
        color_timestamp_us=_frame_optional_int(color_frame, "get_timestamp_us"),
        depth_timestamp_us=_frame_optional_int(depth_frame, "get_timestamp_us"),
        color_frame_index=_frame_optional_int(color_frame, "get_index"),
        depth_frame_index=_frame_optional_int(depth_frame, "get_index"),
        host_timestamp_ns=host_timestamp_ns,
        intrinsics=intrinsics,
        depth_processing=depth_processor.as_dict(),
    )


def get_rgb_frame_bundle(pipeline) -> RgbFrameBundle | None:
    """取得一张RGB手眼帧；任何Depth/PointCloud状态都不会影响此函数。"""
    frames = pipeline.wait_for_frames(3000)
    if frames is None:
        return None
    host_timestamp_ns = time.time_ns()
    try:
        frame_set = frames.as_frame_set()
    except Exception:
        frame_set = frames
    color_frame = frame_set.get_color_frame()
    if color_frame is None:
        return None
    color_bgr = color_frame_to_bgr(color_frame)
    if color_bgr is None:
        return None
    intrinsics = get_camera_intrinsics(pipeline)
    if intrinsics is not None and (
        intrinsics.width != color_bgr.shape[1] or intrinsics.height != color_bgr.shape[0]
    ):
        print(
            f"[WARN] RGB内参尺寸与彩色帧不一致: K={intrinsics.width}x{intrinsics.height}, "
            f"frame={color_bgr.shape[1]}x{color_bgr.shape[0]}"
        )
    return RgbFrameBundle(
        color_bgr=color_bgr,
        color_timestamp_us=_frame_optional_int(color_frame, "get_timestamp_us"),
        color_frame_index=_frame_optional_int(color_frame, "get_index"),
        host_timestamp_ns=host_timestamp_ns,
        intrinsics=intrinsics,
    )


def get_aligned_rgb_depth_xyz(pipeline, align_filter, point_cloud_filter):
    """兼容旧接口：取 `(color_bgr, depth_mm, xyz_map)`；失败返回 None。"""
    bundle = get_aligned_frame_bundle(pipeline, align_filter, point_cloud_filter)
    if bundle is None:
        return None
    return bundle.color_bgr, bundle.depth_mm, bundle.xyz_map_mm
