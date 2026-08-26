#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""料框镜片实时检测前端。

功能：
    * 采集 Gemini 435Le 左/右 IR 黑白相机画面；
    * 实时显示 IR 自动曝光、曝光时间、增益、亮度和激光状态；
    * 使用同目录下的 best.pt 实时推理并叠加检测结果；
    * 支持切换左右 IR、自动/手动曝光、增益、置信度和激光开关；
    * 支持保存当前原图与标注图。

运行环境：项目已有的 lip_env310（Python 3.10 + pyorbbecsdk + ultralytics）。
本文件是独立前端，不修改“料框镜片检测.py”。
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, ttk


PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "best.pt"
CAPTURE_DIR = SCRIPT_DIR / "realtime_captures"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from pyorbbecsdk import (  # type: ignore
        OBFormat,
        OBFrameType,
        OBPropertyID,
        OBSensorType,
        Config,
        Pipeline,
    )
    ORBBEC_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - 在无SDK环境下由界面提示
    ORBBEC_IMPORT_ERROR = exc


@dataclass
class FramePacket:
    """从采集/推理线程传给 Tk 主线程的一帧结果。"""

    raw_bgr: np.ndarray
    annotated_bgr: np.ndarray
    detections: list[dict[str, Any]]
    camera_status: dict[str, Any]
    inference_ms: float
    loop_fps: float
    frame_index: int | None


def _frame_int(frame: Any, method_name: str) -> int | None:
    try:
        return int(getattr(frame, method_name)())
    except Exception:
        return None


def _range_value(value: Any, name: str, default: int) -> int:
    try:
        return int(getattr(value, name))
    except Exception:
        return default


def _frame_to_gray(frame: Any) -> np.ndarray | None:
    """把 Orbbec 的 IR 帧转换为 8-bit 灰度图。"""
    if frame is None:
        return None

    video = frame.as_video_frame()
    width = int(video.get_width())
    height = int(video.get_height())
    fmt = video.get_format()
    data = np.asanyarray(video.get_data())

    if fmt == OBFormat.Y8:
        return np.resize(data, (height, width)).astype(np.uint8)
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

    # 兼容可能出现的 16-bit IR 格式，压到 8-bit 仅用于显示/模型输入。
    raw = np.frombuffer(data, dtype=np.uint16)
    raw = np.resize(raw, (height, width))
    return cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


class OrbbecIRCamera:
    """Gemini 435Le 单路 IR 相机封装。"""

    def __init__(self, side: str = "left") -> None:
        if ORBBEC_IMPORT_ERROR is not None:
            raise RuntimeError(f"无法导入 pyorbbecsdk: {ORBBEC_IMPORT_ERROR}")
        if side not in {"left", "right"}:
            raise ValueError(f"不支持的 IR 相机: {side}")

        self.side = side
        self.pipeline: Any | None = None
        self.device: Any | None = None
        self.profile: Any | None = None
        self._lock = threading.RLock()
        self._stopped = False

        self._laser_id = OBPropertyID.OB_PROP_LASER_BOOL
        self._auto_exposure_id = OBPropertyID.OB_PROP_IR_AUTO_EXPOSURE_BOOL
        self._exposure_id = OBPropertyID.OB_PROP_IR_EXPOSURE_INT
        self._gain_id = OBPropertyID.OB_PROP_IR_GAIN_INT
        self._brightness_id = OBPropertyID.OB_PROP_IR_BRIGHTNESS_INT

    def open(self) -> dict[str, Any]:
        with self._lock:
            self.pipeline = Pipeline()
            sensor_type = (
                OBSensorType.LEFT_IR_SENSOR
                if self.side == "left"
                else OBSensorType.RIGHT_IR_SENSOR
            )
            profiles = self.pipeline.get_stream_profile_list(sensor_type)
            if int(profiles.get_count()) <= 0:
                raise RuntimeError(f"没有找到 {self.side} IR 视频 profile")

            # profile 0 是当前设备的 1280x800@10 Y8，优先保证与测试结果一致。
            self.profile = profiles.get_stream_profile_by_index(0)
            config = Config()
            config.enable_stream(self.profile)
            self.pipeline.start(config)
            self.device = self.pipeline.get_device()
            self._stopped = False

            # 默认关闭红外投射器，避免黑白图出现激光点阵。
            self.set_laser(False)
            return self.status()

    def close(self) -> None:
        with self._lock:
            pipeline = self.pipeline
            self.pipeline = None
            self.device = None
            self.profile = None
            self._stopped = True
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass

    def read(self, timeout_ms: int = 1000) -> tuple[np.ndarray, int | None] | None:
        with self._lock:
            if self.pipeline is None or self._stopped:
                return None
            frames = self.pipeline.wait_for_frames(timeout_ms)
            if frames is None:
                return None
            frame_type = (
                OBFrameType.LEFT_IR_FRAME
                if self.side == "left"
                else OBFrameType.RIGHT_IR_FRAME
            )
            frame = frames.get_frame(frame_type)
            gray = _frame_to_gray(frame)
            if gray is None:
                return None
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return bgr, _frame_int(frame, "get_index")

    def _require_device(self) -> Any:
        if self.device is None or self._stopped:
            raise RuntimeError("相机尚未启动")
        return self.device

    def status(self) -> dict[str, Any]:
        with self._lock:
            device = self._require_device()
            result: dict[str, Any] = {"side": self.side}
            readers = (
                ("auto_exposure", self._auto_exposure_id, device.get_bool_property),
                ("exposure", self._exposure_id, device.get_int_property),
                ("gain", self._gain_id, device.get_int_property),
                ("brightness", self._brightness_id, device.get_int_property),
                ("laser", self._laser_id, device.get_bool_property),
            )
            for name, prop, reader in readers:
                try:
                    value = reader(prop)
                    result[name] = bool(value) if name in {"auto_exposure", "laser"} else int(value)
                except Exception as exc:
                    result[name] = None
                    result[f"{name}_error"] = str(exc)

            try:
                result["exposure_range"] = self._int_range(self._exposure_id)
            except Exception:
                result["exposure_range"] = None
            try:
                result["gain_range"] = self._int_range(self._gain_id)
            except Exception:
                result["gain_range"] = None
            return result

    def _int_range(self, prop: Any) -> dict[str, int]:
        rng = self._require_device().get_int_property_range(prop)
        return {
            "min": _range_value(rng, "min", 0),
            "max": _range_value(rng, "max", 100000),
            "step": max(1, _range_value(rng, "step", 1)),
        }

    def set_laser(self, enabled: bool) -> None:
        with self._lock:
            self._require_device().set_bool_property(self._laser_id, bool(enabled))

    def set_auto_exposure(self, enabled: bool) -> None:
        with self._lock:
            self._require_device().set_bool_property(self._auto_exposure_id, bool(enabled))

    def set_exposure(self, value: int) -> None:
        with self._lock:
            self._require_device().set_int_property(self._exposure_id, int(value))

    def set_gain(self, value: int) -> None:
        with self._lock:
            self._require_device().set_int_property(self._gain_id, int(value))


class RealtimeDetectorApp:
    """Tkinter 主界面；相机和 YOLO 推理始终运行在后台线程。"""

    def __init__(self, root: tk.Tk, model_path: Path) -> None:
        self.root = root
        self.model_path = model_path
        self.root.title("料框镜片实时检测")
        self.root.geometry("1500x900")
        self.root.minsize(1100, 700)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.camera: OrbbecIRCamera | None = None
        self.model: Any | None = None
        self.frame_queue: queue.Queue[FramePacket] = queue.Queue(maxsize=2)
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.error_queue: queue.Queue[str] = queue.Queue()
        self.worker_finished_event = threading.Event()
        self.last_packet: FramePacket | None = None
        self.closing = False
        self._settings_lock = threading.Lock()
        self.conf_threshold = 0.15
        self.selected_side = "left"
        self.manual_exposure_target: int | None = None
        self.manual_gain_target: int | None = None

        self.side_var = tk.StringVar(value="左 IR")
        self.auto_exposure_var = tk.BooleanVar(value=True)
        self.laser_var = tk.BooleanVar(value=False)
        self.conf_var = tk.StringVar(value="0.15")
        self.manual_exposure_var = tk.StringVar(value="")
        self.manual_gain_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="未启动")
        self.device_var = tk.StringVar(value="设备：未连接")
        self.exposure_read_var = tk.StringVar(value="曝光：--")
        self.gain_read_var = tk.StringVar(value="增益：--")
        self.brightness_read_var = tk.StringVar(value="亮度：--")
        self.auto_read_var = tk.StringVar(value="自动曝光：--")
        self.laser_read_var = tk.StringVar(value="激光：--")
        self.performance_var = tk.StringVar(value="帧率：-- | 推理：--")
        self.detection_count_var = tk.StringVar(value="检测数量：--")

        self._build_ui()
        self.root.after(50, self._pump_results)

    @staticmethod
    def _side_key(label: str) -> str:
        return "right" if "右" in label else "left"

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        # 第0行：启动/参数栏；第1行：曝光控制；第2行：实时画面主体。
        self.root.rowconfigure(2, weight=1)

        toolbar = ttk.Frame(self.root, padding=(8, 8, 8, 4))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(10, weight=1)

        ttk.Label(toolbar, text="相机").grid(row=0, column=0, padx=(0, 4))
        self.side_combo = ttk.Combobox(
            toolbar, textvariable=self.side_var, values=("左 IR", "右 IR"),
            state="readonly", width=8,
        )
        self.side_combo.grid(row=0, column=1, padx=(0, 10))

        self.start_button = ttk.Button(toolbar, text="启动实时检测", command=self._start)
        self.start_button.grid(row=0, column=2, padx=(0, 5))
        self.stop_button = ttk.Button(toolbar, text="停止", command=self._stop, state="disabled")
        self.stop_button.grid(row=0, column=3, padx=(0, 10))

        ttk.Checkbutton(
            toolbar, text="自动曝光", variable=self.auto_exposure_var,
            command=self._on_auto_exposure,
        ).grid(row=0, column=4, padx=(0, 8))
        ttk.Checkbutton(
            toolbar, text="激光", variable=self.laser_var,
            command=self._on_laser,
        ).grid(row=0, column=5, padx=(0, 10))

        ttk.Label(toolbar, text="置信度").grid(row=0, column=6, padx=(0, 4))
        ttk.Entry(toolbar, textvariable=self.conf_var, width=7).grid(row=0, column=7, padx=(0, 4))
        ttk.Button(toolbar, text="应用", command=self._apply_confidence).grid(row=0, column=8, padx=(0, 10))
        ttk.Button(toolbar, text="保存当前帧", command=self._save_current).grid(row=0, column=9, padx=(0, 10))
        ttk.Label(toolbar, textvariable=self.status_var, foreground="#555555").grid(
            row=0, column=10, sticky="e",
        )

        settings = ttk.LabelFrame(self.root, text="曝光 / 增益控制", padding=6)
        settings.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))
        settings.columnconfigure(8, weight=1)
        ttk.Label(settings, text="手动曝光(μs)").grid(row=0, column=0, padx=(0, 4))
        self.exposure_entry = ttk.Entry(settings, textvariable=self.manual_exposure_var, width=10)
        self.exposure_entry.grid(row=0, column=1, padx=(0, 4))
        ttk.Button(settings, text="设置曝光", command=self._apply_exposure).grid(row=0, column=2, padx=(0, 12))
        ttk.Label(settings, text="手动增益").grid(row=0, column=3, padx=(0, 4))
        self.gain_entry = ttk.Entry(settings, textvariable=self.manual_gain_var, width=8)
        self.gain_entry.grid(row=0, column=4, padx=(0, 4))
        ttk.Button(settings, text="设置增益", command=self._apply_gain).grid(row=0, column=5, padx=(0, 12))
        ttk.Label(settings, textvariable=self.exposure_read_var).grid(row=0, column=6, padx=(0, 10))
        ttk.Label(settings, textvariable=self.gain_read_var).grid(row=0, column=7, padx=(0, 10))
        ttk.Label(settings, textvariable=self.brightness_read_var).grid(row=0, column=8, sticky="w")

        body = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        video_box = ttk.LabelFrame(body, text="实时画面（黑白 IR + YOLO 分割结果）", padding=5)
        video_box.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        video_box.columnconfigure(0, weight=1)
        video_box.rowconfigure(0, weight=1)
        self.video_label = tk.Label(video_box, text="点击“启动实时检测”", bg="#101010", fg="white")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(body)
        side.grid(row=0, column=1, sticky="nsew")
        side.configure(width=350)
        side.grid_propagate(False)

        info = ttk.LabelFrame(side, text="实时状态", padding=8)
        info.pack(fill="x", pady=(0, 8))
        for variable in (
            self.device_var, self.exposure_read_var, self.gain_read_var,
            self.brightness_read_var, self.auto_read_var, self.laser_read_var,
            self.performance_var, self.detection_count_var,
        ):
            ttk.Label(info, textvariable=variable).pack(anchor="w", pady=2)

        det_box = ttk.LabelFrame(side, text="检测目标", padding=5)
        det_box.pack(fill="both", expand=True, pady=(0, 8))
        det_box.rowconfigure(0, weight=1)
        det_box.columnconfigure(0, weight=1)
        columns = ("class", "conf", "center", "box")
        self.detection_tree = ttk.Treeview(det_box, columns=columns, show="headings", height=15)
        headings = {"class": "类别", "conf": "置信度", "center": "中心", "box": "框"}
        widths = {"class": 75, "conf": 65, "center": 100, "box": 150}
        for name in columns:
            self.detection_tree.heading(name, text=headings[name])
            self.detection_tree.column(name, width=widths[name], anchor="center")
        scroll = ttk.Scrollbar(det_box, orient="vertical", command=self.detection_tree.yview)
        self.detection_tree.configure(yscrollcommand=scroll.set)
        self.detection_tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        log_box = ttk.LabelFrame(side, text="运行日志", padding=5)
        log_box.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_box, height=7, width=42, state="disabled", wrap="word")
        log_scroll = ttk.Scrollbar(log_box, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

    def _append_log(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{stamp}] {text}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _publish_log(self, text: str) -> None:
        self.log_queue.put(text)

    def _start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        if not self.model_path.exists():
            messagebox.showerror("模型不存在", f"找不到模型：\n{self.model_path}")
            return
        try:
            self._apply_confidence()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        self.stop_event.clear()
        self.worker_finished_event.clear()
        # 在 Tk 主线程中先取出字符串，后台线程不直接访问 Tk Variable。
        self.selected_side = self._side_key(self.side_var.get())
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.side_combo.configure(state="disabled")
        self.status_var.set("正在启动相机和模型…")
        self._append_log(f"启动 {self.side_var.get()}，模型：{self.model_path.name}")
        self.worker = threading.Thread(target=self._worker_main, daemon=True)
        self.worker.start()

    def _stop(self) -> None:
        self.stop_event.set()
        camera = self.camera
        if camera is not None:
            camera.close()
        self.status_var.set("正在停止…")
        self.stop_button.configure(state="disabled")

    def _worker_main(self) -> None:
        camera: OrbbecIRCamera | None = None
        try:
            import torch
            from ultralytics import YOLO

            device = 0 if torch.cuda.is_available() else "cpu"
            self._publish_log(f"加载模型，设备：{device}")
            model = YOLO(str(self.model_path))
            self.model = model

            camera = OrbbecIRCamera(self.selected_side)
            self.camera = camera
            status = camera.open()
            self._publish_log(
                f"相机已启动：{status.get('side')} IR，默认激光关闭，"
                f"曝光={status.get('exposure')}，增益={status.get('gain')}"
            )
            self._publish_log(f"模型类别：{getattr(model, 'names', {})}")

            last_loop = time.perf_counter()
            smooth_fps = 0.0
            frame_no = 0
            latest_status = status
            while not self.stop_event.is_set():
                captured = camera.read(timeout_ms=1000)
                if captured is None:
                    continue
                raw_bgr, frame_index = captured
                started = time.perf_counter()
                with self._settings_lock:
                    conf = float(self.conf_threshold)
                results = model.predict(
                    source=raw_bgr,
                    imgsz=1280,
                    conf=conf,
                    device=device,
                    save=False,
                    verbose=False,
                )
                result = results[0]
                annotated = result.plot()
                inference_ms = float((time.perf_counter() - started) * 1000.0)

                now = time.perf_counter()
                instant_fps = 1.0 / max(1e-6, now - last_loop)
                last_loop = now
                smooth_fps = instant_fps if smooth_fps <= 0 else smooth_fps * 0.85 + instant_fps * 0.15

                detections: list[dict[str, Any]] = []
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    xywh = boxes.xywh.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy().astype(int)
                    names = getattr(model, "names", {})
                    for box, center_box, score, class_id in zip(xyxy, xywh, confs, classes):
                        detections.append({
                            "class": str(names.get(int(class_id), class_id)),
                            "class_id": int(class_id),
                            "conf": float(score),
                            "center": (int(center_box[0]), int(center_box[1])),
                            "box": tuple(int(value) for value in box),
                        })

                frame_no += 1
                if frame_no == 1 or frame_no % 5 == 0:
                    try:
                        latest_status = camera.status()
                        with self._settings_lock:
                            target_exposure = self.manual_exposure_target
                            target_gain = self.manual_gain_target
                        # 某些固件在流开始或属性刷新后会把曝光写回默认值，
                        # 手动模式下持续校验并恢复目标值，避免画面自动回调。
                        if target_exposure is not None or target_gain is not None:
                            if latest_status.get("auto_exposure"):
                                camera.set_auto_exposure(False)
                            if target_exposure is not None and latest_status.get("exposure") != target_exposure:
                                camera.set_exposure(target_exposure)
                            if target_gain is not None and latest_status.get("gain") != target_gain:
                                camera.set_gain(target_gain)
                            latest_status = camera.status()
                    except Exception as exc:
                        self._publish_log(f"读取曝光状态失败：{exc}")

                self._publish_packet(FramePacket(
                    raw_bgr=raw_bgr,
                    annotated_bgr=annotated,
                    detections=detections,
                    camera_status=latest_status,
                    inference_ms=inference_ms,
                    loop_fps=smooth_fps,
                    frame_index=frame_index,
                ))
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            self._publish_log(f"后台线程异常：{type(exc).__name__}: {exc}")
            self._publish_log(traceback.format_exc())
            self.error_queue.put(error_text)
        finally:
            if camera is not None:
                camera.close()
            self.camera = None
            self.model = None
            self._publish_log("后台采集线程已停止")
            # 不从后台线程直接调用 Tk；由主线程的 _pump_results 轮询该事件。
            self.worker_finished_event.set()

    def _publish_packet(self, packet: FramePacket) -> None:
        try:
            while True:
                self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frame_queue.put_nowait(packet)
        except queue.Full:
            pass

    def _worker_finished(self) -> None:
        self.worker_finished_event.clear()
        self.worker = None
        if self.closing:
            self.root.destroy()
            return
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.side_combo.configure(state="readonly")
        if self.stop_event.is_set():
            self.status_var.set("已停止")
        else:
            self.status_var.set("异常停止，请查看日志")

    def _pump_results(self) -> None:
        while True:
            try:
                packet = self.frame_queue.get_nowait()
            except queue.Empty:
                break
            self.last_packet = packet
            self._show_packet(packet)

        while True:
            try:
                log = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(log.rstrip())

        while True:
            try:
                error_text = self.error_queue.get_nowait()
            except queue.Empty:
                break
            self.status_var.set("启动失败，请检查运行环境")
            if not self.closing:
                messagebox.showerror("实时检测启动失败", error_text)

        if self.worker is not None and self.worker_finished_event.is_set():
            self._worker_finished()
        if not self.closing:
            self.root.after(50, self._pump_results)
        elif self.worker is not None:
            # 关闭窗口时仍需继续轮询，等后台线程安全退出后再销毁 Tk。
            self.root.after(50, self._pump_results)

    def _show_packet(self, packet: FramePacket) -> None:
        image = cv2.cvtColor(packet.annotated_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(image)
        max_width = max(480, self.video_label.winfo_width() - 10)
        max_height = max(320, self.video_label.winfo_height() - 10)
        pil.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo

        status = packet.camera_status
        side = "左 IR" if status.get("side") == "left" else "右 IR"
        self.device_var.set(f"设备：Gemini 435Le | {side} | 1280×800 Y8")
        self.exposure_read_var.set(self._format_status("曝光", status.get("exposure"), " μs"))
        self.gain_read_var.set(self._format_status("增益", status.get("gain"), ""))
        self.brightness_read_var.set(self._format_status("亮度", status.get("brightness"), ""))
        self.auto_read_var.set(f"自动曝光：{'开' if status.get('auto_exposure') else '关'}")
        self.laser_read_var.set(f"激光：{'开' if status.get('laser') else '关'}")
        self.performance_var.set(
            f"帧率：{packet.loop_fps:.1f} FPS | 推理：{packet.inference_ms:.1f} ms"
        )
        self.detection_count_var.set(f"检测数量：{len(packet.detections)}")
        self.auto_exposure_var.set(bool(status.get("auto_exposure", self.auto_exposure_var.get())))
        self.laser_var.set(bool(status.get("laser", self.laser_var.get())))
        # 输入框是用户编辑区，不能每帧用相机读数覆盖；启动时为空才填充一次。
        if status.get("exposure") is not None and not self.manual_exposure_var.get().strip():
            self.manual_exposure_var.set(str(status["exposure"]))
        if status.get("gain") is not None and not self.manual_gain_var.get().strip():
            self.manual_gain_var.set(str(status["gain"]))

        for item in self.detection_tree.get_children():
            self.detection_tree.delete(item)
        for detection in packet.detections:
            x, y = detection["center"]
            x1, y1, x2, y2 = detection["box"]
            self.detection_tree.insert(
                "", "end",
                values=(
                    detection["class"],
                    f"{detection['conf']:.3f}",
                    f"({x}, {y})",
                    f"({x1},{y1})-({x2},{y2})",
                ),
            )

    @staticmethod
    def _format_status(label: str, value: Any, suffix: str) -> str:
        return f"{label}：{value}{suffix}" if value is not None else f"{label}：--"

    def _apply_confidence(self) -> None:
        value = float(self.conf_var.get().strip())
        if not 0.001 <= value <= 0.99:
            raise ValueError("置信度必须在 0.001 到 0.99 之间")
        with self._settings_lock:
            self.conf_threshold = value
        self._append_log(f"置信度已设置为 {value:.3f}")

    def _on_auto_exposure(self) -> None:
        camera = self.camera
        if camera is None:
            return
        try:
            enabled = bool(self.auto_exposure_var.get())
            if enabled:
                with self._settings_lock:
                    self.manual_exposure_target = None
                    self.manual_gain_target = None
                camera.set_auto_exposure(True)
            else:
                # SDK 关闭 AE 时可能先把曝光重置为默认值，所以关闭后
                # 立刻按输入框中的目标值再写一次。
                camera.set_auto_exposure(False)
                exposure_text = self.manual_exposure_var.get().strip()
                gain_text = self.manual_gain_var.get().strip()
                target_exposure = int(exposure_text) if exposure_text else None
                target_gain = int(gain_text) if gain_text else None
                with self._settings_lock:
                    self.manual_exposure_target = target_exposure
                    self.manual_gain_target = target_gain
                if target_exposure is not None:
                    camera.set_exposure(target_exposure)
                if target_gain is not None:
                    camera.set_gain(target_gain)
            self._append_log(f"自动曝光：{'开' if enabled else '关'}")
        except Exception as exc:
            self._append_log(f"设置自动曝光失败：{exc}")

    def _on_laser(self) -> None:
        camera = self.camera
        if camera is None:
            self.laser_var.set(False)
            return
        try:
            camera.set_laser(bool(self.laser_var.get()))
            self._append_log(f"激光：{'开' if self.laser_var.get() else '关'}")
        except Exception as exc:
            self.laser_var.set(False)
            self._append_log(f"设置激光失败：{exc}")

    def _apply_exposure(self) -> None:
        camera = self.camera
        if camera is None:
            self._append_log("请先启动实时检测")
            return
        try:
            value = int(self.manual_exposure_var.get().strip())
            if self.auto_exposure_var.get():
                self.auto_exposure_var.set(False)
                camera.set_auto_exposure(False)
            camera.set_exposure(value)
            with self._settings_lock:
                self.manual_exposure_target = value
            self._append_log(f"曝光已设置为 {value} μs")
        except Exception as exc:
            self._append_log(f"设置曝光失败：{exc}")

    def _apply_gain(self) -> None:
        camera = self.camera
        if camera is None:
            self._append_log("请先启动实时检测")
            return
        try:
            value = int(self.manual_gain_var.get().strip())
            if self.auto_exposure_var.get():
                self.auto_exposure_var.set(False)
                camera.set_auto_exposure(False)
            camera.set_gain(value)
            with self._settings_lock:
                self.manual_gain_target = value
            self._append_log(f"增益已设置为 {value}")
        except Exception as exc:
            self._append_log(f"设置增益失败：{exc}")

    def _save_current(self) -> None:
        packet = self.last_packet
        if packet is None:
            messagebox.showinfo("没有画面", "请先启动实时检测并等待一帧")
            return
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        raw_path = CAPTURE_DIR / f"ir_raw_{stamp}.png"
        result_path = CAPTURE_DIR / f"ir_detection_{stamp}.jpg"
        cv2.imwrite(str(raw_path), packet.raw_bgr)
        cv2.imwrite(str(result_path), packet.annotated_bgr)
        self._append_log(f"已保存：{raw_path.name} / {result_path.name}")

    def _on_close(self) -> None:
        self.closing = True
        self.stop_event.set()
        if self.camera is not None:
            self.camera.close()
        if self.worker is None or not self.worker.is_alive():
            self.root.destroy()


def main() -> None:
    model_path = MODEL_PATH
    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1]).expanduser().resolve()
    root = tk.Tk()
    RealtimeDetectorApp(root, model_path)
    root.mainloop()


if __name__ == "__main__":
    main()
