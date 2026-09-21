"""固定观察位的鼠标画区窗口。

画区阶段使用持续的 RGB 视频流，操作员可以一边调整观察位一边看当前画面。
点击“确认当前位姿并取图”时，窗口锁定最新一帧，并只读一次机械臂状态，
把 TCP 位姿、关节角、帧时间戳和图片一起保存，供后续复现观察位和审计使用。
这里不会发送任何运动命令。
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from .auto_sector_selection import AutoSectorConfig, validate_region_polygon
from .io_utils import atomic_write_json, jsonable


POSE_RECORD_SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def make_polygon_config(regions, image_size, base=None, pose_records=None):
    """生成多边形扇区配置，并保留画区时记录的机器人位姿。"""
    raw = dict(base or {})
    records = list(raw.get("sector_pose_records", [])) if pose_records is None else list(pose_records)
    raw.update({
        "partition_mode": "polygons", "origin_px": None,
        "regions": sorted(regions, key=lambda r: int(r["sector_id"])),
        "image_size_px": list(image_size), "roi_polygon_px": None,
        "active_sector_ids": None,
        "sector_pose_records": records,
    })
    return AutoSectorConfig.from_dict(raw)


class SectorEditor(tk.Toplevel):
    """持续显示相机 RGB 流的扇区编辑器。"""

    def __init__(self, parent, config_path, on_saved, connection_provider=None):
        super().__init__(parent)
        self.title("鼠标画工作区域")
        self.config_path = Path(config_path)
        self.on_saved = on_saved
        self.connection_provider = connection_provider
        self.base = {}
        if self.config_path.is_file():
            try:
                self.base = json.loads(self.config_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                pass
        if not isinstance(self.base, dict):
            self.base = {}
        self.regions = list(self.base.get("regions", []))
        self.points = []
        self.image = None
        self.frame_metadata = {}
        self.scale = 1.0

        # 相机线程持续更新 latest_frame；Tk 主线程只在 poll 中刷新画布。
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_metadata = {}
        self.latest_frame_key = None
        self.displayed_frame_key = None
        self.stream_stop = threading.Event()
        self.stream_thread = None
        self.stream_running = False
        self.stream_starting = False
        self.pose_busy = False
        self.frame_locked = False
        self.offline_mode = False
        self.current_pose_record_id = None
        self.current_pose_sector_id = None
        self.current_pose_frame_image_path = None
        self.closing = False
        self.events = queue.Queue()

        self.pose_records_path = self.config_path.with_name(
            f"{self.config_path.stem}_pose_records.json"
        )
        self.pose_records = self._load_pose_records()

        self.sid = tk.StringVar(value="1")
        self.status = tk.StringVar(
            value="请启动 RGB 视频流；确认位置时会同时记录最新画面和机械臂当前位姿。"
        )
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill="x")
        self.stream_button = ttk.Button(
            toolbar, text="启动视频流", command=self.toggle_stream
        )
        self.stream_button.pack(side="left")
        # 保留旧属性名，避免外部脚本或旧测试引用 capture_button。
        self.capture_button = self.stream_button
        self.confirm_button = ttk.Button(
            toolbar,
            text="确认当前位姿并取图",
            command=self.confirm_pose_and_frame,
            state="disabled",
        )
        self.confirm_button.pack(side="left", padx=6)
        ttk.Button(toolbar, text="打开已有图像（离线）", command=self.load_image).pack(
            side="left", padx=6
        )
        ttk.Label(toolbar, text="区域编号").pack(side="left")
        ttk.Combobox(
            toolbar,
            textvariable=self.sid,
            values=[str(i) for i in range(1, 7)],
            state="readonly",
            width=4,
        ).pack(side="left")
        ttk.Button(toolbar, text="闭合并保存本区域", command=self.finish_region).pack(
            side="left", padx=6
        )
        ttk.Button(toolbar, text="删除当前编号区域", command=self.delete_region).pack(
            side="left"
        )
        ttk.Button(toolbar, text="保存配置并使用", command=self.save).pack(
            side="left", padx=6
        )
        ttk.Label(
            self,
            text="视频流实时显示当前观察位；确认当前位姿后左键逐点画轮廓，右键撤销一点。重叠/公共边归编号最小的区域；区域外不选孔。",
            padding=6,
        ).pack(anchor="w")
        ttk.Label(
            self,
            text="先选择区域编号并把机械臂调整到该观察位，再点击“确认当前位姿并取图”；记录只读状态，不会运动机械臂。",
            padding=6,
        ).pack(anchor="w")
        ttk.Label(
            self,
            text=f"位姿记录：{self.pose_records_path}",
            foreground="#555555",
            padding=(6, 0, 6, 4),
        ).pack(anchor="w")
        self.canvas = tk.Canvas(
            self, width=960, height=540, background="#333333", highlightthickness=0
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Button-3>", self.undo_point)
        ttk.Label(self, textvariable=self.status, padding=8).pack(anchor="w")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.poll_id = self.after(50, self.poll)

    def _load_pose_records(self):
        """读取主配置和旁车记录，按 record_id 合并，避免重复。"""
        records = []
        configured = self.base.get("sector_pose_records", [])
        if isinstance(configured, list):
            records.extend(item for item in configured if isinstance(item, dict))
        if self.pose_records_path.is_file():
            try:
                payload = json.loads(self.pose_records_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    sidecar = payload.get("records", [])
                elif isinstance(payload, list):
                    sidecar = payload
                else:
                    sidecar = []
                if isinstance(sidecar, list):
                    records.extend(item for item in sidecar if isinstance(item, dict))
            except (ValueError, OSError):
                pass
        result = []
        seen = set()
        for record in records:
            key = str(record.get("record_id", "")) or repr(record)
            if key in seen:
                continue
            seen.add(key)
            result.append(record)
        return result

    def _write_pose_records(self):
        atomic_write_json(
            self.pose_records_path,
            jsonable(
                {
                    "schema_version": POSE_RECORD_SCHEMA_VERSION,
                    "config_path": str(self.config_path),
                    "records": self.pose_records,
                }
            ),
        )

    def destroy(self):
        self.closing = True
        self.stream_stop.set()
        if getattr(self, "poll_id", None):
            try:
                self.after_cancel(self.poll_id)
            except tk.TclError:
                pass
            self.poll_id = None
        super().destroy()

    def close(self):
        if self.pose_busy:
            self.status.set("正在读取机械臂位姿，请稍候再关闭窗口。")
            return
        self.destroy()

    def set_image(self, image, *, reset_points=True, frame_metadata=None):
        if not isinstance(image, np.ndarray) or image.ndim < 2:
            raise ValueError("图像必须是有效的 numpy 彩色图像")
        h, w = image.shape[:2]
        old_size = None if self.image is None else self.image.shape[:2]
        size_changed = old_size is not None and old_size != (h, w)
        configured_size = self.base.get("image_size_px")
        if size_changed or (
            old_size is None
            and configured_size
            and list(configured_size) != [w, h]
        ):
            # 分辨率变化后原像素坐标已失效，必须重新画区，不能静默缩放旧区域。
            self.regions = []
            self.points = []
        elif reset_points:
            self.points = []
        self.image = image
        self.frame_metadata = dict(frame_metadata or {})
        self.scale = min(1.0, 1100 / w, 650 / h)
        sw, sh = max(1, int(w * self.scale)), max(1, int(h * self.scale))
        rgb = cv2.cvtColor(cv2.resize(image, (sw, sh)), cv2.COLOR_BGR2RGB)
        self.photo = tk.PhotoImage(
            data=f"P6\n{sw} {sh}\n255\n".encode() + rgb.tobytes(), format="PPM"
        )
        self.canvas.configure(width=sw, height=sh)
        self.redraw()
        if self.stream_running and not self.offline_mode:
            frame_index = self.frame_metadata.get("color_frame_index")
            suffix = f"帧 {frame_index}" if frame_index is not None else "视频流"
            self.status.set(
                f"RGB {w} × {h}（{suffix}）；已保存区域："
                f"{[r['sector_id'] for r in self.regions]}。"
            )
        else:
            self.status.set(
                f"原图 {w} × {h}；已保存区域："
                f"{[r['sector_id'] for r in self.regions]}。"
            )

    def load_image(self):
        if self.pose_busy or self.stream_starting:
            return
        if self.stream_running:
            self.stop_stream()
        path = filedialog.askopenfilename(
            parent=self,
            title="选择固定观察位的原始相机图像",
            filetypes=[("图像", "*.png *.jpg *.jpeg *.bmp")],
        )
        if path:
            image = cv2.imdecode(
                np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                messagebox.showerror("读取失败", "不能读取图像", parent=self)
            else:
                with self.frame_lock:
                    self.latest_frame = None
                    self.latest_frame_metadata = {}
                    self.latest_frame_key = None
                self.offline_mode = True
                self.set_image(image)

    def toggle_stream(self):
        if self.stream_running or self.stream_starting:
            self.stop_stream()
        else:
            self.start_stream()

    def start_stream(self):
        if self.stream_running or self.stream_starting:
            return
        with self.frame_lock:
            # 重启视频流时不允许把上一轮缓存帧误当成当前观察位。
            self.latest_frame = None
            self.latest_frame_metadata = {}
            self.latest_frame_key = None
            self.displayed_frame_key = None
        self.stream_stop.clear()
        self.offline_mode = False
        self.frame_locked = False
        self.current_pose_record_id = None
        self.current_pose_sector_id = None
        self.current_pose_frame_image_path = None
        self.confirm_button.configure(text="确认当前位姿并取图")
        self.stream_starting = True
        self.stream_button.configure(state="disabled", text="正在启动视频流…")
        self.confirm_button.configure(state="disabled")
        self.status.set("正在启动 RGB 视频流，请稍候……")
        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()

    def stop_stream(self):
        if not self.stream_running and not self.stream_starting:
            return
        self.stream_stop.set()
        self.stream_button.configure(state="disabled", text="正在停止视频流…")
        self.confirm_button.configure(state="disabled")
        self.status.set("正在停止 RGB 视频流……")

    def _stream_worker(self):
        pipeline = None
        error = None
        started = False
        try:
            from .camera import get_rgb_frame_bundle, init_rgb_handeye_pipeline

            pipeline = init_rgb_handeye_pipeline()
            started = True
            self.events.put(("stream_started", None))
            while not self.stream_stop.is_set():
                bundle = get_rgb_frame_bundle(pipeline)
                if bundle is None:
                    continue
                image = bundle.color_bgr.copy()
                metadata = bundle.metadata_dict()
                key = (
                    metadata.get("color_frame_index"),
                    metadata.get("host_timestamp_ns", time.time_ns()),
                )
                with self.frame_lock:
                    self.latest_frame = image
                    self.latest_frame_metadata = metadata
                    self.latest_frame_key = key
        except Exception as exc:  # 相机/SDK错误在窗口中显示，不让Tk线程崩溃
            error = str(exc)
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception as exc:
                    error = f"释放相机失败：{exc}"
            self.events.put(("stream_stopped", {"started": started, "error": error}))

    def _latest_frame_copy(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None, {}
            return self.latest_frame.copy(), dict(self.latest_frame_metadata)

    def confirm_pose_and_frame(self):
        """锁定最新视频帧并读取一次机械臂状态。"""
        if self.pose_busy:
            return
        if self.frame_locked:
            self.frame_locked = False
            self.current_pose_record_id = None
            self.current_pose_sector_id = None
            self.current_pose_frame_image_path = None
            self.confirm_button.configure(text="确认当前位姿并取图")
            self.status.set("已解除当前帧锁定；视频流继续刷新，可调整到下一个扇区。")
            return
        if not self.stream_running:
            self.status.set("请先启动视频流；位姿记录必须绑定视频流中的当前帧。")
            return
        frame, frame_metadata = self._latest_frame_copy()
        if frame is None:
            self.status.set("视频流尚未取得有效帧，请稍候再确认。")
            return
        if self.connection_provider is None:
            messagebox.showerror(
                "无法记录位姿",
                "当前窗口没有机械臂连接参数，不能生成位姿记录。请从工作台打开画区窗口。",
                parent=self,
            )
            return
        try:
            # connection_provider 通常读取 Tk 变量，只在主线程调用。
            connection = dict(self.connection_provider())
        except Exception as exc:
            messagebox.showerror("读取连接参数失败", str(exc), parent=self)
            return
        required = ("ip", "port", "user", "password", "timeout_ms")
        missing = [key for key in required if key not in connection]
        if missing:
            messagebox.showerror(
                "连接参数不完整", f"缺少机械臂连接参数：{', '.join(missing)}", parent=self
            )
            return

        sid = int(self.sid.get())
        captured_at = _now_iso()
        frame_host_timestamp_ns = int(
            frame_metadata.get("host_timestamp_ns", time.time_ns())
        )
        # 先在主线程锁定按钮时看到的最新帧，后续画区坐标与保存图片严格使用它。
        self.frame_locked = True
        self.set_image(frame, reset_points=False, frame_metadata=frame_metadata)
        self.pose_busy = True
        self.confirm_button.configure(state="disabled")
        self.stream_button.configure(state="disabled")
        self.status.set(f"正在读取 S{sid:02d} 的机械臂当前位姿，请保持机械臂静止……")

        def work():
            session = None
            try:
                from .motion_control import AuboMotionSession

                session = AuboMotionSession()
                session.connect(
                    str(connection["ip"]),
                    int(connection["port"]),
                    str(connection["user"]),
                    str(connection["password"]),
                    int(connection["timeout_ms"]),
                )
                # snapshot() 只读取控制器状态，不调用任何运动接口。
                snapshot = session.snapshot()
                robot_snapshot_captured_at = _now_iso()
                capture_dir = self.config_path.parent / f"S{sid:02d}" / "captures"
                capture_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                image_path = capture_dir / (
                    f"{self.config_path.stem}_S{sid:02d}_{stamp}.png"
                )
                success, encoded = cv2.imencode(".png", frame)
                if not success:
                    raise RuntimeError("当前视频帧编码失败")
                encoded.tofile(str(image_path))
                record = {
                    "record_id": f"S{sid:02d}_{stamp}",
                    "schema_version": POSE_RECORD_SCHEMA_VERSION,
                    "sector_id": sid,
                    "captured_at": captured_at,
                    "robot_snapshot_captured_at": robot_snapshot_captured_at,
                    "frame_image_path": str(image_path),
                    "frame_host_timestamp_ns": frame_host_timestamp_ns,
                    "frame_metadata": frame_metadata,
                    "robot_snapshot": snapshot,
                }
                self.events.put(("pose_recorded", record))
            except Exception as exc:
                self.events.put(("pose_error", str(exc)))
            finally:
                if session is not None:
                    try:
                        session.disconnect()
                    except Exception:
                        pass

        threading.Thread(target=work, daemon=True).start()

    def _drain_events(self):
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                return
            if kind == "stream_started":
                self.stream_starting = False
                self.stream_running = True
                self.stream_button.configure(state="normal", text="停止视频流")
                self.confirm_button.configure(state="normal")
                self.status.set("RGB 视频流已启动；调整观察位后点击“确认当前位姿并取图”。")
            elif kind == "stream_stopped":
                self.stream_starting = False
                self.stream_running = False
                self.stream_button.configure(state="normal", text="启动视频流")
                self.confirm_button.configure(state="disabled")
                error = payload.get("error") if isinstance(payload, dict) else None
                if error and not self.closing:
                    self.status.set(f"视频流已停止：{error}")
                    messagebox.showerror("视频流失败", error, parent=self)
                elif not self.closing:
                    self.status.set("视频流已停止。需要实时画面时可再次启动。")
            elif kind == "pose_recorded":
                self.pose_busy = False
                self.pose_records.append(payload)
                self.current_pose_record_id = str(payload.get("record_id"))
                self.current_pose_sector_id = int(payload.get("sector_id"))
                self.current_pose_frame_image_path = payload.get("frame_image_path")
                try:
                    self._write_pose_records()
                except Exception as exc:
                    self.status.set(f"位姿已读取，但旁车记录写入失败：{exc}")
                    messagebox.showerror("位姿记录失败", str(exc), parent=self)
                else:
                    if self.stream_running:
                        self.stream_button.configure(state="normal")
                        self.confirm_button.configure(state="normal")
                        self.confirm_button.configure(text="继续视频流")
                    snapshot = payload.get("robot_snapshot", {})
                    tcp = snapshot.get("tcp_pose_m_rad")
                    joints = snapshot.get("joints_rad")
                    self.status.set(
                        f"S{int(payload['sector_id']):02d} 位姿已记录；"
                        f"TCP={tcp}；关节={joints}。可继续画区。"
                    )
            elif kind == "pose_error":
                self.pose_busy = False
                self.frame_locked = False
                self.current_pose_record_id = None
                self.current_pose_sector_id = None
                self.current_pose_frame_image_path = None
                self.confirm_button.configure(text="确认当前位姿并取图")
                if self.stream_running:
                    self.stream_button.configure(state="normal")
                    self.confirm_button.configure(state="normal")
                self.status.set(f"读取机械臂位姿失败：{payload}")
                if not self.closing:
                    messagebox.showerror("位姿记录失败", str(payload), parent=self)

    def poll(self):
        if self.closing:
            return
        self._drain_events()
        with self.frame_lock:
            key = self.latest_frame_key
            frame = self.latest_frame
            metadata = dict(self.latest_frame_metadata)
        if (
            frame is not None
            and not self.offline_mode
            and not self.frame_locked
            and key != self.displayed_frame_key
        ):
            # worker只替换 latest_frame，不修改已交给Tk的数组；这里可直接显示。
            try:
                self.set_image(frame, reset_points=False, frame_metadata=metadata)
                self.displayed_frame_key = key
            except Exception as exc:
                self.status.set(f"刷新视频帧失败：{exc}")
        self.poll_id = self.after(50, self.poll)

    def redraw(self):
        self.canvas.delete("all")
        if self.image is None:
            return
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        for region in self.regions:
            xy = [float(v) * self.scale for p in region["polygon_px"] for v in p]
            self.canvas.create_polygon(*xy, fill="", outline="#00ff88", width=2)
            self.canvas.create_text(
                xy[0], xy[1], text=f"S{int(region['sector_id']):02d}",
                anchor="nw", fill="#00ff88"
            )
        xy = [v * self.scale for p in self.points for v in p]
        if len(self.points) > 1:
            self.canvas.create_line(*xy, fill="yellow", width=2)
        for x, y in self.points:
            x, y = x * self.scale, y * self.scale
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="yellow")

    def add_point(self, event):
        if self.image is None or self.pose_busy:
            return
        if self.connection_provider is not None and not self.offline_mode and not self.frame_locked:
            self.status.set("请先确认当前位姿并锁定视频帧，再开始画该区域。")
            return
        h, w = self.image.shape[:2]
        self.points.append([
            min(w - 1, max(0, event.x / self.scale)),
            min(h - 1, max(0, event.y / self.scale)),
        ])
        self.redraw()

    def undo_point(self, _event=None):
        if self.points and not self.pose_busy:
            self.points.pop()
            self.redraw()

    def finish_region(self):
        if self.pose_busy:
            self.status.set("正在读取机械臂位姿，请稍候再闭合区域。")
            return False
        sid = int(self.sid.get())
        if self.connection_provider is not None and not self.offline_mode and not (
            self.frame_locked
            and self.current_pose_record_id
            and self.current_pose_sector_id == sid
        ):
            self.status.set(
                f"请先选择 S{sid:02d}，启动视频流并点击“确认当前位姿并取图”，"
                "再闭合该区域。"
            )
            return False
        try:
            validate_region_polygon(self.points)
        except ValueError as exc:
            messagebox.showerror("区域无效", str(exc), parent=self)
            return False
        self.regions = [r for r in self.regions if int(r["sector_id"]) != sid]
        region = {"sector_id": sid, "polygon_px": list(self.points)}
        if self.current_pose_sector_id == sid and self.current_pose_record_id:
            region.update({
                "pose_record_id": self.current_pose_record_id,
                "frame_image_path": self.current_pose_frame_image_path,
            })
        self.regions.append(region)
        self.points = []
        self.redraw()
        self.status.set(f"S{sid:02d} 已闭合；可换编号画下一区域，或保存配置。")
        return True

    def delete_region(self):
        self.regions = [
            r for r in self.regions if int(r["sector_id"]) != int(self.sid.get())
        ]
        self.points = []
        self.redraw()

    def save(self):
        if self.image is None or self.pose_busy:
            self.status.set("请先取得图像，并等待位姿读取完成。")
            return
        if self.points and not self.finish_region():
            return
        if self.connection_provider is not None and not self.offline_mode:
            record_ids = {
                str(record.get("record_id"))
                for record in self.pose_records
                if record.get("record_id")
            }
            missing_pose = [
                f"S{int(region['sector_id']):02d}"
                for region in self.regions
                if str(region.get("pose_record_id", "")) not in record_ids
            ]
            if missing_pose:
                self.status.set(
                    "以下区域还没有绑定视频帧和机械臂位姿："
                    + ", ".join(missing_pose)
                    + "。请逐个选择区域并确认当前位姿。"
                )
                return
        try:
            h, w = self.image.shape[:2]
            config = make_polygon_config(
                self.regions, (w, h), self.base, pose_records=self.pose_records
            )
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            image_path = self.config_path.with_suffix(".reference.png")
            success, encoded = cv2.imencode(".png", self.image)
            if not success:
                raise ValueError("原图编码失败")
            encoded.tofile(str(image_path))
            payload = config.to_dict()
            payload["reference_image_path"] = str(image_path)
            payload["pose_records_path"] = str(self.pose_records_path)
            atomic_write_json(self.config_path, jsonable(payload))
            # 主配置保存成功后旁车也立即落盘，避免只关窗口导致记录丢失。
            self._write_pose_records()
        except (ValueError, OSError, RuntimeError) as exc:
            messagebox.showerror("保存失败", str(exc), parent=self)
            return
        self.on_saved(self.config_path)
        self.destroy()
