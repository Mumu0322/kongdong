#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RGB 手眼界面：采集、诊断求解和样本维护。"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .capture import capture_burst_samples_gui
from .camera import get_rgb_frame_bundle, init_rgb_handeye_pipeline, print_device_info
from .charuco_detect import create_charuco_board, estimate_rgb_board_pose
from .config import CAMERA_CFG, ROBOT_CFG, SOLVE_CFG
from .gui_common import GuiLogWriter
from .io_utils import make_dir
from .quality import evaluate_image_quality
from .robot import AUBO_SESSION, close_aubo_session
from .samples import CalibSample, archive_samples, load_existing_samples, next_available_sample_index
from .solve import format_handeye_summary, solve_and_save
try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    ImageTk = None  # type: ignore

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def load_active_rgb_samples() -> list[CalibSample]:
    """只加载 RGB 样本；旧点云样本移动到可追溯归档目录。"""
    loaded = load_existing_samples() if SOLVE_CFG.load_existing_samples_on_start else []
    rgb_samples = [
        sample for sample in loaded
        if str(sample.calibration_frame or "").strip().lower() == "rgb_camera"
    ]
    legacy_samples = [
        sample for sample in loaded
        if str(sample.calibration_frame or "").strip().lower() != "rgb_camera"
    ]
    if not rgb_samples:
        print("[HANDEYE] 当前没有活动 RGB 样本。")
    if legacy_samples:
        archive_dir = archive_samples(
            legacy_samples,
            rgb_samples,
            reason="legacy_pointcloud_samples_archived_before_rgb_handeye",
            archive_prefix="archived_legacy_pointcloud",
        )
        print(
            f"[MIGRATE] 已将{len(legacy_samples)}个旧点云手眼样本移出活动目录：{archive_dir}；"
            "它们仍可追溯，但不会与RGB样本混合求解。"
        )
    return rgb_samples


class HandEyeGuiPanel(ttk.Frame):
    """可以独立成窗口，也可以当作子页面嵌到别的 Tk 容器（workbench.py 就是这么用的）。"""

    def __init__(self, master: tk.Misc | None = None, autostart: bool = True) -> None:
        super().__init__(master)
        self.display_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.status_queue: "queue.Queue[dict[str, str]]" = queue.Queue()
        self.command_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.worker_sample_dir: Path | None = None
        self.photo_image: Any | None = None

        self.ip_var = tk.StringVar(value=ROBOT_CFG.ip)
        self.port_var = tk.StringVar(value=str(ROBOT_CFG.rpc_port))
        self.user_var = tk.StringVar(value=ROBOT_CFG.user)
        self.password_var = tk.StringVar(value=ROBOT_CFG.password)
        self.controller_pose_var = tk.StringVar(value="未连接")
        self.controller_offset_var = tk.StringVar(value="未读取")
        self._last_pose_poll = 0.0
        self.save_dir_var = tk.StringVar(value=CAMERA_CFG.save_dir)
        self.output_json_var = tk.StringVar(value=SOLVE_CFG.output_json)

        self.camera_status_var = tk.StringVar(value="未启动")
        self.robot_status_var = tk.StringVar(value="未连接")
        self.sample_status_var = tk.StringVar(value="0")
        self.quality_status_var = tk.StringVar(value="等待画面")
        self.output_status_var = tk.StringVar(value=SOLVE_CFG.output_json)
        self.result_var = tk.StringVar(value="尚未求解")

        self._build_ui()
        self.after(120, self._poll_queues)
        if autostart:
            self.after(300, self.start_worker)

    # ---------------- UI 构建 ----------------

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TButton", padding=(8, 5))
        style.configure("TLabelframe", padding=8)

        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        config = ttk.LabelFrame(root, text="标定文件与位姿源")
        config.pack(fill=tk.X)
        config.columnconfigure(1, weight=1)
        ttk.Label(config, text="机械臂连接参数使用工作台顶部统一设置。", foreground="#555555").grid(
            row=0, column=0, columnspan=4, sticky="w", padx=(0, 12), pady=3
        )
        ttk.Label(config, text="位姿源：控制器当前 TCP / 基坐标系").grid(
            row=0, column=4, columnspan=2, sticky="w", pady=3,
        )

        ttk.Label(config, text="采集目录").grid(row=1, column=0, sticky="w", padx=(0, 4), pady=3)
        ttk.Entry(config, textvariable=self.save_dir_var).grid(row=1, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=3)
        ttk.Button(config, text="选择", command=self.browse_save_dir).grid(row=1, column=4, sticky="w", padx=(0, 8), pady=3)
        ttk.Button(config, text="打开目录", command=self.open_save_dir).grid(row=1, column=5, sticky="w", pady=3)

        ttk.Label(config, text="输出JSON").grid(row=2, column=0, sticky="w", padx=(0, 4), pady=3)
        ttk.Entry(config, textvariable=self.output_json_var).grid(row=2, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=3)
        ttk.Button(config, text="选择", command=self.browse_output_json).grid(row=2, column=4, sticky="w", padx=(0, 8), pady=3)

        toolbar = ttk.Frame(root)
        toolbar.pack(fill=tk.X, pady=(8, 8))
        self.start_btn = ttk.Button(toolbar, text="启动相机", command=self.start_worker)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.stop_btn = ttk.Button(toolbar, text="停止相机", command=self.stop_worker)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 14))
        self.connect_btn = ttk.Button(toolbar, text="连接机械臂", command=lambda: self.enqueue_command("connect"))
        self.connect_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.disconnect_btn = ttk.Button(toolbar, text="断开机械臂", command=lambda: self.enqueue_command("disconnect"))
        self.disconnect_btn.pack(side=tk.LEFT, padx=(0, 14))
        self.capture_btn = ttk.Button(
            toolbar, text="采集5帧选1帧", command=lambda: self.enqueue_command("capture"),
        )
        self.capture_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.solve_btn = ttk.Button(toolbar, text="求解标定", command=lambda: self.enqueue_command("solve"))
        self.solve_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.maintenance_btn = ttk.Menubutton(toolbar, text="样本维护")
        maintenance_menu = tk.Menu(self.maintenance_btn, tearoff=False)
        maintenance_menu.add_command(label="归档最后样本", command=lambda: self.enqueue_command("delete_last"))
        self.maintenance_btn.configure(menu=maintenance_menu)
        self.maintenance_btn.pack(side=tk.LEFT)

        status = ttk.LabelFrame(root, text="状态")
        status.pack(fill=tk.X, pady=(0, 8))
        for col in range(8):
            status.columnconfigure(col, weight=1)
        self._status_label(status, 0, 0, "相机", self.camera_status_var)
        self._status_label(status, 0, 2, "机械臂", self.robot_status_var)
        self._status_label(status, 0, 4, "样本数", self.sample_status_var)
        self._status_label(status, 0, 6, "质量", self.quality_status_var)
        self._status_label(status, 1, 0, "输出", self.output_status_var, columnspan=7)
        self._status_label(status, 2, 0, "当前TCP", self.controller_pose_var, columnspan=7)
        self._status_label(status, 3, 0, "实际偏置", self.controller_offset_var, columnspan=7)

        content = ttk.Frame(root)
        content.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(content, text="实时标定画面")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.video_label = ttk.Label(left, anchor="center", background="#111111")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        right_shell = ttk.Frame(content, width=400)
        right_shell.pack(side=tk.RIGHT, fill=tk.Y)
        right_shell.pack_propagate(False)
        right_canvas = tk.Canvas(
            right_shell, highlightthickness=0, borderwidth=0,
        )
        right_scrollbar = ttk.Scrollbar(
            right_shell, orient=tk.VERTICAL, command=right_canvas.yview,
        )
        right_canvas.configure(yscrollcommand=right_scrollbar.set)
        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        right = ttk.Frame(right_canvas)
        right_window = right_canvas.create_window((0, 0), window=right, anchor="nw")
        right.bind(
            "<Configure>",
            lambda _event: right_canvas.configure(scrollregion=right_canvas.bbox("all")),
        )
        right_canvas.bind(
            "<Configure>",
            lambda event: right_canvas.itemconfigure(
                right_window, width=max(1, int(event.width)),
            ),
        )
        self.right_canvas = right_canvas
        self.right_scrollbar = right_scrollbar

        result_frame = ttk.LabelFrame(right, text="标定结果")
        result_frame.pack(fill=tk.X, pady=(0, 8))
        result_label = ttk.Label(result_frame, textvariable=self.result_var, justify=tk.LEFT)
        result_label.pack(fill=tk.X, padx=6, pady=6)
        result_frame.bind(
            "<Configure>", lambda event: result_label.configure(wraplength=max(100, event.width - 32)),
        )

        log_frame = ttk.LabelFrame(right, text="日志")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, height=20, wrap=tk.WORD, font=("Microsoft YaHei UI", 9))
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=6)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 6), pady=6)

        self._update_button_state()

    def _status_label(self, parent: ttk.LabelFrame, row: int, column: int, name: str, var: tk.StringVar, columnspan: int = 1) -> None:
        ttk.Label(parent, text=name + "：").grid(row=row, column=column, sticky="w", padx=(8, 2), pady=3)
        ttk.Label(parent, textvariable=var).grid(row=row, column=column + 1, columnspan=columnspan, sticky="w", pady=3)

    # ---------------- 表单/文件对话框 ----------------

    def browse_save_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=self.save_dir_var.get() or str(os.getcwd()))
        if path:
            self.save_dir_var.set(path)

    def browse_output_json(self) -> None:
        from pathlib import Path

        path = filedialog.asksaveasfilename(
            initialfile=Path(self.output_json_var.get()).name,
            initialdir=str(Path(self.output_json_var.get()).parent),
            defaultextension=".json",
            filetypes=(("JSON 文件", "*.json"), ("所有文件", "*.*")),
        )
        if path:
            self.output_json_var.set(path)

    def open_save_dir(self) -> None:
        if not self.apply_form_config():
            return
        make_dir(CAMERA_CFG.save_dir)
        try:
            os.startfile(CAMERA_CFG.save_dir)  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showerror("打开目录失败", str(exc))

    def apply_form_config(self) -> bool:
        ip = self.ip_var.get().strip()
        user = self.user_var.get().strip()
        password = self.password_var.get()
        save_dir = self.save_dir_var.get().strip()
        output_json = self.output_json_var.get().strip()
        try:
            port = int(self.port_var.get().strip())
            if not ip:
                raise ValueError("机械臂 IP 不能为空")
            if port <= 0 or port > 65535:
                raise ValueError("端口必须在 1-65535")
            if not save_dir:
                raise ValueError("采集目录不能为空")
            if not output_json:
                raise ValueError("输出 JSON 路径不能为空")
        except Exception as exc:
            messagebox.showerror("配置错误", str(exc))
            return False

        configured_dir = Path(save_dir).expanduser().resolve()
        if (
            self.worker_thread is not None
            and self.worker_alive()
            and self.worker_sample_dir is not None
            and configured_dir != self.worker_sample_dir
        ):
            messagebox.showerror(
                "不能切换采集目录",
                "相机运行期间不能更改采集目录。请先停止相机，修改目录后重新启动，"
                "避免内存样本和磁盘样本混用。",
            )
            return False

        ROBOT_CFG.ip = ip
        ROBOT_CFG.rpc_port = port
        ROBOT_CFG.user = user
        ROBOT_CFG.password = password
        CAMERA_CFG.save_dir = save_dir
        SOLVE_CFG.output_json = output_json
        self.output_status_var.set(output_json)
        return True

    def _publish_controller_pose(self, snapshot: dict[str, Any]) -> None:
        pose = snapshot["pose_values"]
        offset = snapshot["actual_tcp_offset_sdk_m_rad"]
        offset_mm_deg = [v * 1000.0 for v in offset[:3]] + np.rad2deg(offset[3:]).tolist()

        def format_pose(values) -> str:
            xyz = ", ".join(f"{v:.3f}" for v in values[:3])
            rpy = ", ".join(f"{v:.3f}" for v in values[3:])
            return f"XYZ [mm]: {xyz}    Rx/Ry/Rz [deg]: {rpy}"

        self._worker_status(
            controller_pose=format_pose(pose), controller_offset=format_pose(offset_mm_deg),
        )

    # ---------------- 后台线程管理 ----------------

    def worker_alive(self) -> bool:
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def start_worker(self) -> None:
        if self.worker_alive():
            return
        if not self.apply_form_config():
            return
        self.worker_sample_dir = Path(CAMERA_CFG.save_dir).expanduser().resolve()
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._worker_main, name="handeye-gui-worker", daemon=True)
        self.worker_thread.start()
        self.camera_status_var.set("启动中")
        self._update_button_state()

    def stop_worker(self) -> None:
        self.stop_event.set()
        self.camera_status_var.set("停止中")
        self._update_button_state()

    def enqueue_command(self, command: str, payload: Any = None) -> None:
        if not self.apply_form_config():
            return
        if not self.worker_alive():
            messagebox.showwarning("未启动", "请先点击“启动相机”。")
            return
        self.command_queue.put((command, payload))

    def _worker_status(self, **kwargs: str) -> None:
        self.status_queue.put({k: str(v) for k, v in kwargs.items()})

    def _put_display(self, image_bgr: np.ndarray) -> None:
        try:
            if self.display_queue.full():
                self.display_queue.get_nowait()
            self.display_queue.put_nowait(image_bgr)
        except Exception:
            pass

    def _drain_worker_commands(
        self, pipeline, board, dictionary,
        next_index: int, samples: list[CalibSample],
    ) -> tuple[int, list[CalibSample]]:
        while True:
            try:
                command, payload = self.command_queue.get_nowait()
            except queue.Empty:
                break

            try:
                if command == "network":
                    ok, message = AUBO_SESSION.network_precheck()
                    print("[NETWORK]", "通过" if ok else "失败", message)
                    self._worker_status(robot="端口可达" if ok else "网络不可达")
                elif command == "connect":
                    AUBO_SESSION.connect()
                    snap = AUBO_SESSION.read_pose_snapshot()
                    pose_values = snap.get("named_pose_values", {})
                    print(f"[AUBO] GUI连接成功：{AUBO_SESSION.robot_name} {AUBO_SESSION.robot_type} pose={pose_values}")
                    self._worker_status(robot=f"已连接 {AUBO_SESSION.robot_name} {AUBO_SESSION.robot_type}")
                    self._publish_controller_pose(snap)
                elif command == "disconnect":
                    close_aubo_session()
                    print("[AUBO] 已断开机械臂")
                    self._worker_status(robot="已断开")
                    self._worker_status(controller_pose="未连接", controller_offset="未读取")
                elif command == "capture":
                    if pipeline is None:
                        print("[GUI] 相机未就绪，不能采集样本。")
                    else:
                        next_index = capture_burst_samples_gui(
                            pipeline, board, dictionary,
                            next_index, samples, publish_display=self._put_display, stop_event=self.stop_event,
                        )
                        self._worker_status(samples=str(len(samples)), result="样本已更新，请重新求解")
                elif command == "solve":
                    result = solve_and_save(samples)
                    if result is not None:
                        self._worker_status(output=SOLVE_CFG.output_json, result=format_handeye_summary(result))
                    else:
                        self._worker_status(result="样本不足，无法求解")
                elif command == "delete_last":
                    if samples:
                        removed = samples[-1]
                        remaining = list(samples[:-1])
                        archive_dir = archive_samples(
                            [removed], remaining, reason="user_removed_last_sample",
                        )
                        samples = remaining
                        next_index = next_available_sample_index(samples)
                        print(
                            f"[GUI] 已归档最后一个样本 index={removed.index}，"
                            f"活动CSV已同步：{archive_dir}"
                        )
                    else:
                        print("[GUI] 当前没有可删除样本。")
                    self._worker_status(samples=str(len(samples)), result="样本已更新，请重新求解")
            except Exception as exc:
                self._worker_status(result=f"操作失败：{exc}")
                print(f"[GUI-ERROR] 命令 {command} 执行失败：{exc}")
                print(traceback.format_exc())
        return next_index, samples

    def _worker_main(self) -> None:
        pipeline = None
        samples: list[CalibSample] = []
        next_index = 1
        try:
            print("=" * 70)
            print("AUBO 眼在手 RGB-PnP 手眼标定 GUI")
            print("Camera save_dir:", CAMERA_CFG.save_dir)
            print("Output json:", SOLVE_CFG.output_json)
            print(f"AUBO: {ROBOT_CFG.ip}:{ROBOT_CFG.rpc_port}, 自动读取控制器当前TCP")
            print("=" * 70)

            self.worker_sample_dir = Path(CAMERA_CFG.save_dir).expanduser().resolve()
            make_dir(CAMERA_CFG.save_dir)
            make_dir(Path(SOLVE_CFG.output_json).parent)
            board, dictionary = create_charuco_board()
            samples = load_active_rgb_samples()
            next_index = next_available_sample_index(samples)
            self._worker_status(samples=str(len(samples)), output=SOLVE_CFG.output_json, result="尚未求解")

            if print_device_info():
                try:
                    pipeline = init_rgb_handeye_pipeline()
                    self._worker_status(camera="运行中")
                    print("[GUI] RGB-only相机流已启动。请连接机械臂，然后采集不同姿态样本。")
                except Exception as exc:
                    from .camera import format_orbbec_error_hint

                    print(format_orbbec_error_hint(exc))
                    self._worker_status(camera="相机初始化失败")
            else:
                self._worker_status(camera="相机不可用")
                print("[GUI] 相机不可用；仍可进行网络诊断或连接机械臂。")

            while not self.stop_event.is_set():
                next_index, samples = self._drain_worker_commands(
                    pipeline, board, dictionary, next_index, samples,
                )
                if AUBO_SESSION.connected and time.monotonic() - self._last_pose_poll >= 1.0:
                    self._last_pose_poll = time.monotonic()
                    try:
                        self._publish_controller_pose(AUBO_SESSION.read_pose_snapshot())
                    except Exception as exc:
                        self._worker_status(controller_pose=f"读取失败：{exc}", controller_offset="未读取")
                if pipeline is None:
                    time.sleep(0.15)
                    continue

                bundle = get_rgb_frame_bundle(pipeline)
                if bundle is None:
                    time.sleep(0.01)
                    continue
                color_bgr = bundle.color_bgr
                pose_result = estimate_rgb_board_pose(color_bgr, bundle.intrinsics, board, dictionary)
                quality = evaluate_image_quality(color_bgr, pose_result)
                self._put_display(pose_result.rgb_overlay)
                self._worker_status(samples=str(len(samples)), quality=f"{quality.score:.1f}/100 {quality.label}")
        except Exception as exc:
            print("[GUI-ERROR] 工作线程异常：", exc)
            print(traceback.format_exc())
            self._worker_status(camera="异常")
        finally:
            try:
                if pipeline is not None:
                    pipeline.stop()
            except Exception:
                pass
            close_aubo_session()
            self.worker_sample_dir = None
            self._worker_status(camera="已停止", robot="已断开", controller_pose="未连接", controller_offset="未读取")
            print("[GUI] 已停止相机并断开 AUBO 会话。")

    # ---------------- UI 刷新 ----------------

    def _poll_queues(self) -> None:
        latest: np.ndarray | None = None
        while True:
            try:
                latest = self.display_queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self._show_image(latest)

        while True:
            try:
                text = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.insert(tk.END, text)
            self.log_text.see(tk.END)

        while True:
            try:
                status = self.status_queue.get_nowait()
            except queue.Empty:
                break
            if "camera" in status:
                self.camera_status_var.set(status["camera"])
            if "robot" in status:
                self.robot_status_var.set(status["robot"])
            if "samples" in status:
                self.sample_status_var.set(status["samples"])
            if "quality" in status:
                self.quality_status_var.set(status["quality"])
            if "output" in status:
                self.output_status_var.set(status["output"])
            if "result" in status:
                self.result_var.set(status["result"])
            if "controller_pose" in status:
                self.controller_pose_var.set(status["controller_pose"])
            if "controller_offset" in status:
                self.controller_offset_var.set(status["controller_offset"])
        self._update_button_state()
        self.after(80, self._poll_queues)

    def _show_image(self, image_bgr: np.ndarray) -> None:
        label_w = max(640, int(self.video_label.winfo_width()))
        label_h = max(360, int(self.video_label.winfo_height()))
        h, w = image_bgr.shape[:2]
        scale = min(label_w / max(1, w), label_h / max(1, h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if Image is not None and ImageTk is not None:
            self.photo_image = ImageTk.PhotoImage(Image.fromarray(rgb))
        else:
            import base64

            ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if not ok:
                return
            data = base64.b64encode(encoded.tobytes()).decode("ascii")
            self.photo_image = tk.PhotoImage(data=data)
        self.video_label.configure(image=self.photo_image)

    def _update_button_state(self) -> None:
        running = self.worker_alive()
        self.start_btn.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_btn.configure(state=tk.NORMAL if running else tk.DISABLED)
        state = tk.NORMAL if running else tk.DISABLED
        for button in (
            self.connect_btn, self.disconnect_btn, self.capture_btn,
            self.solve_btn, self.maintenance_btn,
        ):
            button.configure(state=state)

    def on_close(self) -> None:
        self.stop_event.set()


class HandEyeGuiApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AUBO 眼在手 RGB-PnP 手眼标定")
        self.geometry("1520x920")
        self.minsize(1200, 760)
        self.panel = HandEyeGuiPanel(self, autostart=True)
        self.panel.pack(fill=tk.BOTH, expand=True)
        self.log_queue = self.panel.log_queue
        self.stop_event = self.panel.stop_event
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self) -> None:
        self.panel.on_close()
        self.after(250, self.destroy)


def main_gui() -> None:
    app = HandEyeGuiApp()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = GuiLogWriter(app.log_queue, old_stdout)  # type: ignore[assignment]
    sys.stderr = GuiLogWriter(app.log_queue, old_stderr)  # type: ignore[assignment]
    try:
        app.mainloop()
    finally:
        app.stop_event.set()
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def main() -> None:
    main_gui()
