#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""工作台中的两阶段孔定位页面。

定位计算继续复用 ``run_yolo_eye_in_hand_optimized.py``，但以后台子进程运行：
Tk 主线程不会被相机、YOLO 或机器人运动等待阻塞。大幅运动的 ``m`` 确认改由
本页面的“确认当前运动”按钮发送；小幅闭环仍由原流程自动执行。
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable


PROJECT_DIR = Path(__file__).resolve().parent.parent
LOCALIZATION_SCRIPT = PROJECT_DIR / "run_yolo_eye_in_hand_optimized.py"
DEFAULT_MODEL = Path(r"C:\MM\models\small_silu.pt")
DEFAULT_HANDEYE = Path(r"C:\MM\aubo_tools\data\e7_candidates\e7_handeye_candidate_current.json")
RUNS_DIR = PROJECT_DIR.parent / "data" / "hole_localization_runs"


class HoleLocalizationPanel(ttk.Frame):
    """两阶段 YOLO 孔定位 GUI；连接参数从工作台顶栏实时读取。"""

    def __init__(self, master: tk.Misc, connection_provider: Callable[[], dict[str, Any]]) -> None:
        super().__init__(master, padding=10)
        self.connection_provider = connection_provider
        self.log_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.run_started_at = 0.0
        self.waiting_confirmation = False
        self._poll_id: str | None = None

        self.model_var = tk.StringVar(value=str(DEFAULT_MODEL))
        self.handeye_var = tk.StringVar(value=str(DEFAULT_HANDEYE))
        self.confidence_var = tk.StringVar(value="0.35")
        self.coarse_height_var = tk.StringVar(value="340")
        self.fine_height_var = tk.StringVar(value="260")
        self.coarse_frames_var = tk.StringVar(value="15")
        self.fine_frames_var = tk.StringVar(value="30")
        # 默认三孔：界面会要求逐个点击三个孔，第一孔作为粗定位参考。
        self.hole_count_var = tk.StringVar(value="3")
        self.speed_var = tk.StringVar(value="0.03")
        self.acc_var = tk.StringVar(value="0.10")
        self.execute_var = tk.BooleanVar(value=False)
        self.experimental_var = tk.BooleanVar(value=True)
        self.final_xy_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="待开始：默认仅预览，不会下发机器人运动")
        self.motion_var = tk.StringVar(value="无待确认运动")
        self.result_var = tk.StringVar(value="尚无本次结果")

        self._build()
        self._poll_queue()

    def _build(self) -> None:
        config = ttk.LabelFrame(self, text="两阶段定位参数", padding=8)
        config.pack(fill=tk.X)
        config.columnconfigure(1, weight=1)
        config.columnconfigure(4, weight=1)

        self._path_row(config, 0, "YOLO 模型", self.model_var, self._browse_model)
        self._path_row(config, 1, "手眼结果", self.handeye_var, self._browse_handeye)

        fields = ttk.Frame(config)
        fields.grid(row=2, column=0, columnspan=5, sticky="ew", pady=(6, 0))
        for col, (label, var, width) in enumerate([
            ("置信度", self.confidence_var, 7),
            ("粗定位高度 mm", self.coarse_height_var, 8),
            ("精定位高度 mm", self.fine_height_var, 8),
            ("粗定位帧", self.coarse_frames_var, 6),
            ("精定位帧", self.fine_frames_var, 6),
            ("输出孔数", self.hole_count_var, 6),
            ("速度 m/s", self.speed_var, 7),
            ("加速度 m/s²", self.acc_var, 7),
        ]):
            ttk.Label(fields, text=label).grid(row=0, column=col * 2, sticky="w", padx=(0, 3))
            ttk.Entry(fields, textvariable=var, width=width).grid(row=0, column=col * 2 + 1, sticky="w", padx=(0, 10))

        switches = ttk.Frame(config)
        switches.grid(row=3, column=0, columnspan=5, sticky="w", pady=(7, 0))
        ttk.Checkbutton(switches, text="真实运动（未勾选时仅预览）", variable=self.execute_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(switches, text="允许当前实验手眼结果", variable=self.experimental_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(switches, text="精定位后执行 TCP XY → 基坐标 Z → +Y 0.2 mm", variable=self.final_xy_var).pack(side=tk.LEFT)

        action = ttk.LabelFrame(self, text="运行控制", padding=8)
        action.pack(fill=tk.X, pady=(8, 0))
        self.start_btn = ttk.Button(action, text="开始两阶段定位", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.confirm_btn = ttk.Button(action, text="确认当前运动", command=self.confirm_motion, state=tk.DISABLED)
        self.confirm_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.cancel_btn = ttk.Button(action, text="取消待确认运动", command=self.cancel_pending_motion, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(action, text="打开结果目录", command=self.open_results_dir).pack(side=tk.LEFT)
        ttk.Label(action, textvariable=self.status_var).pack(side=tk.LEFT, padx=(16, 0))

        notice = ttk.LabelFrame(self, text="操作提示", padding=8)
        notice.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(
            notice,
            text=("1. 点击开始后，若使用真实运动，请在本页收到“待确认运动”后再点击确认。"
                  "  2. 相机选孔暂使用弹出的画面：点击目标孔后按 Enter。"
                  "  3. 小幅闭环修正自动执行；运动期间如需急停，请使用机械臂示教器。"),
            justify=tk.LEFT,
            wraplength=1150,
        ).pack(anchor="w")
        ttk.Label(notice, textvariable=self.motion_var, foreground="#a35a00").pack(anchor="w", pady=(5, 0))

        result = ttk.LabelFrame(self, text="最终结果", padding=8)
        result.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(result, textvariable=self.result_var, justify=tk.LEFT, wraplength=1150).pack(anchor="w")

        logs = ttk.LabelFrame(self, text="运行日志", padding=6)
        logs.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.log_text = tk.Text(logs, height=16, wrap="word", state=tk.DISABLED)
        scroll = ttk.Scrollbar(logs, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _path_row(self, parent: ttk.LabelFrame, row: int, label: str, variable: tk.StringVar,
                  callback: Callable[[], None]) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, columnspan=3, sticky="ew", pady=3)
        ttk.Button(parent, text="选择", command=callback).grid(row=row, column=4, sticky="w", padx=(8, 0), pady=3)

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="选择 YOLO 模型", filetypes=[("模型", "*.pt"), ("所有文件", "*.*")])
        if path:
            self.model_var.set(path)

    def _browse_handeye(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="选择手眼结果", filetypes=[("JSON", "*.json"), ("所有文件", "*.*")])
        if path:
            self.handeye_var.set(path)

    def _numbers(self) -> dict[str, float | int]:
        try:
            return {
                "confidence": float(self.confidence_var.get()),
                "coarse_height": float(self.coarse_height_var.get()),
                "fine_height": float(self.fine_height_var.get()),
                "coarse_frames": int(self.coarse_frames_var.get()),
                "fine_frames": int(self.fine_frames_var.get()),
                "speed": float(self.speed_var.get()),
                "acc": float(self.acc_var.get()),
            }
        except ValueError as exc:
            raise ValueError("定位参数必须是有效数字") from exc

    def _build_command(self) -> list[str]:
        if not LOCALIZATION_SCRIPT.is_file():
            raise FileNotFoundError(f"找不到定位脚本：{LOCALIZATION_SCRIPT}")
        model = Path(self.model_var.get().strip())
        handeye = Path(self.handeye_var.get().strip())
        if not model.is_file():
            raise FileNotFoundError(f"YOLO 模型不存在：{model}")
        if not handeye.is_file():
            raise FileNotFoundError(f"手眼结果不存在：{handeye}")
        values = self._numbers()
        connection = self.connection_provider()
        command = [
            # 无缓冲输出，确保 GUI 能在机器人开始运动前看到确认事件。
            sys.executable, "-u", str(LOCALIZATION_SCRIPT),
            "--model", str(model), "--handeye", str(handeye),
            "--confidence", str(values["confidence"]),
            "--coarse-height-mm", str(values["coarse_height"]),
            "--fine-height-mm", str(values["fine_height"]),
            "--coarse-frames", str(values["coarse_frames"]),
            "--fine-frames", str(values["fine_frames"]),
            "--hole-count", str(int(self.hole_count_var.get())),
            "--speed-m-s", str(values["speed"]), "--acc-m-s2", str(values["acc"]),
            "--robot-ip", str(connection["ip"]), "--robot-port", str(connection["port"]),
            "--robot-user", str(connection["user"]), "--robot-password", str(connection["password"]),
            "--robot-timeout-ms", str(connection["timeout_ms"]),
        ]
        command.append("--execute" if self.execute_var.get() else "--no-execute")
        command.append("--allow-experimental-handeye" if self.experimental_var.get() else "--require-validated-handeye")
        command.append("--move-final-xy" if self.final_xy_var.get() else "--no-move-final-xy")
        return command

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("定位进行中", "当前定位流程尚未结束。", parent=self)
            return
        try:
            command = self._build_command()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc), parent=self)
            return
        if self.execute_var.get() and not messagebox.askyesno(
            "确认真实运动",
            "将执行回原点及两阶段定位。大幅运动会在本页等待你的确认。\n\n确认开始吗？",
            parent=self,
        ):
            return
        self._clear_log()
        self.result_var.set("本次定位进行中…")
        self.status_var.set("正在启动定位进程…")
        self.motion_var.set("无待确认运动")
        self.waiting_confirmation = False
        self.confirm_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.DISABLED)
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(PROJECT_DIR),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            self.status_var.set("启动失败")
            messagebox.showerror("无法启动定位", str(exc), parent=self)
            return
        self.run_started_at = time.time()
        self.start_btn.configure(state=tk.DISABLED)
        threading.Thread(target=self._read_process_output, args=(self.process,), daemon=True).start()

    def _read_process_output(self, process: subprocess.Popen[str]) -> None:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                self.log_queue.put(("log", line))
                if "[MOTION_CONFIRM_REQUIRED]" in line:
                    self.log_queue.put(("confirm", line.strip()))
            code = process.wait()
            self.log_queue.put(("finished", code))
        except Exception as exc:
            self.log_queue.put(("worker_error", str(exc)))

    def confirm_motion(self) -> None:
        if not self.waiting_confirmation or self.process is None or self.process.poll() is not None:
            return
        try:
            assert self.process.stdin is not None
            self.process.stdin.write("m\n")
            self.process.stdin.flush()
        except OSError as exc:
            messagebox.showerror("确认发送失败", str(exc), parent=self)
            return
        self.waiting_confirmation = False
        self.confirm_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.DISABLED)
        self.motion_var.set("已确认，等待机器人完成当前动作…")

    def cancel_pending_motion(self) -> None:
        if not self.waiting_confirmation or self.process is None or self.process.poll() is not None:
            return
        try:
            assert self.process.stdin is not None
            self.process.stdin.write("\n")
            self.process.stdin.flush()
        except OSError as exc:
            messagebox.showerror("取消发送失败", str(exc), parent=self)
            return
        self.waiting_confirmation = False
        self.confirm_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.DISABLED)
        self.motion_var.set("已取消待确认运动，流程将安全结束。")

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                    self.status_var.set("定位流程运行中…")
                elif kind == "confirm":
                    self.waiting_confirmation = True
                    self.confirm_btn.configure(state=tk.NORMAL)
                    self.cancel_btn.configure(state=tk.NORMAL)
                    self.motion_var.set("待确认运动：请核对日志中的当前位置和目标位置，然后点击“确认当前运动”。")
                elif kind == "finished":
                    self._finished(int(payload))
                elif kind == "worker_error":
                    self._append_log(f"[GUI] 读取定位进程失败：{payload}\n")
        except queue.Empty:
            pass
        self._poll_id = self.after(80, self._poll_queue)

    def _finished(self, code: int) -> None:
        self.waiting_confirmation = False
        self.confirm_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.DISABLED)
        self.start_btn.configure(state=tk.NORMAL)
        self.process = None
        if code == 0:
            self.status_var.set("定位完成")
            self._show_latest_result()
        else:
            self.status_var.set(f"定位结束，退出码={code}")
            self.result_var.set("本次定位未通过质量门或被取消；请查看运行日志和结果目录。")

    def _show_latest_result(self) -> None:
        reports = sorted(
            (path for path in RUNS_DIR.glob("two-stage-*/report.json") if path.stat().st_mtime >= self.run_started_at - 2.0),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not reports:
            self.result_var.set("流程完成，但未找到本次 JSON 报告。")
            return
        try:
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            final = report.get("final_result", {})
            holes = final.get("holes")
            if isinstance(holes, list) and holes:
                lines = [
                    f"报告：{reports[0].parent}",
                    f"三孔共享精拍 TCP：{self._format_vector(final.get('shared_fine_tcp_pose_m_rad'))}",
                ]
                for item in holes:
                    lines.append(
                        f"孔 {item.get('hole_id', '-')}：中心={self._format_vector(item.get('hole_center_base_mm'))} "
                        f"法向={self._format_vector(item.get('plane_normal_toward_camera_base'))} "
                        f"姿态={self._format_vector(item.get('hole_pose_m_rad'))} "
                        f"孔径={item.get('matched_diameter_mm', '-') } mm"
                    )
                self.result_var.set("\n".join(lines))
                return
            center = final.get("hole_center_base_mm")
            normal = final.get("plane_normal_toward_camera_base")
            tcp = final.get("final_tcp_pose_m_rad")
            diameter = final.get("matched_diameter_mm")
            self.result_var.set(
                f"报告：{reports[0].parent}\n"
                f"孔中心（基坐标 mm）：{self._format_vector(center)}    法向：{self._format_vector(normal)}\n"
                f"最终 TCP（m/rad）：{self._format_vector(tcp)}    孔径类别：{diameter} mm"
            )
        except Exception as exc:
            self.result_var.set(f"读取本次报告失败：{exc}")

    @staticmethod
    def _format_vector(value: Any) -> str:
        if not isinstance(value, (list, tuple)):
            return "-"
        try:
            return "[" + ", ".join(f"{float(item):.3f}" for item in value) + "]"
        except (TypeError, ValueError):
            return str(value)

    def open_results_dir(self) -> None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(RUNS_DIR))  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("打开目录失败", str(exc), parent=self)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def on_close(self) -> None:
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
        # 不会在机器人自动运动期间强杀子进程；若正等待确认，可安全发送取消。
        if self.waiting_confirmation:
            self.cancel_pending_motion()
