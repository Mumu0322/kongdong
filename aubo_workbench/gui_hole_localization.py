#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""工作台中的两阶段孔定位和视野偏移测试页面。

定位计算继续复用两个独立诊断入口，但都以后台子进程运行：
Tk 主线程不会被相机、YOLO 或机器人运动等待阻塞。流程仅在开始检测下一个已选孔时暂停，
由本页面的“开始检测下一个孔”按钮发送继续指令，其余运动自动执行。
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
OFFSET_TEST_SCRIPT = PROJECT_DIR / "run_coarse_to_fine_offset_test.py"
DEFAULT_MODEL = Path(r"C:\MM\models\small_silu.pt")
DEFAULT_HANDEYE = Path(r"C:\MM\aubo_tools\data\e7_candidates\e7_handeye_candidate_current.json")
RUNS_DIR = PROJECT_DIR.parent / "data" / "hole_localization_runs"


class HoleLocalizationPanel(ttk.Frame):
    """两阶段 YOLO 孔定位和单孔视野偏移测试 GUI。"""

    def __init__(self, master: tk.Misc, connection_provider: Callable[[], dict[str, Any]]) -> None:
        super().__init__(master, padding=10)
        self.connection_provider = connection_provider
        self.log_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.process_mode = "two_stage"
        self.run_started_at = 0.0
        self.waiting_confirmation = False
        self.confirmation_kind = ""
        self._poll_id: str | None = None

        self.model_var = tk.StringVar(value=str(DEFAULT_MODEL))
        self.handeye_var = tk.StringVar(value=str(DEFAULT_HANDEYE))
        self.confidence_var = tk.StringVar(value="0.35")
        self.coarse_height_var = tk.StringVar(value="340")
        self.fine_height_var = tk.StringVar(value="260")
        self.coarse_frames_var = tk.StringVar(value="15")
        self.fine_frames_var = tk.StringVar(value="30")
        # 初始孔数由相机窗口中的点击数量决定，按 Enter 结束选择。
        self.speed_var = tk.StringVar(value="0.08")
        self.acc_var = tk.StringVar(value="0.25")
        self.transit_speed_var = tk.StringVar(value="0.15")
        self.transit_acc_var = tk.StringVar(value="0.45")
        self.approach_speed_var = tk.StringVar(value="0.12")
        self.approach_acc_var = tk.StringVar(value="0.35")
        self.offset_radii_var = tk.StringVar(value="0 5 10 15 20")
        self.offset_angles_var = tk.StringVar(value="0 45 90 135 180 225 270 315")
        self.execute_var = tk.BooleanVar(value=False)
        self.experimental_var = tk.BooleanVar(value=True)
        self.final_xy_var = tk.BooleanVar(value=True)
        self.final_target_mode_var = tk.StringVar(value="机械爪模式")
        self.include_final_motion_var = tk.BooleanVar(value=False)
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
            ("精确速度 m/s", self.speed_var, 7),
            ("加速度 m/s²", self.acc_var, 7),
        ]):
            ttk.Label(fields, text=label).grid(row=0, column=col * 2, sticky="w", padx=(0, 3))
            ttk.Entry(fields, textvariable=var, width=width).grid(row=0, column=col * 2 + 1, sticky="w", padx=(0, 10))

        transit_fields = ttk.Frame(config)
        transit_fields.grid(row=3, column=0, columnspan=5, sticky="w", pady=(6, 0))
        for col, (label, var, width) in enumerate([
            ("安全过渡速度 m/s", self.transit_speed_var, 7),
            ("安全过渡加速度 m/s²", self.transit_acc_var, 7),
        ]):
            ttk.Label(transit_fields, text=label).grid(row=0, column=col * 2, sticky="w", padx=(0, 3))
            ttk.Entry(transit_fields, textvariable=var, width=width).grid(
                row=0, column=col * 2 + 1, sticky="w", padx=(0, 10),
            )

        approach_fields = ttk.Frame(config)
        approach_fields.grid(row=4, column=0, columnspan=5, sticky="w", pady=(6, 0))
        for col, (label, var, width) in enumerate([
            ("非接触接近速度 m/s", self.approach_speed_var, 7),
            ("非接触接近加速度 m/s²", self.approach_acc_var, 7),
        ]):
            ttk.Label(approach_fields, text=label).grid(row=0, column=col * 2, sticky="w", padx=(0, 3))
            ttk.Entry(approach_fields, textvariable=var, width=width).grid(
                row=0, column=col * 2 + 1, sticky="w", padx=(0, 10),
            )

        offset_fields = ttk.Frame(config)
        offset_fields.grid(row=5, column=0, columnspan=5, sticky="w", pady=(6, 0))
        for col, (label, var, width) in enumerate([
            ("偏移测试半径 mm", self.offset_radii_var, 24),
            ("偏移测试方向 °", self.offset_angles_var, 38),
        ]):
            ttk.Label(offset_fields, text=label).grid(
                row=0, column=col * 2, sticky="w", padx=(0, 3),
            )
            ttk.Entry(offset_fields, textvariable=var, width=width).grid(
                row=0, column=col * 2 + 1, sticky="w", padx=(0, 14),
            )

        switches = ttk.Frame(config)
        switches.grid(row=6, column=0, columnspan=5, sticky="w", pady=(7, 0))
        ttk.Label(switches, text="最终点模式").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(
            switches,
            textvariable=self.final_target_mode_var,
            values=("机械爪模式", "平常模式"),
            state="readonly",
            width=13,
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(switches, text="真实运动（未勾选时仅预览）", variable=self.execute_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(switches, text="允许当前实验手眼结果", variable=self.experimental_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(switches, text="精定位后执行 TCP XY → 基坐标 Z → +Y 0.2 mm", variable=self.final_xy_var).pack(side=tk.LEFT)
        ttk.Checkbutton(
            switches, text="偏移测试执行最终 XY → Z → +Y 0.2 mm（实机）",
            variable=self.include_final_motion_var,
        ).pack(side=tk.LEFT, padx=(16, 0))

        action = ttk.LabelFrame(self, text="运行控制", padding=8)
        action.pack(fill=tk.X, pady=(8, 0))
        self.start_btn = ttk.Button(action, text="开始两阶段定位", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.offset_start_btn = ttk.Button(
            action, text="偏移容忍度测试（单孔）", command=self.start_offset_test,
        )
        self.offset_start_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.confirm_btn = ttk.Button(action, text="开始检测下一个孔", command=self.confirm_motion, state=tk.DISABLED)
        self.confirm_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.mark_error_btn = ttk.Button(
            action, text="标记当前点有误差并继续",
            command=self.mark_current_point_error, state=tk.DISABLED,
        )
        self.mark_error_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.cancel_btn = ttk.Button(action, text="停止流程", command=self.cancel_pending_motion, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(action, text="打开结果目录", command=self.open_results_dir).pack(side=tk.LEFT)
        ttk.Label(action, textvariable=self.status_var).pack(side=tk.LEFT, padx=(16, 0))

        notice = ttk.LabelFrame(self, text="操作提示", padding=8)
        notice.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(
            notice,
            text=("1. 相机初始画面中选择所有目标孔后按 Enter。"
                  "  2. 每个孔完成后，仅在开始检测下一个已选孔时点击继续。"
                  "  3. 其余运动自动执行；如需急停，请使用机械臂示教器。"),
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
                "transit_speed": float(self.transit_speed_var.get()),
                "transit_acc": float(self.transit_acc_var.get()),
                "approach_speed": float(self.approach_speed_var.get()),
                "approach_acc": float(self.approach_acc_var.get()),
            }
        except ValueError as exc:
            raise ValueError("定位参数必须是有效数字") from exc

    @staticmethod
    def _float_list(value: str, label: str) -> list[float]:
        try:
            values = [float(item) for item in value.replace(",", " ").split()]
        except ValueError as exc:
            raise ValueError(f"{label}必须是空格或逗号分隔的数字") from exc
        if not values:
            raise ValueError(f"{label}不能为空")
        return values

    def _build_command(self, mode: str = "two_stage") -> list[str]:
        script = LOCALIZATION_SCRIPT if mode == "two_stage" else OFFSET_TEST_SCRIPT
        if not script.is_file():
            raise FileNotFoundError(f"找不到定位脚本：{script}")
        model = Path(self.model_var.get().strip())
        handeye = Path(self.handeye_var.get().strip())
        if (mode != "offset" or self.execute_var.get()) and not model.is_file():
            raise FileNotFoundError(f"YOLO 模型不存在：{model}")
        if (mode != "offset" or self.execute_var.get()) and not handeye.is_file():
            raise FileNotFoundError(f"手眼结果不存在：{handeye}")
        values = self._numbers()
        connection = self.connection_provider()
        command = [
            # 无缓冲输出，确保 GUI 能在机器人开始运动前看到确认事件。
            sys.executable, "-u", str(script),
            "--model", str(model), "--handeye", str(handeye),
            "--confidence", str(values["confidence"]),
            "--coarse-height-mm", str(values["coarse_height"]),
            "--fine-height-mm", str(values["fine_height"]),
            "--coarse-frames", str(values["coarse_frames"]),
            "--fine-frames", str(values["fine_frames"]),
            "--speed-m-s", str(values["speed"]), "--acc-m-s2", str(values["acc"]),
            "--transit-speed-m-s", str(values["transit_speed"]),
            "--transit-acc-m-s2", str(values["transit_acc"]),
            "--approach-speed-m-s", str(values["approach_speed"]),
            "--approach-acc-m-s2", str(values["approach_acc"]),
            "--robot-ip", str(connection["ip"]), "--robot-port", str(connection["port"]),
            "--robot-user", str(connection["user"]), "--robot-password", str(connection["password"]),
            "--robot-timeout-ms", str(connection["timeout_ms"]),
        ]
        command.append("--execute" if self.execute_var.get() else "--no-execute")
        command.append(
            "--allow-experimental-handeye"
            if self.experimental_var.get() else "--require-validated-handeye"
        )
        if mode == "offset":
            if self.include_final_motion_var.get() and not self.execute_var.get():
                raise ValueError("偏移测试的最终XY/Z/Y动作必须勾选“真实运动”")
            command.extend([
                "--radii-mm",
                *[str(value) for value in self._float_list(self.offset_radii_var.get(), "偏移测试半径")],
                "--angles-deg",
                *[str(value) for value in self._float_list(self.offset_angles_var.get(), "偏移测试方向")],
            ])
            if self.execute_var.get():
                command.append("--start-confirmed")
            if self.include_final_motion_var.get():
                command.append("--include-final-motion")
        else:
            final_mode = {
                "机械爪模式": "gripper",
                "平常模式": "normal",
            }.get(self.final_target_mode_var.get())
            if final_mode is None:
                raise ValueError("最终点模式必须选择“机械爪模式”或“平常模式”")
            command.extend(["--final-target-mode", final_mode])
            command.append("--move-final-xy" if self.final_xy_var.get() else "--no-move-final-xy")
        return command

    def start(self) -> None:
        self._start_process("two_stage")

    def start_offset_test(self) -> None:
        self._start_process("offset")

    def _start_process(self, mode: str) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("定位进行中", "当前定位流程尚未结束。", parent=self)
            return
        try:
            command = self._build_command(mode)
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc), parent=self)
            return
        offset_confirmation = (
            "将执行最终XY、降Z、基坐标Y+0.2 mm，并在每个采样点完成后安全回升并返回中心。"
            if self.include_final_motion_var.get() else
            "不会执行最终XY、最终Z或基坐标Y+0.2 mm动作。"
        )
        if self.execute_var.get() and not messagebox.askyesno(
            "确认真实运动",
            (
                "将执行回原点、单孔粗定位、下降和横向偏移采集。"
                f"{offset_confirmation}\n"
                "请在相机窗口中只选择一个孔，之后测试会自动运行。\n\n确认开始吗？"
                if mode == "offset" else
                "将执行回原点及两阶段定位。除开始检测下一个已选孔外，运动会自动执行。\n\n确认开始吗？"
            ),
            parent=self,
        ):
            return
        self.process_mode = mode
        self._clear_log()
        self.result_var.set(
            "偏移容忍度测试进行中…" if mode == "offset" else "本次定位进行中…"
        )
        self.status_var.set(
            "正在启动偏移测试进程…" if mode == "offset" else "正在启动定位进程…"
        )
        self.motion_var.set("无待确认运动")
        self.waiting_confirmation = False
        self.confirmation_kind = ""
        self.confirm_btn.configure(text="开始检测下一个孔")
        self.confirm_btn.configure(state=tk.DISABLED)
        self.mark_error_btn.configure(state=tk.DISABLED)
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
        self.offset_start_btn.configure(state=tk.DISABLED)
        threading.Thread(target=self._read_process_output, args=(self.process,), daemon=True).start()

    def _read_process_output(self, process: subprocess.Popen[str]) -> None:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                self.log_queue.put(("log", line))
                if (
                    "[MOTION_CONFIRM_REQUIRED]" in line
                    or "[NEXT_HOLE_CONFIRM_REQUIRED]" in line
                ):
                    self.log_queue.put(("confirm", line.strip()))
                elif "[OFFSET_TEST_CONFIRM_REQUIRED]" in line:
                    self.log_queue.put(("offset_start_confirm", line.strip()))
                elif "[OFFSET_NEXT_CONFIRM_REQUIRED]" in line:
                    self.log_queue.put(("offset_next_confirm", line.strip()))
                elif "[OFFSET_TARGET_CONFIRM_REQUIRED]" in line:
                    self.log_queue.put(("offset_target_confirm", line.strip()))
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
        confirmation_kind = self.confirmation_kind
        self.confirmation_kind = ""
        self.confirm_btn.configure(state=tk.DISABLED, text="开始检测下一个孔")
        self.mark_error_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.DISABLED)
        if confirmation_kind == "offset_start":
            self.motion_var.set("已确认，开始偏移测试…")
        elif confirmation_kind == "offset_next":
            self.motion_var.set("已确认，开始下一个偏移点…")
        elif confirmation_kind == "offset_target":
            self.motion_var.set("已确认，回升并继续偏移测试…")
        else:
            self.motion_var.set("已确认，开始检测下一个孔…")

    def mark_current_point_error(self) -> None:
        if (
            not self.waiting_confirmation
            or self.confirmation_kind != "offset_target"
            or self.process is None
            or self.process.poll() is not None
        ):
            return
        try:
            assert self.process.stdin is not None
            self.process.stdin.write("e\n")
            self.process.stdin.flush()
        except OSError as exc:
            messagebox.showerror("误差标记发送失败", str(exc), parent=self)
            return
        self.waiting_confirmation = False
        self.confirmation_kind = ""
        self.confirm_btn.configure(state=tk.DISABLED, text="开始检测下一个孔")
        self.mark_error_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.DISABLED)
        self.motion_var.set("已标记当前点有误差，回升后继续偏移测试…")

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
        self.confirmation_kind = ""
        self.confirm_btn.configure(state=tk.DISABLED)
        self.mark_error_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.DISABLED)
        self.confirm_btn.configure(text="开始检测下一个孔")
        self.motion_var.set("已取消待确认运动，流程将安全结束。")

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                    self.status_var.set(
                        "偏移容忍度测试运行中…"
                        if self.process_mode == "offset" else "定位流程运行中…"
                    )
                elif kind == "confirm":
                    self.waiting_confirmation = True
                    self.confirmation_kind = "hole"
                    self.confirm_btn.configure(state=tk.NORMAL)
                    self.mark_error_btn.configure(state=tk.DISABLED)
                    self.cancel_btn.configure(state=tk.NORMAL)
                    self.confirm_btn.configure(text="开始检测下一个孔")
                    self.motion_var.set("等待开始下一个孔：请点击“开始检测下一个孔”继续。")
                elif kind == "offset_start_confirm":
                    self.waiting_confirmation = True
                    self.confirmation_kind = "offset_start"
                    self.confirm_btn.configure(state=tk.NORMAL, text="开始偏移测试")
                    self.mark_error_btn.configure(state=tk.DISABLED)
                    self.cancel_btn.configure(state=tk.NORMAL)
                    self.motion_var.set("等待开始偏移测试：请点击“开始偏移测试”继续。")
                elif kind == "offset_next_confirm":
                    self.waiting_confirmation = True
                    self.confirmation_kind = "offset_next"
                    self.confirm_btn.configure(state=tk.NORMAL, text="开始下一个偏移点")
                    self.mark_error_btn.configure(state=tk.DISABLED)
                    self.cancel_btn.configure(state=tk.NORMAL)
                    self.motion_var.set("当前偏移点已完成：请点击“开始下一个偏移点”继续。")
                elif kind == "offset_target_confirm":
                    self.waiting_confirmation = True
                    self.confirmation_kind = "offset_target"
                    self.confirm_btn.configure(state=tk.NORMAL, text="确认并继续")
                    self.mark_error_btn.configure(state=tk.NORMAL)
                    self.cancel_btn.configure(state=tk.NORMAL)
                    self.motion_var.set("已到达最终目标点：请确认后回升并继续。")
                elif kind == "finished":
                    self._finished(int(payload))
                elif kind == "worker_error":
                    self._append_log(f"[GUI] 读取定位进程失败：{payload}\n")
        except queue.Empty:
            pass
        self._poll_id = self.after(80, self._poll_queue)

    def _finished(self, code: int) -> None:
        self.waiting_confirmation = False
        self.confirmation_kind = ""
        self.confirm_btn.configure(state=tk.DISABLED)
        self.mark_error_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.DISABLED)
        self.confirm_btn.configure(text="开始检测下一个孔")
        self.start_btn.configure(state=tk.NORMAL)
        self.offset_start_btn.configure(state=tk.NORMAL)
        self.process = None
        if code == 0:
            self.status_var.set("偏移测试完成" if self.process_mode == "offset" else "定位完成")
            self._show_latest_result()
        else:
            self.status_var.set(f"定位结束，退出码={code}")
            self.result_var.set("本次定位未通过质量门或被取消；请查看运行日志和结果目录。")

    def _show_latest_result(self) -> None:
        if self.process_mode == "offset":
            reports = sorted(
                (
                    path for path in RUNS_DIR.glob("coarse-to-fine-offset-*/report.json")
                    if path.stat().st_mtime >= self.run_started_at - 2.0
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if not reports:
                self.result_var.set("偏移测试完成，但未找到本次 JSON 报告。")
                return
            try:
                report = json.loads(reports[0].read_text(encoding="utf-8"))
                summary = report.get("summary", {})
                lines = [
                    f"报告：{reports[0].parent}\n"
                    f"采样点：{report.get('sample_count', len(report.get('plan', [])))}    "
                    f"最大支持半径：{summary.get('max_supported_radius_mm', '-')} mm    "
                    f"严格通过半径：{summary.get('max_strict_radius_mm', '-')} mm",
                    "绿色=当前精定位质量门通过；橙色=仅严格门未通过但降级通过；红色=失败",
                ]
                manual_error_ids = summary.get("manual_error_sample_ids", [])
                if manual_error_ids:
                    lines.append(
                        "人工标记有误差的点："
                        + ", ".join(str(item) for item in manual_error_ids)
                    )
                motion_summary = report.get("final_motion_summary", {})
                if motion_summary.get("enabled"):
                    step_labels = {
                        "final_xy": "最终XY",
                        "final_z": "最终Z",
                        "final_y_plus_0_2": "最终+Y 0.2 mm",
                    }
                    lines.append("最终动作到位误差（实际TCP−规划TCP）：")
                    for step_name, label in step_labels.items():
                        step = motion_summary.get("steps", {}).get(step_name, {})
                        max_error = step.get("max_translation_error_mm")
                        mean_error = step.get("mean_translation_error_mm")
                        if max_error is not None:
                            lines.append(
                                f"{label}：最大 {float(max_error):.3f} mm，"
                                f"平均 {float(mean_error):.3f} mm"
                            )
                self.result_var.set("\n".join(lines))
            except Exception as exc:
                self.result_var.set(f"读取偏移测试报告失败：{exc}")
            return
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
                deferred_count = int(final.get("deferred_count", 0) or 0)
                completed_count = int(final.get("completed_count", len(holes) - deferred_count) or 0)
                if deferred_count:
                    self.status_var.set(
                        f"定位完成，但有 {deferred_count} 个孔延期，成功 {completed_count} 个"
                    )
                lines = [
                    f"报告：{reports[0].parent}",
                    f"顺序处理孔数：{final.get('hole_count', len(holes))}    "
                    f"成功：{completed_count}    延期：{deferred_count}    "
                    f"最终 TCP：{self._format_vector(final.get('final_tcp_pose_m_rad'))}",
                ]
                for item in holes:
                    quality_status = item.get("fine_quality_status", "strict")
                    status_text = (
                        f"状态={item.get('status', 'completed')}/{quality_status} "
                        if item.get("status") != "completed" or quality_status != "strict"
                        else ""
                    )
                    lines.append(
                        f"孔 {item.get('hole_id', '-')}：{status_text}"
                        f"中心={self._format_vector(item.get('hole_center_base_mm'))} "
                        f"法向={self._format_vector(item.get('plane_normal_toward_camera_base'))} "
                        f"姿态={self._format_vector(item.get('hole_pose_m_rad'))} "
                        f"孔径={item.get('matched_diameter_mm', '-') } mm "
                        f"跟踪={item.get('tracking_identity', '-')}"
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
