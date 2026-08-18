#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z-ERG-20C 旋转电爪 GUI 控制页。

夹爪驱动仍保存在 ``C:/MM/aubo_tools/JiaZhua/z_erg_20c.py``，本模块只负责：

* 读取串口/从站/波特率等连接参数；
* 在后台线程执行 Modbus 操作，避免阻塞 Tk 主线程；
* 展示状态，并在动作失败或夹爪异常时停止当前 GUI 操作。

本页不会自动跟随机械臂运动，也不会在 GUI 启动时连接夹爪。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, X, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Callable


DRIVER_PATH = Path(__file__).resolve().parents[2] / "JiaZhua" / "z_erg_20c.py"
BAUDRATES = (9600, 19200, 38400, 57600, 115200, 153600, 256000)
CURRENT_POLL_INTERVAL_MS = 250

_driver_module: Any | None = None


def load_gripper_driver() -> Any:
    """按固定工作区路径懒加载夹爪驱动，GUI启动时不强制依赖串口。"""

    global _driver_module
    if _driver_module is not None:
        return _driver_module
    if not DRIVER_PATH.exists():
        raise FileNotFoundError(f"夹爪驱动文件不存在：{DRIVER_PATH}")
    module_name = "_aubo_z_erg_20c_driver"
    spec = importlib.util.spec_from_file_location(module_name, DRIVER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载夹爪驱动：{DRIVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _driver_module = module
    return module


class GripperControlPanel(ttk.Frame):
    """Z-ERG-20C 独立控制页。所有串口操作都在后台线程执行。"""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, padding=10)
        self.gripper: Any | None = None
        self.busy = False
        self.closing = False
        self.buttons: list[ttk.Button] = []

        self.port_var = tk.StringVar(value="COM5")
        self.slave_var = tk.StringVar(value="1")
        self.baud_var = tk.StringVar(value="115200")
        self.timeout_var = tk.StringVar(value="1.0")
        self.retries_var = tk.StringVar(value="3")
        self.grip_speed_var = tk.StringVar(value="50")
        self.grip_current_var = tk.StringVar(value="0.2")
        self.grip_position_var = tk.StringVar(value="10")
        self.rotation_speed_var = tk.StringVar(value="720")
        self.rotation_current_var = tk.StringVar(value="0.8")
        self.rotation_angle_var = tk.StringVar(value="90")
        self.rotation_relative_var = tk.StringVar(value="360")
        self.status_var = tk.StringVar(value="未连接")
        self.grip_feedback_var = tk.StringVar(value="夹持实际电流：-- A")
        self.rotation_feedback_var = tk.StringVar(value="旋转实际电流：-- A")
        self.current_monitor_var = tk.StringVar(value="电流监视：未连接")
        self.driver_var = tk.StringVar(value=f"驱动：{DRIVER_PATH}")

        self._current_poll_job: str | None = None
        self._current_poll_inflight = False
        self._current_poll_token = 0
        self._last_current_poll_error: str | None = None

        self._build()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        connection = ttk.LabelFrame(self, text="夹爪连接参数", padding=8)
        connection.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for column in (1, 3, 5, 7):
            connection.columnconfigure(column, weight=1)

        self._entry(connection, 0, 0, "串口", self.port_var, 12)
        self._entry(connection, 0, 2, "从站ID", self.slave_var, 8)
        ttk.Label(connection, text="波特率").grid(row=0, column=4, sticky="w", padx=(12, 4), pady=3)
        ttk.Combobox(
            connection,
            textvariable=self.baud_var,
            values=[str(value) for value in BAUDRATES],
            state="readonly",
            width=10,
        ).grid(row=0, column=5, sticky="ew", padx=(0, 12), pady=3)
        self._entry(connection, 0, 6, "超时(s)", self.timeout_var, 8)
        self._entry(connection, 0, 8, "重试", self.retries_var, 6)

        ttk.Label(connection, textvariable=self.driver_var, foreground="#555555").grid(
            row=1, column=0, columnspan=7, sticky="w", pady=(5, 0)
        )
        self.connect_button = ttk.Button(connection, text="连接并读取状态", command=self.connect_gripper)
        self.connect_button.grid(
            row=1, column=7, sticky="e", padx=(6, 0), pady=(5, 0)
        )
        self.buttons.append(self.connect_button)
        self._button(connection, 1, 8, "断开", self.disconnect_gripper)
        self._button(connection, 1, 9, "刷新状态", self.refresh_status)

        safety = ttk.LabelFrame(self, text="动作控制（不会自动随机器人运动）", padding=8)
        safety.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        safety.columnconfigure(1, weight=1)
        safety.columnconfigure(3, weight=1)
        safety.columnconfigure(5, weight=1)
        ttk.Label(
            safety,
            text="初始化前请确认夹指周围无物体；异常状态会停止当前夹爪操作。",
            foreground="#9a4d00",
        ).grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 6))
        self._button(safety, 1, 0, "初始化校准", self.initialize_gripper)
        self._button(safety, 1, 1, "电机使能", self.enable_motor)
        self._button(safety, 1, 2, "关闭电机", self.disable_motor)
        self._button(safety, 1, 3, "张开", self.open_gripper)
        self._button(safety, 1, 4, "闭合", self.close_gripper)

        grip = ttk.LabelFrame(self, text="夹持参数", padding=8)
        grip.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        for column in (1, 3, 5):
            grip.columnconfigure(column, weight=1)
        self._entry(grip, 0, 0, "速度(mm/s)", self.grip_speed_var, 10)
        self._entry(grip, 0, 2, "电流(A)", self.grip_current_var, 10)
        self._entry(grip, 0, 4, "位置(mm)", self.grip_position_var, 10)
        self._button(grip, 0, 6, "移动到指定位置", self.move_gripper)

        rotation = ttk.LabelFrame(self, text="旋转参数", padding=8)
        rotation.grid(row=3, column=0, sticky="new", pady=(0, 8))
        for column in (1, 3, 5, 7):
            rotation.columnconfigure(column, weight=1)
        self._entry(rotation, 0, 0, "速度(°/s)", self.rotation_speed_var, 10)
        self._entry(rotation, 0, 2, "电流(A)", self.rotation_current_var, 10)
        self._entry(rotation, 0, 4, "绝对角度(°)", self.rotation_angle_var, 10)
        self._button(rotation, 0, 6, "绝对旋转", self.rotate_absolute)
        self._entry(rotation, 1, 0, "相对角度(°)", self.rotation_relative_var, 10)
        self._button(rotation, 1, 2, "相对旋转", self.rotate_relative)

        state = ttk.LabelFrame(self, text="状态与日志", padding=8)
        state.grid(row=4, column=0, sticky="nsew")
        state.columnconfigure(0, weight=1)
        state.columnconfigure(1, weight=1)
        state.columnconfigure(2, weight=1)
        state.rowconfigure(2, weight=1)
        feedback = ttk.Frame(state)
        feedback.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 5))
        feedback.columnconfigure(0, weight=1)
        feedback.columnconfigure(1, weight=1)
        feedback.columnconfigure(2, weight=1)
        ttk.Label(feedback, textvariable=self.status_var, foreground="#1f5f99").grid(
            row=0, column=0, sticky="w", padx=(0, 12)
        )
        ttk.Label(feedback, textvariable=self.grip_feedback_var, foreground="#176b3a").grid(
            row=0, column=1, sticky="w", padx=(0, 12)
        )
        ttk.Label(feedback, textvariable=self.rotation_feedback_var, foreground="#176b3a").grid(
            row=0, column=2, sticky="w"
        )
        ttk.Label(state, textvariable=self.current_monitor_var, foreground="#666666").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(0, 5)
        )
        self.log_text = ScrolledText(state, height=10, wrap="word", state=tk.DISABLED)
        self.log_text.grid(row=2, column=0, columnspan=3, sticky="nsew")

    @staticmethod
    def _entry(parent: tk.Misc, row: int, label_column: int, label: str,
               variable: tk.StringVar, width: int) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=label_column, sticky="w", padx=(0, 4), pady=3)
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=label_column + 1, sticky="ew", padx=(0, 8), pady=3)
        return entry

    def _button(self, parent: tk.Misc, row: int, column: int, text: str, command) -> ttk.Button:
        button = ttk.Button(parent, text=text, command=command)
        button.grid(row=row, column=column, padx=(4, 0), pady=3, sticky="ew")
        self.buttons.append(button)
        return button

    # ------------------------------------------------------------------
    # 参数和日志
    # ------------------------------------------------------------------
    def _connection_settings(self) -> dict[str, Any]:
        port = self.port_var.get().strip()
        if not port:
            raise ValueError("串口不能为空")
        slave_id = int(self.slave_var.get().strip())
        baudrate = int(self.baud_var.get().strip())
        timeout = float(self.timeout_var.get().strip())
        retries = int(self.retries_var.get().strip())
        if not 1 <= slave_id <= 247:
            raise ValueError("从站ID必须在1~247范围内")
        if baudrate not in BAUDRATES:
            raise ValueError(f"波特率必须是：{BAUDRATES}")
        if timeout <= 0 or retries < 1:
            raise ValueError("超时必须大于0，重试次数必须至少为1")
        return {
            "port": port,
            "slave_id": slave_id,
            "baudrate": baudrate,
            "timeout": timeout,
            "retries": retries,
        }

    def _float(self, variable: tk.StringVar, label: str) -> float:
        try:
            return float(variable.get().strip())
        except ValueError as exc:
            raise ValueError(f"{label}必须是数字") from exc

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(END, f"{message}\n")
        self.log_text.see(END)
        self.log_text.configure(state=tk.DISABLED)

    def _show_state(self, state: Any) -> None:
        if isinstance(state, dict):
            self._update_current_feedback_from_state(state)
            self._append_log(json.dumps(state, ensure_ascii=False, indent=2))
        else:
            self._append_log(str(state))

    @staticmethod
    def _format_current(value: Any) -> str:
        try:
            return f"{float(value):.4f} A"
        except (TypeError, ValueError):
            return "-- A"

    def _update_current_feedback_from_state(self, state: dict[str, Any]) -> None:
        if "grip_current_a" in state:
            self.grip_feedback_var.set(
                f"夹持实际电流：{self._format_current(state['grip_current_a'])}"
            )
        if "rot_current_a" in state:
            self.rotation_feedback_var.set(
                f"旋转实际电流：{self._format_current(state['rot_current_a'])}"
            )

    def _reset_current_feedback(self, monitor_text: str = "电流监视：未连接") -> None:
        self.grip_feedback_var.set("夹持实际电流：-- A")
        self.rotation_feedback_var.set("旋转实际电流：-- A")
        self.current_monitor_var.set(monitor_text)

    def _schedule_current_poll(self, delay_ms: int = CURRENT_POLL_INTERVAL_MS) -> None:
        if self.closing or self.gripper is None or self._current_poll_job is not None:
            return
        try:
            self._current_poll_job = self.after(delay_ms, self._poll_current_once)
        except tk.TclError:
            self._current_poll_job = None

    def _start_current_polling(self) -> None:
        self._stop_current_polling()
        if self.gripper is None or self.closing:
            self._reset_current_feedback()
            return
        self._last_current_poll_error = None
        self.current_monitor_var.set(
            f"电流监视：实时更新（{CURRENT_POLL_INTERVAL_MS} ms）"
        )
        self._schedule_current_poll(delay_ms=0)

    def _stop_current_polling(self, monitor_text: str | None = None) -> None:
        self._current_poll_token += 1
        if self._current_poll_job is not None:
            try:
                self.after_cancel(self._current_poll_job)
            except tk.TclError:
                pass
            self._current_poll_job = None
        if monitor_text is not None:
            self.current_monitor_var.set(monitor_text)

    def _poll_current_once(self) -> None:
        self._current_poll_job = None
        gripper = self.gripper
        if self.closing or gripper is None:
            return
        if self._current_poll_inflight:
            self._schedule_current_poll()
            return

        self._current_poll_inflight = True
        poll_token = self._current_poll_token

        def worker() -> None:
            grip_current: float | None = None
            rotation_current: float | None = None
            error: Exception | None = None
            try:
                # 只读两个实际反馈寄存器，避免实时监视反复读取完整状态快照。
                grip_current = float(gripper.read_grip_current())
                rotation_current = float(gripper.read_rotation_current())
            except Exception as exc:
                error = exc
            try:
                self.after(
                    0,
                    lambda: self._finish_current_poll(
                        poll_token, gripper, grip_current, rotation_current, error
                    ),
                )
            except tk.TclError:
                pass

        threading.Thread(target=worker, name="gripper-current-poll", daemon=True).start()

    def _finish_current_poll(
        self,
        poll_token: int,
        gripper: Any,
        grip_current: float | None,
        rotation_current: float | None,
        error: Exception | None,
    ) -> None:
        self._current_poll_inflight = False
        if (
            self.closing
            or poll_token != self._current_poll_token
            or self.gripper is not gripper
        ):
            return

        if error is not None:
            self.grip_feedback_var.set("夹持实际电流：读取失败")
            self.rotation_feedback_var.set("旋转实际电流：读取失败")
            self.current_monitor_var.set("电流监视：读取失败，正在重试")
            error_text = f"{type(error).__name__}: {error}"
            if error_text != self._last_current_poll_error:
                self._append_log(f"[实时电流] 读取失败：{error_text}")
                self._last_current_poll_error = error_text
        else:
            self.grip_feedback_var.set(
                f"夹持实际电流：{self._format_current(grip_current)}"
            )
            self.rotation_feedback_var.set(
                f"旋转实际电流：{self._format_current(rotation_current)}"
            )
            self.current_monitor_var.set(
                f"电流监视：实时更新（{CURRENT_POLL_INTERVAL_MS} ms）"
            )
            self._last_current_poll_error = None
        self._schedule_current_poll()

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        for button in self.buttons:
            button.configure(state=tk.DISABLED if busy else tk.NORMAL)

    # ------------------------------------------------------------------
    # 后台任务
    # ------------------------------------------------------------------
    def _start_worker(
        self,
        label: str,
        work: Callable[[], Any],
        on_success: Callable[[Any], None] | None = None,
    ) -> None:
        if self.busy:
            return
        self._set_busy(True)
        self.status_var.set(f"{label}中...")

        def worker() -> None:
            try:
                result = work()
            except Exception as exc:
                self.after(0, lambda error=exc: self._finish_error(label, error))
            else:
                self.after(0, lambda value=result: self._finish_success(label, value, on_success))

        threading.Thread(target=worker, name=f"gripper-{label}", daemon=True).start()

    def _finish_success(
        self,
        label: str,
        result: Any,
        on_success: Callable[[Any], None] | None,
    ) -> None:
        self._set_busy(False)
        if on_success is not None:
            on_success(result)
        elif result is not None:
            self._show_state(result)
        self.status_var.set(f"{label}完成")
        self._append_log(f"[{label}] 完成")

    def _finish_error(self, label: str, error: Exception) -> None:
        self._set_busy(False)
        self.status_var.set(f"{label}失败")
        self._append_log(f"[{label}] 失败：{type(error).__name__}: {error}")
        if label == "断开夹爪" and self.gripper is not None and not self.closing:
            try:
                if self.gripper.is_connected():
                    self._start_current_polling()
            except Exception:
                pass
        messagebox.showerror("夹爪操作失败", f"{label}失败：\n{error}", parent=self.winfo_toplevel())

    def _require_gripper(self) -> Any:
        if self.gripper is None or not self.gripper.is_connected():
            raise RuntimeError("请先连接夹爪")
        return self.gripper

    def _confirm(self, title: str, message: str) -> bool:
        if self.busy:
            return False
        return messagebox.askyesno(title, message, parent=self.winfo_toplevel())

    # ------------------------------------------------------------------
    # 连接和状态
    # ------------------------------------------------------------------
    def connect_gripper(self) -> None:
        try:
            settings = self._connection_settings()
        except Exception as exc:
            messagebox.showerror("夹爪参数错误", str(exc), parent=self.winfo_toplevel())
            return

        def work() -> tuple[Any, dict[str, Any]]:
            driver = load_gripper_driver()
            gripper = driver.ZErg20C(**settings)
            try:
                gripper.connect()
                state = gripper.read_all_status()
            except Exception:
                try:
                    gripper.close()
                except Exception:
                    pass
                raise
            return gripper, state

        def success(result: tuple[Any, dict[str, Any]]) -> None:
            self.gripper, state = result
            self._append_log(f"驱动已连接：{DRIVER_PATH}")
            self._show_state(state)
            self._start_current_polling()

        self._start_worker("连接夹爪", work, success)

    def disconnect_gripper(self) -> None:
        if self.busy:
            return
        gripper = self.gripper
        if gripper is None:
            self.status_var.set("未连接")
            self._reset_current_feedback()
            return

        self._stop_current_polling("电流监视：停止中")

        def work() -> None:
            gripper.close()

        def success(_result: Any) -> None:
            self.gripper = None
            self._reset_current_feedback()

        self._start_worker("断开夹爪", work, success)

    def refresh_status(self) -> None:
        try:
            gripper = self._require_gripper()
        except Exception as exc:
            messagebox.showerror("夹爪未连接", str(exc), parent=self.winfo_toplevel())
            return
        self._start_worker("刷新状态", gripper.read_all_status)

    # ------------------------------------------------------------------
    # 动作
    # ------------------------------------------------------------------
    def initialize_gripper(self) -> None:
        if not self._confirm("确认初始化", "初始化会让夹指张开校准，请确认夹指周围无物体。"):
            return
        try:
            gripper = self._require_gripper()
        except Exception as exc:
            messagebox.showerror("夹爪未连接", str(exc), parent=self.winfo_toplevel())
            return

        def work() -> dict[str, Any]:
            gripper.initialize(wait=True)
            gripper.enable_motor(True)
            return gripper.read_all_status()

        self._start_worker("初始化校准", work)

    def enable_motor(self) -> None:
        self._run_simple_action("电机使能", lambda gripper: gripper.enable_motor(True))

    def disable_motor(self) -> None:
        if not self._confirm("确认关闭电机", "关闭电机输出可能使当前夹持物失去保持力，确认继续吗？"):
            return
        self._run_simple_action("关闭电机", lambda gripper: gripper.enable_motor(False))

    def open_gripper(self) -> None:
        if not self._confirm("确认张开", "将执行张开夹爪动作，确认周围安全？"):
            return
        try:
            gripper = self._require_gripper()
            speed = self._float(self.grip_speed_var, "夹持速度")
            current = self._float(self.grip_current_var, "夹持电流")
        except Exception as exc:
            messagebox.showerror("夹爪参数错误", str(exc), parent=self.winfo_toplevel())
            return

        def work() -> dict[str, Any]:
            gripper.set_grip_speed(speed)
            gripper.set_grip_current(current)
            gripper.open_gripper(wait=True)
            return gripper.read_all_status()

        self._start_worker("张开夹爪", work)

    def close_gripper(self) -> None:
        if not self._confirm("确认闭合", "将执行闭合夹爪动作，请确认夹持对象和安全范围？"):
            return
        try:
            gripper = self._require_gripper()
            speed = self._float(self.grip_speed_var, "夹持速度")
            current = self._float(self.grip_current_var, "夹持电流")
        except Exception as exc:
            messagebox.showerror("夹爪参数错误", str(exc), parent=self.winfo_toplevel())
            return

        def work() -> dict[str, Any]:
            gripper.set_grip_speed(speed)
            gripper.set_grip_current(current)
            gripper.close_gripper(wait=True)
            return gripper.read_all_status()

        self._start_worker("闭合夹爪", work)

    def move_gripper(self) -> None:
        try:
            gripper = self._require_gripper()
            speed = self._float(self.grip_speed_var, "夹持速度")
            current = self._float(self.grip_current_var, "夹持电流")
            position = self._float(self.grip_position_var, "夹持位置")
        except Exception as exc:
            messagebox.showerror("夹爪参数错误", str(exc), parent=self.winfo_toplevel())
            return

        def work() -> dict[str, Any]:
            gripper.set_grip_speed(speed)
            gripper.set_grip_current(current)
            gripper.grip_to(position, wait=True)
            return gripper.read_all_status()

        self._start_worker("移动夹持位置", work)

    def rotate_absolute(self) -> None:
        if not self._confirm("确认旋转", "将执行夹爪绝对旋转动作，确认旋转范围安全？"):
            return
        try:
            gripper = self._require_gripper()
            speed = self._float(self.rotation_speed_var, "旋转速度")
            current = self._float(self.rotation_current_var, "旋转电流")
            angle = self._float(self.rotation_angle_var, "绝对角度")
        except Exception as exc:
            messagebox.showerror("夹爪参数错误", str(exc), parent=self.winfo_toplevel())
            return

        def work() -> dict[str, Any]:
            gripper.set_rotation_speed(speed)
            gripper.set_rotation_current(current)
            gripper.rotate_to(angle, wait=True)
            return gripper.read_all_status()

        self._start_worker("绝对旋转", work)

    def rotate_relative(self) -> None:
        if not self._confirm("确认相对旋转", "将执行夹爪相对旋转动作，确认旋转范围安全？"):
            return
        try:
            gripper = self._require_gripper()
            speed = self._float(self.rotation_speed_var, "旋转速度")
            current = self._float(self.rotation_current_var, "旋转电流")
            angle = self._float(self.rotation_relative_var, "相对角度")
        except Exception as exc:
            messagebox.showerror("夹爪参数错误", str(exc), parent=self.winfo_toplevel())
            return

        def work() -> dict[str, Any]:
            gripper.set_rotation_speed(speed)
            gripper.set_rotation_current(current)
            gripper.rotate_relative(angle, wait=True)
            return gripper.read_all_status()

        self._start_worker("相对旋转", work)

    def _run_simple_action(self, label: str, action: Callable[[Any], None]) -> None:
        try:
            gripper = self._require_gripper()
        except Exception as exc:
            messagebox.showerror("夹爪未连接", str(exc), parent=self.winfo_toplevel())
            return

        def work() -> dict[str, Any]:
            action(gripper)
            return gripper.read_all_status()

        self._start_worker(label, work)

    def on_close(self) -> None:
        self.closing = True
        self._stop_current_polling("电流监视：已停止")
        gripper = self.gripper
        self.gripper = None
        if gripper is not None:
            try:
                gripper.close()
            except Exception:
                pass
