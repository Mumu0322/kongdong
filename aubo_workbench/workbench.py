#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AUBO 工具工作台：一个窗口里切换机械臂信息、运动控制、TCP 示教、眼在手标定、伞架孔验证页面。

每个页面都是各自模块里的独立 ttk.Frame，workbench 只负责导航栏、
连接参数同步和统一关闭。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import BOTH, LEFT, RIGHT, X, Y, filedialog, messagebox, ttk
from typing import Any

from .gui_common import GuiLogWriter
from .gui_handeye import HandEyeGuiPanel
from .motion_control import AuboMotionPanel
from .robot_info import read_all
from .tcp_teach import TcpTeachPanel


def _network_precheck(ip: str, port: int, timeout_s: float = 1.0) -> tuple[bool, str]:
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        sock.connect((ip, port))
        local_ip, local_port = sock.getsockname()
        return True, f"端口可达，本机出口 {local_ip}:{local_port}"
    except OSError as exc:
        return False, f"无法连接 {ip}:{port}，系统错误：{exc}"
    finally:
        sock.close()


class RobotInfoPanel(ttk.Frame):
    def __init__(self, master: tk.Misc, get_connection) -> None:
        super().__init__(master, padding=10)
        self.get_connection = get_connection
        self.status_var = tk.StringVar(value="未读取")
        self.last_result: dict[str, Any] | None = None
        self.worker_running = False
        self._build()

    def _build(self) -> None:
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=X, pady=(0, 8))
        self.refresh_btn = ttk.Button(toolbar, text="读取机械臂信息", command=self.refresh_info)
        self.refresh_btn.pack(side=LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="网络诊断", command=self.network_diagnostic).pack(side=LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="保存JSON", command=self.save_json).pack(side=LEFT, padx=(0, 12))
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=LEFT)

        body = ttk.Frame(self)
        body.pack(fill=BOTH, expand=True)

        left = ttk.LabelFrame(body, text="摘要")
        left.pack(side=LEFT, fill=Y, padx=(0, 10))
        self.summary = tk.Text(left, width=46, height=28, wrap="word", font=("Microsoft YaHei UI", 10))
        self.summary.pack(fill=BOTH, expand=True, padx=6, pady=6)

        right = ttk.LabelFrame(body, text="完整信息 JSON")
        right.pack(side=RIGHT, fill=BOTH, expand=True)
        self.detail = tk.Text(right, wrap="none", font=("Consolas", 9))
        y_scroll = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self.detail.yview)
        x_scroll = ttk.Scrollbar(right, orient=tk.HORIZONTAL, command=self.detail.xview)
        self.detail.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.detail.pack(side=LEFT, fill=BOTH, expand=True, padx=(6, 0), pady=(6, 0))
        y_scroll.pack(side=RIGHT, fill=Y, padx=(0, 6), pady=(6, 0))
        x_scroll.pack(fill=X, padx=6, pady=(0, 6))

    def network_diagnostic(self) -> None:
        cfg = self.get_connection()
        ok, message = _network_precheck(cfg["ip"], cfg["port"])
        self.status_var.set("网络可达" if ok else "网络不可达")
        (messagebox.showinfo if ok else messagebox.showerror)("网络诊断", message)

    def refresh_info(self) -> None:
        if self.worker_running:
            return
        self.worker_running = True
        self.refresh_btn.configure(state=tk.DISABLED)
        self.status_var.set("读取中...")
        threading.Thread(target=self._refresh_worker, daemon=True).start()

    def _refresh_worker(self) -> None:
        try:
            cfg = self.get_connection()
            ok, message = _network_precheck(cfg["ip"], cfg["port"])
            if not ok:
                raise RuntimeError(message)
            result = read_all(cfg["ip"], cfg["port"], cfg["user"], cfg["password"], cfg["timeout_ms"])
            import time

            result["read_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as exc:
            self.after(0, lambda: self._show_error(exc))
            return
        self.after(0, lambda: self._show_result(result))

    def _show_error(self, exc: Exception) -> None:
        self.worker_running = False
        self.refresh_btn.configure(state=tk.NORMAL)
        self.status_var.set("读取失败")
        message = str(exc)
        self.summary.delete("1.0", tk.END)
        self.summary.insert(tk.END, message)
        messagebox.showerror("读取失败", message)

    def _show_result(self, result: dict[str, Any]) -> None:
        self.worker_running = False
        self.refresh_btn.configure(state=tk.NORMAL)
        self.last_result = result
        self.status_var.set("读取完成")

        self.summary.delete("1.0", tk.END)
        self.summary.insert(tk.END, self._build_summary(result))
        self.detail.delete("1.0", tk.END)
        self.detail.insert(tk.END, json.dumps(result, ensure_ascii=False, indent=2))

    def _build_summary(self, result: dict[str, Any]) -> str:
        lines = [
            f"读取时间：{result.get('read_at', '-')}",
            f"连接：{result.get('connection', {}).get('ip')}:{result.get('connection', {}).get('port')}",
            "",
        ]
        system_info = result.get("system_info", {})
        lines.extend([
            "控制器：",
            f"  软件版本：{system_info.get('control_software_version')}",
            f"  构建日期：{system_info.get('control_software_build_date')}",
            f"  系统时间：{system_info.get('control_system_time')}",
            "",
        ])
        for robot in result.get("robots", []):
            config = robot.get("config", {})
            state = robot.get("state", {})
            manage = robot.get("manage", {})
            lines.extend([
                f"机械臂：{robot.get('name')}",
                f"  类型：{config.get('robot_type')} / 子类型 {config.get('robot_sub_type')}",
                f"  控制柜：{config.get('control_box_type')}    DOF：{config.get('dof')}",
                f"  上电：{state.get('power_on')}    静止：{state.get('steady')}",
                f"  模式：{state.get('robot_mode')} / 安全：{state.get('safety_mode')}",
                f"  碰撞：{state.get('collision_occurred')}    仿真：{manage.get('simulation_enabled')}",
                f"  当前 TCP：{state.get('tcp_pose')}",
                f"  TCP 偏移：{config.get('tcp_offset')}",
                f"  关节角：{state.get('joint_positions_rad')}",
                "",
            ])
        return "\n".join(lines)

    def save_json(self) -> None:
        if self.last_result is None:
            messagebox.showwarning("没有数据", "请先读取机械臂信息。")
            return
        path = filedialog.asksaveasfilename(
            initialdir=str(Path(__file__).resolve().parent.parent / "data"),
            initialfile="aubo_robot_info.json", defaultextension=".json",
            filetypes=(("JSON 文件", "*.json"), ("所有文件", "*.*")),
        )
        if not path:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.last_result, ensure_ascii=False, indent=2), encoding="utf-8")
        self.status_var.set(f"已保存：{path}")


class AuboWorkbench(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AUBO 工具工作台")
        self.geometry("1440x880")
        self.minsize(1160, 720)

        self.ip_var = tk.StringVar(value=os.getenv("AUBO_ROBOT_IP", "192.168.1.100"))
        self.port_var = tk.StringVar(value="30004")
        self.user_var = tk.StringVar(value="AUBO")
        self.password_var = tk.StringVar(value=os.getenv("AUBO_ROBOT_PASSWORD", ""))
        self.timeout_var = tk.StringVar(value="3000")

        self.pages: dict[str, ttk.Frame] = {}
        self.nav_buttons: dict[str, ttk.Button] = {}
        self.current_page = ""
        self.old_stdout: Any | None = None
        self.old_stderr: Any | None = None

        self._build()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.show_page("robot_info")

    def _build(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        top = ttk.LabelFrame(self, text="AUBO 连接参数", padding=8)
        top.pack(fill=X, padx=10, pady=(10, 6))
        ttk.Label(top, text="IP").pack(side=LEFT)
        ttk.Entry(top, textvariable=self.ip_var, width=16).pack(side=LEFT, padx=(4, 10))
        ttk.Label(top, text="端口").pack(side=LEFT)
        ttk.Entry(top, textvariable=self.port_var, width=7).pack(side=LEFT, padx=(4, 10))
        ttk.Label(top, text="账号").pack(side=LEFT)
        ttk.Entry(top, textvariable=self.user_var, width=10).pack(side=LEFT, padx=(4, 10))
        ttk.Label(top, text="密码").pack(side=LEFT)
        ttk.Entry(top, textvariable=self.password_var, width=12, show="*").pack(side=LEFT, padx=(4, 10))
        ttk.Label(top, text="超时ms").pack(side=LEFT)
        ttk.Entry(top, textvariable=self.timeout_var, width=8).pack(side=LEFT, padx=(4, 12))
        ttk.Button(top, text="同步到当前功能", command=self.sync_current_page).pack(side=LEFT, padx=(0, 8))

        body = ttk.Frame(self)
        body.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))

        nav = ttk.Frame(body, width=170)
        nav.pack(side=LEFT, fill=Y, padx=(0, 10))
        nav.pack_propagate(False)
        for key, text in [
            ("robot_info", "机械臂信息"),
            ("motion_control", "机械臂运动"),
            ("tcp_teach", "TCP 示教"),
            ("handeye", "眼在手标定"),
        ]:
            btn = ttk.Button(nav, text=text, command=lambda page=key: self.show_page(page))
            btn.pack(fill=X, pady=(0, 8))
            self.nav_buttons[key] = btn

        self.content = ttk.Frame(body)
        self.content.pack(side=RIGHT, fill=BOTH, expand=True)

    def get_connection(self) -> dict[str, Any]:
        try:
            port = int(self.port_var.get().strip())
            timeout_ms = int(self.timeout_var.get().strip())
        except ValueError as exc:
            raise ValueError("端口和超时必须是数字") from exc
        return {
            "ip": self.ip_var.get().strip(), "port": port,
            "user": self.user_var.get().strip(), "password": self.password_var.get(), "timeout_ms": timeout_ms,
        }

    def show_page(self, key: str) -> None:
        if self.current_page == key:
            return
        page = self.pages.get(key)
        if page is None:
            try:
                page = self._create_page(key)
            except Exception as exc:
                messagebox.showerror("功能加载失败", str(exc))
                return
            self.pages[key] = page
        for existing in self.pages.values():
            existing.pack_forget()
        page.pack(fill=BOTH, expand=True)
        self.current_page = key
        self.sync_current_page(show_errors=False)

    def _create_page(self, key: str) -> ttk.Frame:
        if key == "robot_info":
            return RobotInfoPanel(self.content, self.get_connection)
        if key == "motion_control":
            return AuboMotionPanel(self.content)
        if key == "tcp_teach":
            return TcpTeachPanel(self.content)
        if key == "handeye":
            panel = HandEyeGuiPanel(self.content, autostart=False)
            if self.old_stdout is None:
                self.old_stdout = sys.stdout
                self.old_stderr = sys.stderr
                sys.stdout = GuiLogWriter(panel.log_queue, self.old_stdout)  # type: ignore[assignment]
                sys.stderr = GuiLogWriter(panel.log_queue, self.old_stderr)  # type: ignore[assignment]
            return panel
        raise KeyError(key)

    def sync_current_page(self, show_errors: bool = True) -> None:
        try:
            cfg = self.get_connection()
        except Exception as exc:
            if show_errors:
                messagebox.showerror("连接参数错误", str(exc))
            return
        page = self.pages.get(self.current_page)
        if page is None:
            return
        for attr, value in [
            ("ip_var", cfg["ip"]), ("port_var", str(cfg["port"])),
            ("user_var", cfg["user"]), ("password_var", cfg["password"]),
        ]:
            var = getattr(page, attr, None)
            if isinstance(var, tk.StringVar):
                var.set(value)

    def on_close(self) -> None:
        for page in self.pages.values():
            close = getattr(page, "on_close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        if self.old_stdout is not None:
            sys.stdout = self.old_stdout
        if self.old_stderr is not None:
            sys.stderr = self.old_stderr
        self.destroy()


def main() -> None:
    AuboWorkbench().mainloop()


if __name__ == "__main__":
    main()
