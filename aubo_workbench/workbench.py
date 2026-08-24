#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AUBO 工具工作台：切换机械臂信息、运动控制、TCP 示教、眼在手标定、孔位验证页面。

每个页面都是各自模块里的独立 ttk.Frame，workbench 只负责导航栏、
连接参数同步和统一关闭。夹爪控制单独使用一个非模态窗口，便于同时操作
其他工作台页面。
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import BOTH, LEFT, RIGHT, X, Y, filedialog, messagebox, ttk
from typing import Any

from .gui_common import GuiLogWriter
from .gui_handeye import HandEyeGuiPanel
from .gui_hole_localization import HoleLocalizationPanel
from .gui_tools import ResultsCenterPanel, VisualToolsPanel
from .gripper_control import GripperControlPanel
from .motion_control import AuboMotionPanel
from .paths import (
    DATA_DIR,
    DEFAULT_ROBOT_IP,
    DEFAULT_ROBOT_PASSWORD,
    DEFAULT_ROBOT_PORT,
    DEFAULT_ROBOT_TIMEOUT_MS,
    DEFAULT_ROBOT_USER,
)
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
            initialdir=str(DATA_DIR),
            initialfile="aubo_robot_info.json", defaultextension=".json",
            filetypes=(("JSON 文件", "*.json"), ("所有文件", "*.*")),
        )
        if not path:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.last_result, ensure_ascii=False, indent=2), encoding="utf-8")
        self.status_var.set(f"已保存：{path}")


class CalibrationHubPanel(ttk.Frame):
    """将 TCP 示教与手眼标定收敛到同一个“标定中心”。"""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        notebook = ttk.Notebook(self)
        notebook.pack(fill=BOTH, expand=True)
        self.tcp_panel = TcpTeachPanel(notebook)
        self.handeye_panel = HandEyeGuiPanel(notebook, autostart=False)
        notebook.add(self.tcp_panel, text="TCP 示教")
        notebook.add(self.handeye_panel, text="RGB 手眼标定")

    def sync_connection(self, cfg: dict[str, Any]) -> None:
        for panel in (self.tcp_panel, self.handeye_panel):
            for attr, value in [
                ("ip_var", cfg["ip"]), ("port_var", str(cfg["port"])),
                ("user_var", cfg["user"]), ("password_var", cfg["password"]),
            ]:
                variable = getattr(panel, attr, None)
                if isinstance(variable, tk.StringVar):
                    variable.set(value)

    def on_close(self) -> None:
        for panel in (self.tcp_panel, self.handeye_panel):
            close = getattr(panel, "on_close", None)
            if callable(close):
                close()


class RobotQuickActions(ttk.LabelFrame):
    """工作台顶部的高频机器人操作；每次动作使用独立短连接。"""

    def __init__(self, master: tk.Misc, get_connection) -> None:
        super().__init__(master, text="公共机器人操作", padding=8)
        self.get_connection = get_connection
        self.busy = False
        self.status_var = tk.StringVar(value="未读取机器人状态")
        self._buttons: list[ttk.Button] = []
        self._build()

    def _button(self, text: str, command) -> ttk.Button:
        button = ttk.Button(self, text=text, command=command)
        button.pack(side=LEFT, padx=(0, 6))
        self._buttons.append(button)
        return button

    def _build(self) -> None:
        self._button("刷新状态", self.refresh_status)
        self._button("上电并启动", self.power_on)
        self._button("停止运动", self.stop_motion)
        self._button("清空队列", self.clear_queue)
        self._button("回原点", self.move_home)
        ttk.Label(self, textvariable=self.status_var, foreground="#555555").pack(side=LEFT, padx=(10, 0))

    def _set_busy(self, busy: bool, text: str | None = None) -> None:
        self.busy = busy
        for button in self._buttons:
            button.configure(state=tk.DISABLED if busy else tk.NORMAL)
        if text is not None:
            self.status_var.set(text)

    def _with_session(self, label: str, action, *, confirm: str | None = None) -> None:
        if self.busy:
            return
        if confirm and not messagebox.askyesno(label, confirm, parent=self.winfo_toplevel()):
            return
        try:
            cfg = self.get_connection()
        except Exception as exc:
            messagebox.showerror("连接参数错误", str(exc), parent=self.winfo_toplevel())
            return
        self._set_busy(True, f"{label}：连接机械臂…")

        def work() -> None:
            session = None
            try:
                from .motion_control import AuboMotionSession

                session = AuboMotionSession()
                session.connect(cfg["ip"], cfg["port"], cfg["user"], cfg["password"], cfg["timeout_ms"])
                result = action(session)
                snapshot = session.snapshot()
                message = self._format_snapshot(label, snapshot, result)
            except Exception as exc:
                self.after(0, lambda: self._show_action_error(label, exc))
                return
            finally:
                if session is not None:
                    session.disconnect()
            self.after(0, lambda: self._finish_action(message))

        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _format_snapshot(label: str, snapshot: dict[str, Any], result: Any = None) -> str:
        tcp = snapshot.get("tcp_pose_m_rad", [])
        xyz = [round(float(value) * 1000.0, 2) for value in tcp[:3]] if len(tcp) >= 3 else ["-", "-", "-"]
        suffix = "" if result is None else f"，返回={result}"
        return (
            f"{label}完成：上电={snapshot.get('power_on')}，静止={snapshot.get('steady')}，"
            f"碰撞={snapshot.get('collision')}，TCP XYZ(mm)={xyz}{suffix}"
        )

    def _finish_action(self, message: str) -> None:
        self._set_busy(False, message)

    def _show_action_error(self, label: str, exc: Exception) -> None:
        self._set_busy(False, f"{label}失败")
        messagebox.showerror(f"{label}失败", str(exc), parent=self.winfo_toplevel())

    def refresh_status(self) -> None:
        self._with_session("刷新状态", lambda session: None)

    def power_on(self) -> None:
        self._with_session(
            "上电并启动", lambda session: session.power_on_startup(),
            confirm="将对机械臂上电并启动。确认继续吗？",
        )

    def stop_motion(self) -> None:
        self._with_session(
            "停止运动", lambda session: session.stop_motion(),
            confirm="将向控制器发送停止运动命令。确认继续吗？",
        )

    def clear_queue(self) -> None:
        self._with_session(
            "清空运动队列", lambda session: session.clear_path(),
            confirm="将清空控制器当前运动队列。确认继续吗？",
        )

    def move_home(self) -> None:
        from .motion_control import load_home_point

        home = load_home_point()
        if home is None:
            messagebox.showwarning("未设置原点", "请先在“机器人控制”页面设定原点。", parent=self.winfo_toplevel())
            return
        target = [round(float(value), 4) for value in home.tcp_pose_m_rad[:3]]

        def action(session) -> Any:
            response = session.move_joint(home.joints_rad, math.radians(20.0), math.radians(40.0))
            deadline = time.monotonic() + 45.0
            while time.monotonic() < deadline:
                snapshot = session.snapshot()
                if snapshot.get("collision"):
                    raise RuntimeError("回原点后检测到碰撞标志")
                joints = snapshot.get("joints_rad", [])
                at_home = (
                    len(joints) == len(home.joints_rad)
                    and max(abs(float(actual) - float(target)) for actual, target in zip(joints, home.joints_rad))
                    <= math.radians(0.5)
                )
                if snapshot.get("power_on") and snapshot.get("steady") and at_home:
                    return response
                time.sleep(0.25)
            raise RuntimeError("回原点超时：机械臂未稳定到达保存的关节原点")

        self._with_session(
            "回原点", action,
            confirm=f"将以关节运动回到原点“{home.name}”，目标 TCP XYZ(m)={target}。确认继续吗？",
        )


class AuboWorkbench(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AUBO 工具工作台")
        self.geometry("1440x880")
        self.minsize(1160, 720)

        self.ip_var = tk.StringVar(value=DEFAULT_ROBOT_IP)
        self.port_var = tk.StringVar(value=str(DEFAULT_ROBOT_PORT))
        self.user_var = tk.StringVar(value=DEFAULT_ROBOT_USER)
        self.password_var = tk.StringVar(value=DEFAULT_ROBOT_PASSWORD)
        self.timeout_var = tk.StringVar(value=str(DEFAULT_ROBOT_TIMEOUT_MS))

        self.pages: dict[str, ttk.Frame] = {}
        self.nav_buttons: dict[str, ttk.Button] = {}
        self.current_page = ""
        self.gripper_window: tk.Toplevel | None = None
        self.gripper_panel: GripperControlPanel | None = None
        self.old_stdout: Any | None = None
        self.old_stderr: Any | None = None

        self._build()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.show_page("hole_localization")

    def _build(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        top = ttk.LabelFrame(self, text="统一连接参数", padding=8)
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
        ttk.Button(top, text="同步当前页面", command=self.sync_current_page).pack(side=LEFT, padx=(0, 8))
        ttk.Label(top, text="连接信息只在此处维护；切换页面会自动同步。", foreground="#555555").pack(side=LEFT)

        self.quick_actions = RobotQuickActions(self, self.get_connection)
        self.quick_actions.pack(fill=X, padx=10, pady=(0, 6))

        body = ttk.Frame(self)
        body.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))

        nav = ttk.Frame(body, width=170)
        nav.pack(side=LEFT, fill=Y, padx=(0, 10))
        nav.pack_propagate(False)
        ttk.Label(nav, text="现场执行", foreground="#555555").pack(anchor="w", pady=(2, 5))
        for key, text in [("hole_localization", "孔洞定位")]:
            btn = ttk.Button(nav, text=text, command=lambda page=key: self.show_page(page))
            btn.pack(fill=X, pady=(0, 8))
            self.nav_buttons[key] = btn
        ttk.Label(nav, text="机器人与标定", foreground="#555555").pack(anchor="w", pady=(8, 5))
        for key, text in [
            ("motion_control", "机器人控制"),
            ("gripper", "夹爪控制"),
            ("calibration", "标定中心"),
        ]:
            command = self.open_gripper_window if key == "gripper" else lambda page=key: self.show_page(page)
            btn = ttk.Button(nav, text=text, command=command)
            btn.pack(fill=X, pady=(0, 8))
            self.nav_buttons[key] = btn
        ttk.Label(nav, text="系统", foreground="#555555").pack(anchor="w", pady=(8, 5))
        for key, text in [
            ("visual_tools", "视觉与实验"),
            ("results", "结果中心"),
            ("robot_info", "系统信息"),
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
        if key == "gripper":
            self.open_gripper_window()
            return
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
        if key == "calibration":
            panel = CalibrationHubPanel(self.content)
            if self.old_stdout is None:
                self.old_stdout = sys.stdout
                self.old_stderr = sys.stderr
                sys.stdout = GuiLogWriter(panel.handeye_panel.log_queue, self.old_stdout)  # type: ignore[assignment]
                sys.stderr = GuiLogWriter(panel.handeye_panel.log_queue, self.old_stderr)  # type: ignore[assignment]
            return panel
        if key == "hole_localization":
            return HoleLocalizationPanel(self.content, self.get_connection)
        if key == "visual_tools":
            return VisualToolsPanel(self.content, self.get_connection)
        if key == "results":
            return ResultsCenterPanel(self.content)
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
        sync = getattr(page, "sync_connection", None)
        if callable(sync):
            sync(cfg)
            return
        for attr, value in [
            ("ip_var", cfg["ip"]), ("port_var", str(cfg["port"])),
            ("user_var", cfg["user"]), ("password_var", cfg["password"]),
        ]:
            var = getattr(page, attr, None)
            if isinstance(var, tk.StringVar):
                var.set(value)

    def open_gripper_window(self) -> None:
        """打开独立的夹爪控制窗口，不切换或阻塞主工作台页面。"""

        if self.gripper_window is not None and self.gripper_window.winfo_exists():
            self.gripper_window.deiconify()
            self.gripper_window.lift()
            self.gripper_window.focus_force()
            return

        window = tk.Toplevel(self)
        window.title("夹爪控制 - Z-ERG-20C")
        window.geometry("1040x760")
        window.minsize(900, 650)
        window.protocol("WM_DELETE_WINDOW", self.close_gripper_window)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)

        panel = GripperControlPanel(window)
        panel.grid(row=0, column=0, sticky="nsew")
        self.gripper_window = window
        self.gripper_panel = panel

        # 不使用 transient/grab_set：主工作台和夹爪窗口可以同时操作。
        window.lift()
        window.focus_force()

    def close_gripper_window(self) -> None:
        """关闭夹爪窗口并释放其串口连接。"""

        window = self.gripper_window
        panel = self.gripper_panel
        if panel is not None and panel.busy:
            messagebox.showwarning(
                "夹爪操作进行中",
                "当前夹爪动作尚未完成，请等待动作结束后再关闭窗口。",
                parent=window if window is not None and window.winfo_exists() else self,
            )
            return
        if panel is not None:
            panel.on_close()
        self.gripper_panel = None
        self.gripper_window = None
        if window is not None and window.winfo_exists():
            window.destroy()

    def on_close(self) -> None:
        # 主工作台退出时，夹爪窗口一并关闭；此时不再拦截正在进行的窗口关闭。
        panel = self.gripper_panel
        window = self.gripper_window
        self.gripper_panel = None
        self.gripper_window = None
        if panel is not None:
            try:
                panel.on_close()
            except Exception:
                pass
        if window is not None and window.winfo_exists():
            window.destroy()
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
