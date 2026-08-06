#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""独立 AUBO 机械臂上下电与运动控制 GUI。

运行建议：
    C:\Users\j1005\.conda\envs\lip_env310\python.exe C:\MM\aubo_tools\aubo_workbench_project\run_workbench.py
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, X, Y, messagebox, simpledialog, ttk
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Callable


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
AUBO_TOOLS_DIR = PROJECT_DIR.parent
DATA_DIR = AUBO_TOOLS_DIR / "data"
POINTS_FILE = DATA_DIR / "aubo_motion_points.json"
HOME_POINT_FILE = DATA_DIR / "aubo_home_point.json"
DEFAULT_SDK_DIR = Path(r"C:\MM\third_party\aubo_sdk")

DEFAULT_IP = "192.168.50.200"
DEFAULT_PORT = 30004
DEFAULT_USER = "AUBO"
DEFAULT_PASSWORD = "123456"
DEFAULT_TIMEOUT_MS = 3000
MOTION_FRAME_CHOICES = ("基坐标系", "工具/TCP坐标系")


def add_local_sdk_path() -> Path | None:
    """Add the local AUBO SDK directory for Python 3.10 environments."""
    candidates = [
        PROJECT_DIR / "third_party" / "aubo_sdk",
        AUBO_TOOLS_DIR / "third_party" / "aubo_sdk",
        AUBO_TOOLS_DIR.parent / "third_party" / "aubo_sdk",
        DEFAULT_SDK_DIR,
    ]
    for candidate in candidates:
        if candidate.exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    return None


SDK_DIR = add_local_sdk_path()
SDK_IMPORT_ERROR: Exception | None = None
try:
    import pyaubo_sdk as aubo  # type: ignore
except Exception as exc:  # pragma: no cover - depends on local SDK install
    aubo = None  # type: ignore
    SDK_IMPORT_ERROR = exc


def now_text() -> str:
    return time.strftime("%H:%M:%S")


def rad_to_deg(value: float) -> float:
    return value * 180.0 / math.pi


def deg_to_rad(value: float) -> float:
    return value * math.pi / 180.0


def m_to_mm(value: float) -> float:
    return value * 1000.0


def mm_to_m(value: float) -> float:
    return value / 1000.0


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[row][k] * b[k][col] for k in range(3)) for col in range(3)]
        for row in range(3)
    ]


def _mat_vec_mul(a: list[list[float]], v: list[float]) -> list[float]:
    return [sum(a[row][k] * v[k] for k in range(3)) for row in range(3)]


def _rotx(rad: float) -> list[list[float]]:
    c, s = math.cos(rad), math.sin(rad)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def _roty(rad: float) -> list[list[float]]:
    c, s = math.cos(rad), math.sin(rad)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rotz(rad: float) -> list[list[float]]:
    c, s = math.cos(rad), math.sin(rad)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _axis_rot(axis: int, rad: float) -> list[list[float]]:
    if axis == 0:
        return _rotx(rad)
    if axis == 1:
        return _roty(rad)
    return _rotz(rad)


def rpy_to_matrix(rx: float, ry: float, rz: float) -> list[list[float]]:
    return _mat_mul(_mat_mul(_rotz(rz), _roty(ry)), _rotx(rx))


def matrix_to_rpy(matrix: list[list[float]]) -> list[float]:
    # Inverse of R = Rz(rz) * Ry(ry) * Rx(rx), matching the AUBO RPY usage in the hand-eye code.
    r20 = max(-1.0, min(1.0, -matrix[2][0]))
    ry = math.asin(r20)
    cy = math.cos(ry)
    if abs(cy) > 1e-8:
        rz = math.atan2(matrix[1][0], matrix[0][0])
        rx = math.atan2(matrix[2][1], matrix[2][2])
    else:
        rz = math.atan2(-matrix[0][1], matrix[1][1])
        rx = 0.0
    return [rx, ry, rz]


def sdk_ok(ret: Any) -> bool:
    if aubo is not None:
        try:
            if ret == aubo.AUBO_OK:
                return True
        except Exception:
            pass
    try:
        return int(ret) == 0
    except Exception:
        text = str(ret)
        return text == "0" or text.endswith(".AUBO_OK") or text == "AUBO_OK"


def ret_text(ret: Any) -> str:
    return str(ret)


def read_float(var: tk.StringVar, name: str, min_value: float | None = None) -> float:
    text = var.get().strip()
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数字：{text}") from exc
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} 不能小于 {min_value}")
    return value


@dataclass
class SavedPoint:
    name: str
    created_at: str
    joints_rad: list[float]
    tcp_pose_m_rad: list[float]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SavedPoint":
        return cls(
            name=str(data.get("name", "未命名点")),
            created_at=str(data.get("created_at", "")),
            joints_rad=[float(v) for v in data.get("joints_rad", data.get("joints", [0.0] * 6))],
            tcp_pose_m_rad=[float(v) for v in data.get("tcp_pose_m_rad", data.get("tcp", [0.0] * 6))],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "joints_rad": self.joints_rad,
            "tcp_pose_m_rad": self.tcp_pose_m_rad,
        }


def load_points() -> list[SavedPoint]:
    if not POINTS_FILE.exists():
        return []
    try:
        payload = json.loads(POINTS_FILE.read_text(encoding="utf-8"))
        items = payload.get("points", payload if isinstance(payload, list) else [])
        return [SavedPoint.from_dict(item) for item in items]
    except Exception:
        return []


def save_points(points: list[SavedPoint]) -> None:
    POINTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    POINTS_FILE.write_text(
        json.dumps({"points": [p.to_dict() for p in points]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_home_point() -> SavedPoint | None:
    if not HOME_POINT_FILE.exists():
        return None
    try:
        payload = json.loads(HOME_POINT_FILE.read_text(encoding="utf-8"))
        data = payload.get("home_point", payload)
        return SavedPoint.from_dict(data)
    except Exception:
        return None


def save_home_point(point: SavedPoint) -> None:
    HOME_POINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    HOME_POINT_FILE.write_text(
        json.dumps({"home_point": point.to_dict()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class AuboMotionSession:
    """Thin wrapper around pyaubo_sdk. GUI calls are serialized by one lock."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.client: Any | None = None
        self.robot_if: Any | None = None
        self.state: Any | None = None
        self.motion: Any | None = None
        self.manage: Any | None = None
        self.config: Any | None = None
        self.robot_name = ""

    @property
    def connected(self) -> bool:
        return self.client is not None and self.robot_if is not None

    def connect(self, ip: str, port: int, user: str, password: str, timeout_ms: int) -> str:
        if aubo is None:
            raise RuntimeError(
                "未能导入 pyaubo_sdk。请使用 Python 3.10 运行，或确认 "
                f"AUBO SDK 位于 {SDK_DIR or DEFAULT_SDK_DIR}。"
                f"\n原始错误：{SDK_IMPORT_ERROR}"
            )
        with self.lock:
            self.disconnect()
            client = aubo.RpcClient()
            client.setRequestTimeout(int(timeout_ms))
            ret = client.connect(ip, int(port))
            if not sdk_ok(ret):
                raise RuntimeError(f"连接失败：{ret_text(ret)}")
            ret = client.login(user, password)
            if not sdk_ok(ret):
                try:
                    client.disconnect()
                except Exception:
                    pass
                raise RuntimeError(f"登录失败：{ret_text(ret)}")
            names = list(client.getRobotNames())
            if not names:
                try:
                    client.disconnect()
                except Exception:
                    pass
                raise RuntimeError("已连接控制器，但没有找到机器人名称。")

            self.client = client
            self.robot_name = str(names[0])
            self.robot_if = client.getRobotInterface(self.robot_name)
            self.state = self.robot_if.getRobotState()
            self.motion = self.robot_if.getMotionControl()
            self.manage = self.robot_if.getRobotManage()
            self.config = self.robot_if.getRobotConfig()
            return self.robot_name

    def disconnect(self) -> None:
        with self.lock:
            if self.client is not None:
                try:
                    self.client.disconnect()
                except Exception:
                    pass
            self.client = None
            self.robot_if = None
            self.state = None
            self.motion = None
            self.manage = None
            self.config = None
            self.robot_name = ""

    def require_connected(self) -> None:
        if not self.connected or self.state is None or self.motion is None or self.manage is None:
            raise RuntimeError("尚未连接机械臂。")

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            self.require_connected()
            assert self.state is not None
            snap: dict[str, Any] = {}
            snap["robot_name"] = self.robot_name
            snap["power_on"] = bool(self.state.isPowerOn())
            snap["steady"] = bool(self.state.isSteady())
            snap["collision"] = bool(self.state.isCollisionOccurred())
            snap["robot_mode"] = str(self.state.getRobotModeType())
            snap["safety_mode"] = str(self.state.getSafetyModeType())
            snap["joints_rad"] = [float(v) for v in list(self.state.getJointPositions())]
            snap["tcp_pose_m_rad"] = [float(v) for v in list(self.state.getTcpPose())]
            snap["tcp_speed"] = list(self.state.getTcpSpeed())
            if self.motion is not None:
                try:
                    snap["queue_size"] = int(self.motion.getQueueSize())
                except Exception:
                    snap["queue_size"] = None
            return snap

    def power_on_startup(self) -> list[Any]:
        with self.lock:
            self.require_connected()
            assert self.manage is not None
            ret1 = self.manage.poweron()
            time.sleep(0.2)
            ret2 = self.manage.startup()
            return [ret1, ret2]

    def power_off(self) -> Any:
        with self.lock:
            self.require_connected()
            assert self.manage is not None
            return self.manage.poweroff()

    def clear_path(self) -> Any:
        with self.lock:
            self.require_connected()
            assert self.motion is not None
            return self.motion.clearPath()

    def stop_motion(self) -> Any:
        with self.lock:
            self.require_connected()
            assert self.motion is not None
            return self.motion.stopMove(False, True)

    def move_joint(self, joints_rad: list[float], speed_rad_s: float, acc_rad_s2: float) -> list[Any]:
        with self.lock:
            self.require_connected()
            assert self.motion is not None
            rets: list[Any] = []
            try:
                rets.append(self.motion.clearPath())
            except Exception as exc:
                rets.append(f"clearPath 异常：{exc}")
            rets.append(self.motion.moveJoint([float(v) for v in joints_rad], float(speed_rad_s), float(acc_rad_s2), 0.0, 0.0))
            return rets

    def move_line(self, pose_m_rad: list[float], speed_m_s: float, acc_m_s2: float) -> list[Any]:
        with self.lock:
            self.require_connected()
            assert self.motion is not None
            rets: list[Any] = []
            try:
                rets.append(self.motion.clearPath())
            except Exception as exc:
                rets.append(f"clearPath 异常：{exc}")
            rets.append(self.motion.moveLine([float(v) for v in pose_m_rad], float(speed_m_s), float(acc_m_s2), 0.0, 0.0))
            return rets

    def speed_joint(self, speeds_rad_s: list[float], acc_rad_s2: float, duration_s: float) -> Any:
        with self.lock:
            self.require_connected()
            assert self.motion is not None
            return self.motion.speedJoint([float(v) for v in speeds_rad_s], float(acc_rad_s2), float(duration_s))

    def speed_line(self, speed_m_rad_s: list[float], acc_m_s2: float, duration_s: float) -> Any:
        with self.lock:
            self.require_connected()
            assert self.motion is not None
            return self.motion.speedLine([float(v) for v in speed_m_rad_s], float(acc_m_s2), float(duration_s))

    def freedrive(self, enable: bool) -> Any:
        with self.lock:
            self.require_connected()
            assert self.manage is not None
            return self.manage.freedrive(bool(enable))

    def backdrive(self, enable: bool) -> Any:
        with self.lock:
            self.require_connected()
            assert self.manage is not None
            return self.manage.backdrive(bool(enable))

    def exit_handguide(self) -> Any:
        with self.lock:
            self.require_connected()
            assert self.manage is not None
            return self.manage.exitHandguideMode()


class AuboMotionPanel(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.session = AuboMotionSession()
        self.points = load_points()
        self.home_point = load_home_point()
        self.last_snapshot: dict[str, Any] | None = None

        self.status_token = 0
        self.status_running = False
        self.jog_event: threading.Event | None = None
        self.jog_lock = threading.Lock()

        self.ip_var = tk.StringVar(value=DEFAULT_IP)
        self.port_var = tk.StringVar(value=str(DEFAULT_PORT))
        self.user_var = tk.StringVar(value=DEFAULT_USER)
        self.password_var = tk.StringVar(value=DEFAULT_PASSWORD)
        self.timeout_var = tk.StringVar(value=str(DEFAULT_TIMEOUT_MS))
        self.connection_var = tk.StringVar(value="未连接")

        self.power_var = tk.StringVar(value="-")
        self.steady_var = tk.StringVar(value="-")
        self.mode_var = tk.StringVar(value="-")
        self.safety_var = tk.StringVar(value="-")
        self.collision_var = tk.StringVar(value="-")
        self.queue_var = tk.StringVar(value="-")
        self.home_var = tk.StringVar(value=self.home_label())

        self.joint_speed_var = tk.StringVar(value="10")
        self.joint_acc_var = tk.StringVar(value="30")
        self.joint_step_var = tk.StringVar(value="1")
        self.tcp_speed_var = tk.StringVar(value="20")
        self.tcp_acc_var = tk.StringVar(value="80")
        self.rot_speed_var = tk.StringVar(value="5")
        self.rot_acc_var = tk.StringVar(value="30")
        self.ptp_speed_var = tk.StringVar(value="30")
        self.ptp_acc_var = tk.StringVar(value="60")
        self.line_speed_var = tk.StringVar(value="20")
        self.line_acc_var = tk.StringVar(value="80")
        self.line_step_var = tk.StringVar(value="5")
        self.rot_step_var = tk.StringVar(value="2")
        self.motion_frame_var = tk.StringVar(value="基坐标系")

        self.joint_value_vars = [tk.StringVar(value="-") for _ in range(6)]
        self.tcp_value_vars = [tk.StringVar(value="-") for _ in range(6)]

        self._build()
        self._refresh_points()
        if aubo is None:
            self.log(f"SDK 导入失败：{SDK_IMPORT_ERROR}")
            self.log("请用 Python 3.10 运行，或确认 C:\\MM\\third_party\\aubo_sdk 存在。")
        else:
            self.log(f"SDK 已加载：{getattr(aubo, '__file__', '-')}")

    def _build(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        viewport = ttk.Frame(self)
        viewport.pack(fill=BOTH, expand=True)
        self.main_canvas = tk.Canvas(viewport, highlightthickness=0)
        v_scroll = ttk.Scrollbar(viewport, orient=tk.VERTICAL, command=self.main_canvas.yview)
        h_scroll = ttk.Scrollbar(viewport, orient=tk.HORIZONTAL, command=self.main_canvas.xview)
        self.main_canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        v_scroll.pack(side=RIGHT, fill=Y)
        h_scroll.pack(side=tk.BOTTOM, fill=X)
        self.main_canvas.pack(side=LEFT, fill=BOTH, expand=True)

        self.main_content = ttk.Frame(self.main_canvas)
        self.main_window_id = self.main_canvas.create_window((0, 0), window=self.main_content, anchor="nw")
        self.main_content.bind("<Configure>", self._sync_scroll_region)
        self.main_canvas.bind("<Configure>", self._on_canvas_configure)
        self.main_canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.main_content.bind("<Enter>", lambda _event: self.bind_all("<MouseWheel>", self._on_mouse_wheel))
        self.main_content.bind("<Leave>", lambda _event: self.unbind_all("<MouseWheel>"))

        top = ttk.LabelFrame(self.main_content, text="机器人会话", padding=8)
        top.pack(fill=X, padx=10, pady=(10, 6))
        ttk.Label(top, text="连接参数来自工作台顶部；如有修改，请点击顶部“同步当前页面”。", foreground="#555555").grid(
            row=0, column=0, columnspan=4, sticky="w", padx=(0, 12)
        )
        self.connect_btn = ttk.Button(top, text="连接", command=self.connect_async)
        self.connect_btn.grid(row=0, column=4, sticky="ew", padx=(0, 6))
        ttk.Button(top, text="断开", command=self.disconnect_async).grid(row=0, column=5, sticky="ew", padx=(0, 12))
        ttk.Label(top, textvariable=self.connection_var).grid(row=0, column=6, sticky="w")
        top.columnconfigure(0, weight=1)

        body = ttk.Frame(self.main_content)
        body.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))

        paned = ttk.PanedWindow(body, orient=tk.HORIZONTAL)
        paned.pack(fill=BOTH, expand=True)
        left = ttk.Frame(paned, width=330)
        right = ttk.Frame(paned, width=900)
        paned.add(left, weight=1)
        paned.add(right, weight=3)

        self._build_status(left)
        self._build_power(left)
        self._build_log(left)

        notebook = ttk.Notebook(right)
        notebook.pack(fill=BOTH, expand=True)
        jog_page = ttk.Frame(notebook, padding=10)
        point_page = ttk.Frame(notebook, padding=10)
        line_page = ttk.Frame(notebook, padding=10)
        notebook.add(jog_page, text="按住连续运动")
        notebook.add(point_page, text="采点与点到点")
        notebook.add(line_page, text="相对直线运动")

        self._build_jog_page(jog_page)
        self._build_points_page(point_page)
        self._build_line_page(line_page)

    def _sync_scroll_region(self, _event: tk.Event | None = None) -> None:
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        target_width = max(int(event.width), 1050)
        self.main_canvas.itemconfigure(self.main_window_id, width=target_width)
        self._sync_scroll_region()

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        if event.state & 0x0001:
            self.main_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        else:
            self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _build_status(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="当前状态", padding=8)
        box.pack(fill=X, pady=(0, 8))
        for row, (name, var) in enumerate([
            ("上电", self.power_var),
            ("静止", self.steady_var),
            ("模式", self.mode_var),
            ("安全", self.safety_var),
            ("碰撞", self.collision_var),
            ("队列", self.queue_var),
            ("坐标", self.motion_frame_var),
            ("原始点", self.home_var),
        ]):
            ttk.Label(box, text=name, width=8).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Label(box, textvariable=var, width=28).grid(row=row, column=1, sticky="w", pady=2)

        joints = ttk.LabelFrame(parent, text="关节角 deg", padding=8)
        joints.pack(fill=X, pady=(0, 8))
        for i, var in enumerate(self.joint_value_vars):
            ttk.Label(joints, text=f"J{i + 1}", width=4).grid(row=i, column=0, sticky="w")
            ttk.Label(joints, textvariable=var, width=18).grid(row=i, column=1, sticky="w")

        tcp = ttk.LabelFrame(parent, text="TCP 位姿 mm / deg", padding=8)
        tcp.pack(fill=X, pady=(0, 8))
        for i, name in enumerate(["X", "Y", "Z", "Rx", "Ry", "Rz"]):
            ttk.Label(tcp, text=name, width=4).grid(row=i, column=0, sticky="w")
            ttk.Label(tcp, textvariable=self.tcp_value_vars[i], width=18).grid(row=i, column=1, sticky="w")

    def _build_power(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="电源与安全", padding=8)
        box.pack(fill=X, pady=(0, 8))
        ttk.Button(box, text="上电并启动", command=lambda: self.run_command("上电并启动", self.session.power_on_startup)).pack(fill=X, pady=(0, 6))
        ttk.Button(box, text="断电", command=lambda: self.run_command("断电", self.session.power_off)).pack(fill=X, pady=(0, 6))
        ttk.Button(box, text="停止运动", command=lambda: self.run_command("停止运动", self.session.stop_motion)).pack(fill=X, pady=(0, 6))
        ttk.Button(box, text="设当前为原始点", command=self.set_current_as_home_point).pack(fill=X, pady=(0, 6))
        ttk.Button(box, text="复位到原始点", command=self.reset_to_home_point).pack(fill=X, pady=(0, 6))
        ttk.Button(box, text="清空运动队列", command=lambda: self.run_command("清空运动队列", self.session.clear_path)).pack(fill=X, pady=(0, 6))
        ttk.Button(box, text="进入拖拽模式", command=lambda: self.run_command("进入拖拽模式", lambda: self.session.freedrive(True))).pack(fill=X, pady=(0, 6))
        ttk.Button(box, text="退出拖拽模式", command=lambda: self.run_command("退出拖拽模式", self.session.exit_handguide)).pack(fill=X)

    def _build_log(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="日志", padding=6)
        box.pack(fill=BOTH, expand=True)
        self.log_text = ScrolledText(box, width=44, height=14, font=("Consolas", 9), wrap="word")
        self.log_text.pack(fill=BOTH, expand=True)

    def _build_jog_page(self, parent: ttk.Frame) -> None:
        params = ttk.LabelFrame(parent, text="连续运动参数", padding=8)
        params.pack(fill=X, pady=(0, 10))
        self._grid_entry(params, 0, 0, "关节速度 deg/s", self.joint_speed_var, 10)
        self._grid_entry(params, 0, 2, "关节加速度 deg/s²", self.joint_acc_var, 10)
        self._grid_entry(params, 1, 0, "关节点动步长 deg", self.joint_step_var, 10)
        self._grid_entry(params, 1, 2, "TCP速度 mm/s", self.tcp_speed_var, 10)
        self._grid_entry(params, 2, 0, "TCP加速度 mm/s²", self.tcp_acc_var, 10)
        self._grid_entry(params, 2, 2, "XYZ点动步长 mm", self.line_step_var, 10)
        self._grid_entry(params, 3, 0, "姿态点动步长 deg", self.rot_step_var, 10)
        self._grid_combo(params, 3, 2, "运动坐标系", self.motion_frame_var, MOTION_FRAME_CHOICES, 16)

        joints = ttk.LabelFrame(parent, text="关节连续运动：按住按钮重复发送小步 moveJoint，松开停止", padding=8)
        joints.pack(fill=X, pady=(0, 10))
        for i in range(6):
            ttk.Label(joints, text=f"J{i + 1}", width=5).grid(row=i, column=0, padx=(0, 8), pady=3)
            self._hold_button(joints, "负向", lambda idx=i: self.start_joint_jog(idx, -1), 1, i)
            self._hold_button(joints, "正向", lambda idx=i: self.start_joint_jog(idx, 1), 2, i)

        tcp = ttk.LabelFrame(parent, text="TCP 连续运动：按住按钮重复发送小步 moveLine，松开停止", padding=8)
        tcp.pack(fill=X)
        for i, name in enumerate(["X", "Y", "Z", "Rx", "Ry", "Rz"]):
            ttk.Label(tcp, text=name, width=5).grid(row=i, column=0, padx=(0, 8), pady=3)
            self._hold_button(tcp, "负向", lambda idx=i: self.start_tcp_jog(idx, -1), 1, i)
            self._hold_button(tcp, "正向", lambda idx=i: self.start_tcp_jog(idx, 1), 2, i)

    def _build_points_page(self, parent: ttk.Frame) -> None:
        params = ttk.LabelFrame(parent, text="点到点参数", padding=8)
        params.pack(fill=X, pady=(0, 10))
        self._grid_entry(params, 0, 0, "速度 deg/s", self.ptp_speed_var, 10)
        self._grid_entry(params, 0, 2, "加速度 deg/s²", self.ptp_acc_var, 10)

        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=X, pady=(0, 8))
        ttk.Button(toolbar, text="采集当前点", command=self.capture_point).pack(side=LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="设当前为原始点", command=self.set_current_as_home_point).pack(side=LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="复位到原始点", command=self.reset_to_home_point).pack(side=LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="移动到选中点", command=self.move_to_selected_point).pack(side=LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="删除选中点", command=self.delete_selected_point).pack(side=LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="清空点位", command=self.clear_points).pack(side=LEFT, padx=(0, 6))

        split = ttk.Frame(parent)
        split.pack(fill=BOTH, expand=True)
        list_frame = ttk.Frame(split)
        list_frame.pack(side=LEFT, fill=Y, padx=(0, 10))
        self.point_list = tk.Listbox(list_frame, width=32, height=22)
        self.point_list.pack(side=LEFT, fill=Y)
        scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.point_list.yview)
        scroll.pack(side=RIGHT, fill=Y)
        self.point_list.configure(yscrollcommand=scroll.set)
        self.point_list.bind("<<ListboxSelect>>", lambda _event: self.show_selected_point())

        detail_frame = ttk.LabelFrame(split, text="点位详情", padding=6)
        detail_frame.pack(side=RIGHT, fill=BOTH, expand=True)
        self.point_detail = ScrolledText(detail_frame, font=("Consolas", 10), wrap="word")
        self.point_detail.pack(fill=BOTH, expand=True)

    def _build_line_page(self, parent: ttk.Frame) -> None:
        params = ttk.LabelFrame(parent, text="相对运动参数", padding=8)
        params.pack(fill=X, pady=(0, 10))
        self._grid_entry(params, 0, 0, "直线速度 mm/s", self.line_speed_var, 10)
        self._grid_entry(params, 0, 2, "直线加速度 mm/s²", self.line_acc_var, 10)
        self._grid_entry(params, 1, 0, "XYZ步长 mm", self.line_step_var, 10)
        self._grid_entry(params, 1, 2, "Rx/Ry/Rz步长 deg", self.rot_step_var, 10)
        self._grid_combo(params, 2, 0, "运动坐标系", self.motion_frame_var, MOTION_FRAME_CHOICES, 16)

        moves = ttk.LabelFrame(parent, text="基于当前 TCP 的相对 moveLine", padding=8)
        moves.pack(fill=X)
        for i, name in enumerate(["X", "Y", "Z", "Rx", "Ry", "Rz"]):
            ttk.Label(moves, text=name, width=5).grid(row=i, column=0, padx=(0, 8), pady=3)
            ttk.Button(moves, text="负向一步", command=lambda idx=i: self.relative_line_move(idx, -1)).grid(row=i, column=1, sticky="ew", padx=(0, 6), pady=3)
            ttk.Button(moves, text="正向一步", command=lambda idx=i: self.relative_line_move(idx, 1)).grid(row=i, column=2, sticky="ew", padx=(0, 6), pady=3)
        moves.columnconfigure(1, weight=1)
        moves.columnconfigure(2, weight=1)

    def _grid_entry(self, parent: ttk.Frame, row: int, col: int, label: str, var: tk.StringVar, width: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 4), pady=3)
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=col + 1, sticky="w", padx=(0, 16), pady=3)

    def _grid_combo(
        self,
        parent: ttk.Frame,
        row: int,
        col: int,
        label: str,
        var: tk.StringVar,
        values: tuple[str, ...],
        width: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 4), pady=3)
        ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=width).grid(
            row=row, column=col + 1, sticky="w", padx=(0, 16), pady=3
        )

    def _hold_button(self, parent: ttk.Frame, text: str, on_press: Callable[[], None], col: int, row: int) -> None:
        btn = ttk.Button(parent, text=text)
        btn.grid(row=row, column=col, sticky="ew", padx=(0, 6), pady=3)
        btn.bind("<ButtonPress-1>", lambda _event: on_press())
        btn.bind("<ButtonRelease-1>", lambda _event: self.stop_jog())
        btn.bind("<Leave>", lambda _event: self.stop_jog())
        parent.columnconfigure(col, weight=1)

    def log(self, message: str) -> None:
        self.log_text.insert(END, f"[{now_text()}] {message}\n")
        self.log_text.see(END)

    def home_label(self) -> str:
        if self.home_point is None:
            return "未设置"
        return f"{self.home_point.name} {self.home_point.created_at}".strip()

    def read_connection_inputs(self) -> tuple[str, int, str, str, int]:
        ip = self.ip_var.get().strip()
        if not ip:
            raise ValueError("IP 不能为空")
        try:
            port = int(self.port_var.get().strip())
            timeout = int(self.timeout_var.get().strip())
        except ValueError as exc:
            raise ValueError("端口和超时必须是整数") from exc
        return ip, port, self.user_var.get().strip(), self.password_var.get(), timeout

    def connect_async(self) -> None:
        try:
            ip, port, user, password, timeout = self.read_connection_inputs()
        except Exception as exc:
            messagebox.showerror("连接参数错误", str(exc))
            return
        self.connect_btn.configure(state=tk.DISABLED)
        self.connection_var.set("连接中...")
        self.log(f"连接 {ip}:{port} ...")

        def work() -> None:
            try:
                robot_name = self.session.connect(ip, port, user, password, timeout)
            except Exception as exc:
                self.after(0, lambda: self._connect_failed(exc))
                return
            self.after(0, lambda: self._connect_success(robot_name))

        threading.Thread(target=work, daemon=True).start()

    def _connect_success(self, robot_name: str) -> None:
        self.connect_btn.configure(state=tk.NORMAL)
        self.connection_var.set(f"已连接：{robot_name}")
        self.log(f"连接成功：{robot_name}")
        self.start_status_loop()

    def _connect_failed(self, exc: Exception) -> None:
        self.connect_btn.configure(state=tk.NORMAL)
        self.connection_var.set("连接失败")
        self.log(f"连接失败：{exc}")
        messagebox.showerror("连接失败", str(exc))

    def disconnect_async(self) -> None:
        self.stop_jog()
        self.status_token += 1

        def work() -> None:
            try:
                if self.session.connected:
                    try:
                        self.session.stop_motion()
                    except Exception:
                        pass
                self.session.disconnect()
            finally:
                self.after(0, self._disconnect_done)

        threading.Thread(target=work, daemon=True).start()

    def _disconnect_done(self) -> None:
        self.connection_var.set("未连接")
        self.power_var.set("-")
        self.steady_var.set("-")
        self.mode_var.set("-")
        self.safety_var.set("-")
        self.collision_var.set("-")
        self.queue_var.set("-")
        for var in self.joint_value_vars + self.tcp_value_vars:
            var.set("-")
        self.log("已断开")

    def run_command(self, name: str, func: Callable[[], Any]) -> None:
        def work() -> None:
            self.after(0, lambda: self.log(f"{name} ..."))
            try:
                ret = func()
            except Exception as exc:
                self.after(0, lambda: self._command_failed(name, exc))
                return
            self.after(0, lambda: self._command_done(name, ret))

        threading.Thread(target=work, daemon=True).start()

    def _command_done(self, name: str, ret: Any) -> None:
        if isinstance(ret, list):
            text = ", ".join(ret_text(v) for v in ret)
            ok = all(sdk_ok(v) for v in ret if not isinstance(v, str))
        else:
            text = ret_text(ret)
            ok = sdk_ok(ret)
        self.log(f"{name} 返回：{text}" + ("" if ok else "  [请检查控制器状态]"))

    def _command_failed(self, name: str, exc: Exception) -> None:
        self.log(f"{name} 失败：{exc}")
        messagebox.showerror(f"{name}失败", str(exc))

    def start_status_loop(self) -> None:
        self.status_token += 1
        token = self.status_token
        if self.status_running:
            return
        self.status_running = True

        def loop() -> None:
            try:
                while token == self.status_token and self.session.connected:
                    try:
                        snap = self.session.snapshot()
                    except Exception as exc:
                        self.after(0, lambda e=exc: self.log(f"状态读取失败：{e}"))
                        break
                    self.after(0, lambda s=snap: self.apply_snapshot(s))
                    time.sleep(0.25)
            finally:
                self.status_running = False

        threading.Thread(target=loop, daemon=True).start()

    def apply_snapshot(self, snap: dict[str, Any]) -> None:
        self.last_snapshot = snap
        self.power_var.set("是" if snap["power_on"] else "否")
        self.steady_var.set("是" if snap["steady"] else "否")
        self.mode_var.set(snap["robot_mode"])
        self.safety_var.set(snap["safety_mode"])
        self.collision_var.set("是" if snap["collision"] else "否")
        self.queue_var.set(str(snap.get("queue_size", "-")))
        joints = snap["joints_rad"]
        tcp = snap["tcp_pose_m_rad"]
        for i, value in enumerate(joints[:6]):
            self.joint_value_vars[i].set(f"{rad_to_deg(value): .3f}")
        for i, value in enumerate(tcp[:6]):
            shown = m_to_mm(value) if i < 3 else rad_to_deg(value)
            self.tcp_value_vars[i].set(f"{shown: .3f}")

    def selected_motion_frame(self) -> str:
        return "tool" if self.motion_frame_var.get().startswith("工具") else "base"

    def build_relative_tcp_target(self, pose_m_rad: list[float], index: int, signed_step: float) -> list[float]:
        target = [float(v) for v in pose_m_rad[:6]]
        frame = self.selected_motion_frame()
        rx, ry, rz = target[3:6]
        current_r = rpy_to_matrix(rx, ry, rz)

        if index < 3:
            delta = [0.0, 0.0, 0.0]
            delta[index] = signed_step
            if frame == "tool":
                delta = _mat_vec_mul(current_r, delta)
            for i in range(3):
                target[i] += delta[i]
            return target

        axis = index - 3
        step_r = _axis_rot(axis, signed_step)
        if frame == "tool":
            next_r = _mat_mul(current_r, step_r)
        else:
            next_r = _mat_mul(step_r, current_r)
        target[3:6] = matrix_to_rpy(next_r)
        return target

    def start_joint_jog(self, index: int, sign: int) -> None:
        try:
            speed = deg_to_rad(read_float(self.joint_speed_var, "关节速度", 0.0))
            acc = deg_to_rad(read_float(self.joint_acc_var, "关节加速度", 0.0))
            step = deg_to_rad(read_float(self.joint_step_var, "关节点动步长", 0.0))
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self.start_step_jog("关节", lambda: self._joint_step(index, sign, step, speed, acc))

    def start_tcp_jog(self, index: int, sign: int) -> None:
        try:
            speed = mm_to_m(read_float(self.tcp_speed_var, "TCP速度", 0.0))
            acc = mm_to_m(read_float(self.tcp_acc_var, "TCP加速度", 0.0))
            xyz_step = mm_to_m(read_float(self.line_step_var, "XYZ点动步长", 0.0))
            rot_step = deg_to_rad(read_float(self.rot_step_var, "姿态点动步长", 0.0))
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        step = xyz_step if index < 3 else rot_step
        self.start_step_jog("TCP", lambda: self._tcp_step(index, sign, step, speed, acc))

    def _joint_step(self, index: int, sign: int, step_rad: float, speed_rad_s: float, acc_rad_s2: float) -> list[Any]:
        snap = self.session.snapshot()
        target = list(snap["joints_rad"])
        target[index] += float(sign) * step_rad
        return self.session.move_joint(target, speed_rad_s, acc_rad_s2)

    def _tcp_step(self, index: int, sign: int, step_m_or_rad: float, speed_m_s: float, acc_m_s2: float) -> list[Any]:
        snap = self.session.snapshot()
        target = self.build_relative_tcp_target(snap["tcp_pose_m_rad"], index, float(sign) * step_m_or_rad)
        return self.session.move_line(target, speed_m_s, acc_m_s2)

    def wait_after_step(self, timeout_s: float = 2.0) -> None:
        deadline = time.time() + timeout_s
        time.sleep(0.08)
        while time.time() < deadline:
            try:
                snap = self.session.snapshot()
                if bool(snap.get("steady", False)):
                    return
            except Exception:
                return
            time.sleep(0.04)

    def start_step_jog(self, mode: str, step_func: Callable[[], list[Any]]) -> None:
        self.stop_jog(log_stop=False)
        event = threading.Event()
        event.set()
        with self.jog_lock:
            self.jog_event = event
        self.log(f"开始{mode}分步点动")

        def loop() -> None:
            last_bad: str | None = None
            while event.is_set():
                try:
                    rets = step_func()
                    bad = [ret for ret in rets if not isinstance(ret, str) and not sdk_ok(ret)]
                    if bad:
                        text = ", ".join(ret_text(ret) for ret in bad)
                        if text != last_bad:
                            last_bad = text
                            self.after(0, lambda t=text: self.log(f"{mode}分步点动返回异常：{t}"))
                    self.wait_after_step()
                except Exception as exc:
                    self.after(0, lambda e=exc: self.log(f"{mode}分步点动失败：{e}"))
                    break
                time.sleep(0.03)

        threading.Thread(target=loop, daemon=True).start()

    def stop_jog(self, log_stop: bool = True) -> None:
        with self.jog_lock:
            event = self.jog_event
            self.jog_event = None
        if event is not None:
            event.clear()
            if log_stop:
                self.log("停止连续运动")
            self.run_command("停止运动", self.session.stop_motion)

    def selected_point_index(self) -> int | None:
        sel = self.point_list.curselection()
        if not sel:
            return None
        idx = int(sel[0])
        if idx < 0 or idx >= len(self.points):
            return None
        return idx

    def _refresh_points(self) -> None:
        self.point_list.delete(0, END)
        for point in self.points:
            self.point_list.insert(END, point.name)

    def show_selected_point(self) -> None:
        idx = self.selected_point_index()
        self.point_detail.delete("1.0", END)
        if idx is None:
            return
        p = self.points[idx]
        lines = [
            f"名称：{p.name}",
            f"时间：{p.created_at}",
            "",
            "关节角 deg：",
            json.dumps([round(rad_to_deg(v), 6) for v in p.joints_rad], ensure_ascii=False),
            "",
            "TCP 位姿 mm/deg：",
            json.dumps(
                [round(m_to_mm(v), 6) if i < 3 else round(rad_to_deg(v), 6) for i, v in enumerate(p.tcp_pose_m_rad)],
                ensure_ascii=False,
            ),
            "",
            "SDK 原始 joints_rad：",
            json.dumps(p.joints_rad, ensure_ascii=False),
            "",
            "SDK 原始 tcp_pose_m_rad：",
            json.dumps(p.tcp_pose_m_rad, ensure_ascii=False),
        ]
        self.point_detail.insert(END, "\n".join(lines))

    def set_current_as_home_point(self) -> None:
        snap = self.last_snapshot
        if not snap:
            messagebox.showwarning("没有状态", "请先连接机械臂并等待状态刷新。")
            return
        point = SavedPoint(
            name="原始点",
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            joints_rad=list(snap["joints_rad"]),
            tcp_pose_m_rad=list(snap["tcp_pose_m_rad"]),
        )
        self.home_point = point
        save_home_point(point)
        self.home_var.set(self.home_label())
        self.log("已将当前位置设置为原始点")

    def reset_to_home_point(self) -> None:
        if self.home_point is None:
            messagebox.showwarning("未设置原始点", "请先点击“设当前为原始点”。")
            return
        try:
            speed = deg_to_rad(read_float(self.ptp_speed_var, "点到点速度", 0.0))
            acc = deg_to_rad(read_float(self.ptp_acc_var, "点到点加速度", 0.0))
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        point = self.home_point
        self.run_command("复位到原始点", lambda: self.session.move_joint(point.joints_rad, speed, acc))

    def capture_point(self) -> None:
        snap = self.last_snapshot
        if not snap:
            messagebox.showwarning("没有状态", "请先连接机械臂并等待状态刷新。")
            return
        default_name = f"P{len(self.points) + 1:03d}"
        name = simpledialog.askstring("采集当前点", "点位名称：", initialvalue=default_name, parent=self)
        if not name:
            return
        point = SavedPoint(
            name=name.strip(),
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            joints_rad=list(snap["joints_rad"]),
            tcp_pose_m_rad=list(snap["tcp_pose_m_rad"]),
        )
        self.points.append(point)
        save_points(self.points)
        self._refresh_points()
        self.point_list.selection_clear(0, END)
        self.point_list.selection_set(len(self.points) - 1)
        self.show_selected_point()
        self.log(f"已采集点位：{point.name}")

    def move_to_selected_point(self) -> None:
        idx = self.selected_point_index()
        if idx is None:
            messagebox.showinfo("提示", "请先选择一个点位。")
            return
        point = self.points[idx]
        if not messagebox.askyesno("确认移动", f"确定移动到点位：{point.name}？"):
            return
        try:
            speed = deg_to_rad(read_float(self.ptp_speed_var, "点到点速度", 0.0))
            acc = deg_to_rad(read_float(self.ptp_acc_var, "点到点加速度", 0.0))
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self.run_command(f"移动到点位 {point.name}", lambda: self.session.move_joint(point.joints_rad, speed, acc))

    def delete_selected_point(self) -> None:
        idx = self.selected_point_index()
        if idx is None:
            messagebox.showinfo("提示", "请先选择一个点位。")
            return
        point = self.points[idx]
        if not messagebox.askyesno("删除点位", f"确定删除 {point.name}？"):
            return
        del self.points[idx]
        save_points(self.points)
        self._refresh_points()
        self.point_detail.delete("1.0", END)
        self.log(f"已删除点位：{point.name}")

    def clear_points(self) -> None:
        if not self.points:
            return
        if not messagebox.askyesno("清空点位", "确定清空所有采集点位？"):
            return
        self.points.clear()
        save_points(self.points)
        self._refresh_points()
        self.point_detail.delete("1.0", END)
        self.log("已清空点位")

    def relative_line_move(self, index: int, sign: int) -> None:
        snap = self.last_snapshot
        if not snap:
            messagebox.showwarning("没有状态", "请先连接机械臂并等待状态刷新。")
            return
        try:
            speed = mm_to_m(read_float(self.line_speed_var, "直线速度", 0.0))
            acc = mm_to_m(read_float(self.line_acc_var, "直线加速度", 0.0))
            xyz_step = mm_to_m(read_float(self.line_step_var, "XYZ步长", 0.0))
            rot_step = deg_to_rad(read_float(self.rot_step_var, "姿态步长", 0.0))
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        step = xyz_step if index < 3 else rot_step
        target = self.build_relative_tcp_target(snap["tcp_pose_m_rad"], index, float(sign) * step)
        name = ["X", "Y", "Z", "Rx", "Ry", "Rz"][index]
        frame = self.motion_frame_var.get()
        self.run_command(f"{name} 相对移动({frame})", lambda: self.session.move_line(target, speed, acc))

    def on_close(self) -> None:
        self.status_token += 1
        with self.jog_lock:
            event = self.jog_event
            self.jog_event = None
        if event is not None:
            event.clear()
        try:
            if self.session.connected:
                try:
                    self.session.stop_motion()
                except Exception:
                    pass
                self.session.disconnect()
        finally:
            self.destroy()


class AuboMotionGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AUBO 机械臂上下电与运动控制")
        self.geometry("1320x860")
        self.minsize(760, 520)
        self.panel = AuboMotionPanel(self)
        self.panel.pack(fill=BOTH, expand=True)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self) -> None:
        self.panel.on_close()
        self.destroy()


def main() -> None:
    AuboMotionGui().mainloop()


if __name__ == "__main__":
    main()
