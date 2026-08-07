#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TCP 示教工具：采集若干个法兰位姿，用 4 点法求 TCP 偏移 XYZ，
再用 3 点法（原点/+X点/+Y点）求 Rx/Ry/Rz，最后可选择写入机械臂。

注意：这个会话（TcpTeachSession）会调用 `config.setTcpOffset()`，是全项目里
唯一真正会修改机械臂参数的地方；和 robot.py 里只读的 AuboPoseSession
故意保持两个独立的类，不共享基类，避免“只读手眼位姿读取”不小心跟
“会写 TCP 偏移的示教工具”混用。
"""

from __future__ import annotations

import itertools
import math
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import BOTH, END, LEFT, RIGHT, X, Y, filedialog, messagebox, ttk

from .sdk_paths import add_aubo_sdk_to_path

add_aubo_sdk_to_path()

import pyaubo_sdk as aubo  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_POINTS_FILE = DATA_DIR / "tcp_teach_points.json"


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def fmt(values: list[float], digits: int = 6) -> str:
    return "[" + ", ".join(f"{v:.{digits}f}" for v in values) + "]"


def fmt_xyz_mm(pose: list[float], digits: int = 3) -> str:
    if len(pose) < 3:
        return "-"
    return fmt([value * 1000.0 for value in pose[:3]], digits)


def yes_no(value: bool) -> str:
    return "是" if value else "否"


def vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def vec_dot(a: list[float], b: list[float]) -> float:
    return sum(a[i] * b[i] for i in range(3))


def vec_cross(a: list[float], b: list[float]) -> list[float]:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def vec_norm(v: list[float]) -> float:
    return math.sqrt(vec_dot(v, v))


def vec_normalize(v: list[float], name: str) -> list[float]:
    length = vec_norm(v)
    if length < 1e-6:
        raise RuntimeError(f"{name} 距离过小，无法确定方向")
    return [item / length for item in v]


def mat_transpose(m: list[list[float]]) -> list[list[float]]:
    return [[m[j][i] for j in range(3)] for i in range(3)]


def mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def quat_to_matrix(quat: list[float], order: str) -> list[list[float]]:
    if order == "wxyz":
        w, x, y, z = quat
    else:
        x, y, z, w = quat
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise RuntimeError("四元数长度为 0，无法转换姿态")
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def matrix_to_quat(matrix: list[list[float]], order: str) -> list[float]:
    m = matrix
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w, x, y, z = (m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w, x, y, z = (m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w, x, y, z = (m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s
    return [w, x, y, z] if order == "wxyz" else [x, y, z, w]


def err_text(ret: int) -> str:
    try:
        return aubo.returnValue2Str(ret)
    except Exception:
        return str(ret)


@dataclass
class TeachPoint:
    name: str
    timestamp: str
    tool_pose: list[float]
    tcp_pose: list[float]
    joints: list[float]
    actual_tcp_offset: list[float]


class TcpTeachSession:
    """TCP 示教专用会话：可以读位姿、可以写 TCP 偏移。"""

    def __init__(self) -> None:
        self.client: Any | None = None
        self.robot_if: Any | None = None
        self.state: Any | None = None
        self.config: Any | None = None
        self.math_api: Any | None = None
        self.quaternion_order = "wxyz"
        self.robot_name = ""
        self.robot_type = ""
        self.tcp_offset_cache: list[float] = []

    @property
    def connected(self) -> bool:
        return self.client is not None and bool(self.client.hasConnected())

    def network_precheck(self, ip: str, port: int, timeout_s: float = 1.0) -> tuple[bool, str]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout_s)
        try:
            sock.connect((ip, port))
            local_ip, local_port = sock.getsockname()
            return True, f"端口可达，本机出口 {local_ip}:{local_port}"
        except OSError as exc:
            return False, (
                f"无法连接 {ip}:{port}。\n系统错误: {exc}\n\n"
                "请检查：\n1. 机械臂控制柜是否已上电并启动完成。\n2. 网线是否接在当前电脑的“以太网”口。\n"
                "3. 本机以太网 IP 是否为 192.168.50.196/24。\n4. 机械臂控制器 IP 是否仍为 192.168.50.200。\n"
                "5. 控制器通信服务端口 30004 是否开启。"
            )
        finally:
            sock.close()

    def connect(self, ip: str, port: int, user: str, password: str, timeout_ms: int) -> str:
        self.disconnect()
        ok, message = self.network_precheck(ip, port)
        if not ok:
            raise RuntimeError(message)
        client = aubo.RpcClient()
        client.setRequestTimeout(timeout_ms)
        ret = client.connect(ip, port)
        if ret != aubo.AUBO_OK:
            raise RuntimeError(f"连接失败: {ret} {err_text(ret)}")
        ret = client.login(user, password)
        if ret != aubo.AUBO_OK:
            try:
                client.disconnect()
            finally:
                pass
            raise RuntimeError(f"登录失败: {ret} {err_text(ret)}")
        names = client.getRobotNames()
        if not names:
            client.disconnect()
            raise RuntimeError("未找到机械臂")
        self.client = client
        self.robot_name = names[0]
        self.robot_if = client.getRobotInterface(self.robot_name)
        self.state = self.robot_if.getRobotState()
        self.config = self.robot_if.getRobotConfig()
        self.math_api = client.getMath()
        self.quaternion_order = self.detect_quaternion_order()
        self.robot_type = self.config.getRobotType()
        self.tcp_offset_cache = list(self.config.getTcpOffset())
        return self.robot_name

    def disconnect(self) -> None:
        if self.client is not None:
            try:
                self.client.disconnect()
            except Exception:
                pass
        self.client = None
        self.robot_if = None
        self.state = None
        self.config = None
        self.math_api = None
        self.quaternion_order = "wxyz"
        self.robot_name = ""
        self.robot_type = ""
        self.tcp_offset_cache = []

    def detect_quaternion_order(self) -> str:
        if self.math_api is None:
            return "wxyz"
        quat = list(self.math_api.rpyToQuaternion([0.0, 0.0, 0.0]))
        if len(quat) == 4 and abs(quat[3]) > 0.9:
            return "xyzw"
        return "wxyz"

    def rpy_to_matrix(self, rpy: list[float]) -> list[list[float]]:
        if self.math_api is None:
            raise RuntimeError("未连接机械臂")
        quat = list(self.math_api.rpyToQuaternion(rpy))
        return quat_to_matrix(quat, self.quaternion_order)

    def matrix_to_rpy(self, matrix: list[list[float]]) -> list[float]:
        if self.math_api is None:
            raise RuntimeError("未连接机械臂")
        quat = matrix_to_quat(matrix, self.quaternion_order)
        return list(self.math_api.quaternionToRpy(quat))

    def snapshot(self) -> dict[str, Any]:
        if not self.connected or self.state is None or self.config is None:
            raise RuntimeError("未连接机械臂")
        self.tcp_offset_cache = list(self.config.getTcpOffset())
        return {
            "robot_name": self.robot_name, "robot_type": self.robot_type,
            "safety_mode": str(self.state.getSafetyModeType()), "robot_mode": str(self.state.getRobotModeType()),
            "power_on": bool(self.state.isPowerOn()), "steady": bool(self.state.isSteady()),
            "collision": bool(self.state.isCollisionOccurred()), "tcp_offset": list(self.tcp_offset_cache),
            "actual_tcp_offset": list(self.state.getActualTcpOffset()),
            "tool_pose": list(self.state.getToolPose()), "tcp_pose": list(self.state.getTcpPose()),
            "joints": list(self.state.getJointPositions()),
        }

    def live_snapshot(self) -> dict[str, Any]:
        if not self.connected or self.state is None:
            raise RuntimeError("未连接机械臂")
        return {
            "robot_name": self.robot_name, "robot_type": self.robot_type,
            "safety_mode": str(self.state.getSafetyModeType()), "robot_mode": str(self.state.getRobotModeType()),
            "power_on": bool(self.state.isPowerOn()), "steady": bool(self.state.isSteady()),
            "tcp_offset": list(self.tcp_offset_cache),
            "tool_pose": list(self.state.getToolPose()), "tcp_pose": list(self.state.getTcpPose()),
            "joints": list(self.state.getJointPositions()),
        }

    def capture(self, name: str) -> TeachPoint:
        snap = self.snapshot()
        if not snap["steady"]:
            raise RuntimeError("机械臂尚未静止，请等待停止后再采集")
        if snap["collision"]:
            raise RuntimeError("当前存在碰撞标志，请先清除故障后再采集")
        return TeachPoint(
            name=name, timestamp=now_text(), tool_pose=snap["tool_pose"], tcp_pose=snap["tcp_pose"],
            joints=snap["joints"], actual_tcp_offset=snap["actual_tcp_offset"],
        )

    def identify_tcp(self, points: list[TeachPoint]) -> dict[str, Any]:
        if not self.connected or self.math_api is None:
            raise RuntimeError("未连接机械臂")
        if len(points) < 4:
            raise RuntimeError("至少需要采集 4 个点")

        xyz_offsets: list[list[float]] = []
        failures: list[dict[str, Any]] = []
        for combo in itertools.combinations(points, 4):
            poses = [p.tool_pose for p in combo]
            offset, ret = self.math_api.tcpOffsetIdentify(poses)
            offset = list(offset)
            if ret == aubo.AUBO_OK and len(offset) >= 3:
                xyz_offsets.append(offset[:3])
            else:
                failures.append({
                    "points": [p.name for p in combo],
                    "ret": ret,
                    "message": err_text(ret),
                    "returned_length": len(offset),
                })

        if not xyz_offsets:
            raise RuntimeError(f"所有 4 点组合计算失败: {failures[:3]}")

        avg_xyz = [sum(values) / len(values) for values in zip(*xyz_offsets)]
        spread = [
            math.sqrt(sum((item[i] - avg_xyz[i]) ** 2 for item in xyz_offsets) / len(xyz_offsets))
            for i in range(3)
        ]
        current_rpy = self.tcp_offset_cache[3:6] if len(self.tcp_offset_cache) >= 6 else [0.0, 0.0, 0.0]
        full_offset = avg_xyz + current_rpy
        residuals = self.tip_residuals(points, full_offset)
        return {
            "offset": full_offset,
            "identified_xyz": avg_xyz,
            "offset_spread": spread,
            "valid_combinations": len(xyz_offsets),
            "failed_combinations": len(failures),
            "residuals": residuals,
        }

    def tip_residuals(self, points: list[TeachPoint], offset: list[float]) -> dict[str, Any]:
        if self.math_api is None:
            raise RuntimeError("未连接机械臂")
        tip_positions: list[list[float]] = []
        for point in points:
            tip_pose = list(self.math_api.poseTrans(point.tool_pose, offset))
            tip_positions.append(tip_pose[:3])
        centroid = [sum(axis) / len(tip_positions) for axis in zip(*tip_positions)]
        errors_mm = [math.sqrt(sum((pos[i] - centroid[i]) ** 2 for i in range(3))) * 1000.0 for pos in tip_positions]
        rms = math.sqrt(sum(e * e for e in errors_mm) / len(errors_mm))
        return {"tip_positions": tip_positions, "centroid": centroid, "errors_mm": errors_mm, "rms_mm": rms, "max_mm": max(errors_mm)}

    def apply_tcp(self, offset: list[float]) -> int:
        if not self.connected or self.config is None:
            raise RuntimeError("未连接机械臂")
        return int(self.config.setTcpOffset(offset))

    def tcp_position_from_tool_pose(self, tool_pose: list[float], offset_xyz: list[float]) -> list[float]:
        if self.math_api is None:
            raise RuntimeError("未连接机械臂")
        pose = list(self.math_api.poseTrans(tool_pose, [offset_xyz[0], offset_xyz[1], offset_xyz[2], 0.0, 0.0, 0.0]))
        return pose[:3]

    def identify_orientation(self, origin: TeachPoint, x_point: TeachPoint, y_point: TeachPoint, offset_xyz: list[float]) -> dict[str, Any]:
        if not self.connected or self.math_api is None:
            raise RuntimeError("未连接机械臂")

        p0 = self.tcp_position_from_tool_pose(origin.tool_pose, offset_xyz)
        px = self.tcp_position_from_tool_pose(x_point.tool_pose, offset_xyz)
        py = self.tcp_position_from_tool_pose(y_point.tool_pose, offset_xyz)

        x_raw = vec_sub(px, p0)
        y_raw = vec_sub(py, p0)
        x_len, y_len = vec_norm(x_raw), vec_norm(y_raw)
        x_axis = vec_normalize(x_raw, "+X 方向点")
        y_hint = vec_normalize(y_raw, "+Y 平面点")
        cos_angle = max(-1.0, min(1.0, vec_dot(x_axis, y_hint)))
        angle_deg = math.degrees(math.acos(cos_angle))
        if abs(math.sin(math.radians(angle_deg))) < 0.15:
            raise RuntimeError("原点、+X点、+Y点接近共线，无法稳定计算姿态")

        y_axis = [y_hint[i] - vec_dot(y_hint, x_axis) * x_axis[i] for i in range(3)]
        y_axis = vec_normalize(y_axis, "+Y 正交方向")
        z_axis = vec_normalize(vec_cross(x_axis, y_axis), "+Z 方向")
        y_axis = vec_cross(z_axis, x_axis)

        base_to_tcp = [[x_axis[0], y_axis[0], z_axis[0]], [x_axis[1], y_axis[1], z_axis[1]], [x_axis[2], y_axis[2], z_axis[2]]]
        base_to_flange = self.rpy_to_matrix(origin.tool_pose[3:])
        flange_to_tcp = mat_mul(mat_transpose(base_to_flange), base_to_tcp)
        rpy = self.matrix_to_rpy(flange_to_tcp)

        return {
            "rpy": rpy, "origin_position": p0, "x_position": px, "y_position": py,
            "x_length_mm": x_len * 1000.0, "y_length_mm": y_len * 1000.0, "xy_angle_deg": angle_deg,
        }


class TcpTeachPanel(ttk.Frame):
    def __init__(self, master: tk.Misc | None = None) -> None:
        super().__init__(master)
        self.session = TcpTeachSession()
        self.points: list[TeachPoint] = []
        self.orientation_points: dict[str, TeachPoint] = {}
        self.result_offset: list[float] | None = None
        self.result_base_xyz: list[float] | None = None
        self.poll_after_id: str | None = None
        self.last_poll_error = ""

        self.ip_var = tk.StringVar(value="192.168.50.200")
        self.port_var = tk.StringVar(value="30004")
        self.user_var = tk.StringVar(value="AUBO")
        self.password_var = tk.StringVar(value="123456")
        self.status_var = tk.StringVar(value="未连接")
        self.robot_var = tk.StringVar(value="-")
        self.mode_var = tk.StringVar(value="-")
        self.tcp_offset_var = tk.StringVar(value="-")
        self.flange_xyz_mm_var = tk.StringVar(value="-")
        self.tcp_xyz_mm_var = tk.StringVar(value="-")
        self.tool_pose_var = tk.StringVar(value="-")
        self.tcp_pose_var = tk.StringVar(value="-")
        self.joints_var = tk.StringVar(value="-")
        self.result_var = tk.StringVar(value="-")
        self.result_base_xyz_var = tk.StringVar(value="-")
        self.quality_var = tk.StringVar(value="-")
        self.orient_points_var = tk.StringVar(value="原点: 未采集    +X点: 未采集    +Y点: 未采集")
        self.orient_result_var = tk.StringVar(value="-")

        self._build()
        self.start_polling()

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill=BOTH, expand=True)

        top = ttk.LabelFrame(outer, text="TCP 示教会话", padding=8)
        top.pack(fill=X)
        ttk.Label(top, text="连接参数使用工作台顶部统一设置。", foreground="#555555").pack(side=LEFT, padx=(0, 12))
        self.connect_btn = ttk.Button(top, text="连接机械臂", command=self.toggle_connect)
        self.connect_btn.pack(side=LEFT, padx=(0, 8))
        ttk.Label(top, textvariable=self.status_var).pack(side=LEFT, padx=8)

        info = ttk.LabelFrame(outer, text="机械臂状态")
        info.pack(fill=X, pady=(10, 8))
        self._info_row(info, 0, "机器人", self.robot_var)
        self._info_row(info, 1, "模式/安全", self.mode_var)
        self._info_row(info, 2, "TCP偏移(法兰系)", self.tcp_offset_var)
        self._info_row(info, 3, "基坐标法兰XYZ(mm)", self.flange_xyz_mm_var)
        self._info_row(info, 4, "基坐标TCP XYZ(mm)", self.tcp_xyz_mm_var)
        self._info_row(info, 5, "基坐标法兰位姿", self.tool_pose_var)
        self._info_row(info, 6, "基坐标TCP位姿", self.tcp_pose_var)
        self._info_row(info, 7, "关节角(rad)", self.joints_var)

        mid = ttk.Frame(outer)
        mid.pack(fill=BOTH, expand=True)

        left = ttk.Frame(mid)
        left.pack(side=LEFT, fill=BOTH, expand=True)

        toolbar = ttk.Frame(left)
        toolbar.pack(fill=X, pady=(0, 6))
        # 同一个连接按钮只保留在上方会话区；保留别名兼容已有状态刷新逻辑。
        self.connect_action_btn = self.connect_btn
        ttk.Button(toolbar, text="采集当前点", command=self.capture_point).pack(side=LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="删除选中", command=self.delete_selected).pack(side=LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="清空", command=self.clear_points).pack(side=LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="加载", command=self.load_points).pack(side=LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="保存", command=self.save_points).pack(side=LEFT, padx=(0, 12))
        ttk.Button(toolbar, text="计算TCP", command=self.compute_tcp).pack(side=LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="写入TCP", command=self.apply_tcp).pack(side=LEFT)

        columns = ("name", "time", "tool_xyz", "tool_rpy", "tcp_xyz")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", height=13)
        for col, label, width in [
            ("name", "点位", 80), ("time", "采集时间", 150),
            ("tool_xyz", "基坐标法兰 XYZ(mm)", 220),
            ("tool_rpy", "法兰 RPY(rad)", 220),
            ("tcp_xyz", "基坐标 TCP XYZ(mm)", 220),
        ]:
            self.tree.heading(col, text=label)
            self.tree.column(col, width=width, anchor="w", stretch=True)
        self.tree.pack(fill=BOTH, expand=True)

        result = ttk.LabelFrame(left, text="标定结果")
        result.pack(fill=X, pady=(8, 0))
        self._info_row(result, 0, "TCP偏移(法兰系)", self.result_var)
        self._info_row(result, 1, "公共点基坐标XYZ", self.result_base_xyz_var)
        self._info_row(result, 2, "质量评估", self.quality_var)

        orient = ttk.LabelFrame(left, text="姿态标定（三点法计算 Rx/Ry/Rz）")
        orient.pack(fill=X, pady=(8, 0))
        orient_toolbar = ttk.Frame(orient)
        orient_toolbar.grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(4, 2))
        ttk.Button(orient_toolbar, text="采集方向原点", command=lambda: self.capture_orientation_point("origin")).pack(side=LEFT, padx=(0, 6))
        ttk.Button(orient_toolbar, text="采集+X方向点", command=lambda: self.capture_orientation_point("x")).pack(side=LEFT, padx=(0, 6))
        ttk.Button(orient_toolbar, text="采集+Y平面点", command=lambda: self.capture_orientation_point("y")).pack(side=LEFT, padx=(0, 6))
        ttk.Button(orient_toolbar, text="计算Rx/Ry/Rz", command=self.compute_orientation).pack(side=LEFT, padx=(0, 6))
        ttk.Button(orient_toolbar, text="清空方向点", command=self.clear_orientation_points).pack(side=LEFT)
        self._info_row(orient, 1, "方向点", self.orient_points_var)
        self._info_row(orient, 2, "姿态结果", self.orient_result_var)

        right = ttk.Frame(mid, width=320)
        right.pack(side=RIGHT, fill=Y, padx=(10, 0))
        ttk.Label(right, text="日志").pack(anchor="w")
        self.log_text = tk.Text(right, height=28, width=42, wrap="word")
        self.log_text.pack(fill=BOTH, expand=True)

    def _info_row(self, parent: ttk.Frame, row: int, name: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=name, width=18).grid(row=row, column=0, sticky="w", padx=6, pady=2)
        ttk.Label(parent, textvariable=var).grid(row=row, column=1, sticky="w", padx=6, pady=2)
        parent.columnconfigure(1, weight=1)

    def log(self, text: str) -> None:
        self.log_text.insert(END, f"{now_text()}  {text}\n")
        self.log_text.see(END)

    def set_connection_ui(self, connected: bool) -> None:
        if connected:
            self.status_var.set("已连接")
            self.connect_btn.config(text="断开机械臂")
            self.connect_action_btn.config(text="断开机械臂")
        else:
            self.status_var.set("未连接")
            self.connect_btn.config(text="连接机械臂")
            self.connect_action_btn.config(text="连接机械臂")

    def toggle_connect(self) -> None:
        if self.session.connected:
            self.session.disconnect()
            self.set_connection_ui(False)
            self.log("已断开连接")
            return
        try:
            robot_name = self.session.connect(
                self.ip_var.get().strip(), int(self.port_var.get().strip()),
                self.user_var.get().strip(), self.password_var.get(), timeout_ms=1500,
            )
        except Exception as exc:
            self.status_var.set("连接失败")
            self.log(f"连接失败: {exc}")
            messagebox.showerror("连接失败", str(exc))
            return
        self.set_connection_ui(True)
        self.log(f"连接成功: {robot_name}")

    def start_polling(self) -> None:
        if self.poll_after_id is None:
            self.poll_after_id = self.after(1000, self.poll_robot)

    def stop_polling(self) -> None:
        if self.poll_after_id is not None:
            try:
                self.after_cancel(self.poll_after_id)
            except Exception:
                pass
            self.poll_after_id = None

    def poll_robot(self) -> None:
        self.poll_after_id = None
        if self.session.connected:
            try:
                snap = self.session.live_snapshot()
                self.robot_var.set(f"{snap['robot_name']} / {snap['robot_type']}")
                self.mode_var.set(
                    f"{snap['robot_mode']} / {snap['safety_mode']} / 上电={yes_no(snap['power_on'])} 静止={yes_no(snap['steady'])}"
                )
                self.tcp_offset_var.set(fmt(snap["tcp_offset"]))
                self.flange_xyz_mm_var.set(fmt_xyz_mm(snap["tool_pose"]))
                self.tcp_xyz_mm_var.set(fmt_xyz_mm(snap["tcp_pose"]))
                self.tool_pose_var.set(fmt(snap["tool_pose"]))
                self.tcp_pose_var.set(fmt(snap["tcp_pose"]))
                self.joints_var.set(fmt(snap["joints"]))
                self.last_poll_error = ""
            except Exception as exc:
                message = str(exc)
                self.status_var.set("读取失败")
                if message != self.last_poll_error:
                    self.log(f"读取失败，已断开连接: {message}")
                    self.last_poll_error = message
                self.session.disconnect()
                self.set_connection_ui(False)
        self.poll_after_id = self.after(1000, self.poll_robot)

    def capture_point(self) -> None:
        try:
            point = self.session.capture(f"P{len(self.points) + 1}")
        except Exception as exc:
            self.log(f"采集失败: {exc}")
            messagebox.showerror("采集失败", str(exc))
            return
        self.points.append(point)
        self.refresh_points()
        self.log(f"已采集 {point.name}")

    def refresh_points(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for idx, point in enumerate(self.points):
            self.tree.insert("", END, iid=str(idx), values=(
                point.name,
                point.timestamp,
                fmt_xyz_mm(point.tool_pose),
                fmt(point.tool_pose[3:]),
                fmt_xyz_mm(point.tcp_pose),
            ))

    def selected_indexes(self) -> list[int]:
        indexes = []
        for item in self.tree.selection():
            try:
                indexes.append(int(item))
            except ValueError:
                pass
        return sorted(indexes, reverse=True)

    def delete_selected(self) -> None:
        for idx in self.selected_indexes():
            if 0 <= idx < len(self.points):
                self.points.pop(idx)
        for i, point in enumerate(self.points, start=1):
            point.name = f"P{i}"
        self.refresh_points()
        self.log("已删除选中点")

    def clear_points(self) -> None:
        if self.points and not messagebox.askyesno("清空", "确定清空所有已采集点吗？"):
            return
        self.points.clear()
        self.result_offset = None
        self.result_base_xyz = None
        self.result_var.set("-")
        self.result_base_xyz_var.set("-")
        self.quality_var.set("-")
        self.refresh_points()
        self.log("已清空采集点")

    def update_orientation_points_label(self) -> None:
        labels = {"origin": "原点", "x": "+X点", "y": "+Y点"}
        parts = []
        for key, label in labels.items():
            point = self.orientation_points.get(key)
            parts.append(f"{label}: {'未采集' if point is None else point.timestamp}")
        self.orient_points_var.set("    ".join(parts))

    def capture_orientation_point(self, key: str) -> None:
        names = {"origin": "方向原点", "x": "+X方向点", "y": "+Y平面点"}
        try:
            point = self.session.capture(names[key])
        except Exception as exc:
            self.log(f"{names[key]}采集失败: {exc}")
            messagebox.showerror("采集失败", str(exc))
            return
        self.orientation_points[key] = point
        self.update_orientation_points_label()
        self.log(f"已采集{names[key]}")

    def clear_orientation_points(self) -> None:
        self.orientation_points.clear()
        self.orient_result_var.set("-")
        self.update_orientation_points_label()
        self.log("已清空方向点")

    def compute_orientation(self) -> None:
        missing = [name for key, name in [("origin", "方向原点"), ("x", "+X方向点"), ("y", "+Y平面点")] if key not in self.orientation_points]
        if missing:
            messagebox.showwarning("方向点不足", "请先采集: " + "、".join(missing))
            return

        if self.result_offset is None:
            if not messagebox.askyesno("未计算XYZ", "尚未计算新的 TCP XYZ，是否使用当前机械臂 TCP 的 XYZ 来计算 Rx/Ry/Rz？"):
                return
            try:
                snap = self.session.snapshot()
            except Exception as exc:
                messagebox.showerror("读取失败", str(exc))
                return
            self.result_offset = list(snap["tcp_offset"])

        try:
            result = self.session.identify_orientation(
                self.orientation_points["origin"], self.orientation_points["x"], self.orientation_points["y"], self.result_offset[:3],
            )
        except Exception as exc:
            self.log(f"姿态计算失败: {exc}")
            messagebox.showerror("姿态计算失败", str(exc))
            return

        self.result_offset = self.result_offset[:3] + result["rpy"]
        self.result_var.set(fmt(self.result_offset))
        self.orient_result_var.set(
            f"RxRyRz={fmt(result['rpy'])}, X距离={result['x_length_mm']:.2f} mm, "
            f"Y距离={result['y_length_mm']:.2f} mm, 夹角={result['xy_angle_deg']:.2f} deg"
        )
        self.log(f"Rx/Ry/Rz 计算完成: {fmt(result['rpy'])}")

    def save_points(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            initialdir=str(DATA_DIR), initialfile=DEFAULT_POINTS_FILE.name, defaultextension=".json",
            filetypes=[("JSON 文件", "*.json")],
        )
        if not path:
            return
        import json

        payload = {
            "saved_at": now_text(),
            "points": [asdict(point) for point in self.points],
            "orientation_points": {key: asdict(point) for key, point in self.orientation_points.items()},
            "result_offset": self.result_offset,
            "result_base_xyz": self.result_base_xyz,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.log(f"已保存: {path}")

    def load_points(self) -> None:
        path = filedialog.askopenfilename(initialdir=str(DATA_DIR), filetypes=[("JSON 文件", "*.json")])
        if not path:
            return
        import json

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.points = [TeachPoint(**item) for item in payload.get("points", [])]
        self.orientation_points = {key: TeachPoint(**item) for key, item in payload.get("orientation_points", {}).items()}
        self.result_offset = payload.get("result_offset")
        self.result_base_xyz = payload.get("result_base_xyz")
        self.result_var.set(fmt(self.result_offset) if self.result_offset else "-")
        self.result_base_xyz_var.set(
            f"{fmt_xyz_mm(self.result_base_xyz)} mm" if self.result_base_xyz else "-"
        )
        self.quality_var.set("-")
        self.orient_result_var.set("-")
        self.refresh_points()
        self.update_orientation_points_label()
        self.log(f"已加载: {path}")

    def compute_tcp(self) -> None:
        try:
            result = self.session.identify_tcp(self.points)
        except Exception as exc:
            self.log(f"计算失败: {exc}")
            messagebox.showerror("计算失败", str(exc))
            return
        self.result_offset = result["offset"]
        residuals = result["residuals"]
        self.result_base_xyz = residuals["centroid"]
        spread = result["offset_spread"]
        self.result_var.set(fmt(self.result_offset))
        self.result_base_xyz_var.set(f"{fmt_xyz_mm(self.result_base_xyz)} mm")
        self.quality_var.set(
            f"有效组合={result['valid_combinations']}, 失败组合={result['failed_combinations']}, "
            f"RMS={residuals['rms_mm']:.3f} mm, 最大误差={residuals['max_mm']:.3f} mm, "
            f"XYZ离散度={fmt([v * 1000 for v in spread[:3]], 3)} mm"
        )
        self.log(
            f"TCP XYZ 计算完成: {fmt_xyz_mm(result['identified_xyz'])} mm；"
            f"完整偏移(m/rad): {fmt(self.result_offset)}"
        )

    def apply_tcp(self) -> None:
        if self.result_offset is None:
            messagebox.showwarning("没有结果", "请先计算 TCP。")
            return
        if not messagebox.askyesno("写入 TCP", "确定将此 TCP 偏移写入机械臂吗？\n\n" + fmt(self.result_offset)):
            return
        try:
            ret = self.session.apply_tcp(self.result_offset)
        except Exception as exc:
            self.log(f"写入失败: {exc}")
            messagebox.showerror("写入失败", str(exc))
            return
        self.log(f"写入 TCP 返回: {ret} {err_text(ret)}")
        if ret != aubo.AUBO_OK:
            messagebox.showerror("写入失败", f"{ret} {err_text(ret)}")

    def on_close(self) -> None:
        self.stop_polling()
        self.session.disconnect()


class TcpTeachApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AUBO TCP 示教工具")
        self.geometry("1180x760")
        self.minsize(1040, 680)
        self.panel = TcpTeachPanel(self)
        self.panel.pack(fill=BOTH, expand=True)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self) -> None:
        self.panel.on_close()
        self.destroy()


if __name__ == "__main__":
    TcpTeachApp().mainloop()
