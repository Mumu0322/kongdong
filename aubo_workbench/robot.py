#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手眼标定用的 AUBO 会话：只读 TCP/Tool 位姿，绝不下发运动指令。

注意：项目里还有一个功能完全不同的 TCP 示教会话（可以写 TCP 偏移、
做 4 点/3 点标定），定义在 tcp_teach.py 的 TcpTeachSession 里，
两者故意不合并、不共用基类，避免“只读手眼位姿读取”和“会修改机械臂参数
的示教工具”被误用串线。
"""

from __future__ import annotations

import math
import socket
from collections import OrderedDict
from datetime import datetime
from typing import Any

from .config import ROBOT_CFG
from .geometry import ensure_finite_array, make_transform, rotx, roty, rotz
from .sdk_paths import add_aubo_sdk_to_path, aubo_sdk_hint

SDK_DIR = add_aubo_sdk_to_path()

try:
    import pyaubo_sdk as aubo  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        f"未能导入 pyaubo_sdk。请确认 AUBO SDK 位于以下任一路径：{aubo_sdk_hint()}，"
        "或已安装到当前 Python 环境。"
    ) from exc


def err_text(ret: int) -> str:
    try:
        return aubo.returnValue2Str(ret)
    except Exception:
        return str(ret)


def quat_to_matrix(quat: list[float], order: str):
    if len(quat) != 4:
        raise RuntimeError(f"四元数长度异常: {quat}")
    if order == "wxyz":
        w, x, y, z = [float(v) for v in quat]
    else:
        x, y, z, w = [float(v) for v in quat]
    import numpy as np

    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise RuntimeError("四元数长度为 0，无法转换姿态")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rpy_zyx_fallback_to_matrix(rpy_rad: list[float]):
    """手动位姿兜底转换；在线读取优先使用 AUBO SDK Math.rpyToQuaternion。"""
    rx, ry, rz = [float(v) for v in rpy_rad]
    return rotz(rz) @ roty(ry) @ rotx(rx)


class AuboPoseSession:
    """只读会话：连接/登录、读取一次 TCP 位姿快照、断开。不做任何写操作。"""

    def __init__(self) -> None:
        self.client: Any | None = None
        self.robot_if: Any | None = None
        self.state: Any | None = None
        self.config: Any | None = None
        self.math_api: Any | None = None
        self.robot_name = ""
        self.robot_type = ""
        self.quaternion_order = "wxyz"

    @property
    def connected(self) -> bool:
        return self.client is not None and bool(self.client.hasConnected())

    def network_precheck(self) -> tuple[bool, str]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(float(ROBOT_CFG.network_precheck_timeout_s))
        try:
            sock.connect((ROBOT_CFG.ip, int(ROBOT_CFG.rpc_port)))
            local_ip, local_port = sock.getsockname()
            return True, f"端口可达，本机出口 {local_ip}:{local_port}"
        except OSError as exc:
            return False, f"无法连接 AUBO {ROBOT_CFG.ip}:{ROBOT_CFG.rpc_port}。系统错误: {exc}"
        finally:
            sock.close()

    def connect(self) -> None:
        if self.connected:
            return
        ok, message = self.network_precheck()
        if not ok:
            raise RuntimeError(message)

        client = aubo.RpcClient()
        client.setRequestTimeout(int(ROBOT_CFG.request_timeout_ms))
        ret = client.connect(ROBOT_CFG.ip, int(ROBOT_CFG.rpc_port))
        if ret != aubo.AUBO_OK:
            raise RuntimeError(f"AUBO 连接失败: {ret} {err_text(ret)}")
        ret = client.login(ROBOT_CFG.user, ROBOT_CFG.password)
        if ret != aubo.AUBO_OK:
            try:
                client.disconnect()
            finally:
                pass
            raise RuntimeError(f"AUBO 登录失败: {ret} {err_text(ret)}")

        names = list(client.getRobotNames())
        if not names:
            client.disconnect()
            raise RuntimeError("AUBO SDK 未返回机械臂名称")

        self.client = client
        self.robot_name = str(names[0])
        self.robot_if = client.getRobotInterface(self.robot_name)
        self.state = self.robot_if.getRobotState()
        self.config = self.robot_if.getRobotConfig()
        self.math_api = client.getMath()
        self.quaternion_order = self.detect_quaternion_order()
        try:
            self.robot_type = str(self.config.getRobotType())
        except Exception:
            self.robot_type = ""
        print(f"[AUBO] 已连接 {ROBOT_CFG.ip}:{ROBOT_CFG.rpc_port}, robot={self.robot_name}, type={self.robot_type}")

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
        self.robot_name = ""
        self.robot_type = ""
        self.quaternion_order = "wxyz"

    def detect_quaternion_order(self) -> str:
        if self.math_api is None:
            return "wxyz"
        quat = list(self.math_api.rpyToQuaternion([0.0, 0.0, 0.0]))
        if len(quat) == 4 and abs(float(quat[3])) > 0.9:
            return "xyzw"
        return "wxyz"

    def rpy_to_matrix(self, rpy_rad: list[float]):
        if self.math_api is None:
            return rpy_zyx_fallback_to_matrix(rpy_rad)
        quat = list(self.math_api.rpyToQuaternion([float(v) for v in rpy_rad]))
        return quat_to_matrix(quat, self.quaternion_order)

    def pose_sdk_to_transform_mm(self, pose_m_rad: list[float]):
        import numpy as np

        if len(pose_m_rad) < 6:
            raise RuntimeError(f"AUBO 位姿长度不足 6: {pose_m_rad}")
        values = np.asarray([float(v) for v in pose_m_rad[:6]], dtype=np.float64)
        ensure_finite_array(values, "aubo_pose_m_rad")
        R = self.rpy_to_matrix(values[3:6].tolist())
        t_mm = values[:3] * 1000.0
        return make_transform(R, t_mm)

    def read_pose_snapshot(self) -> dict[str, Any]:
        self.connect()
        if self.state is None:
            raise RuntimeError("AUBO RobotState 未初始化")

        power_on = bool(self.state.isPowerOn())
        steady = bool(self.state.isSteady())
        collision = bool(self.state.isCollisionOccurred())
        tool_pose = [float(v) for v in list(self.state.getToolPose())]
        tcp_pose = [float(v) for v in list(self.state.getTcpPose())]
        selected_pose = tcp_pose if ROBOT_CFG.pose_source.lower() == "tcp" else tool_pose
        pose_values_mm_deg = [
            selected_pose[0] * 1000.0,
            selected_pose[1] * 1000.0,
            selected_pose[2] * 1000.0,
            math.degrees(selected_pose[3]),
            math.degrees(selected_pose[4]),
            math.degrees(selected_pose[5]),
        ]
        named_values = OrderedDict(
            (label, round(float(value), 6))
            for label, value in zip(("X_mm", "Y_mm", "Z_mm", "Rx_deg", "Ry_deg", "Rz_deg"), pose_values_mm_deg)
        )

        snapshot: dict[str, Any] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "controller": {"ip": ROBOT_CFG.ip, "port": ROBOT_CFG.rpc_port, "protocol": "aubo_rpc"},
            "robot_brand": "AUBO",
            "robot_name": self.robot_name,
            "robot_type": self.robot_type,
            "pose_source": ROBOT_CFG.pose_source.lower(),
            "pose_units": {"xyz": "mm", "rpy": "deg"},
            "pose_values": [float(v) for v in pose_values_mm_deg],
            "pose_values_sdk_m_rad": [float(v) for v in selected_pose[:6]],
            "named_pose_values": named_values,
            "tcp_pose_sdk_m_rad": tcp_pose,
            "tool_pose_sdk_m_rad": tool_pose,
            "power_on": power_on,
            "steady": steady,
            "collision": collision,
            "robot_mode": str(self.state.getRobotModeType()),
            "safety_mode": str(self.state.getSafetyModeType()),
        }
        try:
            snapshot["joints_rad"] = [float(v) for v in list(self.state.getJointPositions())]
        except Exception:
            snapshot["joints_rad"] = []
        try:
            snapshot["actual_tcp_offset_sdk_m_rad"] = [float(v) for v in list(self.state.getActualTcpOffset())]
        except Exception:
            snapshot["actual_tcp_offset_sdk_m_rad"] = []
        try:
            if self.config is not None:
                snapshot["configured_tcp_offset_sdk_m_rad"] = [float(v) for v in list(self.config.getTcpOffset())]
        except Exception:
            snapshot["configured_tcp_offset_sdk_m_rad"] = []
        return snapshot


# 全局单例：整个手眼标定流程共用一条只读连接。
AUBO_SESSION = AuboPoseSession()


def close_aubo_session() -> None:
    AUBO_SESSION.disconnect()


def get_capture_pose_transform():
    """返回 (^B T_T, snapshot, status)，平移单位 mm。失败时 T 为 None。"""
    if ROBOT_CFG.robot_pose_read_enable:
        try:
            snap = AUBO_SESSION.read_pose_snapshot()
            if ROBOT_CFG.require_power_on and not bool(snap.get("power_on", False)):
                return None, snap, "aubo_power_off"
            if ROBOT_CFG.require_steady and not bool(snap.get("steady", False)):
                return None, snap, "aubo_not_steady"
            if ROBOT_CFG.reject_collision and bool(snap.get("collision", False)):
                return None, snap, "aubo_collision_flag"
            vals = snap.get("pose_values_sdk_m_rad", [])
            if not isinstance(vals, (list, tuple)) or len(vals) < 6:
                return None, snap, "aubo_pose_values_less_than_6"
            numeric_vals = [float(v) for v in vals[:6]]
            if not all(math.isfinite(v) for v in numeric_vals):
                return None, snap, "aubo_pose_values_non_finite"
            # AUBO 状态接口偶尔会在关节状态有效时瞬时返回全零 Tool/TCP 位姿。
            # 对当前机械臂而言全零位姿不可能是有效采集姿态，必须拒绝，不能作为稳定样本保存。
            if all(abs(v) <= 1e-12 for v in numeric_vals):
                return None, snap, "aubo_pose_values_all_zero"
            T = AUBO_SESSION.pose_sdk_to_transform_mm(numeric_vals)
            return T, snap, "aubo_pose_ok"
        except Exception as exc:
            print("[WARN] 自动读取 AUBO TCP 失败:", exc)

    if ROBOT_CFG.manual_pose_sdk_m_rad is not None:
        vals = [float(v) for v in ROBOT_CFG.manual_pose_sdk_m_rad]
        T = AUBO_SESSION.pose_sdk_to_transform_mm(vals)
        pose_values_mm_deg = [
            vals[0] * 1000.0,
            vals[1] * 1000.0,
            vals[2] * 1000.0,
            math.degrees(vals[3]),
            math.degrees(vals[4]),
            math.degrees(vals[5]),
        ]
        snap = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "controller": {"protocol": "manual_aubo_pose"},
            "robot_brand": "AUBO",
            "pose_source": "manual",
            "pose_units": {"xyz": "mm", "rpy": "deg"},
            "pose_values": [float(v) for v in pose_values_mm_deg],
            "pose_values_sdk_m_rad": vals,
        }
        return T, snap, "manual_aubo_pose_ok"

    return None, None, "no_aubo_pose"


def set_manual_pose_from_console() -> None:
    text = input("请输入当前 AUBO TCP 位姿 x y z rx ry rz，单位 m rad，用空格分隔：\n> ").strip()
    parts = text.replace(",", " ").split()
    if len(parts) != 6:
        print("[WARN] 输入数量不是 6，未更新 manual_pose_sdk_m_rad")
        return
    try:
        vals = tuple(float(v) for v in parts)
    except ValueError:
        print("[WARN] 输入包含非数字，未更新 manual_pose_sdk_m_rad")
        return
    ROBOT_CFG.manual_pose_sdk_m_rad = vals  # type: ignore[assignment]
    print("[INFO] manual_pose_sdk_m_rad 已更新:", vals)
