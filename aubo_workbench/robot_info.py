#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读查询 AUBO 机械臂/控制器信息，不下发任何运动指令。

这是原脚本里通过 `exec(compile(embedded_source, ...))` 硬塞进主文件的
独立小工具，现在是一个普通模块，可以直接 `python -m aubo_workbench.robot_info`
运行，也可以被 workbench.py 的 RobotInfoPanel 当库函数调用。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable

from .sdk_paths import add_aubo_sdk_to_path

add_aubo_sdk_to_path()

import pyaubo_sdk as aubo  # noqa: E402


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return str(value)


def safe_call(fn: Callable[[], Any]) -> Any:
    try:
        return to_jsonable(fn())
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def read_robot(client: Any, robot_name: str) -> dict[str, Any]:
    robot = client.getRobotInterface(robot_name)
    state = robot.getRobotState()
    config = robot.getRobotConfig()
    manage = robot.getRobotManage()
    io = robot.getIoControl()

    return {
        "name": robot_name,
        "config": {
            "robot_type": safe_call(config.getRobotType),
            "robot_sub_type": safe_call(config.getRobotSubType),
            "control_box_type": safe_call(config.getControlBoxType),
            "dof": safe_call(config.getDof),
            "tcp_offset": safe_call(config.getTcpOffset),
            "payload": safe_call(config.getPayload),
            "home_position": safe_call(config.getHomePosition),
            "mounting_pose": safe_call(config.getMountingPose),
            "gravity": safe_call(config.getGravity),
            "collision_level": safe_call(config.getCollisionLevel),
            "default_joint_speed": safe_call(config.getDefaultJointSpeed),
            "default_joint_acc": safe_call(config.getDefaultJointAcc),
            "default_tool_speed": safe_call(config.getDefaultToolSpeed),
            "default_tool_acc": safe_call(config.getDefaultToolAcc),
            "joint_min_positions": safe_call(config.getJointMinPositions),
            "joint_max_positions": safe_call(config.getJointMaxPositions),
            "joint_max_speeds": safe_call(config.getJointMaxSpeeds),
            "joint_max_accelerations": safe_call(config.getJointMaxAccelerations),
            "tcp_max_speeds": safe_call(config.getTcpMaxSpeeds),
            "tcp_max_accelerations": safe_call(config.getTcpMaxAccelerations),
            "has_tcp_force_sensor": safe_call(config.hasTcpForceSensor),
            "has_base_force_sensor": safe_call(config.hasBaseForceSensor),
        },
        "manage": {
            "operational_mode": safe_call(manage.getOperationalMode),
            "control_mode": safe_call(manage.getRobotControlMode),
            "freedrive_enabled": safe_call(manage.isFreedriveEnabled),
            "simulation_enabled": safe_call(manage.isSimulationEnabled),
            "link_mode_enabled": safe_call(manage.isLinkModeEnabled),
        },
        "state": {
            "power_on": safe_call(state.isPowerOn),
            "steady": safe_call(state.isSteady),
            "within_safety_limits": safe_call(state.isWithinSafetyLimits),
            "collision_occurred": safe_call(state.isCollisionOccurred),
            "robot_mode": safe_call(state.getRobotModeType),
            "safety_mode": safe_call(state.getSafetyModeType),
            "joint_positions_rad": safe_call(state.getJointPositions),
            "joint_speeds": safe_call(state.getJointSpeeds),
            "joint_temperatures": safe_call(state.getJointTemperatures),
            "joint_currents": safe_call(state.getJointCurrents),
            "tcp_pose": safe_call(state.getTcpPose),
            "tcp_speed": safe_call(state.getTcpSpeed),
            "tcp_force": safe_call(state.getTcpForce),
            "main_voltage": safe_call(state.getMainVoltage),
            "main_current": safe_call(state.getMainCurrent),
            "robot_voltage": safe_call(state.getRobotVoltage),
            "robot_current": safe_call(state.getRobotCurrent),
            "control_box_temperature": safe_call(state.getControlBoxTemperature),
            "control_box_humidity": safe_call(state.getControlBoxHumidity),
        },
        "io_summary": {
            "standard_di_num": safe_call(io.getStandardDigitalInputNum),
            "standard_do_num": safe_call(io.getStandardDigitalOutputNum),
            "tool_di_num": safe_call(io.getToolDigitalInputNum),
            "tool_do_num": safe_call(io.getToolDigitalOutputNum),
            "standard_ai_num": safe_call(io.getStandardAnalogInputNum),
            "standard_ao_num": safe_call(io.getStandardAnalogOutputNum),
            "tool_ai_num": safe_call(io.getToolAnalogInputNum),
            "tool_ao_num": safe_call(io.getToolAnalogOutputNum),
            "tool_voltage_domain": safe_call(io.getToolVoltageOutputDomain),
        },
    }


def read_all(ip: str, port: int, user: str, password: str, timeout_ms: int) -> dict[str, Any]:
    """连接一次、读完所有信息、断开。供 CLI 和 GUI 共用。"""
    client = aubo.RpcClient()
    client.setRequestTimeout(timeout_ms)
    ret = client.connect(ip, port)
    if ret != aubo.AUBO_OK:
        raise RuntimeError(f"连接失败：{ret} {aubo.returnValue2Str(ret)}")
    try:
        ret = client.login(user, password)
        if ret != aubo.AUBO_OK:
            raise RuntimeError(f"登录失败：{ret} {aubo.returnValue2Str(ret)}")

        system_info = client.getSystemInfo()
        robot_names = list(client.getRobotNames())
        return {
            "connection": {
                "ip": ip, "port": port, "robot_names": robot_names,
                "axis_names": safe_call(client.getAxisNames),
            },
            "system_info": {
                "control_software_version": safe_call(system_info.getControlSoftwareFullVersion),
                "control_software_build_date": safe_call(system_info.getControlSoftwareBuildDate),
                "control_software_version_code": safe_call(system_info.getControlSoftwareVersionCode),
                "interface_version_code": safe_call(system_info.getInterfaceVersionCode),
                "control_system_time": safe_call(system_info.getControlSystemTime),
            },
            "robots": [read_robot(client, name) for name in robot_names],
        }
    finally:
        try:
            client.disconnect()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Read AUBO robot information without motion commands.")
    parser.add_argument("--ip", default=os.getenv("AUBO_ROBOT_IP", "192.168.1.100"))
    parser.add_argument("--port", type=int, default=30004)
    parser.add_argument("--user", default="AUBO")
    parser.add_argument("--password", default=os.getenv("AUBO_ROBOT_PASSWORD", ""))
    parser.add_argument("--timeout-ms", type=int, default=3000)
    args = parser.parse_args()

    try:
        result = read_all(args.ip, args.port, args.user, args.password, args.timeout_ms)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
