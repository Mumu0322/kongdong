from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from aubo_workbench.capture import pose_bracket_report
from aubo_workbench.geometry import transform_to_sdk_pose_m_rad
from aubo_workbench.robot import AuboPoseSession, get_capture_pose_transform


def session_with_offset(offset):
    session = AuboPoseSession()
    flange = [0.2, 0.3, 0.4, 0.1, -0.2, 0.3]
    tcp = transform_to_sdk_pose_m_rad(
        session.pose_sdk_to_transform_mm(flange) @ session.pose_sdk_to_transform_mm(offset)
    )
    session.connect = lambda: None
    session.state = SimpleNamespace(
        isPowerOn=lambda: True, isSteady=lambda: True, isCollisionOccurred=lambda: False,
        getToolPose=lambda: flange, getTcpPose=lambda: tcp,
        getActualTcpOffset=lambda: offset,
        getRobotModeType=lambda: "running", getSafetyModeType=lambda: "normal",
        getJointPositions=lambda: [0.0] * 6,
    )
    session.config = SimpleNamespace(getTcpOffset=lambda: offset)
    return session, tcp


def test_reads_active_tcp_and_offset_without_a_named_tool_selection():
    session, tcp = session_with_offset([0.01, -0.02, 0.21, 0.4, 0.2, -0.1])
    snapshot = session.read_pose_snapshot()
    assert snapshot["pose_source"] == "tcp"
    assert snapshot["pose_source_selection"] == "controller_active_tcp"
    assert snapshot["pose_chain_check"]["ok"]
    assert np.allclose(snapshot["pose_values_sdk_m_rad"], tcp)
    assert snapshot["pose_values_sdk_m_rad"] != snapshot["tool_pose_sdk_m_rad"]


def test_controller_tool_change_is_read_on_next_snapshot():
    first, _ = session_with_offset([0.0] * 6)
    second, tcp = session_with_offset([0.0, 0.0, 0.3, 0.0, 0.0, 0.0])
    old = first.read_pose_snapshot()
    first.state, first.config = second.state, second.config
    new = first.read_pose_snapshot()
    assert new["pose_values_sdk_m_rad"] == tcp
    assert old["actual_tcp_offset_sdk_m_rad"] != new["actual_tcp_offset_sdk_m_rad"]


def test_inconsistent_controller_pose_is_rejected():
    session, _ = session_with_offset([0.0] * 6)
    session.state.getActualTcpOffset = lambda: [0.0, 0.0, 0.02, 0.0, 0.0, 0.0]
    with pytest.raises(RuntimeError, match="读数不一致"):
        session.read_pose_snapshot()


def test_missing_actual_offset_does_not_fall_back_to_flange():
    session, _ = session_with_offset([0.0] * 6)
    session.state.getActualTcpOffset = lambda: []
    with patch("aubo_workbench.robot.AUBO_SESSION", session):
        transform, snapshot, status = get_capture_pose_transform()
    assert transform is None
    assert status == "aubo_pose_read_failed"


def test_tool_switch_during_frame_is_rejected_even_with_same_tcp_pose():
    before = {
        "pose_values": [100.0, 200.0, 300.0, 0.0, 0.0, 0.0],
        "pose_source_selection": "controller_active_tcp",
        "actual_tcp_offset_sdk_m_rad": [0.0] * 6,
    }
    after = dict(before, actual_tcp_offset_sdk_m_rad=[0.0, 0.0, 0.01, 0.0, 0.0, 0.0])
    report = pose_bracket_report(before, after)
    assert not report["ok"]
    assert not report["actual_tcp_offset_stable"]
