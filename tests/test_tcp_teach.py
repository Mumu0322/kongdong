from __future__ import annotations

import unittest

from aubo_workbench.tcp_teach import TeachPoint, TcpTeachSession, fmt_xyz_mm


class _FakeClient:
    def hasConnected(self) -> bool:
        return True


class _FakeMath:
    def __init__(self) -> None:
        self.call_count = 0

    def tcpOffsetIdentify(self, poses: list[list[float]]) -> tuple[list[float], int]:
        del poses
        self.call_count += 1
        delta = self.call_count * 0.000001
        return [0.001 + delta, 0.002 + delta, 0.210 + delta], 0

    def poseTrans(self, tool_pose: list[float], offset: list[float]) -> list[float]:
        return [tool_pose[i] + offset[i] for i in range(3)] + tool_pose[3:6]


def _point(index: int) -> TeachPoint:
    return TeachPoint(
        name=f"P{index + 1}",
        timestamp="2026-07-17 00:00:00",
        tool_pose=[index * 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        tcp_pose=[0.0] * 6,
        joints=[0.0] * 6,
        actual_tcp_offset=[0.0] * 6,
    )


class TcpTeachSessionTests(unittest.TestCase):
    def test_identify_tcp_accepts_three_value_sdk_result(self) -> None:
        session = TcpTeachSession()
        session.client = _FakeClient()
        session.math_api = _FakeMath()
        session.tcp_offset_cache = [0.0, 0.0, 0.0, 0.1, 0.2, 0.3]

        result = session.identify_tcp([_point(index) for index in range(5)])

        self.assertEqual(result["valid_combinations"], 5)
        self.assertEqual(len(result["identified_xyz"]), 3)
        self.assertEqual(len(result["offset"]), 6)
        self.assertEqual(result["offset"][3:], [0.1, 0.2, 0.3])

    def test_fmt_xyz_mm_converts_meters_to_millimeters(self) -> None:
        self.assertEqual(fmt_xyz_mm([0.440783, 0.312765, 0.266545]), "[440.783, 312.765, 266.545]")


if __name__ == "__main__":
    unittest.main()
