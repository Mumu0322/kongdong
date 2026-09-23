import unittest
import json
import threading
import tempfile
from pathlib import Path
from unittest.mock import patch

from aubo_workbench.motion_control import (
    AuboMotionPanel, AuboMotionSession, SavedPoint, load_home_point, load_points,
)


class FakeState:
    def __init__(self, *, power_on=True, steady=True, collision=False, within_limits=True):
        self.power_on = power_on
        self.steady = steady
        self.collision = collision
        self.within_limits = within_limits

    def isPowerOn(self):
        return self.power_on

    def isSteady(self):
        return self.steady

    def isCollisionOccurred(self):
        return self.collision

    def isWithinSafetyLimits(self):
        return self.within_limits


class FakeMotion:
    def __init__(self):
        self.calls = []

    def clearPath(self):
        self.calls.append(("clearPath",))
        return 0

    def moveJoint(self, joints, speed, acc, blend, radius):
        self.calls.append(("moveJoint", joints, speed, acc, blend, radius))
        return 0

    def moveLine(self, pose, speed, acc, blend, radius):
        self.calls.append(("moveLine", pose, speed, acc, blend, radius))
        return 0

    def speedJoint(self, speeds, acc, duration):
        self.calls.append(("speedJoint", speeds, acc, duration))
        return 0

    def speedLine(self, speeds, acc, duration):
        self.calls.append(("speedLine", speeds, acc, duration))
        return 0


def make_session(state):
    session = AuboMotionSession()
    session.client = object()
    session.robot_if = object()
    session.state = state
    session.motion = FakeMotion()
    session.manage = object()
    return session


class MotionSafetyTests(unittest.TestCase):
    def test_incomplete_saved_home_cannot_become_zero_joint_target(self):
        with self.assertRaisesRegex(ValueError, "joints_rad"):
            SavedPoint.from_dict({})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "home.json"
            path.write_text(json.dumps({"home_point": {}}), encoding="utf-8")
            with patch("aubo_workbench.motion_control.HOME_POINT_FILE", path):
                self.assertIsNone(load_home_point())

    def test_legacy_point_list_still_loads_complete_points(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points.json"
            path.write_text(json.dumps([{
                "name": "P1", "joints": [0.1] * 6, "tcp": [0.2] * 6,
            }]), encoding="utf-8")
            with patch("aubo_workbench.motion_control.POINTS_FILE", path):
                points = load_points()
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].joints_rad, [0.1] * 6)

    def test_position_move_requires_power_and_steady_state(self):
        for state in (FakeState(power_on=False), FakeState(steady=False), FakeState(collision=True)):
            session = make_session(state)
            with self.assertRaises(RuntimeError):
                session.move_joint([0.0] * 6, 0.1, 0.2)
            self.assertEqual(session.motion.calls, [])

    def test_position_move_rejects_controller_safety_limit_failure(self):
        session = make_session(FakeState(within_limits=False))
        with self.assertRaises(RuntimeError):
            session.move_line([0.0] * 6, 0.01, 0.02)
        self.assertEqual(session.motion.calls, [])

    def test_position_move_validates_shape_and_finite_values_before_sdk_call(self):
        session = make_session(FakeState())
        with self.assertRaises(ValueError):
            session.move_joint([0.0] * 5, 0.1, 0.2)
        with self.assertRaises(ValueError):
            session.move_line([0.0, float("nan"), 0.0, 0.0, 0.0, 0.0], 0.01, 0.02)
        self.assertEqual(session.motion.calls, [])

    def test_speed_command_does_not_require_steady_but_rejects_collision(self):
        session = make_session(FakeState(steady=False))
        session.speed_joint([0.0] * 6, 0.2, 0.1)
        self.assertEqual(session.motion.calls[0][0], "speedJoint")

        blocked = make_session(FakeState(steady=False, collision=True))
        with self.assertRaises(RuntimeError):
            blocked.speed_line([0.0] * 6, 0.2, 0.1)
        self.assertEqual(blocked.motion.calls, [])

    def test_failed_clear_path_never_sends_position_move(self):
        for move_name in ("move_joint", "move_line"):
            for failure in (42, OSError("controller unavailable")):
                with self.subTest(move_name=move_name, failure=failure):
                    session = make_session(FakeState())

                    def fail_clear():
                        session.motion.calls.append(("clearPath",))
                        if isinstance(failure, Exception):
                            raise failure
                        return failure

                    session.motion.clearPath = fail_clear
                    with self.assertRaisesRegex(RuntimeError, "clearPath"):
                        getattr(session, move_name)([0.0] * 6, 0.1, 0.2)
                    self.assertEqual(session.motion.calls, [("clearPath",)])

    def test_jog_stops_after_sdk_failure(self):
        class FakePanel:
            def __init__(self):
                self.jog_lock = threading.Lock()
                self.jog_event = None
                self.logs = []
                self.stop_calls = 0
                self.session = self

            def stop_jog(self, log_stop=False):
                pass

            def log(self, message):
                self.logs.append(message)

            def after(self, delay, callback):
                callback()

            def wait_after_step(self):
                pass

            def stop_motion(self):
                self.stop_calls += 1

        class InlineThread:
            def __init__(self, target, daemon):
                self.target = target

            def start(self):
                self.target()

        panel = FakePanel()
        step_calls = []

        def rejected_step():
            step_calls.append(1)
            return [0, 13]

        with patch("aubo_workbench.motion_control.threading.Thread", InlineThread):
            AuboMotionPanel.start_step_jog(panel, "TCP", rejected_step)

        self.assertEqual(len(step_calls), 1)
        self.assertEqual(panel.stop_calls, 1)
        self.assertIsNone(panel.jog_event)
        self.assertTrue(any("分步点动失败" in message for message in panel.logs))


if __name__ == "__main__":
    unittest.main()
