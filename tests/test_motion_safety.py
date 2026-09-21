import unittest

from aubo_workbench.motion_control import AuboMotionSession


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


if __name__ == "__main__":
    unittest.main()
