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


class FakeRuntime:
    def __init__(self, calls, status):
        self.calls = calls
        self.status = status

    def getStatus(self):
        return self.status

    def start(self):
        self.calls.append(("runtime.start",))
        return 0

    def stop(self):
        self.calls.append(("runtime.stop",))
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

    def test_blended_path_queues_pose_changes_with_aubo_argument_order(self):
        session = make_session(FakeState())
        poses = [[float(index)] * 6 for index in range(3)]
        result = session.move_line_blended_path(poses, 0.03, 0.35, 0.005)
        self.assertEqual(result, [0, 0, 0, 0])
        self.assertEqual(session.motion.calls, [
            ("clearPath",),
            ("moveLine", poses[0], 0.35, 0.03, 0.005, 0.0),
            ("moveLine", poses[1], 0.35, 0.03, 0.005, 0.0),
            ("moveLine", poses[2], 0.35, 0.03, 0.0, 0.0),
        ])

    def test_blended_path_rejects_unsteady_start_before_queueing(self):
        session = make_session(FakeState(steady=False))
        with self.assertRaises(RuntimeError):
            session.move_line_blended_path([[0.0] * 6, [0.1] * 6], 0.03, 0.35, 0.005)
        self.assertEqual(session.motion.calls, [])

    def test_blended_path_retries_queue_full_without_clearing_path(self):
        session = make_session(FakeState())
        original = session.motion
        attempts = {1: 0}

        def move_line(pose, speed, acc, blend, radius):
            original.calls.append(("moveLine", pose, speed, acc, blend, radius))
            attempts[1] += 1
            return 2 if attempts[1] == 2 else 0

        original.moveLine = move_line
        result = session.move_line_blended_path(
            [[0.0] * 6, [0.1] * 6, [0.2] * 6],
            0.03, 0.35, 0.005, queue_retry_interval_s=0.001,
        )
        self.assertEqual(result, [0, 0, 0, 0])
        self.assertEqual(original.calls[0], ("clearPath",))
        self.assertEqual(sum(call[0] == "clearPath" for call in original.calls), 1)
        self.assertEqual(sum(call[0] == "moveLine" for call in original.calls), 4)

    def test_blended_path_accepts_per_segment_speeds(self):
        session = make_session(FakeState())
        session.move_line_blended_path(
            [[0.0] * 6, [0.1] * 6], 0.03, 0.35, 0.005,
            segment_speeds_m_s=[0.08, 0.03],
        )
        moves = [call for call in session.motion.calls if call[0] == "moveLine"]
        self.assertEqual([moves[0][2], moves[1][2]], [0.35, 0.35])
        self.assertEqual([moves[0][3], moves[1][3]], [0.08, 0.03])

    def test_blended_path_reports_persistent_queue_full_without_clearing_path(self):
        session = make_session(FakeState())
        original = session.motion
        original.moveLine = lambda *args: 2
        with self.assertRaisesRegex(RuntimeError, "队列持续满载"):
            session.move_line_blended_path(
                [[0.0] * 6, [0.1] * 6], 0.03, 0.35, 0.005,
                queue_retry_timeout_s=0.01, queue_retry_interval_s=0.001,
            )
        self.assertEqual(sum(call[0] == "clearPath" for call in original.calls), 1)

    def test_blended_path_resends_ignored_segment_and_logs_it(self):
        session = make_session(FakeState())
        original = session.motion
        responses = iter([0, 13, 13, 0, 0])

        def move_line(pose, speed, acc, blend, radius):
            original.calls.append(("moveLine", pose, speed, acc, blend, radius))
            return next(responses)

        original.moveLine = move_line
        log = []
        poses = [[0.0] * 6, [0.1] * 6, [0.2] * 6]
        result = session.move_line_blended_path(
            poses, 0.03, 0.35, 0.005, queue_retry_interval_s=0.001, queue_log=log,
        )
        self.assertEqual(result, [0, 0, 0, 0])
        moves = [call for call in original.calls if call[0] == "moveLine"]
        self.assertEqual([call[1] for call in moves], [poses[0], poses[1], poses[1], poses[1], poses[2]])
        self.assertEqual(sum(call[0] == "clearPath" for call in original.calls), 1)
        self.assertEqual([item["ignored_retries"] for item in log], [0, 2, 0])
        self.assertEqual(log[1]["attempts"], 3)

    def test_blended_path_accepts_ignore_when_tcp_already_on_target(self):
        # Field run 20260928_101738: segment 1 returned 2, ran anyway, then
        # every resend returned 13 because the arm was already on the target.
        poses = [[0.35, -0.066, 0.237, 0.0, -0.23, 0.0], [0.4, 0.0, 0.2, 0.0, -0.23, 0.0]]
        state = FakeState()
        state.getTcpPose = lambda: [0.35, -0.066, 0.2372, 0.0, -0.2301, 0.0]
        session = make_session(state)
        original = session.motion
        responses = iter([2, 13, 0])

        def move_line(pose, speed, acc, blend, radius):
            original.calls.append(("moveLine", pose, speed, acc, blend, radius))
            return next(responses)

        original.moveLine = move_line
        log = []
        result = session.move_line_blended_path(
            poses, 0.03, 0.35, 0.005, queue_retry_interval_s=0.001, queue_log=log,
        )
        self.assertEqual(result, [0, 13, 0])
        self.assertEqual([item["result"] for item in log], ["ignored_already_reached", "0"])

    def test_blended_path_gives_up_on_persistent_ignore(self):
        session = make_session(FakeState())
        session.motion.moveLine = lambda *args: 13
        log = []
        with self.assertRaisesRegex(RuntimeError, "持续返回13"):
            session.move_line_blended_path(
                [[0.0] * 6, [0.1] * 6], 0.03, 0.35, 0.005,
                ignore_retry_timeout_s=0.01, queue_retry_interval_s=0.001, queue_log=log,
            )
        self.assertEqual(log[0]["result"], "ignored_timeout")

    def test_blended_path_stops_resending_ignored_segment_on_collision(self):
        state = FakeState()
        session = make_session(state)

        def move_line(*args):
            state.collision = True
            return 13

        session.motion.moveLine = move_line
        with self.assertRaisesRegex(RuntimeError, "碰撞"):
            session.move_line_blended_path(
                [[0.0] * 6, [0.1] * 6], 0.03, 0.35, 0.005, queue_retry_interval_s=0.001,
            )

    def test_blended_path_abort_prevents_requeue_after_external_stop(self):
        session = make_session(FakeState())
        abort = threading.Event()
        calls = []

        def move_line(*args):
            calls.append(args)
            abort.set()  # external stopMove happens while this segment is retried
            return 13

        session.motion.moveLine = move_line
        with self.assertRaisesRegex(RuntimeError, "停止请求"):
            session.move_line_blended_path(
                [[0.0] * 6, [0.1] * 6], 0.03, 0.35, 0.005,
                queue_retry_interval_s=0.001, abort_event=abort,
            )
        self.assertEqual(len(calls), 1)

    def _runtime_session(self, *, status="RuntimeState.Stopped", queue_sizes=(3, 1, 0, 0)):
        state = FakeState()
        final = [0.2] * 6
        state.getTcpPose = lambda: final
        session = make_session(state)
        runtime = FakeRuntime(session.motion.calls, status)
        session.runtime = runtime
        sizes = iter(queue_sizes)
        session.motion.getQueueSize = lambda: next(sizes, 0)
        session.motion.getSpeedFraction = lambda: 0.5
        session.motion.stopMove = lambda *args: session.motion.calls.append(("stopMove",)) or 0
        return session, runtime

    def test_blended_path_runs_inside_runtime_machine_until_finished(self):
        session, runtime = self._runtime_session()
        info = {}
        result = session.move_line_blended_path(
            [[0.0] * 6, [0.1] * 6, [0.2] * 6], 0.03, 0.35, 0.005,
            use_runtime_machine=True, completion_settle_s=0.0, path_info=info,
        )
        self.assertEqual(result, [0, 0, 0, 0])
        names = [call[0] for call in session.motion.calls]
        self.assertEqual(names, ["clearPath", "runtime.start", "moveLine", "moveLine", "moveLine", "runtime.stop"])
        self.assertEqual(info["speed_fraction"], 0.5)
        self.assertIn("execution_wait_s", info)

    def test_blended_path_refuses_runtime_machine_that_is_already_running(self):
        session, runtime = self._runtime_session(status="RuntimeState.Running")
        with self.assertRaisesRegex(RuntimeError, "运行机状态"):
            session.move_line_blended_path(
                [[0.0] * 6, [0.2] * 6], 0.03, 0.35, 0.005, use_runtime_machine=True,
            )
        self.assertEqual(session.motion.calls, [])

    def test_blended_path_stops_motion_then_runtime_on_abort_during_execution(self):
        session, runtime = self._runtime_session(queue_sizes=(5,) * 1000)
        abort = threading.Event()
        session.state.steady = True
        timer = threading.Timer(0.05, abort.set)
        timer.start()
        with self.assertRaisesRegex(RuntimeError, "停止请求"):
            session.move_line_blended_path(
                [[0.0] * 6, [0.2] * 6], 0.03, 0.35, 0.005,
                use_runtime_machine=True, abort_event=abort,
            )
        timer.join()
        names = [call[0] for call in session.motion.calls]
        self.assertEqual(names[-2:], ["stopMove", "runtime.stop"])

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
