from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

import run_yolo_eye_in_hand_optimized as runner
from aubo_workbench.camera_stream_health import (
    CameraStreamError, camera_stream_session, monitored_camera_read,
    require_camera_stream_healthy,
)
from aubo_workbench.localization_errors import MotionExecutionError


def test_missing_frame_budget_spans_calls_and_is_latched():
    reader = Mock(return_value=None)
    read = monitored_camera_read(reader)

    @camera_stream_session
    def run():
        assert read() is None
        assert read() is None
        with pytest.raises(CameraStreamError, match="连续3次"):
            read()
        with pytest.raises(CameraStreamError):
            read()
        with pytest.raises(CameraStreamError):
            require_camera_stream_healthy()

    run()
    assert reader.call_count == 3
    # Ending the failed session must not poison a later independent run.
    camera_stream_session(lambda: read())()
    assert reader.call_count == 4


def test_valid_frame_resets_transient_missing_budget():
    frame = SimpleNamespace(intrinsics=object())
    reader = Mock(side_effect=[None, None, frame, None, None, frame])
    read = monitored_camera_read(reader)

    @camera_stream_session
    def run():
        for _ in range(6):
            read()
        require_camera_stream_healthy()

    run()
    assert reader.call_count == 6


def test_reader_outside_localization_preserves_none_behavior():
    reader = monitored_camera_read(lambda: None)
    assert [reader() for _ in range(20)] == [None] * 20


def test_sdk_exception_becomes_fatal_camera_error():
    reader = monitored_camera_read(Mock(side_effect=RuntimeError("device disconnected")))
    with pytest.raises(CameraStreamError, match="device disconnected") as caught:
        camera_stream_session(reader)()
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_missing_intrinsics_does_not_count_as_recovered_frame():
    reader = monitored_camera_read(lambda: SimpleNamespace(intrinsics=None))
    with pytest.raises(CameraStreamError):
        camera_stream_session(lambda: [reader() for _ in range(10)])()


def test_motion_is_not_sent_after_latched_camera_failure():
    motion = Mock()
    reader = monitored_camera_read(lambda: None)

    @camera_stream_session
    def run():
        with pytest.raises(CameraStreamError):
            for _ in range(3):
                reader()
        with pytest.raises(CameraStreamError):
            runner._confirm_and_move_line(
                "next group", np.eye(4), np.eye(4), SimpleNamespace(), motion, Mock(),
            )

    run()
    motion.move_line.assert_not_called()


@pytest.mark.parametrize("failure_phase", ["warmup", "capture", "pose_refinement", "motion"])
def test_stream_loss_aborts_grouped_workflow_before_next_group(tmp_path, failure_phase):
    selected = [
        {
            "hole_id": hole_id,
            "initial_center_px": [640.0, 400.0],
            "initial_center_base_mm": [float(hole_id * 20), 0.0, 340.0],
            "initial_plane_normal_base": [0.0, 0.0, -1.0],
            "initial_detection": {"class_id": 0, "center": [640.0, 400.0]},
        }
        for hole_id in range(1, 5)
    ]
    groups = [selected[:2], selected[2:]]
    args = SimpleNamespace(
        execute=True, reuse_coarse_cache=False, reuse_persistent_coarse_cache=False,
        shared_cache_validation=False, confidence=0.5, optimize_hole_order=False,
    )
    cfg = runner.TwoStageConfig(
        batch_coarse_localization=True, batch_fine_localization=False,
        batch_coarse_frames=15, batch_coarse_min_valid=10,
        batch_coarse_settle_discard_frames=0 if failure_phase == "capture" else 10,
        coarse_settle_delay_s=0.0,
    )
    intrinsics = SimpleNamespace(
        width=1280, height=800, fx=600.0, fy=600.0, cx=640.0, cy=400.0, distortion=(),
    )
    plan = {
        "target_tcp_pose_m_rad": [0.0] * 6,
        "projected_holes_px": {i: np.array([640.0, 400.0]) for i in range(1, 5)},
        "group_bbox_px": [600.0, 350.0, 680.0, 450.0],
        "group_center_px": np.array([640.0, 400.0]),
    }
    report = {"stages": {}, "camera": {}}
    runtime = {"rgbd_pipeline": object(), "align": object(), "chain": object()}
    handeye = SimpleNamespace(T_tcp_rgb_camera=np.eye(4), validated_for_motion=True)
    reader = Mock(return_value=None)
    checked_read = monitored_camera_read(reader)

    def refine(*values):
        if failure_phase == "pose_refinement":
            for _ in range(5):
                checked_read()
        return {
            "group": values[2], "current_tcp": values[3],
            "target": values[4], "geometry": values[5], "report": {},
        }

    def move_to_group(*_args, **_kwargs):
        if failure_phase == "motion":
            raise MotionExecutionError("moveLine 下发失败：[0, -1]")
        return np.eye(4)

    with patch.object(runner, "get_aligned_frame_bundle", checked_read), \
            patch.object(runner, "_split_batch_localization_groups", return_value=(groups, [None, None])), \
            patch.object(runner, "_plan_batch_coarse_group_pose", return_value=(np.eye(4), plan)), \
            patch.object(runner, "_move_to_shared_coarse_pose", side_effect=move_to_group) as move, \
            patch.object(runner, "_refine_shared_coarse_group_pose", side_effect=refine), \
            patch.object(runner, "_save_grouping_plan_visualization", return_value={}), \
            patch.object(runner, "_batch_fine_localization_at_260mm") as fine:
        with pytest.raises(MotionExecutionError if failure_phase == "motion" else CameraStreamError):
            camera_stream_session(runner._run_sequential_hole_workflow)(
                args, handeye, object(), cfg, tmp_path, report,
                runner.TimingRecorder(), [], runtime, object(), object(),
                np.eye(4), selected, intrinsics,
            )
    assert reader.call_count == (0 if failure_phase == "motion" else 3)
    assert move.call_count == 1
    fine.assert_not_called()
    expected_failure = "robot_motion_failed" if failure_phase == "motion" else "camera_stream_unavailable"
    assert report["stages"]["batch_coarse_plan"]["groups"][-1]["failure_type"] == expected_failure


def test_power_off_stops_without_waiting_or_sending_motion():
    motion = Mock()
    motion.snapshot.return_value = {"power_on": False, "steady": True, "collision": False}
    args = SimpleNamespace(speed_m_s=0.08, acc_m_s2=0.25)
    with patch.object(runner.time, "sleep") as sleep:
        with pytest.raises(MotionExecutionError, match="未上电"):
            runner._confirm_and_move_line(
                "next group", np.eye(4), np.eye(4), args, motion, Mock(),
                require_confirmation=False,
            )
    sleep.assert_not_called()
    motion.move_line.assert_not_called()


def test_fine_stream_failure_does_not_trigger_hole_recapture(tmp_path):
    with patch.object(runner, "_capture_fine_burst", side_effect=CameraStreamError("stream lost")) as capture:
        with pytest.raises(CameraStreamError):
            runner._capture_fine_with_recovery(
                object(), object(), 0.5, {}, runner.TwoStageConfig(), tmp_path,
                {}, 1, 1, np.array([640.0, 400.0]), runner.TimingRecorder(), [],
            )
    capture.assert_called_once()
