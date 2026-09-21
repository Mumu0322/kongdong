"""Bounded missing-frame handling for a localization session (no SDK dependency)."""

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from .localization_errors import LocalizationHardwareError


class CameraStreamError(LocalizationHardwareError):
    """The camera cannot supply frames; do not retry this as a hole failure."""
    failure_type = "camera_stream_unavailable"


@dataclass
class _StreamHealth:
    missing_reads: int = 0
    error: CameraStreamError | None = None


_health: ContextVar[_StreamHealth | None] = ContextVar("localization_camera_health", default=None)
MAX_CONSECUTIVE_MISSING_READS = 3


def camera_stream_session(function):
    """Keep the failure budget across warmup/capture/groups, reset on a new run."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _health.set(_StreamHealth())
        try:
            return function(*args, **kwargs)
        finally:
            _health.reset(token)
    return wrapped


def require_camera_stream_healthy():
    state = _health.get()
    if state is not None and state.error is not None:
        raise state.error


def monitored_camera_read(function):
    """Preserve legacy readers outside localization; abort sustained loss inside it."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        state = _health.get()
        if state is None:
            return function(*args, **kwargs)
        require_camera_stream_healthy()
        try:
            bundle = function(*args, **kwargs)
        except CameraStreamError:
            raise
        except Exception as exc:
            state.error = CameraStreamError(
                f"相机取帧异常：{type(exc).__name__}: {exc}；停止本轮定位，不再进入后续分组。"
            )
            raise state.error from exc
        if bundle is not None and bundle.intrinsics is not None:
            state.missing_reads = 0
            return bundle
        state.missing_reads += 1
        print(
            f"[CAMERA_FRAME_MISSING] 连续未取得有效相机帧 "
            f"{state.missing_reads}/{MAX_CONSECUTIVE_MISSING_READS}（含预热取帧）",
            flush=True,
        )
        if state.missing_reads >= MAX_CONSECUTIVE_MISSING_READS:
            state.error = CameraStreamError(
                "相机连续3次未返回有效图像/内参，已停止本轮定位和后续分组运动；"
                "请检查相机连接及设备日志，恢复视频流后重新启动任务。"
            )
            raise state.error
        return bundle
    return wrapped
