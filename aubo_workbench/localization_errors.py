"""Hardware faults must terminate localization rather than skip a hole/group."""

from functools import wraps


class LocalizationHardwareError(RuntimeError):
    failure_type = "hardware_failure"


class MotionExecutionError(LocalizationHardwareError):
    failure_type = "robot_motion_failed"


def motion_failure_boundary(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except LocalizationHardwareError:
            raise
        except Exception as exc:
            raise MotionExecutionError(f"{function.__name__}: {exc}") from exc
    return wrapped
