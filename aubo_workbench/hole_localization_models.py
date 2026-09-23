"""Data models and timing support for two-stage hole localization."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Any

import numpy as np


def infer_timing_category(name: str) -> str:
    """为旧调用点提供稳定的计时分类兜底。

    新调用点可以通过 ``category=`` 显式指定；这里保留名称推断是为了兼容
    已经拆分到多个工作流模块的大量旧计时点。分类只用于报告，不参与定位。
    """
    normalized = str(name).strip().lower()
    if normalized.startswith("artifact_io/"):
        return "artifact_io"
    if "next_cycle" in normalized or "operator" in normalized:
        return "operator_wait"
    if (
        "wait_" in normalized
        or "settle" in normalized
        or "discard" in normalized
        or "verify_steady" in normalized
    ):
        return "safety_wait"
    if any(
        token in normalized
        for token in (
            "navigate",
            "move_to",
            "safe_correction_move",
            "correction_motion",
            "final_motion",
            "return_home",
            "move_home",
        )
    ):
        return "robot_motion"
    if any(
        token in normalized
        for token in (
            "capture",
            "geometry",
            "fusion",
            "pixel_to_base",
            "tilt_center",
            "rgbd_pipeline",
            "pointcloud",
        )
    ):
        return "vision_compute"
    return "other"


# 视觉耗时只描述相机数据和视觉算法本身。机器人运动、停稳/安全等待、
# 人工确认以及文件产物都不属于这些阶段。字符串作为报告 JSON 的稳定接口，
# 不要在 UI 中直接依赖事件名称做二次猜测。
VISUAL_TIMING_STAGES = (
    "camera_startup",
    "frame_acquisition",
    "yolo_inference",
    "pointcloud_processing",
    "ellipse_fitting",
    "coordinate_transform",
    "fusion_quality",
    "final_point_calculation",
    "visual_pipeline_other",
)


def infer_visual_timing_stage(name: str) -> str | None:
    """根据旧计时点名称推断视觉子阶段。

    新计时点应显式传入 ``visual_stage``；这个推断只负责兼容历史报告和
    尚未拆分的旧调用点。返回 ``None`` 表示该事件不计入视觉耗时。
    """
    normalized = str(name).strip().lower().replace("\\", "/")
    if not normalized:
        return None
    if normalized.startswith("artifact_io/"):
        return None
    if normalized.startswith("operator/") or normalized.startswith("robot/"):
        return None
    if any(
        token in normalized
        for token in (
            "wait_steady",
            "verify_steady",
            "settle_delay",
            "settle_buffer",
            "correction_motion",
            "safe_correction_move",
            "navigate_to_",
            "move_to_",
            "move_home",
            "return_home",
            "final_motion",
        )
    ):
        return None
    if normalized.startswith("camera/") and any(
        token in normalized for token in ("start", "restart", "pipeline")
    ):
        return "camera_startup"
    if normalized.endswith("initial_selection/yolo_rgbd_pointcloud"):
        return "visual_pipeline_other"
    if any(
        token in normalized
        for token in (
            "get_rgb_frame",
            "get_aligned_frame",
            "frame_acquisition",
            "discard_settle_frames",
            "flush_rgb_queue",
            "rgb_queue",
        )
    ):
        return "frame_acquisition"
    if any(
        token in normalized
        for token in (
            "yolo",
            "detect",
            "assign_detection",
            "nearest_detection",
        )
    ):
        return "yolo_inference"
    # 未拆分的采集父事件作为视觉流水线兜底；子事件会在时间轴上优先显示。
    if any(
        token in normalized
        for token in (
            "capture_and_localize",
            "capture_all_holes",
            "fine_capture_attempt",
            "/capture",
        )
    ):
        return "visual_pipeline_other"
    if any(
        token in normalized
        for token in (
            "ellipse",
            "fit_hole",
            "fit_circle",
        )
    ):
        return "ellipse_fitting"
    if any(
        token in normalized
        for token in (
            "pointcloud",
            "point_cloud",
            "hole_camera_point",
            "plane",
            "sphere",
            "surface",
        )
    ):
        return "pointcloud_processing"
    if any(
        token in normalized
        for token in (
            "pixel_to_base",
            "camera_transform",
            "project_",
            "projection",
            "undistort",
            "coordinate_transform",
        )
    ):
        return "coordinate_transform"
    if any(
        token in normalized
        for token in (
            "fusion",
            "fuse_",
            "joint_",
            "quality",
            "anchor_correction",
        )
    ):
        return "fusion_quality"
    if any(
        token in normalized
        for token in (
            "tilt_center",
            "final_point",
            "point_calculation",
        )
    ):
        return "final_point_calculation"
    if any(
        token in normalized
        for token in (
            "plan_group_pose",
            "plan_batch",
            "projected_holes",
            "camera_height_to_plane",
        )
    ):
        return "visual_pipeline_other"
    if "geometry" in normalized:
        return "visual_pipeline_other"
    return None


@contextmanager
def visual_measure(
    timing: Any | None,
    name: str,
    visual_stage: str,
    **details: Any,
):
    """在计时对象可用时记录一个视觉叶子阶段，否则保持无开销兼容。"""
    measure = getattr(timing, "measure", None)
    if callable(measure):
        details.setdefault("category", "vision_compute")
        details.setdefault("visual_stage", str(visual_stage))
        details.setdefault("level", "leaf")
        with measure(name, **details):
            yield
    else:
        yield


class TwoStageSelectionCancelled(RuntimeError):
    """用户在回原点后的新一轮初始选孔窗口中按 Esc 退出。"""


class TimingRecorder:
    """记录流程阶段耗时，并在每个阶段结束时同步到运行报告。

    ``measure_artifact`` 用于记录图片、点云归档和报告等文件产物的写盘
    时间。产物写盘属于审计输出，不应算入机械臂/视觉算法的有效节拍；
    它仍会保留在 ``timing.events`` 中，便于定位磁盘或编码性能问题。
    ``effective_wall_elapsed_s`` 是从总墙钟时间扣除已记录产物写盘后的值。
    生产节拍和纯执行时间由报告模块基于事件时间区间计算；不能把所有
    ``events`` 的 elapsed 直接相加，因为共享定位存在父子嵌套计时。
    """

    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self.events: list[dict[str, Any]] = []
        # 运动距离与耗时分开保存。运动事件不直接塞进 ``events``，避免
        # 为了统计距离而改变原有计时事件的嵌套时间轴；最终报告会从这里
        # 汇总 TCP 端点位移，并明确标注其计算口径。
        self.motion_events: list[dict[str, Any]] = []
        self._report: dict[str, Any] | None = None
        self._active_measurements: list[dict[str, Any]] = []
        self._markers: list[dict[str, Any]] = []

    def attach_report(self, report: dict[str, Any]) -> None:
        self._report = report
        self._sync()

    def snapshot(self, *, include_visual: bool = True) -> dict[str, Any]:
        aggregates: dict[str, dict[str, Any]] = {}
        elapsed_since_start = max(0.0, time.perf_counter() - self.started_at)
        artifact_events = [
            event for event in self.events
            if str(event.get("category", "")) == "artifact_io"
            or str(event.get("name", "")).startswith("artifact_io/")
            or bool(event.get("exclude_from_effective", False))
        ]
        for event in self.events:
            name = str(event["name"])
            bucket = aggregates.setdefault(
                name,
                {"count": 0, "total_elapsed_s": 0.0, "max_elapsed_s": 0.0},
            )
            elapsed_s = float(event["elapsed_s"])
            bucket["count"] += 1
            bucket["total_elapsed_s"] += elapsed_s
            bucket["max_elapsed_s"] = max(float(bucket["max_elapsed_s"]), elapsed_s)
        for bucket in aggregates.values():
            bucket["total_elapsed_s"] = round(float(bucket["total_elapsed_s"]), 6)
            bucket["max_elapsed_s"] = round(float(bucket["max_elapsed_s"]), 6)
        artifact_elapsed_s = sum(float(event["elapsed_s"]) for event in artifact_events)
        snapshot = {
            "schema_version": 2,
            "total_elapsed_s": round(elapsed_since_start, 6),
            "effective_wall_elapsed_s": round(
                max(0.0, elapsed_since_start - artifact_elapsed_s), 6,
            ),
            "events": list(self.events),
            "motion_events": list(self.motion_events),
            "markers": list(self._markers),
            "aggregates": aggregates,
            "artifact_io": {
                "total_elapsed_s": round(artifact_elapsed_s, 6),
                "event_count": len(artifact_events),
                "failed_event_count": sum(
                    1 for event in artifact_events
                    if str(event.get("status", "")) != "completed"
                ),
            },
        }
        # 视觉阶段汇总放进原始 report.json，便于不打开简明摘要的调用方
        # 直接读取；使用延迟导入避免 models/report 的循环导入。
        if include_visual:
            try:
                from aubo_workbench.hole_localization_report import (
                    summarize_visual_timing_events,
                )
                snapshot["visual_timing"] = summarize_visual_timing_events(
                    list(self.events)
                )
            except Exception:
                # 计时不能反过来阻断定位主流程；旧环境缺少报告模块时仍返回
                # 完整的原始事件和 artifact_io 统计。
                pass
        try:
            from aubo_workbench.hole_localization_report import (
                summarize_motion_distance,
            )
            snapshot["motion_distance"] = summarize_motion_distance(
                list(self.motion_events),
            )
        except Exception:
            # 同样不让报告增强项影响计时主链路。
            pass
        return snapshot

    def record_motion(
        self,
        name: str,
        start_xyz_mm: Any,
        target_xyz_mm: Any,
        *,
        motion_type: str = "move_line",
        status: str = "completed",
        **details: Any,
    ) -> None:
        """记录一次成功的机器人 TCP 端点运动，用于距离统计。

        ``path_distance_mm`` 是当前已知 TCP 起点到目标点的欧氏距离，
        不是控制器内部真实关节轨迹长度；这对 moveLine 是规划段长度，
        对 moveJoint 是端点位移下界。Z 正方向按基坐标定义为“上升”。
        统计失败或被取消的下发不会调用此方法。
        """
        try:
            start = np.asarray(start_xyz_mm, dtype=np.float64).reshape(-1)
            target = np.asarray(target_xyz_mm, dtype=np.float64).reshape(-1)
            if start.size >= 16:
                start = start.reshape(4, 4)[:3, 3]
            elif start.size >= 3:
                start = start[:3]
            else:
                raise ValueError("start_xyz_mm must have at least 3 values")
            if target.size >= 16:
                target = target.reshape(4, 4)[:3, 3]
            elif target.size >= 3:
                target = target[:3]
            else:
                raise ValueError("target_xyz_mm must have at least 3 values")
            if not np.isfinite(start).all() or not np.isfinite(target).all():
                raise ValueError("motion XYZ contains non-finite values")
        except (TypeError, ValueError, RuntimeError):
            # 距离统计不能阻断真实运动；无效位姿只被忽略并留在原有日志中。
            return

        delta = target - start
        dz = float(delta[2])
        horizontal_mm = float(np.linalg.norm(delta[:2]))
        path_distance_mm = float(np.linalg.norm(delta))
        if dz > 1e-9:
            vertical_direction = "up"
        elif dz < -1e-9:
            vertical_direction = "down"
        else:
            vertical_direction = "level"
        event: dict[str, Any] = {
            "name": str(name),
            "status": str(status),
            "motion_type": str(motion_type),
            "coordinate_frame": "robot_base_mm",
            "distance_basis": "planned_tcp_waypoint_delta",
            "start_xyz_mm": [round(float(value), 6) for value in start],
            "target_xyz_mm": [round(float(value), 6) for value in target],
            "delta_xyz_mm": [round(float(value), 6) for value in delta],
            "path_distance_mm": round(path_distance_mm, 6),
            "horizontal_distance_mm": round(horizontal_mm, 6),
            "vertical_delta_mm": round(dz, 6),
            "vertical_up_mm": round(max(0.0, dz), 6),
            "vertical_down_mm": round(max(0.0, -dz), 6),
            "vertical_direction": vertical_direction,
            "offset_s": round(max(0.0, time.perf_counter() - self.started_at), 6),
        }
        if self._active_measurements:
            event["parent_name"] = str(self._active_measurements[-1]["name"])
            event["depth"] = len(self._active_measurements)
        event.update(details)
        self.motion_events.append(event)
        self._sync()

    def mark(self, name: str, **details: Any) -> None:
        """记录一个零耗时流程边界，供报告定义任务窗口。"""
        marker: dict[str, Any] = {
            "name": str(name),
            "offset_s": round(max(0.0, time.perf_counter() - self.started_at), 6),
        }
        marker.update(details)
        self._markers.append(marker)
        self._sync()

    def _sync(self) -> None:
        if self._report is not None:
            # 事件记录频率可能达到每帧一次；同步检查点只写原始事件，避免
            # 每次 append 都重新计算一次完整视觉时间轴。最终落盘时调用
            # snapshot()（默认 include_visual=True）会补上视觉汇总。
            self._report["timing"] = self.snapshot(include_visual=False)

    def scoped_snapshot(self, prefix: str) -> dict[str, Any]:
        events = [item for item in self.events if str(item["name"]).startswith(prefix)]
        artifact_events = [
            item for item in events
            if str(item.get("category", "")) == "artifact_io"
            or str(item.get("name", "")).startswith("artifact_io/")
            or bool(item.get("exclude_from_effective", False))
        ]
        effective_events = [
            item for item in events
            if not (
                str(item.get("category", "")) == "artifact_io"
                or str(item.get("name", "")).startswith("artifact_io/")
                or bool(item.get("exclude_from_effective", False))
            )
        ]
        snapshot = {
            "prefix": prefix,
            "sum_elapsed_s": round(sum(float(item["elapsed_s"]) for item in events), 6),
            "effective_sum_elapsed_s": round(
                sum(
                    float(item.get("effective_elapsed_s", item["elapsed_s"]))
                    for item in effective_events
                ),
                6,
            ),
            "artifact_io_s": round(
                sum(float(item["elapsed_s"]) for item in artifact_events), 6,
            ),
            "wall_start_offset_s": (
                round(
                    min(float(item["start_offset_s"]) for item in events),
                    6,
                )
                if all("start_offset_s" in item for item in events) and events
                else None
            ),
            "wall_end_offset_s": (
                round(
                    max(float(item["end_offset_s"]) for item in events),
                    6,
                )
                if all("end_offset_s" in item for item in events) and events
                else None
            ),
            "events": events,
            "motion_events": [
                item for item in self.motion_events
                if str(item.get("parent_name", "")).startswith(prefix)
                or str(item.get("name", "")).startswith(prefix)
            ],
        }
        try:
            from aubo_workbench.hole_localization_report import (
                summarize_motion_distance,
                summarize_visual_timing_events,
            )
            snapshot["visual_timing"] = summarize_visual_timing_events(events)
            snapshot["motion_distance"] = summarize_motion_distance(
                snapshot["motion_events"],
            )
        except Exception:
            pass
        return snapshot

    def record(
        self, name: str, elapsed_s: float, status: str = "completed", **details: Any,
    ) -> None:
        now = time.perf_counter()
        end_offset_s = float(details.pop(
            "_end_offset_s", max(0.0, now - self.started_at),
        ))
        start_offset_s = float(details.pop(
            "_start_offset_s", max(0.0, end_offset_s - max(0.0, float(elapsed_s))),
        ))
        depth = int(details.pop("_depth", len(self._active_measurements)))
        parent_name = details.pop("_parent_name", None)
        category = str(details.get("category") or infer_timing_category(name))
        details["category"] = category
        visual_stage = details.get("visual_stage")
        if visual_stage is None and category == "vision_compute":
            visual_stage = infer_visual_timing_stage(name)
        if visual_stage in VISUAL_TIMING_STAGES:
            details["visual_stage"] = str(visual_stage)
        details.setdefault("level", "detail")
        event: dict[str, Any] = {
            "name": name,
            "elapsed_s": round(float(elapsed_s), 6),
            "status": status,
            "start_offset_s": round(max(0.0, start_offset_s), 6),
            "end_offset_s": round(max(max(0.0, start_offset_s), end_offset_s), 6),
            "depth": max(0, depth),
        }
        if parent_name:
            event["parent_name"] = str(parent_name)
        event.update(details)
        self.events.append(event)
        self._sync()
        if visual_stage in VISUAL_TIMING_STAGES:
            print(
                f"[TIMING] {name}: {float(elapsed_s):.3f}s [{status}] [VISION]",
                flush=True,
            )

    @contextmanager
    def measure(self, name: str, **details: Any):
        started_at = time.perf_counter()
        depth = len(self._active_measurements)
        parent_name = (
            str(self._active_measurements[-1]["name"])
            if self._active_measurements else None
        )
        scope: dict[str, Any] = {
            "name": str(name),
            "nested_artifact_io_s": 0.0,
        }
        self._active_measurements.append(scope)
        status = "completed"
        error: str | None = None
        try:
            yield
        except BaseException as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._active_measurements.pop()
            elapsed_s = time.perf_counter() - started_at
            nested_artifact_io_s = min(
                max(0.0, float(scope["nested_artifact_io_s"])),
                max(0.0, float(elapsed_s)),
            )
            details = {
                **details,
                # 与 record() 的 elapsed_s 使用相同精度，避免极短事件因
                # 浮点尾差出现 effective_elapsed_s > elapsed_s。
                "effective_elapsed_s": round(
                    min(elapsed_s, max(0.0, elapsed_s - nested_artifact_io_s)), 6,
                ),
            }
            if nested_artifact_io_s > 0.0:
                details["nested_artifact_io_s"] = round(nested_artifact_io_s, 6)
            if error is not None:
                details = {**details, "error": error}
            self.record(
                name,
                elapsed_s,
                status=status,
                _start_offset_s=max(0.0, started_at - self.started_at),
                _end_offset_s=max(0.0, time.perf_counter() - self.started_at),
                _depth=depth,
                _parent_name=parent_name,
                **details,
            )

    @contextmanager
    def measure_artifact(self, name: str, **details: Any):
        """记录一次文件产物生成，并把它从父计时的有效耗时中扣除。"""
        started_at = time.perf_counter()
        depth = len(self._active_measurements)
        parent_name = (
            str(self._active_measurements[-1]["name"])
            if self._active_measurements else None
        )
        status = "completed"
        error: str | None = None
        try:
            yield
        except BaseException as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed_s = time.perf_counter() - started_at
            # 该写盘操作可能位于多个嵌套的业务计时块中。每个父块都
            # 包含了这段写盘时间，因此每个父块都要扣除同一段实际时长。
            for scope in self._active_measurements:
                scope["nested_artifact_io_s"] += elapsed_s
            event_name = str(name).strip("/")
            if not event_name.startswith("artifact_io/"):
                event_name = f"artifact_io/{event_name}"
            artifact_details = {
                **details,
                "category": "artifact_io",
                "exclude_from_effective": True,
            }
            if error is not None:
                artifact_details["error"] = error
            self.record(
                event_name,
                elapsed_s,
                status=status,
                _start_offset_s=max(0.0, started_at - self.started_at),
                _end_offset_s=max(0.0, time.perf_counter() - self.started_at),
                _depth=depth,
                _parent_name=parent_name,
                **artifact_details,
            )

    def print_summary(self) -> None:
        snapshot = self.snapshot()
        # 现场性能关注视觉链路；机械臂运动和人工/安全等待仍保留在原始
        # events 里供审计，但不再混入终端的主计时摘要。
        try:
            from aubo_workbench.hole_localization_report import (
                summarize_visual_timing_events,
            )
            visual = summarize_visual_timing_events(list(snapshot["events"]))
        except Exception:
            visual = {
                "total_visual_s": 0.0,
                "stage_breakdown": {},
            }
        print(
            f"[TIMING_SUMMARY] visual_total={float(visual.get('total_visual_s', 0.0)):.3f}s",
            flush=True,
        )
        stage_breakdown = visual.get("stage_breakdown") or {}
        for stage, elapsed_s in sorted(
            stage_breakdown.items(),
            key=lambda item: float(item[1]),
            reverse=True,
        ):
            print(
                f"  visual/{stage}: total={float(elapsed_s):.3f}s",
                flush=True,
            )
        motion = snapshot.get("motion_distance") or {}
        if int(motion.get("segment_count", 0) or 0) > 0:
            print(
                "[MOTION_DISTANCE_SUMMARY] "
                f"total={float(motion.get('total_path_mm', 0.0)):.1f}mm "
                f"horizontal={float(motion.get('horizontal_distance_mm', 0.0)):.1f}mm "
                f"up={float(motion.get('vertical_up_mm', 0.0)):.1f}mm "
                f"down={float(motion.get('vertical_down_mm', 0.0)):.1f}mm "
                f"segments={int(motion.get('segment_count', 0) or 0)}",
                flush=True,
            )


@contextmanager
def artifact_measure(
    timing: Any | None, name: str, **details: Any,
):
    """兼容旧调用方的可选产物计时上下文。"""
    measure = getattr(timing, "measure_artifact", None)
    if callable(measure):
        with measure(name, **details):
            yield
    else:
        yield


@dataclass(frozen=True)
class TwoStageConfig:
    coarse_height_mm: float = 340.0
    fine_height_mm: float = 260.0
    # 逐孔精拍回退只需要在260 mm观察位上方留出局部横移净空；最终低位
    # 落位仍使用独立的60 mm安全余量，不能共用该较小值。
    per_hole_fine_safe_z_margin_mm: float = 20.0
    # 默认帧数按最近一次运行的稳定性下调；达到稳定质量门时还会提前结束。
    coarse_frames: int = 10
    fine_frames: int = 20
    height_tolerance_mm: float = 2.0
    center_tolerance_px: float = 5.0
    # 当前 Gemini 深度平面法向跨帧/跨视角重复性约 1–2°；粗阶段不应追逐到 0.5°。
    # 精定位仍锁定最终粗姿态并使用 RGB 进行 XY 微调。
    normal_tolerance_deg: float = 2.0
    max_z_corrections: int = 4
    min_coarse_valid: int = 8
    min_fine_valid: int = 12
    # 镀膜曲面工件的初始环带深度允许少量结构化噪声；粗/精阶段门限保持不变。
    initial_max_plane_rmse_mm: float = 3.5
    max_plane_rmse_mm: float = 3.5
    coarse_settle_frames: int = 5
    # 跨运行缓存现场验证：到位后丢弃相机/末端预热帧，再用更多稳定帧验证。
    cache_validation_settle_discard_frames: int = 10
    cache_validation_frames: int = 8
    cache_validation_min_valid: int = 5
    # 缓存缺失或损坏回退完整粗定位时，到达340 mm后给机械臂/相机的
    # 固定停稳缓冲；直接命中base缓存时不等待、不做现场复核。
    coarse_settle_delay_s: float = 0.6
    # 每次单孔粗拍重拍前重新确认机器人停稳，并丢弃相机队列中的预热帧。
    coarse_recapture_settle_discard_frames: int = 10
    # 单孔粗定位最多允许两次姿态纠偏，最后一次采集只做验证，不再盲目运动。
    coarse_max_corrections: int = 2
    coarse_max_attempt_multiplier: int = 4
    max_coarse_center_scatter_p95_px: float = 0.8
    coarse_min_ring_coverage_ratio: float = 0.75
    coarse_max_ring_gap_deg: float = 60.0
    batch_coarse_quality_singleton_retry: bool = True
    fine_stable_min_frames: int = 15
    fine_stable_center_scatter_p95_px: float = 0.6
    # 相机管线持续运行；精定位只自适应清理旧帧，不固定丢弃一批“预热帧”。
    # 显式设置为正数时仍保留最少丢帧下限，用于现场需要更强稳定性的情况。
    fine_settle_discard_frames: int = 0
    # 单孔精定位质量门失败时只重拍当前孔，不中断整个多孔流程。
    fine_retry_count: int = 2
    # 多次重拍仍略超严格门槛时，允许稳定但降级的结果继续执行并留痕。
    fine_degraded_max_center_scatter_p95_px: float = 0.6
    max_ellipse_residual_px: float = 0.9
    # 粗定位身份关联使用固定锚点；比精拍适当放宽，兼容粗拍时
    # YOLO框中心在局部倾斜和深度噪声下的少量变化。
    multi_coarse_tracking_tolerance_px: float = 70.0
    min_ellipse_coverage_deg: float = 200.0
    max_fine_center_scatter_p95_px: float = 0.35
    # 精定位先使用粗定位点云中心在当前RGB相机中的投影作为身份锚点。
    # 该门限明显小于相邻孔间距，避免密集孔阵列中锁到邻孔。
    fine_pointcloud_anchor_tolerance_px: float = 30.0
    # 反光会让严格椭圆残差门（0.9 px）间歇性失败；只有在点云投影锚点
    # 已锁定目标且拟合仍满足较宽的几何门时，才允许该帧继续使用YOLO中心。
    fine_yolo_fallback_max_ellipse_residual_px: float = 2.5
    fine_yolo_fallback_min_ellipse_coverage_deg: float = 45.0
    enable_tilt_center_correction: bool = True
    tilt_correction_iterations: int = 4
    tilt_correction_samples: int = 240
    max_tilt_correction_mm: float = 3.0
    # 批量粗定位模式参数
    batch_coarse_localization: bool = False
    # 逐帧叠加图只用于诊断，不参与定位计算。默认只保留每次采集的最后一帧，
    # 避免正常运行把大量PNG编码和磁盘写入混入拍摄节拍。
    save_all_capture_overlays: bool = False
    batch_coarse_frames: int = 15
    batch_coarse_min_valid: int = 10
    # 粗定位达到最少有效帧后，再额外保留少量帧确认稳定；满足稳定门即提前结束，
    # 避免每组在已经足够可靠时固定拍满 batch_coarse_frames。
    batch_coarse_early_stop_extra_frames: int = 2
    batch_coarse_min_holes_per_frame: int | None = None
    batch_coarse_view_margin_px: float = 50.0
    shared_cache_validation: bool = False
    shared_cache_validation_frames: int = 3
    shared_cache_validation_min_valid: int = 2
    shared_cache_validation_view_margin_px: float = 50.0
    # 批量移动完成后先丢弃相机队列中的旧帧，避免把运动过程帧融合进固定TCP位姿。
    batch_coarse_settle_discard_frames: int = 10
    # 普通粗定位及第四策略点云采集共用同一套稳定性门；实际高度由
    # coarse_height_mm（第四策略启动时折叠为独立高度）决定。
    max_coarse_tracking_distance_p95_px: float = 15.0
    # 共享粗定位专用的到位后现场复核。该段只在共享粗定位路径读取，
    # 逐孔粗定位、逐孔精定位和共享精定位不受影响。
    batch_coarse_group_pose_refinement: bool = True
    batch_coarse_pose_refine_frames: int = 5
    batch_coarse_pose_refine_min_valid_frames: int = 3
    # 共享粗定位位姿纠偏采用小步闭环：一次拍摄、一次小移动、再拍摄。
    # 340 mm共享组的历史纠偏量达到18 mm/5°左右，2轮无法收敛。
    batch_coarse_pose_refine_max_iterations: int = 8
    batch_coarse_pose_refine_settle_discard_frames: int = 5
    batch_coarse_pose_refine_tracking_tolerance_px: float = 45.0
    batch_coarse_pose_refine_max_center_scatter_p95_px: float = 1.5
    batch_coarse_pose_refine_max_tracking_distance_p95_px: float = 20.0
    batch_coarse_pose_refine_max_reprojection_error_px: float = 8.0
    batch_coarse_pose_refine_pose_tolerance_mm: float = 2.0
    batch_coarse_pose_refine_rotation_tolerance_deg: float = 1.0
    batch_coarse_pose_refine_min_move_mm: float = 0.5
    # 现场复核每一步只允许小幅纠偏；超过门限时先按比例拆步，避免
    # 把18 mm/5°级的整体修正变成一次大范围水平/姿态运动。
    batch_coarse_pose_refine_max_correction_mm: float = 5.0
    # 姿态差较大时允许单步最多修正2°；平移仍受5 mm门限共同约束，
    # 因此现场大多数斜向纠偏实际约为1.4~1.6°，而不是直接跳完整误差。
    batch_coarse_pose_refine_max_correction_rotation_deg: float = 2.0
    # 同一孔组已处于340 mm观察位时，小步纠偏直接从当前位置MoveL；
    # 关闭后恢复抬升、横移、下降的保守路径，便于现场快速回退。
    batch_coarse_pose_refine_direct_motion: bool = True
    # 这是整个共享组闭环允许的累计修正上限，不是单步上限。
    batch_coarse_pose_refine_max_total_correction_mm: float = 25.0
    batch_coarse_pose_refine_max_total_correction_rotation_deg: float = 7.0
    # 建图的安全横移高度余量。建图碰撞风险更高，默认比普通共享观察位
    # 的60 mm余量更保守；实际运动仍受控制器安全状态限制。
    map_build_safe_z_margin_mm: float = 100.0
    # 共享粗定位允许自动拆成多组。单组投影范围过大时，即使全部孔仍在
    # 画面内，也应拆组让每组位于更可靠的中央视野。
    batch_coarse_max_view_span_ratio: float = 0.35
    # 正式共享粗定位分组：每组最多5孔，且必须是相邻、紧凑的空间簇。
    batch_coarse_max_group_size: int = 5
    batch_coarse_group_max_aspect_ratio: float = 2.0
    batch_coarse_group_adjacency_factor: float = 1.8
    # 防止只靠近邻链把远处孔串入同一组。该门限作用于组内孔位在
    # 机器人基坐标XY平面的最大直径；超过后必须拆成更局部的组。
    batch_coarse_group_max_xy_diameter_mm: float = 150.0
    batch_coarse_group_max_normal_spread_deg: float = 3.0
    # 260mm批量精定位：在同一视野内一次采集并精定位全部已选孔。
    batch_fine_localization: bool = True
    batch_fine_frames: int = 8
    batch_fine_min_valid: int = 5
    batch_fine_stable_min_frames: int = 5
    # 260mm下降后RGB pipeline里可能残留运动过程帧；默认按时间戳自适应
    # 清理到实时流，不固定丢弃10帧。显式设为正数时才增加最少丢帧下限。
    batch_fine_settle_discard_frames: int = 0
    # 共享精拍首拍质量不足时，先在当前位置继续补少量帧；仍不满足质量门
    # 才由外层规划新的共享观察位，避免为“只差一两帧”的孔重复移动机械臂。
    batch_fine_inplace_recovery_frames: int = 4
    # 共享精拍某些孔失败时，允许对失败孔子组执行受限位姿调整后补拍；
    # 第二步只在第一步确实被单步运动门限截断且仍失败时执行。
    batch_fine_supplement_rounds: int = 2
    batch_fine_view_margin_px: float = 50.0
    # 精定位优先使用3～4孔的紧凑组：多孔给联合变换提供冗余，并减少组间
    # 长距离运动。首拍失败孔由下方的组内受限位姿调整重新观察。
    batch_fine_max_view_span_ratio: float = 0.55
    batch_fine_max_group_size: int = 4
    batch_fine_group_max_aspect_ratio: float = 1.8
    batch_fine_group_adjacency_factor: float = 1.8
    batch_fine_group_max_normal_spread_deg: float = 5.0
    # 共享精拍的几何圆心必须靠近粗定位投影锚点。检测框中心只用于身份
    # 分配，不能代替真实圆心通过此门，否则会接受“框匹配正确、圆心错误”。
    batch_fine_max_geometric_anchor_distance_px: float = 5.0
    # 联合变换失败时只允许小幅、无歧义的逐孔结果回退；超过此位移的孔
    # 留给空间分组补拍，避免把稳定但错误的圆心写入最终结果或孔位地图。
    batch_fine_fallback_max_coarse_to_fine_xy_mm: float = 3.0
    # 关闭组内位姿调整时，旧的共享补拍路径仍使用该视野跨度拆组。
    batch_fine_supplement_max_view_span_ratio: float = 0.55
    # 共享精拍失败后，不再默认抬升、横移、下降到另一个观察位。根据失败孔
    # 子组的粗定位中心和融合平面法向重算观察位，并在当前260mm附近执行一次
    # 受限MoveL。RZ始终锁定；法向不可靠时只调整平移，不调整RX/RY。
    batch_fine_in_group_pose_adjustment: bool = True
    batch_fine_in_group_max_adjustments: int = 2
    batch_fine_in_group_max_xy_mm: float = 8.0
    batch_fine_in_group_max_z_mm: float = 3.0
    batch_fine_in_group_max_rotation_deg: float = 2.0
    batch_fine_in_group_min_normal_holes: int = 2
    batch_fine_in_group_max_normal_spread_deg: float = 3.0
    # 共享精拍质量门失败后，保留已成功的共享结果；失败孔逐孔补拍，
    # 这样单个坏孔不会把整批结果置为 deferred。
    batch_fine_per_hole_fallback: bool = True
    # 第四种检测策略：只使用独立高度的点云得到孔中心，最终XY只使用
    # ChArUco纠偏；点云中心不可用时记录失败，不进入其它定位流程。
    coarse_direct_final: bool = False
    # 第四策略的试验参数独立于普通两阶段粗定位，便于做高度/分组对照。
    coarse_direct_final_height_mm: float = 340.0
    coarse_direct_final_max_group_size: int = 5
    coarse_direct_final_early_stop_extra_frames: int = 5
    # 到达共享观察位并确认控制器稳定后，额外等待机械臂/工装的微振动衰减。
    coarse_direct_final_settle_delay_s: float = 1.0
    # 纯点云分组搜索和拍摄都必须有边界，避免复杂候选或SDK取帧异常让流程
    # 长时间没有可解释进度。底层单次SDK调用仍受其自身请求超时限制。
    coarse_direct_final_group_planning_timeout_s: float = 10.0
    coarse_direct_final_settle_discard_timeout_s: float = 10.0
    coarse_direct_final_capture_timeout_s: float = 60.0
    coarse_direct_final_steady_timeout_s: float = 45.0
    coarse_direct_final_max_consecutive_frame_failures: int = 3
    # 仅采集评估仍会移动到观察位并采集点云，但不会执行最终XY/Z动作。
    coarse_direct_final_capture_only: bool = False
    # 共享精定位专用：把同一260mm视野内的多个孔作为一个平面刚体
    # 约束联合求解。该开关只在batch_fine_localization路径生效；逐孔
    # 精定位、共享粗定位和其它模式不读取这些字段。
    batch_fine_joint_localization: bool = True
    batch_fine_joint_min_holes: int = 2
    batch_fine_joint_min_valid_frames: int = 5
    # 1.5 mm 会让两孔组中明显互相矛盾的位移仍勉强通过；1.0 mm 可将
    # 这类无冗余、不能估旋转的组合安全地送入补拍/逐孔回退。
    batch_fine_joint_max_residual_mm: float = 1.0
    # 对刚体联合解，门限检查每个内点孔在其实际基坐标位置的最大修正量，
    # 而不是受旋转中心影响的基坐标原点平移参数。
    batch_fine_joint_max_translation_mm: float = 5.0
    batch_fine_joint_max_yaw_deg: float = 3.0
    batch_fine_joint_stable_translation_mm: float = 0.25
    batch_fine_joint_stable_yaw_deg: float = 0.15
    # 联合结果为主；保留小幅单孔残差，兼容实际加工误差和圆心局部偏差。
    batch_fine_joint_local_residual_weight: float = 0.25
    batch_fine_joint_local_residual_limit_mm: float = 0.5
    # 最终共享精定位XY再受粗拍点云支撑中心约束。只在粗精中心一致时
    # 做小幅拉回；实际融合时还会按孔级联合残差自适应降权：视觉联合
    # 越稳定，粗拍点云介入越少，避免把粗定位系统偏差重新带回最终点。
    batch_fine_pointcloud_xy_fusion: bool = True
    batch_fine_pointcloud_xy_weight: float = 0.35
    batch_fine_pointcloud_xy_max_correction_mm: float = 0.6
    batch_fine_pointcloud_xy_agreement_gate_mm: float = 2.5

    @classmethod
    def from_namespace(cls, args: Any) -> "TwoStageConfig":
        """从 argparse/GUI 命名空间构造配置，避免在运行入口重复维护默认值。"""
        defaults = cls()
        aliases = {"fine_retry_count": "fine_retries"}
        values: dict[str, Any] = {}
        for field in fields(defaults):
            source_name = aliases.get(field.name, field.name)
            # 处理带下划线和带连字符的参数名（命令行参数使用连字符）
            alt_source_name = source_name.replace("_", "-")
            if not hasattr(args, source_name) and not hasattr(args, alt_source_name):
                continue
            value = getattr(args, source_name, None)
            if value is None and hasattr(args, alt_source_name):
                value = getattr(args, alt_source_name, None)
            if value is None:
                values[field.name] = None
                continue
            default = getattr(defaults, field.name)
            if isinstance(default, bool):
                value = bool(value)
            elif isinstance(default, int):
                value = int(value)
            elif isinstance(default, float):
                value = float(value)
            elif field.name == "batch_coarse_min_holes_per_frame":
                value = int(value)
            values[field.name] = value
        return cls(**values)

    def validate(self, *, reuse_coarse_cache: bool = True) -> None:
        """校验两阶段定位参数之间的约束。"""

        def positive(value: float, label: str) -> None:
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{label}必须是大于0的有限数字")

        def nonnegative(value: float, label: str) -> None:
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{label}必须是大于等于0的有限数字")

        def ratio(value: float, label: str) -> None:
            if not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{label}必须在(0, 1]内")

        def count_range(value: int, maximum: int, label: str) -> None:
            if int(value) < 1 or int(value) > int(maximum):
                raise ValueError(f"{label}必须在1到{int(maximum)}之间")

        nonnegative(self.fine_settle_discard_frames, "精定位预热丢弃帧数")
        ratio(self.coarse_min_ring_coverage_ratio, "粗定位环带最小覆盖率")
        positive(self.coarse_max_ring_gap_deg, "粗定位环带最大缺口角度")
        if self.coarse_max_ring_gap_deg > 360:
            raise ValueError("粗定位环带最大缺口角度不能超过360度")
        nonnegative(self.coarse_settle_delay_s, "粗定位停稳缓冲时间")
        nonnegative(self.cache_validation_settle_discard_frames, "缓存验证预热丢弃帧数")
        positive(self.cache_validation_frames, "缓存验证帧数")
        count_range(
            self.cache_validation_min_valid,
            self.cache_validation_frames,
            "缓存验证最少有效帧数",
        )
        nonnegative(self.coarse_recapture_settle_discard_frames, "粗定位重拍停稳丢弃帧数")
        nonnegative(self.coarse_max_corrections, "粗定位最大姿态纠偏次数")
        nonnegative(self.fine_retry_count, "精定位重试次数")
        if self.fine_height_mm >= self.coarse_height_mm:
            raise ValueError("精定位高度必须小于粗定位高度")
        if (
            not math.isfinite(float(self.per_hole_fine_safe_z_margin_mm))
            or float(self.per_hole_fine_safe_z_margin_mm) < 10.0
        ):
            raise ValueError("逐孔精拍安全横移高度余量必须是不小于10 mm的有限数")
        if self.coarse_frames < self.min_coarse_valid:
            raise ValueError(
                f"粗定位最大帧数必须不少于最小有效帧数："
                f"{self.coarse_frames} < {self.min_coarse_valid}"
            )
        if self.fine_frames < self.fine_stable_min_frames:
            raise ValueError(
                f"精定位最大帧数必须不少于稳定门帧数："
                f"{self.fine_frames} < {self.fine_stable_min_frames}"
            )

        if self.shared_cache_validation:
            if not reuse_coarse_cache:
                raise ValueError("共享缓存快速验证要求启用粗定位缓存复用")
            positive(self.shared_cache_validation_frames, "共享缓存验证帧数")
            count_range(
                self.shared_cache_validation_min_valid,
                self.shared_cache_validation_frames,
                "共享缓存验证最少有效帧数",
            )
            nonnegative(self.shared_cache_validation_view_margin_px, "共享缓存验证视野边缘余量")
            if self.batch_coarse_localization:
                raise ValueError("共享缓存快速验证不能与批量粗定位同时启用")

        if self.batch_coarse_localization:
            if self.batch_coarse_frames < self.batch_coarse_min_valid:
                raise ValueError(
                    "批量粗定位最大帧数必须不少于最小有效帧数："
                    f"{self.batch_coarse_frames} < {self.batch_coarse_min_valid}"
                )
            nonnegative(self.batch_coarse_early_stop_extra_frames, "批量粗定位提前结束确认帧数")
            nonnegative(self.batch_coarse_settle_discard_frames, "批量粗定位停稳丢弃帧数")
            positive(self.max_coarse_center_scatter_p95_px, "粗定位中心散布P95门限")
            positive(self.max_coarse_tracking_distance_p95_px, "粗定位跟踪距离P95门限")
            nonnegative(self.batch_coarse_view_margin_px, "批量粗定位视野边缘余量")
            positive(self.batch_coarse_pose_refine_frames, "共享粗定位现场复核帧数")
            count_range(
                self.batch_coarse_pose_refine_min_valid_frames,
                self.batch_coarse_pose_refine_frames,
                "共享粗定位现场复核最少有效帧数",
            )
            positive(self.batch_coarse_pose_refine_max_iterations, "共享粗定位现场复核最大迭代次数")
            nonnegative(
                self.batch_coarse_pose_refine_settle_discard_frames,
                "共享粗定位现场复核预热丢弃帧数",
            )
            for value, label in (
                (self.batch_coarse_pose_refine_tracking_tolerance_px, "共享粗定位现场复核跟踪门限"),
                (self.batch_coarse_pose_refine_max_center_scatter_p95_px, "共享粗定位现场复核中心散布门限"),
                (self.batch_coarse_pose_refine_max_tracking_distance_p95_px, "共享粗定位现场复核跟踪距离门限"),
                (self.batch_coarse_pose_refine_max_reprojection_error_px, "共享粗定位现场复核重投影门限"),
                (self.batch_coarse_pose_refine_pose_tolerance_mm, "共享粗定位现场复核位姿门限"),
                (self.batch_coarse_pose_refine_rotation_tolerance_deg, "共享粗定位现场复核姿态门限"),
                (self.batch_coarse_pose_refine_min_move_mm, "共享粗定位现场复核最小移动量"),
                (self.batch_coarse_pose_refine_max_correction_mm, "共享粗定位现场复核最大纠偏平移"),
                (self.batch_coarse_pose_refine_max_correction_rotation_deg, "共享粗定位现场复核最大纠偏旋转"),
                (self.batch_coarse_pose_refine_max_total_correction_mm, "共享粗定位现场复核累计最大纠偏平移"),
                (self.batch_coarse_pose_refine_max_total_correction_rotation_deg, "共享粗定位现场复核累计最大纠偏旋转"),
            ):
                positive(value, label)
            if self.batch_coarse_pose_refine_max_total_correction_mm < self.batch_coarse_pose_refine_max_correction_mm:
                raise ValueError("共享粗定位累计最大纠偏平移不能小于单步最大纠偏平移")
            if self.batch_coarse_pose_refine_max_total_correction_rotation_deg < self.batch_coarse_pose_refine_max_correction_rotation_deg:
                raise ValueError("共享粗定位累计最大纠偏旋转不能小于单步最大纠偏旋转")

        positive(self.map_build_safe_z_margin_mm, "建图安全横移高度余量")
        positive(self.coarse_direct_final_height_mm, "第四策略点云拍摄高度")
        if int(self.coarse_direct_final_max_group_size) < 1 or int(
            self.coarse_direct_final_max_group_size
        ) > 5:
            raise ValueError("第四策略每组最多孔数必须在1到5之间")
        nonnegative(
            self.coarse_direct_final_early_stop_extra_frames,
            "第四策略提前结束确认帧数",
        )
        nonnegative(
            self.coarse_direct_final_settle_delay_s,
            "第四策略停稳后额外等待时间",
        )
        if float(self.coarse_direct_final_settle_delay_s) > 30.0:
            raise ValueError("第四策略停稳后额外等待时间必须不超过30秒")
        for value, label in (
            (self.coarse_direct_final_group_planning_timeout_s, "第四策略分组规划超时"),
            (self.coarse_direct_final_settle_discard_timeout_s, "第四策略旧帧清理超时"),
            (self.coarse_direct_final_capture_timeout_s, "第四策略点云采集超时"),
            (self.coarse_direct_final_steady_timeout_s, "第四策略停稳等待超时"),
        ):
            positive(value, label)
        if int(self.coarse_direct_final_max_consecutive_frame_failures) < 1:
            raise ValueError("第四策略连续取帧失败上限必须至少为1")

        ratio(self.batch_coarse_max_view_span_ratio, "共享粗定位单组视野跨度比例")
        positive(self.batch_coarse_max_group_size, "共享粗定位单组最多孔数")
        if (
            not math.isfinite(self.batch_coarse_group_max_aspect_ratio)
            or self.batch_coarse_group_max_aspect_ratio < 1.0
        ):
            raise ValueError("共享粗定位单组最大长宽比必须是不小于1的有限数字")
        if (
            not math.isfinite(self.batch_coarse_group_adjacency_factor)
            or self.batch_coarse_group_adjacency_factor < 1.0
        ):
            raise ValueError("共享粗定位相邻距离倍数必须是不小于1的有限数字")
        positive(self.batch_coarse_group_max_xy_diameter_mm, "共享粗定位单组最大XY直径")
        nonnegative(self.batch_coarse_group_max_normal_spread_deg, "共享粗定位单组法向离散门限")

        nonnegative(self.batch_fine_view_margin_px, "批量精定位视野边缘余量")
        ratio(self.batch_fine_max_view_span_ratio, "共享精定位单组视野跨度比例")
        positive(self.batch_fine_max_group_size, "共享精定位单组最多孔数")
        if (
            not math.isfinite(self.batch_fine_group_max_aspect_ratio)
            or self.batch_fine_group_max_aspect_ratio < 1.0
        ):
            raise ValueError("共享精定位单组最大长宽比必须是不小于1的有限数字")
        if (
            not math.isfinite(self.batch_fine_group_adjacency_factor)
            or self.batch_fine_group_adjacency_factor < 1.0
        ):
            raise ValueError("共享精定位相邻距离倍数必须是不小于1的有限数字")
        nonnegative(self.batch_fine_group_max_normal_spread_deg, "共享精定位单组法向离散门限")
        positive(self.batch_fine_max_geometric_anchor_distance_px, "共享精定位几何圆心锚点距离门限")
        positive(self.batch_fine_fallback_max_coarse_to_fine_xy_mm, "共享精定位粗精XY安全回退门限")
        ratio(self.batch_fine_supplement_max_view_span_ratio, "共享精定位补拍视野跨度比例")
        nonnegative(
            self.batch_fine_in_group_max_adjustments,
            "共享精定位组内位姿调整次数",
        )
        positive(self.batch_fine_in_group_max_xy_mm, "共享精定位组内XY调整门限")
        nonnegative(self.batch_fine_in_group_max_z_mm, "共享精定位组内Z调整门限")
        nonnegative(
            self.batch_fine_in_group_max_rotation_deg,
            "共享精定位组内姿态调整门限",
        )
        positive(
            self.batch_fine_in_group_min_normal_holes,
            "共享精定位组内姿态调整最少法向孔数",
        )
        nonnegative(
            self.batch_fine_in_group_max_normal_spread_deg,
            "共享精定位组内姿态调整法向离散门限",
        )

        if self.batch_fine_localization:
            positive(self.batch_fine_frames, "批量精定位帧数")
            count_range(
                self.batch_fine_min_valid,
                self.batch_fine_frames,
                "批量精定位最少有效帧数",
            )
            count_range(
                self.batch_fine_stable_min_frames,
                self.batch_fine_frames,
                "批量精定位稳定门帧数",
            )
            nonnegative(self.batch_fine_settle_discard_frames, "批量精定位停稳丢弃帧数")
            nonnegative(self.batch_fine_inplace_recovery_frames, "批量精定位原位补帧数")
            nonnegative(self.batch_fine_supplement_rounds, "批量精定位共享补拍轮数")
            if self.batch_fine_joint_localization:
                if self.batch_fine_joint_min_holes < 2:
                    raise ValueError("共享精定位联合最少孔数必须不少于2")
                count_range(
                    self.batch_fine_joint_min_valid_frames,
                    self.batch_fine_frames,
                    "共享精定位联合稳定帧数",
                )
                for value, label in (
                    (self.batch_fine_joint_max_residual_mm, "共享精定位联合残差门限"),
                    (self.batch_fine_joint_max_translation_mm, "共享精定位联合平移门限"),
                    (self.batch_fine_joint_max_yaw_deg, "共享精定位联合旋转门限"),
                    (self.batch_fine_joint_stable_translation_mm, "共享精定位联合平移稳定门限"),
                    (self.batch_fine_joint_stable_yaw_deg, "共享精定位联合旋转稳定门限"),
                ):
                    positive(value, label)
                if not 0.0 <= self.batch_fine_joint_local_residual_weight <= 1.0:
                    raise ValueError("共享精定位逐孔残差权重必须在0到1之间")
                nonnegative(
                    self.batch_fine_joint_local_residual_limit_mm,
                    "共享精定位逐孔残差限幅",
                )
            if self.batch_fine_pointcloud_xy_fusion:
                if not 0.0 <= self.batch_fine_pointcloud_xy_weight <= 1.0:
                    raise ValueError("共享精定位点云中心融合权重必须在0到1之间")
                nonnegative(
                    self.batch_fine_pointcloud_xy_max_correction_mm,
                    "共享精定位点云中心最大修正量",
                )
                positive(
                    self.batch_fine_pointcloud_xy_agreement_gate_mm,
                    "共享精定位点云中心一致性门限",
                )


@dataclass
class PlaneEstimate:
    point_camera_mm: np.ndarray
    normal_camera: np.ndarray
    rmse_mm: float
    ring_points: int
    surface_model: str = "sphere"
    sphere_center_camera_mm: np.ndarray | None = None
    sphere_radius_mm: float | None = None
    surface_plane_point_camera_mm: np.ndarray | None = None
    points_camera_mm: np.ndarray | None = None
    surface_selection_policy: str = "legacy"
    front_surface_z_mm: float | None = None
    ring_points_raw: int | None = None
    surface_points_selected: int | None = None
    ring_coverage_ratio: float | None = None
    ring_max_gap_deg: float | None = None
    raw_points_camera_mm: np.ndarray | None = None
    raw_pixels: np.ndarray | None = None
    ring_pixels: np.ndarray | None = None
    surface_selected_mask: np.ndarray | None = None


@dataclass
class Observation:
    stage: str
    frame_index: int
    center_px: np.ndarray
    ellipse: dict[str, Any] | None = None
    plane: PlaneEstimate | None = None
    timestamp_ns: int | None = None
    error: str | None = None
    center_source: str = "unknown"
    quality_note: str | None = None
    tracking_distance_px: float | None = None
    geometric_anchor_distance_px: float | None = None
