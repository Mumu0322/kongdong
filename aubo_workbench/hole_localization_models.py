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
        self._report: dict[str, Any] | None = None
        self._active_measurements: list[dict[str, Any]] = []
        self._markers: list[dict[str, Any]] = []

    def attach_report(self, report: dict[str, Any]) -> None:
        self._report = report
        self._sync()

    def snapshot(self) -> dict[str, Any]:
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
        return {
            "schema_version": 2,
            "total_elapsed_s": round(elapsed_since_start, 6),
            "effective_wall_elapsed_s": round(
                max(0.0, elapsed_since_start - artifact_elapsed_s), 6,
            ),
            "events": list(self.events),
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
            self._report["timing"] = self.snapshot()

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
        return {
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
        }

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
        print(
            f"[TIMING] {name}: {float(elapsed_s):.3f}s [{status}]",
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
        print(
            f"[TIMING_SUMMARY] total={float(snapshot['total_elapsed_s']):.3f}s",
            flush=True,
        )
        aggregates = snapshot["aggregates"]
        ranked = sorted(
            aggregates.items(),
            key=lambda item: float(item[1]["total_elapsed_s"]),
            reverse=True,
        )
        for name, item in ranked[:12]:
            print(
                f"  {name}: total={float(item['total_elapsed_s']):.3f}s "
                f"count={int(item['count'])} max={float(item['max_elapsed_s']):.3f}s",
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
    fine_stable_min_frames: int = 15
    fine_stable_center_scatter_p95_px: float = 0.6
    # 机械臂到达精定位高度后，先丢弃相机队列和末端微振动产生的预热帧。
    fine_settle_discard_frames: int = 10
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
    # 所有340mm粗定位（批量和逐孔）共用同一套稳定性门。
    max_coarse_tracking_distance_p95_px: float = 15.0
    # 共享粗定位专用的到位后现场复核。该段只在共享粗定位路径读取，
    # 逐孔粗定位、逐孔精定位和共享精定位不受影响。
    batch_coarse_group_pose_refinement: bool = True
    batch_coarse_pose_refine_frames: int = 5
    batch_coarse_pose_refine_min_valid_frames: int = 3
    batch_coarse_pose_refine_max_iterations: int = 2
    batch_coarse_pose_refine_settle_discard_frames: int = 5
    batch_coarse_pose_refine_tracking_tolerance_px: float = 45.0
    batch_coarse_pose_refine_max_center_scatter_p95_px: float = 1.5
    batch_coarse_pose_refine_max_tracking_distance_p95_px: float = 20.0
    batch_coarse_pose_refine_max_reprojection_error_px: float = 8.0
    batch_coarse_pose_refine_pose_tolerance_mm: float = 2.0
    batch_coarse_pose_refine_rotation_tolerance_deg: float = 1.0
    batch_coarse_pose_refine_min_move_mm: float = 0.5
    # 共享粗定位允许自动拆成多组。单组投影范围过大时，即使全部孔仍在
    # 画面内，也应拆组让每组位于更可靠的中央视野。
    batch_coarse_max_view_span_ratio: float = 0.55
    # 正式共享粗定位分组：每组最多9孔，且必须是相邻、紧凑的空间簇。
    batch_coarse_max_group_size: int = 9
    batch_coarse_group_max_aspect_ratio: float = 2.0
    batch_coarse_group_adjacency_factor: float = 1.8
    # 防止只靠近邻链把远处孔串入同一组。该门限作用于组内孔位在
    # 机器人基坐标XY平面的最大直径；超过后必须拆成更局部的组。
    batch_coarse_group_max_xy_diameter_mm: float = 180.0
    batch_coarse_group_max_normal_spread_deg: float = 8.0
    # 260mm批量精定位：在同一视野内一次采集并精定位全部已选孔。
    batch_fine_localization: bool = True
    batch_fine_frames: int = 8
    batch_fine_min_valid: int = 5
    batch_fine_stable_min_frames: int = 5
    # 260mm下降后RGB pipeline里会残留运动过程帧。现场实测丢3帧只耗时
    # 3.7ms，仍未清空队列；默认与可靠的逐孔精定位一致，至少丢10帧。
    batch_fine_settle_discard_frames: int = 10
    # 共享精拍首拍质量不足时，先在当前位置继续补少量帧；仍不满足质量门
    # 才由外层规划新的共享观察位，避免为“只差一两帧”的孔重复移动机械臂。
    batch_fine_inplace_recovery_frames: int = 4
    # 共享精拍某些孔的严格几何有效帧不足时，允许移动到失败孔的共同
    # 260mm观察位补拍；补拍仍按孔集合共享，不退化为逐孔精拍。
    batch_fine_supplement_rounds: int = 1
    batch_fine_view_margin_px: float = 50.0
    batch_fine_max_view_span_ratio: float = 0.60
    # 正式共享精定位分组：每组最多3孔，优先保证局部视野和几何一致性。
    batch_fine_max_group_size: int = 3
    batch_fine_group_max_aspect_ratio: float = 1.8
    batch_fine_group_adjacency_factor: float = 1.8
    batch_fine_group_max_normal_spread_deg: float = 5.0
    # 共享精拍的几何圆心必须靠近粗定位投影锚点。检测框中心只用于身份
    # 分配，不能代替真实圆心通过此门，否则会接受“框匹配正确、圆心错误”。
    batch_fine_max_geometric_anchor_distance_px: float = 5.0
    # 联合变换失败时只允许小幅、无歧义的逐孔结果回退；超过此位移的孔
    # 留给空间分组补拍，避免把稳定但错误的圆心写入最终结果或孔位地图。
    batch_fine_fallback_max_coarse_to_fine_xy_mm: float = 3.0
    # 首次共享拍摄仍覆盖全部选孔；补拍按更紧凑视野拆组，确保机械臂
    # 真正移动到失败孔上方，而不是在原共同位置附近重复同一画面。
    batch_fine_supplement_max_view_span_ratio: float = 0.60
    # 共享精拍质量门失败后，保留已成功的共享结果；失败孔逐孔补拍，
    # 这样单个坏孔不会把整批结果置为 deferred。
    batch_fine_per_hole_fallback: bool = True
    # 共享精定位专用：把同一260mm视野内的多个孔作为一个平面刚体
    # 约束联合求解。该开关只在batch_fine_localization路径生效；逐孔
    # 精定位、共享粗定位和其它模式不读取这些字段。
    batch_fine_joint_localization: bool = True
    batch_fine_joint_min_holes: int = 2
    batch_fine_joint_min_valid_frames: int = 5
    batch_fine_joint_max_residual_mm: float = 1.5
    batch_fine_joint_max_translation_mm: float = 5.0
    batch_fine_joint_max_yaw_deg: float = 3.0
    batch_fine_joint_stable_translation_mm: float = 0.25
    batch_fine_joint_stable_yaw_deg: float = 0.15
    # 联合结果为主；保留小幅单孔残差，兼容实际加工误差和圆心局部偏差。
    batch_fine_joint_local_residual_weight: float = 0.25
    batch_fine_joint_local_residual_limit_mm: float = 0.5

    @classmethod
    def from_namespace(cls, args: Any) -> "TwoStageConfig":
        """从 argparse/GUI 命名空间构造配置，避免在运行入口重复维护默认值。"""
        defaults = cls()
        aliases = {"fine_retry_count": "fine_retries"}
        values: dict[str, Any] = {}
        for field in fields(defaults):
            source_name = aliases.get(field.name, field.name)
            if not hasattr(args, source_name):
                continue
            value = getattr(args, source_name)
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
            ):
                positive(value, label)

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
