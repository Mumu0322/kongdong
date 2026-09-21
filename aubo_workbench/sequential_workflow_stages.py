"""Compatibility facade for sequential workflow stages.

The context and runtime bridge stay here so the legacy runner and its patch
points remain stable. Heavy stage implementations live in
sequential_workflow_shared and sequential_hole_execution.
"""

from __future__ import annotations

from aubo_workbench import sequential_hole_execution as _hole_execution
from aubo_workbench import sequential_workflow_shared as _shared_stages
from dataclasses import dataclass
from typing import Any


_RUNTIME_DEPENDENCIES = {"init_pipeline"}


def install_runtime(symbols: dict[str, object]) -> None:
    for name in _RUNTIME_DEPENDENCIES:
        if name in symbols:
            globals()[name] = symbols[name]
    runtime = {**symbols, **globals()}
    _shared_stages.install_runtime(runtime)
    _hole_execution.install_runtime(runtime)


@dataclass
class SequentialWorkflowContext:
    """Shared data contract between sequential workflow stages.

    Resource references are stable for one run. Mutable fields such as the
    current TCP and per-hole dictionaries are deliberately written back by each
    stage so the existing report/cache mutation semantics remain unchanged.
    """

    args: Any
    handeye: Any
    model: Any
    cfg: Any
    run_dir: Any
    report: Any
    timing: Any
    rows: Any
    runtime: Any
    pose_session: Any
    motion_session: Any
    current_tcp: Any
    initial_holes: Any
    initial_intrinsics: Any
    fixed_rz_rad: float
    results: Any
    order_ids: list[Any]
    initial_pointcloud_reused_holes: list[Any]
    all_selected_two_capture_mode: bool
    cache_entries: Any
    cache_sources: Any
    cache_source_ids: Any
    cache_gates: Any
    cache_enabled: bool
    persistent_enabled: bool
    persistent_entries: Any
    coarse_cache_dir: Any
    persistent_cache_dir: Any
    cache_built_ids: set[int]
    batch_coarse_results: dict[Any, Any]
    batch_coarse_for_cache: bool
    shared_cache_results: dict[Any, Any]
    shared_cache_failed_ids: set[int]
    invalidated_cache_ids: set[int]
    batch_fine_results: dict[Any, Any]
    batch_fine_plan: dict[Any, Any]

    def ensure_rgbd_pipeline(self) -> tuple[Any, Any, Any]:
        if self.runtime.get("rgbd_pipeline") is None:
            with self.timing.measure("camera/restart_rgbd_pipeline"):
                pipeline, align, chain = init_pipeline()
            self.runtime["rgbd_pipeline"] = pipeline
            self.runtime["align"] = align
            self.runtime["chain"] = chain
        return (
            self.runtime["rgbd_pipeline"],
            self.runtime["align"],
            self.runtime["chain"],
        )


def run_preview_stage(ctx: SequentialWorkflowContext) -> int | None:
    return _shared_stages.run_preview_stage(ctx)


def run_shared_coarse_stage(ctx: SequentialWorkflowContext) -> None:
    return _shared_stages.run_shared_coarse_stage(ctx)


def run_shared_cache_validation_stage(ctx: SequentialWorkflowContext) -> None:
    return _shared_stages.run_shared_cache_validation_stage(ctx)


def run_shared_fine_stage(ctx: SequentialWorkflowContext) -> None:
    return _shared_stages.run_shared_fine_stage(ctx)


def optimize_hole_order_stage(ctx: SequentialWorkflowContext) -> None:
    return _shared_stages.optimize_hole_order_stage(ctx)


def run_per_hole_stage(ctx: SequentialWorkflowContext) -> None:
    return _hole_execution.run_per_hole_stage(ctx)
