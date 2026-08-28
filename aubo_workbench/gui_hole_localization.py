#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""工作台中的两阶段孔定位和诊断工具页面。

定位计算继续复用两个独立诊断入口，但都以后台子进程运行：
Tk 主线程不会被相机、YOLO 或机器人运动等待阻塞。默认仅在开始检测下一个已选孔时暂停，
由本页面的“开始检测下一个孔”按钮发送继续指令；也可以启用自动连续检测。
"""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from .paths import HANDEYE_CANDIDATE_PATH, HOLE_LOCALIZATION_RUNS_DIR, MODEL_PATH

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOCALIZATION_SCRIPT = PROJECT_DIR / "run_yolo_eye_in_hand_optimized.py"
OFFSET_TEST_SCRIPT = PROJECT_DIR / "run_coarse_to_fine_offset_test.py"
DEFAULT_MODEL = MODEL_PATH
DEFAULT_HANDEYE = HANDEYE_CANDIDATE_PATH
RUNS_DIR = HOLE_LOCALIZATION_RUNS_DIR


class ScrollableTab(ttk.Frame):
    """带常驻纵向滚动条的 Notebook 页面容器。

    孔洞定位页的参数较多，在较小窗口中如果直接放进 Notebook，底部控件会
    被裁掉且用户不容易察觉。这个容器让滚动条始终位于页面右侧；鼠标放在
    页面内容上时也可以直接使用滚轮上下查看。
    """

    def __init__(self, master: tk.Misc, *, padding: int = 0) -> None:
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        frame_background = ttk.Style(self).lookup("TFrame", "-background")
        if frame_background:
            self.canvas.configure(background=frame_background)
        self.scrollbar = ttk.Scrollbar(
            self, orient=tk.VERTICAL, command=self.canvas.yview,
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        # 滚动条常驻显示，即使内容暂时没有超出窗口，也能明确提示页面可滚动。
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.inner = ttk.Frame(self.canvas, padding=padding)
        self.inner.columnconfigure(0, weight=1)
        self._window_id = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw",
        )
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_inner_configure(self, _event: tk.Event) -> None:
        scrollregion = self.canvas.bbox("all")
        if scrollregion is not None:
            self.canvas.configure(scrollregion=scrollregion)

    def _on_canvas_configure(self, event: tk.Event) -> None:
        # 内容区至少与可视宽度一致，避免出现无意义的横向空白。
        self.canvas.itemconfigure(self._window_id, width=max(1, int(event.width)))

    def bind_mousewheel(self) -> None:
        """把滚轮绑定到当前页内的所有控件，Entry/Button 上也能滚动。"""
        self._bind_mousewheel_recursive(self.inner)

    def _bind_mousewheel_recursive(self, widget: tk.Misc) -> None:
        # 运行日志已有独立的文本滚动条，不让页面滚动绑定抢走它的滚轮焦点。
        if isinstance(widget, tk.Text):
            return
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(sequence, self._on_mousewheel, add="+")
        for child in widget.winfo_children():
            # 日志文本和它自己的滚动条保留原生滚动行为，避免两个滚动条抢焦点。
            if child is self.scrollbar:
                continue
            self._bind_mousewheel_recursive(child)

    def _on_mousewheel(self, event: tk.Event) -> str:
        if getattr(event, "num", None) == 4:
            units = -3
        elif getattr(event, "num", None) == 5:
            units = 3
        else:
            delta = int(getattr(event, "delta", 0) or 0)
            units = -max(1, abs(delta) // 120) if delta > 0 else max(1, abs(delta) // 120)
        self.canvas.yview_scroll(units, "units")
        return "break"


class HoleLocalizationPanel(ttk.Frame):
    """两阶段 YOLO 孔定位 GUI，以及独立的偏移诊断工具。"""

    def __init__(self, master: tk.Misc, connection_provider: Callable[[], dict[str, Any]]) -> None:
        super().__init__(master, padding=10)
        self.connection_provider = connection_provider
        self.log_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.process_mode = "two_stage"
        self.run_started_at = 0.0
        self.waiting_confirmation = False
        self.confirmation_kind = ""
        self._poll_id: str | None = None

        self.model_var = tk.StringVar(value=str(DEFAULT_MODEL))
        self.handeye_var = tk.StringVar(value=str(DEFAULT_HANDEYE))
        # 三种粗定位策略在界面上使用互斥单选框。旧的 BooleanVar 仍保留为
        # 命令行兼容层，避免外部脚本或历史测试直接构造面板时失效。
        # 保持原 GUI 的默认行为：优先复用已验证的粗定位缓存。
        self.strategy_var = tk.StringVar(value="batch")
        self.advanced_visible_var = tk.BooleanVar(value=False)
        self.confidence_var = tk.StringVar(value="0.35")
        self.coarse_height_var = tk.StringVar(value="340")
        self.fine_height_var = tk.StringVar(value="260")
        self.coarse_frames_var = tk.StringVar(value="15")
        self.fine_frames_var = tk.StringVar(value="30")
        # 初始孔数由相机窗口中的点击数量决定，按 Enter 结束选择。
        self.speed_var = tk.StringVar(value="0.08")
        self.acc_var = tk.StringVar(value="0.25")
        self.transit_speed_var = tk.StringVar(value="0.15")
        self.transit_acc_var = tk.StringVar(value="0.45")
        self.approach_speed_var = tk.StringVar(value="0.12")
        self.approach_acc_var = tk.StringVar(value="0.35")
        self.offset_radii_var = tk.StringVar(value="0 5 10 15 20")
        self.offset_angles_var = tk.StringVar(value="0 45 90 135 180 225 270 315")
        self.batch_coarse_localization_var = tk.BooleanVar(value=True)
        self.batch_coarse_frames_var = tk.StringVar(value="15")
        self.batch_coarse_min_valid_var = tk.StringVar(value="10")
        self.batch_coarse_min_holes_var = tk.StringVar(value="")
        self.batch_coarse_view_margin_var = tk.StringVar(value="50.0")
        self.batch_fine_localization_var = tk.BooleanVar(value=True)
        self.batch_fine_joint_localization_var = tk.BooleanVar(value=True)
        self.batch_fine_frames_var = tk.StringVar(value="8")
        self.batch_fine_min_valid_var = tk.StringVar(value="5")
        self.batch_fine_stable_min_frames_var = tk.StringVar(value="5")
        self.batch_fine_settle_discard_frames_var = tk.StringVar(value="10")
        self.batch_fine_supplement_rounds_var = tk.StringVar(value="1")
        self.batch_fine_view_margin_var = tk.StringVar(value="50.0")
        self.optimize_hole_order_var = tk.BooleanVar(value=False)
        self.execute_var = tk.BooleanVar(value=False)
        self.experimental_var = tk.BooleanVar(value=False)
        self.final_target_mode_var = tk.StringVar(value="机械爪模式")
        self.reuse_coarse_cache_var = tk.BooleanVar(value=True)
        self.reuse_persistent_coarse_cache_var = tk.BooleanVar(value=True)
        self.shared_cache_validation_var = tk.BooleanVar(value=False)
        self.shared_cache_validation_frames_var = tk.StringVar(value="3")
        self.shared_cache_validation_min_valid_var = tk.StringVar(value="2")
        self.shared_cache_validation_view_margin_var = tk.StringVar(value="50.0")
        self.auto_next_hole_var = tk.BooleanVar(value=False)
        self.final_xy_var = tk.BooleanVar(value=False)
        self.include_final_motion_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="待开始：默认仅预览，不会下发机器人运动")
        self.motion_var = tk.StringVar(value="无待确认运动")
        self.result_var = tk.StringVar(value="尚无本次结果")

        self._build()
        self.strategy_var.trace_add("write", self._on_strategy_changed)
        self._apply_strategy()
        for variable in (
            self.reuse_coarse_cache_var, self.reuse_persistent_coarse_cache_var,
            self.shared_cache_validation_var, self.batch_coarse_localization_var,
        ):
            variable.trace_add("write", self._update_legacy_strategy_controls)
        self._update_legacy_strategy_controls()
        self._poll_queue()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", padx=2, pady=(0, 6))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="孔洞定位工作台", font=("TkDefaultFont", 11, "bold")).grid(
            row=0, column=0, sticky="w",
        )
        ttk.Label(header, textvariable=self.status_var, foreground="#555555").grid(
            row=0, column=1, sticky="e",
        )
        ttk.Label(
            header, text="↕ 右侧滚动条 / 鼠标滚轮查看全部内容", foreground="#777777",
        ).grid(row=0, column=2, sticky="e", padx=(14, 0))

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        detection_page = ScrollableTab(self.notebook, padding=10)
        diagnostics_page = ScrollableTab(self.notebook, padding=10)
        monitor_page = ScrollableTab(self.notebook, padding=10)
        self.notebook.add(detection_page, text="孔洞检测")
        self.notebook.add(diagnostics_page, text="诊断工具")
        self.notebook.add(monitor_page, text="运行监控")

        self._build_detection_tab(detection_page.inner)
        self._build_diagnostics_tab(diagnostics_page.inner)
        self._build_monitor_tab(monitor_page.inner)
        for page in (detection_page, diagnostics_page, monitor_page):
            page.bind_mousewheel()

        # 这组按钮始终可见，避免用户在相机窗口等待时找不到确认/停止操作。
        control = ttk.LabelFrame(self, text="流程控制", padding=6)
        control.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        control.columnconfigure(6, weight=1)
        self.confirm_btn = ttk.Button(
            control, text="开始检测下一个孔", command=self.confirm_motion, state=tk.DISABLED,
        )
        self.confirm_btn.grid(row=0, column=0, padx=(0, 6))
        self.mark_error_btn = ttk.Button(
            control, text="标记当前点有误差并继续",
            command=self.mark_current_point_error, state=tk.DISABLED,
        )
        self.mark_error_btn.grid(row=0, column=1, padx=(0, 6))
        self.cancel_btn = ttk.Button(
            control, text="停止流程", command=self.cancel_pending_motion, state=tk.DISABLED,
        )
        self.cancel_btn.grid(row=0, column=2, padx=(0, 12))
        ttk.Button(control, text="打开结果目录", command=self.open_results_dir).grid(row=0, column=3, padx=(0, 6))
        ttk.Label(control, textvariable=self.motion_var, foreground="#a35a00").grid(
            row=1, column=0, columnspan=7, sticky="w", pady=(5, 0),
        )

    def _build_detection_tab(self, parent: ttk.Frame) -> None:
        files = ttk.LabelFrame(parent, text="检测输入", padding=8)
        files.pack(fill=tk.X)
        files.columnconfigure(1, weight=1)
        self._path_row(files, 0, "YOLO 模型", self.model_var, self._browse_model)
        self._path_row(files, 1, "手眼结果", self.handeye_var, self._browse_handeye)

        two_stage = ttk.LabelFrame(parent, text="基础检测参数", padding=8)
        two_stage.pack(fill=tk.X, pady=(8, 0))
        self._grid_fields(two_stage, [
            ("置信度", self.confidence_var, 7),
            ("粗定位高度 mm", self.coarse_height_var, 8),
            ("精定位高度 mm", self.fine_height_var, 8),
            ("粗定位帧", self.coarse_frames_var, 6),
            ("精定位帧", self.fine_frames_var, 6),
            ("精确速度 m/s", self.speed_var, 7),
            ("加速度 m/s²", self.acc_var, 7),
            ("安全过渡速度 m/s", self.transit_speed_var, 7),
            ("安全过渡加速度 m/s²", self.transit_acc_var, 7),
            ("非接触接近速度 m/s", self.approach_speed_var, 7),
            ("非接触接近加速度 m/s²", self.approach_acc_var, 7),
        ], columns=3)

        strategy = ttk.LabelFrame(parent, text="检测策略（互斥）", padding=8)
        strategy.pack(fill=tk.X, pady=(8, 0))
        for row, (label, value) in enumerate((
            ("逐孔检测：每个孔独立粗定位和精定位", "per_hole"),
            ("全部孔共享两次稳定连拍：340 mm 粗定位，260 mm 精定位", "batch"),
            ("复用粗定位缓存：一拍多验证，失败逐孔粗定位", "cache"),
        )):
            ttk.Radiobutton(
                strategy, text=label, variable=self.strategy_var, value=value,
                command=self._apply_strategy,
            ).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Label(
            strategy,
            text="共享精拍首拍信息不足时，会移动到失败孔共同观察位补拍；不转成逐孔精拍。",
            foreground="#555555",
        ).grid(row=3, column=0, sticky="w", pady=(4, 0))

        batch = ttk.LabelFrame(parent, text="高级：批量粗定位参数", padding=8)
        batch.pack(fill=tk.X, pady=(8, 0))
        self._grid_fields(batch, [
            ("批量采集帧数", self.batch_coarse_frames_var, 7),
            ("批量最少有效帧", self.batch_coarse_min_valid_var, 7),
            ("每帧最少孔数", self.batch_coarse_min_holes_var, 7),
            ("视野边缘余量 px", self.batch_coarse_view_margin_var, 7),
        ], columns=2)
        ttk.Checkbutton(
            batch,
            text="260 mm 共享精定位（首拍全部选中孔；失败孔可移动共同位补拍）",
            variable=self.batch_fine_localization_var,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 2))
        ttk.Checkbutton(
            batch,
            text="260 mm 多孔联合精定位（共享XY；保留逐孔残差、倾斜纠偏和ChArUco补偿）",
            variable=self.batch_fine_joint_localization_var,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(2, 2))
        ttk.Label(batch, text="260 mm 精定位视野边缘余量 px").grid(
            row=4, column=0, sticky="w", padx=(0, 4), pady=4,
        )
        ttk.Entry(batch, textvariable=self.batch_fine_view_margin_var, width=7).grid(
            row=4, column=1, sticky="w", padx=(0, 18), pady=4,
        )
        self._grid_fields(batch, [
            ("260 mm 批量帧数", self.batch_fine_frames_var, 7),
            ("260 mm 最少有效帧", self.batch_fine_min_valid_var, 7),
            ("260 mm 稳定门帧数", self.batch_fine_stable_min_frames_var, 7),
            ("260 mm 最少预热丢弃帧", self.batch_fine_settle_discard_frames_var, 7),
            ("260 mm 失败孔共享补拍轮数", self.batch_fine_supplement_rounds_var, 7),
        ], columns=2, start_row=5)

        shared = ttk.LabelFrame(parent, text="高级：缓存验证策略", padding=8)
        self.shared_cache_validation_frame = shared
        shared.pack(fill=tk.X, pady=(8, 0))
        self.shared_cache_validation_check = ttk.Checkbutton(
            shared,
            text="共享 340 mm 一拍多缓存验证（按视野自动分组；通过孔直接到 260 mm）",
            variable=self.shared_cache_validation_var,
        )
        self.shared_cache_validation_check.grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 4))
        ttk.Label(
            shared,
            text="复用缓存策略默认执行此验证；失败孔自动进入逐孔完整粗定位，逐孔结果不覆盖一拍多缓存。",
            foreground="#555555",
        ).grid(row=1, column=0, columnspan=6, sticky="w", pady=(0, 6))
        self._grid_fields(shared, [
            ("每组验证帧数", self.shared_cache_validation_frames_var, 7),
            ("每孔最少有效帧", self.shared_cache_validation_min_valid_var, 7),
            ("视野边缘余量 px", self.shared_cache_validation_view_margin_var, 7),
        ], columns=3, start_row=2)

        options = ttk.LabelFrame(parent, text="高级选项", padding=8)
        options.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(options, text="最终点模式").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Combobox(
            options,
            textvariable=self.final_target_mode_var,
            values=("机械爪模式", "平常模式"),
            state="readonly",
            width=13,
        ).grid(row=0, column=1, sticky="w", padx=(0, 18))
        ttk.Checkbutton(options, text="真实运动", variable=self.execute_var).grid(row=0, column=2, sticky="w", padx=(0, 18))
        ttk.Checkbutton(options, text="允许当前实验手眼结果", variable=self.experimental_var).grid(row=0, column=3, sticky="w")
        ttk.Checkbutton(
            options, text="复用本次一拍多批量粗定位缓存", variable=self.reuse_coarse_cache_var,
        ).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(
            options, text="复用历史一拍多批量粗定位缓存", variable=self.reuse_persistent_coarse_cache_var,
        ).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(options, text="精定位后沿工具 X/Y/Z 方向执行最终移动（慢速）", variable=self.final_xy_var).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(6, 0),
        )
        ttk.Checkbutton(
            options, text="按最近邻优化孔序（工艺允许时启用）",
            variable=self.optimize_hole_order_var,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            options,
            text="当前孔完成后自动检测下一孔（连续运动）",
            variable=self.auto_next_hole_var,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(
            options,
            text="仅作用于本轮已选孔；开启后孔间不再等待人工确认，请确认运行区域安全。",
            foreground="#a35a00",
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(2, 0))
        actions = ttk.LabelFrame(parent, text="开始检测", padding=8)
        actions.pack(fill=tk.X, pady=(8, 0))
        self.start_btn = ttk.Button(actions, text="开始两阶段定位", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(
            actions, text="偏移测试仅用于标定和诊断，不影响正式孔洞检测。", foreground="#555555",
        ).pack(side=tk.LEFT, padx=(16, 0))

    def _build_diagnostics_tab(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="偏移容忍度测试", padding=8)
        panel.pack(fill=tk.X)
        ttk.Label(panel, text="用于单孔偏移采样和标定诊断；不会改变正式孔洞检测策略。",
                  foreground="#555555").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        self._grid_fields(panel, [
            ("偏移测试半径 mm", self.offset_radii_var, 24),
            ("偏移测试方向 °", self.offset_angles_var, 38),
        ], columns=2, start_row=1)
        ttk.Checkbutton(
            panel, text="偏移测试执行最终 XY → Z → +Y 0.3 mm（实机）",
            variable=self.include_final_motion_var,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.offset_start_btn = ttk.Button(panel, text="开始偏移容忍度测试", command=self.start_offset_test)
        self.offset_start_btn.grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )

    def _apply_strategy(self) -> None:
        """把用户选择的互斥策略映射到旧命令行兼容开关。"""
        strategy = self.strategy_var.get()
        self.batch_coarse_localization_var.set(strategy == "batch")
        self.reuse_coarse_cache_var.set(strategy == "cache")
        self.reuse_persistent_coarse_cache_var.set(strategy == "cache")
        self.shared_cache_validation_var.set(strategy == "cache")

    def _on_strategy_changed(self, *_args: Any) -> None:
        self._apply_strategy()

    def _build_monitor_tab(self, parent: ttk.Frame) -> None:
        notice = ttk.LabelFrame(parent, text="操作提示", padding=8)
        notice.pack(fill=tk.X)
        ttk.Label(
            notice,
            text=("两阶段流程：相机初始画面中选择孔后按 Enter；默认每个孔完成后点击继续，"
                  "也可在高级选项启用自动检测下一孔。"
                  "本轮全部完成后机械臂先保持当前位置；点击“开始下一轮检测”后才回到初始点，确认完全停止后再采集第二轮画面。"
                  "如需急停，请使用机械臂示教器。"),
            justify=tk.LEFT,
            wraplength=1050,
        ).pack(anchor="w")

        result = ttk.LabelFrame(parent, text="最终结果", padding=8)
        result.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(result, textvariable=self.result_var, justify=tk.LEFT, wraplength=1050).pack(anchor="w")

        logs = ttk.LabelFrame(parent, text="运行日志", padding=6)
        logs.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.log_text = tk.Text(logs, height=18, wrap="word", state=tk.DISABLED)
        scroll = ttk.Scrollbar(logs, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    @staticmethod
    def _grid_fields(
        parent: ttk.Widget,
        fields: list[tuple[str, tk.StringVar, int]],
        columns: int = 3,
        start_row: int = 0,
    ) -> None:
        for index, (label, variable, width) in enumerate(fields):
            row = start_row + index // columns
            slot = (index % columns) * 2
            ttk.Label(parent, text=label).grid(row=row, column=slot, sticky="w", padx=(0, 4), pady=4)
            ttk.Entry(parent, textvariable=variable, width=width).grid(
                row=row, column=slot + 1, sticky="w", padx=(0, 18), pady=4,
            )

    def _path_row(self, parent: ttk.LabelFrame, row: int, label: str, variable: tk.StringVar,
                  callback: Callable[[], None]) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, columnspan=3, sticky="ew", pady=3)
        ttk.Button(parent, text="选择", command=callback).grid(row=row, column=4, sticky="w", padx=(8, 0), pady=3)

    def _update_legacy_strategy_controls(self, *_args: Any) -> None:
        """Keep the mutually-exclusive 340 mm strategies legible in the GUI."""
        conflict = bool(self.batch_coarse_localization_var.get())
        cache_enabled = bool(self.reuse_coarse_cache_var.get())
        allowed = cache_enabled and not conflict
        if not allowed and self.shared_cache_validation_var.get():
            self.shared_cache_validation_var.set(False)
        state = tk.NORMAL if allowed else tk.DISABLED
        for widget in self.shared_cache_validation_frame.winfo_children():
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="选择 YOLO 模型", filetypes=[("模型", "*.pt"), ("所有文件", "*.*")])
        if path:
            self.model_var.set(path)

    def _browse_handeye(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="选择手眼结果", filetypes=[("JSON", "*.json"), ("所有文件", "*.*")])
        if path:
            self.handeye_var.set(path)

    def _numbers(self) -> dict[str, float | int]:
        try:
            numbers: dict[str, float | int] = {
                "confidence": float(self.confidence_var.get()),
                "coarse_height": float(self.coarse_height_var.get()),
                "fine_height": float(self.fine_height_var.get()),
                "coarse_frames": int(self.coarse_frames_var.get()),
                "fine_frames": int(self.fine_frames_var.get()),
                "speed": float(self.speed_var.get()),
                "acc": float(self.acc_var.get()),
                "transit_speed": float(self.transit_speed_var.get()),
                "transit_acc": float(self.transit_acc_var.get()),
                "approach_speed": float(self.approach_speed_var.get()),
                "approach_acc": float(self.approach_acc_var.get()),
            }
            return numbers
        except ValueError as exc:
            raise ValueError("定位参数必须是有效数字") from exc

    @staticmethod
    def _float_list(value: str, label: str) -> list[float]:
        try:
            values = [float(item) for item in value.replace(",", " ").split()]
        except ValueError as exc:
            raise ValueError(f"{label}必须是空格或逗号分隔的数字") from exc
        if not values:
            raise ValueError(f"{label}不能为空")
        return values

    def _build_command(self, mode: str = "two_stage", execute_override: bool | None = None) -> list[str]:
        is_offset = mode == "offset"
        script = LOCALIZATION_SCRIPT if mode == "two_stage" else OFFSET_TEST_SCRIPT
        if not script.is_file():
            raise FileNotFoundError(f"找不到定位脚本：{script}")
        model = Path(self.model_var.get().strip())
        handeye = Path(self.handeye_var.get().strip())
        execute = self.execute_var.get() if execute_override is None else bool(execute_override)
        if (not is_offset or execute) and not model.is_file():
            raise FileNotFoundError(f"YOLO 模型不存在：{model}")
        if (not is_offset or execute) and not handeye.is_file():
            raise FileNotFoundError(f"手眼结果不存在：{handeye}")
        values = self._numbers()
        connection = self.connection_provider()
        command = [
            # 无缓冲输出，确保 GUI 能在机器人开始运动前看到确认事件。
            sys.executable, "-u", str(script),
            "--model", str(model), "--handeye", str(handeye),
            "--confidence", str(values["confidence"]),
            "--coarse-height-mm", str(values["coarse_height"]),
            "--fine-height-mm", str(values["fine_height"]),
            "--coarse-frames", str(values["coarse_frames"]),
            "--fine-frames", str(values["fine_frames"]),
            "--speed-m-s", str(values["speed"]), "--acc-m-s2", str(values["acc"]),
            "--transit-speed-m-s", str(values["transit_speed"]),
            "--transit-acc-m-s2", str(values["transit_acc"]),
            "--approach-speed-m-s", str(values["approach_speed"]),
            "--approach-acc-m-s2", str(values["approach_acc"]),
            "--robot-ip", str(connection["ip"]), "--robot-port", str(connection["port"]),
            "--robot-user", str(connection["user"]), "--robot-password", str(connection["password"]),
            "--robot-timeout-ms", str(connection["timeout_ms"]),
        ]
        command.append("--execute" if execute else "--no-execute")
        if is_offset:
            command.append(
                "--allow-experimental-handeye"
                if self.experimental_var.get() else "--require-validated-handeye"
            )
            if self.include_final_motion_var.get() and not execute:
                raise ValueError("偏移测试的最终XY/Z/Y动作必须勾选“真实运动”")
            command.extend([
                "--radii-mm",
                *[str(value) for value in self._float_list(self.offset_radii_var.get(), "偏移测试半径")],
                "--angles-deg",
                *[str(value) for value in self._float_list(self.offset_angles_var.get(), "偏移测试方向")],
            ])
            if execute:
                command.append("--start-confirmed")
            if self.include_final_motion_var.get():
                command.append("--include-final-motion")
        else:  # two_stage
            command.append(
                "--allow-experimental-handeye"
                if self.experimental_var.get() else "--require-validated-handeye"
            )
            final_mode = {
                "机械爪模式": "gripper",
                "平常模式": "normal",
            }.get(self.final_target_mode_var.get())
            if final_mode is None:
                raise ValueError("最终点模式必须选择“机械爪模式”或“平常模式”")
            command.extend(["--final-target-mode", final_mode])
            command.append("--move-final-xy" if self.final_xy_var.get() else "--no-move-final-xy")
            # “复用粗定位缓存”策略的语义是跨运行复用一拍多正式缓存。
            # 即使高级兼容开关被旧配置残留为 False，也不能悄悄退化成
            # 仅当前运行缓存，否则下一轮会再次执行一拍多。
            strategy_name = getattr(getattr(self, "strategy_var", None), "get", lambda: "")()
            cache_strategy = strategy_name == "cache"
            reuse_session_cache = bool(self.reuse_coarse_cache_var.get()) or cache_strategy
            reuse_persistent_cache = (
                bool(self.reuse_persistent_coarse_cache_var.get()) or cache_strategy
            )
            command.append(
                "--reuse-coarse-cache"
                if reuse_session_cache else "--no-reuse-coarse-cache"
            )
            command.append(
                "--reuse-persistent-coarse-cache"
                if reuse_persistent_cache else "--no-reuse-persistent-coarse-cache"
            )
            shared_cache_var = getattr(self, "shared_cache_validation_var", None)
            shared_cache_enabled = bool(shared_cache_var is not None and shared_cache_var.get())
            if shared_cache_enabled:
                if not self.reuse_coarse_cache_var.get():
                    raise ValueError("共享快速缓存验证需要启用“复用本次初始多孔局部点云缓存”")
                try:
                    shared_frames = int(getattr(self, "shared_cache_validation_frames_var").get())
                    shared_min_valid = int(getattr(self, "shared_cache_validation_min_valid_var").get())
                    shared_margin = float(getattr(self, "shared_cache_validation_view_margin_var").get())
                except ValueError as exc:
                    raise ValueError("共享快速缓存验证参数必须是有效数字") from exc
                if shared_frames < 1 or shared_min_valid < 1 or shared_min_valid > shared_frames:
                    raise ValueError("共享验证最少有效帧数必须在验证帧数范围内")
                if not math.isfinite(shared_margin) or shared_margin < 0.0:
                    raise ValueError("共享验证视野边缘余量必须是大于等于 0 的有限数字")
                command.extend([
                    "--shared-cache-validation",
                    "--shared-cache-validation-frames", str(shared_frames),
                    "--shared-cache-validation-min-valid", str(shared_min_valid),
                    "--shared-cache-validation-view-margin-px", str(shared_margin),
                ])
            batch_mode_var = getattr(self, "batch_coarse_localization_var", None)
            if batch_mode_var is not None and batch_mode_var.get():
                if shared_cache_enabled:
                    raise ValueError("共享快速缓存验证不能与批量粗定位同时启用")
                try:
                    batch_frames = int(self.batch_coarse_frames_var.get())
                    batch_min_valid = int(self.batch_coarse_min_valid_var.get())
                    batch_view_margin = float(self.batch_coarse_view_margin_var.get())
                    batch_min_holes_text = self.batch_coarse_min_holes_var.get().strip()
                    batch_min_holes = int(batch_min_holes_text) if batch_min_holes_text else None
                except ValueError as exc:
                    raise ValueError("批量粗定位参数必须是有效数字") from exc
                if batch_frames <= 0 or batch_min_valid <= 0:
                    raise ValueError("批量采集帧数和最少有效帧必须大于 0")
                if batch_frames < batch_min_valid:
                    raise ValueError("批量采集帧数必须不少于批量最少有效帧")
                if batch_min_holes is not None and batch_min_holes <= 0:
                    raise ValueError("每帧最少孔数必须大于 0")
                if not math.isfinite(batch_view_margin) or batch_view_margin < 0.0:
                    raise ValueError("视野边缘余量必须是大于等于 0 的有限数字")
                command.append("--batch-coarse-localization")
                command.extend([
                    "--batch-coarse-frames", str(batch_frames),
                    "--batch-coarse-min-valid", str(batch_min_valid),
                    "--batch-coarse-view-margin-px", str(batch_view_margin),
                ])
                if batch_min_holes is not None:
                    command.extend(["--batch-coarse-min-holes-per-frame", str(batch_min_holes)])
                optimize_hole_order_var = getattr(self, "optimize_hole_order_var", None)
                if optimize_hole_order_var is not None and optimize_hole_order_var.get():
                    command.append("--optimize-hole-order")
            fine_batch_var = getattr(self, "batch_fine_localization_var", None)
            fine_batch_enabled = bool(
                fine_batch_var is None or fine_batch_var.get()
            )
            try:
                fine_batch_margin = float(
                    getattr(
                        getattr(self, "batch_fine_view_margin_var", None),
                        "get",
                        lambda: "50.0",
                    )()
                )
            except ValueError as exc:
                raise ValueError("批量精定位视野边缘余量必须是有效数字") from exc
            if not math.isfinite(fine_batch_margin) or fine_batch_margin < 0.0:
                raise ValueError("批量精定位视野边缘余量必须是大于等于 0 的有限数字")
            try:
                fine_batch_frames = int(getattr(
                    getattr(self, "batch_fine_frames_var", None),
                    "get", lambda: "1",
                )())
                fine_batch_min_valid = int(getattr(
                    getattr(self, "batch_fine_min_valid_var", None),
                    "get", lambda: "1",
                )())
                fine_batch_stable_min = int(getattr(
                    getattr(self, "batch_fine_stable_min_frames_var", None),
                    "get", lambda: "1",
                )())
                fine_batch_settle = int(getattr(
                    getattr(self, "batch_fine_settle_discard_frames_var", None),
                    "get", lambda: "10",
                )())
                fine_batch_supplement_rounds = int(getattr(
                    getattr(self, "batch_fine_supplement_rounds_var", None),
                    "get", lambda: "1",
                )())
            except ValueError as exc:
                raise ValueError("批量精定位帧数参数必须是有效整数") from exc
            if fine_batch_frames < 1 or fine_batch_min_valid < 1:
                raise ValueError("批量精定位帧数和最少有效帧数必须大于 0")
            if fine_batch_frames < fine_batch_min_valid:
                raise ValueError("批量精定位帧数必须不少于最少有效帧数")
            if fine_batch_stable_min < 1 or fine_batch_stable_min > fine_batch_frames:
                raise ValueError("批量精定位稳定门帧数必须在批量帧数范围内")
            if fine_batch_settle < 0:
                raise ValueError("批量精定位预热丢弃帧数不能小于 0")
            if fine_batch_supplement_rounds < 0:
                raise ValueError("批量精定位共享补拍轮数不能小于 0")
            command.append(
                "--batch-fine-localization"
                if fine_batch_enabled else "--no-batch-fine-localization"
            )
            joint_batch_var = getattr(
                self, "batch_fine_joint_localization_var", None,
            )
            joint_batch_enabled = bool(
                joint_batch_var is None or joint_batch_var.get()
            )
            command.extend([
                "--batch-fine-view-margin-px", str(fine_batch_margin),
                "--batch-fine-frames", str(fine_batch_frames),
                "--batch-fine-min-valid", str(fine_batch_min_valid),
                "--batch-fine-stable-min-frames", str(fine_batch_stable_min),
                "--batch-fine-settle-discard-frames", str(fine_batch_settle),
                "--batch-fine-supplement-rounds", str(fine_batch_supplement_rounds),
                "--batch-fine-joint-localization"
                if joint_batch_enabled else "--no-batch-fine-joint-localization",
            ])
        return command

    def start(self) -> None:
        self._start_process("two_stage")

    def start_offset_test(self) -> None:
        self._start_process("offset")

    def _start_process(self, mode: str, execute_override: bool | None = None) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("定位进行中", "当前定位流程尚未结束。", parent=self)
            return
        try:
            command = self._build_command(mode, execute_override=execute_override)
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc), parent=self)
            return
        execute = self.execute_var.get() if execute_override is None else bool(execute_override)
        offset_confirmation = (
            "将执行最终XY、降Z、基坐标Y+0.3 mm，并在每个采样点完成后安全回升并返回中心。"
            if self.include_final_motion_var.get() else
            "不会执行最终XY、最终Z或基坐标Y+0.3 mm动作。"
        )
        auto_next_hole = bool(
            getattr(getattr(self, "auto_next_hole_var", None), "get", lambda: False)()
        )
        if execute and not messagebox.askyesno(
            "确认真实运动",
            (
                "将执行回原点、单孔粗定位、下降和横向偏移采集。"
                f"{offset_confirmation}\n"
                "请在相机窗口中只选择一个孔，之后测试会自动运行。\n\n确认开始吗？"
                if mode == "offset" else
                "将执行回原点及两阶段定位；每轮完成后会自动回原点并进入下一轮选孔，"
                "相机和机器人会话保持运行。按选孔窗口 Esc 可结束会话。\n"
                + (
                    "已启用自动检测下一孔：当前轮已选孔之间将连续移动，不再逐孔等待人工确认；"
                    "请确认运行区域安全，并随时准备使用急停。\n"
                    if auto_next_hole else
                    "当前轮每个孔完成后需要人工确认，确认后才会移动到下一个孔。\n"
                )
                + "\n确认开始吗？"
            ),
            parent=self,
        ):
            return
        self.process_mode = mode
        self._clear_log()
        self.result_var.set(
            "偏移容忍度测试进行中…" if mode == "offset" else
            "本次定位进行中…"
        )
        self.status_var.set(
            "正在启动偏移测试进程…" if mode == "offset" else
            "正在启动定位进程…"
        )
        self.motion_var.set("无待确认运动")
        self.waiting_confirmation = False
        self.confirmation_kind = ""
        self.confirm_btn.configure(text="开始检测下一个孔")
        self.confirm_btn.configure(state=tk.DISABLED)
        self.mark_error_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.DISABLED)
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(PROJECT_DIR),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            self.status_var.set("启动失败")
            messagebox.showerror("无法启动定位", str(exc), parent=self)
            return
        self.run_started_at = time.time()
        self.start_btn.configure(state=tk.DISABLED)
        self.offset_start_btn.configure(state=tk.DISABLED)
        threading.Thread(target=self._read_process_output, args=(self.process,), daemon=True).start()

    def _read_process_output(self, process: subprocess.Popen[str]) -> None:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                self.log_queue.put(("log", line))
                if "[NEXT_HOLE_CONFIRM_REQUIRED]" in line:
                    self.log_queue.put(("next_hole_confirm", line.strip()))
                elif "[MOTION_CONFIRM_REQUIRED]" in line:
                    self.log_queue.put(("confirm", line.strip()))
                elif "[INITIAL_SELECTION_REQUIRED]" in line:
                    self.log_queue.put(("initial_selection", line.strip()))
                elif "[NEXT_CYCLE_CONFIRM_REQUIRED]" in line:
                    self.log_queue.put(("next_cycle_confirm", line.strip()))
                elif "[OFFSET_TEST_CONFIRM_REQUIRED]" in line:
                    self.log_queue.put(("offset_start_confirm", line.strip()))
                elif "[OFFSET_NEXT_CONFIRM_REQUIRED]" in line:
                    self.log_queue.put(("offset_next_confirm", line.strip()))
                elif "[OFFSET_TARGET_CONFIRM_REQUIRED]" in line:
                    self.log_queue.put(("offset_target_confirm", line.strip()))
            code = process.wait()
            self.log_queue.put(("finished", code))
        except Exception as exc:
            self.log_queue.put(("worker_error", str(exc)))

    def confirm_motion(self) -> None:
        if not self.waiting_confirmation or self.process is None or self.process.poll() is not None:
            return
        try:
            assert self.process.stdin is not None
            self.process.stdin.write("m\n")
            self.process.stdin.flush()
        except OSError as exc:
            messagebox.showerror("确认发送失败", str(exc), parent=self)
            return
        self.waiting_confirmation = False
        confirmation_kind = self.confirmation_kind
        self.confirmation_kind = ""
        self.confirm_btn.configure(state=tk.DISABLED, text="开始检测下一个孔")
        self.mark_error_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.DISABLED)
        if confirmation_kind == "offset_start":
            self.motion_var.set("已确认，开始偏移测试…")
        elif confirmation_kind == "offset_next":
            self.motion_var.set("已确认，开始下一个偏移点…")
        elif confirmation_kind == "offset_target":
            self.motion_var.set("已确认，回升并继续偏移测试…")
        elif confirmation_kind == "next_cycle":
            self.motion_var.set("已确认，正在回到初始点；到位并完全停止后才采集下一轮画面…")
        else:
            self.motion_var.set("已确认，开始检测下一个孔…")

    def _handle_next_hole_confirmation(self, payload: str) -> None:
        """处理同一轮孔间确认；自动模式只在此事件上发送一次 m。"""
        if self.process_mode != "two_stage":
            self._append_log(
                f"[GUI] 忽略非孔洞检测流程的下一孔确认事件：{payload}\n"
            )
            return
        self.waiting_confirmation = True
        self.confirmation_kind = "hole"
        self.mark_error_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.NORMAL)
        auto_next_hole = bool(
            getattr(getattr(self, "auto_next_hole_var", None), "get", lambda: False)()
        ) and self.process_mode == "two_stage"
        if auto_next_hole:
            self.confirm_btn.configure(state=tk.DISABLED, text="自动进入下一孔…")
            self.motion_var.set("当前孔已完成，正在自动进入下一孔…")
            self.status_var.set("当前孔完成，正在自动移动到下一孔…")
            self._append_log(
                "[GUI] 已启用自动检测下一孔，自动发送 m，当前轮将连续执行。\n"
            )
            # confirm_motion 会检查子进程仍在运行，并在成功发送后清除等待状态。
            self.confirm_motion()
            if self.process is not None and self.process.poll() is None:
                self.status_var.set("已自动进入下一孔，定位流程运行中…")
        else:
            self.confirm_btn.configure(state=tk.NORMAL, text="开始检测下一个孔")
            self.motion_var.set("等待开始下一个孔：请点击“开始检测下一个孔”继续。")

    def mark_current_point_error(self) -> None:
        if (
            not self.waiting_confirmation
            or self.confirmation_kind != "offset_target"
            or self.process is None
            or self.process.poll() is not None
        ):
            return
        try:
            assert self.process.stdin is not None
            self.process.stdin.write("e\n")
            self.process.stdin.flush()
        except OSError as exc:
            messagebox.showerror("误差标记发送失败", str(exc), parent=self)
            return
        self.waiting_confirmation = False
        self.confirmation_kind = ""
        self.confirm_btn.configure(state=tk.DISABLED, text="开始检测下一个孔")
        self.mark_error_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.DISABLED)
        self.motion_var.set("已标记当前点有误差，回升后继续偏移测试…")

    def cancel_pending_motion(self) -> None:
        if not self.waiting_confirmation or self.process is None or self.process.poll() is not None:
            return
        try:
            assert self.process.stdin is not None
            self.process.stdin.write("\n")
            self.process.stdin.flush()
        except OSError as exc:
            messagebox.showerror("取消发送失败", str(exc), parent=self)
            return
        self.waiting_confirmation = False
        self.confirmation_kind = ""
        self.confirm_btn.configure(state=tk.DISABLED)
        self.mark_error_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.DISABLED)
        self.confirm_btn.configure(text="开始检测下一个孔")
        self.motion_var.set("已取消待确认运动，流程将安全结束。")

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                    self.status_var.set(
                        "偏移容忍度测试运行中…"
                        if self.process_mode == "offset" else
                        "定位流程运行中…"
                    )
                elif kind == "initial_selection":
                    self.waiting_confirmation = False
                    self.confirmation_kind = ""
                    self.motion_var.set("机器人已在原点，请在相机窗口重新选择目标孔。")
                    self.status_var.set("已回到原点，等待选择目标孔…")
                elif kind == "next_cycle_confirm":
                    self.waiting_confirmation = True
                    self.confirmation_kind = "next_cycle"
                    self.confirm_btn.configure(state=tk.NORMAL, text="开始下一轮检测")
                    self.mark_error_btn.configure(state=tk.DISABLED)
                    self.cancel_btn.configure(state=tk.NORMAL)
                    self.motion_var.set(
                        "上一轮已完成；机器人保持当前位置，请点击“开始下一轮检测”后回到初始点。"
                    )
                    self.status_var.set("等待确认开始下一轮检测…")
                elif kind == "confirm":
                    self.waiting_confirmation = True
                    self.confirmation_kind = "hole"
                    self.confirm_btn.configure(state=tk.NORMAL)
                    self.mark_error_btn.configure(state=tk.DISABLED)
                    self.cancel_btn.configure(state=tk.NORMAL)
                    self.confirm_btn.configure(text="开始检测下一个孔")
                    self.motion_var.set("等待开始下一个孔：请点击“开始检测下一个孔”继续。")
                elif kind == "next_hole_confirm":
                    self._handle_next_hole_confirmation(str(payload))
                elif kind == "offset_start_confirm":
                    self.waiting_confirmation = True
                    self.confirmation_kind = "offset_start"
                    self.confirm_btn.configure(state=tk.NORMAL, text="开始偏移测试")
                    self.mark_error_btn.configure(state=tk.DISABLED)
                    self.cancel_btn.configure(state=tk.NORMAL)
                    self.motion_var.set("等待开始偏移测试：请点击“开始偏移测试”继续。")
                elif kind == "offset_next_confirm":
                    self.waiting_confirmation = True
                    self.confirmation_kind = "offset_next"
                    self.confirm_btn.configure(state=tk.NORMAL, text="开始下一个偏移点")
                    self.mark_error_btn.configure(state=tk.DISABLED)
                    self.cancel_btn.configure(state=tk.NORMAL)
                    self.motion_var.set("当前偏移点已完成：请点击“开始下一个偏移点”继续。")
                elif kind == "offset_target_confirm":
                    self.waiting_confirmation = True
                    self.confirmation_kind = "offset_target"
                    self.confirm_btn.configure(state=tk.NORMAL, text="确认并继续")
                    self.mark_error_btn.configure(state=tk.NORMAL)
                    self.cancel_btn.configure(state=tk.NORMAL)
                    self.motion_var.set("已到达最终目标点：请确认后回升并继续。")
                elif kind == "finished":
                    self._finished(int(payload))
                elif kind == "worker_error":
                    self._append_log(f"[GUI] 读取定位进程失败：{payload}\n")
        except queue.Empty:
            pass
        self._poll_id = self.after(80, self._poll_queue)

    def _finished(self, code: int) -> None:
        self.waiting_confirmation = False
        self.confirmation_kind = ""
        self.confirm_btn.configure(state=tk.DISABLED)
        self.mark_error_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.DISABLED)
        self.confirm_btn.configure(text="开始检测下一个孔")
        self.start_btn.configure(state=tk.NORMAL)
        self.offset_start_btn.configure(state=tk.NORMAL)
        self.process = None
        if code == 0:
            self.status_var.set(
                "偏移测试完成" if self.process_mode == "offset" else
                "定位完成"
            )
            self._show_latest_result()
        else:
            self.status_var.set(f"定位结束，退出码={code}")
            self.result_var.set("本次定位未通过质量门或被取消；请查看运行日志和结果目录。")

    def _show_latest_result(self) -> None:
        if self.process_mode == "offset":
            reports = sorted(
                (
                    path for path in RUNS_DIR.glob("coarse-to-fine-offset-*/report.json")
                    if path.stat().st_mtime >= self.run_started_at - 2.0
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if not reports:
                self.result_var.set("偏移测试完成，但未找到本次 JSON 报告。")
                return
            try:
                report = json.loads(reports[0].read_text(encoding="utf-8"))
                summary = report.get("summary", {})
                lines = [
                    f"报告：{reports[0].parent}\n"
                    f"采样点：{report.get('sample_count', len(report.get('plan', [])))}    "
                    f"最大支持半径：{summary.get('max_supported_radius_mm', '-')} mm    "
                    f"严格通过半径：{summary.get('max_strict_radius_mm', '-')} mm",
                    "绿色=当前精定位质量门通过；橙色=仅严格门未通过但降级通过；红色=失败",
                ]
                manual_error_ids = summary.get("manual_error_sample_ids", [])
                if manual_error_ids:
                    lines.append(
                        "人工标记有误差的点："
                        + ", ".join(str(item) for item in manual_error_ids)
                    )
                motion_summary = report.get("final_motion_summary", {})
                if motion_summary.get("enabled"):
                    step_labels = {
                        "final_xy": "最终XY",
                        "final_z": "最终Z",
                        "final_y_plus_0_3": "最终+Y 0.3 mm",
                        # 保留历史报告的显示兼容性。
                        "final_y_plus_0_2": "最终+Y 0.2 mm（历史报告）",
                    }
                    lines.append("最终动作到位误差（实际TCP−规划TCP）：")
                    for step_name, label in step_labels.items():
                        step = motion_summary.get("steps", {}).get(step_name, {})
                        max_error = step.get("max_translation_error_mm")
                        mean_error = step.get("mean_translation_error_mm")
                        if max_error is not None:
                            lines.append(
                                f"{label}：最大 {float(max_error):.3f} mm，"
                                f"平均 {float(mean_error):.3f} mm"
                            )
                self.result_var.set("\n".join(lines))
            except Exception as exc:
                self.result_var.set(f"读取偏移测试报告失败：{exc}")
            return
        reports = sorted(
            (path for path in RUNS_DIR.glob("two-stage-*/report.json") if path.stat().st_mtime >= self.run_started_at - 2.0),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not reports:
            self.result_var.set("流程完成，但未找到本次 JSON 报告。")
            return
        try:
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            final = report.get("final_result", {})
            holes = final.get("holes")
            if isinstance(holes, list) and holes:
                deferred_count = int(final.get("deferred_count", 0) or 0)
                completed_count = int(final.get("completed_count", len(holes) - deferred_count) or 0)
                if deferred_count:
                    self.status_var.set(
                        f"定位完成，但有 {deferred_count} 个孔延期，成功 {completed_count} 个"
                    )
                lines = [
                    f"报告：{reports[0].parent}",
            f"批量处理孔数：{final.get('hole_count', len(holes))}    "
                    f"成功：{completed_count}    延期：{deferred_count}    "
                    f"最终 TCP：{self._format_vector(final.get('final_tcp_pose_m_rad'))}",
                ]
                shared = (report.get("stages") or {}).get("shared_cache_validation") or {}
                if shared.get("requested"):
                    groups = shared.get("groups") or []
                    accepted = sum(len(item.get("accepted_holes") or []) for item in groups)
                    fallback = sum(len(item.get("fallback_holes") or []) for item in groups)
                    lines.append(
                        f"共享340mm缓存验证：组数={len(groups)}，直接260mm={accepted}，"
                        f"逐孔340mm回退={fallback}"
                    )
                    for group in groups:
                        lines.append(
                            f"  组{group.get('group_index', '-')}: 孔={group.get('hole_ids', [])} "
                            f"通过={group.get('accepted_holes', [])} 回退={group.get('fallback_holes', [])}"
                        )
                batch_fine = (report.get("stages") or {}).get("batch_fine_plan") or {}
                batch_coarse = (report.get("stages") or {}).get("batch_coarse_plan") or {}
                if batch_coarse.get("enabled"):
                    groups = batch_coarse.get("groups") or []
                    lines.append(
                        f"340mm共享粗定位：组数={len(groups)}，"
                        "全部选中孔共享一个综合中心拍摄位姿"
                    )
                    for group in groups:
                        lines.append(
                            f"  粗定位组{group.get('group_index', '-')}: "
                            f"孔={group.get('hole_ids', [])} "
                            f"综合中心={self._format_vector(group.get('combined_point_base_mm'))} "
                            f"共享拍摄位姿={self._format_vector(group.get('target_tcp_pose_m_rad'))}"
                        )
                if batch_fine.get("enabled"):
                    groups = batch_fine.get("groups") or []
                    lines.append(
                        f"260mm共享拍摄：组数={len(groups)}，每组一个相机位姿；"
                        "孔结果仅为基坐标三维点"
                    )
                    for group in groups:
                        lines.append(
                            f"  组{group.get('group_index', '-')}: "
                            f"孔={group.get('hole_ids', [])} "
                            f"综合中心={self._format_vector(group.get('combined_point_base_mm'))} "
                            f"共享拍摄位姿={self._format_vector(group.get('target_tcp_pose_m_rad'))}"
                        )
                        for supplement in group.get("supplement_captures") or []:
                            lines.append(
                                f"    共享补拍{supplement.get('round', '-')}: "
                                f"孔={supplement.get('hole_ids', [])} "
                                f"通过={supplement.get('accepted_holes', [])} "
                                f"未通过={supplement.get('fallback_holes', [])}"
                            )
                for item in holes:
                    quality_status = item.get("fine_quality_status", "strict")
                    status_text = (
                        f"状态={item.get('status', 'completed')}/{quality_status} "
                        if item.get("status") != "completed" or quality_status != "strict"
                        else ""
                    )
                    lines.append(
                        f"孔 {item.get('hole_id', '-')}：{status_text}"
                        f"中心={self._format_vector(item.get('hole_center_base_mm'))} "
                        f"孔径={item.get('matched_diameter_mm', '-') } mm "
                        f"跟踪={item.get('tracking_identity', '-')}"
                    )
                self.result_var.set("\n".join(lines))
                return
            center = final.get("hole_center_base_mm")
            normal = final.get("plane_normal_toward_camera_base")
            tcp = final.get("final_tcp_pose_m_rad")
            diameter = final.get("matched_diameter_mm")
            self.result_var.set(
                f"报告：{reports[0].parent}\n"
                f"孔中心（基坐标 mm）：{self._format_vector(center)}    法向：{self._format_vector(normal)}\n"
                f"最终 TCP（m/rad）：{self._format_vector(tcp)}    孔径类别：{diameter} mm"
            )
        except Exception as exc:
            self.result_var.set(f"读取本次报告失败：{exc}")

    @staticmethod
    def _format_vector(value: Any) -> str:
        if not isinstance(value, (list, tuple)):
            return "-"
        try:
            return "[" + ", ".join(f"{float(item):.3f}" for item in value) + "]"
        except (TypeError, ValueError):
            return str(value)

    def open_results_dir(self) -> None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(RUNS_DIR))  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("打开目录失败", str(exc), parent=self)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def on_close(self) -> None:
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
        # 不会在机器人自动运动期间强杀子进程；若正等待确认，可安全发送取消。
        if self.waiting_confirmation:
            self.cancel_pending_motion()
