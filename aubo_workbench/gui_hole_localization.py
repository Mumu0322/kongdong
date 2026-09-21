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
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from .auto_sector_selection import load_auto_sector_config
from .hole_map import load_hole_map, resolve_hole_map_path
from .gui_hole_map_selector import HoleMapSelector, collect_hole_map_points
from .io_utils import atomic_write_json
from .paths import (
    HANDEYE_DIAGNOSTIC_PATH,
    HOLE_LOCALIZATION_CURRENT_MAP_PATH,
    HOLE_LOCALIZATION_MAPS_DIR,
    HOLE_LOCALIZATION_RUNS_DIR,
    HOLE_LOCALIZATION_SECTOR_INFO_DIR,
    MODEL_PATH,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOCALIZATION_SCRIPT = PROJECT_DIR / "run_yolo_eye_in_hand_optimized.py"
OFFSET_TEST_SCRIPT = PROJECT_DIR / "run_coarse_to_fine_offset_test.py"
DEFAULT_MODEL = MODEL_PATH
DEFAULT_HANDEYE = HANDEYE_DIAGNOSTIC_PATH
RUNS_DIR = HOLE_LOCALIZATION_RUNS_DIR
AUTO_SECTOR_CONFIG_TEMPLATE = PROJECT_DIR / "configs" / "auto_sector_selection_static_template.json"
AUTO_SECTOR_CONFIG_PATH = HOLE_LOCALIZATION_SECTOR_INFO_DIR / "auto_sector_selection.json"


def _prepare_auto_sector_config() -> Path | None:
    """首次打开GUI时复制一份可编辑的自动分区配置。"""
    if AUTO_SECTOR_CONFIG_PATH.is_file():
        # 早期版本已经可能把旧的 radial 模板复制到该固定路径。当前实验
        # 不再要求伞架中心；只对“无区域且无中心”的默认占位文件做一次
        # 无损迁移，保留已填写过中心的旧 radial 配置。
        try:
            payload = json.loads(AUTO_SECTOR_CONFIG_PATH.read_text(encoding="utf-8"))
            if (
                isinstance(payload, dict)
                and payload.get("partition_mode", "radial") == "radial"
                and payload.get("origin_px") is None
                and not payload.get("regions")
            ):
                payload.update({
                    "partition_mode": "polygons",
                    "regions": [],
                    "image_size_px": None,
                    "sector_pose_records": payload.get("sector_pose_records", []),
                })
                atomic_write_json(AUTO_SECTOR_CONFIG_PATH, payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
            pass
        return AUTO_SECTOR_CONFIG_PATH
    if not AUTO_SECTOR_CONFIG_TEMPLATE.is_file():
        return None
    try:
        AUTO_SECTOR_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(AUTO_SECTOR_CONFIG_TEMPLATE, AUTO_SECTOR_CONFIG_PATH)
    except OSError:
        return AUTO_SECTOR_CONFIG_TEMPLATE
    return AUTO_SECTOR_CONFIG_PATH


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
        # 四种定位策略是 GUI 中唯一的流程选择来源；命令行开关由它直接生成。
        self.strategy_var = tk.StringVar(value="batch")
        self.auto_sector_selection_var = tk.BooleanVar(value=False)
        auto_sector_config = _prepare_auto_sector_config()
        self.auto_sector_config_var = tk.StringVar(
            value=str(auto_sector_config) if auto_sector_config is not None else ""
        )
        self.auto_sector_ids_var = tk.StringVar(value="")
        self.auto_exclude_boundary_var = tk.BooleanVar(value=False)
        self.confidence_var = tk.StringVar(value="0.35")
        self.coarse_height_var = tk.StringVar(value="340")
        self.fine_height_var = tk.StringVar(value="260")
        self.per_hole_fine_safe_z_margin_var = tk.StringVar(value="20.0")
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
        self.batch_coarse_frames_var = tk.StringVar(value="15")
        self.batch_coarse_min_valid_var = tk.StringVar(value="10")
        self.batch_coarse_min_holes_var = tk.StringVar(value="")
        self.batch_coarse_view_margin_var = tk.StringVar(value="50.0")
        self.batch_coarse_pose_refine_max_correction_var = tk.StringVar(value="5.0")
        self.batch_coarse_pose_refine_max_correction_rotation_var = tk.StringVar(value="2.0")
        self.map_build_safe_z_margin_var = tk.StringVar(value="100.0")
        self.map_build_localization_mode_var = tk.StringVar(value="coarse_only")
        self.batch_fine_localization_var = tk.BooleanVar(value=True)
        self.batch_fine_joint_localization_var = tk.BooleanVar(value=True)
        self.batch_fine_joint_max_residual_var = tk.StringVar(value="1.0")
        self.batch_fine_pointcloud_xy_fusion_var = tk.BooleanVar(value=True)
        self.batch_fine_pointcloud_xy_weight_var = tk.StringVar(value="0.35")
        self.batch_fine_pointcloud_xy_max_correction_var = tk.StringVar(value="0.6")
        self.batch_fine_pointcloud_xy_agreement_gate_var = tk.StringVar(value="2.5")
        self.batch_fine_frames_var = tk.StringVar(value="8")
        self.batch_fine_min_valid_var = tk.StringVar(value="5")
        self.batch_fine_stable_min_frames_var = tk.StringVar(value="5")
        self.batch_fine_settle_discard_frames_var = tk.StringVar(value="10")
        self.batch_fine_inplace_recovery_frames_var = tk.StringVar(value="4")
        self.batch_fine_supplement_rounds_var = tk.StringVar(value="2")
        self.batch_fine_view_margin_var = tk.StringVar(value="50.0")
        self.batch_fine_in_group_pose_adjustment_var = tk.BooleanVar(value=True)
        self.batch_fine_in_group_max_adjustments_var = tk.StringVar(value="2")
        self.batch_fine_in_group_max_xy_var = tk.StringVar(value="8.0")
        self.batch_fine_in_group_max_z_var = tk.StringVar(value="3.0")
        self.batch_fine_in_group_max_rotation_var = tk.StringVar(value="2.0")
        self.batch_fine_in_group_min_normal_holes_var = tk.StringVar(value="2")
        self.batch_fine_in_group_max_normal_spread_var = tk.StringVar(value="3.0")
        self.coarse_direct_final_height_var = tk.StringVar(value="340")
        self.coarse_direct_final_max_group_size_var = tk.StringVar(value="5")
        self.coarse_direct_final_early_stop_extra_frames_var = tk.StringVar(value="5")
        self.coarse_direct_final_settle_delay_var = tk.StringVar(value="1.0")
        self.coarse_direct_final_capture_only_var = tk.BooleanVar(value=False)
        self.optimize_hole_order_var = tk.BooleanVar(value=False)
        self.execute_var = tk.BooleanVar(value=False)
        self.experimental_var = tk.BooleanVar(value=True)
        self.shared_cache_validation_var = tk.BooleanVar(value=False)
        self.shared_cache_validation_frames_var = tk.StringVar(value="3")
        self.shared_cache_validation_min_valid_var = tk.StringVar(value="2")
        self.shared_cache_validation_view_margin_var = tk.StringVar(value="50.0")
        self.auto_next_hole_var = tk.BooleanVar(value=False)
        self.final_xy_var = tk.BooleanVar(value=True)
        self.charuco_xy_var = tk.BooleanVar(value=True)
        self.include_final_motion_var = tk.BooleanVar(value=False)
        self.hole_map_path_var = tk.StringVar(
            value=(
                str(HOLE_LOCALIZATION_CURRENT_MAP_PATH)
                if HOLE_LOCALIZATION_CURRENT_MAP_PATH.is_file() else ""
            )
        )
        self.sector_id_var = tk.StringVar(value="1")
        self.hole_map_ids_var = tk.StringVar(value="")
        self.hole_map_selection_summary_var = tk.StringVar(
            value="未单独选择（调用时使用全部有效孔）"
        )
        # 建图默认要求人工确认最终孔集，避免自动扇区筛选错误直接进入运动。
        self.map_hole_selection_mode_var = tk.StringVar(value="manual")
        self.repair_hole_id_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="待开始：默认仅预览，不会下发机器人运动")
        self.motion_var = tk.StringVar(value="无待确认运动")
        self.result_var = tk.StringVar(value="尚无本次结果")

        self._build()
        self.strategy_var.trace_add("write", self._apply_strategy)
        self.sector_id_var.trace_add("write", self._on_hole_map_context_changed)
        self.hole_map_path_var.trace_add("write", self._on_hole_map_context_changed)
        self.hole_map_ids_var.trace_add("write", self._update_hole_map_selection_summary)
        self._apply_strategy()
        self._refresh_repair_hole_choices()
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
        parameters_page = ScrollableTab(self.notebook, padding=10)
        diagnostics_page = ScrollableTab(self.notebook, padding=10)
        monitor_page = ScrollableTab(self.notebook, padding=10)
        self.notebook.add(detection_page, text="检测与地图")
        self.notebook.add(parameters_page, text="参数调优")
        self.notebook.add(diagnostics_page, text="诊断工具")
        self.notebook.add(monitor_page, text="运行监控")

        self._build_detection_tab(detection_page.inner)
        self._build_parameters_tab(parameters_page.inner)
        self._build_diagnostics_tab(diagnostics_page.inner)
        self._build_monitor_tab(monitor_page.inner)
        for page in (detection_page, parameters_page, diagnostics_page, monitor_page):
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

        strategy = ttk.LabelFrame(parent, text="检测策略（互斥）", padding=8)
        strategy.pack(fill=tk.X, pady=(8, 0))
        for row, (label, value) in enumerate((
            ("逐孔检测：每个孔独立粗定位和精定位", "per_hole"),
            ("多孔分组共享定位：340 mm 分组粗定位，260 mm 分组精定位", "batch"),
            ("复用粗定位缓存：一拍多验证，失败逐孔粗定位", "cache"),
            ("第四策略：340 mm点云中心直达（外围最多3孔，内部最多5孔）", "coarse_direct"),
        )):
            ttk.Radiobutton(
                strategy, text=label, variable=self.strategy_var, value=value,
            ).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Label(
            strategy,
            text="第四策略只使用独立高度的点云中心，默认340 mm；选区外围先分组且最多3孔，内部最多5孔，均优先3～5孔，必要时允许1～2孔；所有组保持紧凑并执行长宽比门限；确认停稳后默认等待1秒，最终XY仅使用ChArUco纠偏。",
            foreground="#555555",
        ).grid(row=4, column=0, sticky="w", pady=(4, 0))

        auto_panel = ttk.LabelFrame(parent, text="初始自动分区/选孔实验", padding=8)
        auto_panel.pack(fill=tk.X, pady=(8, 0))
        auto_panel.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            auto_panel,
            text="启用静态伞架自动扇区划分和自动选孔（仅初始观察帧）",
            variable=self.auto_sector_selection_var,
        ).grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 4))
        self._path_row(
            auto_panel, 1, "自动分区配置 JSON", self.auto_sector_config_var,
            self._browse_auto_sector_config,
        )
        ttk.Label(auto_panel, text="活动扇区").grid(
            row=2, column=0, sticky="w", padx=(0, 6), pady=3,
        )
        ttk.Entry(auto_panel, textvariable=self.auto_sector_ids_var, width=24).grid(
            row=2, column=1, sticky="w", pady=3,
        )
        ttk.Label(auto_panel, text="留空=全部已画区域；重叠处归编号最小区域").grid(
            row=2, column=2, columnspan=3, sticky="w", padx=(12, 0), pady=3)
        ttk.Label(
            auto_panel,
            text="鼠标画区无需伞架中心；孔中心在边界上直接归属，区域外不选孔。粗定位观察高度按当前策略单独执行。",
            foreground="#a35a00",
        ).grid(row=3, column=0, columnspan=5, sticky="w", pady=(3, 0))
        ttk.Label(
            auto_panel,
            text=f"扇区信息将按 S01、S02… 分目录保存到：{HOLE_LOCALIZATION_SECTOR_INFO_DIR}",
            foreground="#555555",
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(3, 0))
        ttk.Button(
            auto_panel,
            text="打开扇区信息目录",
            command=self.open_sector_info_dir,
        ).grid(row=4, column=4, sticky="e", padx=(8, 0), pady=(3, 0))
        ttk.Button(
            auto_panel,
            text="鼠标画区 / 修改区域",
            command=self._draw_auto_sector_regions,
        ).grid(row=5, column=4, sticky="e", padx=(8, 0), pady=(3, 0))
        ttk.Label(
            auto_panel,
            text="首次使用：点“鼠标画区”→启动视频流→调整观察位并确认位姿→画区域→保存；自动生成配置文件。",
            foreground="#a35a00",
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(3, 0))
        ttk.Button(
            auto_panel,
            text="打开配置文件",
            command=self._open_auto_sector_config,
        ).grid(row=6, column=4, sticky="e", padx=(8, 0), pady=(3, 0))

        options = ttk.LabelFrame(parent, text="执行与安全", padding=8)
        options.pack(fill=tk.X, pady=(8, 0))
        ttk.Checkbutton(options, text="真实运动", variable=self.execute_var).grid(row=0, column=0, sticky="w", padx=(0, 18))
        ttk.Checkbutton(
            options,
            text="使用当前最新实验手眼结果（含本次运动）",
            variable=self.experimental_var,
        ).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(options, text="执行最终点运动（地图调用固定启用）", variable=self.final_xy_var).grid(
            row=1, column=0, sticky="w", pady=(6, 0),
        )
        ttk.Checkbutton(
            options,
            text="使用 ChArUco XY 纠偏（默认开启）",
            variable=self.charuco_xy_var,
        ).grid(row=1, column=1, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            options, text="按最近邻优化孔序（工艺允许时启用）",
            variable=self.optimize_hole_order_var,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            options,
            text="当前孔完成后自动进入下一孔（两阶段/地图建图/地图调用）",
            variable=self.auto_next_hole_var,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(
            options,
            text="关闭时每个孔完成后等待人工确认；开启后孔间连续运动，请确认运行区域安全。",
            foreground="#a35a00",
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(2, 0))

        map_panel = ttk.LabelFrame(
            parent,
            text="孔位地图（建图可选340 mm粗定位或逐孔粗+精定位；调用时按当前策略执行）",
            padding=8,
        )
        map_panel.pack(fill=tk.X, pady=(8, 0))
        map_panel.columnconfigure(1, weight=1)
        ttk.Label(map_panel, text="当前地图 JSON").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(map_panel, textvariable=self.hole_map_path_var).grid(
            row=0, column=1, columnspan=2, sticky="ew", pady=3,
        )
        ttk.Button(map_panel, text="选择", command=self._browse_hole_map).grid(
            row=0, column=3, sticky="w", padx=(8, 0), pady=3,
        )
        ttk.Label(map_panel, text="扇区").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Combobox(
            map_panel,
            textvariable=self.sector_id_var,
            values=[str(value) for value in range(1, 7)],
            width=6,
            state="readonly",
        ).grid(row=1, column=1, sticky="w", pady=3)
        ttk.Label(
            map_panel,
            text="当前伞架共6个旋转扇区；建图/调用都只处理所选扇区",
            foreground="#555555",
        ).grid(row=1, column=2, columnspan=2, sticky="w", padx=(8, 0), pady=3)
        ttk.Label(map_panel, text="调用孔号").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(
            map_panel, textvariable=self.hole_map_ids_var, width=38, state="readonly",
        ).grid(
            row=2, column=1, sticky="w", pady=3,
        )
        ttk.Button(
            map_panel, text="点击地图选孔", command=self.open_hole_map_selector,
        ).grid(row=2, column=2, sticky="w", padx=(8, 0), pady=3)
        ttk.Label(
            map_panel,
            textvariable=self.hole_map_selection_summary_var,
            foreground="#555555",
        ).grid(row=2, column=3, sticky="w", padx=(8, 0), pady=3)
        ttk.Label(map_panel, text="返修孔号").grid(
            row=3, column=0, sticky="w", padx=(0, 6), pady=3,
        )
        self.repair_hole_combo = ttk.Combobox(
            map_panel,
            textvariable=self.repair_hole_id_var,
            values=(),
            width=10,
            state="readonly",
        )
        self.repair_hole_combo.grid(row=3, column=1, sticky="w", pady=3)
        selection_mode_panel = ttk.LabelFrame(map_panel, text="建图最终选孔方式", padding=5)
        selection_mode_panel.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(5, 0))
        ttk.Radiobutton(
            selection_mode_panel,
            text="自动建图：直接使用自动扇区候选",
            variable=self.map_hole_selection_mode_var,
            value="auto",
        ).grid(row=0, column=0, sticky="w", padx=(0, 14))
        ttk.Radiobutton(
            selection_mode_panel,
            text="人工选孔后建图（推荐）",
            variable=self.map_hole_selection_mode_var,
            value="manual",
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(
            selection_mode_panel,
            text="人工模式只显示自动扇区筛出的候选，按 Enter 确认后才开始340 mm建图。",
            foreground="#555555",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 0))
        build_localization_panel = ttk.LabelFrame(map_panel, text="建图定位方式", padding=5)
        build_localization_panel.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(5, 0))
        ttk.Radiobutton(
            build_localization_panel,
            text="快速建图：只做340 mm粗定位",
            variable=self.map_build_localization_mode_var,
            value="coarse_only",
        ).grid(row=0, column=0, sticky="w", padx=(0, 14))
        ttk.Radiobutton(
            build_localization_panel,
            text="逐孔建图：340 mm粗定位 + 260 mm精定位（推荐）",
            variable=self.map_build_localization_mode_var,
            value="per_hole",
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(
            build_localization_panel,
            text="逐孔模式会每孔移动并保存粗定位与精定位参考；不执行最终安放，地图调用时仍重新精定位。",
            foreground="#555555",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 0))
        ttk.Label(
            map_panel,
            text="从当前扇区粗地图选择一个孔；返修会重新做粗/精定位但不执行最终安放，成功后生成新地图版本。",
            foreground="#a35a00",
        ).grid(row=6, column=0, columnspan=4, sticky="w", pady=(2, 0))
        self.repair_map_btn = ttk.Button(
            map_panel, text="重新定位并更新此孔", command=self.start_hole_map_repair,
        )
        self.repair_map_btn.grid(row=7, column=0, padx=(0, 8), pady=(6, 0), sticky="w")
        self.build_map_btn = ttk.Button(
            map_panel, text="建立孔位地图", command=self.start_hole_map_build,
        )
        self.build_map_btn.grid(row=7, column=1, padx=(0, 8), pady=(6, 0), sticky="w")
        self.execute_map_btn = ttk.Button(
            map_panel, text="调用地图孔位", command=self.start_hole_map_execute,
        )
        self.execute_map_btn.grid(row=7, column=2, padx=(0, 8), pady=(6, 0), sticky="w")
        ttk.Button(map_panel, text="查看点云", command=self.open_hole_map_pointcloud).grid(
            row=7, column=3, sticky="w", padx=(8, 0), pady=(6, 0),
        )
        ttk.Button(map_panel, text="打开三维PLY", command=self.open_hole_map_ply).grid(
            row=8, column=3, sticky="w", padx=(8, 0), pady=(6, 0),
        )
        ttk.Label(
            map_panel,
            text="粗定位用于地图导航；逐孔模式额外保存每孔260 mm精定位参考，调用地图时仍重新采集点云和ChArUco结果。",
            foreground="#555555",
        ).grid(row=8, column=0, columnspan=3, sticky="w", padx=(0, 0), pady=(6, 0))

        actions = ttk.LabelFrame(parent, text="开始检测", padding=8)
        actions.pack(fill=tk.X, pady=(8, 0))
        self.start_btn = ttk.Button(actions, text="开始两阶段定位", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(
            actions, text="详细帧数、速度和分组参数请在“参数调优”页设置。", foreground="#555555",
        ).pack(side=tk.LEFT, padx=(16, 0))

    def _build_parameters_tab(self, parent: ttk.Frame) -> None:
        """集中放置低频参数，避免主检测页面被调试选项淹没。"""
        ttk.Label(
            parent,
            text="这些参数会直接影响定位精度、节拍和运动安全；正常使用时保持默认值即可。",
            foreground="#a35a00",
        ).pack(anchor="w", pady=(0, 6))

        two_stage = ttk.LabelFrame(parent, text="基础定位与运动参数", padding=8)
        two_stage.pack(fill=tk.X)
        self._grid_fields(two_stage, [
            ("置信度", self.confidence_var, 7),
            ("粗定位高度 mm", self.coarse_height_var, 8),
            ("精定位高度 mm", self.fine_height_var, 8),
            ("逐孔精拍横移Z余量 mm", self.per_hole_fine_safe_z_margin_var, 8),
            ("粗定位帧", self.coarse_frames_var, 6),
            ("精定位帧", self.fine_frames_var, 6),
            ("精确速度 m/s", self.speed_var, 7),
            ("加速度 m/s²", self.acc_var, 7),
            ("安全过渡速度 m/s", self.transit_speed_var, 7),
            ("安全过渡加速度 m/s²", self.transit_acc_var, 7),
            ("非接触接近速度 m/s", self.approach_speed_var, 7),
            ("非接触接近加速度 m/s²", self.approach_acc_var, 7),
        ], columns=3)

        batch = ttk.LabelFrame(parent, text="多孔分组共享定位参数", padding=8)
        self.shared_localization_frame = batch
        batch.pack(fill=tk.X, pady=(8, 0))
        self._grid_fields(batch, [
            ("批量采集帧数", self.batch_coarse_frames_var, 7),
            ("批量最少有效帧", self.batch_coarse_min_valid_var, 7),
            ("每帧最少孔数", self.batch_coarse_min_holes_var, 7),
            ("视野边缘余量 px", self.batch_coarse_view_margin_var, 7),
        ], columns=2)
        ttk.Checkbutton(
            batch,
            text=(
                "260 mm 分组共享精定位（默认最多4孔；失败孔进行一次"
                "受限XY/Z/RX/RY组内微调）"
            ),
            variable=self.batch_fine_localization_var,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 2))
        ttk.Checkbutton(
            batch,
            text="260 mm 多孔联合精定位（共享XY；保留逐孔残差、倾斜纠偏和ChArUco补偿）",
            variable=self.batch_fine_joint_localization_var,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(2, 2))
        ttk.Checkbutton(
            batch,
            text=(
                "最终 XY 融合粗拍点云支撑中心（粗精一致才修正，"
                "并按联合残差自适应降权）"
            ),
            variable=self.batch_fine_pointcloud_xy_fusion_var,
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(2, 2))
        ttk.Checkbutton(
            batch,
            text="失败孔启用组内受限XY/Z/RX/RY直接微调（RZ锁定）",
            variable=self.batch_fine_in_group_pose_adjustment_var,
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(2, 2))
        self._grid_fields(batch, [
            ("共享粗定位单步纠偏 mm", self.batch_coarse_pose_refine_max_correction_var, 7),
            ("共享粗定位单步纠偏旋转 °", self.batch_coarse_pose_refine_max_correction_rotation_var, 7),
            ("建图安全横移Z余量 mm", self.map_build_safe_z_margin_var, 7),
        ], columns=2, start_row=6)
        self._grid_fields(batch, [
            ("260 mm 精定位视野边缘余量 px", self.batch_fine_view_margin_var, 7),
            ("260 mm 批量帧数", self.batch_fine_frames_var, 7),
            ("260 mm 最少有效帧", self.batch_fine_min_valid_var, 7),
            ("260 mm 稳定门帧数", self.batch_fine_stable_min_frames_var, 7),
            ("260 mm 最少预热丢弃帧", self.batch_fine_settle_discard_frames_var, 7),
            ("260 mm 原位补帧数", self.batch_fine_inplace_recovery_frames_var, 7),
            ("260 mm 失败孔组内调整/补拍次数", self.batch_fine_supplement_rounds_var, 7),
            ("组内最多调整次数", self.batch_fine_in_group_max_adjustments_var, 7),
            ("组内最大XY mm", self.batch_fine_in_group_max_xy_var, 7),
            ("组内最大Z mm", self.batch_fine_in_group_max_z_var, 7),
            ("组内最大姿态角 °", self.batch_fine_in_group_max_rotation_var, 7),
            ("姿态调整最少法向孔数", self.batch_fine_in_group_min_normal_holes_var, 7),
            ("姿态调整法向离散 °", self.batch_fine_in_group_max_normal_spread_var, 7),
            ("联合孔级残差门限 mm", self.batch_fine_joint_max_residual_var, 7),
            ("点云中心权重", self.batch_fine_pointcloud_xy_weight_var, 7),
            ("点云最大修正 mm", self.batch_fine_pointcloud_xy_max_correction_var, 7),
            ("粗精一致性门限 mm", self.batch_fine_pointcloud_xy_agreement_gate_var, 7),
        ], columns=2, start_row=8)

        direct_info = ttk.LabelFrame(
            parent, text="第四策略：独立高度点云中心直达", padding=8,
        )
        self.coarse_direct_info_frame = direct_info
        direct_info.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(
            direct_info,
            text="默认340 mm；选区最外围孔先分组且每组最多3孔，内部孔每组最多5孔，均优先3～5孔并允许余数为1～2孔；外围组同样执行紧凑度门限，近似直线的三孔会拆分；停稳后额外等待1秒。仅采集评估会移动到观察位并保存点云结果，不执行最终XY/Z动作。",
            foreground="#555555",
        ).grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 2))
        self._grid_fields(direct_info, [
            ("点云拍摄高度 mm", self.coarse_direct_final_height_var, 7),
            ("内部每组最多孔数（1-5；外围最多3）", self.coarse_direct_final_max_group_size_var, 7),
            ("额外确认帧数", self.coarse_direct_final_early_stop_extra_frames_var, 7),
            ("停稳后额外等待 s", self.coarse_direct_final_settle_delay_var, 7),
        ], columns=3, start_row=1)
        ttk.Checkbutton(
            direct_info,
            text="仅采集评估（不执行最终 XY/Z 动作）",
            variable=self.coarse_direct_final_capture_only_var,
        ).grid(row=2, column=0, columnspan=6, sticky="w", pady=(3, 0))

        shared = ttk.LabelFrame(parent, text="缓存检测策略参数", padding=8)
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
            text="仅在检测策略选择“复用粗定位缓存”时生效；失败孔自动进入逐孔完整粗定位。",
            foreground="#555555",
        ).grid(row=1, column=0, columnspan=6, sticky="w", pady=(0, 6))
        self._grid_fields(shared, [
            ("每组验证帧数", self.shared_cache_validation_frames_var, 7),
            ("每孔最少有效帧", self.shared_cache_validation_min_valid_var, 7),
            ("视野边缘余量 px", self.shared_cache_validation_view_margin_var, 7),
        ], columns=3, start_row=2)

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
            panel, text="偏移测试执行最终 XY → Z（实机，不追加Y偏置）",
            variable=self.include_final_motion_var,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.offset_start_btn = ttk.Button(panel, text="开始偏移容忍度测试", command=self.start_offset_test)
        self.offset_start_btn.grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )

    def _apply_strategy(self, *_args: Any) -> None:
        """把互斥策略同步到所有会改变执行路径的开关。"""
        strategy = self.strategy_var.get()
        per_hole_strategy = strategy == "per_hole"
        coarse_direct_strategy = strategy == "coarse_direct"
        # 逐孔模式不能残留共享精定位或联合XY；否则多孔任务仍会被
        # sequential workflow 识别为“两拍共享”并先执行共享粗/精定位。
        fine_enabled = not per_hole_strategy and not coarse_direct_strategy
        self.batch_fine_localization_var.set(fine_enabled)
        self.batch_fine_joint_localization_var.set(fine_enabled)
        pointcloud_fusion_var = getattr(
            self, "batch_fine_pointcloud_xy_fusion_var", None,
        )
        if pointcloud_fusion_var is not None:
            pointcloud_fusion_var.set(fine_enabled)
        cache_strategy = strategy == "cache"
        self.shared_cache_validation_var.set(cache_strategy)
        self._update_strategy_controls()

    def _build_monitor_tab(self, parent: ttk.Frame) -> None:
        notice = ttk.LabelFrame(parent, text="操作提示", padding=8)
        notice.pack(fill=tk.X)
        ttk.Label(
            notice,
            text=("两阶段流程：相机初始画面中选择孔后按 Enter；默认每个孔完成后点击继续，"
                  "也可在“执行与安全”中启用自动检测下一孔。"
                  "地图中的坏孔可在“检测与地图”选择扇区/孔号后执行单孔返修；返修只生成新版本，不覆盖旧地图。"
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

    def _update_strategy_controls(self) -> None:
        """只允许当前选中的策略编辑自己的专属参数。"""
        strategy = self.strategy_var.get()
        coarse_direct_strategy = strategy == "coarse_direct"
        state = tk.NORMAL if strategy == "cache" else tk.DISABLED
        for widget in self.shared_cache_validation_frame.winfo_children():
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass
        # 逐孔策略下共享参数没有执行意义，全部置灰。第四策略仍使用
        # 批量点云采集参数，但高度和每组孔数由其专属面板控制。
        shared_state = tk.DISABLED if strategy == "per_hole" else tk.NORMAL
        for widget in self.shared_localization_frame.winfo_children():
            try:
                if coarse_direct_strategy and shared_state == tk.NORMAL:
                    row = int(widget.grid_info().get("row", -1))
                    widget.configure(state=tk.DISABLED if row >= 2 else tk.NORMAL)
                else:
                    widget.configure(state=shared_state)
            except tk.TclError:
                pass
        direct_frame = getattr(self, "coarse_direct_info_frame", None)
        if direct_frame is not None:
            direct_state = tk.NORMAL if coarse_direct_strategy else tk.DISABLED
            for widget in direct_frame.winfo_children():
                try:
                    widget.configure(state=direct_state)
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

    def _browse_hole_map(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="选择孔位地图",
            initialdir=str(HOLE_LOCALIZATION_MAPS_DIR),
            filetypes=[("孔位地图 JSON", "*.json"), ("所有文件", "*.*")],
        )
        if path:
            self.hole_map_path_var.set(path)

    def _on_hole_map_context_changed(self, *_args: object) -> None:
        """切换地图或扇区后清除旧选择，避免误用另一扇区的孔号。"""
        self.hole_map_ids_var.set("")
        self._refresh_repair_hole_choices()

    def _update_hole_map_selection_summary(self, *_args: object) -> None:
        raw_ids = self.hole_map_ids_var.get().replace(",", " ").split()
        if raw_ids:
            self.hole_map_selection_summary_var.set(f"已选 {len(raw_ids)} 个孔；按点击顺序执行")
        else:
            self.hole_map_selection_summary_var.set("未单独选择（调用时使用全部有效孔）")

    def open_hole_map_selector(self) -> None:
        """打开按真实粗定位 XY 坐标绘制的可点击选孔窗口。"""
        existing = getattr(self, "_hole_map_selector", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return
        raw_path = self.hole_map_path_var.get().strip()
        if not raw_path:
            raw_path = str(HOLE_LOCALIZATION_CURRENT_MAP_PATH)
        try:
            sector_id = int(self.sector_id_var.get())
            if not 1 <= sector_id <= 6:
                raise ValueError("扇区编号必须在 1 到 6 之间")
            map_path = resolve_hole_map_path(raw_path)
            payload = load_hole_map(raw_path)
            points = collect_hole_map_points(payload, sector_id)
            ids_text = self.hole_map_ids_var.get().replace(",", " ").strip()
            if ids_text:
                initial_ids = [int(item) for item in ids_text.split()]
            else:
                # 旧界面中留空表示调用全部有效孔；首次打开保持相同行为，
                # 用户可点“清空”后按需要重新选择。
                initial_ids = [
                    point.hole_id for point in points
                    if point.status in {"ready", "completed"}
                ]

            def apply_selection(selected_ids: list[int]) -> None:
                self.hole_map_ids_var.set(" ".join(str(item) for item in selected_ids))
                self.status_var.set(
                    f"地图选孔完成：S{sector_id:02d} 共选择 {len(selected_ids)} 个孔。"
                )

            self._hole_map_selector = HoleMapSelector(
                self,
                points=points,
                sector_id=sector_id,
                map_name=map_path.parent.name,
                initial_ids=initial_ids,
                on_confirm=apply_selection,
            )
        except Exception as exc:
            messagebox.showerror("打开孔洞地图失败", str(exc), parent=self)

    def _browse_auto_sector_config(self) -> None:
        current_text = str(self.auto_sector_config_var.get()).strip()
        current = Path(current_text).expanduser() if current_text else PROJECT_DIR / "configs"
        initialdir = current.parent if current.is_file() and current.parent.is_dir() else current
        if not initialdir.is_dir():
            initialdir = PROJECT_DIR / "configs"
        path = filedialog.askopenfilename(
            parent=self,
            title="选择静态伞架自动分区配置",
            initialdir=str(initialdir),
            filetypes=[("自动分区 JSON", "*.json"), ("所有文件", "*.*")],
        )
        if path:
            self.auto_sector_config_var.set(path)

    def _draw_auto_sector_regions(self) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showerror("流程正在运行", "请先停止定位流程，再打开画区窗口。", parent=self)
            return
        existing = getattr(self, "_sector_editor", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            return
        from .gui_sector_editor import SectorEditor
        config_path = self.auto_sector_config_var.get().strip() or str(AUTO_SECTOR_CONFIG_PATH)
        if Path(config_path).resolve() == AUTO_SECTOR_CONFIG_TEMPLATE.resolve():
            config_path = str(AUTO_SECTOR_CONFIG_PATH)
        def saved(path):
            self.auto_sector_config_var.set(str(path))
            self.auto_sector_selection_var.set(True)
            self.auto_sector_ids_var.set("")
            self.status_var.set("画区配置已保存，可开始两阶段定位。")
        self._sector_editor = SectorEditor(
            self, config_path, saved, connection_provider=self.connection_provider
        )

    def _create_auto_sector_config(self) -> None:
        """创建或重新选择一份可编辑的自动分区配置模板。"""
        config_path = _prepare_auto_sector_config()
        if config_path is None:
            messagebox.showerror(
                "配置模板不存在",
                f"找不到自动分区模板：{AUTO_SECTOR_CONFIG_TEMPLATE}",
                parent=self,
            )
            return
        self.auto_sector_config_var.set(str(config_path))
        messagebox.showinfo(
            "自动分区配置已准备",
            "配置文件已准备好。请点击“鼠标画区”，用视频流调整观察位、记录机械臂位姿并绘制区域，再勾选自动分区运行。",
            parent=self,
        )

    def _open_auto_sector_config(self) -> None:
        """用系统默认编辑器打开当前自动分区配置。"""
        config_path = Path(self.auto_sector_config_var.get()).expanduser()
        if not config_path.is_file():
            self._create_auto_sector_config()
            config_path = Path(self.auto_sector_config_var.get()).expanduser()
        if not config_path.is_file():
            return
        try:
            os.startfile(str(config_path))  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("打开配置文件失败", str(exc), parent=self)

    def _refresh_repair_hole_choices(self) -> None:
        """根据当前地图/扇区刷新单孔返修下拉框。"""
        combo = getattr(self, "repair_hole_combo", None)
        if combo is None:
            return
        raw_path = self.hole_map_path_var.get().strip()
        if not raw_path:
            raw_path = str(HOLE_LOCALIZATION_CURRENT_MAP_PATH)
        try:
            sector_id = int(self.sector_id_var.get())
            if not 1 <= sector_id <= 6:
                raise ValueError
            payload = load_hole_map(raw_path)
            key = f"S{sector_id:02d}"
            sector = (payload.get("sectors") or {}).get(key, {})
            holes = sector.get("holes") if isinstance(sector, dict) else {}
            values = sorted(
                str(int(hole.get("hole_id")))
                for hole in (holes or {}).values()
                if isinstance(hole, dict)
                and str(hole.get("status", "ready")) in {"ready", "completed"}
                and hole.get("hole_id") is not None
            )
            combo.configure(values=values)
            if self.repair_hole_id_var.get().strip() not in values:
                self.repair_hole_id_var.set(values[0] if values else "")
        except Exception:
            combo.configure(values=())
            self.repair_hole_id_var.set("")

    def open_hole_map_pointcloud(self) -> None:
        """打开当前/选定地图的点云预览JPG。"""
        self._open_hole_map_artifact("preview_jpg", "点云预览")

    def open_hole_map_ply(self) -> None:
        """用系统默认三维查看器打开当前/选定地图的PLY点云。"""
        self._open_hole_map_artifact("ply", "三维点云PLY")

    def _open_hole_map_artifact(self, artifact_name: str, label: str) -> None:
        raw_path = self.hole_map_path_var.get().strip()
        if not raw_path:
            raw_path = str(HOLE_LOCALIZATION_CURRENT_MAP_PATH)
        try:
            map_path = resolve_hole_map_path(raw_path)
            payload = load_hole_map(raw_path)
            artifact = (payload.get("artifacts") or {}).get(artifact_name)
            if int(payload.get("schema_version", -1)) == 4:
                sector_key = f"S{int(self.sector_id_var.get()):02d}"
                artifact = (
                    ((payload.get("artifacts") or {}).get("sectors") or {})
                    .get(sector_key, {})
                    .get(artifact_name)
                )
            if not artifact:
                raise ValueError(f"该地图没有{label}；请重新建立地图")
            preview = map_path.parent / str(artifact)
            if not preview.is_file():
                raise FileNotFoundError(f"{label}不存在：{preview}")
            os.startfile(str(preview))  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showerror(f"打开{label}失败", str(exc), parent=self)

    def _numbers(self) -> dict[str, float | int]:
        try:
            numbers: dict[str, float | int] = {
                "confidence": float(self.confidence_var.get()),
                "coarse_height": float(self.coarse_height_var.get()),
                "fine_height": float(self.fine_height_var.get()),
                "per_hole_fine_safe_z_margin": float(
                    getattr(
                        getattr(self, "per_hole_fine_safe_z_margin_var", None),
                        "get",
                        lambda: "20.0",
                    )()
                ),
                "coarse_frames": int(self.coarse_frames_var.get()),
                "fine_frames": int(self.fine_frames_var.get()),
                "speed": float(self.speed_var.get()),
                "acc": float(self.acc_var.get()),
                "transit_speed": float(self.transit_speed_var.get()),
                "transit_acc": float(self.transit_acc_var.get()),
                "approach_speed": float(self.approach_speed_var.get()),
                "approach_acc": float(self.approach_acc_var.get()),
            }
            if float(numbers["per_hole_fine_safe_z_margin"]) < 10.0:
                raise ValueError("逐孔精拍横移Z余量不能小于10 mm")
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
        is_map_build = mode == "hole_map_build"
        is_map_execute = mode == "hole_map_execute"
        is_map_repair = mode == "hole_map_repair"
        if not (is_offset or mode == "two_stage" or is_map_build or is_map_execute or is_map_repair):
            raise ValueError(f"未知定位流程模式：{mode}")
        script = OFFSET_TEST_SCRIPT if is_offset else LOCALIZATION_SCRIPT
        if not script.is_file():
            raise FileNotFoundError(f"找不到定位脚本：{script}")
        model = Path(self.model_var.get().strip())
        handeye = Path(self.handeye_var.get().strip())
        execute = self.execute_var.get() if execute_override is None else bool(execute_override)
        # 地图调用也要启动相机、加载YOLO和当前手眼；是否执行260 mm精定位
        # 由当前检测策略决定。
        if not is_offset and not model.is_file():
            raise FileNotFoundError(f"YOLO 模型不存在：{model}")
        if not is_offset and not handeye.is_file():
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
            "--per-hole-fine-safe-z-margin-mm",
            str(values.get("per_hole_fine_safe_z_margin", 20.0)),
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
        # 地图调用同样是实时视觉运动流程，遵守实验手眼放行策略。
        if not is_offset:
            command.append(
                "--allow-experimental-handeye"
                if self.experimental_var.get() else "--require-validated-handeye"
            )
        # 自动分区只用于现场初始选孔和建图入口；地图调用/返修使用已有
        # 孔位身份，不能再次用初始候选筛选覆盖目标。
        if mode in {"two_stage", "hole_map_build"}:
            auto_var = getattr(self, "auto_sector_selection_var", None)
            auto_enabled = bool(auto_var is not None and auto_var.get())
            if auto_enabled:
                config_text = str(
                    getattr(
                        getattr(self, "auto_sector_config_var", None),
                        "get",
                        lambda: "",
                    )()
                ).strip()
                if not config_text:
                    raise ValueError("启用自动分区实验必须选择自动分区配置 JSON")
                config_path = Path(config_text)
                if not config_path.is_file():
                    raise FileNotFoundError(f"自动分区配置不存在：{config_path}")
                active_sector_ids = None
                command.extend(["--auto-select-holes", "--auto-sector-config", str(config_path)])
                sector_text = str(
                    getattr(
                        getattr(self, "auto_sector_ids_var", None),
                        "get",
                        lambda: "",
                    )()
                ).replace(",", " ").strip()
                if sector_text:
                    try:
                        sector_ids = [int(item) for item in sector_text.split()]
                    except ValueError as exc:
                        raise ValueError("活动扇区必须是空格或逗号分隔的整数") from exc
                    if not sector_ids or any(item < 1 or item > 6 for item in sector_ids):
                        raise ValueError("活动扇区必须是1到6之间的整数")
                    active_sector_ids = sector_ids
                    command.extend(["--auto-sector-ids", *[str(item) for item in sector_ids]])
                boundary_var = getattr(self, "auto_exclude_boundary_var", None)
                include_boundary_candidates = None
                if boundary_var is not None and boundary_var.get():
                    include_boundary_candidates = False
                    command.append("--auto-exclude-boundary-candidates")
                try:
                    load_auto_sector_config(
                        config_path,
                        active_sector_ids=active_sector_ids,
                        include_boundary_candidates=include_boundary_candidates,
                    )
                except (FileNotFoundError, ValueError) as exc:
                    raise ValueError(
                        f"自动分区配置无法使用：{config_path}\n{exc}\n"
                        "请点击“鼠标画区 / 修改区域”，在固定观察位的图像上画区并保存。"
                    ) from exc
        if is_offset:
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
        elif is_map_execute:
            map_path_text = self.hole_map_path_var.get().strip()
            if not map_path_text:
                map_path_text = str(HOLE_LOCALIZATION_CURRENT_MAP_PATH)
            map_path = Path(map_path_text)
            # current.json 不存在时也允许启动：命令行会把它识别为“当前
            # 扇区无地图”，进入首轮共享粗/精定位建图。
            if not map_path.is_file() and map_path.name.lower() != "current.json":
                raise FileNotFoundError(f"孔位地图不存在：{map_path}")
            try:
                sector_id = int(self.sector_id_var.get())
            except ValueError as exc:
                raise ValueError("扇区编号必须是1到6的整数") from exc
            if not 1 <= sector_id <= 6:
                raise ValueError("扇区编号必须在1到6之间")
            command.extend([
                "--hole-map-mode", "execute",
                "--hole-map-path", str(map_path),
                "--sector-id", str(sector_id),
                # 地图调用成功后固定进入最终点运动；点云中心不可用的孔
                # 会记录为失败，不进入其它定位流程。
                "--move-final-xy",
            ])
            strategy_name = getattr(getattr(self, "strategy_var", None), "get", lambda: "")()
            if strategy_name == "coarse_direct":
                try:
                    direct_height = float(getattr(
                        getattr(self, "coarse_direct_final_height_var", None),
                        "get", lambda: "340",
                    )())
                    direct_group_size = int(getattr(
                        getattr(self, "coarse_direct_final_max_group_size_var", None),
                        "get", lambda: "5",
                    )())
                    direct_extra_frames = int(getattr(
                        getattr(self, "coarse_direct_final_early_stop_extra_frames_var", None),
                        "get", lambda: "5",
                    )())
                    direct_settle_delay = float(getattr(
                        getattr(self, "coarse_direct_final_settle_delay_var", None),
                        "get", lambda: "1.0",
                    )())
                except (TypeError, ValueError) as exc:
                    raise ValueError("第四策略点云参数必须是有效数字") from exc
                if (
                    not math.isfinite(direct_height) or direct_height <= 0.0
                    or direct_group_size < 1 or direct_group_size > 5
                    or direct_extra_frames < 0
                    or not math.isfinite(direct_settle_delay)
                    or direct_settle_delay < 0.0
                    or direct_settle_delay > 30.0
                ):
                    raise ValueError("第四策略要求高度>0、每组孔数在1到5、额外确认帧数>=0、停稳等待在0到30秒")
                command.extend([
                    "--coarse-direct-final",
                    "--coarse-direct-final-height-mm", str(direct_height),
                    "--coarse-direct-final-max-group-size", str(direct_group_size),
                    "--coarse-direct-final-early-stop-extra-frames", str(direct_extra_frames),
                    "--coarse-direct-final-settle-delay-s", str(direct_settle_delay),
                    "--no-batch-fine-localization",
                    "--no-batch-fine-joint-localization",
                    "--no-batch-fine-pointcloud-xy-fusion",
                ])
                if bool(getattr(
                    getattr(self, "coarse_direct_final_capture_only_var", None),
                    "get", lambda: False,
                )()):
                    command.append("--coarse-direct-final-capture-only")
                    command.append("--no-move-final-xy")
            ids_text = self.hole_map_ids_var.get().replace(",", " ").strip()
            if ids_text:
                try:
                    hole_ids = [int(item) for item in ids_text.split()]
                except ValueError as exc:
                    raise ValueError("调用孔号必须是空格或逗号分隔的整数") from exc
                if not hole_ids or any(item <= 0 for item in hole_ids):
                    raise ValueError("调用孔号必须是正整数")
                command.extend(["--hole-ids", *[str(item) for item in hole_ids]])
        elif is_map_repair:
            if not execute:
                raise ValueError("单孔地图返修必须勾选“真实运动”")
            map_path_text = self.hole_map_path_var.get().strip()
            if not map_path_text:
                map_path_text = str(HOLE_LOCALIZATION_CURRENT_MAP_PATH)
            map_path = Path(map_path_text)
            if not map_path.is_file():
                raise FileNotFoundError(f"孔位地图不存在：{map_path}")
            try:
                sector_id = int(self.sector_id_var.get())
                hole_id = int(self.repair_hole_id_var.get())
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("返修扇区和孔号必须是有效整数") from exc
            if not 1 <= sector_id <= 6 or hole_id <= 0:
                raise ValueError("返修扇区必须在1到6之间，孔号必须是正整数")
            command.extend([
                "--hole-map-mode", "repair",
                "--hole-map-path", str(map_path),
                "--sector-id", str(sector_id),
                "--repair-hole-id", str(hole_id),
                "--no-batch-fine-localization",
                "--no-batch-fine-joint-localization",
                "--no-batch-fine-pointcloud-xy-fusion",
                "--no-reuse-coarse-cache",
                "--no-reuse-persistent-coarse-cache",
                "--no-move-final-xy",
            ])
        else:  # two_stage / hole_map_build
            if is_map_build:
                try:
                    sector_id = int(self.sector_id_var.get())
                except ValueError as exc:
                    raise ValueError("扇区编号必须是1到6的整数") from exc
                if not 1 <= sector_id <= 6:
                    raise ValueError("扇区编号必须在1到6之间")
                selection_mode = str(
                    getattr(
                        getattr(self, "map_hole_selection_mode_var", None),
                        "get",
                        lambda: "manual",
                    )()
                ).strip().lower()
                if selection_mode not in {"auto", "manual"}:
                    raise ValueError("建图最终选孔方式必须是自动或人工")
                map_build_localization_mode = str(
                    getattr(
                        getattr(self, "map_build_localization_mode_var", None),
                        "get",
                        lambda: "coarse_only",
                    )()
                ).strip().lower()
                if map_build_localization_mode not in {"coarse_only", "per_hole"}:
                    raise ValueError("建图定位方式必须是快速粗定位或逐孔粗+精定位")
                auto_enabled = bool(
                    getattr(getattr(self, "auto_sector_selection_var", None), "get", lambda: False)()
                )
                if selection_mode == "auto" and not auto_enabled:
                    raise ValueError("自动建图必须先勾选静态伞架自动扇区划分和自动选孔")
                command.extend([
                    "--sector-id", str(sector_id),
                    "--map-hole-selection-mode", selection_mode,
                    "--map-build-localization-mode", map_build_localization_mode,
                ])
                try:
                    max_correction = float(getattr(
                        getattr(self, "batch_coarse_pose_refine_max_correction_var", None),
                        "get", lambda: "5.0",
                    )())
                    max_rotation = float(getattr(
                        getattr(self, "batch_coarse_pose_refine_max_correction_rotation_var", None),
                        "get", lambda: "2.0",
                    )())
                    safe_margin = float(getattr(
                        getattr(self, "map_build_safe_z_margin_var", None),
                        "get", lambda: "100.0",
                    )())
                except (TypeError, ValueError) as exc:
                    raise ValueError("建图安全纠偏参数必须是有效数字") from exc
                if not math.isfinite(max_correction) or max_correction <= 0.0:
                    raise ValueError("共享粗定位最大纠偏必须是大于0的有限数字")
                if not math.isfinite(max_rotation) or max_rotation <= 0.0:
                    raise ValueError("共享粗定位最大纠偏旋转必须是大于0的有限数字")
                if not math.isfinite(safe_margin) or safe_margin <= 0.0:
                    raise ValueError("建图安全横移Z余量必须是大于0的有限数字")
                command.extend([
                    "--batch-coarse-pose-refine-max-correction-mm", str(max_correction),
                    "--batch-coarse-pose-refine-max-correction-rotation-deg", str(max_rotation),
                    "--map-build-safe-z-margin-mm", str(safe_margin),
                ])
            command.append(
                "--no-move-final-xy"
                if is_map_build else
                "--move-final-xy" if self.final_xy_var.get() else "--no-move-final-xy"
            )
            strategy_name = getattr(getattr(self, "strategy_var", None), "get", lambda: "")()
            coarse_direct_strategy = strategy_name == "coarse_direct"
            cache_strategy = strategy_name == "cache"
            # 缓存开关只由检测策略决定；建图始终采集当前扇区的新鲜点云。
            reuse_session_cache = cache_strategy and not is_map_build
            reuse_persistent_cache = cache_strategy and not is_map_build
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
            if coarse_direct_strategy and shared_cache_enabled:
                raise ValueError("第四策略点云中心直达不能启用共享粗定位缓存验证")
            if shared_cache_enabled:
                if not cache_strategy:
                    raise ValueError("共享快速缓存验证要求选择“复用粗定位缓存”检测策略")
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
            if strategy_name in {"batch", "coarse_direct"}:
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
                not is_map_build
                and strategy_name not in {"per_hole", "coarse_direct"}
                and (fine_batch_var is None or fine_batch_var.get())
            )
            if fine_batch_enabled:
                try:
                    fine_batch_margin = float(
                        getattr(
                            getattr(self, "batch_fine_view_margin_var", None),
                            "get",
                            lambda: "50.0",
                        )()
                    )
                    pointcloud_xy_weight = float(getattr(
                        getattr(self, "batch_fine_pointcloud_xy_weight_var", None),
                        "get", lambda: "0.35",
                    )())
                    pointcloud_xy_max_correction = float(getattr(
                        getattr(self, "batch_fine_pointcloud_xy_max_correction_var", None),
                        "get", lambda: "0.6",
                    )())
                    pointcloud_xy_agreement_gate = float(getattr(
                        getattr(self, "batch_fine_pointcloud_xy_agreement_gate_var", None),
                        "get", lambda: "2.5",
                    )())
                    joint_max_residual = float(getattr(
                        getattr(self, "batch_fine_joint_max_residual_var", None),
                        "get", lambda: "1.0",
                    )())
                    fine_in_group_max_xy = float(getattr(
                        getattr(self, "batch_fine_in_group_max_xy_var", None),
                        "get", lambda: "8.0",
                    )())
                    fine_in_group_max_z = float(getattr(
                        getattr(self, "batch_fine_in_group_max_z_var", None),
                        "get", lambda: "3.0",
                    )())
                    fine_in_group_max_rotation = float(getattr(
                        getattr(self, "batch_fine_in_group_max_rotation_var", None),
                        "get", lambda: "2.0",
                    )())
                    fine_in_group_max_normal_spread = float(getattr(
                        getattr(
                            self,
                            "batch_fine_in_group_max_normal_spread_var",
                            None,
                        ),
                        "get", lambda: "3.0",
                    )())
                except ValueError as exc:
                    raise ValueError("批量精定位视野或点云融合参数必须是有效数字") from exc
                if not math.isfinite(fine_batch_margin) or fine_batch_margin < 0.0:
                    raise ValueError("批量精定位视野边缘余量必须是大于等于 0 的有限数字")
                if not math.isfinite(joint_max_residual) or joint_max_residual <= 0.0:
                    raise ValueError("联合孔级残差门限必须是大于 0 的有限数字")
                if not math.isfinite(pointcloud_xy_weight) or not 0.0 <= pointcloud_xy_weight <= 1.0:
                    raise ValueError("点云中心权重必须是 0 到 1 之间的有限数字")
                if (
                    not math.isfinite(pointcloud_xy_max_correction)
                    or pointcloud_xy_max_correction < 0.0
                ):
                    raise ValueError("点云中心最大修正量必须是大于等于 0 的有限数字")
                if (
                    not math.isfinite(pointcloud_xy_agreement_gate)
                    or pointcloud_xy_agreement_gate <= 0.0
                ):
                    raise ValueError("粗精一致性门限必须是大于 0 的有限数字")
                if not math.isfinite(fine_in_group_max_xy) or fine_in_group_max_xy <= 0.0:
                    raise ValueError("组内最大XY调整必须是大于 0 的有限数字")
                if not math.isfinite(fine_in_group_max_z) or fine_in_group_max_z < 0.0:
                    raise ValueError("组内最大Z调整必须是大于等于 0 的有限数字")
                if (
                    not math.isfinite(fine_in_group_max_rotation)
                    or fine_in_group_max_rotation < 0.0
                ):
                    raise ValueError("组内最大姿态调整必须是大于等于 0 的有限数字")
                if (
                    not math.isfinite(fine_in_group_max_normal_spread)
                    or fine_in_group_max_normal_spread < 0.0
                ):
                    raise ValueError("组内法向离散门限必须是大于等于 0 的有限数字")
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
                    fine_batch_inplace_recovery = int(getattr(
                        getattr(self, "batch_fine_inplace_recovery_frames_var", None),
                        "get", lambda: "4",
                    )())
                    fine_batch_supplement_rounds = int(getattr(
                        getattr(self, "batch_fine_supplement_rounds_var", None),
                        "get", lambda: "1",
                    )())
                    fine_in_group_max_adjustments = int(getattr(
                        getattr(
                            self,
                            "batch_fine_in_group_max_adjustments_var",
                            None,
                        ),
                        "get", lambda: "1",
                    )())
                    fine_in_group_min_normal_holes = int(getattr(
                        getattr(
                            self,
                            "batch_fine_in_group_min_normal_holes_var",
                            None,
                        ),
                        "get", lambda: "2",
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
                if fine_batch_inplace_recovery < 0:
                    raise ValueError("批量精定位原位补帧数不能小于 0")
                if fine_batch_supplement_rounds < 0:
                    raise ValueError("批量精定位共享补拍轮数不能小于 0")
                if fine_in_group_max_adjustments < 0:
                    raise ValueError("组内位姿调整次数不能小于 0")
                if fine_in_group_min_normal_holes < 1:
                    raise ValueError("组内姿态调整最少法向孔数必须大于 0")
            command.append(
                "--batch-fine-localization"
                if fine_batch_enabled else "--no-batch-fine-localization"
            )
            joint_batch_var = getattr(
                self, "batch_fine_joint_localization_var", None,
            )
            joint_batch_enabled = bool(
                not is_map_build
                and strategy_name not in {"per_hole", "coarse_direct"}
                and (joint_batch_var is None or joint_batch_var.get())
            )
            pointcloud_fusion_var = getattr(
                self, "batch_fine_pointcloud_xy_fusion_var", None,
            )
            pointcloud_fusion_enabled = bool(
                fine_batch_enabled
                and (pointcloud_fusion_var is None or pointcloud_fusion_var.get())
            )
            in_group_pose_var = getattr(
                self, "batch_fine_in_group_pose_adjustment_var", None,
            )
            in_group_pose_enabled = bool(
                fine_batch_enabled
                and (in_group_pose_var is None or in_group_pose_var.get())
            )
            command.extend([
                *(
                    [
                        "--batch-fine-view-margin-px", str(fine_batch_margin),
                        "--batch-fine-frames", str(fine_batch_frames),
                        "--batch-fine-min-valid", str(fine_batch_min_valid),
                        "--batch-fine-stable-min-frames", str(fine_batch_stable_min),
                        "--batch-fine-settle-discard-frames", str(fine_batch_settle),
                        "--batch-fine-inplace-recovery-frames", str(fine_batch_inplace_recovery),
                        "--batch-fine-supplement-rounds", str(fine_batch_supplement_rounds),
                        "--batch-fine-in-group-max-adjustments", str(fine_in_group_max_adjustments),
                        "--batch-fine-in-group-max-xy-mm", str(fine_in_group_max_xy),
                        "--batch-fine-in-group-max-z-mm", str(fine_in_group_max_z),
                        "--batch-fine-in-group-max-rotation-deg", str(fine_in_group_max_rotation),
                        "--batch-fine-in-group-min-normal-holes", str(fine_in_group_min_normal_holes),
                        "--batch-fine-in-group-max-normal-spread-deg", str(fine_in_group_max_normal_spread),
                        "--batch-fine-joint-max-residual-mm", str(joint_max_residual),
                        "--batch-fine-pointcloud-xy-weight", str(pointcloud_xy_weight),
                        "--batch-fine-pointcloud-xy-max-correction-mm", str(pointcloud_xy_max_correction),
                        "--batch-fine-pointcloud-xy-agreement-gate-mm", str(pointcloud_xy_agreement_gate),
                    ]
                    if fine_batch_enabled else []
                ),
                "--batch-fine-joint-localization"
                if joint_batch_enabled else "--no-batch-fine-joint-localization",
                "--batch-fine-pointcloud-xy-fusion"
                if pointcloud_fusion_enabled else "--no-batch-fine-pointcloud-xy-fusion",
                "--batch-fine-in-group-pose-adjustment"
                if in_group_pose_enabled else "--no-batch-fine-in-group-pose-adjustment",
            ])
            if coarse_direct_strategy and not is_map_build:
                try:
                    direct_height = float(getattr(
                        getattr(self, "coarse_direct_final_height_var", None),
                        "get", lambda: "340",
                    )())
                    direct_group_size = int(getattr(
                        getattr(self, "coarse_direct_final_max_group_size_var", None),
                        "get", lambda: "5",
                    )())
                    direct_extra_frames = int(getattr(
                        getattr(self, "coarse_direct_final_early_stop_extra_frames_var", None),
                        "get", lambda: "5",
                    )())
                    direct_settle_delay = float(getattr(
                        getattr(self, "coarse_direct_final_settle_delay_var", None),
                        "get", lambda: "1.0",
                    )())
                except (TypeError, ValueError) as exc:
                    raise ValueError("第四策略点云参数必须是有效数字") from exc
                if (
                    not math.isfinite(direct_height) or direct_height <= 0.0
                    or direct_group_size < 1 or direct_group_size > 5
                    or direct_extra_frames < 0
                    or not math.isfinite(direct_settle_delay)
                    or direct_settle_delay < 0.0
                    or direct_settle_delay > 30.0
                ):
                    raise ValueError("第四策略要求高度>0、每组孔数在1到5、额外确认帧数>=0、停稳等待在0到30秒")
                command.append("--coarse-direct-final")
                command.extend([
                    "--coarse-direct-final-height-mm", str(direct_height),
                    "--coarse-direct-final-max-group-size", str(direct_group_size),
                    "--coarse-direct-final-early-stop-extra-frames", str(direct_extra_frames),
                    "--coarse-direct-final-settle-delay-s", str(direct_settle_delay),
                ])
                if bool(getattr(
                    getattr(self, "coarse_direct_final_capture_only_var", None),
                    "get", lambda: False,
                )()):
                    command.append("--coarse-direct-final-capture-only")
                    command.append("--no-move-final-xy")
            if is_map_build:
                command.extend([
                    "--hole-map-mode", "build",
                    (
                        "--batch-coarse-localization"
                        if str(
                            getattr(
                                getattr(self, "map_build_localization_mode_var", None),
                                "get",
                                lambda: "coarse_only",
                            )()
                        ).strip().lower() == "coarse_only"
                        else "--no-batch-coarse-localization"
                    ),
                ])
        command.append(
            "--use-charuco-xy-correction"
            if bool(getattr(getattr(self, "charuco_xy_var", None), "get", lambda: True)())
            else "--no-charuco-xy-correction"
        )
        return command

    def start(self) -> None:
        self._start_process("two_stage")

    def start_hole_map_build(self) -> None:
        self._start_process("hole_map_build")

    def start_hole_map_execute(self) -> None:
        # 与固定的地图调用命令保持界面状态一致，避免旧窗口状态显示关闭。
        self.final_xy_var.set(True)
        self._start_process("hole_map_execute")

    def start_hole_map_repair(self) -> None:
        self._refresh_repair_hole_choices()
        self._start_process("hole_map_repair")

    def start_offset_test(self) -> None:
        self._start_process("offset")

    def _start_process(self, mode: str, execute_override: bool | None = None) -> None:
        editor = getattr(self, "_sector_editor", None)
        if editor is not None and editor.winfo_exists():
            messagebox.showerror("画区窗口尚未关闭", "请先保存或关闭画区窗口，再启动定位。", parent=self)
            return
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
            "将执行最终XY和降Z，不追加Y偏置；每个采样点完成后安全回升并返回中心。"
            if self.include_final_motion_var.get() else
            "不会执行最终XY或最终Z动作。"
        )
        auto_next_hole = bool(
            getattr(getattr(self, "auto_next_hole_var", None), "get", lambda: False)()
        )
        coarse_direct_strategy = bool(
            getattr(getattr(self, "strategy_var", None), "get", lambda: "")()
            == "coarse_direct"
        )
        coarse_direct_height = str(getattr(
            getattr(self, "coarse_direct_final_height_var", None),
            "get", lambda: "340",
        )())
        coarse_direct_capture_only = bool(getattr(
            getattr(self, "coarse_direct_final_capture_only_var", None),
            "get", lambda: False,
        )())
        if mode == "hole_map_build":
            map_selection_mode = str(
                getattr(
                    getattr(self, "map_hole_selection_mode_var", None),
                    "get",
                    lambda: "manual",
                )()
            ).strip().lower()
            map_build_localization_mode = str(
                getattr(
                    getattr(self, "map_build_localization_mode_var", None),
                    "get",
                    lambda: "coarse_only",
                )()
            ).strip().lower()
            map_build_description = (
                "逐孔执行340 mm粗定位和260 mm精定位，并保存每孔精定位参考；"
                if map_build_localization_mode == "per_hole" else
                "只执行340 mm粗定位并保存粗定位地图；"
            )
            auto_enabled = bool(
                getattr(getattr(self, "auto_sector_selection_var", None), "get", lambda: False)()
            )
            confirmation_text = (
                f"将对扇区 {self.sector_id_var.get().strip() or '-'} {map_build_description}"
                + (
                    "当前将直接使用自动扇区候选建图；"
                    if map_selection_mode == "auto" and auto_enabled else
                    "当前会在自动扇区候选范围内打开人工选孔窗口，按Enter确认后才开始建图；"
                    if map_selection_mode == "manual" and auto_enabled else
                    "当前将打开人工选孔窗口，按Enter确认后才开始建图；"
                )
                + "建图阶段不会执行最终安放动作；地图保存成功后机械臂自动回原点。"
            )
        elif mode == "hole_map_execute":
            confirmation_text = (
                f"将调用扇区 {self.sector_id_var.get().strip() or '-'} 的粗定位地图；"
                + (
                    f"会在{coarse_direct_height} mm直接使用点云中心，"
                    + (
                        "只做点云采集评估，不执行最终XY/Z动作；"
                        if coarse_direct_capture_only else
                        "最终XY只应用最新ChArUco纠偏；"
                    )
                    + "点云失败孔不再进入其它定位流程，并"
                    if coarse_direct_strategy else
                    "会启动相机并在本轮重新执行260 mm精定位，成功孔随后应用最新ChArUco XY纠偏并"
                )
                + "移动到完整最终点。若该扇区还没有地图，"
                "会进入首轮340 mm粗定位建图。"
                f"调用孔号：{self.hole_map_ids_var.get().strip() or '全部有效孔'}。"
                + (
                    "已开启自动进入下一孔，孔间不等待人工确认；请确认运行区域安全。"
                    if auto_next_hole else
                    "当前孔完成后会等待人工确认，点击“开始执行下一个地图孔”才会继续。"
                )
            )
        elif mode == "hole_map_repair":
            confirmation_text = (
                f"将对扇区 {self.sector_id_var.get().strip() or '-'} 的孔 "
                f"{self.repair_hole_id_var.get().strip() or '-'} 重新执行粗定位和精定位；"
                "共享联合XY、点云融合和旧缓存都会关闭，不执行最终安放动作。"
                "成功后只替换该孔并生成新的地图版本，原地图保留可回退。"
            )
        else:
            confirmation_text = (
                (
                    f"将执行回原点及{coarse_direct_height} mm点云中心直达流程；"
                    + (
                        "仅采集评估，不执行最终XY/Z动作；"
                        if coarse_direct_capture_only else ""
                    )
                    + "点云失败孔只记录失败，不进入其它定位流程，每轮完成后会自动回原点并进入下一轮选孔，"
                    if coarse_direct_strategy else
                    "将执行回原点及两阶段定位；每轮完成后会自动回原点并进入下一轮选孔，"
                )
                + "相机和机器人会话保持运行。按选孔窗口 Esc 可结束会话。\n"
                + (
                    "已启用自动检测下一孔：当前轮已选孔之间将连续移动，不再逐孔等待人工确认；"
                    "请确认运行区域安全，并随时准备使用急停。\n"
                    if auto_next_hole else
                    "当前轮每个孔完成后需要人工确认，确认后才会移动到下一个孔。\n"
                )
            )
        if execute and not messagebox.askyesno(
            "确认真实运动",
            (
                "将执行回原点、单孔粗定位、下降和横向偏移采集。"
                f"{offset_confirmation}\n"
                "请在相机窗口中只选择一个孔，之后测试会自动运行。\n\n确认开始吗？"
                if mode == "offset" else
                confirmation_text + "\n确认开始吗？"
            ),
            parent=self,
        ):
            return
        self.process_mode = mode
        self._clear_log()
        self.result_var.set(
            "偏移容忍度测试进行中…" if mode == "offset" else
            "孔位地图调用进行中…" if mode == "hole_map_execute" else
            "孔位地图单孔返修进行中…" if mode == "hole_map_repair" else
            "孔位地图建立进行中…" if mode == "hole_map_build" else
            "本次定位进行中…"
        )
        self.status_var.set(
            "正在启动偏移测试进程…" if mode == "offset" else
            "正在启动孔位地图调用…" if mode == "hole_map_execute" else
            "正在启动孔位地图单孔返修…" if mode == "hole_map_repair" else
            "正在启动孔位地图建立…" if mode == "hole_map_build" else
            "正在启动定位进程…"
        )
        self.motion_var.set("无待确认运动")
        self.waiting_confirmation = False
        self.confirmation_kind = ""
        self.confirm_btn.configure(text="开始检测下一个孔")
        self.confirm_btn.configure(state=tk.DISABLED)
        self.mark_error_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.DISABLED)
        if hasattr(self, "build_map_btn"):
            self.build_map_btn.configure(state=tk.DISABLED)
        if hasattr(self, "execute_map_btn"):
            self.execute_map_btn.configure(state=tk.DISABLED)
        if hasattr(self, "repair_map_btn"):
            self.repair_map_btn.configure(state=tk.DISABLED)
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
            if hasattr(self, "build_map_btn"):
                self.build_map_btn.configure(state=tk.NORMAL)
            if hasattr(self, "execute_map_btn"):
                self.execute_map_btn.configure(state=tk.NORMAL)
            if hasattr(self, "repair_map_btn"):
                self.repair_map_btn.configure(state=tk.NORMAL)
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
        elif confirmation_kind == "hole" and self.process_mode == "hole_map_execute":
            self.motion_var.set("已确认，开始执行下一个地图孔…")
        elif confirmation_kind == "hole" and self.process_mode == "hole_map_build":
            self.motion_var.set("已确认，开始建立下一个地图孔…")
        else:
            self.motion_var.set("已确认，开始检测下一个孔…")

    def _handle_next_hole_confirmation(self, payload: str) -> None:
        """处理孔间确认；建图和地图调用默认暂停，自动模式只发送一次 m。"""
        if self.process_mode not in {"two_stage", "hole_map_build", "hole_map_execute"}:
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
        )
        is_map_mode = self.process_mode in {"hole_map_build", "hole_map_execute"}
        next_hole_label = "地图孔" if is_map_mode else "检测孔"
        if auto_next_hole:
            self.confirm_btn.configure(state=tk.DISABLED, text=f"自动进入下一{next_hole_label}…")
            self.motion_var.set(f"当前孔已完成，正在自动进入下一{next_hole_label}…")
            self.status_var.set(f"当前孔完成，正在自动移动到下一{next_hole_label}…")
            self._append_log(
                (
                    "[GUI] 已启用自动检测下一孔（地图模式为自动执行下一地图孔），"
                    "自动发送 m，当前轮将连续执行。\n"
                    if is_map_mode else
                    "[GUI] 已启用自动检测下一孔，自动发送 m，当前轮将连续执行。\n"
                )
            )
            # confirm_motion 会检查子进程仍在运行，并在成功发送后清除等待状态。
            self.confirm_motion()
            if self.process is not None and self.process.poll() is None:
                self.status_var.set(f"已自动进入下一{next_hole_label}，定位流程运行中…")
        else:
            button_text = (
                "开始执行下一个地图孔"
                if self.process_mode == "hole_map_execute" else "开始检测下一个孔"
            )
            if self.process_mode == "hole_map_build":
                button_text = "开始建立下一个地图孔"
            self.confirm_btn.configure(state=tk.NORMAL, text=button_text)
            self.motion_var.set(f"等待开始下一个{next_hole_label}：请点击“{button_text}”继续。")

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
        if hasattr(self, "build_map_btn"):
            self.build_map_btn.configure(state=tk.NORMAL)
        if hasattr(self, "execute_map_btn"):
            self.execute_map_btn.configure(state=tk.NORMAL)
        if hasattr(self, "repair_map_btn"):
            self.repair_map_btn.configure(state=tk.NORMAL)
        self.process = None
        if code == 0:
            self.status_var.set(
                "偏移测试完成" if self.process_mode == "offset" else
                "孔位地图调用完成" if self.process_mode == "hole_map_execute" else
                "孔位地图单孔返修完成" if self.process_mode == "hole_map_repair" else
                "孔位地图建立完成" if self.process_mode == "hole_map_build" else
                "定位完成"
            )
            self._show_latest_result()
        else:
            self.status_var.set(f"定位结束，退出码={code}")
            self.result_var.set("本次定位未通过质量门或被取消；请查看运行日志和结果目录。")
            if self.process_mode != "offset":
                reports = sorted(
                    (path for path in RUNS_DIR.glob("two-stage-*/report.json")
                     if path.stat().st_mtime >= self.run_started_at - 2.0),
                    key=lambda path: path.stat().st_mtime, reverse=True,
                )
                if reports:
                    try:
                        report = json.loads(reports[0].read_text(encoding="utf-8"))
                        failure_labels = {
                            "camera_stream_unavailable": "相机取帧故障，本轮已停止",
                            "robot_motion_failed": "机器人运动故障，本轮已停止",
                        }
                        if report.get("failure_type") in failure_labels:
                            self.status_var.set(failure_labels[report["failure_type"]])
                            self.result_var.set(f"{report.get('error', '')}\n报告：{reports[0]}")
                    except (OSError, ValueError):
                        pass

    def _show_latest_result(self) -> None:
        if self.process_mode == "hole_map_execute":
            reports = sorted(
                (
                    path for path in RUNS_DIR.glob("two-stage-*/report.json")
                    if path.stat().st_mtime >= self.run_started_at - 2.0
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if not reports:
                self.result_var.set("孔位地图调用完成，但未找到本次调用报告。")
                return
            try:
                report = json.loads(reports[0].read_text(encoding="utf-8"))
                holes = (report.get("final_result") or {}).get("holes") or []
                lines = [
                    f"粗地图：{(report.get('hole_map_execution') or {}).get('map_path', '-') }\n"
                    f"调用孔号：{[item.get('hole_id') for item in holes]}    "
                    f"完成：{sum(1 for item in holes if item.get('status') == 'completed')}",
                    f"调用报告：{reports[0].parent}",
                ]
                motion_distance = (
                    (report.get("timing") or {}).get("motion_distance")
                    or report.get("motion_distance")
                    or {}
                )
                motion_count = int(motion_distance.get("segment_count", 0) or 0)
                if motion_count > 0:
                    def format_distance(value: Any) -> str:
                        distance_mm = float(value or 0.0)
                        if abs(distance_mm) >= 1000.0:
                            return f"{distance_mm:.1f} mm ({distance_mm / 1000.0:.3f} m)"
                        return f"{distance_mm:.1f} mm"

                    lines.extend([
                        "机械臂运动距离（不计入视觉耗时）：",
                        f"  总路径：{format_distance(motion_distance.get('total_path_mm'))}    "
                        f"水平：{format_distance(motion_distance.get('horizontal_distance_mm'))}",
                        f"  上升：{format_distance(motion_distance.get('vertical_up_mm'))}    "
                        f"下降：{format_distance(motion_distance.get('vertical_down_mm'))}    "
                        f"净Z：{format_distance(motion_distance.get('vertical_net_mm'))}",
                        f"  统计段数：{motion_count}",
                    ])
                else:
                    lines.append("机械臂运动距离：暂无可用的运动端点记录。")
                for item in holes:
                    lines.append(
                        f"孔 {item.get('hole_id', '-')}：{item.get('status', '-')} "
                        f"精定位状态={item.get('fine_quality_status', '-')}"
                    )
                self.result_var.set("\n".join(lines))
            except Exception as exc:
                self.result_var.set(f"读取孔位地图调用报告失败：{exc}")
            return
        if self.process_mode == "hole_map_repair":
            reports = sorted(
                (
                    path for path in RUNS_DIR.glob("two-stage-*/report.json")
                    if path.stat().st_mtime >= self.run_started_at - 2.0
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if not reports:
                self.result_var.set("单孔返修完成，但未找到本次报告。")
                return
            try:
                report = json.loads(reports[0].read_text(encoding="utf-8"))
                repair = report.get("map_repair") or {}
                hole_map = report.get("hole_map") or {}
                lines = [
                    f"单孔返修：S{int(repair.get('sector_id', 0)):02d}-H{int(repair.get('hole_id', 0)):02d}\n"
                    f"方式：重新粗/精定位（不执行最终安放）    状态：{report.get('status', '-')}\n"
                    f"新地图：{hole_map.get('path', repair.get('new_map_path', '-'))}\n"
                    f"当前入口：{hole_map.get('current_path', '未更新')}\n"
                    f"详细报告：{reports[0].parent}",
                ]
                motion_distance = (
                    (report.get("timing") or {}).get("motion_distance")
                    or report.get("motion_distance")
                    or {}
                )
                if int(motion_distance.get("segment_count", 0) or 0) > 0:
                    lines.append(
                        "机械臂运动距离："
                        f"总路径 {float(motion_distance.get('total_path_mm', 0.0)):.1f} mm，"
                        f"上升 {float(motion_distance.get('vertical_up_mm', 0.0)):.1f} mm，"
                        f"下降 {float(motion_distance.get('vertical_down_mm', 0.0)):.1f} mm"
                    )
                self.result_var.set("\n".join(lines))
                if hole_map.get("current_path"):
                    self.hole_map_path_var.set(str(hole_map["current_path"]))
                self._refresh_repair_hole_choices()
            except Exception as exc:
                self.result_var.set(f"读取单孔返修报告失败：{exc}")
            return
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
            summary_path = reports[0].parent / "result_summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary_status = summary.get("status") or {}
                summary_timing = summary.get("timing") or {}
                visual_timing = summary_timing.get("visual_timing") or {}
                if not visual_timing:
                    visual_timing = {
                        "total_visual_s": summary_timing.get(
                            "visual_total_s",
                            (summary_timing.get("task_breakdown") or {}).get(
                                "vision_compute_s", 0.0,
                            ),
                        ),
                        "stage_breakdown": {},
                    }
                visual_session_s = float(
                    summary_timing.get(
                        "visual_session_s",
                        (summary_timing.get("visual_session_timing") or {}).get(
                            "total_visual_s", visual_timing.get("total_visual_s", 0.0)
                        ),
                    )
                )
                motion_distance = summary_timing.get("motion_distance") or {}
                visual_stage_labels = {
                    "camera_startup": "相机启动",
                    "frame_acquisition": "RGB/RGB-D取帧",
                    "yolo_inference": "YOLO检测/关联",
                    "pointcloud_processing": "点云/平面几何",
                    "ellipse_fitting": "椭圆/中心拟合",
                    "coordinate_transform": "像素到基坐标",
                    "fusion_quality": "融合/质量判断",
                    "final_point_calculation": "最终点计算",
                    "visual_pipeline_other": "其它视觉流水线",
                }
                visual_lines = []
                for stage, label in visual_stage_labels.items():
                    value = float(visual_timing.get(f"{stage}_s", 0.0))
                    if value <= 0.0:
                        value = float(
                            (visual_timing.get("stage_breakdown") or {}).get(stage, 0.0)
                        )
                    if value > 0.0:
                        visual_lines.append(f"  {label}：{value:.2f} s")
                lines = [
                    f"简明报告：{summary_path}",
                    f"本轮状态：{summary_status.get('cycle_status', '-') }    "
                    f"会话结束：{summary_status.get('session_end_reason') or summary_status.get('session_status', '-')}",
                    f"孔数：{summary_status.get('selected_holes', 0)}    "
                    f"成功：{summary_status.get('completed_holes', 0)}    "
                    f"延后/失败：{summary_status.get('deferred_holes', 0)}",
                    f"视觉生产节拍（不含机械臂运动/等待）：{float(visual_timing.get('total_visual_s', 0.0)):.2f} s    "
                    f"平均：{float(visual_timing.get('total_visual_s', 0.0)) / max(1, int(summary_status.get('selected_holes', 0))):.2f} s/孔",
                    f"视觉会话总耗时（含相机启动）：{visual_session_s:.2f} s",
                    "视觉阶段拆分：",
                    *visual_lines,
                ]
                motion_count = int(motion_distance.get("segment_count", 0) or 0)
                if motion_count > 0:
                    def format_distance(value: Any) -> str:
                        distance_mm = float(value or 0.0)
                        if abs(distance_mm) >= 1000.0:
                            return f"{distance_mm:.1f} mm ({distance_mm / 1000.0:.3f} m)"
                        return f"{distance_mm:.1f} mm"

                    lines.extend([
                        "",
                        "机械臂运动距离（不计入视觉耗时）：",
                        f"  总路径：{format_distance(motion_distance.get('total_path_mm'))}    "
                        f"水平：{format_distance(motion_distance.get('horizontal_distance_mm'))}",
                        f"  上升：{format_distance(motion_distance.get('vertical_up_mm'))}    "
                        f"下降：{format_distance(motion_distance.get('vertical_down_mm'))}    "
                        f"净Z：{format_distance(motion_distance.get('vertical_net_mm'))}",
                        f"  统计段数：{motion_count}    "
                        f"口径：{motion_distance.get('distance_basis', 'planned_tcp_waypoint_delta')}",
                    ])
                else:
                    lines.extend([
                        "",
                        "机械臂运动距离：暂无可用的运动端点记录（旧报告可能未启用此统计）。",
                    ])
                lines.extend([
                    "",
                    "孔号    来源                  孔自身视觉(s)  共享视觉(s)  合计(s)",
                ])
                for item in summary.get("holes") or []:
                    lines.append(
                        f"H{int(item.get('hole_id', 0)):02d}     "
                        f"{str(item.get('source', '-')):<21} "
                        f"{float(item.get('exclusive_visual_s', item.get('exclusive_time_s', 0.0))):>11.2f}  "
                        f"{float(item.get('shared_visual_s', item.get('shared_allocated_time_s', 0.0))):>9.2f}  "
                        f"{float(item.get('attributed_visual_s', item.get('attributed_total_s', 0.0))):>7.2f}"
                    )
                lines.append(f"详细报告：{reports[0]}")
                self.result_var.set("\n".join(lines))
                return
            final = report.get("final_result", {})
            holes = final.get("holes")
            if isinstance(holes, list) and holes:
                deferred_count = int(final.get("deferred_count", 0) or 0)
                completed_count = int(final.get("completed_count", len(holes) - deferred_count) or 0)
                capture_only_count = int(final.get("capture_only_count", 0) or 0)
                if deferred_count:
                    self.status_var.set(
                        f"定位完成，但有 {deferred_count} 个孔延期，成功 {completed_count} 个"
                    )
                elif capture_only_count and not completed_count:
                    self.status_var.set(
                        f"点云采集评估完成，共 {capture_only_count} 个孔，未执行最终动作"
                    )
                lines = [
                    f"报告：{reports[0].parent}",
            f"批量处理孔数：{final.get('hole_count', len(holes))}    "
                    f"成功：{completed_count}    仅采集：{capture_only_count}    延期：{deferred_count}    "
                    f"最终 TCP：{self._format_vector(final.get('final_tcp_pose_m_rad'))}",
                ]
                map_summary = report.get("hole_map") or {}
                if map_summary:
                    if self.process_mode == "hole_map_build":
                        map_build_mode = str(
                            map_summary.get("map_build_localization_mode", "coarse_only")
                        ).strip().lower()
                        if map_build_mode == "per_hole":
                            lines.append(
                                "建图内容：每孔保存340mm粗定位和260mm精定位参考；最终TCP动作不写入地图。"
                            )
                            lines.append(
                                f"逐孔精定位参考：有效={map_summary.get('fine_reference_ready_holes', [])}，"
                                f"缺失={map_summary.get('fine_reference_missing_holes', [])}"
                            )
                        else:
                            lines.append(
                                "建图内容：仅保存本轮340mm粗定位；260mm精定位/最终TCP不写入地图。"
                            )
                    lines.append(
                        f"孔位地图：{map_summary.get('path', '-')}，"
                        f"有效孔={map_summary.get('ready_holes', [])}，"
                        f"延期孔={map_summary.get('deferred_holes', [])}"
                    )
                    if map_summary.get("ready_sector_ids") is not None:
                        lines.append(
                            f"六扇区已建地图：{map_summary.get('ready_sector_ids', [])}，"
                            f"本次更新扇区={map_summary.get('sector_id', '-')}"
                        )
                    current_path = map_summary.get("current_path")
                    if current_path:
                        self.hole_map_path_var.set(str(current_path))
                        lines.append(f"当前可调用地图：{current_path}")
                    elif self.process_mode == "hole_map_build":
                        reason = map_summary.get("current_update_blocked_reason")
                        lines.append(
                            "本次地图为部分结果，未替换已有完整当前地图。"
                            if not reason else
                            f"当前入口未更新：{reason}。"
                        )
                    artifacts = map_summary.get("artifacts") or {}
                    if artifacts.get("preview_jpg"):
                        lines.append(
                            f"点云预览：{Path(map_summary.get('path', '')).parent / str(artifacts['preview_jpg'])}"
                        )
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
                    coarse_height = batch_coarse.get("target_height_mm")
                    coarse_height_text = (
                        f"{float(coarse_height):g}mm"
                        if coarse_height is not None else "配置高度"
                    )
                    lines.append(
                        f"{coarse_height_text}共享粗定位：组数={len(groups)}，"
                        "每组使用本组孔位的综合中心拍摄位姿"
                    )
                    grouping_visualization = batch_coarse.get("grouping_visualization") or {}
                    if grouping_visualization.get("jpg"):
                        lines.append(f"粗定位分组可视化：{grouping_visualization['jpg']}")
                    for group in groups:
                        lines.append(
                            f"  粗定位组{group.get('group_index', '-')}: "
                            f"孔={group.get('hole_ids', [])} "
                            f"综合中心={self._format_vector(group.get('combined_point_base_mm'))} "
                            f"共享拍摄位姿={self._format_vector(group.get('target_tcp_pose_m_rad'))}"
                        )
                        capture_visualization = group.get("capture_grouping_visualization") or {}
                        if capture_visualization.get("jpg"):
                            lines.append(f"    实际拍摄分组图：{capture_visualization['jpg']}")
                if batch_fine.get("enabled"):
                    groups = batch_fine.get("groups") or []
                    lines.append(
                        f"260mm共享拍摄：组数={len(groups)}，每组一个相机位姿；"
                        "孔结果仅为基坐标三维点"
                    )
                    per_hole_fallback = batch_fine.get("per_hole_fallback_holes") or []
                    if per_hole_fallback:
                        lines.append(
                            f"共享精拍质量门未通过，转逐孔精定位：{per_hole_fallback}"
                        )
                    grouping_visualization = batch_fine.get("grouping_visualization") or {}
                    if grouping_visualization.get("jpg"):
                        lines.append(f"精定位分组可视化：{grouping_visualization['jpg']}")
                    for group in groups:
                        lines.append(
                            f"  组{group.get('group_index', '-')}: "
                            f"孔={group.get('hole_ids', [])} "
                            f"综合中心={self._format_vector(group.get('combined_point_base_mm'))} "
                            f"共享拍摄位姿={self._format_vector(group.get('target_tcp_pose_m_rad'))}"
                        )
                        capture_visualization = group.get("capture_grouping_visualization") or {}
                        if capture_visualization.get("jpg"):
                            lines.append(f"    实际拍摄分组图：{capture_visualization['jpg']}")
                        for supplement in group.get("supplement_captures") or []:
                            lines.append(
                                f"    共享补拍{supplement.get('round', '-')}: "
                                f"孔={supplement.get('hole_ids', [])} "
                                f"通过={supplement.get('accepted_holes', [])} "
                                f"未通过={supplement.get('fallback_holes', [])}"
                            )
                            supplement_visualization = supplement.get("capture_grouping_visualization") or {}
                            if supplement_visualization.get("jpg"):
                                lines.append(f"      补拍分组图：{supplement_visualization['jpg']}")
                raw_motion = (report.get("timing") or {}).get("motion_distance") or {}
                if int(raw_motion.get("segment_count", 0) or 0) > 0:
                    def format_distance(value: Any) -> str:
                        distance_mm = float(value or 0.0)
                        if abs(distance_mm) >= 1000.0:
                            return f"{distance_mm:.1f} mm ({distance_mm / 1000.0:.3f} m)"
                        return f"{distance_mm:.1f} mm"

                    lines.extend([
                        "机械臂运动距离（不计入视觉耗时）：",
                        f"  总路径：{format_distance(raw_motion.get('total_path_mm'))}    "
                        f"水平：{format_distance(raw_motion.get('horizontal_distance_mm'))}",
                        f"  上升：{format_distance(raw_motion.get('vertical_up_mm'))}    "
                        f"下降：{format_distance(raw_motion.get('vertical_down_mm'))}    "
                        f"净Z：{format_distance(raw_motion.get('vertical_net_mm'))}",
                    ])
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

    def open_sector_info_dir(self) -> None:
        HOLE_LOCALIZATION_SECTOR_INFO_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(HOLE_LOCALIZATION_SECTOR_INFO_DIR))  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("打开扇区信息目录失败", str(exc), parent=self)

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
