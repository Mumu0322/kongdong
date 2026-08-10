#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""工作台中的 CAD 配准、孔定位、CAD 运动和视野偏移测试页面。

定位计算继续复用两个独立诊断入口，但都以后台子进程运行：
Tk 主线程不会被相机、YOLO 或机器人运动等待阻塞。流程仅在开始检测下一个已选孔时暂停，
由本页面的“开始检测下一个孔”按钮发送继续指令，其余运动自动执行。
"""

from __future__ import annotations

import json
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

from .paths import (
    CAD_MODEL_PATH,
    CAD_MOTION_RUNS_DIR,
    CAD_REGISTRATION_RUNS_DIR,
    HANDEYE_CANDIDATE_PATH,
    HOLE_LOCALIZATION_RUNS_DIR,
    CAMERA_CALIBRATION_PATH,
    MODEL_PATH,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOCALIZATION_SCRIPT = PROJECT_DIR / "run_yolo_eye_in_hand_optimized.py"
OFFSET_TEST_SCRIPT = PROJECT_DIR / "run_coarse_to_fine_offset_test.py"
CAD_REGISTRATION_SCRIPT = PROJECT_DIR / "run_cad_registration_preview.py"
DEFAULT_MODEL = MODEL_PATH
DEFAULT_HANDEYE = HANDEYE_CANDIDATE_PATH
DEFAULT_CAD_MODEL = CAD_MODEL_PATH
DEFAULT_INTRINSICS = CAMERA_CALIBRATION_PATH
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
    """CAD 配准、CAD 运动、两阶段 YOLO 定位和单孔视野偏移测试 GUI。"""

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
        self.cad_model_var = tk.StringVar(value=str(DEFAULT_CAD_MODEL))
        self.registration_intrinsics_var = tk.StringVar(value=str(DEFAULT_INTRINSICS))
        self.registration_step_var = tk.StringVar(value="")
        self.registration_image_dir_var = tk.StringVar(value="")
        self.registration_mapping_var = tk.StringVar(value="")
        self.registration_frames_var = tk.StringVar(value="5")
        self.registration_match_distance_var = tk.StringVar(value="80")
        self.registration_rmse_var = tk.StringVar(value="1.5")
        self.registration_max_error_var = tk.StringVar(value="3.0")
        self.registration_cross_frame_p95_var = tk.StringVar(value="1.5")
        self.registration_min_valid_var = tk.StringVar(value="3")
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
        self.cad_fine_height_var = tk.StringVar(value="260")
        self.cad_depth_height_var = tk.StringVar(value="340")
        self.cad_depth_frames_var = tk.StringVar(value="8")
        self.cad_depth_min_valid_var = tk.StringVar(value="5")
        self.cad_depth_min_holes_var = tk.StringVar(value="4")
        self.cad_depth_plane_rmse_var = tk.StringVar(value="3.5")
        self.cad_depth_height_p95_var = tk.StringVar(value="2.0")
        self.cad_group_view_margin_var = tk.StringVar(value="35")
        self.cad_fine_frames_var = tk.StringVar(value="12")
        self.cad_fine_min_valid_var = tk.StringVar(value="6")
        self.cad_center_p95_var = tk.StringVar(value="1.5")
        self.cad_match_tolerance_var = tk.StringVar(value="70")
        self.cad_settle_discard_var = tk.StringVar(value="10")
        self.offset_radii_var = tk.StringVar(value="0 5 10 15 20")
        self.offset_angles_var = tk.StringVar(value="0 45 90 135 180 225 270 315")
        self.execute_var = tk.BooleanVar(value=False)
        self.experimental_var = tk.BooleanVar(value=False)
        self.cad_workspace_checked_var = tk.BooleanVar(value=False)
        self.final_xy_var = tk.BooleanVar(value=False)
        self.final_target_mode_var = tk.StringVar(value="机械爪模式")
        self.include_final_motion_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="待开始：默认仅预览，不会下发机器人运动")
        self.motion_var = tk.StringVar(value="无待确认运动")
        self.result_var = tk.StringVar(value="尚无本次结果")

        self._build()
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
        registration_page = ScrollableTab(self.notebook, padding=10)
        cad_page = ScrollableTab(self.notebook, padding=10)
        legacy_page = ScrollableTab(self.notebook, padding=10)
        monitor_page = ScrollableTab(self.notebook, padding=10)
        self.notebook.add(registration_page, text="CAD 配准预览")
        self.notebook.add(cad_page, text="CAD 运动（主流程）")
        self.notebook.add(legacy_page, text="旧定位 / 偏移测试")
        self.notebook.add(monitor_page, text="运行监控")

        self._build_registration_tab(registration_page.inner)
        self._build_cad_tab(cad_page.inner)
        self._build_legacy_tab(legacy_page.inner)
        self._build_monitor_tab(monitor_page.inner)
        for page in (registration_page, cad_page, legacy_page, monitor_page):
            page.bind_mousewheel()

        # 这组按钮始终可见，避免用户在 CAD 外部相机窗口等待时找不到确认/停止操作。
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
        ttk.Button(control, text="打开 CAD 结果目录", command=self.open_cad_results_dir).grid(row=0, column=4, padx=(0, 6))
        ttk.Button(
            control, text="打开最新精定位图", command=self.open_latest_cad_fine_visualization,
        ).grid(row=0, column=5)
        ttk.Label(control, textvariable=self.motion_var, foreground="#a35a00").grid(
            row=1, column=0, columnspan=7, sticky="w", pady=(5, 0),
        )

    def _build_registration_tab(self, parent: ttk.Frame) -> None:
        files = ttk.LabelFrame(parent, text="CAD 配准输入（实时模式只读取 RGB 和 TCP，不会运动）", padding=8)
        files.pack(fill=tk.X)
        files.columnconfigure(1, weight=1)
        self._path_row(files, 0, "RGB 内参", self.registration_intrinsics_var, self._browse_registration_intrinsics)
        self._path_row(files, 1, "YOLO 模型", self.model_var, self._browse_model)
        self._path_row(files, 2, "手眼结果（只读 TCP）", self.handeye_var, self._browse_handeye)
        self._path_row(files, 3, "CAD 模型", self.cad_model_var, self._browse_cad_model)
        self._path_row(files, 4, "STEP（可空，填写后重新解析）", self.registration_step_var, self._browse_registration_step)
        self._path_row(files, 5, "离线图片目录（可空=实时 RGB）", self.registration_image_dir_var, self._browse_registration_image_dir)
        self._path_row(files, 6, "人工映射 JSON（可空=窗口鼠标映射）", self.registration_mapping_var, self._browse_registration_mapping)

        params = ttk.LabelFrame(parent, text="配准质量门", padding=8)
        params.pack(fill=tk.X, pady=(8, 0))
        self._grid_fields(params, [
            ("帧数（实时填 1 或 5）", self.registration_frames_var, 7),
            ("匹配距离 px", self.registration_match_distance_var, 7),
            ("单帧 RMSE px", self.registration_rmse_var, 7),
            ("单帧最大误差 px", self.registration_max_error_var, 7),
            ("跨帧中心 P95 mm", self.registration_cross_frame_p95_var, 7),
            ("最少有效帧", self.registration_min_valid_var, 7),
            ("YOLO 置信度", self.confidence_var, 7),
        ], columns=3)

        actions = ttk.LabelFrame(parent, text="CAD 配准操作", padding=8)
        actions.pack(fill=tk.X, pady=(8, 0))
        self.registration_start_btn = ttk.Button(
            actions, text="启动 CAD 配准预览", command=self.start_cad_registration,
        )
        self.registration_start_btn.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            actions, text="打开 CAD 配准结果", command=self.open_cad_registration_results_dir,
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(
            actions,
            text="实时采集：在弹出的窗口按 C/空格拍摄；每帧按 S/Enter 保存，R 重拍；完成后按窗口提示做人工映射。",
            foreground="#555555",
        ).pack(side=tk.LEFT)

        notice = ttk.LabelFrame(parent, text="配准流程", padding=8)
        notice.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(
            notice,
            text=("1. 先启动配准预览，保持机械臂和工件不动。\n"
                  "2. 采集 5 帧 RGB；若出现映射窗口，先点击左侧 YOLO 孔，再点击右侧 CAD 孔，至少完成 4 组，按 S/Enter 确认。\n"
                  "3. 检查 CAD 圆、孔号和重投影误差；通过后报告会自动填入“CAD 运动（主流程）”页。\n"
                  "4. 配准阶段不会实例化运动会话，也不会下发机器人运动。"),
            justify=tk.LEFT,
            wraplength=1050,
        ).pack(anchor="w")

    def _build_cad_tab(self, parent: ttk.Frame) -> None:
        files = ttk.LabelFrame(parent, text="CAD 运动输入", padding=8)
        files.pack(fill=tk.X)
        files.columnconfigure(1, weight=1)
        self._path_row(files, 0, "YOLO 模型", self.model_var, self._browse_model)
        self._path_row(files, 1, "手眼结果", self.handeye_var, self._browse_handeye)
        self._path_row(files, 2, "CAD 模型", self.cad_model_var, self._browse_cad_model)

        params = ttk.LabelFrame(parent, text="CAD 分组深度与精定位参数", padding=8)
        params.pack(fill=tk.X, pady=(8, 0))
        self._grid_fields(params, [
            ("共同深度高度 mm", self.cad_depth_height_var, 7),
            ("共同深度帧数", self.cad_depth_frames_var, 7),
            ("共同深度最少有效帧", self.cad_depth_min_valid_var, 7),
            ("多孔每帧最少有效孔", self.cad_depth_min_holes_var, 7),
            ("深度平面 RMSE mm", self.cad_depth_plane_rmse_var, 7),
            ("高度跨帧 P95 mm", self.cad_depth_height_p95_var, 7),
            ("视野边缘额外余量 px", self.cad_group_view_margin_var, 7),
            ("260 mm 高度", self.cad_fine_height_var, 7),
            ("采集帧数", self.cad_fine_frames_var, 7),
            ("最少有效帧", self.cad_fine_min_valid_var, 7),
            ("中心 P95 px", self.cad_center_p95_var, 7),
            ("匹配距离 px", self.cad_match_tolerance_var, 7),
            ("稳定丢弃帧", self.cad_settle_discard_var, 7),
        ], columns=3)

        options = ttk.LabelFrame(parent, text="CAD 运动选项", padding=8)
        options.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(options, text="最终点模式").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Combobox(
            options,
            textvariable=self.final_target_mode_var,
            values=("机械爪模式", "平常模式"),
            state="readonly",
            width=13,
        ).grid(row=0, column=1, sticky="w", padx=(0, 18))
        ttk.Checkbutton(
            options, text="真实运动（未勾选时只能预览）", variable=self.execute_var,
        ).grid(row=0, column=2, sticky="w", padx=(0, 18))
        ttk.Checkbutton(
            options, text="允许当前实验手眼结果", variable=self.experimental_var,
        ).grid(row=0, column=3, sticky="w")
        ttk.Checkbutton(
            options, text="精定位后执行 TCP XY → 基坐标 Z → +Y 0.3 mm",
            variable=self.final_xy_var,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            options,
            text="我已检查机器人活动范围、工件和急停条件（真实 CAD 运动必选）",
            variable=self.cad_workspace_checked_var,
        ).grid(row=1, column=3, sticky="w", pady=(6, 0))

        actions = ttk.LabelFrame(parent, text="CAD 操作", padding=8)
        actions.pack(fill=tk.X, pady=(8, 0))
        self.cad_preview_btn = ttk.Button(
            actions, text="CAD 目标预览（不运动）", command=self.start_cad_preview,
        )
        self.cad_preview_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.cad_motion_btn = ttk.Button(
            actions, text="启动 CAD 分组深度 + 260mm 运动", command=self.start_cad_motion,
        )
        self.cad_motion_btn.pack(side=tk.LEFT)
        ttk.Label(
            actions,
            text="真实运动：启动前自动采5帧 RGB 重配准；通过后回原点，点击1个或多个 CAD 圆，再逐孔到CAD 260 mm。",
            foreground="#555555",
        ).pack(side=tk.LEFT, padx=(16, 0))

    def _build_legacy_tab(self, parent: ttk.Frame) -> None:
        files = ttk.LabelFrame(parent, text="旧流程输入", padding=8)
        files.pack(fill=tk.X)
        files.columnconfigure(1, weight=1)
        self._path_row(files, 0, "YOLO 模型", self.model_var, self._browse_model)
        self._path_row(files, 1, "手眼结果", self.handeye_var, self._browse_handeye)

        two_stage = ttk.LabelFrame(parent, text="两阶段定位参数", padding=8)
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

        offset = ttk.LabelFrame(parent, text="偏移容忍度测试参数", padding=8)
        offset.pack(fill=tk.X, pady=(8, 0))
        self._grid_fields(offset, [
            ("偏移测试半径 mm", self.offset_radii_var, 24),
            ("偏移测试方向 °", self.offset_angles_var, 38),
        ], columns=2)

        options = ttk.LabelFrame(parent, text="旧流程选项", padding=8)
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
        ttk.Checkbutton(options, text="精定位后执行 TCP XY → 基坐标 Z → +Y 0.3 mm", variable=self.final_xy_var).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 0),
        )
        ttk.Checkbutton(
            options, text="偏移测试执行最终 XY → Z → +Y 0.3 mm（实机）",
            variable=self.include_final_motion_var,
        ).grid(row=1, column=3, sticky="w", pady=(6, 0))

        actions = ttk.LabelFrame(parent, text="旧流程操作", padding=8)
        actions.pack(fill=tk.X, pady=(8, 0))
        self.start_btn = ttk.Button(actions, text="开始两阶段定位", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.offset_start_btn = ttk.Button(
            actions, text="偏移容忍度测试（单孔）", command=self.start_offset_test,
        )
        self.offset_start_btn.pack(side=tk.LEFT)
        ttk.Label(
            actions, text="旧流程只在需要兼容历史点云方案时使用。", foreground="#555555",
        ).pack(side=tk.LEFT, padx=(16, 0))

    def _build_monitor_tab(self, parent: ttk.Frame) -> None:
        notice = ttk.LabelFrame(parent, text="操作提示", padding=8)
        notice.pack(fill=tk.X)
        ttk.Label(
            notice,
            text=("CAD：真实运动启动前自动用当前位置采集5帧 RGB 并重新配准；通过后回原点，在 CAD 画面中点击一个或多个目标孔，按 Enter/Space 确认。\n"
                  "随后整组孔共同移动到 340 mm；单孔测该孔高度，多孔融合共享高度，再逐孔使用 CAD 260 mm 位姿；每个孔完成后点击“开始下一个 CAD 孔”。\n"
                  "每个孔在 260 mm 会保存 CAD 投影圆、YOLO 框、中心误差箭头和多帧中位数图；运行结束后可点“打开最新精定位图”。\n"
                  "旧流程：相机初始画面中选择孔后按 Enter；每个孔完成后点击继续。如需急停，请使用机械臂示教器。"),
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
    def _grid_fields(parent: ttk.Widget, fields: list[tuple[str, tk.StringVar, int]], columns: int = 3) -> None:
        for index, (label, variable, width) in enumerate(fields):
            row = index // columns
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

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="选择 YOLO 模型", filetypes=[("模型", "*.pt"), ("所有文件", "*.*")])
        if path:
            self.model_var.set(path)

    def _browse_handeye(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="选择手眼结果", filetypes=[("JSON", "*.json"), ("所有文件", "*.*")])
        if path:
            self.handeye_var.set(path)

    def _browse_registration_intrinsics(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="选择 RGB 内参 JSON",
            filetypes=[("JSON", "*.json"), ("所有文件", "*.*")],
        )
        if path:
            self.registration_intrinsics_var.set(path)

    def _browse_registration_step(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="选择 STEP（可取消，使用已有 CAD JSON）",
            filetypes=[("STEP", "*.step *.stp"), ("所有文件", "*.*")],
        )
        if path:
            self.registration_step_var.set(path)

    def _browse_registration_image_dir(self) -> None:
        path = filedialog.askdirectory(parent=self, title="选择离线 RGB 图片目录（可取消使用实时模式）")
        if path:
            self.registration_image_dir_var.set(path)

    def _browse_registration_mapping(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="选择人工映射 JSON（可取消在窗口中点选）",
            filetypes=[("JSON", "*.json"), ("所有文件", "*.*")],
        )
        if path:
            self.registration_mapping_var.set(path)

    def _browse_cad_model(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="选择 CAD 孔位模型",
            filetypes=[("JSON", "*.json"), ("所有文件", "*.*")],
        )
        if path:
            self.cad_model_var.set(path)

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
            cad_defaults: dict[str, tuple[str, float | int, Callable[[str], float | int]]] = {
                "cad_fine_height": ("cad_fine_height_var", 260.0, float),
                "cad_depth_height": ("cad_depth_height_var", 340.0, float),
                "cad_depth_frames": ("cad_depth_frames_var", 8, int),
                "cad_depth_min_valid": ("cad_depth_min_valid_var", 5, int),
                "cad_depth_min_holes": ("cad_depth_min_holes_var", 4, int),
                "cad_depth_plane_rmse": ("cad_depth_plane_rmse_var", 3.5, float),
                "cad_depth_height_p95": ("cad_depth_height_p95_var", 2.0, float),
                "cad_group_view_margin": ("cad_group_view_margin_var", 35.0, float),
                "cad_fine_frames": ("cad_fine_frames_var", 12, int),
                "cad_fine_min_valid": ("cad_fine_min_valid_var", 6, int),
                "cad_center_p95": ("cad_center_p95_var", 1.5, float),
                "cad_match_tolerance": ("cad_match_tolerance_var", 70.0, float),
                "cad_settle_discard": ("cad_settle_discard_var", 10, int),
                "registration_frames": ("registration_frames_var", 5, int),
                "registration_match_distance": ("registration_match_distance_var", 80.0, float),
                "registration_rmse": ("registration_rmse_var", 1.5, float),
                "registration_max_error": ("registration_max_error_var", 3.0, float),
                "registration_cross_frame_p95": ("registration_cross_frame_p95_var", 1.5, float),
                "registration_min_valid": ("registration_min_valid_var", 3, int),
            }
            for key, (attr_name, default, converter) in cad_defaults.items():
                variable = getattr(self, attr_name, None)
                numbers[key] = default if variable is None else converter(variable.get())
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

    def _build_cad_registration_command(self) -> list[str]:
        if not CAD_REGISTRATION_SCRIPT.is_file():
            raise FileNotFoundError(f"找不到 CAD 配准脚本：{CAD_REGISTRATION_SCRIPT}")
        values = self._numbers()
        intrinsics = Path(self.registration_intrinsics_var.get().strip())
        if not intrinsics.is_file():
            raise FileNotFoundError(f"RGB 内参不存在：{intrinsics}")
        model = Path(self.model_var.get().strip())
        if not model.is_file():
            raise FileNotFoundError(f"YOLO 模型不存在：{model}")
        handeye = Path(self.handeye_var.get().strip())
        if not handeye.is_file():
            raise FileNotFoundError(f"手眼结果不存在：{handeye}")
        cad_model = Path(self.cad_model_var.get().strip())
        step_text = self.registration_step_var.get().strip()
        step = Path(step_text) if step_text else None
        if step is not None and not step.is_file():
            raise FileNotFoundError(f"STEP 文件不存在：{step}")
        if step is None and not cad_model.is_file():
            raise FileNotFoundError(f"CAD 模型不存在：{cad_model}")
        image_dir_text = self.registration_image_dir_var.get().strip()
        image_dir = Path(image_dir_text) if image_dir_text else None
        if image_dir is not None and not image_dir.is_dir():
            raise FileNotFoundError(f"离线图片目录不存在：{image_dir}")
        if image_dir is None and int(values["registration_frames"]) not in {1, 5}:
            raise ValueError("实时 CAD 配准帧数只能填 1 或 5")
        mapping_text = self.registration_mapping_var.get().strip()
        mapping = Path(mapping_text) if mapping_text else None
        if mapping is not None and not mapping.is_file():
            raise FileNotFoundError(f"人工映射 JSON 不存在：{mapping}")
        connection = self.connection_provider()
        command = [
            sys.executable, "-u", str(CAD_REGISTRATION_SCRIPT),
            "--intrinsics", str(intrinsics),
            "--cad-model", str(cad_model),
            "--model", str(model),
            "--handeye", str(handeye),
            "--frames", str(values["registration_frames"]),
            "--confidence", str(values["confidence"]),
            "--match-distance-px", str(values["registration_match_distance"]),
            "--rmse-px", str(values["registration_rmse"]),
            "--max-error-px", str(values["registration_max_error"]),
            "--cross-frame-p95-mm", str(values["registration_cross_frame_p95"]),
            "--min-valid-frames", str(values["registration_min_valid"]),
            "--robot-ip", str(connection["ip"]), "--robot-port", str(connection["port"]),
            "--robot-user", str(connection["user"]), "--robot-password", str(connection["password"]),
            "--robot-timeout-ms", str(connection["timeout_ms"]),
        ]
        if step is not None:
            command.extend(["--step", str(step)])
        if image_dir is not None:
            command.extend(["--image-dir", str(image_dir)])
        if mapping is not None:
            command.extend(["--mapping-json", str(mapping)])
        return command

    def _build_command(self, mode: str = "two_stage", execute_override: bool | None = None) -> list[str]:
        if mode == "cad_registration":
            return self._build_cad_registration_command()
        is_cad = mode in {"cad_preview", "cad_motion"}
        is_offset = mode == "offset"
        script = LOCALIZATION_SCRIPT if (mode == "two_stage" or is_cad) else OFFSET_TEST_SCRIPT
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
        if is_cad:
            command.append("--cad-motion")
            command.append(
                "--cad-allow-experimental-handeye"
                if self.experimental_var.get() else "--require-validated-handeye"
            )
            cad_model = Path(self.cad_model_var.get().strip())
            if not cad_model.is_file():
                raise FileNotFoundError(f"CAD 模型不存在：{cad_model}")
            command.extend([
                "--cad-model-json", str(cad_model),
                "--cad-fine-height-mm", str(values["cad_fine_height"]),
                "--cad-depth-height-mm", str(values["cad_depth_height"]),
                "--cad-depth-frames", str(values["cad_depth_frames"]),
                "--cad-depth-min-valid", str(values["cad_depth_min_valid"]),
                "--cad-depth-min-holes", str(values["cad_depth_min_holes"]),
                "--cad-depth-plane-rmse-mm", str(values["cad_depth_plane_rmse"]),
                "--cad-depth-height-p95-mm", str(values["cad_depth_height_p95"]),
                "--cad-group-view-margin-px", str(values["cad_group_view_margin"]),
                "--cad-fine-frames", str(values["cad_fine_frames"]),
                "--cad-fine-min-valid", str(values["cad_fine_min_valid"]),
                "--cad-fine-center-p95-px", str(values["cad_center_p95"]),
                "--cad-yolo-match-tolerance-px", str(values["cad_match_tolerance"]),
                "--cad-settle-discard-frames", str(values["cad_settle_discard"]),
            ])
        elif is_offset:
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
        if is_cad:
            final_mode = {
                "机械爪模式": "gripper",
                "平常模式": "normal",
            }.get(self.final_target_mode_var.get())
            if final_mode is None:
                raise ValueError("最终点模式必须选择“机械爪模式”或“平常模式”")
            command.extend([
                "--final-target-mode", final_mode,
                "--move-final-xy" if self.final_xy_var.get() else "--no-move-final-xy",
            ])
        return command

    def start(self) -> None:
        self._start_process("two_stage")

    def start_cad_registration(self) -> None:
        # 配准阶段始终不执行机器人运动；脚本只读取 TCP 作为 T_base_camera 计算依据。
        self._start_process("cad_registration", execute_override=False)

    def start_cad_preview(self) -> None:
        # 预览按钮始终强制 no-execute，避免误把上方的真实运动勾选带入预览。
        self._start_process("cad_preview", execute_override=False)

    def start_cad_motion(self) -> None:
        if not self.execute_var.get():
            messagebox.showwarning(
                "真实运动未启用",
                "请先勾选“真实运动（未勾选时仅预览）”，再点击 CAD 真实运动按钮。\n"
                "如果只想检查 CAD 投影，请点击“CAD 目标预览（不运动）”。",
                parent=self,
            )
            return
        if not self.cad_workspace_checked_var.get():
            messagebox.showwarning(
                "需要安全确认",
                "请先勾选 CAD 参数区的“我已检查机器人活动范围、工件和急停条件”。",
                parent=self,
            )
            return
        self._start_process("cad_motion", execute_override=True)

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
        if execute and not messagebox.askyesno(
            "确认真实运动",
            (
                "将执行回原点、单孔粗定位、下降和横向偏移采集。"
                f"{offset_confirmation}\n"
                "请在相机窗口中只选择一个孔，之后测试会自动运行。\n\n确认开始吗？"
                if mode == "offset" else
                "将先在当前位置自动采集5帧 RGB 并重新匹配 CAD；通过质量门后回机械臂原点，"
                "再在画面中点击选择一组 CAD 孔；"
                "然后共同居中到340 mm，只采一次RGB-D顶面高度，再逐孔按CAD位姿到260 mm。\n"
                "深度只修正整组共享高度，CAD继续提供孔号、XY和法向；每个孔完成后可点击按钮开始下一个孔。\n"
                "当前手眼结果若未通过生产质量门，本次会标记为实验运动。\n\n确认开始吗？"
                if mode == "cad_motion" else
                "将执行回原点及两阶段定位。除开始检测下一个已选孔外，运动会自动执行。\n\n确认开始吗？"
            ),
            parent=self,
        ):
            return
        self.process_mode = mode
        self._clear_log()
        self.result_var.set(
            "CAD 配准预览进行中…" if mode == "cad_registration" else
            "偏移容忍度测试进行中…" if mode == "offset" else
            "CAD 目标预览进行中…" if mode == "cad_preview" else
            "CAD 分组深度 + 260 mm 运动流程进行中…" if mode == "cad_motion" else
            "本次定位进行中…"
        )
        self.status_var.set(
            "正在启动 CAD 配准进程…" if mode == "cad_registration" else
            "正在启动偏移测试进程…" if mode == "offset" else
            "正在启动 CAD 预览进程…" if mode == "cad_preview" else
            "正在启动 CAD 运动进程…" if mode == "cad_motion" else
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
        self.registration_start_btn.configure(state=tk.DISABLED)
        self.cad_preview_btn.configure(state=tk.DISABLED)
        self.cad_motion_btn.configure(state=tk.DISABLED)
        self.offset_start_btn.configure(state=tk.DISABLED)
        threading.Thread(target=self._read_process_output, args=(self.process,), daemon=True).start()

    def _read_process_output(self, process: subprocess.Popen[str]) -> None:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                self.log_queue.put(("log", line))
                if (
                    "[MOTION_CONFIRM_REQUIRED]" in line
                    or "[NEXT_HOLE_CONFIRM_REQUIRED]" in line
                ):
                    self.log_queue.put(("confirm", line.strip()))
                elif "[NEXT_CAD_HOLE_CONFIRM_REQUIRED]" in line:
                    self.log_queue.put(("cad_hole_confirm", line.strip()))
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
        elif confirmation_kind == "cad_hole":
            self.motion_var.set("已确认，开始下一个 CAD 孔…")
        else:
            self.motion_var.set("已确认，开始检测下一个孔…")

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
                        "CAD 配准预览运行中…"
                        if self.process_mode == "cad_registration" else
                        "偏移容忍度测试运行中…"
                        if self.process_mode == "offset" else
                        "CAD 目标预览运行中…"
                        if self.process_mode == "cad_preview" else
                        "CAD 运动流程运行中…"
                        if self.process_mode == "cad_motion" else
                        "定位流程运行中…"
                    )
                elif kind == "confirm":
                    self.waiting_confirmation = True
                    self.confirmation_kind = "hole"
                    self.confirm_btn.configure(state=tk.NORMAL)
                    self.mark_error_btn.configure(state=tk.DISABLED)
                    self.cancel_btn.configure(state=tk.NORMAL)
                    self.confirm_btn.configure(text="开始检测下一个孔")
                    self.motion_var.set("等待开始下一个孔：请点击“开始检测下一个孔”继续。")
                elif kind == "cad_hole_confirm":
                    self.waiting_confirmation = True
                    self.confirmation_kind = "cad_hole"
                    self.confirm_btn.configure(state=tk.NORMAL, text="开始下一个 CAD 孔")
                    self.mark_error_btn.configure(state=tk.DISABLED)
                    self.cancel_btn.configure(state=tk.NORMAL)
                    self.motion_var.set("当前 CAD 孔已完成：请点击“开始下一个 CAD 孔”继续。")
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
        self.registration_start_btn.configure(state=tk.NORMAL)
        self.cad_preview_btn.configure(state=tk.NORMAL)
        self.cad_motion_btn.configure(state=tk.NORMAL)
        self.offset_start_btn.configure(state=tk.NORMAL)
        self.process = None
        if code == 0 or (self.process_mode == "cad_registration" and code == 2):
            self.status_var.set(
                "CAD 配准完成" if self.process_mode == "cad_registration" and code == 0 else
                "CAD 配准完成，但未通过质量门" if self.process_mode == "cad_registration" else
                "偏移测试完成" if self.process_mode == "offset" else
                "CAD 预览完成" if self.process_mode == "cad_preview" else
                "CAD 运动完成" if self.process_mode == "cad_motion" else
                "定位完成"
            )
            self._show_latest_result()
        else:
            self.status_var.set(f"定位结束，退出码={code}")
            self.result_var.set("本次定位未通过质量门或被取消；请查看运行日志和结果目录。")

    def _show_latest_result(self) -> None:
        if self.process_mode == "cad_registration":
            reports = sorted(
                (
                    path for path in CAD_REGISTRATION_RUNS_DIR.glob("*/cad_registration_report.json")
                    if path.stat().st_mtime >= self.run_started_at - 2.0
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if not reports:
                self.result_var.set("CAD 配准完成，但未找到本次配准报告。")
                return
            report_path = reports[0]
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                result = report.get("result", {})
                frames = result.get("frames", [])
                valid_indices = result.get("valid_frame_indices", [])
                success = bool(result.get("success", False))
                t_camera_cad = result.get("T_camera_cad")
                translation = None
                if isinstance(t_camera_cad, list) and len(t_camera_cad) >= 3:
                    try:
                        translation = [float(t_camera_cad[index][3]) for index in range(3)]
                    except (IndexError, TypeError, ValueError):
                        translation = None
                lines = [
                    f"报告：{report_path.parent}",
                    f"状态：{'PASS' if success else 'FAIL'}    "
                    f"有效帧：{len(valid_indices)}/{len(frames)}    "
                    f"跨帧中心 P95：{result.get('cross_frame_center_p95_mm', '-')} mm",
                    f"配准质量门：{'通过' if result.get('multi_frame_gate_pass') else '未通过'}    "
                    f"运动许可：{bool(report.get('motion_allowed', False))}（Stage 0 始终不运动）",
                ]
                if translation is not None:
                    lines.append(f"T_camera_cad 平移（mm）：{self._format_vector(translation)}")
                mapping_json = report.get("mapping_files", {}).get("json")
                if mapping_json:
                    lines.append(f"映射表：{mapping_json}")
                failures = result.get("failure_reasons", [])
                if failures:
                    lines.append("失败原因：" + "；".join(str(item) for item in failures))
                if success:
                    lines.append("该报告通过预览；下一次 CAD 运动启动时仍会用当前位置 RGB/TCP 重新自动配准。")
                self.result_var.set("\n".join(lines))
            except Exception as exc:
                self.result_var.set(f"读取 CAD 配准报告失败：{exc}")
            return
        if self.process_mode in {"cad_preview", "cad_motion"}:
            reports = sorted(
                (
                    path for path in CAD_MOTION_RUNS_DIR.glob("cad-motion-*/report.json")
                    if path.stat().st_mtime >= self.run_started_at - 2.0
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if not reports:
                self.result_var.set("CAD流程完成，但未找到本次 JSON 报告。")
                return
            try:
                report = json.loads(reports[0].read_text(encoding="utf-8"))
                plan = report.get("stages", {}).get("cad_target_pose_plan", {})
                processed = report.get("stages", {}).get("processed_holes", {})
                holes = processed.get("holes", []) or plan.get("holes", [])
                selected_ids = report.get("selected_hole_ids", [])
                status = report.get("status", "-")
                lines = [
                    f"报告：{reports[0].parent}",
                    f"状态：{status}    真实运动请求：{bool(report.get('motion_requested'))}    "
                    f"已下发运动：{bool(report.get('motion_executed'))}",
                    f"CAD目标孔：{', '.join(str(item) for item in selected_ids) if selected_ids else '-'}",
                ]
                fresh_report = report.get("cad_registration_report")
                if fresh_report:
                    lines.append(f"本次启动自动重新配准报告：{fresh_report}")
                if report.get("cad_registration_policy"):
                    lines.append(f"CAD配准策略：{report.get('cad_registration_policy')}")
                group_depth = report.get("stages", {}).get("cad_group_depth_340", {})
                if group_depth:
                    depth_mode_text = (
                        "单孔高度" if group_depth.get("depth_mode") == "single_hole"
                        else "多孔共享高度"
                    )
                    lines.append(
                        f"340 mm深度（{depth_mode_text}）：{'通过' if group_depth.get('success') else group_depth.get('status', '未通过')}    "
                        f"有效帧：{group_depth.get('valid_frame_count', '-')}/"
                        f"{group_depth.get('total_frame_count', '-')}    "
                        f"高度：{group_depth.get('height_median_mm', '-')} mm    "
                        f"偏差：{group_depth.get('height_offset_mm', '-')} mm"
                    )
                    if group_depth.get("failure_reasons"):
                        lines.append("共同深度失败原因：" + "；".join(
                            str(item) for item in group_depth["failure_reasons"]
                        ))
                initial_tcp = report.get("robot_initial_tcp_pose_m_rad")
                if initial_tcp:
                    lines.append(f"启动时 TCP（m/rad）：{self._format_vector(initial_tcp)}")
                for item in holes:
                    if not isinstance(item, dict):
                        continue
                    hole_id = item.get("hole_id", "-")
                    target = item.get("target_tcp_pose_m_rad") or item.get("cad_260_target_tcp_pose_m_rad")
                    center = item.get("cad_center_base_mm") or item.get("hole_center_base_mm")
                    lines.append(
                        f"孔 {hole_id}：CAD中心={self._format_vector(center)} "
                        f"目标TCP={self._format_vector(target)}"
                    )
                    fine = item.get("fine_yolo")
                    if isinstance(fine, dict):
                        valid_frames = fine.get("valid_frames", "-")
                        total_frames = fine.get("total_frames", "-")
                        scatter = fine.get("center_scatter_p95_px", "-")
                        mean_error = fine.get("mean_distance_to_cad_projection_px", "-")
                        lines.append(
                            f"  260精定位：YOLO有效帧={valid_frames}/{total_frames}，"
                            f"中心P95={scatter} px，CAD→YOLO平均误差={mean_error} px"
                        )
                        visualization = fine.get("visualization", {})
                        if isinstance(visualization, dict):
                            result_image = visualization.get("result_image") or visualization.get("summary_image")
                            if result_image:
                                lines.append(f"  精定位可视化：{result_image}")
                if report.get("error"):
                    lines.append(f"错误：{report['error']}")
                if status == "completed_experimental_handeye":
                    lines.append("警告：本次使用未通过生产验证的实验手眼结果。")
                self.result_var.set("\n".join(lines))
            except Exception as exc:
                self.result_var.set(f"读取 CAD 运动报告失败：{exc}")
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
                    f"顺序处理孔数：{final.get('hole_count', len(holes))}    "
                    f"成功：{completed_count}    延期：{deferred_count}    "
                    f"最终 TCP：{self._format_vector(final.get('final_tcp_pose_m_rad'))}",
                ]
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
                        f"法向={self._format_vector(item.get('plane_normal_toward_camera_base'))} "
                        f"姿态={self._format_vector(item.get('hole_pose_m_rad'))} "
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

    def open_cad_registration_results_dir(self) -> None:
        CAD_REGISTRATION_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(CAD_REGISTRATION_RUNS_DIR))  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("打开 CAD 配准结果目录失败", str(exc), parent=self)

    def open_cad_results_dir(self) -> None:
        CAD_MOTION_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(CAD_MOTION_RUNS_DIR))  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("打开 CAD 结果目录失败", str(exc), parent=self)

    def open_latest_cad_fine_visualization(self) -> None:
        """用系统图片查看器打开最近一次 260 mm 精定位的汇总图。"""

        candidates: list[Path] = []
        for pattern in (
            "cad-motion-*/fine_visualizations/*/summary_result.png",
            "cad-motion-*/fine_visualizations/*/summary.png",
            "cad-motion-*/fine_visualizations/*/frame_*.png",
        ):
            candidates.extend(CAD_MOTION_RUNS_DIR.glob(pattern))
        candidates = [path for path in candidates if path.is_file()]
        if not candidates:
            messagebox.showinfo(
                "尚无精定位图",
                "还没有找到 260 mm 精定位可视化图。请先完成一次 CAD 运动流程。",
                parent=self,
            )
            return
        try:
            latest = max(candidates, key=lambda path: path.stat().st_mtime)
            os.startfile(str(latest))  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("打开精定位图失败", str(exc), parent=self)

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
