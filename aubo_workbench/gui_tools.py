#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主工作台中的独立视觉工具、实验入口和运行结果中心。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Callable

from .paths import (
    CHARUCO_HEIGHT_ERROR_DIR,
    CHARUCO_POINT_EXPERIMENTS_DIR,
    DATA_DIR,
    HANDEYE_CANDIDATE_DIR,
    HOLE_LOCALIZATION_COARSE_CACHE_DIR,
    HOLE_LOCALIZATION_RUNS_DIR,
    TCP_ABSOLUTE_XY_MODEL_DIR,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
LENS_GUI_SCRIPT = PROJECT_DIR / "LiaoKuang" / "料框镜片检测前端.py"
CACHE_GUI_SCRIPT = PROJECT_DIR / "tools" / "visualize_coarse_cache.py"
HANDEYE_POSE_SCRIPT = PROJECT_DIR / "run_handeye_pose_sequence.py"
CHARUCO_EXPERIMENT_SCRIPT = PROJECT_DIR / "run_charuco_height_error_experiment.py"


@dataclass(frozen=True)
class ResultSource:
    key: str
    label: str
    root: Path
    patterns: tuple[str, ...]


RESULT_SOURCES = (
    ResultSource(
        "hole_localization", "两阶段 / 偏移测试", HOLE_LOCALIZATION_RUNS_DIR,
        ("two-stage-*/report.json", "coarse-to-fine-offset-*/report.json"),
    ),
    ResultSource(
        "charuco_height", "ChArUco 高度 / RGB 精度", CHARUCO_HEIGHT_ERROR_DIR,
        ("*/report.json",),
    ),
    ResultSource(
        "charuco_point", "ChArUco 点位实验", CHARUCO_POINT_EXPERIMENTS_DIR,
        ("*/report.json",),
    ),
    ResultSource(
        "tcp_xy", "TCP-XY 模型", TCP_ABSOLUTE_XY_MODEL_DIR,
        ("*/report.json",),
    ),
)


def find_latest_reports(sources: tuple[ResultSource, ...] = RESULT_SOURCES) -> list[tuple[ResultSource, Path]]:
    """返回每类结果最新的报告；不存在的类别不制造占位文件。"""

    found: list[tuple[ResultSource, Path]] = []
    for source in sources:
        candidates: list[Path] = []
        if source.root.is_dir():
            for pattern in source.patterns:
                candidates.extend(path for path in source.root.glob(pattern) if path.is_file())
        if candidates:
            found.append((source, max(candidates, key=lambda path: path.stat().st_mtime)))
    return sorted(found, key=lambda item: item[1].stat().st_mtime, reverse=True)


def report_status(path: Path) -> str:
    """从不同实验报告的常见字段中提取适合列表展示的状态。"""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "报告不可读"
    if not isinstance(payload, dict):
        return "报告格式异常"
    status = payload.get("status")
    if status not in (None, ""):
        return str(status)
    result = payload.get("result")
    if isinstance(result, dict) and "success" in result:
        return "PASS" if bool(result["success"]) else "FAIL"
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("status", "result"):
            if summary.get(key) not in (None, ""):
                return str(summary[key])
    return "已生成"


def _open_path(path: Path, parent: tk.Misc) -> None:
    try:
        os.startfile(str(path))  # type: ignore[attr-defined]
    except OSError as exc:
        messagebox.showerror("打开失败", str(exc), parent=parent)


class VisualToolsPanel(ttk.Frame):
    """启动已有的独立 GUI/交互实验，不把相机循环塞进主 Tk 线程。"""

    def __init__(self, master: tk.Misc, get_connection: Callable[[], dict[str, Any]]) -> None:
        super().__init__(master, padding=14)
        self.get_connection = get_connection
        self.processes: dict[str, subprocess.Popen[Any]] = {}
        self.status_var = tk.StringVar(value="独立工具尚未启动")
        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="视觉与实验工具", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
        ttk.Label(
            self,
            text="工具在独立窗口运行；相机或实验退出后，主工作台仍保持可用。",
            foreground="#555555",
        ).pack(anchor="w", pady=(3, 12))

        vision = ttk.LabelFrame(self, text="视觉检测与诊断", padding=10)
        vision.pack(fill=tk.X)
        self._tool_row(
            vision, 0, "料框镜片实时检测",
            "Gemini 435Le 左/右 IR、实时 YOLO、曝光/增益/激光控制和当前帧保存。",
            "lens", lambda: [sys.executable, str(LENS_GUI_SCRIPT)], LENS_GUI_SCRIPT,
        )
        self._tool_row(
            vision, 1, "粗定位缓存可视化",
            "查看历史孔位局部点云、平面质量与缓存范围，不修改缓存数据。",
            "cache", lambda: [
                sys.executable, str(CACHE_GUI_SCRIPT), "--gui",
                "--cache-dir", str(HOLE_LOCALIZATION_COARSE_CACHE_DIR),
            ], CACHE_GUI_SCRIPT,
        )

        experiments = ttk.LabelFrame(self, text="标定实验（默认不执行机器人运动）", padding=10)
        experiments.pack(fill=tk.X, pady=(10, 0))
        self._tool_row(
            experiments, 0, "手眼 40 点序列预检",
            "校验当前点位计划与进度；实机运动必须在控制台显式追加 --execute。",
            "handeye_pose", self._handeye_preview_command, HANDEYE_POSE_SCRIPT, console=True,
        )
        self._tool_row(
            experiments, 1, "ChArUco 高度 / 视野实验",
            "启动相机测量流程；默认仅测量和预览 Z 修正，不发送 moveLine。",
            "charuco", self._charuco_preview_command, CHARUCO_EXPERIMENT_SCRIPT, console=True,
        )

        folders = ttk.LabelFrame(self, text="常用数据", padding=10)
        folders.pack(fill=tk.X, pady=(10, 0))
        for column, (label, path) in enumerate((
            ("手眼候选", HANDEYE_CANDIDATE_DIR),
            ("ChArUco 实验", CHARUCO_HEIGHT_ERROR_DIR),
            ("粗定位缓存", HOLE_LOCALIZATION_COARSE_CACHE_DIR),
            ("全部运行数据", DATA_DIR),
        )):
            ttk.Button(folders, text=label, command=lambda target=path: _open_path(target, self)).grid(
                row=0, column=column, padx=(0, 8), sticky="w",
            )
        ttk.Label(self, textvariable=self.status_var, foreground="#555555").pack(anchor="w", pady=(12, 0))

    def _tool_row(
        self,
        parent: ttk.LabelFrame,
        row: int,
        title: str,
        description: str,
        key: str,
        command_builder: Callable[[], list[str]],
        required_file: Path,
        *,
        console: bool = False,
    ) -> None:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text=title, font=("TkDefaultFont", 10, "bold")).grid(
            row=row, column=0, sticky="w", padx=(0, 12), pady=6,
        )
        ttk.Label(parent, text=description, foreground="#555555", wraplength=760).grid(
            row=row, column=1, sticky="w", pady=6,
        )
        ttk.Button(
            parent, text="启动", width=10,
            command=lambda: self._launch(key, title, command_builder, required_file, console),
        ).grid(row=row, column=2, sticky="e", padx=(12, 0), pady=6)

    def _connection_args(self) -> list[str]:
        cfg = self.get_connection()
        return [
            "--robot-ip", str(cfg["ip"]), "--robot-port", str(cfg["port"]),
            "--robot-user", str(cfg["user"]), "--robot-password", str(cfg["password"]),
            "--robot-timeout-ms", str(cfg["timeout_ms"]),
        ]

    def _handeye_preview_command(self) -> list[str]:
        cfg = self.get_connection()
        return [
            sys.executable, str(HANDEYE_POSE_SCRIPT),
            "--ip", str(cfg["ip"]), "--port", str(cfg["port"]),
            "--user", str(cfg["user"]), "--password", str(cfg["password"]),
            "--timeout-ms", str(cfg["timeout_ms"]),
        ]

    def _charuco_preview_command(self) -> list[str]:
        return [sys.executable, str(CHARUCO_EXPERIMENT_SCRIPT), *self._connection_args()]

    def _launch(
        self,
        key: str,
        title: str,
        command_builder: Callable[[], list[str]],
        required_file: Path,
        console: bool,
    ) -> None:
        running = self.processes.get(key)
        if running is not None and running.poll() is None:
            self.status_var.set(f"{title} 已在运行，请切换到它的独立窗口。")
            return
        if not required_file.is_file():
            messagebox.showerror("工具不存在", f"未找到：{required_file}", parent=self)
            return
        try:
            command = command_builder()
            flags = subprocess.CREATE_NEW_CONSOLE if console and sys.platform == "win32" else 0
            self.processes[key] = subprocess.Popen(command, cwd=str(PROJECT_DIR), creationflags=flags)
        except Exception as exc:
            messagebox.showerror(f"启动{title}失败", str(exc), parent=self)
            return
        self.status_var.set(f"已启动：{title}")

    def on_close(self) -> None:
        # 独立工具由各自窗口负责释放相机/机器人资源，主窗口不强杀子进程。
        self.processes.clear()


class ResultsCenterPanel(ttk.Frame):
    """集中展示各条流程最近一次报告，并提供可发现的数据入口。"""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, padding=14)
        self.rows: dict[str, Path] = {}
        self.summary_var = tk.StringVar(value="")
        self._build()
        self.refresh()

    def _build(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="结果中心", font=("TkDefaultFont", 13, "bold")).pack(side=tk.LEFT)
        ttk.Button(header, text="刷新", command=self.refresh).pack(side=tk.RIGHT)

        columns = ("kind", "time", "status", "folder")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=14)
        for key, text, width, anchor in (
            ("kind", "功能", 180, "w"), ("time", "更新时间", 155, "center"),
            ("status", "状态", 190, "w"), ("folder", "结果目录", 620, "w"),
        ):
            self.tree.heading(key, text=text)
            self.tree.column(key, width=width, anchor=anchor, stretch=key == "folder")
        scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        scroll.place(in_=self.tree, relx=1.0, rely=0, relheight=1.0, anchor="ne")
        self.tree.bind("<Double-1>", lambda _event: self.open_selected_report())
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self._update_summary())

        actions = ttk.Frame(self)
        actions.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(actions, text="打开报告", command=self.open_selected_report).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(actions, text="打开结果目录", command=self.open_selected_folder).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(actions, text="打开全部运行数据", command=lambda: _open_path(DATA_DIR, self)).pack(side=tk.LEFT)
        ttk.Label(actions, textvariable=self.summary_var, foreground="#555555").pack(side=tk.RIGHT)

    def refresh(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.rows.clear()
        for index, (source, report) in enumerate(find_latest_reports()):
            item_id = f"result-{index}"
            modified = datetime.fromtimestamp(report.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            self.tree.insert(
                "", tk.END, iid=item_id,
                values=(source.label, modified, report_status(report), str(report.parent)),
            )
            self.rows[item_id] = report
        count = len(self.rows)
        self.summary_var.set(f"已发现 {count} 类最新报告" if count else "尚未发现运行报告")

    def _selected_path(self) -> Path | None:
        selected = self.tree.selection()
        return self.rows.get(selected[0]) if selected else None

    def _update_summary(self) -> None:
        path = self._selected_path()
        self.summary_var.set(str(path) if path is not None else f"已发现 {len(self.rows)} 类最新报告")

    def open_selected_report(self) -> None:
        path = self._selected_path()
        if path is None:
            messagebox.showinfo("未选择报告", "请先选择一行运行结果。", parent=self)
            return
        _open_path(path, self)

    def open_selected_folder(self) -> None:
        path = self._selected_path()
        if path is None:
            messagebox.showinfo("未选择报告", "请先选择一行运行结果。", parent=self)
            return
        _open_path(path.parent, self)
