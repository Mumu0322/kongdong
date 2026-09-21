#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""粗定位孔洞地图的可视化选孔窗口。"""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Any, Callable, Iterable


READY_STATUSES = {"ready", "completed"}


@dataclass(frozen=True)
class HoleMapPoint:
    """一个可在二维基坐标地图中显示的孔。"""

    hole_id: int
    x_mm: float
    y_mm: float
    z_mm: float
    status: str


def collect_hole_map_points(payload: dict[str, Any], sector_id: int) -> list[HoleMapPoint]:
    """提取所选扇区中具有有限基坐标的孔，并按孔号排序。"""
    if not isinstance(payload, dict):
        raise ValueError("孔位地图格式无效")
    if int(payload.get("schema_version", -1)) == 4:
        key = f"S{int(sector_id):02d}"
        sector = (payload.get("sectors") or {}).get(key)
        if not isinstance(sector, dict):
            raise ValueError(f"当前地图中没有扇区 {key}")
        holes = sector.get("holes") or {}
    else:
        holes = payload.get("holes") or {}
    records = holes.values() if isinstance(holes, dict) else holes
    points: list[HoleMapPoint] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        center = record.get("coarse_center_base_mm")
        if not isinstance(center, (list, tuple)) or len(center) < 3:
            continue
        try:
            hole_id = int(record["hole_id"])
            x_mm, y_mm, z_mm = (float(center[index]) for index in range(3))
        except (KeyError, TypeError, ValueError):
            continue
        if hole_id <= 0 or not all(math.isfinite(value) for value in (x_mm, y_mm, z_mm)):
            continue
        points.append(HoleMapPoint(
            hole_id=hole_id,
            x_mm=x_mm,
            y_mm=y_mm,
            z_mm=z_mm,
            status=str(record.get("status", "ready")),
        ))
    if not points:
        raise ValueError("所选扇区没有可显示的粗定位孔坐标")
    return sorted(points, key=lambda item: item.hole_id)


def normalize_selected_ids(
    selected_ids: Iterable[int], points: Iterable[HoleMapPoint],
) -> list[int]:
    """保留存在且可调用的孔号，同时保持用户的点击顺序。"""
    ready = {
        point.hole_id for point in points if point.status in READY_STATUSES
    }
    normalized: list[int] = []
    for raw_id in selected_ids:
        hole_id = int(raw_id)
        if hole_id in ready and hole_id not in normalized:
            normalized.append(hole_id)
    return normalized


class HoleMapSelector(tk.Toplevel):
    """按真实基坐标 XY 绘制、点击并排序所选孔。"""

    def __init__(
        self,
        master: tk.Misc,
        *,
        points: list[HoleMapPoint],
        sector_id: int,
        map_name: str,
        initial_ids: Iterable[int],
        on_confirm: Callable[[list[int]], None],
    ) -> None:
        super().__init__(master)
        self.title(f"孔洞地图选孔 - 扇区 S{sector_id:02d}")
        self.geometry("1180x760")
        self.minsize(820, 560)
        self.transient(master.winfo_toplevel())
        self.points = points
        self.point_by_id = {point.hole_id: point for point in points}
        self.ready_ids = [
            point.hole_id for point in points if point.status in READY_STATUSES
        ]
        self.selected_ids = normalize_selected_ids(initial_ids, points)
        self.on_confirm = on_confirm
        self._redraw_job: str | None = None
        self.hover_var = tk.StringVar(
            value="点击孔标记可选择或取消；右侧为提交顺序，主页面启用最近邻优化时会重排。"
        )
        self.count_var = tk.StringVar()

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        header = ttk.Frame(self, padding=(10, 8, 10, 4))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text=f"{map_name}    S{sector_id:02d}    基坐标系 XY 俯视图",
            font=("TkDefaultFont", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="坐标来自 340 mm 粗定位地图；本窗口只选孔，不控制机械臂。",
            foreground="#666666",
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))

        body = ttk.Frame(self, padding=(10, 4, 10, 4))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            body, background="#f7f7f7", highlightthickness=1,
            highlightbackground="#999999", cursor="hand2",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._schedule_redraw)

        side = ttk.LabelFrame(body, text="已选孔（执行顺序）", padding=8)
        side.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        side.rowconfigure(1, weight=1)
        ttk.Label(side, textvariable=self.count_var).grid(row=0, column=0, sticky="w", pady=(0, 5))
        list_frame = ttk.Frame(side)
        list_frame.grid(row=1, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.selected_list = tk.Listbox(list_frame, width=22, height=24, exportselection=False)
        self.selected_list.grid(row=0, column=0, sticky="nsew")
        list_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.selected_list.yview)
        list_scroll.grid(row=0, column=1, sticky="ns")
        self.selected_list.configure(yscrollcommand=list_scroll.set)
        ttk.Button(side, text="移到上一位", command=lambda: self._move_selected(-1)).grid(
            row=2, column=0, sticky="ew", pady=(7, 3),
        )
        ttk.Button(side, text="移到下一位", command=lambda: self._move_selected(1)).grid(
            row=3, column=0, sticky="ew",
        )

        footer = ttk.Frame(self, padding=(10, 4, 10, 10))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.hover_var, foreground="#555555").grid(
            row=0, column=0, columnspan=7, sticky="w", pady=(0, 7),
        )
        ttk.Button(footer, text="全选", command=self._select_all).grid(row=1, column=1, padx=3)
        ttk.Button(footer, text="清空", command=self._clear).grid(row=1, column=2, padx=3)
        ttk.Button(footer, text="反选", command=self._invert).grid(row=1, column=3, padx=3)
        ttk.Button(footer, text="按孔号排序", command=self._sort).grid(row=1, column=4, padx=(3, 18))
        ttk.Button(footer, text="取消", command=self.destroy).grid(row=1, column=5, padx=3)
        ttk.Button(footer, text="确认选择", command=self._confirm).grid(row=1, column=6, padx=(3, 0))

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self._refresh_selection_widgets(redraw=False)
        self.after_idle(self._draw_map)
        self.grab_set()

    def _schedule_redraw(self, _event: tk.Event | None = None) -> None:
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(60, self._draw_map)

    def _draw_map(self) -> None:
        self._redraw_job = None
        canvas = self.canvas
        canvas.delete("all")
        width = max(400, canvas.winfo_width())
        height = max(360, canvas.winfo_height())
        margin_left, margin_right, margin_top, margin_bottom = 72, 35, 35, 62
        xs = [point.x_mm for point in self.points]
        ys = [point.y_mm for point in self.points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        x_span = max(1.0, max_x - min_x)
        y_span = max(1.0, max_y - min_y)
        plot_width = max(1.0, width - margin_left - margin_right)
        plot_height = max(1.0, height - margin_top - margin_bottom)
        scale = min(plot_width / x_span, plot_height / y_span)
        used_width, used_height = x_span * scale, y_span * scale
        x0 = margin_left + (plot_width - used_width) / 2.0
        y0 = margin_top + (plot_height - used_height) / 2.0

        def project(point: HoleMapPoint) -> tuple[float, float]:
            return (
                x0 + (point.x_mm - min_x) * scale,
                y0 + (max_y - point.y_mm) * scale,
            )

        # 轻量网格和坐标方向，帮助从地图上辨认孔的空间位置。
        for index in range(6):
            fraction = index / 5.0
            gx = x0 + used_width * fraction
            gy = y0 + used_height * fraction
            canvas.create_line(gx, y0, gx, y0 + used_height, fill="#dddddd", dash=(2, 4))
            canvas.create_line(x0, gy, x0 + used_width, gy, fill="#dddddd", dash=(2, 4))
            canvas.create_text(gx, y0 + used_height + 18, text=f"{min_x + x_span * fraction:.0f}", fill="#555555")
            canvas.create_text(x0 - 30, y0 + used_height - used_height * fraction, text=f"{min_y + y_span * fraction:.0f}", fill="#555555")
        canvas.create_text(x0 + used_width / 2, height - 12, text="机器人基坐标 X (mm)", fill="#333333")
        canvas.create_text(15, y0 + used_height / 2, text="Y", fill="#333333", font=("TkDefaultFont", 10, "bold"))

        selected_order = {hole_id: index + 1 for index, hole_id in enumerate(self.selected_ids)}
        radius = max(8.0, min(13.0, 0.24 * scale * 30.0))
        for point in self.points:
            px, py = project(point)
            selectable = point.status in READY_STATUSES
            selected = point.hole_id in selected_order
            fill = "#f28e2b" if selected else ("#4e9a51" if selectable else "#aaaaaa")
            outline = "#8a4b00" if selected else "#2f6031"
            tag = f"hole_{point.hole_id}"
            canvas.create_oval(
                px - radius, py - radius, px + radius, py + radius,
                fill=fill, outline=outline, width=2, tags=(tag, "hole"),
            )
            # 孔号始终显示；已选顺序显示在标记上方，避免混淆孔号和执行序号。
            canvas.create_text(
                px, py, text=str(point.hole_id), fill="white",
                font=("TkDefaultFont", 8, "bold"), tags=(tag, "hole"),
            )
            canvas.create_text(
                px, py - radius - 9,
                text=f"#{selected_order[point.hole_id]}" if selected else f"H{point.hole_id:02d}",
                fill="#9a4f00" if selected else "#333333",
                font=("TkDefaultFont", 8, "bold"), tags=(tag, "hole"),
            )
            if selectable:
                canvas.tag_bind(tag, "<Button-1>", lambda _event, hid=point.hole_id: self._toggle(hid))
            canvas.tag_bind(tag, "<Enter>", lambda _event, p=point: self._show_point(p))
            canvas.tag_bind(tag, "<Leave>", lambda _event: self._show_help())

    def _show_point(self, point: HoleMapPoint) -> None:
        self.hover_var.set(
            f"H{point.hole_id:02d}：X={point.x_mm:.3f} mm，Y={point.y_mm:.3f} mm，"
            f"Z={point.z_mm:.3f} mm，状态={point.status}"
        )

    def _show_help(self) -> None:
        self.hover_var.set(
            "点击孔标记可选择或取消；右侧为提交顺序，主页面启用最近邻优化时会重排。"
        )

    def _toggle(self, hole_id: int) -> None:
        if hole_id in self.selected_ids:
            self.selected_ids.remove(hole_id)
        else:
            self.selected_ids.append(hole_id)
        self._refresh_selection_widgets()

    def _select_all(self) -> None:
        self.selected_ids = list(self.ready_ids)
        self._refresh_selection_widgets()

    def _clear(self) -> None:
        self.selected_ids = []
        self._refresh_selection_widgets()

    def _invert(self) -> None:
        selected = set(self.selected_ids)
        self.selected_ids = [hole_id for hole_id in self.ready_ids if hole_id not in selected]
        self._refresh_selection_widgets()

    def _sort(self) -> None:
        self.selected_ids.sort()
        self._refresh_selection_widgets()

    def _move_selected(self, offset: int) -> None:
        selection = self.selected_list.curselection()
        if not selection:
            return
        old_index = int(selection[0])
        new_index = old_index + int(offset)
        if not 0 <= new_index < len(self.selected_ids):
            return
        hole_id = self.selected_ids.pop(old_index)
        self.selected_ids.insert(new_index, hole_id)
        self._refresh_selection_widgets()
        self.selected_list.selection_set(new_index)
        self.selected_list.see(new_index)

    def _refresh_selection_widgets(self, *, redraw: bool = True) -> None:
        self.selected_list.delete(0, tk.END)
        for index, hole_id in enumerate(self.selected_ids, start=1):
            point = self.point_by_id[hole_id]
            self.selected_list.insert(
                tk.END,
                f"{index:02d}. H{hole_id:02d}  ({point.x_mm:.1f}, {point.y_mm:.1f})",
            )
        self.count_var.set(f"已选 {len(self.selected_ids)} / {len(self.ready_ids)} 个有效孔")
        if redraw:
            self._draw_map()

    def _confirm(self) -> None:
        if not self.selected_ids:
            messagebox.showwarning("尚未选孔", "请至少点击选择一个孔。", parent=self)
            return
        self.on_confirm(list(self.selected_ids))
        self.destroy()
