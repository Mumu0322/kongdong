from __future__ import annotations

import unittest
import sys
import tkinter as tk

from aubo_workbench.tcp_teach import TeachPoint, TcpTeachPanel, TcpTeachSession, fmt_xyz_mm


class _FakeClient:
    def hasConnected(self) -> bool:
        return True


class _FakeMath:
    def __init__(self) -> None:
        self.call_count = 0

    def tcpOffsetIdentify(self, poses: list[list[float]]) -> tuple[list[float], int]:
        del poses
        self.call_count += 1
        delta = self.call_count * 0.000001
        return [0.001 + delta, 0.002 + delta, 0.210 + delta], 0

    def poseTrans(self, tool_pose: list[float], offset: list[float]) -> list[float]:
        return [tool_pose[i] + offset[i] for i in range(3)] + tool_pose[3:6]


def _point(index: int) -> TeachPoint:
    return TeachPoint(
        name=f"P{index + 1}",
        timestamp="2026-07-17 00:00:00",
        tool_pose=[index * 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        tcp_pose=[0.0] * 6,
        joints=[0.0] * 6,
        actual_tcp_offset=[0.0] * 6,
    )


class TcpTeachSessionTests(unittest.TestCase):
    def test_identify_tcp_accepts_three_value_sdk_result(self) -> None:
        session = TcpTeachSession()
        session.client = _FakeClient()
        session.math_api = _FakeMath()
        session.tcp_offset_cache = [0.0, 0.0, 0.0, 0.1, 0.2, 0.3]

        result = session.identify_tcp([_point(index) for index in range(5)])

        self.assertEqual(result["valid_combinations"], 5)
        self.assertEqual(len(result["identified_xyz"]), 3)
        self.assertEqual(len(result["offset"]), 6)
        self.assertEqual(result["offset"][3:], [0.1, 0.2, 0.3])

    def test_fmt_xyz_mm_converts_meters_to_millimeters(self) -> None:
        self.assertEqual(fmt_xyz_mm([0.440783, 0.312765, 0.266545]), "[440.783, 312.765, 266.545]")


@unittest.skipUnless(sys.platform == "win32", "Windows desktop layout regression")
class TcpTeachLayoutTests(unittest.TestCase):
    def test_point_table_is_not_covered_by_its_container(self) -> None:
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        # Keep the test window almost transparent; alpha=0 prevents hit testing.
        root.attributes("-alpha", 0.01)
        root.geometry("1300x700+0+0")
        panel = TcpTeachPanel(root)
        panel.pack(fill="both", expand=True)
        panel.stop_polling()
        panel.points = [_point(index) for index in range(12)]
        panel.refresh_points()
        root.deiconify()
        root.lift()

        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)

        canvas = next(w for w in descendants(panel) if isinstance(w, tk.Canvas))
        for scale in (1.333, 2.0):
            with self.subTest(scale=scale):
                root.tk.call("tk", "scaling", scale)
                root.update()
                canvas.yview_moveto(0)
                root.update()
                table_y = panel.tree.winfo_rooty() - canvas.winfo_rooty()
                content_height = float(canvas.cget("scrollregion").split()[3])
                canvas.yview_moveto(max(0, table_y - 40) / content_height)
                panel.tree.yview_moveto(0)
                root.update()
                rows = panel.tree.get_children()
                x, y, width, height = panel.tree.bbox(rows[0])
                for local_y in (y // 2, y + height // 2):
                    hit = root.winfo_containing(
                        panel.tree.winfo_rootx() + x + 10,
                        panel.tree.winfo_rooty() + local_y,
                    )
                    self.assertEqual(hit, panel.tree, "Point header/row is obscured")
                panel.tree.see(rows[-1])
                root.update()
                self.assertTrue(panel.tree.bbox(rows[-1]), "Last point must be scrollable into view")


if __name__ == "__main__":
    unittest.main()
