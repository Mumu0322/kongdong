#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视觉工具页中纯文件扫描逻辑的离线测试。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from aubo_workbench.gui_tools import ResultSource, find_latest_reports, report_status


class ResultsCenterTests(unittest.TestCase):
    def test_finds_latest_report_per_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            older = root / "run-1" / "report.json"
            newer = root / "run-2" / "report.json"
            older.parent.mkdir()
            newer.parent.mkdir()
            older.write_text('{"status": "old"}', encoding="utf-8")
            newer.write_text('{"status": "new"}', encoding="utf-8")
            os.utime(older, (1000, 1000))
            os.utime(newer, (2000, 2000))
            source = ResultSource("sample", "样例", root, ("*/report.json",))

            reports = find_latest_reports((source,))

            self.assertEqual(len(reports), 1)
            self.assertEqual(reports[0][1], newer)

    def test_status_prefers_top_level_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            report.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            self.assertEqual(report_status(report), "completed")

    def test_status_supports_registration_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            report.write_text(json.dumps({"result": {"success": False}}), encoding="utf-8")
            self.assertEqual(report_status(report), "FAIL")

    def test_invalid_json_is_reported_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            report.write_text("not-json", encoding="utf-8")
            self.assertEqual(report_status(report), "报告不可读")

    def test_non_object_json_is_reported_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            report.write_text("[]", encoding="utf-8")
            self.assertEqual(report_status(report), "报告格式异常")


if __name__ == "__main__":
    unittest.main()
