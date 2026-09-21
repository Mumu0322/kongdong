#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""io_utils 中 JSON 归一化与 CSV 落盘的测试。

这两个函数替换了原来散落在 4 个模块里的私有副本，行为在这里固定下来。
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aubo_workbench.io_utils import jsonable, write_dict_rows


class JsonableTests(unittest.TestCase):
    def test_numpy_array_becomes_nested_list(self) -> None:
        self.assertEqual(jsonable(np.array([[1.0, 2.0], [3.0, 4.0]])),
                         [[1.0, 2.0], [3.0, 4.0]])

    def test_numpy_scalars_become_python_numbers(self) -> None:
        self.assertIsInstance(jsonable(np.float64(1.5)), float)
        self.assertIsInstance(jsonable(np.int32(7)), int)

    def test_path_becomes_string(self) -> None:
        self.assertIsInstance(jsonable(Path("a") / "b"), str)

    def test_object_with_to_dict_is_expanded(self) -> None:
        class Payload:
            def to_dict(self) -> dict[str, object]:
                return {"value": np.float64(2.5)}

        self.assertEqual(jsonable(Payload()), {"value": 2.5})

    def test_nested_containers_and_non_string_keys(self) -> None:
        value = {1: [np.int64(2), (np.float32(0.5), Path("c"))]}
        result = jsonable(value)
        self.assertEqual(list(result.keys()), ["1"])
        self.assertEqual(result["1"][0], 2)
        self.assertAlmostEqual(result["1"][1][0], 0.5, places=6)
        self.assertEqual(result["1"][1][1], "c")

    def test_result_is_json_serializable(self) -> None:
        payload = {"matrix": np.eye(2), "path": Path("d"), "n": np.int16(3)}
        json.dumps(jsonable(payload))


class WriteDictRowsTests(unittest.TestCase):
    def test_derives_union_of_keys_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "out.csv"
            write_dict_rows(target, [{"b": 1}, {"a": 2}])
            with target.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["a", "b"])

    def test_explicit_fields_control_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "out.csv"
            write_dict_rows(target, [{"a": 1, "b": 2}], fields=["b", "a"])
            with target.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["b", "a"])
        self.assertEqual(rows[1], ["2", "1"])

    def test_empty_rows_still_write_fallback_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "out.csv"
            write_dict_rows(target, [], fallback_fields=("sample_id", "status"))
            with target.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows, [["sample_id", "status"]])

    def test_uses_utf8_sig_for_excel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "out.csv"
            write_dict_rows(target, [{"名称": "孔位"}])
            self.assertTrue(target.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_creates_missing_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "nested" / "deep" / "out.csv"
            write_dict_rows(target, [{"a": 1}])
            self.assertTrue(target.is_file())

    def test_accepts_generator_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "out.csv"
            write_dict_rows(target, ({"a": index} for index in range(3)))
            with target.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
