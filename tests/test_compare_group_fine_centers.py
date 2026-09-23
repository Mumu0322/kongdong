from __future__ import annotations

import unittest

from tools.compare_group_fine_centers import build_rows


class CompareGroupFineCentersTests(unittest.TestCase):
    def test_selects_runtime_capture_and_compares_both_centers(self) -> None:
        diagnostic = {
            "hole_id": 3,
            "valid": True,
            "paired_center_diagnostic_status": "complete",
            "yolo_point_base_mm": [10.0, 20.0, 2.0],
            "geometric_point_base_mm": [10.3, 20.4, 2.0],
        }
        candidate = {
            "final_result": {"holes": [{
                "hole_id": 3, "hole_center_base_mm": [10.1, 20.0, 2.0],
            }]},
            "stages": {"batch_fine_results": {"groups": [{
                "group_index": 1,
                "hole_ids": [3],
                "capture": {"frame_records": [{"holes": [diagnostic]}]},
                "supplement_captures": [{
                    "round": 1,
                    "hole_ids": [3],
                    "capture": {"frame_records": [{"holes": [diagnostic]}]},
                }],
                "results": {"3": {"success": True, "capture_round": 1}},
            }]}}
        }
        reference = {"final_result": {"holes": [{
            "hole_id": 3, "hole_center_base_mm": [10.0, 20.0, 2.0],
        }]}}

        rows = build_rows(candidate, reference)

        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[0]["selected_by_runtime"])
        self.assertTrue(rows[1]["selected_by_runtime"])
        self.assertAlmostEqual(rows[1]["yolo_minus_reference_xy_mm"], 0.0)
        self.assertAlmostEqual(rows[1]["geometric_minus_reference_xy_mm"], 0.5)
        self.assertAlmostEqual(
            rows[1]["runtime_final_minus_reference_xy_mm"], 0.1,
        )


if __name__ == "__main__":
    unittest.main()
