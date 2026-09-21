from __future__ import annotations

import unittest

from aubo_workbench.gui_hole_map_selector import (
    collect_hole_map_points,
    normalize_selected_ids,
)


class HoleMapSelectorDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = {
            "schema_version": 4,
            "sectors": {
                "S01": {
                    "holes": {
                        "H02": {
                            "hole_id": 2,
                            "status": "ready",
                            "coarse_center_base_mm": [520.0, -30.0, 10.0],
                        },
                        "H01": {
                            "hole_id": 1,
                            "status": "completed",
                            "coarse_center_base_mm": [480.0, 20.0, -5.0],
                        },
                        "H03": {
                            "hole_id": 3,
                            "status": "failed",
                            "coarse_center_base_mm": [550.0, 45.0, 6.0],
                        },
                    }
                }
            },
        }

    def test_collects_real_base_xy_and_sorts_by_hole_id(self) -> None:
        points = collect_hole_map_points(self.payload, 1)
        self.assertEqual([point.hole_id for point in points], [1, 2, 3])
        self.assertEqual((points[0].x_mm, points[0].y_mm, points[0].z_mm), (480.0, 20.0, -5.0))

    def test_selection_keeps_click_order_and_excludes_unavailable_holes(self) -> None:
        points = collect_hole_map_points(self.payload, 1)
        selected = normalize_selected_ids([2, 3, 1, 2], points)
        self.assertEqual(selected, [2, 1])

    def test_missing_sector_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "S02"):
            collect_hole_map_points(self.payload, 2)


if __name__ == "__main__":
    unittest.main()
