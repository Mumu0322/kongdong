from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from aubo_workbench.batch_grouping import (
    group_holes_spatially,
    group_holes_spatially_with_metadata,
)
import run_yolo_eye_in_hand_optimized as localization


def _hole(hole_id: int, x: float, y: float, u: float, v: float) -> dict:
    return {
        "hole_id": hole_id,
        "initial_center_base_mm": np.asarray((x, y, 0.0), dtype=np.float64),
        "initial_center_px": np.asarray((u, v), dtype=np.float64),
        "initial_plane_normal_base": np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
    }


class BatchGroupingTests(unittest.TestCase):
    def test_grid_is_split_by_four_hole_cap_and_all_groups_are_connected(self) -> None:
        holes = [
            _hole(
                row * 4 + column + 1,
                column * 10.0,
                row * 10.0,
                column * 100.0,
                row * 100.0,
            )
            for row in range(3)
            for column in range(4)
        ]

        groups, diagnostics = group_holes_spatially(
            holes,
            max_group_size=4,
            max_aspect_ratio=1.8,
            adjacency_distance_factor=1.8,
            max_normal_spread_deg=5.0,
        )

        self.assertEqual(len(groups), len(diagnostics))
        self.assertTrue(all(len(group) <= 4 for group in groups))
        self.assertEqual(
            sorted(int(hole["hole_id"]) for group in groups for hole in group),
            list(range(1, 13)),
        )
        self.assertTrue(all(item["connected"] for item in diagnostics))
        self.assertTrue(
            all(
                item["aspect_ratio"] is None or item["aspect_ratio"] <= 1.8 + 1e-9
                for item in diagnostics
            )
        )

    def test_long_strip_does_not_remain_one_shared_group(self) -> None:
        holes = [_hole(index + 1, index * 10.0, 0.0, index * 100.0, 100.0) for index in range(6)]

        groups, diagnostics = group_holes_spatially(
            holes,
            max_group_size=4,
            max_aspect_ratio=1.8,
            adjacency_distance_factor=1.8,
        )

        self.assertGreater(len(groups), 1)
        self.assertTrue(all(len(group) <= 2 for group in groups))
        self.assertTrue(all(item["aspect_ratio"] in (None, 1.0) for item in diagnostics))

    def test_group_seed_order_is_front_to_back_then_left_to_right(self) -> None:
        holes = [
            _hole(4, 10.0, 10.0, 200.0, 200.0),
            _hole(2, 0.0, 0.0, 100.0, 100.0),
            _hole(3, 10.0, 0.0, 200.0, 100.0),
            _hole(1, 0.0, 10.0, 100.0, 200.0),
        ]

        groups, _ = group_holes_spatially(
            holes,
            max_group_size=2,
            max_aspect_ratio=2.0,
            adjacency_distance_factor=1.8,
        )

        self.assertEqual([[int(hole["hole_id"]) for hole in group] for group in groups], [[2, 3], [1, 4]])

    def test_planner_rejection_starts_a_new_connected_group(self) -> None:
        holes = [_hole(index + 1, index * 10.0, 0.0, index * 100.0, 100.0) for index in range(4)]

        def planner(group: list[dict]) -> dict:
            return {"ok": len(group) <= 2}

        groups, diagnostics = group_holes_spatially(
            holes,
            planner=planner,
            max_group_size=4,
            max_aspect_ratio=None,
            adjacency_distance_factor=1.8,
        )

        self.assertEqual([[int(hole["hole_id"]) for hole in group] for group in groups], [[1, 2], [3, 4]])
        self.assertEqual(diagnostics[0]["planner_reason"], None)
        self.assertEqual(diagnostics[1]["planner_reason"], None)

    def test_solver_increases_group_count_only_when_lower_bound_is_infeasible(self) -> None:
        holes = [
            _hole(index + 1, index * 10.0, 0.0, index * 100.0, 100.0)
            for index in range(4)
        ]

        groups, diagnostics, metadata = group_holes_spatially_with_metadata(
            holes,
            planner=lambda group: {"ok": len(group) <= 2},
            max_group_size=4,
            adjacency_distance_factor=1.8,
        )

        self.assertEqual(len(groups), 2)
        self.assertEqual(metadata["theoretical_min_group_count"], 1)
        self.assertEqual(metadata["minimum_feasible_group_count"], 2)
        self.assertTrue(metadata["group_count_increased"])
        self.assertTrue(all(item["selected_by_exact_cover"] for item in diagnostics))

    def test_xy_diameter_gate_breaks_a_connected_but_wide_cluster(self) -> None:
        # The points form a U-shaped neighbour chain.  Connectivity and PCA
        # aspect ratio alone are insufficient to reject it as one group.
        holes = [
            _hole(index + 1, x, y, x * 10.0 + 100.0, y * 10.0 + 100.0)
            for index, (x, y) in enumerate(
                [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0),
                 (20.0, 10.0), (20.0, 20.0), (10.0, 20.0), (0.0, 20.0)]
            )
        ]

        groups, diagnostics = group_holes_spatially(
            holes,
            max_group_size=7,
            max_aspect_ratio=2.0,
            adjacency_distance_factor=1.8,
            max_xy_diameter_mm=22.0,
        )

        self.assertGreater(len(groups), 1)
        self.assertTrue(
            all(float(item["xy_diameter_mm"]) <= 22.0 + 1e-9 for item in diagnostics)
        )
        self.assertTrue(
            all(item.get("max_xy_diameter_mm") == 22.0 for item in diagnostics)
        )

    def test_same_group_count_prefers_compact_partition(self) -> None:
        # Four holes are near the origin and four are near (100, 100).  A
        # compactness-optimised exact cover must not join the two clusters just
        # because both partitions use the same minimum number of groups.
        holes = [
            _hole(1, 0.0, 0.0, 100.0, 100.0),
            _hole(2, 10.0, 0.0, 110.0, 100.0),
            _hole(3, 0.0, 10.0, 100.0, 110.0),
            _hole(4, 10.0, 10.0, 110.0, 110.0),
            _hole(5, 100.0, 100.0, 200.0, 200.0),
            _hole(6, 110.0, 100.0, 210.0, 200.0),
            _hole(7, 100.0, 110.0, 200.0, 210.0),
            _hole(8, 110.0, 110.0, 210.0, 210.0),
        ]

        groups, _, metadata = group_holes_spatially_with_metadata(
            holes,
            max_group_size=4,
            max_aspect_ratio=2.0,
            adjacency_distance_factor=1.8,
            max_xy_diameter_mm=25.0,
        )

        self.assertEqual(metadata["minimum_feasible_group_count"], 2)
        self.assertEqual(
            {frozenset(int(hole["hole_id"]) for hole in group) for group in groups},
            {frozenset({1, 2, 3, 4}), frozenset({5, 6, 7, 8})},
        )

    def test_grouping_visualization_writes_png_and_high_quality_jpg(self) -> None:
        holes = [
            _hole(1, 0.0, 0.0, 100.0, 100.0),
            _hole(2, 10.0, 0.0, 150.0, 100.0),
            _hole(3, 0.0, 10.0, 100.0, 150.0),
        ]
        groups, diagnostics = group_holes_spatially(
            holes, max_group_size=4, max_aspect_ratio=2.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "01_home_selected.png"
            cv2.imwrite(str(source), np.full((240, 320, 3), 220, dtype=np.uint8))
            paths = localization._save_grouping_plan_visualization(
                source, root / "batch_fine_grouping_plan", holes, groups,
                "GROUP PLAN", diagnostics,
            )
            self.assertIsNotNone(paths)
            assert paths is not None
            self.assertTrue(Path(paths["png"]).is_file())
            self.assertTrue(Path(paths["jpg"]).is_file())
            self.assertIsNotNone(cv2.imread(paths["jpg"], cv2.IMREAD_COLOR))


if __name__ == "__main__":
    unittest.main()
