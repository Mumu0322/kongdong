from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from aubo_workbench.batch_grouping import (
    classify_selected_hole_boundary_layers,
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
    def test_selected_region_boundary_classifies_outer_layer_only(self) -> None:
        holes = [
            _hole(row * 5 + column + 1, column * 10.0, row * 10.0,
                  column * 100.0, row * 100.0)
            for row in range(5)
            for column in range(5)
        ]

        classified, metadata = classify_selected_hole_boundary_layers(holes)

        self.assertEqual(metadata["coordinate_source"], "base_pca")
        self.assertEqual(metadata["component_count"], 1)
        self.assertEqual(
            metadata["edge_hole_ids"],
            [str(value) for value in (1, 2, 3, 4, 5, 6, 10, 11, 15, 16, 20, 21, 22, 23, 24, 25)],
        )
        self.assertEqual(
            metadata["interior_hole_ids"],
            [str(value) for value in (7, 8, 9, 12, 13, 14, 17, 18, 19)],
        )
        self.assertTrue(all(hole["boundary_class"] == "edge" for hole in classified if int(hole["hole_id"]) in {1, 2, 3, 4, 5, 6, 10, 11, 15, 16, 20, 21, 22, 23, 24, 25}))
        self.assertTrue(all(hole["boundary_class"] == "interior" for hole in classified if int(hole["hole_id"]) in {7, 8, 9, 12, 13, 14, 17, 18, 19}))

    def test_boundary_classification_keeps_disconnected_components_separate(self) -> None:
        holes = [
            _hole(row * 3 + column + 1, column * 10.0, row * 10.0,
                  column * 100.0, row * 100.0)
            for row in range(3)
            for column in range(3)
        ] + [
            _hole(10 + row * 3 + column, 100.0 + column * 10.0, row * 10.0,
                  500.0 + column * 100.0, row * 100.0)
            for row in range(3)
            for column in range(3)
        ]

        classified, metadata = classify_selected_hole_boundary_layers(holes)

        self.assertEqual(metadata["component_count"], 2)
        self.assertEqual(metadata["edge_hole_count"], 16)
        self.assertEqual(metadata["interior_hole_count"], 2)
        self.assertEqual(
            [hole["boundary_class"] for hole in classified if int(hole["hole_id"]) in {5, 14}],
            ["interior", "interior"],
        )

    def test_boundary_classification_keeps_a_concave_notch_on_the_outer_layer(self) -> None:
        # A selected rectangle with its upper-right corner removed has a
        # concave notch.  The two points facing the notch are still boundary
        # points of the selected region; the lower central points are interior.
        coordinates = [
            (column, row)
            for row in range(5)
            for column in range(5)
            if not (column >= 3 and row <= 1)
        ]
        holes = [
            _hole(index + 1, column * 10.0, row * 10.0,
                  column * 100.0, row * 100.0)
            for index, (column, row) in enumerate(coordinates)
        ]

        classified, _ = classify_selected_hole_boundary_layers(holes)
        by_id = {int(hole["hole_id"]): hole for hole in classified}

        self.assertEqual(by_id[6]["boundary_class"], "edge")
        self.assertEqual(by_id[10]["boundary_class"], "edge")
        self.assertEqual(by_id[14]["boundary_class"], "interior")

    def test_boundary_classification_honours_explicit_hole_override(self) -> None:
        holes = [
            _hole(row * 5 + column + 1, column * 10.0, row * 10.0,
                  column * 100.0, row * 100.0)
            for row in range(5)
            for column in range(5)
        ]
        holes[12]["boundary_class_override"] = "edge"
        holes[0]["boundary_class_override"] = "interior"

        classified, metadata = classify_selected_hole_boundary_layers(holes)
        by_id = {int(hole["hole_id"]): hole for hole in classified}

        self.assertEqual(by_id[13]["boundary_class"], "edge")
        self.assertEqual(by_id[1]["boundary_class"], "interior")
        self.assertTrue(by_id[13]["boundary_override"])
        self.assertEqual(metadata["manual_override_count"], 2)

    def test_edge_first_splitter_finishes_outer_phase_before_inner_phase(self) -> None:
        holes = [
            _hole(row * 5 + column + 1, column * 10.0, row * 10.0,
                  column * 100.0, row * 100.0)
            for row in range(5)
            for column in range(5)
        ]

        def planner(group: list[dict], *_args: object) -> tuple[np.ndarray, dict]:
            return np.eye(4), {
                "ok": True,
                "group_bbox_px": np.asarray((0.0, 0.0, 400.0, 400.0)),
                "projected_holes_px": {
                    int(hole["hole_id"]): np.asarray(hole["initial_center_px"], dtype=float)
                    for hole in group
                },
            }

        intrinsics = SimpleNamespace(width=1280, height=800)
        with patch.object(localization, "_plan_batch_coarse_group_pose", side_effect=planner):
            groups, diagnostics, metadata = localization._split_batch_localization_groups_edge_first(
                holes,
                np.eye(4),
                object(),
                0.0,
                intrinsics,
                340.0,
                50.0,
                0.8,
                max_group_size=5,
                max_aspect_ratio=None,
                max_xy_diameter_mm=30.0,
            )

        self.assertEqual(len(groups), len(diagnostics))
        self.assertTrue(metadata["coverage_check"]["ok"])
        self.assertEqual(
            [diagnostic["group_phase"] for diagnostic in diagnostics],
            sorted(
                (diagnostic["group_phase"] for diagnostic in diagnostics),
                key=lambda phase: 0 if phase == "edge_first" else 1,
            ),
        )
        seen_inner = False
        for group in groups:
            classes = {hole["group_boundary_class"] for hole in group}
            self.assertEqual(len(classes), 1)
            group_class = next(iter(classes))
            if group_class == "interior":
                seen_inner = True
                self.assertLessEqual(len(group), 5)
            else:
                self.assertFalse(seen_inner)
                self.assertLessEqual(len(group), 3)

    def test_edge_phase_splits_nearly_collinear_three_hole_group_for_compactness(self) -> None:
        holes = [
            _hole(index + 1, index * 20.0, 0.0, 640.0 + index * 40.0, 400.0)
            for index in range(3)
        ]

        def planner(group: list[dict], *_args: object) -> tuple[np.ndarray, dict]:
            return np.eye(4), {
                "ok": True,
                "group_bbox_px": np.asarray((600.0, 360.0, 720.0, 440.0)),
                "projected_holes_px": {
                    int(hole["hole_id"]): np.asarray(hole["initial_center_px"], dtype=float)
                    for hole in group
                },
            }

        with patch.object(localization, "_plan_batch_coarse_group_pose", side_effect=planner):
            groups, diagnostics, metadata = localization._split_batch_localization_groups_edge_first(
                holes,
                np.eye(4),
                object(),
                0.0,
                SimpleNamespace(width=1280, height=800),
                340.0,
                50.0,
                0.8,
                max_group_size=5,
                max_aspect_ratio=2.0,
                max_xy_diameter_mm=80.0,
            )

        self.assertEqual(
            [[int(hole["hole_id"]) for hole in group] for group in groups],
            [[1], [2, 3]],
        )
        self.assertTrue(all(item["planner_ok"] for item in diagnostics))
        self.assertFalse(metadata["phases"]["edge_first"]["aspect_ratio_gate_disabled_for_edge"])
        self.assertTrue(metadata["phases"]["edge_first"]["aspect_ratio_gate_applied_for_edge"])
        self.assertTrue(
            any(
                int(item["hole_ids"][0]) == 1
                and "max_aspect_ratio" in item["rejection_reasons"]
                for item in diagnostics
            )
        )

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

    def test_direct_policy_prefers_three_to_five_when_compact_cover_exists(self) -> None:
        holes = [
            _hole(1, 0.0, 0.0, 100.0, 100.0),
            _hole(2, 10.0, 0.0, 110.0, 100.0),
            _hole(3, 0.0, 10.0, 100.0, 110.0),
            _hole(4, 100.0, 0.0, 300.0, 100.0),
            _hole(5, 110.0, 0.0, 310.0, 100.0),
            _hole(6, 100.0, 10.0, 300.0, 110.0),
        ]

        groups, _, metadata = group_holes_spatially_with_metadata(
            holes,
            max_group_size=5,
            max_aspect_ratio=2.0,
            adjacency_distance_factor=1.8,
            preferred_min_group_size=3,
        )

        self.assertEqual(
            {frozenset(int(hole["hole_id"]) for hole in group) for group in groups},
            {frozenset({1, 2, 3}), frozenset({4, 5, 6})},
        )
        self.assertEqual(metadata["constraints"]["preferred_min_group_size"], 3)
        self.assertFalse(metadata["search_timed_out"])

    def test_group_search_timeout_returns_complete_audited_fallback(self) -> None:
        holes = [
            _hole(index + 1, float(index % 3) * 10.0, float(index // 3) * 10.0,
                  100.0 + float(index % 3) * 20.0, 100.0 + float(index // 3) * 20.0)
            for index in range(6)
        ]

        groups, _, metadata = group_holes_spatially_with_metadata(
            holes,
            max_group_size=5,
            max_aspect_ratio=2.0,
            adjacency_distance_factor=1.8,
            preferred_min_group_size=3,
            search_timeout_s=1.0e-9,
        )

        self.assertTrue(metadata["search_timed_out"])
        self.assertTrue(metadata["fallback_used"])
        self.assertEqual(
            sorted(int(hole["hole_id"]) for group in groups for hole in group),
            list(range(1, 7)),
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
