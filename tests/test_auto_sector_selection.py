import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aubo_workbench.auto_sector_selection import (
    AutoSectorConfig,
    load_auto_sector_config,
    render_auto_sector_overlay,
    order_holes_row_major,
    row_major_layout,
    sector_angle_and_id,
    select_auto_holes,
)


def detection(x, y, confidence=0.9, size=12):
    half = size / 2.0
    return {
        "class_id": 0,
        "class_name": "hole",
        "confidence": confidence,
        "center": [float(x), float(y)],
        "box": [float(x - half), float(y - half), float(x + half), float(y + half)],
    }


class AutoSectorSelectionTests(unittest.TestCase):
    def test_row_major_order_is_independent_of_detection_order(self):
        items = [
            {"detection_index": 4, "center": [220, 205], "box": [214, 199, 226, 211]},
            {"detection_index": 1, "center": [120, 102], "box": [114, 96, 126, 108]},
            {"detection_index": 3, "center": [220, 101], "box": [214, 95, 226, 107]},
            {"detection_index": 2, "center": [120, 205], "box": [114, 199, 126, 211]},
        ]
        ordered = order_holes_row_major([items[2], items[0], items[3], items[1]])
        self.assertEqual([item["detection_index"] for item in ordered], [1, 3, 2, 4])
        layout = row_major_layout(items)
        self.assertEqual(
            [(item["item"]["detection_index"], item["row"], item["column"]) for item in layout],
            [(1, 1, 1), (3, 1, 2), (2, 2, 1), (4, 2, 2)],
        )

    def test_row_major_order_groups_small_vertical_jitter(self):
        items = [
            {"detection_index": 0, "center": [100, 100], "box": [94, 94, 106, 106]},
            {"detection_index": 1, "center": [140, 104], "box": [134, 98, 146, 110]},
            {"detection_index": 2, "center": [100, 135], "box": [94, 129, 106, 141]},
        ]
        ordered = order_holes_row_major(items)
        self.assertEqual([item["detection_index"] for item in ordered], [0, 1, 2])

    def test_sector_assignment_uses_zero_angle_and_image_y_direction(self):
        config = AutoSectorConfig(origin_px=(100.0, 100.0), sector_count=4)
        self.assertEqual(sector_angle_and_id((110, 100), config)[1], 1)
        self.assertEqual(sector_angle_and_id((100, 90), config)[1], 2)
        self.assertEqual(sector_angle_and_id((90, 100), config)[1], 3)
        self.assertEqual(sector_angle_and_id((100, 110), config)[1], 4)

    def test_boundary_is_explicit_and_can_be_excluded(self):
        config = AutoSectorConfig(
            origin_px=(0.0, 0.0), sector_count=4,
            boundary_margin_deg=2.0, include_boundary_candidates=False,
        )
        result = select_auto_holes([detection(1, 0)], config)
        self.assertEqual(result.selected_indices, [])
        self.assertEqual(result.candidates[0].assignment_status, "pending_boundary")
        self.assertTrue(any("边界" in warning for warning in result.warnings))

    def test_roi_confidence_and_dedup_are_reported(self):
        config = AutoSectorConfig(
            origin_px=(100.0, 100.0), sector_count=4,
            zero_angle_deg=-1.0,
            roi_polygon_px=((80, 80), (120, 80), (120, 120), (80, 120)),
            dedup_distance_px=4.0,
            boundary_margin_deg=0.0,
        )
        result = select_auto_holes([
            detection(110, 100, confidence=0.95),
            detection(111, 100, confidence=0.80),
            detection(90, 100, confidence=0.20),
            detection(140, 100, confidence=0.90),
        ], config)
        self.assertEqual(result.selected_indices, [0])
        self.assertEqual(result.duplicate_groups, [[0, 1]])
        reasons = {item["detection_index"]: item["reason"] for item in result.rejected}
        self.assertEqual(reasons[2], "confidence_below_threshold")
        self.assertEqual(reasons[3], "outside_experiment_roi")

    def test_dedup_requires_box_overlap_so_adjacent_holes_are_retained(self):
        config = AutoSectorConfig(
            origin_px=(0.0, 0.0), sector_count=4, dedup_distance_px=25.0,
            boundary_margin_deg=0.0,
        )
        result = select_auto_holes([
            detection(20, -5, confidence=0.95, size=8),
            detection(35, -5, confidence=0.80, size=8),
            detection(50, -5, confidence=0.70, size=8),
        ], config)
        self.assertEqual(result.selected_indices, [0, 1, 2])
        self.assertEqual(result.duplicate_groups, [])

    def test_dedup_keeps_high_confidence_box_for_same_hole(self):
        config = AutoSectorConfig(
            origin_px=(0.0, 0.0), sector_count=4,
            dedup_distance_px=10.0, boundary_margin_deg=0.0,
        )
        result = select_auto_holes([
            detection(20, -5, confidence=0.95, size=12),
            detection(21, -4, confidence=0.80, size=12),
        ], config)
        self.assertEqual(result.selected_indices, [0])
        self.assertEqual(result.duplicate_groups, [[0, 1]])

    def test_reference_catalogue_reports_complete_or_missing(self):
        config = AutoSectorConfig(
            origin_px=(100.0, 100.0), sector_count=4,
            zero_angle_deg=-1.0,
            dedup_distance_px=4.0,
            boundary_margin_deg=0.0,
            reference_holes=(
                {"global_hole_key": "A", "center_px": [110, 100], "sector_id": 1},
                {"global_hole_key": "B", "center_px": [100, 90], "sector_id": 2},
            ),
        )
        complete = select_auto_holes([detection(110, 100), detection(100, 90)], config)
        self.assertEqual(complete.coverage["status"], "complete")
        self.assertEqual(complete.to_dict()["selected_global_hole_keys"], ["A", "B"])
        incomplete = select_auto_holes([detection(110, 100)], config)
        self.assertEqual(incomplete.coverage["status"], "incomplete")
        self.assertEqual(incomplete.coverage["missing_reference_keys"], ["B"])

    def test_reference_matching_can_reassign_a_previous_candidate(self):
        config = AutoSectorConfig(
            origin_px=(100.0, 100.0), sector_count=4,
            zero_angle_deg=-1.0, boundary_margin_deg=0.0,
            dedup_distance_px=2.0, reference_match_distance_px=12.0,
            reference_holes=(
                {"global_hole_key": "A", "center_px": [100, 100]},
                {"global_hole_key": "B", "center_px": [120, 100]},
            ),
        )
        result = select_auto_holes([
            detection(109, 100, confidence=0.95),
            detection(100, 100, confidence=0.80),
        ], config)
        identities = {
            item.detection_index: item.global_hole_key
            for item in result.candidates
        }
        self.assertEqual(identities, {0: "B", 1: "A"})
        self.assertEqual(result.coverage["status"], "complete")

    def test_boundary_candidates_never_enter_executable_selection(self):
        config = AutoSectorConfig(
            origin_px=(0.0, 0.0), sector_count=4,
            boundary_margin_deg=2.0, include_boundary_candidates=True,
        )
        result = select_auto_holes([detection(20, 0)], config)
        self.assertEqual(result.selected_indices, [])
        self.assertEqual(result.pending_boundary_indices, [0])
        self.assertEqual(
            result.coverage["pending_boundary_detection_indices"], [0]
        )

    def test_reference_coverage_is_limited_to_active_sectors(self):
        config = AutoSectorConfig(
            origin_px=(100.0, 100.0), sector_count=4,
            zero_angle_deg=-1.0, boundary_margin_deg=0.0,
            active_sector_ids=(1,), dedup_distance_px=2.0,
            reference_holes=(
                {"global_hole_key": "A", "center_px": [110, 100], "sector_id": 1},
                {"global_hole_key": "B", "center_px": [100, 90], "sector_id": 2},
            ),
        )
        result = select_auto_holes([detection(110, 100)], config)
        self.assertEqual(result.coverage["status"], "complete")
        self.assertEqual(result.coverage["reference_count"], 1)
        self.assertEqual(result.coverage["out_of_scope_reference_count"], 1)

    def test_overlay_and_config_loader_are_serializable(self):
        config = AutoSectorConfig(
            origin_px=(50.0, 50.0), sector_count=6, boundary_margin_deg=0.0,
        )
        result = select_auto_holes([detection(60, 50)], config)
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        overlay = render_auto_sector_overlay(image, [detection(60, 50)], result)
        self.assertEqual(overlay.shape, image.shape)
        self.assertGreater(int(np.count_nonzero(overlay)), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            loaded = load_auto_sector_config(path, active_sector_ids=[1])
            self.assertEqual(loaded.active_sector_ids, (1,))

    def test_missing_origin_is_rejected(self):
        with self.assertRaises(ValueError):
            AutoSectorConfig.from_dict({"sector_count": 6})

    def test_malformed_detection_is_reported_without_aborting_batch(self):
        config = AutoSectorConfig(origin_px=(0.0, 0.0), boundary_margin_deg=0.0)
        result = select_auto_holes([None, detection(20, -5)], config)
        self.assertEqual(result.selected_indices, [1])
        self.assertEqual(result.rejected[0]["reason"], "malformed_detection")


if __name__ == "__main__":
    unittest.main()
