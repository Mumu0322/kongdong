import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from aubo_workbench.auto_sector_selection import (
    AutoSectorConfig, select_auto_holes, render_auto_sector_overlay,
    validate_selection_image, validate_region_polygon,
)
from aubo_workbench.gui_sector_editor import make_polygon_config
from aubo_workbench.sector_info import write_sector_info_snapshots


def box(x, y):
    return {"center": [x, y], "box": [x-1, y-1, x+1, y+1], "confidence": 0.9}


class PolygonSectorTests(unittest.TestCase):
    def config(self, **overrides):
        raw = {
            "partition_mode": "polygons", "image_size_px": [100, 100],
            "regions": [
                {"sector_id": 2, "polygon_px": [[40, 0], [90, 0], [90, 90], [40, 90]]},
                {"sector_id": 1, "polygon_px": [[0, 0], [50, 0], [50, 90], [0, 90]]},
            ],
        }
        raw.update(overrides)
        return AutoSectorConfig.from_dict(raw)

    def test_overlap_edge_and_vertex_have_unique_owner_without_pending(self):
        result = select_auto_holes([box(45, 40), box(50, 80), box(0, 0), box(80, 80), box(95, 95)], self.config())
        self.assertEqual([c.sector_id for c in result.candidates], [1, 1, 1, 2])
        self.assertEqual(result.selected_indices, [0, 1, 2, 3])
        self.assertEqual(result.pending_boundary_indices, [])
        self.assertTrue(all(c.assignment_status == "confirmed" for c in result.candidates))
        self.assertEqual(result.rejected[0]["reason"], "outside_drawn_regions")

    def test_active_filter_does_not_reassign_overlap_to_another_sector(self):
        result = select_auto_holes([box(45, 40), box(80, 40)], self.config(active_sector_ids=[2]))
        self.assertEqual(result.selected_indices, [1])
        self.assertEqual(result.rejected[0]["sector_id"], 1)

    def test_editor_save_roundtrip_and_archive_need_no_origin(self):
        config = make_polygon_config(self.config().regions, (100, 100))
        result = select_auto_holes([box(20, 20)], config)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        validate_selection_image(image, config)
        overlay = render_auto_sector_overlay(image, [box(20, 20)], result)
        self.assertGreater(np.count_nonzero(overlay), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            loaded = AutoSectorConfig.from_path(path)
            self.assertIsNone(loaded.origin_px)
            write_sector_info_snapshots(result, Path(tmp) / "archive")
            definition = json.loads((Path(tmp) / "archive/S01/sector_definition.json").read_text(encoding="utf-8"))
            self.assertEqual(definition["partition_mode"], "polygons")
            self.assertNotIn("angle_start_deg", definition)

    def test_wrong_image_dimensions_fail_before_selection(self):
        with self.assertRaisesRegex(ValueError, "分辨率"):
            validate_selection_image(np.zeros((50, 100, 3), dtype=np.uint8), self.config())

    def test_pose_records_roundtrip_with_polygon_config(self):
        record = {
            "record_id": "S01_20260908_120000_000",
            "sector_id": 1,
            "frame_host_timestamp_ns": 123,
            "robot_snapshot": {
                "joints_rad": [0.0] * 6,
                "tcp_pose_m_rad": [0.0] * 6,
            },
        }
        config = make_polygon_config(self.config().regions, (100, 100), pose_records=[record])
        self.assertEqual(config.to_dict()["sector_pose_records"], [record])
        loaded = AutoSectorConfig.from_dict(config.to_dict())
        self.assertEqual(loaded.sector_pose_records[0]["record_id"], record["record_id"])

    def test_degenerate_and_crossed_polygons_are_rejected(self):
        for points in ([[0,0], [1,1], [2,2]], [[0,0], [80,70], [0,90], [70,0]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                validate_region_polygon(points)


if __name__ == "__main__":
    unittest.main()
