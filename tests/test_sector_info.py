import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aubo_workbench.auto_sector_selection import AutoSectorConfig, select_auto_holes
from aubo_workbench.sector_info import write_sector_info_snapshots


def _detection(x: float, y: float) -> dict:
    return {
        "center": [x, y],
        "box": [x - 5.0, y - 5.0, x + 5.0, y + 5.0],
        "confidence": 0.9,
        "class_id": 0,
    }


class SectorInfoTests(unittest.TestCase):
    def test_writes_independent_directory_for_every_sector(self) -> None:
        config = AutoSectorConfig(
            origin_px=(100.0, 100.0), sector_count=4,
            zero_angle_deg=-1.0, boundary_margin_deg=0.0,
        )
        result = select_auto_holes([
            _detection(110.0, 100.0),
            _detection(100.0, 90.0),
        ], config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "sector_info"
            manifest = write_sector_info_snapshots(
                result,
                root,
                source_run_dir=Path(directory) / "two-stage-test",
                overlay=np.zeros((40, 50, 3), dtype=np.uint8),
            )
            self.assertEqual(set(manifest["sectors"]), {"S01", "S02", "S03", "S04"})
            self.assertTrue((root / "index.json").is_file())
            self.assertTrue((root / "runs" / "two-stage-test" / "auto_sector_selection.json").is_file())
            self.assertTrue((root / "runs" / "two-stage-test" / "auto_sector_selection_overlay.png").is_file())
            self.assertTrue((root / "S01" / "sector_definition.json").is_file())
            self.assertTrue((root / "S02" / "latest.json").is_file())
            self.assertTrue((root / "S01" / "runs" / "two-stage-test" / "sector_info.json").is_file())
            payload = json.loads((root / "S01" / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["definition"]["sector_key"], "S01")
            self.assertEqual(payload["selected_detection_indices"], [0])


if __name__ == "__main__":
    unittest.main()
