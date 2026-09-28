from __future__ import annotations

import math
import unittest

from tools.blend_motion_probe import parse_args, plan_square, summarize_samples

START = [0.35, -0.066, 0.237, 0.0, -0.23, 0.0]


def trace(points, speed=0.02, dt=0.01, dwell_s=0.0, duplicate_every=0):
    """Sample a polyline at constant speed, optionally resting at each vertex.

    ``duplicate_every`` repeats the previous pose every N samples, as the
    controller does when it is polled faster than it updates.
    """
    samples, t = [], 0.0

    def add(xy):
        nonlocal t
        if duplicate_every and samples and len(samples) % duplicate_every == 0:
            xy = samples[-1]["xyz_m"][:2]
        samples.append({"t_s": t, "xyz_m": [*xy, 0.237], "queue_size": 1, "blending": False})
        t += dt

    for a, b in zip(points, points[1:]):
        n = max(1, int(math.dist(a, b) / (speed * dt)))
        for i in range(n):
            add([a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n])
        for _ in range(int(dwell_s / dt)):
            add(list(b))
    add(list(points[-1]))
    return samples


class BlendMotionProbeTests(unittest.TestCase):
    def test_square_keeps_z_and_orientation_and_returns_to_start(self):
        pts = plan_square(START, 40.0, -1, 1)
        self.assertEqual(len(pts), 4)
        self.assertEqual(pts[-1], START)
        self.assertAlmostEqual(pts[0][0], START[0] - 0.04)
        self.assertAlmostEqual(pts[1][1], START[1] + 0.04)
        for p in pts:
            self.assertEqual(p[2:], START[2:])

    def test_stopped_corners_are_detected(self):
        pts = plan_square(START, 40.0, 1, 1)
        path = [START[:2], *[p[:2] for p in pts]]
        samples = trace(path, dwell_s=0.3)
        summary = summarize_samples(samples, pts, speed_m_s=0.02, blend_m=0.005)
        self.assertEqual([c["verdict"] for c in summary["corners"]], ["stopped"] * 3)
        self.assertFalse(summary["continuous"])
        self.assertAlmostEqual(summary["ideal_constant_speed_s"], 8.0)

    def test_blended_and_dropped_corners(self):
        pts = plan_square(START, 40.0, 1, 1)
        # Corner 2 skipped: the arm cuts from corner 1 straight to corner 3.
        path = [START[:2], pts[0][:2], pts[2][:2], pts[3][:2]]
        summary = summarize_samples(trace(path), pts, speed_m_s=0.02, blend_m=0.005)
        self.assertEqual([c["verdict"] for c in summary["corners"]], ["blended", "dropped", "blended"])

    def test_repeated_controller_poses_are_not_mistaken_for_stops(self):
        pts = plan_square(START, 40.0, 1, 1)
        path = [START[:2], *[p[:2] for p in pts]]
        samples = trace(path, dt=0.008, duplicate_every=3)
        summary = summarize_samples(samples, pts, speed_m_s=0.02, blend_m=0.005)
        self.assertTrue(summary["continuous"])

    def test_args_reject_unsafe_speed_and_size(self):
        with self.assertRaises(SystemExit):
            parse_args(["--speed", "0.1"])
        with self.assertRaises(SystemExit):
            parse_args(["--side-mm", "200"])
        self.assertEqual(parse_args(["--modes", "process"]).modes, ["process"])


if __name__ == "__main__":
    unittest.main()
