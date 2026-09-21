"""Bounded, per-hole coarse recapture scheduling (no robot I/O)."""
from __future__ import annotations


def queue_singleton_recaptures(groups, group, failed_ids, attempted, parent_index):
    queued = []
    for hole in group:
        hole_id = int(hole["hole_id"])
        if hole_id not in failed_ids or hole_id in attempted or hole.get("coarse_quality_retry"):
            continue
        # Keep the original navigation seed: failed geometry must not become
        # an unchecked robot target. The normal planning/motion gates still run.
        retry = dict(hole)
        retry.pop("coarse_pose_refined_center_base_mm", None)
        retry.pop("coarse_pose_refined_normal_base", None)
        retry["coarse_quality_retry"] = True
        retry["coarse_quality_retry_parent_group"] = int(parent_index)
        groups.append([retry])
        attempted.add(hole_id)
        queued.append(hole_id)
    return queued
