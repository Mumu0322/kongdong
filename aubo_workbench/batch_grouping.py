"""Spatial grouping helpers for shared coarse/fine hole localization.

The grouping policy deliberately separates two concerns:

* the home-image coordinates determine a deterministic processing order
  (left-to-right, then front-to-back), and
* the robot/base coordinates determine whether holes are actually neighbours
  and whether a group is compact enough for a shared observation.

This prevents a visually ordered list from accidentally creating long strips
or groups containing holes that are far apart in 3-D space.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np


Planner = Callable[[list[dict[str, Any]]], Mapping[str, Any]]


def _point3(value: Any) -> np.ndarray | None:
    try:
        point = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if point.size < 3 or not np.all(np.isfinite(point[:3])):
        return None
    return point[:3].copy()


def _point2(value: Any) -> tuple[float, float] | None:
    try:
        point = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if point.size < 2 or not np.all(np.isfinite(point[:2])):
        return None
    return float(point[0]), float(point[1])


def _hole_id(hole: Mapping[str, Any], fallback: int) -> str:
    value = hole.get("hole_id", fallback)
    return str(value)


def _scan_key(hole: Mapping[str, Any], fallback: int) -> tuple[float, float, str, int]:
    """Return the deterministic home-image order key.

    ``u`` is the horizontal image coordinate and ``v`` the vertical one.  A
    small quantisation of ``v`` makes the order read as rows: left-to-right in
    one row, then the next row from front to back.  If home pixels are not
    available, the original list position remains a stable fallback.
    """

    point = _point2(hole.get("initial_center_px"))
    if point is None:
        point = _point2(hole.get("home_center_px"))
    if point is None:
        return (float(fallback), float(fallback), _hole_id(hole, fallback), fallback)

    u, v = point
    return (round(v / 25.0), u, _hole_id(hole, fallback), fallback)


def _normal(hole: Mapping[str, Any]) -> np.ndarray | None:
    for key in (
        "planning_normal_base",
        "initial_plane_normal_base",
        "plane_normal_base",
        "normal_base",
    ):
        value = _point3(hole.get(key))
        if value is not None:
            norm = float(np.linalg.norm(value))
            if norm > 1e-9:
                return value / norm
    return None


def _plan_summary(plan: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not plan:
        return None
    summary: dict[str, Any] = {}
    for key in (
        "ok",
        "group_bbox_px",
        "group_center_px",
        "projected_holes_px",
        "projected_bbox_span_px",
        "projected_bbox_span_ratio",
        "view_span_ratio",
        "error",
    ):
        if key not in plan:
            continue
        value = plan[key]
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, Mapping):
            value = dict(value)
        elif isinstance(value, (list, tuple)):
            value = list(value)
        elif isinstance(value, (np.floating, np.integer)):
            value = value.item()
        summary[key] = value
    return summary


def _pairwise_distances(points: Sequence[np.ndarray]) -> np.ndarray:
    count = len(points)
    distances = np.full((count, count), np.inf, dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            distance = float(np.linalg.norm(points[i] - points[j]))
            distances[i, j] = distance
            distances[j, i] = distance
    np.fill_diagonal(distances, 0.0)
    return distances


def _adjacency_limit(distances: np.ndarray, factor: float) -> tuple[float, float]:
    count = distances.shape[0]
    if count <= 1:
        return 0.0, 0.0
    nearest = np.min(np.where(np.eye(count, dtype=bool), np.inf, distances), axis=1)
    nearest = nearest[np.isfinite(nearest) & (nearest > 1e-9)]
    if nearest.size == 0:
        return 0.0, 0.0
    median_nn = float(np.median(nearest))
    # The upper bound avoids turning a sparse set into one connected group.
    limit = median_nn * max(1.0, float(factor))
    return median_nn, limit


def _group_aspect_ratio(points: Sequence[np.ndarray]) -> float:
    if len(points) <= 2:
        return 1.0
    matrix = np.asarray(points, dtype=np.float64)
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    if np.max(np.linalg.norm(centered, axis=1)) <= 1e-9:
        return 1.0
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    if singular_values.size < 2 or singular_values[1] <= 1e-9:
        # A collinear group is deliberately rejected by the aspect-ratio gate.
        # Keep the diagnostic JSON finite as well; Python's default ``Infinity``
        # output is not valid JSON for downstream PLC/report consumers.
        return 1.0e9
    return float(singular_values[0] / singular_values[1])


def _group_diameter(points: Sequence[np.ndarray]) -> float:
    if len(points) <= 1:
        return 0.0
    return float(np.max(_pairwise_distances(points)))


def _group_xy_diameter(points: Sequence[np.ndarray]) -> float:
    """Return the lateral footprint diameter in the robot base XY plane.

    The Z value contains the estimated surface height and can vary because of
    a tilted/uneven workpiece.  It must not make an otherwise local XY cluster
    fail the grouping gate, so compactness is measured in the horizontal
    plane used for shared camera placement.
    """
    if len(points) <= 1:
        return 0.0
    xy_points = [np.asarray(point, dtype=np.float64)[:2] for point in points]
    return float(np.max(_pairwise_distances(xy_points)))


def _group_centroid_radius(points: Sequence[np.ndarray]) -> float:
    if len(points) <= 1:
        return 0.0
    xy_points = np.asarray(
        [np.asarray(point, dtype=np.float64)[:2] for point in points],
        dtype=np.float64,
    )
    centroid = np.mean(xy_points, axis=0, keepdims=True)
    return float(np.max(np.linalg.norm(xy_points - centroid, axis=1)))


def _group_mean_pairwise_distance(points: Sequence[np.ndarray]) -> float:
    if len(points) <= 1:
        return 0.0
    xy_points = [np.asarray(point, dtype=np.float64)[:2] for point in points]
    distances = _pairwise_distances(xy_points)
    values = distances[np.triu_indices(len(xy_points), k=1)]
    return float(np.mean(values)) if values.size else 0.0


def _normal_spread_deg(normals: Sequence[np.ndarray]) -> float:
    if len(normals) <= 1:
        return 0.0
    reference = np.sum(np.asarray(normals, dtype=np.float64), axis=0)
    if np.linalg.norm(reference) <= 1e-9:
        reference = normals[0]
    reference = reference / max(float(np.linalg.norm(reference)), 1e-9)
    angles: list[float] = []
    for normal in normals:
        aligned = normal if float(np.dot(normal, reference)) >= 0.0 else -normal
        cosine = float(np.clip(np.dot(aligned, reference), -1.0, 1.0))
        angles.append(math.degrees(math.acos(cosine)))
    return float(max(angles)) if angles else 0.0


def _depth_span(points: Sequence[np.ndarray], normals: Sequence[np.ndarray]) -> float:
    if len(points) <= 1 or not normals:
        return 0.0
    reference = np.sum(np.asarray(normals, dtype=np.float64), axis=0)
    norm = float(np.linalg.norm(reference))
    if norm <= 1e-9:
        return 0.0
    reference /= norm
    projected = [float(np.dot(point, reference)) for point in points]
    return float(max(projected) - min(projected))


def _group_metrics(
    group: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for hole in group:
        point = _point3(hole.get("initial_center_base_mm"))
        if point is not None:
            points.append(point)
        normal = _normal(hole)
        if normal is not None:
            normals.append(normal)
    return {
        "hole_ids": [_hole_id(hole, index) for index, hole in enumerate(group)],
        "count": len(group),
        "aspect_ratio": _group_aspect_ratio(points) if points else None,
        "diameter_mm": _group_diameter(points) if points else None,
        "xy_diameter_mm": _group_xy_diameter(points) if points else None,
        "centroid_radius_mm": _group_centroid_radius(points) if points else None,
        "mean_pairwise_distance_mm": (
            _group_mean_pairwise_distance(points) if points else None
        ),
        "normal_spread_deg": _normal_spread_deg(normals) if normals else None,
        "depth_span_mm": _depth_span(points, normals) if points and normals else None,
    }


def group_holes_spatially(
    holes: Sequence[Mapping[str, Any]],
    *,
    planner: Planner | None = None,
    max_group_size: int | None = None,
    max_aspect_ratio: float | None = None,
    adjacency_distance_factor: float = 1.8,
    max_normal_spread_deg: float | None = None,
    max_depth_span_mm: float | None = None,
    max_xy_diameter_mm: float | None = None,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Group holes into the minimum number of feasible compact clusters.

    The public two-value return shape is kept for existing callers.  The
    actual solver is an exact-cover search over connected candidate groups;
    callers that need the search audit can use
    :func:`group_holes_spatially_with_metadata`.
    """
    groups, diagnostics, _ = group_holes_spatially_with_metadata(
        holes,
        planner=planner,
        max_group_size=max_group_size,
        max_aspect_ratio=max_aspect_ratio,
        adjacency_distance_factor=adjacency_distance_factor,
        max_normal_spread_deg=max_normal_spread_deg,
        max_depth_span_mm=max_depth_span_mm,
        max_xy_diameter_mm=max_xy_diameter_mm,
    )
    return groups, diagnostics


def group_holes_spatially_with_metadata(
    holes: Sequence[Mapping[str, Any]],
    *,
    planner: Planner | None = None,
    max_group_size: int | None = None,
    max_aspect_ratio: float | None = None,
    adjacency_distance_factor: float = 1.8,
    max_normal_spread_deg: float | None = None,
    max_depth_span_mm: float | None = None,
    max_xy_diameter_mm: float | None = None,
) -> tuple[
    list[list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Return a minimum-cardinality feasible partition and its search audit.

    A group is feasible only when it satisfies all enabled spatial, geometric
    and planner gates.  The solver first tries the theoretical lower bound
    ``ceil(hole_count / max_group_size)`` and then increases the group count
    only when no exact cover exists at the previous count.  This is important
    for the shared-camera workflow: a greedy seed-and-grow pass can create an
    unnecessarily large number of robot moves even when a different compact
    partition would cover the same holes with fewer observations.

    Candidate groups are generated by connected spatial growth, so the exact
    cover search remains limited to physically local groups rather than every
    arbitrary subset.  If no full cover satisfies the planner, a clearly
    marked best-effort partition is returned so the caller can preserve every
    selected hole and send rejected groups to its existing fallback path.
    """

    records = [dict(hole) for hole in holes]
    if not records:
        return [], [], {
            "algorithm": "minimum_feasible_exact_cover_compact_v2",
            "hole_count": 0,
            "theoretical_min_group_count": 0,
            "minimum_feasible_group_count": 0,
            "search_complete": True,
            "fallback_used": False,
        }

    if max_group_size is not None and int(max_group_size) < 1:
        raise ValueError("max_group_size must be at least 1")
    if max_aspect_ratio is not None and float(max_aspect_ratio) < 1.0:
        raise ValueError("max_aspect_ratio must be at least 1.0")
    if float(adjacency_distance_factor) < 1.0:
        raise ValueError("adjacency_distance_factor must be at least 1.0")
    if max_xy_diameter_mm is not None:
        if not math.isfinite(float(max_xy_diameter_mm)) or float(max_xy_diameter_mm) <= 0.0:
            raise ValueError("max_xy_diameter_mm must be a positive finite number")

    indexed = list(enumerate(records))
    indexed.sort(key=lambda item: _scan_key(item[1], item[0]))
    ordered_indices = [index for index, _ in indexed]
    scan_rank = {index: rank for rank, index in enumerate(ordered_indices)}

    fallback_points: list[np.ndarray] = []
    for position, hole in enumerate(records):
        point = _point3(hole.get("initial_center_base_mm"))
        if point is None:
            # Missing base coordinates cannot be spatially related safely.  A
            # unique synthetic point keeps such holes isolated and recoverable.
            point = np.asarray((position * 1_000_000.0, 0.0, 0.0), dtype=np.float64)
        fallback_points.append(point)

    distances = _pairwise_distances(fallback_points)
    xy_distances = _pairwise_distances([
        point[:2] for point in fallback_points
    ])
    median_nn, adjacency_limit = _adjacency_limit(
        distances, float(adjacency_distance_factor)
    )

    def is_adjacent(left: int, right: int) -> bool:
        if adjacency_limit <= 0.0:
            return False
        return bool(distances[left, right] <= adjacency_limit + 1e-9)

    effective_max_group_size = (
        len(records)
        if max_group_size is None
        else min(len(records), int(max_group_size))
    )

    def ordered_tuple(indices: Sequence[int]) -> tuple[int, ...]:
        return tuple(sorted((int(index) for index in indices), key=lambda index: scan_rank[index]))

    def connected(indices: Sequence[int]) -> bool:
        values = list(indices)
        if len(values) <= 1:
            return True
        visited = {values[0]}
        frontier = [values[0]]
        while frontier:
            current = frontier.pop()
            for candidate in values:
                if candidate in visited or not is_adjacent(current, candidate):
                    continue
                visited.add(candidate)
                frontier.append(candidate)
        return len(visited) == len(values)

    geometry_cache: dict[tuple[int, ...], dict[str, Any]] = {}
    plan_cache: dict[tuple[int, ...], dict[str, Any]] = {}

    def evaluate_geometry(indices: Sequence[int]) -> dict[str, Any]:
        key = ordered_tuple(indices)
        cached = geometry_cache.get(key)
        if cached is not None:
            return cached
        values = list(key)
        result: dict[str, Any] = {
            "ok": True,
            "reason": None,
            "details": {},
        }
        if len(values) > effective_max_group_size:
            result.update({"ok": False, "reason": "max_group_size"})
            geometry_cache[key] = result
            return result
        if not connected(values):
            result.update({"ok": False, "reason": "not_connected"})
            geometry_cache[key] = result
            return result

        points = [fallback_points[index] for index in values]
        aspect = _group_aspect_ratio(points)
        result["details"]["aspect_ratio"] = aspect
        if max_aspect_ratio is not None and aspect > float(max_aspect_ratio) + 1e-9:
            result.update({
                "ok": False,
                "reason": "max_aspect_ratio",
                "details": {"aspect_ratio": aspect},
            })
            geometry_cache[key] = result
            return result

        xy_diameter = _group_xy_diameter(points)
        result["details"]["xy_diameter_mm"] = xy_diameter
        result["details"]["centroid_radius_mm"] = _group_centroid_radius(points)
        result["details"]["mean_pairwise_distance_mm"] = (
            _group_mean_pairwise_distance(points)
        )
        if (
            max_xy_diameter_mm is not None
            and xy_diameter > float(max_xy_diameter_mm) + 1e-9
        ):
            result.update({
                "ok": False,
                "reason": "max_xy_diameter_mm",
                "details": {
                    "xy_diameter_mm": xy_diameter,
                    "max_xy_diameter_mm": float(max_xy_diameter_mm),
                    "centroid_radius_mm": result["details"]["centroid_radius_mm"],
                    "mean_pairwise_distance_mm": result["details"]["mean_pairwise_distance_mm"],
                },
            })
            geometry_cache[key] = result
            return result

        normals = [_normal(records[index]) for index in values]
        valid_normals = [normal for normal in normals if normal is not None]
        normal_spread = _normal_spread_deg(valid_normals) if valid_normals else 0.0
        result["details"]["normal_spread_deg"] = normal_spread
        if (
            max_normal_spread_deg is not None
            and valid_normals
            and normal_spread > float(max_normal_spread_deg) + 1e-9
        ):
            result.update({
                "ok": False,
                "reason": "max_normal_spread_deg",
                "details": {"normal_spread_deg": normal_spread},
            })
            geometry_cache[key] = result
            return result

        depth_span = _depth_span(points, valid_normals)
        result["details"]["depth_span_mm"] = depth_span
        if max_depth_span_mm is not None and depth_span > float(max_depth_span_mm) + 1e-9:
            result.update({
                "ok": False,
                "reason": "max_depth_span_mm",
                "details": {"depth_span_mm": depth_span},
            })
            geometry_cache[key] = result
            return result

        geometry_cache[key] = result
        return result

    def evaluate_candidate(indices: Sequence[int]) -> dict[str, Any]:
        key = ordered_tuple(indices)
        cached = plan_cache.get(key)
        if cached is not None:
            return cached
        geometry = evaluate_geometry(key)
        result: dict[str, Any] = {
            "indices": key,
            "mask": sum(1 << int(index) for index in key),
            "geometry_ok": bool(geometry["ok"]),
            "geometry_reason": geometry.get("reason"),
            "geometry_details": dict(geometry.get("details") or {}),
            "planner_ok": True,
            "planner_reason": None,
            "planner": None,
        }
        if not geometry["ok"]:
            result["ok"] = False
            plan_cache[key] = result
            return result
        if planner is not None:
            candidate_holes = [records[index] for index in key]
            try:
                plan = planner(candidate_holes)
            except Exception as exc:  # planner errors must split, not lose holes
                result.update({
                    "ok": False,
                    "planner_ok": False,
                    "planner_reason": "planner_error",
                    "planner": {"error": f"{type(exc).__name__}: {exc}"},
                })
                plan_cache[key] = result
                return result
            plan_summary = _plan_summary(plan)
            if not plan or plan.get("ok", True) is False:
                result.update({
                    "ok": False,
                    "planner_ok": False,
                    "planner_reason": "planner_rejected",
                    "planner": plan_summary,
                })
                plan_cache[key] = result
                return result
            result["planner"] = plan_summary
        result["ok"] = True
        plan_cache[key] = result
        return result

    # Candidate streams are generated lazily.  Eagerly enumerating every
    # connected subset is needlessly expensive for a 9-hole coarse cap; the
    # exact-cover search normally finds a lower-bound solution after inspecting
    # only the largest compact candidates for a few scan-order seeds.
    candidates_by_seed: dict[int, list[dict[str, Any]]] = {}
    soft_candidates_by_seed: dict[int, list[dict[str, Any]]] = {}
    rejection_by_seed: dict[int, dict[str, int]] = {}
    candidate_streams: dict[int, Any] = {}
    candidate_stream_exhausted: set[int] = set()
    generated_seeds: set[int] = set()
    candidate_generation_nodes = 0

    def candidate_key(item: dict[str, Any]) -> tuple[Any, ...]:
        details = item.get("geometry_details", {}) or {}
        return (
            float(details.get("xy_diameter_mm", 1.0e9)),
            float(details.get("centroid_radius_mm", 1.0e9)),
            float(details.get("mean_pairwise_distance_mm", 1.0e9)),
            float(details.get("aspect_ratio", 1.0e9)),
            -len(item["indices"]),
            tuple(scan_rank[index] for index in item["indices"]),
        )

    def ensure_candidate_stream(seed: int) -> None:
        nonlocal candidate_generation_nodes
        if seed in candidate_streams:
            return
        generated_seeds.add(seed)
        seed_rank = scan_rank[seed]
        all_future = {
            index for index in ordered_indices
            if scan_rank[index] > seed_rank
        }
        seen: set[tuple[int, ...]] = set()
        rejection_reasons: dict[str, int] = {}

        def record_rejection(entry: dict[str, Any]) -> None:
            reason = entry.get("geometry_reason") or entry.get("planner_reason")
            if reason:
                rejection_reasons[str(reason)] = rejection_reasons.get(str(reason), 0) + 1

        def enumerate_exact(
            current: tuple[int, ...], target_size: int,
            current_xy_diameter: float = 0.0,
        ) -> Any:
            nonlocal candidate_generation_nodes
            candidate_generation_nodes += 1
            # XY diameter is monotonic when a point is added.  Rejecting the
            # partial branch here is essential for a dense selection: without
            # it the solver would enumerate every large connected subset only
            # to reject it at the leaf, which can stall the live workflow.
            if (
                max_xy_diameter_mm is not None
                and len(current) > 1
                and current_xy_diameter > float(max_xy_diameter_mm) + 1e-9
            ):
                return
            if len(current) == target_size:
                if current in seen:
                    return
                seen.add(current)
                entry = evaluate_candidate(current)
                if entry.get("geometry_ok"):
                    soft_candidates_by_seed.setdefault(seed, []).append(entry)
                    if entry.get("ok"):
                        candidates_by_seed.setdefault(seed, []).append(entry)
                        yield entry
                    else:
                        record_rejection(entry)
                else:
                    record_rejection(entry)
                return

            frontier = [
                index for index in all_future
                if index not in current
                and any(is_adjacent(index, member) for member in current)
            ]
            frontier.sort(key=lambda index: (
                min((distances[index, member] for member in current), default=np.inf),
                scan_rank[index],
            ))
            for candidate in frontier:
                proposed = ordered_tuple((*current, candidate))
                proposed_xy_diameter = max(
                    current_xy_diameter,
                    max(
                        (float(xy_distances[candidate, member]) for member in current),
                        default=0.0,
                    ),
                )
                yield from enumerate_exact(
                    proposed, target_size, proposed_xy_diameter,
                )

        def stream() -> Any:
            # Larger candidates are still generated first to keep the fallback
            # behaviour useful.  The exact-cover solver below evaluates all
            # feasible covers at the selected group count and then chooses the
            # most compact one.
            for target_size in range(effective_max_group_size, 0, -1):
                yield from enumerate_exact((seed,), target_size)

        candidate_streams[seed] = stream()
        candidates_by_seed.setdefault(seed, [])
        soft_candidates_by_seed.setdefault(seed, [])
        rejection_by_seed[seed] = rejection_reasons

    def iter_hard_candidates(seed: int, minimum_size: int = 1) -> Any:
        ensure_candidate_stream(seed)
        hard_candidates = candidates_by_seed[seed]
        yielded = 0
        while True:
            while yielded < len(hard_candidates):
                candidate = hard_candidates[yielded]
                yielded += 1
                if len(candidate["indices"]) >= int(minimum_size):
                    yield candidate
            if seed in candidate_stream_exhausted:
                return
            try:
                next(candidate_streams[seed])
            except StopIteration:
                candidate_stream_exhausted.add(seed)
                # Stable sorting makes compact candidates available first and
                # keeps equal-score choices deterministic.
                hard_candidates.sort(key=candidate_key)
                soft_candidates_by_seed[seed].sort(
                    key=lambda item: (candidate_key(item), 0 if item.get("ok") else 1)
                )
                continue

    def ensure_all_candidates(seed: int) -> None:
        for _ in iter_hard_candidates(seed, 1):
            pass

    full_mask = (1 << len(records)) - 1
    theoretical_minimum = int(math.ceil(
        len(records) / max(1, effective_max_group_size)
    ))
    search_attempts: list[dict[str, Any]] = []
    search_nodes_total = 0

    def first_remaining_index(mask: int) -> int | None:
        for index in ordered_indices:
            if mask & (1 << index):
                return index
        return None

    def solution_score(solution: Sequence[dict[str, Any]]) -> tuple[Any, ...]:
        """Score a partition after its group count has been fixed.

        The old solver returned the first feasible exact cover.  That favoured
        the largest candidate generated from the first scan-order seed and
        allowed a long neighbour chain to win.  This lexicographic score makes
        the widest group the first decision, then resolves ties with the
        centroid radius, average pair distance, shape, size and scan order.
        """
        compactness: list[tuple[float, float, float, float, int, tuple[int, ...]]] = []
        for item in solution:
            details = item.get("geometry_details", {}) or {}
            compactness.append((
                float(details.get("xy_diameter_mm", 1.0e9)),
                float(details.get("centroid_radius_mm", 1.0e9)),
                float(details.get("mean_pairwise_distance_mm", 1.0e9)),
                float(details.get("aspect_ratio", 1.0e9)),
                -len(item["indices"]),
                tuple(scan_rank[index] for index in item["indices"]),
            ))
        compactness.sort(
            key=lambda value: (
                -value[0],
                -value[1],
                -value[2],
                -value[3],
                value[4],
                value[5],
            )
        )
        scan_keys = tuple(sorted(value[5] for value in compactness))
        return (
            tuple(value[0] for value in compactness),
            tuple(value[1] for value in compactness),
            tuple(value[2] for value in compactness),
            tuple(value[3] for value in compactness),
            tuple(value[4] for value in compactness),
            scan_keys,
        )

    def find_exact_cover(target_group_count: int) -> tuple[list[dict[str, Any]] | None, int]:
        nodes = 0
        state_cache: dict[tuple[int, int], list[dict[str, Any]] | None] = {}

        def search(remaining_mask: int, slots: int) -> list[dict[str, Any]] | None:
            nonlocal nodes
            nodes += 1
            remaining_count = remaining_mask.bit_count()
            if remaining_count == 0:
                return [] if slots == 0 else None
            if slots <= 0 or remaining_count < slots:
                return None
            if remaining_count > slots * effective_max_group_size:
                return None
            state = (remaining_mask, slots)
            if state in state_cache:
                cached = state_cache[state]
                return None if cached is None else list(cached)
            seed = first_remaining_index(remaining_mask)
            if seed is None:
                solution = [] if slots == 0 else None
                state_cache[state] = solution
                return None if solution is None else list(solution)
            minimum_candidate_size = max(
                1,
                remaining_count - (slots - 1) * effective_max_group_size,
            )
            best_solution: list[dict[str, Any]] | None = None
            best_score: tuple[Any, ...] | None = None
            for candidate in iter_hard_candidates(seed, minimum_candidate_size):
                candidate_mask = int(candidate["mask"])
                if candidate_mask & remaining_mask != candidate_mask:
                    continue
                next_remaining = remaining_mask ^ candidate_mask
                tail = search(next_remaining, slots - 1)
                if tail is not None:
                    solution = [candidate, *tail]
                    score = solution_score(solution)
                    if best_score is None or score < best_score:
                        best_solution = solution
                        best_score = score
            state_cache[state] = best_solution
            return None if best_solution is None else list(best_solution)

        return search(full_mask, int(target_group_count)), nodes

    selected_entries: list[dict[str, Any]] | None = None
    minimum_feasible_group_count: int | None = None
    for target_group_count in range(theoretical_minimum, len(records) + 1):
        solution, node_count = find_exact_cover(target_group_count)
        search_nodes_total += int(node_count)
        search_attempts.append({
            "group_count": int(target_group_count),
            "feasible": solution is not None,
            "search_nodes": int(node_count),
        })
        if solution is not None:
            selected_entries = solution
            minimum_feasible_group_count = int(target_group_count)
            break

    fallback_used = selected_entries is None
    fallback_reason = None
    if selected_entries is None:
        fallback_reason = "no_exact_cover_under_all_constraints"
        selected_entries = []
        remaining_mask = full_mask
        while remaining_mask:
            seed = first_remaining_index(remaining_mask)
            if seed is None:
                break
            ensure_all_candidates(seed)
            available = [
                item for item in soft_candidates_by_seed.get(seed, [])
                if int(item["mask"]) & remaining_mask == int(item["mask"])
            ]
            if not available:
                # This can only happen when a custom planner rejects even the
                # singleton.  Preserve the hole as an explicitly rejected
                # singleton rather than silently dropping it.
                singleton = evaluate_candidate((seed,))
                singleton = dict(singleton)
                singleton["ok"] = False
                singleton["planner_ok"] = False
                singleton["planner_reason"] = (
                    singleton.get("planner_reason") or "no_feasible_candidate"
                )
                available = [singleton]
            chosen = available[0]
            selected_entries.append(chosen)
            remaining_mask ^= int(chosen["mask"])
        minimum_feasible_group_count = None

    # The exact-cover search is anchored at the first remaining scan-order
    # hole.  Sort the selected entries once more so report order is always the
    # requested left-to-right, front-to-back order, irrespective of recursion.
    selected_entries.sort(
        key=lambda item: min(scan_rank[index] for index in item["indices"])
    )
    groups = [
        [records[index] for index in entry["indices"]]
        for entry in selected_entries
    ]

    search_metadata: dict[str, Any] = {
        "algorithm": "minimum_feasible_exact_cover_compact_v2",
        "hole_count": len(records),
        "theoretical_min_group_count": theoretical_minimum,
        "minimum_feasible_group_count": minimum_feasible_group_count,
        "selected_group_count": len(groups),
        "max_group_size": int(effective_max_group_size),
        "search_complete": not fallback_used,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "search_nodes": int(search_nodes_total),
        "candidate_generation_nodes": int(candidate_generation_nodes),
        "candidate_seeds_generated": len(generated_seeds),
        "selection_objective": (
            "minimum_group_count_then_minimize_sorted_group_xy_diameter_and_"
            "centroid_radius_then_mean_pairwise_distance"
        ),
        "search_attempts": search_attempts,
        "group_count_increased": bool(
            minimum_feasible_group_count is not None
            and minimum_feasible_group_count > theoretical_minimum
        ),
        "group_count_increase_reason": (
            "lower_bound_group_count_has_no_full_feasible_cover"
            if minimum_feasible_group_count is not None
            and minimum_feasible_group_count > theoretical_minimum
            else None
        ),
        "scan_order_policy": "home_image_v_then_u_row_major",
        "constraints": {
            "max_aspect_ratio": max_aspect_ratio,
            "adjacency_distance_factor": float(adjacency_distance_factor),
            "max_normal_spread_deg": max_normal_spread_deg,
            "max_depth_span_mm": max_depth_span_mm,
            "max_xy_diameter_mm": max_xy_diameter_mm,
            "planner_enabled": planner is not None,
        },
    }

    diagnostics: list[dict[str, Any]] = []
    for group_number, (group, entry) in enumerate(
        zip(groups, selected_entries), start=1
    ):
        group_indices = list(entry["indices"])
        group_edges = [
            {
                "from_hole_id": _hole_id(records[left], left),
                "to_hole_id": _hole_id(records[right], right),
                "distance_mm": float(distances[left, right]),
            }
            for offset, left in enumerate(group_indices)
            for right in group_indices[offset + 1 :]
            if is_adjacent(left, right)
        ]
        metrics = _group_metrics(group)
        final_plan = entry.get("planner")
        final_plan_ok = bool(entry.get("planner_ok", True)) and bool(entry.get("geometry_ok", True))
        final_plan_reason = (
            entry.get("planner_reason")
            if not entry.get("geometry_ok", True) else None
        )
        if final_plan_reason is None and not entry.get("geometry_ok", True):
            final_plan_reason = entry.get("geometry_reason")
        diagnostics.append(
            {
                "group_index": group_number,
                "hole_ids": [_hole_id(records[index], index) for index in group_indices],
                "scan_order": [
                    {
                        "rank": scan_rank[index] + 1,
                        "hole_id": _hole_id(records[index], index),
                        "initial_center_px": _point2(records[index].get("initial_center_px")),
                    }
                    for index in group_indices
                ],
                "seed_hole_id": _hole_id(records[group_indices[0]], group_indices[0]),
                "connected": connected(group_indices),
                "adjacency_limit_mm": adjacency_limit,
                "median_nearest_neighbor_mm": median_nn,
                "adjacency_edges": group_edges,
                "rejection_reasons": dict(
                    rejection_by_seed.get(group_indices[0], {})
                ),
                "planner_ok": final_plan_ok,
                "planner_reason": final_plan_reason,
                "planner": final_plan,
                "selected_by_exact_cover": not fallback_used,
                "minimum_feasible_group_count": minimum_feasible_group_count,
                "theoretical_min_group_count": theoretical_minimum,
                "max_xy_diameter_mm": max_xy_diameter_mm,
                **metrics,
            }
        )

    return groups, diagnostics, search_metadata
