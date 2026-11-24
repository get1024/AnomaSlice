"""One-to-one GT/prediction matching by intersection over the smaller box area."""

from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf, isfinite, nextafter
from sys import float_info


def _axis_overlap(a, width, b, other_width):
    # Subtract the distance between starts before adding any absolute coordinates.
    if a <= b:
        return min(other_width, width - (b - a))
    return min(width, other_width - (a - b))


def _eligible_pairs(gt_boxes, prediction_boxes, min_coverage):
    """Sweep left to right, comparing only rectangles with overlapping x ranges."""
    rectangles = [[], []]
    events = []
    for side, boxes in enumerate((gt_boxes, prediction_boxes)):
        for index, box in enumerate(boxes):
            x, y, width, height = box["bbox"]
            rectangles[side].append((x, y, width, height))
            if width > 0 and height > 0:
                # Round outward for candidate discovery only. A tiny valid width
                # at a large coordinate may round x + width back to x.
                end = nextafter(x + width, inf)
                events.extend(((x, True, side, index), (end, False, side, index)))

    active = [set(), set()]
    for _, opening, side, index in sorted(events):  # At equal x, remove ends before adding starts.
        if not opening:
            active[side].remove(index)
            continue
        x, y, width, height = rectangles[side][index]
        for other in active[1 - side]:
            x2, y2, width2, height2 = rectangles[1 - side][other]
            overlap_y = _axis_overlap(y, height, y2, height2)
            overlap_x = _axis_overlap(x, width, x2, width2)
            if overlap_x <= 0 or overlap_y <= 0:
                continue
            intersection = overlap_x * overlap_y
            smaller_area = min(width * height, width2 * height2)
            if float_info.min <= intersection and isfinite(smaller_area):
                # One division preserves exact integer boundaries such as 66/132.
                coverage = intersection / smaller_area
            else:
                # For areas outside the normal float range, both axis ratios
                # are bounded and max(I/A, I/B) still equals I/min(A,B).
                coverage = max((overlap_x / width) * (overlap_y / height),
                               (overlap_x / width2) * (overlap_y / height2))
            if coverage >= min_coverage:
                gt, pred = (other, index) if side else (index, other)
                yield gt, pred, coverage
        active[side].add(index)


@dataclass(slots=True)
class _Edge:
    to: int
    reverse: int
    cost: int
    available: bool


def _add_edge(graph, start, end, cost):
    forward = _Edge(end, len(graph[end]), cost, True)
    reverse = _Edge(start, len(graph[start]), -cost, False)
    graph[start].append(forward)
    graph[end].append(reverse)
    return forward


def _augment(graph, source, sink, potential):
    """Add a cheapest unit of flow, allowing earlier pairs to be reassigned."""
    distance = [inf] * len(graph)
    previous = [None] * len(graph)
    distance[source] = 0
    queue = [(0, source)]
    while queue:
        cost, node = heappop(queue)
        if cost != distance[node]:
            continue
        for edge in graph[node]:
            if not edge.available:
                continue
            # Potentials keep reduced costs nonnegative even on reverse edges.
            candidate = cost + edge.cost + potential[node] - potential[edge.to]
            if candidate < distance[edge.to]:
                distance[edge.to] = candidate
                previous[edge.to] = (node, edge)
                heappush(queue, (candidate, edge.to))
    if previous[sink] is None:
        return False
    for node, cost in enumerate(distance):
        if cost != inf:
            potential[node] += cost
    node = sink
    while node != source:
        parent, edge = previous[node]
        edge.available = False
        graph[node][edge.reverse].available = True
        node = parent
    return True


def _component_pairs(edges):
    gt_ids = sorted({gt for gt, _, _ in edges})
    pred_ids = sorted({pred for _, pred, _ in edges})
    if len(gt_ids) == 1 or len(pred_ids) == 1:
        gt, pred, _ = max(edges, key=lambda edge: edge[2])
        return [(gt, pred)]

    # Unit capacities enforce one-to-one assignments. Maximum flow gives the
    # most matches; minimizing (1 - coverage) gives the highest coverage sum
    # at that cardinality. Scale binary floats exactly to avoid rounding cycles.
    scale = max(score.as_integer_ratio()[1] for _, _, score in edges)
    gt_node = {gt: index for index, gt in enumerate(gt_ids)}
    pred_node = {pred: len(gt_ids) + index for index, pred in enumerate(pred_ids)}
    source = len(gt_ids) + len(pred_ids)
    sink = source + 1
    graph = [[] for _ in range(sink + 1)]
    for node in gt_node.values():
        _add_edge(graph, source, node, 0)
    for node in pred_node.values():
        _add_edge(graph, node, sink, 0)
    assignments = []
    for gt, pred, score in edges:
        numerator, denominator = score.as_integer_ratio()
        cost = scale - numerator * (scale // denominator)
        edge = _add_edge(graph, gt_node[gt], pred_node[pred], cost)
        assignments.append((gt, pred, edge))

    potential = [0] * len(graph)
    while _augment(graph, source, sink, potential):
        pass
    return [(gt, pred) for gt, pred, edge in assignments if not edge.available]


def match_pairs(gt_boxes, prediction_boxes, min_coverage=0.0):
    """Return matched (GT index, prediction index) pairs, independent of class."""
    if not 0 <= min_coverage <= 1:
        raise ValueError("覆盖率阈值须在 0 和 1 之间。")
    by_gt, by_pred = defaultdict(list), defaultdict(list)
    for gt, pred, score in _eligible_pairs(gt_boxes, prediction_boxes, min_coverage):
        by_gt[gt].append((gt, pred, score))
        by_pred[pred].append(gt)

    # Disconnected groups cannot compete for a match. A typical isolated
    # defect needs only the single-edge fast path, even in a large image.
    pairs = []
    seen_gt, seen_pred = set(), set()
    for seed in by_gt:
        if seed in seen_gt:
            continue
        seen_gt.add(seed)
        pending = [seed]
        edges = []
        while pending:
            gt = pending.pop()
            edges.extend(by_gt[gt])
            for _, pred, _ in by_gt[gt]:
                if pred not in seen_pred:
                    seen_pred.add(pred)
                    for neighbor in by_pred[pred]:
                        if neighbor not in seen_gt:
                            seen_gt.add(neighbor)
                            pending.append(neighbor)
        pairs.extend(_component_pairs(edges))
    return sorted(pairs)


def match_counts(gt_boxes, prediction_boxes, min_coverage=0.0):
    """Return (true positives, GT count, prediction count)."""
    gt_boxes, prediction_boxes = tuple(gt_boxes), tuple(prediction_boxes)
    pairs = match_pairs(gt_boxes, prediction_boxes, min_coverage)
    return len(pairs), len(gt_boxes), len(prediction_boxes)
