import copy
from itertools import permutations
import unittest

from detection_metrics import _component_pairs, match_counts, match_pairs


def box(x=0, y=0, width=4, height=4, label="划痕"):
    return {"label": label, "bbox": [x, y, width, height]}


class DetectionMetricTests(unittest.TestCase):
    def test_empty_and_repeated_detections(self):
        self.assertEqual(match_counts([], []), (0, 0, 0))
        self.assertEqual(match_counts([], [box()]), (0, 0, 1))
        self.assertEqual(match_counts([box()], []), (0, 1, 0))
        self.assertEqual(match_counts([box(), box()], [box(), box(), box()]), (2, 2, 3))

    def test_containment_ignores_class_and_uses_smaller_box_area(self):
        big = box(-5, -5, 40, 40, label="未知类别")
        small = box(0.25, 0.5)
        for gt, pred in ((big, small), (small, big)):
            self.assertEqual(match_pairs([gt], [pred], min_coverage=1), [(0, 0)])

    def test_positive_intersection_and_inclusive_strict_threshold(self):
        gt = [box(0.25, 0.5)]
        tiny_overlap = [box(4.24, 0.5)]
        self.assertEqual(match_counts(gt, tiny_overlap), (1, 1, 1))
        self.assertEqual(match_counts(gt, tiny_overlap, 0.5), (0, 1, 1))
        self.assertEqual(match_counts(gt, [box(2.25, 0.5)], 0.5), (1, 1, 1))
        self.assertEqual(match_counts(gt, [box(2.26, 0.5)], 0.5), (0, 1, 1))
        # Intersection 6*11, smaller area 11*12. Multiplying two separately
        # rounded axis ratios can put this exact half just below the threshold.
        self.assertEqual(match_pairs([box(0, 0, 11, 12)], [box(5, 1, 11, 12)], 0.5), [(0, 0)])
        for pred in (box(4.25, 0.5), box(0.25, 4.5), box(4.25, 4.5), box(40, 40)):
            self.assertEqual(match_pairs(gt, [pred]), [])

    def test_finite_boxes_whose_area_underflows_or_overflows(self):
        for size in (1e-300, 1e300):
            with self.subTest(size=size):
                gt = [box(0.0, 0.0, size, size)]
                self.assertEqual(match_pairs(gt, [box(size / 2, 0.0, size, size)], 0.5), [(0, 0)])
                self.assertEqual(match_pairs(gt, [box(size, 0.0, size, size)]), [])
        tiny = [box(1e300, 1e300, 1e-300, 1e-300)]
        self.assertEqual(match_pairs(tiny, tiny, 1), [(0, 0)])

    def test_count_precedes_score_even_when_one_perfect_match_is_available(self):
        gt = [box(0, width=10), box(-9, width=10)]
        pred = [box(0, width=10), box(9, width=10)]
        self.assertEqual(match_pairs(gt, pred), [(0, 1), (1, 0)])
        self.assertEqual(match_pairs(gt, pred, 0.5), [(0, 0)])

    def test_best_total_coverage_can_skip_highest_individual_score(self):
        gt = [box(0, width=5), box(1, width=3)]
        pred = [box(2, width=4), box(3, width=3)]
        # The diagonal's 3/4 + 1/3 loses to the cross-pairing's 2/3 + 2/3.
        originals = copy.deepcopy((gt, pred))
        self.assertEqual(match_pairs(gt, pred), [(0, 1), (1, 0)])
        self.assertEqual(match_pairs(gt, reversed(pred)), [(0, 0), (1, 1)])
        self.assertEqual((gt, pred), originals)

    def test_disconnected_components_preserve_global_indices(self):
        gt = [box(x=0), box(x=10), box(x=0), box(y=20)]
        pred = [box(y=-20), box(x=11), box(x=1.5), box(x=-1.5)]
        pairs = match_pairs(gt, pred)
        self.assertEqual(len(pairs), 3)
        self.assertIn((1, 1), pairs)
        self.assertEqual(sorted(gt_index for gt_index, _ in pairs), [0, 1, 2])
        self.assertEqual(sorted(pred_index for _, pred_index in pairs), [1, 2, 3])

    def test_all_three_by_three_graphs_against_exhaustive_assignments(self):
        indices = [(i, j) for i in range(3) for j in range(3)]
        for mask in range(1, 512):
            # Vary the scores as well as edge presence from graph to graph.
            weights = {pair: ((mask + k * 3) % 8 + 1) / 8
                       for k, pair in enumerate(indices) if mask & (1 << k)}
            actual = _component_pairs([(i, j, w) for (i, j), w in weights.items()])
            self.assertEqual(len({i for i, _ in actual}), len(actual))
            self.assertEqual(len({j for _, j in actual}), len(actual))
            # Every partial matching can be extended to a permutation.
            scores = []
            for permutation in permutations(range(3)):
                valid = [weights[pair] for pair in enumerate(permutation) if pair in weights]
                scores.append((len(valid), sum(valid)))
            self.assertEqual((len(actual), sum(weights[p] for p in actual)), max(scores), mask)

    def test_threshold_must_be_a_proportion(self):
        for threshold in [-0.1, 1.1, float("nan")]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                match_counts([], [], threshold)


if __name__ == "__main__":
    unittest.main()
