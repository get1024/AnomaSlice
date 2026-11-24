"""Dataset metrics scan lifecycle, source matching and missing-layer behavior."""

from collections import Counter
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

from PySide6.QtCore import Qt

from comparison_paths import ComparisonSources
from dataset_metrics import DatasetMetricsScan


class DatasetMetricsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.near = self.root / "xml"
        self.far = self.root / "prediction"
        self.near.mkdir()
        self.far.mkdir()
        self.images = [self.root / f"sample{number}.png" for number in range(1, 5)]
        self.sources = ComparisonSources(self.images[0], None, self.far, images=self.images)

    @staticmethod
    def write_box(directory, image, x=5):
        path = directory / image.with_suffix(".xml").name
        path.write_text("<DefectDetection><Count>1</Count><Detections><Detection>"
                        f"<x>{x}</x><y>7</y><width>3</width><height>9</height><message>A</message>"
                        "</Detection></Detections></DefectDetection>", encoding="utf-8")
        return path

    def scan(self, min_coverage=0.0):
        snapshots = []
        worker = DatasetMetricsScan(self.images, self.sources, min_coverage=min_coverage)
        worker.result_ready.connect(snapshots.append, Qt.DirectConnection)
        worker.start()
        self.assertTrue(worker.wait(10000))
        self.assertEqual(len(snapshots), 1)
        return snapshots[0]

    def test_scan_counts_missing_layers_and_surfaces_invalid_xml(self):
        first, gt_only, prediction_only, broken = self.images
        self.write_box(self.near, first)
        self.write_box(self.far, first)
        self.write_box(self.near, gt_only)
        self.write_box(self.far, prediction_only)
        invalid = self.write_box(self.far, broken)
        invalid.write_bytes(b"not xml")

        result = self.scan()
        self.assertEqual(dict(result.counts), {first: (1, 1, 1), gt_only: (0, 1, 0), prediction_only: (0, 0, 1)})
        self.assertEqual(set(result.errors), {broken})
        self.assertIn("XML", result.errors[broken])
        self.assertFalse((self.near / prediction_only.with_suffix(".xml").name).exists())
        self.assertEqual(invalid.read_bytes(), b"not xml")

    def test_coverage_mode_changes_tp_but_preserves_gt_and_prediction_counts(self):
        image = self.images[0]
        self.write_box(self.near, image)
        self.write_box(self.far, image, x=7)
        self.assertEqual(self.scan(min_coverage=0.0).counts[image], (1, 1, 1))
        self.assertEqual(self.scan(min_coverage=0.5).counts[image], (0, 1, 1))

    def test_scan_indexes_directories_once_and_regular_lookup_stays_fresh(self):
        calls = Counter()
        iterdir = Path.iterdir

        def count_directory(path):
            calls[path] += 1
            return iterdir(path)

        with patch.object(Path, "iterdir", count_directory):
            result = self.scan()
        self.assertEqual(result.errors, {})
        self.assertEqual(calls, {self.near: 1, self.far: 1, self.root: 1})
        adjacent = self.write_box(self.root, self.images[0])
        self.assertEqual(self.sources.resolve(self.images[0]).gt, adjacent)
        self.assertEqual(self.scan().counts[self.images[0]], (0, 1, 0))

    def test_removed_explicit_gt_is_an_error_for_its_image(self):
        bound = self.write_box(self.near, self.images[0])
        self.sources = ComparisonSources(self.images[0], bound, self.far, images=self.images)
        bound.unlink()
        result = self.scan()
        self.assertEqual(set(result.errors), {self.images[0]})
        self.assertNotIn(self.images[0], result.counts)

    def test_cancelled_scan_finishes_without_publishing_a_partial_snapshot(self):
        computing, resume = Event(), Event()
        snapshots = []
        worker = DatasetMetricsScan(self.images, self.sources)
        worker.result_ready.connect(snapshots.append, Qt.DirectConnection)

        def pause_match(*args, **kwargs):
            computing.set()
            resume.wait(5)
            return 0, 0, 0

        with patch("dataset_metrics.match_counts", side_effect=pause_match) as match:
            worker.start()
            try:
                self.assertTrue(computing.wait(5))
                worker.requestInterruption()
            finally:
                resume.set()
                self.assertTrue(worker.wait(10000))
        self.assertEqual(match.call_count, 1)
        self.assertEqual(snapshots, [])
