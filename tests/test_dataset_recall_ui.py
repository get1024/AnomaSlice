"""Keep the dataset summary in step with the editable comparison window."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from app_window import MainWindow
from detection_metrics import match_counts


class DatasetRecallWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory())).resolve()
        self.gt = self.root / "GT"
        self.pred = self.root / "模型"
        images = self.root / "图片"
        for directory in (self.gt, self.pred, images):
            directory.mkdir()
        self.first, self.second = images / "图1.png", images / "图2.png"
        pixels = QImage(120, 100, QImage.Format_RGB32)
        pixels.fill(Qt.gray)
        for path in (self.first, self.second):
            self.assertTrue(pixels.save(str(path)))
        self.enterContext(patch.object(MainWindow, "_get_global_config_path",
                                       return_value=str(self.root / "settings.json")))
        self.question = self.enterContext(patch.object(QMessageBox, "question", return_value=QMessageBox.Discard))
        for method in ("warning", "information", "critical"):
            self.enterContext(patch.object(QMessageBox, method))
        self.window = MainWindow()
        self.window.show()
        self.window.activateWindow()
        self.app.processEvents()

    def tearDown(self):
        self.question.return_value = QMessageBox.Discard
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def write_box(self, folder, image, label, *, x=10, width=30, copies=1):
        path = folder / image.with_suffix(".xml").name
        node = (f"<Detection><x>{x}</x><y>20</y><width>{width}</width><height>40</height>"
                f"<message>{label}</message></Detection>")
        path.write_text(
            f"<DefectDetection><Count>{copies}</Count><Detections>{node * copies}</Detections></DefectDetection>",
            encoding="utf-8",
        )
        return path

    def open_dataset(self, prediction=None):
        self.window._open_source(self.first.parent, self.first,
                                 comparison_sources=(self.gt, prediction or self.pred))

    def wait_for_summary(self, *fragments):
        deadline = time.monotonic() + 5
        label = self.window.dataset_recall
        while time.monotonic() < deadline:
            if label.isVisible() and all(fragment in label.text() for fragment in fragments):
                return
            QTest.qWait(10)
        self.fail(f"Expected visible summary containing {fragments!r}, got {label.text()!r}")

    def assert_counts(self, matched, total, predictions):
        rate = f"{matched / total:.1%}" if total else "—"
        self.wait_for_summary(rate, f"正确匹配 {matched} / 人工标注 {total}",
                              f"漏检 {total - matched}", f"多检 {predictions - matched}")

    def add_matching_box(self):
        canvas = self.window.canvas
        canvas._pending_rect = QRectF(10, 20, 30, 40)
        canvas._open_label_editor()
        canvas._label_editor.setText("划痕")
        QTest.keyClick(canvas._label_editor, Qt.Key_Return)

    def test_unvisited_image_and_unsaved_edits_undo_redo_and_discard(self):
        gt = self.write_box(self.gt, self.first, "划痕")
        pred = self.write_box(self.pred, self.first, "划痕")
        self.write_box(self.gt, self.second, "污渍")
        originals = gt.read_bytes(), pred.read_bytes()
        self.open_dataset()
        self.assert_counts(1, 2, 1)
        self.assertEqual(self.window._current_image_path, str(self.first))
        canvas = self.window.canvas
        canvas.select_annotation(0)
        canvas.delete_selected_annotation()
        self.assert_counts(0, 1, 1)
        self.window._undo_current()
        self.assert_counts(1, 2, 1)
        self.window._redo_current()
        self.assert_counts(0, 1, 1)
        self.question.return_value = QMessageBox.Cancel
        self.window._on_next_image()
        self.assertEqual(self.window._current_image_path, str(self.first))
        self.assert_counts(0, 1, 1)
        self.question.return_value = QMessageBox.Discard
        self.window._on_next_image()
        self.assertEqual(self.window._current_image_path, str(self.second))
        self.assert_counts(1, 2, 1)
        self.assertEqual((gt.read_bytes(), pred.read_bytes()), originals)

    def test_empty_gt_then_save_and_reopen_without_double_counting(self):
        prediction = self.write_box(self.pred, self.first, "划痕")
        before = prediction.read_bytes()
        self.open_dataset()
        self.assert_counts(0, 0, 1)
        self.add_matching_box()
        self.assert_counts(1, 1, 1)
        target = self.gt / "图1.xml"
        self.assertFalse(target.exists())
        self.window._on_save_current()
        self.assertTrue(target.is_file())
        self.window._on_next_image()
        self.assert_counts(1, 1, 1)
        self.open_dataset()
        self.assert_counts(1, 1, 1)
        self.assertEqual(prediction.read_bytes(), before)

    def test_edit_while_initial_scan_is_in_flight_wins_over_disk_snapshot(self):
        self.write_box(self.gt, self.first, "划痕")
        self.write_box(self.pred, self.first, "划痕")
        self.write_box(self.gt, self.second, "污渍")
        snapshot_read, resume_scan = Event(), Event()

        def delayed_counts(gt, predictions, min_coverage):
            counts = match_counts(gt, predictions, min_coverage)
            if not snapshot_read.is_set():
                snapshot_read.set()
                resume_scan.wait(5)
            return counts

        with patch("dataset_metrics.match_counts", side_effect=delayed_counts):
            try:
                self.open_dataset()
                self.assertTrue(snapshot_read.wait(5))
                self.window.canvas.select_annotation(0)
                self.window.canvas.delete_selected_annotation()
            finally:
                resume_scan.set()
            self.assert_counts(0, 1, 1)

    def test_new_source_using_same_images_replaces_totals_and_plain_mode_clears_them(self):
        self.write_box(self.gt, self.first, "划痕")
        self.write_box(self.pred, self.first, "划痕")
        other = self.root / "另一模型"
        other.mkdir()
        self.write_box(other, self.first, "掉漆", x=80)
        self.open_dataset()
        self.assert_counts(1, 1, 1)
        self.open_dataset(other)
        self.assert_counts(0, 1, 1)
        self.window._open_source(self.first.parent, self.first)
        self.assertFalse(self.window.dataset_recall.isVisible())
        self.open_dataset()
        self.assert_counts(1, 1, 1)
        self.window._on_new_project()
        self.assertFalse(self.window.dataset_recall.isVisible())

    def test_switch_rule_recounts_unvisited_image_and_keeps_current_unsaved_gt(self):
        # A narrow prediction is fully inside the box the reviewer will add.
        self.write_box(self.pred, self.first, "掉漆", width=2)
        # Both duplicates overlap the GT by one pixel in width.
        self.write_box(self.gt, self.second, "污渍")
        self.write_box(self.pred, self.second, "其他", x=39, copies=2)
        self.open_dataset()
        self.assertEqual(self.window.recall_rule.currentIndex(), 0)
        self.assert_counts(1, 1, 3)
        self.add_matching_box()
        self.assert_counts(2, 2, 3)
        self.window.recall_rule.setCurrentIndex(1)
        self.assert_counts(1, 2, 3)
        self.assertEqual(self.window._current_image_path, str(self.first))
        self.assertTrue(self.window._boxes_dirty)
        self.assertFalse((self.gt / "图1.xml").exists())
        self.question.assert_not_called()
        self.window.recall_rule.setCurrentIndex(0)
        self.assert_counts(2, 2, 3)

    def test_unreadable_prediction_disables_dataset_rate_and_identifies_image(self):
        self.write_box(self.gt, self.first, "划痕")
        self.write_box(self.gt, self.second, "污渍")
        malformed = self.pred / "图2.xml"
        malformed.write_bytes(b"<broken")
        self.open_dataset()
        self.wait_for_summary("—", "1 张图片无法统计")
        self.assertIn(self.second.name, self.window.dataset_recall.toolTip())
        self.window._on_next_image()
        self.wait_for_summary("—", "1 张图片无法统计")
        self.assertTrue(self.window.canvas.isEnabled())
        self.assertEqual(malformed.read_bytes(), b"<broken")
