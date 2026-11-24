"""Exercise GT/prediction ownership through the window's open, edit and save paths."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGraphicsRectItem, QMessageBox

from app_window import MainWindow


class ComparisonWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        images = self.root / "图片"
        images.mkdir()
        self.first, self.second = images / "图1.png", images / "图2.png"
        image = QImage(160, 120, QImage.Format_RGB32)
        image.fill(Qt.gray)
        for path in (self.first, self.second):
            self.assertTrue(image.save(str(path)))
        self.near = images / "xml"
        self.gt = self.root / "GT"
        self.model = self.root / "预测"
        for folder in (self.near, self.gt, self.model):
            folder.mkdir()

        self.enterContext(patch.object(MainWindow, "_get_global_config_path",
                                       return_value=str(self.root / "settings.json")))
        self.question = self.enterContext(patch("app_window.QMessageBox.question", return_value=QMessageBox.Save))
        for method in ("warning", "information", "critical"):
            self.enterContext(patch(f"app_window.QMessageBox.{method}"))
        self.window = MainWindow()
        self.window.show()
        self.window.activateWindow()
        self.app.processEvents()

    def tearDown(self):
        self.question.return_value = QMessageBox.Discard
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def write_xml(self, folder, image, label, x=4):
        path = folder / image.with_suffix(".xml").name
        path.write_text(
            f"<DefectDetection><Count>1</Count><Detections><Detection index='1'>"
            f"<message>{label}</message><x>{x}</x><y>5</y><width>20</width><height>15</height>"
            "</Detection></Detections></DefectDetection>", encoding="utf-8",
        )
        return path

    def open_comparison(self, gt, prediction):
        with patch("app_window.ComparisonDialog") as factory:
            dialog = factory.return_value
            dialog.exec.side_effect = [True, False]
            dialog.image_path = self.first
            dialog.gt_source = gt
            dialog.prediction_source = prediction
            self.window._open_comparison()
        self.app.processEvents()

    def add_box(self, label):
        canvas = self.window.canvas
        canvas._pending_rect = QRectF(75, 45, 25, 15)
        canvas._open_label_editor()
        canvas._label_editor.setText(label)
        QTest.keyClick(canvas._label_editor, Qt.Key_Return)
        self.app.processEvents()
        self.assertTrue(self.window._boxes_dirty)

    def save_shortcut(self):
        QTest.keyClick(self.window.canvas, Qt.Key_S, Qt.ControlModifier)
        self.app.processEvents()
        self.assertFalse(self.window._boxes_dirty)

    def assert_layers(self, gt_labels, prediction_x):
        canvas = self.window.canvas
        self.assertTrue(canvas.isEnabled())
        self.assertEqual([box["label"] for box in canvas.get_annotations()], gt_labels)
        self.assertEqual(self.window.review_box_list.count(), len(gt_labels))
        boxes = [item for item in canvas._prediction_items if isinstance(item, QGraphicsRectItem)]
        self.assertEqual([item.rect().x() for item in boxes], prediction_x)
        self.assertTrue(all(item.isVisible() for item in boxes))

    @staticmethod
    def saved_labels(path):
        return [node.text for node in ET.parse(path).findall("./Detections/Detection/message")]

    def test_gt_near_image_prediction_separate_shortcut_and_navigation(self):
        gt_file = self.write_xml(self.near, self.first, "GT1")
        pred1 = self.write_xml(self.model, self.first, "P1", x=42)
        pred2 = self.write_xml(self.model, self.second, "P2", x=32)
        original = [pred1.read_bytes(), pred2.read_bytes()]
        self.open_comparison(None, self.model)
        self.assert_layers(["GT1"], [42])
        self.add_box("人工1")
        self.save_shortcut()
        self.assertEqual(self.saved_labels(gt_file), ["GT1", "人工1"])
        self.window._on_next_image()
        self.assert_layers([], [32])
        self.add_box("人工2")
        self.window._on_prev_image()
        self.question.assert_called_once()
        self.assertEqual(self.saved_labels(self.near / "图2.xml"), ["人工2"])
        self.assert_layers(["GT1", "人工1"], [42])
        self.assertEqual([pred1.read_bytes(), pred2.read_bytes()], original)

    def test_prediction_near_image_and_new_gt_in_separate_folder(self):
        pred = self.write_xml(self.near, self.first, "仅预测", x=55)
        before = pred.read_bytes()
        self.window._open_source(self.first.parent, self.first, comparison_sources=(self.gt, None))
        self.assert_layers([], [55])
        self.add_box("实测")
        self.save_shortcut()
        self.assertEqual(self.saved_labels(self.gt / "图1.xml"), ["实测"])
        self.assertEqual(pred.read_bytes(), before)

    def test_explicit_separate_files_and_return_to_plain_image(self):
        gt = self.write_xml(self.gt, self.first, "GT")
        pred = self.write_xml(self.model, self.first, "P", x=36)
        before = pred.read_bytes()
        self.open_comparison(gt, pred)
        self.assert_layers(["GT"], [36])
        self.add_box("补录")
        self.save_shortcut()
        self.assertEqual(self.saved_labels(gt), ["GT", "补录"])
        self.window._open_source(self.first.parent, self.first)
        self.assert_layers([], [])
        self.assertTrue(self.window.prediction_status.isHidden())
        self.add_box("本地")
        self.save_shortcut()
        self.assertEqual(self.saved_labels(self.near / "图1.xml"), ["本地"])
        self.assertEqual(self.saved_labels(gt), ["GT", "补录"])
        self.assertEqual(pred.read_bytes(), before)

    def test_missing_and_malformed_predictions_leave_gt_editable(self):
        pred = self.model / "图2.xml"
        pred.write_bytes(b"<broken")
        self.open_comparison(self.gt, self.model)
        self.assert_layers([], [])
        self.add_box("无预测")
        self.window._on_next_image()
        self.assert_layers([], [])
        self.add_box("坏预测")
        self.save_shortcut()
        self.assertEqual(self.saved_labels(self.gt / "图1.xml"), ["无预测"])
        self.assertEqual(self.saved_labels(self.gt / "图2.xml"), ["坏预测"])
        self.assertFalse((self.model / "图1.xml").exists())
        self.assertEqual(pred.read_bytes(), b"<broken")

    def test_reassigning_dirty_single_xml_as_prediction_offers_only_discard_or_cancel(self):
        pred = self.write_xml(self.near, self.first, "原始预测", x=46)
        before = pred.read_bytes()
        self.window._open_source(self.first.parent, pred)
        self.assert_layers(["原始预测"], [])
        self.add_box("误加")
        self.question.return_value = QMessageBox.Cancel
        self.open_comparison(self.gt, None)
        self.assertEqual(self.question.call_args.args[3], QMessageBox.Discard | QMessageBox.Cancel)
        self.assert_layers(["原始预测", "误加"], [])
        self.assertEqual(pred.read_bytes(), before)
        self.question.return_value = QMessageBox.Discard
        self.open_comparison(self.gt, None)
        self.assert_layers([], [46])
        self.add_box("GT 新框")
        self.save_shortcut()
        self.assertEqual(self.saved_labels(self.gt / "图1.xml"), ["GT 新框"])
        self.assertEqual(pred.read_bytes(), before)
