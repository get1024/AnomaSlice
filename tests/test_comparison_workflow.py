"""Exercise GT/prediction ownership through the window's open, edit and save paths."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGraphicsRectItem, QMessageBox, QPushButton

from app_window import MainWindow
from comparison_dialog import ComparisonDialog


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

    def open_source(self, source, *, gt=None, prediction=None):
        with patch("app_window.ComparisonDialog") as factory:
            dialog = factory.return_value
            dialog.exec.side_effect = [True, False]
            dialog.image_path = source
            dialog.gt_source = gt
            dialog.prediction_source = prediction
            self.window._open_dataset()
        self.app.processEvents()

    def nested_image(self):
        image = self.first.parent / "相机A" / "图3.png"
        image.parent.mkdir()
        image.write_bytes(self.first.read_bytes())
        return image

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
        self.open_source(self.first, prediction=self.model)
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
        self.open_source(self.first, gt=self.gt)
        self.assert_layers([], [55])
        self.add_box("实测")
        self.save_shortcut()
        self.assertEqual(self.saved_labels(self.gt / "图1.xml"), ["实测"])
        self.assertEqual(pred.read_bytes(), before)

    def test_explicit_separate_files_and_return_to_plain_image(self):
        gt = self.write_xml(self.gt, self.first, "GT")
        pred = self.write_xml(self.model, self.first, "P", x=36)
        before = pred.read_bytes()
        self.open_source(self.first, gt=gt, prediction=pred)
        self.assert_layers(["GT"], [36])
        self.add_box("补录")
        self.save_shortcut()
        self.assertEqual(self.saved_labels(gt), ["GT", "补录"])
        self.open_source(self.first)
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
        self.open_source(self.first, gt=self.gt, prediction=self.model)
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
        self.open_source(pred)
        self.assert_layers(["原始预测"], [])
        self.add_box("误加")
        self.question.return_value = QMessageBox.Cancel
        self.open_source(self.first, gt=self.gt)
        self.assertEqual(self.question.call_args.args[3], QMessageBox.Discard | QMessageBox.Cancel)
        self.assert_layers(["原始预测", "误加"], [])
        self.assertEqual(pred.read_bytes(), before)
        self.question.return_value = QMessageBox.Discard
        self.open_source(self.first, gt=self.gt)
        self.assert_layers([], [46])
        self.add_box("GT 新框")
        self.save_shortcut()
        self.assertEqual(self.saved_labels(self.gt / "图1.xml"), ["GT 新框"])
        self.assertEqual(pred.read_bytes(), before)

    def test_folder_with_prediction_only_loads_nested_images_and_their_local_gt(self):
        nested = self.nested_image()
        near = nested.parent / "xml"
        near.mkdir()
        gt = self.write_xml(near, nested, "子目录 GT")
        pred = self.write_xml(self.model, nested, "子目录预测", x=60)
        before = pred.read_bytes()
        self.open_source(self.first.parent, prediction=self.model)
        self.assertEqual(set(self.window._image_list), {str(self.first), str(self.second), str(nested)})
        self.window.review_image_picker.setCurrentIndex(self.window._image_list.index(str(nested)))
        self.assert_layers(["子目录 GT"], [60])
        self.add_box("新增")
        self.save_shortcut()
        self.assertEqual(self.saved_labels(gt), ["子目录 GT", "新增"])
        self.assertEqual(pred.read_bytes(), before)

    def test_attach_and_change_predictions_on_existing_folder_keep_project_and_unsaved_gt(self):
        nested = self.nested_image()
        near = nested.parent / "xml"
        near.mkdir()
        gt = self.write_xml(near, nested, "原始")
        original_gt = gt.read_bytes()
        self.write_xml(self.model, nested, "A", x=51)
        other_model = self.root / "第二版预测"
        other_model.mkdir()
        self.write_xml(other_model, nested, "B", x=61)
        self.open_source(self.first.parent)
        self.assertEqual(set(self.window._image_list), {str(self.first), str(self.second), str(nested)})
        original_list = self.window._image_list.copy()
        index = original_list.index(str(nested))
        self.window.review_image_picker.setCurrentIndex(index)
        self.assert_layers(["原始"], [])
        project = str(self.root / "审核工程.json")
        self.window._save_project_impl(project)
        self.add_box("待保存")

        for model, x in [(self.model, 51), (other_model, 61)]:
            with self.subTest(model=model):
                self.open_source(self.first.parent, prediction=model)
                self.assertEqual(self.window._image_list, original_list)
                self.assertEqual(self.window._current_image_path, str(nested))
                self.assertEqual(self.window.review_image_picker.currentIndex(), index)
                self.assertEqual(self.window.project_manager.project_path, project)
                self.assert_layers(["原始", "待保存"], [x])
                self.assertTrue(self.window._boxes_dirty)
                self.assertEqual(gt.read_bytes(), original_gt)
                self.question.assert_not_called()
        self.window.canvas.annotation_undo()
        self.assert_layers(["原始"], [61])
        self.window.canvas.annotation_redo()
        self.window.save_xml_button.click()
        self.assertEqual(self.saved_labels(gt), ["原始", "待保存"])

    def test_open_gt_xml_directly_selects_its_image_in_plain_mode(self):
        gt = self.write_xml(self.near, self.second, "直接打开")
        self.open_source(gt)
        self.assertEqual(self.window._current_image_path, str(self.second))
        self.assert_layers(["直接打开"], [])
        self.assertTrue(self.window.prediction_status.isHidden())
        self.add_box("复核")
        self.save_shortcut()
        self.assertEqual(self.saved_labels(gt), ["直接打开", "复核"])

    def test_sidebar_opens_one_form_for_folder_and_prediction(self):
        self.write_xml(self.near, self.first, "人工")
        self.write_xml(self.model, self.first, "预测", x=46)
        buttons = [b for b in self.window.annotation_panel.findChildren(QPushButton)
                   if "打开" in b.text()]
        self.assertEqual(len(buttons), 1)

        def fill_form():
            dialog = self.app.activeModalWidget()
            if isinstance(dialog, ComparisonDialog):
                dialog.image_edit.setText(str(self.first.parent))
                dialog.prediction_edit.setText(str(self.model))
                dialog.accept()

        QTimer.singleShot(0, fill_form)
        buttons[0].click()
        self.assert_layers(["人工"], [46])

    def test_folder_and_single_named_xml_start_on_matching_image(self):
        pred = self.write_xml(self.model, self.second, "预测", x=50)
        self.open_source(self.first.parent, prediction=pred)
        self.assertEqual(self.window._current_image_path, str(self.second))
        self.assert_layers([], [50])
