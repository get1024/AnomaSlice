"""Workflow checks spanning the actual window, canvas, and disk persistence."""

import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from annotation_store import LabelCatalog
from app_window import MainWindow
from slicer import SlicerThread, count_mask_pixels


class AppWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.config_patch = patch.object(MainWindow, "_get_global_config_path", return_value=str(self.root / "settings.json"))
        self.config_patch.start()
        self.messages = [patch("app_window.QMessageBox.warning"), patch("app_window.QMessageBox.information"), patch("app_window.QMessageBox.critical"),
                         patch("app_window.QMessageBox.question", return_value=QMessageBox.Save)]
        for message in self.messages:
            message.start()
        self.window = MainWindow()
        self.window.show()
        self.window.activateWindow()
        self.app.processEvents()
        self.images = []
        for name in ("样本1.png", "样本2.png"):
            path = self.root / name
            image = QImage(320, 240, QImage.Format_RGB32)
            image.fill(QColor(100, 100, 100))
            self.assertTrue(image.save(str(path)))
            self.images.append(str(path))
        with patch("app_window.QFileDialog.getOpenFileName", return_value=(self.images[0], "")):
            self.window._open_image()
        self.app.processEvents()

    def tearDown(self):
        self.window._boxes_dirty = False
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        for message in self.messages:
            message.stop()
        self.config_patch.stop()
        self.temp.cleanup()

    def commit(self, label):
        canvas = self.window.canvas
        canvas._pending_rect = QRectF(20, 30, 60, 70)
        canvas._open_label_editor()
        canvas._label_editor.setText(label)
        QTest.keyClick(canvas._label_editor, Qt.Key_Return)
        self.app.processEvents()

    def xml_path(self, index=0):
        return self.root / "xml" / (Path(self.images[index]).stem + ".xml")

    def test_enter_alias_save_xml_undo_redo_navigation_and_reopen(self):
        self.window.label_catalog = LabelCatalog([{"name": "划痕", "aliases": ["1"]}])
        self.window._refresh_catalog()
        self.commit("1")
        self.assertFalse(self.xml_path().exists())
        QTest.keyClick(self.window.canvas, Qt.Key_S, Qt.ControlModifier)
        root = ET.parse(self.xml_path()).getroot()
        node = root.find("./Detections/Detection")
        self.assertEqual(node.findtext("message"), "划痕")
        self.assertEqual([node.findtext(k) for k in ("x","y","width","height")], ["20","30","60","70"])
        self.assertEqual((root.findtext("RawCount"), root.findtext("Count")), ("0", "1"))
        self.assertFalse(self.window.canvas.has_pending_annotation())
        self.window._undo_current()
        self.window._on_save_current()
        self.assertEqual(ET.parse(self.xml_path()).findtext("Count"), "0")
        self.window._redo_current()
        self.window._on_next_image()
        self.assertEqual(ET.parse(self.xml_path()).findtext("Count"), "1")
        self.assertEqual(self.window.canvas.get_annotations(), [])
        self.commit("待定1")
        self.window._on_prev_image()
        self.assertEqual(ET.parse(self.xml_path(1)).findtext("./Detections/Detection/message"), "待定1")
        self.assertEqual(self.window.canvas.get_annotations()[0]["label"], "划痕")
        self.assertFalse(list(self.root.rglob("*.annotations.json")))

    def test_bbox_mode_is_lazy_and_slice_mode_retains_mask_workflow(self):
        self.assertEqual(self.window._patch_coords, [])
        self.assertIsNone(self.window.canvas.get_mask())
        self.assertFalse(self.window.slice_and_next_action.isEnabled())
        self.window.mode_combo.setCurrentIndex(1)
        self.app.processEvents()
        self.assertTrue(self.window._patch_coords)
        self.assertIsNotNone(self.window.canvas.get_mask())
        self.window._on_patch_defect()
        self.assertGreater(count_mask_pixels(self.window.canvas.get_mask()), 0)
        self.window.mode_combo.setCurrentIndex(0)
        self.assertEqual(self.window.canvas.get_annotations(), [])

    def test_save_failure_retains_current_document_until_retry(self):
        with patch.object(self.window._xml_document, "save", side_effect=OSError("read only")):
            self.commit("划痕")
            self.assertTrue(self.window._boxes_dirty)
            self.window._on_next_image()
            self.assertEqual(self.window._current_image_path, self.images[0])
            self.assertEqual(len(self.window.canvas.get_annotations()), 1)
        self.assertTrue(self.window._save_xml())
        self.window._on_next_image()
        self.assertEqual(self.window._current_image_path, self.images[1])

    def test_corrupt_sidecar_is_never_overwritten(self):
        path = Path(self.images[1] + ".annotations.json")
        path.write_text("{broken", encoding="utf-8")
        self.window._on_next_image()
        self.assertTrue(self.window._annotation_load_error)
        self.assertFalse(self.window.canvas.isEnabled())
        self.window._on_save_current()
        self.window._on_prev_image()
        self.assertEqual(path.read_text(encoding="utf-8"), "{broken")
        self.assertTrue(self.window.canvas.isEnabled())

    def test_existing_text_is_not_converted_by_new_aliases(self):
        self.commit("1")
        self.window.label_catalog = LabelCatalog([{"name": "划痕", "aliases": ["1"]}])
        self.window._refresh_catalog()
        self.window._on_next_image()
        self.window._on_prev_image()
        self.assertFalse(self.window._annotation_load_error)
        self.assertEqual(self.window.canvas.get_annotations()[0]["label"], "1")

    def test_empty_project_cannot_overwrite_previous_image(self):
        self.commit("划痕")
        self.window._on_save_current()
        before = self.xml_path().read_bytes()
        project = self.root / "empty.json"
        project.write_text(json.dumps({"version": "1.1", "images": [], "current_index": -1, "config": {}}), encoding="utf-8")
        with patch("app_window.QFileDialog.getOpenFileName", return_value=(str(project), "")):
            self.window._on_open_project()
        self.assertIsNone(self.window._current_image_path)
        self.window._on_save_current()
        self.assertEqual(self.xml_path().read_bytes(), before)

    def test_folder_and_xml_picker_continue_the_same_editable_document(self):
        part = self.root / "部件"
        (part / "xml").mkdir(parents=True)
        image = part / "现场.bmp"
        self.assertTrue(QImage(self.images[0]).save(str(image)))
        path = part / "xml" / "现场.xml"
        original = """<DefectDetection><RawCount>7</RawCount><Count>2</Count><Detections>
        <Detection index="1"><x>20</x><y>30</y><width>60</width><height>70</height><message>污渍</message><confidence>0.412300</confidence><color>#0000FF</color></Detection>
        <Detection index="2" camera="B"><x>20</x><y>30</y><width>60</width><height>70</height><message>污渍</message><confidence>0.789000</confidence><color>#0000FF</color><extra>原始记录</extra></Detection>
        </Detections></DefectDetection>"""
        path.write_text(original, encoding="utf-8")
        with patch("app_window.QFileDialog.getExistingDirectory", return_value=str(self.root)):
            self.window._open_folder()
        self.window.review_image_picker.setCurrentIndex(self.window._image_list.index(str(image)))
        canvas = self.window.canvas
        self.assertEqual(canvas._annotation_items[0].rect(), QRectF(20, 30, 60, 70))
        self.assertEqual(canvas._annotation_items[0].pen().color(), QColor("#0000ff"))
        listing = self.window.review_box_list
        point = listing.visualItemRect(listing.item(0)).center()
        QTest.mouseClick(listing.viewport(), Qt.LeftButton, Qt.NoModifier, point)
        self.assertEqual(canvas.selected_annotation, 0)
        QTest.keyClick(listing, Qt.Key_Delete)
        self.assertEqual(len(canvas.get_annotations()), 1)
        self.assertEqual(path.read_text(encoding="utf-8"), original)
        point = canvas.mapFromScene(QPointF(40, 50))
        QTest.mouseDClick(canvas.viewport(), Qt.LeftButton, Qt.NoModifier, point)
        canvas._label_editor.setText("复核划痕")
        QTest.keyClick(canvas._label_editor, Qt.Key_Return)
        self.commit("人工补标")
        self.window.save_xml_button.click()
        root = ET.parse(path).getroot()
        self.assertEqual((root.findtext("RawCount"), root.findtext("Count")), ("7", "2"))
        first, second = root.findall("Detections/Detection")
        self.assertEqual((first.get("index"), first.get("camera"), first.findtext("extra")), ("1", "B", "原始记录"))
        self.assertEqual((first.findtext("message"), first.findtext("confidence")), ("复核划痕", "0.789000"))
        self.assertEqual((second.get("index"), second.findtext("message")), ("2", "人工补标"))
        with patch("app_window.QFileDialog.getOpenFileName", return_value=(str(path), "")):
            self.window._open_image()
        self.assertEqual(self.window._current_image_path, str(image))
        self.assertEqual([b["label"] for b in canvas.get_annotations()], ["复核划痕", "人工补标"])

    def test_cancel_or_discard_leaving_unsaved_image(self):
        self.commit("待审核")
        with patch("app_window.QMessageBox.question", return_value=QMessageBox.Cancel):
            self.window.review_image_picker.setCurrentIndex(1)
            self.window.mode_combo.setCurrentIndex(1)
            self.assertFalse(self.window.close())
        self.assertEqual((self.window.review_image_picker.currentIndex(), self.window.mode_combo.currentIndex()), (0, 0))
        self.assertTrue(self.window._boxes_dirty)
        with patch("app_window.QMessageBox.question", return_value=QMessageBox.Discard):
            self.window._on_next_image()
        self.assertEqual(self.window._current_image_path, self.images[1])
        self.assertFalse(self.xml_path().exists())

    def test_legacy_boxes_import_once_and_existing_xml_takes_precedence(self):
        boxes = [{"label": "旧标签", "bbox": [1, 2, 10, 12]}]
        legacy = Path(self.window.annotation_store.save(self.images[0], 320, 240, boxes, []))
        data = legacy.read_bytes()
        self.window._load_current_image()
        self.assertEqual(self.window.canvas.get_annotations(), boxes)
        self.window._on_save_current()
        self.window.canvas.select_annotation(0)
        self.window.canvas.delete_selected_annotation()
        self.window._on_save_current()
        self.window._load_current_image()
        self.assertEqual(self.window.canvas.get_annotations(), [])
        self.assertEqual(data, legacy.read_bytes())

    def test_slice_export_still_classifies_using_mask_fraction(self):
        normal, abnormal = self.root / "Normal", self.root / "Abnormal"
        normal.mkdir()
        abnormal.mkdir()
        image = QImage(64, 32, QImage.Format_RGB32)
        image.fill(QColor("black"))
        mask = QImage(64, 32, QImage.Format_Grayscale8)
        mask.fill(0)
        painter = QPainter(mask)
        painter.fillRect(0, 0, 32, 32, QColor("white"))
        painter.end()
        worker = SlicerThread(image, mask, 32, 32, 32, .01, str(normal), str(abnormal), False, None, "sample")
        worker.run()
        self.assertEqual([p.name for p in normal.iterdir()], ["sample_32_0.png"])
        self.assertEqual([p.name for p in abnormal.iterdir()], ["sample_0_0.png"])


if __name__ == "__main__":
    unittest.main()
