"""Workflow checks spanning the actual window, canvas, and disk persistence."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from annotation_store import LabelCatalog
from main import MainWindow, SlicerThread


class AppWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config_patch = patch.object(MainWindow, "_get_global_config_path", return_value=str(self.root / "settings.json"))
        self.config_patch.start()
        self.messages = [patch("main.QMessageBox.warning"), patch("main.QMessageBox.information"), patch("main.QMessageBox.critical")]
        for message in self.messages:
            message.start()
        self.window = MainWindow()
        self.window.show()
        self.window.activateWindow()
        self.app.processEvents()
        self.images = []
        for name in ("样本1.png", "样本2.png"):
            path = self.root / name
            cv2.imencode(".png", np.full((240, 320, 3), 100, np.uint8))[1].tofile(str(path))
            self.images.append(str(path))
        self.window._image_list = self.images
        self.window.project_manager.set_images(self.images)
        self.window._image_index = 0
        self.window._load_current_image()
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

    def read_sidecar(self, index=0):
        return json.loads(Path(self.images[index] + ".annotations.json").read_text(encoding="utf-8"))

    def test_enter_alias_autosave_undo_redo_navigation_and_reopen(self):
        self.window.label_catalog = LabelCatalog([{"name": "划痕", "aliases": ["1"]}])
        self.window._refresh_catalog()
        self.commit("1")
        self.assertEqual(self.read_sidecar()["boxes"], [{"label": "划痕", "bbox": [20, 30, 60, 70]}])
        self.assertFalse(self.window.canvas.has_pending_annotation())
        self.window._undo_current()
        self.assertEqual(self.read_sidecar()["boxes"], [])
        self.window._redo_current()
        self.assertEqual(len(self.read_sidecar()["boxes"]), 1)
        self.window._on_next_image()
        self.assertEqual(self.window.canvas.get_annotations(), [])
        self.commit("待定1")
        self.assertEqual(self.read_sidecar(1)["boxes"][0]["label"], "待定1")
        self.window._on_prev_image()
        self.assertEqual(self.window.canvas.get_annotations()[0]["label"], "划痕")

    def test_bbox_mode_is_lazy_and_slice_mode_retains_mask_workflow(self):
        self.assertEqual(self.window._patch_coords, [])
        self.assertIsNone(self.window.canvas.get_mask())
        self.assertFalse(self.window.slice_and_next_action.isEnabled())
        self.window.mode_combo.setCurrentIndex(1)
        self.app.processEvents()
        self.assertTrue(self.window._patch_coords)
        self.assertIsNotNone(self.window.canvas.get_mask())
        self.window._on_patch_defect()
        self.assertGreater(cv2.countNonZero(self.window.canvas.get_mask()), 0)
        self.window.mode_combo.setCurrentIndex(0)
        self.assertEqual(self.window.canvas.get_annotations(), [])

    def test_save_failure_retains_current_document_until_retry(self):
        with patch.object(self.window.annotation_store, "save", side_effect=OSError("read only")):
            self.commit("划痕")
            self.assertTrue(self.window._boxes_dirty)
            self.window._on_next_image()
            self.assertEqual(self.window._current_image_path, self.images[0])
            self.assertEqual(len(self.window.canvas.get_annotations()), 1)
        self.assertTrue(self.window._flush_annotations())
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
        before = Path(self.images[0] + ".annotations.json").read_bytes()
        project = self.root / "empty.json"
        project.write_text(json.dumps({"version": "1.1", "images": [], "current_index": -1, "config": {}}), encoding="utf-8")
        with patch("main.QFileDialog.getOpenFileName", return_value=(str(project), "")):
            self.window._on_open_project()
        self.assertIsNone(self.window._current_image_path)
        self.window._on_save_current()
        self.assertEqual(Path(self.images[0] + ".annotations.json").read_bytes(), before)

    def test_slice_export_still_classifies_using_mask_fraction(self):
        normal, abnormal = self.root / "Normal", self.root / "Abnormal"
        normal.mkdir()
        abnormal.mkdir()
        image = np.zeros((32, 64, 3), np.uint8)
        mask = np.zeros((32, 64), np.uint8)
        mask[:, :32] = 255
        worker = SlicerThread(image, mask, 32, 32, 32, .01, str(normal), str(abnormal), False, None, "sample")
        worker.run()
        self.assertEqual([p.name for p in normal.iterdir()], ["sample_32_0.png"])
        self.assertEqual([p.name for p in abnormal.iterdir()], ["sample_0_0.png"])


if __name__ == "__main__":
    unittest.main()
