"""Project round-trip tests for GT and prediction roles."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QMessageBox

from app_window import MainWindow


class ComparisonProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_reopening_project_preserves_nearby_prediction_as_read_only(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.enterContext(patch.object(MainWindow, "_get_global_config_path", return_value=str(root / "settings.json")))
        self.enterContext(patch.object(QMessageBox, "information"))
        self.enterContext(patch.object(QMessageBox, "warning"))
        self.enterContext(patch.object(QMessageBox, "critical"))
        window = MainWindow()
        self.addCleanup(window.close)

        image = root / "images" / "sample.png"
        prediction = image.parent / "xml" / "sample.xml"
        gt = root / "ground-truth" / "sample.xml"
        gt.parent.mkdir()
        prediction.parent.mkdir(parents=True)
        pixels = QImage(80, 80, QImage.Format_RGB32)
        pixels.fill(0)
        self.assertTrue(pixels.save(str(image)))
        xml = ("<DefectDetection><Count>1</Count><Detections><Detection>"
               "<x>2</x><y>3</y><width>5</width><height>7</height><message>{}</message>"
               "</Detection></Detections></DefectDetection>")
        prediction_bytes = xml.format("model").encode()
        prediction.write_bytes(prediction_bytes)
        gt.write_bytes(xml.format("human").encode())

        window._open_source(image.parent, image, comparison_sources=(gt.parent, None))
        project = str(root / "review.json")
        window._save_project_impl(project)
        window._on_new_project()
        with patch("app_window.QFileDialog.getOpenFileName", return_value=(project, "")):
            window._on_open_project()

        self.assertEqual(window._xml_document.path, gt)
        self.assertEqual(window.canvas.get_annotations()[0]["label"], "human")
        self.assertTrue(window.canvas._prediction_items)
        window.canvas.select_annotation(0)
        window.canvas.delete_selected_annotation()
        window._on_save_current()
        self.assertEqual(ET.parse(gt).findtext("Count"), "0")
        self.assertEqual(prediction.read_bytes(), prediction_bytes)
