import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QDialog

from comparison_dialog import ComparisonDialog


class ComparisonDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.image = self.root / "sample.PNG"
        sample = QImage(4, 4, QImage.Format_RGB32)
        sample.fill(0)
        self.assertTrue(sample.save(str(self.image)))
        self.gt = self.root / "labels"
        self.gt.mkdir()
        self.prediction = self.root / "sample.XML"
        self.prediction.touch()
        self.dialog = ComparisonDialog(self.image)
        self.addCleanup(self.dialog.deleteLater)

    def test_separate_directory_and_file_sources(self):
        self.dialog.gt_edit.setText(str(self.gt))
        self.dialog.prediction_edit.setText(str(self.prediction))
        self.dialog.accept()
        self.assertEqual(self.dialog.result(), QDialog.Accepted)
        self.assertEqual(self.dialog.image_path, self.image.resolve())
        self.assertEqual(self.dialog.gt_source, self.gt.resolve())
        self.assertEqual(self.dialog.prediction_source, self.prediction.resolve())

    def test_gt_explicit_prediction_near_image(self):
        self.dialog.gt_edit.setText(str(self.gt))
        self.dialog.accept()
        self.assertEqual(self.dialog.result(), QDialog.Accepted)
        self.assertIsNone(self.dialog.prediction_source)

    def test_rejects_invalid_or_identical_sources(self):
        for gt, prediction in [
            ("", ""),
            (self.root / "missing.xml", ""),
            (self.image, self.prediction),
            (self.prediction, self.prediction),
        ]:
            with self.subTest(gt=gt, prediction=prediction), patch("comparison_dialog.QMessageBox.warning") as warning:
                self.dialog.gt_edit.setText(str(gt))
                self.dialog.prediction_edit.setText(str(prediction))
                self.dialog.accept()
                warning.assert_called_once()
                self.assertEqual(self.dialog.result(), QDialog.Rejected)
                self.assertIsNone(self.dialog.image_path)
