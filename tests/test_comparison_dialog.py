import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QDialog, QToolButton

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

    def test_dataset_alone_accepts_folder_image_or_xml(self):
        for source in (self.root, self.image, self.prediction):
            with self.subTest(source=source):
                dialog = ComparisonDialog(source)
                self.addCleanup(dialog.deleteLater)
                dialog.accept()
                self.assertEqual(dialog.result(), QDialog.Accepted)
                self.assertEqual(dialog.image_path, source.resolve())
                self.assertIsNone(dialog.gt_source)
                self.assertIsNone(dialog.prediction_source)

    def test_rejects_empty_undecodable_and_missing_dataset_paths(self):
        invalid = self.root / "not-an-image.png"
        invalid.touch()
        for text in ("", str(invalid), str(self.root / "missing")):
            with self.subTest(text=text), patch("comparison_dialog.QMessageBox.warning") as warning:
                self.dialog.image_edit.setText(text)
                self.dialog.accept()
                warning.assert_called_once()
                self.assertEqual(self.dialog.result(), QDialog.Rejected)

    def test_prefilled_roles_survive_same_dataset_and_clear_when_dataset_changes(self):
        dialog = ComparisonDialog(self.root, gt_source=self.gt, prediction_source=self.prediction,
                                  active_dataset=True, binding_image=self.image)
        self.addCleanup(dialog.deleteLater)
        dialog.image_edit.setText(str(self.root) + os.sep)
        dialog.image_edit.editingFinished.emit()
        self.assertEqual(dialog.gt_edit.text(), str(self.gt))
        self.assertEqual(dialog.prediction_edit.text(), str(self.prediction))
        self.assertIn(self.image.name, dialog.gt_edit.toolTip())
        dialog.image_edit.setText(str(self.image))
        dialog.image_edit.editingFinished.emit()
        self.assertFalse(dialog.gt_edit.text())
        self.assertFalse(dialog.prediction_edit.text())
        self.assertIn("唯一同名图片", dialog.gt_edit.toolTip())
        dialog.prediction_edit.setText(str(self.prediction))
        dialog.accept()
        self.assertIsNone(dialog.gt_source)
        self.assertEqual(dialog.prediction_source, self.prediction.resolve())

    def test_split_buttons_offer_native_folder_and_file_selection(self):
        buttons = self.dialog.findChildren(QToolButton)
        dataset = next(b for b in buttons if b.accessibleName() == "选择图片或文件夹来源")
        prediction = next(b for b in buttons if b.accessibleName() == "选择只读对照标签来源")
        self.dialog.gt_edit.setText(str(self.gt))
        with patch("comparison_dialog.QFileDialog.getExistingDirectory", return_value=str(self.root)):
            dataset.click()
        self.assertEqual(self.dialog.image_edit.text(), str(self.root))
        self.assertFalse(self.dialog.gt_edit.text())
        with patch("comparison_dialog.QFileDialog.getOpenFileName", return_value=(str(self.prediction), "")):
            prediction.menu().actions()[0].trigger()
        self.assertEqual(self.dialog.prediction_edit.text(), str(self.prediction))
        with patch("comparison_dialog.QFileDialog.getOpenFileName", return_value=("", "")):
            dataset.menu().actions()[0].trigger()
        self.assertEqual(self.dialog.prediction_edit.text(), str(self.prediction))
        self.dialog.accept()
        self.assertEqual(self.dialog.result(), QDialog.Accepted)
