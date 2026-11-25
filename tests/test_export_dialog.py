import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog

from export_dialog import ExportDialog, ExportOptions


class ExportDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.gt = [{"label": "旧标签", "bbox": [2, 4, 8, 16]}]
        self.prediction = [{"label": "预测类别", "bbox": [1, 3, 5, 7]}]

    def dialog(self, **kwargs):
        dialog = ExportDialog(gt_boxes=self.gt, **kwargs)
        self.addCleanup(dialog.deleteLater)
        dialog.folder.setText(str(self.root))
        return dialog

    def test_image_and_all_crop_exports_use_chosen_layers(self):
        for mode in ("current_image", "all_images", "all_crops"):
            with self.subTest(mode=mode):
                dialog = self.dialog(prediction_boxes=[])
                dialog.mode.setCurrentIndex(dialog.mode.findData(mode))
                dialog.accept()
                self.assertEqual(dialog.result(), QDialog.Accepted)
                self.assertEqual(dialog.options, ExportOptions(mode, ("gt", "prediction"), self.root))
        single = self.dialog()
        self.assertEqual(single.layers.count(), 1)
        single.accept()
        self.assertEqual(single.options.layers, ("gt",))

    def test_crop_locations_stay_available_when_painted_layers_change(self):
        dialog = self.dialog(prediction_boxes=self.prediction, selected_gt=0)
        dialog.mode.setCurrentIndex(dialog.mode.findData("specified"))
        self.assertEqual(dialog.boxes.item(0).checkState(), Qt.Checked)
        dialog.layers.setCurrentIndex(1)
        self.assertFalse(dialog.boxes.item(0).isHidden())
        dialog.accept()
        self.assertEqual(dialog.options, ExportOptions(
            "selected_boxes", ("prediction",), self.root, frozenset({("gt", 0)})))
        both = self.dialog(prediction_boxes=self.prediction, selected_gt=0)
        both.mode.setCurrentIndex(both.mode.findData("specified"))
        both.layers.setCurrentIndex(0)
        self.assertFalse(both.boxes.item(1).isHidden())
        both.boxes.item(1).setCheckState(Qt.Checked)
        both.accept()
        self.assertEqual(both.options.selected, frozenset({("gt", 0), ("prediction", 0)}))

    def test_class_filter_accepts_catalog_raw_and_manually_entered_names(self):
        dialog = self.dialog(prediction_boxes=self.prediction, labels=["划痕"])
        dialog.mode.setCurrentIndex(dialog.mode.findData("specified"))
        dialog.scope.setCurrentIndex(dialog.scope.findData("class_crops"))
        self.assertEqual([dialog.label.itemText(i) for i in range(dialog.label.count())],
                         ["划痕", "旧标签", "预测类别"])
        dialog.label.setEditText("  其他图片上的类别  ")
        dialog.accept()
        self.assertEqual(dialog.options, ExportOptions(
            "class_crops", ("gt", "prediction"), self.root, label="其他图片上的类别"))

    def test_cancelled_chooser_and_incomplete_options(self):
        dialog = self.dialog()
        with patch("export_dialog.QFileDialog.getExistingDirectory", return_value=""):
            dialog._browse()
        self.assertEqual(dialog.folder.text(), str(self.root))
        with patch("export_dialog.QMessageBox.warning") as warning:
            dialog.mode.setCurrentIndex(dialog.mode.findData("specified"))
            dialog.accept()
            dialog.scope.setCurrentIndex(dialog.scope.findData("class_crops"))
            dialog.label.setEditText("")
            dialog.accept()
            dialog.mode.setCurrentIndex(dialog.mode.findData("current_image"))
            dialog.folder.setText(str(self.root / "missing"))
            dialog.accept()
            self.assertEqual(warning.call_count, 3)
        self.assertEqual(dialog.result(), QDialog.Rejected)
        self.assertIsNone(dialog.options)
