"""Export from the real window and check the PNGs written by its worker."""

import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from app_window import MainWindow
from export_dialog import ExportOptions
from xml_annotations import AnnotationXml


class ExportWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.images = self.root / "数据集"
        self.output = self.root / "导出"
        self.output.mkdir()
        self.enterContext(patch.object(MainWindow, "_get_global_config_path",
                                       return_value=str(self.root / "settings.json")))
        self.enterContext(patch.object(QMessageBox, "question", return_value=QMessageBox.Discard))
        self.enterContext(patch.object(QMessageBox, "information"))
        self.warning = self.enterContext(patch.object(QMessageBox, "warning"))
        self.critical = self.enterContext(patch.object(QMessageBox, "critical"))
        self.window = MainWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def image(self, name, background="black"):
        path = self.images / name
        path.parent.mkdir(exist_ok=True, parents=True)
        image = QImage(100, 80, QImage.Format_RGB32)
        image.fill(QColor(background))
        self.assertTrue(image.save(str(path)))
        return path

    def write_box(self, path, label, width=12):
        AnnotationXml.open_path(path).save([{"label": label, "bbox": [4, 26, width, 10]}])
        return path

    def export(self, mode, layers=("gt",), *, selected=frozenset(), label=""):
        previous = set(self.output.iterdir())
        with patch("app_window.ExportDialog") as factory:
            factory.return_value.exec.return_value = QDialog.Accepted
            factory.return_value.options = ExportOptions(mode, layers, self.output, selected, label)
            self.window.export_action.trigger()
            factory.assert_called_once()
        deadline = time.monotonic() + 10
        while self.window._export_thread is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertIsNone(self.window._export_thread, "Export worker did not finish")
        runs = set(self.output.iterdir()) - previous
        self.assertEqual(len(runs), 1)
        run, = runs
        self.assertTrue(run.is_dir() and run.name.startswith("AnomaSlice-"))
        return sorted(run.rglob("*.png"))

    def test_current_view_and_selected_crop_use_unsaved_gt_without_writing_sources(self):
        image = self.image("当前.png")
        gt = self.write_box(image.with_suffix(".xml"), "原框")
        original = (image.read_bytes(), gt.read_bytes())
        self.window._open_source(self.images)
        canvas = self.window.canvas
        canvas._pending_rect = QRectF(38, 36, 24, 22)
        canvas._open_label_editor()
        canvas._label_editor.setText("划痕")
        QTest.keyClick(canvas._label_editor, Qt.Key_Return)
        self.assertTrue(self.window._boxes_dirty)

        view, = self.export("current_image")
        annotated = QImage(str(view))
        self.assertEqual(annotated.size(), QSize(100, 80))
        self.assertEqual(annotated.pixelColor(38, 53), QColor("red"))
        crop, = self.export("selected_boxes", selected=frozenset({("gt", 1)}))
        self.assertEqual(QImage(str(crop)).size(), QSize(24, 22))
        self.assertTrue(self.window._boxes_dirty)
        self.assertEqual((image.read_bytes(), gt.read_bytes()), original)

    def test_all_images_preserve_duplicate_stems_and_previous_export(self):
        self.image("相机1/样本.png", "blue")
        self.image("相机2/样本.png", "green")
        self.window._open_source(self.images)
        first = self.export("all_images")
        self.assertEqual(len(first), 2)
        backgrounds = {QImage(str(path)).pixelColor(99, 79).name() for path in first}
        self.assertEqual(backgrounds, {QColor("blue").name(), QColor("green").name()})
        snapshot = {path: path.read_bytes() for path in first}
        second = self.export("all_images")
        self.assertEqual(len(second), 2)
        self.assertTrue(set(first).isdisjoint(second))
        self.assertEqual({path: path.read_bytes() for path in first}, snapshot)

    def test_crop_to_gt_box_and_draw_only_predictions_inside_it(self):
        image = self.image("对照.png")
        predictions = self.root / "预测"
        predictions.mkdir()
        AnnotationXml.open_path(image.with_suffix(".xml")).save([
            {"label": "划痕", "bbox": [5, 8, 60, 50]}])
        AnnotationXml.open_path(predictions / "对照.xml").save([
            {"label": "候选", "bbox": [15, 20, 30, 12]}])
        self.window._open_source(self.images, comparison_sources=(None, predictions))
        path, = self.export("selected_boxes", ("prediction",), selected=frozenset({("gt", 0)}))
        crop = QImage(str(path))
        self.assertEqual(crop.size(), QSize(60, 50))
        self.assertEqual(crop.pixelColor(0, 25), QColor("black"))
        predicted_edge = {crop.pixelColor(x, 12).name() for x in range(10, 40)}
        self.assertIn("#00d5e8", predicted_edge)

    def test_export_into_dataset_does_not_add_outputs_to_the_next_session(self):
        source = self.image("图.png")
        self.output = self.images
        self.window._open_source(self.images)
        exported, = self.export("current_image")
        self.window._open_source(self.images)
        self.assertEqual(self.window._image_list, [str(source)])
        self.window._open_source(exported.parent)
        self.assertEqual(self.window._image_list, [str(exported)])

    def test_all_and_class_crops_include_unvisited_images_and_allow_missing_predictions(self):
        first = self.image("图1.png")
        second = self.image("图2.png")
        predictions = self.root / "预测"
        predictions.mkdir()
        gt1 = self.write_box(first.with_suffix(".xml"), "划痕", 14)
        gt2 = self.write_box(second.with_suffix(".xml"), "旧标签", 16)
        pred1 = self.write_box(predictions / "图1.xml", "旧标签", 18)
        originals = {p: p.read_bytes() for p in (gt1, gt2, pred1)}
        self.window._open_source(self.images, comparison_sources=(None, predictions))
        both = ("gt", "prediction")
        all_crops = self.export("all_crops", both)
        self.assertCountEqual([QImage(str(p)).width() for p in all_crops], [14, 16, 18])
        selected = self.export("class_crops", both, label="旧标签")
        self.assertCountEqual([QImage(str(p)).width() for p in selected], [16, 18])
        self.assertEqual(self.window._current_image_path, str(first))
        self.assertEqual({p: p.read_bytes() for p in originals}, originals)
        self.assertFalse((predictions / "图2.xml").exists())

    def test_malformed_prediction_is_reported_only_for_exports_that_use_it(self):
        first = self.image("图1.png")
        second = self.image("图2.png")
        predictions = self.root / "预测"
        predictions.mkdir()
        self.write_box(first.with_suffix(".xml"), "划痕", 15)
        self.write_box(second.with_suffix(".xml"), "旧标签", 17)
        good = self.write_box(predictions / "图1.xml", "预测框", 21)
        broken = predictions / "图2.xml"
        broken.write_bytes(b"<broken")
        before = (good.read_bytes(), broken.read_bytes())
        self.window._open_source(self.images, comparison_sources=(None, predictions))
        self.warning.reset_mock()
        gt_crops = self.export("all_crops")
        self.assertCountEqual([QImage(str(p)).width() for p in gt_crops], [15, 17])
        self.warning.assert_not_called()
        pred, = self.export("all_crops", ("prediction",))
        self.assertEqual(QImage(str(pred)).width(), 21)
        self.warning.assert_called_once()
        self.assertEqual((good.read_bytes(), broken.read_bytes()), before)

    def test_unvisited_legacy_sidecar_and_project_boxes_export_without_xml_conversion(self):
        paths = [self.image(f"图{i}.png") for i in range(1, 4)]
        project = self.root / "旧工程.json"
        project.write_text(json.dumps({
            "version": "1.0", "images": list(map(str, paths)), "current_index": 0, "config": {},
            "annotations": {str(paths[2]): [{"label": "工程缺陷", "bbox": [1, 2, 11, 16]}]},
        }), encoding="utf-8")
        sidecar = Path(self.window.annotation_store.save(
            str(paths[1]), 100, 80, [{"label": "旧标注", "bbox": [2, 3, 18, 12]}]))
        before = (sidecar.read_bytes(), project.read_bytes())
        with patch("app_window.QFileDialog.getOpenFileName", return_value=(str(project), "")):
            self.window._on_open_project()
        self.critical.assert_not_called()
        crops = self.export("all_crops")
        self.assertCountEqual([QImage(str(p)).width() for p in crops], [18, 11])
        self.assertEqual(list(self.images.rglob("*.xml")), [])
        self.assertEqual((sidecar.read_bytes(), project.read_bytes()), before)
