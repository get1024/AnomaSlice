"""Exercise frozen image codecs, metadata, project hashing, and crop export."""

import os
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter
from PySide6.QtWidgets import QPushButton

from annotation_store import LabelCatalog
from app_window import CanvasWidget
from label_dialog import LabelDialog
from project_store import ProjectManager
from slicer import SlicerThread, count_mask_pixels
from visual_export import iter_visual_exports, save_png
from xml_annotations import AnnotationXml, scan_folder


def _check_visual_export(image, directory):
    gt = [{"label": "划痕", "bbox": [12, 30, 64, 56]}]
    pred = [{"label": "划痕", "bbox": [34, 48, 72, 40]}]
    image = image.scaled(128, 96)
    for crop, layers, name in [(False, ("gt", "prediction"), "双层整图"),
                               (True, ("prediction",), "预测缺陷")]:
        (layer, index, rendered), = iter_visual_exports(
            image, gt, pred, crops=crop, layers=layers, selected={("prediction", 0)})
        path = directory / f"{name}.png"
        save_png(rendered, path)
        decoded = QImage(str(path))
        width, height = (72, 40) if crop else (128, 96)
        if (decoded.width(), decoded.height()) != (width, height):
            raise RuntimeError(f"Frozen visual export size error: {name}")
        if crop and (layer, index) != ("prediction", 0):
            raise RuntimeError("Frozen crop selection failed")
        colors = {decoded.pixel(x, y) for x in range(width) for y in range(height)}
        box_colors = {"#00d5e8"} if crop else {"#00d5e8", "#ff0000"}
        if not all(QColor(color).rgb() in colors for color in box_colors):
            raise RuntimeError(f"Missing frozen export box: {name}")
        if not any(min(QColor.fromRgb(pixel).getRgb()[:3]) > 220 for pixel in colors):
            raise RuntimeError(f"Missing frozen export caption: {name}")


def _check_label_dialog(window):
    dialog = LabelDialog(LabelCatalog.defaults(), window)
    try:
        dialog.show()
        QCoreApplication.processEvents()
        buttons = {button.text(): button for button in dialog.findChildren(QPushButton)}
        buttons["添加种类"].click()
        _, name, aliases = dialog.rows[-1]
        for field, text in ((name, "凹坑"), (aliases, "dent")):
            point = field.rect().center()
            local = QPointF(point)
            screen = QPointF(field.mapToGlobal(point))
            for event_type, pressed in ((QEvent.MouseButtonPress, Qt.LeftButton),
                                        (QEvent.MouseButtonRelease, Qt.NoButton)):
                event = QMouseEvent(event_type, local, screen, Qt.LeftButton, pressed, Qt.NoModifier)
                QCoreApplication.sendEvent(field, event)
            field.insert(text)
        dialog.rows[0][0].click()
        buttons["移除所选种类"].click()
        buttons["保存"].click()
        if (dialog.result() != LabelDialog.Accepted or
                len(dialog.catalog.to_list()) != 4 or dialog.catalog.resolve("dent") != "凹坑"):
            raise RuntimeError("Frozen label editing failed")
    finally:
        dialog.close()
        dialog.deleteLater()


def verify_runtime(window):
    with tempfile.TemporaryDirectory(prefix="AnomaSlice-check-") as temp_path:
        directory = Path(temp_path).resolve()
        image_path = os.path.join(directory, "测试图片.png")
        image = QImage(64, 32, QImage.Format_RGB32)
        image.fill(QColor(120, 140, 160))
        if not image.save(image_path):
            raise RuntimeError("PNG writer is unavailable")
        loaded = QImage(image_path)
        if loaded.isNull() or loaded.pixelColor(0, 0) != image.pixelColor(0, 0):
            raise RuntimeError("PNG reader did not preserve the image")
        _check_label_dialog(window)
        mask = QImage(64, 32, QImage.Format_Grayscale8)
        mask.fill(0)
        painter = QPainter(mask)
        painter.fillRect(0, 0, 32, 32, QColor("white"))
        painter.end()
        catalog = LabelCatalog([{"name": "划痕", "aliases": ["1"]}])
        if catalog.resolve("1") != "划痕" or catalog.resolve("待定1") != "待定1":
            raise RuntimeError("Alias resolution failed")
        boxes = [{"label": "划痕", "bbox": [4, 5, 10, 12]}]
        for suffix in ("bmp", "jpg", "tiff", "webp"):
            target = os.path.join(directory, "codec-" + suffix + "." + suffix)
            if not image.save(target) or QImage(target).size() != image.size():
                raise RuntimeError(f"Frozen {suffix} codec failed")
        entry = next(entry for entry in scan_folder(directory) if entry.image == Path(image_path))
        AnnotationXml.open(entry).save(boxes)
        reopened = AnnotationXml.open(entry)
        if reopened.boxes[0]["label"] != "划痕" or reopened.boxes[0]["bbox"] != boxes[0]["bbox"]:
            raise RuntimeError("Annotation XML roundtrip failed")
        window._open_source(directory, entry.target())
        if window.canvas.get_annotations() != reopened.boxes:
            raise RuntimeError("The annotation page did not restore the XML boxes")
        window.canvas.select_annotation(0)
        window.canvas.delete_selected_annotation()
        window._on_save_current()
        if AnnotationXml.open(entry).boxes:
            raise RuntimeError("XML deletion was not saved")
        predictions = directory / "predictions"
        predictions.mkdir()
        model_path = predictions / entry.target().name
        AnnotationXml.open_path(model_path).save(boxes)
        model_bytes = model_path.read_bytes()
        window._open_source(directory, entry.image, comparison_sources=(None, predictions))
        if not window.canvas._prediction_items or window.canvas.get_annotations():
            raise RuntimeError("Frozen application did not separate GT and prediction boxes")
        window.canvas.set_annotations(boxes)
        window.canvas.annotations_changed(boxes)
        deadline = time.monotonic() + 5
        while "100.0%" not in window.dataset_recall.text():
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Frozen dataset recall failed: {window.dataset_recall.text()}")
            QCoreApplication.processEvents()
            time.sleep(0.01)
        window._on_save_current()
        if not AnnotationXml.open(entry).boxes or model_path.read_bytes() != model_bytes:
            raise RuntimeError("Comparison edits did not save exclusively to GT")
        canvas = CanvasWidget()
        try:
            if not canvas.load_image(image_path, image=loaded):
                raise RuntimeError("Image canvas load failed")
            canvas.set_annotation_mode(True)
            canvas.set_annotations(boxes)
            if canvas.get_annotations() != boxes:
                raise RuntimeError("Canvas annotation restore failed")
            canvas.set_mask(mask)
            if count_mask_pixels(canvas.get_mask()) != 1024:
                raise RuntimeError("Canvas mask restore failed")
        finally:
            canvas.close()
        project = ProjectManager()
        project.set_images([image_path])
        project.set_mask(image_path, mask)
        project.annotations[image_path] = boxes
        project_path = os.path.join(directory, "工程.json")
        project.save_project(project_path, 0, {"label_catalog": catalog.to_list()})
        restored = ProjectManager()
        restored.load_project(project_path)
        if count_mask_pixels(restored.get_mask(image_path)) != 1024:
            raise RuntimeError("Project mask PNG roundtrip failed")
        normal, abnormal = os.path.join(directory, "Normal"), os.path.join(directory, "Abnormal")
        os.makedirs(normal)
        os.makedirs(abnormal)
        worker = SlicerThread(image, mask, 32, 32, 32, .01, normal, abnormal, False, None, "测试")
        errors, completed = [], []
        worker.error_signal.connect(errors.append)
        worker.finished_signal.connect(lambda n, a: completed.append((n, a)))
        worker.run()
        if errors or len(completed) != 1 or any(len(paths) != 1 for paths in completed[0]):
            raise RuntimeError(f"Crop export failed: {errors}")
        for path in completed[0][0] + completed[0][1]:
            if QImage(path).size() != QImage(32, 32, QImage.Format_RGB32).size():
                raise RuntimeError("Exported crop dimensions are wrong")
        _check_visual_export(loaded, directory)
