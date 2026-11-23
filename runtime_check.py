"""Exercise frozen image codecs, metadata, project hashing, and crop export."""

import os
import tempfile
from pathlib import Path

from PySide6.QtGui import QColor, QImage, QPainter

from annotation_store import LabelCatalog
from app_window import CanvasWidget
from project_store import ProjectManager
from slicer import SlicerThread, count_mask_pixels
from xml_annotations import AnnotationXml, scan_folder


def verify_runtime(window):
    with tempfile.TemporaryDirectory(prefix="AnomaSlice-check-") as directory:
        image_path = os.path.join(directory, "测试图片.png")
        image = QImage(64, 32, QImage.Format_RGB32)
        image.fill(QColor(120, 140, 160))
        if not image.save(image_path):
            raise RuntimeError("PNG writer is unavailable")
        loaded = QImage(image_path)
        if loaded.isNull() or loaded.pixelColor(0, 0) != image.pixelColor(0, 0):
            raise RuntimeError("PNG reader did not preserve the image")
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
        window._open_source(Path(directory), entry.target())
        if window.canvas.get_annotations() != reopened.boxes:
            raise RuntimeError("The annotation page did not restore the XML boxes")
        window.canvas.select_annotation(0)
        window.canvas.delete_selected_annotation()
        window._on_save_current()
        if AnnotationXml.open(entry).boxes:
            raise RuntimeError("XML deletion was not saved")
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
