"""Export a snapshot of the open dataset without navigating or saving annotations."""

import re
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from visual_export import iter_visual_exports, save_png
from xml_annotations import AnnotationXml, ImageEntry


def _filename_part(text):
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    return cleaned.encode("utf-8")[:90].decode("utf-8", errors="ignore").rstrip(". ")


class ExportJob(QThread):
    progress = Signal(int)

    def __init__(self, options, paths, index, *, current_image, current_gt, current_prediction,
                 comparison, entries, project_boxes, parent=None):
        super().__init__(parent)
        self.options = options
        self.current_path = paths[index]
        if options.mode in ("current_image", "selected_boxes"):
            self.images = [(index + 1, self.current_path)]
        else:
            self.images = list(enumerate(paths, start=1))
        self.current_image = current_image
        self.current_boxes = {"gt": current_gt, "prediction": current_prediction}
        self.read_layers = set(options.layers)
        if options.mode == "selected_boxes":
            self.read_layers.update(layer for layer, _ in options.selected)
        self.resolve = comparison.export_resolver(self.read_layers) if comparison else None
        self.entries = dict(entries)
        self.project_boxes = project_boxes
        self.directory = None
        self.written = 0
        self.skipped = 0
        self.errors = []
        self.cancelled = False

    def _boxes(self, path, image):
        result = {"gt": [], "prediction": []}
        pair = None
        for layer in self.read_layers:
            if path == self.current_path and self.current_boxes[layer] is not None:
                result[layer] = self.current_boxes[layer]
            elif self.resolve:
                pair = pair or self.resolve(path)
                result[layer] = AnnotationXml.open_path(pair[layer]).boxes
            elif layer == "gt":
                entry = ImageEntry.lookup(path, self.entries)
                doc = AnnotationXml.for_image(entry, image.width(), image.height(), self.project_boxes.get(path, []))
                result[layer] = doc.boxes
        return result

    def _write_image(self, number, path, image, boxes):
        mode = self.options.mode
        selected = None
        if mode == "selected_boxes":
            selected = self.options.selected
        elif mode == "class_crops":
            selected = {(layer, i) for layer, records in boxes.items()
                        for i, box in enumerate(records) if box["label"] == self.options.label}
        exports = iter_visual_exports(image, boxes["gt"], boxes["prediction"],
                                      crops=mode not in ("current_image", "all_images"),
                                      layers=self.options.layers, selected=selected)
        base = f"{number:06d}_{_filename_part(Path(path).stem)}"
        for layer, i, rendered in exports:
            if self.isInterruptionRequested():
                break
            suffix = ""
            if layer:
                label = _filename_part(boxes[layer][i]["label"])
                source = "GT" if layer == "gt" else "Pred"
                suffix = f"_{source}-{i + 1}_{label}"
            destination = self.options.directory / f"{base}{suffix}.png"
            save_png(rendered, destination, occupied=self.occupied_names)
            self.directory = destination.parent
            self.written += 1

    def run(self):
        try:
            self.occupied_names = {p.name.casefold() for p in self.options.directory.iterdir()}
            for completed, (number, path) in enumerate(self.images, start=1):
                if self.isInterruptionRequested():
                    break
                try:
                    image = (self.current_image if path == self.current_path and self.current_image is not None
                             else QImage(path))
                    if image.isNull():
                        raise ValueError("无法读取图片")
                    boxes = self._boxes(path, image)
                except (OSError, ValueError) as error:
                    self.skipped += 1
                    if len(self.errors) < 6:
                        self.errors.append(f"{path}：{error}")
                else:
                    self._write_image(number, path, image, boxes)
                self.progress.emit(completed)
        except Exception as error:
            self.errors.append(f"导出中断：{error}")
        finally:
            self.cancelled = self.isInterruptionRequested()
