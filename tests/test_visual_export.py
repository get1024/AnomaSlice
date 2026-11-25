import errno
import os
from concurrent.futures import ThreadPoolExecutor
import tempfile
import unittest
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from visual_export import iter_visual_exports, save_png


class VisualExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.original = QImage(240, 160, QImage.Format_RGB32)
        self.original.fill(QColor("black"))
        self.original.setPixelColor(3, 6, QColor("yellow"))
        self.gt = [{"bbox": [20, 48, 90, 80], "label": "划痕"}]
        self.predictions = [{"bbox": [130, 48, 90, 80], "label": "误检"}]

    def full(self, layers):
        (layer, index, image), = iter_visual_exports(
            self.original, self.gt, self.predictions, crops=False, layers=layers)
        self.assertIsNone(layer)
        self.assertIsNone(index)
        return image

    def test_full_resolution_selected_layers_palette_and_captions(self):
        gt = self.full(("gt",))
        pred = self.full(("prediction",))
        both = self.full(("gt", "prediction"))
        black = QColor("black")
        cyan = QColor("#00d5e8")
        self.assertEqual(both.size(), self.original.size())
        for image in (gt, both):
            self.assertEqual(image.pixelColor(20, 88), QColor("red"))
            self.assertTrue(any(image.pixelColor(x, y).lightness() > 140
                                for x in range(24, 105) for y in range(25, 48)))
        for image in (pred, both):
            edge = {image.pixelColor(x, 48).name() for x in range(133, 216)}
            self.assertEqual(edge, {black.name(), cyan.name()})
        self.assertEqual(gt.pixelColor(133, 48), black)
        self.assertEqual(pred.pixelColor(20, 88), black)
        self.assertEqual(self.original.pixelColor(20, 88), black)
        self.assertEqual(both.pixelColor(3, 6), QColor("yellow"))

    def test_crop_fractional_edges_selection_and_overlapping_layers(self):
        self.gt = [{"bbox": [310, 0, 2, 8], "label": "图外"},
                   {"bbox": [-4.5, 3.75, 120.1, 110], "label": "掉漆"}]
        self.predictions = [{"bbox": [20, 45, 70, 40], "label": "污渍"}]
        crops = list(iter_visual_exports(
            self.original, self.gt, self.predictions, crops=True,
            layers=("gt", "prediction"), selected={("gt", 0), ("gt", 1)}))
        self.assertEqual(len(crops), 1)
        layer, index, image = crops[0]
        self.assertEqual((layer, index, image.size()), ("gt", 1, QSize(116, 111)))
        edge = {image.pixelColor(x, 42).name() for x in range(22, 88)}
        self.assertEqual(edge, {"#000000", "#00d5e8"})
        self.assertTrue(any(image.pixelColor(x, y).lightness() > 180
                            for x in range(5, 80) for y in range(4, 20)))
        self.assertEqual(image.pixelColor(40, 36), QColor("black"))

    def test_tiny_indexed_crop_and_png_encoding_failure(self):
        image = self.original.convertToFormat(QImage.Format_Indexed8)
        boxes = [{"bbox": [8, 8, 1, 1], "label": "自定义", "color": "#123456"}]
        (_, _, crop), = iter_visual_exports(image, boxes, [], crops=True, layers=("gt",))
        self.assertEqual(crop.size(), QSize(1, 1))
        self.assertEqual(crop.pixelColor(0, 0), QColor("#123456"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "裁剪.png"
            save_png(crop, path)
            contents = path.read_bytes()
            self.assertEqual(QImage(str(path)).pixelColor(0, 0), QColor("#123456"))
            with self.assertRaisesRegex(OSError, "裁剪.png"):
                save_png(QImage(), path)
            self.assertEqual(path.read_bytes(), contents)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_simultaneous_writers_preserve_existing_file_and_publish_distinct_complete_pngs(self):
        backgrounds = ("blue", "red", "yellow", "white")
        ready = Barrier(len(backgrounds))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "导出.png"
            path.write_bytes(b"existing")

            def publish(color):
                image = QImage(24, 24, QImage.Format_RGB32)
                image.fill(QColor(color))
                ready.wait(timeout=5)
                # Simulate jobs with stale directory listings.
                return save_png(image, path, occupied=set())

            with ThreadPoolExecutor(max_workers=len(backgrounds)) as pool:
                outputs = list(pool.map(publish, backgrounds))
            self.assertEqual(set(outputs), {path.with_name(f"导出 ({i}).png") for i in range(2, 6)})
            self.assertEqual(set(path.parent.iterdir()), {path, *outputs})
            self.assertEqual(path.read_bytes(), b"existing")
            for output, background in zip(outputs, backgrounds):
                self.assertEqual(QImage(str(output)).pixelColor(23, 23), QColor(background))

    def test_crop_region_can_come_from_hidden_layer(self):
        outer = [{"bbox": [20, 30, 110, 90], "label": "划痕"}]
        inner = [{"bbox": [34, 60, 60, 32], "label": "划痕"}]
        for selected, drawn, gt, prediction, color in (
            ("gt", "prediction", outer, inner, "#00d5e8"),
            ("prediction", "gt", inner, outer, "#ff0000"),
        ):
            with self.subTest(selected=selected):
                (layer, index, crop), = iter_visual_exports(
                    self.original, gt, prediction, crops=True, layers=(drawn,),
                    selected={(selected, 0)})
                self.assertEqual((layer, index, crop.size()), (selected, 0, QSize(110, 90)))
                self.assertEqual(crop.pixelColor(0, 40), QColor("black"))
                edge = {crop.pixelColor(x, 30).name() for x in range(17, 69)}
                self.assertIn(color, edge)

    def test_volume_without_exclusive_rename_or_links_preserves_files_and_cleans_failed_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "图.png"
            path.write_bytes(b"existing")
            with patch("visual_export.sys.platform", "darwin"), \
                 patch("visual_export._macos_rename_exclusive", side_effect=OSError(errno.ENOTSUP, "rename")), \
                 patch("visual_export.os.link", side_effect=OSError(errno.ENOTSUP, "link")):
                new = save_png(self.original, path, occupied=set())
                self.assertEqual(new, path.with_name("图 (2).png"))
                self.assertEqual(QImage(str(new)).pixelColor(3, 6), QColor("yellow"))

                def disk_full(source, output):
                    output.write(source.read(8))
                    raise OSError(errno.ENOSPC, "disk full")

                with patch("visual_export.shutil.copyfileobj", side_effect=disk_full):
                    with self.assertRaises(OSError) as error:
                        save_png(self.original, path)
                self.assertEqual(error.exception.errno, errno.ENOSPC)
                self.assertEqual(set(path.parent.iterdir()), {path, new})
                self.assertEqual(path.read_bytes(), b"existing")

    def test_rename_permission_and_space_errors_do_not_attempt_a_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "图.png"
            for err in (errno.EACCES, errno.ENOSPC):
                with self.subTest(err=err), patch("visual_export.sys.platform", "darwin"), \
                     patch("visual_export._macos_rename_exclusive", side_effect=OSError(err, "failed")), \
                     patch("visual_export.os.link") as link:
                    with self.assertRaises(OSError) as error:
                        save_png(self.original, path)
                    self.assertEqual(error.exception.errno, err)
                    link.assert_not_called()
                    self.assertEqual(list(path.parent.iterdir()), [])

    def test_no_boxes_and_empty_selection(self):
        (_, _, full), = iter_visual_exports(self.original, [], [], crops=False, layers=("gt",))
        self.assertEqual(full, self.original)
        self.assertEqual(list(iter_visual_exports(
            self.original, self.gt, [], crops=True, layers=("gt",), selected=set())), [])
