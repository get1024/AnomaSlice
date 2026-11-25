"""Qt-only image processing keeps legacy slicing and PNG projects compatible."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect
from PySide6.QtGui import QColor, QImage, QPainter

from project_store import ProjectManager
from slicer import SlicerThread, count_mask_pixels
from xml_annotations import scan_folder


class QtSlicerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.normal = self.root / "正常"
        self.abnormal = self.root / "缺陷"

    @staticmethod
    def image(width, height, gray=False):
        image = QImage(width, height, QImage.Format_Grayscale8 if gray else QImage.Format_RGB888)
        image.fill(0 if gray else QColor("red"))
        return image

    def worker(self, image, mask=None, threshold=.01, size=4):
        worker = SlicerThread(
            image, mask, size, size, size, threshold, str(self.normal), str(self.abnormal),
            False, None, "样本",
        )
        self.success, self.errors = [], []
        worker.finished_signal.connect(lambda normal, abnormal: self.success.append((normal, abnormal)))
        worker.error_signal.connect(self.errors.append)
        return worker

    def test_mask_count_ignores_alignment_and_clips_source_rectangles(self):
        mask = self.image(3, 2, True)
        data = mask.bits()
        for y in range(mask.height()):
            data[y * mask.bytesPerLine() + 3] = 255  # padding is not a pixel
        mask.setPixelColor(0, 0, QColor("white"))
        mask.setPixelColor(1, 1, QColor("white"))
        self.assertEqual(count_mask_pixels(mask), 2)
        self.assertEqual(count_mask_pixels(mask, QRect(1, 0, 9, 9)), 1)
        self.assertEqual(count_mask_pixels(mask, QRect(20, 20, 1, 1)), 0)
        self.assertEqual(count_mask_pixels(None), 0)

    def test_exact_threshold_is_normal_and_greater_fraction_is_abnormal(self):
        image = self.image(8, 4)
        mask = self.image(8, 4, True)
        painter = QPainter(mask)
        painter.fillRect(0, 0, 1, 4, QColor("white"))
        painter.fillRect(4, 0, 2, 4, QColor("white"))
        painter.end()
        worker = self.worker(image, mask, threshold=.25)
        worker.run()
        self.assertEqual(self.errors, [])
        self.assertEqual([p.name for p in self.normal.iterdir()], ["样本_0_0.png"])
        self.assertEqual([p.name for p in self.abnormal.iterdir()], ["样本_4_0.png"])
        self.assertEqual(len(self.success), 1)
        output = QImage(str(self.abnormal / "样本_4_0.png"))
        self.assertEqual((output.width(), output.height()), (4, 4))
        self.assertEqual(output.pixelColor(0, 0), QColor("red"))

    def test_optional_mask_and_implicit_image_snapshots(self):
        image = self.image(4, 4)
        worker = self.worker(image)
        image.fill(QColor("blue"))
        worker.run()
        output = QImage(str(self.normal / "样本_0_0.png"))
        self.assertEqual(output.pixelColor(0, 0), QColor("red"))
        self.assertEqual(self.errors, [])

    def test_dataset_scan_distinguishes_new_crops_from_unmarked_camera_images(self):
        image = self.image(4, 4)
        original = self.root / "camera_0_0.png"
        self.assertTrue(image.save(str(original)))
        self.worker(image).run()
        crop = self.normal / "样本_0_0.png"
        self.assertEqual([entry.image for entry in scan_folder(self.root)], [original])
        self.assertEqual([entry.image for entry in scan_folder(self.root, selected=crop)], [crop])
        self.assertEqual([entry.image for entry in scan_folder(self.normal)], [crop])

    def test_size_mismatch_reports_error_without_success(self):
        worker = self.worker(self.image(4, 4), self.image(2, 2, True))
        worker.run()
        self.assertEqual(len(self.errors), 1)
        self.assertIn("尺寸", self.errors[0])
        self.assertEqual(self.success, [])

    def test_write_failure_reports_error_without_success(self):
        worker = self.worker(self.image(4, 4))
        with patch.object(worker, "_save_png", side_effect=OSError("disk full")):
            worker.run()
        self.assertEqual(self.errors, ["disk full"])
        self.assertEqual(self.success, [])

    def test_qt_mask_project_roundtrip_keeps_existing_format(self):
        manager = ProjectManager()
        manager.set_images(["原图.png"])
        mask = self.image(5, 7, True)
        mask.setPixelColor(2, 3, QColor("white"))
        manager.set_mask("原图.png", mask)
        mask.fill(0)
        manager.annotations = {"原图.png": [{"label": "划痕", "bbox": [1, 2, 3, 4]}]}
        path = str(self.root / "标注工程.json")
        manager.save_project(path, 0, {"stride": 128})
        reopened = ProjectManager()
        reopened.load_project(path)
        self.assertEqual(reopened.annotations, manager.annotations)
        self.assertEqual(count_mask_pixels(reopened.get_mask("原图.png")), 1)
        self.assertEqual(reopened.get_mask("原图.png").size(), mask.size())

    def test_mask_autosave_merges_other_images_and_rejects_stale_edits_to_same_image(self):
        owner = ProjectManager()
        owner.set_images(["a.png", "b.png"])
        path = str(self.root / "review.json")
        owner.save_project(path, 0, {})
        second = ProjectManager()
        second.load_project(path)
        mask = self.image(3, 3, True)
        mask.setPixelColor(1, 1, QColor("white"))
        owner.save_image_mask("a.png", mask)
        second.save_image_mask("b.png", mask)
        with self.assertRaisesRegex(ValueError, "掩膜已被修改"):
            second.save_image_mask("a.png", self.image(3, 3, True))
        second.save_project(path, 1, {})
        reopened = ProjectManager()
        reopened.load_project(path)
        self.assertEqual(reopened.get_mask("a.png"), mask)
        self.assertEqual(reopened.get_mask("b.png"), mask)


if __name__ == "__main__":
    unittest.main()
