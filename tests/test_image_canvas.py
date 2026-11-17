"""Check native QImage masks, including legacy inclusive grid boundaries."""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from image_canvas import ScalableImageLabel, SliceCanvas


class ImageCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.temp.name) / "原图.png")
        self.image = QImage(11, 9, QImage.Format_RGB32)
        self.image.fill(QColor("gray"))
        self.assertTrue(self.image.save(self.path))
        self.canvas = SliceCanvas()
        self.assertTrue(self.canvas.load_image(self.path))

    def tearDown(self):
        self.canvas.close()
        self.canvas.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def value(self, x, y):
        return self.canvas.get_mask().pixelColor(x, y).red()

    def test_mask_is_lazy_and_decoded_source_can_be_reused(self):
        self.assertIsNone(self.canvas.get_mask())
        self.assertTrue(self.canvas.load_image("not-read.png", self.image))
        self.canvas.ensure_slice_mask()
        mask = self.canvas.get_mask()
        self.assertEqual(mask.format(), QImage.Format_Grayscale8)
        self.assertEqual(mask.size(), self.image.size())
        self.assertEqual(self.value(10, 8), 0)

    def test_inclusive_fill_clips_at_last_image_pixel(self):
        self.canvas.ensure_slice_mask()
        self.canvas.fill_rect(QRectF(2, 3, 2, 2), 255)
        self.assertEqual(self.value(4, 5), 255)
        self.assertEqual(self.value(5, 5), 0)
        self.assertEqual(self.value(4, 6), 0)
        self.canvas.fill_rect(QRectF(10, 8, 10, 10), 255)
        self.assertEqual(self.value(10, 8), 255)
        self.assertEqual(self.canvas.get_mask().size(), self.image.size())

    def test_snapshots_and_undo_redo_are_isolated(self):
        self.canvas.ensure_slice_mask()
        before = self.canvas.get_mask()
        self.canvas.fill_rect(QRectF(2, 3, 2, 2), 255)
        after = self.canvas.get_mask()
        self.assertEqual(before.pixelColor(2, 3).red(), 0)
        self.assertEqual(after.pixelColor(2, 3).red(), 255)
        after.fill(0)
        self.assertEqual(self.value(2, 3), 255)
        self.canvas.undo()
        self.assertEqual(self.value(2, 3), 0)
        self.canvas.redo()
        self.assertEqual(self.value(2, 3), 255)
        self.canvas.clear_mask()
        self.assertEqual(self.value(2, 3), 0)
        self.canvas.undo()
        self.assertEqual(self.value(2, 3), 255)

    def test_overlay_uses_mask_as_alpha_with_red_foreground(self):
        self.canvas.ensure_slice_mask()
        self.canvas.fill_rect(QRectF(2, 3, 2, 2), 255)
        overlay = self.canvas._mask_item.pixmap().toImage()
        self.assertEqual(overlay.pixelColor(0, 0).alpha(), 0)
        self.assertEqual(overlay.pixelColor(2, 3), QColor(255, 0, 0, 255))
        self.canvas.fill_rect(QRectF(6, 2, 1, 1), 128)
        overlay = self.canvas._mask_item.pixmap().toImage()
        self.assertEqual(overlay.pixelColor(6, 2).alpha(), 128)

    def test_set_mask_does_not_alias_callers_image(self):
        mask = QImage(11, 9, QImage.Format_Grayscale8)
        mask.fill(255)
        self.canvas.set_mask(mask)
        mask.fill(0)
        self.assertEqual(self.value(10, 8), 255)
        self.canvas.fill_rect(QRectF(1, 1, 1, 1), 0)
        self.assertEqual(mask.pixelColor(1, 1).red(), 0)
        self.canvas.reset_view()
        self.assertIsNone(self.canvas.get_mask())
        self.assertTrue(self.canvas._mask_item.pixmap().isNull())

    def test_preview_text_clears_a_previous_picture(self):
        label = ScalableImageLabel()
        label.setPixmap(self.canvas._pixmap_item.pixmap())
        label.setText("标注完成")
        self.assertIsNone(label._original_pixmap)
        self.assertEqual(label.text(), "标注完成")


if __name__ == "__main__":
    unittest.main()
