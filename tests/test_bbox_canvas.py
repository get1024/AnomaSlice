"""Offscreen interaction checks for the bounding-box canvas.

Run with ``python -m unittest discover -s tests -v``. QtTest comes with the
runtime's PySide6 Essentials package, so these checks add no runtime dependency.
"""

import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QAction, QColor, QImage, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMainWindow

from bbox_canvas import BoxCanvasMixin
from main import _SliceCanvas


class _TestCanvas(BoxCanvasMixin, _SliceCanvas):
    pass


class BoxCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.image_path = str(Path(self.tmp.name) / "原图.png")
        image = QImage(500, 400, QImage.Format_RGB32)
        image.fill(QColor("gray"))
        self.assertTrue(image.save(self.image_path))
        self.window = QMainWindow()
        self.canvas = _TestCanvas()
        self.window.setCentralWidget(self.canvas)
        self.window.resize(800, 600)
        self.window.show()
        self.app.processEvents()
        self.assertTrue(self.canvas.load_image(self.image_path))
        self.canvas.set_annotation_mode(True)
        self.canvas.resolve_label = lambda text: {"1": "划痕"}.get(text, text)
        self.changes = []
        self.canvas.annotations_changed = self.changes.append

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.tmp.cleanup()

    def drag(self, start, end):
        start = self.canvas.mapFromScene(QPointF(*start))
        end = self.canvas.mapFromScene(QPointF(*end))
        QTest.mousePress(self.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, start)
        QTest.mouseMove(self.canvas.viewport(), end)
        QTest.mouseRelease(self.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, end)

    def commit(self, text):
        self.assertTrue(self.canvas.has_pending_annotation())
        self.assertTrue(self.canvas._label_editor.hasFocus())
        self.canvas._label_editor.setText(text)
        QTest.keyClick(self.canvas._label_editor, Qt.Key_Return)
        self.assertFalse(self.canvas.has_pending_annotation())

    def wheel(self, point, delta=120):
        event = QWheelEvent(
            QPointF(point), QPointF(self.canvas.viewport().mapToGlobal(point)),
            QPoint(), QPoint(0, delta), Qt.NoButton, Qt.NoModifier,
            Qt.NoScrollPhase, False,
        )
        self.app.sendEvent(self.canvas.viewport(), event)

    def test_exact_alias_matching_and_source_coordinates(self):
        self.drag((100, 100), (200, 180))
        self.commit("1")
        first = self.canvas.get_annotations()[0]
        self.assertEqual(first["label"], "划痕")
        for actual, expected in zip(first["bbox"], [100, 100, 100, 80]):
            self.assertAlmostEqual(actual, expected, delta=1)
        self.drag((250, 100), (350, 180))
        self.commit("待定1")
        self.assertEqual(self.canvas.get_annotations()[1]["label"], "待定1")
        # Saved metadata and callback snapshots cannot mutate the live canvas.
        snapshot = self.canvas.get_annotations()
        snapshot[0]["bbox"][0] = 999
        self.assertNotEqual(self.canvas.get_annotations()[0]["bbox"][0], 999)
        self.changes[0][0]["label"] = "external mutation"
        self.assertEqual(self.canvas.get_annotations()[0]["label"], "划痕")

    def test_reverse_drag_clips_at_source_image_boundary(self):
        self.drag((300, 250), (-50, -50))
        self.commit("边界")
        x, y, width, height = self.canvas.get_annotations()[0]["bbox"]
        self.assertEqual((x, y), (0, 0))
        self.assertAlmostEqual(width, 300, delta=1)
        self.assertAlmostEqual(height, 250, delta=1)
        self.assertLessEqual(x + width, 500)
        self.assertLessEqual(y + height, 400)

    def test_click_does_not_create_box_and_can_select(self):
        self.canvas.set_annotations([{"label": "划痕", "bbox": [100, 100, 100, 80]}])
        point = self.canvas.mapFromScene(QPointF(130, 130))
        QTest.mouseClick(self.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, point)
        self.assertFalse(self.canvas.has_pending_annotation())
        self.assertEqual(self.canvas._selected_annotation, 0)
        QTest.keyClick(self.canvas, Qt.Key_Delete)
        self.assertEqual(self.canvas.get_annotations(), [])
        self.canvas.annotation_undo()
        self.assertEqual(len(self.canvas.get_annotations()), 1)

    def test_undo_redo_and_rename_preserve_mask(self):
        original_mask = self.canvas.get_mask()
        self.drag((100, 100), (200, 200))
        self.commit("1")
        self.canvas.annotation_undo()
        self.assertEqual(self.canvas.get_annotations(), [])
        self.canvas.annotation_redo()
        self.assertEqual(len(self.canvas.get_annotations()), 1)
        point = self.canvas.mapFromScene(QPointF(150, 150))
        QTest.mouseDClick(self.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, point)
        self.commit("待定1")
        self.assertEqual(self.canvas.get_annotations()[0]["label"], "待定1")
        self.canvas.annotation_undo()
        self.assertEqual(self.canvas.get_annotations()[0]["label"], "划痕")
        if original_mask is None:
            self.assertIsNone(self.canvas.get_mask())
        else:
            self.assertTrue((original_mask == self.canvas.get_mask()).all())

    def test_pending_boxes_are_cancelled_without_committing(self):
        self.drag((10, 10), (100, 100))
        QTest.keyClick(self.canvas._label_editor, Qt.Key_Escape)
        self.assertFalse(self.canvas.has_pending_annotation())
        self.assertEqual(self.canvas.get_annotations(), [])
        self.drag((10, 10), (100, 100))
        self.canvas.annotation_undo()
        self.assertFalse(self.canvas.has_pending_annotation())
        self.assertEqual(self.canvas.get_annotations(), [])
        self.drag((10, 10), (100, 100))
        self.canvas.set_annotation_mode(False)
        self.assertFalse(self.canvas.has_pending_annotation())
        self.assertEqual(self.canvas.get_annotations(), [])

    def test_zoom_pan_and_resize_keep_source_geometry(self):
        cursor = QPoint(450, 300)
        before = self.canvas.mapToScene(cursor)
        self.wheel(cursor)
        after = self.canvas.mapToScene(cursor)
        self.assertLess((before - after).manhattanLength(), 3)
        scale = self.canvas.transform().m11()
        self.window.resize(900, 600)
        self.app.processEvents()
        self.assertEqual(self.canvas.transform().m11(), scale)
        before = self.canvas.mapToScene(self.canvas.viewport().rect().center())
        QTest.mousePress(self.canvas.viewport(), Qt.RightButton, Qt.NoModifier, QPoint(400, 300))
        QTest.mouseMove(self.canvas.viewport(), QPoint(450, 330))
        QTest.mouseRelease(self.canvas.viewport(), Qt.RightButton, Qt.NoModifier, QPoint(450, 330))
        after = self.canvas.mapToScene(self.canvas.viewport().rect().center())
        self.assertGreater((before - after).manhattanLength(), 10)
        self.drag((100, 100), (200, 180))
        self.commit("1")
        for actual, expected in zip(self.canvas.get_annotations()[0]["bbox"], [100, 100, 100, 80]):
            self.assertAlmostEqual(actual, expected, delta=2)
        self.canvas.fit_image()
        self.assertFalse(self.canvas._user_view_adjusted)

    def test_editor_owns_enter_and_text_undo_over_window_shortcuts(self):
        triggered = []
        for shortcut in ("Return", "Ctrl+Z", "Delete"):
            action = QAction(self.window)
            action.setShortcut(shortcut)
            action.triggered.connect(lambda checked=False, key=shortcut: triggered.append(key))
            self.window.addAction(action)
        self.drag((100, 100), (200, 200))
        editor = self.canvas._label_editor
        QTest.keyClicks(editor, "abc")
        QTest.keyClick(editor, Qt.Key_Z, Qt.ControlModifier)
        self.assertEqual(editor.text(), "")
        QTest.keyClicks(editor, "scratch")
        QTest.keyClick(editor, Qt.Key_Return)
        self.assertEqual(self.canvas.get_annotations()[0]["label"], "scratch")
        self.assertEqual(triggered, [])

    def test_slice_mode_retains_click_signal_and_new_image_resets_box_history(self):
        self.drag((100, 100), (200, 180))
        self.commit("1")
        self.canvas.set_annotation_mode(False)
        points = []
        self.canvas.canvas_clicked.connect(lambda x, y: points.append((x, y)))
        point = self.canvas.mapFromScene(QPointF(30, 40))
        QTest.mouseClick(self.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, point)
        self.assertEqual(len(points), 1)
        self.assertEqual(len(self.canvas.get_annotations()), 1)
        self.assertTrue(self.canvas.load_image(self.image_path))
        self.canvas.annotation_undo()
        self.assertEqual(self.canvas.get_annotations(), [])


if __name__ == "__main__":
    unittest.main()
