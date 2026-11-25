"""Offscreen interaction checks for the bounding-box canvas.

Run with ``python -m unittest discover -s tests -v``. QtTest comes with the
runtime's PySide6 Essentials package, so these checks add no runtime dependency.
"""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QAction, QColor, QImage, QInputDevice, QNativeGestureEvent, QPointingDevice, QWheelEvent,
)
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMainWindow

from bbox_canvas import BoxCanvasMixin
from image_canvas import SliceCanvas


class _TestCanvas(BoxCanvasMixin, SliceCanvas):
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

    def wheel(self, point, delta=120, pixel_delta_y=0):
        event = QWheelEvent(
            QPointF(point), QPointF(self.canvas.viewport().mapToGlobal(point)),
            QPoint(0, pixel_delta_y), QPoint(0, delta), Qt.NoButton, Qt.NoModifier,
            Qt.NoScrollPhase, False,
        )
        self.app.sendEvent(self.canvas.viewport(), event)

    def trackpad(self):
        return QPointingDevice(
            "Trackpad", 42, QInputDevice.DeviceType.TouchPad, QPointingDevice.PointerType.Finger,
            QInputDevice.Capability.Position, 5, 0,
        )

    def test_mac_two_finger_scroll_moves_image_in_both_axes_without_zoom(self):
        viewport = self.canvas.viewport()
        point = QPoint(370, 260)
        image_point = self.canvas.mapToScene(point)
        delta = QPoint(20, -13)
        scale = self.canvas.transform().m11()
        device = self.trackpad()
        with patch.object(sys, "platform", "darwin"):
            event = QWheelEvent(
                QPointF(point), QPointF(viewport.mapToGlobal(point)), delta, QPoint(30, -30),
                Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False,
                Qt.MouseEventNotSynthesized, device,
            )
            self.app.sendEvent(viewport, event)
            self.assertEqual(self.canvas.transform().m11(), scale)
            self.assertLessEqual((self.canvas.mapFromScene(image_point) - point - delta).manhattanLength(), 2)
            self.wheel(point, pixel_delta_y=15)
            self.assertGreater(self.canvas.transform().m11(), scale)

    def test_mac_native_pinch_zooms_in_and_out_around_fingers(self):
        viewport = self.canvas.viewport()
        point = QPoint(340, 230)
        anchor = self.canvas.mapToScene(point)
        device = self.trackpad()
        with patch.object(sys, "platform", "darwin"):
            for value in (0.25, -0.2):
                before = self.canvas.transform().m11()
                event = QNativeGestureEvent(
                    Qt.ZoomNativeGesture, device, 2, QPointF(point),
                    QPointF(viewport.mapTo(self.window, point)), QPointF(viewport.mapToGlobal(point)),
                    value, QPointF(),
                )
                self.app.sendEvent(viewport, event)
                after = self.canvas.transform().m11()
                if value > 0:
                    self.assertGreater(after, before)
                else:
                    self.assertLess(after, before)
                self.assertLessEqual((self.canvas.mapFromScene(anchor) - point).manhattanLength(), 3)
            self.assertFalse(self.canvas.has_pending_annotation())
            # macOS delivers its accessibility three-finger drag as a primary-button drag.
            self.drag((100, 100), (200, 180))
            self.commit("scratch")
            self.assertEqual(self.canvas.get_annotations()[0]["label"], "scratch")

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

    def test_prediction_overlay_is_visible_but_only_gt_can_be_edited(self):
        gt = [{"label": "GT", "bbox": [100, 100, 100, 80]}]
        predictions = [
            {"label": "overlap", "bbox": [100, 100, 100, 80]},
            {"label": "prediction only", "bbox": [300, 120, 60, 40]},
        ]
        self.canvas.set_annotations(gt)
        self.canvas.set_predictions(predictions)
        outline, caption = self.canvas._prediction_items[:2]
        gt_outline, gt_caption = self.canvas._annotation_items
        self.assertEqual(outline.rect(), gt_outline.rect())
        self.assertGreater(outline.zValue(), gt_outline.zValue())
        self.assertEqual(outline.pen().style(), Qt.DashLine)
        self.assertEqual(gt_outline.pen().style(), Qt.SolidLine)
        self.assertNotEqual(caption.pos(), gt_caption.pos())
        self.assertTrue(all(item.isVisible() for item in self.canvas._prediction_items))

        prediction_only = self.canvas.mapFromScene(QPointF(320, 130))
        QTest.mouseClick(self.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, prediction_only)
        self.assertIsNone(self.canvas.selected_annotation)
        QTest.mouseDClick(self.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, prediction_only)
        self.assertFalse(self.canvas.has_pending_annotation())
        QTest.keyClick(self.canvas, Qt.Key_Delete)
        self.assertEqual(self.changes, [])

        overlap = self.canvas.mapFromScene(QPointF(120, 130))
        QTest.mouseClick(self.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, overlap)
        QTest.keyClick(self.canvas, Qt.Key_Delete)
        self.canvas.annotation_undo()
        self.canvas.annotation_redo()
        self.assertEqual(self.changes, [[], gt, []])
        self.assertEqual(len(self.canvas._prediction_items), 4)
        self.assertIs(outline.scene(), self.canvas.scene())

        self.canvas.set_annotation_mode(False)
        self.assertFalse(outline.isVisible())
        self.canvas.set_annotation_mode(True)
        self.assertTrue(outline.isVisible())
        self.assertTrue(self.canvas.load_image(self.image_path))
        self.assertEqual(self.canvas._prediction_items, [])
        self.assertIsNone(outline.scene())

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
            self.assertEqual(original_mask, self.canvas.get_mask())

    def test_captions_for_tiny_matching_boxes_stay_separate(self):
        for y in (0, 100, 397):
            with self.subTest(y=y):
                boxes = [{"label": "scratch", "bbox": [60, y, 3, 3]}]
                self.canvas.set_annotations(boxes)
                self.canvas.set_predictions(boxes)
                captions = (self.canvas._annotation_items[1], self.canvas._prediction_items[1])
                bounds = [item.deviceTransform(self.canvas.viewportTransform()).mapRect(item.boundingRect())
                          for item in captions]
                self.assertFalse(bounds[0].intersects(bounds[1]))

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

    def test_confirming_existing_label_preserves_text_and_edit_history(self):
        point = self.canvas.mapFromScene(QPointF(130, 130))
        for original, entered in (("1", None), ("划痕", None), ("划痕", "1")):
            with self.subTest(original=original, entered=entered):
                boxes = [{"label": original, "bbox": [100, 100, 100, 80], "source_id": 0}]
                self.canvas.set_annotations(boxes)
                self.changes.clear()
                QTest.mouseDClick(self.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, point)
                self.assertTrue(self.canvas.has_pending_annotation())
                editor = self.canvas._label_editor
                if entered is not None:
                    editor.setText(entered)
                QTest.keyClick(editor, Qt.Key_Return)
                self.assertFalse(self.canvas.has_pending_annotation())
                self.assertEqual(self.canvas.get_annotations(), boxes)
                self.canvas.annotation_undo()
                self.assertEqual(self.changes, [])

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
