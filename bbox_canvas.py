"""Small, image-coordinate bounding-box editor shared with the slicing canvas."""

from copy import deepcopy
import math

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QKeySequence, QPen
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsView,
    QLineEdit,
)

from xml_annotations import label_color


class _BoxLabel(QGraphicsItem):
    """A constant-screen-size caption outside the defect rectangle."""

    def __init__(self, text, color, room_above):
        super().__init__()
        self._text = text
        self._font = QFont("Microsoft YaHei UI", 9)
        metrics = QFontMetricsF(self._font)
        height = metrics.height() + 4
        self.above = room_above >= height + 2
        self._rect = QRectF(0, -height - 2 if self.above else 2, metrics.horizontalAdvance(text) + 10, height)
        self._color = color
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        self.setAcceptedMouseButtons(Qt.NoButton)

    def boundingRect(self):
        return self._rect

    def paint(self, painter, option, widget=None):
        painter.fillRect(self._rect, QColor(24, 28, 35, 225))
        painter.fillRect(QRectF(0, self._rect.y(), 2, self._rect.height()), self._color)
        painter.setFont(self._font)
        painter.setPen(Qt.white)
        painter.drawText(self._rect.adjusted(5, 0, -2, 0), Qt.AlignLeft | Qt.AlignVCenter, self._text)


class BoxCanvasMixin:
    """Add bbox interaction without changing the existing grid/mask workflow.

    ``annotations_changed`` receives a deep-copied list after committed edits.
    ``resolve_label`` is called once with the complete trimmed editor value.
    ``new_label_committed`` receives the resulting canonical label.
    """

    def __init__(self, *args, **kwargs):
        self._annotation_mode = False
        self._annotations = []
        self._annotation_items = []
        self._annotation_undo_stack = []
        self._annotation_redo_stack = []
        self._selected_annotation = None
        self._pending_rect = None
        self._rename_index = None
        self._draft_item = None
        self._box_start = None
        self._box_press_pos = None
        self._pan_pos = None
        self._user_view_adjusted = False
        self._fit_scale = 1.0
        self.resolve_label = lambda text: text
        self.annotations_changed = lambda annotations: None
        self.selection_changed = lambda index: None
        self.new_label_committed = lambda label: None
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self._label_editor = QLineEdit(self.viewport())
        self._label_editor.setPlaceholderText("标签 / 别名 · Enter 确认 · Esc 取消")
        self._label_editor.setMinimumWidth(220)
        self._label_editor.setStyleSheet(
            "QLineEdit { background: #202936; color: white; border: 2px solid #49baff;"
            " padding: 6px 9px; border-radius: 4px; font-size: 13px; }"
        )
        self._label_editor.hide()
        self._label_editor.installEventFilter(self)
        self._label_editor.returnPressed.connect(self._commit_pending_annotation)

    def set_annotation_mode(self, enabled):
        self.cancel_pending_annotation()
        self._annotation_mode = bool(enabled)
        self._grid_item.setVisible(not self._annotation_mode)
        self._mask_item.setVisible(not self._annotation_mode)
        for item in self._annotation_items:
            item.setVisible(self._annotation_mode)
        self.viewport().setCursor(Qt.CrossCursor if self._annotation_mode else Qt.ArrowCursor)

    @property
    def selected_annotation(self):
        return self._selected_annotation

    def select_annotation(self, index):
        selected = index if 0 <= index < len(self._annotations) else None
        if self._selected_annotation != selected:
            self._selected_annotation = selected
            self._render_annotations()

    def get_annotations(self):
        return deepcopy(self._annotations)

    def set_annotations(self, annotations):
        self.cancel_pending_annotation()
        self._annotations = deepcopy(annotations or [])
        self._selected_annotation = None
        self._annotation_undo_stack.clear()
        self._annotation_redo_stack.clear()
        self._render_annotations()

    def has_pending_annotation(self):
        return self._pending_rect is not None or self._rename_index is not None

    def _image_rect(self):
        return self._pixmap_item.boundingRect()

    def _has_image(self):
        return not self._pixmap_item.pixmap().isNull()

    def _clamp_point(self, point):
        rect = self._image_rect()
        return QPointF(
            min(max(point.x(), rect.left()), rect.right()),
            min(max(point.y(), rect.top()), rect.bottom()),
        )

    def _expand_view_scene(self):
        """Leave room for right-button panning even while the whole image fits."""
        if self._has_image():
            rect = self._image_rect()
            scale = max(abs(self.transform().m11()), 0.00001)
            margin = max(rect.width(), rect.height(), self.width() / scale, self.height() / scale) * 3
            self.setSceneRect(rect.adjusted(-margin, -margin, margin, margin))

    def fit_image(self):
        self.cancel_pending_annotation()
        self._user_view_adjusted = False
        if not self._has_image():
            return
        self.resetTransform()
        self._expand_view_scene()
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self._fit_scale = max(abs(self.transform().m11()), 0.00001)
        self.centerOn(self._image_rect().center())

    def _fit_in_view(self):
        if not self._user_view_adjusted:
            self.fit_image()

    def resizeEvent(self, event):
        # The original canvas always refits on resize; preserve an inspected detail.
        QGraphicsView.resizeEvent(self, event)
        if not self._user_view_adjusted:
            self._fit_in_view()
        elif self.has_pending_annotation():
            self._position_label_editor()

    def load_image(self, path, image=None):
        self.cancel_pending_annotation()
        self._user_view_adjusted = False
        self.set_annotations([])
        result = super().load_image(path, image=image)
        if result:
            self.fit_image()
        return result

    def reset_view(self):
        self.set_annotations([])
        self._user_view_adjusted = False
        super().reset_view()

    def wheelEvent(self, event):
        if not self._has_image():
            event.accept()
            return
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if not delta:
            event.accept()
            return
        pos = event.position().toPoint()
        before = self.mapToScene(pos)
        current = max(abs(self.transform().m11()), 0.00001)
        target = current * (1.25 ** (delta / 120.0))
        target = min(max(target, self._fit_scale * 0.1), max(64.0, self._fit_scale * 20))
        self.scale(target / current, target / current)
        self._user_view_adjusted = True
        self._expand_view_scene()
        after = self.mapToScene(pos)
        center = self.mapToScene(self.viewport().rect().center())
        self.centerOn(center + before - after)
        if self.has_pending_annotation():
            self._position_label_editor()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton and self._has_image():
            self._pan_pos = event.position().toPoint()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._annotation_mode and self._has_image():
            point = self.mapToScene(event.position().toPoint())
            if not self._image_rect().contains(point):
                event.accept()
                return
            if self.has_pending_annotation():
                if self._label_editor.text().strip():
                    if not self._commit_pending_annotation():
                        event.accept()
                        return
                else:
                    self.cancel_pending_annotation()
            self.setFocus(Qt.MouseFocusReason)
            self._box_start = self._clamp_point(point)
            self._box_press_pos = event.position().toPoint()
            self._draft_item = QGraphicsRectItem(QRectF(self._box_start, self._box_start))
            pen = QPen(QColor("#49baff"), 2, Qt.DashLine)
            pen.setCosmetic(True)
            self._draft_item.setPen(pen)
            self._draft_item.setBrush(QColor(73, 186, 255, 28))
            self._draft_item.setZValue(20)
            self._scene.addItem(self._draft_item)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_pos is not None:
            pos = event.position().toPoint()
            delta = self.mapToScene(self._pan_pos) - self.mapToScene(pos)
            center = self.mapToScene(self.viewport().rect().center())
            self.centerOn(center + delta)
            self._pan_pos = pos
            self._user_view_adjusted = True
            if self.has_pending_annotation():
                self._position_label_editor()
            event.accept()
            return
        if self._box_start is not None and self._draft_item is not None:
            point = self._clamp_point(self.mapToScene(event.position().toPoint()))
            self._draft_item.setRect(QRectF(self._box_start, point).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self._pan_pos is not None:
            self._pan_pos = None
            self.viewport().setCursor(Qt.CrossCursor if self._annotation_mode else Qt.ArrowCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._box_start is not None:
            end = self._clamp_point(self.mapToScene(event.position().toPoint()))
            rect = QRectF(self._box_start, end).normalized()
            distance = (event.position().toPoint() - self._box_press_pos).manhattanLength()
            self._box_start = None
            self._box_press_pos = None
            if distance < QApplication.startDragDistance() or rect.width() < 1 or rect.height() < 1:
                self._remove_draft()
                self._selected_annotation = self._annotation_at(end)
                self._render_annotations()
            else:
                bounds = self._image_rect()
                left, top = max(0, math.floor(rect.left())), max(0, math.floor(rect.top()))
                right = min(int(bounds.right()), math.ceil(rect.right()))
                bottom = min(int(bounds.bottom()), math.ceil(rect.bottom()))
                self._pending_rect = QRectF(left, top, right - left, bottom - top)
                self._draft_item.setRect(self._pending_rect)
                self._open_label_editor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self._annotation_mode:
            self.cancel_pending_annotation()
            index = self._annotation_at(self.mapToScene(event.position().toPoint()))
            if index is not None:
                self._selected_annotation = index
                self._rename_index = index
                self._render_annotations()
                self._open_label_editor(self._annotations[index]["label"])
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _annotation_at(self, point):
        for index in range(len(self._annotations) - 1, -1, -1):
            if QRectF(*self._annotations[index]["bbox"]).contains(point):
                return index
        return None

    def _remove_draft(self):
        if self._draft_item is not None:
            self._scene.removeItem(self._draft_item)
            self._draft_item = None

    def cancel_pending_annotation(self):
        self._pending_rect = None
        self._rename_index = None
        self._box_start = None
        self._box_press_pos = None
        self._remove_draft()
        if hasattr(self, "_label_editor"):
            self._label_editor.hide()
            self._label_editor.clear()

    def _open_label_editor(self, text=""):
        self._label_editor.setText(text)
        self._label_editor.setToolTip("输入缺陷名称或完整别名；Enter 确认，Esc 取消")
        self._label_editor.setStyleSheet(self._label_editor.styleSheet().replace("#ff7373", "#49baff"))
        self._position_label_editor()
        self._label_editor.show()
        self._label_editor.raise_()
        self._label_editor.setFocus(Qt.OtherFocusReason)
        self._label_editor.selectAll()

    def _position_label_editor(self):
        rect = self._pending_rect
        if rect is None and self._rename_index is not None:
            rect = QRectF(*self._annotations[self._rename_index]["bbox"])
        if rect is None:
            return
        pos = self.mapFromScene(rect.topLeft())
        width = min(290, max(100, self.viewport().width() - 16))
        height = max(36, self._label_editor.sizeHint().height())
        x = max(8, min(pos.x(), self.viewport().width() - width - 8))
        y = max(8, min(pos.y() - height - 3, self.viewport().height() - height - 8))
        self._label_editor.setGeometry(x, y, width, height)

    def _commit_pending_annotation(self):
        if not self.has_pending_annotation():
            return False
        value = self._label_editor.text().strip()
        if not value:
            self._label_editor.setFocus()
            return False
        try:
            label = self.resolve_label(value)
            if not isinstance(label, str) or not label.strip():
                raise ValueError("标签不能为空")
            label = label.strip()
        except (ValueError, TypeError) as error:
            self._label_editor.setToolTip(str(error))
            self._label_editor.setStyleSheet(self._label_editor.styleSheet().replace("#49baff", "#ff7373"))
            self._label_editor.setFocus()
            self._label_editor.selectAll()
            return False
        self._push_annotation_undo()
        if self._rename_index is not None:
            self._annotations[self._rename_index]["label"] = label
            selected = self._rename_index
        else:
            rect = self._pending_rect
            self._annotations.append({
                "label": label,
                "bbox": [int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())],
            })
            selected = len(self._annotations) - 1
        self.cancel_pending_annotation()
        self._selected_annotation = selected
        self._render_annotations()
        self.new_label_committed(label)
        self.annotations_changed(self.get_annotations())
        self.setFocus(Qt.OtherFocusReason)
        return True

    def _push_annotation_undo(self):
        self._annotation_undo_stack.append(self.get_annotations())
        del self._annotation_undo_stack[:-100]
        self._annotation_redo_stack.clear()

    def annotation_undo(self):
        if self.has_pending_annotation() or self._box_start is not None:
            self.cancel_pending_annotation()
            return
        if self._annotation_undo_stack:
            self._annotation_redo_stack.append(self.get_annotations())
            self._annotations = self._annotation_undo_stack.pop()
            self._selected_annotation = None
            self._render_annotations()
            self.annotations_changed(self.get_annotations())

    def annotation_redo(self):
        self.cancel_pending_annotation()
        if self._annotation_redo_stack:
            self._annotation_undo_stack.append(self.get_annotations())
            self._annotations = self._annotation_redo_stack.pop()
            self._selected_annotation = None
            self._render_annotations()
            self.annotations_changed(self.get_annotations())

    def delete_selected_annotation(self):
        if self.has_pending_annotation():
            self.cancel_pending_annotation()
            return
        if self._selected_annotation is None:
            return
        self._push_annotation_undo()
        del self._annotations[self._selected_annotation]
        self._selected_annotation = None
        self._render_annotations()
        self.annotations_changed(self.get_annotations())

    def keyPressEvent(self, event):
        if self._annotation_mode:
            if event.key() == Qt.Key_Escape:
                self.cancel_pending_annotation()
                self._selected_annotation = None
                self._render_annotations()
                event.accept()
                return
            if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                self.delete_selected_annotation()
                event.accept()
                return
        super().keyPressEvent(event)

    def eventFilter(self, watched, event):
        if watched is getattr(self, "_label_editor", None):
            if event.type() == QEvent.ShortcutOverride:
                # Text entry owns its edit keys even when the main window has
                # application actions with the same shortcuts.
                if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape, Qt.Key_Delete, Qt.Key_Backspace) or event.matches(QKeySequence.Undo) or event.matches(QKeySequence.Redo):
                    event.accept()
                    return True
            elif event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape:
                self.cancel_pending_annotation()
                self.setFocus(Qt.OtherFocusReason)
                return True
        return super().eventFilter(watched, event)

    def _render_annotations(self):
        for item in self._annotation_items:
            self._scene.removeItem(item)
        self._annotation_items.clear()
        for index, annotation in enumerate(self._annotations):
            color = QColor(annotation.get("color", ""))
            if not color.isValid():
                color = QColor(label_color(annotation["label"]))
            selected = index == self._selected_annotation
            item = QGraphicsRectItem(QRectF(*annotation["bbox"]))
            pen = QPen(QColor("#ffffff") if selected else color, 3 if selected else 2)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setBrush(QColor(color.red(), color.green(), color.blue(), 25 if selected else 8))
            item.setAcceptedMouseButtons(Qt.NoButton)
            item.setZValue(12 if selected else 10)
            item.setVisible(self._annotation_mode)
            self._scene.addItem(item)
            caption = f'{index + 1} · {annotation["label"]}'
            rect = item.rect()
            label = _BoxLabel(caption, color, rect.top() * self._fit_scale)
            label.setPos(rect.topLeft() if label.above else rect.bottomLeft())
            label.setZValue(13 if selected else 11)
            label.setVisible(self._annotation_mode)
            self._scene.addItem(label)
            self._annotation_items.extend((item, label))
        self.selection_changed(self._selected_annotation)
