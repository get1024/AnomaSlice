"""Native Qt image display and grid-mask editing, independent of application state.

Masks are grayscale QImages. Qt supplies image decoding, painting, alpha
composition and copy-on-write snapshots; no vision or numerical runtime is used.
"""

from PySide6.QtCore import QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
    QLabel, QSizePolicy,
)


class ScalableImageLabel(QLabel):
    """Fit a preview to its available space without changing its aspect ratio."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._original_pixmap = None

    def setPixmap(self, pixmap):
        self._original_pixmap = pixmap
        self.update()

    def setText(self, text):
        self._original_pixmap = None
        super().setText(text)

    def paintEvent(self, event):
        if self._original_pixmap is None or self._original_pixmap.isNull():
            super().paintEvent(event)
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        target = QRectF(self.rect())
        size = self._original_pixmap.size().scaled(target.size().toSize(), Qt.KeepAspectRatio)
        x = (target.width() - size.width()) / 2
        y = (target.height() - size.height()) / 2
        painter.drawPixmap(int(x), int(y), size.width(), size.height(), self._original_pixmap)


class GridItem(QGraphicsItem):
    """Paint only the visible portion of the slicing grid."""

    def __init__(self, rect, stride, patch_width, patch_height):
        super().__init__()
        self._rect = rect
        self._stride = stride
        self._patch_width = patch_width
        self._patch_height = patch_height
        self._pen = QPen(Qt.cyan, 1)
        self._highlight_rect = QRectF()
        self._highlight_pen = QPen(QColor(255, 0, 255), 5)
        self._smart_grid = False

    def set_smart_grid(self, enabled):
        self._smart_grid = bool(enabled)
        self.update()

    def set_highlight_rect(self, rect):
        self._highlight_rect = rect
        self.update()

    def set_rect(self, rect):
        self.prepareGeometryChange()
        self._rect = rect
        self.update()

    def set_stride(self, stride):
        self._stride = max(1, stride)
        self.update()

    def set_patch_size(self, width, height):
        self._patch_width = max(1, width)
        self._patch_height = max(1, height)
        self.update()

    def boundingRect(self):
        return self._rect

    def get_patch_counts(self):
        if self._rect.isNull() or self._stride <= 0:
            return 0, 0
        columns = max(0, (int(self._rect.width()) - self._patch_width) // self._stride + 1)
        rows = max(0, (int(self._rect.height()) - self._patch_height) // self._stride + 1)
        return rows, columns

    def paint(self, painter, option, widget=None):
        if self._rect.isNull() or self._stride <= 0:
            return
        exposed = option.exposedRect
        left = int(max(self._rect.left(), exposed.left()))
        right = int(min(self._rect.right(), exposed.right()))
        top = int(max(self._rect.top(), exposed.top()))
        bottom = int(min(self._rect.bottom(), exposed.bottom()))
        start_x, start_y = int(self._rect.left()), int(self._rect.top())
        column = max(0, (left - self._patch_width - start_x) // self._stride)
        row = max(0, (top - self._patch_height - start_y) // self._stride)
        if start_x + column * self._stride + self._patch_width < left:
            column += 1
        if start_y + row * self._stride + self._patch_height < top:
            row += 1

        def draw_grid():
            y = start_y + row * self._stride
            while y + self._patch_height <= int(self._rect.bottom()) + 1 and y <= bottom:
                x = start_x + column * self._stride
                while x + self._patch_width <= int(self._rect.right()) + 1 and x <= right:
                    painter.drawRect(x, y, self._patch_width, self._patch_height)
                    x += self._stride
                y += self._stride

        if self._smart_grid:
            painter.setPen(QPen(Qt.black, 1))
            draw_grid()
            painter.setPen(QPen(Qt.white, 1, Qt.DashLine))
            draw_grid()
        else:
            painter.setPen(self._pen)
            draw_grid()
        if not self._highlight_rect.isNull():
            painter.setPen(self._highlight_pen)
            painter.drawRect(self._highlight_rect)


class SliceCanvas(QGraphicsView):
    """Image and optional slice mask; bbox interaction is composed separately."""

    canvas_clicked = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._grid_item = GridItem(QRectF(), 128, 256, 256)
        self._scene.addItem(self._grid_item)
        self._mask_item = QGraphicsPixmapItem()
        self._mask_item.setOpacity(0.5)
        self._scene.addItem(self._mask_item)
        self.setBackgroundBrush(Qt.black)
        self._mask = None
        self._undo_stack = []
        self._redo_stack = []

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            self.canvas_clicked.emit(point.x(), point.y())
        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_in_view()

    def _fit_in_view(self):
        if not self._pixmap_item.pixmap().isNull():
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            self.scale(1.25 if event.angleDelta().y() > 0 else 0.8,
                       1.25 if event.angleDelta().y() > 0 else 0.8)
            event.accept()
        else:
            super().wheelEvent(event)

    def load_image(self, path, image=None):
        """Display a decoded QImage when supplied, otherwise decode the path."""
        pixmap = QPixmap.fromImage(image) if image is not None else QPixmap(path)
        if pixmap.isNull():
            return False
        self._pixmap_item.setPixmap(pixmap)
        rect = QRectF(pixmap.rect())
        self._scene.setSceneRect(rect)
        self._grid_item.set_rect(rect)
        self._mask = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_mask_item()
        self._fit_in_view()
        return True

    def ensure_slice_mask(self):
        pixmap = self._pixmap_item.pixmap()
        if self._mask is None and not pixmap.isNull():
            self._mask = QImage(pixmap.width(), pixmap.height(), QImage.Format_Grayscale8)
            self._mask.fill(0)
            self._update_mask_item()

    def set_stride(self, stride):
        self._grid_item.set_stride(stride)

    def set_smart_grid(self, enabled):
        self._grid_item.set_smart_grid(enabled)

    def set_patch_size(self, width, height):
        self._grid_item.set_patch_size(width, height)

    def set_highlight_rect(self, rect):
        self._grid_item.set_highlight_rect(rect)

    def get_patch_counts(self):
        return self._grid_item.get_patch_counts()

    def set_mask_opacity(self, opacity):
        self._mask_item.setOpacity(opacity)

    def clear_mask(self):
        if self._mask is None:
            return
        self.push_undo()
        self._mask.fill(0)
        self._update_mask_item()

    def reset_view(self):
        self._pixmap_item.setPixmap(QPixmap())
        self._mask = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_mask_item()
        self._grid_item.set_rect(QRectF())
        self.scene().setSceneRect(QRectF())

    def push_undo(self):
        if self._mask is None:
            return
        self._undo_stack.append(QImage(self._mask))
        del self._undo_stack[:-20]
        self._redo_stack.clear()

    def undo(self):
        if not self._undo_stack:
            return
        if self._mask is not None:
            self._redo_stack.append(QImage(self._mask))
        self._mask = self._undo_stack.pop()
        self._update_mask_item()

    def redo(self):
        if not self._redo_stack:
            return
        if self._mask is not None:
            self._undo_stack.append(QImage(self._mask))
        self._mask = self._redo_stack.pop()
        self._update_mask_item()

    def _update_mask_item(self):
        if self._mask is None:
            self._mask_item.setPixmap(QPixmap())
            return
        # QImage performs this alpha composition in native code. Grayscale
        # intensity becomes alpha, keeping normal units completely transparent.
        overlay = QImage(self._mask.size(), QImage.Format_ARGB32_Premultiplied)
        overlay.fill(QColor(255, 0, 0))
        overlay.setAlphaChannel(self._mask)
        self._mask_item.setPixmap(QPixmap.fromImage(overlay))

    def fill_rect(self, rect, value):
        if self._mask is None:
            return
        self.push_undo()
        x, y = int(rect.x()), int(rect.y())
        width, height = int(rect.width()), int(rect.height())
        value = max(0, min(255, int(value)))
        painter = QPainter(self._mask)
        # Retain the previous slicer's inclusive endpoint convention so existing
        # mask classification does not change during the dependency removal.
        painter.fillRect(QRect(x, y, width + 1, height + 1), QColor(value, value, value))
        painter.end()
        self._update_mask_item()

    def get_mask(self):
        return QImage(self._mask) if self._mask is not None else None

    def set_mask(self, mask):
        if mask is None:
            return
        self._mask = mask.convertToFormat(QImage.Format_Grayscale8)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_mask_item()
