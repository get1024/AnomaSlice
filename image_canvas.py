"""Native Qt image display and grid-mask editing, independent of application state.

Masks are grayscale QImages. Qt supplies image decoding, painting, alpha
composition and copy-on-write snapshots; no vision or numerical runtime is used.
"""

from dataclasses import dataclass

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


@dataclass(frozen=True)
class ReviewGrid:
    """Stride-sized review cells in row-major order, including partial edge cells."""

    width: int
    height: int
    stride: int

    @property
    def columns(self):
        return (self.width + self.stride - 1) // self.stride

    @property
    def rows(self):
        return (self.height + self.stride - 1) // self.stride

    def __len__(self):
        return self.rows * self.columns

    def cell(self, index):
        row, column = divmod(index, self.columns)
        x, y = column * self.stride, row * self.stride
        return QRect(x, y, min(self.stride, self.width - x), min(self.stride, self.height - y))

    def index_at(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return (int(y) // self.stride) * self.columns + int(x) // self.stride
        return None


class GridItem(QGraphicsItem):
    """Draw grid lines clipped to the exposed region, including partial edge cells."""

    def __init__(self):
        super().__init__()
        self.grid = ReviewGrid(0, 0, 128)
        self.setFlag(QGraphicsItem.ItemUsesExtendedStyleOption)
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

    def set_grid(self, grid):
        self.prepareGeometryChange()
        self.grid = grid
        self.update()

    def boundingRect(self):
        return QRectF(0, 0, self.grid.width, self.grid.height)

    def paint(self, painter, option, widget=None):
        bounds = self.boundingRect()
        exposed = option.exposedRect.intersected(bounds).toAlignedRect()
        if exposed.isEmpty():
            return
        step = self.grid.stride

        def draw_grid():
            painter.drawRect(bounds)
            for x in range(((exposed.left() + step - 1) // step) * step, exposed.right() + 2, step):
                painter.drawLine(x, exposed.top(), x, exposed.bottom() + 1)
            for y in range(((exposed.top() + step - 1) // step) * step, exposed.bottom() + 2, step):
                painter.drawLine(exposed.left(), y, exposed.right() + 1, y)

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
        self._grid_item = GridItem()
        self._scene.addItem(self._grid_item)
        self._mask_item = QGraphicsPixmapItem()
        self._mask_item.setOpacity(0.5)
        self._scene.addItem(self._mask_item)
        self.setBackgroundBrush(Qt.black)
        self._mask = None
        self._saved_mask = None
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
        self._grid_item.set_grid(ReviewGrid(pixmap.width(), pixmap.height(), self.review_grid.stride))
        self._mask = None
        self._saved_mask = None
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
            self.mark_mask_saved()
            self._update_mask_item()

    def set_stride(self, stride):
        self._grid_item.set_grid(ReviewGrid(self.review_grid.width, self.review_grid.height, max(1, stride)))

    def set_smart_grid(self, enabled):
        self._grid_item.set_smart_grid(enabled)

    def set_highlight_rect(self, rect):
        self._grid_item.set_highlight_rect(rect)

    @property
    def review_grid(self):
        return self._grid_item.grid

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
        self._saved_mask = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_mask_item()
        self._grid_item.set_grid(ReviewGrid(0, 0, self.review_grid.stride))
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

    def mark_mask_saved(self):
        self._saved_mask = self.get_mask()

    def has_mask_changes(self):
        return self._mask != self._saved_mask

    def set_mask(self, mask):
        if mask is None:
            return
        self._mask = mask.convertToFormat(QImage.Format_Grayscale8)
        self.mark_mask_saved()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_mask_item()
