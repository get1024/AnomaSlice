"""Draw annotation exports in original image coordinates with Qt."""

import errno
import os
import shutil
import sys
import tempfile
from itertools import count
from pathlib import Path

from PySide6.QtCore import QFile, QIODevice, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QImageWriter, QPainter, QPen

from image_dataset import EXPORT_PNG_KEY
from xml_annotations import label_color


def iter_visual_exports(image, gt_boxes, prediction_boxes, *, crops, layers, selected=None):
    """Yield (layer, zero-based index, image) or (None, None, full image).

    Selected boxes determine crop boundaries independently of the layers drawn.
    With no explicit selection, crop each visible box. Fractional boundaries
    round outwards to include every touched source pixel.
    """
    if image.isNull():
        raise ValueError("无法导出空图像。")
    if not layers or set(layers) - {"gt", "prediction"}:
        raise ValueError("请选择要导出的标注图层。")
    bounds = QRectF(image.rect())
    overlays = []
    regions = []
    for layer, annotations in (("gt", gt_boxes), ("prediction", prediction_boxes)):
        draw_layer = layer in layers
        if not draw_layer and (not crops or selected is None):
            continue
        for index, annotation in enumerate(annotations or ()):
            crop_here = crops and (draw_layer if selected is None else (layer, index) in selected)
            if not draw_layer and not crop_here:
                continue
            rect = QRectF(*annotation["bbox"]).intersected(bounds)
            if rect.isEmpty():
                continue
            if draw_layer:
                overlays.append((layer, index, rect, annotation))
            if crop_here:
                regions.append((layer, index, rect.toAlignedRect()))

    if not crops:
        yield None, None, _render(image, image.rect(), overlays)
    for layer, index, region in regions:
        yield layer, index, _render(image, region, overlays)


def _render(image, region, boxes):
    output = image.copy(region)
    if output.isNull():
        raise MemoryError("图像过大，无法创建导出图像。")
    output.setDevicePixelRatio(1)
    visible = [(layer, index, rect.translated(-region.x(), -region.y()), annotation)
               for layer, index, rect, annotation in boxes if rect.intersects(QRectF(region))]
    if not visible:
        return output

    # QPainter cannot target indexed images. Preserve 16-bit source channels
    # when exporting grayscale and high-depth colour images.
    high_depth = image.depth() > 32 or image.format() == QImage.Format_Grayscale16
    output = output.convertToFormat(QImage.Format_RGBA64 if high_depth else QImage.Format_ARGB32)
    painter = QPainter(output)
    if not painter.isActive():
        raise RuntimeError("无法绘制导出图像。")
    try:
        bounds = QRectF(output.rect())
        font = QFont("PingFang SC" if sys.platform == "darwin" else "Microsoft YaHei UI")
        short_side = min(output.width(), output.height())
        regular_size = max(12, min(40, round(short_side / 65)))
        # Small crops still need room for the defect name. Keep normal labels
        # at a readable size and shrink only when constrained by the output.
        font.setPixelSize(max(6, min(regular_size, int(output.height() * .57),
                                     int(output.width() * .38))))
        painter.setFont(font)
        metrics = QFontMetricsF(font, output)
        stroke = max(1, round(font.pixelSize() / 7))
        captions = []
        for layer, index, rect, annotation in visible:
            if layer == "prediction":
                color = QColor("#00d5e8")
                style = Qt.DashLine
                source = "对照"
            else:
                color = QColor(annotation.get("color") or "")
                if not color.isValid():
                    color = QColor(label_color(annotation["label"]))
                style = Qt.SolidLine
                source = "可编辑"
            pen_width = min(stroke, rect.width(), rect.height())
            pen = QPen(color, pen_width, style)
            painter.setPen(pen)
            inset = pen_width / 2
            if rect.width() <= pen_width or rect.height() <= pen_width:
                painter.fillRect(rect, color)
            else:
                painter.drawRect(rect.adjusted(inset, inset, -inset, -inset))
            name = annotation["label"]
            caption = f'{name} · {source} {index + 1}'
            captions.append((rect.intersected(bounds), name, caption, color, layer == "prediction"))
        for rect, name, caption, color, prediction in captions:
            _draw_caption(painter, bounds, metrics, rect, name, caption, color, prediction)
    finally:
        painter.end()
    return output


def _draw_caption(painter, bounds, metrics, box, name, caption, color, prediction):
    padding = 1 if metrics.height() < 12 else 3
    height = metrics.height() + 2 * padding
    available = max(0, bounds.width() - 2 * padding)
    if metrics.horizontalAdvance(caption) > available:
        caption = metrics.elidedText(name, Qt.ElideRight, available)
    if bounds.height() < height or not caption:
        return
    width = min(bounds.width(), metrics.horizontalAdvance(caption) + 2 * padding)
    x = min(box.left(), bounds.width() - width)
    # Put GT above its box and predictions below. At an image or crop edge
    # the respective caption moves inside the same edge.
    if prediction:
        y = box.bottom() if box.bottom() + height <= bounds.height() else box.bottom() - height
    else:
        y = box.top() - height if box.top() >= height else box.top()
    y = max(0, min(y, bounds.height() - height))
    badge = QRectF(x, y, width, height)
    painter.fillRect(badge, QColor(24, 28, 35, 225))
    painter.fillRect(QRectF(x, y, 2, height), color)
    painter.setPen(Qt.white)
    painter.drawText(badge.adjusted(padding, 0, -padding, 0), Qt.AlignVCenter | Qt.AlignLeft, caption)


def _macos_rename_exclusive(temporary, destination):
    import ctypes
    rename = ctypes.CDLL(None, use_errno=True).renamex_np
    rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
    if rename(os.fsencode(temporary), os.fsencode(destination), 0x00000004):  # RENAME_EXCL
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), str(destination))


def _publish_new_file(temporary, destination):
    """Publish without ever opening an existing destination for writing."""
    if sys.platform == "win32":
        os.rename(temporary, destination)  # MoveFileEx without REPLACE_EXISTING.
        return
    unsupported = {errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL}
    if sys.platform == "darwin":
        try:
            _macos_rename_exclusive(temporary, destination)
            return
        except OSError as error:
            if error.errno not in unsupported:
                raise
    try:
        os.link(temporary, destination)
        return
    except OSError as error:
        if error.errno not in unsupported and error.errno != errno.EPERM:
            raise

    # Some removable volumes and network shares support neither operation.
    # Acquire the name exclusively and write only to the descriptor we created.
    created = None
    try:
        with open(temporary, "rb") as source, destination.open("xb") as output:
            created = os.fstat(output.fileno())
            shutil.copyfileobj(source, output)
    except BaseException:
        if created is not None and destination.exists() and os.path.samestat(created, destination.lstat()):
            destination.unlink(missing_ok=True)
        raise


def save_png(image, path, *, occupied=None):
    """Publish a complete PNG under the first available name, without replacement."""
    path = Path(path)
    if occupied is None:
        occupied = {p.name.casefold() for p in path.parent.iterdir()}
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".anomaslice-write-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            device = QFile()
            if not device.open(stream.fileno(), QIODevice.WriteOnly):
                raise OSError(f"{path.name}：{device.errorString()}")
            try:
                writer = QImageWriter(device, b"png")
                writer.setText(EXPORT_PNG_KEY, "1")
                if not writer.write(image):
                    raise OSError(f"{path.name}：{writer.errorString()}")
                if not device.flush():
                    raise OSError(f"{path.name}：{device.errorString()}")
            finally:
                device.close()
        for number in count(1):
            candidate = path if number == 1 else path.with_name(f"{path.stem} ({number}){path.suffix}")
            key = candidate.name.casefold()
            if key in occupied:
                continue
            try:
                _publish_new_file(temporary, candidate)
            except FileExistsError:
                occupied.add(key)
                continue
            occupied.add(key)
            return candidate
    finally:
        Path(temporary).unlink(missing_ok=True)
