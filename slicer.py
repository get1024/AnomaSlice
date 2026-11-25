"""Crop and classify user-annotated images using Qt's image codecs only."""

from __future__ import annotations

import os

from PySide6.QtCore import QRect, QThread, Signal
from PySide6.QtGui import QImage

from image_dataset import EXPORT_PNG_KEY


def _gray_mask(mask):
    if mask is None or mask.isNull():
        return None
    if mask.format() == QImage.Format_Grayscale8:
        return mask
    return mask.convertToFormat(QImage.Format_Grayscale8)


def _count_mask_region(data, row_stride, x, y, width, height):
    """Count nonzero bytes in C, without interpreting row-alignment padding."""
    total = 0
    for row in range(y, y + height):
        start = row * row_stride + x
        total += width - data[start : start + width].tobytes().count(0)
    return total


def count_mask_pixels(mask, rect=None):
    """Return marked pixels in a QImage mask (or a clipped source rectangle).

    QImage's grayscale rows are padded to a four-byte boundary. The padding is
    deliberately excluded, and counting each row uses the native bytes.count
    implementation instead of a Python loop over individual pixels.
    """
    gray = _gray_mask(mask)
    if gray is None:
        return 0
    region = gray.rect() if rect is None else QRect(rect).intersected(gray.rect())
    if region.isEmpty():
        return 0
    return _count_mask_region(
        gray.constBits(), gray.bytesPerLine(), region.x(), region.y(),
        region.width(), region.height(),
    )


class SlicerThread(QThread):
    """Write fixed-size crops based solely on the person's grayscale mask.

    ``finished_signal`` is emitted only after every requested image is saved.
    ``error_signal`` reports failures; QThread.finished still always lets the
    window clean up the worker. Image and mask snapshots use Qt implicit sharing.
    """

    progress_signal = Signal(int)
    finished_signal = Signal(list, list)
    error_signal = Signal(str)

    def __init__(
        self, image, mask, patch_w, patch_h, stride, threshold, normal_dir,
        abnormal_dir, save_mask, mask_dir, base_name,
    ):
        super().__init__()
        self._image = QImage(image)
        self._mask = QImage(mask) if mask is not None else None
        self._patch_w = patch_w
        self._patch_h = patch_h
        self._stride = stride
        self._threshold = threshold
        self._normal_dir = os.fspath(normal_dir)
        self._abnormal_dir = os.fspath(abnormal_dir)
        self._save_mask = save_mask
        self._mask_dir = os.fspath(mask_dir) if mask_dir else None
        self._base_name = base_name

    @staticmethod
    def _save_png(image, path):
        image.setText(EXPORT_PNG_KEY, "1")
        if not image.save(path, "PNG"):
            raise OSError(f"无法保存切图：{path}")

    def run(self):
        try:
            self._slice()
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            self.error_signal.emit(str(error))

    def _slice(self):
        if self._image.isNull():
            raise ValueError("切图失败：没有可读取的原始图片。")
        for value in (self._patch_w, self._patch_h, self._stride):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("切图尺寸和步长必须是大于零的整数。")
        if not 0 <= self._threshold <= 1:
            raise ValueError("切图阈值必须在 0 至 1 之间。")

        h, w = self._image.height(), self._image.width()
        mask = _gray_mask(self._mask)
        if mask is not None and mask.size() != self._image.size():
            raise ValueError("切图失败：标注掩膜的尺寸与原图不一致。")
        mask_data = mask.constBits() if mask is not None else None
        mask_stride = mask.bytesPerLine() if mask is not None else 0

        os.makedirs(self._normal_dir, exist_ok=True)
        os.makedirs(self._abnormal_dir, exist_ok=True)
        if self._save_mask and self._mask_dir:
            os.makedirs(self._mask_dir, exist_ok=True)

        cols = max(0, (w - self._patch_w) // self._stride + 1)
        rows = max(0, (h - self._patch_h) // self._stride + 1)
        total = max(1, cols * rows)
        patch_area = self._patch_w * self._patch_h
        normal_paths, abnormal_paths = [], []
        count = 0
        for y in range(0, h - self._patch_h + 1, self._stride):
            for x in range(0, w - self._patch_w + 1, self._stride):
                marked = 0 if mask_data is None else _count_mask_region(
                    mask_data, mask_stride, x, y, self._patch_w, self._patch_h,
                )
                is_abnormal = marked / patch_area > self._threshold
                filename = f"{self._base_name}_{x}_{y}.png"
                directory = self._abnormal_dir if is_abnormal else self._normal_dir
                path = os.path.join(directory, filename)
                self._save_png(self._image.copy(x, y, self._patch_w, self._patch_h), path)
                (abnormal_paths if is_abnormal else normal_paths).append(path)
                if is_abnormal and self._save_mask and self._mask_dir and mask is not None:
                    self._save_png(
                        mask.copy(x, y, self._patch_w, self._patch_h),
                        os.path.join(self._mask_dir, filename),
                    )
                count += 1
                self.progress_signal.emit(int(count / total * 100))
        if not count:
            self.progress_signal.emit(100)
        self.finished_signal.emit(normal_paths, abnormal_paths)
