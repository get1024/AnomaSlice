"""Browse classified patches for the current image and move them between output folders."""

from collections import deque
from pathlib import Path

from PySide6.QtCore import QFile, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QListWidget, QListWidgetItem, QMenu, QMessageBox, QTabWidget, QVBoxLayout, QWidget,
)

from image_canvas import ScalableImageLabel


def patch_coordinates(path, image_stem):
    """Recognize filenames produced by SlicerThread, including stems with underscores."""
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
        return None
    parts = path.stem.rsplit("_", 2)
    if len(parts) != 3 or parts[0] != image_stem:
        return None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None


class PatchGallery(QWidget):
    patch_selected = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(320)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs, 1)
        self._lists = {}
        for category in ("Normal", "Abnormal"):
            view = QListWidget()
            view.setViewMode(QListWidget.IconMode)
            view.setResizeMode(QListWidget.Adjust)
            view.setIconSize(QSize(64, 64))
            view.setContextMenuPolicy(Qt.CustomContextMenu)
            view.customContextMenuRequested.connect(lambda pos, key=category: self._menu(key, pos))
            view.itemClicked.connect(self._preview_item)
            tabs.addTab(view, category)
            self._lists[category] = view
        self._preview = ScalableImageLabel()
        self._preview.setStyleSheet("background-color: #222; border: 1px solid #444;")
        layout.addWidget(self._preview, 1)
        self._queue = deque()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._load_batch)
        self.reset()

    def reset(self, normal=(), abnormal=()):
        self._files = {"Normal": set(map(Path, normal)), "Abnormal": set(map(Path, abnormal))}
        self.show_image(None, None, None)

    def project_files(self):
        return tuple(sorted(map(str, self._files[key])) for key in ("Normal", "Abnormal"))

    def show_image(self, patch_stem, normal_dir, abnormal_dir):
        self._timer.stop()
        self._queue.clear()
        self._preview.setText("预览区域")
        self._stem = patch_stem
        self._dirs = {"Normal": normal_dir, "Abnormal": abnormal_dir}
        for key, view in self._lists.items():
            view.clear()
            if self._stem is None:
                continue
            directory = self._dirs[key]
            files = self._files[key]
            if directory:
                try:
                    files.update(p for p in Path(directory).iterdir()
                                 if patch_coordinates(p, self._stem) is not None and p.is_file())
                except (FileNotFoundError, NotADirectoryError):
                    pass
                except OSError as error:
                    view.setToolTip(str(error))
            for path in sorted(p for p in files if patch_coordinates(p, self._stem) is not None):
                if path.is_file():
                    self._queue.append((key, path))
                else:
                    files.discard(path)
        if self._queue:
            self._timer.start(0)

    def _load_batch(self):
        for _ in range(min(50, len(self._queue))):
            key, path = self._queue.popleft()
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                icon = QIcon(pixmap.scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                item = QListWidgetItem(icon, path.name)
                item.setToolTip(str(path))
                self._lists[key].addItem(item)
        if not self._queue:
            self._timer.stop()

    def _preview_item(self, item):
        path = Path(item.toolTip())
        pixmap = QPixmap(str(path))
        if not pixmap.isNull():
            self._preview.setPixmap(pixmap)
            coords = patch_coordinates(path, self._stem)
            if coords is not None:
                self.patch_selected.emit(*coords)

    def _menu(self, key, pos):
        view = self._lists[key]
        item = view.itemAt(pos)
        other = "Abnormal" if key == "Normal" else "Normal"
        if item is None or not self._dirs[other]:
            return
        menu = QMenu(self)
        action = menu.addAction(f"移动到 {other}")
        if menu.exec(view.viewport().mapToGlobal(pos)) != action:
            return
        source = Path(item.toolTip())
        destination = Path(self._dirs[other]) / source.name
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_file = QFile(str(source))
            if not source_file.rename(str(destination)):
                raise OSError(source_file.errorString())
        except OSError as error:
            QMessageBox.warning(self, "移动切图失败", str(error))
            return
        self._files[key].discard(source)
        self._files[other].add(destination)
        view.takeItem(view.row(item))
        item.setToolTip(str(destination))
        self._lists[other].addItem(item)
