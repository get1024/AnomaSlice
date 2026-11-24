"""Open an annotation dataset and optionally attach GT and prediction XML sources."""

import os
from pathlib import Path

from PySide6.QtGui import QImageReader
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QGridLayout, QLabel, QLineEdit,
    QMenu, QMessageBox, QToolButton, QVBoxLayout,
)

from xml_annotations import image_extensions


class ComparisonDialog(QDialog):
    def __init__(self, source=None, parent=None, *, gt_source=None, prediction_source=None,
                 active_dataset=False, binding_image=None):
        super().__init__(parent)
        self.setWindowTitle("打开标注数据")
        self.image_path = self.gt_source = self.prediction_source = None
        layout = QVBoxLayout(self)
        grid = QGridLayout()
        layout.addLayout(grid)
        self.image_edit = self._add_row(grid, 0, "图片或文件夹", source,
                                        "图片文件夹、单张图片或对应的 XML")
        self.gt_edit = self._add_row(grid, 1, "GT（可编辑）", gt_source,
                                     "自动匹配 · 也可指定 GT 的 XML 文件夹或文件")
        self.prediction_edit = self._add_row(grid, 2, "模型预测（只读）", prediction_source,
                                             "可选 · 指定预测 XML 文件夹或文件")
        grid.setColumnStretch(1, 1)
        self._xml_owners = {edit: self._source_key() for edit in (self.gt_edit, self.prediction_edit)}
        self._current_source = self._source_key() if active_dataset else None
        self._binding_image = Path(binding_image) if binding_image else None
        for edit in self._xml_owners:
            edit.textChanged.connect(lambda _, field=edit: self._remember_source(field))
        self.image_edit.editingFinished.connect(self._clear_stale_sources)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self._update_hint()
        layout.addWidget(self.hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("打开")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(650, self.sizeHint().height())

    def _add_row(self, grid, row, label, value, placeholder):
        edit = QLineEdit(str(value) if value else "")
        edit.setPlaceholderText(placeholder)
        edit.setClearButtonEnabled(True)
        caption = QLabel(label)
        caption.setBuddy(edit)
        grid.addWidget(caption, row, 0)
        grid.addWidget(edit, row, 1)
        button = QToolButton()
        button.setText("选文件夹…")
        button.setAccessibleName(f"选择{label}来源")
        button.setPopupMode(QToolButton.MenuButtonPopup)
        button.clicked.connect(lambda: self._choose_folder(edit, label))
        menu = QMenu(button)
        file_action = menu.addAction("选图片或 XML 文件…" if row == 0 else "选 XML 文件…")
        file_action.triggered.connect(lambda: self._choose_file(edit, label))
        button.setMenu(menu)
        grid.addWidget(button, row, 2)
        return edit

    def _source_key(self):
        text = self.image_edit.text().strip()
        return os.path.normcase(os.path.abspath(os.path.expanduser(text))) if text else None

    def _remember_source(self, field):
        self._xml_owners[field] = self._source_key()

    def _update_hint(self):
        current = self._current_source and self._source_key() == self._current_source
        intro = ("可直接更改当前图片列表的 XML 来源。" if current
                 else "只选图片或文件夹即可开始标注。")
        self.hint.setText(intro + "对照模型结果时，通常只需再选预测文件夹。")
        if current and self._binding_image:
            binding = f"单个 XML 绑定到 {self._binding_image.name}。"
        else:
            binding = "打开新文件夹时，单个 XML 绑定到唯一同名图片。异名 XML 请从对应图片打开。"
        source_help = ("选目录时按每张图的文件名匹配。指定一项 XML 来源后，"
                       "另一项会在图片旁及其 xml 子目录中查找。")
        for widget in (self.hint, self.gt_edit, self.prediction_edit):
            widget.setToolTip(binding + source_help)

    def _clear_stale_sources(self):
        current = self._source_key()
        for edit, previous in self._xml_owners.items():
            if previous and previous != current:
                edit.clear()
            self._xml_owners[edit] = current
        self._update_hint()

    def _start_directory(self, edit):
        text = edit.text().strip() or self.image_edit.text().strip()
        if not text:
            return ""
        path = Path(text).expanduser()
        return str(path if path.is_dir() else path.parent)

    def _choose_file(self, edit, label):
        file_filter = "XML 文件 (*.xml)"
        title = f"选择{label} XML 文件"
        if edit is self.image_edit:
            patterns = " ".join(f"*.{ext}" for ext in sorted(image_extensions()))
            file_filter = f"图片或 XML ({patterns} *.xml);;所有文件 (*)"
            title = "选择图片或同名 XML"
        path, _ = QFileDialog.getOpenFileName(self, title, self._start_directory(edit), file_filter)
        if path:
            edit.setText(path)
            if edit is self.image_edit:
                self._clear_stale_sources()

    def _choose_folder(self, edit, label):
        title = "选择图片文件夹（含子目录）" if edit is self.image_edit else f"选择{label} XML 文件夹"
        path = QFileDialog.getExistingDirectory(self, title, self._start_directory(edit))
        if path:
            edit.setText(path)
            if edit is self.image_edit:
                self._clear_stale_sources()

    @staticmethod
    def _source(edit, label):
        text = edit.text().strip()
        if not text:
            return None
        path = Path(text).expanduser().resolve()
        if path.is_dir() or (path.is_file() and path.suffix.lower() == ".xml"):
            return path
        raise ValueError(f"请选择已有的{label} XML 文件或文件夹。")

    def accept(self):
        try:
            text = self.image_edit.text().strip()
            if not text:
                raise ValueError("请选择图片、图片文件夹或同名 XML。")
            image = Path(text).expanduser().resolve()
            if not image.is_dir() and not (image.is_file() and (
                    image.suffix.lower() == ".xml" or QImageReader(str(image)).canRead())):
                raise ValueError("请选择已有的图片、图片文件夹或同名 XML。")
            self._clear_stale_sources()
            gt = self._source(self.gt_edit, " GT")
            prediction = self._source(self.prediction_edit, "模型预测")
            if gt and prediction and gt.samefile(prediction):
                raise ValueError("GT 与预测使用了同一来源，请分别指定各自的 XML 文件或文件夹。")
        except (OSError, RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "无法打开标注数据", str(error))
            return
        self.image_path = image
        self.gt_source = gt
        self.prediction_source = prediction
        super().accept()
