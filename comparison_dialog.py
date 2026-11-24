"""Choose an image and assign editable GT and read-only prediction XML sources."""

from pathlib import Path

from PySide6.QtGui import QImageReader
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QGridLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QVBoxLayout,
)

from xml_annotations import image_extensions


class ComparisonDialog(QDialog):
    def __init__(self, image=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("打开 GT 与预测对照")
        self.image_path = self.gt_source = self.prediction_source = None
        layout = QVBoxLayout(self)
        grid = QGridLayout()
        layout.addLayout(grid)
        self.image_edit = QLineEdit(str(image) if image else "")
        self.image_edit.setPlaceholderText("选择要对照的图片")
        grid.addWidget(QLabel("图片"), 0, 0)
        grid.addWidget(self.image_edit, 0, 1)
        image_button = QPushButton("选择图片…")
        image_button.clicked.connect(self._choose_image)
        grid.addWidget(image_button, 0, 2, 1, 2)

        self.gt_edit = self._add_source_row(grid, 1, "GT（可编辑）")
        self.prediction_edit = self._add_source_row(grid, 2, "模型预测（只读）")
        grid.setColumnStretch(1, 1)
        hint = QLabel("选择文件夹时按图片名匹配 XML。可将其中一项留空，"
                      "在图片所在目录及其 xml 子目录中查找同名 XML。")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("打开对照")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(650, self.sizeHint().height())

    def _add_source_row(self, grid, row, label):
        edit = QLineEdit()
        edit.setPlaceholderText("XML 文件或所在文件夹")
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(edit, row, 1)
        file_button = QPushButton("选文件…")
        file_button.clicked.connect(lambda: self._choose_xml(edit, label))
        grid.addWidget(file_button, row, 2)
        folder_button = QPushButton("选文件夹…")
        folder_button.clicked.connect(lambda: self._choose_folder(edit, label))
        grid.addWidget(folder_button, row, 3)
        return edit

    def _choose_image(self):
        patterns = " ".join(f"*.{ext}" for ext in sorted(image_extensions()))
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", self.image_edit.text(),
                                              f"图片 ({patterns});;所有文件 (*)")
        if path:
            self.image_edit.setText(path)

    def _start_directory(self, edit):
        text = edit.text().strip() or self.image_edit.text().strip()
        if not text:
            return ""
        path = Path(text).expanduser()
        return str(path if path.is_dir() else path.parent)

    def _choose_xml(self, edit, label):
        path, _ = QFileDialog.getOpenFileName(self, f"选择{label} XML",
                                              self._start_directory(edit), "XML 文件 (*.xml)")
        if path:
            edit.setText(path)

    def _choose_folder(self, edit, label):
        path = QFileDialog.getExistingDirectory(self, f"选择{label} XML 文件夹",
                                                self._start_directory(edit))
        if path:
            edit.setText(path)

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
            image = Path(self.image_edit.text().strip()).expanduser().resolve()
            if not image.is_file() or not QImageReader(str(image)).canRead():
                raise ValueError("请选择支持的图片文件。")
            gt = self._source(self.gt_edit, " GT")
            prediction = self._source(self.prediction_edit, "模型预测")
            if gt is None and prediction is None:
                raise ValueError("请至少指定 GT 或模型预测的一项 XML 来源。")
            if gt and prediction and gt.samefile(prediction):
                raise ValueError("GT 与预测使用了同一来源，请分别指定各自的 XML 文件或文件夹。")
        except (OSError, RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "无法打开对照", str(error))
            return
        self.image_path = image
        self.gt_source = gt
        self.prediction_source = prediction
        super().accept()
