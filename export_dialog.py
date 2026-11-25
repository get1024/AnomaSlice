"""Choose annotated images or defect crops to export from the current dataset."""

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout,
)


@dataclass(frozen=True)
class ExportOptions:
    mode: str
    layers: tuple[str, ...]
    directory: Path
    selected: frozenset[tuple[str, int]] = frozenset()
    label: str = ""


class ExportDialog(QDialog):
    def __init__(self, parent=None, *, gt_boxes, prediction_boxes=None, labels=(), selected_gt=None):
        super().__init__(parent)
        self.setWindowTitle("导出标注图片")
        self.setMinimumWidth(490)
        self.options = None
        layout = QVBoxLayout(self)
        self.form = QFormLayout()
        layout.addLayout(self.form)

        self.mode = QComboBox()
        self.mode.addItem("导出当前图片", "current_image")
        self.mode.addItem("导出全部图片", "all_images")
        self.mode.addItem("导出指定缺陷裁剪图", "specified")
        self.mode.addItem("导出全部缺陷裁剪图", "all_crops")
        self.form.addRow("导出内容", self.mode)

        self.layers = QComboBox()
        self.layers.addItem("1 号 XML · 人工标注（GT）", ("gt",))
        if prediction_boxes is not None:
            self.layers.addItem("2 号 XML · 模型预测", ("prediction",))
            self.layers.addItem("两个 XML · GT 和预测", ("gt", "prediction"))
            self.layers.setCurrentIndex(2)
        self.form.addRow("绘制标注", self.layers)

        self.scope = QComboBox()
        self.scope.addItem("勾选当前图片中的缺陷", "selected_boxes")
        self.scope.addItem("按类别筛选整个数据集", "class_crops")
        self.form.addRow("选择方式", self.scope)
        self.boxes = QListWidget()
        self.form.addRow("裁剪范围", self.boxes)
        self.empty_hint = QLabel("当前图片没有可用于裁剪的缺陷框。")
        self.form.addRow(self.empty_hint)

        class_names = dict.fromkeys(labels)
        for layer, title, boxes in (("gt", "GT", gt_boxes), ("prediction", "预测", prediction_boxes or [])):
            for index, box in enumerate(boxes):
                name = box["label"]
                class_names[name] = None
                x, y, width, height = box["bbox"]
                item = QListWidgetItem(f"{title} #{index + 1}　{name}　({x}, {y})　{width} × {height}")
                item.setData(Qt.UserRole, (layer, index))
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                checked = layer == "gt" and index == selected_gt
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                self.boxes.addItem(item)

        self.label = QComboBox()
        self.label.setEditable(True)
        self.label.setInsertPolicy(QComboBox.NoInsert)
        self.label.addItems(list(class_names))
        self.label.lineEdit().setPlaceholderText("选择或输入完整类别名")
        self.form.addRow("缺陷类别", self.label)

        destination = QHBoxLayout()
        self.folder = QLineEdit()
        self.folder.setPlaceholderText("每次导出会在此目录下新建子目录")
        destination.addWidget(self.folder)
        browse = QPushButton("选择文件夹…")
        browse.clicked.connect(self._browse)
        destination.addWidget(browse)
        self.form.addRow("输出目录", destination)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("导出")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for combo in (self.mode, self.scope):
            combo.currentIndexChanged.connect(self._update_fields)
        self._update_fields()

    def _browse(self):
        chosen = QFileDialog.getExistingDirectory(self, "选择输出目录", self.folder.text().strip())
        if chosen:
            self.folder.setText(chosen)

    def _update_fields(self):
        specified = self.mode.currentData() == "specified"
        individual = specified and self.scope.currentData() == "selected_boxes"
        self.form.setRowVisible(self.scope, specified)
        self.form.setRowVisible(self.label, specified and not individual)
        self.form.setRowVisible(self.boxes, individual and self.boxes.count() > 0)
        self.form.setRowVisible(self.empty_hint, individual and self.boxes.count() == 0)

    def accept(self):
        mode = self.mode.currentData()
        if mode == "specified":
            mode = self.scope.currentData()
        selected, label = frozenset(), ""
        if mode == "selected_boxes":
            items = (self.boxes.item(row) for row in range(self.boxes.count()))
            selected = frozenset(tuple(item.data(Qt.UserRole)) for item in items
                                 if item.checkState() == Qt.Checked)
            if not selected:
                QMessageBox.warning(self, "无法导出", "请勾选至少一个缺陷框。")
                return
        if mode == "class_crops":
            label = self.label.currentText().strip()
            if not label:
                QMessageBox.warning(self, "无法导出", "请选择或输入要裁剪的缺陷类别。")
                return
        text = self.folder.text().strip()
        directory = Path(text).expanduser()
        if not text or not directory.is_dir():
            QMessageBox.warning(self, "无法导出", "请选择已有的输出目录。")
            return
        self.options = ExportOptions(mode, tuple(self.layers.currentData()), directory.resolve(), selected, label)
        super().accept()
