"""Small, explicit class/alias editor. No model or service dependency."""

import re

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel,
    QLayout, QLineEdit, QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from annotation_store import LabelCatalog


class LabelDialog(QDialog):
    def __init__(self, catalog, parent=None):
        super().__init__(parent)
        self.setWindowTitle("缺陷种类与别名")
        self.resize(520, 360)
        self.catalog = catalog
        layout = QVBoxLayout(self)
        hint = QLabel("类别表适用于所有图片。每行填写一种缺陷，多个别名用逗号分隔。\n"
                      "标注时输入完整别名即可使用对应的类别，区分大小写。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        contents = QWidget()
        self.fields = QFormLayout(contents)
        self.fields.setSizeConstraint(QLayout.SetMinAndMaxSize)
        self.fields.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        header = QHBoxLayout()
        header.addWidget(QLabel("缺陷种类"), 1)
        header.addWidget(QLabel("别名（可留空）"), 2)
        self.fields.addRow(QLabel("选择"), header)
        self.scroll.setWidget(contents)
        layout.addWidget(self.scroll)
        self.rows = []
        self._populate(catalog)
        row = QHBoxLayout()
        add = QPushButton("添加种类")
        add.clicked.connect(self._add_empty_row)
        remove = QPushButton("移除所选种类")
        remove.clicked.connect(self._remove_rows)
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch()
        restore = QPushButton("恢复默认四类")
        restore.setToolTip("用模型默认的四种缺陷及其别名替换表格内容，点击保存后生效。")
        restore.clicked.connect(lambda: self._populate(LabelCatalog.defaults()))
        row.addWidget(restore)
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("保存")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate(self, catalog):
        while self.rows:
            self._remove_row(len(self.rows) - 1)
        for record in catalog.to_list():
            self._add_row(record["name"], ", ".join(record["aliases"]))

    def _add_row(self, name, aliases):
        selected = QCheckBox()
        selected.setAccessibleName("选择此种类")
        name_edit = QLineEdit(name)
        name_edit.setAccessibleName("缺陷种类")
        alias_edit = QLineEdit(aliases)
        alias_edit.setAccessibleName("别名")
        inputs = QHBoxLayout()
        inputs.addWidget(name_edit, 1)
        inputs.addWidget(alias_edit, 2)
        self.fields.addRow(selected, inputs)
        self.rows.append((selected, name_edit, alias_edit))
        return name_edit

    def _add_empty_row(self):
        editor = self._add_row("", "")
        editor.setFocus()
        # The scroll area knows the new row's geometry after the layout pass.
        QTimer.singleShot(0, editor, lambda: self.scroll.ensureWidgetVisible(editor))

    def _remove_row(self, row):
        selected, _, _ = self.rows.pop(row)
        self.fields.removeRow(selected)

    def _remove_rows(self):
        for row in reversed(range(len(self.rows))):
            if self.rows[row][0].isChecked():
                self._remove_row(row)

    def accept(self):
        QApplication.inputMethod().commit()
        records = []
        for _, name_edit, alias_edit in self.rows:
            name = name_edit.text().strip()
            aliases = alias_edit.text().strip()
            if not name and not aliases:
                continue
            records.append({"name": name, "aliases": [a.strip() for a in re.split(r"[,，]", aliases) if a.strip()]})
        try:
            self.catalog = LabelCatalog(records)
        except ValueError as error:
            QMessageBox.warning(self, "种类或别名冲突", str(error))
            return
        super().accept()
