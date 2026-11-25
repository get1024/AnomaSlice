"""Small, explicit class/alias editor. No model or service dependency."""

import re

from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView,
    QLabel, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
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
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["缺陷种类", "别名（可留空）"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._populate_table(catalog)
        layout.addWidget(self.table)
        row = QHBoxLayout()
        add = QPushButton("添加种类")
        add.clicked.connect(lambda: self._add_row())
        remove = QPushButton("移除所选种类")
        remove.clicked.connect(self._remove_rows)
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch()
        restore = QPushButton("恢复默认四类")
        restore.setToolTip("用模型默认的四种缺陷及其别名替换表格内容，点击保存后生效。")
        restore.clicked.connect(lambda: self._populate_table(LabelCatalog.defaults()))
        row.addWidget(restore)
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("保存")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate_table(self, catalog):
        self.table.setRowCount(0)
        for record in catalog.to_list():
            self._add_row(record["name"], ", ".join(record["aliases"]))
        self.table.clearSelection()

    def _add_row(self, name="", aliases=""):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(name))
        self.table.setItem(row, 1, QTableWidgetItem(aliases))
        self.table.setCurrentCell(row, 0)

    def _remove_rows(self):
        for row in sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(row)

    def accept(self):
        # The Save button can run before the delegate receives focus-out.
        editor = self.table.indexWidget(self.table.currentIndex())
        if editor is not None and self.table.state() == QAbstractItemView.EditingState:
            self.table.commitData(editor)
        records = []
        for row in range(self.table.rowCount()):
            name_item, aliases_item = self.table.item(row, 0), self.table.item(row, 1)
            name = name_item.text().strip() if name_item else ""
            aliases = aliases_item.text().strip() if aliases_item else ""
            if not name and not aliases:
                continue
            records.append({"name": name, "aliases": [a.strip() for a in re.split(r"[,，]", aliases) if a.strip()]})
        try:
            self.catalog = LabelCatalog(records)
        except ValueError as error:
            QMessageBox.warning(self, "种类或别名冲突", str(error))
            return
        super().accept()
