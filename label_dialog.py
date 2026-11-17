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
        hint = QLabel("每行一种缺陷；多个别名用逗号分隔。\n仅完整输入匹配别名：1 → 划痕，待定1 保持原样。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["缺陷种类", "别名（可留空）"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        for record in catalog.to_list():
            self._add_row(record["name"], ", ".join(record["aliases"]))
        layout.addWidget(self.table)
        row = QHBoxLayout()
        add = QPushButton("添加种类")
        add.clicked.connect(lambda: self._add_row())
        remove = QPushButton("移除所选种类")
        remove.clicked.connect(self._remove_rows)
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch()
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("保存")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

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
