import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QPushButton

from annotation_store import LabelCatalog
from label_dialog import LabelDialog


class LabelDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.records = [{"name": "脱落", "aliases": ["2"]}, {"name": "污渍", "aliases": ["3"]}]
        self.catalog = LabelCatalog(self.records)
        self.dialog = LabelDialog(self.catalog)
        self.addCleanup(self.dialog.deleteLater)
        self.addCleanup(self.dialog.close)
        self.dialog.show()
        self.app.processEvents()
        self.buttons = self.dialog.findChild(QDialogButtonBox)

    def click(self, text):
        button = next(button for button in self.dialog.findChildren(QPushButton)
                      if button.text() == text)
        QTest.mouseClick(button, Qt.LeftButton)
        self.app.processEvents()

    def test_restore_then_cancel_keeps_user_settings(self):
        self.click("恢复默认四类")
        self.assertEqual(len(self.dialog.rows), 4)
        self.buttons.button(QDialogButtonBox.Cancel).click()
        self.assertEqual(self.dialog.result(), QDialog.Rejected)
        self.assertEqual(self.dialog.catalog.to_list(), self.records)

    def test_restore_then_save_applies_four_classes_and_their_aliases(self):
        self.click("恢复默认四类")
        self.buttons.button(QDialogButtonBox.Ok).click()
        self.assertEqual(self.dialog.result(), QDialog.Accepted)
        self.assertEqual(self.dialog.catalog.to_list(), LabelCatalog.defaults().to_list())
        for token, english, name in [("0", "drop_paint", "掉漆"), ("1", "scratch", "划痕"),
                                     ("2", "stain", "污渍"), ("3", "other", "其他")]:
            self.assertEqual(self.dialog.catalog.resolve(token), name)
            self.assertEqual(self.dialog.catalog.resolve(english), name)
        self.assertEqual(self.catalog.to_list(), self.records)

    def test_click_each_default_field_then_add_remove_and_save(self):
        self.click("恢复默认四类")
        for _, name, aliases in self.dialog.rows:
            QTest.mouseClick(name, Qt.LeftButton)
            QTest.mouseClick(aliases, Qt.LeftButton)
        self.click("添加种类")
        _, name, aliases = self.dialog.rows[-1]
        QTest.keyClicks(name, "dent")
        QTest.mouseClick(aliases, Qt.LeftButton)
        QTest.keyClicks(aliases, "4, mark")
        for row in (0, 2):
            QTest.mouseClick(self.dialog.rows[row][0], Qt.LeftButton)
        self.click("移除所选种类")
        self.click("保存")
        self.assertEqual(self.dialog.result(), QDialog.Accepted)
        self.assertEqual([r["name"] for r in self.dialog.catalog.to_list()], ["划痕", "其他", "dent"])
        self.assertEqual(self.dialog.catalog.resolve("mark"), "dent")

    def test_delete_all_and_save_empty_catalog(self):
        for selected, _, _ in self.dialog.rows:
            selected.click()
        self.click("移除所选种类")
        self.click("保存")
        self.assertEqual(self.dialog.result(), QDialog.Accepted)
        self.assertEqual(self.dialog.catalog.to_list(), [])

    def test_conflicting_alias_can_be_corrected_before_save(self):
        _, _, aliases = self.dialog.rows[0]
        aliases.setText("3")
        with patch("label_dialog.QMessageBox.warning") as warning:
            self.click("保存")
        warning.assert_called_once()
        self.assertEqual(self.dialog.catalog.to_list(), self.records)
        aliases.setText("2，drop")
        self.click("保存")
        self.assertEqual(self.dialog.result(), QDialog.Accepted)
        self.assertEqual(self.dialog.catalog.resolve("drop"), "脱落")
