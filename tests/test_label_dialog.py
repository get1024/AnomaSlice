import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
        self.restore = next(button for button in self.dialog.findChildren(QPushButton)
                            if button.text() == "恢复默认四类")
        self.buttons = self.dialog.findChild(QDialogButtonBox)

    def test_restore_then_cancel_keeps_user_settings(self):
        self.restore.click()
        self.assertEqual(self.dialog.table.rowCount(), 4)
        self.buttons.button(QDialogButtonBox.Cancel).click()
        self.assertEqual(self.dialog.result(), QDialog.Rejected)
        self.assertEqual(self.dialog.catalog.to_list(), self.records)

    def test_restore_then_save_applies_four_classes_and_their_aliases(self):
        self.restore.click()
        self.buttons.button(QDialogButtonBox.Ok).click()
        self.assertEqual(self.dialog.result(), QDialog.Accepted)
        self.assertEqual(self.dialog.catalog.to_list(), LabelCatalog.defaults().to_list())
        for token, english, name in [("0", "drop_paint", "掉漆"), ("1", "scratch", "划痕"),
                                     ("2", "stain", "污渍"), ("3", "other", "其他")]:
            self.assertEqual(self.dialog.catalog.resolve(token), name)
            self.assertEqual(self.dialog.catalog.resolve(english), name)
        self.assertEqual(self.catalog.to_list(), self.records)
