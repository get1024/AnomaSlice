"""Exercise update actions and window lifecycle with local release responses."""

import hashlib
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QPushButton

from app_window import MainWindow
from update_dialog import UpdateDialog
from update_service import RELEASES_URL


PAYLOAD = b"test update bytes"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def metadata(tag="v1.1.0"):
    asset = {
        "name": "AnomaSlice.exe", "state": "uploaded", "size": len(PAYLOAD),
        "digest": f"sha256:{DIGEST}",
        "browser_download_url": f"{RELEASES_URL}/download/{tag}/AnomaSlice.exe",
    }
    return io.BytesIO(json.dumps({
        "tag_name": tag, "draft": False, "prerelease": False, "assets": [asset],
    }).encode())


class UpdateDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(patch("update_dialog.APP_VERSION", "1.0.0"))
        self.enterContext(patch("update_service.sys", SimpleNamespace(platform="win32")))
        self.enterContext(patch("update_service.platform.machine", return_value="AMD64"))

    def open_dialog(self, responses, *, frozen=True):
        self.fetch = self.enterContext(patch("update_service._open", side_effect=responses))
        self.enterContext(patch("update_dialog.sys", SimpleNamespace(frozen=frozen)))
        dialog = UpdateDialog(self.directory)
        self.addCleanup(dialog.deleteLater)
        self.addCleanup(dialog.close)
        dialog.show()
        self.finish_worker(dialog)
        return dialog

    def finish_worker(self, dialog):
        self.assertTrue(dialog.worker.wait(5000))
        self.app.processEvents()
        self.assertIsNone(dialog.worker)

    def test_failed_check_can_retry_then_report_up_to_date(self):
        dialog = self.open_dialog([OSError("offline"), metadata("v1.0.0")])
        self.assertIn("offline", dialog.status.text())
        self.assertEqual(dialog.action.text(), "重试")
        dialog.action.click()
        self.finish_worker(dialog)
        self.assertIn("最新", dialog.status.text())
        self.assertTrue(dialog.action.isHidden())
        self.assertIsNone(dialog.release)
        self.assertEqual(self.fetch.call_count, 2)

    def test_download_retries_after_bad_checksum_and_accepts_verified_file(self):
        dialog = self.open_dialog([metadata(), io.BytesIO(PAYLOAD.upper()), io.BytesIO(PAYLOAD)])
        self.assertIn("v1.1.0", dialog.status.text())
        self.assertEqual(dialog.action.text(), "下载并重启更新")
        dialog.action.click()
        self.finish_worker(dialog)
        self.assertIn("SHA-256", dialog.status.text())
        self.assertEqual(dialog.action.text(), "重试")
        self.assertIsNone(dialog.download)
        self.assertEqual(list(self.directory.iterdir()), [])
        self.assertTrue(dialog.isVisible())
        dialog.action.click()
        self.finish_worker(dialog)
        self.assertEqual(dialog.result(), QDialog.Accepted)
        self.assertEqual(dialog.download.parent, self.directory)
        self.assertEqual(dialog.download.read_bytes(), PAYLOAD)
        self.assertEqual(dialog.release.sha256, DIGEST)

    def test_cancel_wins_even_after_download_finishes_before_ui_receives_result(self):
        dialog = self.open_dialog([metadata(), io.BytesIO(PAYLOAD)])
        accepted, rejected = [], []
        dialog.accepted.connect(lambda: accepted.append(True))
        dialog.rejected.connect(lambda: rejected.append(True))
        dialog.action.click()
        self.assertTrue(dialog.worker.wait(5000))
        # The worker has verified the file, but its queued finished signal
        # has yet to reach the main thread. A cancel at this point must win.
        dialog.dismiss.click()
        self.app.processEvents()
        self.assertEqual((accepted, rejected), ([], [True]))
        self.assertIsNone(dialog.download)
        self.assertFalse(dialog.isVisible())

    def test_source_build_offers_release_page_without_install_action(self):
        dialog = self.open_dialog([metadata()], frozen=False)
        self.assertIn("源码", dialog.status.text())
        self.assertTrue(dialog.action.isHidden())
        button = next(b for b in dialog.findChildren(QPushButton) if "GitHub" in b.text())
        with patch("update_dialog.QDesktopServices.openUrl") as open_url:
            button.click()
        self.assertEqual(open_url.call_args.args[0].toString(), RELEASES_URL)
        self.assertEqual(self.fetch.call_count, 1)
        self.assertIsNone(dialog.download)


class WindowUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory())).resolve()
        self.enterContext(patch.object(MainWindow, "_get_global_config_path",
                                      return_value=str(self.root / "settings.json")))
        self.dialog_factory = self.enterContext(patch("app_window.UpdateDialog",
                                                      side_effect=self.downloaded_dialog))
        self.launch = self.enterContext(patch("app_window.launch_update"))
        self.window = MainWindow()
        self.window.show()
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.cleanup_window)

    def downloaded_dialog(self, directory, parent):
        self.download = directory / "AnomaSlice.exe"
        self.download.write_bytes(PAYLOAD)
        return Mock(download=self.download, release=SimpleNamespace(sha256=DIGEST),
                    worker=None, **{"exec.return_value": QDialog.Accepted})

    def cleanup_window(self):
        self.window._boxes_dirty = False
        self.window.close()
        self.app.processEvents()

    def test_help_action_hands_off_before_cleaning_download_and_closes(self):
        help_menu = next(a.menu() for a in self.window.menuBar().actions() if a.text() == "帮助")
        action = next(a for a in help_menu.actions() if a.text() == "检查更新…")
        received = []
        self.launch.side_effect = lambda path, system, digest: received.append((path.read_bytes(), system, digest))
        action.trigger()
        self.dialog_factory.assert_called_once()
        self.assertEqual(received, [(PAYLOAD, sys.platform, DIGEST)])
        self.assertFalse(self.window.isVisible())
        self.assertFalse(self.download.parent.exists())
        self.assertIsNone(self.window._pending_update)
        self.assertIsNone(self.window._update_dialog)

    def test_cancel_unsaved_gt_keeps_edits_open_and_discards_download(self):
        path = self.root / "part.png"
        image = QImage(64, 64, QImage.Format_RGB32)
        image.fill(0)
        self.assertTrue(image.save(str(path)))
        self.window._open_dataset_path(path)
        boxes = [{"label": "划痕", "bbox": [1, 2, 10, 12]}]
        self.window.canvas.set_annotations(boxes)
        self.window.canvas.annotations_changed(boxes)
        with patch("app_window.QMessageBox.question", return_value=QMessageBox.Cancel) as prompt:
            self.window.update_action.trigger()
        prompt.assert_called_once()
        self.launch.assert_not_called()
        self.assertTrue(self.window.isVisible())
        self.assertTrue(self.window._boxes_dirty)
        self.assertEqual(self.window.canvas.get_annotations(), boxes)
        self.assertFalse(self.window._xml_document.path.exists())
        self.assertFalse(self.download.parent.exists())
        self.assertIsNone(self.window._pending_update)
