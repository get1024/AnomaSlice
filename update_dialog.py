"""User-initiated update check and download."""

import sys

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout,
)

from app_version import APP_VERSION
from update_service import RELEASES_URL, UpdateCheckWorker, UpdateDownloadWorker


class UpdateDialog(QDialog):
    def __init__(self, directory, parent=None):
        super().__init__(parent)
        self.directory = directory
        self.release = None
        self.download = None
        self.worker = None
        self.cancelling = False
        self.setWindowTitle("检查更新")
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"当前版本：AnomaSlice {APP_VERSION}"))
        self.status = QLabel()
        self.status.setTextFormat(Qt.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        buttons = QHBoxLayout()
        releases = QPushButton("查看 GitHub Release")
        releases.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(RELEASES_URL)))
        buttons.addWidget(releases)
        buttons.addStretch()
        self.action = QPushButton()
        self.action.clicked.connect(self._start)
        buttons.addWidget(self.action)
        self.dismiss = QPushButton("取消")
        self.dismiss.clicked.connect(self.reject)
        buttons.addWidget(self.dismiss)
        layout.addLayout(buttons)
        self._start()

    def _start(self):
        self.action.hide()
        self.dismiss.setText("取消")
        self.progress.show()
        if self.release is None:
            self.status.setText("正在检查 GitHub 最新正式版本…")
            self.progress.setRange(0, 0)
            self.worker = UpdateCheckWorker(APP_VERSION, self)
        else:
            self.status.setText(f"正在下载并校验 {self.release.asset_name}…")
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.worker = UpdateDownloadWorker(self.release, self.directory, self)
            self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def _finished(self):
        worker, self.worker = self.worker, None
        worker.deleteLater()
        self.progress.hide()
        self.dismiss.setText("关闭")
        if self.cancelling or worker.cancelled:
            super().reject()
        elif worker.error:
            self.status.setText(worker.error)
            self.action.setText("重试")
            self.action.show()
        elif self.release is not None:
            self.download = worker.path
            self.accept()
        elif worker.release is None:
            self.status.setText("当前已是最新可用版本。")
        else:
            self.release = worker.release
            self.status.setText(f"可更新至 {self.release.tag}（{self.release.asset_name}，"
                                f"{self.release.size / 1048576:.1f} MiB）。")
            if getattr(sys, "frozen", False):
                self.action.setText("下载并重启更新")
                self.action.show()
            else:
                self.status.setText(self.status.text() + "\n当前从源码运行。可从发布页下载安装包。")

    def reject(self):
        if self.worker is None:
            super().reject()
        else:
            self.cancelling = True
            self.worker.requestInterruption()
            self.dismiss.setEnabled(False)
            self.status.setText("正在取消…")
