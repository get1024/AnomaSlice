"""Live dataset recall for the comparison toolbar."""

from itertools import islice
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QLabel

from dataset_metrics import DatasetMetricsScan
from detection_metrics import match_counts


class DatasetRecallLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTextFormat(Qt.PlainText)
        self.setStyleSheet("font-weight: 600; padding: 0 10px;")
        self.setAccessibleName("数据集缺陷检出率")
        self._workers = set()
        self._scan = None
        self._counts = {}
        self._errors = {}
        self._matched = self._total = self._predicted = 0
        self._min_coverage = 0.0
        self._ready = False
        self.hide()

    def start_dataset(self, images, comparison, min_coverage):
        if self._scan:
            self._scan.requestInterruption()
        self._scan = None
        self._counts.clear()
        self._errors.clear()
        self._matched = self._total = self._predicted = 0
        self._min_coverage = min_coverage
        self._ready = False
        if comparison is None:
            self.clear()
            self.hide()
            return
        self._refresh()
        scan = DatasetMetricsScan(images, comparison, self, min_coverage=min_coverage)
        self._workers.add(scan)
        self._scan = scan
        scan.result_ready.connect(self._accept_scan)
        scan.finished.connect(self._worker_finished)
        scan.start()

    @Slot(object)
    def _accept_scan(self, result):
        if self.sender() is not self._scan:
            return
        # Images opened or edited while scanning already have newer in-memory counts.
        for path, counts in result.counts.items():
            if path not in self._counts and path not in self._errors:
                self._counts[path] = counts
                self._matched += counts[0]
                self._total += counts[1]
                self._predicted += counts[2]
        for path, error in result.errors.items():
            if path not in self._counts and path not in self._errors:
                self._errors[path] = error
        self._ready = True
        self._refresh()

    @Slot()
    def _worker_finished(self):
        worker = self.sender()
        self._workers.discard(worker)
        if self._scan is worker:
            self._scan = None
        worker.deleteLater()

    def stop_scans(self):
        self._scan = None
        for worker in self._workers:
            worker.requestInterruption()
        for worker in self._workers:
            worker.wait()

    def update_image(self, image, boxes, predictions):
        self._replace_image(image, match_counts(boxes, predictions, self._min_coverage))

    def image_error(self, image, error):
        self._replace_image(image, None, error)

    def _replace_image(self, image, counts, error=None):
        path = Path(image).resolve()
        old_matched, old_total, old_predicted = self._counts.pop(path, (0, 0, 0))
        self._matched -= old_matched
        self._total -= old_total
        self._predicted -= old_predicted
        if counts is None:
            self._errors[path] = str(error)
        else:
            self._errors.pop(path, None)
            self._counts[path] = counts
            self._matched += counts[0]
            self._total += counts[1]
            self._predicted += counts[2]
        self._refresh()

    def _refresh(self):
        rule = ("两框交集面积大于零" if self._min_coverage == 0
                else "交集面积 ÷ 两框中较小的面积 ≥ 50%")
        hint = (f"检出率 = 正确匹配数 ÷（正确匹配数 + 漏检数）。当前条件为{rule}。\n"
                "按同图位置一对一匹配，允许类别不同。先匹配最多的缺陷，再最大化小框覆盖率总和。\n"
                "每个未匹配的 GT 计一次漏检，每个未匹配的预测计一次多检。统计包含未保存的 GT 修改。")
        if not self._ready:
            value = "统计中…"
        elif self._errors:
            value = f"—    {len(self._errors)} 张图片无法统计"
            hint += "\n" + "\n".join(f"{p.name}：{e}" for p, e in islice(self._errors.items(), 5))
        else:
            rate = f"{self._matched / self._total:.1%}" if self._total else "—"
            value = (f"{rate}    正确匹配 {self._matched} / 人工标注 {self._total}"
                     f"    漏检 {self._total - self._matched}    多检 {self._predicted - self._matched}")
        self.setText(f"数据集缺陷检出率  {value}")
        self.setToolTip(hint)
