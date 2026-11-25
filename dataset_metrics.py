"""Read a comparison dataset off the UI thread and count matches per image."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from PySide6.QtCore import QThread, Signal

from detection_metrics import match_counts
from xml_annotations import AnnotationXml, xml_io_lock


@dataclass(frozen=True)
class DatasetScanResult:
    counts: Mapping[Path, tuple[int, int, int]]
    errors: Mapping[Path, str]


class DatasetMetricsScan(QThread):
    """Emit one snapshot on completion. A cancelled scan discards partial results."""

    result_ready = Signal(object)

    def __init__(self, images, comparison, parent=None, *, min_coverage=0.0):
        super().__init__(parent)
        self._images = tuple(images)
        self._comparison = comparison
        self._min_coverage = min_coverage

    def run(self):
        counts = {}
        errors = {}
        resolve = self._comparison.scan_resolver()
        for path in self._images:
            if self.isInterruptionRequested():
                return
            image = Path(path)
            try:
                image = image.resolve()
                with xml_io_lock:
                    pair = resolve(image)
                gt = AnnotationXml.open_path(pair.gt).boxes
                predictions = AnnotationXml.open_path(pair.prediction).boxes
                counts[image] = match_counts(gt, predictions, self._min_coverage)
            except Exception as error:
                # Keep individual XML, path and matcher failures visible in the result.
                errors[image] = str(error) or type(error).__name__
        if not self.isInterruptionRequested():
            result = DatasetScanResult(MappingProxyType(counts), MappingProxyType(errors))
            self.result_ready.emit(result)
