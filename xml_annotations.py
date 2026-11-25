"""Read and edit DefectDetection XML while preserving detector metadata."""

import math
from copy import deepcopy
from pathlib import Path
from threading import Lock
from xml.dom import Node, minidom
from xml.parsers.expat import ExpatError

from PySide6.QtCore import QIODevice, QLockFile, QSaveFile
from PySide6.QtGui import QColor

from annotation_store import AnnotationStore
# Keep the original module's dataset API available to existing scripts.
from image_dataset import (
    EXPORT_MARKER as EXPORT_MARKER, EXPORT_PNG_KEY as EXPORT_PNG_KEY, ImageEntry as ImageEntry,
    image_extensions as image_extensions, natural_key as natural_key, scan_folder as scan_folder,
)


# Windows path resolution and file reads can both deny a simultaneous atomic
# rename. Background workers release this lock before parsing or matching boxes.
xml_io_lock = Lock()
_EMPTY = b"""<?xml version="1.0" encoding="utf-8"?>
<DefectDetection>
  <RawCount>0</RawCount>
  <Count>0</Count>
  <Detections/>
</DefectDetection>"""


def label_color(label):
    palette = {"掉漆": "#FFFF00", "划痕": "#FF0000", "污渍": "#0000FF", "其他": "#00FF00"}
    return palette.get(label) or QColor.fromHsv(sum(ord(c) * 37 for c in label) % 360, 180, 255).name()


def _children(node, name):
    return [child for child in node.childNodes
            if child.nodeType == Node.ELEMENT_NODE and child.tagName == name]


def _one(node, name):
    matches = _children(node, name)
    if len(matches) != 1:
        raise ValueError(f"{node.tagName} 中应有一个 {name} 节点")
    return matches[0]


def _text(node, name):
    children = _children(node, name)
    if not children:
        return ""
    return "".join(n.data for n in children[0].childNodes
                   if n.nodeType in (Node.TEXT_NODE, Node.CDATA_SECTION_NODE)).strip()


def _set_text(node, name, value):
    matches = _children(node, name)
    element = matches[0] if matches else node.appendChild(node.ownerDocument.createElement(name))
    for child in list(element.childNodes):
        if child.nodeType in (Node.TEXT_NODE, Node.CDATA_SECTION_NODE):
            element.removeChild(child)
    element.appendChild(node.ownerDocument.createTextNode(value))


class AnnotationXml:
    """A box's source_id identifies its exact XML node, even for identical boxes."""

    def __init__(self, path, source):
        self.path = path
        self._source = source
        try:
            self._dom = minidom.parseString(_EMPTY if source is None else source)
        except (ExpatError, LookupError) as error:
            raise ValueError(f"XML 解析失败：{error}") from error
        root = self._dom.documentElement
        if root.tagName != "DefectDetection":
            raise ValueError("需要 DefectDetection 格式的缺陷 XML")
        _one(root, "Count")
        self._original = []
        for index, node in enumerate(_children(_one(root, "Detections"), "Detection")):
            try:
                rect = [float(_text(node, k)) for k in ("x", "y", "width", "height")]
                if not all(map(math.isfinite, rect)) or min(rect[2:]) <= 0:
                    raise ValueError
            except ValueError as error:
                raise ValueError(f"第 {index + 1} 个框的坐标或尺寸无效") from error
            self._original.append({
                "source_id": index, "bbox": rect, "label": _text(node, "message") or "缺陷",
                "color": _text(node, "color"), "confidence": _text(node, "confidence"),
            })
        self.boxes = deepcopy(self._original)

    @classmethod
    def open(cls, entry):
        return cls.open_path(entry.target(), must_exist=entry.xml is not None)

    @classmethod
    def for_image(cls, entry, width, height, embedded_boxes):
        """Read plain-mode GT, including adjacent JSON and older projects."""
        doc = cls.open(entry)
        if not doc.exists:
            store = AnnotationStore()
            if Path(store.sidecar_path(entry.image)).exists():
                doc.boxes = store.load(entry.image, width, height, include_catalog=False)["boxes"]
            else:
                doc.boxes = store.validate_boxes(embedded_boxes, width, height)
        return doc

    @classmethod
    def open_path(cls, path, *, must_exist=False):
        try:
            with xml_io_lock:
                source = path.read_bytes()
        except FileNotFoundError:
            if must_exist:
                raise
            source = None
        return cls(path, source)

    @property
    def exists(self):
        return self._source is not None

    def save(self, boxes):
        if self.exists and boxes == self.boxes:
            return
        # Keep the opening DOM as a template so undo can restore a deleted node, including
        # its original confidence and custom children, even after an explicit save.
        ids = [box["source_id"] for box in boxes if "source_id" in box]
        if ids != sorted(set(ids)) or any(i < 0 or i >= len(self._original) for i in ids):
            raise ValueError("待保存的框与打开的 XML 节点不对应")
        kept = {box["source_id"]: (index, box) for index, box in enumerate(boxes, 1) if "source_id" in box}
        with self._dom.cloneNode(True) as dom:
            root = dom.documentElement
            parent = _one(root, "Detections")
            for index, node in enumerate(_children(parent, "Detection")):
                if index in kept:
                    ordinal, box = kept[index]
                    node.setAttribute("index", str(ordinal))
                    if box["label"] != self._original[index]["label"]:
                        _set_text(node, "message", box["label"])
                    for name, value, old in zip(("x", "y", "width", "height"), box["bbox"], self._original[index]["bbox"]):
                        if value != old:
                            _set_text(node, name, str(value))
                else:
                    indent = node.previousSibling
                    if indent and indent.nodeType == Node.TEXT_NODE and not indent.data.strip():
                        parent.removeChild(indent)
                    parent.removeChild(node)
            for ordinal, box in enumerate(boxes, 1):
                if "source_id" in box:
                    continue
                node = dom.createElement("Detection")
                node.setAttribute("index", str(ordinal))
                node.setAttribute("source", "manual")
                trailing = parent.lastChild
                if trailing and trailing.nodeType == Node.TEXT_NODE and not trailing.data.strip():
                    parent.removeChild(trailing)
                parent.appendChild(dom.createTextNode("\n    "))
                parent.appendChild(node)
                fields = dict(zip(("x", "y", "width", "height"), box["bbox"]))
                fields.update(message=box["label"], confidence="", color=label_color(box["label"]))
                for name, value in fields.items():
                    node.appendChild(dom.createTextNode("\n      "))
                    _set_text(node, name, str(value))
                node.appendChild(dom.createTextNode("\n    "))
                parent.appendChild(dom.createTextNode("\n  "))
            _set_text(root, "Count", str(len(boxes)))
            encoded = dom.toxml(encoding=self._dom.encoding or "utf-8", standalone=self._dom.standalone)
        source = self._source or b""
        if encoded.startswith(b"<?xml "):
            encoded = encoded.replace(b"?><", b"?>\n<", 1)
        if b"\r\n" in source:
            encoded = encoded.replace(b"\n", b"\r\n")
        if source.startswith(b"\xef\xbb\xbf") and not encoded.startswith(b"\xef\xbb\xbf"):
            encoded = b"\xef\xbb\xbf" + encoded
        try:
            minidom.parseString(encoded).unlink()
        except ExpatError as error:
            raise ValueError(f"标签含 XML 无法保存的字符：{error}") from error
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Serialize cooperating reviewers, then check that the source on disk is still ours.
        lock = QLockFile(str(self.path) + ".anomaslice.lock")
        if not lock.tryLock(0):
            raise OSError("无法锁定 XML，请检查目录写权限或稍后重试。")
        try:
            output = QSaveFile(str(self.path))
            if not output.open(QIODevice.WriteOnly):
                raise OSError(output.errorString())
            if output.write(encoded) != len(encoded):
                output.cancelWriting()
                raise OSError(output.errorString())
            try:
                current = self.path.read_bytes()
            except FileNotFoundError:
                current = None
            if current != self._source:
                output.cancelWriting()
                raise ValueError("XML 已被其他程序修改。请重新打开图片后再审核。")
            with xml_io_lock:
                if not output.commit():
                    raise OSError(output.errorString())
        finally:
            lock.unlock()
        self._source = encoded
        self.boxes = deepcopy(boxes)
