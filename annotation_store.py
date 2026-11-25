"""Small, dependency-free label catalog and per-image annotation storage.

Bounding boxes use original-image pixel coordinates ``[x, y, width, height]``.
Their right and bottom edges are exclusive. Sidecars sit beside their images,
for example ``part.png.annotations.json``; the image itself is never changed.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping


# IDs and model labels from the GeSi Defect module, in model output order.
_DEFAULT_LABELS = (
    ("掉漆", "0", "drop_paint"),
    ("划痕", "1", "scratch"),
    ("污渍", "2", "stain"),
    ("其他", "3", "other"),
)


class AnnotationValidationError(ValueError):
    """A label catalog or annotation document is invalid or incompatible."""


def _text(value, description):
    if not isinstance(value, str) or not value.strip():
        raise AnnotationValidationError(f"{description}不能为空，且必须是文字。")
    return value.strip()


def _integer(value, description, minimum=0):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (isinstance(value, float) and not math.isfinite(value))
        or int(value) != value
        or value < minimum
    ):
        raise AnnotationValidationError(f"{description}必须是大于或等于 {minimum} 的整数。")
    return int(value)


class LabelCatalog:
    """User-defined labels, with exact, case-sensitive whole-token aliases.

    ``resolve(' 1 ')`` can return ``'划痕'`` while ``resolve('待定1')`` returns
    ``'待定1'``. Unknown nonempty text is accepted without changing the catalog.
    """

    @classmethod
    def defaults(cls):
        """Create an independent catalog for a new installation."""
        return cls([{"name": name, "aliases": [class_id, model_label]}
                    for name, class_id, model_label in _DEFAULT_LABELS])

    @classmethod
    def from_legacy(cls, records):
        """Fill missing defaults while preserving every existing token's meaning."""
        catalog = cls(records)
        by_name = {record["name"]: record for record in catalog.to_list()}
        occupied = catalog._names.union(catalog._aliases)
        for name, class_id, model_label in _DEFAULT_LABELS:
            if name in catalog._aliases:
                continue
            record = by_name.setdefault(name, {"name": name, "aliases": []})
            record["aliases"].extend(alias for alias in (class_id, model_label)
                                     if alias not in occupied)
        return cls(list(by_name.values()))

    def __init__(self, records=None):
        if records is None:
            records = []
        if not isinstance(records, (list, tuple)):
            raise AnnotationValidationError("缺陷类别必须是列表。")

        self._records = []
        self._names = set()
        self._aliases = {}
        for index, record in enumerate(records, 1):
            if not isinstance(record, Mapping):
                raise AnnotationValidationError(f"第 {index} 个缺陷类别格式无效。")
            name = _text(record.get("name"), f"第 {index} 个类别名称")
            if name in self._names:
                raise AnnotationValidationError(f"类别名称重复：{name}")
            aliases = record.get("aliases", [])
            if not isinstance(aliases, (list, tuple)):
                raise AnnotationValidationError(f"类别“{name}”的别名必须是列表。")
            self._records.append(
                {"name": name, "aliases": [_text(alias, "别名") for alias in aliases]}
            )
            self._names.add(name)

        # Collect every canonical name first, so aliases cannot shadow a label
        # listed later in the dialog or in a saved document.
        for record in self._records:
            for alias in record["aliases"]:
                if alias in self._names:
                    raise AnnotationValidationError(f"别名“{alias}”与类别名称重复。")
                if alias in self._aliases:
                    raise AnnotationValidationError(f"别名重复：{alias}")
                self._aliases[alias] = record["name"]

    def resolve(self, text):
        """Return the canonical label only when the complete token matches."""
        text = _text(text, "标签")
        return self._aliases.get(text, text)

    def add_label(self, name):
        """Add a canonical label without aliases; existing names are a no-op."""
        name = _text(name, "类别名称")
        if name in self._aliases:
            raise AnnotationValidationError(f"类别名称“{name}”已被用作别名。")
        if name not in self._names:
            self._records.append({"name": name, "aliases": []})
            self._names.add(name)
        return name

    def to_list(self):
        """Return independent JSON-compatible records for dialogs/storage."""
        return [
            {"name": record["name"], "aliases": list(record["aliases"])}
            for record in self._records
        ]


def _validate_boxes(boxes, width, height):
    if not isinstance(boxes, list):
        raise AnnotationValidationError("标注框 boxes 必须是列表。")
    validated = []
    for index, box in enumerate(boxes, 1):
        if not isinstance(box, Mapping):
            raise AnnotationValidationError(f"第 {index} 个标注框格式无效。")
        label = _text(box.get("label"), f"第 {index} 个框的标签")
        bbox = box.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise AnnotationValidationError(f"第 {index} 个框的 bbox 必须是 [x, y, 宽, 高]。")
        x = _integer(bbox[0], f"第 {index} 个框的 x")
        y = _integer(bbox[1], f"第 {index} 个框的 y")
        box_width = _integer(bbox[2], f"第 {index} 个框的宽", minimum=1)
        box_height = _integer(bbox[3], f"第 {index} 个框的高", minimum=1)
        if x + box_width > width or y + box_height > height:
            raise AnnotationValidationError(
                f"第 {index} 个标注框超出原图范围（{width} × {height}）。"
            )
        validated.append({"label": label, "bbox": [x, y, box_width, box_height]})
    return validated


class AnnotationStore:
    """Read/write adjacent UTF-8 JSON sidecars using atomic replacement.

    Missing sidecars return an empty document. Malformed/incompatible sidecars
    raise ``AnnotationValidationError`` and are never silently overwritten by
    this class. I/O failures raise ``OSError`` with the affected sidecar path.
    Callers should stop automatic saving for an image if loading it fails.
    """

    VERSION = 1

    @staticmethod
    def sidecar_path(image_path):
        return os.path.abspath(os.fspath(image_path)) + ".annotations.json"

    @staticmethod
    def validate_boxes(boxes, width, height):
        """Return independent, validated boxes for an original-image size.

        Project files can embed boxes without a sidecar. This gives their
        callers the same validation as ``load`` and ``save`` before display.
        """
        width = _integer(width, "原图宽度", minimum=1)
        height = _integer(height, "原图高度", minimum=1)
        return _validate_boxes(boxes, width, height)

    def _empty_document(self, image_path, width, height):
        return {
            "version": self.VERSION,
            "image": os.path.basename(os.fspath(image_path)),
            "width": _integer(width, "原图宽度", minimum=1),
            "height": _integer(height, "原图高度", minimum=1),
            "boxes": [],
            "label_catalog": [],
        }

    def load(self, image_path, width, height, *, include_catalog=True):
        """Load boxes for the expected image size and optionally the legacy catalog."""
        expected = self._empty_document(image_path, width, height)
        sidecar = self.sidecar_path(image_path)
        try:
            with open(sidecar, "r", encoding="utf-8-sig") as stream:
                document = json.load(stream)
        except FileNotFoundError:
            return expected
        except (json.JSONDecodeError, UnicodeError) as error:
            raise AnnotationValidationError(f"标注文件损坏：{sidecar}\n{error}") from error
        except OSError as error:
            raise OSError(f"无法读取标注文件：{sidecar}\n{error}") from error

        try:
            if not isinstance(document, Mapping):
                raise AnnotationValidationError("标注文件内容必须是 JSON 对象。")
            version = _integer(document.get("version"), "标注文件版本", minimum=1)
            if version != self.VERSION:
                raise AnnotationValidationError(f"不支持的标注文件版本：{version}")
            if document.get("image") != expected["image"]:
                raise AnnotationValidationError("标注文件中的图片名称与当前图片不一致。")
            stored_width = _integer(document.get("width"), "原图宽度", minimum=1)
            stored_height = _integer(document.get("height"), "原图高度", minimum=1)
            if (stored_width, stored_height) != (expected["width"], expected["height"]):
                raise AnnotationValidationError(
                    "标注文件中的图片尺寸与当前图片不一致；请确认图片未被缩放或替换。"
                )
            expected["boxes"] = _validate_boxes(
                document.get("boxes"), expected["width"], expected["height"]
            )
            if include_catalog:
                expected["label_catalog"] = LabelCatalog(
                    document.get("label_catalog", [])
                ).to_list()
        except AnnotationValidationError as error:
            raise AnnotationValidationError(f"标注文件无效：{sidecar}\n{error}") from error
        return expected

    def save(self, image_path, width, height, boxes, label_catalog=None):
        """Validate, atomically save, and return the absolute sidecar path."""
        document = self._empty_document(image_path, width, height)
        document["boxes"] = _validate_boxes(boxes, document["width"], document["height"])
        if isinstance(label_catalog, LabelCatalog):
            label_catalog = label_catalog.to_list()
        document["label_catalog"] = LabelCatalog(label_catalog).to_list()
        sidecar = self.sidecar_path(image_path)
        temporary_path = None
        try:
            # Same-directory replacement stays on the same filesystem, including
            # Windows and network shares that support atomic rename/replace.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=os.path.dirname(sidecar),
                prefix="." + os.path.basename(sidecar) + ".",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = stream.name
                json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, sidecar)
            temporary_path = None
        except OSError as error:
            raise OSError(f"无法保存标注文件：{sidecar}\n{error}") from error
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
        return sidecar
