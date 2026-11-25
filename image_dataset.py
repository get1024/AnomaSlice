"""Discover source images, exported images, and their editable XML destinations."""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtGui import QImageReader


EXPORT_MARKER = ".anomaslice-export"
EXPORT_PNG_KEY = "AnomaSlice-Export"


def image_extensions():
    return {bytes(fmt).decode("ascii") for fmt in QImageReader.supportedImageFormats()}


def natural_key(path):
    return [int(s) if i % 2 else s.casefold()
            for i, s in enumerate(re.split(r"([0-9]+)", str(path)))]


@dataclass(frozen=True)
class ImageEntry:
    image: Path
    xml: Path | None = None
    issue: str = ""

    @staticmethod
    def lookup(path, entries):
        image = Path(path).resolve()
        if image not in entries:
            # Older projects store paths only. Index the directory once on demand.
            entries.update((entry.image, entry) for entry in scan_folder(image.parent, recursive=False,
                                                                         selected=image))
        if image not in entries:
            raise ValueError("图片已移动，请重新打开文件夹。")
        return entries[image]

    def target(self):
        if self.issue:
            raise ValueError(self.issue)
        adjacent = self.image.with_suffix(".xml")
        nested = self.image.parent / "xml" / adjacent.name
        matches = {p for p in (adjacent, nested) if p.is_file()}
        if self.xml is not None:
            # A case-insensitive Mac volume can open XML/Foo.XML via xml/foo.xml,
            # although those spellings are unequal to pathlib on macOS.
            if self.xml.is_file():
                matches = {p for p in matches if str(p).casefold() != str(self.xml).casefold()
                           or not p.samefile(self.xml)}
            matches.add(self.xml)
        if len(matches) > 1:
            raise ValueError("存在多份同名 XML，请确认对应文件后重新打开。")
        return next(iter(matches), nested)


def scan_folder(root, *, recursive=True, selected=None):
    root = Path(root).resolve(strict=True)
    formats = image_extensions()
    images = defaultdict(list)
    exports = defaultdict(list)
    xmls = defaultdict(list)

    def report_error(error):
        raise error

    for directory, subdirs, names in root.walk(on_error=report_error):
        if directory != root and EXPORT_MARKER in names:
            subdirs.clear()
            continue
        if not recursive:
            subdirs[:] = [d for d in subdirs if directory == root and d.casefold() == "xml"]
        for name in names:
            if name.startswith(".anomaslice-write-") and name.endswith(".tmp"):
                continue
            path = directory / name
            ext = path.suffix[1:].lower()
            if ext == "xml":
                xmls[path.stem.casefold()].append(path)
            elif (recursive or directory == root) and (ext in formats or QImageReader.imageFormat(str(path))):
                # Both export sequences and patch coordinates can occur in real
                # camera filenames. The embedded marker establishes ownership.
                generated = (ext == "png" and re.search(r"^\d{6,}_|_\d+_\d+$", path.stem)
                             and QImageReader(str(path)).text(EXPORT_PNG_KEY) == "1")
                group = exports if generated else images
                group[path.stem.casefold()].append(path)

    if not images:
        images = exports
    elif selected is not None:
        selected = Path(selected).resolve()
        matching_exports = exports.get(selected.stem.casefold(), ())
        if selected in matching_exports or (selected.suffix.casefold() == ".xml" and matching_exports
                                            and selected.stem.casefold() not in images):
            images = exports
    entries = []
    for stem, group in images.items():
        candidates = xmls.get(stem, [])
        nearby = {}
        for image in group:
            nearby[image] = [
                xml for xml in candidates
                if xml.parent == image.parent or (
                    xml.parent.name.casefold() == "xml" and xml.parent.parent == image.parent
                )
            ]
        owners = Counter(xml for matches in nearby.values() for xml in matches)
        per_directory = Counter(image.parent for image in group)
        for image, matches in nearby.items():
            if len(matches) == 1 and owners[matches[0]] == 1:
                entries.append(ImageEntry(image, matches[0]))
            elif len(group) == len(candidates) == 1:
                # A unique stem also works for separate image and annotation trees.
                entries.append(ImageEntry(image, candidates[0]))
            elif matches or per_directory[image.parent] > 1 or any(xml not in owners for xml in candidates):
                entries.append(ImageEntry(image, issue="同名图片或 XML 有多份，无法确定对应关系。"))
            else:
                entries.append(ImageEntry(image))
    return sorted(entries, key=lambda entry: natural_key(entry.image.relative_to(root)))
