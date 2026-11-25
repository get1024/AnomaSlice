"""Project state and backwards-compatible JSON/PNG persistence.

This module has no window or canvas dependencies. Masks are ordinary QImages,
and all validation happens before an opened project replaces the active state.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import QIODevice, QLockFile, QSaveFile
from PySide6.QtGui import QImage

from annotation_store import LabelCatalog
from image_dataset import EXPORT_MARKER
from json_storage import write_json_atomic


@contextmanager
def json_write_lock(path):
    """Serialize read-modify-write operations from application instances."""
    lock = QLockFile(os.fspath(path) + ".lock")
    lock.setStaleLockTime(0)  # Mask saves can exceed the default 30-second timeout.
    if not lock.tryLock(0):
        raise OSError(f"无法锁定文件以保存，请检查目录权限或稍后重试：{path}")
    try:
        yield
    finally:
        lock.unlock()


def _read_project(json_path):
    with open(json_path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or "images" not in data or "config" not in data:
        raise ValueError("这不是 AnomaSlice 工程文件；图片标注 JSON 请通过打开对应图片载入。")
    if data.get("version") not in ("1.0", "1.1"):
        raise ValueError("不支持的 AnomaSlice 工程版本。")
    if not isinstance(data["config"], dict):
        raise ValueError("工程参数 config 必须是 JSON 对象。")
    return data


def _catalog_fields(config):
    return {key: deepcopy(config[key]) for key in ("label_catalog", "label_catalog_version")
            if key in config}


class ProjectManager:
    def __init__(self):
        self.project_path = None
        self.images = []
        self._patch_stems = {}
        self.masks = {}
        self._saved_masks = {}
        self.annotations = {}
        self.current_index = -1
        self.config = {}
        self._saved_catalog = {}
        self.normal_dir = None
        self.abnormal_dir = None

    def set_images(self, images):
        self.images = list(images)
        stems = {image: Path(image).stem for image in self.images}
        counts = Counter(stem.casefold() for stem in stems.values())
        reserved = set(counts)
        self._patch_stems = {}
        for image in sorted(image for image, stem in stems.items() if counts[stem.casefold()] > 1):
            identity = os.path.normcase(os.path.abspath(image))
            digest = hashlib.sha256(os.fsencode(identity)).hexdigest()[:20]
            stem = f"{stems[image][:32]}~{digest}"
            # A real image can itself be named like a generated patch prefix.
            while stem.casefold() in reserved:
                stem += "~"
            self._patch_stems[image] = stem
            reserved.add(stem.casefold())

    def patch_stem(self, image):
        return self._patch_stems.get(image, Path(image).stem)

    def set_mask(self, image_path, mask):
        if mask is not None and not mask.isNull():
            self.masks[image_path] = mask.convertToFormat(QImage.Format_Grayscale8)

    def get_mask(self, image_path):
        mask = self.masks.get(image_path)
        return QImage(mask) if mask is not None else None

    @staticmethod
    def _save_mask(mask, path):
        output = QSaveFile(path)
        if not output.open(QIODevice.WriteOnly):
            raise OSError(f"无法写入工程掩膜：{path}\n{output.errorString()}")
        if not mask.save(output, "PNG"):
            output.cancelWriting()
            raise OSError(f"无法保存工程掩膜：{path}")
        if not output.commit():
            raise OSError(f"无法提交工程掩膜：{path}\n{output.errorString()}")

    def _stage_masks(self, json_path, masks, existing_paths):
        """Write outside the live JSON's mask paths until its atomic replacement succeeds."""
        base_dir = os.path.dirname(json_path)
        project_name = os.path.splitext(os.path.basename(json_path))[0]
        mask_dir = os.path.join(base_dir, project_name + "_masks")
        if masks:
            try:
                os.mkdir(mask_dir)
            except FileExistsError:
                if not os.path.isdir(mask_dir):
                    raise
            else:
                # Only newly created output directories are known to contain no source images.
                with open(os.path.join(mask_dir, EXPORT_MARKER), "xb"):
                    pass
        live_paths = {os.path.normcase(os.path.abspath(os.path.join(base_dir, path)))
                      for path in existing_paths.values()}
        mask_paths_map = {}
        for img_path, mask in masks.items():
            name_hash = hashlib.md5(img_path.encode("utf-8")).hexdigest()
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            stem = os.path.join(mask_dir, f"{base_name}_{name_hash}")
            slots = (stem + ".png", stem + "_next.png")
            for save_path in slots:
                if os.path.normcase(save_path) not in live_paths:
                    self._save_mask(mask, save_path)
                    mask_paths_map[img_path] = os.path.relpath(save_path, base_dir)
                    break
            else:
                raise ValueError(f"工程中的多个掩膜引用了同一图片的保存位置：{img_path}")
        return mask_paths_map

    def _check_mask_snapshot(self, image_path, existing_paths):
        relative = existing_paths.get(image_path)
        disk_mask = None
        if relative:
            target = os.path.join(os.path.dirname(self.project_path), relative)
            disk = QImage(target)
            if not disk.isNull():
                disk_mask = disk.convertToFormat(QImage.Format_Grayscale8)
            elif os.path.exists(target):
                raise ValueError(f"无法读取已保存的掩膜：{target}")
        if disk_mask != self._saved_masks.get(image_path):
            raise ValueError("当前图片的工程掩膜已被修改，请重新打开工程。")

    def save_image_mask(self, image_path, mask):
        """Merge one image into the latest project without replacing other reviewers' edits."""
        path = self.project_path
        with json_write_lock(path):
            data = _read_project(path)
            if image_path not in data["images"]:
                raise ValueError("当前图片已从工程中移除，请重新打开工程。")
            paths = self._mask_paths(data)
            self._check_mask_snapshot(image_path, paths)
            paths.update(self._stage_masks(path, {image_path: mask}, paths))
            data["masks"] = paths
            write_json_atomic(path, data)
        self.set_mask(image_path, mask)
        self._saved_masks[image_path] = self.masks[image_path]

    def save_project(self, json_path, current_index, config, normal_files=None, abnormal_files=None,
                     *, current_mask=None):
        json_path = os.path.abspath(os.fspath(json_path))
        masks = dict(self.masks)
        if current_mask is not None:
            masks[self.images[current_index]] = current_mask.convertToFormat(QImage.Format_Grayscale8)
        data = {
            "version": "1.1", "images": self.images, "current_index": current_index,
            "config": config, "annotations": self.annotations,
            "normal_files": normal_files if normal_files is not None else [],
            "abnormal_files": abnormal_files if abnormal_files is not None else [],
        }
        with json_write_lock(json_path):
            same_project = self.project_path and os.path.normcase(json_path) == os.path.normcase(
                os.path.abspath(self.project_path)
            )
            previous = {}
            if same_project or os.path.isfile(json_path):
                try:
                    previous = _read_project(json_path)
                except ValueError:
                    if same_project:
                        raise
            existing_paths = self._mask_paths(previous)
            if same_project:
                self._check_catalog_snapshot(previous["config"])
                changed = {image: mask for image, mask in masks.items() if mask != self._saved_masks.get(image)}
                for image in changed:
                    self._check_mask_snapshot(image, existing_paths)
                data["masks"] = existing_paths | self._stage_masks(json_path, changed, existing_paths)
            else:
                data["masks"] = self._stage_masks(json_path, masks, existing_paths)
            write_json_atomic(json_path, data)
        self.project_path = json_path
        self.masks = masks
        self.config = deepcopy(config)
        self._saved_catalog = _catalog_fields(config)
        self._saved_masks = dict(self.masks)

    @staticmethod
    def _mask_paths(data):
        paths = data.get("masks", {})
        if not isinstance(paths, dict) or any(
            not isinstance(image, str) or not image.strip()
            or not isinstance(path, str) or not path.strip() for image, path in paths.items()
        ):
            raise ValueError("工程中的 masks 必须使用图片路径对应掩膜路径。")
        return paths

    def _check_catalog_snapshot(self, config):
        if _catalog_fields(config) != self._saved_catalog:
            raise ValueError("工程的缺陷种类与别名已被其他操作修改，请重新打开工程后再保存。")

    def save_label_catalog(self, catalog):
        """Update just the catalog in the current project, preserving other on-disk edits."""
        if not self.project_path:
            raise ValueError("请先打开或保存工程。")
        records = catalog.to_list()
        with json_write_lock(self.project_path):
            data = _read_project(self.project_path)
            config = data["config"]
            self._check_catalog_snapshot(config)
            config.update(label_catalog=records, label_catalog_version=LabelCatalog.VERSION)
            write_json_atomic(self.project_path, data)
        self._saved_catalog = _catalog_fields(config)
        self.config.update(_catalog_fields(config))

    def load_project(self, json_path):
        data = _read_project(json_path)

        def path_list(value, description):
            if not isinstance(value, list) or any(
                not isinstance(path, str) or not path.strip() for path in value
            ):
                raise ValueError(f"工程中的{description}必须是有效路径列表。")
            return list(value)

        images = path_list(data["images"], "图片")
        current_index = data.get("current_index", -1)
        if isinstance(current_index, bool) or not isinstance(current_index, int):
            raise ValueError("工程中的当前图片编号必须是整数。")
        if (images and not 0 <= current_index < len(images)) or (not images and current_index != -1):
            raise ValueError("工程中的当前图片编号超出图片列表范围。")
        config = dict(data["config"])
        for key, minimum, maximum in (
            ("patch_width", 1, 8192), ("patch_height", 1, 8192),
            ("stride", 1, 8192), ("threshold", 0, 100),
            ("mask_opacity", 0, 100), ("mode", 0, 1),
            ("label_catalog_version", 1, LabelCatalog.VERSION),
        ):
            if key in config:
                value = config[key]
                if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                    raise ValueError(f"工程参数 {key} 必须是 {minimum} 至 {maximum} 的整数。")
        for key in ("normal_dir", "abnormal_dir"):
            if key in config and not isinstance(config[key], str):
                raise ValueError(f"工程参数 {key} 必须是路径文字。")
        if "smart_grid" in config and not isinstance(config["smart_grid"], bool):
            raise ValueError("工程参数 smart_grid 必须是布尔值。")
        if "label_catalog" in config:
            config["label_catalog"] = LabelCatalog(config["label_catalog"]).to_list()

        annotations = data.get("annotations", {})
        if not isinstance(annotations, dict):
            raise ValueError("工程中的 annotations 必须是 JSON 对象。")
        validated_annotations = {}
        for image_path, boxes in annotations.items():
            if not isinstance(image_path, str) or not image_path.strip() or not isinstance(boxes, list):
                raise ValueError("工程中的标注必须使用图片路径对应标注框列表。")
            validated_boxes = []
            for box in boxes:
                if not isinstance(box, dict) or not isinstance(box.get("label"), str) or not box["label"].strip():
                    raise ValueError("工程中的标注框缺少有效标签。")
                bbox = box.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4 or any(
                    isinstance(value, bool) or not isinstance(value, int) for value in bbox
                ):
                    raise ValueError("工程中的标注框坐标必须是 [x, y, 宽, 高] 整数列表。")
                if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= 0 or bbox[3] <= 0:
                    raise ValueError("工程中的标注框起点不能为负，宽高必须大于零。")
                validated_boxes.append({"label": box["label"].strip(), "bbox": list(bbox)})
            validated_annotations[image_path] = validated_boxes

        normal_files = path_list(data.get("normal_files", []), "Normal 结果")
        abnormal_files = path_list(data.get("abnormal_files", []), "Abnormal 结果")
        mask_map = self._mask_paths(data)
        base_dir = os.path.dirname(json_path)
        masks = {}
        for img_path, mask_rel_path in mask_map.items():
            full_path = os.path.join(base_dir, mask_rel_path)
            if os.path.exists(full_path):
                mask = QImage(full_path)
                if mask.isNull():
                    raise ValueError(f"无法解码工程掩膜：{full_path}")
                masks[img_path] = mask.convertToFormat(QImage.Format_Grayscale8)

        self.project_path = os.path.abspath(os.fspath(json_path))
        self.set_images(images)
        self.current_index = current_index
        self.config = config
        self._saved_catalog = _catalog_fields(data["config"])
        self.annotations = validated_annotations
        self.masks = masks
        self._saved_masks = dict(masks)
        self.normal_dir = config.get("normal_dir")
        self.abnormal_dir = config.get("abnormal_dir")
        return self.current_index, self.config, normal_files, abnormal_files
