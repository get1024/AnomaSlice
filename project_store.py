"""Project state and backwards-compatible JSON/PNG persistence.

This module has no window or canvas dependencies. Masks are ordinary QImages,
and all validation happens before an opened project replaces the active state.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile

from PySide6.QtCore import QIODevice, QSaveFile
from PySide6.QtGui import QImage

from annotation_store import LabelCatalog


class ProjectManager:
    def __init__(self):
        self.project_path = None
        self.images = []
        self.masks = {}
        self.annotations = {}
        self.current_index = -1
        self.config = {}
        self.normal_dir = None
        self.abnormal_dir = None

    def set_images(self, images):
        self.images = list(images)

    def set_mask(self, image_path, mask):
        if mask is not None and not mask.isNull():
            self.masks[image_path] = QImage(mask)

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

    def save_project(self, json_path, current_index, config, normal_files=None, abnormal_files=None):
        json_path = os.path.abspath(os.fspath(json_path))
        base_dir = os.path.dirname(json_path)
        project_name = os.path.splitext(os.path.basename(json_path))[0]
        mask_dir = os.path.join(base_dir, project_name + "_masks")
        if self.masks:
            os.makedirs(mask_dir, exist_ok=True)
        mask_paths_map = {}
        for img_path, mask in self.masks.items():
            name_hash = hashlib.md5(img_path.encode("utf-8")).hexdigest()
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(mask_dir, f"{base_name}_{name_hash}.png")
            self._save_mask(mask, save_path)
            mask_paths_map[img_path] = os.path.relpath(save_path, base_dir)
        data = {
            "version": "1.1", "images": self.images, "current_index": current_index,
            "config": config, "masks": mask_paths_map, "annotations": self.annotations,
            "normal_files": normal_files if normal_files is not None else [],
            "abnormal_files": abnormal_files if abnormal_files is not None else [],
        }
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=base_dir, suffix=".tmp", delete=False,
            ) as stream:
                temporary_path = stream.name
                json.dump(data, stream, indent=4, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, json_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
        self.project_path = json_path

    def load_project(self, json_path):
        with open(json_path, "r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or "images" not in data or "config" not in data:
            raise ValueError("这不是 AnomaSlice 工程文件；图片标注 JSON 请通过打开对应图片载入。")
        if data.get("version") not in ("1.0", "1.1"):
            raise ValueError("不支持的 AnomaSlice 工程版本。")

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
        config = data["config"]
        if not isinstance(config, dict):
            raise ValueError("工程参数 config 必须是 JSON 对象。")
        config = dict(config)
        for key, minimum, maximum in (
            ("patch_width", 1, 8192), ("patch_height", 1, 8192),
            ("stride", 1, 8192), ("threshold", 0, 100),
            ("mask_opacity", 0, 100), ("mode", 0, 1),
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
        mask_map = data.get("masks", {})
        if not isinstance(mask_map, dict) or any(
            not isinstance(image_path, str) or not image_path.strip()
            or not isinstance(mask_path, str) or not mask_path.strip()
            for image_path, mask_path in mask_map.items()
        ):
            raise ValueError("工程中的 masks 必须使用图片路径对应掩膜路径。")
        base_dir = os.path.dirname(json_path)
        masks = {}
        for img_path, mask_rel_path in mask_map.items():
            full_path = os.path.join(base_dir, mask_rel_path)
            if os.path.exists(full_path):
                mask = QImage(full_path)
                if mask.isNull():
                    raise ValueError(f"无法解码工程掩膜：{full_path}")
                masks[img_path] = mask.convertToFormat(QImage.Format_Grayscale8)

        self.project_path = os.fspath(json_path)
        self.images = images
        self.current_index = current_index
        self.config = config
        self.annotations = validated_annotations
        self.masks = masks
        self.normal_dir = config.get("normal_dir")
        self.abnormal_dir = config.get("abnormal_dir")
        return self.current_index, self.config, normal_files, abnormal_files
