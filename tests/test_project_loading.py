"""Project loading must not replace the active document on invalid JSON."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from main import ProjectManager


class ProjectLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "工程.json"
        self.manager = ProjectManager()
        self.manager.project_path = "previous.json"
        self.manager.images = ["previous.png"]
        self.manager.current_index = 0
        self.manager.annotations = {"previous.png": [{"label": "划痕", "bbox": [0, 0, 1, 1]}]}
        self.document = {
            "version": "1.1", "images": ["当前.png"], "current_index": 0,
            "config": {"patch_width": 256, "stride": 128, "label_catalog": []},
            "annotations": {"当前.png": [{"label": "待定1", "bbox": [1, 2, 3, 4]}]},
            "masks": {}, "normal_files": [], "abnormal_files": [],
        }

    def write_document(self, document):
        self.path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    def test_valid_project_loads_annotations_and_config(self):
        self.write_document(self.document)
        current_index, config, normal, abnormal = self.manager.load_project(str(self.path))
        self.assertEqual(current_index, 0)
        self.assertEqual(config["stride"], 128)
        self.assertEqual(self.manager.images, ["当前.png"])
        self.assertEqual(self.manager.annotations, self.document["annotations"])
        self.assertEqual((normal, abnormal), ([], []))

    def test_original_version_and_empty_project_remain_supported(self):
        self.write_document({"version": "1.0", "images": [], "current_index": -1, "config": {}})
        self.manager.load_project(str(self.path))
        self.assertEqual(self.manager.images, [])
        self.assertEqual(self.manager.current_index, -1)

    def test_image_sidecar_is_rejected_without_replacing_active_project(self):
        before = copy.deepcopy(self.manager.__dict__)
        self.write_document({"version": 1, "image": "part.png", "boxes": [], "width": 10, "height": 10})
        with self.assertRaisesRegex(ValueError, "不是 AnomaSlice 工程"):
            self.manager.load_project(str(self.path))
        self.assertEqual(self.manager.__dict__, before)

    def test_malformed_fields_preserve_active_project(self):
        documents = []
        for field, value in (("images", "part.png"), ("current_index", 4),
                             ("current_index", True), ("config", []),
                             ("annotations", []), ("normal_files", "part.png"),
                             ("masks", []), ("version", "99")):
            document = copy.deepcopy(self.document)
            document[field] = value
            documents.append(document)
        for config in ({"mode": 4}, {"stride": "128"}, {"threshold": False},
                       {"normal_dir": None}, {"label_catalog": [{"name": "划痕", "aliases": ["划痕"]}]}):
            document = copy.deepcopy(self.document)
            document["config"] = config
            documents.append(document)
        document = copy.deepcopy(self.document)
        document["annotations"] = {"当前.png": [{"label": "划痕", "bbox": [0, 0, 0, 1]}]}
        documents.append(document)
        for document in documents:
            with self.subTest(document=document):
                before = copy.deepcopy(self.manager.__dict__)
                self.write_document(document)
                with self.assertRaises(ValueError):
                    self.manager.load_project(str(self.path))
                self.assertEqual(self.manager.__dict__, before)

    def test_mask_decode_failure_does_not_commit_new_project(self):
        before = copy.deepcopy(self.manager.__dict__)
        (Path(self.directory.name) / "bad-mask.png").write_bytes(b"not an image")
        self.document["masks"] = {"当前.png": "bad-mask.png"}
        self.write_document(self.document)
        with self.assertRaisesRegex(ValueError, "无法解码工程掩膜"):
            self.manager.load_project(str(self.path))
        self.assertEqual(self.manager.__dict__, before)


if __name__ == "__main__":
    unittest.main()
