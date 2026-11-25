import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from annotation_store import AnnotationStore, AnnotationValidationError, LabelCatalog


class LabelCatalogTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"name": "划痕", "aliases": ["1", "scratch"]},
            {"name": "污渍", "aliases": ["2"]},
        ]

    def test_aliases_only_match_the_complete_stripped_token(self):
        catalog = LabelCatalog(self.records)
        self.assertEqual(catalog.resolve("1"), "划痕")
        self.assertEqual(catalog.resolve(" 1 "), "划痕")
        self.assertEqual(catalog.resolve("scratch"), "划痕")
        for text in ("待定1", "11", "1号", "Scratch", "划痕", "新缺陷"):
            self.assertEqual(catalog.resolve(text), text)

    def test_defaults_use_one_based_shortcuts_and_model_labels(self):
        expected = [
            {"name": "掉漆", "aliases": ["1", "drop_paint"]},
            {"name": "划痕", "aliases": ["2", "scratch"]},
            {"name": "污渍", "aliases": ["3", "stain"]},
            {"name": "其他", "aliases": ["4", "other"]},
        ]
        catalog = LabelCatalog.defaults()
        self.assertEqual(catalog.to_list(), expected)
        for record in expected:
            for token in record["aliases"]:
                self.assertEqual(catalog.resolve(token), record["name"])
        catalog.add_label("脱落")
        self.assertEqual(LabelCatalog.defaults().to_list(), expected)
        self.assertEqual(LabelCatalog().to_list(), [])

    def test_legacy_defaults_preserve_existing_token_owners_and_row_order(self):
        records = [
            {"name": "划痕", "aliases": ["1"]},
            {"name": "脱落", "aliases": ["2", "掉漆"]},
            {"name": "污渍", "aliases": ["3"]},
            {"name": "other", "aliases": []},
        ]
        catalog = LabelCatalog.from_legacy(records)
        self.assertEqual(catalog.to_list(), [
            {"name": "划痕", "aliases": ["1", "scratch"]},
            {"name": "脱落", "aliases": ["2", "掉漆"]},
            {"name": "污渍", "aliases": ["3", "stain"]},
            {"name": "other", "aliases": []},
            {"name": "其他", "aliases": ["4"]},
        ])
        self.assertEqual(LabelCatalog.from_legacy(catalog.to_list()).to_list(), catalog.to_list())
        self.assertEqual(LabelCatalog.from_legacy([]).to_list(), LabelCatalog.defaults().to_list())

    def test_only_untouched_old_defaults_upgrade(self):
        records = LabelCatalog.defaults().to_list()
        for i, record in enumerate(records):
            record["aliases"][0] = str(i)
        config = {"label_catalog": records}
        self.assertEqual(LabelCatalog.from_config(config, None).to_list(), LabelCatalog.defaults().to_list())
        config["label_catalog_version"] = 1
        self.assertEqual(LabelCatalog.from_config(config, None).to_list(), LabelCatalog.defaults().to_list())
        config["label_catalog_version"] = LabelCatalog.VERSION
        self.assertEqual(LabelCatalog.from_config(config, None).to_list(), records)
        records[0]["aliases"].append("peel")
        config["label_catalog_version"] = 1
        self.assertEqual(LabelCatalog.from_config(config, None).to_list(), records)

    def test_alias_conflicts_and_duplicate_names_are_rejected(self):
        bad_records = [
            [{"name": "划痕", "aliases": ["1"]}, {"name": "污渍", "aliases": ["1"]}],
            [{"name": "划痕", "aliases": ["污渍"]}, {"name": "污渍", "aliases": []}],
            [{"name": "划痕", "aliases": []}, {"name": " 划痕 ", "aliases": []}],
            [{"name": "划痕", "aliases": ["1", " 1 "]}],
            [{"name": "划痕", "aliases": ["划痕"]}],
            [{"name": "", "aliases": []}],
            [{"name": "划痕", "aliases": "1"}],
        ]
        for records in bad_records:
            with self.subTest(records=records), self.assertRaises(AnnotationValidationError):
                LabelCatalog(records)

    def test_empty_labels_rejected_and_new_labels_are_idempotent(self):
        catalog = LabelCatalog(self.records)
        for value in ("", "  ", None, 1):
            with self.subTest(value=value), self.assertRaises(AnnotationValidationError):
                catalog.resolve(value)
        self.assertEqual(catalog.add_label(" 待定1 "), "待定1")
        catalog.add_label("待定1")
        self.assertEqual(len(catalog.to_list()), 3)
        with self.assertRaises(AnnotationValidationError):
            catalog.add_label("1")

    def test_catalog_does_not_leak_mutable_references(self):
        catalog = LabelCatalog(self.records)
        self.records[0]["aliases"].append("bad-input")
        exported = catalog.to_list()
        exported[0]["aliases"].append("bad-output")
        self.assertEqual(catalog.resolve("bad-input"), "bad-input")
        self.assertNotIn("bad-output", catalog.to_list()[0]["aliases"])


class AnnotationStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.image = Path(self.directory.name) / "中文零件.png"
        self.image.write_bytes(b"image contents are not modified")
        self.store = AnnotationStore()
        self.boxes = [
            {"label": "划痕", "bbox": [0, 1, 7, 8]},
            {"label": "待定1", "bbox": [9, 9, 1, 1]},
        ]
        self.catalog = [{"name": "划痕", "aliases": ["1"]}]

    def test_missing_sidecar_returns_empty_document(self):
        loaded = self.store.load(self.image, 10, 10)
        self.assertEqual(loaded["boxes"], [])
        self.assertEqual(loaded["label_catalog"], [])
        self.assertEqual(loaded["image"], "中文零件.png")

    def test_unicode_roundtrip_retains_pixel_coordinates(self):
        path = self.store.save(self.image, 10, 10, self.boxes, self.catalog)
        self.assertEqual(path, str(self.image) + ".annotations.json")
        self.assertEqual(self.image.read_bytes(), b"image contents are not modified")
        self.assertIn("划痕", Path(path).read_text(encoding="utf-8"))
        loaded = self.store.load(self.image, 10, 10)
        self.assertEqual(loaded["boxes"], self.boxes)
        self.assertEqual(loaded["label_catalog"], self.catalog)
        self.assertEqual(loaded["version"], 1)

    def test_atomic_replace_failure_preserves_last_saved_document(self):
        path = self.store.save(self.image, 10, 10, self.boxes, self.catalog)
        before = Path(path).read_bytes()
        with patch("annotation_store.os.replace", side_effect=PermissionError("locked")):
            with self.assertRaisesRegex(OSError, "无法保存"):
                self.store.save(self.image, 10, 10, [], self.catalog)
        self.assertEqual(Path(path).read_bytes(), before)
        self.assertEqual(list(Path(self.directory.name).glob("*.tmp")), [])

    def test_boxes_can_load_independently_of_invalid_legacy_aliases(self):
        path = Path(self.store.save(self.image, 10, 10, self.boxes))
        document = json.loads(path.read_text(encoding="utf-8"))
        document["label_catalog"] = [{"name": "划痕", "aliases": ["划痕"]}]
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(AnnotationValidationError):
            self.store.load(self.image, 10, 10)
        loaded = self.store.load(self.image, 10, 10, include_catalog=False)
        self.assertEqual(loaded["boxes"], self.boxes)
        self.assertEqual(loaded["label_catalog"], [])
        document["boxes"][0]["bbox"] = [0, 0, 11, 10]
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(AnnotationValidationError, "超出原图"):
            self.store.load(self.image, 10, 10, include_catalog=False)

    def test_corrupt_file_is_reported_and_left_untouched(self):
        path = Path(self.store.sidecar_path(self.image))
        for contents in ("not JSON", "[]", '{"version": 99}', "{ "):
            with self.subTest(contents=contents):
                path.write_text(contents, encoding="utf-8")
                with self.assertRaises(AnnotationValidationError):
                    self.store.load(self.image, 10, 10)
                self.assertEqual(path.read_text(encoding="utf-8"), contents)

    def test_incompatible_image_size_and_name_are_rejected(self):
        path = self.store.save(self.image, 10, 10, self.boxes)
        with self.assertRaisesRegex(AnnotationValidationError, "尺寸"):
            self.store.load(self.image, 11, 10)
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        document["image"] = "different.png"
        Path(path).write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(AnnotationValidationError, "名称"):
            self.store.load(self.image, 10, 10)

    def test_invalid_boxes_do_not_replace_valid_saved_annotations(self):
        path = self.store.save(self.image, 10, 10, self.boxes)
        before = Path(path).read_bytes()
        for bbox in ([0, 0, 11, 1], [-1, 0, 1, 1], [0, 0, 0, 1],
                     [0, 0, 1.5, 1], [0, 0, float("nan"), 1], [False, 0, 1, 1]):
            with self.subTest(bbox=bbox), self.assertRaises(AnnotationValidationError):
                self.store.save(self.image, 10, 10, [{"label": "划痕", "bbox": bbox}])
            self.assertEqual(Path(path).read_bytes(), before)

    def test_out_of_bounds_loaded_box_is_rejected_without_clamping(self):
        path = self.store.save(self.image, 10, 10, self.boxes)
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        document["boxes"][0]["bbox"] = [9, 9, 2, 2]
        Path(path).write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(AnnotationValidationError, "超出原图"):
            self.store.load(self.image, 10, 10)

    def test_public_box_validator_checks_bounds_and_copies_records(self):
        validated = self.store.validate_boxes(self.boxes, 10, 10)
        self.assertEqual(validated, self.boxes)
        validated[0]["bbox"][0] = 1
        self.assertEqual(self.boxes[0]["bbox"][0], 0)
        with self.assertRaises(AnnotationValidationError):
            self.store.validate_boxes(self.boxes, 9, 9)


if __name__ == "__main__":
    unittest.main()
