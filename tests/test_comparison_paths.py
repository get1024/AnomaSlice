"""Exercise independent GT and prediction locations without writing predictions."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from comparison_paths import ComparisonSources


class ComparisonPathTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.images = [self.root / "1.jpg", self.root / "2.png"]
        for image in self.images:
            image.touch()
        self.near = self.root / "XML"
        self.far = self.root / "predicted"
        self.gt_dir = self.root / "verified"
        for folder in (self.near, self.far, self.gt_dir):
            folder.mkdir()
        self.local = self.near / "1.XML"
        self.other = self.far / "1.xml"
        self.local.touch()
        self.other.touch()

    def sources(self, gt=None, prediction=None):
        return ComparisonSources(self.images[0], gt, prediction, images=self.images)

    def test_usual_reversed_and_two_external_directories(self):
        external_gt = self.gt_dir / "1.xml"
        external_gt.touch()
        for gt, pred, expected in (
            (None, self.far, (self.local, self.other)),
            (self.far, None, (self.other, self.local)),
            (self.gt_dir, self.far, (external_gt, self.other)),
        ):
            with self.subTest(gt=gt, pred=pred):
                pair = self.sources(gt, pred).resolve(self.images[0])
                self.assertEqual((pair.gt, pair.prediction), expected)

    def test_missing_layers_resolve_in_their_own_directories(self):
        sources = self.sources(prediction=self.other)
        pair = sources.resolve(self.images[1])
        self.assertEqual((pair.gt, pair.prediction), (self.near / "2.xml", self.far / "2.xml"))
        self.assertFalse(pair.gt.exists())
        self.assertFalse(pair.prediction.exists())

    def test_auto_matches_beside_image_when_nested_taken_by_other_role(self):
        flat = self.images[0].with_suffix(".xml")
        flat.touch()
        pair = self.sources(prediction=self.local).resolve(self.images[0])
        self.assertEqual((pair.gt, pair.prediction), (flat, self.local))
        with self.assertRaisesRegex(ValueError, "同名"):
            self.sources(prediction=self.far).resolve(self.images[0])

    def test_explicitly_bound_file_can_have_custom_name(self):
        custom = self.gt_dir / "manual.xml"
        custom.touch()
        sources = self.sources(custom, self.other)
        self.assertEqual(sources.resolve(self.images[0]).gt, custom)
        self.assertEqual(sources.resolve(self.images[1]).gt, self.gt_dir / "2.xml")

    def test_filename_bound_to_first_image_cannot_be_claimed_by_next(self):
        bound = self.gt_dir / "2.xml"
        bound.touch()
        sources = self.sources(bound, self.far)
        self.assertEqual(sources.resolve(self.images[0]).gt, bound)
        with self.assertRaisesRegex(ValueError, "已绑定"):
            sources.resolve(self.images[1])

    def test_gt_cannot_point_to_another_images_prediction(self):
        second_prediction = self.far / "2.xml"
        second_prediction.touch()
        with self.assertRaisesRegex(ValueError, "另一张图片"):
            self.sources(second_prediction, self.far).resolve(self.images[0])
        nearby_prediction = self.near / "2.xml"
        nearby_prediction.touch()
        with self.assertRaisesRegex(ValueError, "另一张图片"):
            self.sources(nearby_prediction, None).resolve(self.images[0])

    def test_project_round_trip_preserves_anchor_and_source_kinds(self):
        custom = self.gt_dir / "manual.xml"
        custom.touch()
        sources = self.sources(custom, self.far)
        config = sources.to_config()
        restored = ComparisonSources.from_config(config, images=reversed(self.images))
        for image in self.images:
            self.assertEqual(restored.resolve(image), sources.resolve(image))
        custom.unlink()
        with self.assertRaisesRegex(ValueError, "GT 文件已移动"):
            restored.resolve(self.images[0])
        custom.mkdir()
        with self.assertRaisesRegex(ValueError, "重新选择"):
            ComparisonSources.from_config(config, images=self.images)

    def test_invalid_or_disappeared_sources_do_not_fall_back(self):
        sources = self.sources(None, self.far)
        config = sources.to_config()
        for invalid in (None, {}, {**config, "gt": {}}, {**config, "prediction": None},
                        {**config, "selected_image": "missing.jpg"}):
            with self.assertRaises(ValueError):
                ComparisonSources.from_config(invalid, images=self.images)
        self.other.unlink()
        self.far.rmdir()
        with self.assertRaises(OSError):
            sources.resolve(self.images[1])
        with self.assertRaisesRegex(ValueError, "重新选择"):
            ComparisonSources.from_config(config, images=self.images)

    def test_explicit_conflicts_duplicate_stems_and_hardlink_recheck(self):
        with self.assertRaisesRegex(ValueError, "同一个"):
            self.sources(self.local, self.local).resolve(self.images[0])
        mapped = self.sources(self.near, self.far)
        self.assertEqual(mapped.resolve(self.images[0]).gt, self.local)
        self.other.unlink()
        os.link(self.local, self.other)
        with self.assertRaisesRegex(ValueError, "同一个"):
            mapped.resolve(self.images[0])
        self.images.append(self.root / "1.png")
        with self.assertRaisesRegex(ValueError, "同名图片"):
            self.sources(self.near, self.far).resolve(self.images[0])

    def make_tree(self):
        root = self.root / "images"
        photos = [root / "line1" / "sample.png", root / "line2" / "sample.jpg"]
        for photo in photos:
            photo.parent.mkdir(parents=True)
            photo.touch()
        return root, photos

    def test_duplicate_stems_in_separate_directories_mirror_by_role(self):
        root, photos = self.make_tree()
        pairs = []
        for photo in photos:
            name = photo.with_suffix(".xml").name
            local = photo.parent / "xml" / name
            far = self.far / photo.parent.name / name
            gt = self.gt_dir / photo.parent.name / name
            for path in (local, far, gt):
                path.parent.mkdir()
                path.touch()
            pairs.append((local, far, gt))
        for gt_source, pred_source, indexes in ((None, self.far, (0, 1)),
                                                (self.gt_dir, None, (2, 0)),
                                                (self.gt_dir, self.far, (2, 1))):
            sources = ComparisonSources(photos[0], gt_source, pred_source, images=photos, image_root=root)
            restored = ComparisonSources.from_config(sources.to_config(), images=reversed(photos))
            self.assertEqual(restored.image_root, root)
            self.assertEqual(restored.source_paths_for(photos[1]), (gt_source, pred_source))
            for image, locations in zip(photos, pairs):
                pair = restored.resolve(image)
                self.assertEqual((pair.gt, pair.prediction), tuple(locations[i] for i in indexes))
                self.assertEqual(restored.scan_resolver()(image), pair)
                self.assertTrue(restored.protects_prediction(pair.prediction))
                self.assertFalse(restored.protects_prediction(pair.gt))

    def test_flat_fallback_only_for_unique_stems_and_missing_gt_uses_mirror(self):
        root, photos = self.make_tree()
        unique = photos[1].with_name("other.png")
        unique.touch()
        photos.append(unique)
        flat = self.far / "other.XML"
        flat.touch()
        sources = ComparisonSources(photos[0], self.gt_dir, self.far, images=photos, image_root=root)
        pair = sources.resolve(unique)
        self.assertEqual(pair.prediction, flat)
        self.assertEqual(pair.gt, self.gt_dir / "line2" / "other.xml")
        from xml_annotations import AnnotationXml
        AnnotationXml.open_path(pair.gt).save([])
        self.assertTrue(pair.gt.is_file())
        ambiguous = self.far / "sample.xml"
        ambiguous.touch()
        with self.assertRaisesRegex(ValueError, "多张同名图片"):
            sources.resolve(photos[0])
        self.assertTrue(sources.protects_prediction(ambiguous))
        mirrored = self.far / "line1" / "sample.xml"
        mirrored.parent.mkdir()
        mirrored.touch()
        self.assertEqual(sources.resolve(photos[0]).prediction, mirrored)

    def test_root_image_owns_its_external_xml_and_child_gets_missing_path(self):
        root, photos = self.make_tree()
        root_photo = root / "sample.bmp"
        root_photo.touch()
        flat = self.far / "sample.xml"
        flat.touch()
        photos.append(root_photo)
        sources = ComparisonSources(photos[0], None, self.far, images=photos, image_root=root)
        self.assertEqual(sources.resolve(root_photo).prediction, flat)
        self.assertEqual(sources.resolve(photos[0]).prediction, self.far / "line1" / "sample.xml")
        photos.append(photos[0].with_suffix(".bmp"))
        sources = ComparisonSources(photos[0], None, self.far, images=photos, image_root=root)
        with self.assertRaisesRegex(ValueError, "同名图片"):
            sources.resolve(photos[0])

    def test_overlapping_external_trees_cannot_assign_another_images_prediction_as_gt(self):
        root, photos = self.make_tree()
        root_photo = root / "sample.png"
        root_photo.touch()
        photos.append(root_photo)
        nested = self.far / "line1"
        nested.mkdir()
        sources = ComparisonSources(root_photo, nested, self.far, images=photos, image_root=root)
        with self.assertRaisesRegex(ValueError, "另一张图片"):
            sources.resolve(root_photo)
        (nested / "sample.xml").touch()
        with self.assertRaisesRegex(ValueError, "另一张图片"):
            sources.resolve(root_photo)

    def test_dirty_save_must_not_create_a_new_flat_or_adjacent_prediction(self):
        root, photos = self.make_tree()
        unique = photos[0]
        external = ComparisonSources(unique, None, self.far, images=[unique], image_root=root)
        self.assertTrue(external.protects_prediction(self.far / "sample.xml"))
        nearby = ComparisonSources(unique, self.gt_dir, None, images=[unique], image_root=root)
        self.assertTrue(nearby.protects_prediction(unique.with_suffix(".xml")))
        self.assertFalse(nearby.protects_prediction(self.gt_dir / "line1" / "sample.xml"))

    def test_editable_sources_and_dirty_save_check_include_aliases_and_noncurrent_images(self):
        root, photos = self.make_tree()
        custom = self.gt_dir / "manual.xml"
        custom.touch()
        prediction = self.far / "detector.xml"
        prediction.touch()
        other = self.far / "line2" / "sample.xml"
        other.parent.mkdir()
        other.touch()
        alias = self.root / "old-edits.xml"
        os.link(other, alias)
        sources = ComparisonSources(photos[0], custom, prediction, images=photos, image_root=root)
        self.assertEqual(sources.source_paths_for(photos[0]), (custom, prediction))
        self.assertEqual(sources.source_paths_for(photos[1]), (self.gt_dir, self.far))
        self.assertTrue(sources.protects_prediction(alias))
        self.assertTrue(sources.protects_prediction(prediction))
        self.assertFalse(sources.protects_prediction(custom))

    def test_check_all_ordinary_targets_in_order_with_one_prediction_scan(self):
        root, photos = self.make_tree()
        sources = ComparisonSources(photos[0], self.gt_dir, self.far, images=photos, image_root=root)
        other_prediction = self.far / "line2" / "sample.xml"
        other_prediction.parent.mkdir()
        other_prediction.touch()
        ordinary_gt = self.near / "another.xml"
        alias = self.root / "changed-role.xml"
        os.link(other_prediction, alias)
        missing_prediction = self.far / "line1" / "sample.xml"
        with patch.object(sources, "_prediction_candidates", wraps=sources._prediction_candidates) as read:
            targets = iter([ordinary_gt, alias, missing_prediction])
            self.assertEqual(sources.first_protected_target(targets), alias)
            self.assertEqual(read.call_count, len(photos))
        self.assertEqual(sources.first_protected_target([ordinary_gt, missing_prediction]), missing_prediction)
        self.assertIsNone(sources.first_protected_target([ordinary_gt]))


if __name__ == "__main__":
    unittest.main()
