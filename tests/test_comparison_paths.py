"""Exercise independent GT and prediction locations without writing predictions."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

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


if __name__ == "__main__":
    unittest.main()
