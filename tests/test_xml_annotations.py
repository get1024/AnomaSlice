"""Regression checks for local XML ownership and lossless editing of detector fields."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from PySide6.QtGui import QImage

from xml_annotations import AnnotationXml, ImageEntry, scan_folder


class AnnotationXmlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_recursive_local_pairs_duplicate_names_and_unusual_extensions(self):
        image = QImage(16, 20, QImage.Format_RGB32)
        image.fill(0)
        for unit in ("相机2", "相机10"):
            parent = self.root / unit
            (parent / "XML").mkdir(parents=True)
            self.assertTrue(image.save(str(parent / "样本.BMP")))
            (parent / "XML" / "样本.XML").touch()
        self.assertTrue(image.save(str(self.root / "无后缀"), "PNG"))
        (self.root / "说明.txt").write_text("notes")
        entries = scan_folder(self.root)
        self.assertEqual(len(entries), 3)
        paired = [e for e in entries if e.xml]
        self.assertEqual([e.image.parent.name for e in paired], ["相机2", "相机10"])
        self.assertTrue(all(e.xml.parent.parent == e.image.parent for e in paired))
        for entry in paired:
            self.assertEqual(str(entry.target()), str(entry.xml))
        collision = paired[0].image.with_suffix(".jpg")
        self.assertTrue(image.save(str(collision)))
        ambiguous = [e for e in scan_folder(self.root) if e.image.parent == collision.parent]
        for entry in ambiguous:
            with self.assertRaises(ValueError):
                AnnotationXml.open(entry)

    def test_changed_missing_or_concurrently_created_target_is_never_overwritten(self):
        entry = ImageEntry(self.root / "sample.png")
        boxes = [{"label": "污渍", "bbox": [1, 2, 3, 4]}]
        first = AnnotationXml.open(entry)
        second = AnnotationXml.open(entry)
        first.save(boxes)
        saved = first.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "其他程序"):
            second.save([])
        self.assertEqual(first.path.read_bytes(), saved)
        reopened = AnnotationXml.open(entry)
        first.path.unlink()
        with self.assertRaisesRegex(ValueError, "其他程序"):
            reopened.save([])
        self.assertEqual(list(first.path.parent.iterdir()), [])

    def test_ambiguous_or_malformed_xml_cannot_be_replaced(self):
        image = self.root / "part.jpg"
        image.touch()
        flat = image.with_suffix(".xml")
        flat.write_bytes(b"<broken")
        entry = scan_folder(self.root)[0]
        with self.assertRaisesRegex(ValueError, "XML"):
            AnnotationXml.open(entry)
        (self.root / "xml").mkdir()
        nested = self.root / "xml" / flat.name
        nested.write_bytes(b"<other/>")
        with self.assertRaisesRegex(ValueError, "同名"):
            AnnotationXml.open(scan_folder(self.root)[0])
        self.assertEqual((flat.read_bytes(), nested.read_bytes()), (b"<broken", b"<other/>"))

    def test_deleting_last_box_and_undoing_after_save_restores_original_node(self):
        path = self.root / "part.xml"
        source = b"""<?xml version="1.0" encoding="utf-8"?>
<!--original--><DefectDetection serial="S"><RawCount>5</RawCount><Count>1</Count><Detections>
<Detection index="1" camera="A"><x>02</x><y>3</y><width>4</width><height>5</height>
<message>scratch</message><confidence>0.811000</confidence><sensor/></Detection>
</Detections></DefectDetection>"""
        path.write_bytes(source)
        doc = AnnotationXml.open(ImageEntry(path.with_suffix(".png"), path))
        original = doc.boxes
        doc.save(original)
        self.assertEqual(path.read_bytes(), source)
        doc.save([])
        xml = ET.parse(path)
        self.assertEqual((xml.findtext("Count"), xml.findtext("RawCount")), ("0", "5"))
        self.assertEqual(xml.findall("Detections/Detection"), [])
        doc.save(original)
        restored = ET.parse(path).find("Detections/Detection")
        self.assertEqual((restored.get("camera"), restored.findtext("x"), restored.findtext("confidence")), ("A", "02", "0.811000"))
        self.assertIsNotNone(restored.find("sensor"))
        self.assertIn(b"<!--original-->", path.read_bytes())
