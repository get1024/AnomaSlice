"""Workflow checks spanning the actual window, canvas, and disk persistence."""

import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QMessageBox

from annotation_store import LabelCatalog
from app_window import MainWindow
from label_dialog import LabelDialog
from project_store import ProjectManager
from slicer import SlicerThread, count_mask_pixels


class AppWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.config_patch = patch.object(MainWindow, "_get_global_config_path", return_value=str(self.root / "settings.json"))
        self.config_patch.start()
        self.messages = [patch("app_window.QMessageBox.warning"), patch("app_window.QMessageBox.information"), patch("app_window.QMessageBox.critical"),
                         patch("app_window.QMessageBox.question", return_value=QMessageBox.Save)]
        for message in self.messages:
            message.start()
        self.window = MainWindow()
        self.window.show()
        self.window.activateWindow()
        self.app.processEvents()
        self.images = []
        for name in ("样本1.png", "样本2.png"):
            path = self.root / name
            image = QImage(320, 240, QImage.Format_RGB32)
            image.fill(QColor(100, 100, 100))
            self.assertTrue(image.save(str(path)))
            self.images.append(str(path))
        self.window._open_dataset_path(Path(self.images[0]))
        self.app.processEvents()

    def tearDown(self):
        self.window._boxes_dirty = False
        with patch("app_window.QMessageBox.question", return_value=QMessageBox.Discard):
            self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        for message in self.messages:
            message.stop()
        self.config_patch.stop()
        self.temp.cleanup()

    def commit(self, label):
        canvas = self.window.canvas
        canvas._pending_rect = QRectF(20, 30, 60, 70)
        canvas._open_label_editor()
        canvas._label_editor.setText(label)
        QTest.keyClick(canvas._label_editor, Qt.Key_Return)
        self.app.processEvents()

    def xml_path(self, index=0):
        return self.root / "xml" / (Path(self.images[index]).stem + ".xml")

    def edit_catalog(self, records):
        with patch("app_window.LabelDialog") as dialog:
            dialog.return_value.catalog = LabelCatalog(records)
            dialog.return_value.exec.side_effect = [True, False]
            self.window._edit_labels()

    def test_alias_save_xml_undo_redo_navigation_and_reopen(self):
        canvas = self.window.canvas
        canvas._pending_rect = QRectF(20, 30, 60, 70)
        canvas._open_label_editor()
        QTest.keyClick(canvas._label_editor, Qt.Key_2)
        self.assertFalse(canvas.has_pending_annotation())
        self.assertFalse(self.xml_path().exists())
        QTest.keyClick(self.window.canvas, Qt.Key_S, Qt.ControlModifier)
        root = ET.parse(self.xml_path()).getroot()
        node = root.find("./Detections/Detection")
        self.assertEqual(node.findtext("message"), "划痕")
        self.assertEqual([node.findtext(k) for k in ("x","y","width","height")], ["20","30","60","70"])
        self.assertEqual((root.findtext("RawCount"), root.findtext("Count")), ("0", "1"))
        self.assertFalse(self.window.canvas.has_pending_annotation())
        self.window._undo_current()
        self.window._on_save_current()
        self.assertEqual(ET.parse(self.xml_path()).findtext("Count"), "0")
        self.window._redo_current()
        self.window._on_next_image()
        self.assertEqual(ET.parse(self.xml_path()).findtext("Count"), "1")
        self.assertEqual(self.window.canvas.get_annotations(), [])
        self.commit("待定1")
        self.window._on_prev_image()
        self.assertEqual(ET.parse(self.xml_path(1)).findtext("./Detections/Detection/message"), "待定1")
        self.assertEqual(self.window.canvas.get_annotations()[0]["label"], "划痕")
        self.assertFalse(list(self.root.rglob("*.annotations.json")))

    def test_bbox_mode_is_lazy_and_slice_mode_retains_mask_workflow(self):
        self.assertIsNone(self.window.canvas.get_mask())
        self.assertFalse(self.window.slice_and_next_action.isEnabled())
        self.window.mode_combo.setCurrentIndex(1)
        self.app.processEvents()
        self.assertEqual(len(self.window.canvas.review_grid), 6)
        self.assertIsNotNone(self.window.canvas.get_mask())
        self.window.btn_defect.click()
        self.assertGreater(count_mask_pixels(self.window.canvas.get_mask()), 0)
        self.window.mode_combo.setCurrentIndex(0)
        self.assertEqual(self.window.canvas.get_annotations(), [])

    def test_fast_navigation_replaces_queued_previews_and_preserves_project_gallery(self):
        self.window.mode_combo.setCurrentIndex(1)
        normal = self.root / "Normal"
        normal.mkdir()
        paths = [normal / f"{stem}_0_0.png" for stem in ("样本1", "样本10", "样本2")]
        image = QImage(self.images[0])
        for path in paths:
            self.assertTrue(image.save(str(path)))
        self.window._refresh_gallery()
        self.window._on_next_image()
        gallery = self.window.gallery
        self.assertEqual(gallery.project_files(), ([str(paths[0]), str(paths[2])], []))
        self.app.processEvents()
        view = gallery._lists["Normal"]
        self.assertEqual([view.item(i).text() for i in range(view.count())], [paths[2].name])

    def test_same_stem_images_keep_separate_crops_and_ignore_ambiguous_legacy_files(self):
        dataset = self.root / "images"
        for folder, color in (("A", "red"), ("B", "blue")):
            image_dir = dataset / folder
            image_dir.mkdir(parents=True)
            image = QImage(32, 32, QImage.Format_RGB32)
            image.fill(QColor(color))
            self.assertTrue(image.save(str(image_dir / "样本.png")))
        self.window._open_dataset_path(dataset)
        self.window.mode_combo.setCurrentIndex(1)
        for spin in (self.window.patch_width_spin, self.window.patch_height_spin, self.window.stride_spin):
            spin.setValue(32)
        normal = dataset / "Normal"
        normal.mkdir()
        legacy = normal / "样本_0_0.png"
        self.assertTrue(image.save(str(legacy)))
        original = legacy.read_bytes()
        self.window.gallery.reset([str(legacy)])
        view = self.window.gallery._lists["Normal"]
        outputs = []
        for index, color in enumerate(("red", "blue")):
            self.window._refresh_gallery()
            self.app.processEvents()
            self.assertEqual(view.count(), 0)
            self.window.btn_no_defect.click()
            self.assertTrue(self.window._on_slice())
            self.assertTrue(self.window._slicer_thread.wait(5000))
            self.app.processEvents()
            self.app.processEvents()
            self.assertEqual(view.count(), 1)
            output = Path(view.item(0).toolTip())
            outputs.append(output)
            self.assertEqual(QImage(str(output)).pixelColor(0, 0), QColor(color))
            if index == 0:
                self.window._on_next_image()
        self.assertNotEqual(*outputs)
        self.assertEqual(QImage(str(outputs[0])).pixelColor(0, 0), QColor("red"))
        self.assertEqual(legacy.read_bytes(), original)
        self.window._on_prev_image()
        self.app.processEvents()
        self.assertEqual(view.count(), 1)
        self.assertEqual(view.item(0).toolTip(), str(outputs[0]))

    def test_opening_another_dataset_does_not_copy_old_project_state(self):
        old_image = self.images[0]
        self.window.project_manager.set_mask(old_image, QImage(old_image))
        self.window.project_manager.annotations[old_image] = [{"label": "旧框", "bbox": [0, 0, 1, 1]}]
        new_folder = self.root / "new"
        new_folder.mkdir()
        new_path = new_folder / "样本1.png"
        self.assertTrue(QImage(old_image).save(str(new_path)))
        self.window._open_dataset_path(new_folder)
        self.assertEqual(self.window.project_manager.images, [str(new_path)])
        self.assertFalse(self.window.project_manager.masks)
        self.assertFalse(self.window.project_manager.annotations)

    def test_save_failure_retains_current_document_until_retry(self):
        with patch.object(self.window._xml_document, "save", side_effect=OSError("read only")):
            self.commit("划痕")
            self.assertTrue(self.window._boxes_dirty)
            self.window.review_image_picker.setCurrentIndex(1)
            self.assertEqual(self.window._current_image_path, self.images[0])
            self.assertEqual(self.window.review_image_picker.currentIndex(), 0)
            self.assertEqual(len(self.window.canvas.get_annotations()), 1)
        self.window._on_next_image()
        self.assertEqual(self.window._current_image_path, self.images[1])
        self.assertEqual(ET.parse(self.xml_path()).findtext("Count"), "1")

    def test_slice_navigation_persists_only_edited_masks_and_retries_without_losing_them(self):
        self.window.mode_combo.setCurrentIndex(1)
        project = self.root / "review.json"
        canvas = self.window.canvas
        with patch("app_window.QFileDialog.getSaveFileName", side_effect=[("", ""), (str(project), "")]) as choose:
            self.window._on_next_image()
            self.window._on_prev_image()
            choose.assert_not_called()
            self.window.btn_defect.click()
            first_mask = canvas.get_mask()
            self.window._on_next_image()
            self.assertEqual(self.window._image_index, 0)
            self.assertTrue(canvas.has_mask_changes())
            self.window._on_next_image()
            self.assertEqual(self.window._image_index, 1)
            self.assertEqual(choose.call_count, 2)
        self.window._on_prev_image()
        self.assertEqual(canvas.get_mask(), first_mask)
        canvas.clear_mask()
        with patch("project_store.write_json_atomic", side_effect=PermissionError("locked")):
            self.window._on_next_image()
        self.assertEqual(self.window._image_index, 0)
        self.assertTrue(canvas.has_mask_changes())
        reopened = ProjectManager()
        reopened.load_project(project)
        self.assertEqual(reopened.get_mask(self.images[0]), first_mask)
        self.window._on_next_image()
        self.assertEqual(self.window._image_index, 1)
        reopened.load_project(project)
        self.assertEqual(count_mask_pixels(reopened.get_mask(self.images[0])), 0)
        self.assertIsNone(reopened.get_mask(self.images[1]))
        self.window.btn_defect.click()
        with patch("app_window.QFileDialog.getOpenFileName", return_value=(str(project), "")):
            self.window._on_open_project()
        self.window._on_next_image()
        self.assertGreater(count_mask_pixels(canvas.get_mask()), 0)
        self.window._open_dataset_path(self.root)
        self.assertEqual(self.window._image_list, self.images)

    def test_close_and_new_project_preserve_changed_mask_on_cancel(self):
        self.window.mode_combo.setCurrentIndex(1)
        self.window.btn_defect.click()
        with patch("app_window.QMessageBox.question", return_value=QMessageBox.Cancel):
            self.window._on_new_project()
            self.assertEqual(self.window._image_list, self.images)
            self.assertFalse(self.window.close())
        self.assertTrue(self.window.canvas.has_mask_changes())

    def test_navigation_keys_save_edits_and_allow_typing_a_and_d_in_labels(self):
        canvas = self.window.canvas
        canvas._pending_rect = QRectF(20, 30, 60, 70)
        canvas._open_label_editor()
        QTest.keyClicks(canvas._label_editor, "ad")
        self.assertEqual(canvas._label_editor.text(), "ad")
        self.assertEqual(self.window._image_index, 0)
        QTest.keyClick(canvas._label_editor, Qt.Key_Return)
        with patch("app_window.QMessageBox.question") as question:
            QTest.keyClick(canvas, Qt.Key_D)
            self.assertEqual(self.window._image_index, 1)
            self.assertEqual(ET.parse(self.xml_path()).findtext("./Detections/Detection/message"), "ad")
            QTest.keyClick(canvas, Qt.Key_A)
            self.assertEqual(self.window._image_index, 0)
            self.assertEqual(canvas.get_annotations()[0]["label"], "ad")
            self.assertFalse(self.xml_path(1).exists())
            question.assert_not_called()

    def test_picker_preserves_an_empty_draft_and_saves_a_named_draft(self):
        canvas = self.window.canvas
        canvas._pending_rect = QRectF(20, 30, 60, 70)
        canvas._open_label_editor()
        self.window.review_image_picker.setCurrentIndex(1)
        self.assertEqual(self.window._image_index, 0)
        self.assertEqual(self.window.review_image_picker.currentIndex(), 0)
        self.assertTrue(canvas.has_pending_annotation())
        QTest.keyClicks(canvas._label_editor, "custom")
        self.window.review_image_picker.setCurrentIndex(1)
        self.assertEqual(self.window._image_index, 1)
        self.assertEqual(ET.parse(self.xml_path()).findtext("./Detections/Detection/message"), "custom")

    def test_corrupt_sidecar_is_never_overwritten(self):
        path = Path(self.images[1] + ".annotations.json")
        path.write_text("{broken", encoding="utf-8")
        self.window._on_next_image()
        self.assertTrue(self.window._annotation_load_error)
        self.assertFalse(self.window.canvas.isEnabled())
        self.window._on_save_current()
        self.window._on_prev_image()
        self.assertEqual(path.read_text(encoding="utf-8"), "{broken")
        self.assertTrue(self.window.canvas.isEnabled())

    def test_existing_text_is_not_converted_by_new_aliases(self):
        self.edit_catalog([])
        self.commit("1")
        self.edit_catalog([{"name": "划痕", "aliases": ["1"]}])
        self.window._on_next_image()
        self.window._on_prev_image()
        self.assertFalse(self.window._annotation_load_error)
        self.assertEqual(self.window.canvas.get_annotations()[0]["label"], "1")

    def test_browsing_xml_and_legacy_sidecar_cannot_change_the_catalog(self):
        defaults = LabelCatalog.defaults().to_list()
        self.assertEqual(self.window.label_catalog.to_list(), defaults)
        self.commit("仅本图")
        self.commit("scratch")
        self.window._on_save_current()
        xml = self.xml_path().read_bytes()
        self.window.annotation_store.save(
            self.images[1], 320, 240, [{"bbox": [1, 1, 2, 2], "label": "1"}],
            [{"name": "旧目录", "aliases": ["scratch"]}],
        )
        self.window._on_next_image()
        self.assertEqual(self.window.canvas.get_annotations()[0]["label"], "1")
        self.assertEqual(self.window.label_catalog.to_list(), defaults)
        self.edit_catalog([])
        self.window._on_prev_image()
        self.assertEqual(self.window.label_catalog.to_list(), [])
        self.assertEqual([box["label"] for box in self.window.canvas.get_annotations()], ["仅本图", "划痕"])
        self.assertEqual(self.xml_path().read_bytes(), xml)
        self.window._on_next_image()
        self.assertEqual(self.window.canvas.resolve_label("scratch"), "scratch")
        reopened = MainWindow()
        try:
            self.assertEqual(reopened.label_catalog.to_list(), [])
        finally:
            reopened.close()

    def test_project_catalog_saves_to_project_and_plain_images_return_to_global(self):
        personal = [{"name": "个人分类", "aliases": ["1"]}]
        project_labels = [{"name": "工程分类", "aliases": ["1"]}]
        self.edit_catalog(personal)
        path = self.root / "project.json"
        path.write_text(json.dumps({
            "version": "1.1", "images": self.images, "current_index": 0,
            "config": {"label_catalog_version": 1, "label_catalog": project_labels},
            "annotations": {self.images[0]: [{"label": "历史标签", "bbox": [1, 1, 5, 5]}]},
        }), encoding="utf-8")
        with patch("app_window.QFileDialog.getOpenFileName", return_value=(str(path), "")):
            self.window._on_open_project()
        self.assertEqual(self.window.canvas.get_annotations()[0]["label"], "历史标签")
        self.assertEqual(self.window.label_catalog.to_list(), project_labels)
        self.edit_catalog([])
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["config"]["label_catalog"], [])
        self.window._save_global_config()
        settings = self.root / "settings.json"
        self.assertEqual(json.loads(settings.read_text(encoding="utf-8"))["label_catalog"], personal)
        self.window._open_source(self.root, Path(self.images[1]))
        self.assertEqual(self.window.label_catalog.to_list(), personal)
        self.assertEqual(self.window.canvas.get_annotations(), [])

    def test_migrating_old_settings_keeps_shortcuts_and_later_deletions(self):
        path = self.root / "settings.json"
        path.write_text(json.dumps({"label_catalog": [
            {"name": "脱落", "aliases": ["2"]}, {"name": "污渍", "aliases": ["3"]},
        ]}), encoding="utf-8")
        migrated = MainWindow()
        try:
            for alias, name in (("2", "脱落"), ("3", "污渍"), ("1", "掉漆"),
                                ("scratch", "划痕"), ("other", "其他")):
                self.assertEqual(migrated.canvas.resolve_label(alias), name)
            with patch("app_window.LabelDialog") as dialog:
                dialog.return_value.exec.side_effect = [True, False]
                dialog.return_value.catalog = LabelCatalog([])
                migrated._edit_labels()
        finally:
            migrated.close()
        # The original window predates the edit in the second window.
        with self.assertRaisesRegex(ValueError, "其他窗口"):
            self.window._save_global_config(LabelCatalog.defaults())
        self.window._save_global_config()
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual((saved["label_catalog_version"], saved["label_catalog"]), (LabelCatalog.VERSION, []))

    def test_save_button_keeps_active_field_and_releases_dialog(self):
        def edit_active_field():
            dialog = self.app.activeModalWidget()
            _, _, editor = dialog.rows[0]
            QTest.mouseClick(editor, Qt.LeftButton)
            editor.selectAll()
            QTest.keyClicks(editor, "0, drop_paint, peel")
            buttons = dialog.findChild(QDialogButtonBox)
            QTest.mouseClick(buttons.button(QDialogButtonBox.Ok), Qt.LeftButton)

        QTimer.singleShot(0, edit_active_field)
        self.window.classes_button.click()
        self.assertEqual(self.window.canvas.resolve_label("peel"), "掉漆")
        self.app.sendPostedEvents(None, QEvent.DeferredDelete)
        self.assertEqual(self.window.findChildren(LabelDialog), [])

    def test_empty_project_cannot_overwrite_previous_image(self):
        self.commit("划痕")
        self.window._on_save_current()
        before = self.xml_path().read_bytes()
        project = self.root / "empty.json"
        project.write_text(json.dumps({"version": "1.1", "images": [], "current_index": -1, "config": {}}), encoding="utf-8")
        with patch("app_window.QFileDialog.getOpenFileName", return_value=(str(project), "")):
            self.window._on_open_project()
        self.assertIsNone(self.window._current_image_path)
        self.window._on_save_current()
        self.assertEqual(self.xml_path().read_bytes(), before)

    def test_folder_and_xml_picker_continue_the_same_editable_document(self):
        part = self.root / "部件"
        (part / "xml").mkdir(parents=True)
        image = part / "现场.bmp"
        self.assertTrue(QImage(self.images[0]).save(str(image)))
        path = part / "xml" / "现场.xml"
        original = """<DefectDetection><RawCount>7</RawCount><Count>2</Count><Detections>
        <Detection index="1"><x>20</x><y>30</y><width>60</width><height>70</height><message>污渍</message><confidence>0.412300</confidence><color>#0000FF</color></Detection>
        <Detection index="2" camera="B"><x>20</x><y>30</y><width>60</width><height>70</height><message>污渍</message><confidence>0.789000</confidence><color>#0000FF</color><extra>原始记录</extra></Detection>
        </Detections></DefectDetection>"""
        path.write_text(original, encoding="utf-8")
        self.window._open_dataset_path(self.root)
        self.window.review_image_picker.setCurrentIndex(self.window._image_list.index(str(image)))
        canvas = self.window.canvas
        self.assertEqual(canvas._annotation_items[0].rect(), QRectF(20, 30, 60, 70))
        self.assertEqual(canvas._annotation_items[0].pen().color(), QColor("#0000ff"))
        listing = self.window.review_box_list
        point = listing.visualItemRect(listing.item(0)).center()
        QTest.mouseClick(listing.viewport(), Qt.LeftButton, Qt.NoModifier, point)
        self.assertEqual(canvas.selected_annotation, 0)
        QTest.keyClick(listing, Qt.Key_Delete)
        self.assertEqual(len(canvas.get_annotations()), 1)
        self.assertEqual(path.read_text(encoding="utf-8"), original)
        point = canvas.mapFromScene(QPointF(40, 50))
        QTest.mouseDClick(canvas.viewport(), Qt.LeftButton, Qt.NoModifier, point)
        canvas._label_editor.setText("复核划痕")
        QTest.keyClick(canvas._label_editor, Qt.Key_Return)
        self.commit("人工补标")
        self.window.save_xml_button.click()
        root = ET.parse(path).getroot()
        self.assertEqual((root.findtext("RawCount"), root.findtext("Count")), ("7", "2"))
        first, second = root.findall("Detections/Detection")
        self.assertEqual((first.get("index"), first.get("camera"), first.findtext("extra")), ("1", "B", "原始记录"))
        self.assertEqual((first.findtext("message"), first.findtext("confidence")), ("复核划痕", "0.789000"))
        self.assertEqual((second.get("index"), second.findtext("message")), ("2", "人工补标"))
        self.window._open_dataset_path(path)
        self.assertEqual(self.window._current_image_path, str(image))
        self.assertEqual([b["label"] for b in canvas.get_annotations()], ["复核划痕", "人工补标"])

    def test_cancel_or_discard_leaving_annotation_mode(self):
        self.commit("待审核")
        with patch("app_window.QMessageBox.question", return_value=QMessageBox.Cancel):
            self.window.mode_combo.setCurrentIndex(1)
            self.assertFalse(self.window.close())
        self.assertEqual((self.window.review_image_picker.currentIndex(), self.window.mode_combo.currentIndex()), (0, 0))
        self.assertTrue(self.window._boxes_dirty)
        with patch("app_window.QMessageBox.question", return_value=QMessageBox.Discard):
            self.window.mode_combo.setCurrentIndex(1)
        self.assertEqual(self.window.mode_combo.currentIndex(), 1)
        self.assertEqual(self.window.canvas.get_annotations(), [])
        self.assertFalse(self.xml_path().exists())

    def test_legacy_boxes_import_once_and_existing_xml_takes_precedence(self):
        boxes = [{"label": "旧标签", "bbox": [1, 2, 10, 12]}]
        legacy = Path(self.window.annotation_store.save(self.images[0], 320, 240, boxes, []))
        data = legacy.read_bytes()
        self.window._load_current_image()
        self.assertEqual(self.window.canvas.get_annotations(), boxes)
        self.window._on_save_current()
        self.window.canvas.select_annotation(0)
        self.window.canvas.delete_selected_annotation()
        self.window._on_save_current()
        self.window._load_current_image()
        self.assertEqual(self.window.canvas.get_annotations(), [])
        self.assertEqual(data, legacy.read_bytes())

    def test_slice_export_still_classifies_using_mask_fraction(self):
        normal, abnormal = self.root / "Normal", self.root / "Abnormal"
        normal.mkdir()
        abnormal.mkdir()
        image = QImage(64, 32, QImage.Format_RGB32)
        image.fill(QColor("black"))
        mask = QImage(64, 32, QImage.Format_Grayscale8)
        mask.fill(0)
        painter = QPainter(mask)
        painter.fillRect(0, 0, 32, 32, QColor("white"))
        painter.end()
        worker = SlicerThread(image, mask, 32, 32, 32, .01, str(normal), str(abnormal), False, None, "sample")
        worker.run()
        self.assertEqual([p.name for p in normal.iterdir()], ["sample_32_0.png"])
        self.assertEqual([p.name for p in abnormal.iterdir()], ["sample_0_0.png"])


if __name__ == "__main__":
    unittest.main()
