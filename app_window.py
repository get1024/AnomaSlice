"""Qt window and workflow coordination; image, project, and export services are separate."""

import os
import json
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

from PySide6.QtCore import QRectF, QSignalBlocker, Qt
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QListWidget,
    QListWidgetItem, QLabel, QLineEdit, QMainWindow, QPushButton, QCheckBox,
    QProgressDialog, QSlider, QSizePolicy, QSpinBox, QSplitter, QToolButton, QVBoxLayout,
    QWidget, QProgressBar, QGroupBox, QGridLayout, QMessageBox,
    QComboBox, QStackedWidget,
)

from annotation_store import AnnotationStore, LabelCatalog
from bbox_canvas import BoxCanvasMixin
from comparison_dialog import ComparisonDialog
from comparison_paths import ComparisonSources
from dataset_recall_label import DatasetRecallLabel
from export_dialog import ExportDialog
from export_job import ExportJob
from image_canvas import ScalableImageLabel, SliceCanvas
from label_dialog import LabelDialog
from patch_gallery import PatchGallery
from project_store import ProjectManager, json_write_lock
from json_storage import write_json_atomic
from slicer import SlicerThread
from update_dialog import UpdateDialog
from update_install import launch_update
from image_dataset import ImageEntry, scan_folder
from xml_annotations import AnnotationXml


class CanvasWidget(BoxCanvasMixin, SliceCanvas):
    """Compose rectangle interaction with the native image canvas."""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AnomaSlice — 图片标注与切图")
        self.resize(1400, 900)
        self.project_manager = ProjectManager()
        self._global_catalog = LabelCatalog.defaults()
        self.annotation_store = AnnotationStore()
        self._boxes_dirty = False
        self._annotation_load_error = False
        self._current_image_path = None
        self._current_image = None
        self._image_list = []
        self._image_index = -1
        self._image_entries = {}
        self._dataset_source = None
        self._xml_document = None
        self._comparison = None
        self._slicer_thread = None
        self._export_thread = None
        self._pending_update = None
        self._update_dialog = None
        self._current_patch_index = 0
        self._create_actions()
        self._build_ui()
        self.canvas.annotations_changed = self._on_annotations_changed
        self.canvas.selection_changed = self._sync_review_selection
        self._clear_annotation_state()
        self._load_global_config()
        self._set_catalog(self._global_catalog)
        self._set_mode(0)


    def _create_actions(self):
        # File Actions
        self.new_project_action = QAction("新建工程", self)
        self.new_project_action.triggered.connect(self._on_new_project)

        self.open_project_action = QAction("打开工程...", self)
        self.open_project_action.setShortcut("Ctrl+O")
        self.open_project_action.triggered.connect(self._on_open_project)

        self.save_project_action = QAction("保存工程", self)
        self.save_project_action.setShortcut("Ctrl+S")
        self.save_project_action.triggered.connect(self._on_save_current)

        self.save_project_as_action = QAction("工程另存为...", self)
        self.save_project_as_action.triggered.connect(self._on_save_project_as)

        # Edit Actions
        self.undo_action = QAction("撤销", self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.triggered.connect(self._undo_current)

        self.redo_action = QAction("重做", self)
        self.redo_action.setShortcut("Ctrl+Shift+Z")
        self.redo_action.triggered.connect(self._redo_current)

        # Initialize Canvas early for signal connections
        self.canvas = CanvasWidget()

        self.open_dataset_action = QAction("打开数据集…", self)
        self.open_dataset_action.setShortcut("Ctrl+I")
        self.open_dataset_action.triggered.connect(self._open_dataset)

        self.export_action = QAction("导出图片或缺陷裁剪图…", self)
        self.export_action.setEnabled(False)
        self.export_action.triggered.connect(self._export_images)

        self.slice_and_next_action = QAction("Slice and Next", self)
        self.slice_and_next_action.setShortcut("Return")
        self.slice_and_next_action.triggered.connect(self._on_slice_and_next)

        self.addAction(self.slice_and_next_action)

        for caption, key, navigate in (("上一张", "A", self._on_prev_image),
                                       ("下一张", "D", self._on_next_image)):
            action = QAction(caption, self)
            action.setShortcut(key)
            action.setAutoRepeat(False)
            action.triggered.connect(navigate)
            self.addAction(action)

    def _build_ui(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        file_menu.addAction(self.open_dataset_action)
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(self.new_project_action)
        file_menu.addAction(self.open_project_action)
        file_menu.addAction(self.save_project_action)
        file_menu.addAction(self.save_project_as_action)
        file_menu.addSeparator()

        edit_menu = menubar.addMenu("编辑")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)

        help_menu = menubar.addMenu("帮助")
        self.update_action = help_menu.addAction("检查更新…")
        self.update_action.setMenuRole(QAction.NoRole)
        self.update_action.triggered.connect(self._check_for_updates)

        # Main Layout with Splitter
        self.main_splitter = QSplitter(Qt.Horizontal)

        # 1. Left Panel
        self.control_panel = self._build_control_panel()
        self.annotation_panel = self._build_annotation_panel()
        self.left_stack = QStackedWidget()
        self.left_stack.addWidget(self.annotation_panel)
        self.left_stack.addWidget(self.control_panel)

        # 2. Center Panel (Patch Viewer Only)
        self.patch_viewer = self._build_patch_viewer()
        self.patch_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.annotation_view = QWidget()
        self._annotation_canvas_layout = QVBoxLayout(self.annotation_view)
        self._annotation_canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.center_stack = QStackedWidget()
        self.center_stack.addWidget(self.annotation_view)
        self.center_stack.addWidget(self.patch_viewer)

        # 3. Gallery (Right)
        self.gallery = PatchGallery()
        self.gallery.patch_selected.connect(self._highlight_patch)

        self.main_splitter.addWidget(self.left_stack)
        self.main_splitter.addWidget(self.center_stack)
        self.main_splitter.addWidget(self.gallery)

        # Set stretch factors for main splitter
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)

        # Set initial sizes
        self.main_splitter.setSizes([350, 1000, 320])

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 6, 8, 6)
        top = QHBoxLayout()
        top.addWidget(QLabel("工作模式"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["框选标注", "切图分类"])
        self.mode_combo.currentIndexChanged.connect(self._set_mode)
        top.addWidget(self.mode_combo)
        self.dataset_recall = DatasetRecallLabel()
        top.addWidget(self.dataset_recall)
        self.recall_rule = QComboBox()
        self.recall_rule.setAccessibleName("检出率位置匹配条件")
        self.recall_rule.setToolTip("选择同图可编辑标签框与只读对照框的位置匹配条件")
        self.recall_rule.addItem("有交集", 0.0)
        self.recall_rule.addItem("小框覆盖率 ≥ 50%", 0.5)
        self.recall_rule.currentIndexChanged.connect(self._on_recall_rule_changed)
        top.addWidget(self.recall_rule)
        top.addStretch()
        self.image_position_label = QLabel("打开一张图片后，可继续浏览同文件夹中的图片")
        top.addWidget(self.image_position_label)
        export_button = QToolButton()
        export_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        export_button.setDefaultAction(self.export_action)
        top.addWidget(export_button)
        layout.addLayout(top)
        layout.addWidget(self.main_splitter, 1)
        self.setCentralWidget(container)
        self.statusBar().showMessage("就绪")

    def _build_annotation_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(240)
        panel.setMaximumWidth(330)
        layout = QVBoxLayout(panel)
        title = QLabel("人工框选标注")
        title.setStyleSheet("font-weight: bold; font-size: 18px;")
        layout.addWidget(title)
        native_text = QKeySequence.SequenceFormat.NativeText
        open_key = self.open_dataset_action.shortcut().toString(native_text)
        open_button = QPushButton(f"打开数据集…  ({open_key})")
        open_button.clicked.connect(self.open_dataset_action.trigger)
        layout.addWidget(open_button)
        self.annotation_image_label = QLabel("尚未打开图片")
        self.annotation_image_label.setWordWrap(True)
        self.annotation_image_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.annotation_image_label)
        self.review_image_picker = QComboBox()
        self.review_image_picker.setAccessibleName("当前图片")
        self.review_image_picker.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.review_image_picker.currentIndexChanged.connect(self._go_to_image)
        layout.addWidget(self.review_image_picker)
        self.classes_button = QPushButton("缺陷种类与别名…")
        self.classes_button.clicked.connect(self._edit_labels)
        layout.addWidget(self.classes_button)
        self.catalog_label = QLabel("可直接输入标签，也可先设置种类别名。")
        self.catalog_label.setTextFormat(Qt.PlainText)
        self.catalog_label.setWordWrap(True)
        layout.addWidget(self.catalog_label)
        self.box_count_label = QLabel("标注框：0")
        layout.addWidget(self.box_count_label)
        self.review_box_list = QListWidget()
        self.review_box_list.setAccessibleName("当前图片的缺陷框")
        self.review_box_list.setMaximumHeight(180)
        self.review_box_list.currentRowChanged.connect(self.canvas.select_annotation)
        delete_from_list = QAction(self.review_box_list)
        delete_keys = ["Backspace", "Delete"] if sys.platform == "darwin" else ["Delete"]
        delete_from_list.setShortcuts([QKeySequence(key) for key in delete_keys])
        delete_from_list.setShortcutContext(Qt.WidgetShortcut)
        delete_from_list.triggered.connect(self.canvas.delete_selected_annotation)
        self.review_box_list.addAction(delete_from_list)
        layout.addWidget(self.review_box_list)
        self.prediction_status = QLabel()
        self.prediction_status.setTextFormat(Qt.PlainText)
        self.prediction_status.setWordWrap(True)
        layout.addWidget(self.prediction_status)
        self.annotation_save_label = QLabel()
        self.annotation_save_label.setWordWrap(True)
        layout.addWidget(self.annotation_save_label)
        fit = QPushButton("适应窗口")
        fit.clicked.connect(self.canvas.fit_image)
        layout.addWidget(fit)
        delete_key = delete_from_list.shortcut().toString(native_text) if sys.platform == "darwin" else "Delete"
        self.delete_box_button = QPushButton(f"删除所选框  ({delete_key})")
        self.delete_box_button.clicked.connect(self.canvas.delete_selected_annotation)
        layout.addWidget(self.delete_box_button)
        undo_row = QHBoxLayout()
        undo = QPushButton("撤回")
        undo.clicked.connect(self._undo_current)
        redo = QPushButton("重做")
        redo.clicked.connect(self._redo_current)
        undo_row.addWidget(undo)
        undo_row.addWidget(redo)
        layout.addLayout(undo_row)
        save_key = self.save_project_action.shortcut().toString(native_text)
        self.save_xml_button = QPushButton(f"保存可编辑标签  ({save_key})")
        self.save_xml_button.clicked.connect(self._on_save_current)
        layout.addWidget(self.save_xml_button)
        layout.addStretch()
        undo_key = self.undo_action.shortcut().toString(native_text)
        redo_key = self.redo_action.shortcut().toString(native_text)
        navigation_help = ("触控板：双指滑动平移，双指捏合或张开缩放，三指拖动画框。"
                           if sys.platform == "darwin" else "滚轮缩放，右键拖动平移。")
        self.annotation_help = QLabel(
            f"左键拖动画框，别名自动确认，自由文本按 Enter。\n单击选框，双击改标签，{delete_key} 删除。\n"
            f"{navigation_help}\n{undo_key} 撤回，{redo_key} 重做。"
        )
        self.annotation_help.setWordWrap(True)
        layout.addWidget(self.annotation_help)
        navigation = QHBoxLayout()
        previous = QPushButton("上一张  (A)")
        previous.clicked.connect(self._on_prev_image)
        following = QPushButton("下一张  (D)")
        following.clicked.connect(self._on_next_image)
        navigation.addWidget(previous)
        navigation.addWidget(following)
        layout.addLayout(navigation)
        return panel

    def _set_mode(self, index):
        annotation = index == 0
        if not annotation and not self._settle_xml_changes():
            with QSignalBlocker(self.mode_combo):
                self.mode_combo.setCurrentIndex(0)
            return
        self.canvas.cancel_pending_annotation()
        if annotation:
            self._annotation_canvas_layout.addWidget(self.canvas)
        else:
            self._slice_canvas_layout.insertWidget(self._slice_canvas_index, self.canvas, 1)
        self.left_stack.setCurrentIndex(index)
        self.center_stack.setCurrentIndex(index)
        self.gallery.setVisible(not annotation)
        self.canvas.set_annotation_mode(annotation)
        self._show_dataset_metrics(annotation)
        self.save_project_action.setText("保存可编辑标签" if annotation else "保存工程")
        if not annotation:
            self.canvas.ensure_slice_mask()
        self.slice_and_next_action.setEnabled(not annotation)
        self.main_splitter.setSizes([270 if annotation else 350, 1000, 0 if annotation else 320])
        self.canvas.fit_image()
        if not annotation:
            self._init_patch_review()
        self._refresh_gallery()

    def _undo_current(self):
        focus = QApplication.focusWidget()
        if isinstance(focus, QLineEdit):
            focus.undo()
        elif self.mode_combo.currentIndex() == 0:
            self.canvas.annotation_undo()
        else:
            self.canvas.undo()

    def _redo_current(self):
        focus = QApplication.focusWidget()
        if isinstance(focus, QLineEdit):
            focus.redo()
        elif self.mode_combo.currentIndex() == 0:
            self.canvas.annotation_redo()
        else:
            self.canvas.redo()

    def _on_save_current(self):
        if self.mode_combo.currentIndex() == 0:
            if self.canvas.has_pending_annotation():
                self.statusBar().showMessage("请按 Enter 确认标签，或按 Esc 取消当前框。", 5000)
                return
            self._save_xml()
        else:
            self._on_save_project()

    @property
    def label_catalog(self):
        return self.canvas.label_catalog

    def _set_catalog(self, catalog):
        self.canvas.label_catalog = catalog
        descriptions = []
        for record in self.label_catalog.to_list():
            aliases = ", ".join(record["aliases"])
            descriptions.append(f'{record["name"]}  ←  {aliases}' if aliases else record["name"])
        self.catalog_label.setText("\n".join(descriptions) or "可直接输入标签，也可先设置种类别名。")

    def _edit_labels(self):
        dialog = LabelDialog(self.label_catalog, self)
        project = self.project_manager.project_path
        dialog.setWindowTitle("缺陷种类与别名 · " + (Path(project).name if project else "全局设置"))
        while dialog.exec():
            try:
                if project:
                    self.project_manager.save_label_catalog(dialog.catalog)
                else:
                    self._save_global_config(dialog.catalog)
                    self._global_catalog = dialog.catalog
            except (OSError, ValueError) as error:
                QMessageBox.warning(self, "类别表尚未保存", str(error))
                continue
            self._set_catalog(dialog.catalog)
            break
        dialog.deleteLater()

    def _on_annotations_changed(self, boxes):
        if self._xml_document is None:
            return
        self._boxes_dirty = boxes != self._xml_document.boxes
        self._show_boxes(boxes)
        self.annotation_save_label.setText("有未保存的 XML 修改。" if self._boxes_dirty else self._xml_status())

    def _xml_status(self):
        doc = self._xml_document
        if doc.exists:
            return f"可编辑标签 · {doc.path.name}"
        if doc.boxes:
            return "已载入旧标注，保存后生成 XML。"
        return f"保存后新建可编辑标签 · {doc.path.name}"

    def _save_xml(self):
        doc = self._xml_document
        if doc is None:
            return True
        try:
            image = Path(self._current_image_path).resolve()
            target = (self._comparison.resolve(image).gt if self._comparison
                      else self._image_entries[image].target())
            if target != doc.path:
                raise ValueError("图片的同名 XML 已变化，请重新打开图片。")
            doc.save(self.canvas.get_annotations())
        except (OSError, ValueError) as error:
            self.annotation_save_label.setText(f"XML 保存失败：{error}")
            QMessageBox.warning(self, "XML 尚未保存", str(error))
            return False
        self._boxes_dirty = False
        self.save_xml_button.setEnabled(False)
        self.annotation_save_label.setText(f"可编辑标签已保存，当前 {len(doc.boxes)} 个框。")
        return True

    def _settle_xml_changes(self, *, protect_current=False, autosave=False):
        if not self._boxes_dirty:
            return True
        if protect_current:
            answer = QMessageBox.question(
                self, "对照 XML 有未保存修改",
                "当前编辑的文件已被选作只读对照标签。放弃这些修改并开始对照？",
                QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Cancel,
            )
        elif autosave:
            return self._save_xml()
        else:
            answer = QMessageBox.question(
                self, "保存标注修改", f"将当前修改写入 {self._xml_document.path.name}？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save,
            )
        if answer == QMessageBox.Save:
            return self._save_xml()
        if answer == QMessageBox.Discard:
            self._boxes_dirty = False
            self.canvas.set_annotations(self._xml_document.boxes)
            self._show_boxes(self._xml_document.boxes)
            self.annotation_save_label.setText(self._xml_status())
            return True
        return False

    def _prepare_image_change(self, *, protect_current=False, autosave=False):
        if self._slicer_thread is not None and self._slicer_thread.isRunning():
            self.statusBar().showMessage("正在保存切图，请完成后再切换图片。", 5000)
            return False
        if autosave and self.canvas.has_pending_annotation() and not self.canvas.commit_pending_annotation():
            self.statusBar().showMessage("请填写标签，或按 Esc 取消当前框后切换图片。", 5000)
            return False
        if not self._settle_xml_changes(protect_current=protect_current, autosave=autosave):
            return False
        self.canvas.cancel_pending_annotation()
        if autosave and self._current_image_path and self.canvas.has_mask_changes():
            if self.project_manager.project_path:
                try:
                    self.project_manager.save_image_mask(self._current_image_path, self.canvas.get_mask())
                except (OSError, ValueError) as error:
                    QMessageBox.warning(self, "切图掩膜尚未保存", str(error))
                    return False
                self.canvas.mark_mask_saved()
            else:
                path = self._choose_project_path()
                return bool(path) and self._write_project(path, notify=False)
        return True

    def _load_annotations(self, path, width, height):
        self._clear_annotation_state()
        try:
            entry = ImageEntry.lookup(path, self._image_entries)
            if self._comparison:
                pair = self._comparison.resolve(path)
                self._load_prediction(pair.prediction)
                doc = AnnotationXml.open_path(pair.gt)
            else:
                doc = AnnotationXml.for_image(entry, width, height, self.project_manager.annotations.get(path, []))
            self._xml_document = doc
            self.canvas.set_annotations(doc.boxes)
            self._show_boxes(doc.boxes)
            self.annotation_save_label.setText(self._xml_status())
            self.annotation_save_label.setToolTip(str(doc.path))
        except (OSError, ValueError) as error:
            self._annotation_load_error = True
            if self._comparison:
                self.dataset_recall.image_error(path, error)
            self.canvas.setEnabled(False)
            self.annotation_save_label.setText(f"无法编辑标注：{error}")
            self.annotation_save_label.setToolTip(str(error))

    def _build_patch_viewer(self):
        container = QWidget()
        layout = QVBoxLayout(container)

        # Title
        title = QLabel("单元检视 (Unit Review)")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Patch Display
        self.patch_display_label = ScalableImageLabel()
        self.patch_display_label.setAlignment(Qt.AlignCenter)
        self.patch_display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.patch_display_label.setStyleSheet("border: 2px dashed #444; background-color: #111;")
        layout.addWidget(self.patch_display_label, 1) # Stretch 1

        # Info
        self.patch_info_label = QLabel("进度: -/-")
        self.patch_info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.patch_info_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_no_defect = QPushButton("无缺陷")
        self.btn_defect = QPushButton("有缺陷")

        self.btn_no_defect.setStyleSheet("background-color: #388E3C; color: white; font-weight: bold; padding: 8px;")
        self.btn_defect.setStyleSheet("background-color: #D32F2F; color: white; font-weight: bold; padding: 8px;")

        self.btn_no_defect.clicked.connect(lambda: self._record_patch_annotation(False))
        self.btn_defect.clicked.connect(lambda: self._record_patch_annotation(True))

        btn_layout.addWidget(self.btn_no_defect)
        btn_layout.addWidget(self.btn_defect)
        layout.addLayout(btn_layout)

        # Navigation
        nav_layout = QHBoxLayout()

        self.prev_image_button = QPushButton("上一张大图  (A)")
        self.prev_image_button.clicked.connect(self._on_prev_image)
        nav_layout.addWidget(self.prev_image_button)

        self.next_button = QPushButton("下一张大图  (D)")
        self.next_button.clicked.connect(self._on_next_image)
        nav_layout.addWidget(self.next_button)
        layout.addLayout(nav_layout)

        layout.addStretch()
        return container

    def _build_control_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(300)
        layout = QVBoxLayout(panel)
        layout.setSpacing(10) # Increase spacing slightly

        def add_path_row(title, category=None):
            caption = QLabel(title)
            caption.setStyleSheet("color: #aaa; font-weight: bold; font-size: 11px;")
            layout.addWidget(caption)
            row = QHBoxLayout()
            label = QLabel("未选择")
            label.setWordWrap(True)
            label.setStyleSheet("color: #ddd; border: 1px solid #444; border-radius: 2px; padding: 2px; font-size: 12px;")
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            row.addWidget(label, 1)
            if category:
                browse = QPushButton("...")
                browse.setFixedWidth(30)
                browse.clicked.connect(lambda: self._select_output_dir(category, label))
                row.addWidget(browse)
            layout.addLayout(row)
            return label

        self.source_path_label = add_path_row("原始图源路径")
        self.normal_path_label = add_path_row("Normal 输出路径", "Normal")
        self.abnormal_path_label = add_path_row("Abnormal 输出路径", "Abnormal")

        # Parameters Group (Compact)
        param_group = QGroupBox("参数设置")
        param_layout = QGridLayout(param_group)
        param_layout.setContentsMargins(5, 5, 5, 5)
        param_layout.setSpacing(5)

        self.patch_width_spin = QSpinBox()
        self.patch_height_spin = QSpinBox()
        self.stride_spin = QSpinBox()
        for spin in (self.patch_width_spin, self.patch_height_spin, self.stride_spin):
            spin.setRange(1, 8192)
            spin.setValue(256)
            spin.setKeyboardTracking(False)
        self.stride_spin.setValue(128)

        self.stride_spin.valueChanged.connect(self._update_grid_params)

        self.smart_grid_checkbox = QCheckBox("智能网格")
        self.smart_grid_checkbox.setChecked(False)
        self.smart_grid_checkbox.stateChanged.connect(self._on_smart_grid_changed)

        # Row 0: Patch Size
        param_layout.addWidget(QLabel("尺寸(WxH):"), 0, 0)
        size_layout = QHBoxLayout()
        size_layout.addWidget(self.patch_width_spin)
        size_layout.addWidget(QLabel("x"))
        size_layout.addWidget(self.patch_height_spin)
        param_layout.addLayout(size_layout, 0, 1)

        # Row 1: Stride & Threshold
        param_layout.addWidget(QLabel("步长:"), 1, 0)
        param_layout.addWidget(self.stride_spin, 1, 1)

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 100)
        self.threshold_spin.setValue(1)
        param_layout.addWidget(QLabel("阈值(%):"), 2, 0)
        param_layout.addWidget(self.threshold_spin, 2, 1)

        # Row 3: Smart Grid & Patch Count
        param_layout.addWidget(self.smart_grid_checkbox, 3, 0, 1, 2)

        self.patch_count_label = QLabel("预计: -")
        self.patch_count_label.setStyleSheet("color: #888; font-weight: bold;")
        param_layout.addWidget(self.patch_count_label, 4, 0, 1, 2)

        layout.addWidget(param_group, 0) # No stretch

        # Mask Opacity
        opacity_layout = QHBoxLayout()
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        self.mask_opacity_slider = QSlider(Qt.Horizontal)
        self.mask_opacity_slider.setRange(0, 100)
        self.mask_opacity_slider.setValue(50)
        self.mask_opacity_slider.valueChanged.connect(self._on_mask_opacity_changed)
        opacity_layout.addWidget(QLabel("透明度:"))
        opacity_layout.addWidget(self.mask_opacity_slider)
        layout.addLayout(opacity_layout)

        # Actions Group (Compact)
        action_group = QGroupBox("操作")
        action_layout = QGridLayout(action_group)
        action_layout.setContentsMargins(5, 5, 5, 5)
        action_layout.setSpacing(5)

        self.open_image_btn = QPushButton("打开数据集…")
        self.open_image_btn.clicked.connect(self.open_dataset_action.trigger)

        self.clear_mask_button = QPushButton("清除")
        self.clear_mask_button.clicked.connect(self.canvas.clear_mask)

        action_layout.addWidget(self.open_image_btn, 0, 0)
        action_layout.addWidget(self.clear_mask_button, 0, 1)

        self.slice_button = QPushButton("切分")
        self.slice_button.clicked.connect(self._on_slice)

        action_layout.addWidget(self.slice_button, 1, 0, 1, 2)

        layout.addWidget(action_group, 0)

        # Mini Map (Canvas)
        # Add a label for clarity
        self.current_image_label = QLabel("当前图片: -")
        self.current_image_label.setStyleSheet("color: #FFD700; font-weight: bold; font-size: 14px;") # Gold color
        self.current_image_label.setAlignment(Qt.AlignCenter)
        self.current_image_label.setWordWrap(True)
        layout.addWidget(self.current_image_label)

        layout.addWidget(QLabel("检视窗格导航 (点击方格可跳转):"))

        # Configure Canvas to be small but usable
        self.canvas.setMinimumSize(250, 250)
        self.canvas.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.canvas.canvas_clicked.connect(self._on_canvas_clicked)

        layout.addWidget(self.canvas, 1) # Give it some stretch space
        self._slice_canvas_layout = layout
        self._slice_canvas_index = layout.indexOf(self.canvas)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar, 0)

        return panel

    def _highlight_patch(self, x, y):
        self.canvas.set_highlight_rect(QRectF(x, y, self.patch_width_spin.value(), self.patch_height_spin.value()))

    def _select_output_dir(self, category, label):
        folder = QFileDialog.getExistingDirectory(self, f"选择 {category} 输出文件夹", label.text())
        if folder:
            directory = os.path.normpath(folder)
            if category == "Normal":
                self._set_output_dirs(directory, self.project_manager.abnormal_dir)
            else:
                self._set_output_dirs(self.project_manager.normal_dir, directory)
            self._refresh_gallery()

    def _set_output_dirs(self, normal, abnormal):
        # Older projects could store the UI's empty-path placeholder.
        self.project_manager.normal_dir = normal if normal and normal != "未选择" else None
        self.project_manager.abnormal_dir = abnormal if abnormal and abnormal != "未选择" else None
        self.normal_path_label.setText(self.project_manager.normal_dir or "未选择")
        self.abnormal_path_label.setText(self.project_manager.abnormal_dir or "未选择")

    def _refresh_gallery(self):
        stem = (self.project_manager.patch_stem(self._current_image_path)
                if self.mode_combo.currentIndex() == 1 and self._current_image_path else None)
        self.gallery.show_image(stem, self.project_manager.normal_dir,
                                self.project_manager.abnormal_dir)

    def _clear_annotation_state(self):
        self._xml_document = None
        self._boxes_dirty = False
        self._annotation_load_error = False
        self.canvas.setEnabled(True)
        self._prediction_boxes = None
        self.canvas.set_predictions([])
        self.prediction_status.clear()
        self.prediction_status.setToolTip("")
        self.prediction_status.hide()
        self.canvas.set_annotations([])
        self._show_boxes([])
        self.annotation_save_label.setText("打开图片或文件夹开始标注。")
        self.annotation_save_label.setToolTip("")

    def _sync_review_selection(self, index):
        with QSignalBlocker(self.review_box_list):
            self.review_box_list.setCurrentRow(-1 if index is None else index)
        self.delete_box_button.setEnabled(index is not None)

    def _show_boxes(self, boxes):
        with QSignalBlocker(self.review_box_list):
            self.review_box_list.clear()
            for number, box in enumerate(boxes, 1):
                caption = f'{number} · {box["label"]}'
                if box.get("confidence"):
                    caption += f'    {box["confidence"]}'
                item = QListWidgetItem(caption)
                x, y, w, h = box["bbox"]
                item.setToolTip(f"左上角 ({x:g}, {y:g})　宽 {w:g}　高 {h:g}")
                self.review_box_list.addItem(item)
        self._sync_review_selection(self.canvas.selected_annotation)
        self.box_count_label.setText(f"可编辑标签（实线）：{len(boxes)}")
        doc = self._xml_document
        self.save_xml_button.setEnabled(doc is not None and (self._boxes_dirty or not doc.exists))
        if self._comparison and doc is not None and self._prediction_boxes is not None:
            self.dataset_recall.update_image(self._current_image_path, boxes, self._prediction_boxes)

    def _load_prediction(self, path):
        self.prediction_status.show()
        self.prediction_status.setToolTip(str(path))
        try:
            boxes = AnnotationXml.open_path(path, must_exist=True).boxes
        except FileNotFoundError:
            self._prediction_boxes = []
            self.prediction_status.setText("只读对照标签 · 本图无匹配 XML")
            return
        except (OSError, ValueError) as error:
            self.dataset_recall.image_error(self._current_image_path, error)
            self.prediction_status.setText(f"只读对照标签读取失败：{error}")
            return
        self._prediction_boxes = boxes
        self.canvas.set_predictions(boxes)
        self.prediction_status.setText(f"只读对照标签（青色虚线）：{len(boxes)}")

    def _open_dataset(self):
        active = bool(self._image_list)
        anchor = Path(self._image_list[self._image_index]).resolve() if active else None
        gt = prediction = None
        if self._comparison:
            anchor = self._comparison.selected_image
            gt, prediction = self._comparison.source_paths_for(anchor)
        elif self._xml_document and self._xml_document.exists:
            path = self._xml_document.path
            if path.parent not in (anchor.parent, anchor.parent / "xml"):
                gt = path.parent if path.stem.casefold() == anchor.stem.casefold() else path
        dialog = ComparisonDialog(self._dataset_source, self, gt_source=gt, prediction_source=prediction,
                                  active_dataset=active, binding_image=anchor)
        while dialog.exec():
            try:
                source = dialog.image_path
                roles = (dialog.gt_source, dialog.prediction_source)
                if active and source == self._dataset_source:
                    if roles != (gt, prediction):
                        self._change_xml_sources(anchor, roles)
                else:
                    self._open_dataset_path(source, roles)
            except (OSError, ValueError) as error:
                QMessageBox.warning(self, "无法打开数据集", str(error))
                continue
            break
        dialog.deleteLater()

    def _open_dataset_path(self, source, roles=(None, None)):
        selected = None if source.is_dir() else source
        root = source if selected is None else source.parent
        if selected and selected.suffix.casefold() == ".xml" and root.name.casefold() == "xml":
            root = root.parent
        self._open_source(root, selected, comparison_sources=roles if any(roles) else None)

    def _protect_pending_prediction(self, comparison):
        return bool(self._xml_document and comparison and comparison.protects_prediction(self._xml_document.path))

    def _plain_gt_targets(self):
        for path in self._image_list:
            image = Path(path).resolve()
            entry = self._image_entries.get(image) or ImageEntry(image)
            try:
                yield entry.target()
            except ValueError:
                continue  # Ambiguous entries cannot be edited in plain mode.

    def _change_xml_sources(self, anchor, roles):
        current = Path(self._image_list[self._image_index]).resolve()
        comparison = None
        if any(roles):
            root = self._dataset_source if self._dataset_source.is_dir() else None
            comparison = ComparisonSources(anchor, *roles, images=self._image_list, image_root=root)
            pair = comparison.resolve(current)
            target = pair.gt
        else:
            entry = self._image_entries.get(current)
            target = entry.target() if entry else None
            if self._comparison:
                conflict = self._comparison.first_protected_target(self._plain_gt_targets())
                if conflict:
                    raise ValueError(f"自动匹配会将只读对照文件 {conflict} 用作可编辑标签。请指定可编辑标签来源。")
        protect = self._protect_pending_prediction(comparison)
        keep_gt = self._xml_document is not None and self._xml_document.path == target and not protect
        if not keep_gt and not self._prepare_image_change(protect_current=protect):
            return
        self._comparison = comparison
        self.mode_combo.setCurrentIndex(0)
        self._start_dataset_metrics()
        if keep_gt:
            # Replacing the overlay preserves the editable document, undo stack and zoom.
            self.canvas.set_predictions([])
            self._prediction_boxes = None
            self.prediction_status.hide()
            if comparison:
                self._load_prediction(pair.prediction)
            self._show_boxes(self.canvas.get_annotations())
        elif self._current_image is not None:
            self._load_annotations(self._current_image_path, self._current_image.width(), self._current_image.height())
        else:
            self._load_current_image()

    def _populate_image_picker(self, root=None):
        self.export_action.setEnabled(bool(self._image_list) and self._export_thread is None)
        with QSignalBlocker(self.review_image_picker):
            self.review_image_picker.clear()
            for path in self._image_list:
                name = str(Path(path).relative_to(root)) if root else path
                self.review_image_picker.addItem(name)
            self.review_image_picker.setCurrentIndex(self._image_index)

    def _export_images(self):
        if not self._image_list or self._export_thread is not None:
            return
        if self.canvas.has_pending_annotation():
            QMessageBox.information(self, "请先确认当前框", "请按 Enter 确认标签或按 Esc 取消，再导出图片。")
            return
        gt = self.canvas.get_annotations() if self._xml_document is not None else None
        predictions = self._prediction_boxes
        dialog = ExportDialog(
            self, gt_boxes=gt or [], prediction_boxes=(predictions or []) if self._comparison else None,
            labels=[entry["name"] for entry in self.label_catalog.to_list()],
            selected_gt=self.canvas.selected_annotation,
        )
        accepted = dialog.exec()
        options = dialog.options
        dialog.deleteLater()
        if not accepted:
            return
        worker = ExportJob(options, self._image_list, self._image_index, current_image=self._current_image,
                           current_gt=gt, current_prediction=predictions, comparison=self._comparison,
                           entries=self._image_entries, project_boxes=self.project_manager.annotations, parent=self)
        self._export_thread = worker
        self.export_action.setEnabled(False)
        progress = QProgressDialog("正在导出图片…", "取消", 0, len(worker.images), self)
        self._export_progress = progress
        progress.setWindowTitle("导出标注图片")
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        progress.canceled.connect(worker.requestInterruption)
        worker.progress.connect(progress.setValue)
        worker.finished.connect(self._export_finished)
        progress.show()
        worker.start()

    def _export_finished(self):
        worker = self._export_thread
        self._export_progress.hide()
        self._export_progress.deleteLater()
        self._export_thread = None
        self.export_action.setEnabled(bool(self._image_list))
        title = "导出已取消" if worker.cancelled else "导出结果"
        message = f"已导出 {worker.written} 个 PNG 文件。"
        if worker.directory:
            message += f"\n{worker.directory}"
        elif not worker.errors and not worker.cancelled:
            message += "\n所选范围内没有可裁剪的缺陷框。"
        if worker.errors:
            if worker.skipped:
                message += f"\n跳过 {worker.skipped} 张无法读取的图片或标注。"
            QMessageBox.warning(self, title, message + "\n" + "\n".join(worker.errors))
        else:
            QMessageBox.information(self, title, message)
        worker.deleteLater()

    def _open_source(self, folder, selected=None, *, comparison_sources=None):
        source = selected or folder
        selected_xml = selected is not None and selected.suffix.lower() == ".xml"
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            entries = scan_folder(folder, recursive=selected is None or selected_xml, selected=selected)
        finally:
            QApplication.restoreOverrideCursor()
        if selected_xml:
            matches = [entry for entry in entries if entry.xml == selected]
            if len(matches) != 1:
                raise ValueError("未找到 XML 唯一对应的图片。请选择图片或图片文件夹，再指定 XML 来源。")
            selected = matches[0].image
        paths = [entry.image for entry in entries]
        if not paths or (selected is not None and selected not in paths):
            raise ValueError("所选位置没有可识别的图片。")
        comparison = None
        if comparison_sources is not None:
            gt, prediction = comparison_sources
            if selected_xml and gt is None and source != prediction:
                gt = source
            anchor = selected or self._folder_anchor(paths, (gt, prediction))
            comparison = ComparisonSources(anchor, gt, prediction, images=paths,
                                           image_root=folder if source == folder else None)
            comparison.resolve(anchor)
            selected = anchor
        protect_current = self._protect_pending_prediction(comparison)
        if not self._prepare_image_change(protect_current=protect_current, autosave=True):
            return
        if selected_xml or comparison:
            self.mode_combo.setCurrentIndex(0)
        self._comparison = comparison
        self._dataset_source = source
        self._set_catalog(self._global_catalog)
        self._image_entries = {entry.image: entry for entry in entries}
        self._image_list = [str(path) for path in paths]
        self._start_dataset_metrics()
        self._image_index = paths.index(selected) if selected is not None else 0
        self._populate_image_picker(folder)
        self.project_manager = ProjectManager()
        self.project_manager.set_images(self._image_list)
        self.gallery.reset()
        self.annotation_image_label.setText(f"{folder.name}  ·  {len(paths)} 张图片")
        self.annotation_image_label.setToolTip(str(folder))

        self._set_output_dirs(str(folder / "Normal"), str(folder / "Abnormal"))
        self._load_current_image()

    @staticmethod
    def _folder_anchor(images, roles):
        if len(images) == 1:
            return images[0]
        owners = set()
        for source in roles:
            if source and source.is_file():
                matches = [p for p in images if p.stem.casefold() == source.stem.casefold()]
                if len(matches) == 1:
                    owners.add(matches[0])
                elif matches:
                    raise ValueError(f"{source.name} 对应多张图片，请选择具体图片或指定 XML 文件夹。")
        if len(owners) == 1:
            return owners.pop()
        if any(p and p.is_file() for p in roles):
            raise ValueError("无法确定单个 XML 对应的起始图片。请选择具体图片或指定 XML 文件夹。")
        return images[0]

    def _start_dataset_metrics(self):
        self.dataset_recall.start_dataset(self._image_list, self._comparison, self.recall_rule.currentData())
        self._show_dataset_metrics(self.mode_combo.currentIndex() == 0)

    def _show_dataset_metrics(self, annotation_mode):
        visible = annotation_mode and self._comparison is not None
        self.dataset_recall.setVisible(visible)
        self.recall_rule.setVisible(visible)

    def _on_recall_rule_changed(self):
        if self._comparison is None:
            return
        self._start_dataset_metrics()
        if self._xml_document is not None and self._prediction_boxes is not None:
            self.dataset_recall.update_image(self._current_image_path, self.canvas.get_annotations(), self._prediction_boxes)

    def _update_grid_params(self, stride):
        self.canvas.set_stride(stride)
        self._update_patch_count_label()
        if self.mode_combo.currentIndex() == 1:
            self._init_patch_review()

    def _update_patch_count_label(self):
        # Count Unit Modules
        grid = self.canvas.review_grid
        self.patch_count_label.setText(f"单元总数: {grid.columns} x {grid.rows} = {len(grid)} 个")

    def _on_mask_opacity_changed(self, value: int):
        self.canvas.set_mask_opacity(value / 100.0)

    def _load_current_image(self):
        if self._image_index < 0 or self._image_index >= len(self._image_list):
            return
        path = self._image_list[self._image_index]
        filename = os.path.basename(path)
        current_idx = self._image_index + 1
        total_count = len(self._image_list)
        self.current_image_label.setText(f"当前图片: {filename} ({current_idx}/{total_count})")
        with QSignalBlocker(self.review_image_picker):
            self.review_image_picker.setCurrentIndex(self._image_index)
        self.review_image_picker.setToolTip(path)
        self.image_position_label.setText(f"{filename}  ·  {current_idx}/{total_count}")
        self.source_path_label.setText(path)
        image = QImage(path)
        if image.isNull() or not self.canvas.load_image(path, image=image):
            self._current_image_path = None
            self._current_image = None
            self.canvas.reset_view()
            self._clear_annotation_state()
            self.annotation_save_label.setText("图片读取失败，可继续浏览其他图片。")
            self.annotation_save_label.setToolTip(path)
            self._current_patch_index = 0
            self._update_patch_view()
            self._refresh_gallery()
            QMessageBox.warning(self, "无法显示图片", f"无法读取：{path}")
            return
        self._current_image_path = path
        self._current_image = image

        # Restore mask from ProjectManager
        mask = self.project_manager.get_mask(path)
        if mask is not None:
            self.canvas.set_mask(mask)
        elif self.mode_combo.currentIndex() == 1:
            self.canvas.ensure_slice_mask()

        self._load_annotations(path, image.width(), image.height())

        # Update patch count immediately when image loaded
        self._update_patch_count_label()

        if self.mode_combo.currentIndex() == 1:
            self._init_patch_review()
        else:
            self._current_patch_index = 0

        # Filter gallery to show only current image's patches
        self._refresh_gallery()

    def _on_prev_image(self):
        self._go_to_image(self._image_index - 1)

    def _on_next_image(self):
        self._go_to_image(self._image_index + 1)

    def _go_to_image(self, index):
        if not 0 <= index < len(self._image_list) or self._image_index == index:
            return
        if not self._prepare_image_change(autosave=True):
            with QSignalBlocker(self.review_image_picker):
                self.review_image_picker.setCurrentIndex(self._image_index)
            return
        self._image_index = index
        self._load_current_image()

    def _get_current_config(self):
        return {
            "patch_width": self.patch_width_spin.value(),
            "patch_height": self.patch_height_spin.value(),
            "stride": self.stride_spin.value(),
            "threshold": self.threshold_spin.value(),
            "normal_dir": self.project_manager.normal_dir or "",
            "abnormal_dir": self.project_manager.abnormal_dir or "",
            "label_catalog": self.label_catalog.to_list(),
            "label_catalog_version": LabelCatalog.VERSION,
            "mode": self.mode_combo.currentIndex(),
        }

    def _apply_config(self, config):
        self.patch_width_spin.setValue(config.get("patch_width", 256))
        self.patch_height_spin.setValue(config.get("patch_height", 256))
        self.stride_spin.setValue(config.get("stride", 128))
        self.threshold_spin.setValue(config.get("threshold", 1))
        self._set_output_dirs(config.get("normal_dir"), config.get("abnormal_dir"))
        self.mode_combo.setCurrentIndex(config.get("mode", 0))

    def _on_new_project(self):
        if not self._prepare_image_change() or not self._confirm_project_close():
            return
        self._comparison = None
        self._dataset_source = None
        self._image_entries = {}
        self.project_manager = ProjectManager()
        self._set_output_dirs(None, None)
        self._set_catalog(self._global_catalog)
        self._image_list = []
        self._start_dataset_metrics()
        self._image_index = -1
        self._current_image_path = None
        self._current_image = None
        self.canvas.reset_view()
        self._clear_annotation_state()
        self._populate_image_picker()
        self.annotation_image_label.setText("尚未打开图片")
        self.image_position_label.setText("打开一张图片后，可继续浏览同文件夹中的图片")
        self.source_path_label.setText("未选择")
        self.gallery.reset()
        self._init_patch_review()
        self._update_patch_count_label()
        self.current_image_label.setText("当前图片: -")
        self.setWindowTitle("Industrial Image Slicer & Annotator - New Project")

    def _on_open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开工程", "", "Project Files (*.json)")
        if not path:
            return

        try:
            project, state, comparison = self._read_project_state(path)
            changed_mask = self.canvas.has_mask_changes()
            if not self._prepare_image_change(protect_current=self._protect_pending_prediction(comparison),
                                              autosave=True):
                return
            if changed_mask and Path(self.project_manager.project_path).resolve() == Path(path).resolve():
                # The autosave updated the file we just validated.
                project, state, comparison = self._read_project_state(path)
            current_index, config, normal_files, abnormal_files = state
            self.project_manager = project
            self._comparison = comparison
            self._image_entries = {}
            self._current_image_path = None
            self._current_image = None
            self.canvas.reset_view()
            self._clear_annotation_state()
            self.annotation_image_label.setText(os.path.basename(path))
            self.annotation_image_label.setToolTip(path)
            self.image_position_label.setText("工程中没有可显示的图片")
            self._set_catalog(self._catalog_from_config(config))
            self._apply_config(config)

            self._image_list = self.project_manager.images
            saved_source = config.get("dataset_source")
            saved_path = Path(saved_source) if isinstance(saved_source, str) else None
            self._dataset_source = saved_path if saved_path and saved_path.is_absolute() and saved_path.exists() else None
            if self._dataset_source is None and comparison:
                self._dataset_source = comparison.image_root
            if project.images and self._dataset_source is None:
                try:
                    self._dataset_source = Path(os.path.commonpath([Path(p).resolve().parent for p in project.images]))
                except ValueError:  # An existing project can span multiple Windows drives.
                    self._dataset_source = Path(project.images[current_index]).resolve()
            self._start_dataset_metrics()
            self._image_index = current_index
            self._populate_image_picker()

            self.gallery.reset(normal_files, abnormal_files)

            if 0 <= self._image_index < len(self._image_list):
                self._load_current_image()
            else:
                self.canvas.reset_view()

            self.setWindowTitle(f"Industrial Image Slicer & Annotator - {os.path.basename(path)}")
            QMessageBox.information(self, "提示", "工程加载成功")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载工程失败: {str(e)}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def _read_project_state(path):
        project = ProjectManager()
        state = project.load_project(path)
        config = state[1]
        comparison = (ComparisonSources.from_config(config["xml_comparison"], images=project.images)
                      if "xml_comparison" in config else None)
        return project, state, comparison

    def _confirm_project_close(self):
        if not self.canvas.has_mask_changes() and not (self._pending_update and self.project_manager.masks):
            return True
        choice = QMessageBox.question(self, "保存切图工程", "要在离开前保存当前切图工程吗？",
                                      QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                      QMessageBox.Save)
        return choice == QMessageBox.Discard or (choice == QMessageBox.Save and self._on_save_project())

    def _on_save_project(self):
        if self.project_manager.project_path:
            return self._save_project_impl(self.project_manager.project_path)
        return self._on_save_project_as()

    def _on_save_project_as(self):
        path = self._choose_project_path()
        return bool(path) and self._save_project_impl(path)

    def _choose_project_path(self):
        return QFileDialog.getSaveFileName(self, "保存切图工程", "", "Project Files (*.json)")[0]

    def _save_project_impl(self, path):
        if not self._prepare_image_change():
            return False
        return self._write_project(path)

    def _write_project(self, path, *, notify=True):
        try:
            config = self._get_current_config()
            if self._dataset_source:
                config["dataset_source"] = str(self._dataset_source)
            if self._comparison:
                config["xml_comparison"] = self._comparison.to_config()

            normal_files, abnormal_files = self.gallery.project_files()
            self.project_manager.save_project(
                path, self._image_index, config, normal_files, abnormal_files,
                current_mask=self.canvas.get_mask() if self.canvas.has_mask_changes() else None,
            )
            self.canvas.mark_mask_saved()
            self.setWindowTitle(f"Industrial Image Slicer & Annotator - {os.path.basename(path)}")
            if notify:
                QMessageBox.information(self, "提示", "工程保存成功")
            return True
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存工程失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _on_smart_grid_changed(self, state):
        self.canvas.set_smart_grid(self.smart_grid_checkbox.isChecked())

    def _check_for_updates(self):
        if self._update_dialog is not None:
            return
        if self._export_thread is not None or (self._slicer_thread and self._slicer_thread.isRunning()):
            QMessageBox.information(self, "检查更新", "请在当前导出或切图任务结束后更新。")
            return
        try:
            with TemporaryDirectory(prefix="AnomaSlice-update-") as directory:
                dialog = UpdateDialog(Path(directory), self)
                self._update_dialog = dialog
                try:
                    if dialog.exec() and dialog.download:
                        self._pending_update = (dialog.download, dialog.release.sha256)
                        self.close()
                finally:
                    self._pending_update = None
                    self._update_dialog = None
                    if dialog.worker is not None:
                        dialog.worker.requestInterruption()
                        dialog.worker.wait()
                    dialog.deleteLater()
        except OSError as error:
            QMessageBox.warning(self, "更新未完成", str(error))

    def closeEvent(self, event):
        if self._update_dialog is not None and self._pending_update is None:
            self._update_dialog.reject()
            event.ignore()
            return
        if self._export_thread is not None:
            self.statusBar().showMessage("请等待导出完成，或取消导出后关闭窗口。", 5000)
            event.ignore()
            return
        if self._slicer_thread is not None and self._slicer_thread.isRunning():
            self.statusBar().showMessage("正在保存切图，请完成后再关闭窗口。", 5000)
            event.ignore()
            return
        if self._pending_update and self.canvas.has_pending_annotation():
            QMessageBox.information(self, "更新已暂停", "请先按 Enter 确认正在输入的框，或按 Esc 取消，再更新。")
            event.ignore()
            return
        if not self._prepare_image_change() or not self._confirm_project_close():
            event.ignore()
            return
        try:
            self._save_global_config()
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "全局设置尚未保存", str(error))
            if self._pending_update:
                event.ignore()
                return
        if self._pending_update:
            download, digest = self._pending_update
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                launch_update(download, sys.platform, digest)
            except Exception as error:
                QMessageBox.warning(self, "无法安装更新", str(error))
                event.ignore()
                return
            finally:
                QApplication.restoreOverrideCursor()
        self.dataset_recall.stop_scans()
        super().closeEvent(event)

    def _get_global_config_path(self):
        return os.path.join(os.path.expanduser("~"), ".image_cutting_config.json")

    def _catalog_from_config(self, config):
        return LabelCatalog.from_config(config, self._global_catalog)

    def _save_global_config(self, edited_catalog=None):
        path = self._get_global_config_path()
        with json_write_lock(path):
            try:
                with open(path, encoding="utf-8") as stream:
                    stored = json.load(stream)
            except FileNotFoundError:
                stored = {}
            if not isinstance(stored, dict):
                raise ValueError("全局设置必须是 JSON 对象。")
            # A project or an older window can close after a different window
            # saves its aliases. Only the editor can replace the disk catalog.
            catalog = self._catalog_from_config(stored)
            if edited_catalog is not None:
                if catalog.to_list() != self._global_catalog.to_list():
                    raise ValueError("全局类别表已被其他窗口修改，请重新启动应用后再编辑。")
                catalog = edited_catalog
            config = {
                **stored,
                **self._get_current_config(),
                "label_catalog": catalog.to_list(),
                "smart_grid": self.smart_grid_checkbox.isChecked(),
                "mask_opacity": self.mask_opacity_slider.value()
            }
            write_json_atomic(path, config)

    def _load_global_config(self):
        path = self._get_global_config_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self._global_catalog = self._catalog_from_config(config)
            self._apply_config({k: v for k, v in config.items() if k != "mode"})

            self.smart_grid_checkbox.setChecked(config.get("smart_grid", False))
            self.mask_opacity_slider.setValue(config.get("mask_opacity", 50))
        except Exception as e:
            print(f"Error loading global config: {e}")


    def _on_slice_and_next(self):
        if self.mode_combo.currentIndex() == 1 and self._on_slice():
            self._auto_next = True

    def _on_slice(self):
        if self._current_image is None or self._slicer_thread is not None:
            return

        # Enforce that all patches must be annotated
        if self._current_patch_index < len(self.canvas.review_grid):
            QMessageBox.warning(self, "警告", "请先完成所有 Patch 的标注！")
            return

        normal_dir = self.project_manager.normal_dir
        abnormal_dir = self.project_manager.abnormal_dir
        if not normal_dir or not abnormal_dir:
            QMessageBox.warning(self, "请选择输出目录", "请先选择 Normal 和 Abnormal 输出文件夹。")
            return
        patch_w = self.patch_width_spin.value()
        patch_h = self.patch_height_spin.value()
        stride = self.stride_spin.value()
        threshold = self.threshold_spin.value() / 100.0
        image = self._current_image
        mask = self.canvas.get_mask()
        if mask is None:
            return
        base_name = self.project_manager.patch_stem(self._current_image_path)
        self.progress_bar.setValue(0)
        self.slice_button.setEnabled(False)
        self._slicer_thread = SlicerThread(
            image,
            mask,
            patch_w,
            patch_h,
            stride,
            threshold,
            normal_dir,
            abnormal_dir,
            False,
            None,
            base_name,
        )
        self._slicer_thread.progress_signal.connect(self.progress_bar.setValue)
        self._slicer_thread.error_signal.connect(self._on_slice_error)
        self._slicer_thread.finished_signal.connect(self._on_slice_finished)
        self._slicer_thread.finished.connect(self._on_thread_finished)
        self._slicer_thread.start()
        return True

    def _init_patch_review(self):
        self._current_patch_index = 0
        self._update_patch_view()

    def _update_patch_view(self):
        grid = self.canvas.review_grid
        reviewing = self._current_patch_index < len(grid)
        self.btn_no_defect.setEnabled(reviewing)
        self.btn_defect.setEnabled(reviewing)
        self.slice_button.setEnabled(bool(grid) and not reviewing and self._slicer_thread is None)
        if not reviewing:
            self.canvas.set_highlight_rect(QRectF())
            self.patch_display_label.setText("标注完成" if grid else "无图片/无单元")
            self.patch_info_label.setText(f"进度: {len(grid)}/{len(grid)}" if grid else "进度: -/-")
            return
        rect = grid.cell(self._current_patch_index)
        self.canvas.set_highlight_rect(QRectF(rect))
        self.patch_display_label.setPixmap(QPixmap.fromImage(self._current_image.copy(rect)))
        self.patch_info_label.setText(f"进度: {self._current_patch_index + 1}/{len(grid)}")

    def _record_patch_annotation(self, is_defect):
        grid = self.canvas.review_grid
        if self._current_patch_index < len(grid):
            rect = grid.cell(self._current_patch_index)
            self.canvas.fill_rect(QRectF(rect), 255 if is_defect else 0)
            self._current_patch_index += 1
            self._update_patch_view()

    def _on_canvas_clicked(self, x, y):
        index = self.canvas.review_grid.index_at(x, y)
        if index is not None:
            self._current_patch_index = index
            self._update_patch_view()

    def _on_slice_finished(self, normal_paths, abnormal_paths):
        QMessageBox.information(self, "完成", f"切分完成！\nNormal: {len(normal_paths)}\nAbnormal: {len(abnormal_paths)}")

        # We need to refresh the gallery, but filtering by current image
        self._refresh_gallery()

    def _on_slice_error(self, message):
        self._auto_next = False
        self.statusBar().showMessage("切图保存失败，请检查输出目录。")
        QMessageBox.critical(self, "切图失败", message)

    def _on_thread_finished(self):
        self.slice_button.setEnabled(True)
        self._slicer_thread = None
        if getattr(self, "_auto_next", False):
            self._auto_next = False
            self._on_next_image()
