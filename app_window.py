"""Qt window and workflow coordination; image, project, and export services are separate."""

import os
import shutil
import json
import sys
from pathlib import Path

from PySide6.QtCore import QRectF, QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QAction, QIcon, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QListWidget,
    QListWidgetItem, QLabel, QLineEdit, QMainWindow, QPushButton, QCheckBox,
    QSlider, QSizePolicy, QSpinBox, QSplitter, QTabWidget, QVBoxLayout,
    QWidget, QProgressBar, QMenu, QGroupBox, QGridLayout, QMessageBox,
    QComboBox, QStackedWidget,
)

from annotation_store import AnnotationStore, LabelCatalog
from bbox_canvas import BoxCanvasMixin
from comparison_dialog import ComparisonDialog
from comparison_paths import ComparisonSources
from dataset_recall_label import DatasetRecallLabel
from image_canvas import ScalableImageLabel, SliceCanvas
from label_dialog import LabelDialog
from project_store import ProjectManager, json_write_lock, write_json_atomic
from slicer import SlicerThread
from xml_annotations import AnnotationXml, ImageEntry, scan_folder


class CanvasWidget(BoxCanvasMixin, SliceCanvas):
    """Compose rectangle interaction with the native image canvas."""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AnomaSlice — 图片标注与切图")
        self.resize(1400, 900)
        self.project_manager = ProjectManager()
        self._global_catalog = LabelCatalog.defaults()
        self.label_catalog = self._global_catalog
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
        self._thumbnail_queue = { }
        self._thumbnail_timer = None
        self._slicer_thread = None
        self._patch_coords = []
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

        self.slice_and_next_action = QAction("Slice and Next", self)
        self.slice_and_next_action.setShortcut("Return")
        self.slice_and_next_action.triggered.connect(self._on_slice_and_next)

        self.addAction(self.slice_and_next_action)

    def _build_ui(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        file_menu.addAction(self.open_dataset_action)
        file_menu.addSeparator()
        file_menu.addAction(self.new_project_action)
        file_menu.addAction(self.open_project_action)
        file_menu.addAction(self.save_project_action)
        file_menu.addAction(self.save_project_as_action)
        file_menu.addSeparator()

        edit_menu = menubar.addMenu("编辑")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)

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
        self.gallery = self._build_gallery()

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
        self.recall_rule.setToolTip("选择同图 GT 框与预测框的位置匹配条件")
        self.recall_rule.addItem("有交集", 0.0)
        self.recall_rule.addItem("小框覆盖率 ≥ 50%", 0.5)
        self.recall_rule.currentIndexChanged.connect(self._on_recall_rule_changed)
        top.addWidget(self.recall_rule)
        top.addStretch()
        self.image_position_label = QLabel("打开一张图片后，可继续浏览同文件夹中的图片")
        top.addWidget(self.image_position_label)
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
        self.save_xml_button = QPushButton(f"保存 GT XML  ({save_key})")
        self.save_xml_button.clicked.connect(self._on_save_current)
        layout.addWidget(self.save_xml_button)
        layout.addStretch()
        undo_key = self.undo_action.shortcut().toString(native_text)
        redo_key = self.redo_action.shortcut().toString(native_text)
        navigation_help = ("触控板：双指滑动平移，双指捏合或张开缩放，三指拖动画框。"
                           if sys.platform == "darwin" else "滚轮缩放，右键拖动平移。")
        self.annotation_help = QLabel(
            f"左键拖动画框，Enter 确认标签。\n单击选框，双击改标签，{delete_key} 删除。\n"
            f"{navigation_help}\n{undo_key} 撤回，{redo_key} 重做。"
        )
        self.annotation_help.setWordWrap(True)
        layout.addWidget(self.annotation_help)
        navigation = QHBoxLayout()
        previous = QPushButton("上一张")
        previous.clicked.connect(self._on_prev_image)
        following = QPushButton("下一张")
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
        self.save_project_action.setText("保存 GT XML" if annotation else "保存工程")
        if not annotation:
            self.canvas.ensure_slice_mask()
        self.slice_and_next_action.setEnabled(not annotation)
        self.main_splitter.setSizes([270 if annotation else 350, 1000, 0 if annotation else 320])
        self.canvas.fit_image()
        if not annotation:
            self._init_patch_review()

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

    def _set_catalog(self, catalog):
        self.label_catalog = catalog
        self.canvas.resolve_label = self.label_catalog.resolve
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
            return f"GT · {doc.path.name}"
        if doc.boxes:
            return "已载入旧标注，保存后生成 XML。"
        return f"保存后新建 GT · {doc.path.name}"

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
        self.annotation_save_label.setText(f"GT 已保存，当前 {len(doc.boxes)} 个框。")
        return True

    def _settle_xml_changes(self, *, protect_current=False):
        if not self._boxes_dirty:
            return True
        if protect_current:
            answer = QMessageBox.question(
                self, "预测 XML 有未保存修改",
                "当前编辑的文件已被选作只读预测。放弃这些修改并开始对照？",
                QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Cancel,
            )
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

    def _prepare_image_change(self, *, protect_current=False):
        if self._slicer_thread is not None and self._slicer_thread.isRunning():
            self.statusBar().showMessage("正在保存切图，请完成后再切换图片。", 5000)
            return False
        if not self._settle_xml_changes(protect_current=protect_current):
            return False
        self.canvas.cancel_pending_annotation()
        if self._current_image_path:
            self.project_manager.set_mask(self._current_image_path, self.canvas.get_mask())
        return True

    def _load_annotations(self, path, width, height):
        self._clear_annotation_state()
        try:
            image_path = Path(path).resolve()
            if image_path not in self._image_entries:
                # Older projects store image paths only. Resolve their adjacent annotations on demand.
                entries = scan_folder(image_path.parent, recursive=False)
                self._image_entries.update((entry.image, entry) for entry in entries)
            entry = self._image_entries.get(image_path)
            if entry is None:
                raise ValueError("图片已移动，请重新打开文件夹。")
            if self._comparison:
                pair = self._comparison.resolve(image_path)
                self._load_prediction(pair.prediction)
                doc = AnnotationXml.open_path(pair.gt)
            else:
                doc = AnnotationXml.open(entry)
            if not doc.exists and not self._comparison:
                legacy = self.annotation_store.load(path, width, height, include_catalog=False)
                if os.path.isfile(self.annotation_store.sidecar_path(path)):
                    doc.boxes = legacy["boxes"]
                else:
                    doc.boxes = self.annotation_store.validate_boxes(
                        self.project_manager.annotations.get(path, []), width, height
                    )
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

        self.btn_no_defect.clicked.connect(self._on_patch_no_defect)
        self.btn_defect.clicked.connect(self._on_patch_defect)

        btn_layout.addWidget(self.btn_no_defect)
        btn_layout.addWidget(self.btn_defect)
        layout.addLayout(btn_layout)

        # Navigation
        nav_layout = QHBoxLayout()

        self.prev_image_button = QPushButton("上一张大图")
        self.prev_image_button.clicked.connect(self._on_prev_image)
        nav_layout.addWidget(self.prev_image_button)

        self.next_button = QPushButton("下一张大图")
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

        # Path selection helpers
        def create_path_selector(label_text, label_widget, is_folder=True):
            wrapper = QWidget()
            v_layout = QVBoxLayout(wrapper)
            v_layout.setContentsMargins(0, 0, 0, 0)
            v_layout.setSpacing(2)

            # Label
            lbl = QLabel(label_text)
            lbl.setStyleSheet("color: #aaa; font-weight: bold; font-size: 11px;")
            v_layout.addWidget(lbl)

            # Input + Button in a sub-layout or just add them
            input_container = QWidget()
            input_layout = QHBoxLayout(input_container)
            input_layout.setContentsMargins(0, 0, 0, 0)
            input_layout.setSpacing(2)

            # Configure label widget (QLabel for wrapping)
            label_widget.setWordWrap(True)
            label_widget.setStyleSheet("color: #ddd; border: 1px solid #444; border-radius: 2px; padding: 2px; font-size: 12px;")
            label_widget.setText("未选择")
            # Ensure it doesn't take too much vertical space but can expand
            label_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

            input_layout.addWidget(label_widget, 1)

            btn = QPushButton("...")
            btn.setFixedWidth(30)
            # Each caller connects its own selector, once.
            input_layout.addWidget(btn)

            v_layout.addWidget(input_container)
            return wrapper, btn

        # Source Path
        self.source_path_label = QLabel()
        source_widget, self.source_btn = create_path_selector("原始图源路径", self.source_path_label, False)
        # self.source_btn.clicked.disconnect()
        # Remove the button from this widget since we moved it to Action group
        self.source_btn.setVisible(False)
        layout.addWidget(source_widget, 0)

        # Normal Path
        self.normal_path_label = QLabel()
        normal_widget, self.normal_btn = create_path_selector("Normal 输出路径", self.normal_path_label, True)
        self.normal_btn.clicked.connect(lambda: self._select_normal_dir(self.normal_path_label))
        layout.addWidget(normal_widget, 0)

        # Abnormal Path
        self.abnormal_path_label = QLabel()
        abnormal_widget, self.abnormal_btn = create_path_selector("Abnormal 输出路径", self.abnormal_path_label, True)
        self.abnormal_btn.clicked.connect(lambda: self._select_abnormal_dir(self.abnormal_path_label))
        layout.addWidget(abnormal_widget, 0)

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

        # Debounce timer for parameter changes
        self._param_update_timer = QTimer()
        self._param_update_timer.setSingleShot(True)
        self._param_update_timer.setInterval(300)
        self._param_update_timer.timeout.connect(self._update_grid_params)

        self.stride_spin.valueChanged.connect(self._schedule_param_update)
        self.patch_width_spin.valueChanged.connect(self._schedule_param_update)
        self.patch_height_spin.valueChanged.connect(self._schedule_param_update)

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
        # Remove fixed maximum size if we want it to stretch with panel width,
        # but better to keep it square-ish or proportional.
        # Let's just set a reasonable minimum height.
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

    def _build_gallery(self):
        container = QWidget()
        container.setMinimumWidth(320)
        layout = QVBoxLayout(container)

        # Top: List Widget
        self.tabs = QTabWidget()
        self.normal_list = QListWidget()
        self.abnormal_list = QListWidget()
        self.normal_list.setViewMode(QListWidget.IconMode)
        self.normal_list.setResizeMode(QListWidget.Adjust)
        self.normal_list.setIconSize(QPixmap(64, 64).size())
        self.abnormal_list.setViewMode(QListWidget.IconMode)
        self.abnormal_list.setResizeMode(QListWidget.Adjust)
        self.abnormal_list.setIconSize(QPixmap(64, 64).size())

        # Context Menu
        self.normal_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.normal_list.customContextMenuRequested.connect(lambda pos: self._on_gallery_context_menu(pos, self.normal_list, "normal"))

        self.abnormal_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.abnormal_list.customContextMenuRequested.connect(lambda pos: self._on_gallery_context_menu(pos, self.abnormal_list, "abnormal"))

        # Connect clicks
        self.normal_list.itemClicked.connect(self._on_gallery_item_clicked)
        self.abnormal_list.itemClicked.connect(self._on_gallery_item_clicked)

        self.tabs.addTab(self.normal_list, "Normal")
        self.tabs.addTab(self.abnormal_list, "Abnormal")
        layout.addWidget(self.tabs, 1) # Stretch 1

        # Bottom: Preview Widget
        self.preview_label = QLabel("预览区域")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #222; border: 1px solid #444;")
        self.preview_label.setScaledContents(False) # We will handle scaling manually to keep aspect ratio
        layout.addWidget(self.preview_label, 1) # Stretch 1 (Half height)

        return container

    def _on_gallery_item_clicked(self, item: QListWidgetItem):
        path = item.toolTip()
        if not path or not os.path.exists(path):
            return

        # 1. Show in preview label
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            # Scale to fit label while keeping aspect ratio
            w = self.preview_label.width()
            h = self.preview_label.height()
            scaled = pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_label.setPixmap(scaled)

        # 2. Highlight in Canvas
        # Filename format: {base_name}_{x}_{y}.png
        try:
            filename = os.path.basename(path)
            name_part = os.path.splitext(filename)[0]
            # Split by last two underscores
            parts = name_part.rsplit('_', 2)
            if len(parts) >= 3:
                x = int(parts[1])
                y = int(parts[2])
                patch_w = self.patch_width_spin.value()
                patch_h = self.patch_height_spin.value()
                rect = QRectF(x, y, patch_w, patch_h)
                self.canvas.set_highlight_rect(rect)
                # Optional: Center view on highlight
                # self.canvas.centerOn(rect.center())
        except Exception as e:
            print(f"Error parsing coordinates from filename: {e}")

    def _select_folder(self, line_edit: QLineEdit):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹", line_edit.text())
        if folder:
            line_edit.setText(os.path.normpath(folder))

    def _select_normal_dir(self, label_widget: QLabel):
        folder = QFileDialog.getExistingDirectory(self, "选择Normal输出文件夹", label_widget.text())
        if folder:
            norm_path = os.path.normpath(folder)
            label_widget.setText(norm_path)
            self.project_manager.normal_dir = norm_path

    def _select_abnormal_dir(self, label_widget: QLabel):
        folder = QFileDialog.getExistingDirectory(self, "选择Abnormal输出文件夹", label_widget.text())
        if folder:
            norm_path = os.path.normpath(folder)
            label_widget.setText(norm_path)
            self.project_manager.abnormal_dir = norm_path

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
        self.box_count_label.setText(f"GT 标注框（实线）：{len(boxes)}")
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
            self.prediction_status.setText("预测（只读）· 本图无匹配 XML")
            return
        except (OSError, ValueError) as error:
            self.dataset_recall.image_error(self._current_image_path, error)
            self.prediction_status.setText(f"预测 XML 读取失败：{error}")
            return
        self._prediction_boxes = boxes
        self.canvas.set_predictions(boxes)
        self.prediction_status.setText(f"预测框（青色虚线，只读）：{len(boxes)}")

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
        return bool(self._boxes_dirty and comparison and comparison.protects_prediction(self._xml_document.path))

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
                    raise ValueError(f"自动匹配会将预测文件 {conflict} 用作 GT。请指定 GT 来源。")
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
        with QSignalBlocker(self.review_image_picker):
            self.review_image_picker.clear()
            for path in self._image_list:
                name = str(Path(path).relative_to(root)) if root else path
                self.review_image_picker.addItem(name)
            self.review_image_picker.setCurrentIndex(self._image_index)

    def _open_source(self, folder, selected=None, *, comparison_sources=None):
        source = selected or folder
        selected_xml = selected is not None and selected.suffix.lower() == ".xml"
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            entries = scan_folder(folder, recursive=selected is None or selected_xml)
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
        if not self._prepare_image_change(protect_current=protect_current):
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
        self.project_manager.set_images(self._image_list)
        self.project_manager.project_path = None
        self.annotation_image_label.setText(f"{folder.name}  ·  {len(paths)} 张图片")
        self.annotation_image_label.setToolTip(str(folder))

        # Auto-set Normal/Abnormal paths if not set or user wants convenience
        normal_dir = os.path.join(folder, "Normal")
        abnormal_dir = os.path.join(folder, "Abnormal")

        self.project_manager.normal_dir = normal_dir
        self.project_manager.abnormal_dir = abnormal_dir

        self.normal_path_label.setText(normal_dir)
        self.abnormal_path_label.setText(abnormal_dir)
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

    def _schedule_param_update(self):
        self._param_update_timer.start()

    def _update_grid_params(self):
        # Grid should visualize the Unit Modules (Stride x Stride), not the Patches
        stride = self.stride_spin.value()
        self.canvas.set_patch_size(stride, stride)
        self.canvas.set_stride(stride)
        self._update_patch_count_label()
        if self.mode_combo.currentIndex() == 1:
            self._init_patch_review()

    def _update_patch_count_label(self):
        # Count Unit Modules
        rows, cols = self.canvas.get_patch_counts()
        total = rows * cols
        self.patch_count_label.setText(f"单元总数: {cols} x {rows} = {total} 个")

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
            self._patch_coords = []
            self._current_patch_index = 0
            self._update_patch_view()
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

        # Allocate grid traversal only when the slicing workflow is in use.
        if self.mode_combo.currentIndex() == 1:
            self._init_patch_review()
        else:
            self._patch_coords = []
            self._current_patch_index = 0

        # Filter gallery to show only current image's patches
        self._filter_gallery_by_current_image()

    def _on_prev_image(self):
        self._go_to_image(self._image_index - 1)

    def _on_next_image(self):
        self._go_to_image(self._image_index + 1)

    def _go_to_image(self, index):
        if not 0 <= index < len(self._image_list) or self._image_index == index:
            return
        if not self._prepare_image_change():
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
            "normal_dir": self.normal_path_label.text(),
            "abnormal_dir": self.abnormal_path_label.text(),
            "label_catalog": self.label_catalog.to_list(),
            "label_catalog_version": 1,
            "mode": self.mode_combo.currentIndex(),
        }

    def _apply_config(self, config):
        if not config:
            return
        self.patch_width_spin.setValue(config.get("patch_width", 256))
        self.patch_height_spin.setValue(config.get("patch_height", 256))
        self.stride_spin.setValue(config.get("stride", 128))
        self.threshold_spin.setValue(config.get("threshold", 1))
        if "normal_dir" in config:
            self.normal_path_label.setText(config["normal_dir"])
            self.project_manager.normal_dir = config["normal_dir"]
        if "abnormal_dir" in config:
            self.abnormal_path_label.setText(config["abnormal_dir"])
            self.project_manager.abnormal_dir = config["abnormal_dir"]
        self.mode_combo.setCurrentIndex(config.get("mode", 0))

    def _on_new_project(self):
        if not self._prepare_image_change():
            return
        self._comparison = None
        self._dataset_source = None
        self._image_entries = {}
        self._start_dataset_metrics()
        self.project_manager = ProjectManager()
        self._set_catalog(self._global_catalog)
        self._image_list = []
        self._image_index = -1
        self._current_image_path = None
        self._current_image = None
        self.canvas.reset_view()
        self._clear_annotation_state()
        self._populate_image_picker()
        self.annotation_image_label.setText("尚未打开图片")
        self.image_position_label.setText("打开一张图片后，可继续浏览同文件夹中的图片")
        self.source_path_label.setText("未选择")
        self.normal_list.clear()
        self.abnormal_list.clear()
        self.setWindowTitle("Industrial Image Slicer & Annotator - New Project")

    def _on_open_project(self):
        from PySide6.QtWidgets import QMessageBox
        path, _ = QFileDialog.getOpenFileName(self, "打开工程", "", "Project Files (*.json)")
        if not path:
            return

        try:
            project = ProjectManager()
            current_index, config, normal_files, abnormal_files = project.load_project(path)
            comparison = None
            protect_current = False
            if "xml_comparison" in config:
                comparison = ComparisonSources.from_config(config["xml_comparison"], images=project.images)
                protect_current = self._protect_pending_prediction(comparison)
            if not self._prepare_image_change(protect_current=protect_current):
                return
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

            # Restore Gallery
            self.normal_list.clear()
            self.abnormal_list.clear()
            self._thumbnail_queue.clear()

            if normal_files:
                self._enqueue_thumbnails(self.normal_list, normal_files)
            if abnormal_files:
                self._enqueue_thumbnails(self.abnormal_list, abnormal_files)

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

    def _on_save_project(self):
        if self.project_manager.project_path:
            self._save_project_impl(self.project_manager.project_path)
        else:
            self._on_save_project_as()

    def _on_save_project_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存工程", "", "Project Files (*.json)")
        if not path:
            return
        self._save_project_impl(path)

    def _save_project_impl(self, path):
        from PySide6.QtWidgets import QMessageBox
        if not self._prepare_image_change():
            return
        # Ensure current mask is updated in project manager
        if self._current_image_path:
            self.project_manager.set_mask(self._current_image_path, self.canvas.get_mask())

        try:
            config = self._get_current_config()
            if self._dataset_source:
                config["dataset_source"] = str(self._dataset_source)
            if self._comparison:
                config["xml_comparison"] = self._comparison.to_config()

            # Collect file paths from list widgets
            normal_files = []
            for i in range(self.normal_list.count()):
                item = self.normal_list.item(i)
                normal_files.append(item.toolTip()) # Tooltip stores full path

            abnormal_files = []
            for i in range(self.abnormal_list.count()):
                item = self.abnormal_list.item(i)
                abnormal_files.append(item.toolTip())

            self.project_manager.save_project(path, self._image_index, config, normal_files, abnormal_files)
            self.setWindowTitle(f"Industrial Image Slicer & Annotator - {os.path.basename(path)}")
            QMessageBox.information(self, "提示", "工程保存成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存工程失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def _on_gallery_context_menu(self, pos, list_widget, source_type):
        item = list_widget.itemAt(pos)
        if not item:
            return

        menu = QMenu(self)
        if source_type == "normal":
            action = menu.addAction("移动到 Abnormal")
            target_dir = self.project_manager.abnormal_dir
            target_list = self.abnormal_list
        else:
            action = menu.addAction("移动到 Normal")
            target_dir = self.project_manager.normal_dir
            target_list = self.normal_list

        action.triggered.connect(lambda: self._move_gallery_item(item, list_widget, target_list, target_dir))
        menu.exec(list_widget.mapToGlobal(pos))

    def _move_gallery_item(self, item, source_list, target_list, target_dir):
        source_path = item.toolTip()
        if not os.path.exists(source_path):
             return

        file_name = os.path.basename(source_path)
        target_path = os.path.join(target_dir, file_name)

        try:
            os.makedirs(target_dir, exist_ok=True)
            shutil.move(source_path, target_path)
        except Exception as e:
            print(f"Error moving file: {e}")
            return

        row = source_list.row(item)
        source_list.takeItem(row)

        # Check if we should add to target list based on filter
        # Only add if it belongs to current image (prefix match)
        current_img_name = os.path.splitext(os.path.basename(self._current_image_path))[0]
        if file_name.startswith(current_img_name):
            new_item = QListWidgetItem(item.icon(), file_name)
            new_item.setToolTip(target_path)
            target_list.addItem(new_item)

    def _on_smart_grid_changed(self, state):
        self.canvas.set_smart_grid(self.smart_grid_checkbox.isChecked())

    def closeEvent(self, event):
        if self._slicer_thread is not None and self._slicer_thread.isRunning():
            self.statusBar().showMessage("正在保存切图，请完成后再关闭窗口。", 5000)
            event.ignore()
            return
        if not self._prepare_image_change():
            event.ignore()
            return
        try:
            self._save_global_config()
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "全局设置尚未保存", str(error))
        self.dataset_recall.stop_scans()
        super().closeEvent(event)

    def _get_global_config_path(self):
        return os.path.join(os.path.expanduser("~"), ".image_cutting_config.json")

    def _catalog_from_config(self, config):
        if "label_catalog_version" in config:
            version = config["label_catalog_version"]
            if type(version) is not int or version != 1:
                raise ValueError(f"不支持的类别表版本：{version}")
        if "label_catalog" not in config:
            return self._global_catalog
        if "label_catalog_version" not in config:
            return LabelCatalog.from_legacy(config["label_catalog"])
        return LabelCatalog(config["label_catalog"])

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

            self.patch_width_spin.setValue(config.get("patch_width", 256))
            self.patch_height_spin.setValue(config.get("patch_height", 256))
            self.stride_spin.setValue(config.get("stride", 128))
            self.threshold_spin.setValue(config.get("threshold", 1))
            if "normal_dir" in config:
                self.normal_path_label.setText(config["normal_dir"])
            if "abnormal_dir" in config:
                self.abnormal_path_label.setText(config["abnormal_dir"])
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
        if self._patch_coords and self._current_patch_index < len(self._patch_coords):
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "请先完成所有 Patch 的标注！")
            return

        normal_dir = self.normal_path_label.text().strip()
        abnormal_dir = self.abnormal_path_label.text().strip()
        if not normal_dir or not abnormal_dir:
            return
        save_mask = False
        mask_dir = None
        if save_mask:
            mask_dir = os.path.join(abnormal_dir, "mask")
        patch_w = self.patch_width_spin.value()
        patch_h = self.patch_height_spin.value()
        stride = self.stride_spin.value()
        threshold = self.threshold_spin.value() / 100.0
        image = self._current_image
        mask = self.canvas.get_mask()
        if mask is None:
            return
        w, h = image.width(), image.height()
        base_name = os.path.splitext(os.path.basename(self._current_image_path))[0]
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
            save_mask,
            mask_dir,
            base_name,
        )
        self._slicer_thread.progress_signal.connect(self.progress_bar.setValue)
        self._slicer_thread.error_signal.connect(self._on_slice_error)
        self._slicer_thread.finished_signal.connect(self._on_slice_finished)
        self._slicer_thread.finished.connect(self._on_thread_finished)
        self._slicer_thread.start()
        return True

    def _init_patch_review(self):
        self._patch_coords = []
        self._current_patch_index = 0
        if self._current_image is None:
            self._update_patch_view()
            return

        w, h = self._current_image.width(), self._current_image.height()
        # Iterate based on Unit Modules (Stride x Stride)
        # We tile the image with stride-sized blocks
        stride = self.stride_spin.value()

        # Ensure we cover the whole image, even if it's not a perfect multiple
        # But for annotation grid, it's cleaner to stick to the grid definition
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Ensure we don't go out of bounds (though usually we just clip)
                if y + stride <= h and x + stride <= w:
                    self._patch_coords.append((x, y))
                elif y < h and x < w:
                    # Handle edge cases? For now, let's only include full units or
                    # include partials? The user said "5120x5120 image, 512 stride", which is perfect.
                    # Let's include partials for robustness
                    self._patch_coords.append((x, y))

        self.slice_button.setEnabled(False)
        self._update_patch_view()

    def _update_patch_view(self):
        if not self._patch_coords or self._current_patch_index >= len(self._patch_coords):
            if self._patch_coords and self._current_patch_index >= len(self._patch_coords):
                self.patch_display_label.setText("标注完成")
                self.patch_info_label.setText(f"进度: {len(self._patch_coords)}/{len(self._patch_coords)}")
                self.canvas.set_highlight_rect(QRectF())
                self.slice_button.setEnabled(True)
                self.btn_no_defect.setEnabled(False)
                self.btn_defect.setEnabled(False)
            else:
                self.patch_display_label.setText("无图片/无单元")
                self.patch_info_label.setText("进度: -/-")
                self.btn_no_defect.setEnabled(False)
                self.btn_defect.setEnabled(False)
            return

        self.btn_no_defect.setEnabled(True)
        self.btn_defect.setEnabled(True)
        self.slice_button.setEnabled(False)

        x, y = self._patch_coords[self._current_patch_index]
        # Unit Size = Stride
        stride = self.stride_spin.value()

        # Highlight in canvas (Unit Module)
        # Note: If edge case, width/height might be smaller
        w_img, h_img = self._current_image.width(), self._current_image.height()
        unit_w = min(stride, w_img - x)
        unit_h = min(stride, h_img - y)

        self.canvas.set_highlight_rect(QRectF(x, y, unit_w, unit_h))

        # Extract unit module
        if self._current_image is not None:
            patch = self._current_image.copy(x, y, unit_w, unit_h)
            pixmap = QPixmap.fromImage(patch)

            # Show in label (ScalableImageLabel handles scaling)
            self.patch_display_label.setPixmap(pixmap)

        self.patch_info_label.setText(f"进度: {self._current_patch_index + 1}/{len(self._patch_coords)}")

    def _on_patch_no_defect(self):
        self._record_patch_annotation(is_defect=False)
        self._next_patch()

    def _on_patch_defect(self):
        self._record_patch_annotation(is_defect=True)
        self._next_patch()

    def _record_patch_annotation(self, is_defect):
        if self._current_patch_index >= len(self._patch_coords):
            return

        x, y = self._patch_coords[self._current_patch_index]
        stride = self.stride_spin.value()

        # Fill Unit Module
        w_img, h_img = self._current_image.width(), self._current_image.height()
        unit_w = min(stride, w_img - x)
        unit_h = min(stride, h_img - y)

        self.canvas.fill_rect(QRectF(x, y, unit_w, unit_h), 255 if is_defect else 0)

    def _next_patch(self):
        self._current_patch_index += 1
        self._update_patch_view()

    def _on_prev_patch(self):
        if self._current_patch_index > 0:
            self._current_patch_index -= 1
            self._update_patch_view()

    def _on_canvas_clicked(self, x, y):
        """Navigate to the patch closest to the clicked point."""
        if not self._patch_coords:
            return

        # Simple linear search for closest patch/unit
        # Ideally, we check which unit rect contains the point
        stride = self.stride_spin.value()
        w_img, h_img = self._current_image.width(), self._current_image.height()

        found_index = -1

        for i, (px, py) in enumerate(self._patch_coords):
            # Calculate unit dimensions
            unit_w = min(stride, w_img - px)
            unit_h = min(stride, h_img - py)

            rect = QRectF(px, py, unit_w, unit_h)
            if rect.contains(x, y):
                found_index = i
                break

        if found_index != -1:
            self._current_patch_index = found_index
            self._update_patch_view()

    def _on_slice_finished(self, normal_paths, abnormal_paths):
        QMessageBox.information(self, "完成", f"切分完成！\nNormal: {len(normal_paths)}\nAbnormal: {len(abnormal_paths)}")

        # We need to refresh the gallery, but filtering by current image
        self._filter_gallery_by_current_image()

    def _on_slice_error(self, message):
        self._auto_next = False
        self.statusBar().showMessage("切图保存失败，请检查输出目录。")
        QMessageBox.critical(self, "切图失败", message)

    def _filter_gallery_by_current_image(self):
        """Reload gallery lists but only show patches belonging to the current image."""
        if not self._current_image_path:
            self.normal_list.clear()
            self.abnormal_list.clear()
            return

        # Get current image basename without extension
        # e.g. "image_01.png" -> "image_01"
        current_name = os.path.splitext(os.path.basename(self._current_image_path))[0]

        self.normal_list.clear()
        self.abnormal_list.clear()

        # Helper to load directory and filter
        def load_dir(directory, list_widget):
            if not directory or not os.path.exists(directory):
                return

            # Sort by modification time or name? Name usually keeps order
            try:
                files = sorted(os.listdir(directory))
            except Exception:
                return

            # Collect valid files first
            valid_files = []
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    if f.startswith(current_name):
                        valid_files.append(os.path.join(directory, f))

            # Use the thumbnail queue mechanism to load them
            if valid_files:
                self._enqueue_thumbnails(list_widget, valid_files)

        load_dir(self.project_manager.normal_dir, self.normal_list)
        load_dir(self.project_manager.abnormal_dir, self.abnormal_list)

    def _on_thread_finished(self):
        self.slice_button.setEnabled(True)
        self._slicer_thread = None
        if getattr(self, "_auto_next", False):
            self._auto_next = False
            self._on_next_image()

    def _enqueue_thumbnails(self, list_widget: QListWidget, paths):
        if not paths:
            return
        if list_widget not in self._thumbnail_queue:
            self._thumbnail_queue[list_widget] = []
        self._thumbnail_queue[list_widget].extend(paths)
        if self._thumbnail_timer is None:
            self._thumbnail_timer = QTimer(self)
            self._thumbnail_timer.timeout.connect(self._process_thumbnail_queue)
            self._thumbnail_timer.start(0)

    def _process_thumbnail_queue(self):
        batch_size = 50
        empty_widgets = []
        for list_widget, queue in self._thumbnail_queue.items():
            for _ in range(min(batch_size, len(queue))):
                path = queue.pop(0)
                pixmap = QPixmap(path)
                if pixmap.isNull():
                    continue
                icon = QIcon(pixmap.scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                item = QListWidgetItem(icon, os.path.basename(path))
                item.setToolTip(path)
                list_widget.addItem(item)
            if not queue:
                empty_widgets.append(list_widget)
        for widget in empty_widgets:
            self._thumbnail_queue.pop(widget, None)
        if not self._thumbnail_queue:
            self._thumbnail_timer.stop()
            self._thumbnail_timer = None
