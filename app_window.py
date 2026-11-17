"""Qt window and workflow coordination; image, project, and export services are separate."""

import os
import shutil
import json
import re

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QAction, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QListWidget,
    QListWidgetItem, QLabel, QLineEdit, QMainWindow, QPushButton, QCheckBox,
    QSlider, QSizePolicy, QSpinBox, QSplitter, QTabWidget, QVBoxLayout,
    QWidget, QProgressBar, QMenu, QGroupBox, QGridLayout, QMessageBox,
    QComboBox, QStackedWidget,
)

from annotation_store import AnnotationStore, LabelCatalog
from bbox_canvas import BoxCanvasMixin
from image_canvas import ScalableImageLabel, SliceCanvas
from label_dialog import LabelDialog
from project_store import ProjectManager
from slicer import SlicerThread


class CanvasWidget(BoxCanvasMixin, SliceCanvas):
    """Compose rectangle interaction with the native image canvas."""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AnomaSlice — 图片标注与切图")
        self.resize(1400, 900)
        self.project_manager = ProjectManager()
        self.label_catalog = LabelCatalog([])
        self.annotation_store = AnnotationStore()
        self._boxes_dirty = False
        self._annotation_load_error = False
        self._current_image_path = None
        self._current_image = None
        self._image_list = []
        self._image_index = -1
        self._thumbnail_queue = { }
        self._thumbnail_timer = None
        self._slicer_thread = None
        self._patch_coords = []
        self._current_patch_index = 0
        self._create_actions()
        self._build_ui()
        self.canvas.resolve_label = self.label_catalog.resolve
        self.canvas.annotations_changed = self._on_annotations_changed
        self.canvas.new_label_committed = self._on_label_committed
        self._load_global_config()
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

        self.open_image_action = QAction("打开图片", self)
        self.open_image_action.setShortcut("Ctrl+I")
        self.open_image_action.triggered.connect(self._open_image)

        self.slice_and_next_action = QAction("Slice and Next", self)
        self.slice_and_next_action.setShortcut("Return")
        self.slice_and_next_action.triggered.connect(self._on_slice_and_next)

        self.addAction(self.slice_and_next_action)

    def _build_ui(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        file_menu.addAction(self.open_image_action)
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
        open_button = QPushButton("打开图片…  (Ctrl+I)")
        open_button.clicked.connect(self._open_image)
        layout.addWidget(open_button)
        self.annotation_image_label = QLabel("尚未打开图片")
        self.annotation_image_label.setWordWrap(True)
        self.annotation_image_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.annotation_image_label)
        classes = QPushButton("缺陷种类与别名…")
        classes.clicked.connect(self._edit_labels)
        layout.addWidget(classes)
        self.catalog_label = QLabel("可直接输入标签，也可先设置种类别名。")
        self.catalog_label.setWordWrap(True)
        layout.addWidget(self.catalog_label)
        self.box_count_label = QLabel("标注框：0")
        layout.addWidget(self.box_count_label)
        self.annotation_save_label = QLabel("每次确认后自动保存到图片旁的 JSON 文件。")
        self.annotation_save_label.setWordWrap(True)
        layout.addWidget(self.annotation_save_label)
        fit = QPushButton("适应窗口")
        fit.clicked.connect(self.canvas.fit_image)
        layout.addWidget(fit)
        delete = QPushButton("删除所选框  (Delete)")
        delete.clicked.connect(self.canvas.delete_selected_annotation)
        layout.addWidget(delete)
        undo_row = QHBoxLayout()
        undo = QPushButton("撤回")
        undo.clicked.connect(self._undo_current)
        redo = QPushButton("重做")
        redo.clicked.connect(self._redo_current)
        undo_row.addWidget(undo)
        undo_row.addWidget(redo)
        layout.addLayout(undo_row)
        layout.addStretch()
        help_text = QLabel("滚轮：缩放图片\n左键拖动：画框\n右键拖动：移动图片\nEnter：确认标签　Esc：取消\n单击：选框　双击：改标签\nCtrl+Z：撤回\nCtrl+Shift+Z：重做")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
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
        self.canvas.cancel_pending_annotation()
        if annotation:
            self._annotation_canvas_layout.addWidget(self.canvas)
        else:
            self._slice_canvas_layout.insertWidget(self._slice_canvas_index, self.canvas, 1)
        self.left_stack.setCurrentIndex(index)
        self.center_stack.setCurrentIndex(index)
        self.gallery.setVisible(not annotation)
        self.canvas.set_annotation_mode(annotation)
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
            self._flush_annotations(force=True)
        else:
            self._on_save_project()

    def _refresh_catalog(self):
        self.canvas.resolve_label = self.label_catalog.resolve
        descriptions = []
        for record in self.label_catalog.to_list():
            aliases = ", ".join(record["aliases"])
            descriptions.append(f'{record["name"]}  ←  {aliases}' if aliases else record["name"])
        self.catalog_label.setText("\n".join(descriptions) or "可直接输入标签，也可先设置种类别名。")

    def _edit_labels(self):
        dialog = LabelDialog(self.label_catalog, self)
        if dialog.exec():
            self.label_catalog = dialog.catalog
            self._refresh_catalog()
            self._save_global_config()
            if self._current_image_path and not self._annotation_load_error:
                self._boxes_dirty = True
                self._flush_annotations()

    def _on_label_committed(self, label):
        self.label_catalog.add_label(label)
        self._refresh_catalog()
        self._save_global_config()

    def _on_annotations_changed(self, boxes):
        if not self._current_image_path or self._annotation_load_error:
            return
        self.project_manager.annotations[self._current_image_path] = boxes
        self.box_count_label.setText(f"标注框：{len(boxes)}")
        self._boxes_dirty = True
        self._flush_annotations()

    def _flush_annotations(self, force=False):
        if not self._current_image_path or self._current_image is None:
            return True
        if self._annotation_load_error:
            return True  # Keep unreadable sidecars untouched; navigation remains available.
        if not self._boxes_dirty and not force:
            return True
        w, h = self._current_image.width(), self._current_image.height()
        try:
            boxes = self.canvas.get_annotations()
            saved_path = self.annotation_store.save(
                self._current_image_path, w, h, boxes, self.label_catalog.to_list()
            )
        except (OSError, ValueError) as error:
            self.annotation_save_label.setText(f"保存失败：{error}")
            self.statusBar().showMessage("标注尚未保存，请检查图片目录的写入权限后按 Ctrl+S 重试。")
            return False
        self.project_manager.annotations[self._current_image_path] = boxes
        self._boxes_dirty = False
        self.annotation_save_label.setText("已自动保存 ✓")
        self.annotation_save_label.setToolTip(str(saved_path))
        self.statusBar().showMessage(f"已保存 {len(boxes)} 个标注框", 3000)
        return True

    def _prepare_image_change(self):
        if self._slicer_thread is not None and self._slicer_thread.isRunning():
            self.statusBar().showMessage("正在保存切图，请完成后再切换图片。", 5000)
            return False
        self.canvas.cancel_pending_annotation()
        if not self._flush_annotations():
            QMessageBox.warning(self, "标注尚未保存", "保存失败，当前图片已保留。请检查写入权限后按 Ctrl+S 重试。")
            return False
        if self._current_image_path:
            self.project_manager.set_mask(self._current_image_path, self.canvas.get_mask())
        return True

    def _load_annotations(self, path, width, height):
        self._boxes_dirty = False
        self._annotation_load_error = False
        self.canvas.setEnabled(True)
        try:
            document = self.annotation_store.load(path, width, height)
            boxes = document["boxes"]
            if not os.path.exists(path + ".annotations.json"):
                boxes = self.project_manager.annotations.get(path, boxes)
                boxes = self.annotation_store.validate_boxes(boxes, width, height)
            # Merge portable labels, preserving existing bindings when aliases conflict.
            for record in document.get("label_catalog", []):
                records = self.label_catalog.to_list()
                by_name = {r["name"]: r for r in records}
                if record["name"] not in by_name:
                    try:
                        self.label_catalog.add_label(record["name"])
                    except ValueError:
                        self.statusBar().showMessage("导入类别与当前别名冲突，已有框的标签仍原样保留。", 7000)
                        continue
                for alias in record["aliases"]:
                    records = self.label_catalog.to_list()
                    target = next(r for r in records if r["name"] == record["name"])
                    if alias in target["aliases"]:
                        continue
                    target["aliases"].append(alias)
                    try:
                        self.label_catalog = LabelCatalog(records)
                    except ValueError:
                        self.statusBar().showMessage("部分导入别名发生冲突，已保留当前种类设置。", 7000)
            for box in boxes:
                try:
                    self.label_catalog.add_label(box["label"])
                except ValueError:
                    pass  # Existing annotation text is never re-resolved as an alias.
            self.canvas.set_annotations(boxes)
            self.box_count_label.setText(f"标注框：{len(boxes)}")
            self.annotation_save_label.setText("已载入标注" if boxes else "每次确认后自动保存到图片旁的 JSON 文件。")
            self._refresh_catalog()
        except (OSError, ValueError) as error:
            self._annotation_load_error = True
            self.canvas.set_annotations([])
            self.canvas.setEnabled(False)
            self.box_count_label.setText("标注加载失败")
            self.annotation_save_label.setText("原标注文件已保留，请修复文件后重新打开图片。")
            QMessageBox.warning(self, "无法读取标注", f"{error}\n\n为保护已有标注，当前图片暂时禁止编辑。")

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

        self.open_image_btn = QPushButton("打开图片/文件夹")
        self.open_image_btn.clicked.connect(self._open_image)

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

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        if not self._prepare_image_change():
            return

        # Normalize path separators
        path = os.path.normpath(path)
        folder = os.path.dirname(path)
        filename = os.path.basename(path)

        # Update source path label
        self.source_path_label.setText(folder)

        # Natural sort for images
        def natural_key(text):
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

        files = [
            name for name in os.listdir(folder)
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ]
        files.sort(key=natural_key)

        self._image_list = [
            os.path.normpath(os.path.join(folder, name))
            for name in files
        ]

        # Update Project Manager
        self.project_manager.set_images(self._image_list)
        self.project_manager.project_path = None

        try:
            self._image_index = self._image_list.index(path)
        except ValueError:
            # Fallback: try to find by basename if full path match fails
            target_base = os.path.basename(path)
            for i, p in enumerate(self._image_list):
                if os.path.basename(p) == target_base:
                    self._image_index = i
                    break
            else:
                self._image_index = 0

        # Auto-set Normal/Abnormal paths if not set or user wants convenience
        normal_dir = os.path.join(folder, "Normal")
        abnormal_dir = os.path.join(folder, "Abnormal")

        self.project_manager.normal_dir = normal_dir
        self.project_manager.abnormal_dir = abnormal_dir

        self.normal_path_label.setText(normal_dir)
        self.abnormal_path_label.setText(abnormal_dir)
        self._load_current_image()

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
        # Decode before replacing the visible document so a bad file cannot leave
        # the previous image paired with a new output filename.
        image = QImage(path)
        if image.isNull():
            QMessageBox.warning(self, "无法打开图片", f"无法读取：{path}")
            return
        if not self.canvas.load_image(path, image=image):
            QMessageBox.warning(self, "无法显示图片", f"无法显示：{path}")
            return
        self._current_image_path = path
        self._current_image = image
        # Update current image label
        filename = os.path.basename(path)
        current_idx = self._image_index + 1
        total_count = len(self._image_list)
        self.current_image_label.setText(f"当前图片: {filename} ({current_idx}/{total_count})")
        self.annotation_image_label.setText(filename)
        self.annotation_image_label.setToolTip(path)
        self.image_position_label.setText(f"{filename}  ·  {current_idx}/{total_count}")

        # Restore mask from ProjectManager
        mask = self.project_manager.get_mask(path)
        if mask is not None:
            self.canvas.set_mask(mask)
        elif self.mode_combo.currentIndex() == 1:
            self.canvas.ensure_slice_mask()

        self.source_path_label.setText(path)

        w, h = image.width(), image.height()
        self._load_annotations(path, w, h)

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
        if not self._image_list:
            return

        if not self._prepare_image_change():
            return

        self._image_index = max(self._image_index - 1, 0)
        self._load_current_image()

    def _on_next_image(self):
        if not self._image_list:
            return

        if not self._prepare_image_change():
            return

        self._image_index = min(self._image_index + 1, len(self._image_list) - 1)
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
        if "label_catalog" in config:
            self.label_catalog = LabelCatalog(config["label_catalog"])
            self._refresh_catalog()
        self.mode_combo.setCurrentIndex(config.get("mode", 0))

    def _on_new_project(self):
        if not self._prepare_image_change():
            return
        self.project_manager = ProjectManager()
        self._image_list = []
        self._image_index = -1
        self._current_image_path = None
        self._current_image = None
        self.canvas.reset_view()
        self.canvas.set_annotations([])
        self.canvas.setEnabled(True)
        self._boxes_dirty = False
        self._annotation_load_error = False
        self.annotation_image_label.setText("尚未打开图片")
        self.box_count_label.setText("标注框：0")
        self.annotation_save_label.setText("每次确认后自动保存到图片旁的 JSON 文件。")
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
        if not self._prepare_image_change():
            return

        try:
            current_index, config, normal_files, abnormal_files = self.project_manager.load_project(path)
            self._current_image_path = None
            self._current_image = None
            self._boxes_dirty = False
            self._annotation_load_error = False
            self.canvas.reset_view()
            self.canvas.setEnabled(True)
            self.annotation_image_label.setText("尚未打开图片")
            self.box_count_label.setText("标注框：0")
            self.annotation_save_label.setText("每次确认后自动保存到图片旁的 JSON 文件。")
            self.image_position_label.setText("工程中没有可显示的图片")
            self._apply_config(config)

            self._image_list = self.project_manager.images
            self._image_index = current_index

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
        self._save_global_config()
        super().closeEvent(event)

    def _get_global_config_path(self):
        return os.path.join(os.path.expanduser("~"), ".image_cutting_config.json")

    def _save_global_config(self):
        config = {
            **self._get_current_config(),
            "smart_grid": self.smart_grid_checkbox.isChecked(),
            "mask_opacity": self.mask_opacity_slider.value()
        }
        try:
            with open(self._get_global_config_path(), 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error saving global config: {e}")

    def _load_global_config(self):
        path = self._get_global_config_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
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
