import os
import sys
import shutil
import json
import re

import numpy as np
import cv2
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal, QThread
from PySide6.QtGui import QAction, QActionGroup, QBrush, QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QCheckBox,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QMenu,
    QGroupBox,
    QGridLayout,
    QMessageBox,
)

class ScalableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._original_pixmap = None

    def setPixmap(self, pixmap):
        self._original_pixmap = pixmap
        self.update() # Trigger paintEvent

    def paintEvent(self, event):
        if not self._original_pixmap or self._original_pixmap.isNull():
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Calculate target size keeping aspect ratio
        target_rect = QRectF(self.rect())
        img_size = self._original_pixmap.size()
        scaled_size = img_size.scaled(target_rect.size().toSize(), Qt.KeepAspectRatio)
        
        # Center the image
        x = (target_rect.width() - scaled_size.width()) / 2
        y = (target_rect.height() - scaled_size.height()) / 2
        
        painter.drawPixmap(int(x), int(y), scaled_size.width(), scaled_size.height(), self._original_pixmap)

class GridItem(QGraphicsItem):
    def __init__(self, rect: QRectF, stride: int, patch_width: int, patch_height: int):
        super().__init__()
        self._rect = rect
        self._stride = stride
        self._patch_width = patch_width
        self._patch_height = patch_height
        self._pen = QPen(Qt.cyan)
        self._pen.setWidth(1)
        self._highlight_rect = QRectF()
        # Enhance highlight style: Magenta color, thicker line (5px)
        self._highlight_pen = QPen(QColor(255, 0, 255)) # Magenta
        self._highlight_pen.setWidth(5)
        self._smart_grid = False

    def set_smart_grid(self, enabled: bool):
        self._smart_grid = enabled
        self.update()

    def set_highlight_rect(self, rect: QRectF):
        self._highlight_rect = rect
        self.update()

    def set_rect(self, rect: QRectF):
        self.prepareGeometryChange()
        self._rect = rect
        self.update()

    def set_stride(self, stride: int):
        self._stride = max(1, stride)
        self.update()

    def set_patch_size(self, width: int, height: int):
        self._patch_width = max(1, width)
        self._patch_height = max(1, height)
        self.update()

    def boundingRect(self) -> QRectF:
        return self._rect

    def get_patch_counts(self):
        if self._rect.isNull() or self._stride <= 0:
            return 0, 0
        w = int(self._rect.width())
        h = int(self._rect.height())
        cols = max(0, (w - self._patch_width) // self._stride + 1)
        rows = max(0, (h - self._patch_height) // self._stride + 1)
        return rows, cols

    def paint(self, painter: QPainter, option, widget=None):
        if self._rect.isNull() or self._stride <= 0:
            return
        
        # Viewport culling optimization
        exposed = option.exposedRect
        left_bound = int(max(self._rect.left(), exposed.left()))
        right_bound = int(min(self._rect.right(), exposed.right()))
        top_bound = int(max(self._rect.top(), exposed.top()))
        bottom_bound = int(min(self._rect.bottom(), exposed.bottom()))

        start_x = int(self._rect.left())
        start_y = int(self._rect.top())
        
        start_col = max(0, int((left_bound - self._patch_width - start_x) // self._stride))
        if start_x + start_col * self._stride + self._patch_width < left_bound:
             start_col += 1
             
        start_row = max(0, int((top_bound - self._patch_height - start_y) // self._stride))
        if start_y + start_row * self._stride + self._patch_height < top_bound:
            start_row += 1

        def draw_loop():
            y = start_y + start_row * self._stride
            while y + self._patch_height <= int(self._rect.bottom()) + 1:
                if y > bottom_bound:
                    break
                    
                x = start_x + start_col * self._stride
                while x + self._patch_width <= int(self._rect.right()) + 1:
                    if x > right_bound:
                        break
                    painter.drawRect(x, y, self._patch_width, self._patch_height)
                    x += self._stride
                y += self._stride

        if self._smart_grid:
            # Draw black solid line first
            black_pen = QPen(Qt.black)
            black_pen.setWidth(1)
            painter.setPen(black_pen)
            draw_loop()
            
            # Draw white dashed line on top
            white_pen = QPen(Qt.white)
            white_pen.setWidth(1)
            white_pen.setStyle(Qt.DashLine)
            painter.setPen(white_pen)
            draw_loop()
        else:
            painter.setPen(self._pen)
            draw_loop()
            
        # Draw highlight
        if not self._highlight_rect.isNull():
            painter.setPen(self._highlight_pen)
            painter.drawRect(self._highlight_rect)


class CanvasWidget(QGraphicsView):
    # Signal to notify when a point is clicked (x, y)
    canvas_clicked = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag) # Disable drag since we always fit in view
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        # Disable scrollbars to force fit
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._grid_item = GridItem(QRectF(), 128, 256, 256)
        self._scene.addItem(self._grid_item)
        self._mask_item = QGraphicsPixmapItem()
        self._mask_item.setOpacity(0.5)
        self._scene.addItem(self._mask_item)
        self.setBackgroundBrush(Qt.black)
        self._mask = None
        self._mask_rgba = None
        self._undo_stack = []
        self._redo_stack = []

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.canvas_clicked.emit(scene_pos.x(), scene_pos.y())
        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_in_view()

    def _fit_in_view(self):
        if not self._pixmap_item.pixmap().isNull():
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def load_image(self, path: str):
        # Standard behavior: Ctrl + Wheel for zoom.
        if event.modifiers() & Qt.ControlModifier:
            zoom_in = event.angleDelta().y() > 0
            factor = 1.25 if zoom_in else 0.8
            self.scale(factor, factor)
            event.accept()
        else:
            super().wheelEvent(event)

    def load_image(self, path: str):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return False
        self._pixmap_item.setPixmap(pixmap)
        rect = QRectF(pixmap.rect())
        self._scene.setSceneRect(rect)
        self._grid_item.set_rect(rect)
        self._mask = np.zeros((pixmap.height(), pixmap.width()), dtype=np.uint8)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_mask_item()
        self._fit_in_view() # Use the helper method
        return True

    def set_stride(self, stride: int):
        self._grid_item.set_stride(stride)

    def set_smart_grid(self, enabled: bool):
        self._grid_item.set_smart_grid(enabled)

    def set_patch_size(self, width: int, height: int):
        self._grid_item.set_patch_size(width, height)

    def set_highlight_rect(self, rect: QRectF):
        self._grid_item.set_highlight_rect(rect)

    def get_patch_counts(self):
        return self._grid_item.get_patch_counts()

    def set_mask_opacity(self, opacity: float):
        self._mask_item.setOpacity(opacity)

    def clear_mask(self):
        if self._mask is None:
            return
        self.push_undo()
        self._mask.fill(0)
        self._update_mask_item()

    def reset_view(self):
        self._pixmap_item.setPixmap(QPixmap())
        self._mask = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_mask_item()
        self._grid_item.set_rect(QRectF())
        self.scene().setSceneRect(QRectF())

    def push_undo(self):
        if self._mask is None:
            return
        self._undo_stack.append(self._mask.copy())
        if len(self._undo_stack) > 20:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self):
        if not self._undo_stack:
            return
        if self._mask is not None:
             self._redo_stack.append(self._mask.copy())
        self._mask = self._undo_stack.pop()
        self._update_mask_item()

    def redo(self):
        if not self._redo_stack:
            return
        if self._mask is not None:
            self._undo_stack.append(self._mask.copy())
        self._mask = self._redo_stack.pop()
        self._update_mask_item()

    def _update_mask_item(self):
        if self._mask is None:
            self._mask_item.setPixmap(QPixmap())
            return
        h, w = self._mask.shape
        self._mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        self._mask_rgba[..., 0] = 255
        self._mask_rgba[..., 3] = self._mask
        image = QImage(self._mask_rgba.data, w, h, QImage.Format_RGBA8888)
        self._mask_item.setPixmap(QPixmap.fromImage(image))

    def fill_rect(self, rect: QRectF, value: int):
        if self._mask is None:
            return
        self.push_undo()
        x = int(rect.x())
        y = int(rect.y())
        w = int(rect.width())
        h = int(rect.height())
        cv2.rectangle(self._mask, (x, y), (x + w, y + h), value, -1)
        self._update_mask_item()

    def get_mask(self):
        if self._mask is None:
            return None
        return self._mask.copy()

    def set_mask(self, mask):
        if mask is None:
            return
        # Ensure mask matches image size if possible, or just accept it
        self._mask = mask.copy()
        self._update_mask_item()



class ProjectManager:
    def __init__(self):
        self.project_path = None
        self.images = []
        self.masks = {} # map path -> mask (numpy array)
        self.current_index = -1
        self.config = {}
        self.normal_dir = None
        self.abnormal_dir = None

    def set_images(self, images):
        self.images = images
        # Clean up masks that are not in the new image list? 
        # Maybe keep them in case user adds them back.
    
    def set_mask(self, image_path, mask):
        if mask is not None:
            self.masks[image_path] = mask.copy()
    
    def get_mask(self, image_path):
        return self.masks.get(image_path)
    
    def save_project(self, json_path, current_index, config, normal_files=None, abnormal_files=None):
        base_dir = os.path.dirname(json_path)
        project_name = os.path.splitext(os.path.basename(json_path))[0]
        mask_dir = os.path.join(base_dir, project_name + "_masks")
        os.makedirs(mask_dir, exist_ok=True)
        
        mask_paths_map = {}
        for img_path, mask_arr in self.masks.items():
            # Create a unique filename for the mask
            # Using hash of path to avoid filename collisions if multiple files have same name in different dirs
            import hashlib
            name_hash = hashlib.md5(img_path.encode('utf-8')).hexdigest()
            base_name = os.path.basename(img_path)
            mask_filename = f"{os.path.splitext(base_name)[0]}_{name_hash}.png"
            save_path = os.path.join(mask_dir, mask_filename)
            cv2.imencode(".png", mask_arr)[1].tofile(save_path)
            mask_paths_map[img_path] = os.path.relpath(save_path, base_dir)
            
        data = {
            "version": "1.0",
            "images": self.images,
            "current_index": current_index,
            "config": config,
            "masks": mask_paths_map,
            "normal_files": normal_files if normal_files is not None else [],
            "abnormal_files": abnormal_files if abnormal_files is not None else []
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        self.project_path = json_path

    def load_project(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.project_path = json_path
        self.images = data.get("images", [])
        self.current_index = data.get("current_index", -1)
        self.config = data.get("config", {})
        normal_files = data.get("normal_files", [])
        abnormal_files = data.get("abnormal_files", [])
        
        base_dir = os.path.dirname(json_path)
        mask_map = data.get("masks", {})
        self.masks = {}
        
        for img_path, mask_rel_path in mask_map.items():
            full_path = os.path.join(base_dir, mask_rel_path)
            if os.path.exists(full_path):
                mask = cv2.imdecode(np.fromfile(full_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    self.masks[img_path] = mask
        
        return self.current_index, self.config, normal_files, abnormal_files


class SlicerThread(QThread):
    progress_signal = Signal(int)
    finished_signal = Signal(list, list)

    def __init__(
        self,
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
    ):
        super().__init__()
        self._image = image
        self._mask = mask
        self._patch_w = patch_w
        self._patch_h = patch_h
        self._stride = stride
        self._threshold = threshold
        self._normal_dir = normal_dir
        self._abnormal_dir = abnormal_dir
        self._save_mask = save_mask
        self._mask_dir = mask_dir
        self._base_name = base_name

    def run(self):
        h, w = self._image.shape[:2]
        total = max(1, ((h - self._patch_h) // self._stride + 1) * ((w - self._patch_w) // self._stride + 1))
        count = 0
        normal_paths = []
        abnormal_paths = []
        for y in range(0, h - self._patch_h + 1, self._stride):
            for x in range(0, w - self._patch_w + 1, self._stride):
                img_roi = self._image[y : y + self._patch_h, x : x + self._patch_w]
                mask_roi = self._mask[y : y + self._patch_h, x : x + self._patch_w]
                defect_ratio = float(cv2.countNonZero(mask_roi)) / float(self._patch_w * self._patch_h)
                filename = f"{self._base_name}_{x}_{y}.png"
                is_abnormal = defect_ratio > self._threshold
                
                if is_abnormal:
                    out_path = os.path.join(self._abnormal_dir, filename)
                    # Use imencode/tofile to support Chinese paths
                    cv2.imencode(os.path.splitext(filename)[1], img_roi)[1].tofile(out_path)
                    abnormal_paths.append(out_path)
                    if self._save_mask and self._mask_dir is not None:
                        mask_path = os.path.join(self._mask_dir, filename)
                        cv2.imencode(os.path.splitext(filename)[1], mask_roi)[1].tofile(mask_path)
                else:
                    out_path = os.path.join(self._normal_dir, filename)
                    cv2.imencode(os.path.splitext(filename)[1], img_roi)[1].tofile(out_path)
                    normal_paths.append(out_path)
                count += 1
                progress = int(count / total * 100)
                self.progress_signal.emit(progress)
        self.finished_signal.emit(normal_paths, abnormal_paths)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Industrial Image Slicer & Annotator")
        self.resize(1400, 900)
        self.project_manager = ProjectManager()
        self._create_actions()
        self._build_ui()
        self._load_global_config()
        self._current_image_path = None
        self._current_image = None
        self._image_list = []
        self._image_index = -1
        self._thumbnail_queue = { }
        self._thumbnail_timer = None
        self._slicer_thread = None
        self._patch_coords = []
        self._current_patch_index = 0


    def _create_actions(self):
        # File Actions
        self.new_project_action = QAction("新建工程", self)
        self.new_project_action.triggered.connect(self._on_new_project)
        
        self.open_project_action = QAction("打开工程...", self)
        self.open_project_action.setShortcut("Ctrl+O")
        self.open_project_action.triggered.connect(self._on_open_project)
        
        self.save_project_action = QAction("保存工程", self)
        self.save_project_action.setShortcut("Ctrl+S")
        self.save_project_action.triggered.connect(self._on_save_project)
        
        self.save_project_as_action = QAction("工程另存为...", self)
        self.save_project_as_action.triggered.connect(self._on_save_project_as)

        # Edit Actions
        self.undo_action = QAction("撤销", self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.triggered.connect(lambda: self.canvas.undo())

        self.redo_action = QAction("重做", self)
        self.redo_action.setShortcut("Ctrl+Shift+Z")
        self.redo_action.triggered.connect(lambda: self.canvas.redo())

        # Initialize Canvas early for signal connections
        self.canvas = CanvasWidget()

        self.slice_and_next_action = QAction("Slice and Next", self)
        self.slice_and_next_action.setShortcut("Return")
        self.slice_and_next_action.triggered.connect(self._on_slice_and_next)
        
        self.addAction(self.slice_and_next_action)

    def _build_ui(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
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
        
        # 2. Center Panel (Patch Viewer Only)
        self.patch_viewer = self._build_patch_viewer()
        self.patch_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 3. Gallery (Right)
        self.gallery = self._build_gallery()

        self.main_splitter.addWidget(self.control_panel)
        self.main_splitter.addWidget(self.patch_viewer)
        self.main_splitter.addWidget(self.gallery)
        
        # Set stretch factors for main splitter
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        
        # Set initial sizes
        self.main_splitter.setSizes([350, 1000, 320])

        self.setCentralWidget(self.main_splitter)

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
            if is_folder:
                # Adjust _select_folder to work with QLabel
                btn.clicked.connect(lambda: self._select_folder(label_widget))
            else:
                 pass 
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

        self._load_current_image()
        
        # Auto-set Normal/Abnormal paths if not set or user wants convenience
        normal_dir = os.path.join(folder, "Normal")
        abnormal_dir = os.path.join(folder, "Abnormal")
        
        self.project_manager.normal_dir = normal_dir
        self.project_manager.abnormal_dir = abnormal_dir
        
        self.normal_path_label.setText(normal_dir)
        self.abnormal_path_label.setText(abnormal_dir)

    def _schedule_param_update(self):
        self._param_update_timer.start()

    def _update_grid_params(self):
        # Grid should visualize the Unit Modules (Stride x Stride), not the Patches
        stride = self.stride_spin.value()
        self.canvas.set_patch_size(stride, stride)
        self.canvas.set_stride(stride)
        self._update_patch_count_label()
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
        self._current_image_path = path
        
        # Update current image label
        filename = os.path.basename(path)
        current_idx = self._image_index + 1
        total_count = len(self._image_list)
        self.current_image_label.setText(f"当前图片: {filename} ({current_idx}/{total_count})")
        
        self.canvas.load_image(path)
        
        # Restore mask from ProjectManager
        mask = self.project_manager.get_mask(path)
        if mask is not None:
            self.canvas.set_mask(mask)
            
        self.source_path_label.setText(path)
        
        # Fix for Chinese path reading in OpenCV (Windows)
        try:
            self._current_image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error reading image: {e}")
            self._current_image = None
            
        # Update patch count immediately when image loaded
        self._update_patch_count_label()
        
        # Initialize patch review
        self._init_patch_review()
        
        # Filter gallery to show only current image's patches
        self._filter_gallery_by_current_image()

    def _on_prev_image(self):
        if not self._image_list:
            return
        
        # Save current mask
        if self._current_image_path:
            self.project_manager.set_mask(self._current_image_path, self.canvas.get_mask())
            
        self._image_index = max(self._image_index - 1, 0)
        self._load_current_image()

    def _on_next_image(self):
        if not self._image_list:
            return
        
        # Save current mask
        if self._current_image_path:
            self.project_manager.set_mask(self._current_image_path, self.canvas.get_mask())
            
        self._image_index = min(self._image_index + 1, len(self._image_list) - 1)
        self._load_current_image()

    def _get_current_config(self):
        return {
            "patch_width": self.patch_width_spin.value(),
            "patch_height": self.patch_height_spin.value(),
            "stride": self.stride_spin.value(),
            "threshold": self.threshold_spin.value(),
            "normal_dir": self.normal_path_label.text(),
            "abnormal_dir": self.abnormal_path_label.text()
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

    def _on_new_project(self):
        self.project_manager = ProjectManager()
        self._image_list = []
        self._image_index = -1
        self._current_image_path = None
        self._current_image = None
        self.canvas.reset_view()
        self.source_path_edit.clear()
        self.normal_list.clear()
        self.abnormal_list.clear()
        self.setWindowTitle("Industrial Image Slicer & Annotator - New Project")

    def _on_open_project(self):
        from PySide6.QtWidgets import QMessageBox
        path, _ = QFileDialog.getOpenFileName(self, "打开工程", "", "Project Files (*.json)")
        if not path:
            return
        
        try:
            current_index, config, normal_files, abnormal_files = self.project_manager.load_project(path)
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
        self._save_global_config()
        super().closeEvent(event)

    def _get_global_config_path(self):
        return os.path.join(os.path.expanduser("~"), ".image_cutting_config.json")

    def _save_global_config(self):
        config = {
            "patch_width": self.patch_width_spin.value(),
            "patch_height": self.patch_height_spin.value(),
            "stride": self.stride_spin.value(),
            "threshold": self.threshold_spin.value(),
            "normal_dir": self.normal_path_label.text(),
            "abnormal_dir": self.abnormal_path_label.text(),
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
        self._auto_next = True
        self._on_slice()

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
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(abnormal_dir, exist_ok=True)
        save_mask = False
        mask_dir = None
        if save_mask:
            mask_dir = os.path.join(abnormal_dir, "mask")
            os.makedirs(mask_dir, exist_ok=True)
        patch_w = self.patch_width_spin.value()
        patch_h = self.patch_height_spin.value()
        stride = self.stride_spin.value()
        threshold = self.threshold_spin.value() / 100.0
        image = self._current_image
        mask = self.canvas.get_mask()
        if mask is None:
            return
        h, w = image.shape[:2]
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
        self._slicer_thread.finished_signal.connect(self._on_slice_finished)
        self._slicer_thread.finished.connect(self._on_thread_finished)
        self._slicer_thread.start()

    def _init_patch_review(self):
        self._patch_coords = []
        self._current_patch_index = 0
        if self._current_image is None:
            self._update_patch_view()
            return

        h, w = self._current_image.shape[:2]
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
        h_img, w_img = self._current_image.shape[:2]
        unit_w = min(stride, w_img - x)
        unit_h = min(stride, h_img - y)
        
        self.canvas.set_highlight_rect(QRectF(x, y, unit_w, unit_h))
        
        # Extract unit module
        if self._current_image is not None:
            patch = self._current_image[y : y + unit_h, x : x + unit_w]
            # Convert to QPixmap
            h, w, ch = patch.shape
            bytes_per_line = ch * w
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            image = QImage(patch_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            
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
        h_img, w_img = self._current_image.shape[:2]
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
        h_img, w_img = self._current_image.shape[:2]
        
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


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
