# 工业视觉图像切分标注工具 (Industrial Image Slicer & Annotator) 开发指南

我现在想用 python 写一个 GUI 程序（应该只能是pyQt吧？），这个程序的主要功能是：

1. 可视化切分 一张较大的图像 到 较小的图像，其中切分小图像的大小和切分步长可以动态调整；
2. 切分的大小和步长要以可视化的方式呈现在 原始图像上，方便用户预先判断切分效果，可以用线条或者方框来实现，如果有更好的方式也可以；
3. 由于图片是工业场景下、包含缺陷的照片，所以切分出的、较小的图片会有“Normal”和“Abnormal”两种类型，考虑到切分的图像会有步长和大小设置，所以一个“缺陷”可能会被包含在很多很多张切分出来的小图片中。所以我不希望划分“Normal”和“Abnormal”两种类型的时候要对每一张照片进行手动分类，我希望我手动在原始图片中标记出“缺陷”的位置，程序自动判断，凡是包含“缺陷”的、切分后的小照片，都自动存储在“Abnormal”类型所设置的路径下。“缺陷”的标记方式呢，最好是画笔圈选或者荧光笔覆盖的方式，如果你有更好的方式也可以；
4. GUI上最少包含三个主要区块：a.原始图片区块：让用户标记“缺陷”，点击“切分”button，然后切分完毕后自动显示下一张图片；b.Normal区块：显示Normal路径文件夹下所有被写入的、无缺陷的图片缩略图（像电脑系统资源管理器那样）；c.Abnormal区块：显示Abnormal路径文件夹下所有被写入的、有缺陷的图片缩略图（像电脑系统资源管理器那样）；

我目前就想到了这些需求。这是我写的一个软件开发的指导文档，请你作为参考。在具体开发之前，请你先明确所要进行的todo lists，然后依次执行每一个任务，每一个任务执行之前，要先告诉我“爸爸，我现在在做{任务名称}任务”。

# 项目概述

本项目旨在开发一款基于 Python 的桌面 GUI 应用程序，用于协助工业质检（AOI）场景下的深度学习数据集制作。核心功能是将高分辨率工业图像依据用户定义的规则切分为小图，并根据用户的涂抹标记，自动将小图分类为“正常（Normal）”或“缺陷（Abnormal）”。

用的环境是Conda的imageCutting环境。你可以通过conda env list查看所有环境。

# 技术栈选型

| 组件 | 选型 | 理由 |
|:-:|:-:|:-:|
| 语言 | Python 3.12 | 生态丰富，开发效率高。 |
| GUI 框架 | PySide6 (Qt for Python) | 相比 PyQt6 协议更友好（LGPL），且拥有强大的 Graphics View Framework，完美支持图层、缩放、标注交互。 |
| 并发处理 | QThread / QThreadPool | **(新增)** 切分大图是计算密集型任务，必须在后台线程执行，防止界面卡死。 |
| 图像处理 | OpenCV (cv2) & NumPy | 高性能矩阵运算，处理遮罩（Mask）判断 overlap 速度极快。 |
| 图像加载 | Pillow (PIL) | 用于 UI 显示时的图像转换，配合 Qt 的 QImage。 |
| 配置/数据 | JSON | 保存用户的切分配置、路径设置以及**工程进度状态**。 |

# UI/UX 设计方案

## 整体布局设计

推荐采用 “左主右辅” 或 “三栏式” 布局，采用深色主题（Dark Mode）以减少视觉疲劳。

### 顶部工具栏 (Toolbar)
- 文件操作：新建/打开工程、保存进度、导出数据。
- 标注工具：
  - 画笔 (Brush)：自由涂抹。
  - **(新增) 矩形 (Rectangle)**：快速框选规则缺陷。
  - **(新增) 多边形 (Polygon)**：处理不规则但边缘清晰的缺陷。
  - 橡皮擦 (Eraser)、清除所有 (Clear)。
- 笔刷设置：调节笔刷大小滑动条。
- 视图控制：适应窗口、1:1 显示、网格显隐开关、**网格线自动反色/双色虚线**（适应不同背景）。

### 左侧：参数控制区 (Control Panel)

- 路径设置：原始图源路径、Normal 输出路径、Abnormal 输出路径。
- 切分参数：
  - 切分宽/高 (Patch Size, e.g., 256x256)
  - 步长 (Stride, e.g., 128) - 支持滑动条动态拖动，实时看效果
- 判定阈值 (Overlap Threshold)：缺陷像素占比多少算 Abnormal (0% - 100%)。
- **(新增) 视图设置**：
  - **Mask 透明度 (Opacity)**：滑动条 (0-100%)，调节红色遮罩的透明度，便于查看底部细节。
- 操作按钮：巨大的“执行切分 (Slice)”按钮、“下一张 (Next)”按钮。

### 中间：核心工作区 (Canvas)

- 这是 QGraphicsView 组件。
- 底层：原始图像。
- 中间层：缺陷标记层（支持半透明红色涂抹，透明度可调）。
- 顶层：切分网格预览层（黄色或青色细线，根据步长动态渲染）。

### 右侧/底部：结果预览区 (Gallery)

- 使用 QTabWidget 分为两个标签页：[ Normal ] 和 [ Abnormal ]。
- 内容使用 QListWidget (IconMode)。
- **(新增) 性能优化**：实现**分页加载**或**滚动懒加载**，防止一次性加载数千张缩略图导致内存溢出或界面卡顿。
- 关键交互：双击缩略图可以弹窗查看大图，右键菜单支持“移动到另一类”（用于修正自动分类的错误）。

## 颜色搭配建议 (Dark Theme)

- 背景色：#2D2D2D (深灰，护眼)
- 面板背景：#3C3F41
- 强调色 (Accent)：#007ACC (VSCode 蓝) 或 #4CAF50 (绿色，用于执行按钮)
- 缺陷标记色：RGBA(255, 0, 0, 128) (半透明红，透明度可调)
- 网格线颜色：#00FFFF (青色) 或 #FFFF00 (黄色) - 需与工业灰度图形成高反差。
- Normal 文件夹标识：绿色系图标
- Abnormal 文件夹标识：红色系图标

# 交互逻辑详解

## 核心操作流程

1. **工程管理 (Project Management)**：
   - 建议引入“工程”概念，保存当前处理的文件索引、配置参数和**每张图的 Mask 数据**。
   - 避免程序崩溃或中途退出导致标注数据丢失。

2. 加载：用户打开工程或选择“原始图片文件夹”。程序加载第一张图。

3. 调整：用户调整 Patch Size 和 Stride，画布上实时更新网格线，帮助用户预判切分位置是否会把缺陷切碎。

4. 标记：
   - 滚轮：放大/缩小图片。
   - 按住中键/空格+左键：拖拽平移图片。
   - 标注工具：使用画笔、矩形或多边形工具标记缺陷。程序后台维护一个与原图同尺寸的二值化 Mask。

5. 切分与分类 (算法逻辑) - **(后台线程执行)**：
   - 点击“切分”后，启动后台线程，避免界面卡死。
   - 进度条显示切分进度 (0% -> 100%)。
   - **(新增) 数据闭环**：在保存 Abnormal 小图时，可选项：**同步保存对应的 Mask 小图** (用于语义分割训练)。
   - 逻辑：
     1. 程序按照 Grid 遍历图像。
     2. 对于每一个 Patch $(x, y, w, h)$：
        - 截取原图 ROI。
        - 截取 Mask ROI。
        - 计算 Mask ROI 中白色像素的数量/占比。
        - If pixel_count > threshold: Save to Abnormal_Dir (and optionally save Mask ROI).
        - Else: Save to Normal_Dir.

6. 反馈：切分完成后，右侧预览区刷新显示（懒加载）。

7. 流转：点击“下一张”，自动保存当前状态，清空画布，加载下一张图。

## 快捷键设计

- Ctrl + Z：撤销上一步涂抹。
- [ 和 ]：快速调节笔刷大小。
- Space (按住)：切换为抓手工具（Pan）。
- Enter：执行切分并自动下一张（熟练工模式）。
- **(新增) B / R / P**：切换 画笔(Brush) / 矩形(Rectangle) / 多边形(Polygon)。

# 核心代码架构 (Model-View-Controller)

为了代码可维护性，建议严格分离逻辑。

## Model (数据层)

- **ImageProcessor**: 负责 OpenCV 的图像裁剪、Mask 生成、像素统计、文件保存。
- **ProjectManager**: **(新增)** 负责管理工程状态（当前图片索引、所有图片的 Mask 路径、配置参数）。
- **ConfigManager**: 负责读写 JSON 配置文件。

## View (视图层 - UI)

- **MainWindow**: 主窗口框架。
- **CanvasWidget** (继承自 QGraphicsView):
  - 重写 mousePressEvent, mouseMoveEvent 实现画笔/形状绘制。
  - 重写 wheelEvent 实现以鼠标为中心的缩放。
  - **LayerManager**: 管理原图层、Mask Item 层、Grid Item 层。
- **GalleryWidget**: 封装右侧的缩略图显示逻辑，**实现懒加载**。

## Controller (逻辑控制)

- 连接 View 的信号（如“切分按钮点击”）到 Model 的函数。
- **SlicerThread (QThread)**: **(新增)** 专门负责切分运算的后台线程，发送 `progress_updated(int)` 信号给 UI。
- 处理“下一张”逻辑：检查文件列表索引，通知 View 更新图片。

# 关键技术实现难点与伪代码

## 画笔涂抹与 Mask 同步

不要直接在原图像素上画。要使用 双层绘图。

Qt 界面上：使用 `QGraphicsPathItem` (画笔), `QGraphicsRectItem` (矩形) 等在透明层上绘制。

后台数据：同时在一个 `numpy.zeros (uint8)` 矩阵上，用 `cv2.circle`, `cv2.rectangle`, `cv2.fillPoly` 绘制白色区域。

重点：Qt 的坐标系和 OpenCV 的矩阵坐标系必须严格对应，缩放时要进行坐标映射 (mapToScene)。

## 自动分类算法逻辑 (后台线程版)

```python
class SlicerThread(QThread):
    progress_signal = Signal(int)
    finished_signal = Signal(dict) # 返回统计结果

    def run(self):
        h, w = self.image.shape[:2]
        total_patches = (h // stride) * (w // stride)
        count = 0
        
        for y in range(0, h, self.stride):
            for x in range(0, w, self.stride):
                if self.isInterruptionRequested(): return
                
                # ... (切分逻辑同前) ...
                
                # 3. 保存
                if is_abnormal:
                    cv2.imwrite(img_path, img_roi)
                    if self.save_mask:
                        cv2.imwrite(mask_path, mask_roi) # 同步保存 Mask
                else:
                    cv2.imwrite(normal_path, img_roi)
                
                count += 1
                self.progress_signal.emit(int(count / total_patches * 100))
```

## 性能优化：网格绘制

不要创建成百上千个 QGraphicsRectItem，这会极其消耗内存。
优化方案：
自定义一个 GridItem 继承自 QGraphicsItem，只重写 paint 方法。在 paint 方法中，根据当前的 Viewport（可视区域）计算出需要画哪几条线，只画屏幕看得到的线。

```python
def paint(self, painter, option, widget):
    # 根据 rect (可视区域) 和 stride 计算起止点
    # 使用 painter.drawLine 批量绘制
    pass
```

# 开发阶段规划

## Phase 1 (原型验证):
- 搭建基础 PySide6 窗口。
- 实现图片加载显示。
- 实现 Patch Size / Stride 输入框，并在图上画出静态网格 (GridItem)。

## Phase 2 (核心交互):
- 实现画笔、矩形工具。
- 实现后台 Mask 生成逻辑与同步。
- **实现多线程切分算法**和文件保存。

## Phase 3 (数据闭环与预览):
- 实现右侧 Normal/Abnormal 预览区 (**懒加载**)。
- 实现工程状态保存 (JSON)。
- 添加“下一张/上一张”切换逻辑。

## Phase 4 (优化):
- 增加缩放/漫游功能。
- 增加撤销功能。
- 增加 Mask 透明度调节。
- 美化 UI。

# 总结

这个工具的核心价值在于**“可视化预判”、“半自动分类”和“工程化闭环”**。通过让用户在切分前就能看到网格与缺陷的相对位置，可以极大提高数据集的质量；通过涂抹 Mask 自动分类，大幅降低人工成本；通过多线程和工程管理，确保工具在大规模工业数据处理中的稳定性和流畅性。
