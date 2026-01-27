# 工业视觉图像切分标注工具

工业视觉图像切分标注工具（Industrial Image Slicer & Annotator）是一个基于 Python 的桌面 GUI 应用，主要用于工业质检（AOI）场景下的数据集制作。它可以将高分辨率工业图像按照用户设定的规则切分为小图，并根据用户在原图上的“缺陷”标注，自动将小图分类为 Normal（正常）和 Abnormal（缺陷）。

> 目标：在尽量减少人工逐张分类工作的前提下，高效构建高质量的工业缺陷检测 / 分割数据集。

## 功能特性

- 图像切分
  - 支持将大图切分为固定尺寸的小图（可配置宽、高）。
  - 支持设置步长（Stride），控制小图之间的重叠程度。
  - 切分网格以叠加层的形式实时显示在原图上，方便预判切分效果。

- 缺陷标注与自动分类
  - 支持在原始图像上通过画笔、矩形、多边形等工具标记“缺陷”区域。
  - 内部维护与原图同尺寸的二值 Mask 矩阵（NumPy），用于记录缺陷区域。
  - 切分时自动判断每个小图与 Mask 的重叠比例，根据阈值自动归类为 Normal 或 Abnormal。
  - 可选项：在保存 Abnormal 小图时，同时保存对应的 Mask 小图，用于语义分割任务。

- GUI 交互
  - 采用 PySide6 构建桌面 GUI，整体采用暗色主题以减轻视觉疲劳。
  - 左侧参数控制区：路径设置、切分尺寸、步长、阈值、Mask 透明度等。
  - 中间画布区：基于 QGraphicsView 的多图层 Canvas（原图层 / Mask 层 / 网格层）。
  - 右侧预览区：Normal / Abnormal 两个标签页，以缩略图形式展示切分结果，支持懒加载和翻页。

- 性能与稳定性
  - 使用 QThread / QThreadPool 将切分和保存任务放在后台线程中执行，避免界面卡死。
  - 通过工程管理功能（Project），将当前进度、配置参数、Mask 数据等持久化到 JSON 文件中，防止崩溃或中断导致数据丢失。

## 技术栈

- 语言：Python 3.12
- GUI 框架：PySide6 (Qt for Python)
- 图像处理：OpenCV (cv2)、NumPy
- 图像加载与转换：Pillow (PIL)
- 配置与工程管理：JSON

运行环境建议使用 Conda 独立环境，以便隔离依赖。

## 项目结构（当前原型）

项目当前处于原型开发阶段，目录结构大致如下：

- `src/main.py`：应用入口，后续将逐步演化为包含主窗体、Canvas、Gallery 等组件的主程序文件。
- `intro.md`：详细的开发设计与需求说明文档（推荐开发者优先阅读）。
- `ImageSlicer.spec`：PyInstaller 打包配置文件（用于生成可执行程序）。
- `build_exe.bat`：打包脚本，方便在 Windows 环境下一键构建 exe。

> 注意：随着开发推进，`src/` 目录下会拆分出更多模块（Model / View / Controller），包括 `ImageProcessor`、`ProjectManager`、`ConfigManager`、`CanvasWidget`、`GalleryWidget`、`SlicerThread` 等。

## 快速开始

### 1. 准备运行环境

1. 安装 Anaconda / Miniconda。
2. 创建并激活专用环境（名称以实际为准，例如 `AnomaSlice`）：

   ```bash
   conda create -n AnomaSlice python=3.12
   conda activate AnomaSlice
   ```

3. 安装依赖（示例）：

   ```bash
   pip install pyside6 opencv-python numpy pillow
   ```

> 具体依赖列表可根据 `src/main.py` 和后续模块的实际 import 适当增减。

### 2. 运行程序

在项目根目录下执行：

```bash
cd src
python main.py
```

如果你使用的是 Conda 环境，请确保已经激活对应环境（例如 `conda activate AnomaSlice`）。

## 基本使用流程

1. 启动程序后，先在左侧控制面板中设置：
   - 原始图片文件夹路径；
   - Normal 输出路径；
   - Abnormal 输出路径；
   - 切分尺寸（宽 / 高）、步长（Stride）；
   - 判定阈值（缺陷像素占比多少算 Abnormal）；
   - Mask 透明度等视觉辅助参数。

2. 通过顶部菜单 / 工具栏加载第一张原始图像。

3. 在画布中：
   - 使用滚轮缩放、按空格拖拽移动视图；
   - 使用画笔、矩形、多边形工具在缺陷区域涂抹标注；
   - 实时观察切分网格与缺陷区域的相对位置，必要时调整 Patch Size / Stride。

4. 点击“执行切分 (Slice)”按钮：
   - 后台线程开始遍历所有 Patch；
   - 根据 Mask 重叠比例自动将切分图保存到 Normal / Abnormal 目录；
   - 可选：同步保存 Abnormal 对应的 Mask 小图。

5. 切分完成后：
   - 右侧 Normal / Abnormal 预览区刷新，懒加载显示缩略图；
   - 双击缩略图可放大查看，右键菜单支持将图片在 Normal / Abnormal 之间移动（修正分类）。

6. 点击“下一张 (Next)”按钮：
   - 自动保存当前工程状态（包括 Mask）；
   - 清空画布并加载下一张原始图片。

## 快捷键（规划中）

- `Ctrl + Z`：撤销上一步涂抹。
- `[` / `]`：减小 / 增大笔刷大小。
- `Space`（按住）：抓手工具，拖拽画布平移。
- `Enter`：执行切分并自动进入下一张（面向熟练操作员）。
- `B` / `R` / `P`：切换画笔（Brush）、矩形（Rectangle）、多边形（Polygon）工具。

## 开发者指南

如果你是本项目的开发者或贡献者，建议按以下顺序了解代码：

1. 阅读 [`intro.md`](intro.md)，了解整体需求、UI/UX 设计、交互流程和架构规划。
2. 从 [`src/main.py`](src/main.py) 入手，熟悉应用入口和主窗体结构。
3. 按照 MVC 思路规划和拆分模块：
   - Model：`ImageProcessor`、`ProjectManager`、`ConfigManager`。
   - View：`MainWindow`、`CanvasWidget`、`GalleryWidget` 等。
   - Controller：信号槽连接逻辑、多线程切分控制、工程状态管理等。
4. 优先实现 `intro.md` 中“开发阶段规划”里的 Phase 1 → Phase 4 功能，逐步迭代。

在实现过程中，务必注意：

- 始终在后台线程中执行重计算和 IO 操作，避免阻塞 UI。
- 使用统一的坐标系映射确保 Qt 视图坐标与 OpenCV 中的像素坐标一一对应。
- 在处理大量小图保存、缩略图预览时，关注内存占用与懒加载策略。

## 打包与发布（预留）

项目中已经包含：

- `ImageSlicer.spec`：用于配置 PyInstaller 打包规则；
- `build_exe.bat`：在 Windows 环境下一键打包的脚本。

在功能相对稳定后，可以基于这两个文件：

- 将程序打包为独立的 Windows 可执行文件（`.exe`），方便在生产线电脑上部署；
- 根据需要进一步加入图标、版本信息、多语言等配置。