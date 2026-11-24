# AnomaSlice — Windows 图片标注与切图

原生 Qt 桌面应用，支持人工框选标注与切图分类。所有数据保存在本机。
运行时只依赖 PySide6 Essentials；图片读写、裁剪、绘制均使用 Qt，不使用 OpenCV、NumPy 或模型。

## 下载与运行

[下载最新 AnomaSlice.exe](https://github.com/get1024/AnomaSlice/releases/latest/download/AnomaSlice.exe)
 · [版本说明](https://github.com/get1024/AnomaSlice/releases/latest)

下载后双击运行，也可以将这个 EXE 直接拷给其他 Windows x64 电脑使用。
无需安装 Python、Conda 或额外运行环境，不需要随附 `_internal` 文件夹。

## 框选标注（默认模式）

1. 点击“打开图片或 XML”选择单张图片或已有的缺陷 XML。点击“打开文件夹”可递归浏览整批图片，包括 BMP、JPEG、PNG、TIFF、GIF、WebP 和打包的 Qt 解码器支持的其他格式。
2. 已有 XML 中的缺陷框直接显示在原图上。用图片下拉列表或“上一张 / 下一张”切换图片。**滚轮**缩放，**右键拖动**平移。
3. **左键拖动**新增矩形框，输入标签或别名后按 **Enter** 确认。**Esc** 取消正在输入的框。
4. **双击**框修改标签，**单击**框后按 **Delete** 删除。框重叠时可通过左侧编号列表选中指定框。
5. **Ctrl+Z** 撤回，**Ctrl+Shift+Z** 重做。点“保存 XML”或按 **Ctrl+S** 写入当前图片的标注。切换图片、切换到切图模式或关闭窗口时，未保存的修改会提示处理。

“缺陷种类与别名”中每行填写一个类别，多个别名用逗号分隔。
例如类别“划痕”的别名为 `1, scratch`，输入 `1` 保存为“划痕”，
输入 `待定1` 保持原样。只对完整文本匹配，区分大小写。
可以直接输入新类别；重复或冲突的别名会提示修正。
修改类别表影响后续输入，已有框的标签文字保持原样。

### 同图对照 GT 和模型预测

在“框选标注”中点击 **打开双 XML 对照…**，选择一张图片，再分别设置 **GT（可编辑）** 和 **模型预测（只读）**。两个来源都支持选择 XML 文件或文件夹。

| 文件布局 | GT 来源 | 模型预测来源 |
| --- | --- | --- |
| GT 在图片旁或图片目录的 `xml/` 中 | 留空，自动查找 | 选择预测文件或文件夹 |
| 预测在图片旁或图片目录的 `xml/` 中 | 选择 GT 文件或文件夹 | 留空，自动查找 |
| 两份 XML 均独立存放 | 选择 GT 文件或文件夹 | 选择预测文件或文件夹 |

GT 框使用实线，预测框使用青色虚线并标有“预测”。左侧列表、双击改标签、删除、撤销和 **Ctrl+S** 只作用于 GT。用“上一张 / 下一张”浏览同一目录的图片时，程序按图片主名在各自的来源中匹配 XML。手动选定的文件可以与起始图片异名，后续图片会在该文件所在目录按同名匹配。

某张图片没有预测时仍可编辑 GT。缺少 GT 时，保存会在选定的 GT 目录中新建同名 XML。使用自动查找时，默认写入图片目录的 `xml/` 子目录。工程另存为会记住两个来源。

### XML 的查找和保存

```text
部件/
  样本01.bmp
  xml/
    样本01.xml
```

应用识别图片旁和其 `xml` 子目录里的同名标注。如果图片与 XML 分目录存放，打开共同上级文件夹，也可按唯一同名关系配对。匹配有歧义时，界面会显示原因。

打开已有 XML 后，保存会原地更新该文件。首次标注的图片会在保存时生成 `xml/<图片主名>.xml`。已有的旧版 `.annotations.json` 或工程内嵌框可以在缺少 XML 时读入，下一次保存即得到 XML。

XML 沿用现场的 `DefectDetection` 格式。框使用原图像素坐标 `x, y, width, height`。保存更新 `Count` 和剩余框的 `index`，保留 `RawCount`、检测置信度及扩展字段。新建 XML 的 `RawCount` 为 0。人工新框带有 `source="manual"`，置信度留空。保存失败时可以在当前图片上重试。

## 切图分类

顶部切换到“切图分类”，设置 Patch 宽高、步长及分类阈值。

1. 逐个查看网格单元，点击“有缺陷 / 无缺陷”完成人工标注。
2. 完成后点击“切分”，或按 Enter 切分并进入下一张。
3. 程序按设置尺寸裁切小图，被标为缺陷的面积比例超过阈值时保存到 `Abnormal`，否则保存到 `Normal`。
4. 右侧查看结果，可右键将小图移到另一类别。

人工检查单元由步长决定，导出小图由 Patch 宽高决定，导出小图可以重叠。
框选标签与切图掩膜独立保存。支持中文文件路径和原有 1.0 / 1.1 工程文件。

切图模式下，`Ctrl+S` 保存工程，`Ctrl+O` 打开工程；`Ctrl+I` 打开图片。

## 代码结构

| 模块 | 职责 |
| --- | --- |
| `main.py` | 应用入口 |
| `app_window.py` | 界面和工作流程协调 |
| `image_canvas.py` | Qt 图片显示、网格和切图掩膜 |
| `bbox_canvas.py` | 框选、平移、缩放和标签输入 |
| `xml_annotations.py` | 递归配对、缺陷 XML 读写及并发修改检查 |
| `annotation_store.py` | 类别与别名校验、旧版 JSON 标注兼容 |
| `project_store.py` | 工程读写和兼容性校验 |
| `slicer.py` | Qt 原生裁切、覆盖比例计算和导出 |
| `label_dialog.py` | 类别与别名编辑界面 |

## 从源码运行与构建

使用 Windows x64、Python 3.12：

```powershell
python -m pip install -r requirements-runtime.txt
python main.py
```

构建单文件 Windows EXE：

```powershell
.\build-windows.ps1 -Python C:\path\to\python.exe -InstallDependencies
```

输出为 `dist/AnomaSlice.exe`。构建时排除未使用的 Qt Web、QML、多媒体和视频模块；
单文件封装在启动时把必要的 Qt 运行库释放到系统临时目录，退出后清理。

运行检查：

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
python -m unittest discover -s tests -v
python main.py --smoke-test
```

发布流程先执行测试，再验证打包后 EXE 在隔离 PATH 下独立启动，成功后发布 Release。
