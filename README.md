# AnomaSlice — Windows 图片标注与切图

原生 Qt 桌面应用，支持人工框选标注与切图分类。所有数据保存在本机。
运行时只依赖 PySide6 Essentials；图片读写、裁剪、绘制均使用 Qt，不使用 OpenCV、NumPy 或模型。

## 下载与运行

[下载最新 AnomaSlice.exe](https://github.com/get1024/AnomaSlice/releases/latest/download/AnomaSlice.exe)
 · [版本说明](https://github.com/get1024/AnomaSlice/releases/latest)

下载后双击运行，也可以将这个 EXE 直接拷给其他 Windows x64 电脑使用。
无需安装 Python、Conda 或额外运行环境，不需要随附 `_internal` 文件夹。

## 框选标注（默认模式）

1. 点击“打开图片”，选择文件；之后用“上一张 / 下一张”浏览同文件夹图片。
2. **滚轮**缩放，**右键按住拖动**平移，点击“适应窗口”恢复完整视图。
3. **左键按住拖动**画矩形框，松开后输入框出现在框的左上方。
4. 输入标签或别名，按 **Enter** 确认；**Esc** 取消当前未确认的框。
5. **单击**已有框选中，**Delete** 删除；**双击**已有框修改标签。
6. **Ctrl+Z** 撤回，**Ctrl+Shift+Z** 重做；输入框有焦点时撤回文字编辑。

“缺陷种类与别名”中每行填写一个类别，多个别名用逗号分隔。
例如类别“划痕”的别名为 `1, scratch`，输入 `1` 保存为“划痕”，
输入 `待定1` 保持原样。只对完整文本匹配，区分大小写。
可以直接输入新类别；重复或冲突的别名会提示修正。
修改类别表影响后续输入，已有框的标签文字保持原样。

每次确认、删除、改标签或撤回后自动保存；`Ctrl+S` 可重试保存。
标注文件位于图片旁，名称为 `<原图片完整文件名>.annotations.json`，不修改图片。
保存使用**原图像素坐标** `[x, y, width, height]`，右边界和下边界不包含在框内；
缩放和平移不改变坐标。

```json
{
  "version": 1,
  "image": "sample.png",
  "width": 1920,
  "height": 1080,
  "boxes": [{"label": "划痕", "bbox": [100, 80, 60, 40]}],
  "label_catalog": [{"name": "划痕", "aliases": ["1"]}]
}
```

切换图片会取消尚未确认的框，已确认的框会自动恢复。
保存失败时保留当前图片和编辑内容；标注文件损坏或尺寸不匹配时保留原文件并停止编辑该图。
当前需要调整框尺寸时，可撤回或删除后重新画框。

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
| `annotation_store.py` | 类别/别名校验与原子 JSON 存储 |
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
