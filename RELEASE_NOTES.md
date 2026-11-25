## v1.8.0 — 导出标注图片与缺陷裁剪图

- 点击窗口顶部的 **导出图片或缺陷裁剪图…**，可导出当前图片、全部图片、指定缺陷或全部缺陷。指定缺陷支持勾选当前图上的框，也支持按类别筛选整个数据集。
- 每次导出可选择只绘制 1 号 XML（GT）、只绘制 2 号 XML（预测），或同时绘制两者。整图保留原始分辨率，裁剪图按框截取，均保存为 PNG。当前图会使用已确认但尚未写入 XML 的 GT 编辑。
- 针对 Mac 上点击缺陷种类时的闪退，类别和别名改用独立文本输入框。可以添加新类别、勾选并移除已有类别，也可以保存空类别表。

| 电脑 | 下载文件 |
| --- | --- |
| Windows x64 | `AnomaSlice.exe` |
| Mac · Apple 芯片（M 系列） | `AnomaSlice-macOS-arm64.zip` |
| Mac · Intel 处理器 | `AnomaSlice-macOS-x86_64.zip` |

Windows 下载后双击 EXE。Mac 需要 macOS 14 或更新版本，解压 ZIP 后将 `AnomaSlice.app` 拖入「应用程序」。两个平台均无需安装 Python。

Mac 版使用 ⌘S 保存、⌘Z 撤销，支持用笔记本的 Delete 键删除选中的框。若首次打开时系统提示无法验证开发者，本版尚未经 Apple 公证，请参阅 [Apple 的打开说明](https://support.apple.com/zh-cn/102445)。
