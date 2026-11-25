# -*- mode: python ; coding: utf-8 -*-
"""Single-file Windows build using Qt's native image and widget support."""

from pathlib import Path


project_dir = Path(SPECPATH)

analysis = Analysis(
    [str(project_dir / "main.py")],
    pathex=[str(project_dir)],
    binaries=[],
    datas=[(str(project_dir / "VERSION"), ".")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.QtQuickWidgets",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        "PySide6.QtPdf",
        "PySide6.QtPdfWidgets",
        "PySide6.QtNetwork",
        "PySide6.QtOpenGL",
        "PySide6.QtOpenGLWidgets",
        "PySide6.QtPrintSupport",
        "PySide6.QtSql",
        "PySide6.QtSvg",
        "PySide6.QtSvgWidgets",
        "PySide6.QtTest",
        "PySide6.QtXml",
        "cv2",
        "numpy",
        "PIL",
        "matplotlib",
        "IPython",
        "tkinter",
        "pytest",
    ],
    noarchive=False,
    optimize=1,
)

# Qt 6.12 on Windows uses the OS-provided ICU (System32/icuuc.dll). An activated
# conda base environment can cause dependency analysis to collect its unrelated
# ICU instead; that DLL has version-suffixed exports and breaks QtCore import.
# Leave ICU to Windows, as the original, unfrozen PySide6 process does.
# QFileDialog and QGraphicsView use native/raster widgets. They need neither
# OpenGL's software fallback nor network, PDF, TLS or SQL plugins. Keep Qt's
# installed image codecs so folder discovery matches what the EXE can decode.
def keep_binary(entry):
    destination = Path(entry[0])
    filename = destination.name.lower()
    parts = {part.lower() for part in destination.parts}
    if filename.startswith(("opencv_videoio_ffmpeg", "icudt")) or filename == "icuuc.dll":
        return False
    if filename == "opengl32sw.dll":
        return False
    if "plugins" in parts:
        if "platforms" in parts:
            return filename in {"qwindows.dll", "qoffscreen.dll"}
        if "imageformats" in parts:
            return filename != "qpdf.dll"
        return False
    return True


analysis.binaries = [entry for entry in analysis.binaries if keep_binary(entry)]
analysis.datas = [
    entry for entry in analysis.datas
    if "translations" not in {part.lower() for part in Path(entry[0]).parts}
]

pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    name="AnomaSlice",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
