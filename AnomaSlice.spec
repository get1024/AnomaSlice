# -*- mode: python ; coding: utf-8 -*-
"""Portable Windows build: native Qt widgets, no web/QML/video runtime."""

from pathlib import Path


project_dir = Path(SPECPATH)

analysis = Analysis(
    [str(project_dir / "main.py")],
    pathex=[str(project_dir)],
    binaries=[],
    datas=[],
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
        "matplotlib",
        "IPython",
        "tkinter",
        "pytest",
    ],
    noarchive=False,
    optimize=1,
)

# OpenCV's wheel contains a large optional FFmpeg DLL. This application uses
# only still-image operations, so that video decoder is not distributed.
# Qt 6.12 on Windows uses the OS-provided ICU (System32/icuuc.dll). An activated
# conda base environment can cause dependency analysis to collect its unrelated
# ICU instead; that DLL has version-suffixed exports and breaks QtCore import.
# Leave ICU to Windows, as the original, unfrozen PySide6 process does.
analysis.binaries = [
    entry
    for entry in analysis.binaries
    if not Path(entry[0]).name.lower().startswith(("opencv_videoio_ffmpeg", "icudt"))
    and Path(entry[0]).name.lower() != "icuuc.dll"
]

pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="AnomaSlice",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="AnomaSlice",
)
