# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Pass File Viewer."""

import sys
from pathlib import Path
import vispy

block_cipher = None
src_dir = Path(SPECPATH)
vispy_dir = Path(vispy.__file__).parent

a = Analysis(
    [str(src_dir / 'main.py')],
    pathex=[str(src_dir)],
    binaries=[],
    datas=[
        # vispy GLSL shaders and data files
        (str(vispy_dir / 'glsl'), 'vispy/glsl'),
        (str(vispy_dir / 'io' / '_data'), 'vispy/io/_data'),
    ],
    hiddenimports=[
        'vispy.app.backends._pyqt6',
        'vispy.io',
        'vispy.glsl',
        'PyQt6.sip',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PassFileViewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,        # windowed app, no console
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=str(src_dir / 'app_icon.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PassFileViewer',
)
