# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Pass File Viewer."""

import sys
from pathlib import Path
import vispy
import PyQt6

block_cipher = None
src_dir = Path(SPECPATH)
vispy_dir = Path(vispy.__file__).parent
pyqt6_dir = Path(PyQt6.__file__).parent

a = Analysis(
    [str(src_dir / 'main.py')],
    pathex=[str(src_dir)],
    binaries=[],
    datas=[
        # vispy GLSL shaders and data files
        (str(vispy_dir / 'glsl'), 'vispy/glsl'),
        (str(vispy_dir / 'io' / '_data'), 'vispy/io/_data'),
        # bundled mp3
        (str(src_dir / 'high_skies-the_shape_of_things_to_come.mp3'), '.'),
        # Qt software OpenGL renderer (Mesa llvmpipe) for no-GPU machines
        (str(pyqt6_dir / 'Qt6' / 'bin' / 'opengl32sw.dll'), '.'),
    ],
    hiddenimports=[
        'vispy.app.backends._pyqt6',
        'vispy.io',
        'vispy.glsl',
        'vispy.gloo.gl.glplus',
        'vispy.gloo.gl.gl2',
        'vispy.gloo.gl.desktop',
        'PyQt6.sip',
        'PyQt6.QtMultimedia',
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
    a.binaries,
    a.zipfiles,
    a.datas,
    name='PassFileViewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,        # windowed app, no console
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=str(src_dir / 'app_icon.ico'),
    version=str(src_dir / 'version_info.txt'),
)
