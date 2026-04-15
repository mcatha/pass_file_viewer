"""
Pass File Viewer — entry point.

Usage:
    python main.py                     # launch empty, open via File menu
    python main.py path/to/file.pass   # open a .pass file immediately
"""

import sys
import os
from pathlib import Path

def _check_opengl() -> None:
    """Switch to Qt's software OpenGL if the system has no usable GPU driver."""
    if os.environ.get("QT_OPENGL"):
        return  # user already chose
    try:
        import ctypes
        gl = ctypes.windll.opengl32
        # If we can't even load a basic GL function, driver is broken
        if not gl.wglGetProcAddress:
            raise OSError
    except Exception:
        os.environ["QT_OPENGL"] = "software"

_check_opengl()

# vispy must use PyQt6 backend before any other vispy import
from vispy import app as vispy_app
vispy_app.use_app("pyqt6")

from PyQt6.QtWidgets import QApplication
from main_window import MainWindow


def main() -> None:
    qapp = QApplication(sys.argv)
    qapp.setApplicationName("Pass File Viewer")

    # Accept an optional .pass file path on the command line
    initial_file = None
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.is_file():
            initial_file = str(p)

    window = MainWindow(initial_file=initial_file)
    window.show()
    sys.exit(qapp.exec())


if __name__ == "__main__":
    main()
