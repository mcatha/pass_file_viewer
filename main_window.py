"""
Main application window — menus, status bar, file loading.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence, QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QStatusBar,
    QLabel,
    QApplication,
    QDockWidget,
    QColorDialog,
    QWidgetAction,
    QWidget,
    QHBoxLayout,
    QSlider,
)

from pass_parser import parse_pass_file, PassData
from viewer_widget import ShotViewerWidget
from selection_pane import SelectionPane


class _ParseWorker(QObject):
    """Parses a .pass file on a background thread."""
    finished = pyqtSignal(object, object)  # (PassData, path) or (None, error_str)

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path

    def run(self) -> None:
        try:
            data = parse_pass_file(self._path)
            self.finished.emit(data, self._path)
        except Exception as exc:
            self.finished.emit(None, str(exc))


class MainWindow(QMainWindow):
    """Top-level window for the Pass File Viewer."""

    def __init__(self, initial_file: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Pass File Viewer")
        self.resize(1280, 900)

        # ── central widget ──────────────────────────────────────────
        self._viewer = ShotViewerWidget(self)
        self.setCentralWidget(self._viewer)
        # ── selection pane (right dock) ─────────────────────────────
        self._selection_pane = SelectionPane()
        self._selection_dock = QDockWidget("Selection", self)
        self._selection_dock.setWidget(self._selection_pane)
        self._selection_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._selection_dock)
        self._selection_dock.setVisible(False)  # hidden until first box selection

        # Connect box-selection signal
        self._viewer.box_selected.connect(self._on_box_selection)
        # Connect KD-tree ready signal
        self._viewer.kdtree_ready.connect(self._on_kdtree_ready)
        # Background parse thread state
        self._parse_thread: QThread | None = None
        self._parse_worker: _ParseWorker | None = None
        # ── status bar ──────────────────────────────────────────────
        self._status_label = QLabel("  No file loaded")
        self._status_label.setStyleSheet("color: #aaa;")
        status = QStatusBar(self)
        status.addWidget(self._status_label, 1)
        self.setStatusBar(status)

        # ── menus ───────────────────────────────────────────────────
        self._build_menus()

        # ── load initial file if provided ───────────────────────────
        if initial_file:
            self._open_file(Path(initial_file))

    # ── menu construction ───────────────────────────────────────────

    def _build_menus(self) -> None:
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        open_act = QAction("&Open Pass File…", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._on_open)
        file_menu.addAction(open_act)

        file_menu.addSeparator()

        exit_act = QAction("E&xit", self)
        exit_act.setShortcut(QKeySequence("Alt+F4"))
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        # View menu
        view_menu = menu_bar.addMenu("&View")

        self._lines_act = QAction("Show Shot &Connections", self)
        self._lines_act.setCheckable(True)
        self._lines_act.setChecked(False)
        self._lines_act.setShortcut(QKeySequence("Ctrl+L"))
        self._lines_act.toggled.connect(self._on_toggle_lines)
        view_menu.addAction(self._lines_act)

        reset_act = QAction("&Reset View", self)
        reset_act.setShortcut(QKeySequence("Ctrl+R"))
        reset_act.triggered.connect(self._viewer.reset_view)
        view_menu.addAction(reset_act)

        view_menu.addSeparator()

        self._selection_pane_act = QAction("Show &Selection Pane", self)
        self._selection_pane_act.setCheckable(True)
        self._selection_pane_act.setChecked(False)
        self._selection_pane_act.setShortcut(QKeySequence("Ctrl+S"))
        self._selection_pane_act.toggled.connect(self._selection_dock.setVisible)
        view_menu.addAction(self._selection_pane_act)
        # Keep menu in sync if user closes dock via title-bar X
        self._selection_dock.visibilityChanged.connect(self._selection_pane_act.setChecked)

        view_menu.addSeparator()

        # ── FWHM scale slider ──────────────────────────────────────
        # Logarithmic slider: value ∈ [-200, 200] → scale = 10^(v/100)
        # FWHM nm/µs = 10 * scale  → range [0.1, 1000] nm/µs
        fwhm_container = QWidget()
        fwhm_layout = QHBoxLayout(fwhm_container)
        fwhm_layout.setContentsMargins(8, 2, 8, 2)
        fwhm_label = QLabel("FWHM:")
        fwhm_label.setStyleSheet("color: #ccc;")
        self._fwhm_value_label = QLabel("60.00 nm/\u00b5s")
        self._fwhm_value_label.setFixedWidth(100)
        self._fwhm_value_label.setStyleSheet("color: #ccc;")
        self._fwhm_slider = QSlider(Qt.Orientation.Horizontal)
        self._fwhm_slider.setMinimum(-200)   # 10^(-2) = 0.01× → 0.1 nm/µs
        self._fwhm_slider.setMaximum(200)    # 10^(+2) = 100×  → 1000 nm/µs
        self._fwhm_slider.setValue(78)       # 10^0.78 ≈ 6.0×  → 60 nm/µs
        self._fwhm_slider.setFixedWidth(160)
        self._fwhm_slider.valueChanged.connect(self._on_fwhm_slider)
        fwhm_layout.addWidget(fwhm_label)
        fwhm_layout.addWidget(self._fwhm_slider)
        fwhm_layout.addWidget(self._fwhm_value_label)
        fwhm_action = QWidgetAction(self)
        fwhm_action.setDefaultWidget(fwhm_container)
        view_menu.addAction(fwhm_action)

        view_menu.addSeparator()

        # ── Marker mode toggle ──────────────────────────────────────
        self._disc_mode_act = QAction("&Disc Markers (hard edge + overlay)", self)
        self._disc_mode_act.setCheckable(True)
        self._disc_mode_act.setChecked(False)
        self._disc_mode_act.toggled.connect(self._on_marker_mode_toggle)
        view_menu.addAction(self._disc_mode_act)

        view_menu.addSeparator()

        # ── Colors submenu ──────────────────────────────────────────
        colors_menu = view_menu.addMenu("C&olors")

        self._color_categories: list[dict] = []

        shot_presets = [
            ("Bright Blue",    0.30, 0.60, 1.00, 1.00),
            ("Cyan",           0.20, 0.80, 1.00, 1.00),
            ("Green",          0.25, 0.90, 0.40, 1.00),
            ("Magenta",        0.90, 0.30, 1.00, 1.00),
            ("Orange",         1.00, 0.60, 0.20, 1.00),
            ("Red",            1.00, 0.30, 0.25, 1.00),
            ("Gold",           1.00, 0.80, 0.20, 1.00),
            ("White",          0.85, 0.85, 0.85, 1.00),
        ]
        highlight_presets = [
            ("Gold",           1.00, 0.90, 0.00, 1.00),
            ("White",          1.00, 1.00, 1.00, 1.00),
            ("Magenta",        1.00, 0.35, 0.90, 1.00),
            ("Cyan",           0.20, 1.00, 1.00, 1.00),
            ("Red",            1.00, 0.35, 0.30, 1.00),
            ("Orange",         1.00, 0.65, 0.20, 1.00),
        ]
        box_presets = [
            ("Green",          0.30, 1.00, 0.50, 1.00),
            ("Cyan",           0.20, 1.00, 1.00, 1.00),
            ("White",          1.00, 1.00, 1.00, 1.00),
            ("Magenta",        1.00, 0.35, 0.90, 1.00),
            ("Yellow",         1.00, 1.00, 0.30, 1.00),
            ("Orange",         1.00, 0.65, 0.20, 1.00),
        ]
        line_presets = [
            ("Red",            1.00, 0.40, 0.25, 0.80),
            ("Orange",         1.00, 0.65, 0.20, 0.80),
            ("Yellow",         1.00, 0.95, 0.30, 0.80),
            ("Green",          0.40, 0.90, 0.40, 0.80),
            ("Cyan",           0.30, 0.90, 1.00, 0.80),
            ("Gray",           0.75, 0.75, 0.75, 0.70),
        ]
        sel_line_presets = [
            ("Gold",           1.00, 0.90, 0.00, 1.00),
            ("White",          1.00, 1.00, 1.00, 1.00),
            ("Cyan",           0.20, 1.00, 1.00, 1.00),
            ("Magenta",        1.00, 0.35, 0.90, 1.00),
            ("Green",          0.30, 1.00, 0.50, 1.00),
            ("Orange",         1.00, 0.65, 0.20, 1.00),
        ]

        categories = [
            ("&Shot Color",       "shot_color",         "set_shot_color",         shot_presets),
            ("Click &Highlight",  "selected_color",     "set_selected_color",     highlight_presets),
            ("&Box Highlight",    "box_selected_color", "set_box_selected_color", box_presets),
            ("Connection &Lines", "line_color",         "set_line_color",         line_presets),
            ("Line H&ighlight",   "sel_line_color",     "set_sel_line_color",     sel_line_presets),
        ]

        for menu_label, getter_prop, setter_name, presets in categories:
            sub = colors_menu.addMenu(menu_label)
            group = QActionGroup(self)
            group.setExclusive(True)
            current = getattr(self._viewer, getter_prop)
            cr, cg, cb, ca = current

            for label, r, g, b, a in presets:
                act = QAction(label, self)
                act.setCheckable(True)
                pm = QPixmap(16, 16)
                pm.fill(QColor.fromRgbF(min(r, 1.0), min(g, 1.0), min(b, 1.0)))
                act.setIcon(QIcon(pm))
                act.setData({"rgba": (r, g, b, a), "setter": setter_name,
                             "group_idx": len(self._color_categories)})
                act.triggered.connect(self._on_color_preset)
                group.addAction(act)
                sub.addAction(act)
                if abs(r - cr) < 0.02 and abs(g - cg) < 0.02 and abs(b - cb) < 0.02:
                    act.setChecked(True)

            sub.addSeparator()
            custom_act = QAction("Custom\u2026", self)
            custom_act.setData({"getter": getter_prop, "setter": setter_name,
                                "group_idx": len(self._color_categories)})
            custom_act.triggered.connect(self._on_custom_color)
            sub.addAction(custom_act)

            self._color_categories.append({"group": group})

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        about_act = QAction("&About", self)
        about_act.triggered.connect(self._on_about)
        help_menu.addAction(about_act)

        controls_act = QAction("&Controls", self)
        controls_act.triggered.connect(self._on_controls)
        help_menu.addAction(controls_act)

    # ── slots ───────────────────────────────────────────────────────

    def _on_fwhm_slider(self, value: int) -> None:
        """Logarithmic FWHM slider: value ∈ [-100, 100] → scale ∈ [0.1, 10]."""
        import math
        scale = math.pow(10, value / 100.0)
        # FWHM in nm/µs: base rate is 10 nm/µs
        fwhm_nm_per_us = 0.01 * scale * 1000.0  # _NM_PER_NS_DWELL * scale * 1e3
        self._fwhm_value_label.setText(f"{fwhm_nm_per_us:.2f} nm/\u00b5s")
        self._viewer.set_fwhm_scale(scale)

    def _on_marker_mode_toggle(self, checked: bool) -> None:
        """Switch between Gaussian PSF and hard-disc marker rendering."""
        self._viewer.set_marker_mode('disc' if checked else 'gaussian')

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Pass File",
            "",
            "Pass Files (*.pass);;All Files (*)",
        )
        if path:
            self._open_file(Path(path))

    def _on_toggle_lines(self, checked: bool) -> None:
        self._viewer.set_lines_visible(checked)

    def _on_color_preset(self) -> None:
        act = self.sender()
        if act is not None:
            info = act.data()
            rgba = info["rgba"]
            setter = getattr(self._viewer, info["setter"])
            setter(rgba)

    def _on_custom_color(self) -> None:
        act = self.sender()
        if act is None:
            return
        info = act.data()
        getter_prop = info["getter"]
        setter_name = info["setter"]
        group_idx = info["group_idx"]
        cr, cg, cb, ca = getattr(self._viewer, getter_prop)
        initial = QColor.fromRgbF(cr, cg, cb, ca)
        color = QColorDialog.getColor(
            initial, self, getter_prop.replace("_", " ").title(),
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if color.isValid():
            setter = getattr(self._viewer, setter_name)
            setter((color.redF(), color.greenF(), color.blueF(), color.alphaF()))
            # Uncheck any preset since this is a custom colour
            group = self._color_categories[group_idx]["group"]
            checked = group.checkedAction()
            if checked is not None:
                checked.setChecked(False)

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About Pass File Viewer",
            "<h3>Pass File Viewer</h3>"
            "<p>GPU-accelerated visualiser for binary .pass shot files.</p>"
            "<p>Uses vispy (OpenGL) for rendering millions of shots.</p>",
        )

    def _on_controls(self) -> None:
        QMessageBox.information(
            self,
            "Mouse Controls",
            "<b>Scroll wheel</b> — Zoom in / out<br>"
            "<b>Right-drag</b> — Pan<br>"
            "<b>Left-drag</b> — Box selection<br>"
            "<b>Left-click</b> — Select / deselect single shot<br>"
            "<b>Shift + Left-drag</b> — Rotate (2D)<br>"
            "<b>Hover</b> — Show shot info tooltip<br><br>"
            "<b>Ctrl+L</b> — Toggle shot connection lines<br>"
            "<b>Ctrl+R</b> — Reset view<br>"
            "<b>Ctrl+S</b> — Toggle selection pane",
        )

    # ── file loading ────────────────────────────────────────────────

    def _open_file(self, path: Path) -> None:
        # If a previous parse is still running, wait for it
        if self._parse_thread is not None:
            self._parse_thread.quit()
            self._parse_thread.wait()

        self._status_label.setText(f"  Loading {path.name}…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

        thread = QThread(self)  # parent prevents premature GC
        worker = _ParseWorker(path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_parse_finished)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_parse_thread_done)
        # prevent GC
        self._parse_thread = thread
        self._parse_worker = worker
        self._pending_path = path
        thread.start()

    def _on_parse_thread_done(self) -> None:
        """Called when the parse thread has fully stopped."""
        self._parse_thread = None
        self._parse_worker = None

    def _on_parse_finished(self, result, extra) -> None:
        """Called when background file parsing completes."""
        QApplication.restoreOverrideCursor()

        if result is None:
            # extra is the error string
            QMessageBox.critical(self, "Error", f"Failed to parse file:\n{extra}")
            self._status_label.setText("  Error loading file")
            return

        data: PassData = result
        path: Path = extra

        self._status_label.setText(f"  Rendering {data.count:,} shots…")
        QApplication.processEvents()

        self._viewer.load_data(data)
        self._selection_pane.set_data(data)

        self.setWindowTitle(f"Pass File Viewer — {path.name}")
        self._update_status(data, path)

    def _on_kdtree_ready(self) -> None:
        """Called when the KD-tree is built — update status to show it's fully ready."""
        txt = self._status_label.text()
        if "Building spatial index" in txt:
            txt = txt.replace("  |  Building spatial index…", "")
            self._status_label.setText(txt)

    def _update_status(self, data: PassData, path: Path) -> None:
        h = data.header
        size_mb = path.stat().st_size / (1024 * 1024)
        base = (
            f"  {path.name}  |  "
            f"Shots: {data.count:,}  |  "
            f"Stripe #{h.stripeNumber}  |  "
            f"Resolution: {h.resolution}  |  "
            f"BSS: {h.bss}  |  "
            f"Origin: ({h.stripeOriginX}, {h.stripeOriginY})  |  "
            f"File: {size_mb:.1f} MB"
        )
        # If KD-tree is still building, append a note
        if self._viewer._kdtree is None:
            base += "  |  Building spatial index…"
        self._status_label.setText(base)

    # ── box selection ───────────────────────────────────────────────

    def _on_box_selection(self, indices: list[int]) -> None:
        """Handle box-selection signal from the viewer."""
        self._selection_pane.update_selection(indices)
        if len(indices):
            self._selection_dock.setVisible(True)
