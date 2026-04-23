"""
Main application window — menus, status bar, file loading.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QObject, QPoint, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence, QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QMenu,
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

from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QUrl

import sys as _sys

import numpy as np

from pass_parser import parse_pass_file, PassData
from viewer_widget import ShotViewerWidget
from selection_pane import SelectionPane

# Support both normal execution and PyInstaller bundle
_BASE_DIR = Path(getattr(_sys, '_MEIPASS', Path(__file__).resolve().parent))
_MP3_PATH = str(_BASE_DIR / "high_skies-the_shape_of_things_to_come.mp3")


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


class _MultiParseWorker(QObject):
    """Parses multiple .pass files in parallel on a thread pool."""
    progress = pyqtSignal(int, int)  # (completed, total)
    finished = pyqtSignal(object, object)  # (list[(PassData, Path)], None) or (None, error_str)

    def __init__(self, paths: list[Path]) -> None:
        super().__init__()
        self._paths = paths

    def run(self) -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        n = len(self._paths)
        results: dict[Path, PassData] = {}
        with ThreadPoolExecutor() as pool:
            futures = {pool.submit(parse_pass_file, p): p for p in self._paths}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    data = future.result()
                    results[path] = data
                except Exception as exc:
                    self.finished.emit(None, f"{path.name}: {exc}")
                    return
                self.progress.emit(len(results), n)
        # Return results in the original file order
        ordered = [(results[p], p) for p in self._paths]
        self.finished.emit(ordered, None)


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
        self._selection_pane.content_ready.connect(self._fit_selection_dock)

        # Connect box-selection signal
        self._viewer.box_selected.connect(self._on_box_selection)
        # Connect KD-tree ready signal
        self._viewer.kdtree_ready.connect(self._on_kdtree_ready)
        # Connect stripe right-click signal
        self._viewer.stripe_right_clicked.connect(self._on_stripe_right_clicked)
        # Two-way click-select ↔ selection pane row highlight
        self._viewer.shot_clicked.connect(self._selection_pane.highlight_shot)
        self._selection_pane.shot_activated.connect(self._viewer.click_select_shot)
        # Background parse thread state
        self._parse_thread: QThread | None = None
        self._parse_worker: _ParseWorker | None = None
        self._loaded_files: list[tuple[PassData, Path]] = []
        self._file_selected: set[int] = set()  # file indices currently "file-selected"
        self._next_load_incremental: bool = False
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

        open_act = QAction("&Open Pass Files…", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._on_open)
        file_menu.addAction(open_act)

        incremental_open_act = QAction("&Incremental Open…", self)
        incremental_open_act.triggered.connect(self._on_incremental_open)
        file_menu.addAction(incremental_open_act)

        file_menu.addSeparator()

        exit_act = QAction("E&xit", self)
        exit_act.setShortcut(QKeySequence("Alt+F4"))
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        # View menu
        view_menu = menu_bar.addMenu("&View")

        reset_act = QAction("&Reset View", self)
        reset_act.setShortcut(QKeySequence("Ctrl+R"))
        reset_act.triggered.connect(self._viewer.reset_view)
        view_menu.addAction(reset_act)

        self._lines_act = QAction("Show Shot &Connections", self)
        self._lines_act.setCheckable(True)
        self._lines_act.setChecked(False)
        self._lines_act.setShortcut(QKeySequence("Ctrl+L"))
        self._lines_act.toggled.connect(self._on_toggle_lines)
        view_menu.addAction(self._lines_act)

        # ── Wafer Outline submenu ─────────────────────────────────
        _WAFER_SIZES: list[tuple[int | None, str]] = [
            (None,          "None"),
            (51_000_000,    '2" (51 mm)'),
            (100_000_000,   '4" (100 mm)'),
            (125_000_000,   '5" (125 mm)'),
            (150_000_000,   '6" (150 mm)'),
            (200_000_000,   '8" (200 mm)'),
            (300_000_000,   '12" (300 mm)'),
            (450_000_000,   '18" (450 mm)'),
        ]
        wafer_menu = view_menu.addMenu("&Wafer Outline")
        self._wafer_group = QActionGroup(self)
        self._wafer_group.setExclusive(True)
        self._wafer_actions: list[tuple[QAction, int | None]] = []
        for diameter, label in _WAFER_SIZES:
            act = QAction(label, self)
            act.setCheckable(True)
            if diameter is None:
                act.setChecked(True)
            self._wafer_group.addAction(act)
            wafer_menu.addAction(act)
            self._wafer_actions.append((act, diameter))
        self._wafer_group.triggered.connect(self._on_wafer_outline_select)

        # ── Column Positions submenu ──────────────────────────────
        col_pos_menu = view_menu.addMenu("Display &Column Positions")
        self._col_pos_group = QActionGroup(self)
        self._col_pos_group.setExclusionPolicy(
            QActionGroup.ExclusionPolicy.ExclusiveOptional
        )
        mb200_act = QAction("MB200 Array", self)
        mb200_act.setCheckable(True)
        self._col_pos_group.addAction(mb200_act)
        col_pos_menu.addAction(mb200_act)
        mb300_act = QAction("MB300 Array", self)
        mb300_act.setCheckable(True)
        self._col_pos_group.addAction(mb300_act)
        col_pos_menu.addAction(mb300_act)
        self._col_pos_group.triggered.connect(self._on_column_positions_select)

        view_menu.addSeparator()

        self._selection_pane_act = QAction("Show &Selection Pane", self)
        self._selection_pane_act.setCheckable(True)
        self._selection_pane_act.setChecked(False)
        self._selection_pane_act.setShortcut(QKeySequence("Ctrl+S"))
        self._selection_pane_act.toggled.connect(self._selection_dock.setVisible)
        view_menu.addAction(self._selection_pane_act)
        # Keep menu in sync if user closes dock via title-bar X
        self._selection_dock.visibilityChanged.connect(self._selection_pane_act.setChecked)

        select_all_act = QAction("Select &All Passes", self)
        select_all_act.triggered.connect(self._on_select_all_passes)
        view_menu.addAction(select_all_act)

        deselect_all_act = QAction("&Deselect All Passes", self)
        deselect_all_act.triggered.connect(self._on_deselect_all_passes)
        view_menu.addAction(deselect_all_act)

        view_menu.addSeparator()

        # ── FWHM scale slider ──────────────────────────────────────
        # Logarithmic slider: value ∈ [-200, 200] → scale = 10^(v/100)
        fwhm_container = QWidget()
        fwhm_layout = QHBoxLayout(fwhm_container)
        fwhm_layout.setContentsMargins(8, 2, 8, 2)
        fwhm_label = QLabel("FWHM:")
        fwhm_label.setStyleSheet("color: #ccc;")
        self._fwhm_value_label = QLabel("60.00 nm/\u00b5s")
        self._fwhm_value_label.setFixedWidth(100)
        self._fwhm_value_label.setStyleSheet("color: #ccc;")
        self._fwhm_slider = QSlider(Qt.Orientation.Horizontal)
        self._fwhm_slider.setMinimum(-200)
        self._fwhm_slider.setMaximum(200)
        self._fwhm_slider.setValue(78)        # 10^0.78 ≈ 6.0× → 60 nm/µs
        self._fwhm_slider.setFixedWidth(160)
        self._fwhm_slider.valueChanged.connect(self._on_fwhm_slider)
        fwhm_layout.addWidget(fwhm_label)
        fwhm_layout.addWidget(self._fwhm_slider)
        fwhm_layout.addWidget(self._fwhm_value_label)
        fwhm_action = QWidgetAction(self)
        fwhm_action.setDefaultWidget(fwhm_container)
        view_menu.addAction(fwhm_action)

        view_menu.addSeparator()

        # ── Marker mode select ───────────────────────────────────────
        mode_menu = view_menu.addMenu("Marker &Mode")
        self._mode_group = QActionGroup(self)
        self._mode_group.setExclusive(True)

        self._disc_mode_act = QAction("Disc", self)
        self._disc_mode_act.setCheckable(True)
        self._disc_mode_act.setChecked(True)
        self._mode_group.addAction(self._disc_mode_act)
        mode_menu.addAction(self._disc_mode_act)

        self._gauss_mode_act = QAction("Gaussian", self)
        self._gauss_mode_act.setCheckable(True)
        self._gauss_mode_act.setChecked(False)
        self._mode_group.addAction(self._gauss_mode_act)
        mode_menu.addAction(self._gauss_mode_act)

        self._mode_group.triggered.connect(self._on_marker_mode_select)

        view_menu.addSeparator()

        # ── Disc Alpha Controls submenu ───────────────────────────
        self._disc_alpha_menu = view_menu.addMenu("&Disc Alpha Controls")

        # Overlap white slider (log): base additive white alpha
        # value ∈ [-300, 0] → 10^(v/100) ∈ [0.001, 1.0]
        ow_container = QWidget()
        ow_layout = QHBoxLayout(ow_container)
        ow_layout.setContentsMargins(8, 2, 8, 2)
        ow_label = QLabel("Overlap:")
        ow_label.setStyleSheet("color: #ccc;")
        self._ow_value_label = QLabel("0.050")
        self._ow_value_label.setFixedWidth(60)
        self._ow_value_label.setStyleSheet("color: #ccc;")
        self._ow_slider = QSlider(Qt.Orientation.Horizontal)
        self._ow_slider.setMinimum(-300)
        self._ow_slider.setMaximum(0)
        self._ow_slider.setValue(-130)        # 10^-1.30 ≈ 0.05
        self._ow_slider.setFixedWidth(160)
        self._ow_slider.valueChanged.connect(self._on_ow_slider)
        ow_layout.addWidget(ow_label)
        ow_layout.addWidget(self._ow_slider)
        ow_layout.addWidget(self._ow_value_label)
        ow_action = QWidgetAction(self)
        ow_action.setDefaultWidget(ow_container)
        self._disc_alpha_menu.addAction(ow_action)

        # dpp_low slider (log): disc alpha = 1 below this DPP
        # value ∈ [-200, 600] → 10^(v/100) ∈ [0.01, 1000000]
        disc_lo_container = QWidget()
        disc_lo_layout = QHBoxLayout(disc_lo_container)
        disc_lo_layout.setContentsMargins(8, 2, 8, 2)
        disc_lo_label = QLabel("dpp lo:")
        disc_lo_label.setStyleSheet("color: #ccc;")
        self._disc_lo_value_label = QLabel("0.01")
        self._disc_lo_value_label.setFixedWidth(60)
        self._disc_lo_value_label.setStyleSheet("color: #ccc;")
        self._disc_lo_slider = QSlider(Qt.Orientation.Horizontal)
        self._disc_lo_slider.setMinimum(-200)
        self._disc_lo_slider.setMaximum(600)
        self._disc_lo_slider.setValue(-200)       # 10^-2.00 = 0.01
        self._disc_lo_slider.setFixedWidth(160)
        self._disc_lo_slider.valueChanged.connect(self._on_disc_lo_slider)
        disc_lo_layout.addWidget(disc_lo_label)
        disc_lo_layout.addWidget(self._disc_lo_slider)
        disc_lo_layout.addWidget(self._disc_lo_value_label)
        disc_lo_action = QWidgetAction(self)
        disc_lo_action.setDefaultWidget(disc_lo_container)
        self._disc_alpha_menu.addAction(disc_lo_action)

        # dpp_high slider (log): disc alpha = 0 above this DPP
        # value ∈ [-200, 600] → 10^(v/100) ∈ [0.01, 1000000]
        disc_hi_container = QWidget()
        disc_hi_layout = QHBoxLayout(disc_hi_container)
        disc_hi_layout.setContentsMargins(8, 2, 8, 2)
        disc_hi_label = QLabel("dpp hi:")
        disc_hi_label.setStyleSheet("color: #ccc;")
        self._disc_hi_value_label = QLabel("1e+10")
        self._disc_hi_value_label.setFixedWidth(60)
        self._disc_hi_value_label.setStyleSheet("color: #ccc;")
        self._disc_hi_slider = QSlider(Qt.Orientation.Horizontal)
        self._disc_hi_slider.setMinimum(-200)
        self._disc_hi_slider.setMaximum(1000)
        self._disc_hi_slider.setValue(1000)       # 10^10.00 = 1e10
        self._disc_hi_slider.setFixedWidth(160)
        self._disc_hi_slider.valueChanged.connect(self._on_disc_hi_slider)
        disc_hi_layout.addWidget(disc_hi_label)
        disc_hi_layout.addWidget(self._disc_hi_slider)
        disc_hi_layout.addWidget(self._disc_hi_value_label)
        disc_hi_action = QWidgetAction(self)
        disc_hi_action.setDefaultWidget(disc_hi_container)
        self._disc_alpha_menu.addAction(disc_hi_action)

        # dpp_mid slider (log): intermediate DPP point
        # value ∈ [-200, 1000] → 10^(v/100) ∈ [0.01, 10000000000]
        disc_mid_container = QWidget()
        disc_mid_layout = QHBoxLayout(disc_mid_container)
        disc_mid_layout.setContentsMargins(8, 2, 8, 2)
        disc_mid_label = QLabel("dpp mid:")
        disc_mid_label.setStyleSheet("color: #ccc;")
        self._disc_mid_value_label = QLabel("5000")
        self._disc_mid_value_label.setFixedWidth(60)
        self._disc_mid_value_label.setStyleSheet("color: #ccc;")
        self._disc_mid_slider = QSlider(Qt.Orientation.Horizontal)
        self._disc_mid_slider.setMinimum(-200)
        self._disc_mid_slider.setMaximum(1000)
        self._disc_mid_slider.setValue(370)       # 10^3.70 ≈ 5000
        self._disc_mid_slider.setFixedWidth(160)
        self._disc_mid_slider.valueChanged.connect(self._on_disc_mid_slider)
        disc_mid_layout.addWidget(disc_mid_label)
        disc_mid_layout.addWidget(self._disc_mid_slider)
        disc_mid_layout.addWidget(self._disc_mid_value_label)
        disc_mid_action = QWidgetAction(self)
        disc_mid_action.setDefaultWidget(disc_mid_container)
        self._disc_alpha_menu.addAction(disc_mid_action)

        # f_mid slider (linear): alpha value at dpp_mid
        # value ∈ [0, 100] → 0.00 .. 1.00
        fmid_container = QWidget()
        fmid_layout = QHBoxLayout(fmid_container)
        fmid_layout.setContentsMargins(8, 2, 8, 2)
        fmid_label = QLabel("α mid:")
        fmid_label.setStyleSheet("color: #ccc;")
        self._fmid_value_label = QLabel("0.20")
        self._fmid_value_label.setFixedWidth(60)
        self._fmid_value_label.setStyleSheet("color: #ccc;")
        self._fmid_slider = QSlider(Qt.Orientation.Horizontal)
        self._fmid_slider.setMinimum(0)
        self._fmid_slider.setMaximum(100)
        self._fmid_slider.setValue(20)            # 0.20
        self._fmid_slider.setFixedWidth(160)
        self._fmid_slider.valueChanged.connect(self._on_fmid_slider)
        fmid_layout.addWidget(fmid_label)
        fmid_layout.addWidget(self._fmid_slider)
        fmid_layout.addWidget(self._fmid_value_label)
        fmid_action = QWidgetAction(self)
        fmid_action.setDefaultWidget(fmid_container)
        self._disc_alpha_menu.addAction(fmid_action)

        # Inflate slider (linear): disc stride inflation amplitude
        # value ∈ [0, 2000] → 0.00 .. 20.00 in steps of 0.01
        infl_container = QWidget()
        infl_layout = QHBoxLayout(infl_container)
        infl_layout.setContentsMargins(8, 2, 8, 2)
        infl_label = QLabel("Inflate:")
        infl_label.setStyleSheet("color: #ccc;")
        self._infl_value_label = QLabel("0.50")
        self._infl_value_label.setFixedWidth(60)
        self._infl_value_label.setStyleSheet("color: #ccc;")
        self._infl_slider = QSlider(Qt.Orientation.Horizontal)
        self._infl_slider.setMinimum(0)
        self._infl_slider.setMaximum(2000)     # 0.00 .. 20.00
        self._infl_slider.setValue(50)          # 0.50
        self._infl_slider.setFixedWidth(160)
        self._infl_slider.valueChanged.connect(self._on_infl_slider)
        infl_layout.addWidget(infl_label)
        infl_layout.addWidget(self._infl_slider)
        infl_layout.addWidget(self._infl_value_label)
        infl_action = QWidgetAction(self)
        infl_action.setDefaultWidget(infl_container)
        self._disc_alpha_menu.addAction(infl_action)

        # Edge softness slider (linear): vispy antialias width in pixels
        aa_container = QWidget()
        aa_layout = QHBoxLayout(aa_container)
        aa_layout.setContentsMargins(8, 2, 8, 2)
        aa_label = QLabel("Edge:")
        aa_label.setStyleSheet("color: #ccc;")
        self._aa_value_label = QLabel("2.0")
        self._aa_value_label.setFixedWidth(60)
        self._aa_value_label.setStyleSheet("color: #ccc;")
        self._aa_slider = QSlider(Qt.Orientation.Horizontal)
        self._aa_slider.setMinimum(0)
        self._aa_slider.setMaximum(200)       # 0..20.0 in steps of 0.1
        self._aa_slider.setValue(20)          # 2.0
        self._aa_slider.setFixedWidth(160)
        self._aa_slider.valueChanged.connect(self._on_aa_slider)
        aa_layout.addWidget(aa_label)
        aa_layout.addWidget(self._aa_slider)
        aa_layout.addWidget(self._aa_value_label)
        aa_action = QWidgetAction(self)
        aa_action.setDefaultWidget(aa_container)
        self._disc_alpha_menu.addAction(aa_action)

        # ── Gaussian Alpha Controls submenu ─────────────────────────
        self._gauss_alpha_menu = view_menu.addMenu("&Gaussian Alpha Controls")
        self._gauss_alpha_menu.setEnabled(False)

        # d_ref slider (log): sigmoid midpoint
        dref_container = QWidget()
        dref_layout = QHBoxLayout(dref_container)
        dref_layout.setContentsMargins(8, 2, 8, 2)
        dref_label = QLabel("d_ref:")
        dref_label.setStyleSheet("color: #ccc;")
        self._dref_value_label = QLabel("16.5")
        self._dref_value_label.setFixedWidth(60)
        self._dref_value_label.setStyleSheet("color: #ccc;")
        self._dref_slider = QSlider(Qt.Orientation.Horizontal)
        self._dref_slider.setMinimum(-100)
        self._dref_slider.setMaximum(300)
        self._dref_slider.setValue(122)       # 10^1.22 ≈ 16.5
        self._dref_slider.setFixedWidth(160)
        self._dref_slider.valueChanged.connect(self._on_dref_slider)
        dref_layout.addWidget(dref_label)
        dref_layout.addWidget(self._dref_slider)
        dref_layout.addWidget(self._dref_value_label)
        dref_action = QWidgetAction(self)
        dref_action.setDefaultWidget(dref_container)
        self._gauss_alpha_menu.addAction(dref_action)

        # α max slider (log)
        amax_container = QWidget()
        amax_layout = QHBoxLayout(amax_container)
        amax_layout.setContentsMargins(8, 2, 8, 2)
        amax_label = QLabel("\u03b1 max:")
        amax_label.setStyleSheet("color: #ccc;")
        self._amax_value_label = QLabel("0.330")
        self._amax_value_label.setFixedWidth(60)
        self._amax_value_label.setStyleSheet("color: #ccc;")
        self._amax_slider = QSlider(Qt.Orientation.Horizontal)
        self._amax_slider.setMinimum(-300)
        self._amax_slider.setMaximum(0)
        self._amax_slider.setValue(-48)       # 10^-0.48 ≈ 0.33
        self._amax_slider.setFixedWidth(160)
        self._amax_slider.valueChanged.connect(self._on_amax_slider)
        amax_layout.addWidget(amax_label)
        amax_layout.addWidget(self._amax_slider)
        amax_layout.addWidget(self._amax_value_label)
        amax_action = QWidgetAction(self)
        amax_action.setDefaultWidget(amax_container)
        self._gauss_alpha_menu.addAction(amax_action)

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
            ("Green",          0.30, 1.00, 0.50, 0.50),
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

        # ── Volume menu (top-level) ─────────────────────────────────
        volume_menu = menu_bar.addMenu("V&olume")

        vol_widget = QWidget()
        vol_layout = QHBoxLayout(vol_widget)
        vol_layout.setContentsMargins(8, 2, 8, 2)
        vol_label = QLabel("Vol:")
        vol_layout.addWidget(vol_label)
        self._vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._vol_slider.setRange(0, 100)
        self._vol_slider.setValue(0)
        self._vol_slider.setFixedWidth(140)
        vol_layout.addWidget(self._vol_slider)
        self._vol_value_label = QLabel("0")
        self._vol_value_label.setFixedWidth(28)
        vol_layout.addWidget(self._vol_value_label)
        vol_action = QWidgetAction(self)
        vol_action.setDefaultWidget(vol_widget)
        volume_menu.addAction(vol_action)

        # Set up media player
        self._audio_output = QAudioOutput(self)
        self._audio_output.setVolume(0.0)
        self._media_player = QMediaPlayer(self)
        self._media_player.setAudioOutput(self._audio_output)
        self._media_player.setSource(QUrl.fromLocalFile(_MP3_PATH))
        self._media_player.setLoops(QMediaPlayer.Loops.Infinite)

        self._vol_slider.valueChanged.connect(self._on_vol_slider)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        about_act = QAction("&About", self)
        about_act.triggered.connect(self._on_about)
        help_menu.addAction(about_act)

        controls_act = QAction("&Controls", self)
        controls_act.triggered.connect(self._on_controls)
        help_menu.addAction(controls_act)

    # ── slots ───────────────────────────────────────────────────────

    def _on_vol_slider(self, value: int) -> None:
        """Volume slider 0–100.  0 = muted/stopped, >0 = play at that volume."""
        self._vol_value_label.setText(str(value))
        vol = value / 100.0
        self._audio_output.setVolume(vol)
        if value > 0:
            if self._media_player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                self._media_player.play()
        else:
            self._media_player.pause()

    def _on_fwhm_slider(self, value: int) -> None:
        """Logarithmic FWHM slider: value ∈ [-200, 200] → scale = 10^(v/100)."""
        import math
        scale = math.pow(10, value / 100.0)
        fwhm_nm_per_us = 0.01 * scale * 1000.0
        self._fwhm_value_label.setText(f"{fwhm_nm_per_us:.2f} nm/\u00b5s")
        self._viewer.set_fwhm_scale(scale)

    def _on_dref_slider(self, value: int) -> None:
        """d_ref: slider [-100, 300] → d_ref = 10^(v/100) ∈ [0.1, 1000]."""
        import math
        d_ref = math.pow(10, value / 100.0)
        self._dref_value_label.setText(f"{d_ref:.1f}")
        self._viewer.set_alpha_dref(d_ref)

    def _on_amax_slider(self, value: int) -> None:
        """α max: slider [-300, 0] → α_max = 10^(v/100) ∈ [0.001, 1.0]."""
        import math
        amax = math.pow(10, value / 100.0)
        self._amax_value_label.setText(f"{amax:.3f}")
        self._viewer.set_alpha_max(amax)

    def _on_ow_slider(self, value: int) -> None:
        """Disc overlap white: slider [-300, 0] → 10^(v/100) ∈ [0.001, 1.0]."""
        import math
        ow = math.pow(10, value / 100.0)
        self._ow_value_label.setText(f"{ow:.3f}")
        self._viewer.set_disc_overlap_white(ow)

    def _on_disc_lo_slider(self, value: int) -> None:
        """Disc dpp_low: slider [-200, 600] → 10^(v/100) ∈ [0.01, 1000000]."""
        import math
        v = math.pow(10, value / 100.0)
        self._disc_lo_value_label.setText(f"{v:.2g}")
        self._viewer.set_disc_dref(v)

    def _on_disc_hi_slider(self, value: int) -> None:
        """Disc dpp_high: slider [-200, 1000] → 10^(v/100)."""
        import math
        v = math.pow(10, value / 100.0)
        self._disc_hi_value_label.setText(f"{v:.2g}")
        self._viewer.set_disc_dpp_high(v)

    def _on_disc_mid_slider(self, value: int) -> None:
        """Disc dpp_mid: slider [-200, 1000] → 10^(v/100)."""
        import math
        v = math.pow(10, value / 100.0)
        self._disc_mid_value_label.setText(f"{v:.2g}")
        self._viewer.set_disc_dpp_mid(v)

    def _on_fmid_slider(self, value: int) -> None:
        """α mid: slider [0, 100] → 0.00 .. 1.00."""
        fmid = value / 100.0
        self._fmid_value_label.setText(f"{fmid:.2f}")
        self._viewer.set_disc_f_mid(fmid)

    def _on_infl_slider(self, value: int) -> None:
        """Disc inflate: slider [0, 2000] → 0.00 .. 20.00."""
        amp = value / 100.0
        self._infl_value_label.setText(f"{amp:.2f}")
        self._viewer.set_disc_inflate_amp(amp)

    def _on_aa_slider(self, value: int) -> None:
        """Disc edge softness: slider [0, 200] → 0.0 .. 20.0 px."""
        aa = value / 10.0
        self._aa_value_label.setText(f"{aa:.1f}")
        self._viewer.set_disc_antialias(aa)

    def _on_marker_mode_select(self, action: QAction) -> None:
        """Switch between Gaussian PSF and hard-disc marker rendering."""
        is_disc = (action is self._disc_mode_act)
        self._viewer.set_marker_mode('disc' if is_disc else 'gaussian')
        self._gauss_alpha_menu.setEnabled(not is_disc)
        self._disc_alpha_menu.setEnabled(is_disc)

    def _on_wafer_outline_select(self, action: QAction) -> None:
        """Show or hide the wafer outline circle."""
        for act, diameter in self._wafer_actions:
            if act is action:
                self._viewer.set_wafer_outline(diameter)
                return

    def _on_column_positions_select(self, action: QAction) -> None:
        if action.isChecked():
            array_type = 'MB200' if 'MB200' in action.text() else 'MB300'
            self._viewer.set_column_positions(array_type)
        else:
            self._viewer.set_column_positions(None)

    def _on_open(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Pass File(s)",
            "",
            "Pass Files (*.pass);;All Files (*)",
        )
        if not paths:
            return

        # Validate all files up front
        valid: list[Path] = []
        for p in paths:
            pass_path = Path(p)
            meta_path = pass_path.parent / (pass_path.name + ".meta")

            # Validate meta file if it exists (don't reject files without one —
            # the parser will check for an embedded header as fallback).
            if meta_path.is_file():
                try:
                    from pass_parser import parse_meta_file
                    parse_meta_file(meta_path)
                except Exception as exc:
                    QMessageBox.warning(
                        self,
                        "Invalid Meta File",
                        f"The meta file could not be read:\n{meta_path.name}\n\n{exc}",
                    )
                    continue

            valid.append(pass_path)

        if not valid:
            return

        # Single file: use the fast single-file path
        self._next_load_incremental = False
        if len(valid) == 1:
            self._open_file(valid[0])
            return

        # Multiple files: parse all at once, then render once
        self._open_files_batch(valid)

    def _on_incremental_open(self) -> None:
        """Open additional pass files and add them to the current view."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Pass File(s) to View",
            "",
            "Pass Files (*.pass);;All Files (*)",
        )
        if not paths:
            return

        valid: list[Path] = []
        for p in paths:
            pass_path = Path(p)
            meta_path = pass_path.parent / (pass_path.name + ".meta")
            if meta_path.is_file():
                try:
                    from pass_parser import parse_meta_file
                    parse_meta_file(meta_path)
                except Exception as exc:
                    QMessageBox.warning(
                        self,
                        "Invalid Meta File",
                        f"The meta file could not be read:\n{meta_path.name}\n\n{exc}",
                    )
                    continue
            valid.append(pass_path)

        if not valid:
            return

        self._next_load_incremental = True
        if len(valid) == 1:
            self._open_file(valid[0])
        else:
            self._open_files_batch(valid)

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
            "<p><b>Version 1.2</b> &mdash; March 2026</p>"
            "<p>GPU-accelerated visualiser for binary .pass shot files.<br>"
            "Uses vispy (OpenGL) for rendering millions of shots.</p>"
            "<p>Created by <b>Morgan Catha</b>, Senior Electrical Engineer,<br>"
            "for <b>Multibeam Corporation</b>.</p>"
            "<hr>"
            "<p style='font-size:small;'>Music: <i>The Shape of Things to Come</i> "
            "by <b>High Skies</b>, from the album <i>Sounds of Earth</i> (2010).<br>"
            "<a href='http://highskies.bandcamp.com'>highskies.bandcamp.com</a></p>",
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
            "<b>Shift + Left-click</b> — Place ruler start<br>"
            "<b>Left-click</b> (while ruler active) — Lock ruler end<br>"
            "<b>Right-click</b> — Clear ruler<br>"
            "<b>Hover</b> — Show shot info tooltip<br><br>"
            "<b>Ctrl+L</b> — Toggle shot connection lines<br>"
            "<b>Ctrl+R</b> — Reset view<br>"
            "<b>Ctrl+S</b> — Toggle selection pane",
        )

    # ── file loading ────────────────────────────────────────────────

    def _open_files_batch(self, paths: list[Path]) -> None:
        """Parse multiple files on a background thread, then render once."""
        if self._parse_thread is not None:
            self._parse_thread.quit()
            self._parse_thread.wait()

        n = len(paths)
        self._status_label.setText(f"  Loading {n} files…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

        thread = QThread(self)
        worker = _MultiParseWorker(paths)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(
            lambda cur, tot: self._status_label.setText(
                f"  Parsing file {cur}/{tot}…"
            )
        )
        worker.finished.connect(self._on_multi_parse_finished)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_parse_thread_done)
        self._parse_thread = thread
        self._parse_worker = worker
        thread.start()

    def _on_multi_parse_finished(self, results, error) -> None:
        """Called when batch parsing completes — merge all and render once."""
        QApplication.restoreOverrideCursor()

        if results is None:
            QMessageBox.critical(self, "Error", f"Failed to parse file:\n{error}")
            self._status_label.setText("  Error loading file")
            return

        # Apply stripe origin offsets
        for data, path in results:
            ox = np.float64(data.header.stripeOriginX)
            oy = np.float64(data.header.stripeOriginY)
            data.x = data.x.astype(np.float64) + ox
            data.y = data.y.astype(np.float64) + oy

        # Merge into loaded files list
        _incremental = self._next_load_incremental and bool(self._loaded_files)
        self._next_load_incremental = False
        if _incremental:
            self._loaded_files.extend(results)
        else:
            self._loaded_files = list(results)
            self._file_selected.clear()
        merged = self._merge_loaded_files()

        self._status_label.setText(f"  Rendering {merged.count:,} shots…")
        QApplication.processEvents()

        self._viewer.load_data(merged, keep_origin=_incremental)
        self._selection_pane.set_data(merged)
        _counts = [d.count for d, _ in self._loaded_files]
        self._selection_pane.set_file_boundaries(
            [str(p.name) for _, p in self._loaded_files],
            _counts,
        )
        _offsets: list[int] = []
        _running = 0
        for c in _counts:
            _offsets.append(_running)
            _running += c
        self._viewer.set_file_break_offsets(_offsets)

        # Stripe region metadata
        regions = []
        for d, p in self._loaded_files:
            h = d.header
            regions.append({
                "name": p.name,
                "shots": d.count,
                "originX": h.stripeOriginX,
                "originY": h.stripeOriginY,
                "width": h.stripeWidth,
                "length": h.stripeLength,
                "subFieldHeight": h.subFieldHeight,
                "overlap": h.overlap,
            })
        self._viewer.set_stripe_regions(regions)
        self._restore_pinned_stripes()

        n = len(self._loaded_files)
        self.setWindowTitle(f"Pass File Viewer — {n} files")
        self._update_status_multi(merged)

    def _open_file(self, path: Path) -> None:
        # If a previous parse is still running, wait for it
        if self._parse_thread is not None:
            self._parse_thread.quit()
            self._parse_thread.wait()

        self._status_label.setText(f"  Loading {path.name}…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

        thread = QThread(self)
        worker = _ParseWorker(path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_parse_finished)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_parse_thread_done)
        self._parse_thread = thread
        self._parse_worker = worker
        self._pending_path = path
        thread.start()

    def _on_parse_thread_done(self) -> None:
        """Called when the parse thread has fully stopped."""
        self._parse_thread = None
        self._parse_worker = None

    def closeEvent(self, event) -> None:
        self._media_player.stop()
        if self._parse_thread is not None:
            self._parse_thread.quit()
            self._parse_thread.wait()
        self._viewer.shutdown()
        self._selection_pane.shutdown()
        super().closeEvent(event)

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

        # Apply stripe origin offset → absolute wafer coordinates (float64)
        ox = np.float64(data.header.stripeOriginX)
        oy = np.float64(data.header.stripeOriginY)
        data.x = data.x.astype(np.float64) + ox
        data.y = data.y.astype(np.float64) + oy

        # Incremental merge or replace
        _incremental = self._next_load_incremental and bool(self._loaded_files)
        self._next_load_incremental = False
        if _incremental:
            self._loaded_files.append((data, path))
            merged = self._merge_loaded_files()
        else:
            self._loaded_files = [(data, path)]
            self._file_selected.clear()
            merged = data

        self._status_label.setText(f"  Rendering {merged.count:,} shots…")
        QApplication.processEvents()

        self._viewer.load_data(merged, keep_origin=_incremental)
        self._selection_pane.set_data(merged)
        _counts = [d.count for d, _ in self._loaded_files]
        self._selection_pane.set_file_boundaries(
            [str(p.name) for _, p in self._loaded_files],
            _counts,
        )
        _offsets: list[int] = []
        _running = 0
        for c in _counts:
            _offsets.append(_running)
            _running += c
        self._viewer.set_file_break_offsets(_offsets)

        # Pass stripe region metadata for hover rectangles
        regions = []
        for d, p in self._loaded_files:
            h = d.header
            regions.append({
                "name": p.name,
                "shots": d.count,
                "originX": h.stripeOriginX,
                "originY": h.stripeOriginY,
                "width": h.stripeWidth,
                "length": h.stripeLength,
                "subFieldHeight": h.subFieldHeight,
                "overlap": h.overlap,
            })
        self._viewer.set_stripe_regions(regions)
        self._restore_pinned_stripes()

        # Easter egg: dollar bill green for novus_ordo
        if path.stem.lower() == "novus_ordo":
            self._viewer.set_shot_color((0.33, 0.54, 0.18, 1.0))

        if len(self._loaded_files) == 1:
            self.setWindowTitle(f"Pass File Viewer — {path.name}")
            self._update_status(data, path)
        else:
            n = len(self._loaded_files)
            self.setWindowTitle(f"Pass File Viewer — {n} files")
            self._update_status_multi(merged)

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

    def _merge_loaded_files(self) -> PassData:
        """Concatenate all loaded files into a single PassData."""
        xs = [d.x for d, _ in self._loaded_files]
        ys = [d.y for d, _ in self._loaded_files]
        dwells = [d.dwell for d, _ in self._loaded_files]
        merged_x = np.concatenate(xs)
        merged_y = np.concatenate(ys)
        merged_dwell = np.concatenate(dwells)
        header = self._loaded_files[-1][0].header
        return PassData(
            header=header,
            x=merged_x, y=merged_y, dwell=merged_dwell,
            count=len(merged_x),
        )

    def _update_status_multi(self, merged: PassData) -> None:
        """Status bar text when multiple files are loaded."""
        n_files = len(self._loaded_files)
        total_size = sum(p.stat().st_size for _, p in self._loaded_files)
        size_mb = total_size / (1024 * 1024)
        latest_name = self._loaded_files[-1][1].name
        base = (
            f"  {n_files} files loaded  |  "
            f"Shots: {merged.count:,}  |  "
            f"Latest: {latest_name}  |  "
            f"Total: {size_mb:.1f} MB"
        )
        if self._viewer._kdtree is None:
            base += "  |  Building spatial index…"
        self._status_label.setText(base)

    # ── box selection ───────────────────────────────────────────────

    def _on_box_selection(self, indices: list[int]) -> None:
        """Handle box-selection signal from the viewer."""
        self._selection_pane.update_selection(indices)
        if len(indices):
            self._selection_dock.setVisible(True)

    def _fit_selection_dock(self, width: int) -> None:
        """Resize the selection dock to exactly fit its column content."""
        screen = self.screen()
        if screen is not None:
            max_w = screen.availableGeometry().width() // 2
            width = min(width, max_w)
        self.resizeDocks([self._selection_dock], [width], Qt.Orientation.Horizontal)

    # ── file selection (right-click menu) ───────────────────────────

    def _file_shot_indices(self, file_idx: int) -> np.ndarray:
        """Return the merged-array index range for the given file."""
        offset = sum(self._loaded_files[i][0].count for i in range(file_idx))
        count = self._loaded_files[file_idx][0].count
        return np.arange(offset, offset + count, dtype=np.intp)

    def _on_deselect_all_passes(self) -> None:
        for idx in list(self._file_selected):
            self._viewer.unpin_stripe(idx)
        self._file_selected.clear()
        self._apply_file_selection()

    def _on_select_all_passes(self) -> None:
        if not self._loaded_files:
            return
        for idx in range(len(self._loaded_files)):
            if idx not in self._file_selected:
                self._file_selected.add(idx)
                self._viewer.pin_stripe(idx)
        self._apply_file_selection()

    def _apply_file_selection(self) -> None:
        """Merge all file-selected shot ranges and push to the viewer."""
        if not self._file_selected or not self._loaded_files:
            self._viewer.set_locked_indices(None)
            self._viewer.select_shots(np.empty(0, dtype=np.intp))
            return
        parts = [self._file_shot_indices(i) for i in sorted(self._file_selected)]
        indices = np.concatenate(parts).astype(np.intp)
        self._viewer.set_locked_indices(indices)
        self._viewer.select_shots(indices)

    def _restore_pinned_stripes(self) -> None:
        """Re-pin all file-selected stripes after a set_stripe_regions call."""
        for idx in self._file_selected:
            if idx < len(self._loaded_files):
                self._viewer.pin_stripe(idx)
        self._apply_file_selection()

    def _on_stripe_right_clicked(self, stripe_indices: list, global_pos: QPoint) -> None:
        """Show a context menu listing hovered pass files as checkable items."""
        menu = QMenu(self)
        title = QAction("Select pass files:", self)
        title.setEnabled(False)
        menu.addAction(title)
        menu.addSeparator()
        actions: list[tuple[int, QAction]] = []
        for idx in stripe_indices:
            if idx >= len(self._loaded_files):
                continue
            _, path = self._loaded_files[idx]
            act = QAction(path.name, self)
            act.setCheckable(True)
            act.setChecked(idx in self._file_selected)
            menu.addAction(act)
            actions.append((idx, act))
        if not actions:
            return
        menu.exec(global_pos)
        # Process results: pin/unpin and rebuild selection
        changed = False
        for idx, act in actions:
            if act.isChecked() and idx not in self._file_selected:
                self._file_selected.add(idx)
                self._viewer.pin_stripe(idx)
                changed = True
            elif not act.isChecked() and idx in self._file_selected:
                self._file_selected.discard(idx)
                self._viewer.unpin_stripe(idx)
                changed = True
        if changed:
            self._apply_file_selection()
