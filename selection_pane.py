"""
Scrollable side-pane that lists box-selected shots in a table.

Contents are easily copyable: Ctrl+A selects all rows, Ctrl+C copies to clipboard
in tab-separated format suitable for pasting into spreadsheets.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PyQt6.QtGui import QKeySequence, QAction
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QAbstractItemView,
    QApplication,
    QHeaderView,
    QTableView,
)

from pass_parser import PassData


class _ShotTableModel(QAbstractTableModel):
    """Virtual table model — only formats data for rows Qt actually renders."""

    _COLUMNS = ("Shot #", "X (nm)", "Y (nm)", "Dwell (ns)")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._indices: np.ndarray = np.empty(0, dtype=np.intp)  # sorted shot indices
        self._data: PassData | None = None

    def set_selection(self, data: PassData | None, indices) -> None:
        self.beginResetModel()
        self._data = data
        if isinstance(indices, np.ndarray):
            self._indices = np.sort(indices).astype(np.intp)
        elif indices:
            self._indices = np.array(sorted(indices), dtype=np.intp)
        else:
            self._indices = np.empty(0, dtype=np.intp)
        self.endResetModel()

    # ── QAbstractTableModel interface ──

    def rowCount(self, parent=QModelIndex()):
        return len(self._indices)

    def columnCount(self, parent=QModelIndex()):
        return len(self._COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._COLUMNS[section]
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self._data is None:
            return None
        row, col = index.row(), index.column()
        idx = int(self._indices[row])

        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return f"{idx + 1:,}"
            elif col == 1:
                return f"{self._data.x[idx]:,.0f}"
            elif col == 2:
                return f"{self._data.y[idx]:,.0f}"
            elif col == 3:
                return f"{self._data.dwell[idx]:,.0f}"

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        return None


class SelectionPane(QWidget):
    """Side pane showing box-selected shot data in a copyable table."""

    _COLUMNS = _ShotTableModel._COLUMNS

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header row
        header_row = QHBoxLayout()
        self._header_label = QLabel("Selection  (0 shots)")
        self._header_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        header_row.addWidget(self._header_label)

        copy_btn = QPushButton("Copy All")
        copy_btn.setToolTip("Copy entire table to clipboard (tab-separated)")
        copy_btn.setFixedWidth(80)
        copy_btn.clicked.connect(self._copy_all)
        header_row.addWidget(copy_btn)
        layout.addLayout(header_row)

        # Model + View (virtual rows — scales to any size)
        self._model = _ShotTableModel(self)
        self._table = QTableView()
        self._table.setModel(self._model)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setSortingEnabled(False)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            "QTableView { font-family: Consolas, monospace; font-size: 12px; }"
            "QTableView::item { padding: 2px 6px; }"
        )
        layout.addWidget(self._table)

        # Keyboard shortcut: Ctrl+C copies selected rows
        copy_act = QAction(self)
        copy_act.setShortcut(QKeySequence.StandardKey.Copy)
        copy_act.triggered.connect(self._copy_selected)
        self.addAction(copy_act)

        self._data: PassData | None = None

    def set_data(self, data: PassData) -> None:
        """Store reference to the loaded pass data."""
        self._data = data

    def update_selection(self, indices: list[int]) -> None:
        """Populate the table with the given shot indices (0-based), sorted by shot number."""
        self._model.set_selection(self._data, indices)
        total = len(indices)
        self._header_label.setText(f"Selection  ({total:,} shots)")

    # ── clipboard helpers ───────────────────────────────────────────

    def _rows_to_text(self, rows: list[int]) -> str:
        """Convert table rows to tab-separated text with header."""
        lines = ["\t".join(self._COLUMNS)]
        model = self._model
        for r in sorted(rows):
            cols = []
            for c in range(model.columnCount()):
                val = model.data(model.index(r, c))
                cols.append(val if val else "")
            lines.append("\t".join(cols))
        return "\n".join(lines)

    def _copy_all(self) -> None:
        """Copy entire table to clipboard."""
        row_count = self._model.rowCount()
        if row_count:
            text = self._rows_to_text(list(range(row_count)))
            QApplication.clipboard().setText(text)

    def _copy_selected(self) -> None:
        """Copy selected rows to clipboard."""
        selected = sorted({idx.row() for idx in self._table.selectedIndexes()})
        if selected:
            text = self._rows_to_text(selected)
            QApplication.clipboard().setText(text)
