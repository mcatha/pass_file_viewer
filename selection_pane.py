"""
Scrollable side-pane that lists box-selected shots in a table.

Contents are easily copyable: Ctrl+A selects all rows, Ctrl+C copies to clipboard
in tab-separated format suitable for pasting into spreadsheets.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, QThread, QObject, QTimer, pyqtSignal
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


class _SortWorker(QObject):
    """Sorts an index array on a background thread."""
    finished = pyqtSignal(object)  # emits sorted np.ndarray (np.intp)

    def __init__(self, indices) -> None:
        super().__init__()
        self._indices = indices

    def run(self) -> None:
        if isinstance(self._indices, np.ndarray):
            result = np.sort(self._indices).astype(np.intp)
        else:
            result = np.array(sorted(self._indices), dtype=np.intp)
        self.finished.emit(result)


class _ShotTableModel(QAbstractTableModel):
    """Virtual table model — only formats data for rows Qt actually renders."""

    _COLUMNS = ("Shot #", "File", "X (nm)", "Y (nm)", "Dwell (ns)")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._indices: np.ndarray = np.empty(0, dtype=np.intp)  # sorted shot indices
        self._data: PassData | None = None
        self._file_offsets: np.ndarray = np.array([0], dtype=np.intp)  # start index per file
        self._file_names: list[str] = [""]

    def set_sorted(self, data: PassData | None, indices: np.ndarray) -> None:
        self.beginResetModel()
        self._data = data
        self._indices = indices
        self.endResetModel()

    def set_file_boundaries(self, names: list[str], counts: list[int]) -> None:
        """Set per-file name and shot count so Shot # and File columns work correctly."""
        offsets = [0]
        for c in counts[:-1]:
            offsets.append(offsets[-1] + c)
        self._file_offsets = np.array(offsets, dtype=np.intp)
        self._file_names = list(names)

    def clear(self) -> None:
        self.beginResetModel()
        self._indices = np.empty(0, dtype=np.intp)
        self.endResetModel()

    def sort(self, column: int, order=Qt.SortOrder.AscendingOrder) -> None:
        if self._data is None or len(self._indices) == 0 or column < 0:
            return
        idx = self._indices
        if column == 0:   # per-file shot number
            file_idxs = np.searchsorted(self._file_offsets, idx, side='right') - 1
            keys = idx - self._file_offsets[file_idxs]
        elif column == 1:  # file (sort by load order)
            keys = np.searchsorted(self._file_offsets, idx, side='right') - 1
        elif column == 2:
            keys = self._data.x[idx]
        elif column == 3:
            keys = self._data.y[idx]
        elif column == 4:
            keys = self._data.dwell[idx]
        else:
            return
        order_arr = np.argsort(keys, kind='stable')
        if order == Qt.SortOrder.DescendingOrder:
            order_arr = order_arr[::-1]
        self.beginResetModel()
        self._indices = idx[order_arr]
        self.endResetModel()

    # ── QAbstractTableModel interface ──

    def rowCount(self, parent=QModelIndex()):
        return len(self._indices)

    def columnCount(self, parent=QModelIndex()):
        return len(self._COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal:
            if role == Qt.ItemDataRole.DisplayRole:
                return self._COLUMNS[section]
            if role == Qt.ItemDataRole.TextAlignmentRole:
                return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self._data is None:
            return None
        row, col = index.row(), index.column()
        idx = int(self._indices[row])

        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                file_idx = int(np.searchsorted(self._file_offsets, idx, side='right') - 1)
                shot_num = idx - int(self._file_offsets[file_idx]) + 1
                return f"{shot_num:,}"
            elif col == 1:
                file_idx = int(np.searchsorted(self._file_offsets, idx, side='right') - 1)
                return self._file_names[file_idx] if file_idx < len(self._file_names) else ""
            elif col == 2:
                return f"{self._data.x[idx]:,.0f}"
            elif col == 3:
                return f"{self._data.y[idx]:,.0f}"
            elif col == 4:
                return f"{self._data.dwell[idx]:,.0f}"

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

        return None


class SelectionPane(QWidget):
    """Side pane showing box-selected shot data in a copyable table."""

    _COLUMNS = _ShotTableModel._COLUMNS
    content_ready = pyqtSignal(int)   # emits required pixel width after data loads

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
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setResizeContentsPrecision(50)
        self._table.horizontalHeader().setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
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
        self._sort_thread: QThread | None = None
        self._sort_worker: _SortWorker | None = None
        self._pending_count: int = 0
        self._current_indices: np.ndarray = np.empty(0, dtype=np.intp)

    def set_data(self, data: PassData) -> None:
        """Store reference to the loaded pass data."""
        self._data = data
        self._current_indices = np.empty(0, dtype=np.intp)

    def set_file_boundaries(self, names: list[str], counts: list[int]) -> None:
        """Update per-file name/count info so Shot # and File columns are correct."""
        self._model.set_file_boundaries(names, counts)

    def update_selection(self, indices: list[int]) -> None:
        """Populate the table with the given shot indices (0-based), sorted by shot number."""
        arr = np.asarray(indices, dtype=np.intp)
        if (len(arr) == len(self._current_indices)
                and np.array_equal(np.sort(arr), self._current_indices)):
            return  # identical selection — skip the clear/re-sort flash
        total = len(indices)
        self._pending_count = total
        self._header_label.setText(f"Selection  ({total:,} shots)  —  loading…")
        self._model.clear()
        if self._sort_worker is not None:
            self._sort_worker.finished.disconnect(self._on_sort_ready)
            self._sort_thread.quit()
        thread = QThread(self)
        worker = _SortWorker(indices)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_sort_ready)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_sort_thread_done)
        self._sort_thread = thread
        self._sort_worker = worker
        thread.start()

    def _on_sort_ready(self, sorted_indices: np.ndarray) -> None:
        self._current_indices = sorted_indices   # sorted ascending — used for dedup check
        self._model.set_sorted(self._data, sorted_indices)
        self._header_label.setText(f"Selection  ({self._pending_count:,} shots)")
        QTimer.singleShot(0, self._emit_content_width)

    def _emit_content_width(self) -> None:
        header = self._table.horizontalHeader()
        col_w = sum(header.sectionSize(i) for i in range(header.count()))
        scrollbar_w = self._table.verticalScrollBar().sizeHint().width()
        margins = self.contentsMargins()
        w = col_w + scrollbar_w + margins.left() + margins.right() + 8
        self.content_ready.emit(w)

    def _on_sort_thread_done(self) -> None:
        self._sort_thread = None
        self._sort_worker = None

    def shutdown(self) -> None:
        """Stop any running background thread.  Call before the widget is destroyed."""
        if self._sort_thread is not None:
            self._sort_thread.quit()
            self._sort_thread.wait()
            self._sort_thread = None
            self._sort_worker = None

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
