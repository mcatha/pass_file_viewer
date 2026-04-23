"""
Scrollable side-pane that lists box-selected shots in a table.

Contents are easily copyable: Ctrl+A selects all rows, Ctrl+C copies to clipboard
in tab-separated format suitable for pasting into spreadsheets.

Large selections (> _MAX_VIRTUAL_ROWS) use a virtual window: Qt only ever sees
a capped row count, avoiding the 32-bit integer overflow in QHeaderView::length()
that causes blank cells at ~71 M rows with 30 px default section height.  An
offset scrollbar below the table lets the user navigate the full dataset.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence, QAction
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollBar,
    QAbstractItemView,
    QHeaderView,
    QTableView,
)

from pass_parser import PassData

# Qt uses 32-bit int for pixel positions inside QHeaderView.  At the default
# section height (~30 px) the product rowCount × height overflows at ~71 M rows,
# causing the view to render nothing.  Keeping the virtual window below this
# threshold avoids the overflow on any platform.
_MAX_VIRTUAL_ROWS = 10_000_000   # 10 M rows × 30 px = 300 M << INT_MAX


class _ShotTableModel(QAbstractTableModel):
    """Virtual table model — only formats data for rows Qt actually renders."""

    _COLUMNS = ("Shot #", "File", "X (nm)", "Y (nm)", "Dwell (ns)")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._indices: np.ndarray = np.empty(0, dtype=np.intp)  # sorted shot indices
        self._data: PassData | None = None
        self._file_offsets: np.ndarray = np.array([0], dtype=np.intp)  # start index per file
        self._file_names: list[str] = [""]
        self._offset: int = 0   # first row of the virtual window into _indices

    def set_sorted(self, data: PassData | None, indices: np.ndarray) -> None:
        self.beginResetModel()
        self._data = data
        self._indices = indices
        self._offset = 0
        self.endResetModel()

    def set_file_boundaries(self, names: list[str], counts: list[int]) -> None:
        """Set per-file name and shot count so Shot # and File columns work correctly."""
        offsets = [0]
        for c in counts[:-1]:
            offsets.append(offsets[-1] + c)
        self._file_offsets = np.array(offsets, dtype=np.intp)
        self._file_names = [n[:-5] if n.lower().endswith('.pass') else n for n in names]

    def set_offset(self, offset: int) -> None:
        """Slide the virtual window to start at *offset* within _indices."""
        n = len(self._indices)
        clamped = max(0, min(offset, max(0, n - 1)))
        if clamped == self._offset:
            return
        self.beginResetModel()
        self._offset = clamped
        self.endResetModel()

    def clear(self) -> None:
        self.beginResetModel()
        self._indices = np.empty(0, dtype=np.intp)
        self._offset = 0
        self.endResetModel()

    def sort(self, column: int, order=Qt.SortOrder.AscendingOrder) -> None:
        # Skip sort for very large datasets — np.argsort on millions of rows
        # blocks the main thread.  Qt calls this automatically via updateGeometries()
        # when setSortingEnabled is True, so we must guard here rather than at the call site.
        _SORT_MAX = 500_000
        if self._data is None or len(self._indices) == 0 or column < 0:
            return
        if len(self._indices) > _SORT_MAX:
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
        self._offset = 0
        self.endResetModel()

    # ── QAbstractTableModel interface ──

    def rowCount(self, parent=QModelIndex()):
        n = len(self._indices)
        if n == 0:
            return 0
        return min(n - self._offset, _MAX_VIRTUAL_ROWS)

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
        idx = int(self._indices[self._offset + row])

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
    shot_activated = pyqtSignal(int)  # emits global shot index when a row is clicked

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
        copy_btn.setToolTip("Copy current window to clipboard (tab-separated)")
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
        self._table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerItem)
        self._table.setStyleSheet(
            "QTableView { font-family: Consolas, monospace; font-size: 12px; }"
            "QTableView::item { padding: 2px 6px; }"
        )
        self._table.clicked.connect(self._on_row_clicked)
        layout.addWidget(self._table)

        # Offset scrollbar — shown only when total rows exceed the virtual window
        self._offset_bar = QScrollBar(Qt.Orientation.Horizontal)
        self._offset_bar.setRange(0, 0)
        self._offset_bar.setSingleStep(max(1, _MAX_VIRTUAL_ROWS // 100))
        self._offset_bar.setPageStep(_MAX_VIRTUAL_ROWS)
        self._offset_bar.valueChanged.connect(self._on_offset_changed)
        self._offset_bar.setVisible(False)
        layout.addWidget(self._offset_bar)

        self._offset_label = QLabel("")
        self._offset_label.setStyleSheet("font-size: 11px; color: #aaa;")
        self._offset_label.setVisible(False)
        layout.addWidget(self._offset_label)

        # Keyboard shortcut: Ctrl+C copies selected rows
        copy_act = QAction(self)
        copy_act.setShortcut(QKeySequence.StandardKey.Copy)
        copy_act.triggered.connect(self._copy_selected)
        self.addAction(copy_act)

        self._data: PassData | None = None
        self._current_indices: np.ndarray = np.empty(0, dtype=np.intp)
        self._total_shots: int = 0

    def set_data(self, data: PassData) -> None:
        """Store reference to the loaded pass data."""
        self._data = data
        self._current_indices = np.empty(0, dtype=np.intp)
        self._model.set_sorted(data, self._current_indices)

    def set_file_boundaries(self, names: list[str], counts: list[int]) -> None:
        """Update per-file name/count info so Shot # and File columns are correct."""
        self._model.set_file_boundaries(names, counts)

    def update_selection(self, indices) -> None:
        """Populate the table with the given shot indices (0-based).

        All callers pass already-sorted indices (file-selected indices are sorted
        aranges; box-selected indices come from np.nonzero which is always sorted).
        """
        if self._data is None:
            return  # data not loaded yet; update_selection after set_data will follow

        arr = np.asarray(indices, dtype=np.intp)  # O(1) view when dtype already matches

        # Skip re-render if selection hasn't changed
        if arr is self._current_indices:
            return
        total = len(arr)
        if (total == len(self._current_indices)
                and total < 500_000
                and np.array_equal(arr, self._current_indices)):
            return

        self._current_indices = arr
        self._total_shots = total
        self._header_label.setText(f"Selection  ({total:,} shots)")
        self._model.set_sorted(self._data, arr)
        self._update_offset_bar(total)
        QTimer.singleShot(0, self._emit_content_width)

    def _update_offset_bar(self, total: int) -> None:
        overflow = total > _MAX_VIRTUAL_ROWS
        if overflow:
            max_offset = total - _MAX_VIRTUAL_ROWS
            self._offset_bar.blockSignals(True)
            self._offset_bar.setRange(0, max_offset)
            self._offset_bar.setValue(0)
            self._offset_bar.blockSignals(False)
            self._offset_bar.setVisible(True)
            self._refresh_offset_label(0, total)
        else:
            self._offset_bar.setVisible(False)
            self._offset_label.setVisible(False)

    def _refresh_offset_label(self, offset: int, total: int) -> None:
        end = min(offset + _MAX_VIRTUAL_ROWS, total)
        self._offset_label.setText(
            f"Rows {offset + 1:,} – {end:,} of {total:,}"
        )
        self._offset_label.setVisible(True)

    def _on_offset_changed(self, value: int) -> None:
        self._model.set_offset(value)
        self._refresh_offset_label(value, self._total_shots)

    def _emit_content_width(self) -> None:
        header = self._table.horizontalHeader()
        col_w = sum(header.sectionSize(i) for i in range(header.count()))
        scrollbar_w = self._table.verticalScrollBar().sizeHint().width()
        margins = self.contentsMargins()
        w = col_w + scrollbar_w + margins.left() + margins.right() + 8
        self.content_ready.emit(w)

    def shutdown(self) -> None:
        """No-op — kept for API compatibility."""
        pass

    # ── clipboard helpers ───────────────────────────────────────────

    def _on_row_clicked(self, index) -> None:
        row = index.row()
        actual = self._model._offset + row
        if 0 <= actual < len(self._model._indices):
            self.shot_activated.emit(int(self._model._indices[actual]))

    def highlight_shot(self, global_idx: int) -> None:
        """Highlight and scroll to the row for global_idx, or clear if -1."""
        if global_idx < 0 or len(self._model._indices) == 0:
            self._table.clearSelection()
            return
        rows = np.where(self._model._indices == global_idx)[0]
        if len(rows) == 0:
            self._table.clearSelection()
            return
        actual_row = int(rows[0])
        virtual_row = actual_row - self._model._offset
        if not (0 <= virtual_row < self._model.rowCount()):
            # Navigate the offset window to contain this row
            self._offset_bar.setValue(actual_row)
            virtual_row = 0
        self._table.selectRow(virtual_row)
        self._table.scrollTo(self._model.index(virtual_row, 0))

    def _rows_to_text(self, rows: list[int]) -> str:
        """Convert virtual table rows to tab-separated text with header."""
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
        """Copy current window to clipboard."""
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
