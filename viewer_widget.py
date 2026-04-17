"""
GPU-accelerated shot viewer widget using vispy + OpenGL.

Renders millions of shots as size-scaled markers on a 2D cartesian plane.
Supports zoom (scroll), pan (right-drag), 2D rotate (middle-drag / Shift+left-drag),
and hover tooltips via scipy KD-tree spatial lookup.
"""

from __future__ import annotations

import math as _math
import numpy as np
from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF, QTimer, QThread, QObject, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPolygonF
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

from scipy.spatial import cKDTree
import vispy
try:
    vispy.use(gl='gl+')         # prefer GL3+ backend (instanced rendering)
except Exception:
    vispy.use(gl='gl2')         # fallback for machines without GL3+ support
from vispy import scene
from vispy.scene import visuals
from gaussian_markers import GaussianMarkers
from pass_parser import PassData

# ── constants ───────────────────────────────────────────────────────
# Diameter in nm = dwell_ns * _NM_PER_NS_DWELL  (10 nm per µs)
_NM_PER_NS_DWELL = 0.01
_DISC_SIZE_SCALE = 0.667
_SHOT_COLOR = np.array([0.30, 0.60, 1.00, 1.0])  # bright blue, fully opaque
_ALPHA_NEAR = 1.00     # (legacy, unused — α max replaces this)
_ALPHA_FAR  = 0.005    # Gaussian per-shot alpha at max zoom-out (tiny markers, need visibility)
_ALPHA_MAX  = 0.330    # Gaussian per-shot alpha cap at close zoom
_DISC_OVERLAP_WHITE = 0.05  # disc-mode: additive white alpha per overlap

_SELECTED_COLOR = np.array([1.0, 0.90, 0.0, 1.0])  # bright gold for selected shot
_BOX_SELECTED_COLOR = np.array([0.3, 1.0, 0.5, 1.0])  # bright green for box-selected
_LINE_COLOR = (1.0, 0.40, 0.25, 0.7)      # bright red-ish
_SEL_LINE_COLOR = (1.0, 0.90, 0.0, 0.95)  # bright gold, matching selected shot
_MIN_SEL_PX = 7  # minimum click-selection marker size in screen pixels
_MIN_BOX_SEL_PX = 4  # minimum box-selection marker size in screen pixels
_RUBBER_BAND_PEN = QColor(100, 200, 255, 200)
_RUBBER_BAND_FILL = QColor(100, 200, 255, 40)
_BG_COLOR = "#1e1e1e"

# Unit circle for wafer outline (257 points → 256 segments, closed)
_theta = np.linspace(0, 2 * np.pi, 257)
_UNIT_CIRCLE = np.column_stack([np.cos(_theta), np.sin(_theta)])
_ARROW_COLOR = QColor(200, 200, 200, 220)
_LABEL_COLOR = QColor(255, 255, 255, 255)    # pure white
_ARROW_SIZE = 20       # side‑length of arrowhead triangle in px
_AXIS_LABEL_FONT = QFont("Consolas", 17, QFont.Weight.Bold)
_STRIDE1_DPP = 2.5         # nm/px at which stride = 1 (below = no stride)
_STRIDE_EXPONENT = 0.4     # power curve: <1 ramps fast then flattens, >1 gentle then steep
# ── Mode-specific budget parameters ──
# Gaussian (additive): brightness comes from accumulation, so we need
# high shot density even when data covers few screen pixels.
_GAUSS_SHOTS_PER_PX = 20.0      # per-pixel density target
_GAUSS_MIN_BUDGET   = 0          # no floor
_GAUSS_MAX_RENDERED = 2_097_152  # hard cap (2^21)
# Disc (alpha blend): each shot is fully opaque, so fewer are fine.
_DISC_SHOTS_PER_PX  = 3.0
_DISC_MIN_BUDGET    = 0          # no floor
_DISC_MAX_RENDERED  = 2_097_152
print(f"[INIT] gauss={_GAUSS_SHOTS_PER_PX}/px min={_GAUSS_MIN_BUDGET} cap={_GAUSS_MAX_RENDERED}  disc={_DISC_SHOTS_PER_PX}/px cap={_DISC_MAX_RENDERED}")

_ALPHA_DREF = 16.5      # sigmoid midpoint: DPP where alpha is halfway between far and max
_ALPHA_P = 1.5          # sigmoid steepness (higher = sharper transition)
_STRIDE_INFLATE_AMP = 0.50  # size *= 1 + amp*log10(stride)/(log10(stride)+0.2)


class _RightPanCamera(scene.PanZoomCamera):
    """PanZoomCamera that pans on right-drag instead of zooming.

    Left-drag is reserved for rubber-band selection (handled before the
    camera sees it), so we remap:
      right-drag (button 2) → pan
      scroll wheel           → zoom
      drag-zoom              → disabled
    """

    def viewbox_mouse_event(self, event):
        if event.handled or not self.interactive:
            return

        # Let base class handle scroll / gesture zoom
        from vispy.scene.cameras import BaseCamera
        BaseCamera.viewbox_mouse_event(self, event)

        if event.type == 'mouse_wheel':
            center = self._scene_transform.imap(event.pos)
            self.zoom((1 + self.zoom_factor) ** (-event.delta[1] * 30), center)
            event.handled = True
        elif event.type == 'gesture_zoom':
            center = self._scene_transform.imap(event.pos)
            self.zoom(1 - event.scale, center)
            event.handled = True
        elif event.type == 'mouse_move':
            if event.press_event is None:
                return
            modifiers = event.mouse_event.modifiers
            # Right-drag → pan (original camera uses button 1)
            if 2 in event.buttons and not modifiers:
                p1 = np.array(event.last_event.pos)[:2]
                p2 = np.array(event.pos)[:2]
                p1s = self._transform.imap(p1)
                p2s = self._transform.imap(p2)
                self.pan(p1s - p2s)
                event.handled = True
            else:
                event.handled = False
        elif event.type == 'mouse_press':
            event.handled = event.button in [1, 2]
        else:
            event.handled = False


class _KDTreeWorker(QObject):
    """Builds a cKDTree on a background thread."""
    finished = pyqtSignal(object)  # emits the cKDTree

    def __init__(self, positions: np.ndarray) -> None:
        super().__init__()
        self._positions = positions

    def run(self) -> None:
        tree = cKDTree(self._positions)
        self.finished.emit(tree)


class _RubberBandOverlay(QWidget):
    """Transparent overlay that draws the rubber-band selection rectangle."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        self._rect: QRectF | None = None

    def set_rect(self, rect: QRectF | None) -> None:
        self._rect = rect
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        if self._rect is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(_RUBBER_BAND_PEN, 1.5, Qt.PenStyle.DashLine))
        p.setBrush(QBrush(_RUBBER_BAND_FILL))
        p.drawRect(self._rect)
        p.end()


class _AxisArrowOverlay(QWidget):
    """Transparent overlay that paints arrowheads + labels for rotated axes."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        # Each arrow: (origin_screen_x, origin_screen_y, dir_x, dir_y, label)
        # dir is the unit vector in *screen* space (+Y is down in screen)
        self.arrows: list[tuple[float, float, float, float, str]] = []

    @staticmethod
    def _line_rect_intersections(
        ox: float, oy: float, dx: float, dy: float,
        w: float, h: float, margin: float,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Find where the infinite line through (ox, oy) in direction (dx, dy)
        intersects the viewport inset by *margin*.

        Returns ((near_x, near_y), (far_x, far_y)) — the two intersection
        points where the line crosses the viewport boundary. 'near' is the
        point at lowest t (most negative direction), 'far' at highest t
        (most positive direction).  Returns None if the line doesn't cross."""
        xmin, xmax = margin, w - margin
        ymin, ymax = margin, h - margin
        t_near = -float('inf')
        t_far = float('inf')
        for lo, hi, o, d in [(xmin, xmax, ox, dx), (ymin, ymax, oy, dy)]:
            if abs(d) < 1e-12:
                if o < lo or o > hi:
                    return None          # parallel and outside
                continue
            t1 = (lo - o) / d
            t2 = (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)
        if t_far < t_near:
            return None
        return ((ox + t_near * dx, oy + t_near * dy),
                (ox + t_far * dx, oy + t_far * dy))

    def paintEvent(self, event) -> None:  # noqa: N802
        if not self.arrows:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cw, ch = self.width(), self.height()
        pen = QPen(QColor(0, 0, 0, 200), 2)   # black outline on arrows
        p.setPen(pen)
        p.setBrush(QBrush(_ARROW_COLOR))
        p.setFont(_AXIS_LABEL_FONT)
        s = _ARROW_SIZE
        margin = 6

        # Prepare a QPainterPath-based text outline for high contrast labels
        from PyQt6.QtGui import QPainterPath

        for tx, ty, dx, dy, label in self.arrows:

            # Arrowhead geometry
            nx, ny = -dy, dx  # perpendicular in screen space
            base_x = tx - dx * s
            base_y = ty - dy * s

            # Build arrow + label at nominal positions, get real bounding box,
            # then shift everything to stay on screen.
            gap = s * 1.2
            lcx = base_x - dx * gap
            lcy = base_y - dy * gap

            # Create the label path at nominal position to measure true bounds
            tmp_label = QPainterPath()
            tmp_label.addText(0, 0, _AXIS_LABEL_FONT, label)
            lbr = tmp_label.boundingRect()  # actual glyph bounds

            # Place label so its bounding box is centered on (lcx, lcy)
            lbl_x = lcx - lbr.width() * 0.5 - lbr.x()
            lbl_y = lcy - lbr.height() * 0.5 - lbr.y()

            # Compute true bounding rect of text at that position
            label_path = QPainterPath()
            label_path.addText(lbl_x, lbl_y, _AXIS_LABEL_FONT, label)
            text_rect = label_path.boundingRect()

            # Collect all extreme coordinates (arrow + label)
            pad = 4
            all_min_x = min(tx, base_x - abs(nx) * s * 0.5, text_rect.left())
            all_max_x = max(tx, base_x + abs(nx) * s * 0.5, text_rect.right())
            all_min_y = min(ty, base_y - abs(ny) * s * 0.5, text_rect.top())
            all_max_y = max(ty, base_y + abs(ny) * s * 0.5, text_rect.bottom())

            # Also include the opposite base corner
            for sx2 in [1, -1]:
                all_min_x = min(all_min_x, base_x + sx2 * nx * s * 0.5)
                all_max_x = max(all_max_x, base_x + sx2 * nx * s * 0.5)
                all_min_y = min(all_min_y, base_y + sx2 * ny * s * 0.5)
                all_max_y = max(all_max_y, base_y + sx2 * ny * s * 0.5)

            # Compute shift to keep everything on screen
            shift_x = shift_y = 0.0
            if all_min_x < pad:
                shift_x = pad - all_min_x
            elif all_max_x > cw - pad:
                shift_x = (cw - pad) - all_max_x
            if all_min_y < pad:
                shift_y = pad - all_min_y
            elif all_max_y > ch - pad:
                shift_y = (ch - pad) - all_max_y

            # Apply shift
            tx += shift_x;  ty += shift_y
            base_x += shift_x;  base_y += shift_y

            tri = QPolygonF([
                QPointF(tx, ty),
                QPointF(base_x + nx * s * 0.5, base_y + ny * s * 0.5),
                QPointF(base_x - nx * s * 0.5, base_y - ny * s * 0.5),
            ])
            p.drawPolygon(tri)

            # Redraw label at shifted position
            shifted_label = QPainterPath()
            shifted_label.addText(lbl_x + shift_x, lbl_y + shift_y,
                                  _AXIS_LABEL_FONT, label)
            p.setPen(QPen(QColor(0, 0, 0, 220), 3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(shifted_label)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(_LABEL_COLOR))
            p.drawPath(shifted_label)
            # Restore pen/brush for next arrow
            p.setPen(pen)
            p.setBrush(QBrush(_ARROW_COLOR))

        p.end()


class ShotViewerWidget(QWidget):
    """QWidget wrapper around a vispy SceneCanvas for shot visualisation."""

    # (no artificial display-point cap — stride is derived from zoom level)

    # Emitted when a box selection finishes.  Carries 0-based shot indices (list or ndarray).
    box_selected = pyqtSignal(object)

    # Emitted when the KD-tree finishes building in the background.
    kdtree_ready = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # vispy canvas
        self._canvas = scene.SceneCanvas(
            keys="interactive", bgcolor=_BG_COLOR, parent=self,
        )
        self._view = self._canvas.central_widget.add_view()

        # Camera: right-drag = pan, scroll = zoom, left-drag = rubber-band
        self._camera = _RightPanCamera(aspect=1)
        self._view.camera = self._camera

        # Intermediate node that holds every visual.  We apply rotation
        # to this node so the camera's own transforms are unaffected.
        # Grid MUST be created before _visual_root so it renders behind
        # all markers (GridLines uses additive blend internally).
        self._grid = visuals.GridLines(parent=self._view.scene, color=(0.3, 0.3, 0.3, 0.5))
        self._visual_root = scene.Node(parent=self._view.scene)

        # ── Gaussian markers: additive Gaussian PSF ─────────────────
        self._gauss_markers = GaussianMarkers(parent=self._visual_root,
                                               antialias=2,
                                               scaling='scene', symbol='disc',
                                               method='instanced')
        self._gauss_markers.set_gl_state(blend=True, depth_test=False,
                                          blend_func=('src_alpha', 'one'))

        # ── Disc markers: base (alpha blend) + overlap (additive white) ─
        self._disc_base_markers = visuals.Markers(parent=self._visual_root,
                                                    antialias=2,
                                                    scaling='scene', symbol='disc',
                                                    method='instanced')
        self._disc_base_markers.set_gl_state(blend=True, depth_test=False,
                                              blend_func=('src_alpha', 'one_minus_src_alpha'))

        self._disc_overlap_markers = visuals.Markers(parent=self._visual_root,
                                                       antialias=2,
                                                       scaling='scene', symbol='disc',
                                                       method='instanced')
        self._disc_overlap_markers.set_gl_state(blend=True, depth_test=False,
                                                 blend_func=('src_alpha', 'one'))

        # Gaussian markers hidden by default (disc mode is default)
        self._gauss_markers.visible = False

        # Active layer aliases (set by _apply_marker_mode)
        self._marker_mode: str = 'disc'   # 'gaussian' or 'disc'
        self._markers = self._disc_base_markers
        self._overlap_markers = self._disc_overlap_markers

        # Connection lines render ON TOP of markers
        self._lines = visuals.Line(parent=self._visual_root, color=_LINE_COLOR, width=1)
        self._lines.visible = False

        # Overlay lines for connections to/from selected shot
        self._sel_lines = visuals.Line(parent=self._visual_root, color=_SEL_LINE_COLOR, width=3)
        self._sel_lines.visible = False

        # Ruler line (data-space, white)
        self._ruler_line = visuals.Line(parent=self._visual_root, color=(1, 1, 1, 0.9), width=2)
        self._ruler_line.visible = False
        # Ruler tick marks (perpendicular hash marks)
        self._ruler_ticks = visuals.Line(parent=self._visual_root, color=(1, 1, 1, 0.7), width=1)
        self._ruler_ticks.visible = False

        # Overlay markers for box-selected shots — disc versions
        self._disc_box_sel_markers = visuals.Markers(parent=self._visual_root, antialias=5,
                                              scaling='scene', symbol='disc',
                                              method='instanced')
        self._disc_box_sel_markers.set_gl_state('translucent', depth_test=False)
        self._disc_box_sel_markers.set_data(
            np.zeros((1, 2), dtype=np.float32), size=0, edge_width=0
        )
        self._disc_box_sel_markers.visible = False

        # Overlay markers for box-selected shots — Gaussian versions
        self._gauss_box_sel_markers = GaussianMarkers(parent=self._visual_root, antialias=5,
                                              scaling='scene', symbol='disc',
                                              method='instanced')
        self._gauss_box_sel_markers.set_gl_state('translucent', depth_test=False)
        self._gauss_box_sel_markers.set_data(
            np.zeros((1, 2), dtype=np.float32), size=0, edge_width=0
        )

        # Overlay marker for single-click selected shot — disc version
        self._disc_sel_marker = visuals.Markers(parent=self._visual_root,
                                           antialias=2.5, scaling='scene', symbol='disc',
                                           method='instanced')
        self._disc_sel_marker.set_gl_state('translucent', depth_test=False)
        self._disc_sel_marker.set_data(
            np.zeros((1, 2), dtype=np.float32), size=0, edge_width=0
        )
        self._disc_sel_marker.visible = False

        # Overlay marker for single-click selected shot — Gaussian version
        self._gauss_sel_marker = GaussianMarkers(parent=self._visual_root,
                                           antialias=2.5, scaling='scene', symbol='disc',
                                           method='instanced')
        self._gauss_sel_marker.set_gl_state('translucent', depth_test=False)
        self._gauss_sel_marker.set_data(
            np.zeros((1, 2), dtype=np.float32), size=0, edge_width=0
        )
        self._gauss_sel_marker.visible = False

        # Active selection aliases (set by _apply_marker_mode)
        self._sel_marker = self._disc_sel_marker
        self._box_sel_markers = self._disc_box_sel_markers

        # Origin crosshair and axis lines
        _AXIS_COLOR = (0.6, 0.6, 0.6, 0.8)
        self._axis_half_len: float = 1e6  # grows on zoom-out; load_data resets from data extent
        hl = self._axis_half_len
        self._x_axis = visuals.Line(
            pos=np.array([[-hl, 0], [hl, 0]], dtype=np.float64),
            color=_AXIS_COLOR, width=1, parent=self._visual_root,
        )
        self._y_axis = visuals.Line(
            pos=np.array([[0, -hl], [0, hl]], dtype=np.float64),
            color=_AXIS_COLOR, width=1, parent=self._visual_root,
        )
        # Origin marker
        self._origin_marker = visuals.Markers(parent=self._visual_root)
        self._origin_marker.set_data(
            np.zeros((1, 2), dtype=np.float32),
            size=8, face_color=(1, 1, 1, 0.9), edge_width=0,
        )
        # Wafer outline circle
        self._wafer_diameter_nm: float | None = None
        self._wafer_outline = visuals.Line(
            pos=np.zeros((2, 2), dtype=np.float64),
            color=_AXIS_COLOR, width=1, connect='strip',
            parent=self._visual_root,
        )
        self._wafer_outline.visible = False
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas.native)

        # Rubber-band selection overlay
        self._rubber_overlay = _RubberBandOverlay(self._canvas.native)
        self._rubber_overlay.show()

        # Graphical arrow overlay (painted on top of the GL canvas)
        self._arrow_overlay = _AxisArrowOverlay(self._canvas.native)
        self._arrow_overlay.show()

        # Coordinate readout label at bottom of canvas
        self._coord_label = QLabel(self._canvas.native)
        self._coord_label.setStyleSheet(
            "background-color: rgba(30, 30, 30, 200);"
            "color: #bbb;"
            "padding: 2px 8px;"
            "font-family: Consolas, monospace;"
            "font-size: 11px;"
        )
        self._coord_label.setText("X: — nm   Y: — nm")
        self._coord_label.adjustSize()
        self._coord_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._coord_label.show()
        # Reposition on resize
        self._canvas.events.resize.connect(self._reposition_coord_label)

        # Tooltip style shared by both labels
        _tooltip_style = (
            "background-color: rgba(40, 40, 40, 230);"
            "color: #ddd;"
            "border: 1px solid #666;"
            "border-radius: 4px;"
            "padding: 4px 8px;"
            "font-family: Consolas, monospace;"
            "font-size: 12px;"
        )

        # Persistent tooltip for the click-selected shot
        self._tooltip = QLabel(self._canvas.native)
        self._tooltip.setStyleSheet(_tooltip_style)
        self._tooltip.setVisible(False)
        self._tooltip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Hover tooltip that follows the mouse
        self._hover_tooltip = QLabel(self._canvas.native)
        self._hover_tooltip.setStyleSheet(_tooltip_style)
        self._hover_tooltip.setVisible(False)
        self._hover_tooltip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Ruler distance label
        self._ruler_label = QLabel(self._canvas.native)
        self._ruler_label.setStyleSheet(
            "background-color: rgba(30, 30, 30, 220);"
            "color: #fff;"
            "border: 1px solid #999;"
            "border-radius: 3px;"
            "padding: 2px 6px;"
            "font-family: Consolas, monospace;"
            "font-size: 12px;"
        )
        self._ruler_label.setVisible(False)
        self._ruler_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # On-screen shot count (top-left corner)
        self._shot_count_label = QLabel(self._canvas.native)
        self._shot_count_label.setStyleSheet(
            "background-color: rgba(30, 30, 30, 180);"
            "color: #aaa;"
            "border: none;"
            "padding: 2px 6px;"
            "font-family: Consolas, monospace;"
            "font-size: 11px;"
        )
        self._shot_count_label.move(4, 4)
        self._shot_count_label.setVisible(False)
        self._shot_count_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Data state
        self._data: PassData | None = None
        self._positions: np.ndarray | None = None  # Nx2 float32, centroid-shifted
        self._origin: np.ndarray = np.zeros(2, dtype=np.float64)  # centroid offset
        self._sizes: np.ndarray | None = None
        self._kdtree: cKDTree | None = None
        self._kdtree_thread: QThread | None = None
        self._rotation_deg: float = 0.0
        self._selected_idx: int | None = None       # currently selected shot index
        self._box_selected_indices: np.ndarray = np.empty(0, dtype=np.intp)  # box-selected shot indices
        self._lines_data_set: bool = False           # True once line data is uploaded

        self._all_positions: np.ndarray | None = None  # full (N,2) float32
        self._all_sizes: np.ndarray | float | None = None  # full sizes
        self._raw_dwells: np.ndarray | None = None  # raw dwell values (ns)
        self._fwhm_scale: float = 6.0   # multiplier on _NM_PER_NS_DWELL (default 60 nm/µs)
        self._stride_inflate_amp: float = _STRIDE_INFLATE_AMP  # hardcoded
        self._alpha_comp_power: float = 1.0  # hardcoded: alpha /= stride_scale
        self._alpha_far: float = _ALPHA_FAR  # hardcoded floor
        self._alpha_max: float = _ALPHA_MAX  # close-zoom alpha cap (user-adjustable)
        self._alpha_dref: float = _ALPHA_DREF  # sigmoid midpoint DPP (user-adjustable)
        self._alpha_p: float = _ALPHA_P  # sigmoid steepness (hardcoded)
        self._disc_overlap_white: float = _DISC_OVERLAP_WHITE  # disc overlap alpha (user-adjustable)
        self._disc_dpp_low: float = 0.01    # disc log-linear: f=1 below this DPP
        self._disc_dpp_mid: float = 5000.0   # disc log-linear: intermediate DPP
        self._disc_f_mid: float = 0.2        # disc log-linear: alpha at dpp_mid
        self._disc_dpp_high: float = 1e10    # disc log-linear: f=0 above this DPP
        self._disc_inflate_amp: float = _STRIDE_INFLATE_AMP  # disc stride inflation (user-adjustable)
        self._decim_stride: int = 1  # current display stride (1 = all points)

        # Throttled shot decimation rebuild on zoom/pan
        self._shot_decim_timer = QTimer(self)
        self._shot_decim_timer.setSingleShot(True)
        self._shot_decim_timer.setInterval(100)
        self._shot_decim_timer.timeout.connect(self._update_decim_stride)

        # Mouse-move throttle: avoid KD-tree queries faster than display refresh
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(16)  # ~60 fps cap
        self._hover_timer.timeout.connect(self._do_hover_query)
        self._pending_hover_pos: tuple[float, float] | None = None
        self._pending_hover_px: tuple[int, int] = (0, 0)

        # Stripe region hover state
        self._stripe_regions: list[dict] = []
        self._stripe_aabbs: list[tuple[float, float, float, float]] = []
        self._stripe_rect: visuals.Rectangle | None = None
        self._hovered_stripe_idx: int | None = None

        self._stripe_tooltip = QLabel(self._canvas.native)
        self._stripe_tooltip.setStyleSheet(_tooltip_style)
        self._stripe_tooltip.setVisible(False)
        self._stripe_tooltip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Rubber-band drag state
        self._rubber_dragging = False
        self._rubber_start_px: tuple[float, float] | None = None  # pixel coords
        self._rubber_start_data: np.ndarray | None = None          # data coords

        # Ruler state
        self._ruler_start: np.ndarray | None = None   # data coords (centroid-shifted)
        self._ruler_end: np.ndarray | None = None     # data coords (centroid-shifted)
        self._ruler_active: bool = False               # True while stretching to mouse

        # Mouse tracking for hover
        self._canvas.native.setMouseTracking(True)
        self._canvas.events.mouse_move.connect(self._on_mouse_move)

        # Reposition tooltip on camera changes (pan/zoom)
        self._camera.events.transform_change.connect(self._on_camera_change)
        # PanZoomCamera.transform_change is unreliable, so also listen to draw.
        # The reentrancy guard in _on_camera_change prevents the old infinite loop.
        self._canvas.events.draw.connect(self._on_camera_change)

        # 2D rotation via Shift + left-drag
        self._rotating = False
        self._rotate_start_x = 0
        self._rotate_start_angle = 0.0
        self._rotate_center: np.ndarray | None = None  # data-space pivot
        self._rotate_base_matrix: np.ndarray | None = None  # saved transform at drag start
        self._canvas.events.mouse_press.connect(self._on_mouse_press)
        self._canvas.events.mouse_release.connect(self._on_mouse_release)

    # ── public API ──────────────────────────────────────────────────

    def load_data(self, data: PassData) -> None:
        """Load parsed pass data and render shots."""
        import time as _time
        _t0 = _time.perf_counter()

        self._data = data
        self._lines_data_set = False
        self._kdtree = None
        self._selected_idx = None
        self._box_selected_indices = np.empty(0, dtype=np.intp)
        self._face_colors = _SHOT_COLOR
        self._sel_marker.visible = False
        self._sel_lines.visible = False
        self._box_sel_markers.visible = False

        if data.count == 0:
            self._markers.set_data(np.empty((0, 2)))
            self._overlap_markers.set_data(np.empty((0, 2)))
            self._lines.set_data(np.empty((0, 2)))
            self._all_positions = None
            self._all_sizes = None
            return

        # Build Nx2 position array — centroid-shifted for GPU float32 precision.
        # Input may be float64 (after origin offset) or float32 (raw bitfields).
        # Subtract in float64 to preserve precision, then cast to float32.
        raw_pos = np.column_stack((data.x, data.y))
        self._origin = raw_pos.mean(axis=0).astype(np.float64)
        self._positions = (raw_pos - self._origin).astype(np.float32)
        _t1 = _time.perf_counter()

        # Diameter in data units (nm): FWHM = dwell_ns * _NM_PER_NS_DWELL * _fwhm_scale
        self._raw_dwells = data.dwell
        dwell_sizes = np.maximum(data.dwell * _NM_PER_NS_DWELL * self._fwhm_scale, 1.0).astype(np.float32)
        dmin_sz = dwell_sizes.min()
        dmax_sz = dwell_sizes.max()
        if dmax_sz > dmin_sz:
            self._sizes = dwell_sizes
            self._uniform_size = None  # per-point sizes
        else:
            # All dwells identical → single scalar saves N*4 bytes
            self._uniform_size = float(dmin_sz)
            self._sizes = self._uniform_size
        _t2 = _time.perf_counter()

        # Store full-resolution data; visuals always get decimated subset
        self._all_positions = self._positions
        self._all_sizes = self._sizes
        self._max_shot_size = float(np.max(self._sizes)) if not np.isscalar(self._sizes) else float(self._sizes)
        # ─── Fixed per-shot priority (computed once at load time) ────────
        # Base = uniform random → perfectly proportional decimation.
        # Dwell bias: bigger dwell → priority scaled DOWN by up to 15%
        #   (more likely to survive decimation).
        n_shots = len(self._positions)
        rng = np.random.default_rng(seed=42)
        priority = rng.random(n_shots).astype(np.float32)          # [0, 1)

        # Dwell bias: scale down priorities for bigger-dwell shots
        _DWELL_WEIGHT = 0.15                                       # max 15% reduction
        dwell_range = dmax_sz - dmin_sz
        if dwell_range > 0:
            dwell_norm = ((dwell_sizes - dmin_sz) / dwell_range).astype(np.float32)
            priority *= (1.0 - _DWELL_WEIGHT * dwell_norm)

        self._shot_priority = priority
        # Pre-sort indices by priority so the all-visible decimation path
        # can just slice instead of running argpartition every frame.
        self._priority_sorted = np.argsort(priority).astype(np.intp)
        # Cache data bounding box for density calculations and viewport cull fast-path
        dmin = self._positions.min(axis=0)
        dmax = self._positions.max(axis=0)
        self._data_width = float(dmax[0] - dmin[0])
        self._data_height = float(dmax[1] - dmin[1])
        self._data_xmin = float(dmin[0])
        self._data_xmax = float(dmax[0])
        self._data_ymin = float(dmin[1])
        self._data_ymax = float(dmax[1])

        # Compute colors for current mode
        self._face_colors = _SHOT_COLOR
        self._recompute_base_color()

        # Upload all points to GPU once — pan/zoom is a pure GPU matrix
        # transform with zero CPU cost per frame.
        n_pts = len(self._positions)
        self._upload_all_shots()
        _t3 = _time.perf_counter()

        # Lines data is DEFERRED until the user toggles them on (saves GPU upload)
        if self._lines.visible:
            self._lines.set_data(self._positions, color=_LINE_COLOR, width=1)
            self._lines_data_set = True

        # Build KD-tree in background thread so the UI stays responsive
        self._build_kdtree_async(self._positions)

        # Reposition origin marker and axis lines in centroid-shifted space
        orig_c = -self._origin.astype(np.float64)
        # Axis length: 10× the data diagonal — covers any zoom-out while
        # staying small enough for float32 GPU clip precision.
        data_diag = _math.hypot(float(np.ptp(self._positions[:, 0])),
                                float(np.ptp(self._positions[:, 1])))
        self._axis_half_len = max(data_diag * 10.0, 1e6)  # at least 1 mm
        hl = self._axis_half_len
        self._x_axis.set_data(
            np.array([[orig_c[0] - hl, orig_c[1]],
                       [orig_c[0] + hl, orig_c[1]]], dtype=np.float64))
        self._y_axis.set_data(
            np.array([[orig_c[0], orig_c[1] - hl],
                       [orig_c[0], orig_c[1] + hl]], dtype=np.float64))
        self._origin_marker.set_data(
            orig_c.reshape(1, 2),
            size=8, face_color=(1, 1, 1, 0.9), edge_width=0,
        )
        _t4 = _time.perf_counter()

        # Reset rotation
        self._rotation_deg = 0.0
        self._visual_root.transform = scene.transforms.MatrixTransform()
        self._last_cam_sig = None  # force arrow/axis reposition after origin change
        self._fit_view()
        _t5 = _time.perf_counter()
        print(f"[load] positions: {(_t1-_t0)*1000:.0f}ms  sizes: {(_t2-_t1)*1000:.0f}ms  "
              f"upload: {(_t3-_t2)*1000:.0f}ms  axes: {(_t4-_t3)*1000:.0f}ms  "
              f"fit: {(_t5-_t4)*1000:.0f}ms  total: {(_t5-_t0)*1000:.0f}ms  "
              f"({n_pts:,} pts)")

        # Reposition wafer outline if active (centroid changed)
        if self._wafer_diameter_nm is not None:
            self.set_wafer_outline(self._wafer_diameter_nm)

    def set_stripe_regions(self, regions: list[dict]) -> None:
        """Set stripe region metadata for hover-activated rectangle display."""
        self._stripe_regions = regions
        self._hovered_stripe_idx = None
        self._stripe_tooltip.setVisible(False)

        # Precompute centroid-shifted AABBs
        self._stripe_aabbs = []
        for r in regions:
            x0 = r["originX"] - self._origin[0]
            y0 = r["originY"] - self._origin[1]
            x1 = x0 + r["width"]
            y1 = y0 + r["length"]
            self._stripe_aabbs.append((x0, y0, x1, y1))

        # Create or reset the single reusable rectangle visual
        if self._stripe_rect is not None:
            self._stripe_rect.parent = None
            self._stripe_rect = None
        if regions:
            self._stripe_rect = visuals.Rectangle(
                center=(0, 0), width=1, height=1,
                border_color=(1, 1, 0.5, 0.8),
                color=(0, 0, 0, 0),
                border_width=1,
                parent=self._visual_root,
            )
            self._stripe_rect.visible = False

    def _position_stripe_tooltip(self) -> None:
        """Position the stripe tooltip near the rectangle or in the lower-right."""
        if self._hovered_stripe_idx is None:
            return
        aabb = self._stripe_aabbs[self._hovered_stripe_idx]
        # Try the top-right corner of the rectangle
        tr_corner = np.array([aabb[2], aabb[3]], dtype=np.float64)
        screen = self._data_to_canvas(tr_corner)
        cw = self._canvas.native.width()
        ch = self._canvas.native.height()
        tw = self._stripe_tooltip.width()
        th = self._stripe_tooltip.height()
        pad = 10
        if (screen is not None
                and 0 <= screen[0] + pad + tw <= cw
                and 0 <= screen[1] - th - pad <= ch):
            self._stripe_tooltip.move(int(screen[0]) + pad, int(screen[1]) - th - pad)
        else:
            # Fallback: lower-right corner of the canvas
            self._stripe_tooltip.move(cw - tw - pad, ch - th - pad)

    def set_wafer_outline(self, diameter_nm: float | None) -> None:
        """Show or hide a wafer outline circle of the given diameter (nm)."""
        self._wafer_diameter_nm = diameter_nm
        if diameter_nm is None:
            self._wafer_outline.visible = False
            return
        radius = diameter_nm / 2.0
        center = -self._origin
        pts = _UNIT_CIRCLE * radius + center
        self._wafer_outline.set_data(pts.astype(np.float64))
        self._wafer_outline.visible = True

    def _build_kdtree_async(self, positions: np.ndarray) -> None:
        """Build the KD-tree on a worker thread."""
        # Clean up any previous thread
        if self._kdtree_thread is not None:
            self._kdtree_thread.quit()
            self._kdtree_thread.wait()
        self._kdtree = None
        thread = QThread(self)  # parent prevents premature GC
        worker = _KDTreeWorker(positions)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_kdtree_ready)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_kdtree_thread_done)
        # prevent GC
        self._kdtree_thread = thread
        self._kdtree_worker = worker
        thread.start()

    def _on_kdtree_ready(self, tree: cKDTree) -> None:
        """Called when background KD-tree build completes."""
        self._kdtree = tree
        self.kdtree_ready.emit()

    def _on_kdtree_thread_done(self) -> None:
        """Called when the KD-tree thread has fully stopped."""
        self._kdtree_thread = None
        self._kdtree_worker = None

    def set_lines_visible(self, visible: bool) -> None:
        """Toggle shot connection lines."""
        if visible and self._positions is not None:
            if not self._lines_data_set:
                self._lines.set_data(self._positions, color=_LINE_COLOR, width=1)
                self._lines_data_set = True
        self._lines.visible = visible
        self._update_sel_lines()
        self._canvas.update()

    @property
    def lines_visible(self) -> bool:
        return self._lines.visible

    def set_shot_color(self, rgba: tuple[float, float, float, float]) -> None:
        """Change shot colour and refresh the display."""
        global _SHOT_COLOR
        _SHOT_COLOR = np.array(rgba, dtype=np.float32)
        self._face_colors = _SHOT_COLOR
        self._recompute_base_color()
        if self._all_positions is not None:
            self._upload_all_shots()
            self._canvas.update()

    @property
    def shot_color(self) -> tuple[float, float, float, float]:
        return tuple(_SHOT_COLOR.tolist())

    def _display_size(self, base_sz):
        """Scale base size for selection markers to match data markers visually."""
        if self._marker_mode == 'disc':
            return base_sz * _DISC_SIZE_SCALE
        else:
            # Gaussian selection markers use the same FWHM as data markers
            return base_sz

    def _sel_size(self, base_sz):
        """Return selection marker size: floor-clamped + inflated + min screen px."""
        dpp = self._get_data_per_px()
        if dpp is not None and dpp > 0:
            if self._marker_mode == 'gaussian':
                min_size = dpp * 4.0
            else:
                min_size = dpp
            if np.isscalar(base_sz):
                sz = max(base_sz, min_size)
            else:
                sz = np.maximum(base_sz, min_size)
        else:
            sz = base_sz
        # Apply same stride inflation as the data layer
        stride = self._decim_stride
        amp = self._disc_inflate_amp if self._marker_mode == 'disc' else self._stride_inflate_amp
        if stride > 1 and amp > 0.0:
            ls = _math.log10(stride)
            stride_scale = 1.0 + amp * ls / (ls + 0.2)
            if np.isscalar(sz):
                sz = sz * stride_scale
            else:
                sz = sz * stride_scale
        sz = self._display_size(sz) * 1.05
        # Ensure at least _MIN_SEL_PX screen pixels
        if dpp is not None and dpp > 0:
            min_data = _MIN_SEL_PX * dpp
            if np.isscalar(sz):
                return max(sz, min_data)
            else:
                return np.maximum(sz, min_data)
        return sz

    def _box_sel_size(self, base_sz):
        """Return box-selection marker size with same floor clamp + inflation as data."""
        dpp = self._get_data_per_px()
        if dpp is not None and dpp > 0:
            if self._marker_mode == 'gaussian':
                min_size = dpp * 4.0
            else:
                min_size = dpp
            if np.isscalar(base_sz):
                sz = max(base_sz, min_size)
            else:
                sz = np.maximum(base_sz, min_size)
        else:
            sz = base_sz
        # Apply same stride inflation as the data layer
        stride = self._decim_stride
        amp = self._disc_inflate_amp if self._marker_mode == 'disc' else self._stride_inflate_amp
        if stride > 1 and amp > 0.0:
            ls = _math.log10(stride)
            stride_scale = 1.0 + amp * ls / (ls + 0.2)
            if np.isscalar(sz):
                sz = sz * stride_scale
            else:
                sz = sz * stride_scale
        return self._display_size(sz) * 1.05

    def set_selected_color(self, rgba: tuple[float, float, float, float]) -> None:
        """Change single-click selection colour."""
        global _SELECTED_COLOR
        _SELECTED_COLOR = np.array(rgba, dtype=np.float32)
        if self._sel_marker.visible and self._selected_idx is not None:
            idx = self._selected_idx
            sel_sz = self._uniform_size if self._uniform_size is not None else self._sizes[idx:idx+1]
            self._sel_marker.set_data(
                self._positions[idx:idx+1], size=self._sel_size(sel_sz),
                face_color=_SELECTED_COLOR, edge_width=0)
            self._canvas.update()

    @property
    def selected_color(self) -> tuple[float, float, float, float]:
        return tuple(_SELECTED_COLOR.tolist())

    def set_box_selected_color(self, rgba: tuple[float, float, float, float]) -> None:
        """Change box-selection highlight colour."""
        global _BOX_SELECTED_COLOR
        _BOX_SELECTED_COLOR = np.array(rgba, dtype=np.float32)
        if self._box_sel_markers.visible and len(self._box_selected_indices):
            self._upload_box_sel_markers()
            self._canvas.update()

    @property
    def box_selected_color(self) -> tuple[float, float, float, float]:
        return tuple(_BOX_SELECTED_COLOR.tolist())

    def set_line_color(self, rgba: tuple[float, float, float, float]) -> None:
        """Change connection-line colour."""
        global _LINE_COLOR
        _LINE_COLOR = rgba
        if self._lines.visible and self._positions is not None:
            self._lines.set_data(self._positions, color=_LINE_COLOR, width=1)
            self._canvas.update()

    @property
    def line_color(self) -> tuple[float, float, float, float]:
        c = _LINE_COLOR
        return (c[0], c[1], c[2], c[3])

    def set_sel_line_color(self, rgba: tuple[float, float, float, float]) -> None:
        """Change selected-shot connection-line colour."""
        global _SEL_LINE_COLOR
        _SEL_LINE_COLOR = rgba
        if self._sel_lines.visible:
            self._update_sel_lines()
            self._canvas.update()

    @property
    def sel_line_color(self) -> tuple[float, float, float, float]:
        c = _SEL_LINE_COLOR
        return (c[0], c[1], c[2], c[3])

    def reset_view(self) -> None:
        """Reset camera and rotation to show all data."""
        self._rotation_deg = 0.0
        self._visual_root.transform = scene.transforms.MatrixTransform()
        self._fit_view()

    # ── Marker mode (Gaussian / Disc) ─────────────────────────────────

    def _recompute_base_color(self) -> None:
        """Set self._base_color based on marker mode and current shot color."""
        if self._marker_mode == 'disc':
            rgb = np.clip(_SHOT_COLOR[:3] - self._disc_overlap_white, 0, 1)
            self._base_color = np.array([*rgb, 1.0], dtype=np.float32)
        else:
            self._base_color = np.array([*_SHOT_COLOR[:3], 1.0], dtype=np.float32)

    def _apply_marker_mode(self) -> None:
        """Toggle visibility of Gaussian vs. disc marker layers."""
        gauss = self._marker_mode == 'gaussian'
        self._gauss_markers.visible = gauss
        self._disc_base_markers.visible = not gauss
        self._disc_overlap_markers.visible = not gauss

        # Swap selection marker aliases
        # Hide old selection markers, point aliases to new ones
        old_sel_vis = self._sel_marker.visible
        old_box_vis = self._box_sel_markers.visible
        self._sel_marker.visible = False
        self._box_sel_markers.visible = False

        if gauss:
            self._sel_marker = self._gauss_sel_marker
            self._box_sel_markers = self._gauss_box_sel_markers
            self._disc_sel_marker.visible = False
            self._disc_box_sel_markers.visible = False
        else:
            self._sel_marker = self._disc_sel_marker
            self._box_sel_markers = self._disc_box_sel_markers
            self._gauss_sel_marker.visible = False
            self._gauss_box_sel_markers.visible = False

        # Restore visibility and re-upload selection data if needed
        if old_sel_vis and self._selected_idx is not None:
            idx = self._selected_idx
            sel_sz = self._uniform_size if self._uniform_size is not None else self._sizes[idx:idx+1]
            self._sel_marker.set_data(
                self._positions[idx:idx+1], size=self._sel_size(sel_sz),
                face_color=_SELECTED_COLOR, edge_width=0)
            self._sel_marker.visible = True
        if old_box_vis and len(self._box_selected_indices) > 0:
            self._upload_box_sel_markers()

    @property
    def marker_mode(self) -> str:
        """Current marker rendering mode: 'gaussian' or 'disc'."""
        return self._marker_mode

    def set_marker_mode(self, mode: str) -> None:
        """Switch between Gaussian PSF and hard-disc marker rendering.

        mode: 'gaussian' or 'disc'
        """
        if mode not in ('gaussian', 'disc'):
            raise ValueError(f"Unknown marker mode: {mode!r}")
        if mode == self._marker_mode:
            return
        self._marker_mode = mode
        self._recompute_base_color()
        self._apply_marker_mode()
        # Force full re-upload with new rendering path
        self._last_view_key = None
        if self._all_positions is not None:
            self._update_decim_stride()
        self._canvas.update()

    # ── FWHM scale ──────────────────────────────────────────────────

    @property
    def fwhm_scale(self) -> float:
        """Current FWHM multiplier (1.0 = default dwell→nm conversion)."""
        return self._fwhm_scale

    @property
    def fwhm_nm_per_ns(self) -> float:
        """Effective FWHM in nm per ns of dwell time."""
        return _NM_PER_NS_DWELL * self._fwhm_scale

    def set_fwhm_scale(self, scale: float) -> None:
        """Change the FWHM scale factor and re-upload shot sizes.

        scale: multiplier on the base conversion (_NM_PER_NS_DWELL).
               1.0 = default (10 nm/µs), 2.0 = double size, etc.
        """
        if self._raw_dwells is None:
            return
        self._fwhm_scale = scale
        dwell_sizes = np.maximum(
            self._raw_dwells * _NM_PER_NS_DWELL * scale, 1.0
        ).astype(np.float32)
        dmin_sz = dwell_sizes.min()
        dmax_sz = dwell_sizes.max()
        if dmax_sz > dmin_sz:
            self._sizes = dwell_sizes
            self._uniform_size = None
        else:
            self._uniform_size = float(dmin_sz)
            self._sizes = self._uniform_size
        self._all_sizes = self._sizes
        # Force re-upload with current viewport
        self._last_view_key = None
        self._update_decim_stride()
        self._canvas.update()

    def set_stride_inflate_amp(self, amp: float) -> None:
        """Set the stride inflation amplitude and re-render."""
        self._stride_inflate_amp = amp
        self._last_view_key = None
        self._update_decim_stride()
        self._canvas.update()

    def set_alpha_far(self, value: float) -> None:
        """Set the far-zoom alpha floor and re-render."""
        self._alpha_far = max(0.001, min(value, 1.0))
        self._last_view_key = None
        self._update_decim_stride()
        self._canvas.update()

    def set_alpha_comp_power(self, power: float) -> None:
        """Set the alpha compensation power and re-render."""
        self._alpha_comp_power = power
        self._last_view_key = None
        self._update_decim_stride()
        self._canvas.update()

    def set_alpha_dref(self, value: float) -> None:
        """Set sigmoid midpoint DPP (d_ref) and re-render.

        alpha = alpha_far + (alpha_max - alpha_far) / (1 + (dpp/d_ref)^p)
        """
        self._alpha_dref = max(0.1, value)
        self._last_view_key = None
        self._update_decim_stride()
        self._canvas.update()

    def set_alpha_max(self, value: float) -> None:
        """Set the close-zoom alpha cap and re-render."""
        self._alpha_max = max(0.001, min(value, 1.0))
        self._last_view_key = None
        self._update_decim_stride()
        self._canvas.update()

    def set_disc_overlap_white(self, value: float) -> None:
        """Set the disc-mode base white overlay alpha and re-render."""
        self._disc_overlap_white = max(0.001, min(value, 1.0))
        self._recompute_base_color()
        self._last_view_key = None
        self._update_decim_stride()
        self._canvas.update()

    def set_disc_dref(self, value: float) -> None:
        """Set the disc-mode dpp_low (opaque threshold) and re-render."""
        self._disc_dpp_low = max(0.01, value)
        self._last_view_key = None
        self._update_decim_stride()
        self._canvas.update()

    def set_disc_dpp_high(self, value: float) -> None:
        """Set the disc-mode dpp_high (transparent threshold) and re-render."""
        self._disc_dpp_high = max(0.01, value)
        self._last_view_key = None
        self._update_decim_stride()
        self._canvas.update()

    def set_disc_dpp_mid(self, value: float) -> None:
        """Set the disc-mode dpp_mid (intermediate point DPP) and re-render."""
        self._disc_dpp_mid = max(0.01, value)
        self._last_view_key = None
        self._update_decim_stride()
        self._canvas.update()

    def set_disc_f_mid(self, value: float) -> None:
        """Set the disc-mode f_mid (alpha at dpp_mid) and re-render."""
        self._disc_f_mid = max(0.0, min(1.0, value))
        self._last_view_key = None
        self._update_decim_stride()
        self._canvas.update()

    def set_disc_inflate_amp(self, amp: float) -> None:
        """Set the disc-mode stride inflation amplitude and re-render."""
        self._disc_inflate_amp = max(0.0, amp)
        self._last_view_key = None
        self._update_decim_stride()
        self._canvas.update()

    def set_disc_antialias(self, value: float) -> None:
        """Set the disc edge softness (antialias width in pixels) and re-render."""
        value = max(0.0, value)
        self._disc_base_markers.antialias = value
        self._disc_overlap_markers.antialias = value
        self._canvas.update()

    # ── line decimation ─────────────────────────────────────────────

    def _get_data_per_px(self) -> float | None:
        """Return the current data-units-per-pixel scale, or None."""
        try:
            tr = self._canvas.scene.node_transform(self._visual_root)
            p0 = tr.map([0, 0, 0, 1])
            p1 = tr.map([1, 0, 0, 1])
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            return _math.sqrt(dx * dx + dy * dy)
        except Exception:
            return None

    # ── shot decimation ─────────────────────────────────────────────

    def _priority_indices(self, vis_idx: np.ndarray | None,
                          n_vis: int, stride: float) -> np.ndarray:
        """Return ~n_vis/stride indices chosen by fixed per-shot priority.

        *vis_idx* is an index array into the full position array, or
        ``None`` when every point is visible (avoids allocating a huge
        arange).  *n_vis* is the visible count.
        """
        if stride <= 1.0:
            if vis_idx is None:
                return np.arange(n_vis, dtype=np.intp)
            return vis_idx
        count = max(1, int(n_vis / stride))
        if vis_idx is None:
            # All points visible — slice the pre-sorted priority order (O(1))
            return self._priority_sorted[:count]
        priorities = self._shot_priority[vis_idx]
        top_k = np.argpartition(priorities, count)[:count]
        return vis_idx[top_k]

    def _upload_all_shots(self) -> None:
        """Upload shot data to active marker visuals (initial load, strided)."""
        pos = self._all_positions
        if pos is None or len(pos) == 0:
            self._gauss_markers.set_data(np.empty((0, 2)))
            self._disc_base_markers.set_data(np.empty((0, 2)))
            self._disc_overlap_markers.set_data(np.empty((0, 2)))
            return

        n = len(pos)

        # Initial stride: camera isn't ready yet, use canvas pixel count as budget estimate
        canvas_px = max(self._canvas.native.width() * self._canvas.native.height(), 1)
        if self._marker_mode == 'gaussian':
            spp, min_b, max_r = _GAUSS_SHOTS_PER_PX, _GAUSS_MIN_BUDGET, _GAUSS_MAX_RENDERED
        else:
            spp, min_b, max_r = _DISC_SHOTS_PER_PX, _DISC_MIN_BUDGET, _DISC_MAX_RENDERED
        initial_budget = min(max(min_b, int(canvas_px * spp)), max_r)
        stride = max(1.0, float(round(n / initial_budget)))
        self._decim_stride = stride
        idx = self._priority_indices(None, n, stride)
        dpos = pos[idx]
        self._uploaded_positions = dpos
        dsizes = self._uniform_size if self._uniform_size is not None else self._all_sizes[idx]
        self._uploaded_sizes = dsizes
        self._upload_view(dpos, dsizes, stride=stride)
        self._shot_count_label.setText(f"{len(dpos):,} / {n:,} shots")
        self._shot_count_label.adjustSize()
        self._shot_count_label.setVisible(True)

    def _upload_view(self, dpos: np.ndarray, dsizes,
                     dpp: float = 0.0, stride: float = 1.0) -> None:
        """Upload a (possibly filtered/strided) point subset to GPU.

        dpp: data units (nm) per screen pixel — the true zoom scale.
        stride: decimation stride (>=1). Logged for diagnostics.
        """

        # Ensure each marker is at least a few pixels wide in data units.
        # Gaussians need ~4px to show their smooth falloff; discs need 1px.
        if self._marker_mode == 'gaussian':
            min_size = dpp * 4.0
        else:
            min_size = dpp
        if np.isscalar(dsizes):
            base_sizes = max(dsizes, min_size)
        else:
            base_sizes = np.maximum(dsizes, min_size)

        # Inflate AFTER floor clamp so the scaling is always visible.
        # scale = 1 + A * log10(stride) / (log10(stride) + 0.2)
        # Saturates near 1+A at high stride; exactly 1.0 at stride=1.
        # Use mode-specific amplitude so disc and gaussian can be tuned independently.
        if self._marker_mode == 'gaussian':
            amp = self._stride_inflate_amp
        else:
            amp = self._disc_inflate_amp
        if stride > 1.0 and amp > 0.0:
            ls = _math.log10(stride)
            stride_scale = 1.0 + amp * ls / (ls + 0.2)
            if np.isscalar(base_sizes):
                base_sizes = base_sizes * stride_scale
            else:
                base_sizes = base_sizes * stride_scale

        if self._marker_mode == 'gaussian':
            # Alpha based on rendered size, not raw zoom.
            # effective_dpp = rendered_size / 4  (the dpp at which that size
            # would just hit the 4-px floor clamp).
            # When floor-clamped: effective_dpp = dpp  (alpha tracks zoom)
            # When natural size:  effective_dpp = size/4  (alpha freezes)
            # ── Sigmoid alpha curve ──
            # α = α_far + (α_max − α_far) / (1 + (dpp/d_ref)^p)
            # Smooth S-curve: α_max when zoomed in, α_far when zoomed out,
            # with d_ref controlling where the midpoint is.
            if np.isscalar(dsizes):
                natural_dpp = float(dsizes) / 4.0
            else:
                natural_dpp = float(np.median(dsizes)) / 4.0
            effective_dpp = max(dpp, natural_dpp)

            d_ref = self._alpha_dref
            p = self._alpha_p
            alpha = self._alpha_far + (self._alpha_max - self._alpha_far) / (1.0 + (effective_dpp / d_ref) ** p)
            # Compensate for stride size inflation
            if stride > 1.0:
                ls = _math.log10(stride)
                stride_scale_alpha = 1.0 + self._stride_inflate_amp * ls / (ls + 0.2)
                alpha = max(self._alpha_far, alpha / (stride_scale_alpha ** self._alpha_comp_power))
            self._last_overlap_alpha = alpha
            print(f"[gauss] dpp={dpp:.2f} eff_dpp={effective_dpp:.2f} alpha={alpha:.5f} stride={stride:.2f} d_ref={d_ref:.1f} p={p:.1f}")
            face_color = np.array([*self._base_color[:3], alpha], dtype=np.float32)
            self._gauss_markers.set_data(
                dpos, size=base_sizes,
                face_color=face_color, edge_color=face_color,
                edge_width=0,
            )
        else:
            # Disc mode: two regimes
            # 1) Natural size (zoomed in, dpp < shot_size): base α=1,
            #    overlay α=ow.  Non-overlapping = shot_color, overlapping
            #    = brighter by ow per extra disc.
            # 2) Floor-clamped (zoomed out, dpp ≥ shot_size): base α < 1
            #    via sigmoid, overlay α = ow * same factor.  Density shows
            #    through accumulation; single-disc cancellation still holds
            #    because both layers scale by the same factor f:
            #      (shot_color - ow)*f + bg*(1-f) + ow*f = shot_color*f + bg*(1-f)
            disc_sizes = base_sizes * _DISC_SIZE_SCALE if not np.isscalar(base_sizes) else base_sizes * _DISC_SIZE_SCALE

            ow = self._disc_overlap_white

            # Piecewise log-linear ramp with intermediate point:
            # f=1 when dpp ≤ lo, f=f_mid at dpp_mid, f=0 when dpp ≥ hi.
            # Linear in log-space between each pair.
            dpp_lo = self._disc_dpp_low
            dpp_mid = self._disc_dpp_mid
            dpp_hi = self._disc_dpp_high
            f_mid = self._disc_f_mid
            if dpp <= dpp_lo:
                f = 1.0
            elif dpp >= dpp_hi:
                f = 0.0
            elif dpp_mid <= dpp_lo or dpp_mid >= dpp_hi:
                # mid out of range — fall back to simple lo→hi
                f = 1.0 - (_math.log(dpp) - _math.log(dpp_lo)) / (_math.log(dpp_hi) - _math.log(dpp_lo))
            elif dpp <= dpp_mid:
                t = (_math.log(dpp) - _math.log(dpp_lo)) / (_math.log(dpp_mid) - _math.log(dpp_lo))
                f = 1.0 + (f_mid - 1.0) * t
            else:
                t = (_math.log(dpp) - _math.log(dpp_mid)) / (_math.log(dpp_hi) - _math.log(dpp_mid))
                f = f_mid * (1.0 - t)
            print(f"[disc] dpp={dpp:.2f} lo={dpp_lo:.1f} mid={dpp_mid:.1f} hi={dpp_hi:.1f} f_mid={f_mid:.2f} f={f:.5f} ow={ow:.3f}")

            self._last_overlap_alpha = ow * f

            base_fc = np.array([*self._base_color[:3], f], dtype=np.float32)
            self._disc_base_markers.set_data(
                dpos, size=disc_sizes,
                face_color=base_fc, edge_color=base_fc,
                edge_width=0,
            )
            overlap_color = np.array([1.0, 1.0, 1.0, ow * f], dtype=np.float32)
            self._disc_overlap_markers.set_data(
                dpos, size=disc_sizes,
                face_color=overlap_color, edge_color=overlap_color,
                edge_width=0,
            )

    def _update_decim_stride(self) -> None:
        """Viewport cull + zoom-adaptive stride decimation.

        1. Filter to shots inside the viewport (with margin).
        2. If the visible count exceeds a budget, stride the visible set.
        stride = 1 + ((dpp - stride1_dpp) / stride1_dpp) ^ exponent
        """
        if self._all_positions is None:
            return
        pos = self._all_positions
        n = len(pos)
        if n == 0:
            return

        bounds = self._get_viewport_bounds()
        if bounds is None:
            return
        xmin, xmax, ymin, ymax = bounds

        # Add margin: 5% of viewport + half the largest rendered disc diameter,
        # so shots whose centers are just off-screen but whose discs extend
        # into view aren't culled.  The base layer inflates to at least dpp,
        # so use max(max_size, dpp) for the radius term.
        canvas_px = max(self._canvas.native.width(), 1)
        dpp = self._get_data_per_px() or (float(self._camera.rect.width) / canvas_px)

        if self._uniform_size is not None:
            max_shot_size = self._uniform_size
        else:
            max_shot_size = getattr(self, '_max_shot_size', 1.0)
        half_disc = max(max_shot_size, dpp) * 0.5

        mx = (xmax - xmin) * 0.05 + half_disc
        my = (ymax - ymin) * 0.05 + half_disc

        # Fast path: skip per-point viewport cull if all data fits in viewport.
        # Comparing 4 floats instead of 4×N numpy arrays.
        if (hasattr(self, '_data_xmin')
                and self._data_xmin >= xmin - mx and self._data_xmax <= xmax + mx
                and self._data_ymin >= ymin - my and self._data_ymax <= ymax + my):
            vis_idx = None  # sentinel: all points visible
            n_vis = n
        else:
            vis_mask = ((pos[:, 0] >= xmin - mx) & (pos[:, 0] <= xmax + mx) &
                        (pos[:, 1] >= ymin - my) & (pos[:, 1] <= ymax + my))
            vis_idx = np.nonzero(vis_mask)[0]
            n_vis = len(vis_idx)

        # Stride: budget scales with how many screen pixels the data covers.
        # Per axis, use min(data_extent, viewport_extent) so that a narrow
        # pattern zoomed out doesn't get a viewport-sized budget.
        data_w = getattr(self, '_data_width', xmax - xmin)
        data_h = getattr(self, '_data_height', ymax - ymin)
        vp_w = xmax - xmin
        vp_h = ymax - ymin
        vis_data_w = min(data_w, vp_w) / dpp          # screen px the data fills, X
        vis_data_h = min(data_h, vp_h) / dpp          # screen px the data fills, Y
        data_screen_px = max(vis_data_w * vis_data_h, 1.0)
        if self._marker_mode == 'gaussian':
            spp, min_b, max_r = _GAUSS_SHOTS_PER_PX, _GAUSS_MIN_BUDGET, _GAUSS_MAX_RENDERED
        else:
            spp, min_b, max_r = _DISC_SHOTS_PER_PX, _DISC_MIN_BUDGET, _DISC_MAX_RENDERED
        budget = min(max(min_b, int(data_screen_px * spp)), max_r)
        stride = max(1.0, n_vis / budget)
        self._decim_stride = stride
        # Quantise stride for cache key only — actual stride is continuous
        stride_quant = round(stride * 5) / 5      # 0.2 steps for cache key
        print(f"[stride] n_vis={n_vis} stride={stride:.3f} budget={budget} min_b={min_b} data_px={data_screen_px:.0f} dpp={dpp:.2f} mode={self._marker_mode}")

        # Build a cache key from viewport quantised bounds + stride + quantised dpp
        # Log-quantise dpp so the key changes ~every 5% zoom step
        dpp_quant = round(_math.log2(max(dpp, 0.001)) * 20)
        key = (round(xmin / max(mx, 1)), round(xmax / max(mx, 1)),
               round(ymin / max(my, 1)), round(ymax / max(my, 1)), stride_quant, dpp_quant)

        if key != getattr(self, '_last_view_key', None):
            self._last_view_key = key
            idx = self._priority_indices(vis_idx, n_vis, stride)
            dpos = pos[idx]
            if len(dpos) > 0:
                print(f"[upload] rendered={len(idx)} pos_range=({dpos[:,0].min():.0f}..{dpos[:,0].max():.0f}, {dpos[:,1].min():.0f}..{dpos[:,1].max():.0f})")
            else:
                print(f"[upload] rendered=0")
            self._uploaded_positions = dpos
            if self._uniform_size is not None:
                dsizes = self._uniform_size
            else:
                dsizes = self._all_sizes[idx]
            self._uploaded_sizes = dsizes
            self._upload_view(dpos, dsizes, dpp, stride)

        # Always update status label
        rendered = len(getattr(self, '_uploaded_positions', []))
        alpha = getattr(self, '_last_overlap_alpha', 0.0)
        parts = [f"{n:,} total"]
        parts.append(f"{n_vis:,} visible")
        if stride > 1.0:
            parts.append(f"stride {stride:.1f}")
        parts.append(f"{rendered:,} rendered")
        parts.append(f"dpp {dpp:.2f}")
        parts.append(f"a {alpha:.3f}")
        self._shot_count_label.setText("  |  ".join(parts))
        self._shot_count_label.adjustSize()
        self._shot_count_label.setVisible(True)

    def _get_viewport_bounds(self) -> tuple[float, float, float, float] | None:
        """Return (xmin, xmax, ymin, ymax) in data coords for the visible area."""
        try:
            tr = self._canvas.scene.node_transform(self._visual_root)
            w, h = self._canvas.native.width(), self._canvas.native.height()
            # Map all 4 viewport corners so bounds stay correct under rotation
            corners = [tr.map([0, 0, 0, 1]), tr.map([w, 0, 0, 1]),
                       tr.map([0, h, 0, 1]), tr.map([w, h, 0, 1])]
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            return min(xs), max(xs), min(ys), max(ys)
        except Exception:
            return None

    # ── private helpers ─────────────────────────────────────────────

    def _update_sel_lines(self) -> None:
        """Show/hide thick gold lines to/from selected or box-selected shots."""
        if self._positions is None or not self._lines.visible:
            self._sel_lines.visible = False
            return

        n = len(self._positions)

        # Collect all selected indices (single + box)
        # Skip connection lines for huge box selections (would draw millions of segments)
        if len(self._box_selected_indices) > 10_000:
            self._sel_lines.visible = False
            return

        parts_list: list[np.ndarray] = []
        if self._selected_idx is not None:
            parts_list.append(np.array([self._selected_idx], dtype=np.intp))
        if len(self._box_selected_indices):
            parts_list.append(np.asarray(self._box_selected_indices, dtype=np.intp))

        if not parts_list:
            self._sel_lines.visible = False
            return

        idx_arr = np.unique(np.concatenate(parts_list))

        # Predecessor segments (idx-1 → idx) for idx > 0
        prev_mask = idx_arr > 0
        prev_idx = idx_arr[prev_mask]
        # Successor segments (idx → idx+1) for idx < n-1
        next_mask = idx_arr < (n - 1)
        next_idx = idx_arr[next_mask]

        # Build interleaved start/end pairs
        parts = []
        if len(prev_idx):
            seg = np.empty((len(prev_idx) * 2, 2), dtype=np.float32)
            seg[0::2] = self._positions[prev_idx - 1]
            seg[1::2] = self._positions[prev_idx]
            parts.append(seg)
        if len(next_idx):
            seg = np.empty((len(next_idx) * 2, 2), dtype=np.float32)
            seg[0::2] = self._positions[next_idx]
            seg[1::2] = self._positions[next_idx + 1]
            parts.append(seg)

        if parts:
            self._sel_lines.set_data(
                np.concatenate(parts),
                color=_SEL_LINE_COLOR,
                width=3,
                connect='segments',
            )
            self._sel_lines.visible = True
        else:
            self._sel_lines.visible = False

    def _reposition_coord_label(self, event=None) -> None:
        """Keep the coordinate label anchored to the bottom-left of the canvas."""
        canvas_h = self._canvas.native.height()
        self._coord_label.move(4, canvas_h - self._coord_label.height() - 4)

    def _camera_signature(self) -> tuple:
        """Cheap fingerprint of the current camera state (rect + rotation + canvas size)."""
        r = self._camera.rect
        return (float(r.left), float(r.right), float(r.bottom), float(r.top),
                self._rotation_deg,
                self._canvas.native.width(), self._canvas.native.height())

    def _on_camera_change(self, event=None) -> None:
        """Reposition the pinned tooltip and axis labels when the camera changes."""
        # Skip if the camera hasn't actually moved (breaks the async loop
        # where _arrow_overlay.update() queues a canvas redraw that re-fires
        # the draw event).
        sig = self._camera_signature()
        if sig == getattr(self, '_last_cam_sig', None):
            return
        self._last_cam_sig = sig
        self.__on_camera_change_inner()

    def __on_camera_change_inner(self) -> None:
        # Tooltip tracking
        if self._selected_idx is not None and self._positions is not None:
            screen_pos = self._data_to_canvas(self._positions[self._selected_idx])
            if screen_pos is not None:
                self._tooltip.move(int(screen_pos[0]) + 20, int(screen_pos[1]) - self._tooltip.height() - 10)

        # Reposition stripe tooltip so it stays anchored during pan/zoom
        if self._hovered_stripe_idx is not None:
            self._position_stripe_tooltip()

        # Throttled shot stride update on zoom
        if self._all_positions is not None:
            self._shot_decim_timer.start()

        # Reposition ruler label
        if self._ruler_start is not None:
            self._update_ruler_visual()

        # Refresh selection marker sizes so they stay visible at any zoom
        if self._sel_marker.visible and self._selected_idx is not None:
            idx = self._selected_idx
            sel_sz = self._uniform_size if self._uniform_size is not None else self._sizes[idx:idx+1]
            self._sel_marker.set_data(
                self._positions[idx:idx+1], size=self._sel_size(sel_sz),
                face_color=_SELECTED_COLOR, edge_width=0)
        if self._box_sel_markers.visible and len(self._box_selected_indices):
            self._upload_box_sel_markers()

        # Axis labels at canvas edges where axis lines meet the viewport edge
        self._reposition_axis_labels()

    def _reposition_axis_labels(self) -> None:
        """Recompute axis screen positions and repaint the arrow overlay."""
        cw, ch = self._canvas.native.width(), self._canvas.native.height()
        if cw == 0 or ch == 0:
            return

        self._arrow_overlay.resize(cw, ch)

        # Map the data-space origin and axis unit vectors to screen,
        # accounting for both camera and the visual_root rotation.
        try:
            tr = self._canvas.scene.node_transform(self._visual_root)

            # Direction vectors: use a step proportional to the viewport size
            # so the two sample points are always far apart in screen space,
            # even at extreme zoom-out where 1 data-unit → sub-pixel.
            step = max(float(self._camera.rect.width),
                       float(self._camera.rect.height), 1.0) * 0.1
            o0 = tr.imap([0.0, 0.0, 0, 1])
            px0 = tr.imap([step, 0.0, 0, 1])
            py0 = tr.imap([0.0, step, 0, 1])

            # Screen position of the real data-space origin (for ray start)
            orig_c = -self._origin.astype(np.float64)
            o_scr = tr.imap([float(orig_c[0]), float(orig_c[1]), 0, 1])
            ox, oy = float(o_scr[0]), float(o_scr[1])
        except Exception as exc:
            print(f"[axis] exception: {exc}")
            self._arrow_overlay.arrows = []
            self._arrow_overlay.update()
            return

        def _unit(ax, ay):
            length = _math.hypot(ax, ay)
            if length < 1e-12:
                return 0.0, 0.0
            return ax / length, ay / length

        # Screen-space direction vectors (from near-origin samples)
        xdx, xdy = _unit(float(px0[0]) - float(o0[0]), float(px0[1]) - float(o0[1]))
        ydx, ydy = _unit(float(py0[0]) - float(o0[0]), float(py0[1]) - float(o0[1]))

        # Each axis is an infinite line through the origin. Find both
        # intersection points with the viewport, then assign the +dir
        # arrow to the far point and the -dir arrow to the near point.
        arrows = []
        margin = 6
        for dx, dy, pos_label, neg_label in [
            (xdx, xdy, "X", "−X"),
            (ydx, ydy, "Y", "−Y"),
        ]:
            if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                continue
            hit = self._arrow_overlay._line_rect_intersections(
                ox, oy, dx, dy, cw, ch, margin)
            if hit is not None:
                (near_x, near_y), (far_x, far_y) = hit
                arrows.append((far_x, far_y, dx, dy, pos_label))
                arrows.append((near_x, near_y, -dx, -dy, neg_label))
            else:
                # Line doesn't cross viewport — project far along each
                # direction from the origin and clamp to viewport edge
                # so +/- arrows land on different edges.
                far = max(cw, ch) * 2.0
                for sign, lbl in [(1, pos_label), (-1, neg_label)]:
                    px = ox + sign * dx * far
                    py = oy + sign * dy * far
                    cx = max(margin, min(px, cw - margin))
                    cy = max(margin, min(py, ch - margin))
                    arrows.append((cx, cy, sign * dx, sign * dy, lbl))
        self._arrow_overlay.arrows = arrows
        self._arrow_overlay.update()

        # Clip axis lines to viewport bounds + margin so GPU never
        # handles huge coordinates (prevents float32 rasterizer jitter).
        bounds = self._get_viewport_bounds()
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            # 100% margin keeps lines off-screen but coordinates small
            mx = (xmax - xmin)
            my = (ymax - ymin)
            xmin -= mx;  xmax += mx
            ymin -= my;  ymax += my
            orig_c = -self._origin.astype(np.float64)
            self._x_axis.set_data(
                np.array([[xmin, orig_c[1]],
                           [xmax, orig_c[1]]], dtype=np.float64))
            self._y_axis.set_data(
                np.array([[orig_c[0], ymin],
                           [orig_c[0], ymax]], dtype=np.float64))

    def _data_to_canvas(self, data_xy: np.ndarray) -> np.ndarray | None:
        """Map a data (x, y) point to canvas pixel coordinates via node_transform."""
        try:
            tr = self._canvas.scene.node_transform(self._visual_root)
            screen = tr.imap([float(data_xy[0]), float(data_xy[1]), 0, 1])
            return np.array([float(screen[0]), float(screen[1])])
        except Exception:
            return None

    def _fit_view(self) -> None:
        if self._positions is None or len(self._positions) == 0:
            return
        xmin, ymin = self._positions.min(axis=0)
        xmax, ymax = self._positions.max(axis=0)
        margin_x = max((xmax - xmin) * 0.05, 1.0)
        margin_y = max((ymax - ymin) * 0.05, 1.0)
        self._camera.set_range(
            x=(xmin - margin_x, xmax + margin_x),
            y=(ymin - margin_y, ymax + margin_y),
        )

    def _on_mouse_move(self, event) -> None:
        """Handle hover tooltip, rubber-band drag, and Shift+drag rotation."""
        if self._rotating and event.is_dragging:
            dx = event.pos[0] - self._rotate_start_x
            self._rotation_deg = self._rotate_start_angle + dx * 0.3
            self._apply_rotation()
            return

        # Rubber-band drag update
        if self._rubber_dragging and event.is_dragging:
            sx, sy = self._rubber_start_px
            cx, cy = float(event.pos[0]), float(event.pos[1])
            x0, y0 = min(sx, cx), min(sy, cy)
            w, h = abs(cx - sx), abs(cy - sy)
            self._rubber_overlay.resize(self._canvas.native.width(), self._canvas.native.height())
            self._rubber_overlay.set_rect(QRectF(x0, y0, w, h))
            event.handled = True
            return

        # Update coordinate readout
        try:
            tr = self._canvas.scene.node_transform(self._visual_root)
            pos3 = tr.map(list(event.pos) + [0, 1])
            data_x, data_y = pos3[0], pos3[1]
            self._coord_label.setText(
                f"X: {data_x + self._origin[0]:,.1f} nm   Y: {data_y + self._origin[1]:,.1f} nm"
            )
            self._coord_label.adjustSize()
            self._reposition_coord_label()
        except Exception:
            pass

        # Update ruler endpoint while stretching
        if self._ruler_active and self._ruler_start is not None:
            self._ruler_end = np.array([data_x, data_y], dtype=np.float32)
            self._update_ruler_visual()

        # Stripe region hover (instant AABB check, not throttled)
        self._update_stripe_hover(data_x, data_y)

        if self._kdtree is None and self._data is None:
            return

        # Throttle hover queries to ~60 fps
        self._pending_hover_pos = (data_x, data_y)
        self._pending_hover_px = (int(event.pos[0]), int(event.pos[1]))
        if not self._hover_timer.isActive():
            self._hover_timer.start()

    def _hit_test(self, scene_pos: np.ndarray, min_radius_data: float) -> int | None:
        """Return the index of the shot under *scene_pos*, or None.

        Uses ``query_ball_point`` with a radius equal to the largest
        rendered shot so that every candidate is considered regardless
        of how many smaller shots lie between the click and a large
        shot's centre.  Among all hits the nearest centre wins.
        """
        if self._kdtree is None:
            return None

        disc_mode = self._marker_mode == 'disc'
        scale = _DISC_SIZE_SCALE if disc_mode else 1.0

        # Determine the maximum possible hit radius (half the largest marker)
        if self._uniform_size is not None:
            max_r = self._uniform_size * scale / 2.0
        else:
            max_r = float(np.max(self._sizes)) * scale / 2.0
        search_r = max(max_r, min_radius_data)

        # Cap the search radius to avoid returning millions of candidates
        # when zoomed out.  10× the natural marker radius is generous enough
        # for click-selection; beyond that, individual shots are sub-pixel.
        search_r = min(search_r, max(max_r * 10.0, 1.0))

        idxs = self._kdtree.query_ball_point(scene_pos, r=search_r)
        if not idxs:
            return None

        # Compute distances for all candidates
        centres = self._kdtree.data[idxs]          # (N, 2)
        dists = np.linalg.norm(centres - scene_pos, axis=1)

        # Sort by distance (nearest centre first)
        order = np.argsort(dists)

        for o in order:
            i = idxs[o]
            d = dists[o]
            sz = self._uniform_size if self._uniform_size is not None else self._sizes[i]
            r = sz * scale / 2.0
            if d <= max(r, min_radius_data):
                return int(i)

        return None

    def _update_stripe_hover(self, data_x: float, data_y: float) -> None:
        """Check if cursor is inside a stripe AABB and show/hide rectangle."""
        hovered = None
        for i, (x0, y0, x1, y1) in enumerate(self._stripe_aabbs):
            if x0 <= data_x <= x1 and y0 <= data_y <= y1:
                hovered = i
                break
        if hovered != self._hovered_stripe_idx:
            self._hovered_stripe_idx = hovered
            if hovered is not None and self._stripe_rect is not None:
                aabb = self._stripe_aabbs[hovered]
                cx = (aabb[0] + aabb[2]) / 2
                cy = (aabb[1] + aabb[3]) / 2
                w = aabb[2] - aabb[0]
                h = aabb[3] - aabb[1]
                self._stripe_rect.center = (cx, cy)
                self._stripe_rect.width = w
                self._stripe_rect.height = h
                self._stripe_rect.visible = True
                r = self._stripe_regions[hovered]
                tip = (
                    f"File: {r['name']}\n"
                    f"Shots: {r['shots']:,}\n"
                    f"Origin: ({r['originX']:,}, {r['originY']:,})\n"
                    f"Width: {r['width']:,}   Length: {r['length']:,}\n"
                    f"SubField Height: {r['subFieldHeight']:,}\n"
                    f"Overlap: {r['overlap']:,}"
                )
                self._stripe_tooltip.setText(tip)
                self._stripe_tooltip.adjustSize()
                self._position_stripe_tooltip()
                self._stripe_tooltip.setVisible(True)
            elif self._stripe_rect is not None:
                self._stripe_rect.visible = False
                self._stripe_tooltip.setVisible(False)

    def _do_hover_query(self) -> None:
        """Perform the actual KD-tree hover lookup (throttled)."""
        if self._pending_hover_pos is None:
            return
        data_x, data_y = self._pending_hover_pos
        mx, my = self._pending_hover_px
        self._pending_hover_pos = None

        if self._kdtree is None or self._data is None:
            return

        scene_pos = np.array([data_x, data_y], dtype=np.float32)

        # Minimum hit radius: 5 pixels converted to data units.
        try:
            tr_h = self._canvas.scene.node_transform(self._visual_root)
            p0_h = tr_h.map([0, 0, 0, 1])
            p1_h = tr_h.map([1, 0, 0, 1])
            min_radius_data = 5.0 * abs(p1_h[0] - p0_h[0])
        except Exception:
            min_radius_data = 0.0

        idx = self._hit_test(scene_pos, min_radius_data)

        if idx is not None:
            # Don't show hover tooltip for the already-selected shot
            if idx == self._selected_idx:
                self._hover_tooltip.setVisible(False)
            else:
                shot_num = idx + 1
                sx = self._data.x[idx]
                sy = self._data.y[idx]
                sd = self._data.dwell[idx]
                tip = (
                    f"Shot #{shot_num:,}\n"
                    f"X: {sx:,.0f} nm   Y: {sy:,.0f} nm\n"
                    f"Dwell: {sd:,.0f} ns"
                )
                self._hover_tooltip.setText(tip)
                self._hover_tooltip.adjustSize()
                self._hover_tooltip.move(mx + 15, my - 10)
                self._hover_tooltip.setVisible(True)
        else:
            self._hover_tooltip.setVisible(False)

    def _on_mouse_press(self, event) -> None:
        """Handle click-to-select, rubber-band start, and Shift+drag rotation."""
        # Track right-click start position for ruler clear detection
        if event.button == 2:
            self._right_click_start_px = (float(event.pos[0]), float(event.pos[1]))

        # Shift + left-click → start rotation
        if event.button == 1 and Qt.KeyboardModifier.ShiftModifier in self._get_modifiers(event):
            self._rotating = True
            self._rotate_start_x = event.pos[0]
            self._rotate_start_pos = (float(event.pos[0]), float(event.pos[1]))
            self._rotate_start_angle = self._rotation_deg
            # Save current transform so we can compose delta on top
            tr = self._visual_root.transform
            if hasattr(tr, 'matrix'):
                self._rotate_base_matrix = tr.matrix.copy()
            else:
                self._rotate_base_matrix = np.eye(4)
            # Capture mouse position in data space as rotation pivot
            try:
                tr = self._canvas.scene.node_transform(self._visual_root)
                pos3 = tr.map(list(event.pos) + [0, 1])
                self._rotate_center = np.array([pos3[0], pos3[1]], dtype=np.float64)
            except Exception:
                self._rotate_center = None
            event.handled = True
            return

        # Plain left-click → start rubber-band or single-click select
        if event.button == 1 and not self._get_modifiers(event):
            self._rubber_dragging = True
            self._camera.interactive = False  # prevent pan while dragging
            self._rubber_start_px = (float(event.pos[0]), float(event.pos[1]))
            try:
                tr = self._canvas.scene.node_transform(self._visual_root)
                pos3 = tr.map(list(event.pos) + [0, 1])
                self._rubber_start_data = np.array([pos3[0], pos3[1]], dtype=np.float32)
            except Exception:
                self._rubber_start_data = None
            event.handled = True

    def _on_mouse_release(self, event) -> None:
        # Right-click (not drag) → clear ruler
        if event.button == 2 and self._ruler_start is not None:
            sp = getattr(self, '_right_click_start_px', None)
            if sp is not None:
                ex, ey = float(event.pos[0]), float(event.pos[1])
                if ((ex - sp[0]) ** 2 + (ey - sp[1]) ** 2) ** 0.5 < 5:
                    self._clear_ruler()

        if self._rotating:
            self._rotating = False
            # Tiny Shift+drag → Shift+click → start ruler
            sp = getattr(self, '_rotate_start_pos', None)
            if sp is not None:
                ex, ey = float(event.pos[0]), float(event.pos[1])
                if ((ex - sp[0]) ** 2 + (ey - sp[1]) ** 2) ** 0.5 < 5:
                    try:
                        tr = self._canvas.scene.node_transform(self._visual_root)
                        pos3 = tr.map(list(event.pos) + [0, 1])
                        self._ruler_start = np.array([pos3[0], pos3[1]], dtype=np.float32)
                        self._ruler_end = None
                        self._ruler_active = True
                        self._update_ruler_visual()
                    except Exception:
                        pass
            return

        if self._rubber_dragging:
            self._rubber_dragging = False
            self._camera.interactive = True  # re-enable pan/zoom
            self._rubber_overlay.set_rect(None)  # hide rectangle

            if self._rubber_start_px is None:
                return

            sx, sy = self._rubber_start_px
            ex, ey = float(event.pos[0]), float(event.pos[1])
            drag_dist = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5

            if drag_dist < 5:
                # If ruler is stretching, lock its endpoint
                if self._ruler_active:
                    try:
                        tr = self._canvas.scene.node_transform(self._visual_root)
                        pos3 = tr.map(list(event.pos) + [0, 1])
                        self._ruler_end = np.array([pos3[0], pos3[1]], dtype=np.float32)
                        self._ruler_active = False
                        self._update_ruler_visual()
                    except Exception:
                        pass
                    return
                # Tiny drag → treat as single click
                self._handle_click_select(event)
                return

            # Box selection: map all 4 screen-space corners of the
            # rubber-band rectangle to data space so the selection
            # region stays correct even when the view is rotated.
            if self._positions is None:
                return
            try:
                tr = self._canvas.scene.node_transform(self._visual_root)
                # Screen-space corners of the drawn rectangle
                corners_px = [
                    [sx, sy, 0, 1],
                    [ex, sy, 0, 1],
                    [ex, ey, 0, 1],
                    [sx, ey, 0, 1],
                ]
                quad = np.array(
                    [[tr.map(c)[0], tr.map(c)[1]] for c in corners_px],
                    dtype=np.float64,
                )
            except Exception:
                return

            # Fast AABB pre-filter: only test points inside the quad's
            # bounding box.  For a zoomed-in view on a 45M-shot file,
            # this reduces the candidate set from millions to thousands.
            qx_min, qx_max = quad[:, 0].min(), quad[:, 0].max()
            qy_min, qy_max = quad[:, 1].min(), quad[:, 1].max()
            all_pos = self._positions
            aabb_mask = (
                (all_pos[:, 0] >= qx_min) & (all_pos[:, 0] <= qx_max)
                & (all_pos[:, 1] >= qy_min) & (all_pos[:, 1] <= qy_max)
            )
            candidates = np.nonzero(aabb_mask)[0]
            pos = all_pos[candidates]

            # Point-in-convex-quadrilateral test using cross-product
            # signs.  A point is inside the quad when it is on the
            # same side of every directed edge.
            def _edge_sign(a, b, pts):
                """Return sign of cross(AB, AP) for each point P."""
                return (b[0] - a[0]) * (pts[:, 1] - a[1]) - \
                       (b[1] - a[1]) * (pts[:, 0] - a[0])

            s0 = _edge_sign(quad[0], quad[1], pos)
            s1 = _edge_sign(quad[1], quad[2], pos)
            s2 = _edge_sign(quad[2], quad[3], pos)
            s3 = _edge_sign(quad[3], quad[0], pos)

            # All same sign (or zero) → inside
            mask = (
                ((s0 >= 0) & (s1 >= 0) & (s2 >= 0) & (s3 >= 0))
                | ((s0 <= 0) & (s1 <= 0) & (s2 <= 0) & (s3 <= 0))
            )
            indices = candidates[mask].astype(np.intp)

            self._apply_box_selection(indices)
            event.handled = True

    def _handle_click_select(self, event) -> None:
        """Select or deselect a shot on click."""
        # Clear any box selection first
        if len(self._box_selected_indices):
            self.clear_box_selection()

        if self._kdtree is None or self._data is None or self._positions is None:
            return

        try:
            tr = self._canvas.scene.node_transform(self._visual_root)
            pos3 = tr.map(list(event.pos) + [0, 1])
            scene_pos = np.array([pos3[0], pos3[1]], dtype=np.float32)
        except Exception:
            return

        # Minimum hit radius: 5 pixels converted to data units
        try:
            p0 = tr.map([0, 0, 0, 1])
            p1 = tr.map([1, 0, 0, 1])
            min_radius_data = 5.0 * abs(p1[0] - p0[0])
        except Exception:
            min_radius_data = 0.0

        idx = self._hit_test(scene_pos, min_radius_data)

        # Deselect previous
        if self._selected_idx is not None:
            self._sel_marker.visible = False

        if idx is not None:
            if self._selected_idx == idx:
                # Clicking the same shot again → deselect
                self._selected_idx = None
                self._sel_marker.visible = False
                self._sel_lines.visible = False
                self._tooltip.setVisible(False)
                self._hover_tooltip.setVisible(False)
                self.box_selected.emit([])
            else:
                # Select new shot
                self._selected_idx = idx
                self._hover_tooltip.setVisible(False)
                # Show selected shot as overlay on top
                sel_sz = self._uniform_size if self._uniform_size is not None else self._sizes[idx:idx+1]
                computed_sel = self._sel_size(sel_sz)
                dpp = self._get_data_per_px()
                print(f"[SEL] raw_sz={sel_sz} sel_sz={computed_sel} dpp={dpp} stride={self._decim_stride} mode={self._marker_mode}")
                self._sel_marker.set_data(
                    self._positions[idx:idx+1],
                    size=computed_sel,
                    face_color=_SELECTED_COLOR,
                    edge_width=0,
                )
                self._sel_marker.visible = True
                self._update_sel_lines()
                # Show persistent tooltip
                shot_num = idx + 1
                sx = self._data.x[idx]
                sy = self._data.y[idx]
                sd = self._data.dwell[idx]
                tip = (
                    f"Shot #{shot_num:,}\n"
                    f"X: {sx:,.0f} nm   Y: {sy:,.0f} nm\n"
                    f"Dwell: {sd:,.0f} ns"
                )
                self._tooltip.setText(tip)
                self._tooltip.adjustSize()
                screen_pos = self._data_to_canvas(self._positions[idx])
                if screen_pos is not None:
                    self._tooltip.move(int(screen_pos[0]) + 20, int(screen_pos[1]) - self._tooltip.height() - 10)
                self._tooltip.setVisible(True)
                self.box_selected.emit([idx])
        else:
            # Clicked empty space → deselect
            self._selected_idx = None
            self._sel_marker.visible = False
            self._sel_lines.visible = False
            self._tooltip.setVisible(False)
            self._hover_tooltip.setVisible(False)
            self.box_selected.emit([])

        self._canvas.update()
        event.handled = True

    def _upload_box_sel_markers(self) -> None:
        """Upload box-selection overlay markers, viewport-culled then strided."""
        idx_arr = self._box_selected_indices
        if len(idx_arr) == 0 or self._positions is None:
            return
        # Viewport-cull: only upload markers for selected shots currently on screen
        bounds = self._get_viewport_bounds()
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            mx = (xmax - xmin) * 0.05
            my = (ymax - ymin) * 0.05
            sel_pos = self._positions[idx_arr]
            vis = ((sel_pos[:, 0] >= xmin - mx) & (sel_pos[:, 0] <= xmax + mx) &
                   (sel_pos[:, 1] >= ymin - my) & (sel_pos[:, 1] <= ymax + my))
            idx_arr = idx_arr[vis]
        if len(idx_arr) == 0:
            self._box_sel_markers.set_data(np.empty((0, 2)))
            return
        # Stride so we never upload more than ~500k markers
        stride = max(1, len(idx_arr) // 500_000)
        sub = idx_arr[::stride]
        box_sz = self._uniform_size if self._uniform_size is not None else self._sizes[sub]
        self._box_sel_markers.set_data(
            self._positions[sub],
            size=self._box_sel_size(box_sz),
            face_color=_BOX_SELECTED_COLOR,
            edge_width=0,
        )

    def _apply_box_selection(self, indices: np.ndarray) -> None:
        """Highlight all box-selected shots and emit the signal."""
        # Clear any single-shot selection
        if self._selected_idx is not None:
            self._selected_idx = None
            self._sel_marker.visible = False
            self._sel_lines.visible = False
            self._tooltip.setVisible(False)
            self._hover_tooltip.setVisible(False)

        self._box_sel_markers.visible = False
        self._box_selected_indices = indices

        if len(indices) and self._positions is not None:
            self._upload_box_sel_markers()
            self._box_sel_markers.visible = True

        self._update_sel_lines()
        self._canvas.update()
        self.box_selected.emit(indices)

    def clear_box_selection(self) -> None:
        """Remove box-selection highlighting."""
        if len(self._box_selected_indices):
            self._box_selected_indices = np.empty(0, dtype=np.intp)
            self._box_sel_markers.visible = False
            self._update_sel_lines()
            self._canvas.update()
            self.box_selected.emit([])

    def _apply_rotation(self) -> None:
        """Apply incremental rotation around the stored pivot.

        Composes only the *delta* angle (since drag start) on top of the
        saved base transform so that changing the pivot never causes a
        sudden view jump.
        """
        if self._positions is None or len(self._positions) == 0:
            return
        if self._rotate_center is not None:
            cx, cy = self._rotate_center
        else:
            cx, cy = self._positions.mean(axis=0)

        delta_deg = self._rotation_deg - self._rotate_start_angle

        # Build incremental rotation around the pivot
        inc = scene.transforms.MatrixTransform()
        inc.translate((-cx, -cy, 0))
        inc.rotate(delta_deg, (0, 0, 1))
        inc.translate((cx, cy, 0))

        # Compose on top of the base transform saved at drag start
        base = self._rotate_base_matrix
        if base is not None:
            self._visual_root.transform.matrix = inc.matrix @ base
        else:
            self._visual_root.transform = inc
        self._canvas.update()

    def _update_ruler_visual(self) -> None:
        """Redraw the ruler line, tick marks, and reposition the distance label."""
        if self._ruler_start is None:
            self._ruler_line.visible = False
            self._ruler_ticks.visible = False
            self._ruler_label.setVisible(False)
            return
        end = self._ruler_end if self._ruler_end is not None else self._ruler_start
        pts = np.array([self._ruler_start, end], dtype=np.float32)
        self._ruler_line.set_data(pos=pts, color=(1, 1, 1, 0.9))
        self._ruler_line.visible = True
        dx = float(end[0] - self._ruler_start[0])
        dy = float(end[1] - self._ruler_start[1])
        dist = _math.hypot(dx, dy)

        # Generate tick marks along the ruler
        if dist > 1e-9:
            # Choose a nice tick interval
            raw = dist / 10.0
            mag = 10.0 ** _math.floor(_math.log10(raw)) if raw > 0 else 1.0
            nice = mag
            for candidate in [1, 2, 5, 10]:
                if candidate * mag >= raw:
                    nice = candidate * mag
                    break
            # Perpendicular unit vector
            inv = 1.0 / dist
            ux, uy = dx * inv, dy * inv  # along ruler
            px, py = -uy, ux             # perpendicular
            # Tick half-length: 6 pixels in data units
            dpp = self._get_data_per_px()
            half = 6.0 * dpp
            tick_pts = []
            t = nice
            while t < dist - nice * 0.01:
                cx = self._ruler_start[0] + ux * t
                cy = self._ruler_start[1] + uy * t
                tick_pts.append([cx - px * half, cy - py * half])
                tick_pts.append([cx + px * half, cy + py * half])
                t += nice
            if tick_pts:
                self._ruler_ticks.set_data(
                    pos=np.array(tick_pts, dtype=np.float32),
                    connect='segments', color=(1, 1, 1, 0.7))
                self._ruler_ticks.visible = True
            else:
                self._ruler_ticks.visible = False
        else:
            self._ruler_ticks.visible = False

        if dist >= 1e6:
            txt = f"{dist / 1e6:,.3f} mm"
        elif dist >= 1e3:
            txt = f"{dist / 1e3:,.3f} \u00b5m"
        else:
            txt = f"{dist:,.1f} nm"
        self._ruler_label.setText(txt)
        self._ruler_label.adjustSize()
        mid = (self._ruler_start + end) / 2.0
        screen = self._data_to_canvas(mid)
        if screen is not None:
            self._ruler_label.move(int(screen[0]) + 10, int(screen[1]) - 20)
        self._ruler_label.setVisible(True)

    def _clear_ruler(self) -> None:
        """Remove the ruler."""
        self._ruler_start = None
        self._ruler_end = None
        self._ruler_active = False
        self._ruler_line.visible = False
        self._ruler_ticks.visible = False
        self._ruler_label.setVisible(False)
        self._canvas.update()

    @staticmethod
    def _get_modifiers(event) -> set:
        """Extract Qt keyboard modifiers from a vispy event."""
        mods = set()
        try:
            for m in event.modifiers:
                name = m.name if hasattr(m, 'name') else str(m)
                name_lower = name.lower()
                if "shift" in name_lower:
                    mods.add(Qt.KeyboardModifier.ShiftModifier)
                elif "control" in name_lower:
                    mods.add(Qt.KeyboardModifier.ControlModifier)
        except (AttributeError, TypeError):
            pass
        return mods
