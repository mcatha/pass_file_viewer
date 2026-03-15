"""
Generate a .pass v3 file: e-beam lithography chip layout
blended with surreal fractal art.

Upper half — real IC structures:
  SRAM arrays, ALU datapath, H-tree clock distribution, bus routing.

Lower half — surreal fractals emerging from the chip geometry:
  - Koch snowflake (solid scanline fill)
  - Dragon curve (space-filling fractal path)
  - Hilbert curve (dense space-filling trace)
  - L-system fractal forest (trees growing from the bus lines)
  - Archimedean spiral (sweeping through the scene)

Structured VLSI blends into organic forms — surrealism meets silicon.

Field: 60 × 60 µm  (60000 × 60000 nm).
"""

import struct, math
import numpy as np
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("WARNING: Pillow not installed — logo will be skipped")

# ── parameters ──────────────────────────────────────────────────────
OUTPUT = Path(__file__).parent.parent / "chip_layout.pass"
LOGO_PATH = Path(__file__).parent.parent / "mb-logo-w-tag.png"
FIELD = 60000           # nm
HALF  = FIELD // 2
MAX_DWELL = 16383
ADDR_GRID = 2          # nm — beam address grid resolution.
                       # ALL shot coordinates are snapped to this grid
                       # to guarantee clean line edges (LER control).

shots: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []


def snap(v):
    """Snap coordinate(s) to the beam address grid."""
    return (np.round(np.asarray(v, dtype=np.float64) / ADDR_GRID) * ADDR_GRID).astype(np.int32)


def add(x, y, dwell):
    """Append a shot batch.  Coordinates are snapped to ADDR_GRID."""
    x = snap(x).ravel()
    y = snap(y).ravel()
    d = np.asarray(dwell, dtype=np.int32).ravel()
    if d.size == 1 and x.size > 1:
        d = np.full(x.size, d[0], dtype=np.int32)
    if x.size > 0:
        shots.append((x, y, d))


# ── primitives ──────────────────────────────────────────────────────

def fill_rect(cx, cy, w, h, dwell, pitch):
    """Tile a rectangle with a hex-packed shot grid + interstitial fill.

    Primary grid: hex-packed rows (every odd row offset by pitch/2).
    Interstitial grid: smaller-dwell shots at the centres of the hex
    triangles, filling the gaps between the primary shots.

    This gives near-uniform dose coverage across the feature."""
    if w <= 0 or h <= 0:
        return
    x0 = cx - w / 2
    y0 = cy - h / 2
    diam = dwell * _NM_PER_NS

    # Row spacing for hex packing: pitch * sqrt(3)/2
    row_sp = pitch * 0.866
    if row_sp < ADDR_GRID:
        row_sp = ADDR_GRID

    # --- primary hex grid ---
    gx_base = np.arange(x0 + pitch / 2, x0 + w, pitch)
    gy_all  = np.arange(y0 + row_sp / 2, y0 + h, row_sp)
    if gx_base.size == 0 or gy_all.size == 0:
        return

    all_x, all_y, all_d = [], [], []
    for row_i, yv in enumerate(gy_all):
        offset = (pitch / 2) if (row_i % 2) else 0.0
        gx = gx_base + offset
        gx = gx[gx < x0 + w]  # clip to rect
        if gx.size > 0:
            all_x.append(gx)
            all_y.append(np.full(gx.size, yv))
            all_d.append(np.full(gx.size, dwell, dtype=np.int32))

    # --- interstitial fill shots (smaller dwell, centred in triangles) ---
    # Only add if the feature is large enough to benefit
    if w > diam * 2 and h > diam * 2:
        fill_dwell = max(int(dwell * 0.5), 100)  # half-size fill shots
        ix_off = pitch / 2       # x-offset from primary grid
        iy_off = row_sp / 3      # y-offset (1/3 up triangle height)
        for row_i, yv in enumerate(gy_all[:-1]):
            base_off = (pitch / 2) if (row_i % 2) else 0.0
            # Two interstitial rows per primary row gap
            for fy in [yv + iy_off, yv + 2 * iy_off]:
                if fy > y0 + h:
                    continue
                foff = base_off + ix_off / 2
                fx = gx_base + foff
                fx = fx[(fx > x0) & (fx < x0 + w)]
                if fx.size > 0:
                    all_x.append(fx)
                    all_y.append(np.full(fx.size, fy))
                    all_d.append(np.full(fx.size, fill_dwell, dtype=np.int32))

    if all_x:
        add(np.concatenate(all_x), np.concatenate(all_y),
            np.concatenate(all_d))


def fill_trace_h(x0, y0, x1, width, dwell, pitch):
    """Horizontal trace from x0→x1 at vertical centre y0."""
    if x1 < x0: x0, x1 = x1, x0
    if x1 - x0 < 1: return
    fill_rect((x0 + x1) / 2, y0, x1 - x0, width, dwell, pitch)


def fill_trace_v(x0, y0, y1, width, dwell, pitch):
    """Vertical trace from y0→y1 at horizontal centre x0."""
    if y1 < y0: y0, y1 = y1, y0
    if y1 - y0 < 1: return
    fill_rect(x0, (y0 + y1) / 2, width, y1 - y0, dwell, pitch)


def fill_contact(cx, cy, size, dwell, pitch):
    """Square contact/via."""
    fill_rect(cx, cy, size, size, dwell, pitch)


def fill_trace_along(pts, width, dwell, pitch):
    """Tile a Manhattan polyline (list of (x,y) vertices) with width."""
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if abs(x1 - x0) > abs(y1 - y0):
            fill_trace_h(x0, (y0 + y1) / 2, x1, width, dwell, pitch)
        else:
            fill_trace_v((x0 + x1) / 2, y0, y1, width, dwell, pitch)


# ── dose & pitch ────────────────────────────────────────────────────
# The viewer displays shot diameter = dwell × 0.01 nm.
# We derive pitch (beam step) directly from dwell so they always match:
#   pitch = floor(diameter × OVERLAP_FACTOR)
# ~20% overlap between adjacent shots → uniform dose coverage.
_NM_PER_NS = 0.01           # viewer's dwell-to-diameter mapping
_OVERLAP   = 0.62           # pitch/diameter ratio (~38% overlap)
                             # hex packing + interstitial fill gives
                             # near-uniform coverage at this ratio

def pitch_for(dwell):
    """Beam step size for a given dwell, consistent with viewer."""
    return max(int(dwell * _NM_PER_NS * _OVERLAP), ADDR_GRID)

def diam_for(dwell):
    """Displayed shot diameter for a given dwell."""
    return dwell * _NM_PER_NS

# Dwell per feature type (ns).  Determines both dose AND displayed size.
DOSE_PAD      = 3200    # diam 32 nm — large pads, high dose
DOSE_METAL    = 2500    # diam 25 nm — standard interconnect
DOSE_POLY     = 1500    # diam 15 nm — critical gate lines
DOSE_CONTACT  = 2800    # diam 28 nm — via holes, extra dose
DOSE_ALIGN    = 3500    # diam 35 nm — alignment marks
DOSE_LOGO     = 2200    # diam 22 nm — logo text
DOSE_FRACTAL  = 1800    # diam 18 nm — organic / fractal structures

# Pitches auto-derived from dwell (always consistent with viewer)
PITCH_PAD     = pitch_for(DOSE_PAD)       # 25
PITCH_METAL   = pitch_for(DOSE_METAL)     # 20
PITCH_POLY    = pitch_for(DOSE_POLY)      # 12
PITCH_CONTACT = pitch_for(DOSE_CONTACT)   # 22
PITCH_ALIGN   = pitch_for(DOSE_ALIGN)     # 28
PITCH_FRACTAL = pitch_for(DOSE_FRACTAL)   # 14
PITCH_LOGO    = pitch_for(DOSE_LOGO)      # 17

print(f"  Pitches: pad={PITCH_PAD} metal={PITCH_METAL} poly={PITCH_POLY} "
      f"contact={PITCH_CONTACT} align={PITCH_ALIGN} fractal={PITCH_FRACTAL} "
      f"logo={PITCH_LOGO}")

# =====================================================================
# FRACTAL / ORGANIC GEOMETRY GENERATORS
# =====================================================================

def hilbert_curve(order, size):
    """Generate a Hilbert space-filling curve as a list of (x, y) points.
    Returns points in [0, size] × [0, size]."""
    def _hilbert(x, y, ax, ay, bx, by, pts):
        w = abs(ax + ay)
        h = abs(bx + by)
        dax = 1 if ax > 0 else (-1 if ax < 0 else 0)
        day = 1 if ay > 0 else (-1 if ay < 0 else 0)
        dbx = 1 if bx > 0 else (-1 if bx < 0 else 0)
        dby = 1 if by > 0 else (-1 if by < 0 else 0)
        if h == 1:
            for i in range(w):
                pts.append((x, y))
                x += dax; y += day
            return
        if w == 1:
            for i in range(h):
                pts.append((x, y))
                x += dbx; y += dby
            return
        ax2 = ax // 2; ay2 = ay // 2
        bx2 = bx // 2; by2 = by // 2
        w2 = abs(ax2 + ay2)
        h2 = abs(bx2 + by2)
        if 2 * w > 3 * h:
            if (w2 & 1) and (w > 2):
                ax2 += dax; ay2 += day
            _hilbert(x, y, ax2, ay2, bx, by, pts)
            _hilbert(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by, pts)
        else:
            if (h2 & 1) and (h > 2):
                bx2 += dbx; by2 += dby
            _hilbert(x, y, bx2, by2, ax2, ay2, pts)
            _hilbert(x + bx2, y + by2, ax, ay, bx - bx2, by - by2, pts)
            _hilbert(x + (ax - dax) + (bx2 - dbx),
                     y + (ay - day) + (by2 - dby),
                     -bx2, -by2, -(ax - ax2), -(ay - ay2), pts)
    n = 2 ** order
    pts = []
    _hilbert(0, 0, n, 0, 0, n, pts)
    pts.append(pts[-1])  # close
    scale = size / n
    return [(p[0] * scale, p[1] * scale) for p in pts]


def koch_snowflake(order, size):
    """Generate Koch snowflake vertices (equilateral triangle base)."""
    def _subdivide(p1, p2, depth):
        if depth == 0:
            return [p1]
        dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
        a = (p1[0] + dx / 3, p1[1] + dy / 3)
        b = (p1[0] + dx * 2 / 3, p1[1] + dy * 2 / 3)
        # peak
        px = (p1[0] + p2[0]) / 2 + (p2[1] - p1[1]) * math.sqrt(3) / 6
        py = (p1[1] + p2[1]) / 2 - (p2[0] - p1[0]) * math.sqrt(3) / 6
        pk = (px, py)
        return (_subdivide(p1, a, depth - 1) +
                _subdivide(a, pk, depth - 1) +
                _subdivide(pk, b, depth - 1) +
                _subdivide(b, p2, depth - 1))
    h = size * math.sqrt(3) / 2
    tri = [(0, -h / 3), (size / 2, h * 2 / 3), (-size / 2, h * 2 / 3)]
    pts = []
    for i in range(3):
        pts.extend(_subdivide(tri[i], tri[(i + 1) % 3], order))
    pts.append(pts[0])
    return pts


def htree_points(cx, cy, half_len, depth, width, dwell, pitch):
    """Recursive H-tree clock distribution — fractal branching."""
    if depth <= 0 or half_len < width * 2:
        return
    # horizontal bar
    fill_trace_h(cx - half_len, cy, cx + half_len, width, dwell, pitch)
    # vertical bars at ends
    vlen = half_len * 0.7
    fill_trace_v(cx - half_len, cy - vlen, cy + vlen, width, dwell, pitch)
    fill_trace_v(cx + half_len, cy - vlen, cy + vlen, width, dwell, pitch)
    # contacts at branch points
    fill_contact(cx - half_len, cy, width * 1.5, DOSE_CONTACT, PITCH_CONTACT)
    fill_contact(cx + half_len, cy, width * 1.5, DOSE_CONTACT, PITCH_CONTACT)
    # recurse at 4 endpoints
    nhl = half_len * 0.5
    nw = max(width * 0.75, pitch * 2)
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            htree_points(cx + sx * half_len, cy + sy * vlen,
                         nhl, depth - 1, nw, dwell, pitch)


def serpentine_meander(x0, y0, n_turns, seg_len, spacing, width, dwell, pitch):
    """Serpentine meander resistor — deterministic Manhattan path."""
    cx, cy = x0, y0
    for i in range(n_turns):
        direction = 1 if i % 2 == 0 else -1
        # horizontal run
        fill_trace_h(cx, cy, cx + direction * seg_len, width, dwell, pitch)
        cx += direction * seg_len
        # vertical jog
        if i < n_turns - 1:
            fill_trace_v(cx, cy, cy + spacing, width, dwell, pitch)
            cy += spacing


def lsystem_tree(x0, y0, length, direction, depth, width, dwell, pitch,
                 shrink=0.68):
    """L-system binary tree using ONLY Manhattan segments.
    direction: 0=right, 1=up, 2=left, 3=down.  Branches turn ±1.
    All coordinates grid-snapped for clean LER."""
    if depth <= 0 or length < width * 2:
        return
    dx = [1, 0, -1, 0][direction % 4]
    dy = [0, 1, 0, -1][direction % 4]
    x1 = x0 + dx * length
    y1 = y0 + dy * length
    if dx != 0:
        fill_trace_h(x0, y0, x1, width, dwell, pitch)
    else:
        fill_trace_v(x0, y0, y1, width, dwell, pitch)
    nw = max(width * 0.8, pitch * 2)
    nl = length * shrink
    # Branch: turn left and right (±1 in direction space)
    lsystem_tree(x1, y1, nl, (direction + 1) % 4, depth - 1,
                 nw, dwell, pitch, shrink)
    lsystem_tree(x1, y1, nl, (direction - 1) % 4, depth - 1,
                 nw, dwell, pitch, shrink)


def dragon_curve_pts(order, size):
    """Dragon curve — self-similar space-filling fractal.
    Returns list of (x, y) points, Manhattan-only, scaled to size."""
    turns = [1]
    for _ in range(order - 1):
        turns = turns + [1] + [1 - t for t in reversed(turns)]
    dx_tab = [1, 0, -1, 0]
    dy_tab = [0, 1, 0, -1]
    d = 0
    x, y = 0, 0
    pts = [(0, 0)]
    for t in turns:
        d = (d + (1 if t else -1)) % 4
        x += dx_tab[d]; y += dy_tab[d]
        pts.append((x, y))
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    span = max(xhi - xlo, yhi - ylo, 1)
    s = size / span
    return [((p[0] - xlo) * s, (p[1] - ylo) * s) for p in pts]


def archimedean_spiral_pts(cx, cy, r_max, turns, step):
    """Archimedean spiral — points along the curve at ~step spacing."""
    total_angle = turns * 2 * math.pi
    n_pts = max(int(total_angle * r_max / step), 200)
    pts = []
    for i in range(n_pts + 1):
        theta = total_angle * i / n_pts
        r = r_max * theta / total_angle
        pts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    return pts


# =====================================================================
# CHIP FLOORPLAN — 50 × 50 µm die in 60 × 60 µm field
#
#  +----------------------------------------------------+
#  | pad pad pad pad pad pad pad pad pad pad pad pad     |
#  |  +------+------+------------+------+-------+       |
#  |  | SRAM |H-tree|   ALU /    |H-tree| SRAM  |       |
#  |  | A    |clock | datapath   |clock | B     |       |
#  |  +------+------+-----+------+------+-------+       |
#  |  | Mandel- | Dragon  | Hilbert curve   |           |
#  |  | brot    | curve   |                 |           |
#  |  |      Archimedean spiral overlay      |           |
#  |  |     MULTIBEAM CORPORATION  logo strip       |    |
#  |  +---------------------------------------------+   |
#  | pad pad pad pad pad pad pad pad pad pad pad pad     |
#  +----------------------------------------------------+
# =====================================================================

DIE_CX, DIE_CY = HALF, HALF
DIE_W, DIE_H = 50000, 50000
DIE_L = DIE_CX - DIE_W // 2
DIE_R = DIE_CX + DIE_W // 2
DIE_B = DIE_CY - DIE_H // 2
DIE_T = DIE_CY + DIE_H // 2

PAD_ZONE = 2800
FUNC_L = DIE_L + PAD_ZONE + 400
FUNC_R = DIE_R - PAD_ZONE - 400
FUNC_B = DIE_B + PAD_ZONE + 400
FUNC_T = DIE_T - PAD_ZONE - 400

# =====================================================================
# 1. BOND PADS — ring around die perimeter
# =====================================================================
print("[1/16] Bond pads...")
PAD_SZ = 2000
PAD_PITCH_V = 3400
PAD_OFFSET = 1300

n_pads_h = int((DIE_W - 4000) / PAD_PITCH_V)
for i in range(n_pads_h):
    px = DIE_L + 2000 + i * PAD_PITCH_V + PAD_PITCH_V / 2
    fill_rect(px, DIE_T - PAD_OFFSET, PAD_SZ, PAD_SZ, DOSE_PAD, PITCH_PAD)
    fill_rect(px, DIE_B + PAD_OFFSET, PAD_SZ, PAD_SZ, DOSE_PAD, PITCH_PAD)

n_pads_v = int((DIE_H - 4000) / PAD_PITCH_V)
for i in range(n_pads_v):
    py = DIE_B + 2000 + i * PAD_PITCH_V + PAD_PITCH_V / 2
    fill_rect(DIE_L + PAD_OFFSET, py, PAD_SZ, PAD_SZ, DOSE_PAD, PITCH_PAD)
    fill_rect(DIE_R - PAD_OFFSET, py, PAD_SZ, PAD_SZ, DOSE_PAD, PITCH_PAD)

# =====================================================================
# 2. PAD-TO-CORE TRACES
# =====================================================================
print("[2/16] Pad-to-core traces...")
WIRE_W = 180

for i in range(n_pads_h):
    px = DIE_L + 2000 + i * PAD_PITCH_V + PAD_PITCH_V / 2
    fill_trace_v(px, FUNC_T, DIE_T - PAD_OFFSET - PAD_SZ / 2,
                 WIRE_W, DOSE_METAL, PITCH_METAL)
    fill_trace_v(px, DIE_B + PAD_OFFSET + PAD_SZ / 2, FUNC_B,
                 WIRE_W, DOSE_METAL, PITCH_METAL)

for i in range(n_pads_v):
    py = DIE_B + 2000 + i * PAD_PITCH_V + PAD_PITCH_V / 2
    fill_trace_h(DIE_L + PAD_OFFSET + PAD_SZ / 2, py, FUNC_L,
                 WIRE_W, DOSE_METAL, PITCH_METAL)
    fill_trace_h(FUNC_R, py, DIE_R - PAD_OFFSET - PAD_SZ / 2,
                 WIRE_W, DOSE_METAL, PITCH_METAL)

# =====================================================================
# 3. ALIGNMENT MARKS — crosses in die corners
# =====================================================================
print("[3/16] Alignment marks...")
for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
    ax = DIE_CX + sx * (DIE_W / 2 - 500)
    ay = DIE_CY + sy * (DIE_H / 2 - 500)
    fill_rect(ax, ay, 700, 120, DOSE_ALIGN, PITCH_ALIGN)
    fill_rect(ax, ay, 120, 700, DOSE_ALIGN, PITCH_ALIGN)

# ── Block layout ────────────────────────────────────────────────────
HDIV = FUNC_B + (FUNC_T - FUNC_B) * 0.54
LOGO_TOP = FUNC_B + (FUNC_T - FUNC_B) * 0.15

# Upper row blocks
SRAM_A_W = 7000
HTREE1_W = 2500
SRAM_B_W = 7000
HTREE2_W = 2500
ALU_W = FUNC_R - FUNC_L - SRAM_A_W - SRAM_B_W - HTREE1_W - HTREE2_W

SRAM_A_L = FUNC_L;          SRAM_A_R = SRAM_A_L + SRAM_A_W
HTREE1_L = SRAM_A_R;        HTREE1_R = HTREE1_L + HTREE1_W
ALU_L    = HTREE1_R;        ALU_R    = ALU_L + ALU_W
HTREE2_L = ALU_R;           HTREE2_R = HTREE2_L + HTREE2_W
SRAM_B_L = HTREE2_R;        SRAM_B_R = FUNC_R

# Lower zone — surreal fractals blending with the chip above
_lower_w = FUNC_R - FUNC_L
FRAC_L_W = int(_lower_w * 0.35)       # Mandelbrot zone
FRAC_M_W = int(_lower_w * 0.30)       # Dragon curve zone
FRAC_R_W = _lower_w - FRAC_L_W - FRAC_M_W  # Hilbert curve zone

FRAC_L_L = FUNC_L;                    FRAC_L_R = FRAC_L_L + FRAC_L_W
FRAC_M_L = FRAC_L_R;                  FRAC_M_R = FRAC_M_L + FRAC_M_W
FRAC_R_L = FRAC_M_R;                  FRAC_R_R = FUNC_R

# =====================================================================
# 4. BLOCK OUTLINES
# =====================================================================
print("[4/16] Block outlines...")
OL = 70
fill_trace_h(FUNC_L, FUNC_T, FUNC_R, OL, DOSE_METAL, PITCH_METAL)
fill_trace_h(FUNC_L, FUNC_B, FUNC_R, OL, DOSE_METAL, PITCH_METAL)
fill_trace_v(FUNC_L, FUNC_B, FUNC_T, OL, DOSE_METAL, PITCH_METAL)
fill_trace_v(FUNC_R, FUNC_B, FUNC_T, OL, DOSE_METAL, PITCH_METAL)
fill_trace_h(FUNC_L, HDIV, FUNC_R, OL, DOSE_METAL, PITCH_METAL)
fill_trace_h(FUNC_L, LOGO_TOP, FUNC_R, OL, DOSE_METAL, PITCH_METAL)

for xd in [SRAM_A_R, HTREE1_R, ALU_R, HTREE2_R]:
    fill_trace_v(xd, HDIV, FUNC_T, OL, DOSE_METAL, PITCH_METAL)
for xd in [FRAC_L_R, FRAC_M_R]:
    fill_trace_v(xd, LOGO_TOP, HDIV, OL, DOSE_METAL, PITCH_METAL)

# =====================================================================
# 5. SRAM-A — regular 6T bit-cell array
# =====================================================================
print("[5/16] SRAM array A...")
M = 250   # margin
sa = dict(l=SRAM_A_L+M, r=SRAM_A_R-M, b=HDIV+M, t=FUNC_T-M)

CELL_W, CELL_H = 160, 100
CELL_PX, CELL_PY = 240, 180
n_cx = int((sa['r'] - sa['l']) / CELL_PX)
n_cy = int((sa['t'] - sa['b']) / CELL_PY)

for ci in range(n_cx):
    for cj in range(n_cy):
        x = sa['l'] + ci * CELL_PX + CELL_PX / 2
        y = sa['b'] + cj * CELL_PY + CELL_PY / 2
        fill_rect(x, y, CELL_W, CELL_H, DOSE_POLY, PITCH_POLY)

# Word lines (width >= 2× beam diameter)
WL_W = max(50, int(diam_for(DOSE_METAL) * 2.5))
for cj in range(n_cy):
    wy = sa['b'] + cj * CELL_PY + CELL_PY / 2 - CELL_H / 2 - 30
    fill_trace_h(sa['l'], wy, sa['r'], WL_W, DOSE_METAL, PITCH_METAL)

# Bit line pairs
BL_W = max(40, int(diam_for(DOSE_METAL) * 2))
for ci in range(0, n_cx, 2):
    bx = sa['l'] + ci * CELL_PX + CELL_PX / 2
    fill_trace_v(bx - 35, sa['b'], sa['t'], BL_W, DOSE_METAL, PITCH_METAL)
    fill_trace_v(bx + 35, sa['b'], sa['t'], BL_W, DOSE_METAL, PITCH_METAL)

# =====================================================================
# 6. SRAM-B — mirror of SRAM-A
# =====================================================================
print("[6/16] SRAM array B...")
sb = dict(l=SRAM_B_L+M, r=SRAM_B_R-M, b=HDIV+M, t=FUNC_T-M)
n_bx = int((sb['r'] - sb['l']) / CELL_PX)
n_by = int((sb['t'] - sb['b']) / CELL_PY)

for ci in range(n_bx):
    for cj in range(n_by):
        x = sb['l'] + ci * CELL_PX + CELL_PX / 2
        y = sb['b'] + cj * CELL_PY + CELL_PY / 2
        fill_rect(x, y, CELL_W, CELL_H, DOSE_POLY, PITCH_POLY)

for cj in range(n_by):
    wy = sb['b'] + cj * CELL_PY + CELL_PY / 2 - CELL_H / 2 - 30
    fill_trace_h(sb['l'], wy, sb['r'], WL_W, DOSE_METAL, PITCH_METAL)

for ci in range(0, n_bx, 2):
    bx = sb['l'] + ci * CELL_PX + CELL_PX / 2
    fill_trace_v(bx - 35, sb['b'], sb['t'], BL_W, DOSE_METAL, PITCH_METAL)
    fill_trace_v(bx + 35, sb['b'], sb['t'], BL_W, DOSE_METAL, PITCH_METAL)

# =====================================================================
# 7. H-TREE CLOCK DISTRIBUTION (fractal) — two blocks
# =====================================================================
print("[7/16] H-tree clock distribution...")
for ht_l, ht_r in [(HTREE1_L, HTREE1_R), (HTREE2_L, HTREE2_R)]:
    hcx = (ht_l + ht_r) / 2
    hcy = (HDIV + FUNC_T) / 2
    htree_points(hcx, hcy, (ht_r - ht_l) / 2 - 300, depth=5,
                 width=80, dwell=DOSE_METAL, pitch=PITCH_METAL)

# =====================================================================
# 8. ALU / DATAPATH
# =====================================================================
print("[8/16] ALU / datapath...")
am = 350
al = dict(l=ALU_L+am, r=ALU_R-am, b=HDIV+am, t=FUNC_T-am)

# Register file — 8 rows of 32-bit cells
REG_W_TOTAL = al['r'] - al['l'] - 600
REG_H = 250
REG_GAP = 120
n_regs = 8
reg_block_h = n_regs * REG_H + (n_regs - 1) * REG_GAP
reg_y0 = al['t'] - 400

for ri in range(n_regs):
    ry = reg_y0 - ri * (REG_H + REG_GAP)
    bit_w = REG_W_TOTAL / 32
    for bit in range(32):
        bx = al['l'] + 300 + bit * bit_w + bit_w / 2
        fill_rect(bx, ry, bit_w - 8, REG_H - 16, DOSE_POLY, PITCH_POLY)
    # Address decoder tap
    fill_rect(al['l'] + 130, ry, 160, REG_H, DOSE_METAL, PITCH_METAL)

# Data bus (width >= 2× beam)
BUS_TRACE_W = max(50, int(diam_for(DOSE_METAL) * 2.5))
bus_y_top = reg_y0 - n_regs * (REG_H + REG_GAP) - 100
for b in range(8):
    by = bus_y_top - b * 100
    fill_trace_h(al['l'] + 150, by, al['r'] - 150, BUS_TRACE_W, DOSE_METAL, PITCH_METAL)

# ALU gate array
alu_arr_t = bus_y_top - 8 * 100 - 150
alu_arr_b = al['b'] + 150
GATE_W, GATE_H = 50, 180
GATE_PX, GATE_PY = 120, 300
n_gx = int((al['r'] - al['l'] - 300) / GATE_PX)
n_gy = max(1, int((alu_arr_t - alu_arr_b) / GATE_PY))

for gi in range(n_gx):
    for gj in range(n_gy):
        gx = al['l'] + 150 + gi * GATE_PX + GATE_PX / 2
        gy = alu_arr_b + gj * GATE_PY + GATE_PY / 2
        fill_rect(gx, gy, GATE_W, GATE_H, DOSE_POLY, PITCH_POLY)

# Deterministic metal routing: every row gets a horizontal rail
ROUTE_W = max(40, int(diam_for(DOSE_METAL) * 2))
for gj in range(n_gy):
    gy = alu_arr_b + gj * GATE_PY + GATE_PY / 2 + GATE_H / 2 + 20
    fill_trace_h(al['l'] + 150, gy, al['r'] - 150, ROUTE_W, DOSE_METAL, PITCH_METAL)

# =====================================================================
# 9. MANDELBROT SET — surreal fractal, rotated 90° CW (left zone)
# =====================================================================
print("[9/16] Mandelbrot set (rotated)...")
mb_margin = 100
mb_l = FRAC_L_L + mb_margin
mb_r = FRAC_L_R - mb_margin
mb_b = LOGO_TOP + mb_margin
mb_t = HDIV - mb_margin
# Shift centre down slightly so the set sits lower in the zone
mb_shift_down = int((mb_t - mb_b) * 0.08)
mb_b -= mb_shift_down
mb_t -= mb_shift_down
# Shift up by 1 µm (1000 nm)
mb_b += 1000
mb_t += 1000
mb_w = mb_r - mb_l
mb_h = mb_t - mb_b

# Map pixel grid to complex plane — classic Mandelbrot window
# Real: [-2.2, 0.8], Imag: [-1.2, 1.2] (centred, full set visible)
re_lo, re_hi = -2.2, 0.8
im_lo, im_hi = -1.2, 1.2

# 90° CW rotation: physical X → imaginary axis, physical Y → -real axis
# Aspect-correct: fit the complex window into the zone
aspect_zone_x = mb_w   # physical x span → maps to imaginary span
aspect_zone_y = mb_h   # physical y span → maps to real span
im_span_needed = im_hi - im_lo
re_span_needed = re_hi - re_lo
scale_im = aspect_zone_x / im_span_needed
scale_re = aspect_zone_y / re_span_needed
scale = min(scale_im, scale_re)
# Adjust spans to fill zone
im_span = aspect_zone_x / scale
re_span = aspect_zone_y / scale
im_center = (im_lo + im_hi) / 2
re_center = (re_lo + re_hi) / 2
im_lo_adj = im_center - im_span / 2
im_hi_adj = im_center + im_span / 2
re_lo_adj = re_center - re_span / 2
re_hi_adj = re_center + re_span / 2

# Multi-scale BOUNDARY fill: centre is empty, shots only on the outside
# edge of the set.  At each scale, find interior pixels that border at
# least one exterior pixel (boundary pixels).  Coarse scales produce
# big boundary shots, fine scales resolve fractal edge detail.

MB_SCALES = [
    # (dwell, max_iter, stacks) – coarsest to finest
    # dwell × 0.01 = diameter in nm.  Max = 16383 (0x3FFF).
    (16000, 150, 3),  # 160 nm
    (10000, 130, 2),  # 100 nm
    ( 6000, 110, 2),  #  60 nm
    ( 3500, 100, 1),  #  35 nm
    ( 2000,  90, 1),  #  20 nm
    ( 1200,  85, 1),  #  12 nm
    (  800,  80, 1),  #   8 nm
    (  500,  80, 1),  #   5 nm — finest wispy tendrils
]

mb_total_shots = 0
for sc_dwell, sc_iter, sc_stacks in MB_SCALES:
    sc_diam = sc_dwell * 0.01        # nm
    sc_pitch = max(int(sc_diam * 0.62), ADDR_GRID)
    sc_nx = max(int(mb_w / sc_pitch), 2)
    sc_ny = max(int(mb_h / sc_pitch), 2)

    # Build complex plane at this resolution
    sc_im = np.linspace(im_lo_adj, im_hi_adj, sc_nx)
    sc_re = np.linspace(re_lo_adj, re_hi_adj, sc_ny)
    SC_IM, SC_RE = np.meshgrid(sc_im, sc_re)
    SC_C = SC_RE + 1j * SC_IM

    # Iterate
    SC_Z = np.zeros_like(SC_C)
    SC_M = np.zeros(SC_C.shape, dtype=np.int32)
    for _ in range(sc_iter):
        alive = np.abs(SC_Z) <= 2.0
        SC_Z[alive] = SC_Z[alive] ** 2 + SC_C[alive]
        SC_M[alive] += 1

    # Interior = never escaped
    inside = (SC_M == sc_iter)

    # Boundary = interior pixel with at least one non-interior neighbour
    # Pad with False so edge pixels count as boundary
    padded = np.pad(inside, 1, mode='constant', constant_values=False)
    boundary = inside & ~(
        padded[:-2, 1:-1] & padded[2:, 1:-1] &   # up & down
        padded[1:-1, :-2] & padded[1:-1, 2:]      # left & right
    )

    rows_sc, cols_sc = np.where(boundary)
    if len(rows_sc) > 0:
        sx = (mb_l + (cols_sc / max(sc_nx - 1, 1)) * mb_w).astype(np.int32)
        sy = (mb_b + (rows_sc / max(sc_ny - 1, 1)) * mb_h).astype(np.int32)
        sx = (sx // ADDR_GRID) * ADDR_GRID
        sy = (sy // ADDR_GRID) * ADDR_GRID
        for _ in range(sc_stacks):
            add(sx, sy, sc_dwell)
        mb_total_shots += len(sx) * sc_stacks

print(f"  Mandelbrot (boundary, multi-scale): {mb_total_shots:,} shots "
      f"across {len(MB_SCALES)} scale levels")

# =====================================================================
# 10a. L-SYSTEM FOREST — top half of centre zone
# =====================================================================
print("[10/16] L-system forest + dragon curve...")
drg_margin = 200
zone_m_l = FRAC_M_L + drg_margin
zone_m_r = FRAC_M_R - drg_margin
zone_m_b = LOGO_TOP + drg_margin
zone_m_t = HDIV - drg_margin
zone_m_mid = (zone_m_b + zone_m_t) / 2  # split into top/bottom halves

# L-system forest: trees growing upward, pushed higher & narrower
TREE_W = max(50, int(diam_for(DOSE_FRACTAL) * 2.5))
forest_floor = zone_m_mid + (zone_m_t - zone_m_mid) * 0.35  # shifted up
n_trees = 7
forest_inset = (zone_m_r - zone_m_l) * 0.12  # narrower spread
tree_spacing = (zone_m_r - zone_m_l - 2 * forest_inset) / (n_trees + 1)
for ti in range(n_trees):
    tx = zone_m_l + forest_inset + (ti + 1) * tree_spacing
    # Vary tree height: taller in centre, shorter at edges
    centre_frac = 1.0 - abs(ti - (n_trees - 1) / 2) / ((n_trees - 1) / 2)
    tree_h = (zone_m_t - forest_floor - 100) * (0.55 + 0.45 * centre_frac) * 0.60
    tree_depth = 6 if centre_frac > 0.5 else 5
    lsystem_tree(tx, forest_floor, tree_h, 1, tree_depth,
                 TREE_W, DOSE_FRACTAL, PITCH_FRACTAL, shrink=0.65)

# =====================================================================
# 10b. DRAGON CURVE — rotated 90° CW, bottom half of centre zone
# =====================================================================
drg_half_h = zone_m_mid - zone_m_b - 100  # bottom half height
drg_half_w = zone_m_r - zone_m_l
drg_sz = min(drg_half_w, drg_half_h) * 1.40  # nearly 2× wider
drg_cx = (zone_m_l + zone_m_r) / 2
drg_cy = zone_m_b + drg_half_h / 2

dpts = dragon_curve_pts(order=14, size=drg_sz)
dpts_arr = np.array(dpts)
# Rotate 90° CW: (x, y) → (y, -x)
rotated_x = dpts_arr[:, 1].copy()
rotated_y = -dpts_arr[:, 0].copy()
# Re-centre into [0, drg_sz] range
rotated_x -= rotated_x.min()
rotated_y -= rotated_y.min()
rx_span = rotated_x.max() - rotated_x.min()
ry_span = rotated_y.max() - rotated_y.min()
if rx_span > 0:
    rotated_x = rotated_x / rx_span * drg_sz
if ry_span > 0:
    rotated_y = rotated_y / ry_span * drg_sz

drg_ox = drg_cx - drg_sz / 2
drg_oy = drg_cy - drg_sz / 2

# Interpolate along each segment for continuous dense coverage
dx0 = rotated_x[:-1] + drg_ox
dy0 = rotated_y[:-1] + drg_oy
dx1 = rotated_x[1:] + drg_ox
dy1 = rotated_y[1:] + drg_oy
seg_lens_d = np.abs(dx1 - dx0) + np.abs(dy1 - dy0)
n_steps_d = np.maximum((seg_lens_d / PITCH_FRACTAL).astype(int), 1)

all_dx, all_dy = [], []
for i in range(len(dx0)):
    ts = np.linspace(0, 1, n_steps_d[i] + 1)
    all_dx.append(dx0[i] + (dx1[i] - dx0[i]) * ts)
    all_dy.append(dy0[i] + (dy1[i] - dy0[i]) * ts)
drg_x = np.concatenate(all_dx)
drg_y = np.concatenate(all_dy)
mask_d = ((drg_x >= zone_m_l) & (drg_x <= zone_m_r) &
          (drg_y >= zone_m_b) & (drg_y <= zone_m_t))
add(drg_x[mask_d], drg_y[mask_d], DOSE_FRACTAL)

# =====================================================================
# 11. HILBERT CURVE — surreal space-filling fractal (right zone)
# =====================================================================
print("[11/16] Hilbert curve...")
hilb_margin = 400
hilb_sz = min(FRAC_R_R - FRAC_R_L, HDIV - LOGO_TOP) - 2 * hilb_margin
hilb_cx = (FRAC_R_L + FRAC_R_R) / 2
hilb_cy = (LOGO_TOP + HDIV) / 2
hilb_ox = hilb_cx - hilb_sz / 2
hilb_oy = hilb_cy - hilb_sz / 2

hpts = hilbert_curve(order=6, size=hilb_sz)
hpts_arr = np.array(hpts)

# Interpolate along each segment for continuous dense coverage
hx0 = hpts_arr[:-1, 0] + hilb_ox
hy0 = hpts_arr[:-1, 1] + hilb_oy
hx1 = hpts_arr[1:, 0] + hilb_ox
hy1 = hpts_arr[1:, 1] + hilb_oy
seg_lens = np.abs(hx1 - hx0) + np.abs(hy1 - hy0)
n_steps = np.maximum((seg_lens / PITCH_FRACTAL).astype(int), 1)

all_hx, all_hy = [], []
for i in range(len(hx0)):
    ts = np.linspace(0, 1, n_steps[i] + 1)
    all_hx.append(hx0[i] + (hx1[i] - hx0[i]) * ts)
    all_hy.append(hy0[i] + (hy1[i] - hy0[i]) * ts)
hilb_x = np.concatenate(all_hx)
hilb_y = np.concatenate(all_hy)
mask = ((hilb_x >= FRAC_R_L + hilb_margin) &
        (hilb_x <= FRAC_R_R - hilb_margin) &
        (hilb_y >= LOGO_TOP + hilb_margin) &
        (hilb_y <= HDIV - hilb_margin))
add(hilb_x[mask], hilb_y[mask], DOSE_FRACTAL)

# =====================================================================
# 12. ZONE SEAM CONTACTS — tie fractals to chip at boundaries
# =====================================================================
print("[12/16] Zone seam contacts...")
# Small contact pads along each zone divider, like the chip's via
# arrays are stitching into the fractal world
SEAM_VIA = max(80, int(diam_for(DOSE_CONTACT) * 2.5))
for xd in [FRAC_L_R, FRAC_M_R]:
    n_vias = 6
    vy_span = (HDIV - LOGO_TOP) * 0.6
    vy0 = LOGO_TOP + (HDIV - LOGO_TOP) * 0.2
    for vi in range(n_vias):
        vy = vy0 + vi * vy_span / (n_vias - 1)
        fill_contact(xd, vy, SEAM_VIA, DOSE_CONTACT, PITCH_CONTACT)

# =====================================================================
# 13. (reserved)
# =====================================================================

# =====================================================================
# 14. HORIZONTAL BUS at divider
# =====================================================================
print("[14/16] Horizontal bus...")
HBUS_W = max(45, int(diam_for(DOSE_METAL) * 2.5))
HBUS_VIA = max(55, int(diam_for(DOSE_CONTACT) * 2.5))
for bi in range(6):
    by = HDIV - 180 - bi * 90
    fill_trace_h(FUNC_L + 80, by, FUNC_R - 80, HBUS_W, DOSE_METAL, PITCH_METAL)

# Contact arrays at block boundaries
for xd in [FRAC_L_R, FRAC_M_R, SRAM_A_R, ALU_R]:
    for bi in range(6):
        fill_contact(xd, HDIV - 180 - bi * 90, HBUS_VIA, DOSE_CONTACT, PITCH_CONTACT)

# =====================================================================
# 15. VERTICAL BUSES connecting upper ↔ lower via H-tree blocks
# =====================================================================
print("[15/16] Vertical interconnect...")
for ht_l, ht_r in [(HTREE1_L, HTREE1_R), (HTREE2_L, HTREE2_R)]:
    bus_cx = (ht_l + ht_r) / 2
    n_bl = 6
    bsp = (ht_r - ht_l) / (n_bl + 1)
    VBUS_W = max(45, int(diam_for(DOSE_METAL) * 2.5))
    VBUS_VIA = max(80, int(diam_for(DOSE_CONTACT) * 3))
    for bi in range(n_bl):
        bx = ht_l + (bi + 1) * bsp
        fill_trace_v(bx, HDIV + 150, FUNC_T - 150, VBUS_W, DOSE_METAL, PITCH_METAL)
        fill_contact(bx, HDIV + 300, VBUS_VIA, DOSE_CONTACT, PITCH_CONTACT)
        fill_contact(bx, FUNC_T - 300, VBUS_VIA, DOSE_CONTACT, PITCH_CONTACT)

# =====================================================================
# 16. MULTIBEAM LOGO in clear zone at bottom
# =====================================================================
print("[16/16] Multibeam logo...")
LOGO_CX = (FUNC_L + FUNC_R) / 2
LOGO_CY = (FUNC_B + LOGO_TOP) / 2
logo_avail_w = FUNC_R - FUNC_L - 1000
logo_avail_h = LOGO_TOP - FUNC_B - 600

if HAS_PIL and LOGO_PATH.exists():
    img = Image.open(LOGO_PATH)
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    img = img.convert('L')

    aspect = img.height / img.width
    target_w = min(logo_avail_w, int(logo_avail_h / aspect))
    target_h = int(target_w * aspect)
    logo_pitch = PITCH_LOGO
    px_w = max(target_w // logo_pitch, 10)
    px_h = max(target_h // logo_pitch, 10)
    img = img.resize((px_w, px_h), Image.LANCZOS)
    arr = np.array(img, dtype=np.float64)

    # Hard binary threshold — clean edges
    mask = (255.0 - arr) > 100
    rows, cols = np.where(mask)

    logo_ox = LOGO_CX - target_w / 2
    logo_oy = LOGO_CY - target_h / 2
    lx = (logo_ox + cols * logo_pitch).astype(np.int32)
    ly = (logo_oy + (px_h - rows) * logo_pitch).astype(np.int32)
    add(lx, ly, DOSE_LOGO)
    print(f"  Logo: {len(lx):,} shots")

    # --- 80s speed lines flanking the logo ---
    # Horizontal streaks radiating outward from logo edges,
    # thickest near the logo, tapering thinner as they reach the die edge.
    logo_left  = logo_ox
    logo_right = logo_ox + target_w
    accent_gap = 80
    edge_l = FUNC_L + 200   # left die edge
    edge_r = FUNC_R - 200   # right die edge

    # Stack of lines centred on logo, fanning vertically
    n_lines = 9
    spread = target_h * 0.55   # total vertical spread of the line bundle
    for li in range(n_lines):
        # y position: evenly spaced, symmetrical around LOGO_CY
        frac = (li / (n_lines - 1)) - 0.5   # -0.5 … +0.5
        ly = LOGO_CY + frac * spread * 2

        # Line thickness: thickest at centre (li=4), thinner at edges
        dist_from_centre = abs(frac)
        lw = max(int(80 * (1.0 - dist_from_centre * 1.4)), 20)

        # Left streak: from die edge to just before logo
        fill_trace_h(edge_l, ly, logo_left - accent_gap, lw,
                     DOSE_ALIGN, PITCH_ALIGN)
        # Right streak: from just after logo to die edge
        fill_trace_h(logo_right + accent_gap, ly, edge_r, lw,
                     DOSE_ALIGN, PITCH_ALIGN)
else:
    print("  Logo skipped")

# =====================================================================
# MERGE, CLAMP, WRITE
# =====================================================================
print("\nMerging shots...")
all_x = np.concatenate([s[0] for s in shots]).astype(np.int64)
all_y = np.concatenate([s[1] for s in shots]).astype(np.int64)
all_dwell = np.concatenate([s[2] for s in shots]).astype(np.int64)

np.clip(all_x, 0, 65535, out=all_x)
np.clip(all_y, 0, 0xFFFFFFFF, out=all_y)
np.clip(all_dwell, 1, MAX_DWELL, out=all_dwell)

n = len(all_x)
print(f"Total shots: {n:,}")

# v3 header
header = struct.pack(
    "<IHHiiIIdHHdiQQ",
    0, 3, 0,
    -HALF, -HALF,
    FIELD, FIELD,
    1.0, 0, 0, 0.0, 0,
    n, 0,
)
assert len(header) == 64

print("Packing binary records...")
records = np.zeros(n, dtype=np.uint64)
records |= (all_dwell.astype(np.uint64) & 0x3FFF) << np.uint64(2)
records |= (all_x.astype(np.uint64) & 0xFFFF) << np.uint64(16)
records |= (all_y.astype(np.uint64) & 0xFFFFFFFF) << np.uint64(32)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, 'wb') as f:
    f.write(header)
    f.write(records.tobytes())

sz = OUTPUT.stat().st_size
print(f"\nWritten: {OUTPUT}")
print(f"File size: {sz / 1024 / 1024:.1f} MB")
print(f"Shots: {n:,}")
print(f"Dwell range: {int(all_dwell.min())}-{int(all_dwell.max())} ns")
print(f"X range: {int(all_x.min())}-{int(all_x.max())} nm")
print(f"Y range: {int(all_y.min())}-{int(all_y.max())} nm")
