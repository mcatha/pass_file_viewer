"""
Generate a .pass v3 file with ~1M shots forming complex geometries
in a 30x30 micron grid centered on the origin.

Geometries included:
  - Concentric circles (varying dwell radially)
  - Spiral arms (log-spiral, high dwell)
  - Grid of small squares (checkerboard pattern)
  - Star / polygon outlines
  - Dense filled rectangles with dwell gradient
  - Scattered random shots filling gaps

Dwell times range from 0 to 20 us (0-20000 ns, stored as 14-bit field
with max 16383 — we clamp to that).

Coordinate system: X is uint16 (0-65535), Y is uint32.
Resolution = 1 nm/unit.  30 um = 30000 nm.
Center at origin → stripeOriginX/Y offset so (0,0) maps to the center.
We'll place shots in the range [0, 30000] for both X and Y in file coords,
with the header origin set to -15000 so the viewer sees (-15000..+15000).
"""

import struct
import numpy as np
from pathlib import Path

# ── parameters ──────────────────────────────────────────────────────
OUTPUT = Path(__file__).parent.parent / "complex_geometry_1M.pass"
TARGET_SHOTS = 1_000_000
GRID_SIZE = 30000  # 30 um in nm
CENTER = GRID_SIZE // 2  # 15000

# 14-bit dwell max = 16383;  20 us = 20000 ns → clamp
MAX_DWELL_NS = 16383  # hardware limit (14 bits)
DWELL_20US = 20000

def clamp_dwell(d):
    return np.clip(d, 0, MAX_DWELL_NS).astype(np.uint16)

rng = np.random.default_rng(42)

shots = []  # list of (x, y, dwell) tuples as arrays


# ── 1. Concentric circles with radial dwell gradient ────────────────
print("Generating concentric circles...")
n_rings = 60
for i in range(n_rings):
    r = 1000 + i * 200  # radius from 1000 to 12800 nm
    circumference = 2 * np.pi * r
    n_pts = max(int(circumference / 8), 50)  # ~8 nm spacing
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    x = CENTER + (r * np.cos(theta)).astype(np.int32)
    y = CENTER + (r * np.sin(theta)).astype(np.int32)
    # Dwell increases with radius: 500 ns at center to 15000 ns at edge
    dwell = np.full(n_pts, 500 + int(14500 * i / n_rings), dtype=np.int32)
    shots.append((x, y, dwell))

# ── 2. Logarithmic spiral arms ──────────────────────────────────────
print("Generating spiral arms...")
n_arms = 5
for arm in range(n_arms):
    t = np.linspace(0, 6 * np.pi, 8000)
    a, b = 200, 0.12
    r = a * np.exp(b * t)
    mask = r < 14000
    t, r = t[mask], r[mask]
    angle_offset = arm * 2 * np.pi / n_arms
    x = CENTER + (r * np.cos(t + angle_offset)).astype(np.int32)
    y = CENTER + (r * np.sin(t + angle_offset)).astype(np.int32)
    # High dwell on spirals: 12000-18000 ns
    dwell = (12000 + 6000 * np.sin(t * 3) ** 2).astype(np.int32)
    shots.append((x, y, dwell))

# ── 3. Star polygon outlines ────────────────────────────────────────
print("Generating star polygons...")
for n_points, radius, dwell_val in [(5, 8000, 10000), (7, 10000, 8000), (12, 12000, 6000)]:
    vertices = []
    for i in range(n_points * 2):
        angle = i * np.pi / n_points - np.pi / 2
        r = radius if i % 2 == 0 else radius * 0.4
        vertices.append((CENTER + int(r * np.cos(angle)),
                          CENTER + int(r * np.sin(angle))))
    vertices.append(vertices[0])  # close
    # Interpolate between vertices
    for j in range(len(vertices) - 1):
        x0, y0 = vertices[j]
        x1, y1 = vertices[j + 1]
        dist = max(abs(x1 - x0), abs(y1 - y0))
        n_pts = max(dist // 6, 2)  # ~6 nm spacing
        t = np.linspace(0, 1, n_pts)
        x = (x0 + t * (x1 - x0)).astype(np.int32)
        y = (y0 + t * (y1 - y0)).astype(np.int32)
        dwell = np.full(n_pts, dwell_val, dtype=np.int32)
        shots.append((x, y, dwell))

# ── 4. Checkerboard of small filled squares ─────────────────────────
print("Generating checkerboard pattern...")
sq_size = 400  # 400 nm squares
spacing = 800   # 800 nm pitch
n_cells = GRID_SIZE // spacing
for ix in range(n_cells):
    for iy in range(n_cells):
        if (ix + iy) % 2 != 0:
            continue
        # Fill square with ~6 nm grid
        cx = ix * spacing + spacing // 2
        cy = iy * spacing + spacing // 2
        # Only fill if outside the main circle area (avoid overlap)
        dist_from_center = np.sqrt((cx - CENTER) ** 2 + (cy - CENTER) ** 2)
        if dist_from_center < 13500:
            continue
        gx = np.arange(cx - sq_size // 2, cx + sq_size // 2, 12)
        gy = np.arange(cy - sq_size // 2, cy + sq_size // 2, 12)
        xx, yy = np.meshgrid(gx, gy)
        xx, yy = xx.ravel(), yy.ravel()
        # Gradient dwell: 2000-8000 based on position
        dwell = (2000 + 6000 * (xx - cx + sq_size // 2) / sq_size).astype(np.int32)
        shots.append((xx, yy, dwell))

# ── 5. Dense filled rectangles with dwell gradient ──────────────────
print("Generating filled rectangles...")
rects = [
    (1000, 1000, 5000, 3000),    # (x, y, w, h) in file coords
    (24000, 1000, 5000, 3000),
    (1000, 26000, 5000, 3000),
    (24000, 26000, 5000, 3000),
]
for rx, ry, rw, rh in rects:
    gx = np.arange(rx, rx + rw, 10)
    gy = np.arange(ry, ry + rh, 10)
    xx, yy = np.meshgrid(gx, gy)
    xx, yy = xx.ravel(), yy.ravel()
    # Dwell gradient across rectangle: 0 to 16000 ns
    t = (xx - rx) / rw
    dwell = (t * 16000).astype(np.int32)
    shots.append((xx, yy, dwell))

# ── 6. Text-like block letters "PASS" ───────────────────────────────
print("Generating block letters...")
# Simple 5x7 bitmap font for P, A, S, S
font = {
    'P': [
        "####.",
        "#...#",
        "#...#",
        "####.",
        "#....",
        "#....",
        "#....",
    ],
    'A': [
        ".###.",
        "#...#",
        "#...#",
        "#####",
        "#...#",
        "#...#",
        "#...#",
    ],
    'S': [
        ".####",
        "#....",
        "#....",
        ".###.",
        "....#",
        "....#",
        "####.",
    ],
}
text = "PASS"
letter_w, letter_h = 5, 7
pixel_size = 200  # each "pixel" = 200x200 nm block
gap = 100  # between letters
total_w = len(text) * (letter_w * pixel_size + gap) - gap
start_x = CENTER - total_w // 2
start_y = CENTER + 5500  # above center

for ci, ch in enumerate(text):
    bitmap = font[ch]
    ox = start_x + ci * (letter_w * pixel_size + gap)
    for row_i, row in enumerate(bitmap):
        for col_i, c in enumerate(row):
            if c == '#':
                px = ox + col_i * pixel_size
                py = start_y - row_i * pixel_size
                gx = np.arange(px, px + pixel_size, 15)
                gy = np.arange(py, py + pixel_size, 15)
                xx, yy = np.meshgrid(gx, gy)
                xx, yy = xx.ravel(), yy.ravel()
                dwell = np.full(len(xx), 18000, dtype=np.int32)
                shots.append((xx, yy, dwell))

# ── 7. Random scatter to fill to 1M ────────────────────────────────
current = sum(len(s[0]) for s in shots)
print(f"Geometry shots so far: {current:,}")
remaining = TARGET_SHOTS - current
if remaining > 0:
    print(f"Adding {remaining:,} random scatter shots...")
    x = rng.integers(0, GRID_SIZE, size=remaining).astype(np.int32)
    y = rng.integers(0, GRID_SIZE, size=remaining).astype(np.int32)
    dwell = rng.integers(0, 20001, size=remaining).astype(np.int32)
    shots.append((x, y, dwell))

# ── Merge and clamp ─────────────────────────────────────────────────
all_x = np.concatenate([s[0] for s in shots]).astype(np.int64)
all_y = np.concatenate([s[1] for s in shots]).astype(np.int64)
all_dwell = np.concatenate([s[2] for s in shots]).astype(np.int64)

# Clamp to valid ranges
np.clip(all_x, 0, 65535, out=all_x)       # uint16
np.clip(all_y, 0, 0xFFFFFFFF, out=all_y)  # uint32
np.clip(all_dwell, 0, MAX_DWELL_NS, out=all_dwell)

# Trim or pad to exactly TARGET_SHOTS
if len(all_x) > TARGET_SHOTS:
    idx = rng.choice(len(all_x), TARGET_SHOTS, replace=False)
    idx.sort()
    all_x, all_y, all_dwell = all_x[idx], all_y[idx], all_dwell[idx]

n = len(all_x)
print(f"Total shots: {n:,}")

# ── Build v3 header ─────────────────────────────────────────────────
header = struct.pack(
    "<IHHiiIIdHHdiQQ",
    0,          # stripeSymbol
    3,          # stripeDataVersion
    0,          # stripeNumber
    -15000,     # stripeOriginX  (center the 30um grid on origin)
    -15000,     # stripeOriginY
    30000,      # stripeWidth
    30000,      # stripeLength
    1.0,        # resolution (1 nm/unit)
    0,          # bss
    0,          # subFieldHeight
    0.0,        # maxStageSpeed
    0,          # sortDirection
    n,          # shotCount
    0,          # shapeCount
)
assert len(header) == 64

# ── Build 8-byte shot records ───────────────────────────────────────
print("Packing binary records...")
# mID=0 (bits 0-1), mDwell (bits 2-15), mX (bits 16-31), mY (bits 32-63)
records = np.zeros(n, dtype=np.uint64)
records |= (all_dwell.astype(np.uint64) & 0x3FFF) << np.uint64(2)
records |= (all_x.astype(np.uint64) & 0xFFFF) << np.uint64(16)
records |= (all_y.astype(np.uint64) & 0xFFFFFFFF) << np.uint64(32)

# ── Write file ──────────────────────────────────────────────────────
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, 'wb') as f:
    f.write(header)
    f.write(records.tobytes())

file_size = OUTPUT.stat().st_size
print(f"Written: {OUTPUT}")
print(f"File size: {file_size / 1024 / 1024:.1f} MB")
print(f"Dwell range: {int(all_dwell.min())}-{int(all_dwell.max())} ns")
print(f"X range: {int(all_x.min())}-{int(all_x.max())}")
print(f"Y range: {int(all_y.min())}-{int(all_y.max())}")
