"""Generate MB logo pass files for MB300.

Logo (mb-logo-w-tag.png, 600×145 px) is scaled to 250 mm × 60.4 mm and
centred on the wafer.  One embedded-v4-header .pass file is written per
active (column, master-pass) combination.

Column layout
-------------
The number suffix (1–4) gives the X position; the letter (A–E) gives Y.

  B/C/D rows (4 beams each):  X = +112.5, +37.5, −37.5, −112.5 mm
  A/E rows   (3 beams each):  X = +75, 0, −75 mm
  Y positions: A=+130, B=+65, C=0, D=−65, E=−130 mm

Cell boundaries:
  X: midpoints between adjacent columns → ±150 mm outer (B/C/D), ±112.5 mm (A/E)
  Y: midpoints between adjacent rows    → ±162.5 mm outer, ±97.5, ±32.5 mm inner
All cells are 75 mm wide × 65 mm tall.

Only C-row cells (Y section ±32.5 mm) overlap the logo's ±30.2 mm Y extent.
Outer B/C/D columns (col 1 and col 4) are clamped to ±125 mm logo edge in X.

Stage motion
------------
- Stage steps 60 µm in X between passes.
- For each X position the beam sweeps in Y over the column's full Y section.
- Serpentine: odd-numbered passes scan −Y; even passes scan +Y.
"""

import struct
import math as _math
import numpy as np
from PIL import Image
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE   = Path(__file__).parent
_LOGO   = _HERE.parent / "mb-logo-w-tag.png"
OUT_DIR = _HERE.parent / "logo_passes"

# ── Physical parameters (nm) ──────────────────────────────────────────────────
# Scale logo so its corners fall exactly 1 mm inside the 300 mm wafer edge.
# Logo pixel aspect: 600 × 145.  Corner radius = sqrt((W/2)² + (H/2)²) = 149 mm.
_CORNER_R_NM   = 149_000_000   # 150 mm wafer radius − 1 mm margin
_LOGO_DIAG_PX  = _math.sqrt(600**2 + 145**2)
LOGO_WIDTH_NM  = round(2 * _CORNER_R_NM * 600 / _LOGO_DIAG_PX)
LOGO_HEIGHT_NM = round(LOGO_WIDTH_NM * 145 / 600)

LOGO_X_MIN = -(LOGO_WIDTH_NM  // 2)   # −125_000_000
LOGO_X_MAX =   LOGO_WIDTH_NM  // 2    # +125_000_000
LOGO_Y_MIN = -(LOGO_HEIGHT_NM // 2)   # ≈ −30_208_333
LOGO_Y_MAX =   LOGO_HEIGHT_NM + LOGO_Y_MIN   # ≈ +30_208_334

PITCH_NM      = 1100     # shot pitch
PASS_WIDTH_NM = 60_000   # 60 µm stage X step
DWELL_NS      = 16_383   # 14-bit maximum
HALF          = PITCH_NM // 2

# ── MB300 columns (from viewer_widget._MB300_FIDUCIALS) ──────────────────────
# Each entry: (name, beam_X_nm, x_sec_start, x_sec_end, y_sec_start, y_sec_end)
#
# All 18 columns have equal cell size: 75 mm wide × 65 mm tall.
#
# X section boundaries (midpoints between adjacent X positions):
#   B/C/D: ±75, ±150 mm outer — col 1 and col 4 clamped to logo ±125 mm
#   A/E:   ±37.5, ±112.5 mm (no clamping needed)
# Y section boundaries (midpoints between adjacent Y positions):
#   ±32.5, ±97.5, ±162.5 mm
BEAM_COLUMNS = [
    # name   beam_X            x_sec_start       x_sec_end        y_sec_start      y_sec_end
    # A row (Y = +132 mm); 3 beams at X = +75.1, 0, −75.1 mm
    ('A2',  75_100_000,   37_550_000,  112_650_000,   99_000_000,  165_000_000),
    ('A3',           0,  -37_550_000,   37_550_000,   99_000_000,  165_000_000),
    ('A4', -75_100_000, -112_650_000,  -37_550_000,   99_000_000,  165_000_000),
    # B row (Y = +66 mm); 4 beams at X = +112.65, +37.55, −37.55, −112.65 mm
    ('B1',  112_650_000,  75_100_000,  LOGO_X_MAX,    33_000_000,   99_000_000),
    ('B2',   37_550_000,           0,  75_100_000,    33_000_000,   99_000_000),
    ('B3',  -37_550_000,  -75_100_000,          0,    33_000_000,   99_000_000),
    ('B4', -112_650_000,  LOGO_X_MIN,  -75_100_000,   33_000_000,   99_000_000),
    # C row (Y = 0 mm); 4 beams — only row that overlaps the logo Y extent
    ('C1',  112_650_000,  75_100_000,  LOGO_X_MAX,   -33_000_000,   33_000_000),
    ('C2',   37_550_000,           0,  75_100_000,   -33_000_000,   33_000_000),
    ('C3',  -37_550_000,  -75_100_000,          0,   -33_000_000,   33_000_000),
    ('C4', -112_650_000,  LOGO_X_MIN,  -75_100_000,  -33_000_000,   33_000_000),
    # D row (Y = −66 mm); 4 beams
    ('D1',  112_650_000,  75_100_000,  LOGO_X_MAX,   -99_000_000,  -33_000_000),
    ('D2',   37_550_000,           0,  75_100_000,   -99_000_000,  -33_000_000),
    ('D3',  -37_550_000,  -75_100_000,          0,   -99_000_000,  -33_000_000),
    ('D4', -112_650_000,  LOGO_X_MIN,  -75_100_000,  -99_000_000,  -33_000_000),
    # E row (Y = −132 mm); 3 beams at X = +75.1, 0, −75.1 mm
    ('E2',  75_100_000,   37_550_000,  112_650_000,  -165_000_000,  -99_000_000),
    ('E3',           0,  -37_550_000,   37_550_000,  -165_000_000,  -99_000_000),
    ('E4', -75_100_000, -112_650_000,  -37_550_000,  -165_000_000,  -99_000_000),
]

# ── Master pass sweep ─────────────────────────────────────────────────────────
# All 18 columns are active for the same P_X range: −37.55 mm to +37.55 mm.
P_X_FIRST = -37_550_000
N_MASTER   = 1_252   # ceil(75_100_000 / 60_000)

# ── Image preprocessing ───────────────────────────────────────────────────────
_img  = Image.open(_LOGO).convert("RGBA")
_arr  = np.array(_img)   # (145, 600, 4)
_lum  = (0.299 * _arr[:, :, 0].astype(np.float32)
       + 0.587 * _arr[:, :, 1].astype(np.float32)
       + 0.114 * _arr[:, :, 2].astype(np.float32)) / 255.0
_alp  = _arr[:, :, 3].astype(np.float32) / 255.0
_dark = (_lum * _alp) < 0.5
DARK_MASK = _dark[::-1, :]   # row 0 → bottom of logo on wafer
H_PX, W_PX = DARK_MASK.shape

# ── Pre-compute Y grids for each distinct Y section ───────────────────────────
# Keyed by (y_sec_start, y_sec_end).
_y_grids: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
for _, _, _, _, ys, ye in BEAM_COLUMNS:
    key = (ys, ye)
    if key in _y_grids:
        continue
    sec_len = ye - ys
    Y_LOC = np.arange(HALF, sec_len, PITCH_NM, dtype=np.int64)
    # wafer_Y = y_local + ys  →  image row = (wafer_Y − LOGO_Y_MIN) × H_PX / LOGO_HEIGHT_NM
    IY_raw = (Y_LOC + ys - LOGO_Y_MIN) * H_PX // LOGO_HEIGHT_NM
    valid_y = (IY_raw >= 0) & (IY_raw < H_PX)
    IY_ = np.clip(IY_raw, 0, H_PX - 1).astype(np.int32)
    _y_grids[key] = (Y_LOC, IY_, valid_y)

# ── v4 header (78 bytes, little-endian) ───────────────────────────────────────
_HDR_FMT = "<IHHiiIIdHHdiQQQI??"

def _pack_header(seq_n: int, origin_x: int, origin_y: int,
                 stripe_w: int, stripe_len: int, shot_count: int) -> bytes:
    sort_dir = -1 if seq_n % 2 == 1 else 1
    return struct.pack(
        _HDR_FMT,
        0xB3D11982, 4, seq_n & 0xFFFF,
        origin_x, origin_y,
        stripe_w, stripe_len,
        1.0, PITCH_NM, 0, 0.0, sort_dir,
        shot_count, 0, 0,
        DWELL_NS, False, False,
    )

# ── Main loop ─────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

_dw = np.uint64(DWELL_NS)
total_shots = 0
total_files = 0

print(f"Logo: {W_PX}×{H_PX} px → "
      f"{LOGO_WIDTH_NM/1e6:.0f} mm × {LOGO_HEIGHT_NM/1e6:.2f} mm")
print(f"Pitch: {PITCH_NM} nm  Dwell: {DWELL_NS} ns  "
      f"Passes: {N_MASTER} × {len(BEAM_COLUMNS)} beams")
print(f"Output: {OUT_DIR}\n")

for n in range(1, N_MASTER + 1):
    P_X = P_X_FIRST + (n - 1) * PASS_WIDTH_NM

    for name, beam_x, xs_start, xs_end, ys_start, ys_end in BEAM_COLUMNS:
        x_pass_abs = beam_x + P_X
        x_start    = max(x_pass_abs, xs_start)
        x_end      = min(x_pass_abs + PASS_WIDTH_NM, xs_end)
        if x_end <= x_start:
            continue

        x_local = np.arange(HALF, x_end - x_start, PITCH_NM, dtype=np.int64)
        if len(x_local) == 0:
            continue

        x_abs = x_start + x_local
        IX = np.clip(
            (x_abs - LOGO_X_MIN) * W_PX // LOGO_WIDTH_NM,
            0, W_PX - 1,
        ).astype(np.int32)

        Y_LOCAL, IY, valid_y = _y_grids[(ys_start, ys_end)]
        hit = DARK_MASK[IY[:, None], IX[None, :]]
        hit[~valid_y, :] = False

        sy, sx = np.nonzero(hit)
        if len(sy) == 0:
            continue

        if n % 2 == 1:
            order = np.argsort(-Y_LOCAL[sy], kind='stable')
        else:
            order = np.argsort( Y_LOCAL[sy], kind='stable')

        xl = x_local[sx[order]].astype(np.uint64)
        yl = Y_LOCAL[sy[order]].astype(np.uint64)
        records = (_dw << np.uint64(2)) | (xl << np.uint64(16)) | (yl << np.uint64(32))

        fname = OUT_DIR / f"{name}_{n:04d}.pass"
        with open(fname, "wb") as f:
            f.write(_pack_header(
                n, x_start, ys_start,
                x_end - x_start, ys_end - ys_start,
                len(sy),
            ))
            f.write(records.tobytes())

        total_shots += len(sy)
        total_files += 1

    if n % 100 == 0 or n == N_MASTER:
        print(f"  pass {n:4d}/{N_MASTER}  "
              f"files {total_files:5,}  "
              f"shots {total_shots:>14,.0f}  "
              f"data {total_shots * 8 / 1e9:6.1f} GB")

print(f"\nDone: {total_files:,} files  "
      f"{total_shots:,} shots  "
      f"{total_shots * 8 / 1e9:.1f} GB  →  {OUT_DIR}")
