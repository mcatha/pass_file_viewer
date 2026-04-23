"""Generate MB logo pass files for MB300.

Logo (mb-logo-w-tag.png, 600×145 px) is scaled to 250 mm × 60.4 mm and
centred on the wafer.  One embedded-v4-header .pass file is written per
active (column, master-pass) combination.

Each MB300 column has a rectangular area of responsibility:
  - X section: midpoints between adjacent column X positions
  - Y section: midpoints between adjacent column Y positions (within its row)
All 18 columns have equal cell size (65 mm wide × 75 mm tall).

Stage motion
------------
- Stage steps 60 µm in X between passes.
- For each X position the stage sweeps in Y over the column's Y section.
- Serpentine: odd-numbered passes scan −Y; even passes scan +Y.
"""

import struct
import numpy as np
from PIL import Image
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE   = Path(__file__).parent
_LOGO   = _HERE.parent / "mb-logo-w-tag.png"
OUT_DIR = _HERE.parent / "logo_passes"

# ── Physical parameters (nm) ──────────────────────────────────────────────────
LOGO_WIDTH_NM  = 250_000_000
LOGO_HEIGHT_NM = round(LOGO_WIDTH_NM * 145 / 600)   # ≈ 60_416_667

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
# All 18 columns have equal cell size: 65 mm wide × 75 mm tall.
#
# X section boundaries (midpoints between adjacent X positions):
#   ±162.5 mm (outer edge), ±97.5, ±32.5 mm — clamped to logo at ±125 mm
# Y section boundaries (midpoints between adjacent Y positions per column):
#   B/C/D rows: ±150, ±75, 0 mm
#   A/E rows:   ±112.5, ±37.5 mm
BEAM_COLUMNS = [
    # name   beam_X          x_sec_start     x_sec_end      y_sec_start    y_sec_end
    # A column (X = −130 mm); X section clamped to logo left edge
    ('A2', -130_000_000,  LOGO_X_MIN,     -97_500_000,    37_500_000,   112_500_000),
    ('A3', -130_000_000,  LOGO_X_MIN,     -97_500_000,   -37_500_000,    37_500_000),
    ('A4', -130_000_000,  LOGO_X_MIN,     -97_500_000,  -112_500_000,   -37_500_000),
    # B column (X = −65 mm)
    ('B1',  -65_000_000,  -97_500_000,    -32_500_000,    75_000_000,   150_000_000),
    ('B2',  -65_000_000,  -97_500_000,    -32_500_000,             0,    75_000_000),
    ('B3',  -65_000_000,  -97_500_000,    -32_500_000,   -75_000_000,             0),
    ('B4',  -65_000_000,  -97_500_000,    -32_500_000,  -150_000_000,   -75_000_000),
    # C column (X = 0)
    ('C1',            0,  -32_500_000,     32_500_000,    75_000_000,   150_000_000),
    ('C2',            0,  -32_500_000,     32_500_000,             0,    75_000_000),
    ('C3',            0,  -32_500_000,     32_500_000,   -75_000_000,             0),
    ('C4',            0,  -32_500_000,     32_500_000,  -150_000_000,   -75_000_000),
    # D column (X = +65 mm)
    ('D1',   65_000_000,   32_500_000,     97_500_000,    75_000_000,   150_000_000),
    ('D2',   65_000_000,   32_500_000,     97_500_000,             0,    75_000_000),
    ('D3',   65_000_000,   32_500_000,     97_500_000,   -75_000_000,             0),
    ('D4',   65_000_000,   32_500_000,     97_500_000,  -150_000_000,   -75_000_000),
    # E column (X = +130 mm); X section clamped to logo right edge
    ('E2',  130_000_000,   97_500_000,    LOGO_X_MAX,    37_500_000,   112_500_000),
    ('E3',  130_000_000,   97_500_000,    LOGO_X_MAX,   -37_500_000,    37_500_000),
    ('E4',  130_000_000,   97_500_000,    LOGO_X_MAX,  -112_500_000,   -37_500_000),
]

# ── Master pass sweep ─────────────────────────────────────────────────────────
# Driven by the 65 mm wide central sections (B, C, D).
# A3 activates at pass 626 (beam_X + P_X = x_sec_start → P_X = +5 mm).
# E3 deactivates after pass 459 (beam_X + P_X = x_sec_end → P_X = −5 mm).
P_X_FIRST = -32_500_000
N_MASTER   = 1_084   # ceil(65_000_000 / 60_000)

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
