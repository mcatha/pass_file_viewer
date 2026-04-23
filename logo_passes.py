"""Generate MB logo pass files for MB300.

Logo (mb-logo-w-tag.png, 600×145 px) is scaled to 250 mm × 60.4 mm and
centred on the wafer.  One embedded-v4-header .pass file is written per
active (beam, master-pass) combination.

Each MB300 beam has a rectangular area of responsibility defined by:
  - X section: midpoints between adjacent beam X positions
  - Y section: midpoints between adjacent beam Y positions (within its column)

Of the 18 MB300 beams, 8 have Y sections that overlap the 60.4 mm logo:
  A3, B2, B3, C2, C3, D2, D3, E3

Stage motion
------------
- Stage steps 60 µm in X between passes.
- For each X position the stage sweeps in Y over the beam's Y section.
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

PITCH_NM      = 1270     # shot pitch
PASS_WIDTH_NM = 60_000   # 60 µm stage X step
DWELL_NS      = 16_383   # 14-bit maximum
HALF          = PITCH_NM // 2

# ── MB300 beam columns (from viewer_widget._MB300_FIDUCIALS) ──────────────────
# Each entry: (name, beam_X_nm, x_sec_start, x_sec_end, y_sec_start, y_sec_end)
#
# X section boundaries = midpoints between adjacent column X positions:
#   -125, -97.5, -32.5, +32.5, +97.5, +125 mm
#
# Y section boundaries = midpoints between adjacent beam Y positions per column,
# clamped to logo.  Beams B1/B4/C1/C4/D1/D4 (Y=±112.5 mm) and
# A2/A4/E2/E4 (Y=±75 mm) have sections that don't reach the logo and are omitted.
#
#   Inner columns B/C/D: rows 2 (+37.5 mm) and 3 (−37.5 mm)
#     boundary at Y = 0 (midpoint between +37.5 and −37.5)
#   Outer columns A/E: row 3 (Y = 0)
#     section [−37.5, +37.5] mm covers the full logo height
BEAM_COLUMNS = [
    # name  beam_X          x_sec_start     x_sec_end      y_sec_start  y_sec_end
    ('A3', -130_000_000,  LOGO_X_MIN,     -97_500_000,   LOGO_Y_MIN,  LOGO_Y_MAX),
    ('B2',  -65_000_000,  -97_500_000,    -32_500_000,            0,  LOGO_Y_MAX),
    ('B3',  -65_000_000,  -97_500_000,    -32_500_000,   LOGO_Y_MIN,           0),
    ('C2',            0,  -32_500_000,     32_500_000,            0,  LOGO_Y_MAX),
    ('C3',            0,  -32_500_000,     32_500_000,   LOGO_Y_MIN,           0),
    ('D2',   65_000_000,   32_500_000,     97_500_000,            0,  LOGO_Y_MAX),
    ('D3',   65_000_000,   32_500_000,     97_500_000,   LOGO_Y_MIN,           0),
    ('E3',  130_000_000,   97_500_000,    LOGO_X_MAX,    LOGO_Y_MIN,  LOGO_Y_MAX),
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
_y_grids: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
for _, _, _, _, ys, ye in BEAM_COLUMNS:
    key = (ys, ye)
    if key in _y_grids:
        continue
    sec_len = ye - ys
    Y_LOC = np.arange(HALF, sec_len, PITCH_NM, dtype=np.int64)
    # wafer_Y = y_local + ys  →  image row = (wafer_Y − LOGO_Y_MIN) × H_PX / LOGO_HEIGHT_NM
    IY_ = np.clip(
        (Y_LOC + ys - LOGO_Y_MIN) * H_PX // LOGO_HEIGHT_NM,
        0, H_PX - 1,
    ).astype(np.int32)
    _y_grids[key] = (Y_LOC, IY_)

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

        Y_LOCAL, IY = _y_grids[(ys_start, ys_end)]
        hit = DARK_MASK[IY[:, None], IX[None, :]]

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
