"""Generate MB logo pass files for MB300 5-column array.

Logo (mb-logo-w-tag.png, 600×145 px) is scaled to 250 mm × 60.4 mm and
centred on the wafer.  One embedded-v4-header .pass file is written per
active (column, master-pass) combination.

Stage motion
------------
- The stage scans in Y for each pass (stripe length ≈ 60.4 mm).
- Between passes the stage steps 60 µm in X.
- Serpentine: odd-numbered passes scan in the −Y direction (shots written
  Y-descending); even passes scan in +Y (shots Y-ascending).

Columns
-------
MB300: 5 columns at −130, −65, 0, +65, +130 mm wafer-X.
Each column is responsible for the wafer-X section closest to it
(boundaries at midpoints between adjacent columns, clamped to the logo).
"""

import struct
import numpy as np
from PIL import Image
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).parent
_LOGO    = _HERE.parent / "mb-logo-w-tag.png"
OUT_DIR  = _HERE.parent / "logo_passes"

# ── Physical parameters (nm) ──────────────────────────────────────────────────
LOGO_WIDTH_NM  = 250_000_000                           # 250 mm
LOGO_HEIGHT_NM = round(LOGO_WIDTH_NM * 145 / 600)     # ≈ 60_416_667 nm

LOGO_X_MIN = -(LOGO_WIDTH_NM  // 2)   # −125_000_000 nm
LOGO_X_MAX =   LOGO_WIDTH_NM  // 2    # +125_000_000 nm
LOGO_Y_MIN = -(LOGO_HEIGHT_NM // 2)   # ≈ −30_208_333 nm

PITCH_NM      = 1270      # shot pitch → ≈ 20 % overlap at 1587 nm FWHM beam
PASS_WIDTH_NM = 60_000    # 60 µm — stage X step between passes
DWELL_NS      = 16_383    # max 14-bit value

HALF = PITCH_NM // 2      # 325 nm — shots centred in grid cells

# ── MB300 column layout ───────────────────────────────────────────────────────
_COL_OFFSETS_NM = [
    -130_000_000,
     -65_000_000,
           0,
      65_000_000,
     130_000_000,
]

# Section X boundaries: midpoints between adjacent columns, clamped to logo
_BOUNDS = [LOGO_X_MIN]
for _i in range(len(_COL_OFFSETS_NM) - 1):
    _BOUNDS.append((_COL_OFFSETS_NM[_i] + _COL_OFFSETS_NM[_i + 1]) // 2)
_BOUNDS.append(LOGO_X_MAX)
# _BOUNDS: [-125e6, -97.5e6, -32.5e6, +32.5e6, +97.5e6, +125e6]

# (col_id, col_offset_nm, section_x_start_nm, section_x_end_nm)
COLUMNS = [
    (_i + 1, _COL_OFFSETS_NM[_i], _BOUNDS[_i], _BOUNDS[_i + 1])
    for _i in range(len(_COL_OFFSETS_NM))
]

# ── Master pass sweep ─────────────────────────────────────────────────────────
# Stage positions P_X such that C3 (offset 0) sweeps across its section
# [−32.5 mm, +32.5 mm].  At P_X = −32_500_000, C3 starts; C5 also starts.
# At P_X = +5_000_000, C1 starts.  Total 1_084 steps.
P_X_FIRST = -32_500_000   # nm — first master stage position
N_MASTER   = 1_084        # ceil(65_000_000 / 60_000)

# ── Image preprocessing ───────────────────────────────────────────────────────
_img  = Image.open(_LOGO).convert("RGBA")
_arr  = np.array(_img)         # (145, 600, 4) uint8
_lum  = (0.299 * _arr[:, :, 0].astype(np.float32)
       + 0.587 * _arr[:, :, 1].astype(np.float32)
       + 0.114 * _arr[:, :, 2].astype(np.float32)) / 255.0
_alp  = _arr[:, :, 3].astype(np.float32) / 255.0
_dark = (_lum * _alp) < 0.5    # True = fire the beam
DARK_MASK = _dark[::-1, :]      # flip Y: row 0 → bottom of logo on wafer
H_PX, W_PX = DARK_MASK.shape   # 145, 600

# ── Pre-compute Y shot grid (same for every pass) ─────────────────────────────
Y_LOCAL = np.arange(HALF, LOGO_HEIGHT_NM, PITCH_NM, dtype=np.int64)
IY = np.clip(Y_LOCAL * H_PX // LOGO_HEIGHT_NM, 0, H_PX - 1).astype(np.int32)

# ── v4 header packing (78 bytes, little-endian) ───────────────────────────────
_HDR_FMT = "<IHHiiIIdHHdiQQQI??"

def _pack_header(seq_n: int, origin_x: int, stripe_w: int,
                 shot_count: int) -> bytes:
    sort_dir = -1 if seq_n % 2 == 1 else 1
    return struct.pack(
        _HDR_FMT,
        0xB3D11982,          # stripeSymbol (magic)
        4,                   # stripeDataVersion
        seq_n & 0xFFFF,      # stripeNumber (uint16)
        origin_x,            # stripeOriginX  (int32, wafer nm)
        LOGO_Y_MIN,          # stripeOriginY  (int32, wafer nm)
        stripe_w,            # stripeWidth    (uint32, nm)
        LOGO_HEIGHT_NM,      # stripeLength   (uint32, nm)
        1.0,                 # resolution     (double, nm)
        PITCH_NM,            # bss            (uint16, nm)
        0,                   # subFieldHeight (uint16)
        0.0,                 # maxStageSpeed  (double)
        sort_dir,            # sortDirection  (int32): −1=−Y, +1=+Y
        shot_count,          # shotCount      (uint64)
        0,                   # shapeCount     (uint64)
        0,                   # overlap        (uint64)
        DWELL_NS,            # baseDwellTime  (uint32, ns)
        False,               # debug
        False,               # centerShotPresent
    )

# ── Main loop ─────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

_dwell_u64 = np.uint64(DWELL_NS)

total_shots = 0
total_files = 0

print(f"Logo: {W_PX}×{H_PX} px → "
      f"{LOGO_WIDTH_NM/1e6:.0f} mm × {LOGO_HEIGHT_NM/1e6:.2f} mm")
print(f"Pitch: {PITCH_NM} nm  Dwell: {DWELL_NS} ns  "
      f"Passes: {N_MASTER} master × up to 5 columns")
print(f"Output: {OUT_DIR}\n")

for n in range(1, N_MASTER + 1):
    P_X = P_X_FIRST + (n - 1) * PASS_WIDTH_NM

    for col_id, col_offset, sec_x_start, sec_x_end in COLUMNS:
        x_pass_abs = col_offset + P_X
        x_start    = max(x_pass_abs, sec_x_start)
        x_end      = min(x_pass_abs + PASS_WIDTH_NM, sec_x_end)
        if x_end <= x_start:
            continue

        # Local X shot grid (nm, offset from x_start)
        x_local = np.arange(HALF, x_end - x_start, PITCH_NM, dtype=np.int64)
        if len(x_local) == 0:
            continue

        # Map absolute X positions to image pixel columns
        x_abs  = x_start + x_local
        IX = np.clip((x_abs - LOGO_X_MIN) * W_PX // LOGO_WIDTH_NM,
                     0, W_PX - 1).astype(np.int32)

        # Sample the dark mask over the full (Y × X) grid for this pass
        hit = DARK_MASK[IY[:, None], IX[None, :]]   # (N_Y, N_X) bool

        sy, sx = np.nonzero(hit)
        if len(sy) == 0:
            continue                # blank stripe — skip

        # Serpentine sort: odd passes scan −Y (desc), even passes +Y (asc)
        if n % 2 == 1:
            order = np.argsort(-Y_LOCAL[sy], kind='stable')
        else:
            order = np.argsort( Y_LOCAL[sy], kind='stable')

        xl = x_local[sx[order]].astype(np.uint64)
        yl = Y_LOCAL[sy[order]].astype(np.uint64)

        records = (_dwell_u64 << np.uint64(2)) \
                | (xl << np.uint64(16)) \
                | (yl << np.uint64(32))

        fname = OUT_DIR / f"C{col_id}_{n:04d}.pass"
        with open(fname, "wb") as f:
            f.write(_pack_header(n, x_start, x_end - x_start, len(sy)))
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
