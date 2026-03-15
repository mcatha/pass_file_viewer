"""
Binary parser for .pass files (v3 and v4, auto-detected).

Pass file format:
  - Header: 64 bytes (v3) or 78 bytes (v4), little-endian, packed
  - Stream of 8-byte records (Shot or Shape union)
    - Shot (mID == 0): 2-bit ID, 14-bit dwell, 16-bit X, 32-bit Y
    - Shape (mID != 0): 2-bit ID, 30-bit spare, 32-bit parameter (signed)

Uses numpy for vectorised bitfield extraction — handles millions of records
in seconds.
"""

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── header layouts ──────────────────────────────────────────────────
# v3: 64 bytes   v4: 78 bytes (adds overlap, baseDwellTime, debug, centerShotPresent)
_V3_FMT = "<IHHiiIIdHHdiQQ"
_V4_FMT = "<IHHiiIIdHHdiQQQI??"
_V3_SIZE = struct.calcsize(_V3_FMT)   # 64
_V4_SIZE = struct.calcsize(_V4_FMT)   # 78

_V3_FIELDS = (
    "stripeSymbol",
    "stripeDataVersion",
    "stripeNumber",
    "stripeOriginX",
    "stripeOriginY",
    "stripeWidth",
    "stripeLength",
    "resolution",
    "bss",
    "subFieldHeight",
    "maxStageSpeed",
    "sortDirection",
    "shotCount",
    "shapeCount",
)

_V4_FIELDS = _V3_FIELDS + (
    "overlap",
    "baseDwellTime",
    "debug",
    "centerShotPresent",
)


@dataclass
class PassHeader:
    stripeSymbol: int = 0
    stripeDataVersion: int = 0
    stripeNumber: int = 0
    stripeOriginX: int = 0
    stripeOriginY: int = 0
    stripeWidth: int = 0
    stripeLength: int = 0
    resolution: float = 0.0
    bss: int = 0
    subFieldHeight: int = 0
    maxStageSpeed: float = 0.0
    sortDirection: int = 0
    shotCount: int = 0
    shapeCount: int = 0
    overlap: int = 0
    baseDwellTime: int = 0
    debug: bool = False
    centerShotPresent: bool = False
    version: int = 3  # detected format version


@dataclass
class PassData:
    """Parsed contents of a .pass file."""
    header: PassHeader
    x: np.ndarray       # float32 — shot X coordinates
    y: np.ndarray       # float32 — shot Y coordinates
    dwell: np.ndarray   # float32 — dwell values
    count: int          # number of shots


def _detect_version(raw: bytes) -> int:
    """
    Auto-detect v3 vs v4 by checking which header size yields a record
    count matching the shotCount + shapeCount stored in the header.
    """
    if len(raw) < _V3_SIZE:
        raise ValueError(
            f"File too small ({len(raw)} bytes) for a v3 header "
            f"({_V3_SIZE} bytes)."
        )

    # Parse the v3 portion to read shotCount + shapeCount
    v3_vals = struct.unpack_from(_V3_FMT, raw, 0)
    expected_records = v3_vals[12] + v3_vals[13]  # shotCount + shapeCount

    remaining_v3 = len(raw) - _V3_SIZE
    remaining_v4 = len(raw) - _V4_SIZE

    # Check v3 first: remaining / 8 == expected records
    if remaining_v3 >= 0 and remaining_v3 % 8 == 0:
        if remaining_v3 // 8 == expected_records:
            return 3

    # Check v4
    if remaining_v4 >= 0 and remaining_v4 % 8 == 0:
        v4_vals = struct.unpack_from(_V4_FMT, raw, 0)
        if remaining_v4 // 8 == v4_vals[12] + v4_vals[13]:
            return 4

    # Fallback: whichever gives clean 8-byte alignment
    if remaining_v4 >= 0 and remaining_v4 % 8 == 0:
        return 4
    return 3


def parse_pass_file(path: str | Path) -> PassData:
    """
    Parse a .pass binary file (auto-detects v3 vs v4).

    Parameters
    ----------
    path : str or Path
        Path to the .pass file.

    Returns
    -------
    PassData
        Parsed header and shot arrays.
    """
    path = Path(path)
    file_size = path.stat().st_size

    # Use memory-mapped I/O for large files (>10 MB) to avoid copying
    # the entire file into Python memory — lets the OS page it in on demand.
    if file_size > 10 * 1024 * 1024:
        f = open(path, 'rb')
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        raw = mm
    else:
        raw = path.read_bytes()
        f = None
        mm = None

    try:
        version = _detect_version(raw)

        # ── parse header ────────────────────────────────────────────────
        if version == 4:
            fmt, size, fields = _V4_FMT, _V4_SIZE, _V4_FIELDS
        else:
            fmt, size, fields = _V3_FMT, _V3_SIZE, _V3_FIELDS

        values = struct.unpack_from(fmt, raw, 0)
        field_dict = dict(zip(fields, values))
        field_dict["version"] = version
        header = PassHeader(**field_dict)

        # ── parse records ───────────────────────────────────────────────
        n_records = (len(raw) - size) // 8
        if n_records == 0:
            return PassData(
                header=header,
                x=np.empty(0, dtype=np.float32),
                y=np.empty(0, dtype=np.float32),
                dwell=np.empty(0, dtype=np.float32),
                count=0,
            )

        # Read as uint64 → single contiguous view, no reshape needed.
        # .copy() detaches from the mmap so it can be safely closed afterwards.
        raw64 = np.frombuffer(raw, dtype=np.dtype("<u8"), count=n_records, offset=size).copy()

        # Bitfield extraction via 64-bit arithmetic — avoids creating
        # the intermediate 2-column uint32 array (halves temp memory).
        #   bits  0- 1: mID      (2 bits)
        #   bits  2-15: mDwell  (14 bits)
        #   bits 16-31: mX      (16 bits)
        #   bits 32-63: mY      (32 bits)
        m_id = (raw64 & np.uint64(0x3)).astype(np.uint8)
        m_dwell = ((raw64 >> np.uint64(2)) & np.uint64(0x3FFF)).astype(np.uint16)
        m_x = ((raw64 >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint16)
        m_y = (raw64 >> np.uint64(32)).astype(np.uint32)

        # Keep only shots (mID == 0) — free intermediate arrays early
        shot_mask = m_id == 0
        del m_id, raw64

        x = m_x[shot_mask].astype(np.float32)
        y = m_y[shot_mask].astype(np.float32)
        dwell = m_dwell[shot_mask].astype(np.float32)
        n_shots = int(np.sum(shot_mask))
        del m_x, m_y, m_dwell, shot_mask

        return PassData(
            header=header,
            x=x,
            y=y,
            dwell=dwell,
            count=n_shots,
        )
    finally:
        if mm is not None:
            mm.close()
        if f is not None:
            f.close()
