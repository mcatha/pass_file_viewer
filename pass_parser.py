"""
Binary parser for .pass files and their companion .pass.meta metadata files.

Pass file format:
  - .pass files contain ONLY 8-byte shot/shape records (no header).
  - Metadata lives in a companion .pass.meta file (same name + .meta).
  - .pass.meta follows the MEBL2 packed struct format (little-endian).
  - Legacy synthetic files may embed a header directly in the .pass file;
    these are detected by the magic number 0xb3d11982 at byte 0.

Record format (8 bytes each, Shot or Shape union):
    Shot  (mID == 0): 2-bit ID, 14-bit dwell, 16-bit X, 32-bit Y
    Shape (mID != 0): 2-bit ID, 30-bit spare, 32-bit parameter (signed)

Uses numpy for vectorised bitfield extraction — handles millions of records
in seconds.
"""

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── magic number ────────────────────────────────────────────────────
_STRIPE_SYMBOL = 0xB3D11982

# ── meta / embedded header layouts (little-endian, packed) ──────────
# v3: 64 bytes  v4: 78 bytes  v2.1.1: 88 bytes (appends compression fields)
_V3_FMT = "<IHHiiIIdHHdiQQ"
_V4_FMT = "<IHHiiIIdHHdiQQQI??"
_V211_FMT = "<IHHiiIIdHHdiQQQI????ii"
_V3_SIZE = struct.calcsize(_V3_FMT)     # 64
_V4_SIZE = struct.calcsize(_V4_FMT)     # 78
_V211_SIZE = struct.calcsize(_V211_FMT)  # 88

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

_V211_FIELDS = _V4_FIELDS + (
    "compression",
    "compression2Order",
    "shotsPerBlock",
    "blocksPer2Order",
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
    compression: bool = False
    compression2Order: bool = False
    shotsPerBlock: int = 0
    blocksPer2Order: int = 0
    version: int = 0  # detected meta format version (3, 4, or 211)


@dataclass
class PassData:
    """Parsed contents of a .pass file."""
    header: PassHeader
    x: np.ndarray       # float32 — shot X coordinates
    y: np.ndarray       # float32 — shot Y coordinates
    dwell: np.ndarray   # float32 — dwell values
    count: int          # number of shots


def _parse_header_bytes(raw: bytes) -> PassHeader:
    """Parse a header from raw bytes (meta file or embedded), auto-detecting
    the version by size and validating the magic number."""
    size = len(raw)

    if size >= _V211_SIZE:
        fmt, fields, ver = _V211_FMT, _V211_FIELDS, 211
    elif size >= _V4_SIZE:
        fmt, fields, ver = _V4_FMT, _V4_FIELDS, 4
    elif size >= _V3_SIZE:
        fmt, fields, ver = _V3_FMT, _V3_FIELDS, 3
    else:
        raise ValueError(
            f"Header too small ({size} bytes); minimum is {_V3_SIZE}."
        )

    values = struct.unpack_from(fmt, raw, 0)
    if values[0] != _STRIPE_SYMBOL:
        raise ValueError(
            f"Invalid header magic: 0x{values[0]:08x} "
            f"(expected 0x{_STRIPE_SYMBOL:08x})."
        )

    field_dict = dict(zip(fields, values))
    field_dict["version"] = ver
    return PassHeader(**field_dict)


def parse_meta_file(path: str | Path) -> PassHeader:
    """Parse a .pass.meta file and return the header.

    Parameters
    ----------
    path : str or Path
        Path to the .pass.meta file.

    Returns
    -------
    PassHeader
        Parsed metadata header.
    """
    raw = Path(path).read_bytes()
    return _parse_header_bytes(raw)


def parse_pass_header_only(path: str | Path) -> PassHeader:
    """Read header metadata without decoding shot records.

    Checks for a companion .pass.meta file first (tiny, fast).  If absent,
    reads just the first 78–88 bytes of the .pass file looking for an embedded
    header.  Returns a default PassHeader() if no valid header is found.

    Use this when you need spatial/count metadata for many files without the
    cost of reading gigabytes of shot data.
    """
    path = Path(path)
    meta_path = path.parent / (path.name + ".meta")
    if meta_path.is_file():
        return parse_meta_file(meta_path)
    try:
        with open(path, 'rb') as f:
            magic_bytes = f.read(6)
        if (len(magic_bytes) >= 6
                and struct.unpack_from("<I", magic_bytes, 0)[0] == _STRIPE_SYMBOL):
            version = struct.unpack_from("<H", magic_bytes, 4)[0]
            size = _V211_SIZE if version == 2 else _V4_SIZE
            with open(path, 'rb') as f:
                return _parse_header_bytes(f.read(size))
    except OSError:
        pass
    return PassHeader()


def _read_records(raw, offset: int = 0, shot_stride: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract shot arrays from raw 8-byte records starting at *offset*.

    Returns (x, y, dwell, n_shots) arrays.
    """
    n_records = (len(raw) - offset) // 8
    if n_records == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty.copy(), 0

    # Read as uint64 — .copy() detaches from any mmap
    raw64 = np.frombuffer(
        raw, dtype=np.dtype("<u8"), count=n_records, offset=offset,
    )
    if shot_stride > 1:
        raw64 = raw64[::shot_stride]

    # Bitfield extraction via 64-bit arithmetic:
    #   bits  0- 1: mID      (2 bits)
    #   bits  2-15: mDwell  (14 bits)
    #   bits 16-31: mX      (16 bits)
    #   bits 32-63: mY      (32 bits)
    m_id = (raw64 & np.uint64(0x3)).astype(np.uint8)
    m_dwell = ((raw64 >> np.uint64(2)) & np.uint64(0x3FFF)).astype(np.uint16)
    m_x = ((raw64 >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint16)
    m_y = (raw64 >> np.uint64(32)).astype(np.uint32)

    # Keep only shots (mID == 0)
    shot_mask = m_id == 0
    del m_id, raw64

    x = m_x[shot_mask].astype(np.float32)
    y = m_y[shot_mask].astype(np.float32)
    dwell = m_dwell[shot_mask].astype(np.float32)
    n_shots = int(np.sum(shot_mask))
    del m_x, m_y, m_dwell, shot_mask

    return x, y, dwell, n_shots


def parse_pass_file(path: str | Path, shot_stride: int = 1) -> PassData:
    """Parse a .pass binary file.

    Metadata is read from a companion .pass.meta file if present.
    Otherwise, the .pass file is checked for an embedded header
    (magic number 0xb3d11982 at byte 0). Version 1 headers are
    78 bytes; version 2 headers are 88 bytes. If neither a .meta
    file nor an embedded header is found, a default empty header
    is used and the entire file is treated as shot records.

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

    # ── look for companion .pass.meta ───────────────────────────────
    meta_path = path.parent / (path.name + ".meta")
    if meta_path.is_file():
        header = parse_meta_file(meta_path)
        record_offset = 0
    else:
        # Check for embedded header (magic number at byte 0)
        with open(path, 'rb') as probe:
            magic_bytes = probe.read(6)  # 4 bytes magic + 2 bytes version
        if (len(magic_bytes) >= 6
                and struct.unpack_from("<I", magic_bytes, 0)[0] == _STRIPE_SYMBOL):
            version = struct.unpack_from("<H", magic_bytes, 4)[0]
            if version == 2:
                record_offset = _V211_SIZE
            else:
                record_offset = _V4_SIZE
            with open(path, 'rb') as hf:
                header = _parse_header_bytes(hf.read(record_offset))
        else:
            header = PassHeader()
            record_offset = 0

    # ── memory-map large files ──────────────────────────────────────
    if file_size > 10 * 1024 * 1024:
        f = open(path, 'rb')
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        raw = mm
    else:
        raw = path.read_bytes()
        f = None
        mm = None

    try:
        x, y, dwell, n_shots = _read_records(raw, record_offset, shot_stride)
        return PassData(header=header, x=x, y=y, dwell=dwell, count=n_shots)
    finally:
        if mm is not None:
            mm.close()
        if f is not None:
            f.close()
