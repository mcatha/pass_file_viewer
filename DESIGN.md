# Pass File Viewer — Design Document

**Version 1.4 — April 2026**

## 1. Overview

The **Pass File Viewer** is a GPU-accelerated desktop application for visualising binary `.pass` shot files produced by Multibeam's e-beam lithography column control system. It renders millions of shots as size-scaled markers on a 2D Cartesian plane, with real-time pan, zoom, rotate, interactive selection, and a measurement ruler.

### The central rendering challenge

The viewer must look correct across ~5 orders of magnitude of zoom. At close zoom each shot is many pixels wide and they overlap heavily — additive blending blows out to solid white. At far zoom thousands of shots collapse into a single pixel — if alpha is too low they disappear, too high and sparse regions look the same as dense ones. There is no single blend mode, alpha value, or marker size that works across this range. Every fix at one end creates a problem at the other.

The piecewise alpha curves, priority-based decimation, minimum-size inflation, and mode-specific sizing are all compromises to keep the image reasonable across the full zoom range. None are perfect — they are the least-wrong tradeoffs found so far.

### Two rendering modes

- **Disc mode** (default): two composited layers — an alpha-blend base for stable appearance plus an additive white overlay for overlap brightening. Hard edges let the user distinguish individual shots, identify specific placements, and verify spacing. Per-shot alpha follows a piecewise log-linear curve with user-adjustable breakpoints.
- **Gaussian mode**: each shot is a soft Gaussian point-spread function with FWHM-based sizing. A single additive-blend layer accumulates colour; per-shot alpha follows a sigmoid curve so overlap brightens at close zoom without blowing out when zoomed far. This shows the accumulated dose picture but individual shots blur into each other.

### Key capabilities

| Feature | Description |
|---------|-------------|
| **High-performance rendering** | Instanced OpenGL markers via vispy; loads 10 M+ shot files; renders up to 2 M shots per frame via priority-based decimation |
| **Disc markers (default)** | Hard-edged discs with additive white overlap layer for brightening toward white |
| **Gaussian PSF markers** | Custom fragment shader renders each shot as a smooth Gaussian with FWHM = shot size |
| **Zoom-adaptive alpha** | Mode-specific alpha curves prevent blowout at wide zoom while keeping close-zoom overlap visible |
| **Adjustable alpha curves** | Disc and Gaussian alpha curves exposed via per-mode slider submenus |
| **FWHM slider** | Logarithmic slider (0.1–1000 nm/µs) controls shot size scaling in real time |
| **Priority-based decimation** | Fixed per-shot random priority; mode-specific density budgets |
| **Interactive selection** | Single-click and rubber-band box selection with side-pane data table |
| **Measurement ruler** | Shift+click ruler with perpendicular tick marks at 1-2-5 intervals |
| **Shot connections** | Toggle-able lines between consecutive shots |
| **Colour customisation** | 5-category colour menu with presets and custom colour picker |
| **Hover tooltips** | KD-tree spatial lookup shows shot info on mouse hover |
| **2D rotation** | Shift+left-drag rotates the view; axis arrowheads track orientation |
| **Background music** | Looping ambient audio with volume slider |
| **Software OpenGL fallback** | Bundled Mesa llvmpipe renderer for machines with no GPU |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      main.py                            │
│  Entry point: _check_opengl() → QApplication → MainWindow│
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  main_window.py                         │
│  MainWindow (QMainWindow)                               │
│  ├── Menus: File, View (FWHM slider, Marker Mode,      │
│  │     Disc/Gaussian Alpha Controls, Colors), Volume,   │
│  │     Help                                             │
│  ├── Audio: QMediaPlayer + QAudioOutput (looping mp3)   │
│  ├── Status bar: file info, shot count, spatial index   │
│  ├── Central: ShotViewerWidget                          │
│  └── Right dock: SelectionPane                          │
└──┬──────────────┬──────────────┬────────────────────────┘
   │              │              │
   ▼              ▼              ▼
pass_parser.py  viewer_widget.py  selection_pane.py
                  │
                  ▼
            gaussian_markers.py
```

### Module responsibilities

| Module | Role |
|--------|------|
| `main.py` | Entry point; OpenGL driver check and software fallback; command-line `.pass` file argument |
| `main_window.py` | Qt window, menus (FWHM slider, marker mode, alpha controls, colours, volume), background file parsing, status bar, audio player |
| `pass_parser.py` | Binary `.pass` / `.pass.meta` parser; numpy vectorised bitfield extraction |
| `viewer_widget.py` | GPU rendering, camera, decimation, selection, tooltips, rotation, axis lines, ruler |
| `gaussian_markers.py` | `GaussianMarkers` visual — custom Gaussian PSF fragment + vertex shaders |
| `selection_pane.py` | Virtual table model for box-selected shots; clipboard copy |
| `img_to_pass.py` | Utility: converts PNG images to synthetic `.pass` files using brightness-mapped disc overlap |

---

## 3. File Format — `.pass` + `.pass.meta`

A pass file consists of two companion files:

| File | Contents |
|------|----------|
| `*.pass` | 8-byte shot/shape records, optionally preceded by an embedded header |
| `*.pass.meta` | Metadata header in MEBL2 packed struct format (optional if header is embedded) |

Pass files come in two variants:
1. **Headerless + companion meta**: the `.pass` file is a flat stream of 8-byte records from byte 0; all metadata lives in a companion `.pass.meta` file with the same base name.
2. **Embedded header**: the `.pass` file begins with a header (magic number `0xB3D11982` at byte 0), followed by shot/shape records. Version 1 headers are 78 bytes; version 2 headers are 88 bytes. The `stripeDataVersion` field (offset 4, `uint16`) distinguishes them.

### `.pass.meta` — MEBL2 header format

The meta file is a single packed little-endian struct (`__attribute__((__packed__))`) beginning with magic number `0xB3D11982`. The current format is **v2.1.1** (88 bytes). The parser also accepts older/shorter meta files (64 or 78 bytes) by detecting size, but production files use the full 88-byte layout.

#### Field layout

| Offset | Field | C type | Pack fmt | Description |
|--------|-------|--------|----------|-------------|
| 0 | `stripeSymbol` | `uint32` | `I` | Magic number — must be `0xB3D11982` |
| 4 | `stripeDataVersion` | `uint16` | `H` | Data format version |
| 6 | `stripeNumber` | `uint16` | `H` | Stripe index |
| 8 | `stripeOriginX` | `int32` | `i` | X origin (nm), signed, relative to wafer centre |
| 12 | `stripeOriginY` | `int32` | `i` | Y origin (nm), signed, relative to wafer centre |
| 16 | `stripeWidth` | `uint32` | `I` | Stripe width (nm) |
| 20 | `stripeLength` | `uint32` | `I` | Stripe length (nm) |
| 24 | `resolution` | `double` | `d` | Resolution (nm), IEEE 754 double-precision |
| 32 | `bss` | `uint16` | `H` | Beam step size |
| 34 | `subFieldHeight` | `uint16` | `H` | Sub-field height |
| 36 | `maxStageSpeed` | `double` | `d` | Maximum stage speed |
| 44 | `sortDirection` | `int32` | `i` | `SortDirection` enum (4 bytes) |
| 48 | `shotCount` | `uint64` | `Q` | Number of shot records (`unsigned long`, 8 bytes on Linux/GCC x86_64) |
| 56 | `shapeCount` | `uint64` | `Q` | Number of shape records |
| **v2.1.1 additional fields** | | | | |
| 64 | `overlap` | `uint64` | `Q` | Overlap value |
| 72 | `baseDwellTime` | `uint32` | `I` | Base dwell time (ns) |
| 76 | `debug` | `bool` | `?` | Debug flag |
| 77 | `centerShotPresent` | `bool` | `?` | Centre shot present flag |
| | | | | |
| 78 | `compression` | `bool` | `?` | Compression enabled |
| 79 | `compression2Order` | `bool` | `?` | Second-order compression |
| 80 | `shotsPerBlock` | `int32` | `i` | Shots per compressed block |
| 84 | `blocksPer2Order` | `int32` | `i` | Blocks per second-order group |

Notes:
- Origins (`stripeOriginX`, `stripeOriginY`) are signed — negative values are normal (coordinates relative to wafer centre).
- `unsigned long` fields (`shotCount`, `shapeCount`, `overlap`) are 8 bytes. The format targets Linux/GCC on x86_64 where `unsigned long` = 8 bytes.
- The struct is packed with no padding (`__attribute__((__packed__))`).

### `.pass` — shot/shape records (8 bytes each, little-endian `uint64`)

| Bits | Field | Width | Description |
|------|-------|-------|-------------|
| 0–1 | `mID` | 2 | 0 = Shot, non-zero = Shape |
| 2–15 | `mDwell` | 14 | Dwell time in ns (max 16383) |
| 16–31 | `mX` | 16 | X coordinate |
| 32–63 | `mY` | 32 | Y coordinate |

Only shot records (`mID == 0`) are rendered. Shape records are skipped.

### Parser behaviour

- Metadata source priority: (1) companion `.pass.meta` file, (2) embedded header in the `.pass` file, (3) empty default header.
- When a `.pass.meta` file exists, shot records are read from byte 0 of the `.pass` file.
- When no `.pass.meta` exists but the first 4 bytes match the magic number `0xB3D11982`, the embedded header is parsed and shot records begin after the header (78 or 88 bytes depending on version).
- When neither is available, the entire `.pass` file is treated as shot records with a default empty header.
- Files >10 MB are memory-mapped (`mmap`) to avoid copying into Python memory.
- Bitfield extraction uses 64-bit numpy arithmetic on a contiguous `uint64` view — no intermediate reshape.
- Shot mask applied once; intermediate arrays freed immediately.

---

## 4. Rendering Architecture

### 4.1 OpenGL backend

The viewer prefers the **GL3+ backend** (`gl+`) for instanced rendering. On machines without GL3+ support, it falls back to GL2:

```python
try:
    vispy.use(gl='gl+')      # Prefer GL3+ (instanced rendering)
except Exception:
    vispy.use(gl='gl2')      # Fallback for machines without GL3+
```

On machines with no GPU at all (e.g. Microsoft Basic Display Adapter with only OpenGL 1.1), the entry point (`main.py`) detects the broken driver and sets `QT_OPENGL=software`, which triggers Qt's bundled Mesa llvmpipe software renderer (`opengl32sw.dll`).

### 4.2 Two rendering modes

The user can switch between Disc and Gaussian modes via **View → Marker Mode**. Both modes share the same decimated point set and selection system; only the GPU visuals differ.

#### Disc mode (default)

Two `Markers` visuals composited on the same scene node:

| Property | Base layer | Overlap layer |
|----------|-----------|---------------|
| Blend mode | `(src_alpha, one_minus_src_alpha)` | `(src_alpha, one)` |
| Symbol | Stock vispy disc | Stock vispy disc |
| Face colour | User-chosen shot colour (α scaled by zoom) | White `(1, 1, 1, α)` — zoom-dependent |
| Size | `max(true_size, dpp) × 0.667` — inflated to ≥ 1 px, scaled by `_DISC_SIZE_SCALE` | `true_size × 0.667` — no inflation |
| Purpose | Stable visual at all zooms | Additive brightening only where shots genuinely overlap in data space |

The disc alpha uses a **piecewise log-linear curve** with four regimes:

| Region | Condition | Alpha ($f$) |
|--------|-----------|-------------|
| Opaque | $\text{dpp} \le \text{dpp\_lo}$ | $f = 1.0$ |
| Upper ramp | $\text{dpp\_lo} < \text{dpp} \le \text{dpp\_mid}$ | $f = \text{lerp}(1.0,\ f_{\text{mid}},\ t)$ where $t = \frac{\ln(\text{dpp}) - \ln(\text{dpp\_lo})}{\ln(\text{dpp\_mid}) - \ln(\text{dpp\_lo})}$ |
| Lower ramp | $\text{dpp\_mid} < \text{dpp} < \text{dpp\_hi}$ | $f = \text{lerp}(f_{\text{mid}},\ 0,\ t)$ where $t = \frac{\ln(\text{dpp}) - \ln(\text{dpp\_mid})}{\ln(\text{dpp\_hi}) - \ln(\text{dpp\_mid})}$ |
| Transparent | $\text{dpp} \ge \text{dpp\_hi}$ | $f = 0.0$ |

Base layer RGBA: `(R, G, B, f)`. Overlap layer RGBA: `(1, 1, 1, ow × f)` where `ow` is `_DISC_OVERLAP_WHITE`.

Default breakpoints (all user-adjustable via View → Disc Alpha Controls):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dpp_lo` | 0.01 | DPP below which disc is fully opaque |
| `dpp_mid` | 5000.0 | Intermediate breakpoint |
| `dpp_hi` | 1e10 | DPP above which disc is fully transparent |
| `f_mid` | 0.20 | Alpha value at `dpp_mid` |
| `ow` | 0.05 | Additive white alpha per overlap |

#### Gaussian mode

A single `GaussianMarkers` visual with **additive blending** (`src_alpha, one`).

Each shot is rendered as a smooth Gaussian point-spread function. The `size` parameter is the **FWHM** (Full Width at Half Maximum) — the Gaussian crosses 50 % intensity at exactly `size / 2` from centre.

```
alpha(r) = exp(-4·ln2·r²)     where r = 2·dist / FWHM
```

Per-shot face alpha follows a **sigmoid curve**:

$$\alpha = \alpha_{\text{far}} + \frac{\alpha_{\text{max}} - \alpha_{\text{far}}}{1 + \left(\frac{\text{dpp}_{\text{eff}}}{d_{\text{ref}}}\right)^p}$$

where $\text{dpp}_{\text{eff}} = \max(\text{dpp},\ \text{natural\_size} / 4)$.

| Constant | Default | Adjustable | Description |
|----------|---------|------------|-------------|
| `_ALPHA_FAR` | 0.005 | No | Alpha floor at maximum zoom-out |
| `_ALPHA_MAX` | 0.330 | Yes | Alpha cap at close zoom |
| `_ALPHA_DREF` | 16.5 | Yes | Sigmoid midpoint DPP |
| `_ALPHA_P` | 1.5 | No | Sigmoid steepness exponent |

**Stride compensation**: alpha is divided by a stride-dependent scale factor:

$$\text{stride\_scale} = 1 + \text{amp} \times \frac{\log_{10}(\text{stride})}{\log_{10}(\text{stride}) + 0.2}$$

with `_STRIDE_INFLATE_AMP = 0.50`. This keeps per-shot alpha higher when fewer shots are drawn, partially compensating for fewer overlapping additive contributions.

### 4.3 Gaussian PSF shader (`gaussian_markers.py`)

`GaussianMarkers` subclasses vispy's `MarkersVisual` with two shader modifications:

**Fragment shader**: replaces the stock disc SDF with a Gaussian profile. The SDF call is kept (for vispy template wiring) but its result is unused — alpha comes purely from `exp(-2.7726 · nr²)` where `nr = dist / half_size`. Fragments with `gauss < 0.003` are discarded to save fill rate.

**Vertex shader**: the `total_size` formula is patched from `$v_size + 4*(edgewidth + 1.5*antialias)` to `$v_size * 3.0 + 4.0`, giving the bounding quad enough room for Gaussian tails (the Gaussian at `r = 1.43` hits the 0.003 discard threshold).

### 4.4 Shot sizing

Shot size in data units (nm) is proportional to dwell time, scaled by a user-adjustable FWHM factor:

$$d = \max(\text{dwell\_ns} \times \texttt{\_NM\_PER\_NS\_DWELL} \times \texttt{fwhm\_scale},\ 1.0)$$

| Constant | Default | Description |
|----------|---------|-------------|
| `_NM_PER_NS_DWELL` | 0.01 | Base conversion: 10 nm per µs |
| `fwhm_scale` | 10.0 | Multiplier — default gives 100 nm/µs FWHM |
| `_DISC_SIZE_SCALE` | 0.667 | Additional scale factor applied in disc mode only |

The FWHM slider in the View menu adjusts `fwhm_scale` logarithmically from 0.01× to 100× (0.1–1000 nm/µs effective).

When all dwells are identical (common case), a single scalar size is used instead of a per-point array, saving `N × 4` bytes of GPU upload.

### 4.5 Minimum marker size

Each marker is inflated to a minimum size so sub-pixel shots remain visible:

- **Gaussian mode**: minimum size = `dpp × 4` (Gaussians need ~4 px to show their smooth falloff)
- **Disc mode base layer**: minimum size = `dpp` (1 screen pixel)
- **Disc mode overlap layer**: no inflation — uses true data size only

### 4.6 Centroid shifting

Raw shot coordinates (X up to 65535, Y up to ~60 million nm) would lose precision in float32. All positions are centroid-shifted before upload:

```python
origin = raw_pos.mean(axis=0)  # float64
positions = (raw_pos - origin).astype(float32)  # sub-nm precision
```

The origin is stored separately for tooltip coordinate reconstruction.

### 4.7 Axis lines

Origin crosshair lines (X and Y axes through the file origin) use a **grow-only dynamic length**:

1. At load time, half-length is set to `max(10 × data_diagonal, 1 mm)`.
2. On each camera change, if the viewport diagonal × 2 exceeds the current half-length, the lines are re-uploaded with the larger value.
3. Lines never shrink — this avoids set_data thrash on repeated zoom in/out cycles.

Large fixed lengths (e.g. 1e15) cause GPU float32 clip-pipeline precision loss (axis jitter). The grow-only approach keeps values within safe float32 range while guaranteeing axes extend past screen edges at any zoom level.

---

## 5. Decimation

Decimation runs on every camera change (100 ms debounce timer). The same code path runs at every zoom level. When the visible shot count is small enough, stride = 1 and every visible shot is drawn.

### Mode-specific tuning parameters

| Parameter | Gaussian | Disc |
|-----------|----------|------|
| Shots per pixel | 20.0 | 3.0 |
| Minimum budget | 0 | 0 |
| Hard cap | 2,097,152 (2²¹) | 2,097,152 (2²¹) |

### Algorithm

1. **Viewport cull**: AABB mask keeps only shots inside the visible area + 5 % margin + half the largest disc diameter. Viewport bounds are rotation-aware (4-corner screen → data mapping).
2. **Density-based budget**: the occupied screen area is estimated from `min(data_extent, viewport_extent)` per axis, converted to pixels via `dpp`. Budget = `screen_px × shots_per_px`, capped at the mode's hard cap. No floor — when the data covers only a few pixels, very few shots are rendered.
3. **Stride**: `stride = max(1, n_visible / budget)`. When stride ≤ 1, all visible shots are drawn.
4. **Priority-based selection**: each shot is assigned a fixed random priority at load time. The top `budget` shots by priority are selected via `np.argpartition` (O(n)). A full `np.argsort` of priorities is computed on a background `QThread` after load; once ready, the all-visible fast path slices it in O(1) instead of calling `argpartition`.
5. **Cache gating**: a composite key of quantised viewport bounds + stride + `round(log₂(dpp) × 20)` prevents redundant GPU uploads when the view hasn't meaningfully changed.
6. **All visuals synced**: the same decimated index set is uploaded to all active marker layers.

### Design properties

- **Stable across pan/zoom**: fixed per-shot priorities mean the same shots survive decimation regardless of small viewport shifts — no visible flicker.
- **No rotation artefacts**: viewport bounds use 4-corner mapping; `dpp` is computed from the full transform chain via Euclidean distance, not just the X component.
- **No blowout**: density-based budget limits on-screen overlap count; hard cap prevents GPU overload.

---

## 6. Interaction

### Camera (`_RightPanCamera`)

Extends vispy's `PanZoomCamera`:

| Input | Action |
|-------|--------|
| Scroll wheel | Zoom in/out |
| Right-drag | Pan |
| Left-drag | Rubber-band box selection |
| Shift + left-drag | 2D rotation |
| Shift + left-click | Place ruler start |
| Left-click (ruler active) | Lock ruler end |
| Right-click | Clear ruler |

### Selection system

| Type | Visual | Behaviour |
|------|--------|-----------|
| **Hover** | Floating tooltip | KD-tree `query_ball_point` lookup with radius = `max(max_marker_radius, 5 px in data units)`. Shows shot #, X, Y, dwell. Hidden automatically on click selection. |
| **Click** | Gold overlay marker + tooltip | Select/deselect single shot; highlight connection lines. Mode-aware hit radius: disc uses `size × _DISC_SIZE_SCALE / 2`, Gaussian uses `size / 2`. |
| **Box** | Green overlay markers + side pane | Rubber-band selection mapped through rotation transform; AABB pre-filter then cross-product point-in-quad test run on a background `QThread`; wait cursor shown during computation. Side pane sorts indices on a second background thread with a "loading…" indicator. |

The KD-tree is built asynchronously on a background `QThread` on the currently rendered (decimated) subset, not the full dataset. It is rebuilt each time the decimated set changes (100 ms debounced camera event). Because the rendered set is always ≤ 2M shots, the KD-tree stays small regardless of file size. The returned hit index is remapped through `_rendered_indices` to recover the original data array index. Hover is momentarily unavailable during each rebuild window.

### Selection marker sizing

Selection markers (click and box) are inflated relative to the underlying shot to remain visible at all zoom levels:

1. Floor-clamp to minimum screen pixels: `_MIN_SEL_PX = 7` (click), `_MIN_BOX_SEL_PX = 4` (box)
2. Stride inflation: `size *= 1 + amp × log₁₀(stride) / (log₁₀(stride) + 0.2)` where `amp` is mode-specific (`_disc_inflate_amp` or `_stride_inflate_amp`)
3. Display scale factor applied (0.667 for disc, 1.0 for Gaussian)
4. 5 % extra margin: `size *= 1.05`

### Ruler

A data-space measurement tool for measuring distances between two points.

**Visual elements**:
- `_ruler_line`: solid white line (`(1, 1, 1, 0.9)`, width 2)
- `_ruler_ticks`: perpendicular tick marks (`(1, 1, 1, 0.7)`, width 1) at 1-2-5 intervals
- `_ruler_label`: `QLabel` at the ruler midpoint showing distance

**Interaction**:
1. **Shift + left-click** → places ruler start point, begins active tracking
2. **Mouse move** (while active) → updates ruler end to cursor position, redraws line + ticks + label
3. **Left-click** (while active) → locks ruler end point
4. **Right-click** (without dragging) → clears ruler. Right-click-drag (pan) does not clear.

**Tick mark logic**:
- Compute ideal interval: `10^floor(log₁₀(distance / 10))`
- Choose smallest candidate from `[1, 2, 5, 10] × magnitude` that is ≥ ideal
- Ticks placed at `0, interval, 2×interval, …` along the ruler
- Perpendicular half-length: 6 pixels in data units

**Distance label format**:
- ≥ 1 mm → `"X.XXX mm"`
- ≥ 1 µm → `"X.XXX µm"`
- Otherwise → `"X.X nm"`

The ruler is redrawn on camera changes (zoom/pan/rotate) to keep tick sizes correct.

### Rotation

Rotation is applied as a `MatrixTransform` on a parent `Node` that holds all visuals. The camera's own transforms remain unaffected. Axis arrowheads (`_AxisArrowOverlay`) are painted on a transparent Qt overlay widget and track orientation via line-rectangle intersection. When the axis line intersects the viewport, arrows are placed at both intersection points. When the axis line does not intersect the viewport (origin panned far off-screen), arrows are clamped to the nearest viewport edge so they remain visible at all times.

---

## 7. UI

### Menu structure

```
File
├── Open Pass File… (Ctrl+O)
├── Incremental Open… (add files to current view)
└── Exit (Alt+F4)

View
├── Show Shot Connections (Ctrl+L)
├── Reset View (Ctrl+R)
├── Show Selection Pane (Ctrl+S)
├── ─────────────────────
├── FWHM: [====slider====] 60.00 nm/µs
├── ─────────────────────
├── Marker Mode ►
│   ├── ● Disc
│   └── ○ Gaussian
├── Wafer Outline ►
│   ├── ● None
│   ├── ○ 2" (51 mm)
│   ├── ○ 4" (100 mm)
│   ├── ○ 5" (125 mm)
│   ├── ○ 6" (150 mm)
│   ├── ○ 8" (200 mm)
│   ├── ○ 12" (300 mm)
│   └── ○ 18" (450 mm)
├── ─────────────────────
├── Disc Alpha Controls ►        [enabled in Disc mode]
│   ├── Overlap (ow): 0.050
│   ├── dpp lo: 0.010
│   ├── dpp hi: 1e+10
│   ├── dpp mid: 5000.0
│   ├── α mid (f_mid): 0.20
│   ├── Inflate: 0.50
│   └── Edge (antialias): 2.0
├── Gaussian Alpha Controls ►    [enabled in Gaussian mode]
│   ├── d_ref: 16.5
│   └── α max: 0.330
├── ─────────────────────
└── Colors ►
    ├── Shot Color       (8 presets + Custom…)
    ├── Click Highlight  (6 presets + Custom…)
    ├── Box Highlight    (6 presets + Custom…)
    ├── Connection Lines (6 presets + Custom…)
    └── Line Highlight   (6 presets + Custom…)

Volume
└── [====slider====] 0–100       [volume control for background music]

Help
├── About
└── Controls
```

### File open validation

When the user selects one or more `.pass` files via File → Open (multi-select is supported), each file is validated before loading:

1. **Missing meta file**: if no companion `.pass.meta` exists, the parser checks for an embedded header in the `.pass` file. If neither is found, the file is opened with a default empty header (no origin offset, no stripe metadata).
2. **Invalid meta file**: if the `.pass.meta` file exists but cannot be parsed (wrong magic number, too small, corrupt), a warning dialog is shown with the specific error and the file is not opened.

### Origin offset and coordinate system

After parsing, the stripe origin from the `.pass.meta` header (`stripeOriginX`, `stripeOriginY`) is added to each shot's raw X/Y coordinates, converting them to absolute wafer coordinates in nanometres. This arithmetic uses float64 to avoid precision loss at wafer-scale magnitudes (billions of nm). The viewer's centroid shift then subtracts the mean position and casts to float32 for GPU rendering, preserving sub-nm precision.

### Incremental open

**File → Incremental Open…** opens a separate file dialog whose selected files are appended to the current view instead of replacing it. Each file's shots are independently offset by their stripe origin, so shots from different stripes align in wafer coordinates. The status bar shows the number of loaded files, total shot count, and combined file size. Opening via the normal **File → Open…** always replaces all loaded data.

When adding files incrementally, `viewer.append_data(new_data)` is called instead of `load_data`. This skips all O(N_existing) CPU work:

- Coordinates are centroid-shifted and cast to float32 **only for the new shots** (existing `_all_positions` is preserved)
- Dwell sizes, raw dwells, and per-shot priorities are computed and appended for the new shots only
- The data bounding box is extended (min/max with new shots) rather than recomputed from scratch
- `self._data` is updated by concatenating only the new x/y/dwell arrays onto the existing ones

The GPU re-upload (`set_data`) is still a full replace — vispy doesn't support appending to a vertex buffer and the decimated subset changes when N grows — but the argsort over all priorities is still O(N_total log N_total), so it's kicked off on a background thread as usual. The per-shot camera view is preserved (the origin doesn't change).

### FWHM slider

Logarithmic: slider value ∈ [−200, 200] maps to scale = $10^{v/100}$, giving effective range 0.1–1000 nm/µs. Default is 100 nm/µs (slider value 100).

### Disc Alpha Controls

Each control is a horizontal slider with label and numeric value display, exposed as a submenu of View. These control the piecewise log-linear alpha curve used by the disc renderer.

| Control | Slider Range | Default | Formula | Description |
|---------|-------------|---------|---------|-------------|
| **Overlap (ow)** | [−300, 0] | −130 | $10^{v/100}$ | Additive white alpha per overlap (0.05) |
| **dpp lo** | [−200, 600] | −200 | $10^{v/100}$ | DPP at which disc becomes fully opaque (0.01) |
| **dpp hi** | [−200, 1000] | 1000 | $10^{v/100}$ | DPP at which disc becomes fully transparent (1e10) |
| **dpp mid** | [−200, 1000] | 370 | $10^{v/100}$ | Intermediate DPP breakpoint (5000.0) |
| **α mid (f_mid)** | [0, 100] | 20 | $v/100$ | Alpha value at dpp_mid (0.20) |
| **Inflate** | [0, 2000] | 50 | $v/100$ | Stride inflation amplitude (0.50) |
| **Edge (antialias)** | [0, 200] | 20 | $v/10$ | Vispy antialias width in pixels (2.0) |

### Gaussian Alpha Controls

| Control | Slider Range | Default | Formula | Description |
|---------|-------------|---------|---------|-------------|
| **d_ref** | [−100, 300] | 122 | $10^{v/100}$ | Sigmoid midpoint DPP (16.5) |
| **α max** | [−300, 0] | −48 | $10^{v/100}$ | Per-shot alpha cap at close zoom (0.330) |

### Colour presets

**Shot Color** (8 presets): Bright Blue (default), Cyan, Green, Magenta, Orange, Red, Gold, White — each with Custom… picker.

**Click Highlight**: Gold (default), White, Magenta, Cyan, Red, Orange.

**Box Highlight**: Green (default), Cyan, White, Magenta, Yellow, Orange.

**Connection Lines**: Red (default), Orange, Yellow, Green, Cyan, Gray.

**Line Highlight**: Gold (default), White, Cyan, Magenta, Green, Orange.

### Selection Pane

Dockable side panel with a virtual `QTableView`. Columns: Shot #, File, X (nm), Y (nm), Dwell (ns). Supports Ctrl+A/Ctrl+C for clipboard copy in tab-separated format.

**Qt 32-bit overflow constraint**: `QHeaderView::length()` = `rowCount × defaultSectionSize` uses 32-bit integer arithmetic. At the default section height (~30 px) this overflows `INT_MAX` at ~71 M rows, causing Qt to render a blank viewport. The model therefore reports `rowCount = min(total, _MAX_VIRTUAL_ROWS)` where `_MAX_VIRTUAL_ROWS = 10 000 000` (10 M rows × 30 px = 300 M, well below `INT_MAX`).

When the full selection exceeds `_MAX_VIRTUAL_ROWS`, a horizontal offset scrollbar appears below the table labelled "Rows 1 – 10,000,000 of N". Dragging it shifts the model's `_offset` pointer so the visible window slides through the full `_indices` array. All shots remain accessible; none are dropped. `highlight_shot()` auto-navigates the window when the target row falls outside the current view.

### Status bar

Displays: filename, shot count, stripe number, resolution, BSS, origin, file size. Shows "Building spatial index…" while KD-tree is under construction.

### Shot count overlay

An in-canvas label shows the current decimation state:

```
10,253,771 total  |  8,412,003 visible  |  stride 8.4  |  1,001,429 rendered  |  dpp 33.70  |  a 0.005
```

Fields: total shots in file, visible (viewport-culled), active stride, rendered count, data-per-pixel, current per-shot alpha.

### Wafer outline

**View → Wafer Outline** provides a submenu of standard wafer diameters. The circle is rendered by `_WaferOutlineOverlay`, a transparent `QWidget` child of the GL canvas (same pattern as `_FiducialOverlay`). Three concentric `QPainter.drawEllipse` strokes produce a real alpha falloff:

| Pass | Width | Colour (RGBA) | Purpose |
|------|-------|---------------|---------|
| Outer glow | 10 px | `(255, 80, 80, 40)` | Wide soft halo |
| Mid halo | 3.5 px | `(255, 80, 80, 110)` | Intermediate layer |
| Core | 1.5 px | `(255, 110, 110, 210)` | Bright centre line |

The overlay is updated by `_reposition_wafer_outline()`, called from `__on_camera_change_inner` (every pan/zoom) and from `set_wafer_outline()`. Screen-space centre is computed via `_data_to_canvas(-self._origin)` and radius via `diameter_nm / 2 / nm_per_px`.

**Why QPainter not vispy Line:** Two vispy Line layers composited on identical pixel geometry don't produce visible width-based falloff — the GPU blends them on the same pixels instead of extending the stroke outward. `QPainter.drawEllipse` with different stroke widths draws concentric rings at different radii, which is the correct rendering.

### Stripe region hover and file selection

When pass files are loaded, the viewer stores each file's stripe rectangle metadata (origin, width, length). When the user hovers inside a stripe's bounding box, a yellow rectangle outline appears around that stripe's region and a tooltip displays:

- File name, shot count
- Origin (X, Y), width, length
- SubField height, overlap

The tooltip anchors to the rectangle's top-right corner. If the user zooms in so the rectangle boundary is off-screen, the tooltip moves to the lower-right corner of the canvas. Only one stripe rectangle is shown at a time — the one under the cursor. The rectangle uses a single reusable `visuals.Rectangle` that is repositioned on hover.

**Right-click file selection:** A right-click (small drag) while hovering over one or more stripe regions opens a context menu. Each hovered file's name appears as a checkable item. Checking a file:
1. Selects all its shots (equivalent to rubber-band box-selecting the entire stripe) — pushed to the viewer via `select_shots()` and displayed with green overlay markers and the selection side-pane.
2. Pins the file's boundary rectangle and metadata label permanently in the scene (cyan border), so they remain visible even when the cursor moves away.

Unchecking a file removes its shots from the selection and unpins its boundary. Multiple files can be selected simultaneously; their shot index ranges are concatenated. `MainWindow` tracks the set of file-selected file indices in `_file_selected`. Pinned visuals use separate `visuals.Rectangle` instances (one per pinned file) stored in `_pinned_stripes`. Pinned metadata labels are repositioned on every camera change.

### Column-position fiducial markers

**View → Column Positions** (MB200 / MB300) overlays crosshair + circle markers at the nominal electron-column positions. Markers are painted by `_FiducialOverlay`, a transparent `QWidget` overlay that uses `QPainter` with antialiasing and `drawEllipse` for a perfect circle.

| Property | Value |
|----------|-------|
| Colour | Light yellow `QColor(255, 255, 140, 217)` |
| Arm half-length | 10 mm (`_FIDUCIAL_ARM_NM`) — scales with zoom |
| Circle radius | 3.5 mm (`_FIDUCIAL_CIRCLE_NM`) — intersects the cross arms; scales with zoom |
| Stroke width | 2 px fixed screen space |

Arm length and circle radius are computed each camera change as `nm / nm_per_px`, derived from `camera.rect.width / canvas_pixel_width`, so the geometry scales with zoom. A minimum of 5 px (arm) / 3 px (circle) prevents markers from collapsing to a point when fully zoomed out.

`_reposition_fiducials()` is called on every camera change and whenever `set_column_positions()` is called. It converts each fiducial's world position to screen coordinates via `_data_to_canvas`, clips to the viewport with a margin, and updates `_fiducial_overlay.markers`. Labels are painted directly by the overlay using `QPainterPath.addText`.

**Note:** an earlier implementation used vispy `Line` visuals in world space, but vispy's `Line` does not support boolean break arrays for the `connect` parameter, causing the circle to render incorrectly. QPainter's `drawEllipse` is the correct tool here.

### Coordinate readout

A bottom-left overlay label shows the current mouse position in data space:

```
X: 12,345 nm   Y: 67,890 nm
```

Updated in real time as the mouse moves.

### Audio system

Background music via `QMediaPlayer` + `QAudioOutput`. The Volume menu contains a slider (0–100). Volume = 0 pauses playback; any non-zero value starts playback at that volume. The MP3 is set to infinite loop.

**Track**: "The Shape of Things to Come" by High Skies, from the album *Sounds of Earth* (2010).

---

## 8. Entry Point & Software Rendering

### `main.py`

```python
def _check_opengl() -> None:
    """Switch to Qt's software OpenGL if the system has no usable GPU driver."""
    if os.environ.get("QT_OPENGL"):
        return  # user already chose
    try:
        import ctypes
        gl = ctypes.windll.opengl32
        if not gl.wglGetProcAddress:
            raise OSError
    except Exception:
        os.environ["QT_OPENGL"] = "software"
```

This function runs before any Qt or vispy imports. On machines with no GPU (e.g. Microsoft Basic Display Adapter providing only OpenGL 1.1), it sets `QT_OPENGL=software`, which causes Qt to use its bundled Mesa llvmpipe software renderer (`opengl32sw.dll`). Users can override by pre-setting the `QT_OPENGL` environment variable.

The application accepts an optional `.pass` file path as `sys.argv[1]` for direct launch.

---

## 9. Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.14 |
| GUI framework | PyQt6 | ≥ 6.5 |
| GPU rendering | vispy (gl+ backend with PyOpenGL, gl2 fallback) | ≥ 0.14 |
| Software GL fallback | Mesa llvmpipe via Qt's `opengl32sw.dll` | — |
| Audio | PyQt6 `QMediaPlayer` + `QAudioOutput` | — |
| Spatial indexing | scipy `cKDTree` | ≥ 1.10 |
| Numerics | numpy | ≥ 2.0 |
| Graphics API | OpenGL 3+ (instanced rendering), GL2 fallback, Mesa software | — |
| Packaging | PyInstaller (single-file exe) | 6.18 |

### Why vispy?

- Instanced marker rendering handles millions of points without per-point draw calls
- Scene graph with camera transforms eliminates manual matrix math
- Shader subclassing (`GaussianMarkersVisual`) enables custom fragment shaders without forking
- PyQt6 integration via `vispy.app.use_app("pyqt6")`

---

## 10. Build & Distribution

### PyInstaller spec (`PassFileViewer.spec`)

Single-file windowed exe. Key configuration:

**Bundled data files**:
- `vispy/glsl/` — GLSL shader source files
- `vispy/io/_data/` — vispy built-in data
- `high_skies-the_shape_of_things_to_come.mp3` — background music
- `opengl32sw.dll` — Qt's Mesa software OpenGL renderer (from `PyQt6/Qt6/bin/`)

**Hidden imports** (not auto-detected by PyInstaller):
- `vispy.app.backends._pyqt6` — vispy PyQt6 backend
- `vispy.gloo.gl.glplus` — GL3+ renderer
- `vispy.gloo.gl.gl2` — GL2 fallback
- `vispy.gloo.gl.desktop` — desktop GL
- `PyQt6.sip` — Qt interface language
- `PyQt6.QtMultimedia` — QMediaPlayer + QAudioOutput

**Windows version info**: loaded from `version_info.txt` — v1.2.0.0, "Pass File Viewer", © 2026 Multibeam Corporation.

### Version control

The `pass_viewer/` directory is a Git repository with remote at `https://github.com/mcatha/pass_file_viewer.git`. The parent directory (`Pass files/`) is not under version control — only the source files inside `pass_viewer/` are tracked.

---

## 11. Performance Characteristics

| Operation | Technique | Impact |
|-----------|-----------|--------|
| File parsing | mmap + numpy 64-bit vectorised bitfields (no intermediate copy) | ~2 s for 10M shots |
| KD-tree build | Background QThread on decimated subset (≤ 2M shots); rebuilt per decimation update | Small and fast; scales to any file size |
| GPU upload | Uniform colour scalar (not per-point array); scalar size when dwells identical | Saves N×16+ bytes |
| Zoom-out | Priority-based decimation + 2M hard cap | Keeps rendered count ≤ 2M |
| Viewport cull | AABB mask before decimation | Only process visible shots |
| Connection lines | Decimated line segments (max 500K) | Avoids GPU overload |
| Priority sort | `np.argsort` deferred to background `QThread`; `np.argpartition` O(n) fallback | Main thread unblocked at load |
| Box selection geometry | AABB pre-filter + quad edge test on background `QThread` | Main thread unblocked during large selections |
| Box selection marker upload | Viewport cull skipped for selections >1M; cache key prevents redundant GPU uploads per frame | Pan/zoom stays smooth with large selections active |
| Selection pane sort | `np.sort` on background `QThread`; "loading…" indicator shown immediately | Main thread unblocked for large sorted panes |
| Hover lookup | Query dispatched to background `QThread`; at most one in-flight at a time; latest position queued if busy | Main thread never blocks; tooltip tracks cursor continuously |

---

## 12. Image-to-Pass Converter (`img_to_pass.py`)

Utility script that converts a PNG image to a synthetic `.pass` file. Brightness is mapped to disc overlap density and dwell variation.

### Algorithm

1. Load PNG → grayscale + alpha → brightness = luminance × alpha ∈ [0, 1]
2. Flip Y axis (viewer Y increases upward, image Y is downward)
3. Gamma correction: `brightness = brightness ^ 2.5` (crushes darks for contrast)
4. Scale: `brightness *= 0.7` (reduce overall intensity)
5. Per-pixel shot count: `round(brightness × MAX_LAYERS)` (0–6 discs per grid cell)
6. Per-shot jitter: uniformly random ±`JITTER` nm
7. Per-shot dwell: `base × brightness + ±50% noise`, clamped to 14-bit max

### Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DWELL_MIN` | 50 | Minimum dwell time (ns) |
| `DWELL_MAX` | 8000 | Maximum dwell time (ns) |
| `SCALE` | 30 | Grid cell size (nm per pixel) |
| `MAX_LAYERS` | 6 | Maximum disc overlaps per pixel |
| `JITTER` | 12 | Per-shot random offset (nm), 40 % of cell size |

### Output format

A `.pass` file (headerless 8-byte shot records) and a companion `.pass.meta` file containing the metadata header.

---

## 13. MB300 Logo Pass Generator (`logo_passes.py`)

Generates a set of `.pass` files that write the Multibeam logo (`mb-logo-w-tag.png`, 600×145 px) onto a 300 mm wafer using the MB300 system. One embedded-v4-header `.pass` file is written per active (column, master-pass) combination.

### MB300 column layout

The MB300 has **18 individual beam columns**. The number suffix (1–4) gives the X position; the letter (A–E) gives the Y position.

| Column group | Beam Y | Y section | Beam X positions | X section |
|---|---|---|---|---|
| A2, A3, A4 | +130 mm | +97.5 → +162.5 mm | +75, 0, −75 mm | ±112.5 mm (3 cells, 75 mm each) |
| B1, B2, B3, B4 | +65 mm | +32.5 → +97.5 mm | +112.5, +37.5, −37.5, −112.5 mm | ±150 mm (4 cells, 75 mm each) |
| C1, C2, C3, C4 | 0 mm | −32.5 → +32.5 mm | +112.5, +37.5, −37.5, −112.5 mm | ±150 mm (4 cells, 75 mm each) |
| D1, D2, D3, D4 | −65 mm | −97.5 → −32.5 mm | +112.5, +37.5, −37.5, −112.5 mm | ±150 mm (4 cells, 75 mm each) |
| E2, E3, E4 | −130 mm | −162.5 → −97.5 mm | +75, 0, −75 mm | ±112.5 mm (3 cells, 75 mm each) |

All cells are **75 mm wide × 65 mm tall**. Outer B/C/D columns (col 1 and col 4) are clamped to the logo ±125 mm X boundary for shot generation.

Only C-row cells (Y section ±32.5 mm) overlap the logo's ±30.2 mm Y extent. All other rows produce no shots but are included in the sweep. A `valid_y` boolean mask prevents out-of-range rows from generating spurious edge shots.

### Stage motion

- Master sweep: **N_MASTER = 1 250** stage-X positions at 60 µm steps, P_X from −37.5 mm to +37.44 mm.
- At each stage position n the stage sweeps in Y over the column's full Y section.
- **Serpentine**: odd-numbered passes scan −Y (shots ordered Y descending, `sortDirection = −1`); even passes scan +Y (ascending, `sortDirection = +1`).

### Physical parameters

| Parameter | Value |
|---|---|
| Logo size | 289.7 mm × 70.0 mm, centred on wafer; corners at 149 mm from centre (1 mm inside wafer edge) |
| Shot pitch X | 900 nm |
| Shot pitch Y | 779 nm (hexagonal close-packed: `round(900 × √3/2)`) |
| Shot density | ~1 426 330 shots/mm² |
| Estimated output | ~8.93 B shots, ~71.4 GB |
| Beam FWHM | 1600 nm (dwell 16 000 ns × 100 nm/µs viewer default) |
| Dwell | 16 000 ns (16 µs) |
| Pass width (X step) | 60 000 nm (60 µm) |

### Image sampling

```python
lum      = (0.299*R + 0.587*G + 0.114*B) / 255.0
alpha    = A / 255.0
on_white = lum * alpha + (1.0 - alpha)   # composite onto white background
dark     = on_white < 0.5                # True = write a shot
dark     = dark[::-1, :]                 # flip Y: row 0 → logo bottom on wafer
```

Compositing onto white before thresholding correctly handles transparent pixels
(background) — without it, transparent pixels have `lum × alpha = 0 < 0.5` and
would incorrectly be treated as dark.  The logo has ~30.9 % fill (26 860 / 87 000
pixels are dark after compositing).

Shot positions within each pass use **bilinear interpolation** at fractional image
coordinates (float pixel index with −0.5 offset) for smooth, anti-aliased edges.
Shots are placed on a **hexagonal close-packed grid**: even-indexed X columns use
row grid A; odd X columns use grid B (offset by `PITCH_Y/2` in Y), forming a
triangular lattice with 100 % area coverage (maximum uncovered radius ≈ 736 nm,
less than the 800 nm beam half-width).

### v4 embedded header

78-byte little-endian struct `"<IHHiiIIdHHdiQQQI??"`, magic `0xB3D11982`, version 4. `stripeOriginX` is the clipped pass X start in wafer coordinates; `stripeOriginY` is `LOGO_Y_MIN`. Shot coordinates are local to the stripe origin.

### Output

Files written to `../logo_passes/` (i.e. `Pass files/logo_passes/`) named `{name}_{n:04d}.pass` where `name` is the beam label (e.g. `C2_0001.pass`).

---

## 14. Beam Diagram Scripts

Two standalone matplotlib scripts generate diagrams of the MB300 column layout. Both output to `Pass files/`.

| Script | Output | Purpose |
|---|---|---|
| `beam_diagram.py` | `beam_areas.png` | Annotated diagram showing all 18 column cells with logo boundary overlay (yellow) and colour coding for cells that overlap the logo Y range |
| `beam_diagram_tool.py` | `beam_areas_tool.png` | General tool diagram — no logo references, uniform cell colour, larger cell labels (upper-left corner of each cell to avoid overlapping the beam initial position marker) |

Both show the 300 mm wafer outline as a dashed circle. A/E outer columns are drawn at their full unclipped ±162.5 mm extent to show the true equal-size grid. Beam initial positions (the physical beam location within each cell) are marked with a red `+`.

---

## 15. Easter Eggs

- **novus_ordo.pass**: when a file with stem `novus_ordo` (case-insensitive) is loaded, the shot colour automatically switches to dollar bill green `(0.33, 0.54, 0.18, 1.0)`. The companion `novus_ordo.png` image (detail from a US dollar bill) can be converted to a `.pass` file using `img_to_pass.py`.

---

## 16. File Listing

```
pass_viewer/
├── main.py                                      # Entry point, OpenGL driver check
├── main_window.py                               # QMainWindow: menus, audio, status, file loading
├── viewer_widget.py                             # GPU viewer: rendering, decimation, interaction, ruler
├── gaussian_markers.py                          # GaussianMarkers visual (custom Gaussian PSF shader)
├── selection_pane.py                            # Box-selection data table
├── pass_parser.py                               # Binary .pass / .pass.meta parser
├── img_to_pass.py                               # PNG → .pass image converter
├── logo_passes.py                               # MB300 logo pass generator (18-column, v4 header)
├── beam_diagram.py                              # Column area diagram with logo overlay
├── beam_diagram_tool.py                         # Column area diagram (general tool, no logo refs)
├── generate_test_pass.py                        # Test data generator (random shots)
├── generate_chip_pass.py                        # Test data generator (IC layout + fractals)
├── generate_icon.py                             # App icon generator
├── app_icon.ico                                 # Application icon
├── high_skies-the_shape_of_things_to_come.mp3   # Background music (looping)
├── novus_ordo.png                               # Easter egg source image
├── version_info.txt                             # Windows exe version resource (v1.2.0.0)
├── requirements.txt                             # Python dependencies
├── DESIGN.md                                    # This document
└── PassFileViewer.spec                          # PyInstaller single-file exe spec
```

---

## 17. Known Limitations

1. **No shape record rendering** — only shot records (mID == 0) are displayed
2. **No inter-file zoom-in reload** — zooming into a region does not immediately re-read files at stride 1; the viewport debounce (500 ms) triggers `_update_viewport_files`, which recomputes stride for the new viewport and evicts/reloads as needed
3. **Debug prints** — diagnostic print statements (`[disc]`, `[gauss]`, `[SEL]`, `[stride]`, `[upload]`, `[axis]`, `[INIT]`, `[load]`) are still active in `viewer_widget.py`

---

## 18. Viewport-Aware Lazy Loading

### Problem

Opening a large multi-file pass set requires decoding all shots into float64 x/y
arrays, which can exhaust available RAM.  The MB300 logo write (~8 930 files,
~71 GB, ~8.93 B shots at 900 nm hex pitch) already exceeds memory on 16 GB
machines.  Production IC patterns may be substantially larger — data volumes for
real chip writes are unknown, as this is a new technology.  Target machines have
16–64 GB.

### Solution

**For multi-file opens exceeding `_LAZY_LOAD_BYTES` (400 MB):**

1. **Header scan** — `_HeaderScanWorker` reads only the first 78 bytes of each
   file (or its `.pass.meta`) using a 32-thread pool.  Builds a `FileIndex` of
   `FileEntry` objects (spatial bounding box + shot count) without loading any
   shot data.  Typically completes in 1–5 s for 14 000+ files on a local SSD.

2. **`FileIndex`** — list of `FileEntry` dataclasses with spatial bbox query
   (`FileIndex.query(x0, y0, x1, y1) → list[FileEntry]`).

3. **`FileCache`** — cache of parsed `PassData` objects keyed by `(Path, stride)`
   and bounded by `_RAM_BUDGET_SHOTS = 50_000_000` (≈1.8 GB).  Including stride
   in the key means a zoom change that raises or lowers the stride automatically
   invalidates stale cache entries.  `put()` evicts LRU entries when the
   total shot count would exceed the budget.

4. **`_update_viewport_files(x0, x1, y0, y1)`** — called after header scan and
   on every debounced viewport change (500 ms):
   - Queries `FileIndex` for all candidates intersecting the viewport.
   - Computes `stride = ceil(total_candidate_shots / _RAM_BUDGET_SHOTS)`.
     All candidate files are loaded; stride > 1 thins the shot records so the
     in-RAM total stays within budget.
   - Evicts cached `(path, stride)` pairs no longer in the desired set.
   - Dispatches `_MultiParseWorker` to load any missing `(path, stride)` pairs.

   **Why adaptive stride beats spatial sampling:** spatial sampling opened ~50
   scattered files per viewport update (random I/O across the directory), which
   both looked wrong (thin stripes instead of full pattern) and was slower.
   Adaptive stride opens every file but reads only every N-th record — sequential
   I/O within each file followed by a fast numpy slice is much cheaper than
   scattered seeks.

5. **`_rebuild_viewer_data()`** — merges cached files into a single `PassData`
   and calls `viewer.load_data(merged, fit_view=fit)`.  Uses thin `PassData`
   copies (empty arrays, correct `count`) in `_loaded_files` to avoid storing
   shot coordinates twice.  `fit_view=True` only on first load.

6. **Viewport signal** — `ShotViewerWidget.viewport_rect_changed(x0, x1, y0, y1)`
   is emitted on every camera change and consumed by `MainWindow` to drive
   `_update_viewport_files` via a 500 ms debounce timer.

### Constants

| Constant | Value | Meaning |
|---|---|---|
| `_LAZY_LOAD_BYTES` | 400 000 000 | Total file size threshold to activate lazy loading |
| `_RAM_BUDGET_SHOTS` | 50 000 000 | Max shots kept in RAM cache at once |

### Behaviour at different zoom levels

| Zoom | Files in viewport | Loaded |
|---|---|---|
| Full wafer (all files) | N (all) | All files loaded at stride K — full pattern visible, thinned |
| Sub-region | Fraction of N | All in-viewport files, lower stride (more shots per file) |
| Single stripe | 1 | That file at stride 1 — every individual shot accessible |

---

---

## 19. Threading and State Ownership

### Principle 1 — Every entry path into a subsystem must reset all state that subsystem owns

When an operation "takes ownership" of a resource (the file index, the cache, the loaded data), it must reset that resource on **every path** that initiates a new operation, not just the primary one.

In practice: `_file_index` and `_file_cache` were reset on the lazy-load entry path but not on the single-file or small-batch paths. Any pending viewport debounce timer fires after the new load completes, queries the stale `_file_index`, and reloads the old files over the new data. The fix: reset shared state at the earliest common point — before branching into individual load paths — not inside each branch.

**Rule:** if a variable is cleared in some entry paths but not others, it will carry stale state into the next operation on the paths that skip the clear. State that belongs to an operation must be cleared before that operation starts, unconditionally.

### Principle 2 — Thread completion callbacks must verify ownership before modifying shared state

Qt signals connected to `thread.finished` are queued cross-thread signals. They fire in the main event loop **after** `wait()` returns — potentially after a new thread has already been assigned to the same slot variable. An unconditional clear (`self._parse_thread = None`) will null out the active thread's reference if the stale signal fires second.

The correct pattern — already used for the header scan thread — is to capture the thread reference in a lambda and check identity before acting:

```python
thread.finished.connect(lambda t=thread: self._on_foo_thread_done(t))

def _on_foo_thread_done(self, thread) -> None:
    if self._foo_thread is thread:
        self._foo_thread = None
        self._foo_worker = None
```

**Rule:** every `thread.finished` callback that modifies shared state must confirm it is being called for the thread that currently owns that state. An unconditional clear is always wrong when multiple sequential threads use the same slot variable.

### Principle 3 — When a deferred computation replaces a fallback, it must immediately invalidate the fallback's output

The priority argsort runs on a background thread after load. Before it finishes, `_priority_indices` falls back to a different selection algorithm. When the argsort completes, simply storing the result is not enough — the display still reflects the fallback's output until the next camera event. Any fallback that produces a different result than the real computation must trigger a re-render when the real result arrives.

**Rule:** if a deferred result changes what would be computed, completing that deferred work must invalidate the cache key and re-run the dependent computation, not just store the result for the next query.

---

## 20. Open Questions

This tool is being developed alongside a new technology (MB300 multi-beam EBL
for IC production).  Real chip pattern data does not yet exist.  Several design
assumptions are therefore provisional:

### Data volumes

The only large write jobs tested so far are the MB300 logo test patterns (~71 GB,
~8.93 B shots).  For production IC patterns at 50 nm feature sizes the shot pitch
will be an order of magnitude finer, shot counts per die will be vastly higher,
and the number of dies per wafer is unknown.  Total write job size for a real chip
pattern could range from manageable to orders of magnitude larger than the logo
test.

### Performance bottlenecks

Several potential bottlenecks have been identified but not validated against real
data:

| Concern | Status |
|---|---|
| RAM spike on numpy boolean-mask filtering | Real but low priority at current logo-scale volumes |
| Vispy scene-graph overhead at high file counts | Not a concern — single Visual node used, not one per file |
| Header scan speed for very large file sets | Handled by 32-thread pool; ProcessPoolExecutor not needed for I/O-bound work |
| Zoom-out density representation | Unsolved — splatting fails when decimation stride is large; density texture requires reading all shot data |

### Zoom-out density display

The current approach (priority-based decimation + Gaussian splatting) produces a
faithful image when individual shots are visible, but breaks at extreme zoom-out:
the surviving decimated shots are too far apart to overlap, so the pattern reads
as a sparse point cloud instead of a solid shape.

The correct fix is a pre-integrated density image (2D histogram of all shots),
displayed as a `vispy.scene.visuals.Image` when `dpp > shot_pitch`.  Building
this requires reading all shot data — either a one-time background strided-read
pass at open time, or accepting that the image represents only the loaded cache.
The right trade-off cannot be determined without knowing real chip pattern data
volumes and the storage medium (local SSD vs network share).

**Design principle:** do not build elaborate infrastructure for bottlenecks that
have not materialized in practice.  Defer the density-image implementation until
real IC pattern files exist and the actual zoom-out behaviour can be evaluated.

---

## 21. Glossary
|------|-----------|
| **Shot** | A single electron beam exposure point with X, Y, and dwell time |
| **Dwell** | Duration (ns) the beam stays at a position; determines dose and rendered size |
| **FWHM** | Full Width at Half Maximum — the diameter at which the Gaussian PSF reaches 50 % of peak intensity |
| **Stripe** | A horizontal band of shots that the stage traverses in one pass |
| **BSS** | Beam Step Size — the grid pitch of shot placement |
| **Pass file** | Binary `.pass` file containing packed shot/shape records (no header); accompanied by a `.pass.meta` companion file with stripe metadata in MEBL2 format |
| **Decimation** | Reducing the number of rendered shots via priority-based selection from the visible set |
| **Additive blending** | OpenGL blend mode where source colour is added to framebuffer (`src_alpha, one`) |
| **Alpha blending** | Standard transparency blend that converges to source colour (`src_alpha, one_minus_src_alpha`) |
| **dpp** | Data units per pixel — the true zoom scale, computed from the full vispy transform chain via Euclidean distance (rotation-invariant) |
| **Priority** | Fixed per-shot random rank that determines decimation survival order |
| **Mesa llvmpipe** | Software OpenGL implementation bundled as `opengl32sw.dll`; used when no GPU is available |
