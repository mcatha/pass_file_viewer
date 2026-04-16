# Pass File Viewer — Design Document

**Version 1.3 — April 2026**

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
| **High-performance rendering** | Instanced OpenGL markers via vispy; handles 10 M+ shots at 60 fps |
| **Disc markers (default)** | Hard-edged discs with additive white overlap layer for brightening toward white |
| **Gaussian PSF markers** | Custom fragment shader renders each shot as a smooth Gaussian with FWHM = shot size |
| **Zoom-adaptive alpha** | Mode-specific alpha curves prevent blowout at wide zoom while keeping close-zoom overlap visible |
| **Adjustable alpha curves** | Disc and Gaussian alpha curves exposed via per-mode slider submenus |
| **FWHM slider** | Logarithmic slider (0.1–1000 nm/µs) controls shot size scaling in real time |
| **Priority-based decimation** | Fixed per-shot priority with dwell bias; mode-specific density budgets |
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
| `*.pass` | Headerless flat stream of 8-byte shot/shape records from byte 0 |
| `*.pass.meta` | Metadata header in MEBL2 packed struct format |

The `.pass` file contains **no header** — every byte is part of an 8-byte record. All metadata (stripe geometry, shot counts, resolution, etc.) lives in the companion `.pass.meta` file, which shares the same base name (e.g. `chip.pass` + `chip.pass.meta`).

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

- Metadata is read from the `.pass.meta` companion file. If no meta file exists, an empty header is used.
- The `.pass` file is read from byte 0 — there is no header offset.
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
| `fwhm_scale` | 6.0 | Multiplier — default gives 60 nm/µs FWHM |
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
4. **Priority-based selection**: each shot is assigned a fixed random priority at load time, with a mild dwell bias (15 %): higher-dwell shots get lower priority values (more likely to survive decimation). The top `budget` shots by priority are selected via `np.argpartition` (O(n)).
5. **Cache gating**: a composite key of quantised viewport bounds + stride + `round(log₂(dpp) × 20)` prevents redundant GPU uploads when the view hasn't meaningfully changed.
6. **All visuals synced**: the same decimated index set is uploaded to all active marker layers.

### Design properties

- **Stable across pan/zoom**: fixed per-shot priorities mean the same shots survive decimation regardless of small viewport shifts — no visible flicker.
- **No rotation artefacts**: viewport bounds use 4-corner mapping; `dpp` is computed from the full transform chain via Euclidean distance, not just the X component.
- **No blowout**: density-based budget limits on-screen overlap count; hard cap prevents GPU overload.
- **Importance-aware**: dwell bias keeps larger exposures visible preferentially.

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
| **Box** | Green overlay markers + side pane | Rubber-band selection mapped through rotation transform; cross-product point-in-quad test |

The KD-tree is built asynchronously on a background `QThread` to avoid blocking the UI during load.

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

Rotation is applied as a `MatrixTransform` on a parent `Node` that holds all visuals. The camera's own transforms remain unaffected. Axis arrowheads (`_AxisArrowOverlay`) are painted on a transparent Qt overlay widget and track orientation via ray-rectangle intersection.

---

## 7. UI

### Menu structure

```
File
├── Open Pass File… (Ctrl+O)
├── Incremental Open (checkbox)
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

When the user selects a `.pass` file via File → Open, two checks run before loading:

1. **Missing meta file**: if no companion `.pass.meta` exists in the same directory, a warning dialog is shown and the file is not opened.
2. **Invalid meta file**: if the `.pass.meta` file exists but cannot be parsed (wrong magic number, too small, corrupt), a warning dialog is shown with the specific error and the file is not opened.

### Origin offset and coordinate system

After parsing, the stripe origin from the `.pass.meta` header (`stripeOriginX`, `stripeOriginY`) is added to each shot's raw X/Y coordinates, converting them to absolute wafer coordinates in nanometres. This arithmetic uses float64 to avoid precision loss at wafer-scale magnitudes (billions of nm). The viewer's centroid shift then subtracts the mean position and casts to float32 for GPU rendering, preserving sub-nm precision.

### Incremental open

When **File → Incremental Open** is checked, opening a new file appends its shots to the existing view instead of replacing. Each file's shots are independently offset by their stripe origin, so shots from different stripes align in wafer coordinates. The status bar shows the number of loaded files, total shot count, and combined file size. Unchecking incremental open and opening a file replaces all data.

### FWHM slider

Logarithmic: slider value ∈ [−200, 200] maps to scale = $10^{v/100}$, giving effective range 0.1–1000 nm/µs. Default is 60 nm/µs (slider value 78).

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

Dockable side panel with a virtual `QTableView` (virtual rows — scales to any selection size). Columns: Shot #, X (nm), Y (nm), Dwell (ns). Supports Ctrl+A/Ctrl+C for clipboard copy in tab-separated format.

### Status bar

Displays: filename, shot count, stripe number, resolution, BSS, origin, file size. Shows "Building spatial index…" while KD-tree is under construction.

### Shot count overlay

An in-canvas label shows the current decimation state:

```
10,253,771 total  |  8,412,003 visible  |  stride 8.4  |  1,001,429 rendered  |  dpp 33.70  |  a 0.005
```

Fields: total shots in file, visible (viewport-culled), active stride, rendered count, data-per-pixel, current per-shot alpha.

### Wafer outline

**View → Wafer Outline** provides a submenu of standard wafer diameters: None, 2" (51 mm), 4" (100 mm), 5" (125 mm), 6" (150 mm), 8" (200 mm), 12" (300 mm), 18" (450 mm). Selecting a size draws a circle of that diameter centered on the wafer origin (0, 0 in absolute coordinates). The circle uses a 256-segment `visuals.Line` with the same axis color `(0.6, 0.6, 0.6, 0.8)`. The outline repositions automatically when new data is loaded (since the centroid shift changes). Selecting "None" hides the circle.

### Stripe region hover

When pass files are loaded, the viewer stores each file's stripe rectangle metadata (origin, width, length). When the user hovers inside a stripe's bounding box, a yellow rectangle outline appears around that stripe's region and a tooltip displays:

- File name, shot count
- Origin (X, Y), width, length
- SubField height, overlap

The tooltip anchors to the rectangle's top-right corner. If the user zooms in so the rectangle boundary is off-screen, the tooltip moves to the lower-right corner of the canvas. Only one stripe rectangle is shown at a time — the one under the cursor. The rectangle uses a single reusable `visuals.Rectangle` that is repositioned on hover.

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

---

## 11. Performance Characteristics

| Operation | Technique | Impact |
|-----------|-----------|--------|
| File parsing | mmap + numpy 64-bit vectorised bitfields | ~2 s for 10M shots |
| KD-tree build | Background QThread | Non-blocking UI |
| GPU upload | Uniform colour scalar (not per-point array); scalar size when dwells identical | Saves N×16+ bytes |
| Zoom-out | Priority-based decimation + 2M hard cap | Keeps rendered count ≤ 2M |
| Viewport cull | AABB mask before decimation | Only process visible shots |
| Connection lines | Decimated line segments (max 500K) | Avoids GPU overload |
| Priority select | `np.argpartition` (O(n)) | No full sort needed |
| Hover lookup | Throttled to 60 fps via 16 ms timer | No redundant KD-tree queries |

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

## 13. Easter Eggs

- **novus_ordo.pass**: when a file with stem `novus_ordo` (case-insensitive) is loaded, the shot colour automatically switches to dollar bill green `(0.33, 0.54, 0.18, 1.0)`. The companion `novus_ordo.png` image (detail from a US dollar bill) can be converted to a `.pass` file using `img_to_pass.py`.

---

## 14. File Listing

```
pass_viewer/
├── main.py                                      # Entry point, OpenGL driver check
├── main_window.py                               # QMainWindow: menus, audio, status, file loading
├── viewer_widget.py                             # GPU viewer: rendering, decimation, interaction, ruler
├── gaussian_markers.py                          # GaussianMarkers visual (custom Gaussian PSF shader)
├── selection_pane.py                            # Box-selection data table
├── pass_parser.py                               # Binary .pass / .pass.meta parser
├── img_to_pass.py                               # PNG → .pass image converter
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

## 15. Known Limitations

1. **Single-file viewer** — no multi-file or stripe-sequence viewing
2. **2D only** — no 3D perspective or Z-axis support
3. **No shape record rendering** — only shot records (mID == 0) are displayed
4. **Debug prints** — diagnostic print statements (`[disc]`, `[gauss]`, `[SEL]`, `[stride]`, `[upload]`, `[axis]`, `[INIT]`, `[load]`) are still active in `viewer_widget.py`

---

## 16. Glossary

| Term | Definition |
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
| **Priority** | Fixed per-shot random rank (biased by dwell) that determines decimation survival order |
| **Mesa llvmpipe** | Software OpenGL implementation bundled as `opengl32sw.dll`; used when no GPU is available |
