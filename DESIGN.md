# Pass File Viewer — Design Document

## 1. Overview

The **Pass File Viewer** is a GPU-accelerated desktop application for visualising binary `.pass` shot files produced by Multibeam's e-beam lithography column control system. It renders millions of shots as size-scaled markers on a 2D Cartesian plane, with real-time pan, zoom, rotate, and interactive selection.

### The central rendering challenge

The viewer must look correct across ~5 orders of magnitude of zoom. At close zoom each shot is many pixels wide and they overlap heavily — additive blending blows out to solid white. At far zoom thousands of shots collapse into a single pixel — if alpha is too low they disappear, too high and sparse regions look the same as dense ones. There is no single blend mode, alpha value, or marker size that works across this range. Every fix at one end creates a problem at the other.

The quadratic alpha curve, priority-based decimation, minimum-size inflation, and mode-specific sizing are all compromises to keep the image reasonable across the full zoom range. None are perfect — they are the least-wrong tradeoffs found so far.

### Two rendering modes

- **Gaussian mode** (default): each shot is a soft Gaussian point-spread function with FWHM-based sizing. A single additive-blend layer accumulates colour; per-shot alpha is zoom-dependent so overlap brightens at close zoom without blowing out when zoomed far. This shows the accumulated dose picture but individual shots blur into each other.
- **Disc mode**: two composited layers — an alpha-blend base for stable appearance plus an additive white overlay for overlap brightening. Hard edges let the user distinguish individual shots, identify specific placements, and verify spacing.

### Key capabilities

| Feature | Description |
|---------|-------------|
| **High-performance rendering** | Instanced OpenGL markers via vispy; handles 10 M+ shots at 60 fps |
| **Gaussian PSF markers** | Custom fragment shader renders each shot as a smooth Gaussian with FWHM = shot size |
| **Disc markers (alt mode)** | Hard-edged discs with additive white overlap layer for brightening toward white |
| **Zoom-adaptive alpha** | Quadratic alpha curve prevents blowout at wide zoom while keeping close-zoom overlap visible |
| **FWHM slider** | Logarithmic slider (0.1–1000 nm/µs) controls shot size scaling in real time |
| **Priority-based decimation** | Fixed per-shot priority with dwell bias; stride compensation keeps brightness stable |
| **Interactive selection** | Single-click and rubber-band box selection with side-pane data table |
| **Shot connections** | Toggle-able lines between consecutive shots |
| **Colour customisation** | 5-category colour menu with presets and custom colour picker |
| **Hover tooltips** | KD-tree spatial lookup shows shot info on mouse hover |
| **2D rotation** | Shift+left-drag rotates the view; axis arrowheads track orientation |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      main.py                            │
│  Entry point: creates QApplication, MainWindow          │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  main_window.py                         │
│  MainWindow (QMainWindow)                               │
│  ├── Menus: File, View (FWHM slider, mode toggle,      │
│  │          Colors submenu), Help                       │
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
| `main.py` | Entry point; command-line `.pass` file argument |
| `main_window.py` | Qt window, menus (FWHM slider, marker mode toggle, colours), background file parsing, status bar |
| `pass_parser.py` | Binary `.pass` parser (v3/v4 auto-detect); numpy vectorised bitfield extraction |
| `viewer_widget.py` | GPU rendering, camera, decimation, selection, tooltips, rotation, axis lines |
| `gaussian_markers.py` | `GaussianMarkers` visual — custom Gaussian PSF fragment + vertex shaders |
| `selection_pane.py` | Virtual table model for box-selected shots; clipboard copy |

---

## 3. File Format — `.pass` v3/v4

### Header

| Version | Size | Extra fields over v3 |
|---------|------|----------------------|
| v3 | 64 bytes | — |
| v4 | 78 bytes | `overlap`, `baseDwellTime`, `debug`, `centerShotPresent` |

Version is auto-detected by checking which header size yields a record count matching `shotCount + shapeCount`.

### Records (8 bytes each, little-endian `uint64`)

| Bits | Field | Width | Description |
|------|-------|-------|-------------|
| 0–1 | `mID` | 2 | 0 = Shot, non-zero = Shape |
| 2–15 | `mDwell` | 14 | Dwell time in ns (max 16383) |
| 16–31 | `mX` | 16 | X coordinate |
| 32–63 | `mY` | 32 | Y coordinate |

Only shot records (`mID == 0`) are rendered. Shape records are skipped.

### Parser performance

- Files >10 MB are memory-mapped (`mmap`) to avoid copying into Python memory.
- Bitfield extraction uses 64-bit numpy arithmetic on a contiguous `uint64` view — no intermediate reshape.
- Shot mask applied once; intermediate arrays freed immediately.

---

## 4. Rendering Architecture

### 4.1 Two rendering modes

The user can switch between Gaussian and Disc modes via **View → Disc Markers**. Both modes share the same decimated point set and selection system; only the GPU visuals differ.

#### Gaussian mode (default)

A single `GaussianMarkers` visual with **additive blending** (`src_alpha, one`).

Each shot is rendered as a smooth Gaussian point-spread function. The `size` parameter is the **FWHM** (Full Width at Half Maximum) — the Gaussian crosses 50 % intensity at exactly `size / 2` from centre.

```
alpha(r) = exp(-4·ln2·r²)     where r = 2·dist / FWHM
```

Per-shot face alpha is **zoom-dependent** via a quadratic curve:

$$t = \frac{\text{dpp}^2}{\text{dpp}^2 + \text{REF}^2}$$

$$\alpha = \alpha_{\text{near}} + (\alpha_{\text{far}} - \alpha_{\text{near}}) \cdot t$$

| Constant | Value | Meaning |
|----------|-------|---------|
| `_ALPHA_NEAR` | 1.00 | Alpha at maximum zoom-in (large overlapping Gaussians) |
| `_ALPHA_FAR` | 0.01 | Alpha at maximum zoom-out (tiny markers) |
| `_ALPHA_REF_DPP` | 20.0 nm/px | Midpoint — alpha is halfway between NEAR and FAR here |

At close zoom, `t → 0` and `alpha → 1.0` (full intensity). At wide zoom, `t → 1` and `alpha → 0.01`. The quadratic scaling matches the fact that overlap density grows as $(size/dpp)^2$.

**Stride size compensation**: when decimation stride > 1, only 1/stride of the shots are drawn. With additive blending, total brightness ∝ ρ × α × FWHM². Since ρ drops to ρ/stride, marker FWHM is inflated by √stride so each surviving shot's Gaussian footprint covers stride× the area. This preserves accumulated brightness smoothly — unlike alpha compensation, which can't fill the physical gaps between removed shots.

#### Disc mode (alternative)

Two `Markers` visuals composited on the same scene node:

| Property | Base layer | Overlap layer |
|----------|-----------|---------------|
| Blend mode | `(src_alpha, one_minus_src_alpha)` | `(src_alpha, one)` |
| Shader | Stock vispy disc | Stock vispy disc |
| Face colour | User-chosen shot colour (α = 1.0) | White `(1, 1, 1, α)` — zoom-dependent |
| Size | `max(true_size, dpp) × 0.667` — inflated to ≥ 1 px, scaled by `_DISC_SIZE_SCALE` | `true_size × 0.667` — no inflation |
| Purpose | Stable visual at all zooms | Additive brightening only where shots genuinely overlap in data space |

The overlap alpha uses a $1/(1+x)$ curve:

$$\alpha = \texttt{\_DISC\_OVERLAP\_WHITE} \times \frac{\texttt{\_ALPHA\_REF\_DPP}}{\texttt{dpp} + \texttt{\_ALPHA\_REF\_DPP}}$$

With `_DISC_OVERLAP_WHITE = 0.08` and `_ALPHA_REF_DPP = 20.0 nm/px`.

### 4.2 Gaussian PSF shader (`gaussian_markers.py`)

`GaussianMarkers` subclasses vispy's `MarkersVisual` with two shader modifications:

**Fragment shader**: replaces the stock disc SDF with a Gaussian profile. The SDF call is kept (for vispy template wiring) but its result is unused — alpha comes purely from `exp(-2.7726 · nr²)` where `nr = dist / half_size`. Fragments with `gauss < 0.003` are discarded to save fill rate.

**Vertex shader**: the `total_size` formula is patched from `$v_size + 4*(edgewidth + 1.5*antialias)` to `$v_size * 3.0 + 4.0`, giving the bounding quad enough room for Gaussian tails (the Gaussian at `r = 1.43` hits the 0.003 discard threshold).

### 4.3 Shot sizing

Shot size in data units (nm) is proportional to dwell time, scaled by a user-adjustable FWHM factor:

$$d = \max(\text{dwell\_ns} \times \texttt{\_NM\_PER\_NS\_DWELL} \times \texttt{fwhm\_scale},\ 1.0)$$

| Constant | Default | Description |
|----------|---------|-------------|
| `_NM_PER_NS_DWELL` | 0.01 | Base conversion: 10 nm per µs |
| `fwhm_scale` | 6.0 | Multiplier — default gives 60 nm/µs FWHM |
| `_DISC_SIZE_SCALE` | 0.667 | Additional scale factor applied in disc mode only |

The FWHM slider in the View menu adjusts `fwhm_scale` logarithmically from 0.01× to 100× (0.1–1000 nm/µs effective).

When all dwells are identical (common case), a single scalar size is used instead of a per-point array, saving `N × 4` bytes of GPU upload.

### 4.4 Minimum marker size

Each marker is inflated to a minimum size so sub-pixel shots remain visible:

- **Gaussian mode**: minimum size = `dpp × 4` (Gaussians need ~4 px to show their smooth falloff)
- **Disc mode base layer**: minimum size = `dpp` (1 screen pixel)
- **Disc mode overlap layer**: no inflation — uses true data size only

### 4.5 Centroid shifting

Raw shot coordinates (X up to 65535, Y up to ~60 million nm) would lose precision in float32. All positions are centroid-shifted before upload:

```python
origin = raw_pos.mean(axis=0)  # float64
positions = (raw_pos - origin).astype(float32)  # sub-nm precision
```

The origin is stored separately for tooltip coordinate reconstruction.

### 4.6 Axis lines

Origin crosshair lines (X and Y axes through the file origin) use a **grow-only dynamic length**:

1. At load time, half-length is set to `max(10 × data_diagonal, 1 mm)`.
2. On each camera change, if the viewport diagonal × 2 exceeds the current half-length, the lines are re-uploaded with the larger value.
3. Lines never shrink — this avoids set_data thrash on repeated zoom in/out cycles.

Large fixed lengths (e.g. 1e15) cause GPU float32 clip-pipeline precision loss (axis jitter). The grow-only approach keeps values within safe float32 range while guaranteeing axes extend past screen edges at any zoom level.

---

## 5. Decimation

Decimation runs on every camera change (100 ms debounce timer). The same code path runs at every zoom level. When the visible shot count is small enough, stride = 1 and every visible shot is drawn.

### Tuning parameters

```python
_MAX_SHOTS_PER_PX = 3.0      # target screen-space density (shots per pixel)
_MAX_RENDERED     = 2_097_152  # hard cap on rendered shots (2²¹)
```

### Algorithm

1. **Viewport cull**: AABB mask keeps only shots inside the visible area + 5 % margin + half the largest disc diameter. Viewport bounds are rotation-aware (4-corner screen → data mapping).
2. **Density-based budget**: the occupied screen area is estimated from `min(data_extent, viewport_extent)` per axis, converted to pixels via `dpp`. Budget = `screen_px × _MAX_SHOTS_PER_PX`, capped at `_MAX_RENDERED`. No floor — when the data covers only a few pixels, very few shots are rendered.
3. **Stride**: `stride = max(1, n_visible / budget)`. When stride ≤ 1, all visible shots are drawn.
4. **Priority-based selection**: each shot is assigned a fixed random priority at load time, with a mild dwell bias (15 %): higher-dwell shots get lower priority values (more likely to survive decimation). The top `budget` shots by priority are selected via `np.argpartition` (O(n)).
5. **Stride size compensation** (Gaussian mode): with additive blending, total brightness ∝ density × alpha × FWHM². Decimation reduces density by 1/stride, so marker FWHM is inflated by `sqrt(stride)`, making each surviving shot cover stride× the area. This fills gaps smoothly — unlike alpha-only compensation, which creates bright speckle where shots are locally sparse.
6. **Cache gating**: a composite key of quantised viewport bounds + stride + `round(log₂(dpp) × 20)` prevents redundant GPU uploads when the view hasn't meaningfully changed.
7. **All visuals synced**: the same decimated index set is uploaded to all active marker layers.

### Design properties

- **Stable across pan/zoom**: fixed per-shot priorities mean the same shots survive decimation regardless of small viewport shifts — no visible flicker.
- **Smooth stride transitions**: stride compensation keeps brightness continuous even as decimation kicks in.
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

### Selection system

| Type | Visual | Behaviour |
|------|--------|-----------|
| **Hover** | Floating tooltip | KD-tree nearest-neighbour lookup; shows shot #, X, Y, dwell. Mode-aware hit radius: disc uses `size × _DISC_SIZE_SCALE / 2`, Gaussian uses `size / 2`. Hidden automatically on click selection. |
| **Click** | Gold overlay marker + tooltip | Select/deselect single shot; highlight connection lines. Same mode-aware hit radius as hover. |
| **Box** | Green overlay markers + side pane | Rubber-band selection mapped through rotation transform; cross-product point-in-quad test |

The KD-tree is built asynchronously on a background `QThread` to avoid blocking the UI during load.

### Rotation

Rotation is applied as a `MatrixTransform` on a parent `Node` that holds all visuals. The camera's own transforms remain unaffected. Axis arrowheads (`_AxisArrowOverlay`) are painted on a transparent Qt overlay widget and track orientation via ray-rectangle intersection.

---

## 7. UI

### Menu structure

```
File
├── Open Pass File… (Ctrl+O)
└── Exit (Alt+F4)

View
├── Show Shot Connections (Ctrl+L)
├── Reset View (Ctrl+R)
├── Show Selection Pane (Ctrl+S)
├── ─────────────────────
├── FWHM: [====slider====] 60.00 nm/µs
├── ─────────────────────
├── Disc Markers (hard edge + overlay)     [toggle]
├── ─────────────────────
└── Colors
    ├── Shot Color       (8 presets + Custom…)
    ├── Click Highlight  (6 presets + Custom…)
    ├── Box Highlight    (6 presets + Custom…)
    ├── Connection Lines (6 presets + Custom…)
    └── Line Highlight   (6 presets + Custom…)

Help
├── About
└── Controls
```

The **FWHM slider** is logarithmic: slider value ∈ [−200, 200] maps to scale = $10^{v/100}$, giving effective range 0.1–1000 nm/µs. Default is 60 nm/µs (slider value 78).

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

---

## 8. Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.14 |
| GUI framework | PyQt6 | ≥ 6.5 |
| GPU rendering | vispy (gl+ backend with PyOpenGL) | ≥ 0.14 |
| Spatial indexing | scipy `cKDTree` | ≥ 1.10 |
| Numerics | numpy | ≥ 2.0 |
| Graphics API | OpenGL 3+ (instanced rendering) | — |

### Why vispy?

- Instanced marker rendering handles millions of points without per-point draw calls
- Scene graph with camera transforms eliminates manual matrix math
- Shader subclassing (`GaussianMarkersVisual`) enables custom fragment shaders without forking
- PyQt6 integration via `vispy.app.use_app("pyqt6")`

---

## 9. Performance Characteristics

| Operation | Technique | Impact |
|-----------|-----------|--------|
| File parsing | mmap + numpy 64-bit vectorised bitfields | ~2 s for 10M shots |
| KD-tree build | Background QThread | Non-blocking UI |
| GPU upload | Uniform colour scalar (not per-point array); scalar size when dwells identical | Saves N×16+ bytes |
| Zoom-out | Priority-based decimation + 2M hard cap | Keeps rendered count ≤ 2M |
| Viewport cull | AABB mask before decimation | Only process visible shots |
| Connection lines | Decimated line segments (max 500K) | Avoids GPU overload |
| Priority select | `np.argpartition` (O(n)) | No full sort needed |

---

## 10. File listing

```
pass_viewer/
├── main.py                 # Entry point
├── main_window.py          # QMainWindow: menus, FWHM slider, mode toggle, status, file loading
├── pass_parser.py          # Binary .pass parser (v3/v4)
├── viewer_widget.py        # GPU viewer: rendering, decimation, interaction
├── gaussian_markers.py     # GaussianMarkers visual (custom Gaussian PSF shader)
├── selection_pane.py       # Box-selection data table
├── generate_test_pass.py   # Test data generator (random shots)
├── generate_chip_pass.py   # Test data generator (IC layout + fractals)
├── generate_icon.py        # App icon generator
├── app_icon.ico            # Application icon
├── requirements.txt        # Python dependencies
├── DESIGN.md               # This document
└── PassFileViewer.spec     # PyInstaller spec for distribution
```

---

## 11. Known Limitations

1. **Single-file viewer** — no multi-file or stripe-sequence viewing
2. **2D only** — no 3D perspective or Z-axis support
3. **No shape record rendering** — only shot records (mID == 0) are displayed

---

## 12. Glossary

| Term | Definition |
|------|-----------|
| **Shot** | A single electron beam exposure point with X, Y, and dwell time |
| **Dwell** | Duration (ns) the beam stays at a position; determines dose and rendered size |
| **FWHM** | Full Width at Half Maximum — the diameter at which the Gaussian PSF reaches 50 % of peak intensity |
| **Stripe** | A horizontal band of shots that the stage traverses in one pass |
| **BSS** | Beam Step Size — the grid pitch of shot placement |
| **Pass file** | Binary file containing a header + packed shot/shape records |
| **Decimation** | Reducing the number of rendered shots via priority-based selection from the visible set |
| **Additive blending** | OpenGL blend mode where source colour is added to framebuffer (`src_alpha, one`) |
| **Alpha blending** | Standard transparency blend that converges to source colour (`src_alpha, one_minus_src_alpha`) |
| **dpp** | Data units per pixel — the true zoom scale, computed from the full vispy transform chain via Euclidean distance (rotation-invariant) |
| **Priority** | Fixed per-shot random rank (biased by dwell and spatial density) that determines decimation survival order |
