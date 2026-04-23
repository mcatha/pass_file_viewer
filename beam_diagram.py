"""Generate beam_areas.png: MB300 column areas of responsibility.

All 18 beam columns are drawn as equal 65 mm × 75 mm cells on a shared grid.
The logo boundary (250 mm × 60.4 mm) is overlaid in yellow.
Beam positions are marked with a cross.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

_MM = 1_000_000   # nm per mm (matching viewer_widget convention)

# Full (unclipped) cell boundaries in mm
# X column boundaries: -162.5, -97.5, -32.5, +32.5, +97.5, +162.5
# Y row boundaries:    -150, -75, 0, +75, +150  (B/C/D)
#                      -112.5, -37.5, +37.5, +112.5 (A/E)

# Beam positions in mm (from _MB300_FIDUCIALS)
BEAMS = [
    # name   beam_X  beam_Y   x_cell_start  x_cell_end  y_cell_start  y_cell_end
    # A column (X = −130 mm); full cell extends to −162.5 mm
    ('A2', -130,   75,   -162.5,  -97.5,   37.5,  112.5),
    ('A3', -130,    0,   -162.5,  -97.5,  -37.5,   37.5),
    ('A4', -130,  -75,   -162.5,  -97.5, -112.5,  -37.5),
    # B column (X = −65 mm)
    ('B1',  -65,  112.5,  -97.5,  -32.5,   75.0,  150.0),
    ('B2',  -65,   37.5,  -97.5,  -32.5,    0.0,   75.0),
    ('B3',  -65,  -37.5,  -97.5,  -32.5,  -75.0,    0.0),
    ('B4',  -65, -112.5,  -97.5,  -32.5, -150.0,  -75.0),
    # C column (X = 0)
    ('C1',    0,  112.5,  -32.5,   32.5,   75.0,  150.0),
    ('C2',    0,   37.5,  -32.5,   32.5,    0.0,   75.0),
    ('C3',    0,  -37.5,  -32.5,   32.5,  -75.0,    0.0),
    ('C4',    0, -112.5,  -32.5,   32.5, -150.0,  -75.0),
    # D column (X = +65 mm)
    ('D1',   65,  112.5,   32.5,   97.5,   75.0,  150.0),
    ('D2',   65,   37.5,   32.5,   97.5,    0.0,   75.0),
    ('D3',   65,  -37.5,   32.5,   97.5,  -75.0,    0.0),
    ('D4',   65, -112.5,   32.5,   97.5, -150.0,  -75.0),
    # E column (X = +130 mm); full cell extends to +162.5 mm
    ('E2',  130,   75,    97.5,  162.5,   37.5,  112.5),
    ('E3',  130,    0,    97.5,  162.5,  -37.5,   37.5),
    ('E4',  130,  -75,    97.5,  162.5, -112.5,  -37.5),
]

LOGO_W_MM   = 250.0
LOGO_H_MM   = 250 * 145 / 600   # ≈ 60.42 mm
LOGO_X0     = -LOGO_W_MM / 2    # −125 mm
LOGO_Y0     = -LOGO_H_MM / 2    # ≈ −30.21 mm

fig, ax = plt.subplots(figsize=(14, 10))

# Draw cells (shaded, no fill for cells outside the logo Y range)
for name, bx, by, xs, xe, ys, ye in BEAMS:
    # Check if this cell overlaps the logo Y range
    overlaps_logo = ye > LOGO_Y0 and ys < -LOGO_Y0
    color = '#d0e8ff' if overlaps_logo else '#f0f0f0'
    rect = mpatches.Rectangle(
        (xs, ys), xe - xs, ye - ys,
        linewidth=0.8, edgecolor='#444', facecolor=color, zorder=1,
    )
    ax.add_patch(rect)
    # Label
    cx, cy = (xs + xe) / 2, (ys + ye) / 2
    ax.text(cx, cy, name, ha='center', va='center', fontsize=8, color='#222', zorder=3)
    # Beam cross
    ax.plot(bx, by, '+', color='#c00', markersize=6, markeredgewidth=1.2, zorder=4)

# Logo boundary
logo_rect = mpatches.Rectangle(
    (LOGO_X0, LOGO_Y0), LOGO_W_MM, LOGO_H_MM,
    linewidth=2, edgecolor='gold', facecolor='none', zorder=5, label='Logo boundary',
)
ax.add_patch(logo_rect)

# Grid lines at column and row boundaries
for x in [-162.5, -97.5, -32.5, 32.5, 97.5, 162.5]:
    ax.axvline(x, color='#bbb', linewidth=0.5, zorder=0)
for y in [-150, -112.5, -75, -37.5, 0, 37.5, 75, 112.5, 150]:
    ax.axhline(y, color='#bbb', linewidth=0.5, zorder=0)

# Wafer circle (300 mm diameter)
wafer = plt.Circle((0, 0), 150, fill=False, edgecolor='#888', linewidth=1.5,
                   linestyle='--', zorder=6, label='Wafer edge (300 mm)')
ax.add_patch(wafer)

ax.set_xlim(-175, 175)
ax.set_ylim(-175, 175)
ax.set_aspect('equal')
ax.set_xlabel('Wafer X (mm)')
ax.set_ylabel('Wafer Y (mm)')
ax.set_title('MB300 — beam column areas of responsibility\n'
             'Blue = overlaps logo  |  Grey = outside logo Y range  |  + = beam position')
ax.legend(loc='lower right', fontsize=9)

OUT = Path(__file__).parent.parent / "beam_areas.png"
fig.tight_layout()
fig.savefig(OUT, dpi=100)
print(f"Saved {OUT}")
