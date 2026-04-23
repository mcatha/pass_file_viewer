"""Generate beam_areas_tool.png: MB300 column areas of responsibility (general tool diagram)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Beam positions and full (unclipped) cell boundaries, all in mm
BEAMS = [
    # name   beam_X  beam_Y   x_cell_start  x_cell_end  y_cell_start  y_cell_end
    ('A2', -130,   75,   -162.5,  -97.5,   37.5,  112.5),
    ('A3', -130,    0,   -162.5,  -97.5,  -37.5,   37.5),
    ('A4', -130,  -75,   -162.5,  -97.5, -112.5,  -37.5),
    ('B1',  -65,  112.5,  -97.5,  -32.5,   75.0,  150.0),
    ('B2',  -65,   37.5,  -97.5,  -32.5,    0.0,   75.0),
    ('B3',  -65,  -37.5,  -97.5,  -32.5,  -75.0,    0.0),
    ('B4',  -65, -112.5,  -97.5,  -32.5, -150.0,  -75.0),
    ('C1',    0,  112.5,  -32.5,   32.5,   75.0,  150.0),
    ('C2',    0,   37.5,  -32.5,   32.5,    0.0,   75.0),
    ('C3',    0,  -37.5,  -32.5,   32.5,  -75.0,    0.0),
    ('C4',    0, -112.5,  -32.5,   32.5, -150.0,  -75.0),
    ('D1',   65,  112.5,   32.5,   97.5,   75.0,  150.0),
    ('D2',   65,   37.5,   32.5,   97.5,    0.0,   75.0),
    ('D3',   65,  -37.5,   32.5,   97.5,  -75.0,    0.0),
    ('D4',   65, -112.5,   32.5,   97.5, -150.0,  -75.0),
    ('E2',  130,   75,    97.5,  162.5,   37.5,  112.5),
    ('E3',  130,    0,    97.5,  162.5,  -37.5,   37.5),
    ('E4',  130,  -75,    97.5,  162.5, -112.5,  -37.5),
]

fig, ax = plt.subplots(figsize=(14, 10))

for name, bx, by, xs, xe, ys, ye in BEAMS:
    rect = mpatches.Rectangle(
        (xs, ys), xe - xs, ye - ys,
        linewidth=0.8, edgecolor='#444', facecolor='#d0e8ff', zorder=1,
    )
    ax.add_patch(rect)
    # Label: upper-left corner of cell, inset slightly
    ax.text(xs + 3, ye - 4, name,
            ha='left', va='top', fontsize=11, fontweight='bold', color='#222', zorder=3)
    # Beam initial position marker
    ax.plot(bx, by, '+', color='#c00', markersize=8, markeredgewidth=1.5, zorder=4)

# Grid lines at column and row boundaries
for x in [-162.5, -97.5, -32.5, 32.5, 97.5, 162.5]:
    ax.axvline(x, color='#bbb', linewidth=0.5, zorder=0)
for y in [-150, -112.5, -75, -37.5, 0, 37.5, 75, 112.5, 150]:
    ax.axhline(y, color='#bbb', linewidth=0.5, zorder=0)

# Wafer circle (300 mm diameter)
wafer = plt.Circle((0, 0), 150, fill=False, edgecolor='#888', linewidth=1.5,
                   linestyle='--', zorder=6, label='Wafer edge (300 mm)')
ax.add_patch(wafer)

# Legend proxy for beam marker
beam_marker = plt.Line2D([0], [0], marker='+', color='#c00', linestyle='none',
                         markersize=9, markeredgewidth=1.5, label='Beam initial position')
ax.legend(handles=[wafer, beam_marker], loc='lower right', fontsize=10)

ax.set_xlim(-175, 175)
ax.set_ylim(-175, 175)
ax.set_aspect('equal')
ax.set_xlabel('Wafer X (mm)', fontsize=11)
ax.set_ylabel('Wafer Y (mm)', fontsize=11)
ax.set_title('MB300 — beam column areas of responsibility', fontsize=13)

OUT = Path(__file__).parent.parent / "beam_areas_tool.png"
fig.tight_layout()
fig.savefig(OUT, dpi=100)
print(f"Saved {OUT}")
