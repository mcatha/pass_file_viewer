"""Generate beam_areas_tool.png: MB300 column areas of responsibility (general tool diagram)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Beam positions and full (unclipped) cell boundaries, all in mm.
# Number suffix → X position; letter → Y position.
BEAMS = [
    # name    beam_X   beam_Y   x_cell_start  x_cell_end  y_cell_start  y_cell_end
    # A row (Y = +130 mm); 3 beams
    ('A2',   75,  130,   37.5,  112.5,   97.5,  162.5),
    ('A3',    0,  130,  -37.5,   37.5,   97.5,  162.5),
    ('A4',  -75,  130, -112.5,  -37.5,   97.5,  162.5),
    # B row (Y = +65 mm); 4 beams
    ('B1',  112.5,  65,   75,  150,   32.5,   97.5),
    ('B2',   37.5,  65,    0,   75,   32.5,   97.5),
    ('B3',  -37.5,  65,  -75,    0,   32.5,   97.5),
    ('B4', -112.5,  65, -150,  -75,   32.5,   97.5),
    # C row (Y = 0 mm); 4 beams
    ('C1',  112.5,   0,   75,  150,  -32.5,   32.5),
    ('C2',   37.5,   0,    0,   75,  -32.5,   32.5),
    ('C3',  -37.5,   0,  -75,    0,  -32.5,   32.5),
    ('C4', -112.5,   0, -150,  -75,  -32.5,   32.5),
    # D row (Y = −65 mm); 4 beams
    ('D1',  112.5, -65,   75,  150,  -97.5,  -32.5),
    ('D2',   37.5, -65,    0,   75,  -97.5,  -32.5),
    ('D3',  -37.5, -65,  -75,    0,  -97.5,  -32.5),
    ('D4', -112.5, -65, -150,  -75,  -97.5,  -32.5),
    # E row (Y = −130 mm); 3 beams
    ('E2',   75, -130,   37.5,  112.5, -162.5,  -97.5),
    ('E3',    0, -130,  -37.5,   37.5, -162.5,  -97.5),
    ('E4',  -75, -130, -112.5,  -37.5, -162.5,  -97.5),
]

fig, ax = plt.subplots(figsize=(14, 10))

for name, bx, by, xs, xe, ys, ye in BEAMS:
    rect = mpatches.Rectangle(
        (xs, ys), xe - xs, ye - ys,
        linewidth=0.8, edgecolor='#444', facecolor='#d0e8ff', zorder=1,
    )
    ax.add_patch(rect)
    # Label: upper-left corner of cell, inset slightly
    ax.text(xs + 3, ye - 3, name,
            ha='left', va='top', fontsize=11, fontweight='bold', color='#222', zorder=3)
    # Beam initial position marker
    ax.plot(bx, by, '+', color='#c00', markersize=8, markeredgewidth=1.5, zorder=4)

# Horizontal grid lines at shared Y row boundaries
for y in [-162.5, -97.5, -32.5, 32.5, 97.5, 162.5]:
    ax.axhline(y, color='#bbb', linewidth=0.5, zorder=0)

# Wafer circle (300 mm diameter)
wafer = plt.Circle((0, 0), 150, fill=False, edgecolor='#888', linewidth=1.5,
                   linestyle='--', zorder=6, label='Wafer edge (300 mm)')
ax.add_patch(wafer)

# 200 mm wafer outline centred on C2 (37.5, 0)
wafer200 = plt.Circle((37.5, 0), 100, fill=False, edgecolor='#e07000', linewidth=1.5,
                      linestyle='--', zorder=6, label='Wafer edge (200 mm), centred on C2')
ax.add_patch(wafer200)

# Legend proxy for beam marker
beam_marker = plt.Line2D([0], [0], marker='+', color='#c00', linestyle='none',
                         markersize=9, markeredgewidth=1.5, label='Beam initial position')
ax.legend(handles=[wafer, wafer200, beam_marker], loc='lower right', fontsize=10)

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
