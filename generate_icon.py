"""Generate app icon with a double-spiral motif matching the pass file pattern."""

import math
from pathlib import Path
import numpy as np

ICON_OUT = Path(__file__).parent / "app_icon.ico"
SIZES = [256, 128, 64, 48, 32, 16]


def _make_double_spiral_points(n_points_per_arm=120, n_turns=3.0):
    """Return arrays of (x, y) for a double spiral matching the pass file pattern.

    Two arms start from the center, 180° apart, and wind outward.
    Points are evenly spaced along each arm.
    """
    xs, ys = [], []
    for arm_offset in [0, math.pi]:  # two arms, 180° apart
        t = np.linspace(0.05, n_turns * 2 * math.pi, n_points_per_arm)
        # Radius grows linearly with angle
        r = t / t.max()
        x = r * np.cos(t + arm_offset)
        y = r * np.sin(t + arm_offset)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def render_icon(size: int) -> "Image":
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (30, 30, 30, 255))
    draw = ImageDraw.Draw(img)

    margin = size * 0.10
    half = size / 2.0
    scale = half - margin

    xs, ys = _make_double_spiral_points()

    # All dots same steel-blue color, matching the app
    color = (46, 140, 217, 255)
    r = max(1.2, size / 100)  # dot radius scales with icon size

    for x, y in zip(xs, ys):
        px = half + x * scale
        py = half - y * scale  # flip Y so +Y is up
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color)

    return img


def main():
    from PIL import Image

    images = []
    for sz in SIZES:
        print(f"  Rendering {sz}x{sz} ...")
        images.append(render_icon(sz))

    images[0].save(
        str(ICON_OUT),
        format="ICO",
        sizes=[(s, s) for s in SIZES],
        append_images=images[1:],
    )
    print(f"Icon saved to {ICON_OUT}")


if __name__ == "__main__":
    main()
