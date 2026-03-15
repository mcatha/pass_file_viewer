"""Gaussian point-spread-function markers for vispy.

Drop-in replacement for ``vispy.scene.visuals.Markers`` that renders each
marker with a Gaussian intensity falloff instead of a hard-edged disc.

The marker ``size`` parameter is interpreted as the **FWHM** (Full Width at
Half Maximum) in data units (nm when ``scaling='scene'``).

    alpha(r) = exp(-4·ln2·r²)

where *r* is the normalised radial distance: ``r = 2·d / FWHM``, so the
Gaussian crosses 50% intensity at exactly ``size/2`` from centre.

Profile:
    centre:  1.00
    r = 0.5: 0.50
    edge:    ~0.06
"""

from vispy.visuals.markers import MarkersVisual
from vispy.scene.visuals import create_visual_node
import re as _re

# ── Gaussian fragment shader ──────────────────────────────────────────
# Keeps all declarations and template hooks ($v_size, $pointcoord,
# $marker, $lighting_functions) so vispy's shader wiring is satisfied.
# The SDF call ($marker) is kept but its return value is unused — alpha
# comes purely from the Gaussian profile.

_GAUSSIAN_FRAG = """
#version 120

uniform vec3 u_light_position;
uniform vec3 u_light_color;
uniform float u_light_ambient;
uniform float u_alpha;
uniform float u_antialias;
uniform bool u_spherical;

varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_edgewidth;
varying float v_total_size;
varying float v_depth_middle;
varying float v_alias_ratio;
varying float v_symbol;

bool isnan(float val) {
  return ( val < 0.0 || 0.0 < val || val == 0.0 ) ? false : true;
}

bool isinf(float val) {
    return (val != 0.0 && val * 2.0 == val) ? true : false;
}

// provides apply_lighting and write_depth functions
$lighting_functions

void main()
{
    // Discard zero-size or degenerate markers (same guard as stock shader)
    if ($v_size <= 0. || isnan($v_size) || isinf($v_size))
        discard;

    // Total rendered quad extent — must match vertex shader's total_size
    float size = $v_size * 3.0 + 4.0;

    vec2 pointcoord = $pointcoord;

    // Keep the SDF call so vispy's template-hook system stays wired up.
    float r = $marker(pointcoord, v_total_size, int(v_symbol));

    // ── Gaussian point-spread function (FWHM-based) ──────────
    // Normalised radial distance: 0 at centre, 1 at nominal disc edge.
    // size = FWHM, so half-max is at r = 0.5 (dist = size/2).
    float dist = length((pointcoord - vec2(0.5, 0.5)) * size);
    float half_size = max($v_size * 0.5, 0.001);
    float nr = dist / half_size;

    // k = 4·ln(2) ≈ 2.7726 → exact FWHM: alpha(0.5) = 0.5
    float gauss = exp(-2.7726 * nr * nr);

    // Discard negligible fragments (saves fill rate)
    if (gauss < 0.003)
        discard;

    gl_FragColor = vec4(v_bg_color.rgb, v_bg_color.a * gauss * u_alpha);
}
"""


class GaussianMarkersVisual(MarkersVisual):
    """Markers rendered with a Gaussian intensity profile.

    Subclasses ``MarkersVisual`` — only the fragment shader is replaced
    and the vertex shader's quad sizing is enlarged so the Gaussian tails
    aren't clipped by the bounding quad.

    The Gaussian decays to the discard threshold (0.003) at r ≈ 1.43,
    so the quad must be ≈ 2.86× the marker size.  We replace the stock
    ``total_size`` formula with one that pads by 1.5× the marker size
    on each side (3× total), giving comfortable room.
    """

    # Override both shaders.
    _shaders = dict(MarkersVisual._shaders)
    _shaders['fragment'] = _GAUSSIAN_FRAG

    # Patch vertex shader: replace the total_size line to give the quad
    # enough room for the Gaussian tails.
    #   Stock:  total_size = $v_size + 4.*(v_edgewidth + 1.5*u_antialias)
    #   Ours:   total_size = $v_size * 3.0 + 4.0   (3× marker + small constant)
    _shaders['vertex'] = _re.sub(
        r'float total_size = \$v_size \+ 4\.\s*\*\s*\(v_edgewidth \+ 1\.5 \* u_antialias\);',
        'float total_size = $v_size * 3.0 + 4.0;',
        MarkersVisual._shaders['vertex'],
    )


# Scene-graph node (use this in place of ``scene.visuals.Markers``)
GaussianMarkers = create_visual_node(GaussianMarkersVisual)
