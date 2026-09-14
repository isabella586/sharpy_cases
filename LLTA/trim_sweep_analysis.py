"""
LLTA trim sweep analysis
Objective: Scans SHARPy StaticTrim output folders, extracts trim results and camber per run, 
computes analytical hand-calc alpha(V) and CL(V) / CL(alpha) curves, and produces four saved plots:

    1. alpha vs. u_inf                  SHARPy & hand calc, one pair of curves per camber,
                                        with an overlay of typical cruise-speed. 
    2. CL vs. u_inf                     SHARPy CL (from AeroForcesCalculator) & the single
                                        camber-independent trim CL_required(V) hand calc,
                                        with overlays of typical cruise-speed / CLmax / CLmin.
    3. CL vs. alpha                     SHARPy CL & the wing-alone hand-calc lift curve
                                        (one per camber, theoretical thin-airfoil slope) vs.
                                        a linear regression fit through that camber's own
                                        SHARPy datapoint vs. the ASWing full-aircraft lift
                                        curve (dCL/dalpha, zero-lift angle), plus CLmax/CLmin.
                           
                                        All three line types stay straight (linear) from the
                                        zero-lift angle up until close to the stall angle,
                                        then bend over to peak at CLmax/CLmin - the classic
                                        lift-curve shape - see _stall_shape() below.

    4. wing-tip deflection vs u_inf -   Out of plane (vertical) bending displacement of the wingtip
                                        node at the trim state, relative to its position in the jig
                                        (undeformed) shape, one curve per camber. 

    5. deflected wing shape vs. span -  absolute vertical position of every
                                        wing-side node from root to tip, for the jig shape
                                        and for every u_inf, faceted by camber (u_inf
                                        encoded as a colorbar rather than a per-line
                                        legend). Same SaveData requirement as plot 4.

Requires only numpy, pandas, matplotlib and h5py (already a SHARPy dependency) - no SHARPy import needed, this
only reads the SHARPy generated output files (trim_iterations.txt, the
AeroForcesCalculator forces CSV).

Usage:
    python llta_sweep_analysis.py

Edit the CONFIG block below to desired setup - including the ASWING_* (i.e. prototype)
constants. 
"""

import os
import re
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import llta_geometry as llta

try:
    import h5py
except ImportError:
    h5py = None #Needed?

# ============================================================================
# CONFIG - edit this block
# ============================================================================

OUTPUT_ROOT = "/Users/icc25/sharpy_cases/llta/output"

# Folder naming convention: llta_trim_uinf_{velocity}[_{camber_tag}]
# Accepts both "uinf" and "u_inf" (some earlier runs used the underscore).
# The camber tag is optional - a bare llta_trim_uinf_<V> folder (no camber
# suffix) is treated as the no-camber / symmetric-airfoil baseline case.
# Anything else not matching this pattern (old runs, other cases) is skipped
# and reported.
FOLDER_PATTERN = re.compile(
    r"^llta_trim_u_?inf_(?P<velocity>\d+(?:\.\d+)?)(?:_(?P<camber_tag>camber\d*))?$"
)

# camber_tag -> (M, P) in the generate_naca_camber(M, P) convention (M = max
# camber, % chord; P = its chordwise position, tenths of chord). 
# Add entries here if a new camber tag shows up - alpha_L0 is computed from these, not
# hardcoded, so a new tag just needs its (M, P) added.
# "nocamber" is the symmetric baseline (folder with no camber suffix at all);
# M=0, P=0 makes alpha_L0_deg() return 0.0 automatically.
CAMBER_MP = {
    "camber": {"M": 6, "P": 4},
    "camber2": {"M": 5, "P": 5},
    "nocamber": {"M": 0, "P": 0},
}

# Aircraft / flight-condition constants (LLTA, per the project setup notes)
MASS = llta.aircraft_mass     # kg
G = 9.81
RHO = 1.225                   # kg/m^3
CHORD_MAIN = llta.chord_main  # m
SPAN_MAIN = llta.span_main    # m
OSWALD_E = 0.85

# SHARPy StaticTrim default convergence tolerances (used only to flag/warn on
# runs that didn't actually converge - doesn't affect the extracted values)
FX_TOL = 0.01
FZ_TOL = 0.01
M_TOL = 0.01

# Fraction of the (zero-lift-angle -> stall-angle) range, at each end, over
# which the CL-vs-alpha curves bend from straight into the rounded peak at
# CLmax/CLmin - see _stall_shape(). 0.2 means the last 20% of the way to
# stall is where the curve leaves the straight line; the other 80% (starting
# right at the zero-lift angle) stays exactly linear, matching the "usual
# angles of flight" straight segment in a typical CL-alpha chart.
STALL_TRANSITION_FRAC = 0.2

# Set False to turn off the near-stall bend entirely and just plot pure
# straight lines (slope * (alpha - alpha0)) all the way through, e.g. to read
# off the zero-lift angle as a clean x-intercept without the curve bending
# away near CLmax/CLmin. Flip back to True to restore the blended shape -
# nothing else in the script needs to change either way.
STALL_BLEND_ENABLED = False

# Set False to drop the per-camber "SHARPy data fit" line from the CL-vs-alpha
# plot (plot 3) - two fewer lines and two fewer legend entries, useful if
# that plot feels crowded and you only want the theoretical/ASWing curves
# against the raw SHARPy points.
SHOW_SHARPY_DATA_FIT = False

# Alpha-axis crop (degrees), applied to *both* plots that show alpha: the
# x-axis on the CL-vs-alpha plot (plot 3) and the y-axis on the alpha-vs-u_inf
# plot (plot 1). The underlying computed curves (theory, ASWing, SHARPy data
# fit, hand-calc alpha) are still computed over their full natural range
# (plot 3's alpha_fine spans data range + an 8 deg buffer -- see Plot 3
# below) -- lowering this doesn't change any curve's shape or the
# stall-shape math. What it does change is how much of each curve/series is
# actually *drawn*: every plotted line/point beyond this value is dropped
# entirely (via _trim_to_cap()), not just hidden behind a cropped axis
# window, so a curve visibly ends at this value with real blank space
# before the plot's border (via _cap_axis_with_gap(), using the margin set
# by ALPHA_PLOT_GAP below) instead of running flush into (and looking cut
# off by) the edge of the plot area. If the data on a given plot doesn't
# actually reach this value in the first place, that plot is left
# completely alone -- no trimming, no extra margin, since there'd be
# nothing behind it to leave a gap for. Set to None to fall back to
# matplotlib's automatic (data-driven) axis limit on both plots.
ALPHA_PLOT_XMAX = 15

# Blank margin (degrees) left between where a capped alpha curve's drawn
# data actually stops and the true right/top edge of its plot area -- see
# ALPHA_PLOT_XMAX above. Purely cosmetic (how much breathing room the cut
# line gets), doesn't affect where the data itself is trimmed.
ALPHA_PLOT_GAP = 1.5

# --------------------------------------
# ASWing / prototype reference data - ALL DUMMY PLACEHOLDER VALUES.
# Replace every one of these with the real figures from your ASWing file
# before trusting the overlays drawn from them. Nothing else in the script
# needs to change when you do - these six constants are the only inputs.
# --------------------------------------
CRUISE_SPEED_ASWING = 9          # m/s - vertical line on plots 1 & 2.
CL_MAX_ASWING = 1.34             # Horizontal line on plots 2 & 3.
CL_MIN_ASWING = -1               # Horizontal line on plots 2 & 3.
DCL_DALPHA_ASWING = 4.97            # /rad - Slope of the ASWing lift
                                    # curve on plot 3 (full-aircraft, not
                                    # wing-alone - expect it to differ from
                                    # CL_ALPHA below, which is wing-alone).
ALPHA_L0_ASWING_DEG = -6.2         # deg - Zero-lift angle, horizontal
                                    # line on plot 1 and the x-intercept of the
                                    # ASWing lift curve on plot 3.
                                    # SIGN CONVENTION: negative for a cambered
                                    # airfoil that lifts at alpha=0, matching
                                    # alpha_L0 for 'camber'/'camber2' below
                                    # (e.g. -6.2317 deg). You quoted this as
                                    # "6.2 degrees" - double-check the sign
                                    # matches this convention when you fill in
                                    # the real value.

# ------------------------------------
# Wing-tip deflection (relative to jig shape)
#  - jig (undeformed) shape:  data/structure/ini_info/pos - the FE node
#     positions from the input mesh, not edited by FSI loops.
#   - trimmed (deformed) shape: data/structure/timestep_info/<last>/pos -
#     StaticCoupled.change_trim() rebuilds timestep_info from scratch on
#     every trim iteration, so the last entry after a finished run is the
#     converged final trim state.
#   Both arrays are in the body-fixed 'A' frame (no rigid-body/trim rotation
#   applied), so subtracting them gives pure elastic deformation with the
#   trim alpha rotation already excluded - not a mix of "the wing bent" and
#   "the whole aircraft pitched up".

# Node index (0-based, in the same numbering as the case's FE mesh) of the
# wingtip to track. Leave as None to auto-detect: the node with the largest
# |y| in the jig shape. 
WING_TIP_NODE_INDEX = None

# Axis indices within the (x, y, z) pos array. y = spanwise (see above);
# z = vertical/out-of-plane bending, SHARPy's standard body-fixed-frame
# convention (x ~ forward/chordwise, y spanwise, z up).
SPAN_AXIS_INDEX = 1
VERTICAL_AXIS_INDEX = 2

# Set True to also draw the total 3D tip-displacement magnitude (not just
# the vertical/bending component) on the deflection plot - off by default.
SHOW_TOTAL_TIP_DEFLECTION = False

# Node indices (0-based, same numbering as WING_TIP_NODE_INDEX) making up
# "the wing" for the spanwise deflection-shape plot (plot 5) - every node
# from root to tip. Leave as None to auto-detect:
# every node on the same side (sign of y) as the wingtip node, in the jig
# shape. This is a coordinate-only heuristic - it has no way to know that a
# tail or fuselage node sharing that same y-range and sign isn't actually
# part of the wing. Auto-detect prints the full selected node list (index +
# jig x, y, z), in root-to-tip order, once on the first successful run --
# check it for a sudden jump in x partway through (a sign a non-wing node
# got pulled in) and set this explicitly, ordered root to tip, if so.
WING_NODE_INDICES = None

# Plot 5's x-axis: physical distance from root, |y| in meters (default,
# False), or the spanwise position normalized by the semispan -- |y| /
# (SPAN_MAIN/2), 0 at the root, 1 at the tip (True). Normalized span is
# useful for comparing the *shape* of the bend independent of the aircraft's
# absolute size (e.g. against another aircraft's normalized-span deflection
# plot), or just if that's the preferred convention to read.
# Note: when this is True, the x-axis becomes dimensionless while the y-axis
# (vertical position) stays in meters, so a literal 1:1 aspect ratio no
# longer means anything physical -- the plot drops back to matplotlib's
# automatic aspect in that case, and keeps the 1:1 physical aspect only for
# the default (unnormalized, both-axes-in-meters) mode.
NORMALIZE_WING_SHAPE_SPAN = True

# Plot 5's x-axis maximum in the default (unnormalized, meters) mode. The
# exact semispan (SPAN_MAIN/2 = 1.875 m here) puts a tick right on the
# wingtip, but since it's not a "nice" locator step the ticks before it
# stay evenly spaced at 0.25 m while the final one lands a short 0.125 m
# later -- a visibly uneven last step. Overriding to a round number like
# 2.0 m instead keeps every tick an even 0.25 m apart, at the cost of the
# tip no longer sitting exactly on a labelled tick. Set to None to go back
# to the exact semispan. Ignored when NORMALIZE_WING_SHAPE_SPAN is True --
# normalized mode's 0-1 range already lands on nice ticks either way.
WING_SHAPE_SPAN_AXIS_MAX = 2.0  # m

# Plot 5's y-axis (z, absolute vertical position) is forced to reach at
# least this value as its upper limit, with a tick mark placed exactly
# there -- regardless of how far any single camber/velocity's deformed
# shape actually goes -- so the vertical scale reflects the full expected
# range of the wingtip's travel and a reader can pick that value straight
# off the axis. Set to None to just autoscale as before.
WING_SHAPE_Z_AXIS_MAX = 0.30  # m

OUT_CSV = "llta_sweep_results.csv"
OUT_PLOT_ALPHA_V = "llta_sweep_alpha_vs_v.png"
OUT_PLOT_CL_V = "llta_sweep_cl_vs_v.png"
OUT_PLOT_CL_ALPHA = "llta_sweep_cl_vs_alpha.png"
OUT_PLOT_TIP_DEFL = "llta_sweep_tip_deflection_vs_v.png"
OUT_PLOT_WING_SHAPE = "llta_sweep_wing_shape_vs_span.png"

# ============================================================================
# Derived constants
# ============================================================================

W = MASS * G
S = CHORD_MAIN * SPAN_MAIN
AR = SPAN_MAIN**2 / S
A0 = 2 * np.pi  # /rad, 2D thin-airfoil slope (camber-independent)
CL_ALPHA = A0 / (1 + A0 / (np.pi * OSWALD_E * AR))  # /rad, wing-alone, AR + Oswald corrected

# Shared plot styling
PALETTE = ["#2b6cb0", "#c0392b", "#2e7d32", "#8e44ad"]
INK, MUTED, GRID = "#1a2733", "#5b6b7a", "#d8dee4"
ASWING_COLOR = "#444444"
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("llta_seq_blue", [
    "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 1.0,
})


def _style_axes(ax, title, legend_outside=False):
    ax.set_title(title, fontsize=12, loc="left", color=INK)
    ax.grid(True, color=GRID, lw=0.7, alpha=0.7)
    if legend_outside:
        # For a legend with many entries: place it outside the axes (to the
        # right) instead of letting matplotlib's "best" placement squeeze it
        # in over the data, which is what was cramming/overlapping rows on
        # the busier plots. bbox_inches='tight' on savefig() picks up the
        # extra width automatically, so nothing gets clipped.
        ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED,
                   loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    else:
        ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _ensure_boundary_tick(ax, axis, boundary):
    """Force a tick mark to appear at exactly `boundary` on the given axis
    ('x' or 'y'), on top of whatever ticks matplotlib's default locator
    already picked. The default locator steps in "nice" round numbers
    (0.25, 0.5, 1, 2, 2.5, 5, ...) which generally don't land exactly on a
    physical quantity like a 1.875 m semispan, so without this the last
    stretch of the axis has no tick to read the boundary value off of.

    Note: ax.get_xticks()/get_yticks() can return candidate tick positions
    that fall *outside* the current view limits (the locator plans slightly
    past the data range) -- feeding those straight back into set_xticks()/
    set_yticks() would silently widen the axis to include them. So the
    existing ticks are first filtered down to the current view range before
    the boundary tick is added back in, keeping the visible range exactly
    what set_xlim()/set_ylim() already established."""
    if axis == "x":
        get_ticks, set_ticks, lim = ax.get_xticks, ax.set_xticks, ax.get_xlim()
    else:
        get_ticks, set_ticks, lim = ax.get_yticks, ax.set_yticks, ax.get_ylim()
    lo, hi = min(lim), max(lim)
    ticks = [t for t in get_ticks() if lo - 1e-9 <= t <= hi + 1e-9]
    if not any(np.isclose(t, boundary, atol=1e-6) for t in ticks):
        ticks.append(boundary)
    set_ticks(sorted(ticks))


def _trim_to_cap(x, y, cap, cap_on="x"):
    """Return (x, y) with every point where the array named by `cap_on`
    ('x' or 'y') exceeds `cap` dropped -- so a plotted curve/line's drawn
    extent genuinely stops there, instead of being drawn further and only
    visually clipped by the axis window. This keeps the axis crop and the
    actually-plotted data in agreement with each other: the line itself
    ends at the cap, rather than the crop being a display-only trick
    layered on top of a longer line underneath it that just happens to run
    into the edge of the plot area."""
    x = np.asarray(x)
    y = np.asarray(y)
    mask = (x if cap_on == "x" else y) <= cap
    return x[mask], y[mask]


def _cap_axis_with_gap(ax, axis, cap, gap):
    """Set `axis`'s upper display limit to `cap + gap`. Leaves a blank
    margin of size `gap` beyond wherever a same-`cap`-trimmed curve (see
    _trim_to_cap) actually stops drawing, so the curve visibly ends with
    some empty space in front of the plot's true border, rather than
    running flush into it and looking cut off by the frame. No-op if `cap`
    is None -- the caller is expected to only pass a real cap when the data
    actually reaches past it in the first place, since a gap with nothing
    behind it would just be a smaller plot for no reason."""
    if cap is None:
        return
    if axis == "x":
        ax.set_xlim(right=cap + gap)
    else:
        ax.set_ylim(top=cap + gap)


# ============================================================================
# Hand-calc theory
# ============================================================================

def alpha_L0_deg(M_pct, P_tenths, n=20000):
    """Thin-airfoil zero-lift angle (deg) for a standard NACA 4-digit mean
    camber line, via the closed-form integral. Validated earlier against
    tabulated NACA values (2412 -> -2.08 deg, 4412 -> -4.15 deg, 2212 -> -1.80 deg)."""
    m = M_pct * 1e-2
    p = P_tenths * 1e-1
    if p == 0:
        return 0.0
    theta = np.linspace(1e-9, np.pi - 1e-9, n)
    x = (1 - np.cos(theta)) / 2
    dzdx = np.where(
        x < p,
        (2 * m / p**2) * (p - x),
        (2 * m / (1 - p)**2) * (p - x),
    )
    integrand = dzdx * (np.cos(theta) - 1)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    result_rad = -(1 / np.pi) * trapz(integrand, theta)
    return np.degrees(result_rad)


def alpha_hand_deg(V, a_l0_deg):
    """Hand-calc trim alpha: alpha_L0 + CL_required(V)/CL_alpha, both terms in
    radians internally, converted to degrees once at the end."""
    CL_required = CL_hand(V)
    return a_l0_deg + np.degrees(CL_required / CL_ALPHA)


def CL_hand(V):
    """Trim-equilibrium CL required at this speed: vertical equilibrium
    (L = W) gives CL = 2W / (rho V^2 S). This does NOT depend on camber - it's
    purely a statement of what CL the aircraft needs at this speed, regardless
    of which airfoil is producing it. That's why the CL-vs-V hand calc is a
    single curve, not one per camber (unlike alpha_hand_deg, which does depend
    on camber through alpha_L0). It's also exactly the CL_required already
    used inside alpha_hand_deg() above - pushing alpha_hand_deg() back through
    CL_hand_from_alpha() below returns this same value by construction."""
    return 2 * W / (RHO * V**2 * S)


def _hermite_blend(t, l0, l0_prime, l1, width):
    """Cubic Hermite interpolation for t in [0, 1] from (value=l0, slope=l0_prime)
    at t=0 to (value=l1, slope=0) at t=1, where `width` is the alpha-span the
    t=[0,1] interval covers (needed to scale the slope term correctly). Used to
    round the lift curve over into its peak at CLmax/CLmin with a smooth (C1)
    join - no kink where the straight line hands off to the curved part."""
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    return h00 * l0 + h10 * width * l0_prime + h01 * l1


def _stall_shape(alpha_deg, alpha0_deg, slope_per_deg, cl_max, cl_min,
                  transition_frac=STALL_TRANSITION_FRAC):
    """The classic textbook CL-vs-alpha shape: a straight line from the
    zero-lift angle (alpha0_deg, where CL=0) with slope `slope_per_deg`,
    staying linear all the way out to close to the stall angle, THEN bending
    over the last `transition_frac` of that range to peak exactly at cl_max
    (with zero slope there, like a real lift curve's rounded top) - mirrored
    on the negative side towards cl_min. Beyond the stall angle the curve is
    held flat at cl_max/cl_min rather than declining: there's no post-stall
    data or theory available here (slope, alpha0, CLmax, CLmin don't tell you
    the post-stall drop-off shape), so a flat cap is the honest choice over
    guessing a decline - this only claims to reproduce the pre-stall/at-stall
    shape in the reference image, not full post-stall behaviour.

    The stall angle itself isn't a separate input - it's derived
    self-consistently as where the straight line would reach cl_max (or
    cl_min): alpha_stall = alpha0 + cl_max/slope. That's the only definition
    of "critical angle" consistent with the four numbers already available.

    If STALL_BLEND_ENABLED is False, skips all of the above and returns the
    plain straight line - useful for reading the zero-lift angle off as a
    clean x-intercept without the near-stall bend in the way.
    """
    alpha_deg = np.asarray(alpha_deg, dtype=float)
    cl_linear = slope_per_deg * (alpha_deg - alpha0_deg)

    if not STALL_BLEND_ENABLED:
        return cl_linear

    alpha_stall_pos = alpha0_deg + cl_max / slope_per_deg
    alpha_stall_neg = alpha0_deg + cl_min / slope_per_deg  # cl_min<0 -> below alpha0
    knee_pos = alpha_stall_pos - transition_frac * (alpha_stall_pos - alpha0_deg)
    knee_neg = alpha_stall_neg - transition_frac * (alpha_stall_neg - alpha0_deg)

    # positive-side (towards cl_max) blend
    t_pos = (alpha_deg - knee_pos) / (alpha_stall_pos - knee_pos)
    t_pos_clipped = np.clip(t_pos, 0, 1)
    l0_pos = slope_per_deg * (knee_pos - alpha0_deg)
    blend_pos = _hermite_blend(t_pos_clipped, l0_pos, slope_per_deg, cl_max,
                                alpha_stall_pos - knee_pos)

    # negative-side (towards cl_min) blend - mirror image
    t_neg = (alpha_deg - knee_neg) / (alpha_stall_neg - knee_neg)
    t_neg_clipped = np.clip(t_neg, 0, 1)
    l0_neg = slope_per_deg * (knee_neg - alpha0_deg)
    blend_neg = _hermite_blend(t_neg_clipped, l0_neg, slope_per_deg, cl_min,
                                alpha_stall_neg - knee_neg)

    return np.select(
        [alpha_deg >= alpha_stall_pos, alpha_deg >= knee_pos,
         alpha_deg <= alpha_stall_neg, alpha_deg <= knee_neg],
        [cl_max, blend_pos, cl_min, blend_neg],
        default=cl_linear,
    )


def _linear_fit_deg(alpha_deg_vals, cl_vals):
    """Least-squares straight line through real (alpha_deg, CL) data points -
    returns (slope_per_deg, alpha0_deg) in the same (slope, zero-lift-angle)
    form _stall_shape() expects, or None if there aren't at least 2 valid
    points to fit. This is what makes the SHARPy-data-fit curve on plot 3
    genuinely data-driven, as opposed to the theoretical wing-alone and
    ASWing curves, which use externally-supplied slope/alpha0 values."""
    alpha_deg_vals = np.asarray(alpha_deg_vals, dtype=float)
    cl_vals = np.asarray(cl_vals, dtype=float)
    mask = ~(np.isnan(alpha_deg_vals) | np.isnan(cl_vals))
    if mask.sum() < 2:
        return None
    slope_per_deg, intercept = np.polyfit(alpha_deg_vals[mask], cl_vals[mask], 1)
    if slope_per_deg == 0:
        return None
    alpha0_deg = -intercept / slope_per_deg
    return slope_per_deg, alpha0_deg


def CL_hand_from_alpha(alpha_deg, a_l0_deg):
    """Wing-alone lift curve: linear thin-airfoil theory near the zero-lift
    angle (slope CL_alpha), shaped via _stall_shape() to stay straight until
    close to the (self-consistently derived) stall angle, then round over to
    CLmax/CLmin. NOTE: CLmax/CLmin are only available from the ASWing
    (full-aircraft) data - there's no separate wing-alone stall figure - so
    the wing-alone curve borrows the same bounds. Treat the near-stall bend on
    this curve as illustrative shape, not a rigorous wing-alone stall
    prediction; the straight-line slope and zero-lift angle (which cover the
    whole range any real trimmed SHARPy point falls in) are still the
    validated wing-alone quantities, unchanged from before."""
    slope_per_deg = CL_ALPHA * np.pi / 180
    return _stall_shape(alpha_deg, a_l0_deg, slope_per_deg, CL_MAX_ASWING, CL_MIN_ASWING)


def CL_aswing_from_alpha(alpha_deg):
    """ASWing full-aircraft lift curve: linear near the ASWing zero-lift angle
    (slope dCL/dalpha), shaped via _stall_shape() the same way. Expected to
    sit above the wing-alone hand-calc line in the linear region (tail,
    fuselage and other full-aircraft contributions add to the wing-alone
    slope), which is the point of plotting both together."""
    slope_per_deg = DCL_DALPHA_ASWING * np.pi / 180
    return _stall_shape(alpha_deg, ALPHA_L0_ASWING_DEG, slope_per_deg, CL_MAX_ASWING, CL_MIN_ASWING)


# ============================================================================
# Parsing
# ============================================================================

def parse_trim_iterations(path):
    """Parse SHARPy's trim_iterations.txt (a '|'-delimited table, not a CSV -
    written by cout_utils.TablePrinter). Returns the converged (last) row as a
    dict, plus the raw column values needed for a convergence check."""
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]

    rows = []
    for line in lines:
        s = line.strip()
        if not s.startswith("|"):
            continue
        if set(s.replace("|", "")) <= {"="}:
            continue
        fields = [c.strip() for c in s.strip("|").split("|")]
        try:
            float(fields[0])
        except ValueError:
            continue  # header row
        rows.append(fields)

    if not rows:
        raise ValueError(f"No data rows parsed from {path}")

    last = rows[-1]
    # columns: iter, alpha[deg], elev[deg], thrust, Fx, Fy, Fz, Mx, My, Mz
    return {
        "alpha_deg": float(last[1]),
        "elev_deg": float(last[2]),
        "thrust": float(last[3]),
        "Fx": float(last[4]),
        "Fz": float(last[6]),
        "My": float(last[8]),
        "n_iter": len(rows),
    }


def parse_forces_csv(path, u_inf):
    """Compute CL from the AeroForcesCalculator raw-force output:
    CL = (fz_steady_G + fz_unsteady_G) / (q_ref * S), matching
    AeroForcesCalculator.calculate_coefficients() internally."""
    data = np.genfromtxt(path, delimiter=",", comments="#")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    last = data[-1]
    fz_steady_g = last[3]
    fz_unsteady_g = last[6]
    q_ref = 0.5 * RHO * u_inf**2
    return (fz_steady_g + fz_unsteady_g) / (q_ref * S)


def find_forces_csv(run_dir, folder_name):
    """The expected name is forces_<folder_name>_aeroforces.csv (case_name ==
    folder_name is SHARPy's usual convention). Falls back to a glob if that
    exact file isn't there, in case naming drifts."""
    expected = os.path.join(run_dir, "forces", f"forces_{folder_name}_aeroforces.csv")
    if os.path.exists(expected):
        return expected
    candidates = glob.glob(os.path.join(run_dir, "forces", "*aeroforces*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    return None

def find_savedata_h5(run_dir, folder_name):
    """SaveData writes <output_folder>/savedata/<case_name>.data.h5
    (case_name == folder_name, same convention as the forces CSV). Requires
    'SaveData'. Returns None (not an exception) if missing, same pattern
    as find_forces_csv(), so a run missing this file just skips the
    deflection columns rather than aborting the whole scan."""
    expected = os.path.join(run_dir, "savedata", f"{folder_name}.data.h5")
    if os.path.exists(expected):
        return expected
    candidates = glob.glob(os.path.join(run_dir, "savedata", "*.data.h5"))
    if len(candidates) == 1:
        return candidates[0]
    return None
 
 
_tip_node_reported = False  # print the auto-detected node once, not per-run
_wing_nodes_reported = False  # print the auto-detected wing node list once

def _select_wing_nodes(jig_pos, tip_node): # Needed???
    """Coordinate-based proxy for 'this run's wing nodes' (root to tip),
    used only for the spanwise deflection-shape plot: every node on the
    same side (sign of y) as the wingtip node, sorted by |y| ascending. This
    is a heuristic over coordinates, not a real read of the FE
    connectivity/beam grouping -- it has no way to know that a tail,
    fuselage, or a strut/other structural member sharing that y-range and
    sign isn't part of the wing (confirmed to actually happen on this
    aircraft's real mesh -- see _select_wing_nodes_by_beam() below, which is
    tried first and is what should normally be used instead). Kept as the
    fallback for a savedata file that doesn't have 'beam_number'/
    'connectivities' saved (older SHARPy version, or SaveData settings that
    dropped them). See the WING_NODE_INDICES comment in CONFIG for how to
    override either method entirely."""
    tip_sign = np.sign(jig_pos[tip_node, SPAN_AXIS_INDEX])
    if tip_sign == 0:
        tip_sign = 1.0
    y = jig_pos[:, SPAN_AXIS_INDEX]
    mask = (y * tip_sign) >= -1e-9  # same side, including the root (y ~ 0)
    candidates = np.nonzero(mask)[0]
    order = np.argsort(np.abs(y[candidates]))
    return candidates[order]

def _select_wing_nodes_by_beam(jig_pos, beam_number, connectivities, tip_node):
    """Authoritative wing-node selection, using SHARPy's own FE model instead
    of guessing from coordinates. SHARPy tags every beam ELEMENT with a
    'beam_number' -- an integer identifying which physical structural member
    it belongs to (main wing, tail, a strut, etc.), even within a single-body
    FE mesh; 'connectivities' (num_elem x num_node_elem) gives each element's
    node indices. So: find which element(s) the wingtip node belongs to, read
    THEIR beam_number, then take every node belonging to any element sharing
    that same beam_number -- that is, by construction, exactly the wingtip's
    own structural member and nothing else, no matter what other member
    happens to overlap its spanwise range.
 
    Still ANDed with the same-side-of-root filter _select_wing_nodes() uses,
    as a harmless safety net (relevant only if a single beam_number really
    does span both left and right, e.g. a continuous unbroken beam through
    the root) -- and returned sorted root to tip.
 
    Raises (KeyError via the caller's h5py lookup, propagated here) if
    'beam_number' or 'connectivities' weren't saved in this file -- the
    caller catches this and falls back to _select_wing_nodes()."""
    connectivities = np.asarray(connectivities)
    elem_mask = np.any(connectivities == tip_node, axis=1)
    if not np.any(elem_mask):
        raise ValueError(f"wingtip node {tip_node} not found in any row of 'connectivities' "
                          f"-- can't determine its beam_number")
    tip_beam_num = int(beam_number[elem_mask][0])
    wing_elems = np.nonzero(beam_number == tip_beam_num)[0]
    wing_nodes = np.unique(connectivities[wing_elems].ravel())
 
    tip_sign = np.sign(jig_pos[tip_node, SPAN_AXIS_INDEX])
    if tip_sign == 0:
        tip_sign = 1.0
    same_side = wing_nodes[(jig_pos[wing_nodes, SPAN_AXIS_INDEX] * tip_sign) >= -1e-9]
    order = np.argsort(np.abs(jig_pos[same_side, SPAN_AXIS_INDEX]))
    return same_side[order], tip_beam_num
 
 
def parse_tip_deflection(h5_path, run_id):
    """Wing-tip deflection relative to the jig (undeformed) shape, plus (best
    effort) the full spanwise (root-to-tip) jig/deformed shape used by the
    wing-shape plot -- see the CONFIG block above for exactly what's being
    read and why it's the right quantity.
 
    Returns a dict with the tip-deflection fields always present (or raises,
    same pattern as parse_forces_csv(), if even THOSE can't be computed --
    e.g. the file is missing timestep_info entirely). The wing-shape fields
    ("span_y_jig"/"span_z_jig"/"span_z_deformed") are added on a best-effort
    basis in a separate try/except: a failure there (bad node selection,
    unexpected mesh shape, etc.) is warned about and simply leaves those keys
    out of the dict -- it must NEVER take down the tip-deflection fields
    above, which is what happened when this was all one unguarded block."""
    global _tip_node_reported, _wing_nodes_reported
    with h5py.File(h5_path, "r") as f:
        struct_grp = f["data"]["structure"]
        jig_pos = struct_grp["ini_info"]["pos"][()]
        # h5utils writes a '_read_as' bookkeeping dataset (records the Python
        # class name for reconstruction) into every group it saves, including
        # this one -- it's not a timestep. '_' sorts after '0'-'9' in ASCII,
        # so a plain sorted(...)[-1] picks up '_read_as' instead of the real
        # last timestep and crashes when something tries to .keys() a
        # Dataset. Keep only the all-digit "%05d"-style keys SaveData.
        # save_timestep() actually writes.
        ts_keys = sorted(k for k in struct_grp["timestep_info"].keys() if k.isdigit())
        if not ts_keys:
            raise ValueError("no timestep_info entries in savedata file")
        deformed_pos = struct_grp["timestep_info"][ts_keys[-1]]["pos"][()]
 
        # Structural definition (not per-timestep state) -- SHARPy's Beam
        # class saves these as plain arrays alongside ini_info/timestep_info
        # whenever SaveData's save_struct is True. Read them here, while the
        # file is still open; None if this run's file predates them / never
        # had them (older SHARPy, or a stripped-down savedata write).
        beam_number = struct_grp["beam_number"][()] if "beam_number" in struct_grp else None
        connectivities = struct_grp["connectivities"][()] if "connectivities" in struct_grp else None
 
    node = WING_TIP_NODE_INDEX
    if node is None:
        node = int(np.argmax(np.abs(jig_pos[:, SPAN_AXIS_INDEX])))
 
    jig_xyz = jig_pos[node]
    def_xyz = deformed_pos[node]
    delta = def_xyz - jig_xyz
 
    if WING_TIP_NODE_INDEX is None and not _tip_node_reported:
        print(f"  [tip deflection] auto-detected wingtip node index {node} "
              f"(jig position x={jig_xyz[0]:.4f}, y={jig_xyz[1]:.4f}, "
              f"z={jig_xyz[2]:.4f} m), from {run_id} -- verify this is "
              f"actually the wingtip and not another extremity; set "
              f"WING_TIP_NODE_INDEX explicitly if not.")
        _tip_node_reported = True
 
    semispan = SPAN_MAIN / 2
    result = {
        "wing_tip_node": node,
        "jig_tip_y": float(jig_xyz[SPAN_AXIS_INDEX]),
        "tip_deflection_vertical_m": float(delta[VERTICAL_AXIS_INDEX]),
        "tip_deflection_total_m": float(np.linalg.norm(delta)),
        "tip_deflection_pct_semispan": float(delta[VERTICAL_AXIS_INDEX] / semispan * 100),
    }
 
    # Wing-shape (plot 5) fields -- best effort, isolated from the fields
    # above on purpose. See the docstring: a failure here must only cost the
    # wing-shape plot, never the tip-deflection plot that depends on nothing
    # in this block.
    try:
        wing_nodes = None
        selection_method = None
        if WING_NODE_INDICES is not None:
            wing_nodes = np.array(
                sorted(WING_NODE_INDICES, key=lambda n: abs(jig_pos[n, SPAN_AXIS_INDEX]))
            )
            selection_method = "WING_NODE_INDICES override"
        elif beam_number is not None and connectivities is not None:
            try:
                wing_nodes, beam_num_used = _select_wing_nodes_by_beam(
                    jig_pos, beam_number, connectivities, node)
                selection_method = (f"beam_number == {beam_num_used} (read from this run's own "
                                     f"FE model -- the authoritative method, not a coordinate guess)")
            except Exception as e:
                warnings.warn(f"{run_id}: beam_number-based wing-node selection failed ({e}) -- "
                               f"falling back to the coordinate heuristic for this run")
 
        if wing_nodes is None:
            wing_nodes = _select_wing_nodes(jig_pos, node)
            selection_method = selection_method or (
                "coordinate heuristic: same side of root as the tip, sorted by |y| -- "
                "'beam_number'/'connectivities' weren't in this savedata file, so the "
                "authoritative method wasn't available")
 
        if WING_NODE_INDICES is None and not _wing_nodes_reported:
            print(f"  [wing shape] auto-detected {len(wing_nodes)} wing-side node(s) for the "
                  f"spanwise deflection plot, root to tip (from {run_id}), via {selection_method}:")
            for n in wing_nodes:
                print(f"      node {n:3d}: jig x={jig_pos[n, 0]:.4f}, y={jig_pos[n, 1]:.4f}, "
                      f"z={jig_pos[n, 2]:.4f} m")
            print("      verify this is really just the wing -- watch for a sudden jump in x "
                  "partway through the list (a sign a tail/fuselage/strut node got pulled in) -- "
                  "and set WING_NODE_INDICES explicitly if so.")
            _wing_nodes_reported = True
 
        result["span_y_jig"] = jig_pos[wing_nodes, SPAN_AXIS_INDEX]
        result["span_z_jig"] = jig_pos[wing_nodes, VERTICAL_AXIS_INDEX]
        result["span_z_deformed"] = deformed_pos[wing_nodes, VERTICAL_AXIS_INDEX]
    except Exception as e:
        warnings.warn(f"{run_id}: tip deflection OK, but failed to build the wing-shape "
                       f"data for plot 5 ({e}) -- wing-shape plot will skip this run")
 
    return result

# ============================================================================
# Main scan
# ============================================================================

def main():
    if not os.path.isdir(OUTPUT_ROOT):
        raise SystemExit(f"OUTPUT_ROOT does not exist: {OUTPUT_ROOT}")

    entries = sorted(os.listdir(OUTPUT_ROOT))
    matched, skipped = [], []

    for name in entries:
        full = os.path.join(OUTPUT_ROOT, name)
        if not os.path.isdir(full):
            continue
        m = FOLDER_PATTERN.match(name)
        if not m:
            skipped.append(name)
            continue
        camber_tag = m.group("camber_tag") or "nocamber"
        matched.append((name, full, m.group("velocity"), camber_tag))

    print(f"Found {len(matched)} matching run folders, skipped {len(skipped)} non-matching:")
    for s in skipped:
        print(f"  skipped: {s}")
    print()

    # cache alpha_L0 per camber tag so the integral only runs once each
    alpha_l0_cache = {}

    rows = []
    wing_shape_data = {}  # (camber_tag, u_inf) -> {span_y_jig, span_z_jig, span_z_deformed}
    for folder_name, run_dir, vel_str, camber_tag in matched:
        u_inf = float(vel_str)
        run_id = folder_name

        if camber_tag not in CAMBER_MP:
            warnings.warn(f"{run_id}: unknown camber tag '{camber_tag}', no (M,P) mapping - skipping")
            continue
        if camber_tag not in alpha_l0_cache:
            mp = CAMBER_MP[camber_tag]
            alpha_l0_cache[camber_tag] = alpha_L0_deg(mp["M"], mp["P"])
        a_l0 = alpha_l0_cache[camber_tag]

        trim_path = os.path.join(run_dir, "statictrim", "trim_iterations.txt")
        if not os.path.exists(trim_path):
            warnings.warn(f"{run_id}: missing trim_iterations.txt - skipping")
            continue
        try:
            trim = parse_trim_iterations(trim_path)
        except Exception as e:
            warnings.warn(f"{run_id}: failed to parse trim_iterations.txt ({e}) - skipping")
            continue

        converged = (
            abs(trim["Fx"]) < FX_TOL
            and abs(trim["Fz"]) < FZ_TOL
            and abs(trim["My"]) < M_TOL
        )
        if not converged:
            warnings.warn(
                f"{run_id}: NOT converged within tolerance "
                f"(Fx={trim['Fx']:.4f}, Fz={trim['Fz']:.4f}, My={trim['My']:.4f}) "
                f"- included in output but flagged"
            )

        forces_path = find_forces_csv(run_dir, folder_name)
        cl_sharpy = np.nan
        if forces_path is None:
            warnings.warn(f"{run_id}: no forces CSV found - CL left blank")
        else:
            try:
                cl_sharpy = parse_forces_csv(forces_path, u_inf)
            except Exception as e:
                warnings.warn(f"{run_id}: failed to parse forces CSV ({e})")

        wing_tip_node = np.nan
        jig_tip_y = tip_defl_vert = tip_defl_total = tip_defl_pct = np.nan
        if h5py is None:
            pass  # warned once, after the loop -- see below
        else:
            savedata_path = find_savedata_h5(run_dir, folder_name)
            if savedata_path is None:
                warnings.warn(f"{run_id}: no savedata .h5 found -- tip deflection left "
                               f"blank (add 'SaveData' with save_struct: True to the "
                               f"case's flow list to enable this)")
            else:
                try:
                    defl = parse_tip_deflection(savedata_path, run_id)
                    wing_tip_node = defl["wing_tip_node"]
                    jig_tip_y = defl["jig_tip_y"]
                    tip_defl_vert = defl["tip_deflection_vertical_m"]
                    tip_defl_total = defl["tip_deflection_total_m"]
                    tip_defl_pct = defl["tip_deflection_pct_semispan"]
                    # The wing-shape fields are best-effort inside
                    # parse_tip_deflection() and may be absent even when the
                    # tip-deflection fields above succeeded (it already
                    # warned in that case) -- only populate the plot-5 data
                    # when they actually came back.
                    if "span_y_jig" in defl:
                        wing_shape_data[(camber_tag, u_inf)] = {
                            "span_y_jig": defl["span_y_jig"],
                            "span_z_jig": defl["span_z_jig"],
                            "span_z_deformed": defl["span_z_deformed"],
                        }
                except Exception as e:
                    warnings.warn(f"{run_id}: failed to parse savedata .h5 ({e})")
 
        rows.append({
            "run_id": run_id,
            "u_inf": u_inf,
            "camber_tag": camber_tag,
            "M": CAMBER_MP[camber_tag]["M"],
            "P": CAMBER_MP[camber_tag]["P"],
            "alpha_L0_deg": a_l0,
            "alpha_sharpy_deg": trim["alpha_deg"],
            "elev_deg": trim["elev_deg"],
            "thrust": trim["thrust"],
            "Fx_residual": trim["Fx"],
            "Fz_residual": trim["Fz"],
            "My_residual": trim["My"],
            "converged": converged,
            "CL_sharpy": cl_sharpy,
            "alpha_hand_deg": alpha_hand_deg(u_inf, a_l0),
            "CL_hand": CL_hand(u_inf),
            # wing-alone hand-calc CL evaluated at SHARPy's own trimmed alpha --
            # a convenience column for residual checks, not used directly by
            # any of the three plots below (each plots the continuous curves
            # instead).
            "CL_hand_at_sharpy_alpha": CL_hand_from_alpha(trim["alpha_deg"], a_l0),
            "wing_tip_node": wing_tip_node,
            "jig_tip_y": jig_tip_y,
            "tip_deflection_vertical_m": tip_defl_vert,
            "tip_deflection_total_m": tip_defl_total,
            "tip_deflection_pct_semispan": tip_defl_pct,
        })
 

    if h5py is None:
        warnings.warn("h5py is not installed -- wing-tip deflection columns/plot are "
                       "all blank this run. Install it (pip install h5py; it's already "
                       "a SHARPy dependency, so it should be in the same environment) "
                       "to enable them.")
        
    if not rows:
        raise SystemExit("No runs successfully parsed - check OUTPUT_ROOT and FOLDER_PATTERN.")

    df = pd.DataFrame(rows).sort_values(["camber_tag", "u_inf"]).reset_index(drop=True)

    dupes = df[df.duplicated(subset=["u_inf", "camber_tag"], keep=False)]
    if not dupes.empty:
        warnings.warn(f"Duplicate (u_inf, camber) pairs found - check these run_ids:\n{dupes['run_id'].tolist()}")

    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_CSV}")
    print(df.to_string(index=False))

    n_unconverged = (~df["converged"]).sum()
    if n_unconverged:
        print(f"\nWARNING: {n_unconverged} run(s) did not converge within tolerance - see 'converged' column.")

    camber_tags = sorted(df["camber_tag"].unique())
    v_fine = np.linspace(df["u_inf"].min() * 0.95, df["u_inf"].max() * 1.05, 200)

    # ---------------------------------
    # Plot 1: alpha vs u_inf - one hand-calc + one SHARPy curve per camber,
    # plus ASWing cruise-speed and zero-lift-angle reference lines.
    # ---------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 5.5))

    # Tracks whether ALPHA_PLOT_XMAX actually cuts anything off on this plot
    # (any series' alpha genuinely exceeding it) -- only then do we trim the
    # drawn data and open up a blank margin above the cap; otherwise this
    # plot is left on its normal autoscaled y-axis, untouched.
    alpha1_capped = False

    for i, camber_tag in enumerate(camber_tags):
        color = PALETTE[i % len(PALETTE)]
        sub = df[df["camber_tag"] == camber_tag].sort_values("u_inf")
        a_l0 = alpha_l0_cache[camber_tag]
        mp = CAMBER_MP[camber_tag]

        sharpy_x = sub["u_inf"].to_numpy()
        sharpy_y = sub["alpha_sharpy_deg"].to_numpy()
        hand_x = v_fine
        hand_y = alpha_hand_deg(v_fine, a_l0)
        if ALPHA_PLOT_XMAX is not None:
            if np.any(sharpy_y > ALPHA_PLOT_XMAX) or np.any(hand_y > ALPHA_PLOT_XMAX):
                alpha1_capped = True
            sharpy_x, sharpy_y = _trim_to_cap(sharpy_x, sharpy_y, ALPHA_PLOT_XMAX, cap_on="y")
            hand_x, hand_y = _trim_to_cap(hand_x, hand_y, ALPHA_PLOT_XMAX, cap_on="y")

        ax1.plot(sharpy_x, sharpy_y, "o-", color=color, lw=2, ms=6,
                 label=f"SHARPy, {camber_tag} (M={mp['M']}, P={mp['P']})")
        ax1.plot(hand_x, hand_y, "-", color=color, lw=1.6, alpha=0.6,
                 label=f"hand calc, {camber_tag}")

        unconverged = sub[~sub["converged"]]
        if not unconverged.empty:
            unc_x = unconverged["u_inf"].to_numpy()
            unc_y = unconverged["alpha_sharpy_deg"].to_numpy()
            if ALPHA_PLOT_XMAX is not None:
                unc_x, unc_y = _trim_to_cap(unc_x, unc_y, ALPHA_PLOT_XMAX, cap_on="y")
            if len(unc_x):
                ax1.plot(unc_x, unc_y, "x", color="black", ms=10, mew=2, zorder=5,
                         label="_nolegend_")

    ax1.axvline(CRUISE_SPEED_ASWING, color=ASWING_COLOR, ls=":", lw=1.4,
                label=f"cruise speed (ASWing), {CRUISE_SPEED_ASWING:g} m/s")
    # Zero-lift angle deliberately NOT shown here - it's a fixed alpha, not
    # tied to a trim state at any particular velocity, so a horizontal line on
    # an alpha-vs-V plot doesn't represent anything physically meaningful on
    # this graph. It belongs on the CL-vs-alpha plot instead, where it's the
    # actual x-intercept of the ASWing lift curve (plot 3, below).

    ax1.set_xlabel("u_inf (m/s)")
    ax1.set_ylabel("alpha (deg)")
    _cap_axis_with_gap(ax1, "y", ALPHA_PLOT_XMAX if alpha1_capped else None, ALPHA_PLOT_GAP)
    _style_axes(ax1, "LLTA trim sweep - SHARPy vs. hand calc, by camber")
    fig1.tight_layout()
    fig1.savefig(OUT_PLOT_ALPHA_V, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"\nSaved plot to {OUT_PLOT_ALPHA_V}")

    # ---------------------------------
    # Plot 2: CL vs u_inf - SHARPy CL per camber vs. the single
    # camber-independent trim CL_required(V) hand calc, plus ASWing
    # cruise-speed / CLmax / CLmin reference lines.
    # ---------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5.5))

    for i, camber_tag in enumerate(camber_tags):
        color = PALETTE[i % len(PALETTE)]
        sub = df[df["camber_tag"] == camber_tag].sort_values("u_inf")
        mp = CAMBER_MP[camber_tag]

        ax2.plot(sub["u_inf"], sub["CL_sharpy"], "o-", color=color, lw=2, ms=6,
                 label=f"SHARPy CL, {camber_tag} (M={mp['M']}, P={mp['P']})")

        unconverged = sub[~sub["converged"]]
        if not unconverged.empty:
            ax2.plot(unconverged["u_inf"], unconverged["CL_sharpy"], "x", color="black",
                     ms=10, mew=2, zorder=5, label="_nolegend_")

    # one curve, not per camber - see CL_hand()'s docstring for why
    ax2.plot(v_fine, CL_hand(v_fine), "-", color=INK, lw=1.8, alpha=0.7,
              label="hand calc, CL required (trim)")

    ax2.axvline(CRUISE_SPEED_ASWING, color=ASWING_COLOR, ls=":", lw=1.4,
                label=f"cruise speed (ASWing), {CRUISE_SPEED_ASWING:g} m/s")
    # CLmax/CLmin as one shaded band + one legend entry, instead of two
    # separate full-width lines each carrying their own "(ASWing)" label -
    # reads as "the usable CL envelope" at a glance, and halves the legend
    # rows this pair used to cost.
    ax2.axhspan(CL_MIN_ASWING, CL_MAX_ASWING, color=ASWING_COLOR, alpha=0.07, lw=0,
                label=f"ASWing usable CL range ({CL_MIN_ASWING:g} to {CL_MAX_ASWING:g})")

    ax2.set_xlabel("u_inf (m/s)")
    ax2.set_ylabel("CL")
    _style_axes(ax2, "LLTA trim sweep - CL vs. velocity")
    fig2.tight_layout()
    fig2.savefig(OUT_PLOT_CL_V, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved plot to {OUT_PLOT_CL_V}")

    # ---------------------------------
    # Plot 3: CL vs alpha - SHARPy points vs. the wing-alone hand-calc lift
    # curve (one per camber, theoretical) vs. a linear regression fit through
    # that camber's own SHARPy points vs. the ASWing full-aircraft lift curve,
    # plus CLmax/CLmin. This is where dCL/dalpha (ASWing) actually belongs as
    # a plotted line, since alpha is the x-axis here.
    # ---------------------------------
    fig3, ax3 = plt.subplots(figsize=(9.5, 5.5))

    # Wide enough past the actual data range to show the curves flattening
    # towards CLmax/CLmin, not just their straight-line middle section.
    all_alpha = df["alpha_sharpy_deg"].to_numpy()
    alpha_fine = np.linspace(min(all_alpha.min(), ALPHA_L0_ASWING_DEG) - 8,
                              all_alpha.max() + 8, 300)

    # Whether ALPHA_PLOT_XMAX actually cuts anything off this fine grid --
    # only then do the curves below get trimmed and the axis opened up with
    # a blank margin past the cap; otherwise plot 3 keeps its normal
    # autoscaled x-axis, untouched.
    alpha3_capped = ALPHA_PLOT_XMAX is not None and alpha_fine.max() > ALPHA_PLOT_XMAX

    def _capped(x, y):
        """Trim (x, y) to x <= ALPHA_PLOT_XMAX for display, when that cap is
        active and actually cuts something off; otherwise pass through
        unchanged. Local to plot 3 since it closes over alpha3_capped."""
        if not alpha3_capped:
            return x, y
        return _trim_to_cap(x, y, ALPHA_PLOT_XMAX, cap_on="x")

    for i, camber_tag in enumerate(camber_tags):
        color = PALETTE[i % len(PALETTE)]
        sub = df[df["camber_tag"] == camber_tag].sort_values("alpha_sharpy_deg")
        a_l0 = alpha_l0_cache[camber_tag]
        mp = CAMBER_MP[camber_tag]

        # Labels deliberately don't repeat "(M=x, P=y)" here (unlike plot 1) -
        # with up to 3 cambers x up to 3 line types this legend fills up fast,
        # and color already ties every line/point for a camber together. The
        # M/P mapping is in CAMBER_MP at the top of the script and in plot 1.
        ax3.plot(*_capped(sub["alpha_sharpy_deg"].to_numpy(), sub["CL_sharpy"].to_numpy()),
                 "o", color=color, ms=6, label=f"SHARPy, {camber_tag}")
        ax3.plot(*_capped(alpha_fine, CL_hand_from_alpha(alpha_fine, a_l0)), "-", color=color,
                 lw=1.6, alpha=0.6, label=f"hand calc (theory), {camber_tag}")

        # Second line, genuinely data-driven: least-squares fit through this
        # camber's own SHARPy points (not the theoretical CL_alpha/alpha_L0),
        # shaped the same way near stall. Distinguishes "what theory predicts"
        # from "what a line through the actual runs looks like" - needs at
        # least 2 points with a valid CL (skipped + warned otherwise, e.g. a
        # camber with only one converged run or missing forces files). Set
        # SHOW_SHARPY_DATA_FIT = False at the top of the script to drop this
        # line (and its legend entry) entirely if the plot feels crowded.
        if SHOW_SHARPY_DATA_FIT:
            fit = _linear_fit_deg(sub["alpha_sharpy_deg"], sub["CL_sharpy"])
            if fit is None:
                warnings.warn(f"{camber_tag}: fewer than 2 valid (alpha, CL) points - "
                               f"skipping the SHARPy-data-fit line for this camber")
            else:
                slope_fit, a0_fit = fit
                ax3.plot(*_capped(alpha_fine, _stall_shape(alpha_fine, a0_fit, slope_fit,
                                                            CL_MAX_ASWING, CL_MIN_ASWING)),
                         ":", color=color, lw=2.0, label=f"SHARPy data fit, {camber_tag}")

    ax3.plot(*_capped(alpha_fine, CL_aswing_from_alpha(alpha_fine)), "-", color=ASWING_COLOR, lw=2,
             label=f"ASWing (full-aircraft), dCL/dα={DCL_DALPHA_ASWING:g}/rad")

    # Same shaded-band treatment as plot 2: one legend entry instead of two.
    ax3.axhspan(CL_MIN_ASWING, CL_MAX_ASWING, color=ASWING_COLOR, alpha=0.07, lw=0,
                label=f"ASWing usable CL range ({CL_MIN_ASWING:g} to {CL_MAX_ASWING:g})")

    ax3.set_xlabel("alpha (deg)")
    ax3.set_ylabel("CL")
    _cap_axis_with_gap(ax3, "x", ALPHA_PLOT_XMAX if alpha3_capped else None, ALPHA_PLOT_GAP)
    _style_axes(ax3, "LLTA: CL vs. alpha -- SHARPy, hand calc, SHARPy fit, ASWing", legend_outside=True)
    fig3.tight_layout()
    fig3.savefig(OUT_PLOT_CL_ALPHA, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved plot to {OUT_PLOT_CL_ALPHA}")

    # ------------------------------------------------------------------
    # Plot 4: wing-tip deflection vs u_inf -- vertical (out-of-plane
    # bending) displacement of the wingtip node at the trim state, relative
    # to its jig-shape (undeformed) position, one curve per camber. Skipped
    # entirely (with a message, not an error) if no run produced any
    # deflection data -- e.g. h5py isn't installed, or SaveData isn't in the
    # flow yet -- same graceful-degradation pattern as CL_sharpy above.
    # ------------------------------------------------------------------
    if df["tip_deflection_vertical_m"].notna().any():
        fig4, ax4 = plt.subplots(figsize=(8, 5.5))
 
        for i, camber_tag in enumerate(camber_tags):
            color = PALETTE[i % len(PALETTE)]
            sub = df[df["camber_tag"] == camber_tag].sort_values("u_inf")
            mp = CAMBER_MP[camber_tag]
 
            ax4.plot(sub["u_inf"], sub["tip_deflection_vertical_m"], "o-", color=color, lw=2, ms=6,
                     label=f"{camber_tag} (M={mp['M']}, P={mp['P']})")
            if SHOW_TOTAL_TIP_DEFLECTION:
                ax4.plot(sub["u_inf"], sub["tip_deflection_total_m"], "^--", color=color, lw=1.4,
                         alpha=0.6, ms=5, label=f"{camber_tag}, total 3D magnitude")
 
            unconverged = sub[~sub["converged"]]
            if not unconverged.empty:
                ax4.plot(unconverged["u_inf"], unconverged["tip_deflection_vertical_m"], "x",
                         color="black", ms=10, mew=2, zorder=5, label="_nolegend_")
 
        ax4.axhline(0, color=GRID, lw=1.0)
 
        ax4.set_xlabel("u_inf (m/s)")
        ax4.set_ylabel("wingtip vertical deflection (m), relative to jig shape")
        _style_axes(ax4, "LLTA trim sweep: wingtip deflection vs. velocity")
        fig4.tight_layout()
        fig4.savefig(OUT_PLOT_TIP_DEFL, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"Saved plot to {OUT_PLOT_TIP_DEFL}")
    else:
        print(f"\nNo wing-tip deflection data available in any run -- skipped "
              f"{OUT_PLOT_TIP_DEFL}. Add 'SaveData' (save_struct: True) to your case's "
              f"flow list and re-run the sweep to enable this plot.")

    # ------------------------------------------------------------------
    # Plot 5: deflected wing shape along the span -- absolute vertical (z)
    # position of every wing-side node from root to tip, for the jig shape
    # and for every u_inf, faceted by camber (one subplot per camber tag).
    # u_inf is a continuous quantity, so it's encoded as a colorbar
    # (single hue, light -> dark) rather than one legend line per velocity
    # -- with up to ~10 velocities per camber, a discrete legend here would
    # just reproduce the "too many lines" clutter problem plot 3 already
    # ran into. The jig shape is the one genuinely categorical line (same
    # physical mesh regardless of u_inf) and gets the one legend entry.
    # Requires the same SaveData output as plot 4 -- skipped (with a
    # message, not an error) if no run produced wing-shape data.
    # ------------------------------------------------------------------
    if wing_shape_data:
        cambers_with_shape = sorted({c for (c, _u) in wing_shape_data})
        fig5, axes5 = plt.subplots(len(cambers_with_shape), 1,
                                    figsize=(9, 3.2 * len(cambers_with_shape)),
                                    sharex=True, sharey=True, squeeze=False,
                                    constrained_layout=True)
        axes5 = axes5[:, 0]
 
        all_u = sorted({u for (_c, u) in wing_shape_data})
        norm = plt.Normalize(vmin=min(all_u), vmax=max(all_u))
        cmap = SEQUENTIAL_BLUE

        # See NORMALIZE_WING_SHAPE_SPAN in CONFIG: dividing by the semispan
        # makes the x-axis a dimensionless 0-1 fraction instead of meters, in
        # which case a literal 1:1 aspect ratio against the still-in-meters
        # y-axis wouldn't mean anything physical, so the aspect is only
        # forced to "equal" in the default (both-axes-in-meters) mode.
        x_scale = (SPAN_MAIN / 2) if NORMALIZE_WING_SHAPE_SPAN else 1.0
        x_label = ("normalized span, |y| / semispan (-)" if NORMALIZE_WING_SHAPE_SPAN
                    else "distance from root, |y| (m)")
 
        for ax, camber_tag in zip(axes5, cambers_with_shape):
            entries = {u: d for (c, u), d in wing_shape_data.items() if c == camber_tag}
            u_list = sorted(entries)
 
            y_dist = np.abs(entries[u_list[0]]["span_y_jig"]) / x_scale
            z_jig_ref = entries[u_list[0]]["span_z_jig"]
            for u in u_list[1:]:
                if not np.allclose(entries[u]["span_z_jig"], z_jig_ref, atol=1e-6):
                    warnings.warn(f"{camber_tag}: jig shape differs between u_inf={u_list[0]:g} "
                                   f"and u_inf={u:g} by more than 1e-6 m -- unexpected, since the "
                                   f"jig shape shouldn't depend on trim state; check the mesh is "
                                   f"really identical across this camber's runs")
                    break

            ax.plot(y_dist, z_jig_ref, "--", color=INK, lw=1.0, label="jig shape", zorder=5)
            for u in u_list:
                ax.plot(np.abs(entries[u]["span_y_jig"]) / x_scale, entries[u]["span_z_deformed"],
                         "-", color=cmap(norm(u)), lw=0.9, marker="o", ms=2)

            if not NORMALIZE_WING_SHAPE_SPAN and WING_SHAPE_SPAN_AXIS_MAX is not None:
                x_max = WING_SHAPE_SPAN_AXIS_MAX
            else:
                x_max = (SPAN_MAIN / 2) / x_scale
            ax.set_xlim(0, x_max)
            _ensure_boundary_tick(ax, "x", x_max)
            if WING_SHAPE_Z_AXIS_MAX is not None:
                ax.set_ylim(top=max(ax.get_ylim()[1], WING_SHAPE_Z_AXIS_MAX))
                _ensure_boundary_tick(ax, "y", WING_SHAPE_Z_AXIS_MAX)
            if not NORMALIZE_WING_SHAPE_SPAN:
                # 1:1 scale on x and y -- both are physical positions in
                # meters, so this is what makes the plotted shape actually
                # look like the wing's real proportions rather than a
                # stretched-out cartoon of the bend.
                ax.set_aspect("equal", adjustable="box")
            mp = CAMBER_MP[camber_tag]
            _style_axes(ax, f"{camber_tag} (M={mp['M']}, P={mp['P']})")
            ax.set_ylabel("z (m)")
 
        axes5[-1].set_xlabel(x_label)
 
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig5.colorbar(sm, ax=list(axes5), pad=0.02)
        cbar.set_label("u_inf (m/s)")
 
        fig5.suptitle("LLTA: deflected wing shape vs. jig shape, by camber",
                       x=0.01, ha="left", fontsize=12, color=INK)
        fig5.savefig(OUT_PLOT_WING_SHAPE, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"Saved plot to {OUT_PLOT_WING_SHAPE}")
    else:
        print(f"\nNo wing-shape data available in any run -- skipped {OUT_PLOT_WING_SHAPE}. "
              f"Same requirement as the tip-deflection plot: add 'SaveData' (save_struct: True) "
              f"to your case's flow list and re-run the sweep.")

if __name__ == "__main__":
    main()