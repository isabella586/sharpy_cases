"""
LLTA trim sweep analysis

Scans SHARPy StaticTrim output folders, extracts the trim results for each run, computes
hand-calc alpha(V), CL(V) and CL(alpha) curves, and compares everything with Voltitude's
ASWING trim sweep. Writes nine plots, a results CSV, and a lift-curve slope summary printed
to the console.

Runs are grouped by a free-form RUN TAG taken from the folder name
(llta_trim_u_inf_<V>_<run_tag>). A tag can stand for any investigation: camber, mesh, wake
length, stiffness, a fixed control-surface setting... RUN_TAGS below gives each tag a legend
label and, where it differs from DEFAULT_CAMBER, the camber used by the hand calc.

Plots
    1. alpha vs u_inf            SHARPy per run tag, hand calc per camber, ASWING trim points
    2. CL vs u_inf               SHARPy CL vs the trim requirement CL = 2W/(rho V^2 S), ASWING
    3. CL vs alpha               SHARPy points + straight-line fit per run tag, wing-alone hand
                                 calc, ASWING lift-curve constants and trim points
    4. tip deflection vs u_inf   vertical wingtip displacement from the jig shape
    5. wing shape vs span        jig and deformed wing shape, one panel per run tag
    6. elevator vs u_inf         SHARPy trim elevator vs ASWING (Flap 1)
    7. thrust vs u_inf           SHARPy trim thrust vs ASWING ENG (both in N)
    8. whole aircraft vs wing    per run tag: CL-alpha fit on whole-aircraft CL vs wing-only CL
    9. alpha vs 1/u_inf^2        with L = W, alpha = alpha_0 + (CL V^2 / a) / V^2 is a straight
                                 line: intercept = zero-lift angle, slope -> lift-curve slope a.
                                 Right panel: residuals from each straight-line fit.

All fits (slope summary, plots 3, 8 and 9) use only speeds >= V_MIN_FIT, so post-stall points
(5 and 5.5 m/s, above ASWING's CLmax) do not bias them. Excluded points are still plotted.

Inputs per run folder:  statictrim/trim_iterations.txt, forces/*aeroforces.csv,
savedata/<case>.data.h5 (SaveData with save_struct and save_aero), and the case's
<SHARPY_CASES_DIR>/<case>.sharpy for the StaticTrim convergence tolerances.

Usage:  python trim_sweep_analysis.py   (edit the CONFIG block first)
"""

import os
import re          
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import h5py
from configobj import ConfigObj   # SHARPy's own .sharpy file reader
import llta_geometry as llta

# ============================================================================
# CONFIG
# ============================================================================

OUTPUT_ROOT = "/Users/icc25/sharpy_cases/llta/output"
SHARPY_CASES_DIR = "/Users/icc25/sharpy_cases/LLTA"     # holds <case>.sharpy

# Folder naming: llta_trim_u_inf_<V>[_<run_tag>] ("uinf" also accepted). A folder with no tag
# is the symmetric-aerofoil baseline, tag "nocamber". Non-matching folders are skipped.
FOLDER_PATTERN = re.compile(
    r"^llta_trim_u_?inf_(?P<velocity>\d+(?:\.\d+)?)(?:__(?P<run_tag>[A-Za-z]\w*))?$"
)

# Camber (M, P) in the generate_naca_camber(M, P) convention: M = max camber (% chord),
# P = its position (tenths of chord). Used only for the hand calc's zero-lift angle.
DEFAULT_CAMBER = (6, 4)

# One entry per run tag. All keys optional:
#   label    - legend text (default: the tag itself)
#   camber   - (M, P) if not DEFAULT_CAMBER
#   variable, value - what the tag investigates; written to the CSV for filtering
# Tags found in OUTPUT_ROOT but missing here still run, with DEFAULT_CAMBER and the tag as label.
RUN_TAGS = {
    "nocamber": {"label": "no camber", "camber": (0, 0), "variable": "camber"},
    "camber": {"label": "camber 6/4", "variable": "camber"},
    "camber2": {"label": "camber 5/5", "camber": (5, 5), "variable": "camber"},
    "camber_ccgfixed": {"label": "M0 (ccg fixed)", "variable": "model version"},
    "camber_modelaudit": {"label": "M1 (model audit)", "variable": "model version"},
    "camber_wakelength": {"label": "m = 4, wake rule", "variable": "chordwise panels", "value": 4},
    "camber_chorddiscretisation": {"label": "m = 6", "variable": "chordwise panels", "value": 6},
    "camber_chorddiscretisation16": {"label": "m = 16", "variable": "chordwise panels", "value": 16},
    "camber_fuselagestiffnessupdated": {"label": "fuselage EI/GJ updated", "variable": "boom stiffness"},
    "llta_trim_u_inf_9__baseline": {"label": "alpha offset", "variable": "wing incidence = 0 deg"},
    "llta_trim_u_inf_9__wing_incidence": {"label": "alpha offset", "variable": "wing incidence = -3 deg"},
    "llta_trim_u_inf_9__control_deflection": {"label": "alpha offset", "variable": "control surface incidence = -3 deg"},
}

# Which run tags appear on the plots (legend order = colour order). None = every tag found.
# The CSV always holds every run.
TAGS_TO_PLOT = None

# Fits use only speeds >= this (post-stall points excluded, still plotted).
V_MIN_FIT = 6.0  # m/s

# Fallback StaticTrim tolerances, used only when a case's .sharpy file can't be read.
# They only flag non-converged runs; they don't change any extracted value.
DEFAULT_TRIM_TOL = {"fx_tolerance": 0.01, "fz_tolerance": 0.01, "m_tolerance": 0.01}

# Aircraft / flight condition
MASS = llta.aircraft_mass     # kg
G = 9.81
RHO = 1.225                   # kg/m^3
CHORD_MAIN = llta.chord_main  # m
SPAN_MAIN = llta.span_main    # m
OSWALD_E = 0.85

# Plot options
SHOW_SHARPY_DATA_FIT = True       # plot 3: straight-line fit through each tag's SHARPy points
SHOW_TOTAL_TIP_DEFLECTION = False  # plot 4: also draw the 3D tip displacement magnitude
# Display only: on plots 1 and 3, points and curves beyond this alpha are not drawn and the axis
# stops ALPHA_PLOT_GAP past it. The CSV and every fit still use all the data. None = autoscale.
ALPHA_PLOT_XMAX = 15   # deg
ALPHA_PLOT_GAP = 1.5   # deg

# Plot 5 (wing shape)
NORMALIZE_WING_SHAPE_SPAN = True   # x-axis as |y|/semispan (True) or |y| in m (False)
WING_SHAPE_SPAN_AXIS_MAX = 2.0     # m, x-axis limit in metres mode (None = semispan)
WING_SHAPE_Z_AXIS_MAX = 0.30       # m, minimum z-axis top (None = autoscale)

# Node selection (0-based FE node indices). None = auto-detect: tip = largest |y| in the jig
# shape; wing = nodes on the tip's beam_number (printed once - check it).
WING_TIP_NODE_INDEX = None
WING_NODE_INDICES = None
SPAN_AXIS_INDEX, VERTICAL_AXIS_INDEX = 1, 2   # y spanwise, z up (A frame)

# Aero surfaces (generate_aero_file): 0/1 main wing right/left, 2 fin, 3/4 tailplane.
WING_SURFACES = (0, 1)

# --------------------------------------------------------------------------
# ASWING reference data (Voltitude)
# --------------------------------------------------------------------------
CRUISE_SPEED_ASWING = 9     # m/s, vertical line on plots 1, 2, 6, 7
CL_MAX_ASWING = 1.34        # shaded usable-CL band on plots 2 and 3
CL_MIN_ASWING = -1
# Lift-curve constants from the run file's AVL wing block (dCLda 4.97, alpha 6.2). The sign of
# the zero-lift angle is assumed negative (cambered wing lifting at alpha = 0).
DCL_DALPHA_ASWING = 4.97    # /rad
ALPHA_L0_ASWING_DEG = -6.2  # deg

# ASWING trim sweep (one row per speed). elev_deg = Flap 1; thrust = ENG, in N like SHARPy's.
ASWING_TRIM_SWEEP = [
    {"u_inf": 6.5, "alpha_deg": 4.725, "elev_deg": -5.576, "cl": 1.1093, "thrust": 0.8469},
    {"u_inf": 7.0, "alpha_deg": 2.246, "elev_deg": -3.861, "cl": 0.9565, "thrust": 0.78156},
    {"u_inf": 7.5, "alpha_deg": 0.7525, "elev_deg": -3.106, "cl": 0.8332, "thrust": 0.7593},
    {"u_inf": 8.0, "alpha_deg": -0.399, "elev_deg": -2.582, "cl": 0.7323, "thrust": 0.75272},
    {"u_inf": 8.5, "alpha_deg": -1.309, "elev_deg": -2.204, "cl": 0.6487, "thrust": 0.75824},
    {"u_inf": 9.0, "alpha_deg": -2.031, "elev_deg": -1.937, "cl": 0.5786, "thrust": 0.77377},
    {"u_inf": 9.5, "alpha_deg": -2.603, "elev_deg": -1.76, "cl": 0.5193, "thrust": 0.79779},
    {"u_inf": 10.0, "alpha_deg": -3.051, "elev_deg": -1.657, "cl": 0.4687, "thrust": 0.82919},
]

# Outputs
OUT_CSV = "llta_sweep_results.csv"
OUT_PLOT_ALPHA_V = "llta_sweep_alpha_vs_v.png"
OUT_PLOT_CL_V = "llta_sweep_cl_vs_v.png"
OUT_PLOT_CL_ALPHA = "llta_sweep_cl_vs_alpha.png"
OUT_PLOT_TIP_DEFL = "llta_sweep_tip_deflection_vs_v.png"
OUT_PLOT_WING_SHAPE = "llta_sweep_wing_shape_vs_span.png"
OUT_PLOT_ELEV_V = "llta_sweep_elev_vs_v.png"
OUT_PLOT_THRUST_V = "llta_sweep_thrust_vs_v.png"
OUT_PLOT_WING_VS_TOTAL = "llta_sweep_wing_vs_total_cl_fit.png"
OUT_PLOT_ALPHA_INV_V2 = "llta_sweep_alpha_vs_inv_v2.png"

# ============================================================================
# Derived constants and styling
# ============================================================================

W = MASS * G
S = CHORD_MAIN * SPAN_MAIN
AR = SPAN_MAIN**2 / S
A0 = 2 * np.pi                                       # /rad, 2D thin-aerofoil slope
CL_ALPHA = A0 / (1 + A0 / (np.pi * OSWALD_E * AR))   # /rad, wing alone, finite-AR corrected

# Eight categorical colours, assigned to run tags in plotting order.
PALETTE = ["#2b6cb0", "#c0392b", "#2e7d32", "#8e44ad",
           "#d68910", "#138d90", "#6d4c41", "#d63384"]
INK, MUTED, GRID = "#1a2733", "#5b6b7a", "#d8dee4"
ASWING_COLOR = "#444444"
ASWING_SWEEP_MARKER = "s"
ASWING_SWEEP_KW = dict(marker=ASWING_SWEEP_MARKER, linestyle="none", ms=8, mew=1.8,
                       mfc="none", mec=ASWING_COLOR)
HAND_CALC_STYLES = ["-", "--", "-.", ":"]   # one per distinct camber
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("llta_seq_blue", [
    "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 1.0,
})


def tag_info(tag):
    """RUN_TAGS entry for a tag, with defaults filled in."""
    info = dict(RUN_TAGS.get(tag, {}))
    info.setdefault("label", tag)
    info.setdefault("camber", DEFAULT_CAMBER)
    info.setdefault("variable", "")
    info.setdefault("value", np.nan)
    return info


def _style_axes(ax, title, legend_outside=False):
    ax.set_title(title, fontsize=12, loc="left", color=INK)
    ax.grid(True, color=GRID, lw=0.7, alpha=0.7)
    if legend_outside:   # long legends go to the right of the axes
        ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED,
                  loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    else:
        ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _ensure_boundary_tick(ax, axis, boundary):
    """Add a tick at exactly `boundary` without widening the current axis limits."""
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
    """Drop points beyond `cap` (display only - see ALPHA_PLOT_XMAX)."""
    x, y = np.asarray(x), np.asarray(y)
    mask = (x if cap_on == "x" else y) <= cap
    return x[mask], y[mask]


def _cap_axis_with_gap(ax, axis, cap, gap):
    """Set the axis upper limit to cap + gap (no-op if cap is None)."""
    if cap is None:
        return
    if axis == "x":
        ax.set_xlim(right=cap + gap)
    else:
        ax.set_ylim(top=cap + gap)


def aswing_sweep_frame():
    """ASWING_TRIM_SWEEP as a DataFrame sorted by u_inf, or None if empty."""
    if not ASWING_TRIM_SWEEP:
        return None
    df_a = pd.DataFrame(ASWING_TRIM_SWEEP).sort_values("u_inf").reset_index(drop=True)
    data_cols = [c for c in df_a.columns if c != "u_inf"]
    if df_a[data_cols].isna().all(axis=None):
        return None
    return df_a

# ============================================================================
# Hand calc
# ============================================================================

def alpha_L0_deg(M_pct, P_tenths, n=20000):
    """Thin-aerofoil zero-lift angle (deg) of a NACA 4-digit camber line (closed-form integral).
    Checked against tabulated values: 2412 -> -2.08, 4412 -> -4.15, 2212 -> -1.80 deg."""
    m, p = M_pct * 1e-2, P_tenths * 1e-1
    if p == 0:
        return 0.0
    theta = np.linspace(1e-9, np.pi - 1e-9, n)
    x = (1 - np.cos(theta)) / 2
    dzdx = np.where(x < p, (2 * m / p**2) * (p - x), (2 * m / (1 - p)**2) * (p - x))
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return np.degrees(-(1 / np.pi) * trapz(dzdx * (np.cos(theta) - 1), theta))


def CL_hand(V):
    """Trim CL required for L = W at speed V (independent of camber)."""
    return 2 * W / (RHO * V**2 * S)


def alpha_hand_deg(V, a_l0_deg):
    """Hand-calc trim alpha: alpha_L0 + CL_required(V) / CL_alpha."""
    return a_l0_deg + np.degrees(CL_hand(V) / CL_ALPHA)


def lift_line(alpha_deg, alpha0_deg, slope_per_deg):
    """Straight lift curve CL = slope * (alpha - alpha0). No stall model: SHARPy's UVLM is
    inviscid, and ASWING's CLmax/CLmin are shown as a band instead."""
    return slope_per_deg * (np.asarray(alpha_deg, dtype=float) - alpha0_deg)


def _linear_fit_deg(alpha_deg_vals, cl_vals):
    """Least-squares CL-alpha line -> (slope_per_deg, alpha0_deg), or None (< 2 points)."""
    a = np.asarray(alpha_deg_vals, dtype=float)
    c = np.asarray(cl_vals, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(c))
    if mask.sum() < 2:
        return None
    slope, intercept = np.polyfit(a[mask], c[mask], 1)
    if slope == 0:
        return None
    return slope, -intercept / slope


def _fit_window(sub):
    """Rows of a run's sweep that the fits use (u_inf >= V_MIN_FIT)."""
    return sub[sub["u_inf"] >= V_MIN_FIT]

# ============================================================================
# Parsing
# ============================================================================

def parse_trim_iterations(path):
    """Last row of SHARPy's trim_iterations.txt ('|'-delimited table).
    Columns: iter, alpha [deg], elev [deg], thrust, Fx, Fy, Fz, Mx, My, Mz."""
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s.startswith("|") or set(s.replace("|", "")) <= {"="}:
                continue
            fields = [c.strip() for c in s.strip("|").split("|")]
            try:
                float(fields[0])
            except ValueError:
                continue   # header row
            rows.append(fields)
    if not rows:
        raise ValueError(f"No data rows parsed from {path}")
    last = rows[-1]
    return {"alpha_deg": float(last[1]), "elev_deg": float(last[2]), "thrust": float(last[3]),
            "Fx": float(last[4]), "Fz": float(last[6]), "My": float(last[8]),
            "n_iter": len(rows)}


def read_trim_tolerances(case_name):
    """StaticTrim fx/fz/m tolerances from <SHARPY_CASES_DIR>/<case>.sharpy.
    Returns (tolerances dict, source string); falls back to DEFAULT_TRIM_TOL per missing key."""
    path = os.path.join(SHARPY_CASES_DIR, f"{case_name}.sharpy")
    tol = dict(DEFAULT_TRIM_TOL)
    if not os.path.exists(path):
        return tol, "default (no .sharpy file)"
    section = ConfigObj(path).get("StaticTrim", {})
    found = [k for k in tol if k in section]
    for k in found:
        tol[k] = float(section[k])
    if len(found) == len(tol):
        return tol, ".sharpy"
    return tol, ".sharpy (+ defaults for " + ", ".join(k for k in tol if k not in found) + ")"


def parse_forces_csv(path, u_inf):
    """Whole-aircraft CL from the AeroForcesCalculator CSV:
    (fz_steady_G + fz_unsteady_G) / (q S), as AeroForcesCalculator does internally."""
    data = np.genfromtxt(path, delimiter=",", comments="#")
    last = np.atleast_2d(data)[-1]
    return (last[3] + last[6]) / (0.5 * RHO * u_inf**2 * S)


def _find_one(run_dir, sub, expected, pattern):
    path = os.path.join(run_dir, sub, expected)
    if os.path.exists(path):
        return path
    candidates = glob.glob(os.path.join(run_dir, sub, pattern))
    return candidates[0] if len(candidates) == 1 else None


def find_forces_csv(run_dir, case):
    return _find_one(run_dir, "forces", f"forces_{case}_aeroforces.csv", "*aeroforces*.csv")


def find_savedata_h5(run_dir, case):
    return _find_one(run_dir, "savedata", f"{case}.data.h5", "*.data.h5")


def _digit_keys(grp):
    """SaveData writes a '_read_as' entry into every group; keep only the '%05d' timesteps."""
    return sorted(k for k in grp.keys() if k.isdigit())


_tip_node_reported = False
_wing_nodes_reported = False


def _select_wing_nodes(jig_pos, beam_number, connectivities, tip_node):
    """Wing nodes root to tip: every node on the wingtip's beam_number, same side of the root.
    Falls back to 'same sign of y as the tip' if beam_number/connectivities weren't saved."""
    y = jig_pos[:, SPAN_AXIS_INDEX]
    side = np.sign(y[tip_node]) or 1.0
    if beam_number is not None and connectivities is not None:
        elem = np.any(connectivities == tip_node, axis=1)
        tip_beam = int(beam_number[elem][0])
        nodes = np.unique(connectivities[beam_number == tip_beam].ravel())
        method = f"beam_number == {tip_beam}"
    else:
        nodes = np.arange(len(y))
        method = "same side of root as the tip (no beam_number saved - check for tail nodes)"
    nodes = nodes[y[nodes] * side >= -1e-9]
    return nodes[np.argsort(np.abs(y[nodes]))], method


def parse_savedata(h5_path, u_inf, run_id):
    """From the savedata .h5: tip deflection (deformed minus jig, A frame, so trim rotation is
    excluded), the spanwise wing shape (plot 5) and wing-only CL (plot 8). Missing pieces
    come back as NaN / absent keys rather than errors."""
    global _tip_node_reported, _wing_nodes_reported
    out = {}
    with h5py.File(h5_path, "r") as f:
        st = f["data/structure"]
        jig = st["ini_info/pos"][()]
        ts_keys = _digit_keys(st["timestep_info"])
        deformed = st["timestep_info"][ts_keys[-1]]["pos"][()]   # last entry = converged trim
        beam_number = st["beam_number"][()] if "beam_number" in st else None
        conn = st["connectivities"][()] if "connectivities" in st else None

        steady = unsteady = None
        aero = f["data"].get("aero")
        if aero is not None and "timestep_info" in aero:
            a_keys = _digit_keys(aero["timestep_info"])
            if a_keys:
                ats = aero["timestep_info"][a_keys[-1]]
                if "inertial_steady_forces" in ats and "inertial_unsteady_forces" in ats:
                    steady = ats["inertial_steady_forces"][()]     # [n_surf, 6], G frame
                    unsteady = ats["inertial_unsteady_forces"][()]

    # Tip deflection
    tip = WING_TIP_NODE_INDEX if WING_TIP_NODE_INDEX is not None else \
        int(np.argmax(np.abs(jig[:, SPAN_AXIS_INDEX])))
    if WING_TIP_NODE_INDEX is None and not _tip_node_reported:
        print(f"  [tip] auto-detected wingtip node {tip} at jig "
              f"(x, y, z) = {np.round(jig[tip], 4)} m ({run_id})")
        _tip_node_reported = True
    delta = deformed[tip] - jig[tip]
    out.update(wing_tip_node=tip,
               jig_tip_y=float(jig[tip, SPAN_AXIS_INDEX]),
               tip_deflection_vertical_m=float(delta[VERTICAL_AXIS_INDEX]),
               tip_deflection_total_m=float(np.linalg.norm(delta)),
               tip_deflection_pct_semispan=float(delta[VERTICAL_AXIS_INDEX] / (SPAN_MAIN / 2) * 100))

    # Wing shape (plot 5)
    try:
        if WING_NODE_INDICES is not None:
            nodes = np.array(sorted(WING_NODE_INDICES, key=lambda n: abs(jig[n, SPAN_AXIS_INDEX])))
            method = "WING_NODE_INDICES"
        else:
            nodes, method = _select_wing_nodes(jig, beam_number, conn, tip)
        if WING_NODE_INDICES is None and not _wing_nodes_reported:
            print(f"  [wing shape] {len(nodes)} wing nodes via {method} ({run_id}); "
                  f"root to tip: {nodes.tolist()}")
            _wing_nodes_reported = True
        out.update(span_y_jig=jig[nodes, SPAN_AXIS_INDEX],
                   span_z_jig=jig[nodes, VERTICAL_AXIS_INDEX],
                   span_z_deformed=deformed[nodes, VERTICAL_AXIS_INDEX])
    except Exception as e:
        warnings.warn(f"{run_id}: wing-shape data unavailable ({e})")

    # Wing-only CL (plot 8), same q S reference as the whole-aircraft CL
    out["CL_sharpy_wing_only"] = np.nan
    if steady is not None and max(WING_SURFACES) < steady.shape[0]:
        fz = steady[list(WING_SURFACES), 2].sum() + unsteady[list(WING_SURFACES), 2].sum()
        out["CL_sharpy_wing_only"] = fz / (0.5 * RHO * u_inf**2 * S)
    return out

# ============================================================================
# Main
# ============================================================================

def scan_runs():
    """Parse every matching run folder into one DataFrame (+ wing-shape data for plot 5)."""
    if not os.path.isdir(OUTPUT_ROOT):
        raise SystemExit(f"OUTPUT_ROOT does not exist: {OUTPUT_ROOT}")

    matched, skipped = [], []
    for name in sorted(os.listdir(OUTPUT_ROOT)):
        full = os.path.join(OUTPUT_ROOT, name)
        if not os.path.isdir(full):
            continue
        m = FOLDER_PATTERN.match(name)
        if m:
            matched.append((name, full, float(m.group("velocity")), m.group("run_tag") or "nocamber"))
        else:
            skipped.append(name)
    print(f"Found {len(matched)} run folders; skipped {len(skipped)} non-matching: {skipped}")
    unknown = sorted({t for *_, t in matched if t not in RUN_TAGS})
    if unknown:
        print(f"Run tags not in RUN_TAGS (using DEFAULT_CAMBER {DEFAULT_CAMBER}): {unknown}")

    alpha_l0_cache, rows, wing_shapes = {}, [], {}
    no_forces, no_h5 = [], []   # reported once at the end instead of one warning per run
    for case, run_dir, u_inf, tag in matched:
        info = tag_info(tag)
        M, P = info["camber"]
        a_l0 = alpha_l0_cache.setdefault((M, P), alpha_L0_deg(M, P))

        trim_path = os.path.join(run_dir, "statictrim", "trim_iterations.txt")
        try:
            trim = parse_trim_iterations(trim_path)
        except Exception as e:
            warnings.warn(f"{case}: skipped - can't read trim_iterations.txt ({e})")
            continue

        tol, tol_source = read_trim_tolerances(case)
        converged = (abs(trim["Fx"]) < tol["fx_tolerance"] and abs(trim["Fz"]) < tol["fz_tolerance"]
                     and abs(trim["My"]) < tol["m_tolerance"])
        if not converged:
            warnings.warn(f"{case}: NOT converged (Fx={trim['Fx']:.4f}, Fz={trim['Fz']:.4f}, "
                          f"My={trim['My']:.4f}; tolerances from {tol_source}) - kept but flagged")

        cl = np.nan
        forces_path = find_forces_csv(run_dir, case)
        if forces_path is None:
            no_forces.append(case)
        else:
            cl = parse_forces_csv(forces_path, u_inf)

        sd = {}
        h5_path = find_savedata_h5(run_dir, case)
        if h5_path is None:
            no_h5.append(case)
        else:
            try:
                sd = parse_savedata(h5_path, u_inf, case)
            except Exception as e:
                warnings.warn(f"{case}: failed to read savedata .h5 ({e})")
        if "span_y_jig" in sd:
            wing_shapes[(tag, u_inf)] = {k: sd.pop(k) for k in ("span_y_jig", "span_z_jig", "span_z_deformed")}

        rows.append({
            "run_id": case, "u_inf": u_inf, "run_tag": tag, "run_label": info["label"],
            "variable": info["variable"], "value": info["value"], "M": M, "P": P,
            "alpha_L0_deg": a_l0, "alpha_sharpy_deg": trim["alpha_deg"], "elev_deg": trim["elev_deg"],
            "thrust": trim["thrust"], "Fx_residual": trim["Fx"], "Fz_residual": trim["Fz"],
            "My_residual": trim["My"], "converged": converged,
            "fx_tolerance": tol["fx_tolerance"], "fz_tolerance": tol["fz_tolerance"],
            "m_tolerance": tol["m_tolerance"], "tolerance_source": tol_source,
            "CL_sharpy": cl, "CL_sharpy_wing_only": sd.get("CL_sharpy_wing_only", np.nan),
            "alpha_hand_deg": alpha_hand_deg(u_inf, a_l0), "CL_hand": CL_hand(u_inf),
            "wing_tip_node": sd.get("wing_tip_node", np.nan),
            "jig_tip_y": sd.get("jig_tip_y", np.nan),
            "tip_deflection_vertical_m": sd.get("tip_deflection_vertical_m", np.nan),
            "tip_deflection_total_m": sd.get("tip_deflection_total_m", np.nan),
            "tip_deflection_pct_semispan": sd.get("tip_deflection_pct_semispan", np.nan),
        })

    if no_forces:
        print(f"{len(no_forces)} run(s) without a forces CSV (CL left blank): {no_forces}")
    if no_h5:
        print(f"{len(no_h5)} run(s) without a savedata .h5 (deflection and wing-only CL left "
              f"blank): {no_h5}")
    if not rows:
        raise SystemExit("No runs parsed - check OUTPUT_ROOT and FOLDER_PATTERN.")
    df = pd.DataFrame(rows).sort_values(["run_tag", "u_inf"]).reset_index(drop=True)
    dupes = df[df.duplicated(subset=["u_inf", "run_tag"], keep=False)]
    if not dupes.empty:
        warnings.warn(f"Duplicate (u_inf, run_tag) pairs: {dupes['run_id'].tolist()}")
    return df, wing_shapes, alpha_l0_cache


def main():
    df, wing_shapes, alpha_l0_cache = scan_runs()
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_CSV}")
    print(df[["run_id", "alpha_sharpy_deg", "elev_deg", "thrust", "CL_sharpy", "converged",
              "tolerance_source"]].to_string(index=False))
    if (~df["converged"]).any():
        print(f"\nWARNING: {(~df['converged']).sum()} run(s) not converged - see 'converged'.")

    found = sorted(df["run_tag"].unique())
    run_tags = [t for t in (TAGS_TO_PLOT or found) if t in found]
    missing = [t for t in (TAGS_TO_PLOT or []) if t not in found]
    if missing:
        print(f"TAGS_TO_PLOT entries not found in this sweep: {missing}")
    colour = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(run_tags)}
    label = {t: tag_info(t)["label"] for t in run_tags}
    if len(run_tags) > len(PALETTE):
        print(f"Note: {len(run_tags)} run tags but {len(PALETTE)} colours - colours repeat. "
              f"Use TAGS_TO_PLOT to choose a subset.")
    # Hand-calc curves depend only on camber, so draw one per distinct camber.
    cambers = sorted({tag_info(t)["camber"] for t in run_tags})
    v_fine = np.linspace(df["u_inf"].min() * 0.95, df["u_inf"].max() * 1.05, 200)
    aswing_df = aswing_sweep_frame()

    def runs(tag, sort="u_inf"):
        return df[df["run_tag"] == tag].sort_values(sort)

    # ---- Lift-curve slope summary (console) --------------------------------
    print(f"\n--- CL-alpha straight-line fits (u_inf >= {V_MIN_FIT:g} m/s) ---")
    aswing_fit = None
    if aswing_df is not None:
        a_win = aswing_df[aswing_df["u_inf"] >= V_MIN_FIT][["alpha_deg", "cl"]].dropna()
        aswing_fit = _linear_fit_deg(a_win["alpha_deg"], a_win["cl"])
    if aswing_fit is not None:
        ref_slope, ref_label = np.degrees(aswing_fit[0]), "ASWING trim points"
        print(f"ASWING trim points: dCL/dalpha = {ref_slope:.3f} /rad, alpha0 = {aswing_fit[1]:.2f} deg")
    else:
        ref_slope, ref_label = DCL_DALPHA_ASWING, "ASWING constant"
    for tag in run_tags:
        fit = _linear_fit_deg(*_fit_window(runs(tag))[["alpha_sharpy_deg", "CL_sharpy"]].T.values)
        if fit is not None:
            slope = np.degrees(fit[0])
            print(f"{label[tag]:28s} dCL/dalpha = {slope:.3f} /rad "
                  f"({(slope - ref_slope) / ref_slope * 100:+.1f}% vs {ref_label}), "
                  f"alpha0 = {fit[1]:.2f} deg")

    # ---- Plot 1: alpha vs u_inf ---------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 5.5))
    capped1 = False
    for tag in run_tags:
        sub = runs(tag)
        x, y = sub["u_inf"].to_numpy(), sub["alpha_sharpy_deg"].to_numpy()
        if ALPHA_PLOT_XMAX is not None:
            capped1 |= bool(np.any(y > ALPHA_PLOT_XMAX))
            x, y = _trim_to_cap(x, y, ALPHA_PLOT_XMAX, cap_on="y")
        ax1.plot(x, y, "o-", color=colour[tag], lw=2, ms=6, label=f"SHARPy, {label[tag]}")
        unc = sub[~sub["converged"]]
        if not unc.empty:
            ax1.plot(unc["u_inf"], unc["alpha_sharpy_deg"], "x", color="black", ms=10, mew=2,
                     zorder=5, label="_nolegend_")
    for i, (M, P) in enumerate(cambers):
        x, y = v_fine, alpha_hand_deg(v_fine, alpha_l0_cache[(M, P)])
        if ALPHA_PLOT_XMAX is not None:
            capped1 |= bool(np.any(y > ALPHA_PLOT_XMAX))
            x, y = _trim_to_cap(x, y, ALPHA_PLOT_XMAX, cap_on="y")
        ax1.plot(x, y, HAND_CALC_STYLES[i % 4], color=INK, lw=1.4, alpha=0.6,
                 label=f"hand calc, camber M={M} P={P}")
    if aswing_df is not None:
        a = aswing_df.dropna(subset=["alpha_deg"])
        ax1.plot(a["u_inf"], a["alpha_deg"], **ASWING_SWEEP_KW, label="ASWING trim sweep", zorder=6)
    ax1.axvline(CRUISE_SPEED_ASWING, color=ASWING_COLOR, ls=":", lw=1.4,
                label=f"cruise speed (ASWING), {CRUISE_SPEED_ASWING:g} m/s")
    ax1.set_xlabel("u_inf (m/s)")
    ax1.set_ylabel("alpha (deg)")
    _cap_axis_with_gap(ax1, "y", ALPHA_PLOT_XMAX if capped1 else None, ALPHA_PLOT_GAP)
    _style_axes(ax1, "LLTA trim sweep: alpha vs. velocity", legend_outside=True)
    fig1.tight_layout()
    fig1.savefig(OUT_PLOT_ALPHA_V, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"\nSaved {OUT_PLOT_ALPHA_V}")

    # ---- Plot 2: CL vs u_inf ------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5.5))
    for tag in run_tags:
        sub = runs(tag)
        ax2.plot(sub["u_inf"], sub["CL_sharpy"], "o-", color=colour[tag], lw=2, ms=6,
                 label=f"SHARPy, {label[tag]}")
        unc = sub[~sub["converged"]]
        if not unc.empty:
            ax2.plot(unc["u_inf"], unc["CL_sharpy"], "x", color="black", ms=10, mew=2, zorder=5,
                     label="_nolegend_")
    ax2.plot(v_fine, CL_hand(v_fine), "-", color=INK, lw=1.8, alpha=0.7, label="CL required (L = W)")
    ax2.axvline(CRUISE_SPEED_ASWING, color=ASWING_COLOR, ls=":", lw=1.4,
                label=f"cruise speed (ASWING), {CRUISE_SPEED_ASWING:g} m/s")
    ax2.axhspan(CL_MIN_ASWING, CL_MAX_ASWING, color=ASWING_COLOR, alpha=0.07, lw=0,
                label=f"ASWING usable CL range ({CL_MIN_ASWING:g} to {CL_MAX_ASWING:g})")
    if aswing_df is not None:
        a = aswing_df.dropna(subset=["cl"])
        ax2.plot(a["u_inf"], a["cl"], **ASWING_SWEEP_KW, label="ASWING trim sweep", zorder=6)
    ax2.set_xlabel("u_inf (m/s)")
    ax2.set_ylabel("CL")
    _style_axes(ax2, "LLTA trim sweep: CL vs. velocity", legend_outside=True)
    fig2.tight_layout()
    fig2.savefig(OUT_PLOT_CL_V, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT_PLOT_CL_V}")

    # ---- Plot 3: CL vs alpha -------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(9.5, 5.5))
    all_alpha = df["alpha_sharpy_deg"].to_numpy()
    alpha_fine = np.linspace(min(all_alpha.min(), ALPHA_L0_ASWING_DEG) - 8, all_alpha.max() + 8, 300)
    capped3 = ALPHA_PLOT_XMAX is not None and alpha_fine.max() > ALPHA_PLOT_XMAX

    def _capped(x, y):
        return _trim_to_cap(x, y, ALPHA_PLOT_XMAX, cap_on="x") if capped3 else (x, y)

    for tag in run_tags:
        sub = runs(tag, "alpha_sharpy_deg")
        ax3.plot(*_capped(sub["alpha_sharpy_deg"].to_numpy(), sub["CL_sharpy"].to_numpy()),
                 "o", color=colour[tag], ms=6, label=f"SHARPy, {label[tag]}")
        if SHOW_SHARPY_DATA_FIT:
            fit = _linear_fit_deg(*_fit_window(sub)[["alpha_sharpy_deg", "CL_sharpy"]].T.values)
            if fit is not None:
                ax3.plot(*_capped(alpha_fine, lift_line(alpha_fine, fit[1], fit[0])), ":",
                         color=colour[tag], lw=2.0, label=f"SHARPy fit, {label[tag]}")
    for i, (M, P) in enumerate(cambers):
        ax3.plot(*_capped(alpha_fine, lift_line(alpha_fine, alpha_l0_cache[(M, P)],
                                                CL_ALPHA * np.pi / 180)),
                 HAND_CALC_STYLES[i % 4], color=INK, lw=1.4, alpha=0.6,
                 label=f"hand calc (wing alone), camber M={M} P={P}")
    ax3.plot(*_capped(alpha_fine, lift_line(alpha_fine, ALPHA_L0_ASWING_DEG,
                                            DCL_DALPHA_ASWING * np.pi / 180)),
             "-", color=ASWING_COLOR, lw=2, label=f"ASWING constants, dCL/dα = {DCL_DALPHA_ASWING:g}/rad")
    ax3.axhspan(CL_MIN_ASWING, CL_MAX_ASWING, color=ASWING_COLOR, alpha=0.07, lw=0,
                label=f"ASWING usable CL range ({CL_MIN_ASWING:g} to {CL_MAX_ASWING:g})")
    if aswing_df is not None:
        a = aswing_df.dropna(subset=["alpha_deg", "cl"])
        ax3.plot(*_capped(a["alpha_deg"].to_numpy(), a["cl"].to_numpy()), **ASWING_SWEEP_KW,
                 label="ASWING trim sweep", zorder=6)
    ax3.set_xlabel("alpha (deg)")
    ax3.set_ylabel("CL")
    _cap_axis_with_gap(ax3, "x", ALPHA_PLOT_XMAX if capped3 else None, ALPHA_PLOT_GAP)
    _style_axes(ax3, f"LLTA: CL vs. alpha (fits use u_inf >= {V_MIN_FIT:g} m/s)", legend_outside=True)
    fig3.tight_layout()
    fig3.savefig(OUT_PLOT_CL_ALPHA, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT_PLOT_CL_ALPHA}")

    # ---- Plot 4: tip deflection vs u_inf ------------------------------------
    if df["tip_deflection_vertical_m"].notna().any():
        fig4, ax4 = plt.subplots(figsize=(8, 5.5))
        for tag in run_tags:
            sub = runs(tag)
            ax4.plot(sub["u_inf"], sub["tip_deflection_vertical_m"], "o-", color=colour[tag], lw=2,
                     ms=6, label=label[tag])
            if SHOW_TOTAL_TIP_DEFLECTION:
                ax4.plot(sub["u_inf"], sub["tip_deflection_total_m"], "^--", color=colour[tag],
                         lw=1.4, alpha=0.6, ms=5, label=f"{label[tag]}, 3D magnitude")
            unc = sub[~sub["converged"]]
            if not unc.empty:
                ax4.plot(unc["u_inf"], unc["tip_deflection_vertical_m"], "x", color="black",
                         ms=10, mew=2, zorder=5, label="_nolegend_")
        ax4.axhline(0, color=GRID, lw=1.0)
        ax4.set_xlabel("u_inf (m/s)")
        ax4.set_ylabel("wingtip vertical deflection from jig shape (m)")
        _style_axes(ax4, "LLTA trim sweep: wingtip deflection vs. velocity", legend_outside=True)
        fig4.tight_layout()
        fig4.savefig(OUT_PLOT_TIP_DEFL, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"Saved {OUT_PLOT_TIP_DEFL}")
    else:
        print(f"No tip-deflection data - skipped {OUT_PLOT_TIP_DEFL} (needs SaveData, save_struct).")

    # ---- Plot 5: wing shape vs span, one panel per run tag -------------------
    shape_tags = [t for t in run_tags if any(c == t for (c, _u) in wing_shapes)]
    if shape_tags:
        fig5, axes5 = plt.subplots(len(shape_tags), 1, figsize=(9, 3.2 * len(shape_tags)),
                                   sharex=True, sharey=True, squeeze=False, constrained_layout=True)
        axes5 = axes5[:, 0]
        norm = plt.Normalize(vmin=min(u for _c, u in wing_shapes), vmax=max(u for _c, u in wing_shapes))
        x_scale = (SPAN_MAIN / 2) if NORMALIZE_WING_SHAPE_SPAN else 1.0
        for ax, tag in zip(axes5, shape_tags):
            entries = {u: d for (c, u), d in wing_shapes.items() if c == tag}
            u_list = sorted(entries)
            ref = entries[u_list[0]]
            if any(not np.allclose(entries[u]["span_z_jig"], ref["span_z_jig"], atol=1e-6) for u in u_list):
                warnings.warn(f"{tag}: jig shape differs between speeds - check the meshes match")
            ax.plot(np.abs(ref["span_y_jig"]) / x_scale, ref["span_z_jig"], "--", color=INK, lw=1.0,
                    label="jig shape", zorder=5)
            for u in u_list:
                ax.plot(np.abs(entries[u]["span_y_jig"]) / x_scale, entries[u]["span_z_deformed"], "-",
                        color=SEQUENTIAL_BLUE(norm(u)), lw=0.9, marker="o", ms=2)
            x_max = (WING_SHAPE_SPAN_AXIS_MAX if (not NORMALIZE_WING_SHAPE_SPAN and WING_SHAPE_SPAN_AXIS_MAX)
                     else (SPAN_MAIN / 2) / x_scale)
            ax.set_xlim(0, x_max)
            _ensure_boundary_tick(ax, "x", x_max)
            if WING_SHAPE_Z_AXIS_MAX is not None:
                ax.set_ylim(top=max(ax.get_ylim()[1], WING_SHAPE_Z_AXIS_MAX))
                _ensure_boundary_tick(ax, "y", WING_SHAPE_Z_AXIS_MAX)
            if not NORMALIZE_WING_SHAPE_SPAN:
                ax.set_aspect("equal", adjustable="box")   # true proportions when both axes in m
            _style_axes(ax, label[tag])
            ax.set_ylabel("z (m)")
        axes5[-1].set_xlabel("normalised span, |y| / semispan" if NORMALIZE_WING_SHAPE_SPAN
                             else "distance from root, |y| (m)")
        sm = plt.cm.ScalarMappable(norm=norm, cmap=SEQUENTIAL_BLUE)
        sm.set_array([])
        fig5.colorbar(sm, ax=list(axes5), pad=0.02).set_label("u_inf (m/s)")
        fig5.suptitle("LLTA: deflected wing shape vs. jig shape, by run tag", x=0.01, ha="left",
                      fontsize=12, color=INK)
        fig5.savefig(OUT_PLOT_WING_SHAPE, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"Saved {OUT_PLOT_WING_SHAPE}")
    else:
        print(f"No wing-shape data - skipped {OUT_PLOT_WING_SHAPE} (needs SaveData, save_struct).")

    # ---- Plots 6 and 7: elevator and thrust vs u_inf --------------------------
    for col, ylab, title, out in (
            ("elev_deg", "elevator deflection (deg)", "elevator", OUT_PLOT_ELEV_V),
            ("thrust", "thrust (N)", "thrust", OUT_PLOT_THRUST_V)):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for tag in run_tags:
            sub = runs(tag)
            ax.plot(sub["u_inf"], sub[col], "o-", color=colour[tag], lw=2, ms=6,
                    label=f"SHARPy, {label[tag]}")
        if aswing_df is not None and col in aswing_df:
            a = aswing_df.dropna(subset=[col])
            ax.plot(a["u_inf"], a[col], **ASWING_SWEEP_KW, label="ASWING trim sweep", zorder=6)
        ax.axvline(CRUISE_SPEED_ASWING, color=ASWING_COLOR, ls=":", lw=1.4,
                   label=f"cruise speed (ASWING), {CRUISE_SPEED_ASWING:g} m/s")
        ax.set_xlabel("u_inf (m/s)")
        ax.set_ylabel(ylab)
        _style_axes(ax, f"LLTA trim sweep: {title} vs. velocity, SHARPy vs. ASWING", legend_outside=True)
        fig.tight_layout()
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"Saved {out}")

    # ---- Plot 8: whole-aircraft vs wing-only CL-alpha fit --------------------
    # The whole-aircraft fit includes the tail's trim-varying lift; the wing-only fit (wing
    # surfaces only, from the savedata .h5) is the one to compare with wing-alone zero-lift angles.
    if df["CL_sharpy_wing_only"].notna().any():
        fig8, ax8 = plt.subplots(figsize=(9.5, 5.5))
        print(f"\nWhole-aircraft vs wing-only CL-alpha fits (u_inf >= {V_MIN_FIT:g} m/s):")
        for tag in run_tags:
            sub = runs(tag, "alpha_sharpy_deg")
            win = _fit_window(sub)
            for col, mk, ls, kind in (("CL_sharpy", "o", ":", "whole aircraft"),
                                      ("CL_sharpy_wing_only", "s", "--", "wing only")):
                pts = sub.dropna(subset=[col])
                if pts.empty:
                    continue
                ax8.plot(pts["alpha_sharpy_deg"], pts[col], mk, color=colour[tag], ms=6,
                         mfc=colour[tag] if mk == "o" else "none", mew=1.6, label=f"{kind}, {label[tag]}")
                fit = _linear_fit_deg(win["alpha_sharpy_deg"], win[col])
                if fit is not None:
                    ax8.plot(alpha_fine, lift_line(alpha_fine, fit[1], fit[0]), ls, color=colour[tag],
                             lw=2.0, label=f"{kind} fit, {label[tag]}")
                    print(f"  {label[tag]:28s} {kind:15s} dCL/dalpha = {np.degrees(fit[0]):6.3f} /rad, "
                          f"alpha0 = {fit[1]:7.3f} deg")
        ax8.axvline(ALPHA_L0_ASWING_DEG, color=ASWING_COLOR, ls=":", lw=1.2,
                    label=f"ASWING zero-lift angle, {ALPHA_L0_ASWING_DEG:g} deg")
        ax8.axhline(0, color=GRID, lw=1.0)
        ax8.set_xlabel("alpha_sharpy (deg)")
        ax8.set_ylabel("CL")
        _style_axes(ax8, "LLTA: whole-aircraft vs. wing-only SHARPy fit, per run tag", legend_outside=True)
        fig8.tight_layout()
        fig8.savefig(OUT_PLOT_WING_VS_TOTAL, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"Saved {OUT_PLOT_WING_VS_TOTAL}")
    else:
        print(f"No wing-only CL data - skipped {OUT_PLOT_WING_VS_TOTAL} (needs SaveData, save_aero).")

    # ---- Plot 9: alpha vs 1/u_inf^2 ---------------------------------------------
    # alpha = alpha_0 + k x, x = 1/V^2, k = CL V^2 / a. Apparent lift-curve slope a = mean(CL V^2)/k,
    # using each dataset's own CL, so reference-area and weight differences cancel.
    def fit_inv_v2(v, alpha, cl):
        v, alpha, cl = (np.asarray(t, dtype=float) for t in (v, alpha, cl))
        ok = (v >= V_MIN_FIT) & np.isfinite(alpha)
        if ok.sum() < 2:
            return None
        x = 1.0 / v[ok] ** 2
        k, a0 = np.polyfit(x, alpha[ok], 1)
        clv2 = np.nanmean(cl[ok] * v[ok] ** 2)
        return dict(a0=a0, k=k, x=x, resid=alpha[ok] - (a0 + k * x), clv2=clv2,
                    a=clv2 / np.radians(k), n=int(ok.sum()))

    fig9, (ax9, ax9r) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw=dict(width_ratios=[1.6, 1]))
    fits9, x_max9 = {}, 0.0
    print(f"\n--- alpha = alpha_0 + k / V^2 fits (u_inf >= {V_MIN_FIT:g} m/s) ---")
    print(f"{'series':30s} {'n':>2s} {'alpha_0 (deg)':>13s} {'k (deg m2/s2)':>14s} "
          f"{'CL*V^2':>8s} {'a (/rad)':>9s} {'rms resid':>9s}")
    series = [(f"SHARPy {label[t]}", runs(t), "u_inf", "alpha_sharpy_deg", "CL_sharpy", colour[t])
              for t in run_tags]
    if aswing_df is not None:
        series.append(("ASWING", aswing_df, "u_inf", "alpha_deg", "cl", ASWING_COLOR))
    for name, data, vc, ac, cc, col in series:
        v, al = data[vc].to_numpy(), data[ac].to_numpy()
        x = 1.0 / v ** 2
        x_max9 = max(x_max9, x.max())
        used = v >= V_MIN_FIT
        is_asw = name == "ASWING"
        kw = dict(ASWING_SWEEP_KW) if is_asw else dict(marker="o", ls="none", ms=7, color=col)
        ax9.plot(x[used], al[used], **kw, label=name)
        if (~used).any():
            ax9.plot(x[~used], al[~used], **{**kw, "mfc": "none", "mew": 1.5}, label="_nolegend_")
        fit = fit_inv_v2(v, al, data[cc].to_numpy())
        if fit is None:
            continue
        fits9[name] = fit
        xx = np.linspace(0, x.max() * 1.05, 50)
        ax9.plot(xx, fit["a0"] + fit["k"] * xx, "--" if is_asw else "-", color=col, lw=1.3,
                 alpha=0.85, label="_nolegend_")
        ax9r.plot(fit["x"], fit["resid"], marker=ASWING_SWEEP_MARKER if is_asw else "o",
                  ls="--" if is_asw else "-", color=col, mfc="none" if is_asw else col, ms=6,
                  lw=1.1, label=name)
        print(f"{name:30s} {fit['n']:2d} {fit['a0']:13.2f} {fit['k']:14.1f} {fit['clv2']:8.2f} "
              f"{fit['a']:9.2f} {np.sqrt(np.mean(fit['resid'] ** 2)):9.3f}")
    if "ASWING" in fits9:
        ref = fits9["ASWING"]
        print("Relative to ASWING:")
        for name, fit in fits9.items():
            if name != "ASWING":
                print(f"  {name:30s} d(alpha_0) = {fit['a0'] - ref['a0']:+6.2f} deg, "
                      f"k/k_ASWING = {fit['k'] / ref['k']:.3f}")
    for i, (M, P) in enumerate(cambers):
        xx = np.linspace(1.0 / (df["u_inf"].max() * 1.05) ** 2, x_max9 * 1.05, 50)
        ax9.plot(xx, alpha_hand_deg(1.0 / np.sqrt(xx), alpha_l0_cache[(M, P)]), HAND_CALC_STYLES[i % 4],
                 color=INK, lw=1.4, alpha=0.5, label=f"hand calc, camber M={M} P={P}")
    ax9.set_xlim(left=0)
    sec = ax9.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    v_ticks = [v for v in (5, 6, 7, 8, 9, 10, 15) if 1 / v ** 2 <= ax9.get_xlim()[1]]
    sec.set_xticks([1 / v ** 2 for v in v_ticks])
    sec.set_xticklabels([f"{v:g}" for v in v_ticks])
    sec.set_xlabel("u_inf (m/s)", color=MUTED)
    ax9.axhline(0, color=GRID, lw=1.0)
    ax9.set_xlabel("1 / u_inf^2  (s^2/m^2)")
    ax9.set_ylabel("trim alpha (deg)")
    _style_axes(ax9, "Trim alpha vs 1/u_inf^2 (hollow = excluded from fit; intercept = alpha_0)")
    ax9r.axhline(0, color=GRID, lw=1.0)
    ax9r.set_xlabel("1 / u_inf^2  (s^2/m^2)")
    ax9r.set_ylabel("alpha minus straight-line fit (deg)")
    _style_axes(ax9r, "Residuals from each straight-line fit")
    fig9.tight_layout()
    fig9.savefig(OUT_PLOT_ALPHA_INV_V2, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT_PLOT_ALPHA_INV_V2}")


if __name__ == "__main__":
    main()