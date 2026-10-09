"""
Convert a SHARPy CG position into %MAC, so it can be compared directly against
a %MAC figure quoted by Voltitude (or any other source that specifies CG that
way, which is the near-universal convention for stability/trim work).

No command-line flags -- edit the settings below, then run:

    python compute_percent_mac.py

Must live in the same folder as extract_surface_loads.py.

-----------------------------------------------------------------------------
THE MATH
-----------------------------------------------------------------------------
%MAC is defined purely in the x (chordwise/streamwise) direction, relative to
the wing's Mean Aerodynamic Chord (MAC):

    %MAC = (x_cg - x_LE,MAC) / MAC * 100

where MAC is the length of the mean aerodynamic chord and x_LE,MAC is the
x-location of its leading edge, both in the same aircraft x-datum as x_cg.
0% = MAC leading edge, 100% = MAC trailing edge; conventional aircraft
typically sit somewhere around 20-35% MAC.

For a single straight-tapered wing panel (constant taper ratio, constant
sweep from root to tip) there's a closed-form handbook result:

    lambda   = c_tip / c_root                              (taper ratio)
    MAC      = (2/3) * c_root * (1 + lambda + lambda^2) / (1 + lambda)
    y_MAC    = (b_semi/3) * (1 + 2*lambda) / (1 + lambda)   (spanwise station
                                                              of the MAC,
                                                              measured from
                                                              the root along
                                                              the span)
    x_LE,MAC = x_LE,root + y_MAC * tan(sweep_LE)

This is what MODE = "simple" below uses.

If the real planform isn't a single clean trapezoid (e.g. LLTA has more than
one taper break, or you just want to feed in the actual chord/x_LE stations
straight out of your geometry-generation script instead of re-deriving root
chord/tip chord/sweep by hand), the general strip-integration form handles
any planform:

    MAC      = Integral[ c(y)^2 dy ] / Integral[ c(y) dy ]   over one semispan
    x_LE,MAC = Integral[ c(y) * x_LE(y) dy ] / Integral[ c(y) dy ]

This is what MODE = "stations" below uses (numerical trapezoidal integration
over whatever spanwise stations you give it -- works for multi-panel wings
too, since it doesn't assume a single taper/sweep).

x_cg itself comes from the SAME place as your elevator-derivative script
already got it: the CG saved by a 'Modal' step in a .data.h5 file (via
get_sharpy_modal_cg / resolve_ref_pos, with the hand-buildup cross-check as
a fallback) -- so this reuses a value you've already validated, rather than
recomputing it.
"""
import numpy as np
from extract_surface_loads import resolve_ref_pos, get_wing_le_root_x
import llta_geometry as llta

# =============================================================================
# EDIT THESE SETTINGS, THEN RUN THIS FILE
# =============================================================================

# --- Where does x_cg come from? ---------------------------------------------
# Point this at any .data.h5 that has a saved 'Modal' step (same file you used
# for REF_POS in run_derivative.py works fine). Leave as None and fill in
# X_CG_MANUAL instead if you'd rather type the number in directly.
H5_FILE = "/Users/icc25/sharpy_cases/LLTA/output/llta_model_audit/savedata/llta_model_audit.data.h5"
X_CG_MANUAL = None   # e.g. 0.23810025 -- used only if H5_FILE is None

# --- Wing planform: pick ONE mode -------------------------------------------
MODE = "simple"   # "simple" (single trapezoid) or "stations" (general planform)

# MODE = "simple": fill these in (all in metres/degrees, same x-datum as x_cg)
ROOT_CHORD   = 0.225   # e.g. 0.35
TIP_CHORD    = 0.225   # e.g. 0.20
SEMISPAN     = 1.875   # e.g. 3.0   -- one wing's span, root to tip
SWEEP_LE_DEG = 0.0     # leading-edge sweep, positive = swept back
X_LE_ROOT    = -llta.ea_main * llta.chord_main     # x-location of the ROOT leading edge in your aircraft datum
                                                   # (only used if WING_H5_FILE/WING_SURF below are left as None --
                                                   # set those instead to have this read straight out of your model)

# Optional: instead of typing X_LE_ROOT by hand, pull it straight out of a
# .data.h5 file's own aero grid (guaranteed to be in the same axes as x_cg).
# Leave both as None to just use the X_LE_ROOT value above.
WING_H5_FILE = None   # e.g. same file as H5_FILE above, or any run with this wing meshed
WING_SURF    = 0   # e.g. 0 -- the i_surf index for one of your wing halves

# MODE = "stations": fill these in instead (arrays, root at y=0 to tip at y=semispan,
# one side of the aircraft only -- pull straight from your geometry script)
Y_STATIONS = []   # e.g. [0.0, 1.0, 2.0, 3.0]
CHORDS     = []   # matching chord length at each station
X_LE       = []   # matching LE x-location at each station (same x-datum as x_cg)

# --- Optional: what has Voltitude quoted? -----------------------------------
VOLTITUDE_PERCENT_MAC = None   # e.g. 28.0 -- leave None to skip the comparison

# =============================================================================
# Nothing below this line needs editing.
# =============================================================================


def mac_trapezoidal(root_chord, tip_chord, semispan, sweep_le_deg=0.0, x_le_root=0.0):
    """Closed-form MAC for a single straight-tapered wing panel.

    Returns (mac, x_le_mac, y_mac): MAC length, x-location of its leading
    edge, and the spanwise station (from root, along the span) it sits at.
    """
    lam = tip_chord / root_chord
    mac = (2.0 / 3.0) * root_chord * (1 + lam + lam ** 2) / (1 + lam)
    y_mac = (semispan / 3.0) * (1 + 2 * lam) / (1 + lam)
    x_le_mac = x_le_root + y_mac * np.tan(np.radians(sweep_le_deg))
    return mac, x_le_mac, y_mac


def mac_from_stations(y, chord, x_le):
    """General strip-integration MAC for an arbitrary (possibly multi-panel)
    planform, given spanwise stations on ONE side of the aircraft.

    Returns (mac, x_le_mac).
    """
    y = np.asarray(y, dtype=float)
    chord = np.asarray(chord, dtype=float)
    x_le = np.asarray(x_le, dtype=float)
    order = np.argsort(y)
    y, chord, x_le = y[order], chord[order], x_le[order]

    denom = np.trapz(chord, y)
    mac = np.trapz(chord ** 2, y) / denom
    x_le_mac = np.trapz(chord * x_le, y) / denom
    return mac, x_le_mac


def percent_mac(x_cg, mac, x_le_mac):
    return (x_cg - x_le_mac) / mac * 100.0


if __name__ == '__main__':
    # --- resolve x_cg --------------------------------------------------
    if H5_FILE is not None and not H5_FILE.startswith('/path/to/'):
        cg = resolve_ref_pos(H5_FILE, user_ref=None)
        x_cg = cg[0]
    elif X_CG_MANUAL is not None:
        x_cg = X_CG_MANUAL
        print(f"Using manually-entered x_cg = {x_cg}")
    else:
        raise SystemExit(
            "Set either H5_FILE (to a real path) or X_CG_MANUAL at the top of "
            "this file first."
        )

    # --- resolve x_le_root, if requested from the model itself ------------
    if WING_H5_FILE is not None and WING_SURF is not None:
        print()
        x_le_root = get_wing_le_root_x(WING_H5_FILE, WING_SURF)
        print()
    else:
        x_le_root = X_LE_ROOT

    # --- resolve MAC + its LE location -----------------------------------
    if MODE == "simple":
        missing = [name for name, val in
                   [("ROOT_CHORD", ROOT_CHORD), ("TIP_CHORD", TIP_CHORD), ("SEMISPAN", SEMISPAN)]
                   if val is None]
        if missing:
            raise SystemExit(f"MODE='simple' needs {missing} filled in at the top of this file.")
        mac, x_le_mac, y_mac = mac_trapezoidal(ROOT_CHORD, TIP_CHORD, SEMISPAN,
                                                SWEEP_LE_DEG, x_le_root)
        print(f"Wing planform: single trapezoid, taper = {TIP_CHORD / ROOT_CHORD:.3f}")
        print(f"MAC spanwise station (from root): {y_mac:.4f} m")

    elif MODE == "stations":
        if not (len(Y_STATIONS) >= 2 and len(CHORDS) == len(Y_STATIONS) and len(X_LE) == len(Y_STATIONS)):
            raise SystemExit(
                "MODE='stations' needs Y_STATIONS, CHORDS and X_LE filled in at the top of this "
                "file, all the same length, with at least 2 stations."
            )
        mac, x_le_mac = mac_from_stations(Y_STATIONS, CHORDS, X_LE)
        print(f"Wing planform: {len(Y_STATIONS)}-station numerical integration")

    else:
        raise SystemExit(f"MODE must be 'simple' or 'stations', got {MODE!r}")

    # --- report -----------------------------------------------------------
    pct = percent_mac(x_cg, mac, x_le_mac)

    print()
    print(f"MAC              = {mac:.4f} m")
    print(f"x_LE,MAC         = {x_le_mac:.4f} m")
    print(f"x_cg             = {x_cg:.4f} m")
    print(f"CG position      = {pct:.2f} %MAC")

    if VOLTITUDE_PERCENT_MAC is not None:
        diff = pct - VOLTITUDE_PERCENT_MAC
        print()
        print(f"Voltitude quotes = {VOLTITUDE_PERCENT_MAC:.2f} %MAC")
        print(f"Difference       = {diff:+.2f} %MAC  (SHARPy {'aft' if diff > 0 else 'forward'} of Voltitude)")