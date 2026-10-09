"""
LLTA spanwise lift distribution - SHARPy vs. ASWING (9 m/s trim point)
Objective: investigate the higher SHARPy trim alpha (vs. ASWING) at a fixed u_inf by comparing
HOW the lift is distributed along the span, not just its total.

Reads, per SHARPy run (savedata .data.h5, needs SaveData save_aero: True and save_struct: True):
    data/aero/timestep_info/<last>/forces    per surface [6, M+1, N+1], G frame, at panel vertices
    data/aero/timestep_info/<last>/zeta      per surface [3, M+1, N+1], G frame, vertex positions
    data/aero/timestep_info/<last>/inertial_steady_forces   [n_surf, 6] (closure check only)
    data/structure/timestep_info/<last>/gravity_forces       [n_node, 6] (weight incl. lumped masses)
    data/structure/timestep_info/<last>/pos, beam_number, connectivities
and the ASWING output file (Voltitude's Op.Point 1, 9 m/s): Main_Wing / Elevator_1 beam tables
(y, cl, Fn, Mc', Ms'), with the exponent row above each header applied (true = printed x 10^n).

Plots (2 PNGs) + 1 CSV:
    llta_lift_distribution.png      2x2:
        (a) vertical aero force at every node [N]  - wing and tail, primary run (what was asked for)
        (b) lift per unit span l(y) [N/m]          - nodal force / tributary width; ASWING = q c cl
        (c) l(y) / l_mean                          - shape only ("same CL" comparison), + elliptic
        (d) local cl(y)                             - wing and tail; twist/incidence shows up here
    llta_wing_shear_bending.png     1x2: shear force and bending moment from the tip inwards -
                                    SHARPy aero-only, gravity-only and NET (primary run), net for
                                    the others; ASWING Fn / Mc' (internal beam loads = compare to NET)
    llta_lift_distribution.csv      the spanwise table for every run

Why the nodal force is converted to N/m: SHARPy's UVLM forces sit on panel vertices, so the force
at a node scales with the local node spacing. Spacing changes (e.g. at the 50 % semispan kink) make
raw nodal forces jump even when the loading is smooth. l(y) = F_node / tributary width removes that.

Why gravity is taken as -|F|: gravity at a node is a purely vertical force, so its vertical (G frame)
component is minus its magnitude in whatever frame SHARPy stored it (A or B). This avoids depending
on the storage frame of gravity_forces. The inventory/closure printout checks it against W.

Usage:
    python lift_distribution.py              # uses the CONFIG block below
    python lift_distribution.py --inventory  # just print the h5 contents/shapes of the primary run
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

_trapz = getattr(np, "trapezoid", None) or np.trapz

# ============================================================================
# CONFIG - edit this block
# ============================================================================

OUTPUT_ROOT = "/Users/icc25/sharpy_cases/llta/output"

# Run folders to plot. The FIRST one is the primary run (gets the per-node, aero/gravity/net
# breakdown); the rest are overlaid as lines only. u_inf is parsed from the folder name.
RUNS = [
    #"llta_trim_u_inf_9_camber_chorddiscretisation16",   # m = 16, wake rule  (primary)
    #"llta_trim_u_inf_9_camber_wakelength",              # m = 4,  wake rule
    # "llta_trim_u_inf_5_camber_chorddiscretisation16",
    # "llta_trim_u_inf_10_camber_chorddiscretisation16",
    "llta_trim_u_inf_9__baseline",
    "llta_trim_u_inf_9__wing_incidence"
]
RUN_LABELS = {   # optional pretty labels; falls back to the folder name
    #"llta_trim_u_inf_9_camber_chorddiscretisation16": "SHARPy m=16, 9 m/s",
    #"llta_trim_u_inf_9_camber_wakelength": "SHARPy m=4, 9 m/s",
    "llta_trim_u_inf_9__baseline": "wing incidence = 0 deg",
    "llta_trim_u_inf_9__wing_incidence": "wing incidence = -3 deg",
    "llta_trim_u_inf_9__control_deflection": "control surface incidence = -3 deg",
}

ASWING_FILE = "v9ms"            # Voltitude's 9 m/s ASWING output (same folder as this script)
ASWING_WING_BEAM = "Main_Wing"
ASWING_TAIL_BEAM = "Elevator_1"
ASWING_CHORD_WING = 0.225       # m - ASWING Ref.Chrd. Checked: q*c*int(cl dy) = 12.33 N per side
                                #     vs 24.9/2 = 12.45 N from the ASWING lift split (1 %).

RHO = 1.225
G = 9.81
WING_SURFACES = (0, 1)          # i_surf: 0/1 main wing, 2 fin, 3/4 tailplane (generate_aero_file)
TAIL_SURFACES = (3, 4)
Y_ZERO_TOL = 1e-6               # |y| below this counts as the root/centreline

OUT_PLOT_DIST = "llta_lift_distribution.png"
OUT_PLOT_SHEAR = "llta_wing_shear_bending.png"
OUT_CSV = "llta_lift_distribution.csv"

PALETTE = ["#2b6cb0", "#c0392b", "#2e7d32", "#8e44ad"]   # same order as trim_sweep_analysis.py
ASWING_COLOR = "#444444"
GRID = "#dddddd"

FOLDER_U = re.compile(r"llta_trim_u_?inf_(\d+(?:\.\d+)?)")

# ============================================================================
# Readers
# ============================================================================

def _digit_keys(grp):
    """SaveData/h5utils writes a '_read_as' bookkeeping entry into every group - keep only the
    '%05d' keys (same gotcha as in trim_sweep_analysis.py)."""
    return sorted((k for k in grp.keys() if k.isdigit()), key=int)


def _as_list(node):
    """Per-surface lists are saved either as one stacked dataset or as a group of '%05d'
    datasets (same helper as tail_inflow_check.py)."""
    if isinstance(node, h5py.Dataset):
        return list(node[()])
    return [node[k][()] for k in _digit_keys(node)]


def inventory(h5_path):
    print(f"\n--- h5 inventory: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        for grp_path in ("data/aero/timestep_info", "data/structure/timestep_info"):
            if grp_path not in f:
                print(f"  {grp_path}: MISSING")
                continue
            keys = _digit_keys(f[grp_path])
            print(f"  {grp_path}: {len(keys)} timestep(s), using '{keys[-1]}'")
            ts = f[grp_path][keys[-1]]
            for name in sorted(ts.keys()):
                obj = ts[name]
                if isinstance(obj, h5py.Dataset):
                    print(f"    {name:28s} dataset {obj.shape}")
                else:
                    sub = _digit_keys(obj)
                    shp = obj[sub[0]].shape if sub else "-"
                    print(f"    {name:28s} group of {len(sub)} x {shp}")
        st = f["data/structure"]
        for name in ("beam_number", "connectivities", "num_node", "lumped_mass", "lumped_mass_nodes"):
            if name in st:
                print(f"  data/structure/{name}: {st[name].shape}")


def read_sharpy(h5_path):
    with h5py.File(h5_path, "r") as f:
        a_grp = f["data/aero/timestep_info"]
        ats = a_grp[_digit_keys(a_grp)[-1]]
        zeta = _as_list(ats["zeta"])
        forces = _as_list(ats["forces"])
        isf = ats["inertial_steady_forces"][()] if "inertial_steady_forces" in ats else None

        st = f["data/structure"]
        sts = st["timestep_info"][_digit_keys(st["timestep_info"])[-1]]
        pos = sts["pos"][()]
        grav = sts["gravity_forces"][()] if "gravity_forces" in sts else None
        beam_number = st["beam_number"][()] if "beam_number" in st else None
        conn = st["connectivities"][()] if "connectivities" in st else None
    return dict(zeta=zeta, forces=forces, isf=isf, pos=pos, grav=grav,
                beam_number=beam_number, conn=conn)


def _starboard(t):
    """Rows of the starboard half of a symmetric ASWING beam table. The root appears twice
    (t = -0.00 for the port end, then t = 0.00 for the starboard end, with opposite-signed Fn/Ms),
    and '-0.00' parses as 0.0, so select from the LAST t == 0 row onwards rather than t >= 0."""
    n = len(t)
    zero = np.nonzero(np.abs(t) < 1e-9)[0]
    start = zero[-1] if len(zero) else int(np.argmax(t > 0))
    return np.arange(n) >= start


def parse_aswing_beams(path):
    """Beam tables from an ASWING operating-point printout. Applies the exponent row
    (true value = printed x 10^n) to Mc', Ms', Fn."""
    lines = open(path).read().splitlines()
    beams, header = {}, {}
    for ln in lines:
        m = re.search(r"Dyn\.Pr\.:\s*([\d.Ee+-]+)", ln)
        if m:
            header["q"] = float(m.group(1))
        m = re.search(r"Alpha\s*:\s*([-\d.]+)", ln)
        if m:
            header["alpha"] = float(m.group(1))
    i = 0
    while i < len(lines):
        m = re.match(r"\s*(Surface|Fuselage)\s+Beam\s+\d+:\s+(\S+)", lines[i])
        if not m:
            i += 1
            continue
        exps = [int(v) for v in lines[i + 1].split()]
        hdr = lines[i + 2].split()
        rows, j = [], i + 4
        while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith("..."):
            rows.append([float(v) for v in lines[j].split()])
            j += 1
        arr = np.array(rows)
        d = {h: arr[:, k] for k, h in enumerate(hdr)}
        d["Mc"] = d.pop("Mc'/10") * 10.0 ** exps[0]
        d["Ms"] = d.pop("Ms'/10") * 10.0 ** exps[1]
        d["Fn"] = d.pop("Fn/10") * 10.0 ** exps[2]
        beams[m.group(2)] = d
        i = j
    return beams, header

# ============================================================================
# Spanwise processing
# ============================================================================

def tributary_width(y_abs):
    """Half the distance to each neighbour (half-panel at the ends), in the order given."""
    order = np.argsort(y_abs)
    ys = y_abs[order]
    edges = np.concatenate(([ys[0]], 0.5 * (ys[1:] + ys[:-1]), [ys[-1]]))
    w_sorted = np.diff(edges)
    w = np.empty_like(w_sorted)
    w[order] = w_sorted
    return w


def surface_strips(zeta_s, forces_s, q):
    """Per spanwise vertex line of one surface: |y|, vertical force, chord, l(y), cl(y)."""
    fz = forces_s[2].sum(axis=0)                      # sum chordwise -> [N+1]
    y = zeta_s[1].mean(axis=0)
    chord = np.linalg.norm(zeta_s[[0, 2], -1, :] - zeta_s[[0, 2], 0, :], axis=0)   # LE->TE, x-z
    y_abs = np.abs(y)
    w = tributary_width(y_abs)
    l = np.divide(fz, w, out=np.full_like(fz, np.nan), where=w > 0)
    cl = l / (q * chord)
    order = np.argsort(y_abs)
    return {k: v[order] for k, v in dict(y=y, y_abs=y_abs, fz=fz, chord=chord, w=w, l=l, cl=cl).items()}


def wing_side_nodes(pos, beam_number, conn, sign):
    """Structural nodes on one wing half (excluding the centreline node 0 / root). Uses the beam
    that contains the outermost node on that side, as in trim_sweep_analysis.py."""
    y = pos[:, 1]
    tip = int(np.argmax(sign * y))
    if beam_number is not None and conn is not None:
        elem = np.nonzero(np.any(conn == tip, axis=1))[0][0]
        b = beam_number[elem]
        nodes = np.unique(conn[beam_number == b].ravel())
    else:
        warnings.warn("beam_number/connectivities missing - selecting wing nodes by sign of y only "
                      "(may pick up tail nodes; check the printed wing mass)")
        nodes = np.nonzero(sign * y > Y_ZERO_TOL)[0]
    return nodes[sign * y[nodes] > Y_ZERO_TOL]


def shear_bending(y_load, f_load, stations):
    """Shear S(y0) = sum of loads outboard of y0; bending M(y0) = sum F (y - y0).
    A load exactly at an interior station counts half (it straddles the cut); at the innermost
    load position (the root) it counts in full, since its whole tributary strip is outboard."""
    S, M = np.zeros_like(stations), np.zeros_like(stations)
    y_in = y_load.min()
    for k, y0 in enumerate(stations):
        out = y_load > y0 + 1e-9
        on = np.abs(y_load - y0) <= 1e-9
        S[k] = f_load[out].sum() + (1.0 if abs(y0 - y_in) <= 1e-9 else 0.5) * f_load[on].sum()
        M[k] = (f_load[out] * (y_load[out] - y0)).sum()
    return S, M


def process_run(folder):
    m = FOLDER_U.search(folder)
    if not m:
        raise ValueError(f"can't parse u_inf from folder name '{folder}'")
    u_inf = float(m.group(1))
    q = 0.5 * RHO * u_inf ** 2
    h5_path = os.path.join(OUTPUT_ROOT, folder, "savedata", f"{folder}.data.h5")
    if not os.path.exists(h5_path):
        import glob
        c = glob.glob(os.path.join(OUTPUT_ROOT, folder, "savedata", "*.data.h5"))
        if not c:
            raise FileNotFoundError(f"no savedata .h5 in {folder}")
        h5_path = c[0]
    d = read_sharpy(h5_path)

    surf = {s: surface_strips(d["zeta"][s], d["forces"][s], q) for s in WING_SURFACES + TAIL_SURFACES
            if s < len(d["zeta"])}

    # --- closure checks ------------------------------------------------------
    print(f"\n=== {folder}  (u_inf = {u_inf:g} m/s, q = {q:.2f} Pa)")
    total_aero_z = sum(F[2].sum() for F in d["forces"])
    for s, st in surf.items():
        msg = f"  surface {s}: sum vertical aero force = {st['fz'].sum():8.3f} N"
        if d["isf"] is not None:
            msg += f"   (inertial_steady_forces fz = {d['isf'][s, 2]:8.3f} N)"
        print(msg)
    print(f"  all surfaces: sum vertical aero force = {total_aero_z:.3f} N")

    out = dict(folder=folder, u_inf=u_inf, q=q, surf=surf)
    if d["grav"] is not None:
        g_z = -np.linalg.norm(d["grav"][:, :3], axis=1)        # see module docstring
        W = -g_z.sum()
        print(f"  total weight from gravity_forces = {W:.3f} N ({W / G:.3f} kg);"
              f" aero/W = {total_aero_z / W:.4f}")
        root = np.nonzero(np.abs(d["pos"][:, 1]) < Y_ZERO_TOL)[0]
        print(f"  weight on centreline nodes (incl. node 0 lumped masses) = {-g_z[root].sum():.3f} N")
        for sign, name in ((+1, "y>0"), (-1, "y<0")):
            nodes = wing_side_nodes(d["pos"], d["beam_number"], d["conn"], sign)
            out[f"grav_{name}"] = (np.abs(d["pos"][nodes, 1]), g_z[nodes])
            print(f"  wing half {name}: {len(nodes)} nodes, wing-borne weight = "
                  f"{-g_z[nodes].sum():.3f} N ({-g_z[nodes].sum() / G * 1000:.0f} g)")
    else:
        warnings.warn(f"{folder}: no gravity_forces in savedata - net shear/bending skipped")
    for s in WING_SURFACES:
        if s in surf:
            print(f"  wing surface {s}: half-wing lift = {surf[s]['fz'].sum():.3f} N")
    return out

# ============================================================================
# Main
# ============================================================================

def main():
    if "--inventory" in sys.argv:
        f0 = RUNS[0]
        inventory(os.path.join(OUTPUT_ROOT, f0, "savedata", f"{f0}.data.h5"))
        return

    runs = []
    for folder in RUNS:
        try:
            runs.append(process_run(folder))
        except Exception as e:
            warnings.warn(f"{folder}: skipped ({e})")
    if not runs:
        raise SystemExit("No runs could be read - check OUTPUT_ROOT / RUNS.")

    # --- ASWING -------------------------------------------------------------
    asw = None
    if os.path.exists(ASWING_FILE):
        beams, hdr = parse_aswing_beams(ASWING_FILE)
        q_a = hdr.get("q", 0.5 * RHO * 9.0 ** 2)
        wb, tb = beams[ASWING_WING_BEAM], beams[ASWING_TAIL_BEAM]
        r, rt = _starboard(wb["t"]), _starboard(tb["t"])  # y>0 half, incl. its own root row
        asw = dict(y=wb["y"][r], cl=wb["cl"][r], Fn=wb["Fn"][r], Mc=wb["Mc"][r],
                   l=q_a * ASWING_CHORD_WING * wb["cl"][r],
                   y_t=tb["y"][rt], cl_t=tb["cl"][rt], q=q_a)
        L_half = _trapz(asw["l"], asw["y"])
        print(f"\n=== ASWING {ASWING_FILE}: q = {q_a:.2f} Pa, alpha = {hdr.get('alpha')} deg")
        print(f"  half-wing lift (q c int cl dy) = {L_half:.3f} N; root Fn = {asw['Fn'][0]:.3f} N, "
              f"root Mc' = {asw['Mc'][0]:.3f} N m")
        print(f"  implied wing-borne weight per side = {L_half - asw['Fn'][0]:.3f} N "
              f"({(L_half - asw['Fn'][0]) / G * 1000:.0f} g)")
    else:
        warnings.warn(f"ASWING file '{ASWING_FILE}' not found - plotting SHARPy only")

    label = lambda r: RUN_LABELS.get(r["folder"], r["folder"])
    prim = runs[0]
    semispan = max(prim["surf"][s]["y_abs"].max() for s in WING_SURFACES)

    # --- CSV ----------------------------------------------------------------
    rows = []
    for r in runs:
        for s, st in r["surf"].items():
            for k in range(len(st["y"])):
                rows.append(dict(run=r["folder"], u_inf=r["u_inf"], surface=s, y=st["y"][k],
                                 fz_node=st["fz"][k], trib_width=st["w"][k], l_per_span=st["l"][k],
                                 chord=st["chord"][k], cl_local=st["cl"][k]))
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    # --- Figure 1: distribution 2x2 -----------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    (ax_a, ax_b), (ax_c, ax_d) = axs

    # (a) nodal vertical aero force, primary run, wing + tail, both halves
    for s, mk, nm in ((0, "o", "wing"), (1, "o", "wing"), (3, "s", "tail"), (4, "s", "tail")):
        if s not in prim["surf"]:
            continue
        st = prim["surf"][s]
        side = "y>0" if st["y"].mean() > 0 else "y<0"
        filled = side == "y>0"
        col = PALETTE[0] if nm == "wing" else PALETTE[1]
        ax_a.plot(st["y_abs"], st["fz"], mk + "-", color=col, ms=4, lw=0.8,
                  mfc=col if filled else "none", label=f"{nm}, {side}")
    ax_a.set_title(f"(a) vertical aero force at each node - {label(prim)}")
    ax_a.set_ylabel("F_z per node (N)")

    # (b) l(y), (c) l/l_mean, (d) cl
    for i, r in enumerate(runs):
        col = PALETTE[i % len(PALETTE)]
        for s in WING_SURFACES:
            st = r["surf"][s]
            ls = "-" if st["y"].mean() > 0 else "--"
            lab = label(r) if st["y"].mean() > 0 else "_nolegend_"
            l_mean = st["fz"].sum() / st["y_abs"].max()
            ax_b.plot(st["y_abs"], st["l"], ls, color=col, lw=1.8, label=lab)
            ax_c.plot(st["y_abs"] / st["y_abs"].max(), st["l"] / l_mean, ls, color=col, lw=1.8, label=lab)
            ax_d.plot(st["y_abs"], st["cl"], ls, color=col, lw=1.8, label=lab)
        for s in TAIL_SURFACES:
            if s in r["surf"] and r["surf"][s]["y"].mean() > 0:
                st = r["surf"][s]
                ax_d.plot(st["y_abs"], st["cl"], ":", color=col, lw=1.6,
                          label=f"{label(r)} (tail)" if i == 0 else "_nolegend_")

    eta = np.linspace(0, 1, 200)
    ax_c.plot(eta, 4 / np.pi * np.sqrt(1 - eta ** 2), color="#999999", lw=1.2, label="elliptic")
    if asw is not None:
        kw = dict(marker="s", ls="none", ms=5, mfc="none", mec=ASWING_COLOR, mew=1.2)
        ax_b.plot(asw["y"], asw["l"], **kw, label="ASWING 9 m/s (q c_ref cl)")
        l_mean_a = _trapz(asw["l"], asw["y"]) / asw["y"].max()
        ax_c.plot(asw["y"] / asw["y"].max(), asw["l"] / l_mean_a, **kw, label="ASWING 9 m/s")
        ax_d.plot(asw["y"], asw["cl"], **kw, label="ASWING wing")
        ax_d.plot(asw["y_t"], asw["cl_t"], marker="^", ls="none", ms=5, mfc="none",
                  mec=ASWING_COLOR, mew=1.2, label="ASWING tail")
    ax_b.set_title("(b) lift per unit span (solid y>0, dashed y<0)")
    ax_b.set_ylabel("l(y) (N/m)")
    ax_c.set_title("(c) shape only: l(y) / mean l  (same-CL comparison)")
    ax_c.set_ylabel("l / l_mean")
    ax_c.set_xlabel("eta = |y| / semispan")
    ax_d.set_title("(d) local section cl (dotted = SHARPy tail)")
    ax_d.set_ylabel("cl")
    ax_d.axhline(0, color=GRID, lw=1)
    for ax in (ax_a, ax_b, ax_d):
        ax.set_xlabel("|y| (m)")
    for ax in axs.ravel():
        ax.grid(color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PLOT_DIST, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT_PLOT_DIST}")

    # --- Figure 2: shear / bending -------------------------------------------
    fig2, (ax_s, ax_m) = plt.subplots(1, 2, figsize=(13, 4.8))
    stations = np.linspace(0, semispan, 300)
    for i, r in enumerate(runs):
        col = PALETTE[i % len(PALETTE)]
        st = next(r["surf"][s] for s in WING_SURFACES if r["surf"][s]["y"].mean() > 0)
        Sa, Ma = shear_bending(st["y_abs"], st["fz"], stations)
        if "grav_y>0" in r:
            yg, fg = r["grav_y>0"]
            Sg, Mg = shear_bending(yg, fg, stations)
        else:
            Sg = Mg = None
        if i == 0:
            ax_s.plot(stations, Sa, ":", color=col, lw=1.6, label=f"{label(r)} aero only")
            ax_m.plot(stations, Ma, ":", color=col, lw=1.6, label=f"{label(r)} aero only")
            if Sg is not None:
                ax_s.plot(stations, Sg, "-.", color=col, lw=1.2, label=f"{label(r)} gravity only")
                ax_m.plot(stations, Mg, "-.", color=col, lw=1.2, label=f"{label(r)} gravity only")
        if Sg is not None:
            ax_s.plot(stations, Sa + Sg, "-", color=col, lw=2, label=f"{label(r)} NET")
            ax_m.plot(stations, Ma + Mg, "-", color=col, lw=2, label=f"{label(r)} NET")
            print(f"  {label(r)}: root net shear = {(Sa + Sg)[0]:.3f} N, "
                  f"root net bending = {(Ma + Mg)[0]:.3f} N m "
                  f"(aero-only {Sa[0]:.3f} N / {Ma[0]:.3f} N m)")
    if asw is not None:
        kw = dict(marker="s", ls="none", ms=5, mfc="none", mec=ASWING_COLOR, mew=1.2)
        ax_s.plot(asw["y"], asw["Fn"], **kw, label="ASWING Fn (internal shear)")
        ax_m.plot(asw["y"], asw["Mc"], **kw, label="ASWING Mc' (internal bending)")
    ax_s.set_title("Wing shear force, tip -> root (y>0 half)")
    ax_s.set_ylabel("shear (N)")
    ax_m.set_title("Wing bending moment, tip -> root (y>0 half)")
    ax_m.set_ylabel("bending moment (N m)")
    for ax in (ax_s, ax_m):
        ax.set_xlabel("|y| (m)")
        ax.axhline(0, color=GRID, lw=1)
        ax.grid(color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    fig2.tight_layout()
    fig2.savefig(OUT_PLOT_SHEAR, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT_PLOT_SHEAR}")


if __name__ == "__main__":
    main()