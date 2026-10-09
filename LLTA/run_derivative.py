"""
Easier front end for extract_surface_loads.py's --sweep mode: no command-line
flags, no quoting paths with spaces, no colon-separated strings to get right --
just edit the three settings below and run:

    python run_derivative.py

Must live in the same folder as extract_surface_loads.py.
"""
from extract_surface_loads import derivative_report, _print_derivative_report, resolve_ref_pos

# =============================================================================
# EDIT THESE THREE THINGS, THEN RUN THIS FILE
# =============================================================================

# One (deflection_deg, path_to_data.h5) entry per run. Paste real paths --
# copy them straight from Finder/your output folder, spaces and all, no
# quoting needed since these are just Python strings.
RUNS = [
    (-2, "/Users/icc25/sharpy_cases/LLTA/output/llta_trim_uinf_9_elev_-2_no_trim/savedata/llta_trim_uinf_9_elev_-2_no_trim.data.h5"),
    (0,  "/Users/icc25/sharpy_cases/LLTA/output/llta_trim_uinf_9_elev_0_no_trim/savedata/llta_trim_uinf_9_elev_0_no_trim.data.h5"),
    (2,  "/Users/icc25/sharpy_cases/LLTA/output/llta_trim_uinf_9_elev_+2_no_trim/savedata/llta_trim_uinf_9_elev_+2_no_trim.data.h5"),
]

# Which aero surface indices to sum together (e.g. your two tail halves).
# Run extract_surface_loads.py on a single file first if you need to re-check
# which i_surf values these are.
SURFACES = [3, 4]

# Reference point for moments. Leave as None to auto-resolve (uses the CG from
# a saved 'Modal' step if present, else the hand-buildup CG, else [0,0,0] with
# a warning) -- the SAME point is then reused for every run in the sweep.
# Set to e.g. [0.238, 0.0, 0.032] to force a specific point instead.
REF_POS = None

# Optional: fill these in to also get dCL_tail/ddefl and dCm_tail/ddefl directly
# comparable to ASWing's dCLdF1 -- leave as None to just see the dimensional
# (N, N.m) slopes instead. q = 0.5*RHO*U_INF**2 gets computed for you.
RHO = 1.225      # e.g. 1.225 (kg/m^3)
U_INF = 9    # e.g. 9.0  (m/s) -- must match the u_inf these RUNS were run at
S_TAIL = 2*0.099   # e.g. 0.12 (m^2) -- the TAIL's own reference area, not the aircraft's S_ref
C_TAIL = 0.135   # e.g. 0.15 (m)   -- the TAIL's own reference chord, only needed for the Cm slope

# =============================================================================
# Nothing below this line needs editing.
# =============================================================================

if __name__ == '__main__':
    placeholder_paths = [p for _, p in RUNS if p.startswith("/path/to")]
    if placeholder_paths:
        raise SystemExit(
            "Edit the RUNS list at the top of this file first -- it still has "
            "the placeholder paths in it."
        )

    ref_pos = resolve_ref_pos(RUNS[0][1], user_ref=REF_POS)
    print()
    result = derivative_report(RUNS, ref_pos=ref_pos, surfaces=SURFACES,
                                rho=RHO, u_inf=U_INF, s_tail=S_TAIL, c_tail=C_TAIL)
    _print_derivative_report(result)