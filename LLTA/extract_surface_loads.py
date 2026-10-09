"""
Extract per-surface (or per-node-subset) aerodynamic force AND moment from a SHARPy
`SaveData` output file (`*.data.h5`), for a single timestep.

Why this exists
----------------
`AeroForcesCalculator` writes a per-surface FORCE breakdown into the saved timestep
(`data/aero/timestep_info/<ts>/inertial_steady_forces` and `body_steady_forces`,
shape (n_surf, 6)) -- but it never populates the moment columns (3:6) of those
arrays; only the whole-aircraft total (`total_steady_body_forces`, a single 6-vector)
carries a real moment, computed about a hardcoded [0, 0, 0] reference point in the A
frame (there is no exposed setting to change that reference point).

To get a per-surface (e.g. tail-only) MOMENT about a reference point of your choosing
(e.g. the aircraft CG), this script reconstructs it from the same raw material
AeroForcesCalculator itself reduces: the per-STRUCTURAL-NODE force+moment saved at
`data/structure/timestep_info/<ts>/postproc_node/aero_steady_forces` (shape
(num_node, 6), expressed in the local body-fixed B frame at each node), rotated into
the A frame node-by-node (replicating `StructTimeStepInfo.nodal_b_for_2_a_for`, i.e.
Rodrigues' formula on each node's `psi`/CRV), then reduced with the same formula as
`sharpy.aero.utils.mapping.total_forces_moments`:

    F_total       = sum_i f_i
    M_about_ref   = sum_i [ m_i + (pos_i - ref) x f_i ]

`aero2struct_mapping` (also in the saved file) gives, for each aerodynamic surface
index `i_surf`, exactly which structural node indices belong to it -- so the same
reduction can be restricted to any subset of surfaces (e.g. just your tail halves)
to get a genuine per-surface moment, not just force.

Validated against a real LLTA output: reducing over ALL nodes with ref_pos=[0,0,0]
reproduces the file's own `total_steady_body_forces` to 6 significant figures --
confirming the rotation + reduction here is done the same way SHARPy does it
internally, not an approximation layered on top.

Caveat (check before trusting the split too far): cross-checking the per-surface
FORCE reconstructed this way against AeroForcesCalculator's own per-surface
`inertial_steady_forces` shows a few-percent difference for the larger surfaces
(likely from how `postproc_node['aero_steady_forces']` lumps panel forces onto
nearby structural nodes, vs. AeroForcesCalculator's direct panel-level projection),
and a larger *relative* difference on the smallest-force surface (a fin, most
likely). The whole-aircraft total is exact; the per-surface split is a good,
internally-consistent cross-check, not bit-for-bit identical to a hypothetical
"AeroForcesCalculator with per-surface moments". For a finite-difference across
elevator angles (the actual goal), a small consistent mapping bias mostly cancels
out in the subtraction -- but don't over-trust the absolute per-surface numbers to
better than a few percent, especially for low-force surfaces.

No SHARPy installation is required to run this -- only h5py and numpy.

Usage
-----
    python extract_surface_loads.py path/to/case.data.h5 [--ref x y z] [--ts 0]

Or import and call `extract(...)` from your own sweep-analysis script.
"""
import argparse
import numpy as np
import h5py


def skew(v):
    m = np.zeros((3, 3))
    m[1, 2] = -v[0]; m[2, 0] = -v[1]; m[0, 1] = -v[2]
    m[2, 1] = v[0]; m[0, 2] = v[1]; m[1, 0] = v[2]
    return m


def crv2rotation(psi):
    """Cartesian Rotation Vector -> rotation matrix (Rodrigues' formula), matching
    sharpy.utils.algebra.crv2rotation exactly."""
    n = np.linalg.norm(psi)
    S = skew(psi)
    if n < 1e-8:
        return np.eye(3) + S + 0.5 * S @ S
    return np.eye(3) + (np.sin(n) / n) * S + ((1 - np.cos(n)) / n ** 2) * (S @ S)


def cross3(v, w):
    return np.array([
        v[1] * w[2] - v[2] * w[1],
        -v[0] * w[2] + v[2] * w[0],
        v[0] * w[1] - v[1] * w[0],
    ])


def total_forces_moments(forces_nodes_a, pos_def, ref_pos=np.zeros(3)):
    """Same reduction as sharpy.aero.utils.mapping.total_forces_moments:
    forces_nodes_a: (n, 6) force+moment per node, already in the A frame
    pos_def: (n, 3) node positions in A
    ref_pos: point in A about which to sum moments
    """
    ra = pos_def - ref_pos
    F = forces_nodes_a[:, :3].sum(axis=0)
    M = forces_nodes_a[:, 3:].sum(axis=0) + sum(
        cross3(ra[i], forces_nodes_a[i, :3]) for i in range(len(ra))
    )
    return F, M


def load_case(h5path, ts=0):
    """Loads everything needed from a SHARPy SaveData file for this extraction."""
    f = h5py.File(h5path, 'r')
    tskey = f'{ts:05d}'

    pos = f[f'data/structure/timestep_info/{tskey}/pos'][:]
    psi = f[f'data/structure/timestep_info/{tskey}/psi'][:]
    nodal_b = f[f'data/structure/timestep_info/{tskey}/postproc_node/aero_steady_forces'][:]
    node_master_elem = f['data/structure/node_master_elem'][:]

    # per-surface -> structural node index membership (0-indexed)
    a2s_grp = f['data/aero/aero2struct_mapping']
    surf_nodes = {
        int(k): a2s_grp[k]['_as_array'][:]
        for k in a2s_grp.keys() if k != '_read_as'
    }

    # AeroForcesCalculator's own outputs, for cross-checking
    inertial_steady_forces = f[f'data/aero/timestep_info/{tskey}/inertial_steady_forces'][:]
    body_steady_forces = f[f'data/aero/timestep_info/{tskey}/body_steady_forces'][:]
    total_steady_body_forces = f[f'data/aero/timestep_info/{tskey}/total_steady_body_forces'][:]

    f.close()
    return dict(
        pos=pos, psi=psi, nodal_b=nodal_b, node_master_elem=node_master_elem,
        surf_nodes=surf_nodes,
        inertial_steady_forces=inertial_steady_forces,
        body_steady_forces=body_steady_forces,
        total_steady_body_forces=total_steady_body_forces,
    )


# ---------------------------------------------------------------------------
# CG: (a) read SHARPy's own value straight out of a 'Modal' run, and
#     (b) an independent hand-computable cross-check from the raw mass model.
# ---------------------------------------------------------------------------

def get_sharpy_modal_cg(h5path, ts=0):
    """
    Reads the CG SHARPy itself computed, straight from a saved 'Modal' step
    (sharpy.structure.utils.modalutils.cg(), called on the fully-assembled
    global mass matrix -- i.e. every distributed + lumped mass contribution,
    exactly as SHARPy uses it elsewhere).

    Requires 'Modal' to have been in the case's `flow` (with `SaveData`'s
    `save_struct: True` running afterwards) for this specific run -- if not,
    raises a clear error rather than silently returning something else.
    """
    f = h5py.File(h5path, 'r')
    tskey = f'{ts:05d}'
    modal_path = f'data/structure/timestep_info/{tskey}/modal'
    if modal_path not in f or 'cg' not in f[modal_path]:
        f.close()
        raise KeyError(
            f"No 'modal/cg' found at timestep {tskey} in {h5path}. "
            "Add 'Modal' to this case's `flow` (after the trim/coupled step, "
            "with 'SaveData' running after 'Modal') and re-run, or compute "
            "the CG by hand with mass_buildup_cg() instead."
        )
    cg = f[f'{modal_path}/cg'][:]
    f.close()
    return cg


def mass_buildup_cg(h5path, ts=0):
    """
    Hand-computable cross-check for the CG, built directly from the same raw
    mass-model arrays SHARPy itself assembles (no SHARPy install needed):

      - distributed structural mass along each beam element:
            m_e   = mu_e * L_e            (mu_e from mass_db[elem_mass[e], 0, 0],
                                            L_e = element's physical length)
            r_e   ~= mean of that element's 3 node positions   (centroid approx.)
      - lumped point masses:
            m_k   = lumped_mass[k]
            r_k   = pos[attachment node] + C_AB(node) @ lumped_mass_position[k]
                    (local offset must be rotated B -> A, same rotation used
                    for the aero postproc_node forces elsewhere in this script)

      M_total = sum(m_e) + sum(m_k)
      CG      = ( sum(m_e * r_e) + sum(m_k * r_k) ) / M_total

    Uses the SAME timestep's (possibly deformed) `pos`/`psi` that a 'Modal'
    run at that timestep would use -- the CG is technically evaluated about
    the current deformed shape, not a fixed jig-shape constant, so compare
    like-for-like against get_sharpy_modal_cg() at the same `ts`.

    Returns (cg[3], total_mass).
    """
    f = h5py.File(h5path, 'r')
    tskey = f'{ts:05d}'
    s = f['data/structure']

    pos = s[f'timestep_info/{tskey}/pos'][:]
    psi = s[f'timestep_info/{tskey}/psi'][:]
    node_master_elem = s['node_master_elem'][:]
    connectivities = s['connectivities'][:]
    mass_db = s['mass_db'][:]
    elem_mass = s['elem_mass'][:]

    has_lumped = 'lumped_mass' in s and s['lumped_mass'].shape != ()
    lumped_mass = s['lumped_mass'][:] if has_lumped else np.array([])
    lumped_mass_position = s['lumped_mass_position'][:] if has_lumped else np.zeros((0, 3))
    lumped_mass_nodes = s['lumped_mass_nodes'][:] if has_lumped else np.array([], dtype=int)
    f.close()

    total_m = 0.0
    first_moment = np.zeros(3)

    for i_elem in range(connectivities.shape[0]):
        node_idx = connectivities[i_elem]
        node_pos = pos[node_idx]
        # element length: sort the 3 nodes along their dominant spread axis and
        # sum consecutive segment lengths -- avoids needing to know this
        # SHARPy build's [start, end, mid] ordering convention explicitly.
        spread = node_pos.max(axis=0) - node_pos.min(axis=0)
        axis = np.argmax(spread)
        order = np.argsort(node_pos[:, axis])
        sorted_pos = node_pos[order]
        length = (np.linalg.norm(sorted_pos[1] - sorted_pos[0])
                  + np.linalg.norm(sorted_pos[2] - sorted_pos[1]))
        mu = mass_db[elem_mass[i_elem], 0, 0]
        m_elem = mu * length
        centroid = node_pos.mean(axis=0)
        total_m += m_elem
        first_moment += m_elem * centroid

    for j in range(len(lumped_mass)):
        m = lumped_mass[j]
        inode = lumped_mass_nodes[j]
        i_elem, i_local = node_master_elem[inode]
        cab = crv2rotation(psi[i_elem, i_local, :])
        r_global = pos[inode] + cab @ lumped_mass_position[j]
        total_m += m
        first_moment += m * r_global

    return first_moment / total_m, total_m


def rotate_nodal_b_to_a(nodal_b, psi, node_master_elem):
    """Replicates StructTimeStepInfo.nodal_b_for_2_a_for: rotate each node's local
    (B-frame) force+moment 6-vector into the reference (A) frame."""
    num_node = nodal_b.shape[0]
    nodal_a = np.zeros_like(nodal_b)
    for i in range(num_node):
        i_elem, i_local = node_master_elem[i]
        cab = crv2rotation(psi[i_elem, i_local, :])
        nodal_a[i, 0:3] = cab @ nodal_b[i, 0:3]
        nodal_a[i, 3:6] = cab @ nodal_b[i, 3:6]
    return nodal_a


def extract(h5path, ts=0, ref_pos=np.zeros(3), surface_groups=None, verify=True):
    """
    Returns a dict with:
      - 'per_surface': {i_surf: (F[3], M[3])} about ref_pos, for every surface found
      - 'groups': {group_name: (F[3], M[3])} about ref_pos, if surface_groups given
      - 'aircraft_total': (F[3], M[3]) reconstructed over all nodes (should match
        total_steady_body_forces if verify succeeds)
      - 'verified': bool, whether the all-node reduction matches SHARPy's own total

    surface_groups: optional dict like {'tail': [3, 4], 'wing': [0, 1]} to also sum
      several surfaces together (careful: surfaces sharing a root/junction node --
      check 'surf_nodes' -- will have that node's contribution counted once per
      surface in the group if you list them individually and add manually; this
      function unions node indices within a group so shared nodes aren't
      double-counted for that group's own total).
    """
    d = load_case(h5path, ts=ts)
    nodal_a = rotate_nodal_b_to_a(d['nodal_b'], d['psi'], d['node_master_elem'])

    per_surface = {}
    for i_surf, idx in sorted(d['surf_nodes'].items()):
        F, M = total_forces_moments(nodal_a[idx], d['pos'][idx], ref_pos=ref_pos)
        per_surface[i_surf] = (F, M)

    groups = {}
    if surface_groups:
        for name, surf_list in surface_groups.items():
            idx = np.unique(np.concatenate([d['surf_nodes'][s] for s in surf_list]))
            F, M = total_forces_moments(nodal_a[idx], d['pos'][idx], ref_pos=ref_pos)
            groups[name] = (F, M)

    F_all, M_all = total_forces_moments(nodal_a, d['pos'], ref_pos=np.zeros(3))
    verified = False
    if verify:
        verified = bool(np.allclose(
            np.concatenate([F_all, M_all]),
            d['total_steady_body_forces'],
            atol=1e-6, rtol=1e-5,
        ))

    return dict(
        per_surface=per_surface,
        groups=groups,
        aircraft_total=(F_all, M_all),
        verified=verified,
        surf_nodes=d['surf_nodes'],
        inertial_steady_forces=d['inertial_steady_forces'],  # AeroForcesCalculator's own per-surface FORCE (cols 3:6 always 0)
    )


def _print_report(result):
    print(f"Whole-aircraft total reconstruction matches SHARPy's own total: {result['verified']}")
    print()
    print(f"{'i_surf':>6} {'#nodes':>7}  {'Fx':>10} {'Fy':>10} {'Fz':>10}   {'Mx':>10} {'My':>10} {'Mz':>10}   (force cross-check vs AeroForcesCalculator's own Fx,Fy,Fz)")
    for i_surf, (F, M) in sorted(result['per_surface'].items()):
        n = len(result['surf_nodes'][i_surf])
        afc = result['inertial_steady_forces'][i_surf, :3]
        print(f"{i_surf:>6} {n:>7}  {F[0]:>10.4f} {F[1]:>10.4f} {F[2]:>10.4f}   "
              f"{M[0]:>10.4f} {M[1]:>10.4f} {M[2]:>10.4f}   "
              f"(AFC: {afc[0]:.4f}, {afc[1]:.4f}, {afc[2]:.4f})")
    if result['groups']:
        print()
        for name, (F, M) in result['groups'].items():
            print(f"group '{name}': F = {F}, M = {M}")


def resolve_ref_pos(h5path, ts=0, user_ref=None, quiet=False):
    """Shared CG-resolution logic: user-supplied > saved Modal CG > hand-buildup
    CG > [0,0,0] (last resort, with a loud warning)."""
    def _p(msg):
        if not quiet:
            print(msg)

    if user_ref is not None:
        _p(f"Using user-supplied ref_pos = {user_ref}")
        return np.array(user_ref)
    try:
        ref_pos = get_sharpy_modal_cg(h5path, ts=ts)
        _p(f"Using CG from saved Modal step ({h5path}): {ref_pos}")
        return ref_pos
    except KeyError as e:
        _p(f"({e})")
        try:
            ref_pos, total_m = mass_buildup_cg(h5path, ts=ts)
            _p(f"Using hand-computed mass-buildup CG instead: {ref_pos} (total mass {total_m:.4f} kg)")
            return ref_pos
        except Exception:
            _p("Falling back to ref_pos = [0, 0, 0] (mesh origin, NOT the CG -- "
               "moment slopes about this point are not directly comparable to a "
               "CG-referenced value).")
            return np.zeros(3)


def fit_slope(x, y):
    """Least-squares slope + intercept through (x, y). Works for 2 points (exact
    finite difference) or more (least-squares fit, so one noisy point doesn't
    dictate the whole answer). x, y: 1D arrays of the same length."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept


def derivative_report(files_and_defl, ts=0, ref_pos=None, surfaces=None, group_name='selection',
                       rho=None, u_inf=None, s_tail=None, c_tail=None):
    """
    The actual finite-difference / slope calculation `extract()` on its own does
    NOT do. Give it several (deflection_deg, h5path) pairs at the SAME alpha and
    u_inf (varying only the control surface deflection) and it fits dF/ddefl and
    dM/ddefl for a chosen surface or group across them.

    files_and_defl: list of (deflection_deg, h5path) tuples, e.g.
        [(-2.0, 'run_m2.data.h5'), (0.0, 'run_0.data.h5'), (2.0, 'run_p2.data.h5')]
    surfaces: list of surface indices to sum together (e.g. [3, 4] for your two
        tail halves) -- required.
    ref_pos: the SAME reference point is used for every file in the sweep, on
        purpose -- resolved once (from the first file, if not given explicitly),
        not per-file. Using a different reference point per file would silently
        reintroduce the exact (ref-point x dF/ddefl) artifact discussed earlier,
        since the reduction is only reference-point-INDEPENDENT for a fixed
        reference across the comparison, not fixed per-run.
    rho, u_inf, s_tail, c_tail: optional. If ALL of rho/u_inf/s_tail are given,
        also computes dCL_tail/ddefl = (dFz/ddefl) / (q*s_tail), with
        q = 0.5*rho*u_inf**2 -- the same normalisation AeroForcesCalculator
        itself uses (calculate_coefficients()), just with the tail's own
        reference area instead of the aircraft's S_ref. If c_tail is ALSO
        given, additionally computes dCm_tail/ddefl = (dMy/ddefl)/(q*s_tail*c_tail).
        These are what's directly comparable to ASWing's dCLdF1 (and the
        moment it implies via the tail arm) -- leave these as None to just
        get the dimensional (N, N.m) slopes.
    """
    if surfaces is None:
        raise ValueError("Pass `surfaces` (e.g. [3, 4] for your tail halves) -- "
                          "there's no safe default to difference against.")

    files_and_defl = sorted(files_and_defl, key=lambda t: t[0])
    if ref_pos is None:
        ref_pos = resolve_ref_pos(files_and_defl[0][1], ts=ts)

    defl = []
    F_list = []
    M_list = []
    for d, path in files_and_defl:
        res = extract(path, ts=ts, ref_pos=ref_pos, surface_groups={group_name: surfaces})
        F, M = res['groups'][group_name]
        defl.append(d)
        F_list.append(F)
        M_list.append(M)

    defl = np.array(defl)
    F_arr = np.array(F_list)   # (n_points, 3)
    M_arr = np.array(M_list)   # (n_points, 3)

    result = {'ref_pos': ref_pos, 'deflections_deg': defl, 'F': F_arr, 'M': M_arr}
    labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    data = np.hstack([F_arr, M_arr])
    result['slopes_per_deg'] = {}
    result['slopes_per_rad'] = {}
    for i, lbl in enumerate(labels):
        slope_deg, intercept = fit_slope(defl, data[:, i])
        result['slopes_per_deg'][lbl] = slope_deg
        result['slopes_per_rad'][lbl] = slope_deg * 180.0 / np.pi

    result['q'] = None
    result['dCL_tail_per_rad'] = None
    result['dCm_tail_per_rad'] = None
    if rho is not None and u_inf is not None and s_tail is not None:
        q = 0.5 * rho * u_inf ** 2
        result['q'] = q
        result['dCL_tail_per_rad'] = result['slopes_per_rad']['Fz'] / (q * s_tail)
        if c_tail is not None:
            result['dCm_tail_per_rad'] = result['slopes_per_rad']['My'] / (q * s_tail * c_tail)

    return result


def _print_derivative_report(result):
    print(f"Reference point used for ALL points in this sweep: {result['ref_pos']}")
    print(f"Deflections (deg): {result['deflections_deg']}")
    print()
    print(f"{'':>4} {'per-deg':>14} {'per-rad':>14}")
    for lbl in ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']:
        print(f"{lbl:>4} {result['slopes_per_deg'][lbl]:>14.5f} {result['slopes_per_rad'][lbl]:>14.5f}")

    if result['q'] is not None:
        print()
        print(f"q = 0.5*rho*u_inf^2 = {result['q']:.4f} Pa")
        print(f"dCL_tail/d(delta_e), per rad : {result['dCL_tail_per_rad']:.5f}   "
              f"<-- compare directly against ASWing's dCLdF1 (0.108)")
        if result['dCm_tail_per_rad'] is not None:
            print(f"dCm_tail/d(delta_e), per rad : {result['dCm_tail_per_rad']:.5f}   "
                  f"<-- the moment ASWing's model gets purely from dCLdF1 x tail arm "
                  f"(it has no separate dCMdF1 term)")
    else:
        print()
        print("Compare 'Fz per-rad' against ASWing's dCLdF1 (after normalising both by "
              "q*S_tail), and 'My per-rad' against the moment implied by "
              "dCLdF1 x (tail AC - CG) arm, per the earlier discussion. "
              "(Pass rho/u_inf/s_tail to derivative_report() to get this done "
              "automatically instead.)")


# ---------------------------------------------------------------------------
# Wing leading-edge x-location (root), for converting a CG into %MAC.
# ---------------------------------------------------------------------------

def get_wing_le_root_x(h5path, i_surf, ts=0, verbose=True):
    """
    Reads the x-location of the wing's leading edge AT THE ROOT straight out of
    the saved aerodynamic grid (`data/aero/timestep_info/<ts>/zeta/<i_surf>`),
    in the same A-frame axes/datum as everything else this script extracts
    (x_cg included) -- so it's guaranteed consistent with your CG value,
    unlike re-deriving it by hand from a separate geometry script.

    `zeta` has shape (3, M+1, N+1): (x/y/z, chordwise, spanwise). Two things
    have to be identified automatically rather than assumed, since the raw
    array doesn't label them:

      - which spanwise index is the ROOT (not necessarily index 0): taken as
        whichever spanwise station sits closest to the fuselage centreline
        (min sqrt(y^2 + z^2)).
      - which chordwise index is the LEADING edge (not necessarily index 0):
        determined from the sign of the local freestream `u_ext` at that
        surface -- the LE is whichever chordwise end the flow reaches first,
        so this is read from the actual run rather than assumed from an
        indexing convention that can differ surface-to-surface.

    Returns x_le_root (float). Also prints a sanity check: the local root
    chord length, and (if this surface's structural root node can be
    identified) where that node's x falls as a fraction of the chord --
    should land between 0 and 1 (typically ~0.2-0.35 for a wingbox/elastic
    axis position); if it doesn't, something about the surface picked is off
    (e.g. i_surf is actually a tail/fin, not the wing).
    """
    f = h5py.File(h5path, 'r')
    tskey = f'{ts:05d}'
    skey = f'{i_surf:05d}'
    zeta = f[f'data/aero/timestep_info/{tskey}/zeta/{skey}'][:]
    u_ext = f[f'data/aero/timestep_info/{tskey}/u_ext/{skey}'][:]

    # root = spanwise station closest to the centreline, checked at chordwise index 0
    radial = np.sqrt(zeta[1, 0, :] ** 2 + zeta[2, 0, :] ** 2)
    j_root = int(np.argmin(radial))

    # LE = whichever chordwise end the flow reaches first, from u_ext's own sign
    flow_x = np.mean(u_ext[0, :, j_root])
    x0, x1 = zeta[0, 0, j_root], zeta[0, -1, j_root]
    le_idx, te_idx = (0, -1) if (flow_x >= 0) == (x0 <= x1) else (-1, 0)

    x_le_root = float(zeta[0, le_idx, j_root]) # First chordwise point of the aerogrid. Zeta from aerogrid (stored in interial G frame) - rotated from A by trim attitude. 
    x_te_root = float(zeta[0, te_idx, j_root])
    chord = float(np.linalg.norm(zeta[:, le_idx, j_root] - zeta[:, te_idx, j_root]))

    if verbose:
        print(f"Surface {i_surf}: root spanwise index = {j_root}, "
              f"LE chordwise index = {le_idx if le_idx >= 0 else zeta.shape[1] - 1}")
        print(f"  x_LE,root = {x_le_root:.4f}   x_TE,root = {x_te_root:.4f}   "
              f"local root chord = {chord:.4f}")
        try:
            a2s_grp = f['data/aero/aero2struct_mapping']
            root_node = int(a2s_grp[skey]['_as_array'][0])
            x_node = float(f[f'data/structure/timestep_info/{tskey}/pos'][root_node, 0])
            frac = (x_node - x_le_root) / (x_te_root - x_le_root)
            print(f"  structural root node (#{root_node}) x = {x_node:.4f}  "
                  f"-> {frac * 100:.1f}% of local chord from LE "
                  f"{'(looks sane)' if 0.0 <= frac <= 1.0 else '(!! outside [0,1] -- check i_surf/indices)'}")
        except Exception:
            pass

    f.close()
    return x_le_root


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('h5file', nargs='?', help='Single .data.h5 file to report on')
    ap.add_argument('--ts', type=int, default=0)
    ap.add_argument('--ref', type=float, nargs=3, default=None,
                     help='Reference point (A frame) to take moments about, e.g. the CG. '
                          'If omitted, tries to read it from a saved Modal step '
                          '(data/structure/timestep_info/<ts>/modal/cg); falls back to '
                          'the hand-computed mass-buildup CG if Modal was not run; '
                          'falls back to [0,0,0] only if neither is available.')
    ap.add_argument('--sweep', nargs='+', default=None,
                     help="Do the finite-difference/slope calc instead of a single-file "
                          "report. Pass 'path:deflection_deg' pairs (path FIRST, so a "
                          "negative deflection doesn't get mistaken for another flag), e.g.: "
                          "--sweep run_m2.data.h5:-2 run_0.data.h5:0 run_p2.data.h5:2")
    ap.add_argument('--surfaces', type=int, nargs='+', default=None,
                     help='Surface indices to sum together for --sweep (e.g. --surfaces 3 4 '
                          'for your two tail halves). Required with --sweep.')
    ap.add_argument('--rho', type=float, default=None, help='Air density, for coefficient slopes')
    ap.add_argument('--u-inf', type=float, default=None, help='Freestream speed, for coefficient slopes')
    ap.add_argument('--s-tail', type=float, default=None, help="Tail's own reference area, for coefficient slopes")
    ap.add_argument('--c-tail', type=float, default=None, help="Tail's own reference chord, for the Cm slope too")
    args = ap.parse_args()

    if args.sweep is not None:
        if args.surfaces is None:
            raise SystemExit("--sweep requires --surfaces (e.g. --surfaces 3 4)")
        pairs = []
        for item in args.sweep:
            path, defl_str = item.rsplit(':', 1)
            pairs.append((float(defl_str), path))
        ref_pos = resolve_ref_pos(pairs[0][1], ts=args.ts, user_ref=args.ref)
        print()
        result = derivative_report(pairs, ts=args.ts, ref_pos=ref_pos, surfaces=args.surfaces,
                                    rho=args.rho, u_inf=args.u_inf, s_tail=args.s_tail, c_tail=args.c_tail)
        _print_derivative_report(result)
    else:
        if args.h5file is None:
            raise SystemExit("Provide a .data.h5 file, or use --sweep for the derivative calc.")
        ref_pos = resolve_ref_pos(args.h5file, ts=args.ts, user_ref=args.ref)

        # Print both CG estimates side by side when both are available, as the actual cross-check.
        try:
            cg_modal = get_sharpy_modal_cg(args.h5file, ts=args.ts)
            print(f"  SHARPy (Modal) CG      : {cg_modal}")
        except KeyError:
            cg_modal = None
        try:
            cg_hand, total_m = mass_buildup_cg(args.h5file, ts=args.ts)
            print(f"  Hand mass-buildup CG   : {cg_hand}  (total mass {total_m:.4f} kg)")
        except Exception:
            cg_hand = None
        if cg_modal is not None and cg_hand is not None:
            diff = cg_modal - cg_hand
            print(f"  Difference (Modal - hand): {diff}")
        print()

        result = extract(args.h5file, ts=args.ts, ref_pos=ref_pos)
        _print_report(result)