"""
Rough rigid-body mass, CG and inertia tensor (about the CG) from a SHARPy
SaveData file (*.data.h5), in the same spirit as mass_buildup_cg() in
extract_surface_loads.py -- it reads the same arrays and uses the same
element-length / lumped-mass-rotation conventions, and adds the inertia.

Must live in the same folder as extract_surface_loads.py.

    python rigid_body_inertia.py path/to/case.data.h5 [--ts 0] [--wing-cg-offset 0.02]

What it does
------------
  * distributed mass: each beam element (3 nodes) is split into n_sub equal
    slices along the quadratic curve through its nodes. Each slice carries
    mu * length / n_sub, with mu = mass_db[elem_mass[e], 0, 0].
  * lumped masses: placed at  pos[node] + C_AB(node) @ lumped_mass_position,
    exactly as mass_buildup_cg() does.
  * inertia about the CG:  I = sum m (|d|^2 * Id - d d^T)  (point-mass terms)
    + own inertia of each element (diag of mass_db[..., 3:6, 3:6] * length,
      rotated B -> A with C_AB) + own inertia of each lumped mass (rotated
      B -> A). Own-inertia terms are small; set include_own_inertia=False to
      see how much they matter.

Known limits (first-pass tool, not a replacement for SHARPy's own Modal CG)
---------------------------------------------------------------------------
  * mass_db[0, 0] only: the CG offset stored in the wing mass matrix's
    translation-rotation coupling (cg_offset_main, 0.02 m aft of the elastic
    axis) is NOT picked up from the file. Pass wing_cg_offset_x to add it
    (metres, +x aft). Without it x_cg lands ~0.009 m forward of the true CG.
  * The wing mass_db rotational terms already include a small parallel-axis
    term about the elastic axis, so own-inertia is slightly overstated
    (order 0.3% of Iyy).
  * Uses the (possibly deformed) pos/psi at the chosen timestep, so numbers
    shift slightly with trim deformation (wing tips rise, z_cg moves up).
"""
import argparse
import numpy as np
import h5py
from extract_surface_loads import crv2rotation


def _quad_points(p0, p_end, p_mid, n_sub):
    """Midpoints of n_sub equal slices along the quadratic through
    start (s=0), mid (s=0.5), end (s=1)."""
    s = (np.arange(n_sub) + 0.5) / n_sub
    l0 = 2 * (s - 0.5) * (s - 1.0)
    lm = -4 * s * (s - 1.0)
    l1 = 2 * s * (s - 0.5)
    return l0[:, None] * p0 + lm[:, None] * p_mid + l1[:, None] * p_end


def rigid_body_inertia(h5path, ts=0, n_sub=20, wing_cg_offset_x=0.0,
                       wing_mass_ids=(0, 1), include_own_inertia=True):
    f = h5py.File(h5path, 'r')
    tskey = f'{ts:05d}'
    s = f['data/structure']
    pos = s[f'timestep_info/{tskey}/pos'][:]
    psi = s[f'timestep_info/{tskey}/psi'][:]
    node_master_elem = s['node_master_elem'][:]
    conn = s['connectivities'][:]
    mass_db = s['mass_db'][:]
    elem_mass = s['elem_mass'][:]
    has_lumped = 'lumped_mass' in s and s['lumped_mass'].shape != ()
    lm = s['lumped_mass'][:] if has_lumped else np.array([])
    lm_pos = s['lumped_mass_position'][:] if has_lumped else np.zeros((0, 3))
    lm_nodes = s['lumped_mass_nodes'][:] if has_lumped else np.array([], dtype=int)
    lm_inertia = s['lumped_mass_inertia'][:] if has_lumped else np.zeros((0, 3, 3))
    f.close()

    masses, points, own = [], [], []   # own: list of 3x3 inertias (A frame)

    for e in range(conn.shape[0]):
        n0, n1, nm = conn[e]            # SHARPy order: start, end, mid
        p0, p1, pm = pos[n0], pos[n1], pos[nm]
        length = np.linalg.norm(pm - p0) + np.linalg.norm(p1 - pm)
        blk = mass_db[elem_mass[e]]
        mu = blk[0, 0]
        pts = _quad_points(p0, p1, pm, n_sub)
        if elem_mass[e] in wing_mass_ids and wing_cg_offset_x:
            pts = pts + np.array([wing_cg_offset_x, 0.0, 0.0])
        m_slice = mu * length / n_sub
        for p in pts:
            masses.append(m_slice)
            points.append(p)
        if include_own_inertia:
            cab = crv2rotation(psi[e, 0, :])
            jdiag = np.diag(np.diag(blk[3:6, 3:6])) * length
            own.append(cab @ jdiag @ cab.T)

    for j in range(len(lm)):
        inode = lm_nodes[j]
        i_elem, i_local = node_master_elem[inode]
        cab = crv2rotation(psi[i_elem, i_local, :])
        masses.append(lm[j])
        points.append(pos[inode] + cab @ lm_pos[j])
        if include_own_inertia:
            own.append(cab @ lm_inertia[j] @ cab.T)

    m = np.array(masses)
    r = np.array(points)
    M = m.sum()
    cg = (m[:, None] * r).sum(axis=0) / M
    d = r - cg
    I = np.zeros((3, 3))
    for mi, di in zip(m, d):
        I += mi * (np.dot(di, di) * np.eye(3) - np.outer(di, di))
    for Io in own:
        I += Io
    return dict(mass=M, cg=cg, I=I, Ixx=I[0, 0], Iyy=I[1, 1], Izz=I[2, 2], Ixz=I[0, 2])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('h5file')
    ap.add_argument('--ts', type=int, default=0)
    ap.add_argument('--wing-cg-offset', type=float, default=0.0,
                    help='wing section CG aft of the elastic axis, m (cg_offset_main = 0.02)')
    ap.add_argument('--no-own-inertia', action='store_true')
    a = ap.parse_args()
    r = rigid_body_inertia(a.h5file, ts=a.ts, wing_cg_offset_x=a.wing_cg_offset,
                           include_own_inertia=not a.no_own_inertia)
    print(f"mass = {r['mass']:.4f} kg")
    print(f"cg   = {r['cg']}")
    print(f"Ixx  = {r['Ixx']:.4f} kg m^2   (roll)")
    print(f"Iyy  = {r['Iyy']:.4f} kg m^2   (pitch)")
    print(f"Izz  = {r['Izz']:.4f} kg m^2   (yaw)")
    print(f"Ixz  = {r['Ixz']:.4f} kg m^2")