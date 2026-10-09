#! /usr/bin/env python3
"""
llta_modal_analysis.py
======================
Post-processing of a SHARPy `Modal` run for the LLTA (free-free, linearised about trim,
structural modes only - no aerodynamics). Reads SHARPy's own output files; does NOT re-run SHARPy.

Inputs (set in the CONFIG block below):
  <case>.fem.h5                      - case input: connectivity, beam numbers, BCs, stiffness_db
  <Modal folder>/Mglobal.dat, Kglobal.dat, eigenvectors.dat, frequencies.dat, eigenvalues.dat
  <SaveData .h5>                     - trim-deformed geometry (pos, psi) at the linearisation point

Steps (numbers follow the LLTA modal-analysis process):
  Step 2  Matrix checks      - symmetry, positive definiteness, conditioning, independent re-solve
  Step 2  Mass properties    - total mass, CG, inertia read from the rigid-body block of Mglobal
  Step 3  Rigid-body modes   - rigid fraction c and rigid amplitudes a (see the reference doc
                               "LLTA Modal - Rigid-Body Mode Identification"), count, gap, orthogonality
  Step 4  Elastic modes      - strain energy by component x family (torsion / bending about y_B /
                               bending about z_B) from beam curvatures; kinetic energy by component;
                               symmetric/antisymmetric index; nodal points; coupling flag; label
  Step 5  Hand estimates     - matched to the auto-labelled modes
  Step 6  Aeroelastic   - AsymptoticStability eigenvalues: wake / flight-dynamics / elastic classification,
                          structural-mode tracking, frequency-compression correction, aerodynamic damping,
                          optional low-density reference for numerical damping
  Step 7  Outputs            - mode table (CSV + Markdown), checks log, mode-shape plots, VTK file list

Conventions used (checked against SHARPy source / data, Oct 2026):
  - Flexible DOFs: 6 per node with boundary_conditions != 1, in node order. Translations in the A-frame,
    rotations are Cartesian rotation vector (CRV) increments d_psi.
  - Rigid DOFs: last 10 = [v_A (3), omega_A (3), quaternion (4)]   (modal.py: num_rigid_dof = 10)
  - Rotation increment in the material frame: d_theta_B = T(psi) @ d_psi, with T = crv2tan(psi)
    (verified: T^T diag(j) T reproduces the nodal rotational blocks of Mglobal).
  - Stiffness diag = [EA, GA, GA, GJ, EI_y, EI_z]; curvature kappa_B = d(theta_B)/ds.
"""
import os
import glob
import numpy as np
import scipy.linalg as sl
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================================================
# CONFIG - edit for each run
# =====================================================================================
CASE_NAME = 'llta_modes_aerodynamics'
MODEL_TAG = 'M2 - section inertias'
CASE_ROUTE = os.path.dirname(os.path.realpath(__file__))        # folder holding <case>.fem.h5
OUTPUT_ROUTE = os.path.join(CASE_ROUTE, 'output', CASE_NAME)       # SHARPy output folder for this case
MODAL_DIR = None        # None = auto-find the folder containing Mglobal.dat under OUTPUT_ROUTE
SAVEDATA_FILE = None    # None = auto-find a .h5 file under OUTPUT_ROUTE/savedata
RESULTS_DIR = os.path.join(OUTPUT_ROUTE, 'modal_analysis_results')
MAKE_MODE_PLOTS = True
MAKE_RIGID_ARGAND = True     # Argand (complex-plane) diagrams of the rigid-body modes' state content
ARGAND_DECADES = 3           # Argand arrow length is log10 scaled over this many decades
ANALYSE_STABILITY = True     # read AsymptoticStability output (Step 6) if a stability folder exists
STABILITY_DIR = None         # None = OUTPUT_ROUTE/stability
REFERENCE_STABILITY_DIR = None  # stability folder of a LOW-DENSITY run of the same case (numerical damping),
                             # e.g. os.path.join(CASE_ROUTE, 'output', 'llta_modes_aerodynamics_low_rho', 'stability')
WAKE_STRUCT_FRACTION = 1e-11 # eigenvectors with less structural content than this are labelled wake (aero) states
FLIGHT_MIN_STRUCT_FRACTION = 1e-9  # heave/pitch-dominated eigenvalues below this are wake-ladder states
UNWARP_FREQUENCIES = True    # undo the discrete-time frequency compression: s = (2/dt) tanh(s_obs dt/2)
ARGAND_REFERENCE = 'auto'    # state set to 1 at 0 deg in each diagram: 'auto' = largest state of that mode,
                             # or a state name: 'u','v','w','p','q','r','phi','theta','psi'

# Beam numbers as written by generate_llta_trim.py
BEAM_NAMES = {0: 'R wing', 1: 'L wing', 2: 'fuselage', 3: 'fin', 4: 'R tail', 5: 'L tail'}
GROUP_OF = {'R wing': 'wing', 'L wing': 'wing', 'fuselage': 'fuselage', 'fin': 'fin',
            'R tail': 'tailplane', 'L tail': 'tailplane', 'hub': 'hub'}
# Physical meaning of bending about local y_B / z_B for each group (for labels only)
FAMILY_NAMES = {'wing':      {'torsion': 'torsion', 'bend_y': 'flapwise bending', 'bend_z': 'in-plane bending'},
                'tailplane': {'torsion': 'torsion', 'bend_y': 'flapwise bending', 'bend_z': 'in-plane bending'},
                'fuselage':  {'torsion': 'torsion', 'bend_y': 'vertical bending', 'bend_z': 'lateral bending'},
                'fin':       {'torsion': 'torsion', 'bend_y': 'lateral bending',  'bend_z': 'fore-aft bending'}}

N_RIGID = 10                 # rigid DOFs at the end of the DOF vector (6 velocities + 4 quaternion)
CHORD_REF = 0.225            # length used to weight rotations in the symmetry index (m)
EA_POSITION = 0.35           # elastic axis, fraction of chord from LE (for CG in %MAC)

# Pass/flag thresholds (working values - see the reference doc)
RIGID_C_MIN = 0.99           # rigid fraction for a rigid-body mode
ELASTIC_C_MAX = 0.01         # rigid fraction for an elastic mode
ELASTIC_ORTHO_TOL = 0.1      # |phi_i^T M phi_j| between elastic modes (SHARPy's non-symmetric solver)
COUPLED_THRESHOLD = 0.70     # largest strain-energy share below this -> 'coupled'
SYM_THRESHOLD = 0.90         # |symmetry index| above this -> S or A
COVERAGE_BAND = (0.85, 1.15)  # beam strain energy / (1/2 phi^T K phi) outside this -> flag the mode's label

# Reference values from the hand checks (Step 0 / Step 5)
REF_MASS = 2.372             # kg, hand sum of llta_geometry.py masses
REF_CG_MAC = (35.0, 43.0)    # Voltitude flying CG range, %MAC

# Hand estimates: matched to the n-th auto-labelled mode with the same group/family/symmetry
HAND_ESTIMATES = [
    dict(name='Wing 1st symmetric flapwise bending',     group='wing',     family='bend_y',  sym='S', order=1, f=2.43,  basis='beam FE with central mass'),
    dict(name='Wing 1st antisymmetric flapwise bending', group='wing',     family='bend_y',  sym='A', order=1, f=8.86,  basis='beam FE with tip masses'),
    dict(name='Wing 2nd symmetric flapwise bending',     group='wing',     family='bend_y',  sym='S', order=2, f=13.78, basis='beam FE'),
    dict(name='Wing 2nd antisymmetric flapwise bending', group='wing',     family='bend_y',  sym='A', order=2, f=28.56, basis='beam FE'),
    dict(name='Wing 3rd symmetric flapwise bending',     group='wing',     family='bend_y',  sym='S', order=3, f=36.27, basis='beam FE'),
    dict(name='Fuselage torsion (tail/fin roll)',        group='fuselage', family='torsion', sym='A', order=1, f=6.98,  basis='shaft + rigid tail/fin disc'),
    dict(name='Fuselage vertical bending',               group='fuselage', family='bend_y',  sym='S', order=1, f=8.54,  basis='cantilever + tip mass'),
    dict(name='Fuselage lateral bending',                group='fuselage', family='bend_z',  sym='A', order=1, f=8.54,  basis='cantilever + tip mass'),
    dict(name='Wing 1st symmetric torsion',              group='wing',     family='torsion', sym='S', order=1, f=16.94, basis='clamped shaft, uncoupled'),
    dict(name='Wing 1st antisymmetric torsion',          group='wing',     family='torsion', sym='A', order=1, f=16.94, basis='clamped shaft, uncoupled'),
    dict(name='Wing symmetric in-plane bending',         group='wing',     family='bend_z',  sym='S', order=1, f=48.1,  basis='beam FE, EI_z placeholder'),
]

# Plot colours (categorical slots 1-7, fixed order)
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#8a63d2']
INK, MUTED, GRID = '#1f2937', '#6b7280', '#e5e7eb'


TRIM_QUAT = np.array([1.0, 0.0, 0.0, 0.0])   # A-frame attitude at trim (overwritten from SaveData 'quat')


# =====================================================================================
# Rotation helpers (same formulae as sharpy.utils.algebra)
# =====================================================================================
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def crv2tan(psi):
    n = np.linalg.norm(psi)
    S = skew(psi)
    if n < 1e-8:
        return np.eye(3) - 0.5 * S + S @ S / 6.0
    k1 = (np.cos(n) - 1.0) / n**2
    k2 = (1.0 - np.sin(n) / n) / n**2
    return np.eye(3) + k1 * S + k2 * S @ S


def crv2rotation(psi):
    n = np.linalg.norm(psi)
    if n < 1e-15:
        return np.eye(3)
    k = psi / n
    K = skew(k)
    return np.eye(3) + np.sin(n) * K + (1 - np.cos(n)) * K @ K


def rotation2crv(C):
    ang = np.arccos(np.clip((np.trace(C) - 1) / 2, -1, 1))
    if ang < 1e-12:
        return np.zeros(3)
    if np.pi - ang < 1e-6:                       # 180 deg: axis from symmetric part
        B = (C + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(B), 0, None))
        i = np.argmax(k)
        k = B[:, i] / np.sqrt(B[i, i])
        return ang * k / np.linalg.norm(k)
    k = np.array([C[2, 1] - C[1, 2], C[0, 2] - C[2, 0], C[1, 0] - C[0, 1]]) / (2 * np.sin(ang))
    return ang * k


# =====================================================================================
# Loading
# =====================================================================================
def find_one(root, pattern, what):
    hits = sorted(glob.glob(os.path.join(root, '**', pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(f'No {what} ({pattern}) found under {root}')
    return hits


def load_case():
    fem = h5py.File(os.path.join(CASE_ROUTE, CASE_NAME + '.fem.h5'), 'r')
    case = {k: fem[k][()] for k in ['coordinates', 'connectivities', 'beam_number', 'boundary_conditions',
                                    'stiffness_db', 'elem_stiffness', 'frame_of_reference_delta']}
    fem.close()
    return case


def load_modal():
    mdir = MODAL_DIR or os.path.dirname(find_one(OUTPUT_ROUTE, 'Mglobal.dat', 'Modal output')[0])
    rd = lambda f: np.loadtxt(os.path.join(mdir, f))
    out = dict(dir=mdir, M=rd('Mglobal.dat'), K=rd('Kglobal.dat'), V=rd('eigenvectors.dat'),
               w=rd('frequencies.dat'), lam=rd('eigenvalues.dat'))
    if out['V'].ndim == 1:
        out['V'] = out['V'][:, None]
    return out


def load_trim_geometry(case):
    """Trim-deformed pos (A-frame) and psi from SaveData. Falls back to the undeformed fem.h5 geometry."""
    n_node = case['coordinates'].shape[0]
    n_elem = case['connectivities'].shape[0]
    try:
        files = [SAVEDATA_FILE] if SAVEDATA_FILE else find_one(os.path.join(OUTPUT_ROUTE, 'savedata'), '*.h5', 'SaveData file')
        cands = []
        with h5py.File(files[0], 'r') as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and name.split('/')[-1] == 'pos' and obj.shape == (n_node, 3):
                    parent = name.rsplit('/', 1)[0]
                    if parent + '/psi' in f and f[parent + '/psi'].shape == (n_elem, 3, 3):
                        cands.append(parent)
            f.visititems(visit)
            if not cands:
                raise KeyError('no (pos, psi) pair with the case dimensions')
            # Prefer the last timestep_info entry; ini_info only if nothing else
            def key(p):
                digits = ''.join(c if c.isdigit() else ' ' for c in p.split('/')[-1]).split()
                return ('ini_info' not in p, 'timestep_info' in p, int(digits[-1]) if digits else -1)
            best = sorted(cands, key=key)[-1]
            pos, psi = f[best + '/pos'][()], f[best + '/psi'][()]
            global TRIM_QUAT
            if best + '/quat' in f:
                TRIM_QUAT = np.array(f[best + '/quat'][()], dtype=float)
        return pos, psi, f'SaveData {os.path.basename(files[0])}:/{best}', cands
    except Exception as err:
        pos = case['coordinates'].astype(float)
        psi = np.zeros((n_elem, 3, 3))
        for e, conn in enumerate(case['connectivities']):
            xb = pos[conn[1]] - pos[conn[0]]
            xb /= np.linalg.norm(xb)
            for j in range(3):
                fd = case['frame_of_reference_delta'][e, j]
                yb = fd - (fd @ xb) * xb
                yb /= np.linalg.norm(yb)
                psi[e, j] = rotation2crv(np.column_stack([xb, yb, np.cross(xb, yb)]))
        return pos, psi, f'FALLBACK undeformed fem.h5 geometry ({err})', []


# =====================================================================================
# Model bookkeeping
# =====================================================================================
class Model:
    def __init__(self, case, modal, pos, psi):
        self.case, self.pos, self.psi = case, pos, psi
        self.M, self.K, self.V, self.w, self.lam = modal['M'], modal['K'], modal['V'], modal['w'], modal['lam']
        self.n = self.M.shape[0]
        self.nf = self.n - N_RIGID
        bc = case['boundary_conditions']
        conn = case['connectivities']
        n_node = len(bc)
        # vdof: flexible DOF block of each node (-1 if clamped to the A-frame)
        self.vdof = -np.ones(n_node, dtype=int)
        k = 0
        for nd in range(n_node):
            if bc[nd] != 1:
                self.vdof[nd] = k
                k += 1
        assert 6 * k == self.nf, f'DOF count mismatch: 6 x {k} free nodes != {self.nf}'
        self.hub_nodes = [nd for nd in range(n_node) if bc[nd] == 1]
        # master element / local index of each node (first element in which it appears)
        self.master = {}
        self.node_comp = {}
        for e, c in enumerate(conn):
            for j, nd in enumerate(c):
                self.master.setdefault(nd, (e, j))
                self.node_comp.setdefault(nd, BEAM_NAMES.get(case['beam_number'][e], f'beam {case["beam_number"][e]}'))
        for nd in self.hub_nodes:
            self.node_comp[nd] = 'hub'
        # nodal translational masses from the flexible diagonal; hub = total - sum(flexible)
        self.mnode = {nd: self.M[6 * self.vdof[nd], 6 * self.vdof[nd]] for nd in range(n_node) if self.vdof[nd] >= 0}
        self.m_total = self.M[self.nf, self.nf]
        self.m_hub = self.m_total - sum(self.mnode.values())
        # mirror pairs (y -> -y) for symmetry
        self.mirror = {}
        for a in range(n_node):
            if pos[a, 1] > 1e-6:
                d = np.linalg.norm(pos - pos[a] * np.array([1, -1, 1]), axis=1)
                b = int(np.argmin(d))
                if d[b] < 1e-3 * max(1.0, np.abs(pos).max()):
                    self.mirror[a] = b

    # --- per-mode quantities -------------------------------------------------------
    def dpsi(self, phi, nd):
        return np.zeros(3) if self.vdof[nd] < 0 else phi[6 * self.vdof[nd] + 3: 6 * self.vdof[nd] + 6]

    def du_rel(self, phi, nd):
        return np.zeros(3) if self.vdof[nd] < 0 else phi[6 * self.vdof[nd]: 6 * self.vdof[nd] + 3]

    def frame(self, phi):
        return phi[self.nf:self.nf + 3], phi[self.nf + 3:self.nf + 6]

    def u_abs(self, phi, nd):
        at, ar = self.frame(phi)
        return self.du_rel(phi, nd) + at + np.cross(ar, self.pos[nd])

    def theta_A_rel(self, phi, nd):
        # rotation DOFs are the CRV of the node's MASTER element: d_theta_A = C_AB(psi_m) T(psi_m) d_psi
        em, jm = self.master[nd]
        pm = self.psi[em, jm]
        return crv2rotation(pm) @ crv2tan(pm) @ self.dpsi(phi, nd)

    def theta_B(self, phi, e, j):
        # elastic rotation of node (e, j) expressed in element e's own material frame
        nd = self.case['connectivities'][e, j]
        return crv2rotation(self.psi[e, j]).T @ self.theta_A_rel(phi, nd)

    def theta_A(self, phi, nd):
        _, ar = self.frame(phi)
        return self.theta_A_rel(phi, nd) + ar


# =====================================================================================
# Step 2 - matrix checks and mass properties
# =====================================================================================
def matrix_checks(mdl, log):
    M, K, nf = mdl.M, mdl.K, mdl.nf
    asym = lambda A: np.abs(A - A.T).max() / max(np.abs(A).max(), 1e-30)
    Mff, Kff = M[:nf, :nf], K[:nf, :nf]
    eM = np.linalg.eigvalsh((Mff + Mff.T) / 2)
    eK = np.linalg.eigvals(Kff).real
    log.check('M symmetric (relative asymmetry)', asym(M) < 1e-6, f'{asym(M):.2e}')
    log.check('K_ff symmetric (relative asymmetry)', asym(Kff) < 1e-6, f'{asym(Kff):.2e}',
              note='a small asymmetry is expected when K is linearised about a loaded trim state')
    log.check('M_ff positive definite', eM.min() > 0, f'min eig {eM.min():.3e}')
    log.check('K_ff positive definite (node 0 fixed to A-frame)', eK.min() > 0, f'min eig {eK.min():.3e}')
    cM, cK = np.linalg.cond(Mff), np.linalg.cond(Kff)
    log.check('Condition numbers below 1e12', max(cM, cK) < 1e12, f'cond(M_ff) {cM:.2e}, cond(K_ff) {cK:.2e}')
    Krig = np.abs(K[:, nf:]).max() + np.abs(K[nf:, :]).max()
    log.info(f'K rigid rows/columns max |entry| = {Krig:.2e} (0 => no stiffness on rigid DOFs: gravity off in Modal assembly)')
    q = slice(nf + 6, nf + 10)
    log.info(f'Quaternion DOFs: max coupling to other DOFs in M = {np.abs(M[q, :nf + 6]).max():.2e}, in K = {np.abs(K[q, :]).max():.2e}')

    # mass properties from the rigid block (as modalutils.cg)
    Mrr = M[nf:nf + 6, nf:nf + 6]
    m = Mrr[0, 0]
    cg = -np.array([Mrr[2, 4], Mrr[0, 5], Mrr[1, 3]]) / m
    I0 = Mrr[3:, 3:]
    Icg = I0 - m * (cg @ cg * np.eye(3) - np.outer(cg, cg))
    mac = (EA_POSITION * CHORD_REF + cg[0]) / CHORD_REF * 100
    log.check('Total mass vs hand sum', abs(m - REF_MASS) / REF_MASS < 0.01, f'{m:.4f} kg (hand {REF_MASS} kg)')
    log.check('CG within Voltitude range', REF_CG_MAC[0] <= mac <= REF_CG_MAC[1],
              f'x {cg[0]*1e3:.1f} mm aft of node 0 = {mac:.1f}% MAC; y {cg[1]*1e3:.2f} mm; z {cg[2]*1e3:.1f} mm')
    log.info('Inertia about CG (A-frame axes) [kg m^2]: Ixx {:.4f}, Iyy {:.4f}, Izz {:.4f}, Ixz {:.4f}'.format(
        Icg[0, 0], Icg[1, 1], Icg[2, 2], Icg[0, 2]))
    return dict(mass=m, cg=cg, cg_mac=mac, I_cg=Icg)


def resolve_symmetric(mdl, log):
    """Independent re-solve with symmetrised M, K (quaternion DOFs removed); MAC against SHARPy."""
    nf = mdl.nf
    idx = np.r_[0:nf + 6]
    Ms = ((mdl.M + mdl.M.T) / 2)[np.ix_(idx, idx)]
    Ks = ((mdl.K + mdl.K.T) / 2)[np.ix_(idx, idx)]
    lam, U = sl.eigh(Ks, Ms)
    U_full = np.zeros((mdl.n, len(idx)))
    U_full[idx] = U
    f_mine = np.sqrt(np.abs(lam)) / (2 * np.pi)
    el_sh = [i for i in range(mdl.V.shape[1]) if mdl.w[i] > 1e-6]
    el_mine = [i for i in range(len(lam)) if f_mine[i] > 0.1][:len(el_sh)]
    res = []
    for i, k in zip(el_sh, el_mine):
        a, b = mdl.V[:, i], U_full[:, k]
        mac = (a @ mdl.M @ b) ** 2 / ((a @ mdl.M @ a) * (b @ mdl.M @ b))
        res.append((i, mdl.w[i] / 2 / np.pi, f_mine[k], mac))
    dfmax = max(abs(r[1] - r[2]) / r[1] for r in res)
    macmin = min(r[3] for r in res)
    log.check('Independent symmetric re-solve reproduces SHARPy frequencies (<0.1%)', dfmax < 1e-3,
              f'max rel. diff {dfmax:.2e}; min MAC {macmin:.4f}')
    nz = f_mine[:6]
    log.info(f'Rigid-mode noise level of the symmetric re-solve: max {nz.max():.2e} Hz '
             f'(gap to first elastic: {nz.max() / res[0][1]:.1e})')
    return res


# =====================================================================================
# Step 3 - rigid-body modes
# =====================================================================================
RIGID_NAMES = ['surge (x)', 'sway (y)', 'heave (z)', 'roll', 'pitch', 'yaw',
               'quaternion q0', 'quaternion q1', 'quaternion q2', 'quaternion q3']


def rigid_analysis(mdl, log):
    M, V, nf = mdl.M, mdl.V, mdl.nf
    R = np.zeros((mdl.n, N_RIGID))
    R[nf:, :] = np.eye(N_RIGID)
    Mrr = R.T @ M @ R
    out = []
    for i in range(V.shape[1]):
        p = V[:, i]
        a = np.linalg.solve(Mrr, R.T @ M @ p)
        c = (a @ Mrr @ a) / (p @ M @ p)
        out.append(dict(c=c, a=a, rigid_name=RIGID_NAMES[int(np.argmax(np.abs(a)))] if c > 0.5 else ''))
    is_rigid = np.array([o['c'] >= RIGID_C_MIN for o in out])
    n_rig = int(is_rigid.sum())
    log.check(f'{N_RIGID} rigid-body modes with c >= {RIGID_C_MIN}', n_rig == N_RIGID, f'{n_rig} found')
    log.check('Rigid modes are the lowest modes', np.all(is_rigid[:n_rig]), '')
    el = ~is_rigid
    cmax = max([o['c'] for o, e in zip(out, el) if e], default=0)
    log.check(f'All elastic modes have c <= {ELASTIC_C_MAX}', cmax <= ELASTIC_C_MAX, f'max c = {cmax:.2e}')
    G = V.T @ M @ V
    log.check('Mass normalisation phi^T M phi = 1', np.abs(np.diag(G) - 1).max() < 1e-6, f'max dev {np.abs(np.diag(G) - 1).max():.1e}')
    Ge = G[np.ix_(el, el)] - np.eye(el.sum())
    log.check(f'Elastic modes mutually M-orthogonal (< {ELASTIC_ORTHO_TOL})', np.abs(Ge).max() < ELASTIC_ORTHO_TOL,
              f'max |phi_i^T M phi_j| = {np.abs(Ge).max():.3f}',
              note='non-zero values come from SHARPy\'s non-symmetric solver acting on the slightly asymmetric K')
    Gre = G[np.ix_(~el, el)]
    log.check('Elastic modes carry no rigid-body momentum', np.abs(Gre).max() < 1e-4, f'max |phi_r^T M phi_e| = {np.abs(Gre).max():.1e}')
    log.info('Rigid modes are referenced to node 0 (A-frame); their mutual coupling reflects the CG offset. '
             'Re-run Modal with rigid_modes_ppal_axes: True to see them about the CG.')
    return out, is_rigid


# =====================================================================================
# Step 3b - Argand diagrams of rigid-body mode content (CG / principal axes)
# =====================================================================================
ARGAND_STATES = ['u', 'v', 'w', 'p', 'q', 'r', 'phi', 'theta', 'psi']
ARGAND_UNITS = {'u': 'm/s', 'v': 'm/s', 'w': 'm/s', 'p': 'rad/s', 'q': 'rad/s', 'r': 'rad/s',
                'phi': 'rad', 'theta': 'rad', 'psi': 'rad'}


def rigid_states_cg(mdl, phi, mp):
    """Rigid-body states of one eigenvector, referred to the CG and the principal axes of inertia.

    phi : eigenvector (real or complex). Rigid DOFs (A-frame, node 0): v_A = phi[nf:nf+3],
          omega_A = phi[nf+3:nf+6], quaternion perturbation dq = phi[nf+6:nf+10].
    Translation at the CG:  v_cg = v_A + omega_A x r_cg      (r_cg from node 0, A-frame)
    Attitude from the quaternion (linearised about the trim quaternion q = [q0, qv]):
          d_theta_A = 2 (q0 dqv - dq0 qv - qv x dqv)
    All vectors are then rotated into principal axes: x_P = R_pa^T x_A.
    """
    nf = mdl.nf
    vA, wA, dq = phi[nf:nf + 3], phi[nf + 3:nf + 6], phi[nf + 6:nf + 10]
    q0, qv = TRIM_QUAT[0], TRIM_QUAT[1:]
    att = 2.0 * (q0 * dq[1:] - dq[0] * qv - np.cross(qv, dq[1:]))
    vcg = vA + np.cross(wA, mp['cg'])
    Rt = mp['R_pa'].T
    return np.concatenate([Rt @ vcg, Rt @ wA, Rt @ att])


def principal_axes(mp):
    """Principal axes of the CG inertia tensor, ordered and signed to stay closest to A-frame x, y, z."""
    lam, R = np.linalg.eigh(mp['I_cg'])
    order = [int(np.argmax(np.abs(R[k, :]))) for k in range(3)]
    if len(set(order)) < 3:
        order = [0, 1, 2]
    R = R[:, order]
    R = R * np.sign(np.diag(R))
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return R


def argand_rigid(mdl, rig, is_rigid, mp, outdir, log):
    mp = dict(mp)
    mp['R_pa'] = principal_axes(mp)
    tilt = np.degrees(np.arctan2(mp['R_pa'][2, 0], mp['R_pa'][0, 0]))
    log.info(f'Argand: states referred to CG {np.round(mp["cg"] * 1e3, 1)} mm from node 0, principal axes '
             f'(x_P tilted {tilt:+.2f} deg from x_A about y); trim quaternion {np.round(TRIM_QUAT, 5)}')
    idx = [i for i in range(mdl.V.shape[1]) if is_rigid[i]]
    rows, panels = [], []
    for i in idx:
        z = rigid_states_cg(mdl, mdl.V[:, i].astype(complex), mp)
        if np.abs(z).max() < 1e-12:
            panels.append((i, None, None, None)); rows.append((i, 'none', z)); continue
        k = int(np.argmax(np.abs(z))) if ARGAND_REFERENCE == 'auto' else ARGAND_STATES.index(ARGAND_REFERENCE)
        if abs(z[k]) < 1e-12:                         # requested reference absent in this mode
            k = int(np.argmax(np.abs(z)))
        zn = z / z[k]                                 # reference state -> 1 at 0 deg
        # quaternion modes: how much of |dq| is a real rotation (the rest only changes the quaternion norm)
        dq = mdl.V[mdl.nf + 6:mdl.nf + 10, i]
        qshare = (np.linalg.norm(z[6:9]) / 2) / np.linalg.norm(dq) if np.linalg.norm(dq) > 0.5 * np.linalg.norm(mdl.V[mdl.nf:, i]) else None
        panels.append((i, zn, k, qshare)); rows.append((i, ARGAND_STATES[k], zn))

    # CSV of normalised content (magnitude and phase)
    with open(os.path.join(outdir, 'rigid_mode_argand.csv'), 'w') as f:
        f.write('mode,reference_state,' + ','.join(f'{s}_mag,{s}_phase_deg' for s in ARGAND_STATES) + '\n')
        for i, ref, zn in rows:
            f.write(f'{i},{ref},' + ','.join(f'{abs(c):.6e},{np.degrees(np.angle(c)):.1f}' for c in zn) + '\n')

    # Figure: one Argand panel per rigid mode
    ncol = 5
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.8 * nrow + 1.2))
    axs = np.atleast_2d(axs)
    colours = dict(zip(ARGAND_STATES, COL + ['#8c564b', '#4b5563']))
    for ax, (i, zn, k, qshare) in zip(axs.flat, panels):
        ax.set_aspect('equal')
        ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
        t = np.linspace(0, 2 * np.pi, 200)
        for dec in range(ARGAND_DECADES):              # log-radius rings: 1, 0.1, 0.01, ...
            rr = 1 - dec / ARGAND_DECADES
            ax.plot(rr * np.cos(t), rr * np.sin(t), color=GRID, lw=0.8)
            ax.text(rr * np.cos(np.radians(60)), rr * np.sin(np.radians(60)), f'{10.0 ** -dec:g}', fontsize=6.5, color=MUTED)
        ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(1.42, 0.03, 'Re', fontsize=7, color=MUTED, ha='right', va='bottom')
        ax.text(0.03, 1.42, 'Im', fontsize=7, color=MUTED, ha='left', va='top')
        for sp in ax.spines.values():
            sp.set_color(GRID)
        title = f'Mode {i} - {rig[i]["rigid_name"]}'
        if zn is None:
            ax.text(0, 0, 'no physical rigid motion\n(quaternion normalisation\ncomponent)', ha='center',
                    va='center', fontsize=8, color=MUTED)
            ax.set_title(title, fontsize=9, color=INK, loc='left')
            continue
        shown = sorted([j for j in range(9) if abs(zn[j]) > 10.0 ** -ARGAND_DECADES], key=lambda j: -abs(zn[j]))
        labels_y = {}
        for j in shown:
            c0 = zn[j]
            rad = 1 + np.log10(abs(c0)) / ARGAND_DECADES  # log radius: 1 -> 1, 10^-decades -> 0
            c = rad * np.exp(1j * np.angle(c0))
            ax.annotate('', xy=(c.real, c.imag), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='-|>', color=colours[ARGAND_STATES[j]], lw=2, mutation_scale=10))
            # stagger labels of arrows that share a direction (real modes lie on the real axis)
            key = (round(np.angle(c), 2))
            n = labels_y.get(key, 0); labels_y[key] = n + 1
            ang = np.angle(c)
            ax.text(c.real + 0.06 * np.cos(ang), c.imag + 0.06 * np.sin(ang) + 0.17 * n,
                    f'{ARGAND_STATES[j]} {abs(c0):.3g}', fontsize=7.5, color=INK,
                    ha='left' if c.real >= 0 else 'right', va='bottom')
        sub = f'ref: {ARGAND_STATES[k]} = 1 at 0 deg'
        if qshare is not None:
            sub += f'; rotation = {qshare:.1%} of |dq|'
        ax.set_title(title + '\n' + sub, fontsize=9, color=INK, loc='left')
    for ax in list(axs.flat)[len(panels):]:
        ax.axis('off')
    fig.suptitle(f'{CASE_NAME} ({MODEL_TAG}): rigid-body mode content, CG / principal axes '
                 f'(units: u,v,w m/s; p,q,r rad/s; phi,theta,psi rad)\nArrow length on a log scale: rings at 1, 0.1, 0.01 relative to the reference state; components below '
                 f'{10.0 ** -ARGAND_DECADES:g} hidden',
                 fontsize=10.5, color=INK, x=0.01, ha='left')
    fig.text(0.01, 0.005, 'Undamped Modal eigenvectors are real, so every arrow lies at 0 or 180 deg; phase content appears only for '
             'complex modes (e.g. AsymptoticStability), which this function accepts unchanged.\nSHARPy rotations are about node 0, '
             'so referring them to the CG adds translation (e.g. pitch -> u). Quaternion modes: "rotation %" is the share of the '
             'perturbation that is a real attitude change; the rest only changes the quaternion norm.',
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92), h_pad=2.0)
    fig.savefig(os.path.join(outdir, 'rigid_mode_argand.png'), dpi=150)
    plt.close(fig)
    for i, ref, zn in rows:
        if ref == 'none':
            log.info(f'Argand mode {i}: no physical rigid motion (quaternion normalisation component)')
        else:
            others = ', '.join(f'{ARGAND_STATES[j]} {zn[j].real:+.3g}' for j in range(9)
                               if abs(zn[j]) > 1e-3 and ARGAND_STATES[j] != ref)
            log.info(f'Argand mode {i}: {ref} = 1' + (f'; also {others}' if others else '; pure'))


# =====================================================================================
# Step 4 - elastic mode classification
# =====================================================================================
def strain_energy(mdl, phi):
    """Beam strain energy split by component x family, from d(theta_B)/ds over half-elements."""
    conn, st, es = mdl.case['connectivities'], mdl.case['stiffness_db'], mdl.case['elem_stiffness']
    U = {}
    for e, c in enumerate(conn):
        comp = mdl.node_comp.get(c[2], 'hub')
        D = np.diag(st[es[e]])[3:6]                              # Get GJ, EI_y, EI_z for the element. NOTE: only diagonal is used currently - ignores coupling!
        th = [mdl.theta_B(phi, e, j) for j in range(3)]          # For each node (j), get elastic rotation at that node
        for (ja, jb) in [(0, 2), (2, 1)]:                        # start->mid, mid->end
            L = np.linalg.norm(mdl.pos[c[jb]] - mdl.pos[c[ja]])  # half-element length, from trim-deformed node position (pos)
            kap = (th[jb] - th[ja]) / L                          # curvature - change in rotation across the half element, divided by length
            for fam, k in zip(['torsion', 'bend_y', 'bend_z'], range(3)):
                key = (GROUP_OF[comp], fam)
                U[key] = U.get(key, 0.0) + 0.5 * L * D[k] * kap[k] ** 2
    return U


def kinetic_by_component(mdl, phi):
    T = {}
    for nd in range(len(mdl.case['boundary_conditions'])):
        comp = mdl.node_comp.get(nd, 'hub')
        m = mdl.m_hub if comp == 'hub' else mdl.mnode[nd]
        u = mdl.u_abs(phi, nd)
        T[comp] = T.get(comp, 0.0) + m * u @ u
        if mdl.vdof[nd] >= 0:
            k = 6 * mdl.vdof[nd] + 3
            r = phi[k:k + 3]
            T[comp] += r @ mdl.M[k:k + 3, k:k + 3] @ r
    s = sum(T.values())
    return {k: v / s for k, v in T.items()}


def symmetry_index(mdl, phi, comps=('R wing',)):
    Pu, Pr = np.diag([1, -1, 1]), np.diag([-1, 1, -1])      # mirror y -> -y for vectors / pseudovectors
    num = nr = nl = 0.0
    for a, b in mdl.mirror.items():
        if mdl.node_comp.get(a) not in comps:
            continue
        va = np.r_[mdl.u_abs(phi, a), CHORD_REF * mdl.theta_A(phi, a)]
        vb = np.r_[Pu @ mdl.u_abs(phi, b), CHORD_REF * (Pr @ mdl.theta_A(phi, b))]
        num += va @ vb; nr += va @ va; nl += vb @ vb
    return num / np.sqrt(max(nr * nl, 1e-30))


def wing_line(mdl, comp='R wing'):
    nodes = [nd for nd in range(len(mdl.pos)) if mdl.node_comp.get(nd) in (comp, 'hub')]
    return sorted(nodes, key=lambda nd: abs(mdl.pos[nd, 1]))


def nodal_points(mdl, phi, family):
    nodes = wing_line(mdl)
    if family == 'torsion':
        vals = np.array([mdl.theta_B(phi, *mdl.master[nd])[0] for nd in nodes])
    else:
        vals = np.array([mdl.u_abs(phi, nd)[2] for nd in nodes])
    big = vals[np.abs(vals) > 0.05 * np.abs(vals).max()]
    return int(np.sum(np.sign(big[1:]) != np.sign(big[:-1])))


def classify(mdl, rig, is_rigid, log):
    rows = []
    count = {}
    for i in range(mdl.V.shape[1]):
        f = mdl.w[i] / (2 * np.pi)
        if is_rigid[i]:
            rows.append(dict(mode=i, f=f, kind='rigid', label=f'Rigid: {rig[i]["rigid_name"]}', c=rig[i]['c']))
            continue
        phi = mdl.V[:, i]
        U = strain_energy(mdl, phi)
        Ubeam = sum(U.values())
        cover = Ubeam / (0.5 * mdl.lam[i])
        Uf = {k: v / Ubeam for k, v in U.items()}
        ranked = sorted(Uf.items(), key=lambda kv: -kv[1])
        (g1, fam1), s1 = ranked[0]
        Tk = kinetic_by_component(mdl, phi)
        S_w = symmetry_index(mdl, phi, ('R wing',))
        S_t = symmetry_index(mdl, phi, ('R tail',))
        S_ref = S_w if g1 in ('wing', 'hub') else (S_t if g1 == 'tailplane' else S_w)
        sym = 'S' if S_ref > SYM_THRESHOLD else ('A' if S_ref < -SYM_THRESHOLD else 'mixed')
        key = (g1, fam1, sym)
        count[key] = count.get(key, 0) + 1
        coupled = s1 < COUPLED_THRESHOLD
        name = f'{g1} {FAMILY_NAMES.get(g1, {}).get(fam1, fam1)}'
        label = f'{name} ({sym}, #{count[key]})'
        if coupled:
            (g2, fam2), s2 = ranked[1]
            label += f' + {g2} {FAMILY_NAMES.get(g2, {}).get(fam2, fam2)} [coupled]'
        np_ = nodal_points(mdl, phi, fam1) if g1 == 'wing' else ''
        rows.append(dict(mode=i, f=f, kind='elastic', label=label, group=g1, family=fam1, sym=sym, order=count[key],
                         coupled=coupled, c=rig[i]['c'], S_wing=S_w, S_tail=S_t, nodal=np_, coverage=cover,
                         U=Uf, T=Tk))
        if not (COVERAGE_BAND[0] <= cover <= COVERAGE_BAND[1]):
            rows[-1]['label'] += ' [?]'
            log.flag(f'Mode {i}: beam-curvature strain energy = {cover:.2f} x (1/2 phi^T K phi) - family split unreliable '
                     f'(check geometry source; with undeformed psi, rotations leak into the stiff EI_z term)')
    return rows


def match_hand(rows, log):
    for h in HAND_ESTIMATES:
        hit = [r for r in rows if r['kind'] == 'elastic' and r['group'] == h['group'] and r['family'] == h['family']
               and r['sym'] == h['sym'] and r['order'] == h['order']]
        if hit:
            r = hit[0]
            r.setdefault('hand', []).append(h)
            log.info(f'Hand estimate "{h["name"]}" {h["f"]:.2f} Hz <-> mode {r["mode"]} {r["f"]:.3f} Hz '
                     f'({(r["f"] - h["f"]) / h["f"]:+.1%}; {h["basis"]})')
        else:
            log.info(f'Hand estimate "{h["name"]}" {h["f"]:.2f} Hz: no mode with that label in this run')


# =====================================================================================
# Outputs
# =====================================================================================
class Log:
    def __init__(self):
        self.lines = []

    def check(self, what, ok, detail, note=''):
        s = f'[{"PASS" if ok else "FLAG"}] {what}: {detail}' + (f'  ({note})' if note else '')
        self.lines.append(s); print(s)

    def flag(self, s):
        self.lines.append('[FLAG] ' + s); print('[FLAG] ' + s)

    def info(self, s):
        self.lines.append('[info] ' + s); print('[info] ' + s)


def write_tables(rows, outdir):
    fams = sorted({k for r in rows if r['kind'] == 'elastic' for k in r['U']})
    comps = sorted({k for r in rows if r['kind'] == 'elastic' for k in r['T']})
    hdr = ['mode', 'freq_Hz', 'label', 'symmetry', 'coupled', 'rigid_fraction_c', 'S_index_wing', 'S_index_tail',
           'wing_nodal_points', 'strain_energy_coverage', 'hand_estimate', 'hand_Hz', 'diff_pct'] + \
          [f'SE_{g}_{f}' for g, f in fams] + [f'KE_{c}' for c in comps]
    lines = [','.join(hdr)]
    md = ['| Mode | f (Hz) | Label | Sym | Top strain energy | Top kinetic energy | Hand estimate |', '|---|---|---|---|---|---|---|']
    for r in rows:
        if r['kind'] == 'rigid':
            lines.append(','.join([str(r['mode']), f'{r["f"]:.6f}', r['label'], '', '', f'{r["c"]:.4f}'] + [''] * (len(hdr) - 6)))
            md.append(f'| {r["mode"]} | 0 | {r["label"]} | - | - | - | - |')
            continue
        h = r.get('hand', [])
        hname = '; '.join(x['name'] for x in h)
        hf = '; '.join(f'{x["f"]:.2f}' for x in h)
        hd = '; '.join(f'{(r["f"] - x["f"]) / x["f"] * 100:+.1f}' for x in h)
        lines.append(','.join([str(r['mode']), f'{r["f"]:.6f}', '"' + r['label'] + '"', r['sym'], str(r['coupled']),
                               f'{r["c"]:.2e}', f'{r["S_wing"]:+.3f}', f'{r["S_tail"]:+.3f}', str(r['nodal']),
                               f'{r["coverage"]:.3f}', '"' + hname + '"', hf, hd] +
                              [f'{r["U"].get(k, 0):.4f}' for k in fams] + [f'{r["T"].get(c, 0):.4f}' for c in comps]))
        topU = ', '.join(f'{g} {FAMILY_NAMES.get(g, {}).get(f, f)} {v:.0%}' for (g, f), v in
                         sorted(r['U'].items(), key=lambda kv: -kv[1])[:2] if v > 0.05)
        topT = ', '.join(f'{c} {v:.0%}' for c, v in sorted(r['T'].items(), key=lambda kv: -kv[1])[:3] if v > 0.05)
        hstr = '; '.join(f'{x["f"]:.2f} ({(r["f"] - x["f"]) / x["f"]:+.1%})' for x in h) or '-'
        md.append(f'| {r["mode"]} | {r["f"]:.3f} | {r["label"]} | {r["sym"]} | {topU} | {topT} | {hstr} |')
    open(os.path.join(outdir, 'mode_table.csv'), 'w').write('\n'.join(lines) + '\n')
    open(os.path.join(outdir, 'mode_table.md'), 'w').write('\n'.join(md) + '\n')


def plot_modes(mdl, rows, outdir):
    pdir = os.path.join(outdir, 'mode_shapes')
    os.makedirs(pdir, exist_ok=True)
    wing_nodes = sorted([nd for nd in range(len(mdl.pos)) if mdl.node_comp.get(nd) in ('R wing', 'L wing', 'hub')],
                        key=lambda nd: mdl.pos[nd, 1])
    fus_nodes = sorted([nd for nd in range(len(mdl.pos)) if mdl.node_comp.get(nd) in ('fuselage', 'hub')],
                       key=lambda nd: mdl.pos[nd, 0])
    fin_nodes = sorted([nd for nd in range(len(mdl.pos)) if mdl.node_comp.get(nd) == 'fin'], key=lambda nd: mdl.pos[nd, 2])
    tail_nodes = sorted([nd for nd in range(len(mdl.pos)) if mdl.node_comp.get(nd) in ('R tail', 'L tail')],
                        key=lambda nd: mdl.pos[nd, 1])
    for r in rows:
        if r['kind'] != 'elastic':
            continue
        phi = mdl.V[:, r['mode']]
        U = {nd: mdl.u_abs(phi, nd) for nd in range(len(mdl.pos))}
        scale = max(np.linalg.norm(u) for u in U.values())
        # nose-up twist positive on both wings: x_B points outboard, so flip sign where x_B . y_A < 0
        tw = np.array([mdl.theta_B(phi, *mdl.master[nd])[0] *
                       np.sign(crv2rotation(mdl.psi[mdl.master[nd]])[1, 0] or 1.0) for nd in wing_nodes])
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        y = mdl.pos[wing_nodes, 1]
        for k, (lab, col) in enumerate([('x (fore-aft)', COL[1]), ('y (spanwise)', COL[3]), ('z (vertical)', COL[0])]):
            ax[0].plot(y, [U[nd][k] / scale for nd in wing_nodes], color=col, lw=2, marker='o', ms=3, label=lab)
        ax[0].set_title('Wing absolute displacement (A-frame)', fontsize=10, color=INK, loc='left')
        ax[0].set_xlabel('y_A (m)')
        ax[1].plot(y, tw / max(np.abs(tw).max(), 1e-30), color=COL[2], lw=2, marker='o', ms=3)
        ax[1].set_title('Wing elastic twist, nose-up positive (normalised)', fontsize=10, color=INK, loc='left')
        ax[1].set_xlabel('y_A (m)')
        for nodes, coord, lab, col, k in [(fus_nodes, 0, 'fuselage z', COL[0], 2), (fus_nodes, 0, 'fuselage y', COL[3], 1),
                                          (tail_nodes, 1, 'tailplane z (vs y_A)', COL[4], 2), (fin_nodes, 2, 'fin y (vs z_A)', COL[5], 1)]:
            if nodes:
                ax[2].plot(mdl.pos[nodes, coord], [U[nd][k] / scale for nd in nodes], color=col, lw=2, marker='o', ms=3, label=lab)
        ax[2].set_title('Fuselage, tailplane, fin', fontsize=10, color=INK, loc='left')
        ax[2].set_xlabel('position along component (m)')
        for a in ax:
            a.axhline(0, color=MUTED, lw=0.8); a.grid(color=GRID, lw=0.6)
            a.spines[['top', 'right']].set_visible(False)
            a.tick_params(colors=MUTED, labelsize=8)
        ax[0].legend(fontsize=8, frameon=False); ax[2].legend(fontsize=8, frameon=False)
        fig.suptitle(f'Mode {r["mode"]}  |  {r["f"]:.3f} Hz  |  {r["label"]}', fontsize=12, color=INK, x=0.01, ha='left')
        fig.tight_layout()
        fig.savefig(os.path.join(pdir, f'mode_{r["mode"]:02d}.png'), dpi=130)
        plt.close(fig)

    # Summary: strain-energy composition per elastic mode
    el = [r for r in rows if r['kind'] == 'elastic']
    keys = sorted({k for r in el for k in r['U']}, key=lambda k: -sum(r['U'].get(k, 0) for r in el))
    keep = keys[:6]
    fig, ax = plt.subplots(figsize=(11, 0.38 * len(el) + 1.6))
    left = np.zeros(len(el))
    yy = np.arange(len(el))[::-1]
    for c, k in enumerate(keep + ['other']):
        vals = np.array([r['U'].get(k, 0) if k != 'other' else 1 - sum(r['U'].get(kk, 0) for kk in keep) for r in el])
        g, f = (k if k != 'other' else ('', 'other'))
        lab = f'{g} {FAMILY_NAMES.get(g, {}).get(f, f)}'.strip()
        ax.barh(yy, vals, left=left, color=COL[c] if k != 'other' else '#c7c7c7', edgecolor='white', linewidth=1.5,
                height=0.72, label=lab)
        left += vals
    ax.set_yticks(yy)
    ax.set_yticklabels([f'{r["mode"]}  {r["f"]:.2f} Hz  ({r["sym"]})' for r in el], fontsize=8, color=INK)
    ax.set_xlim(0, 1); ax.set_xlabel('Share of beam strain energy', color=MUTED)
    ax.spines[['top', 'right']].set_visible(False); ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(0, -0.08))
    ax.set_title(f'{CASE_NAME} ({MODEL_TAG}): what deforms in each elastic mode', loc='left', color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'strain_energy_summary.png'), dpi=140)
    plt.close(fig)


# =====================================================================================
# Step 6 - aeroelastic modes (AsymptoticStability output)
# =====================================================================================
LIN_RIGID_NAMES = {9: ['u', 'v', 'w', 'p', 'q', 'r', 'phi', 'theta', 'psi'],
                   10: ['u', 'v', 'w', 'p', 'q', 'r', 'q0', 'q1', 'q2', 'q3']}


def read_linear_settings(log):
    """num_modes, use_euler, dt, newmark_damp, density from the SaveData settings (fallbacks if absent)."""
    out = dict(num_modes=30, use_euler=True, dt=None, newmark_damp=5e-3, density=None, ppal=True)
    try:
        f = find_one(os.path.join(OUTPUT_ROUTE, 'savedata'), '*.h5', 'SaveData file')[0]
        with h5py.File(f, 'r') as h:
            b = 'data/settings/LinearAssembler/linear_system_settings/'
            g = lambda k: h[b + k][()]
            out.update(num_modes=int(g('beam_settings/num_modes')), use_euler=bool(g('beam_settings/use_euler')),
                       dt=float(g('beam_settings/dt')), newmark_damp=float(g('beam_settings/newmark_damp')),
                       density=float(g('aero_settings/density')), ppal=bool(g('beam_settings/rigid_modes_ppal_axes')))
    except Exception as err:
        log.flag(f'Linear settings not read from SaveData ({err}); using defaults {out}')
    return out


def read_stability(sdir, n_struct_rows):
    """Eigenvalues (continuous time, as SHARPy writes them) plus ONLY the structural rows of the eigenvectors.

    The eigenvector files are (n_states x n_evals) text files, tens of MB. They are streamed once: the last
    n_struct_rows rows (structural states q, q_dot) are kept and the squared column norm of every row is summed,
    so the structural share of each eigenvector can be computed without holding the full matrix in memory.
    """
    ev = np.loadtxt(os.path.join(sdir, 'aeroelastic_eigenvalues.dat'))
    lam = ev[:, 0] + 1j * ev[:, 1]

    def stream(fn):
        from collections import deque
        tail, tot, n = deque(maxlen=n_struct_rows), None, 0
        with open(os.path.join(sdir, fn)) as f:
            for line in f:
                v = np.array(line.split(), dtype=float)
                tot = v ** 2 if tot is None else tot + v ** 2
                tail.append(v)
                n += 1
        return np.array(tail), tot, n

    re_, tr, n = stream('aeroelastic_eigenvectors_r.dat')
    im_, ti, _ = stream('aeroelastic_eigenvectors_i.dat')
    return lam, re_ + 1j * im_, tr + ti, n


def unwarp(s, dt):
    """Undo the Newmark/trapezoidal (Tustin-type) frequency compression of a discrete-time eigenvalue.

    SHARPy reports s_obs = ln(z)/dt. For a trapezoidal-type scheme z = (1 + s dt/2)/(1 - s dt/2), so the
    eigenvalue of the continuous system is s = (2/dt) tanh(s_obs dt/2). For a lightly damped mode this is
    f_true = tan(pi f_obs dt)/(pi dt). Exact for the structural (Newmark, average-acceleration) part only.
    """
    return (2.0 / dt) * np.tanh(s * dt / 2.0)


def analyse_eigenvalues(lam, S, tot, lin, freq_struct, struct_labels, log, tag=''):
    """Classify every eigenvalue: integrator (zero), aero/wake, flight dynamics or elastic."""
    N = lin['num_modes']
    n_rig = 9 if lin['use_euler'] else 10
    rnames = LIN_RIGID_NAMES[n_rig]
    q, qd = S[:N], S[N:2 * N]
    # linear modal coordinate j >= n_rig  <->  Modal mode (N_RIGID + j - n_rig)
    coord_name, coord_modal = [], []
    for j in range(N):
        if j < n_rig:
            coord_name.append(rnames[j]); coord_modal.append(None)
        else:
            m = N_RIGID + j - n_rig
            if m < len(freq_struct):
                coord_name.append(f'E{j - n_rig + 1} [{struct_labels.get(m, "")}, {freq_struct[m]:.2f} Hz]')
                coord_modal.append(m)
            else:
                coord_name.append(f'E{j - n_rig + 1} [not in Modal output]'); coord_modal.append(None)
    rows = []
    for k in range(lam.size):
        s = lam[k]
        if s.imag < -1e-9:
            continue
        sc = unwarp(s, lin['dt']) if UNWARP_FREQUENCIES else s
        fobs, fcor = abs(s) / (2 * np.pi), abs(sc) / (2 * np.pi)
        zeta = -s.real / abs(s) if abs(s) > 1e-9 else np.nan
        zeta_c = -sc.real / abs(sc) if abs(sc) > 1e-9 else np.nan
        sfrac = float((np.abs(S[:, k]) ** 2).sum() / tot[k])
        e = np.abs(qd[:, k]) ** 2
        e = e / e.sum() if e.sum() > 0 else e
        rig_share = float(e[:n_rig].sum())
        dom = int(np.argmax(e))
        if abs(s) < 1e-6:
            cat, name = 'zero', 'integrator / neutral (exact zero)'
        elif sfrac < WAKE_STRUCT_FRACTION:
            cat, name = 'aero', 'aerodynamic (wake) state'
        elif rig_share > 0.5:
            cat = 'flight'
            d = rnames[int(np.argmax(e[:n_rig]))]
            if abs(s.imag) < 1e-6:
                name = {'p': 'roll subsidence?', 'psi': 'spiral?', 'phi': 'spiral?', 'r': 'spiral?'}.get(d, f'real root ({d})?')
            elif d == 'p':
                name = 'roll subsidence (wake-coupled pair)?' if zeta > 0.7 else 'roll oscillation?'
            elif d in ('w', 'q', 'theta') and sfrac < FLIGHT_MIN_STRUCT_FRACTION:
                cat, name = 'aero', 'aero lag state coupled to heave/pitch'   # wake-ladder member, not a rigid-body mode
            else:
                name = {'u': 'phugoid?', 'v': 'Dutch roll?', 'r': 'Dutch roll?', 'w': 'short period / heave?',
                        'q': 'short period?', 'theta': 'short period?'}.get(d, f'oscillatory ({d})?')
        else:
            cat, name = 'elastic', coord_name[dom]
        top = np.argsort(-e)[:3]
        rows.append(dict(eig=k, s=s, s_corr=sc, f_obs=fobs, f=fcor, zeta_obs=zeta, zeta=zeta_c, sfrac=sfrac,
                         cat=cat, name=name, dom=dom, share=e, rig_share=rig_share,
                         content=', '.join(f'{coord_name[j].split(" [")[0]} {e[j]:.2f}' for j in top if e[j] > 0.02),
                         q=q[:, k]))
    counts = {c: sum(r['cat'] == c for r in rows) for c in ['zero', 'flight', 'elastic', 'aero']}
    log.info(f'{tag}eigenvalues (upper half-plane): {counts}')
    unstable = [r for r in rows if r['s'].real > 1e-6]
    log.check(f'{tag}no eigenvalue with positive real part (stable at this speed)', not unstable,
              ', '.join(f'eig {r["eig"]} Re={r["s"].real:.3g}' for r in unstable) or 'none')
    return rows, coord_name, coord_modal, n_rig


def track_elastic(rows, coord_modal, n_rig, freq_struct, lin):
    """For each linear elastic coordinate, the eigenvalue carrying most of it (ignoring wake states)."""
    out = []
    cand = [r for r in rows if r['cat'] in ('elastic', 'flight')]
    for j, m in enumerate(coord_modal):
        if j < n_rig:
            continue
        best = max(cand, key=lambda r: r['share'][j]) if cand else None
        if best is None or best['share'][j] < 0.3:
            out.append(dict(coord=j, modal=m, row=None)); continue
        f0 = freq_struct[m] if m is not None else np.nan
        w = 2 * np.pi * best['f']
        out.append(dict(coord=j, modal=m, row=best, f_struct=f0, share=best['share'][j],
                        zeta_num_est=0.5 * lin['newmark_damp'] * w * lin['dt'],
                        trusted=best['f_obs'] < 0.25 / lin['dt']))
    return out


def stability_step(mdl, rows_struct, log):
    sdir = STABILITY_DIR or os.path.join(OUTPUT_ROUTE, 'stability')
    if not os.path.isfile(os.path.join(sdir, 'aeroelastic_eigenvalues.dat')):
        log.info(f'No AsymptoticStability output in {sdir} - aeroelastic step skipped')
        return None
    lin = read_linear_settings(log)
    N = lin['num_modes']
    log.info(f'Linear system: {N} structural modes, use_euler={lin["use_euler"]}, dt={lin["dt"]} s '
             f'(Nyquist {0.5 / lin["dt"]:.0f} Hz, trusted below ~{0.25 / lin["dt"]:.0f} Hz), '
             f'newmark_damp={lin["newmark_damp"]}, density={lin["density"]}')
    freq_struct = mdl.w / (2 * np.pi)
    labels = {r['mode']: r['label'].split(' [')[0] for r in rows_struct if r['kind'] == 'elastic'}
    lam, S, tot, nstates = read_stability(sdir, 2 * N)
    log.info(f'Stability folder {sdir}: {lam.size} eigenvalues, {nstates} states; structural rows = last {2 * N}')
    rows, cname, cmodal, n_rig = analyse_eigenvalues(lam, S, tot, lin, freq_struct, labels, log)
    log.info(f'Linear modal coordinates: {n_rig} rigid ({", ".join(LIN_RIGID_NAMES[n_rig])}) then elastic; '
             f'LinearAssembler re-solves the modes, so linear elastic k = Modal mode {N_RIGID}+k-1')
    trk = track_elastic(rows, cmodal, n_rig, freq_struct, lin)

    # optional low-density reference run -> numerical damping
    ref_trk = None
    if REFERENCE_STABILITY_DIR and os.path.isfile(os.path.join(REFERENCE_STABILITY_DIR, 'aeroelastic_eigenvalues.dat')):
        rl, rS, rtot, _ = read_stability(REFERENCE_STABILITY_DIR, 2 * N)
        rrows, _, _, _ = analyse_eigenvalues(rl, rS, rtot, lin, freq_struct, labels, log, tag='[reference] ')
        ref_trk = {t['coord']: t for t in track_elastic(rrows, cmodal, n_rig, freq_struct, lin)}
        log.info(f'Reference (low-density) run read from {REFERENCE_STABILITY_DIR}: zeta_aero = zeta - zeta_ref')

    # frequency check of the mapping
    for t in trk:
        if t['row'] is not None and t['modal'] is not None and t['trusted']:
            d = (t['row']['f'] - t['f_struct']) / t['f_struct']
            if abs(d) > 0.15:
                log.flag(f'Elastic coordinate {t["coord"]} (Modal mode {t["modal"]}): coupled frequency differs by '
                         f'{d:+.0%} from the structural one - check the mode mapping or a strong aero effect')

    write_stability_tables(rows, trk, ref_trk, lin, RESULTS_DIR)
    plot_root_locus(rows, trk, freq_struct, lin, RESULTS_DIR)
    plot_damping(trk, ref_trk, RESULTS_DIR)
    argand_flight(rows, n_rig, RESULTS_DIR, log)
    for t in trk:
        r = t['row']
        if r is None:
            log.info(f'  coord {t["coord"]:2d}: no non-wake eigenvalue carries it (>30%)')
            continue
        zr = ref_trk.get(t['coord'], {}).get('row') if ref_trk else None
        za = f', zeta_aero {r["zeta"] - zr["zeta"]:.3f}' if zr else ''
        log.info(f'  {cname[t["coord"]]}: {r["f"]:.2f} Hz, zeta {r["zeta"]:.3f}{za} '
                 f'(share {t["share"]:.2f}{"" if t["trusted"] else ", above trusted range"})')
    for r in rows:
        if r['cat'] == 'flight':
            log.info(f'  flight dynamics: {r["name"]} {r["f"]:.3f} Hz, zeta {r["zeta"]:.3f} ({r["content"]})')
    return rows, trk


def write_stability_tables(rows, trk, ref_trk, lin, outdir):
    hdr = ['eig', 're_obs', 'im_obs', 'f_obs_Hz', 'zeta_obs', 'f_corr_Hz', 'zeta_corr', 'category', 'name',
           'structural_fraction', 'rigid_share', 'content']
    with open(os.path.join(outdir, 'aeroelastic_eigenvalues_classified.csv'), 'w') as f:
        f.write(','.join(hdr) + '\n')
        for r in rows:
            f.write(','.join([str(r['eig']), f'{r["s"].real:.6g}', f'{r["s"].imag:.6g}', f'{r["f_obs"]:.4f}',
                              f'{r["zeta_obs"]:.4f}', f'{r["f"]:.4f}', f'{r["zeta"]:.4f}', r['cat'],
                              '"' + r['name'] + '"', f'{r["sfrac"]:.2e}', f'{r["rig_share"]:.3f}',
                              '"' + r['content'] + '"']) + '\n')
    md = ['## Structural modes with aerodynamics', '',
          f'Frequencies corrected for discrete-time compression (dt = {lin["dt"]} s). '
          'zeta_num_est = Newmark numerical damping estimate (0.5 x newmark_damp x omega x dt).', '',
          '| Linear coord | Structural mode | f struct (Hz) | f aero (Hz) | df | zeta | zeta_num_est | zeta_ref | zeta_aero | share | note |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    csv = ['coord,modal_mode,f_struct_Hz,f_aero_Hz,df_pct,zeta,zeta_num_est,zeta_ref,zeta_aero,share,trusted,eig']
    for t in trk:
        r = t['row']
        if r is None:
            md.append(f'| {t["coord"]} | {t["modal"]} | - | - | - | - | - | - | - | - | not found |'); continue
        zr = ref_trk.get(t['coord'], {}).get('row') if ref_trk else None
        zref = f'{zr["zeta"]:.4f}' if zr else '-'
        zaer = f'{r["zeta"] - zr["zeta"]:.4f}' if zr else '-'
        df = (r['f'] - t['f_struct']) / t['f_struct'] * 100 if t['modal'] is not None else np.nan
        note = '' if t['trusted'] else 'above trusted range'
        md.append(f'| {t["coord"]} | {r["name"] if t["modal"] is None else r["name"]} | {t["f_struct"]:.2f} | {r["f"]:.2f} | '
                  f'{df:+.1f}% | {r["zeta"]:.3f} | {t["zeta_num_est"]:.4f} | {zref} | {zaer} | {t["share"]:.2f} | {note} |')
        csv.append(f'{t["coord"]},{t["modal"]},{t["f_struct"]:.4f},{r["f"]:.4f},{df:.2f},{r["zeta"]:.5f},'
                   f'{t["zeta_num_est"]:.5f},{zref},{zaer},{t["share"]:.3f},{t["trusted"]},{r["eig"]}')
    md += ['', '## Flight-dynamics modes (labels are candidates from the dominant rigid state)', '',
           '| Eig | Candidate | f (Hz) | zeta | Re (1/s) | Content |', '|---|---|---|---|---|---|']
    for r in sorted([r for r in rows if r['cat'] in ('flight', 'zero')], key=lambda r: r['f']):
        md.append(f'| {r["eig"]} | {r["name"]} | {r["f"]:.3f} | {r["zeta"]:.3f} | {r["s"].real:.3f} | {r["content"]} |')
    open(os.path.join(outdir, 'aeroelastic_mode_table.md'), 'w').write('\n'.join(md) + '\n')
    open(os.path.join(outdir, 'aeroelastic_mode_table.csv'), 'w').write('\n'.join(csv) + '\n')


def plot_root_locus(rows, trk, freq_struct, lin, outdir):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.2))
    fmax_trust = 0.25 / lin['dt']
    for ax, (fmax, remax, title) in zip(axs, [(fmax_trust * 1.6, None, f'0 - {fmax_trust * 1.6:.0f} Hz'),
                                             (10.5, None, '0 - 10 Hz (flight dynamics and low elastic modes)')]):
        aero = [r for r in rows if r['cat'] == 'aero' and r['f'] <= fmax]
        ax.scatter([r['s_corr'].real for r in aero], [r['s_corr'].imag / (2 * np.pi) for r in aero], s=10,
                   color='#c7c7c7', label='aerodynamic (wake) states', zorder=1)
        for t in trk:
            r = t['row']
            if r is None or t['modal'] is None or r['f'] > fmax:
                continue
            ax.plot([0, r['s_corr'].real], [t['f_struct'], r['s_corr'].imag / (2 * np.pi)], color='#9ec5f4', lw=1, zorder=2)
        el = [r for r in rows if r['cat'] == 'elastic' and r['f'] <= fmax]
        ax.scatter([r['s_corr'].real for r in el], [r['s_corr'].imag / (2 * np.pi) for r in el], s=34,
                   color=COL[0], edgecolor='white', linewidth=1, label='elastic (structural) modes', zorder=4)
        fl = [r for r in rows if r['cat'] == 'flight' and r['f'] <= fmax]
        ax.scatter([r['s_corr'].real for r in fl], [r['s_corr'].imag / (2 * np.pi) for r in fl], s=48, marker='D',
                   color=COL[1], edgecolor='white', linewidth=1, label='flight-dynamics modes', zorder=5)
        fs = [f for f in freq_struct if 0.01 < f <= fmax]
        ax.scatter(np.zeros(len(fs)), fs, marker='_', s=200, color=INK, linewidth=2,
                   label='structural modes (no aero)', zorder=3)
        for r in (fl if fmax < 20 else []):
            ax.annotate(r['name'].replace('?', ''), (r['s_corr'].real, r['s_corr'].imag / (2 * np.pi)),
                        textcoords='offset points', xytext=(6, 4), fontsize=7.5, color=INK)
        ax.axvline(0, color=MUTED, lw=0.8)
        if fmax > 20:
            ax.axhline(fmax_trust, color=COL[4], lw=1, ls='--')
            ax.text(ax.get_xlim()[0] if False else 0, fmax_trust, '  trusted limit', color=COL[4], fontsize=8, va='bottom', ha='right')
        ax.set_ylim(-0.3, fmax)
        ax.set_xlabel('Real part (1/s), corrected', color=MUTED)
        ax.set_ylabel('Frequency (Hz), corrected', color=MUTED)
        ax.set_title(title, loc='left', fontsize=10.5, color=INK)
        ax.grid(color=GRID, lw=0.6); ax.spines[['top', 'right']].set_visible(False); ax.tick_params(colors=MUTED)
    axs[0].legend(fontsize=8, frameon=False, loc='upper left')
    fig.suptitle(f'{CASE_NAME}: aeroelastic eigenvalues (light lines join each structural mode to its aeroelastic mode)',
                 fontsize=11.5, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'aeroelastic_root_locus.png'), dpi=150)
    plt.close(fig)


def plot_damping(trk, ref_trk, outdir):
    tt = [t for t in trk if t['row'] is not None and t['modal'] is not None]
    if not tt:
        return
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(tt) + 1.6))
    y = np.arange(len(tt))[::-1]
    z = np.array([t['row']['zeta'] for t in tt])
    zn = np.array([t['zeta_num_est'] for t in tt])
    ax.barh(y, z, color=[COL[0] if t['trusted'] else '#9ec5f4' for t in tt], height=0.62, label='total damping ratio')
    ax.scatter(zn, y, marker='|', s=160, color=COL[1], linewidth=2.5, zorder=4, label='Newmark numerical estimate')
    if ref_trk:
        zr = [ref_trk.get(t['coord'], {}).get('row') for t in tt]
        ax.scatter([r['zeta'] if r else np.nan for r in zr], y, marker='o', s=26, color=INK, zorder=5,
                   label='low-density reference run')
    for yi, t in zip(y, tt):
        ax.text(t['row']['zeta'] + 0.004, yi, f'{t["row"]["zeta"]:.3f}', va='center', fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{t["f_struct"]:.2f} Hz  ' + t['row']['name'].split(' [', 1)[-1].rsplit(', ', 1)[0].split(' + ')[0]
                        for t in tt],
                       fontsize=8, color=INK)
    ax.set_xlabel('Damping ratio (pale bars: above the trusted frequency range)', color=MUTED)
    ax.grid(axis='x', color=GRID, lw=0.6); ax.spines[['top', 'right']].set_visible(False); ax.tick_params(colors=MUTED)
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    ax.set_title(f'{CASE_NAME}: damping of each structural mode with aerodynamics', loc='left', fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'aeroelastic_damping.png'), dpi=150)
    plt.close(fig)


def argand_flight(rows, n_rig, outdir, log):
    """Argand diagrams of the rigid-state content of each flight-dynamics mode (complex, so phase is real now)."""
    fl = sorted([r for r in rows if r['cat'] == 'flight'], key=lambda r: r['f'])
    if not fl:
        return
    names = LIN_RIGID_NAMES[n_rig]
    colours = dict(zip(names, COL + ['#8c564b', '#4b5563', '#9ca3af']))
    ncol = min(4, len(fl)); nrow = int(np.ceil(len(fl) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.9 * nrow + 1.0))
    axs = np.atleast_1d(axs).ravel()
    t = np.linspace(0, 2 * np.pi, 200)
    for ax, r in zip(axs, fl):
        z = r['q'][:n_rig]
        k = int(np.argmax(np.abs(z)))
        zn = z / z[k]
        ax.set_aspect('equal')
        for dec in range(ARGAND_DECADES):
            rr = 1 - dec / ARGAND_DECADES
            ax.plot(rr * np.cos(t), rr * np.sin(t), color=GRID, lw=0.8)
            ax.text(rr * np.cos(np.radians(60)), rr * np.sin(np.radians(60)), f'{10.0 ** -dec:g}', fontsize=6.5, color=MUTED)
        ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
        ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45); ax.set_xticks([]); ax.set_yticks([])
        for j in sorted(range(n_rig), key=lambda j: -abs(zn[j])):
            if abs(zn[j]) < 10.0 ** -ARGAND_DECADES:
                continue
            rad = 1 + np.log10(abs(zn[j])) / ARGAND_DECADES
            c = rad * np.exp(1j * np.angle(zn[j]))
            ax.annotate('', xy=(c.real, c.imag), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='-|>', color=colours[names[j]], lw=2, mutation_scale=10))
            ax.text(c.real * 1.08, c.imag * 1.08, f'{names[j]} {abs(zn[j]):.2g}\n{np.degrees(np.angle(zn[j])):+.0f} deg',
                    fontsize=7, color=INK, ha='left' if c.real >= 0 else 'right', va='center')
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_title(f'{r["name"]}  eig {r["eig"]}\n{r["f"]:.3f} Hz, zeta {r["zeta"]:.3f}', fontsize=9, color=INK, loc='left')
    for ax in axs[len(fl):]:
        ax.axis('off')
    fig.suptitle(f'{CASE_NAME}: flight-dynamics mode content (rigid modal coordinates, CG / principal axes; '
                 f'u,v,w m/s, p,q,r rad/s, phi,theta,psi rad; log radius, largest state = 1 at 0 deg)',
                 fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(outdir, 'flight_dynamics_argand.png'), dpi=150)
    plt.close(fig)


def list_vtk(outdir, log):
    vtk = sorted(glob.glob(os.path.join(OUTPUT_ROUTE, '**', '*mode*.vt*'), recursive=True))
    with open(os.path.join(outdir, 'paraview_files.txt'), 'w') as f:
        f.write('\n'.join(vtk) + '\n')
    log.info(f'{len(vtk)} mode VTK files listed in paraview_files.txt' if vtk else
             'No mode VTK files found (check write_modes_vtk and the output folder)')


# =====================================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log = Log()
    case = load_case()
    modal = load_modal()
    pos, psi, src, cands = load_trim_geometry(case)
    log.info(f'Case {CASE_NAME} ({MODEL_TAG}); Modal folder {modal["dir"]}')
    (log.info if src.startswith('SaveData') else log.flag)(f'Geometry source: {src}')
    if len(cands) > 1:
        log.info(f'Other (pos, psi) groups found in SaveData: {cands}')
    mdl = Model(case, modal, pos, psi)
    log.info(f'{mdl.n} DOFs = 6 x {mdl.nf // 6} free nodes + {N_RIGID} rigid; {mdl.V.shape[1]} modes; '
             f'hub (node 0 + lumped) mass {mdl.m_hub:.4f} kg')

    log.info('--- Step 2: matrices and mass properties ---')
    mp = matrix_checks(mdl, log)
    resolve_symmetric(mdl, log)
    log.info('--- Step 3: rigid-body modes ---')
    rig, is_rigid = rigid_analysis(mdl, log)
    if MAKE_RIGID_ARGAND:
        argand_rigid(mdl, rig, is_rigid, mp, RESULTS_DIR, log)
    log.info('--- Step 4: elastic modes ---')
    rows = classify(mdl, rig, is_rigid, log)
    cov = [r['coverage'] for r in rows if r['kind'] == 'elastic']
    log.info(f'Beam strain energy / (1/2 phi^T K phi): min {min(cov):.3f}, median {np.median(cov):.3f}, max {max(cov):.3f}')
    log.info('--- Step 5: hand estimates ---')
    match_hand(rows, log)
    for r in rows:
        print(f'  mode {r["mode"]:2d}  {r["f"]:8.3f} Hz  {r["label"]}')

    if ANALYSE_STABILITY:
        log.info('--- Step 6: aeroelastic modes (AsymptoticStability) ---')
        stability_step(mdl, rows, log)

    write_tables(rows, RESULTS_DIR)
    if MAKE_MODE_PLOTS:
        plot_modes(mdl, rows, RESULTS_DIR)
    list_vtk(RESULTS_DIR, log)
    open(os.path.join(RESULTS_DIR, 'checks_log.txt'), 'w').write('\n'.join(log.lines) + '\n')
    print(f'\nResults written to {RESULTS_DIR}')
    return mdl, rows


if __name__ == '__main__':
    main()