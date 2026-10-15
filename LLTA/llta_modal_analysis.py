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
CASE_NAME = 'llta_trim_u_inf_9__wingincidence'
MODEL_TAG = 'M2 - section inertias'
CASE_ROUTE = os.path.dirname(os.path.realpath(__file__))        # folder holding <case>.fem.h5
OUTPUT_ROUTE = os.path.join(CASE_ROUTE, 'output', CASE_NAME)       # SHARPy output folder for this case
MODAL_DIR = None        # None = auto-find the folder containing Mglobal.dat under OUTPUT_ROUTE
SAVEDATA_FILE = None    # None = auto-find a .h5 file under OUTPUT_ROUTE/savedata
RESULTS_DIR = os.path.join(OUTPUT_ROUTE, 'modal_analysis_results')
MAKE_MODE_PLOTS = True
MODE_PLOT_DIAGNOSTIC = True  # keep the full 3-panel A-frame mode plots (mode_shapes/) while the clean plots are refined
CLEAN_MODE_PLOTS = 'all'     # 'all' or a list of mode numbers for the clean publication-style plots (mode_shapes_clean/)
PAPER_FIG_WIDTH_IN = 7.0     # clean mode plots: figure width (in); 7.0 = full text width, 3.5 = single journal column
PAPER_FIG_ASPECT = 0.32      # clean mode plots: height / width - wide and short to suit a high-aspect-ratio wing
PAPER_FONT_PT = 9            # clean mode plots: label font size (pt); ticks/tags 1 pt smaller
ARGAND_DECADES = 3           # flight-dynamics Argand (Step 6): arrow length log10 scaled over this many decades
ANALYSE_STABILITY = True     # read AsymptoticStability output (Step 6) if a stability folder exists
STABILITY_DIR = None         # None = OUTPUT_ROUTE/stability
REFERENCE_STABILITY_DIR = None  # stability folder of a LOW-DENSITY run of the same case (numerical damping),
                             # e.g. os.path.join(CASE_ROUTE, 'output', 'llta_modes_aerodynamics_low_rho', 'stability')
WAKE_STRUCT_FRACTION = 1e-11 # eigenvectors with less structural content than this are labelled wake (aero) states
FLIGHT_MIN_STRUCT_FRACTION = 1e-9  # heave/pitch-dominated eigenvalues below this are wake-ladder states
UNWARP_FREQUENCIES = True    # undo the discrete-time frequency compression: s = (2/dt) tanh(s_obs dt/2)
SHOW_WAKE_STATES = False     # clean eigenvalue figures only; the diagnostic figure always shows the wake states
ELASTIC_ZOOM_FMAX = 10.5     # Hz, upper limit of the clean low-frequency elastic figure
FLIGHT_ZOOM = dict(fmax=2.75, re_min=-17.0)  # clean flight-dynamics figure: damped-frequency and real-part limits

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
# Eigenvalue figures: elastic modes coloured by family (same order as the mode-shape slide), flight dynamics separate
FAMILY_COLOURS = {'wing bending': COL[0], 'wing torsion': COL[1], 'boom / tail': COL[2], 'not in Modal output': MUTED}
FLIGHT_COLOUR = INK   # black diamonds; marker shape separates them from the elastic modes
WAKE_COLOUR, LINK_COLOUR = '#d1d5db', '#c3cad5'


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
# Method used here: rigid fraction c = a^T M_rr a / phi^T M phi, with a = M_rr^-1 (M phi)[last 10 rows]
# (Mglobal.dat + eigenvectors.dat). c = 1 for a rigid-body mode, c = 0 for an elastic mode.
#
# ALTERNATIVE (simpler): use omega from the Modal eigenvalues (frequencies.dat, or Im(lambda) in
# eigenvaluetable.txt). Modal is undamped, so Re(lambda) = 0 for EVERY mode and cannot be used;
# Im(lambda) = omega = 0 for rigid-body modes and omega > 0 for elastic modes.
# For the LLTA the gap is clear (0 vs 15.26 rad/s for the first elastic mode), e.g.
#     is_rigid = mdl.w < 1e-3 * mdl.w.max()
# Caveats: numerical rigid frequencies are only approximately zero, so a threshold is still needed;
# and an omega test does not check that elastic modes carry no rigid-body momentum (a mass-matrix
# check that c provides). Kept: rigid fraction c below; omega test noted as an alternative for now.
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
    """Share of the mode's kinetic energy in each component (lumped: m|u|^2 + theta^T J theta per node).

    INFORMATION ONLY: this does not feed the mode label or the rigid/elastic classification. It is reported
    in mode_table (Top kinetic energy, KE_* columns) to show WHERE THE MASS MOVES, as a complement to the
    strain energy (where the structure deforms). Uses: GVT sensor placement, mass-change sensitivity,
    interpreting hand-estimate mismatches (e.g. hub participation).
    """
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
    """Sign changes of the dominant quantity along the right semi-span (hub to tip), wing modes only.

    INFORMATION ONLY: this does not feed the mode label or the classification. It is reported in mode_table
    (wing_nodal_points) as a cross-check on the '#n' order in the label.
    Quantity: twist about the beam axis (torsion), vertical displacement z (flapwise bending, bend_y),
    fore-aft displacement x (in-plane bending, bend_z).
    """
    nodes = wing_line(mdl)
    if family == 'torsion':
        vals = np.array([mdl.theta_B(phi, *mdl.master[nd])[0] for nd in nodes])
    else:
        comp = 0 if family == 'bend_z' else 2      # in-plane bending -> x (fore-aft); flapwise -> z (vertical)
        vals = np.array([mdl.u_abs(phi, nd)[comp] for nd in nodes])
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


def paper_rc():
    """Matplotlib style for publication figures (LaTeX look): serif Computer Modern, 8-9 pt, black lines,
    inward ticks on all four sides, no grid, no title (the caption goes in the paper).
    Uses real LaTeX if it is installed (exact match to the paper's font); otherwise mathtext Computer Modern."""
    import shutil
    rc = {'font.family': 'serif', 'font.size': PAPER_FONT_PT, 'axes.labelsize': PAPER_FONT_PT,
          'xtick.labelsize': PAPER_FONT_PT - 1, 'ytick.labelsize': PAPER_FONT_PT - 1,
          'legend.fontsize': PAPER_FONT_PT - 1, 'axes.linewidth': 0.6, 'axes.edgecolor': 'k',
          'axes.labelcolor': 'k', 'text.color': 'k', 'xtick.color': 'k', 'ytick.color': 'k',
          'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
          'xtick.major.size': 3.0, 'ytick.major.size': 3.0, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
          'xtick.minor.visible': True, 'ytick.minor.visible': True, 'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
          'axes.grid': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
          'pdf.fonttype': 42, 'ps.fonttype': 42}         # embed TrueType, not Type 3 (journal requirement)
    if shutil.which('latex') and shutil.which('dvipng'):
        rc.update({'text.usetex': True, 'font.serif': ['Computer Modern Roman']})
    else:
        # cmr10 ships with matplotlib, so the Computer Modern look works without a LaTeX installation
        rc.update({'text.usetex': False, 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True,
                   'font.serif': ['cmr10', 'CMU Serif', 'Computer Modern Roman', 'DejaVu Serif']})
    return rc


def wing_out_of_plane(mdl, phi, nodes):
    """Absolute displacement of each wing node normal to its local wing panel (local z_B, + = up).

    Uses the node's master-element section axes at trim, so on the dihedral outer panels the motion is
    measured perpendicular to the panel instead of being split across A-frame y and z.
    """
    out = []
    for nd in nodes:
        zB = crv2rotation(mdl.psi[mdl.master[nd]])[:, 2]
        zB = zB if zB[2] >= 0 else -zB                     # make + point up on both wings
        out.append(mdl.u_abs(phi, nd) @ zB)
    return np.array(out)


def wing_kinks(mdl, nodes, min_deg=5.0):
    """Indices (into nodes) of dihedral kinks: nodes where the UNDEFORMED wing line turns by more than min_deg."""
    xyz = np.asarray(mdl.case['coordinates'])[nodes]
    idx = []
    for k in range(1, len(nodes) - 1):
        a, b = xyz[k] - xyz[k - 1], xyz[k + 1] - xyz[k]
        ang = np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))
        if ang > min_deg:
            idx.append(k)
    return idx                                  # indices into `nodes`


GROUP_WORD = {'fuselage': 'boom', 'tailplane': 'tailplane', 'fin': 'fin', 'wing': 'wing', 'hub': 'hub'}
SECONDARY_COLOUR = '#1f5fa8'    # second quantity on a dual-axis plot (right axis), dashed


def family_word(group, family):
    return FAMILY_NAMES.get(group, {}).get(family, family)


def mode_title(r):
    """Plain-English mode name for figure titles, e.g. '1st symmetric wing flapwise bending'; coupled modes get
    '+ <second type>' from the second-largest strain-energy share."""
    ordinal = {1: '1st', 2: '2nd', 3: '3rd'}.get(r['order'], f'{r["order"]}th')
    sym = {'S': 'symmetric', 'A': 'antisymmetric'}.get(r['sym'], 'mixed-symmetry')
    name = f'{ordinal} {sym} {GROUP_WORD.get(r["group"], r["group"])} {family_word(r["group"], r["family"])}'
    sec = secondary_of(r)
    if sec:
        name += f' + {GROUP_WORD.get(sec[0], sec[0])} {family_word(*sec)}'
    return name


def secondary_of(r):
    """(group, family) of the second-largest strain-energy share for modes flagged [coupled], else None."""
    if not r.get('coupled'):
        return None
    ranked = sorted(r['U'].items(), key=lambda kv: -kv[1])
    return ranked[1][0] if len(ranked) > 1 else None


# ---- quantities of interest ------------------------------------------------------------------
def _section_axes(mdl, nd):
    return crv2rotation(mdl.psi[mdl.master[nd]])            # columns: x_B (beam axis), y_B (chord), z_B (normal)


def wing_quantity(mdl, phi, nodes, family):
    """Wing quantity along the span (total motion = elastic + body motion), in each node's section axes at trim.
    bend_y  -> out-of-plane displacement w (normal to the local panel, + up)
    bend_z  -> in-plane (fore-aft) displacement u (in the panel plane, + aft)
    torsion -> twist theta about the beam axis (+ nose up on both wings)"""
    out = []
    for nd in nodes:
        C = _section_axes(mdl, nd)
        if family == 'bend_y':
            n = C[:, 2] if C[2, 2] >= 0 else -C[:, 2]
            out.append(mdl.u_abs(phi, nd) @ n)
        elif family == 'bend_z':
            c = C[:, 1] if C[0, 1] >= 0 else -C[:, 1]
            out.append(mdl.u_abs(phi, nd) @ c)
        else:
            out.append((C.T @ mdl.theta_A(phi, nd))[0] * np.sign(C[1, 0] or 1.0))
    return np.array(out)


def boom_quantity(mdl, phi, nodes, family):
    """Boom quantity along its length (A-frame; the boom lies along x_A, aft).
    bend_y -> vertical displacement (z_A, + up); bend_z -> lateral displacement (y_A, + starboard);
    torsion -> twist about the boom axis (rotation about x_A)."""
    if family == 'torsion':
        return np.array([mdl.theta_A(phi, nd)[0] for nd in nodes])
    k = 2 if family == 'bend_y' else 1
    return np.array([mdl.u_abs(phi, nd)[k] for nd in nodes])


Y_LABELS = {('wing', 'bend_y'): 'Out-of-plane displacement,\n' + r'$w$ (normalised)',
            ('wing', 'bend_z'): 'In-plane displacement,\n' + r'$u$ (normalised)',
            ('wing', 'torsion'): 'Twist,\n' + r'$\theta$ (normalised)',
            ('fuselage', 'bend_y'): 'Vertical displacement,\n' + r'$w$ (normalised)',
            ('fuselage', 'bend_z'): 'Lateral displacement,\n' + r'$v$ (normalised)',
            ('fuselage', 'torsion'): 'Twist,\n' + r'$\theta$ (normalised)'}


def _norm(v):
    m = np.abs(v).max()
    return v / m if m > 0 else v


def _curve_px(ax, x, y, n=400):
    """Densely resampled curve in display (pixel) coordinates, for label-collision checks."""
    xs = np.linspace(0, 1, n)
    t = np.linspace(0, 1, len(x))
    return ax.transData.transform(np.c_[np.interp(xs, t, x), np.interp(xs, t, y)])


def _tags(ax, x, y, tags, obstacles):
    """Open circle on each reference node plus a text tag, placed automatically: candidate positions around the
    node are tried (inboard-below first, as on mode 10) and the one that overlaps no curve, no other tag and stays
    inside the axes wins. obstacles = list of (N, 2) pixel arrays of every curve drawn on the figure."""
    fig = ax.figure
    for k, _ in tags:
        ax.plot([x[k]], [y[k]], ls='none', marker='o', ms=5.5, mfc='white', mec='k', mew=0.9, zorder=4)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    axbox = ax.get_window_extent(rend)
    pts = np.vstack(obstacles)
    placed = []
    xc = 0.5 * (x.min() + x.max())
    for k, text in tags:
        s_in = -np.sign(x[k] - xc) or 1.0                       # +1: inboard is to the right
        h_in, h_out = ('left', 'right') if s_in > 0 else ('right', 'left')
        cands = [(5 * s_in, -5, h_in, 'top'), (5 * s_in, 5, h_in, 'bottom'),
                 (-5 * s_in, 5, h_out, 'bottom'), (-5 * s_in, -5, h_out, 'top'),
                 (0, 7, 'center', 'bottom'), (0, -7, 'center', 'top')]
        best, best_score = None, None
        for dx, dy, ha, va in cands:
            ann = ax.annotate(text, (x[k], y[k]), xytext=(dx, dy), textcoords='offset points', ha=ha, va=va,
                              fontsize=PAPER_FONT_PT - 1, linespacing=1.0, multialignment='center')
            ann.set_in_layout(False)
            bb = ann.get_window_extent(rend).expanded(1.06, 1.12)
            inside = (bb.x0 >= axbox.x0 and bb.x1 <= axbox.x1 and bb.y0 >= axbox.y0 and bb.y1 <= axbox.y1)
            hits = np.sum((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) & (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1))
            clash = sum(bb.overlaps(o) for o in placed)
            score = hits + 1000 * clash + (0 if inside else 500)
            if best_score is None or score < best_score:
                if best is not None:
                    best[0].remove()
                best, best_score = (ann, bb), score
            else:
                ann.remove()
            if score == 0:
                break
        placed.append(best[1])


def _headroom(ax, *curves):
    v = np.concatenate(curves)
    rng = (v.max() - v.min()) or 1.0
    ax.set_ylim(v.min() - 0.25 * rng, v.max() + 0.25 * rng)


class _Layout:
    """Node lists, normalised positions and reference nodes for the wing and boom panels."""
    def __init__(self, mdl):
        self.wing = sorted([nd for nd in range(len(mdl.pos)) if mdl.node_comp.get(nd) in ('R wing', 'L wing', 'hub')],
                           key=lambda nd: mdl.pos[nd, 1])
        y = mdl.pos[self.wing, 1]
        self.eta = y / np.abs(y).max()
        self.wing_tags = [(int(np.argmin(np.abs(y))), 'centre (hub)')] + \
                         [(k, 'dihedral\nkink') for k in wing_kinks(mdl, self.wing)]
        self.boom = sorted([nd for nd in range(len(mdl.pos)) if mdl.node_comp.get(nd) in ('fuselage', 'hub')],
                           key=lambda nd: mdl.pos[nd, 0])
        x = mdl.pos[self.boom, 0]
        self.xi = (x - x.min()) / ((x.max() - x.min()) or 1.0)
        self.boom_tags = [(0, 'hub'), (len(self.boom) - 1, 'tail')]

    def panel(self, mdl, phi, group, family):
        if group == 'wing':
            return self.eta, _norm(wing_quantity(mdl, phi, self.wing, family)), self.wing_tags, \
                r'Spanwise position, $y/(b/2)$', (-1.05, 1.05)
        return self.xi, _norm(boom_quantity(mdl, phi, self.boom, family)), self.boom_tags, \
            r'Position along boom, $x/L_{\mathrm{boom}}$', (-0.03, 1.03)


def _style_axis(ax, x, ylab, xlab, xlim):
    ax.axhline(0, color='0.6', lw=0.5, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab, linespacing=1.1)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))


def plot_modes_clean(mdl, rows, outdir):
    """Clean, publication-style mode-shape plots: only the quantities of interest for each mode.

    Single type (e.g. wing flapwise bending): one curve on one axis.
    Coupled, same component (e.g. wing torsion + wing in-plane bending): dual y-axis - primary on the left
      (black, solid), secondary on the right (blue, dashed, blue axis).
    Coupled, different components (e.g. boom vertical bending + wing torsion): two panels side by side
      (wing span | boom length), each with its own quantity.
    Sign: SHARPy's eigenvector sign unchanged (same half-cycle as the ParaView mode files).
    Displacement: total perturbation (elastic + body motion) about trim; static dihedral/trim deflection excluded.
    Each quantity is normalised to max |value| = 1 (eigenvector scale is arbitrary).
    Format: paper_rc(); wide and short; title = mode number, name and frequency; PDF (vector) + PNG.
    """
    pdir = os.path.join(outdir, 'mode_shapes_clean')
    os.makedirs(pdir, exist_ok=True)
    L = _Layout(mdl)
    sel = (lambda m: True) if CLEAN_MODE_PLOTS == 'all' else (lambda m: m in CLEAN_MODE_PLOTS)
    for r in rows:
        if r['kind'] != 'elastic' or not sel(r['mode']):
            continue
        prim = ('fuselage' if r['group'] in ('fuselage', 'hub') else r['group'], r['family'])
        sec = secondary_of(r)
        sec = ('fuselage' if sec and sec[0] == 'hub' else sec[0], sec[1]) if sec else None
        if prim not in Y_LABELS or (sec and sec not in Y_LABELS):
            print(f'[clean plot] mode {r["mode"]} ({r["label"]}): component not supported yet - skipped')
            continue
        phi = mdl.V[:, r['mode']]
        title = f'Mode {r["mode"]}: {mode_title(r)}, {r["f"]:.2f} Hz'
        with plt.rc_context(paper_rc()):
            W, H = PAPER_FIG_WIDTH_IN, PAPER_FIG_ASPECT * PAPER_FIG_WIDTH_IN
            if sec is None or sec[0] == prim[0]:
                fig, ax = plt.subplots(figsize=(W, H))
                x, q, tags, xlab, xlim = L.panel(mdl, phi, *prim)
                ax.plot(x, q, color='k', lw=1.0, marker='o', ms=2.2, zorder=3)
                curves = [q]
                if sec:                                          # dual y-axis, same component
                    ax2 = ax.twinx()
                    _, q2, _, _, _ = L.panel(mdl, phi, *sec)
                    ax2.plot(x, q2, color=SECONDARY_COLOUR, lw=1.0, ls='--', marker='o', ms=2.2, mfc='white',
                             zorder=2)
                    ax2.set_ylabel(Y_LABELS[sec], color=SECONDARY_COLOUR, linespacing=1.1)
                    ax2.tick_params(axis='y', which='both', colors=SECONDARY_COLOUR, right=True)
                    ax2.spines['right'].set_color(SECONDARY_COLOUR)
                    ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
                    ax.tick_params(axis='y', which='both', right=False)
                    _headroom(ax2, q, q2)
                    curves.append(q2)
                _style_axis(ax, x, Y_LABELS[prim], xlab, xlim)
                _headroom(ax, *curves)
                ax.set_title(title, fontsize=PAPER_FONT_PT + 1, pad=4)
                fig.tight_layout(pad=0.3)
                obs = [_curve_px(ax, x, q)] + ([_curve_px(ax2, x, q2)] if sec else [])
                _tags(ax, x, q, tags, obs)
            else:                                                # coupled across components: wing | boom
                fig, axs = plt.subplots(1, 2, figsize=(W, H), gridspec_kw=dict(width_ratios=[2, 1]))
                for ax, comp in zip(axs, ('wing', 'fuselage')):
                    fam = prim[1] if prim[0] == comp else sec[1]
                    x, q, tags, xlab, xlim = L.panel(mdl, phi, comp, fam)
                    ax.plot(x, q, color='k', lw=1.0, marker='o', ms=2.2, zorder=3)
                    _style_axis(ax, x, Y_LABELS[(comp, fam)], xlab, xlim)
                    _headroom(ax, q)
                    ax._llta_curve = (x, q, tags)
                fig.suptitle(title, fontsize=PAPER_FONT_PT + 1, y=0.99)
                fig.tight_layout(pad=0.3)
                for ax in axs:
                    x, q, tags = ax._llta_curve
                    _tags(ax, x, q, tags, [_curve_px(ax, x, q)])
            for ext in ('pdf', 'png'):
                fig.savefig(os.path.join(pdir, f'mode_{r["mode"]:02d}.{ext}'), dpi=300)
            plt.close(fig)


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
        if r['kind'] != 'elastic' or not MODE_PLOT_DIAGNOSTIC:
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
    out['u_inf'] = None
    try:
        with h5py.File(f, 'r') as h:
            out['u_inf'] = float(h['data/settings/AerogridLoader/wake_shape_generator_input/u_inf'][()])
    except Exception:
        log.info('Free-stream speed not found in SaveData; figure titles will omit it')
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
    tag_eigenvalues(rows, trk, cmodal, rows_struct)
    has_ref = ref_trk is not None
    plot_root_locus_diagnostic(rows, trk, freq_struct, lin, has_ref, RESULTS_DIR)
    plot_eigenvalues_clean(rows, trk, freq_struct, lin, has_ref, RESULTS_DIR)
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
          'f_n = natural frequency |lambda|/2pi (used for df); f_d = damped frequency Im(lambda)/2pi '
          '(the height plotted in the eigenvalue figures).', '',
          '| Linear coord | Structural mode | f struct (Hz) | f_n aero (Hz) | f_d aero (Hz) | df (f_n) | zeta | zeta_num_est | zeta_ref | zeta_aero | share | note |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    csv = ['coord,modal_mode,f_struct_Hz,f_n_aero_Hz,f_d_aero_Hz,df_pct,zeta,zeta_num_est,zeta_ref,zeta_aero,share,trusted,eig']
    for t in trk:
        r = t['row']
        if r is None:
            md.append(f'| {t["coord"]} | {t["modal"]} | - | - | - | - | - | - | - | - | - | not found |'); continue
        fd = r['s_corr'].imag / (2 * np.pi)
        zr = ref_trk.get(t['coord'], {}).get('row') if ref_trk else None
        zref = f'{zr["zeta"]:.4f}' if zr else '-'
        zaer = f'{r["zeta"] - zr["zeta"]:.4f}' if zr else '-'
        df = (r['f'] - t['f_struct']) / t['f_struct'] * 100 if t['modal'] is not None else np.nan
        note = '' if t['trusted'] else 'above trusted range'
        md.append(f'| {t["coord"]} | {r["name"] if t["modal"] is None else r["name"]} | {t["f_struct"]:.2f} | {r["f"]:.2f} | '
                  f'{fd:.2f} | {df:+.1f}% | {r["zeta"]:.3f} | {t["zeta_num_est"]:.4f} | {zref} | {zaer} | {t["share"]:.2f} | {note} |')
        csv.append(f'{t["coord"]},{t["modal"]},{t["f_struct"]:.4f},{r["f"]:.4f},{fd:.4f},{df:.2f},{r["zeta"]:.5f},'
                   f'{t["zeta_num_est"]:.5f},{zref},{zaer},{t["share"]:.3f},{t["trusted"]},{r["eig"]}')
    md += ['', '## Flight-dynamics modes (labels are candidates from the dominant rigid state)', '',
           '| Eig | Candidate | f_n (Hz) | f_d (Hz) | zeta | Re (1/s) | Content |', '|---|---|---|---|---|---|---|']
    for r in sorted([r for r in rows if r['cat'] in ('flight', 'zero')], key=lambda r: r['f']):
        md.append(f'| {r["eig"]} | {r["name"]} | {r["f"]:.3f} | {r["s_corr"].imag / (2 * np.pi):.3f} | {r["zeta"]:.3f} | '
                  f'{r["s"].real:.3f} | {r["content"]} |')
    open(os.path.join(outdir, 'aeroelastic_mode_table.md'), 'w').write('\n'.join(md) + '\n')
    open(os.path.join(outdir, 'aeroelastic_mode_table.csv'), 'w').write('\n'.join(csv) + '\n')


def plot_family(r):
    """Colour family of a classified Modal (structural) row: wing bending / wing torsion / boom / tail."""
    if r.get('group') == 'wing':
        return 'wing torsion' if r['family'] == 'torsion' else 'wing bending'
    return 'boom / tail'


def short_mode_label(r):
    """Compact label for a structural mode, e.g. '13: S wing torsion 1' (Modal mode number first)."""
    grp = {'fuselage': 'boom', 'tailplane': 'tail'}.get(r['group'], r['group'])
    fam = FAMILY_NAMES.get(r['group'], {}).get(r['family'], r['family']).replace(' bending', ' bend.')
    return f'{r["mode"]}: {r["sym"]} {grp} {fam} {r["order"]}'


def tag_eigenvalues(rows, trk, coord_modal, rows_struct):
    """Add plotting fields: corrected Re and damped frequency for every row; family, label and pairing for elastic rows.

    Paired = the eigenvalue is the one tracked from a Modal mode (track_elastic). Elastic-classified eigenvalues
    that no Modal mode tracks to are 'unpaired' and drawn hollow.
    """
    by_mode = {r['mode']: r for r in rows_struct if r['kind'] == 'elastic'}
    paired = {id(t['row']) for t in trk if t['row'] is not None and t['modal'] is not None}
    for r in rows:
        r['re'] = r['s_corr'].real
        r['fd'] = r['s_corr'].imag / (2 * np.pi)          # damped frequency (Hz) = plotted height
        if r['cat'] != 'elastic':
            continue
        rs = by_mode.get(coord_modal[r['dom']])
        r['plot_family'] = plot_family(rs) if rs else 'not in Modal output'
        r['short'] = short_mode_label(rs) if rs else f'E{r["dom"] - 8} (not in Modal)'
        r['paired'] = id(r) in paired


def trusted_fmax(lin):
    """Upper trusted frequency on the plotted (corrected) axis. The trust test is f_obs < 0.25/dt on the raw axis;
    with the compression correction that limit maps to tan(pi f_obs dt)/(pi dt)."""
    f_obs = 0.25 / lin['dt']
    return np.tan(np.pi * f_obs * lin['dt']) / (np.pi * lin['dt']) if UNWARP_FREQUENCIES else f_obs


def conditions_line(lin, has_ref):
    parts = [CASE_NAME, MODEL_TAG]
    if lin.get('u_inf'):
        parts.append(f'U = {lin["u_inf"]:g} m/s')
    parts.append(f'dt = {lin["dt"]:g} s')
    parts.append('frequencies corrected for discrete-time compression' if UNWARP_FREQUENCIES else 'raw (uncorrected) frequencies')
    if not has_ref:
        parts.append('provisional: no low-density reference run')
    return '  ·  '.join(parts)


def _spread_labels(ax, pts, obstacles=(), min_gap_pt=10.0, pad_pt=8.0, fontsize=7.5):
    """Direct labels: placed left of each point (right if there is no room), nudged upwards to clear other labels
    and plotted markers. pts = [(x, y, text)], obstacles = [(x, y)] in data coordinates."""
    if not pts:
        return
    fig = ax.figure
    fig.canvas.draw()
    to_pt = 72.0 / fig.dpi
    left_pt = ax.get_window_extent().x0 * to_pt
    obs = [tuple(ax.transData.transform(o) * to_pt) for o in obstacles]
    items = []
    for x, y, t in pts:
        xd, yd = ax.transData.transform((x, y)) * to_pt
        w = len(t) * fontsize * 0.52                     # approximate text width (pt)
        side = -1 if xd - pad_pt - w > left_pt else 1
        items.append((yd, xd, x, y, t, w, side))
    items.sort(key=lambda i: i[0])
    placed = []
    for yd, xd, x, y, t, w, side in items:
        x0, x1 = (xd - pad_pt - w, xd - pad_pt) if side < 0 else (xd + pad_pt, xd + pad_pt + w)
        yl = yd
        for _ in range(60):
            clash = any(abs(yl - py) < min_gap_pt and not (x1 < px0 or x0 > px1) for py, px0, px1 in placed) or \
                    any(abs(yl - oy) < 7 and x0 - 5 < ox < x1 + 5 for ox, oy in obs)
            if not clash:
                break
            yl += 2.5
        placed.append((yl, x0, x1))
        ann = ax.annotate(t, (x, y), xytext=(side * pad_pt, yl - yd), textcoords='offset points',
                          ha='right' if side < 0 else 'left', va='center', fontsize=fontsize, color=INK, zorder=6,
                          arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5, shrinkA=0, shrinkB=3)
                          if abs(yl - yd) > 2 else None)
        ann.set_in_layout(False)


def _draw_eigs(ax, rows, trk, freq_struct, fmax, re_min=None, show_wake=True, label_elastic=True,
               label_flight=False, diagnostic=False):
    """Draw one eigenvalue panel. Height = damped frequency Im(lambda)/2pi; points filtered on that same value."""
    def inside(r):
        return -0.05 <= r['fd'] <= fmax and (re_min is None or r['re'] >= re_min)

    if show_wake:
        aero = [r for r in rows if r['cat'] == 'aero' and inside(r)]
        ax.scatter([r['re'] for r in aero], [r['fd'] for r in aero], s=9, color=WAKE_COLOUR, linewidth=0,
                   label='aerodynamic (wake) states', zorder=1)
    fs = [f for f in freq_struct if 0.01 < f <= fmax]
    ax.scatter(np.zeros(len(fs)), fs, marker='_', s=170, color=INK, linewidth=1.6,
               label='structural modes, no aero (Modal)', zorder=3)
    for t in trk:
        r = t['row']
        if r is None or t['modal'] is None or not inside(r):
            continue
        ax.plot([0, r['re']], [t['f_struct'], r['fd']], color=LINK_COLOUR, lw=0.9, zorder=2)

    el = [r for r in rows if r['cat'] == 'elastic' and inside(r)]
    for fam, col in FAMILY_COLOURS.items():
        p = [r for r in el if r['plot_family'] == fam and r['paired']]
        if p:
            ax.scatter([r['re'] for r in p], [r['fd'] for r in p], s=46, color=col, edgecolor='white', linewidth=1.2,
                       label=f'elastic: {fam}', zorder=4)
    unp = [r for r in el if not r['paired']]
    if unp:
        ax.scatter([r['re'] for r in unp], [r['fd'] for r in unp], s=46, facecolor='white',
                   edgecolor=[FAMILY_COLOURS[r['plot_family']] for r in unp], linewidth=1.5,
                   label='elastic, not tracked to a Modal mode', zorder=4)
    fl = [r for r in rows if r['cat'] == 'flight' and inside(r)]
    if fl:
        ax.scatter([r['re'] for r in fl], [r['fd'] for r in fl], s=58, marker='D', color=FLIGHT_COLOUR,
                   edgecolor='white', linewidth=1.2, label='flight dynamics (candidates)', zorder=5)

    ax.axvline(0, color=MUTED, lw=0.8, zorder=0)
    ax.set_ylim(-0.03 * fmax, fmax)
    if re_min is not None:
        ax.set_xlim(re_min, max(0.5, -0.04 * re_min))
    elif label_elastic and not diagnostic:
        lo = min([r['re'] for r in el + fl] + [-1.0])
        ax.set_xlim(1.5 * lo, max(0.5, -0.04 * lo))     # room for labels left of the leftmost points
    ax.set_xlabel(r'Real part, Re($\lambda$) (1/s)', color=MUTED)
    ax.set_ylabel(r'Damped frequency, Im($\lambda$)/2$\pi$ (Hz)', color=MUTED)
    ax.grid(color=GRID, lw=0.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=MUTED)

    labels = []
    if label_elastic:
        for r in el:
            if r['paired']:
                labels.append((r['re'], r['fd'], r['short']))
            elif diagnostic:
                labels.append((r['re'], r['fd'], f'eig {r["eig"]} (untracked, {r["plot_family"]})'))
    if label_flight:
        for r in fl:
            name = r['name'] if diagnostic else \
                r['name'].replace('?', '').replace(' (wake-coupled pair)', ', wake-coupled').strip() + ' (cand.)'
            labels.append((r['re'], r['fd'], name))
    obstacles = [(r['re'], r['fd']) for r in el + fl]
    return labels, obstacles


def plot_root_locus_diagnostic(rows, trk, freq_struct, lin, has_ref, outdir):
    """Everything on: wake states, tracking lines, untracked elastic eigenvalues (with eig index), raw candidate names,
    trusted limit. Two panels: full range and 0-10 Hz."""
    f_tr = trusted_fmax(lin)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.6))
    for ax, fmax, title in [(axs[0], f_tr * 1.25, f'0 - {f_tr * 1.25:.0f} Hz'),
                            (axs[1], ELASTIC_ZOOM_FMAX, f'0 - {ELASTIC_ZOOM_FMAX:g} Hz')]:
        labels, obs = _draw_eigs(ax, rows, trk, freq_struct, fmax, show_wake=True, label_elastic=fmax < 20,
                                 label_flight=fmax < 20, diagnostic=True)
        if fmax > f_tr:
            ax.axhline(f_tr, color=COL[4], lw=1, ls='--', zorder=0)
            ax.text(ax.get_xlim()[0], f_tr, f' trusted limit ({f_tr:.1f} Hz corrected = {0.25 / lin["dt"]:.0f} Hz raw)',
                    color=COL[4], fontsize=8, va='bottom', ha='left')
        ax.set_title(title, loc='left', fontsize=10.5, color=INK)
        _spread_labels(ax, labels, obs)
    axs[0].legend(fontsize=8, frameon=False, loc='lower left')
    fig.suptitle('Aeroelastic eigenvalues, diagnostic view (light lines join each Modal mode to the eigenvalue tracked from it)\n'
                 + conditions_line(lin, has_ref), fontsize=10.5, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'aeroelastic_root_locus_diagnostic.png'), dpi=150)
    plt.close(fig)


def plot_eigenvalues_clean(rows, trk, freq_struct, lin, has_ref, outdir):
    """Three separate presentation figures: trusted range, low-frequency elastic zoom, flight-dynamics zoom.
    Legend sits outside the axes on the right; conditions line under the title."""
    f_tr = trusted_fmax(lin)
    parts = conditions_line(lin, has_ref).split('  ·  ')
    sub = '  ·  '.join(parts[:4]) + '\n' + '  ·  '.join(parts[4:])

    def finish(fig, ax, title, fname):
        ax.legend(fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0))
        ax.set_title(title, loc='left', fontsize=11.5, color=INK, pad=40)
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=7.5, color=MUTED, ha='left', va='bottom',
                linespacing=1.4)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=200)
        plt.close(fig)

    # 1. Trusted range
    fig, ax = plt.subplots(figsize=(11, 7.4))
    labels, obs = _draw_eigs(ax, rows, trk, freq_struct, f_tr, show_wake=SHOW_WAKE_STATES)
    _spread_labels(ax, labels, obs)
    finish(fig, ax, f'Aeroelastic eigenvalues, 0 - {f_tr:.0f} Hz (trusted range)', 'aeroelastic_eigenvalues_full.png')

    # 2. Low-frequency elastic zoom
    fig, ax = plt.subplots(figsize=(11, 6.4))
    labels, obs = _draw_eigs(ax, rows, trk, freq_struct, ELASTIC_ZOOM_FMAX, show_wake=SHOW_WAKE_STATES)
    _spread_labels(ax, labels, obs)
    finish(fig, ax, f'Low-frequency elastic modes, 0 - {ELASTIC_ZOOM_FMAX:g} Hz',
           'aeroelastic_eigenvalues_elastic_zoom.png')

    # 3. Flight dynamics, with an inset for the slow modes near the origin
    fig, ax = plt.subplots(figsize=(11, 6.4))
    labels, obs = _draw_eigs(ax, rows, trk, freq_struct, FLIGHT_ZOOM['fmax'], re_min=FLIGHT_ZOOM['re_min'],
                             show_wake=SHOW_WAKE_STATES, label_flight=True)
    slow = [l for l in labels if l[0] > -1.0 and l[1] < 0.6]
    ins = ax.inset_axes([0.62, 0.36, 0.34, 0.40])
    sl = [r for r in rows if r['cat'] == 'flight' and r['re'] > -1.0 and -0.05 <= r['fd'] < 0.6]
    ins.scatter([r['re'] for r in sl], [r['fd'] for r in sl], s=50, marker='D', color=FLIGHT_COLOUR,
                edgecolor='white', linewidth=1.2, zorder=5)
    ins.axvline(0, color=MUTED, lw=0.8)
    ins.set_xlim(-0.65, 0.12)
    ins.set_ylim(-0.04, 0.55)
    ins.grid(color=GRID, lw=0.6)
    ins.tick_params(colors=MUTED, labelsize=7)
    ins.set_title('slow modes near the origin', fontsize=7.5, color=MUTED, loc='left')
    for x, y, t in slow:
        left_half = x < 0.5 * ins.get_xlim()[0]          # label to the right if the point is in the left half
        ins.annotate(t, (x, y), xytext=(6 if left_half else -6, 5), textcoords='offset points',
                     ha='left' if left_half else 'right', fontsize=7.5, color=INK)
    ax.indicate_inset_zoom(ins, edgecolor=MUTED, alpha=0.6)
    _spread_labels(ax, [l for l in labels if l not in slow], obs)
    finish(fig, ax, f'Flight-dynamics modes, 0 - {FLIGHT_ZOOM["fmax"]:g} Hz', 'aeroelastic_eigenvalues_flight.png')


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
        plot_modes_clean(mdl, rows, RESULTS_DIR)
    list_vtk(RESULTS_DIR, log)
    open(os.path.join(RESULTS_DIR, 'checks_log.txt'), 'w').write('\n'.join(log.lines) + '\n')
    print(f'\nResults written to {RESULTS_DIR}')
    return mdl, rows


if __name__ == '__main__':
    main()