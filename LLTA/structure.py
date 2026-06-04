#! /usr/bin/env python3
import h5py as h5
import numpy as np
import llta_geometry as llta

# --- Wing geometry ---
span_main = llta.span_main              # Full span (m). Semi-span = span_main / 2 - REMEMBER SHARPY USES SPAN MAIN AS SEMI_SPAN!
lambda_main = llta.lambda_main          # Fraction of semi-span that carries dihedral
lambda_dihedral = llta.lambda_dihedral  # Dihedral angle (degrees)

chord_main = llta.chord_main            # Wing chord (m)
ea_main = llta.ea_main                  # Elastic axis as fraction of chord from LE

# --- Tail geometry ---
span_tail = llta.span_tail              # Full tail span (m). One side = span_tail (not halved)
chord_tail = llta.chord_tail            # Tail chord (m)
ea_tail = llta.ea_tail                  # Tail elastic axis as fraction of chord from LE

# --- Fin geometry ---
fin_height = llta.fin_height            # Fin height (m)
chord_fin = llta.chord_fin              # Fin chord (m)
ea_fin = llta.ea_fin                    # Fin elastic axis as fraction of chord from LE

# --- Fuselage geometry ---
length_fuselage = llta.length_fuselage  # Total fuselage length (m)
offset_fuselage = 0.0                   # NOT USED: Vertical offset of fuselage tail end (m)

# --- Structural stiffness ---
ea  = llta.ea       # Axial stiffness (N)
ga  = llta.ga       # Shear stiffness (N)
gj  = llta.gj       # Torsional stiffness (Nm²)
eiy = llta.eiy      # Flapwise bending stiffness (Nm²)
eiz = llta.eiz      # Edgewise bending stiffness (Nm²)

# --- Distributed mass ---
m_bar_main     = llta.m_bar_main   # Wing mass per unit length (kg/m)
j_bar_main     = llta.j_bar_main   # Wing rotational inertia per unit length (kg·m)
cg_offset      = llta.cg_offset    # CG offset from elastic axis (fraction of chord, aft +ve)

m_bar_fuselage = llta.m_bar_fuselage  # Fuselage mass per unit length (kg/m)
j_bar_fuselage = llta.j_bar_fuselage

m_bar_tail     = llta.m_bar_tail   # Tail mass per unit length (kg/m)
j_bar_tail     = llta.j_bar_tail

# --- Stiffness scaling factors ---
sigma_fuselage = 1.0
sigma_tail     = 1.0

# --- Lumped mass values (kg) ---
mass_prop        = 0.05
mass_batteries   = 0.33
mass_pixhawk     = 0.08
mass_additional  = 0.54
mass_launch_handle   = 0.02
mass_camera          = 0.02
mass_camera_counter  = 0.02

# --- Lumped mass positions in the B (body) frame [x, y, z] (m) ---
front_fuselage_pos   = [0.0, -0.279, 0.0]
right_handle_pos     = [-0.023750, 0.0, 0.0]
left_handle_pos      = [-0.023750, 0.0, 0.0]
camera_pos           = [0.0, 0.0, 0.0]
camera_counter_pos   = [0.0, 0.0, 0.0]

# --- Lumped mass inertia tensors (kg·m²) ---
inertia_front_fuselage = np.diag([0.001179, 0.005401, 0.005436])
inertia_launch_handle  = np.diag([1.814e-05, 2.294e-06, 1.814e-05])
inertia_camera         = np.diag([3.143e-06, 3.780e-06, 4.800e-06])


class LLTAStructure:
    """
    This generates the structural (FEM) model for the LLTA aircraft and creates
    the .fem.h5 file required by SHARPy.

    Beam layout (beam_number):
        0: right wing  (inner + outer, shared beam ID)
        1: left wing   (inner + outer, shared beam ID)
        2: fuselage
        3: fin
        4: right tail
        5: left tail
    """

    def __init__(self, case_name, case_route, **kwargs):
        self.case_name  = case_name
        self.case_route = case_route

        # NEEDED??? --- Tuneable parameters (override via kwargs) ---
        self.sigma            = kwargs.get('sigma', 1)
        self.n_elem_multiplier = kwargs.get('n_elem_multiplier', 2)
        self.thrust           = kwargs.get('thrust', 0.)

        # --- Element / node counts (set in generate()) ---
        self.n_elem = None
        self.n_node = None
        self.n_node_elem = 3 # fixed for 3-node elements used for GEBM

        self.n_elem_main = None   # total elements per wing half
        self.n_elem_main1 = None   # inner (flat) portion
        self.n_elem_main2 = None   # outer (dihedral) portion
        self.n_elem_fuselage = None
        self.n_elem_fin = None
        self.n_elem_tail = None

        self.n_node_main = None
        self.n_node_main1 = None
        self.n_node_main2 = None
        self.n_node_fuselage = None
        self.n_node_fin = None
        self.n_node_tail = None

        # --- Control surfaces ---
        self.n_elem_per_elevator = None # Will be needed for control surfaces
        self.n_ailerons_per_wing = 0 #numb_ailerons
        self.n_elev_per_tail_surf = 0 #numb_elevators

        # --- Nodal coordinates (set in generate()) ---
        self.x = None
        self.y = None
        self.z = None

        # --- Key node indices (set in generate(), used by aero.py) ---
        self.end_of_fuselage_node = None

        # --- Element arrays (set in generate()) ---
        self.conn = None
        self.beam_number = None
        self.elem_stiffness = None
        self.elem_mass = None
        self.mass = None   # mass database, shape (n_mass, 6, 6)

        # --- Geometry constants used by aero.py ---
        self.span_main = span_main
        self.span_tail = span_tail
        self.fin_height = fin_height
        self.lambda_main = lambda_main
        self.lambda_dihedral = lambda_dihedral

    # -------------------------------------------------------------------------
    def set_thrust(self, value):
        self.thrust = value

    # -------------------------------------------------------------------------
    def generate(self):
        """
        Build all structural arrays and write the .fem.h5 file.
        Mirrors generate_fem() in generate_llta.py, reorganised as a class method.
        """

        # --- Element counts ---
        n_elem_multiplier = self.n_elem_multiplier
        self.n_elem_main = int(4 * n_elem_multiplier)
        self.n_elem_main1 = round(self.n_elem_main * (1 - lambda_main))
        self.n_elem_main2 = self.n_elem_main - self.n_elem_main1
        self.n_elem_fuselage = int(2 * n_elem_multiplier)
        self.n_elem_fin = int(2 * n_elem_multiplier)
        self.n_elem_tail = int(2 * n_elem_multiplier)

        # --- Node counts per component ---
        self.n_node_main1 = self.n_elem_main1 * (self.n_node_elem - 1) + 1
        self.n_node_main2 = self.n_elem_main2 * (self.n_node_elem - 1) + 1
        self.n_node_main = self.n_node_main1 + self.n_node_main2 - 1
        self.n_node_fuselage = self.n_elem_fuselage * (self.n_node_elem - 1) + 1
        self.n_node_fin = self.n_elem_fin  * (self.n_node_elem - 1) + 1
        self.n_node_tail = self.n_elem_tail * (self.n_node_elem - 1) + 1

        # --- Total element and node counts ---
        self.n_elem = (self.n_elem_main1 * 2
                       + self.n_elem_main2 * 2
                       + self.n_elem_fuselage
                       + self.n_elem_fin
                       + self.n_elem_tail * 2)

        self.n_node = (self.n_node_main1 * 2 - 1          # right + left inner (shared root)
                       + (self.n_node_main2 - 1) * 2      # right + left outer
                       + self.n_node_fuselage - 1          # fuselage (shares root node)
                       + self.n_node_fin - 1          # fin      (shares fuselage-end node)
                       + (self.n_node_tail - 1) * 2)    # right + left tail (share fuselage-end)

        # --- Stiffness and mass matrices ---
        n_stiffness = 3
        n_mass = 3

        base_stiffness_main = self.sigma * np.diag([ea, ga, ga, gj, eiy, eiz])
        base_stiffness_fuselage = base_stiffness_main.copy() * sigma_fuselage
        base_stiffness_fuselage[4, 4] = base_stiffness_fuselage[5, 5]
        base_stiffness_tail = base_stiffness_main.copy() * sigma_tail
        base_stiffness_tail[4, 4] = base_stiffness_tail[5, 5]

        base_mass_main = self.generate_mass_matrix(m_bar_main, j_bar_main)
        base_mass_fuselage = self.generate_mass_matrix(m_bar_fuselage, j_bar_fuselage)
        base_mass_tail = self.generate_mass_matrix(m_bar_tail, j_bar_tail)

        stiffness = np.zeros((n_stiffness, 6, 6))
        self.mass = np.zeros((n_mass, 6, 6))

        stiffness[0, ...] = base_stiffness_main
        stiffness[1, ...] = base_stiffness_fuselage
        stiffness[2, ...] = base_stiffness_tail

        self.mass[0, ...] = base_mass_main
        self.mass[1, ...] = base_mass_fuselage
        self.mass[2, ...] = base_mass_tail

        # --- Derived span values ---
        semi_span  = span_main / 2
        span_main1 = (1.0 - lambda_main) * semi_span
        span_main2 = lambda_main * semi_span
        lam_rad = np.deg2rad(lambda_dihedral)

        # --- Allocate structural arrays ---
        self.x = np.zeros((self.n_node,))
        self.y = np.zeros((self.n_node,))
        self.z = np.zeros((self.n_node,))

        self.beam_number = np.zeros((self.n_elem,), dtype=int)
        frame_of_reference_delta = np.zeros((self.n_elem, self.n_node_elem, 3))
        structural_twist = np.zeros((self.n_elem, self.n_node_elem))
        self.conn = np.zeros((self.n_elem, self.n_node_elem), dtype=int)
        self.elem_stiffness = np.zeros((self.n_elem,), dtype=int)
        self.elem_mass = np.zeros((self.n_elem,), dtype=int)
        boundary_conditions = np.zeros((self.n_node,), dtype=int)
        app_forces = np.zeros((self.n_node, 6))

        # --- Lumped masses ---
        n_lumped_mass = 5
        lumped_mass_nodes    = np.zeros((n_lumped_mass,), dtype=int)
        lumped_mass          = np.zeros((n_lumped_mass,))
        lumped_mass_position = np.zeros((n_lumped_mass, 3))
        lumped_mass_inertia  = np.zeros((n_lumped_mass, 3, 3))

        # front fuselage (node 0)
        lumped_mass_nodes[0]    = 0
        lumped_mass[0]          = mass_prop + mass_batteries + mass_pixhawk + mass_additional
        lumped_mass_position[0] = front_fuselage_pos
        lumped_mass_inertia[0]  = inertia_front_fuselage

        # right wing – launch handle (node 4) and camera (node 16)
        lumped_mass_nodes[1]    = 4
        lumped_mass[1]          = mass_launch_handle
        lumped_mass_position[1] = right_handle_pos
        lumped_mass_inertia[1]  = inertia_launch_handle

        lumped_mass_nodes[2]    = 16
        lumped_mass[2]          = mass_camera
        lumped_mass_position[2] = camera_pos
        lumped_mass_inertia[2]  = inertia_camera

        # left wing – launch handle (node 20) and camera counterweight (node 32)
        lumped_mass_nodes[3]    = 20
        lumped_mass[3]          = mass_launch_handle
        lumped_mass_position[3] = left_handle_pos
        lumped_mass_inertia[3]  = inertia_launch_handle

        lumped_mass_nodes[4]    = 32
        lumped_mass[4]          = mass_camera_counter
        lumped_mass_position[4] = camera_counter_pos
        lumped_mass_inertia[4]  = inertia_camera

        # =====================================================================
        # Build nodal coordinates and connectivity
        # we = working element index offset
        # wn = working node index offset
        # =====================================================================
        we = 0
        wn = 0

        # ------------------------------------------------------------------
        # INNER RIGHT WING  (beam 0)
        # ------------------------------------------------------------------
        self.beam_number[we:we + self.n_elem_main1] = 0
        self.y[wn:wn + self.n_node_main1] = np.linspace(0.0, span_main1, self.n_node_main1)

        for ielem in range(self.n_elem_main1):
            self.conn[we + ielem, :] = (np.ones((3,)) * (we + ielem) * (self.n_node_elem - 1)
                                        + [0, 2, 1])
            for inode in range(self.n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]

        self.elem_stiffness[we:we + self.n_elem_main1] = 0
        self.elem_mass[we:we + self.n_elem_main1]      = 0
        boundary_conditions[0] = 1  # clamped root
        app_forces[0]          = [0, self.thrust, 0, 0, 0, 0]

        we += self.n_elem_main1
        wn += self.n_node_main1

        # ------------------------------------------------------------------
        # OUTER RIGHT WING  (beam 0, dihedral section)
        # ------------------------------------------------------------------
        self.beam_number[we:we + self.n_elem_main2] = 0
        self.y[wn:wn + self.n_node_main2 - 1] = (
            self.y[wn - 1]
            + np.linspace(0.0, np.cos(lam_rad) * span_main2, self.n_node_main2)[1:])
        self.z[wn:wn + self.n_node_main2 - 1] = (
            self.z[wn - 1]
            + np.linspace(0.0, np.sin(lam_rad) * span_main2, self.n_node_main2)[1:])

        for ielem in range(self.n_elem_main2):
            self.conn[we + ielem, :] = (np.ones((3,)) * (we + ielem) * (self.n_node_elem - 1)
                                        + [0, 2, 1])
            for inode in range(self.n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]

        self.elem_stiffness[we:we + self.n_elem_main2] = 0
        self.elem_mass[we:we + self.n_elem_main2]      = 0
        boundary_conditions[wn + self.n_node_main2 - 2] = -1  # free tip

        we += self.n_elem_main2
        wn += self.n_node_main2 - 1

        # ------------------------------------------------------------------
        # INNER LEFT WING  (beam 1)
        # ------------------------------------------------------------------
        self.beam_number[we:we + self.n_elem_main1 - 1] = 1   # note: reproduces original code
        self.y[wn:wn + self.n_node_main1 - 1] = np.linspace(0.0, -span_main1, self.n_node_main1)[1:]

        for ielem in range(self.n_elem_main1):
            self.conn[we + ielem, :] = (np.ones((3,)) * (we + ielem) * (self.n_node_elem - 1)
                                        + [0, 2, 1])
            for inode in range(self.n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
        self.conn[we, 0] = 0  # left wing root shares node 0 with right wing root

        self.elem_stiffness[we:we + self.n_elem_main1] = 0
        self.elem_mass[we:we + self.n_elem_main1]      = 0

        we += self.n_elem_main1
        wn += self.n_node_main1 - 1

        # ------------------------------------------------------------------
        # OUTER LEFT WING  (beam 1, dihedral section)
        # ------------------------------------------------------------------
        self.beam_number[we:we + self.n_elem_main2] = 1
        self.y[wn:wn + self.n_node_main2 - 1] = (
            self.y[wn - 1]
            + np.linspace(0.0, -np.cos(lam_rad) * span_main2, self.n_node_main2)[1:])
        self.z[wn:wn + self.n_node_main2 - 1] = (
            self.z[wn - 1]
            + np.linspace(0.0, np.sin(lam_rad) * span_main2, self.n_node_main2)[1:])

        for ielem in range(self.n_elem_main2):
            self.conn[we + ielem, :] = (np.ones((3,)) * (we + ielem) * (self.n_node_elem - 1)
                                        + [0, 2, 1])
            for inode in range(self.n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]

        self.elem_stiffness[we:we + self.n_elem_main2] = 0
        self.elem_mass[we:we + self.n_elem_main2]      = 0
        boundary_conditions[wn + self.n_node_main2 - 2] = -1  # free tip

        we += self.n_elem_main2
        wn += self.n_node_main2 - 1

        # ------------------------------------------------------------------
        # FUSELAGE  (beam 2)
        # ------------------------------------------------------------------
        self.beam_number[we:we + self.n_elem_fuselage] = 2
        self.x[wn:wn + self.n_node_fuselage - 1] = np.linspace(0.0, length_fuselage,
                                                                 self.n_node_fuselage)[1:]
        self.z[wn:wn + self.n_node_fuselage - 1] = np.linspace(0.0, offset_fuselage,
                                                                 self.n_node_fuselage)[1:]

        for ielem in range(self.n_elem_fuselage):
            self.conn[we + ielem, :] = (np.ones((3,)) * (we + ielem) * (self.n_node_elem - 1)
                                        + [0, 2, 1])
            for inode in range(self.n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
        self.conn[we, 0] = 0  # fuselage root connects to wing root (node 0)

        self.elem_stiffness[we:we + self.n_elem_fuselage] = 1
        self.elem_mass[we:we + self.n_elem_fuselage]      = 1

        we += self.n_elem_fuselage
        wn += self.n_node_fuselage - 1
        self.end_of_fuselage_node = wn - 1   # last fuselage node, shared by fin and tails

        # ------------------------------------------------------------------
        # FIN  (beam 3)
        # ------------------------------------------------------------------
        self.beam_number[we:we + self.n_elem_fin] = 3
        self.x[wn:wn + self.n_node_fin - 1] = self.x[self.end_of_fuselage_node]
        self.z[wn:wn + self.n_node_fin - 1] = (
            self.z[self.end_of_fuselage_node]
            + np.linspace(0.0, fin_height, self.n_node_fin)[1:])

        for ielem in range(self.n_elem_fin):
            self.conn[we + ielem, :] = (np.ones((3,)) * (we + ielem) * (self.n_node_elem - 1)
                                        + [0, 2, 1])
            for inode in range(self.n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        self.conn[we, 0] = self.end_of_fuselage_node  # fin root at end of fuselage

        self.elem_stiffness[we:we + self.n_elem_fin] = 2
        self.elem_mass[we:we + self.n_elem_fin]      = 2

        we += self.n_elem_fin
        wn += self.n_node_fin - 1
        # end_of_fin_node = wn - 1  # available if needed

        # ------------------------------------------------------------------
        # RIGHT TAIL  (beam 4)
        # ------------------------------------------------------------------
        self.beam_number[we:we + self.n_elem_tail] = 4
        self.x[wn:wn + self.n_node_tail - 1] = self.x[self.end_of_fuselage_node]
        self.y[wn:wn + self.n_node_tail - 1] = np.linspace(0.0, span_tail,
                                                             self.n_node_tail)[1:]
        self.z[wn:wn + self.n_node_tail - 1] = self.z[self.end_of_fuselage_node]

        for ielem in range(self.n_elem_tail):
            self.conn[we + ielem, :] = (np.ones((3,)) * (we + ielem) * (self.n_node_elem - 1)
                                        + [0, 2, 1])
            for inode in range(self.n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
        self.conn[we, 0] = self.end_of_fuselage_node  # tail root at end of fuselage

        self.elem_stiffness[we:we + self.n_elem_tail] = 2
        self.elem_mass[we:we + self.n_elem_tail]      = 2
        boundary_conditions[wn + self.n_node_tail - 2] = -1  # free tip

        we += self.n_elem_tail
        wn += self.n_node_tail - 1

        # ------------------------------------------------------------------
        # LEFT TAIL  (beam 5)
        # ------------------------------------------------------------------
        self.beam_number[we:we + self.n_elem_tail] = 5
        self.x[wn:wn + self.n_node_tail - 1] = self.x[self.end_of_fuselage_node]
        self.y[wn:wn + self.n_node_tail - 1] = np.linspace(0.0, -span_tail,
                                                             self.n_node_tail)[1:]
        self.z[wn:wn + self.n_node_tail - 1] = self.z[self.end_of_fuselage_node]

        for ielem in range(self.n_elem_tail):
            self.conn[we + ielem, :] = (np.ones((3,)) * (we + ielem) * (self.n_node_elem - 1)
                                        + [0, 2, 1])
            for inode in range(self.n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
        self.conn[we, 0] = self.end_of_fuselage_node  # tail root at end of fuselage

        self.elem_stiffness[we:we + self.n_elem_tail] = 2
        self.elem_mass[we:we + self.n_elem_tail]      = 2
        boundary_conditions[wn + self.n_node_tail - 2] = -1  # free tip

        we += self.n_elem_tail
        wn += self.n_node_tail - 1

        # =====================================================================
        # Write .fem.h5
        # =====================================================================
        with h5.File(self.case_route + '/' + self.case_name + '.fem.h5', 'a') as h5file:
            h5file.create_dataset('coordinates',
                                  data=np.column_stack((self.x, self.y, self.z)))
            h5file.create_dataset('connectivities',       data=self.conn)
            h5file.create_dataset('num_node_elem',        data=self.n_node_elem)
            h5file.create_dataset('num_node',             data=self.n_node)
            h5file.create_dataset('num_elem',             data=self.n_elem)
            h5file.create_dataset('stiffness_db',         data=stiffness)
            h5file.create_dataset('elem_stiffness',       data=self.elem_stiffness)
            h5file.create_dataset('mass_db',              data=self.mass)
            h5file.create_dataset('elem_mass',            data=self.elem_mass)
            h5file.create_dataset('frame_of_reference_delta', data=frame_of_reference_delta)
            h5file.create_dataset('structural_twist',     data=structural_twist)
            h5file.create_dataset('boundary_conditions',  data=boundary_conditions)
            h5file.create_dataset('beam_number',          data=self.beam_number)
            h5file.create_dataset('app_forces',           data=app_forces)
            h5file.create_dataset('lumped_mass_nodes',    data=lumped_mass_nodes)
            h5file.create_dataset('lumped_mass',          data=lumped_mass)
            h5file.create_dataset('lumped_mass_inertia',  data=lumped_mass_inertia)
            h5file.create_dataset('lumped_mass_position', data=lumped_mass_position)

    # -------------------------------------------------------------------------
    def generate_mass_matrix(self, m_bar, j_bar):
        """Diagonal 6×6 mass matrix for a uniform beam cross-section."""
        return np.diag([m_bar, m_bar, m_bar,
                        j_bar, 0.5 * j_bar, 0.5 * j_bar])

    # -------------------------------------------------------------------------
    def find_index_of_closest_entry(self, array_values, target_value):
        return np.argmin(np.abs(array_values - target_value))

    # -------------------------------------------------------------------------
    def calculate_aircraft_mass(self):
        """
        Print a breakdown of structural and lumped mass, and the centre of
        gravity. Call this after generate() as a sanity check before running
        SHARPy.
        """
        list_elem_mass = []
        center_of_gravity = np.zeros((3,))

        for i_elem in range(self.n_elem):
            start_node = self.conn[i_elem, 0]
            end_node   = self.conn[i_elem, 1]
            length_elem = np.sqrt(
                (self.x[start_node] - self.x[end_node]) ** 2
                + (self.y[start_node] - self.y[end_node]) ** 2
                + (self.z[start_node] - self.z[end_node]) ** 2)
            mid_node = self.conn[i_elem, 2]
            pos = [self.x[mid_node], self.y[mid_node], self.z[mid_node]]
            m_dist = self.mass[self.elem_mass[i_elem], 0, 0]
            mass_elem = m_dist * length_elem
            list_elem_mass.append(mass_elem)
            for i_dim in range(3):
                center_of_gravity[i_dim] += mass_elem * pos[i_dim]

        total_structural_mass = sum(list_elem_mass)

        print(f'\n--- LLTA Mass Breakdown ---')
        print(f'{"Beam":<6} {"Component":<14} {"Mass (kg)":>10}')
        print(f'{"-"*6} {"-"*14} {"-"*10}')
        for i_beam in sorted(set(self.beam_number)):
            beam_mass = sum(
                m for m, b in zip(list_elem_mass, self.beam_number) if b == i_beam)
            labels = {0: 'right wing', 1: 'left wing', 2: 'fuselage',
                    3: 'fin', 4: 'right tail', 5: 'left tail'}
            print(f'{i_beam:<6} {labels.get(i_beam, "?"):<14} {beam_mass:>10.4f}')

        # Lumped masses (from module-level constants, mirroring what generate() uses)
        lumped_masses = [
            mass_prop + mass_batteries + mass_pixhawk + mass_additional,
            mass_launch_handle, mass_camera,
            mass_launch_handle, mass_camera_counter,
        ]
        total_lumped = sum(lumped_masses)
        total_mass   = total_structural_mass + total_lumped

        center_of_gravity /= total_mass
        print(f'Total lumped mass: {total_lumped:.4f} kg')
        print(f'Total aircraft mass: {total_mass:.4f} kg')
        print(f'Centre of gravity [x, y, z]: {center_of_gravity}')

if __name__ == '__main__':
    import os
    fem_file = 'test_output/llta_test.fem.h5'
    if os.path.isfile(fem_file):
        os.remove(fem_file)
    
    s = LLTAStructure('llta_test', 'test_output')
    s.generate()
    s.calculate_aircraft_mass()
    print("n_elem:", s.n_elem)
    print("n_node:", s.n_node)
    print("end_of_fuselage_node:", s.end_of_fuselage_node)