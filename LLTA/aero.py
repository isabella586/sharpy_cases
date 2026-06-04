#! /usr/bin/env python3
import h5py as h5
import numpy as np
from structure import (chord_main, chord_tail, chord_fin,
                       ea_main, ea_tail, ea_fin,
                       span_main)

# Reference area (used by get_settings.py via llta.py's reference_area property)
area_ref = chord_main * span_main


class LLTAAero:
    """
    Generates the aerodynamic model for the LLTA aircraft and writes
    the .aero.h5 file required by SHARPy.

    Surface layout:
        0: right wing
        1: left wing
        2: fin
        3: right tail
        4: left tail

    Control surfaces:
        0: elevator  (right and left tail share one index — one command moves both)
        1: rudder    (fin)
    """

    def __init__(self, m, structure, case_name, case_route, **kwargs):
        """
        Args:
            m (int): Number of chordwise aerodynamic panels.
            structure (LLTAStructure): Populated structure instance (must have
                been initialised; generate() need not have been called yet).
            case_name (str): SHARPy case name.
            case_route (str): Path to the folder where .aero.h5 will be written.

        Keyword Args:
            cs_deflection (float): Initial elevator deflection (rad). Default 0.
            rudder_static_deflection (float): Initial rudder deflection (rad). Default 0.
            polars (list[np.ndarray]): Optional list of (N,4) arrays of
                [AoA (rad), Cl, Cd, Cm] per airfoil for polar corrections.
        """
        self.m          = m
        self.structure  = structure
        self.route      = case_route
        self.case_name  = case_name

        self.cs_deflection           = kwargs.get('cs_deflection', 0.)
        self.rudder_static_deflection = kwargs.get('rudder_static_deflection', 0.)
        self.polars                  = kwargs.get('polars', None)

        self.n_surfaces         = 5
        self.n_control_surfaces = 2

    # -------------------------------------------------------------------------
    def generate(self):
        """
        Build all aerodynamic arrays and write the .aero.h5 file.
        Mirrors generate_aero_file() in generate_llta.py.
        """
        structure    = self.structure
        m            = self.m

        # Pull counts from the structure instance
        n_elem       = structure.n_elem
        n_node       = structure.n_node
        n_node_elem  = structure.n_node_elem
        n_elem_main  = structure.n_elem_main      # total elements per wing half
        n_node_main  = structure.n_node_main      # total nodes per wing half
        n_elem_fuselage = structure.n_elem_fuselage
        n_node_fuselage = structure.n_node_fuselage
        n_elem_fin   = structure.n_elem_fin
        n_node_fin   = structure.n_node_fin
        n_elem_tail  = structure.n_elem_tail
        n_node_tail  = structure.n_node_tail

        # --- Allocate aero arrays ---
        airfoil_distribution = np.zeros((n_elem, n_node_elem), dtype=int)
        surface_distribution = np.zeros((n_elem,), dtype=int) - 1  # -1 = no aero surface
        surface_m            = np.zeros((self.n_surfaces,), dtype=int)
        m_distribution       = 'uniform'
        aero_node            = np.zeros((n_node,), dtype=bool)
        twist                = np.zeros((n_elem, n_node_elem))
        sweep                = np.zeros((n_elem, n_node_elem))
        chord                = np.zeros((n_elem, n_node_elem))
        elastic_axis         = np.zeros((n_elem, n_node_elem))

        # --- Control surface arrays ---
        control_surface          = np.zeros((n_elem, n_node_elem), dtype=int) - 1
        control_surface_type     = np.zeros((self.n_control_surfaces,), dtype=int)
        control_surface_deflection  = np.zeros((self.n_control_surfaces,))
        control_surface_chord    = np.zeros((self.n_control_surfaces,), dtype=int)
        control_surface_hinge_coord = np.zeros((self.n_control_surfaces,), dtype=float)

        # CS 0 — elevator (applied to both right and left tail surfaces)
        control_surface_type[0]        = 0                        # static
        control_surface_deflection[0]  = self.cs_deflection       # rad
        control_surface_chord[0]       = m                        # full chord
        control_surface_hinge_coord[0] = -0.25                    # nondim wrt elastic axis

        # CS 1 — rudder (fin)
        control_surface_type[1]        = 0                        # static
        control_surface_deflection[1]  = self.rudder_static_deflection
        control_surface_chord[1]       = 1
        control_surface_hinge_coord[1] = 0.0

        # =====================================================================
        # Populate arrays — we/wn track element and node offsets, matching
        # the order beams are built in structure.generate()
        # =====================================================================
        we = 0
        wn = 0

        # ------------------------------------------------------------------
        # RIGHT WING  (surface 0, beam 0)
        # ------------------------------------------------------------------
        i_surf = 0
        airfoil_distribution[we:we + n_elem_main, :] = 0
        surface_distribution[we:we + n_elem_main]    = i_surf
        surface_m[i_surf] = m
        aero_node[wn:wn + n_node_main] = True

        temp_chord = np.full(n_node_main, chord_main)
        temp_sweep = np.zeros(n_node_main)

        node_counter = 0
        for i_elem in range(we, we + n_elem_main):
            for i_local_node in range(n_node_elem):
                if i_local_node != 0:
                    node_counter += 1
                chord[i_elem, i_local_node]        = temp_chord[node_counter]
                elastic_axis[i_elem, i_local_node] = ea_main
                sweep[i_elem, i_local_node]        = temp_sweep[node_counter]

        we += n_elem_main
        wn += n_node_main

        # ------------------------------------------------------------------
        # LEFT WING  (surface 1, beam 1)
        # ------------------------------------------------------------------
        i_surf = 1
        airfoil_distribution[we:we + n_elem_main, :] = 0
        surface_distribution[we:we + n_elem_main]    = i_surf
        surface_m[i_surf] = m
        aero_node[wn:wn + n_node_main - 1] = True

        temp_chord = np.full(n_node_main, chord_main)
        node_counter = 0
        for i_elem in range(we, we + n_elem_main):
            for i_local_node in range(n_node_elem):
                if i_local_node != 0:
                    node_counter += 1
                chord[i_elem, i_local_node]        = temp_chord[node_counter]
                elastic_axis[i_elem, i_local_node] = ea_main
                sweep[i_elem, i_local_node]        = -temp_sweep[node_counter]

        we += n_elem_main
        wn += n_node_main - 1

        # ------------------------------------------------------------------
        # FUSELAGE — structural only, no aero surface, skip both counters
        # ------------------------------------------------------------------
        we += n_elem_fuselage
        wn += n_node_fuselage - 1 - 1  # -1 for shared root, -1 to land on end_of_fuselage_node

        # ------------------------------------------------------------------
        # FIN  (surface 2, beam 3)
        # ------------------------------------------------------------------
        i_surf = 2
        airfoil_distribution[we:we + n_elem_fin, :] = 1
        surface_distribution[we:we + n_elem_fin]    = i_surf
        surface_m[i_surf] = m
        aero_node[wn:wn + n_node_fin] = True

        for i_elem in range(we, we + n_elem_fin):
            for i_local_node in range(n_node_elem):
                chord[i_elem, i_local_node]        = chord_fin
                elastic_axis[i_elem, i_local_node] = ea_fin
                control_surface[i_elem, i_local_node] = 1  # rudder

        we += n_elem_fin
        wn += n_node_fin - 1

        # ------------------------------------------------------------------
        # RIGHT TAIL  (surface 3, beam 4)
        # ------------------------------------------------------------------
        i_surf = 3
        airfoil_distribution[we:we + n_elem_tail, :] = 2
        surface_distribution[we:we + n_elem_tail]    = i_surf
        surface_m[i_surf] = m
        aero_node[wn:] = True  # all remaining nodes are aero nodes

        for i_elem in range(we, we + n_elem_tail):
            for i_local_node in range(n_node_elem):
                twist[i_elem, i_local_node]        = 0.
                chord[i_elem, i_local_node]        = chord_tail
                elastic_axis[i_elem, i_local_node] = ea_tail
                control_surface[i_elem, i_local_node] = 0  # elevator

        we += n_elem_tail
        wn += n_node_tail

        # ------------------------------------------------------------------
        # LEFT TAIL  (surface 4, beam 5)
        # ------------------------------------------------------------------
        i_surf = 4
        airfoil_distribution[we:we + n_elem_tail, :] = 2
        surface_distribution[we:we + n_elem_tail]    = i_surf
        surface_m[i_surf] = m
        aero_node[wn:wn + n_node_tail - 1] = True

        for i_elem in range(we, we + n_elem_tail):
            for i_local_node in range(n_node_elem):
                twist[i_elem, i_local_node]        = 0.
                chord[i_elem, i_local_node]        = chord_tail
                elastic_axis[i_elem, i_local_node] = ea_tail
                control_surface[i_elem, i_local_node] = 0  # elevator

        we += n_elem_tail
        wn += n_node_tail

        # =====================================================================
        # Write .aero.h5
        # =====================================================================
        with h5.File(self.route + '/' + self.case_name + '.aero.h5', 'w') as h5file:

            # Airfoils — all flat plates (NACA 0000) for now
            airfoils_group = h5file.create_group('airfoils')
            airfoils_group.create_dataset('0', data=np.column_stack(
                self.generate_naca_camber(M=0, P=0)))   # wing
            airfoils_group.create_dataset('1', data=np.column_stack(
                self.generate_naca_camber(M=0, P=0)))   # fin
            airfoils_group.create_dataset('2', data=np.column_stack(
                self.generate_naca_camber(M=0, P=0)))   # tail

            chord_ds = h5file.create_dataset('chord', data=chord)
            chord_ds.attrs['units'] = 'm'

            twist_ds = h5file.create_dataset('twist', data=twist)
            twist_ds.attrs['units'] = 'rad'

            sweep_ds = h5file.create_dataset('sweep', data=sweep)
            sweep_ds.attrs['units'] = 'rad'

            h5file.create_dataset('airfoil_distribution', data=airfoil_distribution)
            h5file.create_dataset('surface_distribution', data=surface_distribution)
            h5file.create_dataset('surface_m',            data=surface_m)
            h5file.create_dataset('m_distribution',
                                  data=m_distribution.encode('ascii', 'ignore'))
            h5file.create_dataset('aero_node',            data=aero_node)
            h5file.create_dataset('elastic_axis',         data=elastic_axis)
            h5file.create_dataset('control_surface',      data=control_surface)
            h5file.create_dataset('control_surface_deflection',
                                  data=control_surface_deflection)
            h5file.create_dataset('control_surface_chord',data=control_surface_chord)
            h5file.create_dataset('control_surface_hinge_coord',
                                  data=control_surface_hinge_coord)
            h5file.create_dataset('control_surface_type', data=control_surface_type)

            if self.polars is not None:
                polars_group = h5file.create_group('polars')
                for i_airfoil, polar in enumerate(self.polars):
                    polars_group.create_dataset('{:g}'.format(i_airfoil), data=polar)

    # -------------------------------------------------------------------------
    def generate_naca_camber(self, M=0, P=0):
        """
        Generate a NACA 4-digit camber line.
        M=0, P=0 gives a flat plate (symmetric airfoil), used for all
        LLTA surfaces until measured camber data is available.
        """
        mm = M * 1e-2
        p  = P * 1e-1

        def naca(x, mm, p):
            if x < 1e-6:
                return 0.0
            elif x < p:
                return mm / (p * p) * (2 * p * x - x * x)
            elif x > p and x < 1 + 1e-6:
                return mm / ((1 - p) * (1 - p)) * (1 - 2 * p + 2 * p * x - x * x)

        x_vec = np.linspace(0, 1, 1000)
        y_vec = np.array([naca(x, mm, p) for x in x_vec])
        return x_vec, y_vec
