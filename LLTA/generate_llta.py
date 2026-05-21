#! /usr/bin/env python3
import h5py as h5
import numpy as np
import os
import sharpy.utils.algebra as algebra
import llta_geometry as llta
import math
import sharpy.sharpy_main 

case_name = 'llta'
route = os.path.dirname(os.path.realpath(__file__)) + '/'

# EXECUTION
flow = ['BeamLoader',
        'AerogridLoader',
        #'NonLinearStatic',
        #'StaticUvlm',
        'StaticTrim',           # Longitudinal static trim
        #'StaticCoupled',       # Static coupled solver with GECM and UVLM. This needs calling when running DynamicCoupled (ref dynamiccoupled.py solver script)
        'BeamLoads',            # Computation of beam loads and strains for static
        'AerogridPlot',         # Output of aero grid
        'BeamPlot',             # Beam structure and output data for static
        #'DynamicCoupled'       # Uses BeamLoads, BeamPlot and AerogridPlot at each timestep
        'Modal',
        #'LinearAssember',
        #'AsymptoticStability',
        #'SaveData'
        ]

# if free_flight is False, the motion of the centre of the wing is prescribed.
free_flight = True #This is clamp on/off
if not free_flight:
    case_name += '_prescribed'
    amplitude = 0 * np.pi / 180
    period = 3
    case_name += '_amp_' + str(amplitude).replace('.', '') + '_period_' + str(period)

# FLIGHT CONDITIONS
# the simulation is set such that the aircraft flies at a u_inf velocity while
# the air is calm.
u_inf = 6                                               # Freestream velocity (SI units)
rho = 1.225                                             # Density (SI units) - 400ft flight test is 1.231 kg/m3 density. Sea level 1.225

# trim sigma = 1.5                      
alpha = llta.alpha_deg * np.pi / 180                    # Angle of attack (rad)
beta = 0 # Side slip angle                              # Side slip angle (rad)
roll = 0 # Wings level                                  # Initial/static roll angle (rad) - wings level
gravity = True

cs_deflection = -2.08 * np.pi / 180                     # [NEEDS WORK] Control surface delfection
rudder_static_deflection = 0.0                          # Needed?
rudder_step = 0.0 * np.pi / 180                         # Needed? Step input to rudder (fin-tail coupling?)
thrust = 6.16
sigma = 1                                               # Stiffness scaling factor. Previously 1.5
lambda_dihedral = llta.lambda_dihedral * np.pi / 180    # Dihedral angle (rad)
alpha_zero_lift = llta.zero_alpha * np.pi / 180        # Zero alpha, nose-down = negative in SHARPy convetion

# gust settings
gust_intensity = 0.4 
gust_length = 0.5 * u_inf
gust_offset = 0.5 * u_inf

# numerics
n_step = 5                              # Changed Ramp up force applied over defined step count (previously 1)
structural_relaxation_factor = 0.6      # Changed Used in Static coupled solver settings (previously 0.3)
relaxation_factor = 0.35                # Changed Used in Dynamic coupled solver settings (previously 0.2)
tolerance = 1e-6
fsi_tolerance = 1e-4                    # More lenient convergence threshold than the default (1e-5)

num_cores = 2

# MODEL GEOMETRY
# aero
chord_main = llta.chord_main 
chord_tail = llta.chord_tail
chord_fin = llta.chord_fin 

# beam
span_main = llta.span_main / 2  # Semi-span
lambda_main = llta.lambda_main
ea_main = llta.ea_main          # [QUESTION] Position of elastic axis (as % of chord?) 

ea = llta.ea                            # Axial stiffness
ga = llta.ga                            # Shear stiffness
gj = llta.gj                            # Torsional stiffness
eiy = llta.eiy                          # Bending stiffness around the flapwise diration
eiz = llta.eiz                          # Bending stiffness around the edgewise direction
m_bar_main = llta.m_bar_main
j_bar_main = 0.075                      # Not defined by Voltitude - leave for now
ccg_main = (llta.cg_offset * chord_main) - (ea_main * chord_main) # Added during investigation of mass matrix and missing coupling terms. This is the CG offset from the elastic axis.

length_fuselage = llta.length_fuselage
offset_fuselage = 0                     # This is a fuselage that is angled - currently not needed
sigma_fuselage = 1                      # Scaling factor on top of sigma for fuselage stiffness - Voltitude assume same across board
m_bar_fuselage = llta.m_bar_fuselage
j_bar_fuselage = 0.08                   # Not defined by Voltitude - leave for now

span_tail = llta.span_tail
ea_tail = llta.ea_tail
fin_height = llta.fin_height
ea_fin = llta.ea_fin
sigma_tail = 1                          # Scaling factor on top of sigma for fuselage stiffness - Voltitude assume same across board
m_bar_tail = llta.m_bar_tail
j_bar_tail = 0.08                       # Not defined by Voltitude - leave for now

# lumped masses
n_lumped_mass = 5
lumped_mass_nodes = np.zeros((n_lumped_mass,), dtype=int)
lumped_mass = np.zeros((n_lumped_mass,))
lumped_mass_position = np.zeros((n_lumped_mass, 3))
lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))

# lumped masses - front fuselage (node 0)
lumped_mass_nodes[0] = 0 # attached to node 0
lumped_mass[0] = llta.mass_prop + llta.mass_batteries + llta.mass_pixhawk + llta.mass_additional # Changed (prop, batteriesm, pixhawk and 0.54 discrepancy)
lumped_mass_position[0] = llta.front_fuselage_pos # Offset position of payload (forward of node 0, so -ve y axis), average of positions
lumped_mass_inertia[0] = np.diag([0.001179, 0.005401, 0.005436]) # Combined forward fuselage — propeller + battery + electronics. Parallel axis theorem applied to combined CG at x = -0.2384 m

# lumped masses - right wing (node 4 and 16)
lumped_mass_nodes[1] = 4 # attached to node 4
lumped_mass[1] = llta.mass_launch_handle
lumped_mass_position[1] = llta.right_handle_pos # Material FoR, inwards of node 4, -ve local x axis
lumped_mass_inertia[1] = np.diag([1.814e-05, 2.294e-06, 1.814e-05]) # Launch handle (cylinder, spanwise axis) - negligable, not expected to affect results

lumped_mass_nodes[2] = 16 # attached to node 16
lumped_mass[2] = llta.mass_camera
lumped_mass_position[2] = llta.camera_pos
lumped_mass_inertia[2] = np.diag([3.143e-06, 3.780e-06, 4.800e-06]) # Camera (box 40x35x25 mm) - negligable, not expected to affect results

# lumped masses - left wing (node 20 and 32)
lumped_mass_nodes[3] = 20 # attached to node 20
lumped_mass[3] = llta.mass_launch_handle
lumped_mass_position[3] = llta.left_handle_pos # Material FoR, inwards of node 20, -ve local x axis
lumped_mass_inertia[3] = np.diag([1.814e-05, 2.294e-06, 1.814e-05]) # Launch handle (mirror of [1]) - negligable, not expected to affect results

lumped_mass_nodes[4] = 32 # attached to node 32
lumped_mass[4] = llta.mass_camera_counter
lumped_mass_position[4] = llta.camera_counter_pos
lumped_mass_inertia[4] = np.diag([3.143e-06, 3.780e-06, 4.800e-06]) # Same as camera estimates

#print("These are the lumped masses",lumped_mass)
#print("These are the lumped mass nodes",lumped_mass_nodes) #Added
#print("These are the lumped mass positions wrt their nodes", lumped_mass_position)

# DISCRETISATION
# spatial discretisation
# chordiwse panels
m = 4
# spanwise elements
n_elem_multiplier = 2
n_elem_main = int(4 * n_elem_multiplier)
n_elem_tail = int(2 * n_elem_multiplier)
n_elem_fin = int(2 * n_elem_multiplier)
n_elem_fuselage = int(2 * n_elem_multiplier)
n_surfaces = 5

# temporal discretisation
physical_time = 1 #30
tstep_factor = 0.75 #From line below, this is the only variable that makes sense to change, was 1.
dt = chord_main / (m * u_inf * tstep_factor)  # Updated so timestep is shedding of one chord length into wake
n_tstep = round(physical_time / dt)

# END OF INPUT-----------------------------------------------------------------

# beam processing
n_node_elem = 3
span_main1 = (1.0 - lambda_main) * span_main
span_main2 = lambda_main * span_main

n_elem_main1 = round(n_elem_main * (1 - lambda_main))
n_elem_main2 = n_elem_main - n_elem_main1
#print("Number of elements for span main 1:", n_elem_main1)
#print("Number of elements for span main 2:", n_elem_main2)

# total number of elements
n_elem = 0
n_elem += n_elem_main1 + n_elem_main1
n_elem += n_elem_main2 + n_elem_main2
n_elem += n_elem_fuselage
n_elem += n_elem_fin
n_elem += n_elem_tail + n_elem_tail
#print("Total elements:", n_elem)

# number of nodes per part
n_node_main1 = n_elem_main1 * (n_node_elem - 1) + 1
print("Number of nodes for main_elem 1 (inner right wing)", n_node_main1)
n_node_main2 = n_elem_main2 * (n_node_elem - 1) + 1
n_node_main = n_node_main1 + n_node_main2 - 1
n_node_fuselage = n_elem_fuselage * (n_node_elem - 1) + 1
print("This is the number of nodes for the fuselage",n_node_fuselage)
n_node_fin = n_elem_fin * (n_node_elem - 1) + 1
n_node_tail = n_elem_tail * (n_node_elem - 1) + 1

# total number of nodes
n_node = 0
n_node += n_node_main1 + n_node_main1 - 1
n_node += n_node_main2 - 1 + n_node_main2 - 1
n_node += n_node_fuselage - 1
n_node += n_node_fin - 1
n_node += n_node_tail - 1
n_node += n_node_tail - 1
print("The total number of nodes is:", n_node)

# stiffness and mass matrices
n_stiffness = 3
base_stiffness_main = sigma * np.diag([ea, ga, ga, gj, eiy, eiz])
base_stiffness_fuselage = base_stiffness_main.copy() * sigma_fuselage
base_stiffness_fuselage[4, 4] = base_stiffness_fuselage[5, 5] 
base_stiffness_tail = base_stiffness_main.copy() * sigma_tail
base_stiffness_tail[4, 4] = base_stiffness_tail[5, 5] 

n_mass = 3
# This was the original (and only line) -> base_mass_main = np.diag([m_bar_main, m_bar_main, m_bar_main, j_bar_main, 0.5 * j_bar_main, 0.5 * j_bar_main])

# Below section has been added during Mass Matrix Investigation
    # 6×6 mass matrix to include chordwise CG offset
    # SHARPy beam axis is y (spanwise). Chord lies in the x-z plane of the cross-section.
    # Ccg is the chordwise offset of the CG from the beam reference line (positive aft).
    # In SHARPy's cross-section frame, chordwise = local x, so the coupling term is m * ccg_main into position [0,4] and [4,0] (x-translation coupled with ry-rotation).
base_mass_main = np.diag([m_bar_main, m_bar_main, m_bar_main, j_bar_main, 0.5 * j_bar_main, 0.5 * j_bar_main])
#base_mass_main[0, 4] =  m_bar_main * ccg_main   # x-translation / pitch-rotation coupling
#base_mass_main[4, 0] =  m_bar_main * ccg_main   # symmetric (matrix must be symmetric)

base_mass_fuselage = np.diag([m_bar_fuselage,
                              m_bar_fuselage,
                              m_bar_fuselage,
                              j_bar_fuselage,
                              j_bar_fuselage * 0.5,
                              j_bar_fuselage * 0.5])
base_mass_tail = np.diag([m_bar_tail,
                          m_bar_tail,
                          m_bar_tail,
                          j_bar_tail,
                          j_bar_tail * 0.5,
                          j_bar_tail * 0.5])

# PLACEHOLDERS
# beam
x = np.zeros((n_node,))
y = np.zeros((n_node,))
z = np.zeros((n_node,))
beam_number = np.zeros((n_elem,), dtype=int)
frame_of_reference_delta = np.zeros((n_elem, n_node_elem, 3))
structural_twist = np.zeros((n_elem, 3))
conn = np.zeros((n_elem, n_node_elem), dtype=int)
stiffness = np.zeros((n_stiffness, 6, 6))
elem_stiffness = np.zeros((n_elem,), dtype=int)
mass = np.zeros((n_mass, 6, 6))
elem_mass = np.zeros((n_elem,), dtype=int)
boundary_conditions = np.zeros((n_node,), dtype=int)
app_forces = np.zeros((n_node, 6))

# aero
airfoil_distribution = np.zeros((n_elem, n_node_elem), dtype=int)
surface_distribution = np.zeros((n_elem,), dtype=int) - 1
surface_m = np.zeros((n_surfaces,), dtype=int)
m_distribution = 'uniform'
aero_node = np.zeros((n_node,), dtype=bool)
twist = np.zeros((n_elem, n_node_elem))
sweep = np.zeros((n_elem, n_node_elem))
chord = np.zeros((n_elem, n_node_elem,))
elastic_axis = np.zeros((n_elem, n_node_elem,))


# FUNCTIONS-------------------------------------------------------------
def clean_test_files():
    fem_file_name = route + '/' + case_name + '.fem.h5'
    if os.path.isfile(fem_file_name):
        os.remove(fem_file_name)

    dyn_file_name = route + '/' + case_name + '.dyn.h5'
    if os.path.isfile(dyn_file_name):
        os.remove(dyn_file_name)

    aero_file_name = route + '/' + case_name + '.aero.h5'
    if os.path.isfile(aero_file_name):
        os.remove(aero_file_name)

    solver_file_name = route + '/' + case_name + '.sharpy'
    if os.path.isfile(solver_file_name):
        os.remove(solver_file_name)

    flightcon_file_name = route + '/' + case_name + '.flightcon.txt'
    if os.path.isfile(flightcon_file_name):
        os.remove(flightcon_file_name)


def generate_dyn_file():
    global dt
    global n_tstep
    global route
    global case_name
    global num_elem
    global num_node_elem
    global num_node
    global amplitude
    global period
    global free_flight

    dynamic_forces_time = None
    with_dynamic_forces = False
    with_forced_vel = False
    if not free_flight:
        with_forced_vel = True

    if with_dynamic_forces:
        f1 = 100
        dynamic_forces = np.zeros((num_node, 6))
        app_node = [int(num_node_main - 1), int(num_node_main)]
        dynamic_forces[app_node, 2] = f1
        force_time = np.zeros((n_tstep,))
        limit = round(0.05 / dt)
        force_time[50:61] = 1

        dynamic_forces_time = np.zeros((n_tstep, num_node, 6))
        for it in range(n_tstep):
            dynamic_forces_time[it, :, :] = force_time[it] * dynamic_forces

    forced_for_vel = None
    if with_forced_vel:
        forced_for_vel = np.zeros((n_tstep, 6))
        forced_for_acc = np.zeros((n_tstep, 6))
        for it in range(n_tstep):
            # if dt*it < period:
            # forced_for_vel[it, 2] = 2*np.pi/period*amplitude*np.sin(2*np.pi*dt*it/period)
            # forced_for_acc[it, 2] = (2*np.pi/period)**2*amplitude*np.cos(2*np.pi*dt*it/period)

            forced_for_vel[it, 3] = 2 * np.pi / period * amplitude * np.sin(2 * np.pi * dt * it / period)
            forced_for_acc[it, 3] = (2 * np.pi / period) ** 2 * amplitude * np.cos(2 * np.pi * dt * it / period)

    if with_dynamic_forces or with_forced_vel:
        with h5.File(route + '/' + case_name + '.dyn.h5', 'a') as h5file:
            if with_dynamic_forces:
                h5file.create_dataset(
                    'dynamic_forces', data=dynamic_forces_time)
            if with_forced_vel:
                h5file.create_dataset(
                    'for_vel', data=forced_for_vel)
                h5file.create_dataset(
                    'for_acc', data=forced_for_acc)
            h5file.create_dataset(
                'num_steps', data=n_tstep)


def generate_fem():
    stiffness[0, ...] = base_stiffness_main
    stiffness[1, ...] = base_stiffness_fuselage
    stiffness[2, ...] = base_stiffness_tail

    mass[0, ...] = base_mass_main
    mass[1, ...] = base_mass_fuselage
    mass[2, ...] = base_mass_tail

    we = 0
    wn = 0
    # inner right wing
    beam_number[we:we + n_elem_main1] = 0
    y[wn:wn + n_node_main1] = np.linspace(0.0, span_main1, n_node_main1)

    for ielem in range(n_elem_main1):
        conn[we + ielem, :] = ((np.ones((3,)) * (we + ielem) * (n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]

    #print("Connectivies array for inner RIGHT wing)", conn)
    elem_stiffness[we:we + n_elem_main1] = 0
    elem_mass[we:we + n_elem_main1] = 0
    boundary_conditions[0] = 1
    # remember this is in B FoR
    app_forces[0] = [0, thrust, 0, 0, 0, 0] 
    we += n_elem_main1
    wn += n_node_main1

    # outer right wing
    beam_number[we:we + n_elem_main1] = 0
    y[wn:wn + n_node_main2 - 1] = y[wn - 1] + np.linspace(0.0, np.cos(lambda_dihedral) * span_main2, n_node_main2)[1:]
    z[wn:wn + n_node_main2 - 1] = z[wn - 1] + np.linspace(0.0, np.sin(lambda_dihedral) * span_main2, n_node_main2)[1:]
    for ielem in range(n_elem_main2):
        conn[we + ielem, :] = ((np.ones((3,)) * (we + ielem) * (n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
    
    #print("Connectivities array added to show outer RIGHT wing)",conn)
    elem_stiffness[we:we + n_elem_main2] = 0
    elem_mass[we:we + n_elem_main2] = 0
    boundary_conditions[wn + n_node_main2 - 2] = -1
    we += n_elem_main2
    wn += n_node_main2 - 1

    # inner left wing
    beam_number[we:we + n_elem_main1 - 1] = 1
    y[wn:wn + n_node_main1 - 1] = np.linspace(0.0, -span_main1, n_node_main1)[1:]
    for ielem in range(n_elem_main1):
        conn[we + ielem, :] = ((np.ones((3,)) * (we + ielem) * (n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
    conn[we, 0] = 0
    
    #print("Connectivities array added to show inner LEFT wing)",conn)
    elem_stiffness[we:we + n_elem_main1] = 0
    elem_mass[we:we + n_elem_main1] = 0
    we += n_elem_main1
    wn += n_node_main1 - 1

    # outer left wing
    beam_number[we:we + n_elem_main2] = 1
    y[wn:wn + n_node_main2 - 1] = y[wn - 1] + np.linspace(0.0, -np.cos(lambda_dihedral) * span_main2, n_node_main2)[1:]
    z[wn:wn + n_node_main2 - 1] = z[wn - 1] + np.linspace(0.0, np.sin(lambda_dihedral) * span_main2, n_node_main2)[1:]
    for ielem in range(n_elem_main2):
        conn[we + ielem, :] = ((np.ones((3,)) * (we + ielem) * (n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
    
    #print("Connectivities array added to show outer LEFT wing)",conn)
    elem_stiffness[we:we + n_elem_main2] = 0
    elem_mass[we:we + n_elem_main2] = 0
    boundary_conditions[wn + n_node_main2 - 2] = -1
    we += n_elem_main2
    wn += n_node_main2 - 1

    # fuselage
    beam_number[we:we + n_elem_fuselage] = 2
    x[wn:wn + n_node_fuselage - 1] = np.linspace(0.0, length_fuselage, n_node_fuselage)[1:]
    z[wn:wn + n_node_fuselage - 1] = np.linspace(0.0, offset_fuselage, n_node_fuselage)[1:]
    for ielem in range(n_elem_fuselage):
        conn[we + ielem, :] = ((np.ones((3,)) * (we + ielem) * (n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
    conn[we, 0] = 0 #TO DO: Hard coding in assigned node for where the wings are on the fuselage - need to fix so it changes with discretisation

    print("Connectivities array added to show fueslage)",conn)
    elem_stiffness[we:we + n_elem_fuselage] = 1
    elem_mass[we:we + n_elem_fuselage] = 1
    we += n_elem_fuselage
    wn += n_node_fuselage - 1
    global end_of_fuselage_node
    end_of_fuselage_node = wn - 1

    # fin
    beam_number[we:we + n_elem_fin] = 3
    x[wn:wn + n_node_fin - 1] = x[end_of_fuselage_node]
    z[wn:wn + n_node_fin - 1] = z[end_of_fuselage_node] + np.linspace(0.0, fin_height, n_node_fin)[1:]
    for ielem in range(n_elem_fin):
        conn[we + ielem, :] = ((np.ones((3,)) * (we + ielem) * (n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
    conn[we, 0] = end_of_fuselage_node

    #print("Connectivities array added to show fin)",conn)
    elem_stiffness[we:we + n_elem_fin] = 2
    elem_mass[we:we + n_elem_fin] = 2
    we += n_elem_fin
    wn += n_node_fin - 1
    end_of_fin_node = wn - 1
    #print("End fin node: ", end_of_fin_node)

    # right tail - respositioned
    beam_number[we:we + n_elem_tail] = 4
    x[wn:wn + n_node_tail - 1] = x[end_of_fuselage_node] #Set the x coordinates to same as last fin node (but we want fuselage)
    y[wn:wn + n_node_tail - 1] = np.linspace(0.0, span_tail, n_node_tail)[1:]
    z[wn:wn + n_node_tail - 1] = z[end_of_fuselage_node]
    for ielem in range(n_elem_tail):
        conn[we + ielem, :] = ((np.ones((3,)) * (we + ielem) * (n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
    #print("Current link to fin at node: ", end_of_fin_node, "Expect to link at node: ", end_of_fuselage_node)
    conn[we, 0] = end_of_fuselage_node

    #print("Connectivities array added to show RIGHT tail)",conn)
    elem_stiffness[we:we + n_elem_tail] = 2
    elem_mass[we:we + n_elem_tail] = 2
    boundary_conditions[wn + n_node_tail - 2] = -1
    we += n_elem_tail
    wn += n_node_tail - 1

    # left tail - repositioned
    beam_number[we:we + n_elem_tail] = 5
    x[wn:wn + n_node_tail - 1] = x[end_of_fuselage_node]
    y[wn:wn + n_node_tail - 1] = np.linspace(0.0, -span_tail, n_node_tail)[1:]
    z[wn:wn + n_node_tail - 1] = z[end_of_fuselage_node]
    for ielem in range(n_elem_tail):
        conn[we + ielem, :] = ((np.ones((3,)) * (we + ielem) * (n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
    conn[we, 0] = end_of_fuselage_node

    print("Connectivities array added to show LEFT tail)",conn)
    elem_stiffness[we:we + n_elem_tail] = 2
    elem_mass[we:we + n_elem_tail] = 2
    boundary_conditions[wn + n_node_tail - 2] = -1
    we += n_elem_tail
    wn += n_node_tail - 1

    with h5.File(route + '/' + case_name + '.fem.h5', 'a') as h5file:
        coordinates = h5file.create_dataset('coordinates', data=np.column_stack((x, y, z)))
        conectivities = h5file.create_dataset('connectivities', data=conn)
        num_nodes_elem_handle = h5file.create_dataset(
            'num_node_elem', data=n_node_elem)
        num_nodes_handle = h5file.create_dataset(
            'num_node', data=n_node)
        num_elem_handle = h5file.create_dataset(
            'num_elem', data=n_elem)
        stiffness_db_handle = h5file.create_dataset(
            'stiffness_db', data=stiffness)
        stiffness_handle = h5file.create_dataset(
            'elem_stiffness', data=elem_stiffness)
        mass_db_handle = h5file.create_dataset(
            'mass_db', data=mass)
        mass_handle = h5file.create_dataset(
            'elem_mass', data=elem_mass)
        frame_of_reference_delta_handle = h5file.create_dataset(
            'frame_of_reference_delta', data=frame_of_reference_delta)
        structural_twist_handle = h5file.create_dataset(
            'structural_twist', data=structural_twist)
        bocos_handle = h5file.create_dataset(
            'boundary_conditions', data=boundary_conditions)
        beam_handle = h5file.create_dataset(
            'beam_number', data=beam_number)
        app_forces_handle = h5file.create_dataset(
            'app_forces', data=app_forces)
        lumped_mass_nodes_handle = h5file.create_dataset(
            'lumped_mass_nodes', data=lumped_mass_nodes)
        lumped_mass_handle = h5file.create_dataset(
            'lumped_mass', data=lumped_mass)
        lumped_mass_inertia_handle = h5file.create_dataset(
            'lumped_mass_inertia', data=lumped_mass_inertia)
        lumped_mass_position_handle = h5file.create_dataset(
            'lumped_mass_position', data=lumped_mass_position)


def generate_aero_file():
    global x, y, z
    # control surfaces
    n_control_surfaces = 2
    control_surface = np.zeros((n_elem, n_node_elem), dtype=int) - 1
    control_surface_type = np.zeros((n_control_surfaces,), dtype=int)
    control_surface_deflection = np.zeros((n_control_surfaces,))
    control_surface_chord = np.zeros((n_control_surfaces,), dtype=int)
    control_surface_hinge_coord = np.zeros((n_control_surfaces,), dtype=float)

    # control surface type 0 = static
    # control surface type 1 = dynamic
    control_surface_type[0] = 0
    control_surface_deflection[0] = cs_deflection
    control_surface_chord[0] = m
    control_surface_hinge_coord[0] = -0.25  # nondimensional wrt elastic axis (+ towards the trailing edge)

    control_surface_type[1] = 0
    control_surface_deflection[1] = rudder_static_deflection
    control_surface_chord[1] = 1
    control_surface_hinge_coord[1] = -0.  # nondimensional wrt elastic axis (+ towards the trailing edge)

    we = 0
    wn = 0
    # right wing (surface 0, beam 0)
    i_surf = 0
    airfoil_distribution[we:we + n_elem_main, :] = 0
    surface_distribution[we:we + n_elem_main] = i_surf
    surface_m[i_surf] = m
    aero_node[wn:wn + n_node_main] = True
    temp_chord = np.linspace(chord_main, chord_main, n_node_main)
    temp_sweep = np.linspace(0.0, 0 * np.pi / 180, n_node_main)
    node_counter = 0
    for i_elem in range(we, we + n_elem_main):
        for i_local_node in range(n_node_elem):
            if not i_local_node == 0:
                node_counter += 1
            chord[i_elem, i_local_node] = temp_chord[node_counter]
            elastic_axis[i_elem, i_local_node] = ea_main
            sweep[i_elem, i_local_node] = temp_sweep[node_counter]

    we += n_elem_main
    wn += n_node_main

    # left wing (surface 1, beam 1)
    i_surf = 1
    airfoil_distribution[we:we + n_elem_main, :] = 0
    # airfoil_distribution[wn:wn + n_node_main - 1] = 0
    surface_distribution[we:we + n_elem_main] = i_surf
    surface_m[i_surf] = m
    aero_node[wn:wn + n_node_main - 1] = True
    # chord[wn:wn + num_node_main - 1] = np.linspace(main_chord, main_tip_chord, num_node_main)[1:]
    # chord[wn:wn + num_node_main - 1] = main_chord
    # elastic_axis[wn:wn + num_node_main - 1] = main_ea
    temp_chord = np.linspace(chord_main, chord_main, n_node_main)
    node_counter = 0
    for i_elem in range(we, we + n_elem_main):
        for i_local_node in range(n_node_elem):
            if not i_local_node == 0:
                node_counter += 1
            chord[i_elem, i_local_node] = temp_chord[node_counter]
            elastic_axis[i_elem, i_local_node] = ea_main
            sweep[i_elem, i_local_node] = -temp_sweep[node_counter]

    we += n_elem_main
    wn += n_node_main - 1

    we += n_elem_fuselage
    wn += n_node_fuselage - 1 -1
    #
    # # fin (surface 2, beam 3)
    i_surf = 2
    airfoil_distribution[we:we + n_elem_fin, :] = 1
    # airfoil_distribution[wn:wn + n_node_fin] = 0
    surface_distribution[we:we + n_elem_fin] = i_surf
    surface_m[i_surf] = m
    aero_node[wn:wn + n_node_fin] = True
    # chord[wn:wn + num_node_fin] = fin_chord
    for i_elem in range(we, we + n_elem_fin):
        for i_local_node in range(n_node_elem):
            chord[i_elem, i_local_node] = chord_fin
            elastic_axis[i_elem, i_local_node] = ea_fin
            control_surface[i_elem, i_local_node] = 1
    # twist[end_of_fuselage_node] = 0
    # twist[wn:] = 0
    # elastic_axis[wn:wn + num_node_main] = fin_ea
    we += n_elem_fin
    wn += n_node_fin - 1
    #
    # # # right tail (surface 3, beam 4)
    i_surf = 3
    airfoil_distribution[we:we + n_elem_tail, :] = 2
    # airfoil_distribution[wn:wn + n_node_tail] = 0
    surface_distribution[we:we + n_elem_tail] = i_surf
    surface_m[i_surf] = m
    # XXX not very elegant
    aero_node[wn:] = True
    # chord[wn:wn + num_node_tail] = tail_chord
    # elastic_axis[wn:wn + num_node_main] = tail_ea
    for i_elem in range(we, we + n_elem_tail):
        for i_local_node in range(n_node_elem):
            twist[i_elem, i_local_node] = -0
    for i_elem in range(we, we + n_elem_tail):
        for i_local_node in range(n_node_elem):
            chord[i_elem, i_local_node] = chord_tail
            elastic_axis[i_elem, i_local_node] = ea_tail
            control_surface[i_elem, i_local_node] = 0

    we += n_elem_tail
    wn += n_node_tail
    #
    # # left tail (surface 4, beam 5)
    i_surf = 4
    airfoil_distribution[we:we + n_elem_tail, :] = 2
    # airfoil_distribution[wn:wn + n_node_tail - 1] = 0
    surface_distribution[we:we + n_elem_tail] = i_surf
    surface_m[i_surf] = m
    aero_node[wn:wn + n_node_tail - 1] = True
    # chord[wn:wn + num_node_tail] = tail_chord
    # elastic_axis[wn:wn + num_node_main] = tail_ea
    # twist[we:we + num_elem_tail] = -tail_twist
    for i_elem in range(we, we + n_elem_tail):
        for i_local_node in range(n_node_elem):
            twist[i_elem, i_local_node] = -0
    for i_elem in range(we, we + n_elem_tail):
        for i_local_node in range(n_node_elem):
            chord[i_elem, i_local_node] = chord_tail
            elastic_axis[i_elem, i_local_node] = ea_tail
            control_surface[i_elem, i_local_node] = 0
    we += n_elem_tail
    wn += n_node_tail

    '''
    """Debugging nodes for aero vs fem"""
    print("aero_node assignments:")
    for i, has_aero in enumerate(aero_node):
        print(f"  node {i}: aero={has_aero}, x={x[i]:.3f}, y={y[i]:.3f}, z={z[i]:.3f}")

    print("surface_distribution assignments:")
    for i, surf in enumerate(surface_distribution):
        print(f"  elem {i}: surface={surf}")
    '''
    with h5.File(route + '/' + case_name + '.aero.h5', 'a') as h5file:
        airfoils_group = h5file.create_group('airfoils')
        # add one airfoil
        naca_airfoil_main = airfoils_group.create_dataset('0', data=np.column_stack(
            generate_naca_camber(P=0, M=0)))
        naca_airfoil_tail = airfoils_group.create_dataset('1', data=np.column_stack(
            generate_naca_camber(P=0, M=0)))
        naca_airfoil_fin = airfoils_group.create_dataset('2', data=np.column_stack(
            generate_naca_camber(P=0, M=0)))

        # chord
        chord_input = h5file.create_dataset('chord', data=chord)
        dim_attr = chord_input.attrs['units'] = 'm'

        # twist
        twist_input = h5file.create_dataset('twist', data=twist)
        dim_attr = twist_input.attrs['units'] = 'rad'

        # sweep
        sweep_input = h5file.create_dataset('sweep', data=sweep)
        dim_attr = sweep_input.attrs['units'] = 'rad'

        # airfoil distribution
        airfoil_distribution_input = h5file.create_dataset('airfoil_distribution', data=airfoil_distribution)

        surface_distribution_input = h5file.create_dataset('surface_distribution', data=surface_distribution)
        surface_m_input = h5file.create_dataset('surface_m', data=surface_m)
        m_distribution_input = h5file.create_dataset('m_distribution', data=m_distribution.encode('ascii', 'ignore'))

        aero_node_input = h5file.create_dataset('aero_node', data=aero_node)
        elastic_axis_input = h5file.create_dataset('elastic_axis', data=elastic_axis)

        control_surface_input = h5file.create_dataset('control_surface', data=control_surface)
        control_surface_deflection_input = h5file.create_dataset('control_surface_deflection',
                                                                 data=control_surface_deflection)
        control_surface_chord_input = h5file.create_dataset('control_surface_chord', data=control_surface_chord)
        control_surface_hinge_coord_input = h5file.create_dataset('control_surface_hinge_coord',
                                                                  data=control_surface_hinge_coord)
        control_surface_types_input = h5file.create_dataset('control_surface_type', data=control_surface_type)


def generate_naca_camber(M=0, P=0):
    mm = M * 1e-2
    p = P * 1e-1

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


def generate_solver_file():
    file_name = route + '/' + case_name + '.sharpy'
    settings = dict()
    settings['SHARPy'] = {'case': case_name,
                          'route': route,
                          'flow': flow,
                          'write_screen': 'on',
                          'write_log': 'on',
                          'log_folder': route + '/output/',
                          'log_file': case_name + '.log'}

    settings['BeamLoader'] = {'unsteady': 'on',
                              'orientation': algebra.euler2quat(np.array([roll,
                                                                          alpha,
                                                                          beta]))} # Default is: [1.0, 0, 0, 0] 
    settings['AerogridLoader'] = {'unsteady': 'on',
                                  'aligned_grid': 'on',
                                  'mstar': int(20 / tstep_factor),
                                  'freestream_dir': ['1', '0', '0'],
                                  'wake_shape_generator': 'StraightWake',
                                  'wake_shape_generator_input': {'u_inf': u_inf,
                                                                 'u_inf_direction': ['1', '0', '0'],
                                                                 'dt': dt}}

    settings['NonLinearStatic'] = {'print_info': 'off',
                                   'max_iterations': 150, 
                                   'num_load_steps': 1, 
                                   'delta_curved': 1e-1, # What is this setting?
                                   'min_delta': tolerance,
                                   'gravity_on': gravity,
                                   'gravity': 9.81}

    settings['StaticUvlm'] = {'print_info': 'on',
                              'horseshoe': 'off',
                              'num_cores': num_cores,
                              'n_rollup': 0,
                              'rollup_dt': dt,
                              'rollup_aic_refresh': 1,
                              'rollup_tolerance': 1e-4, 
                              'velocity_field_generator': 'SteadyVelocityField',
                              'velocity_field_input': {'u_inf': u_inf,
                                                       'u_inf_direction': [1., 0, 0]},
                              'rho': rho}

    settings['StaticCoupled'] = {'print_info': 'on',
                                 'structural_solver': 'NonLinearStatic',
                                 'structural_solver_settings': settings['NonLinearStatic'],
                                 'aero_solver': 'StaticUvlm',
                                 'aero_solver_settings': settings['StaticUvlm'],
                                 'max_iter': 100,
                                 'n_load_steps': n_step,
                                 'tolerance': fsi_tolerance,
                                 'relaxation_factor': structural_relaxation_factor}

    settings['StaticTrim'] = {'solver': 'StaticCoupled',
                              'solver_settings': settings['StaticCoupled'],
                              'initial_alpha': alpha,
                              'initial_deflection': cs_deflection,
                              'initial_thrust': thrust,
                              'save_info': True}

    settings['NonLinearDynamicCoupledStep'] = {'print_info': 'off',
                                               'max_iterations': 950,
                                               'delta_curved': 1e-1,
                                               'min_delta': tolerance,
                                               'newmark_damp': 5e-3,
                                               'gravity_on': gravity,
                                               'gravity': 9.81,
                                               'num_steps': n_tstep,
                                               'dt': dt,
                                               'initial_velocity': u_inf}

    settings['NonLinearDynamicPrescribedStep'] = {'print_info': 'off',
                                                  'max_iterations': 950,
                                                  'delta_curved': 1e-1,
                                                  'min_delta': tolerance,
                                                  'newmark_damp': 5e-3,
                                                  'gravity_on': gravity,
                                                  'gravity': 9.81,
                                                  'num_steps': n_tstep,
                                                  'dt': dt,
                                                  'initial_velocity': u_inf * int(free_flight)}

    
    #Solver settings for modal analysis - all settings delcared and set to defaults - [COMPARE AGAINST GENERATE_HALE_INTERIM]
    settings['Modal'] = {'print_info': True,
                         'rigid_body_modes': False, #generate_hale.py set to True
                         'use_undamped_modes': True,
                         'NumLambda': 20, #generate_hale.py set to 30, this doubles if use_undamped_modes is set to False
                         'write_modes_vtk': True,
                         'print_matrices': False,
                         'save_data': True,
                         'continuous_eigenvalues': False,
                         'dt': 0, #generate_hale.py set to dt
                         'delta_curved': 0.01,
                         'plot_eigenvalues': False, 
                         'max_rotation_deg': 15.0, 
                         'max_displacement': 0.15, 
                         'use_custom_timestep': -1, 
                         'rigid_modes_ppal_axes': False, 
                         'rigid_modes_cg': False}

    relative_motion = 'off'
    if not free_flight:
        relative_motion = 'on'
    settings['StepUvlm'] = {'print_info': 'on',
                            'num_cores': num_cores,
                            'convection_scheme': 2,
                            'gamma_dot_filtering': 6,
                            'velocity_field_generator': 'GustVelocityField',
                            'velocity_field_input': {'u_inf': u_inf, # previously int(not free_flight) * u_inf,
                                                     'u_inf_direction': [1., 0, 0],
                                                     'gust_shape': '1-cos',
                                                     'gust_parameters': {'gust_length': gust_length,
                                                                         'gust_intensity': gust_intensity * u_inf},
                                                     'offset': gust_offset,
                                                     'relative_motion': relative_motion},
                            'rho': rho,
                            'n_time_steps': n_tstep,
                            'dt': dt}

    if free_flight:
        solver = 'NonLinearDynamicCoupledStep'
    else:
        solver = 'NonLinearDynamicPrescribedStep'
    settings['DynamicCoupled'] = {'structural_solver': solver,
                                  'structural_solver_settings': settings[solver],
                                  'aero_solver': 'StepUvlm',
                                  'aero_solver_settings': settings['StepUvlm'],
                                  'fsi_substeps': 200, #default is 200 - tried changing to divide into more steps before one aero step is done, didn't seem to work
                                  'fsi_tolerance': fsi_tolerance,
                                  'relaxation_factor': relaxation_factor,
                                  'minimum_steps': 1,
                                  'relaxation_steps': 150,
                                  #'structural_substeps': 4, #New setting declared as part of troubleshooting Singular Matrix (tip from Kelvin)
                                  'final_relaxation_factor': 0.5,
                                  'n_time_steps': n_tstep,
                                  'dt': dt,
                                  'include_unsteady_force_contribution': 'on',
                                  'postprocessors': ['BeamLoads', 'BeamPlot', 'AerogridPlot', 'WriteVariablesTime'], #, 'PlotFlowField'],
                                  'postprocessors_settings': {'BeamLoads': {'csv_output': 'on'},
                                                              'BeamPlot': {'include_rbm': True,
                                                                           'include_applied_forces': 'on'},
                                                              'AerogridPlot': {
                                                                  'include_rbm': True,
                                                                  'include_applied_forces': 'on',
                                                                  'minus_m_star': 0},
                                                              #'PlotFlowField': {'velocity_field_input': u_inf}
                                                              'WriteVariablesTime' : {
                                                                'structure_nodes': 16, # Free-end node ID
                                                                'structure_variables': ['pos'],
                                                                  #'cleanup_old_solution': 'on',
                                                                  },
                                                              }}

    settings['BeamLoads'] = {'csv_output': 'off'}

    settings['BeamPlot'] = {'include_rbm': True,
                            'include_FoR': True,
                            'include_applied_forces': 'on'}


    settings['AerogridPlot'] = {'include_rbm': True,
                                'include_forward_motion': 'off',
                                'include_applied_forces': 'on',
                                'minus_m_star': 0,
                                'u_inf': u_inf,
                                'dt': dt}

    settings['LinearAssembler'] = {'linear_system': 'LinearAeroelastic',
                                   'linear_system_settings': {
                                       'beam_settings': {'modal_projection': False,
                                                         'inout_coords': 'nodes',
                                                         'discrete_time': True,
                                                         'newmark_damp': 0.05,
                                                         'discr_method': 'newmark',
                                                         'dt': dt,
                                                         'proj_modes': 'undamped',
                                                         'use_euler': 'off',
                                                         'num_modes': 40,
                                                         'print_info': 'on',
                                                         'gravity': 'on',
                                                         'remove_dofs': []},
                                       'aero_settings': {'dt': dt,
                                                         'integr_order': 2,
                                                         'density': rho,
                                                         'remove_predictor': False,
                                                         'use_sparse': True,
                                                         'remove_inputs': ['u_gust']}
                                   }}

    settings['AsymptoticStability'] = {'print_info': 'on',
                                       'modes_to_plot': [],
                                       'display_root_locus': 'off',
                                       'frequency_cutoff': 0,
                                       'export_eigenvalues': 'off',
                                       'num_evals': 40}
    
    #settings['PlotFlowField'] = {'velocity_field_input': u_inf}

    import configobj
    config = configobj.ConfigObj()
    config.filename = file_name
    for k, v in settings.items():
        config[k] = v
    config.write()

clean_test_files()
generate_fem()
generate_aero_file()
generate_solver_file()
generate_dyn_file()

# Run Sharpy case - added to be able to run Sharpy case automatically and get data object
# Pull out raw data for displacement here to assess instead of visualising in Paraview
data = sharpy.sharpy_main.main(['', route + '/' + case_name + '.sharpy'])

"""Data Analysis [CLEAN UP]"""
#timestep_data = data.structure.timestep_info[-1]
#print(vars(timestep_data).keys())

#print("The steady applied forces are:", timestep_data.pos)

#print("Structure (timestep_info): ", data.structure.timestep_info)
#print("Structure (ini_info): ", data.structure.ini_info)
#print("Settings: ", dir(data.structure))
#print('All variables in data object:', vars(data))

'''def explore(obj, name='data', depth=0, max_depth=3):
    indent = '  ' * depth
    try:
        attrs = vars(obj).keys()
    except TypeError:
        print(f"{indent}{name} = {type(obj)}")
        return
    for attr in attrs:
        child = getattr(obj, attr)
        print(f"{indent}.{attr}  ({type(child).__name__})")
        if depth < max_depth and hasattr(child, '__dict__'):
            explore(child, attr, depth+1, max_depth)

explore(data)'''
'''
boundary_conditions = data.structure.boundary_conditions
print("Boundary conditions:", boundary_conditions)

# Inspect immediately after
stiffness_db = data.structure.stiffness_db
elem_stiffness = data.structure.elem_stiffness
print("Stiffness matrix:", stiffness_db)
print("Element stiffness assignments:", elem_stiffness)

"""mass_db = data.structure.mass_db
elem_mass = data.structure.elem_mass
print("Mass matrix:", mass_db)
print("Element mass assignments:", mass_db)

for i, K in enumerate(stiffness_db):
    print(f'Entry {i}: EI1={K[5,5]:.4e}, EI2={K[4,4]:.4e}, ratio={K[4,4]/K[5,5]:.4f}')

#for i, s in enumerate(elem_stiffness):
#    print(f'  Element {i}: stiffness entry {s}')

# Summary count
for entry in [0, 1, 2]:
    count = np.sum(elem_stiffness == entry)
    print(f'Stiffness entry {entry} used by {count} elements')"""


#Static Trim Analysis (Equilibrium State)
# Displacement of each beam node at the last timestep
ts = data.structure.timestep_info[-1]
print(vars(ts).keys())  # see all available fields

print("Node deformed positions:", ts.pos)          # nodal positions (deformed), shape (n_nodes, 3), which is x,y,z for each node?
#print("Node velocities:", ts.pos_dot)      # nodal velocities
#print(ts.psi)          # CRV (Cartesian Rotation Vectors) at each element. Wing twist, references/coordinate systems compared. Standard 'output'. Think about how these can be compared with ASWing. 

pos_ini = data.structure.ini_info.pos
np.savetxt('pos_ini.txt', pos_ini)
print(pos_ini)'''
