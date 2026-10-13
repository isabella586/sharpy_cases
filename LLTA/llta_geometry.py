import numpy as np

#LLTA Variables - Sharpy Inputs

aircraft_mass = 2.4 # kg (Note this is not a SHARPy input but used for model validation checks)

#alpha_deg = 4.31        # WHAT IS THIS USED FOR - Angle of attack (rad). Voltitude state 5 for LLTA

#cs_deflection = 0       # PLACEHOLDER 

#zero_alpha = -6.2 * np.pi / 180       # USED? Zero-lift AoA (degrees) - TO DO: + or -?

# AERO PANEL GEOMETRY
# main wing
chord_main = 0.225 # From ASWing file (m)
ea_main = 0.35 # From ASWing file (fraction) - ASWing definition is distance/chord from leading edge to s axis (c,n origin)
print("Position of elastic axis on main wing is:", (ea_main*chord_main), "m from leading edge. NOTE: ea_main is set as a fraction") 
cg_offset_main = 0.02 # From ASWing file - main wing CG offset (m) aft of elastic axis
print("Position of the main wing CG is:", (ea_main*chord_main + cg_offset_main), "m from leading edge.")
wing_incidence = -5 * (np.pi/180)

# tail/elevator
chord_tail = 0.135 # From ASWing file (m)
ea_tail = 0.25 # From ASWing file (fraction) - ASWing definition is distance/chord from leading edge to s axis (c,n origin)

# fin/rudder
chord_fin = 0.135 # From ASWing file (m)
ea_fin = 0.25 # From ASWing file (fraction) - ASWing definition is distance/chord from leading edge to s axis (c,n origin)

# BEAM STRUCTURE GEOMETRY
# main wing
span_main = 3.68 # From ASWing file (m) - NOTE: ASWing file states 3.64m. Drawing states 3.75m. HALE default -> 16.0
lambda_main = 0.5 # Proportion of the span that is dihedral
lambda_dihedral = 12.5  # dihedral angle (deg)
t_main = 0.02 # 20mm wing thickness (assumed)

# fuselage
length_fuselage = 0.87 # From LLTA Geometry Dimensions_updated.png (tail lever arm)
print("Fuselage length (m):", length_fuselage)

# tail/elevator
span_tail = 0.72 # NOTE: 0.72m is from ASWing output file of 9m/s trim run. Drawing states 0.735 #2.5
t_tail = 0.02 # 20mm wing thickness (assumed)

# fin/rudder
fin_height = 0.36 # NOTE: 0.36m is from ASWing output file of 9m/s trim run. Drawing states 0.375 #2.5

# STRUCTURAL PROPERTIES
ea = 1e7 # Axial stiffness - Sharpy default
ga = 1e5 # Shear stiffness
gj = 20.87 # Torsional stiffness - Voltitude defined (19.5 old)
eiy = 51.25 # Bending stiffness - Voltitude defined (35.54 old)
eiz = 2e4 #2e4 # Sharpy default (HALE)

gj_fuselage = 7.16 # Voltitude defined
eiy_fuselage = 71.71 # axis changes for the fuselage
eiz_fuselage = 71.71 # Voltitude defined
print("Fuselage stiffness properties:", gj_fuselage, eiy_fuselage, eiz_fuselage)

# MASS PROPERTIES
m_bar_main = 2.83 / 9.81 # From ASWing file (N/m converted to kg/m)
#j_bar_main = 0.075 # [CHECK] previously m_bar_main / 2 # Per Sharpy default, setting this to x10 less order of magnitude
#j_bar_main = 1600 * ((np.pi / 32) * ((0.01414**4)-(0.01176**4))) #  Previously SPAR J (for early spar analysis) -> material density * (pi/32)*(OD^4-ID^4) rotational inertia per unit length (all three rotational DOFs), was carried over from generate_hale -> 0.075 (not correct)
j_spanwise_main = m_bar_main * ((chord_main**2) + (t_main**2)) / 12 # Torsion (spanwise axis), dependant on chord and thickness, assumed 20mm as wing box thickness
j_chordwise_main = m_bar_main * (t_main**2) / 12 # Chordwise axis, dependant on wing box thickness
j_normal_main = m_bar_main * (chord_main**2) / 12 # In-plane (normal axis), dependant on chord
print("Updated j values for main wing:", j_spanwise_main, j_chordwise_main, j_normal_main)

m_bar_tail = 0.9 / 9.81 # From ASWing file (N/m converted to kg/m). NOTE: the fin is the same but not accounted for here??? [TO DO]
#j_bar_tail = 0.08 # Not pulled through yet (Voltitude not defined)
#j_bar_tail = 1600 * ((np.pi / 32) * ((0.01414**4)-(0.01176**4))) #  Previously SPAR J (for early spar analysis) -> material density * (pi/32)*(OD^4-ID^4) rotational inertia per unit length (all three rotational DOFs), was carried over from generate_hale -> 0.075 (not correct)
j_spanwise_tail = m_bar_tail * ((chord_tail**2) + (t_tail**2)) / 12 # Torsion (spanwise axis), dependant on chord and thickness, assumed 20mm as wing box thickness
j_chordwise_tail = m_bar_tail * (t_tail**2) / 12 # Chordwise axis, dependant on wing box thickness
j_normal_tail = m_bar_tail * (chord_tail**2) / 12 # In-plane (normal axis), dependant on chord
print("Updated j values for tail:", j_spanwise_tail, j_chordwise_tail, j_normal_tail)

m_bar_fuselage = 0.686 / 9.81 # From ASWing file (N/m converted to kg/m). NOTE: this is the main fuselage, not front, that is 1.434 from ASWing file
#j_bar_fuselage = 0.08 # Not pulled through yet (Voltitude not defined)
j_bar_fuselage = 1600 * ((np.pi / 32) * ((0.01414**4)-(0.01176**4))) #  SPAR J -> material density * (pi/32)*(OD^4-ID^4) rotational inertia per unit length (all three rotational DOFs), was carried over from generate_hale -> 0.075 (not correct)

# Check parallel axis theorem
assert np.isclose(j_spanwise_main, j_chordwise_main + j_normal_main), "wing j_x != j_y + j_z"
assert np.isclose(j_spanwise_tail, j_chordwise_tail + j_normal_tail), "tail j_x != j_y + j_z"

# LUMPED MASSES
mass_prop = 0.5 / 9.81
mass_batteries = 3.25 / 9.81
mass_pixhawk = 0.77 / 9.81
mass_additional = 0.55
mass_pod_struct = (1.434 / 9.81) * 0.41

mass_launch_handle = 0.02
mass_camera = 0.02
mass_camera_counter = 0.02

# Positions - LOCAL (Material FoR -> x is nodes out to wing tip, y is 
additional_avionics = [0.0, 0.0, 0.0] 
payload_bay = [0.0, 0.238, 0.0]
pod_structure = [0.0, 0.205, 0.0] 
right_handle_pos = [-0.023750, 0.0, 0.0]
left_handle_pos = [-0.023750, 0.0, 0.0]
camera_pos = [0.0, 0.0, 0.0]
camera_counter_pos = [0.0, 0.0, 0.0]

