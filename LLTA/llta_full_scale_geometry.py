import numpy as np

# =====================================================================================
# FULL-SCALE TWIN-TAIL PROTOTYPE (Voltitude HALE) - SHARPy inputs
# Derived from llta_geometry.py (LLTA, 1/3 scale, single boom). Oct 2026.
#
# Every number below that is NOT a Voltitude full-scale value is tagged [GAP-n] and listed in
# the GAP REGISTER. Replace a tagged value with the real number as soon as you have it, delete
# the tag, and the rest of the model follows (generate_llta_fullscale_trim.py only reads this file).
#
# GAP REGISTER
#  GAP-1  Boom spanwise station. Read off an un-scaled oblique CAD view -> assumed at the dihedral kink (50% semi-span).
#  GAP-2  Boom length. Assumed 3 x LLTA tail arm. The CAD booms LOOK proportionally longer (perspective-limited).
#  GAP-3  Tail size per boom. Assumed each boom carries a 3 x LLTA tailplane (centred on the boom) + 3 x LLTA fin.
#         => total tail area is 2 x the scaled LLTA tail. Change span_tail / fin_height / chord_* if split differently.
#  GAP-4  Stiffness (EA, GA, GJ, EI) of wing / boom / tail / fin: Froude-scaled LLTA (Voltitude full-scale values needed).
#  GAP-5  Mass per length (wing, boom, tail/fin) and wing CG offset: scaled LLTA (Voltitude full-scale values needed).
#  GAP-6  Lumped masses (prop, battery, Pixhawk, avionics, pod structure, handles, cameras): all scaled x27 from LLTA.
#         LLTA-specific hardware (launch handles, cameras) is probably NOT on the prototype. Outboard pods not modelled.
#  GAP-7  Propulsion: single thrust vector at node 0 (as LLTA). Number/position of motors on the prototype unknown.
#  GAP-8  Flight condition: u_inf = 6.5 m/s x sqrt(3) (Froude) and sea-level density. Real design cruise speed unknown.
#  GAP-9  Control surfaces: full-chord elevator and rudder (LLTA modelling choice). Both elevators share one trim
#         variable, both rudders share another. No ailerons. Real hinge lines / surface chords unknown.
#  GAP-10 Wing planform (chord, taper, sweep, dihedral angle, airfoil/camber): assumed 3 x LLTA (rectangular, 12.5 deg, M6P4).
#  GAP-11 Boom / pod / tail vertical position: booms assumed in the wing plane (z = 0). CAD shows pods/booms below the wing.
#  GAP-12 Placeholder stiffness (EA, GA, edgewise EI) are scaled with the same law so they stay as 'rigid' as in LLTA.
#  GAP-13 Aircraft mass / CG target: scaled LLTA mass + second tail group. Real mass and CG (% MAC) unknown.
# =====================================================================================

# SCALING (Froude / aeroelastic similitude, LLTA -> full scale)
scale = 3.0                         # full scale / LLTA
s_len = scale                       # length
s_mass = scale**3                   # mass (same density, geometric scaling)  -> lumped masses
s_mpl = scale**2                    # mass per unit length
s_force = scale**3                  # force / axial & shear stiffness (EA, GA) -> N
s_stiff = scale**5                  # EI, GJ = force x length^2
s_inertia = scale**5                # lumped rotational inertia = mass x length^2
s_speed = scale**0.5                # velocity (Froude)
# NOTE: sectional rotational inertia per length (j) is NOT scaled with a factor here: it is recomputed from the
# scaled chord, thickness and mass-per-length below (this gives scale**4, as expected).

llta_cruise_speed = 6.5             # m/s, LLTA confirmed cruise speed
cruise_speed = llta_cruise_speed * s_speed   # [GAP-8] Froude-scaled cruise speed (~11.26 m/s)
print("Froude-scaled cruise speed (m/s):", cruise_speed)

# Voltitude full-scale mass target
aircraft_mass_llta = 2.4            # kg (LLTA, used for validation checks only)
aircraft_mass = aircraft_mass_llta * s_mass   # [GAP-13] scaled single-boom mass (target for the mass check; twin tail adds mass on top)

# AERO PANEL GEOMETRY
# main wing
chord_main = s_len * 0.225          # [GAP-10] 3 x LLTA (m)
ea_main = 0.35                      # fraction of chord from LE - non-dimensional, unchanged [GAP-10]
print("Position of elastic axis on main wing is:", (ea_main*chord_main), "m from leading edge. NOTE: ea_main is set as a fraction")
cg_offset_main = s_len * 0.02       # [GAP-5] 3 x LLTA - main wing CG offset (m) aft of elastic axis
print("Position of the main wing CG is:", (ea_main*chord_main + cg_offset_main), "m from leading edge.")

# tail/elevator (EACH of the two tails)
chord_tail = s_len * 0.135          # [GAP-3]
ea_tail = 0.25                      # fraction of chord

# fin/rudder (EACH of the two fins)
chord_fin = s_len * 0.135           # [GAP-3]
ea_fin = 0.25                       # fraction of chord

# BEAM STRUCTURE GEOMETRY
# main wing
span_main = s_len * 3.68            # [GAP-10] 3 x LLTA beam span (m) (LLTA: ASWing file 3.64, drawing 3.75, model 3.68)
lambda_main = 0.5                   # Proportion of the span that is dihedral [GAP-10]
lambda_dihedral = 12.5              # dihedral angle (deg) [GAP-10] angle is scale-invariant
t_main = s_len * 0.02               # wing thickness (assumed, 3 x LLTA)

# twin booms (replace the single fuselage of LLTA)
boom_at_kink = True                 # [GAP-1] booms attach at the dihedral kink: y_boom = (1 - lambda_main) * semi-span
length_boom = s_len * 0.87          # [GAP-2] boom length from wing spar (node) to tail spar (m) = 3 x LLTA tail lever arm
offset_boom_z = 0.0                 # [GAP-11] vertical offset of the boom end from the wing plane (m)
print("Boom length (m):", length_boom)

# tail/elevator (each tail is centred on its boom)
span_tail = s_len * 0.72            # [GAP-3] full span of EACH tailplane (m). LLTA 0.72 (ASWing 9 m/s trim); drawing 0.735
t_tail = s_len * 0.02               # tail thickness (assumed, 3 x LLTA)

# fin/rudder (one on each boom)
fin_height = s_len * 0.36           # [GAP-3] height of EACH fin (m). LLTA 0.36 (ASWing 9 m/s trim); drawing 0.375

# STRUCTURAL PROPERTIES - Froude scaling of the LLTA values  [GAP-4] [GAP-12]
ea = 1e7 * s_force                  # Axial stiffness - SHARPy default scaled (placeholder)
ga = 1e5 * s_force                  # Shear stiffness (placeholder)
gj = 20.87 * s_stiff                # Torsional stiffness - LLTA Voltitude value x 3^5
eiy = 51.25 * s_stiff               # Bending stiffness (flapwise) - LLTA Voltitude value x 3^5
eiz = 2e4 * s_stiff                 # Edgewise bending - SHARPy default (HALE) placeholder, scaled so it stays ~400x eiy as in LLTA

gj_boom = 7.16 * s_stiff            # LLTA Voltitude value x 3^5
eiy_boom = eiz                      # axis changes for the boom (overwritten by eiz_boom in the generate script)
eiz_boom = 71.71 * s_stiff          # LLTA Voltitude value x 3^5
print("Boom stiffness properties:", gj_boom, eiy_boom, eiz_boom)

# MASS PROPERTIES - mass per length scales with scale**2  [GAP-5]
m_bar_main = s_mpl * 2.83 / 9.81    # From ASWing file (N/m converted to kg/m), scaled
j_spanwise_main = m_bar_main * ((chord_main**2) + (t_main**2)) / 12 # Torsion (spanwise axis)
j_chordwise_main = m_bar_main * (t_main**2) / 12 # Chordwise axis
j_normal_main = m_bar_main * (chord_main**2) / 12 # In-plane (normal axis)
print("Updated j values for main wing:", j_spanwise_main, j_chordwise_main, j_normal_main)

m_bar_tail = s_mpl * 0.9 / 9.81     # Tail AND fin (shared mass entry), scaled
j_spanwise_tail = m_bar_tail * ((chord_tail**2) + (t_tail**2)) / 12
j_chordwise_tail = m_bar_tail * (t_tail**2) / 12
j_normal_tail = m_bar_tail * (chord_tail**2) / 12
print("Updated j values for tail:", j_spanwise_tail, j_chordwise_tail, j_normal_tail)

m_bar_boom = s_mpl * 0.686 / 9.81   # LLTA main fuselage (tail boom) mass per length, scaled. Used for BOTH booms
j_bar_boom = scale**4 * 1600 * ((np.pi / 32) * ((0.01414**4)-(0.01176**4))) # LLTA spar-tube J scaled by scale**4 (geometric)

# Check parallel axis theorem
assert np.isclose(j_spanwise_main, j_chordwise_main + j_normal_main), "wing j_x != j_y + j_z"
assert np.isclose(j_spanwise_tail, j_chordwise_tail + j_normal_tail), "tail j_x != j_y + j_z"

# LUMPED MASSES (kg) - all scaled by scale**3 from LLTA  [GAP-6]
mass_prop = s_mass * 0.5 / 9.81
mass_batteries = s_mass * 3.25 / 9.81
mass_pixhawk = s_mass * 0.77 / 9.81
mass_additional = s_mass * 0.55
mass_pod_struct = s_mass * (1.434 / 9.81) * 0.41

mass_launch_handle = s_mass * 0.02  # [GAP-6] LLTA-specific hardware - probably not on the prototype
mass_camera = s_mass * 0.02         # [GAP-6] LLTA-specific hardware - probably not on the prototype
mass_camera_counter = s_mass * 0.02 # [GAP-6] LLTA-specific hardware - probably not on the prototype

# Positions - LOCAL (Material FoR of the node's beam) - scaled by scale
additional_avionics = [0.0, 0.0, 0.0]
payload_bay = [0.0, s_len * 0.238, 0.0]
pod_structure = [0.0, s_len * 0.205, 0.0]
right_handle_pos = [-s_len * 0.023750, 0.0, 0.0]
left_handle_pos = [-s_len * 0.023750, 0.0, 0.0]
camera_pos = [0.0, 0.0, 0.0]
camera_counter_pos = [0.0, 0.0, 0.0]