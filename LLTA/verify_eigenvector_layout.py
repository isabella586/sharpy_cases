"""
Verify eigenvectors.dat row layout
------------------------------------
Working hypothesis: 394 rows = 65 nodes x 6 DOF (3 translation + 3 rotation,
in that per-node order) + 4 trailing quaternion orientation states.

Two checks below, using only what the eigenvaluetable.txt already told us
(modes 0-9 are rigid, modes 10-29 are elastic, no further assumptions):

  1. The last 4 rows (indices 390-393, the presumed quaternion states)
     should be ~zero for every ELASTIC mode (10-29) - a quaternion
     perturbation has no reason to activate under pure elastic deformation
     with the rigid-body modes decoupled - and can be nonzero for at least
     some of the RIGID modes (0-9), since that's exactly what they'd
     represent physically.

  2. Reshaping the first 390 rows as (65 nodes, 6 DOF) per mode should show
     a physically sensible pattern for at least the clearest rigid modes:
     e.g. one mode should show near-identical translational values across
     all 65 nodes in a single direction, with ~zero elsewhere - a pure
     rigid-body translation looks exactly like that when the layout
     assumption is right, and like noise/nonsense if it's wrong.

Neither check *proves* the layout - but a fail on either is a clear signal
the 65x6+4 assumption needs revisiting before anything is built on it.
"""

import numpy as np

EIGENVECTORS_PATH = 'output/llta_asymptotic_stability/beam_modal_analysis/eigenvectors.dat'
N_NODES = 65
N_DOF_PER_NODE = 6
N_QUAT = 4
RIGID_MODES = list(range(10))     # indices 0-9, per eigenvaluetable.txt
ELASTIC_MODES = list(range(10, 30))  # indices 10-29

phi = np.loadtxt(EIGENVECTORS_PATH)
print(f"eigenvectors.dat shape: {phi.shape}")
assert phi.shape[0] == N_NODES * N_DOF_PER_NODE + N_QUAT, "Row count doesn't match the 65x6+4 hypothesis - stop here."

quat_rows = phi[N_NODES * N_DOF_PER_NODE:, :]  # last 4 rows, all modes

print("\n--- Check 1: quaternion rows (390-393) by mode ---")
print(f"{'mode':>4}  {'rigid?':>7}  {'max |value| in quat rows':>25}")
for m in range(phi.shape[1]):
    is_rigid = m in RIGID_MODES
    max_val = np.max(np.abs(quat_rows[:, m]))
    print(f"{m:>4}  {'yes' if is_rigid else 'no':>7}  {max_val:>25.6e}")

print("\n--- Check 2: per-node DOF pattern for the clearest rigid modes ---")
elastic_body = phi[:N_NODES * N_DOF_PER_NODE, :]
reshaped = elastic_body.reshape(N_NODES, N_DOF_PER_NODE, phi.shape[1])  # (node, dof, mode)

dof_labels = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
for m in RIGID_MODES[:6]:  # first 6 - the classical (non-quaternion-redundant) rigid modes
    print(f"\nMode {m}:")
    for d, label in enumerate(dof_labels):
        col = reshaped[:, d, m]
        print(f"  {label:>8}: mean={col.mean(): .4e}  std={col.std(): .4e}  "
              f"max|.|={np.max(np.abs(col)): .4e}")