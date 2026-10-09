# Output fem.h5 bits
"""
import h5py
import numpy as np
with h5py.File('llta_prescribed_amp_00_period_3.fem.h5', 'r') as f:
    print('=== FEM FILE CONTENTS ===')
    print('num_node:', np.array(f['num_node']))
    print('num_elem:', np.array(f['num_elem']))
    print('stiffness_db:')
    print(np.array(f['stiffness_db']))
    print('mass_db:')
    print(np.array(f['mass_db']))
    print('boundary_conditions:', np.array(f['boundary_conditions']))
    print('coordinates (first 10):', np.array(f['coordinates'])[:10])
    print('coordinates (last 5):', np.array(f['coordinates'])[-5:])
    print('connectivities (first 5):', np.array(f['connectivities'])[:5])
    print('connectivities (last 5):', np.array(f['connectivities'])[-5:])
    print('app_forces (nonzero):', np.argwhere(np.array(f['app_forces']) != 0))
    print('lumped_mass:', np.array(f['lumped_mass']))
    print('lumped_mass_nodes:', np.array(f['lumped_mass_nodes']))
    print('frame_of_reference_delta (elem 0):', np.array(f['frame_of_reference_delta'])[0])
    print('frame_of_reference_delta (elem 24):', np.array(f['frame_of_reference_delta'])[24])
    print('frame_of_reference_delta (elem 30):', np.array(f['frame_of_reference_delta'])[30])
"""
"""
# Rank of stiffness matrix - identify singularity

stiffness_db = [np.array(h5py.File('llta.fem.h5', 'r')['stiffness_db'])]

# stiffness_db is a list of 6x6 matrices, one per property group
for i, K_group in enumerate(stiffness_db):
    K = np.array(K_group)
    # K is shape (3, 6, 6) - take the first matrix, they should all be identical
    K_single = K[0]
    print(f'\nStiffness group {i}:')
    print(K_single)
    print(f'Condition number: {np.linalg.cond(K_single):.3e}')
    print(f'Min diagonal entry: {np.min(np.abs(np.diag(K_single))):.3e}')
    print(f'Max diagonal entry: {np.max(np.abs(np.diag(K_single))):.3e}')
    print(f'Diagonal ratio: {np.max(np.abs(np.diag(K_single))) / np.min(np.abs(np.diag(K_single))):.3e}')
"""

"""
for i, M_group in enumerate(mass_db):
    M = np.array(M_group)
    M_single = M[0]
    # Estimate highest natural frequency per DOF
    K_single = np.array(stiffness_db[min(i, len(stiffness_db)-1)])[0]
    omega_approx = np.sqrt(np.abs(np.diag(K_single)) / np.abs(np.diag(M_single)))
    print(f'\nMass group {i} - approx natural frequencies per DOF (rad/s):')
    print(omega_approx)
    print(f'Highest: {np.max(omega_approx):.3e} rad/s')
    print(f'Lowest:  {np.min(omega_approx):.3e} rad/s')
    print(f'Frequency ratio: {np.max(omega_approx)/np.min(omega_approx):.3e}')
"""

"""
# Assess global assembly has any gaps, disconnected nodes, or incorrectly assigned root constraints that would cause a singular or near-singular global stiffness matrix despite the individual element matrices looking correct
print("=== BOUNDARY CONDITIONS ANALYSIS ===")
bc = np.array(h5py.File('llta.fem.h5', 'r')['boundary_conditions'])
print(f"Total nodes: {len(bc)}")
print(f"Root nodes (bc=1):  {np.where(bc == 1)[0]}")
print(f"Tip nodes  (bc=-1): {np.where(bc == -1)[0]}")
print(f"Interior nodes (bc=0): count = {np.sum(bc == 0)}")

print("\n=== BEAM CONNECTIVITY ANALYSIS ===")
conn = np.array(h5py.File('llta.fem.h5', 'r')['connectivities'])
print(f"Total elements: {len(conn)}")
print(f"All node indices used: {np.unique(conn)}")
print(f"Max node index in connectivity: {np.max(conn)}")
print(f"Total nodes declared: {len(bc)}")

# Check for any node that appears only once (potential disconnected node)
from collections import Counter
all_nodes = conn.flatten()
node_counts = Counter(all_nodes)
orphan_nodes = [n for n, c in node_counts.items() if c == 1]
print(f"\nNodes appearing only once in connectivity (potential tips/roots): {orphan_nodes}")

print("\n=== MEMBER IDENTIFICATION ===")
# Find where beam members start and end by tracking connectivity breaks
elem_start_nodes = conn[:, 0]
elem_end_nodes = conn[:, 1]
print("First node of each element:", elem_start_nodes[:10], "...")
print("Last node of each element:", elem_end_nodes[:10], "...")

# Beam Member Structure

print("=== MEMBER TRACING ===")
np.array(h5py.File('llta.fem.h5', 'r')['connectivities'])

# Trace which nodes each member visits by following connectivity
# Build adjacency from start/end nodes only (ignore midpoints)
from collections import defaultdict

adjacency = defaultdict(set)
for elem in conn:
    start, end, mid = elem
    adjacency[start].add(end)
    adjacency[end].add(start)

# Trace connected components
visited = set()
components = []

def trace(node, component):
    stack = [node]
    while stack:
        n = stack.pop()
        if n not in visited:
            visited.add(n)
            component.append(n)
            for neighbor in adjacency[n]:
                if neighbor not in visited:
                    stack.append(neighbor)

for node in range(len(np.array(h5py.File('llta.fem.h5', 'r')['boundary_conditions']))):
    if node not in visited:
        component = []
        trace(node, component)
        components.append(sorted(component))

print(f"Number of disconnected components: {len(components)}")
for i, comp in enumerate(components):
    bc_vals = np.array(h5py.File('llta.fem.h5', 'r')['boundary_conditions'])[comp]
    print(f"\nComponent {i}: {len(comp)} nodes, "
          f"root nodes: {[n for n in comp if np.array(h5py.File('llta.fem.h5', 'r')['boundary_conditions'])[n] == 1]}, "
          f"tip nodes: {[n for n in comp if np.array(h5py.File('llta.fem.h5', 'r')['boundary_conditions'])[n] == -1]}, "
          f"node range: {min(comp)}-{max(comp)}")
    

with h5py.File('llta.fem.h5', 'r') as f:
    print("=== ALL DATASETS IN FEM FILE ===")
    for key in f.keys():
        data = f[key][()]
        print(f"\n{key}: shape={np.array(data).shape}, dtype={np.array(data).dtype}")
        if np.array(data).size <= 20:
            print(data)
        else:
            print("First few values:", np.array(data).flat[:10])
"""

"""import h5py
with h5py.File('llta_prescribed_amp_00_period_3.fem.h5', 'r') as f:
    mass = f['mass_db'][:]
    print("Full mass matrix:", mass)
    print('Mass matrix diagonal entries (m, m, m, Ixx, Iyy, Izz):')
    for i, m in enumerate(mass):
        print(f'  Section {i}: {m[0,0]:.4f} kg/m')
"""

import h5py
with h5py.File('llta.fem.h5', 'r') as f:
    fdr = f['frame_of_reference_delta'][:]
    print('Full frame_of_reference_delta array:')
    for i, v in enumerate(fdr):
        print(f'  Element {i:2d}: {v}')