"""
DOF-RSS mode classifier
--------------------------
Confirmed layout (see verify_eigenvector_layout.py output):
  eigenvectors.dat rows 0-389   = 65 nodes x 6 DOF (trans_x,y,z, rot_x,y,z)
  eigenvectors.dat rows 390-393 = 4 quaternion-redundancy states
  eigenvaluetable.txt modes 0-9  = rigid (6 classical + 4 quaternion-redundant)
  eigenvaluetable.txt modes 10-29 = elastic

DOF-type meaning (per node, LOCAL beam frame):
  index 0: trans_x   index 1: trans_y   index 2: trans_z
  index 3: rot_x  (torsion, spanwise axis)
  index 4: rot_y  (flatwise bending)
  index 5: rot_z  (edgewise bending)

This mirrors the same beam_number/elem_mass cross-referencing already used
for the mass_db component split - see inspect_mass_matrices.py.
"""

import numpy as np
import h5py

FEM_H5_PATH = "/Users/icc25/sharpy_cases/LLTA/llta_natural_modes.fem.h5"
EIGENVECTORS_PATH = "output/llta_natural_modes/beam_modal_analysis/eigenvectors.dat"

N_NODES = 65
N_DOF_PER_NODE = 6
DOF_LABELS = ["trans_x", "trans_y", "trans_z", "rot_x (torsion)", "rot_y (flatwise)", "rot_z (edgewise)"]

RIGID_MODES = set(range(10))
ELASTIC_MODES = list(range(10, 30))
TOP_N = 3  # how many torsion candidates to track per component - see main wing's spread of similar fractions

# Confirmed component -> beam_number mapping (from your earlier fem.h5 inspection)
COMPONENTS = {
    "Main wing": [0, 1],
    "Fuselage": [2],
    "Fin/rudder": [3],
    "Tail": [4, 5],
}


def build_node_to_beam_map(fem_h5_path):
    """
    Maps each of the 65 node indices to the beam_number(s) of the elements
    that use it. Most nodes belong to exactly one beam_number; junction
    nodes (e.g. wing root meeting the fuselage) may belong to more than
    one - these are flagged rather than silently assigned to just one,
    since mis-assigning a junction node is exactly the kind of heuristic
    error already caught once in this project's tooling (the wing-node
    selection bug in llta_sweep_analysis.py).
    """
    with h5py.File(fem_h5_path, "r") as fem:
        connectivities = fem["connectivities"][()]   # (n_elem, n_node_elem)
        beam_number = fem["beam_number"][()]          # (n_elem,)

    node_to_beams = {n: set() for n in range(N_NODES)}
    for elem_idx, node_ids in enumerate(connectivities):
        bn = beam_number[elem_idx]
        for node_id in node_ids:
            node_to_beams[node_id].add(bn)

    junctions = {n: bns for n, bns in node_to_beams.items() if len(bns) > 1}
    if junctions:
        print(f"Note: {len(junctions)} junction node(s) belong to multiple beam_numbers "
              f"(included in every component they touch): {junctions}")

    return node_to_beams


def nodes_for_component(node_to_beams, beam_numbers):
    target = set(beam_numbers)
    return [n for n, bns in node_to_beams.items() if bns & target]


def classify_mode_for_component(phi_reshaped, mode_idx, node_ids):
    """
    phi_reshaped: (N_NODES, N_DOF_PER_NODE, n_modes)
    Returns (dominant_dof_label, energy_fraction, rss_per_dof)
    """
    sub = phi_reshaped[node_ids, :, mode_idx]  # (n_selected_nodes, 6)
    rss = np.sqrt(np.sum(sub**2, axis=0))       # (6,) - RSS per DOF type
    total_energy = np.sum(rss**2)
    if total_energy == 0:
        return None, 0.0, rss
    dominant = np.argmax(rss)
    fraction = rss[dominant]**2 / total_energy
    return DOF_LABELS[dominant], fraction, rss


def get_top_torsion_candidates(phi_reshaped, node_ids, modes, top_n=3):
    """
    Returns the top_n modes (by energy fraction) whose DOMINANT DOF is
    torsion, sorted descending by fraction. Not just the single best - the
    main wing result showed several modes within a few percent of each
    other, and mode-veering across a sweep can flip which one "wins" from
    run to run. Tracking several candidates lets you see that happening
    instead of silently picking a different mode each time.
    """
    candidates = []
    for m in modes:
        dof_label, fraction, _ = classify_mode_for_component(phi_reshaped, m, node_ids)
        if dof_label is not None and "torsion" in dof_label:
            candidates.append((m, fraction))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_n]


def main():
    node_to_beams = build_node_to_beam_map(FEM_H5_PATH)

    phi = np.loadtxt(EIGENVECTORS_PATH)
    elastic_body = phi[:N_NODES * N_DOF_PER_NODE, :]
    phi_reshaped = elastic_body.reshape(N_NODES, N_DOF_PER_NODE, phi.shape[1])

    for label, beam_numbers in COMPONENTS.items():
        node_ids = nodes_for_component(node_to_beams, beam_numbers)
        print(f"\n=== {label} (beam_number(s) {beam_numbers}, {len(node_ids)} nodes) ===")

        top_candidates = get_top_torsion_candidates(phi_reshaped, node_ids, ELASTIC_MODES, top_n=TOP_N)
        top_modes = {m for m, _ in top_candidates}

        for m in ELASTIC_MODES:
            dof_label, fraction, rss = classify_mode_for_component(phi_reshaped, m, node_ids)
            marker = "  <-- top torsion candidate" if m in top_modes else ""
            print(f"  mode {m:>2}: dominant = {dof_label or '(no energy)'}, "
                  f"fraction = {fraction:.1%}{marker}")

        print(f"\n  Top {TOP_N} torsion candidates for {label} (track ALL of these across the j sweep,"
              f" not just the first):")
        if not top_candidates:
            print("    None found - no mode has torsion as its dominant DOF for this component.")
        else:
            for rank, (m, fraction) in enumerate(top_candidates, start=1):
                print(f"    #{rank}: mode {m} (energy fraction {fraction:.1%})")
            if len(top_candidates) > 1 and (top_candidates[0][1] - top_candidates[1][1]) < 0.05:
                print("    Note: top two candidates are within 5 percentage points of each other -"
                      " a close call worth treating as genuinely ambiguous, not resolved by rank alone.")


if __name__ == "__main__":
    main()