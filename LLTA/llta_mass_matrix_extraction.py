"""
LLTA mass-matrix extraction: fem.h5 vs SaveData .h5
----------------------------------------------------
Section 9.1 (stability setup notes) needs the raw, per-material 6x6 mass
blocks ("mass_db"), not an assembled global matrix. Both mass_db and the
elem_mass/beam_number arrays that let you attribute a block to a specific
component (fuselage / tail / main wing) live in the original fem.h5 input
file, and should be mirrored in the SaveData output.

STAGE 1 confirms this by printing both files' structure and comparing the
raw arrays directly. STAGE 2 uses those (now-confirmed) arrays to build a
per-component summary: mu, j, and the ccg off-diagonal term, ready to sit
alongside your hand-calculated values.

Run Stage 1 first. Fill in SAVEDATA_STRUCTURE_PATH once you've seen the
actual tree printed — don't assume the guess below is right without
checking it against the printed output.
"""

import h5py
import numpy as np

FEM_H5_PATH = "/Users/icc25/sharpy_cases/LLTA/llta_asymptotic_stability.fem.h5"
SAVEDATA_H5_PATH = "/Users/icc25/sharpy_cases/LLTA/output/llta_asymptotic_stability/savedata/llta_asymptotic_stability.data.h5"

# Best-guess path to the mirrored structural data inside the SaveData file.
# CONFIRM against the Stage 1 printout below before trusting this.
SAVEDATA_STRUCTURE_PATH = "data/structure"

# Fill these in once you've identified them from elem_mass / connectivities
# (see Stage 2) — these are NOT guessed, they must come from your own case
# script or a direct look at beam_number.
# A component can be a single beam_number (fuselage) or a list of them
# (main wing / tail, split left/right) — both forms are handled below.
BEAM_NUMBER_FUSELAGE = 2
BEAM_NUMBER_FIN = 3  # single beam, not left/right split (vertical, unmirrored)
BEAM_NUMBER_MAIN_WING = {"right": 0, "left": 1}
BEAM_NUMBER_TAIL = {"right": 4, "left": 5}


# --------------------------------------------------------------------------
# STAGE 1: inspect both files' structure, and cross-check the raw arrays
# --------------------------------------------------------------------------

def print_h5_tree(path):
    print(f"\n--- structure of {path} ---")
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            kind = "GROUP" if isinstance(obj, h5py.Group) else f"DATASET {obj.shape} {obj.dtype}"
            #print(f"  {name}  [{kind}]")
        f.visititems(visitor)


def compare_fem_and_savedata():
    with h5py.File(FEM_H5_PATH, "r") as fem:
        fem_mass_db = fem["mass_db"][()]
        fem_elem_mass = fem["elem_mass"][()]
        fem_beam_number = fem["beam_number"][()]

    with h5py.File(SAVEDATA_H5_PATH, "r") as sd:
        try:
            sd_mass_db = sd[f"{SAVEDATA_STRUCTURE_PATH}/mass_db"][()]
        except KeyError:
            print(
                f"\nCould not find '{SAVEDATA_STRUCTURE_PATH}/mass_db' in the "
                f"SaveData file. Check the tree printed above and update "
                f"SAVEDATA_STRUCTURE_PATH."
            )
            return

    print(f"\nfem.h5 mass_db shape:      {fem_mass_db.shape}")
    print(f"SaveData mass_db shape:    {sd_mass_db.shape}")

    if fem_mass_db.shape == sd_mass_db.shape and np.allclose(fem_mass_db, sd_mass_db):
        print("MATCH: fem.h5 and SaveData mass_db are identical.")
    else:
        print("MISMATCH: the two files do not agree — worth understanding why "
              "before trusting either as a baseline.")

    return fem_mass_db, fem_elem_mass, fem_beam_number


# --------------------------------------------------------------------------
# STAGE 2: per-component summary (mu, j, ccg) once the arrays are confirmed
# --------------------------------------------------------------------------

def _report_one_beam(label, bn, mass_db, elem_mass, beam_number):
    """Print the mass_db entries used by a single beam_number, return them
    keyed by mass_db index so callers can compare across beams."""
    if bn is None:
        print(f"    {label}: beam_number not set — fill this in.")
        return {}

    elem_ids = np.where(beam_number == bn)[0]
    mass_ids = np.unique(elem_mass[elem_ids])

    print(f"    {label} (beam_number={bn}, {len(elem_ids)} elements, "
          f"{len(mass_ids)} distinct mass_db entr{'y' if len(mass_ids)==1 else 'ies'})")

    blocks = {}
    for mid in mass_ids:
        block = mass_db[mid]
        blocks[mid] = block
        mu = np.diag(block[0:3, 0:3])          # translational mass / length
        j = np.diag(block[3:6, 3:6])           # rotational inertia / length
        ccg = block[0:3, 3:6]                  # CG-offset coupling block
        print(f"      mass_db[{mid}]:")
        print(f"        mu (translational diag)     = {mu}")
        print(f"        j  (rotational diag)         = {j}")
        print(f"        ccg (translation-rotation)   = \n{ccg}")
    return blocks


def summarise_by_component(mass_db, elem_mass, beam_number):
    """
    mass_db:      (n_mass_entries, 6, 6) — one raw 6x6 block per material ID
    elem_mass:    (n_elem,) — which mass_db entry each element uses
    beam_number:  (n_elem,) — which structural member each element belongs to

    A component given as a single int (fuselage) is reported directly.
    A component given as a {"left": .., "right": ..} dict (main wing, tail)
    is reported per side, then cross-checked for left/right symmetry —
    a free consistency check independent of the hand-calculated comparison.
    """
    components = {
        "Fuselage": BEAM_NUMBER_FUSELAGE,
        "Fin/rudder": BEAM_NUMBER_FIN,
        "Main wing": BEAM_NUMBER_MAIN_WING,
        "Tail": BEAM_NUMBER_TAIL,
    }

    for label, bn in components.items():
        print(f"\n--- {label} ---")

        if isinstance(bn, dict):
            side_blocks = {
                side: _report_one_beam(side, side_bn, mass_db, elem_mass, beam_number)
                for side, side_bn in bn.items()
            }
            sides = list(side_blocks)
            if len(sides) == 2 and all(side_blocks.values()):
                left_vals = list(side_blocks[sides[0]].values())
                right_vals = list(side_blocks[sides[1]].values())
                if len(left_vals) == len(right_vals) and all(
                    np.allclose(l, r) for l, r in zip(left_vals, right_vals)
                ):
                    print(f"    Left/right symmetry check: MATCH — {sides[0]} and "
                          f"{sides[1]} mass_db blocks are identical.")
                else:
                    print(f"    Left/right symmetry check: MISMATCH — worth "
                          f"understanding why before treating either side as "
                          f"the baseline for the hand-calc comparison.")
        else:
            _report_one_beam(label, bn, mass_db, elem_mass, beam_number)


if __name__ == "__main__":
    print_h5_tree(FEM_H5_PATH)
    print_h5_tree(SAVEDATA_H5_PATH)

    result = compare_fem_and_savedata()
    if result is not None:
        mass_db, elem_mass, beam_number = result
        print(f"\nUnique beam_number values found: {np.unique(beam_number)}")
        summarise_by_component(mass_db, elem_mass, beam_number)