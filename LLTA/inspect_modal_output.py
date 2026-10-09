"""
Stage 1: inspect Modal solver output format (beam_modal_analysis/)
--------------------------------------------------------------------
Before the DOF-RSS torsion-mode classifier can be written, we need to see:
  - how many modes are in each file, and in what order
  - whether eigenvectors.dat is (n_dof x n_modes) or (n_modes x n_dof)
  - what n_dof actually corresponds to - does it include rigid-body DOFs?
    are clamped/constrained DOFs already removed?
  - whether eigenvaluetable.txt / frequencies.dat give a direct
    mode-index -> frequency mapping to sanity-check eigenvectors.dat against

Nothing below is guessed - it only prints what's actually in the files.
Point MODAL_DIR at ONE run's actual output folder (e.g. your current
baseline case, before running the full j sweep) and run this first.
"""

import numpy as np
import os

MODAL_DIR = '/Users/icc25/sharpy_cases/LLTA/output/llta_natural_modes/beam_modal_analysis'


def peek_text_file(path, n_lines=15):
    print(f"\n--- {path} ---")
    if not os.path.exists(path):
        print("  (not found)")
        return
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                print("  ...")
                break
            print(f"  {line.rstrip()}")


def load_possibly_complex(path):
    """
    np.loadtxt's default float parser can't handle values written as Python's
    own complex repr, e.g. '(0.000000e+00+0.000000e+00j)'. Try plain floats
    first (some .dat files here may genuinely just be real numbers); if that
    fails, fall back to Python's own complex() parser token-by-token, which
    does understand the parenthesised form.
    """
    try:
        return np.loadtxt(path)
    except ValueError:
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append([complex(tok) for tok in line.split()])
        return np.array(rows)


def peek_array_file(path):
    print(f"\n--- {path} ---")
    if not os.path.exists(path):
        print("  (not found)")
        return
    arr = load_possibly_complex(path)
    print(f"  dtype: {arr.dtype}")
    print(f"  shape: {arr.shape}")
    if arr.ndim > 1:
        print(f"  first row:    {arr[0]}")
        print(f"  first column: {arr[:, 0]}")
    else:
        print(f"  first values: {arr[:10]}")


if __name__ == "__main__":
    peek_text_file(os.path.join(MODAL_DIR, 'eigenvaluetable.txt'))
    peek_array_file(os.path.join(MODAL_DIR, 'frequencies.dat'))
    peek_array_file(os.path.join(MODAL_DIR, 'eigenvalues.dat'))
    peek_array_file(os.path.join(MODAL_DIR, 'eigenvectors.dat'))