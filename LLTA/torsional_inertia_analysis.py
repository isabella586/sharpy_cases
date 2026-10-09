"""
Stage B: analyse the j_bar_main sweep
----------------------------------------
Mirrors the trim-sweep analysis pattern (OUTPUT_ROOT + FOLDER_PATTERN +
per-run parsing -> rows -> DataFrame -> CSV -> plot).

Key design choice, following the mode-veering discussion: for each run,
we record ALL top-N torsion-flavoured candidates (not just the single
best), and plot them as an unconnected scatter, sized by confidence
(energy fraction). A genuine torsion mode should trace a smooth,
monotonically-decreasing curve as j increases (frequency ~ 1/sqrt(j) for
a simple torsional oscillator); scattered points that don't follow that
trend are very likely other modes that transiently cross the classifier's
threshold, not a real physical signal.

ASSUMPTION, worth checking rather than trusting blindly: the node/element
connectivity (and therefore the node_to_beam_number map) is assumed
IDENTICAL across every run in the sweep, since only j_bar_main changes,
not geometry. This is why the map is built ONCE from one representative
fem.h5 rather than per-run. If the sweep ever also varies something
geometric, this assumption breaks and the map needs rebuilding per run.
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mode_classification import (
    build_node_to_beam_map,
    nodes_for_component,
    get_top_torsion_candidates,
    N_NODES,
    N_DOF_PER_NODE,
    ELASTIC_MODES,
    COMPONENTS,
)

OUTPUT_ROOT = '/Users/icc25/sharpy_cases/LLTA/output'
FOLDER_PATTERN = re.compile(r'^llta_jsweep_(?P<jval>[\d.eE+-]+)$')

# Any one sweep run's fem.h5 - connectivity is assumed identical across
# the whole sweep (see module docstring).
REFERENCE_FEM_H5 = '/Users/icc25/sharpy_cases/LLTA/llta_jsweep_1.026e-03.fem.h5'

TARGET_COMPONENT = "Main wing"   # which COMPONENTS entry to analyse - change as needed
TOP_N = 5                        # wider than classify_modes' default, given how smeared
                                  # the main wing's torsion signature already looked

OUT_CSV = 'jbar_sweep_results.csv'


def parse_eigenvaluetable(path):
    """
    Parses the pipe-delimited eigenvaluetable.txt into {mode_idx: freq_n_hz}.
    Skips header/separator lines; tolerant of the double separator line
    that appeared where the table print ran twice back-to-back in your
    earlier example output.
    """
    freq_by_mode = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith('|'):
                continue
            fields = [c.strip() for c in line.strip('|').split('|')]
            if len(fields) < 4:
                continue
            mode_str, _real, _imag, freq_n_str = fields[0], fields[1], fields[2], fields[3]
            try:
                mode_idx = int(mode_str)
                freq_n = float(freq_n_str)
            except ValueError:
                continue  # header row or separator - not a data row
            freq_by_mode[mode_idx] = freq_n
    return freq_by_mode


def find_file(run_dir, *parts):
    path = os.path.join(run_dir, *parts)
    return path if os.path.exists(path) else None


def main():
    if not os.path.isdir(OUTPUT_ROOT):
        raise SystemExit(f"OUTPUT_ROOT does not exist: {OUTPUT_ROOT}")

    node_to_beams = build_node_to_beam_map(REFERENCE_FEM_H5)
    node_ids = nodes_for_component(node_to_beams, COMPONENTS[TARGET_COMPONENT])

    entries = sorted(os.listdir(OUTPUT_ROOT))
    rows = []
    skipped = []

    for name in entries:
        full = os.path.join(OUTPUT_ROOT, name)
        if not os.path.isdir(full):
            continue
        m = FOLDER_PATTERN.match(name)
        if not m:
            skipped.append(name)
            continue

        j_val = float(m.group("jval"))
        run_id = name

        table_path = find_file(full, "beam_modal_analysis", "eigenvaluetable.txt")
        vec_path = find_file(full, "beam_modal_analysis", "eigenvectors.dat")
        if table_path is None or vec_path is None:
            warnings.warn(f"{run_id}: missing eigenvaluetable.txt or eigenvectors.dat - skipping")
            continue

        freq_by_mode = parse_eigenvaluetable(table_path)

        phi = np.loadtxt(vec_path)
        elastic_body = phi[:N_NODES * N_DOF_PER_NODE, :]
        phi_reshaped = elastic_body.reshape(N_NODES, N_DOF_PER_NODE, phi.shape[1])

        candidates = get_top_torsion_candidates(phi_reshaped, node_ids, ELASTIC_MODES, top_n=TOP_N)

        if not candidates:
            warnings.warn(f"{run_id}: no torsion-dominant mode found for {TARGET_COMPONENT}")
            continue

        for rank, (mode_idx, fraction) in enumerate(candidates, start=1):
            freq_hz = freq_by_mode.get(mode_idx, np.nan)
            rows.append({
                "run_id": run_id,
                "j_bar_main": j_val,
                "rank": rank,
                "mode_idx": mode_idx,
                "fraction": fraction,
                "freq_hz": freq_hz,
            })

    print(f"Found {sum(1 for n in entries if FOLDER_PATTERN.match(n))} matching run folders, "
          f"skipped {len(skipped)} non-matching:")
    for s in skipped:
        print(f"  skipped: {s}")

    if not rows:
        raise SystemExit("No runs successfully parsed - check OUTPUT_ROOT, FOLDER_PATTERN, and REFERENCE_FEM_H5.")

    df = pd.DataFrame(rows).sort_values(["j_bar_main", "rank"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} candidate rows ({df['j_bar_main'].nunique()} runs) to {OUT_CSV}")
    print(df.to_string(index=False))

    # --- plot: unconnected scatter, point size/opacity by confidence -----
    plt.figure(figsize=(8, 5.5))
    sizes = 20 + 180 * df["fraction"]
    sc = plt.scatter(df["j_bar_main"], df["freq_hz"], s=sizes, c=df["fraction"],
                      cmap="viridis", alpha=0.75, edgecolors="none")
    plt.xscale("log")
    plt.xlabel("j_bar_main (kg·m)")
    plt.ylabel("Candidate frequency (Hz)")
    plt.title(f"{TARGET_COMPONENT}: torsion-flavoured candidates across the j sweep\n"
              f"(point size/color = energy fraction - not connected by mode index,"
              f" since mode index identity isn't stable across the sweep)")
    cbar = plt.colorbar(sc)
    cbar.set_label("Energy fraction (dominant DOF = torsion)")
    plt.tight_layout()
    plt.savefig("jbar_sweep_torsion_candidates.png", dpi=150)
    print("\nSaved plot to jbar_sweep_torsion_candidates.png")
    print("\nLook for: a smooth, monotonically-decreasing band of points as j_bar_main "
          "increases - that band is your genuine torsion mode. Scattered points well off "
          "that band are very likely other modes crossing the classifier's threshold "
          "transiently, not real torsion behaviour.")


if __name__ == "__main__":
    main()