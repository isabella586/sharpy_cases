"""
Stage B: analyse the newmark_damp sweep
-------------------------------------------
For each run, loads aeroelastic_eigenvalues.dat (columns: real, imag) and
checks three things against newmark_damp:

  1. Does the size of the 55-80Hz artifact cluster shrink as damping increases?
  2. Does the worst (most positive) real part IN that band trend toward stable
     (negative) as damping increases?
  3. Do the physically meaningful modes BELOW 55Hz (rigid-body + the two real
     modes + the two low-frequency conjugate pairs) stay roughly fixed across
     the sweep? Newmark damping is supposed to be frequency-selective -
     negligible at low frequency - so these should barely move. If they DO
     move noticeably, that's a red flag this setting affects more than just
     the suspected numerical band.

ASSUMPTION worth confirming: each run's stability output sits at
output/<case_name>/stability/aeroelastic_eigenvalues.dat, mirroring the
Modal step's output/<case_name>/beam_modal_analysis/ layout. Adjust
STABILITY_SUBDIR below if that's not where it actually lands.
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_ROOT = '/Users/icc25/sharpy_cases/LLTA/output'
FOLDER_PATTERN = re.compile(r'^llta_newmarksweep_(?P<damp>[\d.eE+-]+)$')
STABILITY_SUBDIR = os.path.join('stability', 'aeroelastic_eigenvalues.dat')

BAND_LO_HZ = 55.0
BAND_HI_HZ = 80.5
PIN_FREQ_HZ = 80.0
PIN_TOL_HZ = 0.001


def load_run(run_dir):
    path = os.path.join(run_dir, STABILITY_SUBDIR)
    if not os.path.exists(path):
        return None
    a = np.loadtxt(path)
    real, imag = a[:, 0], a[:, 1]
    freq_hz = np.abs(imag) / (2 * np.pi)
    return real, imag, freq_hz


def main():
    if not os.path.isdir(OUTPUT_ROOT):
        raise SystemExit(f'OUTPUT_ROOT does not exist: {OUTPUT_ROOT}')

    entries = sorted(os.listdir(OUTPUT_ROOT))
    band_rows = []
    low_freq_rows = []

    for name in entries:
        full = os.path.join(OUTPUT_ROOT, name)
        if not os.path.isdir(full):
            continue
        m = FOLDER_PATTERN.match(name)
        if not m:
            continue
        damp = float(m.group('damp'))

        loaded = load_run(full)
        if loaded is None:
            print(f'{name}: no aeroelastic_eigenvalues.dat found at expected path - skipping')
            continue
        real, imag, freq_hz = loaded

        # --- diagnostic: dump the lowest-frequency modes for the two runs
        # that showed zero rows below 55 Hz, to see whether the rigid-body
        # modes genuinely shifted or just aren't landing at exactly 0.0 -----
        if damp in (0.0, 1e-4):
            order = np.argsort(freq_hz)
            print(f"\n--- lowest-frequency modes for newmark_damp={damp:g} ({name}) ---")
            print(f"{'real':>12} {'imag':>12} {'freq_hz':>12}")
            for i in order[:20]:
                print(f"{real[i]:>12.6f} {imag[i]:>12.6f} {freq_hz[i]:>12.6f}")
            print()

        band_mask = (freq_hz >= BAND_LO_HZ) & (freq_hz <= BAND_HI_HZ)
        pinned_mask = np.isclose(freq_hz, PIN_FREQ_HZ, atol=PIN_TOL_HZ)

        band_rows.append({
            'newmark_damp': damp,
            'n_modes_in_band': int(np.sum(band_mask)),
            'n_pinned_at_80hz': int(np.sum(pinned_mask)),
            'n_unstable_in_band': int(np.sum(band_mask & (real > 1e-9))),
            'max_real_in_band': float(np.max(real[band_mask])) if np.any(band_mask) else np.nan,
            'n_unstable_outside_band': int(np.sum((~band_mask) & (real > 1e-9))),
        })

        # Track physically meaningful modes by FREQUENCY, not row index -
        # mode ordering isn't guaranteed identical across different
        # newmark_damp values even for the same underlying physical system.
        for i in np.where(~band_mask)[0]:
            low_freq_rows.append({'newmark_damp': damp, 'freq_hz': freq_hz[i], 'real': real[i]})

    if not band_rows:
        raise SystemExit('No runs found - check OUTPUT_ROOT, FOLDER_PATTERN, and STABILITY_SUBDIR.')

    df = pd.DataFrame(band_rows).sort_values('newmark_damp').reset_index(drop=True)
    df.to_csv('newmark_sweep_band_summary.csv', index=False)
    print(df.to_string(index=False))

    low_df = pd.DataFrame(low_freq_rows).sort_values(['newmark_damp', 'freq_hz']).reset_index(drop=True)
    low_df.to_csv('newmark_sweep_low_freq_modes.csv', index=False)

    # --- Plot 1: artifact band size / instability count vs damping ---
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(df['newmark_damp'], df['n_pinned_at_80hz'], 'o-', label='modes pinned at 80.000 Hz')
    ax1.plot(df['newmark_damp'], df['n_unstable_in_band'], 's-', label='unstable modes in 55-80Hz band')
    ax1.set_xscale('symlog', linthresh=1e-4)
    ax1.set_xlabel('newmark_damp')
    ax1.set_ylabel('count')
    ax1.legend()
    ax1.set_title('Artifact band size vs newmark_damp')
    fig1.tight_layout()
    fig1.savefig('newmark_sweep_band_counts.png', dpi=150)

    # --- Plot 2: worst instability in the band vs damping ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(df['newmark_damp'], df['max_real_in_band'], 'o-', color='crimson')
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_xscale('symlog', linthresh=1e-4)
    ax2.set_xlabel('newmark_damp')
    ax2.set_ylabel('max real part in 55-80Hz band')
    ax2.set_title('Worst-case instability in artifact band vs newmark_damp')
    fig2.tight_layout()
    fig2.savefig('newmark_sweep_max_real.png', dpi=150)

    # --- Plot 3: do the physically meaningful low-frequency modes stay put? ---
    # Track each mode as an actual trajectory across the sweep, rather than a
    # single undifferentiated scatter cloud - anchor on the reference run
    # closest to the original 5e-3 baseline, then for every other run match
    # each anchor to its nearest-frequency counterpart. This assumes mode
    # identity varies smoothly with newmark_damp (a reasonable assumption for
    # a numerical-damping sweep, unlike a genuine structural property sweep)
    # - worth a visual sanity check on the resulting lines before trusting them.
    ref_damp = min(df['newmark_damp'], key=lambda d: abs(d - 5e-3))
    ref_freqs = sorted(low_df[low_df['newmark_damp'] == ref_damp]['freq_hz'].to_numpy())

    fig3, ax3 = plt.subplots(figsize=(8, 5.5))
    damp_values = sorted(df['newmark_damp'].unique())

    for anchor_freq in ref_freqs:
        traj_damp, traj_real = [], []
        for d in damp_values:
            run_modes = low_df[low_df['newmark_damp'] == d]
            if run_modes.empty:
                continue
            idx = (run_modes['freq_hz'] - anchor_freq).abs().idxmin()
            traj_damp.append(d)
            traj_real.append(run_modes.loc[idx, 'real'])
        ax3.plot(traj_damp, traj_real, 'o-', label=f'~{anchor_freq:.3f} Hz')

    ax3.set_xscale('symlog', linthresh=1e-4)
    ax3.set_xlabel('newmark_damp')
    ax3.set_ylabel('real part')
    ax3.axhline(0, color='black', lw=0.8)
    ax3.legend(fontsize=8, title='mode (by reference frequency)')
    ax3.set_title(f'Low-frequency mode trajectories vs newmark_damp\n(anchored on the {ref_damp:g} run)')
    fig3.tight_layout()
    fig3.savefig('newmark_sweep_lowfreq_stability.png', dpi=150)

    print('\nSaved: newmark_sweep_band_counts.png, newmark_sweep_max_real.png, newmark_sweep_lowfreq_stability.png')


if __name__ == '__main__':
    main()