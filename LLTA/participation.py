"""
Participation factors and mode table for SHARPy eigen-output.

Usage:
    python participation.py EIGENVALUES EIGENVECTORS [--fmax 5] [--out modes_9ms]

Inputs (any of): .dat/.txt (np.loadtxt), .npy, .npz (first array), .h5 (use --key).
  EIGENVALUES : N complex values, or an N x 2 array [real, imag]
  EIGENVECTORS: N x N complex (columns = right eigenvectors), or N x 2N [real | imag]
Check the printed shapes first; if they are not what you expect, edit load_vectors().

Method
  V is (n_states x n_modes), columns = right eigenvectors. Left eigenvectors are not
  available (SHARPy keeps only a subset of modes), so participation of state k in mode i is
      p_ki = |V[k,i]|^2 / sum_k |V[k,i]|^2
  over the chosen states (use --aero to drop the aero states). Compare states of similar
  units only, because the right-eigenvector form depends on state scaling.
"""
import argparse
import numpy as np
import scipy.linalg as sla

# --- EDIT ME: map state NAMES to row indices of your state vector -----------------
# Leave empty to just print the top state indices per mode. Example:
# STATE_GROUPS = {"u": [0], "w": [1], "q": [2], "theta": [3],
#                 "v": [4], "p": [5], "r": [6], "phi": [7], "psi": [8],
#                 "flex": list(range(9, 400))}
STATE_GROUPS = {}

# Which groups define each classical mode (used only if STATE_GROUPS is filled in)
MODE_SIGNATURES = {
    "short period": ["w", "q", "theta"],
    "phugoid": ["u", "theta"],
    "dutch roll": ["v", "p", "r"],
    "roll subsidence": ["p", "phi"],
    "spiral": ["phi", "psi", "r"],
}


def load_array(path, key=None):
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        z = np.load(path)
        return z[key or z.files[0]]
    if path.endswith((".h5", ".hdf5")):
        import h5py
        with h5py.File(path, "r") as f:
            if key is None:
                raise SystemExit("h5 keys: %s  -> pass --key" % list(f.keys()))
            return f[key][()]
    return np.loadtxt(path)


def load_values(path, key=None):
    a = load_array(path, key)
    if np.iscomplexobj(a):
        return a.ravel()
    if a.ndim == 2 and a.shape[1] == 2:
        return a[:, 0] + 1j * a[:, 1]
    return a.ravel().astype(complex)


def load_vectors(path, n, key=None):
    import os
    # SHARPy writes aeroelastic_eigenvectors_r.dat and ..._i.dat: pass the _r file
    if "_r." in os.path.basename(path):
        ipath = path.replace("_r.", "_i.")
        if os.path.exists(ipath):
            a = load_array(path, key) + 1j * load_array(ipath, key)
            if a.shape[1] != n:
                raise SystemExit("Eigenvector shape %s but %d eigenvalues; edit load_vectors()" % (a.shape, n))
            return a
    a = load_array(path, key)
    if np.iscomplexobj(a):
        return a
    if a.shape == (n, 2 * n):
        return a[:, :n] + 1j * a[:, n:]
    if a.shape == (2 * n, n):
        return a[:n] + 1j * a[n:]
    if a.shape == (n, n):
        return a.astype(complex)
    raise SystemExit("Unexpected eigenvector shape %s for N=%d; edit load_vectors()" % (a.shape, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eigenvalues")
    ap.add_argument("eigenvectors")
    ap.add_argument("--key", default=None)
    ap.add_argument("--fmax", type=float, default=5.0, help="max damped freq to analyse [Hz]")
    ap.add_argument("--smin", type=float, default=-10.0, help="min real part to analyse [1/s]")
    ap.add_argument("--out", default="modes")
    ap.add_argument("--aero", type=int, default=0,
                    help="number of aerodynamic states to EXCLUDE from participation (e.g. 8090)")
    ap.add_argument("--aero-pos", choices=["first", "last"], default="first",
                    help="whether the aero states are the first or last rows of the state vector")
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="multiply eigenvalues by this if SHARPy gave discrete-time or non-dimensional values")
    args = ap.parse_args()

    lam = load_values(args.eigenvalues, args.key) * args.time_scale
    n = lam.size
    V = load_vectors(args.eigenvectors, n, args.key)
    print("N eigenvalues:", n, " eigenvector matrix:", V.shape)

    sigma, wd = lam.real, lam.imag
    fd = np.abs(wd) / (2 * np.pi)
    # one of each conjugate pair (imag >= 0), inside the window of interest
    sel = np.where((wd >= -1e-9) & (fd <= args.fmax) & (sigma >= args.smin))[0]
    sel = sel[np.argsort(fd[sel])]
    print("Modes analysed:", sel.size)

    nst = V.shape[0]
    print("State vector length:", nst)
    if args.aero:
        lo, hi = (args.aero, nst) if args.aero_pos == "first" else (0, nst - args.aero)
        print("Excluding %d aero states; keeping rows %d..%d (%d states)" % (args.aero, lo, hi - 1, hi - lo))
    else:
        lo, hi = 0, nst
    off = lo   # printed indices are rows of the FULL state vector

    # Right-eigenvector participation (left eigenvectors need the full A matrix, not saved here)
    Vk = V[lo:hi, :]

    names = list(STATE_GROUPS)
    rows = []
    P_all = np.zeros((sel.size, hi - lo))
    for j, i in enumerate(sel):
        p = np.abs(Vk[:, i]) ** 2
        p = p / p.sum()
        P_all[j] = p
        top_idx = np.argsort(p)[::-1][:3]
        if STATE_GROUPS:
            g = {k: p[[ii - off for ii in idx if 0 <= ii - off < p.size]].sum() for k, idx in STATE_GROUPS.items()}
            top = sorted(g.items(), key=lambda kv: -kv[1])[:3]
            top_s = ", ".join("%s %.2f" % kv for kv in top)
            # classical-mode guess: signature with highest summed participation
            scores = {m: sum(g.get(s, 0) for s in sig) for m, sig in MODE_SIGNATURES.items()}
            guess = max(scores, key=scores.get) if g.get("flex", 0) < 0.5 else "flexible"
        else:
            top_s = ", ".join("x[%d] %.2f" % (k + off, p[k]) for k in top_idx)
            guess = ""
        sf = (np.abs(Vk[:, i]) ** 2).sum() / (np.abs(V[:, i]) ** 2).sum()
        rows.append((i, sigma[i], fd[i], sf, top_s, guess))

    with open(args.out + "_table.csv", "w") as f:
        f.write("index,sigma_1ps,fd_Hz,struct_frac,top_participation,label_guess\n")
        for r in rows:
            f.write('%d,%.4f,%.4f,%.3f,"%s",%s\n' % r)
    np.save(args.out + "_participation.npy", P_all)
    print("\n idx   sigma [1/s]   fd [Hz]  struct_frac  top participants                label guess")
    for r in rows:
        print("%5d  %10.4f  %8.4f   %8.3f   %-40s %s" % r)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(sigma[sel], fd[sel], s=18)
        for (i, s, f_, _sf, _, g) in rows:
            if g and g != "flexible":
                ax.annotate(g, (s, f_), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("Real part, sigma [1/s]")
        ax.set_ylabel("Damped frequency [Hz]")
        ax.set_title("Eigenvalue map with mode labels")
        fig.tight_layout()
        fig.savefig(args.out + "_map.png", dpi=200)
        print("\nWrote", args.out + "_table.csv,", args.out + "_participation.npy,", args.out + "_map.png")
    except ImportError:
        print("matplotlib not installed; skipped the map")


if __name__ == "__main__":
    main()