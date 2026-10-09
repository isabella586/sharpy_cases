"""
tail_inflow_check.py -- flow angle arriving at the horizontal tail, rebuilt from a SHARPy .data.h5

Rebuilds the wing's vortex rings (bound + wake) from the saved static UVLM solution and
evaluates the velocity they induce at the tail collocation points (Biot-Savart).

Self-check printed first: the rebuilt solution must satisfy SHARPy's no-flow-through
condition on the wing and tail (residual ~1e-3 m/s). If it doesn't, don't trust the rest.

Requires SaveData with save_aero and save_wake on (both default in the LLTA case).
Surfaces: 0/1 wing, 2 fin, 3/4 tail. Usage:
    python tail_inflow_check.py M0.data.h5 M1.data.h5
"""
import sys
import h5py
import numpy as np

U_INF = np.array([9.0, 0.0, 0.0])   # set to the run's u_inf


def as_list(g):
    if isinstance(g, h5py.Dataset):
        return list(g[()])
    return [g[k][()] for k in sorted((k for k in g.keys() if k.isdigit()), key=int)]


def segment_velocity(P, A, B, G, eps=1e-4):
    r1, r2 = P - A, P - B
    c = np.cross(r1, r2)
    c2 = (c * c).sum(-1)
    r0 = B - A
    n1, n2 = np.linalg.norm(r1, axis=-1), np.linalg.norm(r2, axis=-1)
    k = G / (4 * np.pi * np.maximum(c2, eps ** 2)) * (
        (r0 * r1).sum(-1) / np.maximum(n1, eps) - (r0 * r2).sum(-1) / np.maximum(n2, eps))
    k = np.where(c2 < eps ** 2, 0.0, k)
    return c * k[..., None]


def ring_segments(Z, gam):
    """Z: (3, M+1, N+1) ring corners; gam: (M, N) ring circulation."""
    A, B, C, D = Z[:, :-1, :-1], Z[:, :-1, 1:], Z[:, 1:, 1:], Z[:, 1:, :-1]
    return [(p.reshape(3, -1).T, q.reshape(3, -1).T, gam.reshape(-1))
            for p, q in [(A, B), (B, C), (C, D), (D, A)]]


def velocity(P, segs):
    v = np.zeros_like(P)
    for A, B, G in segs:
        v += segment_velocity(P[:, None, :], A[None], B[None], G[None]).sum(1)
    return v


def colloc_and_normals(Z):
    col = 0.25 * (Z[:, :-1, :-1] + Z[:, :-1, 1:] + Z[:, 1:, 1:] + Z[:, 1:, :-1])
    d1 = (Z[:, 1:, 1:] - Z[:, :-1, :-1]).reshape(3, -1).T
    d2 = (Z[:, 1:, :-1] - Z[:, :-1, 1:]).reshape(3, -1).T
    n = np.cross(d1, d2)
    return col.reshape(3, -1).T, n / np.linalg.norm(n, axis=1)[:, None]


def deg(w):
    return np.degrees(np.arctan(w / np.linalg.norm(U_INF)))


for path in sys.argv[1:]:
    with h5py.File(path, "r") as f:
        at = f["data/aero/timestep_info/00000"]
        zeta, zstar = as_list(at["zeta"]), as_list(at["zeta_star"])
        # sign flip matches SHARPy's ring orientation to the one used here (verified by the BC check)
        gam = [-x for x in as_list(at["gamma"])]
        gstar = [-x for x in as_list(at["gamma_star"])]

    all_segs = sum([ring_segments(zeta[s], gam[s]) + ring_segments(zstar[s], gstar[s])
                    for s in range(len(zeta))], [])
    print(f"===== {path}")
    for s, name in [(0, "wing"), (3, "tail")]:
        P, n = colloc_and_normals(zeta[s])
        res = np.abs(((velocity(P, all_segs) + U_INF) * n).sum(1)).mean()
        print(f"  BC check {name}: mean |(U+v).n| = {res:.4f} m/s  (should be ~0)")

    wing_bound = ring_segments(zeta[0], gam[0]) + ring_segments(zeta[1], gam[1])
    wing_wake = ring_segments(zstar[0], gstar[0]) + ring_segments(zstar[1], gstar[1])
    wake_end = [(zstar[s][:, -1, 1:].reshape(3, -1).T, zstar[s][:, -1, :-1].reshape(3, -1).T, gstar[s][-1, :])
                for s in (0, 1)]

    P = np.vstack([colloc_and_normals(zeta[3])[0], colloc_and_normals(zeta[4])[0]])
    wake_start, wake_stop = zstar[0][0, 0, :].mean(), zstar[0][0, -1, :].mean()
    print(f"  tail collocation x {P[:, 0].min():.3f}-{P[:, 0].max():.3f} m | "
          f"wing wake {wake_start:.3f}-{wake_stop:.3f} m (length {wake_stop - wake_start:.3f} m)")

    w_bound = velocity(P, wing_bound)[:, 2].mean()
    w_wake = velocity(P, wing_wake)[:, 2].mean()
    w_end = velocity(P, wake_end)[:, 2].mean()
    for label, w in [("wing bound vortices", w_bound), ("wing wake, all", w_wake),
                     ("  of which wake end edge", w_end), ("  rest of wake", w_wake - w_end),
                     ("TOTAL from wing system", w_bound + w_wake)]:
        print(f"  {label:<26s} {deg(w):+.2f} deg  ({'upwash' if w > 0 else 'downwash'})")