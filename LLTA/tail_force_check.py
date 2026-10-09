import h5py, numpy as np

RUNS = {
    "M0": "/Users/icc25/Documents/05_Data/01_Sharpy/19_llta_trim_polarcorrection_investigation/llta_trim_u_inf_9_camber_ccgfixed/savedata/llta_trim_u_inf_9_camber_ccgfixed.data.h5",
    "M1": "/Users/icc25/sharpy_cases/LLTA/output/llta_trim_u_inf_9_camber_modelaudit/savedata/llta_trim_u_inf_9_camber_modelaudit.data.h5",
}

for tag, path in RUNS.items():
    with h5py.File(path, "r") as f:
        ti = f["data/aero/timestep_info"]
        ts = sorted(k for k in ti.keys() if k.isdigit())[-1]   # skip '_read_as'
        F = ti[ts]["inertial_steady_forces"][()]               # [n_surf x 6]
    wing, fin, tail = F[0, 2] + F[1, 2], F[2, 2], F[3, 2] + F[4, 2]
    print(f"{tag}: wing Fz {wing:.2f} N | tail Fz {tail:.2f} N | fin Fz {fin:.2f} N | sum {wing + tail + fin:.2f} N")

def as_list(g):
    if isinstance(g, h5py.Dataset):
        return list(g[()])
    keys = sorted((k for k in g.keys() if k.isdigit()), key=int)
    return [g[k][()] for k in keys]

for tag, path in RUNS.items():
    with h5py.File(path, "r") as f:
        ti = f["data/aero/timestep_info"]
        ts = sorted(k for k in ti.keys() if k.isdigit())[-1]
        zeta, forces = as_list(ti[ts]["zeta"]), as_list(ti[ts]["forces"])
    print(f"--- {tag}")
    for name, surfs in {"wing": [0, 1], "tail": [3, 4]}.items():
        Fz = sum(forces[s][2].sum() for s in surfs)
        x_cp = sum((zeta[s][0] * forces[s][2]).sum() for s in surfs) / Fz
        Z = zeta[surfs[0]]
        j = Z.shape[2] // 2                                       # mid-span strip
        le, te = Z[:, 0, j], Z[:, -1, j]
        incid = np.degrees(np.arctan2(-(te[2] - le[2]), te[0] - le[0]))  # +ve = nose-up
        print(f"{name}: Fz {Fz:.2f} N | x_cp {x_cp:+.3f} m | LE x {le[0]:+.3f} TE x {te[0]:+.3f} m | chord angle {incid:+.2f} deg")