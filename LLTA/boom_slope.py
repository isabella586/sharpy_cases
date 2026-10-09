import glob, h5py, numpy as np

f = h5py.File(glob.glob('/Users/icc25/sharpy_cases/LLTA/output/llta_trim_u_inf_9_camber_fuselagestiffnessupdated/savedata/llta_trim_u_inf_9_camber_fuselagestiffnessupdated.data.h5')[0], 'r')
s = f['data']['structure']
pos0 = s['ini_info']['pos'][()]                          # jig shape, A frame
ts_keys = sorted(k for k in s['timestep_info'].keys() if k.isdigit())
print('timestep keys:', ts_keys)            # expect ['00000'] for a static trim
last = ts_keys[-1]
pos1 = s['timestep_info'][last]['pos'][()]

# boom nodes: on the centreline (y = z = 0), aft of the wing root
tol = 1e-6
boom = np.where((np.abs(pos0[:, 1]) < tol) & (np.abs(pos0[:, 2]) < tol) & (pos0[:, 0] > tol))[0]
boom = boom[np.argsort(pos0[boom, 0])]
end, prev = boom[-1], boom[-2]

slope = lambda p: np.arctan2(p[end, 2] - p[prev, 2], p[end, 0] - p[prev, 0])
pitch = -np.degrees(slope(pos1) - slope(pos0))           # + = tail nose-up
drop = pos1[end, 2] - pos0[end, 2]

print(f"boom end x = {pos0[end, 0]:.3f} m, drop = {drop * 1000:.1f} mm, tail pitch = {pitch:+.2f} deg")

d = pos1 - pos0
tip = np.argmax(pos0[:, 1])
print(f"wing tip y = {pos0[tip, 1]:.3f} m, dz = {d[tip, 2] * 1000:.1f} mm")   # should be tens of mm
print("boom nodes x, dz [mm]:")
for n in boom:
    print(f"  {pos0[n, 0]:.3f}  {d[n, 2] * 1000:+.3f}")