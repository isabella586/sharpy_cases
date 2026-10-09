import subprocess
import numpy as np

# --- j_bar_main sweep values ---------------------------------------------
# Bounds derived from the mu/j hand-calc check on the main wing:
#   J_CURRENT = spar-only value currently in generate_llta_trim.py
#   j_low     = extra (skin/rib) mass assumed close to the spar radius
#   j_high    = extra mass assumed out at the local half-chord (deliberately
#               generous upper limit - see chat working for the derivation)
J_CURRENT = 3.27505026e-6
J_LOW = 1.38236e-5
J_HIGH = 2.67434e-3

N_POINTS = 8
j_values = np.logspace(np.log10(J_CURRENT), np.log10(J_HIGH), N_POINTS)
j_values = sorted(set(j_values) | {J_CURRENT, J_LOW})  # keep both reference points explicit

python_exe = '/opt/miniconda3/envs/sharpy/bin/python'
script_path = '/Users/icc25/sharpy_cases/llta/generate_llta_asymptotic_stability.py'  # CONFIRM: same script as the velocity sweep?

for j_val in j_values:
    case_name = f'llta_jsweep_{j_val:.3e}'.replace('+', '')
    print(f'Running case: {case_name}  (j_bar_main = {j_val:.6e})')

    subprocess.run(
        [python_exe, script_path, '--case_name', case_name, '--j_bar_main', str(j_val)],
        check=True
    )