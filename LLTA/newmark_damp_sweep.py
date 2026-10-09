import subprocess

# 5e-3 = current baseline value, kept in the sweep as a reference point.
newmark_values = [0.0, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2]

python_exe = '/opt/miniconda3/envs/sharpy/bin/python'
script_path = '/Users/icc25/sharpy_cases/LLTA/generate_llta_asymptotic_stability.py'  # CONFIRM this is the right script

for damp in newmark_values:
    case_name = f'llta_newmarksweep_{damp:.3e}'.replace('+', '')
    print(f'Running case: {case_name}  (newmark_damp = {damp})')

    subprocess.run(
        [python_exe, script_path, '--case_name', case_name, '--newmark_damp', str(damp)],
        check=True
    )