# Trim Run
# Loop through u_inf list to get trim results across velocities for LLTA

import subprocess

# renamed case_name as follows for different cambers:
# camber -> P=4, M=6
# camber2 -> P=5, M=6

velocities = [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
#velocities = [3, 4, 10, 10.5, 15, 20]

python_exe = '/opt/miniconda3/envs/sharpy/bin/python'
script_path = '/Users/icc25/sharpy_cases/llta/generate_llta_trim.py'

CCG_TAG = "wingincidence"

for velocity in velocities:
    suffix = f"_{CCG_TAG}" if CCG_TAG else ""
    case_name = f'llta_trim_u_inf_{velocity}_{suffix}'
    print(f'Running case: {case_name}')

    subprocess.run([python_exe, script_path,'--case_name', case_name,'--velocity', str(velocity)], check=True)
