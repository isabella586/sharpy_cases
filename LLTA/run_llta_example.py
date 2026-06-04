from llta import LLTA   # Import the LLTA model (aircraft class) from the repo
from get_settings import get_settings

# -----------------------------
# 1. Define case parameters
# -----------------------------
case_name = 'llta_test_case'
case_route = 'llta/'  # folder where SHARPy input/output files will go
output = 'output/'           # whether to print verbose output

# -----------------------------
# 2. Create FLEXOP model instance
# -----------------------------
llta_model = LLTA(case_name, case_route, output)

# -----------------------------
# 3. Initialize aeroelastic model
# This sets up discretization, structural/aero parameters, etc.
# -----------------------------
llta_model.init_aeroelastic() # this generates just wing and statictrim is only solver run

# -----------------------------
# 4. Generate SHARPy input files
# This writes the .sharpy input file and geometry for SHARPy to run
# -----------------------------
llta_model.generate()

# -----------------------------
# 5. (Optional) Customize solver settings
# You can modify settings here, otherwise defaults are used
# -----------------------------
settings = get_settings(llta_model, flow = ['BeamLoader', 'AerogridLoader', 'StaticCoupled'], dt = 0.1)
llta_model.create_settings(settings) # Expects a dictionary of solver settings

'''# -----------------------------
# 6. Run the simulation
# This calls SHARPy using the generated input file
# -----------------------------
llta_model.run()'''

print("LLTA simulation finished. Check output folder for results!")