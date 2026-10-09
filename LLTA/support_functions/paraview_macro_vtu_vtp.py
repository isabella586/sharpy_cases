import glob, os
from paraview.simple import *

#Define the directory with the saved output files
directory = os.path.dirname(os.path.realpath(__file__)) + '/output'
print("Paraview is looking for .vtu and .vtp files from this directory:", directory)

if directory:
    folder_name = os.path.basename(directory)
    extensions = ["*.vtu", "*.vtp"]
    # Find and sort all .vtu files in the chosen folder
    #all_files = sorted(glob.glob(os.path.join(directory, "*.vtu"), recursive=True))
    
    all_files = sorted(
        f for ext in extensions
        for f in glob.glob(os.path.join(directory, "**/" + ext), recursive=True))
    
    wake_files = [f for f in all_files if "wake" in os.path.basename(f).lower()]
    body_files = [f for f in all_files if "body" in os.path.basename(f).lower()]
    beam_files = [f for f in all_files if "beam" in os.path.basename(f).lower()]

    if wake_files:
        for each_wake_file in wake_files:
            # Load them with the correct reader — no dialog needed
            name = os.path.splitext(os.path.basename(each_wake_file))[0]
            XMLStructuredGridReader(registrationName=f"{folder_name}/wake/{name}", FileName=[each_wake_file])
        print(f"Loaded {len(wake_files)} WAKE file(s) from: {directory}")
    
    if body_files:
        for each_body_file in body_files:
            # Load them with the correct reader — no dialog needed
            name = os.path.splitext(os.path.basename(each_body_file))[0]
            XMLStructuredGridReader(registrationName=f"{folder_name}/body/{name}", FileName=[each_body_file])
        print(f"Loaded {len(body_files)} BODY file(s) from: {directory}")

    if beam_files:
        for each_beam_file in beam_files:
            # Load them with the correct reader — no dialog needed
            name = os.path.splitext(os.path.basename(each_beam_file))[0]
            XMLPolyDataReader(registrationName=f"{folder_name}/beam/{name}", FileName=[each_beam_file])
        print(f"Loaded {len(beam_files)} BEAM file(s) from: {directory}")
    
    if not wake_files and not body_files: 
        print("No .vtu files found in that folder.")

    if not beam_files:
        print("No .vtp files found in that folder.")