

import subprocess
import os

def run_script(script_name):
    print(f"Running {script_name}...")
    result = subprocess.run(['python', script_name], check=True)
    if result.returncode == 0:
        print(f"{script_name} completed successfully.")
    else:
        print(f"Error running {script_name}")

if __name__ == "__main__":
    # Run processing.py
    run_script('processing.py')

    # Check if the output file from processing.py exists
    if os.path.exists('spg_average_features.csv'):
        # Run regression.py
        run_script('regression.py')
    else:
        print("Error: spg_average_features.csv not found. Make sure processing.py ran correctly.")