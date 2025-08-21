# scripts/test_relax.py

import os
import shutil
import subprocess
import sys

# --- CONFIGURATION ---
# Please verify that these paths are correct for your system.

# Get the absolute path to the project's root directory
# This assumes this test script is in a subdirectory of the project root (e.g., scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    # 1. The path to the relaxation script we want to test
    "RELAX_SCRIPT_TO_TEST": "utils/relax.py",

    # 2. The source files to use for the test
    "SOURCE_TOP_FILE": "/Users/taskeenjafri/Projects/EvolModel/data/structures/Short/short.top",
    "SOURCE_DAT_FILE": "/Users/taskeenjafri/Projects/EvolModel/data/structures/Short/short.dat",

    # 3. A temporary directory to run the test in
    "TEST_DIRECTORY": "relax_test_run",
}

def main():
    """Prepares a directory and runs the relax.py script in isolation."""
    print("--- Starting Relaxation Script Test ---")

    # --- 1. SETUP: Build absolute paths and create a clean test directory ---
    relax_script_path = os.path.join(PROJECT_ROOT, CONFIG["RELAX_SCRIPT_TO_TEST"])
    source_top = os.path.join(PROJECT_ROOT, CONFIG["SOURCE_TOP_FILE"])
    source_dat = os.path.join(PROJECT_ROOT, CONFIG["SOURCE_DAT_FILE"])
    test_dir = os.path.join(PROJECT_ROOT, CONFIG["TEST_DIRECTORY"])

    print(f"Test directory will be: {test_dir}")
    # Clean up any previous test run and create a fresh directory
    shutil.rmtree(test_dir, ignore_errors=True)
    os.makedirs(test_dir)

    # --- 2. PREPARE INPUTS: Copy and rename files as the GA would ---
    dest_top = os.path.join(test_dir, "input.top")
    dest_dat = os.path.join(test_dir, "last_conf.dat")

    print(f"Copying {source_top} -> {dest_top}")
    shutil.copy(source_top, dest_top)
    print(f"Copying {source_dat} -> {dest_dat}")
    shutil.copy(source_dat, dest_dat)
    
    if not os.path.exists(dest_top) or not os.path.exists(dest_dat):
        print("❌ ERROR: Failed to prepare input files.")
        return

    # --- 3. EXECUTE: Run the relax.py script on the prepared directory ---
    command = [
        "python3",
        relax_script_path,
        test_dir  # Pass the path to the test directory as the argument
    ]

    print(f"\n▶️  Executing command: {' '.join(command)}")
    try:
        # We use subprocess.run to call your script just like the GA does
        subprocess.run(command, check=True)
        print("\n✅ Test script finished successfully!")

    except FileNotFoundError:
        print(f"❌ FATAL ERROR: Could not find the script to test at '{relax_script_path}'")
        print("   Please check the 'RELAX_SCRIPT_TO_TEST' path in the CONFIG.")
        
    except subprocess.CalledProcessError:
        # If the relax.py script itself crashes, this error will be caught.
        # The traceback from relax.py will be printed automatically.
        print("\n❌ ERROR: The `relax.py` script crashed. See the traceback above for details.")

if __name__ == "__main__":
    main()