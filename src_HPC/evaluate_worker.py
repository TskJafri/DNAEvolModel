# src/evaluate_worker.py

import os
import sys
import shutil
import argparse
import subprocess

# --- Project Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.dna_model import DNAStructure
from src.fitness import FitnessCalculator
from src.target_scaffold import TargetScaffold

def run_relaxation(individual_dir: str, relax_script_path: str) -> bool:
    """Executes the external relaxation script for a single individual."""
    print(f"    -> Running relaxation for: {individual_dir}")
    python_executable = sys.executable
    command = [python_executable, relax_script_path, individual_dir]
    
    # Prepare files for the relaxation script
    top_path = os.path.join(individual_dir, "structure.top")
    dat_path = os.path.join(individual_dir, "structure.dat")
    input_top = os.path.join(individual_dir, "input.top")
    input_dat = os.path.join(individual_dir, "last_conf.dat")

    try:
        shutil.move(top_path, input_top)
        shutil.move(dat_path, input_dat)

        # Run the subprocess
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800) # 30 min timeout

        # Find and restore the relaxed files
        relaxed_dat = os.path.join(individual_dir, "md_relax", "last_conf.dat")
        if not os.path.exists(relaxed_dat):
            raise FileNotFoundError("Relaxation finished, but output .dat file not found.")

        shutil.copy(os.path.join(individual_dir, "md_relax", "input.top"), top_path)
        shutil.copy(relaxed_dat, dat_path)
        return True

    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"    -> ❌ ERROR: Relaxation failed for {individual_dir}.")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"    -> STDOUT: {e.stdout}")
            print(f"    -> STDERR: {e.stderr}")
        else:
            print(f"    -> Reason: {e}")
        return False
    finally:
        # Clean up input files
        if os.path.exists(input_top):
             os.remove(input_top)
        if os.path.exists(input_dat):
             os.remove(input_dat)


def main():
    parser = argparse.ArgumentParser(description="Worker to evaluate a single GA individual.")
    parser.add_argument("individual_dir", type=str, help="Path to the individual's directory.")
    parser.add_argument("--target_xyz", type=str, required=True, help="Path to the target shape (.xyz) file.")
    parser.add_argument("--relax_script", type=str, required=True, help="Path to the relaxation script.")
    parser.add_argument("--weight_shape", type=float, default=1.0)
    parser.add_argument("--weight_integrity", type=float, default=0.1)
    args = parser.parse_args()

    # Default fitness is infinity (worst possible)
    fitness = float('inf')

    # 1. Run Relaxation
    relaxation_succeeded = run_relaxation(args.individual_dir, args.relax_script)

    # 2. Calculate Fitness (only if relaxation worked)
    if relaxation_succeeded:
        print(f"    -> Evaluating fitness for: {args.individual_dir}")
        try:
            target_scaffold = TargetScaffold(args.target_xyz)
            fitness_calculator = FitnessCalculator(
                target_scaffold=target_scaffold,
                weight_shape=args.weight_shape,
                weight_integrity=args.weight_integrity
            )
            dna = DNAStructure(
                os.path.join(args.individual_dir, "structure.top"),
                os.path.join(args.individual_dir, "structure.dat")
            )
            fitness = fitness_calculator.calculate_fitness(dna)
            print(f"    -> Fitness = {fitness:.4f}")
        except Exception as e:
            print(f"    -> ⚠️ WARNING: Could not evaluate. Reason: {e}")
            fitness = float('inf')

    # 3. Save the result
    fitness_file = os.path.join(args.individual_dir, "fitness.txt")
    with open(fitness_file, 'w') as f:
        f.write(str(fitness))
    
    print(f"--- Worker finished for {args.individual_dir} ---")

if __name__ == "__main__":
    main()