#!/bin/bash
#SBATCH --job-name=ga_eval_worker  # Job name
#SBATCH --output=ga_run/slurm_logs/worker_%A_%a.out  # Standard output and error log (%A = job ID, %a = array task ID)
#SBATCH --ntasks=1                 # Run a single task
#SBATCH --cpus-per-task=1          # Number of CPU cores per task
#SBATCH --mem=2gb                  # Job memory request
#SBATCH --time=00:30:00            # Time limit hrs:min:sec (set this to be slightly longer than your relaxation timeout)

# --- Setup ---
# Create log directory if it doesn't exist
mkdir -p ga_run/slurm_logs

# Activate your python environment (IMPORTANT: MODIFY THIS FOR YOUR SYSTEM)
# For Conda:
# source /path/to/your/conda/etc/profile.d/conda.sh
# conda activate your_env_name
# For virtualenv:
# source /path/to/your/venv/bin/activate

echo "Starting SLURM task ${SLURM_ARRAY_TASK_ID} on $(hostname)"

# --- Arguments from the manager script ---
JOB_LIST_FILE=$1
TARGET_XYZ=$2
RELAX_SCRIPT=$3
WEIGHT_SHAPE=$4
WEIGHT_INTEGRITY=$5

# --- Get the specific directory for this task ---
# SLURM_ARRAY_TASK_ID will be a number from 1 to N.
# We use it to pick the corresponding line from our job list file.
INDIVIDUAL_DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${JOB_LIST_FILE}")

if [ -z "$INDIVIDUAL_DIR" ]; then
    echo "Error: Could not find a directory for task ID ${SLURM_ARRAY_TASK_ID} in ${JOB_LIST_FILE}"
    exit 1
fi

# --- Run the Python Worker Script ---
python src/evaluate_worker.py \
    "$INDIVIDUAL_DIR" \
    --target_xyz "$TARGET_XYZ" \
    --relax_script "$RELAX_SCRIPT" \
    --weight_shape "$WEIGHT_SHAPE" \
    --weight_integrity "$WEIGHT_INTEGRITY"

echo "Finished SLURM task ${SLURM_ARRAY_TASK_ID}"