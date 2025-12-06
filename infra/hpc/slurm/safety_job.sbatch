#!/bin/bash
#SBATCH --job-name=SAFETY
#SBATCH --output=logs/safety_%j.out
#SBATCH --error=logs/safety_%j.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00

module load apptainer

echo "Starting Safety Subsystem..."

apptainer exec \
    --bind /scratch:/scratch \
    ../apptainer/safety.sif \
    python3 -m agi_hpc.services.safety
