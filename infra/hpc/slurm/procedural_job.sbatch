#!/bin/bash
#SBATCH --job-name=PROC
#SBATCH --output=logs/proc_%j.out
#SBATCH --error=logs/proc_%j.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00

module load apptainer

echo "Starting Procedural Memory..."

apptainer exec \
    --bind /scratch:/scratch \
    ../apptainer/procedural.sif \
    python3 -m agi_hpc.services.procedural_memory
