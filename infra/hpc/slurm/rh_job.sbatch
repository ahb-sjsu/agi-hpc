#!/bin/bash
#SBATCH --job-name=RH
#SBATCH --output=logs/rh_%j.out
#SBATCH --error=logs/rh_%j.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

module load apptainer

echo "Starting Right Hemisphere..."

apptainer exec \
    --bind /scratch:/scratch \
    ../apptainer/RH.sif \
    python3 -m agi_hpc.services.right_hemisphere
