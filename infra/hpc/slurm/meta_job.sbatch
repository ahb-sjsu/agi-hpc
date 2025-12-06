#!/bin/bash
#SBATCH --job-name=META
#SBATCH --output=logs/meta_%j.out
#SBATCH --error=logs/meta_%j.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00

module load apptainer

echo "Starting Metacognition Engine..."

apptainer exec \
    --bind /scratch:/scratch \
    ../apptainer/meta.sif \
    python3 -m agi_hpc.services.metacognition
