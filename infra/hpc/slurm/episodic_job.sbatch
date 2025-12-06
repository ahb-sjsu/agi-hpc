#!/bin/bash
#SBATCH --job-name=EPI
#SBATCH --output=logs/epi_%j.out
#SBATCH --error=logs/epi_%j.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00

module load apptainer

echo "Starting Episodic Memory..."

apptainer exec \
    --bind /scratch:/scratch \
    ../apptainer/episodic.sif \
    python3 -m agi_hpc.services.episodic_memory
