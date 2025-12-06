#!/bin/bash
#SBATCH --job-name=ENV
#SBATCH --output=logs/env_%j.out
#SBATCH --error=logs/env_%j.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

module load apptainer

echo "Starting Environment Server..."

apptainer exec \
    --bind /scratch:/scratch \
    ../apptainer/env.sif \
    python3 -m agi_hpc.environment.env_server
