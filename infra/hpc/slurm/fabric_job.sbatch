#!/bin/bash
#SBATCH --job-name=FABRIC
#SBATCH --output=logs/fabric_%j.out
#SBATCH --error=logs/fabric_%j.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00

module load apptainer

echo "Starting Event Fabric..."

apptainer exec \
    --bind /scratch:/scratch \
    ../apptainer/fabric.sif \
    python3 -m agi_hpc.services.event_fabric
