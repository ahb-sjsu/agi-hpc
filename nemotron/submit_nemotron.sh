#!/bin/bash
#SBATCH --job-name=nemotron
#SBATCH --output=nemotron-%j.log
#SBATCH --error=nemotron-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

module load cuda/12.2
export PATH=~/nemotron/env/bin:$PATH
export WORK_DIR=~/nemotron
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Nemotron LoRA Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date
nvidia-smi -L

cd ~/nemotron
python train_hpc.py 2>&1 | tee nemotron_output.log

echo "=== Done at $(date) ==="
