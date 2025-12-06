#!/bin/bash
#SBATCH --job-name=AGI_STACK
#SBATCH --output=logs/stack_%j.out
#SBATCH --error=logs/stack_%j.err
#SBATCH --partition=standard
#SBATCH --time=01:00:00

echo "Launching AGI-HPC multi-service stack..."

# Launch memory services first
SEM_JOB=$(sbatch --parsable semantic_job.sbatch)
EPI_JOB=$(sbatch --parsable episodic_job.sbatch)
PROC_JOB=$(sbatch --parsable procedural_job.sbatch)

# Safety + Meta depend on memory
SAFETY_JOB=$(sbatch --parsable --dependency=afterok:$SEM_JOB:$EPI_JOB:$PROC_JOB safety_job.sbatch)
META_JOB=$(sbatch --parsable --dependency=afterok:$SEM_JOB meta_job.sbatch)

# RH + LH depend on safety + meta
RH_JOB=$(sbatch --parsable --dependency=afterok:$SAFETY_JOB:$META_JOB rh_job.sbatch)
LH_JOB=$(sbatch --parsable --dependency=afterok:$RH_JOB lh_job.sbatch)

# Environment last
ENV_JOB=$(sbatch --parsable env_job.sbatch)

echo "Stack submitted!"
echo "LH job: $LH_JOB"
echo "RH job: $RH_JOB"
echo "SEM job: $SEM_JOB"
echo "EPI job: $EPI_JOB"
echo "PROC job: $PROC_JOB"
echo "SAFETY job: $SAFETY_JOB"
echo "META job: $META_JOB"
echo "ENV job: $ENV_JOB"
