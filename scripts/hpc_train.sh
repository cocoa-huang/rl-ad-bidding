#!/bin/bash
#SBATCH --job-name=rl-ad-bidding
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zh2312@nyu.edu

# ---------------------------------------------------------------
# NYU Torch HPC — Training job for rl-ad-bidding
#
# One-time setup required before first run — see README or ask Eric:
#   /scratch/zh2312/rl-ad-bidding/overlay-15GB-500K.ext3  (overlay with conda env)
#   /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif  (Singularity image)
#
# Usage:
#   sbatch scripts/hpc_train.sh
#   sbatch scripts/hpc_train.sh configs/my_experiment.yaml
# ---------------------------------------------------------------

set -euo pipefail

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Started at: $(date)"

CONFIG=${1:-configs/default.yaml}
echo "Config:     $CONFIG"

mkdir -p logs saved_models

OVERLAY=/scratch/zh2312/rl-ad-bidding/overlay-15GB-500K.ext3
SIF=/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif
WORKDIR=/scratch/zh2312/rl-ad-bidding

singularity exec --bind /scratch \
  --overlay ${OVERLAY}:ro \
  ${SIF} \
  /bin/bash -c "
    source /ext3/env.sh
    conda activate rl-ad-bidding
    cd ${WORKDIR}
    python scripts/train.py --config ${CONFIG} --run-name slurm_\${SLURM_JOB_ID}
  "

echo "Finished at: $(date)"
