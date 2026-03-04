#!/bin/bash
#SBATCH --job-name=rl-ad-bidding
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_NETID@nyu.edu

# ---------------------------------------------------------------
# NYU Greene HPC — Training job for rl-ad-bidding
# Usage: sbatch scripts/hpc_train.sh [--config configs/default.yaml]
# ---------------------------------------------------------------

set -euo pipefail

echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Started at:   $(date)"

# --- Environment setup ---
module purge
module load anaconda3/2023.09

conda activate rl-ad-bidding  # replace with your actual env name

# Verify GPU is visible
nvidia-smi

# --- Working directory ---
cd $SLURM_SUBMIT_DIR

# --- Config (override via sbatch --export or positional arg) ---
CONFIG=${1:-configs/default.yaml}
echo "Using config: $CONFIG"

# --- Create log directory if it doesn't exist ---
mkdir -p logs saved_models

# --- Launch training ---
python scripts/train.py \
    --config "$CONFIG" \
    --run-name "slurm_${SLURM_JOB_ID}"

echo "Finished at: $(date)"
