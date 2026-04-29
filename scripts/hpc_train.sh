#!/bin/bash
#SBATCH --job-name=rl-ad-bidding
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zh2312@nyu.edu

# ---------------------------------------------------------------
# NYU Torch HPC — Training job for rl-ad-bidding
#
# One-time setup (run once on login node, no Singularity needed):
#   cd /scratch/zh2312
#   wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
#   sh Miniforge3-Linux-x86_64.sh -b -p /scratch/zh2312/miniforge3
#   source /scratch/zh2312/miniforge3/etc/profile.d/conda.sh
#   conda create -n rl-ad-bidding python=3.9.12 -y
#   conda activate rl-ad-bidding
#   git clone https://github.com/alimama-tech/AuctionNet.git /scratch/zh2312/AuctionNet
#   cd /scratch/zh2312/rl-ad-bidding && git pull && pip install -r requirements.txt
#
# Usage:
#   sbatch scripts/hpc_train.sh
#   sbatch scripts/hpc_train.sh configs/my_experiment.yaml
# ---------------------------------------------------------------

set -euo pipefail

REPO=/scratch/zh2312/rl-ad-bidding

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Started at: $(date)"

CONFIG=${1:-configs/default.yaml}
echo "Config:     $CONFIG"

mkdir -p $REPO/logs $REPO/saved_models

source /scratch/zh2312/miniforge3/etc/profile.d/conda.sh
conda activate rl-ad-bidding

cd $REPO
python scripts/train.py --config ${CONFIG} --run-name slurm_${SLURM_JOB_ID}

echo "Finished at: $(date)"
