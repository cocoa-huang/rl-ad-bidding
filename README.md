# rl-ad-bidding

> Reinforcement learning agents for optimal bidding in real-time ad auctions — NYU grad RL course project.

![Python 3.9](https://img.shields.io/badge/python-3.9.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-EE4C2C.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Overview

This project trains reinforcement learning agents to learn optimal bidding
strategies in real-time ad auctions. We use
[AuctionNet (NeurIPS 2024, Alibaba)](https://github.com/alimama-tech/AuctionNet)
as our primary simulation environment and dataset.

The final system trains a budget-aware PPO policy in a Gymnasium wrapper around
AuctionNet, then evaluates it against AuctionNet pretrained offline-RL policies
and fixed-alpha baselines in a shared simulation loop.

| Metric | Description |
|--------|-------------|
| **ROI** | Total conversion value won / total budget spent |
| **Budget Utilization** | Amount spent / total budget available |
| **Win Rate** | Auctions won / auctions entered |

---

## Final Results

Final results use `configs/gcp-run-9-selective.yaml`, where the PPO policy
chooses both bid intensity and selectivity:

- `action[0] -> alpha in [0, 150]`
- `action[1] -> keep_fraction in [0.05, 1.0]`
- bids are `alpha * pValue` only for the top `keep_fraction` PVs by pValue

The table below comes from `scripts/common_policy_eval.py` over 500 episodes.
All policies run through the same AuctionNet simulation loop and are scored with
the same metrics.

| Policy | Conversions | Cost | Utilization | CPA ↓ | Conv. ROI ↑ | Slot Win | Exposure | Shaped Reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO gcp-run-9-selective | 1349 | 49,775.26 | 66.37% | 36.90 | 0.02710 | 5.31% | 5.05% | 297.78 ± 213.57 |
| AuctionNet pretrained IQL | 1140 | 74,553.77 | 99.41% | 65.40 | 0.01529 | 12.44% | 12.14% | 162.06 ± 184.52 |
| fixed alpha 100 | 1120 | 40,196.19 | 53.59% | 35.89 | 0.02786 | 4.86% | 4.54% | 260.68 ± 190.32 |
| fixed alpha 130 | 1203 | 52,986.68 | 70.65% | 44.05 | 0.02270 | 5.88% | 5.56% | 259.57 ± 197.57 |

Selective PPO outperforms the AuctionNet pretrained IQL baseline on
conversions, CPA, conversion ROI, and shaped reward under the shared evaluator.
Against fixed alpha, PPO has the best shaped reward and a better
reward/utilization tradeoff, but fixed alpha 100 remains slightly better on pure
ROI and CPA by spending less.

---

## Setup

### Prerequisites

- **Python 3.9.12 exactly** (AuctionNet breaks on 3.10+)
- Conda (Miniconda or Anaconda)
- Access to NYU Greene HPC for GPU training (optional for local dev)

### Installation

```bash
# 1. Clone this repo
git clone https://github.com/cocoa-huang/rl-ad-bidding.git
cd rl-ad-bidding

# 2. Clone AuctionNet as a sibling directory (expected at ../AuctionNet)
git clone https://github.com/alimama-tech/AuctionNet.git ../AuctionNet

# 3. Create the conda environment with the exact Python version
conda create -n rl-ad-bidding python=3.9.12
conda activate rl-ad-bidding

# 4. Install Python dependencies
pip install -r requirements.txt
```

> **Note:** The `data/` directory is gitignored. You must generate or download
> auction log data separately. See the Datasets section below.

---

## Project Structure

```
rl-ad-bidding/
├── environment/     # Gymnasium wrapper around AuctionNet's simulation
├── agents/          # RL algorithm implementations (PPO, IQL, BCQ, baselines)
├── evaluation/      # Metrics computation, ablation runner, plot generation
├── scripts/         # SLURM job scripts and data generation helpers
├── notebooks/       # EDA and experiment analysis notebooks
├── configs/         # YAML hyperparameter configs, one file per experiment
├── data/            # Local data files (gitignored — not committed)
├── saved_models/    # Trained model checkpoints (gitignored — not committed)
├── requirements.txt
└── README.md
```

---

## Usage

### Run training locally

```bash
python scripts/train.py --config configs/default.yaml
```

### Evaluate AuctionNet pretrained IQL baseline

This evaluates AuctionNet's official pretrained IQL policy under the key knobs
from a chosen config (budget, ticks, pv_num, player_index).

```bash
python scripts/evaluate_auctionnet_iql.py --config configs/gcp-run-9-selective.yaml
```

### Compare PPO vs IQL vs fixed-alpha in one shared simulator loop

This runs all selected policies inside the same AuctionNet simulation loop and
prints comparable metrics.

```bash
python scripts/common_policy_eval.py \
  --config configs/gcp-run-9-selective.yaml \
  --run-name gcp-run-9-selective \
  --episodes 500 \
  --fixed-alphas 100 130 \
  --output-json results/common_eval_gcp_run_9_selective_500.json
```

### Submit a training job on NYU Greene HPC

```bash
sbatch scripts/hpc_train.sh configs/default.yaml
```

### Run evaluation on a saved checkpoint

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint saved_models/ppo_best.pt
```

All runs are tracked in [Weights & Biases](https://wandb.ai). Set your W&B API
key before training:

```bash
export WANDB_API_KEY=your_key_here
```

---

## Datasets

This project uses auction log data generated by the AuctionNet simulator.
The `data/` directory is gitignored — raw data is never committed to this repo.

To generate data:

```bash
# See AuctionNet repo for data generation scripts and episode assets.
```

To download pre-generated data from shared storage:

```bash
# Not required for the core AuctionNet NeurIPS PV generator path used in this repo.
```

---

## License

MIT
