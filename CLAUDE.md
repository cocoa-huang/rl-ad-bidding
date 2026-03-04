# CLAUDE.md — rl-ad-bidding

## Project

**rl-ad-bidding** — A graduate RL research project (NYU, 6-week) that trains a
reinforcement learning agent to learn optimal bidding strategies in real-time ad
auctions. The agent is evaluated on ROI, budget utilization, and win rate against
multiple baselines.

---

## Tech Stack

- **Python 3.9.12** (strict version requirement — AuctionNet breaks on 3.10+)
- **PyTorch >= 2.0** — neural network training
- **Gymnasium >= 0.29** — environment API standard
- **Weights & Biases** — experiment tracking and dashboards
- **NYU Greene HPC (SLURM)** — GPU training via `scripts/hpc_train.sh`

---

## Folder Structure

```
environment/     — Gymnasium-compliant wrapper around AuctionNet's simulation
agents/          — RL algorithm implementations (PPO, IQL, BCQ, baselines)
evaluation/      — Metrics computation, ablation runner, plot generation
scripts/         — SLURM job scripts and data generation helpers
notebooks/       — EDA and experiment analysis notebooks
configs/         — YAML hyperparameter configs, one file per experiment
data/            — Local data files (gitignored, not committed)
saved_models/    — Trained model checkpoints (gitignored, not committed)
```

---

## Team Roles

| Person | Area | Responsibility |
|--------|------|----------------|
| Person A | `environment/` | Gymnasium wrapper, data pipeline, HPC setup |
| Person B | `agents/` | PPO implementation, baseline agents |
| Person C | `evaluation/` | Metrics, ablations, W&B dashboards |
| Person D | report/ | Literature review, presentation, floats between teams |

---

## Key Design Decisions

- **Offline → online RL training**: train on pre-generated auction logs first,
  then fine-tune in live AuctionNet simulation.
- **All environments must conform to the Gymnasium API** (`reset`/`step`/`render`/`close`).
- **All experiments tracked in W&B** — no orphaned runs.
- **Hyperparameters live in `configs/` YAML files** — never hardcoded in Python files.
- **SLURM scripts in `scripts/`** are the canonical way to launch HPC training jobs.

---

## Core Evaluation Metrics

| Metric | Formula | Goal |
|--------|---------|------|
| **ROI** | total conversion value won / total budget spent | Higher is better |
| **Budget Utilization** | amount spent / total budget available | Target ~100% |
| **Win Rate** | auctions won / auctions entered | Diagnostic metric |

---

## Instructions for Claude Working in This Repo

- Always maintain **Gymnasium API compatibility** in `environment/`.
- **Never hardcode file paths** — use `configs/` and relative paths.
- Every new experiment must have a corresponding **YAML in `configs/`**.
- All training runs must **log to W&B** with a meaningful `run_name`.
- Keep `agents/` **stateless where possible** — state lives in the environment.
- When adding new dependencies, **update `requirements.txt` immediately**.
- **Do not modify anything** in the AuctionNet sibling directory from here.

---

## Known Constraints

- Python must be **exactly 3.9.12** — AuctionNet breaks on 3.10+.
- AuctionNet lives as a **sibling repo**, not inside this directory.
- The `data/` folder is **gitignored** — never commit raw data files.
- NYU Greene HPC uses **SLURM** — all GPU jobs go through `scripts/hpc_train.sh`.
